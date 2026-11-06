"""Cross-attention decoder — tests (#23 MODEL-03).

Acceptance criteria of issue #23 (clean-room V2 implementation):

* Decoder forward pass executes with shapes ``[B, 1, 256]`` query and
  ``[B, N, 256]`` memory, outputting ``[B, 256]``.
* Attention weights correctly mask out padded atoms via
  ``key_padding_mask``.

Beyond the pinned criteria, this suite enforces: the issue-pinned layer
structure (MHA 256/8/dropout 0.1, FFN 256->512->SiLU->Dropout->512->256,
pre-LayerNorm residuals on both sub-layers, 2 stacked layers), the pre-LN
residual identity under zeroed sub-layers (which also distinguishes pre-LN
from post-LN), order-preserving per-graph padding, exact masking semantics
(padded columns exactly zero in every layer/head/graph; outputs bit-identical
under arbitrary padded-slot junk), SE(3) invariance of the decoder context
end-to-end through the real backbone + handle head + decoder stack (FP64 and
the FP32 production path), translation invariance, memory-token /
pocket-atom / ligand-node permutation invariance, batch-composition
invariance, gradient connectivity (through the checkpointed backbone and the
handle head; no cross-graph gradient leaks), the fail-closed input contract
(dtype policy D-V2-005/D-V2-008, shapes, index ranges, cross-field graph
consistency, poisoned inputs), the envelope rule (decision D-V2-012: the
decoder projects no spatial distances at all), the frozen production profile
(1,219,328 parameters), the attention-footprint arithmetic (< 20 MB at the
documented T4 batch target — arithmetic, not a T4 measurement), the audit
interface, and determinism.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from syntree.core.schema import HANDLE_FEATURE_DIM, TensorContractError
from syntree.models.bipartite import (
    BackboneOutput,
    BipartitePaiNN,
    GaussianRadialBasis,
)
from syntree.models.handle_head import HandleSelectionHead
from syntree.models.policy import (
    DECODER_D_MODEL,
    DECODER_DROPOUT,
    DECODER_FFN_DIM,
    DECODER_NUM_HEADS,
    DECODER_NUM_LAYERS,
    CrossAttentionDecoder,
    CrossAttentionDecoderLayer,
    DecoderOutput,
)

# Small configuration used throughout most of the suite (fast); the frozen
# production profile is covered explicitly by TestFrozenProfile. Mirrors the
# #22 suite's small backbone/head configuration so the stacks compose.
D = 16
LAYERS = 2
RADIAL = 6
HEAD_HIDDEN = 8
DEC_D = 16
DEC_HEADS = 4
DEC_FFN = 32


# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_handle_selection.py conventions)
# ---------------------------------------------------------------------------


def random_rotation(generator: torch.Generator) -> torch.Tensor:
    """A random *proper* rotation (orthogonal, det = +1) in float64."""
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def make_node_state(
    n_pocket: int = 10,
    n_ligand: int = 5,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    batch: int = 0,
) -> SimpleNamespace:
    """A minimal single-graph state exposing the eight node-level fields."""
    gen = torch.Generator().manual_seed(seed)
    elements = torch.tensor((1, 6, 7, 8, 9, 15, 16, 17), dtype=torch.long)

    def positions(count: int) -> torch.Tensor:
        return (torch.rand((count, 3), generator=gen, dtype=dtype) * 2.0 - 1.0) * 3.0

    def z_of(count: int) -> torch.Tensor:
        idx = torch.randint(0, len(elements), (count,), generator=gen)
        return elements[idx]

    return SimpleNamespace(
        pocket_pos=positions(n_pocket),
        pocket_z=z_of(n_pocket),
        pocket_charge=torch.zeros(n_pocket, dtype=dtype),
        pocket_batch=torch.full((n_pocket,), batch, dtype=torch.long),
        ligand_pos=positions(n_ligand),
        ligand_z=z_of(n_ligand),
        ligand_charge=torch.zeros(n_ligand, dtype=dtype),
        ligand_batch=torch.full((n_ligand,), batch, dtype=torch.long),
    )


def make_batched_state(
    pocket_counts, ligand_counts, *, seed: int = 0, dtype: torch.dtype = torch.float64
) -> SimpleNamespace:
    """A multi-graph state assembled from per-graph single-graph samples."""
    states = []
    for g, (n_p, n_l) in enumerate(zip(pocket_counts, ligand_counts)):
        states.append(make_node_state(n_p, n_l, seed=seed + g, dtype=dtype, batch=g))
    return SimpleNamespace(
        pocket_pos=torch.cat([s.pocket_pos for s in states]),
        pocket_z=torch.cat([s.pocket_z for s in states]),
        pocket_charge=torch.cat([s.pocket_charge for s in states]),
        pocket_batch=torch.cat([s.pocket_batch for s in states]),
        ligand_pos=torch.cat([s.ligand_pos for s in states]),
        ligand_z=torch.cat([s.ligand_z for s in states]),
        ligand_charge=torch.cat([s.ligand_charge for s in states]),
        ligand_batch=torch.cat([s.ligand_batch for s in states]),
    )


def small_backbone(dtype: torch.dtype = torch.float64, seed: int = 0) -> BipartitePaiNN:
    torch.manual_seed(seed)
    model = BipartitePaiNN(hidden_dim=D, num_layers=LAYERS, num_radial=RADIAL, r_cut=5.0)
    return model.to(dtype).eval()


def small_head(dtype: torch.dtype = torch.float64, seed: int = 1) -> HandleSelectionHead:
    torch.manual_seed(seed)
    head = HandleSelectionHead(
        backbone_dim=D, hidden_dim=HEAD_HIDDEN, num_radial=RADIAL, geom_cutoff=12.0
    )
    return head.to(dtype).eval()


def small_decoder(dtype: torch.dtype = torch.float64, seed: int = 2) -> CrossAttentionDecoder:
    torch.manual_seed(seed)
    decoder = CrossAttentionDecoder(
        backbone_dim=D,
        handle_dim=HEAD_HIDDEN,
        d_model=DEC_D,
        num_layers=2,
        num_heads=DEC_HEADS,
        ffn_dim=DEC_FFN,
        dropout=0.1,
    )
    return decoder.to(dtype).eval()


def make_pool(
    backbone: BackboneOutput,
    counts_per_graph,
    *,
    seed: int = 3,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A deterministic, batch-composition-invariant handle pool.

    Per graph ``g`` the first ``counts_per_graph[g]`` ligand nodes (in node
    order) become the attachment atoms, at their exact node positions (the
    reacting-atom contract). Descriptor row ``r`` is drawn from a generator
    seeded by ``(seed, r)`` alone, so a graph's rows are identical whether
    it is scored alone or inside any batch (fixture-level batch-composition
    invariance — the property under test is the module's, so the fixture
    must not confound it).
    """
    num_pocket = backbone.num_pocket
    ligand_batch = backbone.node_batch[num_pocket:]
    ligand_pos = backbone.node_pos[num_pocket:]
    rows_graph, rows_atom = [], []
    for g, count in enumerate(counts_per_graph):
        nodes = torch.nonzero(ligand_batch == g, as_tuple=False).flatten()
        assert nodes.numel() >= count, "fixture requires enough ligand atoms"
        chosen = nodes[:count]
        rows_atom.append(chosen)
        rows_graph.append(torch.full((count,), g, dtype=torch.long))
    handle_atom = torch.cat(rows_atom) if rows_graph else torch.empty(0, dtype=torch.long)
    handle_graph = torch.cat(rows_graph) if rows_graph else torch.empty(0, dtype=torch.long)
    handle_pos = ligand_pos.index_select(0, handle_atom).clone()
    descriptors = torch.stack(
        [
            torch.randn(
                (HANDLE_FEATURE_DIM,), generator=torch.Generator().manual_seed(seed + 10_000 + r)
            ).to(dtype)
            for r in range(int(handle_atom.size(0)))
        ]
    ) if handle_atom.numel() else torch.empty((0, HANDLE_FEATURE_DIM), dtype=dtype)
    return descriptors, handle_pos, handle_graph, handle_atom


def padded_attention_values(output: DecoderOutput) -> torch.Tensor:
    """All attention weights that fall on padded memory slots."""
    b, n = output.key_padding_mask.shape
    return output.attention_weights.detach().masked_select(
        output.key_padding_mask.view(1, b, 1, 1, n)
    )


# NOTE on loss choices in the gradient tests: the stack closes with a
# LayerNorm, so *each row* of `context` sums to the constant sum(beta) —
# `context.sum()` and `context[i].sum()` have exactly zero gradient w.r.t.
# everything. Gradient tests therefore use weighted losses (a fixed
# linspace weight vector), which is also what any downstream head loss
# will look like.


# ---------------------------------------------------------------------------
# 1. The pinned acceptance criteria (#23)
# ---------------------------------------------------------------------------


class TestAcceptance:
    """[B, 1, 256] query + [B, N, 256] memory -> [B, 256]; padded atoms masked."""

    def test_forward_shapes_at_frozen_profile(self):
        torch.manual_seed(0)
        decoder = CrossAttentionDecoder().eval()  # frozen profile: d_model 256
        assert decoder.d_model == 256
        b, n = 3, 17
        query = torch.randn(b, 1, 256)
        memory = torch.randn(b, n, 256)
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[0, 12:] = True
        mask[2, 16:] = True
        out = decoder(query, memory, mask)
        assert isinstance(out, DecoderOutput)
        assert tuple(out.context.shape) == (b, 256)

    def test_attention_weights_mask_padded_atoms_exactly(self):
        torch.manual_seed(1)
        decoder = CrossAttentionDecoder().eval()
        b, n = 4, 23
        query = torch.randn(b, 1, 256)
        memory = torch.randn(b, n, 256)
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[0, 9:] = True
        mask[1, 4:] = True
        mask[2, 21:] = True
        # graph 3 is fully unpadded
        out = decoder(query, memory, mask)
        assert tuple(out.attention_weights.shape) == (2, b, 8, 1, n)
        padded = padded_attention_values(out)
        assert padded.numel() == 2 * 8 * (14 + 19 + 2)  # layers * heads * pads
        # Exactly zero at every padded column, in every layer, head and graph.
        assert float(padded.abs().max()) == 0.0

    def test_output_is_bit_identical_under_padded_slot_junk(self):
        torch.manual_seed(2)
        decoder = CrossAttentionDecoder().eval()
        b, n = 3, 15
        query = torch.randn(b, 1, 256)
        memory = torch.randn(b, n, 256)
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[0, 7:] = True
        mask[1, 11:] = True
        mask[2, 14:] = True
        out1 = decoder(query, memory, mask)
        junk = memory.clone()
        junk[0, 7:] = 999.0
        junk[1, 11:] = -777.0
        junk[2, 14:] = 1e6  # graph 2 pad slot (single pad)
        out2 = decoder(query, junk, mask)
        assert torch.equal(out1.context, out2.context)
        assert torch.equal(out1.attention_weights, out2.attention_weights)

    def test_attention_rows_sum_to_one_over_real_tokens(self):
        torch.manual_seed(3)
        decoder = CrossAttentionDecoder().eval()
        b, n = 3, 12
        query = torch.randn(b, 1, 256)
        memory = torch.randn(b, n, 256)
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[0, 5:] = True
        mask[2, 9:] = True
        out = decoder(query, memory, mask)
        real = (~mask).float()  # [B, N]
        sums = (out.attention_weights.squeeze(3) * real.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_train_mode_dropout_keeps_padded_columns_zero(self):
        torch.manual_seed(4)
        decoder = CrossAttentionDecoder().train()  # dropout 0.1 active
        b, n = 2, 16
        query = torch.randn(b, 1, 256)
        memory = torch.randn(b, n, 256)
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[0, 10:] = True
        out = decoder(query, memory, mask)
        padded = padded_attention_values(out)
        assert padded.numel() > 0
        assert float(padded.abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 2. The issue-pinned layer structure
# ---------------------------------------------------------------------------


class TestLayerStructure:
    """MHA(256, 8, 0.1) + FFN 256->512->SiLU->Dropout->256 + pre-LN residuals."""

    def test_mha_configuration_is_pinned(self):
        layer = CrossAttentionDecoderLayer()
        mha = layer.cross_attn
        assert isinstance(mha, torch.nn.MultiheadAttention)
        assert mha.embed_dim == 256
        assert mha.num_heads == 8
        assert mha.dropout == 0.1
        assert mha.batch_first is True

    def test_ffn_structure_is_pinned(self):
        layer = CrossAttentionDecoderLayer()
        ffn = layer.ffn
        assert isinstance(ffn[0], torch.nn.Linear)
        assert ffn[0].in_features == 256 and ffn[0].out_features == 512
        assert isinstance(ffn[1], torch.nn.SiLU)
        assert isinstance(ffn[2], torch.nn.Dropout)
        assert ffn[2].p == 0.1
        assert isinstance(ffn[3], torch.nn.Linear)
        assert ffn[3].in_features == 512 and ffn[3].out_features == 256
        assert len(ffn) == 4

    def test_pre_layernorm_on_both_sublayers(self):
        layer = CrossAttentionDecoderLayer()
        assert isinstance(layer.attn_norm, torch.nn.LayerNorm)
        assert isinstance(layer.ffn_norm, torch.nn.LayerNorm)
        assert layer.attn_norm.normalized_shape == (256,)
        assert layer.ffn_norm.normalized_shape == (256,)

    def test_preln_residual_is_identity_when_sublayers_zeroed(self):
        # Zeroing the attention output projection and the FFN's last linear
        # makes both sub-layer contributions exactly zero. With *pre*-LN
        # residuals the layer is then the exact identity; a post-LN variant
        # would return LayerNorm(x) instead.
        torch.manual_seed(5)
        layer = CrossAttentionDecoderLayer().eval()
        with torch.no_grad():
            torch.nn.init.zeros_(layer.cross_attn.out_proj.weight)
            torch.nn.init.zeros_(layer.cross_attn.out_proj.bias)
            torch.nn.init.zeros_(layer.ffn[3].weight)
            torch.nn.init.zeros_(layer.ffn[3].bias)
        x = torch.randn(3, 1, 256)
        memory = torch.randn(3, 9, 256)
        mask = torch.zeros(3, 9, dtype=torch.bool)
        mask[1, 6:] = True
        out, _ = layer(x, memory, mask)
        assert torch.equal(out, x)

    def test_layer_accepts_general_query_length(self):
        # The stack pins Lq = 1 (one handle query per graph); the layer
        # primitive itself supports general Lq.
        torch.manual_seed(6)
        layer = CrossAttentionDecoderLayer(d_model=32, num_heads=4, ffn_dim=64).eval()
        x = torch.randn(2, 3, 32)
        memory = torch.randn(2, 11, 32)
        mask = torch.zeros(2, 11, dtype=torch.bool)
        mask[0, 8:] = True
        out, attn = layer(x, memory, mask)
        assert tuple(out.shape) == (2, 3, 32)
        assert tuple(attn.shape) == (2, 4, 3, 11)

    def test_layer_rejects_incompatible_shapes(self):
        layer = CrossAttentionDecoderLayer()
        x = torch.randn(2, 1, 256)
        memory = torch.randn(2, 9, 256)
        mask = torch.zeros(2, 9, dtype=torch.bool)
        with pytest.raises(TensorContractError, match="batch"):
            layer(torch.randn(3, 1, 256), memory, mask)
        with pytest.raises(TensorContractError, match="d_model"):
            layer(x, torch.randn(2, 9, 128), mask)
        with pytest.raises(TensorContractError, match="key_padding_mask"):
            layer(x, memory, torch.zeros(2, 8, dtype=torch.bool))
        with pytest.raises(TensorContractError, match="bool"):
            layer(x, memory, torch.zeros(2, 9))

    def test_layer_rejects_fully_masked_graph(self):
        layer = CrossAttentionDecoderLayer()
        x = torch.randn(2, 1, 256)
        memory = torch.randn(2, 9, 256)
        mask = torch.zeros(2, 9, dtype=torch.bool)
        mask[1, :] = True  # graph 1 fully padded
        with pytest.raises(TensorContractError, match="fully"):
            layer(x, memory, mask)

    def test_layer_constructor_validation(self):
        with pytest.raises(ValueError, match="divisible"):
            CrossAttentionDecoderLayer(d_model=30, num_heads=4)
        with pytest.raises(ValueError, match="dropout"):
            CrossAttentionDecoderLayer(dropout=1.0)
        with pytest.raises(ValueError, match="num_heads"):
            CrossAttentionDecoderLayer(num_heads=0)


# ---------------------------------------------------------------------------
# 3. Per-graph padding of the flat union memory
# ---------------------------------------------------------------------------


class TestPadding:
    """Order-preserving per-graph padding with an exact key_padding_mask."""

    def _run(self, dtype=torch.float64):
        state = make_batched_state([9, 12, 7], [5, 3, 6], seed=20)
        backbone = small_backbone(dtype=dtype)
        head = small_head(dtype=dtype)
        decoder = small_decoder(dtype=dtype)
        bb = backbone(state)
        pool = make_pool(bb, [3, 2, 4], dtype=dtype)
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)
        return bb, out

    def test_padded_memory_roundtrip_preserves_rows_and_order(self):
        bb, out = self._run()
        for g in range(bb.num_graphs):
            rows = torch.nonzero(bb.node_batch == g, as_tuple=False).flatten()
            count = int(rows.numel())
            assert int(out.num_memory_tokens[g].item()) == count
            expected = bb.scalar.index_select(0, rows)  # original order
            actual = out.memory[g, :count]
            assert torch.equal(actual, expected)
            # padded tail is zero-filled
            if count < out.memory_length:
                assert not bool(out.memory[g, count:].any())

    def test_key_padding_mask_marks_exactly_the_pads(self):
        bb, out = self._run()
        for g in range(bb.num_graphs):
            count = int(out.num_memory_tokens[g].item())
            assert not bool(out.key_padding_mask[g, :count].any())
            assert bool(out.key_padding_mask[g, count:].all())

    def test_memory_length_is_the_largest_graph(self):
        bb, out = self._run()
        counts = torch.bincount(bb.node_batch)
        assert out.memory_length == int(counts.max().item())

    def test_all_padded_graph_rejected_on_dense_path(self):
        torch.manual_seed(7)
        decoder = small_decoder().eval()
        query = torch.randn(2, 1, DEC_D, dtype=torch.float64)
        memory = torch.randn(2, 8, DEC_D, dtype=torch.float64)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[1, :] = True
        with pytest.raises(TensorContractError, match="fully"):
            decoder(query, memory, mask)

    def test_graph_without_nodes_rejected(self):
        # A synthetic BackboneOutput whose node_batch skips a graph: the
        # padding helper must fail closed (every graph needs >= 1 token).
        state = make_batched_state([8, 8], [4, 4], seed=21)
        backbone = small_backbone()
        head = small_head()
        decoder = small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [2, 2])
        hh = head(bb, *pool)
        broken = BackboneOutput(
            scalar=bb.scalar,
            vector=bb.vector,
            node_pos=bb.node_pos,
            node_type=bb.node_type,
            node_batch=torch.where(bb.node_batch == 1, 2, bb.node_batch),
            num_pocket=bb.num_pocket,
            num_graphs=bb.num_graphs,
            graph_context=bb.graph_context,
            pocket_context=bb.pocket_context,
        )
        with pytest.raises(TensorContractError):
            decoder.forward_state(broken, hh)


# ---------------------------------------------------------------------------
# 4. Dense forward: contract and fail-closed matrix
# ---------------------------------------------------------------------------


class TestDenseForward:
    """The [B, 1, D] x [B, N, D] -> [B, D] interface, validated fail-closed."""

    def _sample(self, b=3, n=13, seed=8, dtype=torch.float64):
        torch.manual_seed(seed)
        query = torch.randn(b, 1, DEC_D, dtype=dtype)
        memory = torch.randn(b, n, DEC_D, dtype=dtype)
        mask = torch.zeros(b, n, dtype=torch.bool)
        mask[0, 9:] = True
        mask[2, 11:] = True
        return query, memory, mask

    def test_query_must_have_exactly_one_token(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        with pytest.raises(TensorContractError, match="one token per graph"):
            decoder(torch.randn(3, 2, DEC_D, dtype=torch.float64), memory, mask)

    def test_fp16_rejected_fail_closed(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        with pytest.raises(TensorContractError, match="float32"):
            decoder(query.to(torch.float16), memory, mask)

    def test_dtype_must_match_module_parameters(self):
        decoder = small_decoder(dtype=torch.float32).eval()  # fp32 module
        query, memory, mask = self._sample(dtype=torch.float32)
        out = decoder(query, memory, mask)  # production path OK
        assert tuple(out.context.shape) == (3, DEC_D)
        q64, m64, k64 = self._sample(dtype=torch.float64)
        with pytest.raises(TensorContractError, match="cast the module"):
            decoder(q64, m64, k64)

    def test_fp64_module_accepts_fp64_for_local_verification(self):
        decoder = small_decoder(dtype=torch.float64).eval()
        query, memory, mask = self._sample(dtype=torch.float64)
        out = decoder(query, memory, mask)
        assert bool(torch.isfinite(out.context).all())

    def test_non_tensor_inputs_rejected(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        with pytest.raises(TensorContractError, match="torch.Tensor"):
            decoder([1.0], memory, mask)
        with pytest.raises(TensorContractError, match="torch.Tensor"):
            decoder(query, "memory", mask)
        with pytest.raises(TensorContractError, match="torch.Tensor"):
            decoder(query, memory, None)

    def test_nan_inputs_fail_closed(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        bad_q = query.clone()
        bad_q[0, 0, 0] = float("nan")
        with pytest.raises(TensorContractError, match="non-finite"):
            decoder(bad_q, memory, mask)
        bad_m = memory.clone()
        bad_m[1, 2, 3] = float("inf")
        with pytest.raises(TensorContractError, match="non-finite"):
            decoder(query, bad_m, mask)

    def test_mask_must_be_bool(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        with pytest.raises(TensorContractError, match="bool"):
            decoder(query, memory, mask.to(torch.float32))
        with pytest.raises(TensorContractError, match="bool"):
            decoder(query, memory, torch.ones_like(mask, dtype=torch.long))

    def test_mask_shape_mismatch_rejected(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        with pytest.raises(TensorContractError, match="key_padding_mask"):
            decoder(query, memory, mask[:, :-1])
        with pytest.raises(TensorContractError, match="key_padding_mask"):
            decoder(query, memory, torch.zeros(4, 13, dtype=torch.bool))

    def test_batch_and_width_mismatch_rejected(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        with pytest.raises(TensorContractError, match="batch"):
            decoder(query, torch.randn(2, 13, DEC_D, dtype=torch.float64), mask)
        with pytest.raises(TensorContractError, match="d_model"):
            decoder(query, torch.randn(3, 13, DEC_D + 1, dtype=torch.float64), mask)
        with pytest.raises(TensorContractError, match="d_model"):
            decoder(torch.randn(3, 1, DEC_D + 2, dtype=torch.float64), memory, mask)

    def test_empty_memory_rejected(self):
        decoder = small_decoder().eval()
        query = torch.randn(2, 1, DEC_D, dtype=torch.float64)
        with pytest.raises(TensorContractError, match="at least one slot"):
            decoder(query, torch.randn(2, 0, DEC_D, dtype=torch.float64), torch.zeros(2, 0, dtype=torch.bool))

    def test_single_memory_slot_graph_is_legal(self):
        decoder = small_decoder().eval()
        query = torch.randn(1, 1, DEC_D, dtype=torch.float64)
        memory = torch.randn(1, 1, DEC_D, dtype=torch.float64)
        out = decoder(query, memory, torch.zeros(1, 1, dtype=torch.bool))
        assert tuple(out.context.shape) == (1, DEC_D)
        assert int(out.num_memory_tokens[0].item()) == 1

    def test_eval_determinism_bit_identical(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        out1 = decoder(query, memory, mask)
        out2 = decoder(query, memory, mask)
        assert torch.equal(out1.context, out2.context)
        assert torch.equal(out1.attention_weights, out2.attention_weights)

    def test_state_dict_roundtrip_bit_identical(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        out1 = decoder(query, memory, mask)
        clone = small_decoder().eval()
        clone.load_state_dict(decoder.state_dict())
        out2 = clone(query, memory, mask)
        assert torch.equal(out1.context, out2.context)

    def test_train_mode_is_seeded_reproducible(self):
        decoder = small_decoder().train()  # dropout active
        query, memory, mask = self._sample()
        torch.manual_seed(123)
        out1 = decoder(query, memory, mask)
        torch.manual_seed(123)
        out2 = decoder(query, memory, mask)
        assert torch.equal(out1.context, out2.context)
        # and dropout really is active: a different seed changes the output
        torch.manual_seed(321)
        out3 = decoder(query, memory, mask)
        assert not torch.equal(out1.context, out3.context)

    def test_memory_token_permutation_invariance(self):
        # Permuting the memory slots (with the mask carried along) must not
        # change the refined context: attention is order-equivariant over
        # the token axis and the query is untouched.
        decoder = small_decoder().eval()
        query, memory, mask = self._sample()
        perm = torch.randperm(memory.size(1))
        out1 = decoder(query, memory, mask)
        out2 = decoder(query, memory[:, perm], mask[:, perm])
        assert torch.allclose(out1.context, out2.context, atol=1e-10)
        assert torch.allclose(
            out1.attention_weights[..., perm], out2.attention_weights, atol=1e-12
        )

    def test_no_gradient_leaks_across_graphs(self):
        decoder = small_decoder().eval()
        query, memory, mask = self._sample(b=3)
        memory = memory.detach().requires_grad_(True)
        out = decoder(query, memory, mask)
        weights = torch.linspace(1.0, 2.0, out.context.size(1), dtype=torch.float64)
        (out.context[0] * weights).sum().backward()
        assert memory.grad is not None
        assert float(memory.grad[1].abs().sum()) == 0.0  # graph 1 untouched
        assert float(memory.grad[2].abs().sum()) == 0.0  # graph 2 untouched
        assert float(memory.grad[0].abs().sum()) > 0.0   # graph 0 receives flow


# ---------------------------------------------------------------------------
# 5. The V2-native interface (backbone + handle head -> decoder)
# ---------------------------------------------------------------------------


class TestForwardState:
    """forward_state consumes the #21/#22 surfaces without re-encoding."""

    def test_end_to_end_shapes_and_finiteness(self):
        state = make_batched_state([11, 9], [6, 4], seed=30)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [3, 2])
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)
        assert isinstance(out, DecoderOutput)
        assert tuple(out.context.shape) == (2, DEC_D)
        assert bool(torch.isfinite(out.context).all())
        assert tuple(out.sequence.shape) == (2, 1, DEC_D)
        assert torch.equal(out.sequence.squeeze(1), out.context)
        assert out.num_layers == 2

    def test_fp32_production_path(self):
        state = make_batched_state([10, 8], [5, 4], seed=31, dtype=torch.float32)
        backbone = small_backbone(dtype=torch.float32)
        head = small_head(dtype=torch.float32)
        decoder = small_decoder(dtype=torch.float32)
        bb = backbone(state)
        pool = make_pool(bb, [2, 3], dtype=torch.float32)
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)
        assert tuple(out.context.shape) == (2, DEC_D)
        assert bool(torch.isfinite(out.context).all())

    def test_forward_state_matches_dense_plumbing_exactly(self):
        state = make_batched_state([9, 12], [5, 6], seed=32)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [3, 4])
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)
        out2 = decoder(out.query, out.memory, out.key_padding_mask)
        assert torch.equal(out.context, out2.context)
        assert torch.equal(out.attention_weights, out2.attention_weights)

    def test_query_receives_gradient_from_all_three_contexts(self):
        state = make_batched_state([9, 8], [5, 4], seed=33, dtype=torch.float64)
        backbone = small_backbone().requires_grad_(False)
        head = small_head().requires_grad_(False)
        decoder = small_decoder()
        with torch.no_grad():
            bb = backbone(state)
            pool = make_pool(bb, [2, 2])
            hh = head(bb, *pool)
        bb.scalar.requires_grad_(True)
        bb.graph_context.requires_grad_(True)
        bb.pocket_context.requires_grad_(True)
        hh.pooled_handle_context.requires_grad_(True)
        out = decoder.forward_state(bb, hh)
        weights = torch.linspace(1.0, 2.0, out.context.size(1), dtype=torch.float64)
        (out.context * weights).sum().backward()
        assert bb.scalar.grad is not None and float(bb.scalar.grad.abs().sum()) > 0.0
        assert (
            bb.graph_context.grad is not None
            and float(bb.graph_context.grad.abs().sum()) > 0.0
        )
        assert (
            bb.pocket_context.grad is not None
            and float(bb.pocket_context.grad.abs().sum()) > 0.0
        )
        assert (
            hh.pooled_handle_context.grad is not None
            and float(hh.pooled_handle_context.grad.abs().sum()) > 0.0
        )

    def test_graph_without_handles_yields_finite_context(self):
        # Graph 1 has no active handle rows: its pooled handle context is
        # the zero row and the query degenerates to the backbone contexts.
        state = make_batched_state([10, 9, 8], [5, 5, 4], seed=34)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [3, 0, 2])
        hh = head(bb, *pool)
        assert int(hh.handle_counts[1].item()) == 0
        assert not bool(hh.pooled_handle_context[1].any())
        out = decoder.forward_state(bb, hh)
        assert bool(torch.isfinite(out.context).all())
        assert tuple(out.context.shape) == (3, DEC_D)

    def test_single_graph_batch(self):
        state = make_batched_state([12], [6], seed=35)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [4])
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)
        assert tuple(out.context.shape) == (1, DEC_D)
        assert int(out.num_memory_tokens[0].item()) == 18
        assert out.memory_length == 18  # no padding for a single graph

    def test_se3_invariance_end_to_end_fp64(self):
        state = make_batched_state([10, 11], [6, 5], seed=36)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [3, 2])
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)

        gen = torch.Generator().manual_seed(99)
        rot = random_rotation(gen)
        shift = torch.randn(3, generator=gen, dtype=torch.float64) * 10.0
        moved = SimpleNamespace(
            pocket_pos=state.pocket_pos @ rot.T + shift,
            pocket_z=state.pocket_z,
            pocket_charge=state.pocket_charge,
            pocket_batch=state.pocket_batch,
            ligand_pos=state.ligand_pos @ rot.T + shift,
            ligand_z=state.ligand_z,
            ligand_charge=state.ligand_charge,
            ligand_batch=state.ligand_batch,
        )
        descriptors, handle_pos, handle_graph, handle_atom = pool
        moved_pool = (
            descriptors,
            handle_pos @ rot.T + shift,
            handle_graph,
            handle_atom,
        )
        bb2 = backbone(moved)
        hh2 = head(bb2, *moved_pool)
        out2 = decoder.forward_state(bb2, hh2)
        worst = float((out.context - out2.context).detach().abs().max())
        assert worst < 1e-10  # measured ~1e-13; generous headroom

    def test_translation_invariance_exact(self):
        state = make_batched_state([10, 11], [6, 5], seed=37)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [3, 2])
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)
        shift = torch.tensor([3.5, -12.25, 7.0], dtype=torch.float64)
        moved = SimpleNamespace(
            pocket_pos=state.pocket_pos + shift,
            pocket_z=state.pocket_z,
            pocket_charge=state.pocket_charge,
            pocket_batch=state.pocket_batch,
            ligand_pos=state.ligand_pos + shift,
            ligand_z=state.ligand_z,
            ligand_charge=state.ligand_charge,
            ligand_batch=state.ligand_batch,
        )
        moved_pool = (pool[0], pool[1] + shift, pool[2], pool[3])
        out2 = decoder.forward_state(backbone(moved), head(backbone(moved), *moved_pool))
        worst = float((out.context - out2.context).detach().abs().max())
        assert worst < 1e-10

    def test_se3_invariance_fp32_production_path(self):
        state = make_batched_state([10, 9], [5, 5], seed=38, dtype=torch.float32)
        backbone = small_backbone(dtype=torch.float32)
        head = small_head(dtype=torch.float32)
        decoder = small_decoder(dtype=torch.float32)
        bb = backbone(state)
        pool = make_pool(bb, [3, 2], dtype=torch.float32)
        out = decoder.forward_state(bb, head(bb, *pool))

        gen = torch.Generator().manual_seed(98)
        rot = random_rotation(gen).to(torch.float32)
        shift = (torch.randn(3, generator=gen) * 8.0).to(torch.float32)
        moved = SimpleNamespace(
            pocket_pos=(state.pocket_pos.double() @ rot.double().T).float() + shift,
            pocket_z=state.pocket_z,
            pocket_charge=state.pocket_charge,
            pocket_batch=state.pocket_batch,
            ligand_pos=(state.ligand_pos.double() @ rot.double().T).float() + shift,
            ligand_z=state.ligand_z,
            ligand_charge=state.ligand_charge,
            ligand_batch=state.ligand_batch,
        )
        moved_pool = (
            pool[0],
            (pool[1].double() @ rot.double().T).float() + shift,
            pool[2],
            pool[3],
        )
        bb2 = backbone(moved)
        out2 = decoder.forward_state(bb2, head(bb2, *moved_pool))
        worst = float((out.context - out2.context).detach().abs().max())
        assert worst < 1e-4  # FP32 production tolerance (measured ~1e-6)

    def test_pocket_atom_permutation_invariance(self):
        state = make_batched_state([10, 11], [6, 5], seed=39)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [3, 2])
        out = decoder.forward_state(bb, head(bb, *pool))
        gen = torch.Generator().manual_seed(97)
        perm = torch.randperm(state.pocket_pos.size(0), generator=gen)
        moved = SimpleNamespace(
            pocket_pos=state.pocket_pos[perm],
            pocket_z=state.pocket_z[perm],
            pocket_charge=state.pocket_charge[perm],
            pocket_batch=state.pocket_batch[perm],
            ligand_pos=state.ligand_pos,
            ligand_z=state.ligand_z,
            ligand_charge=state.ligand_charge,
            ligand_batch=state.ligand_batch,
        )
        bb2 = backbone(moved)
        out2 = decoder.forward_state(bb2, head(bb2, *pool))
        assert torch.allclose(out.context, out2.context, atol=1e-10)

    def test_ligand_permutation_with_handle_remap(self):
        state = make_batched_state([10, 11], [6, 5], seed=40)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [3, 2])
        out = decoder.forward_state(bb, head(bb, *pool))
        gen = torch.Generator().manual_seed(96)
        perm = torch.randperm(state.ligand_pos.size(0), generator=gen)
        inv = torch.argsort(perm)  # old index -> new index
        descriptors, handle_pos, handle_graph, handle_atom = pool
        moved = SimpleNamespace(
            pocket_pos=state.pocket_pos,
            pocket_z=state.pocket_z,
            pocket_charge=state.pocket_charge,
            pocket_batch=state.pocket_batch,
            ligand_pos=state.ligand_pos[perm],
            ligand_z=state.ligand_z[perm],
            ligand_charge=state.ligand_charge[perm],
            ligand_batch=state.ligand_batch[perm],
        )
        moved_pool = (
            descriptors,
            handle_pos,  # same physical reacting atoms
            handle_graph,
            inv.index_select(0, handle_atom),  # remapped attachment indices
        )
        bb2 = backbone(moved)
        out2 = decoder.forward_state(bb2, head(bb2, *moved_pool))
        assert torch.allclose(out.context, out2.context, atol=1e-10)

    def test_batch_composition_invariance(self):
        # Graph 0's context must not depend on which other graphs share the
        # batch (the fixture pool is composition-invariant by construction).
        state = make_batched_state([10, 11], [6, 5], seed=41)
        backbone, head, decoder = small_backbone(), small_head(), small_decoder()
        bb = backbone(state)
        pool = make_pool(bb, [3, 2])
        out = decoder.forward_state(bb, head(bb, *pool))

        single = make_node_state(10, 6, seed=41, batch=0)
        bb1 = backbone(single)
        pool1 = make_pool(bb1, [3])
        out1 = decoder.forward_state(bb1, head(bb1, *pool1))
        assert torch.allclose(out.context[0], out1.context[0], atol=1e-10)

    def test_gradient_through_full_stack(self):
        # FP32 production path, gradients enabled end-to-end through the
        # activation-checkpointed backbone, the handle head, and decoder.
        state = make_batched_state([9, 8], [5, 4], seed=42, dtype=torch.float32)
        backbone = small_backbone(dtype=torch.float32)
        head = small_head(dtype=torch.float32)
        decoder = small_decoder(dtype=torch.float32)
        bb = backbone(state)
        pool = make_pool(bb, [2, 2], dtype=torch.float32)
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)
        weights = torch.linspace(1.0, 2.0, out.context.size(1), dtype=torch.float32)
        (out.context * weights).sum().backward()
        assert backbone.atomic_emb.weight.grad is not None
        assert float(backbone.atomic_emb.weight.grad.abs().sum()) > 0.0
        assert head.descriptor_mlp[0].weight.grad is not None
        assert float(head.descriptor_mlp[0].weight.grad.abs().sum()) > 0.0
        assert decoder.query_handle.weight.grad is not None
        assert float(decoder.query_handle.weight.grad.abs().sum()) > 0.0
        assert decoder.layers[0].cross_attn.in_proj_weight.grad is not None
        assert float(decoder.layers[0].cross_attn.in_proj_weight.grad.abs().sum()) > 0.0

    def test_non_backbone_output_rejected(self):
        decoder = small_decoder()
        state = make_batched_state([9, 8], [5, 4], seed=43)
        backbone, head = small_backbone(), small_head()
        bb = backbone(state)
        hh = head(bb, *make_pool(bb, [2, 2]))
        with pytest.raises(TensorContractError, match="BackboneOutput"):
            decoder.forward_state(state, hh)
        with pytest.raises(TensorContractError, match="HandleSelectionOutput"):
            decoder.forward_state(bb, bb)

    def test_cross_field_graph_count_mismatch_rejected(self):
        import dataclasses

        decoder = small_decoder()
        state = make_batched_state([9, 8], [5, 4], seed=44)
        backbone, head = small_backbone(), small_head()
        bb = backbone(state)
        hh = head(bb, *make_pool(bb, [2, 2]))
        broken = dataclasses.replace(hh, num_graphs=hh.num_graphs + 1)
        with pytest.raises(TensorContractError, match="cross-field"):
            decoder.forward_state(bb, broken)

    def test_wrong_surface_widths_rejected(self):
        # A decoder built for a 16-D backbone / 8-D handle surface must
        # reject 12-D / 5-D producer outputs.
        decoder = small_decoder()
        state = make_batched_state([9, 8], [5, 4], seed=45)
        backbone = small_backbone()
        head = small_head()
        bb = backbone(state)
        hh = head(bb, *make_pool(bb, [2, 2]))
        import dataclasses

        narrow_scalar = dataclasses.replace(bb, scalar=bb.scalar[:, :12])
        with pytest.raises(TensorContractError, match="backbone.scalar"):
            decoder.forward_state(narrow_scalar, hh)
        narrow_contexts = dataclasses.replace(
            bb,
            graph_context=bb.graph_context[:, :12],
            pocket_context=bb.pocket_context[:, :12],
        )
        with pytest.raises(TensorContractError, match="graph_context"):
            decoder.forward_state(narrow_contexts, hh)
        narrow_handle = dataclasses.replace(hh, pooled_handle_context=hh.pooled_handle_context[:, :5])
        with pytest.raises(TensorContractError, match="pooled_handle_context"):
            decoder.forward_state(bb, narrow_handle)

    def test_dtype_mismatch_on_state_path_rejected(self):
        # FP64 backbone output into an FP32 decoder: fail closed.
        decoder = small_decoder(dtype=torch.float32)
        state = make_batched_state([9, 8], [5, 4], seed=46)
        backbone, head = small_backbone(), small_head()
        bb = backbone(state)
        hh = head(bb, *make_pool(bb, [2, 2]))
        with pytest.raises(TensorContractError, match="cast the module"):
            decoder.forward_state(bb, hh)

    def test_poisoned_backbone_scalar_fails_closed(self):
        import dataclasses

        decoder = small_decoder()
        state = make_batched_state([9, 8], [5, 4], seed=47)
        backbone, head = small_backbone(), small_head()
        bb = backbone(state)
        hh = head(bb, *make_pool(bb, [2, 2]))
        poisoned = bb.scalar.clone()
        poisoned[0, 0] = float("nan")
        broken = dataclasses.replace(bb, scalar=poisoned)
        with pytest.raises(TensorContractError, match="non-finite"):
            decoder.forward_state(broken, hh)


# ---------------------------------------------------------------------------
# 6. Frozen production profile, audit, and the envelope rule
# ---------------------------------------------------------------------------


class TestFrozenProfile:
    """d=256, 2 layers, 8 heads, FFN 512, dropout 0.1 — audited and pinned."""

    def test_frozen_constants(self):
        assert DECODER_D_MODEL == 256
        assert DECODER_NUM_LAYERS == 2
        assert DECODER_NUM_HEADS == 8
        assert DECODER_FFN_DIM == 512
        assert DECODER_DROPOUT == 0.1

    def test_frozen_profile_parameter_count(self):
        # Hand-verified arithmetic: layer = MHA (4*256^2 + 4*256 = 263,168)
        # + 2 LayerNorms (2*512) + FFN (256*512+512 + 512*256+256 = 262,912)
        # = 527,104; two layers = 1,054,208; query projections
        # (128*256+256) + 2*(256*256+256) = 164,608; out_norm 512.
        decoder = CrossAttentionDecoder()
        audit = decoder.audit()
        assert audit["num_parameters"] == 1_219_328
        assert audit["num_trainable_parameters"] == 1_219_328
        assert audit["all_parameters_trainable"] is True
        assert audit["parameter_dtypes"] == ["torch.float32"]

    def test_audit_reports_decoder_geometry(self):
        audit = CrossAttentionDecoder().audit()
        assert audit["module"] == "CrossAttentionDecoder"
        assert audit["backbone_dim"] == 256
        assert audit["handle_dim"] == 128
        assert audit["d_model"] == 256
        assert audit["num_layers"] == 2
        assert audit["num_heads"] == 8
        assert audit["ffn_dim"] == 512
        assert audit["dropout"] == 0.1
        assert audit["num_buffers"] == 0
        # D-V2-012: the decoder projects no spatial distances at all —
        # every spatial quantity it consumes already passed through the
        # enveloped GaussianRadialBasis upstream (#21/#22, D-V2-011).
        assert audit["projects_spatial_distances"] is False
        assert audit["deterministic_in_eval"] is True

    def test_no_radial_basis_submodule(self):
        # Structural pin of the envelope rule: the decoder owns no radial
        # basis (enveloped or raw) — there is no expansion site at all.
        decoder = CrossAttentionDecoder()
        assert not any(isinstance(m, GaussianRadialBasis) for m in decoder.modules())

    def test_attention_footprint_arithmetic_below_issue_budget(self):
        # Issue #23 budget: attention-map overhead < 20 MB. Arithmetic at
        # the documented T4 batch target (#45 §5.2: batch 12, pocket cap
        # 700) with a 100-atom ligand margin — a *calculation*, not a T4
        # measurement (no T4 execution has occurred; profiling is #43).
        batch, slots = 12, 800
        floats = DECODER_NUM_LAYERS * batch * DECODER_NUM_HEADS * 1 * slots
        bytes_fp32 = floats * 4
        assert floats == 153_600
        assert bytes_fp32 == 614_400  # ~0.59 MB — two orders below budget
        assert bytes_fp32 < 20 * 1024 * 1024

    def test_integration_smoke_frozen_profile(self):
        # The full production-profile stack (real 5,999,104-param backbone
        # + 224,896-param handle head + 1,219,328-param decoder) on a small
        # two-graph state — CPU smoke, no T4 claim.
        torch.manual_seed(0)
        state = make_batched_state(
            [8, 7], [4, 3], seed=50, dtype=torch.float32
        )
        backbone = BipartitePaiNN().to(torch.float32).eval()
        head = HandleSelectionHead().to(torch.float32).eval()
        decoder = CrossAttentionDecoder().to(torch.float32).eval()
        bb = backbone(state)
        pool = make_pool(bb, [2, 2], dtype=torch.float32)
        hh = head(bb, *pool)
        out = decoder.forward_state(bb, hh)
        assert tuple(out.context.shape) == (2, 256)
        assert bool(torch.isfinite(out.context).all())
        assert int(out.num_memory_tokens.sum().item()) == 22
        # Masking holds at the frozen profile too.
        assert float(padded_attention_values(out).abs().max()) == 0.0
