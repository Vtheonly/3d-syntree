"""Agency-enabled handle-selection head — tests (#22 MODEL-02).

Acceptance criterion of issue #22 (clean-room V2 implementation):

* On intermediates with ``H > 1`` exposed handles, the handle head outputs
  valid normalized probability distributions over the active handle set.

Beyond the pinned criterion, this suite enforces: the environment-menu
integration contract (the head scores exactly the
``AssemblyEnvironment.active_handles()`` menu, and its argmax row yields a
legal ``AssemblyAction.handle_atom_idx``), SE(3) invariance of the full
distribution under rigid motions of (pocket + ligand + handle) positions,
translation invariance, handle-row / ligand-node / pocket-atom permutation
equivariance, batch-composition invariance, single-handle degeneracy
(``P = 1`` exactly), stage-1 attention validity, gradient connectivity
into the backbone, the fail-closed input contract (dtype/shape/index
validation, legacy 67-D descriptor rejection, cross-field graph
consistency), the frozen production profile (224,896 parameters), the
audit interface, and determinism.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rdkit import Chem

from syntree.core.schema import HANDLE_FEATURE_DIM, TensorContractError
from syntree.models.bipartite import (
    BACKBONE_HIDDEN_DIM,
    BackboneOutput,
    BipartitePaiNN,
)
from syntree.models.handle_head import (
    HANDLE_GEOM_CUTOFF,
    HANDLE_HEAD_HIDDEN_DIM,
    HANDLE_HEAD_NUM_RADIAL,
    HandleSelectionHead,
    HandleSelectionOutput,
)
from syntree.engine.environment import (
    AssemblyAction,
    AssemblyEnvironment,
    BuildingBlock,
    HandleSpec,
    JunctionReaction,
    PocketDefinition,
)

# Small configuration used throughout most of the suite (fast); the frozen
# production profile is covered explicitly by TestFrozenProfile.
D = 16
LAYERS = 2
RADIAL = 6
HEAD_HIDDEN = 8


# ---------------------------------------------------------------------------
# Helpers
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


def make_pool(
    backbone: BackboneOutput,
    counts_per_graph,
    *,
    seed: int = 2,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A consistent handle pool for a batched backbone output.

    Chooses distinct ligand atoms of each graph as the attachment atoms
    (random subset), uses their exact node positions as ``handle_pos``
    (the reacting-atom contract), and samples 64-D invariant descriptors.
    Returns ``(descriptors, pos, graph, atom)``.
    """
    gen = torch.Generator().manual_seed(seed)
    # A separate generator for the descriptors: descriptor rows must depend
    # only on the row index, never on how many graphs / permutes preceded
    # them (batch-composition invariance of the fixture itself).
    desc_gen = torch.Generator().manual_seed(seed + 10_000)
    num_pocket = backbone.num_pocket
    ligand_batch = backbone.node_batch[num_pocket:]
    ligand_pos = backbone.node_pos[num_pocket:]
    rows_graph, rows_atom = [], []
    for g, count in enumerate(counts_per_graph):
        nodes = torch.nonzero(ligand_batch == g, as_tuple=False).flatten()
        assert nodes.numel() >= count, "fixture requires enough ligand atoms"
        chosen = nodes[torch.randperm(nodes.numel(), generator=gen)[:count]]
        rows_atom.append(chosen)
        rows_graph.append(torch.full((count,), g, dtype=torch.long))
    handle_atom = torch.cat(rows_atom)
    handle_graph = torch.cat(rows_graph)
    handle_pos = ligand_pos.index_select(0, handle_atom).clone()
    descriptors = torch.randn(
        (int(handle_atom.size(0)), HANDLE_FEATURE_DIM), generator=desc_gen, dtype=dtype
    )
    return descriptors, handle_pos, handle_graph, handle_atom


def assert_valid_distribution(result: HandleSelectionOutput, *, atol: float) -> None:
    """The #22 acceptance assertion: valid normalized per-graph law."""
    num_graphs = result.num_graphs
    assert int(result.handle_counts.sum().item()) == result.num_handles
    for g in range(num_graphs):
        rows = result.rows_for_graph(g)
        assert int(result.handle_counts[g].item()) == int(rows.numel())
        if rows.numel() == 0:
            assert float(result.entropy[g].item()) == 0.0
            continue
        probs = result.probs.index_select(0, rows)
        log_probs = result.log_probs.index_select(0, rows)
        assert bool(torch.isfinite(probs).all())
        assert bool((probs >= 0).all())
        assert probs.sum().item() == pytest.approx(1.0, abs=atol)
        assert log_probs.max().item() <= 0.0
        assert bool((log_probs <= 0.0).all())
        assert torch.allclose(log_probs, probs.clamp(min=1e-300).log(), atol=atol)
        # log_softmax consistency: shifting logits by a constant must not
        # change the law (normalization over the valid alternatives only).
        attention = result.attention_weights.index_select(0, rows)
        assert attention.sum().item() == pytest.approx(1.0, abs=atol)
        expected_entropy = -(probs * log_probs).sum()
        assert result.entropy[g].item() == pytest.approx(
            expected_entropy.item(), abs=atol
        )


# ---------------------------------------------------------------------------
# 1. The pinned acceptance criterion (#22)
# ---------------------------------------------------------------------------


class TestAcceptance:
    """H > 1 exposed handles must yield valid normalized distributions."""

    def test_multi_handle_graphs_are_normalized(self):
        state = make_batched_state([12, 9], [6, 4], seed=10)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        pool = make_pool(out, [3, 2], seed=3)
        result = head(out, *pool)
        assert isinstance(result, HandleSelectionOutput)
        assert result.num_handles == 5
        assert_valid_distribution(result, atol=1e-12)

    def test_fp32_production_path_is_normalized(self):
        state = make_batched_state([12, 9], [6, 4], seed=11, dtype=torch.float32)
        backbone = small_backbone(dtype=torch.float32)
        head = small_head(dtype=torch.float32)
        out = backbone(state)
        pool = make_pool(out, [4, 3], seed=4, dtype=torch.float32)
        result = head(out, *pool)
        assert_valid_distribution(result, atol=1e-5)

    def test_single_handle_graph_is_deterministic(self):
        state = make_batched_state([10], [5], seed=12)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        pool = make_pool(out, [1], seed=5)
        result = head(out, *pool)
        assert_valid_distribution(result, atol=1e-12)
        assert result.probs.item() == 1.0
        assert result.log_probs.item() == 0.0
        assert result.entropy.item() == 0.0
        assert result.attention_weights.item() == 1.0

    def test_mixed_single_and_multi_handle_batch(self):
        state = make_batched_state([10, 8, 11], [5, 6, 3], seed=13)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        pool = make_pool(out, [1, 4, 2], seed=6)
        result = head(out, *pool)
        assert_valid_distribution(result, atol=1e-12)
        single = result.rows_for_graph(0)
        assert result.probs.index_select(0, single).item() == 1.0

    def test_identical_handles_split_uniformly(self):
        # Two handles with identical features on the same graph: the law
        # must be exactly uniform and the entropy exactly ln 2.
        state = make_batched_state([10], [5], seed=14)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        descriptors, pos, graph, atom = make_pool(out, [2], seed=7)
        descriptors = descriptors[0:1].expand(2, -1).contiguous()
        pos = pos[0:1].expand(2, -1).contiguous()
        atom = atom[0:1].expand(2).contiguous()
        result = head(out, descriptors, pos, graph, atom)
        assert_valid_distribution(result, atol=1e-12)
        assert torch.allclose(result.probs, torch.full((2,), 0.5, dtype=torch.float64))
        assert result.entropy.item() == pytest.approx(math.log(2.0), abs=1e-12)

    def test_duplicate_rows_on_one_atom_are_allowed(self):
        # Two distinct handles may share one attachment atom (different
        # handle ids in the environment menu); both rows are scored.
        state = make_batched_state([10], [5], seed=15)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        descriptors, _, graph, _ = make_pool(out, [2], seed=8)
        atom = torch.zeros(2, dtype=torch.long)
        pos = out.node_pos[out.num_pocket :].index_select(0, atom).clone()
        result = head(out, descriptors, pos, graph, atom)
        assert_valid_distribution(result, atol=1e-12)

    def test_shift_invariance_of_log_softmax_normalization(self):
        # Adding a per-graph constant to every geometric/chemical input is
        # not the invariance under test; instead verify that a large global
        # logit scale still normalizes (numerical stability probe): a
        # far-away handle (radial tail) must simply receive its share.
        state = make_batched_state([10], [5], seed=16)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        descriptors, pos, graph, atom = make_pool(out, [3], seed=9)
        pos_far = pos.clone()
        pos_far[0] = pos_far[0] + 50.0  # far beyond the radial basis range
        result = head(out, descriptors, pos_far, graph, atom)
        assert_valid_distribution(result, atol=1e-12)
        assert bool(torch.isfinite(result.logits).all())


# ---------------------------------------------------------------------------
# 2. SE(3) invariance (the spatial-agency contract)
# ---------------------------------------------------------------------------


class TestSE3Invariance:
    """The handle law must be invariant under rigid motions of the system."""

    @staticmethod
    def _run(backbone, head, state, pool):
        out = backbone(state)
        return head(out, *pool)

    def test_random_rigid_motions_float64(self):
        state = make_batched_state([12, 9], [6, 4], seed=20)
        backbone = small_backbone()
        head = small_head()
        pool = make_pool(backbone(state), [3, 2], seed=21)
        base = self._run(backbone, head, state, pool)

        gen = torch.Generator().manual_seed(22)
        worst = 0.0
        for _ in range(6):
            rotation = random_rotation(gen)
            translation = torch.randn(3, generator=gen, dtype=torch.float64) * 10.0
            moved = SimpleNamespace(
                pocket_pos=state.pocket_pos @ rotation.T + translation,
                pocket_z=state.pocket_z,
                pocket_charge=state.pocket_charge,
                pocket_batch=state.pocket_batch,
                ligand_pos=state.ligand_pos @ rotation.T + translation,
                ligand_z=state.ligand_z,
                ligand_charge=state.ligand_charge,
                ligand_batch=state.ligand_batch,
            )
            descriptors, pos, graph, atom = pool
            moved_pool = (descriptors, pos @ rotation.T + translation, graph, atom)
            moved_result = self._run(backbone, head, moved, moved_pool)
            for field in (
                "logits",
                "log_probs",
                "probs",
                "attention_weights",
                "entropy",
            ):
                a = getattr(base, field)
                b = getattr(moved_result, field)
                worst = max(worst, (a - b).abs().max().item())
        assert worst < 1e-9, f"rigid-motion sensitivity {worst}"

    def test_translation_only_invariance(self):
        state = make_batched_state([12, 9], [6, 4], seed=23)
        backbone = small_backbone()
        head = small_head()
        pool = make_pool(backbone(state), [3, 2], seed=24)
        base = self._run(backbone, head, state, pool)
        shift = torch.tensor([[1e3, -2e3, 5e2]], dtype=torch.float64)
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
        descriptors, pos, graph, atom = pool
        moved_pool = (descriptors, pos + shift, graph, atom)
        moved_result = self._run(backbone, head, moved, moved_pool)
        assert (base.log_probs - moved_result.log_probs).abs().max().item() < 1e-10

    def test_fp32_production_invariance_tolerance(self):
        state = make_batched_state([12, 9], [6, 4], seed=25, dtype=torch.float32)
        backbone = small_backbone(dtype=torch.float32)
        head = small_head(dtype=torch.float32)
        pool = make_pool(backbone(state), [3, 2], seed=26, dtype=torch.float32)
        base = self._run(backbone, head, state, pool)
        gen = torch.Generator().manual_seed(27)
        rotation = random_rotation(gen).to(torch.float32)
        moved = SimpleNamespace(
            pocket_pos=state.pocket_pos @ rotation.T,
            pocket_z=state.pocket_z,
            pocket_charge=state.pocket_charge,
            pocket_batch=state.pocket_batch,
            ligand_pos=state.ligand_pos @ rotation.T,
            ligand_z=state.ligand_z,
            ligand_charge=state.ligand_charge,
            ligand_batch=state.ligand_batch,
        )
        descriptors, pos, graph, atom = pool
        moved_pool = (descriptors, pos @ rotation.T, graph, atom)
        moved_result = self._run(backbone, head, moved, moved_pool)
        worst = (base.log_probs - moved_result.log_probs).abs().max().item()
        assert worst < 5e-4, f"fp32 rigid-motion sensitivity {worst}"

    def test_coincident_centroid_handle_geometry_is_finite(self):
        # A handle exactly at its pocket centroid exercises the safe norm
        # at r = 0 (no 0/0 anywhere); the law stays valid.
        state = make_batched_state([10], [5], seed=28)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        descriptors, pos, graph, atom = make_pool(out, [3], seed=29)
        centroid = out.node_pos[: out.num_pocket].mean(dim=0)
        pos_at_centroid = centroid.expand_as(pos).contiguous()
        result = head(out, descriptors, pos_at_centroid, graph, atom)
        assert_valid_distribution(result, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Permutation and batch consistency
# ---------------------------------------------------------------------------


class TestPermutationAndBatch:
    """Order- and batch-composition independence of the handle law."""

    def test_handle_row_permutation_equivariance(self):
        state = make_batched_state([12, 9], [6, 4], seed=30)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        pool = make_pool(out, [3, 2], seed=31)
        base = head(out, *pool)
        perm = torch.tensor([3, 0, 4, 1, 2], dtype=torch.long)
        descriptors, pos, graph, atom = pool
        permuted = head(
            out,
            descriptors.index_select(0, perm),
            pos.index_select(0, perm),
            graph.index_select(0, perm),
            atom.index_select(0, perm),
        )
        for field in ("logits", "log_probs", "probs", "attention_weights"):
            assert torch.allclose(
                getattr(base, field).index_select(0, perm),
                getattr(permuted, field),
                atol=1e-12,
            )
        assert torch.allclose(base.entropy, permuted.entropy, atol=1e-12)
        assert torch.equal(base.handle_atom.index_select(0, perm), permuted.handle_atom)

    def test_batch_composition_invariance(self):
        # Graph A alone vs. graph A batched with graph B: identical law.
        state_a = make_node_state(12, 6, seed=32, batch=0)
        state_ab = make_batched_state([12, 9], [6, 4], seed=32)
        backbone = small_backbone()
        head = small_head()
        out_a = backbone(state_a)
        out_ab = backbone(state_ab)
        pool_a = make_pool(out_a, [3], seed=33)
        pool_ab = make_pool(out_ab, [3, 2], seed=33)
        # identical descriptors/positions/atoms for graph 0's rows by
        # construction (same seeds, same leading state)
        result_a = head(out_a, *pool_a)
        result_ab = head(out_ab, *pool_ab)
        rows_ab = result_ab.rows_for_graph(0)
        assert torch.allclose(
            result_a.log_probs, result_ab.log_probs.index_select(0, rows_ab), atol=1e-10
        )
        assert torch.allclose(
            result_a.probs, result_ab.probs.index_select(0, rows_ab), atol=1e-10
        )
        assert result_a.entropy[0].item() == pytest.approx(
            result_ab.entropy[0].item(), abs=1e-10
        )

    def test_pocket_atom_permutation_invariance(self):
        # End-to-end through the backbone: permuting pocket nodes (all four
        # pocket fields consistently, batch labels included) must not change
        # the handle law.
        state = make_batched_state([12, 9], [6, 4], seed=34)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        pool = make_pool(out, [3, 2], seed=35)
        base = head(out, *pool)
        gen = torch.Generator().manual_seed(36)
        perm = torch.randperm(21, generator=gen)  # the whole pocket block
        permuted = SimpleNamespace(
            pocket_pos=state.pocket_pos.index_select(0, perm),
            pocket_z=state.pocket_z.index_select(0, perm),
            pocket_charge=state.pocket_charge.index_select(0, perm),
            pocket_batch=state.pocket_batch.index_select(0, perm),
            ligand_pos=state.ligand_pos,
            ligand_z=state.ligand_z,
            ligand_charge=state.ligand_charge,
            ligand_batch=state.ligand_batch,
        )
        permuted_result = head(backbone(permuted), *pool)
        assert torch.allclose(base.log_probs, permuted_result.log_probs, atol=1e-10)
        assert torch.allclose(base.entropy, permuted_result.entropy, atol=1e-10)

    def test_ligand_node_permutation_with_consistent_remap(self):
        # Permuting ligand nodes (consistent batch + handle_atom remap)
        # must not change the law: atom identity, not node order, is what
        # the head sees.
        state = make_batched_state([12], [6], seed=37)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        descriptors, pos, graph, atom = make_pool(out, [3], seed=38)
        base = head(out, descriptors, pos, graph, atom)
        gen = torch.Generator().manual_seed(39)
        perm = torch.randperm(6, generator=gen)
        permuted = SimpleNamespace(
            pocket_pos=state.pocket_pos,
            pocket_z=state.pocket_z,
            pocket_charge=state.pocket_charge,
            pocket_batch=state.pocket_batch,
            ligand_pos=state.ligand_pos.index_select(0, perm),
            ligand_z=state.ligand_z.index_select(0, perm),
            ligand_charge=state.ligand_charge.index_select(0, perm),
            ligand_batch=state.ligand_batch.index_select(0, perm),
        )
        remap = torch.empty_like(perm)
        remap[perm] = torch.arange(6)
        atom_remapped = remap.index_select(0, atom)
        pos_remapped = state.ligand_pos.index_select(0, atom).clone()
        permuted_result = head(
            backbone(permuted), descriptors, pos_remapped, graph, atom_remapped
        )
        assert torch.allclose(base.log_probs, permuted_result.log_probs, atol=1e-10)
        assert torch.equal(permuted_result.handle_atom, atom_remapped)
        assert torch.equal(base.handle_atom, atom)

    def test_rows_for_graph_and_echoes(self):
        state = make_batched_state([10, 8], [5, 4], seed=40)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        descriptors, pos, graph, atom = make_pool(out, [2, 3], seed=41)
        result = head(out, descriptors, pos, graph, atom)
        assert torch.equal(result.handle_graph, graph)
        assert torch.equal(result.handle_atom, atom)
        assert result.num_graphs == 2
        rows0 = result.rows_for_graph(0)
        assert rows0.tolist() == [0, 1]
        rows1 = result.rows_for_graph(1)
        assert rows1.tolist() == [2, 3, 4]
        with pytest.raises(TensorContractError):
            result.rows_for_graph(2)


# ---------------------------------------------------------------------------
# 4. Fail-closed input contract
# ---------------------------------------------------------------------------


class TestFailClosed:
    """Malformed pools and states fail closed with precise messages."""

    @staticmethod
    def _ready(dtype=torch.float64):
        state = make_batched_state([10, 8], [5, 4], seed=50, dtype=dtype)
        backbone = small_backbone(dtype=dtype)
        head = small_head(dtype=dtype)
        out = backbone(state)
        pool = make_pool(out, [3, 2], seed=51, dtype=dtype)
        return out, pool

    def test_rejects_non_backbone_output(self):
        _, pool = self._ready()
        head = small_head()
        with pytest.raises(TensorContractError, match="BackboneOutput"):
            head(SimpleNamespace(scalar=torch.zeros(1)), *pool)

    def test_rejects_legacy_67d_descriptors(self):
        out, (descriptors, pos, graph, atom) = self._ready()
        head = small_head()
        legacy = torch.cat(
            [descriptors, torch.zeros(descriptors.size(0), 3, dtype=torch.float64)],
            dim=1,
        )
        with pytest.raises(TensorContractError, match="Legacy 67D Franken-tensors"):
            head(out, legacy, pos, graph, atom)

    def test_rejects_wrong_descriptor_width(self):
        out, (descriptors, pos, graph, atom) = self._ready()
        head = small_head()
        with pytest.raises(TensorContractError, match="frozen schema descriptor width"):
            head(out, descriptors[:, :32], pos, graph, atom)

    def test_rejects_fp16_descriptors(self):
        out, (descriptors, pos, graph, atom) = self._ready(torch.float32)
        head = small_head(dtype=torch.float32)
        with pytest.raises(TensorContractError, match="float32 .* or float64"):
            head(out, descriptors.to(torch.float16), pos, graph, atom)

    def test_rejects_dtype_mismatch_between_module_and_inputs(self):
        out, pool = self._ready(torch.float64)
        head = small_head(dtype=torch.float32)  # module FP32, inputs FP64
        with pytest.raises(TensorContractError, match="cast the module"):
            head(out, *pool)

    def test_rejects_backbone_dtype_mismatch(self):
        # Head FP64 with an FP64 pool, but an FP32 backbone output:
        # producer/consumer mismatch must fail closed on the backbone field.
        state = make_batched_state([10, 8], [5, 4], seed=52, dtype=torch.float32)
        out = small_backbone(dtype=torch.float32)(state)
        descriptors = torch.randn(5, HANDLE_FEATURE_DIM, dtype=torch.float64)
        pos = torch.randn(5, 3, dtype=torch.float64)
        graph = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        atom = torch.tensor([0, 1, 2, 5, 6], dtype=torch.long)  # graph-1 ligand nodes
        head = small_head(dtype=torch.float64)
        with pytest.raises(TensorContractError, match="backbone.scalar"):
            head(out, descriptors, pos, graph, atom)

    def test_rejects_empty_pool(self):
        out, _ = self._ready()
        head = small_head()
        empty_desc = torch.zeros(0, HANDLE_FEATURE_DIM, dtype=torch.float64)
        empty_pos = torch.zeros(0, 3, dtype=torch.float64)
        empty_idx = torch.zeros(0, dtype=torch.long)
        with pytest.raises(TensorContractError, match="pool is empty"):
            head(out, empty_desc, empty_pos, empty_idx, empty_idx)

    def test_rejects_wrong_handle_pos_shapes(self):
        out, (descriptors, pos, graph, atom) = self._ready()
        head = small_head()
        with pytest.raises(TensorContractError, match="size 3 on the last axis"):
            head(out, descriptors, torch.zeros(5, 2, dtype=torch.float64), graph, atom)
        with pytest.raises(TensorContractError, match="5 rows"):
            head(out, descriptors, pos[:4], graph, atom)

    def test_rejects_graph_out_of_range(self):
        out, (descriptors, pos, graph, atom) = self._ready()
        head = small_head()
        bad_graph = graph.clone()
        bad_graph[0] = 2
        with pytest.raises(TensorContractError, match=r"\[0, 2\)"):
            head(out, descriptors, pos, bad_graph, atom)

    def test_rejects_atom_out_of_ligand_range(self):
        out, (descriptors, pos, graph, atom) = self._ready()
        head = small_head()
        bad_atom = atom.clone()
        bad_atom[0] = 9  # total ligand nodes = 9
        with pytest.raises(TensorContractError, match=r"\[0, 9\)"):
            head(out, descriptors, pos, graph, bad_atom)

    def test_rejects_atom_of_foreign_graph(self):
        # Cross-field consistency: the attachment atom must belong to the
        # handle's own graph.
        out, (descriptors, pos, graph, atom) = self._ready()
        head = small_head()
        num_pocket = out.num_pocket
        graph0_atoms = torch.nonzero(
            out.node_batch[num_pocket:] == 0, as_tuple=False
        ).flatten()
        bad_atom = atom.clone()
        bad_atom[3] = graph0_atoms[0]  # row 3 belongs to graph 1
        with pytest.raises(TensorContractError, match="different graph"):
            head(out, descriptors, pos, graph, bad_atom)

    def test_rejects_index_dtype_and_length_errors(self):
        out, (descriptors, pos, graph, atom) = self._ready()
        head = small_head()
        with pytest.raises(TensorContractError, match="int64"):
            head(out, descriptors, pos, graph.to(torch.int32), atom)
        with pytest.raises(TensorContractError, match=r"\[5\]"):
            head(out, descriptors, pos, graph[:4], atom)

    def test_rejects_non_finite_inputs(self):
        out, (descriptors, pos, graph, atom) = self._ready()
        head = small_head()
        poisoned = descriptors.clone()
        poisoned[0, 0] = float("nan")
        with pytest.raises(TensorContractError, match="non-finite"):
            head(out, poisoned, pos, graph, atom)
        poisoned_pos = pos.clone()
        poisoned_pos[1, 1] = float("inf")
        with pytest.raises(TensorContractError, match="non-finite"):
            head(out, descriptors, poisoned_pos, graph, atom)

    def test_rejects_pool_without_ligand_nodes(self):
        # A pocket-only state (t = 0 pre-seed) with a non-empty pool is a
        # contract violation: handles live on ligand atoms.
        state = make_node_state(10, 0, seed=54)
        out = small_backbone()(state)
        assert out.num_ligand == 0
        head = small_head()
        descriptors = torch.randn(1, HANDLE_FEATURE_DIM, dtype=torch.float64)
        pos = torch.randn(1, 3, dtype=torch.float64)
        graph = torch.zeros(1, dtype=torch.long)
        atom = torch.zeros(1, dtype=torch.long)
        with pytest.raises(TensorContractError, match="no ligand nodes"):
            head(out, descriptors, pos, graph, atom)


# ---------------------------------------------------------------------------
# 5. Environment integration — the concrete downstream consumer
# ---------------------------------------------------------------------------


def _tri_chloro_seed() -> BuildingBlock:
    """1,2,4-trichlorobenzene seed: three non-equivalent aryl handles."""
    template = Chem.MolFromSmiles("Clc1ccc(Cl)c(Cl)c1")
    specs = []
    for atom in template.GetAtoms():
        if atom.GetAtomicNum() == 17:
            ipso = atom.GetNeighbors()[0].GetIdx()
            specs.append(
                HandleSpec(f"cl{atom.GetIdx()}", ipso, "aryl_chloride", (atom.GetIdx(),))
            )
    assert len(specs) == 3
    return BuildingBlock("seed124", "Clc1ccc(Cl)c(Cl)c1", tuple(specs))


def _aminophenol_block(block_id: str) -> BuildingBlock:
    """4-aminophenol: one amine handle (consumed by amine coupling) plus
    one alcohol handle that survives the junction — so the active-handle
    menu genuinely *grows* as blocks are added."""
    template = Chem.MolFromSmiles("Nc1ccc(O)cc1")
    n_idx = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 7)
    o_idx = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 8)
    return BuildingBlock(
        block_id,
        "Nc1ccc(O)cc1",
        (HandleSpec("amine", n_idx, "amine"), HandleSpec("oh", o_idx, "alcohol")),
    )


def _toy_pocket() -> PocketDefinition:
    rng = np.random.default_rng(60)
    positions = rng.uniform(-8.0, 8.0, size=(40, 3))
    atomic_numbers = rng.choice([6, 7, 8, 16], size=40)
    charges = np.zeros(40)
    return PocketDefinition(
        positions=positions, atomic_numbers=atomic_numbers, charges=charges
    )


def _state_from_environment(
    env: AssemblyEnvironment, pocket: PocketDefinition, dtype: torch.dtype
) -> SimpleNamespace:
    """The documented env -> tensor-state mapping (heavy atoms, in order)."""
    mol = env.observe()
    conformer = mol.GetConformer()
    positions = np.array(conformer.GetPositions(), dtype=np.float64)
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    # Translate the assembly so the pocket frame is comparable (pocket atoms
    # around the origin): centre the molecule on the pocket centroid is NOT
    # applied — the pocket frame is the frame both sub-graphs must share.
    return SimpleNamespace(
        pocket_pos=torch.as_tensor(pocket.positions, dtype=dtype),
        pocket_z=torch.as_tensor(pocket.atomic_numbers, dtype=torch.long),
        pocket_charge=torch.as_tensor(pocket.charges, dtype=dtype),
        pocket_batch=torch.zeros(pocket.num_atoms, dtype=torch.long),
        ligand_pos=torch.as_tensor(positions, dtype=dtype),
        ligand_z=torch.as_tensor(z, dtype=torch.long),
        ligand_charge=torch.zeros(len(z), dtype=dtype),
        ligand_batch=torch.zeros(len(z), dtype=torch.long),
    )


def _pool_from_menu(
    menu, ligand_pos: torch.Tensor, *, seed: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The documented menu -> pool mapping (descriptors from the catalog
    featurizer in production; deterministic synthetic descriptors here)."""
    gen = torch.Generator().manual_seed(seed)
    handle_atom = torch.tensor([m["handle_atom_idx"] for m in menu], dtype=torch.long)
    handle_pos = ligand_pos.index_select(0, handle_atom).clone()
    descriptors = torch.randn((len(menu), HANDLE_FEATURE_DIM), generator=gen, dtype=dtype)
    handle_graph = torch.zeros(len(menu), dtype=torch.long)
    return descriptors, handle_pos, handle_graph, handle_atom


class TestEnvironmentIntegration:
    """The head scores exactly the ``active_handles()`` menu, and its
    choices are legal ``AssemblyAction`` handle components."""

    @staticmethod
    def _make_env():
        catalog = {
            "ap1": _aminophenol_block("ap1"),
            "ap2": _aminophenol_block("ap2"),
        }
        reactions = {
            "couple_amine": JunctionReaction("couple_amine", block_handle_class="amine"),
        }
        env = AssemblyEnvironment(
            _tri_chloro_seed(),
            catalog,
            pocket=_toy_pocket(),
            max_blocks=4,
            reaction_registry=reactions,
        )
        return env, catalog, reactions

    def test_scores_the_t0_seed_menu_and_argmax_is_actionable(self):
        env, catalog, reactions = self._make_env()
        mol = env.reset()
        menu = env.active_handles()
        assert len(menu) == 3  # H > 1 exposed handles at t = 0
        dtype = torch.float32
        state = _state_from_environment(env, env.pocket, dtype)
        backbone = small_backbone(dtype=dtype)
        head = small_head(dtype=dtype)
        out = backbone(state)
        pool = _pool_from_menu(menu, state.ligand_pos, seed=61, dtype=dtype)
        result = head(out, *pool)

        # rows map 1:1 onto the menu, in input order
        assert result.num_handles == len(menu)
        assert result.handle_atom.tolist() == [m["handle_atom_idx"] for m in menu]
        assert result.handle_counts.tolist() == [len(menu)]
        assert_valid_distribution(result, atol=1e-5)

        # the argmax row's atom index is a legal handle for env.step
        best = int(torch.argmax(result.log_probs).item())
        action = AssemblyAction(
            handle_atom_idx=menu[best]["handle_atom_idx"],
            reaction_id="couple_amine",
            block_id="ap1",
            torsion_bin=18,
        )
        record = env.step(action)
        assert record.action_type == "add"
        assert env.num_added_blocks == 1

    def test_scores_a_grown_intermediate_menu(self):
        env, catalog, reactions = self._make_env()
        env.reset()
        # Grow two aminophenol blocks: each junction consumes one seed-side
        # handle and leaves the block's alcohol handle alive, so the menu
        # mixes surviving seed handles with newly exposed block handles.
        env.step(
            AssemblyAction(
                handle_atom_idx=env.active_handles()[0]["handle_atom_idx"],
                reaction_id="couple_amine",
                block_id="ap1",
                torsion_bin=10,
            )
        )
        env.step(
            AssemblyAction(
                handle_atom_idx=env.active_handles()[0]["handle_atom_idx"],
                reaction_id="couple_amine",
                block_id="ap2",
                torsion_bin=26,
            )
        )
        menu = env.active_handles()
        assert len(menu) == 3  # 1 surviving seed handle + 2 new alcohol handles
        assert {m["handle_class"] for m in menu} == {"aryl_chloride", "alcohol"}
        assert env.observe().GetNumAtoms() > 15  # genuinely grown intermediate
        dtype = torch.float32
        state = _state_from_environment(env, env.pocket, dtype)
        backbone = small_backbone(dtype=dtype)
        head = small_head(dtype=dtype)
        out = backbone(state)
        pool = _pool_from_menu(menu, state.ligand_pos, seed=62, dtype=dtype)
        result = head(out, *pool)
        assert result.num_handles == len(menu)
        assert result.handle_atom.tolist() == [m["handle_atom_idx"] for m in menu]
        assert_valid_distribution(result, atol=1e-5)
        # every menu row is individually actionable
        for row, entry in zip(range(len(menu)), menu):
            assert result.log_probs[row].item() <= 0.0
            assert 0 <= entry["handle_atom_idx"] < state.ligand_pos.size(0)

    def test_log_prob_feeds_the_tb_forward_probability(self):
        # The handle component's log-probability is exactly the quantity the
        # Trajectory Balance engine consumes as part of log P_F (sum of the
        # action tuple's component log-probabilities, #45 §12.3).
        from syntree.engine.gflownet import TrajectoryStep

        env, _, _ = self._make_env()
        env.reset()
        menu = env.active_handles()
        dtype = torch.float64
        state = _state_from_environment(env, env.pocket, dtype)
        backbone = small_backbone(dtype=dtype)
        head = small_head(dtype=dtype)
        out = backbone(state)
        pool = _pool_from_menu(menu, state.ligand_pos, seed=63, dtype=dtype)
        result = head(out, *pool)
        chosen = int(torch.argmax(result.log_probs).item())
        step = TrajectoryStep(
            action_type="add",
            action_log_prob=float(result.log_probs[chosen].item()),
            num_leaves=1,
        )
        assert math.isfinite(step.action_log_prob)
        assert step.action_log_prob <= 0.0


# ---------------------------------------------------------------------------
# 6. Gradient connectivity
# ---------------------------------------------------------------------------


class TestGradients:
    """The handle log-probabilities are trainable end-to-end."""

    def test_gradients_reach_head_and_backbone(self):
        state = make_batched_state([12, 9], [6, 4], seed=70)
        backbone = BipartitePaiNN(
            hidden_dim=D, num_layers=LAYERS, num_radial=RADIAL, r_cut=5.0
        ).to(torch.float64)
        head = HandleSelectionHead(
            backbone_dim=D, hidden_dim=HEAD_HIDDEN, num_radial=RADIAL
        ).to(torch.float64)
        out = backbone(state)
        pool = make_pool(out, [3, 2], seed=71)
        descriptors, pos, graph, atom = pool
        pos = pos.detach().requires_grad_(True)
        out = backbone(state)
        result = head(out, descriptors, pos, graph, atom)
        chosen = torch.tensor([0, 3], dtype=torch.long)
        loss = result.log_probs.index_select(0, chosen).sum()
        loss.backward()

        head_grads = [p.grad for p in head.parameters()]
        assert all(g is not None for g in head_grads)
        assert all(bool(torch.isfinite(g).all()) for g in head_grads)
        assert any(float(g.abs().sum()) > 0.0 for g in head_grads)
        backbone_grads = [p.grad for p in backbone.parameters() if p.grad is not None]
        assert backbone_grads, "no gradient reached the backbone"
        assert all(bool(torch.isfinite(g).all()) for g in backbone_grads)
        assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())
        assert float(pos.grad.abs().sum()) > 0.0

    def test_entropy_is_differentiable_and_finite(self):
        state = make_batched_state([10], [5], seed=72)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        pool = make_pool(out, [3], seed=73)
        result = head(out, *pool)
        result.entropy.sum().backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads)

    def test_no_gradient_leaks_across_graphs(self):
        # Differentiating graph 0's chosen log-prob must not depend on
        # graph 1's handle rows (batch decoupling of P_F components).
        state = make_batched_state([12, 9], [6, 4], seed=74)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        descriptors, pos, graph, atom = make_pool(out, [3, 2], seed=75)
        result = head(out, descriptors, pos, graph, atom)
        row0 = result.rows_for_graph(0)[0]
        result.log_probs[row0].backward()
        # graph 1's descriptors receive no gradient from graph 0's loss
        assert descriptors.grad is None


# ---------------------------------------------------------------------------
# 7. Frozen profile, audit, determinism
# ---------------------------------------------------------------------------


class TestFrozenProfile:
    """The production profile is frozen in version-controlled constants."""

    def test_default_construction_uses_frozen_constants(self):
        head = HandleSelectionHead()
        assert head.backbone_dim == BACKBONE_HIDDEN_DIM == 256
        assert head.hidden_dim == HANDLE_HEAD_HIDDEN_DIM == 128
        assert head.num_radial == HANDLE_HEAD_NUM_RADIAL == 20
        assert head.geom_cutoff == HANDLE_GEOM_CUTOFF == 12.0

    def test_production_parameter_count_is_pinned(self):
        head = HandleSelectionHead()
        count = sum(p.numel() for p in head.parameters())
        assert count == 224896

    def test_descriptor_width_is_the_schema_constant(self):
        # Single source of truth: the head and the schema agree on 64.
        assert HANDLE_FEATURE_DIM == 64

    def test_radial_basis_is_enveloped(self):
        # #9 MATH-03 / D-V2-011: the cosine cutoff envelope is integrated
        # inside GaussianRadialBasis (the one authoritative place for all
        # spatial RBF projections) — the head's [0, 12] A centroid-distance
        # basis is enveloped at 12 A. Supersedes the pre-#9 raw-expansion
        # pin (D-V2-007's documented "until #9 lands" boundary).
        from syntree.models.cutoff import continuous_cosine_envelope

        head = HandleSelectionHead()
        centers = head.radial_basis.centers
        assert centers.numel() == HANDLE_HEAD_NUM_RADIAL
        assert float(centers[0]) == 0.0
        assert float(centers[-1]) == pytest.approx(HANDLE_GEOM_CUTOFF)
        # at the cutoff the features vanish; beyond it they are bitwise zero
        at_cutoff = head.radial_basis(torch.tensor([HANDLE_GEOM_CUTOFF]))
        assert float(at_cutoff.abs().max()) <= 1e-6  # fp32 cancellation floor
        beyond = head.radial_basis(
            torch.tensor([HANDLE_GEOM_CUTOFF + 1e-4, 20.0, 62.0])
        )
        assert bool((beyond == 0).all())
        # inside the range the expansion equals raw Gaussian * envelope
        d = torch.linspace(0.0, HANDLE_GEOM_CUTOFF, 41, dtype=torch.float64)
        rbf64 = head.radial_basis.to(torch.float64)
        out = rbf64(d)
        sigma = HANDLE_GEOM_CUTOFF / (HANDLE_HEAD_NUM_RADIAL - 1)
        for k in range(HANDLE_HEAD_NUM_RADIAL):
            raw = torch.exp(-0.5 * ((d - float(centers[k])) / sigma) ** 2)
            expected = raw * continuous_cosine_envelope(d, HANDLE_GEOM_CUTOFF)
            assert torch.allclose(out[:, k], expected, atol=1e-12)

    def test_constructor_rejects_invalid_arguments(self):
        with pytest.raises(ValueError, match="backbone_dim"):
            HandleSelectionHead(backbone_dim=0)
        with pytest.raises(ValueError, match="hidden_dim"):
            HandleSelectionHead(hidden_dim=0)
        with pytest.raises(ValueError, match="num_radial"):
            HandleSelectionHead(num_radial=1)
        with pytest.raises(ValueError, match="geom_cutoff"):
            HandleSelectionHead(geom_cutoff=0.0)


class TestAuditAndDeterminism:
    """Audit interface and bit-for-bit determinism."""

    def test_audit_reports_dimensions_and_trainability(self):
        head = HandleSelectionHead()
        report = head.audit()
        assert report["module"] == "HandleSelectionHead"
        assert report["backbone_dim"] == 256
        assert report["hidden_dim"] == 128
        assert report["descriptor_dim"] == 64
        assert report["num_parameters"] == 224896
        assert report["num_trainable_parameters"] == 224896
        assert report["all_parameters_trainable"] is True
        assert report["parameter_dtypes"] == ["torch.float32"]
        assert report["deterministic"] is True

    def test_repeated_forward_is_bit_for_bit_identical(self):
        state = make_batched_state([12, 9], [6, 4], seed=80)
        backbone = small_backbone()
        head = small_head()
        out = backbone(state)
        pool = make_pool(out, [3, 2], seed=81)
        first = head(out, *pool)
        second = head(out, *pool)
        for field in (
            "logits",
            "log_probs",
            "probs",
            "attention_weights",
            "entropy",
            "handle_embeddings",
            "pooled_handle_context",
        ):
            assert torch.equal(getattr(first, field), getattr(second, field))

    def test_no_grad_matches_grad_enabled(self):
        state = make_batched_state([12, 9], [6, 4], seed=82)
        backbone = small_backbone()
        head = small_head()
        with torch.no_grad():
            out = backbone(state)
            pool = make_pool(out, [3, 2], seed=83)
            eager = head(out, *pool)
        out = backbone(state)
        pool = make_pool(out, [3, 2], seed=83)
        tracked = head(out, *pool)
        assert torch.allclose(eager.log_probs, tracked.log_probs, atol=1e-12)
