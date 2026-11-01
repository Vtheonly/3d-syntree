"""Two-layer stacked cross-attention decoder over the joint pocket+ligand memory.

Clean-room V2 implementation of the cross-attention decoder block (issue #23
MODEL-03; constitution #45 §7.3; decision D-V2-012 in
``docs/agent/DECISIONS.md``).

**Why a stacked decoder (the #23 motivation — pre-reset decision D08,
re-adopted as specification by D-V2-002).** A single-pass cross-attention at
128 dimensions suffers *subspace crowding*: one attention layer at narrow
width cannot separate sterics, electrostatics, and hydrogen-bonding
directions of a protein pocket. V2 upgrades the decision context to
``d_model = 256`` and stacks **two** cross-attention decoder layers, each
refining a *handle query* against the dense padded joint-graph memory (the
union of pocket and ligand atoms encoded by the single authoritative
:class:`~syntree.models.bipartite.BipartitePaiNN` backbone, #21):

1. **Layer 1** attends to cavity bulk and steric boundaries — the coarse
   placement context of the growth point.
2. **Layer 2** refines the query against localized hydrogen-bond and
   salt-bridge partners — the fine chemical-partner context.

Both layers share the identical pinned structure (below); the
bulk-then-local specialisation is *representational* — the second layer
cross-attends with a layer-1-refined query — not two different mechanisms.
This implements exactly the mechanism the issue pins.

**Pinned layer structure (issue #23, verbatim).** Each
:class:`CrossAttentionDecoderLayer` is:

* cross-attention: ``nn.MultiheadAttention(embed_dim=256, num_heads=8,
  dropout=0.1)`` with ``batch_first=True`` — the handle query attends to
  the dense padded joint-graph memory;
* feed-forward refinement: ``Linear(256 -> 512) -> SiLU -> Dropout ->
  Linear(512 -> 256)``;
* **pre-LayerNorm residual connections on both sub-layers** (normalize the
  sub-layer input, add the sub-layer output to the untouched residual
  stream). The stack closes with one final ``LayerNorm`` — standard pre-LN
  practice (GPT-2/ViT-style) so the residual stream handed to downstream
  heads has a stable scale; the per-layer structure is the issue's, the
  closing norm is the stack's.

**Two forward interfaces.**

* :meth:`CrossAttentionDecoder.forward` — the *dense padded* interface
  pinned by the #23 acceptance criteria: ``query [B, 1, 256]`` and
  ``memory [B, N, 256]`` with a ``key_padding_mask [B, N]`` (``True`` =
  padded, non-atom slot), outputting the refined context ``[B, 256]``.
  One query per graph (``Lq = 1``) is the decoder's contracted use; the
  *layer* primitive accepts general ``Lq`` and is pinned so by test.
* :meth:`CrossAttentionDecoder.forward_state` — the V2-native path: it
  consumes the :class:`~syntree.models.bipartite.BackboneOutput` of the
  single authoritative encoder (never a second pocket encoder, #45 §7.2)
  and the handle-side surface of
  :class:`~syntree.models.handle_head.HandleSelectionOutput` (never a
  second handle encoder, D-V2-010), builds the per-graph handle query,
  pads the union memory, and runs the stack.

**Handle query construction (one authoritative embedding).** For graph
``g``::

    q_g = W_handle( pooled_handle_context_g )   [d_handle -> d_model]
        + W_graph ( graph_context_g )           [d       -> d_model]
        + W_pocket( pocket_context_g )          [d       -> d_model]

``pooled_handle_context`` is the #22 attention-weighted aggregate of the
per-handle embeddings (which already carry the enveloped centroid-distance
RBF geometry and the equivariant directional invariants — see
``handle_head.py``); the per-row ``handle_embeddings`` remain available to
the #24 hierarchical heads for per-candidate scoring and are deliberately
not re-encoded here. The backbone contexts are the #21 pooled summaries.
The query is therefore built from invariant quantities only.

**Dense padded memory and masking semantics.** The flat union
representation (``[N, d]`` with graph assignment, the PyG-style layout the
backbone produces) is padded per graph into ``[B, N_max, d]`` with an
order-preserving gather; padded slots are zero-filled and marked
``True`` in ``key_padding_mask``. Consequences (pinned by tests):

* attention weights over padded columns are **exactly 0** in every layer,
  head, and graph (softmax of ``-inf``), in eval *and* train mode;
* the decoder output is *bit-identical* under arbitrary changes to the
  padded slot values (they are excluded from every reduction);
* a graph with **all** slots padded is a contract violation and fails
  closed (softmax over an empty key set is undefined) — every valid V2
  state has at least one pocket atom per graph (#21), so this cannot occur
  for genuine backbone outputs.

**SE(3) invariance argument.** Every consumed tensor is already invariant:
backbone scalar features and both pooled contexts are invariant (#21,
pinned ≤3e-13 over 36 rigid-motion trials), and the #22 handle surface is
invariant (pinned 8.3e-16 FP64 / 5.96e-08 FP32). The decoder applies only
linear projections, LayerNorms, softmax inner-product attention, and the
FFN to those tensors — all invariant operations — so the output context is
rigid-motion invariant end-to-end (pinned by test through the real
backbone + handle head + decoder stack).

**Envelope rule (decision D-V2-011).** The decoder projects **no new
spatial distances**: it consumes no Cartesian coordinates at all. All
geometry it sees already passed through the enveloped
:class:`~syntree.models.bipartite.GaussianRadialBasis` upstream — the
backbone's edge-distance expansion (inside the #21 message passing) and
the handle head's [0, 12] Å centroid-distance expansion plus directional
invariants (inside the #22 embeddings). D-V2-011's rule ("any spatial
distance RBF the decoder projects MUST be produced by
``GaussianRadialBasis``") is therefore satisfied vacuously *and*
structurally: there is no raw distance expansion anywhere in this module.
If a future issue (e.g. #24/#39) adds a query-to-memory distance bias, it
MUST route through ``GaussianRadialBasis`` — never re-implement an
expansion or multiply a raw one by hand.

**Attention-map memory arithmetic (issue #23's < 20 MB budget).** With the
frozen profile (``B`` graphs, ``N`` padded memory slots, 8 heads, 2
layers, ``Lq = 1``) the exposed attention maps occupy
``2 · B · 8 · 1 · N`` floats: at the documented T4 batch target
``B = 12`` and a conservative ``N = 800`` (the #45 §5.2 pocket cap of 700
plus ligand margin) that is 614,400 floats ≈ **0.59 MB in FP32** — two
orders of magnitude below the 20 MB budget, before counting the
``need_weights=False`` production path which materializes none of them.
This is *arithmetic*, not a measurement: no T4 execution has occurred
(CPU dev sandbox; measured T4 profiling is #43 OPS-03).

**Precision policy (decisions D-V2-005 / D-V2-008).** Parameters default
to pure FP32 (the production path on Colab/Kaggle Tesla T4).
``float64`` is accepted *only* for local numerical verification of the
invariance properties at tight tolerance; the input dtype (including the
consumed ``BackboneOutput`` / ``HandleSelectionOutput`` tensors) must
match the module parameter dtype exactly. FP16/BF16 and every other
dtype are rejected fail-closed.

**Determinism.** The pinned ``dropout = 0.1`` (cross-attention weights and
FFN hidden layer) is active only in training mode; ``eval()`` deactivates
it, making the forward deterministic and reproducible (pinned by test).
No other stochastic layers exist.

**Scope boundary.** This module produces the per-graph *decision context*
``[B, d_model]`` (plus auditable attention maps). The hierarchical action
heads (STOP, reaction family/reaction, synthon, stereochemical parity,
torsion bins — #24), the full policy assembly and sampling utilities
(#25), and the seed/anchor selection that may consume the same
pocket-conditioned context (#39) live in their own issues; the #39
addendum to issue #23 explicitly allows consuming the decoder's
pocket-conditioned context, which is exactly ``context`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from syntree.core.schema import TensorContractError
from syntree.models.bipartite import (
    BACKBONE_HIDDEN_DIM,
    BackboneOutput,
)
from syntree.models.handle_head import (
    HANDLE_HEAD_HIDDEN_DIM,
    HandleSelectionOutput,
)

__all__ = [
    "DECODER_D_MODEL",
    "DECODER_NUM_LAYERS",
    "DECODER_NUM_HEADS",
    "DECODER_FFN_DIM",
    "DECODER_DROPOUT",
    "CrossAttentionDecoderLayer",
    "CrossAttentionDecoder",
    "DecoderOutput",
]

# ---------------------------------------------------------------------------
# Frozen production profile (issue #23; #45 §5.2: architecture dimensions are
# frozen in version-controlled configuration, never probed from free VRAM).
# ---------------------------------------------------------------------------

#: Model width of the decoder (query / memory / residual stream). The issue
#: pins the upgrade from the legacy 128-d single-pass head to 256.
DECODER_D_MODEL = 256

#: Number of stacked cross-attention decoder layers (issue #23: "Stack 2
#: sequential decoder layers").
DECODER_NUM_LAYERS = 2

#: Attention heads of each layer's ``nn.MultiheadAttention`` (issue #23).
DECODER_NUM_HEADS = 8

#: Hidden width of the feed-forward refinement block (issue #23:
#: ``Linear(256 -> 512) -> SiLU -> Dropout -> Linear(512 -> 256)``).
DECODER_FFN_DIM = 512

#: Dropout probability inside the cross-attention and the FFN hidden layer
#: (issue #23); active in training mode only.
DECODER_DROPOUT = 0.1

#: Input dtypes accepted by the decoder (D-V2-005 / D-V2-008): FP32
#: production plus FP64 local verification only.
_FLOAT_INPUT_DTYPES = (torch.float32, torch.float64)


# ---------------------------------------------------------------------------
# One decoder layer (the issue-pinned structure)
# ---------------------------------------------------------------------------


class CrossAttentionDecoderLayer(nn.Module):
    """One pre-LayerNorm cross-attention decoder layer (issue #23, verbatim).

    Structure (pinned by tests):

    * ``cross_attn`` — ``nn.MultiheadAttention(d_model, num_heads,
      dropout=dropout, batch_first=True)``: the handle query attends to the
      dense padded joint-graph memory;
    * ``ffn`` — ``Linear(d_model -> ffn_dim) -> SiLU -> Dropout ->
      Linear(ffn_dim -> d_model)``;
    * **pre-LayerNorm residuals on both sub-layers**: the sub-layer input
      is normalized by ``attn_norm`` / ``ffn_norm`` *before* entering the
      sub-layer, and the sub-layer output is added to the untouched
      residual stream. Zeroing both sub-layer outputs makes the layer the
      exact identity (pinned by test — this also distinguishes pre-LN from
      post-LN, where the same zeroing would leave ``LayerNorm(x)``).

    Forward:
        ``query``: ``[B, Lq, d_model]`` (the decoder stack pins ``Lq = 1``;
        the layer primitive accepts any ``Lq >= 1``).
        ``memory``: ``[B, N, d_model]`` dense padded joint-graph memory.
        ``key_padding_mask``: ``[B, N]`` bool, ``True`` = padded slot
        (excluded from attention exactly).
        Returns ``(refined [B, Lq, d_model], attn [B, num_heads, Lq, N])``.

    The layer performs structural validation (tensor-ness, ranks, ``d_model``
    consistency, mask dtype/shape, no fully-masked row); the dtype/device
    policy against the module parameters is enforced by the owning stack.
    """

    def __init__(
        self,
        d_model: int = DECODER_D_MODEL,
        num_heads: int = DECODER_NUM_HEADS,
        ffn_dim: int = DECODER_FFN_DIM,
        dropout: float = DECODER_DROPOUT,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        num_heads = int(num_heads)
        ffn_dim = int(ffn_dim)
        dropout = float(dropout)
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if ffn_dim < 1:
            raise ValueError(f"ffn_dim must be >= 1, got {ffn_dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_p = dropout

        # Pre-LayerNorm on both sub-layers (issue #23).
        self.attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"ffn_dim={self.ffn_dim}, dropout={self.dropout_p}"
        )

    def _validate_shapes(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[int, int, int]:
        if not isinstance(query, torch.Tensor) or not isinstance(
            memory, torch.Tensor
        ):
            raise TensorContractError(
                "[decoder-layer] `query` and `memory` must be torch tensors, "
                f"got {type(query).__name__} and {type(memory).__name__}"
            )
        if not isinstance(key_padding_mask, torch.Tensor):
            raise TensorContractError(
                "[decoder-layer] `key_padding_mask` must be a torch.Tensor, "
                f"got {type(key_padding_mask).__name__}"
            )
        if query.dim() != 3 or memory.dim() != 3:
            raise TensorContractError(
                "[decoder-layer] `query` must be [B, Lq, d_model] and "
                f"`memory` [B, N, d_model], got {tuple(query.shape)} and "
                f"{tuple(memory.shape)}"
            )
        if key_padding_mask.dim() != 2:
            raise TensorContractError(
                "[decoder-layer] `key_padding_mask` must be [B, N], got "
                f"{tuple(key_padding_mask.shape)}"
            )
        batch, length_q, width_q = query.shape
        batch_m, length_n, width_m = memory.shape
        if batch != batch_m:
            raise TensorContractError(
                f"[decoder-layer] `query` batch {batch} does not match "
                f"`memory` batch {batch_m}"
            )
        if key_padding_mask.size(0) != batch or key_padding_mask.size(1) != length_n:
            raise TensorContractError(
                f"[decoder-layer] `key_padding_mask` must be [{batch}, "
                f"{length_n}] to match `memory`, got "
                f"{tuple(key_padding_mask.shape)}"
            )
        if width_q != self.d_model or width_m != self.d_model:
            raise TensorContractError(
                f"[decoder-layer] `query`/`memory` last dimension must be "
                f"d_model = {self.d_model}, got {width_q} and {width_m}"
            )
        if length_q < 1:
            raise TensorContractError(
                "[decoder-layer] `query` must contain at least one token"
            )
        if length_n < 1:
            raise TensorContractError(
                "[decoder-layer] `memory` must contain at least one slot"
            )
        if key_padding_mask.dtype != torch.bool:
            raise TensorContractError(
                "[decoder-layer] `key_padding_mask` must have dtype bool "
                "(True = padded slot), got "
                f"{key_padding_mask.dtype}"
            )
        if key_padding_mask.device != memory.device:
            raise TensorContractError(
                "[decoder-layer] `key_padding_mask` is on device "
                f"{key_padding_mask.device} but `memory` is on "
                f"{memory.device}"
            )
        if query.device != memory.device:
            raise TensorContractError(
                "[decoder-layer] `query` is on device {query.device} but "
                f"`memory` is on {memory.device}"
            )
        if query.dtype != memory.dtype:
            raise TensorContractError(
                "[decoder-layer] `query` dtype {query.dtype} does not match "
                f"`memory` dtype {memory.dtype}"
            )
        if bool(key_padding_mask.all(dim=1).any()):
            fully_masked = torch.nonzero(
                key_padding_mask.all(dim=1), as_tuple=False
            ).flatten().tolist()
            raise TensorContractError(
                f"[decoder-layer] every graph must own at least one real "
                f"(unpadded) memory slot; graph(s) {fully_masked} are fully "
                "padded — softmax over an empty key set is undefined"
            )
        return int(batch), int(length_q), int(length_n)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One pre-LN cross-attention + FFN refinement step; see above."""
        self._validate_shapes(query, memory, key_padding_mask)

        # Sub-layer 1 (pre-LN residual): cross-attention over the padded
        # joint-graph memory. Padded slots receive -inf attention logits,
        # so their weights are exactly zero and their values cannot leak.
        normed_query = self.attn_norm(query)
        attn_out, attn_weights = self.cross_attn(
            normed_query,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        query = query + attn_out

        # Sub-layer 2 (pre-LN residual): feed-forward refinement.
        query = query + self.ffn(self.ffn_norm(query))
        return query, attn_weights


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderOutput:
    """The auditable result of one decoder forward pass.

    Fields:
        context: refined per-graph decision context ``[B, d_model]`` — the
            #24/#25-facing surface (and the pocket-conditioned context the
            #39 addendum to issue #23 allows consuming).
        sequence: the full refined query sequence ``[B, 1, d_model]``
            (``context.unsqueeze(1)``; the stack pins ``Lq = 1``).
        attention_weights: per-layer attention maps
            ``[num_layers, B, num_heads, 1, N]`` — padded columns are
            exactly zero in every layer/head/graph (telemetry + audit).
        query: the input query echo ``[B, 1, d_model]`` (audit; the built
            handle query on the ``forward_state`` path).
        memory: the padded joint-graph memory echo ``[B, N, d_model]``.
        key_padding_mask: ``[B, N]`` bool echo (``True`` = padded slot).
        num_graphs: number of graphs ``B``.
        num_memory_tokens: real (unpadded) memory tokens per graph
            ``[B]`` int64.
        memory_length: padded memory length ``N``.
    """

    context: torch.Tensor  # [B, d_model]
    sequence: torch.Tensor  # [B, 1, d_model]
    attention_weights: torch.Tensor  # [num_layers, B, num_heads, 1, N]
    query: torch.Tensor  # [B, 1, d_model]
    memory: torch.Tensor  # [B, N, d_model]
    key_padding_mask: torch.Tensor  # [B, N] bool
    num_graphs: int
    num_memory_tokens: torch.Tensor  # [B] int64
    memory_length: int

    @property
    def num_layers(self) -> int:
        """Number of stacked decoder layers that produced this output."""
        return int(self.attention_weights.size(0))


# ---------------------------------------------------------------------------
# Per-graph padding of the flat union memory
# ---------------------------------------------------------------------------


def _pad_union_memory(
    memory: torch.Tensor,
    memory_batch: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad the flat union memory into the dense per-graph layout.

    Converts ``memory [N, D]`` with graph assignment ``memory_batch [N]``
    into ``(padded [B, N_max, D], key_padding_mask [B, N_max],
    counts [B])`` where ``key_padding_mask`` is ``True`` exactly at the
    padded slots. The gather is *order-preserving*: each graph's nodes
    appear in their original relative order, so a permutation applied
    consistently to the flat layout carries through unchanged.

    Every graph must own at least one node (fail-closed); ``memory_batch``
    must be int64 with values in ``[0, num_graphs)``.
    """
    if memory.dim() != 2 or memory.size(0) < 1:
        raise TensorContractError(
            f"[decoder] `memory` must be [N, D] with N >= 1, got "
            f"{tuple(memory.shape)}"
        )
    n = memory.size(0)
    if memory_batch.dim() != 1 or memory_batch.size(0) != n:
        raise TensorContractError(
            f"[decoder] `memory_batch` must be [{n}], got "
            f"{tuple(memory_batch.shape)}"
        )
    if memory_batch.dtype not in (torch.int64, torch.long):
        raise TensorContractError(
            f"[decoder] `memory_batch` must have dtype int64, got "
            f"{memory_batch.dtype}"
        )
    if memory_batch.numel():
        lo, hi = int(memory_batch.min()), int(memory_batch.max())
        if lo < 0 or hi >= num_graphs:
            raise TensorContractError(
                f"[decoder] `memory_batch` values must lie in "
                f"[0, {num_graphs}), got range [{lo}, {hi}]"
            )
    counts = torch.bincount(memory_batch, minlength=num_graphs)  # [B]
    if not bool((counts >= 1).all()):
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise TensorContractError(
            f"[decoder] every graph must own at least one memory token; "
            f"graph(s) {empty} have none (every valid V2 state has at "
            "least one pocket atom per graph)"
        )
    n_max = int(counts.max().item())

    # Order-preserving per-graph gather: sort by graph (stable), then rank
    # each node within its graph segment.
    order = torch.argsort(memory_batch, stable=True)  # [N]
    sorted_batch = memory_batch.index_select(0, order)  # [N]
    starts = torch.cumsum(counts, dim=0) - counts  # [B] segment starts
    rank = torch.arange(n, device=memory.device) - starts.index_select(
        0, sorted_batch
    )  # [N] position within each graph's segment

    padded = memory.new_zeros((num_graphs, n_max, memory.size(1)))
    padded[sorted_batch, rank] = memory.index_select(0, order)
    key_padding_mask = torch.ones(
        (num_graphs, n_max), dtype=torch.bool, device=memory.device
    )
    key_padding_mask[sorted_batch, rank] = False
    return padded, key_padding_mask, counts


# ---------------------------------------------------------------------------
# The stacked decoder
# ---------------------------------------------------------------------------


class CrossAttentionDecoder(nn.Module):
    """Two-layer stacked cross-attention decoder (issue #23 MODEL-03).

    The decision-context factory of the V2 policy stack: it refines a
    per-graph *handle query* against the dense padded joint-graph memory
    (pocket ∪ ligand nodes as encoded by the single authoritative
    ``BipartitePaiNN`` backbone) through ``num_layers`` pre-LN
    :class:`CrossAttentionDecoderLayer` blocks and a final ``LayerNorm``.

    Args:
        backbone_dim: width ``d`` of the consumed backbone representations
            (default: the frozen production
            :data:`~syntree.models.bipartite.BACKBONE_HIDDEN_DIM` = 256).
        handle_dim: width of the consumed #22 handle surface (default
            :data:`~syntree.models.handle_head.HANDLE_HEAD_HIDDEN_DIM` =
            128).
        d_model: decoder width (default :data:`DECODER_D_MODEL` = 256).
        num_layers: stacked layers (default :data:`DECODER_NUM_LAYERS` = 2).
        num_heads: attention heads per layer (default
            :data:`DECODER_NUM_HEADS` = 8).
        ffn_dim: FFN hidden width (default :data:`DECODER_FFN_DIM` = 512).
        dropout: dropout inside cross-attention and FFN (default
            :data:`DECODER_DROPOUT` = 0.1; training mode only).

    Forward (dense interface, the #23 acceptance shapes):
        ``query``: ``[B, 1, d_model]`` — one handle query per graph.
        ``memory``: ``[B, N, d_model]`` — dense padded joint-graph memory.
        ``key_padding_mask``: ``[B, N]`` bool, ``True`` = padded slot.
        Returns a :class:`DecoderOutput` whose ``context`` is ``[B,
        d_model]``.

    Forward (V2-native interface):
        :meth:`forward_state` consumes a ``BackboneOutput`` +
        ``HandleSelectionOutput`` pair and runs the same stack on the
        padded union memory.
    """

    def __init__(
        self,
        backbone_dim: int = BACKBONE_HIDDEN_DIM,
        handle_dim: int = HANDLE_HEAD_HIDDEN_DIM,
        d_model: int = DECODER_D_MODEL,
        num_layers: int = DECODER_NUM_LAYERS,
        num_heads: int = DECODER_NUM_HEADS,
        ffn_dim: int = DECODER_FFN_DIM,
        dropout: float = DECODER_DROPOUT,
    ) -> None:
        super().__init__()
        backbone_dim = int(backbone_dim)
        handle_dim = int(handle_dim)
        d_model = int(d_model)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        ffn_dim = int(ffn_dim)
        dropout = float(dropout)
        if backbone_dim < 1:
            raise ValueError(f"backbone_dim must be >= 1, got {backbone_dim}")
        if handle_dim < 1:
            raise ValueError(f"handle_dim must be >= 1, got {handle_dim}")
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads "
                f"({num_heads})"
            )

        self.backbone_dim = backbone_dim
        self.handle_dim = handle_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_p = dropout

        # Handle-query construction from the #22 handle surface (pooled
        # handle context — the attention-weighted aggregate of the
        # authoritative handle embeddings) and the #21 pooled contexts.
        self.query_handle = nn.Linear(handle_dim, d_model)
        self.query_graph = nn.Linear(backbone_dim, d_model)
        self.query_pocket = nn.Linear(backbone_dim, d_model)

        # The pinned stack + the conventional pre-LN closing norm.
        self.layers = nn.ModuleList(
            CrossAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(d_model)

    def extra_repr(self) -> str:
        return (
            f"backbone_dim={self.backbone_dim}, handle_dim={self.handle_dim}, "
            f"d_model={self.d_model}, num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, ffn_dim={self.ffn_dim}, "
            f"dropout={self.dropout_p}"
        )

    # -- audit ---------------------------------------------------------------

    def audit(self) -> Dict[str, Any]:
        """Parameter/dimension audit (constitution #45 §7.3).

        Returns a JSON-compatible dict of module dimensions, parameter
        counts, trainability, the dtype policy, and the envelope-rule
        declaration (this module projects no spatial distances at all —
        decision D-V2-012).
        """
        parameters = list(self.parameters())
        return {
            "module": type(self).__name__,
            "backbone_dim": self.backbone_dim,
            "handle_dim": self.handle_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout_p,
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "num_buffers": len(list(self.buffers())),
            # The decoder consumes no Cartesian coordinates: every spatial
            # quantity it sees already passed through the enveloped
            # GaussianRadialBasis upstream (#21/#22; D-V2-011/D-V2-012).
            "projects_spatial_distances": False,
            "deterministic_in_eval": True,  # dropout only in training mode
        }

    # -- input validation (fail-closed) ---------------------------------------

    def _reference_param(self) -> torch.Tensor:
        return self.query_handle.weight

    def _check_float(
        self,
        name: str,
        tensor: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
        rank: int,
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TensorContractError(
                f"[decoder] `{name}` must be a torch.Tensor, got "
                f"{type(tensor).__name__}"
            )
        if tensor.dtype not in _FLOAT_INPUT_DTYPES:
            raise TensorContractError(
                f"[decoder] `{name}` must have dtype float32 (the production "
                f"path, decision D-V2-005) or float64 (local numerical "
                f"verification only, decision D-V2-008), got {tensor.dtype}"
            )
        if tensor.dtype != dtype:
            raise TensorContractError(
                f"[decoder] `{name}` has dtype {tensor.dtype} but the module "
                f"parameters are {dtype}; cast the module with "
                f".to({tensor.dtype}) for local numerical verification, or "
                "fix the producer"
            )
        if tensor.device != device:
            raise TensorContractError(
                f"[decoder] `{name}` is on device {tensor.device} but the "
                f"module parameters are on {device}"
            )
        if tensor.dim() != rank:
            raise TensorContractError(
                f"[decoder] `{name}` must be {rank}-D, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.numel() and not bool(torch.isfinite(tensor).all()):
            raise TensorContractError(
                f"[decoder] `{name}` contains non-finite values (NaN/Inf); "
                "poisoned inputs fail closed — locate the producer"
            )
        return tensor

    def _validate_dense_inputs(
        self,
        query: Any,
        memory: Any,
        key_padding_mask: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fail-closed validation of the dense padded interface."""
        ref = self._reference_param()
        dtype, device = ref.dtype, ref.device

        query = self._check_float("query", query, dtype=dtype, device=device, rank=3)
        if query.size(-1) != self.d_model:
            raise TensorContractError(
                f"[decoder] `query` must have size {self.d_model} on the "
                f"last axis (d_model), got {tuple(query.shape)}"
            )
        if query.size(1) != 1:
            raise TensorContractError(
                f"[decoder] `query` must have exactly one token per graph "
                f"(the handle query; Lq = 1), got Lq = {query.size(1)} — "
                "the layer primitive supports general Lq, the stack does not"
            )
        batch = int(query.size(0))
        if batch < 1:
            raise TensorContractError(
                "[decoder] `query` must contain at least one graph"
            )

        memory = self._check_float("memory", memory, dtype=dtype, device=device, rank=3)
        if memory.size(0) != batch:
            raise TensorContractError(
                f"[decoder] `memory` must have batch {batch} to match "
                f"`query`, got {tuple(memory.shape)}"
            )
        if memory.size(-1) != self.d_model:
            raise TensorContractError(
                f"[decoder] `memory` must have size {self.d_model} on the "
                f"last axis (d_model), got {tuple(memory.shape)}"
            )
        if memory.size(1) < 1:
            raise TensorContractError(
                "[decoder] `memory` must contain at least one slot per graph"
            )

        if not isinstance(key_padding_mask, torch.Tensor):
            raise TensorContractError(
                "[decoder] `key_padding_mask` must be a torch.Tensor, got "
                f"{type(key_padding_mask).__name__}"
            )
        if key_padding_mask.dtype != torch.bool:
            raise TensorContractError(
                "[decoder] `key_padding_mask` must have dtype bool "
                "(True = padded slot), got "
                f"{key_padding_mask.dtype}"
            )
        if key_padding_mask.device != device:
            raise TensorContractError(
                f"[decoder] `key_padding_mask` is on device "
                f"{key_padding_mask.device} but the module parameters are "
                f"on {device}"
            )
        if key_padding_mask.dim() != 2 or key_padding_mask.shape != (
            memory.size(0),
            memory.size(1),
        ):
            raise TensorContractError(
                f"[decoder] `key_padding_mask` must be [{memory.size(0)}, "
                f"{memory.size(1)}] to match `memory`, got "
                f"{tuple(key_padding_mask.shape)}"
            )
        if bool(key_padding_mask.all(dim=1).any()):
            fully_masked = torch.nonzero(
                key_padding_mask.all(dim=1), as_tuple=False
            ).flatten().tolist()
            raise TensorContractError(
                f"[decoder] every graph must own at least one real "
                f"(unpadded) memory slot; graph(s) {fully_masked} are fully "
                "padded — softmax over an empty key set is undefined"
            )
        return query, memory, key_padding_mask

    def _validate_state_inputs(
        self,
        backbone: Any,
        handle: Any,
    ) -> tuple[BackboneOutput, HandleSelectionOutput]:
        """Fail-closed validation of the V2-native interface."""
        if not isinstance(backbone, BackboneOutput):
            raise TensorContractError(
                "[decoder] `backbone` must be a BackboneOutput (the "
                f"BipartitePaiNN encoder output), got {type(backbone).__name__}"
            )
        if not isinstance(handle, HandleSelectionOutput):
            raise TensorContractError(
                "[decoder] `handle` must be a HandleSelectionOutput (the "
                "HandleSelectionHead output), got "
                f"{type(handle).__name__}"
            )
        if handle.num_graphs != backbone.num_graphs:
            raise TensorContractError(
                f"[decoder] cross-field mismatch: `handle` covers "
                f"{handle.num_graphs} graphs but `backbone` produced "
                f"{backbone.num_graphs} — both must come from the same "
                "state batch"
            )
        ref = self._reference_param()
        dtype, device = ref.dtype, ref.device
        num_graphs = backbone.num_graphs

        # Backbone-side surface: union scalars (the memory), the pooled
        # union context, the pooled pocket context, and the graph index.
        # (Validation only — BackboneOutput is frozen; nothing is mutated.)
        self._check_float(
            "backbone.scalar", backbone.scalar, dtype=dtype, device=device, rank=2
        )
        if backbone.scalar.size(-1) != self.backbone_dim:
            raise TensorContractError(
                f"[decoder] `backbone.scalar` must have width "
                f"{self.backbone_dim} (the module's backbone_dim), got "
                f"{tuple(backbone.scalar.shape)}"
            )
        if backbone.scalar.size(0) < 1:
            raise TensorContractError(
                "[decoder] `backbone.scalar` must contain at least one node"
            )
        if backbone.node_batch.dtype not in (torch.int64, torch.long):
            raise TensorContractError(
                f"[decoder] `backbone.node_batch` must have dtype int64, "
                f"got {backbone.node_batch.dtype}"
            )
        if backbone.node_batch.device != device:
            raise TensorContractError(
                f"[decoder] `backbone.node_batch` is on device "
                f"{backbone.node_batch.device} but the module parameters "
                f"are on {device}"
            )
        if backbone.node_batch.dim() != 1 or backbone.node_batch.size(
            0
        ) != backbone.scalar.size(0):
            raise TensorContractError(
                f"[decoder] `backbone.node_batch` must be "
                f"[{backbone.scalar.size(0)}], got "
                f"{tuple(backbone.node_batch.shape)}"
            )
        if backbone.node_batch.numel():
            lo, hi = int(backbone.node_batch.min()), int(backbone.node_batch.max())
            if lo < 0 or hi >= num_graphs:
                raise TensorContractError(
                    f"[decoder] `backbone.node_batch` values must lie in "
                    f"[0, {num_graphs}), got range [{lo}, {hi}]"
                )
        for name, tensor in (
            ("backbone.graph_context", backbone.graph_context),
            ("backbone.pocket_context", backbone.pocket_context),
        ):
            tensor = self._check_float(name, tensor, dtype=dtype, device=device, rank=2)
            if tensor.size(0) != num_graphs:
                raise TensorContractError(
                    f"[decoder] `{name}` must have {num_graphs} rows (one "
                    f"per graph), got {tuple(tensor.shape)}"
                )
            if tensor.size(-1) != self.backbone_dim:
                raise TensorContractError(
                    f"[decoder] `{name}` must have width "
                    f"{self.backbone_dim} (the module's backbone_dim), got "
                    f"{tuple(tensor.shape)}"
                )

        # Handle-side surface: the pooled handle context (one authoritative
        # embedding — the per-row `handle_embeddings` stay with #24).
        pooled = self._check_float(
            "handle.pooled_handle_context",
            handle.pooled_handle_context,
            dtype=dtype,
            device=device,
            rank=2,
        )
        if pooled.size(0) != num_graphs:
            raise TensorContractError(
                f"[decoder] `handle.pooled_handle_context` must have "
                f"{num_graphs} rows (one per graph), got {tuple(pooled.shape)}"
            )
        if pooled.size(-1) != self.handle_dim:
            raise TensorContractError(
                f"[decoder] `handle.pooled_handle_context` must have width "
                f"{self.handle_dim} (the module's handle_dim), got "
                f"{tuple(pooled.shape)}"
            )
        return backbone, handle

    # -- forward (dense padded interface) --------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> DecoderOutput:
        """Run the stack on the dense padded layout (the #23 shapes).

        ``query [B, 1, d_model]`` + ``memory [B, N, d_model]`` +
        ``key_padding_mask [B, N]`` (True = pad) → :class:`DecoderOutput`
        with ``context [B, d_model]``.
        """
        query, memory, key_padding_mask = self._validate_dense_inputs(
            query, memory, key_padding_mask
        )
        x = query
        attentions = []
        for layer in self.layers:
            x, weights = layer(x, memory, key_padding_mask)
            attentions.append(weights)
        x = self.out_norm(x)
        context = x.squeeze(1)  # Lq == 1 (validated)
        attention = torch.stack(attentions, dim=0)  # [L, B, H, 1, N]
        num_memory_tokens = (~key_padding_mask).long().sum(dim=1)  # [B]
        output = DecoderOutput(
            context=context,
            sequence=x,
            attention_weights=attention,
            query=query,
            memory=memory,
            key_padding_mask=key_padding_mask,
            num_graphs=int(query.size(0)),
            num_memory_tokens=num_memory_tokens,
            memory_length=int(memory.size(1)),
        )
        self._assert_finite(output)
        return output

    # -- forward (V2-native interface) ------------------------------------------

    def forward_state(
        self,
        backbone: BackboneOutput,
        handle: HandleSelectionOutput,
    ) -> DecoderOutput:
        """Run the stack on a V2 state: build the handle query, pad the
        union memory, and refine.

        Consumes the single authoritative encoder's
        :class:`~syntree.models.bipartite.BackboneOutput` (union scalars,
        pooled graph / pocket contexts) and the #22 handle-side surface
        (:attr:`HandleSelectionOutput.pooled_handle_context` — the
        attention-weighted aggregate of the authoritative handle
        embeddings). No second pocket encoder, no second handle encoder.
        """
        backbone, handle = self._validate_state_inputs(backbone, handle)
        num_graphs = backbone.num_graphs

        # Handle query: one per graph, from invariant summaries only.
        query = (
            self.query_handle(handle.pooled_handle_context)
            + self.query_graph(backbone.graph_context)
            + self.query_pocket(backbone.pocket_context)
        )  # [B, d_model]
        query = query.unsqueeze(1)  # [B, 1, d_model]

        # Dense padded joint-graph memory (order-preserving per-graph
        # gather of the union scalars).
        memory, key_padding_mask, _counts = _pad_union_memory(
            backbone.scalar, backbone.node_batch, num_graphs
        )

        return self.forward(query, memory, key_padding_mask)

    # -- fail loudly ----------------------------------------------------------

    def _assert_finite(self, output: DecoderOutput) -> None:
        """Fail loudly on non-finite outputs (#45 §16: never hide NaNs)."""
        for name in ("context", "sequence", "attention_weights"):
            tensor = getattr(output, name)
            if tensor.numel() and not bool(torch.isfinite(tensor).all()):
                raise TensorContractError(
                    f"[decoder] `{name}` contains non-finite values — with "
                    "finite validated inputs this indicates corrupted "
                    "parameters; fail loudly per #45 §16"
                )
