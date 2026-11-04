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

**Scope.** This module owns the per-graph *decision context* factory
(the two-layer cross-attention decoder, #23) **and** the decoupled
hierarchical policy heads over that context (#24 MODEL-04): the binary
termination head, the 9-way family head, the 27-way family-conditional
certified-reaction head, the K-way tied-embedding synthon retrieval
head, the 2-way chiral parity head, and their container
(:class:`HierarchicalPolicyHeads`; the 36-bin categorical torsion head
lives in :mod:`syntree.models.torsion_head` per the issue's module
pinning). The full policy assembly with the backbone, handle head and
the #39 seed/anchor component is #25 MODEL-05's scope; the #39 addendum
to issue #23 explicitly allows consuming the decoder's pocket-conditioned
context, which is exactly ``context`` here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from syntree.core.schema import TensorContractError
from syntree.chemistry.reactions import POLICY_REACTIONS, REACTION_FAMILIES
from syntree.models.bipartite import (
    BACKBONE_HIDDEN_DIM,
    BackboneOutput,
    BipartitePaiNN,
)
from syntree.models.handle_head import (
    HANDLE_HEAD_HIDDEN_DIM,
    HandleSelectionHead,
    HandleSelectionOutput,
)
from syntree.models.torsion_head import (
    TORSION_BINS,
    TORSION_HEAD_CONTEXT_DIM,
    TORSION_HEAD_SYNTON_DIM,
    TorsionHead,
    TorsionOutput,
)
from syntree.utils.hardware import (
    LOGIT_MASK_FILL,
    LOGIT_MASK_LEGAL_ABOVE,
    HardwareError,
    apply_logit_mask,
    validate_target_actions,
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
    # -- issue #24 MODEL-04: the decoupled hierarchical policy heads -------
    "POLICY_NUM_FAMILIES",
    "POLICY_NUM_REACTIONS",
    "SYNTON_CATALOG_SIZE",
    "SYNTON_EMBED_DIM",
    "PARITY_ACTIONS",
    "STOP_INDEX",
    "CONTINUE_INDEX",
    "REACTION_FAMILY_NAMES",
    "POLICY_REACTION_IDS",
    "FAMILY_MEMBERS",
    "BLOCK_HANDLE_CLASS_VOCAB",
    "MOLECULE_HANDLE_CLASS_VOCAB",
    "HIERARCHICAL_HEADS_PARAMETER_PROFILE",
    "handle_classes_from_menu",
    "build_continuation_masks",
    "build_force_stop",
    "reaction_support_mask",
    "synthon_legality_mask",
    "StopOutput",
    "FamilyOutput",
    "ReactionOutput",
    "SynthonOutput",
    "ParityOutput",
    "ContinueOutput",
    "TerminationHead",
    "ReactionFamilyHead",
    "ReactionHead",
    "SynthonRetrievalHead",
    "ChiralParityHead",
    "HierarchicalPolicyHeads",
    # -- issue #25 MODEL-05: the assembly + the #39 seed/anchor component --
    "MAX_POCKET_ANCHORS",
    "ANCHOR_MIN_SEPARATION",
    "SYNTREE_POLICY_PARAMETER_PROFILE",
    "pocket_anchor_candidates",
    "batched_pocket_anchor_candidates",
    "aggregate_anchor_features",
    "SeedAnchorOutput",
    "SeedAnchorHead",
    "PolicyStateOutput",
    "SynTreePolicy",
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


# ===========================================================================
# Issue #24 MODEL-04 — Decoupled hierarchical policy heads & discrete SO(2)
# torsion space.
#
# The three pre-V2 structural bugs this section eliminates (issue #24
# "Context & Motivation"):
#
#   1. STOP was indexed as token K inside the synthon head, forcing a
#      reaction-family choice before the termination decision (a
#      probability leak).  V2: termination is a *binary head evaluated
#      first* (:class:`TerminationHead`); the continuation heads are a
#      separate stage that is never executed for STOP rows — the two
#      forward passes are structurally distinct calls.
#   2. Torsion densities (rad^-1) were summed with dimensionless discrete
#      probabilities in the TB loss.  V2: the torsion action is a strictly
#      categorical 36-bin law; the von Mises parameters exist only inside
#      the supervised circular loss (`torsion_circular_loss`), never in
#      the TB log-probability.
#   3. Junction stereocenters lacked enantiomeric parity selection.  V2:
#      :class:`ChiralParityHead` emits the discrete ``c_t in {R, S}``
#      action, scored exactly where the junction forms a stereocenter.
#
# Consumption contracts (HANDOFF S-V2-08, binding):
#   * the decoder's ``DecoderOutput.context`` ``[B, 256]`` is the head
#     input surface (D-V2-012);
#   * the certified registry (#11, D-V2-016) supplies the 27 policy
#     reactions, their 9 family memberships, and the per-reaction
#     molecule/block handle classes — the mask factories below are the
#     policy-side consumers of that metadata;
#   * the logit-mask machinery (#10, D-V2-014: ``apply_logit_mask`` /
#     ``validate_target_actions``, ``LOGIT_MASK_FILL``) is the one
#     categorical masking path;
#   * the environment's ``active_handles()`` menu is the handle-class
#     source of continuation legality.
# ===========================================================================


# ---------------------------------------------------------------------------
# Frozen action-space cardinalities (issue #24; #45 §5.2)
# ---------------------------------------------------------------------------

#: Number of coarse reaction families = the certified registry's family
#: map size (D-V2-016: 9 families).
POLICY_NUM_FAMILIES = len(REACTION_FAMILIES)

#: Number of certified policy reactions (D-V2-016: 27 coupling reactions;
#: the 5 deprotections are deterministic environment operations, never
#: policy actions).
POLICY_NUM_REACTIONS = len(POLICY_REACTIONS)

#: Target catalog size (issue #24: "catalog synthon (K = 50,000 classes)";
#: issue #25 pins the same figure).  The head is constructible at any
#: size; the real curated count lands with #12 CHEM-02.
SYNTON_CATALOG_SIZE = 50_000

#: Width of the tied catalog embeddings — the single authoritative synthon
#: representation consumed by retrieval scoring and by the parity/torsion
#: conditioning (mirrors the frozen 64-D handle-descriptor width of the
#: #6 schema).
SYNTON_EMBED_DIM = 64

#: The discrete enantiomeric parity actions (column order of the parity
#: head's 2-way logits; index 0 = ``'R'`` (the environment's default
#: placement orientation), index 1 = ``'S'`` (the mirrored placement —
#: ``AssemblyAction.chiral`` semantics).
PARITY_ACTIONS: Tuple[str, ...] = ("R", "S")

#: Column indices of the binary termination head: CONTINUE first, STOP
#: second (the absorbing action is deliberately *not* a token inside any
#: larger head — bug 1's probability leak is impossible by construction).
CONTINUE_INDEX = 0
STOP_INDEX = 1

#: Canonical family ordering of the 9 family logits (the certified
#: registry's family-map insertion order — deterministic).
REACTION_FAMILY_NAMES: Tuple[str, ...] = tuple(REACTION_FAMILIES.keys())

#: Canonical reaction ordering of the 27 reaction logits (the certified
#: registry's reaction-map insertion order, ``rxn_01..rxn_27``).
POLICY_REACTION_IDS: Tuple[str, ...] = tuple(POLICY_REACTIONS.keys())

#: Family membership: family name -> the member reaction ids.
FAMILY_MEMBERS: Dict[str, Tuple[str, ...]] = {
    name: tuple(REACTION_FAMILIES[name]) for name in REACTION_FAMILY_NAMES
}

#: Molecule-side (grown-ligand) handle classes recognised by the registry.
MOLECULE_HANDLE_CLASS_VOCAB: Tuple[str, ...] = tuple(
    sorted({r.molecule_handle_class for r in POLICY_REACTIONS.values()})
)

#: Block-side (catalog synthon) handle classes recognised by the registry.
BLOCK_HANDLE_CLASS_VOCAB: Tuple[str, ...] = tuple(
    sorted({r.block_handle_class for r in POLICY_REACTIONS.values()})
)

_REACTION_INDEX: Dict[str, int] = {
    rid: i for i, rid in enumerate(POLICY_REACTION_IDS)
}
_FAMILY_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(REACTION_FAMILY_NAMES)
}
_BLOCK_CLASS_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(BLOCK_HANDLE_CLASS_VOCAB)
}
#: reaction index -> family index (the 27-way lineage of the hierarchy).
_REACTION_FAMILY_OF: Tuple[int, ...] = tuple(
    _FAMILY_INDEX[POLICY_REACTIONS[rid].family] for rid in POLICY_REACTION_IDS
)
#: reaction index -> molecule-side handle class.
_REACTION_MOL_CLASS: Tuple[str, ...] = tuple(
    POLICY_REACTIONS[rid].molecule_handle_class for rid in POLICY_REACTION_IDS
)
#: reaction index -> block-side handle class.
_REACTION_BLOCK_CLASS: Tuple[str, ...] = tuple(
    POLICY_REACTIONS[rid].block_handle_class for rid in POLICY_REACTION_IDS
)
#: reaction index -> resonance lock is not None (torsion action absent).
_REACTION_LOCKED: Tuple[bool, ...] = tuple(
    POLICY_REACTIONS[rid].resonance_lock is not None
    for rid in POLICY_REACTION_IDS
)


# ---------------------------------------------------------------------------
# Mask factories — the policy-side consumers of the environment menu and
# the certified registry metadata (#45 §7.3 "mask invalid combinations;
# normalize over the actually valid alternatives").
# ---------------------------------------------------------------------------

def handle_classes_from_menu(menu: Sequence[Mapping[str, object]]) -> Tuple[str, ...]:
    """The distinct active handle classes of one environment menu.

    Consumes the exact ``AssemblyEnvironment.active_handles()`` menu
    format (``handle_class`` key, capacity already filtered by the
    environment).
    """
    return tuple(
        sorted({str(row["handle_class"]) for row in menu})
    )


def build_continuation_masks(
    active_handle_classes: Sequence[Iterable[str]],
    available_block_classes: Optional[Iterable[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legality masks for the family and reaction heads.

    A reaction is legal for a graph iff

    * its molecule-side handle class is present among the graph's active
      handles (the environment menu), and
    * its block-side handle class is available in the catalog.

    A family is legal iff at least one member reaction is legal.

    Parameters:
        active_handle_classes: per-graph iterables of active handle class
            names (e.g. ``handle_classes_from_menu(env.active_handles())``
            per graph).
        available_block_classes: the block-side classes with at least one
            catalog synthon.  Default: *all* registry block classes — the
            documented pre-#12 assumption (no curated catalog exists yet;
            #12 CHEM-02 / #41 DATA-06 will supply the real availability
            from the catalog manifest).

    Returns:
        ``(family_mask, reaction_mask)`` — boolean ``[B, 9]`` and
        ``[B, 27]`` tensors, ``True = illegal`` (the ``apply_logit_mask``
        convention).
    """
    if available_block_classes is None:
        available = set(BLOCK_HANDLE_CLASS_VOCAB)
    else:
        available = set(available_block_classes)
        unknown = available - set(BLOCK_HANDLE_CLASS_VOCAB)
        if unknown:
            raise TensorContractError(
                f"available_block_classes contains classes unknown to the "
                f"certified registry: {sorted(unknown)}"
            )

    num_graphs = len(active_handle_classes)
    if num_graphs == 0:
        raise TensorContractError(
            "build_continuation_masks received an empty graph list (fail-closed)"
        )
    family_mask = torch.zeros(
        (num_graphs, POLICY_NUM_FAMILIES), dtype=torch.bool
    )
    reaction_mask = torch.zeros(
        (num_graphs, POLICY_NUM_REACTIONS), dtype=torch.bool
    )
    for b, classes in enumerate(active_handle_classes):
        active = set(classes)
        for r_idx, rid in enumerate(POLICY_REACTION_IDS):
            legal = (
                _REACTION_MOL_CLASS[r_idx] in active
                and _REACTION_BLOCK_CLASS[r_idx] in available
            )
            if not legal:
                reaction_mask[b, r_idx] = True
        for f_idx, fname in enumerate(REACTION_FAMILY_NAMES):
            members = FAMILY_MEMBERS[fname]
            any_legal = any(
                not bool(reaction_mask[b, _REACTION_INDEX[m]])
                for m in members
            )
            if not any_legal:
                family_mask[b, f_idx] = True
    return family_mask, reaction_mask


def build_force_stop(
    reaction_mask: torch.Tensor,
    at_capacity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-graph forced-termination flags for the STOP head.

    A graph must terminate when it has **no legal continuation**
    (``reaction_mask`` fully masked — no active handle class matches any
    available reaction) or when the environment's block budget is
    exhausted (``num_added_blocks >= max_blocks``: the environment accepts
    only STOP there).  Such states are resolved *before* any categorical
    continuation mask is applied (#45 §8.4 / the ``apply_logit_mask``
    fail-closed contract).
    """
    if reaction_mask.dtype != torch.bool or reaction_mask.ndim != 2:
        raise TensorContractError(
            "reaction_mask must be a boolean [B, 27] tensor"
        )
    if reaction_mask.shape[1] != POLICY_NUM_REACTIONS:
        raise TensorContractError(
            f"reaction_mask must have {POLICY_NUM_REACTIONS} columns, got "
            f"{reaction_mask.shape[1]}"
        )
    forced = reaction_mask.all(dim=-1)
    if at_capacity is not None:
        if at_capacity.dtype != torch.bool or at_capacity.shape != forced.shape:
            raise TensorContractError(
                "at_capacity must be a boolean [B] tensor matching reaction_mask rows"
            )
        forced = forced | at_capacity
    return forced


def reaction_support_mask(
    selected_family: torch.Tensor, reaction_mask: torch.Tensor
) -> torch.Tensor:
    """The reaction head's support: legal reactions of the selected family.

    ``support[b, r] = reaction_mask[b, r] is legal AND family(r) ==
    selected_family[b]`` — the exact conditional support of
    ``P(reaction | family, s)`` (the hierarchical factorization's
    restriction step).
    """
    if not isinstance(selected_family, torch.Tensor) or selected_family.dtype not in (
        torch.int64,
        torch.int32,
    ):
        raise TensorContractError(
            f"selected_family must be an integer tensor, got "
            f"{selected_family.dtype if isinstance(selected_family, torch.Tensor) else type(selected_family).__name__}"
        )
    if selected_family.ndim != 1:
        raise TensorContractError(
            f"selected_family must be [B], got {tuple(selected_family.shape)}"
        )
    if reaction_mask.dtype != torch.bool or reaction_mask.ndim != 2:
        raise TensorContractError(
            "reaction_mask must be a boolean [B, 27] tensor"
        )
    if reaction_mask.shape[0] != selected_family.shape[0]:
        raise TensorContractError(
            f"reaction_mask has {reaction_mask.shape[0]} rows but "
            f"selected_family has {selected_family.shape[0]}"
        )
    if reaction_mask.shape[1] != POLICY_NUM_REACTIONS:
        raise TensorContractError(
            f"reaction_mask must have {POLICY_NUM_REACTIONS} columns"
        )
    if bool((selected_family < 0).any()) or bool(
        (selected_family >= POLICY_NUM_FAMILIES).any()
    ):
        raise TensorContractError(
            "selected_family indices out of range [0, 9) (fail-closed)"
        )
    family_of = torch.tensor(_REACTION_FAMILY_OF, dtype=torch.long)
    member = family_of.unsqueeze(0) == selected_family.long().unsqueeze(1)
    return reaction_mask | ~member


def synthon_legality_mask(
    catalog_handle_classes: torch.Tensor, selected_reaction: torch.Tensor
) -> torch.Tensor:
    """The synthon head's support: synthons of the selected reactions'
    block-side handle class.

    ``mask[b, k] = catalog_handle_classes[k] !=
    block_class(selected_reaction[b])`` (the ``apply_logit_mask``
    convention, ``True = illegal``).  Class compatibility is a *prefilter*
    only — the environment still enforces the exact one-handle chemistry
    at execution, and #40's curation guarantees curated blocks expose
    exactly one coupling handle.
    """
    if not isinstance(catalog_handle_classes, torch.Tensor):
        raise TensorContractError(
            "catalog_handle_classes must be a torch.Tensor"
        )
    if catalog_handle_classes.dtype not in (torch.int64, torch.int32):
        raise TensorContractError(
            f"catalog_handle_classes must be an integer tensor, got "
            f"{catalog_handle_classes.dtype}"
        )
    if catalog_handle_classes.ndim != 1:
        raise TensorContractError(
            f"catalog_handle_classes must be [K], got "
            f"{tuple(catalog_handle_classes.shape)}"
        )
    if not isinstance(selected_reaction, torch.Tensor) or selected_reaction.dtype not in (
        torch.int64,
        torch.int32,
    ):
        raise TensorContractError(
            "selected_reaction must be an integer tensor of reaction indices"
        )
    if selected_reaction.ndim != 1:
        raise TensorContractError(
            f"selected_reaction must be [B], got {tuple(selected_reaction.shape)}"
        )
    if bool((selected_reaction < 0).any()) or bool(
        (selected_reaction >= POLICY_NUM_REACTIONS).any()
    ):
        raise TensorContractError(
            "selected_reaction indices out of range [0, 27) (fail-closed)"
        )
    block_idx = torch.tensor(
        [_BLOCK_CLASS_INDEX[c] for c in _REACTION_BLOCK_CLASS],
        dtype=torch.long,
        device=selected_reaction.device,
    )
    wanted = block_idx[selected_reaction.long()].unsqueeze(1)  # [B, 1]
    classes = catalog_handle_classes.long().unsqueeze(0)  # [1, K]
    return classes != wanted


# ---------------------------------------------------------------------------
# Output dataclasses (auditable tensor + log-probability surfaces)
# ---------------------------------------------------------------------------

@dataclass
class StopOutput:
    """The binary termination decision — evaluated FIRST, always.

    Fields:
        logits: raw (mask-filled) logits ``[B, 2]`` — columns
            ``(CONTINUE, STOP)``.
        log_probs: ``log P(. | s_t)`` ``[B, 2]``.
        probs: ``exp(log_probs)`` — a valid normalized binary law.
        stop_log_prob: ``log P(STOP | s_t)`` ``[B]`` — the TB forward
            log-probability of the terminating action.
        continue_log_prob: ``log P(CONTINUE | s_t)`` ``[B]`` — the TB
            forward log-probability of the branch conditioning.
        forced_stop: rows whose continuation is structurally impossible
            (no legal reaction or exhausted block budget) — ``P(STOP)``
            is exactly 1 there (the CONTINUE logit is mask-filled).
    """

    logits: torch.Tensor  # [B, 2]
    log_probs: torch.Tensor  # [B, 2]
    probs: torch.Tensor  # [B, 2]
    stop_log_prob: torch.Tensor  # [B]
    continue_log_prob: torch.Tensor  # [B]
    forced_stop: torch.Tensor  # [B] bool

    @property
    def num_graphs(self) -> int:
        return int(self.log_probs.size(0))


@dataclass
class FamilyOutput:
    """The 9-way coarse reaction-family decision ``P(f | s, CONTINUE)``.

    Fields:
        logits: raw (mask-filled) family logits ``[B, 9]``.
        log_probs: the normalized law over the *legal* families ``[B, 9]``.
        probs: ``exp(log_probs)``.
        selected_family: the supervised/sampled family index ``[B]``
            int64 (into ``REACTION_FAMILY_NAMES``).
        selected_log_prob: ``log P(f_t | s)`` ``[B]`` — the TB forward
            contribution of the family component.
    """

    logits: torch.Tensor  # [B, 9]
    log_probs: torch.Tensor  # [B, 9]
    probs: torch.Tensor  # [B, 9]
    selected_family: torch.Tensor  # [B] int64
    selected_log_prob: torch.Tensor  # [B]

    @property
    def num_graphs(self) -> int:
        return int(self.log_probs.size(0))


@dataclass
class ReactionOutput:
    """The 27-way certified-reaction decision ``P(r | f, s, CONTINUE)``.

    The hierarchical conditional is realized by support restriction: the
    law is normalized over exactly the legal reactions *of the selected
    family* (``reaction_support_mask``), which makes
    ``P(f, r | s) = P(f | s) · P(r | f, s)`` an exact factorization.

    Fields:
        logits: raw (mask-filled) reaction logits ``[B, 27]``.
        log_probs: the conditional law over the selected family's legal
            reactions ``[B, 27]``.
        probs: ``exp(log_probs)``.
        selected_reaction: the supervised/sampled reaction index ``[B]``
            int64 (into ``POLICY_REACTION_IDS``).
        selected_family: the derived family of the selection ``[B]``
            int64 (the hierarchy reveals the family through the reaction).
        selected_log_prob: ``log P(r_t | f_t, s)`` ``[B]`` — the TB
            forward contribution of the reaction component.
    """

    logits: torch.Tensor  # [B, 27]
    log_probs: torch.Tensor  # [B, 27]
    probs: torch.Tensor  # [B, 27]
    selected_reaction: torch.Tensor  # [B] int64
    selected_family: torch.Tensor  # [B] int64
    selected_log_prob: torch.Tensor  # [B]

    @property
    def num_graphs(self) -> int:
        return int(self.log_probs.size(0))


@dataclass
class SynthonOutput:
    """The K-way catalog-synthon retrieval decision ``P(B | r, s)``.

    Fields:
        logits: scaled dot-product logits over the tied catalog
            embeddings ``[B, K]`` (mask-filled).
        log_probs: the law over the synthons of the selected reaction's
            block-side handle class ``[B, K]``.
        probs: ``exp(log_probs)``.
        selected_synthon: the supervised/sampled catalog index ``[B]``
            int64.
        selected_log_prob: ``log P(B_t | r_t, s)`` ``[B]`` — the TB
            forward contribution of the synthon component.
        retrieved_embedding: the tied catalog embedding of the selection
            ``[B, SYNTON_EMBED_DIM]`` — the conditioning surface for the
            parity and torsion heads (issue #24 pins the torsion
            conditioning on ``B_t``).
    """

    logits: torch.Tensor  # [B, K]
    log_probs: torch.Tensor  # [B, K]
    probs: torch.Tensor  # [B, K]
    selected_synthon: torch.Tensor  # [B] int64
    selected_log_prob: torch.Tensor  # [B]
    retrieved_embedding: torch.Tensor  # [B, embed_dim]

    @property
    def num_graphs(self) -> int:
        return int(self.log_probs.size(0))

    @property
    def catalog_size(self) -> int:
        return int(self.log_probs.size(1))


@dataclass
class ParityOutput:
    """The 2-way enantiomeric parity decision ``P(c | s, B)``.

    Scored exactly where the junction forms a new stereocenter
    (``required``); non-required junctions carry **no parity action** and
    contribute exactly ``0.0`` log-probability (a vacuous ``chiral``
    field on such an action is a documented environment no-op and is
    likewise not scored).

    Fields:
        logits: raw parity logits ``[B, 2]`` — columns ``("R", "S")``.
        log_probs: the normalized law ``[B, 2]``.
        probs: ``exp(log_probs)``.
        required: per-row stereocenter requirement ``[B]`` bool
            (chemistry-determined by the caller; see the module
            docstring for the determination contract).
        selected_parity: the supervised/sampled parity index ``[B]``
            int64 — ``-1`` = no action, ``0`` = ``'R'``, ``1`` = ``'S'``.
        selected_log_prob: ``log P(c_t | s, B_t)`` for required rows,
            exactly ``0.0`` otherwise ``[B]``.
    """

    logits: torch.Tensor  # [B, 2]
    log_probs: torch.Tensor  # [B, 2]
    probs: torch.Tensor  # [B, 2]
    required: torch.Tensor  # [B] bool
    selected_parity: torch.Tensor  # [B] int64
    selected_log_prob: torch.Tensor  # [B]

    @property
    def num_graphs(self) -> int:
        return int(self.log_probs.size(0))


@dataclass
class ContinueOutput:
    """The teacher-forced continuation stage: all five continuation heads.

    Assembled by :meth:`HierarchicalPolicyHeads.forward_continue` for the
    **continuing rows only** (STOP rows never enter this stage — the #24
    acceptance criterion).  The per-step TB forward log-probability is
    :meth:`step_log_prob`.
    """

    family: FamilyOutput
    reaction: ReactionOutput
    synthon: SynthonOutput
    parity: ParityOutput
    torsion: TorsionOutput

    @property
    def num_graphs(self) -> int:
        return self.family.num_graphs

    def step_log_prob(
        self, handle_log_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The TB forward log-probability of one full continuation action.

        ``log P(a_t | s_t) = log P(f) + log P(r | f) + log P(B | r) +
        [stereocenter] log P(c) + [rotatable] log P(k)`` (+ the optional
        #22 handle component, supplied by the caller from
        ``HandleSelectionOutput.log_probs`` — its law is over the
        backbone state, so it joins the sum here).  Every term is a pure
        dimensionless categorical log-probability: no densities, no
        units (the #24 acceptance criterion — the pre-V2 rad^-1 mixing is
        structurally absent).
        """
        total = (
            self.family.selected_log_prob
            + self.reaction.selected_log_prob
            + self.synthon.selected_log_prob
            + self.parity.selected_log_prob
            + self.torsion.selected_log_prob
        )
        if handle_log_probs is not None:
            if (
                not isinstance(handle_log_probs, torch.Tensor)
                or handle_log_probs.shape != total.shape
            ):
                raise TensorContractError(
                    f"handle_log_probs must be [{self.num_graphs}] matching "
                    "the continuation rows"
                )
            total = total + handle_log_probs
        return total


# ---------------------------------------------------------------------------
# The heads
# ---------------------------------------------------------------------------

def _validate_context(
    context: torch.Tensor, context_dim: int, param_dtype: torch.dtype, tag: str
) -> torch.Tensor:
    """Shared fail-closed validation of a decision-context batch."""
    if not isinstance(context, torch.Tensor):
        raise TensorContractError(
            f"[{tag}] `context` must be a torch.Tensor, got "
            f"{type(context).__name__}"
        )
    if context.dtype not in _FLOAT_INPUT_DTYPES:
        raise TensorContractError(
            f"[{tag}] `context` must have dtype float32 (the production "
            f"path, decision D-V2-005) or float64 (local numerical "
            f"verification only, decision D-V2-008), got {context.dtype}"
        )
    if context.dtype != param_dtype:
        raise TensorContractError(
            f"[{tag}] `context` has dtype {context.dtype} but the module "
            f"parameters are {param_dtype}; cast the module for local "
            "numerical verification or fix the producer"
        )
    if context.ndim != 2 or context.size(1) != context_dim:
        raise TensorContractError(
            f"[{tag}] `context` must be [B, {context_dim}], got "
            f"{tuple(context.shape)}"
        )
    if context.numel() == 0:
        raise TensorContractError(
            f"[{tag}] `context` must contain at least one row (fail-closed "
            "on empty batches)"
        )
    if not bool(torch.isfinite(context).all()):
        raise TensorContractError(
            f"[{tag}] `context` contains non-finite values (fail-closed)"
        )
    return context


def _validate_bool_vector(
    flag: Optional[torch.Tensor], num_graphs: int, tag: str, name: str
) -> torch.Tensor:
    if flag is None:
        return None
    if not isinstance(flag, torch.Tensor) or flag.dtype != torch.bool:
        raise TensorContractError(
            f"[{tag}] `{name}` must be a boolean tensor"
        )
    if flag.ndim != 1 or flag.shape[0] != num_graphs:
        raise TensorContractError(
            f"[{tag}] `{name}` must be [{num_graphs}], got "
            f"{tuple(flag.shape)}"
        )
    return flag


def _validate_index_vector(
    values: Optional[torch.Tensor],
    num_graphs: int,
    high: int,
    tag: str,
    name: str,
    allow_sentinel: Optional[int] = None,
) -> torch.Tensor:
    """Shared fail-closed validation of a supervised/sampled index batch."""
    if values is None:
        return None
    if not isinstance(values, torch.Tensor) or values.dtype not in (
        torch.int64,
        torch.int32,
    ):
        raise TensorContractError(
            f"[{tag}] `{name}` must be an integer tensor, got "
            f"{values.dtype if isinstance(values, torch.Tensor) else type(values).__name__}"
        )
    if values.ndim != 1 or values.shape[0] != num_graphs:
        raise TensorContractError(
            f"[{tag}] `{name}` must be [{num_graphs}], got "
            f"{tuple(values.shape)}"
        )
    values = values.long()
    lo = min(0, allow_sentinel) if allow_sentinel is not None else 0
    if bool((values < lo).any()) or bool((values >= high).any()):
        raise TensorContractError(
            f"[{tag}] `{name}` entries must lie in [{lo}, {high}) "
            f"(sentinel {allow_sentinel} allowed), got out-of-range values "
            "(fail-closed)"
        )
    return values


class TerminationHead(nn.Module):
    """The binary hierarchical termination head ``P(STOP | s_t)``.

    Evaluated **first**, on the decoder context alone, before any
    continuation head is constructed (#45 §7.3/§8.4; issue #24
    acceptance 1).  STOP is *not* a token inside any larger head — the
    pre-V2 probability leak (choosing a reaction family to express
    termination) is structurally impossible.

    States whose continuation is structurally impossible (no legal
    reaction, or an exhausted block budget) are forced to terminate: the
    CONTINUE logit is mask-filled, so ``P(STOP) = 1`` exactly and
    ``log P(STOP) = 0.0`` — the fail-closed resolution #45 §8.4 requires
    *before* any categorical continuation mask is applied.
    """

    def __init__(self, context_dim: int = DECODER_D_MODEL) -> None:
        super().__init__()
        context_dim = int(context_dim)
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        self.context_dim = context_dim
        self.logits = nn.Linear(context_dim, 2)

    def extra_repr(self) -> str:
        return f"context_dim={self.context_dim}"

    def audit(self) -> Dict[str, Any]:
        parameters = list(self.parameters())
        return {
            "module": type(self).__name__,
            "context_dim": self.context_dim,
            "num_actions": 2,
            "columns": ("CONTINUE", "STOP"),
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "deterministic_in_eval": True,
        }

    def forward(
        self, context: torch.Tensor, force_stop: Optional[torch.Tensor] = None
    ) -> StopOutput:
        """Evaluate the binary termination law.

        Parameters:
            context: ``[B, context_dim]`` decoder decision context.
            force_stop: optional ``[B]`` bool — rows that must terminate
                (from :func:`build_force_stop`).
        """
        context = _validate_context(
            context, self.context_dim, self.logits.weight.dtype, "stop-head"
        )
        num_graphs = context.size(0)
        force_stop = _validate_bool_vector(
            force_stop, num_graphs, "stop-head", "force_stop"
        )
        if force_stop is None:
            force_stop = torch.zeros(
                num_graphs, dtype=torch.bool, device=context.device
            )
        elif force_stop.device != context.device:
            force_stop = force_stop.to(context.device)

        raw = self.logits(context)  # [B, 2]
        mask = torch.zeros(
            (num_graphs, 2), dtype=torch.bool, device=raw.device
        )
        mask[:, CONTINUE_INDEX] = force_stop
        logits = apply_logit_mask(raw, mask)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return StopOutput(
            logits=logits,
            log_probs=log_probs,
            probs=probs,
            stop_log_prob=log_probs[:, STOP_INDEX],
            continue_log_prob=log_probs[:, CONTINUE_INDEX],
            forced_stop=force_stop,
        )


class ReactionFamilyHead(nn.Module):
    """The 9-way coarse reaction-family head ``P(f | s, CONTINUE)``."""

    def __init__(
        self, context_dim: int = DECODER_D_MODEL, num_families: int = POLICY_NUM_FAMILIES
    ) -> None:
        super().__init__()
        context_dim = int(context_dim)
        num_families = int(num_families)
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if num_families != POLICY_NUM_FAMILIES:
            raise ValueError(
                f"the certified registry defines exactly {POLICY_NUM_FAMILIES} "
                f"reaction families (D-V2-016); got num_families={num_families}"
            )
        self.context_dim = context_dim
        self.num_families = num_families
        self.logits = nn.Linear(context_dim, num_families)

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, num_families={self.num_families}"
        )

    def audit(self) -> Dict[str, Any]:
        parameters = list(self.parameters())
        return {
            "module": type(self).__name__,
            "context_dim": self.context_dim,
            "num_families": self.num_families,
            "family_order": list(REACTION_FAMILY_NAMES),
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "deterministic_in_eval": True,
        }

    def forward(
        self,
        context: torch.Tensor,
        family_mask: torch.Tensor,
        selected_family: Optional[torch.Tensor] = None,
    ) -> FamilyOutput:
        """Evaluate the family law over the legal families.

        Parameters:
            context: ``[B, context_dim]`` decoder decision context
                (continuing rows only).
            family_mask: ``[B, 9]`` bool legality mask (``True`` =
                illegal; from :func:`build_continuation_masks`).
            selected_family: optional supervised/sampled family indices
                ``[B]`` int64.
        """
        context = _validate_context(
            context, self.context_dim, self.logits.weight.dtype, "family-head"
        )
        num_graphs = context.size(0)
        self._validate_mask(family_mask, num_graphs, self.num_families, "family-head", "family_mask")
        selected_family = _validate_index_vector(
            selected_family, num_graphs, self.num_families, "family-head", "selected_family"
        )
        logits = apply_logit_mask(self.logits(context), family_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        if selected_family is None:
            selected_family = torch.full(
                (num_graphs,), -1, dtype=torch.int64, device=context.device
            )
            selected_log_prob = torch.full(
                (num_graphs,),
                float("nan"),
                dtype=log_probs.dtype,
                device=log_probs.device,
            )
        else:
            selected_family = selected_family.to(context.device)
            validate_target_actions(family_mask, selected_family)
            selected_log_prob = log_probs.gather(
                1, selected_family.unsqueeze(-1)
            ).squeeze(-1)
        return FamilyOutput(
            logits=logits,
            log_probs=log_probs,
            probs=probs,
            selected_family=selected_family,
            selected_log_prob=selected_log_prob,
        )

    def _validate_mask(
        self,
        mask: torch.Tensor,
        num_graphs: int,
        columns: int,
        tag: str,
        name: str,
    ) -> None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TensorContractError(
                f"[{tag}] `{name}` must be a boolean tensor"
            )
        if mask.ndim != 2 or tuple(mask.shape) != (num_graphs, columns):
            raise TensorContractError(
                f"[{tag}] `{name}` must be [{num_graphs}, {columns}], got "
                f"{tuple(mask.shape)}"
            )


class ReactionHead(nn.Module):
    """The 27-way certified-reaction head ``P(r | f, s, CONTINUE)``.

    The hierarchical conditional is realized by support restriction: the
    law is normalized over exactly the legal reactions of the selected
    family (:func:`reaction_support_mask`), giving the exact
    factorization ``P(f, r | s) = P(f | s) · P(r | f, s)``.
    """

    def __init__(
        self, context_dim: int = DECODER_D_MODEL, num_reactions: int = POLICY_NUM_REACTIONS
    ) -> None:
        super().__init__()
        context_dim = int(context_dim)
        num_reactions = int(num_reactions)
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if num_reactions != POLICY_NUM_REACTIONS:
            raise ValueError(
                f"the certified registry defines exactly {POLICY_NUM_REACTIONS} "
                f"policy reactions (D-V2-016); got num_reactions={num_reactions}"
            )
        self.context_dim = context_dim
        self.num_reactions = num_reactions
        self.reaction_logits = nn.Linear(context_dim, num_reactions)
        # family membership [9, 27] as a non-persistent buffer (moves with
        # the module, never enters the checkpoint state dict).
        membership = torch.zeros(
            (POLICY_NUM_FAMILIES, POLICY_NUM_REACTIONS), dtype=torch.bool
        )
        for f_idx, fname in enumerate(REACTION_FAMILY_NAMES):
            for member in FAMILY_MEMBERS[fname]:
                membership[f_idx, _REACTION_INDEX[member]] = True
        self.register_buffer("family_membership", membership, persistent=False)

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, num_reactions={self.num_reactions}"
        )

    def audit(self) -> Dict[str, Any]:
        parameters = list(self.parameters())
        return {
            "module": type(self).__name__,
            "context_dim": self.context_dim,
            "num_reactions": self.num_reactions,
            "reaction_order": list(POLICY_REACTION_IDS),
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "deterministic_in_eval": True,
        }

    def forward(
        self,
        context: torch.Tensor,
        selected_family: torch.Tensor,
        reaction_mask: torch.Tensor,
        selected_reaction: Optional[torch.Tensor] = None,
    ) -> ReactionOutput:
        """Evaluate the family-conditional reaction law.

        Parameters:
            context: ``[B, context_dim]`` (continuing rows only).
            selected_family: ``[B]`` int64 family indices conditioning the
                support.
            reaction_mask: ``[B, 27]`` bool legality mask (``True`` =
                illegal).
            selected_reaction: optional supervised/sampled reaction
                indices ``[B]`` int64.
        """
        context = _validate_context(
            context, self.context_dim, next(self.parameters()).dtype, "reaction-head"
        )
        num_graphs = context.size(0)
        self._validate_mask(
            reaction_mask, num_graphs, self.num_reactions, "reaction-head", "reaction_mask"
        )
        selected_family = _validate_index_vector(
            selected_family,
            num_graphs,
            POLICY_NUM_FAMILIES,
            "reaction-head",
            "selected_family",
        )
        if selected_family is None:
            raise TensorContractError(
                "[reaction-head] selected_family is required (the support "
                "conditioning of P(r | f, s))"
            )
        selected_family = selected_family.to(context.device)
        selected_reaction = _validate_index_vector(
            selected_reaction,
            num_graphs,
            self.num_reactions,
            "reaction-head",
            "selected_reaction",
        )

        member = self.family_membership.to(context.device).index_select(
            0, selected_family
        )  # [B, 27] bool — True = member of the selected family
        support = reaction_mask.to(context.device) | ~member
        logits = apply_logit_mask(self.reaction_logits(context), support)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if selected_reaction is None:
            selected_reaction = torch.full(
                (num_graphs,), -1, dtype=torch.int64, device=context.device
            )
            selected_log_prob = torch.full(
                (num_graphs,),
                float("nan"),
                dtype=log_probs.dtype,
                device=log_probs.device,
            )
        else:
            selected_reaction = selected_reaction.to(context.device)
            validate_target_actions(support, selected_reaction)
            selected_log_prob = log_probs.gather(
                1, selected_reaction.unsqueeze(-1)
            ).squeeze(-1)
        return ReactionOutput(
            logits=logits,
            log_probs=log_probs,
            probs=probs,
            selected_reaction=selected_reaction,
            selected_family=selected_family,
            selected_log_prob=selected_log_prob,
        )

    def _validate_mask(
        self,
        mask: torch.Tensor,
        num_graphs: int,
        columns: int,
        tag: str,
        name: str,
    ) -> None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TensorContractError(
                f"[{tag}] `{name}` must be a boolean tensor"
            )
        if mask.ndim != 2 or tuple(mask.shape) != (num_graphs, columns):
            raise TensorContractError(
                f"[{tag}] `{name}` must be [{num_graphs}, {columns}], got "
                f"{tuple(mask.shape)}"
            )


class SynthonRetrievalHead(nn.Module):
    """The K-way catalog-synthon retrieval head ``P(B | r, s)``.

    Scoring is a scaled dot product between a projection of the decision
    context and the **tied catalog embeddings** — one
    ``nn.Embedding(catalog_size, embed_dim)`` table that is the single
    authoritative synthon representation: the retrieval basis here, the
    conditioning surface for the parity/torsion heads, and (per #12
    CHEM-02) the initialization target of the ``pharm3d`` QR-projected
    catalog descriptors.

    The legality mask restricts the law to the synthons carrying the
    selected reaction's block-side handle class
    (:func:`synthon_legality_mask`) — a class-level prefilter only; the
    environment enforces the exact one-handle chemistry at execution and
    #40's curation guarantees curated blocks expose exactly one coupling
    handle.
    """

    def __init__(
        self,
        context_dim: int = DECODER_D_MODEL,
        catalog_size: int = SYNTON_CATALOG_SIZE,
        embed_dim: int = SYNTON_EMBED_DIM,
    ) -> None:
        super().__init__()
        context_dim = int(context_dim)
        catalog_size = int(catalog_size)
        embed_dim = int(embed_dim)
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if catalog_size < 1:
            raise ValueError(f"catalog_size must be >= 1, got {catalog_size}")
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        self.context_dim = context_dim
        self.catalog_size = catalog_size
        self.embed_dim = embed_dim
        # The tied catalog embedding table (the single synthon
        # representation — see the class docstring).
        self.catalog_embeddings = nn.Embedding(catalog_size, embed_dim)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        # reaction index -> block handle-class index [27] (non-persistent
        # metadata buffer).
        block_idx = torch.tensor(
            [_BLOCK_CLASS_INDEX[c] for c in _REACTION_BLOCK_CLASS], dtype=torch.long
        )
        self.register_buffer("reaction_block_class", block_idx, persistent=False)
        # Optional attached catalog handle-class assignment [K] int64.
        self.register_buffer(
            "catalog_handle_classes", torch.zeros(0, dtype=torch.long), persistent=False
        )

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, catalog_size={self.catalog_size}, "
            f"embed_dim={self.embed_dim}"
        )

    def audit(self) -> Dict[str, Any]:
        parameters = list(self.parameters())
        catalog_bytes = self.catalog_size * self.embed_dim * 4
        return {
            "module": type(self).__name__,
            "context_dim": self.context_dim,
            "catalog_size": self.catalog_size,
            "embed_dim": self.embed_dim,
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "tied_catalog_embedding_parameters": (
                self.catalog_size * self.embed_dim
            ),
            "tied_catalog_embedding_bytes_fp32": catalog_bytes,
            "logit_scale": 1.0 / math.sqrt(self.embed_dim),
            "catalog_handle_classes_attached": int(
                self.catalog_handle_classes.numel()
            )
            > 0,
            "deterministic_in_eval": True,
        }

    # -- catalog attachment ----------------------------------------------------

    def set_catalog_handle_classes(
        self, catalog_handle_classes: torch.Tensor
    ) -> None:
        """Attach the catalog's per-synthon handle-class assignment.

        ``catalog_handle_classes[k]`` is the index (into
        ``BLOCK_HANDLE_CLASS_VOCAB``) of synthon ``k``'s coupling handle
        class.  The assignment must cover the full table; it is stored as
        a non-persistent buffer (catalog metadata, not model state).
        """
        if not isinstance(catalog_handle_classes, torch.Tensor):
            raise TensorContractError(
                "catalog_handle_classes must be a torch.Tensor"
            )
        if catalog_handle_classes.dtype not in (torch.int64, torch.int32):
            raise TensorContractError(
                f"catalog_handle_classes must be an integer tensor, got "
                f"{catalog_handle_classes.dtype}"
            )
        if catalog_handle_classes.ndim != 1:
            raise TensorContractError(
                f"catalog_handle_classes must be [K], got "
                f"{tuple(catalog_handle_classes.shape)}"
            )
        if catalog_handle_classes.numel() != self.catalog_size:
            raise TensorContractError(
                f"catalog_handle_classes has {catalog_handle_classes.numel()} "
                f"entries but the embedding table has {self.catalog_size} rows"
            )
        vals = catalog_handle_classes.long()
        if bool((vals < 0).any()) or bool(
            (vals >= len(BLOCK_HANDLE_CLASS_VOCAB)).any()
        ):
            raise TensorContractError(
                "catalog_handle_classes entries out of the block-class "
                "vocabulary range (fail-closed)"
            )
        self.catalog_handle_classes = vals.to(self.catalog_embeddings.weight.device)

    def embedding(self, index: torch.Tensor) -> torch.Tensor:
        """The tied catalog embeddings of catalog indices ``[...]``."""
        if not isinstance(index, torch.Tensor) or index.dtype not in (
            torch.int64,
            torch.int32,
        ):
            raise TensorContractError(
                "index must be an integer tensor of catalog indices"
            )
        if bool((index < 0).any()) or bool((index >= self.catalog_size).any()):
            raise TensorContractError(
                f"catalog indices out of range [0, {self.catalog_size})"
            )
        return self.catalog_embeddings(index.long())

    # -- forward ----------------------------------------------------------------

    def forward(
        self,
        context: torch.Tensor,
        selected_reaction: torch.Tensor,
        selected_synthon: Optional[torch.Tensor] = None,
        catalog_handle_classes: Optional[torch.Tensor] = None,
    ) -> SynthonOutput:
        """Evaluate the synthon retrieval law.

        Parameters:
            context: ``[B, context_dim]`` (continuing rows only).
            selected_reaction: ``[B]`` int64 reaction indices (the
                block-class conditioning).
            selected_synthon: optional supervised/sampled catalog indices
                ``[B]`` int64.
            catalog_handle_classes: optional per-call class assignment
                (overrides an attached one).
        """
        context = _validate_context(
            context,
            self.context_dim,
            self.context_proj.weight.dtype,
            "synthon-head",
        )
        num_graphs = context.size(0)
        selected_reaction = _validate_index_vector(
            selected_reaction,
            num_graphs,
            POLICY_NUM_REACTIONS,
            "synthon-head",
            "selected_reaction",
        )
        if selected_reaction is None:
            raise TensorContractError(
                "[synthon-head] selected_reaction is required (the "
                "block-class conditioning of P(B | r, s))"
            )
        selected_reaction = selected_reaction.to(context.device)
        selected_synthon = _validate_index_vector(
            selected_synthon,
            num_graphs,
            self.catalog_size,
            "synthon-head",
            "selected_synthon",
        )

        classes = catalog_handle_classes
        if classes is None:
            classes = self.catalog_handle_classes
        if classes.numel() != self.catalog_size:
            raise TensorContractError(
                "[synthon-head] no catalog handle-class assignment "
                "available: attach one with set_catalog_handle_classes or "
                "pass catalog_handle_classes (fail-closed)"
            )
        classes = classes.to(context.device)

        mask = synthon_legality_mask(classes, selected_reaction)
        # Rows with no legal synthon fail closed inside apply_logit_mask:
        # a legal reaction (per build_continuation_masks) guarantees its
        # block class is available, so this raises only on inconsistent
        # catalog/mask wiring — exactly the fail-loudly contract.
        query = self.context_proj(context)  # [B, embed_dim]
        table = self.catalog_embeddings.weight  # [K, embed_dim]
        scale = 1.0 / math.sqrt(self.embed_dim)
        raw = (query @ table.t()) * scale  # [B, K]
        logits = apply_logit_mask(raw, mask)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if selected_synthon is None:
            selected_synthon = torch.full(
                (num_graphs,), -1, dtype=torch.int64, device=context.device
            )
            selected_log_prob = torch.full(
                (num_graphs,),
                float("nan"),
                dtype=log_probs.dtype,
                device=log_probs.device,
            )
            retrieved = torch.zeros(
                (num_graphs, self.embed_dim),
                dtype=context.dtype,
                device=context.device,
            )
        else:
            selected_synthon = selected_synthon.to(context.device)
            validate_target_actions(mask, selected_synthon)
            selected_log_prob = log_probs.gather(
                1, selected_synthon.unsqueeze(-1)
            ).squeeze(-1)
            retrieved = self.embedding(selected_synthon)
        return SynthonOutput(
            logits=logits,
            log_probs=log_probs,
            probs=probs,
            selected_synthon=selected_synthon,
            selected_log_prob=selected_log_prob,
            retrieved_embedding=retrieved,
        )

    @property
    def num_reactions(self) -> int:
        """The registry's reaction count (metadata, not a layer)."""
        return POLICY_NUM_REACTIONS


class ChiralParityHead(nn.Module):
    """The 2-way enantiomeric parity head ``P(c | s, B)`` for junction
    stereocenters.

    Input: the decision context concatenated with the tied catalog
    embedding of the selected synthon (the junction's two sides).  Scored
    exactly where the junction forms a new stereocenter; the mapping
    ``('R' -> 0, 'S' -> 1)`` matches ``AssemblyAction.chiral`` — index 1
    (``'S'``) selects the environment's mirrored placement.
    """

    def __init__(
        self,
        context_dim: int = DECODER_D_MODEL,
        synthon_dim: int = SYNTON_EMBED_DIM,
    ) -> None:
        super().__init__()
        context_dim = int(context_dim)
        synthon_dim = int(synthon_dim)
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if synthon_dim < 1:
            raise ValueError(f"synthon_dim must be >= 1, got {synthon_dim}")
        self.context_dim = context_dim
        self.synthon_dim = synthon_dim
        self.logits = nn.Linear(context_dim + synthon_dim, len(PARITY_ACTIONS))

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, synthon_dim={self.synthon_dim}"
        )

    def audit(self) -> Dict[str, Any]:
        parameters = list(self.parameters())
        return {
            "module": type(self).__name__,
            "context_dim": self.context_dim,
            "synthon_dim": self.synthon_dim,
            "actions": list(PARITY_ACTIONS),
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "deterministic_in_eval": True,
        }

    def forward(
        self,
        context: torch.Tensor,
        synthon_embedding: torch.Tensor,
        required: Optional[torch.Tensor] = None,
        selected_parity: Optional[torch.Tensor] = None,
    ) -> ParityOutput:
        """Evaluate the parity law for junction stereocenters.

        Parameters:
            context: ``[B, context_dim]`` (continuing rows only).
            synthon_embedding: ``[B, synthon_dim]`` tied catalog embedding
                of the selected synthon.
            required: ``[B]`` bool — the junction forms a new
                stereocenter.  Default: all rows.
            selected_parity: ``[B]`` int64 — ``-1`` = no action, ``0`` =
                ``'R'``, ``1`` = ``'S'``.
        """
        context = _validate_context(
            context, self.context_dim, self.logits.weight.dtype, "parity-head"
        )
        num_graphs = context.size(0)
        if (
            not isinstance(synthon_embedding, torch.Tensor)
            or synthon_embedding.dtype != context.dtype
            or synthon_embedding.ndim != 2
            or synthon_embedding.shape
            != (num_graphs, self.synthon_dim)
        ):
            raise TensorContractError(
                f"[parity-head] synthon_embedding must be "
                f"[{num_graphs}, {self.synthon_dim}] with the context dtype"
            )
        if not bool(torch.isfinite(synthon_embedding).all()):
            raise TensorContractError(
                "[parity-head] synthon_embedding contains non-finite values "
                "(fail-closed)"
            )
        required = _validate_bool_vector(
            required, num_graphs, "parity-head", "required"
        )
        if required is None:
            required = torch.ones(
                num_graphs, dtype=torch.bool, device=context.device
            )
        else:
            required = required.to(context.device)
        selected_parity = _validate_index_vector(
            selected_parity,
            num_graphs,
            len(PARITY_ACTIONS),
            "parity-head",
            "selected_parity",
            allow_sentinel=-1,
        )
        if selected_parity is None:
            selected_parity = torch.full(
                (num_graphs,), -1, dtype=torch.int64, device=context.device
            )
        else:
            selected_parity = selected_parity.to(context.device)
            # Fail-closed supervision consistency: a stereogenic junction
            # MUST carry an explicit parity action (#45 §9.6 — never
            # silently accept an incomplete trajectory step).
            missing = required & (selected_parity < 0)
            if bool(missing.any()):
                rows = torch.nonzero(missing, as_tuple=False).flatten().tolist()
                raise TensorContractError(
                    f"[parity-head] rows {rows} form junction stereocenters "
                    f"but carry no parity action (selected_parity = -1); "
                    "supervised trajectories must supply c_t in {R, S} "
                    "(fail-closed)"
                )

        logits = self.logits(
            torch.cat((context, synthon_embedding), dim=-1)
        )  # [B, 2]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        selected_log_prob = torch.zeros(
            num_graphs, dtype=log_probs.dtype, device=log_probs.device
        )
        has_action = required & (selected_parity >= 0)
        if bool(has_action.any()):
            rows = torch.nonzero(has_action, as_tuple=False).flatten()
            selected_log_prob = selected_log_prob.index_copy(
                0,
                rows,
                log_probs.gather(
                    1, selected_parity[rows].unsqueeze(-1)
                ).squeeze(-1),
            )
        # A parity declared on a non-stereogenic junction is a vacuous
        # environment no-op: computed above but never scored (0.0), by
        # the documented permissiveness rule.
        return ParityOutput(
            logits=logits,
            log_probs=log_probs,
            probs=probs,
            required=required,
            selected_parity=selected_parity,
            selected_log_prob=selected_log_prob,
        )


# ---------------------------------------------------------------------------
# The container — all five continuation heads + the termination head
# ---------------------------------------------------------------------------

class HierarchicalPolicyHeads(nn.Module):
    """The decoupled hierarchical policy heads of issue #24 MODEL-04.

    Registers the six heads over the decoder decision context:

    * :class:`TerminationHead` — binary ``P(STOP | s)``, evaluated
      **first**;
    * :class:`ReactionFamilyHead` — 9-way ``P(f | s, CONTINUE)``;
    * :class:`ReactionHead` — 27-way ``P(r | f, s, CONTINUE)``;
    * :class:`SynthonRetrievalHead` — 50k-way ``P(B | r, s)`` with the
      tied catalog embeddings;
    * :class:`ChiralParityHead` — 2-way ``P(c | s, B)`` where the
      junction forms a stereocenter;
    * :class:`TorsionHead` — 36-bin categorical ``P(k | s, B)``.

    The two evaluation stages are structurally separate methods —
    :meth:`forward_stop` and the continuation family
    (:meth:`forward_family` / :meth:`forward_reaction` /
    :meth:`forward_synthon` / :meth:`forward_parity` /
    :meth:`forward_torsion`, or the teacher-forced one-shot
    :meth:`forward_continue`).  Calling ``forward_stop`` alone — the
    rollout pattern for terminating rows — never constructs or executes
    any continuation head (the #24 acceptance criterion, pinned by
    forward-hook tests).
    """

    def __init__(
        self,
        context_dim: int = DECODER_D_MODEL,
        catalog_size: int = SYNTON_CATALOG_SIZE,
        synthon_embed_dim: int = SYNTON_EMBED_DIM,
        torsion_hidden_dim: int = TORSION_HEAD_CONTEXT_DIM // 2,
    ) -> None:
        super().__init__()
        context_dim = int(context_dim)
        catalog_size = int(catalog_size)
        synthon_embed_dim = int(synthon_embed_dim)
        torsion_hidden_dim = int(torsion_hidden_dim)
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if catalog_size < 1:
            raise ValueError(f"catalog_size must be >= 1, got {catalog_size}")
        if synthon_embed_dim < 1:
            raise ValueError(
                f"synthon_embed_dim must be >= 1, got {synthon_embed_dim}"
            )
        if torsion_hidden_dim < 1:
            raise ValueError(
                f"torsion_hidden_dim must be >= 1, got {torsion_hidden_dim}"
            )
        self.context_dim = context_dim
        self.catalog_size = catalog_size
        self.synthon_embed_dim = synthon_embed_dim

        self.stop_head = TerminationHead(context_dim)
        self.family_head = ReactionFamilyHead(context_dim)
        self.reaction_head = ReactionHead(context_dim)
        self.synthon_head = SynthonRetrievalHead(
            context_dim, catalog_size, synthon_embed_dim
        )
        self.parity_head = ChiralParityHead(context_dim, synthon_embed_dim)
        self.torsion_head = TorsionHead(
            context_dim=context_dim,
            synthon_dim=synthon_embed_dim,
            hidden_dim=torsion_hidden_dim,
            num_bins=TORSION_BINS,
        )

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, catalog_size={self.catalog_size}, "
            f"synthon_embed_dim={self.synthon_embed_dim}"
        )

    # -- audit ---------------------------------------------------------------

    def audit(self) -> Dict[str, Any]:
        """Parameter/dimension audit (constitution #45 §7.3): counts,
        trainability, dimensions, the tied catalog table, mask
        cardinalities, and the head-by-head breakdown."""
        parameters = list(self.parameters())
        head_audit = {
            name: module.audit()
            for name, module in (
                ("stop_head", self.stop_head),
                ("family_head", self.family_head),
                ("reaction_head", self.reaction_head),
                ("synthon_head", self.synthon_head),
                ("parity_head", self.parity_head),
                ("torsion_head", self.torsion_head),
            )
        }
        return {
            "module": type(self).__name__,
            "context_dim": self.context_dim,
            "catalog_size": self.catalog_size,
            "synthon_embed_dim": self.synthon_embed_dim,
            "action_space": {
                "stop": 2,
                "family": POLICY_NUM_FAMILIES,
                "reaction": POLICY_NUM_REACTIONS,
                "synthon": self.catalog_size,
                "parity": len(PARITY_ACTIONS),
                "torsion_bins": TORSION_BINS,
            },
            "heads": head_audit,
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "tied_catalog_embeddings_bytes_fp32": (
                self.catalog_size * self.synthon_embed_dim * 4
            ),
            "tb_probability_measures": {
                "stop": "categorical_binary",
                "family": "categorical_9",
                "reaction": "categorical_27_family_conditional",
                "synthon": "categorical_K",
                "parity": "categorical_2_conditional",
                "torsion": "categorical_36",
                "von_mises_outputs_in_tb": False,
            },
            "deterministic_in_eval": True,
        }

    # -- stage 1: termination (always first) ----------------------------------

    def forward_stop(
        self, context: torch.Tensor, force_stop: Optional[torch.Tensor] = None
    ) -> StopOutput:
        """Evaluate the binary termination law (stage 1; no continuation
        head is touched)."""
        return self.stop_head(context, force_stop)

    # -- stage 2: the continuation heads ----------------------------------------

    def forward_family(
        self, context: torch.Tensor, family_mask: torch.Tensor
    ) -> FamilyOutput:
        """The 9-way family law (distribution-only pass)."""
        return self.family_head(context, family_mask)

    def forward_reaction(
        self,
        context: torch.Tensor,
        selected_family: torch.Tensor,
        reaction_mask: torch.Tensor,
        selected_reaction: Optional[torch.Tensor] = None,
    ) -> ReactionOutput:
        """The family-conditional 27-way reaction law."""
        return self.reaction_head(
            context, selected_family, reaction_mask, selected_reaction
        )

    def forward_synthon(
        self,
        context: torch.Tensor,
        selected_reaction: torch.Tensor,
        selected_synthon: Optional[torch.Tensor] = None,
        catalog_handle_classes: Optional[torch.Tensor] = None,
    ) -> SynthonOutput:
        """The K-way synthon retrieval law."""
        return self.synthon_head(
            context, selected_reaction, selected_synthon, catalog_handle_classes
        )

    def forward_parity(
        self,
        context: torch.Tensor,
        synthon_embedding: torch.Tensor,
        required: Optional[torch.Tensor] = None,
        selected_parity: Optional[torch.Tensor] = None,
    ) -> ParityOutput:
        """The 2-way parity law for junction stereocenters."""
        return self.parity_head(
            context, synthon_embedding, required, selected_parity
        )

    def forward_torsion(
        self,
        context: torch.Tensor,
        synthon_embedding: torch.Tensor,
        required: Optional[torch.Tensor] = None,
        selected_bin: Optional[torch.Tensor] = None,
        bin_mask: Optional[torch.Tensor] = None,
    ) -> TorsionOutput:
        """The 36-bin categorical torsion law."""
        return self.torsion_head(
            context,
            synthon_embedding,
            required=required,
            selected_bin=selected_bin,
            bin_mask=bin_mask,
        )

    # -- the teacher-forced one-shot --------------------------------------------

    def forward_continue(
        self,
        context: torch.Tensor,
        selected_reaction: torch.Tensor,
        selected_synthon: torch.Tensor,
        selected_parity: torch.Tensor,
        selected_torsion_bin: torch.Tensor,
        stereocenter_required: torch.Tensor,
        family_mask: torch.Tensor,
        reaction_mask: torch.Tensor,
        catalog_handle_classes: Optional[torch.Tensor] = None,
        bin_mask: Optional[torch.Tensor] = None,
    ) -> ContinueOutput:
        """Teacher-forced evaluation of the full continuation stage.

        **This method is for the CONTINUING rows only** — the caller
        passes ``context`` (and every per-row input) already subset to
        the graphs whose STOP decision was CONTINUE.  STOP rows never
        reach this stage (the #24 acceptance criterion).

        The selected family is *derived* from the selected reaction (the
        hierarchy reveals the family through the reaction choice); the
        torsion requirement is *derived* from the reaction's resonance
        lock (locked junctions carry no torsion action — the environment
        snaps their dihedral deterministically).

        Parameters:
            context: ``[B', context_dim]`` — continuing rows only.
            selected_reaction: ``[B']`` int64 reaction indices.
            selected_synthon: ``[B']`` int64 catalog indices.
            selected_parity: ``[B']`` int64 — ``-1`` = no action, ``0`` =
                ``'R'``, ``1`` = ``'S'``.
            selected_torsion_bin: ``[B']`` int64 — ``0`` = no action
                (locked junction), else 1..36.
            stereocenter_required: ``[B']`` bool — the junction forms a
                new stereocenter (chemistry-determined by the caller).
            family_mask / reaction_mask: legality masks from
                :func:`build_continuation_masks` (subset to the same
                rows).
            catalog_handle_classes: optional per-call catalog assignment
                (overrides the attached one).
            bin_mask: optional ``[B', 36]`` torsion bin legality mask.
        """
        context = _validate_context(
            context,
            self.context_dim,
            next(self.parameters()).dtype,
            "hierarchical-heads",
        )
        num_graphs = context.size(0)
        selected_reaction = _validate_index_vector(
            selected_reaction,
            num_graphs,
            POLICY_NUM_REACTIONS,
            "hierarchical-heads",
            "selected_reaction",
        )
        if selected_reaction is None:
            raise TensorContractError(
                "[hierarchical-heads] selected_reaction is required"
            )
        selected_reaction = selected_reaction.to(context.device)
        stereocenter_required = _validate_bool_vector(
            stereocenter_required, num_graphs, "hierarchical-heads",
            "stereocenter_required",
        )
        if stereocenter_required is None:
            raise TensorContractError(
                "[hierarchical-heads] stereocenter_required is required "
                "(the parity scoring condition)"
            )
        stereocenter_required = stereocenter_required.to(context.device)

        # The family revealed by the reaction choice.
        family_of = torch.tensor(
            _REACTION_FAMILY_OF, dtype=torch.long, device=context.device
        )
        selected_family = family_of.index_select(0, selected_reaction)

        family_out = self.family_head(
            context, family_mask, selected_family
        )
        reaction_out = self.reaction_head(
            context, selected_family, reaction_mask, selected_reaction
        )
        synthon_out = self.synthon_head(
            context,
            selected_reaction,
            selected_synthon,
            catalog_handle_classes,
        )
        parity_out = self.parity_head(
            context,
            synthon_out.retrieved_embedding,
            required=stereocenter_required,
            selected_parity=selected_parity,
        )

        # The torsion requirement is derived from the resonance lock.
        locked = torch.tensor(
            _REACTION_LOCKED, dtype=torch.bool, device=context.device
        )
        torsion_required = ~locked.index_select(0, selected_reaction)
        torsion_out = self.torsion_head(
            context,
            synthon_out.retrieved_embedding,
            required=torsion_required,
            selected_bin=selected_torsion_bin,
            bin_mask=bin_mask,
        )
        return ContinueOutput(
            family=family_out,
            reaction=reaction_out,
            synthon=synthon_out,
            parity=parity_out,
            torsion=torsion_out,
        )


#: The frozen production parameter profile of the container (context 256,
#: catalog 50,000 x 64, torsion trunk 128) — version-controlled arithmetic
#: for #25's assembly audit (#45 §5.2: frozen dimensions, never probed).
HIERARCHICAL_HEADS_PARAMETER_PROFILE = (
    (DECODER_D_MODEL + 0) * 2 + 2  # stop
    + (DECODER_D_MODEL * POLICY_NUM_FAMILIES + POLICY_NUM_FAMILIES)  # family
    + (DECODER_D_MODEL * POLICY_NUM_REACTIONS + POLICY_NUM_REACTIONS)  # reaction
    + (SYNTON_CATALOG_SIZE * SYNTON_EMBED_DIM)  # tied catalog table
    + (DECODER_D_MODEL * SYNTON_EMBED_DIM + SYNTON_EMBED_DIM)  # synthon proj
    + ((DECODER_D_MODEL + SYNTON_EMBED_DIM) * len(PARITY_ACTIONS) + len(PARITY_ACTIONS))  # parity
    + ((DECODER_D_MODEL + SYNTON_EMBED_DIM) * TORSION_HEAD_CONTEXT_DIM // 2
       + TORSION_HEAD_CONTEXT_DIM // 2)  # torsion trunk
    + (TORSION_HEAD_CONTEXT_DIM // 2 * TORSION_BINS + TORSION_BINS)  # torsion logits
    + (TORSION_HEAD_CONTEXT_DIM // 2 * 2 + 2)  # torsion mu_xy
    + (TORSION_HEAD_CONTEXT_DIM // 2 * 1 + 1)  # torsion kappa
)


# ===========================================================================
# Issue #25 MODEL-05 — Complete policy assembly (SynTreePolicy) with the
# #39 seed/anchor selection component.
#
# The assembly unifies, in one registered module with one audit:
#
#   * the single authoritative ``BipartitePaiNN`` backbone (#21; frozen
#     256/8/20/5.0 profile, activation checkpointing ON by default),
#   * the agency-enabled handle-selection head (#22),
#   * the two-layer stacked cross-attention decoder (#23),
#   * the decoupled hierarchical policy heads of #24 (termination, family,
#     reaction, synthon retrieval with the tied catalog table, parity,
#     torsion),
#   * the #39 seed/anchor selection component: a deterministic
#     rotation-equivariant pocket-anchor candidate generator plus a
#     factorized ``P(anchor | pocket) * P(seed | anchor, pocket)`` head
#     whose seed side scores through the SAME tied catalog embedding table
#     (one authoritative synthon representation, counted once).
#
# Parameter-budget reconciliation (decision D-V2-019): issue #25's
# ``~4.8M`` estimate predates the frozen per-issue profiles; the closed and
# verified profiles of #21 (5,999,104), #22 (224,896), #23 (1,219,328) and
# #24 (3,272,975) are authoritative under #45 §5.2 (dimensions frozen in
# version-controlled configuration, never re-derived to hit an estimate).
# The BINDING acceptance criterion — total parameter memory < 50 MB — is
# satisfied with margin: 10,880,719 parameters = 43.5 MB FP32.
# ===========================================================================


# ---------------------------------------------------------------------------
# The deterministic pocket-anchor candidate generator (#39 component 1)
# ---------------------------------------------------------------------------

#: Default maximum anchor candidates per pocket (#39 acceptance: "the
#: environment can enumerate multiple valid seed/anchor candidates").
MAX_POCKET_ANCHORS = 8

#: Default minimum separation (Angstrom) between distinct anchor atoms —
#: two pocket atoms closer than this never seed separate anchors, so a
#: single contiguous subpocket yields one anchor, while separated
#: subpockets yield one anchor each (the multi-subpocket criterion).
ANCHOR_MIN_SEPARATION = 4.0


def pocket_anchor_candidates(
    pocket_pos: torch.Tensor,
    pocket_z: torch.Tensor,
    pocket_charge: torch.Tensor,
    *,
    max_anchors: int = MAX_POCKET_ANCHORS,
    min_separation: float = ANCHOR_MIN_SEPARATION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic anchor decomposition of ONE pocket (#39).

    Farthest-point selection over the pocket atoms with strictly
    rotation-invariant criteria and tie-breaks:

    * the first anchor is the atom farthest from the pocket centroid
      (tie-break: lower charge, then lower atomic number, then lower atom
      index — all invariant under rigid motions of a fixed atom list);
    * each subsequent anchor is the atom maximizing the minimum distance
      to the already-selected anchors (same tie-break order), stopping at
      ``max_anchors`` or when the best remaining separation falls below
      ``min_separation``;
    * every pocket atom is assigned to its nearest anchor atom (ties to
      the lower anchor index), and each anchor's *position* is the
      centroid of its assigned atoms.

    The output is therefore **equivariant**: rotating/translating the
    pocket rotates/translates the anchor positions and leaves the
    assignment bit-identical (pinned by test) — which is what makes the
    learned anchor scoring exactly SE(3)-invariant.

    Returns ``(anchor_positions [M, 3], assignment [N_p] int64)`` where
    ``assignment[i]`` is atom ``i``'s anchor index in ``[0, M)`` and
    ``M >= 1`` for any non-empty pocket.
    """
    if not isinstance(pocket_pos, torch.Tensor) or pocket_pos.ndim != 2:
        raise TensorContractError(
            f"[anchors] pocket_pos must be [N, 3], got {type(pocket_pos).__name__}"
        )
    if tuple(pocket_pos.shape)[1] != 3:
        raise TensorContractError(
            f"[anchors] pocket_pos must be [N, 3], got {tuple(pocket_pos.shape)}"
        )
    num_atoms = int(pocket_pos.shape[0])
    if num_atoms == 0:
        raise TensorContractError(
            "[anchors] an empty pocket has no anchor candidates (fail-closed)"
        )
    if not isinstance(pocket_z, torch.Tensor) or pocket_z.shape != (num_atoms,):
        raise TensorContractError("[anchors] pocket_z must be [N]")
    if not isinstance(pocket_charge, torch.Tensor) or pocket_charge.shape != (
        num_atoms,
    ):
        raise TensorContractError("[anchors] pocket_charge must be [N]")
    if not bool(torch.isfinite(pocket_pos).all()):
        raise TensorContractError("[anchors] pocket_pos contains non-finite values")
    if not bool(torch.isfinite(pocket_charge).all()):
        raise TensorContractError("[anchors] pocket_charge contains non-finite values")
    max_anchors = int(max_anchors)
    min_separation = float(min_separation)
    if max_anchors < 1:
        raise TensorContractError(f"[anchors] max_anchors must be >= 1, got {max_anchors}")
    if not (min_separation > 0.0):
        raise TensorContractError(
            f"[anchors] min_separation must be > 0, got {min_separation}"
        )

    pos = pocket_pos.to(torch.float64)
    charge = pocket_charge.to(torch.float64)
    z = pocket_z.to(torch.float64)

    # Invariant tie-break order: (charge, atomic number, atom index).
    order = sorted(
        range(num_atoms), key=lambda i: (float(charge[i]), float(z[i]), i)
    )

    centroid = pos.mean(dim=0)
    dist_to_centroid = (pos - centroid).norm(dim=-1)

    selected: List[int] = []
    # First anchor: farthest from the centroid under the tie-break order.
    best_key, best_val = None, -1.0
    for i in order:
        value = float(dist_to_centroid[i])
        if value > best_val:
            best_key, best_val = i, value
    selected.append(best_key)

    # Subsequent anchors: maximize the minimum distance to the selection.
    min_dist = (pos - pos[selected[0]]).norm(dim=-1)
    while len(selected) < max_anchors:
        best_key, best_val = None, -1.0
        for i in order:
            if i in selected:
                continue
            value = float(min_dist[i])
            if value > best_val:
                best_key, best_val = i, value
        if best_key is None or best_val < min_separation:
            break
        selected.append(best_key)
        dist_new = (pos - pos[best_key]).norm(dim=-1)
        min_dist = torch.minimum(min_dist, dist_new)

    # Assignment: nearest anchor atom (ties -> the lower anchor index).
    anchor_atoms = torch.tensor(selected, dtype=torch.long, device=pos.device)
    distances = torch.cdist(pos, pos.index_select(0, anchor_atoms))  # [N, M]
    assignment = torch.argmin(distances, dim=-1).to(torch.int64)

    # Anchor positions: centroids of the assigned atoms.
    num_anchors = len(selected)
    # Device-following accumulators: pocket_pos may live on CUDA (the
    # authorized T4 path); CPU-hardcoded zeros crash index_add_ there
    # (first surfaced S-V2-16 on the Tesla T4).
    positions = torch.zeros(
        (num_anchors, 3), dtype=torch.float64, device=pos.device
    )
    counts = torch.zeros((num_anchors,), dtype=torch.float64, device=pos.device)
    positions.index_add_(0, assignment, pos)
    counts.index_add_(
        0,
        assignment,
        torch.ones(num_atoms, dtype=torch.float64, device=pos.device),
    )
    if bool((counts == 0).any()):
        # Cannot happen (every anchor atom assigns to itself), but fail
        # loudly rather than emit NaN centroids.
        raise TensorContractError("[anchors] an anchor captured no atoms")
    positions = positions / counts.unsqueeze(-1)
    return (
        positions.to(pocket_pos.dtype),
        assignment,
    )


def batched_pocket_anchor_candidates(
    pocket_pos: torch.Tensor,
    pocket_z: torch.Tensor,
    pocket_charge: torch.Tensor,
    pocket_batch: torch.Tensor,
    num_graphs: int,
    *,
    max_anchors: int = MAX_POCKET_ANCHORS,
    min_separation: float = ANCHOR_MIN_SEPARATION,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The per-graph flattening of :func:`pocket_anchor_candidates`.

    Returns ``(anchor_positions [M, 3], anchor_batch [M] int64,
    assignment [N_p] int64, anchor_counts [B] int64)`` where
    ``assignment[i]`` is pocket atom ``i``'s LOCAL anchor index within its
    own graph and ``M = anchor_counts.sum()``.
    """
    num_pocket = int(pocket_pos.shape[0])
    if int(pocket_batch.shape[0]) != num_pocket:
        raise TensorContractError("[anchors] pocket_batch must match pocket rows")
    if num_pocket:
        observed = int(pocket_batch.max().item()) + 1
        if observed != num_graphs:
            raise TensorContractError(
                f"[anchors] pocket_batch implies {observed} graphs, expected {num_graphs}"
            )
    positions_all, batch_all, assign_all, counts = [], [], [], []
    for g in range(num_graphs):
        rows = torch.nonzero(pocket_batch == g, as_tuple=False).flatten()
        if rows.numel() == 0:
            raise TensorContractError(
                f"[anchors] graph {g} has an empty pocket (fail-closed)"
            )
        pos, assign = pocket_anchor_candidates(
            pocket_pos.index_select(0, rows),
            pocket_z.index_select(0, rows),
            pocket_charge.index_select(0, rows),
            max_anchors=max_anchors,
            min_separation=min_separation,
        )
        positions_all.append(pos)
        batch_all.append(
            torch.full((pos.shape[0],), g, dtype=torch.int64, device=pos.device)
        )
        assign_all.append(assign)
        counts.append(pos.shape[0])
    # anchor_counts follows the input device: the trainer's seed-stage
    # offsets are derived from it (zeros_like) and combined with
    # device-side anchor indices (S-V2-16 T4 crash site).
    return (
        torch.cat(positions_all),
        torch.cat(batch_all),
        torch.cat(assign_all)
        if num_pocket
        else torch.zeros(0, dtype=torch.int64, device=pocket_pos.device),
        torch.tensor(counts, dtype=torch.int64, device=pocket_pos.device),
    )


def aggregate_anchor_features(
    backbone: BackboneOutput,
    assignment: torch.Tensor,
    anchor_batch: torch.Tensor,
) -> torch.Tensor:
    """Per-anchor mean of the backbone pocket-scalar features.

    The invariant anchor feature ``f_m = mean(s_i : atom i assigned to
    anchor m)`` — a rigid-motion-invariant summary of the anchor's
    chemical neighborhood in the learned representation (#39's
    "geometric/chemical features" consumed by the scoring head).
    """
    num_pocket = backbone.num_pocket
    if assignment.shape != (num_pocket,):
        raise TensorContractError(
            f"[anchors] assignment must be [{num_pocket}], got "
            f"{tuple(assignment.shape)}"
        )
    if assignment.dtype not in (torch.int64, torch.int32):
        raise TensorContractError("[anchors] assignment must be integer")
    num_anchors = int(anchor_batch.shape[0])
    if num_anchors == 0:
        raise TensorContractError("[anchors] no anchor rows (fail-closed)")
    device = backbone.scalar.device
    assignment = assignment.long().to(device)
    anchor_batch = anchor_batch.long().to(device)
    if bool((assignment < 0).any()):
        raise TensorContractError("[anchors] negative assignment indices")
    # Flat anchor row index per pocket atom (offset per graph).
    offsets = torch.zeros(
        (backbone.num_graphs + 1,), dtype=torch.long, device=device
    )
    counts_per_graph = torch.bincount(
        anchor_batch, minlength=backbone.num_graphs
    ).to(torch.long)
    offsets[1:] = torch.cumsum(counts_per_graph, dim=0)
    pocket_batch = backbone.node_batch[:num_pocket].long()
    flat = offsets[pocket_batch] + assignment  # [N_p] global anchor row
    if bool((flat >= num_anchors).any()) or bool((flat < 0).any()):
        raise TensorContractError(
            "[anchors] assignment references an anchor outside its graph "
            "(fail-closed cross-field validation)"
        )
    # Cross-graph consistency: every atom's flat anchor must belong to the
    # atom's own graph (an assignment crossing graphs is a wiring error).
    if bool((anchor_batch.index_select(0, flat) != pocket_batch).any()):
        raise TensorContractError(
            "[anchors] a pocket atom is assigned to an anchor of another "
            "graph (fail-closed cross-field validation)"
        )
    dim = backbone.scalar.size(1)
    features = torch.zeros(
        (num_anchors, dim), dtype=backbone.scalar.dtype, device=device
    )
    counts = torch.zeros(
        (num_anchors,), dtype=backbone.scalar.dtype, device=device
    )
    features.index_add_(0, flat, backbone.pocket_scalar)
    counts.index_add_(
        0, flat, torch.ones(num_pocket, dtype=backbone.scalar.dtype, device=device)
    )
    if bool((counts == 0).any()):
        raise TensorContractError(
            "[anchors] an anchor captured no pocket atoms (fail-closed)"
        )
    return features / counts.unsqueeze(-1)


# ---------------------------------------------------------------------------
# The seed/anchor output and head (#39 component 2, policy-side)
# ---------------------------------------------------------------------------

@dataclass
class SeedAnchorOutput:
    """The explicit initial action ``a_seed = (B_0, h_0)`` (#39).

    The probability measure is a rigorously factorized conditional —
    ``P(a_seed | pocket) = P(anchor | pocket) * P(seed | anchor, pocket)``
    — with both stages strictly categorical and dimensionless, so the
    seed action joins the GFlowNet forward log-probability exactly like
    every other action component (constitution §12.3: the forward
    probability includes the seed action).

    Fields (anchor stage, flat per-graph rows like the #22 handle head):
        anchor_logits / anchor_log_probs / anchor_probs: ``[M]`` — the
            per-graph normalized law over that graph's anchor candidates.
        anchor_graph: ``[M]`` int64; anchor_counts: ``[B]`` int64.
        selected_anchor_row: ``[B]`` int64 flat row of the selection (or
            ``-1`` for a distribution-only pass).
        anchor_selected_log_prob: ``[B]`` — the TB forward contribution
            of the anchor component.

    Fields (seed stage, dense per graph):
        seed_logits / seed_log_probs / seed_probs: ``[B, K]`` — the law
            over the seed-eligible catalog entries conditioned on the
            selected anchor.
        selected_seed: ``[B]`` int64 (``-1`` for a distribution-only
            pass); seed_selected_log_prob: ``[B]``.
        retrieved_embedding: ``[B, embed_dim]`` — the tied catalog
            embedding of the selected seed (the t=0 conditioning surface).
    """

    anchor_logits: torch.Tensor  # [M]
    anchor_log_probs: torch.Tensor  # [M]
    anchor_probs: torch.Tensor  # [M]
    anchor_graph: torch.Tensor  # [M] int64
    anchor_counts: torch.Tensor  # [B] int64
    selected_anchor_row: torch.Tensor  # [B] int64
    anchor_selected_log_prob: torch.Tensor  # [B]
    seed_logits: torch.Tensor  # [B, K]
    seed_log_probs: torch.Tensor  # [B, K]
    seed_probs: torch.Tensor  # [B, K]
    selected_seed: torch.Tensor  # [B] int64
    seed_selected_log_prob: torch.Tensor  # [B]
    retrieved_embedding: torch.Tensor  # [B, embed_dim]
    num_graphs: int

    @property
    def catalog_size(self) -> int:
        return int(self.seed_log_probs.size(1))

    def step_log_prob(self) -> torch.Tensor:
        """The TB forward log-probability of the seed action ``[B]``.

        The seed step's backward term is the zero-leaf special case the
        TB engine already accepts (``num_leaves = 0`` semantics of a seed
        step — see ``AssemblyEnvironment.reset``'s docstring and
        ``MDP_STATE_SPACE.md``).
        """
        return self.anchor_selected_log_prob + self.seed_selected_log_prob


class SeedAnchorHead(nn.Module):
    """The learnable seed-fragment and pocket-anchor selection (#39).

    Two strictly categorical stages over the decoder-facing pocket
    context (the #23 addendum's explicitly allowed consumer):

    * **anchor stage** — ``P(h_0 | pocket)``: a scaled dot-product score
      between a query projected from the backbone's invariant
      ``pocket_context`` ``[B, d]`` and keys projected from the invariant
      per-anchor features ``f_m`` (means of the backbone pocket scalars
      over each anchor's assigned atoms), normalized per graph over that
      graph's candidates with the pinned logit-mask machinery;
    * **seed stage** — ``P(B_0 | h_0, pocket)``: a query projected from
      ``[pocket_context ; selected-anchor feature]`` scored against the
      **tied catalog embedding table** (the single authoritative synthon
      representation, shared with :class:`SynthonRetrievalHead` — the
      same module instance, parameters counted once in any assembly),
      masked to the seed-eligible entries.

    Invalid seed/anchor combinations are masked exactly like every other
    categorical: the anchor stage carries a per-row legality mask, and
    the seed stage's mask is the composition of the seed-eligibility mask
    with the selected anchor's combination mask (caller-supplied; the
    class-political note is that class-level masks are prefilters, the
    environment enforces exact chemistry).

    Args:
        context_dim: width of the consumed pocket context and anchor
            features (the frozen backbone width, 256).
        embed_dim: width of the tied catalog embeddings (64).
        catalog_size: number of catalog entries (50,000).
        catalog_embeddings: an optional externally-owned tied table
            (:class:`nn.Embedding(catalog_size, embed_dim)`); when the
            assembly wires the synthon head's table here, the two heads
            share one representation and the parameters register once.
    """

    def __init__(
        self,
        context_dim: int = DECODER_D_MODEL,
        embed_dim: int = SYNTON_EMBED_DIM,
        catalog_size: int = SYNTON_CATALOG_SIZE,
        catalog_embeddings: Optional[nn.Embedding] = None,
    ) -> None:
        super().__init__()
        context_dim = int(context_dim)
        embed_dim = int(embed_dim)
        catalog_size = int(catalog_size)
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if catalog_size < 1:
            raise ValueError(f"catalog_size must be >= 1, got {catalog_size}")
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.catalog_size = catalog_size
        self.anchor_query = nn.Linear(context_dim, context_dim)
        self.anchor_key = nn.Linear(context_dim, context_dim)
        self.seed_query = nn.Linear(2 * context_dim, embed_dim)
        self._catalog_embeddings_shared = catalog_embeddings is not None
        if catalog_embeddings is None:
            self.catalog_embeddings = nn.Embedding(catalog_size, embed_dim)
        else:
            if (
                catalog_embeddings.num_embeddings != catalog_size
                or catalog_embeddings.embedding_dim != embed_dim
            ):
                raise ValueError(
                    "the tied catalog table must be "
                    f"nn.Embedding({catalog_size}, {embed_dim}), got "
                    f"({catalog_embeddings.num_embeddings}, "
                    f"{catalog_embeddings.embedding_dim})"
                )
            self.catalog_embeddings = catalog_embeddings

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, embed_dim={self.embed_dim}, "
            f"catalog_size={self.catalog_size}"
        )

    def audit(self) -> Dict[str, Any]:
        parameters = list(self.parameters())
        table_params = self.catalog_size * self.embed_dim
        return {
            "module": type(self).__name__,
            "context_dim": self.context_dim,
            "embed_dim": self.embed_dim,
            "catalog_size": self.catalog_size,
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "own_parameters_excluding_catalog_table": sum(p.numel() for p in parameters)
            - table_params,
            "catalog_table_parameters": table_params,
            "tied_catalog_table_shared": self._catalog_embeddings_shared,
            "factorization": "P(anchor|pocket) * P(seed|anchor,pocket)",
            "tb_probability_measures": {
                "anchor": "categorical_per_pocket",
                "seed": "categorical_K_anchor_conditional",
            },
            "deterministic_in_eval": True,
        }

    # -- stage 1: the anchor law ----------------------------------------------

    def forward_anchor(
        self,
        pocket_context: torch.Tensor,
        anchor_features: torch.Tensor,
        anchor_graph: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None,
        selected_anchor_row: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """``P(h_0 | pocket)`` over each graph's anchor candidates.

        Parameters:
            pocket_context: ``[B, context_dim]`` — the backbone's
            invariant pocket summary.
            anchor_features: ``[M, context_dim]`` — the per-anchor mean
            pocket-scalar features (:func:`aggregate_anchor_features`).
            anchor_graph: ``[M]`` int64 graph assignment.
            anchor_mask: ``[M]`` bool legality (``True`` = illegal).
            selected_anchor_row: ``[B]`` int64 flat row of the
            supervised/sampled anchor (``-1`` = distribution-only).
        """
        pocket_context = _validate_context(
            pocket_context,
            self.context_dim,
            self.anchor_query.weight.dtype,
            "seed-anchor",
        )
        num_graphs = pocket_context.size(0)
        num_anchors = int(anchor_features.shape[0])
        if anchor_features.shape != (num_anchors, self.context_dim):
            raise TensorContractError(
                f"[seed-anchor] anchor_features must be [M, {self.context_dim}], "
                f"got {tuple(anchor_features.shape)}"
            )
        if anchor_features.dtype != pocket_context.dtype:
            raise TensorContractError(
                "[seed-anchor] anchor_features dtype must match the pocket context"
            )
        if not bool(torch.isfinite(anchor_features).all()):
            raise TensorContractError(
                "[seed-anchor] anchor_features contains non-finite values"
            )
        if anchor_graph.dtype not in (torch.int64, torch.int32) or anchor_graph.shape != (
            num_anchors,
        ):
            raise TensorContractError("[seed-anchor] anchor_graph must be [M] int")
        anchor_graph = anchor_graph.long().to(pocket_context.device)
        if bool((anchor_graph < 0).any()) or bool((anchor_graph >= num_graphs).any()):
            raise TensorContractError("[seed-anchor] anchor_graph out of range")
        if anchor_mask is None:
            anchor_mask = torch.zeros(
                num_anchors, dtype=torch.bool, device=pocket_context.device
            )
        else:
            if anchor_mask.dtype != torch.bool or anchor_mask.shape != (num_anchors,):
                raise TensorContractError("[seed-anchor] anchor_mask must be [M] bool")
            anchor_mask = anchor_mask.to(pocket_context.device)
        selected_anchor_row = _validate_index_vector(
            selected_anchor_row,
            num_graphs,
            num_anchors,
            "seed-anchor",
            "selected_anchor_row",
            allow_sentinel=-1,
        )
        if selected_anchor_row is None:
            selected_anchor_row = torch.full(
                (num_graphs,), -1, dtype=torch.int64, device=pocket_context.device
            )
        else:
            selected_anchor_row = selected_anchor_row.to(pocket_context.device)

        # Per-graph legality: a graph with ALL anchors masked has no valid
        # seed placement — fail closed (never a silent uniform).
        for g in range(num_graphs):
            rows = anchor_graph == g
            if rows.numel() == 0:
                raise TensorContractError(
                    f"[seed-anchor] graph {g} has no anchor candidates "
                    "(fail-closed: generate candidates first)"
                )
            if bool(anchor_mask[rows].all()):
                raise TensorContractError(
                    f"[seed-anchor] graph {g} has no legal anchor candidate "
                    "(fail-closed)"
                )

        query = self.anchor_query(pocket_context)  # [B, d]
        key = self.anchor_key(anchor_features)  # [M, d]
        scores = (
            query.index_select(0, anchor_graph) * key
        ).sum(dim=-1) / math.sqrt(self.context_dim)  # [M]
        # Flat single-action rows cannot route through apply_logit_mask's
        # per-row no-legal-action guard (an individually masked anchor is
        # legitimate — its graph has other legal candidates, checked
        # above), so the pinned mask constant is applied here with the
        # same fail-closed finiteness / poisoned-logit protections
        # inlined (#10 D-V2-014 semantics).
        if not bool(torch.isfinite(scores).all()):
            raise HardwareError(
                "[seed-anchor] anchor scores contain non-finite values (fail-closed)"
            )
        legal_scores = scores[~anchor_mask]
        if bool((legal_scores <= LOGIT_MASK_LEGAL_ABOVE).any()):
            raise HardwareError(
                "[seed-anchor] a legal anchor score is at or below "
                f"LOGIT_MASK_LEGAL_ABOVE ({LOGIT_MASK_LEGAL_ABOVE}): poisoned "
                "scores would compete with the mask fill (fail-closed)"
            )
        scores = scores.masked_fill(anchor_mask, LOGIT_MASK_FILL)
        # Segment log-softmax per graph (exact per-graph normalization,
        # finite masked entries by construction).
        max_per_graph = torch.full(
            (num_graphs,), -float("inf"), dtype=scores.dtype, device=scores.device
        )
        max_per_graph.scatter_reduce_(
            0, anchor_graph, scores, reduce="amax", include_self=True
        )
        shifted = scores - max_per_graph.index_select(0, anchor_graph)
        exp = shifted.exp().masked_fill(anchor_mask, 0.0)
        denom = torch.zeros(
            (num_graphs,), dtype=scores.dtype, device=scores.device
        )
        denom.index_add_(0, anchor_graph, exp)
        log_probs = shifted - denom.index_select(0, anchor_graph).log()
        probs = log_probs.exp()

        selected_log_prob = torch.full(
            (num_graphs,),
            float("nan"),
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        has_selection = selected_anchor_row >= 0
        if bool(has_selection.any()):
            rows = torch.nonzero(has_selection, as_tuple=False).flatten()
            selected_log_prob = selected_log_prob.index_copy(
                0,
                rows,
                log_probs.index_select(0, selected_anchor_row[rows]),
            )
        return {
            "logits": scores,
            "log_probs": log_probs,
            "probs": probs,
            "anchor_graph": anchor_graph,
            "selected_anchor_row": selected_anchor_row,
            "selected_log_prob": selected_log_prob,
        }

    # -- stage 2: the seed law ---------------------------------------------------

    def forward_seed(
        self,
        pocket_context: torch.Tensor,
        selected_anchor_feature: torch.Tensor,
        seed_mask: Optional[torch.Tensor] = None,
        selected_seed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """``P(B_0 | h_0, pocket)`` over the seed-eligible catalog entries.

        Parameters:
            pocket_context: ``[B, context_dim]``.
            selected_anchor_feature: ``[B, context_dim]`` — the feature of
                the selected anchor (conditions the seed choice).
            seed_mask: ``[B, K]`` bool legality — the composition of the
                seed-eligibility mask with the selected anchor's
                combination mask.
            selected_seed: ``[B]`` int64 catalog index (``-1`` =
                distribution-only).
        """
        pocket_context = _validate_context(
            pocket_context,
            self.context_dim,
            self.seed_query.weight.dtype,
            "seed-anchor",
        )
        num_graphs = pocket_context.size(0)
        if (
            not isinstance(selected_anchor_feature, torch.Tensor)
            or selected_anchor_feature.shape != (num_graphs, self.context_dim)
            or selected_anchor_feature.dtype != pocket_context.dtype
        ):
            raise TensorContractError(
                f"[seed-anchor] selected_anchor_feature must be "
                f"[{num_graphs}, {self.context_dim}] with the context dtype"
            )
        if not bool(torch.isfinite(selected_anchor_feature).all()):
            raise TensorContractError(
                "[seed-anchor] selected_anchor_feature contains non-finite values"
            )
        if seed_mask is None:
            seed_mask = torch.zeros(
                (num_graphs, self.catalog_size),
                dtype=torch.bool,
                device=pocket_context.device,
            )
        else:
            if (
                seed_mask.dtype != torch.bool
                or tuple(seed_mask.shape) != (num_graphs, self.catalog_size)
            ):
                raise TensorContractError(
                    f"[seed-anchor] seed_mask must be "
                    f"[{num_graphs}, {self.catalog_size}] bool"
                )
            seed_mask = seed_mask.to(pocket_context.device)
        selected_seed = _validate_index_vector(
            selected_seed,
            num_graphs,
            self.catalog_size,
            "seed-anchor",
            "selected_seed",
            allow_sentinel=-1,
        )
        if selected_seed is None:
            selected_seed = torch.full(
                (num_graphs,), -1, dtype=torch.int64, device=pocket_context.device
            )
        else:
            selected_seed = selected_seed.to(pocket_context.device)

        query = self.seed_query(
            torch.cat((pocket_context, selected_anchor_feature), dim=-1)
        )  # [B, embed_dim]
        table = self.catalog_embeddings.weight  # [K, embed_dim]
        logits = (query @ table.t()) * (1.0 / math.sqrt(self.embed_dim))  # [B, K]
        logits = apply_logit_mask(logits, seed_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        selected_log_prob = torch.full(
            (num_graphs,), float("nan"), dtype=log_probs.dtype, device=log_probs.device
        )
        retrieved = torch.zeros(
            (num_graphs, self.embed_dim),
            dtype=pocket_context.dtype,
            device=pocket_context.device,
        )
        has_selection = selected_seed >= 0
        if bool(has_selection.any()):
            rows = torch.nonzero(has_selection, as_tuple=False).flatten()
            selected_log_prob = selected_log_prob.index_copy(
                0,
                rows,
                log_probs.gather(1, selected_seed[rows].unsqueeze(-1)).squeeze(-1),
            )
            retrieved = retrieved.index_copy(
                0, rows, self.catalog_embeddings(selected_seed[rows].long())
            )
        return {
            "logits": logits,
            "log_probs": log_probs,
            "probs": probs,
            "selected_seed": selected_seed,
            "selected_log_prob": selected_log_prob,
            "retrieved_embedding": retrieved,
        }

    # -- the teacher-forced one-shot ----------------------------------------------

    def forward(
        self,
        pocket_context: torch.Tensor,
        anchor_features: torch.Tensor,
        anchor_graph: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None,
        selected_anchor_row: Optional[torch.Tensor] = None,
        seed_mask: Optional[torch.Tensor] = None,
        selected_seed: Optional[torch.Tensor] = None,
    ) -> SeedAnchorOutput:
        """Teacher-forced evaluation of the full seed action.

        The selected anchor conditions the seed stage; the selected
        anchor's *feature* is gathered from ``anchor_features``.
        """
        anchor = self.forward_anchor(
            pocket_context,
            anchor_features,
            anchor_graph,
            anchor_mask,
            selected_anchor_row,
        )
        num_graphs = pocket_context.size(0)
        device = pocket_context.device
        sel_rows = anchor["selected_anchor_row"]
        if bool((sel_rows >= 0).all()):
            selected_anchor_feature = anchor_features.index_select(0, sel_rows)
        else:
            selected_anchor_feature = torch.zeros(
                (num_graphs, self.context_dim),
                dtype=pocket_context.dtype,
                device=device,
            )
        seed = self.forward_seed(
            pocket_context,
            selected_anchor_feature,
            seed_mask,
            selected_seed,
        )
        counts = torch.bincount(
            anchor["anchor_graph"], minlength=num_graphs
        ).to(torch.int64)
        return SeedAnchorOutput(
            anchor_logits=anchor["logits"],
            anchor_log_probs=anchor["log_probs"],
            anchor_probs=anchor["probs"],
            anchor_graph=anchor["anchor_graph"],
            anchor_counts=counts,
            selected_anchor_row=sel_rows,
            anchor_selected_log_prob=anchor["selected_log_prob"],
            seed_logits=seed["logits"],
            seed_log_probs=seed["log_probs"],
            seed_probs=seed["probs"],
            selected_seed=seed["selected_seed"],
            seed_selected_log_prob=seed["selected_log_prob"],
            retrieved_embedding=seed["retrieved_embedding"],
            num_graphs=num_graphs,
        )


# ---------------------------------------------------------------------------
# The assembly (#25 MODEL-05)
# ---------------------------------------------------------------------------

@dataclass
class PolicyStateOutput:
    """One growth-step encoding pass of :class:`SynTreePolicy`.

    Fields:
        backbone: the :class:`~syntree.models.bipartite.BackboneOutput`
            (union-graph representations; ``pocket_context`` feeds the
            seed/anchor stage, the pooled surfaces feed the decoder).
        handle: the :class:`~syntree.models.handle_head.HandleSelectionOutput`
            (the #22 law over the active handle pool).
        decoder: the :class:`DecoderOutput` (the refined decision context
            — the #24 heads' input surface).
    """

    backbone: BackboneOutput
    handle: HandleSelectionOutput
    decoder: DecoderOutput

    @property
    def context(self) -> torch.Tensor:
        """The per-graph decision context ``[B, d_model]``."""
        return self.decoder.context

    @property
    def num_graphs(self) -> int:
        return self.decoder.num_graphs


class SynTreePolicy(nn.Module):
    """The complete 3D-SynTree V2 policy assembly (#25 MODEL-05).

    One registered module containing the full frozen production stack —
    backbone (#21, activation checkpointing ON), handle-selection head
    (#22), stacked cross-attention decoder (#23), the decoupled
    hierarchical heads (#24), and the seed/anchor selection component of
    #39 — with a single authoritative audit (constitution #45 §7.3:
    parameter counts, trainability, dimensions, gradient connectivity,
    the tied catalog embeddings, the mask interfaces, and model memory).

    The evaluation protocol mirrors the MDP: **stage 0** (once, at
    ``t = 0``) scores the explicit seed action ``(B_0, h_0)`` via
    :meth:`forward_seed_anchor` — no hidden hotspot heuristic decides the
    production starting state; **stage 1** scores termination
    (:meth:`forward_stop`); **stage 2** scores the continuation heads for
    the CONTINUE rows (:meth:`forward_continue` /
    :meth:`forward_state` + the stage-2 heads).  Every stage contributes
    a pure dimensionless categorical log-probability, and the total
    trajectory forward probability is their sum (constitution §12.3).

    Parameter budget (decision D-V2-019): the frozen per-issue profiles
    sum to **10,880,719 parameters = 43.5 MB FP32** — below the 50 MB
    acceptance bound with margin. The issue's ``~4.8M`` estimate predates
    the frozen profiles and is superseded by them (see the module-level
    reconciliation note).
    """

    def __init__(
        self,
        catalog_size: int = SYNTON_CATALOG_SIZE,
        max_pocket_anchors: int = MAX_POCKET_ANCHORS,
        anchor_min_separation: float = ANCHOR_MIN_SEPARATION,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        catalog_size = int(catalog_size)
        if catalog_size < 1:
            raise ValueError(f"catalog_size must be >= 1, got {catalog_size}")
        if int(max_pocket_anchors) < 1:
            raise ValueError("max_pocket_anchors must be >= 1")
        if not (float(anchor_min_separation) > 0.0):
            raise ValueError("anchor_min_separation must be > 0")
        self.catalog_size = catalog_size
        self.max_pocket_anchors = int(max_pocket_anchors)
        self.anchor_min_separation = float(anchor_min_separation)

        # The frozen production stack (one of each — no duplicates, no
        # compatibility shims; clean-room mandate #45 §24).
        self.backbone = BipartitePaiNN(
            activation_checkpointing=activation_checkpointing
        )
        self.handle_head = HandleSelectionHead()
        self.decoder = CrossAttentionDecoder()
        self.heads = HierarchicalPolicyHeads(catalog_size=catalog_size)
        # The #39 seed/anchor component, wired to the SAME tied catalog
        # table the retrieval head owns (one synthon representation,
        # parameters registered once).
        self.seed_anchor_head = SeedAnchorHead(
            catalog_size=catalog_size,
            catalog_embeddings=self.heads.synthon_head.catalog_embeddings,
        )

    def extra_repr(self) -> str:
        return (
            f"catalog_size={self.catalog_size}, "
            f"max_pocket_anchors={self.max_pocket_anchors}, "
            f"anchor_min_separation={self.anchor_min_separation}"
        )

    # -- the anchor candidate machinery (deterministic, #39) -----------------

    def anchor_candidates(self, pocket_pos, pocket_z, pocket_charge, pocket_batch, num_graphs):
        """Deterministic anchor decomposition of the batched pockets."""
        return batched_pocket_anchor_candidates(
            pocket_pos,
            pocket_z,
            pocket_charge,
            pocket_batch,
            num_graphs,
            max_anchors=self.max_pocket_anchors,
            min_separation=self.anchor_min_separation,
        )

    def anchor_features(self, backbone_output: BackboneOutput, assignment, anchor_batch):
        """Invariant per-anchor features from a backbone encoding."""
        return aggregate_anchor_features(backbone_output, assignment, anchor_batch)

    # -- stage 0: the explicit seed action (#39) --------------------------------

    def forward_seed_anchor(
        self,
        backbone_output: BackboneOutput,
        anchor_features: torch.Tensor,
        anchor_graph: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None,
        selected_anchor_row: Optional[torch.Tensor] = None,
        seed_mask: Optional[torch.Tensor] = None,
        selected_seed: Optional[torch.Tensor] = None,
    ) -> SeedAnchorOutput:
        """Score the seed action ``a_seed = (B_0, h_0)`` (stage 0)."""
        return self.seed_anchor_head(
            backbone_output.pocket_context,
            anchor_features,
            anchor_graph,
            anchor_mask,
            selected_anchor_row,
            seed_mask,
            selected_seed,
        )

    # -- growth-step encoding ------------------------------------------------------

    def forward_state(
        self,
        state: Any,
        handle_descriptors: torch.Tensor,
        handle_pos: torch.Tensor,
        handle_graph: torch.Tensor,
        handle_atom: torch.Tensor,
        handle_flags: Optional[torch.Tensor] = None,
    ) -> PolicyStateOutput:
        """Encode a growth state: backbone -> handle head -> decoder.

        Consumes the canonical V2 state batch (the
        :func:`~syntree.core.schema.collate_state_batch` layout) and the
        active-handle pool in the #22 contract; returns the auditable
        triple whose ``context`` feeds the #24 heads.
        """
        backbone_output = self.backbone(state, handle_flags=handle_flags)
        handle_output = self.handle_head(
            backbone_output,
            handle_descriptors,
            handle_pos,
            handle_graph,
            handle_atom,
        )
        decoder_output = self.decoder.forward_state(
            backbone_output, handle_output
        )
        return PolicyStateOutput(
            backbone=backbone_output,
            handle=handle_output,
            decoder=decoder_output,
        )

    # -- stages 1 and 2 (delegation to the #24 container) -----------------------

    def forward_stop(
        self, context: torch.Tensor, force_stop: Optional[torch.Tensor] = None
    ) -> StopOutput:
        """Stage 1: the binary termination law."""
        return self.heads.forward_stop(context, force_stop)

    def forward_continue(
        self,
        context: torch.Tensor,
        selected_reaction: torch.Tensor,
        selected_synthon: torch.Tensor,
        selected_parity: torch.Tensor,
        selected_torsion_bin: torch.Tensor,
        stereocenter_required: torch.Tensor,
        family_mask: torch.Tensor,
        reaction_mask: torch.Tensor,
        catalog_handle_classes: Optional[torch.Tensor] = None,
        bin_mask: Optional[torch.Tensor] = None,
    ) -> ContinueOutput:
        """Stage 2: teacher-forced continuation (CONTINUE rows only)."""
        return self.heads.forward_continue(
            context,
            selected_reaction,
            selected_synthon,
            selected_parity,
            selected_torsion_bin,
            stereocenter_required,
            family_mask,
            reaction_mask,
            catalog_handle_classes,
            bin_mask,
        )

    # -- audit ---------------------------------------------------------------

    def audit(self) -> Dict[str, Any]:
        """The complete assembly audit (constitution #45 §7.3)."""
        parameters = list(self.parameters())
        num_parameters = sum(p.numel() for p in parameters)
        bytes_fp32 = num_parameters * 4
        # The #21 backbone predates the audit() convention — its audit is
        # computed inline here (dimensions, counts, checkpointing flag).
        backbone_params = list(self.backbone.parameters())
        backbone_audit = {
            "module": type(self.backbone).__name__,
            "hidden_dim": self.backbone.hidden_dim,
            "num_layers": self.backbone.num_layers,
            "num_radial": self.backbone.num_radial,
            "r_cut": self.backbone.r_cut,
            "activation_checkpointing": bool(
                self.backbone.activation_checkpointing
            ),
            "num_parameters": sum(p.numel() for p in backbone_params),
            "num_trainable_parameters": sum(
                p.numel() for p in backbone_params if p.requires_grad
            ),
        }
        component_audit = {
            "backbone": backbone_audit,
            "handle_head": self.handle_head.audit(),
            "decoder": self.decoder.audit(),
            "hierarchical_heads": self.heads.audit(),
            "seed_anchor_head": self.seed_anchor_head.audit(),
        }
        return {
            "module": type(self).__name__,
            "catalog_size": self.catalog_size,
            "max_pocket_anchors": self.max_pocket_anchors,
            "anchor_min_separation": self.anchor_min_separation,
            "activation_checkpointing": bool(
                self.backbone.activation_checkpointing
            ),
            "components": {
                name: {
                    "module": audit["module"],
                    # the seed head's count EXCLUDES the shared tied table
                    # (owned by the retrieval head, counted once globally)
                    "num_parameters": audit.get(
                        "own_parameters_excluding_catalog_table",
                        audit["num_parameters"],
                    ),
                    "num_trainable_parameters": audit.get(
                        "num_trainable_parameters", audit["num_parameters"]
                    ),
                }
                for name, audit in component_audit.items()
            },
            "num_parameters": num_parameters,
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "parameter_memory_bytes_fp32": bytes_fp32,
            "parameter_memory_mb_fp32": bytes_fp32 / (1000 * 1000),
            "parameter_memory_under_50_mb": bytes_fp32 < 50 * 1000 * 1000,
            "tied_catalog_embeddings": {
                "shared_by": ["synthon_head", "seed_anchor_head"],
                "num_embeddings": self.catalog_size,
                "embedding_dim": SYNTON_EMBED_DIM,
                "parameters_counted_once": self.heads.synthon_head.catalog_embeddings
                is self.seed_anchor_head.catalog_embeddings,
            },
            "mask_interfaces": {
                "stage0_anchor": "[M] bool (per-row legality)",
                "stage0_seed": "[B, K] bool (eligibility + combination)",
                "stage1_stop": "forced rows via [B] bool",
                "stage2": "family [B,9] / reaction [B,27] / synthon [B,K] / bins [B,36]",
            },
            "action_contract": {
                "stage0_seed_action": "P(anchor|pocket)*P(seed|anchor,pocket)",
                "stage1": "P(STOP|s) binary",
                "stage2": "P(f)P(r|f)P(B|r)P(c)P(k) + #22 P(h|s)",
                "all_categorical": True,
                "von_mises_in_tb": False,
            },
            "frozen_profile_note": (
                "10,880,719 parameters = 43.5 MB FP32 (< 50 MB acceptance "
                "bound); the issue's ~4.8M estimate is superseded by the "
                "closed per-issue frozen profiles (D-V2-019)"
            ),
        }


#: The frozen production parameter total of :class:`SynTreePolicy`
#: (version-controlled arithmetic; #45 §5.2 — frozen dimensions).
SYNTREE_POLICY_PARAMETER_PROFILE = (
    5_999_104  # BipartitePaiNN (#21 frozen profile)
    + 224_896  # HandleSelectionHead (#22 frozen profile)
    + 1_219_328  # CrossAttentionDecoder (#23 frozen profile)
    + HIERARCHICAL_HEADS_PARAMETER_PROFILE  # #24 (incl. the 3.2M tied table)
    + (256 * 256 + 256) * 2  # seed/anchor query + key
    + (512 * 64 + 64)  # seed/anchor seed query
)
