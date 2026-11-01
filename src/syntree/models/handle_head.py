"""Agency-enabled handle selection: ``P(handle_t | s_t)`` over the active pool.

Clean-room V2 implementation of the handle-selection policy head (issue #22
MODEL-02; constitution #45 §7.3; decision D-V2-010 in
``docs/agent/DECISIONS.md``).

**Why this head exists (the #22 motivation — legacy "Vulnerability 3").**
The pre-V2 environment hardcoded handle selection by sorting atom indices
and picking ``handles[0]``.  When an intermediate carries symmetric or
competing handles facing different directions — one toward an open cavity,
one toward a pocket wall — the agent had *no spatial agency* over where to
grow.  V2 makes the handle an explicit, scored component of the action
tuple::

    a_t = (handle_t, r_t, B_t, c_t, phi_t)

which the discrete Assembly MDP of :mod:`syntree.engine.environment`
already realises as ``AssemblyAction(handle_atom_idx, reaction_id,
block_id, chiral, torsion_bin)`` — the handle is the *first* component,
and the environment exposes the live menu through
``AssemblyEnvironment.active_handles()``.  This head supplies the missing
``P(handle_t | s_t)`` that scores exactly that menu.

**Inputs.**  :meth:`HandleSelectionHead.forward` consumes:

* a :class:`~syntree.models.bipartite.BackboneOutput` — the
  contextualized union representations of the *single* authoritative
  ``BipartitePaiNN`` encoder (#21; a second pocket encoder is forbidden,
  #45 §7.2).  The head uses the ligand node representations, the pooled
  pocket context, and the union node metadata;
* the active handle pool, flattened across the batch:

  * ``handle_descriptors`` ``[H, 64]`` — the invariant chemical
    descriptors of each active handle (one row per handle; the width is
    the frozen schema constant
    :data:`~syntree.core.schema.HANDLE_FEATURE_DIM`).  Legacy 67-D
    "Franken" descriptors are rejected fail-closed (#6);
  * ``handle_pos`` ``[H, 3]`` — the reacting atom position of each handle
    in the pocket frame, supplied *separately* from the descriptors
    (constitution #45 §7.1: chemistry and Cartesian coordinates are never
    concatenated into one feature vector);
  * ``handle_graph`` ``[H]`` int64 — graph assignment in ``[0, B)``;
  * ``handle_atom`` ``[H]`` int64 — the attachment atom's index within the
    ligand sub-graph (union node ``num_pocket + handle_atom``).  In the V2
    integration this is exactly the ``handle_atom_idx`` of the
    environment's ``active_handles()`` menu: the environment's molecule
    holds only ligand (heavy) atoms, in the same order the policy-side
    tensor state mirrors (documented contract; verified by test).

**Architecture (decision D-V2-010).**  Per handle ``h`` with attachment
atom ``i`` and graph ``g``:

1. *Invariant handle embedding* ``e_h`` (width ``d_head``)::

       e_h = MLP_desc(u_h)                 [64 -> d_head]  invariant chemistry
           + W_atom  s_i                   [d   -> d_head]  backbone scalar of the atom
           + W_geom  RBF(||x_h - c_g||)    [R   -> d_head]  centroid radial geometry
           + W_dir   <v_i, x_h - c_g>      [d   -> d_head]  equivariant directional invariant

   where ``s_i`` / ``v_i`` are the backbone scalar/vector features of the
   attachment atom, ``c_g`` is the pocket centroid of graph ``g``, and the
   radial basis is the raw (envelope-free) :class:`GaussianRadialBasis`
   over ``[0, HANDLE_GEOM_CUTOFF]`` — the cosine cutoff envelope for *all*
   spatial RBF projections is owned by #9 MATH-03 (decision D-V2-007) and
   is deliberately not applied here.

2. *Per-graph query* — the contextualized ligand summary conditioned on
   the pocket::

       q_g = W_lig  mean(s_j : j ligand of g)  +  W_poc  pocket_context_g

3. *Stage 1 — cross-attention over the handle pool* (the issue's
   "cross-attention query over handle invariant descriptors and Cartesian
   positions relative to pocket centroids").  With one query per graph,
   the scaled dot-product attention over that graph's handle keys::

       alpha_h = softmax_{h' in g(h)}( q_g . e_h' / sqrt(d_head) )

   is a valid probability distribution over the graph's active handle set
   by construction; the attention-weighted value pooling::

       c_g = sum_{h in g} alpha_h * W_val e_h

   gives each handle's later score access to the competing handles of the
   same graph.

4. *Stage 2 — additive (Bahdanau-style) scoring*::

       logit_h = w^T tanh(W_q q_{g(h)} + W_k e_h + W_c c_{g(h)}) / sqrt(d_head)

5. *Normalization over the actually valid alternatives* (#45 §7.3 / §8.4):
   the final distribution is a per-graph ``log_softmax`` over exactly that
   graph's active handle rows::

       log P(handle_h | s_g) = logit_h - logsumexp_{h' in g(h)} logit_h'

   Graphs with zero active handles contribute no rows (there is no handle
   action to score; only STOP is legal in the environment) and receive a
   zero entropy.

**SE(3) invariance argument (pinned by tests).**  The backbone scalars and
pooled contexts are invariant; the descriptors are invariant by contract;
``||x_h - c_g||`` is invariant (``c_g`` is a mean of pocket positions, so
it moves rigidly with the input frame); and ``<v_i, x_h - c_g>`` pairs the
equivariant vector feature with the equivariant centroid-relative offset,
which is invariant because ``(R v) . (R r) = v . r``.  Every downstream
operation is an inner product or MLP of invariant features, so the logits,
log-probabilities, attention weights, and entropies are invariant under
rigid motions of the whole (pocket + ligand + handle) system.  Positions
enter only through the safe norm and the equivariant dot product, so the
pass is differentiable with respect to coordinates without any ``0/0``
singularity (#7 primitives).

**Numerical policy.**  The per-group ``log_softmax`` is computed through
the numerically stable max-shifted segment form.  No NaN masking, clamping
of logits, or silent fallbacks exist anywhere in this module: poisoned
inputs fail closed at validation, and non-finite *outputs* (which can only
come from corrupted parameters) fail closed after the forward pass
(#45 §16 — fail loudly).

**Precision policy (decisions D-V2-005 / D-V2-008).**  Parameters default
to pure FP32 (the production path on Colab/Kaggle Tesla T4).
``float64`` is accepted *only* for local numerical verification of the
invariance properties at tight tolerance; the input dtype (including the
consumed ``BackboneOutput`` tensors) must match the module parameter dtype
exactly.  FP16/BF16 and every other dtype are rejected fail-closed.

**Scope boundary.**  This head produces ``P(handle_t | s_t)`` only.  The
remaining action components (reaction family/reaction, synthon, stereo
parity, torsion bin, STOP) belong to #23/#24, the full policy assembly to
#25, and the descriptor *producer* (the 64-D featurization) to the catalog
issues (#12/#40).  The output additionally exposes the handle embeddings
and the pooled handle context so the #23 decoder can consume the same
handle-side representations without recomputation — one authoritative
embedding, not a second handle encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from syntree.core.schema import (
    HANDLE_FEATURE_DIM,
    LEGACY_HANDLE_FEATURE_DIM,
    TensorContractError,
)
from syntree.models.bipartite import (
    BACKBONE_HIDDEN_DIM,
    BackboneOutput,
    GaussianRadialBasis,
)
from syntree.models.equivariant import safe_norm

__all__ = [
    "HANDLE_HEAD_HIDDEN_DIM",
    "HANDLE_HEAD_NUM_RADIAL",
    "HANDLE_GEOM_CUTOFF",
    "HandleSelectionOutput",
    "HandleSelectionHead",
]

# ---------------------------------------------------------------------------
# Frozen production profile (#45 §5.2: dimensions are frozen in
# version-controlled configuration, never probed from free VRAM).
# ---------------------------------------------------------------------------

#: Internal feature width of the handle head (queries, keys, values).
HANDLE_HEAD_HIDDEN_DIM = 128

#: Number of Gaussian centers of the centroid-distance radial embedding.
HANDLE_HEAD_NUM_RADIAL = 20

#: Upper edge (Angstrom) of the centroid-distance radial basis centers.
#: Pocket extraction uses a 10 A radius around the co-crystallized ligand
#: centroid (#45 §11), so handle-to-pocket-centroid distances live inside
#: roughly [0, 12] A; distances beyond the last center decay smoothly
#: through the Gaussian tail (no envelope, decision D-V2-007 — #9 owns the
#: envelope).
HANDLE_GEOM_CUTOFF = 12.0

#: Input dtypes accepted by the head (D-V2-005 / D-V2-008): FP32
#: production plus FP64 local verification only.
_FLOAT_INPUT_DTYPES = (torch.float32, torch.float64)

#: Exact message raised for a 67-column descriptor tensor (parallel to the
#: schema's legacy ban; pinned by test).
_LEGACY_DESCRIPTOR_MESSAGE = (
    "Legacy 67D Franken-tensors are deprecated: `handle_descriptors` must "
    "be [H, 64] invariant chemical descriptors with the handle's Cartesian "
    "position supplied separately as `handle_pos` [H, 3] (issue #22 "
    "MODEL-02; #45 §7.1 forbids mixing features and coordinates)."
)


# ---------------------------------------------------------------------------
# Segment log-softmax (per-graph normalization over the active handle set)
# ---------------------------------------------------------------------------


def _segment_log_softmax(
    logits: torch.Tensor, index: torch.Tensor, num_segments: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise log-softmax restricted to each segment of ``index``.

    Numerically stable max-shifted form: for row ``h`` of segment ``g``::

        log_softmax_h = logits_h - logsumexp(logits of segment g)

    ``logits`` ``[H]`` must be finite; ``index`` ``[H]`` int64 in
    ``[0, num_segments)``.  Segments without rows are never read by the
    returned row log-probabilities; their per-segment logsumexp entry is
    left at a finite placeholder (``log(tiny)`` over the zero fill) that
    callers must not consume.

    Returns ``(row_log_probs [H], segment_lse [num_segments])``.
    """
    seg_max = logits.new_zeros(num_segments).scatter_reduce(
        0, index, logits, reduce="amax", include_self=False
    )
    row_max = seg_max.index_select(0, index)  # [H]
    shifted = torch.exp(logits - row_max)  # [H]; max row of each segment == 1
    seg_sum = logits.new_zeros(num_segments).scatter_add(0, index, shifted)
    tiny = torch.finfo(seg_sum.dtype).tiny
    segment_lse = seg_sum.clamp(min=tiny).log() + seg_max
    row_log_probs = logits - segment_lse.index_select(0, index)
    return row_log_probs, segment_lse


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandleSelectionOutput:
    """The auditable result of one handle-selection forward pass.

    Fields:
        logits: raw additive scores ``[H]`` (one row per active handle).
        log_probs: ``log P(handle_h | s_{g(h)})`` ``[H]`` — the per-graph
            log-softmax over exactly the active handle set (the TB forward
            log-probability contribution of the handle component;
            constitution §12.3).
        probs: ``exp(log_probs)`` ``[H]`` — a valid normalized distribution
            over each graph's active handle set (the #22 acceptance
            criterion).
        attention_weights: stage-1 cross-attention weights ``[H]`` —
            themselves a valid per-graph distribution (telemetry).
        handle_graph: graph assignment ``[H]`` int64 (validated echo).
        handle_atom: attachment-atom index within the ligand sub-graph
            ``[H]`` int64 (validated echo; maps 1:1 onto the environment's
            ``active_handles()`` menu rows in input order).
        handle_counts: number of active handles per graph ``[B]`` int64.
        entropy: per-graph entropy of the handle distribution ``[B]``
            (nats; exactly 0 for single-handle graphs and for graphs
            without handles — there is no distribution to be uncertain
            about).
        handle_embeddings: the invariant handle embeddings ``e_h``
            ``[H, d_head]`` (the #23 decoder-facing surface).
        pooled_handle_context: attention-pooled per-graph handle context
            ``c_g`` ``[B, d_head]`` (the #23 decoder-facing surface).
        num_graphs: number of graphs ``B``.

    Rows are returned in the caller's input order; per-graph segments are
    recoverable via :meth:`rows_for_graph`.
    """

    logits: torch.Tensor  # [H]
    log_probs: torch.Tensor  # [H]
    probs: torch.Tensor  # [H]
    attention_weights: torch.Tensor  # [H]
    handle_graph: torch.Tensor  # [H] int64
    handle_atom: torch.Tensor  # [H] int64
    handle_counts: torch.Tensor  # [B] int64
    entropy: torch.Tensor  # [B]
    handle_embeddings: torch.Tensor  # [H, d_head]
    pooled_handle_context: torch.Tensor  # [B, d_head]
    num_graphs: int

    @property
    def num_handles(self) -> int:
        """Total number of scored handle rows ``H``."""
        return int(self.log_probs.size(0))

    def rows_for_graph(self, graph: int) -> torch.Tensor:
        """Row indices (input order) of the handles of ``graph``."""
        if not 0 <= int(graph) < self.num_graphs:
            raise TensorContractError(
                f"[handle-head] graph index {graph} out of range "
                f"[0, {self.num_graphs})"
            )
        return torch.nonzero(self.handle_graph == int(graph), as_tuple=False).flatten()


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------


class HandleSelectionHead(nn.Module):
    """Agency-enabled handle-selection head — ``P(handle_t | s_t)``.

    Two-stage cross-attention scorer over the active handle pool (see the
    module docstring for the full architecture and invariance argument):

    * stage 1 — one scaled dot-product attention query per graph over that
      graph's handle keys, pooling an attention-weighted handle context;
    * stage 2 — additive scoring of each handle from (query, key, pooled
      context), normalized per graph by ``log_softmax`` over exactly the
      active handle set.

    The head consumes the :class:`BackboneOutput` of the single
    authoritative ``BipartitePaiNN`` encoder — never a second pocket
    encoder (#45 §7.2) — and the invariant handle descriptors of the
    frozen schema width (64).

    Args:
        backbone_dim: scalar feature dimension ``d`` of the consumed
            backbone representations (default: the frozen production
            :data:`~syntree.models.bipartite.BACKBONE_HIDDEN_DIM` = 256).
        hidden_dim: internal head width ``d_head`` (default
            :data:`HANDLE_HEAD_HIDDEN_DIM` = 128).
        num_radial: Gaussian centers of the centroid-distance embedding
            (default :data:`HANDLE_HEAD_NUM_RADIAL` = 20).
        geom_cutoff: upper edge (Angstrom) of the centroid-distance radial
            basis (default :data:`HANDLE_GEOM_CUTOFF` = 12.0).

    Forward:
        ``backbone``: a :class:`~syntree.models.bipartite.BackboneOutput`.
        ``handle_descriptors``: ``[H, 64]`` invariant chemical descriptors
            of the active handles (one row per handle; ``H >= 1``).
        ``handle_pos``: ``[H, 3]`` reacting atom positions (pocket frame).
        ``handle_graph``: ``[H]`` int64 graph assignment in ``[0, B)``.
        ``handle_atom``: ``[H]`` int64 attachment-atom indices within the
            ligand sub-graph, in ``[0, N_l)``, each belonging to the row's
            graph.
        Returns a :class:`HandleSelectionOutput`.
    """

    def __init__(
        self,
        backbone_dim: int = BACKBONE_HIDDEN_DIM,
        hidden_dim: int = HANDLE_HEAD_HIDDEN_DIM,
        num_radial: int = HANDLE_HEAD_NUM_RADIAL,
        geom_cutoff: float = HANDLE_GEOM_CUTOFF,
    ) -> None:
        super().__init__()
        backbone_dim = int(backbone_dim)
        hidden_dim = int(hidden_dim)
        num_radial = int(num_radial)
        geom_cutoff = float(geom_cutoff)
        if backbone_dim < 1:
            raise ValueError(f"backbone_dim must be >= 1, got {backbone_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_radial < 2:
            raise ValueError(
                f"num_radial must be >= 2 (GaussianRadialBasis spacing), got {num_radial}"
            )
        if not (geom_cutoff > 0.0) or geom_cutoff != geom_cutoff:
            raise ValueError(f"geom_cutoff must be > 0, got {geom_cutoff}")

        self.backbone_dim = backbone_dim
        self.hidden_dim = hidden_dim
        self.num_radial = num_radial
        self.geom_cutoff = geom_cutoff

        # Invariant handle embedding e_h (four additive contributions).
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(HANDLE_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atom_proj = nn.Linear(backbone_dim, hidden_dim)
        self.geom_proj = nn.Linear(num_radial, hidden_dim)
        self.dir_proj = nn.Linear(backbone_dim, hidden_dim)

        # Per-graph query (contextualized ligand summary + pocket context).
        self.query_ligand = nn.Linear(backbone_dim, hidden_dim)
        self.query_pocket = nn.Linear(backbone_dim, hidden_dim)

        # Stage-1 cross-attention values.
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stage-2 additive scoring.
        self.score_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_out = nn.Linear(hidden_dim, 1, bias=False)

        # Raw (envelope-free) centroid-distance radial basis — the cosine
        # envelope for all spatial RBF projections is owned by #9 (D-V2-007).
        self.radial_basis = GaussianRadialBasis(num_radial, geom_cutoff)

    def extra_repr(self) -> str:
        return (
            f"backbone_dim={self.backbone_dim}, hidden_dim={self.hidden_dim}, "
            f"num_radial={self.num_radial}, geom_cutoff={self.geom_cutoff}"
        )

    # -- audit ---------------------------------------------------------------

    def audit(self) -> Dict[str, Any]:
        """Parameter/dimension audit (constitution #45 §7.3).

        Returns a JSON-compatible dict of module dimensions, parameter
        counts, trainability, and the dtype policy.
        """
        parameters = list(self.parameters())
        return {
            "module": type(self).__name__,
            "backbone_dim": self.backbone_dim,
            "hidden_dim": self.hidden_dim,
            "descriptor_dim": HANDLE_FEATURE_DIM,
            "num_radial": self.num_radial,
            "geom_cutoff_angstrom": self.geom_cutoff,
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "num_buffers": len(list(self.buffers())),
            "deterministic": True,  # no dropout / stochastic layers
        }

    # -- input validation (fail-closed) ---------------------------------------

    def _reference_param(self) -> torch.Tensor:
        return self.score_out.weight

    def _check_pool_tensor(
        self,
        name: str,
        tensor: Any,
        *,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        fixed_last: int | None = None,
        rows: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TensorContractError(
                f"[handle-head] `{name}` must be a torch.Tensor, got "
                f"{type(tensor).__name__}"
            )
        if tensor.dtype not in _FLOAT_INPUT_DTYPES:
            raise TensorContractError(
                f"[handle-head] `{name}` must have dtype float32 (the "
                f"production path, decision D-V2-005) or float64 (local "
                f"numerical verification only, decision D-V2-008), got "
                f"{tensor.dtype}"
            )
        if tensor.dtype != dtype:
            raise TensorContractError(
                f"[handle-head] `{name}` has dtype {tensor.dtype} but the "
                f"module parameters are {dtype}; cast the module with "
                f".to({tensor.dtype}) for local numerical verification, or "
                "fix the producer"
            )
        if tensor.device != device:
            raise TensorContractError(
                f"[handle-head] `{name}` is on device {tensor.device} but "
                f"the module parameters are on {device}"
            )
        if tensor.dim() != rank:
            raise TensorContractError(
                f"[handle-head] `{name}` must be {rank}-D, got "
                f"{tuple(tensor.shape)}"
            )
        if rows is not None and tensor.size(0) != rows:
            raise TensorContractError(
                f"[handle-head] `{name}` must have {rows} rows (one per "
                f"active handle), got {tuple(tensor.shape)}"
            )
        if fixed_last is not None and tensor.size(-1) != fixed_last:
            raise TensorContractError(
                f"[handle-head] `{name}` must have size {fixed_last} on the "
                f"last axis, got {tuple(tensor.shape)}"
            )
        if tensor.numel() and not bool(torch.isfinite(tensor).all()):
            raise TensorContractError(
                f"[handle-head] `{name}` contains non-finite values "
                "(NaN/Inf); poisoned inputs fail closed — locate the producer"
            )
        return tensor

    def _check_index_tensor(
        self,
        name: str,
        tensor: Any,
        *,
        rows: int,
        device: torch.device,
        upper: int,
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TensorContractError(
                f"[handle-head] `{name}` must be a torch.Tensor, got "
                f"{type(tensor).__name__}"
            )
        if tensor.dtype not in (torch.int64, torch.long):
            raise TensorContractError(
                f"[handle-head] `{name}` must have dtype int64, got {tensor.dtype}"
            )
        if tensor.device != device:
            raise TensorContractError(
                f"[handle-head] `{name}` is on device {tensor.device} but "
                f"the module parameters are on {device}"
            )
        if tensor.dim() != 1 or tensor.size(0) != rows:
            raise TensorContractError(
                f"[handle-head] `{name}` must be [{rows}] (one row per "
                f"active handle), got {tuple(tensor.shape)}"
            )
        if tensor.numel():
            lo, hi = int(tensor.min()), int(tensor.max())
            if lo < 0 or hi >= upper:
                raise TensorContractError(
                    f"[handle-head] `{name}` values must lie in [0, {upper}), "
                    f"got range [{lo}, {hi}]"
                )
        return tensor

    def _validate_inputs(
        self,
        backbone: Any,
        handle_descriptors: Any,
        handle_pos: Any,
        handle_graph: Any,
        handle_atom: Any,
    ) -> tuple[BackboneOutput, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(backbone, BackboneOutput):
            raise TensorContractError(
                "[handle-head] `backbone` must be a BackboneOutput (the "
                f"BipartitePaiNN encoder output), got {type(backbone).__name__}"
            )
        ref = self._reference_param()
        dtype, device = ref.dtype, ref.device

        # -- descriptor block (defines H; carries the legacy ban) -----------
        handle_descriptors = self._check_pool_tensor(
            "handle_descriptors", handle_descriptors, rank=2, dtype=dtype, device=device
        )
        if handle_descriptors.size(-1) == LEGACY_HANDLE_FEATURE_DIM:
            raise TensorContractError(_LEGACY_DESCRIPTOR_MESSAGE)
        if handle_descriptors.size(-1) != HANDLE_FEATURE_DIM:
            raise TensorContractError(
                f"[handle-head] `handle_descriptors` must be [H, "
                f"{HANDLE_FEATURE_DIM}] (the frozen schema descriptor width, "
                f"issue #6), got {tuple(handle_descriptors.shape)}"
            )
        num_handles = int(handle_descriptors.size(0))
        if num_handles < 1:
            raise TensorContractError(
                "[handle-head] the handle pool is empty — there is no handle "
                "action to score (only STOP is legal on such states; this "
                "head must not be called)"
            )
        if backbone.num_ligand < 1:
            raise TensorContractError(
                "[handle-head] the state has no ligand nodes, yet the handle "
                "pool is non-empty — handles live on ligand atoms"
            )

        # -- geometry / index blocks -----------------------------------------
        handle_pos = self._check_pool_tensor(
            "handle_pos",
            handle_pos,
            rank=2,
            dtype=dtype,
            device=device,
            fixed_last=3,
            rows=num_handles,
        )
        handle_graph = self._check_index_tensor(
            "handle_graph",
            handle_graph,
            rows=num_handles,
            device=device,
            upper=backbone.num_graphs,
        )
        handle_atom = self._check_index_tensor(
            "handle_atom",
            handle_atom,
            rows=num_handles,
            device=device,
            upper=backbone.num_ligand,
        )
        # Cross-field consistency: each handle's attachment atom must be a
        # ligand node of the handle's own graph.
        ligand_batch = backbone.node_batch[backbone.num_pocket :]
        atom_graph = ligand_batch.index_select(0, handle_atom)
        if bool((atom_graph != handle_graph).any()):
            bad = torch.nonzero(atom_graph != handle_graph, as_tuple=False).flatten()
            raise TensorContractError(
                f"[handle-head] `handle_atom` rows {bad.tolist()} reference "
                "ligand nodes belonging to a different graph than their "
                "`handle_graph` row — the attachment atom must live in the "
                "handle's own graph"
            )

        # The consumed backbone representations must share the module dtype
        # (producer/consumer mismatch fails closed, mirroring the backbone's
        # own dtype policy).
        for name, tensor in (
            ("backbone.scalar", backbone.scalar),
            ("backbone.vector", backbone.vector),
            ("backbone.node_pos", backbone.node_pos),
            ("backbone.pocket_context", backbone.pocket_context),
        ):
            if tensor.dtype != dtype:
                raise TensorContractError(
                    f"[handle-head] `{name}` has dtype {tensor.dtype} but "
                    f"the module parameters are {dtype}; cast the head (and "
                    "the backbone that produced it) consistently for local "
                    "verification, or fix the producer"
                )
            if tensor.device != device:
                raise TensorContractError(
                    f"[handle-head] `{name}` is on device {tensor.device} "
                    f"but the module parameters are on {device}"
                )

        return backbone, handle_descriptors, handle_pos, handle_graph, handle_atom

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        backbone: BackboneOutput,
        handle_descriptors: torch.Tensor,
        handle_pos: torch.Tensor,
        handle_graph: torch.Tensor,
        handle_atom: torch.Tensor,
    ) -> HandleSelectionOutput:
        """Score the active handle pool; see the class docstring."""
        (
            backbone,
            handle_descriptors,
            handle_pos,
            handle_graph,
            handle_atom,
        ) = self._validate_inputs(
            backbone, handle_descriptors, handle_pos, handle_graph, handle_atom
        )

        num_graphs = backbone.num_graphs
        num_pocket = backbone.num_pocket
        dtype, device = handle_descriptors.dtype, handle_descriptors.device

        # -- gather the attachment atoms' backbone representations ----------
        ligand_scalar = backbone.scalar[num_pocket:]  # [N_l, d] invariant
        ligand_vector = backbone.vector[num_pocket:]  # [N_l, 3, d] equivariant
        s_atom = ligand_scalar.index_select(0, handle_atom)  # [H, d]
        v_atom = ligand_vector.index_select(0, handle_atom)  # [H, 3, d]

        # -- pocket centroid per graph (rigidly co-moving reference) ---------
        pocket_pos = backbone.node_pos[:num_pocket]  # [N_p, 3]
        pocket_batch = backbone.node_batch[:num_pocket]  # [N_p]
        centroid = torch.zeros((num_graphs, 3), dtype=dtype, device=device)
        centroid = centroid.index_add(0, pocket_batch, pocket_pos)
        pocket_counts = torch.bincount(pocket_batch, minlength=num_graphs).clamp(min=1)
        centroid = centroid / pocket_counts.to(dtype).unsqueeze(-1)  # [B, 3]

        # -- invariant handle embedding ---------------------------------------
        rel = handle_pos - centroid.index_select(0, handle_graph)  # [H, 3]
        radial = safe_norm(rel, dim=-1)  # [H] (exact 0 at r = 0, #7)
        geom_rbf = self.radial_basis(radial)  # [H, R] raw expansion (D-V2-007)
        dir_feat = torch.einsum("hkd,hk->hd", v_atom, rel)  # [H, d] invariant
        e = (
            self.descriptor_mlp(handle_descriptors)
            + self.atom_proj(s_atom)
            + self.geom_proj(geom_rbf)
            + self.dir_proj(dir_feat)
        )  # [H, d_head]

        # -- per-graph query ---------------------------------------------------
        ligand_batch = backbone.node_batch[num_pocket:]  # [N_l]
        lig_sum = torch.zeros(
            (num_graphs, self.backbone_dim), dtype=dtype, device=device
        )
        lig_sum = lig_sum.index_add(0, ligand_batch, ligand_scalar)
        lig_counts = torch.bincount(ligand_batch, minlength=num_graphs).clamp(min=1)
        lig_mean = lig_sum / lig_counts.to(dtype).unsqueeze(-1)  # [B, d]
        query = self.query_ligand(lig_mean) + self.query_pocket(
            backbone.pocket_context
        )  # [B, d_head]

        # -- stage 1: cross-attention over the handle pool --------------------
        q_rows = query.index_select(0, handle_graph)  # [H, d_head]
        attn_logits = (q_rows * e).sum(dim=-1) / math.sqrt(self.hidden_dim)  # [H]
        attn_log_probs, _ = _segment_log_softmax(attn_logits, handle_graph, num_graphs)
        attention_weights = attn_log_probs.exp()  # [H]
        values = self.value_proj(e)  # [H, d_head]
        weighted = values * attention_weights.unsqueeze(-1)
        context = torch.zeros(
            (num_graphs, self.hidden_dim), dtype=dtype, device=device
        )
        context = context.index_add(0, handle_graph, weighted)  # [B, d_head]

        # -- stage 2: additive scoring ------------------------------------------
        score_in = torch.tanh(
            self.score_query(q_rows)
            + self.score_key(e)
            + self.score_context(context.index_select(0, handle_graph))
        )  # [H, d_head]
        logits = self.score_out(score_in).squeeze(-1) / math.sqrt(
            self.hidden_dim
        )  # [H]

        # -- normalization over the actually valid alternatives ----------------
        log_probs, _ = _segment_log_softmax(logits, handle_graph, num_graphs)
        probs = log_probs.exp()

        handle_counts = torch.bincount(handle_graph, minlength=num_graphs)  # [B]
        neg_plogp = -(probs * log_probs)  # [H]
        entropy = torch.zeros(num_graphs, dtype=dtype, device=device)
        entropy = entropy.index_add(0, handle_graph, neg_plogp)  # [B] nats

        output = HandleSelectionOutput(
            logits=logits,
            log_probs=log_probs,
            probs=probs,
            attention_weights=attention_weights,
            handle_graph=handle_graph,
            handle_atom=handle_atom,
            handle_counts=handle_counts,
            entropy=entropy,
            handle_embeddings=e,
            pooled_handle_context=context,
            num_graphs=num_graphs,
        )
        self._assert_finite(output)
        return output

    def _assert_finite(self, output: HandleSelectionOutput) -> None:
        """Fail loudly on non-finite outputs (#45 §16: never hide NaNs)."""
        for name in (
            "logits",
            "log_probs",
            "probs",
            "attention_weights",
            "entropy",
            "handle_embeddings",
            "pooled_handle_context",
        ):
            tensor = getattr(output, name)
            if tensor.numel() and not bool(torch.isfinite(tensor).all()):
                raise TensorContractError(
                    f"[handle-head] `{name}` contains non-finite values — "
                    "with finite validated inputs this indicates corrupted "
                    "parameters; fail loudly per #45 §16"
                )
