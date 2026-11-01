"""Bipartite SE(3)-equivariant backbone over the union pocket + ligand graph.

Clean-room V2 implementation of the single authoritative molecular encoder
(issue #21 MODEL-01; constitution #45 §7.2, decision D-V2-004).

**Why bipartite (the #21 motivation).** The pre-V2 single-pocket encoder was
blind to the growing ligand scaffold, so it could not see ligand-ligand and
pocket-ligand steric context — the root cause of severe self-clashes.
``BipartitePaiNN`` therefore encodes the *union* of protein-pocket atoms and
intermediate ligand atoms in one shared heterogeneous interaction graph::

    P = (X_P, Z_P, q_P)        pocket atoms  (N_p >= 1)
    M_t = (X_L, Z_L, q_L)      ligand atoms  (N_l >= 0, empty at t=0)
    V = V_P ∪ V_L              union nodes   (N = N_p + N_l)

with pocket-pocket, ligand-ligand, and pocket-ligand message passing in every
layer.

**Node ordering (binding).** Union nodes are ordered ``[pocket nodes, ligand
nodes]`` — for a collated batch that is ``[pockets of graphs 0..B-1, ligands
of graphs 0..B-1]``. ``node_type`` is ``0`` (pocket) / ``1`` (ligand);
``node_batch`` carries the graph assignment. ``handle_flags`` (optional)
follows the same union ordering.

**Input contract (fail-closed).** :meth:`BipartitePaiNN.forward` consumes the
canonical V2 state produced by ``syntree.core.schema.collate_state_batch``
(the single batching path, issue #6) — or any attribute container exposing
the eight node-level fields (``pocket_pos/z/charge/batch``,
``ligand_pos/z/charge/batch``), e.g. a ``StateTensorContract`` view. The
policy-level fields (``handle_features``, ``global_features``, ``stop_mask``)
are *not* consumed by the encoder and are not required. Every field the
backbone does consume is validated: exact dtype/rank/shape, atomic numbers in
``[1, MAX_ATOMIC_NUMBER]`` (the embedding-table bound), finiteness, batch
indices in ``[0, B)``, and at least one pocket atom per graph. Malformed
input raises :class:`~syntree.core.schema.TensorContractError` with a
precise message.

**Precision policy (decisions D-V2-005 / D-V2-008).** Parameters default to
pure FP32 (the production path on Colab/Kaggle Tesla T4). ``float64`` is
accepted *only* for local numerical verification of equivariance at tight
tolerance (mirroring the #7 test practice); the input dtype must match the
module parameter dtype exactly. FP16/BF16 and every other dtype are rejected
fail-closed. There is no NaN-masking, clamping, or silent fallback anywhere
in this module.

**Geometry operators.** All norms and unit vectors route through
:func:`~syntree.models.equivariant.safe_norm` /
:func:`~syntree.models.equivariant.safe_unit_vector` (#7 MATH-01, #45 §8.1) —
including the radius-graph distance threshold, so coincident coordinates
degrade gracefully (zero direction, bounded gradient) instead of producing
``0/0`` NaNs.

**Radial basis (#9 boundary, decision D-V2-007).** The 20-Gaussian radial
basis produced here is the *raw* expansion ``exp(-((d-mu_k)/(2 sigma))^2/2)``
over centers ``[0, r_cut]``. The smooth cosine cutoff envelope that must
multiply *all* spatial distance RBF projections is owned by #9 MATH-03 and is
deliberately NOT applied here; #9 will integrate it. Consequence (documented,
tested): at ``d = r_cut`` the last Gaussian still evaluates to 1.0 and edge
existence is a hard graph-construction threshold.

**Activation checkpointing (#45 §3.4 / §5.2).** Each message-then-update
layer is wrapped in non-reentrant ``torch.utils.checkpoint`` whenever
gradients are enabled, trading recompute for activation memory toward the
T4 budget. The capability is implemented and correctness-tested here; the
*measured* T4 VRAM benefit is deferred to #43 OPS-03 — no T4 claim is made
in this module.

**Outputs (the #39-facing surface).** :class:`BackboneOutput` carries the
contextualized union representations (scalar ``[N, d]`` invariant, vector
``[N, 3, d]`` equivariant), the union node metadata, and two pooled
per-graph contexts. The per-pocket-node representations are the
candidate-hotspot features and the pooled contexts are the pocket-conditioned
summaries that the seed/anchor policy (#39 MODEL-06) and decoder (#23
MODEL-03) consume — no second pocket encoder may be introduced downstream
(#45 §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from syntree.core.schema import MAX_ATOMIC_NUMBER, TensorContractError
from syntree.models.equivariant import PaiNNInteraction, PaiNNMixing, safe_norm

__all__ = [
    "BACKBONE_HIDDEN_DIM",
    "BACKBONE_NUM_LAYERS",
    "BACKBONE_NUM_RADIAL",
    "BACKBONE_CUTOFF",
    "GaussianRadialBasis",
    "build_union_radius_graph",
    "BackboneOutput",
    "BipartitePaiNN",
]

# ---------------------------------------------------------------------------
# Frozen production profile (issue #21; #45 §5.2: architecture dimensions are
# frozen in version-controlled configuration, never probed from free VRAM).
# ---------------------------------------------------------------------------

#: Scalar feature dimension ``d`` of the production backbone (vector
#: features are ``3 x d`` per atom).
BACKBONE_HIDDEN_DIM = 256

#: Number of equivariant message-then-update layers.
BACKBONE_NUM_LAYERS = 8

#: Number of Gaussian radial basis functions feeding the radial filters.
BACKBONE_NUM_RADIAL = 20

#: Radius-graph cutoff in Angstrom (pocket-pocket, ligand-ligand and
#: pocket-ligand pairs within this distance are connected).
BACKBONE_CUTOFF = 5.0

#: Node-type codes of the union graph.
NODE_TYPE_POCKET = 0
NODE_TYPE_LIGAND = 1

#: Input dtypes accepted by the backbone: FP32 production (D-V2-005) plus
#: FP64 for local numerical verification only (D-V2-008). FP16/BF16 are
#: rejected fail-closed.
_FLOAT_INPUT_DTYPES = (torch.float32, torch.float64)

#: Accepted dtypes for the per-node handle flags.
_HANDLE_FLAG_DTYPES = (torch.long, torch.int64, torch.bool, torch.float32, torch.float64)

#: The eight node-level state fields consumed by the backbone.
_NODE_FIELDS = (
    "pocket_pos",
    "pocket_z",
    "pocket_charge",
    "pocket_batch",
    "ligand_pos",
    "ligand_z",
    "ligand_charge",
    "ligand_batch",
)


# ---------------------------------------------------------------------------
# Gaussian radial basis (the #21 RBF producer; envelope deferred to #9)
# ---------------------------------------------------------------------------


class GaussianRadialBasis(nn.Module):
    """Fixed (non-learnable) Gaussian radial basis expansion.

    Implements the SchNet/PaiNN-style smeared distance expansion

        ``radial_k(d) = exp(-0.5 * ((d - mu_k) / sigma)^2)``

    with ``num_radial`` centers ``mu_k`` evenly spaced on ``[0, r_cut]`` and
    width ``sigma = r_cut / (num_radial - 1)`` (the center spacing). The
    centers and scale are registered buffers (cast with the module, persisted
    in the state dict, deterministic and reproducible); there are no trainable
    parameters.

    **No cutoff envelope (decision D-V2-007):** the output is the raw
    Gaussian expansion. The continuous cosine envelope that must multiply all
    spatial RBF projections is specified and owned by #9 MATH-03 and will be
    integrated there — at ``d = r_cut`` the last Gaussian here still evaluates
    to 1.0 (pinned by test).

    Args:
        num_radial: number of Gaussian centers (>= 2).
        r_cut: upper edge of the center range in Angstrom (> 0).

    Forward:
        ``dist``: tensor of non-negative distances, any rank.
        Returns the Gaussian expansion with one extra trailing axis of size
        ``num_radial`` (e.g. ``[E] -> [E, R]``, ``[B, N] -> [B, N, R]``).
        Negative or non-finite distances raise
        :class:`~syntree.core.schema.TensorContractError` (fail-closed).
    """

    def __init__(self, num_radial: int = BACKBONE_NUM_RADIAL, r_cut: float = BACKBONE_CUTOFF) -> None:
        super().__init__()
        num_radial = int(num_radial)
        r_cut = float(r_cut)
        if num_radial < 2:
            raise ValueError(
                f"GaussianRadialBasis requires num_radial >= 2 (need at least "
                f"two centers to define the spacing), got {num_radial}"
            )
        if not (r_cut > 0.0) or r_cut != r_cut:
            raise ValueError(f"GaussianRadialBasis requires r_cut > 0, got {r_cut}")
        self.num_radial = num_radial
        self.r_cut = r_cut
        centers = torch.linspace(0.0, r_cut, num_radial)
        spacing = r_cut / (num_radial - 1)
        self.register_buffer("centers", centers)  # [R]
        self.register_buffer("_scale", torch.tensor(-0.5 / (spacing * spacing)))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if not isinstance(dist, torch.Tensor):
            raise TensorContractError(
                f"[radial-basis] `dist` must be a torch.Tensor, got "
                f"{type(dist).__name__}"
            )
        if dist.numel() and not bool(torch.isfinite(dist).all()):
            raise TensorContractError(
                "[radial-basis] `dist` contains non-finite values "
                "(NaN/Inf); poisoned inputs fail closed — locate the producer"
            )
        if dist.numel() and bool((dist < 0).any()):
            raise TensorContractError(
                "[radial-basis] `dist` contains negative values; distances "
                "are non-negative by definition — locate the producer"
            )
        diff = dist.unsqueeze(-1) - self.centers  # [..., R]
        return torch.exp(self._scale * diff * diff)


# ---------------------------------------------------------------------------
# The one authoritative edge construction (#45 §7.2)
# ---------------------------------------------------------------------------


def build_union_radius_graph(
    pos: torch.Tensor,
    node_batch: torch.Tensor,
    r_cut: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact radius graph over the union node set (the one authoritative V2
    edge construction; downstream heads must reuse this, not re-implement it).

    Connects every pair of nodes ``i != j`` of the *same* graph whose
    distance is within ``r_cut`` — which yields exactly the three required
    pair families (pocket-pocket, ligand-ligand, pocket-ligand) on the union
    graph. Both directions of every pair are emitted (PaiNN messages are
    directional). Self-loops are excluded. Edges never cross graphs.

    Returns ``(edge_index, edge_vec, edge_dist)``:

    * ``edge_index [2, E]`` — ``edge_index[0]`` is the receiver ``i``,
      ``edge_index[1]`` the sender ``j`` (the :class:`PaiNNInteraction`
      convention: messages flow sender -> receiver);
    * ``edge_vec [E, 3]`` — ``r_j - r_i`` (receiver -> sender), the
      differentiable direction input of the message block;
    * ``edge_dist [E]`` — :func:`safe_norm` of ``edge_vec``, the
      differentiable distance input of the radial basis.

    Numerical properties:

    * The pairwise distance matrix used for the *threshold* is computed
      through :func:`safe_norm` under ``torch.no_grad()`` (edge existence is
      a discrete geometric test; gradients flow only through the returned
      ``edge_vec`` / ``edge_dist`` features). The threshold is therefore
      eps-inclusive: pairs with true distance ``<= r_cut + SAFE_NORM_EPS``
      are kept — a ``1e-5`` Angstrom tolerance, physically negligible.
    * The distance matrix is exactly symmetric (squared-difference sums),
      so the emitted directed edge set is exactly bidirectional.
    * Coincident distinct atoms produce a zero-length edge vector (distance
      exactly 0), which the #7 safe primitives map to a zero direction with
      bounded gradients — no NaN.
    * Construction is ``O(N^2)`` in transient memory (an ``[N, N, 3]``
      buffer under no_grad, ~7 MB at the #43 pocket cap of ~700 atoms in
      FP32); the production pocket cap keeps this bounded.

    Args:
        pos: union node positions ``[N, 3]`` (``N >= 1``), finite.
        node_batch: graph assignment ``[N]`` int64, values ``>= 0``.
        r_cut: connection radius in Angstrom (``> 0``).
    """

    if not isinstance(pos, torch.Tensor) or not isinstance(node_batch, torch.Tensor):
        raise TensorContractError(
            "[radius-graph] `pos` and `node_batch` must be torch tensors, got "
            f"{type(pos).__name__} and {type(node_batch).__name__}"
        )
    if pos.dim() != 2 or pos.size(1) != 3:
        raise TensorContractError(
            f"[radius-graph] `pos` must be [N, 3], got {tuple(pos.shape)}"
        )
    n = pos.size(0)
    if n < 1:
        raise TensorContractError(
            "[radius-graph] `pos` must contain at least one node"
        )
    if node_batch.dim() != 1 or node_batch.size(0) != n:
        raise TensorContractError(
            f"[radius-graph] `node_batch` must be [N={n}], got "
            f"{tuple(node_batch.shape)}"
        )
    if node_batch.dtype not in (torch.int64, torch.long):
        raise TensorContractError(
            f"[radius-graph] `node_batch` must have dtype int64, got "
            f"{node_batch.dtype}"
        )
    if node_batch.numel() and int(node_batch.min()) < 0:
        raise TensorContractError(
            "[radius-graph] `node_batch` values must be >= 0, got minimum "
            f"{int(node_batch.min())}"
        )
    if pos.numel() and not bool(torch.isfinite(pos).all()):
        raise TensorContractError(
            "[radius-graph] `pos` contains non-finite values (NaN/Inf); "
            "poisoned inputs fail closed — locate the producer"
        )
    r_cut = float(r_cut)
    if not (r_cut > 0.0) or r_cut != r_cut:
        raise TensorContractError(f"[radius-graph] `r_cut` must be > 0, got {r_cut}")

    # Structural mask only (no autograd): safe-norm pairwise distances.
    with torch.no_grad():
        pair_diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 3]
        pair_dist = safe_norm(pair_diff, dim=-1)  # [N, N]
        mask = pair_dist <= r_cut
        mask.fill_diagonal_(False)  # no self-loops
        same_graph = node_batch.unsqueeze(0) == node_batch.unsqueeze(1)  # [N, N]
        mask &= same_graph
        receiver, sender = mask.nonzero(as_tuple=True)  # row-major order

    edge_index = torch.stack([receiver, sender], dim=0)  # [2, E]
    # Differentiable geometric features on the emitted edges only.
    edge_vec = pos[sender] - pos[receiver]  # r_j - r_i  [E, 3]
    edge_dist = safe_norm(edge_vec, dim=-1)  # [E]
    return edge_index, edge_vec, edge_dist


# ---------------------------------------------------------------------------
# Per-graph mean pooling
# ---------------------------------------------------------------------------


def _mean_pool(features: torch.Tensor, index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Mean-pool ``features [..., D]`` over graph ``index [N]`` -> ``[B, D]``.

    ``index`` must lie in ``[0, num_graphs)`` and every graph must be
    populated (validated by the caller); the ``clamp(min=1)`` guard is
    belt-and-braces so a future refactor can never manufacture a NaN.
    """
    sums = features.new_zeros((num_graphs, features.size(-1)))
    sums = sums.index_add(0, index, features)
    counts = torch.bincount(index, minlength=num_graphs)
    return sums / counts.clamp(min=1).to(features.dtype).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Backbone output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BackboneOutput:
    """Contextualized union-graph representations produced by
    :class:`BipartitePaiNN`.

    Fields:
        scalar: invariant node features ``[N, d]`` (union ordering:
            all pocket nodes first, then all ligand nodes).
        vector: equivariant node features ``[N, 3, d]``.
        node_pos: union positions ``[N, 3]`` (same ordering).
        node_type: ``[N]`` int64 — ``0`` pocket, ``1`` ligand.
        node_batch: ``[N]`` int64 — graph assignment in ``[0, B)``.
        num_pocket: ``N_p`` — nodes ``[0, N_p)`` are pocket nodes.
        num_graphs: number of graphs ``B``.
        graph_context: per-graph mean of the union scalar features
            ``[B, d]`` (invariant under rigid motions).
        pocket_context: per-graph mean of the *pocket* scalar features
            ``[B, d]`` — the pocket-conditioned summary consumed by the
            seed/anchor policy (#39) alongside the per-pocket-node
            candidate-hotspot features (``pocket_scalar`` / ``pocket_vector``
            / ``node_pos`` restricted to pocket nodes).

    Slice properties: ``pocket_scalar``, ``pocket_vector``, ``ligand_scalar``,
    ``ligand_vector`` (empty tensors when ``N_l = 0``), ``pocket_mask``,
    ``ligand_mask``, ``num_ligand``.
    """

    scalar: torch.Tensor  # [N, d]
    vector: torch.Tensor  # [N, 3, d]
    node_pos: torch.Tensor  # [N, 3]
    node_type: torch.Tensor  # [N] long
    node_batch: torch.Tensor  # [N] long
    num_pocket: int
    num_graphs: int
    graph_context: torch.Tensor  # [B, d]
    pocket_context: torch.Tensor  # [B, d]

    @property
    def num_ligand(self) -> int:
        return int(self.scalar.size(0)) - int(self.num_pocket)

    @property
    def pocket_scalar(self) -> torch.Tensor:
        return self.scalar[: self.num_pocket]

    @property
    def pocket_vector(self) -> torch.Tensor:
        return self.vector[: self.num_pocket]

    @property
    def ligand_scalar(self) -> torch.Tensor:
        return self.scalar[self.num_pocket :]

    @property
    def ligand_vector(self) -> torch.Tensor:
        return self.vector[self.num_pocket :]

    @property
    def pocket_mask(self) -> torch.Tensor:
        return self.node_type == NODE_TYPE_POCKET

    @property
    def ligand_mask(self) -> torch.Tensor:
        return self.node_type == NODE_TYPE_LIGAND


# ---------------------------------------------------------------------------
# One message-then-update layer
# ---------------------------------------------------------------------------


class _PaiNNLayer(nn.Module):
    """One equivariant layer: message passing then atomwise update.

    Composes the #7 blocks in the PaiNN paper order —
    :class:`PaiNNInteraction` (Eqs. 7-8, message + residual) followed by
    :class:`PaiNNMixing` (Eqs. 9-10, atomwise update + residual). The
    backbone owns the composition (decision D-V2-004); this wrapper adds no
    mathematics of its own.
    """

    def __init__(self, hidden_dim: int, num_radial: int) -> None:
        super().__init__()
        self.interaction = PaiNNInteraction(hidden_dim, num_radial)
        self.mixing = PaiNNMixing(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,  # [N, H]
        vector: torch.Tensor,  # [N, 3, H]
        edge_index: torch.Tensor,  # [2, E]
        edge_vec: torch.Tensor,  # [E, 3]
        edge_rbf: torch.Tensor,  # [E, R]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scalar, vector = self.interaction(scalar, vector, edge_index, edge_vec, edge_rbf)
        return self.mixing(scalar, vector)


# ---------------------------------------------------------------------------
# The backbone
# ---------------------------------------------------------------------------


class _ValidatedState(NamedTuple):
    pocket_pos: torch.Tensor
    pocket_z: torch.Tensor
    pocket_charge: torch.Tensor
    pocket_batch: torch.Tensor
    ligand_pos: torch.Tensor
    ligand_z: torch.Tensor
    ligand_charge: torch.Tensor
    ligand_batch: torch.Tensor
    num_graphs: int


class BipartitePaiNN(nn.Module):
    """SE(3)-equivariant bipartite encoder over the union pocket + ligand graph.

    The single authoritative V2 molecular encoder (#21 MODEL-01, #45 §7.2):
    8 PaiNN message-then-update layers at ``d = 256`` with 20 Gaussian RBFs
    over an exact ``r_cut = 5.0`` Angstrom radius graph connecting
    pocket-pocket, ligand-ligand and pocket-ligand pairs of the same state.
    Scalar outputs are invariant and vector outputs equivariant under rigid
    motions (pinned by ``tests/test_bipartite_painn.py``). The encoder runs
    cleanly as a pocket-only encoder when ``N_l = 0`` (the t=0 state).

    Node features (per union atom):

    * shared atomic-number embedding ``Z_i in [1, 100]`` (one table for
      pocket and ligand atoms; row 0 is reserved and never indexed);
    * node-type embedding ``t_i in {0: pocket, 1: ligand}``;
    * physiological formal charge ``q_i in R`` through a bias-free MLP;
    * handle-flag embedding ``h_i in {0, 1}`` (marks reactive-handle atoms;
      defaults to "none marked" — at t=0 there is no ligand to mark).

    Vector features start at zero and acquire geometry exclusively through
    the equivariant message passing. Every norm / unit vector routes through
    the #7 safe primitives.

    Args:
        hidden_dim: scalar feature dimension ``d`` (default: the frozen
            production profile :data:`BACKBONE_HIDDEN_DIM` = 256; vector
            features are ``3 x d``).
        num_layers: number of message-then-update layers (default
            :data:`BACKBONE_NUM_LAYERS` = 8).
        num_radial: number of Gaussian RBFs (default
            :data:`BACKBONE_NUM_RADIAL` = 20).
        r_cut: radius-graph cutoff in Angstrom (default
            :data:`BACKBONE_CUTOFF` = 5.0).
        activation_checkpointing: wrap every layer in non-reentrant
            ``torch.utils.checkpoint`` while gradients are enabled (default
            ``True``; the *measured* T4 memory benefit is #43's mandate —
            no T4 claim is made here).

    Forward:
        ``state``: the canonical V2 state batch (the
        :func:`~syntree.core.schema.collate_state_batch` output) or any
        attribute container exposing the eight node-level fields.
        ``handle_flags``: optional ``[N_p + N_l]`` tensor with values in
        ``{0, 1}`` marking reactive-handle atoms in union ordering (dtype
        int64 / bool / float32 / float64; same device as the module).
        Returns a :class:`BackboneOutput`.
    """

    def __init__(
        self,
        hidden_dim: int = BACKBONE_HIDDEN_DIM,
        num_layers: int = BACKBONE_NUM_LAYERS,
        num_radial: int = BACKBONE_NUM_RADIAL,
        r_cut: float = BACKBONE_CUTOFF,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        num_radial = int(num_radial)
        r_cut = float(r_cut)
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_radial < 2:
            raise ValueError(
                f"num_radial must be >= 2 (GaussianRadialBasis spacing), got "
                f"{num_radial}"
            )
        if not (r_cut > 0.0) or r_cut != r_cut:
            raise ValueError(f"r_cut must be > 0, got {r_cut}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_radial = num_radial
        self.r_cut = r_cut
        self.activation_checkpointing = bool(activation_checkpointing)

        # Shared element table: indices 1..100 (schema MAX_ATOMIC_NUMBER);
        # row 0 is reserved and never indexed (z >= 1 is enforced).
        self.atomic_emb = nn.Embedding(MAX_ATOMIC_NUMBER + 1, hidden_dim)
        self.type_emb = nn.Embedding(2, hidden_dim)
        self.charge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.handle_emb = nn.Embedding(2, hidden_dim)
        self.radial_basis = GaussianRadialBasis(num_radial, r_cut)
        self.layers = nn.ModuleList(
            _PaiNNLayer(hidden_dim, num_radial) for _ in range(num_layers)
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, "
            f"num_radial={self.num_radial}, r_cut={self.r_cut}, "
            f"activation_checkpointing={self.activation_checkpointing}"
        )

    # -- input validation (fail-closed) ------------------------------------

    def _reference_param(self) -> torch.Tensor:
        return self.atomic_emb.weight

    def _extract_and_validate(self, state: Any) -> _ValidatedState:
        """Extract and fail-closed-validate the eight consumed node fields."""

        if state is None:
            raise TensorContractError(
                "[bipartite] received None; the backbone consumes a canonical "
                "V2 state (collate_state_batch output) or an equivalent "
                "container with the 8 node-level fields"
            )
        missing = [name for name in _NODE_FIELDS if getattr(state, name, None) is None]
        if missing:
            raise TensorContractError(
                f"[bipartite] {type(state).__name__} is missing required "
                f"node-level state field(s): {', '.join(missing)}. The "
                "backbone consumes pocket_pos/z/charge/batch and "
                "ligand_pos/z/charge/batch (pass the collate_state_batch "
                "output or an equivalent container; issue #21)"
            )

        ref = self._reference_param()
        expected_dtype = ref.dtype
        expected_device = ref.device

        def check_float(name: str, t: torch.Tensor, *, rank: int, fixed: dict) -> torch.Tensor:
            if not isinstance(t, torch.Tensor):
                raise TensorContractError(
                    f"[bipartite] `{name}` must be a torch.Tensor, got "
                    f"{type(t).__name__}"
                )
            if t.dtype not in _FLOAT_INPUT_DTYPES:
                raise TensorContractError(
                    f"[bipartite] `{name}` must have dtype float32 (the "
                    f"production path, decision D-V2-005) or float64 (local "
                    f"numerical verification only, decision D-V2-008), got "
                    f"{t.dtype}"
                )
            if t.dtype != expected_dtype:
                raise TensorContractError(
                    f"[bipartite] `{name}` has dtype {t.dtype} but the module "
                    f"parameters are {expected_dtype}; cast the module with "
                    f".to({t.dtype}) for local numerical verification, or fix "
                    "the producer"
                )
            if t.device != expected_device:
                raise TensorContractError(
                    f"[bipartite] `{name}` is on device {t.device} but the "
                    f"module parameters are on {expected_device}"
                )
            if t.dim() != rank:
                raise TensorContractError(
                    f"[bipartite] `{name}` must be {rank}-D, got "
                    f"{tuple(t.shape)}"
                )
            for axis, want in fixed.items():
                if t.size(axis) != want:
                    raise TensorContractError(
                        f"[bipartite] `{name}` must have size {want} on axis "
                        f"{axis}, got {tuple(t.shape)}"
                    )
            if t.numel() and not bool(torch.isfinite(t).all()):
                raise TensorContractError(
                    f"[bipartite] `{name}` contains non-finite values "
                    "(NaN/Inf); poisoned inputs fail closed — locate the "
                    "producer"
                )
            return t

        def check_long(name: str, t: torch.Tensor, length: int) -> torch.Tensor:
            if not isinstance(t, torch.Tensor):
                raise TensorContractError(
                    f"[bipartite] `{name}` must be a torch.Tensor, got "
                    f"{type(t).__name__}"
                )
            if t.dtype not in (torch.int64, torch.long):
                raise TensorContractError(
                    f"[bipartite] `{name}` must have dtype int64, got "
                    f"{t.dtype}"
                )
            if t.device != expected_device:
                raise TensorContractError(
                    f"[bipartite] `{name}` is on device {t.device} but the "
                    f"module parameters are on {expected_device}"
                )
            if t.dim() != 1 or t.size(0) != length:
                raise TensorContractError(
                    f"[bipartite] `{name}` must be [{length}], got "
                    f"{tuple(t.shape)}"
                )
            return t

        def check_atomic(name: str, t: torch.Tensor) -> None:
            if t.numel() == 0:
                return
            lo, hi = int(t.min()), int(t.max())
            if lo < 1 or hi > MAX_ATOMIC_NUMBER:
                raise TensorContractError(
                    f"[bipartite] `{name}` atomic numbers must lie in "
                    f"[1, {MAX_ATOMIC_NUMBER}] (the shared element-embedding "
                    f"table bound), got range [{lo}, {hi}]"
                )

        # Pocket block (N_p >= 1: a pocket-less state is meaningless).
        pocket_pos = check_float("pocket_pos", state.pocket_pos, rank=2, fixed={1: 3})
        num_pocket = pocket_pos.size(0)
        if num_pocket < 1:
            raise TensorContractError(
                "[bipartite] `pocket_pos` must contain at least one atom — "
                "a pocket-less state is meaningless"
            )
        pocket_z = check_long("pocket_z", state.pocket_z, num_pocket)
        check_atomic("pocket_z", pocket_z)
        pocket_charge = check_float("pocket_charge", state.pocket_charge, rank=1, fixed={})
        if pocket_charge.size(0) != num_pocket:
            raise TensorContractError(
                f"[bipartite] `pocket_charge` must have length {num_pocket} "
                f"to match `pocket_pos`, got {tuple(pocket_charge.shape)}"
            )
        pocket_batch = check_long("pocket_batch", state.pocket_batch, num_pocket)

        # Ligand block (N_l >= 0: explicitly empty at t=0).
        ligand_pos = check_float("ligand_pos", state.ligand_pos, rank=2, fixed={1: 3})
        num_ligand = ligand_pos.size(0)
        ligand_z = check_long("ligand_z", state.ligand_z, num_ligand)
        check_atomic("ligand_z", ligand_z)
        ligand_charge = check_float("ligand_charge", state.ligand_charge, rank=1, fixed={})
        if ligand_charge.size(0) != num_ligand:
            raise TensorContractError(
                f"[bipartite] `ligand_charge` must have length {num_ligand} "
                f"to match `ligand_pos`, got {tuple(ligand_charge.shape)}"
            )
        ligand_batch = check_long("ligand_batch", state.ligand_batch, num_ligand)

        # Graph count: trust a PyG Batch's num_graphs, otherwise derive
        # from the batch vectors (max index + 1).
        num_graphs = getattr(state, "num_graphs", None)
        if num_graphs is None:
            pooled_max = int(pocket_batch.max())
            if num_ligand:
                pooled_max = max(pooled_max, int(ligand_batch.max()))
            num_graphs = pooled_max + 1
        num_graphs = int(num_graphs)
        if num_graphs < 1:
            raise TensorContractError(
                f"[bipartite] inferred num_graphs must be >= 1, got "
                f"{num_graphs}"
            )

        for name, vec in (
            ("pocket_batch", pocket_batch),
            ("ligand_batch", ligand_batch),
        ):
            if vec.numel() and int(vec.max()) >= num_graphs:
                raise TensorContractError(
                    f"[bipartite] `{name}` indices must lie in "
                    f"[0, {num_graphs}), got maximum {int(vec.max())}"
                )
            if vec.numel() and int(vec.min()) < 0:
                raise TensorContractError(
                    f"[bipartite] `{name}` indices must be >= 0, got minimum "
                    f"{int(vec.min())}"
                )
        # Every graph must own at least one pocket atom: the pocket defines
        # the conditioning context, and pooled pocket statistics would
        # otherwise be ill-defined (0/0).
        pocket_counts = torch.bincount(pocket_batch, minlength=num_graphs)
        if not bool((pocket_counts >= 1).all()):
            empty = torch.nonzero(pocket_counts == 0).flatten().tolist()
            raise TensorContractError(
                f"[bipartite] every graph must contain at least one pocket "
                f"atom; graph(s) {empty} have none (the pocket defines the "
                "conditioning context and pocket pooling would be 0/0)"
            )

        return _ValidatedState(
            pocket_pos=pocket_pos,
            pocket_z=pocket_z,
            pocket_charge=pocket_charge,
            pocket_batch=pocket_batch,
            ligand_pos=ligand_pos,
            ligand_z=ligand_z,
            ligand_charge=ligand_charge,
            ligand_batch=ligand_batch,
            num_graphs=num_graphs,
        )

    def _validate_handle_flags(
        self, handle_flags: Any, n_total: int
    ) -> torch.Tensor:
        """Validate/normalise the per-node handle flags to int64 ``[N]``."""
        ref_device = self._reference_param().device
        if handle_flags is None:
            return torch.zeros(n_total, dtype=torch.long, device=ref_device)
        if not isinstance(handle_flags, torch.Tensor):
            raise TensorContractError(
                f"[bipartite] `handle_flags` must be a torch.Tensor, got "
                f"{type(handle_flags).__name__}"
            )
        if handle_flags.dtype not in _HANDLE_FLAG_DTYPES:
            raise TensorContractError(
                f"[bipartite] `handle_flags` must have dtype int64, bool, "
                f"float32 or float64, got {handle_flags.dtype}"
            )
        if handle_flags.device != ref_device:
            raise TensorContractError(
                f"[bipartite] `handle_flags` is on device "
                f"{handle_flags.device} but the module parameters are on "
                f"{ref_device}"
            )
        if handle_flags.dim() != 1 or handle_flags.numel() != n_total:
            raise TensorContractError(
                f"[bipartite] `handle_flags` must be [{n_total}] (union "
                f"ordering: pocket nodes then ligand nodes), got "
                f"{tuple(handle_flags.shape)}"
            )
        if handle_flags.numel():
            as_float = handle_flags.to(torch.float64)
            if not bool(((as_float == 0) | (as_float == 1)).all()):
                raise TensorContractError(
                    "[bipartite] `handle_flags` values must be exactly 0 or "
                    "1 (reactive-handle indicator), found other values"
                )
        return handle_flags.to(torch.long)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        state: Any,
        handle_flags: torch.Tensor | None = None,
    ) -> BackboneOutput:
        """Encode a (batched) V2 construction state; see the class docstring."""
        v = self._extract_and_validate(state)

        pos = torch.cat([v.pocket_pos, v.ligand_pos], dim=0)  # [N, 3]
        z = torch.cat([v.pocket_z, v.ligand_z], dim=0)  # [N]
        charge = torch.cat([v.pocket_charge, v.ligand_charge], dim=0)  # [N]
        node_batch = torch.cat([v.pocket_batch, v.ligand_batch], dim=0)  # [N]
        n_p, n_l = v.pocket_pos.size(0), v.ligand_pos.size(0)
        n = n_p + n_l
        device = pos.device
        node_type = torch.cat(
            [
                torch.full((n_p,), NODE_TYPE_POCKET, dtype=torch.long, device=device),
                torch.full((n_l,), NODE_TYPE_LIGAND, dtype=torch.long, device=device),
            ]
        )
        handle_index = self._validate_handle_flags(handle_flags, n)

        # The one authoritative interaction graph (#45 §7.2): pocket-pocket,
        # ligand-ligand and pocket-ligand pairs of the same state within
        # r_cut. Runs cleanly at N_l = 0 (pocket-only encoder).
        edge_index, edge_vec, edge_dist = build_union_radius_graph(
            pos, node_batch, self.r_cut
        )
        edge_rbf = self.radial_basis(edge_dist)  # [E, R]

        # Initial features: invariant chemistry only — Cartesian positions
        # never enter scalar channels (#6 contract, #45 §7.1).
        scalar = (
            self.atomic_emb(z)
            + self.type_emb(node_type)
            + self.charge_mlp(charge.unsqueeze(-1))
            + self.handle_emb(handle_index)
        )
        vector = pos.new_zeros((n, 3, self.hidden_dim))

        for layer in self.layers:
            if self.activation_checkpointing and torch.is_grad_enabled():
                scalar, vector = checkpoint(
                    layer,
                    scalar,
                    vector,
                    edge_index,
                    edge_vec,
                    edge_rbf,
                    use_reentrant=False,
                )
            else:
                scalar, vector = layer(scalar, vector, edge_index, edge_vec, edge_rbf)

        graph_context = _mean_pool(scalar, node_batch, v.num_graphs)
        pocket_context = _mean_pool(scalar[:n_p], v.pocket_batch, v.num_graphs)

        return BackboneOutput(
            scalar=scalar,
            vector=vector,
            node_pos=pos,
            node_type=node_type,
            node_batch=node_batch,
            num_pocket=n_p,
            num_graphs=v.num_graphs,
            graph_context=graph_context,
            pocket_context=pocket_context,
        )
