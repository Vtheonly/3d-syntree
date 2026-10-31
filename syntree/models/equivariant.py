"""SE(3)-equivariant message passing: PaiNN layers and radial basis.

Implements the Polarizable Atom Interaction Network (Schutt et al., 2021)
update scheme with scalar features ``s_i`` and vector features ``v_i``
attached to every atom. The layers are E(3)-equivariant: rotating the input
positions rotates the vector features accordingly while leaving scalar
features invariant.

All norms and normalisations route through the non-singular primitives
:func:`safe_norm` / :func:`safe_unit_vector` (#7 MATH-01): the
``eps``-smoothed root ``||x||_eps = sqrt(sum x^2 + eps^2) - eps`` has the
bounded gradient ``x / sqrt(sum x^2 + eps^2)``, which is exactly zero at
``x = 0`` and never divides a zero vector by its own norm. Raw
``torch.norm`` + ``+1e-8`` guards are retired from this module.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Largest magnitude that is exactly representable in fp16 minus headroom
# (max finite fp16 = 65,504). Used as the saturation bound for activations
# so that unbounded residual growth under fp16 autocast saturates cleanly
# instead of overflowing to +/-inf and poisoning gradients with NaN.
# NOTE (#10 MATH-04): fp16 clamping machinery is interim and scheduled for
# removal by the pure-FP32 precision engine.
FP16_ACT_CLAMP = 6.0e4

#: Default smoothing radius of the non-singular norm primitives (#7).
#: Well below any physical inter-atomic distance (Angstrom scale) and well
#: above fp32 rounding noise, so the approximation error is negligible
#: for real geometry while the gradient stays bounded at coincident
#: coordinates.
SAFE_NORM_EPS = 1e-5


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = SAFE_NORM_EPS,
) -> torch.Tensor:
    """Non-singular norm with smooth, bounded gradient as ``x -> 0``.

    ``||x||_eps = sqrt(sum_k x_k^2 + eps^2) - eps``

    with gradient ``x / sqrt(sum_k x_k^2 + eps^2)`` (magnitude bounded by
    1 everywhere, exactly ``0`` at ``x = 0``). Unlike ``torch.norm``, the
    gradient cannot spike or produce ``0/0`` NaNs at coincident
    coordinates or clashing fragments (#7 MATH-01, #45 §8.1).
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + (eps * eps)) - eps


def safe_unit_vector(
    vec: torch.Tensor,
    dim: int = -1,
    eps: float = SAFE_NORM_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Directional unit vectors and distances strictly bounded away from ``0/0``.

    Returns ``(unit, dist)`` where

    * ``unit = vec / sqrt(sum vec^2 + eps^2)`` - a well-defined direction
      for every input, including the zero vector (``unit = 0`` there);
    * ``dist = (sqrt(sum vec^2 + eps^2) - eps).clamp(min=0)`` - the
      :func:`safe_norm` distance, non-negative by construction.

    Both the values and their gradients w.r.t. ``vec`` stay finite for
    coincident coordinates; the gradient magnitude is bounded by
    ``1 / eps`` instead of diverging.
    """
    norm = torch.sqrt(torch.sum(vec * vec, dim=dim, keepdim=True) + (eps * eps))
    unit = vec / norm
    dist = (norm.squeeze(dim) - eps).clamp(min=0.0)
    return unit, dist


def finite_act_clamp(x: torch.Tensor) -> torch.Tensor:
    """Saturate non-finite / oversized activations to the fp16-safe range.

    A no-op for healthy fp32 values, and a hard safety net under fp16
    autocast: any +/-inf (from overflow) becomes +/-6e4 and NaN becomes 0,
    so the loss can stay finite and the finite-loss guard is a last resort
    rather than a routine.
    """
    if not x.is_floating_point():
        return x
    x = torch.nan_to_num(x, nan=0.0, posinf=FP16_ACT_CLAMP, neginf=-FP16_ACT_CLAMP)
    return x.clamp(min=-FP16_ACT_CLAMP, max=FP16_ACT_CLAMP)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope ``0.5 * (cos(pi * d / r_cut) + 1)``."""

    def __init__(self, cutoff: float):
        super().__init__()
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.cutoff = float(cutoff)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d = distances.clamp(min=0.0)
        envelope = 0.5 * (torch.cos(d * math.pi / self.cutoff) + 1.0)
        return torch.where(d < self.cutoff, envelope, torch.zeros_like(envelope))


class RadialBasis(nn.Module):
    """Gaussian radial basis function expansion with cosine cutoff.

    ``phi_k(d) = exp(-beta * (d - mu_k)^2) * cutoff(d)``
    """

    def __init__(self, num_radial: int = 20, cutoff: float = 5.0):
        super().__init__()
        if num_radial <= 0:
            raise ValueError(f"num_radial must be positive, got {num_radial}")
        self.num_radial = int(num_radial)
        self.cutoff = float(cutoff)
        self.register_buffer("means", torch.linspace(0.0, cutoff, num_radial))
        self.beta = float((2.0 / cutoff * num_radial) ** 2)
        self.cutoff_fn = CosineCutoff(cutoff)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.dim() != 1:
            raise ValueError(f"RadialBasis expects 1-D distances, got {tuple(distances.shape)}")
        envelope = self.cutoff_fn(distances)
        rbf = torch.exp(-self.beta * (distances.unsqueeze(-1) - self.means) ** 2)
        return rbf * envelope.unsqueeze(-1)

    def extra_repr(self) -> str:
        return f"num_radial={self.num_radial}, cutoff={self.cutoff:.2f}"


class PaiNNInteraction(nn.Module):
    """Inter-atomic message passing block.

    Filters radial basis features through a shared MLP, then splits the
    output into gates for scalar and vector messages.
    """

    def __init__(self, hidden_dim: int, num_radial: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.filter_mlp = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(
        self,
        scalar: torch.Tensor,          # [N, H]
        vector: torch.Tensor,          # [N, 3, H]
        edge_index: torch.Tensor,      # [2, E]
        edge_vec: torch.Tensor,        # [E, 3]
        edge_rbf: torch.Tensor,        # [E, num_radial]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Message-passing update.

        The directional unit vector is computed by :func:`safe_unit_vector`
        (#7 MATH-01): coincident coordinates (``d -> 0``) yield a zero
        direction with a bounded gradient instead of the ``0/0`` division
        the legacy ``edge_vec / (edge_dist + 1e-8)`` guard produced.
        """
        if scalar.dim() != 2 or vector.dim() != 3 or vector.shape[1] != 3:
            raise ValueError(
                f"Unexpected feature shapes: scalar {tuple(scalar.shape)}, "
                f"vector {tuple(vector.shape)}"
            )
        row, col = edge_index[0], edge_index[1]

        filters = self.filter_mlp(edge_rbf)                     # [E, 3H]
        f_s, f_v, f_vv = torch.split(filters, self.hidden_dim, dim=-1)

        # Non-singular directional unit vectors, [E, 3].
        dir_vec, _ = safe_unit_vector(edge_vec, dim=-1)

        msg_s = scalar[col] * f_s                                # [E, H]
        msg_v = (
            vector[col] * f_v.unsqueeze(1)
            + dir_vec.unsqueeze(-1) * f_vv.unsqueeze(1)
        )                                                        # [E, 3, H]

        agg_s = torch.zeros_like(scalar)
        agg_v = torch.zeros_like(vector)
        agg_s.index_add_(0, row, finite_act_clamp(msg_s).to(agg_s.dtype))
        agg_v.index_add_(0, row, finite_act_clamp(msg_v).to(agg_v.dtype))
        return finite_act_clamp(agg_s), finite_act_clamp(agg_v)


class PaiNNMixing(nn.Module):
    """Intra-atomic update mixing scalar and vector channels.

    Computes scalar norms of the (projected) vector features via
    :func:`safe_norm` (#7 MATH-01), applies an MLP over ``[s, |v|]``, and
    emits per-atom deltas for both channels.
    """

    def __init__(self, hidden_dim: int, epsilon: float = SAFE_NORM_EPS):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        # Smoothing radius of the non-singular norm (NOT an additive bias:
        # the legacy ``torch.norm(v + 1e-8)`` shifted every vector by eps
        # before taking the norm - mathematically wrong and still singular).
        self.epsilon = float(epsilon)
        self.vec_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(
        self, scalar: torch.Tensor, vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vec_proj = self.vec_proj(vector)                        # [N, 3, H]
        # Non-singular vector-feature norms (#7): bounded gradients, no
        # NaN-replacement clamps (fail loudly on genuinely poisoned inputs
        # instead of hiding them, #45 §8).
        v_norm = safe_norm(vec_proj, dim=1, eps=self.epsilon)     # [N, H]

        h = torch.cat([scalar, v_norm], dim=-1)                 # [N, 2H]
        u = self.update_mlp(h)                                  # [N, 3H]
        a_ss, a_sv, a_vv = torch.split(u, self.hidden_dim, dim=-1)

        delta_s = a_sv * v_norm + a_ss
        delta_v = a_vv.unsqueeze(1) * vector
        return scalar + delta_s, vector + delta_v


class PaiNNLayer(nn.Module):
    """One full PaiNN block: interaction (message passing) + mixing.

    Kept as a single module for drop-in compatibility with earlier specs
    that imported ``PaiNNLayer`` directly.
    """

    def __init__(self, hidden_dim: int, num_radial: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.rbf = RadialBasis(num_radial, cutoff)
        self.interaction = PaiNNInteraction(hidden_dim, num_radial)
        self.mixing = PaiNNMixing(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        # Non-singular edge distances (#7): coincident atoms stay finite.
        edge_dist = safe_norm(edge_vec, dim=-1)
        edge_rbf = self.rbf(edge_dist)

        agg_s, agg_v = self.interaction(
            scalar, vector, edge_index, edge_vec, edge_rbf
        )
        scalar = finite_act_clamp(scalar + agg_s)
        vector = finite_act_clamp(vector + agg_v)
        return self.mixing(scalar, vector)


# Batched variant used by the pocket encoder -------------------------------

class BatchedPaiNNLayer(nn.Module):
    """PaiNN block operating on precomputed batched graphs.

    Identical math to :class:`PaiNNLayer` but consumes edge vectors and
    radial basis features computed once by the encoder, avoiding
    recomputation when stacking many layers. (Edge distances are no
    longer a constructor-of-messages input: the interaction derives the
    non-singular direction/distance from ``edge_vec`` itself via
    :func:`safe_unit_vector`, #7 MATH-01.)
    """

    def __init__(self, hidden_dim: int, num_radial: int = 20):
        super().__init__()
        self.interaction = PaiNNInteraction(hidden_dim, num_radial)
        self.mixing = PaiNNMixing(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        agg_s, agg_v = self.interaction(
            scalar, vector, edge_index, edge_vec, edge_rbf
        )
        scalar = finite_act_clamp(scalar + agg_s)
        vector = finite_act_clamp(vector + agg_v)
        scalar, vector = self.mixing(scalar, vector)
        return finite_act_clamp(scalar), finite_act_clamp(vector)



def build_radius_graph(
    pos: torch.Tensor, batch: torch.Tensor, cutoff: float, loop: bool = False
) -> torch.Tensor:
    """Radius graph constructed per sub-graph to prevent O(N_total^2) memory explosion.

    A pure-torch implementation that stays correct on CPU and GPU alike.
    Edges never cross different graphs in a batch.
    """
    if pos.dim() != 2 or pos.size(1) != 3:
        raise ValueError(f"pos must be [N, 3], got {tuple(pos.shape)}")
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    if num_graphs == 1:
        dist_mat = torch.cdist(pos, pos)
        adj = dist_mat <= cutoff
        if not loop:
            adj.fill_diagonal_(False)
        row, col = torch.nonzero(adj, as_tuple=True)
        return torch.stack([row, col], dim=0)

    # If batch is not monotonic (e.g. pocket nodes concatenated with ligand nodes), sort by graph
    is_monotonic = bool((batch[1:] >= batch[:-1]).all().item()) if batch.numel() > 1 else True
    if not is_monotonic:
        perm = torch.argsort(batch, stable=True)
        pos_sorted = pos[perm]
        batch_sorted = batch[perm]
    else:
        perm = None
        pos_sorted = pos
        batch_sorted = batch

    # Compute radius edges per sub-graph using slice offsets (reduces peak memory from O(N_total^2) to sum O(N_b^2))
    rows, cols = [], []
    counts = torch.bincount(batch_sorted, minlength=num_graphs)
    starts = torch.cumsum(counts, dim=0) - counts

    for g in range(num_graphs):
        n_g = int(counts[g].item())
        if n_g <= 1:
            continue
        start_g = int(starts[g].item())
        pos_g = pos_sorted[start_g : start_g + n_g]
        dist_g = torch.cdist(pos_g, pos_g)
        adj_g = dist_g <= cutoff
        if not loop:
            adj_g.fill_diagonal_(False)
        row_g, col_g = torch.nonzero(adj_g, as_tuple=True)
        rows.append(row_g + start_g)
        cols.append(col_g + start_g)

    if not rows:
        return torch.zeros((2, 0), dtype=torch.long, device=pos.device)

    edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
    # Map back to original input indices if sorted
    if perm is not None:
        edge_index = perm[edge_index]
    return edge_index

__all__ = [
    "CosineCutoff",
    "RadialBasis",
    "PaiNNInteraction",
    "PaiNNMixing",
    "PaiNNLayer",
    "BatchedPaiNNLayer",
    "build_radius_graph",
    "finite_act_clamp",
    "FP16_ACT_CLAMP",
    "SAFE_NORM_EPS",
    "safe_norm",
    "safe_unit_vector",
]
