"""SE(3)-equivariant message passing: PaiNN layers and radial basis.

Implements the Polarizable Atom Interaction Network (Schutt et al., 2021)
update scheme with scalar features ``s_i`` and vector features ``v_i``
attached to every atom. The layers are E(3)-equivariant: rotating the input
positions rotates the vector features accordingly while leaving scalar
features invariant.
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
FP16_ACT_CLAMP = 6.0e4


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
        edge_dist: torch.Tensor,       # [E]
        edge_rbf: torch.Tensor,        # [E, num_radial]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if scalar.dim() != 2 or vector.dim() != 3 or vector.shape[1] != 3:
            raise ValueError(
                f"Unexpected feature shapes: scalar {tuple(scalar.shape)}, "
                f"vector {tuple(vector.shape)}"
            )
        row, col = edge_index[0], edge_index[1]

        filters = self.filter_mlp(edge_rbf)                     # [E, 3H]
        f_s, f_v, f_vv = torch.split(filters, self.hidden_dim, dim=-1)

        # fp16 safety: distances are physical (>= 0) but degenerate data can
        # produce coincident atoms (d == 0); clamp so the unit vector and RBF
        # stay finite instead of dividing by ~0.
        edge_dist = edge_dist.clamp(min=1e-6)

        # Directional unit vectors, [E, 3, 1].
        dir_vec = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)

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

    Computes scalar norms of the (projected) vector features, applies an
    MLP over ``[s, |v|]``, and emits per-atom deltas for both channels.
    """

    def __init__(self, hidden_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
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
        # AMP/fp16 safety: torch.norm squares its inputs and fp16 overflows
        # above 65,504, so vector features that grow during training would
        # turn v_norm into inf and poison BOTH channels. Compute the norm in
        # fp32 with a saturation cap, then return to the ambient dtype.
        v_norm = torch.norm(vec_proj.float() + self.epsilon, dim=1)   # [N, H]
        v_norm = torch.nan_to_num(v_norm, nan=0.0, posinf=1e4, neginf=0.0)
        v_norm = v_norm.clamp(max=1e4).to(scalar.dtype)              # [N, H]

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
        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_rbf = self.rbf(edge_dist)

        agg_s, agg_v = self.interaction(
            scalar, vector, edge_index, edge_vec, edge_dist, edge_rbf
        )
        scalar = finite_act_clamp(scalar + agg_s)
        vector = finite_act_clamp(vector + agg_v)
        return self.mixing(scalar, vector)


# Batched variant used by the pocket encoder -------------------------------

class BatchedPaiNNLayer(nn.Module):
    """PaiNN block operating on precomputed batched graphs.

    Identical math to :class:`PaiNNLayer` but consumes edge vectors,
    distances, and radial basis features computed once by the encoder,
    avoiding recomputation when stacking many layers.
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
        edge_dist: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        agg_s, agg_v = self.interaction(
            scalar, vector, edge_index, edge_vec, edge_dist, edge_rbf
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
]
