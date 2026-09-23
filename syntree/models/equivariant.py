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

        # Directional unit vectors, [E, 3, 1].
        dir_vec = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)

        msg_s = scalar[col] * f_s                                # [E, H]
        msg_v = (
            vector[col] * f_v.unsqueeze(1)
            + dir_vec.unsqueeze(-1) * f_vv.unsqueeze(1)
        )                                                        # [E, 3, H]

        agg_s = torch.zeros_like(scalar)
        agg_v = torch.zeros_like(vector)
        agg_s.index_add_(0, row, msg_s.to(agg_s.dtype))
        agg_v.index_add_(0, row, msg_v.to(agg_v.dtype))
        return agg_s, agg_v


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
        v_norm = torch.norm(vec_proj + self.epsilon, dim=1)     # [N, H]

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
        scalar = scalar + agg_s
        vector = vector + agg_v
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
        scalar = scalar + agg_s
        vector = vector + agg_v
        return self.mixing(scalar, vector)


def build_radius_graph(
    pos: torch.Tensor, batch: torch.Tensor, cutoff: float, loop: bool = False
) -> torch.Tensor:
    """Radius graph via brute-force distance computation.

    A pure-torch implementation (no torch-cluster dependency) that stays
    correct on CPU and GPU alike. ``pos`` is ``[N, 3]`` and ``batch`` maps
    each node to its graph; edges never cross graphs.

    Set ``loop=True`` to include self-loops (PaiNN typically omits them).
    """
    if pos.dim() != 2 or pos.size(1) != 3:
        raise ValueError(f"pos must be [N, 3], got {tuple(pos.shape)}")
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

    dist_mat = torch.cdist(pos, pos)                            # [N, N]
    adj = dist_mat <= cutoff
    if not loop:
        adj.fill_diagonal_(False)
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    adj = adj & same_graph
    row, col = torch.nonzero(adj, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


__all__ = [
    "CosineCutoff",
    "RadialBasis",
    "PaiNNInteraction",
    "PaiNNMixing",
    "PaiNNLayer",
    "BatchedPaiNNLayer",
    "build_radius_graph",
]
