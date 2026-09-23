"""SE(3)-equivariant protein pocket encoder (PaiNN backbone).

Processes the pocket point cloud ``(X_P in R^{N x 3}, z in N^N)`` into

* invariant scalar features ``s_i in R^d`` and
* equivariant vector features ``v_i in R^{3 x d}``,

aggregated into a fixed-size pocket context vector via masked mean pooling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from syntree.models.equivariant import (
    BatchedPaiNNLayer,
    RadialBasis,
    build_radius_graph,
)


class PocketEncoder(nn.Module):
    """PaiNN encoder over protein pocket atoms.

    Args:
        hidden_dim: feature dimension ``d``.
        num_layers: number of stacked PaiNN blocks.
        num_radial: Gaussian radial basis count.
        cutoff: interaction cutoff radius (Angstrom).
        max_atomic_number: size of the element embedding table.
        dropout: dropout applied to the final scalar features.

    The encoder optionally consumes per-atom formal charges (kcal/mol-side
    protonation states at pH 7.4, e.g. Asp/Glu -1, Arg/Lys +1). Charges are
    injected through a small MLP added to the element embedding so the
    network can learn salt-bridge formation that raw atomic numbers cannot
    express.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_radial: int = 20,
        cutoff: float = 5.0,
        max_atomic_number: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        self.hidden_dim = int(hidden_dim)
        self.cutoff = float(cutoff)

        self.atomic_embedding = nn.Embedding(max_atomic_number, hidden_dim)
        nn.init.xavier_uniform_(self.atomic_embedding.weight)

        # Formal-charge encoder: two-signed scalar -> hidden modulation.
        # tanh saturates at extreme charges so a misassigned +4 iron never
        # explodes the activation scale. Bias-free on purpose: a neutral
        # pocket (all charges 0) must reproduce the legacy charge-unaware
        # forward pass exactly, for any weight values.
        self.charge_proj = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        self.rbf = RadialBasis(num_radial, cutoff)
        self.layers = nn.ModuleList(
            [BatchedPaiNNLayer(hidden_dim, num_radial) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        pos: torch.Tensor,                       # [N, 3]
        atomic_nums: torch.Tensor,               # [N]
        batch: Optional[torch.Tensor] = None,    # [N]
        charges: Optional[torch.Tensor] = None,  # [N] formal charges (pH 7.4)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a (possibly batched) pocket point cloud.

        Args:
            pos: atom positions ``[N, 3]``.
            atomic_nums: element identifiers ``[N]``.
            batch: graph assignment ``[N]`` (optional; single graph assumed
                when ``None``).
            charges: per-atom formal charges at physiological pH ``[N]``
                (optional). ``None`` is treated as all-neutral, which keeps
                the encoder shape-compatible with legacy checkpoints that
                were trained before protonation-aware featurization.

        Returns:
            ``(scalars, vectors, pooled)`` where ``scalars`` is ``[N, d]``,
            ``vectors`` is ``[N, 3, d]``, and ``pooled`` is the mean-pooled
            ``[B, d]`` pocket context (``[1, d]`` for a single graph).
        """
        if pos.dim() != 2 or pos.size(1) != 3:
            raise ValueError(f"pos must be [N, 3], got {tuple(pos.shape)}")
        if atomic_nums.shape[0] != pos.size(0):
            raise ValueError(
                f"atomic_nums length {atomic_nums.shape[0]} != num atoms {pos.size(0)}"
            )
        if charges is not None and charges.numel() != pos.size(0):
            raise ValueError(
                f"charges length {charges.numel()} != num atoms {pos.size(0)}"
            )
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        edge_index = build_radius_graph(pos, batch, self.cutoff)
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_rbf = self.rbf(edge_dist)                     # [E, num_radial]

        scalar = self.atomic_embedding(atomic_nums)
        if charges is not None:
            charge_feat = charges.to(scalar.dtype).reshape(-1, 1)
            scalar = scalar + self.charge_proj(charge_feat)
        vector = torch.zeros(
            pos.size(0), 3, self.hidden_dim,
            device=pos.device, dtype=scalar.dtype,
        )

        for layer in self.layers:
            scalar, vector = layer(
                scalar, vector, edge_index, edge_vec, edge_dist, edge_rbf
            )

        scalar = self.layer_norm(self.dropout(scalar))

        num_graphs = int(batch.max().item()) + 1
        pooled = torch.zeros(
            num_graphs, self.hidden_dim, device=scalar.device, dtype=scalar.dtype
        )
        pooled.index_add_(0, batch, scalar)
        counts = torch.zeros(num_graphs, device=scalar.device, dtype=scalar.dtype)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=scalar.dtype))
        pooled = pooled / counts.clamp(min=1).unsqueeze(-1)

        return scalar, vector, pooled

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["PocketEncoder"]
