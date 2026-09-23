"""Heterogeneous SE(3)-equivariant protein-ligand encoder (BipartitePaiNN).

Bug report 2/3 fix ("ghost ligand" / "missing heterogeneous graph"): the
policy state must contain the *entire* intermediate ligand, not just the
single reacting handle atom. This module processes the joint system

* pocket atoms ``(X_P, Z_P, q_P)`` with pH-7.4 formal charges, and
* intermediate ligand atoms ``(X_L, Z_L, q_L)``,

in one PaiNN message-passing graph. The radius graph over the union
naturally contains all three chemically meaningful edge types:

* protein-protein edges (pocket packing),
* ligand-ligand edges (the scaffold the model is growing),
* protein-ligand interaction edges (contacts, clashes, salt bridges).

Node types are distinguished with a learned embedding, and the reacting
handle node is flagged so the readout can query it directly. With an empty
ligand block the module degenerates exactly to pocket-only encoding, which
keeps legacy pocket-only call sites working.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from syntree.models.equivariant import (
    BatchedPaiNNLayer,
    RadialBasis,
    build_radius_graph,
)


class BipartitePaiNN(nn.Module):
    """Joint protein-ligand PaiNN encoder.

    Args:
        hidden_dim: feature dimension ``d``.
        num_layers: number of stacked PaiNN blocks.
        num_radial: Gaussian radial basis count.
        cutoff: interaction cutoff radius (Angstrom).
        max_atomic_number: size of the element embedding table (shared
            between protein and ligand atoms).
        dropout: dropout applied to the final scalar features.
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

        # Shared element table + separate node-type and handle-flag tables.
        # Charges use a bias-free projection so a neutral system reproduces
        # the charge-unaware forward pass exactly.
        self.atomic_embedding = nn.Embedding(max_atomic_number, hidden_dim)
        nn.init.xavier_uniform_(self.atomic_embedding.weight)
        self.node_type_embedding = nn.Embedding(2, hidden_dim)
        nn.init.zeros_(self.node_type_embedding.weight)
        self.handle_flag_embedding = nn.Embedding(2, hidden_dim)
        nn.init.zeros_(self.handle_flag_embedding.weight)
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
        pocket_pos: torch.Tensor,                        # [Np, 3]
        pocket_z: torch.Tensor,                          # [Np]
        pocket_batch: torch.Tensor,                      # [Np]
        pocket_charge: Optional[torch.Tensor] = None,    # [Np]
        ligand_pos: Optional[torch.Tensor] = None,       # [Nl, 3]
        ligand_z: Optional[torch.Tensor] = None,         # [Nl]
        ligand_batch: Optional[torch.Tensor] = None,     # [Nl]
        ligand_charge: Optional[torch.Tensor] = None,     # [Nl]
        handle_node: Optional[torch.Tensor] = None,       # [Ntot] bool flag
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the joint (pocket + intermediate ligand) system.

        Args:
            pocket_pos/pocket_z/pocket_batch: pocket atoms.
            pocket_charge: per-atom pH-7.4 formal charges (optional).
            ligand_pos/ligand_z/ligand_batch: intermediate ligand heavy
                atoms in the same pocket-centred frame (optional; empty or
                ``None`` degenerates to pocket-only encoding).
            ligand_charge: per-atom RDKit formal charges (optional).
            handle_node: boolean per-union-atom flag marking the reacting
                handle node (optional).

        Returns:
            ``(scalars, vectors, pooled)`` over the union node set with
            ``scalars`` ``[Ntot, d]``, ``vectors`` ``[Ntot, 3, d]``, and
            ``pooled`` the mean-pooled ``[B, d]`` union context.
        """
        if pocket_pos.dim() != 2 or pocket_pos.size(1) != 3:
            raise ValueError(
                f"pocket_pos must be [Np, 3], got {tuple(pocket_pos.shape)}"
            )
        if pocket_z.shape[0] != pocket_pos.size(0):
            raise ValueError(
                f"pocket_z length {pocket_z.shape[0]} != num atoms {pocket_pos.size(0)}"
            )
        num_pocket = pocket_pos.size(0)

        if ligand_pos is None or ligand_pos.numel() == 0:
            num_ligand = 0
            all_pos = pocket_pos
            all_z = pocket_z
            all_batch = pocket_batch
            all_charge = pocket_charge
        else:
            if ligand_pos.dim() != 2 or ligand_pos.size(1) != 3:
                raise ValueError(
                    f"ligand_pos must be [Nl, 3], got {tuple(ligand_pos.shape)}"
                )
            if ligand_z is None or ligand_z.shape[0] != ligand_pos.size(0):
                raise ValueError(
                    "ligand_z must be provided with matching length for ligand_pos"
                )
            if ligand_batch is None:
                ligand_batch = torch.zeros(
                    ligand_pos.size(0), dtype=torch.long, device=ligand_pos.device
                )
            num_ligand = ligand_pos.size(0)
            all_pos = torch.cat([pocket_pos, ligand_pos], dim=0)
            all_z = torch.cat([pocket_z, ligand_z], dim=0)
            all_batch = torch.cat([pocket_batch, ligand_batch], dim=0)
            if pocket_charge is None and ligand_charge is None:
                all_charge = None
            else:
                pc = (
                    pocket_charge
                    if pocket_charge is not None
                    else torch.zeros(
                        num_pocket, device=pocket_pos.device, dtype=torch.float32
                    )
                )
                lc = (
                    ligand_charge
                    if ligand_charge is not None
                    else torch.zeros(
                        num_ligand, device=ligand_pos.device, dtype=torch.float32
                    )
                )
                all_charge = torch.cat([pc, lc], dim=0)

        num_graphs = int(all_batch.max().item()) + 1
        num_total = all_pos.size(0)

        # Node-type codes: 0 = pocket, 1 = ligand.
        node_type = torch.cat(
            [
                torch.zeros(num_pocket, dtype=torch.long, device=pocket_pos.device),
                torch.ones(num_ligand, dtype=torch.long, device=pocket_pos.device),
            ]
        ) if num_ligand > 0 else torch.zeros(
            num_total, dtype=torch.long, device=pocket_pos.device
        )

        edge_index = build_radius_graph(all_pos, all_batch, self.cutoff)
        row, col = edge_index
        edge_vec = all_pos[col] - all_pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_rbf = self.rbf(edge_dist)                     # [E, num_radial]

        scalar = self.atomic_embedding(all_z) + self.node_type_embedding(node_type)
        if handle_node is not None:
            if handle_node.shape[0] != num_total:
                raise ValueError(
                    f"handle_node flag length {handle_node.shape[0]} != "
                    f"num union atoms {num_total}"
                )
            scalar = scalar + self.handle_flag_embedding(
                handle_node.to(torch.long)
            )
        if all_charge is not None:
            if all_charge.numel() != num_total:
                raise ValueError(
                    f"charge length {all_charge.numel()} != num atoms {num_total}"
                )
            scalar = scalar + self.charge_proj(
                all_charge.to(scalar.dtype).reshape(-1, 1)
            )
        vector = torch.zeros(
            num_total, 3, self.hidden_dim,
            device=all_pos.device, dtype=scalar.dtype,
        )

        for layer in self.layers:
            scalar, vector = layer(
                scalar, vector, edge_index, edge_vec, edge_dist, edge_rbf
            )

        scalar = self.layer_norm(self.dropout(scalar))

        pooled = torch.zeros(
            num_graphs, self.hidden_dim, device=scalar.device, dtype=scalar.dtype
        )
        pooled.index_add_(0, all_batch, scalar)
        counts = torch.zeros(num_graphs, device=scalar.device, dtype=scalar.dtype)
        counts.index_add_(0, all_batch, torch.ones_like(all_batch, dtype=scalar.dtype))
        pooled = pooled / counts.clamp(min=1).unsqueeze(-1)

        return scalar, vector, pooled

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["BipartitePaiNN"]
