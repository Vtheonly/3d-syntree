"""End-to-end 3D-SynTree policy network.

Combines the SE(3)-equivariant pocket encoder, a cross-attention core that
fuses the ligand attachment-handle state with pocket context, the
grammar-masked synthon selection head, and the continuous torsion head:

    pi(a | P, M) = P(r | P, M) * P(B | r, P, M) * p(phi | B, r, P, M)

The forward pass consumes a (batched) PyG ``Data``/``Batch`` object with
``pocket_pos``, ``pocket_z``, ``pocket_batch`` and ``handle_features``
fields plus the synthon embedding table and reaction compatibility mask.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from syntree.chemistry.reactions import REACTION_FAMILY_NAMES
from syntree.models.pocket_encoder import PocketEncoder
from syntree.models.reaction_head import ReactionHead
from syntree.models.synthon_head import SynthonHead
from syntree.models.torsion_head import ContinuousTorsionHead

_HANDLE_CHEMICAL_FEATURE_DIM = 64
_HANDLE_FEATURE_DIM = 67
_HANDLE_POSITION_OFFSET = 64
_NUM_DISTANCE_RBF = 16


class SynTreePolicy(nn.Module):
    """Joint policy for structure-based synthon assembly.

    Components:
        1. SE(3)-equivariant pocket encoding (PaiNN).
        2. Ligand-handle / pocket cross-attention fusion.
        3. Reaction-masked synthon selection.
        4. Continuous dihedral torsion prediction (von Mises).

    Config keys (``config["model"]``): ``hidden_dim``,
    ``num_equivariant_layers``, ``num_radial_basis``, ``cutoff_radius``,
    ``synthon_embedding_dim``, ``num_attention_heads``, ``max_atomic_number``,
    ``dropout``.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("model", config)
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        num_layers = int(model_cfg.get("num_equivariant_layers", 4))
        num_radial = int(model_cfg.get("num_radial_basis", 20))
        cutoff = float(model_cfg.get("cutoff_radius", 5.0))
        synthon_dim = int(model_cfg.get("synthon_embedding_dim", hidden_dim))
        num_heads = int(model_cfg.get("num_attention_heads", 4))
        max_z = int(model_cfg.get("max_atomic_number", 100))
        dropout = float(model_cfg.get("dropout", 0.1))

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_attention_heads ({num_heads})"
            )
        if synthon_dim != hidden_dim:
            raise ValueError(
                f"synthon_embedding_dim ({synthon_dim}) must currently equal "
                f"hidden_dim ({hidden_dim}); set both to the same value."
            )

        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        # 1. Pocket encoder.
        self.pocket_encoder = PocketEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_radial=num_radial,
            cutoff=cutoff,
            max_atomic_number=max_z,
            dropout=dropout,
        )

        # 2. Ligand handle state -> query space.
        self.handle_proj = nn.Linear(_HANDLE_CHEMICAL_FEATURE_DIM, hidden_dim)
        self.handle_norm = nn.LayerNorm(hidden_dim)

        # Spatial conditioning uses rotation/translation-invariant radial
        # basis functions of the distance from each pocket atom to the
        # current reacting handle. This gives attention an explicit local
        # geometric signal without injecting frame-dependent absolute xyz.
        self.register_buffer(
            "distance_centers", torch.linspace(0.0, cutoff, _NUM_DISTANCE_RBF),
            persistent=False,
        )
        self.distance_width = max(cutoff / (_NUM_DISTANCE_RBF - 1), 0.25)
        self.spatial_distance_proj = nn.Sequential(
            nn.Linear(_NUM_DISTANCE_RBF, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        # Cross-attention: query = chemical handle state, key/value = spatially
        # enriched pocket atoms.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Reaction-family selection head.
        self.reaction_head = ReactionHead(
            hidden_dim=hidden_dim,
            num_families=len(REACTION_FAMILY_NAMES),
            dropout=dropout,
        )

        # 4. Synthon selection head.
        self.synthon_head = SynthonHead(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )

        # 5. Torsion head.
        self.torsion_head = ContinuousTorsionHead(hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    # Pocket encoding (exposed for generators / probing)
    # ------------------------------------------------------------------
    def encode_pocket(
        self,
        pocket_pos: torch.Tensor,
        pocket_z: torch.Tensor,
        pocket_batch: Optional[torch.Tensor] = None,
    ):
        return self.pocket_encoder(pocket_pos, pocket_z, pocket_batch)

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------
    def forward(
        self,
        batch_data,
        synthon_embeddings: torch.Tensor,
        synthon_compatibility_mask: Optional[torch.Tensor] = None,
        reaction_compatibility_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the full policy.

        Args:
            batch_data: PyG ``Data``/``Batch`` with ``pocket_pos`` ``[N,3]``,
                ``pocket_z`` ``[N]``, ``pocket_batch`` ``[N]`` (optional),
                and ``handle_features`` ``[B, 67]`` (or ``[67]``; first 64 are
                chemical features and final 3 are pocket-frame xyz).
            synthon_embeddings: ``[K, hidden_dim]`` catalog embeddings.
            synthon_compatibility_mask: ``[B, K]`` additive logit mask
                (``0`` legal, ``-1e9`` illegal).

        Returns:
            Dict with ``synthon_logits`` ``[B, K]``, ``synthon_log_probs``,
            ``torsion_mu`` ``[B]``, ``torsion_kappa`` ``[B]``, and
            ``pocket_context`` ``[B, d]``.
        """
        pocket_pos = batch_data.pocket_pos
        pocket_z = batch_data.pocket_z
        pocket_batch = getattr(batch_data, "pocket_batch", None)
        if pocket_batch is None:
            # PyG names the follow-batch vector after the attribute it
            # tracks (``pocket_pos`` -> ``pocket_pos_batch``).
            pocket_batch = getattr(batch_data, "pocket_pos_batch", None)
        if pocket_batch is None:
            pocket_batch = torch.zeros(
                pocket_pos.size(0), dtype=torch.long, device=pocket_pos.device
            )

        handle_features = batch_data.handle_features
        if handle_features.dim() == 1:
            if handle_features.size(0) == _HANDLE_FEATURE_DIM:
                # Single un-batched graph.
                handle_features = handle_features.unsqueeze(0)
            elif handle_features.size(0) % _HANDLE_FEATURE_DIM == 0:
                # PyG concatenated per-graph vectors: [B * 64] -> [B, 64].
                handle_features = handle_features.view(-1, _HANDLE_FEATURE_DIM)
            else:
                raise ValueError(
                    f"handle_features has unsupported size {handle_features.size(0)}; "
                    f"expected a multiple of {_HANDLE_FEATURE_DIM} (chemical 64 + xyz 3)"
                )
        if handle_features.size(-1) != _HANDLE_FEATURE_DIM:
            raise ValueError(
                f"handle_features must have last dim {_HANDLE_FEATURE_DIM}, "
                f"got {handle_features.size(-1)}"
            )
        batch_size = handle_features.size(0)
        chemical_handle_features = handle_features[..., :_HANDLE_CHEMICAL_FEATURE_DIM]
        handle_position = handle_features[..., _HANDLE_POSITION_OFFSET:_HANDLE_POSITION_OFFSET + 3]

        # 1. Pocket encoding.
        pocket_s, pocket_v, pocket_pooled = self.encode_pocket(
            pocket_pos, pocket_z, pocket_batch
        )
        num_graphs = pocket_pooled.size(0)
        if num_graphs != batch_size:
            raise ValueError(
                f"handle batch ({batch_size}) != pocket graphs ({num_graphs})"
            )

        # 2. Cross-attention: handle query attends to its graph's pocket atoms.
        query = self.handle_norm(self.handle_proj(chemical_handle_features)).unsqueeze(1)  # [B, 1, d]

        # Add handle-to-pocket geometry to each pocket atom before attention.
        # Distances are invariant under global translation/rotation, while the
        # resulting learned key enrichment makes the attention local to the
        # actual attachment point.
        per_atom_handle = handle_position[pocket_batch]
        handle_dist = torch.linalg.vector_norm(
            pocket_pos - per_atom_handle, dim=-1
        ).clamp(min=0.0, max=self.cutoff)
        centers = self.distance_centers.to(handle_dist)
        radial = torch.exp(
            -0.5 * ((handle_dist.unsqueeze(-1) - centers) / self.distance_width) ** 2
        )
        pocket_s = pocket_s + self.spatial_distance_proj(radial)

        # Chunked per-graph attention via MultiheadAttention requires [B, N, d];
        # build it by scattering pocket atoms into padded per-graph tensors.
        counts = torch.bincount(pocket_batch, minlength=num_graphs)
        max_atoms = int(counts.max().item()) if counts.numel() else 0
        padded = pocket_s.new_zeros(num_graphs, max(max_atoms, 1), self.hidden_dim)
        valid = pocket_s.new_zeros(num_graphs, max(max_atoms, 1))
        # Per-graph local index: node position within its own graph.
        graph_starts = torch.cumsum(counts, 0) - counts
        local_idx = torch.arange(pocket_pos.size(0), device=pocket_pos.device) - graph_starts[pocket_batch]
        padded[pocket_batch, local_idx] = pocket_s
        valid[pocket_batch, local_idx] = 1.0

        key_padding_mask = valid == 0.0  # [B, max_atoms] True = ignore
        context, _ = self.cross_attention(
            query, padded, padded, key_padding_mask=key_padding_mask
        )
        context = context.squeeze(1)  # [B, d]

        # 3. Synthon selection.
        reaction_logits, reaction_log_probs = self.reaction_head(
            context, reaction_compatibility_mask
        )

        stop_mask = getattr(batch_data, "stop_mask", None)
        if stop_mask is None:
            stop_mask = torch.zeros(batch_size, dtype=context.dtype, device=context.device)
        else:
            stop_mask = stop_mask.view(batch_size).to(context.dtype)

        logits, log_probs = self.synthon_head(
            context, synthon_embeddings, synthon_compatibility_mask, stop_mask=stop_mask
        )

        # 4. Torsion prediction.
        mu, kappa = self.torsion_head(context, pocket_v, pocket_batch)

        return {
            "reaction_logits": reaction_logits,
            "reaction_log_probs": reaction_log_probs,
            "synthon_logits": logits,
            "synthon_log_probs": log_probs,
            "torsion_mu": mu,
            "torsion_kappa": kappa,
            "pocket_context": context,
            "state_value": self.value_head(context).squeeze(-1),
        }

    # ------------------------------------------------------------------
    # Convenience inference helpers
    # ------------------------------------------------------------------
    def act(
        self,
        batch_data,
        synthon_embeddings: torch.Tensor,
        synthon_compatibility_mask: Optional[torch.Tensor] = None,
        reaction_compatibility_mask: Optional[torch.Tensor] = None,
        synthon_masks_by_reaction: Optional[torch.Tensor] = None,
        sample: bool = False,
        temperature: float = 1.0,
    ):
        """Select a reaction family, synthon, and dihedral."""
        out = self.forward(
            batch_data,
            synthon_embeddings,
            synthon_compatibility_mask,
            reaction_compatibility_mask,
        )

        reaction_logits = out["reaction_logits"]
        if sample:
            sampling_temperature = max(temperature, 1e-6)
            reaction_log_probs_for_action = F.log_softmax(
                reaction_logits / sampling_temperature, dim=-1
            )
            reaction_family_idx = torch.multinomial(
                reaction_log_probs_for_action.exp(), 1
            ).squeeze(-1)
        else:
            reaction_log_probs_for_action = F.log_softmax(reaction_logits, dim=-1)
            reaction_family_idx = reaction_logits.argmax(dim=-1)

        logits = out["synthon_logits"]
        if synthon_masks_by_reaction is not None:
            if synthon_masks_by_reaction.dim() != 3:
                raise ValueError(
                    "synthon_masks_by_reaction must be [B, F, K]"
                )
            b = reaction_family_idx.size(0)
            rows = torch.arange(b, device=logits.device)
            selected_mask = synthon_masks_by_reaction[
                rows, reaction_family_idx
            ]
            stop_mask = getattr(batch_data, "stop_mask", None)
            if stop_mask is None:
                stop_mask = torch.zeros(reaction_family_idx.size(0), device=logits.device)
            logits, _ = self.synthon_head(
                out["pocket_context"],
                synthon_embeddings,
                selected_mask,
                stop_mask=stop_mask,
            )

        if sample:
            sampling_temperature = max(temperature, 1e-6)
            action_log_probs_for_action = F.log_softmax(
                logits / sampling_temperature, dim=-1
            )
            action_idx = torch.multinomial(
                action_log_probs_for_action.exp(), 1
            ).squeeze(-1)
        else:
            action_log_probs_for_action = F.log_softmax(logits, dim=-1)
            action_idx = logits.argmax(dim=-1)
        stop = action_idx.eq(synthon_embeddings.size(0))
        synthon_idx = torch.where(
            stop,
            torch.full_like(action_idx, -1),
            action_idx,
        )
        reaction_log_probs = reaction_log_probs_for_action.gather(
            -1, reaction_family_idx.unsqueeze(-1)
        ).squeeze(-1)
        action_log_probs = action_log_probs_for_action.gather(
            -1, action_idx.unsqueeze(-1)
        ).squeeze(-1)
        joint_log_prob = torch.where(
            stop,
            action_log_probs,
            reaction_log_probs + action_log_probs,
        )

        phi = ContinuousTorsionHead.sample(
            out["torsion_mu"], out["torsion_kappa"]
        )
        return {
            "reaction_family_idx": reaction_family_idx,
            "synthon_idx": synthon_idx,
            "action_idx": action_idx,
            "stop": stop,
            "dihedral": phi,
            "joint_log_prob": joint_log_prob,
            "synthon_log_prob": action_log_probs,
            "state_value": out["state_value"],
            "reaction_logits": reaction_logits,
            "logits": logits,
            "torsion_mu": out["torsion_mu"],
            "torsion_kappa": out["torsion_kappa"],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["SynTreePolicy"]
