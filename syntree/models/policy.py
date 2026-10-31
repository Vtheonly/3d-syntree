"""End-to-end 3D-SynTree policy network.

Combines the heterogeneous SE(3)-equivariant pocket+ligand encoder
(:class:`~syntree.models.bipartite.BipartitePaiNN`), a cross-attention core
that fuses the ligand attachment-handle state with the *entire* joint
protein-ligand graph, the grammar-masked synthon selection head, and the
synthon-conditioned continuous torsion head:

    pi(a | P, M) = P(r | P, M) * P(B | r, P, M) * p(phi | B, r, P, M)

Bug-report 2/3 fixes baked into this module:

* **Ghost ligand eliminated** - the input graph contains every intermediate
  ligand atom, not just the reacting handle, so the policy can see its own
  self-collisions and physicochemical budget.
* **Torsion conditioned on the selected synthon** (Flaw 2) - the dihedral
  head consumes the embedding of the synthon that will actually be attached.
* **STOP sees global ligand features** (Flaw 3) - molecular weight,
  heavy-atom count and cavity-occupation ratio feed both the query and a
  dedicated termination bias.
* **No Franken-tensors** (Tell 1 / #6 ARCH-02) - handle chemical features
  (64-dim) and the handle position (3-dim) travel as separate fields
  (``handle_features`` + ``handle_pos``). The legacy 67-dim
  chemical+coordinate concatenation is a *hard error*: the policy rejects
  it via the frozen V2 state contract instead of splitting it.

The forward pass consumes a PyG ``Data``/``Batch`` object validated
against the frozen V2 state schema
(:mod:`syntree.core.schema`, issue #6): all 13 canonical tensors
(``pocket_pos``/``pocket_z``/``pocket_charge``/``pocket_batch``,
``ligand_pos``/``ligand_z``/``ligand_charge``/``ligand_batch``,
``handle_features`` ``[B, 64]``, ``handle_pos`` ``[B, 3]``,
``global_features`` ``[B, 4]``, ``stop_mask`` ``[B]``) must be present
with exact dtypes and shapes - there are no implicit batch-vector
fallbacks, no silent zeros defaults and no legacy shape tolerance.
Optional non-contract context: ``handle_nodes`` ``[B]`` (index of the
reacting handle within its graph's ligand nodes, ``-1`` when absent).
Plus the synthon embedding table and reaction compatibility mask.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from syntree.chemistry.reactions import REACTION_FAMILY_NAMES
from syntree.core.schema import (
    GLOBAL_FEATURE_DIM,
    HANDLE_FEATURE_DIM,
    StateTensorContract,
    validate_state_batch,
)
from syntree.models.bipartite import BipartitePaiNN
from syntree.models.reaction_head import ReactionHead
from syntree.models.synthon_head import SynthonHead
from syntree.models.torsion_head import ContinuousTorsionHead

# Re-exported for backwards-compatible imports of the dimension contract.
_HANDLE_CHEMICAL_FEATURE_DIM = HANDLE_FEATURE_DIM
_NUM_DISTANCE_RBF = 16


class SynTreePolicy(nn.Module):
    """Joint policy for structure-based synthon assembly.

    Components:
        1. Heterogeneous SE(3)-equivariant pocket+ligand encoding
           (BipartitePaiNN) with the handle node explicitly flagged.
        2. Handle / joint-graph cross-attention fusion with global
           ligand-size features.
        3. Reaction-masked synthon selection (with a globally informed
           STOP logit).
        4. Synthon-conditioned continuous dihedral torsion prediction
           (von Mises).

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
        # OOM fix (T4): recompute PaiNN layers during backward instead of
        # retaining all 8 layers' message tensors (see BipartitePaiNN).
        grad_ckpt = bool(model_cfg.get("gradient_checkpointing", False))

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

        # 1. Joint heterogeneous pocket+ligand encoder.
        self.joint_encoder = BipartitePaiNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_radial=num_radial,
            cutoff=cutoff,
            max_atomic_number=max_z,
            dropout=dropout,
            gradient_checkpointing=grad_ckpt,
        )

        # 2. Ligand handle state -> query space. Chemical handle features
        #    plus the global ligand-size features; the joint-encoded handle
        #    node is added on top when the ligand graph is present.
        self.handle_proj = nn.Linear(
            _HANDLE_CHEMICAL_FEATURE_DIM + GLOBAL_FEATURE_DIM, hidden_dim
        )
        self.handle_norm = nn.LayerNorm(hidden_dim)
        self.handle_node_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.handle_node_proj.weight)

        # Global ligand-size features -> direct termination bias (Flaw 3).
        self.global_stop_proj = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Spatial conditioning uses rotation/translation-invariant radial
        # basis functions of the distance from each joint-graph atom to the
        # current reacting handle.
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

        # Cross-attention: query = chemical handle state, key/value =
        # spatially enriched joint (pocket + ligand) atoms.
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

        # 5. Synthon-conditioned torsion head.
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
        pocket_batch: torch.Tensor = None,
        pocket_charge: Optional[torch.Tensor] = None,
    ):
        """Pocket-only convenience wrapper around the joint encoder."""
        if pocket_batch is None:
            pocket_batch = torch.zeros(
                pocket_pos.size(0), dtype=torch.long, device=pocket_pos.device
            )
        return self.joint_encoder(
            pocket_pos, pocket_z, pocket_batch, pocket_charge
        )

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------
    def forward(
        self,
        batch_data,
        synthon_embeddings: torch.Tensor,
        synthon_compatibility_mask: Optional[torch.Tensor] = None,
        reaction_compatibility_mask: Optional[torch.Tensor] = None,
        synthon_embedding_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the full policy.

        Args:
            batch_data: PyG ``Data``/``Batch`` object conforming to the
                frozen V2 state schema
                (:func:`syntree.core.schema.validate_state_batch`, issue
                #6): all 13 canonical tensors with exact dtypes/shapes.
                Legacy partial states and 67-dim Franken tensors raise
                ``TensorContractError`` (a ``ValueError``). Optional
                context: ``handle_nodes`` ``[B]`` (index of the reacting
                handle within its graph's ligand nodes, ``-1`` absent).
            synthon_embeddings: ``[K, hidden_dim]`` catalog embeddings.
            synthon_compatibility_mask: ``[B, K]`` additive logit mask
                (``0`` legal, ``-1e9`` illegal).
            synthon_embedding_input: ``[B, hidden_dim]`` teacher-forcing
                synthon embedding for the torsion head (training); the
                learned null synthon is used when omitted.

        Returns:
            Dict with ``synthon_logits`` ``[B, K]``, ``synthon_log_probs``,
                ``torsion_mu`` ``[B]``, ``torsion_kappa`` ``[B]``,
                ``pocket_context`` ``[B, d]``, ``state_value`` ``[B]`` and
                the joint graph tensors used by :meth:`act` to re-run the
                torsion head after action selection.
        """
        # Fail-closed contract check (#6 ARCH-02): every canonical field
        # present with exact dtype/shape; no fallbacks, no legacy 67-dim
        # tolerance, no implicit zeros defaults.
        state: StateTensorContract = validate_state_batch(
            batch_data, where="SynTreePolicy.forward"
        )

        pocket_pos = state.pocket_pos
        pocket_z = state.pocket_z
        pocket_batch = state.pocket_batch
        pocket_charge = state.pocket_charge
        ligand_pos = state.ligand_pos
        ligand_z = state.ligand_z
        ligand_batch = state.ligand_batch
        ligand_charge = state.ligand_charge
        chemical_handle_features = state.handle_features
        handle_position = state.handle_pos.to(pocket_pos.dtype)
        global_features = state.global_features
        batch_size = state.batch_size
        has_ligand = state.num_ligand > 0

        handle_nodes = getattr(batch_data, "handle_nodes", None)

        # ------------------------------------------------------------------
        # 1. Joint pocket+ligand encoding (the ghost-ligand fix).
        # ------------------------------------------------------------------
        num_pocket = state.num_pocket
        handle_flag = None
        handle_union_index = None  # union-graph index of each graph's handle node (-1 -> 0)
        valid = None
        if has_ligand:
            num_ligand = state.num_ligand
            num_total = num_pocket + num_ligand
            handle_flag = torch.zeros(
                num_total, dtype=torch.long, device=pocket_pos.device
            )
            if handle_nodes is not None:
                handle_nodes = handle_nodes.view(batch_size).to(torch.long)
                # Local ligand index -> global union index.
                lig_counts = torch.bincount(
                    ligand_batch, minlength=batch_size
                )
                lig_starts = torch.cumsum(lig_counts, 0) - lig_counts
                graphs = torch.arange(batch_size, device=ligand_batch.device)
                local_union = lig_starts[graphs] + handle_nodes.clamp(min=0)
                valid = (handle_nodes >= 0) & (handle_nodes < lig_counts[graphs])
                handle_union_index = torch.where(
                    valid, local_union, torch.zeros_like(local_union)
                )
                handle_flag = handle_flag.index_put(
                    ((handle_union_index + num_pocket).to(handle_flag.device),),
                    valid.to(handle_flag.dtype).to(handle_flag.device),
                )
        else:
            ligand_pos = ligand_z = ligand_batch = ligand_charge = None

        joint_s, joint_v, joint_pooled = self.joint_encoder(
            pocket_pos,
            pocket_z,
            pocket_batch,
            pocket_charge,
            ligand_pos,
            ligand_z,
            ligand_batch,
            ligand_charge,
            handle_flag,
        )
        union_batch = (
            torch.cat([pocket_batch, ligand_batch], dim=0)
            if ligand_batch is not None
            else pocket_batch
        )
        num_graphs = joint_pooled.size(0)
        if num_graphs != batch_size:
            raise ValueError(
                f"handle batch ({batch_size}) != pocket graphs ({num_graphs})"
            )

        # ------------------------------------------------------------------
        # 2. Query: chemical handle state + global features + the
        #    joint-encoded handle node (when a ligand graph exists).
        # ------------------------------------------------------------------
        query_input = torch.cat(
            [chemical_handle_features.to(joint_s.dtype), global_features.to(joint_s.dtype)],
            dim=-1,
        )
        query = self.handle_norm(self.handle_proj(query_input))  # [B, d]

        # FIX F-02/F-06: Add projected handle node BEFORE unsqueezing to [B, 1, d]
        if handle_union_index is not None and has_ligand:
            handle_node_scalar = joint_s[
                (handle_union_index + num_pocket).to(joint_s.device)
            ]  # [B, d]
            valid_mask = (
                valid.unsqueeze(-1) if valid is not None else torch.ones_like(query)
            )
            query = query + torch.where(
                valid_mask,
                self.handle_node_proj(handle_node_scalar),
                torch.zeros_like(query),
            )
        query = query.unsqueeze(1)  # Strict shape: [B, 1, d]

        # Add handle-to-graph geometry to every joint atom before attention.
        # Distances are invariant under global translation/rotation, and the
        # learned key enrichment makes attention local to the attachment
        # point.
        all_pos = (
            torch.cat([pocket_pos, ligand_pos], dim=0)
            if ligand_pos is not None
            else pocket_pos
        )
        per_atom_handle = handle_position[union_batch]
        handle_dist = torch.linalg.vector_norm(
            all_pos - per_atom_handle, dim=-1
        ).clamp(min=0.0, max=self.cutoff)
        centers = self.distance_centers.to(handle_dist)
        radial = torch.exp(
            -0.5 * ((handle_dist.unsqueeze(-1) - centers) / self.distance_width) ** 2
        )
        joint_s = joint_s + self.spatial_distance_proj(radial)

        # Chunked per-graph attention via MultiheadAttention requires
        # [B, N, d]; build it by scattering joint atoms into padded
        # per-graph tensors. ``union_batch`` is NOT sorted (pocket nodes of
        # all graphs precede ligand nodes of all graphs after the
        # concatenation), so per-graph local indices are computed via a
        # stable sort.
        counts = torch.bincount(union_batch, minlength=num_graphs)
        max_atoms = int(counts.max().item()) if counts.numel() else 0
        padded = joint_s.new_zeros(num_graphs, max(max_atoms, 1), self.hidden_dim)
        valid = joint_s.new_zeros(num_graphs, max(max_atoms, 1))
        graph_starts = torch.cumsum(counts, 0) - counts
        perm = torch.argsort(union_batch, stable=True)
        local_idx_sorted = (
            torch.arange(union_batch.numel(), device=joint_s.device)
            - graph_starts[union_batch[perm]]
        )
        local_idx = torch.empty_like(local_idx_sorted)
        local_idx[perm] = local_idx_sorted
        padded[union_batch, local_idx] = joint_s
        valid[union_batch, local_idx] = 1.0

        key_padding_mask = valid == 0.0  # [B, max_atoms] True = ignore
        context, _ = self.cross_attention(
            query, padded, padded, key_padding_mask=key_padding_mask
        )
        context = context.squeeze(1)  # [B, d]

        # ------------------------------------------------------------------
        # 3. Reaction + synthon selection with a globally informed STOP.
        # ------------------------------------------------------------------
        reaction_logits, reaction_log_probs = self.reaction_head(
            context, reaction_compatibility_mask
        )

        # Validated [B] additive STOP mask (schema contract, #6).
        stop_mask = state.stop_mask.to(context.dtype)

        stop_logit_bias = self.global_stop_proj(
            global_features.to(context.dtype)
        ).squeeze(-1)
        logits, log_probs = self.synthon_head(
            context,
            synthon_embeddings,
            synthon_compatibility_mask,
            stop_mask=stop_mask,
            stop_logit_bias=stop_logit_bias,
        )

        # ------------------------------------------------------------------
        # 4. Synthon-conditioned torsion prediction (Flaw 2).
        # ------------------------------------------------------------------
        mu, kappa = self.torsion_head(
            context, joint_v, union_batch, synthon_embedding_input
        )

        return {
            "reaction_logits": reaction_logits,
            "reaction_log_probs": reaction_log_probs,
            "synthon_logits": logits,
            "synthon_log_probs": log_probs,
            "torsion_mu": mu,
            "torsion_kappa": kappa,
            "pocket_context": context,
            "state_value": self.value_head(context).squeeze(-1),
            # Exposed for act() to re-run the torsion head after action
            # selection without recomputing the whole forward pass.
            "joint_vectors": joint_v,
            "union_batch": union_batch,
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
        """Select a reaction family, synthon, and synthon-conditioned dihedral."""
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
            global_features = getattr(batch_data, "global_features", None)
            if global_features is not None:
                global_features = global_features.view(b, -1).to(logits.dtype)
                stop_logit_bias = self.global_stop_proj(global_features).squeeze(-1)
            else:
                stop_logit_bias = None
            logits, _ = self.synthon_head(
                out["pocket_context"],
                synthon_embeddings,
                selected_mask,
                stop_mask=stop_mask,
                stop_logit_bias=stop_logit_bias,
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

        # Re-run the torsion head conditioned on the SELECTED synthon
        # (Flaw 2): the dihedral must depend on what is being attached.
        selected_synthon_emb = torch.where(
            stop.unsqueeze(-1),
            self.torsion_head.null_synthon.unsqueeze(0).expand(
                action_idx.size(0), -1
            ),
            synthon_embeddings[action_idx.clamp(max=synthon_embeddings.size(0) - 1)],
        )
        mu, kappa = self.torsion_head(
            out["pocket_context"],
            out["joint_vectors"],
            out["union_batch"],
            selected_synthon_emb,
        )
        phi = ContinuousTorsionHead.sample(mu, kappa)
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
            "torsion_mu": mu,
            "torsion_kappa": kappa,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["SynTreePolicy", "GLOBAL_FEATURE_DIM"]
