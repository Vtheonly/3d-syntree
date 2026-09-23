"""Gym-like molecular assembly MDP (bug report 3, upgrade 1).

The step logic that used to be scattered across the generator, conformer
engine and trainer is wrapped in one rigorous environment:

* ``reset``    - load a pocket, embed + place the seed synthon at a
                 hotspot-conditioned position, return the first joint
                 protein-ligand observation.
* ``legal_action_masks`` - the deterministic chemistry grammar for the
                 current state (reaction families, synthons, STOP gate).
* ``step``      - execute one assembly action: reaction validation and
                 execution via RDKit SMARTS, constrained-embedding scaffold
                 anchoring, resonance-aware synthon-conditioned dihedral
                 rotation; return the updated joint observation ``S_{t+1}``,
                 a dense physics reward ``r_t``, the ``done`` flag and an
                 info dict.

The environment is policy-free: actions come from the caller (the policy or
a test), which makes it directly usable for PPO rollouts, offline
evaluation and unit tests of the chemistry itself.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data

from syntree.chemistry.conformer import vdw_radius
from syntree.chemistry.hotspots import PocketHotspotFeaturizer
from syntree.chemistry.reactions import REACTION_FAMILY_NAMES
from syntree.data.featurizer import MolecularFeaturizer

logger = logging.getLogger(__name__)


@dataclass
class AssemblyAction:
    """One assembly decision emitted by the policy.

    Attributes:
        reaction_family_idx: index into ``REACTION_FAMILY_NAMES``.
        synthon_idx: catalog index of the synthon to attach; ``-1`` encodes
            the STOP action.
        dihedral_rad: requested torsion around the new junction bond
            (radians, ``[-pi, pi)``).
        action_idx: raw synthon-head output (``len(catalog)`` = STOP);
            kept for PPO log-prob alignment.
    """

    reaction_family_idx: int
    synthon_idx: int
    dihedral_rad: float
    action_idx: int = -1

    @property
    def is_stop(self) -> bool:
        return self.synthon_idx < 0


@dataclass
class EnvInfo:
    """Per-step diagnostics returned by :meth:`MolecularAssemblyEnv.step`."""

    reaction: Optional[str] = None
    reaction_family: Optional[str] = None
    handle: Optional[str] = None
    synthon_id: Optional[str] = None
    synthon_smiles: Optional[str] = None
    dihedral_applied_rad: Optional[float] = None
    junction_angle_rad: Optional[float] = None
    torsion_mode: Optional[str] = None
    rotated_bond: Optional[Tuple[int, int]] = None
    clash: float = 0.0
    contact_energy: float = 0.0
    stop_reason: Optional[str] = None
    new_atoms: int = 0
    terminated_by: Optional[str] = None

    def as_dict(self) -> Dict:
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                out[key] = list(value)
            else:
                out[key] = value
        return out


class MolecularAssemblyEnv:
    """Rigorous MDP environment wrapping RDKit and the Enamine catalog.

    The environment owns every environment-side concern (pocket, seed,
    chemistry grammar, reaction execution, 3D assembly, physics scoring)
    and is entirely agnostic of the policy network.
    """

    def __init__(
        self,
        rxn_engine,
        conformer_engine,
        catalog,
        validator,
        config: dict,
        device: torch.device = torch.device("cpu"),
        dense_reward: bool = True,
    ):
        self.rxn_engine = rxn_engine
        self.conformer_engine = conformer_engine
        self.catalog = catalog
        self.validator = validator
        self.config = config
        self.device = device
        self.dense_reward = bool(dense_reward)

        self.max_steps = int(
            config.get("data", {}).get("max_steps_per_molecule", 3)
        )
        self.terminal_cap_min_mw = float(
            config.get("data", {}).get("terminal_cap_min_mw", 250.0)
        )
        # Deterministic seed picking shared with SBDDGenerator semantics.
        self.seed_rng = np.random.default_rng(
            int(config.get("system", {}).get("seed", 42))
        )

        # Rollout state.
        self.pocket_mol: Optional[Chem.Mol] = None
        self.pocket_pdb_path: Optional[str] = None
        self.current_mol: Optional[Chem.Mol] = None
        self.recipe: List[Dict] = []
        self.step_count = 0
        self.horizon = self.max_steps
        self.done = False
        self.total_clash = 0.0
        self._prev_contact_energy = 0.0

        # Cached pocket tensors.
        self._pocket_tensors: Dict[str, torch.Tensor] = {}
        self._pocket_coords_np: Optional[np.ndarray] = None
        self._pocket_vdw: Optional[np.ndarray] = None
        self._pocket_volume: float = 0.0
        self._pocket_centroid: Optional[torch.Tensor] = None
        self._hotspots = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(
        self,
        pocket_pdb_path: Optional[str] = None,
        pocket_mol: Optional[Chem.Mol] = None,
        seed_synthon_idx: Optional[int] = None,
        max_steps: Optional[int] = None,
        optimize_torsion_grid: bool = False,
    ) -> Data:
        """Start a new episode: load pocket, place the seed, observe.

        Either ``pocket_pdb_path`` or ``pocket_mol`` must be provided.
        """
        if pocket_mol is None:
            if not pocket_pdb_path:
                raise ValueError("reset requires pocket_pdb_path or pocket_mol")
            try:
                pocket_mol = Chem.MolFromPDBFile(pocket_pdb_path, removeHs=False)
            except OSError as exc:
                raise ValueError(
                    f"Cannot read pocket file {pocket_pdb_path!r}: {exc}"
                ) from exc
            if pocket_mol is None or pocket_mol.GetNumAtoms() == 0:
                raise ValueError(f"Cannot read pocket from {pocket_pdb_path}")
        elif pocket_pdb_path is None:
            pocket_pdb_path = "<in-memory>"

        self.pocket_mol = pocket_mol
        self.pocket_pdb_path = pocket_pdb_path
        self.step_count = 0
        self.done = False
        self.total_clash = 0.0
        self._prev_contact_energy = 0.0
        self.horizon = int(max_steps) if max_steps is not None else self.max_steps
        self._optimize_torsion_grid = bool(optimize_torsion_grid)
        # A non-positive horizon means "seed only": no growth steps are
        # legal, so the episode is born terminal.
        self.done = self.horizon <= 0

        feats = MolecularFeaturizer.featurize_pocket(pocket_mol)
        self._pocket_tensors = {
            "pocket_pos": feats["pocket_pos"].to(self.device),
            "pocket_z": feats["pocket_z"].to(self.device),
            "pocket_charge": feats["pocket_charge"].to(self.device),
        }
        self._pocket_coords_np = feats["pocket_pos"].numpy()
        self._pocket_vdw = np.array(
            [vdw_radius(int(z)) for z in feats["pocket_z"].tolist()],
            dtype=np.float64,
        )
        self._pocket_volume = MolecularFeaturizer.vdw_sphere_volume(pocket_mol)
        self._pocket_centroid = feats["centroid"].view(-1)
        self._hotspots = PocketHotspotFeaturizer().featurize(
            pocket_mol, center=feats["centroid"].view(-1)
        )

        # Seed fragment.
        if seed_synthon_idx is None:
            seed_synthon_idx = self._pick_seed_synthon()
        seed_synthon_idx = int(seed_synthon_idx)
        current = self.catalog.get_mol(seed_synthon_idx, explicit_hs=False)
        current = self.conformer_engine.embed_product(current)
        if current is None:
            raise RuntimeError("Failed to embed the seed synthon in 3D.")

        seed_handles = self.rxn_engine.detect_handles(current)
        seed_handle_type = seed_handles[0].handle_type if seed_handles else None
        # Place the seed at a contact-rich position near the pocket interface.
        self._place_seed_in_pocket(
            current,
            seed_handle_type=seed_handle_type,
            seed_handle_atom_index=(
                seed_handles[0].primary_atom if seed_handles else None
            ),
        )
        self.current_mol = Chem.AddHs(current, addCoords=True)
        self.recipe = [
            {
                "step": 0,
                "action": "seed",
                "synthon_id": self.catalog.get_id(seed_synthon_idx),
                "smiles": self.catalog.get_smiles(seed_synthon_idx),
                "handle": None,
            }
        ]
        return self.observe()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observe(self) -> Data:
        """Build the policy-ready joint observation for the current state."""
        if self.current_mol is None:
            raise RuntimeError("call reset() before observe()")

        handles = self.rxn_engine.rank_handles(self.current_mol)
        target_handle = handles[0] if handles else None
        allow_stop = not self._below_cap()

        pocket = self._pocket_tensors
        # The intermediate ligand is ALWAYS part of the state (ghost-ligand
        # fix); only the handle-specific fields degenerate when no reactive
        # handle remains.
        lig = MolecularFeaturizer.featurize_ligand(
            self.current_mol, center=self._pocket_centroid
        )
        if target_handle is None:
            handle_feat = torch.zeros(1, 64, device=self.device)
            handle_pos = torch.zeros(1, 3, device=self.device)
            handle_node = -1
        else:
            handle_feat = MolecularFeaturizer.featurize_handle(
                self.current_mol,
                target_handle.atom_indices,
                target_handle.handle_type,
            ).unsqueeze(0).to(self.device)
            handle_pos = MolecularFeaturizer.featurize_handle_position(
                self.current_mol,
                target_handle.atom_indices,
                reference_center=self._pocket_centroid,
            ).unsqueeze(0).to(self.device)
            handle_node = lig["heavy_atom_map"].get(
                int(target_handle.primary_atom), -1
            )

        global_feats = MolecularFeaturizer.ligand_global_features(
            self.current_mol, pocket_volume=self._pocket_volume
        ).unsqueeze(0).to(self.device)

        return Data(
            pocket_pos=pocket["pocket_pos"],
            pocket_z=pocket["pocket_z"],
            pocket_charge=pocket["pocket_charge"],
            pocket_batch=torch.zeros(
                pocket["pocket_pos"].size(0), dtype=torch.long, device=self.device
            ),
            ligand_pos=lig["ligand_pos"].to(self.device),
            ligand_z=lig["ligand_z"].to(self.device),
            ligand_charge=lig["ligand_charge"].to(self.device),
            ligand_batch=torch.zeros(
                lig["ligand_pos"].size(0), dtype=torch.long, device=self.device
            ),
            handle_features=handle_feat,
            handle_pos=handle_pos,
            handle_nodes=torch.tensor(
                [handle_node], dtype=torch.long, device=self.device
            ),
            global_features=global_feats,
            stop_mask=torch.tensor(
                0.0 if allow_stop else -1e9,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------
    def _below_cap(self) -> bool:
        if self.current_mol is None:
            return True
        current_mw = float(
            Descriptors.MolWt(Chem.RemoveHs(Chem.Mol(self.current_mol)))
        )
        return current_mw < self.terminal_cap_min_mw

    def legal_action_masks(self) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Deterministic chemistry grammar for the current state.

        Returns:
            ``(reaction_mask, synthon_masks, has_handle)`` where
            ``reaction_mask`` is ``[1, F]`` additive and ``synthon_masks``
            is ``[1, F, K]`` additive (STOP is governed by the observation's
            ``stop_mask``, built from the molecular-weight cap).
        """
        handles = self.rxn_engine.rank_handles(self.current_mol)
        if not handles or self.done:
            empty_r = torch.full(
                (1, len(REACTION_FAMILY_NAMES)), -1e9, device=self.device
            )
            empty_s = torch.full(
                (1, len(REACTION_FAMILY_NAMES), len(self.catalog)), -1e9,
                device=self.device,
            )
            return empty_r, empty_s, False

        target_handle = handles[0]
        if not target_handle.allowed_reactions:
            empty_r = torch.full(
                (1, len(REACTION_FAMILY_NAMES)), -1e9, device=self.device
            )
            empty_s = torch.full(
                (1, len(REACTION_FAMILY_NAMES), len(self.catalog)), -1e9,
                device=self.device,
            )
            return empty_r, empty_s, False

        below_cap = self._below_cap()
        allow_stop = not below_cap
        require_remaining_handle = (self.step_count + 1) < self.horizon and below_cap

        reaction_mask = self.catalog.get_reaction_family_compatibility_mask(
            device=self.device,
            core_handle=target_handle.handle_type,
            require_remaining_handle=require_remaining_handle,
            allow_terminal=allow_stop,
        ).unsqueeze(0)
        synthon_masks = self.catalog.get_reaction_family_masks(
            device=self.device,
            core_handle=target_handle.handle_type,
            require_remaining_handle=require_remaining_handle,
            allow_terminal=allow_stop,
        ).unsqueeze(0)
        return reaction_mask, synthon_masks, True

    def _preferred_reaction(
        self, reaction_family: str, core_handle: str
    ) -> Optional[str]:
        """Concrete RDKit backend for a reaction family (None = illegal)."""
        from syntree.chemistry.reactions import (
            REACTION_FAMILY_MEMBERS,
            REACTION_SIDES,
        )

        members = REACTION_FAMILY_MEMBERS.get(reaction_family, ())
        for reaction in members:
            if core_handle in REACTION_SIDES[reaction]:
                return reaction
        return None

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------
    def step(
        self,
        action: AssemblyAction,
    ) -> Tuple[Data, float, bool, EnvInfo]:
        """Execute one assembly action.

        Returns:
            ``(observation, reward, done, info)``. The observation is the
            joint protein-ligand state ``S_{t+1}`` ready for the policy;
            ``reward`` is a dense physics shaping reward (contact-energy
            improvement minus clash increase) - terminal reward components
            (docking, QED, ...) are added by the caller; ``done`` is True
            when the episode ended (STOP, dead chemistry, or horizon).
        """
        if self.current_mol is None or self.done:
            raise RuntimeError("step() called before reset() or after done")

        info = EnvInfo(clash=self.total_clash)
        self.step_count += 1
        step = self.step_count

        # Horizon guard BEFORE executing: a step beyond the horizon is
        # illegal, not an extra growth step.
        if step > self.horizon:
            self.done = True
            info.stop_reason = "horizon"
            info.terminated_by = "horizon"
            return self.observe(), 0.0, True, info

        handles = self.rxn_engine.rank_handles(self.current_mol)
        if not handles:
            self.done = True
            info.stop_reason = "no_handles"
            info.terminated_by = "no_handles"
            return self.observe(), 0.0, True, info
        target_handle = handles[0]

        if action.is_stop:
            # STOP is only legal above the molecular-weight cap; the
            # observation's stop_mask already encodes this for the policy.
            if self._below_cap():
                logger.debug(
                    "STOP requested below the MW cap; continuing growth instead."
                )
            else:
                self.done = True
                info.stop_reason = "policy_stop"
                info.terminated_by = "policy_stop"
                self.recipe.append(
                    {"step": step, "action": "stop", "reason": "policy_stop"}
                )
                return self.observe(), 0.0, True, info

        if not target_handle.allowed_reactions:
            self.done = True
            info.stop_reason = "no_legal_reactions"
            info.terminated_by = "no_legal_reactions"
            return self.observe(), 0.0, True, info

        below_cap = self._below_cap()
        allow_stop = not below_cap
        require_remaining_handle = step < self.horizon and below_cap

        family = (
            REACTION_FAMILY_NAMES[action.reaction_family_idx]
            if 0 <= action.reaction_family_idx < len(REACTION_FAMILY_NAMES)
            else None
        )
        if family is None:
            self.done = True
            info.stop_reason = "invalid_family"
            info.terminated_by = "invalid_family"
            return self.observe(), 0.0, True, info
        chosen_rxn = self._preferred_reaction(family, target_handle.handle_type)
        if chosen_rxn is None:
            self.done = True
            info.stop_reason = "family_without_backend"
            info.terminated_by = "family_without_backend"
            return self.observe(), 0.0, True, info

        synthon_idx = int(action.synthon_idx)
        if not (0 <= synthon_idx < len(self.catalog)):
            self.done = True
            info.stop_reason = "invalid_synthon"
            info.terminated_by = "invalid_synthon"
            return self.observe(), 0.0, True, info

        # Deterministic chemistry guard: the chosen synthon must be legal
        # for the selected family and the current handle.
        selected_mask = self.catalog.get_reaction_family_mask(
            family,
            device=self.device,
            core_handle=target_handle.handle_type,
            require_remaining_handle=require_remaining_handle,
            allow_terminal=allow_stop,
        )
        if selected_mask[synthon_idx].item() < -1e8:
            logger.warning(
                "Action violates the deterministic chemistry mask; ending rollout."
            )
            self.done = True
            info.stop_reason = "chemistry_mask_violation"
            info.terminated_by = "chemistry_mask_violation"
            return self.observe(), 0.0, True, info

        synthon_mol = self.catalog.get_mol(synthon_idx, explicit_hs=False)

        # Chemical execution on implicit-H copies (heavy-atom indices are
        # preserved so the scaffold conformer stays valid).
        core_noH = Chem.RemoveHs(Chem.Mol(self.current_mol))
        result = self.rxn_engine.apply_reaction(core_noH, synthon_mol, chosen_rxn)
        if result is None:
            self.done = True
            info.stop_reason = "reaction_failed"
            info.terminated_by = "reaction_failed"
            return self.observe(), 0.0, True, info

        # 3D assembly: constrained-embedding scaffold anchoring +
        # resonance-aware synthon-conditioned torsion.
        product = self.conformer_engine.embed_product(
            result.product,
            core_atom_map=result.core_atom_map,
            core_reference=core_noH,
        )
        if product is None:
            self.done = True
            info.stop_reason = "embedding_failed"
            info.terminated_by = "embedding_failed"
            return self.observe(), 0.0, True, info

        core_atom, synthon_atom = result.junction_bond
        _, junction_angle, rotated_bond = (
            self.conformer_engine.apply_junction_torsion(
                product,
                (core_atom, synthon_atom),
                float(action.dihedral_rad),
                pocket_coords=self._pocket_coords_np,
                pocket_vdw=self._pocket_vdw,
            )
        )

        if self._optimize_torsion_grid:
            _, _, clash = self.conformer_engine.optimize_dihedral(
                product,
                (core_atom, synthon_atom),
                pocket_coords=self._pocket_coords_np,
                pocket_vdw=self._pocket_vdw,
            )
            self.total_clash = float(clash)
        else:
            self.total_clash = float(
                self.conformer_engine.compute_steric_clash_np(
                    np.array(
                        Chem.RemoveHs(Chem.Mol(product)).GetConformer().GetPositions(),
                        dtype=np.float64,
                    ),
                    self._pocket_coords_np,
                    ligand_vdw=np.array(
                        [
                            vdw_radius(int(a.GetAtomicNum()))
                            for a in Chem.RemoveHs(Chem.Mol(product)).GetAtoms()
                        ],
                        dtype=np.float64,
                    ),
                    pocket_vdw=self._pocket_vdw,
                )
            )

        self.current_mol = product
        self.recipe.append(
            {
                "step": step,
                "action": "react",
                "reaction": chosen_rxn,
                "reaction_family": family,
                "handle": target_handle.handle_type,
                "synthon_id": self.catalog.get_id(synthon_idx),
                "synthon_smiles": self.catalog.get_smiles(synthon_idx),
                "dihedral_applied_rad": float(action.dihedral_rad),
                "junction_angle_rad": float(junction_angle),
                "torsion_mode": (
                    "planar_snap_adjacent"
                    if rotated_bond != (core_atom, synthon_atom)
                    else "direct"
                ),
                "rotated_bond": (
                    list(rotated_bond) if rotated_bond else None
                ),
                "new_atoms": len(result.synthon_atom_map) + len(result.new_atoms),
            }
        )

        info.reaction = chosen_rxn
        info.reaction_family = family
        info.handle = target_handle.handle_type
        info.synthon_id = self.catalog.get_id(synthon_idx)
        info.synthon_smiles = self.catalog.get_smiles(synthon_idx)
        info.dihedral_applied_rad = float(action.dihedral_rad)
        info.junction_angle_rad = float(junction_angle)
        info.torsion_mode = (
            "planar_snap_adjacent"
            if rotated_bond != (core_atom, synthon_atom)
            else "direct"
        )
        info.rotated_bond = rotated_bond
        info.new_atoms = len(result.synthon_atom_map) + len(result.new_atoms)

        # Dense physics reward: contact-energy improvement minus clash.
        contact = self.contact_energy()
        info.clash = self.total_clash
        info.contact_energy = contact
        if self.dense_reward:
            contact_improvement = math.tanh(
                (self._prev_contact_energy - contact) / 5.0
            )
            reward = float(contact_improvement - 0.01 * self.total_clash)
        else:
            reward = 0.0
        self._prev_contact_energy = contact

        # Horizon / dead-chemistry termination.
        if self.step_count >= self.horizon:
            self.done = True
            info.stop_reason = "horizon"
            info.terminated_by = "horizon"
        else:
            next_handles = self.rxn_engine.rank_handles(self.current_mol)
            if not next_handles:
                self.done = True
                info.stop_reason = "no_handles"
                info.terminated_by = "no_handles"
            else:
                r_mask, s_masks, has_handle = self.legal_action_masks()
                if not bool((r_mask > -1e8).any().item()):
                    self.done = True
                    info.stop_reason = "no_legal_reactions"
                    info.terminated_by = "no_legal_reactions"

        return self.observe(), reward, self.done, info

    # ------------------------------------------------------------------
    # Physics / results
    # ------------------------------------------------------------------
    def contact_energy(self) -> float:
        """Soft-core Lennard-Jones contact energy of the current ligand."""
        if self.current_mol is None:
            return 0.0
        try:
            ligand_noH = Chem.RemoveHs(Chem.Mol(self.current_mol))
            ligand_coords = np.array(
                ligand_noH.GetConformer().GetPositions(), dtype=np.float64
            )
            ligand_vdw = np.array(
                [vdw_radius(int(a.GetAtomicNum())) for a in ligand_noH.GetAtoms()],
                dtype=np.float64,
            )
            return float(
                self.conformer_engine.compute_lennard_jones_np(
                    ligand_coords, self._pocket_coords_np, ligand_vdw,
                    self._pocket_vdw,
                )
            )
        except Exception:
            return 0.0

    def validate(self) -> Dict[str, bool]:
        """Run the chemical validator on the current ligand against the pocket."""
        return self.validator.validate(
            self.current_mol,
            pocket_coords=self._pocket_coords_np,
            pocket_vdw=self._pocket_vdw,
        )

    def smiles(self) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(self.current_mol)))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pick_seed_synthon(self) -> int:
        valid = [
            i
            for i in range(len(self.catalog))
            if self.rxn_engine.handle_types(self.catalog.get_mol(i, explicit_hs=False))
        ]
        if not valid:
            raise RuntimeError("Catalog contains no handle-bearing synthons")
        return int(self.seed_rng.integers(0, len(valid)))

    def _place_seed_in_pocket(
        self,
        mol: Chem.Mol,
        seed_handle_type: Optional[str],
        seed_handle_atom_index: Optional[int],
        search_radius: float = 4.0,
        grid: int = 9,
        contact_distance: float = 3.2,
        contact_width: float = 0.8,
    ) -> None:
        """Translate a seed to a contact-rich, clash-avoiding pocket position.

        Delegates to the same placement heuristic used by SBDDGenerator
        (single source of truth for the seeding physics).
        """
        from syntree.engine.generator import SBDDGenerator

        SBDDGenerator._place_seed_in_pocket(
            mol,
            self._pocket_coords_np,
            pocket_vdw=self._pocket_vdw,
            hotspots=self._hotspots,
            seed_handle_type=seed_handle_type,
            seed_handle_atom_index=seed_handle_atom_index,
            search_radius=search_radius,
            grid=grid,
            contact_distance=contact_distance,
            contact_width=contact_width,
        )


__all__ = ["MolecularAssemblyEnv", "AssemblyAction", "EnvInfo"]
