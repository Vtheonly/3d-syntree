"""Build reaction-validated, multi-step imitation-learning trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import zlib

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from syntree.chemistry.reactions import HANDLE_NAMES, REACTION_FAMILY_NAMES, ReactionEngine
from syntree.data.featurizer import MolecularFeaturizer, GLOBAL_FEATURE_DIM
from syntree.data.decomposer import RetrosyntheticTrajectoryExtractor


@dataclass(frozen=True)
class TrajectoryStep:
    """One expert action in a forward synthesis trajectory."""

    step: int
    core_mol: Chem.Mol
    synthon_index: int
    reaction_family: str
    reaction_name: str
    core_handle_type: str
    target_dihedral: float


class TrajectoryDataset(torch.utils.data.Dataset):
    """Load cached PyG expert states with deterministic trajectory splits."""

    def __init__(self, path: str, split: str = "train"):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train/val/test")
        self.path = str(path)
        self.split = split
        self.states = torch.load(self.path, map_location="cpu", weights_only=False)
        self.states = [
            state for state in self.states
            if self._belongs_to_split(state)
        ]
        if not self.states:
            raise RuntimeError(f"No trajectory states found for split={split!r} in {path}")

    def _belongs_to_split(self, state: Data) -> bool:
        raw = int(getattr(state, "trajectory_hash", torch.tensor(0)).item())
        bucket = raw % 1000
        assigned = "train" if bucket < 800 else "val" if bucket < 900 else "test"
        return assigned == self.split

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        sample = self.states[index]
        if hasattr(sample, "pocket_pos") and sample.pocket_pos is not None:
            sample.num_nodes = sample.pocket_pos.size(0)
        for k in list(sample.keys()):
            v = sample[k]
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                sample[k] = v.unsqueeze(0)
        return sample

    @property
    def num_synthetic(self):
        return 0


class RetrosyntheticTrajectoryBuilder:
    """Convert an observed ligand into a validated forward action trajectory."""

    def __init__(self, catalog, max_steps: int = 4):
        self.catalog = catalog
        self.max_steps = int(max_steps)
        self.extractor = RetrosyntheticTrajectoryExtractor(
            catalog, max_steps=self.max_steps
        )
        self.engine = self.extractor.fragmenter.engine

    def build(
        self,
        ligand: Chem.Mol,
        pocket: Optional[Chem.Mol] = None,
        trajectory_id: str = "",
    ) -> Tuple[List[Data], Dict]:
        if ligand is None or ligand.GetNumAtoms() == 0:
            return [], {"trajectory_id": trajectory_id, "status": "invalid_ligand"}
        if ligand.GetNumConformers() == 0:
            return [], {"trajectory_id": trajectory_id, "status": "missing_conformer"}

        ligand_center = MolecularFeaturizer.ligand_center(ligand)
        if pocket is None or pocket.GetNumAtoms() == 0:
            return [], {"trajectory_id": trajectory_id, "status": "missing_pocket"}

        pocket_features = MolecularFeaturizer.featurize_pocket(
            pocket, center=ligand_center
        )

        reverse_records: List[Tuple[Chem.Mol, object]] = []
        current = Chem.Mol(ligand)
        seen = set()

        for _ in range(self.max_steps):
            key = Chem.MolToSmiles(
                Chem.RemoveHs(Chem.Mol(current)),
                canonical=True,
                isomericSmiles=False,
            )
            if key in seen:
                break
            seen.add(key)

            target = self.extractor.extract_step(current)
            if target is None:
                break
            reverse_records.append((Chem.Mol(current), target))
            current = Chem.Mol(target.core_mol)

        if not reverse_records:
            return [], {
                "trajectory_id": trajectory_id,
                "status": "no_valid_decomposition",
                "num_actions": 0,
            }

        forward_records = list(reversed(reverse_records))
        samples: List[Data] = []

        for step_idx, (product_state, target) in enumerate(forward_records):
            handles = [
                h
                for h in self.engine.detect_handles(target.core_mol)
                if h.handle_type == target.core_handle_type
            ]
            if not handles:
                continue
            handle = handles[0]

            handle_features = MolecularFeaturizer.featurize_handle(
                target.core_mol,
                handle.atom_indices,
                target.core_handle_type,
                reference_center=ligand_center,
            )
            try:
                lig_feats = MolecularFeaturizer.featurize_ligand(
                    target.core_mol, center=ligand_center
                )
                handle_node_idx = lig_feats["heavy_atom_map"].get(
                    int(handle.primary_atom), -1
                )
                handle_pos = MolecularFeaturizer.featurize_handle_position(
                    target.core_mol, handle.atom_indices,
                    reference_center=ligand_center,
                )
                global_feats = MolecularFeaturizer.ligand_global_features(
                    target.core_mol,
                    pocket_volume=MolecularFeaturizer.vdw_sphere_volume(pocket),
                )
            except (ValueError, RuntimeError):
                lig_feats = {
                    "ligand_pos": torch.zeros(0, 3),
                    "ligand_z": torch.zeros(0, dtype=torch.long),
                    "ligand_charge": torch.zeros(0),
                    "heavy_atom_map": {},
                }
                handle_node_idx = -1
                handle_pos = torch.zeros(3)
                global_feats = torch.zeros(GLOBAL_FEATURE_DIM)

            samples.append(
                Data(
                    num_nodes=pocket_features["pocket_pos"].size(0),
                    pocket_pos=pocket_features["pocket_pos"],
                    pocket_z=pocket_features["pocket_z"],
                    pocket_charge=pocket_features["pocket_charge"],
                    ligand_pos=lig_feats["ligand_pos"],
                    ligand_z=lig_feats["ligand_z"],
                    ligand_charge=lig_feats["ligand_charge"],
                    handle_features=handle_features,
                    handle_pos=handle_pos.view(1, 3) if handle_pos.dim() == 1 else handle_pos,
                    handle_nodes=torch.tensor([handle_node_idx], dtype=torch.long),
                    global_features=global_feats.view(1, GLOBAL_FEATURE_DIM),
                    target_synthon=torch.tensor(
                        [int(target.synthon_index)], dtype=torch.long
                    ),
                    target_reaction_family_idx=torch.tensor(
                        [REACTION_FAMILY_NAMES.index(target.reaction_family)],
                        dtype=torch.long,
                    ),
                    target_core_handle_idx=torch.tensor(
                        [HANDLE_NAMES.index(target.core_handle_type)],
                        dtype=torch.long,
                    ),
                    target_dihedral=torch.tensor(
                        [float(target.target_dihedral)], dtype=torch.float32
                    ),
                    target_stop=torch.tensor([False], dtype=torch.bool),
                    stop_mask=torch.tensor([0.0], dtype=torch.float32),
                    trajectory_step=torch.tensor([step_idx], dtype=torch.long),
                    trajectory_hash=torch.tensor(
                        [zlib.crc32(str(trajectory_id).encode("utf-8"))],
                        dtype=torch.long,
                    ),
                    is_real_sample=torch.tensor([True], dtype=torch.bool),
                )
            )

        terminal_handles = self.engine.detect_handles(current)
        if terminal_handles and samples:
            terminal_handle = terminal_handles[0]
            terminal_features = MolecularFeaturizer.featurize_handle(
                current,
                terminal_handle.atom_indices,
                terminal_handle.handle_type,
                reference_center=ligand_center,
            )
            try:
                term_lig_feats = MolecularFeaturizer.featurize_ligand(
                    current, center=ligand_center
                )
                term_handle_node = term_lig_feats["heavy_atom_map"].get(
                    int(terminal_handle.primary_atom), -1
                )
                term_handle_pos = MolecularFeaturizer.featurize_handle_position(
                    current, terminal_handle.atom_indices,
                    reference_center=ligand_center,
                )
                term_global = MolecularFeaturizer.ligand_global_features(
                    current,
                    pocket_volume=MolecularFeaturizer.vdw_sphere_volume(pocket),
                )
            except (ValueError, RuntimeError):
                term_lig_feats = {
                    "ligand_pos": torch.zeros(0, 3),
                    "ligand_z": torch.zeros(0, dtype=torch.long),
                    "ligand_charge": torch.zeros(0),
                }
                term_handle_node = -1
                term_handle_pos = torch.zeros(3)
                term_global = torch.zeros(GLOBAL_FEATURE_DIM)
            samples.append(
                Data(
                    num_nodes=pocket_features["pocket_pos"].size(0),
                    pocket_pos=pocket_features["pocket_pos"],
                    pocket_z=pocket_features["pocket_z"],
                    pocket_charge=pocket_features["pocket_charge"],
                    ligand_pos=term_lig_feats["ligand_pos"],
                    ligand_z=term_lig_feats["ligand_z"],
                    ligand_charge=term_lig_feats["ligand_charge"],
                    handle_features=terminal_features,
                    handle_pos=term_handle_pos.view(1, 3) if term_handle_pos.dim() == 1 else term_handle_pos,
                    handle_nodes=torch.tensor(
                        [term_handle_node], dtype=torch.long
                    ),
                    global_features=term_global.view(1, GLOBAL_FEATURE_DIM),
                    target_synthon=torch.tensor([0], dtype=torch.long),
                    target_reaction_family_idx=torch.tensor(
                        [REACTION_FAMILY_NAMES.index(REACTION_FAMILY_NAMES[0])],
                        dtype=torch.long,
                    ),
                    target_core_handle_idx=torch.tensor(
                        [HANDLE_NAMES.index(terminal_handle.handle_type)],
                        dtype=torch.long,
                    ),
                    target_dihedral=torch.tensor([0.0], dtype=torch.float32),
                    target_stop=torch.tensor([True], dtype=torch.bool),
                    stop_mask=torch.tensor([0.0], dtype=torch.float32),
                    trajectory_step=torch.tensor(
                        [len(samples)], dtype=torch.long
                    ),
                    trajectory_hash=torch.tensor(
                        [zlib.crc32(str(trajectory_id).encode("utf-8"))],
                        dtype=torch.long,
                    ),
                    is_real_sample=torch.tensor([True], dtype=torch.bool),
                )
            )

        status = "ok" if samples else "no_valid_states"
        return samples, {
            "trajectory_id": trajectory_id,
            "status": status,
            "num_actions": sum(
                int(not bool(sample.target_stop.item())) for sample in samples
            ),
            "num_states": len(samples),
            "reaction_families": [
                REACTION_FAMILY_NAMES[int(sample.target_reaction_family_idx.item())]
                for sample in samples
                if not bool(sample.target_stop.item())
            ],
        }


__all__ = ["TrajectoryStep", "TrajectoryDataset", "RetrosyntheticTrajectoryBuilder"]