"""Build reaction-validated, multi-step imitation-learning trajectories.

A trajectory is obtained by recursively inverting supported forward reactions
on an observed co-crystallized ligand. Each accepted inversion is required to
match an actual catalog synthon and to replay through the real reaction engine.

Approximate catalog similarity is used only to discover candidate synthons;
supervised targets are accepted only after exact forward replay reproduces the
observed molecular connectivity. This prevents fuzzy matching from becoming
fabricated ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import zlib

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from syntree.chemistry.reactions import HANDLE_NAMES, REACTION_FAMILY_NAMES, ReactionEngine
from syntree.data.featurizer import MolecularFeaturizer
from syntree.data.fragmenter import ReactionConstrainedFragmenter, SUPPORTED_RETRO_FAMILIES


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
        return self.states[index]

    @property
    def num_synthetic(self):
        return 0


class RetrosyntheticTrajectoryBuilder:
    """Convert an observed ligand into a validated forward action trajectory."""

    def __init__(self, catalog, max_steps: int = 4):
        self.catalog = catalog
        self.max_steps = int(max_steps)
        self.engine = ReactionEngine()
        self.fragmenter = ReactionConstrainedFragmenter(
            catalog, reaction_engine=self.engine
        )

    def build(
        self,
        ligand: Chem.Mol,
        pocket: Optional[Chem.Mol] = None,
        trajectory_id: str = "",
    ) -> Tuple[List[Data], Dict]:
        """Build PyG expert states from a co-crystallized ligand.

        The reverse search repeatedly removes one catalog-compatible synthon.
        The accepted records are reversed to recover the forward order:
        seed -> linker/cap -> ... -> observed ligand.

        A final explicit STOP state is emitted when the decomposition cannot
        continue but the resulting scaffold still exposes a reactive handle.
        """
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

            target = self.fragmenter.find_target(current)
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

            samples.append(
                Data(
                    pocket_pos=pocket_features["pocket_pos"],
                    pocket_z=pocket_features["pocket_z"],
                    handle_features=handle_features,
                    target_synthon=torch.tensor(
                        int(target.synthon_index), dtype=torch.long
                    ),
                    target_reaction_family_idx=torch.tensor(
                        REACTION_FAMILY_NAMES.index(target.reaction_family),
                        dtype=torch.long,
                    ),
                    target_core_handle_idx=torch.tensor(
                        HANDLE_NAMES.index(target.core_handle_type),
                        dtype=torch.long,
                    ),
                    target_dihedral=torch.tensor(
                        float(target.target_dihedral), dtype=torch.float32
                    ),
                    target_stop=torch.tensor(False, dtype=torch.bool),
                    stop_mask=torch.tensor(0.0, dtype=torch.float32),
                    trajectory_step=torch.tensor(step_idx, dtype=torch.long),
                    trajectory_hash=torch.tensor(
                        zlib.crc32(str(trajectory_id).encode("utf-8")),
                        dtype=torch.long,
                    ),
                    is_real_sample=torch.tensor(True, dtype=torch.bool),
                )
            )

        # If the final recoverable scaffold still has a reactive handle, teach
        # the policy that an explicit STOP is a valid terminal action.
        terminal_handles = self.engine.detect_handles(current)
        if terminal_handles and samples:
            terminal_handle = terminal_handles[0]
            terminal_features = MolecularFeaturizer.featurize_handle(
                current,
                terminal_handle.atom_indices,
                terminal_handle.handle_type,
                reference_center=ligand_center,
            )
            samples.append(
                Data(
                    pocket_pos=pocket_features["pocket_pos"],
                    pocket_z=pocket_features["pocket_z"],
                    handle_features=terminal_features,
                    target_synthon=torch.tensor(0, dtype=torch.long),
                    target_reaction_family_idx=torch.tensor(
                        REACTION_FAMILY_NAMES.index(
                            REACTION_FAMILY_NAMES[0]
                        ),
                        dtype=torch.long,
                    ),
                    target_core_handle_idx=torch.tensor(
                        HANDLE_NAMES.index(terminal_handle.handle_type),
                        dtype=torch.long,
                    ),
                    target_dihedral=torch.tensor(0.0, dtype=torch.float32),
                    target_stop=torch.tensor(True, dtype=torch.bool),
                    stop_mask=torch.tensor(0.0, dtype=torch.float32),
                    trajectory_step=torch.tensor(
                        len(samples), dtype=torch.long
                    ),
                    trajectory_hash=torch.tensor(
                        zlib.crc32(str(trajectory_id).encode("utf-8")),
                        dtype=torch.long,
                    ),
                    is_real_sample=torch.tensor(True, dtype=torch.bool),
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
