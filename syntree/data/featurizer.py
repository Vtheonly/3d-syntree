"""Atomic featurization for protein pockets and ligand attachment handles.

Pocket atoms are encoded as ``(position, atomic number)`` point clouds,
centroid-centered for translation invariance. Attachment handles are encoded
as fixed-length flat feature vectors summarising the reacting atom and its
chemical environment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem

# Dimensionality of the flat handle feature vector.
HANDLE_FEATURE_DIM = 64

# Offset layout of the handle feature vector:
#   [0, 16)   one-hot atomic number (capped)
#   [16, 24)  one-hot degree
#   [24, 26)  aromaticity flag, ring-membership flag
#   [26, 34)  one-hot formal charge in [-3, +4]
#   [34, 44)  one-hot total valence
#   [44, 52)  one-hot num H neighbours
#   [52, 56)  handle-class one-hot start (classes 0..7)
#   [56, 64)  neighbour element histogram (C, N, O, S, halogens, other)
_ATOMIC_NUM_SLOTS = 16
_DEGREE_OFFSET = 16
_AROMATIC_OFFSET = 24
_CHARGE_OFFSET = 26
_VALENCE_OFFSET = 34
_NUMH_OFFSET = 44
_HANDLE_CLASS_OFFSET = 52
_NEIGHBOR_OFFSET = 56

_HANDLE_CLASSES = [
    "carboxylic_acid",
    "primary_secondary_amine",
    "aryl_halide",
    "boronic_acid",
    "aldehyde",
    "alcohol",
    "alkyne",
    "azide",
]


class MolecularFeaturizer:
    """Static helpers turning RDKit objects into model-ready tensors."""

    # ------------------------------------------------------------------
    # Pockets
    # ------------------------------------------------------------------
    @staticmethod
    def featurize_pocket(
        pocket_mol: Chem.Mol,
        max_distance: float = 8.0,
        center: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert a protein pocket (PDB-derived Mol) into a point cloud.

        Args:
            pocket_mol: RDKit molecule read from a pocket PDB. Must carry a
                conformer.
            max_distance: unused placeholder kept for API symmetry; pockets
                are expected to be pre-trimmed to this radius.
            center: optional external centroid (e.g. the ligand centroid)
                used to center the cloud instead of the pocket's own mean.

        Returns:
            Dict with ``pocket_pos`` ``[N, 3]``, ``pocket_z`` ``[N]`` and
            ``centroid`` ``[1, 3]``.
        """
        if pocket_mol is None:
            raise ValueError("featurize_pocket requires a valid Mol")
        if pocket_mol.GetNumConformers() == 0:
            raise ValueError("pocket_mol must have 3D coordinates (a conformer)")

        conf = pocket_mol.GetConformer()
        positions: List[List[float]] = []
        atomic_nums: List[int] = []
        for atom in pocket_mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            positions.append([pos.x, pos.y, pos.z])
            atomic_nums.append(atom.GetAtomicNum())

        pos_t = torch.tensor(positions, dtype=torch.float32)
        z_t = torch.tensor(atomic_nums, dtype=torch.long)

        if center is None:
            centroid = pos_t.mean(dim=0, keepdim=True)
        else:
            centroid = center.to(pos_t.dtype).view(1, 3)
        pos_t = pos_t - centroid

        return {
            "pocket_pos": pos_t,
            "pocket_z": z_t,
            "centroid": centroid,
        }

    # ------------------------------------------------------------------
    # Handles
    # ------------------------------------------------------------------
    @staticmethod
    def featurize_handle(
        mol: Chem.Mol,
        handle_atom_indices: Sequence[int],
        handle_type: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode an attachment handle into a fixed 64-dim vector.

        The vector captures the reacting atom's element, degree, aromaticity,
        ring membership, formal charge, valence, hydrogen count, handle
        class, and the element histogram of its neighbourhood.
        """
        feats = np.zeros(HANDLE_FEATURE_DIM, dtype=np.float32)
        if mol is None or not handle_atom_indices:
            return torch.from_numpy(feats)

        atom = mol.GetAtomWithIdx(int(handle_atom_indices[0]))

        z = atom.GetAtomicNum()
        if 0 < z <= _ATOMIC_NUM_SLOTS:
            feats[z - 1] = 1.0

        feats[_DEGREE_OFFSET + min(atom.GetDegree(), 7)] = 1.0

        if atom.GetIsAromatic():
            feats[_AROMATIC_OFFSET] = 1.0
        if atom.IsInRing():
            feats[_AROMATIC_OFFSET + 1] = 1.0

        charge = int(atom.GetFormalCharge())
        charge_slot = charge + 3  # maps [-3, 4] -> [0, 7]
        if 0 <= charge_slot <= 7:
            feats[_CHARGE_OFFSET + charge_slot] = 1.0

        feats[_VALENCE_OFFSET + min(atom.GetTotalValence(), 9)] = 1.0

        feats[_NUMH_OFFSET + min(atom.GetTotalNumHs(), 7)] = 1.0

        if handle_type is not None and handle_type in _HANDLE_CLASSES:
            feats[_HANDLE_CLASS_OFFSET + _HANDLE_CLASSES.index(handle_type)] = 1.0

        # Neighbourhood element histogram.
        for nbr in atom.GetNeighbors():
            nz = nbr.GetAtomicNum()
            if nz == 6:
                feats[_NEIGHBOR_OFFSET] += 1.0
            elif nz == 7:
                feats[_NEIGHBOR_OFFSET + 1] += 1.0
            elif nz == 8:
                feats[_NEIGHBOR_OFFSET + 2] += 1.0
            elif nz == 16:
                feats[_NEIGHBOR_OFFSET + 3] += 1.0
            elif nz in (9, 17, 35, 53):
                feats[_NEIGHBOR_OFFSET + 4] += 1.0
            else:
                feats[_NEIGHBOR_OFFSET + 5] += 1.0

        return torch.from_numpy(feats)

    # ------------------------------------------------------------------
    # Ligands (training targets)
    # ------------------------------------------------------------------
    @staticmethod
    def ligand_center(ligand_mol: Chem.Mol) -> torch.Tensor:
        """Centroid of a ligand's heavy atoms (requires a conformer)."""
        if ligand_mol.GetNumConformers() == 0:
            raise ValueError("ligand_mol must have 3D coordinates")
        conf = ligand_mol.GetConformer()
        pos = np.array(
            [
                list(conf.GetAtomPosition(a.GetIdx()))
                for a in ligand_mol.GetAtoms()
                if a.GetAtomicNum() > 1
            ],
            dtype=np.float32,
        )
        if len(pos) == 0:
            pos = np.array(
                [list(conf.GetAtomPosition(a.GetIdx())) for a in ligand_mol.GetAtoms()],
                dtype=np.float32,
            )
        return torch.from_numpy(pos.mean(axis=0))

    @staticmethod
    def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
        """Robust SMILES parsing helper (returns None on failure)."""
        try:
            return Chem.MolFromSmiles(smiles, sanitize=sanitize)
        except Exception:
            return None


__all__ = ["MolecularFeaturizer", "HANDLE_FEATURE_DIM"]
