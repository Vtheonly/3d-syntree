"""Atomic featurization for pockets, ligand states and attachment handles.

Pocket atoms are encoded as ``(position, atomic number, formal charge)``
point clouds, centroid-centered for translation invariance. X-ray pockets
arrive without hydrogens and with neutral heavy atoms; formal charges at
physiological pH (Asp/Glu -1, Arg/Lys +1, protonated His +1) are assigned
from PDB residue/atom names so the policy can learn salt bridges.

Intermediate ligand states are featurized as heavy-atom point clouds in the
same pocket-centered frame (the "ghost ligand" fix: the policy now sees
the molecule it has already grown), together with global scalar features
(molecular weight, heavy-atom count, cavity-occupation ratio) that feed
the termination decision.

Attachment handles are encoded as fixed-length flat chemical feature
vectors (64-dim). The reacting atom's 3D position travels as a separate
``handle_pos`` tensor - positions and invariant features are never mixed
in the same tensor (bug report 3 / Tell 1).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

# Dimensionality of the flat handle feature vector (chemical features only;
# the handle xyz is returned separately as `handle_pos`).
HANDLE_FEATURE_DIM = 64
HANDLE_CHEMICAL_FEATURE_DIM = 64

# Dimensionality of the global ligand-state features:
# [MW/500, heavy_atoms/50, ligand_volume/pocket_volume, has_ligand].
GLOBAL_FEATURE_DIM = 4

# Offset layout of the handle feature vector:
#   [0, 16)   one-hot atomic number (capped)
#   [16, 24)  one-hot degree
#   [24, 26)  aromaticity flag, ring-membership flag
#   [26, 34)  one-hot formal charge in [-3, +4]
#   [34, 44)  one-hot total valence
#   [44, 52)  one-hot num H neighbours
#   [52, 60)  handle-class one-hot (legacy classes 0..7)
#   [56, 62)  neighbour element histogram (C, N, O, S, halogens, other)
#   [62, 64)  extended handle classes 8..9 (sulfonyl_chloride, alkyl_halide)
#
# KNOWN LEGACY ALIASING (kept deliberately for data compatibility): the
# original layout booked the 8-wide handle-class one-hot at offset 52 while
# the neighbour element histogram starts at 56, so classes 4..7 (aldehyde,
# alcohol, alkyne, azide) overlap slots 56..59 with the C/N/O/S neighbour
# counts. Every existing dataset shard, processed cache and checkpoint was
# produced under those semantics, so re-mapping the slots would silently
# change the meaning of stored 64-dim vectors. The two tasklist-Priority-4
# handle classes are therefore placed in the two genuinely unused padding
# slots 62..63, which keeps ``HANDLE_FEATURE_DIM`` at 64 and leaves every
# historical byte identical. In newly *written* samples slots 62/63 are
# simply newly-informative (weights for always-zero inputs receive zero
# gradient, so old checkpoints transition gracefully).
_ATOMIC_NUM_SLOTS = 16
_DEGREE_OFFSET = 16
_AROMATIC_OFFSET = 24
_CHARGE_OFFSET = 26
_VALENCE_OFFSET = 34
_NUMH_OFFSET = 44
_HANDLE_CLASS_OFFSET = 52
_NEIGHBOR_OFFSET = 56
_EXTENDED_HANDLE_CLASS_OFFSET = 62

# Legacy handle classes 0..7 (order frozen: slots 52..59 in stored data).
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

# Extended handle classes 8..9 -> unused padding slots 62..63 (see layout
# note above). Appending further handle classes is NOT possible without
# widening HANDLE_FEATURE_DIM (breaking stored data); use the reserved
# slots sparingly.
_EXTENDED_HANDLE_CLASSES = [
    "sulfonyl_chloride",
    "alkyl_halide",
]

# Explicit handle-name -> slot map. Legacy classes keep their historical
# (aliased) slots; extended classes take the padding slots.
_HANDLE_CLASS_SLOTS = {
    **{name: _HANDLE_CLASS_OFFSET + i for i, name in enumerate(_HANDLE_CLASSES)},
    **{name: _EXTENDED_HANDLE_CLASS_OFFSET + i
       for i, name in enumerate(_EXTENDED_HANDLE_CLASSES)},
}

# Public mirror used by tests to pin the layout contract.
HANDLE_CLASS_SLOTS = dict(_HANDLE_CLASS_SLOTS)

# ---------------------------------------------------------------------------
# Physiological protonation states (pH 7.4) for standard amino-acid
# residues, expressed as formal charges assigned to the side-chain heavy
# atoms that carry the residue's charge. X-ray PDB files do not resolve
# hydrogens and RDKit parses their heavy atoms as neutral, which would make
# a PaiNN encoder blind to salt-bridge formation.
# ---------------------------------------------------------------------------
_PROTONATION_RULES = {
    # residue: {atom name -> fractional formal charge}; the values sum to the
    # residue's net charge at pH 7.4.
    "ASP": {"OD1": -0.5, "OD2": -0.5},
    "GLU": {"OE1": -0.5, "OE2": -0.5},
    "ARG": {"NE": 1.0 / 3.0, "NH1": 1.0 / 3.0, "NH2": 1.0 / 3.0},
    "LYS": {"NZ": 1.0},
    # Protonated histidine (HIP) only; neutral HID/HIE carry no net charge.
    "HIP": {"ND1": 0.5, "NE2": 0.5},
}


def assign_pocket_formal_charges(pocket_mol: Chem.Mol) -> np.ndarray:
    """Per-atom formal charges at pH 7.4 for a PDB-derived pocket.

    Charge assignment mirrors the PDB2PQR convention for standard residues
    (Asp/Glu carboxylates -1 shared by both oxygens, Arg guanidinium +1
    shared across the three nitrogens, Lys ammonium +1 on NZ, protonated His
    +1 shared by both ring nitrogens). Atoms with no PDB residue info keep
    their RDKit formal charge (SMILES-derived molecules are unaffected).

    Returns:
        ``[N]`` float32 array of formal charges.
    """
    charges = np.zeros(pocket_mol.GetNumAtoms(), dtype=np.float32)
    for atom in pocket_mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            charges[atom.GetIdx()] = float(atom.GetFormalCharge())
            continue
        residue = info.GetResidueName().strip().upper()
        name = info.GetName().strip().upper()
        rule = _PROTONATION_RULES.get(residue)
        if rule and name in rule:
            charges[atom.GetIdx()] = rule[name]
        else:
            charges[atom.GetIdx()] = float(atom.GetFormalCharge())
    return charges


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
            ``pocket_charge`` ``[N]`` (formal charges at pH 7.4).
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
        charge_t = torch.from_numpy(assign_pocket_formal_charges(pocket_mol))

        if center is None:
            centroid = pos_t.mean(dim=0, keepdim=True)
        else:
            centroid = center.to(pos_t.dtype).view(1, 3)
        pos_t = pos_t - centroid

        return {
            "pocket_pos": pos_t,
            "pocket_z": z_t,
            "pocket_charge": charge_t,
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
        reference_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode an attachment handle into a fixed 64-dim chemical vector.

        Only invariant chemical descriptors of the reacting atom and its
        environment are emitted; the handle's 3D position is returned
        separately by :meth:`featurize_handle_position` (bug report 3 /
        Tell 1: positions and features never share a tensor). The
        ``reference_center`` argument is accepted for call-site
        compatibility and ignored.
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

        if handle_type is not None and handle_type in _HANDLE_CLASS_SLOTS:
            feats[_HANDLE_CLASS_SLOTS[handle_type]] = 1.0

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

    @staticmethod
    def featurize_handle_position(
        mol: Chem.Mol,
        handle_atom_indices: Sequence[int],
        reference_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """3D position of the reacting handle atom in the pocket frame.

        Returns ``[3]`` float tensor; zero when the molecule has no
        conformer or no handle is given (degenerate states).
        """
        if mol is None or not handle_atom_indices or mol.GetNumConformers() == 0:
            return torch.zeros(3, dtype=torch.float32)
        conf = mol.GetConformer()
        handle_pos = np.array(
            conf.GetAtomPosition(int(handle_atom_indices[0])), dtype=np.float32
        )
        if reference_center is not None:
            ref = torch.as_tensor(reference_center, dtype=torch.float32).view(-1)
            if ref.numel() != 3:
                raise ValueError("reference_center must contain exactly three coordinates")
            handle_pos = handle_pos - ref.detach().cpu().numpy()
        return torch.from_numpy(handle_pos.astype(np.float32))

    # ------------------------------------------------------------------
    # Intermediate ligand states (the "ghost ligand" fix)
    # ------------------------------------------------------------------
    @staticmethod
    def featurize_ligand(
        ligand_mol: Optional[Chem.Mol],
        center: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Heavy-atom point cloud of the intermediate ligand.

        Args:
            ligand_mol: the molecule grown so far (requires a conformer;
                ``None``/empty yields empty tensors).
            center: the pocket reference center (the same centroid used by
                :meth:`featurize_pocket`) so ligand and pocket coordinates
                live in one frame.

        Returns:
            Dict with ``ligand_pos`` ``[M, 3]``, ``ligand_z`` ``[M]``,
            ``ligand_charge`` ``[M]`` (RDKit formal charges) and
            ``heavy_atom_map`` (original atom index -> ligand node index;
            needed to locate the handle node in the ligand graph because
            ``RemoveHs`` renumbers atoms).
        """
        if ligand_mol is None or ligand_mol.GetNumAtoms() == 0:
            return {
                "ligand_pos": torch.zeros(0, 3, dtype=torch.float32),
                "ligand_z": torch.zeros(0, dtype=torch.long),
                "ligand_charge": torch.zeros(0, dtype=torch.float32),
                "heavy_atom_map": {},
            }
        if ligand_mol.GetNumConformers() == 0:
            raise ValueError("featurize_ligand requires 3D coordinates")

        heavy = Chem.RemoveHs(Chem.Mol(ligand_mol))
        # RemoveHs renumbers atoms; the ligand node index of a heavy atom is
        # simply the number of heavy atoms preceding it (RemoveHs preserves
        # the relative order of the remaining atoms).
        heavy_map: Dict[int, int] = {}
        lig_node = 0
        for atom in ligand_mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                heavy_map[atom.GetIdx()] = lig_node
                lig_node += 1
        conf = heavy.GetConformer()
        positions, zs, charges = [], [], []
        for atom in heavy.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            zs.append(atom.GetAtomicNum())
            charges.append(float(atom.GetFormalCharge()))

        pos_t = torch.tensor(positions, dtype=torch.float32)
        if center is not None:
            ref = torch.as_tensor(center, dtype=torch.float32).view(1, 3)
            pos_t = pos_t - ref
        return {
            "ligand_pos": pos_t,
            "ligand_z": torch.tensor(zs, dtype=torch.long),
            "ligand_charge": torch.tensor(charges, dtype=torch.float32),
            "heavy_atom_map": heavy_map,
        }

    @staticmethod
    def vdw_sphere_volume(mol: Optional[Chem.Mol]) -> float:
        """Sum of per-heavy-atom van der Waals sphere volumes (A^3).

        Overlapping spheres overcount, but as a *ratio* feature (ligand vs
        pocket occupation) the overcounting largely cancels and the value
        stays deterministic and cheap.
        """
        from syntree.chemistry.conformer import vdw_radius

        if mol is None or mol.GetNumAtoms() == 0:
            return 0.0
        heavy = Chem.RemoveHs(Chem.Mol(mol))
        total = 0.0
        for atom in heavy.GetAtoms():
            r = vdw_radius(atom.GetAtomicNum())
            total += 4.0 / 3.0 * math.pi * r ** 3
        return float(total)

    @staticmethod
    def ligand_global_features(
        ligand_mol: Optional[Chem.Mol],
        pocket_volume: Optional[float] = None,
    ) -> torch.Tensor:
        """Global ligand-state features for the policy readout (Flaw 3).

        ``[MW/500, heavy_atoms/50, ligand_volume/pocket_volume, has_ligand]``
        - the termination (STOP) decision needs to know how much molecule
        has already been built relative to the cavity it must fill.
        """
        if ligand_mol is None or ligand_mol.GetNumAtoms() == 0:
            return torch.zeros(GLOBAL_FEATURE_DIM, dtype=torch.float32)
        heavy = Chem.RemoveHs(Chem.Mol(ligand_mol))
        mw = float(Descriptors.MolWt(heavy)) / 500.0
        n_heavy = heavy.GetNumAtoms() / 50.0
        if pocket_volume and pocket_volume > 0.0:
            vol_ratio = (
                MolecularFeaturizer.vdw_sphere_volume(heavy) / float(pocket_volume)
            )
        else:
            vol_ratio = 0.0
        return torch.tensor(
            [mw, n_heavy, vol_ratio, 1.0], dtype=torch.float32
        )

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


__all__ = [
    "MolecularFeaturizer",
    "HANDLE_FEATURE_DIM",
    "HANDLE_CHEMICAL_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
    "HANDLE_CLASS_SLOTS",
]
