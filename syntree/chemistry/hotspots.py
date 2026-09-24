"""Lightweight pharmacophore hotspot detection for protein pockets.

The implementation is intentionally deterministic and dependency-light. It
uses PDB residue identity plus atom chemistry to identify approximate positive,
negative, donor, acceptor, and hydrophobic interaction hotspots. These are
placement priors, not binding-affinity predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class PocketHotspot:
    """One interaction hotspot represented by a pocket atom."""

    atom_index: int
    position: Tuple[float, float, float]
    kind: str
    residue: str
    score: float


# Residues with strong charge character at physiological conditions.
_POSITIVE_RESIDUES = {"LYS", "ARG"}
_NEGATIVE_RESIDUES = {"ASP", "GLU"}
_HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO"}

# Handle type -> preferred hotspot classes for initial seeding.
HANDLE_HOTSPOT_PREFERENCES: Dict[str, Tuple[str, ...]] = {
    "carboxylic_acid": ("positive", "donor"),
    "primary_secondary_amine": ("negative", "acceptor"),
    "boronic_acid": ("donor", "acceptor"),
    "alcohol": ("donor", "acceptor"),
    "aldehyde": ("donor", "hydrophobic"),
    "aryl_halide": ("hydrophobic",),
    "alkyne": ("hydrophobic",),
    "azide": ("positive", "donor"),
    # tasklist Priority 4 handles: sulfonyl chlorides present a polarised,
    # strongly electrophilic SO2 group whose two oxygens want donor/positive
    # environments (mirroring carboxylic acids); sp3 alkyl halides are small
    # greasy electrophiles that seed best into hydrophobic subpockets.
    "sulfonyl_chloride": ("positive", "donor"),
    "alkyl_halide": ("hydrophobic",),
}


class PocketHotspotFeaturizer:
    """Extract coarse pharmacophore hotspots from an RDKit PDB molecule."""

    def featurize(
        self,
        pocket_mol: Chem.Mol,
        center: Optional[Sequence[float]] = None,
    ) -> List[PocketHotspot]:
        if pocket_mol is None or pocket_mol.GetNumAtoms() == 0:
            return []
        if pocket_mol.GetNumConformers() == 0:
            raise ValueError("pocket_mol must have 3D coordinates")

        origin = np.asarray(center if center is not None else (0.0, 0.0, 0.0), dtype=float)
        if origin.size != 3:
            raise ValueError("center must contain three coordinates")

        conf = pocket_mol.GetConformer()
        hotspots: List[PocketHotspot] = []
        for atom in pocket_mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            residue = info.GetResidueName().strip().upper() if info else ""
            element = atom.GetAtomicNum()
            name = info.GetName().strip().upper() if info else ""

            kinds: List[Tuple[str, float]] = []
            if residue in _POSITIVE_RESIDUES and element == 7:
                kinds.append(("positive", 1.0))
            if residue in _NEGATIVE_RESIDUES and element == 8:
                kinds.append(("negative", 1.0))

            # H-bond donor/acceptor priors are residue-aware and intentionally
            # conservative because hydrogens are commonly absent from PDBs.
            if element in (7, 8, 16):
                if residue not in _NEGATIVE_RESIDUES or element != 8:
                    kinds.append(("donor", 0.65))
                if residue not in _POSITIVE_RESIDUES or element != 7:
                    kinds.append(("acceptor", 0.60))

            if residue in _HYDROPHOBIC_RESIDUES and element == 6:
                # Aromatic/branched side-chain carbons are useful geometric
                # anchors for hydrophobic fragments.
                kinds.append(("hydrophobic", 0.45))

            pos = np.asarray(conf.GetAtomPosition(atom.GetIdx()), dtype=float) - origin
            for kind, score in kinds:
                hotspots.append(
                    PocketHotspot(
                        atom_index=atom.GetIdx(),
                        position=(float(pos[0]), float(pos[1]), float(pos[2])),
                        kind=kind,
                        residue=residue,
                        score=float(score),
                    )
                )
        return hotspots

    def rank_for_handle(
        self,
        hotspots: Iterable[PocketHotspot],
        handle_type: str,
        pocket_center: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> List[PocketHotspot]:
        preferred = HANDLE_HOTSPOT_PREFERENCES.get(handle_type, ("hydrophobic",))
        preferred_rank = {kind: idx for idx, kind in enumerate(preferred)}
        center = np.asarray(pocket_center, dtype=float)

        ranked = [h for h in hotspots if h.kind in preferred_rank]
        ranked.sort(
            key=lambda h: (
                preferred_rank[h.kind],
                -h.score,
                float(np.linalg.norm(np.asarray(h.position) - center)),
                h.atom_index,
            )
        )
        return ranked


__all__ = [
    "PocketHotspot",
    "PocketHotspotFeaturizer",
    "HANDLE_HOTSPOT_PREFERENCES",
]
