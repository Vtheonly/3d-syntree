"""Reaction-validated retrosynthetic supervision for unified datasets.

The repository's ReactionConstrainedFragmenter is the chemistry single source
of truth. Product-only transformations that cannot be inferred safely are
rejected instead of being assigned synthetic labels.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from rdkit import Chem

from syntree.data.fragmenter import (
    ReactionConstrainedFragmenter,
    RetrosyntheticTarget,
    SUPPORTED_RETRO_FAMILIES,
)


class RetrosyntheticTrajectoryExtractor:
    """Build deterministic expert trajectories from co-crystallized ligands."""

    def __init__(self, catalog, max_steps: int = 4):
        self.catalog = catalog
        self.max_steps = max(1, int(max_steps))
        self.fragmenter = ReactionConstrainedFragmenter(catalog)

    def extract_step(self, ligand_mol: Chem.Mol) -> Optional[RetrosyntheticTarget]:
        target = self.fragmenter.find_target(ligand_mol)
        if target is None or not np.isfinite(float(target.target_dihedral)):
            return None
        return target

    def extract(self, ligand_mol: Chem.Mol) -> List[RetrosyntheticTarget]:
        if ligand_mol is None or ligand_mol.GetNumConformers() == 0:
            return []

        current = Chem.Mol(ligand_mol)
        steps: List[RetrosyntheticTarget] = []
        seen = set()

        for _ in range(self.max_steps):
            key = Chem.MolToSmiles(
                Chem.RemoveHs(current), canonical=True, isomericSmiles=True
            )
            if key in seen:
                break
            seen.add(key)

            target = self.extract_step(current)
            if target is None:
                break
            if target.core_mol.GetNumConformers() == 0:
                break

            steps.append(target)
            current = Chem.Mol(target.core_mol)

        return steps

    @property
    def supported_retro_families(self):
        return tuple(SUPPORTED_RETRO_FAMILIES)


__all__ = ["RetrosyntheticTrajectoryExtractor"]
