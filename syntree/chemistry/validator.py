"""Chemical and physical validity checks (PoseBusters-equivalent).

Covers valence sanity, forbidden motifs (peroxides, pentavalent carbons),
physicochemical bounds, and — when a protein pocket is supplied —
inter-molecular steric clash sanity.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from syntree.chemistry.conformer import vdw_radius

# SMARTS for chemically unstable / explosive motifs commonly produced by
# unconstrained generative models.
_UNSTABLE_SMARTS = {
    "peroxide": "[OX2][OX2]",
    "ozonide": "[OX2][OX2][OX2]",
    "diazo": "[N-]=[N+]",
    "cumulated_alkene_cumulene_long": "C=C=C=C",  # extended cumulenes
}


class ChemicalValidator:
    """Executes PoseBusters-style sanity filters on generated molecules."""

    MAX_MW = 650.0
    MIN_FSP3 = 0.30
    MAX_FSP3 = 1.0
    MAX_ABS_FORMAL_CHARGE = 3
    MAX_HEAVY_ATOMS = 55

    @staticmethod
    def validate(
        mol: Chem.Mol,
        pocket_coords: Optional[np.ndarray] = None,
        pocket_vdw: Optional[np.ndarray] = None,
        min_inter_dist: float = 1.0,
    ) -> Dict[str, bool]:
        """Run the full battery of checks.

        Args:
            mol: Molecule to validate (a conformer is required only for the
                clash check).
            pocket_coords: optional ``[N, 3]`` protein pocket coordinates.
            pocket_vdw: optional per-atom pocket radii.
            min_inter_dist: minimum acceptable ligand-pocket atom distance.

        Returns:
            Dictionary of check name -> passed?  Every value is ``True``
            only when the corresponding physical/chemical law holds.
        """
        checks: Dict[str, bool] = {
            "valid_rdkit_mol": mol is not None,
            "valid_sanitization": False,
            "no_pentavalent_carbons": True,
            "no_unstable_motifs": True,
            "reasonable_mw": False,
            "valid_fsp3": False,
            "reasonable_charge": True,
            "reasonable_heavy_atoms": True,
            "has_conformer": True,
            "no_pocket_clash": True,
        }
        if mol is None:
            return {k: False for k in checks}

        try:
            probe = Chem.Mol(mol)
            Chem.SanitizeMol(probe)
            checks["valid_sanitization"] = True
        except Exception:
            return checks

        # --- valence grammar ------------------------------------------
        for atom in probe.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetExplicitValence() > 4:
                checks["no_pentavalent_carbons"] = False
            if abs(atom.GetFormalCharge()) > ChemicalValidator.MAX_ABS_FORMAL_CHARGE:
                checks["reasonable_charge"] = False
        if probe.GetNumHeavyAtoms() > ChemicalValidator.MAX_HEAVY_ATOMS:
            checks["reasonable_heavy_atoms"] = False

        # --- unstable motifs -------------------------------------------
        for name, smarts in _UNSTABLE_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and probe.HasSubstructMatch(pattern):
                checks["no_unstable_motifs"] = False

        # --- physicochemical bounds ------------------------------------
        mw = Descriptors.MolWt(probe)
        checks["reasonable_mw"] = mw <= ChemicalValidator.MAX_MW

        fsp3 = Lipinski.FractionCSP3(probe)
        checks["valid_fsp3"] = ChemicalValidator.MIN_FSP3 <= fsp3 <= ChemicalValidator.MAX_FSP3

        # --- geometry ---------------------------------------------------
        if probe.GetNumConformers() == 0:
            checks["has_conformer"] = False
        elif pocket_coords is not None and len(pocket_coords) > 0:
            lig = np.array(probe.GetConformer().GetPositions(), dtype=np.float64)
            if pocket_vdw is None:
                pocket_vdw = np.full(len(pocket_coords), vdw_radius(6))
            diff = lig[:, None, :] - np.asarray(pocket_coords, dtype=np.float64)[None, :, :]
            dists = np.sqrt((diff ** 2).sum(-1))
            # A clash is any atom pair closer than `min_inter_dist` A.
            checks["no_pocket_clash"] = bool((dists >= min_inter_dist).all())

        return checks

    @staticmethod
    def is_all_valid(
        mol: Chem.Mol,
        pocket_coords: Optional[np.ndarray] = None,
        pocket_vdw: Optional[np.ndarray] = None,
    ) -> bool:
        """True only when every check passes."""
        res = ChemicalValidator.validate(mol, pocket_coords, pocket_vdw)
        return all(res.values())

    @staticmethod
    def failure_reasons(mol: Chem.Mol) -> Dict[str, str]:
        """Human-readable diagnostics for the failed checks of ``mol``."""
        checks = ChemicalValidator.validate(mol)
        reasons = {
            "valid_rdkit_mol": "molecule object is None",
            "valid_sanitization": "RDKit sanitization failed (valence errors)",
            "no_pentavalent_carbons": "carbon with explicit valence > 4",
            "no_unstable_motifs": "unstable motif present (peroxide/ozonide/diazo/cumulene)",
            "reasonable_mw": f"molecular weight > {ChemicalValidator.MAX_MW} Da",
            "valid_fsp3": f"Fsp3 outside [{ChemicalValidator.MIN_FSP3}, 1.0]",
            "reasonable_charge": "formal charge magnitude > 3",
            "reasonable_heavy_atoms": f"more than {ChemicalValidator.MAX_HEAVY_ATOMS} heavy atoms",
            "has_conformer": "no 3D conformer available",
            "no_pocket_clash": "steric clash with pocket atoms",
        }
        return {k: msg for k, msg in reasons.items() if not checks.get(k, False)}

    # ------------------------------------------------------------------
    # Convenience descriptors used by evaluation pipelines
    # ------------------------------------------------------------------
    @staticmethod
    def descriptors(mol: Chem.Mol) -> Dict[str, float]:
        """Standard property block for reporting (MW, logP, Fsp3, TPSA,
        HBD/HBA, rotatable bonds)."""
        if mol is None:
            return {}
        return {
            "molecular_weight": float(Descriptors.MolWt(mol)),
            "clogp": float(Descriptors.MolLogP(mol)),
            "fsp3": float(Lipinski.FractionCSP3(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hbd": float(Lipinski.NumHDonors(mol)),
            "hba": float(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        }


__all__ = ["ChemicalValidator"]
