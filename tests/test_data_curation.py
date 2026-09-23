"""Tests for standardized structural curation."""
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem

from syntree.data.curation import (
    PocketExtractionConfig,
    ligand_quality_flags,
    recenter_ligand,
    standardize_pocket,
)


def _ligand():
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    conf_id = AllChem.EmbedMolecule(mol, randomSeed=7)
    assert conf_id >= 0
    return mol


def test_standardize_pocket_removes_hetero_atoms_and_recenters(tmp_path: Path):
    protein = tmp_path / "protein.pdb"
    protein.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n"
        "ATOM      2  N   ALA A   1       2.000   2.000   3.000  1.00 20.00           N\n"
        "HETATM    3  O   HOH A 101       1.100   2.100   3.100  1.00 20.00           O\n"
        "END\n"
    )
    ligand = _ligand()
    conf = ligand.GetConformer()
    for i in range(ligand.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(1.0, 2.0, 3.0))

    out = tmp_path / "pocket.pdb"
    meta = standardize_pocket(
        str(protein),
        ligand,
        str(out),
        PocketExtractionConfig(radius_angstrom=2.0, recenter=True),
    )
    text = out.read_text()
    assert "HETATM" not in text
    assert meta["recentered"] is True
    assert meta["atom_count"] == 2
    assert "  0.000" in text


def test_recenter_ligand_uses_same_origin():
    ligand = _ligand()
    out = recenter_ligand(ligand, (1.0, 2.0, 3.0))
    p = out.GetConformer().GetAtomPosition(0)
    original = ligand.GetConformer().GetAtomPosition(0)
    assert np.allclose(
        [p.x, p.y, p.z],
        [original.x - 1.0, original.y - 2.0, original.z - 3.0],
    )


def test_artifact_residue_is_flagged():
    ligand = _ligand()
    info = Chem.AtomPDBResidueInfo(" C  ", 1, " ", "GOL")
    ligand.GetAtomWithIdx(0).SetMonomerInfo(info)
    assert "crystallization_artifact" in ligand_quality_flags(ligand)
