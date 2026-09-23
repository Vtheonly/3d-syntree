"""Unit tests for the molecular featurizer."""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.data.featurizer import (
    HANDLE_FEATURE_DIM,
    MolecularFeaturizer,
)


@pytest.fixture(scope="module")
def pocket_mol():
    mol = Chem.MolFromSmiles("CC(=O)NCC(=O)O")  # tiny "protein" fragment
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


class TestPocketFeaturization:
    def test_shapes_and_types(self, pocket_mol):
        feats = MolecularFeaturizer.featurize_pocket(pocket_mol)
        n_atoms = pocket_mol.GetNumAtoms()
        assert feats["pocket_pos"].shape == (n_atoms, 3)
        assert feats["pocket_z"].shape == (n_atoms,)
        assert feats["pocket_pos"].dtype == torch.float32
        assert feats["pocket_z"].dtype == torch.int64

    def test_centroid_centering(self, pocket_mol):
        feats = MolecularFeaturizer.featurize_pocket(pocket_mol)
        centroid = feats["pocket_pos"].mean(dim=0)
        assert torch.allclose(centroid, torch.zeros(3), atol=1e-5)

    def test_external_center(self, pocket_mol):
        center = torch.tensor([5.0, -3.0, 1.0])
        feats = MolecularFeaturizer.featurize_pocket(pocket_mol, center=center)
        raw = torch.tensor(
            [list(pocket_mol.GetConformer().GetAtomPosition(i))
             for i in range(pocket_mol.GetNumAtoms())]
        )
        expected = raw - center
        assert torch.allclose(feats["pocket_pos"], expected, atol=1e-4)

    def test_atomic_numbers_correct(self, pocket_mol):
        feats = MolecularFeaturizer.featurize_pocket(pocket_mol)
        expected = [a.GetAtomicNum() for a in pocket_mol.GetAtoms()]
        assert feats["pocket_z"].tolist() == expected

    def test_requires_conformer(self):
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ValueError, match="conformer"):
            MolecularFeaturizer.featurize_pocket(mol)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            MolecularFeaturizer.featurize_pocket(None)

    def test_pdb_pocket(self, tmp_path):
        pdb = tmp_path / "p.pdb"
        pdb.write_text(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C\n"
            "ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O\n"
        )
        mol = Chem.MolFromPDBFile(str(pdb), removeHs=False)
        feats = MolecularFeaturizer.featurize_pocket(mol)
        assert feats["pocket_pos"].shape == (4, 3)
        assert set(feats["pocket_z"].tolist()) <= {7, 6, 8}


class TestHandleFeaturization:
    def test_dimension(self):
        # 64 chemical + 3 xyz (pocket-frame position) = 67.
        assert HANDLE_FEATURE_DIM == 67

    def test_chemical_and_position_layout(self, pocket_mol):
        feats = MolecularFeaturizer.featurize_handle(pocket_mol, [0])
        assert feats.shape == (HANDLE_FEATURE_DIM,)
        # Chemical block is one-hot / count based (small non-negative ints).
        assert feats[:64].sum() >= 2
        # Positional block holds finite coordinates.
        assert torch.isfinite(feats[64:67]).all()

    def test_empty_indices_give_zero_vector(self, pocket_mol):
        feats = MolecularFeaturizer.featurize_handle(pocket_mol, [])
        assert feats.shape == (HANDLE_FEATURE_DIM,)
        assert feats.abs().sum() == 0

    def test_none_mol_gives_zero_vector(self):
        feats = MolecularFeaturizer.featurize_handle(None, [0])
        assert feats.abs().sum() == 0

    def test_atomic_number_slot(self, pocket_mol):
        # Atom 0 is carbon (z=6) -> slot 5 set.
        feats = MolecularFeaturizer.featurize_handle(pocket_mol, [0])
        assert feats[5] == 1.0
        assert feats[:5].sum() == 0

    def test_deterministic(self, pocket_mol):
        f1 = MolecularFeaturizer.featurize_handle(pocket_mol, [0])
        f2 = MolecularFeaturizer.featurize_handle(pocket_mol, [0])
        assert torch.equal(f1, f2)

    def test_different_atoms_differ(self, pocket_mol):
        # C vs O within the same molecule.
        fc = MolecularFeaturizer.featurize_handle(pocket_mol, [0])
        fo = MolecularFeaturizer.featurize_handle(pocket_mol, [2])
        assert not torch.equal(fc, fo)

    def test_handle_type_slot(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("OC(=O)C1CCCCC1"))
        assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
        feats = MolecularFeaturizer.featurize_handle(mol, [0], "carboxylic_acid")
        assert feats.sum() > 2  # element + degree + handle class + neighbors


class TestLigandCenter:
    def test_center_of_heavy_atoms(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        center = MolecularFeaturizer.ligand_center(mol)
        heavy = torch.tensor(
            [
                list(mol.GetConformer().GetAtomPosition(a.GetIdx()))
                for a in mol.GetAtoms()
                if a.GetAtomicNum() > 1
            ]
        )
        assert torch.allclose(center, heavy.mean(dim=0), atol=1e-5)

    def test_requires_conformer(self):
        with pytest.raises(ValueError):
            MolecularFeaturizer.ligand_center(Chem.MolFromSmiles("CCO"))


class TestSmilesParsing:
    def test_valid(self):
        assert MolecularFeaturizer.smiles_to_mol("CCO") is not None

    def test_invalid(self):
        assert MolecularFeaturizer.smiles_to_mol("not_a_smiles") is None
