"""Unit tests for the molecular featurizer."""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.data.featurizer import (
    HANDLE_CLASS_SLOTS,
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
        # Chemical features only (64): the handle xyz travels separately in
        # `handle_pos` (bug report 3 / Tell 1 - no Franken tensors).
        assert HANDLE_FEATURE_DIM == 64

    def test_chemical_and_position_layout(self, pocket_mol):
        feats = MolecularFeaturizer.featurize_handle(pocket_mol, [0])
        assert feats.shape == (HANDLE_FEATURE_DIM,)
        # Chemical block is one-hot / count based (small non-negative ints).
        assert feats.sum() >= 2
        assert feats.min() >= 0.0
        # The position is returned by the dedicated method.
        pos = MolecularFeaturizer.featurize_handle_position(pocket_mol, [0])
        assert pos.shape == (3,)
        assert torch.isfinite(pos).all()

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


class TestHandleClassSlotLayout:
    """Pinned layout of the 64-dim handle vector, including the extended
    Priority-4 classes in the reserved padding slots 62/63 (see
    syntree/data/featurizer.py for the documented legacy aliasing)."""

    def test_feature_dim_unchanged(self):
        assert HANDLE_FEATURE_DIM == 64

    def test_legacy_class_slots_byte_compatible(self):
        """Classes 0..7 keep their historical slots 52..59 exactly, even
        though 56..59 alias the neighbour histogram (documented legacy)."""
        legacy = {
            "carboxylic_acid": 52,
            "primary_secondary_amine": 53,
            "aryl_halide": 54,
            "boronic_acid": 55,
            "aldehyde": 56,
            "alcohol": 57,
            "alkyne": 58,
            "azide": 59,
        }
        assert HANDLE_CLASS_SLOTS == {
            **legacy,
            "sulfonyl_chloride": 62,
            "alkyl_halide": 63,
        }

    def test_slot_map_covers_every_grammar_handle(self):
        """Every handle the reaction grammar can detect must have a slot."""
        from syntree.chemistry.reactions import HANDLE_NAMES
        assert set(HANDLE_NAMES) == set(HANDLE_CLASS_SLOTS)

    def test_sulfonyl_chloride_uses_reserved_slot(self, pocket_mol):
        mol = Chem.MolFromSmiles("CS(=O)(=O)Cl")
        feats = MolecularFeaturizer.featurize_handle(mol, [0], "sulfonyl_chloride")
        assert feats.shape == (64,)
        assert feats[62].item() == 1.0
        assert feats[63].item() == 0.0

    def test_alkyl_halide_uses_reserved_slot(self):
        mol = Chem.MolFromSmiles("CCBr")
        feats = MolecularFeaturizer.featurize_handle(mol, [1], "alkyl_halide")
        assert feats[63].item() == 1.0
        assert feats[62].item() == 0.0

    def test_legacy_handle_slots_unchanged(self):
        """An amine handle still writes slot 53 (not 62/63): old data and
        new data agree byte-for-byte for legacy handle classes."""
        mol = Chem.MolFromSmiles("C1CC(N)CC1")
        feats = MolecularFeaturizer.featurize_handle(
            mol, [2], "primary_secondary_amine"
        )
        assert feats[53].item() == 1.0
        assert feats[62].item() == 0.0
        assert feats[63].item() == 0.0

    def test_extended_slots_never_collide_with_neighbour_histogram(self):
        """Slots 62/63 must not receive neighbour-count contributions."""
        # Carbon tetrachloride: 4 halogen neighbours + alkyl halide class.
        # (Atom index 1 is the carbon; SMILES atom 0 is the first Cl.)
        mol = Chem.MolFromSmiles("ClC(Cl)(Cl)Cl")
        feats = MolecularFeaturizer.featurize_handle(mol, [1], "alkyl_halide")
        # 4 halogen neighbours land in the histogram slot 60 only.
        assert feats[60].item() == 4.0
        # Slot 63 holds the pure one-hot; slot 62 untouched.
        assert feats[63].item() == 1.0
        assert feats[62].item() == 0.0
