"""Unit tests for the chemical validator."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.chemistry.validator import ChemicalValidator


@pytest.fixture(scope="module")
def validator():
    return ChemicalValidator()


def _with_conformer(smiles: str, seed: int = 42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


class TestValidMolecules:
    @pytest.mark.parametrize(
        "smiles",
        [
            "OC(=O)C1CCCCC1",           # sp3 acid
            "CC(C)(C)C(=O)NC1CC2(CC1)COC2",  # spiro amide
            "O=C(OC1CCCCC1)C1CCCCC1",   # ester
        ],
    )
    def test_clean_molecules_pass(self, validator, smiles):
        mol = _with_conformer(smiles)
        checks = validator.validate(mol)
        assert all(checks.values()), validator.failure_reasons(mol)
        assert validator.is_all_valid(mol)

    def test_druglike_amide_passes_all(self, validator):
        mol = _with_conformer("O=C(C1CCCCC1)NC1CC2(CC1)COC2")
        assert validator.is_all_valid(mol)


class TestInvalidMolecules:
    def test_none_molecule_fails_everything(self, validator):
        checks = validator.validate(None)
        assert not any(checks.values())

    def test_unsanitizable_molecule(self, validator):
        mol = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
        checks = validator.validate(mol)
        assert checks["valid_sanitization"] is False
        assert not validator.is_all_valid(mol)

    def test_peroxide_detected(self, validator):
        mol = _with_conformer("CCOOCC")
        checks = validator.validate(mol)
        assert checks["no_unstable_motifs"] is False

    def test_diazomethane_like_detected(self, validator):
        mol = Chem.MolFromSmiles("C=[N+]=[N-]")
        if mol is not None:
            checks = validator.validate(mol)
            assert checks["no_unstable_motifs"] is False

    def test_overweight_molecule(self, validator):
        mol = Chem.MolFromSmiles("C" + "C" * 60)  # ~880 Da
        checks = validator.validate(mol)
        assert checks["reasonable_mw"] is False
        assert checks["reasonable_heavy_atoms"] is False

    def test_flat_molecule_fails_fsp3(self, validator):
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene, Fsp3 = 0
        checks = validator.validate(mol)
        assert checks["valid_fsp3"] is False

    def test_high_formal_charge_fails_overall(self, validator):
        """A hypervalent formally-charged species cannot pass validation
        (sanitization or the charge bound rejects it)."""
        mol = Chem.MolFromSmiles("[N+](C)(C)(C)(C)C", sanitize=False)
        assert not validator.is_all_valid(mol)


class TestPocketClash:
    def test_no_clash_when_distant(self, validator):
        mol = _with_conformer("CCO")
        pos = np.array(mol.GetConformer().GetPositions()) + np.array([50.0, 0, 0])
        pocket = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        # Teleport the molecule far from the pocket.
        conf = mol.GetConformer()
        from rdkit.Geometry import Point3D

        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(*pos[i]))
        checks = validator.validate(mol, pocket_coords=pocket)
        assert checks["no_pocket_clash"] is True

    def test_clash_detected_when_overlapping(self, validator):
        mol = _with_conformer("CCO")
        pocket = np.array(mol.GetConformer().GetPositions())[:3]  # same spot
        checks = validator.validate(mol, pocket_coords=pocket)
        assert checks["no_pocket_clash"] is False

    def test_missing_conformer_flagged(self, validator):
        mol = Chem.MolFromSmiles("CCO")  # no conformer
        checks = validator.validate(mol)
        assert checks["has_conformer"] is False


class TestFailureReasons:
    def test_reasons_match_failed_checks(self, validator):
        mol = Chem.MolFromSmiles("c1ccccc1")  # flat, no conformer
        checks = validator.validate(mol)
        failed = {k for k, v in checks.items() if not v}
        reasons = set(validator.failure_reasons(mol).keys())
        assert reasons == failed


class TestDescriptors:
    def test_descriptor_block(self, validator):
        mol = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        d = validator.descriptors(mol)
        assert d["molecular_weight"] == pytest.approx(128.17, abs=0.5)
        # 7 carbons: 6 ring sp3 + 1 carbonyl sp2.
        assert d["fsp3"] == pytest.approx(6 / 7)
        # 7 carbons + 2 oxygens.
        assert d["heavy_atoms"] == 9
        assert d["rotatable_bonds"] >= 0
        assert all(isinstance(v, float) for v in d.values())

    def test_none_mol_gives_empty(self, validator):
        assert validator.descriptors(None) == {}
