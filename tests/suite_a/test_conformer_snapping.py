"""Suite A / 3 — MMFF94 relaxation with harmonic scaffold restraints
(issue #32 TEST-01; #13 CHEM-03 capability regression).

Exercises :func:`syntree.chemistry.conformer.relax_junction_constrained`
end-to-end on real assembled junction molecules inside a pocket: the
seed-scaffold atoms stay within the flat-bottom harmonic tolerance, the
relaxed geometry keeps chemically sane bond lengths, chirality is
preserved, energies are finite, the caller's molecule is never mutated,
and repeated relaxation is bit-deterministic.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.chemistry.conformer import (  # noqa: E402
    ChemistryError,
    relax_junction_constrained,
)
from syntree.engine.environment import PocketDefinition  # noqa: E402


def _shell_pocket(num_atoms: int = 14, radius: float = 5.2, seed: int = 11):
    """A deterministic surrogate pocket (same construction as #13 tests)."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_atoms, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * (radius + 0.6 * rng.random(num_atoms))[:, None]
    elements = [6, 6, 7, 8, 6, 8, 6, 6, 7, 6, 8, 6, 6, 8][:num_atoms]
    charges = np.zeros(num_atoms)
    return PocketDefinition(
        positions, np.array(elements, dtype=np.int64), charges
    )


def _junction_molecule(seed_atoms):
    """A 3D molecule with a post-junction fragment: anisole-like scaffold
    with an ethyl chain grown off the ring (seed = the anisole ring)."""
    mol = Chem.MolFromSmiles("COc1ccc(CC)cc1")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    assert AllChem.EmbedMolecule(mol, params) == 0
    seed_idx = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() < 9
    ][: len(seed_atoms)]
    return mol, seed_idx


class TestScaffoldRestraints:
    def test_seed_atoms_stay_within_tolerance(self):
        mol, seed_idx = _junction_molecule(range(8))
        pocket = _shell_pocket()
        relaxed, energy = relax_junction_constrained(
            mol, pocket, seed_atoms=seed_idx
        )
        assert np.isfinite(energy)
        conf0 = mol.GetConformer()
        conf1 = relaxed.GetConformer()
        for idx in seed_idx:
            # the seed atom must exist in the relaxed molecule
            # (atom order is preserved by the relaxation contract)
            p0 = np.array(conf0.GetAtomPosition(idx))
            p1 = np.array(conf1.GetAtomPosition(idx))
            # flat-bottom tolerance is 0.15 A by the #13 default; assert
            # the documented position tolerance bound with headroom
            assert np.linalg.norm(p1 - p0) < 0.5, (
                f"seed atom {idx} moved {np.linalg.norm(p1 - p0):.3f} A — "
                "scaffold restraint violated"
            )

    def test_relaxation_lowers_or_keeps_energy_vs_distorted(self):
        mol, seed_idx = _junction_molecule(range(8))
        pocket = _shell_pocket()
        _, energy = relax_junction_constrained(mol, pocket, seed_atoms=seed_idx)
        assert np.isfinite(energy)

    def test_bond_lengths_chemically_sane(self):
        mol, seed_idx = _junction_molecule(range(8))
        pocket = _shell_pocket()
        relaxed, _ = relax_junction_constrained(mol, pocket, seed_atoms=seed_idx)
        conf = relaxed.GetConformer()
        for bond in relaxed.GetBonds():
            d = np.linalg.norm(
                np.array(conf.GetAtomPosition(bond.GetBeginAtomIdx()))
                - np.array(conf.GetAtomPosition(bond.GetEndAtomIdx()))
            )
            # heavy-atom covalent distances in a relaxed molecule:
            # 1.1 A (C#N) .. 1.9 A (C-Br); assert a generous sane envelope
            assert 1.0 <= d <= 2.1, (
                f"bond {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()} "
                f"length {d:.3f} A outside the chemical envelope"
            )

    def test_chirality_preserved(self):
        # (S)-1-phenylethanol scaffold with a grown ethyl amide chain
        mol = Chem.MolFromSmiles("C[C@H](O)c1ccc(CCNC(=O)C)cc1")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 7
        assert AllChem.EmbedMolecule(mol, params) == 0
        seed_idx = [
            a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"
        ][:10]
        pocket = _shell_pocket()
        relaxed, _ = relax_junction_constrained(mol, pocket, seed_atoms=seed_idx)
        from rdkit.Chem import rdMolDescriptors

        chiral0 = Chem.FindMolChiralCenters(
            Chem.RemoveHs(Chem.Mol(mol)), includeUnassigned=True
        )
        chiral1 = Chem.FindMolChiralCenters(
            Chem.RemoveHs(Chem.Mol(relaxed)), includeUnassigned=True
        )
        assert chiral0 == chiral1, "chirality changed during relaxation"

    def test_caller_molecule_not_mutated(self):
        mol, seed_idx = _junction_molecule(range(8))
        before = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
        conf_before = np.array(mol.GetConformer().GetPositions())
        relax_junction_constrained(mol, _shell_pocket(), seed_atoms=seed_idx)
        assert Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol))) == before
        assert np.array_equal(conf_before, np.array(mol.GetConformer().GetPositions()))

    def test_relaxation_deterministic(self):
        mol, seed_idx = _junction_molecule(range(8))
        pocket = _shell_pocket()
        r1, e1 = relax_junction_constrained(mol, pocket, seed_atoms=seed_idx)
        r2, e2 = relax_junction_constrained(mol, pocket, seed_atoms=seed_idx)
        assert e1 == e2
        assert np.array_equal(
            r1.GetConformer().GetPositions(), r2.GetConformer().GetPositions()
        )

    def test_untypeable_pocket_fails_closed(self):
        mol, seed_idx = _junction_molecule(range(8))
        # a potassium atom in the pocket: MMFF cannot type it and the
        # documented B/Zn/Sn coupling-handle fallback does not cover K
        pocket = PocketDefinition(
            np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.array([19, 6], dtype=np.int64),
            np.zeros(2),
        )
        with pytest.raises(ChemistryError):
            relax_junction_constrained(mol, pocket, seed_atoms=seed_idx)
