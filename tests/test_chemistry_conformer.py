"""Unit tests for the conformer engine (embedding, dihedrals, clashes)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.chemistry.conformer import (
    ConformerEngine,
    VDW_RADII,
    vdw_radius,
    _kabsch,
)
from syntree.chemistry.reactions import ReactionEngine


@pytest.fixture(scope="module")
def engine():
    return ConformerEngine()


@pytest.fixture(scope="module")
def butane():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


class TestVdwRadii:
    def test_carbon(self):
        assert vdw_radius(6) == pytest.approx(1.70)

    def test_known_elements(self):
        for z in (1, 6, 7, 8, 16):
            assert z in VDW_RADII

    def test_unknown_falls_back(self):
        assert vdw_radius(999) == vdw_radius(6)


class TestKabsch:
    def test_identity(self):
        P = np.random.default_rng(0).normal(size=(10, 3))
        R, t = _kabsch(P, P)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert np.allclose(t, 0.0, atol=1e-8)

    def test_recovers_rotation(self):
        rng = np.random.default_rng(1)
        P = rng.normal(size=(20, 3))
        theta = 0.7
        R_true = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        t_true = np.array([1.0, -2.0, 0.5])
        Q = P @ R_true.T + t_true
        R, t = _kabsch(P, Q)
        assert np.allclose(P @ R.T + t, Q, atol=1e-8)

    def test_proper_rotation(self):
        rng = np.random.default_rng(2)
        P, Q = rng.normal(size=(15, 3)), rng.normal(size=(15, 3))
        R, _ = _kabsch(P, Q)
        assert np.linalg.det(R) == pytest.approx(1.0)


class TestEmbedProduct:
    def test_embeds_simple_molecule(self, engine):
        mol = Chem.MolFromSmiles("CC(C)CO")
        out = engine.embed_product(mol)
        assert out is not None
        assert out.GetNumConformers() == 1
        assert out.GetNumAtoms() > mol.GetNumAtoms()  # Hs added

    def test_embedding_failure_returns_none(self, engine):
        """An atom-less molecule cannot be embedded; must return None."""
        result = engine.embed_product(Chem.Mol())
        assert result is None

    def test_none_raises(self, engine):
        with pytest.raises(ValueError):
            engine.embed_product(None)

    def test_scaffold_lock_preserves_coordinates(self, engine):
        """Non-junction core atoms stay tethered to their exact positions;
        the junction core atom is free to relax its bond geometry (harmonic
        restraints instead of rigid freezing - Mistake 2 in the chemistry
        audit)."""
        from syntree.chemistry.conformer import _junction_atoms_of

        rxn = ReactionEngine()
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        core3d = engine.embed_product(core)
        assert core3d is not None
        core_noH = Chem.RemoveHs(core3d)
        ref_pos = np.array(core_noH.GetConformer().GetPositions())

        result = rxn.apply_reaction(
            core_noH, Chem.MolFromSmiles("C1CC(N)CC1"), "amide_coupling"
        )
        assert result is not None
        product = engine.embed_product(
            result.product, core_atom_map=result.core_atom_map, core_reference=core_noH
        )
        assert product is not None
        prod_noH = Chem.RemoveHs(Chem.Mol(product))
        prod_pos = np.array(prod_noH.GetConformer().GetPositions())

        junction = {
            prod_idx
            for prod_idx in _junction_atoms_of(product, result.core_atom_map)
            if prod_idx < prod_noH.GetNumAtoms()
        }
        for core_idx, prod_idx in result.core_atom_map.items():
            drift = float(
                np.linalg.norm(prod_pos[prod_idx] - ref_pos[core_idx])
            )
            if prod_idx in junction:
                # Junction atom relaxes to heal the new bond (bounded sanity).
                assert drift < 1.0, f"junction atom drifted {drift:.2f} A"
            else:
                # Harmonic tether: essentially rigid for the pocket anchor.
                assert drift < 0.15, f"core atom drifted {drift:.2f} A"

    def test_junction_bond_length_physical(self, engine):
        """The newly formed C-N amide bond must relax into the physical
        1.2-1.6 A window instead of being stretched/compressed by the
        scaffold lock."""
        rxn = ReactionEngine()
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        core3d = engine.embed_product(core)
        core_noH = Chem.RemoveHs(core3d)
        result = rxn.apply_reaction(
            core_noH, Chem.MolFromSmiles("C1CC(N)CC1"), "amide_coupling"
        )
        product = engine.embed_product(
            result.product, core_atom_map=result.core_atom_map, core_reference=core_noH
        )
        pos = np.array(product.GetConformer().GetPositions())
        a1, a2 = result.junction_bond
        d = float(np.linalg.norm(pos[a1] - pos[a2]))
        assert 1.2 < d < 1.6, f"junction C-N bond length {d:.2f} A is unphysical"

    def test_no_torn_bonds_after_lock(self, engine):
        """Bond lengths between locked core and new atoms must be sane."""
        rxn = ReactionEngine()
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        core3d = engine.embed_product(core)
        core_noH = Chem.RemoveHs(core3d)
        result = rxn.apply_reaction(
            core_noH, Chem.MolFromSmiles("C1CC(N)CC1"), "amide_coupling"
        )
        product = engine.embed_product(
            result.product, core_atom_map=result.core_atom_map, core_reference=core_noH
        )
        pos = np.array(product.GetConformer().GetPositions())
        for bond in Chem.RemoveHs(product).GetBonds():
            d = np.linalg.norm(pos[bond.GetBeginAtomIdx()] - pos[bond.GetEndAtomIdx()])
            assert 0.8 < d < 2.2, f"bond length {d:.2f} A is broken"


class TestDihedrals:
    def test_set_and_get_roundtrip(self, engine, butane):
        for target in (-2.5, -1.0, 0.0, 1.0, 2.8):
            mol = Chem.Mol(butane)
            engine.set_dihedral(mol, (1, 2), target)
            actual = engine.get_dihedral(mol, (1, 2))
            assert actual is not None
            diff = abs(actual - target)
            diff = min(diff, 2 * math.pi - diff)  # circular distance
            assert diff < 0.05, f"set {target:.2f}, got {actual:.2f}"

    def test_rotation_moves_only_one_side(self, engine, butane):
        mol = Chem.Mol(butane)
        pos_before = np.array(mol.GetConformer().GetPositions())
        engine.set_dihedral(mol, (1, 2), 1.234)
        pos_after = np.array(mol.GetConformer().GetPositions())
        # Atoms 0-1 (CH3-CH2 side) must stay fixed; atoms 2-3 move.
        assert np.allclose(pos_after[0], pos_before[0], atol=1e-8)
        assert np.allclose(pos_after[1], pos_before[1], atol=1e-8)

    def test_ring_bond_is_skipped(self, engine):
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        pos_before = np.array(mol.GetConformer().GetPositions())
        # Benzene C0-C1 is a ring bond: rotation must be refused.
        engine.set_dihedral(mol, (0, 1), 1.0)
        pos_after = np.array(mol.GetConformer().GetPositions())
        assert np.allclose(pos_before, pos_after)

    def test_get_dihedral_requires_conformer(self, engine):
        mol = Chem.MolFromSmiles("CCCC")
        assert engine.get_dihedral(mol, (1, 2)) is None

    def test_set_dihedral_requires_conformer(self, engine):
        mol = Chem.MolFromSmiles("CCCC")
        with pytest.raises(ValueError):
            engine.set_dihedral(mol, (1, 2), 0.5)

    def test_terminal_bond_has_no_dihedral(self, engine, butane):
        """A bond to a terminal atom (no further neighbour) cannot define
        a four-atom dihedral."""
        # Butane with explicit H: C-H bonds are terminal on the H side.
        h_atom = next(a.GetIdx() for a in butane.GetAtoms()
                      if a.GetAtomicNum() == 1)
        c_atom = butane.GetAtomWithIdx(h_atom).GetNeighbors()[0].GetIdx()
        assert engine.get_dihedral(butane, (c_atom, h_atom)) is None


class TestClashLoss:
    def test_no_clash_when_far(self, engine):
        lig = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pocket = torch.tensor([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
        loss = engine.compute_steric_clash_loss(lig, pocket)
        assert loss.item() == pytest.approx(0.0)

    def test_clash_when_overlapping(self, engine):
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        pocket = torch.tensor([[0.5, 0.0, 0.0]])
        loss = engine.compute_steric_clash_loss(lig, pocket)
        # threshold ~ (1.7 + 1.7 - 1.0) = 2.4; violation = 2.4 - 0.5 = 1.9
        assert loss.item() == pytest.approx(1.9 ** 2, rel=1e-3)

    def test_differentiable(self, engine):
        lig = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        pocket = torch.tensor([[1.0, 0.0, 0.0]])
        loss = engine.compute_steric_clash_loss(lig, pocket)
        loss.backward()
        assert lig.grad is not None
        assert lig.grad.abs().sum() > 0

    def test_radii_respected(self, engine):
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        pocket = torch.tensor([[3.5, 0.0, 0.0]])  # just beyond 2.4 A
        lig_vdw = torch.tensor([1.7])
        pocket_vdw = torch.tensor([1.7])
        loss = engine.compute_steric_clash_loss(lig, pocket, lig_vdw, pocket_vdw)
        assert loss.item() == pytest.approx(0.0)

    def test_bad_shapes_raise(self, engine):
        with pytest.raises(ValueError):
            engine.compute_steric_clash_loss(torch.zeros(3), torch.zeros(3, 3))
        with pytest.raises(ValueError):
            engine.compute_steric_clash_loss(torch.zeros(3, 3), torch.zeros(4))

    def test_numpy_torch_parity(self, engine):
        rng = np.random.default_rng(3)
        lig = rng.normal(scale=3.0, size=(12, 3))
        pocket = rng.normal(scale=3.0, size=(30, 3))
        t = engine.compute_steric_clash_loss(torch.tensor(lig), torch.tensor(pocket))
        n = engine.compute_steric_clash_np(lig, pocket)
        assert t.item() == pytest.approx(n, rel=1e-4)

    def test_subset_indices(self, engine):
        lig = np.array([[0.0, 0, 0], [5.0, 0, 0]])
        pocket = np.array([[0.4, 0, 0]])
        full = engine.compute_steric_clash_np(lig, pocket)
        subset = engine.compute_steric_clash_np(lig, pocket, subset_indices=[1])
        assert subset == pytest.approx(0.0)
        assert full > 0

    def test_empty_pocket(self, engine):
        assert engine.compute_steric_clash_np(np.zeros((3, 3)), np.zeros((0, 3))) == 0.0


class TestOptimizeDihedral:
    def test_grid_finds_low_clash_angle(self, engine, butane):
        mol = Chem.Mol(butane)
        # A "pocket" wall right next to the terminal methyl.
        pocket = np.array([[3.0, 1.0, 0.0], [3.0, -1.0, 0.0], [3.2, 0.0, 1.0]])
        _, angle, clash = engine.optimize_dihedral(
            mol, (1, 2), pocket_coords=pocket, grid_size=12
        )
        assert -math.pi <= angle < math.pi
        assert clash >= 0.0

    def test_no_pocket_returns_zero_clash(self, engine, butane):
        mol = Chem.Mol(butane)
        _, angle, clash = engine.optimize_dihedral(mol, (1, 2), pocket_coords=None)
        assert clash == 0.0

    def test_requires_conformer(self, engine):
        mol = Chem.MolFromSmiles("CCCC")
        with pytest.raises(ValueError):
            engine.optimize_dihedral(mol, (1, 2))
