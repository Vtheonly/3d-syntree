"""Biophysical correctness battery (bug report 4 testing protocol).

These tests go far beyond "does the code run": they verify the *physics and
chemistry* of every generation step:

* Amide/ester junctions stay resonance-planar (tasklist Test 2).
* Stereocenters on sp3-rich synthons never invert during 3D embedding
  (tasklist Test 3).
* Generated conformers are not trapped in high-strain geometries
  (tasklist Test 4, MMFF94 delta-energy protocol).
* The Lennard-Jones contact potential has a real attractive well (depth,
  zero-crossing, far-field decay, torch/numpy parity, differentiability).
* Pocket protonation states at pH 7.4 are assigned from PDB residue names
  (Asp/Glu -1, Arg/Lys +1, protonated His +1).
* The PaiNN pocket encoder actually consumes the formal charges.
* The RL contact reward component rewards pocket occupation, not solvent.
* Handle ranking follows nucleophilicity tiers (aliphatic amine beats
  aniline).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D

from syntree.chemistry.conformer import (
    ConformerEngine,
    is_conjugated_junction,
)
from syntree.chemistry.reactions import ReactionEngine
from syntree.data.featurizer import (
    MolecularFeaturizer,
    assign_pocket_formal_charges,
)
from syntree.models.pocket_encoder import PocketEncoder


@pytest.fixture(scope="module")
def engine() -> ConformerEngine:
    return ConformerEngine()


@pytest.fixture(scope="module")
def rxn() -> ReactionEngine:
    return ReactionEngine()


@pytest.fixture(scope="module")
def generator(tiny_config_with_assets):
    import json
    import os

    from syntree.engine.generator import SBDDGenerator
    from syntree.models.policy import SynTreePolicy

    torch.manual_seed(0)
    cfg = json.loads(json.dumps(tiny_config_with_assets))
    model = SynTreePolicy(cfg)
    return SBDDGenerator(model, cfg, torch.device("cpu"), output_dir=None)


@pytest.fixture()
def pocket_path(assets_dir):
    import os

    return os.path.join(assets_dir["crossdocked_dir"], "sample_pocket.pdb")


def _embed(smiles: str, seed: int = 42):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return ConformerEngine.embed_product(mol, random_seed=seed)


# ---------------------------------------------------------------------------
# Tasklist Test 2: amide & ester planarity
# ---------------------------------------------------------------------------
class TestAmideEsterPlanarity:
    AMIDE_SMARTS = "[C:1](=[O:2])-[N:3]"

    @staticmethod
    def _amide_twists(mol: Chem.Mol):
        """Max deviation (deg) of any amide C(=O)-N dihedral from planarity."""
        conf = mol.GetConformer()
        pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[N:3]")
        worst = 0.0
        for c_idx, o_idx, n_idx in mol.GetSubstructMatches(pattern):
            c_nbrs = [
                a.GetIdx()
                for a in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                if a.GetIdx() not in (o_idx, n_idx)
            ]
            n_nbrs = [
                a.GetIdx()
                for a in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                if a.GetIdx() != c_idx
            ]
            if not c_nbrs or not n_nbrs:
                continue
            deg = abs(
                rdMolTransforms.GetDihedralDeg(
                    conf, c_nbrs[0], c_idx, n_idx, n_nbrs[0]
                )
            )
            # Trans (180) or cis (0); report distance to the nearest planar form.
            deviation = min(abs(deg - 180.0), abs(deg - 0.0), abs(deg - 360.0))
            worst = max(worst, deviation)
        return worst

    def test_amide_junctions_are_strictly_planar(self, engine, rxn):
        """Every newly formed amide bond satisfies resonance planarity (<15 deg)."""
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        core3d = engine.embed_product(core)
        core_noH = Chem.RemoveHs(core3d)
        result = rxn.apply_reaction(
            core_noH, Chem.MolFromSmiles("C1CC(N)CC1"), "amide_coupling"
        )
        assert result is not None
        product = engine.embed_product(
            result.product,
            core_atom_map=result.core_atom_map,
            core_reference=core_noH,
        )
        assert product is not None
        deviation = self._amide_twists(product)
        assert deviation < 15.0, (
            f"Amide bond twisted {deviation:.1f} deg from planar; "
            "violates molecular-orbital resonance requirement"
        )

    def test_arbitrary_policy_angle_never_twists_amide(self, engine, rxn):
        """apply_junction_torsion must refuse a 90 deg twist of an amide bond."""
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        core3d = engine.embed_product(core)
        core_noH = Chem.RemoveHs(core3d)
        result = rxn.apply_reaction(
            core_noH, Chem.MolFromSmiles("C1CC(N)CC1"), "amide_coupling"
        )
        product = engine.embed_product(
            result.product,
            core_atom_map=result.core_atom_map,
            core_reference=core_noH,
        )
        junction = result.junction_bond
        assert is_conjugated_junction(product, junction)

        _, junction_angle, _ = ConformerEngine.apply_junction_torsion(
            product, junction, math.pi / 2  # request a forbidden 90 deg twist
        )
        # The junction snaps to a planar state, not the requested angle.
        planar_dev = min(
            abs(junction_angle - math.pi), abs(junction_angle - 0.0),
            abs(junction_angle + math.pi),
        )
        assert planar_dev < 1e-6, (
            f"junction angle {junction_angle:.3f} rad is not planar"
        )
        assert abs(junction_angle - math.pi / 2) > 0.1

    def test_ester_junctions_are_strictly_planar(self, engine, rxn):
        """Ester C(=O)-O bonds are resonance-locked exactly like amides."""
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        core3d = engine.embed_product(core)
        core_noH = Chem.RemoveHs(core3d)
        result = rxn.apply_reaction(
            core_noH, Chem.MolFromSmiles("CC(C)CO"), "esterification"
        )
        assert result is not None
        product = engine.embed_product(
            result.product,
            core_atom_map=result.core_atom_map,
            core_reference=core_noH,
        )
        assert product is not None
        conf = product.GetConformer()
        a1, a2 = result.junction_bond
        bond = product.GetBondBetweenAtoms(a1, a2)
        # Ester O-C junction: verify via the O-C(=O) dihedral chain.
        c_idx = a1 if product.GetAtomWithIdx(a1).GetAtomicNum() == 6 else a2
        o_idx = a2 if c_idx == a1 else a1
        carbonyl_o = next(
            n.GetIdx()
            for n in product.GetAtomWithIdx(c_idx).GetNeighbors()
            if n.GetAtomicNum() == 8
            and product.GetBondBetweenAtoms(c_idx, n.GetIdx()).GetBondTypeAsDouble() == 2.0
        )
        o_nbr = next(
            n.GetIdx()
            for n in product.GetAtomWithIdx(o_idx).GetNeighbors()
            if n.GetIdx() != c_idx
        )
        deg = abs(
            rdMolTransforms.GetDihedralDeg(conf, carbonyl_o, c_idx, o_idx, o_nbr)
        )
        deviation = min(abs(deg - 180.0), abs(deg - 0.0), abs(deg - 360.0))
        assert deviation < 15.0, (
            f"Ester junction twisted {deviation:.1f} deg from planar"
        )

    def test_free_single_bond_still_rotates_directly(self, engine):
        """A true sp3-sp3 single bond must accept the policy angle as-is."""
        mol = _embed("CCCCO")
        conf = mol.GetConformer()
        # C1-C2 bond (heavy-atom indices after explicit H: heavy atoms first)
        heavy = Chem.RemoveHs(Chem.Mol(mol))
        assert heavy.GetNumAtoms() == 5
        bond = (0, 1)
        assert not is_conjugated_junction(mol, bond)
        _, applied, rotated = ConformerEngine.apply_junction_torsion(
            mol, bond, 1.234
        )
        assert rotated == bond
        assert abs(applied - 1.234) < 0.05

    def test_generated_ligands_keep_amides_planar(
        self, generator, pocket_path
    ):
        """End-to-end: amide dihedrals of full generator rollouts stay planar."""
        result = generator.generate_ligand(
            pocket_path, seed_synthon_idx=0, max_steps=2
        )
        mol = result["rdkit_mol"]
        deviation = self._amide_twists(mol)
        assert deviation < 15.0, (
            f"Generated ligand has an amide twisted {deviation:.1f} deg"
        )


# ---------------------------------------------------------------------------
# Tasklist Test 3: stereochemical integrity
# ---------------------------------------------------------------------------
class TestStereocenterPreservation:
    def test_chiral_synthon_retains_configuration(self, engine, rxn):
        """(S)-alanine amide coupling must not invert during 3D generation."""
        chiral_synthon = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")  # (S)-alanine
        core = Chem.MolFromSmiles("c1ccccc1N")
        result = rxn.apply_reaction(core, chiral_synthon, "amide_coupling")
        assert result is not None

        product_3d = ConformerEngine.embed_product(
            result.product, random_seed=7
        )
        assert product_3d is not None

        Chem.AssignStereochemistry(product_3d, force=True, cleanIt=True)
        centers = Chem.FindMolChiralCenters(
            product_3d, useLegacyImplementation=False
        )
        assert len(centers) == 1, f"expected 1 stereocenter, got {centers}"
        assert centers[0][1] == "S", (
            f"Chiral center inverted during 3D conformer generation: {centers}"
        )

    def test_3d_conformer_agrees_with_graph_cip(self, engine, rxn):
        """The 3D coordinates themselves must realize the (S) configuration."""
        chiral_synthon = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        core = Chem.MolFromSmiles("c1ccccc1N")
        result = rxn.apply_reaction(core, chiral_synthon, "amide_coupling")
        product_3d = ConformerEngine.embed_product(
            result.product, random_seed=123
        )
        probe = Chem.Mol(product_3d)
        Chem.AssignStereochemistryFrom3D(probe)
        centers = Chem.FindMolChiralCenters(
            probe, useLegacyImplementation=False
        )
        assert len(centers) == 1
        assert centers[0][1] == "S", (
            "Embedded 3D geometry realizes the opposite enantiomer"
        )

    @pytest.mark.parametrize("seed", [0, 7, 99, 2024])
    def test_embedding_is_stereo_deterministic(self, seed):
        """Repeated embeddings never flip the stereocenter (enforceChirality)."""
        mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
        product = ConformerEngine.embed_product(mol, random_seed=seed)
        Chem.AssignStereochemistryFrom3D(product)
        centers = Chem.FindMolChiralCenters(
            product, useLegacyImplementation=False
        )
        assert centers[0][1] == "S"


# ---------------------------------------------------------------------------
# Tasklist Test 4: conformer strain energy (MMFF94 delta-E)
# ---------------------------------------------------------------------------
class TestConformerStrainEnergy:
    def test_conformer_strain_energy(self, engine, rxn):
        """As-embedded conformer must be within 15 kcal/mol of its relaxed
        local minimum (no unphysical high-strain geometry)."""
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        core3d = engine.embed_product(core)
        core_noH = Chem.RemoveHs(core3d)
        result = rxn.apply_reaction(
            core_noH, Chem.MolFromSmiles("C1CC(N)CC1"), "amide_coupling"
        )
        bound = engine.embed_product(
            result.product,
            core_atom_map=result.core_atom_map,
            core_reference=core_noH,
        )
        assert bound is not None

        props = AllChem.MMFFGetMoleculeProperties(bound)
        assert props is not None
        ff_bound = AllChem.MMFFGetMoleculeForceField(bound, props)
        bound_energy = ff_bound.CalcEnergy()

        free = Chem.Mol(bound)
        AllChem.MMFFOptimizeMolecule(free, maxIters=2000)
        ff_free = AllChem.MMFFGetMoleculeForceField(free, props)
        free_energy = ff_free.CalcEnergy()

        strain = bound_energy - free_energy
        assert strain < 15.0, (
            f"Ligand trapped in unphysical high-strain conformation "
            f"(strain: {strain:.2f} kcal/mol)"
        )


# ---------------------------------------------------------------------------
# Lennard-Jones contact potential (Mistake 5)
# ---------------------------------------------------------------------------
class TestLennardJones:
    def test_well_depth_and_location(self):
        """Single C-C pair: V has its -epsilon minimum at 2^(1/6)*sigma."""
        # Bondi vdW radii: C = 1.70 A -> sigma = 3.40 A.
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        poc = torch.tensor([[3.40 * 2 ** (1.0 / 6.0), 0.0, 0.0]])
        energy = ConformerEngine.compute_lennard_jones_energy(
            lig, poc,
            ligand_vdw=torch.tensor([1.70]),
            pocket_vdw=torch.tensor([1.70]),
            epsilon=0.10,
        )
        assert energy.item() == pytest.approx(-0.10, abs=1e-4)

    def test_zero_at_contact_distance(self):
        """V(sigma) = 0 exactly: the vdW contact distance is the zero crossing
        (the linearization only acts below 0.8*sigma, far from here)."""
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        poc = torch.tensor([[3.40, 0.0, 0.0]])
        energy = ConformerEngine.compute_lennard_jones_energy(
            lig, poc,
            ligand_vdw=torch.tensor([1.70]),
            pocket_vdw=torch.tensor([1.70]),
        )
        assert abs(energy.item()) < 1e-4

    def test_repulsive_when_overlapping(self):
        """Closer than sigma -> positive (repulsive wall)."""
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        poc = torch.tensor([[2.0, 0.0, 0.0]])
        energy = ConformerEngine.compute_lennard_jones_energy(
            lig, poc,
            ligand_vdw=torch.tensor([1.70]),
            pocket_vdw=torch.tensor([1.70]),
        )
        assert energy.item() > 0.0

    def test_far_field_decays_to_zero(self):
        """Atoms drifting into open solvent get ~zero energy (not reward)."""
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        poc = torch.tensor([[10.0, 0.0, 0.0]])
        energy = ConformerEngine.compute_lennard_jones_energy(
            lig, poc,
            ligand_vdw=torch.tensor([1.70]),
            pocket_vdw=torch.tensor([1.70]),
        )
        assert abs(energy.item()) < 1e-3

    def test_full_overlap_is_bounded(self):
        """The linearized repulsive wall keeps a total overlap finite and
        moderate (~650*epsilon) instead of exploding to 1e12."""
        lig = torch.tensor([[0.0, 0.0, 0.0]])
        poc = torch.tensor([[0.0, 0.0, 0.0]])
        energy = ConformerEngine.compute_lennard_jones_energy(
            lig, poc,
            ligand_vdw=torch.tensor([1.70]),
            pocket_vdw=torch.tensor([1.70]),
            epsilon=0.10,
        )
        assert torch.isfinite(energy).item()
        assert 0.0 < energy.item() < 1e4
        # Analytic bound for the C1-linear continuation at r = 0:
        # V(0) = eps * (52*c^-12 - 28*c^-6) with c = 0.8.
        c = 0.8
        expected = 0.10 * (52.0 * c ** -12 - 28.0 * c ** -6)
        assert energy.item() == pytest.approx(expected, rel=1e-3)

    def test_linear_branch_is_c1_continuous(self):
        """Energy and first derivative match at the 0.8*sigma switch point.

        The gradient just below the switch (linear branch) must equal the
        gradient just above it (exact 6-12 branch) and both must equal the
        analytic slope 24*eps*(c^-7 - 2*c^-13)/sigma.
        """
        lig = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        sigma = 3.40
        eps, c = 0.10, 0.8
        # dV/dr at the switch; the ligand sits at the origin so
        # dr/dx_lig = -1 and dE/dx_lig = -dV/dr.
        analytic_slope = 24.0 * eps * (c ** -7 - 2.0 * c ** -13) / sigma
        expected_grad = -analytic_slope

        grads = {}
        for scale in (0.799, 0.801):  # just below / just above the switch
            poc = torch.tensor([[sigma * scale, 0.0, 0.0]])
            lig.grad = None
            e = ConformerEngine.compute_lennard_jones_energy(
                lig, poc,
                ligand_vdw=torch.tensor([1.70]),
                pocket_vdw=torch.tensor([1.70]),
            )
            e.backward()
            grads[scale] = lig.grad[0, 0].item()

        # Linear branch: gradient is exactly the analytic slope.
        assert grads[0.799] == pytest.approx(expected_grad, rel=1e-4)
        # Exact branch: gradient approaches the same slope from above.
        assert grads[0.801] == pytest.approx(expected_grad, rel=0.05)
        # C1 continuity: the two one-sided gradients agree to a few percent.
        assert (
            abs(grads[0.799] - grads[0.801]) < 0.05 * abs(expected_grad)
        )

    def test_torch_numpy_parity(self):
        rng = np.random.default_rng(0)
        lig = rng.normal(scale=3.0, size=(12, 3))
        poc = rng.normal(scale=3.0, size=(25, 3))
        lv = rng.uniform(1.2, 1.9, size=12)
        pv = rng.uniform(1.2, 1.9, size=25)
        e_t = ConformerEngine.compute_lennard_jones_energy(
            torch.tensor(lig, dtype=torch.float32),
            torch.tensor(poc, dtype=torch.float32),
            torch.tensor(lv, dtype=torch.float32),
            torch.tensor(pv, dtype=torch.float32),
        ).item()
        e_n = ConformerEngine.compute_lennard_jones_np(lig, poc, lv, pv)
        assert e_t == pytest.approx(e_n, rel=1e-3, abs=1e-3)

    def test_differentiable(self):
        """The energy is differentiable w.r.t. ligand coordinates."""
        lig = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        poc = torch.tensor([[3.8, 0.0, 0.0]])
        energy = ConformerEngine.compute_lennard_jones_energy(lig, poc)
        energy.backward()
        assert lig.grad is not None
        assert torch.isfinite(lig.grad).all()
        # Slightly inside the well minimum (3.8 < 3.816): the gradient is
        # nonzero, pushing the atom back toward the potential minimum.
        assert lig.grad.abs().max().item() > 0.0

    def test_empty_inputs(self):
        e = ConformerEngine.compute_lennard_jones_energy(
            torch.zeros(0, 3), torch.zeros(5, 3)
        )
        assert e.item() == 0.0
        assert ConformerEngine.compute_lennard_jones_np(
            np.zeros((0, 3)), np.zeros((5, 3))
        ) == 0.0

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            ConformerEngine.compute_lennard_jones_energy(
                torch.zeros(3), torch.zeros(5, 3)
            )
        with pytest.raises(ValueError):
            ConformerEngine.compute_lennard_jones_energy(
                torch.zeros(5, 3), torch.zeros(4)
            )


# ---------------------------------------------------------------------------
# Pocket protonation states (Mistake 6)
# ---------------------------------------------------------------------------
def _pocket_mol_from_block(block: str) -> Chem.Mol:
    import tempfile, os

    with tempfile.NamedTemporaryFile(
        "w", suffix=".pdb", delete=False
    ) as f:
        f.write(block)
        path = f.name
    try:
        mol = Chem.MolFromPDBFile(path, removeHs=False)
    finally:
        os.unlink(path)
    assert mol is not None and mol.GetNumAtoms() > 0
    return mol


class TestPocketFormalCharges:
    @staticmethod
    def _atom_charge_map(mol: Chem.Mol):
        out = {}
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            key = (
                info.GetResidueName().strip() + "/" + info.GetName().strip()
                if info is not None
                else f"atom{atom.GetIdx()}"
            )
            out[key] = atom.GetIdx()
        return out

    def test_asp_carboxylate_is_negative(self):
        mol = _pocket_mol_from_block(
            "ATOM      1  N   ASP A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  ASP A   1       1.458   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  CB  ASP A   1       2.000   1.400   0.000  1.00 20.00           C\n"
            "ATOM      4  CG  ASP A   1       3.500   1.400   0.000  1.00 20.00           C\n"
            "ATOM      5  OD1 ASP A   1       4.100   0.400   0.000  1.00 20.00           O\n"
            "ATOM      6  OD2 ASP A   1       4.100   2.400   0.000  1.00 20.00           O\n"
            "END\n"
        )
        charges = assign_pocket_formal_charges(mol)
        idx = self._atom_charge_map(mol)
        assert charges[idx["ASP/OD1"]] == pytest.approx(-0.5)
        assert charges[idx["ASP/OD2"]] == pytest.approx(-0.5)
        # Net residue charge = -1.
        assert charges.sum() == pytest.approx(-1.0)

    def test_glu_lys_arg_charges(self):
        mol = _pocket_mol_from_block(
            "ATOM      1  OE1 GLU A   1       0.000   0.000   0.000  1.00 20.00           O\n"
            "ATOM      2  OE2 GLU A   1       1.200   0.000   0.000  1.00 20.00           O\n"
            "ATOM      3  NZ  LYS A   2       3.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      4  NE  ARG A   3       5.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      5  NH1 ARG A   3       6.300   0.000   0.000  1.00 20.00           N\n"
            "ATOM      6  NH2 ARG A   3       5.000   1.300   0.000  1.00 20.00           N\n"
            "ATOM      7  CG  VAL A   4       7.000   0.000   0.000  1.00 20.00           C\n"
            "END\n"
        )
        charges = assign_pocket_formal_charges(mol)
        idx = self._atom_charge_map(mol)
        assert charges[idx["GLU/OE1"]] == pytest.approx(-0.5)
        assert charges[idx["GLU/OE2"]] == pytest.approx(-0.5)
        assert charges[idx["LYS/NZ"]] == pytest.approx(1.0)
        assert (
            charges[idx["ARG/NE"]]
            + charges[idx["ARG/NH1"]]
            + charges[idx["ARG/NH2"]]
        ) == pytest.approx(1.0)
        # Neutral residue untouched.
        assert charges[idx["VAL/CG"]] == 0.0

    def test_protonated_histidine(self):
        mol = _pocket_mol_from_block(
            "ATOM      1  ND1 HIP A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  NE2 HIP A   1       1.400   0.000   0.000  1.00 20.00           N\n"
            "END\n"
        )
        charges = assign_pocket_formal_charges(mol)
        idx = self._atom_charge_map(mol)
        assert charges[idx["HIP/ND1"]] == pytest.approx(0.5)
        assert charges[idx["HIP/NE2"]] == pytest.approx(0.5)

    def test_smiles_molecules_keep_rdkit_charges(self):
        """Molecules without PDB residue info keep RDKit formal charges."""
        mol = Chem.MolFromSmiles("C[NH3+]")
        charges = assign_pocket_formal_charges(mol)
        assert charges[mol.GetSubstructMatch(Chem.MolFromSmarts("[N+]"))[0]] == 1.0
        assert charges.sum() == pytest.approx(1.0)

    def test_featurize_pocket_returns_charge(self):
        mol = _pocket_mol_from_block(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
            "END\n"
        )
        feats = MolecularFeaturizer.featurize_pocket(mol)
        assert "pocket_charge" in feats
        assert feats["pocket_charge"].shape == feats["pocket_z"].shape
        assert feats["pocket_charge"].dtype == torch.float32
        assert feats["pocket_charge"].abs().sum().item() == 0.0  # neutral gly


# ---------------------------------------------------------------------------
# PocketEncoder charge consumption (Mistake 6 wiring)
# ---------------------------------------------------------------------------
class TestPocketEncoderCharges:
    @staticmethod
    def _encoder():
        torch.manual_seed(0)
        return PocketEncoder(hidden_dim=16, num_layers=1, num_radial=6)

    def test_charge_changes_encoding(self):
        enc = self._encoder().eval()
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
        z = torch.tensor([6, 7, 8])
        s0, _, p0 = enc(pos, z)
        s1, _, p1 = enc(pos, z, charges=torch.tensor([0.0, 0.0, 0.0]))
        s2, _, p2 = enc(pos, z, charges=torch.tensor([1.0, -1.0, 0.0]))
        # Neutral charges reproduce the legacy (None) pathway.
        assert torch.allclose(s0, s1)
        assert torch.allclose(p0, p1)
        # Charged residues modulate the encoding.
        assert not torch.allclose(s0, s2)
        assert not torch.allclose(p0, p2)

    def test_charged_atoms_change_more_than_neutral_ones(self):
        enc = self._encoder().eval()
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
        z = torch.tensor([6, 6, 6])
        s0, _, _ = enc(pos, z)
        s1, _, _ = enc(pos, z, charges=torch.tensor([0.0, 1.0, 0.0]))
        delta = (s1 - s0).norm(dim=-1)
        # The charged atom's features move most (message passing mixes a
        # little into neighbours, but the direct effect dominates).
        assert delta[1] >= delta.max() * 0.9

    def test_charge_length_mismatch_raises(self):
        enc = self._encoder()
        pos = torch.zeros(3, 3)
        z = torch.tensor([6, 6, 6])
        with pytest.raises(ValueError):
            enc(pos, z, charges=torch.zeros(2))


# ---------------------------------------------------------------------------
# RL contact reward component (Mistake 5 wiring)
# ---------------------------------------------------------------------------
class TestContactReward:
    @staticmethod
    def _reward(cfg=None):
        from syntree.engine.rl import ThreeDReward

        return ThreeDReward(cfg)

    def test_favourable_contact_positive(self):
        reward = self._reward()
        out = reward.compute(
            Chem.MolFromSmiles("C"), contact_energy=-10.0
        )
        assert out["contact"] > 0.5

    def test_solvent_drift_near_zero(self):
        reward = self._reward()
        out = reward.compute(Chem.MolFromSmiles("C"), contact_energy=0.0)
        assert out["contact"] == pytest.approx(0.0, abs=1e-6)

    def test_clash_negative(self):
        reward = self._reward()
        out = reward.compute(
            Chem.MolFromSmiles("C"), contact_energy=+50.0
        )
        assert out["contact"] < -0.99

    def test_component_in_total_reward(self):
        reward = self._reward({"contact_weight": 2.0})
        base = reward.compute(Chem.MolFromSmiles("C"), contact_energy=0.0)
        fav = reward.compute(Chem.MolFromSmiles("C"), contact_energy=-100.0)
        assert fav["reward"] > base["reward"] + 1.0

    def test_generator_reports_contact_energy(self, generator, pocket_path):
        result = generator.generate_ligand(
            pocket_path, seed_synthon_idx=0, max_steps=1
        )
        assert "contact_energy" in result
        assert np.isfinite(result["contact_energy"])


# ---------------------------------------------------------------------------
# Chemoselectivity (Mistake 4)
# ---------------------------------------------------------------------------
class TestChemoselectivity:
    def test_aliphatic_amine_beats_aniline(self, rxn):
        """A molecule bearing both an aliphatic amine and an aniline must
        route through the aliphatic (more nucleophilic) nitrogen."""
        mol = Chem.MolFromSmiles("NCCc1ccccc1N")
        ranked = rxn.rank_handles(mol)
        assert ranked, "no handles detected"
        assert ranked[0].handle_type == "primary_secondary_amine"
        first = ranked[0].primary_atom
        aromatic_neighbors = any(
            nbr.GetIsAromatic()
            for nbr in mol.GetAtomWithIdx(first).GetNeighbors()
        )
        assert not aromatic_neighbors, (
            "growth routed through the aniline instead of the aliphatic amine"
        )

    def test_ranking_is_deterministic(self, rxn):
        mol = Chem.MolFromSmiles("NCCc1ccc(N)cc1")
        r1 = [h.primary_atom for h in rxn.rank_handles(mol)]
        r2 = [h.primary_atom for h in rxn.rank_handles(mol)]
        assert r1 == r2

    def test_tier_ordering(self, rxn):
        mol = Chem.MolFromSmiles("NCCc1ccc(Br)cc1")  # amine + aryl bromide
        ranked = rxn.rank_handles(mol)
        tiers = [h.reactivity_tier for h in ranked]
        assert tiers == sorted(tiers)
        assert ranked[0].handle_type == "primary_secondary_amine"
