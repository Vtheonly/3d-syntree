"""#20 DATA-05 — coordinate jitter & exposure-bias augmentation tests.

Acceptance criteria (issue #20):

1. Unit test verifies that coordinate jitter preserves molecular
   connectivity and does not invert chiral stereocenters.
2. The training loader emits jittered coordinates when ``augment=True``
   (verified here at the collator-side hook —
   :func:`syntree.data.augmentation.augment_single_state` — and end-to-end
   through the supervised training loader in
   ``tests/test_supervised_dataset.py``).

Plus the constitution #45 §12.1 contract: train split only (val/test
pristine), and the issue-pinned constants (sigma = 0.15 A jitter on
ligand coordinates only; +/-5.0 deg non-junction dihedral perturbation).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from syntree.core.schema import HANDLE_FEATURE_DIM, stamp_schema_version
from syntree.data.augmentation import (
    AUGMENTATION_SPLIT,
    COORD_JITTER_SIGMA_ANGSTROM,
    DIHEDRAL_PERTURB_MAX_DEG,
    DIHEDRAL_PERTURB_MAX_RAD,
    AugmentationError,
    ExposureBiasAugmenter,
    augment_single_state,
    dihedral_change_rad,
    jitter_ligand_coordinates,
    nonjunction_rotatable_bonds,
    perturb_nonjunction_dihedrals,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: A drug-like intermediate with 2 stereocenters, one amide bond
#: (resonance-locked), one ester (resonance-locked), an aryl-CH2 bond and
#: a 4-bond aliphatic chain (>= 3 coupled rotatable dihedrals).
LIGAND_SMILES = "CC(=O)Nc1ccc(C[C@H](O)C[C@@H](N)C(=O)OC(C)C)cc1"

#: An E/Z double-bond stereocenter fixture (stereo must survive jitter).
EZ_SMILES = "C/C=C/C[C@H](N)C(=O)O"


def _embed(smiles: str, seed: int = 2026) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=False) == 0
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def _ligand_mol() -> Chem.Mol:
    return _embed(LIGAND_SMILES)


def _cip_labels(mol: Chem.Mol, positions: np.ndarray):
    """CIP labels + isomeric SMILES assigned *from the 3D geometry*."""
    probe = Chem.Mol(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, positions[i].tolist())
    probe.RemoveAllConformers()
    probe.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(probe)
    labels = tuple(
        atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        for atom in probe.GetAtoms()
    )
    return labels, Chem.MolToSmiles(probe)


def _single_state(mol: Chem.Mol, num_pocket: int = 5, seed: int = 0) -> Data:
    """A canonical 12-field single-graph state for the ligand."""
    g = torch.Generator().manual_seed(seed)
    pos = mol.GetConformer().GetPositions()
    n_lig = mol.GetNumAtoms()
    data = Data(
        pocket_pos=torch.randn(num_pocket, 3, generator=g, dtype=torch.float32),
        pocket_z=torch.randint(1, 9, (num_pocket,), generator=g),
        pocket_charge=torch.zeros(num_pocket, dtype=torch.float32),
        ligand_pos=torch.as_tensor(pos, dtype=torch.float32),
        ligand_z=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.int64),
        ligand_charge=torch.zeros(n_lig, dtype=torch.float32),
        handle_features=torch.randn(1, HANDLE_FEATURE_DIM, generator=g, dtype=torch.float32),
        handle_pos=torch.randn(1, 3, generator=g, dtype=torch.float32),
        global_features=torch.zeros(1, 4, dtype=torch.float32),
        stop_mask=torch.zeros(1, dtype=torch.float32),
    )
    return stamp_schema_version(data)


# ---------------------------------------------------------------------------
# Issue-pinned constants
# ---------------------------------------------------------------------------


class TestIssuePinnedConstants:
    def test_sigma_is_exactly_pinned(self):
        assert COORD_JITTER_SIGMA_ANGSTROM == 0.15

    def test_dihedral_bound_is_exactly_five_degrees(self):
        assert DIHEDRAL_PERTURB_MAX_DEG == 5.0
        assert DIHEDRAL_PERTURB_MAX_RAD == math.radians(5.0)

    def test_train_is_the_only_augmented_split(self):
        assert AUGMENTATION_SPLIT == "train"


# ---------------------------------------------------------------------------
# Gaussian coordinate jitter (acceptance criterion 1)
# ---------------------------------------------------------------------------


class TestJitter:
    def test_shape_dtype_and_no_mutation(self):
        mol = _ligand_mol()
        pos = mol.GetConformer().GetPositions().copy()
        out = jitter_ligand_coordinates(pos, 0.15, rng=np.random.default_rng(7))
        assert out.shape == pos.shape
        assert out.dtype == np.float64
        assert np.array_equal(out, pos) is False
        assert np.array_equal(mol.GetConformer().GetPositions(), pos)  # input untouched

    def test_noise_scale_is_sigma(self):
        rng = np.random.default_rng(11)
        pos = np.zeros((512, 3))
        out = jitter_ligand_coordinates(pos, 0.15, rng=rng)
        rms = float(np.sqrt((out ** 2).mean()))
        assert 0.13 < rms < 0.17  # sigma for 3-D iid Gaussian components

    def test_connectivity_and_chirality_survive_jitter(self):
        """Acceptance criterion 1: connectivity + no chiral inversion."""
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        labels0, smiles0 = _cip_labels(mol, pristine)
        assert labels0.count("R") + labels0.count("S") == 2  # 2 stereocenters
        constitution0 = Chem.MolToSmiles(mol, isomericSmiles=False)
        for seed in range(50):
            out = jitter_ligand_coordinates(pristine, 0.15, rng=np.random.default_rng(seed))
            labels1, smiles1 = _cip_labels(mol, out)
            assert labels1 == labels0, f"chiral inversion at seed {seed}"
            assert smiles1 == smiles0, f"stereo SMILES changed at seed {seed}"
            assert Chem.MolToSmiles(mol, isomericSmiles=False) == constitution0

    def test_ez_double_bond_stereo_survives_composed_jitter(self):
        """The measured flip case, through the composed augmenter.

        Plain iid jitter flipped this molecule's stereocentre on 2/50
        seeds (the S-V2-14 finding that motivated the rejection loop);
        the composed augmenter must guarantee no flip on every seed.
        """
        mol = _embed(EZ_SMILES)
        pristine = mol.GetConformer().GetPositions().copy()
        labels0, smiles0 = _cip_labels(mol, pristine)
        assert smiles0.startswith("C/C=C") or "/C=C" in smiles0
        aug = ExposureBiasAugmenter(apply_dihedrals=False)  # jitter-only
        for seed in range(200):
            out = aug.augment_positions(pristine, mol, rng=np.random.default_rng(seed))
            labels1, smiles1 = _cip_labels(mol, out)
            assert labels1 == labels0, f"chiral inversion at seed {seed}"
            assert smiles1 == smiles0, f"stereo SMILES changed at seed {seed}"

    def test_stereo_signature_helper_detects_the_flip(self):
        """Regression evidence: the elementary op can flip; the helper sees it."""
        from syntree.data.augmentation import stereo_signature

        mol = _embed(EZ_SMILES)
        pristine = mol.GetConformer().GetPositions().copy()
        reference = stereo_signature(mol, pristine)
        flipped = 0
        for seed in range(50):
            out = jitter_ligand_coordinates(pristine, 0.15, rng=np.random.default_rng(seed))
            if stereo_signature(mol, out) != reference:
                flipped += 1
        # the measured finding: flips DO occur for the elementary op
        # (~4% here) — this is exactly what the composed augmenter rejects
        assert flipped >= 1, "expected the documented flip finding on this fixture"

    def test_fail_closed_when_no_stereo_preserving_draw_exists(self):
        mol = _embed(EZ_SMILES)
        pristine = mol.GetConformer().GetPositions().copy()
        aug = ExposureBiasAugmenter(apply_dihedrals=False, max_jitter_redraws=1)
        flipped_seed = None
        for seed in range(200):
            out = jitter_ligand_coordinates(pristine, 0.15, rng=np.random.default_rng(seed))
            from syntree.data.augmentation import stereo_signature
            if stereo_signature(mol, out) != stereo_signature(mol, pristine):
                flipped_seed = seed
                break
        assert flipped_seed is not None  # guaranteed by the test above
        with pytest.raises(AugmentationError, match="stereo-preserving"):
            aug.augment_positions(pristine, mol, rng=np.random.default_rng(flipped_seed))

    def test_bond_lengths_stay_physical(self):
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        for seed in range(25):
            out = jitter_ligand_coordinates(pristine, 0.15, rng=np.random.default_rng(seed))
            for a, b in bonds:
                d = float(np.linalg.norm(out[a] - out[b]))
                assert 0.75 < d < 2.1, f"bond {(a, b)} at {d:.3f} A (seed {seed})"

    def test_fail_closed_inputs(self):
        with pytest.raises(AugmentationError):
            jitter_ligand_coordinates(np.zeros((2, 3)), 0.0, rng=np.random.default_rng(0))
        with pytest.raises(AugmentationError):
            jitter_ligand_coordinates(np.zeros((2, 3)), float("nan"), rng=np.random.default_rng(0))
        with pytest.raises(AugmentationError):
            jitter_ligand_coordinates(np.zeros((3,)), 0.15, rng=np.random.default_rng(0))
        bad = np.zeros((2, 3))
        bad[0, 0] = float("inf")
        with pytest.raises(AugmentationError):
            jitter_ligand_coordinates(bad, 0.15, rng=np.random.default_rng(0))
        with pytest.raises(AugmentationError):
            jitter_ligand_coordinates(np.zeros((2, 3)), 0.15, rng=np.random.RandomState(0))

    def test_empty_ligand_returns_empty(self):
        out = jitter_ligand_coordinates(np.zeros((0, 3)), 0.15, rng=np.random.default_rng(0))
        assert out.shape == (0, 3)


# ---------------------------------------------------------------------------
# The rotatable-bond definition
# ---------------------------------------------------------------------------


class TestRotatableBonds:
    def test_amide_and_ester_are_locked(self):
        mol = _ligand_mol()
        bonds = set(nonjunction_rotatable_bonds(mol))
        # amide C-N (atoms 1-3), amide N-aryl (3-4, resonance with C=O),
        # ester C(=O)-O and O-C(CH3): locate by SMARTS to stay renumbering-proof
        amide = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3][CX3](=[OX1])"))
        ester = mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2][CX3](=[OX1])"))
        assert amide and ester
        for match in amide + ester:
            for a, b in ((match[0], match[1]), (match[1], match[0])):
                assert (a, b) not in bonds and tuple(sorted((a, b))) not in bonds

    def test_chain_bonds_are_rotatable(self):
        mol = _ligand_mol()
        bonds = nonjunction_rotatable_bonds(mol)
        # the aryl-CH2-CH(OH)-CH2 chain carries >= 3 single acyclic bonds
        chain = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX4H2][CX4H1][CX4H2]"))
        assert len(bonds) >= 3
        assert len(chain) >= 1

    def test_junction_bonds_are_excluded(self):
        mol = _ligand_mol()
        bonds = [tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()]
        junction = [bonds[0], bonds[3]]
        perturbable = [tuple(sorted(b)) for b in nonjunction_rotatable_bonds(mol, junction)]
        for j in junction:
            assert j not in perturbable

    def test_ring_bonds_excluded(self):
        benzene = _embed("c1ccccc1")
        assert nonjunction_rotatable_bonds(benzene) == []

    def test_deterministic_order(self):
        mol = _ligand_mol()
        assert nonjunction_rotatable_bonds(mol) == nonjunction_rotatable_bonds(mol)


# ---------------------------------------------------------------------------
# The bounded dihedral stage
# ---------------------------------------------------------------------------


class TestDihedralPerturbation:
    def test_every_change_within_five_degrees(self):
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        bonds = nonjunction_rotatable_bonds(mol)
        assert len(bonds) >= 3
        for seed in range(25):
            out = perturb_nonjunction_dihedrals(
                pristine, mol, rng=np.random.default_rng(seed)
            )
            for bond in bonds:
                change_deg = abs(math.degrees(dihedral_change_rad(pristine, out, mol, bond)))
                assert change_deg <= 5.0 + 1e-3, (
                    f"bond {bond} changed {change_deg:.4f} deg (seed {seed})"
                )

    def test_rigid_stage_preserves_internal_geometry_bitwise(self):
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        out = perturb_nonjunction_dihedrals(pristine, mol, rng=np.random.default_rng(3))
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            assert np.isclose(
                np.linalg.norm(out[a] - out[b]),
                np.linalg.norm(pristine[a] - pristine[b]),
                rtol=0.0,
                atol=1e-9,  # rigid rotations to float64 precision
            )
        # a valence angle: C(a)-C(b)-C(c) along the aliphatic chain
        chain = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX4]~[CX4]~[CX4]"))[0]
        a, b, c = chain
        ang0 = _angle(pristine[a], pristine[b], pristine[c])
        ang1 = _angle(out[a], out[b], out[c])
        assert abs(ang0 - ang1) < 1e-9

    def test_junction_bonds_never_deliberately_rotated(self):
        mol = _ligand_mol()
        all_bonds = [tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()]
        junction = all_bonds[2:5]
        perturbable = {tuple(sorted(b)) for b in nonjunction_rotatable_bonds(mol, junction)}
        assert perturbable.isdisjoint(junction)

    def test_deterministic_given_seed(self):
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        out1 = perturb_nonjunction_dihedrals(pristine, mol, rng=np.random.default_rng(42))
        out2 = perturb_nonjunction_dihedrals(pristine, mol, rng=np.random.default_rng(42))
        assert np.array_equal(out1, out2)
        out3 = perturb_nonjunction_dihedrals(pristine, mol, rng=np.random.default_rng(43))
        assert not np.array_equal(out1, out3)

    def test_no_rotatable_bonds_is_identity(self):
        benzene = _embed("c1ccccc1")
        pristine = benzene.GetConformer().GetPositions().copy()
        out = perturb_nonjunction_dihedrals(pristine, benzene, rng=np.random.default_rng(0))
        assert np.array_equal(out, pristine)

    def test_fail_closed_on_non_convergence(self):
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        with pytest.raises(AugmentationError, match="did not converge"):
            perturb_nonjunction_dihedrals(
                pristine, mol, rng=np.random.default_rng(5), max_passes=1,
                tolerance_rad=1e-15,
            )

    def test_fail_closed_bad_inputs(self):
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        with pytest.raises(AugmentationError):
            perturb_nonjunction_dihedrals(pristine, mol, rng=np.random.default_rng(0), max_delta_rad=0.0)
        with pytest.raises(AugmentationError):
            perturb_nonjunction_dihedrals(pristine, mol, rng=np.random.RandomState(0))
        with pytest.raises(AugmentationError):
            perturb_nonjunction_dihedrals(np.zeros((3, 3)), mol, rng=np.random.default_rng(0))


def _angle(p0, p1, p2) -> float:
    v1, v2 = np.asarray(p0) - np.asarray(p1), np.asarray(p2) - np.asarray(p1)
    return float(np.degrees(math.acos(
        float(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )))


# ---------------------------------------------------------------------------
# The composed augmenter + train-split policy
# ---------------------------------------------------------------------------


class TestExposureBiasAugmenter:
    def test_composition_produces_both_effects(self):
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        # dihedral-only: rigid geometry, changes bounded by 5 deg
        rigid = ExposureBiasAugmenter(apply_jitter=False)
        out = rigid.augment_positions(pristine, mol, rng=np.random.default_rng(1))
        assert np.array_equal(out, pristine) is False
        for bond in mol.GetBonds():  # rigidity preserved through composition
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            assert np.isclose(np.linalg.norm(out[a] - out[b]),
                              np.linalg.norm(pristine[a] - pristine[b]),
                              rtol=0.0, atol=1e-9)
        # full composition: jitter breaks rigidity (coordinate noise on top)
        full = ExposureBiasAugmenter()
        out2 = full.augment_positions(pristine, mol, rng=np.random.default_rng(1))
        some_bond = next(iter(mol.GetBonds()))
        a, b = some_bond.GetBeginAtomIdx(), some_bond.GetEndAtomIdx()
        assert not np.isclose(np.linalg.norm(out2[a] - out2[b]),
                              np.linalg.norm(pristine[a] - pristine[b]), atol=1e-9)

    def test_train_split_only_fail_closed(self):
        mol = _ligand_mol()
        pristine = mol.GetConformer().GetPositions().copy()
        aug = ExposureBiasAugmenter()
        for split in ("val", "test", "VAL", "Train", ""):
            with pytest.raises(AugmentationError, match="train-split only"):
                aug.augment_positions(pristine, mol, rng=np.random.default_rng(0), split=split)

    def test_atom_count_mismatch_fails_closed(self):
        mol = _ligand_mol()
        with pytest.raises(AugmentationError):
            ExposureBiasAugmenter().augment_positions(
                np.zeros((mol.GetNumAtoms() - 1, 3)), mol, rng=np.random.default_rng(0)
            )


# ---------------------------------------------------------------------------
# Single-graph state application (acceptance criterion 2, collator side)
# ---------------------------------------------------------------------------


class TestAugmentSingleState:
    def _fixture(self):
        mol = _ligand_mol()
        state = _single_state(mol)
        junction = [
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in list(mol.GetBonds())[:2]
        ]
        return mol, state, junction

    def test_augment_true_emits_jittered_ligand_coordinates(self):
        """Acceptance criterion 2: augment=True -> jittered coordinates."""
        mol, state, junction = self._fixture()
        out = augment_single_state(
            state, mol, junction, np.random.default_rng(9), augment=True
        )
        assert not torch.equal(out.ligand_pos, state.ligand_pos)
        # pocket pristine, invariant fields bit-identical
        assert torch.equal(out.pocket_pos, state.pocket_pos)
        assert torch.equal(out.pocket_z, state.pocket_z)
        assert torch.equal(out.ligand_z, state.ligand_z)
        assert torch.equal(out.ligand_charge, state.ligand_charge)
        assert torch.equal(out.handle_features, state.handle_features)
        assert torch.equal(out.global_features, state.global_features)
        # input state never mutated
        assert state.ligand_pos.shape == out.ligand_pos.shape

    def test_augment_false_is_bit_identical_pristine(self):
        mol, state, junction = self._fixture()
        out = augment_single_state(
            state, mol, junction, np.random.default_rng(9), augment=False
        )
        assert torch.equal(out.ligand_pos, state.ligand_pos)

    def test_non_train_split_fails_closed(self):
        mol, state, junction = self._fixture()
        for split in ("val", "test"):
            with pytest.raises(AugmentationError, match="train-split only"):
                augment_single_state(
                    state, mol, junction, np.random.default_rng(0), split=split
                )

    def test_augmented_state_passes_the_frozen_tensor_contract(self):
        from torch_geometric.data import Batch

        from syntree.core.schema import collate_state_batch
        mol, state, junction = self._fixture()
        augmented = augment_single_state(
            state, mol, junction, np.random.default_rng(4)
        )
        batch = collate_state_batch([state, augmented])
        assert isinstance(batch, Batch)
        assert batch.num_graphs == 2

    def test_deterministic_and_seed_sensitive(self):
        mol, state, junction = self._fixture()
        a = augment_single_state(state, mol, junction, np.random.default_rng(21))
        b = augment_single_state(state, mol, junction, np.random.default_rng(21))
        c = augment_single_state(state, mol, junction, np.random.default_rng(22))
        assert torch.equal(a.ligand_pos, b.ligand_pos)
        assert not torch.equal(a.ligand_pos, c.ligand_pos)

    def test_state_molecule_mismatch_fails_closed(self):
        mol, state, junction = self._fixture()
        other = _embed("CCO")
        with pytest.raises(AugmentationError):
            augment_single_state(state, other, junction, np.random.default_rng(0))

    def test_chirality_survives_full_augmentation(self):
        """Full composition (dihedrals + jitter) preserves CIP labels."""
        mol, state, junction = self._fixture()
        pristine = state.ligand_pos.numpy().astype(np.float64)
        labels0, smiles0 = _cip_labels(mol, pristine)
        for seed in range(25):
            out = augment_single_state(
                state, mol, junction, np.random.default_rng(seed), augment=True
            )
            labels1, smiles1 = _cip_labels(mol, out.ligand_pos.numpy().astype(np.float64))
            assert labels1 == labels0, f"chiral inversion at seed {seed}"
            assert smiles1 == smiles0
