"""Flat-bottom soft-core LJ clash + volume floors — issue #14 CHEM-04 acceptance.

Pins the contracts of :mod:`syntree.chemistry.conformer` and its engine
integration:

* **acceptance 1**: at ``r = 0.5 Å`` the flat-bottom soft-core potential
  stays bounded (``< 500 kcal/mol``) where the rigid ``r^-12`` wall
  diverges past ``10^7`` — for the C-C reference pair *and for every
  supported element pair* (the epsilon ceiling makes the bound universal);
* **acceptance 2**: terminal molecules under volume evaluation incur the
  explicit heavy-atom floor penalty below 12 heavy atoms (assembled
  molecules from the real environment, plus the cavity-ratio floor);
* the pinned functional form (flat bottom beyond contact, ``delta = 0.5
  Å^6``, bounded spike, documented ``r_vdw`` jump), analytic gradient,
  torch autograd mirror, atom typing + Lorentz-Berthelot mixing with the
  ceiling;
* the deterministic grid cavity-volume estimator and the two floor
  penalties' shapes/bounds/fail-closed behaviour;
* the engine wiring (D-V2-015): the terminal sub-bin torsion refinement
  objective consumes the production clash potential, and the stationary
  reward ``S_vol`` carries both floors without gaining any batch
  dependence.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.chemistry.conformer import (
    CLASH_SPIKE_CEILING,
    DEFAULT_ATOM_TYPE,
    FLAT_BOTTOM_DELTA,
    HEAVY_ATOM_FLOOR,
    HEAVY_ATOM_FLOOR_WEIGHT,
    LJ_WELL_DEPTHS,
    VDW_RADII,
    VOLUME_RATIO_FLOOR,
    ChemistryError,
    SoftCoreClashPotential,
    atom_type,
    flat_bottom_soft_core_lj,
    flat_bottom_soft_core_lj_gradient,
    flat_bottom_soft_core_lj_torch,
    heavy_atom_floor_penalty,
    pair_epsilon,
    pocket_cavity_volume,
    soft_core_clash_energy,
    volume_fill_ratio,
    volume_floor_penalty,
)
from syntree.engine.environment import (
    JunctionRecord,
    PocketDefinition,
    bin_center,
    set_junction_dihedral,
    soft_core_pocket_energy,
)
from syntree.engine.rl import RewardError, ThreeDReward

# Verified fixtures shared with the engine reconciliation suite (same
# tests directory; pytest's prepend import mode makes them importable).
from test_gflownet_reconciliation import (  # noqa: F401 - shared fixtures
    _fresh_two_step_env,
    _rollout,
    _shell_pocket as _engine_shell_pocket,
)

# ---------------------------------------------------------------------------
# Reference parameters
# ---------------------------------------------------------------------------

C_RADIUS, C_EPS = atom_type(6)
H_RADIUS, H_EPS = atom_type(1)
SIGMA_CC = C_RADIUS + C_RADIUS  # 3.40 Å


# ---------------------------------------------------------------------------
# 1. The pinned potential form — issue #14 acceptance criterion 1
# ---------------------------------------------------------------------------


class TestFlatBottomPotential:
    def test_bounded_at_half_angstrom_cc_pair(self):
        # The pinned acceptance point: a C-C pair at r = 0.5 Å under the
        # *system* pair parameters (Lorentz-Berthelot depth capped by the
        # bounded-spike ceiling).
        eps_cc = pair_epsilon(C_EPS, C_EPS, SIGMA_CC)
        energy = float(flat_bottom_soft_core_lj(0.5, SIGMA_CC, eps_cc))
        assert 0.0 < energy < 500.0, f"C-C V(0.5) = {energy}"

    def test_rigid_wall_would_diverge_at_half_angstrom(self):
        # The contrast the issue pins: the rigid r^-12 wall at the same
        # distance and parameters is > 1e7 kcal/mol.
        eps_cc = pair_epsilon(C_EPS, C_EPS, SIGMA_CC)
        rigid = 4.0 * eps_cc * (SIGMA_CC / 0.5) ** 12
        assert rigid > 1e7

    def test_bounded_for_every_supported_element_pair(self):
        # The epsilon ceiling makes the acceptance bound universal: the
        # total-overlap spike of ANY pair (including Br/I metals in the
        # tables) is <= 500 kcal/mol, hence V(0.5) <= 484.85.
        elements = sorted(set(VDW_RADII) & set(LJ_WELL_DEPTHS))
        assert len(elements) >= 10
        for z_i in elements:
            for z_j in elements:
                r_i, e_i = atom_type(z_i)
                r_j, e_j = atom_type(z_j)
                sigma = r_i + r_j
                eps = pair_epsilon(e_i, e_j, sigma)
                v_half = float(flat_bottom_soft_core_lj(0.5, sigma, eps))
                v_zero = float(flat_bottom_soft_core_lj(0.0, sigma, eps))
                assert v_half < 500.0, f"Z={z_i}/Z={z_j} V(0.5) = {v_half}"
                assert v_zero <= CLASH_SPIKE_CEILING + 1e-9

    def test_flat_bottom_is_bitwise_zero_beyond_contact(self):
        for r in (SIGMA_CC, SIGMA_CC + 1e-9, SIGMA_CC + 5.0, 100.0):
            assert float(flat_bottom_soft_core_lj(r, SIGMA_CC, C_EPS)) == 0.0

    def test_pinned_form_inside_the_branch(self):
        delta = FLAT_BOTTOM_DELTA
        for r in (0.1, 0.5, 1.0, 2.0, 3.39):
            expected = 4.0 * C_EPS * SIGMA_CC ** 6 / (r ** 6 + delta)
            got = float(flat_bottom_soft_core_lj(r, SIGMA_CC, C_EPS))
            assert got == pytest.approx(expected, rel=1e-12)

    def test_delta_constant_is_pinned(self):
        assert FLAT_BOTTOM_DELTA == 0.5  # Å^6, issue-pinned
        assert CLASH_SPIKE_CEILING == 500.0

    def test_monotone_repulsion_as_distance_shrinks(self):
        values = [float(flat_bottom_soft_core_lj(r, SIGMA_CC, C_EPS)) for r in (3.0, 2.0, 1.0, 0.5, 0.1)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_documented_jump_at_contact_is_bounded_and_small(self):
        inside = float(flat_bottom_soft_core_lj(SIGMA_CC - 1e-9, SIGMA_CC, C_EPS))
        outside = float(flat_bottom_soft_core_lj(SIGMA_CC + 1e-9, SIGMA_CC, C_EPS))
        assert outside == 0.0
        assert inside <= 4.0 * C_EPS + 1e-9  # <= ~0.34 kcal/mol for C-C

    def test_vectorised_evaluation(self):
        r = np.array([0.5, 1.0, 3.4, 5.0])
        out = flat_bottom_soft_core_lj(r, np.full(4, SIGMA_CC), np.full(4, C_EPS))
        assert out.shape == (4,)
        assert out[2] == 0.0 and out[3] == 0.0 and out[0] > out[1] > 0.0

    def test_fail_closed_on_bad_inputs(self):
        with pytest.raises(ChemistryError):
            flat_bottom_soft_core_lj(float("nan"), SIGMA_CC, C_EPS)
        with pytest.raises(ChemistryError):
            flat_bottom_soft_core_lj(1.0, 0.0, C_EPS)
        with pytest.raises(ChemistryError):
            flat_bottom_soft_core_lj(1.0, SIGMA_CC, -0.1)
        with pytest.raises(ChemistryError):
            flat_bottom_soft_core_lj(1.0, SIGMA_CC, C_EPS, delta=0.0)


class TestPotentialGradient:
    def test_gradient_matches_finite_difference(self):
        delta = FLAT_BOTTOM_DELTA
        for r in (0.3, 0.8, 1.5, 2.5, 3.0):
            h = 1e-6
            fd = (
                float(flat_bottom_soft_core_lj(r + h, SIGMA_CC, C_EPS))
                - float(flat_bottom_soft_core_lj(r - h, SIGMA_CC, C_EPS))
            ) / (2 * h)
            analytic = float(flat_bottom_soft_core_lj_gradient(r, SIGMA_CC, C_EPS))
            assert analytic == pytest.approx(fd, rel=1e-4, abs=1e-6)

    def test_gradient_zero_at_origin_and_on_flat_bottom(self):
        assert float(flat_bottom_soft_core_lj_gradient(0.0, SIGMA_CC, C_EPS)) == 0.0
        for r in (SIGMA_CC, 4.0, 10.0):
            assert float(flat_bottom_soft_core_lj_gradient(r, SIGMA_CC, C_EPS)) == 0.0

    def test_gradient_is_nonpositive_with_analytic_maximum(self):
        # |dV/dr| = 24 eps sigma^6 r^5 / (r^6 + delta)^2 is maximised at
        # r* = (5 delta / 7)^(1/6) (maximise x^(5/6)/(x+delta)^2 over x=r^6).
        eps_cc = pair_epsilon(C_EPS, C_EPS, SIGMA_CC)
        r = np.linspace(0.0, SIGMA_CC, 2000)
        grad = flat_bottom_soft_core_lj_gradient(r, np.full_like(r, SIGMA_CC), np.full_like(r, eps_cc))
        assert np.all(grad <= 0.0)
        assert np.all(np.isfinite(grad))
        r_star = (5.0 * FLAT_BOTTOM_DELTA / 7.0) ** (1.0 / 6.0)
        expected_max = 24.0 * eps_cc * SIGMA_CC ** 6 * r_star ** 5 / (r_star ** 6 + FLAT_BOTTOM_DELTA) ** 2
        assert np.abs(grad).max() == pytest.approx(expected_max, rel=1e-3)

    def test_torch_autograd_matches_analytic_gradient(self):
        r = torch.tensor([0.3, 0.8, 1.5, 2.5, 3.0], dtype=torch.float64, requires_grad=True)
        sigma = torch.full((5,), SIGMA_CC, dtype=torch.float64)
        eps = torch.full((5,), C_EPS, dtype=torch.float64)
        out = flat_bottom_soft_core_lj_torch(r, sigma, eps)
        out.sum().backward()
        analytic = flat_bottom_soft_core_lj_gradient(
            r.detach().numpy(), np.full(5, SIGMA_CC), np.full(5, C_EPS)
        )
        assert np.allclose(r.grad.numpy(), analytic, rtol=1e-10, atol=1e-12)

    def test_torch_mirror_values_match_numpy_core(self):
        r = np.array([0.2, 1.0, 2.9, 3.4, 6.0])
        numpy_out = flat_bottom_soft_core_lj(r, np.full(5, SIGMA_CC), np.full(5, C_EPS))
        torch_out = flat_bottom_soft_core_lj_torch(
            torch.tensor(r, dtype=torch.float64),
            torch.full((5,), SIGMA_CC, dtype=torch.float64),
            torch.full((5,), C_EPS, dtype=torch.float64),
        ).numpy()
        assert np.allclose(numpy_out, torch_out, rtol=0.0, atol=0.0)


class TestAtomTypingAndMixing:
    def test_bondi_radii_table_is_pinned(self):
        assert VDW_RADII[1] == 1.10   # H
        assert VDW_RADII[6] == 1.70   # C
        assert VDW_RADII[8] == 1.52   # O
        assert VDW_RADII[17] == 1.75  # Cl
        assert VDW_RADII[53] == 1.98  # I

    def test_well_depth_table_is_pinned(self):
        assert LJ_WELL_DEPTHS[1] == 0.015
        assert LJ_WELL_DEPTHS[6] == 0.086
        assert LJ_WELL_DEPTHS[16] == 0.250

    def test_lorentz_berthelot_with_ceiling(self):
        # Heavy-heavy pairs sit at the ceiling; light pairs keep LB.
        sigma_cc = 2 * VDW_RADII[6]
        lb_cc = math.sqrt(C_EPS * C_EPS)
        assert pair_epsilon(C_EPS, C_EPS, sigma_cc) == pytest.approx(
            CLASH_SPIKE_CEILING * FLAT_BOTTOM_DELTA / (4 * sigma_cc ** 6), rel=1e-12
        )
        sigma_hc = VDW_RADII[1] + VDW_RADII[6]
        lb_hc = math.sqrt(H_EPS * C_EPS)
        assert pair_epsilon(H_EPS, C_EPS, sigma_hc) == pytest.approx(lb_hc, rel=1e-12)

    def test_unknown_element_policies(self):
        assert atom_type(30) == DEFAULT_ATOM_TYPE  # Zn: documented fallback
        strict = SoftCoreClashPotential(unknown_element="raise")
        with pytest.raises(ChemistryError, match="Z=30"):
            strict.pair_parameters(
                np.array([6]), np.array([30])
            )
        # default policy falls back (the D-V2-009 oracle precedent)
        default = SoftCoreClashPotential()
        sigma, eps = default.pair_parameters(np.array([6]), np.array([30]))
        assert sigma[0, 0] == pytest.approx(VDW_RADII[6] + DEFAULT_ATOM_TYPE[0])

    def test_fail_closed_on_bad_atomic_numbers(self):
        with pytest.raises(ChemistryError):
            atom_type(0)
        potential = SoftCoreClashPotential()
        with pytest.raises(ChemistryError):
            potential.pair_parameters(np.array([-1]), np.array([6]))


# ---------------------------------------------------------------------------
# 2. The pair-matrix clash engine
# ---------------------------------------------------------------------------


def _simple_pocket() -> PocketDefinition:
    """Four carbon atoms in a plane at ~4 Å from the origin."""
    positions = np.array(
        [
            [4.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, -4.0, 0.0],
        ],
        dtype=np.float64,
    )
    return PocketDefinition(positions, np.array([6, 6, 6, 6], dtype=np.int64), np.zeros(4))


class TestClashEngine:
    def test_zero_energy_when_no_contact(self):
        # Ligand atom at the centre: 4 Å from every pocket atom > sigma_CC.
        ligand = np.array([[0.0, 0.0, 0.0]])
        lig_z = np.array([6])
        pocket = _simple_pocket()
        assert (
            soft_core_clash_energy(ligand, lig_z, pocket.positions, pocket.atomic_numbers)
            == 0.0
        )

    def test_positive_energy_under_overlap(self):
        ligand = np.array([[3.0, 0.0, 0.0]])  # 1 Å from a pocket carbon
        lig_z = np.array([6])
        pocket = _simple_pocket()
        energy = soft_core_clash_energy(ligand, lig_z, pocket.positions, pocket.atomic_numbers)
        assert energy > 0.0
        assert energy < 4 * CLASH_SPIKE_CEILING  # bounded even at hard overlap

    def test_total_equals_pair_matrix_sum(self):
        ligand = np.array([[3.2, 0.3, -0.4], [0.0, 3.9, 0.0]])
        lig_z = np.array([6, 8])
        pocket = _simple_pocket()
        potential = SoftCoreClashPotential()
        matrix = potential.pair_energy(ligand, lig_z, pocket.positions, pocket.atomic_numbers)
        total = potential.total_energy(ligand, lig_z, pocket.positions, pocket.atomic_numbers)
        assert total == pytest.approx(float(matrix.sum()), rel=1e-15)
        per_atom = potential.per_ligand_atom_energy(ligand, lig_z, pocket.positions, pocket.atomic_numbers)
        assert per_atom.shape == (2,)
        assert per_atom.sum() == pytest.approx(total, rel=1e-15)

    def test_deterministic(self):
        ligand = np.array([[3.2, 0.3, -0.4]])
        lig_z = np.array([6])
        pocket = _simple_pocket()
        first = soft_core_clash_energy(ligand, lig_z, pocket.positions, pocket.atomic_numbers)
        second = soft_core_clash_energy(ligand, lig_z, pocket.positions, pocket.atomic_numbers)
        assert first == second

    def test_configuration_is_frozen_semantics(self):
        potential = SoftCoreClashPotential(delta=0.6, spike_ceiling=400.0)
        assert potential.delta == 0.6
        assert potential.spike_ceiling == 400.0
        with pytest.raises(ChemistryError):
            SoftCoreClashPotential(delta=0.0)
        with pytest.raises(ChemistryError):
            SoftCoreClashPotential(unknown_element="ignore")

    def test_fail_closed_on_poisoned_geometry(self):
        pocket = _simple_pocket()
        with pytest.raises(ChemistryError):
            soft_core_clash_energy(
                np.array([[float("nan"), 0.0, 0.0]]), np.array([6]),
                pocket.positions, pocket.atomic_numbers,
            )
        with pytest.raises(ChemistryError):
            soft_core_clash_energy(
                np.zeros((1, 3)), np.array([6, 6]), pocket.positions, pocket.atomic_numbers
            )
        with pytest.raises(ChemistryError):
            soft_core_clash_energy(
                np.zeros((1, 3)), np.array([6]), np.zeros((0, 3)), np.array([], dtype=np.int64)
            )


# ---------------------------------------------------------------------------
# 3. Cavity volume and the floor penalties
# ---------------------------------------------------------------------------


class TestCavityVolume:
    def test_shell_pocket_has_positive_cavity(self):
        # The engine tests' 14-atom shell pocket: the interior void counts.
        pocket = _engine_shell_pocket()
        volume = pocket_cavity_volume(pocket.positions, pocket.atomic_numbers)
        assert volume > 0.0
        # sanity: strictly less than the envelope sphere itself
        centroid = pocket.positions.mean(axis=0)
        offsets = np.linalg.norm(pocket.positions - centroid, axis=1)
        r_env = float(np.max(offsets)) + max(
            VDW_RADII.get(int(z), DEFAULT_ATOM_TYPE[0]) for z in pocket.atomic_numbers
        )
        sphere = (4.0 / 3.0) * math.pi * r_env ** 3
        assert volume < sphere

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        positions = rng.uniform(-4, 4, size=(8, 3))
        numbers = np.full(8, 6, dtype=np.int64)
        first = pocket_cavity_volume(positions, numbers)
        second = pocket_cavity_volume(positions, numbers)
        assert first == second

    def test_monotone_in_pocket_extent(self):
        numbers = np.array([6, 6, 6, 6, 6, 6], dtype=np.int64)
        small = pocket_cavity_volume(
            np.array([[2.0, 0, 0], [-2.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0], [0, 0, 2.0], [0, 0, -2.0]], dtype=np.float64),
            numbers,
        )
        large = pocket_cavity_volume(
            np.array([[4.0, 0, 0], [-4.0, 0, 0], [0, 4.0, 0], [0, -4.0, 0], [0, 0, 4.0], [0, 0, -4.0]], dtype=np.float64),
            numbers,
        )
        assert large > small

    def test_grid_spacing_changes_resolution_not_determinism(self):
        rng = np.random.default_rng(5)
        positions = rng.uniform(-3, 3, size=(6, 3))
        numbers = np.full(6, 7, dtype=np.int64)
        coarse = pocket_cavity_volume(positions, numbers, grid_spacing=1.0)
        fine = pocket_cavity_volume(positions, numbers, grid_spacing=0.5)
        # same order of magnitude (documented estimator resolution)
        assert 0.2 * fine < coarse < 5.0 * fine

    def test_fail_closed(self):
        with pytest.raises(ChemistryError):
            pocket_cavity_volume(np.zeros((2, 3)), np.array([6, 0]))
        with pytest.raises(ChemistryError):
            pocket_cavity_volume(np.zeros((0, 3)), np.array([], dtype=np.int64))
        with pytest.raises(ChemistryError):
            pocket_cavity_volume(np.zeros((2, 3)), np.array([6, 6]), grid_spacing=0.0)
        with pytest.raises(ChemistryError):
            pocket_cavity_volume(
                np.array([[float("inf"), 0, 0], [0, 0, 0]]), np.array([6, 6])
            )


class TestVolumeFloorPenalties:
    def test_ratio_floor_pinned_constant(self):
        assert VOLUME_RATIO_FLOOR == 0.35

    def test_no_penalty_at_or_above_the_ratio(self):
        assert volume_floor_penalty(35.0, 100.0) == 0.0
        assert volume_floor_penalty(80.0, 100.0) == 0.0

    def test_penalty_below_the_ratio_is_bounded_and_monotone(self):
        p50 = volume_floor_penalty(17.5, 100.0)   # ratio 0.175 = floor/2
        p0 = volume_floor_penalty(0.0, 100.0)
        p10 = volume_floor_penalty(10.0, 100.0)
        assert p50 == pytest.approx(-1.0)
        assert p0 == pytest.approx(-2.0)
        assert p10 < p50 < 0.0
        assert -2.0 <= p10 <= 0.0

    def test_fill_ratio(self):
        assert volume_fill_ratio(30.0, 100.0) == pytest.approx(0.3)
        with pytest.raises(ChemistryError):
            volume_fill_ratio(1.0, 0.0)
        with pytest.raises(ChemistryError):
            volume_fill_ratio(float("nan"), 1.0)

    def test_ratio_penalty_fail_closed(self):
        with pytest.raises(ChemistryError):
            volume_floor_penalty(1.0, 100.0, ratio_floor=0.0)
        with pytest.raises(ChemistryError):
            volume_floor_penalty(1.0, -5.0)

    def test_heavy_atom_floor_pinned_constant(self):
        assert HEAVY_ATOM_FLOOR == 12
        assert HEAVY_ATOM_FLOOR_WEIGHT == 2.0

    def test_heavy_atom_floor_shape(self):
        # Acceptance criterion 2: < 12 heavy atoms incurs an explicit penalty.
        assert heavy_atom_floor_penalty(12) == 0.0
        assert heavy_atom_floor_penalty(30) == 0.0
        for n in range(0, 12):
            penalty = heavy_atom_floor_penalty(n)
            assert penalty < 0.0, f"n={n} must incur a penalty"
        assert heavy_atom_floor_penalty(0) == pytest.approx(-2.0)
        assert heavy_atom_floor_penalty(6) == pytest.approx(-1.0)
        assert heavy_atom_floor_penalty(11) == pytest.approx(-2.0 * (1 / 12))
        # monotone: fewer atoms -> stronger penalty
        penalties = [heavy_atom_floor_penalty(n) for n in range(0, 13)]
        assert all(b >= a for a, b in zip(penalties, penalties[1:]))

    def test_heavy_atom_floor_fail_closed(self):
        with pytest.raises(ChemistryError):
            heavy_atom_floor_penalty(-1)
        with pytest.raises(ChemistryError):
            heavy_atom_floor_penalty(5, floor=0)
        with pytest.raises(ChemistryError):
            heavy_atom_floor_penalty(5, weight=-1.0)


# ---------------------------------------------------------------------------
# 4. Engine integration (D-V2-015): reward + torsion refinement
# ---------------------------------------------------------------------------


def _shell_pocket() -> PocketDefinition:
    rng = np.random.default_rng(11)
    directions = rng.normal(size=(14, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * (5.2 + 0.6 * rng.random(14))[:, None]
    elements = [6, 6, 7, 8, 6, 8, 6, 6, 7, 6, 8, 6, 6, 8]
    charges = np.zeros(14)
    charges[2] = -0.8
    charges[8] = 0.8
    return PocketDefinition(positions, np.array(elements, dtype=np.int64), charges)


def _embedded(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None and AllChem.EmbedMolecule(mol, randomSeed=13) == 0
    return mol


class TestRewardVolumeIntegration:
    def test_small_molecule_incurring_explicit_heavy_atom_penalty(self):
        # Acceptance criterion 2: a synthesized-style molecule under volume
        # evaluation with < 12 heavy atoms incurs an explicit penalty.
        pocket = _shell_pocket()
        engine = ThreeDReward(pocket)
        small = _embedded("c1ccccc1")  # benzene: 6 heavy atoms
        reward, components = engine.compute_with_components(small)
        assert components["heavy_atom_count"] == 6
        assert components["heavy_atom_penalty"] == pytest.approx(-1.0)
        assert components["S_vol"] < 0.0
        assert components["S_vol"] == pytest.approx(
            components["ratio_penalty"] + components["heavy_atom_penalty"]
        )

    def test_molecule_at_or_above_the_heavy_atom_floor(self):
        pocket = _shell_pocket()
        engine = ThreeDReward(pocket)
        large = _embedded("CCCCCCCCCCCC")  # dodecane: 12 heavy atoms
        _, components = engine.compute_with_components(large)
        assert components["heavy_atom_count"] == 12
        assert components["heavy_atom_penalty"] == 0.0

    def test_cavity_floor_is_derived_once_and_stationary(self):
        pocket = _shell_pocket()
        engine_a = ThreeDReward(pocket)
        engine_b = ThreeDReward(pocket)
        assert engine_a.pocket_cavity_volume == engine_b.pocket_cavity_volume
        assert engine_a.volume_floor == pytest.approx(
            VOLUME_RATIO_FLOOR * engine_a.pocket_cavity_volume
        )
        molecule = _embedded("c1ccccc1")
        # reward invariant to "batch composition": no batch argument exists
        assert engine_a.compute(molecule) == engine_b.compute(molecule)

    def test_volume_telemetry_is_complete(self):
        engine = ThreeDReward(_shell_pocket())
        _, components = engine.compute_with_components(_embedded("c1ccccc1"))
        assert components["pocket_cavity_volume_angstrom3"] > 0.0
        assert components["volume_ratio"] is not None and 0.0 < components["volume_ratio"] < 1.0
        assert components["volume_ratio_floor"] == VOLUME_RATIO_FLOOR
        assert components["volume_angstrom3"] > 0.0

    def test_explicit_floor_still_supported(self):
        engine = ThreeDReward(_shell_pocket(), volume_floor_angstrom3=120.0)
        assert engine.volume_floor == 120.0
        assert engine.pocket_cavity_volume is None  # explicit floor wins
        _, components = engine.compute_with_components(_embedded("c1ccccc1"))
        assert components["S_vol"] <= 0.0
        assert components["heavy_atom_penalty"] == pytest.approx(-1.0)

    def test_no_pocket_keeps_graph_only_reward(self):
        engine = ThreeDReward()
        _, components = engine.compute_with_components(_embedded("c1ccccc1"))
        assert components["S_vol"] == 0.0
        assert components["heavy_atom_penalty"] == 0.0
        assert components["volume_angstrom3"] is None

    def test_volume_fully_disabled(self):
        engine = ThreeDReward(_shell_pocket(), volume_ratio_floor=None)
        assert engine.volume_floor is None
        _, components = engine.compute_with_components(_embedded("c1ccccc1"))
        assert components["S_vol"] == 0.0

    def test_configuration_validation(self):
        with pytest.raises(RewardError):
            ThreeDReward(_shell_pocket(), volume_ratio_floor=0.0)
        with pytest.raises(RewardError):
            ThreeDReward(_shell_pocket(), heavy_atom_floor=0)
        with pytest.raises(RewardError):
            ThreeDReward(_shell_pocket(), heavy_atom_floor_weight=-1.0)
        with pytest.raises(RewardError):
            ThreeDReward(_shell_pocket(), cavity_grid_spacing=0.0)


class TestRefinementUsesProductionPotential:
    def test_environment_assembled_molecule_runs_the_full_oracle(self):
        # End-to-end via the engine suite's verified fixtures: the two-step
        # assembly, the terminal oracle refining against the production
        # clash potential, and the reward scoring the relaxed molecule —
        # all finite, with the #14 volume floors active.
        env = _fresh_two_step_env(pocket=_engine_shell_pocket())
        _rollout(env, bins=(27, 9))
        relaxed, _, diagnostics = env.finalize_terminal_molecule(return_diagnostics=True)
        assert diagnostics["junctions"], "the refinement must have run"
        for entry in diagnostics["junctions"]:
            assert math.isfinite(entry["objective_after"])
            assert entry["objective_after"] <= entry["objective_before"] + 1e-9

        reward_engine = ThreeDReward(_engine_shell_pocket())
        reward, components = reward_engine.compute_with_components(relaxed)
        assert math.isfinite(reward) and reward >= 1e-4
        # the assembled target (p-dichlorobenzene seed minus both leaving
        # chlorides + methylamine + methanol) has 10 heavy atoms — the
        # explicit heavy-atom floor penalty of acceptance criterion 2 is
        # incurred by a real synthesized molecule under volume evaluation.
        assert components["heavy_atom_count"] == 10
        assert components["heavy_atom_penalty"] == pytest.approx(-2.0 * (12 - 10) / 12)
        assert components["S_vol"] < 0.0
        assert components["pocket_cavity_volume_angstrom3"] > 0.0
        assert components["volume_ratio"] is not None

    def test_objective_values_come_from_the_production_potential(self):
        # The refinement objective must equal the flat-bottom clash energy
        # of the exact pose (not the 12-6 docking surrogate).
        pocket = _engine_shell_pocket()
        env = _fresh_two_step_env(pocket=pocket)
        _rollout(env, bins=(27, 9))
        _, _, diagnostics = env.finalize_terminal_molecule(return_diagnostics=True)
        entry = diagnostics["junctions"][0]
        # reproduce the pre-refinement pose and check both candidate energies
        probe = Chem.Mol(env.molecule)
        junction = JunctionRecord.from_dict(
            json.loads(probe.GetProp("syn_tree_junctions"))[0]
        )
        set_junction_dihedral(probe, junction, bin_center(entry["selected_bin"]))
        positions = np.array(probe.GetConformer().GetPositions(), dtype=np.float64)
        numbers = np.array([a.GetAtomicNum() for a in probe.GetAtoms()], dtype=np.int64)
        production = soft_core_clash_energy(positions, numbers, pocket.positions, pocket.atomic_numbers)
        legacy = soft_core_pocket_energy(positions, numbers, pocket)
        assert entry["objective_before"] == pytest.approx(production, rel=1e-12)
        assert legacy != production  # the two roles are genuinely different physics

    def test_docking_surrogate_remains_available(self):
        # soft_core_pocket_energy keeps its docking-surrogate role (D-V2-015)
        pocket = _simple_pocket()
        # placed near the attractive minimum of one pocket carbon
        ligand = np.array([[0.2, 0.0, 0.0]])
        energy = soft_core_pocket_energy(ligand, np.array([6]), pocket)
        assert math.isfinite(energy)
        assert energy < 0.0  # the attractive branch is the docking surrogate's job
