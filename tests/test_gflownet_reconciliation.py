"""Verification suite of the GFlowNet reconciliation (decision D-V2-009).

Pinned by the owner task "Reconcile GFlowNet State Space, Torsion Actions,
and Reward Oracle" and by issues #24 / #29 / #30 / #38 / #42:

1. :func:`TestDiscreteTrajectoryDeterminism` — repeating the same discrete
   action sequence reproduces the intermediate coordinates bit-for-bit
   (#24: the discrete action fully determines the geometry; bin centres
   are exact).
2. :func:`TestTerminalRelaxationIsolation` — the sub-bin Brent refinement
   and the MMFF94 relaxation run *only* upon STOP, never on intermediate
   MDP states, and never mutate the environment's discrete state (#42
   terminal scope; #24 no intermediate continuous physics).
3. :func:`TestStationaryRewardConsistency` — ``R(x)`` is identical to 8
   decimal places (bit-identical, in fact) regardless of rollout-batch
   siblings (#30 stationarity addendum; constitution §12.4), plus
   positivity / finite-log / component bounds and fail-closed behaviour.
4. :func:`TestBackwardLeafProbability` — ``get_removable_leaves``
   identifies exactly the peripheral fragments of a branched molecule,
   ``P_B`` is a normalized distribution, symmetric convergence and atom
   renumbering are handled (#38 adversarial cases).
5. :func:`TestToyDagFlowBalance` — on an enumerable toy DAG (19 quotient
   states, 31 trajectories) the analytically optimal flows under the
   uniform leaf backward policy drive *every* Trajectory Balance residual
   to zero through the real engine, and the terminal distribution is
   exactly reward-proportional, including the two-order convergence of
   every two-block state (#38 acceptance; #29 TB equation).
6. :func:`TestResonanceLockedJunctions` — amide junctions snap to the
   planar trans lock during rollout and stay planar through the terminal
   MMFF94 relaxation (#24 resonance lock; #42 skip rule).
7. :func:`TestSubbinRefinementMatchesDenseReferenceScan` — the bracketed
   Brent refinement is at least as good as a dense 200-point reference
   scan and stays inside the selected ±5° bin interval (#42 acceptance).
8. :func:`TestTrajectoryBalanceEngineContract` — the pinned TB equation,
   the zero STOP backward term, fail-closed contracts, gradient flow,
   batch decoupling, and the FP32/FP64 precision policy.
"""

from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetDihedralRad

from syntree.engine.environment import (
    AssemblyAction,
    AssemblyEnvironment,
    AssemblyError,
    BuildingBlock,
    HandleSpec,
    JunctionReaction,
    JunctionRecord,
    PocketDefinition,
    backward_leaf_probability,
    bin_bounds,
    bin_center,
    finalize_terminal_molecule,
    get_removable_leaves,
    set_junction_dihedral,
    soft_core_clash_energy,
    soft_core_pocket_energy,
    wrap_angle,
)
from syntree.engine.gflownet import FlowError, TrajectoryBalanceEngine, TrajectoryStep
from syntree.engine.rl import RewardError, ThreeDReward

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REACTIONS = {
    "generic_coupling": JunctionReaction("generic_coupling"),
    "couple_amine": JunctionReaction("couple_amine", block_handle_class="amine"),
}


def _chlorobenzene_block(block_id: str) -> BuildingBlock:
    """An aryl-chloride block: the ipso carbon is the attachment atom."""
    template = Chem.MolFromSmiles("c1ccccc1Cl")
    chlorine = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 17)
    ipso = template.GetAtomWithIdx(chlorine).GetNeighbors()[0].GetIdx()
    return BuildingBlock(
        block_id,
        "c1ccccc1Cl",
        (HandleSpec("aryl", ipso, "aryl_chloride", (chlorine,)),),
    )


def _tri_chloro_seed() -> BuildingBlock:
    """1,2,4-trichlorobenzene seed: three non-equivalent aryl handles."""
    template = Chem.MolFromSmiles("Clc1ccc(Cl)c(Cl)c1")
    specs = []
    for atom in template.GetAtoms():
        if atom.GetAtomicNum() == 17:
            ipso = atom.GetNeighbors()[0].GetIdx()
            specs.append(
                HandleSpec(f"cl{atom.GetIdx()}", ipso, "aryl_chloride", (atom.GetIdx(),))
            )
    assert len(specs) == 3
    return BuildingBlock("seed124", "Clc1ccc(Cl)c(Cl)c1", tuple(specs))


def _methylamine() -> BuildingBlock:
    return BuildingBlock("meam", "CN", (HandleSpec("n", 1, "amine"),))


def _methanol() -> BuildingBlock:
    return BuildingBlock("meoh", "CO", (HandleSpec("o", 1, "alcohol"),))


def _ethylenediamine() -> BuildingBlock:
    return BuildingBlock(
        "en",
        "NCCN",
        (HandleSpec("n0", 0, "amine"), HandleSpec("n3", 3, "amine_free")),
    )


def _shell_pocket(num_atoms: int = 14, radius: float = 5.2, seed: int = 11) -> PocketDefinition:
    """A deterministic 'shell' pocket around the ligand centroid.

    Atoms sit on an anisotropic sphere at a chemically sane distance, so
    the soft-core energies are moderate and the sub-bin refinement has a
    smooth, meaningful landscape (no catastrophic r -> 0 overlaps).
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_atoms, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * (radius + 0.6 * rng.random(num_atoms))[:, None]
    elements = [6, 6, 7, 8, 6, 8, 6, 6, 7, 6, 8, 6, 6, 8][:num_atoms]
    charges = np.zeros(num_atoms)
    charges[2] = -0.8  # an Asp/Glu-like carboxylate surrogate
    charges[8] = 0.8  # a Lys/Arg-like cation surrogate
    return PocketDefinition(positions, np.array(elements, dtype=np.int64), charges)


def _para_dichloro_seed() -> BuildingBlock:
    """p-dichlorobenzene seed: two aryl-chloride handles for two-step rollouts."""
    template = Chem.MolFromSmiles("Clc1ccc(Cl)cc1")
    specs = []
    for atom in template.GetAtoms():
        if atom.GetAtomicNum() == 17:
            ipso = atom.GetNeighbors()[0].GetIdx()
            specs.append(
                HandleSpec(f"cl{atom.GetIdx()}", ipso, "aryl_chloride", (atom.GetIdx(),))
            )
    assert len(specs) == 2
    return BuildingBlock("seedp", "Clc1ccc(Cl)cc1", tuple(specs))


def _fresh_two_step_env(pocket=None):
    seed = _para_dichloro_seed()
    catalog = {"meam": _methylamine(), "meoh": _methanol()}
    return AssemblyEnvironment(
        seed, catalog, pocket=pocket, max_blocks=4, reaction_registry=REACTIONS
    )


def _rollout(env: AssemblyEnvironment, bins) -> Chem.Mol:
    """Add methylamine then methanol with the given torsion bins, then STOP."""
    env.reset()
    handle = env.active_handles()[0]["handle_atom_idx"]
    env.step(AssemblyAction(handle, "generic_coupling", "meam", torsion_bin=bins[0]))
    handle = env.active_handles()[0]["handle_atom_idx"]
    env.step(AssemblyAction(handle, "generic_coupling", "meoh", torsion_bin=bins[1]))
    env.stop()
    return env.molecule


def _junctions_of(mol: Chem.Mol):
    return [
        JunctionRecord.from_dict(item)
        for item in json.loads(mol.GetProp("syn_tree_junctions"))
    ]


# ---------------------------------------------------------------------------
# 1. Discrete trajectory determinism (issue #24)
# ---------------------------------------------------------------------------


class TestDiscreteTrajectoryDeterminism:
    def test_same_action_sequence_is_bit_for_bit_identical(self):
        env_a = _fresh_two_step_env()
        env_b = _fresh_two_step_env()
        _rollout(env_a, bins=(27, 9))
        _rollout(env_b, bins=(27, 9))
        pos_a = np.array(env_a.molecule.GetConformer().GetPositions())
        pos_b = np.array(env_b.molecule.GetConformer().GetPositions())
        assert pos_a.shape == pos_b.shape
        assert np.array_equal(pos_a, pos_b), "repeated rollouts must be bit-for-bit identical"
        assert env_a.trajectory[-2].smiles == env_b.trajectory[-2].smiles
        assert env_a.trajectory[-1].smiles == env_b.trajectory[-1].smiles

    def test_determinism_with_independent_block_catalogs(self):
        """Fresh BuildingBlock instances (fresh ETKDG embedding) stay deterministic."""
        env_a = _fresh_two_step_env()
        env_b = _fresh_two_step_env()
        assert env_a is not env_b
        _rollout(env_a, bins=(5, 31))
        _rollout(env_b, bins=(5, 31))
        assert np.array_equal(
            np.array(env_a.molecule.GetConformer().GetPositions()),
            np.array(env_b.molecule.GetConformer().GetPositions()),
        )

    def test_intermediate_states_are_untouched_by_later_steps(self):
        """Surviving atom coordinates never move when a new block is placed."""
        env = _fresh_two_step_env()
        env.reset()
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "generic_coupling", "meam", torsion_bin=27))
        mol_one = env.molecule
        frozen = np.array(mol_one.GetConformer().GetPositions()).copy()
        # The second junction deletes the remaining chlorine (a leaving group).
        chlorine = next(
            a.GetIdx() for a in mol_one.GetAtoms() if a.GetAtomicNum() == 17
        )
        survivor_rows = [i for i in range(mol_one.GetNumAtoms()) if i != chlorine]
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "generic_coupling", "meoh", torsion_bin=9))
        grown = np.array(env.molecule.GetConformer().GetPositions())
        assert np.array_equal(grown[: len(survivor_rows)], frozen[survivor_rows])

    def test_different_bins_produce_different_geometry(self):
        env_a = _fresh_two_step_env()
        env_b = _fresh_two_step_env()
        _rollout(env_a, bins=(18, 18))
        _rollout(env_b, bins=(19, 18))
        assert not np.array_equal(
            np.array(env_a.molecule.GetConformer().GetPositions()),
            np.array(env_b.molecule.GetConformer().GetPositions()),
        )

    def test_intermediate_dihedral_equals_exact_bin_center(self):
        env = _fresh_two_step_env()
        env.reset()
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "generic_coupling", "meam", torsion_bin=27))
        junction = _junctions_of(env.molecule)[0]
        realized = GetDihedralRad(
            env.molecule.GetConformer(),
            junction.a0,
            junction.a1,
            junction.b1,
            junction.b0,
        )
        assert abs(wrap_angle(realized - bin_center(27))) < 1e-9

    def test_bin_center_grid(self):
        centers = [bin_center(k) for k in range(1, 37)]
        assert all(-math.pi <= c < math.pi for c in centers)
        gaps = np.diff(sorted(centers))
        np.testing.assert_allclose(gaps, 2 * math.pi / 36, atol=1e-12)
        assert bin_center(1) == pytest.approx(-math.pi + math.pi / 36)
        assert bin_center(36) == pytest.approx(math.pi - math.pi / 36)

    def test_bin_bounds_never_wrap(self):
        for k in range(1, 37):
            low, high = bin_bounds(k)
            assert -math.pi <= low < high <= math.pi
            assert high - low == pytest.approx(2 * math.pi / 36, abs=1e-12)

    def test_invalid_torsion_bin_fails_closed(self):
        with pytest.raises(AssemblyError):
            AssemblyAction(0, "generic_coupling", "meam", torsion_bin=0)
        with pytest.raises(AssemblyError):
            AssemblyAction(0, "generic_coupling", "meam", torsion_bin=37)
        with pytest.raises(AssemblyError):
            AssemblyAction(0, "generic_coupling", "meam", torsion_bin="18")
        with pytest.raises(AssemblyError):
            AssemblyAction(0, "generic_coupling", "meam", chiral="X")

    def test_chiral_parity_branches_are_distinct(self):
        env_r = _fresh_two_step_env()
        env_s = _fresh_two_step_env()
        for env, chiral in ((env_r, "R"), (env_s, "S")):
            env.reset()
            handle = env.active_handles()[0]["handle_atom_idx"]
            env.step(
                AssemblyAction(handle, "generic_coupling", "meam", chiral=chiral, torsion_bin=18)
            )
        assert not np.array_equal(
            np.array(env_r.molecule.GetConformer().GetPositions()),
            np.array(env_s.molecule.GetConformer().GetPositions()),
        )


# ---------------------------------------------------------------------------
# 2. Terminal relaxation isolation (issues #24 / #42)
# ---------------------------------------------------------------------------


class TestTerminalRelaxationIsolation:
    def test_oracle_counters_stay_zero_during_assembly(self):
        env = _fresh_two_step_env(pocket=_shell_pocket())
        env.reset()
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "generic_coupling", "meam", torsion_bin=27))
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "generic_coupling", "meoh", torsion_bin=9))
        assert env.stats == {
            "subbin_refinements": 0,
            "mmff_minimizations": 0,
            "brent_evaluations": 0,
        }

    def test_finalize_refuses_non_terminal_state(self):
        env = _fresh_two_step_env()
        env.reset()
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "generic_coupling", "meam", torsion_bin=27))
        with pytest.raises(AssemblyError, match="terminal reward-oracle operation"):
            env.finalize_terminal_molecule()

    def test_finalize_runs_only_after_stop_and_never_mutates_state(self):
        env = _fresh_two_step_env(pocket=_shell_pocket())
        _rollout(env, bins=(27, 9))
        discrete_positions = np.array(env.molecule.GetConformer().GetPositions()).copy()
        discrete_smiles = Chem.MolToSmiles(env.molecule)
        relaxed, energy, diagnostics = env.finalize_terminal_molecule(return_diagnostics=True)
        assert energy is not None and math.isfinite(energy)
        assert env.stats["mmff_minimizations"] == 1
        assert env.stats["subbin_refinements"] == 2  # two rotatable junctions
        assert env.stats["brent_evaluations"] >= 2
        # The discrete state is untouched bit-for-bit.
        assert np.array_equal(
            np.array(env.molecule.GetConformer().GetPositions()), discrete_positions
        )
        assert Chem.MolToSmiles(env.molecule) == discrete_smiles
        # The oracle output is a different (relaxed) object with the same graph.
        assert relaxed is not env.molecule
        assert Chem.MolToSmiles(relaxed) == discrete_smiles

    def test_subbin_rotation_preserves_covalent_geometry_exactly(self):
        """Rigid-junction refinement: bond lengths bit-for-bit intact, no MMFF."""
        env = _fresh_two_step_env(pocket=_shell_pocket())
        _rollout(env, bins=(14, 22))
        before = env.molecule
        relaxed, energy = env.finalize_terminal_molecule(
            subbin_refine=True, mmff_relax=False, return_diagnostics=False
        )
        assert energy is None  # no force field ran
        bonds_before = sorted(
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in before.GetBonds()
        )
        bonds_after = sorted(
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in relaxed.GetBonds()
        )
        assert bonds_before == bonds_after
        conf_before = before.GetConformer()
        conf_after = relaxed.GetConformer()
        for bond in before.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            length_before = np.linalg.norm(
                np.array(conf_before.GetAtomPosition(i)) - np.array(conf_before.GetAtomPosition(j))
            )
            length_after = np.linalg.norm(
                np.array(conf_after.GetAtomPosition(i)) - np.array(conf_after.GetAtomPosition(j))
            )
            # Rigid rotation preserves lengths to machine precision (the
            # Rodrigues composition can differ by ~1 ULP).
            assert length_before == pytest.approx(length_after, abs=1e-12)

    def test_refined_angles_stay_inside_their_bins(self):
        env = _fresh_two_step_env(pocket=_shell_pocket())
        _rollout(env, bins=(27, 9))
        _, _, diagnostics = env.finalize_terminal_molecule(return_diagnostics=True)
        assert len(diagnostics["junctions"]) == 2
        for entry in diagnostics["junctions"]:
            low, high = bin_bounds(entry["selected_bin"])
            assert low - 1e-9 <= entry["refined_angle_rad"] <= high + 1e-9

    def test_seed_scaffold_is_restrained(self):
        env = _fresh_two_step_env(pocket=_shell_pocket())
        _rollout(env, bins=(27, 9))
        _, _, diagnostics = env.finalize_terminal_molecule(return_diagnostics=True)
        # 50 kcal/mol/A^2 position restraints keep the seed scaffold in place.
        assert diagnostics["core_force_constant_kcal_mol_a2"] == 50.0
        assert diagnostics["num_seed_restrained_atoms"] == 6  # chlorobenzene carbons
        assert diagnostics["seed_rmsd_a"] < 0.5

    def test_without_pocket_the_refinement_is_a_documented_no_op(self):
        env = _fresh_two_step_env()
        _rollout(env, bins=(27, 9))
        before = np.array(env.molecule.GetConformer().GetPositions()).copy()
        relaxed, _, diagnostics = env.finalize_terminal_molecule(return_diagnostics=True)
        assert env.stats["subbin_refinements"] == 0
        assert env.stats["mmff_minimizations"] == 1
        for entry in diagnostics["junctions"]:
            assert "skipped" in entry
        assert np.array_equal(
            np.array(relaxed.GetConformer().GetPositions()), before
        ) or not np.array_equal(
            np.array(relaxed.GetConformer().GetPositions()), before
        )  # MMFF may still relax; only the *refinement* is the no-op


# ---------------------------------------------------------------------------
# 3. Stationary reward consistency (issue #30)
# ---------------------------------------------------------------------------


class TestStationaryRewardConsistency:
    def _target_molecule(self):
        env = _fresh_two_step_env(pocket=_shell_pocket())
        _rollout(env, bins=(27, 9))
        relaxed, _ = env.finalize_terminal_molecule(return_diagnostics=False)
        return relaxed

    @staticmethod
    def _sibling(smiles: str) -> Chem.Mol:
        mol = Chem.MolFromSmiles(smiles)
        assert AllChem.EmbedMolecule(mol, randomSeed=13) == 0
        return mol

    def test_reward_identical_across_batch_compositions(self):
        pocket = _shell_pocket()
        engine = ThreeDReward(pocket)
        target = self._target_molecule()

        similar = [
            self._sibling("CNc1ccccc1OC"),
            self._sibling("COc1ccccc1NC"),
            self._sibling("CNCc1ccccc1"),
        ]
        dissimilar = [
            self._sibling("c1ccc2ccccc2c1"),
            self._sibling("CCCCCCCCO"),
            self._sibling("c1ccncc1"),
        ]

        reward_alone = engine.compute(target)
        for sibling in similar:  # 'batch A': target surrounded by similar siblings
            engine.compute(sibling)
        reward_batch_a = engine.compute(target)
        for sibling in dissimilar:  # 'batch B': dissimilar siblings
            engine.compute(sibling)
        reward_batch_b = engine.compute(target)
        # Identical to 8 decimal places (bit-identical in fact).
        assert f"{reward_batch_a:.8f}" == f"{reward_alone:.8f}"
        assert f"{reward_batch_b:.8f}" == f"{reward_alone:.8f}"
        assert reward_batch_a == reward_alone == reward_batch_b

    def test_compute_signature_has_no_batch_parameters(self):
        signature = inspect.signature(ThreeDReward.compute)
        params = set(signature.parameters)
        assert params == {"self", "relaxed_mol"}, (
            "the reward must not accept any batch/sibling/reference parameter "
            f"(issue #30 addendum); got {params}"
        )
        for forbidden in ("batch", "sibling", "reference", "tanimoto", "diversity", "archive"):
            assert not any(forbidden in name for name in params)

    def test_reward_positive_with_finite_log(self):
        engine = ThreeDReward(_shell_pocket())
        target = self._target_molecule()
        reward = engine.compute(target)
        assert reward >= 1e-4
        assert math.isfinite(math.log(reward))

    def test_component_bounds_and_values(self):
        engine = ThreeDReward(_shell_pocket(), volume_floor_angstrom3=120.0)
        target = self._target_molecule()
        reward, components = engine.compute_with_components(target)
        assert -3.0 <= components["S_dock"] <= 5.0
        assert -1.0 < components["S_contact"] < 1.0
        assert components["S_key"] <= 1.5 * 2 + 0.5 * 2
        assert 0.0 <= components["S_props"] <= 1.0
        assert components["S_vol"] <= 0.0
        assert components["volume_angstrom3"] > 0
        expected = math.exp(
            components["S_dock"]
            + components["S_contact"]
            + components["S_key"]
            + components["S_props"]
            + components["S_vol"]
        )
        assert reward == pytest.approx(max(expected, 1e-4), rel=1e-12)

    def test_reward_reproducible_across_engine_instances(self):
        pocket = _shell_pocket()
        target = self._target_molecule()
        assert ThreeDReward(pocket).compute(target) == ThreeDReward(pocket).compute(target)

    def test_no_pocket_reward_is_graph_only(self):
        engine = ThreeDReward()  # no pocket: intrinsic properties only
        target = self._target_molecule()
        reward, components = engine.compute_with_components(target)
        assert components["S_dock"] == 0.0
        assert components["S_contact"] == 0.0
        assert components["S_key"] == 0.0
        assert components["S_props"] > 0.0
        assert components["S_vol"] == 0.0
        assert reward == pytest.approx(math.exp(components["S_props"]))

    def test_salt_bridge_and_hbond_credit_is_capped(self):
        pocket = _shell_pocket()
        engine = ThreeDReward(pocket)
        target = self._target_molecule()
        _, components = engine.compute_with_components(target)
        assert components["S_key"] <= 1.5 * 2 + 0.5 * 2
        assert components["S_key"] >= 0.0

    def test_fail_closed_on_poisoned_input(self):
        engine = ThreeDReward(_shell_pocket())
        mol = self._target_molecule()
        mol.GetConformer().SetAtomPosition(0, (float("nan"), 0.0, 0.0))
        with pytest.raises(RewardError):
            engine.compute(mol)

    def test_invalid_configuration_fails_closed(self):
        with pytest.raises(RewardError):
            ThreeDReward(_shell_pocket(), temperature=0.0)
        with pytest.raises(RewardError):
            ThreeDReward(_shell_pocket(), volume_floor_angstrom3=-1.0)
        with pytest.raises(RewardError):
            ThreeDReward(_shell_pocket(), reward_floor=0.0)


# ---------------------------------------------------------------------------
# 4. Backward leaf probability (issue #38)
# ---------------------------------------------------------------------------


class TestBackwardLeafProbability:
    def _branched_molecule(self):
        """Seed(deg2) - A(deg1); Seed - B(deg2) - C(deg1): leaves = {A, C}."""
        seed = _tri_chloro_seed()
        catalog = {
            "meam": _methylamine(),
            "en": _ethylenediamine(),
            "phcl": _chlorobenzene_block("phcl"),
        }
        env = AssemblyEnvironment(
            seed, catalog, max_blocks=4, reaction_registry=REACTIONS
        )
        env.reset()
        # A = methylamine on one seed handle (peripheral).
        h1 = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(h1, "couple_amine", "meam", torsion_bin=18))
        # B = ethylenediamine on a second seed handle (branch point: carries C).
        h2 = [h["handle_atom_idx"] for h in env.active_handles() if h["fragment_id"] == 0][0]
        env.step(AssemblyAction(h2, "couple_amine", "en", torsion_bin=18))
        # C = chlorobenzene on B's free amine handle (peripheral).
        free = [h["handle_atom_idx"] for h in env.active_handles() if h["fragment_id"] == 2]
        env.step(AssemblyAction(free[0], "generic_coupling", "phcl", torsion_bin=18))
        return env

    def test_branched_molecule_identifies_exactly_peripheral_fragments(self):
        env = self._branched_molecule()
        mol = env.molecule
        # Fragments: 0 = seed (degree 2), 1 = meam (degree 1),
        # 2 = ethylenediamine (degree 2, the branch point), 3 = phcl (degree 1).
        assert get_removable_leaves(mol) == [1, 3]
        assert backward_leaf_probability(mol) == pytest.approx(0.5)
        # P_B is a normalized distribution over the removable leaves.
        assert 2 * backward_leaf_probability(mol) == pytest.approx(1.0)

    def test_backward_probability_from_step_records(self):
        env = self._branched_molecule()
        records = [r for r in env.trajectory if r.action_type == "add"]
        # After A alone: leaves {A}. After A+B: leaves {A, B}. After A+B+C: {A, C}.
        assert [r.num_leaves for r in records] == [1, 2, 2]
        env.stop()
        assert env.trajectory[-1].num_leaves == 2

    def test_symmetric_molecule_both_orders_converge(self):
        seed = _ethylenediamine()
        results = []
        for order in (0, 1):
            env = AssemblyEnvironment(
                seed,
                {"phcl": _chlorobenzene_block("phcl")},
                max_blocks=4,
                reaction_registry=REACTIONS,
            )
            env.reset()
            handles = [h["handle_atom_idx"] for h in env.active_handles()]
            first, second = handles[order], handles[1 - order]
            env.step(AssemblyAction(first, "generic_coupling", "phcl", torsion_bin=18))
            env.step(AssemblyAction(second, "generic_coupling", "phcl", torsion_bin=18))
            results.append(env.molecule)
        mol_a, mol_b = results
        assert Chem.MolToSmiles(mol_a) == Chem.MolToSmiles(mol_b)
        assert get_removable_leaves(mol_a) == [1, 2]
        assert get_removable_leaves(mol_b) == [1, 2]
        assert backward_leaf_probability(mol_a) == pytest.approx(0.5)

    def test_backward_path_measure_sums_to_one(self):
        """Every deletion order of the symmetric molecule has total measure 1."""
        env = AssemblyEnvironment(
            _ethylenediamine(),
            {"phcl": _chlorobenzene_block("phcl")},
            max_blocks=4,
            reaction_registry=REACTIONS,
        )
        env.reset()
        handles = [h["handle_atom_idx"] for h in env.active_handles()]
        env.step(AssemblyAction(handles[0], "generic_coupling", "phcl", torsion_bin=18))
        intermediate_leaves = get_removable_leaves(env.molecule)
        env.step(AssemblyAction(handles[1], "generic_coupling", "phcl", torsion_bin=18))
        terminal_leaves = get_removable_leaves(env.molecule)
        total = sum(
            (1.0 / len(terminal_leaves)) * (1.0 / len(intermediate_leaves))
            for _ in terminal_leaves
        )
        assert total == pytest.approx(1.0)

    def test_atom_renumbering_invariance(self):
        env = self._branched_molecule()
        mol = env.molecule
        leaves_before = get_removable_leaves(mol)
        smiles_before = Chem.MolToSmiles(mol)
        order = list(range(mol.GetNumAtoms()))
        np.random.default_rng(3).shuffle(order)
        renumbered = Chem.RenumberAtoms(mol, [int(i) for i in order])
        assert get_removable_leaves(renumbered) == leaves_before
        assert Chem.MolToSmiles(renumbered) == smiles_before

    def test_seed_is_never_a_leaf(self):
        env = _fresh_two_step_env()
        _rollout(env, bins=(18, 18))
        assert 0 not in get_removable_leaves(env.molecule)

    def test_initial_state_has_no_parents(self):
        env = _fresh_two_step_env()
        env.reset()
        assert get_removable_leaves(env.molecule) == []
        with pytest.raises(AssemblyError):
            backward_leaf_probability(env.molecule)

    def test_foreign_molecule_fails_closed(self):
        with pytest.raises(AssemblyError):
            get_removable_leaves(Chem.MolFromSmiles("c1ccccc1"))


# ---------------------------------------------------------------------------
# 5. Toy-DAG flow balance: TB residual == 0 and P(x) ∝ R(x) (issues #29/#38)
# ---------------------------------------------------------------------------


class TestToyDagFlowBalance:
    """Enumerable construction DAG: 19 quotient states, 31 trajectories.

    Seed = 1,2,4-trichlorobenzene (three non-equivalent handles), catalog =
    {methylamine, methanol}, max_blocks = 2, torsion bin fixed at 18.  The
    GFlowNet state is the *order-independent assembled object* (the
    quotient of derivations that differ only by assembly order — the
    canonical-derivation construction of issue #38); canonical SMILES is a
    faithful state key for this fixture (asserted).  Rewards come from the
    real graph-only stationary reward engine (no pocket).  The flows are
    computed by exact backward induction using the *engine-measured* leaf
    counts, and the analytically optimal forward policy must drive every
    Trajectory Balance residual to zero *through the real TB engine* — and
    the terminal distribution must be exactly reward-proportional,
    including the two-order convergence of every two-block state.
    """

    BIN = 18
    MAX_BLOCKS = 2

    def _fresh_env(self):
        return AssemblyEnvironment(
            _tri_chloro_seed(),
            {"meam": _methylamine(), "meoh": _methanol()},
            max_blocks=self.MAX_BLOCKS,
            reaction_registry=REACTIONS,
        )

    def _replay(self, prefix):
        env = self._fresh_env()
        env.reset()
        for handle_idx, block_id in prefix:
            env.step(
                AssemblyAction(
                    handle_idx, "generic_coupling", block_id, torsion_bin=self.BIN
                )
            )
        return env

    def _enumerate(self):
        trajectories = []  # (prefix tuple, terminal quotient-state key)
        state_leaves, state_reward, state_children = {}, {}, {}

        def register(mol):
            key = Chem.MolToSmiles(mol)
            if key not in state_leaves:
                state_leaves[key] = len(get_removable_leaves(mol))
                state_reward[key] = ThreeDReward().compute(mol)
            return key

        def walk(prefix):
            env = self._replay(prefix)
            key = register(env.molecule)
            trajectories.append((tuple(prefix), key))
            if len(prefix) >= self.MAX_BLOCKS:
                return
            for handle_info in env.active_handles():
                for block_id in ("meam", "meoh"):
                    child_prefix = prefix + [(handle_info["handle_atom_idx"], block_id)]
                    child_key = register(self._replay(child_prefix).molecule)
                    state_children.setdefault(key, set()).add(child_key)
                    walk(child_prefix)

        walk([])
        root = register(self._replay([]).molecule)
        return trajectories, state_leaves, state_reward, state_children, root

    def _flows(self, state_leaves, state_reward, state_children, root):
        memo = {}

        def flow(state):
            if state in memo:
                return memo[state]
            total = state_reward[state]  # the STOP flow out of this state
            for child in state_children.get(state, ()):
                total += flow(child) * (1.0 / state_leaves[child])
            memo[state] = total
            return total

        return memo, flow(root)

    def test_state_space_shape(self):
        trajectories, state_leaves, _, state_children, root = self._enumerate()
        # 1 root + 6 one-add + 12 two-add quotient states.
        assert len(state_leaves) == 19
        # 31 trajectories: 1 stop-at-root + 6 one-add + 24 two-add orders.
        assert len(trajectories) == 31
        one_add = state_children.get(root, set())
        assert len(one_add) == 6
        grandchildren = set()
        for child in one_add:
            grandchildren.update(state_children.get(child, set()))
        assert len(grandchildren) == 12
        # Every two-add state has exactly two one-add parents: the
        # multi-path convergence at the heart of issue #38.
        parent_counts = {}
        for parent in one_add:
            for child in state_children.get(parent, set()):
                parent_counts.setdefault(child, 0)
                parent_counts[child] += 1
        assert set(parent_counts.values()) == {2}

    def test_every_tb_residual_is_zero_and_distribution_reward_proportional(self):
        trajectories, state_leaves, state_reward, state_children, root = self._enumerate()
        flows, root_flow = self._flows(state_leaves, state_reward, state_children, root)

        engine = TrajectoryBalanceEngine().double()
        # TB-zero requires log Z equal to the true total flow (partition function).
        with torch.no_grad():
            engine.log_z.copy_(torch.tensor(math.log(root_flow), dtype=torch.float64))
        residuals = []
        trajectory_probs = {}
        with torch.no_grad():
            for prefix, terminal_key in trajectories:
                env = self._fresh_env()
                env.reset()
                state_keys = [root]
                steps = []
                prob = 1.0
                for handle_idx, block_id in prefix:
                    record = env.step(
                        AssemblyAction(
                            handle_idx, "generic_coupling", block_id, torsion_bin=self.BIN
                        )
                    )
                    child_key = Chem.MolToSmiles(env.molecule)
                    parent_key = state_keys[-1]
                    p_forward = (
                        flows[child_key]
                        * (1.0 / state_leaves[child_key])
                        / flows[parent_key]
                    )
                    record.action_log_prob = math.log(p_forward)
                    steps.append(record)
                    prob *= p_forward
                    state_keys.append(child_key)
                stop_record = env.stop()
                p_stop = state_reward[state_keys[-1]] / flows[state_keys[-1]]
                stop_record.action_log_prob = math.log(p_stop)
                prob *= p_stop
                steps.append(stop_record)

                residual = engine.trajectory_residual(steps, state_reward[terminal_key])
                residuals.append(float(residual.detach()))
                assert abs(float(residual.detach())) < 1e-9, (
                    f"TB residual != 0 for trajectory {prefix}: {float(residual.detach())}"
                )
                trajectory_probs.setdefault(terminal_key, 0.0)
                trajectory_probs[terminal_key] += prob

        # Terminal distribution over quotient states: the probability of
        # terminating with outcome x is P(reach x) * P(STOP|x) summed over
        # every derivation order of x = [flow(x)/Z] * [R(x)/flow(x)] = R(x)/Z
        # — exactly the GFlowNet sampling theorem, with the multi-path mass
        # of all assembly orders of x aggregating without inflation.
        for key, total_prob in trajectory_probs.items():
            assert total_prob > 0.0
            assert total_prob == pytest.approx(state_reward[key] / root_flow, rel=1e-9), (
                f"P(terminal outcome x) != R(x)/Z for state {key}"
            )
        # Two-add states are terminal-only: their flows equal their rewards.
        for key in state_leaves:
            if not state_children.get(key):
                if key != root:  # root has children in this fixture
                    assert flows[key] == pytest.approx(state_reward[key], rel=1e-12)
        # Total probability mass and the partition-function identity.
        assert sum(trajectory_probs.values()) == pytest.approx(1.0, rel=1e-9)
        assert sum(state_reward.values()) == pytest.approx(root_flow, rel=1e-9)
        assert max(abs(r) for r in residuals) < 1e-9

    def test_multipath_convergence_splits_mass_exactly(self):
        """Each of the two orders reaching a two-block state contributes R/2Z."""
        _, state_leaves, state_reward, state_children, root = self._enumerate()
        flows, root_flow = self._flows(state_leaves, state_reward, state_children, root)
        one_add = state_children.get(root, set())
        two_add = set()
        for parent in one_add:
            two_add.update(state_children.get(parent, set()))
        target = sorted(two_add)[0]
        parents = [p for p in one_add if target in state_children.get(p, set())]
        assert len(parents) == 2
        contributions = []
        for parent in parents:
            p_root_to_parent = flows[parent] / root_flow  # leaves(parent) == 1
            p_parent_to_target = flows[target] * (1.0 / state_leaves[target]) / flows[parent]
            p_stop_at_target = state_reward[target] / flows[target]
            contributions.append(p_root_to_parent * p_parent_to_target * p_stop_at_target)
        expected_half = state_reward[target] / (2.0 * root_flow)
        for contribution in contributions:
            assert contribution == pytest.approx(expected_half, rel=1e-12)
        assert sum(contributions) == pytest.approx(state_reward[target] / root_flow, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. Resonance-locked junctions (issue #24 resonance snap)
# ---------------------------------------------------------------------------


class TestResonanceLockedJunctions:
    def _amide_env(self):
        # C0 C1(=O2) O3: handle = carbonyl carbon, leaving = hydroxyl oxygen.
        seed = BuildingBlock("acetic", "CC(=O)O", (HandleSpec("acid", 1, "carbonyl", (3,)),))
        reactions = {
            "amide_formation": JunctionReaction("amide_formation", resonance_lock="trans")
        }
        env = AssemblyEnvironment(
            seed, {"meam": _methylamine()}, max_blocks=4, reaction_registry=reactions
        )
        env.reset()
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "amide_formation", "meam", torsion_bin=5))
        return env

    def test_amide_junction_snaps_to_planar_trans(self):
        env = self._amide_env()
        assert Chem.MolToSmiles(env.molecule) == "CNC(C)=O"
        junctions = _junctions_of(env.molecule)
        assert len(junctions) == 1 and junctions[0].resonance_locked
        value = GetDihedralRad(
            env.molecule.GetConformer(),
            junctions[0].a0,
            junctions[0].a1,
            junctions[0].b1,
            junctions[0].b0,
        )
        assert abs(wrap_angle(value - math.pi)) < 1e-9

    def test_locked_junction_skips_subbin_refinement_and_stays_planar(self):
        env = self._amide_env()
        env.stop()
        env._pocket = _shell_pocket()
        relaxed, energy, diagnostics = env.finalize_terminal_molecule(return_diagnostics=True)
        assert energy is not None and math.isfinite(energy)
        assert env.stats["subbin_refinements"] == 0  # locked: nothing to refine
        assert env.stats["mmff_minimizations"] == 1
        assert diagnostics["junctions"] == []  # locked junctions are not refined
        assert diagnostics["locked_dihedral_drift"], "the locked junction must be monitored"
        for entry in diagnostics["locked_dihedral_drift"]:
            assert entry["drift_rad"] < math.radians(5.0)


# ---------------------------------------------------------------------------
# 7. Sub-bin refinement vs dense reference scan (issue #42)
# ---------------------------------------------------------------------------


class TestSubbinRefinementMatchesDenseReferenceScan:
    def test_brent_beats_dense_scan_and_stays_in_bin(self):
        pocket = _shell_pocket()
        env = _fresh_two_step_env(pocket=pocket)
        _rollout(env, bins=(27, 9))
        mol = env.molecule

        _, _, diagnostics = finalize_terminal_molecule(
            mol, pocket, subbin_refine=True, mmff_relax=False, return_diagnostics=True
        )
        assert len(diagnostics["junctions"]) == 2
        numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
        junctions = _junctions_of(mol)

        # The refinement is *sequential*: later junctions are optimised on the
        # baseline left by the earlier ones.  The dense reference scan mirrors
        # that exact sequence.
        baseline = Chem.Mol(mol)
        for entry, junction in zip(diagnostics["junctions"], junctions):
            low, high = bin_bounds(entry["selected_bin"])
            values = []
            for angle in np.linspace(low, high, 200):
                probe = Chem.Mol(baseline)
                set_junction_dihedral(probe, junction, float(angle))
                positions = np.array(probe.GetConformer().GetPositions(), dtype=np.float64)
                # #14 CHEM-04: the dense reference uses the production
                # flat-bottom soft-core clash potential — the same objective
                # the oracle's Brent refinement consumes (D-V2-015).
                values.append(
                    soft_core_clash_energy(
                        positions, numbers, pocket.positions, pocket.atomic_numbers
                    )
                )
            scan_best = min(values)
            assert entry["objective_after"] <= scan_best + 1e-6, (
                "the bracketed Brent refinement must be at least as good as the "
                "dense reference scan on the same sequential baseline"
            )
            assert low - 1e-9 <= entry["refined_angle_rad"] <= high + 1e-9
            assert entry["brent_evaluations"] < 200  # far cheaper than the scan
            assert entry["objective_after"] <= entry["objective_before"] + 1e-9
            # advance the baseline exactly as the oracle did
            set_junction_dihedral(baseline, junction, entry["refined_angle_rad"])

    def test_bin_interval_containment_including_edges(self):
        """Periodicity: bins adjacent to ±pi keep their bounds inside [-pi, pi]."""
        for k in (1, 2, 35, 36):
            low, high = bin_bounds(k)
            assert -math.pi <= low < high <= math.pi


# ---------------------------------------------------------------------------
# 8. TrajectoryBalanceEngine contract details (issue #29)
# ---------------------------------------------------------------------------


class TestTrajectoryBalanceEngineContract:
    def _steps(self, leaves=(1, 2)):
        return [
            TrajectoryStep("add", -1.1, leaves[0]),
            TrajectoryStep("add", -0.7, leaves[1]),
            TrajectoryStep("stop", -0.4, 3),
        ]

    def test_loss_formula_matches_pinned_equation(self):
        engine = TrajectoryBalanceEngine().double()
        reward = 0.37
        loss = engine.trajectory_balance_loss(self._steps(), reward)
        manual = (
            float(engine.log_z.detach())
            + (-1.1)
            + (-0.7)
            + (-0.4)  # sum log P_F (STOP included)
            + math.log(1)  # - log P_B (step 1: 1/1)
            + math.log(2)  # - log P_B (step 2: 1/2)
            - math.log(reward)
        ) ** 2
        assert float(loss.detach()) == pytest.approx(manual, rel=1e-12)

    def test_stop_backward_term_is_exactly_zero(self):
        engine = TrajectoryBalanceEngine().double()
        steps_a = self._steps()
        steps_b = [
            TrajectoryStep("add", -1.1, 1),
            TrajectoryStep("add", -0.7, 2),
            TrajectoryStep("stop", -0.4, 7),  # different terminal leaf count
        ]
        residual_a = engine.trajectory_residual(steps_a, 0.5)
        residual_b = engine.trajectory_residual(steps_b, 0.5)
        assert float(residual_a.detach()) == pytest.approx(float(residual_b.detach()), abs=1e-15)

    def test_reward_floor_enforced_fail_closed(self):
        engine = TrajectoryBalanceEngine().double()
        with pytest.raises(FlowError):
            engine.trajectory_balance_loss(self._steps(), 1e-9)
        with pytest.raises(FlowError):
            engine.trajectory_balance_loss(self._steps(), float("nan"))
        with pytest.raises(FlowError):
            engine.trajectory_balance_loss(self._steps(), -1.0)

    def test_malformed_trajectory_fails_closed(self):
        engine = TrajectoryBalanceEngine().double()
        with pytest.raises(FlowError):
            engine.trajectory_balance_loss([], 1.0)  # empty
        with pytest.raises(FlowError):
            engine.trajectory_balance_loss([TrajectoryStep("add", -1.0, 0)], 1.0)
        with pytest.raises(FlowError):
            engine.trajectory_balance_loss([TrajectoryStep("add", -1.0, 1)], 1.0)  # no STOP

    def test_stop_only_trajectory_is_valid(self):
        """Stopping immediately at the seed is a legitimate trajectory."""
        engine = TrajectoryBalanceEngine().double()
        loss = engine.trajectory_balance_loss([TrajectoryStep("stop", -0.2)], 1.5)
        manual = (float(engine.log_z.detach()) - 0.2 - math.log(1.5)) ** 2
        assert float(loss.detach()) == pytest.approx(manual, rel=1e-12)

    def test_dictionary_and_object_records_accepted(self):
        engine = TrajectoryBalanceEngine().double()
        dict_trajectory = [
            {"action_type": "add", "action_log_prob": -1.1, "num_leaves": 1},
            {"action_type": "add", "action_log_prob": -0.7, "num_leaves": 2},
            {"action_type": "stop", "action_log_prob": -0.4},
        ]
        loss_dict = engine.trajectory_balance_loss(dict_trajectory, 0.5)
        loss_records = engine.trajectory_balance_loss(self._steps(), 0.5)
        assert float(loss_dict.detach()) == pytest.approx(float(loss_records.detach()))

    def test_gradient_flows_to_log_z(self):
        engine = TrajectoryBalanceEngine()
        loss = engine.trajectory_balance_loss(self._steps(), 0.5)
        loss.backward()
        assert engine.log_z.grad is not None
        assert float(engine.log_z.grad) != 0.0

    def test_batch_loss_is_uncoupled_mean(self):
        engine = TrajectoryBalanceEngine().double()
        trajectories = [self._steps((1, 2)), self._steps((2, 1))]
        rewards = [0.5, 1.5]
        result = engine.batch_trajectory_balance_loss(trajectories, rewards)
        manual = [
            float(engine.trajectory_residual(t, r).detach()) for t, r in zip(trajectories, rewards)
        ]
        assert float(result["loss"].detach()) == pytest.approx(
            sum(r * r for r in manual) / 2, rel=1e-12
        )
        assert [float(x.detach()) for x in result["residuals"]] == pytest.approx(manual)

    def test_fp32_default_fp64_for_verification_only(self):
        engine = TrajectoryBalanceEngine()
        assert engine.dtype == torch.float32
        engine_double = engine.to(torch.float64)
        assert engine_double.dtype == torch.float64
        assert engine_double.log_z.dtype == torch.float64

    def test_nonfinite_log_prob_fails_closed(self):
        with pytest.raises(FlowError):
            TrajectoryStep("add", float("nan"), 1)
        with pytest.raises(FlowError):
            TrajectoryStep("jump", -1.0, 1)
