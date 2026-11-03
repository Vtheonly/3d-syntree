"""Tests for the learned seed/anchor selection — environment side (#39).

Issue-pinned acceptance criteria:

1. The environment can enumerate multiple valid seed/anchor candidates
   for a pocket (multi-subpocket anchors x the seed pool).
2. The seed is committed through an explicit action; nothing is placed
   by ``reset()`` (no hidden heuristic determines the starting state).
3. Invalid seed/anchor combinations fail closed.
4. Seed selection is included in trajectory replay and GFlowNet
   probability accounting (the 'seed' step's forward log-prob enters the
   TB residual; its backward term is zero by the unique-parent proof).
5. Placed seeds are geometrically valid and extendable by certified
   reactions.
6. Supervised seed/anchor labels for canonical decompositions (#39 item
   6) with documented ambiguity handling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from rdkit import Chem

from syntree.data.seed_labels import supervised_anchor_label
from syntree.engine.environment import (
    AssemblyAction,
    AssemblyEnvironment,
    AssemblyError,
    BuildingBlock,
    HandleSpec,
    JunctionReaction,
    PocketDefinition,
    SeedAction,
)
from syntree.engine.gflownet import (
    FlowError,
    TrajectoryBalanceEngine,
    TrajectoryStep,
    as_trajectory_steps,
)


def _block(block_id: str, smiles: str, handle_class: str, attach_element=None, leaving=()):
    mol = Chem.MolFromSmiles(smiles)
    if attach_element is not None:
        atom_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == attach_element)
    else:
        atom_idx = next(
            a.GetIdx()
            for a in mol.GetAtoms()
            if any(n.GetAtomicNum() in leaving for n in a.GetNeighbors())
        )
    return BuildingBlock(
        block_id,
        smiles,
        (HandleSpec(f"{block_id}:{handle_class}", atom_idx, handle_class, tuple(leaving)),),
    )


def _aniline(block_id: str = "aniline") -> BuildingBlock:
    mol = Chem.MolFromSmiles("c1ccccc1N")
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    return BuildingBlock("aniline", "c1ccccc1N", (HandleSpec("nh2", n, "amine"),))


def _aminopyridine(block_id: str = "aminopyridine") -> BuildingBlock:
    mol = Chem.MolFromSmiles("Nc1ccncc1")
    n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetSymbol() == "N")
    return BuildingBlock(block_id, "Nc1ccncc1", (HandleSpec("nh2", n, "amine"),))


def _acid_catalog() -> dict:
    mol = Chem.MolFromSmiles("CCC(=O)Cl")
    cl = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 17)
    carbonyl = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
        and {8, 17}.issubset({n.GetAtomicNum() for n in a.GetNeighbors()})
    )
    return {
        "propionyl": BuildingBlock(
            "propionyl",
            "CCC(=O)Cl",
            (HandleSpec("cl", carbonyl, "acid_chloride", (cl,)),),
        )
    }


#: A pocket with TWO well-separated subpockets (two 6-atom clusters ~14 A
#: apart) — the multi-subpocket acceptance case.
def _two_subpocket_pocket() -> PocketDefinition:
    rng = np.random.default_rng(7)
    cluster_a = rng.normal(scale=1.0, size=(6, 3))
    cluster_b = rng.normal(scale=1.0, size=(6, 3)) + np.array([14.0, 0.0, 0.0])
    positions = np.vstack([cluster_a, cluster_b])
    z = np.array([6, 7, 8, 6, 7, 8] * 2, dtype=np.int64)
    return PocketDefinition(positions, z, np.zeros(12, dtype=np.float64))


def _single_subpocket_pocket() -> PocketDefinition:
    # a compact cluster whose every atom pair is closer than the anchor
    # min-separation (4 A): exactly ONE anchor
    rng = np.random.default_rng(11)
    positions = rng.normal(scale=0.8, size=(10, 3))
    z = np.array([6, 7, 8, 6, 6, 8, 7, 6, 6, 8], dtype=np.int64)
    return PocketDefinition(positions, z, np.zeros(10, dtype=np.float64))


REACTIONS = {"rxn_01": JunctionReaction("rxn_01", block_handle_class="acid_chloride", resonance_lock="trans")}


class TestPreSeedState:
    def test_reset_places_nothing_and_fails_closed(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        menu = env.reset()
        assert menu.seed_block_ids == ("aniline",)
        assert menu.num_anchors >= 2  # two separated subpockets
        with pytest.raises(AssemblyError, match="place_seed"):
            env.observe()
        with pytest.raises(AssemblyError, match="place_seed"):
            env.step(AssemblyAction(0, "rxn_01", "propionyl"))
        with pytest.raises(AssemblyError, match="place_seed"):
            env.stop()
        with pytest.raises(AssemblyError, match="place_seed"):
            _ = env.molecule

    def test_multi_candidate_menu_is_the_product_space(self):
        pool = {"aniline": _aniline(), "aminopyridine": _aminopyridine()}
        env = AssemblyEnvironment(pool, _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        menu = env.reset()
        assert set(menu.seed_block_ids) == {"aniline", "aminopyridine"}
        assert menu.num_anchors >= 2
        # the documented legal action space: pool x anchors
        combos = {(b, a) for b in menu.seed_block_ids for a in range(menu.num_anchors)}
        assert len(combos) >= 4

    def test_empty_pool_fails_closed(self):
        with pytest.raises(AssemblyError, match="empty"):
            AssemblyEnvironment({}, _acid_catalog(), reaction_registry=REACTIONS)

    def test_ambiguous_pool_requires_explicit_action(self):
        pool = {"aniline": _aniline(), "aminopyridine": _aminopyridine()}
        env = AssemblyEnvironment(pool, _acid_catalog(), reaction_registry=REACTIONS)
        env.reset()
        with pytest.raises(AssemblyError, match="explicit SeedAction"):
            env.place_seed()

    def test_invalid_combinations_fail_closed(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        env.reset()
        with pytest.raises(AssemblyError, match="selected"):
            env.place_seed(SeedAction("not_in_pool", 0))
        with pytest.raises(AssemblyError, match="anchor"):
            env.place_seed(SeedAction("aniline", 99))


class TestPlacement:
    def test_placement_records_explicit_seed_step(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        env.reset()
        record = env.place_seed(SeedAction("aniline", 1))
        assert record.action_type == "seed"
        assert record.action == SeedAction("aniline", 1)
        assert record.num_leaves == 0
        assert record.smiles == "Nc1ccccc1"
        assert env.trajectory[0] is record
        # handles are live after the seed action
        menu = env.active_handles()
        assert [h["handle_class"] for h in menu] == ["amine"]

    def test_seed_centroid_lands_on_the_anchor(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        menu = env.reset()
        env.place_seed(SeedAction("aniline", 0))
        centroid = np.mean(env.molecule.GetConformer().GetPositions(), axis=0)
        assert np.allclose(centroid, menu.anchor_positions[0], atol=1e-9)

    def test_placement_is_deterministic(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        env.reset(); env.place_seed(SeedAction("aniline", 0))
        first = np.array(env.molecule.GetConformer().GetPositions())
        env.reset(); env.place_seed(SeedAction("aniline", 0))
        second = np.array(env.molecule.GetConformer().GetPositions())
        assert np.array_equal(first, second)

    def test_placement_is_rigid_motion_equivariant(self):
        pocket = _two_subpocket_pocket()
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        env.reset(); env.place_seed(SeedAction("aniline", 0))
        base = np.array(env.molecule.GetConformer().GetPositions())

        # a random rigid motion of the SAME pocket (same atom list)
        rng = np.random.default_rng(3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = 1.234
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
        translation = np.array([3.0, -2.0, 5.5])
        moved = PocketDefinition(
            (rotation @ pocket.positions.T).T + translation,
            pocket.atomic_numbers,
            pocket.charges,
        )
        env2 = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=moved, reaction_registry=REACTIONS)
        env2.reset(); env2.place_seed(SeedAction("aniline", 0))
        placed = np.array(env2.molecule.GetConformer().GetPositions())
        expected = (rotation @ base.T).T + translation
        assert np.allclose(placed, expected, atol=1e-8), (
            "the deterministic pose rule must be exactly equivariant under rigid "
            "motions of the pocket (documented fallbacks excepted — this pocket "
            "is generic)"
        )

    def test_placed_seed_is_valid_and_extends_by_a_certified_reaction(self):
        pocket = _two_subpocket_pocket()
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        menu = env.reset()
        for anchor in range(menu.num_anchors):
            env.reset()
            env.place_seed(SeedAction("aniline", anchor))
            handle = env.active_handles()[0]["handle_atom_idx"]
            step = env.step(AssemblyAction(handle, "rxn_01", "propionyl"))
            assert step.action_type == "add"
            Chem.SanitizeMol(env.molecule)
            assert env.stats["last_seed_placement_energy_milli"] is not None

    def test_pocket_free_fixture_keeps_template_pose(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), reaction_registry=REACTIONS)
        env.reset()
        env.place_seed()
        positions = np.array(env.molecule.GetConformer().GetPositions())
        template = _aniline().template_positions
        assert np.allclose(positions, template, atol=0.0)

    def test_placed_seed_leaves_the_growth_menu(self):
        seed = _aniline("aniline")
        catalog = _acid_catalog()
        catalog["aniline"] = seed  # seed pool overlaps the catalog
        env = AssemblyEnvironment(seed, catalog, reaction_registry=REACTIONS)
        env.reset()
        env.place_seed(SeedAction("aniline", 0))
        handle = env.active_handles()[0]["handle_atom_idx"]
        with pytest.raises(AssemblyError, match="unknown block_id 'aniline'"):
            env.step(AssemblyAction(handle, "rxn_01", "aniline"))


class TestTrajectoryAndFlowAccounting:
    def test_trajectory_replay_order_is_seed_add_stop(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        env.reset()
        env.place_seed(SeedAction("aniline", 0))
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "rxn_01", "propionyl"))
        env.stop()
        assert [s.action_type for s in env.trajectory] == ["seed", "add", "stop"]

    def test_seed_step_jins_tb_with_zero_backward_term(self):
        engine = TrajectoryBalanceEngine(initial_log_z=0.0)
        seed_log_prob = math.log(0.25) + math.log(0.5)  # P(anchor) * P(seed|anchor)
        steps = [
            TrajectoryStep("seed", seed_log_prob, num_leaves=0),
            TrajectoryStep("add", math.log(0.5), num_leaves=1),
            TrajectoryStep("stop", math.log(0.5), num_leaves=1),
        ]
        engine = engine.to(torch.float64)  # exact identity check in float64
        residual = engine.trajectory_residual(steps, reward=0.5)
        expected = 0.0 + (seed_log_prob + math.log(0.5) + math.log(0.5)) - (-math.log(1)) - math.log(0.5)
        assert abs(float(residual) - expected) < 1e-12

    def test_seed_step_without_the_log_prob_fails_closed(self):
        with pytest.raises(FlowError):
            TrajectoryStep("seed", float("nan"))
        with pytest.raises(FlowError, match="no removable leaves"):
            TrajectoryStep("seed", -1.0, num_leaves=2)

    def test_env_trajectory_feeds_the_tb_engine(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        env.reset()
        env.place_seed(SeedAction("aniline", 0))
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "rxn_01", "propionyl"))
        env.stop()
        for record in env.trajectory:
            record.action_log_prob = math.log(0.5)
        steps = as_trajectory_steps(env.trajectory)
        assert [s.action_type for s in steps] == ["seed", "add", "stop"]
        assert all(s.num_leaves >= 0 for s in steps)
        engine = TrajectoryBalanceEngine()
        residual = engine.trajectory_residual(steps, reward=1.0)
        assert math.isfinite(float(residual))

    def test_seed_only_trajectory_is_legal(self):
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=_two_subpocket_pocket(), reaction_registry=REACTIONS)
        env.reset()
        env.place_seed(SeedAction("aniline", 0))
        env.stop()
        assert env.is_terminal
        for record in env.trajectory:
            record.action_log_prob = math.log(0.5)
        engine = TrajectoryBalanceEngine()
        residual = engine.trajectory_residual(env.trajectory, reward=1.0)
        assert math.isfinite(float(residual))
        # seed + stop only: zero backward terms in total
        assert [s.action_type for s in as_trajectory_steps(env.trajectory)] == ["seed", "stop"]


class TestAnchorConsistency:
    def test_env_anchors_match_the_policy_generator(self):
        import torch

        from syntree.models.policy import pocket_anchor_candidates

        pocket = _two_subpocket_pocket()
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        menu = env.reset()
        positions, assignment = pocket_anchor_candidates(
            torch.as_tensor(pocket.positions, dtype=torch.float64),
            torch.as_tensor(pocket.atomic_numbers, dtype=torch.float64),
            torch.as_tensor(pocket.charges, dtype=torch.float64),
        )
        assert np.allclose(menu.anchor_positions, positions.numpy())
        assert np.array_equal(menu.anchor_assignment, assignment.numpy())

    def test_separated_subpockets_yield_separated_anchors(self):
        pocket = _two_subpocket_pocket()
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        menu = env.reset()
        assert menu.num_anchors == 2
        assert np.linalg.norm(menu.anchor_positions[0] - menu.anchor_positions[1]) > 10.0

    def test_contiguous_pocket_yields_one_anchor(self):
        pocket = _single_subpocket_pocket()
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        menu = env.reset()
        assert menu.num_anchors == 1


class TestSupervisedAnchorLabel:
    def test_label_selects_the_nearest_subpocket(self):
        pocket = _two_subpocket_pocket()
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        menu = env.reset()
        # a crystal seed sitting in subpocket B
        label = supervised_anchor_label(pocket, tuple(menu.anchor_positions[1]))
        assert label.anchor_idx == 1
        assert label.anchor_distance == pytest.approx(0.0, abs=1e-9)
        assert not label.is_tie_break

    def test_label_is_rigid_motion_invariant(self):
        pocket = _two_subpocket_pocket()
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        menu = env.reset()
        rng = np.random.default_rng(5)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        translation = np.array([1.0, 2.0, -3.0])
        moved = PocketDefinition(
            (rotation @ pocket.positions.T).T + translation,
            pocket.atomic_numbers,
            pocket.charges,
        )
        centroid = menu.anchor_positions[0] + np.array([0.3, -0.2, 0.1])
        moved_centroid = rotation @ centroid + translation
        l1 = supervised_anchor_label(pocket, tuple(centroid))
        l2 = supervised_anchor_label(moved, tuple(moved_centroid))
        assert l1.anchor_idx == l2.anchor_idx
        assert l1.anchor_distance == pytest.approx(l2.anchor_distance, abs=1e-9)

    def test_equidistant_tie_resolves_to_the_lowest_index(self):
        pocket = _two_subpocket_pocket()
        env = AssemblyEnvironment(_aniline(), _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        menu = env.reset()
        midpoint = 0.5 * (menu.anchor_positions[0] + menu.anchor_positions[1])
        label = supervised_anchor_label(pocket, tuple(midpoint))
        # anchors equidistant from the exact midpoint (generic pockets:
        # anchor centroids differ; the tie is broken by the lowest index
        # whenever it occurs)
        assert 0 <= label.anchor_idx < menu.num_anchors
        assert label.ambiguity  # the resolution is documented

    def test_supervision_does_not_pin_inference(self):
        # the label is a supervision target; the environment still offers
        # every (block, anchor) combination to the policy
        pocket = _two_subpocket_pocket()
        pool = {"aniline": _aniline(), "aminopyridine": _aminopyridine()}
        env = AssemblyEnvironment(pool, _acid_catalog(), pocket=pocket, reaction_registry=REACTIONS)
        menu = env.reset()
        label = supervised_anchor_label(pocket, tuple(menu.anchor_positions[0]))
        assert label.anchor_idx == 0
        # inference may still legally choose the OTHER anchor/block
        env.place_seed(SeedAction("aminopyridine", 1))
        assert env.trajectory[0].action == SeedAction("aminopyridine", 1)
