"""Suite B / 5 — GFlowNet flow invariants (issue #33 TEST-02).

The issue-pinned battery: **verify the deterministic reverse DAG step
and the Trajectory Balance loss behaviour** (#29 FLOW-01 core, #38
FLOW-04 semantics; constitution #45 §12.3).

* The reverse step is the **uniform leaf deletion policy** — deterministic
  (sorted removable-leaf enumeration), exactly ``1/k`` per parent over
  the ``k`` removable leaves of the state, with seed fragments excluded
  by construction.
* The TB residual is the exact dimensionless identity
  ``log Z + sum log P_F - sum log P_B - log R(x)`` with the uniform-leaf
  backward term ``-log(num_leaves)`` per 'add' step and zero backward
  terms for the seed and STOP steps (unique parents, proven in
  ``docs/architecture/MDP_STATE_SPACE.md``).
* The loss is the squared residual: zero at exact flow, deterministic
  bit-for-bit, gradient-bearing in ``log Z``, and fail-closed on any
  malformed trajectory/reward.
* Multi-path consistency: symmetric reaction orderings of the same
  terminal molecule admit the *same* optimal ``log Z`` (the #38
  multi-parent fix the uniform-leaf policy delivers).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.engine.environment import (  # noqa: E402
    AssemblyError,
    FRAGMENT_PROP,
    SEED_PROP,
    backward_leaf_probability,
    get_removable_leaves,
)
from syntree.engine.gflownet import (  # noqa: E402
    FlowError,
    TrajectoryBalanceEngine,
    TrajectoryStep,
    as_trajectory_steps,
)

_REWARD_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# The deterministic reverse DAG step (uniform leaf deletion)
# ---------------------------------------------------------------------------


def _fragmented_mol(layout: str) -> Chem.Mol:
    """A minimal assembled molecule carrying V2 fragment properties.

    ``"linear"``: seed(F0) - F1 - F2 (only F2 is a removable leaf).
    ``"star"``:   seed(F0) with two identical branches F1, F2 (both are
                  removable leaves — the symmetric multi-parent case).
    """
    rw = Chem.RWMol()
    if layout == "linear":
        plan = [(0, 0, True), (1, 0, True), (2, 1, False), (3, 2, False)]
        bonds = [(0, 1), (1, 2), (2, 3)]
    elif layout == "star":
        plan = [(0, 0, True), (1, 0, True), (2, 1, False), (3, 2, False)]
        bonds = [(0, 1), (1, 2), (1, 3)]
    else:
        raise ValueError(layout)
    for atom_idx, fragment, is_seed in plan:
        atom = Chem.Atom(6)
        atom.SetProp(FRAGMENT_PROP, str(fragment))
        if is_seed:
            atom.SetProp(SEED_PROP, "1")
        rw.AddAtom(atom)
    for begin, end in bonds:
        rw.AddBond(begin, end, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


class TestReverseStep:
    def test_linear_molecule_has_the_single_true_leaf(self):
        mol = _fragmented_mol("linear")
        assert get_removable_leaves(mol) == [2]  # F2 only (F1 is mid-chain)
        assert backward_leaf_probability(mol) == 1.0

    def test_seed_fragment_is_never_removable(self):
        for layout in ("linear", "star"):
            mol = _fragmented_mol(layout)
            assert 0 not in get_removable_leaves(mol)

    def test_symmetric_molecule_spreads_uniform_backward_mass(self):
        mol = _fragmented_mol("star")
        leaves = get_removable_leaves(mol)
        assert leaves == [1, 2]
        assert backward_leaf_probability(mol) == pytest.approx(0.5)
        # the parents' backward probabilities sum to 1 over the state
        assert backward_leaf_probability(mol) * len(leaves) == pytest.approx(1.0)

    def test_reverse_enumeration_is_deterministic(self):
        mol = _fragmented_mol("star")
        assert get_removable_leaves(mol) == get_removable_leaves(mol)
        assert backward_leaf_probability(mol) == backward_leaf_probability(mol)

    def test_removing_a_leaf_yields_a_connected_predecessor(self):
        for layout, leaf in (("linear", 2), ("star", 1), ("star", 2)):
            mol = _fragmented_mol(layout)
            keep = [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if int(atom.GetProp(FRAGMENT_PROP)) != leaf
            ]
            rw = Chem.RWMol(mol)
            for idx in sorted(keep, reverse=True):
                pass  # placeholder to keep linters calm about ordering
            removal = Chem.RWMol(mol)
            for atom_idx in sorted(
                set(range(mol.GetNumAtoms())) - set(keep), reverse=True
            ):
                removal.RemoveAtom(atom_idx)
            predecessor = removal.GetMol()
            Chem.SanitizeMol(predecessor)
            assert predecessor.GetNumAtoms() == len(keep)
            # connected: a single fragment-spanning walk reaches every atom
            visited = {0}
            frontier = [0]
            while frontier:
                current = frontier.pop()
                for neighbor in predecessor.GetAtomWithIdx(current).GetNeighbors():
                    if neighbor.GetIdx() not in visited:
                        visited.add(neighbor.GetIdx())
                        frontier.append(neighbor.GetIdx())
            assert visited == set(range(predecessor.GetNumAtoms()))
            # every remaining atom still carries its fragment provenance
            for atom in predecessor.GetAtoms():
                assert atom.HasProp(FRAGMENT_PROP)

    def test_unassembled_molecule_fails_closed(self):
        bare = Chem.MolFromSmiles("CCC")
        with pytest.raises(AssemblyError):
            get_removable_leaves(bare)
        with pytest.raises(AssemblyError):
            backward_leaf_probability(bare)


# ---------------------------------------------------------------------------
# The Trajectory Balance loss behaviour
# ---------------------------------------------------------------------------


def _trajectory(order: str = "ab") -> list:
    """A seed + two adds + stop trajectory with fixed log-probabilities.

    ``order`` permutes the two assembly steps (the symmetric-molecule
    multi-path fixture): both orderings end in the same terminal x, each
    step keeps its own forward log-probability, and the recorded
    ``num_leaves`` follows the *product state* of each action (the first
    add leaves one removable leaf, the second leaves two — in either
    order).
    """
    log_probs = {"a": -0.8, "b": -1.4}
    first, second = order[0], order[1]
    return [
        TrajectoryStep(action_type="seed", action_log_prob=-2.0),
        TrajectoryStep(
            action_type="add", action_log_prob=log_probs[first], num_leaves=1
        ),
        TrajectoryStep(
            action_type="add", action_log_prob=log_probs[second], num_leaves=2
        ),
        TrajectoryStep(action_type="stop", action_log_prob=-0.3),
    ]


def _optimal_log_z(trajectory, reward: float) -> float:
    steps = as_trajectory_steps(trajectory)
    sum_pf = sum(step.action_log_prob for step in steps)
    sum_pb = -sum(
        math.log(step.num_leaves) for step in steps if step.action_type == "add"
    )
    return math.log(reward) - sum_pf + sum_pb



def _det(tensor) -> float:
    """Detach-then-float: the production engine returns grad-bearing tensors."""
    return float(tensor.detach())


class TestTrajectoryBalanceLoss:
    def test_residual_is_the_exact_identity(self):
        engine = TrajectoryBalanceEngine(initial_log_z=1.7)
        trajectory = _trajectory()
        reward = 0.4
        residual = _det(engine.trajectory_residual(trajectory, reward))
        expected = (
            1.7  # log Z
            + (-2.0 - 0.8 - 1.4 - 0.3)  # sum log P_F
            - (-math.log(1) - math.log(2))  # sum log P_B (uniform leaf)
            - math.log(reward)
        )
        # the engine runs the production FP32 path: 1e-6 relative is exact
        # for values of this magnitude (float64 verification lives in the
        # #38 reconciliation battery)
        assert residual == pytest.approx(expected, rel=1e-6)

    def test_seed_and_stop_carry_no_backward_term(self):
        engine = TrajectoryBalanceEngine(initial_log_z=0.0)
        base = [
            TrajectoryStep(action_type="seed", action_log_prob=-1.0),
            TrajectoryStep(action_type="add", action_log_prob=-1.0, num_leaves=3),
            TrajectoryStep(action_type="stop", action_log_prob=-1.0),
        ]
        # rewriting the seed/stop log-probs changes only the forward sum
        tweaked = [
            TrajectoryStep(action_type="seed", action_log_prob=-2.0),
            TrajectoryStep(action_type="add", action_log_prob=-1.0, num_leaves=3),
            TrajectoryStep(action_type="stop", action_log_prob=-3.0),
        ]
        difference = _det(engine.trajectory_residual(tweaked, 1.0)) - _det(
            engine.trajectory_residual(base, 1.0)
        )
        assert difference == pytest.approx(-3.0, rel=1e-6)

    def test_leaf_count_sensitivity_is_the_uniform_ratio(self):
        engine = TrajectoryBalanceEngine(initial_log_z=0.0)
        with_three = [
            TrajectoryStep(action_type="seed", action_log_prob=-1.0),
            TrajectoryStep(action_type="add", action_log_prob=-1.0, num_leaves=3),
            TrajectoryStep(action_type="stop", action_log_prob=-1.0),
        ]
        with_four = [
            TrajectoryStep(action_type="seed", action_log_prob=-1.0),
            TrajectoryStep(action_type="add", action_log_prob=-1.0, num_leaves=4),
            TrajectoryStep(action_type="stop", action_log_prob=-1.0),
        ]
        difference = _det(engine.trajectory_residual(with_four, 1.0)) - _det(
            engine.trajectory_residual(with_three, 1.0)
        )
        # four leaves dilute the backward mass: the residual rises by
        # log(4/3) — exactly the uniform-ratio bookkeeping
        assert difference == pytest.approx(math.log(4.0 / 3.0), rel=1e-6)

    def test_loss_is_squared_residual_and_zero_at_exact_flow(self):
        trajectory = _trajectory()
        reward = 0.4
        engine = TrajectoryBalanceEngine(
            initial_log_z=_optimal_log_z(trajectory, reward)
        )
        loss = engine.trajectory_balance_loss(trajectory, reward)
        assert float(loss) == pytest.approx(0.0, abs=1e-7)
        engine_off = TrajectoryBalanceEngine(initial_log_z=0.0)
        residual = float(engine_off.trajectory_residual(trajectory, reward))
        torch.testing.assert_close(
            engine_off.trajectory_balance_loss(trajectory, reward),
            torch.tensor(residual ** 2),
        )

    def test_gradients_flow_into_log_z_exactly(self):
        trajectory = _trajectory()
        engine = TrajectoryBalanceEngine(initial_log_z=0.0)
        loss = engine.trajectory_balance_loss(trajectory, 0.4)
        loss.backward()
        residual = _det(engine.trajectory_residual(_trajectory(), 0.4))
        assert engine.log_z.grad is not None
        assert float(engine.log_z.grad) == pytest.approx(2.0 * residual, rel=1e-6)

    def test_execution_is_bit_deterministic(self):
        trajectory = _trajectory()
        first = TrajectoryBalanceEngine(initial_log_z=0.5)
        second = TrajectoryBalanceEngine(initial_log_z=0.5)
        loss_a = first.trajectory_balance_loss(trajectory, 0.4)
        loss_b = second.trajectory_balance_loss(trajectory, 0.4)
        assert torch.equal(loss_a, loss_b)
        assert torch.equal(
            first.trajectory_residual(trajectory, 0.4),
            second.trajectory_residual(trajectory, 0.4),
        )

    def test_fail_closed_contracts(self):
        engine = TrajectoryBalanceEngine()
        trajectory = _trajectory()
        with pytest.raises(FlowError):
            engine.trajectory_residual(trajectory, _REWARD_FLOOR / 10)  # below floor
        with pytest.raises(FlowError):
            engine.trajectory_residual([], 1.0)  # empty trajectory
        unended = trajectory[:-1]
        with pytest.raises(FlowError, match="STOP"):
            engine.trajectory_residual(unended, 1.0)
        malformed = [
            {"action_type": "add", "action_log_prob": -1.0, "num_leaves": 0}
        ] + [
            {"action_type": step.action_type, "action_log_prob": step.action_log_prob,
             "num_leaves": step.num_leaves}
            for step in trajectory
        ]
        with pytest.raises(FlowError):
            engine.trajectory_residual(malformed, 1.0)
        with pytest.raises(FlowError):
            TrajectoryStep(action_type="add", action_log_prob=float("nan"))

    def test_batch_loss_is_the_mean_and_batch_invariant(self):
        """Reward stationarity (#30): siblings never change a residual."""
        engine = TrajectoryBalanceEngine(initial_log_z=0.3)
        trajectory_a, trajectory_b = _trajectory("ab"), _trajectory("ba")
        alone = engine.trajectory_balance_loss(trajectory_a, 0.4)
        batch = engine.batch_trajectory_balance_loss(
            [trajectory_a, trajectory_b], [0.4, 0.9]
        )
        expected_mean = (
            _det(alone) + _det(engine.trajectory_balance_loss(trajectory_b, 0.9))
        ) / 2
        assert float(batch["loss"]) == pytest.approx(expected_mean, rel=1e-6)
        # the same trajectory in a different batch context: identical residual
        solo_residual = _det(engine.trajectory_residual(trajectory_a, 0.4))
        other_batch = engine.batch_trajectory_balance_loss(
            [trajectory_b, trajectory_a], [0.9, 0.4]
        )
        assert float(other_batch["residuals"][1]) == solo_residual

    def test_symmetric_orderings_admit_the_same_optimal_log_z(self):
        """The #38 multi-parent pin: P(x) proportional R(x) per terminal graph."""
        reward = 0.4
        forward_a, forward_b = _trajectory("ab"), _trajectory("ba")
        assert _optimal_log_z(forward_a, reward) == pytest.approx(
            _optimal_log_z(forward_b, reward), rel=1e-12
        )
        engine = TrajectoryBalanceEngine(
            initial_log_z=_optimal_log_z(forward_a, reward)
        )
        # at the shared optimum BOTH orderings' residuals vanish (FP32
        # production path: 1e-5 absolute is exact for these magnitudes)
        assert _det(engine.trajectory_residual(forward_a, reward)) == pytest.approx(
            0.0, abs=1e-5
        )
        assert _det(engine.trajectory_residual(forward_b, reward)) == pytest.approx(
            0.0, abs=1e-5
        )

    def test_step_normalisation_accepts_mappings_and_objects(self):
        mapping = [
            {"action_type": "seed", "action_log_prob": -1.0},
            {"action_type": "add", "action_log_prob": -1.0, "num_leaves": 2},
            {"action_type": "stop", "action_log_prob": -1.0},
        ]
        steps = as_trajectory_steps(mapping)
        assert [step.action_type for step in steps] == ["seed", "add", "stop"]
        assert steps[1].num_leaves == 2
        with pytest.raises(FlowError):
            as_trajectory_steps([{"action_type": "seed"}])  # missing log-prob
