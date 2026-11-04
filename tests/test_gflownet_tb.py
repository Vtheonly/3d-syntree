"""#29 FLOW-01 acceptance battery: Trajectory Balance on synthetic DAG rollouts.

The issue's two acceptance criteria, verbatim:

1. *``tests/test_gflownet_tb.py`` passes: Flow balance holds on synthetic
   DAG rollouts* — the enumerable 19-state / 31-trajectory construction
   DAG (the same fixture as ``test_gflownet_reconciliation.py``), now
   driven through a **trainable torch policy** whose tensor log-probabilities
   feed the Trajectory Balance engine: at the analytically optimal policy
   and the true partition function every TB residual is exactly zero, and
   a policy trained from random init converges to the reward-proportional
   terminal distribution ``P(x) = R(x)/Z`` (the GFlowNet sampling theorem).
2. *Backward pass cleanly optimizes policy parameters and partition
   function ``log Z_theta``* — gradients reach both optimizer param groups
   of :class:`~syntree.engine.gflownet.GFlowNetTrainer` (the policy group
   and the ``lr_Z = 1e-2`` log-partition group) and the update reduces the
   TB loss.

Plus the production-path battery: the seed-conditioned
:class:`~syntree.engine.gflownet.LogPartitionHead` contracts, the
differentiable tensor log-probability path of
:class:`~syntree.engine.gflownet.TrajectoryBalanceEngine`, the
:class:`~syntree.engine.rollout.PolicyRolloutDriver` end-to-end (the real
``SynTreePolicy`` driving the real ``AssemblyEnvironment``), and the
trainer's fail-closed/checkpoint contracts.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch import nn

from syntree.engine.environment import (
    AssemblyAction,
    AssemblyEnvironment,
    BuildingBlock,
    HandleSpec,
    PocketDefinition,
)
from syntree.engine.gflownet import (
    FlowError,
    GFlowNetProfile,
    GFlowNetTrainer,
    LogPartitionHead,
    TBTrajectory,
    TrajectoryBalanceEngine,
    TrajectoryStep,
)
from syntree.engine.rl import ThreeDReward
from syntree.models.policy import BLOCK_HANDLE_CLASS_VOCAB, SynTreePolicy
from tests.test_gflownet_reconciliation import (
    REACTIONS,
    _chlorobenzene_block,
    _methanol,
    _methylamine,
    _shell_pocket,
    _tri_chloro_seed,
)

# ---------------------------------------------------------------------------
# The synthetic DAG (19 quotient states / 31 trajectories) + its flows
# ---------------------------------------------------------------------------

BIN = 18
MAX_BLOCKS = 2
BLOCK_IDS = ("meam", "meoh")


def _fresh_env():
    return AssemblyEnvironment(
        _tri_chloro_seed(),
        {"meam": _methylamine(), "meoh": _methanol()},
        max_blocks=MAX_BLOCKS,
        reaction_registry=REACTIONS,
    )


def _replay(prefix):
    env = _fresh_env()
    env.reset()
    env.place_seed()
    for handle_idx, block_id in prefix:
        env.step(AssemblyAction(handle_idx, "generic_coupling", block_id, torsion_bin=BIN))
    return env


def _enumerate_dag():
    """(trajectories, state_leaves, state_reward, state_children, root)."""
    trajectories = []
    state_leaves, state_reward, state_children = {}, {}, {}

    def register(mol):
        key = Chem.MolToSmiles(mol)
        if key not in state_leaves:
            state_leaves[key] = len(
                __import__(
                    "syntree.engine.environment", fromlist=["get_removable_leaves"]
                ).get_removable_leaves(mol)
            )
            state_reward[key] = ThreeDReward().compute(mol)
        return key

    def walk(prefix):
        env = _replay(prefix)
        key = register(env.molecule)
        trajectories.append((tuple(prefix), key))
        if len(prefix) >= MAX_BLOCKS:
            return
        for handle_info in env.active_handles():
            for block_id in BLOCK_IDS:
                child = prefix + [(handle_info["handle_atom_idx"], block_id)]
                child_key = register(_replay(child).molecule)
                state_children.setdefault(key, set()).add(child_key)
                walk(child)

    walk([])
    root = register(_replay([]).molecule)
    return trajectories, state_leaves, state_reward, state_children, root


def _dag_flows(state_leaves, state_reward, state_children, root):
    memo = {}

    def flow(state):
        if state in memo:
            return memo[state]
        total = state_reward[state]
        for child in state_children.get(state, ()):
            total += flow(child) * (1.0 / state_leaves[child])
        memo[state] = total
        return total

    return memo, flow(root)


def _legal_actions(env):
    """The (handle, block) action menu of a toy state, deterministic order."""
    return [
        (int(h["handle_atom_idx"]), block_id)
        for h in sorted(env.active_handles(), key=lambda x: x["handle_atom_idx"])
        for block_id in BLOCK_IDS
    ]


def _handle_tag(mol, atom_idx):
    """An atom-order-invariant identity of a handle within its quotient state.

    The canonical SMILES of the molecule with the handle atom mapped makes
    canonicalization place the map marker deterministically — two assembly
    orders reaching the same quotient state produce the same tag for the
    corresponding handle (the #38 order-independence, as an action identity).
    """
    probe = Chem.Mol(mol)
    probe.GetAtomWithIdx(atom_idx).SetAtomMapNum(99)
    return Chem.MolToSmiles(probe)


def _menu_actions(env):
    """The order-invariant action menu of a toy state: (handle tag, block)."""
    return sorted(
        (_handle_tag(env.molecule, int(h["handle_atom_idx"])), block_id)
        for h in env.active_handles()
        for block_id in BLOCK_IDS
    )


class ToyDagPolicy(nn.Module):
    """A tabular policy over the DAG's quotient states (canonical SMILES keys).

    ``stop_logits``/``action_logits`` are the per-state parameters; the
    per-state law is the softmax over ``[action logits..., STOP logit]``.
    Tensor log-probabilities gathered from the law keep the autograd graph
    — the on-policy TB surface the trainer optimizes.
    """

    def __init__(self, state_ids, action_slices):
        super().__init__()
        self.state_ids = list(state_ids)
        self.state_index = {key: i for i, key in enumerate(self.state_ids)}
        self.action_slices = action_slices  # state index -> (lo, hi) into action_logits
        total = sum(hi - lo for lo, hi in action_slices)
        self.stop_logits = nn.Parameter(torch.zeros(len(state_ids)))
        self.action_logits = nn.Parameter(torch.zeros(total))

    def law(self, state_key: str) -> torch.Tensor:
        """``[k+1]`` log-probabilities over ``[actions..., STOP]``."""
        index = self.state_index[state_key]
        lo, hi = self.action_slices[index]
        logits = torch.cat(
            [self.action_logits[lo:hi], self.stop_logits[index:index + 1]]
        )
        return torch.log_softmax(logits, dim=-1)


def _build_toy_policy(dtype=torch.float64):
    """Enumerate the DAG and build the tabular policy over its states."""
    trajectories, state_leaves, state_reward, state_children, root = _enumerate_dag()
    # every reachable state key (root + descendants)
    keys = [root]
    seen = {root}
    frontier = [root]
    while frontier:
        state = frontier.pop()
        for child in state_children.get(state, ()):
            if child not in seen:
                seen.add(child)
                keys.append(child)
                frontier.append(child)
    # action slices per state (from a replayed env at that state)
    slices = []
    cursor = 0
    action_menus = {}
    # map state key -> an env producing it (root replay + one prefix each)
    producers = {root: []}
    for prefix, key in trajectories:
        producers.setdefault(key, list(prefix))
    for key in keys:
        env = _replay(producers[key])
        # a state at the block budget has no legal continuation (only STOP)
        if len(producers[key]) >= MAX_BLOCKS:
            actions = []
        else:
            actions = _menu_actions(env)
            if len(set(actions)) != len(actions):
                raise AssertionError(
                    "symmetric handles collapsed to one menu action (the "
                    "toy DAG fixture assumes distinguishable handles)"
                )
        action_menus[key] = actions
        slices.append((cursor, cursor + len(actions)))
        cursor += len(actions)
    policy = ToyDagPolicy(keys, slices)
    return (
        policy.to(dtype=dtype),
        trajectories,
        state_leaves,
        state_reward,
        state_children,
        root,
        action_menus,
        producers,
    )


def _tb_trajectory(prefix, terminal_key, reward, policy, action_menus, producers,
                   state_children, state_leaves, state_reward, root, dtype):
    """One enumerated trajectory as a differentiable TBTrajectory.

    The step log-probabilities are gathered from the toy policy's laws
    (tensor, graph-connected); the seed step's probability is exactly 0.0
    (the single-candidate pool makes the seed action deterministic).  The
    action rows are identified by the order-invariant (handle tag, block)
    menu entries, so every assembly order reaching a quotient state indexes
    the same law.
    """
    env = _fresh_env()
    env.reset()
    env.place_seed()
    steps = [TrajectoryStep("seed", torch.zeros((), dtype=dtype), 0)]
    state_key = root
    for handle_idx, block_id in prefix:
        tag = _handle_tag(env.molecule, handle_idx)
        row = action_menus[state_key].index((tag, block_id))
        log_prob = policy.law(state_key)[row]
        record = env.step(
            AssemblyAction(handle_idx, "generic_coupling", block_id, torsion_bin=BIN)
        )
        steps.append(TrajectoryStep("add", log_prob, record.num_leaves))
        state_key = Chem.MolToSmiles(env.molecule)
    law = policy.law(state_key)
    steps.append(TrajectoryStep("stop", law[-1], 0))
    return TBTrajectory(
        steps=steps,
        reward=reward,
        pocket_latent=torch.zeros(256, dtype=dtype),
        seed_latent=torch.zeros(64, dtype=dtype),
        terminal_smiles=terminal_key,
    )


# ---------------------------------------------------------------------------
# 1. The seed-conditioned partition function (#29)
# ---------------------------------------------------------------------------


class TestLogPartitionHead:
    def test_two_layer_architecture_and_lightweight_budget(self):
        head = LogPartitionHead()
        layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
        assert len(layers) == 2
        assert layers[0].in_features == 256 + 64
        assert layers[0].out_features == 256
        assert layers[1].out_features == 1
        count = sum(p.numel() for p in head.parameters())
        assert count == (256 + 64) * 256 + 256 + 256 + 1
        assert count < 0.01 * 10_880_719  # < 1% of the policy assembly

    def test_zero_initialized_output_starts_at_documented_zero(self):
        head = LogPartitionHead()
        out = head(torch.zeros(3, 256), torch.randn(3, 64))
        assert out.shape == (3,)
        assert torch.equal(out, torch.zeros(3))

    def test_gradient_flows_through_both_latents(self):
        head = LogPartitionHead()
        head.output.weight.data.normal_()
        pocket = torch.randn(2, 256, requires_grad=True)
        seed = torch.randn(2, 64, requires_grad=True)
        head(pocket, seed).sum().backward()
        assert pocket.grad is not None and bool((pocket.grad != 0).any())
        assert seed.grad is not None and bool((seed.grad != 0).any())

    def test_fail_closed_on_bad_inputs(self):
        head = LogPartitionHead()
        with pytest.raises(FlowError):
            head(torch.zeros(2, 128), torch.zeros(2, 64))  # wrong pocket dim
        with pytest.raises(FlowError):
            head(torch.zeros(2, 256), torch.zeros(3, 64))  # batch mismatch
        with pytest.raises(FlowError):
            head(torch.zeros(0, 256), torch.zeros(0, 64))  # empty batch
        bad = torch.zeros(2, 256)
        bad[0, 0] = float("nan")
        with pytest.raises(FlowError):
            head(bad, torch.zeros(2, 64))
        with pytest.raises(FlowError):
            head("no", torch.zeros(2, 64))

    def test_state_dict_round_trip(self):
        head = LogPartitionHead()
        head.trunk.weight.data.normal_()
        clone = LogPartitionHead()
        clone.load_state_dict(head.state_dict())
        x = torch.randn(2, 256)
        y = torch.randn(2, 64)
        assert torch.equal(head(x, y), clone(x, y))


# ---------------------------------------------------------------------------
# 2. The differentiable tensor log-probability path
# ---------------------------------------------------------------------------


class TestTensorLogProbPath:
    def _steps(self, log_prob_seed, log_prob_add, log_prob_stop):
        return [
            TrajectoryStep("seed", log_prob_seed, 0),
            TrajectoryStep("add", log_prob_add, 2),
            TrajectoryStep("stop", log_prob_stop, 2),
        ]

    def test_tensor_path_is_differentiable(self):
        engine = TrajectoryBalanceEngine().double()
        lp = torch.tensor([math.log(0.4)], dtype=torch.float64, requires_grad=True)
        steps = self._steps(lp[0], math.log(0.3), math.log(0.9))
        engine.trajectory_balance_loss(steps, 0.5).backward()
        assert lp.grad is not None and float(lp.grad) != 0.0

    def test_float_and_tensor_paths_agree(self):
        engine = TrajectoryBalanceEngine().double()
        float_residual = engine.trajectory_residual(
            self._steps(math.log(0.4), math.log(0.3), math.log(0.9)), 0.5
        )
        tensor_residual = engine.trajectory_residual(
            self._steps(
                torch.tensor(math.log(0.4), dtype=torch.float64),
                torch.tensor(math.log(0.3), dtype=torch.float64),
                torch.tensor(math.log(0.9), dtype=torch.float64),
            ),
            0.5,
        )
        assert float(float_residual) == pytest.approx(
            float(tensor_residual.detach()), abs=1e-12
        )

    def test_per_trajectory_log_z_batch(self):
        engine = TrajectoryBalanceEngine().double()
        trajectories = [
            self._steps(math.log(0.4), math.log(0.3), math.log(0.9)),
            self._steps(math.log(0.2), math.log(0.5), math.log(0.1)),
        ]
        log_z = torch.tensor([0.7, -0.3], dtype=torch.float64, requires_grad=True)
        out = engine.batch_trajectory_balance_loss(trajectories, [0.5, 0.25], log_z=log_z)
        assert out["residuals"].shape == (2,)
        out["loss"].backward()
        assert log_z.grad is not None and bool((log_z.grad != 0).any())

    def test_fail_closed_contracts(self):
        engine = TrajectoryBalanceEngine().double()
        with pytest.raises(FlowError):
            TrajectoryStep("add", torch.tensor(float("nan")), 2)
        with pytest.raises(FlowError):
            TrajectoryStep("add", torch.zeros(2), 2)  # not a single element
        with pytest.raises(FlowError):
            TrajectoryStep("add", torch.tensor(0.5, dtype=torch.float16), 2)
        with pytest.raises(FlowError):
            engine.batch_trajectory_balance_loss(
                [self._steps(0.1, 0.2, 0.3)], [0.5],
                log_z=torch.zeros(2, dtype=torch.float64),  # batch mismatch
            )
        with pytest.raises(FlowError):
            engine.batch_trajectory_balance_loss(
                [self._steps(0.1, 0.2, 0.3)], [0.5],
                log_z=torch.zeros(1, 2, dtype=torch.float64),  # rank-2
            )


# ---------------------------------------------------------------------------
# 3. Flow balance on the synthetic DAG (acceptance criterion 1)
# ---------------------------------------------------------------------------


class TestToyDagFlowBalance:
    @pytest.fixture(scope="class")
    def dag(self):
        return _build_toy_policy()

    def test_state_space_shape_unchanged(self, dag):
        (_policy, trajectories, state_leaves, _sr, _sc, _root) = dag[:6]
        assert len(state_leaves) == 19
        assert len(trajectories) == 31

    def test_optimal_policy_drives_every_residual_to_zero(self, dag):
        """The flow identity through the *tensor* path: exact zero."""
        (
            policy, trajectories, state_leaves, state_reward,
            state_children, root, action_menus, producers,
        ) = dag
        flows, root_flow = _dag_flows(state_leaves, state_reward, state_children, root)
        # set the parameters to the optimal law: p*(a|s) = F(child)/leaves(child)/F(s)
        with torch.no_grad():
            for key in policy.state_ids:
                actions = action_menus[key]
                if not actions:
                    # at-capacity state: STOP is the only action (p=1)
                    policy.stop_logits[policy.state_index[key]] = 0.0
                    continue
                logits = {}
                env = _replay(producers[key])
                for handle in env.active_handles():
                    tag = _handle_tag(env.molecule, int(handle["handle_atom_idx"]))
                    for block_id in BLOCK_IDS:
                        child_key = Chem.MolToSmiles(
                            _replay(
                                producers[key]
                                + [(int(handle["handle_atom_idx"]), block_id)]
                            ).molecule
                        )
                        p = (
                            flows[child_key]
                            * (1.0 / state_leaves[child_key])
                            / flows[key]
                        )
                        logits[(tag, block_id)] = math.log(p)
                p_stop = state_reward[key] / flows[key]
                logits["stop"] = math.log(p_stop)
                lo_slice, hi_slice = policy.action_slices[policy.state_index[key]]
                policy.action_logits[lo_slice:hi_slice] = torch.tensor(
                    [logits[a] for a in actions], dtype=policy.stop_logits.dtype
                )
                policy.stop_logits[policy.state_index[key]] = logits["stop"]
        engine = TrajectoryBalanceEngine().double()
        residuals = []
        for prefix, terminal_key in trajectories:
            traj = _tb_trajectory(
                prefix, terminal_key, state_reward[terminal_key], policy,
                action_menus, producers, state_children, state_leaves,
                state_reward, root, torch.float64,
            )
            residual = engine.trajectory_residual(
                traj.steps, traj.reward,
                log_z=torch.tensor(math.log(root_flow), dtype=torch.float64),
            )
            residuals.append(float(residual.detach()))
            assert abs(residuals[-1]) < 1e-9, (
                f"TB residual != 0 for trajectory {prefix}: {residuals[-1]}"
            )
        assert max(abs(r) for r in residuals) < 1e-9

    def test_training_optimizes_policy_and_log_z(self, dag, tmp_path):
        """Acceptance criterion 2: both param groups optimize; the loss drops
        orders of magnitude; the learned logZ approaches the true value."""
        (
            _fp64_policy, trajectories, state_leaves, state_reward,
            state_children, root, action_menus, producers,
        ) = dag
        flows, root_flow = _dag_flows(state_leaves, state_reward, state_children, root)
        # a fresh fp32 tabular policy (random init) + the real trainer
        torch.manual_seed(2026)
        policy = ToyDagPolicy(_fp64_policy.state_ids, _fp64_policy.action_slices)
        for parameter in policy.parameters():
            parameter.data.normal_(std=0.01)
        trainer = GFlowNetTrainer(
            policy, run_dir=str(tmp_path), profile=GFlowNetProfile(seed=2026)
        )
        # the lr_Z group is the #29-pinned 1e-2
        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(1e-2)
        assert trainer.optimizer.param_groups[1]["name"] == "log_z"
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)

        def enumerated_batch():
            return [
                _tb_trajectory(
                    prefix, key, state_reward[key], policy, action_menus,
                    producers, state_children, state_leaves, state_reward,
                    root, torch.float32,
                )
                for prefix, key in trajectories
            ]

        first = trainer.train_batch(enumerated_batch())["loss"]
        for _ in range(300):
            record = trainer.train_batch(enumerated_batch())
        final = record["loss"]
        assert final < first / 100.0, (
            f"TB loss did not drop orders of magnitude: {first} -> {final}"
        )
        assert abs(record["log_z_mean"] - math.log(root_flow)) < 0.35, (
            f"learned logZ {record['log_z_mean']} != true {math.log(root_flow)}"
        )
        # checkpoint round-trip: the log Z_theta state survives exactly
        trainer.save_checkpoint()
        fresh = GFlowNetTrainer(
            ToyDagPolicy(policy.state_ids, policy.action_slices),
            run_dir=str(tmp_path), profile=GFlowNetProfile(seed=2026), resume=True,
        )
        for (k1, v1), (k2, v2) in zip(
            trainer.log_partition.state_dict().items(),
            fresh.log_partition.state_dict().items(),
        ):
            assert k1 == k2 and torch.equal(v1, v2)

    def test_learned_terminal_distribution_is_reward_proportional(self, dag, tmp_path):
        """The GFlowNet sampling theorem through real sampled rollouts."""
        (
            _fp64_policy, trajectories, state_leaves, state_reward,
            state_children, root, action_menus, producers,
        ) = dag
        flows, root_flow = _dag_flows(state_leaves, state_reward, state_children, root)
        torch.manual_seed(2026)
        policy = ToyDagPolicy(_fp64_policy.state_ids, _fp64_policy.action_slices)
        for parameter in policy.parameters():
            parameter.data.normal_(std=0.01)
        trainer = GFlowNetTrainer(
            policy, run_dir=str(tmp_path / "dist"), profile=GFlowNetProfile(seed=7)
        )
        for _ in range(400):
            trainer.train_batch([
                _tb_trajectory(
                    prefix, key, state_reward[key], policy, action_menus,
                    producers, state_children, state_leaves, state_reward,
                    root, torch.float32,
                )
                for prefix, key in trajectories
            ])
        policy.eval()
        generator = torch.Generator().manual_seed(123)
        counts = {}
        n = 2000
        with torch.no_grad():
            for _ in range(n):
                env = _fresh_env()
                env.reset()
                env.place_seed()
                state_key = Chem.MolToSmiles(env.molecule)
                while True:
                    law = policy.law(state_key)
                    draw = int(torch.multinomial(law.exp(), 1, generator=generator))
                    actions = action_menus[state_key]
                    if draw == len(actions):  # STOP
                        env.stop()
                        break
                    tag, block_id = actions[draw]
                    handle_idx = next(
                        int(h["handle_atom_idx"])
                        for h in env.active_handles()
                        if _handle_tag(env.molecule, int(h["handle_atom_idx"])) == tag
                    )
                    env.step(AssemblyAction(
                        handle_idx, "generic_coupling", block_id, torsion_bin=BIN
                    ))
                    state_key = Chem.MolToSmiles(env.molecule)
                counts[state_key] = counts.get(state_key, 0) + 1
        # total variation distance vs R(x)/Z
        tv = 0.0
        for key, reward in state_reward.items():
            target = reward / root_flow
            empirical = counts.get(key, 0) / n
            tv += abs(empirical - target)
        tv /= 2.0
        assert tv < 0.15, f"empirical terminal distribution off by TV={tv:.4f}"


# ---------------------------------------------------------------------------
# 4. The production rollout driver (the real policy drives the real MDP)
# ---------------------------------------------------------------------------

AMIDE_ROWS = [
    ("SEED_PPD", "Nc1ccc(N)cc1", "", "amine"),
    ("BLK_PROPIONYL", "CCC(=O)Cl", "acid_chloride", ""),
    ("BLK_BENZOYL", "O=C(Cl)c1ccccc1", "acid_chloride", ""),
]


def _amide_fixture():
    """(policy, driver, pocket, block ids, classes, seed rows)."""
    from syntree.engine.rollout import (
        PolicyRolloutDriver,
        building_block_from_catalog_row,
        seed_block_from_catalog_row,
    )

    class_index = {name: i for i, name in enumerate(BLOCK_HANDLE_CLASS_VOCAB)}
    sentinel = len(BLOCK_HANDLE_CLASS_VOCAB)
    classes = []
    seed_rows = []
    available = set()
    for _bid, _smi, syn, seed in AMIDE_ROWS:
        primary = syn.split(",")[0].strip()
        if primary in class_index:
            classes.append(class_index[primary])
            available.add(primary)
        else:
            classes.append(sentinel)
        seed_rows.append(bool(seed))
    policy = SynTreePolicy(catalog_size=len(AMIDE_ROWS))
    driver = PolicyRolloutDriver(
        policy,
        catalog_block_ids=[row[0] for row in AMIDE_ROWS],
        catalog_block_handle_classes=torch.tensor(classes, dtype=torch.int64),
        catalog_seed_rows=torch.tensor(seed_rows, dtype=torch.bool),
        available_block_classes=sorted(available),
        generator=torch.Generator().manual_seed(2026),
    )
    pocket = _shell_pocket(num_atoms=14, radius=6.0, seed=5)
    return policy, driver, pocket


def _amide_env(pocket, max_blocks=2):
    from syntree.engine.rollout import (
        building_block_from_catalog_row,
        seed_block_from_catalog_row,
    )

    seed = seed_block_from_catalog_row("SEED_PPD", "Nc1ccc(N)cc1")
    catalog = {
        "SEED_PPD": seed,
        "BLK_PROPIONYL": building_block_from_catalog_row(
            "BLK_PROPIONYL", "CCC(=O)Cl", "acid_chloride"
        ),
        "BLK_BENZOYL": building_block_from_catalog_row(
            "BLK_BENZOYL", "O=C(Cl)c1ccccc1", "acid_chloride"
        ),
    }
    return AssemblyEnvironment(seed, catalog, pocket=pocket, max_blocks=max_blocks)


class TestPolicyRolloutDriver:
    def test_end_to_end_rollout_produces_valid_tb_trajectories(self, tmp_path):
        policy, driver, pocket = _amide_fixture()
        envs = [_amide_env(pocket), _amide_env(pocket), _amide_env(pocket)]
        trajectories = driver.rollout(envs)
        assert len(trajectories) == 3
        for traj in trajectories:
            assert traj.steps[0].action_type == "seed"
            assert traj.steps[-1].action_type == "stop"
            adds = [s for s in traj.steps if s.action_type == "add"]
            assert len(adds) <= 2
            for step in traj.steps:
                assert isinstance(step.action_log_prob, torch.Tensor)
                assert step.action_log_prob.requires_grad
                assert bool(torch.isfinite(step.action_log_prob).all())
                if step.action_type == "add":
                    assert step.num_leaves >= 1
            assert traj.reward >= 1e-4 and math.isfinite(traj.reward)
            assert traj.terminal_smiles
            assert traj.pocket_latent.shape == (256,)
            assert traj.seed_latent.shape == (64,)

    def test_rollout_is_deterministic_under_a_fixed_generator(self):
        _policy, reference, pocket = _amide_fixture()
        from syntree.engine.rollout import PolicyRolloutDriver

        # ONE policy instance shared by both runs (identical weights)
        shared = SynTreePolicy(catalog_size=len(AMIDE_ROWS))

        def run():
            driver = PolicyRolloutDriver(
                shared,
                catalog_block_ids=[r[0] for r in AMIDE_ROWS],
                catalog_block_handle_classes=reference.catalog_block_handle_classes.cpu(),
                catalog_seed_rows=reference.seed_mask.logical_not().cpu(),
                available_block_classes=["acid_chloride"],
                generator=torch.Generator().manual_seed(99),
            )
            return driver.rollout([_amide_env(pocket), _amide_env(pocket)])

        first, second = run(), run()
        for a, b in zip(first, second):
            assert a.terminal_smiles == b.terminal_smiles
            assert a.reward == b.reward
            assert [float(s.action_log_prob) for s in a.steps] == pytest.approx(
                [float(s.action_log_prob.detach()) for s in b.steps], rel=1e-6
            )

    def test_seed_and_synthon_legality_masks_hold(self):
        policy, driver, pocket = _amide_fixture()
        envs = [_amide_env(pocket) for _ in range(4)]
        for env in envs:
            # the seed row is the only seed-eligible row
            menu = env.seed_menu()
            assert menu.seed_block_ids == ("SEED_PPD",)
        trajectories = driver.rollout(envs)
        for traj, env in zip(trajectories, envs):
            seed_record = env.trajectory[0]
            assert seed_record.action.block_id == "SEED_PPD"
            for record in env.trajectory:
                if record.action_type == "add":
                    # the seed row (amine class) is never a legal retrieval
                    # target for the acid-chloride reactions
                    assert record.action.block_id != "SEED_PPD"
                    assert record.action.block_id in ("BLK_PROPIONYL", "BLK_BENZOYL")

    def test_max_blocks_budget_forces_stop(self):
        policy, driver, pocket = _amide_fixture()
        env = _amide_env(pocket, max_blocks=1)
        traj = driver.rollout([env])[0]
        adds = [s for s in traj.steps if s.action_type == "add"]
        assert len(adds) <= 1
        assert env.is_terminal

    def test_tb_loss_backward_reaches_policy_and_log_partition(self, tmp_path):
        policy, driver, pocket = _amide_fixture()
        trajectories = driver.rollout([_amide_env(pocket), _amide_env(pocket)])
        trainer = GFlowNetTrainer(
            policy, run_dir=str(tmp_path), profile=GFlowNetProfile(seed=0)
        )
        # direct gradient-flow evidence FIRST (the clean backward of
        # acceptance criterion 2, on the production path); retain the
        # rollout graph so the trainer's own update can backward after it
        pocket_lat = torch.stack([t.pocket_latent for t in trajectories])
        seed_lat = torch.stack([t.seed_latent for t in trajectories])
        log_z = trainer.log_partition(pocket_lat, seed_lat)
        loss = trainer.engine.batch_trajectory_balance_loss(
            [list(t.steps) for t in trajectories],
            [t.reward for t in trajectories],
            log_z=log_z,
        )["loss"]
        loss.backward(retain_graph=True)
        policy_grads = [
            p.grad for p in trainer.policy.parameters() if p.grad is not None
        ]
        assert policy_grads and any(float(g.abs().sum()) > 0 for g in policy_grads)
        head_grads = [
            p.grad for p in trainer.log_partition.parameters() if p.grad is not None
        ]
        assert head_grads and any(float(g.abs().sum()) > 0 for g in head_grads)
        trainer.optimizer.zero_grad(set_to_none=True)
        # the trainer's full update consumes the same rollout cleanly
        record = trainer.train_batch(trajectories)
        assert math.isfinite(record["loss"])
        assert record["batch_size"] == 2

    def test_pocket_free_environment_fails_closed(self):
        policy, driver, _pocket = _amide_fixture()
        env = _amide_env(None)
        with pytest.raises(Exception, match="pocket-conditioned"):
            driver.rollout([env])

    def test_catalog_size_mismatch_fails_closed(self):
        from syntree.engine.rollout import PolicyRolloutDriver, RolloutError

        with pytest.raises(RolloutError):
            PolicyRolloutDriver(
                SynTreePolicy(catalog_size=8),
                catalog_block_ids=[r[0] for r in AMIDE_ROWS],
                catalog_block_handle_classes=torch.zeros(3, dtype=torch.int64),
                catalog_seed_rows=torch.ones(3, dtype=torch.bool),
            )


# ---------------------------------------------------------------------------
# 5. Trainer contracts
# ---------------------------------------------------------------------------


class TestGFlowNetTrainerContracts:
    def _tiny_batch(self, n=2):
        steps = [
            TrajectoryStep("seed", torch.zeros(()), 0),
            TrajectoryStep("add", torch.tensor(-1.2), 1),
            TrajectoryStep("stop", torch.tensor(-0.7), 1),
        ]
        return [
            TBTrajectory(
                steps=list(steps),
                reward=0.5,
                pocket_latent=torch.zeros(256),
                seed_latent=torch.zeros(64),
            )
            for _ in range(n)
        ]

    def test_empty_batch_fails_closed(self, tmp_path):
        policy = SynTreePolicy(catalog_size=3)
        trainer = GFlowNetTrainer(policy, run_dir=str(tmp_path))
        with pytest.raises(FlowError):
            trainer.train_batch([])

    def test_sub_floor_reward_fails_closed(self, tmp_path):
        policy = SynTreePolicy(catalog_size=3)
        trainer = GFlowNetTrainer(policy, run_dir=str(tmp_path))
        batch = self._tiny_batch(1)
        batch[0].reward = 1e-6
        with pytest.raises(FlowError):
            trainer.train_batch(batch)

    def test_non_finite_latent_fails_closed(self, tmp_path):
        policy = SynTreePolicy(catalog_size=3)
        trainer = GFlowNetTrainer(policy, run_dir=str(tmp_path))
        batch = self._tiny_batch(1)
        batch[0] = TBTrajectory(
            steps=batch[0].steps,
            reward=0.5,
            pocket_latent=torch.full((256,), float("nan")),
            seed_latent=torch.zeros(64),
        )
        with pytest.raises(FlowError):
            trainer.train_batch(batch)

    def test_checkpoint_integrity_and_profile_gate(self, tmp_path):
        policy = SynTreePolicy(catalog_size=3)
        trainer = GFlowNetTrainer(policy, run_dir=str(tmp_path))
        trainer.train_batch(self._tiny_batch())
        trainer.save_checkpoint()
        assert os.path.exists(trainer.checkpoint_path + ".sha256")
        # a different profile refuses to resume (frozen-profile gate)
        with pytest.raises(FlowError):
            GFlowNetTrainer(
                policy, run_dir=str(tmp_path),
                profile=GFlowNetProfile(learning_rate=1e-3), resume=True,
            )
        # a corrupted sidecar refuses to resume
        with open(trainer.checkpoint_path + ".sha256", "w") as handle:
            handle.write("0" * 64 + "\n")
        with pytest.raises(FlowError):
            GFlowNetTrainer(policy, run_dir=str(tmp_path), resume=True)

    def test_resume_restores_exact_log_z_and_progress(self, tmp_path):
        policy = SynTreePolicy(catalog_size=3)
        trainer = GFlowNetTrainer(policy, run_dir=str(tmp_path))
        trainer.train_batch(self._tiny_batch())
        trainer.train_batch(self._tiny_batch())
        trainer.snapshot_rng()
        trainer.save_checkpoint()
        before = {
            k: v.clone() for k, v in trainer.log_partition.state_dict().items()
        }
        updates = trainer.progress.updates
        fresh = GFlowNetTrainer(
            SynTreePolicy(catalog_size=3), run_dir=str(tmp_path), resume=True
        )
        assert fresh.progress.updates == updates
        for key, value in fresh.log_partition.state_dict().items():
            assert torch.equal(value, before[key])

    def test_telemetry_keeps_ram_and_vram_separate(self, tmp_path):
        policy = SynTreePolicy(catalog_size=3)
        trainer = GFlowNetTrainer(policy, run_dir=str(tmp_path))
        record = trainer.telemetry()
        assert "host_memory" in record and "gpu_memory" in record
        assert record["host_memory"] is not None
        trainer.log_telemetry()
        assert os.path.exists(os.path.join(str(tmp_path), "telemetry.jsonl"))
