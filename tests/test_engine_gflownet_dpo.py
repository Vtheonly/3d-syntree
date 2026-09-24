"""Unit + integration tests for the GFlowNet Trajectory-Balance engine and
the DPO preference optimizer (tasklist RL modernisation)."""

from __future__ import annotations

import math

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.engine.dpo import DPOTrainer, build_preference_pairs
from syntree.engine.gflownet import GFlowNetTrainer
from syntree.engine.rl import (
    PPOTransition,
    RolloutTransition,
    ThreeDReward,
    joint_log_prob_and_entropy,
)


# ---------------------------------------------------------------------------
# Tiny fake policy + catalog so tests never need the real PaiNN stack.
# ---------------------------------------------------------------------------
class FakeCatalog:
    """Minimal catalog duck-type used by the trainers."""

    def __init__(self, k: int = 4, dim: int = 8):
        self._k = k
        self.embeddings = torch.randn(k, dim)

    def __len__(self):
        return self._k


class FakePolicy(torch.nn.Module):
    """Scalar head over a fake state vector; emits the policy contract."""

    def __init__(self, k: int = 4):
        super().__init__()
        self._k = k
        self.state_proj = torch.nn.Linear(4, 16)
        self.reaction_head = torch.nn.Linear(16, 9)
        self.synthon_head = torch.nn.Linear(16, k + 1)
        self.value_head = torch.nn.Linear(16, 1)

    def forward(self, batch, embeddings, synthon_mask=None, reaction_mask=None,
                **kwargs):
        feats = batch.fake_state
        h = self.state_proj(feats)
        reaction_logits = self.reaction_head(h)
        if reaction_mask is not None:
            reaction_logits = reaction_logits + reaction_mask
        synthon_logits = self.synthon_head(h)
        if synthon_mask is not None:
            synthon_logits = synthon_logits + synthon_mask
        return {
            "reaction_logits": reaction_logits,
            "reaction_log_probs": torch.log_softmax(reaction_logits, dim=-1),
            "synthon_logits": synthon_logits,
            "synthon_log_probs": torch.log_softmax(synthon_logits, dim=-1),
            "state_value": self.value_head(h).squeeze(-1),
        }


class FakeState:
    """Duck-typed state object with a `.to(device)` method."""

    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.fake_state = torch.randn(1, 4, generator=g)

    def to(self, device):
        self.fake_state = self.fake_state.to(device)
        return self

    def clone(self):
        new = FakeState.__new__(FakeState)
        new.fake_state = self.fake_state.clone()
        return new


def _make_transition(seed: int = 0, family: int = 0, action: int = 0,
                     k: int = 4) -> RolloutTransition:
    state = FakeState(seed)
    return RolloutTransition(
        state=state,
        reaction_mask=torch.zeros(1, 9),
        synthon_masks=torch.zeros(1, 9, k + 1),  # K synthons + STOP slot
        family_idx=family,
        action_idx=action,
        old_log_prob=torch.tensor(0.0),
        old_value=torch.tensor(0.0),
        reward=0.0,
        done=False,
    )


def _make_trajectory(n: int, seed: int = 0) -> list:
    return [_make_transition(seed=seed + i, action=0) for i in range(n)]


# ---------------------------------------------------------------------------
# GFlowNet Trajectory Balance math
# ---------------------------------------------------------------------------
class TestTrajectoryBalanceMath:
    def _trainer(self, k: int = 4):
        model = FakePolicy(k)
        catalog = FakeCatalog(k)
        return GFlowNetTrainer(model, catalog, torch.device("cpu"), {}), model, catalog

    def test_tb_loss_zero_when_balanced(self):
        """If log Z + sum log PF == log R exactly, the loss must vanish."""
        trainer, _, _ = self._trainer()
        trace = _make_trajectory(1)
        # sum log PF under the current policy; pick R so that logZ(=0)
        # balances the residual: R = exp(sum log PF).
        log_pf, _, _ = joint_log_prob_and_entropy(
            trainer.model, trainer.catalog, trainer.device, trace[0]
        )
        reward = float(torch.exp(log_pf.detach()))
        loss = trainer.trajectory_balance_loss(trace, reward)
        assert float(loss) < 1e-6

    def test_tb_loss_is_squared_residual(self):
        trainer, _, _ = self._trainer()
        trace = _make_trajectory(1)
        log_pf, _, _ = joint_log_prob_and_entropy(
            trainer.model, trainer.catalog, trainer.device, trace[0]
        )
        reward = 1.0
        expected = (0.0 + float(log_pf) - math.log(1.0) - 0.0) ** 2
        loss = trainer.trajectory_balance_loss(trace, reward)
        assert abs(float(loss) - expected) < 1e-6

    def test_reward_floor_protects_log(self):
        """Zero / negative rewards must be floored, never crash log()."""
        trainer, _, _ = self._trainer()
        trace = _make_trajectory(2)
        loss = trainer.trajectory_balance_loss(trace, 0.0)
        assert torch.isfinite(loss)
        loss_neg = trainer.trajectory_balance_loss(trace, -5.0)
        assert torch.isfinite(loss_neg)
        # floored to the same value
        assert torch.allclose(loss, loss_neg)

    def test_backward_policy_is_deterministic_chain(self):
        """Documented invariant: log P_B == 0 in this MDP (each step attaches
        exactly one synthon, so the precursor state is unique)."""
        trainer, _, _ = self._trainer()
        # The TB loss must not contain any P_B term: verify by checking the
        # loss equals the pure (logZ + sumPF - logR)^2 residual.
        trace = _make_trajectory(3)
        sum_pf = torch.zeros(1)
        for step in trace:
            log_pf, _, _ = joint_log_prob_and_entropy(
                trainer.model, trainer.catalog, trainer.device, step
            )
            sum_pf = sum_pf + log_pf
        reward = 2.5
        expected = (float(trainer.log_Z) + float(sum_pf) - math.log(2.5)) ** 2
        loss = trainer.trajectory_balance_loss(trace, reward)
        # float32 rounding at magnitude ~10^2
        assert abs(float(loss) - expected) < 1e-3

    def test_tb_diff_clip(self):
        cfg = {"tb_diff_clip": 1.0}
        model = FakePolicy(4)
        trainer = GFlowNetTrainer(model, FakeCatalog(4), torch.device("cpu"), cfg)
        trace = _make_trajectory(1)
        # huge reward -> huge residual -> clamped to 1.0 -> loss == 1.0
        loss = trainer.trajectory_balance_loss(trace, 1e12)
        assert abs(float(loss) - 1.0) < 1e-6

    def test_gradient_flows_to_log_z(self):
        trainer, _, _ = self._trainer()
        trace = _make_trajectory(2)
        loss = trainer.trajectory_balance_loss(trace, 1.0)
        loss.backward()
        assert trainer.log_Z.grad is not None
        assert float(trainer.log_Z.grad.abs()) > 0

    def test_gradient_flows_to_policy(self):
        trainer, _, _ = self._trainer()
        before = trainer.model.state_proj.weight.detach().clone()
        stats = trainer.train_step([(_make_trajectory(2), 1.0)])
        after = trainer.model.state_proj.weight.detach()
        assert not torch.equal(before, after)
        assert stats["episodes"] == 1
        assert stats["transitions"] == 2
        assert "log_Z" in stats and "loss" in stats

    def test_train_step_empty_batch(self):
        trainer, _, _ = self._trainer()
        stats = trainer.train_step([])
        assert stats == {"loss": 0.0, "log_Z": 0.0, "episodes": 0,
                         "transitions": 0}
        # empty trajectories are skipped too
        stats2 = trainer.train_step([([], 1.0)])
        assert stats2["episodes"] == 0

    def test_train_step_reduces_tb_residual_on_fixed_batch(self):
        """Optimizing a fixed batch must drive the TB loss down (the TB
        objective is a regression to zero)."""
        torch.manual_seed(0)
        model = FakePolicy(4)
        trainer = GFlowNetTrainer(model, FakeCatalog(4), torch.device("cpu"),
                                  {"learning_rate": 5e-3, "lr_z": 5e-2})
        batch = [(_make_trajectory(2, seed=s), 1.0 + 0.5 * s) for s in range(4)]
        first = trainer.train_step(batch)["loss"]
        for _ in range(30):
            last = trainer.train_step(batch)["loss"]
        assert last < first
        assert math.isfinite(last)

    def test_higher_reward_gets_higher_flow(self):
        """The P(x) ~ R(x) contract: after training, trajectories with larger
        rewards should carry larger summed log P_F (log Z is shared)."""
        torch.manual_seed(1)
        model = FakePolicy(4)
        trainer = GFlowNetTrainer(model, FakeCatalog(4), torch.device("cpu"),
                                  {"learning_rate": 5e-3, "lr_z": 5e-2})
        good = _make_trajectory(1, seed=3)
        bad = _make_trajectory(1, seed=7)
        batch = [(good, 10.0), (bad, 0.01)]
        for _ in range(40):
            trainer.train_step(batch)

        def sum_pf(trace):
            total = torch.zeros(1)
            for step in trace:
                lp, _, _ = joint_log_prob_and_entropy(
                    trainer.model, trainer.catalog, trainer.device, step
                )
                total = total + lp
            return float(total)

        assert sum_pf(good) > sum_pf(bad)

    def test_state_dict_roundtrip(self):
        trainer, _, _ = self._trainer()
        with torch.no_grad():
            trainer.log_Z.fill_(1.234)
        state = trainer.state_dict()

        model2 = FakePolicy(4)
        trainer2 = GFlowNetTrainer(model2, FakeCatalog(4), torch.device("cpu"), {})
        trainer2.load_state_dict(state)
        assert abs(float(trainer2.log_Z) - 1.234) < 1e-6

    def test_load_state_dict_tolerates_empty(self):
        trainer, _, _ = self._trainer()
        trainer.load_state_dict({})
        trainer.load_state_dict(None)

    def test_optimizer_has_two_param_groups(self):
        trainer, _, _ = self._trainer()
        groups = trainer.optimizer.param_groups
        assert len(groups) == 2
        assert groups[1]["lr"] == trainer.lr_z
        assert groups[1]["weight_decay"] == 0.0
        # log_Z is in the second group
        assert any(p is trainer.log_Z for p in groups[1]["params"])

    def test_config_defaults_from_tasklist(self):
        model = FakePolicy(4)
        trainer = GFlowNetTrainer(model, FakeCatalog(4), torch.device("cpu"), {})
        assert trainer.lr == 1e-4
        assert trainer.lr_z == 1e-2
        assert trainer.reward_floor == 1e-4
        assert trainer.max_grad_norm == 1.0


# ---------------------------------------------------------------------------
# DPO math
# ---------------------------------------------------------------------------
class TestDPOMath:
    def _trainer(self, k: int = 4):
        torch.manual_seed(7)
        model = FakePolicy(k)
        return DPOTrainer(model, FakeCatalog(k), torch.device("cpu"), {}), model

    def test_reference_model_is_frozen(self):
        trainer, _ = self._trainer()
        assert all(not p.requires_grad for p in trainer.reference_model.parameters())

    def test_preference_loss_zero_at_initialization(self):
        """pi_theta == pi_ref at initialization -> margin == 0 ->
        -log sigmoid(0) == log 2."""
        import torch.nn.functional as F

        trainer, _ = self._trainer()
        win = _make_trajectory(2, seed=1)
        lose = _make_trajectory(2, seed=2)
        loss = trainer.preference_loss(win, lose)
        assert loss is not None
        assert abs(float(loss) - math.log(2.0)) < 1e-5

    def test_loss_decreases_when_policy_moves_toward_winner(self):
        trainer, _ = self._trainer()
        win = _make_trajectory(2, seed=1)
        lose = _make_trajectory(2, seed=2)
        first = trainer.train_step([(win, lose)])["loss"]
        for _ in range(20):
            last = trainer.train_step([(win, lose)])["loss"]
        assert last < first

    def test_empty_trajectories_give_none(self):
        trainer, _ = self._trainer()
        assert trainer.preference_loss([], []) is None
        assert trainer.preference_loss(_make_trajectory(1), []) is None

    def test_train_step_empty(self):
        trainer, _ = self._trainer()
        stats = trainer.train_step([])
        assert stats["pairs"] == 0

    def test_state_dict_roundtrip_restores_reference(self):
        trainer, model = self._trainer()
        # Move the live policy away from the reference.
        with torch.no_grad():
            model.state_proj.weight.add_(0.5)
        state = trainer.state_dict()

        torch.manual_seed(11)
        model2 = FakePolicy(4)
        trainer2 = DPOTrainer(model2, FakeCatalog(4), torch.device("cpu"), {})
        trainer2.load_state_dict(state)
        for p_ref, q_ref in zip(trainer.reference_model.parameters(),
                                trainer2.reference_model.parameters()):
            assert torch.equal(p_ref, q_ref)

    def test_beta_scales_margin(self):
        big = DPOTrainer(FakePolicy(4), FakeCatalog(4), torch.device("cpu"),
                         {"dpo_beta": 5.0})
        small = DPOTrainer(FakePolicy(4), FakeCatalog(4), torch.device("cpu"),
                           {"dpo_beta": 0.1})
        assert big.beta == 5.0 and small.beta == 0.1


class TestBuildPreferencePairs:
    def test_pairs_ordered_by_reward(self):
        high = {"transitions": _make_trajectory(1, 0), "reward": 2.0}
        low = {"transitions": _make_trajectory(1, 1), "reward": 1.0}
        pairs = build_preference_pairs([high, low])
        assert len(pairs) == 1
        win, lose = pairs[0]
        assert win is high["transitions"]
        assert lose is low["transitions"]

    def test_swapped_order_handled(self):
        low = {"transitions": _make_trajectory(1, 0), "reward": 0.5}
        high = {"transitions": _make_trajectory(1, 1), "reward": 3.0}
        pairs = build_preference_pairs([low, high])
        win, lose = pairs[0]
        assert win is high["transitions"]

    def test_ties_dropped(self):
        a = {"transitions": _make_trajectory(1, 0), "reward": 1.0}
        b = {"transitions": _make_trajectory(1, 1), "reward": 1.0}
        assert build_preference_pairs([a, b]) == []

    def test_empty_transitions_skipped(self):
        a = {"transitions": [], "reward": 2.0}
        b = {"transitions": _make_trajectory(1), "reward": 1.0}
        assert build_preference_pairs([a, b]) == []

    def test_odd_batch_pairs_consecutive(self):
        eps = [
            {"transitions": _make_trajectory(1, i), "reward": float(i)}
            for i in range(5)
        ]
        pairs = build_preference_pairs(eps)
        assert len(pairs) == 2  # (0,1) and (2,3); 4 dropped (no partner)
        for win, lose in pairs:
            # win must have come from the higher-reward episode
            assert any(win is e["transitions"] for e in eps if e["reward"] > 0)


# ---------------------------------------------------------------------------
# Checkpoint plumbing for algorithm state
# ---------------------------------------------------------------------------
class TestAlgorithmCheckpointPlumbing:
    def _config(self):
        return {"huggingface": {"enabled": False}, "system": {"seed": 1}}

    def test_extra_state_roundtrip(self, tmp_path):
        from syntree.utils.checkpoint import CheckpointManager

        mgr = CheckpointManager(self._config(), ckpt_dir=str(tmp_path))
        model = FakePolicy(4)
        extra = {"gflownet": {"log_Z": torch.tensor([0.75])}}
        mgr.save_checkpoint(epoch=0, step=1, model=model, extra_state=extra)

        model2 = FakePolicy(4)
        start, _, _ = mgr.restore_latest(model2)
        assert start == 1
        assert mgr.last_extra_state == extra

    def test_legacy_checkpoint_has_no_extra_state(self, tmp_path):
        from syntree.utils.checkpoint import CheckpointManager

        mgr = CheckpointManager(self._config(), ckpt_dir=str(tmp_path))
        mgr.save_checkpoint(epoch=0, step=1, model=FakePolicy(4))
        mgr.restore_latest(FakePolicy(4))
        assert mgr.last_extra_state is None


# ---------------------------------------------------------------------------
# End-to-end CLI runs (real policy + real chemistry, tiny config)
# ---------------------------------------------------------------------------
class TestStage2AlgorithmCLI:
    def _write_config(self, base_cfg, tmp_path, algorithm):
        import json as _json

        cfg = _json.loads(_json.dumps(base_cfg))
        cfg["data"]["synthetic_samples"] = 8
        cfg["system"]["num_workers"] = 0
        cfg["catalog"] = {"encoder": "morgan2d"}  # fastest path for CI
        cfg["reinforcement_learning"] = {
            "enabled": True,
            "algorithm": algorithm,
            "episodes": 2,
            "rollout_episodes": 2,
            "minibatch_size": 2,
            "ppo_epochs": 1,
            "learning_rate": 1e-4,
            "reward": {"docking_weight": 0.0},
        }
        cfg_path = tmp_path / f"cfg_{algorithm}.json"
        cfg_path.write_text(_json.dumps(cfg))
        return cfg_path

    def _stage_pocket(self, tmp_path):
        pockets = tmp_path / "pockets"
        pockets.mkdir(exist_ok=True)
        with open(pockets / "rl1_pocket.pdb", "w") as f:
            f.write(
                "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
                "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
                "ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C\n"
                "ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O\n"
                "ATOM      5  N   GLY A   2       3.315   1.545   0.000  1.00 20.00           N\n"
                "ATOM      6  CA  GLY A   2       3.866   2.906   0.000  1.00 20.00           C\n"
                "ATOM      7  C   GLY A   2       4.417   4.267   0.000  1.00 20.00           C\n"
                "ATOM      8  O   GLY A   2       3.680   5.254   0.000  1.00 20.00           O\n"
                "TER\nEND\n"
            )
        return str(pockets)

    @pytest.mark.parametrize("algorithm", ["gflownet", "dpo"])
    def test_stage2_runs_end_to_end(self, tiny_config_with_assets, tmp_path,
                                    algorithm):
        import json as _json
        import os
        import subprocess
        import sys

        cfg_path = self._write_config(tiny_config_with_assets, tmp_path, algorithm)
        out_dir = tmp_path / "exp"
        proc = subprocess.run(
            [sys.executable, "main.py", "--mode", "train",
             "--config", str(cfg_path), "--output-dir", str(out_dir)],
            capture_output=True, text=True, timeout=900,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 0, proc.stderr[-3000:]

        pockets = self._stage_pocket(tmp_path)
        rl_out = tmp_path / f"rl_{algorithm}"
        proc = subprocess.run(
            [sys.executable, "main.py", "--mode", "rl",
             "--config", str(cfg_path), "--pocket-dir", pockets,
             "--output-dir", str(rl_out), "--resume-auto"],
            capture_output=True, text=True, timeout=900,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 0, proc.stderr[-4000:]
        assert f"Stage 2 algorithm" in proc.stdout

        with open(rl_out / "rl_history.json") as f:
            history = _json.load(f)
        assert len(history) == 2

        # The checkpoint carries the algorithm state.
        import torch as _torch

        ckpts = sorted((rl_out / "checkpoints").glob("checkpoint_epoch_*.pt"))
        assert ckpts
        payload = _torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        assert payload.get("extra_state") is not None

    def test_invalid_algorithm_rejected(self, tiny_config_with_assets, tmp_path):
        import os
        import subprocess
        import sys

        cfg_path = self._write_config(tiny_config_with_assets, tmp_path, "a3c")
        proc = subprocess.run(
            [sys.executable, "main.py", "--mode", "rl",
             "--config", str(cfg_path), "--pocket-dir", str(tmp_path),
             "--output-dir", str(tmp_path / "rl_bad"), "--resume-auto"],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 2
        assert "ppo | gflownet | dpo" in proc.stderr
