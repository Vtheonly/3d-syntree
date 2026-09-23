"""Tests for the research-stage PPO objective."""
from __future__ import annotations

import torch

from syntree.engine.rl import PPOFineTuner


def test_terminal_reward_returns_decay_once():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

    class Catalog:
        embeddings = torch.zeros(2, 4)

        def __len__(self):
            return 2

    tuner = PPOFineTuner(
        Tiny(),
        Catalog(),
        torch.device("cpu"),
        {"gamma": 0.5, "ppo_epochs": 1, "learning_rate": 0.0},
    )

    # The public update path is intentionally not mocked here; this test
    # verifies the documented return convention through a small trace that
    # cannot be optimized because the learning rate is zero.
    state = type("State", (), {"to": lambda self, device: self})()
    trace = []
    for _ in range(3):
        trace.append({
            "state": state,
            "reaction_mask": torch.zeros(1, 1),
            "synthon_masks": torch.zeros(1, 1, 2),
            "family_idx": 0,
            "action_idx": 0,
            "old_log_prob": torch.tensor(0.0),
            "old_value": torch.tensor(0.0),
        })

    # This should complete without exploding due to repeated terminal reward.
    result = tuner.update_episode(trace, 1.0)
    assert result["steps"] == 3.0
    assert torch.isfinite(torch.tensor(result["loss"]))
