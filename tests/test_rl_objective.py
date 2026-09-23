"""Tests for the research-stage PPO objective."""
from __future__ import annotations

import torch

from syntree.engine.rl import PPOFineTuner


def test_terminal_reward_returns_decay_once():
    returns = PPOFineTuner.terminal_returns(1.0, horizon=3, gamma=0.5)
    assert torch.allclose(
        returns,
        torch.tensor([0.25, 0.5, 1.0]),
    )


def test_empty_terminal_returns():
    returns = PPOFineTuner.terminal_returns(1.0, horizon=0, gamma=0.99)
    assert returns.numel() == 0
