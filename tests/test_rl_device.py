"""Regression test for device propagation in PPO rollout collection."""

from __future__ import annotations

import torch
from torch import nn

from syntree.engine.rl import collect_episode


class FakeEnv:
    def __init__(self):
        self.done = False

    def reset(self, **kwargs):
        self.done = False
        return torch.zeros(4)

    def legal_action_masks(self):
        return (
            torch.zeros(1),
            torch.zeros(1, 1, 1),
            True,
        )

    def step(self, action):
        self.done = True
        return torch.zeros(4), 0.0, True, {}


class DeviceCheckingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(1, device="meta"))

    def act(
        self,
        observation,
        synthon_embeddings,
        reaction_compatibility_mask,
        synthon_masks_by_reaction,
        sample=True,
        temperature=1.0,
    ):
        device = self.anchor.device
        assert observation.device == device
        assert synthon_embeddings.device == device
        assert reaction_compatibility_mask.device == device
        assert synthon_masks_by_reaction.device == device
        return {
            "reaction_family_idx": torch.tensor([0]),
            "action_idx": torch.tensor([0]),
            "synthon_idx": torch.tensor([0]),
            "dihedral": torch.tensor([0.0]),
            "joint_log_prob": torch.tensor([0.0]),
            "state_value": torch.tensor([0.0]),
        }


class FakeCatalog:
    def __init__(self):
        self.embeddings = torch.randn(3, 5)


def test_collect_episode_moves_catalog_and_masks_to_model_device():
    model = DeviceCheckingModel()
    transitions, _ = collect_episode(FakeEnv(), model, FakeCatalog())

    assert len(transitions) == 1
    assert transitions[0].done is True
