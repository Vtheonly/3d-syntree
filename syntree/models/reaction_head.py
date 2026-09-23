"""Reaction-family prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReactionHead(nn.Module):
    """Predicts the synthesis reaction family from pocket/handle context."""

    def __init__(self, hidden_dim: int, num_families: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim <= 0 or num_families <= 0:
            raise ValueError("hidden_dim and num_families must be positive")
        self.num_families = int(num_families)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, num_families),
        )

    def forward(self, context: torch.Tensor, compatibility_mask=None):
        logits = self.net(context)
        if compatibility_mask is not None:
            if compatibility_mask.shape != logits.shape:
                raise ValueError(
                    f"reaction mask shape {tuple(compatibility_mask.shape)} "
                    f"!= logits shape {tuple(logits.shape)}"
                )
            logits = logits + compatibility_mask
        return logits, F.log_softmax(logits, dim=-1)


__all__ = ["ReactionHead"]
