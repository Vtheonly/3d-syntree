"""Grammar-masked synthon selection head.

Cross-attends from the ligand attachment-handle query to the pre-encoded
synthon library embeddings, then applies the reaction-compatibility mask
before the softmax:

    p_synthon = Softmax(q_u E_B^T / sqrt(d) + M_rxn)

where ``M_rxn(u, j) = 0`` when synthon ``j`` can legally react with handle
``u`` and ``-inf`` otherwise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SynthonHead(nn.Module):
    """Reaction-masked synthon scoring head.

    Args:
        hidden_dim: dimensionality of queries and synthon embeddings.
        num_heads: multi-head attention head count (must divide hidden_dim).
        dropout: attention dropout.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must divide hidden_dim ({hidden_dim})"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.stop_embedding = nn.Parameter(torch.randn(hidden_dim) * (hidden_dim ** -0.5))
        self.stop_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        query: torch.Tensor,                 # [B, d] handle/pocket context
        synthon_embeddings: torch.Tensor,    # [K, d] catalog embeddings
        rxn_compatibility_mask: Optional[torch.Tensor] = None,  # [B, K]
        stop_mask: Optional[torch.Tensor] = None,               # [B] or [B, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute masked synthon logits and log-probabilities.

        Args:
            query: context vector for each graph in the batch.
            synthon_embeddings: catalog embedding table.
            rxn_compatibility_mask: additive logit mask; ``0`` keeps a
                synthon legal, ``-1e9`` (or any large negative) removes it.

        Returns:
            ``(logits, log_probs)`` both ``[B, K + 1]``. Index ``K`` is STOP.
        """
        if query.dim() != 2:
            raise ValueError(f"query must be [B, d], got {tuple(query.shape)}")
        if synthon_embeddings.dim() != 2:
            raise ValueError(
                f"synthon_embeddings must be [K, d], got {tuple(synthon_embeddings.shape)}"
            )
        if synthon_embeddings.size(1) != self.hidden_dim:
            raise ValueError(
                f"synthon embedding dim {synthon_embeddings.size(1)} != "
                f"hidden_dim {self.hidden_dim}"
            )

        b, k = query.size(0), synthon_embeddings.size(0)

        q = self.q_proj(query).view(b, self.num_heads, self.head_dim)   # [B, h, dh]
        keys = self.k_proj(synthon_embeddings).view(k, self.num_heads, self.head_dim)
        vals = self.v_proj(synthon_embeddings).view(k, self.num_heads, self.head_dim)

        # Attention scores per head: [B, h, K]
        scores = torch.einsum("bhd,khd->bhk", q, keys) / math.sqrt(self.head_dim)

        if rxn_compatibility_mask is not None:
            if rxn_compatibility_mask.shape != (b, k):
                raise ValueError(
                    f"mask shape {tuple(rxn_compatibility_mask.shape)} != ({b}, {k})"
                )
            scores = scores + rxn_compatibility_mask.unsqueeze(1)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Context per head -> concat -> output projection.
        context = torch.einsum("bhk,khd->bhd", attn, vals)              # [B, h, dh]
        context = context.reshape(b, self.hidden_dim)
        out = self.out_proj(context)

        # Final scoring: dot product between projected context and the raw
        # synthon embeddings (tied readout).
        logits = torch.matmul(out, synthon_embeddings.t()) / math.sqrt(self.hidden_dim)
        if rxn_compatibility_mask is not None:
            logits = logits + rxn_compatibility_mask

        # Explicit STOP action. It is conditioned on the same policy context
        # and receives its own learned embedding/bias, making termination a
        # learned action instead of an implicit “no handles left” side effect.
        stop_logit = (out * self.stop_embedding.unsqueeze(0)).sum(-1) / math.sqrt(self.hidden_dim)
        stop_logit = stop_logit + self.stop_bias
        if stop_mask is not None:
            stop_mask = stop_mask.view(b, 1).to(logits.dtype)
            stop_logit = stop_logit + stop_mask.squeeze(1)
        logits = torch.cat([logits, stop_logit.unsqueeze(-1)], dim=-1)

        log_probs = F.log_softmax(logits, dim=-1)
        return logits, log_probs


def make_compatibility_mask(
    allowed: torch.Tensor, num_synthons: int, device=None, mask_value: float = -1e9
) -> torch.Tensor:
    """Build an additive ``[B, K]`` mask from boolean allowed-flags.

    Args:
        allowed: ``[B, K]`` boolean tensor (True = legal synthon).
        num_synthons: catalog size ``K`` (checked against ``allowed``).
    """
    if allowed.shape[-1] != num_synthons:
        raise ValueError(
            f"allowed has {allowed.shape[-1]} columns, expected {num_synthons}"
        )
    mask = torch.zeros_like(allowed, dtype=torch.float32)
    mask[~allowed.bool()] = mask_value
    if device is not None:
        mask = mask.to(device)
    return mask


__all__ = ["SynthonHead", "make_compatibility_mask"]
