"""FP16-safe additive logit-mask constants shared across the grammar engine.

Colab T4 incident (Stage-1 ``loss nan`` / ``val_loss 98,958,339``):

The reaction-grammar masks historically used ``-1e9`` as the additive logit
fill value. IEEE 754 half precision (binary16) can only represent magnitudes
up to 65,504, so under fp16 autocast every masked logit silently overflowed
to ``-inf``. Combined with a masked *ground-truth* action this produced
``+inf`` cross-entropy and NaN gradients; even with a legal target,
``-inf`` logits that survived into attention denominators produced NaN
weights, and validation cross-entropy on masked targets inflated the
reported ``val_loss`` to the 1e8 range.

Semantics of the fix:

* ``LOGIT_MASK_FILL = -1e4`` is exactly representable in fp16
  (|value| <= 65,504) and still yields a softmax probability of
  ``exp(-1e4) ~ 0`` (below fp16 resolution), so masking behavior is
  mathematically unchanged while every intermediate stays finite.
* ``LOGIT_MASK_LEGAL_ABOVE = -1e3`` is the legality threshold used by
  boolean checks such as ``(mask > LOGIT_MASK_LEGAL_ABOVE).any()`` /
  ``mask[idx] < LOGIT_MASK_LEGAL_ABOVE`` to separate legal entries
  (``0.0``) from masked entries (``-1e4``). The legacy ``-1e8`` threshold
  must NOT be used with the new fill value: ``-1e4 > -1e8`` is True, so a
  masked entry would be misclassified as legal.

Every mask producer (``chemistry.catalog``, ``engine.environment``) and every
legality check (``engine.rl``, ``engine.generator``, ``engine.environment``)
must source these constants instead of hard-coding sentinel values.
"""

from __future__ import annotations

import torch

#: Additive logit-mask fill value for illegal/unsupported actions.
#: FP16-safe (fp16 max finite magnitude is 65,504).
LOGIT_MASK_FILL = -1e4

#: Entries strictly above this threshold are considered legal (mask == 0.0).
LOGIT_MASK_LEGAL_ABOVE = -1e3


def mask_is_legal(mask: torch.Tensor) -> torch.Tensor:
    """Boolean legality map: True where the additive mask keeps an action legal.

    Args:
        mask: additive logit mask (any shape, any float dtype).

    Returns:
        Boolean tensor with True at legal entries (``mask`` above
        :data:`LOGIT_MASK_LEGAL_ABOVE`, i.e. not masked out by
        :data:`LOGIT_MASK_FILL`).
    """
    return mask > LOGIT_MASK_LEGAL_ABOVE


def has_any_legal(mask: torch.Tensor) -> bool:
    """Whether at least one entry of an additive mask is legal."""
    return bool(mask_is_legal(mask).any().item())


__all__ = [
    "LOGIT_MASK_FILL",
    "LOGIT_MASK_LEGAL_ABOVE",
    "mask_is_legal",
    "has_any_legal",
]
