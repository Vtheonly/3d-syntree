"""Continuous cosine cutoff envelope for spatial radial basis projections.

Clean-room V2 implementation of issue #9 MATH-03 (constitution #45 §8.3,
decision D-V2-011).

**Why the envelope (the #9 motivation — legacy "Vulnerability 7").** The
pre-V2 policy clamped handle-to-atom distances with
``clamp(min=0.0, max=cutoff)`` and projected them through a raw Gaussian
radial basis. Without an envelope that decays to zero at the cutoff, every
spectator atom from ``cutoff + eps`` to arbitrarily far away received an
identical, non-zero RBF bias feature (the clamped value saturates the
nearest Gaussian center), corrupting cross-attention keys with unphysical
background noise. The cosine envelope kills that failure mode exactly:
beyond the cutoff the feature is *identically* zero (not merely small), and
inside the cutoff it decays smoothly to zero with a continuous first
derivative, so radial features taper to zero at the graph-construction
boundary instead of appearing/disappearing abruptly.

**The envelope (issue #9 pinned form).** For distance ``d`` and cutoff
``r_cut``::

    f_cut(d) = 0.5 * (cos(pi * d / r_cut) + 1)   for d <= r_cut
    f_cut(d) = 0                                   for d > r_cut

Properties (pinned by ``tests/test_cosine_cutoff.py``):

* ``f_cut(0) = 1`` exactly (``cos(0) + 1 = 2``).
* ``f_cut(d) -> 0`` quadratically as ``d -> r_cut`` from below
  (``0.5 * (1 - cos(pi * delta / r_cut)) ~ (pi*delta/r_cut)^2 / 4`` for
  ``delta = r_cut - d``); on the CPU reference platforms the value at
  ``d = r_cut`` is exactly ``0.0`` in float32 and float64.
* ``f_cut(d) = 0`` *identically* (bitwise) for every ``d > r_cut`` — the
  ``torch.where`` branch selects an exact zero tensor, so no spectator at
  any distance beyond the cutoff contributes anything.
* ``C^1`` at the seam: the derivative ``-0.5 * (pi/r_cut) *
  sin(pi * d / r_cut)`` tends to ``0`` linearly as ``d -> r_cut^-`` and is
  exactly ``0`` beyond the cutoff, so ``radial(d) = RBF(d) * f_cut(d)`` has
  a continuous derivative approaching the boundary (the pinned #9
  acceptance criterion).
* Reflection identity: ``f_cut(d) + f_cut(r_cut - d) = 1`` on ``[0, r_cut]``
  (cosine symmetry), used as an analytic cross-check.

**One authoritative place (decision D-V2-011).** The envelope is
implemented exactly once, here, and is applied inside
:class:`syntree.models.bipartite.GaussianRadialBasis` — the single V2
producer of *all* spatial distance RBF projections (the #21 backbone's
edge-distance expansion and the #22 handle head's centroid-distance
expansion over ``[0, 12]`` Angstrom). Downstream consumers therefore
cannot forget to apply it: there is no raw-Gaussian escape hatch anywhere
in the V2 model stack. The #23 cross-attention decoder must likewise route
its spatial RBF through ``GaussianRadialBasis`` rather than re-implementing
an expansion.

**Numerical policy (constitution #45 §8).** No NaN masking, arbitrary
clamping of *valid* inputs, or silent fallbacks: non-finite distances and
non-floating dtypes fail closed with explicit errors, and the only clamp
is the issue-mandated ``clamp(min=0.0)`` that maps negative roundoff to
zero (documented semantics of the pinned reference implementation). The
function is parameter-free, deterministic, and dtype-preserving.
"""

from __future__ import annotations

import math

import torch

from syntree.core.schema import TensorContractError

__all__ = [
    "continuous_cosine_envelope",
]


def continuous_cosine_envelope(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth ``C^1``-continuous cosine cutoff envelope (issue #9 pinned form).

    Computes::

        f_cut(d) = 0.5 * (cos(pi * d / cutoff) + 1)   if d <= cutoff
        f_cut(d) = 0                                    if d > cutoff

    Args:
        dist: tensor of distances, any rank, floating dtype. Negative values
            are clipped to zero (the pinned reference semantics — distances
            are non-negative by definition, and tiny negative roundoff maps
            to ``f_cut(0) = 1``); non-finite values fail closed.
        cutoff: the cutoff radius ``r_cut`` (same units as ``dist``); must be
            a finite positive number.

    Returns:
        Tensor of the envelope values with the same shape and dtype as
        ``dist``, bounded in ``[0, 1]``, exactly zero (bitwise) for every
        element with ``d > cutoff``.

    Raises:
        TensorContractError: ``dist`` is not a ``torch.Tensor`` of a floating
            dtype, or contains non-finite values (poisoned inputs fail
            closed — locate the producer).
        ValueError: ``cutoff`` is not a finite positive number.
    """
    if not isinstance(dist, torch.Tensor):
        raise TensorContractError(
            f"[cosine-cutoff] `dist` must be a torch.Tensor, got "
            f"{type(dist).__name__}"
        )
    if not dist.is_floating_point():
        raise TensorContractError(
            "[cosine-cutoff] `dist` must have a floating dtype "
            f"(float32/float64 per the V2 precision policy), got "
            f"{dist.dtype}"
        )
    if dist.numel() and not bool(torch.isfinite(dist).all()):
        raise TensorContractError(
            "[cosine-cutoff] `dist` contains non-finite values "
            "(NaN/Inf); poisoned inputs fail closed — locate the producer"
        )
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(
            f"continuous_cosine_envelope requires a finite cutoff > 0, got "
            f"{cutoff}"
        )

    # Issue #9 pinned reference implementation.
    clipped_dist = dist.clamp(min=0.0)
    envelope = 0.5 * (torch.cos(clipped_dist * (math.pi / cutoff)) + 1.0)
    return torch.where(clipped_dist <= cutoff, envelope, torch.zeros_like(envelope))
