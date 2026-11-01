"""Stable von Mises circular-torsion mathematics (issue #8 MATH-02).

Clean-room V2 implementation of the numerically stable ``log I_0(kappa)``
primitive and the circular torsion objective (constitution #45 §8.2;
issues #8 → #24 MODEL-04 → #26/#32).

**Why this module exists (the #8 motivation — legacy "Singularity 2").**
The pre-V2 torsion head evaluated ``log I_0(kappa)`` with an unrolled
20-iteration polynomial series *through autograd*: differentiating 20
successive powers of ``kappa`` diverged badly for ``kappa > 10`` (the
series terms grow like ``(kappa^2/4)^k / (k!)^2`` before they decay, so
the backward pass amplified rounding error catastrophically).  Worse, the
legacy code branched with ``torch.where`` *inside* the autograd graph, so
gradients were evaluated across **both** the active and the inactive
branch — a masked ``log(kappa -> 0) = -inf`` multiplied by a zero mask
factor produced NaN gradients that silently poisoned training.  V2 fixes
both failures structurally:

1. ``log I_0`` is a custom :class:`torch.autograd.Function`
   (:class:`LogBesselI0`) whose **forward** evaluates the pinned two-branch
   approximation (power series below 12, asymptotic expansion at/above 12,
   both computed internally in float64) and whose **backward** is the
   *analytic* Amos' ratio — no unrolled-series differentiation exists
   anywhere in the graph.
2. The ``torch.where`` branch selection happens **inside** the custom
   Function's forward, which autograd never tracks; the backward is a
   single closed-form expression that is finite for every valid ``kappa``.
   There is no inactive-branch gradient path at all, hence no NaN leakage
   (the legacy failure mode is structurally impossible, and this is pinned
   by tests at the branch seam).

**The pinned mathematics (issue #8 body).**

Forward (values, ``kappa >= 0``)::

    log I_0(kappa) = log( sum_{k>=0} (kappa^2/4)^k / (k!)^2 )
                   = kappa - 0.5*(log(2*pi*kappa)) + log(1 + u + 4.5u^2*8 ... )

concretely the issue pins the two branches

* power series (``kappa < 12``): 22-term Horner-style accumulation in
  float64 with an early-exit once every term falls below ``1e-12`` of the
  running total (converges to float64 rounding for ``kappa < 12``);
* asymptotic expansion (``kappa >= 12``):::

      log I_0(kappa) = kappa - 0.5*(log(2*pi) + log(kappa))
                              + log1p( u * (1 + u*(4.5 + u*37.5)) ),
      u = 1/(8*kappa)

  whose ``1 + u + 4.5 u^2 * 8^{-2}...`` coefficients match the true
  ``I_0`` asymptotic series ``1 + 1/(8k) + 9/(128 k^2) + 225/(3072 k^3)``
  **exactly** through ``O(kappa^-3)``; the truncation error of the next
  term is ``~0.112 / kappa^4``, i.e. ``~5.4e-6`` at the ``kappa = 12``
  seam and vanishingly small beyond — which is precisely how the pinned
  tolerance (absolute error ``< 1e-5`` against ``scipy.special.i0e`` over
  ``kappa in [0, 100]``) is met with margin at the seam.

Backward (the analytic Amos' ratio approximation, issue-pinned)::

    d/dkappa log I_0(kappa) = I_1(kappa)/I_0(kappa) = A(kappa)
    A(kappa) ~= kappa / (0.5 + sqrt(kappa^2 + 2.25))

``A`` is **strictly bounded in [0, 1) for every ``kappa >= 0``** (proof:
``A < 1`` iff ``sqrt(kappa^2 + 2.25) > kappa - 0.5``; for ``kappa <= 0.5``
the right side is non-positive while the left is positive, and for
``kappa > 0.5`` squaring gives ``kappa^2 + 2.25 > kappa^2 - kappa + 0.25``,
i.e. ``kappa + 2 > 0``, always true).  ``A(0) = 0`` exactly, so the
gradient at the degenerate uniform distribution is exactly zero, and the
denominator is bounded below by ``2.0`` — the backward is finite and
clamp-free by construction.  This is an **approximation, not an exact
identity** (#45 §8.2): its deviation from the trusted reference ratio
``I_1/I_0 = scipy.special.i1e/i0e`` is measured and pinned by the tests
over ``kappa in [0, 100]`` (measured maximum absolute error 3.30e-2 at
``kappa ~ 2.5``; 7.9e-4 at ``kappa = 30``, 2.0e-4 at 60, 7.4e-5 at 100 —
decaying like ``O(kappa^-2)``), and the forward itself carries a measured
maximum error of **6.5e-6** against the ``i0e`` reference over the same
range (below the pinned 1e-5 tolerance with margin).  The same bounded
Amos form is standard circular-statistics practice.

**Fail-closed domain (constitution §8, decisions D-V2-004/D-V2-005).**
The von Mises concentration is non-negative by definition; ``kappa`` must
be a finite floating tensor with ``kappa >= 0``.  NaN/Inf/negative inputs
raise instead of being clamped: the issue snippet's defensive clamps
(``clamp(min=1e-6)``, ``clamp(min=1e-30)``) are provably dead on the valid
domain (the series total is ``>= 1``; the asymptotic branch is gated to
``kappa >= 12``; ``A`` has no singularity at ``kappa = 0``) and are
replaced by exact ``torch.where`` branch gating plus explicit validation —
identical values on the valid domain, strict rejection outside it.

**Consumers.**  #24 MODEL-04 consumes :func:`log_bessel_i0` /
:func:`von_mises_loss` for the circular torsion loss over the 36-bin
action space (the discrete-bin circular loss per #45 §12.1); #32 TEST-01
aggregates the tests here into Suite A.  The head *architecture* (over
``DecoderOutput.context``) remains #24's scope — this module owns the
mathematics only.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "VON_MISES_SERIES_THRESHOLD",
    "AMOS_RATIO_CONSTANT",
    "LOG_2PI",
    "LogBesselI0",
    "log_bessel_i0",
    "amos_ratio",
    "von_mises_log_prob",
    "von_mises_nll",
    "von_mises_loss",
]

#: Branch threshold pinned by issue #8: power series below, asymptotic
#: expansion at/above.  Calibrated so the asymptotic truncation error
#: (~0.112/kappa^4) is below the pinned 1e-5 tolerance exactly at the seam.
VON_MISES_SERIES_THRESHOLD = 12.0

#: The ``2.25`` inside the Amos ratio square root (issue-pinned form).
AMOS_RATIO_CONSTANT = 2.25

#: ``log(2*pi)`` — the von Mises normalisation constant.
LOG_2PI = math.log(2.0 * math.pi)


def _validate_kappa(kappa: torch.Tensor, name: str = "kappa") -> torch.Tensor:
    """Fail-closed domain check for a concentration tensor ``kappa >= 0``."""
    if not isinstance(kappa, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(kappa).__name__}")
    if not torch.is_floating_point(kappa):
        raise TypeError(
            f"{name} must be a floating tensor, got dtype {kappa.dtype}"
        )
    if kappa.numel() and not bool(torch.isfinite(kappa).all()):
        raise ValueError(f"{name} contains non-finite values (fail-closed)")
    if kappa.numel() and bool((kappa < 0).any()):
        raise ValueError(
            f"{name} contains negative values: the von Mises concentration is "
            f"non-negative by definition (fail-closed)"
        )
    return kappa


class LogBesselI0(torch.autograd.Function):
    """``log I_0(kappa)`` with the analytic Amos' ratio backward (#8 pinned).

    Forward — two branches, both evaluated internally in float64:

    * ``kappa < 12``: power series ``log(sum_k (kappa^2/4)^k / (k!)^2)``
      (up to 22 terms, early-exit at ``1e-12`` relative contribution);
    * ``kappa >= 12``: asymptotic expansion
      ``kappa - 0.5*(log(2 pi) + log kappa) + log1p(u (1 + u (4.5 + 37.5 u)))``
      with ``u = 1/(8 kappa)``.

    The branch selection uses exact ``torch.where`` gating (the inactive
    branch receives gated *safe* inputs — zeros for the series, ``>= 1``
    for the logarithm — so no infinity is ever materialised), and the
    output is cast back to the input dtype.

    Backward — the issue-pinned bounded Amos' ratio::

        d log I_0 / d kappa  ->  A(kappa) = kappa / (0.5 + sqrt(kappa^2 + 2.25))

    computed in float64 and cast to the input dtype.  ``A`` is strictly in
    ``[0, 1)`` for all ``kappa >= 0`` (see module docstring for the proof),
    so the backward is finite, non-negative, and bounded *by mathematics*
    — no clamps, no NaN masking, and no unrolled-series differentiation.

    Domain: finite floating tensors with ``kappa >= 0`` (validated in
    forward; fail-closed otherwise).
    """

    @staticmethod
    def forward(ctx, kappa: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        kappa = _validate_kappa(kappa)
        ctx.save_for_backward(kappa)

        k = kappa.to(torch.float64)
        small_mask = k < VON_MISES_SERIES_THRESHOLD

        # ---- power series branch (kappa < 12), gated to safe inputs ------
        # Elements in the asymptotic regime are gated to 0: their series
        # value (which would overflow/grow huge before converging) is never
        # selected by the where() below, so the gate changes no output.
        k_small = torch.where(small_mask, k, torch.zeros_like(k))
        k2 = (k_small * 0.5) ** 2
        tot = torch.ones_like(k_small)
        term = torch.ones_like(k_small)
        for i in range(1, 22):
            term = term * k2 / (i * i)
            tot = tot + term
            if bool((term < 1e-12 * tot).all()):
                break
        res_small = torch.log(tot)

        # ---- asymptotic branch (kappa >= 12), gated to safe inputs --------
        # Elements in the series regime are gated to 1.0: log(1) is finite,
        # and the where() below never selects their asymptotic value.
        k_large = torch.where(small_mask, torch.ones_like(k), k)
        inv = 1.0 / (8.0 * k_large)
        res_large = k_large - 0.5 * (
            math.log(2.0 * math.pi) + torch.log(k_large)
        ) + torch.log1p(inv * (1.0 + inv * (4.5 + inv * 37.5)))

        return torch.where(small_mask, res_small, res_large).to(kappa.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (kappa,) = ctx.saved_tensors
        # Amos' bounded ratio, evaluated in float64.  The denominator is
        # bounded below by 0.5 + sqrt(2.25) = 2.0 > 0 for every kappa >= 0,
        # and the ratio is strictly in [0, 1) — no clamp is required.
        k = kappa.to(torch.float64)
        ratio = k / (0.5 + torch.sqrt(k * k + AMOS_RATIO_CONSTANT))
        return grad_output * ratio.to(kappa.dtype)


def log_bessel_i0(kappa: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log I_0(kappa)`` (issue #8 public wrapper).

    Parameters
    ----------
    kappa:
        Concentration tensor, finite, floating, ``>= 0`` (fail-closed).

    Returns
    -------
    torch.Tensor
        ``log I_0(kappa)`` with the input dtype; gradients flow through
        the analytic Amos' ratio (see :class:`LogBesselI0`).
    """
    return LogBesselI0.apply(kappa)


def amos_ratio(kappa: torch.Tensor) -> torch.Tensor:
    """The bounded Amos' ratio approximation ``A(kappa)`` (issue-pinned).

    ``A(kappa) = I_1(kappa)/I_0(kappa) ~ kappa / (0.5 + sqrt(kappa^2 + 2.25))``,
    strictly in ``[0, 1)`` for all ``kappa >= 0``; ``A(0) = 0`` exactly.
    Returned as a plain (autograd-trackable) expression so consumers can
    inspect or reuse the same approximation the backward uses — the value
    is an *approximation* of ``I_1/I_0`` (see the module docstring for the
    measured error against ``scipy.special.i1e/i0e``).
    """
    kappa = _validate_kappa(kappa)
    k = kappa.to(torch.float64)
    ratio = k / (0.5 + torch.sqrt(k * k + AMOS_RATIO_CONSTANT))
    return ratio.to(kappa.dtype)


def _validate_angle_tensors(
    mu: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor
) -> tuple:
    """Validate and promote the von Mises objective's arguments."""
    for name, tensor in (("mu", mu), ("kappa", kappa), ("phi", phi)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must be a floating tensor, got dtype {tensor.dtype}")
        if tensor.numel() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values (fail-closed)")
    _validate_kappa(kappa)
    try:
        torch.broadcast_shapes(mu.shape, kappa.shape, phi.shape)
    except RuntimeError as error:
        raise ValueError(
            f"mu/kappa/phi shapes are not broadcastable: "
            f"{tuple(mu.shape)}, {tuple(kappa.shape)}, {tuple(phi.shape)}"
        ) from error
    target_dtype = torch.promote_types(
        torch.promote_types(mu.dtype, kappa.dtype), phi.dtype
    )
    return mu.to(target_dtype), kappa.to(target_dtype), phi.to(target_dtype)


def von_mises_log_prob(
    mu: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Exact von Mises log-density on the circle.

    ``log p(phi | mu, kappa) = kappa * cos(phi - mu) - log(2 pi) - log I_0(kappa)``

    with ``log I_0`` routed through the stable :func:`log_bessel_i0`
    primitive, so the backward is the analytic Amos' ratio — the stable
    circular objective the #24 torsion loss consumes.  ``mu``/``phi`` may
    be any finite angles (the density is ``2 pi``-periodic in both);
    ``kappa`` must be finite and ``>= 0`` (fail-closed).
    """
    mu, kappa, phi = _validate_angle_tensors(mu, kappa, phi)
    return kappa * torch.cos(phi - mu) - LOG_2PI - log_bessel_i0(kappa)


def von_mises_nll(
    mu: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Elementwise negative log-density ``-log p(phi | mu, kappa)``."""
    return -von_mises_log_prob(mu, kappa, phi)


def von_mises_loss(
    mu: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Mean von Mises negative log-likelihood — the circular torsion loss.

    The scalar training objective for torsion supervision (#45 §12.1:
    "a circular loss for torsion bins/angles consistent with the selected
    action formulation").  Shapes must broadcast; the result is the mean
    over all elements.  Fails closed on empty inputs (a mean over nothing
    is a masked NaN, not a loss).
    """
    nll = von_mises_nll(mu, kappa, phi)
    if nll.numel() == 0:
        raise ValueError("von_mises_loss received empty angle tensors (fail-closed)")
    return nll.mean()
