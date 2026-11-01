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
aggregates the tests here into Suite A.

**The #24 head architecture (this module's second scope).**  Issue #24
adds the *policy-level* torsion head here: the strictly categorical
36-bin distribution ``P(phi_t = k | s_t, B_t) = Softmax(z_torsion)[k]``
over the decoder decision context ``s_t`` and the tied catalog embedding
of the selected synthon ``B_t`` (the issue pins the conditioning on
``B_t`` explicitly).  The categorical is the *only* probability measure
this head contributes to Trajectory Balance — the auxiliary von Mises
parameters ``(mu, kappa)`` exist solely for the supervised circular loss
(:func:`torsion_circular_loss`) and never enter the TB log-probability
(the pre-V2 failure of mixing a ``rad^-1`` density with dimensionless
discrete probabilities — the #24 motivation, vulnerability 2 — is
structurally impossible here: :meth:`TorsionOutput.selected_log_prob` is
a pure categorical log-probability).

``mu`` is produced as the angle of a 2D vector routed through the
project's certified non-singular :func:`safe_unit_vector` primitive
(#7 MATH-01): the atan2 of a smoothed unit vector has bounded gradients
for every input, including the zero vector — no 0/0 direction, ever.
``kappa`` is the softplus of a raw scalar — a smooth parameterization of
the non-negative concentration domain (mirroring the issue-pinned Amos
form), not a clamp.  The learned ``kappa`` is anchored by the supervised
angular residual (its NLE optimum is ``A^{-1}(cos Delta)`` for residual
``Delta``, finite unless ``mu`` hits the target exactly), and the #20
coordinate/dihedral augmentation keeps supervised angles non-degenerate;
TB never sees it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from syntree.models.equivariant import safe_unit_vector
from syntree.utils.hardware import HardwareError, apply_logit_mask

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
    "TORSION_BINS",
    "TORSION_BIN_WIDTH",
    "TORSION_HEAD_CONTEXT_DIM",
    "TORSION_HEAD_SYNTON_DIM",
    "TORSION_HEAD_HIDDEN_DIM",
    "torsion_bin_center",
    "torsion_bin_centers",
    "torsion_circular_loss",
    "TorsionOutput",
    "TorsionHead",
]

#: Branch threshold pinned by issue #8: power series below, asymptotic
#: expansion at/above.  Calibrated so the asymptotic truncation error
#: (~0.112/kappa^4) is below the pinned 1e-5 tolerance exactly at the seam.
VON_MISES_SERIES_THRESHOLD = 12.0

#: The ``2.25`` inside the Amos ratio square root (issue-pinned form).
AMOS_RATIO_CONSTANT = 2.25

#: ``log(2*pi)`` — the von Mises normalisation constant.
LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# The discrete SO(2) action space (issue #24 MODEL-04)
# ---------------------------------------------------------------------------

#: Number of discrete torsion bins (issue #24: ``N_phi = 36``, 10-degree
#: resolution over ``[-pi, pi)``).
TORSION_BINS = 36

#: Width of one torsion bin in radians (``2*pi/36``).
TORSION_BIN_WIDTH = 2.0 * math.pi / TORSION_BINS

#: Production context width consumed by :class:`TorsionHead` — the decoder
#: decision context (:data:`~syntree.models.policy.DECODER_D_MODEL` = 256;
#: equality is pinned by test, not by import, to keep the module import
#: graph acyclic).
TORSION_HEAD_CONTEXT_DIM = 256

#: Production width of the conditioning synthon embedding — the tied
#: catalog embedding width (:data:`~syntree.models.policy.SYNTON_EMBED_DIM`
#: = 64; pinned equal by test).
TORSION_HEAD_SYNTON_DIM = 64

#: Hidden width of the torsion head trunk.
TORSION_HEAD_HIDDEN_DIM = 128


def torsion_bin_center(bin_index: int) -> float:
    """Exact static centre of torsion bin ``bin_index`` (1-based, 1..36).

    ``phi_k = -pi + (k - 0.5) * 2*pi/36`` — the canonical bin centres used
    for *all* intermediate 3D geometry during rollout (issue #24: the
    discrete action determines the geometry deterministically).  This is
    the models-side twin of the engine's
    :func:`syntree.engine.environment.bin_center`; exact equality for all
    36 bins is pinned by test (the two modules deliberately do not import
    each other — the contract is enforced by the test suite instead).
    """
    if isinstance(bin_index, bool) or not isinstance(bin_index, (int, float)):
        raise ValueError(
            f"torsion bin index must be an integer in [1, {TORSION_BINS}], "
            f"got {bin_index!r}"
        )
    if not 1 <= bin_index <= TORSION_BINS:
        raise ValueError(
            f"torsion bin index must be in [1, {TORSION_BINS}], got {bin_index}"
        )
    return -math.pi + (bin_index - 0.5) * TORSION_BIN_WIDTH


def torsion_bin_centers() -> Tuple[float, ...]:
    """The 36 canonical bin centres, in bin order ``k = 1..36``."""
    return tuple(torsion_bin_center(k) for k in range(1, TORSION_BINS + 1))


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


# ---------------------------------------------------------------------------
# The 36-bin categorical torsion head (issue #24 MODEL-04)
# ---------------------------------------------------------------------------

#: Column order of the parity-free torsion action: bins are 1-based in the
#: environment's ``AssemblyAction.torsion_bin`` and 0-based inside the
#: ``[B, 36]`` logit tensors (``bin k`` <-> column ``k - 1``).
_TORSION_FLOAT_DTYPES = (torch.float32, torch.float64)


def _validate_head_inputs(
    context: torch.Tensor,
    synthon_embedding: torch.Tensor,
    *,
    context_dim: int,
    synthon_dim: int,
    param_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fail-closed validation of the torsion head's two input tensors."""
    for name, tensor, dim in (
        ("context", context, context_dim),
        ("synthon_embedding", synthon_embedding, synthon_dim),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"[torsion-head] `{name}` must be a torch.Tensor, got "
                f"{type(tensor).__name__}"
            )
        if tensor.dtype not in _TORSION_FLOAT_DTYPES:
            raise TypeError(
                f"[torsion-head] `{name}` must have dtype float32 (the "
                f"production path) or float64 (local numerical verification "
                f"only), got {tensor.dtype}"
            )
        if tensor.dtype != param_dtype:
            raise TypeError(
                f"[torsion-head] `{name}` has dtype {tensor.dtype} but the "
                f"module parameters are {param_dtype}; cast the module for "
                "local numerical verification or fix the producer"
            )
        if tensor.ndim != 2 or tensor.size(1) != dim:
            raise ValueError(
                f"[torsion-head] `{name}` must be [B, {dim}], got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.numel() == 0:
            raise ValueError(
                f"[torsion-head] `{name}` must contain at least one row "
                "(fail-closed on empty batches)"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"[torsion-head] `{name}` contains non-finite values "
                "(fail-closed)"
            )
    if context.shape[0] != synthon_embedding.shape[0]:
        raise ValueError(
            f"[torsion-head] context has {context.shape[0]} rows but "
            f"synthon_embedding has {synthon_embedding.shape[0]}"
        )
    return context, synthon_embedding


@dataclass
class TorsionOutput:
    """The auditable result of one torsion-head forward pass.

    Fields:
        logits: raw (already mask-filled) bin logits ``[B, 36]``.
        log_probs: ``log P(phi_t = k | s_t, B_t)`` ``[B, 36]`` — the
            strictly categorical, dimensionless action law over the 36
            bins (the TB forward log-probability contribution of the
            torsion component).
        probs: ``exp(log_probs)`` ``[B, 36]``.
        mu: the head's auxiliary circular-mean prediction ``[B]`` in
            ``[-pi, pi)`` — supervision geometry only, never a TB
            probability.
        kappa: the head's auxiliary von Mises concentration ``[B] >= 0``
            — supervision geometry only, never a TB probability.
        required: per-row flag — whether the selected junction is
            rotatable (``resonance_lock is None``) and therefore carries
            a torsion action at all ``[B]`` bool.
        selected_bin: the supervised/sampled bin, 1-based like
            ``AssemblyAction.torsion_bin``; ``0`` marks a row whose
            junction is resonance-locked (no torsion action) ``[B]``
            int64.
        selected_log_prob: ``log P(selected_bin | s, B)`` for required
            rows, exactly ``0.0`` for locked rows (no action, no
            log-probability — the resonance lock is environment
            determinism, not a policy choice) ``[B]``.
        num_bins: the number of bins (36).
    """

    logits: torch.Tensor  # [B, 36]
    log_probs: torch.Tensor  # [B, 36]
    probs: torch.Tensor  # [B, 36]
    mu: torch.Tensor  # [B]
    kappa: torch.Tensor  # [B]
    required: torch.Tensor  # [B] bool
    selected_bin: torch.Tensor  # [B] int64
    selected_log_prob: torch.Tensor  # [B]

    @property
    def num_bins(self) -> int:
        """Number of discrete torsion bins (36)."""
        return int(self.log_probs.size(1))

    @property
    def num_graphs(self) -> int:
        """Number of scored rows ``B``."""
        return int(self.log_probs.size(0))


def torsion_circular_loss(
    mu: torch.Tensor, kappa: torch.Tensor, target_bins: torch.Tensor
) -> torch.Tensor:
    """The supervised circular loss over discrete torsion-bin targets.

    Implements #45 §12.1 ("a circular loss for torsion bins/angles
    consistent with the selected action formulation"): the supervised
    target is the *bin* (the action formulation), the loss evaluates the
    stable von Mises NLL (:func:`von_mises_loss`, #8 MATH-02) at that
    bin's exact centre angle, and the angular residual is measured
    *circularly* — bin 1 and bin 36 are 10 degrees apart, not 350.

    Parameters:
        mu: predicted angles ``[B]`` (any finite values; periodicity
            handled by the von Mises density).
        kappa: predicted concentrations ``[B] >= 0``.
        target_bins: int64 ``[B]`` 1-based bin indices in ``[1, 36]``;
            entries ``0`` mark rows without a torsion action
            (resonance-locked junctions) and are excluded from the loss.

    Fails closed when no row carries a torsion target (a mean over
    nothing is a masked NaN, not a loss).
    """
    if not isinstance(target_bins, torch.Tensor):
        raise TypeError(
            f"target_bins must be a torch.Tensor, got {type(target_bins).__name__}"
        )
    if target_bins.ndim != 1:
        raise ValueError(
            f"target_bins must be [B], got {tuple(target_bins.shape)}"
        )
    if target_bins.dtype not in (torch.int64, torch.int32):
        raise TypeError(
            f"target_bins must be an integer tensor, got {target_bins.dtype}"
        )
    keep = target_bins != 0
    vals = target_bins[keep]
    if vals.numel() == 0:
        raise ValueError(
            "torsion_circular_loss received no torsion targets (all rows "
            "resonance-locked or empty — fail-closed)"
        )
    if bool((vals < 0).any()) or bool((vals > TORSION_BINS).any()):
        raise ValueError(
            f"torsion bin targets must be 0 (no action) or in "
            f"[1, {TORSION_BINS}], got out-of-range values (fail-closed)"
        )
    centers = torch.tensor(
        torsion_bin_centers(), dtype=mu.dtype, device=mu.device
    )
    phi_target = centers[(vals - 1).long()]
    return von_mises_loss(mu[keep], kappa[keep], phi_target)


class TorsionHead(nn.Module):
    """The discrete 36-bin SO(2) torsion head (issue #24 MODEL-04).

    ``P(phi_t = k | s_t, B_t) = Softmax(z_torsion)[k]`` where ``z_torsion``
    is produced from the concatenation of the decoder decision context
    ``s_t`` (``[B, context_dim]``) and the tied catalog embedding of the
    selected synthon ``B_t`` (``[B, synthon_dim]``) — the conditioning the
    issue pins — through a small SiLU trunk.

    Auxiliary supervision outputs (never TB probabilities):

    * ``mu`` — the angle of a 2D vector routed through
      :func:`safe_unit_vector` (bounded gradients everywhere, #7);
    * ``kappa`` — ``softplus`` of a raw scalar (non-negative by smooth
      parameterization).

    Args:
        context_dim: width of the consumed decision context (production:
        256, :data:`TORSION_HEAD_CONTEXT_DIM`).
        synthon_dim: width of the consumed synthon embedding (production:
        64, :data:`TORSION_HEAD_SYNTON_DIM`).
        hidden_dim: trunk width (production: 128).
        num_bins: number of discrete bins (production: 36; must match the
            engine's ``TORSION_BINS``).
    """

    def __init__(
        self,
        context_dim: int = TORSION_HEAD_CONTEXT_DIM,
        synthon_dim: int = TORSION_HEAD_SYNTON_DIM,
        hidden_dim: int = TORSION_HEAD_HIDDEN_DIM,
        num_bins: int = TORSION_BINS,
    ) -> None:
        super().__init__()
        context_dim = int(context_dim)
        synthon_dim = int(synthon_dim)
        hidden_dim = int(hidden_dim)
        num_bins = int(num_bins)
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        if synthon_dim < 1:
            raise ValueError(f"synthon_dim must be >= 1, got {synthon_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        self.context_dim = context_dim
        self.synthon_dim = synthon_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins

        self.trunk = nn.Sequential(
            nn.Linear(context_dim + synthon_dim, hidden_dim),
            nn.SiLU(),
        )
        self.bin_logits = nn.Linear(hidden_dim, num_bins)
        self.mu_xy = nn.Linear(hidden_dim, 2)
        self.kappa_raw = nn.Linear(hidden_dim, 1)

    def extra_repr(self) -> str:
        return (
            f"context_dim={self.context_dim}, "
            f"synthon_dim={self.synthon_dim}, "
            f"hidden_dim={self.hidden_dim}, num_bins={self.num_bins}"
        )

    # -- audit ---------------------------------------------------------------

    def audit(self) -> dict:
        """Parameter/dimension audit (constitution #45 §7.3)."""
        parameters = list(self.parameters())
        return {
            "module": type(self).__name__,
            "context_dim": self.context_dim,
            "synthon_dim": self.synthon_dim,
            "hidden_dim": self.hidden_dim,
            "num_bins": self.num_bins,
            "num_parameters": sum(p.numel() for p in parameters),
            "num_trainable_parameters": sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            "all_parameters_trainable": all(p.requires_grad for p in parameters),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "auxiliary_von_mises_outputs": ["mu", "kappa"],
            "tb_probability_measure": "categorical_over_bins",
            "deterministic_in_eval": True,
        }

    # -- forward ----------------------------------------------------------------

    def forward(
        self,
        context: torch.Tensor,
        synthon_embedding: torch.Tensor,
        *,
        required: Optional[torch.Tensor] = None,
        selected_bin: Optional[torch.Tensor] = None,
        bin_mask: Optional[torch.Tensor] = None,
    ) -> TorsionOutput:
        """Evaluate the categorical torsion law (and auxiliary geometry).

        Parameters:
            context: ``[B, context_dim]`` decision context ``s_t``.
            synthon_embedding: ``[B, synthon_dim]`` tied catalog embedding
                of the selected synthon ``B_t``.
            required: ``[B]`` bool — junction rotatable (carries a torsion
                action).  Default: all rows required.
            selected_bin: ``[B]`` int64 1-based supervised/sampled bin
                (``0`` = not required / no action).  Default: ``0``
                everywhere (distribution-only pass).
            bin_mask: optional ``[B, num_bins]`` bool legality mask
                (``True`` = illegal).  Default: all bins legal.

        Returns a :class:`TorsionOutput`.
        """
        context, synthon_embedding = _validate_head_inputs(
            context,
            synthon_embedding,
            context_dim=self.context_dim,
            synthon_dim=self.synthon_dim,
            param_dtype=next(self.parameters()).dtype,
        )
        num_graphs = context.size(0)

        if required is None:
            required = torch.ones(
                num_graphs, dtype=torch.bool, device=context.device
            )
        else:
            required = self._validate_flag(required, num_graphs, context.device)

        if selected_bin is None:
            # Distribution-only pass: no supervision to check.
            selected_bin = torch.zeros(
                num_graphs, dtype=torch.int64, device=context.device
            )
            bins_supplied = False
        else:
            selected_bin = self._validate_bins(selected_bin, num_graphs, context.device)
            bins_supplied = True

        if bin_mask is not None:
            if (
                not isinstance(bin_mask, torch.Tensor)
                or bin_mask.dtype != torch.bool
                or bin_mask.shape != (num_graphs, self.num_bins)
            ):
                raise ValueError(
                    f"[torsion-head] bin_mask must be a boolean "
                    f"[{num_graphs}, {self.num_bins}] tensor"
                )

        h = self.trunk(torch.cat((context, synthon_embedding), dim=-1))
        logits = self.bin_logits(h)  # [B, 36]
        if bin_mask is not None:
            logits = apply_logit_mask(logits, bin_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Auxiliary circular prediction: mu is the angle of a 2D vector
        # routed through the non-singular safe_unit_vector primitive.
        xy = self.mu_xy(h)  # [B, 2]
        unit, _dist = safe_unit_vector(xy, dim=-1)
        mu = torch.atan2(unit[..., 1], unit[..., 0])  # [B] in [-pi, pi]
        kappa = F.softplus(self.kappa_raw(h)).squeeze(-1)  # [B] >= 0

        selected_log_prob = torch.zeros(
            num_graphs, dtype=log_probs.dtype, device=log_probs.device
        )
        has_selection = required & (selected_bin != 0)
        if bool(has_selection.any()):
            rows = torch.nonzero(has_selection, as_tuple=False).flatten()
            selected_log_prob = selected_log_prob.index_copy(
                0,
                rows,
                log_probs.gather(
                    1, (selected_bin[rows] - 1).unsqueeze(-1)
                ).squeeze(-1),
            )
        # Cross-field consistency (fail-closed): a required row MUST carry
        # a bin; a locked row's bin (if any) is a vacuous environment
        # default and contributes nothing.  Enforced only when supervision
        # was supplied — a distribution-only pass carries no bins to check.
        if bins_supplied and bool((required & (selected_bin == 0)).any()):
            bad = torch.nonzero(
                required & (selected_bin == 0), as_tuple=False
            ).flatten().tolist()
            raise ValueError(
                f"[torsion-head] rows {bad} require a torsion action but "
                f"carry selected_bin = 0 (fail-closed: supervised "
                "trajectories must supply the torsion bin)"
            )

        return TorsionOutput(
            logits=logits,
            log_probs=log_probs,
            probs=probs,
            mu=mu,
            kappa=kappa,
            required=required,
            selected_bin=selected_bin,
            selected_log_prob=selected_log_prob,
        )

    # -- the supervised circular objective -----------------------------------

    def circular_loss(
        self, output: TorsionOutput, target_bins: torch.Tensor
    ) -> torch.Tensor:
        """Circular von Mises loss at the supervised bin centres (#8+#24).

        Delegates to :func:`torsion_circular_loss` on the output's
        auxiliary ``(mu, kappa)`` — the *training* objective, distinct
        from the categorical TB log-probability.
        """
        if output.mu.shape != target_bins.shape:
            raise ValueError(
                f"target_bins must be [{output.num_graphs}], got "
                f"{tuple(target_bins.shape)}"
            )
        return torsion_circular_loss(output.mu, output.kappa, target_bins)

    # -- validation helpers ----------------------------------------------------

    def _validate_flag(
        self, flag: torch.Tensor, num_graphs: int, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(flag, torch.Tensor) or flag.dtype != torch.bool:
            raise ValueError(
                "[torsion-head] `required` must be a boolean tensor"
            )
        if flag.ndim != 1 or flag.shape[0] != num_graphs:
            raise ValueError(
                f"[torsion-head] `required` must be [{num_graphs}], got "
                f"{tuple(flag.shape)}"
            )
        return flag.to(device)

    def _validate_bins(
        self, bins: torch.Tensor, num_graphs: int, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(bins, torch.Tensor) or bins.dtype not in (
            torch.int64,
            torch.int32,
        ):
            raise TypeError(
                f"[torsion-head] selected_bin must be an integer tensor, "
                f"got {bins.dtype if isinstance(bins, torch.Tensor) else type(bins).__name__}"
            )
        if bins.ndim != 1 or bins.shape[0] != num_graphs:
            raise ValueError(
                f"[torsion-head] selected_bin must be [{num_graphs}], got "
                f"{tuple(bins.shape)}"
            )
        bins = bins.long().to(device)
        if bool((bins < 0).any()) or bool((bins > self.num_bins).any()):
            raise ValueError(
                f"[torsion-head] selected_bin entries must be in "
                f"[0, {self.num_bins}] (0 = no torsion action), got "
                f"out-of-range values (fail-closed)"
            )
        return bins
