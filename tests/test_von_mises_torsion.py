"""Von Mises circular-torsion mathematics — issue #8 MATH-02 acceptance.

Pins the contracts of :mod:`syntree.models.torsion_head`:

* **forward accuracy** (issue #8 acceptance 1): ``log I_0`` matches the
  trusted ``scipy.special.i0e`` reference within absolute tolerance
  ``< 1e-5`` across ``kappa in [0, 100]`` — tested both in log space
  (``log I_0`` vs ``kappa + log(i0e(kappa))``) and in the scaled space
  the acceptance criterion names literally (``exp(log I_0 - kappa)`` vs
  ``i0e(kappa)``);
* **backward boundedness** (issue #8 acceptance 2): autograd gradients of
  ``log I_0`` are finite, non-negative, and strictly bounded in ``[0, 1)``
  at every test point, including ``kappa = 0`` (exactly zero) and very
  large concentrations;
* **approximation honesty** (#45 §8.2): the Amos backward is compared
  against the trusted reference ratio ``I_1/I_0 = i1e/i0e`` and its
  maximum deviation is *measured and pinned* (an approximation, not an
  exact identity);
* **branch-seam safety** (the legacy "Singularity 2" NaN leak): finite
  values and finite gradients on both sides of the ``kappa = 12`` seam;
* **von Mises objective**: the log-density normalises (numeric integral
  over ``[-pi, pi)``), matches ``scipy.stats.vonmises``, peaks at the
  mean, and has finite gradients in every argument;
* **fail-closed domain** (constitution §8): NaN/Inf/negative/non-floating
  inputs raise instead of being clamped.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import special, stats

from syntree.models.torsion_head import (
    AMOS_RATIO_CONSTANT,
    LOG_2PI,
    VON_MISES_SERIES_THRESHOLD,
    LogBesselI0,
    amos_ratio,
    log_bessel_i0,
    von_mises_log_prob,
    von_mises_loss,
    von_mises_nll,
)

# ---------------------------------------------------------------------------
# Trusted numerical references (scipy)
# ---------------------------------------------------------------------------


def _ref_log_i0(kappa: float) -> float:
    """log I_0(kappa) from scipy's exponentially-scaled i0e (stable for all k)."""
    return float(kappa + math.log(special.i0e(kappa)))


def _ref_ratio(kappa: float) -> float:
    """I_1(kappa)/I_0(kappa) from scipy (the exponential scalings cancel)."""
    return float(special.i1e(kappa) / special.i0e(kappa))


def _test_kappas() -> np.ndarray:
    """Dense + targeted grid over the pinned range [0, 100]."""
    dense = np.linspace(0.0, 100.0, 2001)
    targeted = np.array(
        [
            0.0,
            1e-9,
            1e-6,
            1e-3,
            0.01,
            0.1,
            0.5,
            1.0,
            2.0,
            3.0,
            5.0,
            8.0,
            11.999,
            12.0,
            12.001,
            15.0,
            20.0,
            30.0,
            50.0,
            80.0,
            100.0,
        ]
    )
    rng = np.random.default_rng(20261102)
    random_pts = rng.uniform(0.0, 100.0, size=200)
    return np.unique(np.concatenate([dense, targeted, random_pts]))


# ---------------------------------------------------------------------------
# 1. Forward values — issue #8 acceptance criterion 1
# ---------------------------------------------------------------------------


class TestLogBesselI0Forward:
    def test_forward_matches_i0e_reference_log_space(self):
        kappas = _test_kappas()
        got = log_bessel_i0(torch.tensor(kappas, dtype=torch.float64))
        ref = torch.tensor([_ref_log_i0(k) for k in kappas], dtype=torch.float64)
        worst = float((got - ref).abs().max())
        assert worst < 1e-5, f"log-space i0e reference error {worst}"

    def test_forward_matches_i0e_reference_scaled_space(self):
        # The literal acceptance form: forward values (mapped through the
        # same exponential scaling i0e uses) vs scipy.special.i0e itself.
        kappas = _test_kappas()
        got = torch.exp(log_bessel_i0(torch.tensor(kappas, dtype=torch.float64)) - torch.tensor(kappas, dtype=torch.float64))
        ref = torch.tensor(special.i0e(kappas), dtype=torch.float64)
        worst = float((got - ref).abs().max())
        assert worst < 1e-5, f"scaled-space i0e reference error {worst}"

    def test_forward_accuracy_in_fp32_output(self):
        # fp64 internals meet the pinned tolerance even when the tensor
        # dtype is fp32 (the production precision, #45 §5.1).  The output
        # itself is rounded to fp32 (~6e-6 absolute at log I_0(100) ~ 96.7),
        # so the comparison tolerance carries that representation floor.
        kappas = _test_kappas()
        got = log_bessel_i0(torch.tensor(kappas, dtype=torch.float32))
        ref = torch.tensor([_ref_log_i0(k) for k in kappas], dtype=torch.float32)
        worst = float((got.double() - ref.double()).abs().max())
        assert worst < 2e-5, f"fp32-output i0e reference error {worst}"

    def test_zero_kappa_is_exact_zero(self):
        assert float(log_bessel_i0(torch.zeros(3, dtype=torch.float64))[1]) == 0.0

    def test_small_kappa_quadratic_behaviour(self):
        # log I_0(k) = k^2/4 - k^4/64 + O(k^6) as k -> 0: the relative
        # deviation of the leading term is k^2/16 (6.25e-6 at k = 0.01).
        for k in (1e-4, 1e-3, 1e-2):
            got = float(log_bessel_i0(torch.tensor([k], dtype=torch.float64)))
            assert got == pytest.approx(k * k / 4.0, rel=1e-5, abs=1e-15)

    def test_large_kappa_asymptotic_behaviour(self):
        # Deep in the asymptotic regime the error falls like O(k^-4):
        # ~1.8e-8 at kappa = 50, ~1.1e-9 at kappa = 100 (the truncation
        # of the pinned 3-term asymptotic series).
        for k in (50.0, 100.0):
            got = float(log_bessel_i0(torch.tensor([k], dtype=torch.float64)))
            assert got == pytest.approx(_ref_log_i0(k), abs=1e-7)

    def test_branch_seam_is_within_tolerance(self):
        seam = VON_MISES_SERIES_THRESHOLD
        below = float(log_bessel_i0(torch.tensor([seam - 1e-9], dtype=torch.float64)))
        above = float(log_bessel_i0(torch.tensor([seam + 1e-9], dtype=torch.float64)))
        # The two branches disagree only by the asymptotic truncation error
        # at the seam (~5.4e-6 by design of the pinned threshold).
        assert abs(below - above) < 1e-5

    def test_dtype_preserved(self):
        for dtype in (torch.float32, torch.float64):
            kappa = torch.tensor([0.5, 12.0, 40.0], dtype=dtype)
            assert log_bessel_i0(kappa).dtype == dtype

    def test_deterministic_and_pure(self):
        kappa = torch.tensor([0.3, 7.5, 12.0, 65.0], dtype=torch.float64)
        first = log_bessel_i0(kappa)
        second = log_bessel_i0(kappa)
        assert torch.equal(first, second)

    def test_works_under_no_grad(self):
        kappa = torch.tensor([1.0, 20.0], dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            out = log_bessel_i0(kappa)
        assert out.requires_grad is False
        assert float(out[0]) == pytest.approx(_ref_log_i0(1.0), abs=1e-12)

    def test_batched_shapes_preserved(self):
        kappa = torch.rand(3, 5, dtype=torch.float64) * 100.0
        assert log_bessel_i0(kappa).shape == (3, 5)

    def test_fail_closed_on_non_finite(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError):
                log_bessel_i0(torch.tensor([bad], dtype=torch.float64))

    def test_fail_closed_on_negative_kappa(self):
        with pytest.raises(ValueError):
            log_bessel_i0(torch.tensor([-1e-9], dtype=torch.float64))

    def test_fail_closed_on_non_floating_dtype(self):
        with pytest.raises(TypeError):
            log_bessel_i0(torch.tensor([0, 1, 12]))

    def test_fail_closed_on_non_tensor(self):
        with pytest.raises(TypeError):
            log_bessel_i0(3.0)


# ---------------------------------------------------------------------------
# 2. Backward gradient — issue #8 acceptance criterion 2
# ---------------------------------------------------------------------------


class TestAmosBackward:
    def _grad_at(self, kappa: float) -> float:
        k = torch.tensor([kappa], dtype=torch.float64, requires_grad=True)
        out = log_bessel_i0(k)
        out.backward()
        return float(k.grad)

    def test_gradient_finite_nonnegative_below_one(self):
        for kappa in _test_kappas():
            grad = self._grad_at(float(kappa))
            assert math.isfinite(grad), f"non-finite gradient at kappa={kappa}"
            assert 0.0 <= grad < 1.0, f"gradient {grad} outside [0, 1) at kappa={kappa}"

    def test_gradient_at_zero_kappa_is_exactly_zero(self):
        assert self._grad_at(0.0) == 0.0

    def test_gradient_at_very_large_kappa_stays_bounded(self):
        # Far beyond the pinned range: the Amos ratio still tends to 1 from
        # below and never overflows.
        for kappa in (1e3, 1e5):
            grad = self._grad_at(kappa)
            assert 0.0 <= grad < 1.0
            assert math.isfinite(grad)

    def test_gradient_monotone_in_kappa(self):
        kappas = np.linspace(0.01, 60.0, 400)
        grads = [self._grad_at(float(k)) for k in kappas]
        assert all(b >= a for a, b in zip(grads, grads[1:]))

    def test_gradient_bounded_for_fp32_production_dtype(self):
        kappa = torch.rand(512, dtype=torch.float32) * 100.0
        kappa.requires_grad_(True)
        log_bessel_i0(kappa).sum().backward()
        grad = kappa.grad
        assert bool(torch.isfinite(grad).all())
        assert bool((grad >= 0).all())
        assert bool((grad < 1.0).all())

    def test_gradient_matches_trusted_reference_within_pinned_error(self):
        # #45 §8.2: the Amos form is an approximation — measure its error
        # against scipy's I_1/I_0 over the relevant range and pin the bound.
        kappas = _test_kappas()
        grads = np.array([self._grad_at(float(k)) for k in kappas])
        refs = np.array([_ref_ratio(float(k)) for k in kappas])
        worst = float(np.abs(grads - refs).max())
        # Measured maximum: ~3.2e-2 near kappa ~ 2 (mid-range worst case).
        assert worst < 4.0e-2, f"Amos ratio deviation from I1/I0: {worst}"

    def test_gradient_asymptotic_error_small_for_large_kappa(self):
        for kappa in (30.0, 60.0, 100.0):
            grad = self._grad_at(kappa)
            assert grad == pytest.approx(_ref_ratio(kappa), abs=1e-2)

    def test_gradient_small_kappa_half_kappa_behaviour(self):
        # I_1/I_0 -> kappa/2 as kappa -> 0; the Amos form reproduces the
        # same leading behaviour (denominator -> 0.5 + 1.5 = 2).
        for kappa in (1e-4, 1e-3, 1e-2):
            assert self._grad_at(kappa) == pytest.approx(kappa / 2.0, rel=1e-3)

    def test_no_nan_at_branch_seam_gradients(self):
        # The legacy failure ("Singularity 2") leaked NaN through the
        # inactive torch.where branch during backward.  The custom Function
        # has a single analytic backward: no seam, no leak.
        for kappa in (11.9, 11.999, 12.0, 12.001, 12.1):
            k = torch.tensor([kappa], dtype=torch.float64, requires_grad=True)
            out = log_bessel_i0(k)
            out.backward()
            assert math.isfinite(float(k.grad))

    def test_gradient_shapes_and_batching(self):
        kappa = torch.rand(4, 7, dtype=torch.float64, requires_grad=True)
        out = log_bessel_i0(kappa)
        grad_out = torch.rand_like(out)
        out.backward(grad_out)
        assert kappa.grad.shape == (4, 7)
        assert bool(torch.isfinite(kappa.grad).all())

    def test_gradient_of_sum_decomposes_per_element(self):
        kappas = torch.tensor([0.5, 3.0, 25.0], dtype=torch.float64)
        k = kappas.clone().requires_grad_(True)
        log_bessel_i0(k).sum().backward()
        singles = []
        for value in kappas.tolist():
            kk = torch.tensor([value], dtype=torch.float64, requires_grad=True)
            log_bessel_i0(kk).backward()
            singles.append(float(kk.grad))
        assert torch.allclose(k.grad, torch.tensor(singles, dtype=torch.float64))

    def test_amos_ratio_helper_matches_backward_value(self):
        kappas = torch.tensor([0.0, 0.7, 2.5, 12.0, 45.0], dtype=torch.float64)
        k = kappas.clone().requires_grad_(True)
        log_bessel_i0(k).sum().backward()
        assert torch.allclose(k.grad, amos_ratio(kappas), atol=0.0, rtol=0.0)

    def test_amos_ratio_helper_is_bounded(self):
        kappas = torch.linspace(0.0, 1e4, 2001, dtype=torch.float64)
        ratio = amos_ratio(kappas)
        assert bool((ratio >= 0).all()) and bool((ratio < 1).all())
        assert float(ratio[0]) == 0.0

    def test_amos_ratio_constant_is_pinned(self):
        assert AMOS_RATIO_CONSTANT == 2.25
        assert VON_MISES_SERIES_THRESHOLD == 12.0
        assert LOG_2PI == math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# 3. The von Mises objective (the "von Mises Loss" of the issue title)
# ---------------------------------------------------------------------------


class TestVonMisesObjective:
    def test_log_prob_matches_scipy_vonmises(self):
        rng = np.random.default_rng(8)
        for kappa in (0.1, 1.0, 5.0, 20.0, 80.0):
            for mu in (0.0, -1.3, 2.7):
                phis = rng.uniform(-math.pi, math.pi, size=17)
                got = von_mises_log_prob(
                    torch.tensor([mu] * 17, dtype=torch.float64),
                    torch.tensor([kappa] * 17, dtype=torch.float64),
                    torch.tensor(phis, dtype=torch.float64),
                ).numpy()
                ref = stats.vonmises.logpdf(phis, kappa, loc=mu)
                # scipy's own logpdf uses a high-precision log I_0; our
                # pinned branches carry < 1e-5 absolute error.
                assert np.abs(got - ref).max() < 1e-4, f"kappa={kappa}, mu={mu}"

    def test_log_prob_normalizes_over_the_circle(self):
        # Numeric integral of p over [-pi, pi) is 1 (the normalisation the
        # stable log I_0 must deliver).  Explicit trapezoid (no deprecated
        # trapz helpers).
        grid = np.linspace(-math.pi, math.pi, 40001)
        grid_t = torch.tensor(grid, dtype=torch.float64)
        for kappa in (0.5, 2.0, 10.0):
            for mu in (0.0, 1.1):
                log_p = von_mises_log_prob(
                    torch.tensor([mu] * grid.size, dtype=torch.float64),
                    torch.tensor([kappa] * grid.size, dtype=torch.float64),
                    grid_t,
                )
                p = log_p.exp()
                integral = float(((p[1:] + p[:-1]) * 0.5 * (grid_t[1:] - grid_t[:-1])).sum())
                assert integral == pytest.approx(1.0, rel=2e-3), f"kappa={kappa}, mu={mu}"

    def test_log_prob_peaks_at_the_mean(self):
        kappa = torch.tensor([3.0], dtype=torch.float64)
        mu = torch.tensor([0.4], dtype=torch.float64)
        at_mean = von_mises_log_prob(mu, kappa, mu)
        assert float(at_mean) > float(von_mises_log_prob(mu, kappa, mu + 0.2))
        assert float(at_mean) > float(von_mises_log_prob(mu, kappa, mu - 0.7))
        # Value at the mean: kappa - log(2 pi) - log I_0(kappa).
        assert float(at_mean) == pytest.approx(
            3.0 - LOG_2PI - _ref_log_i0(3.0), abs=1e-5
        )

    def test_uniform_distribution_at_zero_kappa(self):
        # kappa = 0: p = 1/(2 pi) exactly.
        mu = torch.tensor([0.0], dtype=torch.float64)
        kappa = torch.tensor([0.0], dtype=torch.float64)
        phi = torch.tensor([1.234], dtype=torch.float64)
        assert float(von_mises_log_prob(mu, kappa, phi)) == pytest.approx(-LOG_2PI, abs=1e-12)

    def test_nll_is_negated_log_prob(self):
        mu = torch.tensor([0.2], dtype=torch.float64)
        kappa = torch.tensor([4.0], dtype=torch.float64)
        phi = torch.tensor([-0.9], dtype=torch.float64)
        assert float(von_mises_nll(mu, kappa, phi)) == pytest.approx(
            -float(von_mises_log_prob(mu, kappa, phi)), rel=1e-15
        )

    def test_loss_minimum_at_the_mean(self):
        mu = torch.tensor([-0.5], dtype=torch.float64)
        kappa = torch.tensor([2.0], dtype=torch.float64)
        at = float(von_mises_loss(mu, kappa, mu))
        off = float(von_mises_loss(mu, kappa, mu + torch.tensor([0.5], dtype=torch.float64)))
        assert at < off

    def test_loss_gradients_finite_in_all_arguments(self):
        mu = torch.tensor([0.3, -1.2], dtype=torch.float64, requires_grad=True)
        kappa = torch.tensor([0.7, 9.0], dtype=torch.float64, requires_grad=True)
        phi = torch.tensor([0.1, 2.5], dtype=torch.float64)
        loss = von_mises_loss(mu, kappa, phi)
        loss.backward()
        assert mu.grad is not None and kappa.grad is not None
        assert bool(torch.isfinite(mu.grad).all())
        assert bool(torch.isfinite(kappa.grad).all())

    def test_loss_gradient_against_finite_difference(self):
        # The analytic kappa-gradient (Amos ratio) must track the true
        # derivative of the *implemented* forward within the documented
        # approximation error (the forward is the exact log I_0; the
        # backward is the Amos approximation — the pinned design).
        mu = torch.tensor([0.4], dtype=torch.float64)
        kappa = torch.tensor([1.7], dtype=torch.float64, requires_grad=True)
        phi = torch.tensor([0.1], dtype=torch.float64)
        von_mises_nll(mu, kappa, phi).backward()
        analytic = float(kappa.grad)
        h = 1e-6
        fd = (
            float(von_mises_nll(mu, kappa.detach() + h, phi))
            - float(von_mises_nll(mu, kappa.detach() - h, phi))
        ) / (2 * h)
        # d nll/d kappa = A(kappa) - cos(phi - mu); the finite difference
        # uses the exact log I_0, so the gap is the Amos error at kappa=1.7.
        assert analytic == pytest.approx(fd, abs=5e-2)

    def test_broadcasting_across_batch(self):
        mu = torch.zeros(4, dtype=torch.float64)
        kappa = torch.full((4,), 2.0, dtype=torch.float64)
        phi = torch.linspace(-math.pi, math.pi, 4, dtype=torch.float64)
        out = von_mises_log_prob(mu, kappa, phi)
        assert out.shape == (4,)

    def test_fail_closed_on_bad_inputs(self):
        good = torch.tensor([0.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            von_mises_log_prob(good, torch.tensor([-1.0], dtype=torch.float64), good)
        with pytest.raises(ValueError):
            von_mises_log_prob(
                torch.tensor([float("nan")], dtype=torch.float64), good + 1.0, good
            )
        with pytest.raises(TypeError):
            von_mises_log_prob(good, torch.tensor([1], dtype=torch.int64), good)
        with pytest.raises(ValueError):
            # non-broadcastable shapes
            von_mises_log_prob(
                torch.zeros(3, dtype=torch.float64),
                torch.zeros(4, dtype=torch.float64) + 1.0,
                torch.zeros(3, dtype=torch.float64),
            )

    def test_fail_closed_on_empty_loss(self):
        mu = torch.zeros(0, dtype=torch.float64)
        kappa = torch.zeros(0, dtype=torch.float64) + 1.0
        with pytest.raises(ValueError):
            von_mises_loss(mu, kappa, mu)

    def test_objective_is_deterministic(self):
        mu = torch.tensor([0.9], dtype=torch.float64)
        kappa = torch.tensor([12.5], dtype=torch.float64)
        phi = torch.tensor([-2.2], dtype=torch.float64)
        first = von_mises_nll(mu, kappa, phi)
        second = von_mises_nll(mu, kappa, phi)
        assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# 4. Integration with the V2 numerical policy
# ---------------------------------------------------------------------------


class TestNumericalPolicyIntegration:
    def test_no_autograd_through_unrolled_series(self):
        # The pre-V2 failure was differentiating a 20-term power series.
        # The custom Function must NOT record the branch arithmetic: check
        # that the backward is a single closed-form evaluation by verifying
        # gradient correctness on a tensor mixing both branches at once.
        kappas = torch.tensor([0.5, 5.0, 12.0, 40.0, 99.0], dtype=torch.float64)
        k = kappas.clone().requires_grad_(True)
        out = log_bessel_i0(k)
        probe = torch.tensor([1.0, -2.0, 0.5, 3.0, -0.25], dtype=torch.float64)
        (out * probe).sum().backward()
        expected = amos_ratio(kappas) * probe
        assert torch.allclose(k.grad, expected, rtol=0.0, atol=0.0)

    def test_module_contains_no_clamp_machinery(self):
        # Constitution §8 / D-V2-005: stability by construction, not clamps.
        # AST-level check (docstrings may legitimately *mention* the
        # removed legacy clamps).
        import ast
        import pathlib

        import syntree.models.torsion_head as torsion_head_module

        source = pathlib.Path(torsion_head_module.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in ("clamp", "nan_to_num", "clip"):
                    raise AssertionError(
                        f"banned numerical band-aid call: {name} at line {node.lineno}"
                    )

    def test_saved_tensor_roundtrip_through_function(self):
        kappa = torch.tensor([7.0], dtype=torch.float64, requires_grad=True)
        out = LogBesselI0.apply(kappa)
        out.backward()
        # The saved tensor is the original input (validation guarantees
        # kappa >= 0, so the backward trust domain holds).
        assert float(out.detach()) == pytest.approx(_ref_log_i0(7.0), abs=1e-12)
        assert float(kappa.grad) == pytest.approx(_ref_ratio(7.0), abs=4e-2)
