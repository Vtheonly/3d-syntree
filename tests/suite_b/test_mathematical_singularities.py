"""Suite B / 1 — mathematical singularity invariants (issue #33 TEST-02).

The three issue-pinned singularity batteries (constitution #45 §8):

1. **Coincident atoms** — ``d_ij = 0.0`` must produce *finite* features
   and *exactly zero* gradients through :func:`safe_norm` (the
   non-singular PaiNN norm, #7 MATH-01): no 0/0, no NaN, no clamp.
2. **von Mises backward** — ``log I_0(kappa)`` gradients evaluate
   cleanly to the bounded Amos' ratio ``A(kappa) in [0, 1)`` for
   ``kappa in [0, 100]`` (#8 MATH-02), matching the public
   :func:`amos_ratio` approximation the backward actually uses.
3. **Continuous cosine cutoff** — the envelope drops to *exactly zero*
   at ``r_cut`` (bitwise) and its integrated radial basis vanishes just
   beyond the boundary (#9 MATH-03, D-V2-011).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.models.bipartite import GaussianRadialBasis  # noqa: E402
from syntree.models.cutoff import continuous_cosine_envelope  # noqa: E402
from syntree.models.equivariant import SAFE_NORM_EPS, safe_norm, safe_unit_vector  # noqa: E402
from syntree.models.torsion_head import amos_ratio, log_bessel_i0  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Coincident atoms: finite features, exactly-zero gradients
# ---------------------------------------------------------------------------


class TestCoincidentAtoms:
    def test_zero_distance_value_is_finite_and_zero(self):
        coincident = torch.zeros(64, 3, dtype=torch.float64, requires_grad=True)
        norms = safe_norm(coincident)
        assert bool(torch.isfinite(norms).all())
        assert torch.equal(norms, torch.zeros_like(norms))

    def test_zero_distance_gradient_is_exactly_zero(self):
        coincident = torch.zeros(64, 3, dtype=torch.float64, requires_grad=True)
        safe_norm(coincident).sum().backward()
        assert coincident.grad is not None
        assert torch.equal(coincident.grad, torch.zeros_like(coincident.grad))

    def test_tiny_distances_stay_finite_with_bounded_gradients(self):
        for scale in (1e-12, 1e-8, 1e-5, 1e-3):
            vec = torch.full((32, 3), scale, dtype=torch.float64, requires_grad=True)
            norms = safe_norm(vec)
            assert bool(torch.isfinite(norms).all())
            safe_norm(vec).sum().backward()
            assert bool(torch.isfinite(vec.grad).all())
            assert float(vec.grad.abs().max()) <= 1.0 + 1e-12  # bounded by 1

    def test_safe_unit_vector_at_the_zero_vector(self):
        zero = torch.zeros(16, 3, dtype=torch.float64, requires_grad=True)
        unit, dist = safe_unit_vector(zero)
        assert bool(torch.isfinite(unit).all()) and bool(torch.isfinite(dist).all())
        assert torch.equal(unit, torch.zeros_like(unit))  # defined direction
        assert float(unit.norm(dim=-1).max()) < 1.0  # strictly bounded
        assert float(dist.max()) == 0.0
        unit.sum().backward()
        assert bool(torch.isfinite(zero.grad).all())

    def test_smooth_norm_error_is_bounded_by_eps_at_physical_distances(self):
        # |sqrt(d^2+eps^2) - eps - d| <= eps by construction; at physical
        # bond scales (~1.5 A vectors) the relative error is ~eps/d ~ 4e-6
        physical = torch.full((8, 3), 1.5, dtype=torch.float64)
        smoothed = safe_norm(physical, eps=SAFE_NORM_EPS)
        exact = torch.linalg.norm(physical, dim=-1)
        error = (smoothed - exact).abs()
        assert float(error.max()) <= SAFE_NORM_EPS
        assert float((error / exact).max()) < 1e-5


# ---------------------------------------------------------------------------
# 2. von Mises backward: A(kappa) in [0, 1) over kappa in [0, 100]
# ---------------------------------------------------------------------------


def _kappa_grid() -> torch.Tensor:
    values = np.concatenate(
        [
            np.linspace(0.0, 100.0, 201),
            np.array([1e-9, 1e-6, 1e-3, 0.01, 11.999, 12.0, 12.001, 50.0, 99.999]),
        ]
    )
    return torch.tensor(np.unique(values), dtype=torch.float64)


class TestVonMisesBackward:
    def test_backward_gradient_is_the_amos_ratio(self):
        kappa = _kappa_grid().requires_grad_(True)
        log_bessel_i0(kappa).sum().backward()
        assert kappa.grad is not None
        torch.testing.assert_close(kappa.grad, amos_ratio(_kappa_grid()))

    def test_ratio_is_bounded_in_unit_interval(self):
        ratio = amos_ratio(_kappa_grid())
        assert bool(torch.isfinite(ratio).all())
        assert float(ratio.min()) >= 0.0
        assert float(ratio.max()) < 1.0  # strictly below 1 at every finite kappa

    def test_ratio_extremes_and_monotonicity(self):
        assert float(amos_ratio(torch.zeros(1, dtype=torch.float64))[0]) == 0.0
        assert float(amos_ratio(torch.tensor([100.0], dtype=torch.float64))[0]) < 1.0
        grid = _kappa_grid()
        ratio = amos_ratio(grid)
        assert bool((torch.diff(ratio) > 0).all())  # strictly increasing

    def test_forward_and_backward_are_finite_over_the_range(self):
        kappa = _kappa_grid().requires_grad_(True)
        values = log_bessel_i0(kappa)
        assert bool(torch.isfinite(values).all())
        values.sum().backward()
        assert bool(torch.isfinite(kappa.grad).all())

    def test_large_kappa_asymptotics_stay_bounded(self):
        huge = torch.tensor([1e3, 1e4, 1e5], dtype=torch.float64)
        ratio = amos_ratio(huge)
        assert bool(torch.isfinite(ratio).all())
        assert float(ratio.max()) < 1.0


# ---------------------------------------------------------------------------
# 3. Continuous cosine cutoff at the boundary
# ---------------------------------------------------------------------------


class TestCosineCutoffBoundary:
    def test_envelope_is_bitwise_zero_at_the_cutoff(self):
        cutoff = 5.0
        at_boundary = torch.tensor([cutoff], dtype=torch.float64)
        value = continuous_cosine_envelope(at_boundary, cutoff)
        assert torch.equal(value, torch.zeros_like(value))

    def test_envelope_is_zero_beyond_and_one_at_origin(self):
        cutoff = 5.0
        distances = torch.tensor([5.0 + 1e-4, 6.0, 100.0], dtype=torch.float64)
        values = continuous_cosine_envelope(distances, cutoff)
        assert torch.equal(values, torch.zeros_like(values))
        origin = continuous_cosine_envelope(torch.zeros(1, dtype=torch.float64), cutoff)
        assert torch.equal(origin, torch.ones_like(origin))

    def test_envelope_decays_continuously_into_the_boundary(self):
        cutoff = 5.0
        # approaching r_cut from inside: f -> 0 with a vanishing slope
        for delta in (1e-2, 1e-4, 1e-6, 1e-8):
            inside = torch.tensor([cutoff - delta], dtype=torch.float64)
            value = float(continuous_cosine_envelope(inside, cutoff)[0])
            assert 0.0 <= value < 0.5 * (math.pi * delta / cutoff) ** 2 + 1e-15

    def test_envelope_is_c1_across_an_interior_point(self):
        cutoff = 5.0
        center = 2.5
        h = 1e-6
        below = torch.tensor([center - h], dtype=torch.float64)
        above = torch.tensor([center + h], dtype=torch.float64)
        slope = float(
            (
                continuous_cosine_envelope(above, cutoff)
                - continuous_cosine_envelope(below, cutoff)
            )[0]
        ) / (2 * h)
        expected = -0.5 * math.pi / cutoff * math.sin(math.pi * center / cutoff)
        assert abs(slope - expected) < 1e-6

    def test_integrated_radial_basis_vanishes_past_the_boundary(self):
        basis = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        just_past = torch.tensor([5.0 + 1e-4], dtype=torch.float64)
        features = basis(just_past)
        assert torch.equal(features, torch.zeros_like(features))
        inside = torch.tensor([4.9], dtype=torch.float64)
        assert bool((basis(inside) > 0).all())
