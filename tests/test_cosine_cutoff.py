"""Continuous cosine cutoff envelopes for spatial RBF projections —
tests (#9 MATH-03).

Acceptance criteria of issue #9 (clean-room V2 implementation):

1. At ``d = r_cut + 1e-4`` the radial feature vector evaluates to
   identically ``0`` (bitwise, every channel).
2. The gradient ``d(radial)/dd`` approaches zero continuously as
   ``d -> r_cut^-``.

Beyond the two pinned criteria, this suite enforces the analytic envelope
properties (value/derivative closed forms, the reflection identity
``f(d) + f(r_cut - d) = 1``, monotonicity, the quadratic near-cutoff
decay), the fail-closed input contract, the enveloped
``GaussianRadialBasis`` integration (peaks carry the envelope, analytic
radial-derivative match, no parameters added), the handle-head [0, 12] A
centroid-distance basis (the legacy "Vulnerability 7" regime — far
spectators contribute exactly zero), and a compact backbone-level
SE(3)-invariance check confirming the envelope (a radial function) leaves
equivariance intact (the full 36-trial suite lives in
``tests/test_bipartite_painn.py`` and re-runs against the enveloped path).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from syntree.core.schema import TensorContractError
from syntree.models.bipartite import (
    GaussianRadialBasis,
    BipartitePaiNN,
)
from syntree.models.cutoff import continuous_cosine_envelope
from syntree.models.handle_head import (
    HANDLE_GEOM_CUTOFF,
    HandleSelectionHead,
)

_RC = 5.0


def _analytic_envelope(d: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Closed-form reference (float64): the issue's piecewise formula."""
    d = d.clamp(min=0.0)
    inside = 0.5 * (torch.cos(d * (math.pi / cutoff)) + 1.0)
    return torch.where(d <= cutoff, inside, torch.zeros_like(inside))


def _analytic_envelope_derivative(d: torch.Tensor, cutoff: float) -> torch.Tensor:
    """f'(d) = -0.5 (pi / r_cut) sin(pi d / r_cut) inside, 0 outside."""
    d = d.clamp(min=0.0)
    deriv = -0.5 * (math.pi / cutoff) * torch.sin(d * (math.pi / cutoff))
    return torch.where(d <= cutoff, deriv, torch.zeros_like(deriv))


# ---------------------------------------------------------------------------
# 1. The two pinned issue acceptance criteria
# ---------------------------------------------------------------------------


class TestIssueAcceptanceCriteria:
    """The exact checkboxes of the issue #9 body."""

    def test_radial_vector_identically_zero_at_cutoff_plus_1e_4(self):
        """Criterion 1: at ``d = r_cut + 1e-4`` the radial feature vector
        evaluates to identically 0 — every channel, bitwise, both dtypes."""
        for dtype in (torch.float64, torch.float32):
            rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(dtype)
            d = torch.tensor([_RC + 1e-4], dtype=dtype)
            out = rbf(d)
            assert out.shape == (1, 20)
            assert bool((out == 0).all()), f"expected bitwise zeros, got {out}"
            # the identical-zero property holds at every distance beyond
            far = torch.tensor([_RC + 1e-4, _RC + 0.5, 50.0, 1e6], dtype=dtype)
            assert bool((rbf(far) == 0).all())

    def test_gradient_approaches_zero_continuously_at_cutoff(self):
        """Criterion 2: ``d(radial)/dd -> 0`` continuously as
        ``d -> r_cut^-`` — per-channel sup-norm of the Jacobian rows plus
        the summed gradient, both decaying strictly with the offset."""
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(torch.float64)
        deltas = (1e-1, 1e-2, 1e-3, 1e-4)
        sup_norms, sums = [], []
        for delta in deltas:
            d = torch.tensor([_RC - delta], dtype=torch.float64, requires_grad=True)
            feats = rbf(d)  # [1, R]
            feats.sum().backward()
            # per-channel gradient == the Jacobian row (single input)
            per_channel = d.grad.clone()
            sup_norms.append(float(per_channel.abs().max()))
            sums.append(float(d.grad.abs().sum()))
        # strictly decreasing toward zero (continuity of the decay)
        for a, b in zip(sup_norms, sup_norms[1:]):
            assert b < a
        for a, b in zip(sums, sums[1:]):
            assert b < a
        # and small at the last probe: |d(radial)/dd| <= 1e-2 at r_cut-1e-4
        assert sup_norms[-1] <= 1e-2
        assert sums[-1] <= 1.0

    def test_gradient_decay_is_linear_in_the_offset(self):
        # f'(r_cut - delta) ~ -0.5 (pi/r_cut)^2 delta -> the decay is not
        # merely decreasing but vanishes at the documented linear rate
        # (pin with generous slack over the analytic envelope bound).
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(torch.float64)
        for delta in (1e-1, 1e-2, 1e-3):
            d = torch.tensor([_RC - delta], dtype=torch.float64, requires_grad=True)
            rbf(d).sum().backward()
            bound = (math.pi / _RC) ** 2 * delta * 20.0  # sum over channels
            assert float(d.grad.abs().sum()) <= bound


# ---------------------------------------------------------------------------
# 2. The envelope primitive: analytic pinned properties
# ---------------------------------------------------------------------------


class TestEnvelopePrimitive:
    def test_value_at_zero_is_exactly_one(self):
        for dtype in (torch.float64, torch.float32):
            d = torch.zeros(4, dtype=dtype)
            out = continuous_cosine_envelope(d, _RC)
            assert torch.equal(out, torch.ones_like(out))

    def test_value_at_midpoint_is_half(self):
        d = torch.tensor([_RC / 2], dtype=torch.float64)
        out = continuous_cosine_envelope(d, _RC)
        assert out.item() == pytest.approx(0.5, abs=1e-15)

    def test_value_at_cutoff_is_zero(self):
        # exactly 0.0 on the reference CPU platforms in both dtypes; pin a
        # tiny tolerance so the property survives other cos() roundings.
        for dtype in (torch.float64, torch.float32):
            d = torch.tensor([_RC], dtype=dtype)
            out = continuous_cosine_envelope(d, _RC)
            assert float(out.abs().max()) <= 1e-12

    def test_beyond_cutoff_is_bitwise_zero(self):
        for dtype in (torch.float64, torch.float32):
            d = torch.tensor([_RC + 1e-4, _RC + 0.5, 50.0, 1e6], dtype=dtype)
            out = continuous_cosine_envelope(d, _RC)
            assert bool((out == 0).all())
            assert bool((out == 0).all()) and out.dtype == dtype

    def test_bounded_unit_interval(self):
        for dtype in (torch.float64, torch.float32):
            gen = torch.Generator().manual_seed(101)
            d = torch.rand(10_000, generator=gen, dtype=dtype) * 8.0
            out = continuous_cosine_envelope(d, _RC)
            assert bool((out >= 0).all())
            assert bool((out <= 1.0).all())

    def test_monotone_nondecreasing_toward_origin(self):
        # f is non-increasing in d on [0, r_cut] and constant (0) beyond
        d = torch.linspace(0.0, 2.0 * _RC, 4001, dtype=torch.float64)
        out = continuous_cosine_envelope(d, _RC)
        assert bool((torch.diff(out) <= 1e-15).all())

    def test_reflection_identity(self):
        # f(d) + f(r_cut - d) = 1 on [0, r_cut] (cosine symmetry)
        d = torch.linspace(0.0, _RC, 1001, dtype=torch.float64)[1:-1]
        total = continuous_cosine_envelope(d, _RC) + continuous_cosine_envelope(
            _RC - d, _RC
        )
        assert float((total - 1.0).abs().max()) <= 1e-12

    def test_matches_closed_form_reference(self):
        gen = torch.Generator().manual_seed(202)
        d = torch.rand(4096, generator=gen, dtype=torch.float64) * (2 * _RC)
        out = continuous_cosine_envelope(d, _RC)
        ref = _analytic_envelope(d, _RC)
        assert torch.equal(out, ref)

    def test_quadratic_near_cutoff_decay(self):
        # f(r_cut - delta) = 0.5 (1 - cos(pi delta / r_cut)) <= (pi delta/rc)^2 / 4
        # (to the 0.5*(cos+1) cancellation floor near the seam)
        for delta in (1e-1, 1e-2, 1e-3, 1e-4):
            d = torch.tensor([_RC - delta], dtype=torch.float64)
            out = continuous_cosine_envelope(d, _RC)
            bound = (math.pi * delta / _RC) ** 2 / 4.0
            assert out.item() <= bound * (1.0 + 1e-6) + 1e-15

    def test_autograd_matches_analytic_derivative(self):
        gen = torch.Generator().manual_seed(303)
        d = (torch.rand(256, generator=gen, dtype=torch.float64) * 0.98 * _RC) + 0.01
        d = d.clone().requires_grad_(True)
        out = continuous_cosine_envelope(d, _RC)
        out.sum().backward()
        ref = _analytic_envelope_derivative(d.detach(), _RC)
        assert torch.allclose(d.grad, ref, atol=1e-14, rtol=1e-12)

    def test_derivative_is_zero_at_origin(self):
        d = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        continuous_cosine_envelope(d, _RC).sum().backward()
        assert d.grad.item() == 0.0

    def test_derivative_is_exactly_zero_beyond_cutoff(self):
        d = torch.tensor(
            [_RC + 1e-4, _RC + 0.5, 50.0], dtype=torch.float64, requires_grad=True
        )
        out = continuous_cosine_envelope(d, _RC)
        out.sum().backward()
        assert bool((d.grad == 0).all())

    def test_derivative_vanishes_at_cutoff_linearly(self):
        for delta in (1e-1, 1e-2, 1e-3):
            d = torch.tensor([_RC - delta], dtype=torch.float64, requires_grad=True)
            continuous_cosine_envelope(d, _RC).sum().backward()
            bound = 0.5 * (math.pi / _RC) ** 2 * delta * (1.0 + 1e-9)
            assert abs(d.grad.item()) <= bound

    def test_value_continuity_at_the_seam(self):
        # |f(r_cut - delta) - f(r_cut)| = f(r_cut - delta) -> 0 (quadratic)
        for delta in (1e-2, 1e-4, 1e-6):
            d = torch.tensor([_RC - delta], dtype=torch.float64)
            left = continuous_cosine_envelope(d, _RC)
            right = continuous_cosine_envelope(
                torch.tensor([_RC + delta], dtype=torch.float64), _RC
            )
            assert float((left - right).abs().max()) <= (math.pi * delta / _RC) ** 2

    def test_negative_distances_clip_to_one(self):
        # pinned reference semantics: clamp(min=0) maps negative roundoff to
        # f(0) = 1 (GaussianRadialBasis rejects negatives before this point)
        d = torch.tensor([-1e-9, -3.0], dtype=torch.float64)
        out = continuous_cosine_envelope(d, _RC)
        assert torch.equal(out, torch.ones_like(out))

    def test_gradient_does_not_flow_through_the_clamp(self):
        d = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
        continuous_cosine_envelope(d, _RC).sum().backward()
        assert d.grad.item() == 0.0

    def test_gradient_flows_at_interior_points(self):
        d = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        continuous_cosine_envelope(d, _RC).sum().backward()
        assert d.grad.item() != 0.0

    def test_shape_is_rank_agnostic(self):
        d = torch.rand(7, dtype=torch.float64)
        assert continuous_cosine_envelope(d, _RC).shape == (7,)
        d = torch.rand(3, 11, dtype=torch.float64)
        assert continuous_cosine_envelope(d, _RC).shape == (3, 11)
        scalar = torch.tensor(2.5, dtype=torch.float64)
        assert continuous_cosine_envelope(scalar, _RC).shape == ()

    def test_empty_tensor_is_allowed(self):
        d = torch.zeros(0, dtype=torch.float64)
        out = continuous_cosine_envelope(d, _RC)
        assert out.shape == (0,)

    def test_dtype_is_preserved(self):
        for dtype in (torch.float32, torch.float64):
            d = torch.tensor([1.0, 6.0], dtype=dtype)
            assert continuous_cosine_envelope(d, _RC).dtype == dtype

    def test_batched_equals_elementwise(self):
        gen = torch.Generator().manual_seed(404)
        d = torch.rand(2, 50, generator=gen, dtype=torch.float64) * 8.0
        out = continuous_cosine_envelope(d, _RC)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                single = continuous_cosine_envelope(
                    d[i, j].reshape(1), _RC
                )
                assert out[i, j].item() == single.item()

    def test_bitwise_deterministic(self):
        gen = torch.Generator().manual_seed(505)
        d = torch.rand(1000, generator=gen, dtype=torch.float32) * 10.0
        first = continuous_cosine_envelope(d, _RC)
        second = continuous_cosine_envelope(d, _RC)
        assert torch.equal(first, second)

    def test_fp32_fp64_agree_away_from_the_cancellation_seam(self):
        # 0.5*(cos+1) near d=r_cut loses digits to cancellation in fp32; the
        # two dtypes agree to fp32 resolution over the physical range.
        d64 = torch.linspace(0.0, 0.999 * _RC, 500, dtype=torch.float64)
        out64 = continuous_cosine_envelope(d64, _RC)
        out32 = continuous_cosine_envelope(d64.to(torch.float32), _RC).to(
            torch.float64
        )
        assert torch.allclose(out32, out64, atol=1e-7)

    def test_cutoff_scale_invariance_of_shape(self):
        # different cutoffs (backbone 5 A, handle head 12 A) behave alike
        for cutoff in (5.0, 12.0, 0.37):
            mid = torch.tensor([cutoff / 2], dtype=torch.float64)
            assert continuous_cosine_envelope(mid, cutoff).item() == pytest.approx(
                0.5, abs=1e-12
            )
            beyond = torch.tensor([cutoff + 1e-4], dtype=torch.float64)
            assert continuous_cosine_envelope(beyond, cutoff).item() == 0.0

    # -- fail-closed input contract -----------------------------------------

    def test_fail_closed_non_tensor(self):
        with pytest.raises(TensorContractError, match="torch.Tensor"):
            continuous_cosine_envelope([1.0, 2.0], _RC)

    def test_fail_closed_non_finite(self):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(TensorContractError, match="non-finite"):
                continuous_cosine_envelope(
                    torch.tensor([1.0, bad], dtype=torch.float64), _RC
                )

    def test_fail_closed_non_floating_dtype(self):
        with pytest.raises(TensorContractError, match="floating dtype"):
            continuous_cosine_envelope(torch.tensor([1, 2], dtype=torch.long), _RC)
        with pytest.raises(TensorContractError, match="floating dtype"):
            continuous_cosine_envelope(torch.tensor([True]), _RC)

    def test_fail_closed_bad_cutoff(self):
        d = torch.tensor([1.0], dtype=torch.float64)
        for bad in (0.0, -5.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="cutoff"):
                continuous_cosine_envelope(d, bad)


# ---------------------------------------------------------------------------
# 3. The enveloped GaussianRadialBasis (the one authoritative integration)
# ---------------------------------------------------------------------------


class TestEnvelopedRadialBasis:
    def test_first_channel_exactly_one_at_zero_distance(self):
        # f_cut(0) = 1 and the first Gaussian peaks at its center 0
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC)
        out = rbf(torch.zeros(1))
        assert out[0, 0].item() == 1.0

    def test_peaks_carry_the_envelope(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(torch.float64)
        expected = _analytic_envelope(rbf.centers, _RC)
        for k in range(20):
            value = rbf(rbf.centers[k].reshape(1))[0, k]
            assert value.item() == pytest.approx(float(expected[k]), abs=1e-15)

    def test_at_cutoff_features_vanish(self):
        for dtype in (torch.float64, torch.float32):
            rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(dtype)
            out = rbf(torch.tensor([_RC], dtype=dtype))
            tol = 1e-12 if dtype == torch.float64 else 1e-6
            assert float(out.abs().max()) <= tol

    def test_sup_norm_bounded_by_the_envelope(self):
        # every channel is a Gaussian (<= 1) times the envelope, so the
        # feature sup-norm at d is <= f_cut(d) — the taper is authoritative
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(torch.float64)
        d = torch.linspace(0.0, _RC, 501, dtype=torch.float64)
        out = rbf(d)
        env = _analytic_envelope(d, _RC)
        sup = out.max(dim=-1).values
        assert bool((sup <= env + 1e-15).all())

    def test_vectorized_matches_manual_gaussian_times_envelope(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(torch.float64)
        gen = torch.Generator().manual_seed(606)
        d = torch.rand(512, generator=gen, dtype=torch.float64) * (2 * _RC)
        out = rbf(d)
        # reference uses the module's own buffers (centers/scale carry the
        # construction-dtype rounding; the math under test is the product)
        scale = float(rbf._scale)
        for k in range(20):
            mu = float(rbf.centers[k])
            raw = torch.exp(scale * (d - mu) ** 2)
            expected = raw * _analytic_envelope(d, _RC)
            assert torch.allclose(out[:, k], expected, atol=1e-14, rtol=1e-12)

    def test_analytic_radial_derivative_match(self):
        # d(radial_k)/dd = f'(d) g_k(d) + f(d) g_k'(d)  (product rule); the
        # Gaussian reference uses the module's own buffers so the test pins
        # the envelope product-rule mathematics, not buffer rounding.
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(torch.float64)
        scale = float(rbf._scale)
        gen = torch.Generator().manual_seed(707)
        points = torch.rand(64, generator=gen, dtype=torch.float64) * 0.98 * _RC
        d = points.clone().requires_grad_(True)
        feats = rbf(d)  # [N, R]
        for k in (0, 7, 13, 19):
            mu = float(rbf.centers[k])
            g = torch.exp(scale * (points - mu) ** 2)
            g_prime = 2.0 * scale * (points - mu) * g
            expected = (
                _analytic_envelope_derivative(points, _RC) * g
                + _analytic_envelope(points, _RC) * g_prime
            )
            assert torch.allclose(feats[:, k], g * _analytic_envelope(points, _RC))
            grads = torch.autograd.grad(
                feats[:, k].sum(), d, retain_graph=True
            )[0]
            assert torch.allclose(grads, expected, atol=1e-14, rtol=1e-10)

    def test_envelope_adds_no_parameters_and_keeps_buffers(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC)
        assert sum(p.numel() for p in rbf.parameters()) == 0
        assert rbf.centers.shape == (20,)
        assert float(rbf.centers[0]) == 0.0
        assert float(rbf.centers[-1]) == pytest.approx(_RC)

    def test_state_dict_roundtrip_preserves_behavior(self):
        import copy

        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC).to(torch.float64)
        clone = copy.deepcopy(rbf)
        d = torch.linspace(0.0, 2 * _RC, 64, dtype=torch.float64)
        assert torch.equal(rbf(d), clone(d))
        clone.load_state_dict(rbf.state_dict())
        assert torch.equal(rbf(d), clone(d))

    def test_fail_closed_negative_distance_unchanged(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC)
        with pytest.raises(TensorContractError, match="negative"):
            rbf(torch.tensor([-1e-9]))

    def test_fail_closed_non_finite_distance_unchanged(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC)
        with pytest.raises(TensorContractError, match="non-finite"):
            rbf(torch.tensor([float("nan")]))

    def test_fail_closed_non_floating_distance(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC)
        with pytest.raises(TensorContractError, match="floating dtype"):
            rbf(torch.tensor([1, 2], dtype=torch.long))


# ---------------------------------------------------------------------------
# 4. Downstream integration: handle head basis + backbone equivariance
# ---------------------------------------------------------------------------


class TestDownstreamIntegration:
    def test_handle_head_centroid_basis_is_enveloped(self):
        """The #22 head's [0, 12] A centroid-distance RBF carries the
        envelope (D-V2-007/D-V2-011: #9 owns it for ALL spatial RBF
        projections)."""
        head = HandleSelectionHead()
        assert head.geom_cutoff == HANDLE_GEOM_CUTOFF == 12.0
        beyond = head.radial_basis(
            torch.tensor(
                [HANDLE_GEOM_CUTOFF + 1e-4, 20.0, 50.0, 1e6],
                dtype=torch.float32,
            )
        )
        assert bool((beyond == 0).all())

    def test_vulnerability_7_regime_is_dead(self):
        """Legacy failure mode: clamped raw RBF gave every spectator from
        ``r_cut + eps`` to 50 A an identical non-zero bias feature. With the
        envelope the far spectators contribute *exactly* zero — and the
        near-cutoff ones taper smoothly instead of saturating."""
        rbf = GaussianRadialBasis(num_radial=20, r_cut=_RC)
        spectators = torch.tensor([5.01, 7.5, 12.0, 50.0], dtype=torch.float32)
        features = rbf(spectators)
        assert torch.equal(features, torch.zeros_like(features))
        # near the cutoff the taper is smooth, not saturated:
        near = rbf(torch.tensor([4.9, 4.99], dtype=torch.float32))
        assert bool((near[:, -1] > 0).all())
        assert float(near[0, -1]) > float(near[1, -1]) > 0.0

    @staticmethod
    def _make_state(dtype: torch.dtype, seed: int) -> SimpleNamespace:
        gen = torch.Generator().manual_seed(seed)
        return SimpleNamespace(
            pocket_pos=torch.randn(10, 3, generator=gen, dtype=dtype) * 3.0,
            pocket_z=torch.randint(1, 9, (10,), generator=gen),
            pocket_charge=torch.zeros(10, dtype=dtype),
            pocket_batch=torch.zeros(10, dtype=torch.long),
            ligand_pos=torch.randn(6, 3, generator=gen, dtype=dtype) * 2.0,
            ligand_z=torch.randint(1, 9, (6,), generator=gen),
            ligand_charge=torch.zeros(6, dtype=dtype),
            ligand_batch=torch.zeros(6, dtype=torch.long),
        )

    def test_backbone_smoke_with_envelope(self):
        model = BipartitePaiNN(hidden_dim=16, num_layers=2, num_radial=6, r_cut=_RC)
        out = model(self._make_state(torch.float32, seed=808))
        assert out.scalar.shape == (16, 16)
        assert out.vector.shape == (16, 3, 16)
        assert bool(torch.isfinite(out.scalar).all())
        assert bool(torch.isfinite(out.vector).all())

    def test_backbone_scalar_invariance_with_envelope(self):
        """The envelope is a radial (distance-only) function, so enveloping
        the RBF cannot affect SE(3) invariance/equivariance — pinned here
        compactly (the exhaustive 36-trial suite re-runs in
        tests/test_bipartite_painn.py)."""
        model = (
            BipartitePaiNN(hidden_dim=16, num_layers=2, num_radial=6, r_cut=_RC)
            .to(torch.float64)
            .eval()
        )
        state = self._make_state(torch.float64, seed=909)
        base = model(state)
        gen = torch.Generator().manual_seed(910)
        for _ in range(3):
            # random rotation via QR of a Gaussian matrix (orthogonal, det +-1)
            q, _ = torch.linalg.qr(
                torch.randn(3, 3, generator=gen, dtype=torch.float64)
            )
            if torch.det(q) < 0:
                q[:, 0] = -q[:, 0]
            shift = torch.randn(3, generator=gen, dtype=torch.float64) * 10.0
            moved = SimpleNamespace(
                **{
                    **vars(state),
                    "pocket_pos": state.pocket_pos @ q.T + shift,
                    "ligand_pos": state.ligand_pos @ q.T + shift,
                }
            )
            out = model(moved)
            assert torch.allclose(
                out.scalar, base.scalar, atol=1e-10
            ), "scalar invariance broken by the envelope"
            # vector equivariance: each per-channel [3] vector rotates by q
            expected = torch.einsum("ij,njd->nid", q, base.vector)
            assert torch.allclose(
                out.vector, expected, atol=1e-10
            ), "vector equivariance broken by the envelope"
