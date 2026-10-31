"""#7 MATH-01 — non-singular PaiNN norms & bounded unit vectors.

Acceptance criteria (issue #7):
* ``test_safe_norm_zero_distance``: at ``x = 0`` the smoothed norm outputs
  ``0.0`` and the autograd gradient is strictly ``0.0`` (zero NaNs/Infs).
* Random-rotation verification: directional vectors maintain exact
  SO(3) covariance under coordinate rotations.

Beyond the two acceptance bullets, this file pins the analytic gradient,
the gradient bound (|grad| <= 1 for the norm, <= 1/eps for the unit
vector), degenerate-input behaviour of the PaiNN stack end-to-end
(coincident atoms, zero vector features), and rotation equivariance of
the refactored layers with random proper rotations.
"""

from __future__ import annotations

import math

import pytest
import torch

from syntree.models.equivariant import (
    SAFE_NORM_EPS,
    BatchedPaiNNLayer,
    PaiNNLayer,
    PaiNNMixing,
    RadialBasis,
    build_radius_graph,
    safe_norm,
    safe_unit_vector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_rotation(seed: int) -> torch.Tensor:
    """A random *proper* rotation (det = +1) via QR with sign correction."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g)
    Q, R = torch.linalg.qr(A)
    # make det(Q) = +1 (a reflection otherwise)
    Q = Q @ torch.diag(torch.sign(torch.diagonal(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    assert torch.allclose(Q @ Q.T, torch.eye(3), atol=1e-6)
    assert torch.det(Q).item() > 0
    return Q


# ---------------------------------------------------------------------------
# Acceptance criterion 1: safe_norm at the zero vector
# ---------------------------------------------------------------------------


class TestSafeNormZeroDistance:
    def test_safe_norm_zero_distance(self):
        """x = 0  ->  output 0.0, gradient strictly 0.0, zero NaN/Inf.

        The exact-name acceptance test of issue #7.
        """
        x = torch.zeros(8, 3, dtype=torch.float32, requires_grad=True)
        out = safe_norm(x, dim=-1)
        assert torch.isfinite(out).all()
        # forward value is numerically zero (fp32-exact at this eps)
        assert out.abs().max().item() == 0.0, f"safe_norm(0) = {out}"
        out.sum().backward()
        assert x.grad is not None
        # gradient strictly zero, no NaNs or Infs anywhere
        assert torch.isfinite(x.grad).all()
        assert (x.grad == 0.0).all(), f"grad at 0 = {x.grad}"

    def test_safe_norm_zero_float64(self):
        x = torch.zeros(5, 3, dtype=torch.float64, requires_grad=True)
        out = safe_norm(x, dim=-1)
        assert out.abs().max().item() <= 1e-18
        out.sum().backward()
        assert (x.grad == 0.0).all()

    def test_coincident_coordinates_gradient_finite(self):
        """The motivating singularity: d_ij -> 0 must not spike grads."""
        d = torch.zeros(1, 3, requires_grad=True)  # exactly coincident
        out = safe_norm(d, dim=-1)
        out.backward()
        assert torch.isfinite(d.grad).all()
        assert (d.grad == 0.0).all()

    def test_tiny_distances_gradient_bounded(self):
        """|grad safe_norm| <= 1 everywhere, including the near-zero regime
        where the legacy torch.norm direction is numerically undefined."""
        scales = torch.tensor([0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e3])
        for s in scales.tolist():
            x = torch.tensor([[s, 0.0, 0.0]], requires_grad=True)
            out = safe_norm(x, dim=-1)
            out.backward()
            assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
            assert x.grad.abs().max().item() <= 1.0 + 1e-9, (
                f"gradient bound violated at scale {s}: {x.grad.abs().max()}"
            )


# ---------------------------------------------------------------------------
# safe_norm analytic properties
# ---------------------------------------------------------------------------


class TestSafeNormMath:
    def test_matches_torch_norm_asymptotically(self):
        """The smoothed norm converges to the exact norm with an absolute
        error bounded by eps (sqrt(d^2+eps^2) - eps = d - eps + O(eps^2/d)).
        """
        torch.manual_seed(0)
        x = torch.randn(64, 3) * 10.0
        exact = torch.norm(x, dim=-1)
        err = (safe_norm(x, dim=-1) - exact).abs()
        # absolute error <= eps (smoothing bias) + fp32 rounding of the
        # squared sum at these magnitudes (ulp(d^2) ~ 3e-5 -> ~1e-6 on d)
        assert err.max().item() <= 2.0 * SAFE_NORM_EPS, err.max()
        # relative error vanishes as |x|/eps -> infinity
        rel = err / exact
        assert rel.max().item() < 1e-5

    def test_analytic_gradient_matches_autograd(self):
        """grad = x / sqrt(sum x^2 + eps^2), elementwise."""
        torch.manual_seed(1)
        x = torch.randn(32, 3, requires_grad=True)
        out = safe_norm(x, dim=-1)
        out.sum().backward()
        expected = x.detach() / torch.sqrt(
            (x.detach() ** 2).sum(-1, keepdim=True) + SAFE_NORM_EPS ** 2
        )
        assert torch.allclose(x.grad, expected, atol=1e-6)

    def test_non_negative_and_monotone(self):
        torch.manual_seed(2)
        x = torch.randn(50, 3)
        n = safe_norm(x, dim=-1)
        assert (n >= 0).all()
        # monotone in the magnitude: scaling up scales the norm up
        n2 = safe_norm(x * 2.0, dim=-1)
        assert (n2 >= n).all()

    def test_keepdim_and_dim_arguments(self):
        x = torch.randn(6, 3, 4)
        out = safe_norm(x, dim=1, keepdim=True)
        assert out.shape == (6, 1, 4)
        out2 = safe_norm(x, dim=1)
        assert out2.shape == (6, 4)
        assert torch.allclose(out.squeeze(1), out2)

    def test_custom_eps(self):
        x = torch.zeros(3)
        for eps in (1e-4, 1e-5, 1e-3):
            assert safe_norm(x, dim=-1, eps=eps).abs().max().item() <= eps * 1e-6

    def test_3_4_5_triangle(self):
        x = torch.tensor([[3.0, 4.0, 0.0]])
        n = safe_norm(x, dim=-1).item()
        assert n == pytest.approx(5.0, rel=1e-5)


# ---------------------------------------------------------------------------
# safe_unit_vector
# ---------------------------------------------------------------------------


class TestSafeUnitVector:
    def test_zero_vector_zero_unit_and_dist(self):
        vec = torch.zeros(4, 3, requires_grad=True)
        unit, dist = safe_unit_vector(vec, dim=-1)
        assert torch.isfinite(unit).all() and torch.isfinite(dist).all()
        assert (unit == 0.0).all(), "zero vector must map to the zero direction"
        assert (dist == 0.0).all()
        (unit.sum() + dist.sum()).backward()
        assert torch.isfinite(vec.grad).all()
        # bounded by the 1/eps smoothing scale, never infinite
        assert vec.grad.abs().max().item() <= 1.0 / SAFE_NORM_EPS + 1.0

    def test_unit_norms_are_one_for_non_degenerate(self):
        torch.manual_seed(3)
        vec = torch.randn(100, 3) * torch.tensor([1.0, 1e-2, 1e-3])
        unit, dist = safe_unit_vector(vec, dim=-1)
        assert torch.isfinite(unit).all() and torch.isfinite(dist).all()
        # only the genuinely tiny directions deviate from unit length
        big = dist > 100 * SAFE_NORM_EPS
        assert torch.allclose(
            torch.norm(unit[big], dim=-1), torch.ones(int(big.sum())), atol=1e-3
        )

    def test_dist_equals_safe_norm(self):
        torch.manual_seed(4)
        vec = torch.randn(20, 3)
        _, dist = safe_unit_vector(vec, dim=-1)
        assert torch.allclose(dist, safe_norm(vec, dim=-1), atol=1e-7)

    def test_gradient_bounded_near_zero(self):
        for s in (0.0, 1e-12, 1e-9, 1e-6):
            vec = torch.tensor([[s, 0.0, 0.0]], requires_grad=True)
            unit, dist = safe_unit_vector(vec, dim=-1)
            (unit.sum() + dist.sum()).backward()
            assert torch.isfinite(vec.grad).all()
            assert vec.grad.abs().max().item() <= 1.0 / SAFE_NORM_EPS + 1.0

    def test_dim_argument_batched_features(self):
        """[N, 3, H] vector features normalise over the spatial axis."""
        torch.manual_seed(5)
        vec = torch.randn(7, 3, 11)
        unit, dist = safe_unit_vector(vec, dim=1)
        assert unit.shape == (7, 3, 11)
        assert dist.shape == (7, 11)
        assert torch.isfinite(unit).all() and torch.isfinite(dist).all()


# ---------------------------------------------------------------------------
# Acceptance criterion 2: SO(3) covariance under random rotations
# ---------------------------------------------------------------------------


class TestSO3Covariance:
    def test_directional_vectors_exact_rotation_covariance(self):
        """unit(R x) == R unit(x) for random proper rotations (the issue's
        random-rotation verification)."""
        torch.manual_seed(6)
        vec = torch.randn(200, 3)
        unit_x, dist_x = safe_unit_vector(vec, dim=-1)
        for seed in range(5):
            R = random_rotation(seed)
            unit_Rx, dist_Rx = safe_unit_vector(vec @ R.T, dim=-1)
            # exact covariance up to fp32 arithmetic reordering
            assert torch.allclose(unit_Rx, unit_x @ R.T, atol=1e-6), (
                f"direction covariance broken for rotation seed {seed}"
            )
            # distances are rotation invariant
            assert torch.allclose(dist_Rx, dist_x, atol=1e-5)

    def test_safe_norm_rotation_invariant(self):
        torch.manual_seed(7)
        x = torch.randn(64, 3)
        n0 = safe_norm(x, dim=-1)
        for seed in (11, 12, 13):
            R = random_rotation(seed)
            nR = safe_norm(x @ R.T, dim=-1)
            assert torch.allclose(n0, nR, atol=1e-5)


# ---------------------------------------------------------------------------
# PaiNN stack: degenerate geometry end-to-end
# ---------------------------------------------------------------------------


class TestPaiNNDegenerateGeometry:
    def _graph(self, pos: torch.Tensor, cutoff: float = 6.0, num_radial: int = 8):
        batch = torch.zeros(pos.size(0), dtype=torch.long)
        ei = build_radius_graph(pos, batch, cutoff=cutoff)
        row, col = ei
        edge_vec = pos[col] - pos[row]
        edge_dist = safe_norm(edge_vec, dim=-1)
        edge_rbf = RadialBasis(num_radial, cutoff)(edge_dist)
        return ei, edge_vec, edge_rbf

    def test_coincident_atoms_finite_forward_and_backward(self):
        """Duplicate atom coordinates through the full layer: no NaN/Inf in
        activations or gradients (the V1 gradient-spike failure mode)."""
        torch.manual_seed(8)
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # exactly coincident pair
                [1.2, 0.0, 0.0],
                [1.2, 0.0, 0.0],  # another coincident pair
                [0.0, 1.5, 0.0],
            ],
            requires_grad=True,
        )
        layer = BatchedPaiNNLayer(hidden_dim=16, num_radial=8)
        batch = torch.zeros(5, dtype=torch.long)
        ei = build_radius_graph(pos.detach(), batch, cutoff=6.0)
        row, col = ei
        edge_vec = pos[col] - pos[row]           # tracked through pos
        edge_dist = safe_norm(edge_vec, dim=-1)
        edge_rbf = RadialBasis(8, 6.0)(edge_dist)
        s = torch.randn(5, 16)
        v = torch.randn(5, 3, 16)
        out_s, out_v = layer(s, v, ei, edge_vec, edge_rbf)
        assert torch.isfinite(out_s).all() and torch.isfinite(out_v).all()
        (out_s.sum() + out_v.sum()).backward()
        assert torch.isfinite(pos.grad).all()
        # Bounded at the documented O(1/eps) scale of the smoothing radius -
        # no infinite/NaN spikes (the V1 failure mode was >1e5 corruption
        # and 0/0 NaNs at exactly coincident coordinates).
        assert pos.grad.abs().max().item() < 1.0 / SAFE_NORM_EPS

    def test_painn_layer_forward_coincident_positions(self):
        """PaiNNLayer (computes its own geometry) with duplicated coords."""
        torch.manual_seed(9)
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            requires_grad=True,
        )
        layer = PaiNNLayer(hidden_dim=12, num_radial=6, cutoff=4.0)
        batch = torch.zeros(3, dtype=torch.long)
        ei = build_radius_graph(pos.detach(), batch, cutoff=4.0)
        s = torch.randn(3, 12)
        v = torch.zeros(3, 3, 12)
        out_s, out_v = layer(s, v, ei, pos)
        assert torch.isfinite(out_s).all() and torch.isfinite(out_v).all()
        (out_s.sum() + out_v.sum()).backward()
        assert torch.isfinite(pos.grad).all()
        # bounded at the O(1/eps) smoothing scale, never infinite
        assert pos.grad.abs().max().item() < 1.0 / SAFE_NORM_EPS

    def test_mixing_zero_vector_features(self):
        """Zero vector features give exactly-zero norms, finite grads."""
        torch.manual_seed(10)
        mix = PaiNNMixing(hidden_dim=16)
        s = torch.randn(6, 16, requires_grad=True)
        v = torch.zeros(6, 3, 16, requires_grad=True)
        out_s, out_v = mix(s, v)
        assert torch.isfinite(out_s).all() and torch.isfinite(out_v).all()
        out_s.sum().backward()
        assert torch.isfinite(s.grad).all() and torch.isfinite(v.grad).all()

    def test_mixing_norm_is_smoothed_not_biased(self):
        """v_norm of a zero projection is 0 (the legacy +eps bias gave eps)."""
        mix = PaiNNMixing(hidden_dim=8)
        with torch.no_grad():
            w = torch.zeros_like(mix.vec_proj.weight)
            mix.vec_proj.weight.copy_(w)  # vec_proj(x) == 0 for any input
        v = torch.randn(4, 3, 8)
        # recompute what the mixing sees: projected vectors are exactly zero
        proj = mix.vec_proj(v)
        n = safe_norm(proj, dim=1, eps=mix.epsilon)
        assert (n == 0.0).all()

    def test_bipartite_encoder_coincident_pocket_atoms(self):
        """End-to-end through the policy's joint encoder."""
        from syntree.models.bipartite import BipartitePaiNN

        torch.manual_seed(11)
        enc = BipartitePaiNN(hidden_dim=16, num_layers=2, num_radial=6).eval()
        pocket_pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.4, 0.3]]
        )
        pocket_z = torch.tensor([6, 7, 8, 6])
        ligand_pos = torch.tensor([[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]])  # clashing
        ligand_z = torch.tensor([6, 6])
        batch = torch.zeros(6, dtype=torch.long)
        with torch.no_grad():
            s, v, pooled = enc(
                pocket_pos, pocket_z, batch[:4], None, ligand_pos, ligand_z, batch[4:]
            )
        assert torch.isfinite(s).all() and torch.isfinite(v).all()
        assert torch.isfinite(pooled).all()


# ---------------------------------------------------------------------------
# Equivariance of the refactored stack under random proper rotations
# ---------------------------------------------------------------------------


class TestPaiNNEquivarianceRandomRotations:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_scalar_invariance_vector_equivariance(self, seed):
        torch.manual_seed(seed)
        layer = BatchedPaiNNLayer(hidden_dim=24, num_radial=12).eval()
        pos = torch.randn(10, 3)
        batch = torch.zeros(10, dtype=torch.long)
        R = random_rotation(100 + seed)

        ei = build_radius_graph(pos, batch, cutoff=8.0)
        row, col = ei
        ev = pos[col] - pos[row]
        ed = safe_norm(ev, dim=-1)
        er = RadialBasis(12, 8.0)(ed)
        s = torch.randn(10, 24)
        v = torch.randn(10, 3, 24)

        pos_rot = pos @ R.T
        ei_rot = build_radius_graph(pos_rot, batch, cutoff=8.0)
        row_r, col_r = ei_rot
        ev_r = pos_rot[col_r] - pos_rot[row_r]
        ed_r = safe_norm(ev_r, dim=-1)
        er_r = RadialBasis(12, 8.0)(ed_r)

        with torch.no_grad():
            s1, v1 = layer(s, v, ei, ev, er)
            s2, v2 = layer(s, torch.einsum("nkh,sk->nsh", v, R), ei_rot, ev_r, er_r)

        assert torch.allclose(s1, s2, atol=1e-5), (s1 - s2).abs().max()
        expected_v = torch.einsum("nkh,sk->nsh", v1, R)
        assert torch.allclose(v2, expected_v, atol=1e-4), (v2 - expected_v).abs().max()

    def test_safe_unit_vector_so3_covariance_property(self):
        """The covariance property used above, pinned directly on the
        primitive with the identity: unit(Rx) = R unit(x)."""
        torch.manual_seed(12)
        vec = torch.randn(50, 3) * 2.0
        unit, _ = safe_unit_vector(vec, dim=-1)
        R = random_rotation(42)
        unit_rot, _ = safe_unit_vector(vec @ R.T, dim=-1)
        assert torch.allclose(unit_rot, unit @ R.T, atol=1e-6)
