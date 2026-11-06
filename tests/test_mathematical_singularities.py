"""Non-singular PaiNN equivariant norms & bounded unit vectors — tests
(#7 MATH-01).

Acceptance criteria of issue #7 (clean-room V2 implementation):

1. ``tests/test_mathematical_singularities.py::test_safe_norm_zero_distance``
   passes: at ``x = 0`` the output is exactly ``0.0`` and the autograd
   gradient is strictly ``0.0`` (zero NaNs/Infs).
2. Random rotation verification confirms directional vectors maintain
   exact SO(3) covariance under coordinate rotations.

Beyond the two pinned criteria, this suite enforces the analytic-gradient
identity, the |∇| ≤ 1 bound, torch.norm asymptotics, boundedness of unit
vectors across 30 orders of magnitude, degenerate geometry (coincident
atoms, zero vector features, empty edge sets) through the PaiNN blocks,
and scalar invariance / vector equivariance / translation invariance of
the blocks under random rigid motions.
"""

from __future__ import annotations

import pytest
import torch

from syntree.models.equivariant import (
    SAFE_NORM_EPS,
    PaiNNInteraction,
    PaiNNMixing,
    safe_norm,
    safe_unit_vector,
)

H = 8  # hidden dimension used throughout the block tests
R = 6  # radial basis width used throughout the block tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_rotation(generator: torch.Generator) -> torch.Tensor:
    """A random *proper* rotation (orthogonal, det = +1) in float64."""
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # Fix the sign convention so the decomposition is well-conditioned.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotate_vectors(rotation: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Apply R to the leading spatial dimension of ``[..., 3, H]`` features."""
    return torch.einsum("ab,nbh->nah", rotation, vectors)


def make_graph(
    num_nodes: int = 7,
    num_edges: int = 9,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    coincident: bool = False,
):
    """Random scalar/vector features, positions, and a j->i edge set."""
    gen = torch.Generator().manual_seed(seed)
    pos = torch.randn(num_nodes, 3, generator=gen, dtype=dtype)
    if coincident and num_nodes >= 2:
        # Atom 1 exactly coincides with atom 0 (a clash): d -> 0.
        with torch.no_grad():
            pos[1] = pos[0]
    scalar = torch.randn(num_nodes, H, generator=gen, dtype=dtype)
    vector = torch.randn(num_nodes, 3, H, generator=gen, dtype=dtype)
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen)
    if coincident:
        # Guarantee at least one edge on the coincident pair (both ways).
        src = torch.cat([src, torch.tensor([0, 1])])
        dst = torch.cat([dst, torch.tensor([1, 0])])
    edge_index = torch.stack([dst, src])  # messages flow src(j) -> dst(i)
    edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
    edge_rbf = torch.randn(edge_index.size(1), R, generator=gen, dtype=dtype)
    return pos, scalar, vector, edge_index, edge_vec, edge_rbf


def make_layers(*, seed: int = 0, dtype: torch.dtype = torch.float64):
    torch.manual_seed(seed)
    interaction = PaiNNInteraction(H, R).to(dtype)
    mixing = PaiNNMixing(H).to(dtype)
    return interaction, mixing


# ---------------------------------------------------------------------------
# safe_norm
# ---------------------------------------------------------------------------


class TestSafeNorm:
    def test_safe_norm_zero_distance(self):
        """Issue #7 acceptance criterion 1 (exact test name pinned).

        At x = 0: output exactly 0.0, autograd gradient strictly 0.0,
        zero NaNs/Infs anywhere (forward and backward).
        """
        x = torch.zeros(64, 3, dtype=torch.float32, requires_grad=True)
        out = safe_norm(x, dim=-1)
        assert torch.equal(out, torch.zeros_like(out))  # exactly 0.0
        assert bool(torch.isfinite(out).all())
        out.sum().backward()
        assert x.grad is not None
        assert torch.equal(x.grad, torch.zeros_like(x.grad))  # strictly 0.0
        assert bool(torch.isfinite(x.grad).all())

    def test_zero_distance_float64(self):
        x = torch.zeros(16, 3, dtype=torch.float64, requires_grad=True)
        out = safe_norm(x, dim=-1)
        assert torch.equal(out, torch.zeros_like(out))
        out.sum().backward()
        assert torch.equal(x.grad, torch.zeros_like(x.grad))

    def test_analytic_gradient_identity(self):
        """Autograd matches the closed form x / sqrt(sum x^2 + eps^2)."""
        gen = torch.Generator().manual_seed(11)
        x = torch.randn(64, 3, generator=gen, dtype=torch.float64, requires_grad=True)
        safe_norm(x, dim=-1).sum().backward()
        analytic = x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + SAFE_NORM_EPS**2)
        assert torch.allclose(x.grad, analytic, atol=1e-12)

    @pytest.mark.parametrize(
        "scale", [0.0, 1e-30, 1e-12, 1e-8, 1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6]
    )
    def test_gradient_magnitude_bounded_by_one(self, scale):
        gen = torch.Generator().manual_seed(12)
        x = (torch.randn(32, 3, generator=gen) * scale).requires_grad_(True)
        safe_norm(x, dim=-1).sum().backward()
        assert bool(torch.isfinite(x.grad).all())
        assert x.grad.abs().max().item() <= 1.0 + 1e-6

    def test_asymptotically_matches_torch_norm(self):
        gen = torch.Generator().manual_seed(13)
        x = torch.randn(128, 3, generator=gen) * 10.0  # ||x|| ~ 10..50
        diff = (safe_norm(x, dim=-1) - torch.norm(x, dim=-1)).abs()
        assert diff.max().item() <= 2e-5  # <= eps + fp32 rounding
        rel = diff / torch.norm(x, dim=-1)
        assert rel.max().item() < 1e-5

    def test_dim_and_keepdim_variants(self):
        gen = torch.Generator().manual_seed(14)
        v = torch.randn(5, 3, 7, generator=gen)
        assert safe_norm(v, dim=1).shape == (5, 7)
        assert safe_norm(v, dim=1, keepdim=True).shape == (5, 1, 7)
        assert safe_norm(v, dim=-1).shape == (5, 3)
        assert torch.allclose(
            safe_norm(v, dim=1), safe_norm(v, dim=1, keepdim=True).squeeze(1)
        )

    def test_matches_manual_formula(self):
        gen = torch.Generator().manual_seed(15)
        x = torch.randn(64, 3, generator=gen)
        manual = torch.sqrt((x * x).sum(-1) + SAFE_NORM_EPS**2) - SAFE_NORM_EPS
        assert torch.equal(safe_norm(x, dim=-1), manual)


# ---------------------------------------------------------------------------
# safe_unit_vector
# ---------------------------------------------------------------------------


class TestSafeUnitVector:
    def test_zero_vector_yields_zero_direction_and_distance(self):
        v = torch.zeros(16, 3, dtype=torch.float32, requires_grad=True)
        unit, dist = safe_unit_vector(v, dim=-1)
        assert torch.equal(unit, torch.zeros_like(unit))
        assert torch.equal(dist, torch.zeros_like(dist))
        (unit.sum() + dist.sum()).backward()
        assert x_grad_finite_and_bounded(v.grad)

    @pytest.mark.parametrize(
        "magnitude", [0.0, 1e-30, 1e-12, 1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6]
    )
    def test_unit_norm_strictly_bounded(self, magnitude):
        gen = torch.Generator().manual_seed(21)
        v = torch.randn(64, 3, generator=gen)
        v = v / torch.norm(v, dim=-1, keepdim=True) * magnitude
        unit, dist = safe_unit_vector(v, dim=-1)
        assert bool(torch.isfinite(unit).all())
        assert bool(torch.isfinite(dist).all())
        assert torch.norm(unit, dim=-1).max().item() <= 1.0 + 1e-6
        assert (dist >= 0).all()
        assert torch.allclose(dist, safe_norm(v, dim=-1), atol=1e-7)
        if magnitude == 0.0:
            assert torch.equal(unit, torch.zeros_like(unit))
            assert torch.equal(dist, torch.zeros_like(dist))

    def test_unit_direction_correct_at_large_magnitudes(self):
        gen = torch.Generator().manual_seed(22)
        v = torch.randn(32, 3, generator=gen) * 10.0
        unit, _ = safe_unit_vector(v, dim=-1)
        reference = v / torch.norm(v, dim=-1, keepdim=True)
        assert torch.allclose(unit, reference, atol=1e-4)

    def test_gradients_finite_at_tiny_magnitudes(self):
        gen = torch.Generator().manual_seed(23)
        v = (torch.randn(32, 3, generator=gen) * 1e-30).requires_grad_(True)
        unit, dist = safe_unit_vector(v, dim=-1)
        (unit.sum() + dist.sum()).backward()
        assert x_grad_finite_and_bounded(v.grad)


def x_grad_finite_and_bounded(
    grad: torch.Tensor, *, bound: float | None = None
) -> bool:
    """Finite gradients bounded at the documented O(1/eps) scale.

    ``bound`` defaults to the primitive-level ``1/eps + 1``; block-level
    callers pass ``H/eps`` because up to ``H`` channels accumulate through
    the direction path (each contributing at most ``|gate|/eps``).
    """
    if bound is None:
        bound = 1.0 / SAFE_NORM_EPS + 1.0
    assert grad is not None
    assert bool(torch.isfinite(grad).all())
    assert grad.abs().max().item() <= bound + 1e-3
    return True


# ---------------------------------------------------------------------------
# SO(3) covariance (issue #7 acceptance criterion 2)
# ---------------------------------------------------------------------------


class TestSO3Covariance:
    def test_directional_vectors_exact_rotation_covariance(self):
        """Issue #7 acceptance criterion 2 (class/name semantics pinned).

        For independent random proper rotations, directional unit vectors
        transform covariantly (unit(Rx) == R·unit(x)) and distances are
        invariant (dist(Rx) == dist(x)).
        """
        gen = torch.Generator().manual_seed(20261101)
        for trial in range(5):
            rotation = random_rotation(gen)
            assert torch.det(rotation).item() == pytest.approx(1.0, abs=1e-9)
            assert torch.allclose(
                rotation @ rotation.T, torch.eye(3, dtype=torch.float64), atol=1e-9
            )
            x = torch.randn(200, 3, generator=gen, dtype=torch.float64)
            unit, dist = safe_unit_vector(x, dim=-1)
            unit_rot, dist_rot = safe_unit_vector(x @ rotation.T, dim=-1)
            assert torch.allclose(unit_rot, unit @ rotation.T, atol=1e-10)
            assert torch.allclose(dist_rot, dist, atol=1e-12)

    def test_directional_covariance_float32(self):
        gen = torch.Generator().manual_seed(20261102)
        rotation = random_rotation(gen).to(torch.float32)
        x = torch.randn(200, 3, generator=gen)
        unit, _ = safe_unit_vector(x, dim=-1)
        unit_rot, _ = safe_unit_vector(x @ rotation.T, dim=-1)
        assert torch.allclose(unit_rot, unit @ rotation.T, atol=1e-6)

    def test_safe_norm_rotation_invariant(self):
        gen = torch.Generator().manual_seed(20261103)
        for _ in range(5):
            rotation = random_rotation(gen)
            x = torch.randn(128, 3, generator=gen, dtype=torch.float64)
            assert torch.allclose(
                safe_norm(x @ rotation.T, dim=-1),
                safe_norm(x, dim=-1),
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# Degenerate geometry through the PaiNN blocks
# ---------------------------------------------------------------------------


class TestPaiNNBlocksSingularity:
    def test_interaction_coincident_atoms_finite(self):
        """Exactly coincident atom pairs through PaiNNInteraction: all
        activations and position gradients finite, bounded at the
        documented O(1/eps) scale (the deprecated torch.norm path was
        unbounded/NaN there)."""
        interaction, _ = make_layers(seed=31)
        pos, scalar, vector, edge_index, edge_vec, edge_rbf = make_graph(
            coincident=True, seed=32
        )
        pos = pos.clone().requires_grad_(True)
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        out_s, out_v = interaction(scalar, vector, edge_index, edge_vec, edge_rbf)
        assert bool(torch.isfinite(out_s).all())
        assert bool(torch.isfinite(out_v).all())
        (out_s.sum() + out_v.sum()).backward()
        assert bool(torch.isfinite(pos.grad).all())
        # Bounded: O(1/eps) direction gradient x O(1) gates x O(H) channels,
        # orders of magnitude below the unbounded legacy failure regime.
        assert pos.grad.abs().max().item() < 1e6

    def test_interaction_zero_edge_vectors_finite(self):
        interaction, _ = make_layers(seed=33)
        _, scalar, vector, edge_index, _, edge_rbf = make_graph(seed=34)
        edge_vec = torch.zeros(edge_index.size(1), 3, dtype=torch.float64)
        edge_vec = edge_vec.requires_grad_(True)
        out_s, out_v = interaction(scalar, vector, edge_index, edge_vec, edge_rbf)
        assert bool(torch.isfinite(out_s).all() and torch.isfinite(out_v).all())
        (out_s.sum() + out_v.sum()).backward()
        # Block-level bound: H channels can each contribute |gate|/eps
        # through the direction path, so the documented scale is H/eps
        # (observed ~1.1e5 << 8e5 at default initialisation) — bounded,
        # versus the unbounded/NaN legacy torch.norm failure mode.
        assert x_grad_finite_and_bounded(edge_vec.grad, bound=H / SAFE_NORM_EPS)

    def test_mixing_zero_vector_features_finite(self):
        _, mixing = make_layers(seed=35)
        gen = torch.Generator().manual_seed(36)
        scalar = torch.randn(6, H, generator=gen, dtype=torch.float64)
        vector = torch.zeros(6, 3, H, dtype=torch.float64, requires_grad=True)
        out_s, out_v = mixing(scalar, vector)
        assert bool(torch.isfinite(out_s).all() and torch.isfinite(out_v).all())
        (out_s.sum() + out_v.sum()).backward()
        assert bool(torch.isfinite(vector.grad).all())

    def test_empty_edge_set_is_identity_residual(self):
        interaction, _ = make_layers(seed=37)
        _, scalar, vector, _, _, _ = make_graph(seed=38)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_vec = torch.zeros(0, 3, dtype=torch.float64)
        edge_rbf = torch.zeros(0, R, dtype=torch.float64)
        out_s, out_v = interaction(scalar, vector, edge_index, edge_vec, edge_rbf)
        assert torch.equal(out_s, scalar)
        assert torch.equal(out_v, vector)

    def test_no_nan_replacement_anywhere(self):
        """Fail loudly on genuinely poisoned inputs (#45 §8): no NaN masking."""
        interaction, _ = make_layers(seed=39)
        pos, scalar, vector, edge_index, edge_vec, edge_rbf = make_graph(seed=40)
        scalar = scalar.clone()
        scalar[0, 0] = float("nan")
        out_s, _ = interaction(scalar, vector, edge_index, edge_vec, edge_rbf)
        assert not bool(torch.isfinite(out_s).all())  # poison propagates visibly


# ---------------------------------------------------------------------------
# Equivariance of the PaiNN blocks (random rigid motions)
# ---------------------------------------------------------------------------


class TestPaiNNBlocksEquivariance:
    ROTATIONS = 3
    ATOL_SCALAR = 1e-9
    ATOL_VECTOR = 1e-8

    def test_interaction_scalar_invariant_vector_equivariant(self):
        interaction, _ = make_layers(seed=41)
        pos, scalar, vector, edge_index, edge_vec, edge_rbf = make_graph(seed=42)
        s_out, v_out = interaction(scalar, vector, edge_index, edge_vec, edge_rbf)
        gen = torch.Generator().manual_seed(43)
        for _ in range(self.ROTATIONS):
            rotation = random_rotation(gen)
            pos_rot = pos @ rotation.T
            edge_vec_rot = pos_rot[edge_index[1]] - pos_rot[edge_index[0]]
            vector_rot = rotate_vectors(rotation, vector)
            s_rot, v_rot = interaction(
                scalar, vector_rot, edge_index, edge_vec_rot, edge_rbf
            )
            assert torch.allclose(s_rot, s_out, atol=self.ATOL_SCALAR)
            assert torch.allclose(
                v_rot, rotate_vectors(rotation, v_out), atol=self.ATOL_VECTOR
            )

    def test_interaction_translation_invariant(self):
        interaction, _ = make_layers(seed=44)
        pos, scalar, vector, edge_index, edge_vec, edge_rbf = make_graph(seed=45)
        s_out, v_out = interaction(scalar, vector, edge_index, edge_vec, edge_rbf)
        shift = torch.tensor([[1.5, -2.0, 0.25]], dtype=torch.float64)
        pos_shifted = pos + shift
        edge_vec_shifted = pos_shifted[edge_index[1]] - pos_shifted[edge_index[0]]
        s_shift, v_shift = interaction(
            scalar, vector, edge_index, edge_vec_shifted, edge_rbf
        )
        assert torch.allclose(s_shift, s_out, atol=1e-12)
        assert torch.allclose(v_shift, v_out, atol=1e-12)

    def test_mixing_scalar_invariant_vector_equivariant(self):
        _, mixing = make_layers(seed=46)
        _, scalar, vector, *_ = make_graph(seed=47)
        s_out, v_out = mixing(scalar, vector)
        gen = torch.Generator().manual_seed(48)
        for _ in range(self.ROTATIONS):
            rotation = random_rotation(gen)
            s_rot, v_rot = mixing(scalar, rotate_vectors(rotation, vector))
            assert torch.allclose(s_rot, s_out, atol=self.ATOL_SCALAR)
            assert torch.allclose(
                v_rot, rotate_vectors(rotation, v_out), atol=self.ATOL_VECTOR
            )

    def test_message_then_update_composition_equivariant(self):
        """A full PaiNN layer (message block followed by update block)
        preserves scalar invariance and vector equivariance."""
        interaction, mixing = make_layers(seed=49)
        pos, scalar, vector, edge_index, edge_vec, edge_rbf = make_graph(seed=50)
        s_out, v_out = mixing(*interaction(scalar, vector, edge_index, edge_vec, edge_rbf))
        gen = torch.Generator().manual_seed(51)
        for _ in range(self.ROTATIONS):
            rotation = random_rotation(gen)
            pos_rot = pos @ rotation.T
            edge_vec_rot = pos_rot[edge_index[1]] - pos_rot[edge_index[0]]
            s_rot, v_rot = mixing(
                *interaction(
                    scalar,
                    rotate_vectors(rotation, vector),
                    edge_index,
                    edge_vec_rot,
                    edge_rbf,
                )
            )
            assert torch.allclose(s_rot, s_out, atol=self.ATOL_SCALAR)
            assert torch.allclose(
                v_rot, rotate_vectors(rotation, v_out), atol=self.ATOL_VECTOR
            )


# ---------------------------------------------------------------------------
# Fail-closed shape contracts of the blocks
# ---------------------------------------------------------------------------


class TestBlockShapeContracts:
    def test_interaction_rejects_malformed_inputs(self):
        interaction, _ = make_layers(seed=61)
        pos, scalar, vector, edge_index, edge_vec, edge_rbf = make_graph(seed=62)
        with pytest.raises(ValueError, match=r"vector"):
            interaction(scalar, vector.reshape(7, -1), edge_index, edge_vec, edge_rbf)
        with pytest.raises(ValueError, match=r"edge_vec"):
            interaction(scalar, vector, edge_index, edge_vec[:, :2], edge_rbf)
        with pytest.raises(ValueError, match=r"edge_rbf"):
            interaction(scalar, vector, edge_index, edge_vec, edge_rbf[:, :-1])
        bad_index = edge_index.clone()
        bad_index[0, 0] = 999  # out of range
        with pytest.raises(ValueError, match=r"edge_index"):
            interaction(scalar, vector, bad_index, edge_vec, edge_rbf)

    def test_mixing_rejects_malformed_inputs(self):
        _, mixing = make_layers(seed=63)
        _, scalar, vector, *_ = make_graph(seed=64)
        with pytest.raises(ValueError):
            mixing(scalar, vector.reshape(7, -1))
        with pytest.raises(ValueError):
            mixing(scalar.unsqueeze(0), vector)

    def test_constructor_validates_dimensions(self):
        with pytest.raises(ValueError):
            PaiNNInteraction(0, R)
        with pytest.raises(ValueError):
            PaiNNInteraction(H, 0)
        with pytest.raises(ValueError):
            PaiNNMixing(0)
        with pytest.raises(ValueError):
            PaiNNMixing(H, epsilon=0.0)
