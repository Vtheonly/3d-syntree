"""Unit tests for the equivariant backbone (PaiNN layers, RBF, radius graph)."""

from __future__ import annotations

import math

import pytest
import torch

from syntree.models.equivariant import (
    BatchedPaiNNLayer,
    CosineCutoff,
    PaiNNLayer,
    PaiNNMixing,
    RadialBasis,
    build_radius_graph,
)


class TestCosineCutoff:
    def test_zero_at_origin(self):
        fn = CosineCutoff(5.0)
        assert fn(torch.tensor(0.0)).item() == pytest.approx(1.0)

    def test_zero_beyond_cutoff(self):
        fn = CosineCutoff(5.0)
        assert fn(torch.tensor(5.0)).item() == pytest.approx(0.0)
        assert fn(torch.tensor(7.0)).item() == 0.0

    def test_monotone(self):
        fn = CosineCutoff(5.0)
        d = torch.linspace(0, 5, 50)
        vals = fn(d)
        assert (vals[1:] <= vals[:-1] + 1e-7).all()

    def test_bad_cutoff_raises(self):
        with pytest.raises(ValueError):
            CosineCutoff(0.0)


class TestRadialBasis:
    def test_output_shape(self):
        rbf = RadialBasis(num_radial=20, cutoff=5.0)
        out = rbf(torch.rand(100) * 5)
        assert out.shape == (100, 20)

    def test_zero_beyond_cutoff(self):
        rbf = RadialBasis(num_radial=20, cutoff=5.0)
        out = rbf(torch.tensor([6.0, 10.0]))
        assert out.abs().sum() == 0.0

    def test_2d_input_raises(self):
        rbf = RadialBasis(num_radial=20, cutoff=5.0)
        with pytest.raises(ValueError):
            rbf(torch.rand(10, 10))

    def test_bad_num_radial_raises(self):
        with pytest.raises(ValueError):
            RadialBasis(num_radial=0)

    def test_orthogonality_ish(self):
        """Distinct distances activate distinct basis peaks."""
        rbf = RadialBasis(num_radial=10, cutoff=5.0)
        out = rbf(torch.tensor([0.5, 2.5, 4.5]))
        peaks = out.argmax(dim=1)
        assert peaks[0] < peaks[1] < peaks[2]


class TestRadiusGraph:
    def test_no_cross_graph_edges(self):
        pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [50.0, 0, 0], [50.5, 0, 0]])
        batch = torch.tensor([0, 0, 1, 1])
        ei = build_radius_graph(pos, batch, cutoff=5.0)
        for a, b in ei.t().tolist():
            assert batch[a] == batch[b]

    def test_self_edges_excluded(self):
        pos = torch.randn(10, 3)
        ei = build_radius_graph(pos, None, cutoff=100.0)
        assert not (ei[0] == ei[1]).any()

    def test_self_edges_included_when_loop(self):
        pos = torch.randn(5, 3)
        ei = build_radius_graph(pos, None, cutoff=100.0, loop=True)
        assert (ei[0] == ei[1]).sum() == 5

    def test_symmetry(self):
        pos = torch.randn(20, 3)
        ei = build_radius_graph(pos, None, cutoff=3.0)
        pairs = set(map(tuple, ei.t().tolist()))
        for a, b in pairs:
            assert (b, a) in pairs

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            build_radius_graph(torch.rand(5, 2), None, cutoff=1.0)


class TestPaiNNLayers:
    @pytest.fixture()
    def graph(self):
        torch.manual_seed(0)
        pos = torch.randn(15, 3)
        batch = torch.zeros(15, dtype=torch.long)
        ei = build_radius_graph(pos, batch, cutoff=5.0)
        row, col = ei
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1)
        rbf = RadialBasis(16, 5.0)
        edge_rbf = rbf(edge_dist)
        return ei, edge_vec, edge_dist, edge_rbf

    def test_shapes_batched(self, graph):
        ei, edge_vec, edge_dist, edge_rbf = graph
        layer = BatchedPaiNNLayer(hidden_dim=32, num_radial=16)
        s = torch.randn(15, 32)
        v = torch.randn(15, 3, 32)
        s2, v2 = layer(s, v, ei, edge_vec, edge_dist, edge_rbf)
        assert s2.shape == (15, 32)
        assert v2.shape == (15, 3, 32)

    def test_shapes_legacy(self, graph):
        ei, _, _, _ = graph
        layer = PaiNNLayer(hidden_dim=32, num_radial=16, cutoff=5.0)
        pos = torch.randn(15, 3)
        s = torch.randn(15, 32)
        v = torch.randn(15, 3, 32)
        s2, v2 = layer(s, v, ei, pos)
        assert s2.shape == (15, 32)
        assert v2.shape == (15, 3, 32)

    def test_gradients_flow(self, graph):
        ei, edge_vec, edge_dist, edge_rbf = graph
        layer = BatchedPaiNNLayer(hidden_dim=32, num_radial=16)
        s = torch.randn(15, 32, requires_grad=True)
        v = torch.randn(15, 3, 32, requires_grad=True)
        s2, v2 = layer(s, v, ei, edge_vec, edge_dist, edge_rbf)
        (s2.sum() + v2.sum()).backward()
        assert s.grad is not None and s.grad.abs().sum() > 0
        assert v.grad is not None and v.grad.abs().sum() > 0

    def test_bad_vector_shape_raises(self, graph):
        ei, edge_vec, edge_dist, edge_rbf = graph
        layer = BatchedPaiNNLayer(hidden_dim=32, num_radial=16)
        s = torch.randn(15, 32)
        v = torch.randn(15, 2, 32)  # wrong spatial dim
        with pytest.raises(ValueError):
            layer(s, v, ei, edge_vec, edge_dist, edge_rbf)


class TestEquivariance:
    """The core mathematical property: E(3)-equivariance of PaiNN."""

    @pytest.fixture()
    def layer(self):
        torch.manual_seed(7)
        return BatchedPaiNNLayer(hidden_dim=24, num_radial=12)

    @pytest.fixture()
    def inputs(self):
        torch.manual_seed(3)
        pos = torch.randn(12, 3)
        batch = torch.zeros(12, dtype=torch.long)
        ei = build_radius_graph(pos, batch, cutoff=10.0)  # dense: stable edges
        row, col = ei
        edge_vec = pos[col] - pos[row]
        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_rbf = RadialBasis(12, 10.0)(edge_dist)
        return pos, batch, ei, edge_vec, edge_dist, edge_rbf

    @staticmethod
    def _rot(theta: float, axis: int = 2) -> torch.Tensor:
        c, s = math.cos(theta), math.sin(theta)
        R = torch.eye(3)
        i, j = (axis + 1) % 3, (axis + 2) % 3
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        return R

    def test_scalar_invariance_under_rotation(self, layer, inputs):
        pos, batch, ei, edge_vec, edge_dist, edge_rbf = inputs
        s = torch.randn(12, 24)
        v = torch.randn(12, 3, 24)
        layer.eval()
        R = self._rot(0.9)
        with torch.no_grad():
            s1, _ = layer(s, v, ei, edge_vec, edge_dist, edge_rbf)
            pos2 = pos @ R.T
            ei2 = build_radius_graph(pos2, batch, cutoff=10.0)
            row, col = ei2
            ev2 = pos2[col] - pos2[row]
            ed2 = torch.norm(ev2, dim=-1)
            er2 = RadialBasis(12, 10.0)(ed2)
            # Equivariance scenario: BOTH positions and vector inputs rotate.
            v_rot = torch.einsum("nkh,sk->nsh", v, R)
            s2, _ = layer(s, v_rot, ei2, ev2, ed2, er2)
        assert torch.allclose(s1, s2, atol=1e-5), (s1 - s2).abs().max()

    def test_vector_equivariance_under_rotation(self, layer, inputs):
        pos, batch, ei, edge_vec, edge_dist, edge_rbf = inputs
        s = torch.randn(12, 24)
        v = torch.randn(12, 3, 24)
        layer.eval()
        R = self._rot(1.3, axis=0)
        with torch.no_grad():
            _, v1 = layer(s, v, ei, edge_vec, edge_dist, edge_rbf)
            pos2 = pos @ R.T
            ei2 = build_radius_graph(pos2, batch, cutoff=10.0)
            row, col = ei2
            ev2 = pos2[col] - pos2[row]
            ed2 = torch.norm(ev2, dim=-1)
            er2 = RadialBasis(12, 10.0)(ed2)
            # Equivariance scenario: rotated positions AND rotated vectors in.
            v_rot = torch.einsum("nkh,sk->nsh", v, R)
            _, v2 = layer(s, v_rot, ei2, ev2, ed2, er2)
            # v2 must equal v1 rotated: v1 @ R.T (spatial index at dim 1)
            expected = torch.einsum("nkh,sk->nsh", v1, R)
        assert torch.allclose(v2, expected, atol=1e-4), (v2 - expected).abs().max()

    def test_translation_invariance(self, layer, inputs):
        pos, batch, ei, edge_vec, edge_dist, edge_rbf = inputs
        s = torch.randn(12, 24)
        v = torch.randn(12, 3, 24)
        layer.eval()
        t = torch.tensor([3.0, -2.0, 7.0])
        with torch.no_grad():
            s1, v1 = layer(s, v, ei, edge_vec, edge_dist, edge_rbf)
            pos2 = pos + t
            ei2 = build_radius_graph(pos2, batch, cutoff=10.0)
            row, col = ei2
            ev2 = pos2[col] - pos2[row]
            ed2 = torch.norm(ev2, dim=-1)
            er2 = RadialBasis(12, 10.0)(ed2)
            s2, v2 = layer(s, v, ei2, ev2, ed2, er2)
        assert torch.allclose(s1, s2, atol=1e-5)
        assert torch.allclose(v1, v2, atol=1e-5)


class TestPaiNNMixing:
    def test_scalar_invariance(self):
        torch.manual_seed(1)
        mix = PaiNNMixing(hidden_dim=16)
        mix.eval()
        s = torch.randn(6, 16)
        v = torch.randn(6, 3, 16)
        R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with torch.no_grad():
            s1, v1 = mix(s, v)
            s2, v2 = mix(s, torch.einsum("nkh,sk->nsh", v, R))
        assert torch.allclose(s1, s2, atol=1e-6)
        assert torch.allclose(
            v2, torch.einsum("nkh,sk->nsh", v1, R), atol=1e-5
        )
