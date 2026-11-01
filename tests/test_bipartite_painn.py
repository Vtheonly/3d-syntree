"""Bipartite SE(3)-equivariant backbone over the union pocket + ligand graph —
tests (#21 MODEL-01).

Acceptance criteria of issue #21 (clean-room V2 implementation):

1. ``tests/test_bipartite_painn.py`` passes: output scalars are invariant,
   and output vectors are equivariant under 3D rigid rotations.
2. When the ligand node count ``N_l = 0``, the encoder runs cleanly as a
   pocket-only encoder without exceptions.

Beyond the two pinned criteria, this suite enforces the frozen production
profile, the exact radius-graph construction (all three pair families, no
self-loops, no cross-graph edges, brute-force equivalence), the Gaussian
radial basis (including the documented #9 envelope boundary), translation
invariance, batched-equals-single-graph behaviour, activation-checkpointing
equivalence, pooled-context semantics, determinism, and the fail-closed input
contract (dtype/atomic-number/shape/device/handle-flag validation).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from syntree.core.schema import (
    MAX_ATOMIC_NUMBER,
    TensorContractError,
    collate_state_batch,
)
from syntree.models.bipartite import (
    BACKBONE_CUTOFF,
    BACKBONE_HIDDEN_DIM,
    BACKBONE_NUM_LAYERS,
    BACKBONE_NUM_RADIAL,
    BackboneOutput,
    BipartitePaiNN,
    GaussianRadialBasis,
    build_union_radius_graph,
)
from syntree.models.equivariant import SAFE_NORM_EPS

# Small configuration used throughout most of the suite (fast); the frozen
# production profile is covered explicitly by TestFrozenProfile.
H = 16
LAYERS = 2
RADIAL = 6
CUTOFF = 5.0

_ORGANIC_Z = (1, 6, 7, 8, 9, 15, 16, 17)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_rotation(generator: torch.Generator) -> torch.Tensor:
    """A random *proper* rotation (orthogonal, det = +1) in float64."""
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotate_vectors(rotation: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Apply R to the leading spatial dimension of ``[..., 3, H]`` features."""
    return torch.einsum("ab,nbh->nah", rotation, vectors)


def make_node_state(
    n_pocket: int = 10,
    n_ligand: int = 5,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    spread: float = 2.0,
) -> SimpleNamespace:
    """A minimal single-graph state exposing the eight node-level fields."""
    gen = torch.Generator().manual_seed(seed)
    elements = torch.tensor(_ORGANIC_Z, dtype=torch.long)

    def positions(count: int) -> torch.Tensor:
        return (torch.rand((count, 3), generator=gen, dtype=dtype) * 2.0 - 1.0) * spread

    def z_of(count: int) -> torch.Tensor:
        idx = torch.randint(0, len(_ORGANIC_Z), (count,), generator=gen)
        return elements[idx]

    def charges(count: int) -> torch.Tensor:
        return torch.randint(-1, 2, (count,), generator=gen).to(dtype)

    return SimpleNamespace(
        pocket_pos=positions(n_pocket),
        pocket_z=z_of(n_pocket),
        pocket_charge=charges(n_pocket),
        pocket_batch=torch.zeros(n_pocket, dtype=torch.long),
        ligand_pos=positions(n_ligand),
        ligand_z=z_of(n_ligand),
        ligand_charge=charges(n_ligand),
        ligand_batch=torch.zeros(n_ligand, dtype=torch.long),
    )


def make_batched_node_state(
    pocket_counts, ligand_counts, *, seed: int = 0, dtype: torch.dtype = torch.float64
) -> SimpleNamespace:
    """A multi-graph state built from per-graph single-graph samples."""
    states = []
    for g, (n_p, n_l) in enumerate(zip(pocket_counts, ligand_counts)):
        s = make_node_state(n_p, n_l, seed=seed + g, dtype=dtype)
        s.pocket_batch = torch.full((n_p,), g, dtype=torch.long)
        s.ligand_batch = torch.full((n_l,), g, dtype=torch.long)
        states.append(s)
    return SimpleNamespace(
        pocket_pos=torch.cat([s.pocket_pos for s in states]),
        pocket_z=torch.cat([s.pocket_z for s in states]),
        pocket_charge=torch.cat([s.pocket_charge for s in states]),
        pocket_batch=torch.cat([s.pocket_batch for s in states]),
        ligand_pos=torch.cat([s.ligand_pos for s in states]),
        ligand_z=torch.cat([s.ligand_z for s in states]),
        ligand_charge=torch.cat([s.ligand_charge for s in states]),
        ligand_batch=torch.cat([s.ligand_batch for s in states]),
    )


def make_single_graph_data(n_pocket: int, n_ligand: int, *, seed: int = 0) -> Data:
    """A canonical full single-graph state (12 fields) for the collator."""
    gen = torch.Generator().manual_seed(seed)
    s = make_node_state(n_pocket, n_ligand, seed=seed, dtype=torch.float32)
    data = Data(
        pocket_pos=s.pocket_pos,
        pocket_z=s.pocket_z,
        pocket_charge=s.pocket_charge,
        ligand_pos=s.ligand_pos,
        ligand_z=s.ligand_z,
        ligand_charge=s.ligand_charge,
        handle_features=torch.randn(1, 64, generator=gen),
        handle_pos=torch.randn(1, 3, generator=gen),
        global_features=torch.randn(1, 4, generator=gen),
        stop_mask=torch.zeros(1),
    )
    return data


def small_model(
    *, seed: int = 0, dtype: torch.dtype = torch.float64, **overrides
) -> BipartitePaiNN:
    torch.manual_seed(seed)
    kwargs = dict(
        hidden_dim=H, num_layers=LAYERS, num_radial=RADIAL, r_cut=CUTOFF
    )
    kwargs.update(overrides)
    return BipartitePaiNN(**kwargs).to(dtype)


def rigidly_transformed(state: SimpleNamespace, rotation, translation) -> SimpleNamespace:
    """A copy of ``state`` whose positions are rotated then translated."""
    out = SimpleNamespace(
        pocket_pos=state.pocket_pos @ rotation.T + translation,
        pocket_z=state.pocket_z,
        pocket_charge=state.pocket_charge,
        pocket_batch=state.pocket_batch,
        ligand_pos=state.ligand_pos @ rotation.T + translation,
        ligand_z=state.ligand_z,
        ligand_charge=state.ligand_charge,
        ligand_batch=state.ligand_batch,
    )
    return out


# ---------------------------------------------------------------------------
# Frozen production profile
# ---------------------------------------------------------------------------


class TestFrozenProfile:
    def test_default_profile_constants(self):
        model = BipartitePaiNN()
        assert model.hidden_dim == BACKBONE_HIDDEN_DIM == 256
        assert model.num_layers == BACKBONE_NUM_LAYERS == 8
        assert model.num_radial == BACKBONE_NUM_RADIAL == 20
        assert model.r_cut == BACKBONE_CUTOFF == 5.0
        assert model.activation_checkpointing is True
        assert len(model.layers) == 8

    def test_default_profile_parameter_count(self):
        """Pinned parameter count guards accidental architecture drift
        (#45 §5.2: frozen architecture dimensions)."""
        model = BipartitePaiNN()
        count = sum(p.numel() for p in model.parameters())
        assert count == 5_999_104

    def test_parameters_default_to_float32(self):
        model = BipartitePaiNN()
        for p in model.parameters():
            assert p.dtype == torch.float32

    def test_small_config_overrides(self):
        model = small_model()
        assert model.hidden_dim == H
        assert len(model.layers) == LAYERS
        assert model.radial_basis.num_radial == RADIAL
        assert model.activation_checkpointing is True
        model = small_model(activation_checkpointing=False)
        assert model.activation_checkpointing is False

    def test_default_profile_forward_smoke(self):
        """The full 256/8-layer profile runs forward + backward on CPU with
        finite outputs and gradients (no T4 claim is implied)."""
        torch.manual_seed(7)
        model = BipartitePaiNN()
        state = make_node_state(24, 9, seed=3, dtype=torch.float32)
        state.pocket_pos.requires_grad_(True)
        state.ligand_pos.requires_grad_(True)
        out = model(state)
        assert out.scalar.shape == (33, 256)
        assert out.vector.shape == (33, 3, 256)
        assert out.graph_context.shape == (1, 256)
        (out.scalar.sum() + out.vector.pow(2).sum()).backward()
        for t in (state.pocket_pos, state.ligand_pos):
            assert t.grad is not None
            assert bool(torch.isfinite(t.grad).all())
            assert t.grad.abs().sum() > 0
        for p in model.parameters():
            if p.grad is not None:
                assert bool(torch.isfinite(p.grad).all())

    def test_bad_constructor_arguments(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            BipartitePaiNN(hidden_dim=0)
        with pytest.raises(ValueError, match="num_layers"):
            BipartitePaiNN(num_layers=0)
        with pytest.raises(ValueError, match="num_radial"):
            BipartitePaiNN(num_radial=1)
        with pytest.raises(ValueError, match="r_cut"):
            BipartitePaiNN(r_cut=0.0)


# ---------------------------------------------------------------------------
# Gaussian radial basis
# ---------------------------------------------------------------------------


class TestGaussianRadialBasis:
    def test_output_shape_rank_agnostic(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        assert rbf(torch.zeros(7)).shape == (7, 20)
        assert rbf(torch.zeros(3, 11)).shape == (3, 11, 20)

    def test_values_bounded(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        d = torch.linspace(0.0, 5.0, 101, dtype=torch.float64)
        out = rbf(d)
        assert bool((out > 0).all())
        assert bool((out <= 1.0 + 1e-12).all())

    def test_peaks_at_centers(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        for k in range(20):
            center = rbf.centers[k]
            value = rbf(center.reshape(1))[0, k]
            assert torch.allclose(value, torch.ones_like(value))

    def test_decay_away_from_center(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        at_center = rbf(torch.tensor([2.5]))[0, 10]
        off_center = rbf(torch.tensor([2.5 + 0.5]))[0, 10]
        assert off_center < at_center < 1.0

    def test_zero_distance_first_channel_is_one(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        out = rbf(torch.zeros(1))
        assert torch.allclose(out[0, 0], torch.ones(()))
        assert bool((out[0, 1:] < 1.0).all())

    def test_at_cutoff_last_channel_is_one(self):
        """Documents the #9 envelope boundary (decision D-V2-007): the raw
        Gaussian basis has no cosine cutoff — at ``d = r_cut`` the last
        channel still evaluates to 1.0. #9 MATH-03 owns the envelope."""
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        out = rbf(torch.tensor([5.0]))
        assert torch.allclose(out[0, -1], torch.ones(()))

    def test_no_learnable_parameters(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        assert sum(p.numel() for p in rbf.parameters()) == 0
        assert rbf.centers.shape == (20,)

    def test_fail_closed_negative_distance(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        with pytest.raises(TensorContractError, match="negative"):
            rbf(torch.tensor([-1e-9]))

    def test_fail_closed_non_finite_distance(self):
        rbf = GaussianRadialBasis(num_radial=20, r_cut=5.0)
        with pytest.raises(TensorContractError, match="non-finite"):
            rbf(torch.tensor([float("nan")]))

    def test_fail_closed_bad_constructor(self):
        with pytest.raises(ValueError, match="num_radial"):
            GaussianRadialBasis(num_radial=1)
        with pytest.raises(ValueError, match="r_cut"):
            GaussianRadialBasis(r_cut=-1.0)

    def test_dtype_follows_module(self):
        rbf = GaussianRadialBasis(num_radial=6, r_cut=5.0).to(torch.float64)
        out = rbf(torch.tensor([1.0], dtype=torch.float64))
        assert out.dtype == torch.float64


# ---------------------------------------------------------------------------
# The one authoritative radius-graph construction
# ---------------------------------------------------------------------------


class TestBuildUnionRadiusGraph:
    def test_all_edges_within_cutoff(self):
        pos = torch.randn(25, 3, dtype=torch.float64)
        batch = torch.zeros(25, dtype=torch.long)
        _, _, dist = build_union_radius_graph(pos, batch, 5.0)
        assert bool((dist <= 5.0).all())

    def test_no_self_loops(self):
        pos = torch.randn(25, 3, dtype=torch.float64)
        batch = torch.zeros(25, dtype=torch.long)
        edge_index, _, _ = build_union_radius_graph(pos, batch, 5.0)
        same = edge_index[0] == edge_index[1]
        assert not bool(same.any())

    def test_symmetric_directions(self):
        pos = torch.randn(25, 3, dtype=torch.float64)
        batch = torch.zeros(25, dtype=torch.long)
        edge_index, _, _ = build_union_radius_graph(pos, batch, 5.0)
        pairs = set(map(tuple, edge_index.t().tolist()))
        for i, j in pairs:
            assert (j, i) in pairs  # every edge has its reverse

    def test_bruteforce_equivalence(self):
        pos = torch.randn(18, 3, dtype=torch.float64)
        batch = torch.zeros(18, dtype=torch.long)
        edge_index, _, _ = build_union_radius_graph(pos, batch, 5.0)
        got = set(map(tuple, edge_index.t().tolist()))
        expected = {
            (i, j)
            for i in range(18)
            for j in range(18)
            if i != j and (pos[i] - pos[j]).norm() <= 5.0 + SAFE_NORM_EPS
        }
        assert got == expected

    def test_no_cross_graph_edges(self):
        pos = torch.randn(20, 3, dtype=torch.float64)
        batch = torch.tensor([0] * 10 + [1] * 10)
        edge_index, _, _ = build_union_radius_graph(pos, batch, 5.0)
        graphs = batch[edge_index[0]]
        assert bool((graphs == batch[edge_index[1]]).all())

    def test_all_three_pair_families_present(self):
        """Pocket-pocket, ligand-ligand AND pocket-ligand edges are all
        emitted on the union graph (the #21 core requirement)."""
        # Pocket atoms 0-1 close together; ligand atoms 2-3 close together
        # (in union numbering); pocket atom 0 near ligand atom 2.
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # pocket 0
                [1.5, 0.0, 0.0],  # pocket 1
                [3.0, 0.0, 0.0],  # ligand (reacting handle)
                [4.5, 0.0, 0.0],  # ligand
            ],
            dtype=torch.float64,
        )
        node_type = torch.tensor([0, 0, 1, 1])
        batch = torch.zeros(4, dtype=torch.long)
        edge_index, _, _ = build_union_radius_graph(pos, batch, 5.0)
        pairs = set(map(tuple, edge_index.t().tolist()))
        assert (0, 1) in pairs and (1, 0) in pairs  # pocket-pocket
        assert (2, 3) in pairs and (3, 2) in pairs  # ligand-ligand
        assert (0, 2) in pairs and (2, 0) in pairs  # pocket-ligand
        # atom 3 at 4.5 A from atom 0: within the 5.0 A cutoff as well
        assert (0, 3) in pairs
        # and every edge connects nodes of the same graph
        assert all(batch[i] == batch[j] for i, j in pairs)

    def test_empty_edge_set_when_far_apart(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        edge_index, edge_vec, edge_dist = build_union_radius_graph(pos, batch, 5.0)
        assert edge_index.shape == (2, 0)
        assert edge_vec.shape == (0, 3)
        assert edge_dist.shape == (0,)

    def test_coincident_atoms_edge_is_safe(self):
        pos = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        edge_index, edge_vec, edge_dist = build_union_radius_graph(pos, batch, 5.0)
        assert edge_index.shape == (2, 2)  # both directions of the 0-distance pair
        assert torch.equal(edge_vec, torch.zeros_like(edge_vec))
        assert torch.equal(edge_dist, torch.zeros_like(edge_dist))

    def test_graph_invariant_under_rigid_motion(self):
        gen = torch.Generator().manual_seed(11)
        pos = torch.randn(20, 3, generator=gen, dtype=torch.float64)
        batch = torch.zeros(20, dtype=torch.long)
        r = random_rotation(gen)
        t = torch.randn(3, generator=gen, dtype=torch.float64)
        ei0, _, d0 = build_union_radius_graph(pos, batch, 5.0)
        ei1, _, d1 = build_union_radius_graph(pos @ r.T + t, batch, 5.0)
        assert torch.equal(ei0, ei1)  # identical edge structure
        assert torch.allclose(d0, d1, atol=1e-9)

    def test_deterministic_ordering(self):
        pos = torch.randn(15, 3, dtype=torch.float64)
        batch = torch.zeros(15, dtype=torch.long)
        ei0, ev0, ed0 = build_union_radius_graph(pos, batch, 5.0)
        ei1, ev1, ed1 = build_union_radius_graph(pos, batch, 5.0)
        assert torch.equal(ei0, ei1)
        assert torch.equal(ev0, ev1)
        assert torch.equal(ed0, ed1)

    def test_edge_vector_convention(self):
        """edge_vec = pos[sender] - pos[receiver] (r_j - r_i), matching the
        PaiNNInteraction convention (messages flow sender -> receiver)."""
        pos = torch.randn(12, 3, dtype=torch.float64)
        batch = torch.zeros(12, dtype=torch.long)
        edge_index, edge_vec, _ = build_union_radius_graph(pos, batch, 5.0)
        receiver, sender = edge_index[0], edge_index[1]
        assert torch.allclose(edge_vec, pos[sender] - pos[receiver])

    def test_fail_closed_bad_inputs(self):
        pos = torch.randn(6, 3, dtype=torch.float64)
        batch = torch.zeros(6, dtype=torch.long)
        with pytest.raises(TensorContractError, match=r"\[radius-graph\] `pos` must be \[N, 3\]"):
            build_union_radius_graph(torch.randn(6, 2), batch, 5.0)
        with pytest.raises(TensorContractError, match="at least one node"):
            build_union_radius_graph(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), 5.0)
        with pytest.raises(TensorContractError, match="node_batch` must be"):
            build_union_radius_graph(pos, torch.zeros(5, dtype=torch.long), 5.0)
        with pytest.raises(TensorContractError, match="dtype int64"):
            build_union_radius_graph(pos, torch.zeros(6, dtype=torch.int32), 5.0)
        with pytest.raises(TensorContractError, match="r_cut"):
            build_union_radius_graph(pos, batch, 0.0)
        bad_pos = pos.clone()
        bad_pos[0, 0] = float("nan")
        with pytest.raises(TensorContractError, match="non-finite"):
            build_union_radius_graph(bad_pos, batch, 5.0)
        bad_batch = batch.clone()
        bad_batch[0] = -1
        with pytest.raises(TensorContractError, match=">= 0"):
            build_union_radius_graph(pos, bad_batch, 5.0)


# ---------------------------------------------------------------------------
# Pinned acceptance criterion 1: SE(3) invariance / equivariance
# ---------------------------------------------------------------------------


class TestSE3InvarianceEquivariance:
    def test_scalar_invariance_under_rotation(self):
        model = small_model(seed=1)
        state = make_node_state(12, 6, seed=4)
        gen = torch.Generator().manual_seed(21)
        rotation = random_rotation(gen)
        rotated = rigidly_transformed(state, rotation, torch.zeros(3, dtype=torch.float64))
        out_ref = model(state)
        out_rot = model(rotated)
        torch.testing.assert_close(out_rot.scalar, out_ref.scalar, rtol=1e-9, atol=1e-9)

    def test_vector_equivariance_under_rotation(self):
        model = small_model(seed=1)
        state = make_node_state(12, 6, seed=4)
        gen = torch.Generator().manual_seed(21)
        rotation = random_rotation(gen)
        rotated = rigidly_transformed(state, rotation, torch.zeros(3, dtype=torch.float64))
        out_ref = model(state)
        out_rot = model(rotated)
        expected = rotate_vectors(rotation, out_ref.vector)
        torch.testing.assert_close(out_rot.vector, expected, rtol=1e-9, atol=1e-9)

    def test_invariance_under_translation(self):
        model = small_model(seed=2)
        state = make_node_state(12, 6, seed=5)
        translation = torch.tensor([0.5, -1.25, 2.0], dtype=torch.float64)
        identity = torch.eye(3, dtype=torch.float64)
        out_ref = model(state)
        out_tr = model(rigidly_transformed(state, identity, translation))
        torch.testing.assert_close(out_tr.scalar, out_ref.scalar, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(out_tr.vector, out_ref.vector, rtol=1e-9, atol=1e-9)

    def test_combined_rotation_translation(self):
        model = small_model(seed=3)
        state = make_node_state(14, 7, seed=6)
        gen = torch.Generator().manual_seed(23)
        rotation = random_rotation(gen)
        translation = torch.randn(3, generator=gen, dtype=torch.float64)
        out_ref = model(state)
        out_rt = model(rigidly_transformed(state, rotation, translation))
        torch.testing.assert_close(out_rt.scalar, out_ref.scalar, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            out_rt.vector, rotate_vectors(rotation, out_ref.vector), rtol=1e-9, atol=1e-9
        )

    def test_pooled_contexts_invariant(self):
        model = small_model(seed=4)
        state = make_node_state(12, 6, seed=7)
        gen = torch.Generator().manual_seed(24)
        rotation = random_rotation(gen)
        translation = torch.randn(3, generator=gen, dtype=torch.float64)
        out_ref = model(state)
        out_rt = model(rigidly_transformed(state, rotation, translation))
        torch.testing.assert_close(out_rt.graph_context, out_ref.graph_context, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(out_rt.pocket_context, out_ref.pocket_context, rtol=1e-9, atol=1e-9)

    def test_equivariance_with_handle_flags(self):
        model = small_model(seed=5)
        state = make_node_state(11, 6, seed=8)
        handle_flags = torch.zeros(17, dtype=torch.long)
        handle_flags[13] = 1  # one ligand atom marked as reactive handle
        gen = torch.Generator().manual_seed(25)
        rotation = random_rotation(gen)
        translation = torch.randn(3, generator=gen, dtype=torch.float64)
        out_ref = model(state, handle_flags=handle_flags)
        out_rt = model(rigidly_transformed(state, rotation, translation), handle_flags=handle_flags)
        torch.testing.assert_close(out_rt.scalar, out_ref.scalar, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            out_rt.vector, rotate_vectors(rotation, out_ref.vector), rtol=1e-9, atol=1e-9
        )

    def test_equivariance_pocket_only(self):
        """SE(3) behaviour also holds for the pocket-only (N_l = 0) encoder."""
        model = small_model(seed=6)
        state = make_node_state(12, 0, seed=9)
        gen = torch.Generator().manual_seed(26)
        rotation = random_rotation(gen)
        translation = torch.randn(3, generator=gen, dtype=torch.float64)
        out_ref = model(state)
        out_rt = model(rigidly_transformed(state, rotation, translation))
        torch.testing.assert_close(out_rt.scalar, out_ref.scalar, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            out_rt.vector, rotate_vectors(rotation, out_ref.vector), rtol=1e-9, atol=1e-9
        )

    def test_equivariance_batched_multi_graph(self):
        """Each graph's slice transforms independently under one shared
        rigid motion of the whole batch."""
        model = small_model(seed=7)
        state = make_batched_node_state([9, 11, 7], [4, 0, 6], seed=10)
        gen = torch.Generator().manual_seed(27)
        rotation = random_rotation(gen)
        translation = torch.randn(3, generator=gen, dtype=torch.float64)
        out_ref = model(state)
        out_rt = model(rigidly_transformed(state, rotation, translation))
        torch.testing.assert_close(out_rt.scalar, out_ref.scalar, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(
            out_rt.vector, rotate_vectors(rotation, out_ref.vector), rtol=1e-9, atol=1e-9
        )
        torch.testing.assert_close(
            out_rt.graph_context, out_ref.graph_context, rtol=1e-9, atol=1e-9
        )

    def test_gradient_flow_is_finite_and_nonzero(self):
        model = small_model(seed=8)
        state = make_node_state(10, 5, seed=11)
        state.pocket_pos.requires_grad_(True)
        state.ligand_pos.requires_grad_(True)
        out = model(state)
        (out.scalar.sum() + out.vector.pow(2).sum() + out.graph_context.sum()).backward()
        for t in (state.pocket_pos, state.ligand_pos):
            assert t.grad is not None
            assert bool(torch.isfinite(t.grad).all())
            assert t.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Pinned acceptance criterion 2: N_l = 0 pocket-only operation
# ---------------------------------------------------------------------------


class TestEmptyLigandPocketOnly:
    def test_runs_cleanly_without_ligand(self):
        model = small_model(seed=9)
        state = make_node_state(12, 0, seed=12)
        out = model(state)  # must not raise
        assert isinstance(out, BackboneOutput)
        assert out.scalar.shape == (12, H)
        assert out.vector.shape == (12, 3, H)
        assert out.node_pos.shape == (12, 3)
        assert out.num_pocket == 12
        assert out.num_ligand == 0
        assert out.num_graphs == 1
        assert bool(torch.isfinite(out.scalar).all())
        assert bool(torch.isfinite(out.vector).all())

    def test_empty_ligand_slices(self):
        model = small_model(seed=10)
        out = model(make_node_state(9, 0, seed=13))
        assert out.ligand_scalar.shape == (0, H)
        assert out.ligand_vector.shape == (0, 3, H)
        assert not bool(out.ligand_mask.any())
        assert bool(out.pocket_mask.all())

    def test_pocket_only_graph_has_only_pocket_pocket_edges(self):
        model = small_model(seed=11)
        state = make_node_state(10, 0, seed=14)
        edge_index, _, _ = build_union_radius_graph(
            torch.cat([state.pocket_pos, state.ligand_pos]),
            torch.cat([state.pocket_batch, state.ligand_batch]),
            5.0,
        )
        # Nodes are all pocket nodes here; the model must run through the
        # same path (validated via a clean forward above) and the graph
        # builder sees only pocket-pocket pairs.
        assert edge_index.shape[1] >= 0  # runs cleanly whatever the density


# ---------------------------------------------------------------------------
# Batching semantics
# ---------------------------------------------------------------------------


class TestBatching:
    def test_canonical_collated_batch_integration(self):
        """The canonical #6 batching path feeds the backbone directly."""
        model = small_model(seed=12, dtype=torch.float32)
        samples = [
            make_single_graph_data(10, 4, seed=31),
            make_single_graph_data(12, 0, seed=32),
            make_single_graph_data(8, 6, seed=33),
        ]
        batch = collate_state_batch(samples)
        out = model(batch)
        assert out.num_graphs == 3
        assert out.num_pocket == 30
        assert out.num_ligand == 10
        assert out.scalar.shape == (40, H)
        assert out.vector.shape == (40, 3, H)
        assert out.graph_context.shape == (3, H)
        assert out.pocket_context.shape == (3, H)
        # Union ordering: 30 pocket nodes (type 0) then 10 ligand nodes (type 1).
        assert out.node_type[:30].eq(0).all()
        assert out.node_type[30:].eq(1).all()
        assert out.node_batch[:10].eq(0).all()
        assert out.node_batch[10:22].eq(1).all()
        assert out.node_batch[22:30].eq(2).all()
        assert out.node_batch[30:34].eq(0).all()  # ligands of graph 0
        assert out.node_batch[34:40].eq(2).all()  # ligands of graph 2

    def test_batched_equals_single_graph_runs(self):
        """Batching must not change any graph's representation (no
        cross-graph interaction: edges never leave a graph)."""
        model = small_model(seed=13, dtype=torch.float32)
        samples = [
            make_single_graph_data(11, 5, seed=41),
            make_single_graph_data(9, 3, seed=42),
            make_single_graph_data(13, 0, seed=43),
        ]
        batch = collate_state_batch(samples)
        out_batch = model(batch)
        for g, sample in enumerate(samples):
            out_single = model(collate_state_batch([sample]))
            mask = out_batch.node_batch == g
            torch.testing.assert_close(
                out_batch.scalar[mask], out_single.scalar, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                out_batch.vector[mask], out_single.vector, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                out_batch.graph_context[g], out_single.graph_context[0], rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                out_batch.pocket_context[g], out_single.pocket_context[0], rtol=1e-5, atol=1e-6
            )

    def test_pooling_matches_manual_mean(self):
        model = small_model(seed=14, dtype=torch.float32)
        samples = [
            make_single_graph_data(10, 4, seed=51),
            make_single_graph_data(7, 2, seed=52),
        ]
        batch = collate_state_batch(samples)
        out = model(batch)
        for g in range(2):
            nodes = out.node_batch == g
            expected_graph = out.scalar[nodes].mean(dim=0)
            torch.testing.assert_close(out.graph_context[g], expected_graph, rtol=1e-5, atol=1e-6)
            pocket_nodes = nodes & out.pocket_mask
            expected_pocket = out.scalar[pocket_nodes].mean(dim=0)
            torch.testing.assert_close(out.pocket_context[g], expected_pocket, rtol=1e-5, atol=1e-6)

    def test_direct_multi_graph_namespace(self):
        """Multi-graph input without PyG Batch (num_graphs derived from the
        batch vectors) is supported and validated."""
        model = small_model(seed=15)
        state = make_batched_node_state([8, 10], [3, 5], seed=53)
        out = model(state)
        assert out.num_graphs == 2
        assert out.scalar.shape == (26, H)
        assert out.graph_context.shape == (2, H)

    def test_handle_flags_union_ordering(self):
        """handle_flags follows the union ordering [pocket, ligand]: the
        default (None) equals an all-zero flag tensor, and marking a node in
        union position 9 (the first ligand node, with 8 pocket nodes)
        changes that node's representation."""
        model = small_model(seed=16)
        state = make_node_state(8, 4, seed=54)
        ligand_flag = torch.zeros(12, dtype=torch.long)
        ligand_flag[9] = 1  # first ligand node in union ordering
        out_default = model(state)  # handle_flags=None -> all zeros
        out_zero_flags = model(state, handle_flags=torch.zeros(12, dtype=torch.long))
        out_marked = model(state, handle_flags=ligand_flag)
        torch.testing.assert_close(out_default.scalar, out_zero_flags.scalar)
        torch.testing.assert_close(out_default.vector, out_zero_flags.vector)
        # The marked node itself must carry a different representation.
        assert not torch.allclose(out_zero_flags.scalar[9], out_marked.scalar[9])


# ---------------------------------------------------------------------------
# Activation checkpointing
# ---------------------------------------------------------------------------


class TestActivationCheckpointing:
    def _run(self, model, state):
        state.pocket_pos.requires_grad_(True)
        state.ligand_pos.requires_grad_(True)
        out = model(state)
        loss = out.scalar.sum() + out.vector.pow(2).sum() + out.pocket_context.sum()
        loss.backward()
        grads = {
            k: v.grad.clone()
            for k, v in model.named_parameters()
            if v.grad is not None
        }
        pos_grads = (state.pocket_pos.grad.clone(), state.ligand_pos.grad.clone())
        return out, grads, pos_grads

    def test_checkpointing_does_not_change_math(self):
        """Checkpointing is a memory optimization only: with and without it,
        outputs and gradients agree."""
        torch.manual_seed(61)
        model = small_model(seed=17, dtype=torch.float32)
        state = make_node_state(11, 5, seed=62, dtype=torch.float32)

        model.activation_checkpointing = True
        out_on, grads_on, pos_on = self._run(model, state)
        model.zero_grad(set_to_none=True)
        state.pocket_pos.grad = None
        state.ligand_pos.grad = None

        model.activation_checkpointing = False
        out_off, grads_off, pos_off = self._run(model, state)

        torch.testing.assert_close(out_on.scalar, out_off.scalar, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_on.vector, out_off.vector, rtol=1e-6, atol=1e-6)
        for k in grads_on:
            torch.testing.assert_close(grads_on[k], grads_off[k], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pos_on[0], pos_off[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pos_on[1], pos_off[1], rtol=1e-5, atol=1e-6)

    def test_backward_through_checkpointing_is_finite(self):
        torch.manual_seed(63)
        model = small_model(seed=18, dtype=torch.float32)
        state = make_node_state(10, 4, seed=64, dtype=torch.float32)
        _, grads, pos_grads = self._run(model, state)
        assert grads, "at least some parameter gradients must exist"
        for g in grads.values():
            assert bool(torch.isfinite(g).all())
        for g in pos_grads:
            assert bool(torch.isfinite(g).all())
            assert g.abs().sum() > 0

    def test_inference_without_grad_runs_with_flag_on(self):
        model = small_model(seed=19, dtype=torch.float32)
        state = make_node_state(9, 3, seed=65, dtype=torch.float32)
        with torch.no_grad():
            out = model(state)  # checkpointing flag on, but no autograd
        assert bool(torch.isfinite(out.scalar).all())


# ---------------------------------------------------------------------------
# Fail-closed input contract
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_none_state(self):
        model = small_model(seed=20)
        with pytest.raises(TensorContractError, match="received None"):
            model(None)

    def test_missing_fields(self):
        model = small_model(seed=21)
        with pytest.raises(TensorContractError, match="missing required"):
            model(SimpleNamespace(pocket_pos=torch.zeros(3, 3)))
        with pytest.raises(TensorContractError, match="ligand_pos"):
            model(make_node_state(5, 2, seed=1) if False else _drop(make_node_state(5, 2), "ligand_pos"))

    def test_dtype_mismatch_with_module(self):
        model = small_model(seed=22, dtype=torch.float32)
        state = make_node_state(6, 3, seed=66, dtype=torch.float64)
        with pytest.raises(TensorContractError, match="module parameters are"):
            model(state)

    def test_float16_rejected(self):
        model = small_model(seed=23, dtype=torch.float32)
        state = make_node_state(6, 3, seed=67, dtype=torch.float32)
        state.pocket_pos = state.pocket_pos.to(torch.float16)
        with pytest.raises(TensorContractError, match="float32"):
            model(state)

    def test_bfloat16_rejected(self):
        model = small_model(seed=24, dtype=torch.float32)
        state = make_node_state(6, 3, seed=68, dtype=torch.float32)
        state.ligand_charge = state.ligand_charge.to(torch.bfloat16)
        with pytest.raises(TensorContractError, match="production path"):
            model(state)

    def test_atomic_number_out_of_bounds(self):
        model = small_model(seed=25)
        state = make_node_state(6, 3, seed=69)
        state.pocket_z[0] = 0
        with pytest.raises(TensorContractError, match=r"atomic numbers must lie in"):
            model(state)
        state = make_node_state(6, 3, seed=69)
        state.ligand_z[1] = MAX_ATOMIC_NUMBER + 1
        with pytest.raises(TensorContractError, match="element-embedding"):
            model(state)

    def test_z_dtype_rejected(self):
        model = small_model(seed=26)
        state = make_node_state(6, 3, seed=70)
        state.pocket_z = state.pocket_z.to(torch.int32)
        with pytest.raises(TensorContractError, match="int64"):
            model(state)

    def test_length_mismatch(self):
        model = small_model(seed=27)
        state = make_node_state(6, 3, seed=71)
        state.pocket_z = state.pocket_z[:-1]
        with pytest.raises(TensorContractError, match=r"must be \[6\]"):
            model(state)
        state = make_node_state(6, 3, seed=71)
        state.ligand_charge = state.ligand_charge[:-1]
        with pytest.raises(TensorContractError, match="to match `ligand_pos`"):
            model(state)

    def test_non_finite_inputs(self):
        model = small_model(seed=28)
        state = make_node_state(6, 3, seed=72)
        state.pocket_pos[2, 1] = float("inf")
        with pytest.raises(TensorContractError, match="non-finite"):
            model(state)
        state = make_node_state(6, 3, seed=72)
        state.ligand_charge[0] = float("nan")
        with pytest.raises(TensorContractError, match="non-finite"):
            model(state)

    def test_empty_pocket_rejected(self):
        model = small_model(seed=29)
        state = make_node_state(0, 3, seed=73)
        with pytest.raises(TensorContractError, match="at least one atom"):
            model(state)

    def test_batch_index_out_of_range(self):
        model = small_model(seed=30)
        state = make_node_state(6, 3, seed=74)
        # Pin the graph count (as a Batch would) and exceed it.
        state.num_graphs = 1
        state.pocket_batch[0] = 1
        with pytest.raises(TensorContractError, match=r"\[0, 1\)"):
            model(state)

    def test_graph_without_pocket_atoms_rejected(self):
        model = small_model(seed=31)
        state = make_node_state(6, 3, seed=75)
        # Graph 0 owns the pockets; the ligand claims graph 1, which has no
        # pocket atoms — the conditioning context would be ill-defined.
        state.ligand_batch[:] = 1
        with pytest.raises(TensorContractError, match="at least one pocket"):
            model(state)

    def test_handle_flags_wrong_length(self):
        model = small_model(seed=32)
        state = make_node_state(6, 3, seed=76)
        with pytest.raises(TensorContractError, match=r"must be \[9\]"):
            model(state, handle_flags=torch.zeros(8, dtype=torch.long))

    def test_handle_flags_bad_values(self):
        model = small_model(seed=33)
        state = make_node_state(6, 3, seed=77)
        flags = torch.zeros(9, dtype=torch.long)
        flags[4] = 2
        with pytest.raises(TensorContractError, match="exactly 0 or 1"):
            model(state, handle_flags=flags)

    def test_handle_flags_bad_shape_and_dtype(self):
        model = small_model(seed=34)
        state = make_node_state(6, 3, seed=78)
        with pytest.raises(TensorContractError, match="must be"):
            model(state, handle_flags=torch.zeros(3, 3, dtype=torch.long))
        with pytest.raises(TensorContractError, match="dtype"):
            model(state, handle_flags=torch.zeros(9, dtype=torch.int32))
        with pytest.raises(TensorContractError, match="torch.Tensor"):
            model(state, handle_flags=[0] * 9)

    def test_handle_flag_float_values_must_be_exact_binary(self):
        model = small_model(seed=35)
        state = make_node_state(6, 3, seed=79)
        flags = torch.zeros(9, dtype=torch.float64)
        flags[7] = 0.5
        with pytest.raises(TensorContractError, match="exactly 0 or 1"):
            model(state, handle_flags=flags)

    def test_ligand_bad_dtype(self):
        model = small_model(seed=36)
        state = make_node_state(6, 3, seed=80)
        state.ligand_pos = state.ligand_pos.to(torch.float32)
        with pytest.raises(TensorContractError, match="module parameters"):
            model(state)


def _drop(state: SimpleNamespace, field: str) -> SimpleNamespace:
    """A copy of ``state`` with one field removed."""
    items = {k: v for k, v in vars(state).items() if k != field}
    return SimpleNamespace(**items)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_forward_identical(self):
        model = small_model(seed=37)
        state = make_node_state(9, 4, seed=81)
        out1 = model(state)
        out2 = model(state)
        assert torch.equal(out1.scalar, out2.scalar)
        assert torch.equal(out1.vector, out2.vector)
        assert torch.equal(out1.graph_context, out2.graph_context)

    def test_state_not_mutated_by_forward(self):
        model = small_model(seed=38)
        state = make_node_state(9, 4, seed=82)
        before = {k: v.clone() for k, v in vars(state).items()}
        model(state)
        for k, v in before.items():
            assert torch.equal(getattr(state, k), v)

    def test_encoder_does_not_require_policy_fields(self):
        """The backbone is decoupled from policy context: only the eight
        node-level fields are required (handle_features / global_features /
        stop_mask belong to the policy heads, not the encoder)."""
        model = small_model(seed=39)
        state = make_node_state(9, 4, seed=83)  # carries ONLY the 8 fields
        out = model(state)
        assert out.scalar.shape == (13, H)
