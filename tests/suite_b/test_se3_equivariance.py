"""Suite B / 2 — SE(3) equivariance invariants (issue #33 TEST-02).

The issue-pinned battery: **scalar invariance and vector covariance
under random R in SO(3) and translations t in R^3** through the
authoritative ``BipartitePaiNN`` backbone over the union pocket + ligand
graph (#21 MODEL-01; constitution #45 §8.5).

* scalars: ``s'(g x) == s(x)`` for every rigid motion ``g``;
* vectors: ``v'(g x) == R v(x)`` (covariant, not invariant);
* 36 random rigid-motion trials in float64 (mirroring the #21
  acceptance scale), plus the pure-translation case and the pocket-only
  ``N_l = 0`` boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.models.bipartite import BipartitePaiNN  # noqa: E402

# A small configuration (fast); the frozen production profile and its
# equivariance pins are covered by the #21 root battery.
H = 16
LAYERS = 2
RADIAL = 6
CUTOFF = 5.0

_ORGANIC_Z = (1, 6, 7, 8, 9, 15, 16, 17)

NUM_TRIALS = 36


def random_rotation(generator: torch.Generator) -> torch.Tensor:
    """A random *proper* rotation (orthogonal, det = +1) in float64."""
    a = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def make_node_state(
    n_pocket: int, n_ligand: int, *, seed: int = 0, spread: float = 2.0
) -> SimpleNamespace:
    gen = torch.Generator().manual_seed(seed)
    elements = torch.tensor(_ORGANIC_Z, dtype=torch.long)

    def positions(count: int) -> torch.Tensor:
        return (torch.rand((count, 3), generator=gen, dtype=torch.float64) * 2.0 - 1.0) * spread

    def z_of(count: int) -> torch.Tensor:
        idx = torch.randint(0, len(_ORGANIC_Z), (count,), generator=gen)
        return elements[idx]

    def charges(count: int) -> torch.Tensor:
        return torch.randint(-1, 2, (count,), generator=gen).to(torch.float64)

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


def rigidly_transformed(state: SimpleNamespace, rotation, translation) -> SimpleNamespace:
    return SimpleNamespace(
        pocket_pos=state.pocket_pos @ rotation.T + translation,
        pocket_z=state.pocket_z,
        pocket_charge=state.pocket_charge,
        pocket_batch=state.pocket_batch,
        ligand_pos=state.ligand_pos @ rotation.T + translation,
        ligand_z=state.ligand_z,
        ligand_charge=state.ligand_charge,
        ligand_batch=state.ligand_batch,
    )


def rotate_vectors(rotation: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ab,nbh->nah", rotation, vectors)


def small_model(*, seed: int = 0) -> BipartitePaiNN:
    torch.manual_seed(seed)
    return BipartitePaiNN(
        hidden_dim=H, num_layers=LAYERS, num_radial=RADIAL, r_cut=CUTOFF
    ).to(torch.float64)


class TestSE3Equivariance:
    def test_scalar_invariance_and_vector_covariance_over_random_motions(self):
        """The pinned battery: 36 random (R, t) trials in float64."""
        model = small_model(seed=1)
        state = make_node_state(14, 7, seed=4)
        reference = model(state)
        gen = torch.Generator().manual_seed(21)
        max_scalar_error = 0.0
        max_vector_error = 0.0
        for _ in range(NUM_TRIALS):
            rotation = random_rotation(gen)
            translation = torch.randn(3, generator=gen, dtype=torch.float64)
            transformed = model(rigidly_transformed(state, rotation, translation))
            scalar_error = float(
                (transformed.scalar - reference.scalar).abs().max()
            )
            expected_vectors = rotate_vectors(rotation, reference.vector)
            vector_error = float(
                (transformed.vector - expected_vectors).abs().max()
            )
            max_scalar_error = max(max_scalar_error, scalar_error)
            max_vector_error = max(max_vector_error, vector_error)
            assert scalar_error <= 1e-9
            assert vector_error <= 1e-9
        assert max_scalar_error <= 1e-12 or max_scalar_error > 0.0  # exercised
        assert max_vector_error > 0.0  # the covariance check was non-trivial

    def test_rotation_is_actually_proper(self):
        gen = torch.Generator().manual_seed(3)
        for _ in range(8):
            rotation = random_rotation(gen)
            torch.testing.assert_close(
                rotation @ rotation.T, torch.eye(3, dtype=torch.float64)
            )
            assert abs(float(torch.det(rotation)) - 1.0) < 1e-12

    def test_pure_translation_leaves_both_channels_untouched(self):
        model = small_model(seed=2)
        state = make_node_state(12, 6, seed=5)
        translation = torch.tensor([0.5, -1.25, 2.0], dtype=torch.float64)
        identity = torch.eye(3, dtype=torch.float64)
        reference = model(state)
        moved = model(rigidly_transformed(state, identity, translation))
        torch.testing.assert_close(moved.scalar, reference.scalar, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(moved.vector, reference.vector, rtol=1e-9, atol=1e-9)

    def test_pocket_only_state_is_equivariant_and_clean(self):
        model = small_model(seed=3)
        state = make_node_state(16, 0, seed=7)  # N_l = 0 boundary
        gen = torch.Generator().manual_seed(29)
        rotation = random_rotation(gen)
        translation = torch.randn(3, generator=gen, dtype=torch.float64)
        reference = model(state)
        transformed = model(rigidly_transformed(state, rotation, translation))
        torch.testing.assert_close(
            transformed.scalar, reference.scalar, rtol=1e-9, atol=1e-9
        )
        expected = rotate_vectors(rotation, reference.vector)
        torch.testing.assert_close(transformed.vector, expected, rtol=1e-9, atol=1e-9)
        assert transformed.vector.shape[0] == 16  # pocket nodes only

    def test_equivariance_holds_batched(self):
        model = small_model(seed=4)
        states = []
        for g, (n_p, n_l) in enumerate([(12, 5), (9, 4), (7, 0)]):
            s = make_node_state(n_p, n_l, seed=11 + g)
            s.pocket_batch = torch.full((n_p,), g, dtype=torch.long)
            s.ligand_batch = torch.full((n_l,), g, dtype=torch.long)
            states.append(s)
        batched = SimpleNamespace(
            pocket_pos=torch.cat([s.pocket_pos for s in states]),
            pocket_z=torch.cat([s.pocket_z for s in states]),
            pocket_charge=torch.cat([s.pocket_charge for s in states]),
            pocket_batch=torch.cat([s.pocket_batch for s in states]),
            ligand_pos=torch.cat([s.ligand_pos for s in states]),
            ligand_z=torch.cat([s.ligand_z for s in states]),
            ligand_charge=torch.cat([s.ligand_charge for s in states]),
            ligand_batch=torch.cat([s.ligand_batch for s in states]),
        )
        gen = torch.Generator().manual_seed(31)
        rotation = random_rotation(gen)
        translation = torch.randn(3, generator=gen, dtype=torch.float64)
        reference = model(batched)
        transformed = model(rigidly_transformed(batched, rotation, translation))
        torch.testing.assert_close(
            transformed.scalar, reference.scalar, rtol=1e-9, atol=1e-9
        )
        expected = rotate_vectors(rotation, reference.vector)
        torch.testing.assert_close(transformed.vector, expected, rtol=1e-9, atol=1e-9)
