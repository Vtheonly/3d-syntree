"""Suite A / 5 — bit-exact determinism across PyTorch, NumPy, and Python
RNGs (issue #32 TEST-01; the canonical seeding module is
``syntree/utils/seeding.py``).

Regression battery for the V2 reproducibility contract: identical seeds
reproduce identical streams bit-for-bit in every framework, the three
framework streams are independently derived (not mutually correlated),
snapshot/restore reproduces exact continuation, and mis-seeding is
rejected fail-closed.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.utils.seeding import (  # noqa: E402
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


class TestBitExactDeterminism:
    def test_torch_stream_reproducible(self):
        def draw():
            seed_everything(1234)
            return torch.randn(64)

        assert torch.equal(draw(), draw())

    def test_numpy_stream_reproducible(self):
        def draw():
            seed_everything(1234)
            return np.random.default_rng(0).random(64), np.random.random(64)

        a1, b1 = draw()
        a2, b2 = draw()
        assert np.array_equal(a1, a2)
        assert np.array_equal(b1, b2)

    def test_python_stream_reproducible(self):
        def draw():
            seed_everything(1234)
            return [random.random() for _ in range(64)]

        assert draw() == draw()

    def test_cross_framework_joint_reproducibility(self):
        """A fixed work order draws identical values from all streams."""

        def work():
            seed_everything(2026)
            t = torch.randint(0, 1000, (8,))
            n = np.random.randint(0, 1000, size=8)
            p = [random.randint(0, 1000) for _ in range(8)]
            return t.tolist(), n.tolist(), p

        assert work() == work()

    def test_different_seeds_diverge(self):
        seed_everything(1)
        a = torch.randn(32)
        seed_everything(2)
        b = torch.randn(32)
        assert not torch.allclose(a, b)


class TestStreamIndependence:
    def test_framework_streams_are_derived_independently(self):
        """torch / numpy / python seeds are offsets apart, so framework
        streams never alias each other."""
        state = seed_everything(1_000_000)
        assert state.torch_seed == 1_000_001
        assert state.numpy_seed == 1_000_002
        assert state.python_seed == 1_000_003

    def test_generator_not_correlated_with_legacy_same_seed(self):
        """Seeding everything with s must not equal seeding each framework
        with s directly (correlated streams are a reproducibility hazard)."""
        seed_everything(555)
        via_module = torch.randn(8)
        torch.manual_seed(555)
        direct = torch.randn(8)
        assert not torch.allclose(via_module, direct)


class TestSnapshotRestore:
    def test_snapshot_restore_continues_exactly(self):
        seed_everything(99)
        _ = torch.randn(16)  # advance the streams
        _ = np.random.random(16)
        _ = [random.random() for _ in range(16)]
        snap = capture_rng_state()
        a = (torch.randn(16), np.random.random(16), random.random())
        restore_rng_state(snap)
        b = (torch.randn(16), np.random.random(16), random.random())
        assert torch.equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
        assert a[2] == b[2]

    def test_restore_rejects_foreign_objects(self):
        with pytest.raises(TypeError):
            restore_rng_state("not-a-snapshot")

    def test_checkpoint_style_resume(self):
        """The trainer pattern: seed, do work, snapshot, more work; later
        restore the snapshot and replay the same 'more work' exactly."""
        seed_everything(31337)
        torch.randn(64)
        snapshot = capture_rng_state()
        first = torch.randn(64)
        # ... session interruption ...
        seed_everything(0)  # streams wander in between
        torch.randn(64)
        restore_rng_state(snapshot)
        replay = torch.randn(64)
        assert torch.equal(first, replay)


class TestFailClosedSeeding:
    def test_non_integer_seed_rejected(self):
        with pytest.raises(TypeError):
            seed_everything("2026")

    def test_bool_seed_rejected(self):
        with pytest.raises(TypeError):
            seed_everything(True)

    def test_negative_seed_rejected(self):
        with pytest.raises(ValueError):
            seed_everything(-1)

    def test_out_of_range_seed_rejected(self):
        with pytest.raises(ValueError):
            seed_everything(2**63)


class TestSeededPipelines:
    def test_seeded_tensor_pipeline_end_to_end(self):
        """A representative deterministic pipeline: embedding init +
        dataloader-style shuffling reproduce bit-exactly."""

        def pipeline(seed):
            state = seed_everything(seed)
            weights = torch.randn(8, 4)
            order = np.random.permutation(16)
            dropout = (torch.rand(8, 4) > 0.5).float()
            return state, weights, order.tolist(), dropout

        s1, w1, o1, d1 = pipeline(42)
        s2, w2, o2, d2 = pipeline(42)
        assert s1 == s2
        assert torch.equal(w1, w2)
        assert o1 == o2
        assert torch.equal(d1, d2)
