"""Unit tests for the pocket encoder, synthon head, and torsion head."""

from __future__ import annotations

import math

import pytest
import torch

from syntree.models.pocket_encoder import PocketEncoder
from syntree.models.synthon_head import SynthonHead, make_compatibility_mask
from syntree.models.torsion_head import ContinuousTorsionHead, log_bessel_i0


class TestPocketEncoder:
    @pytest.fixture()
    def encoder(self):
        torch.manual_seed(0)
        return PocketEncoder(hidden_dim=32, num_layers=2, num_radial=12,
                             cutoff=5.0)

    def test_single_graph_shapes(self, encoder):
        pos = torch.randn(20, 3)
        z = torch.randint(1, 9, (20,))
        s, v, pooled = encoder(pos, z)
        assert s.shape == (20, 32)
        assert v.shape == (20, 3, 32)
        assert pooled.shape == (1, 32)

    def test_batched_shapes(self, encoder):
        n1, n2 = 15, 25
        pos = torch.randn(n1 + n2, 3)
        z = torch.randint(1, 9, (n1 + n2,))
        batch = torch.tensor([0] * n1 + [1] * n2)
        s, v, pooled = encoder(pos, z, batch)
        assert s.shape == (n1 + n2, 32)
        assert pooled.shape == (2, 32)

    def test_pooling_is_mean(self, encoder):
        pos = torch.randn(10, 3)
        z = torch.randint(1, 9, (10,))
        encoder.eval()
        with torch.no_grad():
            s, _, pooled = encoder(pos, z)
        assert torch.allclose(pooled[0], s.mean(dim=0), atol=1e-5)

    def test_length_mismatch_raises(self, encoder):
        with pytest.raises(ValueError):
            encoder(torch.randn(10, 3), torch.randint(1, 9, (9,)))

    def test_bad_pos_shape_raises(self, encoder):
        with pytest.raises(ValueError):
            encoder(torch.randn(10, 2), torch.randint(1, 9, (10,)))

    def test_invalid_dims_raise(self):
        with pytest.raises(ValueError):
            PocketEncoder(hidden_dim=0)
        with pytest.raises(ValueError):
            PocketEncoder(num_layers=0)

    def test_gradients_flow(self, encoder):
        pos = torch.randn(12, 3)
        z = torch.randint(1, 9, (12,))
        s, v, pooled = encoder(pos, z)
        s.sum().backward()
        assert encoder.atomic_embedding.weight.grad is not None

    def test_num_parameters(self, encoder):
        assert encoder.num_parameters() > 0


class TestSynthonHead:
    @pytest.fixture()
    def head(self):
        torch.manual_seed(0)
        return SynthonHead(hidden_dim=32, num_heads=4)

    def test_output_shapes(self, head):
        q = torch.randn(3, 32)
        emb = torch.randn(50, 32)
        logits, log_probs = head(q, emb)
        # K synthons + 1 learned STOP action.
        assert logits.shape == (3, 51)
        assert log_probs.shape == (3, 51)

    def test_log_probs_normalized(self, head):
        q = torch.randn(2, 32)
        emb = torch.randn(30, 32)
        _, log_probs = head(q, emb)
        assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2), atol=1e-4)

    def test_mask_blocks_entries(self, head):
        head.eval()
        q = torch.randn(1, 32)
        emb = torch.randn(6, 32)
        mask = torch.zeros(1, 6)
        mask[0, 2:] = -1e9
        with torch.no_grad():
            logits, log_probs = head(q, emb, mask)
        assert (log_probs[0, 2:6] < -1e6).all()
        # Mass is shared between the legal synthons and the STOP action.
        assert float(log_probs[0].exp().sum()) == pytest.approx(1.0, abs=1e-4)
        assert float(log_probs[0, :2].exp().sum()) < 1.0

    def test_stop_mask_blocks_stop(self, head):
        """A negative stop_mask removes the STOP action from the support."""
        head.eval()
        q = torch.randn(2, 32)
        emb = torch.randn(6, 32)
        with torch.no_grad():
            _, log_probs = head(q, emb, stop_mask=torch.full((2,), -1e9))
        assert (log_probs[:, 6] < -1e6).all()
        assert torch.allclose(
            log_probs[:, :6].exp().sum(dim=-1), torch.ones(2), atol=1e-4
        )

    def test_fully_masked_is_finite(self, head):
        """Even a fully-masked row must not produce NaN (extreme -1e9)."""
        q = torch.randn(1, 32)
        emb = torch.randn(4, 32)
        mask = torch.full((1, 4), -1e9)
        logits, _ = head(q, emb, mask)
        assert torch.isfinite(logits).all()

    def test_bad_shapes_raise(self, head):
        with pytest.raises(ValueError):
            head(torch.randn(3), torch.randn(50, 32))
        with pytest.raises(ValueError):
            head(torch.randn(3, 32), torch.randn(50, 16))
        with pytest.raises(ValueError):
            head(torch.randn(3, 32), torch.randn(50, 32),
                 torch.zeros(3, 50, 1))

    def test_invalid_head_count_raises(self):
        with pytest.raises(ValueError):
            SynthonHead(hidden_dim=30, num_heads=4)

    def test_make_compatibility_mask(self):
        allowed = torch.tensor([[True, False, True], [False, False, True]])
        mask = make_compatibility_mask(allowed, num_synthons=3)
        assert mask[0, 0] == 0.0 and mask[0, 1] == -1e9
        assert mask[1, 2] == 0.0 and mask[1, 0] == -1e9

    def test_make_mask_wrong_cols_raise(self):
        with pytest.raises(ValueError):
            make_compatibility_mask(torch.ones(2, 5), num_synthons=4)


class TestLogBesselI0:
    def test_zero(self):
        assert log_bessel_i0(torch.tensor(0.0)).item() == pytest.approx(0.0)

    def test_small_matches_series(self):
        x = torch.tensor(0.5)
        expected = math.log(sum((x.item() ** 2 / 4) ** k / (math.factorial(k) ** 2)
                                for k in range(20)))
        assert log_bessel_i0(x).item() == pytest.approx(expected, rel=1e-4)

    def test_large_matches_asymptotic(self):
        x = torch.tensor(30.0)
        expected = x.item() - 0.5 * math.log(2 * math.pi * x.item())
        assert log_bessel_i0(x).item() == pytest.approx(expected, rel=1e-3)

    def test_monotone_increasing(self):
        x = torch.linspace(0.1, 50, 100)
        vals = log_bessel_i0(x)
        assert (vals[1:] > vals[:-1]).all()

    def test_nonnegative(self):
        x = torch.rand(100) * 20
        assert (log_bessel_i0(x) >= -1e-6).all()


class TestContinuousTorsionHead:
    @pytest.fixture()
    def head(self):
        torch.manual_seed(0)
        return ContinuousTorsionHead(hidden_dim=32)

    def test_output_shapes_and_ranges(self, head):
        ctx = torch.randn(4, 32)
        mu, kappa = head(ctx)
        assert mu.shape == (4,)
        assert kappa.shape == (4,)
        assert ((mu >= -math.pi) & (mu < math.pi)).all()
        assert (kappa >= 0.1).all()

    def test_vector_context_accepted(self, head):
        ctx = torch.randn(2, 32)
        vec = torch.randn(10, 3, 32)
        batch = torch.tensor([0] * 6 + [1] * 4)
        mu, kappa = head(ctx, vec, batch)
        assert mu.shape == (2,)

    def test_scalar_shape_validation(self, head):
        with pytest.raises(ValueError):
            head(torch.randn(4, 16))  # wrong dim
        with pytest.raises(ValueError):
            head(torch.randn(4, 32), torch.randn(10, 2, 32))

    def test_loss_zero_at_perfect_prediction(self):
        mu = torch.tensor([0.5])
        kappa = torch.tensor([20.0])
        target = torch.tensor([0.5])
        loss = ContinuousTorsionHead.loss_fn(mu, kappa, target)
        assert loss.item() < 0.05  # near the entropy floor, not zero

    def test_loss_high_when_far(self):
        mu = torch.tensor([0.0])
        kappa = torch.tensor([20.0])
        target = torch.tensor([math.pi])
        loss = ContinuousTorsionHead.loss_fn(mu, kappa, target)
        assert loss.item() > 30.0

    def test_loss_shape_validation(self):
        with pytest.raises(ValueError):
            ContinuousTorsionHead.loss_fn(
                torch.zeros(3), torch.ones(3), torch.zeros(4)
            )

    def test_log_prob_matches_manual(self):
        mu = torch.tensor(0.3)
        kappa = torch.tensor(2.0)
        phi = torch.tensor(1.1)
        lp = ContinuousTorsionHead.log_prob(mu, kappa, phi)
        # Manual von Mises log density with I0(2) = 2.2795853...
        manual = 2.0 * math.cos(1.1 - 0.3) - math.log(2 * math.pi * 2.2795853)
        assert lp.item() == pytest.approx(manual, rel=1e-3)

    def test_sampling_within_range(self):
        mu = torch.tensor([0.0, 1.0, -2.0, 3.0])
        kappa = torch.tensor([1.0, 5.0, 20.0, 0.5])
        samples = ContinuousTorsionHead.sample(mu, kappa)
        assert samples.shape == (4,)
        assert ((samples >= -math.pi) & (samples < math.pi)).all()

    def test_sampling_concentrates(self):
        """High kappa -> samples cluster near mu."""
        mu = torch.zeros(2000)
        kappa = torch.full((2000,), 30.0)
        samples = ContinuousTorsionHead.sample(mu, kappa)
        assert samples.abs().mean() < 0.3

    def test_training_reduces_loss(self, head):
        """A quick optimization sanity check on the head itself."""
        target = torch.tensor([1.0])
        opt = torch.optim.Adam(head.parameters(), lr=0.05)
        losses = []
        for _ in range(60):
            opt.zero_grad()
            mu, kappa = head(torch.zeros(1, 32))
            loss = ContinuousTorsionHead.loss_fn(mu, kappa, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]
