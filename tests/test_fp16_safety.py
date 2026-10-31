"""Regression tests for AMP/fp16 numerical safety.

Background (Colab T4 incident, fp16 autocast): ``torch.norm`` squares its
inputs and fp16 overflows above 65,504, so vector features with per-component
magnitude ~150 turned the torsion head's pooled statistics into inf/NaN at
epoch 4 (``torsion_nll: nan`` while ``synthon_ce`` stayed finite, because the
scalar path never squares anything). These tests pin the hardening: fp32
statistics with saturation caps in the torsion head and PaiNN mixing, fp32
loss/Bessel math, and learnable (not pure-noise) synthetic supervision.
"""

from __future__ import annotations

import math

import pytest
import torch

from syntree.models.equivariant import PaiNNMixing
from syntree.models.torsion_head import ContinuousTorsionHead, log_bessel_i0


class TestTorsionHeadFp16Safety:
    @pytest.fixture()
    def head(self):
        torch.manual_seed(0)
        return ContinuousTorsionHead(hidden_dim=32)

    def test_large_fp16_vectors_finite(self, head):
        """Per-component magnitude 300: naive fp16 norm squares to 270k > 65504.

        The old implementation overflowed to inf, then std() produced NaN and
        poisoned the loss. The hardened path must return finite mu/kappa.
        """
        ctx = torch.randn(4, 32)
        vec = (torch.randn(15, 3, 32) * 300.0).half()
        batch = torch.tensor([0] * 6 + [1] * 5 + [2] * 4)
        mu, kappa = head(ctx, vec, batch)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(kappa).all()
        assert ((mu >= -math.pi) & (mu < math.pi)).all()
        assert ((kappa >= 0.1) & (kappa <= 50.0)).all()

    def test_inf_nan_inputs_do_not_propagate(self, head):
        """Even a corrupted vector channel (inf/NaN entries) stays contained."""
        ctx = torch.randn(3, 32)
        vec = torch.randn(9, 3, 32)
        vec[0, 0, :] = float("inf")
        vec[4, 1, :] = float("nan")
        batch = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
        mu, kappa = head(ctx, vec, batch)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(kappa).all()

    def test_head_under_cpu_bf16_autocast(self, head):
        """Inside an autocast region the head still executes its fp32 path."""
        ctx = torch.randn(4, 32)
        vec = torch.randn(12, 3, 32) * 200.0
        batch = torch.tensor([0] * 6 + [1] * 4 + [2] * 2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mu, kappa = head(ctx, vec, batch)
            loss = ContinuousTorsionHead.loss_fn(mu, kappa, torch.randn(4))
        assert torch.isfinite(loss)
        assert torch.isfinite(mu).all() and torch.isfinite(kappa).all()

    def test_pool_stats_match_reference_loop(self):
        """Vectorized pooling reproduces the original per-graph loop exactly:
        mean/max/unbiased-std over all n_g * d norm entries, zeros for empty
        graphs, zero std for singleton groups."""
        torch.manual_seed(7)
        n, d, b = 37, 8, 5
        vec = torch.randn(n, 3, d)
        # Ragged assignment: graph 3 empty, graph 4 single node (37 = 12+9+15+1).
        batch = torch.tensor(
            [0] * 12 + [1] * 9 + [2] * 15 + [4]
        )
        stats = ContinuousTorsionHead._pool_vector_stats(vec, batch, b)

        norms = torch.norm(vec, dim=1)
        for g in range(b):
            sel = norms[batch == g]
            if sel.numel() == 0:
                expected = torch.zeros(3)
            else:
                flat = sel.reshape(-1)
                expected = torch.stack(
                    [
                        flat.mean(),
                        flat.max(),
                        flat.std() if flat.numel() > 1 else torch.zeros(()),
                    ]
                )
            assert torch.allclose(stats[g], expected, atol=1e-5), f"graph {g}"

    def test_loss_fn_fp16_inputs_finite_and_accurate(self):
        mu = (torch.randn(16) * 2).half()
        kappa = (torch.rand(16) * 49 + 0.5).half()
        target = (torch.rand(16) * 2 * math.pi - math.pi).half()
        loss = ContinuousTorsionHead.loss_fn(mu, kappa, target)
        assert torch.isfinite(loss)
        ref = ContinuousTorsionHead.loss_fn(mu.float(), kappa.float(), target.float())
        assert loss.item() == pytest.approx(ref.item(), rel=1e-3)

    def test_log_bessel_i0_huge_arguments(self):
        x = torch.tensor([0.0, 1.0, 5.0, 50.0, 1e4, 1e6])
        out = log_bessel_i0(x)
        assert torch.isfinite(out).all()
        assert (out >= -1e-6).all()
        # Asymptotics: log I0(x) ~ x - 0.5*log(2*pi*x) for large x.
        for xv, ov in zip([50.0, 1e4, 1e6], out[-3:].tolist()):
            assert ov == pytest.approx(xv - 0.5 * math.log(2 * math.pi * xv), rel=1e-4)

    def test_backward_through_fp16_statistics(self, head):
        """Gradients flow through the fp32 statistics path without NaN."""
        ctx = torch.randn(4, 32, requires_grad=True)
        vec = (torch.randn(10, 3, 32) * 250.0).half().float().requires_grad_(True)
        batch = torch.tensor([0] * 4 + [1] * 3 + [2] * 3)
        mu, kappa = head(ctx, vec, batch)
        loss = ContinuousTorsionHead.loss_fn(mu, kappa, torch.randn(4))
        loss.backward()
        assert torch.isfinite(ctx.grad).all()
        assert torch.isfinite(vec.grad).all()


class TestPaiNNMixingFp16Safety:
    def test_large_fp16_vectors_finite(self):
        torch.manual_seed(1)
        mixing = PaiNNMixing(hidden_dim=16)
        scalar = torch.randn(20, 16)
        vector = (torch.randn(20, 3, 16) * 300.0).half().float()
        out_s, out_v = mixing(scalar, vector)
        assert torch.isfinite(out_s).all()
        assert torch.isfinite(out_v).all()

    def test_small_values_unchanged(self):
        """The non-singular norm must not alter ordinary fp32 results.

        Reference uses the #7 MATH-01 primitive (safe_norm) that the mixing
        now routes through; the legacy reference ``torch.norm(v + eps)``
        was retired with the singularity fix.
        """
        torch.manual_seed(2)
        mixing = PaiNNMixing(hidden_dim=16)
        scalar = torch.randn(10, 16)
        vector = torch.randn(10, 3, 16)
        out_s, out_v = mixing(scalar, vector)

        from syntree.models.equivariant import safe_norm

        vec_proj = mixing.vec_proj(vector)
        v_norm = safe_norm(vec_proj, dim=1, eps=mixing.epsilon)
        h = torch.cat([scalar, v_norm], dim=-1)
        u = mixing.update_mlp(h)
        a_ss, a_sv, a_vv = torch.split(u, mixing.hidden_dim, dim=-1)
        ref_s = scalar + a_sv * v_norm + a_ss
        ref_v = vector + a_vv.unsqueeze(1) * vector
        assert torch.allclose(out_s, ref_s, atol=1e-6)
        assert torch.allclose(out_v, ref_v, atol=1e-6)
