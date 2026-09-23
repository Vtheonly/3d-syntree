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

from syntree.data.crossdocked import CrossDockedDataset
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
        """The fp32-norm shortcut must not alter ordinary fp32 results."""
        torch.manual_seed(2)
        mixing = PaiNNMixing(hidden_dim=16)
        scalar = torch.randn(10, 16)
        vector = torch.randn(10, 3, 16)
        out_s, out_v = mixing(scalar, vector)

        vec_proj = mixing.vec_proj(vector)
        v_norm = torch.norm(vec_proj + mixing.epsilon, dim=1)
        h = torch.cat([scalar, v_norm], dim=-1)
        u = mixing.update_mlp(h)
        a_ss, a_sv, a_vv = torch.split(u, mixing.hidden_dim, dim=-1)
        ref_s = scalar + a_sv * v_norm + a_ss
        ref_v = vector + a_vv.unsqueeze(1) * vector
        assert torch.allclose(out_s, ref_s, atol=1e-6)
        assert torch.allclose(out_v, ref_v, atol=1e-6)


class TestLearnableSyntheticSupervision:
    """The synthetic targets must carry signal (not pure noise), so a Colab
    smoke run actually verifies that the policy learns; and they must remain
    deterministic across processes (crc32 split salt, seeded projections)."""

    @pytest.fixture()
    def dataset(self, tmp_path, assets_dir):
        from syntree.chemistry.catalog import SynthonCatalog

        catalog = SynthonCatalog(
            assets_dir["catalog_path"], embedding_dim=32, validate_handles=False
        )
        return CrossDockedDataset(
            str(tmp_path / "ds"), catalog=catalog, num_synthetic=256, seed=3
        )

    def test_targets_in_range(self, dataset):
        n_catalog = len(dataset.catalog)
        for i in range(len(dataset)):
            t = dataset[i].target_synthon.item()
            assert 0 <= t < n_catalog
            d = dataset[i].target_dihedral.item()
            assert -math.pi <= d < math.pi

    def test_targets_are_deterministic_functions_of_features(self, dataset):
        """Identical handle features + pocket stats must map to the identical
        target (the mapping is fixed, not resampled per access)."""
        a, b = dataset[0], dataset[31]
        # Rebuild the mapping by hand for sample a.
        feats = torch.cat(
            [
                a.handle_features,
                torch.stack(
                    [
                        a.pocket_pos[:, 0].mean(),
                        a.pocket_pos[:, 1].mean(),
                        a.pocket_pos[:, 2].mean(),
                        a.pocket_pos.abs().mean(),
                        a.pocket_z.float().mean(),
                    ]
                ),
            ]
        )
        g = torch.Generator().manual_seed(3 + 7919)
        w_syn = torch.randn(69, generator=g)
        w_dih = torch.randn(69, generator=g)
        scale = math.sqrt(69)
        n_catalog = len(dataset.catalog)
        expect_syn = int(
            torch.floor(torch.sigmoid(3.0 * torch.dot(feats, w_syn) / scale) * n_catalog)
            .clamp(0, n_catalog - 1)
        )
        expect_dih = math.pi * math.tanh(2.0 * torch.dot(feats, w_dih) / scale)
        assert a.target_synthon.item() == expect_syn
        assert a.target_dihedral.item() == pytest.approx(expect_dih, abs=1e-5)
        # Sanity: different samples usually differ.
        assert a.target_synthon.item() != b.target_synthon.item() or (
            a.target_dihedral.item() != b.target_dihedral.item()
        )

    def test_targets_carry_signal(self, dataset):
        """Correlated inputs -> correlated targets: a leave-one-out 1-NN
        predictor on raw features must clearly beat the uniform-random floor
        on the continuous dihedral target (with the old pure-noise targets the
        expected MAE is pi/2 ~ 1.571). The synthon head's learnability is
        already pinned exactly by the deterministic-mapping test above."""
        import numpy as np

        n = len(dataset)
        feats, dih = [], []
        for i in range(n):
            s = dataset[i]
            feats.append(
                torch.cat(
                    [s.handle_features, s.pocket_pos.mean(0), s.pocket_z.float().mean().unsqueeze(0)]
                )
            )
            dih.append(s.target_dihedral.item())
        feats = torch.tensor(np.asarray([f.numpy() for f in feats]))
        dih = torch.tensor(dih)

        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
        dist = torch.cdist(feats, feats)
        dist.fill_diagonal_(float("inf"))
        nn_idx = dist.argmin(dim=1)
        diff = (dih[nn_idx] - dih + math.pi) % (2 * math.pi) - math.pi
        mae = diff.abs().mean().item()
        assert mae < 1.25, f"1-NN dihedral MAE {mae:.3f} shows no learnable signal"

    def test_train_val_splits_differ_but_reproduce(self, tmp_path, assets_dir):
        from syntree.chemistry.catalog import SynthonCatalog

        catalog = SynthonCatalog(
            assets_dir["catalog_path"], embedding_dim=32, validate_handles=False
        )
        d1 = CrossDockedDataset(str(tmp_path / "a"), catalog=catalog, num_synthetic=16, seed=5)
        d2 = CrossDockedDataset(str(tmp_path / "b"), catalog=catalog, num_synthetic=16, seed=5)
        for i in range(16):
            assert d1[i].target_synthon == d2[i].target_synthon
            assert d1[i].target_dihedral == d2[i].target_dihedral
        d3 = CrossDockedDataset(
            str(tmp_path / "c"), catalog=catalog, num_synthetic=16, seed=5, split="val"
        )
        # Different split salt -> different sampling stream.
        assert not torch.equal(d1[0].handle_features, d3[0].handle_features)
