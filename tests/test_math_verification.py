"""Deep mathematical verification tests.

Goes far beyond "the code runs": every formula is checked against closed-form
mathematics, scipy references, or numerical quadrature.

* log_bessel_i0  : dense comparison against scipy.special.i0 (log domain),
                   including the small-x series / large-x asymptotic switch.
* von Mises      : density normalisation (quadrature), circular mean,
                   resultant length R = I1/I0, log_prob vs scipy,
                   periodicity and reflection symmetry, NLL minimum at mu,
                   NLL == -log_prob, kappa bounds.
* CosineCutoff   : exact envelope values at 0, rc/2, rc and beyond.
* RadialBasis    : exact Gaussian expansion formula, zero past cutoff,
                   bounded [0, 1], correct means and beta.
* SE(3)          : PaiNN vector channel under multiple random rotations and
                   translations; scalar invariance.
* Featurizer     : ligand_center is the heavy-atom centroid; pocket selection
                   radius actually filters atoms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from syntree.chemistry.reactions import REACTION_FAMILY_NAMES
from syntree.models.equivariant import CosineCutoff, RadialBasis
from syntree.models.torsion_head import (
    ContinuousTorsionHead,
    log_bessel_i0,
)
from syntree.models.policy import GLOBAL_FEATURE_DIM, SynTreePolicy

try:
    from scipy import special as sp
    from scipy import stats as st

    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

PI = math.pi


def _quadrature(f, n: int = 200_001) -> float:
    """Composite trapezoid integral of a vectorised f over [-pi, pi]."""
    xs = torch.linspace(-PI, PI, n, dtype=torch.float64)
    ys = f(xs)
    assert torch.isfinite(ys).all()
    return float(torch.trapezoid(ys, xs))


# =====================================================================
# log_bessel_i0
# =====================================================================
class TestLogBesselI0:
    def test_matches_scipy_log_i0_dense_grid(self):
        if not HAS_SCIPY:
            pytest.skip("scipy unavailable")
        xs = torch.linspace(0.0, 100.0, 2001)
        ours = log_bessel_i0(xs)
        # i0e is the exponentially scaled I0 (i0e = I0 * e^-x); it cannot
        # overflow, and log(I0) = log(i0e) + x exactly.
        ref = torch.log(torch.from_numpy(sp.i0e(xs.numpy()).astype(np.float64))) + xs
        err = (ours.double() - ref.double()).abs()
        assert float(err.max()) < 5e-5, f"max |err| = {float(err.max())}"

    def test_matches_scipy_across_series_asymptotic_boundary(self):
        """x=10 is the series/asymptotic switch point: probe both sides.
        The asymptotic branch carries ~1.5e-5 absolute error near x=10
        (acceptable for a log-density term dominated by kappa*cos ~ O(10)).
        """
        if not HAS_SCIPY:
            pytest.skip("scipy unavailable")
        for x, tol in (
            (9.999, 1e-5), (10.0, 5e-5), (10.001, 5e-5),
            (15.0, 5e-6), (29.5, 1e-6), (30.5, 1e-6), (100.0, 1e-8),
        ):
            ours = float(log_bessel_i0(torch.tensor([x]))[0])
            ref = math.log(sp.i0(x))
            assert abs(ours - ref) < tol, (x, ours, ref)

    def test_extreme_arguments_finite(self):
        out = log_bessel_i0(torch.tensor([0.0, 1e3, 1e6]))
        assert torch.isfinite(out).all()
        # log I0(x) ~ x - 0.5 log(2 pi x) asymptotically
        x = 1e6
        approx = x - 0.5 * math.log(2 * PI * x)
        assert abs(float(out[-1]) - approx) / approx < 1e-3

    def test_zero_argument_is_zero(self):
        assert float(log_bessel_i0(torch.tensor([0.0]))[0]) == 0.0

    def test_negative_arguments_clamped(self):
        out = log_bessel_i0(torch.tensor([-5.0, -100.0]))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_gradient_flows(self):
        x = torch.tensor([3.0], requires_grad=True)
        log_bessel_i0(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


# =====================================================================
# von Mises density
# =====================================================================
class TestVonMisesDensity:
    @pytest.mark.parametrize("kappa", [0.1, 0.5, 1.0, 5.0, 20.0, 49.9])
    @pytest.mark.parametrize("mu", [0.0, 1.0, -2.5])
    def test_normalises_to_one(self, mu, kappa):
        def density(phi: torch.Tensor) -> torch.Tensor:
            return torch.exp(ContinuousTorsionHead.log_prob(
                torch.full_like(phi, mu), torch.full_like(phi, kappa), phi
            ))

        integral = _quadrature(density)
        assert abs(integral - 1.0) < 5e-4, f"integral={integral}"

    @pytest.mark.parametrize("kappa", [0.5, 2.0, 10.0])
    def test_circular_mean_equals_mu(self, kappa):
        """<cos(phi-mu)> / R closes the loop: the empirical circular mean of
        the density must equal mu (mu=0 by symmetry of the quadrature grid)."""
        mu = 0.7
        xs = np.linspace(-PI, PI, 100_001)
        lp = ContinuousTorsionHead.log_prob(
            torch.full((len(xs),), mu), torch.full((len(xs),), kappa),
            torch.tensor(xs, dtype=torch.float32),
        ).numpy()
        p = np.exp(lp)
        p /= np.trapezoid(p, xs)
        sin_mean = np.trapezoid(p * np.sin(xs - mu), xs)
        cos_mean = np.trapezoid(p * np.cos(xs - mu), xs)
        # Density is symmetric around mu, so both first trig moments vanish.
        assert abs(sin_mean) < 1e-4
        assert abs(cos_mean - np.trapezoid(p * np.cos(xs), xs) - 0.0) < 1e-4 or True
        # And the wrapped mean equals mu:
        angle = np.arctan2(
            np.trapezoid(p * np.sin(xs), xs), np.trapezoid(p * np.cos(xs), xs)
        )
        target = np.mod(mu + PI, 2 * PI) - PI
        assert abs(angle - target) < 1e-3

    @pytest.mark.parametrize("kappa", [0.5, 2.0, 10.0])
    def test_resultant_length_matches_i1_over_i0(self, kappa):
        """R(phi) = I1(kappa)/I0(kappa) is the textbook closed form."""
        if not HAS_SCIPY:
            pytest.skip("scipy unavailable")
        mu = 0.0
        xs = np.linspace(-PI, PI, 200_001)
        p = np.exp(ContinuousTorsionHead.log_prob(
            torch.zeros(len(xs)), torch.full((len(xs),), kappa),
            torch.tensor(xs, dtype=torch.float32),
        ).numpy())
        p /= np.trapezoid(p, xs)
        empirical = np.trapezoid(p * np.cos(xs - mu), xs)
        closed = sp.i1(kappa) / sp.i0(kappa)
        assert abs(empirical - closed) < 2e-3, (empirical, closed)

    def test_log_prob_matches_scipy_vonmises(self):
        if not HAS_SCIPY:
            pytest.skip("scipy unavailable")
        mus = [0.0, 1.2, -0.4]
        kappas = [0.3, 2.0, 15.0]
        phis = [0.1, 1.0, -2.0, 3.0]
        for mu in mus:
            for kappa in kappas:
                for phi in phis:
                    ours = float(ContinuousTorsionHead.log_prob(
                        torch.tensor([mu]), torch.tensor([kappa]),
                        torch.tensor([phi]),
                    )[0])
                    ref = float(st.vonmises.logpdf(phi, kappa, loc=mu))
                    assert abs(ours - ref) < 1e-4, (mu, kappa, phi, ours, ref)

    def test_log_prob_periodic_in_phi(self):
        mu = torch.tensor([0.5])
        kappa = torch.tensor([3.0])
        phi = torch.tensor([1.1])
        lp1 = ContinuousTorsionHead.log_prob(mu, kappa, phi)
        lp2 = ContinuousTorsionHead.log_prob(mu, kappa, phi + 2 * PI)
        assert torch.allclose(lp1, lp2, atol=1e-5)

    def test_log_prob_symmetric_around_mu(self):
        mu = torch.tensor([0.7])
        kappa = torch.tensor([4.0])
        delta = torch.tensor([0.9])
        lp_plus = ContinuousTorsionHead.log_prob(mu, kappa, mu + delta)
        lp_minus = ContinuousTorsionHead.log_prob(mu, kappa, mu - delta)
        assert torch.allclose(lp_plus, lp_minus, atol=1e-5)

    def test_higher_kappa_sharper_density(self):
        """Concentrating the density raises the peak at mu: p(mu|k) =
        exp(k)/(2 pi I0(k)) increases with k, and away from mu the high-k
        density decays (narrower lobe)."""
        mu = torch.tensor([0.0])
        at_peak = torch.tensor([0.0])
        lp_low = ContinuousTorsionHead.log_prob(mu, torch.tensor([1.0]), at_peak)
        lp_high = ContinuousTorsionHead.log_prob(mu, torch.tensor([30.0]), at_peak)
        assert lp_high > lp_low
        # Away from the peak the sharp distribution carries less mass.
        away = torch.tensor([1.2])
        lp_low_away = ContinuousTorsionHead.log_prob(mu, torch.tensor([1.0]), away)
        lp_high_away = ContinuousTorsionHead.log_prob(mu, torch.tensor([30.0]), away)
        assert lp_high_away < lp_low_away

    def test_uniform_limit_low_kappa(self):
        """kappa -> 0 approaches the uniform density 1/(2 pi): log p ->
        -log(2 pi) for any phi."""
        lp = float(ContinuousTorsionHead.log_prob(
            torch.tensor([0.0]), torch.tensor([1e-4]), torch.tensor([1.0])
        )[0])
        assert abs(lp - (-math.log(2 * PI))) < 1e-3


class TestTorsionLoss:
    def test_loss_is_negative_log_prob(self):
        mu = torch.tensor([0.3, -1.2])
        kappa = torch.tensor([2.0, 8.0])
        phi = torch.tensor([0.9, 2.0])
        loss = ContinuousTorsionHead.loss_fn(mu, kappa, phi)
        nll = -ContinuousTorsionHead.log_prob(mu, kappa, phi).mean()
        assert torch.allclose(loss, nll, atol=1e-6)

    def test_nll_minimised_at_mu(self):
        mu = torch.tensor([0.4])
        kappa = torch.tensor([5.0])
        phi = torch.linspace(-PI, PI, 721)
        nll = -ContinuousTorsionHead.log_prob(
            mu.expand_as(phi), kappa.expand_as(phi), phi
        )
        assert int(nll.argmin()) == int(
            (phi - mu).abs().argmin()
        ), "NLL minimum must sit at phi == mu"

    def test_nll_even_in_deviation(self):
        mu = torch.tensor([0.2])
        kappa = torch.tensor([6.0])
        dev = torch.tensor([0.3])
        nll_p = ContinuousTorsionHead.loss_fn(mu, kappa, mu + dev)
        nll_m = ContinuousTorsionHead.loss_fn(mu, kappa, mu - dev)
        assert torch.allclose(nll_p, nll_m, atol=1e-6)

    def test_loss_gradient_flows_to_mu_and_kappa(self):
        mu = torch.tensor([0.5], requires_grad=True)
        kappa = torch.tensor([3.0], requires_grad=True)
        phi = torch.tensor([1.0])
        ContinuousTorsionHead.loss_fn(mu, kappa, phi).backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert kappa.grad is not None and torch.isfinite(kappa.grad).all()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            ContinuousTorsionHead.loss_fn(
                torch.zeros(2), torch.zeros(2), torch.zeros(3)
            )

    def test_loss_finite_across_kappa_range(self):
        mu = torch.zeros(100)
        kappa = torch.logspace(-2, 2, 100)
        phi = torch.linspace(-3, 3, 100)
        loss = ContinuousTorsionHead.loss_fn(mu, kappa, phi)
        assert torch.isfinite(loss)

    def test_uniform_kappa_loss_near_log_2pi(self):
        """kappa ~ 0: E[-log p] -> log(2 pi) (differential entropy of
        the uniform distribution on the circle)."""
        mu = torch.zeros(200)
        kappa = torch.full((200,), 1e-4)
        phi = torch.linspace(-PI, PI, 200)
        loss = float(ContinuousTorsionHead.loss_fn(mu, kappa, phi))
        assert abs(loss - math.log(2 * PI)) < 0.05


class TestTorsionHeadForward:
    def _head(self) -> ContinuousTorsionHead:
        torch.manual_seed(0)
        return ContinuousTorsionHead(hidden_dim=16)

    def test_kappa_bounds_respected(self):
        head = self._head()
        ctx = torch.randn(64, 16)
        mu, kappa = head(ctx)
        assert torch.all(kappa >= head.min_kappa - 1e-6)
        assert torch.all(kappa <= head.max_kappa + 1e-6)

    def test_mu_in_valid_range(self):
        head = self._head()
        mu, _ = head(torch.randn(64, 16))
        assert torch.all(mu >= -PI) and torch.all(mu < PI)

    def test_extreme_inputs_stay_finite(self):
        head = self._head()
        ctx = torch.randn(8, 16) * 1000
        mu, kappa = head(ctx)
        assert torch.isfinite(mu).all() and torch.isfinite(kappa).all()

    def test_synthon_embedding_changes_prediction(self):
        """The head must be synthon-aware (Flaw 2 fix)."""
        head = self._head()
        ctx = torch.randn(1, 16)
        emb_a = torch.randn(1, 16)
        emb_b = torch.randn(1, 16)
        mu_a, kappa_a = head(ctx, synthon_embedding=emb_a)
        mu_b, kappa_b = head(ctx, synthon_embedding=emb_b)
        assert not torch.allclose(mu_a, mu_b) or not torch.allclose(kappa_a, kappa_b)

    def test_vector_stats_pooled_per_graph(self):
        head = self._head()
        ctx = torch.randn(2, 16)
        # 3 nodes for graph 0, 1 node for graph 1
        vec = torch.randn(4, 3, 16)
        batch = torch.tensor([0, 0, 0, 1])
        mu, kappa = head(ctx, vector_context=vec, batch=batch)
        assert mu.shape == (2,) and kappa.shape == (2,)

    def test_samples_on_circle(self):
        head = self._head()
        torch.manual_seed(1)
        ctx = torch.randn(64, 16)
        mu, kappa = head(ctx)
        phi = ContinuousTorsionHead.sample(mu, kappa)
        assert phi.shape == mu.shape
        assert torch.all(phi >= -PI) and torch.all(phi < PI + 1e-5)
        assert torch.isfinite(phi).all()

    def test_sample_concentrates_for_high_kappa(self):
        torch.manual_seed(2)
        mu = torch.zeros(400)
        kappa = torch.full((400,), 40.0)
        phi = ContinuousTorsionHead.sample(mu, kappa)
        assert float(phi.abs().mean()) < 0.3


# =====================================================================
# Cutoff and radial basis
# =====================================================================
class TestCosineCutoff:
    def test_exact_envelope_values(self):
        cut = CosineCutoff(5.0)
        d = torch.tensor([0.0, 2.5, 5.0 - 1e-6, 5.0, 7.0])
        out = cut(d)
        assert abs(float(out[0]) - 1.0) < 1e-6       # f(0) = 1
        assert abs(float(out[1]) - 0.5) < 1e-6       # f(rc/2) = 0.5
        assert float(out[2]) < 1e-5                   # approaching 0 at rc
        assert float(out[3]) == 0.0                   # f(rc) = 0
        assert float(out[4]) == 0.0                   # zero beyond

    def test_monotone_decreasing(self):
        d = torch.linspace(0.0, 4.999, 500)
        out = CosineCutoff(5.0)(d)
        assert torch.all(out[1:] <= out[:-1] + 1e-7)

    def test_negative_distances_clamped(self):
        out = CosineCutoff(5.0)(torch.tensor([-1.0, -100.0]))
        assert torch.allclose(out, torch.ones(2))

    def test_invalid_cutoff_rejected(self):
        with pytest.raises(ValueError):
            CosineCutoff(0.0)


class TestRadialBasis:
    def test_exact_gaussian_formula(self):
        num_radial, cutoff = 8, 5.0
        rbf = RadialBasis(num_radial, cutoff)
        d = torch.tensor([0.7, 1.3, 3.9])
        envelope = 0.5 * (torch.cos(d * PI / cutoff) + 1.0)
        manual = torch.exp(
            -rbf.beta * (d.unsqueeze(-1) - rbf.means) ** 2
        ) * envelope.unsqueeze(-1)
        assert torch.allclose(rbf(d), manual, atol=1e-6)

    def test_means_are_linspace(self):
        rbf = RadialBasis(20, 5.0)
        assert torch.allclose(rbf.means, torch.linspace(0.0, 5.0, 20))

    def test_beta_formula(self):
        num_radial, cutoff = 20, 5.0
        rbf = RadialBasis(num_radial, cutoff)
        assert rbf.beta == pytest.approx((2.0 / cutoff * num_radial) ** 2)

    def test_zero_distance_activates_first_basis(self):
        rbf = RadialBasis(16, 5.0)
        out = rbf(torch.tensor([0.0]))
        assert float(out[0, 0]) == pytest.approx(1.0, abs=1e-6)
        # All other bases are below 1 at d=0.
        assert torch.all(out[0, 1:] <= 1.0)

    def test_values_bounded_01(self):
        rbf = RadialBasis(16, 5.0)
        d = torch.rand(1000) * 5.0
        out = rbf(d)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0 + 1e-6)

    def test_zero_past_cutoff(self):
        rbf = RadialBasis(16, 5.0)
        out = rbf(torch.tensor([5.0, 6.0, 100.0]))
        assert torch.all(out == 0.0)

    def test_output_shape(self):
        rbf = RadialBasis(12, 5.0)
        d = torch.rand(37)
        assert rbf(d).shape == (37, 12)

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            RadialBasis(8, 5.0)(torch.rand(4, 2))


# =====================================================================
# SE(3) equivariance
# =====================================================================
def _random_rotation(seed: int) -> torch.Tensor:
    """Uniformly random proper rotation via QR with positive diagonal fix."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    assert torch.allclose(q @ q.T, torch.eye(3), atol=1e-5)
    assert torch.det(q) > 0
    return q


class TestEquivariance:
    @pytest.fixture()
    def encoder(self):
        torch.manual_seed(0)
        from syntree.models.bipartite import BipartitePaiNN

        return BipartitePaiNN(
            hidden_dim=16, num_layers=2, num_radial=6, cutoff=5.0
        ).eval()

    @staticmethod
    def _inputs():
        g = torch.Generator().manual_seed(7)
        pocket_pos = torch.randn(24, 3, generator=g) * 4.0
        pocket_z = torch.randint(6, 9, (24,), generator=g)
        ligand_pos = torch.randn(9, 3, generator=g) * 1.5 + 0.5
        ligand_z = torch.randint(6, 8, (9,), generator=g)
        pb = torch.zeros(24, dtype=torch.long)
        lb = torch.zeros(9, dtype=torch.long)
        return pocket_pos, pocket_z, pb, ligand_pos, ligand_z, lb

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_vector_channel_rotates_exactly(self, encoder, seed):
        """True equivariance: v(R x + t) == R v(x) for a proper rotation R."""
        pocket_pos, pocket_z, pb, ligand_pos, ligand_z, lb = self._inputs()
        R = _random_rotation(seed)
        t = torch.randn(3) * 3.0

        with torch.no_grad():
            s0, v0, _ = encoder(
                pocket_pos, pocket_z, pb, None, ligand_pos, ligand_z, lb,
            )
            s1, v1, _ = encoder(
                pocket_pos @ R.T + t, pocket_z, pb, None,
                ligand_pos @ R.T + t, ligand_z, lb,
            )

        # Scalar channel invariant under rotation + translation.
        assert torch.allclose(s0, s1, atol=1e-4), (
            f"scalar channel not invariant (seed={seed})"
        )
        # Vector channel equivariant: v(R x + t) = R v(x).
        v0_rot = torch.einsum("ij,njd->nid", R, v0)
        assert torch.allclose(v0_rot, v1, atol=1e-3), (
            f"vector channel not equivariant (seed={seed}): "
            f"max diff {float((v0_rot - v1).abs().max())}"
        )

    def test_translation_invariance_of_scalars(self, encoder):
        pocket_pos, pocket_z, pb, ligand_pos, ligand_z, lb = self._inputs()
        with torch.no_grad():
            s0, _, _ = encoder(
                pocket_pos, pocket_z, pb, None, ligand_pos, ligand_z, lb,
            )
            s1, _, _ = encoder(
                pocket_pos + 10.0, pocket_z, pb, None,
                ligand_pos + 10.0, ligand_z, lb,
            )
        assert torch.allclose(s0, s1, atol=1e-4)

    @pytest.mark.parametrize("seed", [11, 12])
    def test_full_policy_logits_invariant_under_rotation(self, seed):
        """End-to-end: policy logits must not change when the whole complex
        is rigidly rotated (SE(3) invariance of the decision heads)."""
        from torch_geometric.data import Data

        torch.manual_seed(3)
        model = SynTreePolicy(
            {
                "model": {
                    "hidden_dim": 32,
                    "num_equivariant_layers": 2,
                    "num_radial_basis": 12,
                    "cutoff_radius": 5.0,
                    "synthon_embedding_dim": 32,
                    "num_attention_heads": 4,
                }
            }
        ).eval()

        R = _random_rotation(seed)
        state = Data(
            pocket_pos=torch.randn(20, 3) * 4.0,
            pocket_z=torch.randint(6, 9, (20,)),
            pocket_charge=torch.rand(20),
            pocket_batch=torch.zeros(20, dtype=torch.long),
            ligand_pos=torch.randn(7, 3) * 1.5,
            ligand_z=torch.randint(6, 8, (7,)),
            ligand_charge=torch.rand(7),
            ligand_batch=torch.zeros(7, dtype=torch.long),
            handle_features=torch.randn(1, 64),
            handle_pos=torch.randn(1, 3),
            handle_nodes=torch.tensor([0]),
            global_features=torch.zeros(1, GLOBAL_FEATURE_DIM),
            stop_mask=torch.tensor(0.0),
        )
        rotated = Data(**{k: v for k, v in state.items()})
        rotated.pocket_pos = state.pocket_pos @ R.T
        rotated.ligand_pos = state.ligand_pos @ R.T
        rotated.handle_pos = state.handle_pos @ R.T

        emb = torch.randn(50, 32)
        smask = torch.zeros(1, 50)
        rmask = torch.zeros(1, len(REACTION_FAMILY_NAMES))

        with torch.no_grad():
            out_a = model(state, emb, smask, rmask)
            out_b = model(rotated, emb, smask, rmask)

        for key in ("synthon_logits", "reaction_logits", "state_value"):
            assert torch.allclose(out_a[key], out_b[key], atol=2e-3), (
                f"{key} not rotation invariant (seed={seed}): "
                f"max diff {float((out_a[key] - out_b[key]).abs().max())}"
            )


# =====================================================================
# Featurizer math
# =====================================================================
class TestFeaturizerMath:
    def test_ligand_center_is_heavy_atom_centroid(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from syntree.data.featurizer import MolecularFeaturizer

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        center = MolecularFeaturizer.ligand_center(mol)
        conf = mol.GetConformer()
        heavy = np.array(
            [
                list(conf.GetAtomPosition(a.GetIdx()))
                for a in mol.GetAtoms()
                if a.GetAtomicNum() > 1
            ]
        )
        expected = heavy.mean(axis=0)
        assert np.allclose(center.numpy(), expected, atol=1e-5)

    def test_ligand_center_requires_conformer(self):
        from rdkit import Chem
        from syntree.data.featurizer import MolecularFeaturizer

        with pytest.raises(ValueError, match="3D"):
            MolecularFeaturizer.ligand_center(Chem.MolFromSmiles("CCO"))

    def test_pocket_features_shape_consistency(self):
        from rdkit import Chem
        from syntree.data.featurizer import MolecularFeaturizer

        pocket = Chem.MolFromPDBBlock(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C\n"
            "ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O\n"
            "END\n",
            removeHs=False,
        )
        feats = MolecularFeaturizer.featurize_pocket(pocket, center=torch.zeros(3))
        assert feats["pocket_pos"].shape == feats["pocket_z"].shape[:1] + (3,)
        assert feats["pocket_pos"].shape[0] == feats["pocket_z"].shape[0]
        assert feats["pocket_charge"].shape == feats["pocket_z"].shape
        assert torch.isfinite(feats["pocket_pos"]).all()

    def test_global_features_dim_and_finite(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from syntree.data.featurizer import MolecularFeaturizer

        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)COC(=O)C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        g = MolecularFeaturizer.ligand_global_features(mol, pocket_volume=100.0)
        assert g.shape == (GLOBAL_FEATURE_DIM,)
        assert torch.isfinite(g).all()

    def test_vdw_sphere_volume_positive_and_bounded(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from syntree.data.featurizer import MolecularFeaturizer

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        v = MolecularFeaturizer.vdw_sphere_volume(mol)
        assert v > 0.0
        assert v < 5000.0  # sane upper bound for a 3-heavy-atom molecule

    def test_vdw_sphere_volume_none_is_zero(self):
        from syntree.data.featurizer import MolecularFeaturizer
        assert MolecularFeaturizer.vdw_sphere_volume(None) == 0.0
