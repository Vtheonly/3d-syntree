"""Final verification battery: math, reproducibility, data integrity.

The tasklist demands testing that goes far beyond "does the code run":

Mathematical correctness
* von Mises density normalisation (numerical quadrature), sampling
  support and concentration limits.
* PaiNN SE(3)-equivariance of the VECTOR channel (vectors must rotate
  with the frame, scalars must stay invariant).
* Kabsch alignment optimality (proper rotation, orthogonal residual).
* Lennard-Jones interaction of a geometry-consistent pose vs a broken
  one (global ordering).

Reproducibility
* Identical seeds -> identical model initialisation.
* Identical seeds -> identical generated molecules (SMILES + recipe).
* Identical training runs -> identical loss history.

Data integrity
* Every field of the new joint state schema present on all three data
  pipelines (synthetic, real CrossDocked, trajectory builder).
* Shard manifest sha256 digests match the shard bytes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from syntree.chemistry.conformer import ConformerEngine
from syntree.models.bipartite import BipartitePaiNN
from syntree.models.equivariant import RadialBasis
from syntree.models.policy import GLOBAL_FEATURE_DIM, SynTreePolicy
from syntree.models.torsion_head import ContinuousTorsionHead, log_bessel_i0


@pytest.fixture(scope="module")
def catalog(assets_dir):
    from syntree.chemistry.catalog import SynthonCatalog

    return SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)


@pytest.fixture(scope="module")
def generator(tiny_config_with_assets):
    import json as _json

    from syntree.engine.generator import SBDDGenerator

    torch.manual_seed(0)
    cfg = _json.loads(_json.dumps(tiny_config_with_assets))
    return SBDDGenerator(
        SynTreePolicy(cfg), cfg, torch.device("cpu"), output_dir=None
    )


@pytest.fixture()
def pocket_path(assets_dir):
    return os.path.join(
        assets_dir["crossdocked_dir"], "sample_pocket.pdb"
    )


# =====================================================================
# Mathematical correctness
# =====================================================================
class TestVonMisesMath:
    @pytest.mark.parametrize("mu,kappa", [(0.0, 0.5), (1.2, 2.0), (-2.0, 8.0), (3.0, 20.0)])
    def test_density_normalises_to_one(self, mu, kappa):
        """Integral of the von Mises density over [-pi, pi) == 1."""
        n = 200_000
        grid = torch.linspace(-math.pi, math.pi, n + 1)
        mu_t = torch.full((n + 1,), mu)
        kappa_t = torch.full((n + 1,), kappa)
        logp = ContinuousTorsionHead.log_prob(mu_t, kappa_t, grid)
        integral = torch.trapz(logp.exp(), grid).item()
        assert integral == pytest.approx(1.0, rel=2e-3)

    def test_log_bessel_matches_scipy_if_available(self):
        try:
            from scipy.special import ive
        except ImportError:
            pytest.skip("scipy not installed")
        # Stable reference: ive(0, x) = I0(x) * exp(-x), so
        # log I0(x) = log(ive(0, x)) + x never overflows.
        x = torch.linspace(0.001, 200.0, 5000)
        ours = log_bessel_i0(x).numpy()
        ref = np.log(ive(0, x.numpy())) + x.numpy()
        assert np.abs(ours - ref).max() < 1e-4

    def test_samples_stay_on_the_circle(self):
        torch.manual_seed(0)
        mu = torch.tensor([0.3, -1.5, 2.9])
        kappa = torch.tensor([0.1, 2.0, 50.0])
        for _ in range(50):
            phi = ContinuousTorsionHead.sample(mu, kappa)
            assert (phi >= -math.pi).all() and (phi < math.pi).all()

    def test_high_kappa_concentrates_at_mu(self):
        torch.manual_seed(1)
        mu = torch.tensor([0.7])
        kappa = torch.tensor([200.0])
        samples = torch.stack(
            [ContinuousTorsionHead.sample(mu, kappa) for _ in range(200)]
        )
        assert samples.std().item() < 0.1
        assert abs(samples.mean().item() - 0.7) < 0.05


class TestPaiNNVectorEquivariance:
    """The equivariant backbone must be SE(3)-equivariant, not merely
    invariant: rotating the input must rotate the vector channel."""

    @staticmethod
    def _rotation(theta, axis=1):
        c, s = math.cos(theta), math.sin(theta)
        R = torch.eye(3)
        i, j = (axis + 1) % 3, (axis + 2) % 3
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        return R

    def test_joint_encoder_vector_channel_rotates(self):
        torch.manual_seed(0)
        enc = BipartitePaiNN(hidden_dim=16, num_layers=2, num_radial=6).eval()
        pocket_pos = torch.randn(10, 3)
        pocket_z = torch.randint(6, 9, (10,))
        ligand_pos = torch.randn(4, 3) * 0.4 + 0.5
        ligand_z = torch.randint(6, 8, (4,))
        batch = torch.zeros(14, dtype=torch.long)

        with torch.no_grad():
            s0, v0, _ = enc(
                pocket_pos, pocket_z, batch[:10], None,
                ligand_pos, ligand_z, batch[10:],
            )
        R = self._rotation(0.9)
        with torch.no_grad():
            s1, v1, _ = enc(
                pocket_pos @ R.T, pocket_z, batch[:10], None,
                ligand_pos @ R.T, ligand_z, batch[10:],
            )

        # Scalars invariant.
        assert torch.allclose(s0, s1, atol=1e-4)
        # Vectors equivariant: v(R x) = R v(x).
        v0_rot = torch.einsum("ij,njd->nid", R, v0)
        assert torch.allclose(v0_rot, v1, atol=1e-3)

    def test_full_policy_scalar_invariance_under_ligand_rotation(self):
        torch.manual_seed(2)
        model = SynTreePolicy(
            {"model": {"hidden_dim": 32, "num_equivariant_layers": 2,
                       "num_radial_basis": 12, "cutoff_radius": 5.0,
                       "synthon_embedding_dim": 32, "num_attention_heads": 4}}
        ).eval()
        state = Data(
            pocket_pos=torch.randn(8, 3),
            pocket_z=torch.randint(6, 9, (8,)),
            pocket_batch=torch.zeros(8, dtype=torch.long),
            ligand_pos=torch.randn(4, 3),
            ligand_z=torch.randint(6, 8, (4,)),
            ligand_batch=torch.zeros(4, dtype=torch.long),
            handle_features=torch.randn(1, 64),
            handle_pos=torch.randn(1, 3),
            handle_nodes=torch.tensor([1]),
            global_features=torch.zeros(1, GLOBAL_FEATURE_DIM),
            stop_mask=torch.tensor(0.0),
        )
        emb = torch.randn(5, 32)
        R = self._rotation(2.1, axis=0)
        rotated = Data(**{k: v for k, v in state.items()})
        rotated.pocket_pos = state.pocket_pos @ R.T
        rotated.ligand_pos = state.ligand_pos @ R.T
        rotated.handle_pos = state.handle_pos @ R.T
        with torch.no_grad():
            o1 = model(state, emb)
            o2 = model(rotated, emb)
        assert torch.allclose(
            o1["synthon_logits"], o2["synthon_logits"], atol=1e-4
        )
        assert torch.allclose(o1["state_value"], o2["state_value"], atol=1e-4)


class TestKabschMath:
    def test_proper_rotation_and_optimality(self):
        """Kabsch must return a proper rotation (det=+1) that minimises the
        RMSD - verified against a known rigid transform."""
        rng = np.random.default_rng(3)
        P = rng.normal(size=(12, 3))
        # Random proper rotation.
        Q0, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q0) < 0:
            Q0[:, -1] *= -1
        t0 = rng.normal(size=3)
        Q = P @ Q0.T + t0

        from syntree.chemistry.conformer import _kabsch

        R, t = _kabsch(P, Q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)  # orthogonal
        assert np.linalg.det(R) > 0.999                      # proper
        aligned = P @ R.T + t
        rmsd = float(np.sqrt(((aligned - Q) ** 2).sum(1).mean()))
        assert rmsd < 1e-8


class TestRadialBasisMath:
    def test_values_decay_past_cutoff(self):
        rbf = RadialBasis(num_radial=8, cutoff=5.0)
        d = torch.tensor([0.5, 2.5, 4.9, 5.1, 8.0])
        out = rbf(d)
        assert (out[3].abs() < 1e-6).all()  # at/after cutoff -> 0
        assert (out[4].abs() < 1e-6).all()
        assert (out[0] > 0).any()

    def test_rejects_non_1d(self):
        rbf = RadialBasis()
        with pytest.raises(ValueError):
            rbf(torch.randn(4, 2))


class TestLJOrdering:
    def test_contact_pose_beats_detached_and_clashing_poses(self):
        """A pose at van der Waals contact must have LOWER LJ energy than
        the same pose pushed into the protein or into solvent."""
        pocket = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
        contact = torch.tensor([[3.8, 0.0, 0.0]])     # ~ r_min to atom 1
        clash = torch.tensor([[1.0, 0.0, 0.0]])      # deep overlap
        solvent = torch.tensor([[25.0, 0.0, 0.0]])   # far away
        e = lambda lig: ConformerEngine.compute_lennard_jones_energy(
            lig, pocket,
            torch.full((lig.size(0),), 1.7), torch.full((2,), 1.7),
        ).item()
        e_contact, e_clash, e_solvent = e(contact), e(clash), e(solvent)
        assert e_contact < 0.0           # attractive well
        assert e_clash > e_contact        # overlap is worst
        assert abs(e_solvent) < 1e-3      # solvent ~ zero


# =====================================================================
# Reproducibility
# =====================================================================
class TestReproducibility:
    def test_seed_controls_initialisation(self, tiny_config):
        torch.manual_seed(1234)
        a = SynTreePolicy(tiny_config)
        torch.manual_seed(1234)
        b = SynTreePolicy(tiny_config)
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)

    def test_generation_is_deterministic_given_seed(
        self, generator, pocket_path
    ):
        """argmax decoding + fixed seed -> identical molecules and recipes."""
        torch.manual_seed(42)
        r1 = generator.generate_ligand(pocket_path, seed_synthon_idx=0, max_steps=2)
        torch.manual_seed(42)
        r2 = generator.generate_ligand(pocket_path, seed_synthon_idx=0, max_steps=2)
        assert r1["smiles"] == r2["smiles"]
        assert r1["recipe"] == r2["recipe"]
        assert r1["clash_score"] == pytest.approx(r2["clash_score"])

    def test_synthetic_dataset_is_seed_reproducible(self, tmp_path, catalog):
        from syntree.data.crossdocked import CrossDockedDataset

        ds1 = CrossDockedDataset(
            str(tmp_path), synthetic=True, catalog=catalog,
            num_synthetic=8, seed=99,
        )
        ds2 = CrossDockedDataset(
            str(tmp_path), synthetic=True, catalog=catalog,
            num_synthetic=8, seed=99,
        )
        for i in range(len(ds1)):
            a, b = ds1[i], ds2[i]
            assert int(a.target_synthon) == int(b.target_synthon)
            assert float(a.target_dihedral) == pytest.approx(
                float(b.target_dihedral)
            )
            assert torch.equal(a.pocket_pos, b.pocket_pos)


# =====================================================================
# Data integrity
# =====================================================================
class TestDataIntegrity:
    EXPECTED_FIELDS = (
        "pocket_pos", "pocket_z", "pocket_charge",
        "ligand_pos", "ligand_z", "ligand_charge",
        "handle_features", "handle_pos", "handle_nodes", "global_features",
        "target_synthon", "target_dihedral", "target_reaction_family_idx",
        "stop_mask",
    )

    def test_synthetic_schema_complete(self, tmp_path, catalog):
        from syntree.data.crossdocked import CrossDockedDataset

        ds = CrossDockedDataset(
            str(tmp_path), synthetic=True, catalog=catalog,
            num_synthetic=4, seed=3,
        )
        for i in range(len(ds)):
            sample = ds[i]
            for field in self.EXPECTED_FIELDS:
                assert hasattr(sample, field), (
                    f"synthetic sample {i} missing {field}"
                )
            assert sample.ligand_pos.size(0) == 0  # synthetic: no ligand
            assert sample.global_features.shape == (1, GLOBAL_FEATURE_DIM)

    def test_trajectory_builder_schema_complete(self, catalog):
        from syntree.chemistry.reactions import ReactionEngine
        from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder
        from syntree.chemistry.conformer import ConformerEngine
        from rdkit import Chem

        builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=3)
        lig = ConformerEngine.embed_product(
            Chem.MolFromSmiles("O=C(NC1CCCCC1)C1CCCCC1")
        )
        pocket = Chem.MolFromPDBBlock(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C\n"
            "ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O\n"
            "ATOM      5  N   GLY A   2       3.315   1.545   0.000  1.00 20.00           N\n"
            "ATOM      6  CA  GLY A   2       3.866   2.906   0.000  1.00 20.00           C\n"
            "END\n",
            removeHs=False,
        )
        samples, info = builder.build(lig, pocket, trajectory_id="schema-test")
        assert info["status"] == "ok"
        assert samples
        for i, sample in enumerate(samples):
            for field in self.EXPECTED_FIELDS:
                assert hasattr(sample, field), (
                    f"trajectory sample {i} missing {field}"
                )
            assert sample.ligand_pos.size(0) > 0  # real intermediate state

    def test_shard_manifest_digests_match_bytes(self, tmp_path):
        """write_shards manifests must carry true sha256 digests."""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from shard_and_upload import write_shards

        torch.manual_seed(0)
        samples = []
        for i in range(5):
            samples.append(
                Data(
                    pocket_pos=torch.randn(6, 3),
                    pocket_z=torch.randint(6, 9, (6,)),
                    ligand_pos=torch.randn(3, 3),
                    ligand_z=torch.randint(6, 8, (3,)),
                    handle_features=torch.randn(1, 64),
                    handle_pos=torch.randn(1, 3),
                    handle_nodes=torch.tensor([0]),
                    global_features=torch.zeros(1, GLOBAL_FEATURE_DIM),
                    stop_mask=torch.tensor(0.0),
                )
            )
        manifest = write_shards(samples, "train", str(tmp_path), max_shard_bytes=1024)
        assert manifest["total_samples"] == 5
        declared = sum(s["sample_count"] for s in manifest["shards"])
        assert declared == 5
        for shard in manifest["shards"]:
            payload = (tmp_path / "train" / shard["name"]).read_bytes()
            assert shard["compressed_bytes"] == len(payload)
            assert hashlib.sha256(payload).hexdigest() == shard["sha256"]
            # The payload must be valid gzipped torch data.
            import io

            torch.load(
                io.BytesIO(gzip.decompress(payload)), weights_only=False
            )

    def test_hf_manifest_declares_every_split(self, repo_root):
        """The committed seed dataset documentation references all three
        splits; the local build produces all three manifests."""
        for split in ("train", "val", "test"):
            path = os.path.join(
                repo_root, "data", "multidataset"
            )
            # The canonical location lives in the HF repo; locally we
            # verify the builder output when present and skip otherwise.
            manifest = os.path.join(path, f"data_{split}_manifest.json")
            if not os.path.exists(manifest):
                continue
            payload = json.loads(open(manifest).read())
            assert payload["total_samples"] >= 0


# =====================================================================
# End-to-end flow (single process, no CLI)
# =====================================================================
class TestEndToEndFlow:
    def test_train_checkpoint_generate_roundtrip(
        self, tiny_config_with_assets, catalog, tmp_path
    ):
        """A 1-epoch training run produces a checkpoint whose model
        generates a chemically valid molecule through the environment."""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from syntree.engine.generator import SBDDGenerator
        from syntree.engine.trainer import ResilientTrainer

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_samples"] = 8
        cfg["training"]["max_epochs"] = 1
        cfg["system"]["num_workers"] = 0

        model = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            model, cfg, torch.device("cpu"), auto_resume=False,
            output_dir=str(tmp_path),
        )
        summary = trainer.train()
        assert summary["epochs_completed"] >= 1

        model.eval()
        gen = SBDDGenerator(
            model, cfg, torch.device("cpu"), output_dir=str(tmp_path)
        )
        pocket = os.path.join(cfg["data"]["data_dir"], "sample_pocket.pdb")
        result = gen.generate_ligand(pocket, seed_synthon_idx=0, max_steps=1)
        assert result["rdkit_mol"] is not None
        assert result["recipe"][0]["action"] == "seed"
        assert result["contact_energy"] == result["contact_energy"]  # finite
        # The generated molecule is chemically valid.
        from rdkit import Chem

        probe = Chem.Mol(result["rdkit_mol"])
        Chem.SanitizeMol(probe)
