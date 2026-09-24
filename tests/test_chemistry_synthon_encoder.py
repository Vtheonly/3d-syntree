"""Unit tests for the 3D-pharmacophore synthon encoder (tasklist Priority 1)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from rdkit import Chem

from syntree.chemistry.catalog import SynthonCatalog
from syntree.chemistry.synthon_encoder import (
    ENCODER_VERSION,
    ENGINEERED_DIM,
    RAW_DIM,
    SUPPORTED_ENCODERS,
    build_projection,
    catalog_ids_hash,
    compute_pharm3d_descriptor,
    encode_catalog_morgan2d,
    encode_catalog_pharm3d,
)

SMILES = ["CCO", "CCBr", "CS(=O)(=O)Cl", "C1CC(N)CC1", "c1ccccc1", "CCCCO"]


# ---------------------------------------------------------------------------
# Descriptor contract
# ---------------------------------------------------------------------------
class TestDescriptorContract:
    def test_layout_constants(self):
        assert RAW_DIM == 2048 + ENGINEERED_DIM + 1
        assert ENGINEERED_DIM == 89
        assert ENCODER_VERSION == "pharm3d-v1"
        assert SUPPORTED_ENCODERS == ("pharm3d", "morgan2d")

    def test_descriptor_shape_and_finiteness(self):
        desc, has3d = compute_pharm3d_descriptor("CCO", 42)
        assert desc.shape == (RAW_DIM,)
        assert desc.dtype == np.float32
        assert bool(np.isfinite(desc).all())
        assert has3d is True

    def test_descriptor_deterministic(self):
        d1, _ = compute_pharm3d_descriptor("CCCO", 7)
        d2, _ = compute_pharm3d_descriptor("CCCO", 7)
        assert np.array_equal(d1, d2)

    def test_descriptor_seed_sensitivity(self):
        d1, _ = compute_pharm3d_descriptor("CCCO", 7)
        d2, _ = compute_pharm3d_descriptor("CCCO", 8)
        assert not np.array_equal(d1, d2)

    def test_identical_molecules_get_identical_descriptors(self):
        """Duplicate catalog rows must not diverge (crc32 content seed)."""
        d1, _ = compute_pharm3d_descriptor("C1CC(N)CC1", 42)
        d2, _ = compute_pharm3d_descriptor("C1CC(N)CC1", 42)
        assert np.array_equal(d1, d2)

    def test_distinct_molecules_differ(self):
        d1, _ = compute_pharm3d_descriptor("CCO", 42)
        d2, _ = compute_pharm3d_descriptor("CCCO", 42)
        assert not np.array_equal(d1, d2)

    def test_3d_block_is_informative(self):
        """The 3D sub-block (shape/charges/directions) must carry signal."""
        desc, has3d = compute_pharm3d_descriptor("CS(=O)(=O)Cl", 42)
        assert has3d
        # shape block [47, 53) + dipole [53, 54) + charges [54, 58)
        engineered = desc[2048:2048 + ENGINEERED_DIM]
        assert float(np.linalg.norm(engineered[47:58])) > 1e-6

    def test_unparseable_smiles_yields_zero_descriptor(self):
        desc, has3d = compute_pharm3d_descriptor("not_a_smiles", 42)
        assert not np.any(desc)
        assert has3d is False


# ---------------------------------------------------------------------------
# Projection mathematics
# ---------------------------------------------------------------------------
class TestProjection:
    def test_orthonormal_columns_up_to_jl_scale(self):
        raw, dim = RAW_DIM, 64
        p = build_projection(raw, dim, 42)
        assert p.shape == (raw, dim)
        gram = p.T @ p
        expected = np.eye(dim) * (raw / dim)
        assert np.abs(gram - expected).max() < 1e-3

    def test_projection_deterministic(self):
        assert np.array_equal(build_projection(256, 16, 3), build_projection(256, 16, 3))

    def test_projection_seed_changes(self):
        assert not np.array_equal(build_projection(256, 16, 3), build_projection(256, 16, 4))

    def test_distance_preservation(self):
        """Johnson-Lindenstrauss property: the scaled orthonormal projection
        preserves pairwise distances in expectation (relative distortion for
        k=128 is ~sqrt(2/k) ~ 12.5%; assert a generous 30% bound)."""
        q = build_projection(512, 128, 5)
        rng = np.random.default_rng(0)
        points = rng.standard_normal((8, 512))
        projected = points @ q
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d_orig = np.linalg.norm(points[i] - points[j])
                d_proj = np.linalg.norm(projected[i] - projected[j])
                assert abs(d_proj - d_orig) / d_orig < 0.30

    def test_projected_norm_preserved_in_expectation(self):
        """E[||P x||^2] == ||x||^2 for the JL-scaled projection."""
        q = build_projection(512, 64, 9)
        rng = np.random.default_rng(2)
        ratios = []
        for _ in range(64):
            x = rng.standard_normal(512)
            ratios.append((np.linalg.norm(q.T @ x) / np.linalg.norm(x)) ** 2)
        assert abs(float(np.mean(ratios)) - 1.0) < 0.15

    def test_projection_is_bounded(self):
        q = build_projection(512, 32, 5)
        x = np.random.default_rng(1).standard_normal(512)
        y = q.T @ x
        # JL scaling: projected norm concentrates around ||x||.
        assert 0.4 * np.linalg.norm(x) < np.linalg.norm(y) < 1.6 * np.linalg.norm(x)

    def test_wide_fallback(self):
        """embedding_dim > raw_dim falls back to a scaled Gaussian."""
        q = build_projection(8, 16, 42)
        assert q.shape == (8, 16)
        assert np.isfinite(q).all()


# ---------------------------------------------------------------------------
# Catalog-level encoding
# ---------------------------------------------------------------------------
class TestCatalogEncoding:
    def test_embeddings_normalized_and_deterministic(self):
        emb1, info1 = encode_catalog_pharm3d(SMILES, 32, 42)
        emb2, _ = encode_catalog_pharm3d(SMILES, 32, 42)
        assert emb1.shape == (len(SMILES), 32)
        assert torch.equal(emb1, emb2)
        norms = emb1.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        assert info1["num_with_conformer"] == len(SMILES)
        assert info1["encoder_version"] == ENCODER_VERSION

    def test_duplicate_rows_identical_embeddings(self):
        emb, _ = encode_catalog_pharm3d(["CCO", "CCO", "CCBr"], 16, 42)
        assert torch.equal(emb[0], emb[1])
        assert not torch.equal(emb[0], emb[2])

    def test_seed_changes_embeddings(self):
        emb1, _ = encode_catalog_pharm3d(SMILES, 16, 42)
        emb2, _ = encode_catalog_pharm3d(SMILES, 16, 43)
        assert not torch.equal(emb1, emb2)

    def test_morgan2d_matches_legacy_projection(self):
        """The legacy encoder must stay bit-for-bit reproducible."""
        from rdkit import DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        rng = np.random.default_rng(42)
        projection = rng.standard_normal((2048, 24)).astype(np.float32)
        projection /= np.sqrt(2048)
        rows = np.zeros((len(SMILES), 24), dtype=np.float32)
        for i, s in enumerate(SMILES):
            mol = Chem.MolFromSmiles(s)
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp = gen.GetFingerprint(mol)
            vec = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, vec)
            rows[i] = vec @ projection
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        legacy = torch.from_numpy(rows / norms)

        modern = encode_catalog_morgan2d(SMILES, 24, 42)
        assert torch.equal(legacy, modern)

    def test_pharm3d_differs_from_morgan2d(self):
        emb_3d, _ = encode_catalog_pharm3d(SMILES, 16, 42)
        emb_2d = encode_catalog_morgan2d(SMILES, 16, 42)
        assert not torch.equal(emb_3d, emb_2d)

    def test_vram_budget_math(self):
        """tasklist claim (decimal MB): 100k x 512-dim float32 ~ 204.8 MB,
        negligible on a 96 GB RTX 6000 Ada."""
        bytes_ = 100_000 * 512 * 4
        assert abs(bytes_ / 1_000_000 - 204.8) < 0.5
        assert bytes_ < 0.003 * 96 * 1024**3  # <0.3% of an RTX 6000 Ada

    def test_ids_hash_stable(self):
        assert catalog_ids_hash(["a", "b"]) == catalog_ids_hash(["a", "b"])
        assert catalog_ids_hash(["a", "b"]) != catalog_ids_hash(["b", "a"])


# ---------------------------------------------------------------------------
# SynthonCatalog integration (config plumbing + cache)
# ---------------------------------------------------------------------------
class TestCatalogIntegration:
    def test_default_encoder_is_pharm3d(self, assets_dir):
        cat = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=16)
        assert cat.encoder_name == "pharm3d"
        assert cat.embedding_signature["encoder"] == "pharm3d"
        assert cat.embedding_signature["encoder_version"] == ENCODER_VERSION
        assert cat.embedding_signature["dim"] == 16
        assert cat.embedding_signature["num_synthons"] == cat.num_synthons

    def test_unknown_encoder_rejected(self, assets_dir):
        with pytest.raises(ValueError, match="Unknown synthon encoder"):
            SynthonCatalog(assets_dir["catalog_path"], embedding_dim=8, encoder="uni-mol")

    def test_legacy_encoder_available(self, assets_dir):
        cat = SynthonCatalog(
            assets_dir["catalog_path"], embedding_dim=16, encoder="morgan2d"
        )
        assert cat.encoder_name == "morgan2d"
        assert cat.embeddings.shape == (cat.num_synthons, 16)

    def test_encoder_cache_roundtrip(self, assets_dir, tmp_path):
        import shutil as _shutil

        # Copy the catalog so cache files land in a writable tmp dir.
        dst = str(tmp_path / "catalog.parquet")
        _shutil.copy(assets_dir["catalog_path"], dst)

        cat1 = SynthonCatalog(dst, embedding_dim=16, encoder="pharm3d")
        cache = str(tmp_path / "catalog_embeddings_16d_pharm3d.h5")
        assert os.path.exists(cache)

        cat2 = SynthonCatalog(dst, embedding_dim=16, encoder="pharm3d")
        assert cat2.embedding_info.get("from_cache") is True
        assert torch.equal(cat1.embeddings, cat2.embeddings)

    def test_encoder_cache_invalidated_on_seed_change(self, tmp_path):
        import pandas as pd

        path = str(tmp_path / "c.parquet")
        pd.DataFrame(
            {
                "id": ["A", "B"],
                "smiles": ["CCO", "CCBr"],
                "fsp3": [1.0, 1.0],
                "mw": [46.0, 109.0],
                "primary_handle": ["alcohol", "alkyl_halide"],
            }
        ).to_parquet(path, index=False)

        cat1 = SynthonCatalog(path, embedding_dim=8, seed=1)
        cat2 = SynthonCatalog(path, embedding_dim=8, seed=2)
        assert not torch.equal(cat1.embeddings, cat2.embeddings)
        # Each seed gets its own cache file.
        assert os.path.exists(str(tmp_path / "c_embeddings_8d_pharm3d.h5"))

    def test_encoder_cache_invalidated_on_catalog_change(self, tmp_path):
        import pandas as pd

        path = str(tmp_path / "c.parquet")
        frame = pd.DataFrame(
            {
                "id": ["A", "B"],
                "smiles": ["CCO", "CCBr"],
                "fsp3": [1.0, 1.0],
                "mw": [46.0, 109.0],
                "primary_handle": ["alcohol", "alkyl_halide"],
            }
        )
        frame.to_parquet(path, index=False)
        cat1 = SynthonCatalog(path, embedding_dim=8)

        # Rewrite the catalog with a different id set.
        frame.loc[1, "id"] = "B2"
        frame.to_parquet(path, index=False)
        cat2 = SynthonCatalog(path, embedding_dim=8)
        assert cat2.embedding_signature["ids_hash"] != cat1.embedding_signature["ids_hash"]

    def test_cache_skipped_gracefully_on_readonly_dir(self, tmp_path, monkeypatch):
        import pandas as pd

        path = str(tmp_path / "ro" / "c.parquet")
        os.makedirs(str(tmp_path / "ro"), exist_ok=True)
        pd.DataFrame(
            {
                "id": ["A"],
                "smiles": ["CCO"],
                "fsp3": [1.0],
                "mw": [46.0],
                "primary_handle": ["alcohol"],
            }
        ).to_parquet(path, index=False)

        real_replace = os.replace

        def deny_replace(src, dst, *args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(os, "replace", deny_replace)
        cat = SynthonCatalog(path, embedding_dim=8)  # must not raise
        assert cat.embeddings.shape == (1, 8)
        monkeypatch.setattr(os, "replace", real_replace)

    def test_cache_disabled(self, tmp_path):
        import pandas as pd

        path = str(tmp_path / "c.parquet")
        pd.DataFrame(
            {
                "id": ["A"],
                "smiles": ["CCO"],
                "fsp3": [1.0],
                "mw": [46.0],
                "primary_handle": ["alcohol"],
            }
        ).to_parquet(path, index=False)
        SynthonCatalog(path, embedding_dim=8, cache_embeddings=False)
        assert not os.path.exists(str(tmp_path / "c_embeddings_8d_pharm3d.h5"))


# ---------------------------------------------------------------------------
# Checkpoint signature plumbing
# ---------------------------------------------------------------------------
class TestCheckpointSignature:
    def _config(self):
        return {
            "huggingface": {"enabled": False},
            "system": {"seed": 1},
        }

    def test_signature_roundtrip(self, tmp_path):
        from syntree.utils.checkpoint import CheckpointManager

        model = torch.nn.Linear(4, 3)
        mgr = CheckpointManager(self._config(), ckpt_dir=str(tmp_path))
        sig = {"encoder": "pharm3d", "dim": 8, "ids_hash": "abc", "num_synthons": 5}
        mgr.save_checkpoint(
            epoch=0, step=1, model=model, model_config={},
            catalog_signature=sig,
        )
        assert mgr.last_catalog_signature is None  # set only on restore

        model2 = torch.nn.Linear(4, 3)
        start, _, _ = mgr.restore_latest(model2)
        assert start == 1
        assert mgr.last_catalog_signature == sig

    def test_legacy_checkpoint_has_no_signature(self, tmp_path):
        from syntree.utils.checkpoint import CheckpointManager

        model = torch.nn.Linear(4, 3)
        mgr = CheckpointManager(self._config(), ckpt_dir=str(tmp_path))
        mgr.save_checkpoint(epoch=0, step=1, model=model)
        mgr.restore_latest(torch.nn.Linear(4, 3))
        assert mgr.last_catalog_signature is None

    def test_trainer_warns_on_encoder_mismatch(self, assets_dir, capsys):
        from syntree.engine.trainer import ResilientTrainer

        class FakeCkpt:
            last_catalog_signature = {
                "encoder": "morgan2d", "dim": 8, "ids_hash": "zzz",
                "num_synthons": 3,
            }

        class FakeCatalog:
            embedding_signature = {
                "encoder": "pharm3d", "dim": 8, "ids_hash": "abc",
                "num_synthons": 5,
            }

        trainer = object.__new__(ResilientTrainer)
        trainer.ckpt_manager = FakeCkpt()
        trainer.catalog = FakeCatalog()
        ResilientTrainer._warn_on_catalog_signature_mismatch(trainer)
        out = capsys.readouterr().out
        assert "morgan2d" in out and "pharm3d" in out and "WARNING" in out

    def test_trainer_silent_when_signature_matches(self, assets_dir, capsys):
        from syntree.engine.trainer import ResilientTrainer

        class FakeCkpt:
            last_catalog_signature = {
                "encoder": "pharm3d", "dim": 8, "ids_hash": "abc",
                "num_synthons": 5,
            }

        class FakeCatalog:
            embedding_signature = dict(FakeCkpt.last_catalog_signature)

        trainer = object.__new__(ResilientTrainer)
        trainer.ckpt_manager = FakeCkpt()
        trainer.catalog = FakeCatalog()
        ResilientTrainer._warn_on_catalog_signature_mismatch(trainer)
        assert "WARNING" not in capsys.readouterr().out

    def test_trainer_silent_for_legacy_checkpoints(self, assets_dir, capsys):
        from syntree.engine.trainer import ResilientTrainer

        class FakeCkpt:
            last_catalog_signature = None

        trainer = object.__new__(ResilientTrainer)
        trainer.ckpt_manager = FakeCkpt()
        trainer.catalog = None
        ResilientTrainer._warn_on_catalog_signature_mismatch(trainer)
        assert capsys.readouterr().out == ""
