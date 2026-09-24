"""Fail-closed real-data guard tests for ResilientTrainer.

The tasklist demands the trainer refuse to train when:
* the training dataset is a 2-sample stub (len < min_real_samples),
* the local backend silently fell back to the synthetic smoke dataset.

These tests verify the guard fires with an actionable message, fails CLOSED
when the config keys are missing, and stays quiet when explicitly disabled.
"""

from __future__ import annotations

import json
import sys

import pytest
import torch
from torch.utils.data import Dataset

from syntree.engine import trainer as trainer_mod
from syntree.engine.trainer import ResilientTrainer
from syntree.models.policy import SynTreePolicy


class _StubShardedHF(Dataset):
    """Minimal stand-in for ShardedHuggingFaceDataset."""

    def __init__(self, repo_id="", split="train", cache_dir="./hf_cache",
                 revision="main", token=None, max_cached_shards=4, n: int = 0):
        self.repo_id = repo_id
        self.split = split
        self.n = int(n)
        assert repo_id, "repo_id must be validated"

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        raise AssertionError("guard tests must never iterate the dataset")


def _patch_hf_backend(monkeypatch, n_train: int, n_val: int = 2):
    calls = {"train": 0, "val": 0}
    real_class = trainer_mod.ShardedHuggingFaceDataset

    class _Factory(_StubShardedHF):
        def __init__(self, repo_id, split, cache_dir, revision, token,
                     max_cached_shards):
            calls[split] += 1
            super().__init__(
                repo_id=repo_id, split=split, n=n_train if split == "train" else n_val,
            )

    # Only fake out the loader for train/val construction, keep isinstance
    # semantics intact for the sampler branch.
    monkeypatch.setattr(trainer_mod, "ShardedHuggingFaceDataset", _Factory)
    return calls


def _guard_config(tiny_config_with_assets, **data_overrides) -> dict:
    cfg = json.loads(json.dumps(tiny_config_with_assets))
    cfg["data"]["backend"] = "huggingface"
    cfg["data"]["huggingface"] = {
        "repo_id": "JJKK1212/3d-syntree-multidataset",
        "revision": "main",
        "cache_dir": "./hf_cache",
        "max_cached_shards": 4,
    }
    cfg["data"].pop("synthetic_fallback", None)
    cfg["data"].pop("synthetic_samples", None)
    cfg["training"]["max_epochs"] = 1
    cfg["training"]["auto_scale"] = {"enabled": False}
    cfg["data"].update(data_overrides)
    return cfg


def _make_trainer(cfg, tmp_path):
    torch.manual_seed(0)
    model = SynTreePolicy(cfg)
    return ResilientTrainer(
        model, cfg, torch.device("cpu"),
        auto_resume=False, output_dir=str(tmp_path / "exp"),
    )


class TestStubDatasetGuard:
    def test_two_sample_stub_is_rejected(self, tiny_config_with_assets, monkeypatch, tmp_path):
        """The exact production bug: a 2-sample HF stub must abort training."""
        _patch_hf_backend(monkeypatch, n_train=2, n_val=2)
        cfg = _guard_config(
            tiny_config_with_assets,
            require_real_data=True, min_real_samples=50,
        )
        with pytest.raises(RuntimeError) as excinfo:
            _make_trainer(cfg, tmp_path)
        msg = str(excinfo.value)
        assert "FATAL" in msg
        assert "min required: 50" in msg
        assert "build_full_dataset.py" in msg
        assert "2 training samples" in msg

    def test_guard_fails_closed_when_keys_missing(self, tiny_config_with_assets, monkeypatch, tmp_path):
        """No require_real_data / min_real_samples keys -> defaults True / 50."""
        _patch_hf_backend(monkeypatch, n_train=2, n_val=2)
        cfg = _guard_config(tiny_config_with_assets)
        cfg["data"].pop("require_real_data", None)
        cfg["data"].pop("min_real_samples", None)
        with pytest.raises(RuntimeError, match="FATAL"):
            _make_trainer(cfg, tmp_path)

    def test_counts_only_train_split(self, tiny_config_with_assets, monkeypatch, tmp_path):
        """A small val split must not trip the guard (only train is checked)."""
        _patch_hf_backend(monkeypatch, n_train=60, n_val=2)
        cfg = _guard_config(
            tiny_config_with_assets,
            require_real_data=True, min_real_samples=50,
        )
        trainer = _make_trainer(cfg, tmp_path)
        assert len(trainer.dataset) == 60
        assert len(trainer.val_dataset) == 2

    def test_real_dataset_passes_guard(self, tiny_config_with_assets, monkeypatch, tmp_path):
        _patch_hf_backend(monkeypatch, n_train=50, n_val=5)
        cfg = _guard_config(
            tiny_config_with_assets,
            require_real_data=True, min_real_samples=50,
        )
        trainer = _make_trainer(cfg, tmp_path)
        assert len(trainer.dataset) == 50

    def test_guard_disabled_allows_stub(self, tiny_config_with_assets, monkeypatch, tmp_path):
        """Explicit opt-out (smoke mode) keeps small runs possible."""
        _patch_hf_backend(monkeypatch, n_train=2, n_val=2)
        cfg = _guard_config(
            tiny_config_with_assets,
            require_real_data=False, min_real_samples=50,
        )
        trainer = _make_trainer(cfg, tmp_path)
        assert len(trainer.dataset) == 2

    def test_boundary_is_inclusive(self, tiny_config_with_assets, monkeypatch, tmp_path):
        """len == min_real_samples must pass (only strictly smaller fails)."""
        _patch_hf_backend(monkeypatch, n_train=49, n_val=4)
        cfg = _guard_config(
            tiny_config_with_assets,
            require_real_data=True, min_real_samples=49,
        )
        trainer = _make_trainer(cfg, tmp_path)
        assert len(trainer.dataset) == 49


class TestSyntheticFallbackGuard:
    def test_local_synthetic_fallback_rejected(self, tiny_config_with_assets, tmp_path):
        """Production mode must refuse the silent synthetic smoke dataset."""
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_fallback"] = True
        cfg["data"]["synthetic_samples"] = 100  # even a large synthetic set...
        cfg["data"]["require_real_data"] = True
        cfg["data"]["min_real_samples"] = 50
        cfg["data"].pop("backend", None)  # local path
        with pytest.raises(RuntimeError) as excinfo:
            _make_trainer(cfg, tmp_path)
        assert "SYNTHETIC" in str(excinfo.value)

    def test_local_synthetic_allowed_when_opted_out(self, tiny_config_with_assets, tmp_path):
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["require_real_data"] = False
        cfg["data"].pop("backend", None)
        trainer = _make_trainer(cfg, tmp_path)
        assert bool(trainer.dataset.use_synthetic) is True


class TestTrajectoryBackendGuard:
    def test_trajectory_backend_guarded(self, tiny_config_with_assets, tmp_path, workspace):
        """trajectory_pt backend samples come from real data, but a stub
        below the threshold must still be refused."""
        from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder
        from syntree.chemistry.catalog import SynthonCatalog
        from syntree.chemistry.conformer import ConformerEngine
        from rdkit import Chem

        catalog = SynthonCatalog(
            tiny_config_with_assets["data"]["synthon_catalog_path"],
            embedding_dim=32,
        )
        builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=2)
        lig = ConformerEngine.embed_product(
            Chem.MolFromSmiles("O=C(NC1CCCCC1)C1CCCCC1")
        )
        pocket = Chem.MolFromPDBFile(
            tiny_config_with_assets["data"]["data_dir"] + "/sample_pocket.pdb",
            removeHs=False,
        )
        # 'stub-guard-b' (crc32%1000=31) lands in train; 'val-trajectory-3'
        # (868) lands in val, so both TrajectoryDataset splits construct.
        states, meta = builder.build(lig, pocket, trajectory_id="stub-guard-b")
        val_states, _ = builder.build(lig, pocket, trajectory_id="val-trajectory-3")
        assert states, "fixture must produce at least one state"
        states = states + val_states

        path = str(tmp_path / "stub_trajectory.pt")
        torch.save(states, path)

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"].pop("backend", None)
        cfg["data"]["trajectory_dataset_path"] = path
        cfg["data"]["require_real_data"] = True
        cfg["data"]["min_real_samples"] = 50

        with pytest.raises(RuntimeError, match="FATAL"):
            _make_trainer(cfg, tmp_path)


class TestProductionConfigContract:
    def test_train_colab_config_is_fail_closed(self, repo_root):
        path = f"{repo_root}/configs/train_colab_12h.json"
        cfg = json.load(open(path))
        assert cfg["data"]["require_real_data"] is True
        assert cfg["data"]["min_real_samples"] == 50
        assert cfg["data"]["synthetic_fallback"] is False
        assert cfg["data"]["synthetic_samples"] == 0
        assert cfg["data"]["backend"] == "huggingface"
        assert cfg["data"]["huggingface"]["repo_id"] == "JJKK1212/3d-syntree-multidataset"

    def test_default_config_declares_guard_key(self, repo_root):
        """The reference profile must state its guard position explicitly."""
        cfg = json.load(open(f"{repo_root}/configs/default_config.json"))
        assert "require_real_data" in cfg["data"]
