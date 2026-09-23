"""Unit tests for GPU auto-scaling.

Covers the model-size tiers, the batch-size autotuner, the CPU no-op path of
the trainer's auto-scale hook, and persistence of the (possibly scaled) model
architecture inside checkpoints. CUDA-only code paths degrade to no-ops on
CPU, which is exactly what the CPU tests assert.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from syntree.utils.hardware import (
    MODEL_TIERS,
    autotune_batch_size,
    free_vram_bytes,
    scale_model_config,
)


BASE_MODEL_CFG = {
    "hidden_dim": 128,
    "num_equivariant_layers": 4,
    "num_radial_basis": 20,
    "cutoff_radius": 5.0,
    "synthon_embedding_dim": 128,
    "num_attention_heads": 4,
    "max_atomic_number": 100,
    "dropout": 0.1,
}


class TestScaleModelConfig:
    @pytest.mark.parametrize(
        "free_gb,hidden,heads,layers",
        [
            (24.0, 256, 8, 8),  # A100-class
            (14.4, 256, 8, 8),  # T4 with most VRAM free
            (12.0, 256, 8, 8),  # tier boundary is inclusive
            (9.0, 192, 6, 6),   # mid tier
            (8.0, 192, 6, 6),   # tier boundary is inclusive
            (5.0, 160, 4, 5),   # small GPU
            (1.0, 128, 4, 4),   # portable default
            (0.0, 128, 4, 4),   # CPU fallback
        ],
    )
    def test_tier_selection(self, free_gb, hidden, heads, layers):
        scaled = scale_model_config(BASE_MODEL_CFG, free_gb=free_gb)
        assert scaled["hidden_dim"] == hidden
        assert scaled["num_attention_heads"] == heads
        assert scaled["num_equivariant_layers"] == layers
        assert scaled["synthon_embedding_dim"] == hidden  # must track hidden

    def test_does_not_mutate_input(self):
        original = dict(BASE_MODEL_CFG)
        scale_model_config(BASE_MODEL_CFG, free_gb=20.0)
        assert BASE_MODEL_CFG == original

    def test_preserves_non_tier_keys(self):
        scaled = scale_model_config(BASE_MODEL_CFG, free_gb=14.0)
        assert scaled["cutoff_radius"] == 5.0
        assert scaled["dropout"] == 0.1
        assert scaled["num_radial_basis"] == 20
        assert scaled["max_atomic_number"] == 100

    def test_all_tiers_build_valid_policy(self):
        from syntree.models.policy import SynTreePolicy

        for min_gb, overrides in MODEL_TIERS:
            cfg = {"model": scale_model_config(BASE_MODEL_CFG, free_gb=min_gb)}
            policy = SynTreePolicy(cfg)  # raises on invalid head/hidden combos
            assert policy.hidden_dim == overrides["hidden_dim"]

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
    def test_cpu_default_uses_portable_tier(self):
        scaled = scale_model_config(BASE_MODEL_CFG)  # free_gb=None -> query
        assert scaled["hidden_dim"] == 128


class TestAutotuneBatchSize:
    def test_cpu_noop_keeps_start_batch(self):
        result = autotune_batch_size(
            probe_fn=lambda b: None,
            dataset_size=1000,
            start_batch=16,
            device=torch.device("cpu"),
        )
        assert result["batch_size"] == 16
        assert result["peak_bytes"] == 0.0

    def test_non_cuda_device_noop(self):
        result = autotune_batch_size(
            probe_fn=lambda b: None,
            dataset_size=1000,
            start_batch=32,
            device=torch.device("cpu"),
        )
        assert result["batch_size"] == 32

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
    def test_free_vram_bytes_zero_on_cpu(self):
        assert free_vram_bytes() == 0


class TestTrainerAutoscale:
    def test_cpu_trainer_with_autoscale_enabled_is_noop(
        self, tiny_config_with_assets, tmp_path
    ):
        """On CPU the auto-scale hook must not touch batch/accum settings."""
        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["training"]["auto_scale"] = {
            "enabled": True,
            "target_vram_fraction": 0.85,
            "max_batch_size": 4096,
            "scale_model": True,
        }
        cfg["data"]["batch_size"] = 4
        model = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            model, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        )
        assert trainer.batch_size == 4
        assert trainer.accum_steps == cfg["data"]["accumulate_grad_batches"]

    def test_trainer_without_autoscale_key_unchanged(
        self, tiny_config_with_assets, tmp_path
    ):
        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        model = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            model, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        )
        assert trainer.batch_size == cfg["data"]["batch_size"]


class TestCheckpointModelConfig:
    def test_roundtrip(self, tmp_path):
        from syntree.utils.checkpoint import CheckpointManager

        mgr = CheckpointManager(
            {"huggingface": {"enabled": False}},
            ckpt_dir=str(tmp_path / "ckpt"),
        )
        model = torch.nn.Linear(4, 4)
        model_cfg = {"hidden_dim": 256, "num_attention_heads": 8}
        mgr.save_checkpoint(
            epoch=3, step=9, model=model, model_config=model_cfg
        )
        assert mgr.read_model_config() == model_cfg

    def test_missing_checkpoint_returns_none(self, tmp_path):
        from syntree.utils.checkpoint import CheckpointManager

        mgr = CheckpointManager(
            {"huggingface": {"enabled": False}},
            ckpt_dir=str(tmp_path / "empty"),
        )
        assert mgr.read_model_config() is None

    def test_legacy_checkpoint_without_config_returns_none(self, tmp_path):
        """Checkpoints written before architecture persistence -> None."""
        from syntree.utils.checkpoint import CheckpointManager

        ckpt_dir = tmp_path / "legacy"
        ckpt_dir.mkdir()
        torch.save(
            {"epoch": 0, "model_state_dict": {}},
            ckpt_dir / "checkpoint_epoch_0.pt",
        )
        with open(ckpt_dir / "manifest.json", "w") as f:
            json.dump({"latest_epoch": 0, "latest_step": 0}, f)

        mgr = CheckpointManager(
            {"huggingface": {"enabled": False}}, ckpt_dir=str(ckpt_dir)
        )
        assert mgr.read_model_config() is None

    def test_trainer_persists_model_config(self, tiny_config_with_assets, tmp_path):
        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy
        from syntree.utils.checkpoint import CheckpointManager

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["training"]["max_epochs"] = 1
        model = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            model, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        )
        trainer.train()

        mgr = CheckpointManager(
            {"huggingface": {"enabled": False}},
            ckpt_dir=str(tmp_path / "checkpoints"),
        )
        assert mgr.read_model_config() == cfg["model"]


class TestConfigsShipAutoScale:
    def test_colab_config_auto_scale_enabled(self, repo_root):
        path = os.path.join(repo_root, "configs", "train_colab_12h.json")
        with open(path) as f:
            cfg = json.load(f)
        auto = cfg["training"]["auto_scale"]
        assert auto["enabled"] is True
        assert 0.5 <= auto["target_vram_fraction"] <= 0.95
        # Fail-closed production contract (docs/tasklist.md): the Colab
        # profile must never fall back to synthetic data and must require a
        # real dataset above the stub threshold.
        assert cfg["data"]["synthetic_samples"] == 0
        assert cfg["data"]["synthetic_fallback"] is False
        assert cfg["data"]["require_real_data"] is True
        assert cfg["data"]["min_real_samples"] >= 50

    def test_default_config_auto_scale_documented(self, repo_root):
        path = os.path.join(repo_root, "configs", "default_config.json")
        with open(path) as f:
            cfg = json.load(f)
        # Present but disabled: the reference profile stays bit-reproducible.
        assert cfg["training"]["auto_scale"]["enabled"] is False
