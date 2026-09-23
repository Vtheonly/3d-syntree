"""Unit tests for the resilient trainer (short CPU runs)."""

from __future__ import annotations

import json
import os

import pytest
import torch

from syntree.engine.trainer import ResilientTrainer, seed_everything
from syntree.models.policy import SynTreePolicy


@pytest.fixture()
def trained_artifacts(tiny_config_with_assets, tmp_path):
    """Run a 2-epoch training in a temp workspace; return paths + config."""
    cfg = json.loads(json.dumps(tiny_config_with_assets))
    cfg["data"]["synthetic_samples"] = 12
    cfg["training"]["max_epochs"] = 2

    torch.manual_seed(0)
    model = SynTreePolicy(cfg)
    trainer = ResilientTrainer(
        model, cfg, torch.device("cpu"),
        auto_resume=False, output_dir=str(tmp_path),
    )
    summary = trainer.train()
    return {
        "summary": summary,
        "output_dir": str(tmp_path),
        "config": cfg,
        "model": model,
    }


class TestSeedEverything:
    def test_reproducible_tensor_generation(self):
        seed_everything(123)
        a = torch.rand(10)
        seed_everything(123)
        b = torch.rand(10)
        assert torch.equal(a, b)


class TestTrainingRun:
    def test_summary_structure(self, trained_artifacts):
        s = trained_artifacts["summary"]
        assert s["epochs_completed"] == 2
        assert s["final_train_loss"] > 0

    def test_metrics_written(self, trained_artifacts):
        out = trained_artifacts["output_dir"]
        assert os.path.exists(os.path.join(out, "latest_metrics.json"))
        assert os.path.exists(os.path.join(out, "history.json"))
        assert os.path.exists(os.path.join(out, "metrics.jsonl"))

        history = json.load(open(os.path.join(out, "history.json")))
        assert len(history) == 2
        assert {"epoch", "train_loss", "train_synthon_ce",
                "train_torsion_nll"} <= set(history[0])

    def test_jsonl_log_readable(self, trained_artifacts):
        out = trained_artifacts["output_dir"]
        lines = open(os.path.join(out, "metrics.jsonl")).read().strip().splitlines()
        assert len(lines) >= 2
        for line in lines:
            record = json.loads(line)
            assert "wall_time" in record and "train_loss" in record

    def test_checkpoints_written(self, trained_artifacts):
        ckpt_dir = os.path.join(trained_artifacts["output_dir"], "checkpoints")
        files = os.listdir(ckpt_dir)
        assert "manifest.json" in files
        epochs = [f for f in files if f.startswith("checkpoint_epoch_")]
        assert len(epochs) >= 1
        manifest = json.load(open(os.path.join(ckpt_dir, "manifest.json")))
        assert manifest["latest_epoch"] >= 1

    def test_checkpoint_content(self, trained_artifacts):
        ckpt_dir = os.path.join(trained_artifacts["output_dir"], "checkpoints")
        manifest = json.load(open(os.path.join(ckpt_dir, "manifest.json")))
        path = os.path.join(ckpt_dir, f"checkpoint_epoch_{manifest['latest_epoch']}.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state
        assert "rng_states" in state
        assert state["epoch"] == manifest["latest_epoch"]


class TestResume:
    def test_auto_resume_continues(self, trained_artifacts):
        cfg = trained_artifacts["config"]
        model = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            model, cfg, torch.device("cpu"),
            auto_resume=True, output_dir=trained_artifacts["output_dir"],
        )
        assert trainer.start_epoch == 2  # continues after the 2 done epochs
        # With max_epochs=2 the loop body never executes; that is the point.
        summary = trainer.train()
        assert summary["epochs_completed"] == 2

    def test_fresh_directory_starts_at_zero(self, tiny_config_with_assets, tmp_path):
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        model = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            model, cfg, torch.device("cpu"),
            auto_resume=True, output_dir=str(tmp_path / "fresh"),
        )
        assert trainer.start_epoch == 0
        assert trainer.global_step == 0

    def test_restored_weights_match(self, trained_artifacts):
        cfg = trained_artifacts["config"]
        original = trained_artifacts["model"]
        fresh = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            fresh, cfg, torch.device("cpu"),
            auto_resume=True, output_dir=trained_artifacts["output_dir"],
        )
        for p_orig, p_fresh in zip(original.parameters(), fresh.parameters()):
            assert torch.allclose(p_orig, p_fresh, atol=1e-6)


class TestTimeBudget:
    def test_budget_expires_gracefully(self, tiny_config_with_assets, tmp_path):
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["training"]["time_budget_hours"] = 1e-6  # ~3.6 ms: expires instantly
        cfg["training"]["max_epochs"] = 5
        model = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            model, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        )
        summary = trainer.train()
        # The run must terminate without raising and still write artifacts.
        assert "epochs_completed" in summary
        assert os.path.exists(os.path.join(str(tmp_path), "history.json"))
