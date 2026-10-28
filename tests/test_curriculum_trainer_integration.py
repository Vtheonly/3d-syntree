"""End-to-end curriculum integration through the real trainer (CPU, tiny).

Runs the actual ResilientTrainer on the synthetic smoke dataset with the
curriculum enabled and verifies the wiring contract from
docs/CURRICULUM.md:
  * steps/epoch are identical with and without curriculum (budget-neutral),
  * per-epoch metrics entries carry the curriculum fields,
  * checkpoints carry the curriculum state (extra_state) and a resumed
    trainer restores the stage,
  * per-level validation metrics are emitted,
  * console logs expose config, counts, and progression.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syntree.engine.trainer import ResilientTrainer  # noqa: E402
from syntree.models.policy import SynTreePolicy  # noqa: E402


CURRICULUM_CFG = {
    "enabled": True,
    "n_levels": 9,
    "n_stages": 3,
    "min_weight": 0.05,
    "sigma": 1.8,
    "seed": 42,
    "milestone_metric": "val_loss",
    "lower_better": True,
    "improve_thresh": 0.005,
    "patience": 1,
    "feature_sample_stride": 1,
    "cache_path": None,  # filled per-test with the workspace tmp dir
}


def make_config(tiny_config_with_assets, tmp_path, curriculum=True, epochs=2):
    cfg = json.loads(json.dumps(tiny_config_with_assets))
    cfg["data"]["synthetic_samples"] = 12
    cfg["training"]["max_epochs"] = epochs
    if curriculum:
        cur = json.loads(json.dumps(CURRICULUM_CFG))
        cur["cache_path"] = str(tmp_path / "curriculum_levels.json")
        cfg["curriculum"] = cur
    return cfg


def build_trainer(cfg, tmp_path, auto_resume=False, seed=0):
    torch.manual_seed(seed)
    model = SynTreePolicy(cfg)
    return ResilientTrainer(
        model, cfg, torch.device("cpu"),
        auto_resume=auto_resume, output_dir=str(tmp_path),
    )


class TestStepsPerEpochInvariance:
    def test_curriculum_keeps_step_count(self, tiny_config_with_assets, tmp_path):
        cfg_off = make_config(tiny_config_with_assets, tmp_path, curriculum=False)
        cfg_on = make_config(tiny_config_with_assets, tmp_path, curriculum=True)
        trainer_off = build_trainer(cfg_off, tmp_path / "off")
        trainer_on = build_trainer(cfg_on, tmp_path / "on")
        # identical batch count per epoch -> LR schedule and budget unchanged
        assert len(trainer_off.loader) == len(trainer_on.loader)
        # the sampler draws exactly len(dataset) indices per epoch
        assert trainer_on.loader.sampler.num_samples == len(trainer_on.dataset)
        assert trainer_on.loader.sampler.num_samples == len(trainer_off.dataset)


class TestCurriculumRun:
    @pytest.fixture()
    def curriculum_run(self, tiny_config_with_assets, tmp_path):
        cfg = make_config(tiny_config_with_assets, tmp_path)
        trainer = build_trainer(cfg, tmp_path)
        summary = trainer.train()
        return {"cfg": cfg, "trainer": trainer, "summary": summary,
                "output_dir": str(tmp_path)}

    def test_run_completes_finite(self, curriculum_run):
        summary = curriculum_run["summary"]
        assert summary["epochs_completed"] == 2
        assert summary["final_train_loss"] is not None
        assert all(
            isinstance(v, float) and v == v  # not NaN
            for v in [summary["final_train_loss"], summary["best_val_loss"]]
        )

    def test_history_carries_curriculum_fields(self, curriculum_run):
        path = os.path.join(curriculum_run["output_dir"], "history.json")
        history = json.load(open(path))
        assert len(history) == 2
        for entry in history:
            assert "curriculum_stage" in entry
            assert "curriculum_center" in entry
            assert "curriculum_level_weights" in entry
            assert "curriculum_level_counts" in entry
            weights = entry["curriculum_level_weights"]
            assert abs(sum(weights.values()) - 1.0) < 1e-6
            # floor respected in the sampled mixture
            for lv, w in weights.items():
                assert w >= 0.05 - 1e-9
            # every sampled index belongs to a train level pool
            counts = entry["curriculum_level_counts"]
            assert sum(counts.values()) == len(curriculum_run["trainer"].dataset)

    def test_per_level_val_metrics_present(self, curriculum_run):
        path = os.path.join(curriculum_run["output_dir"], "history.json")
        history = json.load(open(path))
        level_keys = [
            key for entry in history for key in entry
            if key.startswith("L") and "/val_loss" in key
        ]
        assert level_keys, "per-level validation metrics must be reported"

    def test_metrics_jsonl_has_curriculum_rows(self, curriculum_run):
        path = os.path.join(curriculum_run["output_dir"], "metrics.jsonl")
        rows = [json.loads(line) for line in open(path)]
        assert rows
        assert all("curriculum_stage" in row for row in rows)
        assert all("phase" in row for row in rows)

    def test_checkpoint_carries_curriculum_state(self, curriculum_run):
        ckpt_dir = os.path.join(curriculum_run["output_dir"], "checkpoints")
        ckpts = [
            name for name in os.listdir(ckpt_dir)
            if name.startswith("checkpoint_epoch_") and name.endswith(".pt")
        ]
        assert ckpts
        latest = max(
            ckpts,
            key=lambda n: int(n.replace("checkpoint_epoch_", "").replace(".pt", "")),
        )
        payload = torch.load(
            os.path.join(ckpt_dir, latest), map_location="cpu", weights_only=False
        )
        extra = payload.get("extra_state") or {}
        assert "curriculum" in extra
        assert "stage" in extra["curriculum"]["sampler"]

    def test_console_logs_expose_curriculum(
        self, tiny_config_with_assets, tmp_path, capsys
    ):
        # Inline run: capsys reliably captures this test's own phase.
        cfg = make_config(tiny_config_with_assets, tmp_path)
        build_trainer(cfg, tmp_path).train()
        out = capsys.readouterr().out
        assert "[curriculum] enabled:" in out
        assert "train level counts:" in out
        assert "sampled:" in out
        assert "stage" in out

    def test_stage_progression_monotone(self, curriculum_run):
        path = os.path.join(curriculum_run["output_dir"], "history.json")
        history = json.load(open(path))
        stages = [entry["curriculum_stage"] for entry in history]
        assert stages == sorted(stages), "stage may never go backwards"


class TestResumeRestoresCurriculumState:
    def test_auto_resume_restores_stage(
        self, tiny_config_with_assets, tmp_path, capsys
    ):
        cfg = make_config(tiny_config_with_assets, tmp_path)
        first = build_trainer(cfg, tmp_path)
        first.train()
        # force a non-zero stage for the resume assertion
        stage = first.curriculum.sampler.stage

        second = build_trainer(cfg, tmp_path, auto_resume=True)
        out = capsys.readouterr().out
        assert "RESUMED" in out
        assert second.curriculum is not None
        assert second.curriculum.sampler.stage == stage
        assert second.start_epoch == 2


class TestFreshStartWithCurriculum:
    def test_fresh_curriculum_run_archives_previous_state(
        self, tiny_config_with_assets, tmp_path, capsys
    ):
        cfg = make_config(tiny_config_with_assets, tmp_path)
        build_trainer(cfg, tmp_path).train()

        cfg2 = make_config(tiny_config_with_assets, tmp_path)
        trainer2 = build_trainer(cfg2, tmp_path, auto_resume=False)
        out = capsys.readouterr().out
        assert "FRESH START CONFIRMED" in out
        assert trainer2.start_epoch == 0
        archives = list((tmp_path / "checkpoints" / "archive").glob("*/"))
        assert archives, "first run must be archived, not deleted"
