"""Fresh-start semantics (2026-10-29 restart-from-scratch mandate).

Covers the three safeguards required by docs/HANDOFF_2026-10-29.md:
  1. --fresh / training.fresh_start archives previous local checkpoints
     (nothing deleted) and explicitly confirms a from-scratch start,
  2. --resume-auto + any fresh-start request is a hard error,
  3. restore_latest REFUSES checkpoints with non-finite model/optimizer
     tensors instead of silently applying them (the 2026-10-28 incident:
     "Restored verified checkpoint: epoch 15" -> loss=NaN from batch 0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pytest
import torch

from syntree.utils.checkpoint import CheckpointManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import resolve_start_mode  # noqa: E402


HF_OFF = {"huggingface": {"enabled": False, "repo_id": "", "push_every_n_epochs": 2}}


def _manager(tmp_path, config_extra=None):
    cfg = dict(HF_OFF)
    if config_extra:
        cfg.update(config_extra)
    return CheckpointManager(cfg, ckpt_dir=str(tmp_path / "checkpoints"))


def _seed_checkpoint_state(mgr, epoch=0, weights=None):
    """Write a structurally consistent local checkpoint state."""
    ckpt_path = mgr.ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
    payload = {
        "epoch": epoch,
        "step": (epoch + 1) * 7,
        "model_state_dict": (
            weights if weights is not None else {"w": torch.ones(3)}
        ),
        "optimizer_state_dict": None,
        "metrics": {"val_loss": 1.234},
        "model_config": {"hidden_dim": 32},
    }
    torch.save(payload, ckpt_path)
    mgr._atomic_json_write(
        mgr.ckpt_dir / mgr.MANIFEST_NAME,
        {"latest_epoch": epoch, "best_metric": 1.234},
    )
    mgr._atomic_json_write(
        mgr.ckpt_dir / mgr.PROGRESS_NAME,
        {
            "version": 1,
            "mode": "epoch",
            "completed_epochs": list(range(epoch + 1)),
            "last_sequential_epoch": epoch,
            "history": [{"epoch": e} for e in range(epoch + 1)],
        },
    )
    return payload


class TestArchiveLocal:
    def test_archives_without_deleting(self, tmp_path):
        mgr = _manager(tmp_path)
        _seed_checkpoint_state(mgr, epoch=0)
        _seed_checkpoint_state(mgr, epoch=1)
        original = {
            p.name: p.stat().st_size
            for p in sorted(mgr.ckpt_dir.glob("checkpoint_epoch_*.pt"))
        }
        assert len(original) == 2

        dest = mgr.archive_local(reason="fresh-start")

        assert dest is not None and dest.is_dir()
        # every artifact survived the move (byte sizes preserved)
        archived = {p.name for p in sorted(dest.glob("*.pt"))}
        assert archived == set(original)
        for name, size in original.items():
            assert (dest / name).stat().st_size == size
        # manifest + progress were archived too
        assert (dest / mgr.MANIFEST_NAME).exists()
        assert (dest / mgr.PROGRESS_NAME).exists()
        # nothing remains at the authoritative paths -> no silent restore
        assert not (mgr.ckpt_dir / mgr.MANIFEST_NAME).exists()
        assert not (mgr.ckpt_dir / mgr.PROGRESS_NAME).exists()
        assert not list(mgr.ckpt_dir.glob("checkpoint_epoch_*.pt"))

    def test_nothing_to_archive_returns_none(self, tmp_path):
        mgr = _manager(tmp_path)
        assert mgr.archive_local() is None

    def test_resume_after_archive_starts_from_epoch_zero(self, tmp_path):
        mgr = _manager(tmp_path)
        _seed_checkpoint_state(mgr, epoch=0)
        mgr.archive_local(reason="fresh-start")
        model = torch.nn.Linear(3, 2)
        epoch, step, best = mgr.restore_latest(model)
        assert epoch == 0 and step == 0 and best == float("inf")


class TestRestoreRefusesPoisonedCheckpoint:
    def test_nan_model_weights_refused_loudly(self, tmp_path):
        mgr = _manager(tmp_path)
        _seed_checkpoint_state(
            mgr,
            epoch=0,
            weights={"w": torch.tensor([1.0, float("nan"), 3.0])},
        )
        model = torch.nn.Linear(3, 2)
        with pytest.raises(RuntimeError) as excinfo:
            mgr.restore_latest(model)
        msg = str(excinfo.value)
        assert "non-finite" in msg
        assert "REFUSED" in msg
        # the poisoned evidence file is preserved untouched
        assert (mgr.ckpt_dir / "checkpoint_epoch_0.pt").exists()

    def test_nan_optimizer_state_refused(self, tmp_path):
        mgr = _manager(tmp_path)
        _seed_checkpoint_state(mgr, epoch=0)
        payload = torch.load(
            mgr.ckpt_dir / "checkpoint_epoch_0.pt", weights_only=False
        )
        payload["optimizer_state_dict"] = {
            "state": {0: {"exp_avg": torch.full((3,), float("inf"))}}
        }
        torch.save(payload, mgr.ckpt_dir / "checkpoint_epoch_0.pt")
        model = torch.nn.Linear(3, 2)
        with pytest.raises(RuntimeError):
            mgr.restore_latest(model, optimizer=torch.optim.AdamW(model.parameters()))

    def test_finite_checkpoint_still_restores(self, tmp_path):
        mgr = _manager(tmp_path)
        _seed_checkpoint_state(mgr, epoch=1)
        model = torch.nn.Linear(3, 2)
        next_epoch, step, best = mgr.restore_latest(model)
        assert next_epoch == 2
        assert step == (1 + 1) * 7
        assert best == pytest.approx(1.234)


class TestStartModeResolution:
    def _args(self, fresh=False, resume=False):
        return argparse.Namespace(fresh=fresh, resume_auto=resume)

    def test_cli_fresh_conflicts_with_resume(self):
        with pytest.raises(SystemExit):
            resolve_start_mode(self._args(fresh=True, resume=True), {})

    def test_config_fresh_conflicts_with_resume(self):
        with pytest.raises(SystemExit):
            resolve_start_mode(
                self._args(resume=True),
                {"training": {"fresh_start": True}},
            )

    def test_fresh_without_resume_ok(self):
        mode = resolve_start_mode(self._args(fresh=True), {})
        assert mode == {"fresh": True, "resume": False, "source": "cli --fresh"}

    def test_config_fresh_alone_ok(self):
        mode = resolve_start_mode(
            self._args(), {"training": {"fresh_start": True}}
        )
        assert mode["fresh"] is True
        assert mode["source"] == "config training.fresh_start"

    def test_neither_means_plain_fresh(self):
        mode = resolve_start_mode(self._args(), {})
        assert mode["fresh"] is False and mode["resume"] is False


class TestTrainerFreshRun:
    def test_fresh_run_archives_and_confirms(
        self, tiny_config_with_assets, tmp_path, capsys
    ):
        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_samples"] = 12
        cfg["training"]["max_epochs"] = 1

        torch.manual_seed(0)
        model = SynTreePolicy(cfg)
        ResilientTrainer(
            model, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        ).train()

        ckpt_dir = tmp_path / "checkpoints"
        before = {p.name for p in sorted(ckpt_dir.glob("*.pt"))}
        assert before, "first run must produce checkpoints"

        torch.manual_seed(0)
        model2 = SynTreePolicy(cfg)
        trainer2 = ResilientTrainer(
            model2, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        )
        out = capsys.readouterr().out
        assert "FRESH START CONFIRMED" in out
        assert "no checkpoint restored" in out
        assert "ARCHIVED" in out or "archived" in out
        assert trainer2.start_epoch == 0
        assert trainer2._first_nonfinite_param() is None
        assert "param health" in out

        archives = list((ckpt_dir / "archive").glob("*/"))
        assert archives, "previous run state must be archived, not deleted"
        archived_names = {p.name for p in archives[-1].glob("*.pt")}
        assert before.issubset(archived_names)

    def test_fresh_resume_invariant_never_restores_old_weights(
        self, tiny_config_with_assets, tmp_path
    ):
        """A fresh run's weights must differ from a previous run's weights."""
        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_samples"] = 12
        cfg["training"]["max_epochs"] = 1

        torch.manual_seed(0)
        first = SynTreePolicy(cfg)
        ResilientTrainer(
            first, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        ).train()
        trained_w = {n: p.detach().clone() for n, p in first.named_parameters()}

        # Pristine reference: same seed, no training, no restore.
        torch.manual_seed(0)
        pristine = SynTreePolicy(cfg)

        torch.manual_seed(0)
        second = SynTreePolicy(cfg)
        ResilientTrainer(
            second, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        )
        # The fresh trainer must equal the pristine re-initialization and
        # must NOT equal the previously trained weights (proof that no old
        # state leaked into the fresh run). Constant/untrained parameters
        # (e.g. zero-initialized flag embeddings) are excluded - they never
        # change, so they cannot distinguish fresh from restored.
        pristine_params = dict(pristine.named_parameters())
        changed = [
            name for name, trained in trained_w.items()
            if not torch.allclose(trained.detach(), pristine_params[name].detach())
        ]
        assert changed, "the first training run must have moved some weights"
        second_params = dict(second.named_parameters())
        for name in changed:
            assert torch.allclose(
                second_params[name].detach(), pristine_params[name].detach()
            ), f"{name}: fresh init differs from the pristine re-init"
            assert not torch.allclose(
                second_params[name].detach(), trained_w[name].detach()
            ), f"{name}: fresh run appears to have restored trained weights"
