"""Unit tests for checkpointing, hardware, and logging utilities."""

from __future__ import annotations

import json
import os

import pytest
import torch

from syntree.utils.checkpoint import CheckpointManager, verify_hf_sync
from syntree.utils.hardware import configure_runtime_environment, set_determinism
from syntree.utils.logger import StructuredLogger, configure_logging


@pytest.fixture()
def ckpt_cfg(tiny_config, tmp_path):
    cfg = json.loads(json.dumps(tiny_config))
    cfg["huggingface"]["enabled"] = False
    return cfg


@pytest.fixture()
def toy_model():
    return torch.nn.Linear(4, 2)


class TestCheckpointManager:
    def test_save_creates_checkpoint_and_manifest(self, ckpt_cfg, toy_model,
                                                  tmp_path):
        mgr = CheckpointManager(ckpt_cfg, ckpt_dir=str(tmp_path / "ck"))
        opt = torch.optim.Adam(toy_model.parameters())
        path = mgr.save_checkpoint(epoch=3, step=30, model=toy_model,
                                   optimizer=opt, metrics={"val_loss": 1.5})
        assert os.path.exists(path)
        manifest = json.load(open(str(tmp_path / "ck" / "manifest.json")))
        assert manifest["latest_epoch"] == 3
        assert manifest["latest_step"] == 30

    def test_save_then_restore_roundtrip(self, ckpt_cfg, tmp_path):
        torch.manual_seed(0)
        model_a = torch.nn.Linear(4, 2)
        opt_a = torch.optim.Adam(model_a.parameters(), lr=0.01)
        mgr = CheckpointManager(ckpt_cfg, ckpt_dir=str(tmp_path / "ck"))
        for _ in range(3):  # move optimizer state away from init
            opt_a.step()
        mgr.save_checkpoint(epoch=1, step=10, model=model_a, optimizer=opt_a,
                            metrics={"val_loss": 2.0})

        model_b = torch.nn.Linear(4, 2)
        opt_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
        mgr_b = CheckpointManager(ckpt_cfg, ckpt_dir=str(tmp_path / "ck"))
        start_epoch, step, best = mgr_b.restore_latest(model_b, opt_b)
        assert start_epoch == 2
        assert step == 10
        assert best == 2.0
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(pa, pb)
        assert opt_b.state_dict()["param_groups"][0]["lr"] == \
            opt_a.state_dict()["param_groups"][0]["lr"]

    def test_restore_fresh_returns_zeros(self, ckpt_cfg, tmp_path):
        mgr = CheckpointManager(ckpt_cfg, ckpt_dir=str(tmp_path / "empty"))
        model = torch.nn.Linear(4, 2)
        epoch, step, best = mgr.restore_latest(model)
        assert epoch == 0 and step == 0
        assert best == float("inf")

    def test_pruning_keeps_last_n(self, ckpt_cfg, toy_model, tmp_path):
        mgr = CheckpointManager(ckpt_cfg, keep_last_n=2,
                                ckpt_dir=str(tmp_path / "ck"))
        for epoch in range(5):
            mgr.save_checkpoint(epoch=epoch, step=epoch, model=toy_model)
        files = [f for f in os.listdir(str(tmp_path / "ck"))
                 if f.startswith("checkpoint_epoch_")]
        assert len(files) == 2
        assert "checkpoint_epoch_4.pt" in files
        assert "checkpoint_epoch_3.pt" in files
        assert "checkpoint_epoch_0.pt" not in files

    def test_rng_state_roundtrip(self, ckpt_cfg, tmp_path):
        """Restoring the saved RNG state replays the same random stream."""
        # Build the model FIRST: nn.Linear initialisation consumes RNG draws,
        # and we need a pristine seed-99 state at save time.
        model = torch.nn.Linear(2, 1)
        torch.manual_seed(99)
        mgr = CheckpointManager(ckpt_cfg, ckpt_dir=str(tmp_path / "ck"))
        mgr.save_checkpoint(epoch=0, step=0, model=model)

        # Destroy the stream.
        torch.manual_seed(7)
        _ = torch.rand(100)

        # Restore and draw.
        state = torch.load(str(tmp_path / "ck" / "checkpoint_epoch_0.pt"),
                           map_location="cpu", weights_only=False)
        CheckpointManager._restore_rng(state["rng_states"])
        after = torch.rand(5)

        # Reference: fresh seed 99 stream.
        torch.manual_seed(99)
        reference = torch.rand(5)
        assert torch.allclose(after, reference)

    def test_metrics_persisted(self, ckpt_cfg, toy_model, tmp_path):
        mgr = CheckpointManager(ckpt_cfg, ckpt_dir=str(tmp_path / "ck"))
        path = mgr.save_checkpoint(epoch=0, step=1, model=toy_model,
                                   metrics={"val_loss": 0.7, "val_acc": 0.4})
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["metrics"]["val_loss"] == 0.7

    def test_hf_disabled_without_token(self, tiny_config, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cfg = json.loads(json.dumps(tiny_config))
        cfg["huggingface"]["enabled"] = True
        cfg["huggingface"]["repo_id"] = ""
        mgr = CheckpointManager(cfg, ckpt_dir=str(tmp_path / "ck"))
        assert mgr.enabled is False  # graceful degradation


class TestVerifyHfSync:
    def test_no_token_reports_error(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        status = verify_hf_sync("user/repo")
        assert status["sync_ok"] is False
        assert "HF_TOKEN not set" in status["error"]

    def test_bad_repo_reports_error(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_invalid_token_for_tests")
        status = verify_hf_sync("definitely/not-a-real-repo-xyz")
        assert status["sync_ok"] is False
        assert "error" in status
        assert status["repo_id"] == "definitely/not-a-real-repo-xyz"


class TestHardware:
    def test_cpu_profile(self):
        info = configure_runtime_environment({"device": "auto"})
        assert info["device"] == "cpu"
        assert info["cuda_available"] is False
        assert info["precision"] == "fp32"

    def test_explicit_cpu(self):
        info = configure_runtime_environment({"device": "cpu"})
        assert info["device"] == "cpu"

    def test_cuda_requested_but_unavailable(self):
        info = configure_runtime_environment({"device": "cuda:0"})
        assert info["device"] == "cpu"  # graceful fallback

    def test_default_config(self):
        info = configure_runtime_environment()
        assert "torch_version" in info

    def test_set_determinism_reproducible(self):
        set_determinism(5, deterministic=True)
        a = torch.rand(3)
        set_determinism(5, deterministic=True)
        b = torch.rand(3)
        assert torch.equal(a, b)
        set_determinism(0, deterministic=False)  # restore benchmark mode


class TestStructuredLogger:
    def test_jsonl_roundtrip(self, tmp_path):
        log = StructuredLogger(str(tmp_path / "m.jsonl"), experiment_name="t")
        log.log({"epoch": 0, "loss": 1.0})
        log.log({"epoch": 1, "loss": 0.5})
        records = log.read_all()
        assert len(records) == 2
        assert records[0]["loss"] == 1.0
        assert records[1]["experiment"] == "t"
        assert "wall_time" in records[0]

    def test_corrupt_lines_skipped(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text('{"a": 1}\nnot json\n{"b": 2}\n')
        log = StructuredLogger(str(path))
        assert len(log.read_all()) == 2

    def test_creates_parent_dirs(self, tmp_path):
        log = StructuredLogger(str(tmp_path / "deep" / "nested" / "m.jsonl"))
        log.log({"x": 1})
        assert os.path.exists(str(tmp_path / "deep" / "nested" / "m.jsonl"))


class TestConfigureLogging:
    def test_idempotent(self):
        configure_logging()
        root1 = len(__import__("logging").getLogger().handlers)
        configure_logging()
        root2 = len(__import__("logging").getLogger().handlers)
        assert root1 == root2

    def test_file_handler(self, tmp_path):
        configure_logging(log_file=str(tmp_path / "run.log"))
        __import__("logging").getLogger("syntree.test").info("hello file")
        assert "hello file" in open(str(tmp_path / "run.log")).read()
