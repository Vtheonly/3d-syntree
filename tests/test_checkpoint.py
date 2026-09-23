"""Regression tests for authoritative sequential checkpoint recovery."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from syntree.utils.checkpoint import CheckpointManager


class FakeHub:
    def __init__(self, files):
        self.files = set(files)
        self.uploads = []

    def list_repo_files(self, repo_id, repo_type="model"):
        return sorted(self.files)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type="model"):
        self.uploads.append(path_in_repo)
        self.files.add(path_in_repo)


def _manager(tmp_path, hub):
    cfg = {
        "huggingface": {
            "enabled": False,
            "repo_id": "owner/test-checkpoints",
            "push_every_n_epochs": 2,
        }
    }
    manager = CheckpointManager(cfg, ckpt_dir=str(tmp_path))
    manager.enabled = True
    manager.api = hub
    manager.token = "test-token"
    return manager


def _write_stale_local_state(tmp_path, epoch=39):
    model = nn.Linear(2, 2)
    torch.save(
        {
            "epoch": epoch,
            "step": epoch + 1,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 9.0},
        },
        tmp_path / f"checkpoint_epoch_{epoch}.pt",
    )
    (tmp_path / "checkpoint_best.pt").write_bytes(b"stale")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"latest_epoch": epoch}),
        encoding="utf-8",
    )
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "completed_epochs": list(range(epoch + 1)),
                "last_sequential_epoch": epoch,
                "history": [],
            }
        ),
        encoding="utf-8",
    )


def test_remote_empty_purges_stale_local_epoch_39(tmp_path):
    _write_stale_local_state(tmp_path, epoch=39)
    manager = _manager(tmp_path, FakeHub([]))

    start_epoch, step, best = manager.restore_latest(nn.Linear(2, 2))

    assert (start_epoch, step) == (0, 0)
    assert best == float("inf")
    assert not list(tmp_path.glob("checkpoint_*.pt"))
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "progress.json").exists()


def test_remote_progress_gap_is_rejected(tmp_path, monkeypatch):
    remote = Path(tmp_path) / "remote"
    remote.mkdir()

    model = nn.Linear(2, 2)
    torch.save(
        {
            "epoch": 2,
            "step": 8,
            "model_state_dict": model.state_dict(),
            "metrics": {"val_loss": 1.0},
        },
        remote / "checkpoint_epoch_2.pt",
    )
    (remote / "manifest.json").write_text(
        json.dumps({"latest_epoch": 2}),
        encoding="utf-8",
    )
    (remote / "progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "completed_epochs": [0, 2],
                "last_sequential_epoch": 2,
                "history": [],
            }
        ),
        encoding="utf-8",
    )

    hub = FakeHub(
        {
            "manifest.json",
            "progress.json",
            "checkpoints/checkpoint_epoch_2.pt",
        }
    )
    manager = _manager(tmp_path / "local", hub)

    def fake_download(repo_id, filename, token=None):
        if filename == "manifest.json":
            return str(remote / "manifest.json")
        if filename == "progress.json":
            return str(remote / "progress.json")
        return str(remote / "checkpoint_epoch_2.pt")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    start_epoch, step, best = manager.restore_latest(nn.Linear(2, 2))

    assert (start_epoch, step) == (0, 0)
    assert best == float("inf")
    assert not (manager.ckpt_dir / "manifest.json").exists()
    assert not (manager.ckpt_dir / "progress.json").exists()


def test_valid_remote_state_restores_and_materializes_checkpoint(tmp_path, monkeypatch):
    remote = Path(tmp_path) / "remote"
    local = Path(tmp_path) / "local"
    remote.mkdir()

    source = nn.Linear(2, 2)
    torch.save(
        {
            "epoch": 2,
            "step": 17,
            "model_state_dict": source.state_dict(),
            "metrics": {"val_loss": 0.75},
        },
        remote / "checkpoint_epoch_2.pt",
    )
    (remote / "manifest.json").write_text(
        json.dumps({"latest_epoch": 2, "latest_step": 17}),
        encoding="utf-8",
    )
    (remote / "progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "completed_epochs": [0, 1, 2],
                "last_sequential_epoch": 2,
                "history": [],
            }
        ),
        encoding="utf-8",
    )

    hub = FakeHub(
        {
            "manifest.json",
            "progress.json",
            "checkpoints/checkpoint_epoch_2.pt",
        }
    )
    manager = _manager(local, hub)

    # A stale local ghost must be removed even when the remote state is valid.
    local.mkdir(parents=True, exist_ok=True)
    (local / "checkpoint_epoch_39.pt").write_bytes(b"ghost")

    def fake_download(repo_id, filename, token=None):
        if filename == "manifest.json":
            return str(remote / "manifest.json")
        if filename == "progress.json":
            return str(remote / "progress.json")
        return str(remote / "checkpoint_epoch_2.pt")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    target = nn.Linear(2, 2)
    start_epoch, step, best = manager.restore_latest(target)

    assert (start_epoch, step) == (3, 17)
    assert best == 0.75
    assert (local / "checkpoint_epoch_2.pt").exists()
    assert not (local / "checkpoint_epoch_39.pt").exists()


def test_final_sync_flushes_earlier_async_uploads(tmp_path):
    hub = FakeHub([])
    manager = _manager(tmp_path, hub)
    model = nn.Linear(2, 2)

    manager.save_checkpoint(
        epoch=0,
        step=1,
        model=model,
        metrics={"val_loss": 2.0},
    )
    manager.save_checkpoint(
        epoch=1,
        step=2,
        model=model,
        metrics={"val_loss": 1.0},
        final=True,
    )

    manager.wait_for_uploads()

    assert hub.uploads[:3] == [
        "checkpoints/checkpoint_epoch_0.pt",
        "progress.json",
        "manifest.json",
    ]
    assert hub.uploads[-3:] == [
        "checkpoints/checkpoint_epoch_1.pt",
        "progress.json",
        "manifest.json",
    ]
    assert manager._upload_futures == []
