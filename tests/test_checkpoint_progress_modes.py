"""Progress-mode validation tests for CheckpointManager.

Epoch mode (trainer): completed epochs must be strictly contiguous.
A gap (e.g. [0, 2]) proves a lost record -> reject the whole state.
Episode mode (Stage 2 RL): only strict monotonicity is required because
RL checkpoints intentionally fire every ``checkpoint_every`` episodes.
Mode mismatches are rejected in both directions.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from syntree.utils.checkpoint import CheckpointManager

CFG = {"huggingface": {"enabled": False}}


def _progress(completed, last_seq=None, mode="epoch", history=None):
    return {
        "version": 1,
        "mode": mode,
        "completed_epochs": list(completed),
        "last_sequential_epoch": (
            completed[-1] if completed else -1 if last_seq is None else last_seq
        ),
        "history": history if history is not None else [],
    }


class TestEpochModeValidation:
    def test_contiguous_progress_valid(self):
        valid, last = CheckpointManager._validate_progress(_progress([0, 1, 2]))
        assert valid and last == 2

    def test_single_epoch_valid(self):
        valid, last = CheckpointManager._validate_progress(_progress([0]))
        assert valid and last == 0

    def test_empty_progress_valid(self):
        valid, last = CheckpointManager._validate_progress(_progress([]))
        assert valid and last == -1

    def test_gap_is_rejected(self):
        """The production bug: [0, 2] means epoch 1's record was lost."""
        valid, last = CheckpointManager._validate_progress(_progress([0, 2]))
        assert not valid

    def test_mid_gap_is_rejected(self):
        valid, _ = CheckpointManager._validate_progress(_progress([0, 1, 2, 4]))
        assert not valid

    def test_duplicate_is_rejected(self):
        valid, _ = CheckpointManager._validate_progress(_progress([0, 1, 1]))
        assert not valid

    def test_decreasing_is_rejected(self):
        valid, _ = CheckpointManager._validate_progress(_progress([2, 1]))
        assert not valid

    def test_last_seq_mismatch_is_rejected(self):
        p = _progress([0, 1, 2])
        p["last_sequential_epoch"] = 5
        valid, _ = CheckpointManager._validate_progress(p)
        assert not valid

    def test_negative_entries_rejected(self):
        valid, _ = CheckpointManager._validate_progress(_progress([-1, 0]))
        assert not valid

    def test_bool_entries_rejected(self):
        p = _progress([0, 1])
        p["completed_epochs"] = [False, 1]
        valid, _ = CheckpointManager._validate_progress(p)
        assert not valid

    def test_non_list_completed_rejected(self):
        p = _progress([0])
        p["completed_epochs"] = "0,1,2"
        valid, _ = CheckpointManager._validate_progress(p)
        assert not valid

    def test_non_dict_rejected(self):
        valid, _ = CheckpointManager._validate_progress(None)
        assert not valid

    def test_legacy_progress_without_mode_defaults_to_epoch(self):
        p = _progress([0, 1])
        p.pop("mode")
        valid, last = CheckpointManager._validate_progress(p)
        assert valid and last == 1

    def test_unknown_mode_rejected(self):
        p = _progress([0, 1], mode="bogus")
        valid, _ = CheckpointManager._validate_progress(p)
        assert not valid


class TestEpisodeModeValidation:
    def test_gapped_episodes_valid(self):
        """RL saves every N episodes: [15, 31, 47] is legitimate."""
        valid, last = CheckpointManager._validate_progress(
            _progress([15, 31, 47], mode="episode"), expected_mode="episode"
        )
        assert valid and last == 47

    def test_contiguous_episodes_valid(self):
        valid, last = CheckpointManager._validate_progress(
            _progress([0, 1, 2], mode="episode"), expected_mode="episode"
        )
        assert valid and last == 2

    def test_mode_mismatch_epoch_manager_reads_episode_progress(self):
        valid, _ = CheckpointManager._validate_progress(
            _progress([15, 31], mode="episode"), expected_mode="epoch"
        )
        assert not valid

    def test_mode_mismatch_episode_manager_reads_epoch_progress(self):
        valid, _ = CheckpointManager._validate_progress(
            _progress([0, 1], mode="epoch"), expected_mode="episode"
        )
        assert not valid

    def test_episode_decreasing_rejected(self):
        valid, _ = CheckpointManager._validate_progress(
            _progress([31, 15], mode="episode"), expected_mode="episode"
        )
        assert not valid


class TestManagerConstruction:
    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="progress_mode"):
            CheckpointManager(CFG, ckpt_dir=str(tmp_path / "a"), progress_mode="bogus")

    def test_default_mode_is_epoch(self, tmp_path):
        mgr = CheckpointManager(CFG, ckpt_dir=str(tmp_path / "b"))
        assert mgr.progress_mode == "episode" if False else mgr.progress_mode == "epoch"

    def test_mode_persisted_in_progress_file(self, tmp_path):
        mgr = CheckpointManager(CFG, ckpt_dir=str(tmp_path / "c"))
        model = nn.Linear(2, 2)
        mgr.save_checkpoint(epoch=0, step=1, model=model, metrics={})
        progress = json.loads((tmp_path / "c" / "progress.json").read_text())
        assert progress["mode"] == "epoch"

    def test_episode_mode_persisted(self, tmp_path):
        mgr = CheckpointManager(
            CFG, ckpt_dir=str(tmp_path / "d"), progress_mode="episode"
        )
        model = nn.Linear(2, 2)
        mgr.save_checkpoint(epoch=15, step=16, model=model, metrics={})
        progress = json.loads((tmp_path / "d" / "progress.json").read_text())
        assert progress["mode"] == "episode"
        assert progress["completed_epochs"] == [15]


class TestEpisodeRecordingSequence:
    def test_rl_style_gapped_saves_record_and_restore(self, tmp_path):
        """Simulate the exact main.py RL flow: checkpoints at episodes
        15, 31, 47 must record without exceptions and restore to 48."""
        mgr = CheckpointManager(
            CFG, ckpt_dir=str(tmp_path / "rl"), progress_mode="episode"
        )
        model = nn.Linear(2, 2)
        for episode in (15, 31, 47):
            mgr.save_checkpoint(
                epoch=episode, step=episode + 1, model=model, metrics={"reward": 0.5}
            )

        progress = json.loads((tmp_path / "rl" / "progress.json").read_text())
        assert progress["completed_epochs"] == [15, 31, 47]
        assert progress["mode"] == "episode"

        mgr2 = CheckpointManager(
            CFG, ckpt_dir=str(tmp_path / "rl"), progress_mode="episode"
        )
        start_episode, step, _ = mgr2.restore_latest(nn.Linear(2, 2))
        assert start_episode == 48
        assert step == 48

    def test_epoch_manager_rejects_gapped_rl_progress(self, tmp_path):
        """An epoch-mode manager must refuse gapped progress: purge + fresh."""
        mgr = CheckpointManager(
            CFG, ckpt_dir=str(tmp_path / "mixed"), progress_mode="episode"
        )
        model = nn.Linear(2, 2)
        for episode in (15, 31):
            mgr.save_checkpoint(epoch=episode, step=episode + 1, model=model, metrics={})

        epoch_mgr = CheckpointManager(CFG, ckpt_dir=str(tmp_path / "mixed"))
        start, step, best = epoch_mgr.restore_latest(nn.Linear(2, 2))
        assert (start, step) == (0, 0)
        assert best == float("inf")
        # The unusable state must be purged, not left behind.
        assert not (tmp_path / "mixed" / "progress.json").exists()


class TestEpochRecordingRules:
    def test_epoch_manager_records_contiguous_run(self, tmp_path):
        mgr = CheckpointManager(CFG, ckpt_dir=str(tmp_path / "e"))
        model = nn.Linear(2, 2)
        for epoch in (0, 1, 2, 3):
            mgr.save_checkpoint(epoch=epoch, step=epoch + 1, model=model, metrics={})
        progress = json.loads((tmp_path / "e" / "progress.json").read_text())
        assert progress["completed_epochs"] == [0, 1, 2, 3]

    def test_out_of_order_epoch_recording_raises(self, tmp_path):
        """Re-saving the *current* epoch is legal (trainer._finalize does it),
        but re-recording an OLDER completed epoch must raise."""
        mgr = CheckpointManager(CFG, ckpt_dir=str(tmp_path / "f"))
        model = nn.Linear(2, 2)
        mgr.save_checkpoint(epoch=0, step=1, model=model, metrics={})
        mgr.save_checkpoint(epoch=1, step=2, model=model, metrics={})
        # Idempotent re-save of the latest epoch: allowed.
        mgr.save_checkpoint(epoch=1, step=2, model=model, metrics={})
        # Older epoch: out of order -> raise.
        with pytest.raises(RuntimeError, match="out of order"):
            mgr.save_checkpoint(epoch=0, step=1, model=model, metrics={})

    def test_epoch_gap_recording_raises(self, tmp_path):
        """Manually injecting a gap must poison the file for the next save."""
        mgr = CheckpointManager(CFG, ckpt_dir=str(tmp_path / "g"))
        model = nn.Linear(2, 2)
        mgr.save_checkpoint(epoch=0, step=1, model=model, metrics={})
        # Corrupt: inject a gap (simulates a lost partial write).
        path = tmp_path / "g" / "progress.json"
        progress = json.loads(path.read_text())
        progress["completed_epochs"] = [0, 2]
        progress["last_sequential_epoch"] = 2
        path.write_text(json.dumps(progress))
        with pytest.raises(RuntimeError, match="refusing to record"):
            mgr.save_checkpoint(epoch=3, step=4, model=model, metrics={})
