"""Unit tests for core.manifest (atomic writes) and core.checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.checksums import (
    compute_digest,
    compute_sha256,
    pinned_hash_algo,
    verify_file,
    verify_pinned,
)
from core.manifest import FileRecord, ManifestManager, ShardRecord


class TestManifestManager:
    def test_roundtrip(self, tmp_path: Path):
        data = {"b": 2, "a": [1, 2, 3], "nested": {"x": "y"}}
        target = tmp_path / "manifest.json"
        ManifestManager.save_atomic(data, target)
        assert ManifestManager.load(target) == data

    def test_atomic_write_creates_parents(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "m.json"
        ManifestManager.save_atomic({"ok": True}, target)
        assert target.is_file()

    def test_no_temp_files_left_behind(self, tmp_path: Path):
        for i in range(5):
            ManifestManager.save_atomic({"i": i}, tmp_path / "m.json")
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []
        assert ManifestManager.load(tmp_path / "m.json") == {"i": 4}

    def test_load_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ManifestManager.load(tmp_path / "absent.json")

    def test_file_record_dataclass(self):
        rec = FileRecord(path="x", size_bytes=1, sha256="0" * 64)
        assert rec.metadata == {} and rec.source_url is None


class TestShardRecordCompat:
    """kaggle_loader requires name/sample_count/sha256 on every shard."""

    def test_defaults_and_loader_keys(self):
        rec = ShardRecord(
            shard_id=0,
            filename="train_shard_0000.pt.gz",
            sample_count=10,
            compressed_bytes=123,
            sha256="a" * 64,
            source_complex_ids=["c1"],
            split="train",
            verified=True,
            sample_start=0,
        )
        assert rec.name == "train_shard_0000.pt.gz"
        assert rec.sample_end == 10
        assert rec.sample_count > 0 and rec.sha256


class TestChecksums:
    def test_sha256_known_vector(self, tmp_path: Path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"abc")
        assert compute_sha256(p) == hashlib.sha256(b"abc").hexdigest()

    def test_md5_and_sha256(self, tmp_path: Path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"hello world")
        assert compute_digest(p, "md5") == hashlib.md5(b"hello world").hexdigest()
        assert compute_digest(p, "sha256") == hashlib.sha256(b"hello world").hexdigest()

    def test_verify_file_strict(self, tmp_path: Path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"payload")
        good = hashlib.sha256(b"payload").hexdigest()
        assert verify_file(p, good)
        assert verify_file(p, good.upper())  # case-insensitive
        assert not verify_file(p, "0" * 64)
        assert not verify_file(tmp_path / "missing", good)

    def test_verify_missing_file(self, tmp_path: Path):
        assert not verify_file(tmp_path / "nope", "0" * 64)

    def test_pinned_hash_algo_detection(self):
        # The legacy crossdocked pin is MD5-length despite its "sha256" name.
        assert pinned_hash_algo("59416d06c2c366f6e05b91fdff8584f7") == "md5"
        assert pinned_hash_algo("a" * 64) == "sha256"
        with pytest.raises(ValueError):
            pinned_hash_algo("abc123")

    def test_verify_pinned_md5(self, tmp_path: Path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"raw-archive")
        md5 = hashlib.md5(b"raw-archive").hexdigest()
        assert verify_pinned(p, md5)
        assert not verify_pinned(p, "0" * 32)
