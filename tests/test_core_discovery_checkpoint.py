"""Unit tests for core.discovery and core.checkpointing (resume + tamper)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.discovery import DatasetDiscovery
from core.manifest import ManifestManager


RAW_MANIFEST = {
    "dataset_type": "3d_syntree_raw_unprocessed",
    "version": 1,
    "created_at": 0.0,
    "files": {
        "crossdocked": {"filename": "crossdocked_pocket10.tar.gz", "size_bytes": 1, "hash": "x", "hash_algo": "md5", "source_url": "u"},
        "enamine_catalog": {"filename": "enamine_3d_subset.parquet", "size_bytes": 1, "hash": "y", "hash_algo": "sha256", "source_url": "u"},
    },
}


class TestRawDiscovery:
    def test_manifest_discovery(self, tmp_path: Path):
        ds = tmp_path / "mounted" / "raw"
        ds.mkdir(parents=True)
        (ds / "raw_manifest.json").write_text(json.dumps(RAW_MANIFEST))
        found = DatasetDiscovery.discover_raw_dataset(tmp_path)
        assert found == ds.resolve()

    def test_manifest_missing_sources_rejected_then_signature_fallback(self, tmp_path: Path):
        # Manifest declares no sources -> not trusted; but exact signature
        # files present -> fallback path certifies it.
        ds = tmp_path / "raw"
        ds.mkdir(parents=True)
        bad = dict(RAW_MANIFEST)
        bad["files"] = {}
        (ds / "raw_manifest.json").write_text(json.dumps(bad))
        (ds / "crossdocked_pocket10.tar.gz").write_bytes(b"x")
        (ds / "enamine_3d_subset.parquet").write_bytes(b"y")
        assert DatasetDiscovery.discover_raw_dataset(tmp_path) == ds.resolve()

    def test_wrong_dataset_type_rejected(self, tmp_path: Path):
        ds = tmp_path / "raw"
        ds.mkdir(parents=True)
        bad = dict(RAW_MANIFEST)
        bad["dataset_type"] = "something_else"
        (ds / "raw_manifest.json").write_text(json.dumps(bad))
        with pytest.raises(FileNotFoundError):
            DatasetDiscovery.discover_raw_dataset(tmp_path)

    def test_no_candidate_raises(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            DatasetDiscovery.discover_raw_dataset(tmp_path)

    def test_missing_root_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            DatasetDiscovery.discover_raw_dataset(tmp_path / "absent")

    def test_crafted_discovery_requires_train_and_val(self, tmp_path: Path):
        ds = tmp_path / "crafted"
        (ds / "train").mkdir(parents=True)
        (ds / "train" / "manifest.json").write_text(json.dumps({"shards": [{}]}))
        with pytest.raises(FileNotFoundError):
            DatasetDiscovery.discover_crafted_dataset(tmp_path)
        (ds / "val").mkdir()
        (ds / "val" / "manifest.json").write_text(json.dumps({"shards": [{}]}))
        assert DatasetDiscovery.discover_crafted_dataset(tmp_path) == ds.resolve()


class TestCheckpointing:
    @pytest.fixture()
    def mgr_cls(self):
        pytest.importorskip("torch")
        from core.checkpointing import ShardCheckpointManager

        return ShardCheckpointManager

    def _samples(self, n: int, tag: int = 0):
        return [{"i": i, "tag": tag, "payload": [tag] * 10} for i in range(n)]

    def test_write_manifest_loader_compat(self, tmp_path: Path, mgr_cls):
        mgr = mgr_cls(tmp_path, "train", max_shard_bytes=10 * 1024 * 1024)
        mgr.write_shard(self._samples(5), ["c1", "c2"])
        manifest = ManifestManager.load(tmp_path / "train" / "manifest.json")
        # Contract required by syntree.data.kaggle_loader:
        assert manifest["format"] == "torch-pyg-list+gzip"
        assert manifest["total_samples"] == 5
        shard = manifest["shards"][0]
        assert shard["name"] == shard["filename"]
        assert shard["sample_count"] == 5 and shard["sample_count"] > 0
        assert len(shard["sha256"]) == 64
        assert (tmp_path / "train" / shard["name"]).is_file()
        # Declared total must equal sum of counts (loader fails closed otherwise).
        assert manifest["total_samples"] == sum(s["sample_count"] for s in manifest["shards"])

    def test_resume_skips_completed_complexes(self, tmp_path: Path, mgr_cls):
        mgr = mgr_cls(tmp_path, "train")
        mgr.write_shard(self._samples(4), ["c1", "c2"])
        mgr.write_shard(self._samples(4, tag=1), ["c3"])

        fresh = mgr_cls(tmp_path, "train")
        assert fresh.is_complex_completed("c1")
        assert fresh.is_complex_completed("c2")
        assert fresh.is_complex_completed("c3")
        assert not fresh.is_complex_completed("c4")
        assert fresh.total_samples == 8
        assert len(fresh.shards) == 2

    def test_tampered_shard_not_trusted(self, tmp_path: Path, mgr_cls):
        mgr = mgr_cls(tmp_path, "train")
        mgr.write_shard(self._samples(4), ["c1"])
        mgr.write_shard(self._samples(4, tag=1), ["c2"])

        # Corrupt the FIRST shard: resume must stop at the gap and trust
        # nothing from it onward (fail-closed).
        first = tmp_path / "train" / "train_shard_0000.pt.gz"
        first.write_bytes(b"tampered")
        fresh = mgr_cls(tmp_path, "train")
        assert not fresh.is_complex_completed("c1")
        assert not fresh.is_complex_completed("c2")
        assert fresh.total_samples == 0

    def test_corrupt_manifest_resets_state(self, tmp_path: Path, mgr_cls):
        mgr = mgr_cls(tmp_path, "train")
        mgr.write_shard(self._samples(3), ["c1"])
        (tmp_path / "train" / "manifest.json").write_text("{not json")
        fresh = mgr_cls(tmp_path, "train")
        assert fresh.total_samples == 0
        assert not fresh.is_complex_completed("c1")

    def test_write_rejects_empty(self, tmp_path: Path, mgr_cls):
        mgr = mgr_cls(tmp_path, "train")
        with pytest.raises(ValueError):
            mgr.write_shard([], ["c1"])

    def test_sample_ranges_are_contiguous(self, tmp_path: Path, mgr_cls):
        mgr = mgr_cls(tmp_path, "val")
        mgr.write_shard(self._samples(7), ["a"])
        mgr.write_shard(self._samples(3, tag=1), ["b"])
        manifest = ManifestManager.load(tmp_path / "val" / "manifest.json")
        ranges = [(s["sample_start"], s["sample_end"]) for s in manifest["shards"]]
        assert ranges == [(0, 7), (7, 10)]
        assert manifest["total_samples"] == 10
