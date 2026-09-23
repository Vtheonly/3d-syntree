"""Offline tests for the production Hugging Face asset synchronization path."""
from __future__ import annotations

import json
import sys
import types

import pytest

import scripts.download_assets as assets


def test_hf_sync_rejects_missing_manifest(monkeypatch, tmp_path):
    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, **kwargs):
            return ["enamine_3d_subset.parquet"]

    fake_hub = types.SimpleNamespace(
        HfApi=FakeApi,
        hf_hub_download=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("download should not run")
        ),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(RuntimeError, match="missing required shard manifests"):
        assets.sync_hf_dataset("fake/repo", str(tmp_path))


def test_hf_sync_records_remote_metadata(monkeypatch, tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    catalog = remote / "enamine_3d_subset.parquet"
    catalog.write_bytes(b"catalog")

    manifests = {}
    for split in ("train", "val", "test"):
        payload = {
            "total_samples": 2,
            "shards": [{"name": "x.pt.gz", "sample_count": 2}],
        }
        path = remote / f"{split}.json"
        path.write_text(json.dumps(payload))
        manifests[f"data/{split}/manifest.json"] = path

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, **kwargs):
            return [
                "enamine_3d_subset.parquet",
                "data/train/manifest.json",
                "data/val/manifest.json",
                "data/test/manifest.json",
            ]

    def fake_download(**kwargs):
        filename = kwargs["filename"]
        if filename == "enamine_3d_subset.parquet":
            return str(catalog)
        return str(manifests[filename])

    fake_hub = types.SimpleNamespace(
        HfApi=FakeApi,
        hf_hub_download=fake_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = assets.sync_hf_dataset("fake/repo", str(tmp_path / "out"))
    assert result["lazy_shards"] is True
    assert result["synthetic_fallback_used"] is False
    assert result["splits"]["train"]["total_samples"] == 2
    assert (tmp_path / "out" / "enamine_3d_subset.parquet").exists()
