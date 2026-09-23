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


def test_hf_sync_stages_rl_target_pockets(monkeypatch, tmp_path):
    """Landmine 3: a dataset repo with targets/ gets its pocket PDBs staged
    into <output>/test_pockets for the RL docking-reward loop."""
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

    target_a = remote / "t1_pocket.pdb"
    target_b = remote / "t2_pocket.pdb"
    target_a.write_text("ATOM\nEND\n")
    target_b.write_text("ATOM\nEND\n")

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, **kwargs):
            return [
                "enamine_3d_subset.parquet",
                "data/train/manifest.json",
                "data/val/manifest.json",
                "data/test/manifest.json",
                "targets/t1_pocket.pdb",
                "targets/t2_pocket.pdb",
            ]

    def fake_download(**kwargs):
        filename = kwargs["filename"]
        if filename == "enamine_3d_subset.parquet":
            return str(catalog)
        if filename.startswith("targets/"):
            return str(remote / filename.split("/")[-1])
        return str(manifests[filename])

    fake_hub = types.SimpleNamespace(
        HfApi=FakeApi,
        hf_hub_download=fake_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = assets.sync_hf_dataset("fake/repo", str(tmp_path / "out"))
    pockets_dir = tmp_path / "out" / "test_pockets"
    assert pockets_dir.is_dir()
    assert sorted(p.name for p in pockets_dir.iterdir()) == [
        "t1_pocket.pdb", "t2_pocket.pdb",
    ]
    assert result["rl_pocket_count"] == 2
    assert result["rl_pocket_dir"] == str(pockets_dir)


def test_hf_sync_without_targets_still_works(monkeypatch, tmp_path):
    """Repositories without a targets/ folder keep working (the RL stage
    then requires an explicit --pocket-dir)."""
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
    assert result["rl_pocket_count"] == 0
    assert result["rl_pocket_dir"] is None
    assert not (tmp_path / "out" / "test_pockets").exists()
