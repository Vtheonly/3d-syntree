"""Data-integrity tests for the sharded Hugging Face dataset pipeline.

Verifies:
* write_shards manifests: totals, digests, index contiguity, byte bounds,
  deterministic output (gzip mtime pinned).
* ShardedHuggingFaceDataset: index mapping, round-trip equality, sha256
  corruption rejection, sample-count mismatch rejection, bounded cache
  eviction, lazy shard loading.
* ShardAwareShuffleSampler: exact coverage, per-epoch determinism, epoch
  variation.
* sync_hf_dataset: empty-split rejection, stub-dataset gate.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.shard_and_upload import serialize_samples, write_shards  # noqa: E402
import syntree.data.hf_loader as hf_loader  # noqa: E402
from syntree.data.hf_loader import (  # noqa: E402
    ShardAwareShuffleSampler,
    ShardedHuggingFaceDataset,
)


def _make_samples(n: int, seed: int = 0) -> list:
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(n):
        pocket = torch.randn(8, 3, generator=g)
        samples.append(
            Data(
                pocket_pos=pocket,
                pocket_z=torch.randint(6, 9, (8,), generator=g),
                ligand_pos=torch.randn(4, 3, generator=g),
                ligand_z=torch.randint(6, 8, (4,), generator=g),
                target_synthon=torch.tensor([i % 10], dtype=torch.long),
                target_stop=torch.tensor([i % 3 == 0]),
                num_nodes=8,
                trajectory_hash=torch.tensor([i], dtype=torch.long),
            )
        )
    return samples


class TestSerializeDeterminism:
    def test_mtime_pinned_bit_reproducible(self):
        """gzip header timestamps must not leak into shard digests."""
        a = serialize_samples(_make_samples(5))
        b = serialize_samples(_make_samples(5))
        assert a == b
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()

    def test_different_data_differs(self):
        assert serialize_samples(_make_samples(5)) != serialize_samples(
            _make_samples(5, seed=1)
        )

    def test_output_is_gzip(self):
        payload = serialize_samples(_make_samples(3))
        assert payload[:2] == b"\x1f\x8b"
        with gzip.GzipFile(fileobj=__import__("io").BytesIO(payload)) as f:
            assert f.read(2)  # decompressible


class TestWriteShards:
    def test_manifest_totals(self, tmp_path):
        manifest = write_shards(_make_samples(37), "train", str(tmp_path))
        assert manifest["total_samples"] == 37
        assert manifest["split"] == "train"
        declared = sum(s["sample_count"] for s in manifest["shards"])
        assert declared == 37

    def test_digests_match_bytes(self, tmp_path):
        manifest = write_shards(_make_samples(20), "val", str(tmp_path))
        for shard in manifest["shards"]:
            payload = (tmp_path / "val" / shard["name"]).read_bytes()
            assert shard["compressed_bytes"] == len(payload)
            assert shard["sha256"] == hashlib.sha256(payload).hexdigest()

    def test_index_contiguity(self, tmp_path):
        manifest = write_shards(_make_samples(50), "train", str(tmp_path))
        shards = sorted(manifest["shards"], key=lambda s: s["sample_start"])
        start = 0
        for shard in shards:
            assert shard["sample_start"] == start
            assert shard["sample_end"] == start + shard["sample_count"]
            start = shard["sample_end"]
        assert start == 50

    def test_byte_bounds_enforced(self, tmp_path):
        max_bytes = 4 * 1024
        manifest = write_shards(
            _make_samples(40), "train", str(tmp_path), max_shard_bytes=max_bytes,
            coarse_batch_size=8,
        )
        for shard in manifest["shards"]:
            if not shard.get("oversize_single_sample"):
                assert shard["compressed_bytes"] <= max_bytes

    def test_oversize_single_sample_flagged(self, tmp_path):
        manifest = write_shards(
            _make_samples(1), "train", str(tmp_path), max_shard_bytes=1,
        )
        assert len(manifest["shards"]) == 1
        assert manifest["shards"][0]["oversize_single_sample"] is True

    def test_empty_split_manifest(self, tmp_path):
        """Zero-sample splits still write a manifest (fail-closed downstream)."""
        manifest = write_shards([], "test", str(tmp_path))
        assert manifest["total_samples"] == 0
        assert manifest["shards"] == []
        assert (tmp_path / "test" / "manifest.json").exists()

    def test_bit_reproducible_output(self, tmp_path):
        m1 = write_shards(_make_samples(25), "train", str(tmp_path / "a"))
        m2 = write_shards(_make_samples(25), "train", str(tmp_path / "b"))
        d1 = [(s["name"], s["sha256"]) for s in m1["shards"]]
        d2 = [(s["name"], s["sha256"]) for s in m2["shards"]]
        assert d1 == d2

    def test_sample_round_trip(self, tmp_path):
        originals = _make_samples(9)
        manifest = write_shards(originals, "train", str(tmp_path))
        for shard in manifest["shards"]:
            with gzip.open(tmp_path / "train" / shard["name"], "rb") as f:
                loaded = torch.load(f, weights_only=False)
            assert len(loaded) == shard["sample_count"]
            for a, b in zip(loaded, loaded):
                pass
        # Full order-preserving round trip.
        loaded_all = []
        for shard in sorted(manifest["shards"], key=lambda s: s["sample_start"]):
            with gzip.open(tmp_path / "train" / shard["name"], "rb") as f:
                loaded_all.extend(torch.load(f, weights_only=False))
        assert len(loaded_all) == len(originals)
        for orig, back in zip(originals, loaded_all):
            assert torch.equal(orig.pocket_pos, back.pocket_pos)
            assert torch.equal(orig.target_synthon, back.target_synthon)
            assert bool(back.target_stop) == bool(orig.target_stop)


# =====================================================================
# ShardedHuggingFaceDataset against a fake Hub
# =====================================================================
class FakeHub:
    """Serves a local directory as if it were an HF dataset repo."""

    def __init__(self, root: Path):
        self.root = root

    def download(self, **kwargs):
        filename = kwargs["filename"]
        local = self.root / filename
        if not local.exists():
            raise FileNotFoundError(f"not in fake repo: {filename}")
        return str(local)


@pytest.fixture()
def fake_repo(tmp_path):
    root = tmp_path / "remote"
    samples = _make_samples(24)
    data_root = root / "data"
    # Small shard budget so the fixture yields multiple shards per split
    # (needed by the cache-eviction and shard-aware sampler tests).
    small = 6 * 1024
    write_shards(samples, "train", str(data_root), max_shard_bytes=small)
    write_shards(samples[:6], "val", str(data_root), max_shard_bytes=small)
    write_shards(samples[6:10], "test", str(data_root), max_shard_bytes=small)
    return FakeHub(root)


def _patch_hub(monkeypatch, hub: FakeHub):
    monkeypatch.setattr(hf_loader, "hf_hub_download", hub.download)


class TestShardedHuggingFaceDataset:
    def test_length_matches_manifest(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        assert len(ds) == 24

    def test_sample_round_trip_equality(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        for idx in (0, 7, 23):
            sample = ds[idx]
            assert torch.isfinite(sample.pocket_pos).all()
            assert sample.pocket_pos.size(0) == 8
            assert int(sample.target_synthon.item()) == idx % 10
            assert hasattr(sample, "num_nodes")

    def test_index_error_out_of_range(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        with pytest.raises(IndexError):
            ds[len(ds)]
        with pytest.raises(IndexError):
            ds[-len(ds) - 1]

    def test_negative_indexing(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        assert ds[-1].pocket_pos.shape == ds[len(ds) - 1].pocket_pos.shape

    def test_sha256_corruption_rejected(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        cache = tmp_path / "cache"
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(cache),
        )
        ds[0]  # force shard download
        shard_file = cache / "train" / ds.shards[0]["name"]
        payload = bytearray(shard_file.read_bytes())
        payload[-1] ^= 0xFF  # corrupt one byte
        shard_file.write_bytes(bytes(payload))
        # New dataset instance so the in-memory cache does not mask it.
        ds2 = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(cache),
        )
        with pytest.raises(IOError, match="SHA-256"):
            ds2[0]

    def test_sample_count_mismatch_rejected(self, monkeypatch, tmp_path):
        root = tmp_path / "remote"
        samples = _make_samples(8)
        manifest = write_shards(samples, "train", str(root / "data"))
        # Lie in the manifest: claim more samples than the shard holds.
        # Internally consistent totals pass the constructor check, so the
        # lie must be caught when the shard bytes are actually loaded.
        for shard in manifest["shards"]:
            shard["sample_count"] = 99
        manifest["total_samples"] = 99 * len(manifest["shards"])
        (root / "data" / "train" / "manifest.json").write_text(json.dumps(manifest))
        hub = FakeHub(root)
        _patch_hub(monkeypatch, hub)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        assert len(ds) == 99 * len(manifest["shards"])
        with pytest.raises(ValueError, match="sample count mismatch"):
            ds[0]

    def test_total_samples_mismatch_rejected(self, monkeypatch, tmp_path):
        root = tmp_path / "remote"
        samples = _make_samples(8)
        manifest = write_shards(samples, "train", str(root / "data"))
        manifest["total_samples"] = 123  # inconsistent with shards
        (root / "data" / "train" / "manifest.json").write_text(json.dumps(manifest))
        _patch_hub(monkeypatch, FakeHub(root))
        with pytest.raises(ValueError, match="does not match shard counts"):
            ShardedHuggingFaceDataset(
                "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
            )

    def test_no_shards_rejected(self, monkeypatch, tmp_path):
        root = tmp_path / "remote"
        (root / "data" / "train").mkdir(parents=True)
        (root / "data" / "train" / "manifest.json").write_text(json.dumps(
            {"total_samples": 0, "shards": []}
        ))
        _patch_hub(monkeypatch, FakeHub(root))
        with pytest.raises(RuntimeError, match="No shards declared"):
            ShardedHuggingFaceDataset(
                "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
            )

    def test_zero_sample_count_rejected(self, monkeypatch, tmp_path):
        root = tmp_path / "remote"
        samples = _make_samples(4)
        manifest = write_shards(samples, "train", str(root / "data"))
        manifest["shards"][0]["sample_count"] = 0
        (root / "data" / "train" / "manifest.json").write_text(json.dumps(manifest))
        _patch_hub(monkeypatch, FakeHub(root))
        with pytest.raises(ValueError, match="positive"):
            ShardedHuggingFaceDataset(
                "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
            )

    def test_invalid_split_rejected(self):
        with pytest.raises(ValueError, match="train/val/test"):
            ShardedHuggingFaceDataset("fake/repo", split="other")

    def test_bounded_cache_eviction(self, monkeypatch, fake_repo, tmp_path):
        """Only max_cached_shards shards stay in memory."""
        _patch_hub(monkeypatch, fake_repo)
        cache = tmp_path / "cache"
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(cache),
            max_cached_shards=1,
        )
        assert len(ds.shards) >= 2, "fixture should produce multiple shards"
        # Visit every sample (crosses shard boundaries).
        for i in range(len(ds)):
            ds[i]
        assert len(ds._cache) <= 1
        assert len(ds._cache_order) <= 1

    def test_lazy_loading_no_early_download(self, monkeypatch, fake_repo, tmp_path):
        """Constructing the dataset must not download any shard."""
        calls = []

        def tracking_download(**kwargs):
            calls.append(kwargs["filename"])
            return fake_repo.download(**kwargs)

        monkeypatch.setattr(hf_loader, "hf_hub_download", tracking_download)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        assert calls == ["data/train/manifest.json"]


class TestShardAwareShuffleSampler:
    def test_covers_every_index_exactly_once(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        sampler = ShardAwareShuffleSampler(ds, seed=42)
        indices = list(iter(sampler))
        assert sorted(indices) == list(range(len(ds)))
        assert len(indices) == len(set(indices))

    def test_per_epoch_determinism(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        sampler = ShardAwareShuffleSampler(ds, seed=7)
        sampler.set_epoch(0)
        e0a = list(iter(sampler))
        e0b = list(iter(sampler))
        assert e0a == e0b

    def test_epoch_changes_order(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        sampler = ShardAwareShuffleSampler(ds, seed=7)
        sampler.set_epoch(0)
        e0 = list(iter(sampler))
        sampler.set_epoch(1)
        e1 = list(iter(sampler))
        assert e0 != e1
        assert sorted(e0) == sorted(e1)

    def test_length_matches_dataset(self, monkeypatch, fake_repo, tmp_path):
        _patch_hub(monkeypatch, fake_repo)
        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train", cache_dir=str(tmp_path / "cache"),
        )
        assert len(ShardAwareShuffleSampler(ds, seed=1)) == len(ds)


# =====================================================================
# Asset-sync integrity gate
# =====================================================================
def _make_repo_files(tmp_path, train_n, val_n, test_n):
    import scripts.download_assets as assets

    remote = tmp_path / "repo"
    remote.mkdir()
    (remote / "enamine_3d_subset.parquet").write_bytes(b"catalog-bytes")
    for split, n in (("train", train_n), ("val", val_n), ("test", test_n)):
        (remote / "data" / split).mkdir(parents=True)
        (remote / f"data/{split}/manifest.json").write_text(
            json.dumps({"total_samples": n, "shards": [{"name": "s.pt.gz", "sample_count": n}]})
        )
    return remote


def _patch_assets_hub(monkeypatch, remote: Path):
    def fake_download(**kwargs):
        local = remote / kwargs["filename"]
        assert local.exists(), kwargs["filename"]
        return str(local)

    def fake_list_repo_files(**kwargs):
        return sorted(
            str(p.relative_to(remote)) for p in remote.rglob("*") if p.is_file()
        )

    fake = types.SimpleNamespace(
        HfApi=lambda token=None: types.SimpleNamespace(
            list_repo_files=fake_list_repo_files
        ),
        hf_hub_download=fake_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    import scripts.download_assets as assets

    return assets


class TestAssetSyncIntegrity:
    def test_empty_test_split_rejected(self, monkeypatch, tmp_path):
        remote = _make_repo_files(tmp_path, train_n=100, val_n=10, test_n=0)
        # An empty split declares zero samples and no shards.
        (remote / "data/test/manifest.json").write_text(
            json.dumps({"total_samples": 0, "shards": []})
        )
        assets = _patch_assets_hub(monkeypatch, remote)
        with pytest.raises(RuntimeError, match="Invalid test manifest"):
            assets.sync_hf_dataset("fake/repo", str(tmp_path / "out"))

    def test_declared_mismatch_rejected(self, monkeypatch, tmp_path):
        remote = _make_repo_files(tmp_path, 100, 10, 10)
        manifest = json.loads((remote / "data/train/manifest.json").read_text())
        manifest["total_samples"] = 500  # lies
        (remote / "data/train/manifest.json").write_text(json.dumps(manifest))
        assets = _patch_assets_hub(monkeypatch, remote)
        with pytest.raises(RuntimeError, match="Invalid train manifest"):
            assets.sync_hf_dataset("fake/repo", str(tmp_path / "out"))

    def test_stub_gate_rejects_small_train(self, monkeypatch, tmp_path):
        remote = _make_repo_files(tmp_path, train_n=2, val_n=2, test_n=2)
        assets = _patch_assets_hub(monkeypatch, remote)
        with pytest.raises(RuntimeError, match="stub dataset"):
            assets.sync_hf_dataset(
                "fake/repo", str(tmp_path / "out"), min_train_samples=50
            )

    def test_stub_gate_passes_for_real_dataset(self, monkeypatch, tmp_path):
        remote = _make_repo_files(tmp_path, train_n=15420, val_n=1920, test_n=1920)
        assets = _patch_assets_hub(monkeypatch, remote)
        result = assets.sync_hf_dataset(
            "fake/repo", str(tmp_path / "out"), min_train_samples=50
        )
        assert result["splits"]["train"]["total_samples"] == 15420
        assert result["synthetic_fallback_used"] is False

    def test_verification_printed(self, monkeypatch, tmp_path, capsys):
        remote = _make_repo_files(tmp_path, 100, 10, 10)
        assets = _patch_assets_hub(monkeypatch, remote)
        assets.sync_hf_dataset("fake/repo", str(tmp_path / "out"))
        out = capsys.readouterr().out
        assert "train=100 samples" in out
        assert "val=10 samples" in out
