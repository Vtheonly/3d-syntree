"""Tests for the unified sharded dataset layer (#19 DATA-04).

Issue-pinned acceptance criteria:

1. Shard generation produces valid bounded ``.pt.gz`` files matching
   SHA-256 digests in ``manifest.json`` (byte-bounded packing, verified
   against the actual emitted file sizes; deterministic bytes).
2. ``ShardedDataset`` loads samples interchangeably from local paths or
   Hugging Face streaming with zero data corruption (the HF path is
   exercised by wiring ``hf_hub_download`` to a simulated remote —
   identical bytes, digest-verified).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from syntree.data.sharded import (  # noqa: E402
    DEFAULT_SHARD_BOUND_BYTES,
    ShardAwareShuffleSampler,
    ShardIntegrityError,
    ShardManifestError,
    ShardManifest,
    ShardedDataset,
    build_shards,
    estimate_record_bytes,
    load_shard_manifest,
)
from syntree.data.shards import read_shard, write_shard  # noqa: E402


def _records(n: int = 64, seed: int = 2026) -> list:
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "complex_id": f"cpx_{i:04d}",
            "split": "train",
            "coords": torch.randn(8 + (i % 5), 3, generator=generator),
            "energy": -12.0 + 0.25 * i,
            "smiles": "CC(=O)Nc1ccc(N)cc1",
        }
        for i in range(n)
    ]


def _build(tmp_path, *, n=64, bound=2048, prefix="traj-train", seed=2026):
    out_dir = str(tmp_path / "shards")
    records = _records(n, seed=seed)
    manifest = build_shards(
        records, out_dir, shard_bound_bytes=bound, prefix=prefix,
        source={"kind": "test", "split": "train"},
    )
    return out_dir, records, manifest


class TestBuildShards:
    def test_default_bound_is_500mb(self):
        assert DEFAULT_SHARD_BOUND_BYTES == 500 * 1024 * 1024

    def test_every_shard_respects_the_byte_bound(self, tmp_path):
        out_dir, records, manifest = _build(tmp_path, bound=2048)
        assert len(manifest.shards) >= 4  # 64 records forced multi-shard
        for entry in manifest.shards:
            path = os.path.join(out_dir, entry.name)
            assert os.path.getsize(path) == entry.bytes
            assert entry.bytes <= 2048, "emitted shard exceeds the bound"

    def test_manifest_digests_match_files(self, tmp_path):
        out_dir, records, manifest = _build(tmp_path, bound=2048)
        from syntree.data.shards import shard_sha256

        for entry in manifest.shards:
            assert shard_sha256(os.path.join(out_dir, entry.name)) == entry.sha256
        # manifest.json written with all integrity fields
        doc = json.load(open(os.path.join(out_dir, "manifest.json")))
        assert doc["num_records"] == len(records)
        assert {s["name"] for s in doc["shards"]} == {e.name for e in manifest.shards}
        for shard in doc["shards"]:
            assert set(shard) >= {"name", "records", "sha256", "bytes"}

    def test_record_counts_partition_the_stream(self, tmp_path):
        out_dir, records, manifest = _build(tmp_path, bound=2048)
        assert manifest.num_records == len(records)
        # order is preserved across shards: concatenated shards == input order
        restored = []
        for entry in manifest.shards:
            restored += read_shard(os.path.join(out_dir, entry.name))
        assert [r["complex_id"] for r in restored] == [r["complex_id"] for r in records]

    def test_deterministic_rebuild_is_byte_identical(self, tmp_path):
        out_dir, records, manifest = _build(tmp_path / "a", bound=2048)
        out_dir2, _, manifest2 = _build(tmp_path / "b", bound=2048)
        assert [e.sha256 for e in manifest.shards] == [e.sha256 for e in manifest2.shards]
        assert [e.records for e in manifest.shards] == [e.records for e in manifest2.shards]

    def test_single_oversized_record_fails_closed(self, tmp_path):
        huge = {"blob": torch.zeros(4096)}  # serializes > 1024 B bound
        with pytest.raises(ShardIntegrityError, match="no valid packing"):
            build_shards([huge], str(tmp_path / "x"), shard_bound_bytes=1024)
        # nothing was written
        assert not any((tmp_path / "x").glob("*.pt.gz"))

    def test_compressed_bound_violation_fails_closed(self, tmp_path, monkeypatch):
        # pathological: a payload that barely fits uncompressed but whose
        # compressed form would exceed a bound already satisfied by the
        # estimate — simulate by patching the size probe to under-report
        import syntree.data.sharded as sharded

        real = sharded.estimate_record_bytes
        records = _records(8)
        monkeypatch.setattr(
            sharded, "estimate_record_bytes", lambda r: real(r) // 8
        )
        # pick a bound above the (under-reported) estimate but below the
        # real compressed size of the whole batch
        total_real = sum(real(r) for r in records)
        bound = max(real(r) for r in records) // 2
        assert bound < total_real
        with pytest.raises(ShardIntegrityError, match="byte bound"):
            build_shards(records, str(tmp_path / "y"), shard_bound_bytes=bound)


class TestShardedDatasetLocal:
    def test_local_load_round_trip_zero_corruption(self, tmp_path):
        out_dir, records, _ = _build(tmp_path, bound=2048)
        dataset = ShardedDataset(out_dir)
        assert len(dataset) == len(records)
        for i in (0, 1, 17, 32, 63):
            got = dataset[i]
            assert got["complex_id"] == records[i]["complex_id"]
            assert torch.equal(got["coords"], records[i]["coords"])
            assert got["energy"] == records[i]["energy"]

    def test_preload_to_ram_round_trip(self, tmp_path):
        out_dir, records, _ = _build(tmp_path, bound=2048)
        dataset = ShardedDataset(out_dir, preload_to_ram=True)
        assert len(dataset._entries) == len(dataset.manifest.shards)
        assert [dataset[i]["complex_id"] for i in range(len(records))] == [
            r["complex_id"] for r in records
        ]

    def test_lazy_lru_keeps_memory_bounded(self, tmp_path):
        out_dir, records, _ = _build(tmp_path, bound=2048)
        dataset = ShardedDataset(out_dir, max_open_shards=1)
        for i in range(len(records)):
            dataset[i]
        assert len(dataset._entries) <= 1

    def test_data_loader_integration_with_sampler(self, tmp_path):
        # uniform tensor shapes so the default collate stacks them (the
        # collate contract belongs to the #26 trainer, not this layer)
        out_dir = str(tmp_path / "shards")
        generator = torch.Generator().manual_seed(31)
        records = [
            {"complex_id": f"cpx_{i:04d}", "energy": float(i),
             "coords": torch.randn(8, 3, generator=generator)}
            for i in range(48)
        ]
        build_shards(records, out_dir, shard_bound_bytes=2048, prefix="traj-train")
        dataset = ShardedDataset(out_dir, max_open_shards=1)
        sampler = ShardAwareShuffleSampler(dataset, seed=7, epoch=0)
        loader = DataLoader(dataset, batch_size=8, sampler=sampler)
        seen = []
        for batch in loader:
            seen += [int(e) for e in batch["energy"]]
        # every record exactly once, energies map 1:1 onto the input set
        assert sorted(seen) == sorted(int(r["energy"]) for r in records)

    def test_corrupted_shard_fails_closed(self, tmp_path):
        out_dir, records, _ = _build(tmp_path, bound=2048)
        target = os.path.join(out_dir, dataset_first_shard_name(out_dir))
        with open(target, "r+b") as handle:
            handle.seek(200)
            byte = handle.read(1)
            handle.seek(200)
            handle.write(bytes([byte[0] ^ 0xFF]))
        dataset = ShardedDataset(out_dir, max_open_shards=1)
        with pytest.raises(ShardIntegrityError, match="digest mismatch"):
            dataset[0]

    def test_manifest_record_count_mismatch_fails_closed(self, tmp_path):
        out_dir, records, _ = _build(tmp_path, bound=2048)
        path = os.path.join(out_dir, "manifest.json")
        doc = json.load(open(path))
        doc["shards"][0]["records"] += 1  # lie about the count
        json.dump(doc, open(path, "w"))
        dataset = ShardedDataset(out_dir, max_open_shards=1)
        with pytest.raises(ShardIntegrityError, match="records"):
            dataset[0]

    def test_wrong_schema_version_fails_closed(self, tmp_path):
        out_dir, records, _ = _build(tmp_path, bound=2048)
        path = os.path.join(out_dir, "manifest.json")
        doc = json.load(open(path))
        doc["schema_version"] = "sharded-dataset-v0.9"
        json.dump(doc, open(path, "w"))
        with pytest.raises(ShardManifestError, match="schema"):
            ShardedDataset(out_dir)


def dataset_first_shard_name(out_dir: str) -> str:
    return load_shard_manifest(out_dir).shards[0].name


class TestShardedDatasetHubStreaming:
    def test_hub_streaming_is_interchangeable_with_local(self, tmp_path, monkeypatch):
        """Missing local shards are fetched via hf_hub_download and yield
        byte-identical records (zero corruption) — the simulated remote is
        a directory with the exact uploaded bytes."""
        import huggingface_hub

        out_dir, records, _ = _build(tmp_path / "build", bound=2048)
        # the "remote repo" is a separate directory (repo layout keeps
        # splits as subdirectories, like the real upload); the local root
        # has ONLY the manifest (shards missing -> streaming path)
        remote = tmp_path / "remote" / "train"
        local = tmp_path / "local"
        local.mkdir()
        import shutil

        for name in os.listdir(out_dir):
            if name.endswith(".pt.gz"):
                remote.mkdir(parents=True, exist_ok=True)
                shutil.copy(os.path.join(out_dir, name), str(remote / name))
        shutil.copy(os.path.join(out_dir, "manifest.json"), str(remote / "manifest.json"))
        shutil.copy(os.path.join(out_dir, "manifest.json"), str(local / "manifest.json"))

        def fake_download(*, repo_id, filename, repo_type, token, local_dir, **kw):
            assert repo_id == "some/repo" and repo_type == "dataset"
            assert filename.startswith("train/")  # repo_subdir composed
            source = os.path.join(str(remote), os.path.basename(filename))
            destination = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(source, destination)
            return destination

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

        dataset = ShardedDataset(str(local), repo_id="some/repo", repo_subdir="train")
        assert len(dataset) == len(records)
        # full pass: every record identical through the streaming path
        for i in range(len(records)):
            got = dataset[i]
            assert got["complex_id"] == records[i]["complex_id"]
            assert torch.equal(got["coords"], records[i]["coords"])
        # every fetched shard was normalized into the canonical flat layout
        for entry in dataset.manifest.shards:
            assert os.path.isfile(os.path.join(str(local), entry.name)), entry.name

    def test_missing_shard_without_repo_fails_closed(self, tmp_path):
        out_dir, records, _ = _build(tmp_path / "build", bound=2048)
        local = tmp_path / "local"
        local.mkdir()
        import shutil

        shutil.copy(os.path.join(out_dir, "manifest.json"), str(local / "manifest.json"))
        dataset = ShardedDataset(str(local))  # no repo_id
        with pytest.raises(ShardIntegrityError, match="no repo_id"):
            dataset[0]


class TestShardAwareShuffleSampler:
    def _dataset(self, tmp_path):
        out_dir, records, _ = _build(tmp_path, bound=2048)
        return ShardedDataset(out_dir), records

    def test_full_unbiased_permutation_each_epoch(self, tmp_path):
        dataset, records = self._dataset(tmp_path)
        sampler = ShardAwareShuffleSampler(dataset, seed=3, epoch=0)
        order = list(iter(sampler))
        assert sorted(order) == list(range(len(records)))
        assert len(order) == len(set(order))

    def test_shard_locality_no_cross_shard_seeking(self, tmp_path):
        dataset, records = self._dataset(tmp_path)
        assert len(dataset.manifest.shards) >= 4
        order = list(iter(ShardAwareShuffleSampler(dataset, seed=3, epoch=0)))
        shards_seen = [dataset.shard_of(i)[0] for i in order]
        # each shard's indices form ONE contiguous block in the yielded order
        positions: dict = {}
        for pos, shard in enumerate(shards_seen):
            positions.setdefault(shard, []).append(pos)
        for shard, pos_list in positions.items():
            assert pos_list == list(
                range(pos_list[0], pos_list[0] + len(pos_list))
            ), f"shard {shard} indices are split across the epoch"

    def test_epoch_seeded_determinism(self, tmp_path):
        dataset, records = self._dataset(tmp_path)
        a1 = list(iter(ShardAwareShuffleSampler(dataset, seed=11, epoch=0)))
        a2 = list(iter(ShardAwareShuffleSampler(dataset, seed=11, epoch=0)))
        b = list(iter(ShardAwareShuffleSampler(dataset, seed=11, epoch=1)))
        assert a1 == a2, "same seed + epoch must reproduce the same order"
        assert a1 != b or len(records) < 3, "epochs should re-shuffle"

    def test_global_rng_state_is_not_disturbed(self, tmp_path):
        dataset, _ = self._dataset(tmp_path)
        from syntree.utils.seeding import capture_rng_state, restore_rng_state

        snapshot = capture_rng_state()
        try:
            list(iter(ShardAwareShuffleSampler(dataset, seed=99, epoch=0)))
            mid = capture_rng_state()
            assert torch.equal(snapshot.torch, mid.torch)
        finally:
            restore_rng_state(snapshot)

    def test_rejects_non_sharded_dataset(self, tmp_path):
        with pytest.raises(TypeError, match="ShardedDataset"):
            ShardAwareShuffleSampler([1, 2, 3])


class TestManifestContract:
    def test_estimate_record_bytes_is_exact(self):
        record = {"a": torch.randn(5, 3), "b": "text"}
        import io as _io

        buffer = _io.BytesIO()
        torch.save(record, buffer)
        assert estimate_record_bytes(record) == buffer.getbuffer().nbytes

    def test_manifest_round_trip(self, tmp_path):
        out_dir, records, manifest = _build(tmp_path, bound=4096)
        loaded = load_shard_manifest(out_dir)
        assert loaded == manifest
        assert loaded.source == {"kind": "test", "split": "train"}
