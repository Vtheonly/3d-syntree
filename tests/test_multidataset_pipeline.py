from pathlib import Path

import torch
from torch_geometric.data import Data

from syntree.data.hf_loader import ShardedHuggingFaceDataset
from scripts.shard_and_upload import write_shards


def test_byte_bounded_shards_have_manifest_ranges(tmp_path):
    samples = [Data(x=torch.arange(i + 1, dtype=torch.float32)) for i in range(9)]
    manifest = write_shards(samples, "train", str(tmp_path), max_shard_bytes=2048)

    assert manifest["total_samples"] == 9
    assert manifest["shards"]
    assert all(s["sample_end"] - s["sample_start"] == s["sample_count"]
               for s in manifest["shards"])
    assert all(s["name"].endswith(".pt.gz") for s in manifest["shards"])


def test_loader_handles_variable_shard_sizes(monkeypatch, tmp_path):
    samples = [Data(x=torch.tensor([i])) for i in range(5)]
    manifest = write_shards(samples, "train", str(tmp_path), max_shard_bytes=2048)

    import syntree.data.hf_loader as module
    monkeypatch.setattr(
        module,
        "hf_hub_download",
        lambda **kwargs: str(tmp_path / "train" / "manifest.json"),
    )
    ds = ShardedHuggingFaceDataset(
        "fake/repo",
        split="train",
        cache_dir=str(tmp_path / "cache"),
    )

    import shutil
    for shard in manifest["shards"]:
        dst = ds.cache_dir / Path(shard["name"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path / "train" / shard["name"], dst)

    assert len(ds) == 5
    assert int(ds[4].x.item()) == 4
