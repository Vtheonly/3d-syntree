"""Manifest-indexed lazy Hugging Face dataset loading."""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class ShardedHuggingFaceDataset(Dataset):
    """Lazy map-style loader for gzip-compressed PyG shards."""

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        cache_dir: str = "./hf_cache",
        revision: str = "main",
        token: Optional[str] = None,
        max_cached_shards: int = 2,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train/val/test")

        self.repo_id = repo_id
        self.split = split
        self.revision = revision
        self.token = token
        self.cache_dir = Path(cache_dir).expanduser() / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cached_shards = max(1, int(max_cached_shards))

        manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"data/{split}/manifest.json",
            repo_type="dataset",
            revision=revision,
            token=token,
            local_dir=str(self.cache_dir),
        )
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise RuntimeError(f"No shards declared for {repo_id} split={split}")

        self._starts: List[int] = []
        self._ends: List[int] = []
        total = 0
        for shard in self.shards:
            count = int(shard["sample_count"])
            if count <= 0:
                raise ValueError("Shard sample_count must be positive")
            self._starts.append(total)
            total += count
            self._ends.append(total)

        self.total_samples = int(self.manifest.get("total_samples", total))
        if self.total_samples != total:
            raise ValueError("Manifest sample total does not match shard counts")

        self._cache: Dict[int, List] = {}
        self._cache_order: List[int] = []

    def __len__(self) -> int:
        return self.total_samples

    @property
    def shard_ranges(self) -> Sequence[Tuple[int, int]]:
        return tuple(zip(self._starts, self._ends))

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self._ends, idx)
        return shard_idx, idx - self._starts[shard_idx]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_shard(self, shard_idx: int) -> List:
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]

        meta = self.shards[shard_idx]
        local_path = self.cache_dir / Path(meta["name"]).name
        if not local_path.exists():
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f"data/{self.split}/{meta['name']}",
                    repo_type="dataset",
                    revision=self.revision,
                    token=self.token,
                    local_dir=str(self.cache_dir),
                )
            )
            if downloaded_path != local_path:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded_path, local_path)

        expected = meta.get("sha256")
        if expected and self._sha256(local_path) != expected:
            raise IOError(f"SHA-256 mismatch for {local_path.name}")

        with gzip.open(local_path, "rb") as f:
            data = torch.load(f, weights_only=False)

        if not isinstance(data, list):
            raise TypeError(f"{local_path.name} does not contain a sample list")
        if len(data) != int(meta["sample_count"]):
            raise ValueError(f"Shard {local_path.name} sample count mismatch")

        self._cache[shard_idx] = data
        self._cache_order.append(shard_idx)
        while len(self._cache_order) > self.max_cached_shards:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return data

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(int(idx))
        sample = self._load_shard(shard_idx)[local_idx]
        # Ensure num_nodes is defined on loaded sample to protect PyG Batch collation
        if hasattr(sample, "pocket_pos") and sample.pocket_pos is not None:
            sample.num_nodes = sample.pocket_pos.size(0)
        for k in list(sample.keys()):
            v = sample[k]
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                sample[k] = v.unsqueeze(0)
        return sample


class ShardAwareShuffleSampler(Sampler):
    """Shuffle shard order, then indices within each shard."""

    def __init__(self, dataset: ShardedHuggingFaceDataset, seed: int = 42):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shard_ids = list(range(len(self.dataset.shard_ranges)))
        rng.shuffle(shard_ids)
        for shard_idx in shard_ids:
            start, end = self.dataset.shard_ranges[shard_idx]
            indices = list(range(start, end))
            rng.shuffle(indices)
            yield from indices


__all__ = ["ShardedHuggingFaceDataset", "ShardAwareShuffleSampler"]