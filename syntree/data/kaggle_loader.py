"""Offline dataset loading for the RTX 6000 / Kaggle accelerator environment.

The RTX 6000 Ada sessions are **fully offline**: any code path touching the
network (``huggingface_hub``, ``urllib``, ``pip``) crashes immediately. This
module reads the exact same manifest + shard format that
``scripts/shard_and_upload.py`` produces (and that
:class:`~syntree.data.hf_loader.ShardedHuggingFaceDataset` streams from the
Hub) directly from a mounted directory such as
``/kaggle/input/3d-syntree-dataset``:

    <dataset_dir>/
        enamine_3d_subset.parquet        # synthon catalog
        train/manifest.json              # {"shards": [{name, sample_count, sha256}], ...}
        train/train_shard_0000.pt.gz
        ...
        val/...
        test/...

Two flavours are provided:

* :class:`KaggleInMemoryDataset` (default, ``data.preload_to_ram: true``) -
  decompresses every shard once and keeps all samples in system RAM (the
  tasklist's 175 GB RAM exploitation; a full 30k-complex dataset is only
  ~8-15 GB), eliminating data-loading latency entirely.
* :class:`KaggleLazyDataset` (``data.preload_to_ram: false``) - LRU shard
  cache for RAM-constrained hosts, identical semantics otherwise.

Both verify SHA-256 digests and per-shard sample counts against the
manifest before trusting the data (data integrity must not depend on
network transport), fail closed on missing/corrupt shards, and never
import ``huggingface_hub`` or open a single socket.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

VALID_SPLITS = ("train", "val", "test")


def _read_manifest(dataset_dir: str, split: str) -> dict:
    """Load and validate a split manifest (fail-closed)."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")
    split_dir = Path(dataset_dir) / split
    manifest_file = split_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_file}. Stage the offline dataset "
            f"directory (build it with scripts/build_full_dataset.py + "
            f"scripts/shard_and_upload.py and mount it, e.g. at "
            f"/kaggle/input/3d-syntree-dataset)."
        )
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt manifest {manifest_file}: {exc}") from exc

    shards = list(manifest.get("shards", []))
    if not shards:
        raise ValueError(f"No shards found in {manifest_file}")

    total = 0
    for shard in shards:
        count = int(shard["sample_count"])
        if count <= 0:
            raise ValueError(
                f"Shard {shard.get('name')} declares non-positive sample_count"
            )
        total += count
    declared_total = int(manifest.get("total_samples", total))
    if declared_total != total:
        raise ValueError(
            f"Manifest total_samples ({declared_total}) does not match the sum "
            f"of shard sample_counts ({total})"
        )
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_shard_file(path: Path, expected_count: Optional[int] = None) -> List:
    """Decompress + deserialize one shard with integrity checks."""
    with gzip.open(path, "rb") as f:
        data = torch.load(f, weights_only=False)
    if not isinstance(data, list):
        raise TypeError(f"{path.name} does not contain a sample list")
    if expected_count is not None and len(data) != int(expected_count):
        raise ValueError(
            f"Shard {path.name} contains {len(data)} samples but its manifest "
            f"entry declares {expected_count}"
        )
    return data


def _fix_sample(sample) -> None:
    """Same num_nodes / 0-dim tensor fixups as the HF loader (in place)."""
    if hasattr(sample, "pocket_pos") and sample.pocket_pos is not None:
        sample.num_nodes = sample.pocket_pos.size(0)
    for k in list(sample.keys()):
        v = sample[k]
        if isinstance(v, torch.Tensor) and v.dim() == 0:
            sample[k] = v.unsqueeze(0)


class _KaggleShardIndex:
    """Shared manifest/index bookkeeping for the offline loaders."""

    def __init__(self, dataset_dir: str, split: str, verify_sha256: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.split_dir = self.dataset_dir / split
        self.verify_sha256 = bool(verify_sha256)
        self.manifest = _read_manifest(dataset_dir, split)
        self.shards: List[dict] = list(self.manifest["shards"])

        self._starts: List[int] = []
        self._ends: List[int] = []
        total = 0
        for shard in self.shards:
            count = int(shard["sample_count"])
            self._starts.append(total)
            total += count
            self._ends.append(total)
        self.total_samples = total

        # Fail closed on missing shard files up front: an offline run must
        # never discover a missing shard six hours into training.
        for shard in self.shards:
            path = self.split_dir / Path(shard["name"]).name
            if not path.exists():
                raise FileNotFoundError(f"Missing shard file: {path}")

    def __len__(self) -> int:
        return self.total_samples

    @property
    def shard_ranges(self) -> Sequence[Tuple[int, int]]:
        return tuple(zip(self._starts, self._ends))

    def locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += self.total_samples
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self._ends, idx)
        return shard_idx, idx - self._starts[shard_idx]

    def shard_path(self, shard_idx: int) -> Path:
        return self.split_dir / Path(self.shards[shard_idx]["name"]).name

    def load_shard_verified(self, shard_idx: int) -> List:
        """Load a shard with SHA-256 + sample-count verification."""
        path = self.shard_path(shard_idx)
        expected = self.shards[shard_idx].get("sha256")
        if expected and self.verify_sha256:
            actual = _sha256(path)
            if actual != expected:
                raise IOError(
                    f"SHA-256 mismatch for {path.name}: manifest expects "
                    f"{expected}, file hashes to {actual}. The offline dataset "
                    f"shard is corrupt - re-stage the dataset."
                )
        return _load_shard_file(
            path, expected_count=int(self.shards[shard_idx]["sample_count"])
        )


class KaggleInMemoryDataset(Dataset):
    """Offline, in-RAM dataset for mounted shard directories.

    Loads *all* shards of a split into system RAM once (the tasklist's
    175 GB RAM exploitation for Kaggle's RTX 6000 Ada sessions), verifying
    every SHA-256 digest and sample count on the way in. Data access during
    training is then a plain list index - zero I/O, zero decompression,
    zero network.
    """

    def __init__(self, dataset_dir: str, split: str = "train",
                 verify_sha256: bool = True):
        self._index = _KaggleShardIndex(dataset_dir, split, verify_sha256)
        self.split = split

        logger.info(
            "[kaggle_loader] preloading %d shards (%d samples) for split "
            "'%s' from %s into RAM...",
            len(self._index.shards), self._index.total_samples, split,
            self._index.split_dir,
        )
        self.samples: List = []
        for shard_idx in range(len(self._index.shards)):
            data = self._index.load_shard_verified(shard_idx)
            for sample in data:
                _fix_sample(sample)
            self.samples.extend(data)
        logger.info(
            "[kaggle_loader] %d '%s' samples resident in RAM.",
            len(self.samples), split,
        )
        if not self.samples:
            raise ValueError(
                f"Offline split '{split}' in {dataset_dir} is empty"
            )

    # Dataset contract -------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[int(idx)]

    # Trainer-facing properties ----------------------------------------
    @property
    def num_synthetic(self) -> int:
        return 0

    @property
    def use_synthetic(self) -> bool:
        return False

    @property
    def manifest(self) -> dict:
        return self._index.manifest

    @property
    def shard_ranges(self):
        return self._index.shard_ranges


class KaggleLazyDataset(Dataset):
    """Offline lazy shard loader with an LRU cache.

    Same manifest format and integrity checks as
    :class:`KaggleInMemoryDataset`, but shards are decompressed on demand
    and at most ``max_cached_shards`` stay in RAM - for hosts without the
    RAM headroom to hold the whole split.
    """

    def __init__(self, dataset_dir: str, split: str = "train",
                 max_cached_shards: int = 2, verify_sha256: bool = True):
        self._index = _KaggleShardIndex(dataset_dir, split, verify_sha256)
        self.split = split
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._cache: Dict[int, List] = {}
        self._cache_order: List[int] = []

    def __len__(self) -> int:
        return self._index.total_samples

    def _shard(self, shard_idx: int) -> List:
        if shard_idx in self._cache:
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)
            return self._cache[shard_idx]
        data = self._index.load_shard_verified(shard_idx)
        for sample in data:
            _fix_sample(sample)
        self._cache[shard_idx] = data
        self._cache_order.append(shard_idx)
        while len(self._cache_order) > self.max_cached_shards:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return data

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._index.locate(int(idx))
        return self._shard(shard_idx)[local_idx]

    # Trainer-facing properties ----------------------------------------
    @property
    def num_synthetic(self) -> int:
        return 0

    @property
    def use_synthetic(self) -> bool:
        return False

    @property
    def manifest(self) -> dict:
        return self._index.manifest

    @property
    def shard_ranges(self):
        return self._index.shard_ranges


__all__ = ["KaggleInMemoryDataset", "KaggleLazyDataset"]
