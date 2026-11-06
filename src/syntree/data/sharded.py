"""Unified sharded dataset layer (#19 DATA-04) on the canonical shard format.

One authoritative high-throughput data layer for the whole V2 system
(decision D14: the four competing V1 loaders — ``crossdocked.py``,
``kaggle_loader.py``, ``hf_loader.py``, … — are replaced by exactly this
module).  It builds strictly on the canonical shard file format of
:mod:`syntree.data.shards` (#32 Suite A primitive — deterministic gzip
``mtime=0`` + SHA-256): the format is NOT re-implemented here.

Components
----------
* :func:`build_shards` — byte-bounded packing of records into ``.pt.gz``
  shards under ``<out_dir>`` plus an integrity ``manifest.json`` (per
  shard: name, record count, byte size, SHA-256; fail-closed on any
  shard exceeding the bound).
* :class:`ShardedDataset` — one :class:`torch.utils.data.Dataset` over
  a shard directory: local mount operation (Kaggle ``/kaggle/input`` or
  local NVMe), on-demand ``hf_hub_download`` streaming of missing
  shards when a ``repo_id`` is configured, optional ``preload_to_ram``
  decompression on high-RAM nodes, and digest verification on every
  shard read (corruption never yields data, #45 §15).
* :class:`ShardAwareShuffleSampler` — epoch-seeded shuffle that
  permutes **shard order first, then intra-shard indices**, eliminating
  cross-shard seeking while remaining a full unbiased permutation of
  every index exactly once per epoch.

The manifest is the integrity contract between the producer (the
``scripts/shard_and_upload.py`` driver) and every consumer (the #26
trainer, the #27 curriculum, evaluation): shard digests are re-verified
at load time, never trusted from filenames alone.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from syntree.data.shards import ShardIntegrityError, read_shard, write_shard

__all__ = [
    "DEFAULT_SHARD_BOUND_BYTES",
    "ShardEntry",
    "ShardManifest",
    "ShardManifestError",
    "build_shards",
    "ShardedDataset",
    "ShardAwareShuffleSampler",
    "estimate_record_bytes",
    "load_shard_manifest",
]

#: The issue-pinned shard bound: 500 MB (``500 * 1024**2`` bytes).
DEFAULT_SHARD_BOUND_BYTES = 500 * 1024 * 1024

MANIFEST_SCHEMA_VERSION = "sharded-dataset-v1.0"
MANIFEST_NAME = "manifest.json"


class ShardManifestError(ValueError):
    """Fail-closed manifest structural error (missing/malformed fields)."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardEntry:
    """One shard's integrity record."""

    name: str
    records: int
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ShardManifest:
    """The shard-directory integrity contract."""

    schema_version: str
    created_utc: str
    shard_bound_bytes: int
    shards: Tuple[ShardEntry, ...]
    source: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_records(self) -> int:
        return sum(entry.records for entry in self.shards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "shard_bound_bytes": self.shard_bound_bytes,
            "num_records": self.num_records,
            "shards": [
                {
                    "name": entry.name,
                    "records": entry.records,
                    "sha256": entry.sha256,
                    "bytes": entry.bytes,
                }
                for entry in self.shards
            ],
            "source": self.source,
        }

    def write(self, out_dir: str, *, manifest_name: str = MANIFEST_NAME) -> str:
        """Atomically write the manifest JSON into ``out_dir``."""
        path = os.path.join(out_dir, manifest_name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
        return path


def load_shard_manifest(
    root: str, *, manifest_name: str = MANIFEST_NAME
) -> ShardManifest:
    """Load + structurally validate the manifest at ``root`` (fail closed)."""
    path = os.path.join(root, manifest_name)
    if not os.path.isfile(path):
        raise ShardManifestError(f"manifest not found: {path} (fail-closed)")
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    required = {"schema_version", "shard_bound_bytes", "shards"}
    missing = required - set(doc)
    if missing:
        raise ShardManifestError(f"manifest missing fields: {sorted(missing)}")
    if doc["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ShardManifestError(
            f"manifest schema {doc['schema_version']!r} != {MANIFEST_SCHEMA_VERSION!r} "
            "(fail-closed version guard)"
        )
    entries = []
    for shard in doc["shards"]:
        try:
            entries.append(
                ShardEntry(
                    name=str(shard["name"]),
                    records=int(shard["records"]),
                    sha256=str(shard["sha256"]),
                    bytes=int(shard["bytes"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ShardManifestError(f"malformed shard entry {shard!r}") from error
    return ShardManifest(
        schema_version=doc["schema_version"],
        created_utc=str(doc.get("created_utc", "")),
        shard_bound_bytes=int(doc["shard_bound_bytes"]),
        shards=tuple(entries),
        source=dict(doc.get("source") or {}),
    )


# ---------------------------------------------------------------------------
# Byte-bounded packing
# ---------------------------------------------------------------------------


def estimate_record_bytes(record: Any) -> int:
    """The exact serialized size of ``record`` under ``torch.save``.

    Deterministic (same record -> same size), so packing decisions are a
    pure function of the record stream.
    """
    buffer = io.BytesIO()
    torch.save(record, buffer)
    return buffer.getbuffer().nbytes


def build_shards(
    records: Sequence[Any],
    out_dir: str,
    *,
    shard_bound_bytes: int = DEFAULT_SHARD_BOUND_BYTES,
    prefix: str = "shard",
    source: Optional[Dict[str, Any]] = None,
    manifest_name: str = MANIFEST_NAME,
) -> ShardManifest:
    """Pack ``records`` into byte-bounded ``.pt.gz`` shards + a manifest.

    Deterministic greedy packing: records are appended to the current
    shard until the *serialized* payload would exceed ``shard_bound_bytes``
    (each record's exact ``torch.save`` size is measured once); the shard
    is then written through the canonical :func:`write_shard` (deterministic
    gzip, ``mtime=0``) and its **actual compressed byte size is verified
    against the bound, failing closed** — the bound is a hard contract on
    the emitted files, never an aspiration.  A record whose serialized
    size alone exceeds the bound has no valid packing and fails closed
    immediately (the caller must raise the bound or split the record
    upstream) — no oversized shard is ever written.

    Returns the written :class:`ShardManifest` (also written to disk).
    """
    if shard_bound_bytes <= 0:
        raise ValueError(f"shard_bound_bytes must be positive: {shard_bound_bytes}")
    os.makedirs(out_dir, exist_ok=True)

    entries: List[ShardEntry] = []
    batch: List[Any] = []
    batch_bytes = 0
    serial = 0

    def flush() -> None:
        nonlocal batch, batch_bytes, serial
        if not batch:
            return
        name = f"{prefix}-{serial:05d}.pt.gz"
        path = os.path.join(out_dir, name)
        digest = write_shard(path, batch)
        size = os.path.getsize(path)
        if size > shard_bound_bytes:
            raise ShardIntegrityError(
                f"shard {name} is {size} bytes > bound {shard_bound_bytes}: the "
                "compressed payload exceeded the byte bound — raise the bound "
                "or split the record stream (fail-closed)"
            )
        entries.append(
            ShardEntry(name=name, records=len(batch), sha256=digest, bytes=size)
        )
        batch, batch_bytes = [], 0
        serial += 1

    for record in records:
        size = estimate_record_bytes(record)
        if size > shard_bound_bytes:
            raise ShardIntegrityError(
                f"a single record serializes to {size} bytes > shard bound "
                f"{shard_bound_bytes} — no valid packing exists (fail-closed)"
            )
        if batch and batch_bytes + size > shard_bound_bytes:
            flush()
        batch.append(record)
        batch_bytes += size
    flush()

    manifest = ShardManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        shard_bound_bytes=shard_bound_bytes,
        shards=tuple(entries),
        source=dict(source or {}),
    )
    manifest.write(out_dir, manifest_name=manifest_name)
    return manifest


# ---------------------------------------------------------------------------
# The unified dataset
# ---------------------------------------------------------------------------


class ShardedDataset(Dataset):
    """One dataset over a shard directory, local or HF-streamed.

    Parameters
    ----------
    root:
        The shard directory (local mount — Kaggle ``/kaggle/input/...``,
        local NVMe, or any local cache directory).
    repo_id:
        Optional Hugging Face dataset repo.  When a shard file is missing
        locally it is fetched on demand with ``hf_hub_download`` into the
        local directory first (streaming-on-demand semantics) — the
        digest from the manifest is verified after the fetch, exactly as
        for locally present files.
    repo_subdir:
        Optional path prefix of this dataset inside the repo (e.g.
        ``"train"`` when the repo stores splits as subdirectories).  The
        fetched file is normalized to the canonical local position
        ``<root>/<shard name>`` after download.
    preload_to_ram:
        When ``True``, every shard is decompressed (and digest-verified)
        up front and held in host memory — the high-RAM-node fast path.
        When ``False``, shards are loaded lazily behind a small LRU of
        open shard payloads (``max_open_shards``), bounded RAM.
    max_open_shards:
        LRU capacity of the lazy path.  ``1`` gives pure sequential
        locality (the ``ShardAwareShuffleSampler`` makes this optimal);
        higher values trade RAM for random-access resilience.
    verify:
        Verify each shard's SHA-256 against the manifest on load
        (default: on — corruption never yields data).
    """

    def __init__(
        self,
        root: str,
        *,
        manifest_name: str = MANIFEST_NAME,
        repo_id: Optional[str] = None,
        repo_subdir: Optional[str] = None,
        hf_token: Optional[str] = None,
        preload_to_ram: bool = False,
        max_open_shards: int = 1,
        verify: bool = True,
    ) -> None:
        self.root = root
        self.repo_id = repo_id
        self.repo_subdir = repo_subdir
        self.hf_token = hf_token
        self.verify = verify
        self.manifest = load_shard_manifest(root, manifest_name=manifest_name)
        self._shard_names = tuple(entry.name for entry in self.manifest.shards)
        # global index -> (shard, offset) via cumulative offsets
        self._offsets: List[int] = [0]
        for entry in self.manifest.shards:
            self._offsets.append(self._offsets[-1] + entry.records)
        self._entries: Dict[int, List[Any]] = {}  # shard idx -> records
        self._lru: List[int] = []
        self._max_open_shards = max(1, int(max_open_shards))
        self.preload_to_ram = bool(preload_to_ram)
        if self.preload_to_ram:
            for shard_index in range(len(self._shard_names)):
                self._load_shard(shard_index)

    # -- shard access ------------------------------------------------------

    def _shard_path(self, shard_index: int) -> str:
        return os.path.join(self.root, self._shard_names[shard_index])

    def _fetch_from_hub(self, shard_index: int) -> str:
        """Fetch a missing shard via ``hf_hub_download`` (on demand)."""
        if self.repo_id is None:
            raise ShardIntegrityError(
                f"shard {self._shard_names[shard_index]!r} missing under "
                f"{self.root} and no repo_id is configured for streaming "
                "(fail-closed)"
            )
        try:
            from huggingface_hub import hf_hub_download  # optional dependency
        except ImportError as error:  # pragma: no cover - environment-specific
            raise ShardIntegrityError(
                "huggingface_hub is not installed — install the 'hf' extra "
                "to enable shard streaming"
            ) from error
        remote_name = (
            f"{self.repo_subdir}/{self._shard_names[shard_index]}"
            if self.repo_subdir
            else self._shard_names[shard_index]
        )
        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=remote_name,
            repo_type="dataset",
            token=self.hf_token,
            local_dir=self.root,
        )
        path = str(path)
        # normalize to the canonical local position: hf_hub_download keeps
        # the repo-relative subpath under local_dir, the shard contract is
        # a flat <root>/<shard name> layout
        canonical = self._shard_path(shard_index)
        if os.path.abspath(path) != os.path.abspath(canonical):
            os.makedirs(self.root, exist_ok=True)
            os.replace(path, canonical)
        return canonical

    def _load_shard(self, shard_index: int) -> List[Any]:
        cached = self._entries.get(shard_index)
        if cached is not None:
            return cached
        path = self._shard_path(shard_index)
        if not os.path.isfile(path):
            path = self._fetch_from_hub(shard_index)
        expected = (
            self.manifest.shards[shard_index].sha256 if self.verify else None
        )
        records = read_shard(path, expected_sha256=expected)
        if len(records) != self.manifest.shards[shard_index].records:
            raise ShardIntegrityError(
                f"shard {self._shard_names[shard_index]} holds {len(records)} "
                f"records, manifest says "
                f"{self.manifest.shards[shard_index].records} (fail-closed)"
            )
        self._entries[shard_index] = records
        if not self.preload_to_ram:
            self._lru.append(shard_index)
            while len(self._lru) > self._max_open_shards:
                self._entries.pop(self._lru.pop(0), None)
        return records

    # -- dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} records")
        # binary search over cumulative offsets
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        shard_index = lo
        offset = index - self._offsets[shard_index]
        return self._load_shard(shard_index)[offset]

    def shard_of(self, index: int) -> Tuple[int, int]:
        """Global ``index`` -> ``(shard_index, intra_shard_offset)``."""
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} records")
        lo, hi = 0, len(self._offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._offsets[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        return lo, index - self._offsets[lo]


# ---------------------------------------------------------------------------
# Shard-aware shuffle sampling
# ---------------------------------------------------------------------------


class ShardAwareShuffleSampler(Sampler[int]):
    """Epoch-seeded shuffle: shard order first, then intra-shard indices.

    Yields a full unbiased permutation — every global index exactly once
    per epoch — while preserving shard locality: indices are visited
    shard-by-shard (no cross-shard seeking), matching the lazy-load LRU
    of :class:`ShardedDataset` (``max_open_shards=1`` optimum).

    Seeding follows the canonical :mod:`syntree.utils.seeding` offset
    convention (torch stream = ``seed + 1``; here composed with the epoch:
    ``torch_seed = seed + 1 + epoch``) and draws from a private
    ``torch.Generator`` so seeding the sampler never disturbs the global
    RNG streams (checkpoint continuation, #45 §13.6).
    """

    def __init__(
        self,
        dataset: "ShardedDataset",
        *,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        if not isinstance(dataset, ShardedDataset):
            raise TypeError(
                f"ShardAwareShuffleSampler needs a ShardedDataset, got "
                f"{type(dataset).__name__}"
            )
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = int(epoch)

    def _generator(self) -> torch.Generator:
        generator = torch.Generator()
        # canonical torch-stream offset (seed + 1), composed with the epoch
        generator.manual_seed((self.seed + 1 + self.epoch) % (2**63 - 1))
        return generator

    def __iter__(self) -> Iterator[int]:
        generator = self._generator()
        order = torch.randperm(len(self.dataset.manifest.shards), generator=generator)
        for shard_index in order.tolist():
            entry = self.dataset.manifest.shards[shard_index]
            base = self.dataset._offsets[shard_index]
            local = torch.randperm(entry.records, generator=generator).tolist()
            for offset in local:
                yield base + offset

    def __len__(self) -> int:
        return len(self.dataset)
