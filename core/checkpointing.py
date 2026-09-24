"""Shard-level incremental checkpointing and resume management."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.checksums import compute_sha256, verify_file
from core.manifest import ManifestManager, ShardRecord


def serialize_samples(samples: Sequence[Any]) -> bytes:
    """Serialize a sample list into deterministic gzipped torch bytes.

    ``mtime=0`` pins the gzip header timestamp so identical data always
    produces identical SHA-256 digests (see scripts/shard_and_upload.py).
    ``torch`` is imported lazily so the rest of ``core`` stays importable
    on CPU-only manifest tooling machines.
    """
    import torch  # local import: keeps core importable without torch

    buffer = io.BytesIO()
    torch.save(list(samples), buffer, _use_new_zipfile_serialization=True)
    return gzip.compress(buffer.getvalue(), compresslevel=6, mtime=0)


class ShardCheckpointManager:
    """Manages idempotent shard creation, incremental saving, and validation.

    On construction the manager re-reads ``<split>/manifest.json``, verifies
    every declared shard on disk by SHA-256 and replays the complex IDs that
    are already safely committed, stopping at the first missing/corrupt shard
    (fail-closed: a partial write never looks completed).
    """

    def __init__(
        self,
        output_dir: Path | str,
        split: str,
        max_shard_bytes: int = 500 * 1024 * 1024,
    ) -> None:
        self.root_dir = Path(output_dir).resolve()
        self.output_dir = self.root_dir / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.max_shard_bytes = int(max_shard_bytes)
        self.manifest_path = self.output_dir / "manifest.json"
        self.shards: List[ShardRecord] = []
        self.completed_complex_ids: set[str] = set()
        self._load_existing_state()

    # ------------------------------------------------------------------
    # Resume state
    # ------------------------------------------------------------------
    def _load_existing_state(self) -> None:
        if not self.manifest_path.is_file():
            return
        try:
            data = ManifestManager.load(self.manifest_path)
        except Exception:
            self.shards = []
            self.completed_complex_ids = set()
            return

        for idx, s in enumerate(data.get("shards", [])):
            try:
                filename = s.get("filename") or s.get("name")
                shard_path = self.output_dir / Path(filename).name
                if not (
                    shard_path.is_file()
                    and verify_file(shard_path, s["sha256"])
                    and int(s["sample_count"]) > 0
                ):
                    break  # gap or corrupt tail: never trust past it
                record = ShardRecord(
                    shard_id=int(s.get("shard_id", idx)),
                    filename=filename,
                    sample_count=int(s["sample_count"]),
                    compressed_bytes=int(s.get("compressed_bytes", 0)),
                    sha256=s["sha256"],
                    source_complex_ids=list(s.get("source_complex_ids", [])),
                    split=s.get("split", self.split),
                    verified=True,
                    name=s.get("name", filename),
                    sample_start=int(s.get("sample_start", 0)),
                    sample_end=int(s.get("sample_end", 0)),
                )
                self.shards.append(record)
                self.completed_complex_ids.update(record.source_complex_ids)
            except Exception:
                break

    def is_complex_completed(self, complex_id: str) -> bool:
        return complex_id in self.completed_complex_ids

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def _next_sample_start(self) -> int:
        return sum(s.sample_count for s in self.shards)

    def _write_batch(
        self,
        samples: Sequence[Any],
        start: int,
        source_complexes: Sequence[str],
        shard_id: int,
    ) -> List[ShardRecord]:
        """Serialize a batch, bisecting when it exceeds ``max_shard_bytes``."""
        payload = serialize_samples(samples)
        if len(payload) > self.max_shard_bytes and len(samples) > 1:
            mid = len(samples) // 2
            left = self._write_batch(samples[:mid], start, source_complexes, shard_id)
            right = self._write_batch(
                samples[mid:], start + mid, source_complexes, shard_id + len(left)
            )
            return left + right

        filename = f"{self.split}_shard_{shard_id:04d}.pt.gz"
        dest_path = self.output_dir / filename
        # Atomic file: write to a temp name, then rename into place.
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        tmp_path.write_bytes(payload)
        tmp_path.replace(dest_path)

        record = ShardRecord(
            shard_id=shard_id,
            filename=filename,
            sample_count=len(samples),
            compressed_bytes=len(payload),
            sha256=compute_sha256(dest_path),
            source_complex_ids=list(source_complexes),
            split=self.split,
            verified=True,
            name=filename,
            sample_start=start,
            sample_end=start + len(samples),
        )
        self.shards.append(record)
        self.completed_complex_ids.update(source_complexes)
        return [record]

    def write_shard(
        self, samples: Sequence[Any], source_complexes: Sequence[str]
    ) -> List[ShardRecord]:
        """Compress samples into deterministic .pt.gz shard(s) and register them.

        Returns every :class:`ShardRecord` written (a batch larger than
        ``max_shard_bytes`` is bisected into multiple shards). The manifest is
        persisted atomically after the file(s) are on disk, so a crash between
        the two leaves unlisted files that the next resume simply overwrites.
        """
        items = list(samples)
        if not items:
            raise ValueError("write_shard requires at least one sample")
        start = self._next_sample_start()
        written = self._write_batch(items, start, list(source_complexes), len(self.shards))
        self._persist_manifest()
        return written

    def _persist_manifest(self) -> None:
        manifest_data: Dict[str, Any] = {
            "format": "torch-pyg-list+gzip",
            "version": 2,
            "split": self.split,
            "total_samples": sum(s.sample_count for s in self.shards),
            "max_shard_bytes": self.max_shard_bytes,
            "shards": [self._record_asdict(s) for s in self.shards],
        }
        ManifestManager.save_atomic(manifest_data, self.manifest_path)

    @staticmethod
    def _record_asdict(record: ShardRecord) -> Dict[str, Any]:
        return {
            "shard_id": record.shard_id,
            "name": record.name,
            "filename": record.filename,
            "sample_start": record.sample_start,
            "sample_count": record.sample_count,
            "sample_end": record.sample_end,
            "compressed_bytes": record.compressed_bytes,
            "sha256": record.sha256,
            "source_complex_ids": list(record.source_complex_ids),
            "split": record.split,
            "verified": record.verified,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def total_samples(self) -> int:
        return sum(s.sample_count for s in self.shards)

    def manifest_dict(self) -> Dict[str, Any]:
        return {
            "format": "torch-pyg-list+gzip",
            "version": 2,
            "split": self.split,
            "total_samples": self.total_samples,
            "max_shard_bytes": self.max_shard_bytes,
            "shards": [self._record_asdict(s) for s in self.shards],
        }
