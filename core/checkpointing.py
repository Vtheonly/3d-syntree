"""Shard-level incremental checkpointing and resume management."""

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Set
import torch

from core.checksums import compute_sha256, verify_file
from core.manifest import ManifestManager, ShardRecord


class ShardCheckpointManager:
    """Manages idempotent shard creation, incremental saving, and validation."""

    def __init__(self, output_dir: Path | str, split: str, max_shard_bytes: int = 500 * 1024 * 1024):
        self.output_dir = Path(output_dir).resolve() / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.max_shard_bytes = max_shard_bytes
        self.manifest_path = self.output_dir / "manifest.json"
        self.shards: List[ShardRecord] = []
        self.completed_complex_ids: Set[str] = set()
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        if not self.manifest_path.is_file():
            return
        try:
            data = ManifestManager.load(self.manifest_path)
            for s in data.get("shards", []):
                shard_path = self.output_dir / s["filename"]
                if shard_path.is_file() and verify_file(shard_path, s["sha256"]):
                    record = ShardRecord(**s)
                    self.shards.append(record)
                    self.completed_complex_ids.update(record.source_complex_ids)
                else:
                    break
        except Exception:
            self.shards.clear()
            self.completed_complex_ids.clear()

    def is_complex_completed(self, complex_id: str) -> bool:
        return complex_id in self.completed_complex_ids

    def write_shard(self, samples: List[Any], source_complexes: List[str]) -> ShardRecord:
        """Compress samples into a deterministic .pt.gz shard and register it atomically."""
        shard_id = len(self.shards)
        filename = f"{self.split}_shard_{shard_id:04d}.pt.gz"
        dest_path = self.output_dir / filename

        buffer = io.BytesIO()
        torch.save(samples, buffer, _use_new_zipfile_serialization=True)
        compressed = gzip.compress(buffer.getvalue(), compresslevel=6, mtime=0)

        with open(dest_path, "wb") as f:
            f.write(compressed)

        sha = compute_sha256(dest_path)
        record = ShardRecord(
            shard_id=shard_id,
            filename=filename,
            sample_count=len(samples),
            compressed_bytes=len(compressed),
            sha256=sha,
            source_complex_ids=list(source_complexes),
            split=self.split,
            verified=True,
        )
        self.shards.append(record)
        self.completed_complex_ids.update(source_complexes)
        self._persist_manifest()
        return record

    def _persist_manifest(self) -> None:
        manifest_data = {
            "format": "torch-pyg-list+gzip",
            "version": 2,
            "split": self.split,
            "total_samples": sum(s.sample_count for s in self.shards),
            "max_shard_bytes": self.max_shard_bytes,
            "shards": [s.__dict__ for s in self.shards],
        }
        ManifestManager.save_atomic(manifest_data, self.manifest_path)