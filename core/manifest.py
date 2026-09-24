"""Atomic JSON manifest management for raw sources and processed shards."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FileRecord:
    path: str
    size_bytes: int
    sha256: str
    source_url: Optional[str] = None
    created_at: float = field(default_factory=lambda: 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardRecord:
    shard_id: int
    filename: str
    sample_count: int
    compressed_bytes: int
    sha256: str
    source_complex_ids: List[str]
    split: str
    verified: bool = False


class ManifestManager:
    """Read, validate, and atomically persist dataset manifests."""

    @staticmethod
    def save_atomic(data: Dict[str, Any], target_path: Path | str) -> None:
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2, sort_keys=True)
            temp_name = tf.name
        os.replace(temp_name, target)

    @staticmethod
    def load(path: Path | str) -> Dict[str, Any]:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Manifest not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)