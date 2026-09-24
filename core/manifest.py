"""Atomic JSON manifest management for raw sources and processed shards."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ManifestManager:
    """Read, validate, and atomically persist dataset manifests."""

    @staticmethod
    def save_atomic(data: Dict[str, Any], target_path: Path | str) -> None:
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=target.parent, delete=False, encoding="utf-8"
            ) as tf:
                json.dump(data, tf, indent=2, sort_keys=True)
                temp_name = tf.name
            os.replace(temp_name, target)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    @staticmethod
    def load(path: Path | str) -> Dict[str, Any]:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Manifest not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)


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
    """One compressed shard, compatible with ``syntree.data.kaggle_loader``.

    The loader contract (``kaggle_loader._read_manifest`` /
    ``_KaggleShardIndex``) requires ``name``, ``sample_count`` and
    ``sha256`` on every shard entry; the extra fields power Craft Mode
    resume bookkeeping. ``filename`` stays a synonym of ``name`` for
    compatibility with manifests written by earlier tooling.
    """

    shard_id: int
    filename: str
    sample_count: int
    compressed_bytes: int
    sha256: str
    source_complex_ids: List[str]
    split: str
    verified: bool = False
    name: str = ""
    sample_start: int = 0
    sample_end: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.filename
        if not self.sample_end:
            self.sample_end = self.sample_start + self.sample_count
