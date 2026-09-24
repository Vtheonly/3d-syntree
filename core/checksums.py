"""High-throughput streaming SHA-256 verification."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(file_path: Path | str, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Stream a file through SHA-256 using 4MB chunks to prevent memory bloat."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file(file_path: Path | str, expected_sha256: str) -> bool:
    """Return True only if file exists and its hash matches strictly."""
    p = Path(file_path)
    if not p.is_file():
        return False
    return compute_sha256(p).lower() == expected_sha256.strip().lower()