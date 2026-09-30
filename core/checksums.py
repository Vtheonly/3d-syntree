"""High-throughput streaming checksum verification (SHA-256 & MD5)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(file_path: Path | str, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Stream a file through SHA-256 using 4MB chunks."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_md5(file_path: Path | str, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Stream a file through MD5."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {p}")
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file(file_path: Path | str, expected_hash: str) -> bool:
    """Return True if the file exists and matches expected hash (auto-detects MD5 vs SHA-256)."""
    p = Path(file_path)
    if not p.is_file():
        return False
    expected = expected_hash.strip().lower()
    if len(expected) == 32:
        return compute_md5(p).lower() == expected
    elif len(expected) == 64:
        return compute_sha256(p).lower() == expected
    return False