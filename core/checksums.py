"""High-throughput streaming file digests and strict verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 4 * 1024 * 1024


def compute_sha256(file_path: Path | str, chunk_size: int = _CHUNK) -> str:
    """Stream a file through SHA-256 using 4MB chunks to prevent memory bloat."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_digest(
    file_path: Path | str, algo: str = "sha256", chunk_size: int = _CHUNK
) -> str:
    """Stream a file through ``algo`` (``sha256`` or ``md5``)."""
    algo = algo.lower()
    if algo not in ("sha256", "md5"):
        raise ValueError(f"Unsupported digest algorithm: {algo}")
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {p}")
    hasher = hashlib.new(algo)
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def pinned_hash_algo(pinned: str) -> str:
    """Infer the digest algorithm of a pinned hex value by its length.

    32 hex chars = MD5, 64 hex chars = SHA-256. Several legacy pins in this
    project are MD5-length despite being labelled "sha256"; failing to
    detect that would make every verification spuriously fail.
    """
    value = pinned.strip().lower()
    if len(value) == 32:
        return "md5"
    if len(value) == 64:
        return "sha256"
    raise ValueError(
        f"Pinned hash must be 32 (MD5) or 64 (SHA-256) hex chars, got {len(value)}"
    )


def verify_file(file_path: Path | str, expected_sha256: str) -> bool:
    """Return True only if file exists and its SHA-256 matches strictly."""
    p = Path(file_path)
    if not p.is_file():
        return False
    return compute_sha256(p).lower() == expected_sha256.strip().lower()


def verify_pinned(file_path: Path | str, pinned: str) -> bool:
    """Verify a file against a pinned digest of inferred length (MD5/SHA-256)."""
    p = Path(file_path)
    if not p.is_file():
        return False
    return compute_digest(p, pinned_hash_algo(pinned)).lower() == pinned.strip().lower()
