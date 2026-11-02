"""Canonical compressed-shard I/O primitive (Suite A / #32; #19 foundation).

The unified data layer (issue #19 DATA-04) shards trajectory records into
byte-bounded ``.pt.gz`` shards with SHA-256 integrity.  This module is
the **one authoritative** shard file-format implementation (constitution
#45 §24.3 — one responsibility, one representation):

* :func:`write_shard` — serialize records with ``torch.save``, compress
  with **deterministic gzip** (``mtime=0``, level 9) so identical records
  produce byte-identical shards, and return the SHA-256 of the compressed
  bytes (the digest ``manifest.json`` records).
* :func:`shard_sha256` — the digest of an existing shard file.
* :func:`read_shard` — decompress, ``torch.load`` (``weights_only=False``
  is required for arbitrary record payloads; shards are produced by this
  project's own pipeline), and — when ``expected_sha256`` is given —
  verify the digest first, failing closed on any mismatch (corruption is
  never silently tolerated, #45 §15 "never trust a filename alone").

The ``ShardedDataset`` abstraction (byte-bounded packing, shuffle
sampling, HF streaming) builds on top of these primitives in
``syntree/data/sharded.py`` (#19) — it does not re-implement the format.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
from typing import Any, List, Sequence

import torch

__all__ = ["write_shard", "read_shard", "shard_sha256", "ShardIntegrityError"]


class ShardIntegrityError(ValueError):
    """Fail-closed shard corruption / mismatch error."""


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_sha256(path: str) -> str:
    """The SHA-256 hex digest of the shard file at ``path``."""
    return _sha256_of_file(path)


def write_shard(path: str, records: Sequence[Any]) -> str:
    """Serialize ``records`` into a deterministic ``.pt.gz`` shard.

    Returns the SHA-256 hex digest of the written file.  The gzip wrapper
    pins ``mtime=0`` (and no filename header) so that the same record
    payload always yields the same bytes — the reproducibility contract
    behind manifest digests.
    """
    buffer = io.BytesIO()
    torch.save(list(records), buffer)
    tmp = path + ".tmp"
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as gz:
            gz.write(buffer.getvalue())
    os.replace(tmp, path)
    return _sha256_of_file(path)


def read_shard(path: str, expected_sha256: str = None) -> List[Any]:
    """Read a ``.pt.gz`` shard back; verify its digest when given.

    Raises :class:`ShardIntegrityError` on digest mismatch or a missing
    file — never returns partially trusted data.
    """
    if not os.path.exists(path):
        raise ShardIntegrityError(f"shard not found: {path} (fail-closed)")
    if expected_sha256 is not None:
        actual = _sha256_of_file(path)
        if actual != expected_sha256:
            raise ShardIntegrityError(
                f"shard digest mismatch for {path}: expected {expected_sha256}, "
                f"got {actual} — the file is corrupt or truncated (fail-closed)"
            )
    with open(path, "rb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            payload = gz.read()
    records = torch.load(io.BytesIO(payload), weights_only=False)
    if not isinstance(records, list):
        raise ShardIntegrityError(
            f"shard {path} does not contain a record list (got "
            f"{type(records).__name__}) — unexpected format (fail-closed)"
        )
    return records
