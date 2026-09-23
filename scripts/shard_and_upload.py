#!/usr/bin/env python3
"""Pack PyG samples into compressed byte-bounded shards and optionally upload."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Iterable, List

import torch
from huggingface_hub import HfApi


def serialize_samples(samples: List) -> bytes:
    buffer = io.BytesIO()
    torch.save(samples, buffer, _use_new_zipfile_serialization=True)
    return gzip.compress(buffer.getvalue(), compresslevel=6)


def _write_batch(batch: List, split: str, out_dir: Path, start: int,
                 shard_index: int, max_bytes: int, records: List[dict]) -> int:
    payload = serialize_samples(batch)
    if len(payload) > max_bytes and len(batch) > 1:
        mid = len(batch) // 2
        shard_index = _write_batch(
            batch[:mid], split, out_dir, start, shard_index, max_bytes, records
        )
        return _write_batch(
            batch[mid:], split, out_dir, start + mid, shard_index, max_bytes, records
        )

    name = f"{split}_shard_{shard_index:04d}.pt.gz"
    path = out_dir / name
    path.write_bytes(payload)
    records.append({
        "name": name,
        "sample_start": int(start),
        "sample_count": len(batch),
        "sample_end": int(start + len(batch)),
        "compressed_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "oversize_single_sample": bool(len(payload) > max_bytes),
    })
    return shard_index + 1


def write_shards(
    samples: Iterable,
    split: str,
    output_dir: str,
    max_shard_bytes: int = 500 * 1024 * 1024,
    coarse_batch_size: int = 256,
) -> dict:
    """Write shards without serializing the growing shard for every sample."""
    items = list(samples)
    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []

    shard_index = 0
    sample_start = 0
    current: List = []

    for i in range(0, len(items), int(coarse_batch_size)):
        chunk = items[i:i + int(coarse_batch_size)]
        candidate = current + chunk
        if current and len(serialize_samples(candidate)) > max_shard_bytes:
            shard_index = _write_batch(
                current, split, out_dir, sample_start, shard_index,
                max_shard_bytes, records
            )
            sample_start += len(current)
            current = list(chunk)
        else:
            current = candidate

    if current:
        _write_batch(
            current, split, out_dir, sample_start, shard_index,
            max_shard_bytes, records
        )

    manifest = {
        "format": "torch-pyg-list+gzip",
        "version": 1,
        "split": split,
        "total_samples": len(items),
        "max_shard_bytes": int(max_shard_bytes),
        "shards": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def upload_split(
    manifest: dict,
    split_dir: Path,
    repo_id: str,
    token: str,
    delete_after_upload: bool = False,
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    split = manifest["split"]
    for shard in manifest["shards"]:
        path = split_dir / shard["name"]
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"data/{split}/{path.name}",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        if delete_after_upload:
            path.unlink()
    api.upload_file(
        path_or_fileobj=str(split_dir / "manifest.json"),
        path_in_repo=f"data/{split}/manifest.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="./data/shards")
    parser.add_argument("--repo-id")
    parser.add_argument("--max-shard-gb", type=float, default=0.50)
    parser.add_argument("--delete-after-upload", action="store_true")
    args = parser.parse_args()

    samples = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(samples, list):
        raise TypeError("Input .pt must contain a list of PyG Data objects")

    manifest = write_shards(
        samples,
        args.split,
        args.output_dir,
        max_shard_bytes=int(args.max_shard_gb * 1024**3),
    )
    print(json.dumps(manifest, indent=2))

    if args.repo_id:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --repo-id is supplied")
        upload_split(
            manifest,
            Path(args.output_dir) / args.split,
            args.repo_id,
            token,
            delete_after_upload=args.delete_after_upload,
        )
        print(f"Uploaded split {args.split} to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
