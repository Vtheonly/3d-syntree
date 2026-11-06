#!/usr/bin/env python3
"""Shard + upload driver (#19 DATA-04, issue-pinned script).

Packs the #18 trajectory outputs (``trajectories_{train,val,test}.pt``)
into byte-bounded ``.pt.gz`` shard directories (one per split, each with
its integrity ``manifest.json``) via the canonical sharding layer
(:mod:`syntree.data.sharded` — itself built on the #32 Suite A shard
format), and — when ``--upload`` is given — syncs the shard directories
to a Hugging Face dataset repository (``HfApi.upload_file``, token from
``HF_TOKEN`` per constitution #45 §15: credentials only via environment
or secure prompt, never committed or logged).

Fail-closed integrity: every shard's SHA-256 is verified against the
manifest at build time and after upload; ``--verify-only`` re-verifies a
shard directory (or a remote repo, downloading its manifest) without
uploading.

Example
-------
    python scripts/shard_and_upload.py \\
        --trajectories work/data/trajectories \\
        --out-dir work/data/shards

    python scripts/shard_and_upload.py \\
        --trajectories work/data/trajectories \\
        --out-dir work/data/shards \\
        --upload --repo-id JJKK1212/3d-syntree-v2-trajectories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from syntree.data.sharded import (  # noqa: E402
    DEFAULT_SHARD_BOUND_BYTES,
    ShardIntegrityError,
    ShardManifestError,
    build_shards,
    load_shard_manifest,
)
from syntree.data.shards import shard_sha256  # noqa: E402

SPLITS = ("train", "val", "test")


def log(message: str) -> None:
    print(f"[shard_and_upload] {message}", flush=True)


def _hf_token(args: argparse.Namespace) -> Optional[str]:
    """HF token from the environment only (never an argument, never logged)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set — credential handling per #45 §15 requires "
            "the environment; refusing to upload without it"
        )
    return token


def build(args: argparse.Namespace) -> int:
    if not os.path.isdir(args.trajectories):
        raise SystemExit(f"trajectories directory not found: {args.trajectories}")
    inputs = {}
    for split in SPLITS:
        path = os.path.join(args.trajectories, f"trajectories_{split}.pt")
        if not os.path.isfile(path):
            raise SystemExit(f"missing #18 output: {path} (fail-closed)")
        records = torch.load(path, weights_only=False)
        if not isinstance(records, list):
            raise SystemExit(f"{path} does not hold a record list (fail-closed)")
        inputs[split] = records
        log(f"{split}: {len(records)} trajectory records")

    bound = args.bound_bytes or DEFAULT_SHARD_BOUND_BYTES
    summary: Dict[str, object] = {
        "schema": "shard-and-upload-v1.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shard_bound_bytes": bound,
        "splits": {},
        "inputs": {
            "trajectories_dir": os.path.abspath(args.trajectories),
            "record_counts": {s: len(inputs[s]) for s in SPLITS},
        },
    }
    for split in SPLITS:
        out_dir = os.path.join(args.out_dir, split)
        source = {
            "kind": "syntree-trajectories",
            "split": split,
            "records": len(inputs[split]),
        }
        manifest = build_shards(
            inputs[split],
            out_dir,
            shard_bound_bytes=bound,
            prefix=f"traj-{split}",
            source=source,
        )
        total_bytes = sum(entry.bytes for entry in manifest.shards)
        summary["splits"][split] = {
            "shards": len(manifest.shards),
            "records": manifest.num_records,
            "bytes": total_bytes,
            "max_shard_bytes": max((e.bytes for e in manifest.shards), default=0),
            "digests": [entry.sha256 for entry in manifest.shards],
        }
        log(
            f"{split}: {len(manifest.shards)} shards, {manifest.num_records} records, "
            f"{total_bytes} bytes (max shard {max((e.bytes for e in manifest.shards), default=0)} B)"
        )

    summary_path = os.path.join(args.out_dir, "shard_run.json")
    tmp = summary_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, summary_path)
    log(f"summary: {summary_path}")

    if args.upload:
        upload(args)
    return 0


def _verify_dir(root: str) -> Dict[str, object]:
    manifest = load_shard_manifest(root)
    for entry in manifest.shards:
        path = os.path.join(root, entry.name)
        if not os.path.isfile(path):
            raise ShardIntegrityError(f"missing shard: {path} (fail-closed)")
        actual = shard_sha256(path)
        if actual != entry.sha256:
            raise ShardIntegrityError(
                f"digest mismatch for {path}: manifest {entry.sha256}, file {actual}"
            )
    return {
        "shards": len(manifest.shards),
        "records": manifest.num_records,
        "bound_bytes": manifest.shard_bound_bytes,
    }


def upload(args: argparse.Namespace) -> int:
    """Sync each split's shard directory to the HF dataset repo (private)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:  # pragma: no cover - environment-specific
        raise SystemExit("huggingface_hub is not installed (pip install 'syntree[hf]')")
    token = _hf_token(args)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    log(f"repo ready: {args.repo_id} (private)")

    for split in SPLITS:
        root = os.path.join(args.out_dir, split)
        info = _verify_dir(root)
        manifest_path = os.path.join(root, "manifest.json")
        api.upload_file(
            path_or_fileobj=manifest_path,
            path_in_repo=f"{split}/manifest.json",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        manifest = load_shard_manifest(root)
        for entry in manifest.shards:
            api.upload_file(
                path_or_fileobj=os.path.join(root, entry.name),
                path_in_repo=f"{split}/{entry.name}",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
        # verify the uploaded copy by digest of the re-downloaded file
        for entry in manifest.shards:
            remote = api.hf_hub_download(
                repo_id=args.repo_id,
                filename=f"{split}/{entry.name}",
                repo_type="dataset",
                token=token,
            )
            actual = shard_sha256(str(remote))
            if actual != entry.sha256:
                raise ShardIntegrityError(
                    f"post-upload digest mismatch for {split}/{entry.name}: "
                    f"manifest {entry.sha256}, remote {actual} (fail-closed)"
                )
        log(
            f"uploaded + verified {split}: {info['shards']} shards, "
            f"{info['records']} records, all digests match"
        )
    return 0


def verify_only(args: argparse.Namespace) -> int:
    for split in SPLITS:
        root = os.path.join(args.out_dir, split)
        info = _verify_dir(root)
        log(f"verified {split}: {info['shards']} shards, {info['records']} records, 0 mismatches")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", required=True, help="#18 output directory")
    parser.add_argument("--out-dir", required=True, help="shard output root")
    parser.add_argument(
        "--bound-bytes",
        type=int,
        default=None,
        help=f"shard byte bound (default {DEFAULT_SHARD_BOUND_BYTES} = 500 MB)",
    )
    parser.add_argument("--upload", action="store_true", help="sync shards to the HF repo")
    parser.add_argument("--repo-id", default=None, help="HF dataset repo (required with --upload)")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing shard directory")
    args = parser.parse_args(argv)

    if args.verify_only:
        return verify_only(args)
    if args.upload and not args.repo_id:
        raise SystemExit("--upload requires --repo-id (credential handling via HF_TOKEN env)")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
