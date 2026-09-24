#!/usr/bin/env python3
"""Download Mode: Resilient acquisition and preservation of raw, unmanipulated data.

Executes strictly in online cloud instances (Colab). Zero expensive chemistry.
Streams the raw CrossDocked2020 archive and the Enamine catalog with HTTP
resume (aria2c, curl fallback), verifies pinned digests (MD5 or SHA-256,
auto-detected by pin length), writes an immutable ``raw_manifest.json`` and
publishes the untouched payload to a Hugging Face dataset repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.checksums import compute_digest, pinned_hash_algo
from core.manifest import ManifestManager

RAW_SOURCES = {
    "crossdocked": {
        "url": "https://huggingface.co/datasets/Yukk1Zz/if3-crossdocked2020/resolve/main/crossdocked_pocket10.tar.gz",
        "expected_filename": "crossdocked_pocket10.tar.gz",
        # Legacy pin: 32 hex chars = MD5 (auto-detected, NOT sha256).
        "expected_sha256": "59416d06c2c366f6e05b91fdff8584f7",
        "size_mb": 1620,
    },
    "enamine_catalog": {
        "url": "https://huggingface.co/datasets/JJKK1212/3d-syntree-multidataset/resolve/main/enamine_3d_subset.parquet",
        "expected_filename": "enamine_3d_subset.parquet",
        "expected_sha256": None,
        "size_mb": 5,
    },
}

README_TEXT = """---
license: mit
task_categories:
- graph-ml
tags:
- biology
- chemistry
- raw-sbdd-sources
pretty_name: 3D-SynTree Raw Unprocessed Sources
size_categories:
- 10K<n<100K
---

# 3D-SynTree: Raw SBDD Source Repositories

This repository contains **unprocessed, raw structural biology archives**.
No chemical filtering, structural trimming, retrosynthetic decomposition, or
tensor conversion has been performed on these files.

### Contents:
- `crossdocked_pocket10.tar.gz`: Complete, unmodified CrossDocked2020 10A pocket complexes.
- `enamine_3d_subset.parquet`: Certified canonical Enamine building blocks catalog.
- `raw_manifest.json`: Pinned-digest signed provenance and integrity registry.

*Consumers: Import this raw dataset into Kaggle to run Craft Mode.*
"""


class RawDatasetAcquisition:
    """Manages resumable downloads, checksum auditing, and raw HF publishing."""

    def __init__(self, workspace_dir: Path | str, hf_repo_id: str, hf_token: str):
        from huggingface_hub import HfApi

        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def _verify(self, path: Path, pinned: str | None) -> bool:
        if not pinned:
            return path.is_file()  # unpinned source: existence is all we can check
        return (
            compute_digest(path, pinned_hash_algo(pinned)).lower()
            == pinned.strip().lower()
        )

    def _download_resume(self, url: str, dest: Path) -> None:
        """Resumable multi-connection download via aria2c, curl as fallback."""
        if shutil.which("aria2c"):
            cmd = [
                "aria2c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "-c",  # Enable HTTP range resume
                "-d", str(self.workspace_dir),
                "-o", dest.name,
                url,
            ]
        elif shutil.which("curl"):
            cmd = ["curl", "-L", "-C", "-", "--fail", "-o", str(dest), url]
        else:
            raise RuntimeError(
                "Neither aria2c nor curl is available; install aria2 "
                "(apt-get install -y aria2) and retry."
            )
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Download failed (exit {res.returncode}) from {url}")

    def download_source(self, key: str, config: dict) -> Path:
        dest_file = self.workspace_dir / config["expected_filename"]
        pinned = config.get("expected_sha256")
        algo = pinned_hash_algo(pinned) if pinned else None
        print(f"\n[Acquire] Downloading {key} -> {dest_file.name}...")

        if dest_file.exists():
            print(f"[Check] Existing file found: {dest_file.name}. Verifying...")
            if self._verify(dest_file, pinned):
                digest = compute_digest(dest_file, algo) if algo else "n/a"
                shown = digest[:12] if algo else "unpinned"
                print(f"[Verified] Existing file valid ({algo}: {shown}...). Skipping download.")
                return dest_file
            print(f"[Warning] Checksum mismatch ({algo} pin) or unpinned file; redownloading...")

        self._download_resume(config["url"], dest_file)

        if pinned:
            digest = compute_digest(dest_file, algo)
            if digest.lower() != pinned.strip().lower():
                raise IOError(
                    f"Corrupt download for {dest_file.name}: expected {algo} "
                    f"{pinned}, got {digest}"
                )
            print(f"[Success] {dest_file.name} verified ({algo}: {digest[:12]}...).")
        else:
            print(f"[Success] {dest_file.name} downloaded (no pinned digest to verify).")
        return dest_file

    # ------------------------------------------------------------------
    # Manifest + upload
    # ------------------------------------------------------------------
    def generate_raw_manifest(self, sources: dict | None = None) -> Path:
        print("\n[Manifest] Generating immutable raw provenance manifest...")
        records = {}
        for key, cfg in (sources or RAW_SOURCES).items():
            f_path = self.workspace_dir / cfg["expected_filename"]
            if not f_path.is_file():
                raise FileNotFoundError(
                    f"Cannot write manifest: {f_path.name} was never downloaded"
                )
            pinned = cfg.get("expected_sha256")
            algo = pinned_hash_algo(pinned) if pinned else "sha256"
            records[key] = {
                "filename": cfg["expected_filename"],
                "size_bytes": f_path.stat().st_size,
                "hash": compute_digest(f_path, algo),
                "hash_algo": algo,
                "source_url": cfg["url"],
            }

        manifest = {
            "dataset_type": "3d_syntree_raw_unprocessed",
            "version": 1,
            "created_at": time.time(),
            "files": records,
        }
        manifest_path = self.workspace_dir / "raw_manifest.json"
        ManifestManager.save_atomic(manifest, manifest_path)
        print(f"[Manifest] Saved -> {manifest_path}")
        return manifest_path

    def upload_raw_to_hf(self) -> None:
        from huggingface_hub import HfApi  # noqa: F811 (re-exported via api)

        print(f"\n[Upload] Publishing untouched raw dataset to HF: {self.hf_repo_id}...")
        self.api.create_repo(repo_id=self.hf_repo_id, repo_type="dataset", exist_ok=True)

        (self.workspace_dir / "README.md").write_text(README_TEXT, encoding="utf-8")

        self.api.upload_folder(
            folder_path=str(self.workspace_dir),
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            commit_message="Publish verified raw SBDD datasets and provenance manifest",
        )
        print(f"[Upload] Complete: https://huggingface.co/datasets/{self.hf_repo_id}")


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Mode: Raw SBDD Acquisition")
    parser.add_argument("--workspace", default=None, help="Local directory to stage raw files")
    parser.add_argument("--hf-repo", default=None, help="Target Hugging Face Dataset repository")
    parser.add_argument("--config", default=None, help="Optional YAML config (configs/download.yaml)")
    parser.add_argument(
        "--skip-upload", action="store_true", help="Download + manifest only, no HF publish"
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    workspace = args.workspace or cfg.get("workspace_dir") or "./raw_workspace"
    repo_id = args.hf_repo or cfg.get("hf_repo_id")
    if not repo_id and not args.skip_upload:
        parser.error("--hf-repo (or hf_repo_id in the config) is required")

    token = os.environ.get("HF_TOKEN")
    if not args.skip_upload and not token:
        print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr)
        return 1

    sources = RAW_SOURCES
    if cfg.get("sources"):
        # Config overlays the defaults (same logical keys).
        for key, override in cfg["sources"].items():
            base = sources.get(key, {})
            sources = dict(sources)  # shallow copy once is enough below
            sources[key] = {
                "url": override.get("url", base.get("url")),
                "expected_filename": override.get("filename", base.get("expected_filename")),
                "expected_sha256": override.get(
                    "sha256", base.get("expected_sha256")
                ),
                "size_mb": override.get("size_mb", base.get("size_mb")),
            }

    if args.skip_upload:
        # Local-only mode: no token, no HF client needed.
        acquisition = RawDatasetAcquisition.__new__(RawDatasetAcquisition)
        acquisition.workspace_dir = Path(workspace).resolve()
        acquisition.workspace_dir.mkdir(parents=True, exist_ok=True)
        acquisition.hf_repo_id = repo_id or "(local-only)"
        acquisition.hf_token = ""
        acquisition.api = None
    else:
        acquisition = RawDatasetAcquisition(workspace, repo_id, token)

    for key, src in sources.items():
        acquisition.download_source(key, src)

    acquisition.generate_raw_manifest(sources)
    if not args.skip_upload:
        acquisition.upload_raw_to_hf()
    print("\n[Done] Download Mode finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
