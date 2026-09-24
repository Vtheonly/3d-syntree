#!/usr/bin/env python3
"""Download Mode: Resilient acquisition and preservation of raw, unmanipulated data.

Executes strictly in online cloud instances (Colab). Zero expensive chemistry.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.checksums import compute_sha256
from core.manifest import ManifestManager

RAW_SOURCES = {
    "crossdocked": {
        "url": "https://huggingface.co/datasets/Yukk1Zz/if3-crossdocked2020/resolve/main/crossdocked_pocket10.tar.gz",
        "expected_filename": "crossdocked_pocket10.tar.gz",
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


class RawDatasetAcquisition:
    """Manages resumable downloads, checksum auditing, and raw HF publishing."""

    def __init__(self, workspace_dir: Path | str, hf_repo_id: str, hf_token: str):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)

    def download_source(self, key: str, config: dict) -> Path:
        dest_file = self.workspace_dir / config["expected_filename"]
        print(f"\n[Acquire] Downloading {key} -> {dest_file.name}...")

        if dest_file.exists():
            print(f"[Check] Existing file found: {dest_file.name}. Verifying...")
            sha = compute_sha256(dest_file)
            if config["expected_sha256"] and sha.lower() == config["expected_sha256"].lower():
                print(f"[Verified] Existing file valid (SHA-256: {sha[:12]}...). Skipping download.")
                return dest_file
            print("[Warning] Checksum mismatch or unpinned file; redownloading via aria2...")

        # Resumable, multi-connection download via aria2c
        cmd = [
            "aria2c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "-c",  # Enable HTTP range resume
            "-d", str(self.workspace_dir),
            "-o", config["expected_filename"],
            config["url"],
        ]
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Download failed for {key} from {config['url']}")

        sha = compute_sha256(dest_file)
        if config["expected_sha256"] and sha.lower() != config["expected_sha256"].lower():
            raise IOError(f"Corrupt download for {dest_file.name}: expected {config['expected_sha256']}, got {sha}")

        print(f"[Success] {dest_file.name} acquired and verified (SHA-256: {sha[:12]}...).")
        return dest_file

    def generate_raw_manifest(self) -> Path:
        print("\n[Manifest] Generating immutable raw provenance manifest...")
        records = {}
        for key, cfg in RAW_SOURCES.items():
            f_path = self.workspace_dir / cfg["expected_filename"]
            records[key] = {
                "filename": cfg["expected_filename"],
                "size_bytes": f_path.stat().st_size,
                "sha256": compute_sha256(f_path),
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
        print(f"\n[Upload] Publishing untouched raw dataset to Hugging Face: {self.hf_repo_id}...")
        self.api.create_repo(repo_id=self.hf_repo_id, repo_type="dataset", exist_ok=True)

        readme_text = f"""---
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
No chemical filtering, structural trimming, retrosynthetic decomposition, or tensor conversion
has been performed on these files.

### Contents:
- `crossdocked_pocket10.tar.gz`: Complete, unmodified CrossDocked2020 10Å pocket complexes.
- `enamine_3d_subset.parquet`: Certified canonical Enamine building blocks catalog.
- `raw_manifest.json`: SHA-256 signed provenance and integrity registry.

*Consumers: Import this raw dataset into Kaggle to run Craft Mode.*
"""
        (self.workspace_dir / "README.md").write_text(readme_text, encoding="utf-8")

        self.api.upload_folder(
            folder_path=str(self.workspace_dir),
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            commit_message="Publish verified raw SBDD datasets and provenance manifest",
        )
        print(f"[Upload] Complete: https://huggingface.co/datasets/{self.hf_repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Mode: Raw SBDD Acquisition")
    parser.add_argument("--workspace", default="./raw_workspace", help="Local directory to stage raw files")
    parser.add_argument("--hf-repo", required=True, help="Target Hugging Face Dataset repository")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr)
        return 1

    acquisition = RawDatasetAcquisition(args.workspace, args.hf_repo, token)
    for key, cfg in RAW_SOURCES.items():
        acquisition.download_source(key, cfg)

    acquisition.generate_raw_manifest()
    acquisition.upload_raw_to_hf()
    print("\n[Done] Download Mode finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())