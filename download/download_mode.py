#!/usr/bin/env python3
"""Download Mode: Resilient acquisition and preservation of raw, unmanipulated data.

Executes strictly in online cloud instances (Colab). Zero expensive chemistry.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.checksums import compute_sha256, verify_file
from core.manifest import ManifestManager

RAW_SOURCES = {
    "crossdocked": {
        "url": "https://huggingface.co/datasets/Yukk1Zz/if3-crossdocked2020/resolve/main/crossdocked_pocket10.tar.gz",
        "expected_filename": "crossdocked_pocket10.tar.gz",
        "expected_hash": "59416d06c2c366f6e05b91fdff8584f7",  # Zenodo MD5
    },
    "enamine_catalog": {
        "url": "https://huggingface.co/datasets/JJKK1212/3d-syntree-multidataset/resolve/main/enamine_real_vendor_export.csv.gz",
        "fallback_url": "https://huggingface.co/datasets/JJKK1212/3d-syntree-multidataset/resolve/main/enamine_3d_subset.parquet",
        "expected_filename": "enamine_real_vendor_export.csv.gz",
        "expected_hash": None,
    },
}


class RawDatasetAcquisition:
    def __init__(self, workspace_dir: Path | str, hf_repo_id: str, hf_token: str):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)

    def download_source(self, key: str, config: dict) -> Path:
        dest_file = self.workspace_dir / config["expected_filename"]
        print(f"\n[Acquire] Downloading {key} -> {dest_file.name}...")

        if dest_file.exists() and dest_file.stat().st_size > 1000:
            if config.get("expected_hash") and verify_file(dest_file, config["expected_hash"]):
                print(f"[Verified] Existing file matches checksum. Skipping download.")
                return dest_file

        url = config["url"]
        cmd = [
            "aria2c", "-x", "16", "-s", "16", "-k", "1M", "-c",
            "-d", str(self.workspace_dir),
            "-o", config["expected_filename"],
            url,
        ]
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0 or not dest_file.exists() or dest_file.stat().st_size == 0:
            if "fallback_url" in config:
                print(f"[Fallback] Primary URL failed; attempting fallback: {config['fallback_url']}")
                fallback_name = Path(config["fallback_url"]).name
                dest_file = self.workspace_dir / fallback_name
                config["expected_filename"] = fallback_name
                cmd[-2] = fallback_name
                cmd[-1] = config["fallback_url"]
                res = subprocess.run(cmd, check=True)
            else:
                raise RuntimeError(f"Download failed for {key} from {url}")

        if config.get("expected_hash") and not verify_file(dest_file, config["expected_hash"]):
            raise IOError(f"Checksum mismatch for {dest_file.name}")

        sha = compute_sha256(dest_file)
        print(f"[Success] {dest_file.name} acquired (SHA-256: {sha[:12]}...).")
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
            "version": 3,
            "created_at": time.time(),
            "files": records,
        }
        manifest_path = self.workspace_dir / "raw_manifest.json"
        ManifestManager.save_atomic(manifest, manifest_path)
        print(f"[Manifest] Saved -> {manifest_path}")
        return manifest_path

    def upload_raw_to_hf(self) -> None:
        print(f"\n[Upload] Publishing raw dataset to Hugging Face: {self.hf_repo_id}...")
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

# 3D-SynTree: Raw Unprocessed SBDD Sources

Contains untouched structural biology archives and authentic commercial building block stock.
Zero in silico crafting or retrosynthetic manipulation has been performed on these files.
"""
        (self.workspace_dir / "README.md").write_text(readme_text, encoding="utf-8")
        self.api.upload_folder(
            folder_path=str(self.workspace_dir),
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            commit_message="Publish raw CrossDocked archive, vendor inventory, and raw manifest",
        )
        print(f"[Upload] Complete: https://huggingface.co/datasets/{self.hf_repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Mode: Raw SBDD Acquisition")
    parser.add_argument("--workspace", default="./raw_workspace")
    parser.add_argument("--hf-repo", required=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())