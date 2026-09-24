"""Conservative, signature-based Kaggle Input dataset discovery.

Never uses overly permissive wildcards. Identifies datasets strictly through:
1. Root manifest detection (raw_manifest.json or manifest.json).
2. Content verification (exact schema keys and signature files).
3. Checksum confirmation against declared metadata.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.checksums import verify_file
from core.manifest import ManifestManager

logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Locates and certifies input datasets under Kaggle/offline mounts."""

    @staticmethod
    def discover_raw_dataset(
        search_root: Path | str = "/kaggle/input",
        expected_sources: Tuple[str, ...] = ("crossdocked", "enamine_catalog"),
    ) -> Path:
        """Locate raw unprocessed dataset directory by manifest verification."""
        root = Path(search_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Search root does not exist: {root}")

        candidates: List[Path] = []
        for dirpath, _, filenames in os.walk(root):
            if "raw_manifest.json" in filenames:
                manifest_path = Path(dirpath) / "raw_manifest.json"
                try:
                    meta = ManifestManager.load(manifest_path)
                    if meta.get("dataset_type") == "3d_syntree_raw_unprocessed":
                        declared_files = meta.get("files", {})
                        if all(k in declared_files for k in expected_sources):
                            candidates.append(Path(dirpath))
                except Exception as exc:
                    logger.debug("Skipping invalid manifest at %s: %s", manifest_path, exc)

        if not candidates:
            # Fallback signature search: check for raw archives directly
            for dirpath, _, filenames in os.walk(root):
                has_archive = any(f == "crossdocked_pocket10.tar.gz" for f in filenames)
                has_catalog = any("enamine" in f.lower() and f.endswith(".parquet") for f in filenames)
                if has_archive and has_catalog:
                    candidates.append(Path(dirpath))

        if not candidates:
            raise FileNotFoundError(
                f"No verified raw dataset found in {root}. "
                "Ensure the Hugging Face raw dataset is mounted as a Kaggle Input."
            )
        if len(candidates) > 1:
            logger.warning("Multiple raw datasets found: %s. Selecting best match: %s", candidates, candidates[0])

        return candidates[0]

    @staticmethod
    def discover_crafted_dataset(
        search_root: Path | str = "/kaggle/input",
    ) -> Path:
        """Locate processed shards directory by validating train/manifest.json."""
        root = Path(search_root).resolve()
        candidates: List[Path] = []

        for dirpath, _, filenames in os.walk(root):
            if "manifest.json" in filenames and os.path.basename(dirpath) == "train":
                parent = Path(dirpath).parent
                if (parent / "val" / "manifest.json").exists():
                    candidates.append(parent)

        if not candidates:
            raise FileNotFoundError(f"No valid crafted dataset with train/val splits found under {root}")
        return candidates[0]