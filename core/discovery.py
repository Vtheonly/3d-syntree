"""Conservative, signature-based Kaggle Input dataset discovery."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

from core.manifest import ManifestManager

logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Locates and certifies input datasets under Kaggle/offline mounts."""

    @staticmethod
    def discover_raw_dataset(
        search_root: Path | str = "/kaggle/input",
        expected_sources: Tuple[str, ...] = ("crossdocked", "enamine_catalog"),
    ) -> Path:
        """Locate raw unprocessed dataset directory by manifest verification or content signature."""
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
                        declared = meta.get("files", {})
                        if all(k in declared for k in expected_sources):
                            candidates.append(Path(dirpath))
                except Exception as exc:
                    logger.debug("Skipping invalid manifest at %s: %s", manifest_path, exc)

        if not candidates:
            # Fallback signature search: check for raw archives directly
            for dirpath, _, filenames in os.walk(root):
                has_archive = any(f == "crossdocked_pocket10.tar.gz" for f in filenames)
                has_catalog = any(
                    ("enamine" in f.lower() or "building_block" in f.lower())
                    and f.endswith((".parquet", ".csv", ".csv.gz", ".sdf"))
                    for f in filenames
                )
                if has_archive and has_catalog:
                    candidates.append(Path(dirpath))

        if not candidates:
            raise FileNotFoundError(
                f"No verified raw dataset found in {root}. "
                "Ensure your raw-sources dataset is attached as a Kaggle Input."
            )
        return candidates[0]

    @staticmethod
    def discover_codebase(search_root: Path | str = "/kaggle/input") -> Path:
        """Locate the 3D-SynTree codebase root."""
        root = Path(search_root).resolve()
        candidates: List[Path] = []
        for dirpath, dirs, files in os.walk(root):
            if "main.py" in files and "syntree" in dirs:
                candidates.append(Path(dirpath))
        if not candidates:
            raise FileNotFoundError(f"3D-SynTree codebase not found under {root}. Attach your repository as a Kaggle Input.")
        return candidates[0]