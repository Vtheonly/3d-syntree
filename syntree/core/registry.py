"""Centralized, immutable registry for 3D-SynTree V2 models, datasets, and HF repos."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

HF_ORG = "JJKK1212"


class CanonicalRepositories:
    """Single source of truth for Hugging Face Hub repositories."""
    CHECKPOINTS = f"{HF_ORG}/3d-syntree-checkpoints"
    MULTIDATASET = f"{HF_ORG}/3d-syntree-multidataset"
    RAW_SOURCES = f"{HF_ORG}/3d-syntree-raw-sources"
    ALL = (CHECKPOINTS, MULTIDATASET, RAW_SOURCES)


@dataclass(frozen=True)
class ModelSchema:
    name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_radial: int
    cutoff: float
    synthon_dim: int
    target_hardware: str


CANONICAL_MODELS: Dict[str, ModelSchema] = {
    "3d-syntree-base": ModelSchema(
        name="3d-syntree-base",
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_radial=20,
        cutoff=5.0,
        synthon_dim=128,
        target_hardware="Colab T4 (15 GB VRAM) / Portable Baseline / CPU",
    ),
    # V2 (#5 ARCH-01): the '3d-syntree-large' (V100/A10G) and
    # '3d-syntree-h100' schemas were removed - #45 §5 restricts supported
    # GPU targets to Google Colab Tesla T4 and Kaggle Tesla T4 only.
}


@dataclass(frozen=True)
class DatasetSchema:
    identifier: str
    catalog_filename: str
    manifest_relpath: str
    split_strategy: str
    description: str


CANONICAL_DATASETS: Dict[str, DatasetSchema] = {
    "cd2020-v1-clean": DatasetSchema(
        identifier="cd2020-v1-clean",
        catalog_filename="enamine_3d_subset.parquet",
        manifest_relpath="data/train/manifest.json",
        split_strategy="random_80_10_10",
        description="CrossDocked2020 pocket10 curated with 1.0A RMSD threshold",
    ),
    "multidataset-v1-30seq": DatasetSchema(
        identifier="multidataset-v1-30seq",
        catalog_filename="enamine_3d_subset.parquet",
        manifest_relpath="data/train/manifest.json",
        split_strategy="mmseqs2_30_percent_identity",
        description="Unified CrossDocked, PDBbind, and BindingMOAD clustered at 30% sequence identity",
    ),
    "multidataset-v2-full": DatasetSchema(
        identifier="multidataset-v2-full",
        catalog_filename="enamine_50k_subset.parquet",
        manifest_relpath="data/train/manifest.json",
        split_strategy="mmseqs2_30_percent_identity",
        description="Full multi-source structural corpus with expanded 50,000 synthon catalog",
    ),
}


class CanonicalFilenames:
    CKPT_STEP = "checkpoint_epoch_{epoch:03d}_step_{step:07d}.pt"
    CKPT_EPOCH = "checkpoint_epoch_{epoch:03d}.pt"
    CKPT_BEST = "checkpoint_best.pt"
    MODEL_MANIFEST = "manifest.json"
    MODEL_PROGRESS = "progress.json"
    DATA_SHARD = "{split}_shard_{shard_id:04d}.pt.gz"
    DATA_MANIFEST = "manifest.json"
    DATASET_STATE = "dataset_state.json"
    CATALOG = "enamine_3d_subset.parquet"

# ----------------------------------------------------------------------
# 5. SUPPORTED DATA BACKENDS
# ----------------------------------------------------------------------
# V2 (#5 ARCH-01): 'local' (CrossDockedDataset) and 'kaggle_offline'
# loaders were purged; the multi-source local pipeline is rebuilt by
# #16 DATA-01 / #19 DATA-04.
SUPPORTED_DATA_BACKENDS: Tuple[str, ...] = (
    "huggingface",
    "trajectory_pt",
)

def run_preflight_audit(config: dict, min_disk_gb: float = 1.0) -> Dict[str, bool]:
    results = {"model_valid": False, "dataset_valid": False, "disk_space_ok": False, "catalog_exists": False}
    model_name = config.get("canonical_model_name")
    if model_name:
        if model_name not in CANONICAL_MODELS:
            raise ValueError(f"Unknown canonical model {model_name!r}. Allowed: {list(CANONICAL_MODELS.keys())}")
        results["model_valid"] = True
    else:
        results["model_valid"] = True

    dataset_name = config.get("canonical_dataset_name")
    if dataset_name:
        if dataset_name not in CANONICAL_DATASETS:
            raise ValueError(f"Unknown canonical dataset {dataset_name!r}. Allowed: {list(CANONICAL_DATASETS.keys())}")
        results["dataset_valid"] = True
    else:
        results["dataset_valid"] = True

    free_bytes = shutil.disk_usage(os.getcwd()).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < min_disk_gb:
        raise OSError(f"Insufficient disk space ({free_gb:.2f} GB). Minimum: {min_disk_gb:.1f} GB.")
    results["disk_space_ok"] = True

    cat_path = config.get("data", {}).get("synthon_catalog_path")
    if cat_path and Path(cat_path).exists():
        results["catalog_exists"] = True
    return results