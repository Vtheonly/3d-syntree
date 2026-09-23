Go to this repository first: https://github.com/Vtheonly/3d-syntree

Then solve **all the issues listed in this bug report**. Make sure to do **super, super, super extensive testing**, including comprehensive unit tests and integration tests. I want you to thoroughly verify everything and make sure the fixes do not introduce any new issues.

""""""""
To transition to **full-scale production mode**, we must address three essential components:

1. **Fail-Closed Configuration & Guards**: Prevent the system from ever silently training on a 2-sample stub or falling back to synthetic data.
2. **Automated End-to-End Dataset Builder**: Download the full CrossDocked2020 dataset, extract 10 Å pockets, validate retrosynthetic trajectories via RDKit, shard into compressed byte-bounded chunks, and push thousands of real samples to `JJKK1212/3d-syntree-multidataset`.
3. **Colab Execution & Training Profile**: Stream the full dataset directly from Hugging Face for multi-hour training with automatic GPU scaling, resilient atomic checkpointing, and no data leaks.

---

### File 1: `configs/train_colab_12h.json` (Full Production Configuration)

Replace the entire contents of `configs/train_colab_12h.json` with:

```json
{
  "system": {
    "project_name": "3D-SynTree",
    "seed": 42,
    "device": "auto",
    "mixed_precision": "fp16",
    "tf32": true,
    "num_workers": 2
  },
  "huggingface": {
    "enabled": true,
    "repo_id": "JJKK1212/3d-syntree-checkpoints",
    "push_every_n_epochs": 2,
    "private": false
  },
  "data": {
    "backend": "huggingface",
    "dataset_name": "unified_multidataset",
    "data_dir": "./data/crossdocked",
    "synthon_catalog_path": "./data/enamine_3d_subset.parquet",
    "require_real_data": true,
    "min_real_samples": 50,
    "synthetic_fallback": false,
    "synthetic_samples": 0,
    "huggingface": {
      "repo_id": "JJKK1212/3d-syntree-multidataset",
      "revision": "main",
      "cache_dir": "./hf_cache",
      "max_cached_shards": 4
    },
    "batch_size": 32,
    "accumulate_grad_batches": 2,
    "max_steps_per_molecule": 3,
    "terminal_cap_min_mw": 250.0,
    "val_fraction": 0.1,
    "trajectory_dataset_path": null,
    "multidataset": {
      "processed_manifest_path": "./data/multidataset/processed_manifest.jsonl",
      "split_manifest_path": "./data/multidataset/split_manifest.json",
      "max_resolution": 2.5,
      "pocket_radius_angstrom": 10.0,
      "min_pocket_atoms": 100
    }
  },
  "model": {
    "hidden_dim": 256,
    "num_equivariant_layers": 8,
    "num_radial_basis": 20,
    "cutoff_radius": 5.0,
    "synthon_embedding_dim": 256,
    "num_attention_heads": 8,
    "max_atomic_number": 100,
    "dropout": 0.1
  },
  "training": {
    "max_epochs": 40,
    "time_budget_hours": 11.5,
    "learning_rate": 0.0003,
    "weight_decay": 0.00001,
    "lr_scheduler": "cosine_warmup",
    "warmup_epochs": 2,
    "grad_clip_norm": 1.0,
    "keep_last_n_checkpoints": 5,
    "loss_weights": {
      "synthon_ce": 1.0,
      "torsion_nll": 0.5,
      "steric_clash": 0.1,
      "reaction_ce": 0.5
    },
    "eval_interval_epochs": 1,
    "auto_scale": {
      "enabled": true,
      "target_vram_fraction": 0.85,
      "max_batch_size": 2048,
      "scale_model": true,
      "lr_scale_rule": "linear"
    }
  },
  "reinforcement_learning": {
    "enabled": false,
    "pocket_dir": "./data/test_pockets",
    "episodes": 256,
    "ppo_epochs": 4,
    "clip_epsilon": 0.2,
    "gamma": 0.99,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "learning_rate": 0.00001,
    "max_grad_norm": 1.0,
    "temperature": 1.0,
    "checkpoint_every": 16,
    "reward": {
      "docking_weight": 1.0,
      "clash_weight": 0.25,
      "contact_weight": 0.25,
      "fsp3_weight": 0.15,
      "qed_weight": 0.15,
      "validity_weight": 0.5
    }
  },
  "evaluation": {
    "num_test_pockets": 100,
    "docking_engine": "gnina",
    "exhaustiveness": 8,
    "run_posebusters": true,
    "run_aizynthfinder": false,
    "aizynthfinder_config": null
  },
  "catalog": {
    "min_fsp3": 0.40,
    "max_mw": 220.0
  }
}
```

---

#### File 2: `scripts/build_full_dataset.py` (New Automated Builder & Uploader)

Create `scripts/build_full_dataset.py` in your repository. This script automates downloading the real CrossDocked2020 dataset, extracting pockets, decomposing ligands with verified forward RDKit replay, partitioning into splits, compressing into shards, and uploading thousands of samples to `JJKK1212/3d-syntree-multidataset`:

```python
#!/usr/bin/env python3
"""Build and upload the complete real production dataset to Hugging Face.

Workflow:
  1. Download full CrossDocked2020 (Zenodo ~4 GB compressed) if not present locally.
  2. Extract 10 Å pockets and sanitize ligands.
  3. Extract reaction-validated retrosynthetic trajectories against the Enamine catalog.
  4. Assign cluster-safe train/val/test splits (80/10/10).
  5. Package into gzip-compressed shards (~500 MB max) with SHA-256 manifests.
  6. Stage target pockets for Stage 2 RL docking evaluation.
  7. Upload clean shards, manifests, exact catalog, and targets to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import torch
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.fragmenter import ReactionConstrainedFragmenter
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder
from scripts.shard_and_upload import write_shards, upload_split

ZENODO_CROSSDOCKED_URL = (
    "https://zenodo.org/records/6458305/files/crossdocked_pocket10.tar.gz"
)


def download_and_extract_crossdocked(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "crossdocked_pocket10.tar.gz"
    extracted_marker = raw_dir / ".extracted"

    if not extracted_marker.exists():
        if not archive_path.exists():
            print(f"[download] Fetching CrossDocked2020 from {ZENODO_CROSSDOCKED_URL}...")
            urllib.request.urlretrieve(ZENODO_CROSSDOCKED_URL, archive_path)
            print(f"[download] Archive downloaded: {archive_path}")

        print(f"[extract] Extracting {archive_path.name} into {raw_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)
        extracted_marker.touch()
        print("[extract] Extraction complete.")
    else:
        print(f"[extract] Found existing extracted dataset in {raw_dir}")

    return raw_dir


def discover_pairs(data_dir: Path) -> List[tuple[Path, Path, str]]:
    pairs = []
    # Search for <name>_pocket10.pdb (or <name>_pocket.pdb) and matching <name>.sdf / <name>_ligand.sdf
    for p in data_dir.rglob("*.pdb"):
        stem = p.stem
        if "pocket" not in stem.lower():
            continue
        # Find companion SDF in same folder
        parent = p.parent
        sdfs = list(parent.glob("*.sdf"))
        if not sdfs:
            continue
        # Prefer SDF with matching prefix or largest file
        companion = sdfs[0]
        for s in sdfs:
            if s.stem.startswith(stem.split("_")[0]):
                companion = s
                break
        cid = f"{parent.name}_{stem}"
        pairs.append((p, companion, cid))

    return sorted(pairs, key=lambda x: x[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="./raw_data/crossdocked")
    parser.add_argument("--catalog", default="./data/enamine_3d_subset.parquet")
    parser.add_argument("--output-dir", default="./data/full_shards")
    parser.add_argument("--repo-id", default="JJKK1212/3d-syntree-multidataset")
    parser.add_argument("--max-complexes", type=int, default=None,
                        help="Optional limit for dry runs; None processes the full dataset")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--upload", action="store_true",
                        help="Upload directly to Hugging Face Hub (requires HF_TOKEN)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if args.upload and not token:
        print("ERROR: --upload requires HF_TOKEN environment variable", file=sys.stderr)
        return 1

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"[catalog] Generating canonical 3D catalog: {catalog_path}...")
        from scripts.download_assets import build_synthetic_catalog
        build_synthetic_catalog(str(catalog_path.parent), num_copies=40)

    catalog = SynthonCatalog(str(catalog_path))
    builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=args.max_steps)

    raw_path = download_and_extract_crossdocked(Path(args.raw_dir))
    all_pairs = discover_pairs(raw_path)
    if not all_pairs:
        print(f"ERROR: No pocket/ligand pairs discovered in {raw_path}", file=sys.stderr)
        return 1

    if args.max_complexes:
        all_pairs = all_pairs[:args.max_complexes]

    print(f"[process] Discovered {len(all_pairs)} complexes. Extracting reaction trajectories...")

    rng = np.random.default_rng(42)
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)

    # 80% train, 10% val, 10% test
    n_train = int(len(all_pairs) * 0.80)
    n_val = int(len(all_pairs) * 0.10)

    split_map = {}
    for idx, i in enumerate(indices):
        split = "train" if idx < n_train else "val" if idx < (n_train + n_val) else "test"
        split_map[all_pairs[i][2]] = split

    states_by_split = {"train": [], "val": [], "test": []}
    target_pockets: List[Path] = []
    accepted_complexes = 0

    for idx, (pocket_path, ligand_path, cid) in enumerate(all_pairs):
        split = split_map[cid]
        try:
            pocket_mol = Chem.MolFromPDBFile(str(pocket_path), removeHs=False)
            suppl = Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=True)
            ligand_mol = next((m for m in suppl if m is not None), None)
            if pocket_mol is None or ligand_mol is None or ligand_mol.GetNumConformers() == 0:
                continue

            states, meta = builder.build(ligand_mol, pocket_mol, trajectory_id=cid)
            if states:
                states_by_split[split].extend(states)
                accepted_complexes += 1
                if split in ("train", "val") and len(target_pockets) < 30:
                    target_pockets.append(pocket_path)

        except Exception as exc:
            continue

        if (idx + 1) % 500 == 0 or idx == len(all_pairs) - 1:
            print(
                f"  [{idx + 1}/{len(all_pairs)}] Accepted complexes: {accepted_complexes} | "
                f"Train states: {len(states_by_split['train'])} | "
                f"Val: {len(states_by_split['val'])} | Test: {len(states_by_split['test'])}"
            )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for split in ("train", "val", "test"):
        samples = states_by_split[split]
        print(f"[shard] Writing {len(samples)} samples for split '{split}'...")
        manifest = write_shards(
            samples,
            split,
            str(out_root),
            max_shard_bytes=500 * 1024 * 1024,
        )
        manifests[split] = manifest

    # Stage RL evaluation targets
    targets_dir = out_root / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    for p in target_pockets:
        shutil.copy2(p, targets_dir / f"{p.stem}_pocket.pdb")

    # Copy catalog to output root
    shutil.copy2(catalog_path, out_root / "enamine_3d_subset.parquet")

    summary = {
        "status": "success",
        "total_complexes_evaluated": len(all_pairs),
        "accepted_complexes": accepted_complexes,
        "splits": {s: len(states_by_split[s]) for s in states_by_split},
        "target_pockets_staged": len(target_pockets),
    }
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print(json.dumps(summary, indent=2))
    print("=" * 60)

    if args.upload:
        print(f"[upload] Uploading complete real dataset to Hugging Face: {args.repo_id}...")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

        # Upload catalog and targets
        api.upload_file(
            path_or_fileobj=str(out_root / "enamine_3d_subset.parquet"),
            path_in_repo="enamine_3d_subset.parquet",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        for tp in targets_dir.glob("*.pdb"):
            api.upload_file(
                path_or_fileobj=str(tp),
                path_in_repo=f"targets/{tp.name}",
                repo_id=args.repo_id,
                repo_type="dataset",
            )

        # Upload shards and manifests for each split
        for split in ("train", "val", "test"):
            upload_split(
                manifests[split],
                out_root / split,
                args.repo_id,
                token=token,
            )

        print(f"[upload] Successfully uploaded complete real dataset to {args.repo_id}!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

#### File 3: `syntree/engine/trainer.py` (Enforce Real Dataset Guard)

In `syntree/engine/trainer.py`, update `ResilientTrainer.__init__` around lines 90–120 so it **fails closed** if real data is requested but unavailable or smaller than the required threshold:

```python
        # Data loading backend setup
        val_fraction = float(data_cfg.get("val_fraction", 0.1))
        trajectory_path = data_cfg.get("trajectory_dataset_path")
        self.data_backend = str(data_cfg.get("backend", "local")).lower()

        if self.data_backend == "huggingface":
            hf_cfg = dict(data_cfg.get("huggingface", {}))
            repo_id = str(hf_cfg.get("repo_id", "")).strip()
            if not repo_id:
                raise ValueError(
                    "data.huggingface.repo_id is required when data.backend='huggingface'"
                )
            token = os.environ.get("HF_TOKEN") or hf_cfg.get("token")
            self.dataset = ShardedHuggingFaceDataset(
                repo_id=repo_id,
                split="train",
                cache_dir=str(hf_cfg.get("cache_dir", "./hf_cache")),
                revision=str(hf_cfg.get("revision", "main")),
                token=token,
                max_cached_shards=int(hf_cfg.get("max_cached_shards", 4)),
            )
            self.val_dataset = ShardedHuggingFaceDataset(
                repo_id=repo_id,
                split="val",
                cache_dir=str(hf_cfg.get("cache_dir", "./hf_cache")),
                revision=str(hf_cfg.get("revision", "main")),
                token=token,
                max_cached_shards=int(hf_cfg.get("max_cached_shards", 4)),
            )
        elif trajectory_path:
            self.data_backend = "trajectory_pt"
            self.dataset = TrajectoryDataset(trajectory_path, split="train")
            self.val_dataset = TrajectoryDataset(trajectory_path, split="val")
        else:
            self.dataset = CrossDockedDataset(
                data_cfg["data_dir"],
                split="train",
                catalog=self.catalog,
                num_synthetic=int(data_cfg.get("synthetic_samples", 100)),
                synthetic_fallback=bool(data_cfg.get("synthetic_fallback", False)),
                split_manifest_path=data_cfg.get("split_manifest_path"),
            )
            self.val_dataset = CrossDockedDataset(
                data_cfg["data_dir"],
                split="val",
                catalog=self.catalog,
                num_synthetic=max(8, int(self.dataset.num_synthetic * val_fraction)),
                synthetic_fallback=bool(data_cfg.get("synthetic_fallback", False)),
                split_manifest_path=data_cfg.get("split_manifest_path"),
            )

        # STRICT REAL-DATA GUARD: Reject toy stubs and synthetic fallbacks in production
        require_real = bool(data_cfg.get("require_real_data", True))
        min_samples = int(data_cfg.get("min_real_samples", 50))
        if require_real and len(self.dataset) < min_samples:
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"FATAL: Production mode requires a real dataset, but the dataset in\n"
                f"'{self.data_backend}' contains only {len(self.dataset)} training samples (min required: {min_samples}).\n\n"
                f"Training will NOT proceed on a stub or placeholder dataset.\n"
                f"To build and upload the complete real dataset, run:\n"
                f"  python scripts/build_full_dataset.py --upload\n"
                f"{'='*70}\n"
            )
```

---

#### File 4: `scripts/download_assets.py` (Validation Gate on Asset Sync)

In `scripts/download_assets.py`, update `sync_hf_dataset` around lines 270–305 so asset synchronization validates that the training split is genuine:

```python
    split_stats = {}
    for split, manifest_file in required_manifests.items():
        local_manifest = Path(hf_hub_download(
            repo_id=repo_id,
            filename=manifest_file,
            repo_type="dataset",
            revision=revision,
            token=token,
            local_dir=str(output_root / "_hf_manifests"),
        ))
        manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        shards = manifest.get("shards", [])
        total = int(manifest.get("total_samples", -1))
        declared = sum(int(s["sample_count"]) for s in shards)
        if total != declared or not shards:
            raise RuntimeError(
                f"Invalid {split} manifest in {repo_id}: "
                f"total_samples={total}, shard_count={len(shards)}, declared={declared}"
            )
        split_stats[split] = {
            "total_samples": total,
            "shard_count": len(shards),
            "max_shard_bytes": manifest.get("max_shard_bytes"),
            "manifest": manifest_file,
        }

    # Verify that the dataset is not a 2-sample stub
    train_count = split_stats.get("train", {}).get("total_samples", 0)
    print(f"[assets] Dataset split verified: train={train_count} samples, val={split_stats['val']['total_samples']} samples")
```

---

#### File 5: `run_3d_syntree.ipynb` (Colab Notebook Workflow)

Update **Cell 5** in `run_3d_syntree.ipynb` so it checks the sample count and prompts the user if the dataset has not been populated with full shards yet:

```python
# CELL 5: Sync Clean Sharded Dataset & Synthon Catalog from HF Dataset Hub
DATASET_REPO = RUNTIME_CONFIG["hf_dataset_repo_id"]
print(f"Synchronizing preprocessed assets from HF Dataset: {DATASET_REPO}...")

!python scripts/download_assets.py \
    --dataset-repo {DATASET_REPO} \
    --dataset-revision main \
    --output-dir ./data

import json
with open("./data/assets_manifest.json") as f:
    meta = json.load(f)

train_samples = meta["splits"]["train"]["total_samples"]
print(f"\nAsset Sync Complete: {train_samples} training samples available from {DATASET_REPO}.")

if train_samples < 50:
    print(
        f"\nWARNING: {DATASET_REPO} contains only {train_samples} samples (stub dataset).\n"
        "To run full multi-hour production training, build and upload the real dataset using:\n"
        "  !python scripts/build_full_dataset.py --upload\n"
    )
```

And add an optional **Dataset Build Cell** directly in the notebook before Cell 5 if you wish to run the full dataset creation directly inside Google Colab:

```python
# OPTIONAL CELL: Build and Upload Complete Real Dataset (Run Once)
# Uncomment the line below to download real CrossDocked from Zenodo, extract pockets,
# generate reaction trajectories, and upload all shards to Hugging Face:

# !python scripts/build_full_dataset.py --upload
```

---

### Step-by-Step Instructions to Run Full-Scale Training

1. **Copy the updated files into your repository**:
   - `configs/train_colab_12h.json`
   - `scripts/build_full_dataset.py`
   - `syntree/engine/trainer.py`
   - `scripts/download_assets.py`
   - `run_3d_syntree.ipynb`

2. **Generate and Upload the Complete Real Dataset (One-Time Execution)**:
   You can run this on your local machine, an HPC cluster, or in a Colab terminal:
   ```bash
   export HF_TOKEN="your_huggingface_write_token"
   python scripts/build_full_dataset.py --upload
   ```
   *This downloads the real CrossDocked complexes, extracts valid 3D trajectories, and uploads complete compressed `.pt.gz` shards to `JJKK1212/3d-syntree-multidataset`.*

3. **Launch Production Training in Colab**:
   - In `run_3d_syntree.ipynb`, run **Cell 1 through Cell 6**.
   - Cell 5 will sync the catalog and verify thousands of real training samples.
   - In **Cell 6.5**, set `FRESH_TRAINING = True` for the first run on the new dataset.
   - Run **Cell 7**.
   
4. **Result**:
   The trainer will log:
   ```text
   [trainer] starting run: budget=11.50h, epochs=40, synthons=85, train=15420, val=1920
   [trainer] model: hidden_dim=256, layers=8, heads=8 | batch=32 x accum=2 | steps/epoch=240
   ```
   Each epoch will process 240 real batches, running continuously for several hours on the T4 GPU and checkpointing to `JJKK1212/3d-syntree-checkpoints` without falling back to synthetic or stub data.



   ### Why It Is Still Training in 1 Minute

Look at lines 17 and 18 of the log:

```text
[trainer] starting run: budget=11.50h, epochs=40, synthons=85, train=2, val=2, resume_epoch=0
[trainer] model: hidden_dim=256, layers=8, heads=8 | batch=2 x accum=16 | steps/epoch=1 | amp=True
```

Notice **`train=2, val=2`** and **`steps/epoch=1`**:
- Your Hugging Face repository `JJKK1212/3d-syntree-multidataset` **still contains only 2 training molecules**.
- Because there are only 2 molecules in the entire dataset, **1 epoch is only 1 step**.
- 40 epochs took **20 seconds** to train.
- And then the script spent the next **8 minutes** uploading 20 checkpoints (70 MB each = 1.4 GB total) to Hugging Face!

It is **impossible** for any machine learning model to train for hours when the dataset only has 2 molecules.

---

### The Solution

To train for hours on real data, you must **download the full CrossDocked2020 dataset (22,500 complexes), process them, and upload the full shards to `JJKK1212/3d-syntree-multidataset`**.

Below are the **four exact files** you need:
1. **`scripts/build_full_dataset.py`**: A complete, automated script that downloads CrossDocked2020 directly from Zenodo, extracts pockets, validates retrosynthetic trajectories with RDKit, shards them into 500 MB `.pt.gz` chunks, and uploads thousands of real samples to `JJKK1212/3d-syntree-multidataset`.
2. **`syntree/engine/trainer.py`**: Updated with a fail-closed guard that **refuses** to train if the dataset is a 2-sample stub.
3. **`configs/train_colab_12h.json`**: Production configuration with `require_real_data: true`.
4. **`run_3d_syntree.ipynb`**: Updated notebook with **Cell 4.5** so you can run the full real dataset build and upload directly in Google Colab.

---

### Complete File 1: `scripts/build_full_dataset.py`

Create `scripts/build_full_dataset.py` in your repository:

```python
#!/usr/bin/env python3
"""Automated pipeline: download full real CrossDocked2020, preprocess, and upload to Hugging Face.

Workflow:
  1. Download CrossDocked2020 (~4 GB tar.gz from Zenodo).
  2. Extract pocket and ligand pairs.
  3. Extract reaction-validated retrosynthetic trajectories against the Enamine catalog.
  4. Partition into train/val/test splits (80/10/10).
  5. Package into 500 MB compressed shards with SHA-256 manifests.
  6. Stage target pocket PDBs for Stage 2 RL docking evaluation.
  7. Upload clean shards, manifests, exact catalog, and targets to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder
from scripts.shard_and_upload import write_shards, upload_split

ZENODO_CROSSDOCKED_URL = (
    "https://zenodo.org/records/6458305/files/crossdocked_pocket10.tar.gz"
)


def download_and_extract(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "crossdocked_pocket10.tar.gz"
    marker = raw_dir / ".extracted_marker"

    if not marker.exists():
        if not archive_path.exists():
            print(f"[download] Fetching real CrossDocked2020 from {ZENODO_CROSSDOCKED_URL}...")
            urllib.request.urlretrieve(ZENODO_CROSSDOCKED_URL, archive_path)
            print(f"[download] Archive downloaded ({archive_path.stat().st_size / 1e9:.2f} GB)")

        print(f"[extract] Extracting {archive_path.name} into {raw_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)
        marker.touch()
        print("[extract] Extraction complete.")
    else:
        print(f"[extract] Found existing extracted dataset in {raw_dir}")

    return raw_dir


def discover_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    """Discover all pocket-ligand pairs in the extracted CrossDocked directory."""
    pairs = []
    print("[discover] Scanning extracted complexes...")
    for pocket_path in data_dir.rglob("*.pdb"):
        stem = pocket_path.stem
        if "pocket" not in stem.lower():
            continue
        parent = pocket_path.parent
        sdfs = list(parent.glob("*.sdf"))
        if not sdfs:
            continue

        # Pair with companion ligand SDF
        ligand_path = sdfs[0]
        prefix = stem.split("_")[0]
        for s in sdfs:
            if s.stem.startswith(prefix):
                ligand_path = s
                break

        cid = f"{parent.name}_{stem}"
        pairs.append((pocket_path, ligand_path, cid))

    pairs.sort(key=lambda x: x[2])
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="./raw_data/crossdocked")
    parser.add_argument("--catalog", default="./data/enamine_3d_subset.parquet")
    parser.add_argument("--output-dir", default="./data/full_shards")
    parser.add_argument("--repo-id", default="JJKK1212/3d-syntree-multidataset")
    parser.add_argument("--max-complexes", type=int, default=None,
                        help="Optional limit for testing; omit to process the entire dataset")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--upload", action="store_true",
                        help="Upload directly to Hugging Face Hub (requires HF_TOKEN)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if args.upload and not token:
        print("ERROR: --upload requires HF_TOKEN environment variable", file=sys.stderr)
        return 1

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"[catalog] Generating canonical 3D catalog: {catalog_path}...")
        from scripts.download_assets import build_synthetic_catalog
        build_synthetic_catalog(str(catalog_path.parent), num_copies=40)

    catalog = SynthonCatalog(str(catalog_path))
    builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=args.max_steps)

    extracted_dir = download_and_extract(Path(args.raw_dir))
    all_pairs = discover_pairs(extracted_dir)
    if not all_pairs:
        print(f"ERROR: No pocket/ligand pairs found in {extracted_dir}", file=sys.stderr)
        return 1

    if args.max_complexes:
        all_pairs = all_pairs[:args.max_complexes]

    print(f"[process] Processing {len(all_pairs)} complexes through reaction-constrained decomposition...")

    rng = np.random.default_rng(42)
    shuffled_idx = rng.permutation(len(all_pairs))

    # 80% train, 10% val, 10% test
    n_train = int(len(all_pairs) * 0.80)
    n_val = int(len(all_pairs) * 0.10)

    split_map = {}
    for pos, idx in enumerate(shuffled_idx):
        split = "train" if pos < n_train else "val" if pos < (n_train + n_val) else "test"
        split_map[all_pairs[idx][2]] = split

    states_by_split = {"train": [], "val": [], "test": []}
    target_pockets: List[Path] = []
    accepted = 0

    pbar = tqdm(all_pairs, desc="Extracting trajectories")
    for pocket_path, ligand_path, cid in pbar:
        split = split_map[cid]
        try:
            pocket_mol = Chem.MolFromPDBFile(str(pocket_path), removeHs=False)
            suppl = Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=True)
            ligand_mol = next((m for m in suppl if m is not None), None)
            if pocket_mol is None or ligand_mol is None or ligand_mol.GetNumConformers() == 0:
                continue

            states, meta = builder.build(ligand_mol, pocket_mol, trajectory_id=cid)
            if states:
                states_by_split[split].extend(states)
                accepted += 1
                if split in ("train", "val") and len(target_pockets) < 50:
                    target_pockets.append(pocket_path)

            pbar.set_postfix({
                "accepted": accepted,
                "train_states": len(states_by_split["train"]),
                "val_states": len(states_by_split["val"]),
            })
        except Exception:
            continue

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for split in ("train", "val", "test"):
        samples = states_by_split[split]
        print(f"\n[shard] Writing {len(samples)} samples for split '{split}'...")
        manifest = write_shards(
            samples,
            split,
            str(out_root),
            max_shard_bytes=500 * 1024 * 1024,
        )
        manifests[split] = manifest

    # Stage RL evaluation target pockets
    targets_dir = out_root / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    for p in target_pockets:
        shutil.copy2(p, targets_dir / f"{p.stem}_pocket.pdb")

    # Copy canonical catalog
    shutil.copy2(catalog_path, out_root / "enamine_3d_subset.parquet")

    summary = {
        "status": "success",
        "total_complexes_evaluated": len(all_pairs),
        "accepted_complexes": accepted,
        "splits": {s: len(states_by_split[s]) for s in states_by_split},
        "target_pockets_staged": len(target_pockets),
    }
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print(json.dumps(summary, indent=2))
    print("=" * 60)

    if args.upload:
        print(f"\n[upload] Uploading complete real dataset to Hugging Face: {args.repo_id}...")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

        api.upload_file(
            path_or_fileobj=str(out_root / "enamine_3d_subset.parquet"),
            path_in_repo="enamine_3d_subset.parquet",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        for tp in targets_dir.glob("*.pdb"):
            api.upload_file(
                path_or_fileobj=str(tp),
                path_in_repo=f"targets/{tp.name}",
                repo_id=args.repo_id,
                repo_type="dataset",
            )

        for split in ("train", "val", "test"):
            upload_split(
                manifests[split],
                out_root / split,
                args.repo_id,
                token=token,
            )

        print(f"\n[upload] SUCCESS: Complete real dataset uploaded to {args.repo_id}!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

#### File 2: `syntree/engine/trainer.py` (Fail-Closed Real Data Guard)

Update `syntree/engine/trainer.py` lines 85–130 so the trainer **fails with a clear error** if it detects a 2-sample stub dataset:

```python
        # Data loading backend setup
        val_fraction = float(data_cfg.get("val_fraction", 0.1))
        trajectory_path = data_cfg.get("trajectory_dataset_path")
        self.data_backend = str(data_cfg.get("backend", "local")).lower()

        if self.data_backend == "huggingface":
            hf_cfg = dict(data_cfg.get("huggingface", {}))
            repo_id = str(hf_cfg.get("repo_id", "")).strip()
            if not repo_id:
                raise ValueError(
                    "data.huggingface.repo_id is required when data.backend='huggingface'"
                )
            token = os.environ.get("HF_TOKEN") or hf_cfg.get("token")
            self.dataset = ShardedHuggingFaceDataset(
                repo_id=repo_id,
                split="train",
                cache_dir=str(hf_cfg.get("cache_dir", "./hf_cache")),
                revision=str(hf_cfg.get("revision", "main")),
                token=token,
                max_cached_shards=int(hf_cfg.get("max_cached_shards", 4)),
            )
            self.val_dataset = ShardedHuggingFaceDataset(
                repo_id=repo_id,
                split="val",
                cache_dir=str(hf_cfg.get("cache_dir", "./hf_cache")),
                revision=str(hf_cfg.get("revision", "main")),
                token=token,
                max_cached_shards=int(hf_cfg.get("max_cached_shards", 4)),
            )
        elif trajectory_path:
            self.data_backend = "trajectory_pt"
            self.dataset = TrajectoryDataset(trajectory_path, split="train")
            self.val_dataset = TrajectoryDataset(trajectory_path, split="val")
        else:
            self.dataset = CrossDockedDataset(
                data_cfg["data_dir"],
                split="train",
                catalog=self.catalog,
                num_synthetic=int(data_cfg.get("synthetic_samples", 100)),
                synthetic_fallback=bool(data_cfg.get("synthetic_fallback", False)),
                split_manifest_path=data_cfg.get("split_manifest_path"),
            )
            self.val_dataset = CrossDockedDataset(
                data_cfg["data_dir"],
                split="val",
                catalog=self.catalog,
                num_synthetic=max(8, int(self.dataset.num_synthetic * val_fraction)),
                synthetic_fallback=bool(data_cfg.get("synthetic_fallback", False)),
                split_manifest_path=data_cfg.get("split_manifest_path"),
            )

        # STRICT REAL-DATA GUARD: Refuse to train on 2-sample stubs in production
        require_real = bool(data_cfg.get("require_real_data", True))
        min_samples = int(data_cfg.get("min_real_samples", 50))
        if require_real and len(self.dataset) < min_samples:
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"FATAL: Production mode requires the full real dataset, but '{self.data_backend}'\n"
                f"only contains {len(self.dataset)} training samples (minimum required: {min_samples}).\n\n"
                f"Training will NOT proceed on a stub or placeholder dataset.\n"
                f"To build and upload the complete real dataset, run Cell 4.5 in the notebook or:\n"
                f"  python scripts/build_full_dataset.py --upload\n"
                f"{'='*70}\n"
            )
```

---

#### File 3: `run_3d_syntree.ipynb` (Colab Notebook with Build Cell)

Replace `run_3d_syntree.ipynb` with this version. It adds **Cell 4.5**, allowing you to trigger the full real dataset download, extraction, and upload with one click right in Colab:

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RC5O_z05JEOx"
      },
      "source": [
        "# 3D-SynTree: Execution & Training Engine\n",
        "**Structure-Based Molecular Design via Reaction-Constrained Synthon Assembly**\n",
        "\n",
        "This notebook executes the production training pipeline for the `3d-syntree` framework.\n",
        "\n",
        "**Workflow:**\n",
        "1. Environment & HF Authentication\n",
        "2. Clone & Sync Repository\n",
        "3. Dependency Installation & GNINA Oracle Setup\n",
        "4. (Optional) Build & Upload Complete Real Dataset to HF Hub\n",
        "5. Sync Preprocessed Dataset Shards & Synthon Catalog\n",
        "6. Hardware Verification & Fresh Mode Setup\n",
        "7. Launch Multi-Hour Production Training\n",
        "8. Checkpoint Diagnostics\n",
        "9. Stage 2 Chemistry-Constrained PPO Fine-Tuning\n",
        "10. Comparative Benchmark Battery"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "--LqfWJlJEO0"
      },
      "outputs": [],
      "source": [
        "# CELL 1: Environment & Hugging Face Authentication\n",
        "import os, json\n",
        "\n",
        "HF_TOKEN = os.environ.get(\"HF_TOKEN\", \"\")\n",
        "if not HF_TOKEN:\n",
        "    try:\n",
        "        from google.colab import userdata\n",
        "        HF_TOKEN = userdata.get(\"HF_TOKEN\")\n",
        "    except Exception:\n",
        "        try:\n",
        "            from kaggle_secrets import UserSecretsClient\n",
        "            HF_TOKEN = UserSecretsClient().get_secret(\"HF_TOKEN\")\n",
        "        except Exception:\n",
        "            HF_TOKEN = \"\"\n",
        "\n",
        "if not HF_TOKEN:\n",
        "    from getpass import getpass\n",
        "    HF_TOKEN = getpass(\"\\nEnter your HuggingFace WRITE token: \").strip()\n",
        "\n",
        "os.environ[\"HF_TOKEN\"] = HF_TOKEN\n",
        "\n",
        "if HF_TOKEN:\n",
        "    from huggingface_hub import whoami\n",
        "    try:\n",
        "        info = whoami(token=HF_TOKEN)\n",
        "        user = info.get(\"name\", \"?\")\n",
        "        print(f\"HF token verified for user '{user}'.\")\n",
        "    except Exception as e:\n",
        "        raise RuntimeError(f\"HF token validation failed: {e}\")\n",
        "\n",
        "RUNTIME_CONFIG = {\n",
        "    \"repo_url\": \"https://github.com/Vtheonly/3d-syntree.git\",\n",
        "    \"branch\": \"main\",\n",
        "    \"hf_dataset_repo_id\": \"JJKK1212/3d-syntree-multidataset\",\n",
        "    \"hf_model_repo_id\": \"JJKK1212/3d-syntree-checkpoints\",\n",
        "    \"config_override\": {\n",
        "        \"huggingface\": {\n",
        "            \"enabled\": bool(HF_TOKEN),\n",
        "            \"repo_id\": \"JJKK1212/3d-syntree-checkpoints\",\n",
        "            \"push_every_n_epochs\": 2,\n",
        "            \"private\": False\n",
        "        },\n",
        "        \"data\": {\n",
        "            \"backend\": \"huggingface\",\n",
        "            \"require_real_data\": True,\n",
        "            \"min_real_samples\": 50,\n",
        "            \"synthetic_fallback\": False,\n",
        "            \"huggingface\": {\n",
        "                \"repo_id\": \"JJKK1212/3d-syntree-multidataset\",\n",
        "                \"revision\": \"main\",\n",
        "                \"cache_dir\": \"./hf_cache\",\n",
        "                \"max_cached_shards\": 4\n",
        "            }\n",
        "        }\n",
        "    }\n",
        "}\n",
        "\n",
        "with open(\"runtime_config.json\", \"w\") as f:\n",
        "    json.dump(RUNTIME_CONFIG, f, indent=2)\n",
        "print(\"Runtime configuration generated successfully.\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a6AYhstLJEO2"
      },
      "outputs": [],
      "source": [
        "# CELL 2: Clone Repository\n",
        "import os\n",
        "\n",
        "REPO_DIR = \"3d-syntree\"\n",
        "REPO_URL = RUNTIME_CONFIG[\"repo_url\"]\n",
        "BRANCH = RUNTIME_CONFIG[\"branch\"]\n",
        "\n",
        "if not os.path.exists(REPO_DIR):\n",
        "    print(f\"Cloning {REPO_URL} (branch: {BRANCH})...\")\n",
        "    !git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}\n",
        "else:\n",
        "    print(f\"{REPO_DIR} directory already exists.\")\n",
        "\n",
        "%cd {REPO_DIR}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pElHLrW9JEO2"
      },
      "outputs": [],
      "source": [
        "# CELL 3: Sync to Latest Git Commit\n",
        "BRANCH = RUNTIME_CONFIG[\"branch\"]\n",
        "!git fetch --all --prune\n",
        "!git checkout {BRANCH}\n",
        "!git reset --hard origin/{BRANCH}\n",
        "!git clean -fd\n",
        "!git log -1 --oneline"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ywgYCz19JEO3"
      },
      "outputs": [],
      "source": [
        "# CELL 4: Install Dependencies & Setup GNINA Docking Oracle\n",
        "!wget -q https://github.com/gnina/gnina/releases/download/v1.1/gnina -O /usr/local/bin/gnina && chmod +x /usr/local/bin/gnina\n",
        "!/usr/local/bin/gnina --version || echo 'gnina unavailable'\n",
        "\n",
        "!pip install --quiet --upgrade pip\n",
        "!pip install --quiet -r requirements.txt\n",
        "!pip install --quiet -e .\n",
        "\n",
        "import rdkit, torch, torch_geometric, huggingface_hub\n",
        "print(\n",
        "    f\"Environment Verified: PyTorch {torch.__version__} | \"\n",
        "    f\"PyG {torch_geometric.__version__} | RDKit {rdkit.__version__} | \"\n",
        "    f\"Hugging Face Hub {huggingface_hub.__version__}\"\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "build-real-dataset-cell"
      },
      "outputs": [],
      "source": [
        "# CELL 4.5: (RUN ONCE) Build & Upload Complete Real Dataset to HF Hub\n",
        "# Set BUILD_REAL_DATASET = True to download full CrossDocked2020 (~4GB from Zenodo),\n",
        "# extract 10A pockets, validate reaction trajectories with RDKit, and upload to HF.\n",
        "# Once uploaded, set this back to False for all future training sessions.\n",
        "BUILD_REAL_DATASET = False\n",
        "\n",
        "if BUILD_REAL_DATASET:\n",
        "    print(\"Starting automated full-scale dataset build and upload to Hugging Face...\")\n",
        "    !python scripts/build_full_dataset.py \\\n",
        "        --repo-id JJKK1212/3d-syntree-multidataset \\\n",
        "        --upload\n",
        "else:\n",
        "    print(\"Skipping dataset build. Using preprocessed shards from Hugging Face.\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "x9VpfievJEO3"
      },
      "outputs": [],
      "source": [
        "# CELL 5: Sync Clean Sharded Dataset & Synthon Catalog from HF Dataset Hub\n",
        "DATASET_REPO = RUNTIME_CONFIG[\"hf_dataset_repo_id\"]\n",
        "print(f\"Synchronizing preprocessed assets from HF Dataset: {DATASET_REPO}...\")\n",
        "\n",
        "!python scripts/download_assets.py \\\n",
        "    --dataset-repo {DATASET_REPO} \\\n",
        "    --dataset-revision main \\\n",
        "    --output-dir ./data\n",
        "\n",
        "import json\n",
        "with open(\"./data/assets_manifest.json\") as f:\n",
        "    meta = json.load(f)\n",
        "\n",
        "train_samples = meta[\"splits\"][\"train\"][\"total_samples\"]\n",
        "print(f\"\\nDataset Sync Complete: {train_samples} training samples available from {DATASET_REPO}.\")\n",
        "assert train_samples >= 50, (\n",
        "    f\"ERROR: {DATASET_REPO} only contains {train_samples} samples (2-sample stub).\\n\"\n",
        "    \"Set BUILD_REAL_DATASET = True in Cell 4.5 to download and upload the real CrossDocked dataset!\"\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4sa3sckgJEO4"
      },
      "outputs": [],
      "source": [
        "# CELL 6: Hardware Verification\n",
        "from syntree.utils.hardware import configure_runtime_environment, free_vram_bytes\n",
        "import json\n",
        "\n",
        "device_info = configure_runtime_environment()\n",
        "print(\"Hardware Execution Profile:\")\n",
        "print(json.dumps(device_info, indent=2))\n",
        "assert device_info[\"device\"].startswith(\"cuda\"), \"ERROR: No GPU detected! Go to Runtime > Change runtime type > GPU.\"\n",
        "free_gb = free_vram_bytes() / 1024**3\n",
        "print(f\"Free VRAM: {free_gb:.2f} GB -> Target 85% utilization: {0.85 * free_gb:.2f} GB\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "checkpoint-clean-reset"
      },
      "outputs": [],
      "source": [
        "# CELL 6.5: Training Mode Configuration (Fresh Run vs Auto-Resume)\n",
        "# Set FRESH_TRAINING = True for your first run on the full real dataset (starts from epoch 0).\n",
        "# Set FRESH_TRAINING = False if you get disconnected and want to resume.\n",
        "FRESH_TRAINING = True\n",
        "\n",
        "import os, shutil\n",
        "from huggingface_hub import HfApi\n",
        "\n",
        "if FRESH_TRAINING:\n",
        "    checkpoint_dir = os.path.join(\"experiments\", \"checkpoints\")\n",
        "    for path in (\n",
        "        checkpoint_dir,\n",
        "        os.path.join(\"experiments\", \"latest_metrics.json\"),\n",
        "        os.path.join(\"experiments\", \"progress.json\"),\n",
        "        os.path.join(\"experiments\", \"history.json\"),\n",
        "    ):\n",
        "        if os.path.isdir(path):\n",
        "            shutil.rmtree(path)\n",
        "        elif os.path.exists(path):\n",
        "            os.remove(path)\n",
        "\n",
        "    token = os.environ.get(\"HF_TOKEN\")\n",
        "    model_repo = RUNTIME_CONFIG.get(\"hf_model_repo_id\")\n",
        "    if token and model_repo:\n",
        "        try:\n",
        "            api = HfApi(token=token)\n",
        "            remote_files = api.list_repo_files(repo_id=model_repo, repo_type=\"model\")\n",
        "            deleted = 0\n",
        "            for f in remote_files:\n",
        "                if f in (\"manifest.json\", \"progress.json\") or f.startswith(\"checkpoints/\"):\n",
        "                    api.delete_file(path_in_repo=f, repo_id=model_repo, repo_type=\"model\")\n",
        "                    deleted += 1\n",
        "            print(f\"Fresh mode: purged {deleted} old checkpoint file(s) from HF Hub ({model_repo}).\")\n",
        "        except Exception as e:\n",
        "            print(f\"Notice: Remote repo check: {e}\")\n",
        "\n",
        "    TRAIN_FLAG = \"--fresh\"\n",
        "    print(\"Training mode: FRESH RUN (--fresh, starting from epoch 0).\")\n",
        "else:\n",
        "    TRAIN_FLAG = \"--resume-auto\"\n",
        "    print(\"Training mode: AUTO-RESUME (--resume-auto, continuing from latest checkpoint).\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "l7SvCIlDJEO4"
      },
      "outputs": [],
      "source": [
        "# CELL 7: Launch Multi-Hour Production Training\n",
        "print(f\"Starting 3D-SynTree production training engine ({TRAIN_FLAG})...\")\n",
        "\n",
        "!python main.py \\\n",
        "    --mode train \\\n",
        "    --config configs/train_colab_12h.json \\\n",
        "    --runtime-config ../runtime_config.json \\\n",
        "    $TRAIN_FLAG"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RBdZFS7TJEO4"
      },
      "outputs": [],
      "source": [
        "# CELL 8: Status & Checkpoint Sync Verification\n",
        "from syntree.utils.checkpoint import verify_hf_sync\n",
        "import json, os\n",
        "\n",
        "MODEL_REPO = RUNTIME_CONFIG[\"hf_model_repo_id\"]\n",
        "status = verify_hf_sync(MODEL_REPO)\n",
        "\n",
        "print(\"=== SESSION STATUS ===\")\n",
        "print(f\"Hugging Face Model Repo: {MODEL_REPO}\")\n",
        "print(f\"Latest Remote Checkpoint: {status['latest_remote_checkpoint']}\")\n",
        "print(f\"Total Epochs Completed:   {status['epochs_completed']}\")\n",
        "print(f\"Sync Operational:         {status['sync_ok']}\")\n",
        "\n",
        "if os.path.exists(\"experiments/latest_metrics.json\"):\n",
        "    with open(\"experiments/latest_metrics.json\") as f:\n",
        "        print(\"\\nLatest Validation Metrics:\")\n",
        "        print(json.dumps(json.load(f), indent=2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OXsm4UXfJEO5"
      },
      "source": [
        "## Stage 2: Chemistry-Constrained PPO Fine-Tuning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lR0daEzhJEO5"
      },
      "outputs": [],
      "source": [
        "# CELL 9: Stage 2 PPO\n",
        "print(\"Starting chemistry-constrained PPO fine-tuning...\")\n",
        "!python main.py \\\n",
        "    --mode rl \\\n",
        "    --config configs/rl_colab_12h.json \\\n",
        "    --pocket-dir ./data/test_pockets \\\n",
        "    --resume-auto"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MkgoKiGYJEO5"
      },
      "source": [
        "## Comparative Evaluation Battery"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bJQzxPDuJEO6"
      },
      "outputs": [],
      "source": [
        "# CELL 10: Comparative Benchmark\n",
        "!python scripts/run_comparative_benchmark.py \\\n",
        "    --manifest ./benchmarks/targets.jsonl \\\n",
        "    --outputs ./benchmarks/outputs \\\n",
        "    --methods 3d-syntree \\\n",
        "    --limit 100"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
```

---

### How to Run Full-Scale Training Right Now

1. Copy the code for the three files above (`configs/train_colab_12h.json`, `scripts/build_full_dataset.py`, `syntree/engine/trainer.py`, and `run_3d_syntree.ipynb`) into your repository.
2. Open `run_3d_syntree.ipynb` in Colab.
3. In **Cell 4.5**, set:
   ```python
   BUILD_REAL_DATASET = True
   ```
   Run Cell 4.5. It will:
   - Download the real 4 GB CrossDocked2020 archive from Zenodo.
   - Decompose thousands of protein-ligand complexes into certified reaction trajectories.
   - Upload the complete `.pt.gz` shards to `JJKK1212/3d-syntree-multidataset`.
   *(Once finished, set `BUILD_REAL_DATASET = False` because you never need to do it again).*
4. Run **Cell 5**. It will sync the dataset and confirm thousands of real training samples.
5. In **Cell 6.5**, keep `FRESH_TRAINING = True`.
6. Run **Cell 7**.
   
The model will launch production training on thousands of real complexes, scaling to fill your T4 GPU memory, and will run continuously for hours.
""""""""


Tokens / credentials

Never paste live tokens into tracked files — GitHub push protection rejects the
push and the credential is considered compromised the moment it is committed.

Provide them at runtime instead:

* Hugging Face write token: `export HF_TOKEN=...` (or a Colab/Kaggle secret named
  `HF_TOKEN`; the pipeline reads it in that order).
* GitHub personal access token: `gh auth login`, or `export GITHUB_TOKEN=...`.

If you need them on disk, keep them in an untracked file such as
`.secrets/tokens.env` (already matched by `.gitignore`).

> NOTE: an earlier revision of this file committed a live Hugging Face token and
> a GitHub PAT. They were removed before that commit was ever pushed and both
> credentials must be revoked/rotated.


Make sure to do a ton of testing—extensive, thorough testing of the mathematical formulas, data compatibility, data availability, data integrity, training and inference pipelines, and everything else involved in the project.

Do not just test whether the code runs. Verify that the mathematical logic is correct, the datasets are compatible and complete, the data flows correctly through the entire pipeline, and the results are consistent and reproducible.



Before making changes,push and merge the code after each commit and be careful of conflicts in code and logic
Before making changes,push and merge the code after each commit and be careful of conflicts in code and logic
Before making changes,push and merge the code after each commit and be careful of conflicts in code and logic
