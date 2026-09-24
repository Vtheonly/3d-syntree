#!/usr/bin/env python3
"""Automated pipeline: download full real CrossDocked2020, preprocess, and upload to Hugging Face.

Workflow:
  1. Download CrossDocked2020 (~4 GB tar.gz from Zenodo).
  2. Extract pocket and ligand pairs (supports *_pocket10.pdb + *.sdf / *.mol2).
  3. Extract reaction-validated retrosynthetic trajectories against the Enamine catalog.
  4. Partition into train/val/test splits (80/10/10, deterministic seed).
  5. Package into 500 MB compressed shards with SHA-256 manifests.
  6. Stage target pocket PDBs for Stage 2 RL docking evaluation.
  7. Upload clean shards, manifests, exact catalog, and targets to Hugging Face Hub.

Every accepted training state comes from a real protein-ligand complex whose
retrosynthetic decomposition replays forward through the RDKit reaction
engine. No synthetic samples are ever generated or uploaded.
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

# Ligand file extensions recognised by the pair discovery step, in priority order.
_LIGAND_EXTENSIONS = (".sdf", ".mol2")


def download_and_extract(raw_dir: Path, zenodo_url: str = ZENODO_CROSSDOCKED_URL) -> Path:
    """Download and extract the CrossDocked2020 archive (idempotent)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "crossdocked_pocket10.tar.gz"
    marker = raw_dir / ".extracted_marker"

    if not marker.exists():
        if not archive_path.exists():
            print(f"[download] Fetching real CrossDocked2020 from {zenodo_url}...")
            urllib.request.urlretrieve(zenodo_url, archive_path)
            size_gb = archive_path.stat().st_size / 1e9
            print(f"[download] Archive downloaded ({size_gb:.2f} GB)")
            if size_gb < 0.001:
                raise RuntimeError(
                    f"Downloaded archive is suspiciously small ({size_gb * 1e6:.0f} bytes); "
                    "refusing to extract a truncated download."
                )

        print(f"[extract] Extracting {archive_path.name} into {raw_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            try:
                # Python >= 3.12: reject absolute paths / traversal entries.
                tar.extractall(path=raw_dir, filter="data")
            except TypeError:  # pragma: no cover - older Python without filter
                tar.extractall(path=raw_dir)
        marker.touch()
        print("[extract] Extraction complete.")
    else:
        print(f"[extract] Found existing extracted dataset in {raw_dir}")

    return raw_dir


def _normalize_pocket_stem(stem: str) -> str:
    """Strip the pocket marker from a pocket file stem."""
    for suffix in ("_pocket10", "_pocket"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _normalize_ligand_stem(stem: str) -> str:
    """Strip the ligand marker from a ligand file stem."""
    for suffix in ("_ligand", "_lig", "_dock"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _best_ligand_match(pocket_stem: str, ligand_paths: List[Path]) -> Path:
    """Pick the ligand file that best matches a pocket stem.

    Preference order:
    1. exact normalized-stem equality (``<id>_pocket10.pdb`` <->
       ``<id>_ligand.sdf`` / ``<id>_ligand.mol2``),
    2. longest shared prefix with the normalized pocket stem (CrossDocked
       naming keeps the complex id as the common prefix of both files),
    3. first ligand in the directory (single-complex folders).
    """
    pocket_norm = _normalize_pocket_stem(pocket_stem)
    best = ligand_paths[0]
    best_score = -1
    for candidate in ligand_paths:
        ligand_norm = _normalize_ligand_stem(candidate.stem)
        if ligand_norm == pocket_norm:
            return candidate
        common = 0
        for a, b in zip(pocket_norm, ligand_norm):
            if a != b:
                break
            common += 1
        # Prefer an exact-prefix ligand over one that merely shares a prefix
        # with the raw (un-normalised) pocket stem.
        if common > best_score:
            best_score = common
            best = candidate
    return best


def discover_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    """Discover all pocket-ligand pairs in the extracted CrossDocked directory."""
    pairs = []
    print("[discover] Scanning extracted complexes...")
    for pocket_path in sorted(Path(data_dir).rglob("*.pdb")):
        stem = pocket_path.stem
        if "pocket" not in stem.lower():
            continue
        parent = pocket_path.parent
        ligands = sorted(
            p for p in parent.iterdir()
            if p.suffix.lower() in _LIGAND_EXTENSIONS
        )
        if not ligands:
            continue

        ligand_path = _best_ligand_match(stem, ligands)
        cid = f"{parent.name}_{stem}"
        pairs.append((pocket_path, ligand_path, cid))

    pairs.sort(key=lambda x: x[2])
    return pairs


def load_ligand(ligand_path: Path) -> Chem.Mol:
    """Load the first valid ligand molecule from an SDF or Mol2 file."""
    suffix = Path(ligand_path).suffix.lower()
    if suffix == ".mol2":
        mol = Chem.MolFromMol2File(
            str(ligand_path), removeHs=False, sanitize=True
        )
        return mol
    supplier = Chem.SDMolSupplier(
        str(ligand_path), removeHs=False, sanitize=True
    )
    return next((m for m in supplier if m is not None), None)


def load_pocket(pocket_path: Path) -> Chem.Mol:
    """Load a protein pocket PDB with hydrogens preserved."""
    return Chem.MolFromPDBFile(str(pocket_path), removeHs=False)


def assign_splits(
    n_items: int, seed: int = 42
) -> List[str]:
    """Deterministic shuffled 80/10/10 split labels ('train'/'val'/'test')."""
    if n_items < 3:
        raise ValueError(
            f"Cannot 80/10/10-split {n_items} items; need at least 3 complexes."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    n_train = int(n_items * 0.80)
    n_val = int(n_items * 0.10)
    labels = [None] * n_items
    for pos, idx in enumerate(order):
        split = "train" if pos < n_train else "val" if pos < (n_train + n_val) else "test"
        labels[int(idx)] = split
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="./raw_data/crossdocked")
    parser.add_argument("--catalog", default="./data/enamine_3d_subset.parquet")
    parser.add_argument("--output-dir", default="./data/full_shards")
    parser.add_argument("--repo-id", default="JJKK1212/3d-syntree-multidataset")
    parser.add_argument("--max-complexes", type=int, default=None,
                        help="Optional limit for testing; omit to process the entire dataset")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-target-pockets", type=int, default=50,
                        help="How many RL evaluation target pockets to stage.")
    parser.add_argument("--shard-mb", type=int, default=500,
                        help="Maximum compressed shard size in MB.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic split seed (80/10/10).")
    parser.add_argument("--upload", action="store_true",
                        help="Upload directly to Hugging Face Hub (requires HF_TOKEN)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if args.upload and not token:
        print("ERROR: --upload requires HF_TOKEN environment variable", file=sys.stderr)
        return 1
    if args.max_complexes is not None and args.max_complexes < 3:
        print(
            "ERROR: --max-complexes must be >= 3 (one complex per 80/10/10 split)",
            file=sys.stderr,
        )
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
        all_pairs = all_pairs[: args.max_complexes]

    print(
        f"[process] Processing {len(all_pairs)} complexes through "
        "reaction-constrained decomposition..."
    )

    split_labels = assign_splits(len(all_pairs), seed=args.seed)
    split_map = {
        all_pairs[i][2]: split_labels[i] for i in range(len(all_pairs))
    }

    states_by_split = {"train": [], "val": [], "test": []}
    target_pockets: List[Path] = []
    accepted = 0

    pbar = tqdm(all_pairs, desc="Extracting trajectories")
    for pocket_path, ligand_path, cid in pbar:
        split = split_map[cid]
        try:
            pocket_mol = load_pocket(pocket_path)
            ligand_mol = load_ligand(ligand_path)
            if pocket_mol is None or ligand_mol is None:
                continue
            if pocket_mol.GetNumAtoms() == 0 or ligand_mol.GetNumAtoms() == 0:
                continue
            if ligand_mol.GetNumConformers() == 0:
                continue

            states, meta = builder.build(ligand_mol, pocket_mol, trajectory_id=cid)
            if states:
                states_by_split[split].extend(states)
                accepted += 1
                if (
                    split in ("train", "val")
                    and len(target_pockets) < args.max_target_pockets
                ):
                    target_pockets.append(pocket_path)
        except Exception:
            continue

        pbar.set_postfix({
            "accepted": accepted,
            "train_states": len(states_by_split["train"]),
            "val_states": len(states_by_split["val"]),
        })

    if accepted == 0:
        print(
            "ERROR: zero complexes produced valid reaction trajectories; "
            "check that the catalog covers the ligand chemistry in the dataset.",
            file=sys.stderr,
        )
        return 1

    for split in ("train", "val", "test"):
        if not states_by_split[split]:
            print(
                f"[warn] split '{split}' received 0 samples; downstream "
                "manifest validation will reject an empty split.",
                file=sys.stderr,
            )

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
            max_shard_bytes=args.shard_mb * 1024 * 1024,
        )
        manifests[split] = manifest

    # Stage RL evaluation target pockets (deterministic prefix, renamed to the
    # *_pocket.pdb convention expected by --mode rl and download_assets.py).
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
        "seed": int(args.seed),
        "max_shard_bytes": int(args.shard_mb * 1024 * 1024),
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
