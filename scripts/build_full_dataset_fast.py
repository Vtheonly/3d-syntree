#!/usr/bin/env python3
"""High-speed multi-core dataset builder for CrossDocked2020.

Features:
- Fast system tar extraction (~15s).
- Single-pass filesystem indexing via os.walk (<10s).
- Multi-core multiprocessing pool utilizing all available CPU cores.
- Deterministic 80/10/10 splits and byte-bounded compressed shards.
- Stages RL target pockets and copies canonical synthon catalog.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder
from scripts.shard_and_upload import write_shards

_LIGAND_EXTENSIONS = (".sdf", ".mol2")

_worker_catalog = None
_worker_builder = None


def _init_worker(catalog_path: str, max_steps: int):
    """Initialize worker process with its own RDKit catalog and builder."""
    global _worker_catalog, _worker_builder
    _worker_catalog = SynthonCatalog(catalog_path)
    _worker_builder = RetrosyntheticTrajectoryBuilder(_worker_catalog, max_steps=max_steps)


def _process_item(item: Tuple[str, str, str, str]):
    """Worker task: load pocket + ligand, extract trajectories."""
    p_path, l_path, cid, split = item
    global _worker_builder
    try:
        pocket_mol = Chem.MolFromPDBFile(p_path, removeHs=False)
        if l_path.lower().endswith(".mol2"):
            ligand_mol = Chem.MolFromMol2File(l_path, removeHs=False, sanitize=True)
        else:
            supp = Chem.SDMolSupplier(l_path, removeHs=False, sanitize=True)
            ligand_mol = next((m for m in supp if m is not None), None)

        if pocket_mol is None or ligand_mol is None:
            return None
        if pocket_mol.GetNumAtoms() == 0 or ligand_mol.GetNumAtoms() == 0 or ligand_mol.GetNumConformers() == 0:
            return None

        states, meta = _worker_builder.build(ligand_mol, pocket_mol, trajectory_id=cid)
        if states:
            return (split, states, p_path)
    except Exception:
        pass
    return None


def _normalize_stem(stem: str) -> str:
    for suffix in ("_pocket10", "_pocket", "_ligand", "_lig", "_dock"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _best_ligand(pocket_stem: str, ligands: List[str]) -> str:
    p_norm = _normalize_stem(pocket_stem)
    best = ligands[0]
    best_score = -1
    for cand in ligands:
        c_norm = _normalize_stem(Path(cand).stem)
        if c_norm == p_norm:
            return cand
        common = 0
        for a, b in zip(p_norm, c_norm):
            if a != b:
                break
            common += 1
        if common > best_score:
            best_score = common
            best = cand
    return best


def fast_discover_pairs(data_dir: Path) -> List[Tuple[str, str, str]]:
    """Single-pass C-level directory walk (<10 seconds)."""
    print(f"[discover] Single-pass indexing of {data_dir}...")
    t0 = time.time()
    pairs = []
    for root, _, files in os.walk(data_dir):
        pockets = [f for f in files if "pocket" in f.lower() and f.endswith(".pdb")]
        if not pockets:
            continue
        ligands = [f for f in files if f.endswith(_LIGAND_EXTENSIONS)]
        if not ligands:
            continue
        parent_name = os.path.basename(root)
        for p in pockets:
            p_stem = p[:-4]
            best_l = _best_ligand(p_stem, ligands)
            pairs.append((
                os.path.join(root, p),
                os.path.join(root, best_l),
                f"{parent_name}_{p_stem}"
            ))
    pairs.sort(key=lambda x: x[2])
    print(f"[discover] Found {len(pairs)} pairs in {time.time() - t0:.2f}s!")
    return pairs


def fast_extract(raw_dir: Path):
    """Fast extraction using system tar (~15s) with Python tarfile fallback."""
    archive = raw_dir / "crossdocked_pocket10.tar.gz"
    marker = raw_dir / ".extracted_marker"
    if marker.exists():
        print("[extract] Already extracted (marker verified).")
        return
    if not archive.exists():
        raise FileNotFoundError(f"Archive missing at {archive}")

    print(f"[extract] Extracting {archive.name} via high-speed system tar...")
    t0 = time.time()
    try:
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(raw_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("[extract] System tar unavailable; falling back to Python tarfile...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=raw_dir)
    marker.touch()
    print(f"[extract] Extraction complete in {time.time() - t0:.2f}s!")


def main():
    parser = argparse.ArgumentParser(description="Accelerated CrossDocked Trajectory Builder")
    parser.add_argument("--raw-dir", default="./raw_data/crossdocked")
    parser.add_argument("--catalog", default="./data/enamine_3d_subset.parquet")
    parser.add_argument("--output-dir", default="./data/full_shards")
    parser.add_argument("--max-complexes", type=int, default=25000)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--shard-mb", type=int, default=500)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fast_extract(raw_dir)
    pairs = fast_discover_pairs(raw_dir)

    if not pairs:
        print(f"ERROR: No pocket/ligand pairs found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_complexes and len(pairs) > args.max_complexes:
        print(f"[subsample] Capping to {args.max_complexes} diverse complexes for optimal yield.")
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(pairs), size=args.max_complexes, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]

    n_items = len(pairs)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_items)
    n_train = int(n_items * 0.80)
    n_val = int(n_items * 0.10)
    split_map = {
        pairs[idx][2]: ("train" if pos < n_train else "val" if pos < (n_train + n_val) else "test")
        for pos, idx in enumerate(order)
    }

    work_items = [(p, l, cid, split_map[cid]) for p, l, cid in pairs]
    print(f"[parallel] Processing {len(work_items)} complexes using {args.workers} CPU workers...")

    states_by_split = {"train": [], "val": [], "test": []}
    target_pockets = []
    accepted = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(str(args.catalog), args.max_steps)) as pool:
        for res in tqdm(pool.imap_unordered(_process_item, work_items, chunksize=32), total=len(work_items), desc="Decomposing trajectories"):
            if res is not None:
                split, states, p_path = res
                states_by_split[split].extend(states)
                accepted += 1
                if split in ("train", "val") and len(target_pockets) < 50:
                    target_pockets.append(p_path)

    print(f"\n[done] Extraction finished in {(time.time() - t0)/60:.2f} mins! Accepted {accepted}/{len(work_items)} complexes.")
    print(f"[states] Train: {len(states_by_split['train'])}, Val: {len(states_by_split['val'])}, Test: {len(states_by_split['test'])}")

    for split in ("train", "val", "test"):
        samples = states_by_split[split]
        print(f"[shard] Packaging {len(samples)} states for split '{split}'...")
        write_shards(samples, split, str(out_dir), max_shard_bytes=args.shard_mb * 1024 * 1024)

    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    for tp in target_pockets:
        shutil.copy2(tp, targets_dir / f"{Path(tp).stem}_pocket.pdb")

    shutil.copy2(args.catalog, out_dir / "enamine_3d_subset.parquet")
    print(f"[success] All shards and target pockets written to {out_dir}")


if __name__ == "__main__":
    main()