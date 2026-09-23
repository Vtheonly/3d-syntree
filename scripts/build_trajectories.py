#!/usr/bin/env python3
"""Build reaction-validated trajectories from standardized multi-dataset records.

Canonical path:
  preprocess_multidataset.py -> dump_all_fastas.py -> MMseqs2 -> split_clusters.py
  -> build_trajectories.py -> shard_and_upload.py

A legacy CrossDocked directory mode is kept for backwards compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
import torch
from rdkit import Chem

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.multidataset import iter_jsonl
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_first_molecule(path: str):
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
    return next((mol for mol in supplier if mol is not None), None)


def build_unified(args) -> int:
    split_frame = pd.read_parquet(args.split_manifest)
    split_map = dict(zip(split_frame["sample_id"].astype(str), split_frame["split"].astype(str)))
    rows = [
        row for row in iter_jsonl(args.manifest)
        if split_map.get(str(row["sample_id"])) == args.split
    ]
    if not rows:
        raise RuntimeError(f"No records assigned to split={args.split!r}")

    catalog = SynthonCatalog(args.catalog)
    builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=args.max_steps)
    states: List = []
    source_counts = Counter()
    status_counts = Counter()

    for row in rows:
        pocket = Chem.MolFromPDBFile(str(row["processed_pocket_path"]), removeHs=False)
        ligand = load_first_molecule(str(row["processed_ligand_path"]))
        if pocket is None or ligand is None:
            status_counts["invalid_pair"] += 1
            continue

        generated, metadata = builder.build(
            ligand, pocket, trajectory_id=str(row["sample_id"])
        )
        if len(generated) < args.min_states:
            status_counts[metadata.get("status", "rejected")] += 1
            continue

        for sample in generated:
            sample.dataset_source = str(row["source"])
            sample.dataset_sample_id = str(row["sample_id"])
            sample.dataset_resolution = float(row["resolution"] or 0.0)
            sample.dataset_split = str(args.split)

        states.extend(generated)
        source_counts[str(row["source"])] += len(generated)
        status_counts["accepted"] += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(states, output)
    manifest = {
        "split": args.split,
        "input_records": len(rows),
        "states_written": len(states),
        "source_state_counts": dict(source_counts),
        "trajectory_status_counts": dict(status_counts),
        "catalog": str(args.catalog),
        "catalog_sha256": sha256_file(args.catalog),
        "max_steps": args.max_steps,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2))
    return 0


def build_legacy(args) -> int:
    catalog = SynthonCatalog(args.catalog)
    builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=args.max_steps)
    pocket_paths = sorted(
        os.path.join(args.data_dir, name)
        for name in os.listdir(args.data_dir)
        if name.endswith("_pocket.pdb")
    )
    all_states: List = []
    records = {
        "data_dir": args.data_dir,
        "pairs_seen": 0,
        "trajectories_accepted": 0,
        "states_written": 0,
        "skipped": {},
    }

    for pocket_path in pocket_paths:
        records["pairs_seen"] += 1
        ligand_path = pocket_path.replace("_pocket.pdb", "_ligand.sdf")
        if not os.path.exists(ligand_path):
            records["skipped"]["missing_ligand"] = records["skipped"].get("missing_ligand", 0) + 1
            continue
        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=False)
        ligand = load_first_molecule(ligand_path)
        if pocket is None or ligand is None:
            records["skipped"]["invalid_pair"] = records["skipped"].get("invalid_pair", 0) + 1
            continue
        states, metadata = builder.build(
            ligand, pocket, trajectory_id=os.path.basename(pocket_path).replace("_pocket.pdb", "")
        )
        if len(states) < args.min_states:
            reason = metadata.get("status", "rejected")
            records["skipped"][reason] = records["skipped"].get(reason, 0) + 1
            continue
        all_states.extend(states)
        records["trajectories_accepted"] += 1
        records["states_written"] += len(states)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_states, output)
    output.with_suffix(".json").write_text(json.dumps(records, indent=2, sort_keys=True))
    print(json.dumps(records, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--split-manifest")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--catalog", default="./data/enamine_3d_subset.parquet")
    parser.add_argument("--output", default="./data/trajectories.pt")
    parser.add_argument("--data-dir", default="./data/crossdocked")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--min-states", type=int, default=1)
    args = parser.parse_args()
    unified = args.manifest is not None or args.split_manifest is not None
    if unified:
        if not args.manifest or not args.split_manifest or not args.split:
            parser.error("--manifest, --split-manifest and --split are required together")
        return build_unified(args)
    return build_legacy(args)


if __name__ == "__main__":
    raise SystemExit(main())