#!/usr/bin/env python3
"""Build an offline reaction-validated imitation-learning trajectory dataset.

The notebook/CLI only orchestrates execution; decomposition logic lives in
syntree.data.trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import torch
from rdkit import Chem

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder


def load_first_molecule(path: str):
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
    return next((mol for mol in supplier if mol is not None), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 3D-SynTree expert trajectories")
    parser.add_argument("--data-dir", default="./data/crossdocked")
    parser.add_argument("--catalog", default="./data/enamine_3d_subset.parquet")
    parser.add_argument("--output", default="./data/trajectories.pt")
    parser.add_argument("--manifest", default="./data/trajectories_manifest.json")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--min-states", type=int, default=1)
    args = parser.parse_args()

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
        "catalog": args.catalog,
        "max_steps": args.max_steps,
        "pairs_seen": 0,
        "trajectories_accepted": 0,
        "states_written": 0,
        "skipped": {},
    }

    for pocket_path in pocket_paths:
        records["pairs_seen"] += 1
        ligand_path = pocket_path.replace("_pocket.pdb", "_ligand.sdf")
        if not os.path.exists(ligand_path):
            records["skipped"]["missing_ligand"] = records["skipped"].get(
                "missing_ligand", 0
            ) + 1
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=False)
        ligand = load_first_molecule(ligand_path)
        if pocket is None or ligand is None:
            records["skipped"]["invalid_pair"] = records["skipped"].get(
                "invalid_pair", 0
            ) + 1
            continue

        states, metadata = builder.build(
            ligand,
            pocket,
            trajectory_id=os.path.basename(pocket_path).replace("_pocket.pdb", ""),
        )
        if len(states) < args.min_states:
            reason = metadata.get("status", "rejected")
            records["skipped"][reason] = records["skipped"].get(reason, 0) + 1
            continue

        all_states.extend(states)
        records["trajectories_accepted"] += 1
        records["states_written"] += len(states)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(all_states, args.output)
    with open(args.manifest, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[trajectory] accepted trajectories: {records['trajectories_accepted']}")
    print(f"[trajectory] states written: {records['states_written']}")
    print(f"[trajectory] dataset: {args.output}")
    print(f"[trajectory] manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
