#!/usr/bin/env python
"""#35 EVAL-02 — freeze the comparative benchmark target manifest.

The frozen target manifest defines **the identical test targets** every
method (3D-SynTree, TargetDiff, DiffSBDD, SyntheMol) is evaluated on
(#35: "evaluate exported structures ... using identical evaluation
code").  It is generated — never hand-edited — from the immutable #18
artifacts: the split trajectory records (``trajectories_test.pt``) and
the processed pocket artifacts they reference.

The honest frozen set at the current project stage is the **33
test-split targets with accepted forward-replay trajectories** (the
#18 census over 170,955 accepted complexes accepted 194 trajectories
overall — 33 in test; the issue's nominal "100 held-out targets" is the
design target, and the manifest records the actual count per the
constitution's no-fabricated-counts rule).

Usage::

    python scripts/make_frozen_targets.py \
        --trajectories work/data/eval/bundle/trajectories_test.pt \
        --processed-dir work/data/eval/bundle/processed \
        --output work/data/eval/frozen_targets.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#35: freeze the comparative benchmark targets"
    )
    parser.add_argument(
        "--trajectories",
        required=True,
        help="trajectories_test.pt (the #18 split artifact)",
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="the bundle processed/ dir with pocket artifacts",
    )
    parser.add_argument("--output", required=True, help="manifest JSON path")
    parser.add_argument(
        "--split",
        default="test",
        help="which split artifact the manifest freezes (default: test)",
    )
    args = parser.parse_args()

    import torch

    records = torch.load(args.trajectories, weights_only=False)
    if not records:
        raise SystemExit(f"error: no trajectory records in {args.trajectories}")

    targets = []
    missing = []
    for record in records:
        target_id = record["complex_id"]
        pocket = os.path.join(args.processed_dir, record["pocket_artifact"])
        if not os.path.exists(pocket):
            missing.append(pocket)
            continue
        targets.append(
            {
                "target_id": target_id,
                "pocket": os.path.abspath(pocket),
                "reference_ligand_smiles": record["ligand_smiles"],
                "notes": (
                    "accepted #18 forward-replay trajectory; "
                    "seed block " + str(record.get("seed_block_id"))
                ),
            }
        )
    if missing:
        raise SystemExit(
            f"error: {len(missing)} pocket artifacts missing, e.g. {missing[0]}"
        )
    manifest = {
        "split": args.split,
        "source": {
            "trajectories": os.path.abspath(args.trajectories),
            "processed_dir": os.path.abspath(args.processed_dir),
        },
        "num_targets": len(targets),
        "honesty_note": (
            "the frozen set is the accepted-trajectory subset of the "
            f"{args.split} split (the #18 census); the #35 design target "
            "of 100 held-out targets is not yet reachable — the manifest "
            "records the actual count"
        ),
        "targets": targets,
    }
    with open(args.output, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[targets] frozen {len(targets)} {args.split}-split targets")
    print(f"[targets] manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
