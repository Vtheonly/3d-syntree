#!/usr/bin/env python3
"""Paired multi-target benchmark for 3D-SynTree and external SBDD baselines.

The script intentionally does not reimplement TargetDiff, DiffSBDD, or other
published generators. It evaluates their exported SDFs through the same
EvaluationPipeline, so each method receives the identical target pockets and
metric implementation.

Manifest JSONL schema:
    {"target_id": "1abc", "pocket": "pockets/1abc.pdb",
     "reference_ligand": "ligands/1abc.sdf"}

Method layout:
    <root>/<method>/<target_id>/ligand_*.sdf
    <root>/<method>/<target_id>/recipe_*.json   # optional

Example:
    python scripts/run_comparative_benchmark.py \
      --manifest ./benchmarks/targets.jsonl \
      --outputs ./benchmarks/outputs \
      --methods 3d-syntree,targetdiff,synthemol \
      --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdkit import Chem

from syntree.engine.evaluator import EvaluationPipeline


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def load_target_molecules(target_dir: Path):
    mols, recipes = [], []
    for sdf in sorted(target_dir.glob("ligand_*.sdf")):
        supplier = Chem.SDMolSupplier(str(sdf), removeHs=False)
        mol = supplier[0] if len(supplier) else None
        mols.append(mol)
        recipe = target_dir / sdf.name.replace("ligand_", "recipe_").replace(".sdf", ".json")
        recipes.append(json.loads(recipe.read_text()) if recipe.exists() else None)
    return mols, recipes


def aggregate(records: List[Dict]) -> Dict:
    scalar_keys = [
        "chemical_validity",
        "posebusters_pass_rate",
        "mean_fsp3",
        "fsp3_ge_0.42_rate",
        "mean_mw",
        "recipe_completeness",
        "uniqueness",
        "mean_pairwise_tanimoto_distance",
        "novelty",
        "mean_vina_score",
        "retrosynthetic_feasibility",
    ]
    out = {"targets_evaluated": len(records)}
    for key in scalar_keys:
        values = [
            float(r[key]) for r in records
            if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)
        ]
        if values:
            out[key] = float(np.mean(values))
            out[key + "_std"] = float(np.std(values))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired SBDD multi-target benchmark")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outputs", required=True,
                        help="root containing one directory per method")
    parser.add_argument("--methods", required=True,
                        help="comma-separated method directory names")
    parser.add_argument("--config", default="configs/default_config.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="./experiments/comparative_benchmark.json")
    args = parser.parse_args()

    targets = load_jsonl(args.manifest)[: max(0, args.limit)]
    if not targets:
        raise SystemExit("No targets found in manifest.")

    config = json.loads(Path(args.config).read_text())
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    output_root = Path(args.outputs)
    results = {"manifest": args.manifest, "target_count": len(targets), "methods": {}}

    for method in methods:
        method_records = []
        method_root = output_root / method
        for target in targets:
            target_id = str(target["target_id"])
            target_dir = method_root / target_id
            mols, recipes = load_target_molecules(target_dir)
            if not mols:
                continue

            pocket = Path(target["pocket"])
            if not pocket.is_absolute():
                pocket = Path(args.manifest).resolve().parent / pocket

            evaluator = EvaluationPipeline(
                config,
                output_dir=str(Path(args.output).parent / "per_target" / method / target_id),
            )
            report = evaluator.evaluate(
                mols,
                recipes,
                pocket_pdb_path=str(pocket) if pocket.exists() else None,
            )
            report["target_id"] = target_id
            method_records.append(report)

        results["methods"][method] = {
            "summary": aggregate(method_records),
            "per_target": method_records,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        method: data["summary"]
        for method, data in results["methods"].items()
    }, indent=2))
    print(f"Full comparative report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
