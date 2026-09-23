#!/usr/bin/env python3
"""Benchmark runner: full evaluation battery over generated molecules.

Wraps :class:`syntree.engine.evaluator.EvaluationPipeline` with CLI
argument parsing, loads generated SDFs + recipes, and prints a compact
summary table aligned with the project's target metrics.

Usage:
    python scripts/run_benchmarks.py --sdf-dir outputs/ \
        --pocket data/crossdocked/sample_pocket.pdb \
        --config configs/default_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description="3D-SynTree benchmark runner")
    parser.add_argument("--sdf-dir", type=str, default="./outputs",
                        help="directory containing ligand_XXX.sdf + recipe_XXX.json")
    parser.add_argument("--pocket", type=str, default=None,
                        help="pocket PDB used for docking evaluation")
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--output-dir", type=str, default="./experiments/eval")
    args = parser.parse_args()

    if not os.path.isdir(args.sdf_dir):
        print(f"error: SDF directory not found: {args.sdf_dir}", file=sys.stderr)
        return 2

    with open(args.config) as f:
        config = json.load(f)

    from rdkit import Chem

    from syntree.engine.evaluator import EvaluationPipeline

    # Load molecules and recipes.
    mols, recipes = [], []
    for name in sorted(os.listdir(args.sdf_dir)):
        if name.startswith("ligand_") and name.endswith(".sdf"):
            supplier = Chem.SDMolSupplier(os.path.join(args.sdf_dir, name), removeHs=False)
            mol = supplier[0] if len(supplier) else None
            mols.append(mol)
            recipe_path = os.path.join(
                args.sdf_dir, name.replace("ligand_", "recipe_").replace(".sdf", ".json")
            )
            recipes.append(
                json.load(open(recipe_path)) if os.path.exists(recipe_path) else None
            )

    if not mols:
        print(f"error: no ligand_*.sdf files under {args.sdf_dir}", file=sys.stderr)
        return 2

    pipeline = EvaluationPipeline(config, output_dir=args.output_dir)
    pocket = args.pocket or os.path.join(config["data"]["data_dir"], "sample_pocket.pdb")
    summary = pipeline.evaluate(
        mols, recipes, pocket_pdb_path=pocket if os.path.exists(pocket) else None
    )

    # Pretty summary.
    print("\n" + "=" * 60)
    print("3D-SynTree BENCHMARK SUMMARY")
    print("=" * 60)
    rows = [
        ("Molecules evaluated", summary.get("num_molecules", len(mols))),
        ("Chemical validity", f"{summary.get('chemical_validity', 0):.1%}"),
        ("PoseBusters pass rate", f"{summary.get('posebusters_pass_rate', 0):.1%}"),
        ("Mean Fsp3", f"{summary.get('mean_fsp3', 0):.3f}"),
        ("Fsp3 >= 0.42 rate", f"{summary.get('fsp3_ge_0.42_rate', 0):.1%}"),
        ("Mean MW (Da)", f"{summary.get('mean_mw', 0):.1f}"),
        ("Recipe completeness", f"{summary.get('recipe_completeness', 0):.1%}"),
        ("Retrosynthesis (proxy)", f"{summary.get('retrosynthetic_feasibility', 0):.1%}"),
    ]
    if "mean_vina_score" in summary:
        rows.append(("Mean docking score", f"{summary['mean_vina_score']:.2f} kcal/mol"))
    for label, value in rows:
        print(f"  {label:<26}: {value}")
    print("=" * 60)
    print(f"Full report: {summary.get('report_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
