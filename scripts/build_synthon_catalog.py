#!/usr/bin/env python3
"""Build a production synthon catalog from a real vendor/exported SDF or CSV.

The repository no longer fabricates an Enamine-sized catalog. Use a real
vendor/library export and run in strict mode for thesis data preparation.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syntree.chemistry.reactions import HANDLE_SMARTS, ReactionEngine


def _rows_from_sdf(path: str) -> Iterable[dict]:
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        cid = mol.GetProp("ID") if mol.HasProp("ID") else f"VENDOR-{idx:08d}"
        yield {"id": cid, "mol": mol}


def _rows_from_csv(path: str) -> Iterable[dict]:
    frame = pd.read_csv(path)
    if "smiles" not in frame.columns:
        raise ValueError("CSV catalog must contain a 'smiles' column")
    for idx, row in frame.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        yield {"id": str(row.get("id", f"VENDOR-{idx:08d}")), "mol": mol}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Curate a real building-block export")
    parser.add_argument("--input", required=True, help="SDF or CSV vendor/library export")
    parser.add_argument("--output", required=True, help="Output parquet")
    parser.add_argument("--min-fsp3", type=float, default=0.40)
    parser.add_argument("--max-mw", type=float, default=220.0)
    parser.add_argument("--min-heavy-atoms", type=int, default=4)
    parser.add_argument("--strict-min-size", type=int, default=50000)
    parser.add_argument("--allow-small", action="store_true",
                        help="development/smoke-test mode; disables the size gate")
    parser.add_argument(
        "--exempt-handles",
        default="aryl_halide,boronic_acid",
        help="comma-separated handles exempt from the Fsp3 floor",
    )
    args = parser.parse_args(argv)

    exempt = {value.strip() for value in args.exempt_handles.split(",") if value.strip()}
    engine = ReactionEngine()
    suffix = os.path.splitext(args.input)[1].lower()
    rows = _rows_from_sdf(args.input) if suffix in {".sdf", ".sd"} else _rows_from_csv(args.input)

    output = []
    seen = set()
    for raw in rows:
        mol = raw["mol"]
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        if canonical in seen:
            continue

        heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
        mw = float(Descriptors.MolWt(mol))
        fsp3 = float(Lipinski.FractionCSP3(mol))
        handles = sorted(set(engine.handle_types(mol)))
        if not handles:
            continue
        if heavy < args.min_heavy_atoms or mw < 80.0 or mw > args.max_mw:
            continue
        if fsp3 < args.min_fsp3 and not (set(handles) & exempt):
            continue

        seen.add(canonical)
        for handle in handles:
            if handle not in HANDLE_SMARTS:
                continue
            output.append(
                {
                    "id": f"{raw['id']}::{handle}",
                    "smiles": Chem.MolToSmiles(mol, canonical=True),
                    "fsp3": fsp3,
                    "mw": mw,
                    "primary_handle": handle,
                    "handle_types": "|".join(handles),
                }
            )

    unique_count = len(seen)
    if not output:
        raise RuntimeError("No real building blocks survived curation")
    if unique_count < args.strict_min_size and not args.allow_small:
        raise RuntimeError(
            f"Only {unique_count} unique building blocks survived; "
            f"production mode requires at least {args.strict_min_size}. "
            "Use --allow-small only for development."
        )

    frame = pd.DataFrame(output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(
        f"[catalog] wrote {len(frame)} handle-indexed rows from "
        f"{unique_count} unique building blocks to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
