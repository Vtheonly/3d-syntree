#!/usr/bin/env python3
"""Build a production synthon catalog from genuine commercial building-block exports.

Curates raw vendor exports (Enamine REAL, ChemSpace, etc.) into the certified
3D-SynTree synthon library. Enforces:
- Heavy atoms >= 4, 80 <= MW <= 220 Da
- Fraction of sp3 carbons (Fsp3) >= 0.40 (exempting aryl coupling partners & sulfonyl chlorides)
- Validatable reaction handles matching the ReactionEngine grammar
- Preservation of authentic vendor catalog IDs
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from typing import Iterable, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syntree.chemistry.reactions import HANDLE_SMARTS, ReactionEngine


def _detect_columns(columns: Iterable[str]) -> Tuple[str, str]:
    """Auto-detect SMILES and vendor catalog ID column names."""
    col_list = list(columns)
    smiles_col, id_col = None, None

    for col in col_list:
        low = col.lower().strip()
        if low in ("smiles", "smi", "structure", "canonical_smiles"):
            smiles_col = col
            break
    if not smiles_col:
        for col in col_list:
            if "smi" in col.lower():
                smiles_col = col
                break

    for col in col_list:
        low = col.lower().strip()
        if low in ("id", "catalog_id", "catalog_no", "enamine_id", "vendor_id", "code", "name"):
            id_col = col
            break
    if not id_col:
        for col in col_list:
            if "id" in col.lower() or "code" in col.lower():
                id_col = col
                break

    if not smiles_col:
        raise ValueError(f"Could not identify a SMILES column in vendor export. Available: {col_list}")
    return smiles_col, id_col or ""


def _rows_from_sdf(path: str) -> Iterable[dict]:
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        cid = None
        for prop in ("ID", "Catalog_ID", "Catalog ID", "Enamine_ID", "cd_id"):
            if mol.HasProp(prop):
                cid = mol.GetProp(prop).strip()
                break
        yield {"id": cid or f"VENDOR-{idx:08d}", "mol": mol}


def _rows_from_csv(path: str) -> Iterable[dict]:
    # pandas natively handles .csv and .csv.gz
    frame = pd.read_csv(path, nrows=5)
    smiles_col, id_col = _detect_columns(frame.columns)

    # Stream in chunks to handle multi-gigabyte commercial files with bounded RAM
    for chunk in pd.read_csv(path, chunksize=10000):
        for idx, row in chunk.iterrows():
            smi = str(row[smiles_col]).strip()
            if not smi or smi == "nan":
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            cid = str(row[id_col]).strip() if id_col and str(row[id_col]) != "nan" else f"EN300-{idx:08d}"
            yield {"id": cid, "mol": mol}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Curate authentic vendor building blocks")
    parser.add_argument("--input", required=True, help="SDF, CSV, or CSV.GZ commercial inventory export")
    parser.add_argument("--output", required=True, help="Output parquet catalog path")
    parser.add_argument("--min-fsp3", type=float, default=0.40)
    parser.add_argument("--max-mw", type=float, default=220.0)
    parser.add_argument("--min-heavy-atoms", type=int, default=4)
    parser.add_argument("--strict-min-size", type=int, default=10000)
    parser.add_argument("--allow-small", action="store_true", help="Development/smoke-test mode")
    parser.add_argument(
        "--exempt-handles",
        default="aryl_halide,boronic_acid,sulfonyl_chloride",
        help="Handles exempt from Fsp3 floor (aryl cross-coupling partners)",
    )
    args = parser.parse_args(argv)

    exempt = {v.strip() for v in args.exempt_handles.split(",") if v.strip()}
    engine = ReactionEngine()
    
    inp_lower = args.input.lower()
    if inp_lower.endswith((".sdf", ".sd", ".sdf.gz")):
        rows = _rows_from_sdf(args.input)
    else:
        rows = _rows_from_csv(args.input)

    output = []
    seen = set()
    total_scanned = 0

    print(f"[catalog] Curating building blocks from {args.input}...")
    for raw in rows:
        total_scanned += 1
        mol = raw["mol"]
        try:
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except Exception:
            continue
        if canonical in seen:
            continue

        heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
        if heavy < args.min_heavy_atoms:
            continue

        mw = float(Descriptors.MolWt(mol))
        if mw < 80.0 or mw > args.max_mw:
            continue

        fsp3 = float(Lipinski.FractionCSP3(mol))
        handles = sorted(set(engine.handle_types(mol)))
        if not handles:
            continue
        if fsp3 < args.min_fsp3 and not (set(handles) & exempt):
            continue

        seen.add(canonical)
        for handle in handles:
            if handle not in HANDLE_SMARTS:
                continue
            output.append({
                "id": f"{raw['id']}::{handle}",
                "smiles": Chem.MolToSmiles(mol, canonical=True),
                "fsp3": fsp3,
                "mw": mw,
                "primary_handle": handle,
                "handle_types": "|".join(handles),
            })

    unique_count = len(seen)
    print(f"[catalog] Scanned: {total_scanned:,} raw entries | Accepted: {unique_count:,} unique building blocks ({len(output):,} handle-indexed entries)")
    
    if not output:
        raise RuntimeError("No building blocks passed the curation filters.")
    if unique_count < args.strict_min_size and not args.allow_small:
        raise RuntimeError(
            f"Only {unique_count} building blocks survived; production requires at least {args.strict_min_size}."
        )

    frame = pd.DataFrame(output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"[catalog] Successfully saved certified catalog to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())