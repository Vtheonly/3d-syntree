#!/usr/bin/env python3
"""Stage 1: sanitize and standardize CrossDocked2020, BindingMOAD and PDBbind.

The datasets themselves are not downloaded by this repository because their
distribution terms and layouts differ. Supply a source manifest or a common
directory layout. The output is a unified JSONL manifest plus 10 A protein
pockets and normalized ligand SDF files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from rdkit import Chem

from syntree.data.multidataset import (
    extract_protein_pocket,
    read_ligand,
    stable_sample_id,
    validate_ligand,
)


def _load_rows(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_pairs(source: str, root: Path) -> List[Dict]:
    proteins = {}
    ligands = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        stem = p.stem
        if p.suffix.lower() == ".pdb":
            key = stem[:-8] if stem.endswith("_protein") else stem
            proteins[key] = p
        elif p.suffix.lower() in {".sdf", ".mol2"}:
            key = stem[:-7] if stem.endswith("_ligand") else stem
            ligands[key] = p

    return [
        {
            "source": source,
            "complex_id": key,
            "protein_path": str(proteins[key]),
            "ligand_path": str(ligands[key]),
        }
        for key in sorted(set(proteins) & set(ligands))
    ]


def iter_rows(args) -> Iterable[Dict]:
    for source, manifest_arg, dir_arg in (
        ("crossdocked2020", args.crossdocked_manifest, args.crossdocked_dir),
        ("bindingmoad", args.bindingmoad_manifest, args.bindingmoad_dir),
        ("pdbbind", args.pdbbind_manifest, args.pdbbind_dir),
    ):
        if manifest_arg:
            for row in _load_rows(Path(manifest_arg)):
                record = dict(row)
                record.setdefault("source", source)
                yield record
        elif dir_arg:
            yield from discover_pairs(source, Path(dir_arg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossdocked-dir")
    parser.add_argument("--bindingmoad-dir")
    parser.add_argument("--pdbbind-dir")
    parser.add_argument("--crossdocked-manifest")
    parser.add_argument("--bindingmoad-manifest")
    parser.add_argument("--pdbbind-manifest")
    parser.add_argument("--output-dir", default="./data/multidataset")
    parser.add_argument("--manifest-out", default="./data/multidataset/processed_manifest.jsonl")
    parser.add_argument("--cutoff-radius", type=float, default=10.0)
    parser.add_argument("--min-pocket-atoms", type=int, default=100)
    parser.add_argument("--min-mw", type=float, default=150.0)
    parser.add_argument("--max-mw", type=float, default=800.0)
    parser.add_argument("--min-heavy-atoms", type=int, default=10)
    parser.add_argument("--max-resolution", type=float, default=2.5)
    parser.add_argument("--strict-resolution-metadata", action="store_true")
    parser.add_argument("--pdbbind-subset", default="refined")
    args = parser.parse_args()

    rows = list(iter_rows(args))
    if not rows:
        parser.error("Provide at least one source directory or manifest.")

    accepted, skipped = [], {}
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for row in rows:
        source = str(row.get("source", "")).strip().lower()
        complex_id = str(row.get("complex_id") or row.get("pdb_id") or "").strip()
        protein = Path(str(row.get("protein_path", "")))
        ligand_path = Path(str(row.get("ligand_path", "")))
        if not complex_id or not protein.exists() or not ligand_path.exists():
            skipped["missing_input"] = skipped.get("missing_input", 0) + 1
            continue

        raw_resolution = row.get("resolution")
        try:
            resolution = float(raw_resolution) if raw_resolution not in (None, "", "nan") else None
        except (ValueError, TypeError):
            resolution = None
        if resolution is not None and resolution > args.max_resolution:
            skipped["resolution"] = skipped.get("resolution", 0) + 1
            continue
        if args.strict_resolution_metadata and resolution is None:
            skipped["missing_resolution"] = skipped.get("missing_resolution", 0) + 1
            continue

        ligand = read_ligand(str(ligand_path))
        valid, descriptors = validate_ligand(
            ligand,
            min_mw=args.min_mw,
            max_mw=args.max_mw,
            min_heavy_atoms=args.min_heavy_atoms,
        )
        if not valid:
            skipped["ligand_filter"] = skipped.get("ligand_filter", 0) + 1
            continue

        safe_source = "".join(c if c.isalnum() or c in "-_" else "_" for c in source)
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in complex_id)
        out_dir = output_root / safe_source
        pocket_path = out_dir / f"{safe_id}_pocket.pdb"
        ligand_out = out_dir / f"{safe_id}_ligand.sdf"

        if not extract_protein_pocket(
            str(protein),
            ligand,
            str(pocket_path),
            cutoff_radius=args.cutoff_radius,
            min_atoms=args.min_pocket_atoms,
        ):
            skipped["pocket_size"] = skipped.get("pocket_size", 0) + 1
            continue

        writer = Chem.SDWriter(str(ligand_out))
        writer.write(ligand)
        writer.close()

        accepted.append({
            "sample_id": f"{source}:{complex_id}",
            "sample_hash": stable_sample_id(source, complex_id),
            "source": source,
            "complex_id": complex_id,
            "protein_path": str(protein.resolve()),
            "ligand_path": str(ligand_path.resolve()),
            "processed_pocket_path": str(pocket_path.resolve()),
            "processed_ligand_path": str(ligand_out.resolve()),
            "resolution": resolution,
            "subset": row.get("subset") or (args.pdbbind_subset if source == "pdbbind" else None),
            "mw": descriptors["mw"],
            "heavy_atoms": int(descriptors["heavy_atoms"]),
        })

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in accepted:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "accepted": len(accepted),
        "input_rows": len(rows),
        "skipped": skipped,
        "sources": sorted({r["source"] for r in accepted}),
        "pocket_cutoff_angstrom": args.cutoff_radius,
        "min_pocket_atoms": args.min_pocket_atoms,
        "ligand_filter": {
            "min_mw": args.min_mw,
            "max_mw": args.max_mw,
            "min_heavy_atoms": args.min_heavy_atoms,
        },
        "max_resolution_when_available": args.max_resolution,
        "manifest": str(manifest_path),
    }
    manifest_path.with_name("preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
