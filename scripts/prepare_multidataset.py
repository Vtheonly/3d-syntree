#!/usr/bin/env python3
"""Normalize multiple structural datasets into one leakage-safe training set.

Input JSONL rows:
  complex_id, source, protein_pdb, ligand_sdf
Optional: protein_key, sequence, affinity_type, affinity_value, affinity_unit,
pose_rmsd, split.

When split is an external benchmark name (for example "casf2016_test"), its
protein cluster is locked to that benchmark split and excluded from train/val/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syntree.data.curation import (
    CuratedComplexRecord,
    PocketExtractionConfig,
    extract_protein_sequence,
    ligand_quality_flags,
    load_first_ligand,
    recenter_ligand,
    standardize_pocket,
    write_jsonl,
    write_ligand_sdf,
)
from syntree.data.splits import (
    assert_no_cluster_overlap,
    assign_with_locked_splits,
    cluster_with_mmseqs2,
    save_split_manifest,
    write_fasta,
)


def _load_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            required = {"complex_id", "source", "protein_pdb", "ligand_sdf"}
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare unified SBDD datasets")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", default="./data/unified")
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    args = parser.parse_args(argv)

    rows = _load_rows(args.input_manifest)
    out_root = os.path.abspath(args.output_dir)
    pair_dir = os.path.join(out_root, "pairs")
    work_dir = os.path.join(out_root, "splitting")
    os.makedirs(pair_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    normalized = []
    rejected = []
    used_ids = set()

    for row in rows:
        complex_id = str(row["complex_id"])
        source = str(row["source"])
        if complex_id in used_ids:
            rejected.append({
                "complex_id": complex_id,
                "source": source,
                "error": "duplicate complex_id",
            })
            continue

        try:
            ligand = load_first_ligand(str(row["ligand_sdf"]))
            flags = ligand_quality_flags(ligand)
            fatal = {
                "missing_3d_conformer",
                "descriptor_failure",
                "metal_containing",
                "crystallization_artifact",
                "too_small",
                "low_molecular_weight",
            }
            if fatal.intersection(flags):
                raise ValueError(
                    f"fatal quality flags: {sorted(fatal.intersection(flags))}"
                )

            protein_path = str(row["protein_pdb"])
            sequence = (
                "".join(str(row["sequence"]).split()).upper()
                if row.get("sequence")
                else extract_protein_sequence(protein_path)
            )
            if len(sequence) < 20:
                raise ValueError("protein sequence is missing or implausibly short")

            pocket_pdb = os.path.join(pair_dir, f"{complex_id}_pocket.pdb")
            ligand_out = os.path.join(pair_dir, f"{complex_id}_ligand.sdf")
            pocket_meta = standardize_pocket(
                protein_path,
                ligand,
                pocket_pdb,
                PocketExtractionConfig(radius_angstrom=args.radius, recenter=True),
            )
            write_ligand_sdf(
                recenter_ligand(ligand, pocket_meta["centroid"]),
                ligand_out,
            )

            protein_key = str(row.get("protein_key") or "")
            if not protein_key:
                protein_key = "seq:" + hashlib.sha256(sequence.encode()).hexdigest()[:24]

            normalized.append(
                CuratedComplexRecord(
                    complex_id=complex_id,
                    source=source,
                    protein_pdb=os.path.abspath(protein_path),
                    ligand_sdf=os.path.abspath(ligand_out),
                    pocket_pdb=os.path.abspath(pocket_pdb),
                    sequence=sequence,
                    protein_key=protein_key,
                    affinity_type=row.get("affinity_type"),
                    affinity_value=row.get("affinity_value"),
                    affinity_unit=row.get("affinity_unit"),
                    pose_rmsd=row.get("pose_rmsd"),
                    quality_flags=flags,
                ).to_dict()
            )
            used_ids.add(complex_id)
        except Exception as exc:
            rejected.append({
                "complex_id": complex_id,
                "source": source,
                "error": str(exc),
            })

    if not normalized:
        raise RuntimeError("No complexes survived curation")

    normalized_path = os.path.join(out_root, "normalized_manifest.jsonl")
    with open(normalized_path, "w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, sort_keys=True) + "
")

    fasta_path = os.path.join(work_dir, "proteins.fasta")
    write_fasta(normalized, fasta_path)
    clusters = cluster_with_mmseqs2(
        fasta_path,
        work_dir,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        mmseqs_bin=args.mmseqs_bin,
    )

    accepted_ids = {str(row["complex_id"]) for row in normalized}
    locked_splits = {
        str(row["complex_id"]): str(row["split"])
        for row in rows
        if row.get("split")
        and str(row["split"]) not in {"train", "val", "test"}
        and str(row["complex_id"]) in accepted_ids
    }
    assignments = assign_with_locked_splits(
        clusters,
        normalized,
        locked_splits=locked_splits,
        fractions=(0.80, 0.10, 0.10),
    )
    assert_no_cluster_overlap(clusters, assignments, normalized)

    split_path = save_split_manifest(
        os.path.join(out_root, "split_manifest.json"),
        assignments,
        clusters,
        normalized,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
    )

    final_records = []
    for row in normalized:
        final_records.append(
            CuratedComplexRecord(
                complex_id=row["complex_id"],
                source=row["source"],
                protein_pdb=row["protein_pdb"],
                ligand_sdf=row["ligand_sdf"],
                pocket_pdb=row["pocket_pdb"],
                sequence=row["sequence"],
                protein_key=row["protein_key"],
                affinity_type=row.get("affinity_type"),
                affinity_value=row.get("affinity_value"),
                affinity_unit=row.get("affinity_unit"),
                pose_rmsd=row.get("pose_rmsd"),
                quality_flags=row.get("quality_flags", []),
                split=assignments[row["complex_id"]],
            )
        )
    write_jsonl(final_records, os.path.join(out_root, "curated_manifest.jsonl"))

    with open(os.path.join(out_root, "rejection_report.json"), "w", encoding="utf-8") as handle:
        json.dump(rejected, handle, indent=2, sort_keys=True)

    print(f"[multidataset] input complexes: {len(rows)}")
    print(f"[multidataset] accepted: {len(normalized)}")
    print(f"[multidataset] rejected: {len(rejected)}")
    print(f"[multidataset] locked benchmark complexes: {len(locked_splits)}")
    print(f"[multidataset] split manifest: {split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
