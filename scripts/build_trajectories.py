#!/usr/bin/env python3
"""Trajectory extraction driver (#18 DATA-03, issue-pinned script).

Ingests the processed complexes from ``processed_manifest.jsonl`` with
their split assignment from ``unified_splits_30seq_id.parquet`` (#17),
extracts verified assembly trajectories through the #18 extraction
engine (the #41 index for candidate discovery, exact RDKit forward
replay as the final acceptance test), and writes the issue-pinned
``trajectories_{train,val,test}.pt`` outputs.

Resume + bounded sessions (constitution #45 §13.8)
--------------------------------------------------
* Every processed complex (accepted trajectory or explicit failure) is
  appended to the per-split accumulators ``trajectories_{split}.jsonl``
  and the failure log ``extraction_failures.jsonl`` and skipped on
  re-run — interrupted work continues without repetition.
* ``--max-seconds`` bounds each invocation (the sandbox recycles long
  calls; the driver is re-invoked until done).
* ``--finalize`` assembles the ``.pt`` outputs + the run manifest with
  measured throughput and the full failure-code histogram.

Version fail-closed: the index is loaded against the exact catalog HDF5
(sha256) and the live certified registry digest; mismatches abort before
any chemistry runs.

Example
-------
    python scripts/build_trajectories.py \\
        --manifest work/data/processed/processed_manifest.jsonl \\
        --splits work/data/splits/unified_splits_30seq_id.parquet \\
        --catalog work/catalog/enamine_real_50k_pharm3d_d256.h5 \\
        --index work/catalog/synthon_index.json \\
        --out-dir work/data/trajectories --max-seconds 540
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402

from syntree.chemistry.synthon_index import SynthonIndex, SynthonIndexError  # noqa: E402
from syntree.data.trajectories import (  # noqa: E402
    CatalogView,
    ExtractionFailure,
    TrajectoryRecord,
    derive_junction_specs,
    extract_trajectory,
    trajectory_to_dict,
)

RDLogger.DisableLog("rdApp.warning")


def log(message: str) -> None:
    print(f"[build_trajectories] {message}", flush=True)


def read_ligand(path: str) -> Optional[Chem.Mol]:
    """Parse the first molecule of a (possibly gzipped) ligand SDF."""
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="strict") as handle:
            text = handle.read()
    except OSError:
        return None
    supplier = Chem.SDMolSupplier()
    supplier.SetData(text, sanitize=True, removeHs=True, strictParsing=True)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def iter_manifest_records(path: str) -> Iterator[Dict[str, object]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class Accumulators:
    """Append-only per-split + failure logs with resume bookkeeping."""

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.paths = {
            split: os.path.join(out_dir, f"trajectories_{split}.jsonl")
            for split in ("train", "val", "test")
        }
        self.failure_path = os.path.join(out_dir, "extraction_failures.jsonl")
        self.done: Dict[str, str] = {}  # complex_id -> 'ok' | 'failed'
        self.counts: Dict[str, int] = {}
        for split, path in self.paths.items():
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            record = json.loads(line)
                            self.done[record["complex_id"]] = "ok"
                            self.counts[f"ok_{split}"] = self.counts.get(f"ok_{split}", 0) + 1
        if os.path.exists(self.failure_path):
            with open(self.failure_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self.done[record["complex_id"]] = "failed"
                        self.counts[record["failure_code"]] = self.counts.get(record["failure_code"], 0) + 1

    def append_ok(self, record: TrajectoryRecord) -> None:
        with open(self.paths[record.split], "a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory_to_dict(record), sort_keys=True) + "\n")
        self.done[record.complex_id] = "ok"
        self.counts[f"ok_{record.split}"] = self.counts.get(f"ok_{record.split}", 0) + 1

    def append_failure(self, failure: ExtractionFailure) -> None:
        with open(self.failure_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "complex_id": failure.complex_id,
                        "split": failure.split,
                        "ligand_smiles": failure.ligand_smiles,
                        "failure_code": failure.failure_code,
                        "detail": failure.detail,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        self.done[failure.complex_id] = "failed"
        self.counts[failure.failure_code] = self.counts.get(failure.failure_code, 0) + 1


def finalize(accumulators: Accumulators, out_dir: str, started: float, extra: Dict[str, object]) -> None:
    """Assemble the .pt outputs + the run manifest from the accumulators."""
    import torch

    summary: Dict[str, object] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": dict(sorted(accumulators.counts.items())),
        "totals": {
            "processed": len(accumulators.done),
            "accepted": sum(v for k, v in accumulators.counts.items() if k.startswith("ok_")),
            "failed": sum(1 for v in accumulators.done.values() if v == "failed"),
        },
    }
    for split in ("train", "val", "test"):
        path = accumulators.paths[split]
        records: List[Dict[str, object]] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        out_path = os.path.join(out_dir, f"trajectories_{split}.pt")
        torch.save(records, out_path)
        summary[f"trajectories_{split}"] = {
            "records": len(records),
            "bytes": os.path.getsize(out_path),
        }
        log(f"trajectories_{split}.pt: {len(records)} records")
    with open(os.path.join(out_dir, "extraction_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(summary | extra, handle, indent=2, sort_keys=True)
    log(f"manifest: {json.dumps(summary['totals'], sort_keys=True)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="process at most N pending complexes this run")
    parser.add_argument("--max-seconds", type=float, default=None, help="bounded run time")
    parser.add_argument("--finalize", action="store_true", help="assemble the .pt outputs and exit")
    parser.add_argument("--require-complete", action="store_true", help="finalize only when no pending complexes remain")
    args = parser.parse_args(argv)

    started = time.time()
    accumulators = Accumulators(args.out_dir)

    # -- version fail-closed: index <-> catalog <-> registry ----------------
    index = SynthonIndex.load(args.index, catalog_h5_path=args.catalog)
    import h5py

    with h5py.File(args.catalog, "r") as h5:
        columns = {
            key: [
                v.decode("utf-8") if isinstance(v, bytes) else str(v)
                for v in h5[f"catalog/{key}"][()]
            ]
            for key in ("block_ids", "smiles")
        }
    if tuple(columns["block_ids"]) != index.block_ids:
        raise SynthonIndexError("catalog view and index are not row-aligned (fail-closed)")
    catalog = CatalogView(
        block_ids=tuple(columns["block_ids"]), smiles=tuple(columns["smiles"])
    )
    log(f"index verified against catalog sha + live registry; {index.catalog_size} rows")

    specs = derive_junction_specs()
    log(f"graft specs derived: {len(specs)} reactions (cross-pair verified)")

    # -- split assignment ----------------------------------------------------
    import pandas as pd

    split_table = pd.read_parquet(args.splits)
    split_of = dict(zip(split_table["complex_id"], split_table["split"]))
    log(f"split table: {len(split_of)} complexes "
        f"({ {s: int((split_table['split'] == s).sum()) for s in ('train','val','test')} })")

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    pending = 0
    processed = 0
    accepted = 0
    no_conformer_rejections = 0
    time_bounded = args.max_seconds is not None

    for record in iter_manifest_records(args.manifest):
        complex_id = str(record["complex_id"])
        if str(record.get("status")) != "accepted":
            continue
        if complex_id in accumulators.done:
            continue
        split = split_of.get(complex_id)
        if split is None:
            # a manifest complex without a split assignment: the split
            # table is authoritative (#17); skip with an explicit count
            accumulators.counts["no_split_assignment"] = (
                accumulators.counts.get("no_split_assignment", 0) + 1
            )
            continue
        pending += 1
        if args.limit is not None and processed >= args.limit:
            pending -= 1
            continue
        if time_bounded and (time.time() - started) > args.max_seconds:
            break

        artifacts = record.get("artifacts") or {}
        ligand_rel = (artifacts or {}).get("ligand")
        pocket_rel = (artifacts or {}).get("pocket", "")
        if not ligand_rel:
            failure = ExtractionFailure(complex_id, split, "", "unparseable_ligand", "no ligand artifact path")
            accumulators.append_failure(failure)
            processed += 1
            continue
        ligand_path = os.path.join(manifest_dir, str(ligand_rel))
        ligand = read_ligand(ligand_path)
        outcome = extract_trajectory(
            ligand,
            complex_id=complex_id,
            split=split,
            index=index,
            catalog=catalog,
            pocket_artifact=str(pocket_rel),
            specs=specs,
        )
        if isinstance(outcome, TrajectoryRecord):
            accumulators.append_ok(outcome)
            accepted += 1
        else:
            accumulators.append_failure(outcome)
        processed += 1
        if processed % 200 == 0:
            rate = processed / max(1e-9, time.time() - started)
            log(f"  {processed} processed ({accepted} accepted) — {rate:.2f}/s")

    elapsed = time.time() - started
    log(
        f"run complete: {processed} newly processed, {accepted} accepted, "
        f"{elapsed:.1f}s ({processed / elapsed:.2f}/s measured); "
        f"total done {len(accumulators.done)}"
    )

    remaining = sum(
        1
        for record in iter_manifest_records(args.manifest)
        if str(record.get("status")) == "accepted"
        and str(record["complex_id"]) not in accumulators.done
        and str(record["complex_id"]) in split_of
    )
    log(f"pending after this run: {remaining}")

    if args.finalize:
        if args.require_complete and remaining > 0:
            log("finalize requested with --require-complete but work remains; not finalizing")
            return 1
        finalize(
            accumulators,
            args.out_dir,
            started,
            {
                "measured": {
                    "this_run_seconds": round(elapsed, 2),
                    "this_run_processed": processed,
                    "this_run_throughput_per_s": round(processed / elapsed, 3) if elapsed else None,
                    "pending_after_run": remaining,
                },
                "inputs": {
                    "manifest": os.path.basename(args.manifest),
                    "splits": os.path.basename(args.splits),
                    "catalog": os.path.basename(args.catalog),
                    "index": os.path.basename(args.index),
                    "index_sha256_pinned": index.catalog_sha256,
                },
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
