#!/usr/bin/env python3
"""Hashed inverted index build driver (#41 DATA-06, issue-pinned script).

Builds the versioned synthon index over the immutable Enamine REAL 50k
catalog HDF5, persists it as a byte-deterministic JSON companion artifact,
and runs the #41 acceptance benchmark protocol:

1. **Exhaustive self-retrieval** — every (catalog row, certified class)
   pair is queried through the index; the row itself must be returned
   ("no valid match is lost", exhaustively over the whole catalog).
2. **Brute-force equivalence** — a deterministic random sample of queries
   is answered by both the index and the linear-scan reference
   (:func:`brute_force_lookup` / :func:`brute_force_seed_lookup`); the
   results must be equal as sets.
3. **Measured performance** — build time, storage footprint, candidate
   reduction, lookup and exact-match throughput (measured on this
   machine; the T4 re-measurement lands with the authorized runtime of
   #36/#43 — never hardcode the audit's illustrative estimates).

The build is atomic and reproducible: identical catalog + registry inputs
produce a byte-identical artifact (verified on every run); the timestamp
lives in the sidecar manifest only.

Example
-------
    python scripts/build_synthon_index.py \\
        --catalog work/catalog/enamine_real_50k_pharm3d_d256.h5 \\
        --out work/catalog/synthon_index.json \\
        --manifest work/catalog/synthon_index_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syntree.chemistry.reactions import POLICY_REACTIONS  # noqa: E402
from syntree.chemistry.synthon_index import (  # noqa: E402
    FORMULA_HASH_VERSION,
    INDEX_SCHEMA_VERSION,
    SynthonIndex,
    SynthonIndexError,
    brute_force_lookup,
    brute_force_seed_lookup,
)


def log(message: str) -> None:
    print(f"[build_synthon_index] {message}", flush=True)


def sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path: str) -> Tuple[Dict[str, list], Dict[str, object]]:
    import h5py

    with h5py.File(path, "r") as h5:
        columns = {
            key: [
                v.decode("utf-8") if isinstance(v, bytes) else str(v)
                for v in h5[f"catalog/{key}"][()]
            ]
            for key in ("block_ids", "smiles", "synthon_classes", "seed_classes")
        }
        attrs = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in h5.attrs.items()}
    return columns, attrs


def benchmark(
    index: SynthonIndex,
    columns: Dict[str, list],
    *,
    sample_queries: int,
    seed: int,
) -> Dict[str, object]:
    """The #41 acceptance benchmark protocol (all numbers measured)."""
    rng = random.Random(seed)
    n = index.catalog_size
    parents = list(index.parents)
    synthon_classes = columns["synthon_classes"]
    seed_classes = columns["seed_classes"]

    # -- 1. exhaustive self-retrieval ------------------------------------
    t0 = time.perf_counter()
    self_checks = 0
    lost: List[str] = []
    class_to_reactions: Dict[str, List[str]] = {}
    for rid, reaction in POLICY_REACTIONS.items():
        class_to_reactions.setdefault(str(reaction.block_handle_class), []).append(rid)
    for row in range(n):
        for handle_class in (c for c in synthon_classes[row].split("|") if c):
            for rid in class_to_reactions[handle_class]:
                if row not in index.lookup(rid, parents[row]):
                    lost.append(f"row {row} ({index.block_ids[row]}) lost for {rid}")
                self_checks += 1
        for seed_class in (c for c in seed_classes[row].split("|") if c):
            if row not in index.seed_lookup(seed_class, parents[row]):
                lost.append(f"row {row} ({index.block_ids[row]}) lost for seed {seed_class}")
            self_checks += 1
    self_retrieval_seconds = time.perf_counter() - t0
    if lost:
        raise SynthonIndexError(
            f"exhaustive self-retrieval FAILED: {len(lost)} lost matches, "
            f"first: {lost[0]}"
        )
    log(f"exhaustive self-retrieval: {self_checks} checks, 0 lost "
        f"({self_retrieval_seconds:.2f} s)")

    # -- 2. brute-force equivalence on a deterministic sample -------------
    query_pool: List[Tuple[str, str]] = []  # (reaction_id or seed:<cls>, parent)
    for row in range(n):
        for handle_class in (c for c in synthon_classes[row].split("|") if c):
            for rid in class_to_reactions[handle_class]:
                query_pool.append((rid, parents[row]))
        for seed_class in (c for c in seed_classes[row].split("|") if c):
            query_pool.append((f"seed:{seed_class}", parents[row]))
    sample = rng.sample(query_pool, min(sample_queries, len(query_pool)))
    t0 = time.perf_counter()
    mismatches = 0
    for reaction_id, parent in sample:
        if reaction_id.startswith("seed:"):
            got = index.seed_lookup(reaction_id[5:], parent)
            want = brute_force_seed_lookup(
                seed_classes=seed_classes,
                parents=parents,
                seed_class=reaction_id[5:],
                query_parent_smiles=parent,
            )
        else:
            got = index.lookup(reaction_id, parent)
            want = brute_force_lookup(
                synthon_classes=synthon_classes,
                parents=parents,
                reaction_id=reaction_id,
                query_parent_smiles=parent,
            )
        if set(got) != set(want):
            mismatches += 1
            if mismatches <= 3:
                log(f"  MISMATCH {reaction_id} {parent}: indexed {got} vs brute {want}")
    brute_seconds = time.perf_counter() - t0
    if mismatches:
        raise SynthonIndexError(
            f"brute-force equivalence FAILED: {mismatches}/{len(sample)} mismatches"
        )
    log(f"brute-force equivalence: {len(sample)} sampled queries, 0 mismatches "
        f"({brute_seconds:.2f} s)")

    # -- 3. measured throughput and candidate reduction --------------------
    t0 = time.perf_counter()
    repeats = max(1, (200_000 // max(1, len(sample))) * 1)
    for _ in range(repeats):
        for reaction_id, parent in sample:
            if reaction_id.startswith("seed:"):
                index.seed_lookup(reaction_id[5:], parent)
            else:
                index.lookup(reaction_id, parent)
    lookup_ops = repeats * len(sample)
    lookup_seconds = time.perf_counter() - t0
    lookup_us = 1e6 * lookup_seconds / lookup_ops

    t0 = time.perf_counter()
    exact_ops = 0
    for _ in range(repeats):
        for reaction_id, parent in sample:
            rows = (
                index.seed_lookup(reaction_id[5:], parent)
                if reaction_id.startswith("seed:")
                else index.lookup(reaction_id, parent)
            )
            index.exact_candidates(parent, rows)
            exact_ops += 1
    exact_seconds = time.perf_counter() - t0
    exact_us = 1e6 * exact_seconds / exact_ops

    candidate_sizes = []
    for reaction_id, parent in sample:
        rows = (
            index.seed_lookup(reaction_id[5:], parent)
            if reaction_id.startswith("seed:")
            else index.lookup(reaction_id, parent)
        )
        candidate_sizes.append(len(rows))
    mean_candidates = sum(candidate_sizes) / len(candidate_sizes)
    return {
        "protocol": {
            "exhaustive_self_retrieval_checks": self_checks,
            "exhaustive_self_retrieval_seconds": round(self_retrieval_seconds, 3),
            "brute_force_sample_size": len(sample),
            "brute_force_seconds": round(brute_seconds, 3),
            "benchmark_seed": seed,
        },
        "measured": {
            "lookup_throughput_ops_per_s": round(lookup_ops / lookup_seconds, 1),
            "lookup_latency_us": round(lookup_us, 2),
            "exact_stage_throughput_ops_per_s": round(exact_ops / exact_seconds, 1),
            "exact_stage_latency_us": round(exact_us, 2),
            "mean_candidates_per_query": round(mean_candidates, 3),
            "max_candidates_per_query": max(candidate_sizes),
            "full_scan_rows": n,
            "mean_candidate_reduction_factor": round(n / mean_candidates, 1)
            if mean_candidates
            else None,
        },
        "machine_note": (
            "measured in the CPU dev sandbox (single-threaded); the supported "
            "T4 environments re-measure under #36/#43 when authorized — these "
            "are real measurements, not the audit's illustrative estimates"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, help="immutable catalog HDF5 path")
    parser.add_argument("--out", required=True, help="output index JSON path")
    parser.add_argument("--manifest", default=None, help="sidecar manifest JSON path")
    parser.add_argument("--sample-queries", type=int, default=2000)
    parser.add_argument("--benchmark-seed", type=int, default=2026)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="verify an existing index artifact (versions + benchmark) without rebuilding",
    )
    args = parser.parse_args(argv)

    columns, attrs = load_catalog(args.catalog)
    catalog_sha = sha256_of_file(args.catalog)
    log(f"catalog {args.catalog}: {len(columns['block_ids'])} rows, sha256 {catalog_sha[:12]}…")

    if args.verify_only:
        index = SynthonIndex.load(args.out, catalog_h5_path=args.catalog)
        log("version checks passed (schema, formula-hash, catalog sha, live registry)")
        bench = benchmark(index, columns, sample_queries=args.sample_queries, seed=args.benchmark_seed)
    else:
        t0 = time.perf_counter()
        index = SynthonIndex.build(
            block_ids=columns["block_ids"],
            smiles=columns["smiles"],
            synthon_classes=columns["synthon_classes"],
            seed_classes=columns["seed_classes"],
            catalog_sha256=catalog_sha,
            catalog_schema_version=attrs.get("catalog_schema_version", ""),
            catalog_registry_digest=attrs.get("registry_digest", ""),
        )
        build_seconds = time.perf_counter() - t0
        log(f"index built in {build_seconds:.2f} s: {index.stats()}")

        index_sha = index.save(args.out)
        # byte-determinism: rebuild and compare artifacts
        rebuilt = SynthonIndex.build(
            block_ids=columns["block_ids"],
            smiles=columns["smiles"],
            synthon_classes=columns["synthon_classes"],
            seed_classes=columns["seed_classes"],
            catalog_sha256=catalog_sha,
            catalog_schema_version=attrs.get("catalog_schema_version", ""),
            catalog_registry_digest=attrs.get("registry_digest", ""),
        )
        tmp = args.out + ".determinism-check"
        rebuilt_sha = rebuilt.save(tmp)
        if rebuilt_sha != index_sha:
            raise SynthonIndexError("rebuild is not byte-deterministic (fail-closed)")
        os.remove(tmp)
        log(f"byte-deterministic rebuild verified (sha256 {index_sha[:12]}…)")

        bench = benchmark(
            index, columns, sample_queries=args.sample_queries, seed=args.benchmark_seed
        )
        bench["measured"]["build_seconds"] = round(build_seconds, 3)
        bench["measured"]["artifact_bytes"] = os.path.getsize(args.out)
        bench["measured"]["artifact_sha256"] = index_sha

    if args.manifest:
        manifest = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "formula_hash_version": FORMULA_HASH_VERSION,
            "catalog_path": os.path.basename(args.catalog),
            "catalog_sha256": catalog_sha,
            "catalog_schema_version": attrs.get("catalog_schema_version", ""),
            "catalog_registry_digest_historical": attrs.get("registry_digest", ""),
            "live_registry_digest": index.registry_digest_value,
            "index_sha256": (
                bench["measured"].get("artifact_sha256")
                if "measured" in bench and "artifact_sha256" in bench.get("measured", {})
                else sha256_of_file(args.out)
            ),
            "stats": index.stats(),
            "benchmark": bench,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = args.manifest + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, args.manifest)
        log(f"manifest written: {args.manifest}")

    log("benchmark: " + json.dumps(bench.get("measured", bench), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
