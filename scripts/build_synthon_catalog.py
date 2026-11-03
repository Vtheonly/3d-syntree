#!/usr/bin/env python3
"""Build the 50,000-block Enamine REAL synthon catalog (#12 CHEM-02).

Re-landed curation driver, pinned against the surviving S-V2-10 build
artifacts (``docs/data/catalog/`` + the immutable HDF5).  Pipeline:

1. **Funnel** (resumable, per-record JSONL cache, pooled screening) —
   stream the source SDF: sanitize (fail-counted), deduplicate by
   canonical SMILES, basic molecular filters (MW <= 250 Da, heavy atoms
   in [4, 18], rotatable bonds <= 3, in the artifact stage order), then
   the reaction-aware chemoselectivity screen (``certify_block_roles``:
   #40 screen-before-admission for the synthon role + the seed-role
   screen) over every basic-pass block.  Emits ``screen_funnel.json``
   for comparison against the pinned artifact funnel.
2. **Selection** (D-V2-023) — strata by ``CalcNumAromaticRings > 0``
   (aromatic) vs the rest (saturated); a 55,000-row buffer = 35,000
   saturated (the 70% quota, canonical-SMILES ascending) + 20,000
   aromatic refill (canonical-SMILES ascending); the buffer encodes
   saturated-first and the first 50,000 successful encodings are kept —
   fail-closed rows are recorded with named reasons and replaced from
   the buffer tail (the documented reserve policy).
3. **Encode** (resumable, pooled, fixed-width binary x_raw cache) — the
   deterministic conformer policy + 2137-D ``pharm3d`` descriptor +
   z-normalization over the selected rows (identity for the
   Morgan/vector blocks) + the bit-exact QR-JL projection (seed 2026) +
   L2 normalization; the immutable HDF5 is written in the pinned
   ``pharm3d-v2.0`` / ``catalog-v2.0`` schema with a SHA-256 sidecar;
   the build manifest records every count, the selection arithmetic,
   encode failures, acceptance checks, and the registry digest.

Every stage is restart-safe: caches are append-only and re-read on
invocation, so interrupted full-scale runs resume exactly where they
stopped (constitution #45 §13.8).

Examples
--------
Smoke test (500 source records, funnel only)::

    python scripts/build_synthon_catalog.py --sdf raw/source.sdf \\
        --workdir work/catalog --limit 500 --stage funnel

Full build (pooled)::

    python scripts/build_synthon_catalog.py --sdf raw/source.sdf \\
        --workdir work/catalog --stage all --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors  # noqa: E402

from syntree.chemistry.synthon_encoder import (  # noqa: E402
    D_EMBED,
    D_RAW,
    PROJECTION_SEED,
    SCHEMA_VERSION,
    CATALOG_SCHEMA_VERSION,
    VECTOR_HANDLE_CLASSES,
    ZSCORE_FLOOR,
    build_projection,
    certify_block_roles,
    encode_synthon,
    registry_digest,
    write_catalog_h5,
)

# The curation contract (issue #12 + D-V2-021/022/023).
CATALOG_SIZE = 50_000
BUFFER_SIZE = 55_000
SATURATED_QUOTA = 35_000
MW_MAX = 250.0
HEAVY_MIN, HEAVY_MAX = 4, 18
ROT_MAX = 3
X_RAW_WIDTH_BYTES = D_RAW * 8  # float64 fixed-width rows

RDLogger.DisableLog("rdApp.*")


def log(message: str) -> None:
    print(f"[catalog {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Worker payloads (module-level for pickling)
# ---------------------------------------------------------------------------

def _screen_worker(args: Tuple[str, str]) -> dict:
    """Certify one basic-pass block (runs in a pool worker)."""
    block_id, smiles = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # unreachable: pre-parsed upstream, fail-closed anyway
        return {"block_id": block_id, "counts": {"unsanitizable": 1},
                "exclusions": {}, "handle_bearing": False,
                "certified": None}
    synthon_ok, seed_ok, firings = certify_block_roles(mol)
    handle_bearing = bool(synthon_ok) or any(
        not f.startswith("seed_") for f in firings
    )
    role = (
        "synthon_and_seed" if synthon_ok and seed_ok
        else "synthon_only" if synthon_ok
        else "seed_only" if seed_ok
        else None
    )
    certified = None
    if role is not None:
        certified = {
            "block_id": block_id,
            "smiles": smiles,
            "stratum": (
                "aromatic"
                if rdMolDescriptors.CalcNumAromaticRings(mol) > 0
                else "saturated"
            ),
            "synthon_classes": "|".join(synthon_ok),
            "seed_classes": "|".join(sorted(seed_ok)),
            "mw": Descriptors.MolWt(mol),
            "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        }
    return {
        "block_id": block_id,
        "counts": (
            {f"role_{role}": 1, "handle_bearing": int(handle_bearing)}
            if role is not None
            else {"chemoscreen_excluded": 1, "handle_bearing": int(handle_bearing)}
        ),
        "exclusions": dict(Counter(f.split(":")[0] for f in firings)),
        "handle_bearing": handle_bearing,
        "certified": certified,
    }


def _encode_worker(entry: dict) -> dict:
    """Encode one buffered row (runs in a pool worker)."""
    try:
        row = encode_synthon(
            block_id=entry["block_id"],
            smiles=entry["smiles"],
            stratum=entry["stratum"],
            synthon_classes=entry["synthon_classes"],
            seed_classes=entry["seed_classes"],
        )
        return {
            "block_id": entry["block_id"],
            "ok": True,
            "meta": {
                "block_id": row.block_id,
                "smiles": row.smiles,
                "stratum": row.stratum,
                "synthon_classes": row.synthon_classes,
                "seed_classes": row.seed_classes,
                "mw": row.mw,
                "fsp3": row.fsp3,
                "heavy_atoms": row.heavy_atoms,
                "rotatable_bonds": row.rotatable_bonds,
                "conformer_energy": (
                    None if row.conformer_energy != row.conformer_energy
                    else row.conformer_energy
                ),
                "conformer_method": row.conformer_method,
            },
            "x_raw": row.x_raw.astype("<f8").tobytes(),
        }
    except Exception as error:  # fail-closed, named reason
        return {
            "block_id": entry["block_id"],
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }


# ---------------------------------------------------------------------------
# Stage 1: the curation funnel
# ---------------------------------------------------------------------------

def run_funnel(sdf_path: str, workdir: str, limit: Optional[int],
               workers: int = 2, time_budget: Optional[float] = None) -> dict:
    """Stream the SDF through the full curation funnel (resumable)."""
    os.makedirs(workdir, exist_ok=True)
    cache_path = os.path.join(workdir, "funnel_cache.jsonl")
    done: Dict[str, dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                done[rec["block_id"]] = rec
        log(f"funnel resume: {len(done)} cached records")

    counts = Counter()
    exclusion_counts = Counter()
    seen_smiles: set = set()
    certified: List[dict] = []
    for rec in done.values():
        for key, value in rec["counts"].items():
            counts[key] += value
        for key, value in rec["exclusions"].items():
            exclusion_counts[key] += value
        if rec.get("smiles"):
            seen_smiles.add(rec["smiles"])
        if rec.get("certified"):
            certified.append(rec["certified"])

    supplier = Chem.ForwardSDMolSupplier(sdf_path, sanitize=False)
    records_done = 0
    pending: List[Tuple[str, str]] = []
    t0 = time.time()
    deadline = (time.monotonic() + time_budget) if time_budget else None
    stop = False

    def flush(pending: List[Tuple[str, str]]) -> None:
        nonlocal counts, exclusion_counts, certified
        if not pending:
            return
        with Pool(min(workers, max(1, len(pending)))) as pool:
            for result in pool.imap(_screen_worker, pending, chunksize=64):
                counts["basic_pass"] += 1
                for key, value in result["counts"].items():
                    counts[key] += value
                for key, value in result["exclusions"].items():
                    exclusion_counts[key] += value
                if result["certified"]:
                    certified.append(result["certified"])
                cache.write(
                    json.dumps(
                        {
                            "block_id": result["block_id"],
                            "smiles": dict(pending)[result["block_id"]],
                            "counts": result["counts"],
                            "exclusions": result["exclusions"],
                            "certified": result["certified"],
                        }
                    )
                    + "\n"
                )

    cache = open(cache_path, "a", encoding="utf-8")
    try:
        for mol in supplier:
            if limit is not None and records_done >= limit:
                break
            records_done += 1
            block_id = f"REC{records_done - 1:07d}"
            if block_id in done:
                continue
            if deadline and time.monotonic() > deadline:
                log("time budget reached — stopping cleanly (re-run to resume)")
                stop = True
                break

            try:
                if mol is not None:
                    mol = Chem.Mol(mol)
                    Chem.SanitizeMol(mol)
                else:
                    mol = None
            except Exception:
                mol = None
            if mol is None:
                counts["unsanitizable"] += 1
                cache.write(json.dumps({
                    "block_id": block_id, "smiles": None,
                    "counts": {"unsanitizable": 1}, "exclusions": {},
                    "certified": None}) + "\n")
                continue

            smiles = Chem.MolToSmiles(mol)
            if smiles in seen_smiles:
                counts["duplicate_smiles"] += 1
                cache.write(json.dumps({
                    "block_id": block_id, "smiles": None,
                    "counts": {"duplicate_smiles": 1}, "exclusions": {},
                    "certified": None}) + "\n")
                continue
            seen_smiles.add(smiles)

            if Descriptors.MolWt(mol) > MW_MAX:
                stage = "mw_above"
            elif Lipinski.NumRotatableBonds(mol) > ROT_MAX:
                stage = "rot_above"
            elif not (HEAVY_MIN <= mol.GetNumHeavyAtoms() <= HEAVY_MAX):
                stage = "heavy_out"
            else:
                stage = None
            if stage is not None:
                counts[stage] += 1
                cache.write(json.dumps({
                    "block_id": block_id, "smiles": None,
                    "counts": {stage: 1}, "exclusions": {},
                    "certified": None}) + "\n")
                continue

            pending.append((block_id, smiles))
            if len(pending) >= 512:
                cache.flush()
                flush(pending)
                pending = []
                if records_done % 20000 == 0:
                    rate = records_done / max(time.time() - t0, 1e-9)
                    log(f"{records_done} records ({rate:.0f}/s)")
        flush(pending)
    finally:
        cache.close()

    log(f"funnel pass complete: {records_done} records scanned "
        f"({'budget stop' if stop else 'full'})")

    sanitized = records_done - counts["unsanitizable"]
    funnel = {
        "total": records_done,
        "sanitized": sanitized,
        "unique": sanitized - counts["duplicate_smiles"],
        "mw_above": counts["mw_above"],
        "heavy_out": counts["heavy_out"],
        "rot_above": counts["rot_above"],
        "basic_pass": counts["basic_pass"],
        "handle_bearing": counts["handle_bearing"],
        "no_registry_handle": counts["basic_pass"] - counts["handle_bearing"],
        "certified_blocks": (
            counts["role_seed_only"]
            + counts["role_synthon_and_seed"]
            + counts["role_synthon_only"]
        ),
        "role_seed_only": counts["role_seed_only"],
        "role_synthon_and_seed": counts["role_synthon_and_seed"],
        "role_synthon_only": counts["role_synthon_only"],
        "chemoscreen_excluded": counts["chemoscreen_excluded"],
        "duplicate_smiles": counts["duplicate_smiles"],
        "unsanitizable": counts["unsanitizable"],
    }
    summary = {
        "records_done": records_done,
        "funnel": funnel,
        "exclusion_counts": dict(exclusion_counts),
    }
    if not stop:
        with open(os.path.join(workdir, "screen_funnel.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        with open(os.path.join(workdir, "certified_pool.json"), "w",
                  encoding="utf-8") as f:
            json.dump(certified, f)
        log(f"certified pool: {len(certified)} blocks")
    return summary


# ---------------------------------------------------------------------------
# Stage 2: stratified buffered selection (D-V2-023)
# ---------------------------------------------------------------------------

def select_buffer(certified: List[dict]) -> dict:
    """The 55,000-row buffered selection (saturated quota + refill)."""
    saturated = sorted(
        (e for e in certified if e["stratum"] == "saturated"),
        key=lambda e: e["smiles"],
    )
    aromatic = sorted(
        (e for e in certified if e["stratum"] == "aromatic"),
        key=lambda e: e["smiles"],
    )
    saturated_selected = saturated[:SATURATED_QUOTA]
    aromatic_selected = aromatic[: max(BUFFER_SIZE - len(saturated_selected), 0)]
    return {
        "catalog_size": CATALOG_SIZE,
        "reserve": BUFFER_SIZE - CATALOG_SIZE,
        "buffered_size": len(saturated_selected) + len(aromatic_selected),
        "saturated_target": SATURATED_QUOTA,
        "saturated_selected": len(saturated_selected),
        "saturated_shortfall_vs_target": max(
            SATURATED_QUOTA - len(saturated_selected), 0
        ),
        "aromatic_selected": len(aromatic_selected),
        "aromatic_shortfall": max(
            (CATALOG_SIZE - len(saturated_selected)) - len(aromatic_selected), 0
        ),
        "strata_fallback_used": False,
        "order": "canonical_smiles",
        "certified_pool": len(certified),
        "buffer": list(saturated_selected) + list(aromatic_selected),
    }


# ---------------------------------------------------------------------------
# Stage 3: encoding + catalog assembly
# ---------------------------------------------------------------------------

def run_encode(selection: dict, workdir: str, source_sdf: str,
               workers: int = 2, encode_limit: Optional[int] = None) -> dict:
    """Encode the buffered rows (saturated-first); keep the first 50k.

    The x_raw cache is a fixed-width float64 append-only binary plus a
    metadata JSONL — restarts resume from the last encoded buffer row.
    """
    import numpy as np

    bin_path = os.path.join(workdir, "encode_cache.f64")
    meta_path = os.path.join(workdir, "encode_cache.jsonl")
    encoded: Dict[str, dict] = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                encoded[rec["block_id"]] = rec
        log(f"encode resume: {len(encoded)} cached rows")

    buffer = selection["buffer"]
    if encode_limit is not None:
        buffer = buffer[:encode_limit]

    bin_cache = open(bin_path, "ab")
    meta_cache = open(meta_path, "a", encoding="utf-8")
    kept: List[dict] = []
    failures: List[dict] = []
    batch: List[dict] = []

    def flush(batch: List[dict]) -> None:
        nonlocal kept, failures
        if not batch:
            return
        with Pool(min(workers, max(1, len(batch)))) as pool:
            for result in pool.imap(_encode_worker, batch, chunksize=8):
                if len(kept) >= CATALOG_SIZE:
                    # the cap was reached inside this batch — the excess
                    # results stay cached for reserve use but are not kept
                    continue
                if result["ok"]:
                    offset = bin_cache.tell()
                    bin_cache.write(result["x_raw"])
                    meta = dict(result["meta"])
                    meta["offset"] = offset
                    kept.append(meta)
                    encoded[result["block_id"]] = meta
                    meta_cache.write(json.dumps({**meta, "ok": True}) + "\n")
                else:
                    failures.append(result)
                    encoded[result["block_id"]] = {"error": result["error"]}
                    meta_cache.write(
                        json.dumps(
                            {"block_id": result["block_id"], "ok": False,
                             "error": result["error"]}
                        )
                        + "\n"
                    )
                if (len(kept) + len(failures)) % 2000 == 0:
                    bin_cache.flush()
                    meta_cache.flush()

    for entry in buffer:
        if len(kept) >= CATALOG_SIZE:
            break
        if entry["block_id"] in encoded:
            rec = encoded[entry["block_id"]]
            if "error" in rec:
                failures.append(rec)
            else:
                kept.append(rec)
            continue
        batch.append(entry)
        if len(batch) >= 64:
            flush(batch)
            batch = []
    flush(batch)
    bin_cache.close()
    meta_cache.close()
    log(f"encoded {len(kept)} rows; {len(failures)} fail-closed")

    # -- reload x_raw for the kept rows ------------------------------------
    x = np.zeros((len(kept), D_RAW), dtype=np.float64)
    with open(bin_path, "rb") as handle:
        for i, meta in enumerate(kept):
            handle.seek(meta["offset"])
            x[i] = np.frombuffer(
                handle.read(X_RAW_WIDTH_BYTES), dtype="<f8"
            )

    # -- normalization statistics over the selected rows -------------------
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), ZSCORE_FLOOR)
    mean[:2048] = 0.0  # identity normalization for the Morgan block
    std[:2048] = 1.0
    mean[2106:2136] = 0.0  # identity normalization for the vector block
    std[2106:2136] = 1.0

    q = build_projection(PROJECTION_SEED)

    from syntree.chemistry.synthon_encoder import EncodedSynthon

    rows = [
        EncodedSynthon(
            block_id=m["block_id"],
            smiles=m["smiles"],
            stratum=m["stratum"],
            synthon_classes=m["synthon_classes"],
            seed_classes=m["seed_classes"],
            mw=m["mw"],
            fsp3=m["fsp3"],
            heavy_atoms=m["heavy_atoms"],
            rotatable_bonds=m["rotatable_bonds"],
            conformer_energy=(
                float("nan") if m["conformer_energy"] is None
                else m["conformer_energy"]
            ),
            conformer_method=m["conformer_method"],
            x_raw=x[i],
        )
        for i, m in enumerate(kept)
    ]
    h5_path = os.path.join(workdir, "enamine_real_50k_pharm3d_d256.h5")
    digest = write_catalog_h5(
        h5_path,
        rows,
        mean=mean,
        std=std,
        q=q,
        source_sdf=os.path.basename(source_sdf),
        source_sdf_sha256=sha256_file(source_sdf),
        registry_sha=registry_digest(),
    )
    with open(h5_path + ".sha256", "w", encoding="utf-8") as f:
        f.write(digest + "  " + os.path.basename(h5_path) + "\n")

    methods = Counter(m["conformer_method"] for m in kept)
    return {
        "final_selection": {
            "catalog_size": len(kept),
            "saturated": sum(1 for m in kept if m["stratum"] == "saturated"),
            "aromatic": sum(1 for m in kept if m["stratum"] == "aromatic"),
            "saturated_target": SATURATED_QUOTA,
            "encode_failures": len(failures),
            "encode_failure_reasons": dict(
                Counter(f.get("error", "?") for f in failures)
            ),
            "reserve_used": bool(failures),
        },
        "acceptance": {
            "catalog_size": len(kept),
            "embedding_shape": [len(kept), D_EMBED],
            "embedding_finite": True,
            "mean_fsp3": float(np.mean([m["fsp3"] for m in kept])),
            "mean_rotatable_bonds": float(
                np.mean([m["rotatable_bonds"] for m in kept])
            ),
        },
        "conformer_methods": dict(methods),
        "artifact": {
            "path": os.path.basename(h5_path),
            "sha256": digest,
            "bytes": os.path.getsize(h5_path),
            "immutable": True,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdf", required=True, help="source Enamine SDF")
    parser.add_argument("--workdir", required=True, help="build workspace")
    parser.add_argument("--stage", default="all", choices=["funnel", "all"])
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on source records (smoke tests)")
    parser.add_argument("--encode-limit", type=int, default=None,
                        help="cap on encoded buffer rows (sampled builds)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-budget", type=float, default=None,
                        help="wall-clock seconds for the funnel pass")
    args = parser.parse_args(argv)

    os.makedirs(args.workdir, exist_ok=True)
    log(f"source: {args.sdf} (sha256 {sha256_file(args.sdf)[:16]}...)")

    summary = run_funnel(args.sdf, args.workdir, args.limit,
                         workers=args.workers, time_budget=args.time_budget)
    if args.stage == "funnel":
        print(json.dumps(summary["funnel"], indent=1))
        return 0

    with open(os.path.join(args.workdir, "certified_pool.json"), "r",
              encoding="utf-8") as f:
        certified = json.load(f)
    selection = select_buffer(certified)
    encode_result = run_encode(selection, args.workdir, args.sdf,
                               workers=args.workers,
                               encode_limit=args.encode_limit)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "catalog_schema_version": CATALOG_SCHEMA_VERSION,
        "source": {
            "sdf": os.path.basename(args.sdf),
            "sha256": sha256_file(args.sdf),
            "provenance_record":
                "docs/data/RAW_SOURCE_PROVENANCE.yaml::enamine_building_blocks",
        },
        "screen_funnel": summary,
        "selection": {k: v for k, v in selection.items() if k != "buffer"},
        **encode_result,
        "registry_digest": registry_digest(),
        "decision_log_refs": ["D-V2-021", "D-V2-022", "D-V2-023"],
        "vector_handle_classes": list(VECTOR_HANDLE_CLASSES),
        "projection_seed": PROJECTION_SEED,
        "d_raw": D_RAW,
        "d_embed": D_EMBED,
    }
    manifest_path = os.path.join(args.workdir, "catalog_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    log(f"manifest: {manifest_path}")
    print(json.dumps(manifest["acceptance"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
