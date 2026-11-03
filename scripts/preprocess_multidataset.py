#!/usr/bin/env python3
"""Multi-source preprocessing driver (#16 DATA-01, issue-pinned script).

Standardizes the three raw sources pinned in
``docs/data/RAW_SOURCE_PROVENANCE.yaml`` into uniform
``<id>_pocket.pdb(.gz)`` / ``<id>_ligand.sdf(.gz)`` artifacts plus the
unified provenance manifest ``processed_manifest.jsonl``:

* ``crossdocked``  — CrossDocked2020 pocket10 mirror (streamed tar.gz;
  archive-level RMSD <= 1.0 A filter).
* ``pdbbind``      — PDBbind v2020 Refined mirror (zip; X-ray,
  resolution <= 2.5 A enforced per entry via the RCSB entry API).
* ``bindingmoad``  — BindingMOAD affinity annotations + structures
  fetched per PDB id from RCSB (X-ray only).

Design notes (D-V2-021)
-----------------------
* **Streaming**: the CrossDocked tar is consumed as a stream (no multi-GB
  extraction); PDBbind is read directly from the zip.  Disk-bounded runs
  may write gzip-compressed artifacts (``--gzip``); the manifest records
  the exact artifact path per complex, and ``--no-gzip`` (default)
  restores the issue's plain-text layout on roomier runtimes.
* **Resume**: every completed complex (accepted or rejected) is recorded
  in the append-only manifest and skipped on re-run; interrupted tails
  are repaired by :class:`ManifestWriter`.  MOAD PDB ids whose whole
  expected complex set is complete are skipped without re-downloading.
* **Fail-closed provenance**: source archive checksums are verified
  against the pinned provenance record before anything is processed;
  resolution/method lookups come from the RCSB entry API (cached in
  ``resolution_cache.json``); every rejection carries an exact reason
  code.  Nothing is fabricated (constitution #45 §11).

Examples
--------
Smoke test (50 complexes per source)::

    python scripts/preprocess_multidataset.py --source all --limit 50 \\
        --raw-dir raw --out-dir data/processed

Full-scale run with the disk-safe gzip layout::

    python scripts/preprocess_multidataset.py --source all \\
        --raw-dir raw --out-dir data/processed --gzip --workers 2
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syntree.data.preprocessing import (  # noqa: E402
    RESOLUTION_RE_LIMIT_ANGSTROM,
    LigandRejection,
    ManifestRecord,
    ManifestWriter,
    complex_id_for,
    ligand_from_mol_block,
    ligand_from_pdb_residue,
    parse_conect_edges,
    parse_experiment_method,
    parse_pdb_residue_lines,
    parse_resolution,
    standardize_complex,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROVENANCE_YAML = os.path.join(REPO_ROOT, "docs", "data", "RAW_SOURCE_PROVENANCE.yaml")
USER_AGENT = "3d-syntree-v2/2.0 (multi-source preprocessing; academic SBDD research)"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"


def log(message: str) -> None:
    print(f"[preprocess {time.strftime('%H:%M:%S')}] {message}", flush=True)


def md5_of_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_provenance() -> Dict[str, dict]:
    """Parse the pinned provenance record (fail closed when unreadable)."""
    import yaml

    with open(PROVENANCE_YAML, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    sources = doc.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise SystemExit(f"malformed provenance record: {PROVENANCE_YAML}")
    return sources


def verify_source_checksums(raw_dir: str, sources: Dict[str, dict]) -> Dict[str, str]:
    """Verify every pinned archive checksum; return {source: filename}."""
    usable: Dict[str, str] = {}
    for name, spec in sources.items():
        filename = spec.get("filename")
        algo = spec.get("hash_algo")
        expected = spec.get("expected_hash")
        if not filename:
            continue
        path = os.path.join(raw_dir, filename)
        if not os.path.exists(path):
            continue
        if algo and expected:
            observed = md5_of_file(path) if algo == "md5" else sha256_of_file(path)
            if observed != expected:
                raise SystemExit(
                    f"CHECKSUM MISMATCH for {name}: expected {expected}, "
                    f"observed {observed} — refusing to process (fail closed)"
                )
            log(f"verified {name}: {filename} ({algo} ok)")
        else:
            log(f"warning: {name} has no pinned checksum; using {filename} unverified")
        usable[name] = filename
    return usable


def http_get(url: str, *, timeout: int = 90, retries: int = 3) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_error}")


# ---------------------------------------------------------------------------
# RCSB entry API (resolution / method) with a persistent cache
# ---------------------------------------------------------------------------

def fetch_entry_metadata(
    pdb_ids: Sequence[str], cache_path: str, *, batch_size: int = 100
) -> Dict[str, Dict[str, object]]:
    """Resolution + method for every PDB id (RCSB GraphQL, batched).

    ``resolution_combined`` / ``exptl`` per entry; ids missing from RCSB
    map to ``{"missing": true}`` (recorded, never guessed).  The cache
    file is atomic-write JSON and is re-used across sessions.
    """
    cache: Dict[str, Dict[str, object]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            cache = json.load(handle)
    todo = [pid.upper() for pid in pdb_ids if pid.upper() not in cache]
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        query = (
            "query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id "
            "rcsb_entry_info{resolution_combined} exptl{method}}}"
        )
        body = json.dumps({"query": query, "variables": {"ids": batch}}).encode()
        request = urllib.request.Request(
            RCSB_GRAPHQL,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        payload = json.loads(urllib.request.urlopen(request, timeout=90).read())
        found = set()
        for entry in payload.get("data", {}).get("entries") or []:
            rcsb_id = entry.get("rcsb_id")
            if not rcsb_id:
                continue
            resolutions = (entry.get("rcsb_entry_info") or {}).get(
                "resolution_combined"
            )
            methods = [m.get("method") for m in entry.get("exptl") or []]
            cache[rcsb_id.upper()] = {
                "resolution": resolutions[0] if resolutions else None,
                "method": methods[0] if methods else None,
            }
            found.add(rcsb_id.upper())
        for missing in set(batch) - found:
            cache[missing] = {"resolution": None, "method": None, "missing": True}
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(cache, handle)
        os.replace(tmp, cache_path)
        log(f"resolution cache: {len(cache)} entries")
    return cache


# ---------------------------------------------------------------------------
# Worker payload (picklable dict -> manifest record dict)
# ---------------------------------------------------------------------------

def _record_from_payload(payload: dict) -> Dict[str, object]:
    """Build the common record skeleton from a payload (metadata only)."""
    return {
        "complex_id": payload["complex_id"],
        "source": payload["source"],
        "pdb_id": payload.get("pdb_id"),
        "target_id": payload.get("target_id"),
        "receptor_pdb_id": payload.get("receptor_pdb_id"),
        "method": payload.get("method"),
        "resolution_angstrom": payload.get("resolution_angstrom"),
        "rmsd_angstrom": payload.get("rmsd_angstrom"),
        "affinity_uM": payload.get("affinity_uM"),
        "provenance": list(payload.get("provenance") or []),
    }


def _process_payload(payload: dict) -> dict:
    """Worker-side standardization; returns the record as a plain dict.

    Ligand parse/filter rejections become ``rejected`` records (an exact
    reason code is a *result*, not a crash); anything unexpected raises
    and fails the run loudly.
    """
    skeleton = _record_from_payload(payload)
    try:
        if payload["ligand_kind"] == "mol_block":
            ligand = ligand_from_mol_block(
                payload["ligand_text"], payload.get("ligand_text_mol2")
            )
        elif payload["ligand_kind"] == "pdb_residue":
            conect = {
                int(k): tuple(int(d) for d in v)
                for k, v in (payload.get("ligand_conect") or {}).items()
            }
            ligand = ligand_from_pdb_residue(
                payload["ligand_lines"],
                template_smiles=payload.get("template_smiles"),
                conect=conect or None,
            )
        else:  # pragma: no cover - guarded by the producers
            raise ValueError(f"unknown ligand_kind {payload['ligand_kind']!r}")
        record = standardize_complex(
            complex_id=payload["complex_id"],
            source=payload["source"],
            protein_pdb_text=payload["protein_pdb_text"],
            ligand=ligand,
            out_dir=payload["out_dir"],
            pdb_id=payload.get("pdb_id"),
            target_id=payload.get("target_id"),
            receptor_pdb_id=payload.get("receptor_pdb_id"),
            method=payload.get("method"),
            resolution_angstrom=payload.get("resolution_angstrom"),
            rmsd_angstrom=payload.get("rmsd_angstrom"),
            affinity_uM=payload.get("affinity_uM"),
            provenance=payload.get("provenance"),
            gzip_output=payload["gzip_output"],
            radius=payload["radius"],
        )
        return json.loads(record.to_json_line())
    except LigandRejection as rejection:
        skeleton.update(
            {
                "status": "rejected",
                "reason_code": rejection.reason_code,
                "detail": str(rejection.detail)[:300],
            }
        )
        return skeleton


def _run_pool(
    payloads: Iterator[dict],
    manifest: ManifestWriter,
    *,
    workers: int,
    progress_every: int = 2000,
) -> Tuple[int, int]:
    accepted = rejected = processed = 0
    if workers <= 1:
        for payload in payloads:
            record = _process_payload(payload)
            manifest.append(ManifestRecord(**record))
            accepted += record["status"] == "accepted"
            rejected += record["status"] == "rejected"
            processed += 1
            if processed % progress_every == 0:
                log(f"processed {processed} (acc {accepted} / rej {rejected})")
    else:
        # BOUNDED submission: Executor.map would eagerly consume the whole
        # payload generator (measured: 183k x ~47KB tar payloads = GBs of
        # queued work items).  A sliding window keeps memory flat and lets
        # _time_bounded stop the producer between submissions.
        from collections import deque

        window = max(4 * workers, 8)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            inflight: deque = deque()

            def drain(one: bool = False) -> None:
                nonlocal accepted, rejected, processed
                while inflight and (one or len(inflight) >= window):
                    record = inflight.popleft().result()
                    manifest.append(ManifestRecord(**record))
                    accepted += record["status"] == "accepted"
                    rejected += record["status"] == "rejected"
                    processed += 1
                    if processed % progress_every == 0:
                        log(
                            f"processed {processed} "
                            f"(acc {accepted} / rej {rejected})"
                        )

            for payload in payloads:
                if len(inflight) >= window:
                    drain()
                inflight.append(pool.submit(_process_payload, payload))
            drain_all = True
            while inflight:
                drain(one=True)
    return accepted, rejected


def _pending_filter(
    payloads: Iterator[dict], manifest: ManifestWriter
) -> Iterator[dict]:
    for payload in payloads:
        if "__direct_record__" in payload:
            record = payload["__direct_record__"]
            if not manifest.already_processed(record["complex_id"]):
                manifest.append(ManifestRecord(**record))
            continue
        if not manifest.already_processed(payload["complex_id"]):
            yield payload


def _time_bounded(payloads: Iterator[dict], seconds: Optional[float]) -> Iterator[dict]:
    """Stop yielding once the wall-clock budget is spent (resume-safe).

    Constitution #45 §13.8: expensive preprocessing must continue across
    sessions.  The manifest already records every completed complex, so
    stopping here loses nothing — the next invocation resumes exactly
    after the last processed record.
    """
    if seconds is None:
        yield from payloads
        return
    deadline = time.monotonic() + seconds
    for payload in payloads:
        if time.monotonic() > deadline:
            log("time budget reached — stopping cleanly (re-run to resume)")
            return
        yield payload


# ---------------------------------------------------------------------------
# Source 1: CrossDocked2020 pocket10 (streamed tar.gz)
# ---------------------------------------------------------------------------

def iter_crossdocked_payloads(
    archive_path: str,
    *,
    out_dir: str,
    gzip_output: bool,
    entry_cache: Dict[str, Dict[str, object]],
    limit: Optional[int],
) -> Iterator[dict]:
    """Stream the pocket10 tar.gz; yield one payload per pocket/ligand pair.

    The tar is consumed in streaming mode (``r|gz``); members are
    buffered per target directory (tar members are directory-grouped) and
    emitted when the directory completes — bounded memory, no
    extraction.  A full pre-pass for the receptor-id inventory happens in
    :func:`main` before this generator runs.
    """
    emitted = 0
    current_dir: Optional[str] = None
    buffered: Dict[str, Dict[str, object]] = {}

    def flush_dir() -> Iterator[dict]:
        nonlocal emitted
        target = os.path.basename(current_dir or "")
        for basename in sorted(buffered):
            pocket_text = buffered[basename]["pocket"]
            ligand_text = buffered[basename]["ligand"]
            if ligand_text is None:
                continue  # unpaired member — impossible post-check, guarded
            if limit is not None and emitted >= limit:
                return
            receptor = basename.split("_")[0]
            receptor_pdb_id = (
                receptor.upper() if len(receptor) == 4 and receptor.isalnum() else None
            )
            meta = entry_cache.get(receptor_pdb_id or "", {})
            yield {
                "complex_id": complex_id_for("crossdocked", target, basename),
                "source": "crossdocked",
                "ligand_kind": "mol_block",
                "ligand_text": ligand_text,
                "protein_pdb_text": pocket_text,
                "out_dir": out_dir,
                "pdb_id": receptor_pdb_id,
                "target_id": target,
                "receptor_pdb_id": receptor_pdb_id,
                "method": meta.get("method"),
                "resolution_angstrom": meta.get("resolution"),
                "rmsd_angstrom": None,
                "affinity_uM": None,
                "provenance": [
                    f"archive member: {current_dir}/{basename}",
                    "per-pose RMSD not carried by the pinned pocket10 mirror; "
                    "archive-level RMSD<=1.0 A filter applies (provenance doc)",
                    "resolution/method = receptor crystal structure (RCSB); "
                    "the pose is docked, not crystallographic",
                    "protein_chains are pocket substrings (pocket-only mirror)",
                ],
                "gzip_output": gzip_output,
                "radius": 10.0,
            }
            emitted += 1

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            directory = os.path.dirname(member.name)
            if directory != current_dir:
                if buffered:
                    yield from flush_dir()
                buffered = {}
                current_dir = directory
            if not member.isfile():
                continue
            if limit is not None and emitted >= limit:
                return
            if member.name.endswith("_pocket10.pdb"):
                base = os.path.basename(member.name)[: -len("_pocket10.pdb")]
                slot = buffered.setdefault(base, {"pocket": "", "ligand": None})
                slot["pocket"] = archive.extractfile(member).read().decode(
                    "utf-8", "replace"
                )
            elif member.name.endswith(".sdf"):
                base = os.path.basename(member.name)[: -len(".sdf")]
                slot = buffered.setdefault(base, {"pocket": "", "ligand": None})
                slot["ligand"] = archive.extractfile(member).read().decode(
                    "utf-8", "replace"
                )
        if buffered and (limit is None or emitted < limit):
            yield from flush_dir()


def collect_crossdocked_receptors(archive_path: str) -> Set[str]:
    """One streaming pass collecting every receptor PDB id (for RCSB lookups)."""
    receptor_ids: Set[str] = set()
    with tarfile.open(archive_path, mode="r|gz") as scan:
        for member in scan:
            if member.isfile() and member.name.endswith(".sdf"):
                base = os.path.basename(member.name)[: -len(".sdf")]
                receptor = base.split("_")[0]
                if len(receptor) == 4 and receptor.isalnum():
                    receptor_ids.add(receptor.upper())
    return receptor_ids


# ---------------------------------------------------------------------------
# Source 2: PDBbind v2020 Refined (zip)
# ---------------------------------------------------------------------------

def iter_pdbbind_payloads(
    archive_path: str,
    *,
    out_dir: str,
    gzip_output: bool,
    entry_cache: Dict[str, Dict[str, object]],
    limit: Optional[int],
) -> Iterator[dict]:
    """Yield one payload per PDBbind complex (read directly from the zip)."""
    emitted = 0
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        top = next(
            (name.split("/")[0] for name in names if "/" in name), "pbpp-2020"
        )
        folders = sorted(
            {
                name.split("/")[1]
                for name in names
                if name.count("/") >= 2 and name.split("/")[0] == top
            }
        )
        for folder in folders:
            if limit is not None and emitted >= limit:
                return
            prefix = f"{top}/{folder}"
            protein_name = f"{prefix}/{folder}_protein.pdb"
            if protein_name not in names:
                continue
            protein_text = archive.read(protein_name).decode("utf-8", "replace")
            ligand_name = f"{prefix}/{folder}_ligand.sdf"
            ligand_text = (
                archive.read(ligand_name).decode("utf-8", "replace")
                if ligand_name in names
                else ""
            )
            pdb_id = folder.upper()
            meta = entry_cache.get(pdb_id, {})
            method = meta.get("method")
            resolution = meta.get("resolution")
            mol2_name = f"{prefix}/{folder}_ligand.mol2"
            mol2_text = (
                archive.read(mol2_name).decode("utf-8", "replace")
                if mol2_name in names
                else None
            )
            # issue-pinned quality gate: X-ray AND resolution <= 2.5 A,
            # re-verified per entry against RCSB (unverifiable -> reject)
            policy_rejection = None
            if method is None or "X-RAY" not in str(method).upper():
                policy_rejection = (
                    "METHOD_NOT_XRAY",
                    f"RCSB method for {pdb_id}: {method!r}",
                )
            elif resolution is None:
                policy_rejection = (
                    "RESOLUTION_MISSING",
                    f"no RCSB resolution for {pdb_id}",
                )
            elif float(resolution) > RESOLUTION_RE_LIMIT_ANGSTROM:
                policy_rejection = (
                    "RESOLUTION_ABOVE_LIMIT",
                    f"{resolution} > {RESOLUTION_RE_LIMIT_ANGSTROM} A",
                )
            if policy_rejection is not None:
                yield {
                    "__direct_record__": {
                        "complex_id": complex_id_for("pdbbind", folder),
                        "source": "pdbbind",
                        "status": "rejected",
                        "reason_code": policy_rejection[0],
                        "detail": policy_rejection[1],
                        "pdb_id": pdb_id,
                        "receptor_pdb_id": pdb_id,
                        "provenance": [
                            "PDBbind refined-set membership re-verified against "
                            "the RCSB entry API (issue #16: X-ray, <= 2.5 A)"
                        ],
                    }
                }
                emitted += 1
                continue
            yield {
                "complex_id": complex_id_for("pdbbind", folder),
                "source": "pdbbind",
                "ligand_kind": "mol_block",
                "ligand_text": ligand_text,
                "ligand_text_mol2": mol2_text,
                "protein_pdb_text": protein_text,
                "out_dir": out_dir,
                "pdb_id": pdb_id,
                "target_id": folder,
                "receptor_pdb_id": pdb_id,
                "method": meta.get("method"),
                "resolution_angstrom": meta.get("resolution"),
                "rmsd_angstrom": None,
                "affinity_uM": None,
                "provenance": [
                    f"archive member: {prefix}/",
                    "RMSD not applicable (crystallographic co-complex)",
                    "resolution/method from the RCSB entry API (mirror files "
                    "carry no headers)",
                    "PDBbind v2020 Refined Set membership (X-ray, resolution "
                    "<= 2.5 A per the refined-set criteria); re-verified "
                    "per entry against RCSB",
                ],
                "gzip_output": gzip_output,
                "radius": 10.0,
            }
            emitted += 1


# ---------------------------------------------------------------------------
# Source 3: BindingMOAD (annotations + RCSB structures)
# ---------------------------------------------------------------------------

def _moad_expected_complexes(annotations_path: str) -> Dict[str, List[dict]]:
    """Parse the final BindingMOAD binding-data CSV into per-PDB ligand specs.

    Source: ``bindingmoad_final.csv`` (the GitHub mirror
    ``Mugsie1969/BindingMOAD`` of the final BindingMOAD binding data —
    site sunset 2024-07-01, PDB snapshot 2022-01-31).  Unlike the older
    ``moad.parquet`` annotations mirror (whose ``name`` column is a
    protein-family *representative* PDB id — measured: only 13% of its
    annotation rows resolve against the named structure), this CSV
    carries the true per-row PDB id, ligand component id and the
    measured affinity.

    Dedup policy (documented, D-V2-021): one complex per (PDB, ligand
    component); multi-instance copies in symmetric chains collapse to
    the deterministic first instance by (chain, resSeq) — every instance
    is recorded in the manifest provenance.
    """
    import csv

    unit_to_uM = {
        "uM": 1.0, "nM": 1e-3, "mM": 1e3, "pM": 1e-6, "fM": 1e-9, "M": 1e6,
    }
    grouped: Dict[str, List[dict]] = {}
    with open(annotations_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pdbFile", "name", "measurement", "symbol", "magnitude", "units"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"MOAD CSV missing columns: {sorted(missing)}")
        for row in reader:
            pdb_id = (row["pdbFile"] or "").strip().upper()
            comp = (row["name"] or "").strip().upper()
            if len(pdb_id) != 4 or not comp:
                continue
            try:
                magnitude = float(row["magnitude"])
            except (TypeError, ValueError):
                continue
            units = (row["units"] or "").strip()
            measurement = (row["measurement"] or "").strip()
            if units == "M^-1":  # association constant Ka -> Kd = 1/Ka
                if magnitude <= 0:
                    continue
                affinity_uM = 1e6 / magnitude
            elif units in unit_to_uM:
                affinity_uM = magnitude * unit_to_uM[units]
            else:
                continue
            grouped.setdefault(pdb_id, []).append(
                {
                    "comp": comp,
                    "components": tuple(comp.split()),
                    "measurement": measurement,
                    "symbol": (row["symbol"] or "").strip(),
                    "raw": magnitude,
                    "units": units,
                    "affinity_uM": affinity_uM,
                }
            )
    return grouped


def fetch_chemcomp_smiles(
    comp_ids: Sequence[str],
    cache_path: str,
    *,
    workers: int = 4,
    max_seconds: Optional[float] = None,
) -> Tuple[Dict[str, str], bool]:
    """SMILES (stereo-preserving when available) per ligand component id.

    Bond-order templates for :func:`ligand_from_pdb_residue`; cached in
    an incrementally flushed JSON file so interrupted sessions resume
    without repeating lookups (constitution #45 §13.8).  Missing
    components map to ``""`` (recorded; the complex is then rejected
    fail-closed with LIGAND_BOND_PERCEPTION_FAILURE).  Returns the cache
    and whether every requested id was resolved (``False`` means the
    time budget ran out — re-run to continue).
    """
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as handle:
            cache = json.load(handle)
    todo = [c for c in comp_ids if c not in cache]
    complete = True

    def flush() -> None:
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(cache, handle)
        os.replace(tmp, cache_path)

    def fetch(comp: str) -> Tuple[str, Optional[str]]:
        """One component id -> SMILES; ``None`` = transient failure (defer).

        A definitive RCSB verdict is cached, including an honest empty
        SMILES (200 without a descriptor, or 404 = unknown component).
        Transient failures (timeouts, 5xx, rate-limiting) are retried
        with backoff and, if they persist, returned as ``None`` so they
        are *never* cached: a temporary outage must not fabricate a
        permanent ``LIGAND_BOND_PERCEPTION_FAILURE`` verdict (fail
        closed, constitution #45 §11 — record, never guess).
        """
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp}"
        # A multi-component assembly name (e.g. "HEM SO4", peptide chains)
        # has no per-component template by construction — it is rejected
        # upstream with LIGAND_MULTI_COMPONENT and must not be fetched
        # (a space in the path is a deterministic URL error, not a
        # transient failure).  Definitive empty SMILES, never deferred.
        if not comp or any(ch.isspace() for ch in comp):
            return comp, ""
        last_error: Optional[BaseException] = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=20) as response:
                    data = json.loads(response.read())
                descriptor = data.get("rcsb_chem_comp_descriptor") or {}
                return (
                    comp,
                    descriptor.get("SMILES_stereo")
                    or descriptor.get("SMILES")
                    or "",
                )
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return comp, ""  # definitive: unknown component
                last_error = error
            except Exception as error:  # noqa: BLE001 - retried below
                last_error = error
            time.sleep(1.0 * (attempt + 1))
        log(f"chemcomp fetch deferred after retries: {comp} ({last_error})")
        return comp, None

    if todo:
        log(f"chemcomp SMILES lookups: {len(todo)} components to fetch")
        deadline = None if max_seconds is None else time.monotonic() + max_seconds
        done = 0
        deferred = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for comp, smiles in pool.map(fetch, todo):
                if smiles is None:
                    deferred += 1
                    continue
                cache[comp] = smiles
                done += 1
                if done % 100 == 0:
                    flush()
                    log(f"chemcomp cache: {done}/{len(todo)} fetched")
                if deadline is not None and time.monotonic() > deadline:
                    complete = False
                    remaining = [
                        c for c in todo if c not in cache
                    ]
                    log(
                        f"chemcomp time budget reached: {done} fetched, "
                        f"{len(remaining)} deferred to the next session"
                    )
                    break
        flush()
        if deferred:
            complete = False
            log(
                f"chemcomp: {deferred} components deferred "
                f"(transient failures; re-run retries them)"
            )
        log(f"chemcomp cache: {len(cache)} components "
            f"({sum(1 for v in cache.values() if not v)} without SMILES)")
    return cache, complete


def iter_bindingmoad_payloads(
    annotations_path: str,
    *,
    out_dir: str,
    gzip_output: bool,
    limit: Optional[int],
    manifest: ManifestWriter,
    download_workers: int = 4,
    template_budget_seconds: Optional[float] = None,
) -> Iterator[dict]:
    """Yield one payload per (PDB, ligand) complex; structures from RCSB.

    Source: the final BindingMOAD binding-data CSV (true per-row PDB ids;
    see :func:`_moad_expected_complexes`).  X-ray only (issue #16:
    "X-ray complexes resolved via RCSB"); resolution is parsed from the
    structure header.  PDB ids whose whole expected complex set is
    already in the manifest are skipped without re-downloading.
    Downloads run in a small thread pool ahead of the process-pool
    consumers; bond-order templates come from the RCSB chemcomp cache.
    """
    grouped = _moad_expected_complexes(annotations_path)
    unique_groups = sum(len({spec["comp"] for spec in specs}) for specs in grouped.values())
    log(
        f"MOAD: {len(grouped)} unique PDB ids, "
        f"{unique_groups} unique (pdb, comp) complexes"
    )
    smiles_cache_path = os.path.join(out_dir, "chemcomp_smiles_cache.json")
    all_comps = sorted({spec["comp"] for specs in grouped.values() for spec in specs})
    smiles_cache, smiles_complete = fetch_chemcomp_smiles(
        all_comps, smiles_cache_path, max_seconds=template_budget_seconds
    )
    if not smiles_complete:
        log("MOAD deferred: template cache incomplete — re-run to continue")
        return

    emitted = 0
    pdb_ids = sorted(grouped)

    def expected_done(pdb_id: str) -> bool:
        for spec in grouped[pdb_id]:
            cid = complex_id_for("bindingmoad", pdb_id, spec["comp"])
            if not manifest.already_processed(cid):
                return False
        return True

    multi_total = {
        spec["comp"]
        for specs in grouped.values()
        for spec in specs
        if len(spec["components"]) > 1
    }
    if multi_total:
        log(
            f"MOAD: {len(multi_total)} multi-component ligand assemblies "
            f"deferred with LIGAND_MULTI_COMPONENT (documented limitation)"
        )

    # multi-component rejections need no structure download: record them first
    multi_emitted = 0
    for pdb_id in pdb_ids:
        for spec in grouped[pdb_id]:
            if len(spec["components"]) <= 1:
                continue
            cid = complex_id_for("bindingmoad", pdb_id, spec["comp"])
            if manifest.already_processed(cid):
                continue
            if limit is not None and emitted >= limit:
                return
            manifest.append(
                ManifestRecord(
                    complex_id=cid,
                    source="bindingmoad",
                    status="rejected",
                    reason_code="LIGAND_MULTI_COMPONENT",
                    detail=f"multi-component ligand assembly: {spec['comp']}",
                    pdb_id=pdb_id,
                    affinity_uM=spec["affinity_uM"],
                    provenance=[
                        "deferred: combined-molecule bond-order templates are "
                        "not available from the per-component chemcomp API "
                        "(documented limitation, D-V2-021)"
                    ],
                )
            )
            emitted += 1
            multi_emitted += 1
    log(f"MOAD: {multi_emitted} multi-component assemblies recorded as deferred")

    def only_multi(pdb_id: str) -> bool:
        return all(len(spec["components"]) > 1 for spec in grouped[pdb_id])

    pending_pdbs = [
        pid
        for pid in pdb_ids
        if not expected_done(pid) and not only_multi(pid)
    ]
    log(f"MOAD: {len(pending_pdbs)} PDB ids remain after the resume scan")

    def fetch_pdb(pdb_id: str) -> Tuple[str, Optional[str]]:
        try:
            text = http_get(RCSB_DOWNLOAD.format(pdb_id=pdb_id)).decode(
                "utf-8", "replace"
            )
            return pdb_id, text
        except RuntimeError:
            return pdb_id, None

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = [
            pool.submit(fetch_pdb, pid)
            for pid in pending_pdbs[: 100 + 4 * download_workers]
        ]
        next_index = len(futures)
        while futures:
            pdb_id, text = futures.pop(0).result()
            if next_index < len(pending_pdbs):
                futures.append(pool.submit(fetch_pdb, pending_pdbs[next_index]))
                next_index += 1
            if text is None:
                for comp in sorted({s["comp"] for s in grouped[pdb_id]}):
                    if limit is not None and emitted >= limit:
                        return
                    yield {
                        "__direct_record__": {
                            "complex_id": complex_id_for("bindingmoad", pdb_id, comp),
                            "source": "bindingmoad",
                            "status": "rejected",
                            "reason_code": "SOURCE_FILE_NOT_FOUND",
                            "detail": "RCSB download failed after retries",
                            "pdb_id": pdb_id,
                            "provenance": [f"structure {pdb_id} unavailable at RCSB"],
                        }
                    }
                    emitted += 1
                continue
            method = parse_experiment_method(text)
            resolution = parse_resolution(text)
            is_xray = method is not None and "X-RAY" in method.upper()
            by_comp: Dict[str, dict] = {}
            instances: Dict[str, List[str]] = {}
            for spec in grouped[pdb_id]:
                by_comp.setdefault(spec["comp"], spec)
                instances.setdefault(spec["comp"], []).append(
                    f"{spec['measurement']}{spec['symbol']}{spec['raw']}{spec['units']}"
                )
            for comp, spec in sorted(by_comp.items()):
                if limit is not None and emitted >= limit:
                    return
                if len(spec["components"]) > 1:
                    # documented limitation: multi-component ligand assemblies
                    # (peptides, oligosaccharides) need combined templates the
                    # per-component chemcomp API cannot provide — fail closed
                    yield {
                        "__direct_record__": {
                            "complex_id": complex_id_for("bindingmoad", pdb_id, comp),
                            "source": "bindingmoad",
                            "status": "rejected",
                            "reason_code": "LIGAND_MULTI_COMPONENT",
                            "detail": f"multi-component ligand assembly: {comp}",
                            "pdb_id": pdb_id,
                            "affinity_uM": spec["affinity_uM"],
                            "provenance": [
                                "deferred: combined-molecule bond-order templates "
                                "are not available from the per-component chemcomp "
                                "API (documented limitation, D-V2-021)"
                            ],
                        }
                    }
                    emitted += 1
                    continue
                # deterministic instance: first HETATM occurrence by (chain, resseq)
                residue = parse_pdb_residue_lines(text, resname=comp)
                residue.sort(key=lambda line: (line[21:22], line[22:27].strip()))
                if residue:
                    first_key = (residue[0][21:22], residue[0][22:27].strip())
                    ligand_lines = [
                        line
                        for line in residue
                        if (line[21:22], line[22:27].strip()) == first_key
                    ]
                else:
                    ligand_lines = []
                ligand_serials = [int(line[6:11]) for line in ligand_lines]
                ligand_conect = {
                    str(serial): list(edges)
                    for serial, edges in parse_conect_edges(
                        text, ligand_serials
                    ).items()
                }
                instances_note = ";".join(sorted(instances[comp])[:8])
                yield {
                    "complex_id": complex_id_for("bindingmoad", pdb_id, comp),
                    "source": "bindingmoad",
                    "ligand_kind": "pdb_residue",
                    "ligand_lines": ligand_lines,
                    "ligand_conect": ligand_conect,
                    "template_smiles": smiles_cache.get(comp) or None,
                    "protein_pdb_text": text,
                    "out_dir": out_dir,
                    "pdb_id": pdb_id,
                    "target_id": pdb_id,
                    "receptor_pdb_id": pdb_id,
                    "method": method,
                    "resolution_angstrom": resolution,
                    "rmsd_angstrom": None,
                    "affinity_uM": spec["affinity_uM"],
                    "provenance": [
                        f"RCSB structure {pdb_id}; ligand component {comp}; "
                        f"affinity measurements: {instances_note}",
                        "annotations: final BindingMOAD binding-data CSV "
                        "(GitHub mirror Mugsie1969/BindingMOAD, site sunset "
                        "2024-07-01, PDB snapshot 2022-01-31)",
                        "RMSD not applicable (crystallographic co-complex)",
                        "X-ray filter applied from the EXPDTA header; resolution "
                        "parsed from REMARK 2 when present",
                        "bond-order template from the RCSB chemcomp descriptor",
                    ],
                    "gzip_output": gzip_output,
                    "radius": 10.0,
                    "reject_method_not_xray": not is_xray,
                }
                emitted += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default="all",
        help="comma list of: crossdocked, pdbbind, bindingmoad, all",
    )
    parser.add_argument("--raw-dir", default="raw", help="directory with source files")
    parser.add_argument("--out-dir", default="data/processed", help="output directory")
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest path (default: <out-dir>/processed_manifest.jsonl)",
    )
    parser.add_argument("--workers", type=int, default=2, help="process workers")
    parser.add_argument("--gzip", action="store_true", help="gzip artifacts (disk-safe)")
    parser.add_argument(
        "--limit", type=int, default=None, help="per-source cap (smoke tests)"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip archive checksum verification"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="wall-clock seconds for processing per source (resume-safe stop)",
    )
    parser.add_argument(
        "--template-budget",
        type=float,
        default=None,
        help="seconds allowed for chemcomp template lookups per run (resume-safe)",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = args.manifest or os.path.join(
        args.out_dir, "processed_manifest.jsonl"
    )
    manifest = ManifestWriter(manifest_path)
    log(f"manifest: {manifest_path} (resume state: {manifest.counts})")

    sources_selected = (
        {"crossdocked", "pdbbind", "bindingmoad"}
        if args.source == "all"
        else {s.strip() for s in args.source.split(",")}
    )
    unknown = sources_selected - {"crossdocked", "pdbbind", "bindingmoad"}
    if unknown:
        raise SystemExit(f"unknown sources: {sorted(unknown)}")

    provenance = load_provenance()
    usable = (
        {} if args.no_verify else verify_source_checksums(args.raw_dir, provenance)
    )

    cache_path = os.path.join(args.out_dir, "resolution_cache.json")
    summary: Dict[str, object] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": sorted(sources_selected),
        "gzip": bool(args.gzip),
        "limit": args.limit,
        "raw_files": {},
    }

    # ---- PDBbind ----------------------------------------------------------
    if "pdbbind" in sources_selected:
        filename = usable.get("pdbbind_refined_2020") or "pbpp-2020.zip"
        archive = os.path.join(args.raw_dir, filename)
        if not os.path.exists(archive):
            log(f"SKIP pdbbind: {archive} not found")
        else:
            summary["raw_files"]["pdbbind"] = {
                "file": filename,
                "bytes": os.path.getsize(archive),
                "md5_verified": "pdbbind_refined_2020" in usable,
            }
            with zipfile.ZipFile(archive) as probe:
                pdb_ids = sorted(
                    {
                        name.split("/")[1]
                        for name in probe.namelist()
                        if name.count("/") >= 2
                    }
                )
            entry_cache = fetch_entry_metadata(pdb_ids, cache_path)
            log(f"pdbbind: {len(pdb_ids)} complexes in the mirror")
            payloads = _time_bounded(
                _pending_filter(
                    iter_pdbbind_payloads(
                        archive,
                        out_dir=args.out_dir,
                        gzip_output=args.gzip,
                        entry_cache=entry_cache,
                        limit=args.limit,
                    ),
                    manifest,
                ),
                args.time_budget,
            )
            accepted, rejected = _run_pool(payloads, manifest, workers=args.workers)
            log(f"pdbbind done: accepted {accepted}, rejected {rejected}")

    # ---- CrossDocked ------------------------------------------------------
    if "crossdocked" in sources_selected:
        filename = usable.get("crossdocked") or "crossdocked.tar.gz"
        archive = os.path.join(args.raw_dir, filename)
        if not os.path.exists(archive):
            log(f"SKIP crossdocked: {archive} not found")
        else:
            summary["raw_files"]["crossdocked"] = {
                "file": filename,
                "bytes": os.path.getsize(archive),
                "md5_verified": "crossdocked" in usable,
            }
            # the receptor inventory is scanned once and cached on disk;
            # resume runs skip the multi-minute streaming pre-pass
            receptors_cache = os.path.join(args.out_dir, "crossdocked_receptors.json")
            if os.path.exists(receptors_cache):
                with open(receptors_cache, "r", encoding="utf-8") as handle:
                    receptor_ids = set(json.load(handle))
                log(
                    f"crossdocked: receptor inventory cached "
                    f"({len(receptor_ids)} ids)"
                )
            else:
                log("crossdocked: scanning receptor ids (full streaming pre-pass) ...")
                receptor_ids = collect_crossdocked_receptors(archive)
                tmp = receptors_cache + ".tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(sorted(receptor_ids), handle)
                os.replace(tmp, receptors_cache)
                log(f"crossdocked: {len(receptor_ids)} unique receptor PDB ids")
            entry_cache = fetch_entry_metadata(sorted(receptor_ids), cache_path)
            payloads = _time_bounded(
                _pending_filter(
                    iter_crossdocked_payloads(
                        archive,
                        out_dir=args.out_dir,
                        gzip_output=args.gzip,
                        entry_cache=entry_cache,
                        limit=args.limit,
                    ),
                    manifest,
                ),
                args.time_budget,
            )
            accepted, rejected = _run_pool(payloads, manifest, workers=args.workers)
            log(f"crossdocked done: accepted {accepted}, rejected {rejected}")

    # ---- BindingMOAD ------------------------------------------------------
    if "bindingmoad" in sources_selected:
        filename = usable.get("bindingmoad_final") or "bindingmoad_final.csv"
        annotations = os.path.join(args.raw_dir, filename)
        if not os.path.exists(annotations):
            log(f"SKIP bindingmoad: {annotations} not found")
        else:
            summary["raw_files"]["bindingmoad"] = {
                "file": filename,
                "bytes": os.path.getsize(annotations),
                "checksum_pinned": False,
            }
            raw_payloads = iter_bindingmoad_payloads(
                annotations,
                out_dir=args.out_dir,
                gzip_output=args.gzip,
                limit=args.limit,
                manifest=manifest,
                template_budget_seconds=args.template_budget,
            )

            def moad_payloads() -> Iterator[dict]:
                for payload in raw_payloads:
                    if "__direct_record__" in payload:
                        record = payload["__direct_record__"]
                        if not manifest.already_processed(record["complex_id"]):
                            manifest.append(ManifestRecord(**record))
                        continue
                    if payload.pop("reject_method_not_xray", False):
                        cid = payload["complex_id"]
                        if not manifest.already_processed(cid):
                            manifest.append(
                                ManifestRecord(
                                    complex_id=cid,
                                    source=payload["source"],
                                    status="rejected",
                                    reason_code="METHOD_NOT_XRAY",
                                    detail=(
                                        "EXPDTA method is not X-ray: "
                                        f"{payload.get('method')}"
                                    ),
                                    pdb_id=payload.get("pdb_id"),
                                    resolution_angstrom=payload.get(
                                        "resolution_angstrom"
                                    ),
                                    provenance=payload.get("provenance"),
                                )
                            )
                        continue
                    yield payload

            accepted, rejected = _run_pool(
                _time_bounded(
                    _pending_filter(moad_payloads(), manifest), args.time_budget
                ),
                manifest,
                workers=args.workers,
                progress_every=500,
            )
            log(f"bindingmoad done: accepted {accepted}, rejected {rejected}")

    # ---- Run summary ------------------------------------------------------
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["manifest_counts"] = manifest.counts
    summary_path = os.path.join(args.out_dir, "preprocessing_run.json")
    tmp = summary_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    os.replace(tmp, summary_path)
    log(f"summary: {summary_path} -> {manifest.counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
