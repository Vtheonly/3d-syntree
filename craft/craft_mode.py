#!/usr/bin/env python3
"""Craft Mode: High-performance structural processing, retrosynthesis, tensor generation.

Executes strictly on Kaggle (offline or online). Auto-discovers the raw
dataset mounted from Download Mode, extracts the CrossDocked archive, runs
the verified chemistry funnel (structural QC -> chemical QC ->
reaction-constrained retrosynthetic decomposition) across a process pool,
and commits states to resumable SHA-256-verified gzip shards.

Reuse contract (see Phase 1 audit): all chemistry/tensor logic lives in
``syntree.*``; this module only orchestrates, measures and checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
import rdkit.Chem.rdFingerprintGenerator as rdFPGen
from tqdm import tqdm

from core.checkpointing import ShardCheckpointManager
from core.discovery import DatasetDiscovery
from core.logging import MeasuredDataFunnel
from core.manifest import ManifestManager
from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.curation import (
    PocketExtractionConfig,
    extract_protein_sequence,
    ligand_quality_flags,
    standardize_pocket,
)
from syntree.data.fragmenter import ReactionConstrainedFragmenter, canonical_smiles
from syntree.data.splits import (
    assert_no_cluster_overlap,
    cluster_with_mmseqs2,
    deterministic_cluster_assignments,
    save_split_manifest,
    write_fasta,
)
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder

_LIGAND_EXTENSIONS = (".sdf", ".mol2")
_HARD_REJECT_FLAGS = {
    "crystallization_artifact",
    "missing_3d_conformer",
    "descriptor_failure",
    "too_small",
    "metal_containing",
}
SPLITS = ("train", "val", "test")

DEFAULT_QC: Dict[str, Any] = {
    "structural": {
        "max_resolution_angstrom": 2.5,
        "pocket_cutoff_radius": 10.0,
        "min_pocket_atoms": 100,
    },
    "chemical": {"min_mw": 150.0, "max_mw": 800.0, "min_heavy_atoms": 10},
}

# Per-process worker state (set by _init_worker in every pool child).
_worker_builder: Optional[RetrosyntheticTrajectoryBuilder] = None
_worker_qc: Dict[str, Any] = DEFAULT_QC
_catalog_fps: list = []
_morgan_gen = None


def _resolution_from_pdb(pdb_path: str) -> Optional[float]:
    """Parse REMARK 2 RESOLUTION if present (CrossDocked PDBs may omit it)."""
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(30):
                line = fh.readline()
                if not line:
                    break
                if line.startswith("REMARK 2 RESOLUTION"):
                    parts = line.split()
                    for i, tok in enumerate(parts):
                        if tok == "RESOLUTION" and i + 1 < len(parts):
                            return float(parts[i + 1])
    except Exception:
        return None
    return None


def _init_worker(catalog_path: str, max_steps: int, qc: Dict[str, Any]) -> None:
    """Initialise catalog/builder once per pool child.

    Also installs the precomputed-Morgan-fingerprint fast path for
    ``ReactionConstrainedFragmenter._catalog_candidates``: the stock method
    re-parses every catalog SMILES per lookup, which is O(catalog) RDKit
    parses per retro step; the patch is behaviour-identical (same exact
    lookup, same 0.85 Tanimoto gate, same sort order) but runs one C++
    bulk-similarity call over precomputed fingerprints.
    """
    global _worker_builder, _worker_qc, _catalog_fps, _morgan_gen
    RDLogger.DisableLog("rdApp.*")
    _worker_qc = qc
    catalog = SynthonCatalog(catalog_path)
    _worker_builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=max_steps)
    _morgan_gen = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)
    _catalog_fps = []
    for smiles in catalog.df["smiles"]:
        mol = Chem.MolFromSmiles(str(smiles))
        _catalog_fps.append(_morgan_gen.GetFingerprint(mol) if mol else None)

    def fast_catalog_candidates(self, synthon_mol: Chem.Mol, min_tanimoto: float = 0.85):
        exact = self._catalog_lookup.get(canonical_smiles(synthon_mol), ())
        if exact:
            return tuple(sorted(exact))
        fp = _morgan_gen.GetFingerprint(synthon_mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, _catalog_fps)
        scored = [
            (sim, idx)
            for idx, sim in enumerate(sims)
            if sim >= min_tanimoto and _catalog_fps[idx] is not None
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(idx for _, idx in scored)

    ReactionConstrainedFragmenter._catalog_candidates = fast_catalog_candidates


def _load_ligand(ligand_path: str) -> Optional[Chem.Mol]:
    if ligand_path.lower().endswith(".mol2"):
        return Chem.MolFromMol2File(ligand_path, removeHs=False, sanitize=True)
    supp = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)
    return next((m for m in supp if m is not None), None)


def _process_complex(item: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """Run one complex through the full funnel; return a stage-tagged result.

    Stages (in order): ``reject_structural`` (unreadable files, missing 3D
    ligand, resolution/atom-count gates, failed 10A pocket re-trim) ->
    ``reject_chemical`` (Ro5/artifact/metal/size gates) -> ``reject_retro``
    (no valid decomposition) -> ``ok``.
    """
    pocket_path, ligand_path, cid, split = item
    result: Dict[str, Any] = {
        "stage": "reject_structural",
        "reason": "",
        "split": split,
        "cid": cid,
        "pocket": pocket_path,
        "states": None,
    }
    qc = _worker_qc
    struct_cfg = qc.get("structural", {})
    chem_cfg = qc.get("chemical", {})
    try:
        # ---- Stage 1a: load ------------------------------------------------
        pocket_mol = Chem.MolFromPDBFile(pocket_path, removeHs=False)
        ligand_mol = _load_ligand(ligand_path)
        if pocket_mol is None or ligand_mol is None:
            result["reason"] = "unreadable_pocket_or_ligand"
            return result
        if pocket_mol.GetNumAtoms() == 0 or ligand_mol.GetNumAtoms() == 0:
            result["reason"] = "zero_atoms"
            return result
        if ligand_mol.GetNumConformers() == 0:
            result["reason"] = "missing_3d_conformer"
            return result

        # ---- Stage 1b: resolution gate (when declared) ---------------------
        max_res = struct_cfg.get("max_resolution_angstrom")
        if max_res:
            resolution = _resolution_from_pdb(pocket_path)
            if resolution is not None and resolution > float(max_res):
                result["stage"] = "reject_structural"
                result["reason"] = f"resolution_{resolution}A"
                return result

        # ---- Stage 1c: 10A ligand-centroid pocket re-trim -------------------
        # recenter=False keeps the original pocket frame so the ligand's 3D
        # coordinates stay aligned with the pocket (recentering would shift
        # only the pocket and corrupt the complex geometry).
        tmp_out = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdb", delete=False
            ) as tmp:
                tmp_out = tmp.name
            standardize_pocket(
                pocket_path,
                ligand_mol,
                tmp_out,
                PocketExtractionConfig(
                    radius_angstrom=float(struct_cfg.get("pocket_cutoff_radius", 10.0)),
                    remove_hetero_atoms=True,
                    recenter=False,
                ),
            )
            std_pocket = Chem.MolFromPDBFile(tmp_out, removeHs=False)
        except Exception as exc:
            result["reason"] = f"pocket_trim_failed:{exc.__class__.__name__}"
            return result
        finally:
            if tmp_out and os.path.exists(tmp_out):
                os.unlink(tmp_out)
        if std_pocket is None or std_pocket.GetNumAtoms() == 0:
            result["reason"] = "empty_standardized_pocket"
            return result
        min_atoms = int(struct_cfg.get("min_pocket_atoms", 100))
        if std_pocket.GetNumAtoms() < min_atoms:
            result["reason"] = f"pocket_atoms_{std_pocket.GetNumAtoms()}<{min_atoms}"
            return result

        # ---- Stage 2: chemical QC ------------------------------------------
        flags = ligand_quality_flags(ligand_mol)
        hard = sorted(set(flags) & _HARD_REJECT_FLAGS)
        if hard:
            result["stage"] = "reject_chemical"
            result["reason"] = "flags:" + ",".join(hard)
            return result
        try:
            from rdkit.Chem import Descriptors

            mw = float(Descriptors.MolWt(ligand_mol))
        except Exception:
            result["stage"] = "reject_chemical"
            result["reason"] = "descriptor_failure"
            return result
        min_mw = float(chem_cfg.get("min_mw", 150.0))
        max_mw = float(chem_cfg.get("max_mw", 800.0))
        if not (min_mw <= mw <= max_mw):
            result["stage"] = "reject_chemical"
            result["reason"] = f"mw_{mw:.1f}"
            return result
        heavy = sum(1 for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1)
        if heavy < int(chem_cfg.get("min_heavy_atoms", 10)):
            result["stage"] = "reject_chemical"
            result["reason"] = f"heavy_atoms_{heavy}"
            return result

        # ---- Stage 3: retrosynthetic decomposition --------------------------
        assert _worker_builder is not None, "worker builder not initialised"
        states, meta = _worker_builder.build(
            ligand_mol, std_pocket, trajectory_id=cid
        )
        if not states:
            result["stage"] = "reject_retro"
            result["reason"] = str(meta.get("status", "no_states"))
            return result
        result["stage"] = "ok"
        result["reason"] = str(meta.get("status", "ok"))
        result["states"] = states
        return result
    except Exception as exc:
        result["stage"] = "reject_retro"
        result["reason"] = f"error:{exc.__class__.__name__}"
        return result


# ---------------------------------------------------------------------------
# Pair discovery (single-pass, longest-prefix ligand matching)
# ---------------------------------------------------------------------------

def _normalize_stem(stem: str) -> str:
    for suffix in ("_pocket10", "_pocket", "_ligand", "_lig", "_dock"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _best_ligand(pocket_stem: str, ligands: List[str]) -> str:
    """Pick the ligand file that best matches a pocket stem.

    Preference: exact normalized-stem equality, then longest shared prefix
    (CrossDocked keeps the complex id as the common prefix), then first.
    """
    p_norm = _normalize_stem(pocket_stem)
    best = ligands[0]
    best_score = -1
    for cand in ligands:
        c_norm = _normalize_stem(Path(cand).stem)
        if c_norm == p_norm:
            return cand
        common = 0
        for a, b in zip(p_norm, c_norm):
            if a != b:
                break
            common += 1
        if common > best_score:
            best_score = common
            best = cand
    return best


def scan_pairs(root_dir: Path | str) -> List[Tuple[str, str, str]]:
    """Single-pass pocket/ligand pair scan over an extracted CrossDocked tree."""
    print(f"[Discover] Single-pass scan of {root_dir}...")
    t0 = time.time()
    pairs: List[Tuple[str, str, str]] = []
    for root, _, files in os.walk(str(root_dir)):
        pockets = [f for f in files if "pocket" in f.lower() and f.endswith(".pdb")]
        if not pockets:
            continue
        ligands = sorted(f for f in files if f.endswith(_LIGAND_EXTENSIONS))
        if not ligands:
            continue
        parent_name = os.path.basename(root)
        for p in sorted(pockets):
            p_stem = p[:-4]
            best_l = _best_ligand(p_stem, ligands)
            pairs.append(
                (os.path.join(root, p), os.path.join(root, best_l), f"{parent_name}_{p_stem}")
            )
    pairs.sort(key=lambda x: x[2])
    print(f"[Discover] Found {len(pairs)} pairs in {time.time() - t0:.2f}s")
    return pairs


# ---------------------------------------------------------------------------
# Leakage-safe splits (MMseqs2 when available, identity clusters otherwise)
# ---------------------------------------------------------------------------

def _stable_cid_bucket(cid: str, seed: int) -> str:
    digest = int.from_bytes(
        hashlib.blake2b(f"{seed}:{cid}".encode(), digest_size=8).digest(), "big"
    )
    return SPLITS[digest % 3]


def _load_or_create_splits(
    pairs: List[Tuple[str, str, str]], output_dir: Path, seed: int
) -> Tuple[Dict[str, str], str, int]:
    """Return (complex_id -> split, policy label, unique protein count).

    A persisted ``split_manifest.json`` is authoritative across resumes so a
    restart can never silently reassign splits. First run clusters protein
    sequences with MMseqs2 (fail-closed in the library); when the binary is
    unavailable (typical offline Kaggle image) every distinct sequence forms
    its own identity cluster - exact-sequence leakage stays locked while the
    recorded policy states honestly what was used.
    """
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.is_file():
        data = ManifestManager.load(manifest_path)
        assignments = {str(k): str(v) for k, v in data.get("assignments", {}).items()}
        policy = str(data.get("split_policy", {}).get("method", "persisted"))
        missing = [cid for _, _, cid in pairs if cid not in assignments]
        if missing:
            print(
                f"[Split] WARNING: {len(missing)} new complexes not covered by the "
                "persisted manifest; assigning via stable hash."
            )
            for cid in missing:
                assignments[cid] = _stable_cid_bucket(cid, seed)
            data["assignments"] = dict(sorted(assignments.items()))
            ManifestManager.save_atomic(data, manifest_path)
        return assignments, f"{policy} (persisted)", len(assignments)

    rows: List[Dict[str, str]] = []
    protein_keys = set()
    for pocket_path, _, cid in pairs:
        try:
            seq = extract_protein_sequence(pocket_path)
        except Exception:
            seq = ""
        if seq:
            key = hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]
        else:
            # Malformed/unparseable pocket: pin an inert per-complex cluster so
            # it can never leak via a shared (unknown) sequence identity.
            key = f"empty_{cid}"
            seq = "X" * 25
        protein_keys.add(key)
        rows.append(
            {"complex_id": cid, "protein_key": key, "sequence": seq, "source": "crossdocked"}
        )

    work_dir = output_dir / ".split_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        fasta_path = str(work_dir / "proteins.fasta")
        write_fasta(rows, fasta_path)
        clusters = cluster_with_mmseqs2(fasta_path, str(work_dir))
        policy = "MMseqs2 (min_seq_id=0.30, coverage=0.80)"
    except Exception as exc:
        clusters = {r["protein_key"]: r["protein_key"] for r in rows}
        policy = (
            "identity-sequence clustering (MMseqs2 unavailable: "
            f"{type(exc).__name__})"
        )
        print(f"[Split] WARNING: {policy}; exact-sequence leakage lock still enforced.")

    assignments = deterministic_cluster_assignments(
        clusters, rows, seed=seed, fractions=(0.80, 0.10, 0.10)
    )
    assert_no_cluster_overlap(clusters, assignments, rows)
    save_split_manifest(str(manifest_path), assignments, clusters, rows, seed=seed)
    # save_split_manifest hard-codes method='MMseqs2'; record the real policy.
    data = ManifestManager.load(manifest_path)
    data.setdefault("split_policy", {})["method"] = policy
    ManifestManager.save_atomic(data, manifest_path)
    counts = {s: sum(1 for v in assignments.values() if v == s) for s in SPLITS}
    print(f"[Split] policy={policy} counts={counts}")
    return assignments, policy, len(protein_keys)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class CraftEngine:
    """Structural QC, chemical filtering, retrosynthesis and resumable sharding."""

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        catalog_path: Path,
        *,
        max_complexes: Optional[int] = None,
        workers: Optional[int] = None,
        shard_mb: int = 500,
        max_steps: int = 4,
        seed: int = 42,
        qc: Optional[Dict[str, Any]] = None,
        flush_states: int = 5000,
        max_target_pockets: int = 50,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = Path(catalog_path)
        self.max_complexes = max_complexes
        self.workers = int(workers or mp.cpu_count())
        self.shard_mb = int(shard_mb)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.qc = qc or DEFAULT_QC
        self.flush_states = int(flush_states)
        self.max_target_pockets = int(max_target_pockets)
        self.funnel = MeasuredDataFunnel()
        self.split_policy = ""
        self._rejected_path = self.output_dir / "rejected_complexes.json"
        self._rejected: Dict[str, str] = {}
        if self._rejected_path.is_file():
            try:
                self._rejected = {
                    str(k): str(v)
                    for k, v in ManifestManager.load(self._rejected_path).items()
                }
            except Exception:
                self._rejected = {}

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def unpack_raw_archive(self) -> Path:
        archive = self.raw_dir / "crossdocked_pocket10.tar.gz"
        extracted_dir = self.raw_dir / "extracted"
        marker = extracted_dir / ".extracted_marker"
        if not archive.is_file():
            raise FileNotFoundError(f"Raw archive not found: {archive}")
        extracted_dir.mkdir(parents=True, exist_ok=True)

        if not marker.is_file():
            print(f"[Extract] Unpacking {archive.name} via system tar...")
            t0 = time.time()
            try:
                subprocess.run(
                    ["tar", "-xzf", str(archive), "-C", str(extracted_dir)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                print("[Extract] system tar failed; falling back to Python tarfile...")
                with tarfile.open(archive, "r:gz") as tar:
                    try:
                        tar.extractall(path=extracted_dir, filter="data")
                    except TypeError:  # Python < 3.12 has no filter kwarg
                        tar.extractall(path=extracted_dir)
            marker.write_text(str(time.time()), encoding="utf-8")
            print(f"[Extract] Completed in {time.time() - t0:.2f}s")
        else:
            print("[Extract] Archive already extracted (marker found).")
        return extracted_dir

    # ------------------------------------------------------------------
    # Rejected-complex persistence (so failures are not reprocessed on resume)
    # ------------------------------------------------------------------
    def _save_rejected(self) -> None:
        if self._rejected:
            ManifestManager.save_atomic(self._rejected, self._rejected_path)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_crafting(self) -> None:
        extracted = self.unpack_raw_archive()
        all_pairs = scan_pairs(extracted)
        if not all_pairs:
            raise RuntimeError(f"No pocket/ligand pairs found under {extracted}")
        self.funnel.raw_source_records = len(all_pairs)

        if self.max_complexes and len(all_pairs) > self.max_complexes:
            rng = np.random.default_rng(self.seed)
            idx = sorted(rng.choice(len(all_pairs), size=self.max_complexes, replace=False))
            all_pairs = [all_pairs[i] for i in idx]

        assignments, policy, unique_proteins = _load_or_create_splits(
            all_pairs, self.output_dir, self.seed
        )
        self.split_policy = policy
        self.funnel.deduplication_before = len(all_pairs)
        self.funnel.deduplication_after = unique_proteins

        managers = {
            s: ShardCheckpointManager(
                self.output_dir, s, max_shard_bytes=self.shard_mb * 1024 * 1024
            )
            for s in SPLITS
        }
        work_items = [
            (p, l, c, assignments[c])
            for p, l, c in all_pairs
            if assignments.get(c, "train") in SPLITS
            and not managers[assignments[c]].is_complex_completed(c)
            and c not in self._rejected
        ]
        rejected_in_scope = sum(1 for _, _, c in all_pairs if c in self._rejected)
        print(
            f"[Resume] Total pairs: {len(all_pairs)}, "
            f"already committed: {len(all_pairs) - len(work_items) - rejected_in_scope}, "
            f"previously rejected: {rejected_in_scope}, "
            f"remaining: {len(work_items)}"
        )
        if not work_items:
            print("[Resume] Nothing left to process; finalising outputs.")

        self.funnel.structural_qc_input = len(work_items)
        self.funnel.retrosynthesis_candidates = len(work_items)

        state_buffers: Dict[str, List[Any]] = {s: [] for s in SPLITS}
        id_buffers: Dict[str, List[str]] = {s: [] for s in SPLITS}
        counts = {"reject_structural": 0, "reject_chemical": 0, "reject_retro": 0, "ok": 0}

        def _flush(split: str) -> None:
            if state_buffers[split]:
                managers[split].write_shard(
                    state_buffers[split], id_buffers[split]
                )
                state_buffers[split].clear()
                id_buffers[split].clear()
                self._save_rejected()

        if work_items:
            with mp.Pool(
                processes=self.workers,
                initializer=_init_worker,
                initargs=(str(self.catalog_path), self.max_steps, self.qc),
            ) as pool:
                results = pool.imap_unordered(_process_complex, work_items, chunksize=32)
                for res in tqdm(results, total=len(work_items), desc="Crafting Trajectories"):
                    stage = res["stage"]
                    counts[stage] = counts.get(stage, 0) + 1
                    if stage != "ok":
                        self._rejected[res["cid"]] = f"{stage}:{res['reason']}"
                        continue
                    split = res["split"]
                    states = res["states"]
                    state_buffers[split].extend(states)
                    id_buffers[split].append(res["cid"])
                    if len(state_buffers[split]) >= self.flush_states:
                        _flush(split)

        for split in SPLITS:
            _flush(split)
        self._save_rejected()

        # ---- Funnel accounting (this run + lifetime committed totals) ------
        self.funnel.structural_qc_rejected = counts["reject_structural"]
        self.funnel.structural_qc_passed = (
            self.funnel.structural_qc_input - self.funnel.structural_qc_rejected
        )
        self.funnel.chemical_qc_input = self.funnel.structural_qc_passed
        self.funnel.chemical_qc_rejected = counts["reject_chemical"]
        self.funnel.chemical_qc_passed = (
            self.funnel.chemical_qc_input - self.funnel.chemical_qc_rejected
        )
        self.funnel.retrosynthesis_decomposed = counts["ok"]
        self.funnel.retrosynthesis_failed = counts["reject_retro"]
        self.funnel.valid_complexes = sum(
            len(m.completed_complex_ids) for m in managers.values()
        )
        self.funnel.train_states = managers["train"].total_samples
        self.funnel.val_states = managers["val"].total_samples
        self.funnel.test_states = managers["test"].total_samples
        self.funnel.trajectory_states_total = (
            self.funnel.train_states + self.funnel.val_states + self.funnel.test_states
        )

        # ---- Stage targets + catalog for offline training -----------------
        targets_dir = self.output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        staged = {p.name for p in targets_dir.glob("*.pdb")}
        for pocket_path, _, cid in all_pairs:
            if len(staged) >= self.max_target_pockets:
                break
            split = assignments.get(cid)
            if split not in ("train", "val"):
                continue
            if managers[split].is_complex_completed(cid):
                dest = targets_dir / f"{Path(pocket_path).stem}_pocket.pdb"
                if not dest.exists():
                    shutil.copy2(pocket_path, dest)
                    staged.add(dest.name)
        shutil.copy2(self.catalog_path, self.output_dir / "enamine_3d_subset.parquet")

        # ---- Output manifest ---------------------------------------------
        output_manifest = {
            "dataset_type": "3d_syntree_crafted",
            "version": 2,
            "created_at": time.time(),
            "split_policy": policy,
            "raw_dataset": str(self.raw_dir),
            "catalog": "enamine_3d_subset.parquet",
            "max_steps": self.max_steps,
            "seed": self.seed,
            "qc": self.qc,
            "funnel": self.funnel.to_dict(),
            "rejected_count": len(self._rejected),
            "targets": sorted(staged),
            "splits": {
                s: {
                    "total_samples": managers[s].total_samples,
                    "shard_count": len(managers[s].shards),
                }
                for s in SPLITS
            },
        }
        ManifestManager.save_atomic(output_manifest, self.output_dir / "craft_output_manifest.json")

        print(f"\n[Output] Dataset staged at {self.output_dir}")
        print(f"[Split] policy: {policy}")
        if len(work_items) < len(all_pairs):
            print("[Note] QC/retro counters above cover the current (resume) run; "
                  "committed state/complex totals are lifetime values.")
        self.funnel.print_report()


# ---------------------------------------------------------------------------
# Verification smoke test (same code path as production training)
# ---------------------------------------------------------------------------

def run_verification_smoke_test(
    catalog_path: Path, output_dir: Path, min_catalog: int = 50
) -> None:
    """End-to-end verification pass: catalog -> dataset -> policy fwd/bwd.

    Uses synthetic states from ``CrossDockedDataset`` so it can run before
    any shards exist, exercising the exact collate + ``SynTreePolicy``
    forward/backward path that training uses (collate lives on
    ``ResilientTrainer``, NOT on ``CrossDockedDataset``).
    """
    print("\n" + "=" * 70)
    print("RUNNING ISOLATED VERIFICATION TEST PASS (Same Code Path)")
    print("=" * 70)

    from syntree.data.crossdocked import CrossDockedDataset
    from syntree.engine.trainer import ResilientTrainer
    from syntree.models.policy import SynTreePolicy

    cat = SynthonCatalog(str(catalog_path), embedding_dim=128)
    if len(cat) < min_catalog:
        raise RuntimeError(
            f"Verification Failed: catalog {catalog_path} has {len(cat)} synthons "
            f"(< {min_catalog}); is this the certified Enamine parquet?"
        )
    print(f"[Verify 1/3] Catalog loaded ({len(cat)} synthons, dim=128).")

    test_config = {
        "model": {
            "hidden_dim": 128,
            "num_equivariant_layers": 2,
            "num_radial_basis": 16,
            "cutoff_radius": 5.0,
            "synthon_embedding_dim": 128,
            "num_attention_heads": 4,
            "max_atomic_number": 100,
            "dropout": 0.0,
        }
    }
    model = SynTreePolicy(test_config)
    print("[Verify 2/3] Policy model instantiated successfully.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = CrossDockedDataset(
        str(output_dir), split="train", catalog=cat, num_synthetic=8, synthetic=True
    )
    items = [ds[i] for i in range(min(4, len(ds)))]
    batch = ResilientTrainer._collate(items)
    preds = model(batch, cat.embeddings, None, None)
    if "synthon_logits" not in preds:
        raise RuntimeError("Verification Failed: policy returned no synthon_logits")
    loss = preds["synthon_logits"].sum()
    loss.backward()
    print("[Verify 3/3] Forward + Backward gradient pass verified cleanly.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Craft Mode: Resilient SBDD Trajectory Processing"
    )
    parser.add_argument(
        "--search-root", default=None, help="Root mount directory to discover the raw dataset"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for crafted shards",
    )
    parser.add_argument("--config", default=None, help="Optional YAML config (configs/craft.yaml)")
    parser.add_argument("--max-complexes", type=int, default=None,
                        help="Optional limit for development smoke runs")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--shard-mb", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verify-only", action="store_true",
                        help="Run the verification suite only")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the verification smoke test")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    search_root = args.search_root or cfg.get("search_root") or "/kaggle/input"
    output_dir = Path(
        args.output_dir
        or cfg.get("output_dir")
        or "/kaggle/working/3d-syntree-dataset"
    ).resolve()

    # 1. Conservative discovery (manifest-certified, signature fallback)
    raw_dataset_dir = DatasetDiscovery.discover_raw_dataset(search_root)
    print(f"[Discovery] Verified raw dataset located at: {raw_dataset_dir}")

    catalog_path = raw_dataset_dir / "enamine_3d_subset.parquet"
    if not catalog_path.is_file():
        candidates = sorted(raw_dataset_dir.glob("enamine*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No Enamine catalog parquet found in {raw_dataset_dir}"
            )
        catalog_path = candidates[0]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Verification smoke test (same code path as training)
    if not args.skip_verify:
        run_verification_smoke_test(catalog_path, output_dir)
    if args.verify_only:
        return 0

    # 3. Full production crafting
    qc = {
        "structural": {
            **DEFAULT_QC["structural"],
            **(cfg.get("structural_qc") or {}),
        },
        "chemical": {**DEFAULT_QC["chemical"], **(cfg.get("chemical_qc") or {})},
    }
    splits_cfg = cfg.get("splits") or {}
    engine = CraftEngine(
        raw_dataset_dir,
        output_dir,
        catalog_path,
        max_complexes=args.max_complexes,
        workers=args.workers,
        shard_mb=args.shard_mb or int(cfg.get("shard_mb", 500)),
        max_steps=args.max_steps or int(cfg.get("max_steps", 4)),
        seed=args.seed or int(splits_cfg.get("seed", 42)),
        qc=qc,
    )
    engine.run_crafting()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
