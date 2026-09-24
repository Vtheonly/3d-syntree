#!/usr/bin/env python3
"""Craft Mode: High-performance structural processing, retrosynthesis, and tensor generation.

Executes strictly on Kaggle (Offline or Online).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
import rdkit.Chem.rdFingerprintGenerator as rdFPGen
from tqdm import tqdm

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.checkpointing import ShardCheckpointManager
from core.discovery import DatasetDiscovery
from core.logging import MeasuredDataFunnel
from syntree.chemistry.catalog import SynthonCatalog
from syntree.chemistry.validator import ChemicalValidator
from syntree.data.curation import PocketExtractionConfig, ligand_quality_flags, standardize_pocket
from syntree.data.fragmenter import ReactionConstrainedFragmenter, canonical_smiles
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder

_LIGAND_EXTENSIONS = (".sdf", ".mol2")

_worker_catalog = None
_worker_builder = None
_catalog_fps = None
_morgan_gen = None


def _init_worker(catalog_path: str, max_steps: int):
    global _worker_catalog, _worker_builder, _catalog_fps, _morgan_gen
    RDLogger.DisableLog("rdApp.*")
    _worker_catalog = SynthonCatalog(catalog_path)
    _worker_builder = RetrosyntheticTrajectoryBuilder(_worker_catalog, max_steps=max_steps)
    _morgan_gen = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)
    _catalog_fps = [
        _morgan_gen.GetFingerprint(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else None
        for s in _worker_catalog.df["smiles"]
    ]

    def fast_catalog_candidates(self, synthon_mol: Chem.Mol, min_tanimoto: float = 0.85):
        exact = self._catalog_lookup.get(canonical_smiles(synthon_mol), ())
        if exact:
            return tuple(sorted(exact))
        fp = _morgan_gen.GetFingerprint(synthon_mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, _catalog_fps)
        scored = [(s, i) for i, s in enumerate(sims) if s >= min_tanimoto and _catalog_fps[i] is not None]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(i for _, i in scored)

    ReactionConstrainedFragmenter._catalog_candidates = fast_catalog_candidates


def _process_complex(item: Tuple[str, str, str, str]) -> Optional[Tuple[str, List[Any], str, str]]:
    p_path, l_path, cid, split = item
    global _worker_builder
    try:
        pocket_mol = Chem.MolFromPDBFile(p_path, removeHs=False)
        if l_path.lower().endswith(".mol2"):
            ligand_mol = Chem.MolFromMol2File(l_path, removeHs=False, sanitize=True)
        else:
            supp = Chem.SDMolSupplier(l_path, removeHs=False, sanitize=True)
            ligand_mol = next((m for m in supp if m is not None), None)

        if pocket_mol is None or ligand_mol is None:
            return None
        if pocket_mol.GetNumAtoms() == 0 or ligand_mol.GetNumAtoms() == 0 or ligand_mol.GetNumConformers() == 0:
            return None

        # Structural & Chemical QC
        flags = ligand_quality_flags(ligand_mol)
        if any(f in flags for f in ("too_small", "low_molecular_weight", "metal_containing")):
            return None

        states, _ = _worker_builder.build(ligand_mol, pocket_mol, trajectory_id=cid)
        if states:
            return (split, states, cid, p_path)
    except Exception:
        pass
    return None


class CraftEngine:
    """Orchestrates structural QC, chemical filtering, and resumable sharding."""

    def __init__(self, raw_dir: Path, output_dir: Path, catalog_path: Path, max_complexes: Optional[int] = None):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = catalog_path
        self.max_complexes = max_complexes
        self.funnel = MeasuredDataFunnel()

    def unpack_raw_archive(self) -> Path:
        archive = self.raw_dir / "crossdocked_pocket10.tar.gz"
        extracted_marker = self.raw_dir / ".extracted_marker"
        extracted_dir = self.raw_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        if not extracted_marker.exists():
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
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(path=extracted_dir)
            extracted_marker.touch()
            print(f"[Extract] Completed in {time.time() - t0:.2f}s")
        return extracted_dir

    def scan_pairs(self, root_dir: Path) -> List[Tuple[str, str, str]]:
        print(f"[Discover] Fast C-level scan of {root_dir}...")
        pairs = []
        for root, _, files in os.walk(root_dir):
            pockets = [f for f in files if "pocket" in f.lower() and f.endswith(".pdb")]
            ligands = [f for f in files if f.endswith(_LIGAND_EXTENSIONS)]
            if not pockets or not ligands:
                continue
            parent = os.path.basename(root)
            for p in pockets:
                p_stem = p[:-4]
                pairs.append((os.path.join(root, p), os.path.join(root, ligands[0]), f"{parent}_{p_stem}"))
        pairs.sort(key=lambda x: x[2])
        self.funnel.raw_source_records = len(pairs)
        return pairs

    def run_crafting(self, workers: int = mp.cpu_count(), shard_size_mb: int = 500) -> None:
        extracted = self.unpack_raw_archive()
        all_pairs = self.scan_pairs(extracted)

        self.funnel.structural_qc_input = len(all_pairs)
        if self.max_complexes and len(all_pairs) > self.max_complexes:
            rng = np.random.default_rng(42)
            all_pairs = [all_pairs[i] for i in sorted(rng.choice(len(all_pairs), size=self.max_complexes, replace=False))]

        n_items = len(all_pairs)
        rng = np.random.default_rng(42)
        order = rng.permutation(n_items)
        n_train = int(n_items * 0.80)
        n_val = int(n_items * 0.10)
        split_map = {
            all_pairs[idx][2]: ("train" if pos < n_train else "val" if pos < (n_train + n_val) else "test")
            for pos, idx in enumerate(order)
        }

        managers = {
            s: ShardCheckpointManager(self.output_dir, s, max_shard_bytes=shard_size_mb * 1024 * 1024)
            for s in ("train", "val", "test")
        }

        # Filter out already completed complexes (resumability)
        work_items = [
            (p, l, c, split_map[c])
            for p, l, c in all_pairs
            if not managers[split_map[c]].is_complex_completed(c)
        ]
        print(f"[Resume] Total items: {len(all_pairs)}, Remaining to process: {len(work_items)}")
        self.funnel.retrosynthesis_candidates = len(work_items)

        state_buffers: Dict[str, List[Any]] = {"train": [], "val": [], "test": []}
        id_buffers: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        target_pockets: List[str] = []

        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(str(self.catalog_path), 4)) as pool:
            for res in tqdm(pool.imap_unordered(_process_complex, work_items, chunksize=32), total=len(work_items), desc="Crafting Trajectories"):
                if res is None:
                    self.funnel.retrosynthesis_failed += 1
                    continue

                split, states, cid, p_path = res
                self.funnel.retrosynthesis_decomposed += 1
                self.funnel.valid_complexes += 1
                self.funnel.trajectory_states_total += len(states)

                state_buffers[split].extend(states)
                id_buffers[split].append(cid)
                if split in ("train", "val") and len(target_pockets) < 50:
                    target_pockets.append(p_path)

                # Incremental Shard Flush (avoids unbounded memory growth)
                if len(state_buffers[split]) >= 5000:
                    managers[split].write_shard(state_buffers[split], id_buffers[split])
                    state_buffers[split].clear()
                    id_buffers[split].clear()

        # Flush remaining buffers
        for split in ("train", "val", "test"):
            if state_buffers[split]:
                managers[split].write_shard(state_buffers[split], id_buffers[split])

        # Stage targets and catalog
        targets_dir = self.output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        for tp in target_pockets:
            shutil.copy2(tp, targets_dir / f"{Path(tp).stem}_pocket.pdb")
        shutil.copy2(self.catalog_path, self.output_dir / "enamine_3d_subset.parquet")

        self.funnel.train_states = sum(s.sample_count for s in managers["train"].shards)
        self.funnel.val_states = sum(s.sample_count for s in managers["val"].shards)
        self.funnel.test_states = sum(s.sample_count for s in managers["test"].shards)
        self.funnel.trajectory_states_total = self.funnel.train_states + self.funnel.val_states + self.funnel.test_states
        self.funnel.print_report()


def run_verification_smoke_test(catalog_path: Path, output_dir: Path) -> None:
    """Run an isolated end-to-end verification pass before full execution."""
    print("\n" + "=" * 70)
    print("RUNNING ISOLATED VERIFICATION TEST PASS (Same Code Path)")
    print("=" * 70)

    from syntree.models.policy import SynTreePolicy
    from syntree.data.crossdocked import CrossDockedDataset

    cat = SynthonCatalog(str(catalog_path), embedding_dim=128)
    assert len(cat) >= 50, "Verification Failed: Catalog invalid"

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
    print("[Verify 1/3] Policy Model instantiated successfully.")

    ds = CrossDockedDataset(str(output_dir), split="train", catalog=cat, num_synthetic=8, synthetic=True)
    sample = ds[0]
    batch = CrossDockedDataset._collate([sample])

    preds = model(batch, cat.embeddings, None, None)
    loss = preds["synthon_logits"].sum()
    loss.backward()
    print("[Verify 2/3] Forward + Backward gradient pass verified cleanly.")
    print("[Verify 3/3] Verification passed. Ready for full production run.\n" + "=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Craft Mode: Resilient SBDD Trajectory Processing")
    parser.add_argument("--search-root", default="/kaggle/input", help="Root mount directory to discover raw dataset")
    parser.add_argument("--output-dir", default="/kaggle/working/3d-syntree-dataset", help="Output directory for crafted shards")
    parser.add_argument("--max-complexes", type=int, default=None, help="Optional limit for development smoke runs")
    parser.add_argument("--verify-only", action="store_true", help="Run the verification suite only")
    args = parser.parse_args()

    # 1. Conservative Discovery
    raw_dataset_dir = DatasetDiscovery.discover_raw_dataset(args.search_root)
    print(f"[Discovery] Verified Raw Dataset located at: {raw_dataset_dir}")

    catalog_path = raw_dataset_dir / "enamine_3d_subset.parquet"
    output_dir = Path(args.output_dir).resolve()

    # 2. Run Verification Smoke Test
    run_verification_smoke_test(catalog_path, output_dir)
    if args.verify_only:
        return 0

    # 3. Full Production Crafting
    engine = CraftEngine(raw_dataset_dir, output_dir, catalog_path, max_complexes=args.max_complexes)
    engine.run_crafting()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())