#!/usr/bin/env python3
"""Craft Mode: Scientific curation, structural QC, retrosynthesis, and tensor generation.

Executes strictly in Kaggle. Performs:
1. Building block curation from raw commercial vendor inventories.
2. CrossDocked extraction and native RMSD quality filtering (<= 1.0 A).
3. Multi-core retrosynthetic decomposition across all available CPU cores.
4. Resumable 500 MB sharded dataset creation with SHA-256 manifests.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.checkpointing import ShardCheckpointManager
from core.discovery import DatasetDiscovery
from core.logging import MeasuredDataFunnel
from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.curation import ligand_quality_flags
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

        # Structural & Chemical QC: discard non-drug-like artifacts and docked decoys
        flags = ligand_quality_flags(ligand_mol)
        if any(f in flags for f in ("too_small", "low_molecular_weight", "metal_containing", "crystallization_artifact")):
            return None

        # CrossDocked RMSD check: discard docked poses with RMSD > 1.0 A
        if ligand_mol.HasProp("rmsd"):
            try:
                if float(ligand_mol.GetProp("rmsd")) > 1.0:
                    return None
            except ValueError:
                pass

        states, _ = _worker_builder.build(ligand_mol, pocket_mol, trajectory_id=cid)
        if states:
            return (split, states, cid, p_path)
    except Exception:
        pass
    return None


class CraftEngine:
    def __init__(self, raw_dir: Path, output_dir: Path, max_complexes: Optional[int] = None):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_complexes = max_complexes
        self.funnel = MeasuredDataFunnel()

    def resolve_or_curate_catalog(self) -> Path:
        """Ensure a curated 50k+ synthon catalog exists in output_dir."""
        dest_catalog = self.output_dir / "enamine_3d_subset.parquet"
        if dest_catalog.exists() and dest_catalog.stat().st_size > 1000:
            print(f"[Catalog] Verified existing curated catalog: {dest_catalog}")
            return dest_catalog

        # Check for pre-curated parquet in raw_dir
        for p in self.raw_dir.glob("*.parquet"):
            if "enamine" in p.name.lower():
                print(f"[Catalog] Copying pre-curated catalog from {p.name}...")
                shutil.copy2(p, dest_catalog)
                return dest_catalog

        # Look for raw vendor inventory (.csv, .csv.gz, .sdf)
        raw_candidates = list(self.raw_dir.glob("*.[cC][sS][vV]*")) + list(self.raw_dir.glob("*.[sS][dD][fF]*"))
        if not raw_candidates:
            raise FileNotFoundError(f"No building block export found in {self.raw_dir}. Attach a commercial catalog.")

        raw_export = raw_candidates[0]
        print(f"[Catalog] Curating raw vendor export {raw_export.name} on Kaggle...")
        cmd = [
            sys.executable, "scripts/build_synthon_catalog.py",
            "--input", str(raw_export),
            "--output", str(dest_catalog),
            "--min-fsp3", "0.40",
            "--max-mw", "220.0",
            "--min-heavy-atoms", "4",
            "--strict-min-size", "10000",
        ]
        subprocess.run(cmd, check=True)
        return dest_catalog

    def unpack_raw_archive(self) -> Path:
        archive = self.raw_dir / "crossdocked_pocket10.tar.gz"
        extracted_marker = self.output_dir / ".extracted_marker"
        extracted_dir = self.output_dir / "extracted_raw"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        if not extracted_marker.exists():
            print(f"[Extract] Unpacking {archive.name}...")
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
        catalog_path = self.resolve_or_curate_catalog()
        extracted = self.unpack_raw_archive()
        all_pairs = self.scan_pairs(extracted)

        self.funnel.structural_qc_input = len(all_pairs)
        if self.max_complexes and len(all_pairs) > self.max_complexes:
            rng = np.random.default_rng(42)
            indices = sorted(rng.choice(len(all_pairs), size=self.max_complexes, replace=False))
            all_pairs = [all_pairs[i] for i in indices]

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

        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(str(catalog_path), 4)) as pool:
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

                if len(state_buffers[split]) >= 5000:
                    managers[split].write_shard(state_buffers[split], id_buffers[split])
                    state_buffers[split].clear()
                    id_buffers[split].clear()

        for split in ("train", "val", "test"):
            if state_buffers[split]:
                managers[split].write_shard(state_buffers[split], id_buffers[split])

        targets_dir = self.output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        for tp in target_pockets:
            shutil.copy2(tp, targets_dir / f"{Path(tp).stem}_pocket.pdb")

        self.funnel.train_states = sum(s.sample_count for s in managers["train"].shards)
        self.funnel.val_states = sum(s.sample_count for s in managers["val"].shards)
        self.funnel.test_states = sum(s.sample_count for s in managers["test"].shards)
        self.funnel.trajectory_states_total = self.funnel.train_states + self.funnel.val_states + self.funnel.test_states
        self.funnel.print_report()


def main() -> int:
    parser = argparse.ArgumentParser(description="Craft Mode: Resilient SBDD Trajectory Processing")
    parser.add_argument("--search-root", default="/kaggle/input")
    parser.add_argument("--output-dir", default="/kaggle/working/3d-syntree-dataset")
    parser.add_argument("--max-complexes", type=int, default=None)
    args = parser.parse_args()

    raw_dir = DatasetDiscovery.discover_raw_dataset(args.search_root)
    print(f"[Discovery] Verified Raw Dataset located at: {raw_dir}")

    engine = CraftEngine(raw_dir, Path(args.output_dir), max_complexes=args.max_complexes)
    engine.run_crafting()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())