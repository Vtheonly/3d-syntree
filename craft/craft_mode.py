#!/usr/bin/env python3
"""Craft Mode: Scientific curation, structural QC, retrosynthesis, and tensor generation.

Executes in local or cloud environments (e.g. Kaggle). Performs:
1. Building block discovery and curation from raw commercial vendor inventories.
2. CrossDocked archive extraction (or existing directory scanning) and RMSD filtering.
3. Multi-core retrosynthetic trajectory construction with zero-mmap IPC serialization.
4. Resumable 500 MB sharded dataset creation with SHA-256 manifests.
5. Direct publishing of verified datasets to Hugging Face Hub (optional).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
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

# Ensure codebase root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.checkpointing import ShardCheckpointManager
from core.checksums import compute_sha256
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


def _init_worker(catalog_path: str, max_steps: int) -> None:
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


def _process_complex(item: Tuple[str, str, str, str]) -> Optional[Tuple[str, bytes, str, str, int]]:
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

        # Structural & Chemical QC: discard non-drug artifacts
        flags = ligand_quality_flags(ligand_mol)
        if any(f in flags for f in ("too_small", "low_molecular_weight", "metal_containing", "crystallization_artifact")):
            return None

        # CrossDocked RMSD filter: discard poses with RMSD > 1.0 A
        if ligand_mol.HasProp("rmsd"):
            try:
                if float(ligand_mol.GetProp("rmsd")) > 1.0:
                    return None
            except ValueError:
                pass

        states, _ = _worker_builder.build(ligand_mol, pocket_mol, trajectory_id=cid)
        if states:
            # Serialize to raw bytes to bypass PyTorch multiprocessing shared-memory/mmap limits
            buf = io.BytesIO()
            torch.save(states, buf)
            return (split, buf.getvalue(), cid, p_path, len(states))
    except Exception:
        pass
    return None


class CraftEngine:
    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        max_complexes: Optional[int] = None,
        shard_mb: int = 500,
        seed: int = 42,
    ):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_complexes = max_complexes
        self.shard_mb = shard_mb
        self.seed = seed
        self.funnel = MeasuredDataFunnel()

    def resolve_or_curate_catalog(self) -> Path:
        """Ensure a curated 50k+ synthon catalog exists in output_dir."""
        dest_catalog = self.output_dir / "enamine_3d_subset.parquet"
        if dest_catalog.exists() and dest_catalog.stat().st_size > 1000:
            print(f"[Catalog] Verified existing curated catalog: {dest_catalog}")
            return dest_catalog

        # Check for pre-curated parquet in raw_dir
        for p in list(self.raw_dir.glob("*.parquet")) + list(self.raw_dir.parent.glob("*.parquet")):
            if "enamine" in p.name.lower():
                print(f"[Catalog] Copying pre-curated catalog from {p.name}...")
                shutil.copy2(p, dest_catalog)
                return dest_catalog

        # Look for raw vendor inventory (.csv, .csv.gz, .sdf)
        raw_candidates = (
            list(self.raw_dir.glob("*.[cC][sS][vV]*"))
            + list(self.raw_dir.glob("*.[sS][dD][fF]*"))
            + list(self.raw_dir.parent.glob("**/*vendor*"))
        )
        if not raw_candidates:
            raise FileNotFoundError(f"No building block export found in {self.raw_dir}. Attach a commercial catalog.")

        raw_export = raw_candidates[0]
        print(f"[Catalog] Curating raw vendor export {raw_export.name}...")
        cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "build_synthon_catalog.py"),
            "--input", str(raw_export),
            "--output", str(dest_catalog),
            "--min-fsp3", "0.40",
            "--max-mw", "220.0",
            "--min-heavy-atoms", "4",
            "--strict-min-size", "10000",
        ]
        subprocess.run(cmd, check=True)
        return dest_catalog

    def resolve_raw_sources(self) -> Path:
        """Locate existing extracted folder or unpack crossdocked archive."""
        cd_candidates = [p for p in self.raw_dir.glob("*crossdocked*") if not p.name.startswith(".")]
        if not cd_candidates:
            cd_candidates = [p for p in self.raw_dir.parent.glob("**/*crossdocked*") if not p.name.startswith(".")]
        if not cd_candidates:
            raise FileNotFoundError(f"No CrossDocked archive or directory found in {self.raw_dir}!")

        cd_src = cd_candidates[0]
        is_extracted = cd_src.is_dir() and any(f.endswith(".pdb") for _, _, files in os.walk(cd_src) for f in files)
        if is_extracted:
            print(f"[Extract] Source is already extracted ({cd_src.name}). Scanning directly...")
            return cd_src

        extracted_dir = self.output_dir / "extracted_raw"
        extracted_marker = extracted_dir / ".extracted_marker"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        if not extracted_marker.exists():
            print(f"[Extract] Unpacking {cd_src.name}...")
            t0 = time.time()
            res = os.system(f"tar -xf '{cd_src}' -C '{extracted_dir}' > /dev/null 2>&1")
            if res != 0:
                os.system(f"tar -xzf '{cd_src}' -C '{extracted_dir}' > /dev/null 2>&1")
            extracted_marker.touch()
            print(f"[Extract] Completed in {time.time() - t0:.2f}s")
        return extracted_dir

    def scan_pairs(self, scan_dir: Path) -> List[Tuple[str, str, str]]:
        print(f"[Discover] Fast C-level scan of {scan_dir}...")
        t0 = time.time()
        pairs = []
        for root, _, files in os.walk(scan_dir):
            pockets = [f for f in files if "pocket" in f.lower() and f.endswith(".pdb")]
            ligands = [f for f in files if f.endswith(_LIGAND_EXTENSIONS)]
            if not pockets or not ligands:
                continue
            parent = os.path.basename(root)
            for p in pockets:
                p_stem = p[:-4]
                # Best matching ligand
                best_lig = ligands[0]
                for lig in ligands:
                    if p_stem in lig or lig[:-4] in p_stem:
                        best_lig = lig
                        break
                pairs.append((os.path.join(root, p), os.path.join(root, best_lig), f"{parent}_{p_stem}"))

        pairs.sort(key=lambda x: x[2])
        self.funnel.raw_source_records = len(pairs)
        print(f"[Discover] Found {len(pairs):,} raw complexes in {time.time() - t0:.2f}s")
        return pairs

    def run_crafting(self, workers: int = mp.cpu_count()) -> None:
        catalog_path = self.resolve_or_curate_catalog()
        scan_dir = self.resolve_raw_sources()
        all_pairs = self.scan_pairs(scan_dir)

        if not all_pairs:
            raise RuntimeError(f"No valid complexes found in {scan_dir}!")

        self.funnel.structural_qc_input = len(all_pairs)
        if self.max_complexes and len(all_pairs) > self.max_complexes:
            rng = np.random.default_rng(self.seed)
            indices = sorted(rng.choice(len(all_pairs), size=self.max_complexes, replace=False))
            all_pairs = [all_pairs[i] for i in indices]
            print(f"[Sample] Subsampled {len(all_pairs):,} complexes for trajectory building.")

        # Deterministic 80/10/10 splits
        n_items = len(all_pairs)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n_items)
        n_train = int(n_items * 0.80)
        n_val = int(n_items * 0.10)
        split_map = {
            all_pairs[idx][2]: ("train" if pos < n_train else "val" if pos < (n_train + n_val) else "test")
            for pos, idx in enumerate(order)
        }

        managers = {
            s: ShardCheckpointManager(self.output_dir, s, max_shard_bytes=self.shard_mb * 1024 * 1024)
            for s in ("train", "val", "test")
        }

        work_items = [
            (p, l, cid, split_map[cid])
            for p, l, cid in all_pairs
            if not managers[split_map[cid]].is_complex_completed(cid)
        ]
        print(f"[Resume] Total items: {len(all_pairs):,} | Remaining to process: {len(work_items):,}")
        self.funnel.retrosynthesis_candidates = len(work_items)

        state_bytes_buf: Dict[str, List[bytes]] = {"train": [], "val": [], "test": []}
        id_buf: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        counts_buf: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        target_pockets: List[str] = []

        def flush_split(split_name: str) -> None:
            if not state_bytes_buf[split_name]:
                return
            # Deserialization on main heap
            deserialized_samples = []
            for raw_b in state_bytes_buf[split_name]:
                deserialized_samples.extend(torch.load(io.BytesIO(raw_b), weights_only=False))
            managers[split_name].write_shard(deserialized_samples, id_buf[split_name])
            state_bytes_buf[split_name].clear()
            id_buf[split_name].clear()
            counts_buf[split_name] = 0

        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(str(catalog_path), 4), maxtasksperchild=500) as pool:
            for res in tqdm(pool.imap_unordered(_process_complex, work_items, chunksize=16), total=len(work_items), desc="Crafting Trajectories"):
                if res is None:
                    self.funnel.retrosynthesis_failed += 1
                    continue

                split, states_bytes, cid, p_path, count = res
                self.funnel.retrosynthesis_decomposed += 1
                self.funnel.valid_complexes += 1

                state_bytes_buf[split].append(states_bytes)
                id_buf[split].append(cid)
                counts_buf[split] += count

                if split in ("train", "val") and len(target_pockets) < 50:
                    target_pockets.append(p_path)

                if counts_buf[split] >= 2500:
                    flush_split(split)

        for s in ("train", "val", "test"):
            flush_split(s)

        # Stage target pockets for Stage 2 RL / Docking
        targets_dir = self.output_dir / "targets"
        targets_dir.mkdir(parents=True, exist_ok=True)
        for tp in target_pockets:
            shutil.copy2(tp, targets_dir / f"{Path(tp).stem}_pocket.pdb")

        shutil.copy2(catalog_path, self.output_dir / "enamine_3d_subset.parquet")

        self.funnel.train_states = sum(s.sample_count for s in managers["train"].shards)
        self.funnel.val_states = sum(s.sample_count for s in managers["val"].shards)
        self.funnel.test_states = sum(s.sample_count for s in managers["test"].shards)
        self.funnel.trajectory_states_total = self.funnel.train_states + self.funnel.val_states + self.funnel.test_states
        self.funnel.print_report()

    def upload_to_hf(self, repo_id: str, token: Optional[str] = None) -> None:
        """Upload the crafted dataset directly to Hugging Face Hub."""
        hf_token = token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN must be provided to upload to Hugging Face.")

        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        print(f"\n[Upload] Publishing dataset to https://huggingface.co/datasets/{repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        readme_path = self.output_dir / "README.md"
        if not readme_path.exists():
            readme_text = f"""---
license: mit
task_categories:
- graph-ml
tags:
- biology
- chemistry
- sbdd-trajectories
pretty_name: 3D-SynTree Verified Multi-Dataset Trajectories
size_categories:
- 10K<n<100K
---

# 3D-SynTree: Curated SBDD Trajectories

Reaction-validated retrosynthetic trajectories with 3D crystal coordinates and exact forward-replay verification.
"""
            readme_path.write_text(readme_text, encoding="utf-8")

        # Exclude raw unpacked files from the upload
        api.upload_folder(
            folder_path=str(self.output_dir),
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=["extracted_raw/**", ".extracted_marker"],
            commit_message="Publish verified 3D-SynTree sharded dataset, catalog, and targets",
        )
        print(f"[Upload] Complete: https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Craft Mode: Resilient SBDD Trajectory Processing")
    parser.add_argument("--search-root", default="/kaggle/input", help="Directory where input datasets are mounted")
    parser.add_argument("--raw-dir", default=None, help="Explicit raw dataset path (auto-discovered if omitted)")
    parser.add_argument("--output-dir", default="/kaggle/working/3d-syntree-dataset")
    parser.add_argument("--max-complexes", type=int, default=None)
    parser.add_argument("--shard-mb", type=int, default=500)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-repo", default=None, help="Hugging Face dataset repo (e.g. user/3d-syntree-multidataset)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (or HF_TOKEN env var)")
    parser.add_argument("--upload", action="store_true", help="Upload to Hugging Face upon completion")
    parser.add_argument("--upload-only", action="store_true", help="Skip crafting and upload existing output_dir")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.upload_only:
        if not args.hf_repo:
            print("ERROR: --hf-repo is required when using --upload-only", file=sys.stderr)
            return 1
        engine = CraftEngine(Path("."), out_dir)
        engine.upload_to_hf(args.hf_repo, args.hf_token)
        return 0

    if args.raw_dir:
        raw_dir = Path(args.raw_dir).resolve()
    else:
        raw_dir = DatasetDiscovery.discover_raw_dataset(args.search_root)

    print(f"[Discovery] Raw source directory: {raw_dir}")
    engine = CraftEngine(
        raw_dir=raw_dir,
        output_dir=out_dir,
        max_complexes=args.max_complexes,
        shard_mb=args.shard_mb,
        seed=args.seed,
    )
    engine.run_crafting(workers=args.workers)

    if args.upload:
        if not args.hf_repo:
            print("ERROR: --upload requires --hf-repo", file=sys.stderr)
            return 1
        engine.upload_to_hf(args.hf_repo, args.hf_token)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())