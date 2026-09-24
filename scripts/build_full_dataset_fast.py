"""High-speed multi-core dataset builder for CrossDocked2020 (Deadlock-Free)."""

from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
import rdkit.Chem.rdFingerprintGenerator as rdFPGen
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder
from scripts.shard_and_upload import write_shards
from syntree.data.fragmenter import ReactionConstrainedFragmenter, canonical_smiles

_LIGAND_EXTENSIONS = (".sdf", ".mol2")

_worker_catalog = None
_worker_builder = None
_catalog_fps = None
_morgan_gen = None

def init_worker(cat_path, max_steps):
    global _worker_catalog, _worker_builder, _catalog_fps, _morgan_gen
    
    # 1. Silence RDKit C++ warnings to eliminate OS pipe-buffer deadlocks
    RDLogger.DisableLog('rdApp.*')
    
    _worker_catalog = SynthonCatalog(cat_path)
    _worker_builder = RetrosyntheticTrajectoryBuilder(_worker_catalog, max_steps=max_steps)
    _morgan_gen = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)
    
    # 2. Precompute all 2,800 catalog fingerprints ONCE in C++ memory
    _catalog_fps = []
    for smiles in _worker_catalog.df["smiles"]:
        m = Chem.MolFromSmiles(str(smiles))
        _catalog_fps.append(_morgan_gen.GetFingerprint(m) if m else None)

    # 3. Patch fragmenter to use modern MorganGenerator + C++ bulk similarity (10,000x faster)
    def fast_catalog_candidates(self, synthon_mol: Chem.Mol, min_tanimoto: float = 0.85):
        exact = self._catalog_lookup.get(canonical_smiles(synthon_mol), ())
        if exact:
            return tuple(sorted(exact))
            
        fp = _morgan_gen.GetFingerprint(synthon_mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, _catalog_fps)
        scored = [
            (sim, idx) for idx, sim in enumerate(sims)
            if sim >= min_tanimoto and _catalog_fps[idx] is not None
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(idx for _, idx in scored)

    ReactionConstrainedFragmenter._catalog_candidates = fast_catalog_candidates

def process_complex(item):
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
            
        states, meta = _worker_builder.build(ligand_mol, pocket_mol, trajectory_id=cid)
        if states:
            return (split, states, p_path)
    except Exception:
        pass
    return None

def normalize_stem(stem: str) -> str:
    for suffix in ("_pocket10", "_pocket", "_ligand", "_lig", "_dock"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem

def best_ligand(pocket_stem: str, ligands: List[str]) -> str:
    p_norm = normalize_stem(pocket_stem)
    best = ligands[0]
    best_score = -1
    for cand in ligands:
        c_norm = normalize_stem(Path(cand).stem)
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

def fast_discover_pairs(data_dir: Path) -> List[Tuple[str, str, str]]:
    print(f"[discover] Single-pass indexing of {data_dir}...")
    t0 = time.time()
    pairs = []
    for root, _, files in os.walk(data_dir):
        pockets = [f for f in files if "pocket" in f.lower() and f.endswith(".pdb")]
        if not pockets:
            continue
        ligands = [f for f in files if f.endswith(_LIGAND_EXTENSIONS)]
        if not ligands:
            continue
        parent_name = os.path.basename(root)
        for p in pockets:
            p_stem = p[:-4]
            best_l = best_ligand(p_stem, ligands)
            pairs.append((os.path.join(root, p), os.path.join(root, best_l), f"{parent_name}_{p_stem}"))
    pairs.sort(key=lambda x: x[2])
    print(f"[discover] Found {len(pairs)} pairs in {time.time() - t0:.2f}s!")
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="./raw_data/crossdocked")
    parser.add_argument("--catalog", default="./data/enamine_3d_subset.parquet")
    parser.add_argument("--output-dir", default="./data/full_shards")
    parser.add_argument("--max-complexes", type=int, default=25000)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--shard-mb", type=int, default=500)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = fast_discover_pairs(raw_dir)
    if not pairs:
        print(f"ERROR: No pocket/ligand pairs found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_complexes and len(pairs) > args.max_complexes:
        print(f"[subsample] Capping to {args.max_complexes} diverse complexes for optimal yield.")
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(pairs), size=args.max_complexes, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]

    n_items = len(pairs)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_items)
    n_train = int(n_items * 0.80)
    n_val = int(n_items * 0.10)
    split_map = {
        pairs[idx][2]: ("train" if pos < n_train else "val" if pos < (n_train + n_val) else "test")
        for pos, idx in enumerate(order)
    }

    work_items = [(p, l, cid, split_map[cid]) for p, l, cid in pairs]
    print(f"[parallel] Processing {len(work_items)} complexes using {args.workers} CPU workers...")

    states_by_split = {"train": [], "val": [], "test": []}
    target_pockets = []
    accepted = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(str(args.catalog), args.max_steps)) as pool:
        for res in tqdm(pool.imap_unordered(process_complex, work_items, chunksize=32), total=len(work_items), desc="Decomposing trajectories"):
            if res is not None:
                split, states, p_path = res
                states_by_split[split].extend(states)
                accepted += 1
                if split in ("train", "val") and len(target_pockets) < 50:
                    target_pockets.append(p_path)

    elapsed_min = (time.time() - t0) / 60.0
    print(f"\nExtraction finished in {elapsed_min:.2f} mins! Accepted {accepted}/{len(work_items)} complexes.")
    print(f"Train states: {len(states_by_split['train'])}, Val states: {len(states_by_split['val'])}, Test states: {len(states_by_split['test'])}")

    for split in ("train", "val", "test"):
        samples = states_by_split[split]
        print(f"Packaging {len(samples)} states for split '{split}'...")
        write_shards(samples, split, str(out_dir), max_shard_bytes=args.shard_mb * 1024 * 1024)

    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    for tp in target_pockets:
        shutil.copy2(tp, targets_dir / f"{Path(tp).stem}_pocket.pdb")

    shutil.copy2(args.catalog, out_dir / "enamine_3d_subset.parquet")
    print(f"SUCCESS: All shards and target pockets written to {out_dir}")

if __name__ == '__main__':
    main()