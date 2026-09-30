#!/usr/bin/env python3
"""Isolated Verification Smoke-Test for the 3D-SynTree environment and architecture."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

# Ensure codebase root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.crossdocked import CrossDockedDataset
from syntree.models.policy import SynTreePolicy


def collate_pyg(items):
    clean_items = []
    for item in items:
        data = item.clone()
        if hasattr(data, "pocket_pos") and data.pocket_pos is not None:
            data.num_nodes = data.pocket_pos.size(0)
        elif hasattr(data, "pos") and data.pos is not None:
            data.num_nodes = data.pos.size(0)
        for k in list(data.keys()):
            v = data[k]
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                data[k] = v.unsqueeze(0)
        clean_items.append(data)
    batch = Batch.from_data_list(clean_items, follow_batch=["pocket_pos", "ligand_pos"])
    if not hasattr(batch, "pocket_batch") or batch.pocket_batch is None:
        batch.pocket_batch = getattr(batch, "pocket_pos_batch", None)
    if not hasattr(batch, "ligand_batch") or batch.ligand_batch is None:
        batch.ligand_batch = getattr(batch, "ligand_pos_batch", None)
    return batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify 3D-SynTree environment and architecture")
    parser.add_argument("--catalog", default=None, help="Path to synthon catalog parquet file")
    args = parser.parse_args()

    print("=" * 70)
    print("3D-SYNTREE PIPELINE & HARDWARE VERIFICATION")
    print("=" * 70)

    import rdkit
    import torch_geometric
    import syntree

    print(f"  PyTorch:         {torch.__version__}")
    print(f"  PyG:             {torch_geometric.__version__}")
    print(f"  RDKit:           {rdkit.__version__}")
    print(f"  3D-SynTree:      {syntree.__version__}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device:          {device_name}")
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        print(f"  VRAM Available:  {free_bytes / 1024**3:.2f} GB / {total_bytes / 1024**3:.2f} GB")

    # Catalog verification
    catalog_path = args.catalog
    if not catalog_path or not Path(catalog_path).exists():
        # Fallback search
        candidates = list(Path("/kaggle/input").rglob("*.parquet")) + list(Path("data").rglob("*.parquet"))
        if candidates:
            catalog_path = str(candidates[0])
        else:
            from scripts.download_assets import build_synthetic_catalog
            catalog_path = build_synthetic_catalog("/tmp/verify_catalog", num_copies=2)

    print(f"\n[1/3] Loading synthon catalog: {catalog_path}")
    cat = SynthonCatalog(str(catalog_path), embedding_dim=128)
    if len(cat) < 5:
        raise ValueError(f"Catalog has only {len(cat)} entries; verification failed.")
    print(f"      Verified catalog with {len(cat):,} synthons.")

    # Model instantiation
    print("\n[2/3] Instantiating SynTreePolicy test architecture...")
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
    print("      Model instantiated successfully.")

    # Forward + Backward check
    print("\n[3/3] Testing forward and backward gradient passes...")
    ds = CrossDockedDataset("/tmp/verify_smoke", split="train", catalog=cat, num_synthetic=4, synthetic=True)
    batch = collate_pyg([ds[0]])
    preds = model(batch, cat.embeddings, None, None)
    loss = preds["synthon_logits"].sum() + preds["reaction_logits"].sum()
    loss.backward()

    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    if not has_grads:
        raise RuntimeError("Backward pass completed but no gradients were computed!")
    print("      Forward + Backward gradient propagation verified.")

    print("\n" + "=" * 70)
    print("ENVIRONMENT & ARCHITECTURE FULLY CERTIFIED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())