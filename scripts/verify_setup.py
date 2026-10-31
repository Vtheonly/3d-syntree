#!/usr/bin/env python3
"""Isolated Verification Smoke-Test for the 3D-SynTree environment and architecture."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure codebase root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from torch_geometric.data import Data

from syntree.chemistry.catalog import SynthonCatalog
from syntree.core.schema import collate_state_batch
from syntree.models.policy import SynTreePolicy


def collate_pyg(items):
    """Canonical V2 batch collation (#6 ARCH-02): single path via
    syntree.core.schema.collate_state_batch (validated, versioned,
    fail-closed - rejects legacy 67-dim Franken tensors)."""
    return collate_state_batch(items)


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
    # V2 (#5 ARCH-01): the synthetic CrossDockedDataset smoke loader was
    # purged; a minimal synthetic pocket/ligand graph is built directly.
    torch.manual_seed(0)
    n_pocket, n_ligand = 24, 6
    sample = Data(
        pocket_pos=torch.randn(n_pocket, 3) * 8.0,
        pocket_z=torch.randint(1, 30, (n_pocket,)),
        ligand_pos=torch.randn(n_ligand, 3) * 2.0,
        ligand_z=torch.randint(1, 30, (n_ligand,)),
        handle_features=torch.randn(1, 64),
        handle_pos=torch.randn(1, 3),
        handle_nodes=torch.zeros(1, dtype=torch.long),
        global_features=torch.randn(1, 4),
    )
    batch = collate_pyg([sample])
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