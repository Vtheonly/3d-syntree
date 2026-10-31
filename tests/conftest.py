"""Shared pytest fixtures for the 3D-SynTree test suite."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rdkit import RDLogger

RDLogger.DisableLog("rdkit.*")


@pytest.fixture(scope="session")
def repo_root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def tiny_config() -> dict:
    """CPU-friendly model/training configuration used across tests."""
    return {
        "system": {
            "project_name": "3D-SynTree-Test",
            "seed": 42,
            "device": "cpu",
            "mixed_precision": "fp32",
            "num_workers": 0,
        },
        "huggingface": {
            "enabled": False,
            "repo_id": "test/3d-syntree-checkpoints",
            "push_every_n_epochs": 2,
            "private": True,
        },
        "data": {
            "dataset_name": "crossdocked2020",
            "data_dir": "",  # filled by assets fixture
            "synthon_catalog_path": "",  # filled by assets fixture
            "max_pocket_distance": 8.0,
            "max_steps_per_molecule": 2,
            "batch_size": 4,
            "accumulate_grad_batches": 1,
            "val_fraction": 0.25,
            "synthetic_fallback": True,
            "synthetic_samples": 12,
            # Explicit smoke-test opt-out of the production fail-closed guard
            # (ResilientTrainer refuses synthetic/stub data by default).
            "require_real_data": False,
            "min_real_samples": 2,
        },
        "model": {
            "hidden_dim": 32,
            "num_equivariant_layers": 2,
            "num_radial_basis": 12,
            "cutoff_radius": 5.0,
            "synthon_embedding_dim": 32,
            "num_attention_heads": 4,
            "max_atomic_number": 100,
            "dropout": 0.0,
        },
        "training": {
            "max_epochs": 2,
            "time_budget_hours": 0.25,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "lr_scheduler": "cosine_warmup",
            "warmup_epochs": 1,
            "grad_clip_norm": 1.0,
            "keep_last_n_checkpoints": 2,
            "loss_weights": {"synthon_ce": 1.0, "torsion_nll": 0.5, "steric_clash": 0.1},
            "eval_interval_epochs": 1,
        },
        "evaluation": {
            "num_test_pockets": 4,
            "docking_engine": "gnina",
            "exhaustiveness": 1,
            "run_posebusters": True,
            "run_aizynthfinder": True,
        },
    }


@pytest.fixture(scope="session")
def assets_dir(tmp_path_factory):
    """Offline asset bundle: catalog parquet + sample pocket PDB."""
    base = tmp_path_factory.mktemp("assets")
    data_dir = str(base / "data")
    out = str(base / "data" / "crossdocked")
    os.makedirs(out, exist_ok=True)

    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    from syntree.chemistry.reactions import ReactionEngine

    engine = ReactionEngine()
    synthons = [
        # sp3-rich
        ("OC(=O)C1CCCCC1", "carboxylic_acid"),
        ("CC(C)(C)C(=O)O", "carboxylic_acid"),
        ("C1CC(N)CC1", "primary_secondary_amine"),
        ("NCC1CCCCC1", "primary_secondary_amine"),
        ("CC1(O)CCCCC1", "alcohol"),
        ("CC(C)CO", "alcohol"),
        ("C1CCC(CC1)C=O", "aldehyde"),
        ("CCCCCC#C", "alkyne"),
        ("CCCCCN=[N+]=[N-]", "azide"),
        # aryl partners (Fsp3-exempt)
        ("Brc1ccc(F)cc1", "aryl_halide"),
        ("B(c1ccccc1)(O)O", "boronic_acid"),
        # tasklist Priority 4 families: sulfonamide + sp3 alkylation
        ("CS(=O)(=O)Cl", "sulfonyl_chloride"),
        ("Cc1ccc(S(=O)(=O)Cl)cc1", "sulfonyl_chloride"),
        ("NCCBr", "alkyl_halide"),
        ("CCBr", "alkyl_halide"),
    ]
    rows = []
    for idx, (smiles, handle) in enumerate(synthons):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert handle in engine.handle_types(mol)
        rows.append(
            {
                "id": f"TEST-{idx:04d}",
                "smiles": smiles,
                "fsp3": float(Lipinski.FractionCSP3(mol)),
                "mw": float(Descriptors.MolWt(mol)),
                "primary_handle": handle,
            }
        )
    catalog_path = os.path.join(data_dir, "enamine_3d_subset.parquet")
    pd.DataFrame(rows).to_parquet(catalog_path, index=False)

    pocket_path = os.path.join(out, "sample_pocket.pdb")
    with open(pocket_path, "w") as f:
        f.write(
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C\n"
            "ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O\n"
            "ATOM      5  N   GLY A   2       3.315   1.545   0.000  1.00 20.00           N\n"
            "ATOM      6  CA  GLY A   2       3.866   2.906   0.000  1.00 20.00           C\n"
            "ATOM      7  C   GLY A   2       4.417   4.267   0.000  1.00 20.00           C\n"
            "ATOM      8  O   GLY A   2       3.680   5.254   0.000  1.00 20.00           O\n"
            "TER\nEND\n"
        )
    return {"data_dir": data_dir, "crossdocked_dir": out, "catalog_path": catalog_path}


@pytest.fixture(scope="session")
def smoke_trajectory_path(assets_dir, tmp_path_factory):
    """Small real trajectory .pt dataset for trainer smoke tests.

    V2 (#5 ARCH-01): the synthetic CrossDockedDataset smoke loader was
    purged; trainer tests use the remaining production backend
    (trajectory_pt) fed by RetrosyntheticTrajectoryBuilder states, so the
    smoke path exercises the real pocket+ligand featurization pipeline.
    """
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from syntree.chemistry.catalog import SynthonCatalog
    from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder

    catalog = SynthonCatalog(
        assets_dir["catalog_path"], embedding_dim=32, validate_handles=False
    )
    builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=2)
    pocket = Chem.MolFromPDBBlock(
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C\n"
        "ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O\n"
        "ATOM      5  N   GLY A   2       3.315   1.545   0.000  1.00 20.00           N\n"
        "ATOM      6  CA  GLY A   2       3.866   2.906   0.000  1.00 20.00           C\n"
        "ATOM      7  C   GLY A   2       4.417   4.267   0.000  1.00 20.00           C\n"
        "ATOM      8  O   GLY A   2       3.680   5.254   0.000  1.00 20.00           O\n"
        "TER\nEND\n",
        removeHs=False,
    )
    products = [
        "O=C(NC1CCCCC1)C1CCCCC1",
        "CC(C)(C)C(=O)NCC1CCCCC1",
        "CC(C)COC(=O)C1CCCCC1",
        "CC(C)COC(=O)C(C)(C)C",
    ]
    states = []
    n = 0
    for _ in range(12):
        for smi in products:
            n += 1
            ligand = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(ligand, randomSeed=42)
            traj_states, _meta = builder.build(
                ligand, pocket, trajectory_id=f"smoke_{n:03d}"
            )
            states.extend(traj_states)
    # Deterministic hash-bucket split: ensure every split is non-empty and
    # train is large enough for the smoke training runs.
    n_train = sum(1 for s in states if int(s.trajectory_hash.item()) % 1000 < 800)
    n_val = sum(
        1 for s in states
        if 800 <= int(s.trajectory_hash.item()) % 1000 < 900
    )
    assert n_train >= 20 and n_val >= 2, (
        f"smoke trajectory fixture too small: train={n_train}, val={n_val}"
    )
    out = tmp_path_factory.mktemp("smoke_traj") / "smoke_trajectories.pt"
    torch.save(states, str(out))
    return str(out)


@pytest.fixture(scope="session")
def tiny_config_with_assets(tiny_config, assets_dir, smoke_trajectory_path):
    cfg = json.loads(json.dumps(tiny_config))
    cfg["data"]["data_dir"] = assets_dir["crossdocked_dir"]
    cfg["data"]["synthon_catalog_path"] = assets_dir["catalog_path"]
    # V2 (#5 ARCH-01): trajectory_pt backend replaces the purged synthetic
    # CrossDocked loader as the smoke-test data source.
    cfg["data"]["backend"] = "trajectory_pt"
    cfg["data"]["trajectory_dataset_path"] = smoke_trajectory_path
    cfg["data"].pop("synthetic_fallback", None)
    cfg["data"].pop("synthetic_samples", None)
    return cfg


@pytest.fixture()
def workspace(tmp_path):
    """Fresh working directory for tests that write artifacts."""
    return str(tmp_path)
