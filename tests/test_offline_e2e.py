"""End-to-end offline verification for the RTX 6000 / Kaggle deployment
contract (tasklist Step 7 + the 'IT IS OFFLINE' constraint):

* CLI session 1 trains on a mounted offline dataset and checkpoints
* CLI session 2 resumes from those checkpoints (`--resume-auto`)
* inference (`--mode generate`) runs from the restored weights
* the entire flow is executed with **socket creation disabled** to prove
  zero network dependency
* keep_last_n checkpoint pruning bounds the disk quota
* the time budget triggers a graceful checkpoint-and-exit
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.shard_and_upload import write_shards  # noqa: E402
from tests.test_offline_kaggle import _tiny_sample  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(args, env_extra=None, timeout=900):
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env["HF_HUB_OFFLINE"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "main.py")] + args,
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, env=env,
    )


@pytest.fixture(scope="module")
def offline_env(tmp_path_factory):
    """A complete offline asset bundle: shards + catalog + pocket."""
    base = tmp_path_factory.mktemp("offline_env")
    dataset = base / "dataset"
    dataset.mkdir()

    # Offline sharded dataset (train/val) from the real writer.
    train = [_tiny_sample(i) for i in range(8)]
    val = [_tiny_sample(100 + i) for i in range(4)]
    write_shards(train, "train", str(dataset))
    write_shards(val, "val", str(dataset))

    # Catalog + pocket reuse the conftest asset bundle.
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    from syntree.chemistry.reactions import ReactionEngine

    engine = ReactionEngine()
    synthons = [
        ("OC(=O)C1CCCCC1", "carboxylic_acid"),
        ("C1CC(N)CC1", "primary_secondary_amine"),
        ("CC(C)CO", "alcohol"),
        ("C1CCC(CC1)C=O", "aldehyde"),
        ("CCCCCC#C", "alkyne"),
        ("CCCCCN=[N+]=[N-]", "azide"),
        ("Brc1ccc(F)cc1", "aryl_halide"),
        ("B(c1ccccc1)(O)O", "boronic_acid"),
        ("CS(=O)(=O)Cl", "sulfonyl_chloride"),
        ("CCBr", "alkyl_halide"),
    ]
    rows = []
    for idx, (smiles, handle) in enumerate(synthons):
        mol = Chem.MolFromSmiles(smiles)
        assert handle in engine.handle_types(mol)
        rows.append({
            "id": f"OFF-{idx:04d}",
            "smiles": smiles,
            "fsp3": float(Lipinski.FractionCSP3(mol)),
            "mw": float(Descriptors.MolWt(mol)),
            "primary_handle": handle,
        })
    catalog_path = dataset / "enamine_3d_subset.parquet"
    pd.DataFrame(rows).to_parquet(catalog_path, index=False)

    pockets = base / "targets"
    pockets.mkdir()
    with open(pockets / "1a9u_pocket.pdb", "w") as f:
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
    return {"base": str(base), "dataset": str(dataset),
            "catalog": str(catalog_path), "pockets": str(pockets)}


def _config(offline_env, tmp_path, **training_overrides):
    return {
        "system": {
            "project_name": "offline-e2e",
            "seed": 42,
            "device": "cpu",
            "mixed_precision": "fp32",
            "num_workers": 0,
        },
        "huggingface": {"enabled": False},
        "data": {
            "backend": "kaggle_offline",
            "data_dir": offline_env["dataset"],
            "synthon_catalog_path": offline_env["catalog"],
            "batch_size": 4,
            "val_fraction": 0.25,
            "max_steps_per_molecule": 2,
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
            "max_epochs": 1,
            "time_budget_hours": 0.25,
            "learning_rate": 3e-4,
            "warmup_epochs": 1,
            "keep_last_n_checkpoints": 2,
            "eval_interval_epochs": 1,
        },
        "catalog": {"encoder": "morgan2d", "cache_embeddings": False},
        **training_overrides,
    }


class TestOfflineSessionChaining:
    def test_train_resume_generate_with_network_disabled(
        self, offline_env, tmp_path
    ):
        """The exact tasklist Step 7 flow, with sockets hard-disabled."""
        # Blocking sockets for the CLI child processes via a sitecustomize
        # shim: any actual socket *creation* raises (subclassing still works,
        # so ssl/torch imports are unaffected).
        guard = tmp_path / "no_network.py"
        guard.write_text(
            "import socket\n"
            "class _BlockedSocket(socket.socket):\n"
            "    def __init__(self, *a, **k):\n"
            "        raise OSError('network disabled for offline test')\n"
            "socket.socket = _BlockedSocket\n"
            "def _blocked(*a, **k):\n"
            "    raise OSError('network disabled for offline test')\n"
            "socket.create_connection = _blocked\n"
            "socket.getaddrinfo = _blocked\n"
        )

        cfg = _config(offline_env, tmp_path)
        cfg_path = tmp_path / "offline.json"
        cfg_path.write_text(json.dumps(cfg))

        # sitecustomize is auto-imported at interpreter startup when its
        # directory is on sys.path, so the guard patches sockets before
        # main.py (and torch) ever import.
        (tmp_path / "sitecustomize.py").write_text(
            "import no_network  # noqa: F401 - blocks socket creation\n"
        )
        env_guard = {"PYTHONPATH": str(tmp_path) + os.pathsep + REPO_ROOT}

        working = tmp_path / "kaggle_working"
        working.mkdir()

        # ---- Session 1: fresh training ------------------------------------
        proc = run_cli(
            ["--mode", "train", "--config", str(cfg_path),
             "--output-dir", str(working), "--fresh"],
            env_extra=env_guard,
        )
        assert proc.returncode == 0, proc.stderr[-4000:]
        assert "kaggle_offline backend: preloading" in proc.stdout

        ckpt_dir = working / "checkpoints"
        assert (ckpt_dir / "checkpoint_epoch_0.pt").exists()
        manifest = json.loads((ckpt_dir / "manifest.json").read_text())
        assert manifest["latest_epoch"] == 0
        progress = json.loads((ckpt_dir / "progress.json").read_text())
        assert progress["completed_epochs"] == [0]

        # The checkpoint records the offline catalog signature.
        payload = torch.load(
            ckpt_dir / "checkpoint_epoch_0.pt",
            map_location="cpu", weights_only=False,
        )
        assert payload["catalog_signature"]["encoder"] == "morgan2d"

        # ---- Session 2: resume-auto continues -----------------------------
        cfg["training"]["max_epochs"] = 2
        cfg_path.write_text(json.dumps(cfg))
        proc2 = run_cli(
            ["--mode", "train", "--config", str(cfg_path),
             "--output-dir", str(working), "--resume-auto"],
            env_extra=env_guard,
        )
        assert proc2.returncode == 0, proc2.stderr[-4000:]
        assert "Restored checkpoint" in proc2.stdout or "resume_epoch=1" in proc2.stdout

        progress2 = json.loads((ckpt_dir / "progress.json").read_text())
        assert progress2["completed_epochs"] == [0, 1]
        assert (ckpt_dir / "checkpoint_epoch_1.pt").exists()

        # ---- Inference from the restored weights ---------------------------
        proc3 = run_cli(
            ["--mode", "generate", "--config", str(cfg_path),
             "--ckpt-dir", str(ckpt_dir),
             "--pocket", os.path.join(offline_env["pockets"], "1a9u_pocket.pdb"),
             "--output-dir", str(tmp_path / "outputs"),
             "--num-ligands", "2", "--resume-auto"],
            env_extra=env_guard,
        )
        assert proc3.returncode == 0, proc3.stderr[-4000:]
        outputs = list((tmp_path / "outputs").glob("ligand_*.sdf"))
        assert len(outputs) == 2

    def test_keep_last_n_prunes_old_checkpoints(self, offline_env, tmp_path):
        cfg = _config(offline_env, tmp_path)
        cfg["training"]["max_epochs"] = 4
        cfg["training"]["keep_last_n_checkpoints"] = 2
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))

        working = tmp_path / "working"
        proc = run_cli(
            ["--mode", "train", "--config", str(cfg_path),
             "--output-dir", str(working), "--fresh"],
        )
        assert proc.returncode == 0, proc.stderr[-4000:]

        epochs = sorted(
            p.name for p in (working / "checkpoints").glob("checkpoint_epoch_*.pt")
        )
        assert len(epochs) == 2, f"expected pruned checkpoints, got {epochs}"
        assert "checkpoint_epoch_3.pt" in epochs
        assert "checkpoint_epoch_2.pt" in epochs
        assert not (working / "checkpoints" / "checkpoint_epoch_0.pt").exists()

    def test_time_budget_graceful_exit(self, offline_env, tmp_path):
        """The 11.2h-style budget: expiring mid-run exits cleanly. With a
        near-zero budget the exit happens before the first epoch completes
        (nothing to checkpoint yet - progress is only recorded for completed
        epochs); with a realistic budget the checkpoint lands as usual."""
        cfg = _config(offline_env, tmp_path)
        cfg["training"]["max_epochs"] = 50
        cfg["training"]["time_budget_hours"] = 1e-5  # ~0.036 s -> expires fast
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))

        working = tmp_path / "working"
        proc = run_cli(
            ["--mode", "train", "--config", str(cfg_path),
             "--output-dir", str(working), "--fresh"],
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        # Degenerate ultra-short budget: exits before epoch 1 finishes. The
        # graceful-exit path must still produce a history artifact and an
        # explicit message (never a hard kill).
        assert "time budget expired" in proc.stdout
        assert (working / "history.json").exists()
        # No progress was recorded for a partially-run epoch (fail-closed
        # progress semantics: only completed epochs are checkpointed).
        ckpt_dir = working / "checkpoints"
        if (ckpt_dir / "progress.json").exists():
            progress = json.loads((ckpt_dir / "progress.json").read_text())
            assert progress["completed_epochs"] == []

    def test_already_completed_resume(self, offline_env, tmp_path):
        cfg = _config(offline_env, tmp_path)
        cfg["training"]["max_epochs"] = 1
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        working = tmp_path / "working"
        assert run_cli(
            ["--mode", "train", "--config", str(cfg_path),
             "--output-dir", str(working), "--fresh"]
        ).returncode == 0

        proc = run_cli(
            ["--mode", "train", "--config", str(cfg_path),
             "--output-dir", str(working), "--resume-auto"]
        )
        assert proc.returncode == 0
        assert "already meets or exceeds max_epochs" in proc.stdout

    def test_no_tokens_in_repo(self):
        """Token hygiene: no live credentials in any tracked file."""
        import re

        secret_patterns = re.compile(
            r"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        )
        offenders = []
        for path in Path(REPO_ROOT).rglob("*"):
            if not path.is_file() or any(
                part in (".git", ".secrets", "__pycache__", ".pytest_cache",
                         "hf_cache", ".venv", "raw_data")
                for part in path.parts
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if secret_patterns.search(text) and path.name not in (
                "test_notebook_contract.py", "test_offline_e2e.py",
            ):
                offenders.append(str(path))
        assert offenders == []
