"""Integration tests for the unified CLI and project scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(args, cwd=REPO_ROOT, timeout=600):
    return subprocess.run(
        [sys.executable, os.path.join(cwd, "main.py")] + args,
        cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )


class TestInfoMode:
    def test_info_lists_reactions(self):
        proc = run_cli(["--mode", "info"])
        assert proc.returncode == 0
        assert "amide_coupling" in proc.stdout
        assert "click_triazole" in proc.stdout

    def test_missing_config_fails_cleanly(self):
        proc = run_cli(["--mode", "info", "--config", "nope.json"])
        assert proc.returncode == 2
        assert "config not found" in proc.stderr


class TestTrainMode:
    def test_short_train_run(self, tiny_config_with_assets, tmp_path):
        cfg_path = tmp_path / "cfg.json"
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_samples"] = 8
        cfg["training"]["max_epochs"] = 1
        cfg["system"]["num_workers"] = 0
        cfg_path.write_text(json.dumps(cfg))

        out_dir = tmp_path / "exp"
        proc = run_cli([
            "--mode", "train", "--config", str(cfg_path),
            "--output-dir", str(out_dir),
        ])
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert os.path.exists(out_dir / "history.json")
        assert "training summary" in proc.stdout


class TestGenerateMode:
    def test_generate_with_trained_checkpoint(self, tiny_config_with_assets,
                                               tmp_path):
        cfg_path = tmp_path / "cfg.json"
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_samples"] = 8
        cfg["training"]["max_epochs"] = 1
        cfg["system"]["num_workers"] = 0
        cfg_path.write_text(json.dumps(cfg))

        out_dir = tmp_path / "exp"
        gen_dir = tmp_path / "gen"
        # Train 1 epoch.
        proc = run_cli(["--mode", "train", "--config", str(cfg_path),
                        "--output-dir", str(out_dir)])
        assert proc.returncode == 0, proc.stderr[-2000:]
        # Generate using the checkpoint produced by the run above.
        proc = run_cli([
            "--mode", "generate", "--config", str(cfg_path),
            "--output-dir", str(gen_dir), "--resume-auto", "--num-ligands", "2",
            "--ckpt-dir", str(out_dir / "checkpoints"),
        ])
        assert proc.returncode == 0, proc.stderr[-2000:]
        sdfs = [f for f in os.listdir(gen_dir) if f.endswith(".sdf")]
        assert len(sdfs) == 2
        assert os.path.exists(gen_dir / "batch_summary.json")

    def test_missing_catalog_fails_cleanly(self, tmp_path):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({
            "data": {"data_dir": "x", "synthon_catalog_path": "missing.parquet"},
        }))
        proc = run_cli(["--mode", "generate", "--config", str(cfg_path)])
        assert proc.returncode == 2
        assert "download_assets" in proc.stderr


class TestEvaluateMode:
    def test_evaluate_generated(self, tiny_config_with_assets, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        gen_dir = tmp_path / "gen"
        gen_dir.mkdir()
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)CO"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        w = Chem.SDWriter(str(gen_dir / "ligand_000.sdf"))
        w.write(mol)
        w.close()
        (gen_dir / "recipe_000.json").write_text(
            json.dumps([{"step": 0, "action": "seed", "synthon_id": "X"}])
        )

        cfg_path = tmp_path / "cfg.json"
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg_path.write_text(json.dumps(cfg))
        proc = run_cli([
            "--mode", "evaluate", "--config", str(cfg_path),
            "--sdf-dir", str(gen_dir),
            "--output-dir", str(tmp_path / "eval"),
        ])
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "chemical_validity" in proc.stdout

    def test_evaluate_empty_dir_fails(self, tmp_path, tiny_config_with_assets):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(tiny_config_with_assets))
        empty = tmp_path / "empty"
        empty.mkdir()
        proc = run_cli(["--mode", "evaluate", "--config", str(cfg_path),
                        "--sdf-dir", str(empty)])
        assert proc.returncode == 2


class TestDownloadAssets:
    def test_offline_catalog_creation(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts",
                                          "download_assets.py"),
             "--skip-download", "--output-dir", str(tmp_path),
             "--catalog-copies", "3"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert os.path.exists(tmp_path / "enamine_3d_subset.parquet")
        assert os.path.exists(tmp_path / "crossdocked" / "sample_pocket.pdb")
        assert os.path.exists(tmp_path / "assets_manifest.json")

        import pandas as pd

        df = pd.read_parquet(tmp_path / "enamine_3d_subset.parquet")
        assert len(df) > 0
        assert set(df.columns) >= {"id", "smiles", "fsp3", "mw", "primary_handle"}

    def test_idempotent_second_run(self, tmp_path):
        cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts",
                                            "download_assets.py"),
               "--skip-download", "--output-dir", str(tmp_path),
               "--catalog-copies", "3"]
        subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, timeout=300)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=300)
        assert "already present" in proc.stdout


class TestBenchmarksScript:
    def test_runs_over_generated_dir(self, tiny_config_with_assets, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        gen_dir = tmp_path / "gen"
        gen_dir.mkdir()
        mol = Chem.AddHs(Chem.MolFromSmiles("OC(=O)C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=1)
        w = Chem.SDWriter(str(gen_dir / "ligand_000.sdf"))
        w.write(mol)
        w.close()
        (gen_dir / "recipe_000.json").write_text(
            json.dumps([{"step": 0, "action": "seed", "synthon_id": "A"}])
        )

        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(tiny_config_with_assets))
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts",
                                          "run_benchmarks.py"),
             "--sdf-dir", str(gen_dir), "--config", str(cfg_path),
             "--output-dir", str(tmp_path / "eval")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "BENCHMARK SUMMARY" in proc.stdout
        assert os.path.exists(tmp_path / "eval" / "evaluation_report.json")


class TestPackaging:
    def test_package_importable(self):
        import syntree

        assert syntree.__version__ == "0.1.0"

    def test_all_submodules_import(self):
        import syntree.chemistry
        import syntree.data
        import syntree.engine
        import syntree.models
        import syntree.utils

        assert syntree.chemistry.ReactionEngine
        assert syntree.data.CrossDockedDataset
        assert syntree.engine.ResilientTrainer
        assert syntree.models.SynTreePolicy
        assert syntree.utils.CheckpointManager

    def test_requirements_file_exists(self):
        assert os.path.exists(os.path.join(REPO_ROOT, "requirements.txt"))

    def test_configs_are_valid_json(self):
        for name in os.listdir(os.path.join(REPO_ROOT, "configs")):
            if name.endswith(".json"):
                with open(os.path.join(REPO_ROOT, "configs", name)) as f:
                    json.load(f)


class TestRLMode:
    def test_rl_discovers_test_pockets_dir(self, tiny_config_with_assets,
                                           tmp_path):
        """Landmine 3: with the HF shard backend the data dir has no loose
        pocket PDBs; --mode rl must discover the test_pockets/ folder staged
        by download_assets.py and run a complete (tiny) PPO cycle."""
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_samples"] = 8
        cfg["system"]["num_workers"] = 0
        cfg["reinforcement_learning"] = {
            "enabled": True,
            "episodes": 1,
            "rollout_episodes": 1,
            "minibatch_size": 2,
            "ppo_epochs": 1,
            "learning_rate": 1e-4,
            "reward": {"docking_weight": 0.0},
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))

        out_dir = tmp_path / "exp"
        proc = run_cli(["--mode", "train", "--config", str(cfg_path),
                       "--output-dir", str(out_dir)])
        assert proc.returncode == 0, proc.stderr[-2000:]

        # Stage the HF-style layout: pockets live in test_pockets/.
        pockets_dir = out_dir / "test_pockets"
        pockets_dir.mkdir(parents=True, exist_ok=True)
        src_pocket = os.path.join(cfg["data"]["data_dir"], "sample_pocket.pdb")
        with open(src_pocket) as f:
            payload = f.read()
        with open(pockets_dir / "rl1_pocket.pdb", "w") as f:
            f.write(payload)

        # The data dir itself must NOT contain loose pocket PDBs (the whole
        # point of the fix): only the nested test_pockets/ folder.
        rl_out = tmp_path / "rl"
        proc = run_cli([
            "--mode", "rl", "--config", str(cfg_path),
            "--output-dir", str(rl_out), "--resume-auto",
            "--ckpt-dir", str(out_dir / "checkpoints"),
        ])
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "RL using 1 pockets" in proc.stdout
        assert os.path.exists(rl_out / "rl_history.json")

    def test_rl_explicit_pocket_dir_flag(self, tiny_config_with_assets,
                                         tmp_path):
        """--pocket-dir overrides every other discovery source."""
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_samples"] = 8
        cfg["system"]["num_workers"] = 0
        cfg["reinforcement_learning"] = {
            "enabled": True,
            "episodes": 1,
            "rollout_episodes": 1,
            "minibatch_size": 2,
            "ppo_epochs": 1,
            "learning_rate": 1e-4,
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))

        out_dir = tmp_path / "exp"
        proc = run_cli(["--mode", "train", "--config", str(cfg_path),
                       "--output-dir", str(out_dir)])
        assert proc.returncode == 0, proc.stderr[-2000:]

        custom = tmp_path / "custom_targets"
        custom.mkdir()
        src_pocket = os.path.join(cfg["data"]["data_dir"], "sample_pocket.pdb")
        with open(src_pocket) as f:
            payload = f.read()
        with open(custom / "x_pocket.pdb", "w") as f:
            f.write(payload)

        rl_out = tmp_path / "rl"
        proc = run_cli([
            "--mode", "rl", "--config", str(cfg_path),
            "--output-dir", str(rl_out), "--resume-auto",
            "--ckpt-dir", str(out_dir / "checkpoints"),
            "--pocket-dir", str(custom),
        ])
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert f"RL using 1 pockets from {custom}" in proc.stdout
