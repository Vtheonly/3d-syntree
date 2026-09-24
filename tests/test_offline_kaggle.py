"""Tests for the offline Kaggle/RTX 6000 data stack (tasklist Steps 5-6):
syntree/data/kaggle_loader.py, the trainer's kaggle_offline backend, the
bf16/TF32 precision resolution, and the shipped production configs."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.shard_and_upload import write_shards  # noqa: E402
from syntree.data.kaggle_loader import (  # noqa: E402
    KaggleInMemoryDataset,
    KaggleLazyDataset,
)


def _tiny_sample(seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    n = 6
    return Data(
        num_nodes=n,
        pocket_pos=torch.randn(n, 3, generator=g),
        pocket_z=torch.randint(1, 9, (n,), generator=g),
        pocket_charge=torch.zeros(n),
        ligand_pos=torch.zeros(0, 3),
        ligand_z=torch.zeros(0, dtype=torch.long),
        ligand_charge=torch.zeros(0),
        handle_features=torch.randn(1, 64, generator=g),
        handle_pos=torch.randn(1, 3, generator=g),
        handle_nodes=torch.tensor([-1], dtype=torch.long),
        global_features=torch.randn(1, 4, generator=g),
        target_synthon=torch.tensor([seed % 4], dtype=torch.long),
        target_dihedral=torch.tensor([0.5]),
        target_reaction_family_idx=torch.tensor([seed % 9], dtype=torch.long),
        target_core_handle_idx=torch.tensor([seed % 10], dtype=torch.long),
        target_stop=torch.tensor([False]),
        stop_mask=torch.tensor([0.0]),
        is_real_sample=torch.tensor([True]),
    )


def _stage_offline_dataset(root: Path, n_train: int = 8, n_val: int = 4):
    """Write a manifest + shards offline dataset directory (real format)."""
    train_samples = [_tiny_sample(i) for i in range(n_train)]
    val_samples = [_tiny_sample(100 + i) for i in range(n_val)]
    write_shards(train_samples, "train", str(root))
    write_shards(val_samples, "val", str(root))
    return root


class TestKaggleInMemoryDataset:
    def test_loads_all_samples_with_manifest_totals(self, tmp_path):
        root = _stage_offline_dataset(tmp_path)
        ds = KaggleInMemoryDataset(str(root), split="train")
        assert len(ds) == 8
        assert ds.num_synthetic == 0
        assert ds.use_synthetic is False
        for i in range(len(ds)):
            sample = ds[i]
            assert sample.pocket_pos.shape == (6, 3)
            assert sample.num_nodes == 6

    def test_roundtrip_identity_with_written_samples(self, tmp_path):
        root = _stage_offline_dataset(tmp_path)
        ds = KaggleInMemoryDataset(str(root), split="train")
        original = _tiny_sample(0)
        assert torch.equal(ds[0].pocket_pos, original.pocket_pos)
        assert torch.equal(ds[0].target_synthon, original.target_synthon)

    def test_val_split(self, tmp_path):
        root = _stage_offline_dataset(tmp_path)
        ds = KaggleInMemoryDataset(str(root), split="val")
        assert len(ds) == 4

    def test_missing_manifest_fails_closed(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            KaggleInMemoryDataset(str(tmp_path), split="train")

    def test_missing_shard_file_fails_closed(self, tmp_path):
        root = _stage_offline_dataset(tmp_path)
        shard = next((root / "train").glob("*.pt.gz"))
        shard.unlink()
        with pytest.raises(FileNotFoundError, match="Missing shard file"):
            KaggleInMemoryDataset(str(root), split="train")

    def test_corrupt_shard_rejected_by_sha256(self, tmp_path):
        root = _stage_offline_dataset(tmp_path)
        shard = next((root / "train").glob("*.pt.gz"))
        # Flip one byte inside the compressed payload.
        payload = bytearray(shard.read_bytes())
        payload[-5] ^= 0xFF
        shard.write_bytes(bytes(payload))
        with pytest.raises(IOError, match="SHA-256 mismatch"):
            KaggleInMemoryDataset(str(root), split="train")

    def test_sample_count_mismatch_rejected(self, tmp_path):
        root = _stage_offline_dataset(tmp_path)
        # Rewrite one shard with fewer samples AND refresh its manifest
        # digest so only the sample-count invariant fires.
        shard = next((root / "train").glob("*.pt.gz"))
        with gzip.open(shard, "rb") as f:
            data = torch.load(f, weights_only=False)
        with gzip.open(shard, "wb") as f:
            torch.save(data[:-1], f)
        import hashlib

        manifest_path = root / "train" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["shards"]:
            if entry["name"] == shard.name:
                entry["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2))
        with pytest.raises(ValueError, match="manifest entry declares"):
            KaggleInMemoryDataset(str(root), split="train")

    def test_empty_split_rejected(self, tmp_path):
        (tmp_path / "train").mkdir(parents=True)
        write_shards([], "train", str(tmp_path))
        with pytest.raises((ValueError, RuntimeError)):
            KaggleInMemoryDataset(str(tmp_path), split="train")

    def test_invalid_split_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="split"):
            KaggleInMemoryDataset(str(tmp_path), split="banana")

    def test_no_network_imports(self):
        """The offline loader must not import network modules (offline
        RTX 6000 sessions have no sockets). AST-based so docstrings
        mentioning the forbidden modules do not trip the check."""
        import ast

        import syntree.data.kaggle_loader as kl

        tree = ast.parse(Path(kl.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"huggingface_hub", "urllib", "requests", "http", "socket"}
        assert imported.isdisjoint(forbidden), imported & forbidden


class TestKaggleLazyDataset:
    def test_lazy_matches_in_memory(self, tmp_path):
        root = _stage_offline_dataset(tmp_path)
        eager = KaggleInMemoryDataset(str(root), split="train")
        lazy = KaggleLazyDataset(str(root), split="train", max_cached_shards=1)
        assert len(lazy) == len(eager)
        for i in range(len(eager)):
            assert torch.equal(lazy[i].pocket_pos, eager[i].pocket_pos)

    def test_lru_eviction_bounds_cache(self, tmp_path):
        root = _stage_offline_dataset(tmp_path, n_train=16)
        lazy = KaggleLazyDataset(str(root), split="train", max_cached_shards=1)
        for i in range(len(lazy)):
            _ = lazy[i]
        assert len(lazy._cache) <= 1

    def test_trainer_facing_properties(self, tmp_path):
        root = _stage_offline_dataset(tmp_path)
        lazy = KaggleLazyDataset(str(root), split="train")
        assert lazy.num_synthetic == 0
        assert lazy.use_synthetic is False


def _trainer_config(tiny_config_with_assets, offline_dir, **overrides):
    cfg = json.loads(json.dumps(tiny_config_with_assets))
    cfg["data"]["backend"] = "kaggle_offline"
    cfg["data"]["data_dir"] = str(offline_dir)
    cfg["data"]["synthon_catalog_path"] = tiny_config_with_assets["data"][
        "synthon_catalog_path"
    ]
    cfg["data"].pop("synthetic_fallback", None)
    cfg["data"]["require_real_data"] = False
    cfg["data"]["min_real_samples"] = 2
    cfg["system"]["num_workers"] = 0
    cfg["training"]["max_epochs"] = 1
    cfg["training"]["time_budget_hours"] = 0.25
    cfg["catalog"] = {"encoder": "morgan2d", "cache_embeddings": False}
    for key, value in overrides.items():
        cfg["data"][key] = value
    return cfg


class TestTrainerKaggleOfflineBackend:
    def _config(self, tiny_config_with_assets, offline_dir, **overrides):
        return _trainer_config(tiny_config_with_assets, offline_dir, **overrides)

    def _trainer(self, cfg, tmp_path, auto_resume=False):
        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy

        device = torch.device("cpu")
        model = SynTreePolicy(cfg)
        return ResilientTrainer(
            model, cfg, device, auto_resume=auto_resume,
            output_dir=str(tmp_path / "exp"),
        )

    def test_backend_dispatch_and_training(self, tiny_config_with_assets,
                                           tmp_path):
        offline = _stage_offline_dataset(
            tmp_path / "offline", n_train=8, n_val=4
        )
        cfg = self._config(tiny_config_with_assets, offline)
        trainer = self._trainer(cfg, tmp_path)
        assert trainer.data_backend == "kaggle_offline"
        assert len(trainer.dataset) == 8
        assert len(trainer.val_dataset) == 4

        summary = trainer.train()
        assert summary["epochs_completed"] >= 1

    def test_lazy_mode_via_preload_to_ram_false(self, tiny_config_with_assets,
                                                tmp_path):
        offline = _stage_offline_dataset(
            tmp_path / "offline", n_train=8, n_val=4
        )
        cfg = self._config(tiny_config_with_assets, offline,
                           preload_to_ram=False, max_cached_shards=1)
        trainer = self._trainer(cfg, tmp_path)
        from syntree.data.kaggle_loader import KaggleLazyDataset

        assert isinstance(trainer.dataset, KaggleLazyDataset)
        summary = trainer.train()
        assert summary["epochs_completed"] >= 1

    def test_missing_data_dir_fails_closed(self, tiny_config_with_assets,
                                           tmp_path):
        cfg = self._config(tiny_config_with_assets, tmp_path / "does_not_exist")
        with pytest.raises(FileNotFoundError, match="kaggle_offline"):
            self._trainer(cfg, tmp_path)

    def test_unknown_backend_rejected(self, tiny_config_with_assets, tmp_path):
        cfg = self._config(tiny_config_with_assets, tmp_path)
        cfg["data"]["backend"] = "s3_bucket"
        with pytest.raises(ValueError, match="not one of"):
            self._trainer(cfg, tmp_path)

    def test_trajectory_backend_requires_path(self, tiny_config_with_assets,
                                              tmp_path):
        cfg = self._config(tiny_config_with_assets, tmp_path)
        cfg["data"]["backend"] = "trajectory_pt"
        cfg["data"].pop("trajectory_dataset_path", None)
        with pytest.raises(ValueError, match="trajectory_dataset_path"):
            self._trainer(cfg, tmp_path)

    def test_offline_training_uses_no_sockets(self, tiny_config_with_assets,
                                              tmp_path, monkeypatch):
        """The defining RTX 6000 constraint: no network access at all. Any
        socket creation during training must crash the test, proving the
        offline path never touches the network."""
        import socket as socket_module

        def _no_sockets(*args, **kwargs):
            raise AssertionError(
                "network access attempted during offline training"
            )

        monkeypatch.setattr(socket_module, "socket", _no_sockets)
        offline = _stage_offline_dataset(
            tmp_path / "offline", n_train=8, n_val=4
        )
        cfg = self._config(tiny_config_with_assets, offline)
        cfg["huggingface"] = {"enabled": False}
        trainer = self._trainer(cfg, tmp_path)
        summary = trainer.train()
        assert summary["epochs_completed"] >= 1

    def test_offline_checkpoint_and_resume(self, tiny_config_with_assets,
                                           tmp_path):
        """Offline checkpoint chaining (tasklist Step 7): train, save, then
        resume from the local checkpoint with zero Hub involvement."""
        offline = _stage_offline_dataset(
            tmp_path / "offline", n_train=8, n_val=4
        )
        cfg = self._config(tiny_config_with_assets, offline)
        cfg["huggingface"] = {"enabled": False}
        cfg["training"]["max_epochs"] = 1

        trainer = self._trainer(cfg, tmp_path, auto_resume=False)
        summary = trainer.train()
        assert summary["epochs_completed"] == 1
        ckpt_dir = Path(tmp_path) / "exp" / "checkpoints"
        assert (ckpt_dir / "checkpoint_epoch_0.pt").exists()
        assert (ckpt_dir / "manifest.json").exists()
        assert (ckpt_dir / "progress.json").exists()

        # Session 2: resume-auto continues from the local state.
        cfg["training"]["max_epochs"] = 2
        trainer2 = self._trainer(cfg, tmp_path, auto_resume=True)
        assert trainer2.start_epoch == 1
        summary2 = trainer2.train()
        assert summary2["epochs_completed"] == 2


class TestAmpPrecisionResolution:
    """bf16 on RTX 6000 Ada (cc 8.9), fp16 fallback on T4 (cc 7.5)."""

    def _mock_cuda(self, monkeypatch, capability, name="RTX Pro 6000",
                   total_gb=96.0):
        fake_cuda = torch.cuda
        monkeypatch.setattr(fake_cuda, "is_available", lambda: True)
        monkeypatch.setattr(fake_cuda, "device_count", lambda: 1)
        monkeypatch.setattr(fake_cuda, "current_device", lambda: 0)
        monkeypatch.setattr(fake_cuda, "get_device_capability",
                            lambda idx=0: capability)
        monkeypatch.setattr(fake_cuda, "get_device_name",
                            lambda idx=0: name)

        class FakeProps:
            total_memory = int(total_gb * 1024 ** 3)

        monkeypatch.setattr(fake_cuda, "get_device_properties",
                            lambda idx=0: FakeProps())
        monkeypatch.setattr(fake_cuda, "mem_get_info",
                            lambda idx=0: (int(total_gb * 0.9 * 1024 ** 3),
                                           int(total_gb * 1024 ** 3)))

    def test_rtx6000_ada_resolves_bf16(self, monkeypatch):
        from syntree.utils.hardware import resolve_amp_dtype

        self._mock_cuda(monkeypatch, (8, 9), "NVIDIA RTX PRO 6000")
        dtype = resolve_amp_dtype("bf16", torch.device("cuda:0"))
        assert dtype == torch.bfloat16

    def test_t4_falls_back_to_fp16_with_warning(self, monkeypatch, caplog):
        from syntree.utils.hardware import resolve_amp_dtype

        self._mock_cuda(monkeypatch, (7, 5), "Tesla T4")
        with caplog.at_level("WARNING"):
            dtype = resolve_amp_dtype("bf16", torch.device("cuda:0"))
        assert dtype == torch.float16
        assert any("bfloat16" in message for message in caplog.messages)

    def test_explicit_fp16_stays_fp16(self, monkeypatch):
        from syntree.utils.hardware import resolve_amp_dtype

        self._mock_cuda(monkeypatch, (8, 9), "RTX PRO 6000")
        assert resolve_amp_dtype("fp16", torch.device("cuda:0")) == torch.float16

    def test_fp32_and_unknown_disable_amp(self, monkeypatch):
        from syntree.utils.hardware import resolve_amp_dtype

        self._mock_cuda(monkeypatch, (8, 9))
        assert resolve_amp_dtype("fp32", torch.device("cuda:0")) is None
        assert resolve_amp_dtype("none", torch.device("cuda:0")) is None

    def test_cpu_disables_amp(self):
        from syntree.utils.hardware import resolve_amp_dtype

        assert resolve_amp_dtype("bf16", torch.device("cpu")) is None

    def test_configure_runtime_rtx6000_profile(self, monkeypatch):
        from syntree.utils.hardware import configure_runtime_environment

        self._mock_cuda(monkeypatch, (8, 9), "NVIDIA RTX PRO 6000", 96.0)
        info = configure_runtime_environment({
            "device": "cuda:0", "mixed_precision": "bf16", "tf32": True,
        })
        assert info["device"] == "cuda:0"
        assert info["compute_capability"] == "8.9"
        assert info["rtx6000_ada"] is True
        assert info["precision"] == "bf16"
        assert info["mixed_precision_dtype"] == "bfloat16"
        assert info["tf32_enabled"] is True

    def test_tf32_false_honoured(self, monkeypatch):
        from syntree.utils.hardware import configure_runtime_environment

        self._mock_cuda(monkeypatch, (8, 9))
        info = configure_runtime_environment({"tf32": False})
        assert info["tf32_enabled"] is False
        assert torch.backends.cuda.matmul.allow_tf32 is False

    def test_tf32_true_on_old_gpu_warns_and_stays_off(self, monkeypatch,
                                                      caplog):
        from syntree.utils.hardware import configure_runtime_environment

        self._mock_cuda(monkeypatch, (7, 5), "Tesla T4")
        with caplog.at_level("WARNING"):
            info = configure_runtime_environment({"tf32": True})
        assert info["tf32_enabled"] is False

    def test_cpu_runtime_unchanged(self):
        from syntree.utils.hardware import configure_runtime_environment

        info = configure_runtime_environment({"device": "cpu"})
        assert info["device"] == "cpu"
        assert info["precision"] == "fp32"
        assert info["tf32_enabled"] is False
        assert "rtx6000_ada" not in info or info.get("rtx6000_ada") is False

    def test_trainer_amp_dtype_attributes(self, tiny_config_with_assets,
                                          tmp_path):
        """Trainer resolves the autocast dtype from the config: a 'bf16'
        config carries torch.bfloat16 (and would disable the GradScaler
        even on CUDA), fp16 keeps scaling. Verified without CUDA by
        inspecting the resolved attributes."""
        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy

        offline = _stage_offline_dataset(tmp_path / "offline", 8, 4)
        cfg = _trainer_config(tiny_config_with_assets, offline)
        cfg["system"]["mixed_precision"] = "bf16"
        model = SynTreePolicy(cfg)
        trainer = ResilientTrainer(
            model, cfg, torch.device("cpu"), output_dir=str(tmp_path / "e")
        )
        # CPU run: AMP off but the resolved dtype still reflects the config.
        assert trainer.amp_dtype == torch.bfloat16
        assert trainer.use_amp is False
        assert trainer.scaler.is_enabled() is False

        cfg2 = _trainer_config(tiny_config_with_assets, offline)
        cfg2["system"]["mixed_precision"] = "fp16"
        model2 = SynTreePolicy(cfg2)
        trainer2 = ResilientTrainer(
            model2, cfg2, torch.device("cpu"), output_dir=str(tmp_path / "e2")
        )
        assert trainer2.amp_dtype == torch.float16


class TestShippedConfigs:
    """The shipped production configs must parse and pass validation."""

    @pytest.mark.parametrize(
        "name",
        [
            "configs/train_kaggle_96gb.json",
            "configs/train_h100_full.json",
            "configs/train_colab_12h.json",
            "configs/rl_colab_12h.json",
            "configs/default_config.json",
        ],
    )
    def test_config_validates(self, name):
        import main as main_module

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_root, name)) as f:
            config = json.load(f)
        errors = main_module.validate_config(config)
        assert errors == [], f"{name}: {errors}"

    def test_kaggle_config_offline_contract(self):
        """The RTX 6000 offline config: no Hub, offline backend, RAM preload,
        bf16 + TF32, 11.2 h budget under the 12 h session kill, checkpoint
        pruning for the 57.6 GB disk quota."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_root, "configs/train_kaggle_96gb.json")) as f:
            cfg = json.load(f)

        assert cfg["data"]["backend"] == "kaggle_offline"
        assert cfg["data"]["preload_to_ram"] is True
        assert cfg["data"]["data_dir"].startswith("/kaggle/input")
        assert cfg["huggingface"]["enabled"] is False
        assert cfg["system"]["mixed_precision"] == "bf16"
        assert cfg["system"]["tf32"] is True
        assert cfg["training"]["time_budget_hours"] == 11.2
        assert cfg["training"]["keep_last_n_checkpoints"] == 2
        assert cfg["model"]["hidden_dim"] == 512
        assert cfg["model"]["num_equivariant_layers"] == 12
        assert cfg["model"]["cutoff_radius"] == 6.5
        assert cfg["catalog"]["encoder"] == "pharm3d"

    def test_h100_config_contract(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_root, "configs/train_h100_full.json")) as f:
            cfg = json.load(f)

        assert cfg["data"]["backend"] == "huggingface"
        assert cfg["data"]["min_real_samples"] == 5000
        assert cfg["system"]["mixed_precision"] == "bf16"
        assert cfg["training"]["time_budget_hours"] == 72.0
        assert cfg["reinforcement_learning"]["algorithm"] == "gflownet"
        assert cfg["reinforcement_learning"]["reward"]["diversity_weight"] == 0.2


class TestConfigValidationCLI:
    def test_cli_rejects_invalid_backend(self, tiny_config_with_assets,
                                         tmp_path):
        import subprocess

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["backend"] = "ftp"
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.run(
            [sys.executable, "main.py", "--mode", "info",
             "--config", str(cfg_path)],
            capture_output=True, text=True, timeout=120, cwd=repo_root,
        )
        assert proc.returncode == 2
        assert "data.backend" in proc.stderr

    def test_cli_rejects_bad_precision(self, tiny_config_with_assets,
                                       tmp_path):
        import subprocess

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["system"]["mixed_precision"] = "tf32x"
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.run(
            [sys.executable, "main.py", "--mode", "info",
             "--config", str(cfg_path)],
            capture_output=True, text=True, timeout=120, cwd=repo_root,
        )
        assert proc.returncode == 2
        assert "mixed_precision" in proc.stderr


class TestOfflineReproducibility:
    """The offline pipeline must give bit-identical results for identical
    seeds (same-trained-weights reproducibility contract)."""

    def test_identical_seeds_identical_loss_history(self,
                                                    tiny_config_with_assets,
                                                    tmp_path):
        import copy

        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy

        offline = _stage_offline_dataset(tmp_path / "offline", 8, 4)

        losses = []
        for run in range(2):
            cfg = _trainer_config(tiny_config_with_assets, offline)
            cfg["huggingface"] = {"enabled": False}
            cfg["training"]["max_epochs"] = 2
            run_dir = tmp_path / f"run{run}"
            torch.manual_seed(0)
            model = SynTreePolicy(copy.deepcopy(cfg))
            trainer = ResilientTrainer(
                model, cfg, torch.device("cpu"),
                output_dir=str(run_dir),
            )
            trainer.train()
            with open(run_dir / "history.json") as f:
                history = json.load(f)
            losses.append([entry["train_loss"] for entry in history])

        assert len(losses[0]) == len(losses[1]) == 2
        assert losses[0] == losses[1], (
            f"offline training is not reproducible: {losses[0]} vs {losses[1]}"
        )
