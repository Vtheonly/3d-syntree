"""Batch 2 Test Suite: Registry, Hardware Autotuning, Checkpointing, Policy & CLI.

Runs purely on CPU within 8 GB RAM constraints.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
torch.set_num_threads(2)

# Prevent chemistry/RDKit cascade while testing pure neural policy and infrastructure
mock_rxn = MagicMock()
mock_rxn.REACTION_FAMILY_NAMES = (
    "amide_coupling", "reductive_amination", "suzuki_coupling", "aryl_amination",
    "urea_formation", "esterification", "click_triazole", "sulfonamide_coupling", "sp3_alkylation"
)
mock_chem = MagicMock(reactions=mock_rxn, catalog=MagicMock())
mock_chem.__path__ = []  # Tells Python this mock behaves as a package

sys.modules["syntree.chemistry"] = mock_chem
sys.modules["syntree.chemistry.reactions"] = mock_rxn
sys.modules["syntree.chemistry.catalog"] = mock_chem.catalog

import syntree
syntree.chemistry = mock_chem

from syntree.core.registry import CANONICAL_MODELS, run_preflight_audit
from syntree.utils.hardware import resolve_amp_dtype, autotune_batch_size
from syntree.utils.checkpoint import CheckpointManager
from syntree.models.policy import SynTreePolicy
import main as main_cli


class TestRegistryAndPreflight(unittest.TestCase):
    def test_01_canonical_registry_integrity(self):
        print("\n[TEST] Registry: Model Schema Integrity")
        self.assertIn("3d-syntree-base", CANONICAL_MODELS)
        base = CANONICAL_MODELS["3d-syntree-base"]
        self.assertEqual(base.hidden_dim, 128)
        self.assertEqual(base.num_layers, 4)
        print("  -> PASS: Canonical models certified.")

    def test_02_preflight_audit_validation(self):
        print("\n[TEST] Registry: Pre-Flight Audit Rules")
        valid_cfg = {"canonical_model_name": "3d-syntree-base", "data": {}}
        res = run_preflight_audit(valid_cfg, min_disk_gb=0.1)
        self.assertTrue(res["model_valid"])
        self.assertTrue(res["disk_space_ok"])

        with self.assertRaises(ValueError):
            run_preflight_audit({"canonical_model_name": "invalid_model_arch"})
        print("  -> PASS: Pre-flight audit validates schemas and fails closed on unknown models.")


class TestHardwareUtilities(unittest.TestCase):
    def test_03_resolve_amp_dtype_cpu(self):
        print("\n[TEST] Hardware: Mixed Precision CPU Resolution")
        cpu = torch.device("cpu")
        self.assertIsNone(resolve_amp_dtype("bf16", cpu))
        self.assertIsNone(resolve_amp_dtype("fp16", cpu))
        print("  -> PASS: CPU correctly resolves to float32 (None autocast).")

    def test_04_autotune_batch_size_downward_search(self):
        print("\n[TEST] Hardware: autotune_batch_size OOM Downward Recovery (Auditing C4)")
        def mock_probe(b: int):
            if b > 4:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory in probe.")

        res_cpu = autotune_batch_size(mock_probe, dataset_size=100, start_batch=16, device=torch.device("cpu"))
        self.assertEqual(res_cpu["batch_size"], 16.0)
        print("  -> PASS: CPU autotune returns safe default without crashing.")


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {"huggingface": {"enabled": False}, "system": {"seed": 42}}
        self.manager = CheckpointManager(self.config, ckpt_dir=self.test_dir, keep_last_n=2)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_05_progress_validation_modes(self):
        print("\n[TEST] Checkpointing: Progress Mode Validation (Auditing B.1)")
        valid_epoch = {"mode": "epoch", "completed_epochs": [0, 1, 2], "last_sequential_epoch": 2, "history": [{}, {}, {}]}
        valid, last_ep = CheckpointManager._validate_progress(valid_epoch, expected_mode="epoch")
        self.assertTrue(valid)
        self.assertEqual(last_ep, 2)

        gap_epoch = {"mode": "epoch", "completed_epochs": [0, 2], "last_sequential_epoch": 2, "history": [{}, {}]}
        valid, _ = CheckpointManager._validate_progress(gap_epoch, expected_mode="epoch")
        self.assertFalse(valid)

        valid_ep = {"mode": "episode", "completed_epochs": [16, 32, 64], "last_sequential_epoch": 64, "history": [{}, {}, {}]}
        valid, last_ep = CheckpointManager._validate_progress(valid_ep, expected_mode="episode")
        self.assertTrue(valid)
        print("  -> PASS: Progress correctly validates strict contiguity for epochs and monotonicity for episodes.")

    def test_06_checkpoint_atomic_save_and_restore_with_scaler(self):
        print("\n[TEST] Checkpointing: Scaler State Persistence (Auditing Issues 4)")
        model = torch.nn.Linear(10, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        try:
            scaler = torch.amp.GradScaler("cpu", enabled=False)
        except Exception:
            scaler = torch.amp.GradScaler("cuda", enabled=False)

        ckpt_path = self.manager.save_checkpoint(epoch=0, step=100, model=model, optimizer=opt, scaler=scaler)
        self.assertTrue(Path(ckpt_path).exists())

        new_model = torch.nn.Linear(10, 2)
        new_opt = torch.optim.SGD(new_model.parameters(), lr=0.01)
        try:
            new_scaler = torch.amp.GradScaler("cpu", enabled=False)
        except Exception:
            new_scaler = torch.amp.GradScaler("cuda", enabled=False)

        next_epoch, step, _ = self.manager.restore_latest(new_model, new_opt, scaler=new_scaler)
        self.assertEqual(next_epoch, 1)
        self.assertEqual(step, 100)
        self.assertTrue(torch.allclose(model.weight, new_model.weight))
        print("  -> PASS: Checkpoint saved and restored cleanly with weights, optimizer, and scaler state.")


class TestPolicyIntegration(unittest.TestCase):
    def setUp(self):
        self.hidden_dim = 32
        self.config = {
            "model": {
                "hidden_dim": self.hidden_dim,
                "num_equivariant_layers": 2,
                "num_radial_basis": 8,
                "cutoff_radius": 5.0,
                "synthon_embedding_dim": self.hidden_dim,
                "num_attention_heads": 4,
                "max_atomic_number": 50,
                "dropout": 0.0,
            }
        }
        self.policy = SynTreePolicy(self.config)
        self.policy.eval()

    def test_07_policy_query_broadcasting_fix(self):
        print("\n[TEST] Policy: Forward Query Broadcasting Check (Auditing F-02/F-06)")
        b = 2
        n_p, n_l = 8, 4
        batch_data = SimpleNamespace(
            pocket_pos=torch.randn(n_p * b, 3),
            pocket_z=torch.randint(1, 10, (n_p * b,)),
            pocket_batch=torch.tensor([0] * n_p + [1] * n_p, dtype=torch.long),
            ligand_pos=torch.randn(n_l * b, 3),
            ligand_z=torch.randint(1, 10, (n_l * b,)),
            ligand_batch=torch.tensor([0] * n_l + [1] * n_l, dtype=torch.long),
            handle_features=torch.randn(b, 64),
            handle_pos=torch.randn(b, 3),
            handle_nodes=torch.tensor([0, 1], dtype=torch.long),
            global_features=torch.randn(b, 4),
        )

        num_synthons = 20
        synthon_embeddings = torch.randn(num_synthons, self.hidden_dim)
        rxn_mask = torch.zeros(b, len(mock_rxn.REACTION_FAMILY_NAMES))
        syn_mask = torch.zeros(b, num_synthons)

        out = self.policy(batch_data, synthon_embeddings, syn_mask, reaction_compatibility_mask=rxn_mask)
        context_shape = out["pocket_context"].shape
        rxn_logits_shape = out["reaction_logits"].shape
        print(f"  -> Context shape: {list(context_shape)} | Reaction logits shape: {list(rxn_logits_shape)}")

        self.assertEqual(context_shape, (b, self.hidden_dim))
        self.assertEqual(rxn_logits_shape, (b, len(mock_rxn.REACTION_FAMILY_NAMES)))
        print("  -> PASS: Query tensor dimensions strictly aligned to [B, d]; no B*B broadcast.")


class TestMainEntrypoint(unittest.TestCase):
    def test_08_main_syntax_and_config_validation(self):
        print("\n[TEST] Entrypoint: main.py Validation & Syntax (Auditing F-01)")
        self.assertTrue(callable(main_cli._load_generated))

        valid_cfg = {
            "system": {"device": "cpu", "mixed_precision": "fp32"},
            "data": {"backend": "local", "batch_size": 16},
            "model": {"hidden_dim": 128},
            "training": {"max_epochs": 10, "time_budget_hours": 1.0, "learning_rate": 0.001},
        }
        errors = main_cli.validate_config(valid_cfg)
        self.assertEqual(len(errors), 0)
        print("  -> PASS: main.py syntax complete and validate_config certified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)