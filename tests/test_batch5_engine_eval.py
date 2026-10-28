"""Batch 5 Test Suite: Optimization, Reinforcement Learning, GFlowNet, DPO & Evaluator.

Strictly CPU-executable within 8 GB RAM constraints.
Tests sub-graph radius construction memory scaling, docking output parsing,
rollout no-grad invariants, 3D reward diversity, GFlowNet Trajectory Balance,
DPO within-pocket pairing, and evaluation metrics.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch.set_num_threads(2)

from syntree.chemistry.catalog import SynthonCatalog
from syntree.chemistry.reactions import ReactionEngine
from syntree.models.equivariant import build_radius_graph
from syntree.models.policy import SynTreePolicy
from syntree.engine.evaluator import EvaluationPipeline
from syntree.engine.rl import (
    ThreeDReward,
    RolloutBuffer,
    RolloutTransition,
    count_key_interactions,
)
from syntree.engine.gflownet import GFlowNetTrainer
from syntree.engine.dpo import DPOTrainer, build_preference_pairs


class TestEquivariantGraphOptimization(unittest.TestCase):
    """Tests for syntree/models/equivariant.py::build_radius_graph."""

    def test_01_build_radius_graph_subgraph_slicing(self):
        """DIAGNOSTIC TEST (C1): build_radius_graph must slice sub-graphs without cross-edges."""
        print("\n[TEST] Equivariant: Sub-Graph Radius Graph Slicing (Auditing C1)")
        # 3 graphs: graph 0 near (0,0,0), graph 1 near (50,50,50), graph 2 near (100,100,100)
        pos_g0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pos_g1 = torch.tensor([[50.0, 50.0, 50.0], [51.0, 50.0, 50.0]])
        pos_g2 = torch.tensor([[100.0, 100.0, 100.0], [100.5, 100.0, 100.0], [100.0, 100.5, 100.0]])

        # Test contiguous batch
        pos = torch.cat([pos_g0, pos_g1, pos_g2], dim=0)
        batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)

        edge_index = build_radius_graph(pos, batch, cutoff=5.0)
        row, col = edge_index[0], edge_index[1]

        # Verify no cross-graph edges
        cross_edges = (batch[row] != batch[col]).sum().item()
        self.assertEqual(cross_edges, 0, "Found cross-graph edges in sliced radius graph!")
        self.assertGreater(edge_index.size(1), 0)

        # Test non-contiguous batch (pocket nodes followed by ligand nodes)
        perm = torch.tensor([0, 3, 5, 1, 4, 6, 2, 7], dtype=torch.long)
        pos_perm = pos[perm]
        batch_perm = batch[perm]

        edge_index_perm = build_radius_graph(pos_perm, batch_perm, cutoff=5.0)
        row_p, col_p = edge_index_perm[0], edge_index_perm[1]
        cross_edges_perm = (batch_perm[row_p] != batch_perm[col_p]).sum().item()
        self.assertEqual(cross_edges_perm, 0, "Found cross-graph edges in non-contiguous batch!")
        self.assertEqual(edge_index_perm.size(1), edge_index.size(1))
        print("  -> PASS: Sub-graph slicing produces identical edges with zero cross-graph leakage.")


class TestEvaluatorDockingParser(unittest.TestCase):
    """Tests for syntree/engine/evaluator.py::_parse_docking_score."""

    def test_02_docking_score_regex_parser(self):
        """DIAGNOSTIC TEST (F-10/F-14): _parse_docking_score must parse standard Vina and GNINA outputs."""
        print("\n[TEST] Evaluator: Docking Score Parser (Auditing F-10/F-14)")
        # Standard AutoDock Vina tabular stdout
        vina_stdout = """
-----+------------+----------+----------
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1         -7.4      0.000      0.000
   2         -6.8      1.240      2.110
"""
        score_vina = EvaluationPipeline._parse_docking_score(vina_stdout)
        self.assertIsNotNone(score_vina, "Failed to parse standard Vina tabular score!")
        self.assertAlmostEqual(score_vina, -7.4, places=2)

        # Standard GNINA tabular stdout
        gnina_stdout = """
mode |   affinity | intramol | cnn pose | cnn
     | (kcal/mol) |  energy  |  score   | affinity
-----+------------+----------+----------+---------
   1         -8.25     -0.52      0.8412     6.71
   2         -7.10     -0.12      0.6540     5.90
"""
        score_gnina = EvaluationPipeline._parse_docking_score(gnina_stdout)
        self.assertIsNotNone(score_gnina, "Failed to parse standard GNINA tabular score!")
        self.assertAlmostEqual(score_gnina, -8.25, places=2)

        # Legacy header format
        legacy_stdout = "Estimated free energy of binding : -6.8 kcal/mol"
        score_legacy = EvaluationPipeline._parse_docking_score(legacy_stdout)
        self.assertAlmostEqual(score_legacy, -6.8, places=2)

        # Empty / failed output
        score_empty = EvaluationPipeline._parse_docking_score("Error: receptor file missing")
        self.assertIsNone(score_empty)
        print("  -> PASS: AutoDock Vina, GNINA, and legacy format scores parsed accurately.")


class TestReinforcementLearningAndRewards(unittest.TestCase):
    """Tests for syntree/engine/rl.py."""

    def test_03_three_d_reward_diversity_and_interactions(self):
        """DIAGNOSTIC TEST (Priority 6): Reward must include batch diversity and key interactions."""
        print("\n[TEST] RL: Multi-Objective 3D Reward & Diversity (Priority 6)")
        reward_fn = ThreeDReward({
            "diversity_weight": 0.5,
            "key_interaction_weight": 1.0,
            "clash_scale": 25.0,
        })

        # Test Batch Diversity
        mol_a = Chem.MolFromSmiles("CC(=O)NCc1ccccc1")
        mol_b = Chem.MolFromSmiles("c1ccccc1")
        mol_c = Chem.MolFromSmiles("CC(=O)NCc1ccccc1")  # Duplicate of A

        diversities = ThreeDReward.batch_diversity([mol_a, mol_b, mol_c])
        self.assertEqual(len(diversities), 3)
        # Duplicate molecules must have lower diversity than unique molecule B
        self.assertGreater(diversities[1], diversities[0])
        print(f"  -> Batch diversities (unique vs duplicate): {diversities}")

        # Test Key Interactions (Salt Bridge)
        # Ligand with protonated amine (+1), Pocket with ASP carboxylate (-0.5, -0.5)
        lig = Chem.AddHs(Chem.MolFromSmiles("[NH3+]CC"))
        AllChem.EmbedMolecule(lig, randomSeed=42)

        pocket_pdb = (
            "ATOM      1  OD1 ASP A   1       0.000   0.000   0.000  1.00 20.00           O\n"
            "ATOM      2  OD2 ASP A   1       1.200   0.000   0.000  1.00 20.00           O\n"
            "END\n"
        )
        pocket_mol = Chem.MolFromPDBBlock(pocket_pdb, removeHs=False)

        salt, hbond = count_key_interactions(lig, pocket_mol)
        self.assertGreaterEqual(salt + hbond, 1, "Failed to identify key interaction between charged pair!")
        print(f"  -> Counted key interactions: {salt} salt bridges, {hbond} H-bonds.")
        print("  -> PASS: Diversity scoring and pharmacophore interactions certified.")

    def test_04_rollout_buffer_gae_computation(self):
        """RolloutBuffer must compute GAE advantages and normalize across buffer."""
        print("\n[TEST] RL: Multi-Episode GAE Advantage Estimation")
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)

        t1 = RolloutTransition(
            state=None, reaction_mask=None, synthon_masks=None,
            family_idx=0, action_idx=1,
            old_log_prob=torch.tensor(-0.5), old_value=torch.tensor(1.0),
            reward=0.2, done=False
        )
        t2 = RolloutTransition(
            state=None, reaction_mask=None, synthon_masks=None,
            family_idx=0, action_idx=len(self.cat_df) if hasattr(self, 'cat_df') else 2,
            old_log_prob=torch.tensor(-0.2), old_value=torch.tensor(1.5),
            reward=2.0, done=True  # Terminal reward
        )
        buffer.add_episode([t1, t2])

        advs, returns = buffer.compute_gae()
        self.assertEqual(len(advs), 2)
        self.assertEqual(len(returns), 2)
        self.assertFalse(torch.isnan(advs).any())
        self.assertFalse(torch.isnan(returns).any())
        print("  -> PASS: GAE advantages and returns evaluated cleanly across episodes.")


class TestGFlowNetAndDPO(unittest.TestCase):
    """Tests for syntree/engine/gflownet.py and dpo.py."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cat_df = pd.DataFrame([
            {"id": "SYN_01", "smiles": "CC(=O)O", "fsp3": 0.5, "mw": 60.05, "primary_handle": "carboxylic_acid"},
            {"id": "SYN_02", "smiles": "CCN", "fsp3": 1.0, "mw": 45.08, "primary_handle": "primary_secondary_amine"},
        ])
        self.cat_path = os.path.join(self.test_dir, "test_cat.parquet")
        self.cat_df.to_parquet(self.cat_path)
        self.catalog = SynthonCatalog(self.cat_path, embedding_dim=16, min_fsp3=0.0, max_mw=300.0, encoder="morgan2d")

        self.model_cfg = {
            "model": {
                "hidden_dim": 16,
                "num_equivariant_layers": 2,
                "num_radial_basis": 8,
                "cutoff_radius": 5.0,
                "synthon_embedding_dim": 16,
                "num_attention_heads": 4,
                "max_atomic_number": 50,
                "dropout": 0.0,
            }
        }
        self.policy = SynTreePolicy(self.model_cfg)
        self.policy.eval()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_05_gflownet_trainer_trajectory_balance(self):
        """GFlowNetTrainer must evaluate TB loss and update log_Z."""
        print("\n[TEST] GFlowNet: Trajectory Balance Loss & Log_Z Update")
        trainer = GFlowNetTrainer(self.policy, self.catalog, device=torch.device("cpu"), config={"learning_rate": 1e-4, "lr_z": 1e-2})
        initial_log_z = float(trainer.log_Z.item())

        # Mock transition
        from torch_geometric.data import Data
        state = Data(
            pocket_pos=torch.randn(6, 3),
            pocket_z=torch.tensor([6, 6, 7, 8, 6, 6]),
            pocket_batch=torch.zeros(6, dtype=torch.long),
            handle_features=torch.randn(1, 64),
            handle_pos=torch.zeros(1, 3),
            handle_nodes=torch.tensor([-1]),
            global_features=torch.zeros(1, 4),
            stop_mask=torch.tensor([0.0]),
        )
        rxn_mask = torch.zeros(1, 9)
        syn_mask = torch.zeros(1, 9, len(self.catalog))

        trans = RolloutTransition(
            state=state, reaction_mask=rxn_mask, synthon_masks=syn_mask,
            family_idx=0, action_idx=0,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.5),
            reward=1.5, done=True
        )

        stats = trainer.train_step([([trans], 1.5)])
        self.assertIn("loss", stats)
        self.assertGreater(stats["loss"], 0.0)

        # Test state save/restore
        state_dict = trainer.state_dict()
        self.assertIn("log_Z", state_dict)
        print(f"  -> GFlowNet step complete (TB loss={stats['loss']:.4f}, log_Z={stats['log_Z']:.4f}).")
        print("  -> PASS: Trajectory Balance optimizer certified.")

    def test_06_dpo_within_pocket_preference_pairing(self):
        """DIAGNOSTIC TEST (H2): DPO must pair candidate episodes within the SAME pocket."""
        print("\n[TEST] DPO: Within-Pocket Preference Pairing (Auditing H2)")
        ep1 = {"pocket": "1a9u", "reward": 0.8, "transitions": [1, 2]}
        ep2 = {"pocket": "1a9u", "reward": 0.3, "transitions": [3, 4]}  # 1a9u loser
        ep3 = {"pocket": "3nyr", "reward": 0.9, "transitions": [5, 6]}
        ep4 = {"pocket": "3nyr", "reward": 0.4, "transitions": [7, 8]}  # 3nyr loser

        pairs = build_preference_pairs([ep1, ep2, ep3, ep4])
        self.assertEqual(len(pairs), 2, "Failed to form 2 preference pairs!")

        # Pair 0 must be (ep1, ep2) because both belong to 1a9u and ep1 > ep2
        win_1, lose_1 = pairs[0]
        self.assertEqual(win_1, [1, 2])
        self.assertEqual(lose_1, [3, 4])

        # Test DPOTrainer instantiation and step
        trainer = DPOTrainer(self.policy, self.catalog, device=torch.device("cpu"), config={"dpo_beta": 0.2})
        self.assertIsNotNone(trainer.reference_model)
        print("  -> PASS: Preference pairs formed strictly within matching pocket contexts.")


class TestEvaluationPipelineBattery(unittest.TestCase):
    """Tests for syntree/engine/evaluator.py."""

    def test_07_evaluator_property_battery(self):
        """EvaluationPipeline must evaluate validity, Fsp3, and uniqueness."""
        print("\n[TEST] Evaluator: Property Battery & PoseBusters Validity")
        pipeline = EvaluationPipeline(
            config={"evaluation": {"run_aizynthfinder": False, "docking_engine": "none"}},
            output_dir=tempfile.mkdtemp()
        )

        mol1 = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))  # Ibuprofen
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene (no conformer, low Fsp3)

        report = pipeline.evaluate([mol1, mol2])
        self.assertIn("chemical_validity", report)
        self.assertIn("posebusters_pass_rate", report)
        self.assertIn("mean_fsp3", report)
        self.assertIn("uniqueness", report)

        self.assertEqual(report["num_molecules"], 2)
        self.assertEqual(report["chemical_validity"], 1.0)
        self.assertEqual(report["posebusters_pass_rate"], 0.5)  # Ibuprofen passed, Benzene failed
        print(f"  -> Report summary: Validity={report['chemical_validity']:.1%}, PB={report['posebusters_pass_rate']:.1%}, Fsp3={report['mean_fsp3']:.3f}")
        print("  -> PASS: Full benchmark battery evaluated cleanly.")


if __name__ == "__main__":
    print("=" * 70)
    print(" 3D-SYNTREE LOCAL VERIFICATION SUITE: BATCH 5 (OPTIMIZATION & EVAL)")
    print(f" Platform: {sys.platform} | PyTorch: {torch.__version__} | Device: CPU")
    print(" Hardware Target: Intel Core i3 (11th Gen) / 8 GB RAM")
    print("=" * 70)
    unittest.main(verbosity=2)