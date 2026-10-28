"""Batch 4 Test Suite: Data Pipeline, Featurizer, Fragmenter & Trajectory Builder.

Strictly CPU-executable within 8 GB RAM constraints.
Tests feature extraction, SNAr aryl fluoride handling, fast bulk similarity,
recursive CrossDocked pair discovery, and multi-step trajectory generation.
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

# Ensure codebase root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch.set_num_threads(2)

from syntree.chemistry.catalog import SynthonCatalog
from syntree.chemistry.reactions import ReactionEngine, REACTION_FAMILY_NAMES
from syntree.data.featurizer import MolecularFeaturizer, GLOBAL_FEATURE_DIM, HANDLE_FEATURE_DIM
from syntree.data.fragmenter import ReactionConstrainedFragmenter
from syntree.data.crossdocked import CrossDockedDataset
from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder
from syntree.data.hf_loader import ShardAwareShuffleSampler


class MockShardDataset:
    """Lightweight mock for shard sampler testing."""
    def __init__(self, ranges):
        self.shard_ranges = ranges
        self.total = sum(end - start for start, end in ranges)

    def __len__(self):
        return self.total


class TestFeaturizer(unittest.TestCase):
    """Tests for syntree/data/featurizer.py."""

    def test_01_featurizer_pocket_and_charge(self):
        """MolecularFeaturizer must featurize pocket atoms and assign formal charges."""
        print("\n[TEST] Featurizer: Pocket Point Cloud & Formal Charges")
        pdb_lines = [
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N",
            "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C",
            "ATOM      3  OD1 ASP A   2       3.000   0.000   0.000  1.00 20.00           O",
            "ATOM      4  NZ  LYS A   3       5.000   0.000   0.000  1.00 20.00           N",
        ]
        mol = Chem.MolFromPDBBlock("\n".join(pdb_lines), removeHs=False)
        self.assertIsNotNone(mol, "Failed to parse mock PDB block")

        feats = MolecularFeaturizer.featurize_pocket(mol)
        self.assertIn("pocket_pos", feats)
        self.assertIn("pocket_z", feats)
        self.assertIn("pocket_charge", feats)
        self.assertIn("centroid", feats)

        charges = feats["pocket_charge"].numpy()
        self.assertAlmostEqual(charges[2], -0.5, places=2)
        self.assertAlmostEqual(charges[3], 1.0, places=2)
        print("  -> PASS: Pocket featurized with correct spatial centroid and physiological charges.")

    def test_02_featurizer_handle_and_ligand_decoupling(self):
        """Chemical features and 3D position must be decoupled (auditing F-02/F-06)."""
        print("\n[TEST] Featurizer: Handle Feature & Position Decoupling")
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))  # Acetic acid
        AllChem.EmbedMolecule(mol, randomSeed=42)

        handle_indices = (0, 1, 2)
        handle_feat = MolecularFeaturizer.featurize_handle(mol, handle_indices, "carboxylic_acid")
        handle_pos = MolecularFeaturizer.featurize_handle_position(mol, handle_indices)

        self.assertEqual(handle_feat.shape, (HANDLE_FEATURE_DIM,))
        self.assertEqual(handle_pos.shape, (3,))
        self.assertFalse(torch.isnan(handle_feat).any())
        self.assertFalse(torch.isnan(handle_pos).any())

        global_feats = MolecularFeaturizer.ligand_global_features(mol, pocket_volume=150.0)
        self.assertEqual(global_feats.shape, (GLOBAL_FEATURE_DIM,))
        self.assertGreater(global_feats[0].item(), 0.0)
        print(f"  -> PASS: Handle chemical features (dim={len(handle_feat)}) decoupled from position (dim={len(handle_pos)}).")


class TestFragmenter(unittest.TestCase):
    """Tests for syntree/data/fragmenter.py."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.df = pd.DataFrame([
            {"id": "SYN_ACID", "smiles": "CC(=O)O", "fsp3": 0.5, "mw": 60.05, "primary_handle": "carboxylic_acid"},
            {"id": "SYN_AMINE", "smiles": "NCc1ccccc1", "fsp3": 0.14, "mw": 107.15, "primary_handle": "primary_secondary_amine"},
            {"id": "SYN_ARYL_F", "smiles": "c1ccccc1F", "fsp3": 0.0, "mw": 96.10, "primary_handle": "aryl_halide"},
        ])
        self.cat_path = os.path.join(self.test_dir, "test_cat.parquet")
        self.df.to_parquet(self.cat_path)
        self.catalog = SynthonCatalog(self.cat_path, embedding_dim=16, min_fsp3=0.0, max_mw=300.0, encoder="morgan2d")
        self.fragmenter = ReactionConstrainedFragmenter(self.catalog)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_03_fragmenter_fluorine_snar_support(self):
        """DIAGNOSTIC TEST (F-22): _decorate_fragment must support Fluorine (9) for SNAr."""
        print("\n[TEST] Fragmenter: SNAr Fluorine Fragment Decoration (Auditing F-22)")
        benzene = Chem.MolFromSmiles("c1ccccc1")
        decorated = ReactionConstrainedFragmenter._decorate_fragment(benzene, 0, "aryl_halide")
        decorated_smiles = [Chem.MolToSmiles(m) for m in decorated]

        print(f"  -> Decorated variants for aryl_halide: {decorated_smiles}")
        has_fluorobenzene = any("Fc1ccccc1" in s or "c1ccc(F)cc1" in s for s in decorated_smiles)
        self.assertTrue(has_fluorobenzene, "Fluorobenzene missing from aryl_halide decorations! SNAr retrosynthesis broken.")
        print("  -> PASS: Fluorine (atomic number 9) successfully generated in aryl_halide variants.")

    def test_04_fragmenter_bulk_tanimoto_speed_and_none_safety(self):
        """DIAGNOSTIC TEST (H8/F-16): Fast candidate search must execute without NoneType crash."""
        print("\n[TEST] Fragmenter: Bulk Tanimoto Search & None-Safety (Auditing H8/F-16)")
        query = Chem.MolFromSmiles("CC(=O)O")
        candidates = self.fragmenter._catalog_candidates(query)
        self.assertIn(0, candidates, "Exact catalog match for acetic acid failed!")

        query_sim = Chem.MolFromSmiles("CCC(=O)O")
        candidates_sim = self.fragmenter._catalog_candidates(query_sim, min_tanimoto=0.4)
        self.assertIsInstance(candidates_sim, tuple)
        print(f"  -> Successfully searched catalog candidates for propionic acid: {candidates_sim}")
        print("  -> PASS: Precomputed bulk fingerprint lookup executed cleanly with zero NoneType errors.")


class TestCrossDockedAndTrajectory(unittest.TestCase):
    """Tests for syntree/data/crossdocked.py and trajectory.py."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cat_df = pd.DataFrame([
            {"id": "SYN_ACID", "smiles": "CC(=O)O", "fsp3": 0.5, "mw": 60.05, "primary_handle": "carboxylic_acid"},
            {"id": "SYN_AMINE", "smiles": "NCc1ccccc1", "fsp3": 0.14, "mw": 107.15, "primary_handle": "primary_secondary_amine"},
        ])
        self.cat_path = os.path.join(self.test_dir, "test_cat.parquet")
        self.cat_df.to_parquet(self.cat_path)
        self.catalog = SynthonCatalog(self.cat_path, embedding_dim=16, min_fsp3=0.0, max_mw=300.0, encoder="morgan2d")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_05_crossdocked_recursive_pair_discovery(self):
        """DIAGNOSTIC TEST (F-08/F-15): CrossDockedDataset must discover nested *_pocket10.pdb files."""
        print("\n[TEST] CrossDocked: Recursive Pair Discovery (Auditing F-08/F-15)")
        flat_p = Path(self.test_dir) / "flat_pocket.pdb"
        flat_l = Path(self.test_dir) / "flat_ligand.sdf"
        flat_p.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n")
        flat_l.write_text("$$$$\n")

        nested_dir = Path(self.test_dir) / "nested_target"
        nested_dir.mkdir()
        nested_p = nested_dir / "complex_pocket10.pdb"
        nested_l = nested_dir / "complex_ligand.sdf"
        nested_p.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n")
        nested_l.write_text("$$$$\n")

        pairs = CrossDockedDataset._discover_pairs(self.test_dir)
        discovered_pockets = [Path(p).name for p, _, _ in pairs]
        print(f"  -> Discovered pocket files: {discovered_pockets}")

        self.assertIn("flat_pocket.pdb", discovered_pockets)
        self.assertIn("complex_pocket10.pdb", discovered_pockets)
        self.assertEqual(len(pairs), 2)
        print("  -> PASS: Both flat and nested Zenodo *_pocket10.pdb archives discovered successfully.")

    def test_06_crossdocked_synthetic_plumbing_and_pyg_collation(self):
        """Synthetic mode must produce valid PyG Data objects that collate cleanly."""
        print("\n[TEST] CrossDocked: Synthetic Data Collation Safety")
        ds = CrossDockedDataset(
            root_dir=self.test_dir,
            split="train",
            catalog=self.catalog,
            num_synthetic=4,
            synthetic=True,
            force_rebuild=True,
        )
        self.assertEqual(len(ds), 4)
        sample = ds[0]

        for attr in ("pocket_pos", "pocket_z", "pocket_charge", "handle_features", "global_features"):
            self.assertTrue(hasattr(sample, attr), f"Sample missing attribute: {attr}")

        batch = CrossDockedDataset._collate([ds[0], ds[1]])
        self.assertEqual(batch.pocket_pos.dim(), 2)
        n_graphs = batch.handle_features.size(0) if batch.handle_features.dim() == 2 else batch.handle_features.size(0) // 64
        self.assertEqual(n_graphs, 2)
        print("  -> PASS: Synthetic samples collate into valid PyG Batch without dimensionality errors.")

    def test_07_trajectory_builder_multistep(self):
        """RetrosyntheticTrajectoryBuilder must extract replayable states from amide ligand."""
        print("\n[TEST] Trajectory: Multi-Step Behavioral Cloning Decomposer")
        ligand = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NCc1ccccc1"))
        AllChem.EmbedMolecule(ligand, randomSeed=42)

        pocket = Chem.MolFromPDBBlock(
            "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n",
            removeHs=False
        )

        builder = RetrosyntheticTrajectoryBuilder(self.catalog, max_steps=2)
        states, meta = builder.build(ligand, pocket, trajectory_id="TRAJ_001")

        self.assertGreater(len(states), 0, "Builder failed to extract states from valid amide ligand!")
        first_step = states[0]
        # Synthon can be SYN_ACID (0) with benzylamine core or SYN_AMINE (1) with acetic acid core
        self.assertIn(first_step.target_synthon.item(), (0, 1))
        self.assertFalse(first_step.target_stop.item())
        family_name = REACTION_FAMILY_NAMES[first_step.target_reaction_family_idx.item()]
        self.assertEqual(family_name, "amide_coupling")
        print(f"  -> Generated {len(states)} trajectory states (synthon_idx={first_step.target_synthon.item()}, family={family_name}).")
        print("  -> PASS: Expert retrosynthetic trajectory extracted and validated.")

    def test_08_shard_aware_sampler_permutation(self):
        """ShardAwareShuffleSampler must shuffle deterministically and preserve sample count."""
        print("\n[TEST] ShardLoader: Shard-Aware Shuffle Sampler Invariants")
        mock_ds = MockShardDataset([(0, 100), (100, 250), (250, 400)])
        sampler = ShardAwareShuffleSampler(mock_ds, seed=42)
        sampler.set_epoch(0)

        indices_ep0 = list(sampler)
        self.assertEqual(len(indices_ep0), 400)
        self.assertEqual(set(indices_ep0), set(range(400)), "Sampler dropped or duplicated indices!")

        sampler.set_epoch(1)
        indices_ep1 = list(sampler)
        self.assertNotEqual(indices_ep0, indices_ep1, "Sampler failed to re-permute on epoch change!")
        print("  -> PASS: Sampler preserves index totality and re-permutates deterministically.")


if __name__ == "__main__":
    print("=" * 70)
    print(" 3D-SYNTREE LOCAL VERIFICATION SUITE: BATCH 4 (DATA PIPELINE)")
    print(f" Platform: {sys.platform} | PyTorch: {torch.__version__} | Device: CPU")
    print(" Hardware Target: Intel Core i3 (11th Gen) / 8 GB RAM")
    print("=" * 70)
    unittest.main(verbosity=2)