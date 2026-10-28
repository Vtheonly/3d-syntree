"""Batch 3 Test Suite: Chemical Physics, Reaction Grammar & MDP Environment.

Runs purely on CPU within 8 GB RAM constraints.
"""

from __future__ import annotations

import math
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

from syntree.chemistry.reactions import (
    HANDLE_SMARTS,
    REACTION_FAMILY_NAMES,
    ReactionEngine,
)
from syntree.chemistry.conformer import (
    ConformerEngine,
    is_conjugated_junction,
)
from syntree.chemistry.validator import ChemicalValidator
from syntree.chemistry.catalog import SynthonCatalog
from syntree.engine.environment import (
    AssemblyAction,
    MolecularAssemblyEnv,
)


class TestReactionGrammar(unittest.TestCase):
    """Tests for syntree/chemistry/reactions.py."""

    def setUp(self):
        self.engine = ReactionEngine()

    def test_01_aryl_fluoride_handle_detection(self):
        """DIAGNOSTIC TEST (F-22): Aryl fluorides must be detected as reactive aryl_halide handles for SNAr."""
        print("\n[TEST] Chemistry: Aryl Fluoride Detection (Auditing F-22)")
        mol = Chem.MolFromSmiles("c1ccccc1F")  # Fluorobenzene
        handles = self.engine.detect_handles(mol)
        handle_types = [h.handle_type for h in handles]
        
        print(f"  -> Detected handles for fluorobenzene: {handle_types}")
        self.assertIn("aryl_halide", handle_types, "Fluorobenzene not detected as aryl_halide! SNAr chemistry broken.")
        print("  -> PASS: Aryl fluoride correctly classified under aryl_halide handle.")

    def test_02_order_reactants_bifunctional_core(self):
        """DIAGNOSTIC TEST (F-09/F-13): order_reactants must check both reactants symmetrically on bifunctional cores."""
        print("\n[TEST] Chemistry: Bifunctional Core Symmetrical Ordering (Auditing F-09/F-13)")
        # Bifunctional core: Glycine (contains both carboxylic acid and primary amine)
        glycine = Chem.MolFromSmiles("NCC(=O)O")
        # Monofunctional synthon: Benzoic acid (contains carboxylic acid only)
        benzoic_acid = Chem.MolFromSmiles("c1ccccc1C(=O)O")
        
        # When coupling glycine and benzoic acid via amide coupling:
        # Glycine must act as the amine (Slot B) and Benzoic acid as the acid (Slot A)
        ordering = self.engine.order_reactants(glycine, benzoic_acid, "amide_coupling")
        self.assertIsNotNone(ordering, "order_reactants returned None for valid coupling partners!")
        
        reactant_a, reactant_b = ordering
        # Reactant A must be benzoic acid (carboxylic acid); Reactant B must be glycine (amine)
        self.assertEqual(reactant_a.GetNumAtoms(), benzoic_acid.GetNumAtoms())
        self.assertEqual(reactant_b.GetNumAtoms(), glycine.GetNumAtoms())
        print("  -> PASS: Reactants correctly ordered without slot inversion.")

    def test_03_forward_reaction_execution_and_provenance(self):
        """ReactionEngine must execute certified reactions and maintain atom provenance maps."""
        print("\n[TEST] Chemistry: Forward Reaction Execution & Atom Provenance")
        acid = Chem.MolFromSmiles("CC(=O)O")  # Acetic acid
        amine = Chem.MolFromSmiles("NCc1ccccc1")  # Benzylamine
        
        res = self.engine.apply_reaction(acid, amine, "amide_coupling")
        self.assertIsNotNone(res)
        
        # Product must be N-benzylacetamide
        smiles = Chem.MolToSmiles(res.product)
        self.assertIn("CC(=O)NCc1ccccc1", smiles)
        self.assertEqual(len(res.junction_bond), 2)
        self.assertGreater(len(res.core_atom_map), 0)
        self.assertGreater(len(res.synthon_atom_map), 0)
        print(f"  -> PASS: Amide product '{smiles}' formed with valid junction {res.junction_bond}.")


class TestConformerAndValidation(unittest.TestCase):
    """Tests for syntree/chemistry/conformer.py and validator.py."""

    def test_04_conformer_resonance_detection_and_embedding(self):
        """ConformerEngine must detect resonance-locked amide bonds and protect them from arbitrary rotation."""
        print("\n[TEST] Conformer: Resonance-Locked Amide Protection")
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC"))  # N-methylacetamide
        mol = ConformerEngine.embed_product(mol)
        self.assertIsNotNone(mol, "Embedding failed on simple amide.")
        
        # Bond (1, 3) is the C(=O)-N single bond
        c_idx, n_idx = 1, 3
        is_locked = is_conjugated_junction(mol, (c_idx, n_idx))
        self.assertTrue(is_locked, "Amide C-N bond failed to be identified as resonance-locked!")
        
        # Attempting set_dihedral on a locked junction must refuse to twist the amide bond
        old_angle = ConformerEngine.get_dihedral(mol, (c_idx, n_idx))
        mol = ConformerEngine.set_dihedral(mol, (c_idx, n_idx), 0.5)
        new_angle = ConformerEngine.get_dihedral(mol, (c_idx, n_idx))
        self.assertAlmostEqual(old_angle, new_angle, places=3, msg="set_dihedral twisted a resonance-locked amide bond!")
        print("  -> PASS: Resonance-locked amide bond identified and protected from unphysical rotation.")

    def test_05_chemical_validator_rules(self):
        """ChemicalValidator must validate sanitized drug-like molecules and reject unphysical motifs."""
        print("\n[TEST] Validator: PoseBusters-Style Validity Filters")
        # 1. Valid molecule with 3D conformer and Fsp3 >= 0.30: Ibuprofen (Fsp3 = 0.46)
        ibuprofen = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))
        AllChem.EmbedMolecule(ibuprofen, randomSeed=42)
        self.assertTrue(ChemicalValidator.is_all_valid(ibuprofen), "Valid 3D Ibuprofen rejected by validator!")

        # 2. Low-Fsp3 rejection: 2D Aspirin (Fsp3 = 0.11 < 0.30)
        aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        checks_aspirin = ChemicalValidator.validate(aspirin)
        self.assertFalse(checks_aspirin["valid_fsp3"], "Validator failed to flag low Fsp3 on Aspirin!")

        # 3. Invalid molecule: Pentavalent carbon
        pentavalent = Chem.MolFromSmiles("C(C)(C)(C)(C)C", sanitize=False)
        checks = ChemicalValidator.validate(pentavalent)
        self.assertFalse(checks["valid_sanitization"])

        # 4. Invalid molecule: Explosive peroxide motif (-O-O-)
        peroxide = Chem.MolFromSmiles("COOC")
        checks_peroxide = ChemicalValidator.validate(peroxide)
        self.assertFalse(checks_peroxide["no_unstable_motifs"])
        print("  -> PASS: Validated 3D Ibuprofen; rejected low-Fsp3, pentavalent carbons, and peroxides.")


class TestEnvironmentAndCatalog(unittest.TestCase):
    """Tests for syntree/chemistry/catalog.py and engine/environment.py."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Build minimal synthetic catalog with sp3-rich synthons
        self.catalog_data = pd.DataFrame([
            {"id": "SYN_001", "smiles": "CC(=O)O", "fsp3": 0.5, "mw": 60.05, "primary_handle": "carboxylic_acid"},
            {"id": "SYN_002", "smiles": "CCN", "fsp3": 1.0, "mw": 45.08, "primary_handle": "primary_secondary_amine"},
            {"id": "SYN_003", "smiles": "c1ccccc1Br", "fsp3": 0.0, "mw": 157.0, "primary_handle": "aryl_halide"},
        ])
        self.catalog_path = os.path.join(self.test_dir, "test_catalog.parquet")
        self.catalog_data.to_parquet(self.catalog_path)
        
        self.catalog = SynthonCatalog(
            catalog_path=self.catalog_path,
            embedding_dim=16,
            min_fsp3=0.0,
            max_mw=250.0,
            encoder="morgan2d",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_06_pick_seed_synthon_catalog_index(self):
        """DIAGNOSTIC TEST (C3/B.4): _pick_seed_synthon must return catalog index, not array position."""
        print("\n[TEST] Environment: Seed Synthon Catalog Index Resolution (Auditing C3/B.4)")
        env = MolecularAssemblyEnv(
            rxn_engine=ReactionEngine(),
            conformer_engine=ConformerEngine(),
            catalog=self.catalog,
            validator=ChemicalValidator(),
            config={"data": {"max_steps_per_molecule": 3, "terminal_cap_min_mw": 100.0}},
            device=torch.device("cpu"),
        )
        
        for _ in range(10):
            seed_idx = env._pick_seed_synthon()
            # Index must be a valid row in self.catalog
            self.assertIn(seed_idx, range(len(self.catalog)))
            mol = self.catalog.get_mol(seed_idx)
            self.assertGreater(len(env.rxn_engine.detect_handles(mol)), 0, "Selected seed lacks reactive handles!")
        print("  -> PASS: Verified 10 seed samplings resolved to valid handle-bearing catalog indices.")

    def test_07_env_step_pre_cap_stop_handling(self):
        """DIAGNOSTIC TEST (F-17): Proposing STOP below MW cap must not fall through to invalid_synthon."""
        print("\n[TEST] Environment: Pre-Cap STOP Action Handling (Auditing F-17)")
        env = MolecularAssemblyEnv(
            rxn_engine=ReactionEngine(),
            conformer_engine=ConformerEngine(),
            catalog=self.catalog,
            validator=ChemicalValidator(),
            config={"data": {"max_steps_per_molecule": 3, "terminal_cap_min_mw": 300.0}},  # High cap
            device=torch.device("cpu"),
        )
        
        # Mock seed molecule
        seed_mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
        seed_mol = ConformerEngine.embed_product(seed_mol)
        env.current_mol = seed_mol
        env.step_count = 0
        env.done = False
        env.horizon = 3
        env._pocket_coords_np = np.zeros((10, 3))
        env._pocket_vdw = np.full(10, 1.7)
        env._pocket_volume = 100.0
        env._pocket_tensors = {
            "pocket_pos": torch.zeros(10, 3, dtype=torch.float32),
            "pocket_z": torch.full((10,), 6, dtype=torch.long),
            "pocket_charge": torch.zeros(10, dtype=torch.float32),
        }

        # Action: STOP (synthon_idx = -1) while MW is ~60 Da (below cap 300 Da)
        stop_action = AssemblyAction(reaction_family_idx=0, synthon_idx=-1, dihedral_rad=0.0)
        obs, reward, done, info = env.step(stop_action)
        
        print(f"  -> STOP below cap result: done={done}, stop_reason='{info.stop_reason}'")
        self.assertNotEqual(info.stop_reason, "invalid_synthon", "STOP below cap fell through to invalid_synthon!")
        self.assertFalse(done, "Episode prematurely terminated on rejected STOP below cap!")
        self.assertEqual(info.stop_reason, "stop_below_cap_rejected")
        print("  -> PASS: STOP below cap rejected gracefully; episode remains active for continued growth.")

    def test_08_catalog_filtering_and_compatibility_masks(self):
        """SynthonCatalog must build valid reaction family masks of shape [F, K]."""
        print("\n[TEST] Catalog: Compatibility Mask Generation")
        masks = self.catalog.get_reaction_family_masks(core_handle="primary_secondary_amine")
        num_families = len(REACTION_FAMILY_NAMES)
        num_synthons = len(self.catalog)
        
        self.assertEqual(masks.shape, (num_families, num_synthons))
        # Family 0 (amide_coupling) with amine core handle should allow carboxylic acids (SYN_001, index 0)
        self.assertGreater(masks[0, 0].item(), -1e8, "SYN_001 (acid) should be legal for amide coupling with amine core!")
        print(f"  -> PASS: Generated [{num_families}, {num_synthons}] compatibility mask with correct grammar filtering.")


if __name__ == "__main__":
    unittest.main(verbosity=2)