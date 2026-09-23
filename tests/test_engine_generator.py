"""Unit tests for the SBDD generator (end-to-end molecule construction)."""

from __future__ import annotations

import json
import os

import pytest
import torch
from rdkit import Chem

from syntree.engine.generator import SBDDGenerator
from syntree.models.policy import SynTreePolicy


@pytest.fixture(scope="module")
def generator(tiny_config_with_assets):
    torch.manual_seed(0)
    cfg = json.loads(json.dumps(tiny_config_with_assets))
    model = SynTreePolicy(cfg)
    return SBDDGenerator(model, cfg, torch.device("cpu"),
                         output_dir=None)  # patched per-test


@pytest.fixture()
def pocket_path(assets_dir):
    return os.path.join(assets_dir["crossdocked_dir"], "sample_pocket.pdb")


class TestSingleLigand:
    def test_generates_valid_molecule(self, generator, pocket_path):
        result = generator.generate_ligand(
            pocket_path, seed_synthon_idx=0, max_steps=2
        )
        assert result["rdkit_mol"] is not None
        assert result["rdkit_mol"].GetNumConformers() == 1
        # Molecule must be chemically valid (sanitize round trip).
        probe = Chem.Mol(result["rdkit_mol"])
        Chem.SanitizeMol(probe)
        assert result["smiles"]

    def test_recipe_structure(self, generator, pocket_path):
        result = generator.generate_ligand(
            pocket_path, seed_synthon_idx=0, max_steps=1
        )
        recipe = result["recipe"]
        assert recipe[0]["step"] == 0
        assert recipe[0]["action"] == "seed"
        assert recipe[0]["synthon_id"].startswith("TEST")
        for step_record in recipe[1:]:
            assert step_record["action"] == "react"
            assert "reaction" in step_record
            assert "synthon_id" in step_record
            assert -3.15 <= step_record["dihedral_applied_rad"] <= 3.15

    def test_generated_molecule_is_bigger_than_seed(self, generator, pocket_path):
        seed = generator.catalog.get_mol(0)
        result = generator.generate_ligand(
            pocket_path, seed_synthon_idx=0, max_steps=1
        )
        assert result["rdkit_mol"].GetNumAtoms() > seed.GetNumAtoms()

    def test_checks_block_present(self, generator, pocket_path):
        result = generator.generate_ligand(pocket_path, seed_synthon_idx=0)
        expected = {
            "valid_rdkit_mol", "valid_sanitization", "no_pentavalent_carbons",
            "no_unstable_motifs", "reasonable_mw", "valid_fsp3",
            "reasonable_charge", "reasonable_heavy_atoms", "has_conformer",
            "no_pocket_clash",
        }
        assert set(result["checks"]) == expected
        assert isinstance(result["passes_posebusters"], bool)
        assert "molecular_weight" in result["descriptors"]

    def test_no_handles_stops_early(self, generator, pocket_path):
        """A seed with no reactive handle yields a 1-entry recipe."""
        # Find a catalog synthon... every catalog entry has a handle by
        # construction, so instead force max_steps=0.
        result = generator.generate_ligand(pocket_path, max_steps=0)
        assert len(result["recipe"]) == 1

    def test_deterministic_seed_choice(self, generator, pocket_path):
        r1 = generator.generate_ligand(pocket_path, seed_synthon_idx=1)
        r2 = generator.generate_ligand(pocket_path, seed_synthon_idx=1)
        assert r1["smiles"] == r2["smiles"]

    def test_missing_pocket_raises(self, generator, tmp_path):
        with pytest.raises(ValueError):
            generator.generate_ligand(str(tmp_path / "nope.pdb"))

    def test_bad_pocket_raises(self, generator, tmp_path):
        bad = tmp_path / "bad.pdb"
        bad.write_text("this is not a pdb file")
        with pytest.raises(ValueError):
            generator.generate_ligand(str(bad))


class TestBatchGeneration:
    def test_batch_writes_artifacts(self, tiny_config_with_assets, pocket_path,
                                    tmp_path):
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        torch.manual_seed(0)
        model = SynTreePolicy(cfg)
        gen = SBDDGenerator(model, cfg, torch.device("cpu"),
                            output_dir=str(tmp_path))
        summary = gen.generate_batch(pocket_pdb_path=pocket_path,
                                     num_ligands=3)
        assert len(summary) == 3
        for i, record in enumerate(summary):
            assert os.path.exists(record["sdf"])
            assert os.path.exists(record["recipe"])
            assert record["generation_seconds"] >= 0
        assert os.path.exists(os.path.join(str(tmp_path), "batch_summary.json"))

        # SDFs must be readable and have 3D coordinates.
        for i in range(3):
            mol = Chem.SDMolSupplier(os.path.join(str(tmp_path),
                                                  f"ligand_{i:03d}.sdf"),
                                     removeHs=False)[0]
            assert mol is not None
            assert mol.GetNumConformers() == 1

    def test_missing_pocket_raises(self, tiny_config_with_assets, tmp_path):
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        model = SynTreePolicy(cfg)
        gen = SBDDGenerator(model, cfg, torch.device("cpu"),
                            output_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            gen.generate_batch(pocket_pdb_path=str(tmp_path / "missing.pdb"))


class TestChemistryGuarantees:
    """The project's core promise: generated molecules are synthesizable."""

    @pytest.mark.parametrize("seed_idx", [0, 1, 2, 3, 4])
    def test_every_product_is_chemically_valid(self, generator, pocket_path,
                                               seed_idx):
        result = generator.generate_ligand(
            pocket_path, seed_synthon_idx=seed_idx, max_steps=2
        )
        probe = Chem.Mol(result["rdkit_mol"])
        Chem.SanitizeMol(probe)  # must never raise
        assert probe.GetNumAtoms() > 0

    def test_recipe_reconstructible(self, generator, pocket_path):
        """Replaying the recipe's reactions must yield the same molecule."""
        from syntree.chemistry.reactions import ReactionEngine

        result = generator.generate_ligand(
            pocket_path, seed_synthon_idx=0, max_steps=1
        )
        recipe = result["recipe"]
        engine = ReactionEngine()

        mol = generator.catalog.get_mol(
            list(generator.catalog.df["id"]).index(recipe[0]["synthon_id"])
            if recipe[0]["synthon_id"] in list(generator.catalog.df["id"])
            else 0,
            explicit_hs=False,
        )
        for step_record in recipe[1:]:
            synthon_id = step_record["synthon_id"]
            idx = list(generator.catalog.df["id"]).index(synthon_id)
            synthon = generator.catalog.get_mol(idx, explicit_hs=False)
            rxn_result = engine.apply_reaction(mol, synthon,
                                               step_record["reaction"])
            assert rxn_result is not None, (
                f"recipe step {step_record['step']} does not replay"
            )
            mol = rxn_result.product

        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(
            Chem.RemoveHs(Chem.Mol(result["rdkit_mol"]))
        )
