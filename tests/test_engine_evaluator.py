"""Unit tests for the evaluation pipeline."""

from __future__ import annotations

import json
import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.engine.evaluator import EvaluationPipeline


def _mol_with_conformer(smiles: str, seed: int = 42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


@pytest.fixture()
def pipeline(tiny_config, tmp_path):
    # Deep-copy: tests mutate evaluation flags and must not leak state into
    # the session-scoped tiny_config used by other tests.
    import copy

    return EvaluationPipeline(
        copy.deepcopy(tiny_config), output_dir=str(tmp_path / "eval")
    )


class TestEvaluate:
    def test_metrics_computed(self, pipeline, tmp_path):
        mols = [
            _mol_with_conformer("OC(=O)C1CCCCC1"),   # Fsp3 = 6/7
            _mol_with_conformer("O=C(OC1CCCCC1)C1CCCCC1"),  # Fsp3 = 12/13
        ]
        recipes = [
            [{"step": 0, "action": "seed", "synthon_id": "A"},
             {"step": 1, "action": "react", "reaction": "esterification"}],
            [{"step": 0, "action": "seed", "synthon_id": "B"}],
        ]
        report = pipeline.evaluate(mols, recipes)
        assert report["num_molecules"] == 2
        assert report["chemical_validity"] == pytest.approx(1.0)
        assert report["recipe_completeness"] == pytest.approx(1.0)
        assert report["mean_fsp3"] == pytest.approx((6 / 7 + 12 / 13) / 2)
        assert report["fsp3_ge_0.42_rate"] == pytest.approx(1.0)
        # AiZynthFinder is unavailable in the test environment: the metric is
        # reported as None with an explicit status instead of a fake proxy.
        assert report["retrosynthetic_feasibility"] is None
        assert "unavailable" in report["retrosynthetic_feasibility_status"]

    def test_report_file_written(self, pipeline):
        mols = [_mol_with_conformer("CCO")]
        pipeline.evaluate(mols, [None])
        report_path = os.path.join(pipeline.output_dir, "evaluation_report.json")
        assert os.path.exists(report_path)
        report = json.load(open(report_path))
        assert report["num_molecules"] == 1
        assert "per_molecule" in report
        assert report["per_molecule"][0]["smiles"] == "CCO"

    def test_invalid_molecule_counted(self, pipeline):
        good = _mol_with_conformer("CCO")
        report = pipeline.evaluate([good, None], [None, None])
        assert report["chemical_validity"] == pytest.approx(0.5)

    def test_flat_molecule_fails_fsp3(self, pipeline):
        flat = _mol_with_conformer("c1ccccc1")  # benzene
        report = pipeline.evaluate([flat], [None])
        assert report["fsp3_ge_0.42_rate"] == pytest.approx(0.0)

    def test_recipe_mismatch_raises(self, pipeline):
        with pytest.raises(ValueError, match="match"):
            pipeline.evaluate([Chem.MolFromSmiles("C")], [])

    def test_per_molecule_diagnostics(self, pipeline):
        mols = [_mol_with_conformer("CC(C)CO")]
        report = pipeline.evaluate(mols, [None])
        entry = report["per_molecule"][0]
        assert entry["descriptors"]["molecular_weight"] > 50
        assert entry["recipe_steps"] == 0
        assert entry["recipe_complete"] is False


class TestDockingHelpers:
    def test_tool_availability_structure(self, pipeline):
        tools = pipeline._tool_availability()
        assert set(tools) == {"gnina", "vina", "smina", "aizynthfinder", "obabel"}
        assert all(isinstance(v, bool) for v in tools.values())

    def test_parse_vina_score(self):
        stdout = (
            "some header\n"
            "Estimated free energy of binding:  -7.3 kcal/mol\n"
            "more lines\n"
        )
        assert EvaluationPipeline._parse_docking_score(stdout) == pytest.approx(-7.3)

    def test_parse_gnina_score(self):
        stdout = "Estimated Affinity:  -8.1\n"
        assert EvaluationPipeline._parse_docking_score(stdout) == pytest.approx(-8.1)

    def test_parse_no_score(self):
        assert EvaluationPipeline._parse_docking_score("nothing here") is None

    def test_parse_malformed_line(self):
        assert EvaluationPipeline._parse_docking_score(
            "Estimated free energy of binding:  garbage"
        ) is None


class TestRetroFeasibility:
    """The retrosynthesis metric is a genuine AiZynthFinder solve rate or
    nothing at all: recipe completeness must never masquerade as
    retrosynthetic feasibility."""

    def test_unavailable_without_aizynthfinder(self, pipeline, monkeypatch):
        import syntree.engine.evaluator as ev

        monkeypatch.setattr(ev.shutil, "which", lambda name: None)
        per_mol = [
            {"recipe_complete": True, "smiles": "CCO"},
            {"recipe_complete": True, "smiles": "CCC"},
            {"recipe_complete": False, "smiles": "CCCC"},
        ]
        assert pipeline._retro_feasibility(per_mol) is None
        assert pipeline._retro_status == "unavailable_missing_aizynthcli"

    def test_disabled_reports_none(self, pipeline, monkeypatch):
        import syntree.engine.evaluator as ev

        monkeypatch.setattr(
            ev.shutil, "which", lambda name: "/usr/bin/aizynthcli"
        )
        pipeline.config.setdefault("evaluation", {})["run_aizynthfinder"] = False
        assert pipeline._retro_feasibility([{"smiles": "CCO"}]) is None
        assert pipeline._retro_status == "disabled"

    def test_missing_config_reports_none(self, pipeline, monkeypatch):
        import syntree.engine.evaluator as ev

        monkeypatch.setattr(
            ev.shutil, "which", lambda name: "/usr/bin/aizynthcli"
        )
        assert pipeline._retro_feasibility([{"smiles": "CCO"}]) is None
        assert pipeline._retro_status == "unavailable_missing_config"
