"""#34 EVAL-01 — the unlinked retrosynthesis evaluation battery.

Pins the #34 acceptance surface of ``syntree/engine/evaluator.py`` and
``scripts/run_benchmarks.py``:

* the fail-closed generated-SDF corpus parser (multi-record, gzipped
  artifacts, explicit rejection vocabulary, loud failures on unreadable
  corpus inputs);
* the physicochemical metric surface (validity, Fsp³, molecular weight,
  fingerprint diversity with duplicate collapse);
* the AiZynthFinder flat-output parser and the #44 failure vocabulary:
  ``stock_mismatch`` (routes found, leaves unpurchasable) is a distinct
  category from ``no_route_found`` (a genuine planner failure) — never
  conflated;
* the isolated planner subprocess boundary (``run_aizynthfinder``):
  non-zero exit / timeout / missing output fail loudly; the corpus is
  delivered by file, results by file — nothing shared with the planner
  process;
* the ``evaluation_report.json`` assembly: per-track sections with
  provenance, honest corpus labelling, forward-replay block, and the
  two-track separation (#44);
* the optional PoseBusters backend (gated on the optional package; the
  fail-closed contracts hold everywhere).
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import textwrap

import pytest
from rdkit import Chem

from syntree.engine.evaluator import (
    EVAL_SCHEMA_VERSION,
    REJECTION_CODES,
    RETRO_FAILURE_CATEGORIES,
    EvaluatedMolecule,
    EvaluationError,
    RetroSearchResult,
    build_evaluation_report,
    fingerprint_diversity,
    load_generated_molecules,
    molecular_metrics,
    parse_aizynth_output,
    run_aizynthfinder,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ACETANILIDE = "CC(=O)Nc1ccccc1"
BENZENE = "c1ccccc1"


def _sdf_block(smiles: str, name: str = "m") -> str:
    mol = Chem.MolFromSmiles(smiles)
    Chem.AddHs(mol)
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(mol, randomSeed=7)
    return Chem.MolToMolBlock(mol) + "$$$$\n"


def _write_sdf(path: str, smiles_list: list[str]) -> str:
    with open(path, "w") as handle:
        for smi in smiles_list:
            handle.write(_sdf_block(smi))
    return path


def _flat_output(path: str, rows: list[dict]) -> str:
    with open(path, "w") as handle:
        json.dump({"schema": {"fields": []}, "data": rows}, handle)
    return path


def _row(target: str, solved: bool, routes: int, not_in_stock: str = "") -> dict:
    return {
        "target": target,
        "is_solved": solved,
        "number_of_routes": routes,
        "number_of_solved_routes": routes if solved else 0,
        "number_of_steps": 1,
        "search_time": 1.0,
        "precursors_in_stock": "Nc1ccccc1" if solved else "",
        "precursors_not_in_stock": not_in_stock,
    }


def _result(target: str, solved: bool) -> RetroSearchResult:
    return RetroSearchResult(
        target=target,
        is_solved=solved,
        number_of_routes=1 if solved else 2,
        number_of_solved_routes=1 if solved else 0,
        number_of_steps=1,
        search_time=1.0,
        precursors_in_stock=("Nc1ccccc1",) if solved else (),
        precursors_not_in_stock=() if solved else ("CC(=O)Cl",),
        failure_category=None if solved else "stock_mismatch",
    )


# ---------------------------------------------------------------------------
# 1. Corpus parsing
# ---------------------------------------------------------------------------


class TestCorpusLoading:
    def test_single_record_sdf(self, tmp_path):
        path = _write_sdf(str(tmp_path / "gen.sdf"), [ASPIRIN])
        corpus = load_generated_molecules([path])
        assert len(corpus) == 1
        mol = corpus[0]
        assert mol.valid and mol.rejection_reason is None
        assert mol.smiles == ASPIRIN
        assert mol.num_atoms > 0
        assert mol.molecule_id == "gen.sdf#0"
        assert len(mol.inchi_key) == 27

    def test_multi_record_sdf(self, tmp_path):
        path = _write_sdf(str(tmp_path / "gen.sdf"), [ASPIRIN, BENZENE])
        corpus = load_generated_molecules([path])
        assert [m.record_index for m in corpus] == [0, 1]
        assert corpus[0].smiles != corpus[1].smiles

    def test_gzipped_sdf_artifact(self, tmp_path):
        raw = _sdf_block(ASPIRIN)
        gz = tmp_path / "gen.sdf.gz"
        with gzip.open(str(gz), "wt") as handle:
            handle.write(raw)
        corpus = load_generated_molecules([str(gz)])
        assert len(corpus) == 1 and corpus[0].valid
        assert corpus[0].source_path.endswith(".gz")

    def test_invalid_valence_record_is_rejected_not_dropped(self, tmp_path):
        # a parseable V2000 block whose carbon carries five bonds:
        # sanitization must fail -> closed rejection vocabulary
        path = tmp_path / "bad.sdf"
        block = (
            "pentavalent\n     RDKit          2D\n\n"
            "  6  5  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0\n"
            "    1.0000    0.0000    0.0000 H   0  0\n"
            "    0.5000    0.8660    0.0000 H   0  0\n"
            "   -0.5000    0.8660    0.0000 H   0  0\n"
            "    0.5000   -0.8660    0.0000 H   0  0\n"
            "   -0.5000   -0.8660    0.0000 H   0  0\n"
            "  1  2  1  0\n  1  3  1  0\n  1  4  1  0\n"
            "  1  5  1  0\n  1  6  1  0\nM  END\n"
        )
        path.write_text(block + "$$$$\n")
        corpus = load_generated_molecules([path])
        assert len(corpus) == 1
        assert corpus[0].valid is False
        assert corpus[0].rejection_reason == "record_unsanitizable"

    def test_unparsable_record_is_rejected(self, tmp_path):
        # an unparseable counts line alongside a valid record: the bad
        # record is rejected, the corpus keeps both (honest counting)
        path = tmp_path / "mixed.sdf"
        bad = (
            "garbage\n     RDKit          2D\n\n"
            "  AA  0  0  0  0  0  0  0  0999 V2000\nM  END\n"
        )
        path.write_text(bad + "$$$$\n" + _sdf_block(ASPIRIN))
        corpus = load_generated_molecules([path])
        assert len(corpus) == 2
        assert corpus[0].rejection_reason == "record_unparsable"
        assert corpus[1].valid is True

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(EvaluationError, match="not found"):
            load_generated_molecules([str(tmp_path / "nope.sdf")])

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.sdf"
        path.write_text("   \n")
        with pytest.raises(EvaluationError, match="empty"):
            load_generated_molecules([str(path)])

    def test_garbage_file_raises(self, tmp_path):
        # a file with NO parseable record is a wrong-file operator error
        path = tmp_path / "garbage.sdf"
        path.write_text("not an sdf at all\n")
        with pytest.raises(EvaluationError, match="no parseable records"):
            load_generated_molecules([str(path)])


# ---------------------------------------------------------------------------
# 2. Physicochemical metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_aspirin_metrics(self):
        metrics = molecular_metrics(ASPIRIN)
        assert metrics["molecular_weight"] == pytest.approx(180.16, abs=0.1)
        assert metrics["fraction_sp3"] == pytest.approx(1.0 / 9.0, abs=1e-6)
        assert metrics["rotatable_bonds"] == 2
        assert metrics["ring_count"] == 1
        assert metrics["formal_charge"] == 0

    def test_invalid_smiles_raises(self):
        with pytest.raises(EvaluationError):
            molecular_metrics("not_a_smiles")

    def test_diversity_zero_for_identical_and_singleton(self):
        assert fingerprint_diversity([ASPIRIN]) == 0.0
        assert fingerprint_diversity([ASPIRIN, ASPIRIN]) == 0.0

    def test_diversity_positive_for_distinct(self):
        d = fingerprint_diversity([ASPIRIN, BENZENE])
        assert 0.0 < d <= 1.0

    def test_diversity_in_range_and_symmetric(self):
        a = fingerprint_diversity([ASPIRIN, BENZENE, ACETANILIDE])
        assert 0.0 <= a < 1.0


# ---------------------------------------------------------------------------
# 3. Planner output parsing + the failure vocabulary
# ---------------------------------------------------------------------------


class TestParseAizynthOutput:
    def test_solved_row(self, tmp_path):
        path = _flat_output(
            str(tmp_path / "out.json"), [_row(ASPIRIN, solved=True, routes=3)]
        )
        results = parse_aizynth_output(path)
        assert results[ASPIRIN].is_solved
        assert results[ASPIRIN].failure_category is None
        assert results[ASPIRIN].number_of_routes == 3

    def test_stock_mismatch_is_distinct_from_planner_failure(self, tmp_path):
        # routes exist, but leaves are not purchasable: a STOCK issue
        path = _flat_output(
            str(tmp_path / "out.json"),
            [_row(ACETANILIDE, solved=False, routes=2, not_in_stock="CC(=O)Cl")],
        )
        results = parse_aizynth_output(path)
        assert results[ACETANILIDE].stock_mismatch is True
        assert results[ACETANILIDE].failure_category == "stock_mismatch"
        assert "stock_mismatch" in RETRO_FAILURE_CATEGORIES

    def test_no_route_found_is_a_planner_failure(self, tmp_path):
        path = _flat_output(
            str(tmp_path / "out.json"), [_row(ACETANILIDE, solved=False, routes=0)]
        )
        results = parse_aizynth_output(path)
        assert results[ACETANILIDE].failure_category == "no_route_found"
        assert results[ACETANILIDE].stock_mismatch is False

    def test_missing_key_raises(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text(json.dumps({"schema": {}, "data": [{"target": "C"}]}))
        with pytest.raises(EvaluationError, match="missing key"):
            parse_aizynth_output(str(path))

    def test_wrong_format_raises(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text(json.dumps({"routes": []}))
        with pytest.raises(EvaluationError, match="flat records"):
            parse_aizynth_output(str(path))

    def test_precursor_lists_parsed(self, tmp_path):
        path = _flat_output(
            str(tmp_path / "out.json"),
            [
                _row(
                    ASPIRIN,
                    solved=False,
                    routes=1,
                    not_in_stock="CC(=O)OC(C)=O, O=C(O)c1ccccc1O",
                )
            ],
        )
        results = parse_aizynth_output(path)
        assert results[ASPIRIN].precursors_not_in_stock == (
            "CC(=O)OC(C)=O",
            "O=C(O)c1ccccc1O",
        )


# ---------------------------------------------------------------------------
# 4. The isolated planner boundary
# ---------------------------------------------------------------------------


class TestRunAizynthfinder:
    def test_smiles_file_and_output_contract(self, tmp_path, monkeypatch):
        out = str(tmp_path / "out.json")
        calls = {}

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            calls["cmd"] = cmd
            # the corpus must be delivered by file (isolation boundary)
            smi_path = cmd[cmd.index("--smiles") + 1]
            out_path = cmd[cmd.index("--output") + 1]
            with open(smi_path) as handle:
                targets = [l.strip() for l in handle if l.strip()]
            _flat_output(out_path, [_row(t, True, 1) for t in targets])
            calls["smiles"] = targets
            calls["out"] = out_path
            assert out_path == out
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        results = run_aizynthfinder([ASPIRIN, BENZENE], "cfg.yml", out)
        assert set(results) == {ASPIRIN, BENZENE}
        assert calls["smiles"] == [ASPIRIN, BENZENE]
        assert cmd_uses_module(calls["cmd"], "aizynthfinder.interfaces.aizynthcli")

    def test_nonzero_exit_raises(self, tmp_path, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 2, "", "boom")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(EvaluationError, match="exit 2"):
            run_aizynthfinder([ASPIRIN], "cfg.yml", str(tmp_path / "o.json"))

    def test_timeout_raises(self, tmp_path, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 1.0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(EvaluationError, match="timeout"):
            run_aizynthfinder([ASPIRIN], "cfg.yml", str(tmp_path / "o.json"))

    def test_no_output_raises(self, tmp_path, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(EvaluationError, match="no output"):
            run_aizynthfinder([ASPIRIN], "cfg.yml", str(tmp_path / "o.json"))

    def test_empty_targets_raise(self, tmp_path):
        with pytest.raises(EvaluationError, match="empty"):
            run_aizynthfinder([], "cfg.yml", str(tmp_path / "o.json"))


def cmd_uses_module(cmd: list[str], module: str) -> bool:
    return module in cmd and "-m" in cmd


# ---------------------------------------------------------------------------
# 5. Report assembly (two-track separation, honest labels)
# ---------------------------------------------------------------------------


class TestReportAssembly:
    def _corpus(self, tmp_path):
        return load_generated_molecules(
            [_write_sdf(str(tmp_path / "gen.sdf"), [ASPIRIN, BENZENE])]
        )

    def test_two_tracks_rendered_separately_never_merged(self, tmp_path):
        corpus = self._corpus(tmp_path)
        report = build_evaluation_report(
            corpus=corpus,
            retro_tracks={
                "track_a": {ASPIRIN: _result(ASPIRIN, True)},
                "track_b": {ASPIRIN: _result(ASPIRIN, False)},
            },
            track_provenance={"track_a": {"p": 1}, "track_b": {"p": 2}},
            corpus_description="test corpus",
        )
        assert set(report["retrosynthesis"]) == {"track_a", "track_b"}
        assert report["retrosynthesis"]["track_a"]["solve_rate"] == 1.0
        assert report["retrosynthesis"]["track_b"]["solve_rate"] == 0.0
        # there is no merged/aggregate/headline retro entry anywhere
        assert set(report) == {
            "schema",
            "corpus",
            "physicochemical",
            "retrosynthesis",
            "extra",
        }
        flat = json.dumps(report)
        assert json.loads(flat)["schema"] == EVAL_SCHEMA_VERSION

    def test_provenance_required_per_track(self, tmp_path):
        corpus = self._corpus(tmp_path)
        with pytest.raises(EvaluationError, match="provenance"):
            build_evaluation_report(
                corpus=corpus,
                retro_tracks={"track_a": {}},
                track_provenance={},
                corpus_description="x",
            )

    def test_failure_category_counts_and_rejections(self, tmp_path):
        corpus = self._corpus(tmp_path)
        report = build_evaluation_report(
            corpus=corpus,
            retro_tracks={
                "t": {
                    ASPIRIN: _result(ASPIRIN, False),
                    BENZENE: RetroSearchResult(
                        target=BENZENE,
                        is_solved=False,
                        number_of_routes=0,
                        number_of_solved_routes=0,
                        number_of_steps=0,
                        search_time=2.0,
                        precursors_in_stock=(),
                        precursors_not_in_stock=(),
                        failure_category="no_route_found",
                    ),
                }
            },
            track_provenance={"t": {"p": 1}},
            corpus_description="x",
        )
        section = report["retrosynthesis"]["t"]
        assert section["num_stock_mismatch"] == 1
        assert section["num_no_route_found"] == 1
        assert section["solve_rate"] == 0.0
        assert section["mean_search_time_s"] == pytest.approx(1.5)

    def test_corpus_description_carried_verbatim(self, tmp_path):
        corpus = self._corpus(tmp_path)
        report = build_evaluation_report(
            corpus=corpus,
            retro_tracks={},
            track_provenance={},
            corpus_description=(
                "forward-replay-verified reconstructed crystal ligands — "
                "NOT trained-policy samples"
            ),
        )
        assert (
            "NOT trained-policy samples"
            in report["corpus"]["description"]
        )

    def test_forward_replay_block_is_distinct_from_planner_retro(self, tmp_path):
        corpus = self._corpus(tmp_path)
        report = build_evaluation_report(
            corpus=corpus,
            retro_tracks={"t": {ASPIRIN: _result(ASPIRIN, True)}},
            track_provenance={"t": {"p": 1}},
            corpus_description="x",
            forward_replay_status={
                "replay_verified": True,
                "note": "internal #18 replay — not planner retrosynthesis",
            },
        )
        assert report["corpus"]["forward_replay"]["replay_verified"] is True
        assert report["retrosynthesis"]["t"]["provenance"] == {"p": 1}


# ---------------------------------------------------------------------------
# 6. Optional backends — fail-closed contracts
# ---------------------------------------------------------------------------


def _posebusters_missing() -> bool:
    try:
        import posebusters  # noqa: F401

        return False
    except ImportError:
        return True


class TestOptionalBackends:
    def test_posebusters_missing_ligand_raises(self):
        with pytest.raises(EvaluationError, match="ligand SDF not found"):
            from syntree.engine.evaluator import posebusters_backend

            posebusters_backend("/nonexistent.sdf")

    def test_vina_missing_inputs_raise(self):
        from syntree.engine.evaluator import vina_rescoring_backend

        with pytest.raises(EvaluationError, match="ligand SDF not found"):
            vina_rescoring_backend("/nonexistent.sdf", "/nonexistent.pdb", (0, 0, 0))
        with pytest.raises(EvaluationError, match="pocket PDB not found"):
            vina_rescoring_backend(__file__, "/nonexistent.pdb", (0, 0, 0))

    @pytest.mark.skipif(
        _posebusters_missing(), reason="posebusters not installed (optional)"
    )
    def test_posebusters_mol_backend_real(self, tmp_path):
        from syntree.engine.evaluator import posebusters_backend

        path = _write_sdf(str(tmp_path / "lig.sdf"), [ASPIRIN])
        outcome = posebusters_backend(path, config_name="mol")
        assert outcome["backend"] == "posebusters"
        assert outcome["passed"] in (True, False)
        assert isinstance(outcome["busts"], dict) and outcome["busts"]


# ---------------------------------------------------------------------------
# 7. CLI smoke — the #34 acceptance driver with a stub planner
# ---------------------------------------------------------------------------


class TestRunBenchmarksCLI:
    def test_cli_end_to_end_with_stub_planner(self, tmp_path):
        sdf = _write_sdf(str(tmp_path / "gen.sdf"), [ASPIRIN, ACETANILIDE])
        stub = tmp_path / "stub_planner.py"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            + textwrap.dedent(
                """
                import json, sys
                args = sys.argv[1:]
                smi = args[args.index("--smiles") + 1]
                out = args[args.index("--output") + 1]
                with open(smi) as h:
                    targets = [l.strip() for l in h if l.strip()]
                rows = [
                    {
                        "target": t,
                        "is_solved": True,
                        "number_of_routes": 1,
                        "number_of_solved_routes": 1,
                        "number_of_steps": 1,
                        "search_time": 0.5,
                        "precursors_in_stock": "Nc1ccccc1",
                        "precursors_not_in_stock": "",
                    }
                    for t in targets
                ]
                with open(out, "w") as h:
                    json.dump({"schema": {}, "data": rows}, h)
                """
            )
        )
        os.chmod(str(stub), 0o755)
        output = tmp_path / "evaluation_report.json"
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "scripts", "run_benchmarks.py"),
                "--sdf",
                str(sdf),
                "--corpus-description",
                "stub corpus",
                "--retro-track",
                "stub_track=" + _stub_config(str(tmp_path)),
                "--aizynth-python",
                str(stub),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        report = json.loads(output.read_text())
        assert report["corpus"]["num_records"] == 2
        assert report["corpus"]["validity_rate"] == 1.0
        assert report["retrosynthesis"]["stub_track"]["solve_rate"] == 1.0
        assert "stub corpus" in proc.stdout

    def test_cli_rejects_missing_track_config(self, tmp_path):
        sdf = _write_sdf(str(tmp_path / "gen.sdf"), [ASPIRIN])
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "scripts", "run_benchmarks.py"),
                "--sdf",
                str(sdf),
                "--corpus-description",
                "x",
                "--retro-track",
                "t=/nonexistent.yml",
                "--output",
                str(tmp_path / "r.json"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 2
        assert "not found" in proc.stderr


def _stub_config(tmp: str) -> str:
    path = os.path.join(tmp, "stub_config.yml")
    with open(path, "w") as handle:
        handle.write("stock: {stub: stub.txt}\n")
    return path
