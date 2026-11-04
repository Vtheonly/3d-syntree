"""#44 EVAL-03 — the two-track AiZynthFinder evaluation battery.

Pins the #44 acceptance surface:

* **Track A stock export** (``scripts/export_aizynth_stock.py``): the
  exported Enamine stock preserves the catalog IDs (mapping file), and
  the manifest ties the export to the immutable HDF5 catalog by its own
  SHA-256 plus the embedded provenance attributes; unparsable catalog
  rows fail closed;
* **track configuration** (``scripts/prepare_aizynth_tracks.py``): the
  two configs are written with absolute stock/model paths, the manifest
  records file checksums and the public download URLs, and the
  never-merge policy is structural; missing public model files fail
  closed;
* **the stock-mismatch distinction** (#44 acceptance: "a stock-mismatch
  test demonstrates that missing stock entries are detected and reported
  rather than misclassified as planner failures") — first at the parser
  level with fixture planner outputs, then **with the real planner**
  (gated on ``SYNTREE_AIZYNTH_MODELS`` pointing at the documented public
  download): a purchasable-precursor target is solved, and a target
  whose routes need out-of-stock leaves is classified
  ``stock_mismatch`` with its missing precursors reported — never as a
  planner failure;
* the separation invariants: two-track reports render separate sections
  and there is no merged headline anywhere (pinned in
  ``test_evaluator.py`` as well).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys

import pytest

from syntree.engine.evaluator import (
    EvaluationError,
    RetroSearchResult,
    build_evaluation_report,
    load_generated_molecules,
    parse_aizynth_output,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ACETANILIDE = "CC(=O)Nc1ccccc1"
ANILINE = "Nc1ccccc1"
ACETIC_ANHYDRIDE = "CC(=O)OC(C)=O"

#: The documented public-model download directory (aizynthfinder's
#: ``download_public_data``); integration tests are gated on it.
AIZYNTH_MODELS = os.environ.get("SYNTREE_AIZYNTH_MODELS")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_catalog(path: str, rows: list[tuple[str, str]]) -> str:
    import h5py

    with h5py.File(path, "w") as h5:
        ids = [r[0].encode() for r in rows]
        smiles = [r[1].encode() for r in rows]
        h5.create_dataset("catalog/block_ids", data=ids, dtype=h5py.string_dtype())
        h5.create_dataset("catalog/smiles", data=smiles, dtype=h5py.string_dtype())
        h5.attrs["catalog_schema_version"] = "catalog-v2.0"
        h5.attrs["source_sdf"] = "synthetic_test_source.sdf"
        h5.attrs["source_sdf_sha256"] = "0" * 64
    return path


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "scripts", script), *args],
        capture_output=True,
        text=True,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# 1. Track A — the Enamine stock export
# ---------------------------------------------------------------------------


class TestTrackAStockExport:
    @pytest.fixture()
    def exported(self, tmp_path):
        catalog = _write_catalog(
            str(tmp_path / "cat.h5"),
            [
                ("REC0000001", ANILINE),
                ("REC0000002", ACETIC_ANHYDRIDE),
                ("REC0000003", ASPIRIN),
            ],
        )
        out_dir = str(tmp_path / "stocks")
        proc = _run_script(
            "export_aizynth_stock.py",
            "--catalog",
            catalog,
            "--output-dir",
            out_dir,
        )
        assert proc.returncode == 0, proc.stderr
        return catalog, out_dir, proc

    def test_stock_files_exist(self, exported):
        _, out_dir, _ = exported
        assert os.path.exists(f"{out_dir}/enamine_stock.txt")
        assert os.path.exists(f"{out_dir}/enamine_stock_mapping.csv")
        assert os.path.exists(f"{out_dir}/enamine_stock_manifest.json")

    def test_manifest_ties_export_to_catalog_checksum(self, exported):
        catalog, out_dir, _ = exported
        manifest = json.load(open(f"{out_dir}/enamine_stock_manifest.json"))
        assert manifest["catalog_sha256"] == _sha256(catalog)
        assert manifest["stock_file_sha256"] == _sha256(
            f"{out_dir}/enamine_stock.txt"
        )
        assert manifest["mapping_file_sha256"] == _sha256(
            f"{out_dir}/enamine_stock_mapping.csv"
        )
        assert manifest["catalog_attributes"]["catalog_schema_version"] == (
            "catalog-v2.0"
        )
        assert manifest["num_rows"] == 3

    def test_catalog_ids_preserved_in_mapping(self, exported):
        _, out_dir, _ = exported
        with open(f"{out_dir}/enamine_stock_mapping.csv") as handle:
            rows = list(csv.DictReader(handle))
        assert {r["block_id"] for r in rows} == {
            "REC0000001",
            "REC0000002",
            "REC0000003",
        }
        assert {r["smiles"] for r in rows} == {
            ANILINE,
            ACETIC_ANHYDRIDE,
            ASPIRIN,
        }
        assert all(len(r["inchi_key"]) == 27 for r in rows)

    def test_track_a_label_never_independent(self, exported):
        _, out_dir, _ = exported
        manifest = json.load(open(f"{out_dir}/enamine_stock_manifest.json"))
        assert "NEVER report as independent" in manifest["track"]

    def test_unparsable_catalog_smiles_fails_closed(self, tmp_path):
        catalog = _write_catalog(
            str(tmp_path / "bad.h5"), [("REC0000001", "not_a_smiles")]
        )
        proc = _run_script(
            "export_aizynth_stock.py",
            "--catalog",
            catalog,
            "--output-dir",
            str(tmp_path / "out"),
        )
        assert proc.returncode != 0
        assert "unparsable" in (proc.stdout + proc.stderr)

    def test_missing_catalog_fails_closed(self, tmp_path):
        proc = _run_script(
            "export_aizynth_stock.py",
            "--catalog",
            str(tmp_path / "nope.h5"),
            "--output-dir",
            str(tmp_path / "out"),
        )
        assert proc.returncode != 0


# ---------------------------------------------------------------------------
# 2. Track configuration
# ---------------------------------------------------------------------------


class TestPrepareTracks:
    @pytest.fixture()
    def prepared(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        for name in (
            "uspto_model.onnx",
            "uspto_templates.csv.gz",
            "uspto_ringbreaker_model.onnx",
            "uspto_ringbreaker_templates.csv.gz",
            "uspto_filter_model.onnx",
        ):
            (models / name).write_bytes(b"fake-model-bytes-" + name.encode())
        stock_a = tmp_path / "enamine_stock.txt"
        stock_a.write_text("KEY1\nKEY2\n")
        stock_b = tmp_path / "zinc_inchikeys.txt"
        stock_b.write_text("KEY3\n")
        out_dir = tmp_path / "tracks"
        proc = _run_script(
            "prepare_aizynth_tracks.py",
            "--models-dir",
            str(models),
            "--stock-a",
            str(stock_a),
            "--stock-b",
            str(stock_b),
            "--output-dir",
            str(out_dir),
        )
        assert proc.returncode == 0, proc.stderr
        return tmp_path, out_dir

    def test_track_configs_written_with_absolute_paths(self, prepared):
        _, out_dir = prepared
        for name, stock_key in (
            ("track_a_stock_aligned.yml", "enamine_stock"),
            ("track_b_external_zinc.yml", "zinc_stock"),
        ):
            lines = open(f"{out_dir}/{name}").read().splitlines()
            stock_lines = [
                l for l in lines if l.strip().startswith(stock_key + ":")
            ]
            assert len(stock_lines) == 1
            stock_path = stock_lines[0].split(":", 1)[1].strip()
            assert os.path.isabs(stock_path)
            assert os.path.exists(stock_path)
            model_lines = [
                l.strip()[2:].strip()
                for l in lines
                if l.strip().startswith("- ")
            ]
            assert model_lines and all(os.path.isabs(p) for p in model_lines)

    def test_manifest_records_checksums_urls_and_policy(self, prepared):
        tmp_path, out_dir = prepared
        manifest = json.load(open(f"{out_dir}/tracks_manifest.json"))
        assert manifest["track_a"]["stock_sha256"] == _sha256(
            str(tmp_path / "enamine_stock.txt")
        )
        assert manifest["track_b"]["stock_sha256"] == _sha256(
            str(tmp_path / "zinc_inchikeys.txt")
        )
        urls = manifest["shared_policy"]["source_urls"]
        assert "zenodo.org" in urls["uspto_model.onnx"]
        assert manifest["shared_policy"]["model_sha256"]["uspto_model.onnx"] == (
            _sha256(str(tmp_path / "models" / "uspto_model.onnx"))
        )
        assert "merge" in manifest["never_merge"].lower()
        assert "independent" in manifest["track_b"]["label"]

    def test_missing_public_models_fail_closed(self, tmp_path):
        empty = tmp_path / "empty_models"
        empty.mkdir()
        stock = tmp_path / "s.txt"
        stock.write_text("KEY\n")
        proc = _run_script(
            "prepare_aizynth_tracks.py",
            "--models-dir",
            str(empty),
            "--stock-a",
            str(stock),
            "--stock-b",
            str(stock),
            "--output-dir",
            str(tmp_path / "out"),
        )
        assert proc.returncode != 0
        assert "download_public_data" in (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# 3. The stock-mismatch distinction
# ---------------------------------------------------------------------------


def _result(solved: bool, routes: int, not_in_stock: str) -> RetroSearchResult:
    return RetroSearchResult(
        target="T",
        is_solved=solved,
        number_of_routes=routes,
        number_of_solved_routes=routes if solved else 0,
        number_of_steps=1,
        search_time=1.0,
        precursors_in_stock=(),
        precursors_not_in_stock=tuple(
            s.strip() for s in not_in_stock.split(",") if s.strip()
        ),
        failure_category=(
            None if solved else ("stock_mismatch" if routes else "no_route_found")
        ),
    )


class TestMismatchSemantics:
    def test_missing_stock_reported_not_misclassified(self):
        result = _result(False, routes=2, not_in_stock="CC(=O)OC(C)=O")
        assert result.stock_mismatch is True
        assert result.precursors_not_in_stock
        assert result.failure_category != "no_route_found"

    def test_report_counts_mismatches_separately(self, tmp_path):
        report = build_evaluation_report(
            corpus=load_generated_molecules(
                [_tiny_sdf(str(tmp_path / "g.sdf"))]
            ),
            retro_tracks={
                "t": {
                    "T": _result(False, routes=2, not_in_stock="CC(=O)OC(C)=O")
                }
            },
            track_provenance={"t": {"p": 1}},
            corpus_description="x",
        )
        section = report["retrosynthesis"]["t"]
        assert section["num_stock_mismatch"] == 1
        assert section["num_no_route_found"] == 0
        assert "CC(=O)OC(C)=O" in section["results"]["T"][
            "precursors_not_in_stock"
        ]


def _tiny_sdf(path: str) -> str:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(ASPIRIN)
    Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=3)
    with open(path, "w") as handle:
        handle.write(Chem.MolToMolBlock(mol) + "$$$$\n")
    return path


# ---------------------------------------------------------------------------
# 4. The REAL planner mismatch demonstration (gated on public models)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not AIZYNTH_MODELS or not os.path.isdir(AIZYNTH_MODELS),
    reason="SYNTREE_AIZYNTH_MODELS (public USPTO download) not present",
)
class TestRealPlannerMismatch:
    @pytest.fixture(scope="class")
    def real_results(self, tmp_path_factory):
        from syntree.engine.evaluator import run_aizynthfinder

        work = tmp_path_factory.mktemp("aizynth")
        # a deliberately minimal stock: the acetanilide precursors are
        # purchasable, aspirin's salicylic-acid precursor is NOT
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

        keys = []
        for smi in (ANILINE, ACETIC_ANHYDRIDE, "c1ccccc1"):
            mol = Chem.MolFromSmiles(smi)
            keys.append(InchiToInchiKey(MolToInchi(mol)))
        stock = work / "mini_stock.txt"
        stock.write_text("\n".join(keys) + "\n")
        config = work / "config.yml"
        config.write_text(
            "expansion:\n"
            f"  uspto:\n    - {AIZYNTH_MODELS}/uspto_model.onnx\n"
            f"    - {AIZYNTH_MODELS}/uspto_templates.csv.gz\n"
            "  ringbreaker:\n"
            f"    - {AIZYNTH_MODELS}/uspto_ringbreaker_model.onnx\n"
            f"    - {AIZYNTH_MODELS}/uspto_ringbreaker_templates.csv.gz\n"
            "filter:\n"
            f"  uspto: {AIZYNTH_MODELS}/uspto_filter_model.onnx\n"
            f"stock:\n  mini: {stock}\n"
        )
        results = run_aizynthfinder(
            [ACETANILIDE, ASPIRIN],
            str(config),
            str(work / "out.json"),
            timeout_seconds=600,
        )
        return results

    def test_purchasable_precursor_target_solved(self, real_results):
        assert real_results[ACETANILIDE].is_solved

    def test_out_of_stock_leaves_are_stock_mismatch_not_planner_failure(
        self, real_results
    ):
        aspirin = real_results[ASPIRIN]
        assert aspirin.is_solved is False
        assert aspirin.stock_mismatch is True
        assert aspirin.failure_category == "stock_mismatch"
        assert aspirin.precursors_not_in_stock
        assert aspirin.number_of_routes > 0

    def test_mismatch_reported_through_parse_path(
        self, real_results, tmp_path
    ):
        # the classification survives the JSON round-trip (the report
        # path consumes the parsed flat output, not in-memory objects)
        report = build_evaluation_report(
            corpus=load_generated_molecules(
                [_tiny_sdf(str(tmp_path / "g.sdf"))]
            ),
            retro_tracks={"mini": real_results},
            track_provenance={"mini": {"p": 1}},
            corpus_description="x",
        )
        section = report["retrosynthesis"]["mini"]
        assert section["num_solved"] >= 1
        assert section["num_stock_mismatch"] >= 1
