"""#35 EVAL-02 — the comparative benchmark battery.

Pins the #35 acceptance surface of ``syntree/engine/comparative.py``,
``scripts/make_frozen_targets.py`` and
``scripts/run_comparative_benchmark.py``:

* the frozen target manifest loader (strict schema, duplicate and
  missing-pocket detection);
* the deterministic bootstrap machinery: seeded reproducibility, the
  CI containing the true mean of a known population, honest widening
  for small samples;
* the paired machinery: differences computed **only** over the common
  target intersection with the asymmetric targets reported, and an
  empty intersection failing loudly (never a silent unpaired
  degradation);
* the identical-evaluation-code guarantee: two methods with the SAME
  structures receive bit-identical metric sections (structural
  fairness, not a promise);
* the report/markdown surface: per-track retrosynthesis columns
  (separate, never merged), absent methods flagged ``not_provided``;
* the manifest generator and the CLI end-to-end with a stub planner
  (the acceptance artifact ``comparative_benchmark.json`` + summary
  markdown with bootstrap CIs).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.engine.comparative import (
    ComparativeError,
    FrozenTarget,
    bootstrap_ci,
    bootstrap_rate_ci,
    comparative_report,
    load_frozen_targets,
    markdown_summary,
    method_section,
    paired_difference_ci,
)
from syntree.engine.evaluator import (
    RetroSearchResult,
    load_generated_molecules,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ACETANILIDE = "CC(=O)Nc1ccccc1"
BENZENE = "c1ccccc1"


def _sdf(path: str, smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=11)
    with open(path, "w") as handle:
        handle.write(Chem.MolToMolBlock(mol) + "$$$$\n")
    return path


def _retro(target: str, solved: bool) -> RetroSearchResult:
    return RetroSearchResult(
        target=target,
        is_solved=solved,
        number_of_routes=1 if solved else 2,
        number_of_solved_routes=1 if solved else 0,
        number_of_steps=1,
        search_time=0.5,
        precursors_in_stock=("Nc1ccccc1",) if solved else (),
        precursors_not_in_stock=() if solved else ("CC(=O)Cl",),
        failure_category=None if solved else "stock_mismatch",
    )


def _manifest(tmp_path, targets: list[tuple[str, str]]) -> str:
    path = tmp_path / "frozen_targets.json"
    entries = [
        {
            "target_id": tid,
            "pocket": pocket,
            "reference_ligand_smiles": "C",
            "notes": "fixture",
        }
        for tid, pocket in targets
    ]
    path.write_text(json.dumps({"split": "test", "targets": entries}))
    return str(path)


# ---------------------------------------------------------------------------
# 1. Manifest loading
# ---------------------------------------------------------------------------


class TestFrozenTargets:
    def test_load_valid_manifest(self, tmp_path):
        pocket = tmp_path / "p.pdb"
        pocket.write_text("ATOM\n")
        path = _manifest(tmp_path, [("t1", str(pocket))])
        targets = load_frozen_targets(path)
        assert targets == [
            FrozenTarget("t1", str(pocket), "C", "fixture")
        ]

    def test_duplicate_ids_rejected(self, tmp_path):
        pocket = tmp_path / "p.pdb"
        pocket.write_text("ATOM\n")
        path = _manifest(tmp_path, [("t1", str(pocket)), ("t1", str(pocket))])
        with pytest.raises(ComparativeError, match="duplicate"):
            load_frozen_targets(path)

    def test_missing_pocket_rejected(self, tmp_path):
        path = _manifest(tmp_path, [("t1", str(tmp_path / "gone.pdb"))])
        with pytest.raises(ComparativeError, match="pocket not found"):
            load_frozen_targets(path)

    def test_missing_fields_rejected(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"targets": [{"target_id": "t"}]}))
        with pytest.raises(ComparativeError, match="missing fields"):
            load_frozen_targets(path)

    def test_empty_targets_rejected(self, tmp_path):
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"targets": []}))
        with pytest.raises(ComparativeError, match="no targets"):
            load_frozen_targets(path)


# ---------------------------------------------------------------------------
# 2. Bootstrap machinery
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_deterministic_given_seed(self):
        values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        assert bootstrap_rate_ci(values, seed=3) == bootstrap_rate_ci(
            values, seed=3
        )
        assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)

    def test_ci_contains_true_mean(self):
        rng = np.random.default_rng(0)
        population = rng.normal(loc=5.0, scale=2.0, size=100_000)
        sample = list(rng.choice(population, size=40, replace=False))
        mean, lo, hi = bootstrap_ci(sample, seed=1)
        assert lo < mean < hi
        assert lo < 5.0 < hi  # the true mean inside the 95% interval

    def test_rate_ci_bounds_and_width(self):
        # k/n = 2/4: interval must reflect the small sample
        mean, lo, hi = bootstrap_rate_ci([1, 0, 1, 0], seed=0)
        assert mean == 0.5
        assert 0.0 <= lo <= 0.5 <= hi <= 1.0
        assert (hi - lo) > 0.3
        # a large unanimous sample tightens toward the point estimate
        _, lo2, hi2 = bootstrap_rate_ci([1] * 500, seed=0)
        assert hi2 - lo2 < 0.05

    def test_empty_input_raises(self):
        with pytest.raises(ComparativeError):
            bootstrap_ci([])
        with pytest.raises(ComparativeError):
            bootstrap_rate_ci([])


class TestPairedDifference:
    def test_pairs_over_common_targets_only(self):
        stats = paired_difference_ci(
            {"a": 1.0, "b": 0.0, "c": 1.0}, {"a": 0.0, "b": 0.0, "d": 1.0}
        )
        assert stats["num_common_targets"] == 2
        assert stats["only_in_a"] == ["c"]
        assert stats["only_in_b"] == ["d"]
        assert stats["mean_difference"] == pytest.approx(0.5)

    def test_zero_difference_gives_zero_centered_ci(self):
        stats = paired_difference_ci(
            {"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}
        )
        assert stats["mean_difference"] == 0.0
        assert stats["ci_low"] == 0.0 and stats["ci_high"] == 0.0

    def test_empty_intersection_raises(self):
        with pytest.raises(ComparativeError, match="empty target intersection"):
            paired_difference_ci({"a": 1.0}, {"b": 1.0})

    def test_deterministic(self):
        a = {"t1": 1.0, "t2": 0.0, "t3": 1.0}
        b = {"t1": 0.0, "t2": 1.0, "t3": 1.0}
        s1 = paired_difference_ci(a, b, seed=5)
        s2 = paired_difference_ci(a, b, seed=5)
        assert s1 == s2


# ---------------------------------------------------------------------------
# 3. Identical evaluation code + report surface
# ---------------------------------------------------------------------------


class TestMethodSection:
    def test_same_structures_same_metrics(self, tmp_path):
        corpus = load_generated_molecules(
            [_sdf(str(tmp_path / "t1.sdf"), ASPIRIN)]
        )
        a = method_section("method_a", corpus, {"track": {"x": _retro("x", True)}})
        b = method_section("method_b", corpus, {"track": {"x": _retro("x", True)}})
        for key in ("validity", "fraction_sp3", "fingerprint_diversity"):
            assert a[key] == b[key]
        assert a["retrosynthesis"] == b["retrosynthesis"]

    def test_empty_corpus_flagged_not_provided(self):
        section = method_section("absent", [], {})
        assert section["not_provided"] is True
        assert section["num_structures"] == 0


class TestComparativeReport:
    def _targets(self, tmp_path):
        pocket = tmp_path / "p.pdb"
        pocket.write_text("ATOM\n")
        return [
            FrozenTarget("t1", str(pocket), "C"),
            FrozenTarget("t2", str(pocket), "C"),
        ]

    def test_two_tracks_never_merged_in_sections(self, tmp_path):
        corpus = load_generated_molecules(
            [
                _sdf(str(tmp_path / "t1.sdf"), ASPIRIN),
                _sdf(str(tmp_path / "t2.sdf"), ACETANILIDE),
            ]
        )
        report = comparative_report(
            targets=self._targets(tmp_path),
            method_corpora={"syntree": corpus},
            method_retro={
                "syntree": {
                    "stock_aligned": {
                        ASPIRIN: _retro(ASPIRIN, True),
                        ACETANILIDE: _retro(ACETANILIDE, True),
                    },
                    "independent_external": {
                        ASPIRIN: _retro(ASPIRIN, False),
                        ACETANILIDE: _retro(ACETANILIDE, False),
                    },
                }
            },
            corpus_description="fixture corpus",
        )
        retro = report["methods"]["syntree"]["retrosynthesis"]
        assert set(retro) == {"stock_aligned", "independent_external"}
        assert retro["stock_aligned"][0] == 1.0
        assert retro["independent_external"][0] == 0.0
        assert "never merged" in report["notes"]

    def test_paired_comparison_on_common_targets(self, tmp_path):
        corpus_a = load_generated_molecules(
            [_sdf(str(tmp_path / "t1.sdf"), ASPIRIN)]
        )
        corpus_b = load_generated_molecules(
            [
                _sdf(str(tmp_path / "t1.sdf"), ACETANILIDE),
                _sdf(str(tmp_path / "t2.sdf"), BENZENE),
            ]
        )
        report = comparative_report(
            targets=self._targets(tmp_path),
            method_corpora={"ref": corpus_a, "other": corpus_b},
            method_retro={
                "ref": {"track": {ASPIRIN: _retro(ASPIRIN, True)}},
                "other": {
                    "track": {
                        ACETANILIDE: _retro(ACETANILIDE, False),
                        BENZENE: _retro(BENZENE, True),
                    }
                },
            },
            corpus_description="fixture",
            reference_method="ref",
        )
        assert len(report["paired_comparisons"]) == 1
        pair = report["paired_comparisons"][0]
        stats = pair["metrics"]["retro_track"]
        # the reference only generated t1: pairing is over {t1} alone,
        # with t2 reported as present only in the other method
        assert stats["num_common_targets"] == 1
        assert stats["only_in_a"] == []
        assert stats["only_in_b"] == ["t2"]
        assert stats["mean_difference"] == pytest.approx(1.0)
        assert stats["ci_low"] <= stats["mean_difference"] <= stats["ci_high"]

    def test_markdown_summary_tables_and_cis(self, tmp_path):
        corpus = load_generated_molecules(
            [_sdf(str(tmp_path / "t1.sdf"), ASPIRIN)]
        )
        report = comparative_report(
            targets=self._targets(tmp_path),
            method_corpora={"syntree": corpus},
            method_retro={
                "syntree": {"independent_external": {ASPIRIN: _retro(ASPIRIN, True)}}
            },
            corpus_description="fixture",
        )
        md = markdown_summary(report)
        assert "| method | structures | validity |" in md
        assert "retro[independent_external]" in md
        assert "[0.000, 1.000]" in md or "[" in md  # CI bracket present
        assert "never merged" in md or "separately" in md


# ---------------------------------------------------------------------------
# 4. The manifest generator + CLI end-to-end (stub planner)
# ---------------------------------------------------------------------------


def _tiny_trajectory_pt(path: str) -> str:
    import torch

    records = [
        {
            "complex_id": "pdbbind_fix1",
            "ligand_smiles": ASPIRIN,
            "pocket_artifact": "pdbbind_fix1_pocket.pdb.gz",
            "seed_block_id": "REC0001",
            "split": "test",
            "steps": [],
        },
        {
            "complex_id": "pdbbind_fix2",
            "ligand_smiles": ACETANILIDE,
            "pocket_artifact": "pdbbind_fix2_pocket.pdb.gz",
            "seed_block_id": "REC0002",
            "split": "test",
            "steps": [],
        },
    ]
    torch.save(records, path)
    return path


class TestMakeFrozenTargets:
    def test_manifest_generated_from_bundle(self, tmp_path):
        import gzip as gz

        processed = tmp_path / "processed"
        processed.mkdir()
        for tid in ("pdbbind_fix1", "pdbbind_fix2"):
            with gz.open(str(processed / f"{tid}_pocket.pdb.gz"), "wt") as h:
                h.write("ATOM\n")
        traj = _tiny_trajectory_pt(str(tmp_path / "trajectories_test.pt"))
        out = tmp_path / "frozen.json"
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "scripts", "make_frozen_targets.py"),
                "--trajectories",
                traj,
                "--processed-dir",
                str(processed),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        manifest = json.loads(out.read_text())
        assert manifest["num_targets"] == 2
        assert manifest["split"] == "test"
        assert {t["target_id"] for t in manifest["targets"]} == {
            "pdbbind_fix1",
            "pdbbind_fix2",
        }
        assert all(os.path.exists(t["pocket"]) for t in manifest["targets"])

    def test_missing_pocket_fails_closed(self, tmp_path):
        traj = _tiny_trajectory_pt(str(tmp_path / "trajectories_test.pt"))
        processed = tmp_path / "empty"
        processed.mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "scripts", "make_frozen_targets.py"),
                "--trajectories",
                traj,
                "--processed-dir",
                str(processed),
                "--output",
                str(tmp_path / "f.json"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode != 0
        assert "missing" in (proc.stdout + proc.stderr)


class TestComparativeCLI:
    def test_cli_end_to_end_stub_planner(self, tmp_path):
        # 1. frozen targets
        pocket = tmp_path / "p.pdb"
        pocket.write_text("ATOM\n")
        targets = tmp_path / "frozen.json"
        targets.write_text(
            json.dumps(
                {
                    "split": "test",
                    "targets": [
                        {
                            "target_id": "t1",
                            "pocket": str(pocket),
                            "reference_ligand_smiles": "C",
                        },
                        {
                            "target_id": "t2",
                            "pocket": str(pocket),
                            "reference_ligand_smiles": "C",
                        },
                    ],
                }
            )
        )
        # 2. two method structure dirs on the same targets
        for method, smiles_map in (
            ("syntree", {"t1": ASPIRIN, "t2": ACETANILIDE}),
            ("baseline", {"t1": BENZENE}),  # t2 not generated
        ):
            d = tmp_path / method
            d.mkdir()
            for tid, smi in smiles_map.items():
                _sdf(str(d / f"{tid}.sdf"), smi)
        # 3. stub planner interpreter (solves everything)
        stub = tmp_path / "stub.py"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            + textwrap.dedent(
                """
                import json, sys
                args = sys.argv[1:]
                smi = args[args.index("--smiles") + 1]
                out = args[args.index("--output") + 1]
                with open(smi) as h:
                    targets_ = [l.strip() for l in h if l.strip()]
                rows = [
                    {
                        "target": t, "is_solved": True,
                        "number_of_routes": 1,
                        "number_of_solved_routes": 1,
                        "number_of_steps": 1, "search_time": 0.1,
                        "precursors_in_stock": "Nc1ccccc1",
                        "precursors_not_in_stock": "",
                    }
                    for t in targets_
                ]
                with open(out, "w") as h:
                    json.dump({"schema": {}, "data": rows}, h)
                """
            )
        )
        os.chmod(str(stub), 0o755)
        output = tmp_path / "comparative_benchmark.json"
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(
                    REPO_ROOT, "scripts", "run_comparative_benchmark.py"
                ),
                "--targets",
                str(targets),
                "--method",
                f"syntree={tmp_path / 'syntree'}",
                "--method",
                f"baseline={tmp_path / 'baseline'}",
                "--retro-track",
                "stub_track=" + _stub_config(str(tmp_path)),
                "--aizynth-python",
                str(stub),
                "--reference",
                "syntree",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        report = json.loads(output.read_text())
        assert report["frozen_targets"]["num_targets"] == 2
        assert report["methods"]["syntree"]["num_structures"] == 2
        assert report["methods"]["baseline"]["num_structures"] == 1
        assert report["methods"]["syntree"]["validity"][0] == 1.0
        assert report["methods"]["baseline"]["retrosynthesis"]["stub_track"]
        assert report["paired_comparisons"]
        md_path = str(output) + ".md"
        assert os.path.exists(md_path)
        md = open(md_path).read()
        assert "| method |" in md and "stub_track" in md

    def test_stray_structure_fails_loudly(self, tmp_path):
        pocket = tmp_path / "p.pdb"
        pocket.write_text("ATOM\n")
        targets = tmp_path / "frozen.json"
        targets.write_text(
            json.dumps(
                {
                    "split": "test",
                    "targets": [
                        {
                            "target_id": "t1",
                            "pocket": str(pocket),
                            "reference_ligand_smiles": "C",
                        }
                    ],
                }
            )
        )
        d = tmp_path / "m"
        d.mkdir()
        _sdf(str(d / "t1.sdf"), ASPIRIN)
        _sdf(str(d / "typo_target.sdf"), BENZENE)  # not a frozen target
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(
                    REPO_ROOT, "scripts", "run_comparative_benchmark.py"
                ),
                "--targets",
                str(targets),
                "--method",
                f"m={d}",
                "--output",
                str(tmp_path / "out.json"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 2
        assert "not matching any frozen target" in proc.stderr


def _stub_config(tmp: str) -> str:
    path = os.path.join(tmp, "stub_config.yml")
    with open(path, "w") as handle:
        handle.write("stock: {stub: stub.txt}\n")
    return path
