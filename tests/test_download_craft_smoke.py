"""Smoke tests for Download Mode and Craft Mode entry points.

Heavy dependencies (torch/rdkit/pandas) gate the corresponding tests via
``pytest.importorskip`` so the suite still collects on lightweight hosts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_craft():
    """Import craft.craft_mode, skipping when heavy deps are unavailable."""
    for mod in ("torch", "torch_geometric", "rdkit", "pandas", "tqdm"):
        pytest.importorskip(mod)
    import craft.craft_mode as m

    return m


# ---------------------------------------------------------------------------
# Download Mode
# ---------------------------------------------------------------------------

class TestDownloadMode:
    def test_module_compiles_and_sources_pinned(self):
        pytest.importorskip("huggingface_hub")
        from download import download_mode

        assert "crossdocked" in download_mode.RAW_SOURCES
        assert "enamine_catalog" in download_mode.RAW_SOURCES
        pin = download_mode.RAW_SOURCES["crossdocked"]["expected_sha256"]
        assert len(pin) in (32, 64)  # MD5- or SHA-256-length pin

    def test_manifest_written_with_hash_algo(self, tmp_path: Path):
        pytest.importorskip("huggingface_hub")
        from download.download_mode import RAW_SOURCES, RawDatasetAcquisition

        # Build the object without touching the Hub (offline unit test).
        acq = RawDatasetAcquisition.__new__(RawDatasetAcquisition)
        acq.workspace_dir = tmp_path
        acq.hf_repo_id = "local/test"
        acq.hf_token = ""
        acq.api = None

        for key, cfg in RAW_SOURCES.items():
            (tmp_path / cfg["expected_filename"]).write_bytes(key.encode())

        manifest_path = acq.generate_raw_manifest(RAW_SOURCES)
        data = json.loads(manifest_path.read_text())
        assert data["dataset_type"] == "3d_syntree_raw_unprocessed"
        assert set(data["files"]) == {"crossdocked", "enamine_catalog"}
        assert data["files"]["crossdocked"]["hash_algo"] == "md5"
        assert data["files"]["enamine_catalog"]["hash_algo"] == "sha256"
        assert data["files"]["crossdocked"]["source_url"]

    def test_manifest_refuses_missing_file(self, tmp_path: Path):
        pytest.importorskip("huggingface_hub")
        from download.download_mode import RawDatasetAcquisition

        acq = RawDatasetAcquisition.__new__(RawDatasetAcquisition)
        acq.workspace_dir = tmp_path
        with pytest.raises(FileNotFoundError):
            acq.generate_raw_manifest(
                {"crossdocked": {"expected_filename": "nope.tar.gz", "url": "u", "expected_sha256": None}}
            )


# ---------------------------------------------------------------------------
# Craft Mode
# ---------------------------------------------------------------------------

class TestCraftModePureLogic:
    """Discovery/split helpers that do not require the full chemistry stack."""

    def _pairs(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        pairs = []
        for i in range(3):
            d = root / f"protein{i}"
            d.mkdir()
            (d / f"complex{i}_pocket10.pdb").write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            )
            (d / f"complex{i}_ligand.sdf").write_text("")
            pairs.append(i)
        return pairs

    def test_scan_pairs_and_ligand_matching(self, tmp_path: Path):
        m = _import_craft()
        _best_ligand, _normalize_stem, scan_pairs = (
            m._best_ligand, m._normalize_stem, m.scan_pairs
        )

        self._pairs(tmp_path / "extracted")
        pairs = scan_pairs(tmp_path / "extracted")
        assert len(pairs) == 3
        for pocket, ligand, cid in pairs:
            assert "pocket" in Path(pocket).name
            assert ligand.endswith("_ligand.sdf")
            assert cid.startswith("protein")
        # Deterministic ordering by complex id.
        assert [c for _, _, c in pairs] == sorted(c for _, _, c in pairs)

        assert _normalize_stem("abc_pocket10") == "abc"
        assert _normalize_stem("abc_ligand") == "abc"
        assert _best_ligand("abc_pocket10", ["zzz_ligand.sdf", "abc_ligand.sdf"]) == "abc_ligand.sdf"
        # Longest shared prefix wins when no exact match exists.
        assert _best_ligand("abc_pocket10", ["abx_ligand.sdf", "abc_lig.sdf"]) == "abc_lig.sdf"

    def test_resolution_parser(self, tmp_path: Path):
        _resolution_from_pdb = _import_craft()._resolution_from_pdb

        pdb = tmp_path / "x.pdb"
        pdb.write_text("HEADER    TEST\nREMARK 2 RESOLUTION.    1.90 ANGSTROMS.\n")
        assert _resolution_from_pdb(str(pdb)) == pytest.approx(1.90)
        pdb.write_text("HEADER    NO RESOLUTION REMARK\n")
        assert _resolution_from_pdb(str(pdb)) is None

    def test_stable_split_buckets(self):
        m = _import_craft()
        SPLITS, _stable_cid_bucket = m.SPLITS, m._stable_cid_bucket

        assert _stable_cid_bucket("c1", 42) == _stable_cid_bucket("c1", 42)
        assert _stable_cid_bucket("c1", 42) in SPLITS
        assert _stable_cid_bucket("c1", 42) != _stable_cid_bucket("c2", 42)


class TestCraftSmoke:
    def test_verification_smoke_test_passes(self, tmp_path: Path, request):
        """Catalog -> synthetic dataset -> policy forward/backward (prod path)."""
        run_verification_smoke_test = _import_craft().run_verification_smoke_test

        # Deferred: assets_dir imports pandas, which must be skip-checked first.
        assets_dir = request.getfixturevalue("assets_dir")
        run_verification_smoke_test(
            Path(assets_dir["catalog_path"]), tmp_path / "out", min_catalog=1
        )
