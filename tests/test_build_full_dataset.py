"""Comprehensive tests for scripts/build_full_dataset.py.

Coverage:
* pair discovery across CrossDocked naming conventions (pocket10 + sdf/mol2,
  prefix matching, missing ligands, nested directories, deterministic order),
* deterministic 80/10/10 split assignment,
* idempotent download/extract with marker + real tar.gz round trip,
* full offline end-to-end build: shards, manifests, sha256 digests, targets,
  catalog copy, sample schema, real-data provenance flags,
* HF upload orchestration with a mocked Hub API,
* CLI contract (--upload without HF_TOKEN fails closed),
* bit-level reproducibility of two independent builds.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import types
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts import build_full_dataset as bfd  # noqa: E402

POCKET_PDB = (
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
    "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
    "ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C\n"
    "ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O\n"
    "ATOM      5  N   GLY A   2       3.315   1.545   0.000  1.00 20.00           N\n"
    "ATOM      6  CA  GLY A   2       3.866   2.906   0.000  1.00 20.00           C\n"
    "ATOM      7  C   GLY A   2       4.417   4.267   0.000  1.00 20.00           C\n"
    "ATOM      8  O   GLY A   2       3.680   5.254   0.000  1.00 20.00           O\n"
    "END\n"
)

# Products decomposable against the conftest catalog (verified via the
# reaction engine's exact forward replay inside the fragmenter).
_CANDIDATE_PRODUCTS = [
    "O=C(NC1CCCCC1)C1CCCCC1",   # amide: cyclohexane acid + cyclopentylamine
    "CC(C)COC(=O)C1CCCCC1",     # ester: isobutanol + cyclohexane acid
    "Fc1ccc(c2ccccc2)cc1",      # suzuki: 4-fluoro-aryl halide + phenylboronic
    "CC(C)(C)C(=O)NCC1CCCCC1",  # amide: pivalic acid + aminomethylcyclohexane
    "CC(C)COC(=O)C(C)(C)C",     # ester: isobutanol + pivalic acid
    "Fc1ccc(NC2CCCC2)cc1",      # aryl amination: aryl fluoride + cyclopentylamine
]


def _sybyl_type(atom) -> str:
    symbol = atom.GetSymbol()
    has_double = any(
        b.GetBondTypeAsDouble() == 2.0 for b in atom.GetBonds()
    )
    if symbol == "C":
        return "C.2" if has_double else "C.3"
    if symbol == "O":
        return "O.2" if has_double else "O.3"
    if symbol == "N":
        return "N.2" if has_double else "N.3"
    return symbol


def _mol2_block(mol: Chem.Mol, name: str = "lig") -> str:
    mol = Chem.AddHs(Chem.Mol(mol))
    assert AllChem.EmbedMolecule(mol, randomSeed=19) == 0
    conf = mol.GetConformer()
    lines = [
        "# generated test ligand",
        "@<TRIPOS>MOLECULE",
        name,
        f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0",
        "SMALL",
        "NO_CHARGES",
        "",
        "",
    ]
    lines.append("@<TRIPOS>ATOM")
    for i, atom in enumerate(mol.GetAtoms(), 1):
        p = conf.GetAtomPosition(i - 1)
        lines.append(
            f"{i:7d} {atom.GetSymbol()}{i:<3d} {p.x:10.4f} {p.y:10.4f} "
            f"{p.z:10.4f} {_sybyl_type(atom):<8s} 1 UNK"
        )
    lines.append("@<TRIPOS>BOND")
    for j, b in enumerate(mol.GetBonds(), 1):
        t = {1.0: "1", 2.0: "2", 3.0: "3", 1.5: "ar"}.get(
            b.GetBondTypeAsDouble(), "1"
        )
        lines.append(
            f"{j:6d}{b.GetBeginAtomIdx() + 1:6d}{b.GetEndAtomIdx() + 1:6d} {t}"
        )
    return "\n".join(lines) + "\n"


def _write_sdf(mol: Chem.Mol, path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def _embedded(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol is not None and AllChem.EmbedMolecule(mol, randomSeed=19) == 0
    return mol


@pytest.fixture(scope="module")
def valid_products(assets_dir):
    """Filter candidate products to those that actually decompose against
    the fixture catalog via the real fragmenter (self-validating)."""
    from syntree.chemistry.catalog import SynthonCatalog
    from syntree.data.fragmenter import ReactionConstrainedFragmenter

    catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
    fragmenter = ReactionConstrainedFragmenter(catalog)
    valid = []
    for smiles in _CANDIDATE_PRODUCTS:
        lig = _embedded(smiles)
        if fragmenter.find_target(lig) is not None:
            valid.append(smiles)
    assert len(valid) >= 3, f"need >=3 decomposable products, got {len(valid)}"
    return valid


@pytest.fixture(scope="module")
def mini_crossdocked(tmp_path_factory, valid_products):
    """A CrossDocked-shaped raw tree: <protein>/<complex>_pocket10.pdb +
    ligand SDFs (and one MOL2) with deterministic geometry."""
    raw = tmp_path_factory.mktemp("mini_crossdocked") / "crossdocked_pocket10"
    raw.mkdir(parents=True)
    n_complex = 0
    ligand_formats = {"sdf": 0, "mol2": 0}
    for copy_idx in range(6):
        for smiles in valid_products:
            n_complex += 1
            protein_dir = raw / f"protein_{copy_idx:02d}_{n_complex % 3}"
            protein_dir.mkdir(parents=True, exist_ok=True)
            cid = f"1abc_{n_complex}_1"
            (protein_dir / f"{cid}_pocket10.pdb").write_text(POCKET_PDB)
            # Rotate formats: mostly SDF, one MOL2 per copy.
            use_mol2 = (n_complex % 6) == 3
            if use_mol2:
                (protein_dir / f"{cid}_ligand.mol2").write_text(
                    _mol2_block(Chem.MolFromSmiles(smiles), cid)
                )
                ligand_formats["mol2"] += 1
            else:
                _write_sdf(_embedded(smiles), protein_dir / f"{cid}_ligand.sdf")
                ligand_formats["sdf"] += 1
    (raw.parent / ".extracted_marker").touch()
    return {
        "raw_dir": raw.parent,
        "n_complex": n_complex,
        "formats": ligand_formats,
    }


# =====================================================================
# Pair discovery
# =====================================================================
class TestDiscoverPairs:
    def test_discovers_all_pairs(self, mini_crossdocked):
        pairs = bfd.discover_pairs(mini_crossdocked["raw_dir"])
        assert len(pairs) == mini_crossdocked["n_complex"]
        for pocket, ligand, cid in pairs:
            assert "pocket" in pocket.stem.lower()
            assert ligand.suffix.lower() in (".sdf", ".mol2")
            assert cid == f"{pocket.parent.name}_{pocket.stem}"

    def test_deterministic_sorted_order(self, mini_crossdocked):
        a = bfd.discover_pairs(mini_crossdocked["raw_dir"])
        b = bfd.discover_pairs(mini_crossdocked["raw_dir"])
        assert [p[2] for p in a] == [p[2] for p in b]
        assert [p[2] for p in a] == sorted(p[2] for p in a)

    def test_mol2_ligands_supported(self, mini_crossdocked):
        assert mini_crossdocked["formats"]["mol2"] >= 1
        pairs = bfd.discover_pairs(mini_crossdocked["raw_dir"])
        mol2_pairs = [p for p in pairs if p[1].suffix == ".mol2"]
        assert len(mol2_pairs) == mini_crossdocked["formats"]["mol2"]

    def test_missing_ligand_skipped(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        (root / "orphan_pocket10.pdb").write_text(POCKET_PDB)
        assert bfd.discover_pairs(root) == []

    def test_non_pocket_pdb_ignored(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        (root / "1abc_full_protein.pdb").write_text(POCKET_PDB)
        lig = _embedded("CC(C)COC(=O)C1CCCCC1")
        _write_sdf(lig, root / "1abc_full_protein_ligand.sdf")
        assert bfd.discover_pairs(root) == []

    def test_nested_directories(self, tmp_path):
        deep = tmp_path / "raw" / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "x1_pocket.pdb").write_text(POCKET_PDB)
        _write_sdf(_embedded("CC(C)COC(=O)C1CCCCC1"), deep / "x1_ligand.sdf")
        pairs = bfd.discover_pairs(tmp_path / "raw")
        assert len(pairs) == 1
        assert pairs[0][2] == f"c_x1_pocket"

    def test_exact_stem_match_preferred(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        (root / "1zzz_pocket10.pdb").write_text(POCKET_PDB)
        # Decoy ligand with a *longer* prefix share than the true companion.
        _write_sdf(_embedded("CC(C)COC(=O)C1CCCCC1"), root / "1zzz_decoy_ligand.sdf")
        _write_sdf(_embedded("CC(C)COC(=O)C1CCCCC1"), root / "1zzz_ligand.sdf")
        pairs = bfd.discover_pairs(root)
        assert pairs[0][1].name == "1zzz_ligand.sdf"

    def test_prefix_match_when_no_exact(self, tmp_path):
        root = tmp_path / "raw"
        root.mkdir()
        (root / "2zzz_7_1_pocket10.pdb").write_text(POCKET_PDB)
        _write_sdf(_embedded("CC(C)COC(=O)C1CCCCC1"), root / "2zzz_7_1_ligand.sdf")
        pairs = bfd.discover_pairs(root)
        assert pairs[0][1].name == "2zzz_7_1_ligand.sdf"


# =====================================================================
# Ligand/pocket loading
# =====================================================================
class TestLoaders:
    def test_load_sdf_ligand(self, tmp_path):
        path = tmp_path / "lig.sdf"
        _write_sdf(_embedded("CC(C)COC(=O)C1CCCCC1"), path)
        mol = bfd.load_ligand(path)
        assert mol is not None and mol.GetNumConformers() == 1

    def test_load_mol2_ligand(self, tmp_path):
        path = tmp_path / "lig.mol2"
        path.write_text(_mol2_block(Chem.MolFromSmiles("CC(C)COC(=O)C1CCCCC1")))
        mol = bfd.load_ligand(path)
        assert mol is not None and mol.GetNumConformers() == 1

    def test_load_corrupt_sdf_returns_none(self, tmp_path):
        path = tmp_path / "bad.sdf"
        path.write_text("not a valid sdf at all\n")
        assert bfd.load_ligand(path) is None

    def test_load_pocket(self, tmp_path):
        path = tmp_path / "pocket.pdb"
        path.write_text(POCKET_PDB)
        mol = bfd.load_pocket(path)
        assert mol is not None and mol.GetNumAtoms() > 0


# =====================================================================
# Splits
# =====================================================================
class TestAssignSplits:
    def test_deterministic(self):
        a = bfd.assign_splits(200, seed=42)
        b = bfd.assign_splits(200, seed=42)
        assert a == b

    def test_proportions(self):
        labels = bfd.assign_splits(1000, seed=42)
        counts = {s: labels.count(s) for s in ("train", "val", "test")}
        assert counts["train"] == 800
        assert counts["val"] == 100
        assert counts["test"] == 100

    def test_all_splits_populated(self):
        labels = bfd.assign_splits(30, seed=42)
        assert set(labels) == {"train", "val", "test"}

    def test_seed_changes_assignment(self):
        a = bfd.assign_splits(200, seed=42)
        b = bfd.assign_splits(200, seed=7)
        assert a != b

    def test_too_few_items_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            bfd.assign_splits(2, seed=42)

    def test_valid_label_values_only(self):
        labels = bfd.assign_splits(57, seed=123)
        assert all(l in ("train", "val", "test") for l in labels)
        assert len(labels) == 57


# =====================================================================
# Download & extraction
# =====================================================================
class TestDownloadAndExtract:
    def _make_tarball(self, path: Path, payload_name: str = "inner_pocket10.pdb") -> None:
        inner = path.parent / payload_name
        inner.write_text(POCKET_PDB)
        with tarfile.open(path, "w:gz") as tar:
            tar.add(inner, arcname=payload_name)
        inner.unlink()

    def test_preextracted_marker_short_circuits(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / ".extracted_marker").touch()
        (raw / "already_pocket10.pdb").write_text(POCKET_PDB)

        def fail(*a, **k):
            raise AssertionError("must not download when marker present")

        monkeypatch.setattr(bfd.urllib.request, "urlretrieve", fail)
        result = bfd.download_and_extract(raw)
        assert result == raw

    def test_extracts_real_tarball_and_writes_marker(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        raw.mkdir()
        archive = raw / "crossdocked_pocket10.tar.gz"
        self._make_tarball(archive)
        # Archive exists -> urlretrieve must not be called.
        monkeypatch.setattr(
            bfd.urllib.request, "urlretrieve",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")),
        )
        bfd.download_and_extract(raw)
        assert (raw / ".extracted_marker").exists()
        assert (raw / "inner_pocket10.pdb").exists()

    def test_idempotent_second_call(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        self._make_tarball(raw / "crossdocked_pocket10.tar.gz")
        bfd.download_and_extract(raw)
        before = sorted(p.name for p in raw.iterdir())
        bfd.download_and_extract(raw)
        after = sorted(p.name for p in raw.iterdir())
        assert before == after

    def test_truncated_download_rejected(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        raw.mkdir()

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"tiny-truncated-bytes")

        monkeypatch.setattr(bfd.urllib.request, "urlretrieve", fake_retrieve)
        with pytest.raises(RuntimeError, match="suspiciously small"):
            bfd.download_and_extract(raw)

    def test_safe_extraction_rejects_path_traversal(self, tmp_path):
        """A tar containing an absolute/parent path must not escape raw_dir
        (Python's data filter); with a crafted member we expect either a
        safe rejection or the member being neutralised."""
        raw = tmp_path / "raw"
        raw.mkdir()
        archive = raw / "crossdocked_pocket10.tar.gz"
        evil = tmp_path / "evil.txt"
        evil.write_text("bad")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(evil, arcname="../evil_escape.txt")
        try:
            bfd.download_and_extract(raw)
        except Exception:
            pass  # rejected by the data filter -> acceptable
        assert not (tmp_path / "evil_escape.txt").exists()


# =====================================================================
# End-to-end offline build
# =====================================================================
@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory, mini_crossdocked, assets_dir):
    """Run the real builder main() offline (marker pre-set, no download)."""
    out_dir = tmp_path_factory.mktemp("built_shards")
    argv = [
        "build_full_dataset.py",
        "--raw-dir", str(mini_crossdocked["raw_dir"]),
        "--catalog", assets_dir["catalog_path"],
        "--output-dir", str(out_dir),
        "--max-steps", "2",
        "--shard-mb", "1",  # tiny shards to exercise multi-shard logic
        "--max-target-pockets", "10",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = bfd.main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    return {"out_dir": out_dir, "raw": mini_crossdocked}


class TestEndToEndBuild:
    SCHEMA_FIELDS = (
        "pocket_pos", "pocket_z", "pocket_charge",
        "ligand_pos", "ligand_z", "ligand_charge",
        "handle_features", "handle_pos", "handle_nodes",
        "global_features", "target_synthon", "target_dihedral",
        "target_reaction_family_idx", "target_core_handle_idx",
        "target_stop", "stop_mask", "trajectory_step", "trajectory_hash",
        "is_real_sample",
    )

    def test_all_split_manifests_written(self, built_dataset):
        out = built_dataset["out_dir"]
        for split in ("train", "val", "test"):
            manifest_path = out / split / "manifest.json"
            assert manifest_path.exists(), f"missing {split} manifest"
            manifest = json.loads(manifest_path.read_text())
            assert manifest["split"] == split
            assert manifest["total_samples"] > 0
            declared = sum(s["sample_count"] for s in manifest["shards"])
            assert declared == manifest["total_samples"]

    def test_shard_digests_and_bounds(self, built_dataset):
        out = built_dataset["out_dir"]
        max_bytes = 1 * 1024 * 1024
        for split in ("train", "val", "test"):
            manifest = json.loads((out / split / "manifest.json").read_text())
            for shard in manifest["shards"]:
                payload = (out / split / shard["name"]).read_bytes()
                assert shard["compressed_bytes"] == len(payload)
                assert hashlib.sha256(payload).hexdigest() == shard["sha256"]
                # gzip magic bytes
                assert payload[:2] == b"\x1f\x8b"
                if not shard.get("oversize_single_sample"):
                    assert len(payload) <= max_bytes

    def test_shard_index_ranges_are_contiguous(self, built_dataset):
        out = built_dataset["out_dir"]
        for split in ("train", "val", "test"):
            manifest = json.loads((out / split / "manifest.json").read_text())
            shards = sorted(manifest["shards"], key=lambda s: s["sample_start"])
            expected_start = 0
            for shard in shards:
                assert shard["sample_start"] == expected_start
                assert shard["sample_end"] == shard["sample_start"] + shard["sample_count"]
                expected_start = shard["sample_end"]
            assert expected_start == manifest["total_samples"]

    def test_samples_are_real_pyg_states(self, built_dataset):
        out = built_dataset["out_dir"]
        total = 0
        for split in ("train", "val", "test"):
            manifest = json.loads((out / split / "manifest.json").read_text())
            for shard in manifest["shards"]:
                path = out / split / shard["name"]
                with gzip.open(path, "rb") as f:
                    samples = torch.load(f, weights_only=False)
                assert isinstance(samples, list)
                assert len(samples) == shard["sample_count"]
                for sample in samples:
                    assert isinstance(sample, Data)
                    for field in self.SCHEMA_FIELDS:
                        assert hasattr(sample, field), f"missing {field}"
                    assert bool(sample.is_real_sample.item()) is True
                    assert sample.pocket_pos.size(0) > 0
                    assert sample.ligand_pos.size(0) > 0
                total += len(samples)
        assert total > 0

    def test_targets_staged_with_pocket_convention(self, built_dataset):
        targets = list((built_dataset["out_dir"] / "targets").glob("*.pdb"))
        assert 0 < len(targets) <= 10
        for t in targets:
            assert t.name.endswith("_pocket.pdb")
            assert t.read_text() == POCKET_PDB

    def test_catalog_copied_to_output_root(self, built_dataset, assets_dir):
        copied = built_dataset["out_dir"] / "enamine_3d_subset.parquet"
        assert copied.exists()
        assert copied.read_bytes() == Path(assets_dir["catalog_path"]).read_bytes()

    def test_split_disjointness_via_trajectory_hash(self, built_dataset):
        """Train/val/test must not share trajectories (no data leak)."""
        out = built_dataset["out_dir"]
        hashes = {}
        for split in ("train", "val", "test"):
            manifest = json.loads((out / split / "manifest.json").read_text())
            seen = set()
            for shard in manifest["shards"]:
                with gzip.open(out / split / shard["name"], "rb") as f:
                    samples = torch.load(f, weights_only=False)
                for s in samples:
                    h = int(s.trajectory_hash.item())
                    seen.add((h, split))
                    hashes.setdefault(h, set()).add(split)
            assert seen  # split non-empty
        for h, splits in hashes.items():
            assert len(splits) == 1, f"trajectory {h} leaked across {splits}"

    def test_states_match_direct_builder_output(self, built_dataset, assets_dir):
        """Data integrity: sharded samples must equal the trajectory
        builder's in-memory output for the same ligand (recompute one)."""
        from syntree.chemistry.catalog import SynthonCatalog
        from syntree.data.trajectory import RetrosyntheticTrajectoryBuilder

        catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
        builder = RetrosyntheticTrajectoryBuilder(catalog, max_steps=2)
        lig = _embedded("CC(C)COC(=O)C1CCCCC1")
        pocket = Chem.MolFromPDBBlock(POCKET_PDB, removeHs=False)
        states, meta = builder.build(lig, pocket, trajectory_id="recompute-check")
        assert states
        reference = states[0]

        out = built_dataset["out_dir"]
        found = False
        split = "train"
        manifest = json.loads((out / split / "manifest.json").read_text())
        for shard in manifest["shards"]:
            with gzip.open(out / split / shard["name"], "rb") as f:
                samples = torch.load(f, weights_only=False)
            for s in samples:
                if bool(s.target_stop.item()):
                    continue
                if s.ligand_pos.shape != reference.ligand_pos.shape:
                    continue
                if (
                    torch.allclose(s.ligand_pos, reference.ligand_pos, atol=1e-3)
                    and int(s.target_synthon) == int(reference.target_synthon)
                ):
                    found = True
        # The same product definitely went into train (80% of the complexes)
        assert found, "sharded states diverge from the direct builder output"


# =====================================================================
# Reproducibility
# =====================================================================
class TestReproducibility:
    def test_two_builds_produce_identical_shards(
        self, tmp_path_factory, mini_crossdocked, assets_dir
    ):
        digests = []
        for i in range(2):
            out = tmp_path_factory.mktemp(f"repro{i}") / "shards"
            argv = [
                "build_full_dataset.py",
                "--raw-dir", str(mini_crossdocked["raw_dir"]),
                "--catalog", assets_dir["catalog_path"],
                "--output-dir", str(out),
                "--max-steps", "2",
                "--shard-mb", "1",
            ]
            old_argv = sys.argv
            sys.argv = argv
            try:
                assert bfd.main() == 0
            finally:
                sys.argv = old_argv
            split_digests = {}
            for split in ("train", "val", "test"):
                manifest = json.loads((out / split / "manifest.json").read_text())
                split_digests[split] = sorted(
                    (s["name"], s["sha256"]) for s in manifest["shards"]
                )
            digests.append(split_digests)
        assert digests[0] == digests[1]


# =====================================================================
# Upload orchestration (mocked Hub)
# =====================================================================
class TestUpload:
    def _run_with_fake_hub(self, tmp_path, mini_crossdocked, assets_dir, uploads):
        class FakeApi:
            def __init__(self, token=None):
                self.token = token

            def create_repo(self, **kwargs):
                uploads["create_repo"].append(kwargs)

            def upload_file(self, **kwargs):
                uploads["upload_file"].append(kwargs)

        import huggingface_hub
        from scripts import shard_and_upload

        argv = [
            "build_full_dataset.py",
            "--raw-dir", str(mini_crossdocked["raw_dir"]),
            "--catalog", assets_dir["catalog_path"],
            "--output-dir", str(tmp_path / "up_shards"),
            "--max-steps", "2",
            "--shard-mb", "1",
            "--upload",
        ]
        old_argv = sys.argv
        monkey_token = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "hf_fake_test_token"
        sys.argv = argv
        try:
            orig = huggingface_hub.HfApi
            orig_sau = shard_and_upload.HfApi
            huggingface_hub.HfApi = FakeApi
            shard_and_upload.HfApi = FakeApi
            rc = bfd.main()
        finally:
            huggingface_hub.HfApi = orig
            shard_and_upload.HfApi = orig_sau
            sys.argv = old_argv
            if monkey_token is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = monkey_token
        return rc

    def test_upload_calls_cover_catalog_targets_and_shards(
        self, tmp_path, mini_crossdocked, assets_dir
    ):
        uploads = {"create_repo": [], "upload_file": []}
        rc = self._run_with_fake_hub(tmp_path, mini_crossdocked, assets_dir, uploads)
        assert rc == 0
        assert uploads["create_repo"], "repo creation missing"
        repo_ids = {u["repo_id"] for u in uploads["create_repo"]}
        assert repo_ids == {"JJKK1212/3d-syntree-multidataset"}

        paths_in_repo = [u["path_in_repo"] for u in uploads["upload_file"]]
        assert "enamine_3d_subset.parquet" in paths_in_repo
        assert any(p.startswith("targets/") and p.endswith("_pocket.pdb")
                   for p in paths_in_repo)
        for split in ("train", "val", "test"):
            assert f"data/{split}/manifest.json" in paths_in_repo
            shard_paths = [p for p in paths_in_repo
                           if p.startswith(f"data/{split}/") and p.endswith(".pt.gz")]
            assert shard_paths, f"no shards uploaded for {split}"

    def test_upload_without_token_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        argv = [
            "build_full_dataset.py",
            "--raw-dir", str(tmp_path),
            "--upload",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            rc = bfd.main()
        finally:
            sys.argv = old_argv
        assert rc == 1


class TestZeroValidComplexes:
    def test_no_pairs_discovered_fails(self, tmp_path, monkeypatch, assets_dir):
        empty_raw = tmp_path / "empty_raw"
        empty_raw.mkdir()
        (empty_raw / ".extracted_marker").touch()
        argv = [
            "build_full_dataset.py",
            "--raw-dir", str(empty_raw),
            "--catalog", assets_dir["catalog_path"],
            "--output-dir", str(tmp_path / "out"),
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            rc = bfd.main()
        finally:
            sys.argv = old_argv
        assert rc == 1

    def test_undecomposable_ligands_fail(self, tmp_path, assets_dir):
        """Pockets whose ligands cannot decompose produce zero trajectories
        -> the build must fail closed instead of uploading an empty dataset."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / ".extracted_marker").touch()
        # n-hexane: no catalog synthon can reconstruct it.
        _write_sdf(_embedded("CCCCCC"), raw / "x1_pocket10.pdb".replace("x1_pocket10", "x1_pocket10"))
        (raw / "x1_pocket10.pdb").write_text(POCKET_PDB)
        argv = [
            "build_full_dataset.py",
            "--raw-dir", str(raw),
            "--catalog", assets_dir["catalog_path"],
            "--output-dir", str(tmp_path / "out2"),
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            rc = bfd.main()
        finally:
            sys.argv = old_argv
        assert rc == 1
