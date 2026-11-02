"""Tests for the multi-source preprocessing library (#16 DATA-01).

Covers the module contracts pinned by the issue and D-V2-021: pocket
extraction policy (10 A sphere, water/additive stripping, metal keep),
ligand filters with exact rejection codes, PDB-ligand reconstruction
with template bond orders, chain-sequence extraction, manifest
resume semantics, and deterministic gzip artifact writing.
"""

from __future__ import annotations

import gzip
import json
import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.data.preprocessing import (
    LIGAND_MIN_HEAVY_ATOMS,
    LIGAND_MW_RANGE,
    LigandRejection,
    ManifestRecord,
    ManifestWriter,
    apply_ligand_filters,
    chain_sequences,
    classify_resname,
    complex_id_for,
    extract_pocket_lines,
    ligand_centroid,
    ligand_from_mol_block,
    ligand_from_pdb_residue,
    parse_conect_edges,
    parse_experiment_method,
    parse_pdb_residue_lines,
    parse_resolution,
    standardize_complex,
)


# ---------------------------------------------------------------------------
# Fixtures: a small hand-built protein-like PDB around a drug-like ligand
# ---------------------------------------------------------------------------

def _atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
    record: str = "ATOM  ",
) -> str:
    return (
        f"{record}{serial:5d} {name:>4s} {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
    )


def _ibuprofen_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    assert mol is not None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return Chem.RemoveHs(mol)


def _protein_pdb_text() -> str:
    """A 4-residue 'protein' plus waters, an additive, a metal and a
    foreign ligand, laid out around the origin."""
    lines = ["HEADER    TEST PROTEIN", "EXPDTA    X-RAY DIFFRACTION"]
    # residues straddling the origin (inside the 10 A sphere)
    lines.append(_atom_line(1, "N", "ALA", "A", 1, 3.0, 0.0, 0.0, "N"))
    lines.append(_atom_line(2, "CA", "ALA", "A", 1, 4.0, 0.5, 0.0, "C"))
    lines.append(_atom_line(3, "N", "GLY", "A", 2, -3.0, 1.0, 0.0, "N"))
    lines.append(_atom_line(4, "CA", "GLY", "A", 2, -4.0, 1.5, 0.5, "C"))
    # modified residue inside the sphere (kept as protein)
    lines.append(
        _atom_line(5, "SE", "MSE", "A", 3, 0.0, 4.0, 0.0, "SE", record="HETATM")
    )
    # residue far outside the sphere (dropped by distance)
    lines.append(_atom_line(6, "N", "LEU", "A", 4, 50.0, 0.0, 0.0, "N"))
    # water inside the sphere (stripped by policy)
    lines.append(
        _atom_line(7, "O", "HOH", "A", 5, 1.0, 1.0, 1.0, "O", record="HETATM")
    )
    # glycerol additive inside the sphere (stripped by policy)
    lines.append(
        _atom_line(8, "C1", "GOL", "A", 6, -1.0, 1.0, 1.0, "C", record="HETATM")
    )
    # catalytic zinc inside the sphere (kept as metal)
    lines.append(
        _atom_line(9, "ZN", "ZN", "A", 7, 0.0, -3.0, 0.0, "ZN", record="HETATM")
    )
    # a foreign ligand-like HETATM inside the sphere (stripped)
    lines.append(
        _atom_line(10, "CA", "LIG", "A", 8, 2.0, -2.0, 0.0, "C", record="HETATM")
    )
    # sodium salt (stripped as salt, NOT kept as metal)
    lines.append(
        _atom_line(11, "NA", "NA", "A", 9, 0.5, -0.5, 0.0, "NA", record="HETATM")
    )
    return "\n".join(lines) + "\nEND\n"


class TestResidueClassification:
    def test_buckets(self):
        assert classify_resname("ALA") == "protein"
        assert classify_resname("MSE") == "protein"
        assert classify_resname("HOH") == "water"
        assert classify_resname("GOL") == "additive"
        assert classify_resname("NA") == "additive"
        assert classify_resname("CL") == "additive"
        assert classify_resname("ZN") == "metal"
        assert classify_resname("MG") == "metal"
        assert classify_resname("STI") == "ligand"


class TestPocketExtraction:
    def test_sphere_policy(self):
        ligand = _ibuprofen_mol()
        centroid = ligand_centroid(ligand)
        pocket = extract_pocket_lines(
            _protein_pdb_text().splitlines(), centroid, radius=10.0
        )
        residues = {line[17:20].strip() for line in pocket}
        # protein residues near the origin are kept
        assert {"ALA", "GLY", "MSE"} <= residues
        # waters, additives, salts and ligand-like HETATM are stripped
        assert "HOH" not in residues
        assert "GOL" not in residues
        assert "NA" not in residues
        assert "LIG" not in residues
        # far-away residue dropped by the distance criterion
        assert "LEU" not in residues
        # catalytic metal kept
        assert "ZN" in residues
        # every kept line is ATOM/HETATM with parseable coordinates
        for line in pocket:
            assert line.startswith(("ATOM", "HETATM"))
            float(line[30:38]), float(line[38:46]), float(line[46:54])

    def test_empty_pocket_when_protein_far(self):
        ligand = _ibuprofen_mol()
        centroid = ligand_centroid(ligand)
        far_text = _atom_line(1, "N", "ALA", "A", 1, 500.0, 0.0, 0.0, "N")
        assert extract_pocket_lines([far_text], centroid) == []


class TestChainSequences:
    def test_sequences_and_restriction(self):
        text = _protein_pdb_text()
        pocket = extract_pocket_lines(text.splitlines(), [0.0, 0.0, 0.0])
        chains = chain_sequences(pocket)
        assert set(chains) == {"A"}
        assert set(chains["A"]) <= set("AGX")  # ALA, GLY, MSE->X
        # restriction: only chains present in the pocket are emitted
        full = chain_sequences(text.splitlines())
        assert set(full) == {"A"}

    def test_modified_residue_maps_to_x(self):
        assert chain_sequences(
            [_atom_line(1, "SE", "MSE", "A", 1, 0, 0, 0, "SE")]
        ) == {"A": "X"}


class TestHeaderParsing:
    def test_resolution_and_method(self):
        text = (
            "EXPDTA    X-RAY DIFFRACTION\n"
            "REMARK   2 RESOLUTION.    1.45 ANGSTROMS.\n"
        )
        assert parse_resolution(text) == 1.45
        assert parse_experiment_method(text) == "X-RAY DIFFRACTION"

    def test_missing_resolution_returns_none(self):
        assert parse_resolution("EXPDTA    X-RAY DIFFRACTION\n") is None
        assert parse_resolution("") is None


class TestLigandParsingAndFilters:
    def test_mol_block_roundtrip(self):
        ligand = _ibuprofen_mol()
        block = Chem.MolToMolBlock(ligand)
        parsed = ligand_from_mol_block(block + "$$$$\n")
        assert parsed.GetNumAtoms() == ligand.GetNumAtoms()

    def test_mol_block_parse_failure_code(self):
        with pytest.raises(LigandRejection) as exc:
            ligand_from_mol_block("not a mol block")
        assert exc.value.reason_code == "LIGAND_SDF_PARSE_FAILURE"

    def test_filters_pass(self):
        meta = apply_ligand_filters(_ibuprofen_mol())
        assert LIGAND_MW_RANGE[0] <= meta["molecular_weight_da"] <= LIGAND_MW_RANGE[1]
        assert meta["heavy_atoms"] >= LIGAND_MIN_HEAVY_ATOMS
        assert set(meta) == {
            "smiles",
            "molecular_weight_da",
            "heavy_atoms",
            "rotatable_bonds",
            "fraction_sp3",
            "formula",
            "formal_charge",
            "num_rings",
        }

    def test_filter_too_few_heavy_atoms(self):
        with pytest.raises(LigandRejection) as exc:
            apply_ligand_filters(Chem.MolFromSmiles("CCO"))
        assert exc.value.reason_code == "LIGAND_TOO_FEW_HEAVY_ATOMS"

    def test_filter_mw_below_range(self):
        # decane: 10 heavy atoms, MW 142.3 < 150
        with pytest.raises(LigandRejection) as exc:
            apply_ligand_filters(Chem.MolFromSmiles("CCCCCCCCCC"))
        assert exc.value.reason_code == "LIGAND_MW_BELOW_RANGE"

    def test_filter_mw_above_range(self):
        # C60 alkane: MW 842.8 > 800
        with pytest.raises(LigandRejection) as exc:
            apply_ligand_filters(Chem.MolFromSmiles("C" * 60))
        assert exc.value.reason_code == "LIGAND_MW_ABOVE_RANGE"

    def test_pdb_residue_with_template(self):
        # ethanolamine-like C-C-N residue with 3D coordinates + CONECT
        lines = [
            _atom_line(1, "C1", "XXX", "A", 900, 0.0, 0.0, 0.0, "C", "HETATM"),
            _atom_line(2, "C2", "XXX", "A", 900, 1.52, 0.0, 0.0, "C", "HETATM"),
            _atom_line(3, "N", "XXX", "A", 900, 2.10, 1.20, 0.0, "N", "HETATM"),
        ]
        conect = parse_conect_edges(
            "CONECT    1    2\nCONECT    2    1    3\nCONECT    3    2\n",
            [1, 2, 3],
        )
        mol = ligand_from_pdb_residue(lines, template_smiles="CCN", conect=conect)
        assert Chem.MolToSmiles(mol) == "CCN"
        assert mol.GetNumConformers() == 1

    def test_pdb_residue_proximity_fallback(self):
        # no CONECT table: proximity bonding supplies the connectivity
        lines = [
            _atom_line(1, "C1", "XXX", "A", 900, 0.0, 0.0, 0.0, "C", "HETATM"),
            _atom_line(2, "C2", "XXX", "A", 900, 1.52, 0.0, 0.0, "C", "HETATM"),
            _atom_line(3, "O", "XXX", "A", 900, 1.52, 1.43, 0.0, "O", "HETATM"),
        ]
        mol = ligand_from_pdb_residue(lines, template_smiles="CCO")
        assert Chem.MolToSmiles(mol) == "CCO"

    def test_pdb_residue_without_template_rejects(self):
        # no usable annotation SMILES -> fail-closed (never guessed bonds)
        lines = [
            _atom_line(1, "C1", "XXX", "A", 900, 0.0, 0.0, 0.0, "C", "HETATM"),
            _atom_line(2, "C2", "XXX", "A", 900, 1.52, 0.0, 0.0, "C", "HETATM"),
        ]
        with pytest.raises(LigandRejection) as exc:
            ligand_from_pdb_residue(lines)
        assert exc.value.reason_code == "LIGAND_BOND_PERCEPTION_FAILURE"

    def test_pdb_residue_template_mismatch(self):
        lines = [
            _atom_line(1, "C1", "XXX", "A", 900, 0.0, 0.0, 0.0, "C", "HETATM"),
            _atom_line(2, "C2", "XXX", "A", 900, 1.52, 0.0, 0.0, "C", "HETATM"),
        ]
        with pytest.raises(LigandRejection) as exc:
            ligand_from_pdb_residue(lines, template_smiles="CCO")
        assert exc.value.reason_code == "LIGAND_TEMPLATE_MISMATCH"

    def test_conect_edges_parsing(self):
        text = (
            "CONECT  101  102  103\n"
            "CONECT  102  101\n"
            "CONECT  103  101  999\n"
            "CONECT  999  103\n"
        )
        edges = parse_conect_edges(text, [101, 102, 103])
        assert edges == {101: (102, 103), 102: (101,), 103: (101,)}

    def test_pdb_residue_not_found(self):
        with pytest.raises(LigandRejection) as exc:
            ligand_from_pdb_residue([])
        assert exc.value.reason_code == "LIGAND_RESIDUE_NOT_FOUND"

    def test_altloc_filtering(self):
        # altLoc occupies column 17 (index 16); an alternate-conformer
        # line with altLoc B is excluded from ligand extraction
        primary = (
            "HETATM  999  C1  XXX A 900    0.000   0.000   1.500  1.00 20.00          C"
        )
        altconf = primary[:16] + "B" + primary[17:]
        assert altconf[16] == "B" and altconf[17:20] == "XXX"
        got = parse_pdb_residue_lines(
            primary + "\n" + altconf + "\n", resname="XXX", chain="A", resseq="900"
        )
        assert len(got) == 1
        assert float(got[0][30:38]) == 0.0


class TestStandardizeComplex:
    def test_accepted_complex_writes_artifacts(self, tmp_path):
        ligand = _ibuprofen_mol()
        record = standardize_complex(
            complex_id="test_src_cx1",
            source="test_src",
            protein_pdb_text=_protein_pdb_text(),
            ligand=ligand,
            out_dir=str(tmp_path),
            pdb_id="1TST",
            method="X-RAY DIFFRACTION",
            resolution_angstrom=1.45,
            rmsd_angstrom=None,
            affinity_uM=3.2,
            provenance=["unit test"],
        )
        assert record.status == "accepted"
        assert record.resolution_angstrom == 1.45
        assert record.rmsd_angstrom is None
        assert record.affinity_uM == 3.2
        assert record.ligand["heavy_atoms"] == ligand.GetNumAtoms()
        assert record.pocket["n_atoms"] > 0
        assert record.protein_chains and "A" in record.protein_chains
        pocket_path = tmp_path / record.artifacts["pocket"]
        ligand_path = tmp_path / record.artifacts["ligand"]
        # artifacts are recorded relative to out_dir (portable manifest)
        assert not os.path.isabs(record.artifacts["pocket"])
        assert os.path.exists(pocket_path) and os.path.exists(ligand_path)
        # pocket content: kept residues only (per-line resname column)
        with open(pocket_path) as handle:
            pocket_out_lines = handle.read().splitlines()
        residues = {line[17:20].strip() for line in pocket_out_lines if line.startswith(("ATOM", "HETATM"))}
        assert "MSE" in residues and "ZN" in residues
        assert "HOH" not in residues
        assert "GOL" not in residues
        assert "LIG" not in residues
        assert "NA" not in residues
        # ligand round-trips through the written SDF
        reloaded = Chem.SDMolSupplier(ligand_path, removeHs=False)
        mols = [m for m in reloaded if m is not None]
        assert len(mols) == 1
        assert mols[0].GetNumAtoms() == ligand.GetNumAtoms()

    def test_rejected_ligand_record(self, tmp_path):
        record = standardize_complex(
            complex_id="test_src_small",
            source="test_src",
            protein_pdb_text=_protein_pdb_text(),
            ligand=Chem.MolFromSmiles("CCO"),
            out_dir=str(tmp_path),
        )
        assert record.status == "rejected"
        assert record.reason_code == "LIGAND_TOO_FEW_HEAVY_ATOMS"
        assert record.artifacts is None
        assert not os.path.exists(str(tmp_path / "test_src_small_pocket.pdb"))

    def test_gzip_artifacts_are_deterministic(self, tmp_path):
        ligand = _ibuprofen_mol()
        kwargs = dict(
            complex_id="test_gz_cx",
            source="test_src",
            protein_pdb_text=_protein_pdb_text(),
            ligand=ligand,
            gzip_output=True,
        )
        standardize_complex(out_dir=str(tmp_path / "a"), **kwargs)
        standardize_complex(out_dir=str(tmp_path / "b"), **kwargs)
        for name in ("test_gz_cx_pocket.pdb.gz", "test_gz_cx_ligand.sdf.gz"):
            first = (tmp_path / "a" / name).read_bytes()
            second = (tmp_path / "b" / name).read_bytes()
            assert first == second, f"non-deterministic gzip artifact {name}"
        with gzip.open(tmp_path / "a" / "test_gz_cx_pocket.pdb.gz", "rt") as handle:
            assert "ZN" in handle.read()

    def test_complex_id_scheme(self):
        assert complex_id_for("pdbbind", "10gs") == "pdbbind_10gs"
        assert complex_id_for("crossdocked", "T", "1ady_A_rec_x") == "crossdocked_T_1ady_A_rec_x"


class TestManifestWriter:
    def _record(self, cid: str, status: str = "accepted") -> ManifestRecord:
        return ManifestRecord(complex_id=cid, source="test", status=status)

    def test_append_and_resume(self, tmp_path):
        path = str(tmp_path / "manifest.jsonl")
        writer = ManifestWriter(path)
        writer.append(self._record("a"))
        writer.append(self._record("b", "rejected"))
        assert writer.already_processed("a")
        assert writer.status_of("b") == "rejected"
        assert writer.counts == {"accepted": 1, "rejected": 1}
        # a fresh writer (new session) sees the same resume state
        resumed = ManifestWriter(path)
        assert resumed.already_processed("a")
        assert resumed.counts == {"accepted": 1, "rejected": 1}

    def test_truncated_tail_is_repaired(self, tmp_path):
        path = str(tmp_path / "manifest.jsonl")
        writer = ManifestWriter(path)
        writer.append(self._record("a"))
        # simulate an interrupted write: garbage partial line
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"complex_id": "b", "stat')
        writer2 = ManifestWriter(path)
        assert writer2.already_processed("a")
        assert not writer2.already_processed("b")
        writer2.append(self._record("b"))
        with open(path, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        assert [rec["complex_id"] for rec in lines] == ["a", "b"]

    def test_record_roundtrip_json(self, tmp_path):
        path = str(tmp_path / "manifest.jsonl")
        writer = ManifestWriter(path)
        record = ManifestRecord(
            complex_id="x",
            source="s",
            status="accepted",
            ligand={"heavy_atoms": 22},
            protein_chains={"A": "AGX"},
        )
        writer.append(record)
        with open(path, encoding="utf-8") as handle:
            loaded = json.loads(handle.readline())
        assert loaded["complex_id"] == "x"
        assert loaded["ligand"]["heavy_atoms"] == 22
        assert loaded["protein_chains"] == {"A": "AGX"}
        assert ManifestRecord(**loaded).complex_id == "x"


class TestPoliciesAreComplete:
    def test_rejection_code_vocabulary(self):
        # every reason code the module can emit is documented
        from syntree.data import preprocessing as pp

        codes = (
            pp.LIGAND_REJECTION_CODES
            | pp.POCKET_REJECTION_CODES
            | pp.SOURCE_REJECTION_CODES
        )
        assert "LIGAND_MW_BELOW_RANGE" in codes
        assert "POCKET_EMPTY" in codes
        assert "METHOD_NOT_XRAY" in codes
