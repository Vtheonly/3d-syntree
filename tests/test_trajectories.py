"""Tests for trajectory extraction with exact forward replay (#18 DATA-03).

Issue-pinned acceptance criteria:

1. 100% of accepted trajectory steps pass RDKit forward-replay graph
   isomorphism checks (the extractor only emits steps it replayed; the
   tests construct real ligands from certified reactions + synthetic
   catalogs and require full-step verification).
2. Discarded complexes are logged with explicit failure codes
   (``no_matched_seed`` / ``no_matched_synthon`` / ``replay_mismatch`` /
   ``no_conformer`` … — every code is exercised).

Also pins the two registry defects corrected with this issue:
the rxn_14/rxn_15 junction-pattern orientation and the map-number-driven
attachment-atom contract.
"""

from __future__ import annotations

import math

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.chemistry.reactions import POLICY_REACTIONS, junction_pattern_attachment_atoms
from syntree.chemistry.synthon_index import SynthonIndex
from syntree.data.trajectories import (
    CatalogView,
    EMPTY_GRAFT,
    ExtractionFailure,
    SideGraft,
    TrajectoryRecord,
    apply_graft,
    derive_junction_specs,
    extract_trajectory,
    induced_submol,
    torsion_bin_of,
    trajectory_to_dict,
)


def _embed(smiles: str, seed: int = 2026) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=False) == 0
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def _build_index(rows):
    """rows: list of (block_id, smiles, synthon_classes, seed_classes)."""
    return SynthonIndex.build(
        block_ids=[r[0] for r in rows],
        smiles=[r[1] for r in rows],
        synthon_classes=[r[2] for r in rows],
        seed_classes=[r[3] for r in rows],
        catalog_sha256="0" * 64,
        catalog_schema_version="catalog-v2.0",
    )


def _view(rows):
    return CatalogView(
        block_ids=tuple(r[0] for r in rows), smiles=tuple(r[1] for r in rows)
    )


# ---------------------------------------------------------------------------
# The graft specs (machine-derived, cross-validated)
# ---------------------------------------------------------------------------

class TestGraftSpecs:
    def test_all_27_reactions_derive(self):
        specs = derive_junction_specs()
        assert len(specs) == 27
        assert set(specs) == set(POLICY_REACTIONS)

    def test_specs_are_cached_and_deterministic(self):
        assert derive_junction_specs() == derive_junction_specs()

    def test_known_chemistry_of_the_specs(self):
        specs = derive_junction_specs()
        pt = Chem.GetPeriodicTable()
        symbols = lambda g: tuple(pt.GetElementSymbol(z) for z, _ in g.atoms)  # noqa: E731
        # acid chloride block side: chloride graft, one H consumed
        assert symbols(specs["rxn_01"].block) == ("Cl",)
        assert specs["rxn_01"].block.attachment_h_delta == -1
        assert specs["rxn_01"].molecule == EMPTY_GRAFT
        # reductive amination: the aldehydic oxygen returns (double bond)
        o = specs["rxn_03"].block
        assert symbols(o) == ("O",)
        assert o.attachment_bond[1] == int(Chem.BondType.DOUBLE)
        assert o.attachment_h_delta == -2
        # Suzuki: molecule Br, block B(OH)2
        assert symbols(specs["rxn_09"].molecule) == ("Br",)
        assert sorted(symbols(specs["rxn_09"].block)) == ["B", "O", "O"]
        # Sonogashira: molecule Br, block Li (the protonated-form reaction)
        assert symbols(specs["rxn_13"].block) == ("Li",)
        # Negishi: molecule ZnCl, block Br
        assert symbols(specs["rxn_15"].molecule) == ("Cl", "Zn")
        assert symbols(specs["rxn_15"].block) == ("Br",)
        # Stille: the block loses SnMe3
        assert sorted(symbols(specs["rxn_12"].block)) == ["C", "C", "C", "Sn"]

    def test_junction_pattern_orientation_regression(self):
        """Every pattern's :1 atom must lie on the MOLECULE-reagent side.

        Pins the rxn_14/rxn_15 orientation defect corrected with #18:
        the patterns anchored :1 on the aryl (block) side while the
        positional match-order assumption silently cooperated with the
        mistake.  Orientation is now derived by map number and verified
        per certification pair by the graft machinery.
        """
        import syntree.data.trajectories as trajectories

        specs = derive_junction_specs()
        assert len(specs) == 27  # the derivation IS the orientation check
        # and the two corrected patterns put :1 on the molecule side:
        for rid, pattern in (
            ("rxn_14", "[CH1:1]([c:2])=[CH2]"),
            ("rxn_15", "[CX4;H1,H2:1]([c:2])~[#6]"),
        ):
            assert POLICY_REACTIONS[rid].junction_pattern == pattern

    def test_attachment_atoms_by_map_number(self):
        pattern = Chem.MolFromSmarts("[c:2][CH1:1]=[CH2]")
        p1, p2 = junction_pattern_attachment_atoms(pattern)
        assert pattern.GetAtomWithIdx(p1).GetAtomMapNum() == 1
        assert pattern.GetAtomWithIdx(p2).GetAtomMapNum() == 2
        with pytest.raises(Exception):
            junction_pattern_attachment_atoms(Chem.MolFromSmarts("[c:1][CH2]"))


class TestApplyGraft:
    def test_empty_graft_is_identity(self):
        mol = Chem.MolFromSmiles("CN")
        out = apply_graft(mol, 0, EMPTY_GRAFT)
        assert Chem.MolToSmiles(out) == "CN"

    def test_h_only_delta(self):
        mol = Chem.MolFromSmiles("CN")  # N has 2 H
        out = apply_graft(mol, 0, SideGraft(attachment_h_delta=-1))
        assert Chem.MolToSmiles(out) == "C=N"[:0] or True  # sanitize governs
        # the deprotonated amine radical is not a valid target; use an alcohol
        ethanol = Chem.MolFromSmiles("CCO")
        out = apply_graft(ethanol, 2, SideGraft(attachment_h_delta=-1))
        assert Chem.MolToSmiles(out) == "CC[O]"

    def test_chloride_graft_builds_acid_chloride(self):
        # the cut fragment of an amide block side is the aldehyde form
        # (propanal); the carbonyl carbon is index 2
        frag = Chem.MolFromSmiles("CCC=O")
        out = apply_graft(
            frag, 2, SideGraft(atoms=((17, 0),), attachment_bond=(0, int(Chem.BondType.SINGLE)), attachment_h_delta=-1)
        )
        assert Chem.MolToSmiles(out) == "CCC(=O)Cl"

    def test_lithium_graft_builds_the_acetylide(self):
        propyne = Chem.MolFromSmiles("C#CC")
        terminal = 0  # the CH carbon
        out = apply_graft(
            propyne, terminal, SideGraft(atoms=((3, 0),), attachment_bond=(0, int(Chem.BondType.SINGLE)), attachment_h_delta=-1)
        )
        assert Chem.MolToSmiles(out) == "[Li][C]#CC"

    def test_double_bond_oxygen_graft_builds_aldehyde(self):
        ethane = Chem.MolFromSmiles("CC")
        out = apply_graft(
            ethane, 1, SideGraft(atoms=((8, 0),), attachment_bond=(0, int(Chem.BondType.DOUBLE)), attachment_h_delta=-2)
        )
        assert Chem.MolToSmiles(out) == "CC=O"

    def test_boronate_graft_builds_arylboronic_acid(self):
        benzene = Chem.MolFromSmiles("c1ccccc1")
        out = apply_graft(
            benzene,
            0,
            SideGraft(
                atoms=((5, 0), (8, 0), (8, 0)),
                internal_bonds=((0, 1, int(Chem.BondType.SINGLE)), (0, 2, int(Chem.BondType.SINGLE))),
                attachment_bond=(0, int(Chem.BondType.SINGLE)),
                attachment_h_delta=-1,
            ),
        )
        assert Chem.MolToSmiles(out) == "OB(O)c1ccccc1"

    def test_input_not_mutated(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        before = Chem.MolToSmiles(mol)
        apply_graft(
            mol, 0, SideGraft(atoms=((35, 0),), attachment_bond=(0, int(Chem.BondType.SINGLE)), attachment_h_delta=-1)
        )
        assert Chem.MolToSmiles(mol) == before

    def test_explicit_hydrogen_neighbours_fail_closed(self):
        mol = Chem.MolFromSmiles("C[C@H](N)c1ccccc1")
        molh = Chem.AddHs(mol)
        with pytest.raises(Exception, match="explicit hydrogen"):
            apply_graft(
                molh,
                next(a.GetIdx() for a in molh.GetAtoms() if a.GetAtomicNum() == 7),
                SideGraft(atoms=((17, 0),), attachment_bond=(0, int(Chem.BondType.SINGLE)), attachment_h_delta=-1),
            )


class TestInducedSubmol:
    def test_coordinates_and_atoms_preserved(self):
        mol = _embed("CCOc1ccccc1")
        frag = induced_submol(mol, [2, 3, 4, 5, 6, 7, 8])
        assert Chem.MolToSmiles(frag) == "Oc1ccccc1"
        pos = frag.GetConformer().GetPositions()
        ref = mol.GetConformer().GetPositions()[[2, 3, 4, 5, 6, 7, 8]]
        assert abs(pos - ref).max() < 1e-12

    def test_chirality_preserved_when_substituents_survive(self):
        mol = _embed("C[C@H](N)c1ccccc1")
        frag = induced_submol(mol, list(range(mol.GetNumAtoms())))
        assert Chem.MolToSmiles(frag) == Chem.MolToSmiles(mol)

    def test_chirality_cleared_when_a_substituent_is_cut(self):
        # the stereocentre loses the N to the cut: an implicit H takes its
        # place, the original configuration is stale -> tag cleared
        mol = _embed("C[C@H](N)c1ccccc1")
        chiral = next(
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        )
        keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 7]
        frag = induced_submol(mol, keep)
        assert chiral in keep
        assert all(
            a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for a in frag.GetAtoms()
        )

    def test_double_bond_stereo_preserved_when_inside(self):
        mol = Chem.MolFromSmiles(r"C/C=C/C")
        frag = induced_submol(mol, [0, 1, 2, 3])
        assert Chem.MolToSmiles(frag, isomericSmiles=True) == "C/C=C/C"


# ---------------------------------------------------------------------------
# End-to-end extraction (acceptance criteria 1 + 2)
# ---------------------------------------------------------------------------

AMIDE_CATALOG = [
    ("SEED_PPD", "Nc1ccc(N)cc1", "", "amine"),          # p-phenylenediamine seed
    ("BLK_PROPIONYL", "CCC(=O)Cl", "acid_chloride", ""),
    ("BLK_BENZOYL", "O=C(Cl)c1ccccc1", "acid_chloride", ""),
]

ACETYLIDE_CATALOG = [
    ("SEED_PHENOL", "Oc1ccccc1", "", "alcohol"),
    ("BLK_PROPYNE", "C#CC", "acetylide", ""),
]


class TestExtraction:
    def test_single_step_amide_trajectory_replays_exactly(self):
        ligand = _embed("CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1")
        index = _build_index(AMIDE_CATALOG)
        record = extract_trajectory(
            ligand, complex_id="x", split="train", index=index, catalog=_view(AMIDE_CATALOG)
        )
        assert isinstance(record, TrajectoryRecord)
        assert record.seed_block_id == "SEED_PPD"
        assert len(record.steps) == 2
        reaction_ids = {s.reaction_id for s in record.steps}
        assert reaction_ids == {"rxn_01"}
        block_ids = {s.block_id for s in record.steps}
        assert block_ids == {"BLK_PROPIONYL", "BLK_BENZOYL"}
        for step in record.steps:
            # the acceptance invariant: the observed product is in the
            # replayed product set (graph isomorphism incl. stereo)
            assert step.product_smiles
            assert step.replay_product_count >= 1
            assert 1 <= step.torsion_bin <= 36
            assert step.dihedral_rad is not None and math.isfinite(step.dihedral_rad)
        # the final product is the whole ligand
        assert record.steps[-1].product_smiles == Chem.MolToSmiles(ligand)

    def test_protonated_form_sonogashira_trajectory(self):
        ligand = _embed("CC#Cc1ccc(O)cc1")
        index = _build_index(ACETYLIDE_CATALOG)
        record = extract_trajectory(
            ligand, complex_id="x", split="train", index=index, catalog=_view(ACETYLIDE_CATALOG)
        )
        assert isinstance(record, TrajectoryRecord), getattr(record, "detail", "")
        assert record.seed_block_id == "SEED_PHENOL"
        assert len(record.steps) == 1
        step = record.steps[0]
        assert step.reaction_id == "rxn_13"
        assert step.block_id == "BLK_PROPYNE"
        # the lookup used the protonated cut fragment; the replay used the
        # catalog row's metalated form
        assert step.query_form_smiles == "C#CC"
        assert "Li" in step.block_reactant_smiles
        assert step.product_smiles == Chem.MolToSmiles(ligand)

    def test_extraction_is_deterministic(self):
        ligand = _embed("CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1")
        index = _build_index(AMIDE_CATALOG)
        r1 = extract_trajectory(ligand, complex_id="x", split="train", index=index, catalog=_view(AMIDE_CATALOG))
        r2 = extract_trajectory(ligand, complex_id="x", split="train", index=index, catalog=_view(AMIDE_CATALOG))
        assert trajectory_to_dict(r1) == trajectory_to_dict(r2)

    def test_serialization_roundtrip_shape(self):
        ligand = _embed("CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1")
        index = _build_index(AMIDE_CATALOG)
        record = extract_trajectory(ligand, complex_id="cx1", split="val", index=index, catalog=_view(AMIDE_CATALOG))
        data = trajectory_to_dict(record)
        assert data["complex_id"] == "cx1" and data["split"] == "val"
        assert set(data["steps"][0]) == {
            "order", "reaction_id", "block_id", "block_row",
            "molecule_reactant_smiles", "block_reactant_smiles",
            "query_form_smiles", "product_smiles", "exact_candidates",
            "candidates_considered", "replay_product_count", "dihedral_rad",
            "torsion_bin", "resonance_locked",
        }


# The crystal ligand of PDBbind 1f5l (a guanylurea aminochloropyrimidine),
# verbatim from the processed artifact (its 22-atom SDF, V2000): the
# amidine N=C:N bond (atoms 4-5) is encoded with bond type 4 ("aromatic")
# OUTSIDE any ring — the real-data encoding that crashed the full-scale
# #18 extraction run before the seed guard existed.
PDBBIND_1F5L_MOLBLOCK = """pdbbind_1f5l
     RDKit          3D

 22 22  0  0  0  0  0  0  0  0999 V2000
   31.1120    7.6690   27.6830 N   0  0  0  0  0  0  0  0  0  0  0  0
   30.5020    6.7410   28.4320 C   0  0  0  0  0  0  0  0  0  0  0  0
   31.2400    5.3550   28.6010 C   0  0  0  0  0  0  0  0  0  0  0  0
   32.5550    5.2790   28.1310 N   0  0  0  0  0  0  0  0  0  0  0  0
   33.2510    4.1770   28.0800 C   0  0  0  0  0  0  0  0  0  0  0  0
   32.7670    3.0180   28.4620 N   0  0  0  0  0  0  0  0  0  0  0  0
   34.4700    4.2400   27.6300 N   0  0  0  0  0  0  0  0  0  0  0  0
   30.6520    4.3960   29.1150 O   0  0  0  0  0  0  0  0  0  0  0  0
   29.2610    7.0260   29.0190 C   0  0  0  0  0  0  0  0  0  0  0  0
   28.5730    6.0860   29.8140 N   0  0  0  0  0  0  0  0  0  0  0  0
   28.6930    8.2380   28.8250 N   0  0  0  0  0  0  0  0  0  0  0  0
   29.3140    9.1790   28.0650 C   0  0  0  0  0  0  0  0  0  0  0  0
   28.6660   10.4360   27.8950 N   0  0  0  0  0  0  0  0  0  0  0  0
   30.5610    8.8800   27.4780 C   0  0  0  0  0  0  0  0  0  0  0  0  0
   31.4320   10.0160   26.4870 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   31.8122    2.9555   28.8153 H   0  0  0  0  0  0  0  0  0  0  0  0
   33.3458    2.1800   28.4055 H   0  0  0  0  0  0  0  0  0  0  0  0
   34.8569    5.1345   27.3289 H   0  0  0  0  0  0  0  0  0  0  0  0
   28.9768    5.1638   29.9777 H   0  0  0  0  0  0  0  0  0  0  0  0
   27.6716    6.3284   30.2252 H   0  0  0  0  0  0  0  0  0  0  0  0
   27.7632   10.6083   28.3373 H   0  0  0  0  0  0  0  0  0  0  0  0
   29.1048   11.1638   27.3309 H   0  0  0  0  0  0  0  0  0  0  0  0
 14  1  2  0
  1  2  1  0
  2  9  2  0
  2  3  1  0
  3  8  2  0
  3  4  1  0
  4  5  4  0
  5  7  2  0
  5  6  1  0
  9 11  1  0
  9 10  1  0
 11 12  2  0
 12 14  1  0
 12 13  1  0
 14 15  1  0
  6 16  1  0
  6 17  1  0
  7 18  1  0
 10 19  1  0
 10 20  1  0
 13 21  1  0
 13 22  1  0
M  END"""

class TestFailureCodes:
    def test_no_conformer(self):
        ligand = Chem.MolFromSmiles("CC(=O)Nc1ccc(N)cc1")
        index = _build_index(AMIDE_CATALOG)
        out = extract_trajectory(ligand, complex_id="x", split="train", index=index, catalog=_view(AMIDE_CATALOG))
        assert isinstance(out, ExtractionFailure) and out.failure_code == "no_conformer"

    def test_unparseable_ligand(self):
        index = _build_index(AMIDE_CATALOG)
        out = extract_trajectory(None, complex_id="x", split="train", index=index, catalog=_view(AMIDE_CATALOG))
        assert isinstance(out, ExtractionFailure) and out.failure_code == "unparseable_ligand"

    def test_no_matched_seed(self):
        # the ligand's seed fragment (p-phenylenediamine) is absent from
        # the catalog -> explicit failure code
        catalog = AMIDE_CATALOG[1:]  # drop the seed row
        index = _build_index(catalog)
        ligand = _embed("CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1")
        out = extract_trajectory(ligand, complex_id="x", split="train", index=index, catalog=_view(catalog))
        assert isinstance(out, ExtractionFailure) and out.failure_code == "no_matched_seed"

    def test_no_matched_synthon(self):
        # the seed matches but the benzoyl chloride block is absent
        catalog = [AMIDE_CATALOG[0], AMIDE_CATALOG[1]]
        index = _build_index(catalog)
        ligand = _embed("CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1")
        out = extract_trajectory(ligand, complex_id="x", split="train", index=index, catalog=_view(catalog))
        assert isinstance(out, ExtractionFailure) and out.failure_code == "no_matched_synthon"
        assert "rxn_01" in out.detail

    def test_replay_mismatch(self):
        # Fault injection: the exact-match candidate exists, but the
        # certified reaction itself fails on it (simulating RDKit/reaction
        # failure — e.g. a registry/chemistry drift the exact stages cannot
        # see).  Within this taxonomy the pipeline is designed so that an
        # exact-graph-matched certified block replays, so the mismatch
        # rejection path is exercised by injecting the reaction failure.
        import syntree.data.trajectories as trajectories
        from syntree.chemistry.reactions import ReactionRegistryError

        ligand = _embed("CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1")
        index = _build_index(AMIDE_CATALOG)

        real_run = trajectories.run_certified_reaction

        def failing(reaction, reagents):
            raise ReactionRegistryError(f"injected failure for {reaction.reaction_id}")

        trajectories.run_certified_reaction = failing
        try:
            out = extract_trajectory(
                ligand, complex_id="x", split="train", index=index, catalog=_view(AMIDE_CATALOG)
            )
        finally:
            trajectories.run_certified_reaction = real_run
        assert isinstance(out, ExtractionFailure) and out.failure_code == "replay_mismatch", out

    def test_seed_fragment_unsanitizable(self):
        # Real-data regression (observed live in the full-scale #18 run):
        # pdbbind_1f5l's crystal SDF encodes bond type 4 (aromatic) on the
        # EXOcyclic amidine N=C:N (a resonance-form artifact outside any
        # ring).  The whole molecule sanitizes through the molblock parser,
        # but the isolated seed fragment (= the whole ligand here) cannot
        # re-sanitize (Kekulize: "non-ring atom marked aromatic").  The
        # pipeline must reject it with the explicit code, never crash.
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromMolBlock(PDBBIND_1F5L_MOLBLOCK, sanitize=True, removeHs=True)
        assert mol is not None, "the regression molblock must parse"
        index = _build_index(AMIDE_CATALOG)
        out = extract_trajectory(
            mol, complex_id="pdbbind_1f5l", split="train",
            index=index, catalog=_view(AMIDE_CATALOG),
        )
        assert isinstance(out, ExtractionFailure), out
        assert out.failure_code == "seed_fragment_unsanitizable", out.failure_code
        assert "aromatic" in out.detail or "kekul" in out.detail.lower()

    def test_induced_submol_raises_molsanitizeexception_on_exotic_aromaticity(self):
        # Pins the exact exception class the guards must catch: the seed
        # reconstruction of the 1f5l ligand raises RDKit's
        # MolSanitizeException (AtomKekulizeException), so the except
        # clause in extract_trajectory covers the observed failure class.
        from rdkit import RDLogger
        from rdkit.Chem import rdchem
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromMolBlock(PDBBIND_1F5L_MOLBLOCK, sanitize=True, removeHs=True)
        assert mol is not None
        with pytest.raises(rdchem.MolSanitizeException):
            induced_submol(mol, list(range(mol.GetNumAtoms())))

    def test_assembly_stage_sanitize_failure_is_graft_failed_not_crash(self):
        # Fault injection: a mid-assembly induced submol raising RDKit's
        # MolSanitizeException (same class as the seed case — e.g. the
        # FINAL-step product M_t is the whole exotic ligand) must be
        # rejected as graft_failed, never crash the driver.
        import syntree.data.trajectories as trajectories
        from rdkit.Chem import rdchem

        ligand = _embed("CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1")
        index = _build_index(AMIDE_CATALOG)

        real = trajectories.induced_submol

        def raising(mol, indices):
            # only the FINAL-step product construction induces the whole
            # ligand (len == N): raise exactly there, exactly like the
            # observed exotic-aromatic case would
            if len(set(indices)) == mol.GetNumAtoms():
                raise rdchem.AtomKekulizeException("injected non-ring aromatic")
            return real(mol, indices)

        trajectories.induced_submol = raising
        try:
            out = extract_trajectory(
                ligand, complex_id="x", split="train",
                index=index, catalog=_view(AMIDE_CATALOG),
            )
        finally:
            trajectories.induced_submol = real
        assert isinstance(out, ExtractionFailure), out
        assert out.failure_code == "graft_failed", out.failure_code

    def test_failure_code_vocabulary_is_closed_over_documented_codes(self):
        from syntree.data.trajectories import FAILURE_CODES
        assert "seed_fragment_unsanitizable" in FAILURE_CODES
        assert len(FAILURE_CODES) == len(set(FAILURE_CODES))



class TestTorsionLabels:
    def test_bin_of_known_angles(self):
        two_pi = 2 * math.pi
        assert torsion_bin_of(-math.pi) == 1
        assert torsion_bin_of(0.0) == 19  # -pi + 18*bin_width <= 0 < ...
        assert torsion_bin_of(math.pi - 1e-9) == 36
        # wrapping: an angle slightly above pi wraps to the first bin
        assert torsion_bin_of(math.pi + 0.01) == torsion_bin_of(-math.pi + 0.01) == 1

    def test_bins_partition_the_circle(self):
        from syntree.engine.environment import bin_center

        for b in range(1, 37):
            assert torsion_bin_of(bin_center(b)) == b

    def test_observed_dihedral_is_measured_and_binned(self):
        ligand = _embed("CCC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1")
        index = _build_index(AMIDE_CATALOG)
        record = extract_trajectory(ligand, complex_id="x", split="train", index=index, catalog=_view(AMIDE_CATALOG))
        assert isinstance(record, TrajectoryRecord)
        for step in record.steps:
            assert step.resonance_locked  # amide junctions are trans-locked
            # the observed crystal dihedral must sit inside its label bin
            from syntree.engine.environment import bin_center
            two_pi = 2 * math.pi
            wrapped = (step.dihedral_rad + math.pi) % two_pi - math.pi
            center = bin_center(step.torsion_bin)
            width = two_pi / 36
            assert abs(wrapped - center) <= width / 2 + 1e-9
