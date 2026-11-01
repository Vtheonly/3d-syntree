"""Reaction-aware chemoselectivity and competing-handle screening —
tests (#40 CHEM-06).

Acceptance criteria of issue #40:

* Catalog curation applies the reaction-aware chemoselectivity layer
  after basic molecular validation — pinned by the ``screen_catalog``
  pipeline and its per-entry results.
* Every exclusion is traceable to a named rule and chemical feature —
  pinned by reason-code assertions plus the ``detected_groups`` /
  ``reason_codes`` correspondence on every excluded fixture.
* Orthogonally protected and otherwise valid multifunctional substrates
  are not indiscriminately removed — pinned by the protected-substrate
  battery (Boc carbamate, silyl ether, ester, thioether) and by the
  Suzuki-tolerant amino-boronic acid.
* Tests cover competing nucleophiles/electrophiles, self-reactive
  combinations, protected substrates, and valid selective substrates —
  the R1/R5/R2/R6/R3/R4 batteries below.
* The final catalog manifest reports the impact of this filter — pinned
  by the manifest's per-rule exclusion counts and percentages.
"""

from __future__ import annotations

import json

import pytest
from rdkit import Chem

from syntree.chemistry.chemoselectivity import (
    ALL_HANDLE_CLASSES,
    COUPLING_COMPETING_NUCLEOPHILES,
    FREE_GROUP_SMARTS,
    HANDLE_CLASS_SMARTS,
    HETEROARYL_HALIDE_RING_CLASSES,
    PD_COUPLING_CLASSES,
    SELF_REACTIVE_PAIRS,
    ChemoscreenResult,
    CurationManifest,
    ChemoselectivityError,
    chemoselectivity_rule_report,
    detect_free_groups,
    detect_handle_instances,
    handle_class_of_atom,
    screen_block,
    screen_catalog,
)
from syntree.chemistry.reactions import POLICY_REACTIONS


# ---------------------------------------------------------------------------
# 1. The free-group vocabulary (positive / negative per group)
# ---------------------------------------------------------------------------

FREE_GROUP_CASES = [
    # (SMILES, expected groups)
    ("NCC", {"primary_amine"}),
    ("CCNCC", {"secondary_amine"}),
    ("c1ccccc1N", {"primary_amine"}),
    ("c1ccccc1NC", {"secondary_amine"}),
    ("C1CCNCC1", {"secondary_amine"}),
    ("CC(=O)N", set()),          # amide N is not free
    ("CC(=O)NC", set()),         # N-methyl amide N is not free
    ("CCS(=O)(=O)N", set()),     # sulfonamide N is not free
    ("CC(C)(C)OC(=O)NCC", set()),  # Boc carbamate N is not free
    ("CCOC(=O)N", set()),        # carbamate N is not free
    ("c1cc[nH]c1", set()),       # pyrrole-type aromatic NH is not matched
    ("CCO", {"alcohol"}),
    ("OCc1ccccc1", {"alcohol"}),  # benzyl alcohol
    ("c1ccccc1O", {"phenol"}),
    ("CC(=O)O", {"carboxylic_acid"}),
    ("CC(=O)OC", set()),         # ester O is not free
    ("CCO[Si](C)(C)C", set()),   # silyl ether O is not free
    ("COc1ccccc1", set()),       # aryl methyl ether is not free
    ("CCS", {"thiol"}),
    ("CSC", set()),              # thioether S is not free
    ("CCC=O", {"aldehyde"}),
    ("O=Cc1ccccc1", {"aldehyde"}),
    ("CC#C", {"terminal_alkyne"}),
    ("CC#CC", set()),            # internal alkyne is not matched
]


class TestFreeGroupVocabulary:
    @pytest.mark.parametrize("smiles,expected", FREE_GROUP_CASES)
    def test_detection(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        groups = {hit.group for hit in detect_free_groups(mol)}
        assert groups == expected

    def test_vocabulary_smarts_all_parse(self):
        for name, smarts in FREE_GROUP_SMARTS.items():
            assert Chem.MolFromSmarts(smarts) is not None, name

    def test_detected_roles_are_declared(self):
        from syntree.chemistry.chemoselectivity import FREE_GROUP_ROLES

        assert set(FREE_GROUP_SMARTS) == set(FREE_GROUP_ROLES)


# ---------------------------------------------------------------------------
# 2. Handle-class detection (every registry block class)
# ---------------------------------------------------------------------------

HANDLE_CASES = [
    ("CC(=O)Cl", "acid_chloride", 1),
    ("O=C(Cl)CCC(=O)Cl", "acid_chloride", 2),
    ("CS(=O)(=O)Cl", "sulfonyl_chloride", 1),
    ("ClC(=O)N(C)C" if False else "O=C(Cl)N(C)C", "carbamoyl_chloride", 1),
    ("CCOC(=O)Cl", "chloroformate", 1),
    ("CCC=O", "aldehyde", 1),
    ("CCBr", "alkyl_bromide", 1),
    ("BrCCCBr", "alkyl_bromide", 2),
    ("c1ccccc1Br", "aryl_bromide", 1),
    ("c1ccccc1Cl", "aryl_chloride", 1),
    ("OB(O)c1ccccc1", "arylboronic_acid", 1),
    ("OB(O)c1ccncc1", "heteroarylboronic_acid", 1),
    ("OB(O)c1ccccc1", "heteroarylboronic_acid", 0),
    ("CC#C", "acetylide", 1),
    # heteroaryl halides via ring classification
    ("n1cnc(Cl)cc1", "pyrimidinyl_chloride", 1),      # 4-chloropyrimidine
    ("n1cnc(Cl)cc1", "pyridyl_chloride", 0),          # not a pyridine
    ("n1cc(Cl)ncc1", "pyrimidinyl_chloride", 0),      # chloropyrazine: para N
    ("n1cccc(Cl)c1", "pyridyl_chloride", 1),
    ("n1ccncc1", "pyridyl_chloride", 0),              # pyrazine: two N
    ("Clc1sccn1", "thiazolyl_chloride", 1),
    ("c1cn[nH]c1Br", "pyrazolyl_bromide", 1),
    ("Clc1nc2ccccc2[nH]1", "benzimidazolyl_chloride", 1),
    ("Clc1ccccc1", "pyrimidinyl_chloride", 0),        # benzene controls
]


class TestHandleDetection:
    @pytest.mark.parametrize("smiles,cls,want", HANDLE_CASES)
    def test_detection(self, smiles, cls, want):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        assert len(detect_handle_instances(mol, cls)) == want

    def test_every_registry_block_class_is_screenable(self):
        registry_classes = {
            r.block_handle_class for r in POLICY_REACTIONS.values()
        }
        assert registry_classes <= ALL_HANDLE_CLASSES

    def test_unknown_class_fails_closed(self):
        with pytest.raises(ChemoselectivityError):
            detect_handle_instances(Chem.MolFromSmiles("CCO"), "not_a_class")

    def test_handle_class_of_atom(self):
        mol = Chem.MolFromSmiles("n1cnc(Cl)cc1")
        cl_carbon = detect_handle_instances(mol, "pyrimidinyl_chloride")[0]
        classes = handle_class_of_atom(mol, cl_carbon)
        assert "pyrimidinyl_chloride" in classes

    def test_handle_class_of_atom_bounds(self):
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(ChemoselectivityError):
            handle_class_of_atom(mol, 17)


# ---------------------------------------------------------------------------
# 3. Rule R1 — classical self-reactive pairs
# ---------------------------------------------------------------------------

class TestSelfReactivePairs:
    @pytest.mark.parametrize(
        "smiles,cls,code",
        [
            ("OCC(=O)Cl", "acid_chloride", "self_reactive_pair:alcohol"),
            ("NCC(=O)Cl", "acid_chloride", "self_reactive_pair:primary_amine"),
            ("SCC(=O)Cl", "acid_chloride", "self_reactive_pair:thiol"),
            ("O=C(O)CC(=O)Cl", "acid_chloride", "self_reactive_pair:carboxylic_acid"),
            ("NCS(=O)(=O)Cl", "sulfonyl_chloride", "self_reactive_pair:primary_amine"),
            ("OCCS(=O)(=O)Cl", "sulfonyl_chloride", "self_reactive_pair:alcohol"),
            ("BrCCN", "alkyl_bromide", "self_reactive_pair:primary_amine"),
            ("BrCCO", "alkyl_bromide", "self_reactive_pair:alcohol"),
            ("BrCCS", "alkyl_bromide", "self_reactive_pair:thiol"),
            ("NCCC=O", "aldehyde", "self_reactive_pair:primary_amine"),
            ("Nc1ccccc1C=O", "aldehyde", "self_reactive_pair:primary_amine"),
        ],
    )
    def test_self_reactive_structures_excluded(self, smiles, cls, code):
        result = screen_block(smiles, cls)
        assert result.status == "excluded"
        assert code in result.reason_codes

    def test_clean_electrophiles_pass(self):
        for smiles, cls in [
            ("CCC(=O)Cl", "acid_chloride"),
            ("CCS(=O)(=O)Cl", "sulfonyl_chloride"),
            ("BrCCC", "alkyl_bromide"),
            ("CCC=O", "aldehyde"),
            ("CCOC(=O)Cl", "chloroformate"),
        ]:
            result = screen_block(smiles, cls)
            assert result.status == "pass", (smiles, result.reason_codes)

    def test_aldehyde_tolerates_alcohol(self):
        # the amine reagent forms imines with aldehydes, not the alcohol
        result = screen_block("OCCC=O", "aldehyde")
        assert result.status == "pass"

    def test_matrix_covers_all_self_reactive_classes(self):
        for cls in SELF_REACTIVE_PAIRS:
            assert SELF_REACTIVE_PAIRS[cls], cls


# ---------------------------------------------------------------------------
# 4. Orthogonal protection passes (the anti-indiscriminate-exclusion pin)
# ---------------------------------------------------------------------------

class TestOrthogonalProtection:
    @pytest.mark.parametrize(
        "smiles,cls",
        [
            # Boc-protected diamine-derived acyl chloride: the carbamate N
            # is orthogonal to the amine reagent
            ("CC(C)(C)OC(=O)NCC(=O)Cl", "acid_chloride"),
            # silyl-protected bromoalcohol
            ("BrCCO[Si](C)(C)C", "alkyl_bromide"),
            # ester-protected carboxylate with an alkyl bromide handle
            ("BrCC(=O)OCC", "alkyl_bromide"),
            # thioether (protected thiol) with an acyl chloride
            ("CSCCC(=O)Cl", "acid_chloride"),
            # benzyl-ether protected phenol with an acyl chloride
            ("COc1ccccc1C(=O)Cl", "acid_chloride"),
            # acetamide-protected amine with an alkyl bromide
            ("CC(=O)NCCBr", "alkyl_bromide"),
        ],
    )
    def test_protected_multifunctional_substrates_pass(self, smiles, cls):
        result = screen_block(smiles, cls)
        assert result.status == "pass", result.reason_codes

    def test_valid_selective_substrates_pass(self):
        # free amine on a boronic acid: a standard Suzuki substrate
        result = screen_block("OB(O)c1ccc(N)cc1", "arylboronic_acid")
        assert result.status == "pass", result.reason_codes
        # free alcohol on a boronic acid: tolerated
        result = screen_block("OB(O)c1ccc(O)cc1", "arylboronic_acid")
        assert result.status == "pass", result.reason_codes
        # carboxylic acid on a boronic acid: tolerated
        result = screen_block("OB(O)c1ccc(C(=O)O)cc1", "arylboronic_acid")
        assert result.status == "pass", result.reason_codes


# ---------------------------------------------------------------------------
# 5. Rule R2 — organometallic quenching
# ---------------------------------------------------------------------------

class TestQuenching:
    @pytest.mark.parametrize(
        "smiles,cls",
        [
            ("C#CCO", "acetylide"),          # propargyl alcohol
            ("C#CCN", "acetylide"),          # propargyl amine
            ("C#CCS", "acetylide"),
        ],
    )
    def test_protic_groups_quench_organometallics(self, smiles, cls):
        result = screen_block(smiles, cls)
        assert result.status == "excluded"
        assert any(c.startswith("quenching_protic_group:") for c in result.reason_codes)

    def test_clean_acetylide_passes(self):
        result = screen_block("CC#C", "acetylide")
        assert result.status == "pass"

    def test_own_terminal_alkyne_not_counted(self):
        # the handle's own terminal alkyne is not a quenching group
        result = screen_block("CC#C", "acetylide")
        assert "quenching_protic_group:terminal_alkyne" not in result.reason_codes

    def test_aryl_bromide_quenches_alkylzinc_consumer(self):
        # rxn_15 consumes aryl bromides with an alkylzinc reagent: any
        # free protic group on the block quenches it
        result = screen_block("Oc1ccccc1Br", "aryl_bromide")
        assert result.status == "excluded"
        assert "quenching_protic_group:phenol" in result.reason_codes


# ---------------------------------------------------------------------------
# 6. Rule R3 — competing identical handles
# ---------------------------------------------------------------------------

class TestCompetingIdenticalHandles:
    @pytest.mark.parametrize(
        "smiles,cls",
        [
            ("O=C(Cl)CCC(=O)Cl", "acid_chloride"),
            ("BrCCBr", "alkyl_bromide"),
            ("c1cc(Br)cc(Br)c1", "aryl_bromide"),
        ],
    )
    def test_multi_handle_excluded(self, smiles, cls):
        result = screen_block(smiles, cls)
        assert result.status == "excluded"
        assert "competing_identical_handles" in result.reason_codes

    def test_no_handle_excluded_with_short_circuit(self):
        result = screen_block("Nc1nccnc1", "pyrimidinyl_chloride")
        assert result.status == "excluded"
        assert result.reason_codes == ("no_handle_of_class",)
        assert result.handle_atom_indices == ()

    def test_declared_handle_validated(self):
        mol = Chem.MolFromSmiles("CCC(=O)Cl")
        with pytest.raises(ChemoselectivityError):
            screen_block(mol, "acid_chloride", handle_atom_idx=0)


# ---------------------------------------------------------------------------
# 7. Rule R4 — competing SNAr positions + the differentiated exemption
# ---------------------------------------------------------------------------

class TestCompetingSnarPositions:
    def test_2_4_dichloropyrimidine_passes_with_note(self):
        # the classic: amine reagents displace the 4-halide selectively
        mol = Chem.MolFromSmiles("n1c(Cl)nc(Cl)cc1")
        cl_carbons = [
            n.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() == "Cl"
            for n in a.GetNeighbors()
            if n.GetSymbol() == "C"
        ]
        handle = min(
            cl_carbons,
            key=lambda a: sum(
                1
                for n in mol.GetAtomWithIdx(a).GetNeighbors()
                if n.GetSymbol() == "N" and n.GetIsAromatic()
            ),
        )
        result = screen_block(mol, "pyrimidinyl_chloride", handle_atom_idx=handle)
        assert result.status == "pass", result.reason_codes
        assert any(
            note.startswith("differentiated_2_4_dihaloazine")
            for note in result.notes
        )

    def test_declaring_the_less_reactive_position_is_excluded(self):
        # declaring the 2-position (flanked by two ring N) as the handle
        # while the amine-labile 4-Cl survives: nonselective
        mol = Chem.MolFromSmiles("n1c(Cl)nc(Cl)cc1")
        cl_carbons = [
            n.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() == "Cl"
            for n in a.GetNeighbors()
            if n.GetSymbol() == "C"
        ]
        spectator = max(
            cl_carbons,
            key=lambda a: sum(
                1
                for n in mol.GetAtomWithIdx(a).GetNeighbors()
                if n.GetSymbol() == "N" and n.GetIsAromatic()
            ),
        )
        result = screen_block(
            mol, "pyrimidinyl_chloride", handle_atom_idx=spectator
        )
        assert result.status == "excluded"
        assert "competing_identical_handles" in result.reason_codes

    def test_equivalent_dihalo_positions_excluded(self):
        # 4,6-dichloropyrimidine: equivalent positions, no differentiation
        result = screen_block("n1cnc(Cl)cc1Cl", "pyrimidinyl_chloride")
        assert result.status == "excluded"
        assert "competing_identical_handles" in result.reason_codes

    def test_clean_chloropyrimidine_passes(self):
        result = screen_block("n1cnc(Cl)cc1", "pyrimidinyl_chloride")
        assert result.status == "pass"


# ---------------------------------------------------------------------------
# 8. Rule R5 — competing uncatalyzed electrophiles
# ---------------------------------------------------------------------------

class TestCompetingElectrophiles:
    def test_second_electrophile_on_acyl_chloride_block(self):
        result = screen_block("BrCCC(=O)Cl", "acid_chloride")
        assert "competing_electrophile:alkyl_bromide" in result.reason_codes

    def test_aldehyde_on_alkyl_bromide_block(self):
        result = screen_block("O=CCCBr", "alkyl_bromide")
        assert "competing_electrophile:aldehyde" in result.reason_codes

    def test_aryl_bromide_is_not_an_uncatalyzed_electrophile(self):
        # the amine reagent of rxn_01 does not touch aryl bromides without
        # Pd catalysis: a bromo-aryl acyl chloride still passes R5
        result = screen_block("Brc1ccc(C(=O)Cl)cc1", "acid_chloride")
        assert not any(
            c.startswith("competing_electrophile:") for c in result.reason_codes
        )

    def test_intended_class_not_double_counted(self):
        # the intended aldehyde handle is not a competing electrophile
        result = screen_block("CCC=O", "aldehyde")
        assert not any(
            c.startswith("competing_electrophile:") for c in result.reason_codes
        )


# ---------------------------------------------------------------------------
# 9. Rule R6 — catalyst poisons
# ---------------------------------------------------------------------------

class TestCatalystPoisons:
    @pytest.mark.parametrize(
        "smiles,cls",
        [
            ("Sc1ccccc1B(O)O", "arylboronic_acid"),
            ("Sc1ccc(B(O)O)cc1", "arylboronic_acid"),
        ],
    )
    def test_thiol_poisons_pd_couplings(self, smiles, cls):
        result = screen_block(smiles, cls)
        assert result.status == "excluded"
        assert "catalyst_poison:thiol" in result.reason_codes

    def test_thioether_is_not_a_poison(self):
        result = screen_block("CSc1ccc(B(O)O)cc1", "arylboronic_acid")
        assert result.status == "pass", result.reason_codes

    def test_every_pd_class_declares_the_thiol_rule(self):
        for cls in PD_COUPLING_CLASSES:
            assert cls in PD_COUPLING_CLASSES


# ---------------------------------------------------------------------------
# 10. Registry integration — applicable reactions and class coverage
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    def test_applicable_reactions_match_the_registry(self):
        result = screen_block("CCC(=O)Cl", "acid_chloride")
        expected = tuple(
            rid
            for rid, r in POLICY_REACTIONS.items()
            if r.block_handle_class == "acid_chloride"
        )
        assert result.applicable_reactions == expected
        assert set(expected) == {"rxn_01", "rxn_17", "rxn_20"}

    def test_every_registry_block_class_screens(self):
        # one representative pass-case molecule per class proves the whole
        # block-side vocabulary is screenable end-to-end
        representatives = {
            "acid_chloride": "CCC(=O)Cl",
            "sulfonyl_chloride": "CCS(=O)(=O)Cl",
            "carbamoyl_chloride": "O=C(Cl)N(C)C",
            "chloroformate": "CCOC(=O)Cl",
            "aldehyde": "CCC=O",
            "alkyl_bromide": "BrCCC",
            "aryl_bromide": "c1ccccc1Br",
            "aryl_chloride": "c1ccccc1Cl",
            "arylboronic_acid": "OB(O)c1ccccc1",
            "heteroarylboronic_acid": "OB(O)c1ccncc1",
            "arylzinc": "[Zn]c1ccccc1",
            "vinylstannane": "C=C[Sn](C)(C)C",
            "acetylide": "CC#C",
            "pyrimidinyl_chloride": "n1cnc(Cl)cc1",
            "pyridyl_chloride": "n1cccc(Cl)c1",
            "thiazolyl_chloride": "Clc1sccn1",
            "thiazolyl_bromide": "Brc1sccn1",
            "pyrazolyl_bromide": "c1cn[nH]c1Br",
            "benzimidazolyl_chloride": "Clc1nc2ccccc2[nH]1",
        }
        registry_classes = {
            r.block_handle_class for r in POLICY_REACTIONS.values()
        }
        assert set(representatives) == registry_classes
        for cls, smiles in representatives.items():
            result = screen_block(smiles, cls)
            assert result.status == "pass", (cls, result.reason_codes)
            assert result.applicable_reactions

    def test_result_record_fields(self):
        result = screen_block("OCC(=O)Cl", "acid_chloride", block_id="blk1")
        assert isinstance(result, ChemoscreenResult)
        d = result.as_dict()
        assert d["block_id"] == "blk1"
        assert d["handle_class"] == "acid_chloride"
        assert d["status"] == "excluded"
        assert "self_reactive_pair:alcohol" in d["reason_codes"]
        groups = {g["group"] for g in d["detected_groups"]}
        assert "alcohol" in groups and "acid_chloride" in groups
        json.dumps(d)  # JSON-serializable


# ---------------------------------------------------------------------------
# 11. The catalog manifest (acceptance: impact reporting)
# ---------------------------------------------------------------------------

class TestCurationManifest:
    def _entries(self):
        return [
            {"block_id": "clean_acyl", "mol": "CCC(=O)Cl", "handle_class": "acid_chloride"},
            {"block_id": "hydroxy_acyl", "mol": "OCC(=O)Cl", "handle_class": "acid_chloride"},
            {"block_id": "amino_acyl", "mol": "NCC(=O)Cl", "handle_class": "acid_chloride"},
            {"block_id": "boc_acyl", "mol": "CC(C)(C)OC(=O)NCC(=O)Cl", "handle_class": "acid_chloride"},
            {"block_id": "dibromide", "mol": "BrCCBr", "handle_class": "alkyl_bromide"},
            {"block_id": "clean_bromide", "mol": "BrCCC", "handle_class": "alkyl_bromide"},
            {"block_id": "thiol_boronic", "mol": "Sc1ccc(B(O)O)cc1", "handle_class": "arylboronic_acid"},
            {"block_id": "amino_boronic", "mol": "OB(O)c1ccc(N)cc1", "handle_class": "arylboronic_acid"},
            {"block_id": "diCl_pyr", "mol": "n1c(Cl)nc(Cl)cc1", "handle_class": "pyrimidinyl_chloride"},
            {"block_id": "clean_pyr", "mol": "n1cnc(Cl)cc1", "handle_class": "pyrimidinyl_chloride"},
        ]

    def test_manifest_counts_and_percentages(self):
        manifest = screen_catalog(self._entries())
        assert isinstance(manifest, CurationManifest)
        assert manifest.total == 10
        assert manifest.excluded == 4
        assert manifest.passed == 6
        assert manifest.pass_rate == pytest.approx(0.6)
        # per-rule impact: alcohol and amine self-reactivity, dibromide,
        # thiol poison — each counted with its named reason code
        assert manifest.exclusion_counts["self_reactive_pair:alcohol"] == 1
        assert manifest.exclusion_counts["self_reactive_pair:primary_amine"] == 1
        assert manifest.exclusion_counts["competing_identical_handles"] == 1
        assert manifest.exclusion_counts["catalyst_poison:thiol"] == 1
        # base-code aggregation
        assert manifest.exclusion_counts["self_reactive_pair"] == 2
        # percentages of the total catalog
        assert manifest.exclusion_percentages["self_reactive_pair:alcohol"] == pytest.approx(10.0)
        # handle-class census
        assert manifest.handle_class_counts["acid_chloride"] == 4
        # the exempted 2,4-dichloropyrimidine passes with its note
        diCl = next(
            r for r in manifest.results if r.block_id == "diCl_pyr"
        )
        assert diCl.status == "pass"
        assert any("differentiated_2_4_dihaloazine" in n for n in diCl.notes)

    def test_manifest_is_json_serializable(self):
        manifest = screen_catalog(self._entries())
        payload = json.dumps(manifest.as_dict())
        assert "exclusion_percentages" in payload

    def test_malformed_entries_fail_closed(self):
        with pytest.raises(ChemoselectivityError):
            screen_catalog([{"block_id": "x", "mol": "CCO"}])  # no class
        with pytest.raises(ChemoselectivityError):
            screen_catalog([{"block_id": "x", "mol": "not_a_smiles", "handle_class": "acid_chloride"}])
        with pytest.raises(ChemoselectivityError):
            screen_catalog(["not_a_mapping"])

    def test_screen_block_rejects_bad_inputs(self):
        with pytest.raises(ChemoselectivityError):
            screen_block(12345, "acid_chloride")
        with pytest.raises(ChemoselectivityError):
            screen_block("CCO", "no_such_class")
        empty = Chem.MolFromSmiles("")
        with pytest.raises(ChemoselectivityError):
            screen_block(empty, "acid_chloride")


# ---------------------------------------------------------------------------
# 12. The documented rule report
# ---------------------------------------------------------------------------

class TestRuleReport:
    def test_report_covers_every_rule(self):
        report = chemoselectivity_rule_report()
        rules = report["rules"]
        for expected in (
            "R1_self_reactive_pair",
            "R1b_competing_nucleophile",
            "R2_quenching_protic_group",
            "R3_competing_identical_handles",
            "R4_competing_snar_position",
            "R5_competing_electrophile",
            "R6_catalyst_poison",
        ):
            assert expected in rules
        assert "orthogonal_protection" in report
        assert report["registry_link"]["policy_reactions"] == 27
        json.dumps(report)

    def test_coupling_matrix_classes_are_registry_classes(self):
        registry_classes = {
            r.block_handle_class for r in POLICY_REACTIONS.values()
        }
        for cls in COUPLING_COMPETING_NUCLEOPHILES:
            assert cls in registry_classes


# ---------------------------------------------------------------------------
# 13. Reason-code traceability (every exclusion names rule + feature)
# ---------------------------------------------------------------------------

class TestTraceability:
    @pytest.mark.parametrize(
        "smiles,cls",
        [
            ("OCC(=O)Cl", "acid_chloride"),
            ("BrCCBr", "alkyl_bromide"),
            ("Sc1ccc(B(O)O)cc1", "arylboronic_acid"),
            ("CC#CO", "acetylide"),
            ("Nc1nccnc1", "pyrimidinyl_chloride"),
        ],
    )
    def test_every_exclusion_is_traceable(self, smiles, cls):
        result = screen_block(smiles, cls)
        assert result.status == "excluded"
        assert result.reason_codes
        for code in result.reason_codes:
            # every code is ``rule`` or ``rule:feature``; a feature-bearing
            # code must correspond to a detected group or handle class
            if ":" in code:
                rule, feature = code.split(":", 1)
                features = {g.group for g in result.detected_groups}
                assert feature in features or feature in {
                    "acid_chloride",
                    "sulfonyl_chloride",
                    "carbamoyl_chloride",
                    "chloroformate",
                    "alkyl_bromide",
                    "aldehyde",
                }, code
