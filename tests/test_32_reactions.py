"""Verification suite of the certified 32-reaction taxonomy (#11 CHEM-01).

Pinned by issue #11's acceptance criteria and constitution #45 §9.1:

1. :func:`TestTaxonomy` — the issue-pinned shape: 32 reactions
   (8 C-N + 7 C-C + 6 C-O/C-S + 6 heterocycle + 5 deprotections), the 9
   coarse families with exact membership (rxn_14/rxn_15 in
   ``cross_coupling_cc`` per D-V2-016), and the complete #45 §9.1
   metadata (handle classes, leaving groups, stereo behaviour, validity
   constraints, failure behaviour, references, junction patterns).
2. :func:`TestForwardReplay` — **100% round-trip forward-replay
   validation on standardized reagent pairs in RDKit** for every one of
   the 32 reactions (the #11 acceptance criterion 1), including all 13
   stereo pairs.
3. :func:`TestStereoRetention` — chiral centres adjacent to newly formed
   bonds retain their tetrahedral configuration (the #11 acceptance
   criterion 2): explicit CIP-code comparison reagent-vs-product.
4. :func:`TestDeprotections` — deterministic, policy-free execution of
   rxn_28..rxn_32; absence is a no-op; the input is never mutated.
5. :func:`TestEngineIntegration` — the certified registry drives the
   real Assembly MDP end to end (product graphs, resonance locks,
   deterministic trans snap) and ``DEFAULT_REACTION_REGISTRY`` *is* the
   certified taxonomy (the generic interim was superseded, D-V2-016),
   with the Trajectory Balance engine consuming certified-registry
   trajectories (the MDP/TB contracts of D-V2-009 are unchanged).
"""

from __future__ import annotations

import math

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetDihedralRad

from syntree.chemistry.reactions import (
    CERTIFIED_REACTIONS,
    DEPROTECTIONS,
    POLICY_REACTIONS,
    REACTION_CATEGORIES,
    REACTION_FAMILIES,
    ReactionRegistryError,
    apply_all_deprotections,
    apply_deprotection,
    build_junction_reaction_registry,
    canonical_product_smiles,
    certify_reaction,
    certify_registry,
    run_certified_reaction,
)
from syntree.engine.environment import (
    AssemblyAction,
    AssemblyEnvironment,
    AssemblyError,
    BuildingBlock,
    DEFAULT_REACTION_REGISTRY,
    HandleSpec,
    PocketDefinition,
    _read_junctions,
    bin_center,
    wrap_angle,
)
from syntree.engine.gflownet import TrajectoryBalanceEngine, as_trajectory_steps


def _junctions_of(mol: Chem.Mol):
    return _read_junctions(mol)

# ---------------------------------------------------------------------------
# Fixtures: catalog blocks whose handles carry the certified classes
# ---------------------------------------------------------------------------


def _block(block_id: str, smiles: str, handle_class: str, leaving: bool = True) -> BuildingBlock:
    """A one-handle block; the attachment atom is located from the leaving
    atom (halogen/O) or is the hetero atom itself (amine/alcohol/thiol)."""
    template = Chem.MolFromSmiles(smiles)
    assert template is not None
    atom_idx = None
    leaving_atoms: tuple = ()
    if leaving:
        # attachment = the neighbour of the highest-numbered leaving atom
        # (halogen, acid O, B); leaving set = that atom alone or B+2O.
        if handle_class in ("arylboronic_acid", "heteroarylboronic_acid"):
            boron = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 5)
            atom_idx = next(n.GetIdx() for n in template.GetAtomWithIdx(boron).GetNeighbors() if n.GetAtomicNum() == 6)
            leaving_atoms = (
                boron,
                *(n.GetIdx() for n in template.GetAtomWithIdx(boron).GetNeighbors() if n.GetAtomicNum() == 8),
            )
        elif handle_class == "aldehyde":
            oxygen = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 8)
            atom_idx = next(n.GetIdx() for n in template.GetAtomWithIdx(oxygen).GetNeighbors())
            leaving_atoms = (oxygen,)
        elif handle_class == "acetylide":
            lithium = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 3)
            carbon = next(n.GetIdx() for n in template.GetAtomWithIdx(lithium).GetNeighbors())
            atom_idx = carbon
            leaving_atoms = (lithium,)
        elif handle_class == "alkylzinc":
            zinc = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 30)
            atom_idx = next(n.GetIdx() for n in template.GetAtomWithIdx(zinc).GetNeighbors() if n.GetAtomicNum() == 6)
            leaving_atoms = (zinc,)
        else:
            halogen = max(
                (a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() in (17, 35, 8) and handle_class.endswith(("chloride", "bromide"))),
                default=None,
            )
            if halogen is None:
                # acid/sulfonyl chlorides, chloroformates, carbamoyl chloride
                chlorine = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == 17)
                anchor = template.GetAtomWithIdx(chlorine).GetNeighbors()[0]
                atom_idx = anchor.GetIdx()
                leaving_atoms = (chlorine,)
            else:
                anchor = template.GetAtomWithIdx(halogen).GetNeighbors()[0]
                atom_idx = anchor.GetIdx()
                leaving_atoms = (halogen,)
    else:
        hetero = {"amine": 7, "alcohol": 8, "thiol": 16}[handle_class]
        atom_idx = next(a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum() == hetero)
        leaving_atoms = ()
    return BuildingBlock(block_id, smiles, (HandleSpec(f"h{atom_idx}", atom_idx, handle_class, leaving_atoms),))


CATALOG = {
    "acid_chloride": _block("acid_chloride", "CC(=O)Cl", "acid_chloride"),
    "sulfonyl_chloride": _block("sulfonyl_chloride", "CS(=O)(=O)Cl", "sulfonyl_chloride"),
    "aldehyde": _block("aldehyde", "CC=O", "aldehyde"),
    "aryl_bromide": _block("aryl_bromide", "Brc1ccccc1", "aryl_bromide"),
    "aryl_chloride": _block("aryl_chloride", "Clc1ccccc1", "aryl_chloride"),
    "carbamoyl_chloride": _block("carbamoyl_chloride", "CN(C)C(=O)Cl", "carbamoyl_chloride"),
    "alkyl_bromide": _block("alkyl_bromide", "CCBr", "alkyl_bromide"),
    "chloroformate": _block("chloroformate", "COC(=O)Cl", "chloroformate"),
    "arylboronic_acid": _block("arylboronic_acid", "OB(O)c1ccccc1", "arylboronic_acid"),
    "heteroarylboronic_acid": _block("heteroarylboronic_acid", "OB(O)c1cccnc1", "heteroarylboronic_acid"),
    "vinyl_bromide": _block("vinyl_bromide", "C=CBr", "vinyl_bromide"),
    "pyrimidinyl_chloride": _block("pyrimidinyl_chloride", "Clc1ncccn1", "pyrimidinyl_chloride"),
    "pyridyl_chloride": _block("pyridyl_chloride", "Clc1ccccn1", "pyridyl_chloride"),
    "thiazolyl_chloride": _block("thiazolyl_chloride", "Clc1nccs1", "thiazolyl_chloride"),
    "thiazolyl_bromide": _block("thiazolyl_bromide", "Brc1nccs1", "thiazolyl_bromide"),
    "benzimidazolyl_chloride": _block("benzimidazolyl_chloride", "Clc1nc2ccccc2[nH]1", "benzimidazolyl_chloride"),
    "pyrazolyl_bromide": _block("pyrazolyl_bromide", "Cn1ncc(Br)c1", "pyrazolyl_bromide"),
    "alcohol": _block("alcohol", "CCO", "alcohol", leaving=False),
    "thiol": _block("thiol", "CCS", "thiol", leaving=False),
    "amine": _block("amine", "CN", "amine", leaving=False),
}


def _seed(block_id: str, smiles: str, handle_class: str, leaving: bool | None = None) -> BuildingBlock:
    if leaving is None:
        leaving = handle_class not in ("amine", "alcohol", "thiol")
    return _block(block_id, smiles, handle_class, leaving=leaving)


# ---------------------------------------------------------------------------
# 1. Taxonomy shape (issue-pinned)
# ---------------------------------------------------------------------------


class TestTaxonomy:
    def test_counts(self):
        assert len(CERTIFIED_REACTIONS) == 32
        assert len(POLICY_REACTIONS) == 27
        assert len(DEPROTECTIONS) == 5
        assert set(CERTIFIED_REACTIONS) == {f"rxn_{i:02d}" for i in range(1, 33)}

    def test_categories(self):
        assert REACTION_CATEGORIES["C-N"] == tuple(f"rxn_{i:02d}" for i in range(1, 9))
        assert REACTION_CATEGORIES["C-C"] == tuple(f"rxn_{i:02d}" for i in range(9, 16))
        assert REACTION_CATEGORIES["C-O/C-S"] == tuple(f"rxn_{i:02d}" for i in range(16, 22))
        assert REACTION_CATEGORIES["heterocycle"] == tuple(f"rxn_{i:02d}" for i in range(22, 28))
        assert REACTION_CATEGORIES["deprotection"] == tuple(f"rxn_{i:02d}" for i in range(28, 33))

    def test_families_exact_membership(self):
        # Issue-pinned lists plus the D-V2-016 additions (rxn_14/rxn_15 in
        # cross_coupling_cc, the only C-C family).
        assert REACTION_FAMILIES["amide_and_carbamate"] == ("rxn_01", "rxn_08")
        assert REACTION_FAMILIES["sulfonamide_coupling"] == ("rxn_02",)
        assert REACTION_FAMILIES["reductive_amination"] == ("rxn_03",)
        assert REACTION_FAMILIES["aryl_amination"] == ("rxn_04", "rxn_05")
        assert REACTION_FAMILIES["urea_formation"] == ("rxn_06",)
        assert REACTION_FAMILIES["sp3_alkylation"] == ("rxn_07", "rxn_16", "rxn_19")
        assert REACTION_FAMILIES["cross_coupling_cc"] == (
            "rxn_09", "rxn_10", "rxn_11", "rxn_12", "rxn_13", "rxn_14", "rxn_15",
        )
        assert REACTION_FAMILIES["ester_and_ether"] == ("rxn_17", "rxn_18", "rxn_20", "rxn_21")
        assert REACTION_FAMILIES["heterocycle_click"] == tuple(f"rxn_{i:02d}" for i in range(22, 28))
        assert len(REACTION_FAMILIES) == 9

    def test_resonance_locks(self):
        locked = {rid for rid, r in POLICY_REACTIONS.items() if r.resonance_lock == "trans"}
        assert locked == {"rxn_01", "rxn_06", "rxn_08", "rxn_17", "rxn_20"}
        assert all(r.resonance_lock is None for rid, r in POLICY_REACTIONS.items() if rid not in locked)

    def test_metadata_complete(self):
        for reaction in CERTIFIED_REACTIONS.values():
            assert reaction.reaction_smarts
            assert AllChem.ReactionFromSmarts(reaction.reaction_smarts) is not None
            assert reaction.block_handle_class
            assert reaction.leaving_group_molecule and reaction.leaving_group_block
            assert reaction.stereo_behavior
            assert reaction.reference
            assert reaction.validity_constraints
            assert reaction.failure_behavior
            assert reaction.certification
            if reaction.policy_action:
                assert reaction.molecule_handle_class
                assert reaction.junction_pattern is not None
                assert reaction.junction_priority > 0
                assert AllChem.MolFromSmarts(reaction.junction_pattern) is not None

    def test_deprotections_have_no_policy_action(self):
        for rule in DEPROTECTIONS.values():
            assert rule.policy_action is False
            assert rule.family is None

    def test_heterocycle_provenance_documented(self):
        for rid in REACTION_FAMILIES["heterocycle_click"]:
            assert POLICY_REACTIONS[rid].cyclization_provenance, rid

    def test_junction_patterns_match_their_own_products(self):
        for reaction in POLICY_REACTIONS.values():
            pattern = Chem.MolFromSmarts(reaction.junction_pattern)
            pair = reaction.certification[0]
            products = run_certified_reaction(reaction, pair.reagents)
            assert any(len(p.GetSubstructMatches(pattern)) >= 1 for p in products), (
                f"{reaction.reaction_id}: the junction pattern must recognize the "
                f"product of its own certification pair"
            )


# ---------------------------------------------------------------------------
# 2. 100% round-trip forward replay (acceptance criterion 1)
# ---------------------------------------------------------------------------


class TestForwardReplay:
    def test_certify_all_32(self):
        reports = certify_registry()
        assert len(reports) == 32
        pairs = sum(len(r["pairs"]) for r in reports)
        assert pairs == 48, "48 standardized reagent pairs are certified"
        stereo = sum(1 for r in reports for p in r["pairs"] if p["stereo_pair"])
        assert stereo == 16

    @pytest.mark.parametrize("reaction_id", sorted(CERTIFIED_REACTIONS))
    def test_round_trip_per_reaction(self, reaction_id):
        report = certify_reaction(CERTIFIED_REACTIONS[reaction_id])
        assert report["pairs"], reaction_id

    def test_failure_is_closed(self):
        # A reagent set that does not match must raise, never skip.
        with pytest.raises(ReactionRegistryError):
            run_certified_reaction(POLICY_REACTIONS["rxn_01"], ("CCO", "CC(=O)Cl"))
        with pytest.raises(ReactionRegistryError):
            run_certified_reaction(POLICY_REACTIONS["rxn_09"], ("Brc1ccccc1", "CCO"))
        with pytest.raises(ReactionRegistryError):
            run_certified_reaction(POLICY_REACTIONS["rxn_01"], ("not a smiles", "CC(=O)Cl"))


# ---------------------------------------------------------------------------
# 3. Stereo retention (acceptance criterion 2)
# ---------------------------------------------------------------------------


class TestStereoRetention:
    @pytest.mark.parametrize("reaction_id", sorted(CERTIFIED_REACTIONS))
    def test_cip_retained(self, reaction_id):
        reaction = CERTIFIED_REACTIONS[reaction_id]
        stereo_pairs = [p for p in reaction.certification if p.stereo_pair]
        if not stereo_pairs:
            pytest.skip(f"{reaction_id}: no stereo pair designed")
        for pair in stereo_pairs:
            products = run_certified_reaction(reaction, pair.reagents)
            reagent = Chem.MolFromSmiles(pair.reagents[0])
            Chem.AssignStereochemistry(reagent, cleanIt=True, force=True)
            reagent_cips = {
                a.GetIdx(): a.GetProp("_CIPCode")
                for a in reagent.GetAtoms()
                if a.HasProp("_CIPCode")
            }
            assert reagent_cips, f"{reaction_id}: the stereo pair reagent has no CIP centre"
            product = products[0]
            Chem.AssignStereochemistry(product, cleanIt=True, force=True)
            product_cips = {
                a.GetProp("_CIPCode") for a in product.GetAtoms() if a.HasProp("_CIPCode")
            }
            for code in reagent_cips.values():
                assert code in product_cips, (
                    f"{reaction_id}: CIP {code} lost — the tetrahedral configuration "
                    f"adjacent to the new bond was not retained"
                )


# ---------------------------------------------------------------------------
# 4. Deprotections
# ---------------------------------------------------------------------------


class TestDeprotections:
    @pytest.mark.parametrize("rule_id", sorted(DEPROTECTIONS))
    def test_applies_deterministically(self, rule_id):
        rule = DEPROTECTIONS[rule_id]
        substrate = Chem.MolFromSmiles(rule.certification[0].reagents[0])
        once = apply_deprotection(substrate, rule)
        twice = apply_deprotection(once, rule)  # already deprotected: no-op
        assert canonical_product_smiles(once) == rule.certification[0].expected_product
        assert canonical_product_smiles(twice) == rule.certification[0].expected_product
        # the input is never mutated
        assert substrate.GetNumAtoms() == Chem.MolFromSmiles(
            rule.certification[0].reagents[0]
        ).GetNumAtoms()

    def test_noop_on_unrelated_molecule(self):
        rule = DEPROTECTIONS["rxn_28"]
        plain = Chem.MolFromSmiles("CCOc1ccccc1")
        out = apply_deprotection(plain, rule)
        assert canonical_product_smiles(out) == canonical_product_smiles(plain)

    def test_sequence(self):
        # A bifunctional substrate carrying both a Boc and an Alloc group:
        # the fixed-order sequence removes both, deterministically.
        substrate = Chem.MolFromSmiles("CC(C)(C)OC(=O)NCCNC(=O)OCC=C")
        out = apply_all_deprotections(substrate)
        assert canonical_product_smiles(out) == "NCCN"  # ethylenediamine

    def test_policy_action_rules_rejected(self):
        with pytest.raises(ReactionRegistryError):
            apply_deprotection(Chem.MolFromSmiles("CN"), POLICY_REACTIONS["rxn_01"])


# ---------------------------------------------------------------------------
# 5. Engine integration: the certified registry drives the real MDP
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_default_registry_is_the_certified_taxonomy(self):
        assert set(DEFAULT_REACTION_REGISTRY) == set(POLICY_REACTIONS)
        assert "generic_coupling" not in DEFAULT_REACTION_REGISTRY  # superseded interim
        assert DEFAULT_REACTION_REGISTRY["rxn_01"].resonance_lock == "trans"
        assert DEFAULT_REACTION_REGISTRY["rxn_01"].block_handle_class == "acid_chloride"
        assert DEFAULT_REACTION_REGISTRY["rxn_02"].resonance_lock is None
        assert DEFAULT_REACTION_REGISTRY["rxn_13"].block_handle_class == "acetylide"

    def _rollout(self, seed, reaction_id, block_id, pocket=None):
        env = AssemblyEnvironment(
            seed, {block_id: CATALOG[block_id]}, pocket=pocket, max_blocks=4
        )
        env.reset()
        env.place_seed()
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, reaction_id, block_id, torsion_bin=18))
        env.stop()
        return env

    @pytest.mark.parametrize(
        "seed_id, seed_smiles, seed_class, reaction_id, block_id, expected",
        [
            ("amine", "CN", "amine", "rxn_01", "acid_chloride", "CNC(C)=O"),
            ("amine", "CN", "amine", "rxn_03", "aldehyde", "CCNC"),
            ("amine", "CN", "amine", "rxn_07", "alkyl_bromide", "CCNC"),
            ("amine", "CN", "amine", "rxn_22", "pyrimidinyl_chloride", "CNc1ncccn1"),
            ("amine", "CN", "amine", "rxn_25", "pyridyl_chloride", "CNc1ccccn1"),
            ("alcohol", "CCO", "alcohol", "rxn_17", "acid_chloride", "CCOC(C)=O"),
            ("alcohol", "CCO", "alcohol", "rxn_16", "alkyl_bromide", "CCOCC"),
            ("thiol", "CCS", "thiol", "rxn_20", "acid_chloride", "CCSC(C)=O"),
            ("aryl_bromide", "Brc1ccccc1", "aryl_bromide", "rxn_09", "arylboronic_acid", "c1ccc(-c2ccccc2)cc1"),
            ("vinyl_bromide", "C=CBr", "vinyl_bromide", "rxn_14", "arylboronic_acid", "C=Cc1ccccc1"),
        ],
    )
    def test_end_to_end_products(self, seed_id, seed_smiles, seed_class, reaction_id, block_id, expected):
        seed = _seed(seed_id, seed_smiles, seed_class)
        env = self._rollout(seed, reaction_id, block_id)
        assert Chem.MolToSmiles(env.molecule) == expected
        junctions = _junctions_of(env.molecule)
        assert len(junctions) == 1
        locked = POLICY_REACTIONS[reaction_id].resonance_lock == "trans"
        assert junctions[0].resonance_locked is locked

    def test_amide_junction_snaps_trans_with_certified_registry(self):
        seed = _seed("amine", "CN", "amine")
        env = self._rollout(seed, "rxn_01", "acid_chloride")
        junction = _junctions_of(env.molecule)[0]
        value = GetDihedralRad(
            env.molecule.GetConformer(), junction.a0, junction.a1, junction.b1, junction.b0
        )
        assert abs(wrap_angle(value - math.pi)) < 1e-9  # certified trans lock

    def test_rotatable_junction_honours_the_torsion_bin(self):
        seed = _seed("amine", "CN", "amine")
        env = AssemblyEnvironment(seed, {"alkyl_bromide": CATALOG["alkyl_bromide"]}, max_blocks=4)
        env.reset()
        env.place_seed()
        handle = env.active_handles()[0]["handle_atom_idx"]
        env.step(AssemblyAction(handle, "rxn_07", "alkyl_bromide", torsion_bin=27))
        junction = _junctions_of(env.molecule)[0]
        value = GetDihedralRad(
            env.molecule.GetConformer(), junction.a0, junction.a1, junction.b1, junction.b0
        )
        assert abs(wrap_angle(value - bin_center(27))) < 1e-9

    def test_unknown_reaction_fails_closed(self):
        seed = _seed("amine", "CN", "amine")
        env = AssemblyEnvironment(seed, {"acid_chloride": CATALOG["acid_chloride"]}, max_blocks=4)
        env.reset()
        env.place_seed()
        handle = env.active_handles()[0]["handle_atom_idx"]
        with pytest.raises(AssemblyError):
            env.step(AssemblyAction(handle, "generic_coupling", "acid_chloride"))
        with pytest.raises(AssemblyError):
            env.step(AssemblyAction(handle, "amide_formation", "acid_chloride"))

    def test_wrong_block_handle_fails_closed(self):
        # rxn_01 requires an acid_chloride block-side handle; an aldehyde
        # block must be rejected (aldehyde handles belong to rxn_03).
        seed = _seed("amine", "CN", "amine")
        env = AssemblyEnvironment(seed, {"aldehyde": CATALOG["aldehyde"]}, max_blocks=4)
        env.reset()
        env.place_seed()
        handle = env.active_handles()[0]["handle_atom_idx"]
        with pytest.raises(AssemblyError):
            env.step(AssemblyAction(handle, "rxn_01", "aldehyde"))

    def test_certified_registry_through_the_tb_engine(self):
        seed = _seed("amine", "CN", "amine")
        env = self._rollout(seed, "rxn_01", "acid_chloride")
        # A uniform forward policy over the (single-action) menu; the STOP
        # log-probability is the complement of the add branch.
        for record in env.trajectory:
            if record.action_type == "add":
                record.action_log_prob = math.log(0.5)
            else:
                record.action_log_prob = math.log(0.5)
        engine = TrajectoryBalanceEngine(initial_log_z=0.0)
        steps = as_trajectory_steps(env.trajectory)
        assert [s.action_type for s in steps] == ["seed", "add", "stop"]
        residual = engine.trajectory_residual(steps, reward=1.5)
        assert torch_isfinite(residual)
        # The recorded trajectory carries the certified reaction id in its
        # action record (the TB contract is registry-agnostic).
        assert env.trajectory[0].action_type == "seed"  # the #39 seed action
        assert env.trajectory[1].action.reaction_id == "rxn_01"
        assert env.trajectory[2].action is None  # STOP


def torch_isfinite(tensor) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all())
