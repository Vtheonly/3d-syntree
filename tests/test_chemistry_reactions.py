"""Unit tests for the SMARTS reaction engine."""

from __future__ import annotations

import pytest
from rdkit import Chem

from syntree.chemistry.reactions import (
    HANDLE_NAMES,
    HANDLE_SMARTS,
    HANDLE_TO_REACTIONS,
    REACTION_FAMILY_MEMBERS,
    REACTION_FAMILY_NAMES,
    REACTION_PARTNER_HANDLES,
    REACTION_SIDES,
    REACTION_TEMPLATES,
    HandleInfo,
    ReactionEngine,
    ReactionResult,
    _tag_atoms,
    _clear_tags,
    grammar_fingerprint,
)

# Valid reactant pairs per reaction (core, synthon).
VALID_PAIRS = {
    "amide_coupling": ("OC(=O)C1CCCCC1", "C1CC(N)CC1"),
    "reductive_amination": ("C1CCC(CC1)C=O", "C1CC(N)CC1"),
    "suzuki_coupling": ("Brc1ccc(F)cc1", "B(c1ccccc1)(O)O"),
    "snar": ("Fc1ccc(Br)cc1", "C1CC(N)CC1"),
    "urea_formation": ("C1CC(N)CC1", "NCC1CCCCC1"),
    "buchwald_hartwig": ("IC1=CC=CC=C1", "C1CC(N)CC1"),
    "esterification": ("OC(=O)C1CCCCC1", "CC(C)CO"),
    "click_triazole": ("CCCCCC#C", "CCCCCN=[N+]=[N-]"),
    # tasklist Priority 4 additions
    "sulfonamide_coupling": ("CS(=O)(=O)Cl", "C1CC(N)CC1"),
    "sp3_alkylation": ("NCCBr", "C1CC(N)CC1"),
}


@pytest.fixture(scope="module")
def engine():
    return ReactionEngine()


class TestTemplateIntegrity:
    def test_all_ten_templates_present(self):
        assert len(REACTION_TEMPLATES) == 10
        expected = {
            "amide_coupling", "reductive_amination", "suzuki_coupling",
            "snar", "urea_formation", "buchwald_hartwig",
            "esterification", "click_triazole",
            "sulfonamide_coupling", "sp3_alkylation",
        }
        assert set(REACTION_TEMPLATES) == expected

    def test_legacy_reaction_order_preserved(self):
        """Family indices 0..6 are baked into old datasets: append-only."""
        legacy = (
            "amide_coupling", "reductive_amination", "suzuki_coupling",
            "aryl_amination", "urea_formation", "esterification", "click_triazole",
        )
        assert REACTION_FAMILY_NAMES[:7] == legacy
        assert REACTION_FAMILY_NAMES[7:] == ("sulfonamide_coupling", "sp3_alkylation")

    def test_legacy_handle_order_preserved(self):
        """Handle indices 0..7 are baked into old datasets: append-only."""
        legacy = (
            "carboxylic_acid", "primary_secondary_amine", "aryl_halide",
            "boronic_acid", "aldehyde", "alcohol", "alkyne", "azide",
        )
        assert HANDLE_NAMES[:8] == legacy
        assert HANDLE_NAMES[8:] == ("sulfonyl_chloride", "alkyl_halide")

    def test_grammar_fingerprint_stable_and_sensitive(self):
        fp = grammar_fingerprint()
        assert len(fp) == 8
        assert all(c in "0123456789abcdef" for c in fp)
        assert grammar_fingerprint() == fp  # deterministic within a session

    def test_templates_compile(self, engine):
        assert set(engine.reactions) == set(REACTION_TEMPLATES)

    def test_sides_cover_all_reactions(self):
        assert set(REACTION_SIDES) == set(REACTION_TEMPLATES)

    def test_handle_to_reactions_consistent_with_sides(self):
        for rxn, (side_a, side_b) in REACTION_SIDES.items():
            assert rxn in HANDLE_TO_REACTIONS[side_a]
            assert rxn in HANDLE_TO_REACTIONS[side_b]

    def test_partner_handles_symmetric(self):
        for handle, partners in REACTION_PARTNER_HANDLES.items():
            for rxn, partner_list in partners.items():
                assert handle in [
                    h for h in REACTION_PARTNER_HANDLES if rxn in REACTION_PARTNER_HANDLES[h]
                ]

    def test_invalid_template_rejected(self):
        with pytest.raises(ValueError):
            ReactionEngine(templates={"broken": "not>>a>><reaction"})


class TestHandleDetection:
    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("OC(=O)C1CCCCC1", {"carboxylic_acid"}),
            ("C1CC(N)CC1", {"primary_secondary_amine"}),
            ("Brc1ccc(F)cc1", {"aryl_halide"}),
            ("B(c1ccccc1)(O)O", {"boronic_acid"}),
            ("C1CCC(CC1)C=O", {"aldehyde"}),
            ("CC(C)CO", {"alcohol"}),
            ("CCCCCC#C", {"alkyne"}),
            ("CCCCCN=[N+]=[N-]", {"azide"}),
            # Amide nitrogen must NOT count as an amine handle.
            ("O=C(NCC)C1CCCCC1", set()),
            # tasklist Priority 4 handles
            ("CS(=O)(=O)Cl", {"sulfonyl_chloride"}),
            ("Cc1ccc(S(=O)(=O)Cl)cc1", {"sulfonyl_chloride"}),
            ("CCBr", {"alkyl_halide"}),
            ("NCCBr", {"alkyl_halide", "primary_secondary_amine"}),
            ("ClCC(Cl)Cl", {"alkyl_halide"}),
            # Acyl / vinyl / aryl halides must NOT be alkyl halides.
            ("CC(=O)Cl", set()),
            ("C=CCl", set()),
            ("c1ccc(Cl)cc1", {"aryl_halide"}),
        ],
    )
    def test_handle_detection(self, engine, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert engine.handle_types(mol) == expected

    def test_multi_handle_molecule(self, engine):
        # Lysine analogue: amine + acid.
        mol = Chem.MolFromSmiles("NCCCC(=O)O")
        assert engine.handle_types(mol) == {"carboxylic_acid", "primary_secondary_amine"}

    def test_handle_info_structure(self, engine):
        mol = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        handles = engine.detect_handles(mol)
        assert len(handles) == 1
        info = handles[0]
        assert isinstance(info, HandleInfo)
        assert info.handle_type == "carboxylic_acid"
        assert info.primary_atom in info.atom_indices
        assert "amide_coupling" in info.allowed_reactions
        assert "esterification" in info.allowed_reactions

    def test_none_mol_raises(self, engine):
        with pytest.raises(ValueError):
            engine.detect_handles(None)


class TestReactionExecution:
    @pytest.mark.parametrize("reaction", sorted(VALID_PAIRS))
    def test_reaction_produces_sanitized_product(self, engine, reaction):
        core, synthon = VALID_PAIRS[reaction]
        result = engine.apply_reaction(
            Chem.MolFromSmiles(core), Chem.MolFromSmiles(synthon), reaction
        )
        assert result is not None, f"{reaction} produced no product"
        assert isinstance(result, ReactionResult)
        assert result.product.GetNumAtoms() > 0
        assert result.reaction_name == reaction

    @pytest.mark.parametrize("reaction", sorted(VALID_PAIRS))
    def test_junction_is_real_bond(self, engine, reaction):
        core, synthon = VALID_PAIRS[reaction]
        result = engine.apply_reaction(
            Chem.MolFromSmiles(core), Chem.MolFromSmiles(synthon), reaction
        )
        a1, a2 = result.junction_bond
        assert 0 <= a1 < result.product.GetNumAtoms()
        assert 0 <= a2 < result.product.GetNumAtoms()
        assert result.product.GetBondBetweenAtoms(a1, a2) is not None

    @pytest.mark.parametrize("reaction", sorted(VALID_PAIRS))
    def test_atom_provenance(self, engine, reaction):
        core, synthon = VALID_PAIRS[reaction]
        core_mol, synthon_mol = Chem.MolFromSmiles(core), Chem.MolFromSmiles(synthon)
        result = engine.apply_reaction(core_mol, synthon_mol, reaction)
        # All core atoms that survive the reaction must be mapped.
        assert len(result.core_atom_map) >= min(3, core_mol.GetNumAtoms() - 2)
        for core_idx, prod_idx in result.core_atom_map.items():
            assert core_idx < core_mol.GetNumAtoms()
            assert prod_idx < result.product.GetNumAtoms()
            assert (
                result.product.GetAtomWithIdx(prod_idx).GetAtomicNum()
                == core_mol.GetAtomWithIdx(core_idx).GetAtomicNum()
            )

    def test_inputs_not_mutated(self, engine):
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        synthon = Chem.MolFromSmiles("C1CC(N)CC1")
        core_smiles_before = Chem.MolToSmiles(core)
        engine.apply_reaction(core, synthon, "amide_coupling")
        assert Chem.MolToSmiles(core) == core_smiles_before
        # No isotope provenance leaked onto the reactants.
        assert all(a.GetIsotope() == 0 for a in core.GetAtoms())
        assert all(a.GetIsotope() == 0 for a in synthon.GetAtoms())

    def test_product_has_no_provenance_tags(self, engine):
        result = engine.apply_reaction(
            Chem.MolFromSmiles("OC(=O)C1CCCCC1"),
            Chem.MolFromSmiles("C1CC(N)CC1"),
            "amide_coupling",
        )
        for atom in result.product.GetAtoms():
            assert atom.GetIsotope() == 0
            assert atom.GetAtomMapNum() == 0

    def test_product_is_chemically_valid(self, engine):
        """The core selling point: every product must sanitize cleanly."""
        for reaction, (core, synthon) in VALID_PAIRS.items():
            result = engine.apply_reaction(
                Chem.MolFromSmiles(core), Chem.MolFromSmiles(synthon), reaction
            )
            assert result is not None
            probe = Chem.Mol(result.product)
            Chem.SanitizeMol(probe)  # must not raise

    def test_reactant_ordering_swapped(self, engine):
        """Amine-first arguments must still couple (ordering logic)."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("C1CC(N)CC1"),  # synthon passed as core
            Chem.MolFromSmiles("OC(=O)C1CCCCC1"),  # acid passed as synthon
            "amide_coupling",
        )
        assert result is not None
        # Now the amine is the 'core' provenance side (6 heavy atoms:
        # 5 ring carbons + 1 nitrogen).
        assert len(result.core_atom_map) == 6

    def test_incompatible_reactants_return_none(self, engine):
        """An acid + an alcohol cannot do amide coupling."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("OC(=O)C1CCCCC1"),
            Chem.MolFromSmiles("CC(C)CO"),
            "amide_coupling",
        )
        assert result is None

    def test_unknown_reaction_raises(self, engine):
        with pytest.raises(ValueError, match="Unknown reaction"):
            engine.apply_reaction(
                Chem.MolFromSmiles("C"), Chem.MolFromSmiles("C"), "diels_alder"
            )

    def test_none_reactants_raise(self, engine):
        with pytest.raises(ValueError):
            engine.apply_reaction(None, Chem.MolFromSmiles("C"), "amide_coupling")

    def test_empty_reactants_return_none(self, engine):
        empty = Chem.Mol()  # valid but atom-less
        result = engine.apply_reaction(empty, Chem.MolFromSmiles("C"),
                                       "amide_coupling")
        assert result is None

    def test_suzuki_boryl_side_as_core(self, engine):
        """Boronic acid as the growing core, aryl bromide as incoming."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("B(c1ccccc1)(O)O"),
            Chem.MolFromSmiles("Brc1ccc(F)cc1"),
            "suzuki_coupling",
        )
        assert result is not None
        assert "Fc1ccc" in Chem.MolToSmiles(result.product)


class TestTagHelpers:
    def test_tag_and_clear_roundtrip(self):
        mol = Chem.MolFromSmiles("CCO")
        tagged = _tag_atoms(mol, 1000)
        assert [a.GetIsotope() for a in tagged.GetAtoms()] == [1000, 1001, 1002]
        cleaned = _clear_tags(tagged)
        assert all(a.GetIsotope() == 0 for a in cleaned.GetAtoms())

    def test_tag_does_not_mutate_original(self):
        mol = Chem.MolFromSmiles("CCO")
        _tag_atoms(mol, 1000)
        assert all(a.GetIsotope() == 0 for a in mol.GetAtoms())


class TestChainAssembly:
    def test_two_step_growth(self, engine):
        """Grow an amide then react the product further."""
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        amine = Chem.MolFromSmiles("NCC1CCCCC1")
        step1 = engine.apply_reaction(core, amine, "amide_coupling")
        assert step1 is not None
        # The product retains the cyclohexylamine's terminal amine? No -
        # aminomethylcyclohexane's N is consumed. Detect remaining handles.
        handles = engine.detect_handles(step1.product)
        handle_types = {h.handle_type for h in handles}
        # The amide product has no free amine left (both reactants consumed).
        assert "primary_secondary_amine" not in handle_types

    def test_amide_product_reacts_as_alcohol(self, engine):
        """Multi-functional growth: hydroxy-amine leaves an alcohol handle."""
        core = Chem.MolFromSmiles("OC(=O)C1CCCCC1")
        hydroxy_amine = Chem.MolFromSmiles("CC(O)CN")
        step1 = engine.apply_reaction(core, hydroxy_amine, "amide_coupling")
        assert step1 is not None
        handles = {h.handle_type for h in engine.detect_handles(step1.product)}
        assert "alcohol" in handles
        # Esterify the free alcohol.
        acid = Chem.MolFromSmiles("CC(C)(C)C(=O)O")
        step2 = engine.apply_reaction(step1.product, acid, "esterification")
        assert step2 is not None
        smiles = Chem.MolToSmiles(step2.product)
        assert "OC(=O)C(C)(C)C" in smiles or "C(C)(C)C(=O)O" in smiles


class TestSulfonamideAndAlkylationChemistry:
    """tasklist Priority 4: sulfonamide coupling + sp3 alkylation."""

    def test_sulfonamide_product_connectivity(self, engine):
        """MsCl + cyclopentylamine -> N-cyclopentylmethanesulfonamide."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("CS(=O)(=O)Cl"),
            Chem.MolFromSmiles("C1CC(N)CC1"),
            "sulfonamide_coupling",
        )
        assert result is not None
        assert Chem.MolToSmiles(result.product) == "CS(=O)(=O)NC1CCCC1"

    def test_sulfonamide_secondary_amine_h_count(self, engine):
        """The product N must carry exactly one implicit H (secondary
        sulfonamide), i.e. RDKit recomputes H counts from product valence."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("CS(=O)(=O)Cl"),
            Chem.MolFromSmiles("CCN"),
            "sulfonamide_coupling",
        )
        assert result is not None
        product = result.product
        n_atom = next(
            a for a in product.GetAtoms()
            if a.GetAtomicNum() == 7 and any(
                n.GetAtomicNum() == 16 for n in a.GetNeighbors()
            )
        )
        assert n_atom.GetTotalNumHs() == 1
        assert Chem.MolToSmiles(product) == "CCNS(C)(=O)=O"

    def test_sulfonamide_reversed_reactant_order(self, engine):
        """Amine on the core, sulfonyl chloride as the incoming synthon."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("C1CC(N)CC1"),
            Chem.MolFromSmiles("CS(=O)(=O)Cl"),
            "sulfonamide_coupling",
        )
        assert result is not None
        assert Chem.MolToSmiles(result.product) == "CS(=O)(=O)NC1CCCC1"

    def test_tosyl_anilide_chemistry(self, engine):
        """TsCl + aniline -> tosylanilide (celecoxib/sulfonamide-drug style).
        Aromatic amines must be accepted by the sulfonamide template."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("Cc1ccc(S(=O)(=O)Cl)cc1"),
            Chem.MolFromSmiles("Nc1ccccc1"),
            "sulfonamide_coupling",
        )
        assert result is not None
        assert (
            Chem.MolToSmiles(result.product)
            == "Cc1ccc(S(=O)(=O)Nc2ccccc2)cc1"
        )

    def test_sulfonamide_no_new_atoms(self, engine):
        """All product atoms are mapped (S, both O, N, and R-groups); the Cl
        leaving group is deleted, so new_atoms must be empty and the
        junction is a direct S-N bond."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("CS(=O)(=O)Cl"),
            Chem.MolFromSmiles("C1CC(N)CC1"),
            "sulfonamide_coupling",
        )
        assert result.new_atoms == ()
        a, b = result.junction_bond
        product = result.product
        assert {product.GetAtomWithIdx(a).GetAtomicNum(),
                product.GetAtomWithIdx(b).GetAtomicNum()} == {16, 7}

    def test_sulfonamide_leaves_free_handle_for_growth(self, engine):
        """2-aminoethylsulfonyl chloride leaves an amine handle so the
        trajectory can continue through the sulfonamide junction."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("O=S(=O)(Cl)CCN"),
            Chem.MolFromSmiles("CCBr"),
            "sp3_alkylation",
        )
        assert result is not None
        handles = {h.handle_type for h in engine.detect_handles(result.product)}
        assert "primary_secondary_amine" in handles

    def test_alkylation_product_connectivity(self, engine):
        """Bromoethylamine + cyclopentylamine -> secondary diamine."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("NCCBr"),
            Chem.MolFromSmiles("C1CC(N)CC1"),
            "sp3_alkylation",
        )
        assert result is not None
        assert Chem.MolToSmiles(result.product) == "NCCNC1CCCC1"

    def test_alkylation_reversed_reactant_order(self, engine):
        """Amine core + bromoethane synthon (N-ethylation)."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("C1CC(N)CC1"),
            Chem.MolFromSmiles("CCBr"),
            "sp3_alkylation",
        )
        assert result is not None
        assert Chem.MolToSmiles(result.product) == "CCNC1CCCC1"

    def test_alkylation_junction_is_c_n_bond(self, engine):
        result = engine.apply_reaction(
            Chem.MolFromSmiles("CCBr"),
            Chem.MolFromSmiles("C1CC(N)CC1"),
            "sp3_alkylation",
        )
        assert result is not None
        a, b = result.junction_bond
        product = result.product
        assert {product.GetAtomWithIdx(a).GetAtomicNum(),
                product.GetAtomWithIdx(b).GetAtomicNum()} == {6, 7}

    def test_alkylation_excludes_acyl_and_vinyl_halides(self, engine):
        """Acyl chloride 'cores' carry no alkyl_halide handle, so the
        reaction must refuse to run rather than mislabel amide formation."""
        result = engine.apply_reaction(
            Chem.MolFromSmiles("CC(=O)Cl"),
            Chem.MolFromSmiles("C1CC(N)CC1"),
            "sp3_alkylation",
        )
        assert result is None

    def test_new_handles_in_allowed_reactions(self):
        from syntree.chemistry.reactions import HANDLE_TO_REACTIONS
        assert "sulfonamide_coupling" in HANDLE_TO_REACTIONS["sulfonyl_chloride"]
        assert "sulfonamide_coupling" in HANDLE_TO_REACTIONS["primary_secondary_amine"]
        assert "sp3_alkylation" in HANDLE_TO_REACTIONS["alkyl_halide"]
        assert "sp3_alkylation" in HANDLE_TO_REACTIONS["primary_secondary_amine"]

    def test_new_families_are_single_member(self):
        assert REACTION_FAMILY_MEMBERS["sulfonamide_coupling"] == ("sulfonamide_coupling",)
        assert REACTION_FAMILY_MEMBERS["sp3_alkylation"] == ("sp3_alkylation",)

    def test_sulfonyl_chloride_is_high_reactivity_tier(self):
        """Sulfonyl chloride (tier 2) outranks acids/aldehydes (tier 3) so
        growth routes through the hyper-reactive handle first."""
        from syntree.chemistry.reactions import HANDLE_REACTIVITY_TIERS
        assert HANDLE_REACTIVITY_TIERS["sulfonyl_chloride"] == 2
        assert HANDLE_REACTIVITY_TIERS["alkyl_halide"] == 3
        assert HANDLE_REACTIVITY_TIERS["sulfonyl_chloride"] < HANDLE_REACTIVITY_TIERS["carboxylic_acid"]

    def test_rank_handles_orders_new_handles_correctly(self, engine):
        """Sulfonyl chloride (tier 2) outranks the weaker electrophiles
        (acid/aldehyde/alkyl-halide, tier 3) but stays below the tier-1
        amine nucleophile - matching the existing chemoselectivity ladder."""
        # Sulfonyl chloride + carboxylic acid: growth routes through SO2Cl.
        mol_acid = Chem.MolFromSmiles("O=S(=O)(Cl)CCC(=O)O")
        ranked_acid = engine.rank_handles(mol_acid)
        assert ranked_acid[0].handle_type == "sulfonyl_chloride"

        # Sulfonyl chloride + amine: the amine nucleophile still wins.
        mol_amine = Chem.MolFromSmiles("O=S(=O)(Cl)CCN")
        ranked_amine = engine.rank_handles(mol_amine)
        assert ranked_amine[0].handle_type == "primary_secondary_amine"
        assert (
            next(h.handle_type for h in ranked_amine if h.handle_type == "sulfonyl_chloride")
            == "sulfonyl_chloride"
        )
