"""Suite A / 1 — forward execution + isotope-provenance mapping of every
certified reaction (issue #32 TEST-01; constitution #45 §16 Suite A).

Capability regression over the complete certified taxonomy (#11): all 32
reactions (27 policy couplings + 5 deterministic deprotections) execute
forward on their standardized certification pairs and reproduce the
pinned expected products atom-for-atom.  Provenance is proven by
isotope tagging: molecule-side and block-side carbons carry
element-encoded mass tags (``100 + atom index`` / ``200 + atom index``),
and every tag observed in a product must decode to a live reactant atom
— the mapping through the transformation is auditable, and repeated
execution yields identical tag sets (bit-stable determinism).
"""

import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.chemistry.reactions import (  # noqa: E402
    CERTIFIED_REACTIONS,
    DEPROTECTIONS,
    POLICY_REACTIONS,
    canonical_product_smiles,
    run_certified_reaction,
)

#: Isotope tag encodings: (offset, side).  Both encode CARBON atoms only
#: (chemically neutral tags: pure carbons exist in every certification
#: reagent of the taxonomy).
MOL_SIDE_TAG = 100
BLOCK_SIDE_TAG = 200


def _tag_carbons(smiles: str, offset: int):
    """Return (mol, {atom_idx}) with C atoms tagged by ``offset + idx``."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparseable reagent {smiles!r}"
    provenance = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C":
            atom.SetIsotope(offset + atom.GetIdx())
            provenance.add(atom.GetIdx())
    return mol, provenance


def _decode_tag(isotope: int):
    """Decode a mass tag back to (side, atom index) or None.

    Tag ranges are disjoint: mol-side carbons carry ``100 + idx``
    (``idx < 100`` — molecules in this domain have < 100 heavy atoms),
    block-side carbons ``200 + idx``.
    """
    if MOL_SIDE_TAG <= isotope < MOL_SIDE_TAG + 100:
        return "mol", isotope - MOL_SIDE_TAG
    if BLOCK_SIDE_TAG <= isotope < BLOCK_SIDE_TAG + 100:
        return "block", isotope - BLOCK_SIDE_TAG
    return None


def _run_tagged(reaction, pair):
    """Execute one certification pair with isotope-tagged carbons."""
    reagents = list(pair.reagents)
    if len(reagents) == 2:
        mol_a, prov_a = _tag_carbons(reagents[0], MOL_SIDE_TAG)
        mol_b, prov_b = _tag_carbons(reagents[1], BLOCK_SIDE_TAG)
        reactants = (mol_a, mol_b)
    else:
        mol_a, prov_a = _tag_carbons(reagents[0], MOL_SIDE_TAG)
        prov_b = set()
        reactants = (mol_a,)
    rxn = AllChem.ReactionFromSmarts(reaction.reaction_smarts)
    assert rxn is not None, f"{reaction.reaction_id}: unparseable SMARTS"
    products = []
    for row in rxn.RunReactants(reactants):
        product = row[0]
        Chem.SanitizeMol(product)
        products.append(product)
    return products, prov_a, prov_b


class TestTaxonomyCoverage:
    def test_all_32_certified_reactions_present(self):
        assert len(POLICY_REACTIONS) == 27
        assert len(DEPROTECTIONS) == 5
        assert len(CERTIFIED_REACTIONS) == 32

    @pytest.mark.parametrize("reaction_id", sorted(CERTIFIED_REACTIONS))
    def test_reaction_has_certification_pairs(self, reaction_id):
        reaction = CERTIFIED_REACTIONS[reaction_id]
        assert reaction.certification, f"{reaction_id}: no certification pairs"


class TestForwardExecution:
    @pytest.mark.parametrize("reaction_id", sorted(CERTIFIED_REACTIONS))
    def test_certification_pairs_reproduce_expected_products(self, reaction_id):
        """Every standardized pair rebuilds its pinned expected product.

        Both sides are compared in the registry's canonical form: the
        pinned expected SMILES is re-canonicalized so that tetrahedral
        ``@``/``@@`` markers — which flip with atom ordering — compare
        on stereochemical content, not on serialization order.
        """
        reaction = CERTIFIED_REACTIONS[reaction_id]
        for pair in reaction.certification:
            products = run_certified_reaction(reaction, pair.reagents)
            canonical = {canonical_product_smiles(p) for p in products}
            expected_mol = Chem.MolFromSmiles(pair.expected_product)
            assert expected_mol is not None, (
                f"{reaction_id}: pinned expected product does not parse: "
                f"{pair.expected_product!r}"
            )
            expected = canonical_product_smiles(expected_mol)
            assert expected in canonical, (
                f"{reaction_id}: expected product {expected!r} not among "
                f"executed products {sorted(canonical)}"
            )


class TestIsotopeProvenance:
    @pytest.mark.parametrize("reaction_id", sorted(CERTIFIED_REACTIONS))
    def test_product_tags_decode_to_reactant_atoms(self, reaction_id):
        """Every mass tag in a product decodes to a live reactant carbon;
        junction products carry provenance from BOTH sides (couplings)."""
        reaction = CERTIFIED_REACTIONS[reaction_id]
        pair = reaction.certification[0]
        products, prov_a, prov_b = _run_tagged(reaction, pair)
        assert products, f"{reaction_id}: no products from tagged reagents"
        product = products[0]
        sides_seen = set()
        for atom in product.GetAtoms():
            if atom.GetIsotope():
                decoded = _decode_tag(atom.GetIsotope())
                assert decoded is not None, (
                    f"{reaction_id}: unexplained isotope {atom.GetIsotope()}"
                )
                side, idx = decoded
                prov = prov_a if side == "mol" else prov_b
                assert idx in prov, (
                    f"{reaction_id}: tag {atom.GetIsotope()} does not decode "
                    "to a reactant atom (provenance broken)"
                )
                sides_seen.add(side)
        if len(pair.reagents) == 2 and prov_a and prov_b:
            # coupling products must carry atoms from both partners
            assert sides_seen == {"mol", "block"}, (
                f"{reaction_id}: product provenance missing a reactant side "
                f"({sides_seen})"
            )

    @pytest.mark.parametrize("reaction_id", sorted(CERTIFIED_REACTIONS))
    def test_provenance_deterministic_across_repeats(self, reaction_id):
        """Repeating the tagged execution yields identical tag multisets."""
        reaction = CERTIFIED_REACTIONS[reaction_id]
        pair = reaction.certification[0]

        def tag_signature():
            products, _, _ = _run_tagged(reaction, pair)
            sig = []
            for product in products:
                sig.append(
                    tuple(
                        sorted(
                            a.GetIsotope()
                            for a in product.GetAtoms()
                            if a.GetIsotope()
                        )
                    )
                )
            return tuple(sorted(sig))

        assert tag_signature() == tag_signature()


class TestFailClosedExecution:
    def test_mismatched_reagents_raise(self):
        from syntree.chemistry.reactions import ReactionRegistryError

        reaction = POLICY_REACTIONS["rxn_01"]  # amine + acid chloride
        with pytest.raises(ReactionRegistryError):
            # two amines cannot undergo the acid-chloride coupling
            run_certified_reaction(reaction, ("CN", "CCN"))

    def test_unparseable_reagent_raises(self):
        from syntree.chemistry.reactions import ReactionRegistryError

        reaction = POLICY_REACTIONS["rxn_01"]
        with pytest.raises(ReactionRegistryError):
            run_certified_reaction(reaction, ("not a smiles", "CC(=O)Cl"))
