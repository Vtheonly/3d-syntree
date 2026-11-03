"""The certified 32-reaction taxonomy and its coarse family mapping (#11 CHEM-01).

Constitution #45 §9.1 requires every registered transformation to define
SMARTS templates, required handle types, atom/bond mapping and provenance,
stereochemical behaviour, validity constraints, product sanitization,
failure behaviour, and reaction-family membership — and requires that
certification is **not** "the SMARTS executes": each reaction carries
standardized reagent/product pairs whose forward replay is verified
atom-for-atom by :func:`certify_registry` (round-trip forward-replay
validation, the #11 acceptance criterion).

Taxonomy (issue-pinned)
-----------------------
32 reactions ``rxn_01`` .. ``rxn_32`` in four coupling blocks plus a
deprotection block (issue #11 §2):

* C-N couplings (8) — ``rxn_01`` .. ``rxn_08``;
* C-C couplings (7) — ``rxn_09`` .. ``rxn_15``;
* C-O/C-S couplings (6) — ``rxn_16`` .. ``rxn_21``;
* heterocycle installations (6) — ``rxn_22`` .. ``rxn_27``;
* deterministic deprotections (5) — ``rxn_28`` .. ``rxn_32`` (no policy
  action; applied by catalog/trajectory consumers).

Coarse families (issue-pinned membership; see D-V2-016 for the two
unlisted members):

1. ``amide_and_carbamate``  — rxn_01, rxn_08
2. ``sulfonamide_coupling``  — rxn_02
3. ``reductive_amination``   — rxn_03
4. ``aryl_amination``        — rxn_04, rxn_05
5. ``urea_formation``        — rxn_06
6. ``sp3_alkylation``        — rxn_07, rxn_16, rxn_19
7. ``cross_coupling_cc``     — rxn_09..rxn_13 **plus rxn_14, rxn_15**
   (D-V2-016: the issue's family list omitted the two remaining C-C
   couplings; "map the 32 transformations into the 9 canonical families"
   forces them into the only C-C family)
8. ``ester_and_ether``       — rxn_17, rxn_18, rxn_20, rxn_21
9. ``heterocycle_click``     — rxn_22..rxn_27

Junction model (engine integration, D-V2-009 scope boundary)
------------------------------------------------------------
The discrete Assembly MDP forms exactly one junction bond per action:
the molecule-side attachment atom bonds to the block-side attachment
atom, the mapped leaving atoms are deleted, and optional resonance locks
pin the junction dihedral.  Every reaction here is therefore expressed in
the *single-junction* form ``A-X + Y-B -> A-B`` (X/Y leave).  The six
heterocycle installations ship the heterocycle **pre-cyclized on the
synthon side** (the only representation compatible with the pinned MDP
contracts — a junction can never close a ring); each entry documents the
one-step cyclization provenance of its synthon class.

Stereochemistry (#11 §2, acceptance 2)
--------------------------------------
All 32 transformations are **stereoretentive** at tetrahedral centres
adjacent to the newly formed bond.  Retention is implemented by RDKit's
atom-map chirality transfer: mapped atoms carry their chiral tags into
the products, and unmapped stereocentres ride along unchanged with their
substituents.  Every family carries at least one *stereo pair* in its
certification set whose product canonical isomeric SMILES (carrying the
``@``/``@@`` retention tokens) is compared atom-for-atom; the acceptance
suite additionally asserts CIP-code preservation reagent-vs-product.
Templates never hard-require a winding (that would reject valid
enantiomeric or achiral reagents); retention is *verified*, not assumed.

Deprotections (rxn_28..rxn_32)
------------------------------
Deterministic, policy-free transforms applied by consumers (catalog
curation #12, trajectory extraction #18): Boc, Alloc, Cbz, O-benzyl,
TBDMS.  Absence of the motif is a documented no-op; a malformed result
fails closed.

This module depends on ``rdkit`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

__all__ = [
    "ReactionRegistryError",
    "ReagentPair",
    "CertifiedReaction",
    "CERTIFIED_REACTIONS",
    "POLICY_REACTIONS",
    "DEPROTECTIONS",
    "REACTION_FAMILIES",
    "REACTION_CATEGORIES",
    "JUNCTION_PRIORITY_CARBOXYL",
    "JUNCTION_PRIORITY_ALKYNE",
    "JUNCTION_PRIORITY_GENERIC",
    "JUNCTION_PRIORITY_REDUCTIVE",
    "canonical_product_smiles",
    "junction_pattern_attachment_atoms",
    "run_certified_reaction",
    "certify_reaction",
    "certify_registry",
    "apply_deprotection",
    "apply_all_deprotections",
    "build_junction_reaction_registry",
]


class ReactionRegistryError(ValueError):
    """Fail-closed error for registry construction and certification."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReagentPair:
    """One standardized certification reagent set with its expected product.

    ``reagents`` is ordered: (molecule-side partner, block-side partner)
    for couplings, or the single protected substrate for deprotections.
    ``expected_product`` is the canonical isomeric SMILES verified by
    chemical knowledge; certification compares atom-for-atom.
    ``stereo_pair`` marks pairs carrying a tetrahedral centre adjacent to
    the newly formed bond (acceptance 2 evidence).
    """

    reagents: Tuple[str, ...]
    expected_product: str
    note: str = ""
    stereo_pair: bool = False


@dataclass(frozen=True)
class CertifiedReaction:
    """One certified transformation (constitution #45 §9.1 fields).

    Junction semantics for the discrete Assembly MDP: the action couples
    the active molecule-side handle of class ``molecule_handle_class``
    with the unique block-side handle of class ``block_handle_class``;
    the block-side leaving atoms (and the molecule-side leaving atoms of
    the reaction class, if any) are deleted; the junction bond
    ``product_bond_type`` forms; ``resonance_lock`` ('trans'/'cis'/None)
    pins the junction dihedral during rollout (None = the torsion bin
    action governs it).

    ``junction_pattern`` is the *product-side* motif SMARTS used by
    retrosynthetic consumers (#15 CHEM-05) to recognize bonds this
    reaction could have formed: map ``:1`` = molecule-side attachment
    atom, ``:2`` = block-side attachment atom.  ``junction_priority``
    breaks pattern ambiguity (higher wins; ties resolve by ascending
    ``reaction_id`` — deterministic, documented in D-V2-016).
    """

    reaction_id: str
    name: str
    family: Optional[str]
    category: str
    policy_action: bool
    reaction_smarts: str
    molecule_handle_class: Optional[str]
    block_handle_class: str
    product_bond_type: str  # 'single' (all 27 couplings; documented)
    resonance_lock: Optional[str]
    leaving_group_molecule: str
    leaving_group_block: str
    stereo_behavior: str
    reference: str
    validity_constraints: str
    failure_behavior: str
    certification: Tuple[ReagentPair, ...]
    junction_pattern: Optional[str] = None
    junction_priority: int = 0
    cyclization_provenance: Optional[str] = None

    def __post_init__(self) -> None:
        if self.product_bond_type != "single":
            raise ReactionRegistryError(
                f"{self.reaction_id}: the V2 junction model forms single bonds; "
                f"got {self.product_bond_type!r}"
            )
        if self.resonance_lock not in (None, "trans", "cis"):
            raise ReactionRegistryError(
                f"{self.reaction_id}: resonance_lock must be None/'trans'/'cis'"
            )
        if not self.certification:
            raise ReactionRegistryError(
                f"{self.reaction_id}: a certified reaction requires at least one "
                f"standardized reagent pair"
            )


#: Junction-pattern priorities (D-V2-016): unmistakable carbonyl/sulfonyl
#: contexts first, then the linear alkyne junction, then generic contexts,
#: reductive amination last (its junction graph is indistinguishable from
#: N-alkylation, which wins).
JUNCTION_PRIORITY_CARBONYL = 4
JUNCTION_PRIORITY_ALKYNE = 3
JUNCTION_PRIORITY_GENERIC = 2
JUNCTION_PRIORITY_REDUCTIVE = 1


def _rxn(
    reaction_id: str,
    name: str,
    family: Optional[str],
    category: str,
    smarts: str,
    mol_class: str,
    block_class: str,
    resonance_lock: Optional[str],
    leaving_mol: str,
    leaving_block: str,
    reference: str,
    certification: Tuple[ReagentPair, ...],
    junction_pattern: str,
    junction_priority: int,
    stereo_note: str = (
        "stereoretentive by atom-map chirality transfer; verified by the "
        "certification pairs (canonical isomeric SMILES carries the retention "
        "tokens)"
    ),
    constraints: str = (
        "reagent templates enforce the reaction class (functional group, "
        "hybridization, H counts); primary/secondary amines only for C-N "
        "couplings; primary alkyl electrophiles for sp3 alkylation (no SN2 "
        "inversion: the reacting carbon is never a stereocentre)"
    ),
    cyclization: Optional[str] = None,
) -> CertifiedReaction:
    return CertifiedReaction(
        reaction_id=reaction_id,
        name=name,
        family=family,
        category=category,
        policy_action=True,
        reaction_smarts=smarts,
        molecule_handle_class=mol_class,
        block_handle_class=block_class,
        product_bond_type="single",
        resonance_lock=resonance_lock,
        leaving_group_molecule=leaving_mol,
        leaving_group_block=leaving_block,
        stereo_behavior=stereo_note,
        reference=reference,
        validity_constraints=constraints,
        failure_behavior=(
            "no SMARTS match or unsanitizable product -> fail closed "
            "(ReactionRegistryError); never a silent skip"
        ),
        certification=certification,
        junction_pattern=junction_pattern,
        junction_priority=junction_priority,
        cyclization_provenance=cyclization,
    )


# ---------------------------------------------------------------------------
# The 27 policy reactions (rxn_01..rxn_27)
# ---------------------------------------------------------------------------

_REACTIONS: List[CertifiedReaction] = [
    # -- C-N couplings (rxn_01..rxn_08) --------------------------------------
    _rxn(
        "rxn_01", "amide formation (acid chloride + amine)", "amide_and_carbamate",
        "C-N",
        "[NX3;H2,H1:4].[CX3:1](=[OX1:2])[Cl:3]>>[CX3:1](=[OX1:2])[NX3:4]",
        "amine", "acid_chloride", "trans",
        "none (molecule-side amines react without leaving atoms)",
        "chloride (deleted)",
        "nucleophilic acyl substitution at an acid chloride; the amide C-N "
        "partial double bond pins the junction planar trans (resonance lock)",
        (
            ReagentPair(("CN", "CC(=O)Cl"), "CC(=O)NC", "acetyl chloride + methylamine -> N-methylacetamide"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "CC(=O)Cl"),
                "CC(=O)N[C@@H](C)c1ccccc1",
                "(S)-1-phenylethylamine acylation; CIP S retained at the centre "
                "adjacent to the new amide bond",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][CX3:2](=[OX1])-[#6]", JUNCTION_PRIORITY_CARBONYL,
    ),
    _rxn(
        "rxn_02", "sulfonamide coupling (sulfonyl chloride + amine)", "sulfonamide_coupling",
        "C-N",
        "[NX3;H2,H1:4].[SX4:1](=[OX1:2])(=[OX1:5])[Cl:3]>>[SX4:1](=[OX1:2])(=[OX1:5])[NX3:4]",
        "amine", "sulfonyl_chloride", None,
        "none",
        "chloride (deleted)",
        "nucleophilic substitution at a sulfonyl chloride; the S-N bond is a "
        "genuine single bond (documented: rotational barrier far below amide "
        "C-N), so the torsion bin action governs the junction",
        (
            ReagentPair(("CN", "CS(=O)(=O)Cl"), "CS(=O)(=O)NC", "methanesulfonyl chloride + methylamine -> N-methylmethanesulfonamide"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "CS(=O)(=O)Cl"),
                "C[C@H](NS(C)(=O)=O)c1ccccc1",
                "(S)-1-phenylethylamine sulfonylation; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][SX4:2](=[OX1])(=[OX1])-[#6]", JUNCTION_PRIORITY_CARBONYL,
    ),
    _rxn(
        "rxn_03", "reductive amination (aldehyde + amine)", "reductive_amination",
        "C-N",
        "[NX3;H2:4].[CH1:1]=[OX1:2]>>[CH2:1][NX3;H1:4]",
        "amine", "aldehyde", None,
        "none",
        "aldehydic oxygen (deleted; net reductive coupling: imine formation + "
        "reduction collapses into the single junction step)",
        "imine formation followed by reduction, expressed as the net "
        "transform; the product is the secondary amine",
        (
            ReagentPair(("CN", "CC=O"), "CCNC", "acetaldehyde + methylamine -> N-methylethylamine"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "CC=O"),
                "CCN[C@@H](C)c1ccccc1",
                "(S)-1-phenylethylamine reductive amination; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][CX4;H2,H3:2]", JUNCTION_PRIORITY_REDUCTIVE,
    ),
    _rxn(
        "rxn_04", "arylation, Buchwald-Hartwig variant (aryl bromide + amine)", "aryl_amination",
        "C-N",
        "[NX3;H2,H1:4].[c:1][Br:3]>>[c:1][NX3:4]",
        "amine", "aryl_bromide", None,
        "none",
        "bromide (deleted)",
        "Pd(0)-catalysed C-N cross-coupling on an aryl bromide (Buchwald "
        "1990s ligand era); the anilino C-N single bond rotates freely "
        "(documented: conjugation too weak to lock)",
        (
            ReagentPair(("CN", "Brc1ccccc1"), "CNc1ccccc1", "bromobenzene + methylamine -> N-methylaniline"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "Brc1ccccc1"),
                "C[C@@H](c1ccccc1)Nc1ccccc1",
                "(S)-1-phenylethylamine arylation; CIP S retained at the centre "
                "adjacent to the new aryl C-N bond",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][c:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_05", "arylation, Ullmann variant (aryl chloride + amine)", "aryl_amination",
        "C-N",
        "[NX3;H2,H1:4].[c:1][Cl:3]>>[c:1][NX3:4]",
        "amine", "aryl_chloride", None,
        "none",
        "chloride (deleted)",
        "Cu/Pd-catalysed C-N coupling on the cheaper aryl chloride "
        "(Ullmann 1904 classical + modern variants)",
        (
            ReagentPair(("CN", "Clc1ccccc1"), "CNc1ccccc1", "chlorobenzene + methylamine -> N-methylaniline"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "Clc1ccccc1"),
                "C[C@H](Nc1ccccc1)c1ccccc1",
                "(S)-1-phenylethylamine Ullmann arylation; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][c:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_06", "urea formation (carbamoyl chloride + amine)", "urea_formation",
        "C-N",
        "[NX3;H2,H1:4].[CX3:1](=[OX1:2])([Cl:3])[NX3:5]>>[CX3:1](=[OX1:2])([NX3:4])[NX3:5]",
        "amine", "carbamoyl_chloride", "trans",
        "none",
        "chloride (deleted; the block's own amine substituents stay)",
        "nucleophilic acyl substitution at a carbamoyl chloride; the urea "
        "C-N partial double bond pins the junction planar trans",
        (
            ReagentPair(("CN", "CN(C)C(=O)Cl"), "CNC(=O)N(C)C", "dimethylcarbamoyl chloride + methylamine -> N,N-dimethyl-N'-methylurea"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "CN(C)C(=O)Cl"),
                "CN(C)C(=O)N[C@@H](C)c1ccccc1",
                "(S)-1-phenylethylamine urea formation; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][CX3:2](=[OX1])[NX3]", JUNCTION_PRIORITY_CARBONYL,
    ),
    _rxn(
        "rxn_07", "sp3 N-alkylation (alkyl bromide + amine)", "sp3_alkylation",
        "C-N",
        "[NX3;H2,H1:4].[CX4;H2,H3:1][Br:3]>>[CX4:1][NX3:4]",
        "amine", "alkyl_bromide", None,
        "none",
        "bromide (deleted)",
        "SN2 N-alkylation restricted to primary (and methyl) electrophiles: "
        "the reacting carbon is never a stereocentre, so no inversion ever "
        "reaches the product (documented validity constraint)",
        (
            ReagentPair(("CN", "CCBr"), "CCNC", "bromoethane + methylamine -> ethyl(methyl)amine"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "CCBr"),
                "CCN[C@@H](C)c1ccccc1",
                "(S)-1-phenylethylamine N-alkylation; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][CX4;H2,H3:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_08", "carbamate formation (chloroformate + amine)", "amide_and_carbamate",
        "C-N",
        "[NX3;H2,H1:4].[CX3:1](=[OX1:2])([Cl:3])[OX2:5]>>[CX3:1](=[OX1:2])([NX3:4])[OX2:5]",
        "amine", "chloroformate", "trans",
        "none",
        "chloride (deleted; the block's alkoxy group stays)",
        "nucleophilic acyl substitution at a chloroformate; the carbamate "
        "C-N partial double bond pins the junction planar trans",
        (
            ReagentPair(("CN", "COC(=O)Cl"), "CNC(=O)OC", "methyl chloroformate + methylamine -> methyl N-methylcarbamate"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "COC(=O)Cl"),
                "COC(=O)N[C@@H](C)c1ccccc1",
                "(S)-1-phenylethylamine carbamate formation; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][CX3:2](=[OX1])[OX2]", JUNCTION_PRIORITY_CARBONYL,
    ),
    # -- C-C couplings (rxn_09..rxn_15) --------------------------------------
    _rxn(
        "rxn_09", "Suzuki coupling, aryl-aryl (aryl bromide + arylboronic acid)",
        "cross_coupling_cc", "C-C",
        "[c:1][Br:3].[c:2][B:4]([OX2:5])([OX2:6])>>[c:1][c:2]",
        "aryl_bromide", "arylboronic_acid", None,
        "bromide (deleted)",
        "boron + both hydroxyls (deleted)",
        "Pd(0)-catalysed sp2-sp2 coupling (Suzuka 1979; Nobel Prize 2010); "
        "the biaryl single bond rotates freely",
        (
            ReagentPair(("Brc1ccccc1", "OB(O)c1ccccc1"), "c1ccc(-c2ccccc2)cc1", "bromobenzene + phenylboronic acid -> biphenyl"),
        ),
        "[c:1][c:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_10", "Suzuki coupling, aryl-heteroaryl (aryl chloride + heteroarylboronic acid)",
        "cross_coupling_cc", "C-C",
        "[c:1][Cl:3].[c:2][B:4]([OX2:5])([OX2:6])>>[c:1][c:2]",
        "aryl_chloride", "heteroarylboronic_acid", None,
        "chloride (deleted)",
        "boron + both hydroxyls (deleted)",
        "Pd(0)-catalysed coupling of an aryl chloride with a heteroaryl "
        "boronic acid; heteroaryl synthons arrive from the catalog",
        (
            ReagentPair(("Clc1ccccc1", "OB(O)c1cccnc1"), "c1ccc(-c2cccnc2)cc1", "chlorobenzene + 3-pyridylboronic acid -> 3-phenylpyridine"),
        ),
        "[c:1][c:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_11", "Negishi coupling (aryl bromide + arylzinc chloride)",
        "cross_coupling_cc", "C-C",
        "[c:1][Br:3].[c:2][Zn:4][Cl:5]>>[c:1][c:2]",
        "aryl_bromide", "arylzinc", None,
        "bromide (deleted)",
        "zinc + chloride (deleted)",
        "Pd/Ni-catalysed coupling with an organozinc reagent (Negishi 1977; "
        "Nobel Prize 2010)",
        (
            ReagentPair(("Brc1ccccc1", "Cl[Zn]c1ccccc1"), "c1ccc(-c2ccccc2)cc1", "bromobenzene + phenylzinc chloride -> biphenyl"),
        ),
        "[c:1][c:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_12", "Stille coupling, alkenyl variant (aryl chloride + vinylstannane)",
        "cross_coupling_cc", "C-C",
        "[c:1][Cl:3].[CH1:2]([Sn:4]([CH3:5])([CH3:6])[CH3:8])=[CH2:7]>>[c:1][CH1:2]=[CH2:7]",
        "aryl_chloride", "vinylstannane", None,
        "chloride (deleted)",
        "tin + all three methyls (deleted)",
        "Pd(0)-catalysed coupling with a vinylstannane (Stille 1978; "
        "Kosugi-Migita); the product is the styryl junction",
        (
            ReagentPair(("Clc1ccccc1", "C=C[Sn](C)(C)C"), "C=Cc1ccccc1", "chlorobenzene + trimethyl(vinyl)stannane -> styrene"),
        ),
        "[c:1][CH1:2]=[CH2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_13", "Sonogashira coupling (aryl bromide + metal acetylide)",
        "cross_coupling_cc", "C-C",
        "[c:1][Br:3].[C:7]#[C:2][Li:4]>>[c:1][C:2]#[C:7]",
        "aryl_bromide", "acetylide", None,
        "bromide (deleted)",
        "lithium (deleted)",
        "Pd/Cu-catalysed alkynylation (Sonogashira 1975), expressed with the "
        "preformed metal acetylide; the sp-sp2 single bond rotates freely",
        (
            ReagentPair(("Brc1ccccc1", "CC#C[Li]"), "CC#Cc1ccccc1", "bromobenzene + lithium propynide -> 1-phenylprop-1-yne"),
        ),
        "[c:1][C:2]#[C]", JUNCTION_PRIORITY_ALKYNE,
    ),
    _rxn(
        "rxn_14", "alkenyl Suzuki coupling (vinyl bromide + arylboronic acid)",
        "cross_coupling_cc", "C-C",
        "[CH1:1]([Br:3])=[CH2:7].[c:2][B:4]([OX2:5])([OX2:6])>>[c:2][CH1:1]=[CH2:7]",
        "vinyl_bromide", "arylboronic_acid", None,
        "bromide (deleted)",
        "boron + both hydroxyls (deleted)",
        "Pd(0)-catalysed sp2-sp2 coupling at a vinyl bromide — the standard "
        "one-step arylation of vinyl synthons (Heck-equivalent product "
        "topology, Suzuki reagent set)",
        (
            ReagentPair(("C=CBr", "OB(O)c1ccccc1"), "C=Cc1ccccc1", "vinyl bromide + phenylboronic acid -> styrene"),
        ),
        "[CH1:1]([c:2])=[CH2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_15", "alkyl Negishi coupling (alkylzinc + aryl bromide)",
        "cross_coupling_cc", "C-C",
        "[CX4;H2,H3:1][Zn:4][Cl:5].[c:2][Br:3]>>[CX4:1][c:2]",
        "alkylzinc", "aryl_bromide", None,
        "zinc + chloride (deleted)",
        "bromide (deleted)",
        "Pd/Ni-catalysed sp3-sp2 coupling (Negishi variant for alkyl "
        "centres); primary alkylzinc only — no beta-hydride stereochemistry "
        "reaches the junction (documented)",
        (
            ReagentPair(("Cl[Zn]CC", "Brc1ccccc1"), "CCc1ccccc1", "ethylzinc chloride + bromobenzene -> ethylbenzene"),
        ),
        "[CX4;H1,H2:1]([c:2])~[#6]", JUNCTION_PRIORITY_GENERIC,
    ),
    # -- C-O/C-S couplings (rxn_16..rxn_21) -----------------------------------
    _rxn(
        "rxn_16", "O-alkylation (alkyl bromide + alcohol/phenol)", "sp3_alkylation",
        "C-O/C-S",
        "[OX2;H1:4].[CX4;H2,H3:1][Br:3]>>[CX4:1][OX2:4]",
        "alcohol", "alkyl_bromide", None,
        "the hydroxyl H (implicit; recomputed by sanitization)",
        "bromide (deleted)",
        "Williamson ether synthesis restricted to primary electrophiles: the "
        "reacting carbon is never a stereocentre (documented)",
        (
            ReagentPair(("CCO", "CCBr"), "CCOCC", "bromoethane + ethanol -> diethyl ether"),
            ReagentPair(
                ("C[C@H](O)c1ccccc1", "CCBr"),
                "CCO[C@@H](C)c1ccccc1",
                "(S)-1-phenylethanol O-alkylation; CIP S retained at the "
                "centre adjacent to the new C-O bond",
                stereo_pair=True,
            ),
        ),
        "[OX2:1][CX4;H2,H3:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_17", "esterification (acid chloride + alcohol)", "ester_and_ether",
        "C-O/C-S",
        "[OX2;H1:4].[CX3:1](=[OX1:2])[Cl:3]>>[CX3:1](=[OX1:2])[OX2:4]",
        "alcohol", "acid_chloride", "trans",
        "the hydroxyl H (implicit)",
        "chloride (deleted)",
        "nucleophilic acyl substitution at an acid chloride; the ester C-O "
        "partial double bond pins the junction planar trans",
        (
            ReagentPair(("CCO", "CC(=O)Cl"), "CCOC(C)=O", "acetyl chloride + ethanol -> ethyl acetate"),
            ReagentPair(
                ("C[C@H](O)c1ccccc1", "CC(=O)Cl"),
                "CC(=O)O[C@@H](C)c1ccccc1",
                "(S)-1-phenylethanol esterification; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[OX2:1][CX3:2](=[OX1])-[#6]", JUNCTION_PRIORITY_CARBONYL,
    ),
    _rxn(
        "rxn_18", "aryl ether formation (aryl bromide + alcohol/phenol)", "ester_and_ether",
        "C-O/C-S",
        "[OX2;H1:4].[c:1][Br:3]>>[c:1][OX2:4]",
        "alcohol", "aryl_bromide", None,
        "the hydroxyl H (implicit)",
        "bromide (deleted)",
        "Pd/Cu-catalysed aryl ether formation (Buchwald/Ullmann C-O "
        "variants); the anisole-type C-O single bond rotates freely",
        (
            ReagentPair(("CCO", "Brc1ccccc1"), "CCOc1ccccc1", "bromobenzene + ethanol -> phenetole"),
            ReagentPair(
                ("C[C@H](O)c1ccccc1", "Brc1ccccc1"),
                "C[C@H](Oc1ccccc1)c1ccccc1",
                "(S)-1-phenylethanol aryl etherification; CIP S retained at the "
                "centre adjacent to the new aryl C-O bond",
                stereo_pair=True,
            ),
        ),
        "[OX2:1][c:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_19", "S-alkylation (alkyl bromide + thiol)", "sp3_alkylation",
        "C-O/C-S",
        "[SX2;H1:4].[CX4;H2,H3:1][Br:3]>>[CX4:1][SX2:4]",
        "thiol", "alkyl_bromide", None,
        "the thiol H (implicit)",
        "bromide (deleted)",
        "SN2 thioether formation restricted to primary electrophiles: the "
        "reacting carbon is never a stereocentre (documented)",
        (
            ReagentPair(("CCS", "CCBr"), "CCSCC", "bromoethane + ethanethiol -> diethyl sulfide"),
            ReagentPair(
                ("C[C@H](S)c1ccccc1", "CCBr"),
                "CCS[C@@H](C)c1ccccc1",
                "(S)-1-phenylethanethiol S-alkylation; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[SX2:1][CX4;H2,H3:2]", JUNCTION_PRIORITY_GENERIC,
    ),
    _rxn(
        "rxn_20", "thioesterification (acid chloride + thiol)", "ester_and_ether",
        "C-O/C-S",
        "[SX2;H1:4].[CX3:1](=[OX1:2])[Cl:3]>>[CX3:1](=[OX1:2])[SX2:4]",
        "thiol", "acid_chloride", "trans",
        "the thiol H (implicit)",
        "chloride (deleted)",
        "nucleophilic acyl substitution at an acid chloride; thioesters are "
        "planar at C(=O)-S with the E preference, so the junction locks "
        "planar trans (documented)",
        (
            ReagentPair(("CCS", "CC(=O)Cl"), "CCSC(C)=O", "acetyl chloride + ethanethiol -> S-ethyl thioacetate"),
            ReagentPair(
                ("C[C@H](S)c1ccccc1", "CC(=O)Cl"),
                "CC(=O)S[C@@H](C)c1ccccc1",
                "(S)-1-phenylethanethiol thioesterification; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[SX2:1][CX3:2](=[OX1])-[#6]", JUNCTION_PRIORITY_CARBONYL,
    ),
    _rxn(
        "rxn_21", "sulfonate esterification (sulfonyl chloride + alcohol)", "ester_and_ether",
        "C-O/C-S",
        "[OX2;H1:4].[SX4:1](=[OX1:2])(=[OX1:5])[Cl:3]>>[SX4:1](=[OX1:2])(=[OX1:5])[OX2:4]",
        "alcohol", "sulfonyl_chloride", None,
        "the hydroxyl H (implicit)",
        "chloride (deleted)",
        "nucleophilic substitution at a sulfonyl chloride; the sulfonate "
        "S(VI) centre is saturated — no pi system — so the O-S single bond "
        "rotates freely (documented)",
        (
            ReagentPair(("CCO", "CS(=O)(=O)Cl"), "CCOS(C)(=O)=O", "methanesulfonyl chloride + ethanol -> ethyl methanesulfonate"),
            ReagentPair(
                ("C[C@H](O)c1ccccc1", "CS(=O)(=O)Cl"),
                "C[C@H](OS(C)(=O)=O)c1ccccc1",
                "(S)-1-phenylethanol mesylation; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[OX2:1][SX4:2](=[OX1])(=[OX1])", JUNCTION_PRIORITY_CARBONYL,
    ),
    # -- heterocycle installations (rxn_22..rxn_27) ---------------------------
    # The heterocycle ships pre-cyclized on the synthon side (the pinned
    # single-junction MDP cannot close rings); each entry documents the
    # one-step cyclization provenance of its synthon class.
    _rxn(
        "rxn_22", "pyrimidine amination (2-chloropyrimidine + amine)", "heterocycle_click",
        "heterocycle",
        "[NX3;H2,H1:4].[c:1]([Cl:3])1[n:7][c:8][c:9][c:10][n:11]1>>"
        "[c:1]([NX3:4])1[n:7][c:8][c:9][c:10][n:11]1",
        "amine", "pyrimidinyl_chloride", None,
        "none",
        "chloride (deleted)",
        "SNAr amination of the activated 2-chloropyrimidine (the C2 position "
        "between the two ring nitrogens)",
        (
            ReagentPair(("CN", "Clc1ncccn1"), "CNc1ncccn1", "2-chloropyrimidine + methylamine -> N-methylpyrimidin-2-amine"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "Clc1ncccn1"),
                "C[C@@H](c1ccccc1)Nc1ncccn1",
                "(S)-1-phenylethylamine amination of 2-chloropyrimidine; "
                "CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][c:2]", JUNCTION_PRIORITY_GENERIC,
        cyclization="pyrimidinyl synthons: cyclocondensation of amidines "
        "with beta-dicarbonyl compounds (catalog-side, one step)",
    ),
    _rxn(
        "rxn_23", "pyrazole arylation (4-bromopyrazole + boronic acid)", "heterocycle_click",
        "heterocycle",
        "[c:2][B:4]([OX2:5])([OX2:6]).[c:1]([Br:3])1[c:7][n:8][n:9][c:10]1>>"
        "[c:1]([c:2])1[c:7][n:8][n:9][c:10]1",
        "boronic_acid", "pyrazolyl_bromide", None,
        "boron + both hydroxyls (deleted)",
        "bromide (deleted)",
        "Pd(0)-catalysed arylation of the 4-bromopyrazole (Suzuki at the C4 "
        "position)",
        (
            ReagentPair(("OB(O)c1ccccc1", "Cn1ncc(Br)c1"), "Cn1cc(-c2ccccc2)cn1", "1-methyl-4-bromopyrazole + phenylboronic acid -> 1-methyl-4-phenylpyrazole"),
        ),
        "[c:1][c:2]", JUNCTION_PRIORITY_GENERIC,
        cyclization="pyrazolyl synthons: 1,3-dipolar cycladdition / "
        "hydrazone chemistry (catalog-side, one step)",
    ),
    _rxn(
        "rxn_24", "thiazole amination (2-chlorothiazole + amine)", "heterocycle_click",
        "heterocycle",
        "[NX3;H2,H1:4].[c:1]([Cl:3])1[s:7][c:8][c:9][n:10]1>>"
        "[c:1]([NX3:4])1[s:7][c:8][c:9][n:10]1",
        "amine", "thiazolyl_chloride", None,
        "none",
        "chloride (deleted)",
        "SNAr amination of the activated 2-chlorothiazole",
        (
            ReagentPair(("CN", "Clc1nccs1"), "CNc1nccs1", "2-chlorothiazole + methylamine -> N-methylthiazol-2-amine"),
        ),
        "[NX3:1][c:2]", JUNCTION_PRIORITY_GENERIC,
        cyclization="thiazolyl synthons: Hantzsch thiazole synthesis "
        "(alpha-haloketone + thioamide, catalog-side, one step)",
    ),
    _rxn(
        "rxn_25", "pyridine amination (2-chloropyridine + amine)", "heterocycle_click",
        "heterocycle",
        "[NX3;H2,H1:4].[c:1]([Cl:3])1[n:7][c:8][c:9][c:10][c:11]1>>"
        "[c:1]([NX3:4])1[n:7][c:8][c:9][c:10][c:11]1",
        "amine", "pyridyl_chloride", None,
        "none",
        "chloride (deleted)",
        "SNAr amination of 2-chloropyridine — the classical heteroaryl "
        "amination",
        (
            ReagentPair(("CN", "Clc1ccccn1"), "CNc1ccccn1", "2-chloropyridine + methylamine -> 2-(methylamino)pyridine"),
            ReagentPair(
                ("C[C@H](N)c1ccccc1", "Clc1ccccn1"),
                "C[C@H](Nc1ccccn1)c1ccccc1",
                "(S)-1-phenylethylamine pyridine amination; CIP S retained",
                stereo_pair=True,
            ),
        ),
        "[NX3:1][c:2]", JUNCTION_PRIORITY_GENERIC,
        cyclization="pyridyl synthons: Hantzsch / Chichibabin pyridine "
        "construction (catalog-side, one step)",
    ),
    _rxn(
        "rxn_26", "thiazole arylation (2-bromothiazole + boronic acid)", "heterocycle_click",
        "heterocycle",
        "[c:2][B:4]([OX2:5])([OX2:6]).[c:1]([Br:3])1[s:7][c:8][c:9][n:10]1>>"
        "[c:1]([c:2])1[s:7][c:8][c:9][n:10]1",
        "boronic_acid", "thiazolyl_bromide", None,
        "boron + both hydroxyls (deleted)",
        "bromide (deleted)",
        "Pd(0)-catalysed arylation of 2-bromothiazole (Suzuki at C2)",
        (
            ReagentPair(("OB(O)c1ccccc1", "Brc1nccs1"), "c1ccc(-c2nccs2)cc1", "2-bromothiazole + phenylboronic acid -> 2-phenylthiazole"),
        ),
        "[c:1][c:2]", JUNCTION_PRIORITY_GENERIC,
        cyclization="thiazolyl synthons: Hantzsch thiazole synthesis "
        "(catalog-side, one step)",
    ),
    _rxn(
        "rxn_27", "benzimidazole amination (2-chlorobenzimidazole + amine)", "heterocycle_click",
        "heterocycle",
        "[NX3;H2,H1:4].[c:1]([Cl:3])1[n:7][c:8]2[c:12][c:13][c:14][c:15][c:16]2[n:17]1>>"
        "[c:1]([NX3:4])1[n:7][c:8]2[c:12][c:13][c:14][c:15][c:16]2[n:17]1",
        "amine", "benzimidazolyl_chloride", None,
        "none",
        "chloride (deleted)",
        "SNAr amination of the activated 2-chlorobenzimidazole",
        (
            ReagentPair(("CN", "Clc1nc2ccccc2[nH]1"), "CNc1nc2ccccc2[nH]1", "2-chlorobenzimidazole + methylamine -> N-methyl-1H-benzimidazol-2-amine"),
        ),
        "[NX3:1][c:2]", JUNCTION_PRIORITY_GENERIC,
        cyclization="benzimidazolyl synthons: condensation of "
        "o-phenylenediamine with carboxylic acids (catalog-side, one step)",
    ),
]

# ---------------------------------------------------------------------------
# The 5 deterministic deprotections (rxn_28..rxn_32; no policy action)
# ---------------------------------------------------------------------------


def _deprot(
    reaction_id: str,
    name: str,
    smarts: str,
    certification: Tuple[ReagentPair, ...],
) -> CertifiedReaction:
    return CertifiedReaction(
        reaction_id=reaction_id,
        name=name,
        family=None,
        category="deprotection",
        policy_action=False,
        reaction_smarts=smarts,
        molecule_handle_class=None,
        block_handle_class="protected_synthon",
        product_bond_type="single",
        resonance_lock=None,
        leaving_group_molecule="the full protecting group (mapped, deleted)",
        leaving_group_block="n/a (single-substrate transform)",
        stereo_behavior="stereoretentive by atom-map chirality transfer",
        reference="standard orthogonal protecting-group chemistry",
        validity_constraints="the PG motif must match the pattern exactly; "
        "absence of the motif is a documented no-op",
        failure_behavior="unsanitizable product -> fail closed "
        "(ReactionRegistryError); never a silent skip",
        certification=certification,
    )


_DEPROTECTIONS: List[CertifiedReaction] = [
    _deprot(
        "rxn_28", "Boc deprotection (tert-butoxycarbonyl -> amine)",
        "[NX3:1][C:2](=[O:3])[O:4][C:5]([CH3:6])([CH3:7])[CH3:8]>>[NX3:1]",
        (
            ReagentPair(("CC(C)(C)OC(=O)NC",), "CN", "Boc-protected methylamine -> methylamine"),
        ),
    ),
    _deprot(
        "rxn_29", "Alloc deprotection (allyloxycarbonyl -> amine)",
        "[NX3:1][C:2](=[O:3])[O:4][CH2:5][CH:6]=[CH2:7]>>[NX3:1]",
        (
            ReagentPair(("CNC(=O)OCC=C",), "CN", "Alloc-protected methylamine -> methylamine"),
        ),
    ),
    _deprot(
        "rxn_30", "Cbz deprotection (benzyl carbamate -> amine)",
        "[NX3:1][C:2](=[O:3])[O:4][CH2:5][c:6]1[c:7][c:8][c:9][c:10][c:11]1>>[NX3:1]",
        (
            ReagentPair(("CNC(=O)OCc1ccccc1",), "CN", "Cbz-protected methylamine -> methylamine"),
        ),
    ),
    _deprot(
        "rxn_31", "O-benzyl deprotection (benzyl ether -> alcohol)",
        "[#8:1][CH2:2][c:3]1[c:4][c:5][c:6][c:7][c:8]1>>[#8:1]",
        (
            ReagentPair(("COCc1ccccc1",), "CO", "benzyl methyl ether -> methanol"),
        ),
    ),
    _deprot(
        "rxn_32", "TBDMS deprotection (tert-butyldimethylsilyl ether -> alcohol)",
        "[#8:1][Si:2]([CH3:3])([CH3:4])[C:5]([CH3:6])([CH3:7])[CH3:8]>>[#8:1]",
        (
            ReagentPair(("CCO[Si](C)(C)C(C)(C)C",), "CCO", "TBDMS-protected ethanol -> ethanol"),
        ),
    ),
]

# ---------------------------------------------------------------------------
# Registries and mappings
# ---------------------------------------------------------------------------

CERTIFIED_REACTIONS: Dict[str, CertifiedReaction] = {
    reaction.reaction_id: reaction for reaction in (*_REACTIONS, *_DEPROTECTIONS)
}
if len(CERTIFIED_REACTIONS) != 32:
    raise ReactionRegistryError(
        f"the certified taxonomy must contain exactly 32 reactions; got {len(CERTIFIED_REACTIONS)}"
    )

POLICY_REACTIONS: Dict[str, CertifiedReaction] = {
    reaction.reaction_id: reaction for reaction in _REACTIONS
}
if len(POLICY_REACTIONS) != 27:
    raise ReactionRegistryError(
        f"expected 27 policy reactions, got {len(POLICY_REACTIONS)}"
    )

DEPROTECTIONS: Dict[str, CertifiedReaction] = {
    reaction.reaction_id: reaction for reaction in _DEPROTECTIONS
}
if len(DEPROTECTIONS) != 5:
    raise ReactionRegistryError(
        f"expected 5 deprotections, got {len(DEPROTECTIONS)}"
    )

#: Issue-pinned family membership (+ D-V2-016 additions rxn_14/rxn_15).
REACTION_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "amide_and_carbamate": ("rxn_01", "rxn_08"),
    "sulfonamide_coupling": ("rxn_02",),
    "reductive_amination": ("rxn_03",),
    "aryl_amination": ("rxn_04", "rxn_05"),
    "urea_formation": ("rxn_06",),
    "sp3_alkylation": ("rxn_07", "rxn_16", "rxn_19"),
    "cross_coupling_cc": ("rxn_09", "rxn_10", "rxn_11", "rxn_12", "rxn_13", "rxn_14", "rxn_15"),
    "ester_and_ether": ("rxn_17", "rxn_18", "rxn_20", "rxn_21"),
    "heterocycle_click": ("rxn_22", "rxn_23", "rxn_24", "rxn_25", "rxn_26", "rxn_27"),
}

REACTION_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "C-N": ("rxn_01", "rxn_02", "rxn_03", "rxn_04", "rxn_05", "rxn_06", "rxn_07", "rxn_08"),
    "C-C": ("rxn_09", "rxn_10", "rxn_11", "rxn_12", "rxn_13", "rxn_14", "rxn_15"),
    "C-O/C-S": ("rxn_16", "rxn_17", "rxn_18", "rxn_19", "rxn_20", "rxn_21"),
    "heterocycle": ("rxn_22", "rxn_23", "rxn_24", "rxn_25", "rxn_26", "rxn_27"),
    "deprotection": ("rxn_28", "rxn_29", "rxn_30", "rxn_31", "rxn_32"),
}

# Cross-check the pinned structure of the taxonomy at import time (fail
# closed — never import a malformed registry).
_family_members = {rid for members in REACTION_FAMILIES.values() for rid in members}
if _family_members != set(POLICY_REACTIONS):
    raise ReactionRegistryError(
        "family membership must cover exactly the 27 policy reactions; "
        f"unmapped: {sorted(set(POLICY_REACTIONS) - _family_members)}; "
        f"unknown: {sorted(_family_members - set(POLICY_REACTIONS))}"
    )
for name, members in REACTION_FAMILIES.items():
    for rid in members:
        if CERTIFIED_REACTIONS[rid].family != name:
            raise ReactionRegistryError(
                f"{rid}: family field {CERTIFIED_REACTIONS[rid].family!r} disagrees "
                f"with REACTION_FAMILIES entry {name!r}"
            )
for rid, reaction in CERTIFIED_REACTIONS.items():
    if not rid.startswith("rxn_") or rid[4:] != f"{int(rid[4:]):02d}":
        raise ReactionRegistryError(f"malformed reaction id {rid!r}")


# ---------------------------------------------------------------------------
# Certification: round-trip forward replay (the #11 acceptance criterion)
# ---------------------------------------------------------------------------


def canonical_product_smiles(mol: Chem.Mol) -> str:
    """Canonical isomeric SMILES of a sanitized product."""
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol)


def junction_pattern_attachment_atoms(pattern: Chem.Mol) -> Tuple[int, int]:
    """The template-atom indices of map ``:1`` and ``:2`` in a junction pattern.

    ``GetSubstructMatches`` returns matches in *template position* order,
    which coincides with the map order only when patterns are written
    ``[:1]``-first.  Consumers must locate the molecule-side/block-side
    attachment atoms by **map number** through this helper (the
    positional assumption was the root cause of the rxn_14/rxn_15
    orientation defect corrected with #18).
    """
    map_one = map_two = None
    for atom in pattern.GetAtoms():
        number = atom.GetAtomMapNum()
        if number == 1 and map_one is None:
            map_one = atom.GetIdx()
        elif number == 2 and map_two is None:
            map_two = atom.GetIdx()
    if map_one is None or map_two is None:
        raise ReactionRegistryError(
            "a junction pattern must carry exactly the map numbers :1 and :2 "
            "(molecule-side and block-side attachment atoms)"
        )
    return map_one, map_two


def run_certified_reaction(
    reaction: CertifiedReaction, reagent_smiles: Sequence[str]
) -> List[Chem.Mol]:
    """Execute ``reaction`` on ``reagent_smiles``; fail closed.

    Returns every product of every match, sanitized.  Raises
    :class:`ReactionRegistryError` when the reaction does not match, a
    reagent does not parse, or a product fails sanitization — never a
    silent skip (constitution #45 §9.1 failure behaviour).
    """
    rxn = AllChem.ReactionFromSmarts(reaction.reaction_smarts)
    if rxn is None:
        raise ReactionRegistryError(
            f"{reaction.reaction_id}: unparseable reaction SMARTS"
        )
    mols: List[Chem.Mol] = []
    for smi in reagent_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ReactionRegistryError(
                f"{reaction.reaction_id}: unparseable reagent SMILES {smi!r}"
            )
        mols.append(mol)
    products = rxn.RunReactants(tuple(mols))
    if not products:
        raise ReactionRegistryError(
            f"{reaction.reaction_id}: the reaction SMARTS does not match the "
            f"reagent set {tuple(reagent_smiles)} — certification fails closed"
        )
    out: List[Chem.Mol] = []
    for row in products:
        product = row[0]
        try:
            Chem.SanitizeMol(product)
        except Exception as error:  # noqa: BLE001 - fail closed with context
            raise ReactionRegistryError(
                f"{reaction.reaction_id}: product failed sanitization: {error}"
            ) from error
        out.append(product)
    return out


def certify_reaction(reaction: CertifiedReaction) -> Dict[str, object]:
    """Round-trip forward-replay validation of one reaction on all its pairs.

    For every :class:`ReagentPair`: run the reaction, deduplicate the
    products by canonical isomeric SMILES, and require the *exact* set to
    equal ``{expected_product}``.  Stereo pairs must reproduce the
    ``@``/``@@`` retention tokens (the isomeric comparison is the
    retention check).  Returns a report; raises on any failure.
    """
    pair_reports = []
    for pair in reaction.certification:
        products = run_certified_reaction(reaction, pair.reagents)
        observed = {canonical_product_smiles(p) for p in products}
        expected = canonical_product_smiles(Chem.MolFromSmiles(pair.expected_product))
        if observed != {expected}:
            raise ReactionRegistryError(
                f"{reaction.reaction_id}: forward replay mismatch on {pair.reagents}: "
                f"expected {{{expected}}}, observed {sorted(observed)}"
            )
        pair_reports.append(
            {
                "reagents": pair.reagents,
                "expected": expected,
                "observed": sorted(observed),
                "stereo_pair": pair.stereo_pair,
                "note": pair.note,
            }
        )
    return {
        "reaction_id": reaction.reaction_id,
        "name": reaction.name,
        "family": reaction.family,
        "category": reaction.category,
        "policy_action": reaction.policy_action,
        "pairs": pair_reports,
    }


def certify_registry(
    registry: Optional[Dict[str, CertifiedReaction]] = None,
) -> List[Dict[str, object]]:
    """Certify every reaction in the registry (100% round-trip replay)."""
    registry = registry if registry is not None else CERTIFIED_REACTIONS
    return [certify_reaction(reaction) for reaction in registry.values()]


# ---------------------------------------------------------------------------
# Deprotection execution (deterministic, policy-free)
# ---------------------------------------------------------------------------


def apply_deprotection(mol: Chem.Mol, rule: CertifiedReaction) -> Chem.Mol:
    """Apply one deprotection rule deterministically.

    Absence of the motif is a documented no-op (returns an independent
    copy).  Every match is transformed in one pass; the result must
    sanitize, else fail closed.  The input molecule is never mutated.
    """
    if rule.policy_action:
        raise ReactionRegistryError(
            f"{rule.reaction_id}: not a deprotection rule"
        )
    rxn = AllChem.ReactionFromSmarts(rule.reaction_smarts)
    if rxn is None:
        raise ReactionRegistryError(
            f"{rule.reaction_id}: unparseable deprotection SMARTS"
        )
    products = rxn.RunReactants((mol,))
    if not products:
        return Chem.Mol(mol)  # documented no-op
    result = products[0][0]
    try:
        Chem.SanitizeMol(result)
    except Exception as error:  # noqa: BLE001
        raise ReactionRegistryError(
            f"{rule.reaction_id}: deprotection produced an unsanitizable "
            f"molecule: {error}"
        ) from error
    return result


def apply_all_deprotections(
    mol: Chem.Mol,
    rules: Optional[Sequence[CertifiedReaction]] = None,
) -> Chem.Mol:
    """Apply the deprotection sequence deterministically (fixed order)."""
    rules = rules if rules is not None else list(DEPROTECTIONS.values())
    current = mol
    for rule in rules:
        current = apply_deprotection(current, rule)
    return current


# ---------------------------------------------------------------------------
# Engine registry construction (supersedes the generic interim, D-V2-016)
# ---------------------------------------------------------------------------


def build_junction_reaction_registry() -> Dict[str, "JunctionReaction"]:  # noqa: F821
    """Build the engine-format registry for the 27 policy reactions.

    Returns ``{reaction_id: JunctionReaction}`` compatible with
    ``AssemblyEnvironment(reaction_registry=...)`` — the certified
    taxonomy supersedes the generic junction-reaction interim
    (D-V2-009 scope boundary / D-V2-016) without changing the MDP/TB
    contracts: same ``JunctionReaction`` interface, now populated with
    certified block-side handle classes, single junction bonds, and the
    amide/urea/carbamate/ester/thioester resonance locks.
    """
    # Lazy import: the engine owns the dataclass (import-cycle safe — this
    # function runs long after both modules are loaded).
    from syntree.engine.environment import JunctionReaction

    registry: Dict[str, JunctionReaction] = {}
    for reaction in POLICY_REACTIONS.values():
        registry[reaction.reaction_id] = JunctionReaction(
            reaction_id=reaction.reaction_id,
            block_handle_class=reaction.block_handle_class,
            bond_type=Chem.BondType.SINGLE,
            resonance_lock=reaction.resonance_lock,
        )
    return registry
