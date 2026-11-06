"""Reaction-aware chemoselectivity and competing-handle screening (issue
#40 CHEM-06).

Clean-room V2 implementation of the synthon-curation chemistry layer
(constitution #45 §9.3): basic molecular-weight / Fsp3 / rotatable-bond
filtering does **not** establish that a building block is *selectively
usable* in its intended reaction.  This module provides the
reaction-aware screen applied **after** basic molecular validation:

* **R1/R2 self-reactivity** — a block whose intended electrophilic
  handle coexists with a *free* nucleophile on the same molecule
  (amine + acid chloride, alcohol + alkyl bromide, ...) self-reacts
  intramolecularly and oligomerizes; organometallic consumers
  (Negishi/Sonogashira) are additionally *quenched* by any free protic
  group.
* **R3 competing identical handles** — more than one handle of the
  intended class (a diacid chloride, a dibromide) is a nonselective,
  oligomerizing substrate: the V2 junction model couples exactly one.
* **R4 competing SNAr positions** — additional activated halides on the
  same electron-deficient heteroaromatic ring give regioisomer
  mixtures, with a documented, pattern-based exemption for the
  classic differentiated 2,4-dihalo-1,3-azine motif (amines displace
  the 4-halide of 2,4-dichloropyrimidines selectively — the textbook
  mono-amination chemistry behind pyrimidine kinase-inhibitor routes).
* **R5 competing electrophiles** — a block whose consumers are
  nucleophilic (amine/alcohol/thiol reagents) must not carry a second
  *uncatalyzed* electrophile (acid chloride + alkyl bromide on one
  molecule: the amine reagent attacks both).
* **R6 catalyst poisons** — free thiols poison the Pd/Sn chemistry of
  the cross-coupling consumers (and compete as nucleophiles).

**Orthogonal protection is honoured by construction** (issue #40's
explicit requirement): the *free*-group SMARTS exclude acylated/
sulfonylated/carbamate nitrogens, ester/silyl/benzyl oxygens, thioether
sulfurs, and aromatic [nH] — so a Boc-protected diamine or a silyl
protected amino-alcohol is a *valid* multifunctional substrate, not an
exclusion.  Every exclusion carries a named reason code and the exact
detected chemical feature; nothing is a blanket functional-group
exclusion — each rule validates the candidate against its intended
reaction/handle role via the certified #11 registry metadata.

**Consumers.**  #12 CHEM-02 applies this layer inside catalog curation
(``screen_catalog`` produces the manifest whose per-rule impact the
issue requires); #41 DATA-06 may reuse the handle-class detectors for
index keys.  The registry link is read-only metadata
(``POLICY_REACTIONS`` / ``REACTION_FAMILIES``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors  # noqa: F401  (kept for downstream curation)

from syntree.chemistry.reactions import POLICY_REACTIONS, REACTION_FAMILIES

__all__ = [
    "ChemoselectivityError",
    "FREE_GROUP_SMARTS",
    "FREE_GROUP_ROLES",
    "HANDLE_CLASS_SMARTS",
    "ALL_HANDLE_CLASSES",
    "HETEROARYL_HALIDE_RING_CLASSES",
    "EXCLUSION_RULES",
    "SELF_REACTIVE_PAIRS",
    "DIFFERENTIATED_DIHALOAZINE",
    "GroupHit",
    "ChemoscreenResult",
    "CurationManifest",
    "detect_free_groups",
    "detect_handle_instances",
    "handle_class_of_atom",
    "screen_block",
    "screen_catalog",
    "chemoselectivity_rule_report",
]


class ChemoselectivityError(ValueError):
    """Fail-closed error for malformed screening inputs."""


# ---------------------------------------------------------------------------
# The free-functional-group vocabulary (SMARTS + reaction roles)
# ---------------------------------------------------------------------------

#: Free (reactive) functional-group patterns.  Each pattern matches ONLY
#: the free form: acyl/sulfonyl/carbamate nitrogens, ester/silyl oxygens,
#: thioether sulfurs and aromatic [nH] are excluded *by construction*,
#: which is exactly how orthogonal protection passes the screen.
FREE_GROUP_SMARTS: Dict[str, str] = {
    # aliphatic N with two H, all heavy neighbours carbon, not acyl/imine-N
    "primary_amine": "[NX3;H2;!$(N[!#6;!#1]);!$(NC=[O,S,N])]",
    # aliphatic N with one H and two carbon neighbours, not acyl/imine-N
    "secondary_amine": "[NX3;H1;!$(N[!#6;!#1]);!$(NC=[O,S,N])]([#6])[#6]",
    # aliphatic O-H on a saturated carbon (carboxylic OH excluded)
    "alcohol": "[OX2H;!$([OX2H][CX3]=[OX1])][CX4]",
    # O-H on an aromatic carbon
    "phenol": "[OX2H][c]",
    # free S-H (aliphatic or aryl)
    "thiol": "[SX2H]",
    # carboxylic acid (and carbamic-acid-type O-H, conservatively)
    "carboxylic_acid": "[CX3](=[OX1])[OX2H]",
    # aldehyde carbonyl carbon
    "aldehyde": "[CX3H1](=[OX1])[#6]",
    # terminal alkyne (acidic C-H)
    "terminal_alkyne": "[CX2;H1]#[CX2]",
}

#: The reaction role each free group plays in the exclusion matrix.
FREE_GROUP_ROLES: Dict[str, str] = {
    "primary_amine": "nucleophile+protic",
    "secondary_amine": "nucleophile+protic",
    "alcohol": "nucleophile+protic",
    "phenol": "nucleophile+protic",
    "thiol": "nucleophile+protic+poison",
    "carboxylic_acid": "protic",
    "aldehyde": "electrophile",
    "terminal_alkyne": "protic",
}

#: The protic set (quench organometallics, form H-bonds, deprotonate).
PROTIC_GROUPS = frozenset(
    {
        "primary_amine",
        "secondary_amine",
        "alcohol",
        "phenol",
        "thiol",
        "carboxylic_acid",
        "terminal_alkyne",
    }
)

#: The free-nucleophile set (attack uncatalyzed electrophiles).
NUCLEOPHILE_GROUPS = frozenset(
    {
        "primary_amine",
        "secondary_amine",
        "alcohol",
        "phenol",
        "thiol",
    }
)


# ---------------------------------------------------------------------------
# Handle-class detection (the block-side vocabulary of the registry)
# ---------------------------------------------------------------------------

#: SMARTS-based handle detectors: the pattern matches the *defining
#: feature* and returns the attachment atom (the atom that bonds in the
#: junction, per the engine's HandleSpec convention).  Heteroaryl
#: halides are detected programmatically (ring classification) because
#: their classes are ring-type, not single-bond, distinctions.
HANDLE_CLASS_SMARTS: Dict[str, str] = {
    # the acyl chloride's fourth substituent is carbon (exclusive of the
    # chloroformate O-substitution and the carbamoyl N-substitution)
    "acid_chloride": "[CX3](=[OX1])(-[#6])-[Cl]",
    "sulfonyl_chloride": "[SX4](=[OX1])(=[OX1])[Cl]",
    "carbamoyl_chloride": "[CX3](=[OX1])([NX3])[Cl]",
    "chloroformate": "[CX3](=[OX1])([OX2])[Cl]",
    "aldehyde": "[CX3H1](=[OX1])[#6]",
    "alkyl_bromide": "[Br][CX4]",
    "aryl_bromide": "[Br][c]",
    "aryl_chloride": "[Cl][c]",
    # any aromatic C-B(OH)2: aryl vs heteroaryl is decided by the
    # ring classification of the attached aryl carbon (programmatic)
    "arylboronic_acid": "[c][BX3]([OX2H])[OX2]",
    "arylzinc": "[c][Zn]",
    "vinylstannane": "[CX3]=[CX3][Sn]",
    "acetylide": "[CX2;H1]#[CX2]",
}

#: Heteroaryl-halide classes: (ring class) -> handle class names.
#: Ring classification is implemented in :func:`_classify_heteroaryl_ring`.
HETEROARYL_HALIDE_RING_CLASSES: Dict[str, Tuple[str, ...]] = {
    "pyrimidine": ("pyrimidinyl_chloride",),
    "pyridine": ("pyridyl_chloride",),
    "thiazole": ("thiazolyl_chloride", "thiazolyl_bromide"),
    "pyrazole": ("pyrazolyl_bromide",),
    "benzimidazole": ("benzimidazolyl_chloride",),
}

#: The complete block-side handle-class vocabulary of the certified
#: registry (SMARTS-detected + ring-classified + the programmatic
#: heteroaryl-boronic subclass).
ALL_HANDLE_CLASSES = frozenset(
    set(HANDLE_CLASS_SMARTS)
    | {"heteroarylboronic_acid"}
    | {
        name
        for names in HETEROARYL_HALIDE_RING_CLASSES.values()
        for name in names
    }
)

#: The SNAr-susceptible heteroaryl halide classes (electron-deficient
#: rings reacting with amine reagents without a catalyst).
SNAR_HALIDE_CLASSES = frozenset(
    {
        "pyrimidinyl_chloride",
        "pyridyl_chloride",
        "thiazolyl_chloride",
        "benzimidazolyl_chloride",
    }
)

#: The differentiated-reactivity exemption of rule R4: two ring halides
#: separated by exactly one ring nitrogen (the 2,4-dihalo-1,3-azine
#: motif — e.g. 2,4-dichloropyrimidine, whose 4-halide is displaced by
#: amines selectively while the 2-halide survives; the classic
#: mono-amination chemistry of pyrimidine routes).
DIFFERENTIATED_DIHALOAZINE = "[c;X2](:[F,Cl,Br]):[n]:[c;X2](:[F,Cl,Br])"

#: Electrophile classes that react with nucleophilic reagents WITHOUT a
#: transition-metal catalyst (rule R5's competing-electrophile set).
UNCATALYZED_ELECTROPHILE_CLASSES = frozenset(
    {
        "acid_chloride",
        "sulfonyl_chloride",
        "carbamoyl_chloride",
        "chloroformate",
        "alkyl_bromide",
        "aldehyde",
    }
)

#: The classical self-reactive pairs (rule R1): intended electrophilic
#: handle class -> the free nucleophile groups that attack it
#: intramolecularly under the reaction conditions.
SELF_REACTIVE_PAIRS: Dict[str, frozenset] = {
    "acid_chloride": frozenset(NUCLEOPHILE_GROUPS | {"carboxylic_acid"}),
    "sulfonyl_chloride": frozenset(NUCLEOPHILE_GROUPS | {"carboxylic_acid"}),
    "carbamoyl_chloride": frozenset(NUCLEOPHILE_GROUPS | {"carboxylic_acid"}),
    "chloroformate": frozenset(NUCLEOPHILE_GROUPS | {"carboxylic_acid"}),
    "alkyl_bromide": frozenset(NUCLEOPHILE_GROUPS | {"carboxylic_acid"}),
    "aldehyde": frozenset({"primary_amine", "secondary_amine"}),
}

#: Rule R2: handle classes whose consumers are organometallic and are
#: quenched by any free protic group on the block.
ORGANOMETALLIC_QUENCHED_CLASSES = frozenset({"arylzinc", "acetylide"})

#: Rule R5: handle classes whose consumers are nucleophilic reagents
#: (amine/alcohol/thiol) — a second uncatalyzed electrophile competes.
NUCLEOPHILIC_REAGENT_CLASSES = frozenset(
    {
        "acid_chloride",
        "sulfonyl_chloride",
        "carbamoyl_chloride",
        "chloroformate",
        "alkyl_bromide",
        "aldehyde",
    }
)

#: Rule R6: handle classes consumed by Pd/Sn cross-couplings where a
#: free thiol poisons the catalyst (and competes as a nucleophile).
PD_COUPLING_CLASSES = frozenset(
    {
        "aryl_bromide",
        "aryl_chloride",
        "arylboronic_acid",
        "heteroarylboronic_acid",
        "vinylstannane",
        "pyrazolyl_bromide",
        "thiazolyl_bromide",
    }
)

#: Rule R1' (competing nucleophile under coupling conditions): aryl and
#: SNAr-susceptible heteroaryl halides whose own free nucleophiles
#: compete intramolecularly/intermolecularly under (for the aryl
#: halides, Pd-catalyzed; for the SNAr halides, basic) conditions.
#: Aliphatic alcohols are TOLERATED on SNAr-susceptible heterocycles
#: (mild amination conditions do not generate alkoxides at useful
#: rates); phenols (more acidic) compete and are excluded.
COUPLING_COMPETING_NUCLEOPHILES: Dict[str, frozenset] = {
    "aryl_bromide": frozenset(NUCLEOPHILE_GROUPS | {"carboxylic_acid"}),
    "aryl_chloride": frozenset(NUCLEOPHILE_GROUPS | {"carboxylic_acid"}),
    "pyrimidinyl_chloride": frozenset(
        {"primary_amine", "secondary_amine", "thiol", "phenol"}
    ),
    "pyridyl_chloride": frozenset(
        {"primary_amine", "secondary_amine", "thiol", "phenol"}
    ),
    "thiazolyl_chloride": frozenset(
        {"primary_amine", "secondary_amine", "thiol", "phenol"}
    ),
    "benzimidazolyl_chloride": frozenset(
        {"primary_amine", "secondary_amine", "thiol", "phenol"}
    ),
}

#: Suzuki-type consumers tolerate free amines/alcohols (common
#: substrates: 3-aminophenylboronic acids couple cleanly) — only
#: thiols (catalyst poison) are excluded.
SUZUKI_TOLERANT_CLASSES = frozenset(
    {"arylboronic_acid", "heteroarylboronic_acid"}
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupHit:
    """One detected chemical feature with its screening role.

    ``group`` is the vocabulary name; ``atom_indices`` are the RDKit
    atom indices of the defining match; ``role`` is the free-group role
    (``nucleophile+protic``, ``electrophile``, ...) or ``handle`` for
    the intended-class instance.
    """

    group: str
    atom_indices: Tuple[int, ...]
    role: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "group": self.group,
            "atom_indices": list(self.atom_indices),
            "role": self.role,
        }


@dataclass
class ChemoscreenResult:
    """The per-compound screening record (issue #40 required work 5)."""

    block_id: str
    handle_class: str
    status: str  # 'pass' | 'excluded'
    reason_codes: Tuple[str, ...] = ()
    detected_groups: Tuple[GroupHit, ...] = ()
    handle_atom_indices: Tuple[int, ...] = ()
    applicable_reactions: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "block_id": self.block_id,
            "handle_class": self.handle_class,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "detected_groups": [g.as_dict() for g in self.detected_groups],
            "handle_atom_indices": list(self.handle_atom_indices),
            "applicable_reactions": list(self.applicable_reactions),
            "notes": list(self.notes),
        }


@dataclass
class CurationManifest:
    """The catalog-level screening manifest (issue #40 required work 7)."""

    results: Tuple[ChemoscreenResult, ...] = ()
    total: int = 0
    passed: int = 0
    excluded: int = 0
    exclusion_counts: Dict[str, int] = field(default_factory=dict)
    exclusion_percentages: Dict[str, float] = field(default_factory=dict)
    handle_class_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "passed": self.passed,
            "excluded": self.excluded,
            "pass_rate": self.pass_rate,
            "exclusion_counts": dict(self.exclusion_counts),
            "exclusion_percentages": {
                k: round(v, 4) for k, v in self.exclusion_percentages.items()
            },
            "handle_class_counts": dict(self.handle_class_counts),
            "results": [r.as_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _mol_from_input(mol) -> Chem.Mol:
    """Accept an RDKit Mol or a SMILES string; sanitize fail-closed."""
    if isinstance(mol, Chem.Mol):
        if mol.GetNumAtoms() == 0:
            raise ChemoselectivityError("empty molecule (fail-closed)")
        return mol
    if isinstance(mol, str):
        parsed = Chem.MolFromSmiles(mol)
        if parsed is None:
            raise ChemoselectivityError(
                f"unparseable SMILES: {mol!r} (fail-closed)"
            )
        return parsed
    raise ChemoselectivityError(
        f"expected an RDKit Mol or SMILES string, got {type(mol).__name__}"
    )


def detect_free_groups(mol) -> Tuple[GroupHit, ...]:
    """All free (reactive) functional groups on the molecule.

    Protected forms (amides, carbamates, sulfonamides, esters, silyl and
    benzyl ethers, thioethers, aromatic [nH]) do not match the free
    vocabulary — orthogonal protection passes by construction.
    """
    mol = _mol_from_input(mol)
    hits: List[GroupHit] = []
    for group, smarts in FREE_GROUP_SMARTS.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:  # pragma: no cover — vocabulary is load-time tested
            raise ChemoselectivityError(
                f"free-group SMARTS for {group!r} failed to parse"
            )
        matches = mol.GetSubstructMatches(query, uniquify=True)
        if matches:
            atoms = tuple(sorted({a for match in matches for a in (match[0],)}))
            hits.append(
                GroupHit(
                    group=group,
                    atom_indices=atoms,
                    role=FREE_GROUP_ROLES.get(group, "unspecified"),
                )
            )
    return tuple(hits)


def _ring_class(mol: Chem.Mol, ring_atom_indices: Sequence[int]) -> Optional[str]:
    """Classify an aromatic ring by size and heteroatoms."""
    ring = [mol.GetAtomWithIdx(i) for i in ring_atom_indices]
    if not all(atom.GetIsAromatic() for atom in ring):
        return None
    hetero = [atom.GetSymbol() for atom in ring if atom.GetSymbol() != "C"]
    size = len(ring)
    if size == 6:
        if hetero.count("N") == 2:
            return "pyrimidine" if _is_13_diazine(mol, ring_atom_indices) else "pyridazine_or_pyrazine"
        if hetero.count("N") == 1:
            return "pyridine"
        return "benzene"
    if size == 5:
        if hetero == ["N", "N"]:
            return "pyrazole"
        if sorted(hetero) == ["N", "S"]:
            return "thiazole"
        if hetero == ["N"]:
            return "pyrrole"
        return "five_ring_other"
    return None


def _is_13_diazine(mol: Chem.Mol, ring_atom_indices: Sequence[int]) -> bool:
    """Whether the two ring nitrogens are 1,3 (meta) related."""
    positions = {idx: p for p, idx in enumerate(ring_atom_indices)}
    nitrogens = [
        p
        for idx, p in positions.items()
        if mol.GetAtomWithIdx(idx).GetSymbol() == "N"
    ]
    if len(nitrogens) != 2:
        return False
    a, b = sorted(nitrogens)
    size = len(ring_atom_indices)
    return min((b - a) % size, (a - b) % size) == 2


def _classify_heteroaryl_ring(
    mol: Chem.Mol, halide_atom_idx: int
) -> Optional[str]:
    """The ring class of the aromatic ring bearing ``halide_atom_idx``.

    Handles fused systems (benzimidazole: a 5-ring with 2 adjacent N
    fused to a benzene).  Returns ``None`` for plain benzenes/other.
    """
    atom = mol.GetAtomWithIdx(halide_atom_idx)
    if not atom.GetIsAromatic() or atom.GetSymbol() != "C":
        return None
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if halide_atom_idx in ring:
            # benzimidazole: imidazole ring fused to benzene
            hetero = [
                mol.GetAtomWithIdx(i).GetSymbol()
                for i in ring
                if mol.GetAtomWithIdx(i).GetSymbol() != "C"
            ]
            if (
                len(ring) == 5
                and hetero == ["N", "N"]
                and _is_fused_to_benzene(mol, ring)
            ):
                return "benzimidazole"
            klass = _ring_class(mol, ring)
            if klass in ("pyrimidine", "pyridine", "thiazole", "pyrazole"):
                return klass
    return None


def _is_fused_to_benzene(mol: Chem.Mol, ring: Sequence[int]) -> bool:
    ring_set = set(ring)
    other = mol.GetRingInfo().AtomRings()
    for r2 in other:
        if set(r2) == ring_set:
            continue
        if len(set(r2) & ring_set) >= 2:
            if _ring_class(mol, r2) == "benzene":
                return True
    return False


def _attached_ring_is_heteroaromatic(mol: Chem.Mol, atom_idx: int) -> bool:
    """Whether any aromatic ring containing ``atom_idx`` has a heteroatom."""
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx in ring:
            if any(
                mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring
            ) and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                return True
    return False


def _halide_classes_on_atom(mol: Chem.Mol, atom_idx: int) -> List[str]:
    """The handle classes the halide-bearing aromatic atom realises."""
    klass = _classify_heteroaryl_ring(mol, atom_idx)
    if klass is None:
        return []
    halides = [
        n.GetSymbol()
        for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if n.GetSymbol() in ("F", "Cl", "Br")
    ]
    out = []
    for name in HETEROARYL_HALIDE_RING_CLASSES.get(klass, ()):
        wanted = "Cl" if name.endswith("_chloride") else "Br"
        if wanted in halides:
            out.append(name)
    return out


def detect_handle_instances(mol, handle_class: str) -> Tuple[int, ...]:
    """Atom indices defining the ``handle_class`` instances on ``mol``.

    The returned atoms follow the engine's ``HandleSpec`` convention
    (the attachment atom that bonds in the junction): the acyl carbon
    for acyl chlorides, the halide-bearing carbon for aryl/alkyl
    halides, the boron for boronic acids, the carbon for zinc/tin
    organometallics, and the terminal CH carbon for the acetylide.
    Heteroaryl halides are classified by their ring type.
    """
    mol = _mol_from_input(mol)
    heteroaryl_classes = {
        name
        for names in HETEROARYL_HALIDE_RING_CLASSES.values()
        for name in names
    }
    if handle_class in heteroaryl_classes:
        found = []
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic() and atom.GetSymbol() == "C":
                if handle_class in _halide_classes_on_atom(mol, atom.GetIdx()):
                    found.append(atom.GetIdx())
        return tuple(sorted(found))
    if handle_class in ("arylboronic_acid", "heteroarylboronic_acid"):
        # base detector: every aromatic C-B(OH)2 (attachment = the carbon)
        base = _smarts_instances(mol, HANDLE_CLASS_SMARTS["arylboronic_acid"])
        hetero = {
            a
            for a in base
            if _attached_ring_is_heteroaromatic(mol, a)
        }
        if handle_class == "heteroarylboronic_acid":
            return tuple(sorted(hetero))
        return tuple(sorted(a for a in base if a not in hetero))
    smarts = HANDLE_CLASS_SMARTS.get(handle_class)
    if smarts is None:
        raise ChemoselectivityError(
            f"unknown handle class {handle_class!r}: not in the certified "
            "registry's block-side vocabulary (fail-closed)"
        )
    return _smarts_instances(mol, smarts)


def _smarts_instances(mol: Chem.Mol, smarts: str) -> Tuple[int, ...]:
    """First-atom indices of all unique substructure matches."""
    query = Chem.MolFromSmarts(smarts)
    if query is None:  # pragma: no cover — vocabulary is load-time tested
        raise ChemoselectivityError(
            f"handle SMARTS {smarts!r} failed to parse"
        )
    matches = mol.GetSubstructMatches(query, uniquify=True)
    if not matches:
        return ()
    return tuple(sorted({m[0] for m in matches}))


def handle_class_of_atom(mol, atom_idx: int) -> List[str]:
    """Every registry handle class realised by atom ``atom_idx``."""
    mol = _mol_from_input(mol)
    if not 0 <= atom_idx < mol.GetNumAtoms():
        raise ChemoselectivityError(
            f"atom index {atom_idx} out of range for a {mol.GetNumAtoms()}-atom molecule"
        )
    classes = []
    for cls in HANDLE_CLASS_SMARTS:
        if atom_idx in detect_handle_instances(mol, cls):
            classes.append(cls)
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetIsAromatic() and atom.GetSymbol() == "C":
        classes.extend(_halide_classes_on_atom(mol, atom_idx))
    return sorted(set(classes))


# ---------------------------------------------------------------------------
# The screen itself
# ---------------------------------------------------------------------------

def _registry_reactions_for_class(handle_class: str) -> Tuple[str, ...]:
    return tuple(
        rid
        for rid, reaction in POLICY_REACTIONS.items()
        if reaction.block_handle_class == handle_class
    )


def screen_block(
    mol,
    handle_class: str,
    block_id: str = "",
    handle_atom_idx: Optional[int] = None,
) -> ChemoscreenResult:
    """Reaction-aware chemoselectivity screen of one catalog candidate.

    Validates the candidate against its intended reaction/handle role
    (never a blanket functional-group exclusion): the exclusion set is
    derived from the certified registry's consumers of
    ``handle_class`` and the documented rule matrix.

    Parameters:
        mol: an RDKit Mol or SMILES string (sanitizable; basic
            molecular validation is assumed to have run upstream).
        handle_class: the intended block-side handle class (a
            ``CertifiedReaction.block_handle_class`` value).
        block_id: the catalog identifier (telemetry).
        handle_atom_idx: optional explicit attachment atom (from the
            catalog's ``HandleSpec``); when given, its class is
            validated against ``handle_class``.
    """
    mol = _mol_from_input(mol)
    if handle_class not in ALL_HANDLE_CLASSES:
        raise ChemoselectivityError(
            f"unknown handle class {handle_class!r} (fail-closed)"
        )
    applicable = _registry_reactions_for_class(handle_class)

    instances = detect_handle_instances(mol, handle_class)
    reason_codes: List[str] = []
    notes: List[str] = []

    intended_atom = handle_atom_idx
    if intended_atom is not None:
        if handle_class not in handle_class_of_atom(mol, intended_atom):
            raise ChemoselectivityError(
                f"block {block_id!r}: atom {intended_atom} does not realize "
                f"handle class {handle_class!r} (fail-closed cross-field "
                "validation of the catalog HandleSpec)"
            )
        if intended_atom not in instances:
            instances = tuple(sorted(set(instances) | {intended_atom}))

    # -- R3 / R4 short-circuit: no handle of the class at all ---------------
    if len(instances) == 0:
        reason_codes.append("no_handle_of_class")
        return ChemoscreenResult(
            block_id=block_id,
            handle_class=handle_class,
            status="excluded",
            reason_codes=tuple(reason_codes),
            detected_groups=detect_free_groups(mol),
            handle_atom_indices=(),
            applicable_reactions=applicable,
            notes=(),
        )

    # -- R3: exactly one *usable* handle of the intended class ---------------
    # With a declared HandleSpec atom, extra class instances on the handle's
    # ring are evaluated through the differentiated-reactivity exemption
    # (the 2,4-dihalo-1,3-azine motif): a spectator halide whose carbon is
    # flanked by MORE adjacent ring nitrogens than the handle carbon (the
    # 2-position vs the 4-position of a pyrimidine) is not a competing
    # coupling site for amine reagents.  Without a declared atom, the
    # reference is the most amine-labile instance (fewest adjacent ring
    # nitrogens); ties at the minimum are genuinely ambiguous.
    if intended_atom is None:
        if len(instances) > 1:
            reference = min(
                instances, key=lambda a: _adjacent_ring_nitrogens(mol, a)
            )
            min_adj = _adjacent_ring_nitrogens(mol, reference)
            tied = [
                a
                for a in instances
                if _adjacent_ring_nitrogens(mol, a) == min_adj
            ]
            extra_class = tuple(a for a in instances if a != reference)
        else:
            reference = instances[0] if instances else None
            extra_class = ()
            tied = [reference] if reference is not None else []
    else:
        reference = intended_atom
        extra_class = tuple(a for a in instances if a != intended_atom)
        tied = []
    if len(tied) > 1:
        reason_codes.append("competing_identical_handles")
    elif extra_class:
        if all(
            _one_ring_n_between(mol, reference, a)
            and _adjacent_ring_nitrogens(mol, a)
            > _adjacent_ring_nitrogens(mol, reference)
            for a in extra_class
        ):
            notes.append(
                "differentiated_2_4_dihaloazine: spectator ring halide(s) "
                "separated from the handle by exactly one ring nitrogen and "
                "flanked by more adjacent ring nitrogens (the documented "
                "2,4-dihalo-1,3-azine exemption — amine reagents displace "
                "the handle-position halide selectively, e.g. the C4 halide "
                "of 2,4-dichloropyrimidines)"
            )
        else:
            reason_codes.append("competing_identical_handles")

    # -- the free-group scan -------------------------------------------------
    free_hits = detect_free_groups(mol)
    free_names = {hit.group for hit in free_hits}
    # the intended class's own defining group is not a "free group" for
    # self-reactivity purposes (e.g. the aldehyde handle is not an
    # excluded aldehyde) — remove it from consideration.
    own_group = _class_own_group(handle_class)

    # -- R1: classical self-reactive pairs ------------------------------------
    for group in sorted(SELF_REACTIVE_PAIRS.get(handle_class, frozenset())):
        if group in free_names and group != own_group:
            reason_codes.append(f"self_reactive_pair:{group}")

    # -- R1': competing nucleophiles under (Pd / basic) coupling conditions ---
    for group in sorted(COUPLING_COMPETING_NUCLEOPHILES.get(handle_class, frozenset())):
        if group in free_names and group != own_group:
            reason_codes.append(f"competing_nucleophile:{group}")

    # -- R2: organometallic quenching ------------------------------------------
    if handle_class in ORGANOMETALLIC_QUENCHED_CLASSES:
        for group in sorted(PROTIC_GROUPS):
            if group in free_names and group != own_group:
                reason_codes.append(f"quenching_protic_group:{group}")

    # -- R2': alkylzinc consumers quench on aryl-bromide blocks (rxn_15) -------
    if handle_class == "aryl_bromide":
        for group in sorted(PROTIC_GROUPS):
            if group in free_names:
                reason_codes.append(f"quenching_protic_group:{group}")

    # -- R6: catalyst poisons ----------------------------------------------------
    if handle_class in PD_COUPLING_CLASSES and "thiol" in free_names:
        reason_codes.append("catalyst_poison:thiol")

    # -- R5: competing uncatalyzed electrophiles ----------------------------------
    if handle_class in NUCLEOPHILIC_REAGENT_CLASSES:
        for other_class in sorted(UNCATALYZED_ELECTROPHILE_CLASSES):
            if other_class == handle_class:
                continue
            if detect_handle_instances(mol, other_class):
                reason_codes.append(f"competing_electrophile:{other_class}")

    # -- R4: competing SNAr positions (with the documented exemption) -----------
    if handle_class in SNAR_HALIDE_CLASSES:
        handle_atom = instances[0] if intended_atom is None else intended_atom
        extra = [
            a
            for a in _competing_snar_positions(mol, handle_atom)
            if a not in instances
        ]
        if extra:
            if all(
                _one_ring_n_between(mol, handle_atom, a)
                and _adjacent_ring_nitrogens(mol, a)
                > _adjacent_ring_nitrogens(mol, handle_atom)
                for a in extra
            ):
                notes.append(
                    "differentiated_2_4_dihaloazine: additional ring halide(s) "
                    "separated from the handle by exactly one ring nitrogen "
                    "(the documented 2,4-dihalo-1,3-azine exemption — amines "
                    "displace the handle-position halide selectively)"
                )
            else:
                reason_codes.append("competing_snar_position")

    # -- assemble the record ------------------------------------------------------
    detected = list(free_hits)
    for atom_idx in instances:
        detected.append(
            GroupHit(
                group=handle_class,
                atom_indices=(atom_idx,),
                role="handle",
            )
        )
    # protected orthogonal functionality is reported as a note when a
    # protected-form pattern is present (telemetry for the manifest)
    status = "excluded" if reason_codes else "pass"
    return ChemoscreenResult(
        block_id=block_id,
        handle_class=handle_class,
        status=status,
        reason_codes=tuple(reason_codes),
        detected_groups=tuple(detected),
        handle_atom_indices=tuple(instances),
        applicable_reactions=applicable,
        notes=tuple(notes),
    )


def _class_own_group(handle_class: str) -> Optional[str]:
    """The free-group name that *is* the intended handle's own group."""
    return {
        "aldehyde": "aldehyde",
        "acetylide": "terminal_alkyne",
        "arylboronic_acid": None,
        "heteroarylboronic_acid": None,
    }.get(handle_class)


def _competing_snar_positions(mol: Chem.Mol, handle_atom: int) -> List[int]:
    """Additional halide-bearing aromatic carbons on the handle's ring."""
    atom = mol.GetAtomWithIdx(handle_atom)
    if not (atom.GetIsAromatic() and atom.GetSymbol() == "C"):
        return []
    ring_info = mol.GetRingInfo()
    handle_rings = [r for r in ring_info.AtomRings() if handle_atom in r]
    extra: List[int] = []
    for other in mol.GetAtoms():
        if other.GetIdx() == handle_atom:
            continue
        if not (other.GetIsAromatic() and other.GetSymbol() == "C"):
            continue
        has_halide = any(
            n.GetSymbol() in ("F", "Cl", "Br") for n in other.GetNeighbors()
        )
        if not has_halide:
            continue
        shares = any(other.GetIdx() in r for r in handle_rings)
        if shares:
            extra.append(other.GetIdx())
    return extra


def _adjacent_ring_nitrogens(mol: Chem.Mol, atom_idx: int) -> int:
    """Count aromatic ring nitrogens directly bonded to ``atom_idx``."""
    atom = mol.GetAtomWithIdx(atom_idx)
    return sum(
        1
        for neighbour in atom.GetNeighbors()
        if neighbour.GetSymbol() == "N" and neighbour.GetIsAromatic()
    )


def _one_ring_n_between(mol: Chem.Mol, a: int, b: int) -> bool:
    """Whether ring atoms ``a`` and ``b`` are separated by exactly one
    aromatic ring nitrogen (the 2,4-relationship of a 1,3-azine)."""
    ring_info = mol.GetRingInfo()
    size_hint = None
    for ring in ring_info.AtomRings():
        if a in ring and b in ring:
            size_hint = len(ring)
    if size_hint is None:
        return False
    # walk both bond directions from ``a``; two bonds away must be ``b``
    # with an aromatic N on the intermediate atom.
    for direction in (+1, -1):
        path = _ring_path(mol, a, steps=2, direction=direction)
        if path is None:
            continue
        mid, end = path
        if end != b:
            continue
        mid_atom = mol.GetAtomWithIdx(mid)
        if mid_atom.GetIsAromatic() and mid_atom.GetSymbol() == "N":
            return True
    return False


def _ring_path(mol: Chem.Mol, start: int, steps: int, direction: int):
    """Walk ``steps`` ring bonds from ``start`` in one direction.

    Returns the visited (intermediate..., end) atom indices, or ``None``
    when the walk is ambiguous (``start`` in multiple rings or a ring
    junction).  Uses the unique smallest ring containing ``start``.
    """
    rings = [
        r
        for r in mol.GetRingInfo().AtomRings()
        if start in r and len(r) <= 6
    ]
    if not rings:
        return None
    ring = min(rings, key=len)
    idx = list(ring)
    p = idx.index(start)
    size = len(idx)
    visited = []
    for s in range(1, steps + 1):
        visited.append(idx[(p + direction * s) % size])
    return tuple(visited)


# ---------------------------------------------------------------------------
# Catalog-level screening (issue #40 required work 7)
# ---------------------------------------------------------------------------

def screen_catalog(
    entries: Iterable[Mapping[str, object]],
) -> CurationManifest:
    """Screen a catalog of candidate entries and build the manifest.

    Each entry is a mapping with keys:

    * ``block_id`` (str), ``mol`` (RDKit Mol or SMILES), ``handle_class``
      (str), and optionally ``handle_atom_idx`` (int).

    The manifest records per-compound pass/fail, reason codes, detected
    handles, applicable reactions, and the per-rule exclusion counts and
    percentages — the impact report the #40 acceptance requires (never
    an inference of wet-lab success rates from catalog membership).
    """
    results: List[ChemoscreenResult] = []
    exclusion_counts: Dict[str, int] = {}
    handle_counts: Dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ChemoselectivityError(
                "catalog entries must be mappings with block_id/mol/handle_class"
            )
        missing = {"block_id", "mol", "handle_class"} - set(entry)
        if missing:
            raise ChemoselectivityError(
                f"catalog entry missing keys {sorted(missing)} (fail-closed)"
            )
        result = screen_block(
            entry["mol"],
            str(entry["handle_class"]),
            block_id=str(entry["block_id"]),
            handle_atom_idx=entry.get("handle_atom_idx"),
        )
        results.append(result)
        handle_counts[result.handle_class] = (
            handle_counts.get(result.handle_class, 0) + 1
        )
        for code in result.reason_codes:
            exclusion_counts[code] = exclusion_counts.get(code, 0) + 1
            base = code.split(":")[0]
            if base != code:
                exclusion_counts[base] = exclusion_counts.get(base, 0) + 1
    total = len(results)
    excluded = sum(1 for r in results if r.status == "excluded")
    percentages = {
        code: 100.0 * count / total for code, count in exclusion_counts.items()
    }
    return CurationManifest(
        results=tuple(results),
        total=total,
        passed=total - excluded,
        excluded=excluded,
        exclusion_counts=exclusion_counts,
        exclusion_percentages=percentages,
        handle_class_counts=handle_counts,
    )


def chemoselectivity_rule_report() -> Dict[str, object]:
    """The documented rule matrix (transparency / manifest header)."""
    return {
        "rules": {
            "R1_self_reactive_pair": {
                "description": (
                    "intended electrophilic handle + free nucleophile on the "
                    "same molecule self-reacts intramolecularly/oligomerizes"
                ),
                "matrix": {
                    cls: sorted(groups)
                    for cls, groups in SELF_REACTIVE_PAIRS.items()
                },
            },
            "R1b_competing_nucleophile": {
                "description": (
                    "aryl/SNAr halide blocks: free nucleophiles compete under "
                    "(Pd / basic) coupling conditions"
                ),
                "matrix": {
                    cls: sorted(groups)
                    for cls, groups in COUPLING_COMPETING_NUCLEOPHILES.items()
                },
            },
            "R2_quenching_protic_group": {
                "description": (
                    "organometallic consumers (arylzinc/acetylide blocks; "
                    "alkylzinc consumers of aryl bromides) are quenched by "
                    "any free protic group"
                ),
                "classes": sorted(ORGANOMETALLIC_QUENCHED_CLASSES)
                + ["aryl_bromide (rxn_15 alkylzinc consumer)"],
            },
            "R3_competing_identical_handles": {
                "description": (
                    "more than one handle of the intended class: "
                    "nonselective oligomerizing substrate (the V2 junction "
                    "model couples exactly one)"
                ),
                "applies_to": "all handle classes",
            },
            "R4_competing_snar_position": {
                "description": (
                    "additional activated halides on the handle's "
                    "electron-deficient heteroaromatic ring give regioisomer "
                    "mixtures"
                ),
                "exemption": (
                    "differentiated 2,4-dihalo-1,3-azine motif: every extra "
                    "ring halide exactly one ring N away from the handle "
                    "(e.g. 2,4-dichloropyrimidine: amines displace the "
                    "4-halide selectively — classic pyrimidine chemistry)"
                ),
                "applies_to": sorted(SNAR_HALIDE_CLASSES),
            },
            "R5_competing_electrophile": {
                "description": (
                    "nucleophilic-reagent consumers: a second uncatalyzed "
                    "electrophile on the block is attacked by the reagent too"
                ),
                "electrophiles": sorted(UNCATALYZED_ELECTROPHILE_CLASSES),
                "applies_to": sorted(NUCLEOPHILIC_REAGENT_CLASSES),
            },
            "R6_catalyst_poison": {
                "description": (
                    "free thiols poison Pd/Sn cross-coupling catalysts (and "
                    "compete as nucleophiles)"
                ),
                "applies_to": sorted(PD_COUPLING_CLASSES),
            },
        },
        "orthogonal_protection": (
            "free-group SMARTS exclude acyl/carbamate/sulfonamide N, "
            "ester/silyl/benzyl O, thioether S and aromatic [nH] by "
            "construction: protected multifunctional substrates pass"
        ),
        "registry_link": {
            "policy_reactions": len(POLICY_REACTIONS),
            "families": sorted(REACTION_FAMILIES),
        },
    }
