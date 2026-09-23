"""SMARTS reaction templates, reactive-handle detection, and deterministic
forward-reaction execution with atom-level provenance tracking.

The engine implements 8 robust medicinal-chemistry reaction classes. Every
``apply_reaction`` call tags reactant atoms with unique **isotope** labels
(core scaffold: ``1000 + i``; incoming synthon: ``2000 + j``) which RDKit
propagates onto product atoms unchanged. Isotopes are used instead of
atom-map numbers because RDKit rewrites the maps of template-matched atoms
during ``RunReactants``, destroying that provenance. The isotope tags give
exact atom correspondences even when the reaction deletes leaving groups
(e.g. the water lost in amide coupling), enabling faithful 3D coordinate
inheritance downstream; tags are cleared from the returned product.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Reaction grammar: SMARTS templates for the 8 certified reaction classes.
#
# NOTE ON SYNTAX: atom-map numbers are placed at the END of each bracket
# expression (``[N;!H0;!H3:2]``) because RDKit >= 2026 rejects map numbers
# followed by H-count predicates (``[N:2;H2,H1]``). H counts are expressed
# as negations (``!H0``/``!H3``) for the same reason.
# --------------------------------------------------------------------------
REACTION_TEMPLATES: Dict[str, str] = {
    "amide_coupling": "[C:1](=O)[OH].[N;!H0;!H3;!$(NC=O):2]>>[C:1](=O)[N:2]",
    "reductive_amination": "[C;H1:1]=O.[N;!H0;!H3;!$(NC=O):2]>>[C:1][N:2]",
    "suzuki_coupling": "[c:1][Br,I,Cl].[c:2][B]([OH,O])[OH,O]>>[c:1][c:2]",
    "snar": "[c:1][F,Cl].[N;!H0;!H3;!$(NC=O):2]>>[c:1][N:2]",
    "urea_formation": "[N;!H0;!H3:1].[N;!H0;!H3:2]>>[N:1]C(=O)[N:2]",
    "buchwald_hartwig": "[c:1][Br,I,Cl].[N;!H0;!H3:2]>>[c:1][N:2]",
    "esterification": "[C:1](=O)[OH].[O;H1;!$(O[B]):2]>>[C:1](=O)[O:2]",
    "click_triazole": "[C:1]C#C.[N:2]=[N+]=[N-]>>[C:1]c1cn([N:2])nn1",
}

# SMARTS identifying reactive attachment points (handles) on a molecule.
# H counts use negated predicates (!H0/!H3) for parser compatibility.
HANDLE_SMARTS: Dict[str, str] = {
    "carboxylic_acid": "[C](=O)[OH]",
    "primary_secondary_amine": "[N;!H0;!H3;!$(NC=O)]",
    "aryl_halide": "[c][Br,I,Cl]",
    "boronic_acid": "[c][B]([OH,O])[OH,O]",
    "aldehyde": "[CH]=O",
    "alcohol": "[O;!H0;!$(O[B]);!$(OC=O)]",
    "alkyne": "[C]#[CH]",
    "azide": "[N]=[N+]=[N-]",
}

# The two complementary reactant slots for every reaction. ``side_a`` is the
# molecule matched by the first reactant pattern of the SMARTS template.
REACTION_SIDES: Dict[str, Tuple[str, str]] = {
    "amide_coupling": ("carboxylic_acid", "primary_secondary_amine"),
    "reductive_amination": ("aldehyde", "primary_secondary_amine"),
    "suzuki_coupling": ("aryl_halide", "boronic_acid"),
    "snar": ("aryl_halide", "primary_secondary_amine"),
    "urea_formation": ("primary_secondary_amine", "primary_secondary_amine"),
    "buchwald_hartwig": ("aryl_halide", "primary_secondary_amine"),
    "esterification": ("carboxylic_acid", "alcohol"),
    "click_triazole": ("alkyne", "azide"),
}

# Handle -> reaction -> set of partner handle types that a synthon must
# expose in order to be a legal partner in that reaction.
REACTION_PARTNER_HANDLES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    handle: {} for handle in HANDLE_SMARTS
}
for _rxn, (_side_a, _side_b) in REACTION_SIDES.items():
    _a_partners = REACTION_PARTNER_HANDLES[_side_a].get(_rxn, ())
    REACTION_PARTNER_HANDLES[_side_a][_rxn] = tuple(sorted(set(_a_partners + (_side_b,))))
    _b_partners = REACTION_PARTNER_HANDLES[_side_b].get(_rxn, ())
    REACTION_PARTNER_HANDLES[_side_b][_rxn] = tuple(sorted(set(_b_partners + (_side_a,))))

# Compatibility mapping: handle type -> reactions a molecule bearing that
# handle can participate in (derived from REACTION_SIDES, single source of
# truth).
HANDLE_TO_REACTIONS: Dict[str, List[str]] = {
    handle: sorted(partners.keys()) for handle, partners in REACTION_PARTNER_HANDLES.items()
}

# Reaction-family classes used by the policy. SNAr and Buchwald-Hartwig are
# deliberately one family because the final product graph does not contain
# enough information to distinguish which experimental conditions were used.
REACTION_FAMILY_MEMBERS: Dict[str, Tuple[str, ...]] = {
    "amide_coupling": ("amide_coupling",),
    "reductive_amination": ("reductive_amination",),
    "suzuki_coupling": ("suzuki_coupling",),
    "aryl_amination": ("snar", "buchwald_hartwig"),
    "urea_formation": ("urea_formation",),
    "esterification": ("esterification",),
    "click_triazole": ("click_triazole",),
}
REACTION_FAMILY_NAMES: Tuple[str, ...] = tuple(REACTION_FAMILY_MEMBERS.keys())
REACTION_FAMILY_FOR: Dict[str, str] = {
    reaction: family
    for family, reactions in REACTION_FAMILY_MEMBERS.items()
    for reaction in reactions
}
HANDLE_NAMES: Tuple[str, ...] = tuple(HANDLE_SMARTS.keys())

# Map numbers used for atom provenance tracking (chosen far above anything a
# template would use).
_CORE_MAP_OFFSET = 1000
_SYNTHON_MAP_OFFSET = 2000


# Nucleophilicity / chemoselectivity tiers used to rank competing handles on
# a growing ligand (medicinal-chemistry practice: an aliphatic amine is ~10^6
# times more reactive in amide couplings than an aniline, so a coupling
# partner must never be attached through the weaker nucleophile while a
# stronger one sits idle).
HANDLE_REACTIVITY_TIERS: Dict[str, int] = {
    "primary_secondary_amine": 1,  # aliphatic amines (pKa ~10.5)
    "carboxylic_acid": 3,          # strong electrophile handle
    "aldehyde": 3,                 # strong electrophile handle
    "alcohol": 4,                  # weak nucleophile (esterification only)
    "alkyne": 4,
    "azide": 4,
    "aryl_halide": 5,              # metal-catalysed cross-coupling only
    "boronic_acid": 5,             # metal-catalysed cross-coupling only
}

# Aromatic (aniline-type) amines are demoted one tier below their aliphatic
# analogues: the lone pair delocalises into the ring (pKa ~4.5).
_AROMATIC_AMINE_DEMOTION = 1


@dataclass(frozen=True)
class HandleInfo:
    """A single detected reactive attachment point."""

    handle_type: str
    atom_indices: Tuple[int, ...]
    allowed_reactions: Tuple[str, ...]

    @property
    def primary_atom(self) -> int:
        """Index of the atom that participates in the new bond."""
        return self.atom_indices[0]

    @property
    def reactivity_tier(self) -> int:
        """Rank within a molecule's competing handles (lower = more reactive).

        Aliphatic amines (tier 1) beat aromatic anilines (tier 2), which beat
        electrophilic coupling handles (tier 3), weak nucleophiles (tier 4)
        and metal-catalysed cross-coupling partners (tier 5)."""
        return HANDLE_REACTIVITY_TIERS.get(self.handle_type, 4)


@dataclass
class ReactionResult:
    """Successful outcome of a forward reaction step.

    Attributes:
        product: Sanitized product molecule (atom maps cleared).
        junction_bond: Product indices ``(core_atom, synthon_atom)`` of the
            newly formed bond connecting scaffold and synthon.
        core_atom_map: Original core atom index -> product atom index.
        synthon_atom_map: Original synthon atom index -> product atom index.
        reaction_name: Name of the executed reaction.
        new_atoms: Product atom indices created by the reaction itself
            (e.g. the inserted carbonyl of urea formation).
    """

    product: Chem.Mol
    junction_bond: Tuple[int, int]
    core_atom_map: Dict[int, int] = field(default_factory=dict)
    synthon_atom_map: Dict[int, int] = field(default_factory=dict)
    reaction_name: str = ""
    new_atoms: Tuple[int, ...] = ()


def _tag_atoms(mol: Chem.Mol, offset: int) -> Chem.Mol:
    """Return a copy of ``mol`` whose atoms carry unique provenance isotope
    labels (``offset + atom index``).

    Isotopes are used instead of atom-map numbers because RDKit rewrites
    the maps of template-matched atoms during ``RunReactants``, destroying
    the provenance, while isotope labels pass through unchanged.
    """
    tagged = Chem.Mol(mol)
    for atom in tagged.GetAtoms():
        atom.SetIsotope(offset + atom.GetIdx())
    return tagged


def _clear_tags(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of ``mol`` with provenance isotopes and atom maps
    removed."""
    cleaned = Chem.Mol(mol)
    for atom in cleaned.GetAtoms():
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
    return cleaned


class ReactionEngine:
    """Executes the 8 certified reactions with full atom provenance.

    The engine is stateless apart from the compiled SMARTS objects, and is
    therefore safe to share across threads and processes.
    """

    def __init__(self, templates: Optional[Dict[str, str]] = None):
        self.templates = dict(templates or REACTION_TEMPLATES)
        self.reactions: Dict[str, rdChemReactions.ChemicalReaction] = {}
        for name, smarts in self.templates.items():
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            if rxn is None:
                raise ValueError(f"Invalid reaction SMARTS for '{name}': {smarts}")
            try:
                rxn.Initialize()
            except Exception:  # pragma: no cover - Initialize is idempotent
                pass
            self.reactions[name] = rxn

        self.handle_patterns: Dict[str, Chem.Mol] = {}
        for name, smarts in HANDLE_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                raise ValueError(f"Invalid handle SMARTS for '{name}': {smarts}")
            self.handle_patterns[name] = pattern

    # ------------------------------------------------------------------
    # Handle detection
    # ------------------------------------------------------------------
    def detect_handles(self, mol: Chem.Mol) -> List[HandleInfo]:
        """Identify every reactive handle present on ``mol``.

        Returns a deterministic, de-duplicated list ordered by handle type
        (insertion order of :data:`HANDLE_SMARTS`) and then by atom index.
        """
        if mol is None:
            raise ValueError("detect_handles requires a valid Mol")

        seen: set = set()
        detected: List[HandleInfo] = []
        for handle_name, pattern in self.handle_patterns.items():
            matches = mol.GetSubstructMatches(pattern, uniquify=True)
            for match in sorted(matches):
                key = (handle_name, match)
                if key in seen:
                    continue
                seen.add(key)
                detected.append(
                    HandleInfo(
                        handle_type=handle_name,
                        atom_indices=tuple(int(i) for i in match),
                        allowed_reactions=tuple(HANDLE_TO_REACTIONS.get(handle_name, [])),
                    )
                )
        return detected

    def rank_handles(self, mol: Chem.Mol) -> List[HandleInfo]:
        """Order detected handles by chemical reactivity (chemoselectivity).

        Sort key: (effective tier, handle type, primary atom index). Aromatic
        amines are demoted below aliphatic amines so an acid coupling is
        always routed through the more nucleophilic nitrogen of a molecule
        bearing both an aliphatic amine and an aniline.
        """
        handles = self.detect_handles(mol)

        def sort_key(info: HandleInfo):
            tier = info.reactivity_tier
            if info.handle_type == "primary_secondary_amine":
                atom = mol.GetAtomWithIdx(info.primary_atom)
                if atom.GetIsAromatic() or any(
                    nbr.GetIsAromatic() and nbr.GetAtomicNum() == 6
                    for nbr in atom.GetNeighbors()
                ):
                    tier += _AROMATIC_AMINE_DEMOTION
            return (tier, info.handle_type, info.primary_atom)

        return sorted(handles, key=sort_key)

    def handle_types(self, mol: Chem.Mol) -> set:
        """Set of handle type names present on ``mol``."""
        return {info.handle_type for info in self.detect_handles(mol)}

    # ------------------------------------------------------------------
    # Reactant ordering helpers
    # ------------------------------------------------------------------
    def order_reactants(
        self, core_mol: Chem.Mol, synthon_mol: Chem.Mol, reaction_name: str
    ) -> Optional[Tuple[Chem.Mol, Chem.Mol]]:
        """Place ``core_mol`` / ``synthon_mol`` into the correct reactant
        slots demanded by ``reaction_name``.

        Returns ``(reactant1, reactant2)`` or ``None`` when neither ordering
        is chemically plausible (i.e. the core exposes none of the reaction's
        handles).
        """
        side_a, side_b = REACTION_SIDES[reaction_name]
        core_types = self.handle_types(core_mol)
        if side_a in core_types:
            return core_mol, synthon_mol
        if side_b in core_types:
            return synthon_mol, core_mol
        return None

    # ------------------------------------------------------------------
    # Reaction execution
    # ------------------------------------------------------------------
    def apply_reaction(
        self,
        core_mol: Chem.Mol,
        synthon_mol: Chem.Mol,
        reaction_name: str,
    ) -> Optional[ReactionResult]:
        """Execute ``reaction_name`` between ``core_mol`` and ``synthon_mol``.

        The core's 3D scaffold is *not* modified; the returned product is a
        fresh molecule whose atoms can be traced back to either reactant via
        ``core_atom_map`` / ``synthon_atom_map``.

        Args:
            core_mol: Growing ligand scaffold (any protonation state).
            synthon_mol: Incoming building block.
            reaction_name: One of :data:`REACTION_TEMPLATES`.

        Returns:
            :class:`ReactionResult` on success, ``None`` when the reaction
            produces no sanitizable product.
        """
        if core_mol is None or synthon_mol is None:
            raise ValueError("apply_reaction requires two valid Mol objects")
        if reaction_name not in self.reactions:
            raise ValueError(
                f"Unknown reaction '{reaction_name}'. "
                f"Available: {sorted(self.reactions)}"
            )
        if core_mol.GetNumAtoms() == 0 or synthon_mol.GetNumAtoms() == 0:
            return None

        ordering = self.order_reactants(core_mol, synthon_mol, reaction_name)
        if ordering is None:
            logger.debug(
                "Core exposes no handle for reaction '%s'; skipping.", reaction_name
            )
            return None
        reactant_a, reactant_b = ordering

        # Tag provenance on copies so the inputs remain untouched.
        tagged_a = _tag_atoms(reactant_a, _CORE_MAP_OFFSET if reactant_a is core_mol else _SYNTHON_MAP_OFFSET)
        tagged_b = _tag_atoms(reactant_b, _CORE_MAP_OFFSET if reactant_b is core_mol else _SYNTHON_MAP_OFFSET)

        rxn = self.reactions[reaction_name]
        try:
            product_sets = rxn.RunReactants((tagged_a, tagged_b))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Reaction execution error (%s): %s", reaction_name, exc)
            return None

        for product_set in product_sets or ():
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                except Exception:
                    continue
                result = self._build_result(
                    product, reaction_name, core_mol, synthon_mol
                )
                if result is not None:
                    return result

        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_result(
        self,
        product: Chem.Mol,
        reaction_name: str,
        core_mol: Chem.Mol,
        synthon_mol: Chem.Mol,
    ) -> Optional[ReactionResult]:
        core_atom_map: Dict[int, int] = {}
        synthon_atom_map: Dict[int, int] = {}
        new_atoms: List[int] = []

        for idx, atom in enumerate(product.GetAtoms()):
            isotope = atom.GetIsotope()
            if isotope == 0:
                new_atoms.append(idx)
            elif _CORE_MAP_OFFSET <= isotope < _SYNTHON_MAP_OFFSET:
                core_atom_map[isotope - _CORE_MAP_OFFSET] = idx
            elif isotope >= _SYNTHON_MAP_OFFSET:
                synthon_atom_map[isotope - _SYNTHON_MAP_OFFSET] = idx

        if not core_atom_map:
            logger.debug("Product retained no core atoms for '%s'.", reaction_name)
            return None

        # The junction bond connects a core-derived atom to a synthon-derived
        # atom (or, for atom-inserting reactions such as the click triazole,
        # to a freshly created atom bonded to synthon atoms).
        junction = self._find_junction(product, core_atom_map, synthon_atom_map, new_atoms)
        if junction is None:
            logger.debug("No junction bond found for '%s'.", reaction_name)
            return None

        cleaned = _clear_tags(product)
        return ReactionResult(
            product=cleaned,
            junction_bond=junction,
            core_atom_map=core_atom_map,
            synthon_atom_map=synthon_atom_map,
            reaction_name=reaction_name,
            new_atoms=tuple(new_atoms),
        )

    @staticmethod
    def _find_junction(
        product: Chem.Mol,
        core_atom_map: Dict[int, int],
        synthon_atom_map: Dict[int, int],
        new_atoms: Sequence[int],
    ) -> Optional[Tuple[int, int]]:
        core_indices = set(core_atom_map.values())
        synthon_indices = set(synthon_atom_map.values())
        new_set = set(new_atoms)

        # Preferred: direct bond between a core atom and a synthon atom.
        best_direct: Optional[Tuple[int, int]] = None
        for bond in product.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in core_indices and a2 in synthon_indices:
                best_direct = (a1, a2)
                break
            if a2 in core_indices and a1 in synthon_indices:
                best_direct = (a2, a1)
                break
        if best_direct is not None:
            return best_direct

        # Fallback (atom-inserting reactions, e.g. click triazole): the bond
        # between a newly created atom and the synthon fragment, preferring
        # bonds where a second new atom is already attached to the core.
        for bond in product.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in new_set and a2 in synthon_indices:
                return (a1, a2)
            if a2 in new_set and a1 in synthon_indices:
                return (a2, a1)
        return None


__all__ = [
    "REACTION_TEMPLATES",
    "HANDLE_SMARTS",
    "REACTION_SIDES",
    "REACTION_PARTNER_HANDLES",
    "HANDLE_TO_REACTIONS",
    "REACTION_FAMILY_MEMBERS",
    "REACTION_FAMILY_NAMES",
    "REACTION_FAMILY_FOR",
    "HANDLE_NAMES",
    "HANDLE_REACTIVITY_TIERS",
    "HandleInfo",
    "ReactionResult",
    "ReactionEngine",
]
