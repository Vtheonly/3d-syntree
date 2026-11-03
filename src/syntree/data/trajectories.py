"""Trajectory extraction with exact forward-replay verification (#18 DATA-03).

Issue #18: supervised behavioral cloning needs ground-truth step
trajectories, but templates extracted without forward verification
produce hallucinations.  This module turns one crystal ligand into a
verified assembly trajectory over the certified 32-reaction taxonomy
(#11) and the immutable 50k Enamine catalog (#12), using the #41 hashed
inverted index for candidate discovery.

The extraction contract (issue-pinned)
--------------------------------------
1. **Canonical decomposition** (#15 ``canonical_decomposition``) splits
   the ligand at certified junction bonds; the assembly order is
   seed-first, periphery-last (the exact reverse of the deterministic
   peel).
2. **Reactant reconstruction by auto-derived graft specs.**  For every
   policy reaction, the molecule-side and block-side *reactant forms*
   are derived from its certification pairs: the reactant equals the
   cut fragment plus a small **graft** (the leaving group: Cl, Br, B(OH)2,
   ZnCl, SnMe3, Li, …) attached at the junction atom with a documented
   H-count delta.  The spec is derived once, cross-validated on **every**
   certification pair of the reaction (all pairs must imply the identical
   spec and round-trip exactly), and cached.  No reverse SMARTS are
   executed at extraction time — the graft is applied to the specific
   fragment, so multi-junction ligands are unambiguous by construction.
3. **Indexed candidate discovery** (#41): the catalog query form is the
   grafted block reactant (for the documented ``rxn_13`` exception — see
   :data:`PROTONATED_CATALOG_FORM_REACTIONS` — the protonated precursor
   form, because the certified SMARTS consumes the lithiated acetylide
   while the catalog sells the terminal alkyne).  The index returns the
   formula-keyed prefilter; :meth:`SynthonIndex.exact_candidates` keeps
   only exact canonical-parent matches.
4. **Exact forward replay** (the acceptance invariant): the step is
   accepted **only** if ``run_certified_reaction`` on the reconstructed
   molecule-side reactant plus the selected catalog synthon produces a
   product set containing a product graph-isomorphic (canonical
   isomeric SMILES) to the observed crystal intermediate ``M_t`` —
   atom identity, bond order, charge and stereochemistry included
   (constitution #45 §9.6).  Rejections carry explicit failure codes;
   nothing is silently accepted.
5. **Observed junction dihedrals** are computed from the crystal
   coordinates (the ligand conformer) with the environment's reference-
   neighbour rule, and labelled with the same 36-bin categorical torsion
   space the policy consumes (#24).

Failure codes (explicit, logged per complex; never silent)
----------------------------------------------------------
``unparseable_ligand`` / ``decomposition_failed`` / ``no_conformer`` /
``no_matched_seed`` / ``no_matched_synthon`` / ``replay_mismatch`` /
``graft_failed`` / ``dihedral_reference_missing``.

This module depends on ``rdkit`` and on the fragmenter/index modules.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolTransforms import GetDihedralRad

from syntree.chemistry.reactions import (
    POLICY_REACTIONS,
    CertifiedReaction,
    canonical_product_smiles,
    junction_pattern_attachment_atoms,
    run_certified_reaction,
)
from syntree.chemistry.synthon_encoder import detect_seed_classes
from syntree.chemistry.synthon_index import SynthonIndex, parent_fragment
from syntree.data.fragmenter import CanonicalDecomposition, canonical_decomposition

__all__ = [
    "PROTONATED_CATALOG_FORM_REACTIONS",
    "FAILURE_CODES",
    "SideGraft",
    "EMPTY_GRAFT",
    "JunctionSpec",
    "CatalogView",
    "StepLabel",
    "TrajectoryRecord",
    "ExtractionFailure",
    "derive_junction_specs",
    "induced_submol",
    "extract_trajectory",
    "trajectory_to_dict",
]

#: Reactions whose certified catalog blocks are sold in the **protonated
#: precursor form** while the certified reaction SMARTS consumes the
#: metalated reagent.  ``rxn_13`` (Sonogashira with the lithium
#: acetylide): the catalog's ``acetylide`` class stores terminal alkynes
#: ``H-C#C-R`` (verified on the immutable artifact); the certified
#: SMARTS ``[c:1][Br:3].[C:7]#[C:2][Li:4]`` consumes ``R-C#C-Li``.  The
#: catalog *lookup* uses the cut fragment (protonated); the *replay*
#: uses the catalog row's metalated form (the graft applied to the
#: row's own parent, so the block actually used is the catalog entry).
#: The metalation is part of the reaction's certified semantics — its
#: certification pair is ``('Brc1ccccc1', 'CC#C[Li]')``.
PROTONATED_CATALOG_FORM_REACTIONS = frozenset({"rxn_13"})

#: The complete, closed failure-code vocabulary (issue acceptance 2).
FAILURE_CODES = (
    "unparseable_ligand",
    "decomposition_failed",
    "no_conformer",
    "no_matched_seed",
    "no_matched_synthon",
    "replay_mismatch",
    "graft_failed",
    "dihedral_reference_missing",
)


class TrajectoryExtractionError(ValueError):
    """Fail-closed extraction error (malformed inputs, never a silent skip)."""


# ---------------------------------------------------------------------------
# Graft specs: the auto-derived leaving-group reconstruction
# ---------------------------------------------------------------------------

#: RDKit bond-type constants used in serialized specs.
_SINGLE = int(Chem.BondType.SINGLE)
_DOUBLE = int(Chem.BondType.DOUBLE)
_TRIPLE = int(Chem.BondType.TRIPLE)


@dataclass(frozen=True)
class SideGraft:
    """The leaving-group graft of one reaction side.

    ``atoms`` lists the graft atoms in attachment order as
    ``(atomic_number, formal_charge)``; ``internal_bonds`` are
    ``(i, j, bond_type)`` between graft slots; the single
    ``attachment_bond`` is ``(graft_slot, bond_type)`` from that graft
    atom to the junction atom; ``attachment_h_delta`` is the H-count
    change of the junction atom between the cut fragment (auto-repaired
    with implicit H) and the reactant form.
    """

    atoms: Tuple[Tuple[int, int], ...] = ()
    internal_bonds: Tuple[Tuple[int, int, int], ...] = ()
    attachment_bond: Tuple[int, int] = (0, _SINGLE)
    attachment_h_delta: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.atoms

    def key(self) -> tuple:
        return (self.atoms, self.internal_bonds, self.attachment_bond, self.attachment_h_delta)


EMPTY_GRAFT = SideGraft()


def _sanitize_product(mol: Chem.Mol) -> Chem.Mol:
    """Sanitize a constructed molecule; repair split-ring aromaticity.

    Constructing a fragment from an aromatic ring can leave aromatic
    flags whose ring perception needs a second pass (the same repair the
    reverse-reaction prototype needed); anything else fails closed.
    """
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:  # noqa: BLE001 - the documented repair path
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                ),
            )
            Chem.SetAromaticity(mol)
            Chem.SanitizeMol(mol)
            return mol
        except Exception as error:  # noqa: BLE001
            raise TrajectoryExtractionError(
                f"constructed molecule failed sanitization: {error}"
            ) from error


def apply_graft(
    fragment: Chem.Mol, attachment_idx: int, graft: SideGraft
) -> Chem.Mol:
    """Attach a leaving-group graft to a cut fragment (deterministic).

    The input is never mutated; the returned molecule is sanitized.
    Fails closed if the junction atom carries explicit-H neighbours the
    spec cannot represent.
    """
    if graft.is_empty and graft.attachment_h_delta == 0:
        return Chem.Mol(fragment)
    atom = fragment.GetAtomWithIdx(attachment_idx)
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 1:
            raise TrajectoryExtractionError(
                "explicit hydrogen neighbours are not supported by the graft "
                "model (remove explicit Hs before extraction)"
            )
    rw = Chem.RWMol(fragment)
    new_indices: List[int] = []
    for atomic_number, charge in graft.atoms:
        new_atom = Chem.Atom(int(atomic_number))
        new_atom.SetFormalCharge(int(charge))
        new_indices.append(rw.AddAtom(new_atom))
    for i, j, bond_type in graft.internal_bonds:
        rw.AddBond(new_indices[i], new_indices[j], Chem.BondType(int(bond_type)))
    if graft.atoms:
        slot, bond_type = graft.attachment_bond
        rw.AddBond(new_indices[slot], attachment_idx, Chem.BondType(int(bond_type)))
    current_h = atom.GetTotalNumHs()
    target = rw.GetAtomWithIdx(attachment_idx)
    target.SetNoImplicit(True)
    target.SetNumExplicitHs(int(current_h + graft.attachment_h_delta))
    return _sanitize_product(rw.GetMol())


@dataclass(frozen=True)
class JunctionSpec:
    """The per-reaction reactant reconstruction spec (both sides)."""

    reaction_id: str
    molecule: SideGraft
    block: SideGraft


def _derive_side_graft(
    reactant: Chem.Mol, fragment: Chem.Mol, attachment_idx: int
) -> SideGraft:
    """Derive the graft of one side from a certification pair.

    ``reactant`` is the pair's reagent molecule on this side;
    ``fragment`` is the corresponding cut fragment of the pair's expected
    product with ``attachment_idx`` its junction atom.  The fragment must
    match as a heavy-atom substructure of the reactant; the unmatched
    reactant atoms are the graft (an empty graft means the cut-repaired
    fragment *is* the reactant, possibly up to a documented H delta).
    Fails closed when no match follows the fragment-plus-graft model.
    """
    # uniquify=False: a symmetric fragment (e.g. a phenyl) matches the
    # reagent in every symmetry rotation; only the rotations aligning the
    # attachment atom with the grafted position are valid, and default
    # uniquification may keep only an invalid one.
    matches = reactant.GetSubstructMatches(fragment, uniquify=False)
    if not matches:
        raise TrajectoryExtractionError(
            "certification pair does not follow the fragment+graft model: "
            "the cut fragment is not a substructure of the reagent"
        )
    expected_reactant = Chem.MolToSmiles(reactant)
    # the graft is a *structural* spec: round-trip comparisons are
    # stereo-blind (an acylation can flip a neighbouring centre's CIP
    # label @/@@ without changing the configuration the graft cares about;
    # stereo rigour is enforced by the runtime replay, not here).
    expected_reactant_flat = Chem.MolToSmiles(reactant, isomericSmiles=False)
    for match in matches:
        # match[query_idx] = reactant idx; anchor = the junction atom's image
        anchor = match[attachment_idx]
        matched = set(match)
        graft_atoms = [
            idx for idx in range(reactant.GetNumAtoms()) if idx not in matched
        ]
        if not graft_atoms:
            # the reactant may be the cut fragment itself up to an H delta
            delta = (
                reactant.GetAtomWithIdx(anchor).GetTotalNumHs()
                - fragment.GetAtomWithIdx(attachment_idx).GetTotalNumHs()
            )
            candidate = SideGraft(attachment_h_delta=delta)
            try:
                rebuilt = apply_graft(fragment, attachment_idx, candidate)
            except TrajectoryExtractionError:
                continue
            if Chem.MolToSmiles(rebuilt, isomericSmiles=False) == expected_reactant_flat:
                return candidate
            continue
        # bonds: graft-internal, and graft -> matched (must hit ONLY the anchor)
        internal: List[Tuple[int, int, int]] = []
        attachment: Optional[Tuple[int, int]] = None
        slot_of = {reactant_idx: slot for slot, reactant_idx in enumerate(graft_atoms)}
        ok = True
        for reactant_idx in graft_atoms:
            for neighbor in reactant.GetAtomWithIdx(reactant_idx).GetNeighbors():
                other = neighbor.GetIdx()
                if other in slot_of:
                    a, b = sorted((slot_of[reactant_idx], slot_of[other]))
                    internal.append((a, b, int(reactant.GetBondBetweenAtoms(reactant_idx, other).GetBondType())))
                elif other in matched:
                    if other != anchor or attachment is not None:
                        ok = False  # bridges to a non-anchor matched atom
                        break
                    attachment = (
                        slot_of[reactant_idx],
                        int(reactant.GetBondBetweenAtoms(reactant_idx, other).GetBondType()),
                    )
            if not ok:
                break
        if not ok or attachment is None:
            continue
        atoms = tuple(
            (
                reactant.GetAtomWithIdx(idx).GetAtomicNum(),
                reactant.GetAtomWithIdx(idx).GetFormalCharge(),
            )
            for idx in graft_atoms
        )
        # H delta at the anchor (other matched atoms must agree)
        delta = (
            reactant.GetAtomWithIdx(anchor).GetTotalNumHs()
            - fragment.GetAtomWithIdx(attachment_idx).GetTotalNumHs()
        )
        for query_idx, reactant_idx in enumerate(match):
            if query_idx == attachment_idx:
                continue
            if (
                reactant.GetAtomWithIdx(reactant_idx).GetTotalNumHs()
                != fragment.GetAtomWithIdx(query_idx).GetTotalNumHs()
            ):
                ok = False
                break
        if not ok:
            continue
        candidate = SideGraft(
            atoms=atoms,
            internal_bonds=tuple(sorted(set(internal))),
            attachment_bond=attachment,
            attachment_h_delta=delta,
        )
        # round-trip verification: grafting the fragment must reproduce
        # the reagent exactly (stereo-blind canonical comparison — see above)
        rebuilt = apply_graft(fragment, attachment_idx, candidate)
        if Chem.MolToSmiles(rebuilt, isomericSmiles=False) == expected_reactant_flat:
            return candidate
    raise TrajectoryExtractionError(
        f"could not derive a consistent graft for reagent {expected_reactant!r}"
    )


def _pair_fragments(
    reaction: CertifiedReaction, pair
) -> Tuple[Chem.Mol, int, Chem.Mol, int, Chem.Mol]:
    """Cut a certification pair's product at its junction.

    Returns ``(M_fragment, a1_in_M, F_fragment, a2_in_F, product)``.
    The junction is located by trying every legal claim of the reaction's
    junction pattern and verifying the split reproduces the reagents up
    to the graft model.
    """
    product = Chem.MolFromSmiles(pair.expected_product)
    if product is None:
        raise TrajectoryExtractionError("unparseable certification product")
    molecule_reagent = Chem.MolFromSmiles(pair.reagents[0])
    block_reagent = Chem.MolFromSmiles(pair.reagents[1])
    pattern = Chem.MolFromSmarts(reaction.junction_pattern)
    seen_bonds = set()
    # attachment atoms by MAP NUMBER (:1 molecule side, :2 block side) —
    # template position order is not the contract; uniquify=False because
    # symmetric junction patterns (e.g. the urea N-C(=O)-N of rxn_06)
    # match in both orientations and RDKit's automorphism dedup would
    # drop the second one.
    p1, p2 = junction_pattern_attachment_atoms(pattern)
    seen_orientations = set()
    for match in product.GetSubstructMatches(pattern, uniquify=False):
        a1, a2 = int(match[p1]), int(match[p2])
        bond = product.GetBondBetweenAtoms(a1, a2)
        if bond is None or bond.IsInRing():
            continue
        # dedup per ORIENTATION (ordered pair): symmetric patterns claim
        # the same bond in both orientations and both must be attempted.
        if (a1, a2) in seen_orientations:
            continue
        seen_orientations.add((a1, a2))
        m_side = _component_without(product, a1, a2)
        f_side = _component_without(product, a2, a1)
        if a1 not in m_side or a2 not in f_side:
            continue
        M = induced_submol(product, sorted(m_side))
        F = induced_submol(product, sorted(f_side))
        a1_m = sorted(m_side).index(a1)
        a2_f = sorted(f_side).index(a2)
        # the molecule reagent must correspond to the :1 side and the block
        # reagent to the :2 side (try both orientations of asymmetric claims)
        for ma, fa, m_reagent, f_reagent in (
            (M, F, molecule_reagent, block_reagent),
        ):
            try:
                _derive_side_graft(m_reagent, ma, a1_m)
                _derive_side_graft(f_reagent, fa, a2_f)
                return ma, a1_m, fa, a2_f, product
            except TrajectoryExtractionError:
                continue
    raise TrajectoryExtractionError(
        f"{reaction.reaction_id}: no junction split of {pair.expected_product!r} "
        "reproduces the certification reagents under the graft model"
    )


def _component_without(mol: Chem.Mol, start: int, blocked: int) -> set:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in mol.GetAtomWithIdx(node).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx != blocked and idx not in seen:
                seen.add(idx)
                stack.append(idx)
    return seen


_SPECS_LOCK = threading.Lock()
_SPECS_CACHE: Optional[Dict[str, JunctionSpec]] = None


def derive_junction_specs(
    registry: Optional[Sequence[CertifiedReaction]] = None,
) -> Dict[str, JunctionSpec]:
    """Derive + cross-validate the graft spec of every policy reaction.

    For each reaction, every certification pair must yield the **same**
    spec (graft atoms, bonds, H deltas identical) and round-trip exactly;
    a single inconsistent pair fails the whole derivation (fail-closed —
    the extraction machinery refuses to run on uncertified chemistry).
    """
    global _SPECS_CACHE
    with _SPECS_LOCK:
        if _SPECS_CACHE is not None and registry is None:
            return dict(_SPECS_CACHE)
        reactions = list(registry) if registry is not None else list(POLICY_REACTIONS.values())
        specs: Dict[str, JunctionSpec] = {}
        for reaction in reactions:
            mol_grafts = set()
            blk_grafts = set()
            mol_graft: Optional[SideGraft] = None
            blk_graft: Optional[SideGraft] = None
            for pair in reaction.certification:
                M, a1_m, F, a2_f, _product = _pair_fragments(reaction, pair)
                mg = _derive_side_graft(Chem.MolFromSmiles(pair.reagents[0]), M, a1_m)
                bg = _derive_side_graft(Chem.MolFromSmiles(pair.reagents[1]), F, a2_f)
                mol_grafts.add(mg.key())
                blk_grafts.add(bg.key())
                mol_graft, blk_graft = mg, bg
            if len(mol_grafts) != 1 or len(blk_grafts) != 1 or mol_graft is None:
                raise TrajectoryExtractionError(
                    f"{reaction.reaction_id}: certification pairs imply "
                    f"{len(mol_grafts)} molecule grafts / {len(blk_grafts)} block "
                    "grafts — the graft model requires exactly one of each"
                )
            specs[reaction.reaction_id] = JunctionSpec(
                reaction_id=reaction.reaction_id,
                molecule=mol_graft,
                block=blk_graft,
            )
        if registry is None:
            _SPECS_CACHE = specs
        return specs


# ---------------------------------------------------------------------------
# Induced subgraph with coordinates, chirality and bond stereo
# ---------------------------------------------------------------------------

def induced_submol(mol: Chem.Mol, atom_indices: Sequence[int]) -> Chem.Mol:
    """The induced subgraph over ``atom_indices`` (deterministic order).

    Preserves atomic chiral tags, bond stereo (when fully inside the
    fragment) and the first conformer's coordinates (re-indexed).  Atoms
    are added in ascending original order, so ``sorted(indices).index(i)``
    maps an original index to its fragment index.
    """
    keep = sorted(set(int(i) for i in atom_indices))
    if not keep:
        raise TrajectoryExtractionError("cannot extract an empty induced subgraph")
    index_set = set(keep)
    rw = Chem.RWMol()
    mapping: Dict[int, int] = {}
    for old in keep:
        mapping[old] = rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old)))
    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if b in mapping and e in mapping:
            rw.AddBond(mapping[b], mapping[e], bond.GetBondType())
    new_mol = rw.GetMol()
    # copy double-bond stereo when both stereo atoms survive the cut
    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if b in mapping and e in mapping:
            stereo_atoms = list(bond.GetStereoAtoms())  # atom indices
            if stereo_atoms and all(a in mapping for a in stereo_atoms):
                new_bond = new_mol.GetBondBetweenAtoms(mapping[b], mapping[e])
                new_bond.SetStereo(bond.GetStereo())
                new_bond.SetStereoAtoms(
                    *(mapping[a] for a in stereo_atoms)
                )
    if mol.GetNumConformers():
        conformer = Chem.Conformer(len(keep))
        positions = mol.GetConformer().GetPositions()
        for old in keep:
            conformer.SetAtomPosition(mapping[old], positions[old])
        new_mol.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(new_mol)
    # Chiral tags survive the copy; an atom that LOST a heavy neighbour to
    # the cut had its substituent set changed (an implicit H took the
    # place), so its original configuration tag is stale and is cleared —
    # the fragment is a different molecule there.  E/Z stereo is kept only
    # when both defining atoms survive (above).  cleanIt=False then keeps
    # the copied, still-valid stereo flags (cleanIt would strip them).
    for old, new in mapping.items():
        orig_atom = mol.GetAtomWithIdx(old)
        if orig_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        lost_heavy = any(
            neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in mapping
            for neighbor in orig_atom.GetNeighbors()
        )
        if lost_heavy:
            new_mol.GetAtomWithIdx(new).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    Chem.AssignStereochemistry(new_mol, cleanIt=False, force=True)
    return new_mol


# ---------------------------------------------------------------------------
# The catalog view + trajectory records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CatalogView:
    """The catalog columns the replay needs (row-aligned with the index)."""

    block_ids: Tuple[str, ...]
    smiles: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.block_ids) != len(self.smiles):
            raise TrajectoryExtractionError("catalog columns disagree in length")


@dataclass
class StepLabel:
    """One verified assembly step of a trajectory."""

    order: int                       # 1-based assembly order
    reaction_id: str
    block_id: str                    # the selected catalog synthon
    block_row: int
    molecule_reactant_smiles: str    # reconstructed M_{t-1} + molecule graft
    block_reactant_smiles: str       # the replay reagent actually used
    query_form_smiles: str           # the catalog lookup form
    product_smiles: str              # observed intermediate M_t (canonical)
    exact_candidates: int            # candidates surviving the exact stage
    candidates_considered: int       # candidates replayed before acceptance
    replay_product_count: int        # products of the accepted replay
    dihedral_rad: Optional[float]    # observed crystal junction dihedral
    torsion_bin: Optional[int]       # the 36-bin categorical label (1..36)
    resonance_locked: bool


@dataclass
class TrajectoryRecord:
    """One fully verified trajectory (all steps passed exact replay)."""

    complex_id: str
    split: str
    ligand_smiles: str
    seed_block_id: str
    seed_row: int
    seed_class: str                  # the certified seed class used
    seed_smiles: str                 # the seed fragment (canonical)
    steps: List[StepLabel] = field(default_factory=list)
    pocket_artifact: str = ""
    #: The crystal centroid of the root fragment (#39's supervised anchor
    #: label input; empty tuple when unavailable).
    seed_centroid: Tuple[float, float, float] = ()


@dataclass
class ExtractionFailure:
    """One rejected complex with its explicit failure code."""

    complex_id: str
    split: str
    ligand_smiles: str
    failure_code: str
    detail: str = ""


def trajectory_to_dict(record: TrajectoryRecord) -> Dict[str, object]:
    """The serialization form for ``trajectories_{split}.pt``."""
    return {
        "complex_id": record.complex_id,
        "split": record.split,
        "ligand_smiles": record.ligand_smiles,
        "seed_block_id": record.seed_block_id,
        "seed_row": record.seed_row,
        "seed_class": record.seed_class,
        "seed_smiles": record.seed_smiles,
        "seed_centroid": list(record.seed_centroid),
        "pocket_artifact": record.pocket_artifact,
        "steps": [
            {
                "order": s.order,
                "reaction_id": s.reaction_id,
                "block_id": s.block_id,
                "block_row": s.block_row,
                "molecule_reactant_smiles": s.molecule_reactant_smiles,
                "block_reactant_smiles": s.block_reactant_smiles,
                "query_form_smiles": s.query_form_smiles,
                "product_smiles": s.product_smiles,
                "exact_candidates": s.exact_candidates,
                "candidates_considered": s.candidates_considered,
                "replay_product_count": s.replay_product_count,
                "dihedral_rad": s.dihedral_rad,
                "torsion_bin": s.torsion_bin,
                "resonance_locked": s.resonance_locked,
            }
            for s in record.steps
        ],
    }


# ---------------------------------------------------------------------------
# Torsion labels (the environment's 36-bin space, #24)
# ---------------------------------------------------------------------------

def _reference_neighbor(atom: Chem.Atom, excluded: int, prefer_oxygen: bool) -> Optional[int]:
    """The deterministic dihedral reference neighbour (the env's rule).

    Highest atomic number wins; oxygen is preferred for resonance-locked
    (planar) junctions; ties break on the lower index.  ``None`` when the
    atom has no other heavy neighbour.
    """
    best = None
    best_key = None
    for neighbor in atom.GetNeighbors():
        idx = neighbor.GetIdx()
        if idx == excluded or neighbor.GetAtomicNum() == 1:
            continue
        key = (
            (1 if (prefer_oxygen and neighbor.GetAtomicNum() == 8) else 0),
            neighbor.GetAtomicNum(),
            -idx,
        )
        if best_key is None or key > best_key:
            best_key, best = key, idx
    return best


def torsion_bin_of(angle: float) -> int:
    """The 1-based 36-bin categorical label of an angle in [-pi, pi).

    The bins partition the circle as ``[-pi + (k-1)w, -pi + k*w)`` with
    ``w = 2*pi/36`` — the bin whose centre (``bin_center``) is nearest
    the wrapped angle.  This defines the supervision convention for the
    observed crystal dihedral (#18): the label is the bin that would
    realise the observed geometry at its canonical centre.
    """
    two_pi = 2.0 * math.pi
    wrapped = (angle + math.pi) % two_pi - math.pi  # [-pi, pi)
    offset = (wrapped + math.pi) / two_pi * 36.0    # [0, 36)
    index = int(math.floor(offset)) + 1
    if index > 36:  # the offset==36-eps rounding edge
        index = 36
    if index < 1:
        index = 1
    return index


# ---------------------------------------------------------------------------
# The extractor
# ---------------------------------------------------------------------------

def extract_trajectory(
    ligand: Chem.Mol,
    *,
    complex_id: str,
    split: str,
    index: SynthonIndex,
    catalog: CatalogView,
    pocket_artifact: str = "",
    specs: Optional[Dict[str, JunctionSpec]] = None,
) -> object:
    """Extract + verify one ligand's trajectory.

    Returns a :class:`TrajectoryRecord` (every step replay-verified) or
    an :class:`ExtractionFailure` with an explicit code.  The ligand
    must carry its crystal conformer for torsion labels.
    """
    if ligand is None or ligand.GetNumAtoms() == 0:
        return ExtractionFailure(complex_id, split, "", "unparseable_ligand")
    ligand_smiles = Chem.MolToSmiles(ligand)
    if not ligand.GetNumConformers():
        return ExtractionFailure(complex_id, split, ligand_smiles, "no_conformer")

    specs = specs if specs is not None else derive_junction_specs()

    try:
        decomp = canonical_decomposition(ligand)
    except Exception as error:  # noqa: BLE001 - fail closed with a code
        return ExtractionFailure(
            complex_id, split, ligand_smiles, "decomposition_failed", str(error)[:200]
        )

    # -- the seed step ------------------------------------------------------
    seed_mol = induced_submol(ligand, decomp.seed_atom_indices)
    seed_smiles = Chem.MolToSmiles(seed_mol)
    seed_centroid = tuple(
        float(x)
        for x in np.mean(
            ligand.GetConformer().GetPositions()[list(decomp.seed_atom_indices)],
            axis=0,
        )
    )
    seed_classes = detect_seed_classes(seed_mol)
    seed_rows: List[int] = []
    seed_class_used = ""
    for seed_class in seed_classes:
        rows = index.seed_lookup(seed_class, seed_smiles)
        exact = index.exact_candidates(seed_smiles, rows)
        if exact:
            seed_rows = list(exact)
            seed_class_used = seed_class
            break
    if not seed_rows:
        return ExtractionFailure(
            complex_id,
            split,
            ligand_smiles,
            "no_matched_seed",
            f"seed {seed_smiles!r} (classes {seed_classes}) not in the certified seed pool",
        )
    seed_row = seed_rows[0]

    # -- assembly steps (core -> periphery) ---------------------------------
    record = TrajectoryRecord(
        complex_id=complex_id,
        split=split,
        ligand_smiles=ligand_smiles,
        seed_block_id=catalog.block_ids[seed_row],
        seed_row=seed_row,
        seed_class=seed_class_used,
        seed_smiles=seed_smiles,
        pocket_artifact=pocket_artifact,
        seed_centroid=seed_centroid,
    )

    accumulated = set(decomp.seed_atom_indices)
    assembly_steps = list(reversed(decomp.steps))  # core -> periphery
    for order, step in enumerate(assembly_steps, start=1):
        reaction = POLICY_REACTIONS[step.reaction_id]
        spec = specs[step.reaction_id]
        a1, a2 = step.junction  # original ligand indices
        fragment_atoms = sorted(step.removed_atom_indices)
        if a2 not in step.removed_atom_indices or a1 not in accumulated:
            return ExtractionFailure(
                complex_id, split, ligand_smiles, "decomposition_failed",
                f"junction orientation broken at assembly step {order}",
            )
        before = sorted(accumulated)
        m_atom = before.index(a1)
        f_atom = fragment_atoms.index(a2)
        try:
            M_prev = induced_submol(ligand, before)
            F = induced_submol(ligand, fragment_atoms)
            M_r = apply_graft(M_prev, m_atom, spec.molecule)
            B_t = apply_graft(F, f_atom, spec.block)
        except TrajectoryExtractionError as error:
            return ExtractionFailure(
                complex_id, split, ligand_smiles, "graft_failed", str(error)[:200]
            )
        m_r_smiles = Chem.MolToSmiles(M_r)
        b_t_smiles = Chem.MolToSmiles(B_t)
        accumulated |= set(step.removed_atom_indices)
        M_t = induced_submol(ligand, sorted(accumulated))
        product_smiles = Chem.MolToSmiles(M_t)

        # -- indexed candidate discovery + exact stage ----------------------
        query_form = (
            Chem.MolToSmiles(F) if reaction.reaction_id in PROTONATED_CATALOG_FORM_REACTIONS
            else b_t_smiles
        )
        rows = index.lookup(reaction.reaction_id, query_form)
        exact = index.exact_candidates(query_form, rows)
        if not exact:
            return ExtractionFailure(
                complex_id, split, ligand_smiles, "no_matched_synthon",
                f"step {order} ({reaction.reaction_id}): no certified catalog "
                f"synthon matches the reconstructed block {query_form!r}",
            )
        accepted: Optional[StepLabel] = None
        last_error = ""
        considered = 0
        for row in exact:
            considered += 1
            if reaction.reaction_id in PROTONATED_CATALOG_FORM_REACTIONS:
                # replay with the catalog row's own metalated form: the
                # row's parent is graph-identical to the cut fragment; the
                # graft attaches at the a2 image under the exact-match
                # isomorphism.
                parent = Chem.MolFromSmiles(parent_fragment(catalog.smiles[row]))
                image = parent.GetSubstructMatch(F)
                if not image or len(image) != F.GetNumAtoms():
                    last_error = "protonated-form isomorphism lost"
                    continue
                reagent = apply_graft(parent, image[f_atom], spec.block)
                reagent_smiles = Chem.MolToSmiles(reagent)
            else:
                reagent_smiles = catalog.smiles[row]
            try:
                products = run_certified_reaction(reaction, (m_r_smiles, reagent_smiles))
            except Exception as error:  # noqa: BLE001 - a rejected candidate
                last_error = f"{type(error).__name__}: {str(error)[:120]}"
                continue
            observed = {canonical_product_smiles(Chem.Mol(p)) for p in products}
            if product_smiles in observed:
                accepted = StepLabel(
                    order=order,
                    reaction_id=reaction.reaction_id,
                    block_id=catalog.block_ids[row],
                    block_row=row,
                    molecule_reactant_smiles=m_r_smiles,
                    block_reactant_smiles=reagent_smiles,
                    query_form_smiles=query_form,
                    product_smiles=product_smiles,
                    exact_candidates=len(exact),
                    candidates_considered=considered,
                    replay_product_count=len(products),
                    dihedral_rad=None,
                    torsion_bin=None,
                    resonance_locked=reaction.resonance_lock is not None,
                )
                break
        if accepted is None:
            return ExtractionFailure(
                complex_id, split, ligand_smiles, "replay_mismatch",
                f"step {order} ({reaction.reaction_id}): {len(exact)} exact "
                f"candidates, none replays to the observed product "
                f"({last_error})",
            )

        # -- observed junction dihedral from the crystal conformer ----------
        a1_t = sorted(accumulated).index(a1)
        a2_t = sorted(accumulated).index(a2)
        prefer_o = accepted.resonance_locked
        a0 = _reference_neighbor(M_t.GetAtomWithIdx(a1_t), a2_t, prefer_o)
        b0 = _reference_neighbor(M_t.GetAtomWithIdx(a2_t), a1_t, prefer_o)
        if a0 is None or b0 is None:
            return ExtractionFailure(
                complex_id, split, ligand_smiles, "dihedral_reference_missing",
                f"step {order}: no heavy reference neighbour for the dihedral",
            )
        dihedral = GetDihedralRad(M_t.GetConformer(), a0, a1_t, a2_t, b0)
        accepted.dihedral_rad = float(dihedral)
        accepted.torsion_bin = torsion_bin_of(float(dihedral))
        record.steps.append(accepted)

    return record
