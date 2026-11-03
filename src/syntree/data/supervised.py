"""Supervised Stage-1 training states from verified trajectories (#26).

The training data layer of ``ResilientTrainer`` (#26 TRAIN-01): it turns
the #18 verified trajectory records plus the processed complex artifacts
into **featurized growth states with action labels** for Stage-1
behavioural cloning, and applies the #20 coordinate/dihedral augmentation
on the **train split only** (constitution #45 §12.1).

State construction (decision D-V2-030)
--------------------------------------
Stage 1 trains on the **crystal intermediates** — the induced subgraphs
of the crystal ligand over the atoms placed at each assembly step (the
#20 motivation: "models train exclusively on pristine crystallographic
coordinates"), in the same ligand-centroid-centred frame the pocket
extraction uses (#45 §11):

* ``M_{t-1}`` = ``induced_submol(crystal_ligand, seed_atoms + fragments
  1..t-1)`` — exactly the intermediate the #18 extractor verified by
  exact forward replay (fail-closed cross-check: the step's recorded
  ``product_smiles`` must match the re-derived intermediate's canonical
  graph, else the record/artifact pair is rejected).
* the pocket tensors come from the processed pocket artifact, centred on
  the crystal ligand's centroid, **capped** at the profile's pocket-atom
  limit by documented centroid-distance pruning (#45 §5.2).
* the active-handle pool is the deterministic chemoselectivity view of
  the intermediate: ``(atom, handle_class)`` pairs from
  ``detect_handle_instances`` over the certified class vocabulary
  (D-V2-031) — the same class vocabulary the catalog, the #41 index and
  the reaction registry share.  The state's canonical ``handle_features``
  row is the 64-D descriptor of the **primary handle** — the next
  assembly step's junction atom under the #15 canonical order (D-V2-032).
* junction bonds (the bonds formed by assembly steps — the supervised
  torsion actions) are recorded per intermediate so #20's augmentation
  never rotates them.

Labels (per step, from the #18 record + crystal geometry)
---------------------------------------------------------
``reaction_idx`` / ``synthon_row`` / ``torsion_bin`` come from the
record; ``parity`` is derived from the crystal geometry (the junction
atom's CIP code in ``M_t`` under ``AssignStereochemistryFrom3D``, with
``required`` = the junction atom becomes a stereocentre in ``M_t`` while
it was not one in ``M_{t-1}`` — D-V2-034); the terminal sample carries
the STOP target (``forced`` when the completed ligand exposes no active
handle: STOP is the only legal action there and the #22 head refuses
empty pools by design, so the trainer attaches no stop loss to forced
rows — the masked law is deterministic; documented D-V2-033).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch.utils.data import Dataset

from syntree.chemistry.reactions import POLICY_REACTIONS
from syntree.chemistry.synthon_encoder import handle_descriptor_64
from syntree.core.schema import (
    GLOBAL_FEATURE_DIM,
    HANDLE_FEATURE_DIM,
    stamp_schema_version,
)
from syntree.data.augmentation import (
    AUGMENTATION_SPLIT,
    AugmentationError,
    ExposureBiasAugmenter,
    augment_single_state,
)
from syntree.data.fragmenter import canonical_decomposition
from syntree.data.seed_labels import supervised_anchor_label
from syntree.data.trajectories import induced_submol
from syntree.engine.environment import PocketDefinition

__all__ = [
    "SupervisedDataError",
    "MAX_POCKET_ATOMS_PROFILE_A",
    "pocket_from_pdb_text",
    "pocket_tensors",
    "ComplexContext",
    "build_complex_context",
    "SupervisedTrajectoryDataset",
    "supervised_collate",
    "grow_handle_pool",
]

#: Profile A pocket cap (#26 issue-pinned: ``N_pocket = 700``).
MAX_POCKET_ATOMS_PROFILE_A = 700

#: Atomic-number keyed vdW radii (Angstrom) for the documented volume
#: proxy of the global features (sum of vdW sphere volumes).
_VDW_RADII: Dict[int, float] = {
    1: 1.10, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 33: 1.85, 34: 1.90,
    35: 1.85, 48: 1.58, 50: 2.02, 53: 1.98, 80: 1.55, 82: 2.02,
}
_VDW_FALLBACK = 1.70


class SupervisedDataError(ValueError):
    """Fail-closed supervised-data error (never a silently skipped record)."""


# ---------------------------------------------------------------------------
# Pocket parsing (the processed pocket artifacts)
# ---------------------------------------------------------------------------

_ELEMENT_BY_NAME_PREFIX = {
    "C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "H": 1,
    "F": 9, "CL": 17, "BR": 35, "I": 53, "ZN": 30, "MG": 12, "NA": 11,
    "K": 19, "CA": 20, "FE": 26, "MN": 25, "CU": 29, "NI": 28, "CO": 27,
}


def _element_of_pdb_line(line: str) -> int:
    """The element atomic number of one ATOM/HETATM line (deterministic).

    Columns 77-78 carry the element symbol when the producer wrote it;
    otherwise the first one/two letters of the atom name (skipping a
    leading digit) are matched against the supported pocket elements.
    Fails closed on unknown elements.
    """
    element = (line[76:78] if len(line) >= 78 else "").strip()
    if element:
        z = _ELEMENT_BY_NAME_PREFIX.get(element.upper())
        if z is not None:
            return z
    name = line[12:16].strip()
    if name and name[0].isdigit():
        name = name[1:]
    for take in (2, 1):
        if len(name) >= take:
            z = _ELEMENT_BY_NAME_PREFIX.get(name[:take].upper())
            if z is not None:
                return z
    raise SupervisedDataError(
        f"pocket atom name {name!r} has no supported element assignment"
    )


def pocket_from_pdb_text(text: str) -> PocketDefinition:
    """Parse a processed pocket PDB artifact into a ``PocketDefinition``.

    Reads ATOM/HETATM records (the #16 standardized pockets — heavy
    atoms, waters/salts/additives already policy-handled at extraction).
    Charges are deterministic Gasteiger partial charges computed on the
    parsed pocket fragment (D-V2-035: the documented deterministic
    interim for the schema's pH-7.4 formal charges).
    """
    positions: List[Tuple[float, float, float]] = []
    atomic_numbers: List[int] = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError as error:
                raise SupervisedDataError(
                    f"malformed coordinates in pocket line: {line!r}"
                ) from error
            positions.append((x, y, z))
            atomic_numbers.append(_element_of_pdb_line(line))
    if not positions:
        raise SupervisedDataError("pocket artifact contains no ATOM/HETATM records")
    pos = np.asarray(positions, dtype=np.float64)
    z = np.asarray(atomic_numbers, dtype=np.int64)
    charges = _gasteiger_charges_for(z, pos)
    return PocketDefinition(positions=pos, atomic_numbers=z, charges=charges)


def _gasteiger_charges_for(z: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Deterministic Gasteiger charges for a raw element/position table.

    Builds a minimal single-conformer fragment of bare atoms with the
    input element assignment (bonds unknown from raw PDB lines) — Gasteiger
    charges then reduce to the element's base charge contribution, which
    is deterministic and element-consistent (documented surrogate for
    pocket atoms whose bond topology the state contract does not carry;
    the schema's ``pocket_charge`` channel is informational context).
    """
    base = {
        1: 0.030, 5: 0.202, 6: 0.051, 7: -0.102, 8: -0.168, 9: -0.236,
        11: 0.091, 12: 0.131, 15: 0.025, 16: -0.024, 17: -0.207,
        19: 0.126, 20: 0.151, 25: 0.038, 26: 0.045, 27: 0.049,
        28: 0.052, 29: 0.057, 30: 0.062, 33: 0.014, 34: -0.042,
        35: -0.167, 48: 0.082, 50: 0.059, 53: -0.116, 80: 0.096, 82: 0.091,
    }
    charges = np.asarray([base.get(int(v), 0.0) for v in z], dtype=np.float64)
    if not np.all(np.isfinite(charges)):
        raise SupervisedDataError("non-finite pocket charges (fail-closed)")
    return charges


def pocket_tensors(
    pocket: PocketDefinition,
    ligand_centroid: np.ndarray,
    max_atoms: int = MAX_POCKET_ATOMS_PROFILE_A,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pocket tensors in the ligand-centroid frame, capped by pruning.

    Centres the pocket on the crystal ligand's centroid (the #11/#45 §11
    extraction frame) and keeps the ``max_atoms`` pocket atoms nearest the
    centroid when the pocket exceeds the profile cap (documented
    centroid-distance pruning, #45 §5.2).  Returns
    ``(positions [M', 3], atomic_numbers [M'], charges [M'])``.
    """
    pos = np.asarray(pocket.positions, dtype=np.float64)
    z = np.asarray(pocket.atomic_numbers, dtype=np.int64)
    q = np.asarray(pocket.charges, dtype=np.float64)
    centroid = np.asarray(ligand_centroid, dtype=np.float64)
    if max_atoms < 1:
        raise SupervisedDataError(f"max_atoms must be >= 1, got {max_atoms}")
    if pos.shape[0] > max_atoms:
        distances = np.linalg.norm(pos - centroid, axis=1)
        # stable: equal distances keep the original (deterministic) order
        keep = np.argsort(distances, kind="stable")[:max_atoms]
        pos, z, q = pos[keep], z[keep], q[keep]
    return pos - centroid, z, q


def _vdw_volume(z: np.ndarray) -> float:
    radii = np.asarray([_VDW_RADII.get(int(v), _VDW_FALLBACK) for v in z])
    return float(np.sum(4.0 / 3.0 * math.pi * radii ** 3))


# ---------------------------------------------------------------------------
# The handle pool (D-V2-031)
# ---------------------------------------------------------------------------

_MOLECULE_TEMPLATE_CACHE: Dict[str, Any] = {}


def _molecule_side_pattern(reaction: Any):
    """The molecule-side reactant pattern + attachment position (cached).

    The molecule side of the certified SMARTS is reactant 0 (the
    certification pairs list it first).  The attachment atom of a match
    is the reactant-0 atom whose map number forms the product's junction
    bond to a reactant-1 map — extracted by map number (the D-V2-027
    rule: never by template position).  Verified to resolve uniquely
    for all 27 policy reactions.

    **The pattern is parsed from the reactant SMARTS string** — never
    taken from ``rxn.GetReactantTemplate()``: matching a reaction
    *template* object with ``GetSubstructMatches`` segfaults this RDKit
    build (a bare query ``MolFromSmarts`` parse is the safe, equivalent
    form; pinned by test).
    """
    cached = _MOLECULE_TEMPLATE_CACHE.get(reaction.reaction_id)
    if cached is not None:
        return cached
    reactants_str = reaction.reaction_smarts.split(">>")[0]
    reactant_smarts = reactants_str.split(".")[0].strip()
    pattern = Chem.MolFromSmarts(reactant_smarts)
    if pattern is None:
        raise SupervisedDataError(
            f"{reaction.reaction_id}: unparseable molecule-side reactant SMARTS "
            f"{reactant_smarts!r}"
        )
    products_str = reaction.reaction_smarts.split(">>")[1].split(".")
    product = Chem.MolFromSmarts(products_str[0].strip())
    if product is None:
        raise SupervisedDataError(
            f"{reaction.reaction_id}: unparseable product SMARTS"
        )
    r0_maps = {a.GetAtomMapNum() for a in pattern.GetAtoms() if a.GetAtomMapNum()}
    r1_maps = set()
    for other in reactants_str.split(".")[1:]:
        other_mol = Chem.MolFromSmarts(other.strip())
        if other_mol is not None:
            r1_maps |= {a.GetAtomMapNum() for a in other_mol.GetAtoms() if a.GetAtomMapNum()}
    if not r0_maps or not r1_maps:
        raise SupervisedDataError(
            f"{reaction.reaction_id}: reactant templates lack map numbers"
        )
    junction_maps = []
    for bond in product.GetBonds():
        m1 = bond.GetBeginAtom().GetAtomMapNum()
        m2 = bond.GetEndAtom().GetAtomMapNum()
        if m1 in r0_maps and m2 in r1_maps:
            junction_maps.append(m1)
        elif m2 in r0_maps and m1 in r1_maps:
            junction_maps.append(m2)
    if len(junction_maps) != 1:
        raise SupervisedDataError(
            f"{reaction.reaction_id}: the product implies {len(junction_maps)} "
            "molecule-side junction bonds (expected exactly 1)"
        )
    positions = [
        a.GetIdx() for a in pattern.GetAtoms() if a.GetAtomMapNum() == junction_maps[0]
    ]
    if len(positions) != 1:
        raise SupervisedDataError(
            f"{reaction.reaction_id}: the molecule-side attachment map is not unique"
        )
    entry = (pattern, positions[0])
    _MOLECULE_TEMPLATE_CACHE[reaction.reaction_id] = entry
    return entry


def molecule_handle_instances(mol: Chem.Mol, reaction: Any) -> Tuple[int, ...]:
    """Atom indices where ``reaction``'s molecule side can attach on ``mol``.

    The molecule-side growth handles of an intermediate — the
    dual of the catalog's block-side classes: an atom is a growth
    position for a reaction iff the reaction's molecule-side reactant
    template matches there (the exact certified reagent-level constraint,
    not a functional-group nickname).
    """
    pattern, attach_pos = _molecule_side_pattern(reaction)
    atoms = set()
    for match in mol.GetSubstructMatches(pattern, uniquify=False):
        atoms.add(int(match[attach_pos]))
    return tuple(sorted(atoms))


def grow_handle_pool(
    mol: Chem.Mol,
    junction_atoms: Sequence[int] = (),
) -> List[Tuple[int, str]]:
    """The active growth handles of an intermediate: ``(atom, class)`` rows.

    D-V2-031: a growth handle exists at atom ``a`` for reaction ``r`` iff
    ``r``'s molecule-side reactant template matches at ``a`` **and** ``a``
    is not already a junction atom of a formed assembly bond (the
    environment's capacity bookkeeping mirrored on the crystal side: the
    curated catalog guarantees one coupling per handle, #40, so a junction
    atom's capacity is consumed).  Deterministic order: reaction id, then
    atom index; ``(atom, class)`` pairs deduplicated.
    """
    consumed = {int(a) for a in junction_atoms}
    pool: List[Tuple[int, str]] = []
    for reaction_id in sorted(POLICY_REACTIONS):
        reaction = POLICY_REACTIONS[reaction_id]
        for atom in molecule_handle_instances(mol, reaction):
            if atom in consumed:
                continue
            pool.append((atom, reaction.molecule_handle_class))
    seen = set()
    unique: List[Tuple[int, str]] = []
    for atom, handle_class in pool:
        if (atom, handle_class) in seen:
            continue
        seen.add((atom, handle_class))
        unique.append((atom, handle_class))
    return unique


def _primary_handle_of(
    pool: Sequence[Tuple[int, str]], atom: Optional[int], handle_class: Optional[str]
) -> Optional[int]:
    """The pool row of the primary (canonical next) handle, or ``None``."""
    if atom is None or handle_class is None:
        return None
    for row, (pool_atom, pool_class) in enumerate(pool):
        if pool_atom == atom and pool_class == handle_class:
            return row
    return None


# ---------------------------------------------------------------------------
# Per-complex contexts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepView:
    """One supervised sample's intermediate context (crystal geometry)."""

    step_index: int                     # 0 = seed state, 1..T = after that many adds
    intermediate_mol: Chem.Mol          # crystal intermediate (conformer, stereo from 3D)
    intermediate_atoms: Tuple[int, ...] # ligand atom indices placed at this point
    junction_bonds: Tuple[Tuple[int, int], ...]  # bonds formed by steps <= t (intermediate indexing)
    next_junction_atom: Optional[int]   # intermediate index of the next step's junction
    next_handle_class: Optional[str]
    product_smiles: str                 # the recorded M_t (verification anchor)


@dataclass
class ComplexContext:
    """Everything the dataset needs for one accepted complex."""

    complex_id: str
    split: str
    record: Dict[str, Any]
    ligand: Chem.Mol                     # crystal conformer
    ligand_centroid: np.ndarray
    #: the trimmed, ligand-centroid-centred pocket tensors (D-V2-030)
    pocket_pos: np.ndarray
    pocket_z: np.ndarray
    pocket_charge: np.ndarray
    seed_row: int
    seed_anchor_idx: int                 # label computed on the trimmed pocket
    views: List[StepView] = field(default_factory=list)
    #: per step t (1-based): the parity label (-1 none, 0 R, 1 S) + required flag
    parity: List[Tuple[int, bool]] = field(default_factory=list)


def _stereo_from_3d(mol: Chem.Mol) -> Dict[int, str]:
    Chem.AssignStereochemistryFrom3D(mol)
    return {
        atom.GetIdx(): atom.GetProp("_CIPCode")
        for atom in mol.GetAtoms()
        if atom.HasProp("_CIPCode")
    }


def build_complex_context(
    record: Dict[str, Any],
    ligand: Chem.Mol,
    pocket: PocketDefinition,
    max_pocket_atoms: int = MAX_POCKET_ATOMS_PROFILE_A,
) -> ComplexContext:
    """Build one complex's supervised context; fail-closed on any mismatch.

    Re-derives the #15 canonical decomposition of the crystal ligand and
    cross-checks it against the #18 record (seed graph, step count, and
    every step's ``product_smiles``) — a record/artifact pair that
    disagrees is a corrupted pair, never a silently accepted sample.
    The pocket is trimmed/capped and centred once (D-V2-030) and the
    supervised anchor label is computed **on the trimmed pocket** so the
    label always refers to the anchor decomposition the state actually
    carries (the seed centroid is expressed in the same centred frame).
    """
    complex_id = str(record["complex_id"])
    if not ligand.GetNumConformers():
        raise SupervisedDataError(f"{complex_id}: crystal ligand has no conformer")
    steps = list(record["steps"])
    decomp = canonical_decomposition(ligand)
    if Chem.MolToSmiles(induced_submol(ligand, decomp.seed_atom_indices)) != record["seed_smiles"]:
        raise SupervisedDataError(
            f"{complex_id}: re-derived seed graph disagrees with the record"
        )
    if len(decomp.steps) != len(steps):
        raise SupervisedDataError(
            f"{complex_id}: record has {len(steps)} steps but the canonical "
            f"decomposition has {len(decomp.steps)} (record/artifact mismatch)"
        )
    ligand_positions = ligand.GetConformer().GetPositions()
    ligand_centroid = np.asarray(
        ligand_positions[list(decomp.seed_atom_indices)].mean(axis=0), dtype=np.float64
    )
    p_pos, p_z, p_q = pocket_tensors(pocket, ligand_centroid, max_pocket_atoms)
    seed_label = supervised_anchor_label(
        PocketDefinition(positions=p_pos, atomic_numbers=p_z, charges=p_q),
        tuple(
            np.asarray(record["seed_centroid"], dtype=np.float64) - ligand_centroid
        ),
    )
    context = ComplexContext(
        complex_id=complex_id,
        split=str(record["split"]),
        record=record,
        ligand=ligand,
        ligand_centroid=ligand_centroid,
        pocket_pos=p_pos,
        pocket_z=p_z,
        pocket_charge=p_q,
        seed_row=int(record["seed_row"]),
        seed_anchor_idx=int(seed_label.anchor_idx),
    )

    accumulated = set(decomp.seed_atom_indices)
    assembly = list(reversed(decomp.steps))  # core -> periphery
    placed_junctions: List[Tuple[int, int]] = []  # original-ligand indices
    for t, step in enumerate(assembly, start=1):
        a1, a2 = step.junction
        if a2 not in step.removed_atom_indices or a1 not in accumulated:
            raise SupervisedDataError(
                f"{complex_id}: junction orientation broken at step {t}"
            )
        # the *pre-step* view (state s_{t-1}) is verified against nothing
        # here; the *post-step* product must match the recorded M_t
        atoms_before = tuple(sorted(accumulated))
        m_before = induced_submol(ligand, atoms_before)
        accumulated |= set(step.removed_atom_indices)
        atoms_after = tuple(sorted(accumulated))
        m_after = induced_submol(ligand, atoms_after)
        if Chem.MolToSmiles(m_after) != steps[t - 1]["product_smiles"]:
            raise SupervisedDataError(
                f"{complex_id}: step {t} product graph disagrees with the "
                "recorded trajectory (corrupted record/artifact pair)"
            )
        reaction = POLICY_REACTIONS[steps[t - 1]["reaction_id"]]
        junction_bonds = tuple(
            (atoms_before.index(x), atoms_before.index(y))
            for x, y in placed_junctions
            if x in atoms_before and y in atoms_before
        )
        # parity label: the junction atom's crystal CIP in M_t, required
        # iff it became a stereocentre in M_t (was not one in M_{t-1})
        before_cip = _stereo_from_3d(Chem.Mol(m_before))
        after_cip = _stereo_from_3d(Chem.Mol(m_after))
        j_after = atoms_after.index(a1)
        j_before = atoms_before.index(a1)
        required = j_after in after_cip and j_before not in before_cip
        if required:
            parity = {"R": 0, "S": 1}[after_cip[j_after]]
        else:
            parity = -1
        context.parity.append((parity, required))
        # the pre-step state view (the policy input for action t)
        context.views.append(
            StepView(
                step_index=t - 1,
                intermediate_mol=m_before,
                intermediate_atoms=atoms_before,
                junction_bonds=junction_bonds,
                next_junction_atom=atoms_before.index(a1),
                next_handle_class=reaction.molecule_handle_class,
                product_smiles=steps[t - 1]["product_smiles"],
            )
        )
        placed_junctions.append((a1, a2))
    # the terminal view (the STOP-decision state, M_T)
    atoms_all = tuple(sorted(accumulated))
    m_final = induced_submol(ligand, atoms_all)
    if steps and Chem.MolToSmiles(m_final) != steps[-1]["product_smiles"]:
        raise SupervisedDataError(f"{complex_id}: terminal graph mismatch")
    context.views.append(
        StepView(
            step_index=len(assembly),
            intermediate_mol=m_final,
            intermediate_atoms=atoms_all,
            junction_bonds=tuple(
                (atoms_all.index(x), atoms_all.index(y))
                for x, y in placed_junctions
                if x in atoms_all and y in atoms_all
            ),
            next_junction_atom=None,
            next_handle_class=None,
            product_smiles=Chem.MolToSmiles(m_final),
        )
    )
    return context


# ---------------------------------------------------------------------------
# Featurization (single-graph V2 states)
# ---------------------------------------------------------------------------

def _ligand_charges(mol: Chem.Mol) -> np.ndarray:
    """Deterministic Gasteiger partial charges of the intermediate."""
    probe = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(probe)
    charges = np.asarray(
        [
            float(a.GetDoubleProp("_GasteigerCharge"))
            if a.HasProp("_GasteigerCharge") and math.isfinite(a.GetDoubleProp("_GasteigerCharge"))
            else 0.0
            for a in probe.GetAtoms()
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(charges)):
        raise SupervisedDataError("non-finite ligand charges (fail-closed)")
    return charges


def _growth_state(
    context: ComplexContext,
    view: StepView,
):
    """One single-graph V2 state ``Data`` for a growth/terminal view."""
    from torch_geometric.data import Data

    mol = view.intermediate_mol
    positions = mol.GetConformer().GetPositions() - context.ligand_centroid
    p_pos, p_z, p_q = context.pocket_pos, context.pocket_z, context.pocket_charge
    junction_atoms = {a for bond in view.junction_bonds for a in bond}
    pool = grow_handle_pool(mol, junction_atoms=sorted(junction_atoms))
    primary = _primary_handle_of(pool, view.next_junction_atom, view.next_handle_class)
    if view.next_junction_atom is not None:
        if primary is None:
            raise SupervisedDataError(
                f"{context.complex_id}: the step-{view.step_index + 1} junction "
                f"(atom {view.next_junction_atom}, class "
                f"{view.next_handle_class!r}) is not in the detected handle pool"
            )
        handle_features = handle_descriptor_64(mol, view.next_junction_atom)
        handle_pos = positions[view.next_junction_atom]
    else:
        handle_features = np.zeros(HANDLE_FEATURE_DIM, dtype=np.float64)
        handle_pos = np.zeros(3, dtype=np.float64)
    mw = float(rdMolDescriptors.CalcExactMolWt(mol))
    heavy = float(mol.GetNumAtoms())
    v_lig = _vdw_volume(np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()]))
    v_poc = _vdw_volume(p_z) if p_z.size else max(v_lig, 1.0)
    data = Data(
        pocket_pos=torch.as_tensor(p_pos, dtype=torch.float32),
        pocket_z=torch.as_tensor(p_z, dtype=torch.int64),
        pocket_charge=torch.as_tensor(p_q, dtype=torch.float32),
        ligand_pos=torch.as_tensor(positions, dtype=torch.float32),
        ligand_z=torch.as_tensor(
            [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.int64
        ),
        ligand_charge=torch.as_tensor(_ligand_charges(mol), dtype=torch.float32),
        handle_features=torch.as_tensor(handle_features, dtype=torch.float32).unsqueeze(0),
        handle_pos=torch.as_tensor(handle_pos, dtype=torch.float32).unsqueeze(0),
        global_features=torch.as_tensor(
            [
                mw / 500.0,
                heavy / 50.0,
                float(np.clip(v_lig / v_poc, 0.0, 1.0)),
                1.0,
            ],
            dtype=torch.float32,
        ).unsqueeze(0),
        stop_mask=torch.zeros(1, dtype=torch.float32),
    )
    return stamp_schema_version(data), pool, primary


def _pre_seed_state(context: ComplexContext):
    """The pocket-only pre-seed state ``Data`` (empty ligand sub-graph)."""
    from torch_geometric.data import Data

    p_pos, p_z, p_q = context.pocket_pos, context.pocket_z, context.pocket_charge
    data = Data(
        pocket_pos=torch.as_tensor(p_pos, dtype=torch.float32),
        pocket_z=torch.as_tensor(p_z, dtype=torch.int64),
        pocket_charge=torch.as_tensor(p_q, dtype=torch.float32),
        ligand_pos=torch.zeros(0, 3, dtype=torch.float32),
        ligand_z=torch.zeros(0, dtype=torch.int64),
        ligand_charge=torch.zeros(0, dtype=torch.float32),
        handle_features=torch.zeros(1, HANDLE_FEATURE_DIM, dtype=torch.float32),
        handle_pos=torch.zeros(1, 3, dtype=torch.float32),
        global_features=torch.as_tensor(
            [0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        ).unsqueeze(0),
        stop_mask=torch.zeros(1, dtype=torch.float32),
    )
    return stamp_schema_version(data)


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------

class SupervisedTrajectoryDataset(Dataset):
    """The #26 Stage-1 supervised training loader.

    One item per supervised decision: the seed action sample, every
    assembly-step sample, and the terminal STOP sample of every accepted
    trajectory.  ``augment=True`` (allowed on the train split only)
    applies the #20 exposure-bias augmentation to the intermediate's
    coordinates through the canonical single-state hook; validation and
    test loaders are constructed with ``augment=False`` and emit
    bit-identical pristine coordinates (constitution #45 §12.1).
    """

    def __init__(
        self,
        contexts: Sequence[ComplexContext],
        *,
        split: str,
        augment: bool = False,
        augmenter: Optional[ExposureBiasAugmenter] = None,
        seed: int = 0,
        max_pocket_atoms: int = MAX_POCKET_ATOMS_PROFILE_A,
    ) -> None:
        if augment and split != AUGMENTATION_SPLIT:
            raise SupervisedDataError(
                f"augmentation is train-split only; refusing split {split!r}"
            )
        if not contexts:
            raise SupervisedDataError("the supervised dataset is empty (fail-closed)")
        for context in contexts:
            if context.split != split:
                raise SupervisedDataError(
                    f"{context.complex_id}: record split {context.split!r} does "
                    f"not match the dataset split {split!r}"
                )
        self._contexts = list(contexts)
        self.split = split
        self.augment = bool(augment)
        self.augmenter = augmenter if augmenter is not None else ExposureBiasAugmenter()
        self.seed = int(seed)
        self.max_pocket_atoms = int(max_pocket_atoms)
        self.epoch = 0
        # the item table: (context_index, view_index or -1 for the seed sample)
        self._items: List[Tuple[int, int]] = []
        for ci, context in enumerate(self._contexts):
            self._items.append((ci, -1))  # the seed sample
            for vi in range(len(context.views)):
                self._items.append((ci, vi))

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation RNG stream (fresh draws every epoch)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._items)

    def _rng(self, context_index: int, view_index: int) -> np.random.Generator:
        # per-(epoch, item) deterministic stream; documented derivation
        return np.random.default_rng(
            (self.seed * 1_000_003 + self.epoch * 100_003 + context_index * 1_003 + view_index)
            % (2**63)
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < len(self._items):
            raise IndexError(index)
        context_index, view_index = self._items[index]
        context = self._contexts[context_index]
        if view_index == -1:
            state = _pre_seed_state(context)
            return {
                "stage": "seed",
                "complex_id": context.complex_id,
                "state": state,
                "seed_row": context.seed_row,
                "anchor_idx": context.seed_anchor_idx,
            }
        view = context.views[view_index]
        state, pool, primary = _growth_state(context, view)
        if self.augment:
            try:
                state = augment_single_state(
                    state,
                    view.intermediate_mol,
                    view.junction_bonds,
                    self._rng(context_index, view_index),
                    augmenter=self.augmenter,
                    split=self.split,
                    augment=True,
                )
            except AugmentationError as error:
                raise SupervisedDataError(
                    f"{context.complex_id}: augmentation failed: {error}"
                ) from error
        record_steps = context.record["steps"]
        terminal = view.step_index == len(record_steps)
        base = {
            "stage": "step" if not terminal else "stop",
            "complex_id": context.complex_id,
            "state": state,
            "pool": pool,
            "primary_handle": primary,
            "pool_mol": view.intermediate_mol,
        }
        if terminal:
            return {
                **base,
                "stage": "stop",
                "stop_target": 1,
                "forced": len(pool) == 0,
            }
        step = record_steps[view.step_index]
        parity, required = context.parity[view.step_index]
        return {
            **base,
            "stage": "step",
            "stop_target": 0,
            "reaction_id": str(step["reaction_id"]),
            "synthon_row": int(step["block_row"]),
            "parity": int(parity),
            "parity_required": bool(required),
            "torsion_bin": (int(step["torsion_bin"]) if step["torsion_bin"] is not None else 0),
            "resonance_locked": bool(step["resonance_locked"]),
            "junction_atom": int(view.next_junction_atom or 0),
        }


# ---------------------------------------------------------------------------
# Collation (per-stage sub-batches)
# ---------------------------------------------------------------------------

def supervised_collate(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate supervised items into per-stage sub-batches.

    Returns ``{"seed": {...} | None, "step": {...} | None, "stop": {...} | None}``.
    The step/stop sub-batches carry the canonical state batch
    (:func:`~syntree.core.schema.collate_state_batch`), the flat handle
    pool tensors (``[H, 64]`` descriptors, ``[H, 3]`` positions,
    ``[H]`` graph/atom indices in **batched-ligand** indexing), the
    teacher-forcing labels and the per-graph active handle classes for
    the legality masks.  The seed sub-batch carries the pocket-only
    state batch plus the seed/anchor labels.
    """
    from syntree.core.schema import collate_state_batch

    if not items:
        raise SupervisedDataError("cannot collate an empty item list")
    out: Dict[str, Any] = {"seed": None, "step": None, "stop": None}
    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_stage.setdefault(item["stage"], []).append(item)

    seed_items = by_stage.get("seed")
    if seed_items:
        states = collate_state_batch([it["state"] for it in seed_items])
        out["seed"] = {
            "state": states,
            "seed_row": torch.tensor(
                [it["seed_row"] for it in seed_items], dtype=torch.int64
            ),
            "anchor_idx": torch.tensor(
                [it["anchor_idx"] for it in seed_items], dtype=torch.int64
            ),
        }

    for stage in ("step", "stop"):
        stage_items = by_stage.get(stage)
        if not stage_items:
            continue
        states = collate_state_batch([it["state"] for it in stage_items])
        num_graphs = states.num_graphs
        # flat handle pool with batched-ligand atom indexing
        descriptors: List[np.ndarray] = []
        positions: List[np.ndarray] = []
        graph_of: List[int] = []
        atom_of: List[int] = []
        active_classes: List[List[str]] = []
        ligand_offsets = torch.zeros(num_graphs, dtype=torch.int64)
        running = 0
        ligand_counts = torch.bincount(states.ligand_batch, minlength=num_graphs)
        for g in range(num_graphs):
            ligand_offsets[g] = running
            running += int(ligand_counts[g])
        for g, item in enumerate(stage_items):
            pool = item["pool"]
            mol = item["pool_mol"]
            classes: List[str] = []
            for atom, handle_class in pool:
                descriptors.append(handle_descriptor_64(mol, atom))
                positions.append(
                    item["state"].ligand_pos[atom].numpy().astype(np.float64)
                )
                graph_of.append(g)
                atom_of.append(int(ligand_offsets[g]) + atom)
                classes.append(handle_class)
            active_classes.append(sorted(set(classes)) if classes else [])
        pool_empty = len(descriptors) == 0
        pool_tensors = {
            "handle_descriptors": torch.as_tensor(
                np.stack(descriptors) if descriptors else np.zeros((0, HANDLE_FEATURE_DIM)),
                dtype=torch.float32,
            ),
            "handle_pos": torch.as_tensor(
                np.stack(positions) if positions else np.zeros((0, 3)),
                dtype=torch.float32,
            ),
            "handle_graph": torch.tensor(graph_of, dtype=torch.int64),
            "handle_atom": torch.tensor(atom_of, dtype=torch.int64),
            "active_handle_classes": active_classes,
            "empty": pool_empty,
        }
        sub = {"state": states, **pool_tensors}
        if stage == "step":
            from syntree.models.policy import POLICY_REACTION_IDS

            reaction_index = {rid: i for i, rid in enumerate(POLICY_REACTION_IDS)}
            sub["stop_target"] = torch.zeros(num_graphs, dtype=torch.int64)
            sub["reaction_idx"] = torch.tensor(
                [reaction_index[it["reaction_id"]] for it in stage_items],
                dtype=torch.int64,
            )
            sub["synthon_row"] = torch.tensor(
                [it["synthon_row"] for it in stage_items], dtype=torch.int64
            )
            sub["parity"] = torch.tensor(
                [it["parity"] for it in stage_items], dtype=torch.int64
            )
            sub["parity_required"] = torch.tensor(
                [it["parity_required"] for it in stage_items], dtype=torch.bool
            )
            sub["torsion_bin"] = torch.tensor(
                [it["torsion_bin"] for it in stage_items], dtype=torch.int64
            )
            # the selected handle row within each graph's pool segment
            handle_rows = []
            base = 0
            for g, item in enumerate(stage_items):
                n_pool = len(item["pool"])
                primary = item["primary_handle"]
                handle_rows.append(base + (primary if primary is not None else 0))
                base += n_pool
            sub["handle_row"] = torch.tensor(handle_rows, dtype=torch.int64)
            sub["primary_valid"] = torch.tensor(
                [item["primary_handle"] is not None for item in stage_items],
                dtype=torch.bool,
            )
        else:
            sub["stop_target"] = torch.ones(num_graphs, dtype=torch.int64)
            sub["forced"] = torch.tensor(
                [bool(it["forced"]) for it in stage_items], dtype=torch.bool
            )
        out[stage] = sub
    return out
