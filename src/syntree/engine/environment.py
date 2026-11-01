"""Discrete Assembly MDP and the terminal geometry-refinement reward oracle.

Clean-room V2 implementation of the *reconciled* two-level GFlowNet
environment (owner task "Reconcile GFlowNet State Space, Torsion Actions,
and Reward Oracle"; issues #24 / #29 / #30 / #38 / #42; constitution #45
§10.2, §12.3, §12.4; decision D-V2-009 in ``docs/agent/DECISIONS.md``).

**Level 1 — the discrete GFlowNet MDP (pure probability space).**
The state is strictly the *Discrete Assembly Derivation*::

    s_t = ( B_0, {(h_i, r_i, B_i, c_i, k_i)}_{i=1..t} )

a seed building block plus the ordered sequence of discrete actions that
assembled it.  Every action component is categorical and dimensionless:

* ``h_t`` — which active attachment handle of the current molecule reacts;
* ``r_t`` — which registered junction reaction forms the bond;
* ``B_t`` — which catalog building block is added;
* ``c_t`` — discrete enantiomeric parity choice at the junction (``'R'`` /
  ``'S'`` / ``None``);
* ``k_t`` — the torsion bin in ``{1..36}`` (10-degree resolution over
  ``[-pi, pi)``);
* ``STOP`` — the explicit absorbing termination action (#45 §8.4).

During rollout the 3D coordinates are generated **deterministically from
the exact discrete bin centers** ``phi_k = -pi + (k - 0.5) * 2*pi / 36``
(issue #24).  Amide/ester junctions declared by the reaction snap strictly
to the planar trans (180-degree) resonance lock.  **No continuous
optimisation, no force field, and no stochastic noise ever touches an
intermediate MDP state** — repeating an action sequence reproduces the
intermediate coordinates bit-for-bit (tested).

**Level 2 — the physical reward oracle (continuous energy evaluation).**
Upon ``STOP`` the discrete terminal molecule is handed to
:func:`finalize_terminal_molecule`, which (and *only* which) executes the
continuous physics of issues #42 and #13 *outside* the probability space:

1. bounded sub-bin refinement of each rotatable junction dihedral within
   its selected ±5-degree interval via a bracketed Brent optimiser against
   the soft-core pocket potential (issue #42, terminal scope);
2. scaffold-constrained MMFF94 relaxation with position restraints of
   50 kcal/mol/Å² on the seed scaffold atoms (issue #13, terminal scope);
3. the relaxed structure is what the stationary reward of
   :class:`~syntree.engine.rl.ThreeDReward` scores (issue #30).

Because the refinement is a deterministic function of the discrete terminal
state, the reward ``R(x)`` is a valid, stationary, positive scalar function
of ``x`` and the pocket — Trajectory Balance over the discrete actions
remains exact while the physical ligand still receives continuous
geometric refinement before scoring.

**Backward policy (issue #38).**  The backward transition is the *Uniform
Leaf Deletion Policy*::

    P_B(s_{t-1} | s_t) = 1 / |RemovableLeafSynthons(s_t)|

where removable leaf synthons are the non-seed building blocks whose
fragment has exactly one junction bond in the fragment junction graph
(:func:`get_removable_leaves`).  This restores exact flow balance in the
presence of symmetric molecules and alternative reaction orderings
(multi-parent convergence) and is consumed by
:class:`~syntree.engine.gflownet.TrajectoryBalanceEngine`.

**Scope boundaries (deliberate, D-V2-009 / D-V2-015; updated for #11).**
The certified 32-reaction taxonomy of #11 (CHEM-01,
``syntree/chemistry/reactions.py``) **is wired in as the default reaction
registry** (``DEFAULT_REACTION_REGISTRY`` = the 27 policy reactions of the
certified taxonomy; the generic interim junction model it replaced was
superseded and removed, D-V2-016).  The MDP/TB contracts above are
unchanged: same single-junction execution path, same resonance locks —
now certified (amide/urea/carbamate/ester/thioester trans locks), with
the amide/ester trans snap pinned by the reconciliation tests.  The
production flat-bottom soft-core clash potential of #14 (CHEM-04,
``syntree/chemistry/conformer.py``) **is wired in**: the sub-bin torsion
refinement objective below evaluates
``syntree.chemistry.conformer.soft_core_clash_energy`` — the bounded,
flat-bottom, repulsive-only form with the documented atom typing and the
500 kcal/mol per-pair spike ceiling (issue #14's follow-up addendum
demand).  ``soft_core_pocket_energy`` remains exported in its
*docking-surrogate* role (attractive 12-6 + softened Coulomb) for the
reward's ``S_dock``/``S_contact`` terms — that is a different injection
point owned by #30 FLOW-02 and the evaluation stack (a real docking
engine supersedes it there).  :class:`~syntree.engine.rl.ThreeDReward`
documents its surrogate.

This module depends on ``rdkit``, ``numpy``, and (for the terminal oracle
only) ``scipy``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetDihedralRad
from scipy.optimize import minimize_scalar

from syntree.chemistry.conformer import soft_core_clash_energy
from syntree.chemistry.reactions import build_junction_reaction_registry

__all__ = [
    "TORSION_BINS",
    "TORSION_BIN_WIDTH",
    "SUBBIN_HALF_WIDTH",
    "AssemblyError",
    "bin_center",
    "bin_bounds",
    "wrap_angle",
    "HandleSpec",
    "JunctionReaction",
    "DEFAULT_REACTION_REGISTRY",
    "BuildingBlock",
    "AssemblyAction",
    "JunctionRecord",
    "StepRecord",
    "PocketDefinition",
    "AssemblyEnvironment",
    "get_removable_leaves",
    "backward_leaf_probability",
    "finalize_terminal_molecule",
    "set_junction_dihedral",
    "soft_core_pocket_energy",
    "soft_core_clash_energy",
    "coupled_charge_energy",
    "SOFT_CORE_DELTA",
    "POCKET_CUTOFF",
    "FRAGMENT_PROP",
    "SEED_PROP",
    "JUNCTIONS_PROP",
]

# ---------------------------------------------------------------------------
# Torsion bin mathematics (issue #24 — the single source of truth)
# ---------------------------------------------------------------------------

#: Number of categorical torsion bins over [-pi, pi) (10-degree resolution).
TORSION_BINS = 36

#: Angular width of one torsion bin in radians (10 degrees).
TORSION_BIN_WIDTH = 2.0 * math.pi / TORSION_BINS

#: Half-width of the sub-bin refinement interval (±5 degrees, issue #42).
SUBBIN_HALF_WIDTH = TORSION_BIN_WIDTH / 2.0


def wrap_angle(angle: float) -> float:
    """Wrap ``angle`` (radians) into ``[-pi, pi)``."""
    two_pi = 2.0 * math.pi
    return (angle + math.pi) % two_pi - math.pi


def bin_center(bin_index: int) -> float:
    """Exact static centre of torsion bin ``bin_index`` (1-based, 1..36).

    ``phi_k = -pi + (k - 0.5) * 2*pi / 36`` — the canonical bin centres used
    for *all* intermediate 3D geometry during rollout (issue #24: the
    discrete action determines the geometry deterministically).
    """
    if isinstance(bin_index, bool) or not isinstance(bin_index, (int, np.integer)):
        raise AssemblyError(
            f"torsion bin index must be an integer in [1, {TORSION_BINS}], got {bin_index!r}"
        )
    if not 1 <= bin_index <= TORSION_BINS:
        raise AssemblyError(
            f"torsion bin index must be in [1, {TORSION_BINS}], got {bin_index}"
        )
    return -math.pi + (bin_index - 0.5) * TORSION_BIN_WIDTH


def bin_bounds(bin_index: int) -> Tuple[float, float]:
    """Inclusive ``(low, high)`` refinement bounds of bin ``bin_index``.

    The bounds are ``center ± 5 degrees``.  Because centres live in
    ``[-175°, +175°]``, the bounds always stay inside ``[-pi, pi]`` — the
    refinement interval never wraps (documented periodicity guarantee,
    issue #42 acceptance).
    """
    center = bin_center(bin_index)
    return center - SUBBIN_HALF_WIDTH, center + SUBBIN_HALF_WIDTH


class AssemblyError(ValueError):
    """Fail-closed error for malformed actions, blocks, or geometry.

    Mirrors the ``TensorContractError`` convention of
    :mod:`syntree.core.schema`: precise message, no silent fallback, no
    NaN masking (#45 §8).
    """


# ---------------------------------------------------------------------------
# Building blocks, handles, reactions
# ---------------------------------------------------------------------------

#: Atom property carrying the fragment (building-block) index.
FRAGMENT_PROP = "syn_tree_fragment"

#: Atom property flagging seed-scaffold atoms (excluded from leaf removal).
SEED_PROP = "syn_tree_seed"

#: Molecule property carrying the serialised junction records (JSON list).
JUNCTIONS_PROP = "syn_tree_junctions"


@dataclass(frozen=True)
class HandleSpec:
    """One attachment handle of a building block.

    ``atom_idx`` is the *attachment atom* that keeps bonding in the product;
    ``leaving_atoms`` are atom indices deleted from the block when the
    junction forms (e.g. the chloride of an aryl chloride).  ``handle_class``
    is matched against :attr:`JunctionReaction.block_handle_class` to pick
    the block-side handle for a reaction.  ``capacity`` is how many junctions
    this handle can form (default 1; #11 CHEM-01 will certify the real
    chemoselectivity rules of #40).
    """

    handle_id: str
    atom_idx: int
    handle_class: str = "generic"
    leaving_atoms: Tuple[int, ...] = ()
    capacity: int = 1

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise AssemblyError(
                f"handle {self.handle_id!r}: capacity must be >= 1, got {self.capacity}"
            )
        if self.atom_idx < 0:
            raise AssemblyError(
                f"handle {self.handle_id!r}: atom_idx must be >= 0, got {self.atom_idx}"
            )


@dataclass(frozen=True)
class JunctionReaction:
    """A registered junction reaction.

    ``block_handle_class`` selects the unique block-side handle whose class
    matches; ``None`` requires the block to expose exactly one handle.
    ``resonance_lock``: ``'trans'`` (amide/ester — snap the junction
    dihedral to planar 180°), ``'cis'`` (0°), or ``None`` (rotatable — the
    torsion bin action governs the dihedral).  The interface is the MDP
    contract carrier; ``DEFAULT_REACTION_REGISTRY`` populates it with the
    certified 32-reaction taxonomy of #11 (CHEM-01, D-V2-016).
    """

    reaction_id: str
    block_handle_class: Optional[str] = None
    bond_type: int = Chem.BondType.SINGLE
    resonance_lock: Optional[str] = None

    def __post_init__(self) -> None:
        if self.resonance_lock not in (None, "trans", "cis"):
            raise AssemblyError(
                f"reaction {self.reaction_id!r}: resonance_lock must be None, "
                f"'trans', or 'cis', got {self.resonance_lock!r}"
            )

    @property
    def locked_dihedral(self) -> Optional[float]:
        if self.resonance_lock is None:
            return None
        return math.pi if self.resonance_lock == "trans" else 0.0


#: Default registry: the 27 policy reactions of the certified 32-reaction
#: taxonomy of #11 CHEM-01 (``syntree/chemistry/reactions.py``), built in
#: the engine's ``JunctionReaction`` format.  This supersedes and removed
#: the generic interim junction model (D-V2-016): there is exactly one
#: authoritative reaction registry in V2.
DEFAULT_REACTION_REGISTRY: Dict[str, JunctionReaction] = build_junction_reaction_registry()


@dataclass
class BuildingBlock:
    """A catalog building block with a deterministic template conformer.

    The template conformer is generated exactly once (ETKDG, fixed
    ``embed_seed``) and cached; every placement during rollout is a rigid
    transform of this cached template, which is what makes repeated action
    sequences bit-for-bit reproducible.  Cross-version/cross-platform
    coordinate equality is deliberately *not* claimed — the determinism
    contract is per RDKit build (decision D-V2-009).
    """

    block_id: str
    smiles: str
    handles: Tuple[HandleSpec, ...]
    embed_seed: int = 2026

    def __post_init__(self) -> None:
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise AssemblyError(f"block {self.block_id!r}: unparseable SMILES {self.smiles!r}")
        if AllChem.EmbedMolecule(mol, randomSeed=self.embed_seed, useRandomCoords=False) != 0:
            raise AssemblyError(f"block {self.block_id!r}: ETKDG embedding failed")
        n = mol.GetNumAtoms()
        seen_ids = set()
        for handle in self.handles:
            if handle.atom_idx >= n or any(idx >= n for idx in handle.leaving_atoms):
                raise AssemblyError(
                    f"block {self.block_id!r}: handle {handle.handle_id!r} indexes "
                    f"outside a {n}-atom template"
                )
            if handle.atom_idx in handle.leaving_atoms:
                raise AssemblyError(
                    f"block {self.block_id!r}: handle {handle.handle_id!r} lists its own "
                    f"attachment atom as a leaving atom"
                )
            if handle.handle_id in seen_ids:
                raise AssemblyError(
                    f"block {self.block_id!r}: duplicate handle id {handle.handle_id!r}"
                )
            seen_ids.add(handle.handle_id)
        object.__setattr__(self, "_template_mol", mol)
        object.__setattr__(
            self,
            "_template_positions",
            np.array(mol.GetConformer().GetPositions(), dtype=np.float64),
        )

    @property
    def template_mol(self) -> Chem.Mol:
        return self._template_mol  # type: ignore[attr-defined]

    @property
    def template_positions(self) -> np.ndarray:
        return self._template_positions  # type: ignore[attr-defined]


@dataclass(frozen=True)
class AssemblyAction:
    """One discrete MDP action ``(h_t, r_t, B_t, c_t, k_t)`` (issue #24).

    All components are categorical and dimensionless; ``torsion_bin`` is
    1-based in ``[1, 36]``.
    """

    handle_atom_idx: int
    reaction_id: str
    block_id: str
    chiral: Optional[str] = None
    torsion_bin: int = 18

    def __post_init__(self) -> None:
        if self.chiral not in (None, "R", "S"):
            raise AssemblyError(
                f"chiral parity must be None, 'R', or 'S', got {self.chiral!r}"
            )
        # Validate the torsion bin eagerly (fail-closed before any geometry).
        bin_center(self.torsion_bin)


@dataclass(frozen=True)
class JunctionRecord:
    """Geometry/provenance record of one formed junction bond.

    ``a1``/``b1`` are the attachment atoms (molecule side / block side),
    ``a0``/``b0`` the reference neighbours defining the junction dihedral
    ``(a0, a1, b1, b0)``, and ``distal`` the atom set beyond ``b1`` used by
    the terminal sub-bin refinement.  Atom indices refer to the molecule as
    assembled *after* the junction formed.
    """

    fragment_id: int
    a1: int
    b1: int
    a0: int
    b0: int
    distal: Tuple[int, ...]
    torsion_bin: int
    resonance_locked: bool
    reaction_id: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "fragment_id": self.fragment_id,
            "a1": self.a1,
            "b1": self.b1,
            "a0": self.a0,
            "b0": self.b0,
            "distal": list(self.distal),
            "torsion_bin": self.torsion_bin,
            "resonance_locked": self.resonance_locked,
            "reaction_id": self.reaction_id,
        }

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> "JunctionRecord":
        return JunctionRecord(
            fragment_id=int(data["fragment_id"]),
            a1=int(data["a1"]),
            b1=int(data["b1"]),
            a0=int(data["a0"]),
            b0=int(data["b0"]),
            distal=tuple(int(i) for i in data["distal"]),
            torsion_bin=int(data["torsion_bin"]),
            resonance_locked=bool(data["resonance_locked"]),
            reaction_id=str(data["reaction_id"]),
        )


@dataclass
class StepRecord:
    """One recorded MDP step for Trajectory Balance.

    ``num_leaves`` is the removable-leaf count of the state *after* the
    action — the count the uniform leaf deletion backward policy needs for
    ``P_B(s_{t-1} | s_t)`` (issue #38).  For the ``STOP`` step the backward
    term is exactly zero (the absorbing terminal has the unique pre-STOP
    parent; proven in ``docs/architecture/MDP_STATE_SPACE.md`` §6), so its
    leaf count is telemetry only.
    """

    action_type: str  # 'add' | 'stop'
    action: Optional[AssemblyAction] = None
    action_log_prob: Optional[float] = None
    num_leaves: int = 0
    smiles: str = ""
    done: bool = False

    def __post_init__(self) -> None:
        if self.action_type not in ("add", "stop"):
            raise AssemblyError(
                f"action_type must be 'add' or 'stop', got {self.action_type!r}"
            )
        if self.action_type == "add" and not isinstance(self.action, AssemblyAction):
            raise AssemblyError("an 'add' step must carry an AssemblyAction")


# ---------------------------------------------------------------------------
# Pocket definition and the oracle-internal soft-core potential
# ---------------------------------------------------------------------------

#: Soft-core vdW parameters (sigma in Å, epsilon in kcal/mol) for the
#: oracle-internal surrogate.  Values follow customary MMFF-like orders of
#: magnitude and are documented constants — the production soft-core /
#: cavity-volume machinery is owned by #14 CHEM-04 and supersedes this via
#: injection when it lands (decision D-V2-009 boundary).
_ORACLE_VDW: Dict[int, Tuple[float, float]] = {
    1: (2.38, 0.015),
    6: (3.36, 0.086),
    7: (3.31, 0.170),
    8: (3.07, 0.170),
    9: (3.07, 0.150),
    15: (3.90, 0.200),
    16: (3.60, 0.250),
    17: (3.64, 0.250),
    35: (3.90, 0.260),
    53: (4.15, 0.300),
}
_ORACLE_VDW_FALLBACK = (3.40, 0.100)

#: Soft-core displacement (Å) that regularises the r -> 0 singularity.
SOFT_CORE_DELTA = 0.5

#: Distance beyond which oracle pocket interactions are cut (Å).
POCKET_CUTOFF = 12.0

#: Coulomb constant in kcal·Å/(mol·e²).
COULOMB_CONSTANT = 332.06371


@dataclass
class PocketDefinition:
    """A fixed pocket conditioning context (point-atom surrogate).

    ``positions [M, 3]`` (Å), ``atomic_numbers [M]``, ``charges [M]`` (e).
    The pocket is part of the *fixed conditioning context* of the reward
    (constitution §12.4): identical molecule + identical pocket ⇒ identical
    reward, independent of rollout-batch siblings.
    """

    positions: np.ndarray
    atomic_numbers: np.ndarray
    charges: np.ndarray

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        self.atomic_numbers = np.ascontiguousarray(self.atomic_numbers, dtype=np.int64)
        self.charges = np.ascontiguousarray(self.charges, dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise AssemblyError("pocket positions must be [M, 3]")
        m = self.positions.shape[0]
        if m == 0:
            raise AssemblyError("pocket must contain at least one atom")
        if self.atomic_numbers.shape != (m,) or self.charges.shape != (m,):
            raise AssemblyError(
                f"pocket field lengths disagree: {m} positions, "
                f"{self.atomic_numbers.shape} atomic numbers, {self.charges.shape} charges"
            )
        if not np.all(np.isfinite(self.positions)):
            raise AssemblyError("pocket positions contain non-finite values")
        if np.any(self.atomic_numbers < 1):
            raise AssemblyError("pocket atomic numbers must be >= 1")

    @property
    def num_atoms(self) -> int:
        return int(self.positions.shape[0])


def _vdw(atomic_number: int) -> Tuple[float, float]:
    return _ORACLE_VDW.get(int(atomic_number), _ORACLE_VDW_FALLBACK)


def soft_core_pocket_energy(
    ligand_positions: np.ndarray,
    ligand_atomic_numbers: np.ndarray,
    pocket: PocketDefinition,
) -> float:
    """Soft-core Lennard-Jones interaction energy of a ligand pose (kcal/mol).

    ``E = sum_ij 4 sqrt(eps_i eps_j) [ (sigma_ij / r_tilde)^12
    - (sigma_ij / r_tilde)^6 ]`` with Lorentz-Berthelot mixing
    ``sigma_ij = (sigma_i + sigma_j) / 2`` and the soft-core displacement
    ``r_tilde = sqrt(r^2 + delta^2)`` (``delta = 0.5`` Å), cut at 12 Å.
    Deterministic, bounded (``|E| <= sum_ij 4 sqrt(eps_i eps_j)``), and
    finite for every input — the r -> 0 singularity is regularised.

    **Role after #14 (D-V2-015):** this is now the *docking surrogate*
    consumed only by :class:`~syntree.engine.rl.ThreeDReward`'s ``S_dock``
    and ``S_contact`` (attractive 12-6 + softened Coulomb — the affinity
    semantics a clash penalty cannot express).  The **production clash
    physics** of #14 CHEM-04 lives in
    :mod:`syntree.chemistry.conformer` (flat-bottom, repulsive-only,
    bounded spike) and is what the terminal sub-bin torsion refinement
    objective consumes.  A real docking engine (evaluation stack / #30)
    supersedes *this* function without touching the MDP/TB contracts.
    """
    ligand_positions = np.asarray(ligand_positions, dtype=np.float64)
    if ligand_positions.ndim != 2 or ligand_positions.shape[1] != 3:
        raise AssemblyError("ligand positions must be [N, 3]")
    deltas = ligand_positions[:, None, :] - pocket.positions[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", deltas, deltas)
    mask = r2 < POCKET_CUTOFF ** 2
    r_tilde = np.sqrt(r2 + SOFT_CORE_DELTA ** 2)
    sig_i = np.array([_vdw(int(z))[0] for z in ligand_atomic_numbers])
    eps_i = np.array([_vdw(int(z))[1] for z in ligand_atomic_numbers])
    sig_j = np.array([_vdw(int(z))[0] for z in pocket.atomic_numbers])
    eps_j = np.array([_vdw(int(z))[1] for z in pocket.atomic_numbers])
    sigma_ij = 0.5 * (sig_i[:, None] + sig_j[None, :])
    eps_ij = np.sqrt(eps_i[:, None] * eps_j[None, :])
    x = sigma_ij / r_tilde
    energy = np.where(mask, 4.0 * eps_ij * (x ** 12 - x ** 6), 0.0)
    total = float(np.sum(energy))
    if not math.isfinite(total):
        raise AssemblyError("soft-core pocket energy is non-finite (poisoned input)")
    return total


def coupled_charge_energy(
    ligand_positions: np.ndarray,
    ligand_charges: np.ndarray,
    pocket: PocketDefinition,
) -> float:
    """Softened Coulomb interaction energy of a ligand pose (kcal/mol).

    ``E = k_c * sum_ij q_i q_j / r_tilde`` with the same ``delta = 0.5`` Å
    softening and 12 Å cutoff.  Ligand charges are supplied by the caller
    (the reward uses deterministic Gasteiger charges).
    """
    ligand_positions = np.asarray(ligand_positions, dtype=np.float64)
    ligand_charges = np.asarray(ligand_charges, dtype=np.float64)
    deltas = ligand_positions[:, None, :] - pocket.positions[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", deltas, deltas)
    mask = r2 < POCKET_CUTOFF ** 2
    r_tilde = np.sqrt(r2 + SOFT_CORE_DELTA ** 2)
    qq = ligand_charges[:, None] * pocket.charges[None, :]
    energy = np.where(mask, COULOMB_CONSTANT * qq / r_tilde, 0.0)
    total = float(np.sum(energy))
    if not math.isfinite(total):
        raise AssemblyError("charge interaction energy is non-finite (poisoned input)")
    return total


# ---------------------------------------------------------------------------
# Leaf semantics — the uniform leaf deletion backward policy (issue #38)
# ---------------------------------------------------------------------------

def _fragment_partition(mol: Chem.Mol) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group atom indices by fragment id; return ``(groups, seed_fragments)``."""
    groups: Dict[int, List[int]] = {}
    seeds: List[int] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if not atom.HasProp(FRAGMENT_PROP):
            raise AssemblyError(
                f"atom {idx} lacks the '{FRAGMENT_PROP}' property — the molecule was "
                f"not produced by the V2 assembly environment"
            )
        frag = int(atom.GetProp(FRAGMENT_PROP))
        groups.setdefault(frag, []).append(idx)
        if atom.HasProp(SEED_PROP):
            seeds.append(frag)
    return groups, sorted(set(seeds))


def get_removable_leaves(mol: Chem.Mol) -> List[int]:
    """Removable leaf synthons of an assembled molecule (fragment ids).

    A fragment is a *removable leaf* when it is **not** the seed scaffold
    and has exactly one junction bond (a bond whose endpoints lie in
    different fragments) in the fragment junction graph.  Removing such a
    fragment always leaves a valid, connected predecessor state — the
    backward action of the uniform leaf deletion policy.  Handles the
    multi-parent convergence of symmetric molecules and alternative
    reaction orderings exactly (issue #38).
    """
    groups, seed_fragments = _fragment_partition(mol)
    junction_degree = {frag: 0 for frag in groups}
    for bond in mol.GetBonds():
        frag_a = int(bond.GetBeginAtom().GetProp(FRAGMENT_PROP))
        frag_b = int(bond.GetEndAtom().GetProp(FRAGMENT_PROP))
        if frag_a != frag_b:
            junction_degree[frag_a] += 1
            junction_degree[frag_b] += 1
    return sorted(
        frag
        for frag, degree in junction_degree.items()
        if degree == 1 and frag not in seed_fragments
    )


def backward_leaf_probability(mol: Chem.Mol) -> float:
    """``P_B(s_{t-1} | s_t)`` under the uniform leaf deletion policy.

    ``1 / |removable leaves|`` — a proper probability distribution over the
    valid parents of the state, which is exactly what restores Trajectory
    Balance flow conservation (#38 / #29).
    """
    n_leaves = len(get_removable_leaves(mol))
    if n_leaves == 0:
        raise AssemblyError(
            "backward_leaf_probability called on a state with no removable leaves "
            "(initial state or non-assembled molecule)"
        )
    return 1.0 / n_leaves


# ---------------------------------------------------------------------------
# Deterministic geometry helpers
# ---------------------------------------------------------------------------

_CANONICAL_BASIS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
)


def _unit(vec: np.ndarray) -> np.ndarray:
    """Deterministic unit vector; fails closed on (near-)zero input."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise AssemblyError("cannot normalise a (near-)zero vector")
    return vec / norm


def _deterministic_perpendicular(u: np.ndarray) -> np.ndarray:
    """A unit vector orthogonal to ``u``, chosen by a fixed canonical rule.

    Uses the canonical basis vector least aligned with ``u`` (lowest index
    on ties) and ``cross(u, e_k)`` — bit-for-bit reproducible.
    """
    dots = [abs(float(np.dot(u, e))) for e in _CANONICAL_BASIS]
    k = int(np.argmin(dots))
    return _unit(np.cross(u, _CANONICAL_BASIS[k]))


def _skew(axis: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )


def _minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The unique minimal (Rodrigues) rotation mapping unit ``source`` to ``target``."""
    source = _unit(source)
    target = _unit(target)
    axis = np.cross(source, target)
    axis_norm = float(np.linalg.norm(axis))
    cos_t = float(np.dot(source, target))
    angle = math.atan2(axis_norm, cos_t)
    if axis_norm < 1e-12:
        if cos_t > 0.0:
            return np.eye(3)
        # Anti-parallel: rotate pi around a deterministic perpendicular axis.
        axis = _deterministic_perpendicular(source)
        return _rotation_about_axis(axis, math.pi)
    axis = axis / axis_norm
    return _rotation_about_axis(axis, angle)


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-hand Rodrigues rotation matrix about unit ``axis`` by ``angle``."""
    axis = _unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    return c * np.eye(3) + s * _skew(axis) + (1.0 - c) * np.outer(axis, axis)


def _hybridization_angle(atom: Chem.Atom) -> float:
    """Deterministic bond angle from atom hybridisation (sp2 -> 120°, else 109.47°)."""
    hyb = atom.GetHybridization()
    if hyb in (Chem.HybridizationType.SP2, Chem.HybridizationType.SP):
        return 2.0 * math.pi / 3.0
    return math.acos(-1.0 / 3.0)  # 109.47° (sp3 / unspecified default)


def _covalent_bond_length(z_a: int, z_b: int) -> float:
    """Deterministic bond length: sum of RDKit covalent radii (Å)."""
    table = Chem.GetPeriodicTable()
    return table.GetRcovalent(int(z_a)) + table.GetRcovalent(int(z_b))


def _reference_neighbor(atom: Chem.Atom, excluded: Sequence[int], prefer_oxygen: bool) -> int:
    """Deterministic dihedral reference neighbour of an attachment atom.

    Prefers a double-bonded oxygen (so a resonance-locked junction measures
    the chemically meaningful O=C-N-X torsion); otherwise the lowest-index
    neighbour not in ``excluded`` (the leaving atoms).  Fails closed when no
    candidate survives.
    """
    candidates = [
        neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetIdx() not in excluded
    ]
    if not candidates:
        raise AssemblyError(
            f"attachment atom {atom.GetIdx()} has no surviving neighbour to define "
            f"the junction dihedral"
        )
    if prefer_oxygen:
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (
                other.GetIdx() in candidates
                and other.GetAtomicNum() == 8
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                return other.GetIdx()
    return min(candidates)


def _bfs_distal_atoms(mol: Chem.Mol, start: int, blocked_bond: Tuple[int, int]) -> List[int]:
    """All atoms reachable from ``start`` without crossing ``blocked_bond``."""
    a, b = blocked_bond
    visited = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = neighbor.GetIdx()
            if (current, idx) in ((a, b), (b, a)) or idx in visited:
                continue
            visited.add(idx)
            stack.append(idx)
    return sorted(visited)


# ---------------------------------------------------------------------------
# The assembly environment (Level 1: discrete MDP)
# ---------------------------------------------------------------------------


class _ActiveHandle:
    """Internal bookkeeping for one live attachment handle in the molecule."""

    __slots__ = (
        "atom_idx",
        "fragment_id",
        "handle_id",
        "handle_class",
        "leaving_atoms",
        "remaining_capacity",
    )

    def __init__(
        self,
        atom_idx: int,
        fragment_id: int,
        handle_id: str,
        handle_class: str,
        leaving_atoms: Tuple[int, ...],
        remaining_capacity: int,
    ) -> None:
        self.atom_idx = atom_idx
        self.fragment_id = fragment_id
        self.handle_id = handle_id
        self.handle_class = handle_class
        self.leaving_atoms = leaving_atoms
        self.remaining_capacity = remaining_capacity


class AssemblyEnvironment:
    """The discrete assembly MDP with the terminal reward oracle.

    Usage (trainer-facing contract)::

        env = AssemblyEnvironment(seed_block, catalog, pocket=..., max_blocks=4)
        env.reset()
        for action in policy(env.observe()):     # discrete actions only
            step = env.step(action)              # deterministic bin-centre geometry
        stop_record = env.stop()                 # absorbing terminal
        relaxed, energy, diag = env.finalize_terminal_molecule()  # Level 2 oracle

    The environment never mutates intermediate geometry with a force field
    or continuous optimiser (issue #24 contract); the oracle runs only
    after ``STOP`` and only on a deep copy (issue #42 terminal scope).
    """

    def __init__(
        self,
        seed_block: BuildingBlock,
        catalog: Mapping[str, BuildingBlock],
        pocket: Optional[PocketDefinition] = None,
        max_blocks: int = 4,
        reaction_registry: Optional[Mapping[str, JunctionReaction]] = None,
    ) -> None:
        if max_blocks < 1:
            raise AssemblyError(f"max_blocks must be >= 1, got {max_blocks}")
        self._seed_block = seed_block
        self._catalog = dict(catalog)
        if seed_block.block_id in self._catalog:
            raise AssemblyError(
                f"seed block {seed_block.block_id!r} must not also appear in the catalog"
            )
        self._pocket = pocket
        self._max_blocks = int(max_blocks)
        self._reactions = dict(reaction_registry or DEFAULT_REACTION_REGISTRY)
        self._mol: Optional[Chem.Mol] = None
        self._junctions: List[JunctionRecord] = []
        self._active_handles: List[_ActiveHandle] = []
        self._num_fragments = 0
        self.trajectory: List[StepRecord] = []
        #: Oracle telemetry (issue #42 logging): refinement / MMFF counters.
        self.stats: Dict[str, int] = {
            "subbin_refinements": 0,
            "mmff_minimizations": 0,
            "brent_evaluations": 0,
        }
        self._terminal = False

    # -- state access --------------------------------------------------------

    @property
    def molecule(self) -> Chem.Mol:
        if self._mol is None:
            raise AssemblyError("reset() must be called before accessing the molecule")
        return self._mol

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    @property
    def pocket(self) -> Optional[PocketDefinition]:
        return self._pocket

    @property
    def num_added_blocks(self) -> int:
        return sum(1 for step in self.trajectory if step.action_type == "add")

    def observe(self) -> Chem.Mol:
        """The current intermediate molecule (deterministic bin-centre geometry)."""
        return self.molecule

    def active_handles(self) -> List[Dict[str, object]]:
        """The active ``(handle_atom_idx, handle_id, class, capacity)`` menu."""
        return [
            {
                "handle_atom_idx": handle.atom_idx,
                "fragment_id": handle.fragment_id,
                "handle_id": handle.handle_id,
                "handle_class": handle.handle_class,
                "remaining_capacity": handle.remaining_capacity,
            }
            for handle in self._active_handles
        ]

    # -- MDP transitions ------------------------------------------------------

    def reset(self) -> Chem.Mol:
        """Reset to the initial state: the seed scaffold alone (fragment 0).

        The trajectory starts empty: its records are the *assembly* actions
        and the terminating STOP.  The initial seed/anchor *selection* action
        of #39 (MODEL-06) will be prepended with its own log-probability
        when that head exists — the TB engine already accepts it as a step
        record with ``num_leaves = 0`` semantics of a seed step.
        """
        template = self._seed_block.template_mol
        positions = self._seed_block.template_positions
        rw = Chem.RWMol(template)
        conformer = Chem.Conformer(rw.GetNumAtoms())
        for idx in range(rw.GetNumAtoms()):
            conformer.SetAtomPosition(idx, positions[idx])
        rw.AddConformer(conformer, assignId=True)
        for idx in range(rw.GetNumAtoms()):
            rw.GetAtomWithIdx(idx).SetIntProp(FRAGMENT_PROP, 0)
            rw.GetAtomWithIdx(idx).SetProp(SEED_PROP, "1")
        product = rw.GetMol()
        Chem.SanitizeMol(product)
        self._mol = product
        self._junctions = []
        self._num_fragments = 1
        self._terminal = False
        self.trajectory = []
        self.stats = {"subbin_refinements": 0, "mmff_minimizations": 0, "brent_evaluations": 0}
        self._active_handles = [
            _ActiveHandle(
                atom_idx=handle.atom_idx,
                fragment_id=0,
                handle_id=handle.handle_id,
                handle_class=handle.handle_class,
                leaving_atoms=tuple(handle.leaving_atoms),
                remaining_capacity=handle.capacity,
            )
            for handle in self._seed_block.handles
        ]
        return product

    def step(self, action: AssemblyAction) -> StepRecord:
        """Apply one discrete assembly action; returns the step record.

        Geometry contract (issue #24): the product conformer is constructed
        deterministically from the cached block template by a rigid
        transform realising the exact bin-centre dihedral (or the resonance
        lock).  Existing atom coordinates are never touched.
        """
        if self._terminal:
            raise AssemblyError("the trajectory is terminal; reset() to start a new one")
        if self.num_added_blocks >= self._max_blocks:
            raise AssemblyError(
                f"max_blocks={self._max_blocks} reached; only STOP is a legal action"
            )
        reaction = self._reactions.get(action.reaction_id)
        if reaction is None:
            raise AssemblyError(
                f"unknown reaction_id {action.reaction_id!r}; registered: "
                f"{sorted(self._reactions)}"
            )
        block = self._catalog.get(action.block_id)
        if block is None:
            raise AssemblyError(
                f"unknown block_id {action.block_id!r}; catalog: {sorted(self._catalog)}"
            )
        handle = next(
            (h for h in self._active_handles if h.atom_idx == action.handle_atom_idx), None
        )
        if handle is None:
            raise AssemblyError(
                f"handle_atom_idx {action.handle_atom_idx} is not an active handle; "
                f"active: {[h['handle_atom_idx'] for h in self.active_handles()]}"
            )
        if handle.remaining_capacity < 1:
            raise AssemblyError(
                f"handle {handle.handle_id!r} at atom {handle.atom_idx} is exhausted"
            )

        # -- locate the block-side handle --------------------------------------
        if reaction.block_handle_class is None:
            matches = list(block.handles)
        else:
            matches = [h for h in block.handles if h.handle_class == reaction.block_handle_class]
        if len(matches) != 1:
            raise AssemblyError(
                f"reaction {reaction.reaction_id!r} requires exactly one block handle "
                f"of class {reaction.block_handle_class!r}; block {block.block_id!r} "
                f"exposes {len(matches)}"
            )
        block_handle = matches[0]

        mol = self.molecule
        a1 = action.handle_atom_idx
        a1_leaving = set(handle.leaving_atoms)
        a1_atom = mol.GetAtomWithIdx(a1)
        locked = reaction.locked_dihedral is not None
        prefer_o = locked
        a0 = _reference_neighbor(a1_atom, sorted(a1_leaving), prefer_oxygen=prefer_o)

        template = block.template_positions
        b1 = block_handle.atom_idx
        b1_leaving = set(block_handle.leaving_atoms)
        b1_atom = block.template_mol.GetAtomWithIdx(b1)
        b0 = _reference_neighbor(b1_atom, sorted(b1_leaving), prefer_oxygen=prefer_o)

        # -- deterministic rigid placement ---------------------------------------
        positions = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
        p_a0, p_a1 = positions[a0], positions[a1]
        u = _unit(p_a0 - p_a1)
        w = _deterministic_perpendicular(u)
        if action.chiral == "S":
            w = -w
        angle_a = _hybridization_angle(a1_atom)
        angle_b = _hybridization_angle(b1_atom)
        bond_length = _covalent_bond_length(a1_atom.GetAtomicNum(), b1_atom.GetAtomicNum())
        d = math.cos(angle_a) * u + math.sin(angle_a) * w
        p_b1 = p_a1 + bond_length * d

        dihedral = reaction.locked_dihedral
        if dihedral is None:
            dihedral = bin_center(action.torsion_bin)
        m = p_a0 - p_a1
        m_perp = _unit(m - float(np.dot(m, d)) * d)
        e_perp = math.cos(dihedral) * m_perp + math.sin(dihedral) * np.cross(d, m_perp)
        e_target = math.cos(angle_b) * (-d) + math.sin(angle_b) * e_perp
        e_t = _unit(template[b0] - template[b1])
        rotation = _minimal_rotation(e_t, e_target)
        placed = p_b1 + (template - template[b1]) @ rotation.T

        # -- assemble the product -------------------------------------------------
        new_fragment_id = self._num_fragments
        product, mol_map, block_map = self._assemble_product(
            mol,
            positions,
            block,
            placed,
            a1,
            a1_leaving,
            b1,
            b1_leaving,
            reaction,
            new_fragment_id,
        )

        # -- invariant check: the realised dihedral matches the target ------------
        new_a0, new_a1 = mol_map[a0], mol_map[a1]
        new_b0, new_b1 = block_map[b0], block_map[b1]
        realized = GetDihedralRad(product.GetConformer(), new_a0, new_a1, new_b1, new_b0)
        if abs(wrap_angle(realized - dihedral)) > 1e-6:
            raise AssemblyError(
                f"internal geometry invariant violated: junction dihedral "
                f"{realized:.8f} rad does not match the target {dihedral:.8f} rad"
            )

        # -- remap the historical junction records through the deletion map ------
        remapped_junctions: List[JunctionRecord] = []
        for record in self._junctions:
            try:
                remapped_junctions.append(
                    JunctionRecord(
                        fragment_id=record.fragment_id,
                        a1=mol_map[record.a1],
                        b1=mol_map[record.b1],
                        a0=mol_map[record.a0],
                        b0=mol_map[record.b0],
                        distal=tuple(mol_map[i] for i in record.distal),
                        torsion_bin=record.torsion_bin,
                        resonance_locked=record.resonance_locked,
                        reaction_id=record.reaction_id,
                    )
                )
            except KeyError as error:
                raise AssemblyError(
                    f"a historical junction atom ({error}) was deleted by this "
                    f"action's leaving group; overlapping leaving sets are not a "
                    f"supported fixture"
                ) from error
        self._junctions = remapped_junctions

        # -- distal atom set (for terminal sub-bin refinement) ---------------------
        distal = tuple(_bfs_distal_atoms(product, new_b1, (new_a1, new_b1)))
        self._junctions.append(
            JunctionRecord(
                fragment_id=new_fragment_id,
                a1=new_a1,
                b1=new_b1,
                a0=new_a0,
                b0=new_b0,
                distal=distal,
                torsion_bin=int(action.torsion_bin),
                resonance_locked=locked,
                reaction_id=reaction.reaction_id,
            )
        )
        _write_junctions(product, self._junctions)
        Chem.AssignStereochemistryFrom3D(product)

        # -- update active-handle bookkeeping ---------------------------------------
        stale_handles = []
        for handle_entry in self._active_handles:
            if handle_entry.atom_idx in mol_map:
                handle_entry.atom_idx = mol_map[handle_entry.atom_idx]
                handle_entry.leaving_atoms = tuple(
                    mol_map[i] for i in handle_entry.leaving_atoms if i in mol_map
                )
            else:
                stale_handles.append(handle_entry)
        if stale_handles:
            raise AssemblyError(
                f"a live handle attachment atom was deleted by this action's "
                f"leaving group (handles: "
                f"{[h.handle_id for h in stale_handles]}); overlapping leaving "
                f"sets are not a supported fixture"
            )
        if handle.remaining_capacity > 1:
            handle.remaining_capacity -= 1
        else:
            self._active_handles.remove(handle)
        for block_side in block.handles:
            if block_side.handle_id == block_handle.handle_id:
                if block_side.capacity > 1:
                    self._active_handles.append(
                        _ActiveHandle(
                            atom_idx=block_map[block_side.atom_idx],
                            fragment_id=new_fragment_id,
                            handle_id=block_side.handle_id,
                            handle_class=block_side.handle_class,
                            leaving_atoms=tuple(
                                block_map[i] for i in block_side.leaving_atoms if i in block_map
                            ),
                            remaining_capacity=block_side.capacity - 1,
                        )
                    )
                continue
            self._active_handles.append(
                _ActiveHandle(
                    atom_idx=block_map[block_side.atom_idx],
                    fragment_id=new_fragment_id,
                    handle_id=block_side.handle_id,
                    handle_class=block_side.handle_class,
                    leaving_atoms=tuple(
                        block_map[i] for i in block_side.leaving_atoms if i in block_map
                    ),
                    remaining_capacity=block_side.capacity,
                )
            )

        self._num_fragments += 1
        self._mol = product
        num_leaves = len(get_removable_leaves(product))
        record = StepRecord(
            action_type="add",
            action=action,
            num_leaves=num_leaves,
            smiles=Chem.MolToSmiles(product),
            done=False,
        )
        self.trajectory.append(record)
        return record

    def stop(self) -> StepRecord:
        """Emit the absorbing STOP action (issue #24 hierarchical termination)."""
        if self._terminal:
            raise AssemblyError("the trajectory is already terminal")
        if self._mol is None:
            raise AssemblyError("reset() must be called before stop()")
        mol = self.molecule
        num_leaves = len(get_removable_leaves(mol))
        record = StepRecord(
            action_type="stop",
            action=None,
            num_leaves=num_leaves,
            smiles=Chem.MolToSmiles(mol),
            done=True,
        )
        self.trajectory.append(record)
        self._terminal = True
        return record

    # -- Level 2: the terminal reward oracle -------------------------------------

    def finalize_terminal_molecule(
        self,
        mol: Optional[Chem.Mol] = None,
        *,
        subbin_refine: bool = True,
        mmff_relax: bool = True,
        brent_xatol: float = 1e-3,
        mmff_max_iterations: int = 200,
        core_force_constant: float = 50.0,
        core_max_displacement: float = 0.3,
        return_diagnostics: bool = False,
    ):
        """Terminal reward-oracle geometry (issues #42 and #13, terminal scope).

        Must be called only after ``STOP`` (fail-closed).  Executes on a
        deep copy: the discrete intermediate/terminal molecule owned by the
        environment is never mutated.  Pipeline: bounded ±5° sub-bin Brent
        refinement of each rotatable junction against the soft-core pocket
        potential, then scaffold-constrained MMFF94 relaxation with
        position restraints ``core_force_constant`` (kcal/mol/Å², default
        50) on the seed-scaffold atoms.  Returns ``(relaxed_mol, energy)``
        or ``(relaxed_mol, energy, diagnostics)``.
        """
        if mol is None:
            if not self._terminal:
                raise AssemblyError(
                    "finalize_terminal_molecule() is a terminal reward-oracle operation: "
                    "it may only run after the STOP action (issue #42 terminal scope)"
                )
            mol = self.molecule
        return finalize_terminal_molecule(
            mol,
            pocket=self._pocket,
            subbin_refine=subbin_refine,
            mmff_relax=mmff_relax,
            brent_xatol=brent_xatol,
            mmff_max_iterations=mmff_max_iterations,
            core_force_constant=core_force_constant,
            core_max_displacement=core_max_displacement,
            return_diagnostics=return_diagnostics,
            stats=self.stats,
        )

    # -- internals ---------------------------------------------------------------

    def _assemble_product(
        self,
        mol: Chem.Mol,
        positions: np.ndarray,
        block: BuildingBlock,
        placed: np.ndarray,
        a1: int,
        a1_leaving: set,
        b1: int,
        b1_leaving: set,
        reaction: JunctionReaction,
        new_fragment_id: int,
    ) -> Tuple[Chem.Mol, Dict[int, int], Dict[int, int]]:
        """Rebuild the product molecule with a deterministic atom order.

        Atom order: surviving molecule atoms (original order) followed by
        surviving block atoms (template order).  Surviving molecule atoms
        keep their coordinates bit-for-bit; block atoms receive the rigid
        placement.  Fragment props are transferred; junction history is
        written by the caller after index remapping.
        """
        surviving_mol = [i for i in range(mol.GetNumAtoms()) if i not in a1_leaving]
        surviving_block = [
            i for i in range(block.template_mol.GetNumAtoms()) if i not in b1_leaving
        ]
        mol_map = {old: new for new, old in enumerate(surviving_mol)}
        block_map = {old: len(surviving_mol) + new for new, old in enumerate(surviving_block)}

        rw = Chem.RWMol()
        for old in surviving_mol:
            rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old)))
        for old in surviving_block:
            rw.AddAtom(Chem.Atom(block.template_mol.GetAtomWithIdx(old)))
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin in mol_map and end in mol_map:
                rw.AddBond(mol_map[begin], mol_map[end], bond.GetBondType())
        for bond in block.template_mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin in block_map and end in block_map:
                rw.AddBond(block_map[begin], block_map[end], bond.GetBondType())
        rw.AddBond(mol_map[a1], block_map[b1], reaction.bond_type)

        for old in surviving_mol:
            atom = rw.GetAtomWithIdx(mol_map[old])
            atom.SetIntProp(FRAGMENT_PROP, int(mol.GetAtomWithIdx(old).GetProp(FRAGMENT_PROP)))
            if mol.GetAtomWithIdx(old).HasProp(SEED_PROP):
                atom.SetProp(SEED_PROP, "1")
        for old in surviving_block:
            rw.GetAtomWithIdx(block_map[old]).SetIntProp(FRAGMENT_PROP, new_fragment_id)

        total = len(surviving_mol) + len(surviving_block)
        conformer = Chem.Conformer(total)
        for old in surviving_mol:
            conformer.SetAtomPosition(mol_map[old], positions[old])
        for old in surviving_block:
            conformer.SetAtomPosition(block_map[old], placed[old])
        rw.AddConformer(conformer, assignId=True)
        product = rw.GetMol()
        try:
            Chem.SanitizeMol(product)
        except Exception as error:  # noqa: BLE001 - fail-closed with context
            raise AssemblyError(
                f"action produced a chemically invalid product (sanitization failed): {error}"
            ) from error
        if product.GetNumAtoms() != total:  # paranoia: sanitization must not drop atoms
            raise AssemblyError("sanitization silently changed the atom count")
        _write_junctions(product, self._junctions)
        return product, mol_map, block_map


def _write_junctions(mol: Chem.Mol, junctions: Sequence[JunctionRecord]) -> None:
    mol.SetProp(JUNCTIONS_PROP, json.dumps([j.as_dict() for j in junctions]))


def _read_junctions(mol: Chem.Mol) -> List[JunctionRecord]:
    if not mol.HasProp(JUNCTIONS_PROP):
        raise AssemblyError(
            f"molecule lacks the '{JUNCTIONS_PROP}' property; only assembled molecules "
            f"can enter the terminal reward oracle"
        )
    return [JunctionRecord.from_dict(item) for item in json.loads(mol.GetProp(JUNCTIONS_PROP))]


# ---------------------------------------------------------------------------
# Terminal reward oracle (Level 2: continuous physics, STOP-gated)
# ---------------------------------------------------------------------------


def set_junction_dihedral(mol: Chem.Mol, junction: JunctionRecord, value: float) -> None:
    """Rotate the distal subtree so the junction dihedral equals ``value``.

    A rigid rotation about the junction bond axis: covalent bond lengths and
    valence angles are preserved bit-for-bit (issue #42 acceptance — the
    refinement may never distort internal covalent geometry).  Public
    primitive so the dense-reference-scan tests of #42 can evaluate the
    objective exactly as the optimiser does.
    """
    conformer = mol.GetConformer()
    p_a1 = np.array(conformer.GetAtomPosition(junction.a1))
    p_b1 = np.array(conformer.GetAtomPosition(junction.b1))
    axis = _unit(p_b1 - p_a1)
    current = GetDihedralRad(conformer, junction.a0, junction.a1, junction.b1, junction.b0)
    delta = wrap_angle(value - current)
    rotation = _rotation_about_axis(axis, delta)
    for idx in junction.distal:
        p = np.array(conformer.GetAtomPosition(idx))
        q = p_a1 + rotation @ (p - p_a1)
        conformer.SetAtomPosition(idx, q)


def finalize_terminal_molecule(
    mol: Chem.Mol,
    pocket: Optional[PocketDefinition] = None,
    *,
    subbin_refine: bool = True,
    mmff_relax: bool = True,
    brent_xatol: float = 1e-3,
    mmff_max_iterations: int = 200,
    core_force_constant: float = 50.0,
    core_max_displacement: float = 0.3,
    return_diagnostics: bool = False,
    stats: Optional[Dict[str, int]] = None,
):
    """Terminal Sub-Bin Geometry Refinement + scaffold-constrained MMFF94.

    This is the *whole* continuous physics of the reconciled design and it
    lives entirely on the terminal side of the MDP (issues #42/#13 as
    re-scoped by the owner reconciliation, decision D-V2-009).  The input
    molecule is deep-copied first — the caller's (and the environment's)
    discrete state is never mutated.

    Sub-bin refinement (issue #42): each rotatable junction dihedral is
    refined inside its selected ±5° interval by a **coarse-scan-seeded
    bracketed Brent optimiser** — a deterministic 9-point grid scan over
    the interval selects the bracket (the best point and its neighbours),
    then ``scipy.optimize.minimize_scalar(method='bounded')`` (Brent's
    method) converges inside that bracket.  The algorithm is named
    accurately per the #42 mandate (it is *not* a plain five-point scan,
    and the Brent phase is a true bracketed optimiser); the two-phase
    design exists because the soft-core objective can be multimodal over
    the interval and a pure Brent run may lock into a local basin.
    Resonance-locked junctions are skipped (planar by construction);
    without a pocket the objective is constant and refinement is a
    documented no-op.  MMFF94 relaxation applies position restraints of
    ``core_force_constant`` (default 50 kcal/mol/Å²) to the seed-scaffold
    atoms.

    Returns ``(relaxed_mol, energy)`` or, with ``return_diagnostics``, the
    triple ``(relaxed_mol, energy, diagnostics)`` where diagnostics records
    per-junction refinement telemetry (selected bin, refined angle,
    objective before/after, optimiser evaluations) and geometry QA (seed
    RMSD, locked-dihedral drift), per issue #42 acceptance.
    """
    work = Chem.Mol(mol)  # deep copy: isolation from the discrete state
    junctions = _read_junctions(work)
    diagnostics: Dict[str, object] = {"junctions": [], "subbin_refine": bool(subbin_refine)}

    if subbin_refine:
        for junction in junctions:
            if junction.resonance_locked:
                continue
            low, high = bin_bounds(junction.torsion_bin)
            entry: Dict[str, object] = {
                "fragment_id": junction.fragment_id,
                "selected_bin": junction.torsion_bin,
                "bin_bounds_rad": [low, high],
            }
            if pocket is None:
                entry["refined_angle_rad"] = bin_center(junction.torsion_bin)
                entry["skipped"] = "no pocket context: the objective is constant"
                diagnostics["junctions"].append(entry)
                continue

            conformer = work.GetConformer()
            numbers = np.array(
                [atom.GetAtomicNum() for atom in work.GetAtoms()], dtype=np.int64
            )

            def objective(value: float) -> float:
                set_junction_dihedral(work, junction, float(value))
                positions = np.array(conformer.GetPositions(), dtype=np.float64)
                # Production flat-bottom soft-core clash potential (#14
                # CHEM-04 / D-V2-015): bounded spike, exact zero beyond
                # contact — the #42 refinement's local torsion objective.
                return soft_core_clash_energy(
                    positions, numbers, pocket.positions, pocket.atomic_numbers
                )

            initial = bin_center(junction.torsion_bin)
            energy_before = objective(initial)
            # Phase 1 — deterministic 9-point coarse scan selects the bracket
            # (the objective can be multimodal over the 10-degree interval).
            coarse_grid = list(np.linspace(low, high, 9))
            coarse_values = [objective(float(angle)) for angle in coarse_grid]
            best_index = int(np.argmin(coarse_values))
            bracket_low = coarse_grid[max(best_index - 1, 0)]
            bracket_high = coarse_grid[min(best_index + 1, len(coarse_grid) - 1)]
            coarse_best_angle = float(coarse_grid[best_index])
            coarse_best_value = float(coarse_values[best_index])
            # Phase 2 — bracketed Brent (scipy fminbound) inside that bracket.
            result = minimize_scalar(
                objective,
                bounds=(bracket_low, bracket_high),
                method="bounded",
                options={"xatol": brent_xatol},
            )
            if float(result.fun) <= coarse_best_value:
                refined_angle, refined_value = float(result.x), float(result.fun)
            else:
                refined_angle, refined_value = coarse_best_angle, coarse_best_value
            set_junction_dihedral(work, junction, refined_angle)
            entry.update(
                {
                    "initial_angle_rad": initial,
                    "refined_angle_rad": refined_angle,
                    "objective_before": energy_before,
                    "objective_after": refined_value,
                    "coarse_evaluations": len(coarse_grid),
                    "brent_evaluations": int(result.nfev),
                    "brent_bracket_rad": [bracket_low, bracket_high],
                }
            )
            diagnostics["junctions"].append(entry)
            if stats is not None:
                stats["subbin_refinements"] += 1
                stats["brent_evaluations"] += int(result.nfev) + len(coarse_grid)
            if not (low - 1e-9 <= refined_angle <= high + 1e-9):
                raise AssemblyError(
                    f"sub-bin refinement escaped its bin interval: {refined_angle} not in "
                    f"[{low}, {high}]"
                )

    energy: Optional[float] = None
    if mmff_relax:
        seed_atoms = [atom.GetIdx() for atom in work.GetAtoms() if atom.HasProp(SEED_PROP)]
        relaxed = Chem.AddHs(work, addCoords=True)
        if not AllChem.MMFFHasAllMoleculeParams(relaxed):
            raise AssemblyError(
                "MMFF94 parameters unavailable for the terminal molecule; the reward "
                "oracle refuses to score an unsupportable structure (fail-closed)"
            )
        properties = AllChem.MMFFGetMoleculeProperties(relaxed, mmffVariant="MMFF94")
        force_field = AllChem.MMFFGetMoleculeForceField(relaxed, properties)
        for idx in seed_atoms:
            force_field.MMFFAddPositionConstraint(
                idx, core_max_displacement, core_force_constant
            )
        force_field.Minimize(maxIts=mmff_max_iterations)
        energy = float(force_field.CalcEnergy())
        work = Chem.RemoveHs(relaxed)
        if stats is not None:
            stats["mmff_minimizations"] += 1
        diagnostics["mmff_energy_kcal_mol"] = energy
        diagnostics["mmff_max_iterations"] = mmff_max_iterations
        diagnostics["core_force_constant_kcal_mol_a2"] = core_force_constant
        diagnostics["num_seed_restrained_atoms"] = len(seed_atoms)

    if return_diagnostics:
        if mmff_relax:
            diagnostics["seed_rmsd_a"] = _seed_rmsd(mol, work)
            locked_drift = []
            conformer = work.GetConformer()
            for junction in junctions:
                if junction.resonance_locked:
                    value = GetDihedralRad(
                        conformer, junction.a0, junction.a1, junction.b1, junction.b0
                    )
                    locked_drift.append(
                        {
                            "fragment_id": junction.fragment_id,
                            "drift_rad": abs(wrap_angle(value - math.pi)),
                        }
                    )
            diagnostics["locked_dihedral_drift"] = locked_drift
        return work, energy, diagnostics
    return work, energy


def _seed_rmsd(reference: Chem.Mol, other: Chem.Mol) -> float:
    """RMSD of the seed-scaffold heavy atoms between two molecules (Å)."""
    ref_idx = [a.GetIdx() for a in reference.GetAtoms() if a.HasProp(SEED_PROP)]
    if not ref_idx:
        return 0.0
    ref_conf = reference.GetConformer()
    other_conf = other.GetConformer()
    total = 0.0
    for idx in ref_idx:
        delta = np.array(ref_conf.GetAtomPosition(idx)) - np.array(other_conf.GetAtomPosition(idx))
        total += float(np.dot(delta, delta))
    return math.sqrt(total / len(ref_idx))
