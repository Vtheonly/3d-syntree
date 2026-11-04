"""The policy-driven rollout driver — Stage-2 on-policy sampling (#29 FLOW-01).

This is the production bridge between the categorical policy
(:class:`~syntree.models.policy.SynTreePolicy`) and the discrete assembly
MDP (:class:`~syntree.engine.environment.AssemblyEnvironment`): it drives
complete trajectories ``a_seed -> (h, r, B, c, k)* -> STOP``, records the
**tensor** forward log-probabilities of every action component (the
Trajectory Balance sum of #29), computes the stationary terminal reward,
and returns :class:`~syntree.engine.gflownet.TBTrajectory` objects for
:class:`~syntree.engine.gflownet.GFlowNetTrainer`.

**The forward probability measure (constitution §12.3, #29 body).**  One
trajectory contributes::

    log P_F = log P(anchor | P) + log P(seed | anchor, P)          # a_seed
            + sum_t [ log P(CONT) + log P(h_t | s_t)
                      + log P(f_t | s_t) + log P(r_t | f_t, s_t)
                      + log P(B_t | r_t, s_t) + [req] log P(c_t)
                      + [rot] log P(k_t | s_t, B_t) ]
            + log P(STOP | s_T)

exactly the issue-pinned decomposition.  Every term is a pure
dimensionless categorical log-probability gathered from the policy's own
normalized laws (one source of truth — the same heads the supervised
trainer trains); the sub-bin torsion refinement of #42 and the
deterministic seed pose of D-V2-028 are environment-side geometry with no
probability measure (never added to the sum).

**State construction (decision D-V2-041).**  Rollout states use the frozen
#6 schema through the same collation path
(:func:`~syntree.core.schema.collate_state_batch`) as the supervised
loader:

* the pocket is **capped once at construction** by the documented
  centroid-distance pruning (Profile A cap 700, centred on the pocket's
  own centroid — the rollout frame reference, since no crystal ligand
  exists; the environment must be constructed with the same capped pocket
  so its anchor menu and the policy's anchor candidates coincide);
* growth states live in the **placed-seed centroid frame** (the rollout
  analog of the supervised crystal-ligand-centroid frame, D-V2-030);
* the ``handle_features``/``handle_pos`` policy-level state fields carry
  the *undecided primary handle* convention (zeros — the same value the
  supervised terminal/stop rows carry; the backbone does not consume
  these fields, and the pool enters the #22 head through its own inputs).

**The active handle pool (D-V2-031 mirrored).**  The pool is the
molecule-side growth-handle construction of the supervised loader
(:func:`~syntree.data.supervised.grow_handle_pool` — reaction-template
matches at non-junction atoms), intersected with the environment's active
capacity bookkeeping: a row survives only if its atom is an active
environment handle with remaining capacity (every sampled handle is
therefore executable; the environment still enforces the exact chemistry
fail-closed).

**The parity action (#24 contract at rollout).**  Whether a junction
creates a stereocentre is determined *before* the action commits, by
stepping a **cloned environment state** (the env's own assembly code —
one source of truth) with ``chiral=None`` and reading the junction atom's
CIP status; where required, the policy samples the R/S branch and the
sampled branch is applied to the real environment; where not required,
``chiral=None`` is applied and the parity head contributes exactly 0.0
(the supervised rule — no vacuous parity action ever perturbs the
geometry of a non-stereogenic junction, so terminal objects stay a
deterministic function of the scored action tuple).  The prediction is
verified fail-closed against the realized product of the real step.

**The catalog bridge.**  :func:`building_block_from_catalog_row` derives
a :class:`~syntree.engine.environment.BuildingBlock` (attachment atom +
leaving atoms + class) from a catalog row's SMILES and synthon class by
matching the certified registry's block-side reactant SMARTS (parsed from
the string — never from reaction template objects, D-V2-031's pinned
RDKit segfault hazard).  #40's curation guarantees curated blocks expose
exactly one coupling handle of their class; any ambiguity fails closed.

This module depends on ``torch``, ``rdkit`` and ``numpy``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch_geometric.data import Data

from syntree.chemistry.reactions import POLICY_REACTIONS
from syntree.chemistry.synthon_encoder import handle_descriptor_64
from syntree.core.schema import (
    HANDLE_FEATURE_DIM,
    collate_state_batch,
    stamp_schema_version,
)
from syntree.data.supervised import _vdw_volume, grow_handle_pool
from syntree.engine.environment import (
    FRAGMENT_PROP,
    AssemblyAction,
    AssemblyEnvironment,
    AssemblyError,
    BuildingBlock,
    DEFAULT_REACTION_REGISTRY,
    HandleSpec,
    PocketDefinition,
    SeedAction,
)
from syntree.engine.gflownet import FlowError, TBTrajectory, TrajectoryStep
from syntree.engine.rl import ThreeDReward
from syntree.models.policy import (
    POLICY_REACTION_IDS,
    SynTreePolicy,
    build_continuation_masks,
    build_force_stop,
)

__all__ = [
    "RolloutError",
    "block_side_handle_spec",
    "building_block_from_catalog_row",
    "seed_block_from_catalog_row",
    "PolicyRolloutDriver",
    "MAX_ROLLOUT_POCKET_ATOMS",
]

#: The Profile-A pocket cap for rollout conditioning (#45 s5.2).
MAX_ROLLOUT_POCKET_ATOMS = 700

#: The block-side reactant-SMARTS cache: class -> (pattern, attach idx,
#: leaving pattern idxs).  Parsed from the certified SMARTS *string*
#: (D-V2-031: reaction template mols segfault GetSubstructMatches).
_BLOCK_TEMPLATE_CACHE: Dict[str, Any] = {}


class RolloutError(ValueError):
    """Fail-closed rollout error (invalid action, state, or wiring)."""


# ---------------------------------------------------------------------------
# The catalog bridge: catalog row -> BuildingBlock
# ---------------------------------------------------------------------------


def block_side_handle_spec(mol: Chem.Mol, handle_class: str) -> HandleSpec:
    """Derive the block-side ``HandleSpec`` of ``mol`` for ``handle_class``.

    Matches the certified registry's block-side reactant SMARTS (the first
    reaction whose ``block_handle_class`` is ``handle_class``): the
    attachment atom is the matched atom carrying the junction map number
    (the block-side atom of the product's junction bond); the leaving
    atoms are the matched atoms whose map numbers do not survive into the
    product (e.g. the chloride of an acid chloride).  The curated catalog
    guarantees exactly one coupling handle per class (#40): zero or
    multiple matches fail closed.
    """
    cached = _BLOCK_TEMPLATE_CACHE.get(handle_class)
    if cached is None:
        candidates = [
            reaction
            for reaction in POLICY_REACTIONS.values()
            if reaction.block_handle_class == handle_class
        ]
        if not candidates:
            raise RolloutError(
                f"no certified reaction consumes the block-side handle class "
                f"{handle_class!r} (unknown class — fail-closed)"
            )
        reaction = sorted(candidates, key=lambda r: r.reaction_id)[0]
        reactants_str = reaction.reaction_smarts.split(">>")[0]
        parts = [p.strip() for p in reactants_str.split(".")]
        block_smarts = parts[1]
        pattern = Chem.MolFromSmarts(block_smarts)
        if pattern is None:
            raise RolloutError(
                f"{reaction.reaction_id}: unparseable block-side reactant "
                f"SMARTS {block_smarts!r}"
            )
        product = Chem.MolFromSmarts(
            reaction.reaction_smarts.split(">>")[1].split(".")[0].strip()
        )
        if product is None:
            raise RolloutError(
                f"{reaction.reaction_id}: unparseable product SMARTS"
            )
        block_maps = {a.GetAtomMapNum() for a in pattern.GetAtoms() if a.GetAtomMapNum()}
        mol_maps = {
            a.GetAtomMapNum()
            for a in Chem.MolFromSmarts(parts[0]).GetAtoms()
            if a.GetAtomMapNum()
        } if Chem.MolFromSmarts(parts[0]) is not None else set()
        if not block_maps or not mol_maps:
            raise RolloutError(
                f"{reaction.reaction_id}: reactant templates lack map numbers"
            )
        junction_map = None
        for bond in product.GetBonds():
            m1 = bond.GetBeginAtom().GetAtomMapNum()
            m2 = bond.GetEndAtom().GetAtomMapNum()
            if m1 in block_maps and m2 in mol_maps:
                junction_map = m1
            elif m2 in block_maps and m1 in mol_maps:
                junction_map = m2
        if junction_map is None:
            raise RolloutError(
                f"{reaction.reaction_id}: the product implies no block-side "
                "junction bond (expected exactly 1)"
            )
        product_maps = {
            a.GetAtomMapNum() for a in product.GetAtoms() if a.GetAtomMapNum()
        }
        attach_positions = [
            a.GetIdx() for a in pattern.GetAtoms() if a.GetAtomMapNum() == junction_map
        ]
        if len(attach_positions) != 1:
            raise RolloutError(
                f"{reaction.reaction_id}: the block-side attachment map is "
                "not unique"
            )
        leaving_positions = [
            a.GetIdx()
            for a in pattern.GetAtoms()
            if a.GetAtomMapNum() and a.GetAtomMapNum() not in product_maps
        ]
        cached = (pattern, attach_positions[0], tuple(leaving_positions))
        _BLOCK_TEMPLATE_CACHE[handle_class] = cached

    pattern, attach_pos, leaving_positions = cached
    matches = mol.GetSubstructMatches(pattern, uniquify=False)
    if not matches:
        raise RolloutError(
            f"the block does not expose a {handle_class!r} coupling handle "
            "(catalog/class mismatch — fail-closed)"
        )
    # The curated catalog guarantees ONE coupling handle of the class
    # (#40 CHEM-06): distinct attachment atoms = an uncurated
    # multifunctional block.
    attachments = sorted({int(m[attach_pos]) for m in matches})
    if len(attachments) != 1:
        raise RolloutError(
            f"the block exposes {len(attachments)} distinct {handle_class!r} "
            f"handles (atoms {attachments}); the curated catalog guarantees "
            "exactly one coupling handle per class (#40) — fail-closed"
        )
    attach_atom = attachments[0]
    leaving: Set[int] = set()
    for match in matches:
        if int(match[attach_pos]) != attach_atom:
            continue
        leaving.update(int(match[i]) for i in leaving_positions)
    leaving.discard(attach_atom)
    return HandleSpec(
        handle_id=f"{handle_class}@{attach_atom}",
        atom_idx=attach_atom,
        handle_class=handle_class,
        leaving_atoms=tuple(sorted(leaving)),
    )


def building_block_from_catalog_row(
    block_id: str, smiles: str, synthon_class: str
) -> BuildingBlock:
    """A catalog row as an environment :class:`BuildingBlock`.

    ``synthon_class`` is the row's primary certified class (the first
    entry of the catalog's comma-separated class list, the same convention
    as the Stage-1 mask builder); the block-side handle is derived from
    the certified registry (:func:`block_side_handle_spec`).
    """
    primary = synthon_class.split(",")[0].strip()
    if not primary:
        raise RolloutError(
            f"catalog row {block_id!r} has an empty synthon class (seed-only "
            "rows are not growth blocks — do not build them)"
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RolloutError(f"catalog row {block_id!r}: unparseable SMILES {smiles!r}")
    spec = block_side_handle_spec(mol, primary)
    return BuildingBlock(block_id, smiles, (spec,))


def seed_block_from_catalog_row(block_id: str, smiles: str) -> BuildingBlock:
    """A seed-eligible catalog row as a seed :class:`BuildingBlock`.

    Seed blocks are **not** curated coupling blocks: they may expose
    several growth sites (the #39 seed pool is the seed-eligible subset of
    the catalog, not the one-handle-per-class coupling subset).  Their
    handles are therefore the *molecule-side* growth-handle construction
    (the exact :func:`~syntree.data.supervised.grow_handle_pool` semantics
    the policy's pool uses — D-V2-031): one :class:`HandleSpec` per
    ``(atom, class)`` growth site, capacity 1, no leaving atoms (the
    molecule-side atom keeps bonding; hydrogens leave implicitly).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RolloutError(f"seed row {block_id!r}: unparseable SMILES {smiles!r}")
    pool = grow_handle_pool(mol)
    if not pool:
        raise RolloutError(
            f"seed row {block_id!r} exposes no molecule-side growth handle "
            "(a seed must offer at least one growth site — fail-closed)"
        )
    specs = tuple(
        HandleSpec(f"{cls}@{atom}", atom, cls, (), 1) for atom, cls in pool
    )
    return BuildingBlock(block_id, smiles, specs)


# ---------------------------------------------------------------------------
# State construction (D-V2-041)
# ---------------------------------------------------------------------------


def _junction_atoms_of(mol: Chem.Mol) -> Set[int]:
    """The junction atoms of an environment intermediate.

    A junction bond is a bond between two different assembly fragments
    (the ``FRAGMENT_PROP`` bookkeeping the environment writes); its two
    atoms are exactly the ``a1``/``b1`` pairs of the junction records.
    """
    junctions: Set[int] = set()
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.HasProp(FRAGMENT_PROP) and end.HasProp(FRAGMENT_PROP):
            if begin.GetIntProp(FRAGMENT_PROP) != end.GetIntProp(FRAGMENT_PROP):
                junctions.add(begin.GetIdx())
                junctions.add(end.GetIdx())
    return junctions


def _cip_codes(mol: Chem.Mol) -> Set[int]:
    """The atom indices carrying a CIP code (post stereo assignment)."""
    return {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.HasProp("_CIPCode")
    }


def _pre_seed_state(
    pocket_pos: np.ndarray,
    pocket_z: np.ndarray,
    pocket_charge: np.ndarray,
) -> Data:
    """The pocket-only pre-seed state (the #39 candidate state)."""
    data = Data(
        pocket_pos=torch.as_tensor(pocket_pos, dtype=torch.float32),
        pocket_z=torch.as_tensor(pocket_z, dtype=torch.int64),
        pocket_charge=torch.as_tensor(pocket_charge, dtype=torch.float32),
        ligand_pos=torch.zeros(0, 3, dtype=torch.float32),
        ligand_z=torch.zeros(0, dtype=torch.int64),
        ligand_charge=torch.zeros(0, dtype=torch.float32),
        handle_features=torch.zeros(1, HANDLE_FEATURE_DIM, dtype=torch.float32),
        handle_pos=torch.zeros(1, 3, dtype=torch.float32),
        global_features=torch.zeros(1, 4, dtype=torch.float32),
        stop_mask=torch.zeros(1, dtype=torch.float32),
    )
    return stamp_schema_version(data)


def _growth_state(
    env: AssemblyEnvironment,
    pocket_pos: np.ndarray,
    pocket_z: np.ndarray,
    pocket_charge: np.ndarray,
    frame_origin: np.ndarray,
) -> Tuple[Data, List[Tuple[int, str]]]:
    """One decision state + its active handle pool (D-V2-041 / D-V2-031).

    The pool rows are the molecule-side growth handles of the current
    intermediate intersected with the environment's active capacity
    bookkeeping.  The primary-handle policy fields carry zeros (the
    undecided-handle convention; the backbone does not consume them).
    """
    mol = env.molecule
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    ligand_pos = positions - frame_origin
    pool = grow_handle_pool(mol, junction_atoms=sorted(_junction_atoms_of(mol)))
    active = {
        int(h["handle_atom_idx"]): int(h["remaining_capacity"])
        for h in env.active_handles()
    }
    rows = [(atom, cls) for atom, cls in pool if active.get(atom, 0) >= 1]
    mw = float(rdMolDescriptors.CalcExactMolWt(mol))
    heavy = float(mol.GetNumAtoms())
    v_lig = _vdw_volume(np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()]))
    v_poc = _vdw_volume(pocket_z) if pocket_z.size else max(v_lig, 1.0)
    data = Data(
        pocket_pos=torch.as_tensor(pocket_pos - frame_origin, dtype=torch.float32),
        pocket_z=torch.as_tensor(pocket_z, dtype=torch.int64),
        pocket_charge=torch.as_tensor(pocket_charge, dtype=torch.float32),
        ligand_pos=torch.as_tensor(ligand_pos, dtype=torch.float32),
        ligand_z=torch.as_tensor(
            [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.int64
        ),
        ligand_charge=torch.as_tensor(
            _ligand_charges(mol), dtype=torch.float32
        ),
        handle_features=torch.zeros(1, HANDLE_FEATURE_DIM, dtype=torch.float32),
        handle_pos=torch.zeros(1, 3, dtype=torch.float32),
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
    return stamp_schema_version(data), rows


def _ligand_charges(mol: Chem.Mol) -> np.ndarray:
    """Deterministic Gasteiger partial charges (the supervised convention)."""
    probe = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(probe)
    charges = np.asarray(
        [
            (
                float(a.GetDoubleProp("_GasteigerCharge"))
                if a.HasProp("_GasteigerCharge")
                and math.isfinite(a.GetDoubleProp("_GasteigerCharge"))
                else 0.0
            )
            for a in probe.GetAtoms()
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(charges)):
        raise RolloutError("non-finite ligand charges (fail-closed)")
    return charges


# ---------------------------------------------------------------------------
# The parity preview (clone the env's step state; one source of truth)
# ---------------------------------------------------------------------------


def _clone_step_state(env: AssemblyEnvironment) -> AssemblyEnvironment:
    """A shallow clone of an environment's *step state* for previews.

    The heavy immutable members (the block catalog, the seed pool, the
    pocket, the reaction registry) are shared by reference; only the
    per-trajectory mutable state (molecule, junction records, active
    handles, trajectory log, stats) is deep-copied.  The clone exists so
    the parity-required preview steps the **environment's own assembly
    code** instead of a reimplementation (single source of truth); it is
    never returned to the caller and its trajectory is discarded.
    """
    clone = object.__new__(AssemblyEnvironment)
    deep_keys = {"_mol", "_junctions", "_active_handles", "trajectory", "stats"}
    for key, value in env.__dict__.items():
        if key in deep_keys:
            setattr(clone, key, copy.deepcopy(value))
        else:
            setattr(clone, key, value)
    return clone


def _preview_parity_required(
    env: AssemblyEnvironment, action: AssemblyAction, handle_atom: int
) -> bool:
    """Whether the junction of ``action`` creates a stereocentre (#24 rule).

    Steps a cloned environment state with ``chiral=None`` (the 'R'
    geometry branch — the branch choice flips the realised CIP code, never
    the *existence* of the stereocentre) and applies the supervised rule:
    required iff the junction atom was not a stereocentre before and is
    one in the product.  The torsion bin does not enter the junction
    atom's stereo sense (the junction-bond direction is fixed by the
    hybridization geometry; only the distal block rotates about it).
    """
    before = _cip_codes(env.molecule)
    clone = _clone_step_state(env)
    try:
        clone.step(action)
    except AssemblyError as error:
        raise RolloutError(
            f"the preview step rejected the action {action!r}: {error} "
            "(mask drift — the policy sampled an illegal action)"
        ) from error
    after = _cip_codes(clone.molecule)
    # the junction atom index is stable across the step for the growing
    # molecule side (leaving groups of the *molecule-side* handle were
    # already absent from the intermediate; the env validates this)
    return handle_atom not in before and handle_atom in after


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclass
class _RolloutEnvState:
    """The driver's per-environment bookkeeping during one rollout."""

    env: AssemblyEnvironment
    pocket_pos: np.ndarray
    pocket_z: np.ndarray
    pocket_charge: np.ndarray
    frame_origin: np.ndarray
    steps: List[Any] = field(default_factory=list)
    pocket_latent: Optional[torch.Tensor] = None
    seed_latent: Optional[torch.Tensor] = None
    terminal_smiles: str = ""
    reward: Optional[float] = None
    done: bool = False


class PolicyRolloutDriver:
    """Drives batched on-policy rollouts for Trajectory Balance (#29).

    Construction requires the same catalog surfaces as the Stage-1 trainer
    (the immutable #12 catalog's per-row block handle classes and seed
    eligibility — the legality/eligibility masks) plus the row→block-id
    mapping (``catalog_block_ids``) so sampled catalog rows execute as
    environment actions.

    ``rollout(envs)`` resets each environment (they must carry a
    **pocket**, capped at ``max_pocket_atoms`` — rollout conditioning is
    pocket-conditioned by definition, D-V2-041) and drives the batch in
    lockstep: the seed/anchor stage once, then the growth loop
    (STOP law -> handle law -> family -> reaction -> synthon -> parity ->
    torsion, each sampled from the policy's own normalized laws with the
    driver's ``torch.Generator``), until every environment reaches STOP.
    Terminal molecules are finalized through the environment's Level-2
    oracle (sub-bin refinement + scaffold-constrained MMFF94) and rewarded
    with the stationary :class:`~syntree.engine.rl.ThreeDReward` of the
    environment's pocket (``reward_factory`` overrides; the engine is
    cached per pocket object).

    Determinism: identical (policy weights, generator state, envs) ⇒
    bit-identical trajectories — every categorical draw goes through the
    single generator and the environment is deterministic by construction.
    """

    def __init__(
        self,
        policy: SynTreePolicy,
        *,
        catalog_block_ids: Sequence[str],
        catalog_block_handle_classes: torch.Tensor,
        catalog_seed_rows: torch.Tensor,
        available_block_classes: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        max_pocket_atoms: int = MAX_ROLLOUT_POCKET_ATOMS,
        reward_factory: Optional[
            Callable[[AssemblyEnvironment], ThreeDReward]
        ] = None,
        finalize_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(catalog_block_ids) != int(catalog_block_handle_classes.numel()):
            raise RolloutError(
                f"catalog_block_ids ({len(catalog_block_ids)}) disagrees with "
                f"the class tensor ({catalog_block_handle_classes.numel()})"
            )
        if int(catalog_seed_rows.numel()) != len(catalog_block_ids):
            raise RolloutError(
                f"catalog_seed_rows ({catalog_seed_rows.numel()}) disagrees "
                f"with the catalog size ({len(catalog_block_ids)})"
            )
        if len(catalog_block_ids) != policy.catalog_size:
            raise RolloutError(
                f"the catalog surfaces ({len(catalog_block_ids)} rows) do not "
                f"match the policy's catalog_size ({policy.catalog_size}) — "
                "the tied embedding table and the masks must agree"
            )
        self.policy = policy
        self.device = torch.device(device) if device is not None else next(
            policy.parameters()
        ).device
        self.catalog_block_ids = list(catalog_block_ids)
        self.catalog_block_handle_classes = (
            catalog_block_handle_classes.to(torch.int64).to(self.device)
        )
        # the mask convention is True = ILLEGAL (apply_logit_mask)
        self.seed_mask = (~catalog_seed_rows.to(torch.bool)).to(self.device)
        self.available_block_classes = list(available_block_classes) \
            if available_block_classes is not None else None
        self.generator = generator if generator is not None else torch.Generator()
        self.max_pocket_atoms = int(max_pocket_atoms)
        self._reward_factory = reward_factory
        self._reward_cache: Dict[int, ThreeDReward] = {}
        self._finalize_kwargs = dict(finalize_kwargs or {})

    # -- sampling helpers ------------------------------------------------------

    def _sample(self, probs: torch.Tensor) -> int:
        """One categorical draw from a 1-D law (deterministic generator)."""
        index = torch.multinomial(probs.detach().to("cpu"), 1, generator=self.generator)
        return int(index.item())

    def _sample_batch(self, probs: torch.Tensor) -> List[int]:
        """Per-row categorical draws from a ``[B, K]`` law."""
        draws = torch.multinomial(
            probs.detach().to("cpu"), 1, generator=self.generator
        )
        return [int(v) for v in draws.flatten().tolist()]

    # -- rewards -----------------------------------------------------------------

    def _reward_engine(self, env: AssemblyEnvironment) -> ThreeDReward:
        if self._reward_factory is not None:
            engine = self._reward_factory(env)
        else:
            key = id(env.pocket)
            engine = self._reward_cache.get(key)
            if engine is None:
                engine = ThreeDReward(env.pocket)
                self._reward_cache[key] = engine
        return engine

    # -- the rollout ---------------------------------------------------------------

    def rollout(self, envs: Sequence[AssemblyEnvironment]) -> List[TBTrajectory]:
        if not envs:
            raise RolloutError("the rollout batch is empty (fail-closed)")
        states: List[_RolloutEnvState] = []
        for env in envs:
            pocket = env.pocket
            if pocket is None:
                raise RolloutError(
                    "rollout environments must be pocket-conditioned (the "
                    "Stage-2 MDP is defined on 3D pockets — D-V2-041); a "
                    "pocket-free environment is a unit-test fixture only"
                )
            if pocket.num_atoms > self.max_pocket_atoms:
                raise RolloutError(
                    f"the environment pocket has {pocket.num_atoms} atoms > "
                    f"the {self.max_pocket_atoms} cap — construct rollout "
                    "environments with a capped pocket (D-V2-041)"
                )
            env.reset()
            # the raw (capped) pocket arrays in the environment's own frame;
            # each stage applies its own centering (D-V2-041)
            pos = np.asarray(pocket.positions, dtype=np.float64)
            z = np.asarray(pocket.atomic_numbers, dtype=np.int64)
            q = np.asarray(pocket.charges, dtype=np.float64)
            centroid = pos.mean(axis=0)
            if pos.shape[0] > self.max_pocket_atoms:
                distances = np.linalg.norm(pos - centroid, axis=1)
                keep = np.argsort(distances, kind="stable")[: self.max_pocket_atoms]
                pos, z, q = pos[keep], z[keep], q[keep]
            states.append(
                _RolloutEnvState(
                    env=env,
                    pocket_pos=pos,
                    pocket_z=z,
                    pocket_charge=q,
                    frame_origin=centroid,
                )
            )
        was_training = self.policy.training
        self.policy.eval()
        try:
            self._seed_stage(states)
            self._growth_loop(states)
            self._terminate(states)
        finally:
            if was_training:
                self.policy.train()
        trajectories = []
        for state in states:
            if state.reward is None:
                raise RolloutError(
                    "a rollout finished without a reward (internal error)"
                )
            trajectories.append(
                TBTrajectory(
                    steps=list(state.steps),
                    reward=state.reward,
                    pocket_latent=state.pocket_latent,
                    seed_latent=state.seed_latent,
                    terminal_smiles=state.terminal_smiles,
                )
            )
        return trajectories

    # -- stage 0: the seed/anchor action (#39) --------------------------------------

    def _seed_stage(self, states: List[_RolloutEnvState]) -> None:
        batch = collate_state_batch(
            [
                _pre_seed_state(
                    s.pocket_pos - s.frame_origin, s.pocket_z, s.pocket_charge
                )
                for s in states
            ]
        ).to(self.device)
        backbone = self.policy.backbone(batch)
        anchor_pos, anchor_batch, assignment, counts = self.policy.anchor_candidates(
            batch.pocket_pos,
            batch.pocket_z,
            batch.pocket_charge,
            batch.pocket_batch,
            batch.num_graphs,
        )
        del anchor_pos
        features = self.policy.anchor_features(backbone, assignment, anchor_batch)
        anchor_out = self.policy.seed_anchor_head.forward_anchor(
            backbone.pocket_context, features, anchor_batch, None, None
        )
        # sample the anchor row per graph (flat segments)
        offsets = torch.zeros_like(counts)
        offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
        anchor_rows: List[int] = []
        for g in range(batch.num_graphs):
            lo = int(offsets[g])
            hi = lo + int(counts[g])
            if hi <= lo:
                raise RolloutError(
                    "a rollout pocket produced zero anchor candidates "
                    "(fail-closed — the #39 menu must be non-empty)"
                )
            row = self._sample(anchor_out["probs"][lo:hi])
            anchor_rows.append(lo + row)
        selected_rows = torch.tensor(anchor_rows, dtype=torch.int64, device=self.device)
        anchor_log_prob = anchor_out["log_probs"].index_select(0, selected_rows)
        # the selected anchor's feature conditions the seed law (#39)
        selected_features = features.index_select(0, selected_rows)
        seed_out = self.policy.seed_anchor_head.forward_seed(
            backbone.pocket_context, selected_features, self.seed_mask.expand(
                batch.num_graphs, -1
            ), None
        )
        seed_rows = self._sample_batch(seed_out["probs"])
        selected_seeds = torch.tensor(seed_rows, dtype=torch.int64, device=self.device)
        seed_log_prob = seed_out["log_probs"].gather(
            1, selected_seeds.unsqueeze(1)
        ).squeeze(1)
        # the logZ conditioning latents: the pooled pocket context + the
        # tied catalog embedding of the selected seed (#29)
        seed_latents = self.policy.seed_anchor_head.catalog_embeddings(
            selected_seeds
        )
        for g, state in enumerate(states):
            local_anchor = anchor_rows[g] - int(offsets[g])
            block_id = self.catalog_block_ids[seed_rows[g]]
            menu = state.env.seed_menu()
            if block_id not in menu.seed_block_ids:
                raise RolloutError(
                    f"the sampled seed row {seed_rows[g]} maps to block "
                    f"{block_id!r} which is not in the environment's seed "
                    "pool (catalog/env mismatch — fail-closed)"
                )
            if local_anchor >= menu.num_anchors:
                raise RolloutError(
                    "the sampled anchor index exceeds the environment's "
                    "anchor menu (capped-pocket mismatch — construct the "
                    "env with the capped pocket, D-V2-041)"
                )
            record = state.env.place_seed(SeedAction(block_id, local_anchor))
            record.action_log_prob = float(
                (anchor_log_prob[g] + seed_log_prob[g]).detach()
            )
            state.steps.append(
                _tensor_step("seed", anchor_log_prob[g] + seed_log_prob[g], 0)
            )
            state.pocket_latent = backbone.pocket_context[g].detach()
            state.seed_latent = seed_latents[g].detach()
            seed_centroid = np.asarray(
                state.env.molecule.GetConformer().GetPositions(), dtype=np.float64
            ).mean(axis=0)
            state.frame_origin = seed_centroid

    # -- the growth loop -------------------------------------------------------------

    def _growth_loop(self, states: List[_RolloutEnvState]) -> None:
        active = [s for s in states if not s.done]
        while active:
            active = self._growth_step(active)

    def _growth_step(
        self, states: List[_RolloutEnvState]
    ) -> List[_RolloutEnvState]:
        # split by pool availability (the #22 head refuses empty pools)
        with_pool: List[_RolloutEnvState] = []
        forced: List[_RolloutEnvState] = []
        for state in states:
            mol = state.env.molecule
            pool = grow_handle_pool(mol, junction_atoms=sorted(_junction_atoms_of(mol)))
            capacity = {
                int(h["handle_atom_idx"]): int(h["remaining_capacity"])
                for h in state.env.active_handles()
            }
            rows = [(a, c) for a, c in pool if capacity.get(a, 0) >= 1]
            if rows:
                with_pool.append(state)
            else:
                forced.append(state)
        # empty-pool terminals: STOP is structurally forced (D-V2-033) —
        # log P(STOP) = 1 contributes exactly 0.0
        for state in forced:
            record = state.env.stop()
            state.steps.append(
                _tensor_step("stop", torch.zeros((), device=self.device), record.num_leaves)
            )
            state.done = True
        if not with_pool:
            return []
        # build the decision batch
        datas = []
        pools: List[List[Tuple[int, str]]] = []
        for state in with_pool:
            data, rows = _growth_state(
                state.env,
                state.pocket_pos,
                state.pocket_z,
                state.pocket_charge,
                state.frame_origin,
            )
            datas.append(data)
            pools.append(rows)
        batch = collate_state_batch(datas).to(self.device)
        descriptors: List[torch.Tensor] = []
        handle_pos: List[torch.Tensor] = []
        graph_of: List[int] = []
        atom_of: List[int] = []
        ligand_offsets = torch.zeros(batch.num_graphs, dtype=torch.int64)
        running = 0
        ligand_counts = torch.bincount(
            batch.ligand_batch, minlength=batch.num_graphs
        )
        for g in range(batch.num_graphs):
            ligand_offsets[g] = running
            running += int(ligand_counts[g])
        for g, (state, rows) in enumerate(zip(with_pool, pools)):
            mol = state.env.molecule
            positions = np.asarray(
                mol.GetConformer().GetPositions(), dtype=np.float64
            ) - state.frame_origin
            for atom, _cls in rows:
                descriptors.append(
                    torch.as_tensor(
                        handle_descriptor_64(mol, atom), dtype=torch.float32
                    )
                )
                handle_pos.append(
                    torch.as_tensor(positions[atom], dtype=torch.float32)
                )
                graph_of.append(g)
                atom_of.append(int(ligand_offsets[g]) + atom)
        pool_descriptors = torch.stack(descriptors).to(self.device)
        pool_pos = torch.stack(handle_pos).to(self.device)
        pool_graph = torch.tensor(graph_of, dtype=torch.int64, device=self.device)
        pool_atom = torch.tensor(atom_of, dtype=torch.int64, device=self.device)
        state_out = self.policy.forward_state(
            batch, pool_descriptors, pool_pos, pool_graph, pool_atom
        )
        # the legality masks (the supervised construction)
        family_mask, reaction_mask = build_continuation_masks(
            [[cls for _a, cls in rows] for rows in pools],
            self.available_block_classes,
        )
        at_capacity = torch.tensor(
            [
                self._at_capacity(state)
                for state in with_pool
            ],
            dtype=torch.bool,
        )
        force_stop = build_force_stop(reaction_mask, at_capacity=at_capacity)
        stop_out = self.policy.forward_stop(
            state_out.context, force_stop.to(self.device)
        )
        stop_draws = self._sample_batch(stop_out.probs)
        continuing: List[int] = []
        for g, draw in enumerate(stop_draws):
            state = with_pool[g]
            if draw == 1:
                record = state.env.stop()
                record.action_log_prob = float(stop_out.stop_log_prob[g].detach())
                state.steps.append(
                    _tensor_step("stop", stop_out.stop_log_prob[g], record.num_leaves)
                )
                state.done = True
            else:
                if bool(force_stop[g]):
                    raise RolloutError(
                        "the policy sampled CONTINUE on a forced-stop row "
                        "(the mask must make P(CONTINUE)=0 — internal error)"
                    )
                continuing.append(g)
        if not continuing:
            return []
        # subset to the continuing rows (STOP rows never touch the
        # continuation heads — the #24 isolation contract)
        ctx = state_out.context.index_select(
            0, torch.tensor(continuing, dtype=torch.int64, device=self.device)
        )
        handle_out = state_out.handle
        # -- handle law (per-graph segments over the flat pool) ---------------
        handle_log_probs: List[torch.Tensor] = []
        selected_handles: List[int] = []
        counts = [len(rows) for rows in pools]
        offsets = np.zeros(len(pools), dtype=np.int64)
        offsets[1:] = np.cumsum(counts)[:-1]
        for g in continuing:
            lo, hi = int(offsets[g]), int(offsets[g]) + counts[g]
            row = self._sample(handle_out.probs[lo:hi])
            handle_log_probs.append(handle_out.log_probs[lo + row])
            selected_handles.append(pools[g][row][0])
        # -- family -> reaction -> synthon -> parity -> torsion -----------------
        fam_mask, rxt_mask = build_continuation_masks(
            [[cls for _a, cls in pools[g]] for g in continuing],
            self.available_block_classes,
        )
        family_out = self.policy.heads.forward_family(ctx, fam_mask.to(self.device))
        family_draws = self._sample_batch(family_out.probs)
        family_rows = torch.tensor(family_draws, dtype=torch.int64, device=self.device)
        family_log_prob = family_out.log_probs.gather(1, family_rows.unsqueeze(1)).squeeze(1)
        # the reaction head restricts its own law to the selected family's
        # legal support (the hierarchical conditional, #24) — sample it
        # directly; an all-masked family fails closed inside the head
        reaction_out = self.policy.heads.forward_reaction(
            ctx, family_rows, rxt_mask.to(self.device), None
        )
        reaction_draws = self._sample_batch(reaction_out.probs)
        reaction_rows = torch.tensor(reaction_draws, dtype=torch.int64, device=self.device)
        reaction_log_prob = reaction_out.log_probs.gather(
            1, reaction_rows.unsqueeze(1)
        ).squeeze(1)
        synthon_out = self.policy.heads.forward_synthon(
            ctx,
            reaction_rows,
            None,
            self.catalog_block_handle_classes,
        )
        synthon_draws = self._sample_batch(synthon_out.probs)
        synthon_rows = torch.tensor(synthon_draws, dtype=torch.int64, device=self.device)
        synthon_log_prob = synthon_out.log_probs.gather(
            1, synthon_rows.unsqueeze(1)
        ).squeeze(1)
        # the tied table gather is the #24 conditioning surface for the
        # parity/torsion heads
        table = self.policy.heads.synthon_head.catalog_embeddings
        embeddings = table(synthon_rows)
        # -- parity (pre-step required determination via the cloned state) -----
        parity_log_probs: List[Optional[torch.Tensor]] = []
        chirals: List[Optional[str]] = []
        for i, g in enumerate(continuing):
            state = with_pool[g]
            reaction_id = POLICY_REACTION_IDS[reaction_draws[i]]
            block_id = self.catalog_block_ids[synthon_draws[i]]
            handle_atom = selected_handles[i]
            preview_action = AssemblyAction(
                handle_atom, reaction_id, block_id, None, 18
            )
            required = _preview_parity_required(state.env, preview_action, handle_atom)
            if required:
                parity_out = self.policy.heads.forward_parity(
                    ctx[i : i + 1],
                    embeddings[i : i + 1],
                    torch.tensor([True], device=self.device),
                    None,
                )
                draw = self._sample(parity_out.probs[0])
                chirals.append(("R", "S")[draw])
                parity_log_probs.append(parity_out.log_probs[0, draw])
            else:
                chirals.append(None)
                parity_log_probs.append(None)
        # -- torsion (rotatable junctions only — the #24 contract) -------------
        torsion_log_probs: List[Optional[torch.Tensor]] = []
        bins: List[int] = []
        for i, g in enumerate(continuing):
            reaction_id = POLICY_REACTION_IDS[reaction_draws[i]]
            lock = DEFAULT_REACTION_REGISTRY[reaction_id].resonance_lock
            if lock is None:
                torsion_out = self.policy.heads.forward_torsion(
                    ctx[i : i + 1],
                    embeddings[i : i + 1],
                    torch.tensor([True], device=self.device),
                    None,
                )
                draw = self._sample(torsion_out.probs[0])
                bins.append(draw + 1)  # 1-based bins
                torsion_log_probs.append(torsion_out.log_probs[0, draw])
            else:
                bins.append(18)  # ignored by the env (resonance lock)
                torsion_log_probs.append(None)
        # -- commit the actions ---------------------------------------------------
        next_active: List[_RolloutEnvState] = []
        for i, g in enumerate(continuing):
            state = with_pool[g]
            reaction_id = POLICY_REACTION_IDS[reaction_draws[i]]
            block_id = self.catalog_block_ids[synthon_draws[i]]
            action = AssemblyAction(
                selected_handles[i],
                reaction_id,
                block_id,
                chirals[i],
                bins[i],
            )
            try:
                record = state.env.step(action)
            except AssemblyError as error:
                raise RolloutError(
                    f"the environment rejected a sampled action {action!r}: "
                    f"{error} (mask drift — fail-closed)"
                ) from error
            # post-hoc parity verification (the preview's fail-closed check)
            if chirals[i] is not None:
                if selected_handles[i] not in _cip_codes(state.env.molecule):
                    raise RolloutError(
                        "parity preview said required but the realized "
                        "product has no CIP at the junction atom (preview "
                        "bug — fail-closed, pin with a regression test)"
                    )
            elif selected_handles[i] in _cip_codes(state.env.molecule):
                raise RolloutError(
                    "parity preview said not-required but the realized "
                    "product carries CIP at the junction atom (preview "
                    "bug — fail-closed, pin with a regression test)"
                )
            total = (
                stop_out.continue_log_prob[g]
                + handle_log_probs[i]
                + family_log_prob[i]
                + reaction_log_prob[i]
                + synthon_log_prob[i]
                + (parity_log_probs[i] if parity_log_probs[i] is not None else 0.0)
                + (torsion_log_probs[i] if torsion_log_probs[i] is not None else 0.0)
            )
            record.action_log_prob = float(total.detach())
            state.steps.append(_tensor_step("add", total, record.num_leaves))
            next_active.append(state)
        return next_active

    def _at_capacity(self, state: _RolloutEnvState) -> bool:
        """The environment's block budget is exhausted (only STOP remains)."""
        return state.env.num_added_blocks >= state.env.max_blocks

    # -- terminal: finalize + reward ------------------------------------------------

    def _terminate(self, states: List[_RolloutEnvState]) -> None:
        for state in states:
            if not state.env.is_terminal:
                raise RolloutError(
                    "a rollout state reached _terminate without STOP "
                    "(internal error — the growth loop must close every env)"
                )
            state.terminal_smiles = state.env.trajectory[-1].smiles
            relaxed, _energy = state.env.finalize_terminal_molecule(
                **self._finalize_kwargs
            )
            engine = self._reward_engine(state.env)
            reward = float(engine.compute(relaxed))
            if not math.isfinite(reward) or reward < engine.reward_floor:
                raise RolloutError(
                    f"the terminal reward {reward!r} is non-finite or below "
                    "the floor (poisoned terminal — fail-closed)"
                )
            state.reward = reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tensor_step(action_type: str, log_prob: Any, num_leaves: int) -> TrajectoryStep:
    """A :class:`TrajectoryStep` carrying a tensor (or float) log-prob."""
    return TrajectoryStep(
        action_type=action_type,
        action_log_prob=log_prob,
        num_leaves=num_leaves,
    )
