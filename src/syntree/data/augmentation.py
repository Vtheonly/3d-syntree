"""Coordinate jitter & exposure-bias augmentation for training batches.

Clean-room V2 implementation of issue #20 DATA-05 (constitution #45 §12.1
"Implement the #20 coordinate/dihedral augmentation only on the training
split, preserving connectivity and stereochemistry").

**The exposure-bias problem (the #20 motivation).**  Stage-1 behavioural
cloning trains on pristine crystallographic intermediates — every state
the policy ever sees during training has perfect crystal geometry.  At
rollout time the policy consumes its *own* assembly outputs: ETKDG
distance-geometry placements and categorical bin-centre torsions whose
coordinates carry spatial error, and the error compounds with every step
(covariate shift / exposure bias).  The policy must therefore be trained
on *imperfect* intermediates:

1. **Gaussian coordinate jitter** — iid ``N(0, sigma^2)`` noise with
   ``sigma = 0.15`` A applied to the intermediate ligand coordinates
   ``X_L`` **only**.  The protein pocket coordinates ``X_P`` are part of
   the fixed conditioning context (constitution #45 §12.4) and stay
   bit-identical (pristine).
2. **Bounded non-junction dihedral perturbation** — every *non-junction
   rotatable* dihedral is rotated by a random amount within
   ``[-5.0, +5.0]`` degrees.  Junction dihedrals — the bonds formed by
   assembly steps — are the supervised torsion-action targets (#24's
   36-bin space; #18's ``dihedral_rad`` labels) and are **never
   deliberately rotated**.

**Train-split only (fail-closed).**  Validation and test batches stay
completely pristine (#45 §12.1: "Validation and test data must not
receive training augmentation").  The augmenter refuses to run on any
split other than ``"train"``.

**What is preserved (the acceptance contract).**

* The molecular graph is never touched: the augmentation is a pure
  coordinate-space transform, so connectivity (atom set, bonds, charges)
  is preserved by construction and verified by test against the
  canonical SMILES.
* Chiral stereocenters are not inverted: at ``sigma = 0.15`` A the noise
  is an order of magnitude below bond lengths (~1.5 A), and the dihedral
  stage is a *rigid* rotation of one side of a rotatable bond (bond
  lengths and valence angles preserved bit-for-bit).  Tests verify CIP
  labels assigned from 3D (``AssignStereochemistryFrom3D``) survive.
* Every perturbed dihedral's total change is within the +/-5 deg bound
  *measured against the pristine geometry* — the sequential rotations
  are coupled (rotating one bond moves the reference atoms of bonds in
  its distal subtree), so the implementation uses Gauss-Seidel passes
  that set every bond exactly to its pristine-plus-delta target and a
  final fail-closed verification that no bond's realised change exceeded
  the bound.

Determinism: all randomness flows through one ``numpy.random.Generator``
supplied by the caller (the training loader derives it per sample/epoch
from :mod:`syntree.utils.seeding`), so augmented batches are exactly
reproducible from the seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

__all__ = [
    "COORD_JITTER_SIGMA_ANGSTROM",
    "DIHEDRAL_PERTURB_MAX_DEG",
    "DIHEDRAL_PERTURB_MAX_RAD",
    "AUGMENTATION_SPLIT",
    "AugmentationError",
    "jitter_ligand_coordinates",
    "nonjunction_rotatable_bonds",
    "dihedral_change_rad",
    "perturb_nonjunction_dihedrals",
    "ExposureBiasAugmenter",
    "augment_single_state",
]

#: Issue-pinned Gaussian jitter standard deviation (Angstrom).
COORD_JITTER_SIGMA_ANGSTROM = 0.15

#: Issue-pinned maximum dihedral perturbation (degrees, inclusive).
DIHEDRAL_PERTURB_MAX_DEG = 5.0

#: The same bound in radians (the internal unit).
DIHEDRAL_PERTURB_MAX_RAD = math.radians(DIHEDRAL_PERTURB_MAX_DEG)

#: The only split that may receive augmentation (fail-closed otherwise).
AUGMENTATION_SPLIT = "train"


class AugmentationError(ValueError):
    """Fail-closed augmentation error (never a silently skipped sample)."""


# ---------------------------------------------------------------------------
# Elementary geometry helpers (numpy, float64)
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap to ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def dihedral_rad(p0, p1, p2, p3) -> float:
    """The signed dihedral ``(p0, p1, p2, p3)`` in ``[-pi, pi)``.

    The standard stable ``atan2`` formulation (no ``acos`` — no precision
    loss near 0/pi and no domain errors for degenerate inputs).
    """
    b0 = np.asarray(p0, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    b1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    b2 = np.asarray(p3, dtype=np.float64) - np.asarray(p2, dtype=np.float64)
    b1n = b1 / max(np.linalg.norm(b1), 1e-300)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return math.atan2(y, x)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise AugmentationError(
            "cannot normalise a zero-length bond axis (degenerate geometry)"
        )
    return vec / norm


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix for a unit ``axis`` and ``angle``."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _reference_neighbor(mol: Chem.Mol, atom_idx: int, excluded: int) -> Optional[int]:
    """The deterministic dihedral reference neighbour.

    The environment's rule (``syntree/engine/environment.py``): highest
    atomic number wins among heavy neighbours other than ``excluded``;
    ties break on the lower index.  ``None`` when no other heavy
    neighbour exists (the dihedral is undefined — such bonds are skipped).
    """
    best: Optional[int] = None
    best_key: Optional[Tuple[int, int]] = None
    for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx == excluded or neighbor.GetAtomicNum() == 1:
            continue
        key = (neighbor.GetAtomicNum(), -idx)
        if best_key is None or key > best_key:
            best_key, best = key, idx
    return best


def _component_beyond(mol: Chem.Mol, start: int, blocked: int) -> List[int]:
    """Atoms reachable from ``start`` without traversing ``blocked``."""
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in mol.GetAtomWithIdx(node).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx != blocked and idx not in seen:
                seen.add(idx)
                stack.append(idx)
    return sorted(seen)


# ---------------------------------------------------------------------------
# The rotatable-bond definition (documented, self-contained)
# ---------------------------------------------------------------------------

#: Double-bond partners that make an adjacent single bond planar
#: (resonance-locked): amide/ester/thioamide/sulfonamide/imidoyl systems.
_RESONANCE_ACCEPTORS = (7, 8, 16)  # N, O, S
_PLANAR_HETEROATOMS = (7, 8, 16)  # N, O, S


def _is_resonance_locked(mol: Chem.Mol, a: int, b: int) -> bool:
    """Whether the single bond ``a-b`` is locked planar by resonance.

    True iff one endpoint is N/O/S single-bonded to a carbon that is
    double/triple-bonded to another N/O/S — the amide (N-C=O), ester
    (O-C=O), thioamide (N-C=S) and imidoyl (N-C=N) conjugation that the
    environment locks via ``JunctionReaction.resonance_lock``.  Those
    bonds are chemically *not* freely rotatable and are excluded.
    """
    for x, y in ((a, b), (b, a)):
        atom = mol.GetAtomWithIdx(x)
        if atom.GetAtomicNum() not in _PLANAR_HETEROATOMS:
            continue
        for partner in atom.GetNeighbors():
            if partner.GetIdx() != y:
                continue
            for second in partner.GetNeighbors():
                if second.GetIdx() == x:
                    continue
                bond = mol.GetBondBetweenAtoms(partner.GetIdx(), second.GetIdx())
                if bond is None:
                    continue
                if (
                    bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
                    and second.GetAtomicNum() in _RESONANCE_ACCEPTORS
                ):
                    return True
    return False


def nonjunction_rotatable_bonds(
    mol: Chem.Mol,
    junction_bonds: Sequence[Tuple[int, int]] = (),
) -> List[Tuple[int, int]]:
    """The perturbable dihedral bonds of ``mol`` (deterministic order).

    A bond is *perturbable* iff

    1. it is a single, non-ring bond,
    2. both endpoints have at least one other heavy neighbour (the
       dihedral's reference atoms exist on both sides),
    3. it is not resonance-locked (amide/ester/imidoyl-type
       heteroatom-to-sp2-carbon bonds — see :func:`_is_resonance_locked`),
       and
    4. it is **not** a junction bond — a bond formed by an assembly step
       (passed by the caller as atom-index pairs; matched in both
       orientations).  Junction dihedrals are the supervised torsion
       actions and are never deliberately rotated.

    Returns the bonds as ``(a, b)`` tuples in ascending atom order (the
    molecule-side atom first is irrelevant — the caller sees a set).
    """
    junction = {tuple(sorted((int(a), int(b)))) for a, b in junction_bonds}
    bonds: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        key = tuple(sorted((a, b)))
        if key in junction:
            continue
        if _reference_neighbor(mol, a, b) is None or _reference_neighbor(mol, b, a) is None:
            continue
        if _is_resonance_locked(mol, a, b):
            continue
        bonds.append(key)
    return sorted(bonds)


# ---------------------------------------------------------------------------
# Stage 1: Gaussian coordinate jitter (ligand only)
# ---------------------------------------------------------------------------

def jitter_ligand_coordinates(
    ligand_positions: np.ndarray,
    sigma: float = COORD_JITTER_SIGMA_ANGSTROM,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add iid Gaussian noise ``N(0, sigma^2)`` to ligand coordinates.

    Parameters:
        ligand_positions: ``[N, 3]`` array (Angstrom) — the intermediate
            ligand coordinates ``X_L`` only; the pocket is never passed
            here and stays pristine by construction.
        sigma: the noise standard deviation (issue-pinned 0.15 A).
        rng: the caller's deterministic stream.

    Returns a **new** ``[N, 3]`` float64 array; the input is never
    mutated.  Fails closed on malformed shapes, non-finite inputs and
    non-positive ``sigma``.
    """
    positions = np.asarray(ligand_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise AugmentationError(
            f"ligand positions must be [N, 3], got {positions.shape}"
        )
    if not np.all(np.isfinite(positions)):
        raise AugmentationError("ligand positions contain non-finite values")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise AugmentationError(f"sigma must be finite and > 0, got {sigma!r}")
    if not isinstance(rng, np.random.Generator):
        raise AugmentationError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    if positions.shape[0] == 0:
        return positions.copy()
    noise = rng.normal(0.0, sigma, size=positions.shape)
    return positions + noise


# ---------------------------------------------------------------------------
# Stage 2: bounded non-junction dihedral perturbation
# ---------------------------------------------------------------------------

def dihedral_change_rad(
    pristine: np.ndarray,
    perturbed: np.ndarray,
    mol: Chem.Mol,
    bond: Tuple[int, int],
) -> float:
    """The wrapped change of one bond's dihedral between two geometries.

    Uses the same deterministic reference-neighbour rule as the
    perturbation itself, so the measurement the tests assert on is the
    measurement the transform realises.
    """
    a, b = bond
    a0 = _reference_neighbor(mol, a, b)
    b0 = _reference_neighbor(mol, b, a)
    if a0 is None or b0 is None:
        raise AugmentationError(
            f"bond {bond} has no dihedral reference neighbour on both sides"
        )
    before = dihedral_rad(pristine[a0], pristine[a], pristine[b], pristine[b0])
    after = dihedral_rad(perturbed[a0], perturbed[a], perturbed[b], perturbed[b0])
    return _wrap_angle(after - before)


def perturb_nonjunction_dihedrals(
    positions: np.ndarray,
    mol: Chem.Mol,
    junction_bonds: Sequence[Tuple[int, int]] = (),
    *,
    rng: np.random.Generator,
    max_delta_rad: float = DIHEDRAL_PERTURB_MAX_RAD,
    max_passes: int = 8,
    tolerance_rad: float = 1e-6,
) -> np.ndarray:
    """Rotate every non-junction rotatable dihedral within ``+/- max_delta``.

    Draw one delta ``U(-max_delta, +max_delta)`` per perturbable bond and
    set every bond's dihedral to its **pristine value plus its delta**
    (Gauss-Seidel: bonds are visited periphery-first — ascending distal
    subtree size — because inner rotations move outer reference atoms,
    and the passes iterate until every bond sits exactly at its target).
    Rigid rotations about bond axes: covalent bond lengths and valence
    angles are preserved bit-for-bit.

    Fails closed (:class:`AugmentationError`) if the iteration does not
    converge within ``max_passes`` or if the final verification finds any
    bond whose realised change exceeds the bound — the ``+/-5 deg``
    contract is enforced, never assumed.
    """
    pos = np.asarray(positions, dtype=np.float64).copy()
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise AugmentationError(f"positions must be [N, 3], got {pos.shape}")
    if pos.shape[0] != mol.GetNumAtoms():
        raise AugmentationError(
            f"geometry has {pos.shape[0]} atoms but the molecule has "
            f"{mol.GetNumAtoms()} — the connectivity and coordinates must "
            "describe the same intermediate"
        )
    if not isinstance(rng, np.random.Generator):
        raise AugmentationError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    max_delta = float(max_delta_rad)
    if not math.isfinite(max_delta) or max_delta <= 0.0:
        raise AugmentationError(f"max_delta_rad must be finite and > 0, got {max_delta!r}")

    bonds = nonjunction_rotatable_bonds(mol, junction_bonds)
    if not bonds:
        return pos

    # Reference atoms and distal subtrees (deterministic; the distal side
    # of bond (a, b) is the component beyond b, rotated about the axis).
    refs: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    distal: Dict[Tuple[int, int], np.ndarray] = {}
    for a, b in bonds:
        a0 = _reference_neighbor(mol, a, b)
        b0 = _reference_neighbor(mol, b, a)
        if a0 is None or b0 is None:  # pragma: no cover - filtered above
            raise AugmentationError(f"bond {(a, b)} lost its reference neighbour")
        refs[(a, b)] = (a0, a, b, b0)
        distal[(a, b)] = np.array(_component_beyond(mol, b, a), dtype=np.int64)

    pristine = {
        bond: dihedral_rad(pos[r[0]], pos[r[1]], pos[r[2]], pos[r[3]])
        for bond, r in refs.items()
    }
    deltas = {bond: float(rng.uniform(-max_delta, max_delta)) for bond in bonds}
    targets = {bond: _wrap_angle(pristine[bond] + deltas[bond]) for bond in bonds}

    # Periphery-first: rotating an inner bond moves the reference atoms of
    # bonds inside its distal subtree, so outer (peripheral) bonds are set
    # first and inner bonds correct the residual in later passes.
    order = sorted(bonds, key=lambda bond: (distal[bond].size, bond))

    def set_bond(bond: Tuple[int, int]) -> None:
        a0, a, b, b0 = refs[bond]
        current = dihedral_rad(pos[a0], pos[a], pos[b], pos[b0])
        delta = _wrap_angle(targets[bond] - current)
        if abs(delta) <= 1e-15:
            return
        axis = _unit(pos[b] - pos[a])
        rotation = _rotation_matrix(axis, delta)
        idx = distal[bond]
        pos[idx] = pos[a] + (pos[idx] - pos[a]) @ rotation.T

    for _ in range(max(1, int(max_passes))):
        for bond in order:
            set_bond(bond)
        residual = max(
            abs(_wrap_angle(
                dihedral_rad(pos[r[0]], pos[r[1]], pos[r[2]], pos[r[3]])
                - targets[bond]
            ))
            for bond, r in refs.items()
        )
        if residual <= tolerance_rad:
            break
    else:
        raise AugmentationError(
            f"dihedral targeting did not converge within {max_passes} passes "
            f"(residual {residual:.3e} rad) — refusing to emit a sample that "
            "violates the +/-5 deg bound"
        )

    # Fail-closed final verification: every realised change within bound.
    for bond in bonds:
        change = abs(dihedral_change_rad(np.asarray(positions, dtype=np.float64), pos, mol, bond))
        if change > max_delta + tolerance_rad:
            raise AugmentationError(
                f"bond {bond} changed by {math.degrees(change):.4f} deg, "
                f"exceeding the +-{math.degrees(max_delta):.1f} deg bound"
            )
    return pos


# ---------------------------------------------------------------------------
# Stereo-signature preservation (the measured flip finding)
# ---------------------------------------------------------------------------

def stereo_signature(mol: Chem.Mol, positions: np.ndarray) -> Tuple[Tuple[str, ...], str]:
    """The 3D stereochemistry signature of ``positions`` on ``mol``.

    ``(per-atom CIP labels, isomeric SMILES assigned from the 3D
    geometry)`` — the pair the preservation contract compares.  Both are
    computed by ``AssignStereochemistryFrom3D`` on a probe copy; the
    input molecule and array are never mutated.

    **Why this exists (measured, S-V2-14).**  Naive iid Gaussian jitter
    at the issue-pinned ``sigma = 0.15`` A **can invert a stereocenter's
    3D signature**: the chirality volume of a typical sp3 centre
    (~0.39 A^3 with ~1.5 A bonds) is only ~2.5x the noise-induced
    volume sigma, and on the E/Z fixture molecule
    ``C/C=C/C[C@H](N)C(=O)O`` plain jitter flipped the CIP label on
    2 of 50 seeds.  Issue #20 requires the jitter to "not invert chiral
    stereocenters", so the composed augmenter rejects-and-redraws any
    jitter draw whose signature differs from the pristine crystal
    signature (:meth:`ExposureBiasAugmenter.augment_positions`).
    """
    probe = Chem.Mol(mol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, positions[i].tolist())
    probe.RemoveAllConformers()
    probe.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(probe)
    labels = tuple(
        atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        for atom in probe.GetAtoms()
    )
    return labels, Chem.MolToSmiles(probe)


# ---------------------------------------------------------------------------
# The composed augmenter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExposureBiasAugmenter:
    """The #20 training-split augmentation: jitter + bounded dihedrals.

    Composition order (documented, deterministic): the rigid dihedral
    stage runs first so its ``+/-5 deg`` contract is measured on a
    geometry untouched by noise, then the Gaussian jitter is added on
    top (the rollout-error model: structured torsion error + unstructured
    coordinate noise).

    The rigid dihedral stage preserves every local chirality volume
    exactly (provable: every stereocentre either sits fully on one side
    of the rotation cut — rigid motion — or is centred on the axis atom
    whose fixed bond vector lies along the rotation axis — the mixed
    volume is rotation-invariant).  The jitter stage is **stereo-
    preserving by rejection resampling**: a draw whose 3D stereo
    signature (:func:`stereo_signature`) differs from the pristine
    crystal signature is discarded and redrawn (bounded attempts;
    fail-closed — the measured ~4%% flip rate makes the expected number
    of draws ~1.05).
    """

    sigma_angstrom: float = COORD_JITTER_SIGMA_ANGSTROM
    dihedral_max_deg: float = DIHEDRAL_PERTURB_MAX_DEG
    apply_jitter: bool = True
    apply_dihedrals: bool = True
    max_jitter_redraws: int = 100

    def __post_init__(self) -> None:
        if not (float(self.sigma_angstrom) > 0.0):
            raise AugmentationError("sigma_angstrom must be > 0")
        if not (float(self.dihedral_max_deg) > 0.0):
            raise AugmentationError("dihedral_max_deg must be > 0")
        if int(self.max_jitter_redraws) < 1:
            raise AugmentationError("max_jitter_redraws must be >= 1")

    def augment_positions(
        self,
        positions: np.ndarray,
        mol: Chem.Mol,
        junction_bonds: Sequence[Tuple[int, int]] = (),
        *,
        rng: np.random.Generator,
        split: str = AUGMENTATION_SPLIT,
    ) -> np.ndarray:
        """Augment one intermediate's coordinates (train split only).

        ``mol`` supplies the connectivity (rotatable-bond detection,
        distal subtrees); ``positions`` is the pristine ``[N, 3]``
        crystal geometry of the *same* molecule.  Returns a new array;
        the inputs are never mutated.
        """
        if split != AUGMENTATION_SPLIT:
            raise AugmentationError(
                f"augmentation is train-split only (#45 §12.1); refusing "
                f"split {split!r} — validation and test batches stay pristine"
            )
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape[0] != mol.GetNumAtoms():
            raise AugmentationError(
                f"geometry has {pos.shape[0]} atoms but the molecule has "
                f"{mol.GetNumAtoms()} — the connectivity and coordinates "
                "must describe the same intermediate"
            )
        if self.apply_dihedrals:
            pos = perturb_nonjunction_dihedrals(
                pos,
                mol,
                junction_bonds,
                rng=rng,
                max_delta_rad=math.radians(float(self.dihedral_max_deg)),
            )
        if self.apply_jitter:
            pos = self._jitter_stereo_preserving(pos, mol, rng)
        return pos

    def _jitter_stereo_preserving(
        self, pos: np.ndarray, mol: Chem.Mol, rng: np.random.Generator
    ) -> np.ndarray:
        """Gaussian jitter with the #20 no-inversion guarantee.

        Draws from :func:`jitter_ligand_coordinates` until the 3D stereo
        signature (:func:`stereo_signature`) matches the reference
        (measured flip probability ~4% at ``sigma = 0.15`` A — the loop
        converges in ~1.05 draws on average).  Fails closed after
        ``max_jitter_redraws`` rejected draws (never silently emits an
        inverted stereocentre).
        """
        reference = stereo_signature(mol, pos)
        last = None
        for _ in range(max(1, int(self.max_jitter_redraws))):
            last = jitter_ligand_coordinates(pos, float(self.sigma_angstrom), rng=rng)
            if stereo_signature(mol, last) == reference:
                return last
        raise AugmentationError(
            f"could not draw a stereo-preserving jitter within "
            f"{self.max_jitter_redraws} attempts (sigma="
            f"{self.sigma_angstrom} A) — refusing to emit a state with an "
            "inverted stereocentre (issue #20 acceptance)"
        )


# ---------------------------------------------------------------------------
# Single-graph state application (the collator-side hook)
# ---------------------------------------------------------------------------

def augment_single_state(
    state,
    mol: Chem.Mol,
    junction_bonds: Sequence[Tuple[int, int]],
    rng: np.random.Generator,
    *,
    augmenter: Optional[ExposureBiasAugmenter] = None,
    split: str = AUGMENTATION_SPLIT,
    augment: bool = True,
):
    """Return an augmented copy of one single-graph V2 state sample.

    The state is a ``torch_geometric.data.Data`` in the canonical 12-field
    layout (``syntree/core/schema.py``).  Only ``ligand_pos`` changes (the
    intermediate ligand coordinates ``X_L``); every other field —
    especially ``pocket_pos`` (the fixed conditioning context) and the
    invariant chemical features — is carried over bit-identically (the
    chemical identity of the state is unchanged; the augmentation models
    *input-coordinate* noise, not feature noise).

    ``augment=False`` returns the state unchanged (bit-identical ligand
    coordinates — the pristine validation/test path); ``augment=True``
    on a non-train split fails closed.
    """
    import torch

    if not augment:
        return state
    if split != AUGMENTATION_SPLIT:
        raise AugmentationError(
            f"augmentation is train-split only (#45 §12.1); refusing "
            f"split {split!r}"
        )
    ligand_pos = getattr(state, "ligand_pos", None)
    if ligand_pos is None:
        raise AugmentationError(
            "the state carries no `ligand_pos` — only canonical V2 states "
            "(syntree/core/schema.py) can be augmented"
        )
    if not isinstance(ligand_pos, torch.Tensor) or ligand_pos.dim() != 2:
        raise AugmentationError(
            f"ligand_pos must be a [N, 3] tensor, got {type(ligand_pos).__name__}"
        )
    if ligand_pos.shape[0] != mol.GetNumAtoms():
        raise AugmentationError(
            f"ligand_pos has {ligand_pos.shape[0]} atoms but the connectivity "
            f"molecule has {mol.GetNumAtoms()} — state and molecule disagree"
        )
    aug = augmenter if augmenter is not None else ExposureBiasAugmenter()
    pristine = ligand_pos.detach().cpu().numpy().astype(np.float64)
    augmented = aug.augment_positions(pristine, mol, junction_bonds, rng=rng, split=split)
    if not np.all(np.isfinite(augmented)):
        raise AugmentationError("augmentation produced non-finite coordinates")
    out = state.clone()
    out.ligand_pos = torch.as_tensor(augmented, dtype=ligand_pos.dtype)
    return out
