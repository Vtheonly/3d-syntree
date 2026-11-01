"""Production pocket physics: flat-bottom soft-core LJ clash + volume floors.

Clean-room V2 implementation of issue #14 CHEM-04 (constitution #45 §10.4
"Soft-core pocket interactions"; decision D-V2-015 in
``docs/agent/DECISIONS.md``; follow-up addendum of issue #14: this module
is the soft-core pocket-contact/clash energy that feeds the
pocket-conditioned torsion refinement of #42 CHEM-07).

**Why flat-bottom soft-core (the #14 motivation — legacy "Vulnerability 5").**
Rigid protein crystal structures penalise atom contact with an extreme
``r^-12`` repulsion: a single C-C pair at 0.5 Å contributes
``> 10^7 kcal/mol`` under the rigid Lennard-Jones wall.  A generative
policy trained against that wall learns the **"shrinking ligand"
collapse** — tiny, contracted fragments hovering in the cavity centre,
because touching the walls anywhere is catastrophically expensive.  The
issue-pinned remedy is a *flat-bottom soft-core* potential::

    V_soft(r) = 0                                  for r >= r_vdw
    V_soft(r) = 4 eps sigma^6 / (r^6 + delta)      for r <  r_vdw

with ``delta = 0.5 A^6`` bounding the maximum repulsive spike under total
atomic overlap.  Properties (all pinned by tests):

* **Flat bottom** — beyond the contact distance the energy is exactly
  zero (bitwise): neutral geometry is never penalised, so MMFF owns the
  intramolecular energetics and the policy is not pushed off the walls.
* **Bounded spike** — ``V_soft(0) = 4 eps sigma^6 / delta`` and
  ``V_soft(0.5) = 4 eps sigma^6 / (0.5^6 + 0.5) <= 484.85 kcal/mol`` for
  every supported element pair *by construction* (the acceptance bound):
  the per-pair depth is capped at
  ``eps_ij = min(sqrt(eps_i eps_j), V_max * delta / (4 sigma_ij^6))``
  with ``V_max = 500 kcal/mol``.  The rigid wall at the same distance is
  ``> 10^7`` — nine orders of magnitude removed.
* **Repulsive-only** — the attractive branch is deliberately discarded
  (that is what "flat-bottom" means here); contact enrichment / affinity
  semantics live in the reward's docking surrogate (#30 FLOW-02 owns its
  production closure) and in MMFF, not in the clash penalty.
* **Piecewise-smooth** — ``dV/dr = -24 eps sigma^6 r^5 / (r^6 + delta)^2``
  inside the branch (finite everywhere, exactly zero at ``r = 0``, bounded
  by construction); at ``r = r_vdw`` the pinned form has a *finite jump*
  of ``4 eps sigma^6 / (r_vdw^6 + delta) ~ 4 eps`` (<= ~0.17 kcal/mol for
  a capped C-C pair) — documented, bounded, and irrelevant to the
  scan-seeded bracketed Brent refinement of #42, whose coarse phase
  straddles it by design.

**Atom typing (documented; issue addendum mandate).**  Per-element van
der Waals radii are the Bondi (1964) values re-adopted from the pre-reset
conformer specification (decision D-V2-002 — specification knowledge, not
code); per-element well depths are the D-V2-009 oracle table
(generic OPLS/AMBER-scale kcal/mol values).  Pair parameters use
Lorentz-Berthelot mixing (``sigma_ij = r_i + r_j``,
``eps_ij = sqrt(eps_i eps_j)``) *capped* by the bounded-spike ceiling
above.  The ceiling is active exactly for the heavy-heavy pairs
(``sigma_ij >= ~3.1 A``) where the flat-bottom form's epsilon is a
penalty scale rather than a thermodynamic well depth (the attractive
branch is discarded anyway); light pairs keep their Lorentz-Berthelot
depth.  Unknown elements (pocket metals/ions) follow the documented
policy of the class: ``"fallback"`` (default — the D-V2-009 oracle
fallback ``(1.70 A, 0.100 kcal/mol)``, matching the surrogate the
production potential supersedes) or ``"raise"`` (strict).

**Cavity-volume machinery (issue-pinned).**  The shrinking-ligand defence
has a second flank: a *volume floor*.  ``pocket_cavity_volume`` is a
deterministic grid estimator of the pocket's void volume (points inside
the pocket's envelope sphere, outside every pocket atom's vdW sphere);
``volume_floor_penalty`` penalises ``V_lig / V_poc < 0.35`` and
``heavy_atom_floor_penalty`` adds the explicit heavy-atom-count guard
(penalty for ``< 12`` heavy atoms).  Both are wired into
:class:`syntree.engine.rl.ThreeDReward` as the production ``S_vol`` (the
reward's configuration stays stationary: ``V_poc`` is derived once at
construction from the fixed pocket context).

**Numerical policy (constitution §8).**  Fail-closed everywhere:
non-finite coordinates, malformed shapes, atomic numbers < 1, non-positive
volumes and radii raise :class:`ChemistryError`; there are no NaN
replacements and no clamps on valid inputs.  Deterministic: every function
is pure arithmetic over float64 numpy arrays (plus an optional torch
mirror for autograd consumers), bit-reproducible for identical inputs.

This module depends on ``numpy`` (engine path) and ``torch`` (the
autograd mirror only).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

__all__ = [
    "ChemistryError",
    "FLAT_BOTTOM_DELTA",
    "CLASH_SPIKE_CEILING",
    "VOLUME_RATIO_FLOOR",
    "HEAVY_ATOM_FLOOR",
    "HEAVY_ATOM_FLOOR_WEIGHT",
    "VDW_RADII",
    "LJ_WELL_DEPTHS",
    "DEFAULT_ATOM_TYPE",
    "atom_type",
    "flat_bottom_soft_core_lj",
    "flat_bottom_soft_core_lj_gradient",
    "flat_bottom_soft_core_lj_torch",
    "pair_epsilon",
    "SoftCoreClashPotential",
    "soft_core_clash_energy",
    "pocket_cavity_volume",
    "volume_fill_ratio",
    "volume_floor_penalty",
    "heavy_atom_floor_penalty",
]


class ChemistryError(ValueError):
    """Fail-closed chemistry/physics error (malformed or poisoned input)."""


#: Softening constant of the pinned flat-bottom form (Å^6): bounds the
#: maximum repulsive spike under total atomic overlap (issue #14).
FLAT_BOTTOM_DELTA = 0.5

#: Maximum per-pair repulsive spike at total overlap, ``V(0) = 4 eps
#: sigma^6 / delta`` (kcal/mol) — the issue #14 acceptance bound, enforced
#: for every supported element pair by the epsilon ceiling.
CLASH_SPIKE_CEILING = 500.0

#: Cavity-volume filling floor: ligands with ``V_lig / V_poc`` below this
#: ratio are penalised upon termination (issue #14).
VOLUME_RATIO_FLOOR = 0.35

#: Heavy-atom-count floor: terminal molecules with fewer heavy atoms incur
#: an explicit volume-evaluation penalty (issue #14 acceptance 2).
HEAVY_ATOM_FLOOR = 12

#: Weight of the heavy-atom floor penalty (kcal/mol-equivalent reward
#: units): ``-weight * max(0, (floor - n) / floor)``.
HEAVY_ATOM_FLOOR_WEIGHT = 2.0

#: Bondi (1964) van der Waals radii (Å), re-adopted from the pre-reset
#: conformer specification (D-V2-002).  Keys are atomic numbers.
VDW_RADII: Dict[int, float] = {
    1: 1.10,   # H
    5: 1.92,   # B
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    11: 2.27,  # Na
    12: 1.73,  # Mg
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    19: 2.75,  # K
    20: 2.31,  # Ca
    30: 2.39,  # Zn
    34: 1.90,  # Se
    35: 1.85,  # Br
    53: 1.98,  # I
}

#: Per-element Lennard-Jones well depths (kcal/mol) — the D-V2-009 oracle
#: table (generic OPLS/AMBER-scale values), now the production conformer
#: potential's atom typing.
LJ_WELL_DEPTHS: Dict[int, float] = {
    1: 0.015,
    6: 0.086,
    7: 0.170,
    8: 0.170,
    9: 0.150,
    15: 0.200,
    16: 0.250,
    17: 0.250,
    35: 0.260,
    53: 0.300,
}

#: Documented fallback atom type for elements outside the tables
#: (the D-V2-009 oracle fallback: carbon-scale contact, generic depth).
DEFAULT_ATOM_TYPE = (1.70, 0.100)


def atom_type(atomic_number: int) -> Tuple[float, float]:
    """``(vdW radius [A], well depth [kcal/mol])`` for an element.

    Unknown elements follow the *documented fallback* (carbon-scale);
    use :class:`SoftCoreClashPotential` with ``unknown_element="raise"``
    for strict typing.
    """
    z = int(atomic_number)
    if z < 1:
        raise ChemistryError(f"atomic numbers must be >= 1, got {atomic_number}")
    radius = VDW_RADII.get(z)
    depth = LJ_WELL_DEPTHS.get(z)
    if radius is None or depth is None:
        return DEFAULT_ATOM_TYPE
    return (radius, depth)


# ---------------------------------------------------------------------------
# The pinned flat-bottom soft-core Lennard-Jones potential
# ---------------------------------------------------------------------------


def flat_bottom_soft_core_lj(
    r: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
    delta: float = FLAT_BOTTOM_DELTA,
) -> np.ndarray:
    """The issue-pinned flat-bottom soft-core LJ potential (kcal/mol).

    ``V(r) = 0`` for ``r >= sigma`` (the flat bottom — bitwise zero) and
    ``V(r) = 4 epsilon sigma^6 / (r^6 + delta)`` for ``r < sigma``.
    Accepts scalars or broadcastable arrays; returns non-negative float64
    values.  Fail-closed on non-finite inputs and non-positive sigma /
    epsilon / delta.
    """
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(sigma)) or np.any(~np.isfinite(epsilon)):
        raise ChemistryError("non-finite distance/sigma/epsilon input (fail-closed)")
    if np.any(sigma <= 0.0):
        raise ChemistryError("sigma must be positive (fail-closed)")
    if np.any(epsilon < 0.0):
        raise ChemistryError("epsilon must be non-negative (fail-closed)")
    if not (math.isfinite(delta) and delta > 0.0):
        raise ChemistryError(f"delta must be finite and > 0, got {delta}")
    inside = r < sigma
    sigma6 = sigma ** 6
    repulsive = 4.0 * epsilon * sigma6 / (r ** 6 + delta)
    return np.where(inside, repulsive, 0.0)


def flat_bottom_soft_core_lj_gradient(
    r: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
    delta: float = FLAT_BOTTOM_DELTA,
) -> np.ndarray:
    """Analytic ``dV/dr`` of the pinned potential (kcal/mol/Å).

    ``dV/dr = -24 epsilon sigma^6 r^5 / (r^6 + delta)^2`` inside the
    branch (finite everywhere, exactly zero at ``r = 0``), and exactly
    zero on and beyond the flat bottom.  The jump at ``r = sigma`` is a
    documented bounded discontinuity of the pinned form (see the module
    docstring), not a gradient.
    """
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(sigma)) or np.any(~np.isfinite(epsilon)):
        raise ChemistryError("non-finite distance/sigma/epsilon input (fail-closed)")
    if np.any(sigma <= 0.0) or np.any(epsilon < 0.0):
        raise ChemistryError("sigma must be positive and epsilon non-negative (fail-closed)")
    if not (math.isfinite(delta) and delta > 0.0):
        raise ChemistryError(f"delta must be finite and > 0, got {delta}")
    inside = r < sigma
    sigma6 = sigma ** 6
    grad = -24.0 * epsilon * sigma6 * r ** 5 / (r ** 6 + delta) ** 2
    return np.where(inside, grad, 0.0)


def flat_bottom_soft_core_lj_torch(
    r: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    delta: float = FLAT_BOTTOM_DELTA,
) -> torch.Tensor:
    """Autograd-trackable torch mirror of the pinned potential.

    Identical mathematics to :func:`flat_bottom_soft_core_lj` implemented
    with differentiable elementwise torch ops (the "differentiable
    potential" of the issue body): gradients with respect to ``r``,
    ``sigma`` and ``epsilon`` flow through autograd and match the analytic
    :func:`flat_bottom_soft_core_lj_gradient` (test-pinned).  The engine
    path uses the numpy core for deterministic float64 evaluation; this
    mirror exists for model-side consumers that need gradients.
    """
    if not (math.isfinite(delta) and delta > 0.0):
        raise ChemistryError(f"delta must be finite and > 0, got {delta}")
    inside = r < sigma
    sigma6 = sigma ** 6
    repulsive = 4.0 * epsilon * sigma6 / (r ** 6 + delta)
    return torch.where(inside, repulsive, torch.zeros_like(repulsive))


def pair_epsilon(
    epsilon_i: float, epsilon_j: float, sigma_ij: float
) -> float:
    """Lorentz-Berthelot depth capped by the bounded-spike ceiling.

    ``eps_ij = min(sqrt(eps_i * eps_j), V_max * delta / (4 sigma_ij^6))``
    — the #14 acceptance bound (``V(0) <= 500`` kcal/mol per pair) holds
    for every pair by construction.
    """
    if epsilon_i < 0.0 or epsilon_j < 0.0:
        raise ChemistryError("well depths must be non-negative (fail-closed)")
    if not (math.isfinite(sigma_ij) and sigma_ij > 0.0):
        raise ChemistryError(f"sigma_ij must be finite and positive, got {sigma_ij}")
    lorentz_berthelot = math.sqrt(epsilon_i * epsilon_j)
    ceiling = CLASH_SPIKE_CEILING * FLAT_BOTTOM_DELTA / (4.0 * sigma_ij ** 6)
    return min(lorentz_berthelot, ceiling)


# ---------------------------------------------------------------------------
# The pair-matrix clash potential (the engine-facing production module)
# ---------------------------------------------------------------------------


class SoftCoreClashPotential:
    """The production pocket clash potential over ligand-pocket atom pairs.

    Configuration is frozen at construction (a changed potential is a
    changed objective — construct a new instance); the defaults are the
    issue-pinned values.  All evaluations are deterministic float64 numpy
    arithmetic and fail closed on malformed or poisoned input.
    """

    def __init__(
        self,
        *,
        delta: float = FLAT_BOTTOM_DELTA,
        spike_ceiling: float = CLASH_SPIKE_CEILING,
        unknown_element: str = "fallback",
    ) -> None:
        if not (math.isfinite(delta) and delta > 0.0):
            raise ChemistryError(f"delta must be finite and > 0, got {delta}")
        if not (math.isfinite(spike_ceiling) and spike_ceiling > 0.0):
            raise ChemistryError(f"spike_ceiling must be finite and > 0, got {spike_ceiling}")
        if unknown_element not in ("fallback", "raise"):
            raise ChemistryError(
                f"unknown_element policy must be 'fallback' or 'raise', got {unknown_element!r}"
            )
        self._delta = float(delta)
        self._spike_ceiling = float(spike_ceiling)
        self._unknown_element = unknown_element

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def spike_ceiling(self) -> float:
        return self._spike_ceiling

    @property
    def unknown_element(self) -> str:
        return self._unknown_element

    # -- atom typing -----------------------------------------------------------

    def _element_parameters(
        self, atomic_numbers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        radii = np.empty(atomic_numbers.shape[0], dtype=np.float64)
        depths = np.empty(atomic_numbers.shape[0], dtype=np.float64)
        for index, z in enumerate(atomic_numbers.tolist()):
            z = int(z)
            if z < 1:
                raise ChemistryError(f"atomic numbers must be >= 1, got {z}")
            radius = VDW_RADII.get(z)
            depth = LJ_WELL_DEPTHS.get(z)
            if radius is None or depth is None:
                if self._unknown_element == "raise":
                    raise ChemistryError(
                        f"element Z={z} is outside the documented atom-typing tables "
                        f"(strict policy; VDW_RADII/LJ_WELL_DEPTHS list "
                        f"{sorted(set(VDW_RADII) & set(LJ_WELL_DEPTHS))})"
                    )
                radius, depth = DEFAULT_ATOM_TYPE
            radii[index] = radius
            depths[index] = depth
        return radii, depths

    def pair_parameters(
        self,
        ligand_atomic_numbers: np.ndarray,
        pocket_atomic_numbers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(sigma [N, M], epsilon [N, M])`` pair parameter matrices.

        ``sigma_ij = r_i + r_j`` (Lorentz contact distance) and
        ``epsilon_ij = min(sqrt(eps_i eps_j), ceiling(sigma_ij))`` (the
        bounded-spike cap).
        """
        lig_z = np.asarray(ligand_atomic_numbers, dtype=np.int64)
        poc_z = np.asarray(pocket_atomic_numbers, dtype=np.int64)
        self._validate_numbers(lig_z, "ligand")
        self._validate_numbers(poc_z, "pocket")
        lig_r, lig_e = self._element_parameters(lig_z)
        poc_r, poc_e = self._element_parameters(poc_z)
        sigma = lig_r[:, None] + poc_r[None, :]
        lorentz_berthelot = np.sqrt(lig_e[:, None] * poc_e[None, :])
        ceiling = self._spike_ceiling * self._delta / (4.0 * sigma ** 6)
        epsilon = np.minimum(lorentz_berthelot, ceiling)
        return sigma, epsilon

    # -- energies ---------------------------------------------------------------

    def pair_energy(
        self,
        ligand_positions: np.ndarray,
        ligand_atomic_numbers: np.ndarray,
        pocket_positions: np.ndarray,
        pocket_atomic_numbers: np.ndarray,
    ) -> np.ndarray:
        """Per-pair clash energy matrix ``[N, M]`` (kcal/mol, >= 0)."""
        lig_pos, poc_pos = self._validate_geometry(
            ligand_positions, pocket_positions
        )
        lig_z = np.asarray(ligand_atomic_numbers, dtype=np.int64)
        poc_z = np.asarray(pocket_atomic_numbers, dtype=np.int64)
        self._validate_numbers(lig_z, "ligand")
        self._validate_numbers(poc_z, "pocket")
        if lig_pos.shape[0] != lig_z.shape[0] or poc_pos.shape[0] != poc_z.shape[0]:
            raise ChemistryError(
                f"position/atomic-number count mismatch: {lig_pos.shape[0]} ligand "
                f"positions vs {lig_z.shape[0]} numbers, {poc_pos.shape[0]} pocket "
                f"positions vs {poc_z.shape[0]} numbers"
            )
        sigma, epsilon = self.pair_parameters(lig_z, poc_z)
        deltas = lig_pos[:, None, :] - poc_pos[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
        return flat_bottom_soft_core_lj(r, sigma, epsilon, self._delta)

    def total_energy(
        self,
        ligand_positions: np.ndarray,
        ligand_atomic_numbers: np.ndarray,
        pocket_positions: np.ndarray,
        pocket_atomic_numbers: np.ndarray,
    ) -> float:
        """Total ligand-pocket clash energy (kcal/mol, >= 0, bounded)."""
        total = float(
            np.sum(
                self.pair_energy(
                    ligand_positions,
                    ligand_atomic_numbers,
                    pocket_positions,
                    pocket_atomic_numbers,
                )
            )
        )
        if not math.isfinite(total):
            raise ChemistryError("clash energy is non-finite (poisoned input)")
        return total

    def per_ligand_atom_energy(
        self,
        ligand_positions: np.ndarray,
        ligand_atomic_numbers: np.ndarray,
        pocket_positions: np.ndarray,
        pocket_atomic_numbers: np.ndarray,
    ) -> np.ndarray:
        """Per-ligand-atom clash energy ``[N]`` (kcal/mol, >= 0)."""
        return self.pair_energy(
            ligand_positions,
            ligand_atomic_numbers,
            pocket_positions,
            pocket_atomic_numbers,
        ).sum(axis=1)

    # -- validation ---------------------------------------------------------------

    @staticmethod
    def _validate_numbers(numbers: np.ndarray, side: str) -> None:
        if numbers.ndim != 1:
            raise ChemistryError(
                f"{side} atomic numbers must be 1-D, got shape {numbers.shape}"
            )
        if numbers.size == 0:
            raise ChemistryError(f"{side} atomic numbers must not be empty")
        if np.any(numbers < 1):
            raise ChemistryError(f"{side} atomic numbers must be >= 1")

    @staticmethod
    def _validate_geometry(
        ligand_positions: np.ndarray, pocket_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lig = np.asarray(ligand_positions, dtype=np.float64)
        poc = np.asarray(pocket_positions, dtype=np.float64)
        if lig.ndim != 2 or lig.shape[1] != 3:
            raise ChemistryError(f"ligand positions must be [N, 3], got {lig.shape}")
        if poc.ndim != 2 or poc.shape[1] != 3:
            raise ChemistryError(f"pocket positions must be [M, 3], got {poc.shape}")
        if lig.shape[0] == 0:
            raise ChemistryError("ligand positions must not be empty")
        if poc.shape[0] == 0:
            raise ChemistryError("pocket positions must not be empty")
        if not np.all(np.isfinite(lig)) or not np.all(np.isfinite(poc)):
            raise ChemistryError("positions contain non-finite values (fail-closed)")
        return lig, poc


#: Module-level default instance (documented fallback atom-typing policy,
#: issue-pinned delta and ceiling) — the engine-facing convenience path.
_DEFAULT_POTENTIAL = SoftCoreClashPotential()


def soft_core_clash_energy(
    ligand_positions: np.ndarray,
    ligand_atomic_numbers: np.ndarray,
    pocket_positions: np.ndarray,
    pocket_atomic_numbers: np.ndarray,
    potential: Optional[SoftCoreClashPotential] = None,
) -> float:
    """Total flat-bottom soft-core clash energy of a ligand pose (kcal/mol).

    Convenience wrapper mirroring the engine surrogate's signature shape
    (array-based to keep this package engine-agnostic — no import of
    ``syntree.engine`` and hence no import cycle).  Uses the module
    default potential unless one is supplied.
    """
    engine = potential if potential is not None else _DEFAULT_POTENTIAL
    return engine.total_energy(
        ligand_positions,
        ligand_atomic_numbers,
        pocket_positions,
        pocket_atomic_numbers,
    )


# ---------------------------------------------------------------------------
# Cavity-volume machinery (issue #14)
# ---------------------------------------------------------------------------


def pocket_cavity_volume(
    pocket_positions: np.ndarray,
    pocket_atomic_numbers: np.ndarray,
    *,
    grid_spacing: float = 0.5,
    unknown_element: str = "fallback",
) -> float:
    """Deterministic grid estimate of the pocket's void (cavity) volume.

    Method (documented estimator, units Å³):

    1. the pocket's envelope sphere is centred at the centroid of the
       pocket atoms with radius ``R_env = max_i |x_i - c| + r_i`` (every
       vdW shell is enclosed);
    2. a regular cubic grid with spacing ``grid_spacing`` covers the
       envelope sphere;
    3. a grid point is *cavity* when it lies inside the envelope sphere
       AND outside every pocket atom's vdW sphere;
    4. ``V_poc = count * grid_spacing^3``.

    Deterministic (pure integer/float arithmetic, fixed iteration order),
    monotone in the pocket's spatial extent, and O(grid x atoms).  It is a
    *floor-calibration scale*, not a geometric constant: it overestimates
    for fully open/exposed surfaces (the exterior region inside the
    envelope sphere counts).  For large pockets prefer a coarser
    ``grid_spacing`` (cost falls with the cube of the spacing).
    """
    poc = np.asarray(pocket_positions, dtype=np.float64)
    numbers = np.asarray(pocket_atomic_numbers, dtype=np.int64)
    if poc.ndim != 2 or poc.shape[1] != 3:
        raise ChemistryError(f"pocket positions must be [M, 3], got {poc.shape}")
    if numbers.ndim != 1 or numbers.shape[0] != poc.shape[0]:
        raise ChemistryError(
            f"pocket field mismatch: {poc.shape[0]} positions vs {numbers.shape}"
        )
    if poc.shape[0] == 0:
        raise ChemistryError("the pocket must contain at least one atom")
    if np.any(numbers < 1):
        raise ChemistryError("pocket atomic numbers must be >= 1")
    if not np.all(np.isfinite(poc)):
        raise ChemistryError("pocket positions contain non-finite values (fail-closed)")
    if not (math.isfinite(grid_spacing) and grid_spacing > 0.0):
        raise ChemistryError(f"grid_spacing must be finite and > 0, got {grid_spacing}")
    if unknown_element not in ("fallback", "raise"):
        raise ChemistryError(
            f"unknown_element policy must be 'fallback' or 'raise', got {unknown_element!r}"
        )

    radii = np.empty(numbers.shape[0], dtype=np.float64)
    for index, z in enumerate(numbers.tolist()):
        z = int(z)
        radius = VDW_RADII.get(z)
        if radius is None:
            if unknown_element == "raise":
                raise ChemistryError(f"element Z={z} is outside VDW_RADII (strict policy)")
            radius = DEFAULT_ATOM_TYPE[0]
        radii[index] = radius

    centroid = poc.mean(axis=0)
    offsets = np.linalg.norm(poc - centroid, axis=1)
    r_env = float(np.max(offsets + radii))

    half = r_env + grid_spacing
    steps = int(math.ceil(2.0 * half / grid_spacing)) + 1
    axis = np.linspace(-half, half, steps)
    grid_x, grid_y, grid_z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack(
        [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1
    )
    inside_envelope = np.einsum("ij,ij->i", points, points) <= r_env * r_env
    candidate = points[inside_envelope]
    if candidate.shape[0] == 0:
        raise ChemistryError("empty cavity grid: the pocket envelope is degenerate")
    # Vectorised outside-vdW test in chunks (bounded memory for big pockets).
    outside_all = np.ones(candidate.shape[0], dtype=bool)
    chunk = max(1, int(4_000_000 // max(1, poc.shape[0])))
    for start in range(0, candidate.shape[0], chunk):
        block = candidate[start : start + chunk]
        deltas = block[:, None, :] - poc[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
        outside = np.all(distances >= radii, axis=1)
        outside_all[start : start + chunk] = outside
    count = int(np.count_nonzero(outside_all))
    volume = count * grid_spacing ** 3
    if not math.isfinite(volume):
        raise ChemistryError("cavity volume is non-finite (poisoned input)")
    return volume


def volume_fill_ratio(ligand_volume: float, pocket_volume: float) -> float:
    """``V_lig / V_poc`` (fail-closed on non-positive pocket volume)."""
    if not math.isfinite(ligand_volume) or ligand_volume < 0.0:
        raise ChemistryError(f"ligand volume must be finite and >= 0, got {ligand_volume}")
    if not math.isfinite(pocket_volume) or pocket_volume <= 0.0:
        raise ChemistryError(
            f"pocket volume must be finite and > 0, got {pocket_volume}"
        )
    return ligand_volume / pocket_volume


def volume_floor_penalty(
    ligand_volume: float,
    pocket_volume: float,
    ratio_floor: float = VOLUME_RATIO_FLOOR,
) -> float:
    """Cavity-volume filling floor penalty (<= 0, bounded below by -2).

    ``penalty = -2 * max(0, 1 - (V_lig / V_poc) / ratio_floor)`` — zero
    when the ligand fills at least ``ratio_floor`` of the cavity, linearly
    penalised below it.  The shape matches the legacy ``S_vol`` term of
    :class:`syntree.engine.rl.ThreeDReward` (bounded in ``[-2, 0]``).
    """
    if not (math.isfinite(ratio_floor) and 0.0 < ratio_floor <= 1.0):
        raise ChemistryError(f"ratio_floor must be in (0, 1], got {ratio_floor}")
    ratio = volume_fill_ratio(ligand_volume, pocket_volume)
    return -2.0 * max(0.0, 1.0 - ratio / ratio_floor)


def heavy_atom_floor_penalty(
    num_heavy_atoms: int,
    floor: int = HEAVY_ATOM_FLOOR,
    weight: float = HEAVY_ATOM_FLOOR_WEIGHT,
) -> float:
    """Explicit heavy-atom-count floor penalty (<= 0; issue #14 acceptance 2).

    ``penalty = -weight * max(0, (floor - n) / floor)`` — zero at
    ``n >= floor`` (default 12), strictly negative below it, bounded by
    ``-weight``.  The anti-shrinking guard that is independent of the
    pocket's size (a small pocket cannot excuse a fragment).
    """
    n = int(num_heavy_atoms)
    if n < 0:
        raise ChemistryError(f"heavy atom count must be >= 0, got {num_heavy_atoms}")
    if int(floor) < 1:
        raise ChemistryError(f"floor must be >= 1, got {floor}")
    if not (math.isfinite(weight) and weight >= 0.0):
        raise ChemistryError(f"weight must be finite and >= 0, got {weight}")
    return -float(weight) * max(0.0, (float(floor) - float(n)) / float(floor))
