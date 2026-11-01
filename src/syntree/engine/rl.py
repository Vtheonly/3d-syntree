"""Stationary multi-objective terminal reward for Trajectory Balance.

Clean-room V2 implementation of the *reconciled* reward oracle (issue #30
FLOW-02 as amended by its stationarity addendum and the owner
reconciliation task; constitution #45 §12.4/§12.5; decision D-V2-009 in
``docs/agent/DECISIONS.md``).

**Stationarity is the contract.**  The Trajectory Balance target
``R(x)`` must be a stationary, reproducible, positive function of the
terminal molecule ``x`` and the *fixed* conditioning context (the pocket
and the frozen scoring configuration) — and of nothing else.  The
batch-level Tanimoto diversity term of the pre-addendum formulation is
therefore **removed**: it made ``R(x)`` depend on arbitrary rollout-batch
siblings, shifting the target distribution ``P(x) ∝ R(x)`` every batch and
destabilising the learnable ``log Z``.  :meth:`ThreeDReward.compute`
accepts *no* batch or sibling argument at all — the stationarity is
structural, not conventional (tested: identical to 8 decimal places across
batch compositions).

Population diversity is not lost: GFlowNet sampling proportional to
``R(x)`` naturally samples across modes (the anti-mode-collapse property
that motivated Stage 2), and population diversity metrics are reported as
evaluation telemetry by the evaluation suite (issue #35), never injected
into the per-trajectory loss.

**Reward formulation** (all terms stationary; units and signs documented)::

    R(x) = exp( ( S_dock + S_contact + S_key + S_props + S_vol ) / T_reward )

* ``S_dock`` — docking surrogate: ``clamp(-ΔG_dock / 2.0, -3.0, +5.0)``
  where ``ΔG_dock`` is the pocket interaction energy (soft-core LJ +
  softened Coulomb, kcal/mol) of the *relaxed* terminal pose.  A real
  docking engine replaces the surrogate when the evaluation stack lands;
  the clamp, units, and sign convention are the #30-specified ones.
* ``S_contact`` — soft-core contact enrichment: ``tanh(-E_LJ_att / 5.0)``
  over the *attractive-only* pocket LJ component (bounded in ``(-1, 1)``).
* ``S_key`` — validated pharmacophore bonus: ``+1.5`` per verified salt
  bridge (opposite pocket charges within 4.0 Å, capped at 2) plus
  ``+0.5 * min(N_hbond, 2)`` over heavy-atom N/O donor–acceptor contacts
  within 3.5 Å (documented surrogate rules; #14/#35 refine the
  pharmacophore validation).
* ``S_props`` — ``0.5 * QED + 0.5 * tanh(max(0, Fsp3 - 0.40) / 0.20)``
  (the #30-specified physicochemical balance, anti-"grease-ball").
* ``S_vol`` — the #14 CHEM-04 production volume floors (two parts, both
  stationary — the pocket cavity volume is derived **once at construction**
  from the fixed pocket context):
  the cavity-volume filling floor ``-2 * max(0, 1 - (V_lig / V_poc) /
  0.35)`` (penalises under-filling of the pocket cavity; ``0`` when the
  ligand reaches the 0.35 fill ratio) **plus** the explicit heavy-atom
  guard ``-2 * max(0, (12 - N_heavy) / 12)`` (penalises ``< 12`` heavy
  atoms — the anti-shrinking defence that is independent of pocket
  size).  An explicit ``volume_floor_angstrom3`` overrides the derived
  cavity floor (legacy configuration escape hatch); without a pocket and
  without an explicit floor the volume evaluation is inactive (``S_vol =
  0``, the graph-only reward).  (Replaces the earlier clash term per the
  owner reconciliation: steric clash already enters through the soft-core
  ``ΔG_dock`` surrogate with the correct sign.)

Bounds: the component sum lies in ``(-inf, +11]`` (each term bounded
above: 5 + 1 + 4 + 1 + 0; ``S_vol`` is bounded below by ``-4`` — two
floor terms of ``-2`` each), so ``R`` is finite; the hard floor
``R_min = 1e-4`` keeps ``log R`` finite (issue #30 acceptance).

**Positivity and fail-closed behaviour.**  Non-finite component inputs
raise :class:`RewardError` (constitution §8: genuinely poisoned inputs
fail loudly — no ``nan_to_num`` anywhere).  No reward component can
override chemical validity: invalid molecules never reach the oracle (the
environment sanitizes every product, #45 §12.5).

This module depends on ``rdkit`` and ``numpy``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed

from syntree.chemistry.conformer import (
    HEAVY_ATOM_FLOOR,
    HEAVY_ATOM_FLOOR_WEIGHT,
    VOLUME_RATIO_FLOOR,
    heavy_atom_floor_penalty,
    pocket_cavity_volume,
    volume_fill_ratio,
    volume_floor_penalty,
)
from syntree.engine.environment import (
    AssemblyError,
    PocketDefinition,
    coupled_charge_energy,
    soft_core_pocket_energy,
)

__all__ = [
    "RewardError",
    "ThreeDReward",
    "REWARD_FLOOR",
    "SALT_BRIDGE_CUTOFF",
    "HBOND_CUTOFF",
    "MAX_SALT_BRIDGES",
    "MAX_HBONDS",
    "VOLUME_RATIO_FLOOR",
    "HEAVY_ATOM_FLOOR",
    "HEAVY_ATOM_FLOOR_WEIGHT",
]

#: Hard reward floor: ``R(x) >= 1e-4`` so ``log R(x)`` stays finite (#30).
REWARD_FLOOR = 1e-4

#: Salt-bridge heavy-atom distance criterion (Å).
SALT_BRIDGE_CUTOFF = 4.0

#: Hydrogen-bond heavy-atom donor–acceptor criterion (Å).
HBOND_CUTOFF = 3.5

#: Maximum salt bridges credited in S_key (each worth +1.5).
MAX_SALT_BRIDGES = 2

#: Maximum hydrogen bonds credited in S_key (each worth +0.5).
MAX_HBONDS = 2

#: Upper bound of every component (documentation constant; sum <= 11).
_COMPONENT_BOUNDS = {
    "S_dock": 5.0,
    "S_contact": 1.0,
    "S_key": 1.5 * MAX_SALT_BRIDGES + 0.5 * MAX_HBONDS,
    "S_props": 1.0,
    "S_vol": 0.0,
}


class RewardError(ValueError):
    """Fail-closed reward error (non-finite physics, malformed input)."""


class ThreeDReward:
    """The stationary terminal reward ``R(x)`` of the V2 design.

    Construct once per pocket and frozen configuration; call
    :meth:`compute` (or :meth:`compute_with_components`) per terminal
    molecule.  The configuration is immutable after construction — a
    changed objective is a *changed training target* and must be a new,
    explicitly-versioned instance (constitution §12.4).
    """

    def __init__(
        self,
        pocket: Optional[PocketDefinition] = None,
        *,
        temperature: float = 1.0,
        reward_floor: float = REWARD_FLOOR,
        volume_floor_angstrom3: Optional[float] = None,
        volume_ratio_floor: Optional[float] = VOLUME_RATIO_FLOOR,
        heavy_atom_floor: int = HEAVY_ATOM_FLOOR,
        heavy_atom_floor_weight: float = HEAVY_ATOM_FLOOR_WEIGHT,
        cavity_grid_spacing: float = 0.5,
    ) -> None:
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise RewardError(f"temperature must be finite and > 0, got {temperature}")
        if reward_floor <= 0.0 or not math.isfinite(reward_floor):
            raise RewardError(f"reward_floor must be finite and > 0, got {reward_floor}")
        if volume_floor_angstrom3 is not None and volume_floor_angstrom3 <= 0.0:
            raise RewardError(
                f"volume_floor_angstrom3 must be > 0 or None, got {volume_floor_angstrom3}"
            )
        if volume_ratio_floor is not None and not (
            0.0 < volume_ratio_floor <= 1.0 and math.isfinite(volume_ratio_floor)
        ):
            raise RewardError(
                f"volume_ratio_floor must be in (0, 1] or None, got {volume_ratio_floor}"
            )
        if int(heavy_atom_floor) < 1:
            raise RewardError(
                f"heavy_atom_floor must be >= 1, got {heavy_atom_floor}"
            )
        if not (math.isfinite(heavy_atom_floor_weight) and heavy_atom_floor_weight >= 0.0):
            raise RewardError(
                f"heavy_atom_floor_weight must be finite and >= 0, got {heavy_atom_floor_weight}"
            )
        if not (math.isfinite(cavity_grid_spacing) and cavity_grid_spacing > 0.0):
            raise RewardError(
                f"cavity_grid_spacing must be finite and > 0, got {cavity_grid_spacing}"
            )
        self._pocket = pocket
        self._temperature = float(temperature)
        self._reward_floor = float(reward_floor)
        self._volume_ratio_floor = (
            float(volume_ratio_floor) if volume_ratio_floor is not None else None
        )
        self._heavy_atom_floor = int(heavy_atom_floor)
        self._heavy_atom_floor_weight = float(heavy_atom_floor_weight)
        # Stationarity (#45 §12.4): the cavity volume is a function of the
        # *fixed* pocket context, derived exactly once at construction —
        # identical molecules + identical pockets keep identical rewards.
        self._pocket_cavity_volume: Optional[float] = None
        if (
            pocket is not None
            and volume_floor_angstrom3 is None
            and self._volume_ratio_floor is not None
        ):
            self._pocket_cavity_volume = pocket_cavity_volume(
                pocket.positions,
                pocket.atomic_numbers,
                grid_spacing=cavity_grid_spacing,
            )
        self._volume_floor = (
            float(volume_floor_angstrom3)
            if volume_floor_angstrom3 is not None
            else (
                self._volume_ratio_floor * self._pocket_cavity_volume
                if self._pocket_cavity_volume is not None
                else None
            )
        )

    # -- configuration access -------------------------------------------------

    @property
    def pocket(self) -> Optional[PocketDefinition]:
        return self._pocket

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def reward_floor(self) -> float:
        return self._reward_floor

    @property
    def volume_floor(self) -> Optional[float]:
        return self._volume_floor

    @property
    def pocket_cavity_volume(self) -> Optional[float]:
        """The pocket's derived cavity volume (Å³, None unless derived)."""
        return self._pocket_cavity_volume

    @property
    def volume_ratio_floor(self) -> Optional[float]:
        return self._volume_ratio_floor

    @property
    def heavy_atom_floor(self) -> int:
        return self._heavy_atom_floor

    @property
    def heavy_atom_floor_weight(self) -> float:
        return self._heavy_atom_floor_weight

    # -- the reward -------------------------------------------------------------

    def compute(self, relaxed_mol: Chem.Mol) -> float:
        """Stationary scalar reward ``R(x) >= reward_floor`` of a relaxed terminal molecule.

        Identical molecule + identical pocket + identical frozen
        configuration ⇒ identical reward — regardless of any batch context
        (there is no batch context in the signature at all).
        """
        reward, _ = self.compute_with_components(relaxed_mol)
        return reward

    def compute_with_components(self, relaxed_mol: Chem.Mol) -> tuple:
        """``(R(x), components)`` — the reward plus fully labelled telemetry.

        The components dictionary records every term's inputs and value
        (units documented per key) for logging, checkpoint provenance, and
        the independent component tests of issue #30's acceptance.
        """
        if relaxed_mol is None or relaxed_mol.GetNumAtoms() == 0:
            raise RewardError("the terminal molecule is empty or missing")
        conformer = relaxed_mol.GetConformer()
        positions = np.array(conformer.GetPositions(), dtype=np.float64)
        numbers = np.array(
            [atom.GetAtomicNum() for atom in relaxed_mol.GetAtoms()], dtype=np.int64
        )
        if not np.all(np.isfinite(positions)):
            raise RewardError("terminal molecule coordinates contain non-finite values")

        # -- S_dock: docking surrogate (soft-core LJ + softened Coulomb) --------
        if self._pocket is not None:
            ligand_charges = self._gasteiger_charges(relaxed_mol)
            e_lj = soft_core_pocket_energy(positions, numbers, self._pocket)
            e_elec = coupled_charge_energy(positions, ligand_charges, self._pocket)
            delta_g = e_lj + e_elec
            s_dock = self._clamp(-delta_g / 2.0, -3.0, 5.0)
            # -- S_contact: attractive-only soft-core contact enrichment --------
            e_attractive = min(e_lj, 0.0)
            s_contact = math.tanh(-e_attractive / 5.0)
            # -- S_key: salt bridges + hydrogen bonds ----------------------------
            s_key = self._pharmacophore_bonus(relaxed_mol, positions, numbers)
        else:
            delta_g, e_lj, e_elec = 0.0, 0.0, 0.0
            s_dock = 0.0
            s_contact = 0.0
            s_key = 0.0

        # -- S_props: QED + Fsp3 -------------------------------------------------
        qed_value = float(qed(relaxed_mol))
        fsp3_value = self._fraction_sp3(relaxed_mol)
        s_props = 0.5 * qed_value + 0.5 * math.tanh(max(0.0, fsp3_value - 0.40) / 0.20)

        # -- S_vol: the #14 production volume floors (ratio + heavy atom) -------
        num_heavy = sum(1 for atom in relaxed_mol.GetAtoms() if atom.GetAtomicNum() > 1)
        if self._volume_floor is not None:
            volume = float(AllChem.ComputeMolVolume(relaxed_mol, confId=conformer.GetId()))
            if self._pocket_cavity_volume is not None:
                ratio_penalty = volume_floor_penalty(
                    volume, self._pocket_cavity_volume, self._volume_ratio_floor
                )
            else:
                # explicit-floor configuration (legacy escape hatch)
                ratio_penalty = -2.0 * max(
                    0.0, (self._volume_floor - volume) / self._volume_floor
                )
            heavy_penalty = heavy_atom_floor_penalty(
                num_heavy,
                floor=self._heavy_atom_floor,
                weight=self._heavy_atom_floor_weight,
            )
            s_vol = ratio_penalty + heavy_penalty
        else:
            volume = None
            ratio_penalty = 0.0
            heavy_penalty = 0.0
            s_vol = 0.0

        total = s_dock + s_contact + s_key + s_props + s_vol
        if not math.isfinite(total):
            raise RewardError(
                f"non-finite reward component sum ({total}); refusing to score a "
                f"poisoned terminal state (fail-closed)"
            )
        reward = math.exp(total / self._temperature)
        if not math.isfinite(reward):
            raise RewardError("reward overflow: component sum exceeds representable exp()")
        reward = max(reward, self._reward_floor)

        components: Dict[str, Any] = {
            "S_dock": s_dock,
            "S_contact": s_contact,
            "S_key": s_key,
            "S_props": s_props,
            "S_vol": s_vol,
            "sum": total,
            "temperature": self._temperature,
            "reward": reward,
            "delta_g_kcal_mol": delta_g,
            "e_lj_kcal_mol": e_lj,
            "e_elec_kcal_mol": e_elec,
            "qed": qed_value,
            "fsp3": fsp3_value,
            "volume_angstrom3": volume,
            "volume_floor_angstrom3": self._volume_floor,
            "volume_ratio": (
                volume_fill_ratio(volume, self._pocket_cavity_volume)
                if volume is not None and self._pocket_cavity_volume is not None
                else None
            ),
            "volume_ratio_floor": self._volume_ratio_floor,
            "pocket_cavity_volume_angstrom3": self._pocket_cavity_volume,
            "heavy_atom_count": num_heavy,
            "heavy_atom_floor": self._heavy_atom_floor,
            "heavy_atom_penalty": heavy_penalty,
            "ratio_penalty": ratio_penalty,
            "pocket_atoms": 0 if self._pocket is None else self._pocket.num_atoms,
        }
        return reward, components

    # -- component implementations ------------------------------------------------

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return min(max(value, low), high)

    @staticmethod
    def _gasteiger_charges(mol: Chem.Mol) -> np.ndarray:
        """Deterministic per-atom Gasteiger charges (fail-closed on failure)."""
        work = Chem.Mol(mol)
        try:
            AllChem.ComputeGasteigerCharges(work)
        except Exception as error:  # noqa: BLE001 - fail-closed with context
            raise RewardError(f"Gasteiger charge computation failed: {error}") from error
        charges = np.array(
            [float(atom.GetDoubleProp("_GasteigerCharge")) for atom in work.GetAtoms()],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(charges)):
            raise RewardError("Gasteiger charges are non-finite (poisoned input)")
        return charges

    @staticmethod
    def _fraction_sp3(mol: Chem.Mol) -> float:
        """Fraction of sp3 carbons over all carbons (0.0 when no carbon)."""
        carbons = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        if not carbons:
            return 0.0
        sp3 = sum(
            1
            for atom in carbons
            if atom.GetHybridization() == Chem.HybridizationType.SP3
        )
        return sp3 / len(carbons)

    def _pharmacophore_bonus(
        self,
        mol: Chem.Mol,
        positions: np.ndarray,
        numbers: np.ndarray,
    ) -> float:
        """Salt-bridge + hydrogen-bond bonus against the pocket (S_key).

        Documented surrogate rules (the full pharmacophore validation is
        owned by the #14/#35 stack):

        * **Salt bridge** — a ligand atom with |Gasteiger charge| >= 0.5 e
          within ``SALT_BRIDGE_CUTOFF`` of a pocket atom carrying an
          opposite-sign formal charge of |q| >= 0.5 e.  Worth ``+1.5``,
          capped at ``MAX_SALT_BRIDGES`` credits.
        * **Hydrogen bond** — a ligand N/O within ``HBOND_CUTOFF`` of a
          pocket N/O (heavy-atom donor–acceptor surrogate).  Worth
          ``+0.5``, capped at ``MAX_HBONDS`` credits.
        """
        assert self._pocket is not None
        charges = self._gasteiger_charges(mol)
        pocket = self._pocket
        deltas = positions[:, None, :] - pocket.positions[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))

        ligand_charged = np.abs(charges) >= 0.5
        pocket_charged = np.abs(pocket.charges) >= 0.5
        opposite = charges[:, None] * pocket.charges[None, :] < 0.0
        salt_pairs = (distances < SALT_BRIDGE_CUTOFF) & ligand_charged[:, None] & pocket_charged[None, :] & opposite
        num_salt = int(min(np.count_nonzero(salt_pairs), MAX_SALT_BRIDGES * 10_000))

        ligand_polar = np.isin(numbers, (7, 8))
        pocket_polar = np.isin(pocket.atomic_numbers, (7, 8))
        hbond_pairs = (distances < HBOND_CUTOFF) & ligand_polar[:, None] & pocket_polar[None, :]
        num_hbond = int(min(np.count_nonzero(hbond_pairs), MAX_HBONDS * 10_000))

        return 1.5 * min(num_salt, MAX_SALT_BRIDGES) + 0.5 * min(num_hbond, MAX_HBONDS)


# Re-exported for the convenience of trainer imports.
_ = AssemblyError  # (environment error type re-exported for type checkers)
