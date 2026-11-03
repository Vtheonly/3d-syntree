"""Approach C quantile curriculum learning engine (#27 TRAIN-02).

Clean-room V2 implementation of issue #27 (constitution #45 §12.1:
"Implement #27 Approach C: nine quantile-balanced difficulty levels using
documented physical/chemical features and a non-zero per-level probability
floor of at least 0.03"; decision D-V2-037 in ``docs/agent/DECISIONS.md``).

**The problem (issue #27 context, Decision D16).**  Uniform sampling
forces early updates to struggle with bulky, high-clash fragments.
Curriculum learning accelerates convergence by biasing early training
toward rigid, low-clash fragments while guaranteeing full dataset
coverage through a permanent probability floor.

**Difficulty features (the five issue-named physical/chemical axes).**
Per supervised item (:class:`~syntree.data.supervised.\
SupervisedTrajectoryDataset` item table entry), computed by
:func:`difficulty_features`:

1. ``pocket_atoms`` — the capped pocket atom count of the complex
   (bigger conditioning environment = harder);
2. ``rotatable_bonds`` — RDKit strict rotatable bonds of the crystal
   intermediate the decision is taken on (flexibility = harder);
3. ``rings`` — SSSR ring count of that intermediate (structural
   complexity = harder);
4. ``pocket_charge_magnitude`` — ``sum |q|`` over the pocket's Gasteiger
   charges (electrostatic environment complexity = harder);
5. ``handle_rarity`` — ``1 - count(class)/max_count`` of the item's
   primary handle class over the training split's items (rarer = harder;
   handle-free items — the pre-seed and terminal STOP samples — sit at
   the common end, ``0.0``, documented D-V2-037).

Features are z-scored over the training split (a constant column
contributes nothing — the documented degenerate-axis behaviour) and
combined with **equal weights** (the plain mean of the z-scores; the
issue names the axes but not weights, and only the induced *ranking*
matters for quantile binning — D-V2-037).

**The 9 quantile-balanced levels (L1..L9).**  Items are sorted by the
scalar difficulty score and split into ``num_levels`` contiguous groups
of sizes as equal as possible (a *balanced* quantile assignment: every
level holds ``N // 9`` or ``N // 9 + 1`` items).  Construction fails
closed when fewer than 9 items exist.

**The 6-stage level distribution (Approach C).**  Stage ``s in [1, 6]``
centres its sampling mass at

``mu(s) = 1 + (s - 1) * (L - 1) / (S - 1)``  (linear 1 -> 9),

using a truncated Gaussian kernel over the integer levels with width
``kernel_sigma = 2.0`` levels.  The stage distribution applies the
permanent probability floor exactly (D-V2-038):

``p(level) = (1 - L * floor) * kernel(level) + floor``

— a mixture with the uniform distribution, which sums to 1 *by
construction* and keeps every level at **exactly** ``floor >= 0.03``
(the naive "clamp-then-renormalize" floor can silently dilute below the
promised bound and is not used).

**Milestone-driven stage advancement (the issue's exact rule).**  After
each epoch's validation pass, :meth:`CurriculumSampler.\
observe_validation_loss` compares against the running minimum
validation loss:

* improvement ``>= 0.005`` advances the stage (the milestone);
* otherwise the patience counter increments, and **two** consecutive
  non-improving epochs advance the stage anyway (the fallback);
* the stage never regresses and never exceeds ``num_stages``.

**Exact step count and determinism.**  Every epoch draws **exactly**
``N = len(dataset)`` samples: a multinomial draw of ``N`` level picks
from the stage distribution, then uniform picks within each level's
pool, then a full shuffle.  All randomness flows through a NumPy
``Generator`` derived deterministically from ``(seed, stage, epoch)``
(mirroring the dataset's per-item stream pattern), so re-iterating an
epoch reproduces the identical sequence — the property the trainer's
exact-resume contract (#45 §13.7) relies on; no mid-epoch generator
state ever needs to be persisted.

**Sampler/curriculum checkpoint state (#45 §12.1).**  The full state —
stage, epoch, running-best validation loss, patience, transition log,
per-epoch level histograms, and the frozen configuration — round-trips
through :meth:`CurriculumSampler.state_dict` /
:meth:`CurriculumSampler.load_state_dict` and is carried inside the
trainer checkpoint payload under the ``"curriculum"`` key (the #43
session-safe-resume contract lists sampler/curriculum state as a
required checkpoint component).

This module depends on ``numpy`` and (for ingestion only) the RDKit
descriptors of already-materialized intermediates — no training-time
GPU state is involved, so the CPU dev sandbox verifies it completely.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from torch.utils.data import Sampler

__all__ = [
    "CurriculumError",
    "CURRICULUM_NUM_LEVELS",
    "CURRICULUM_NUM_STAGES",
    "CURRICULUM_PROBABILITY_FLOOR",
    "CURRICULUM_KERNEL_SIGMA",
    "CURRICULUM_SPLIT",
    "CURRICULUM_FEATURE_NAMES",
    "VAL_IMPROVEMENT_MILESTONE",
    "STAGE_PATIENCE",
    "CURRICULUM_STATE_SCHEMA_VERSION",
    "difficulty_features",
    "standardize_features",
    "difficulty_scores",
    "quantile_levels",
    "stage_center",
    "level_distribution",
    "CurriculumSampler",
]

#: Issue-pinned level count: "Compute 9 quantile-balanced bins (L1..L9)".
CURRICULUM_NUM_LEVELS = 9

#: Issue-pinned stage count: "Centers sampling mass at mu(s) across 6 stages".
CURRICULUM_NUM_STAGES = 6

#: Issue-pinned permanent probability floor: "epsilon >= 0.03".
CURRICULUM_PROBABILITY_FLOOR = 0.03

#: Kernel width (levels) of the stage distribution (D-V2-038).
CURRICULUM_KERNEL_SIGMA = 2.0

#: Curriculum levels are computed over the training split only (#45 §12.1).
CURRICULUM_SPLIT = "train"

#: The five issue-named physical/chemical difficulty axes (D-V2-037).
CURRICULUM_FEATURE_NAMES: Tuple[str, ...] = (
    "pocket_atoms",
    "rotatable_bonds",
    "rings",
    "pocket_charge_magnitude",
    "handle_rarity",
)

#: Issue-pinned milestone: advance when validation loss improves by >= 0.005.
VAL_IMPROVEMENT_MILESTONE = 0.005

#: Issue-pinned fallback patience: advance after 2 non-improving epochs.
STAGE_PATIENCE = 2

#: Schema version of the checkpointable curriculum state.
CURRICULUM_STATE_SCHEMA_VERSION = 1


class CurriculumError(ValueError):
    """Fail-closed curriculum error (never a silently degraded sampler)."""


# ---------------------------------------------------------------------------
# Difficulty ingestion and level assignment
# ---------------------------------------------------------------------------

def difficulty_features(dataset: Any) -> np.ndarray:
    """The ``[N, 5]`` per-item difficulty features of a supervised dataset.

    Ingests the five issue-named axes over the dataset's documented
    deterministic item table (``_items`` / ``_contexts`` — the contract
    ``SupervisedTrajectoryDataset`` defines and the #26 trainer already
    consumes).  Fails closed on any non-train split (constitution #45
    §12.1 — curriculum levels are a *training* device; validation/test
    distributions must stay pristine), on a missing item table, and on
    any non-finite or negative feature value.
    """
    split = getattr(dataset, "split", None)
    if split != CURRICULUM_SPLIT:
        raise CurriculumError(
            f"curriculum difficulty levels are computed on the train split "
            f"only (#45 s12.1); refusing split {split!r}"
        )
    items = getattr(dataset, "_items", None)
    contexts = getattr(dataset, "_contexts", None)
    if items is None or contexts is None:
        raise CurriculumError(
            "difficulty_features ingests a SupervisedTrajectoryDataset (the "
            "documented deterministic item table was not found on "
            f"{type(dataset).__name__})"
        )
    from rdkit.Chem import rdMolDescriptors  # local import: ingestion only

    # primary-handle-class frequency over the split's items (the rarity axis)
    class_counts: Counter = Counter()
    item_classes: List[Optional[str]] = []
    for context_index, view_index in items:
        handle_class: Optional[str] = None
        if view_index >= 0:
            handle_class = contexts[context_index].views[view_index].next_handle_class
        item_classes.append(handle_class)
        if handle_class is not None:
            class_counts[handle_class] += 1
    max_count = max(class_counts.values()) if class_counts else 0

    rows: List[List[float]] = []
    for (context_index, view_index), handle_class in zip(items, item_classes):
        context = contexts[context_index]
        if view_index >= 0:
            intermediate = context.views[view_index].intermediate_mol
            rotatable = int(rdMolDescriptors.CalcNumRotatableBonds(intermediate))
            rings = int(intermediate.GetRingInfo().NumRings())
        else:
            # the pre-seed decision: no intermediate exists yet
            rotatable, rings = 0, 0
        if handle_class is not None and max_count > 0:
            rarity = 1.0 - (class_counts[handle_class] / max_count)
        else:
            rarity = 0.0  # handle-free items sit at the common end (D-V2-037)
        pocket_charge = np.asarray(context.pocket_charge, dtype=np.float64)
        rows.append(
            [
                float(len(context.pocket_z)),
                float(rotatable),
                float(rings),
                float(np.abs(pocket_charge).sum()),
                float(rarity),
            ]
        )
    features = np.asarray(rows, dtype=np.float64)
    expected = (len(items), len(CURRICULUM_FEATURE_NAMES))
    if features.shape != expected:
        raise CurriculumError(
            f"difficulty features have shape {features.shape}, expected "
            f"{expected} (fail-closed)"
        )
    if not np.all(np.isfinite(features)):
        raise CurriculumError("non-finite difficulty features (fail-closed)")
    if np.any(features < 0.0):
        raise CurriculumError(
            "negative difficulty features — every axis is a count or a "
            "magnitude (fail-closed)"
        )
    return features


def standardize_features(features: np.ndarray) -> np.ndarray:
    """Z-score each feature column over the population (documented).

    A column with zero standard deviation contributes nothing (the
    documented degenerate-axis behaviour — an axis on which the split is
    uniform carries no difficulty information; D-V2-037).
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(CURRICULUM_FEATURE_NAMES):
        raise CurriculumError(
            f"features must be [N, {len(CURRICULUM_FEATURE_NAMES)}] over the "
            f"axes {CURRICULUM_FEATURE_NAMES}; got shape {features.shape}"
        )
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    safe_std = np.where(std > 0.0, std, 1.0)
    z = (features - mean) / safe_std
    return np.where(std > 0.0, z, 0.0)


def difficulty_scores(features: np.ndarray) -> np.ndarray:
    """The scalar difficulty score: the equal-weight mean of z-scored axes.

    Only the induced ranking matters for quantile binning (D-V2-037).
    """
    return standardize_features(features).mean(axis=1)


def quantile_levels(
    scores: Sequence[float], num_levels: int = CURRICULUM_NUM_LEVELS
) -> np.ndarray:
    """Balanced quantile level assignment (1..num_levels) over sorted scores.

    Every level receives ``N // num_levels`` or ``N // num_levels + 1``
    items — the "quantile-balanced" property holds exactly, including
    under score ties (stable sort, deterministic).  Fails closed when
    fewer than ``num_levels`` items exist (some level would be empty and
    could never be sampled).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise CurriculumError("scores must be a 1-D array of difficulty values")
    num_items = scores.size
    if num_items < int(num_levels):
        raise CurriculumError(
            f"{num_items} items cannot populate {num_levels} non-empty "
            "quantile levels (fail-closed — an empty level could never be "
            "sampled, violating the all-levels-per-epoch contract)"
        )
    order = np.argsort(scores, kind="stable")
    base, extra = divmod(num_items, int(num_levels))
    levels = np.empty(num_items, dtype=np.int64)
    start = 0
    for level in range(1, int(num_levels) + 1):
        count = base + (1 if level <= extra else 0)
        levels[order[start : start + count]] = level
        start += count
    return levels


# ---------------------------------------------------------------------------
# The 6-stage level distribution
# ---------------------------------------------------------------------------

def stage_center(
    stage: int,
    num_levels: int = CURRICULUM_NUM_LEVELS,
    num_stages: int = CURRICULUM_NUM_STAGES,
) -> float:
    """The stage's distribution centre ``mu(s)`` (linear 1 -> num_levels)."""
    stage = int(stage)
    if not 1 <= stage <= int(num_stages):
        raise CurriculumError(
            f"stage must be in [1, {num_stages}]; got {stage}"
        )
    if num_stages < 2:
        raise CurriculumError("num_stages must be >= 2")
    return 1.0 + (stage - 1) * (int(num_levels) - 1) / (int(num_stages) - 1)


def level_distribution(
    stage: int,
    *,
    num_levels: int = CURRICULUM_NUM_LEVELS,
    num_stages: int = CURRICULUM_NUM_STAGES,
    floor: float = CURRICULUM_PROBABILITY_FLOOR,
    kernel_sigma: float = CURRICULUM_KERNEL_SIGMA,
) -> np.ndarray:
    """The stage's probability vector over the levels (floor-exact).

    ``p = (1 - L * floor) * kernel + floor`` — a mixture with the uniform
    distribution: sums to 1 by construction and keeps *every* level at
    exactly ``floor`` or above, at every stage, including the extreme
    centres ``mu(1) = 1`` and ``mu(S) = L`` (D-V2-038).
    """
    if not (0.0 <= float(floor) < 1.0 / int(num_levels)):
        raise CurriculumError(
            f"floor must satisfy 0 <= floor < 1/num_levels = "
            f"{1.0 / int(num_levels):.6f}; got {floor}"
        )
    if float(kernel_sigma) <= 0.0:
        raise CurriculumError(f"kernel_sigma must be > 0; got {kernel_sigma}")
    center = stage_center(stage, num_levels, num_stages)
    levels = np.arange(1, int(num_levels) + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * ((levels - center) / float(kernel_sigma)) ** 2)
    kernel = kernel / kernel.sum()
    return (1.0 - int(num_levels) * float(floor)) * kernel + float(floor)


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

class CurriculumSampler(Sampler[int]):
    """The Approach C curriculum sampler over a supervised train split.

    Iterating yields **exactly** ``N = len(features)`` dataset indices per
    epoch (multinomial level draws from the stage distribution, uniform
    within-level picks, full shuffle), deterministically derived from
    ``(seed, stage, epoch)`` so an epoch re-iterates bit-identically —
    the property the trainer's exact-resume contract consumes.  The
    milestone machine (:meth:`observe_validation_loss`) advances the
    stage on validation improvement ``>= 0.005`` with a patience-2
    fallback, and the whole state round-trips through
    :meth:`state_dict` for checkpoint persistence (#45 §12.1).
    """

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        *,
        seed: int = 2026,
        num_levels: int = CURRICULUM_NUM_LEVELS,
        num_stages: int = CURRICULUM_NUM_STAGES,
        floor: float = CURRICULUM_PROBABILITY_FLOOR,
        kernel_sigma: float = CURRICULUM_KERNEL_SIGMA,
        milestone: float = VAL_IMPROVEMENT_MILESTONE,
        stage_patience: int = STAGE_PATIENCE,
    ) -> None:
        features_array = np.asarray(features, dtype=np.float64)
        if features_array.ndim != 2 or features_array.shape[1] != len(
            CURRICULUM_FEATURE_NAMES
        ):
            raise CurriculumError(
                f"features must be [N, {len(CURRICULUM_FEATURE_NAMES)}] over "
                f"the axes {CURRICULUM_FEATURE_NAMES}; got shape "
                f"{features_array.shape}"
            )
        if not np.all(np.isfinite(features_array)):
            raise CurriculumError("non-finite difficulty features (fail-closed)")
        num_levels = int(num_levels)
        num_stages = int(num_stages)
        if num_levels < 2:
            raise CurriculumError("num_levels must be >= 2")
        if num_stages < 2:
            raise CurriculumError("num_stages must be >= 2")
        if float(floor) < CURRICULUM_PROBABILITY_FLOOR:
            raise CurriculumError(
                f"the permanent per-level probability floor must be at least "
                f"{CURRICULUM_PROBABILITY_FLOOR} (issue #27); got {floor}"
            )
        if not (0.0 <= float(floor) < 1.0 / num_levels):
            raise CurriculumError(
                f"floor must satisfy 0 <= floor < 1/num_levels = "
                f"{1.0 / num_levels:.6f}; got {floor}"
            )
        if float(kernel_sigma) <= 0.0:
            raise CurriculumError(f"kernel_sigma must be > 0; got {kernel_sigma}")
        if float(milestone) <= 0.0:
            raise CurriculumError(f"milestone must be > 0; got {milestone}")
        if int(stage_patience) < 1:
            raise CurriculumError(f"stage_patience must be >= 1; got {stage_patience}")
        if not (0 <= int(seed) < 2**63):
            raise CurriculumError(f"seed out of range [0, 2^63-1]: {seed}")

        self.features = features_array
        self.seed = int(seed)
        self.num_levels = num_levels
        self.num_stages = num_stages
        self.floor = float(floor)
        self.kernel_sigma = float(kernel_sigma)
        self.milestone = float(milestone)
        self.stage_patience = int(stage_patience)

        self.scores = difficulty_scores(features_array)
        self.levels = quantile_levels(self.scores, num_levels)
        self._pools: List[np.ndarray] = [
            np.flatnonzero(self.levels == level) for level in range(1, num_levels + 1)
        ]
        if any(pool.size == 0 for pool in self._pools):
            raise CurriculumError(
                "an empty level pool exists (fail-closed — every level must "
                "be sampleable in every epoch)"
            )

        # the milestone-machine state (checkpointable)
        self.stage = 1
        self.epoch = 0
        self.best_val_loss: Optional[float] = None
        self.patience = 0
        self.transitions: List[Dict[str, Any]] = []
        self._epoch_level_counts: Dict[int, List[int]] = {}

    # -- construction from a supervised dataset ------------------------------

    @classmethod
    def from_dataset(cls, dataset: Any, **overrides: Any) -> "CurriculumSampler":
        """Build the sampler from a *train* supervised dataset's features."""
        return cls(difficulty_features(dataset), **overrides)

    # -- the epoch draw --------------------------------------------------------

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch of subsequent draws (fresh deterministic stream)."""
        epoch = int(epoch)
        if epoch < 0:
            raise CurriculumError(f"epoch must be >= 0; got {epoch}")
        self.epoch = epoch

    def _epoch_rng(self) -> np.random.Generator:
        # deterministic per-(seed, stage, epoch) stream — mirrors the
        # supervised dataset's per-(epoch, item) derivation
        return np.random.default_rng(
            (self.seed * 1_000_003 + self.stage * 10_007 + self.epoch * 100_003)
            % (2**63)
        )

    def level_probabilities(self) -> np.ndarray:
        """The active stage's probability vector over the levels."""
        return level_distribution(
            self.stage,
            num_levels=self.num_levels,
            num_stages=self.num_stages,
            floor=self.floor,
            kernel_sigma=self.kernel_sigma,
        )

    @property
    def last_level_counts(self) -> List[int]:
        """The level histogram of the most recent epoch's draw (logs)."""
        return list(self._epoch_level_counts.get(self.epoch, [0] * self.num_levels))

    def level_histogram(self, epoch: int) -> List[int]:
        """The recorded level histogram of one past epoch's draw."""
        return list(self._epoch_level_counts.get(int(epoch), [0] * self.num_levels))

    def __iter__(self):
        probabilities = self.level_probabilities()
        rng = self._epoch_rng()
        counts = rng.multinomial(self.num_samples, probabilities)
        drawn: List[int] = []
        for level in range(1, self.num_levels + 1):
            count = int(counts[level - 1])
            if count == 0:
                continue
            pool = self._pools[level - 1]
            picks = pool[rng.integers(0, pool.size, size=count)]
            drawn.extend(int(index) for index in picks)
        order = np.asarray(drawn, dtype=np.int64)
        rng.shuffle(order)
        self._epoch_level_counts[self.epoch] = [int(v) for v in counts]
        return iter(int(index) for index in order)

    def __len__(self) -> int:
        return self.num_samples

    # -- the milestone machine ---------------------------------------------------

    def observe_validation_loss(self, val_loss: float) -> bool:
        """Consume one epoch's validation loss; return whether stage advanced.

        Milestone rule (issue #27): advance when the loss improves on the
        running best by ``>= milestone`` (0.005); otherwise increment
        patience and advance after ``stage_patience`` (2) consecutive
        non-improving epochs.  The stage never regresses and never
        exceeds ``num_stages``; the running best only tightens.
        """
        val_loss = float(val_loss)
        if not math.isfinite(val_loss):
            raise CurriculumError(
                f"validation loss must be finite; got {val_loss} (fail-closed, "
                "#45 s8 — never a silently skipped epoch)"
            )
        advanced = False
        if self.best_val_loss is None:
            # the first observation is the baseline; no advancement
            self.best_val_loss = val_loss
            self.patience = 0
            return False
        improvement = self.best_val_loss - val_loss
        if improvement >= self.milestone:
            advanced = self._advance(reason="milestone", improvement=improvement)
            self.patience = 0
        else:
            self.patience += 1
            if self.patience >= self.stage_patience:
                advanced = self._advance(
                    reason="patience_fallback", improvement=improvement
                )
                self.patience = 0
        self.best_val_loss = min(self.best_val_loss, val_loss)
        return advanced

    def _advance(self, reason: str, improvement: float) -> bool:
        if self.stage >= self.num_stages:
            return False  # terminal stage: the curriculum is fully annealed
        self.stage += 1
        self.transitions.append(
            {
                "epoch": self.epoch,
                "from_stage": self.stage - 1,
                "to_stage": self.stage,
                "reason": reason,
                "improvement": float(improvement),
            }
        )
        return True

    # -- the checkpointable state --------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """The full continuation state (#45 s12.1 / s13.6 — JSON-able)."""
        return {
            "schema_version": CURRICULUM_STATE_SCHEMA_VERSION,
            "seed": self.seed,
            "num_levels": self.num_levels,
            "num_stages": self.num_stages,
            "floor": self.floor,
            "kernel_sigma": self.kernel_sigma,
            "milestone": self.milestone,
            "stage_patience": self.stage_patience,
            "stage": self.stage,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "patience": self.patience,
            "transitions": [dict(entry) for entry in self.transitions],
            "epoch_level_counts": {
                str(epoch): list(counts)
                for epoch, counts in sorted(self._epoch_level_counts.items())
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the exact continuation state (fail-closed on mismatch)."""
        if not isinstance(state, dict):
            raise CurriculumError(
                f"curriculum state must be a dict; got {type(state).__name__}"
            )
        if state.get("schema_version") != CURRICULUM_STATE_SCHEMA_VERSION:
            raise CurriculumError(
                "curriculum state schema version mismatch (expected "
                f"{CURRICULUM_STATE_SCHEMA_VERSION}, got "
                f"{state.get('schema_version')!r}) — refusing to resume"
            )
        config_keys = (
            "seed",
            "num_levels",
            "num_stages",
            "floor",
            "kernel_sigma",
            "milestone",
            "stage_patience",
        )
        for key in config_keys:
            if key not in state:
                raise CurriculumError(f"curriculum state lacks the key {key!r}")
        active = {
            "seed": self.seed,
            "num_levels": self.num_levels,
            "num_stages": self.num_stages,
            "floor": self.floor,
            "kernel_sigma": self.kernel_sigma,
            "milestone": self.milestone,
            "stage_patience": self.stage_patience,
        }
        for key, expected in active.items():
            if state[key] != expected:
                raise CurriculumError(
                    f"curriculum configuration mismatch on {key!r}: checkpoint "
                    f"has {state[key]!r}, active sampler has {expected!r} "
                    "(fail-closed — do not mix curricula)"
                )
        stage = int(state.get("stage", 0))
        if not 1 <= stage <= self.num_stages:
            raise CurriculumError(f"restored stage out of range [1, {self.num_stages}]: {stage}")
        patience = int(state.get("patience", 0))
        if not 0 <= patience < self.stage_patience:
            raise CurriculumError(f"restored patience out of range: {patience}")
        best = state.get("best_val_loss", None)
        if best is not None and not math.isfinite(float(best)):
            raise CurriculumError("restored best_val_loss is not finite (fail-closed)")
        self.stage = stage
        self.epoch = int(state.get("epoch", 0))
        self.patience = patience
        self.best_val_loss = None if best is None else float(best)
        self.transitions = [dict(entry) for entry in state.get("transitions", [])]
        self._epoch_level_counts = {
            int(epoch): [int(v) for v in counts]
            for epoch, counts in state.get("epoch_level_counts", {}).items()
        }
