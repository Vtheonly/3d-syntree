"""The pure-FP32 resilient trainer for the Tesla T4 profile (#26 TRAIN-01).

Clean-room V2 implementation of issue #26: a **static-hardware-profile**
supervised (Stage-1 behavioural cloning) trainer — frozen configuration,
strict FP32 (the #10 fail-closed precision engine), gradient
accumulation, fail-closed divergence handling, complete telemetry for
both host RAM and GPU VRAM (kept strictly separate, per the #26 remote
telemetry addendum), and atomic checkpoints carrying the full
continuation state that #43 (OPS-03) needs for session-safe resume.

**What was deliberately NOT ported from the legacy trainer.**  The
pre-V2 ``nan_shield.py``, batch-skipping loops, dynamic batch-size
auto-scaling, FP16 + ``GradScaler`` and silent precision fallbacks
(D-V2-005, the #10 T4 GradScaler collapse forensics) do not enter V2 at
all: this trainer **fails closed immediately** on any non-finite loss or
gradient (:class:`TrainingDivergenceError`) — a divergence is a bug to
fix, never a batch to skip (constitution #45 §8/§12.5: "Do not hide
NaNs by catching exceptions, replacing values, or skipping batches").

Static Profile A (``configs/train_colab_t4.json``; #45 §5.2 — dimensions
frozen in version-controlled configuration, never probed from VRAM):

* micro-batch size ``B = 12``;
* gradient accumulation steps ``6`` (effective batch 72);
* max pocket atoms ``700`` via documented centroid-distance pruning
  (applied on the data side, ``syntree/data/supervised.py``);
* strict FP32 — no autocast, no ``GradScaler`` (asserted against the
  ``RuntimeConfig`` of :func:`syntree.utils.hardware.configure_runtime_environment`);
* AdamW (lr ``3e-4``, weight decay ``1e-5``), gradient clipping norm 1.0.

**The R3 profile extension (``colab_t4_profile_a_r3``, owner-directed
2026-11-06, D-V2-050):** AdamW weight decay ``1e-2``; a cosine LR schedule
``3e-4 → 3e-5`` with a scale-aware warmup (a frozen fraction of a frozen
step horizon — never re-shaped by cross-session epoch extension,
D-V2-048); validation-based early stopping (patience 5, ``min_delta``
0.10 absolute in the measured ~12–16 validation-loss units) with
best-validation checkpoint selection (``best_state.pt``, same atomic
SHA-256 contract); and the anchor-margin monitor (warn −800 / graceful
stop −700 / hard guard −1000, degradation-increasing energy-like
convention — no pre-existing ``-1000`` guard was found in either branch,
and no Stage-1 quantity lives on that scale, so the thresholds ship in
the documented calibration-pending mode until an environment-side
energy source is wired; see :class:`AnchorMarginMonitor`).  Every R3
field defaults to its Profile-A value, and the Profile-A digest is
bit-identical to the pre-R3 digest, so all persisted Profile-A
checkpoints (``runs/stage1_t4_session*``) remain resumable.

Losses (Stage-1 behavioural cloning, #45 §12.1): discrete
cross-entropies over the strictly categorical laws (seed/anchor, STOP,
handle, family/reaction, synthon, parity) plus the circular von Mises
torsion loss over the 36-bin targets (:func:`torsion_circular_loss`,
#8/#24) — all pure dimensionless log-probability terms, exactly the
components the Trajectory Balance forward sum will consume in Stage 2.

Checkpoint contract (the #43 interface): model + optimizer + LR
scheduler state, global step, epoch, RNG states (torch/NumPy/Python via
:mod:`syntree.utils.seeding`), the data-progress state (epoch + batch
index + sampler epoch) and the profile hash — written atomically
(tmp + rename) with a SHA-256 sidecar, fail-closed on any component
mismatch at load.  Resume continues at the exact next sample without
silently skipping or repeating (#45 §13.6-7).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from syntree.models.policy import (
    POLICY_REACTION_IDS,
    SynTreePolicy,
    build_continuation_masks,
)
from syntree.models.torsion_head import torsion_circular_loss
from syntree.engine.curriculum import CurriculumSampler
from syntree.utils.hardware import RuntimeConfig, configure_runtime_environment
from syntree.utils.seeding import (
    capture_rng_state,
    restore_rng_state,
    rng_state_from_dict,
    rng_state_to_dict,
    seed_everything,
)

__all__ = [
    "TrainerProfile",
    "TrainingDivergenceError",
    "TrainerStateError",
    "AnchorMarginMonitor",
    "ANCHOR_MARGIN_OK",
    "ANCHOR_MARGIN_WARN",
    "ANCHOR_MARGIN_GRACEFUL_STOP",
    "ANCHOR_MARGIN_HARD_GUARD",
    "host_memory_telemetry",
    "gpu_memory_telemetry",
    "ResilientTrainer",
    "load_profile",
]

#: The profile configuration file of the supported production runtime.
DEFAULT_PROFILE_PATH = os.path.join("configs", "train_colab_t4.json")


class TrainingDivergenceError(RuntimeError):
    """Fail-closed divergence: a non-finite loss/gradient/parameter.

    Raised immediately — never a skipped batch, never a NaN replacement
    (#45 §8: "Fail loudly, preserve the failing example and logs,
    identify the mathematical/software cause, fix it, and add a
    regression test").
    """


class TrainerStateError(RuntimeError):
    """Fail-closed checkpoint/state error (missing or mismatched)."""


@dataclass(frozen=True)
class TrainerProfile:
    """The static training profile (Profile A — Tesla T4 FP32; R3 extension).

    Every field is frozen in version-controlled configuration; nothing is
    probed from free VRAM at runtime (#45 §5.2).

    **The R3 extension (``colab_t4_profile_a_r3``, owner-directed
    2026-11-06).**  Beyond Profile A, the R3 profile freezes: AdamW weight
    decay ``1e-2``, a cosine learning-rate schedule ``3e-4 → 3e-5`` with a
    scale-aware warmup (``warmup_fraction`` of the frozen horizon
    ``schedule_total_steps``), validation-based early stopping (patience 5,
    ``min_delta`` 0.10 in the *measured* validation-loss units — the
    #26 T-chain scale is ~12–16, so 0.10 ≈ 0.6–0.8 % relative),
    best-validation checkpoint selection (``best_state.pt``), and the
    anchor-margin monitor thresholds.  All R3 fields carry
    backward-compatible defaults (schedule off / stopping off / monitor
    off) so the frozen Profile A and its persisted checkpoints are bit-
    compatible with this class.
    """

    micro_batch_size: int = 12
    gradient_accumulation_steps: int = 6
    max_pocket_atoms: int = 700
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    precision: str = "fp32"
    seed: int = 2026
    max_epochs: int = 2
    num_workers: int = 0
    log_interval_steps: int = 10
    checkpoint_interval_steps: int = 200
    time_budget_seconds: Optional[int] = None
    profile_name: str = "colab_t4_profile_a"
    # -- the R3 extension (all defaults preserve Profile A exactly) --------
    #: ``"none"`` = constant Profile-A learning rate; ``"cosine_warmup"`` =
    #: the R3 schedule (per-step semantics → digest-pinned when active).
    lr_schedule: str = "none"
    #: The cosine floor (R3: 3e-5).  Must satisfy ``0 < min_lr < learning_rate``.
    min_lr: float = 3e-5
    #: Scale-aware warmup: ``ceil(warmup_fraction × schedule_total_steps)``
    #: optimizer steps ramping linearly ``0 → learning_rate``.
    warmup_fraction: float = 0.05
    #: The FROZEN cosine horizon in optimizer steps (R3: 252 = 12 epochs ×
    #: the measured 21 steps/epoch of the #18 census at B=12/accum 6).
    #: Deliberately NOT derived from ``max_epochs`` at runtime: #43/D-V2-048
    #: legalized cross-session epoch extension, so the schedule horizon is
    #: itself frozen config; steps beyond it hold at ``min_lr``.
    schedule_total_steps: int = 252
    #: Validation-based early stopping (R3).  ``0`` = disabled.  Patience is
    #: in *completed epochs without a ≥ min_delta validation improvement*.
    #: Run-length control (like ``max_epochs``), NOT per-step semantics —
    #: excluded from the digest per the D-V2-046 precedent.
    early_stopping_patience: int = 0
    #: Minimum absolute validation-loss improvement (same units as
    #: ``_validation_loss``; measured T4 scale ~12–16 → 0.10 ≈ 0.6–0.8 %).
    early_stopping_min_delta: float = 0.10
    #: The R3 anchor-margin monitor thresholds (owner-specified constants:
    #: warn −800 / graceful-stop −700 / hard guard −1000).  ``None`` = the
    #: monitor is fully off.  See :class:`AnchorMarginMonitor` for the
    #: documented convention and the D-V2-050 scale finding.
    anchor_margin_warn: Optional[float] = None
    anchor_margin_graceful_stop: Optional[float] = None
    anchor_margin_hard_guard: Optional[float] = None
    #: ``"none"`` = thresholds are recorded but no Stage-1 energy-scale
    #: source exists (loud calibration-pending telemetry, never silent —
    #: D-V2-050); ``"external"`` = the caller must supply
    #: ``anchor_margin_fn`` (fail-closed) and the monitor fully gates.
    anchor_margin_source: str = "none"

    #: Profile fields that define training semantics and therefore
    #: checkpoint compatibility (#26 frozen Profile A).  The operational
    #: session-scheduling knobs — ``time_budget_seconds``, ``num_workers``,
    #: ``log_interval_steps``, ``checkpoint_interval_steps`` and
    #: ``max_epochs`` — are deliberately EXCLUDED: #43 requires
    #: configurable time-aware checkpointing and cross-session
    #: continuation (a run whose epochs extend past one session's
    #: lifetime), so a checkpoint saved under one session budget or epoch
    #: bound must resume under another (D-V2-046/D-V2-048, caught live by
    #: the T-chain on 2026-11-05: a 40 s budget checkpoint refused a
    #: budgetless resume).  max_epochs bounds run LENGTH, not what a step
    #: IS; the per-step contract (batch/accum/precision/optimizer/seed)
    #: still fails closed.
    _DIGEST_FIELDS: ClassVar[tuple] = (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_pocket_atoms",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "precision",
        "seed",
        "profile_name",
    )
    #: R3 per-step scheduling semantics — hashed ONLY when a schedule is
    #: active, so the frozen Profile A digest (and every persisted
    #: Profile-A checkpoint, S-V2-21/22) is bit-identical to before, while
    #: any R3 schedule change still fails closed at resume.  Early
    #: stopping and the anchor-margin thresholds are run-LENGTH/monitoring
    #: control (they change WHEN a run stops, never what a step IS) and
    #: stay out of the digest per the D-V2-046 precedent.
    _R3_DIGEST_FIELDS: ClassVar[tuple] = (
        "lr_schedule",
        "min_lr",
        "warmup_fraction",
        "schedule_total_steps",
    )

    def __post_init__(self) -> None:
        if int(self.micro_batch_size) < 1:
            raise ValueError("micro_batch_size must be >= 1")
        if int(self.gradient_accumulation_steps) < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if int(self.max_pocket_atoms) < 1:
            raise ValueError("max_pocket_atoms must be >= 1")
        if not (float(self.learning_rate) > 0.0):
            raise ValueError("learning_rate must be > 0")
        if not (float(self.gradient_clip_norm) > 0.0):
            raise ValueError("gradient_clip_norm must be > 0")
        if self.precision != "fp32":
            raise ValueError(
                f"Profile A is strict FP32 (#45 s5.1); precision={self.precision!r} "
                "is rejected (no FP16/BF16/AMP exists anywhere in V2)"
            )
        if int(self.max_epochs) < 1:
            raise ValueError("max_epochs must be >= 1")
        # -- the R3 extension validation --------------------------------
        if self.lr_schedule not in ("none", "cosine_warmup"):
            raise ValueError(
                f"lr_schedule must be 'none' or 'cosine_warmup' (got "
                f"{self.lr_schedule!r})"
            )
        if self.lr_schedule == "cosine_warmup":
            if not (0.0 < float(self.min_lr) < float(self.learning_rate)):
                raise ValueError(
                    "cosine_warmup requires 0 < min_lr < learning_rate "
                    f"(got min_lr={self.min_lr!r}, "
                    f"learning_rate={self.learning_rate!r})"
                )
            if not (0.0 < float(self.warmup_fraction) < 1.0):
                raise ValueError(
                    "warmup_fraction must be in (0, 1) — the scale-aware "
                    f"warmup (got {self.warmup_fraction!r})"
                )
            if int(self.schedule_total_steps) < 1:
                raise ValueError("schedule_total_steps must be >= 1")
        if int(self.early_stopping_patience) < 0:
            raise ValueError("early_stopping_patience must be >= 0 (0 = off)")
        if float(self.early_stopping_min_delta) < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        _thresholds = (
            self.anchor_margin_warn,
            self.anchor_margin_graceful_stop,
            self.anchor_margin_hard_guard,
        )
        if self.anchor_margin_source not in ("none", "external"):
            raise ValueError(
                "anchor_margin_source must be 'none' or 'external' (got "
                f"{self.anchor_margin_source!r})"
            )
        if any(t is not None for t in _thresholds) and not all(
            t is not None for t in _thresholds
        ):
            raise ValueError(
                "the anchor-margin monitor requires all three thresholds "
                "(warn / graceful_stop / hard_guard) to be set together"
            )
        if all(t is not None for t in _thresholds):
            warn, stop, guard = (
                float(self.anchor_margin_warn),
                float(self.anchor_margin_graceful_stop),
                float(self.anchor_margin_hard_guard),
            )
            # Degradation-increasing energy-like convention (see
            # AnchorMarginMonitor): value rises toward 0 as anchoring
            # degrades, so the warn must sit BELOW the graceful stop, and
            # the hard guard (implausibly favorable) below both.
            if not (guard < warn < stop):
                raise ValueError(
                    "anchor-margin thresholds must satisfy "
                    f"hard_guard < warn < graceful_stop on the "
                    f"degradation-increasing convention (got guard={guard}, "
                    f"warn={warn}, stop={stop})"
                )
            if self.anchor_margin_source == "none":
                # thresholds recorded but no source: legal only as the
                # documented calibration-pending mode (D-V2-050) — the
                # trainer will state this loudly at fit() start.
                pass

    @property
    def effective_batch_size(self) -> int:
        return int(self.micro_batch_size) * int(self.gradient_accumulation_steps)

    def digest(self) -> str:
        """Stable SHA-256 of the frozen profile (checkpoint compatibility).

        Only :data:`TrainerProfile._DIGEST_FIELDS` (the training-semantics
        contract) are hashed.  Operational scheduling knobs are excluded so
        that resuming under a different session budget, worker count or
        telemetry cadence stays legal (#43 continuation contract); any
        change to the actual training semantics still fails closed.
        When a learning-rate schedule is ACTIVE
        (``lr_schedule != "none"``), the R3 schedule semantics
        (:data:`TrainerProfile._R3_DIGEST_FIELDS`) join the digest; with the
        default ``"none"`` schedule the digest is bit-identical to the
        pre-R3 Profile A digest, so every persisted Profile-A checkpoint
        (S-V2-21/22 T-chain, ``runs/stage1_t4_session*``) stays resumable.
        """
        payload = {name: getattr(self, name) for name in self._DIGEST_FIELDS}
        if self.lr_schedule != "none":
            payload.update(
                {name: getattr(self, name) for name in self._R3_DIGEST_FIELDS}
            )
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()


def load_profile(path: str = DEFAULT_PROFILE_PATH) -> TrainerProfile:
    """Load a ``TrainerProfile`` from a frozen JSON configuration file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return TrainerProfile(**data)


# ---------------------------------------------------------------------------
# The R3 anchor-margin monitor
# ---------------------------------------------------------------------------

#: Monitor verdicts, in escalation order.
ANCHOR_MARGIN_OK = "ok"
ANCHOR_MARGIN_WARN = "warn"
ANCHOR_MARGIN_GRACEFUL_STOP = "graceful_stop"
ANCHOR_MARGIN_HARD_GUARD = "hard_guard"


@dataclass(frozen=True)
class AnchorMarginMonitor:
    """The R3 anchor-margin run-health monitor (fail-closed, loud).

    **Convention (D-V2-050, recorded 2026-11-06).**  The monitored
    quantity is an energy-like *anchor margin* where larger-magnitude
    negative values mean better anchoring and degradation moves the value
    UP toward zero:

    * ``value >= warn`` — degradation warning (telemetry record).
    * ``value >= graceful_stop`` — controlled graceful stop at the next
      batch boundary (``stop_reason="anchor_margin_graceful_stop"``).
    * ``value <= hard_guard`` — implausibly favorable margin (a
      degenerate/numerically broken anchoring, e.g. a physically
      impossible energy): fail-closed :class:`TrainingDivergenceError`.

    Hence the required ordering ``hard_guard < warn < graceful_stop``
    (the owner's R3 constants: ``-1000 < -800 < -700``).

    **Verified finding (D-V2-050):** no pre-existing ``-1000`` anchor
    guard exists anywhere in the V2 or legacy ``main`` codebases (both
    searched this session), and no Stage-1 supervised quantity lives on
    this energy scale (measured loss scale ~12–16; seed/anchor
    log-probabilities ~0–25).  The thresholds are therefore carried as
    owner-specified constants in the R3 profile with
    ``anchor_margin_source="none"`` — the monitor records its
    configuration loudly (calibration-pending telemetry, never a silent
    gate) until an environment-side energy source is wired via
    ``anchor_margin_source="external"`` + ``anchor_margin_fn``.
    """

    warn: float
    graceful_stop: float
    hard_guard: float

    def classify(self, value: float) -> str:
        """Classify one margin value (escalation order: guard > stop > warn)."""
        if not math.isfinite(value):
            # A non-finite margin is a divergence — never skipped (#45 s8).
            raise TrainingDivergenceError(
                f"non-finite anchor-margin value ({value}) — failing closed"
            )
        if value <= self.hard_guard:
            return ANCHOR_MARGIN_HARD_GUARD
        if value >= self.graceful_stop:
            return ANCHOR_MARGIN_GRACEFUL_STOP
        if value >= self.warn:
            return ANCHOR_MARGIN_WARN
        return ANCHOR_MARGIN_OK

    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "warn": float(self.warn),
            "graceful_stop": float(self.graceful_stop),
            "hard_guard": float(self.hard_guard),
        }



# ---------------------------------------------------------------------------
# Telemetry (host RAM and GPU VRAM strictly separate)
# ---------------------------------------------------------------------------

def host_memory_telemetry() -> Dict[str, Optional[float]]:
    """Host (system) RAM telemetry in MiB — never GPU memory.

    Uses ``psutil`` when installed, otherwise ``/proc/meminfo`` (Linux),
    with ``None`` marking unavailable fields.  Host RAM and GPU VRAM are
    separate resource pools and are never conflated (#45 §28.4).
    """
    out: Dict[str, Optional[float]] = {
        "host_total_mib": None,
        "host_available_mib": None,
        "host_used_mib": None,
        "process_rss_mib": None,
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        out["host_total_mib"] = round(vm.total / 1024 ** 2, 1)
        out["host_available_mib"] = round(vm.available / 1024 ** 2, 1)
        out["host_used_mib"] = round(vm.used / 1024 ** 2, 1)
        out["process_rss_mib"] = round(psutil.Process().memory_info().rss / 1024 ** 2, 1)
        return out
    except Exception:  # noqa: BLE001 - documented /proc fallback
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            info = {}
            for line in handle:
                key, _, value = line.partition(":")
                info[key.strip()] = float(value.strip().split()[0]) / 1024.0
        out["host_total_mib"] = round(info.get("MemTotal", 0.0), 1) or None
        available = info.get("MemAvailable", info.get("MemFree", 0.0))
        out["host_available_mib"] = round(available, 1) or None
        out["host_used_mib"] = round(
            info.get("MemTotal", 0.0) - available, 1
        ) or None
    except OSError:
        pass
    return out


def gpu_memory_telemetry(device: Optional[torch.device] = None) -> Dict[str, Optional[float]]:
    """GPU VRAM telemetry in MiB — never host memory.

    PyTorch process counters (allocated/reserved and their peaks) plus
    the runtime-level free/total pair (``mem_get_info``).  ``None``
    fields mark a non-CUDA runtime (the CPU dev sandbox reports no GPU
    numbers at all — an honest absence, never an inferred value).
    """
    if not torch.cuda.is_available():
        return {
            "gpu_name": None,
            "gpu_allocated_mib": None,
            "gpu_reserved_mib": None,
            "gpu_peak_allocated_mib": None,
            "gpu_peak_reserved_mib": None,
            "gpu_free_mib": None,
            "gpu_total_mib": None,
        }
    dev = device if device is not None else torch.device("cuda")
    free, total = torch.cuda.mem_get_info(dev)
    return {
        "gpu_name": torch.cuda.get_device_name(dev),
        "gpu_allocated_mib": round(torch.cuda.memory_allocated(dev) / 1024 ** 2, 1),
        "gpu_reserved_mib": round(torch.cuda.memory_reserved(dev) / 1024 ** 2, 1),
        "gpu_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated(dev) / 1024 ** 2, 1
        ),
        "gpu_peak_reserved_mib": round(
            torch.cuda.max_memory_reserved(dev) / 1024 ** 2, 1
        ),
        "gpu_free_mib": round(free / 1024 ** 2, 1),
        "gpu_total_mib": round(total / 1024 ** 2, 1),
    }


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------

@dataclass
class _Progress:
    """The resumable data-progress state (#45 §13.7)."""

    epoch: int = 0
    batch_in_epoch: int = 0
    global_step: int = 0


class ResilientTrainer:
    """The static-profile Stage-1 trainer (#26 TRAIN-01).

    Usage (the canonical Stage-1 entry point behind
    ``scripts/train_stage1.py``)::

        trainer = ResilientTrainer(policy, profile, run_dir=..., device=...,
                                   catalog_block_handle_classes=...,
                                   catalog_seed_rows=...)
        trainer.fit(train_loader, val_loader=...)
    """

    def __init__(
        self,
        policy: SynTreePolicy,
        profile: TrainerProfile,
        *,
        run_dir: str,
        device: Optional[torch.device] = None,
        catalog_block_handle_classes: Optional[torch.Tensor] = None,
        catalog_seed_rows: Optional[torch.Tensor] = None,
        available_block_classes: Optional[Sequence[str]] = None,
        curriculum: Optional[CurriculumSampler] = None,
        anchor_margin_fn: Optional[Callable[[], float]] = None,
        resume: bool = False,
    ) -> None:
        self.profile = profile
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        # -- the strict hardware gate (#10 MATH-04) -------------------------
        self.runtime: RuntimeConfig = configure_runtime_environment(
            device=device,
            mixed_precision=profile.precision,
        )
        if self.runtime.precision != "fp32" or self.runtime.grad_scaler or self.runtime.autocast_enabled:
            raise TrainerStateError(
                "the hardware gate did not enforce strict FP32 without "
                f"GradScaler/autocast (got {self.runtime.summary()!r}) — "
                "refusing to train (#45 s5.1)"
            )
        self.device = torch.device(self.runtime.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        # -- determinism -----------------------------------------------------
        self.seed_state = seed_everything(profile.seed, device=self.device)

        self.policy = policy.to(device=self.device, dtype=torch.float32)
        self.policy.train()
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=float(profile.learning_rate),
            weight_decay=float(profile.weight_decay),
        )
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if profile.lr_schedule == "cosine_warmup":
            self.scheduler = self._build_cosine_warmup_scheduler()

        # -- the R3 anchor-margin monitor (D-V2-050) -------------------------
        # thresholds set + source "external": full gating, anchor_margin_fn
        # required at fit() (fail-closed).  thresholds set + source "none":
        # the documented calibration-pending mode — loud telemetry, no gate.
        self.anchor_margin_monitor: Optional[AnchorMarginMonitor] = None
        if profile.anchor_margin_warn is not None:
            self.anchor_margin_monitor = AnchorMarginMonitor(
                warn=float(profile.anchor_margin_warn),
                graceful_stop=float(profile.anchor_margin_graceful_stop),
                hard_guard=float(profile.anchor_margin_hard_guard),
            )
        self._anchor_margin_fn = anchor_margin_fn

        # -- the R3 early-stopping state (run-control; restore-if-present) ---
        self._best_val_loss: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._epochs_since_improvement: int = 0

        # -- catalog surfaces for the legality masks ------------------------
        if catalog_block_handle_classes is not None:
            self.catalog_block_handle_classes = (
                catalog_block_handle_classes.to(torch.int64).to(self.device)
            )
        else:
            self.catalog_block_handle_classes = None
        if catalog_seed_rows is not None:
            self.catalog_seed_rows = catalog_seed_rows.to(torch.bool).to(self.device)
        else:
            self.catalog_seed_rows = None
        self.available_block_classes = available_block_classes

        # -- the #27 Approach C curriculum (optional; train-split only) -----
        self.curriculum = curriculum

        self.progress = _Progress()
        self.history: List[Dict[str, Any]] = []
        self._rng_snapshot = capture_rng_state()
        self._started = time.monotonic()
        if resume:
            self.load_checkpoint(self.checkpoint_path)

    # -- paths ---------------------------------------------------------------

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, "trainer_state.pt")

    @property
    def best_state_path(self) -> str:
        """The R3 best-validation checkpoint (best_state.pt)."""
        return os.path.join(self.run_dir, "best_state.pt")

    def _build_cosine_warmup_scheduler(
        self,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """The R3 cosine schedule ``learning_rate → min_lr`` with a
        scale-aware warmup, as a :class:`~torch.optim.lr_scheduler.LambdaLR`
        over the AdamW base learning rate.

        * warmup steps ``= ceil(warmup_fraction × schedule_total_steps)``
          (the "scale-aware" form: the ramp length is a frozen fraction
          of the frozen horizon — it scales with the run, never with the
          session);
        * linear ``0 → learning_rate`` ramp during warmup (step 0 uses
          ``1/warmup`` of the base rate so the very first update is
          non-zero);
        * cosine decay ``learning_rate → min_lr`` over the remaining
          horizon;
        * steps beyond the horizon hold at ``min_lr`` (D-V2-048: the
          horizon is frozen config, so cross-session epoch extension
          never re-shapes the schedule).
        """
        profile = self.profile
        total = max(1, int(profile.schedule_total_steps))
        warmup = max(1, math.ceil(float(profile.warmup_fraction) * total))
        base = float(profile.learning_rate)
        floor = float(profile.min_lr)
        span = base - floor

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            if progress >= 1.0:
                return floor / base
            decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return (floor + span * decay) / base

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # -- losses ----------------------------------------------------------------

    def _seed_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        states = batch["state"].to(self.device)
        seed_rows = batch["seed_row"].to(self.device)
        anchor_idx = batch["anchor_idx"].to(self.device)
        backbone = self.policy.backbone(states)
        anchor_pos, anchor_batch, assignment, counts = self.policy.anchor_candidates(
            states.pocket_pos, states.pocket_z, states.pocket_charge,
            states.pocket_batch, states.num_graphs,
        )
        del anchor_pos
        features = self.policy.anchor_features(backbone, assignment, anchor_batch)
        # flat anchor row of each graph's local selection
        offsets = torch.zeros_like(counts)
        offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
        selected_anchor_row = offsets + anchor_idx
        seed_mask = None
        if self.catalog_seed_rows is not None:
            # the legality-mask convention is True = ILLEGAL
            # (apply_logit_mask); catalog_seed_rows marks eligibility
            seed_mask = (~self.catalog_seed_rows).unsqueeze(0).expand(
                states.num_graphs, -1
            )
        out = self.policy.forward_seed_anchor(
            backbone,
            features,
            anchor_batch,
            None,
            selected_anchor_row=selected_anchor_row,
            seed_mask=seed_mask,
            selected_seed=seed_rows,
        )
        return -(out.anchor_selected_log_prob + out.seed_selected_log_prob).mean()

    def _stop_step_loss(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        states = batch["state"].to(self.device)
        pool = {
            "handle_descriptors": batch["handle_descriptors"].to(self.device),
            "handle_pos": batch["handle_pos"].to(self.device),
            "handle_graph": batch["handle_graph"].to(self.device),
            "handle_atom": batch["handle_atom"].to(self.device),
        }
        stop_target = batch["stop_target"].to(self.device)
        if pool["handle_descriptors"].shape[0] == 0:
            # empty handle pools: STOP is structurally forced and the #22
            # head refuses empty pools by design — no stop loss attaches to
            # forced rows (the masked law is deterministic; D-V2-033).
            if stage == "stop" and not bool(batch["forced"].all()):
                raise TrainerStateError(
                    "an empty pool appeared on a non-forced stop row "
                    "(data inconsistency — fail closed)"
                )
            if stage == "step":
                raise TrainerStateError(
                    "a step row has an empty handle pool (the primary handle "
                    "must exist — data inconsistency, fail closed)"
                )
            return torch.zeros((), device=self.device)
        state_out = self.policy.forward_state(
            states,
            pool["handle_descriptors"],
            pool["handle_pos"],
            pool["handle_graph"],
            pool["handle_atom"],
        )
        force_stop = None
        if stage == "stop":
            forced = batch["forced"].to(self.device)
            if bool(forced.any()):
                force_stop = forced
        stop_out = self.policy.forward_stop(state_out.context, force_stop)
        loss = torch.nn.functional.nll_loss(
            stop_out.log_probs, stop_target.long()
        )
        if stage == "step":
            # the continuation components (teacher-forced)
            reaction_idx = batch["reaction_idx"].to(self.device)
            synthon_row = batch["synthon_row"].to(self.device)
            parity = batch["parity"].to(self.device)
            parity_required = batch["parity_required"].to(self.device)
            torsion_bin = batch["torsion_bin"].to(self.device)
            family_mask, reaction_mask = build_continuation_masks(
                batch["active_handle_classes"],
                self.available_block_classes,
            )
            continue_out = self.policy.forward_continue(
                state_out.context,
                reaction_idx,
                synthon_row,
                parity,
                torsion_bin,
                parity_required,
                family_mask.to(self.device),
                reaction_mask.to(self.device),
                self.catalog_block_handle_classes,
            )
            loss = loss + (
                -continue_out.family.selected_log_prob
                - continue_out.reaction.selected_log_prob
                - continue_out.synthon.selected_log_prob
                - continue_out.parity.selected_log_prob
            ).mean() + torsion_circular_loss(
                continue_out.torsion.mu,
                continue_out.torsion.kappa,
                torsion_bin,
            )
            # the handle component: -log P(selected handle) over the pool law
            handle_out = state_out.handle
            handle_row = batch["handle_row"].to(self.device)
            loss = loss - handle_out.log_probs.index_select(0, handle_row).mean()
        return loss

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """The per-stage losses of one collated supervised batch."""
        parts: Dict[str, torch.Tensor] = {}
        if batch.get("seed") is not None:
            parts["seed"] = self._seed_loss(batch["seed"])
        if batch.get("step") is not None:
            parts["step"] = self._stop_step_loss(batch["step"], stage="step")
        if batch.get("stop") is not None:
            parts["stop"] = self._stop_step_loss(batch["stop"], stage="stop")
        if not parts:
            raise TrainerStateError("a collated batch carried no stage sub-batch")
        total = torch.stack(list(parts.values())).sum()
        parts["total"] = total
        return parts

    # -- the fail-closed guards -----------------------------------------------

    @staticmethod
    def _assert_finite(tensor: torch.Tensor, what: str) -> None:
        if not bool(torch.isfinite(tensor).all()):
            raise TrainingDivergenceError(
                f"non-finite {what} detected ({tensor.dtype} tensor with "
                "non-finite values) — failing closed immediately: no batch "
                "skipping, no NaN replacement, no silent fallback (#45 s8, "
                "#26 'remove nan_shield.py and batch-skipping loops')"
            )

    # -- telemetry ----------------------------------------------------------

    def telemetry(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """One machine-readable telemetry record (#26 telemetry addendum).

        Host RAM and GPU VRAM are separate sections and are never inferred
        from one another; the measurement source of every field is stated
        in its name.
        """
        record: Dict[str, Any] = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile": self.profile.profile_name,
            "epoch": self.progress.epoch,
            "global_step": self.progress.global_step,
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "learning_rate": float(
                self.optimizer.param_groups[0]["lr"]
            ),
            "runtime": self.runtime.summary(),
            "host_memory": host_memory_telemetry(),
            "gpu_memory": gpu_memory_telemetry(self.device
                                               if str(self.device).startswith("cuda")
                                               else None),
        }
        if batch is not None:
            for stage in ("seed", "step", "stop"):
                sub = batch.get(stage)
                if sub is None:
                    continue
                states = sub["state"]
                record.setdefault("batch", {})[stage] = {
                    "num_graphs": int(states.num_graphs),
                    "num_pocket_atoms": int(states.pocket_pos.shape[0]),
                    "num_ligand_atoms": int(states.ligand_pos.shape[0]),
                    "num_handles": int(sub["handle_descriptors"].shape[0])
                    if stage != "seed" else 0,
                }
        return record

    def _log(self, record: Dict[str, Any]) -> None:
        path = os.path.join(self.run_dir, "telemetry.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    # -- checkpoints ----------------------------------------------------------

    def save_checkpoint(self, path: Optional[str] = None, *, extra: Optional[Dict[str, Any]] = None) -> str:
        """Atomically persist the full continuation state (#43 interface).

        The payload is written atomically (tmp + rename) with a separate
        SHA-256 sidecar file; ``load_checkpoint`` re-verifies the digest
        before restoring anything.
        """
        path = path or self.checkpoint_path
        payload = {
            "profile": asdict(self.profile),
            "profile_digest": self.profile.digest(),
            "model": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "progress": asdict(self.progress),
            "history": self.history,
            "rng": rng_state_to_dict(self._rng_snapshot),
            "runtime_summary": self.runtime.summary(),
            "curriculum": (
                self.curriculum.state_dict() if self.curriculum is not None else None
            ),
            # the R3 early-stopping run-control state (best/validation
            # tracking) — restore-if-present, never required (a fresh run
            # has none; the state is control, not per-step semantics)
            "early_stopping": self._early_stopping_state_dict(),
            "extra": extra or {},
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        digest = hashlib.sha256()
        with open(tmp, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        os.replace(tmp, path)
        sidecar = path + ".sha256"
        tmp_sidecar = sidecar + ".tmp"
        with open(tmp_sidecar, "w", encoding="utf-8") as handle:
            handle.write(digest.hexdigest() + "\n")
        os.replace(tmp_sidecar, sidecar)
        return path

    def load_checkpoint(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Restore the exact continuation state (fail-closed on mismatch)."""
        path = path or self.checkpoint_path
        if not os.path.exists(path):
            raise TrainerStateError(f"checkpoint not found: {path}")
        sidecar = path + ".sha256"
        if not os.path.exists(sidecar):
            raise TrainerStateError(
                "checkpoint sidecar missing (integrity unverifiable) — "
                "refusing to resume"
            )
        expected = open(sidecar, "r", encoding="utf-8").read().strip()
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise TrainerStateError(
                "checkpoint integrity check failed (SHA-256 mismatch) — "
                "refusing to resume from a corrupt file"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["profile_digest"] != self.profile.digest():
            raise TrainerStateError(
                "checkpoint profile digest does not match the active profile "
                "(frozen-profile incompatibility — do not mix profiles)"
            )
        self.policy.load_state_dict(payload["model"])
        self.policy.to(device=self.device, dtype=torch.float32)
        self.optimizer.load_state_dict(payload["optimizer"])
        # the R3 LR-schedule state: symmetric fail-closed — a checkpoint
        # saved under an active schedule must resume under the same
        # schedule (the profile digest already pins this, but a silent
        # drop of scheduler state is still refused loudly)
        checkpoint_scheduler = payload.get("scheduler", None)
        if checkpoint_scheduler is not None and self.scheduler is None:
            raise TrainerStateError(
                "the checkpoint carries LR-scheduler state while no "
                "schedule is active (wiring mismatch — fail-closed, R3)"
            )
        if self.scheduler is not None:
            if checkpoint_scheduler is None:
                raise TrainerStateError(
                    "an active LR schedule cannot resume a checkpoint that "
                    "carries no scheduler state (fail-closed, R3)"
                )
            self.scheduler.load_state_dict(checkpoint_scheduler)
        self.progress = _Progress(**payload["progress"])
        self.history = list(payload["history"])
        restore_rng_state(rng_state_from_dict(payload["rng"]))
        # the R3 early-stopping run-control state: restore-if-present
        # (control state, not per-step semantics — a fresh run carries
        # none, and patience/min_delta may legitimately differ on resume)
        self._load_early_stopping_state(payload.get("early_stopping", None))
        # the sampler/curriculum state (#45 s12.1/s13.6): a symmetric
        # fail-closed check — a curriculum run must resume its exact
        # curriculum state, and a non-curriculum run must not silently
        # ingest one
        curriculum_state = payload.get("curriculum", None)
        if self.curriculum is not None:
            if not curriculum_state:
                raise TrainerStateError(
                    "the checkpoint carries no curriculum state while a "
                    "curriculum sampler is active — refusing to silently "
                    "restart the curriculum from stage 1 (fail-closed, #27)"
                )
            self.curriculum.load_state_dict(curriculum_state)
        elif curriculum_state:
            raise TrainerStateError(
                "the checkpoint carries curriculum state but no curriculum "
                "sampler is active (wiring mismatch — fail-closed, #27)"
            )
        return payload.get("extra", {})

    # -- the training loop ------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run the Stage-1 training loop (static profile, fail-closed).

        Supports exact resume: pass a ``DataLoader`` whose dataset is the
        same ``SupervisedTrajectoryDataset`` (identical item order — the
        dataset's item table is deterministic) and ``epoch``/
        ``batch_in_epoch`` continue from the persisted progress without
        skipping or repeating samples (#45 §13.7).  ``time_budget_seconds``
        triggers a controlled graceful stop at a batch boundary with a
        final checkpoint (#43 contract).
        """
        if getattr(train_loader.dataset, "split", None) != "train":
            raise TrainerStateError(
                "the train loader's dataset must be the train split "
                "(augmentation is train-only; fail-closed)"
            )
        # -- the R3 wiring checks (fail-closed, loud) -----------------------
        if self._early_stopping_active() and val_loader is None:
            raise TrainerStateError(
                "the R3 early stopping requires a validation loader "
                "(fail-closed)"
            )
        if (
            val_loader is not None
            and getattr(val_loader.dataset, "split", None) is not None
            and getattr(val_loader.dataset, "split", None) != "val"
        ):
            raise TrainerStateError(
                "the validation signal must come from the val split "
                "(fail-closed)"
            )
        if self.anchor_margin_monitor is not None:
            if self.profile.anchor_margin_source == "external":
                if self._anchor_margin_fn is None:
                    raise TrainerStateError(
                        "the anchor-margin monitor is armed "
                        "(anchor_margin_source='external') but no "
                        "anchor_margin_fn was supplied — refusing to run "
                        "ungated (fail-closed, D-V2-050)"
                    )
            else:
                # source == "none": the documented calibration-pending mode
                # — recorded LOUDLY, never a silent gate (D-V2-050)
                self._log(
                    {
                        "phase": "anchor_margin",
                        "anchor_margin_mode": "calibration_pending",
                        "note": (
                            "thresholds recorded but inactive: no Stage-1 "
                            "energy-scale source exists; wire "
                            "anchor_margin_source='external' + "
                            "anchor_margin_fn to gate (D-V2-050)"
                        ),
                        **self.anchor_margin_monitor.thresholds,
                    }
                )
        if self.curriculum is not None:
            if val_loader is None:
                raise TrainerStateError(
                    "the curriculum requires a validation loader — stage "
                    "advancement is validation-milestone-driven (#27; "
                    "fail-closed)"
                )
            if getattr(val_loader.dataset, "split", None) != "val":
                raise TrainerStateError(
                    "the curriculum's validation signal must come from the "
                    "val split (fail-closed)"
                )
            if train_loader.sampler is not self.curriculum:
                raise TrainerStateError(
                    "the train loader must be wired with the curriculum "
                    "sampler (DataLoader(dataset, sampler=curriculum, ...)); "
                    "got a different sampler (fail-closed, #27)"
                )
        deadline = (
            time.monotonic() + float(self.profile.time_budget_seconds)
            if self.profile.time_budget_seconds is not None
            else None
        )
        start_epoch = self.progress.epoch
        stop_reason = "max_epochs_reached"
        self.policy.train()
        for epoch in range(start_epoch, self.profile.max_epochs):
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = "time_budget"
                break
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if self.curriculum is not None:
                self.curriculum.set_epoch(epoch)
            self.progress.epoch = epoch
            batches = list(train_loader)
            for batch_index, batch in enumerate(batches):
                if batch_index < self.progress.batch_in_epoch:
                    continue  # exact resume: skip already-consumed batches
                if deadline is not None and time.monotonic() >= deadline:
                    stop_reason = "time_budget"
                    break
                batch = self._to_device(batch)
                losses = self.compute_loss(batch)
                loss = losses["total"] / self.profile.gradient_accumulation_steps
                self._assert_finite(loss.detach(), "loss")
                loss.backward()
                self.progress.batch_in_epoch = batch_index + 1
                if (batch_index + 1) % self.profile.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                self._rng_snapshot = capture_rng_state()
                self.progress.global_step += 1
                self.history.append(
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "step": self.progress.global_step,
                        **{
                            name: float(value.detach())
                            for name, value in losses.items()
                        },
                    }
                )
                if self.progress.global_step % self.profile.log_interval_steps == 0:
                    self._log(self.telemetry(batch))
                    anchor_verdict = self._check_anchor_margin()
                    if anchor_verdict == ANCHOR_MARGIN_GRACEFUL_STOP:
                        stop_reason = "anchor_margin_graceful_stop"
                        break
                if (
                    self.profile.checkpoint_interval_steps
                    and self.progress.global_step % self.profile.checkpoint_interval_steps == 0
                ):
                    self.save_checkpoint()
            else:
                # epoch completed on schedule: flush a trailing partial
                # accumulation, reset the in-epoch cursor and advance the
                # epoch counter PAST the completed epoch.  The advance is
                # what makes a resumed COMPLETED run a no-op instead of a
                # silent re-train of the final epoch (D-V2-047, caught live
                # by the T-chain on 2026-11-05: a step-42 max_epochs
                # checkpoint resumed and silently re-ran epoch 1 to step 63).
                self._optimizer_step()
                self.progress.batch_in_epoch = 0
                self.progress.epoch = epoch + 1
                # -- the R3 end-of-epoch validation pass ---------------------
                # one validation sweep per completed epoch feeds both the
                # #27 curriculum milestones and the R3 early stopping /
                # best-state selection (never two sweeps).
                validate_now = val_loader is not None and (
                    self.curriculum is not None or self._early_stopping_active()
                )
                if validate_now:
                    val_loss = self._validation_loss(val_loader)
                    if self.curriculum is not None:
                        advanced = self.curriculum.observe_validation_loss(val_loss)
                        self._log(
                            {
                                "phase": "curriculum",
                                **self.telemetry(),
                                "validation_loss": val_loss,
                                "stage": self.curriculum.stage,
                                "stage_advanced": advanced,
                                "level_counts": self.curriculum.last_level_counts,
                                "level_probabilities": [
                                    round(float(value), 6)
                                    for value in self.curriculum.level_probabilities()
                                ],
                            }
                        )
                    if self._early_stopping_active():
                        record = self._observe_validation(val_loss)
                        self._log(
                            {"phase": "validation", **self.telemetry(), **record}
                        )
                        if record["early_stop"]:
                            stop_reason = "early_stopping"
                            break
                elif val_loader is not None and stop_reason == "max_epochs_reached":
                    self._log({"phase": "validation", **self.telemetry()})
                continue
            break  # time-budget break inside the batch loop
        self.save_checkpoint(extra={"stop_reason": stop_reason})
        # ``progress.epoch`` counts COMPLETED epochs (it advances past an
        # epoch only when that epoch finishes on schedule; a mid-epoch
        # stop leaves it at the in-progress epoch index = completed count).
        return {
            "stop_reason": stop_reason,
            "epochs_completed": self.progress.epoch,
            "global_step": self.progress.global_step,
            "final_losses": self.history[-1] if self.history else None,
            "checkpoint": self.checkpoint_path,
            "best_state": (
                self.best_state_path
                if self._best_epoch is not None
                else None
            ),
            "early_stopping": self._early_stopping_state_dict(),
            "curriculum": (
                {
                    "stage": self.curriculum.stage,
                    "transitions": len(self.curriculum.transitions),
                }
                if self.curriculum is not None
                else None
            ),
        }

    # -- the R3 early stopping / best-state machinery -------------------------

    def _early_stopping_active(self) -> bool:
        return int(self.profile.early_stopping_patience) >= 1

    def _early_stopping_state_dict(self) -> Optional[Dict[str, Any]]:
        """The best/validation tracking state (None before any sweep)."""
        if self._best_val_loss is None:
            return None
        return {
            "best_val_loss": self._best_val_loss,
            "best_epoch": self._best_epoch,
            "epochs_since_improvement": self._epochs_since_improvement,
            "min_delta": float(self.profile.early_stopping_min_delta),
            "patience": int(self.profile.early_stopping_patience),
        }

    def _load_early_stopping_state(
        self, state: Optional[Dict[str, Any]]
    ) -> None:
        """Restore-if-present (run-control state, never required)."""
        if not state:
            self._best_val_loss = None
            self._best_epoch = None
            self._epochs_since_improvement = 0
            return
        self._best_val_loss = (
            float(state["best_val_loss"])
            if state.get("best_val_loss") is not None
            else None
        )
        self._best_epoch = state.get("best_epoch")
        self._epochs_since_improvement = int(
            state.get("epochs_since_improvement", 0)
        )

    def _observe_validation(self, val_loss: float) -> Dict[str, Any]:
        """One end-of-epoch validation observation (R3).

        Absolute ``min_delta`` in ``_validation_loss`` units (the measured
        T4 scale is ~12–16, so the R3 0.10 ≈ 0.6–0.8 % relative — a
        deliberately conservative noise floor); the relative improvement
        is recorded alongside so both readings are always visible.
        On every improvement the best model is atomically persisted as
        ``best_state.pt`` (same SHA-256 sidecar contract as the
        continuation checkpoint).
        """
        improved = (
            self._best_val_loss is None
            or val_loss
            <= self._best_val_loss - float(self.profile.early_stopping_min_delta)
        )
        if improved:
            previous = self._best_val_loss
            self._best_val_loss = val_loss
            self._best_epoch = self.progress.epoch - 1  # the just-completed epoch
            self._epochs_since_improvement = 0
            self.save_checkpoint(
                self.best_state_path,
                extra={
                    "kind": "best_val",
                    "val_loss": val_loss,
                    "epoch": self._best_epoch,
                    "previous_best": previous,
                },
            )
        else:
            self._epochs_since_improvement += 1
        return {
            "validation_loss": val_loss,
            "best_val_loss": self._best_val_loss,
            "best_epoch": self._best_epoch,
            "epochs_since_improvement": self._epochs_since_improvement,
            "improved": improved,
            "relative_improvement_vs_best": (
                round(
                    (self._best_val_loss - val_loss)
                    / max(abs(self._best_val_loss), 1e-12),
                    6,
                )
                if self._best_val_loss is not None
                else None
            ),
            "early_stop": (
                self._epochs_since_improvement
                >= int(self.profile.early_stopping_patience)
            ),
        }

    # -- the R3 anchor-margin observation (external source only) -------------

    def _check_anchor_margin(self) -> Optional[str]:
        """Sample the external anchor-margin source and classify it.

        Called at the telemetry cadence.  Returns the verdict (or None when
        the monitor is inactive); the hard guard raises
        :class:`TrainingDivergenceError` and the graceful stop is surfaced
        via the returned verdict for the caller to act on.
        """
        if (
            self.anchor_margin_monitor is None
            or self.profile.anchor_margin_source != "external"
            or self._anchor_margin_fn is None
        ):
            return None
        value = float(self._anchor_margin_fn())
        verdict = self.anchor_margin_monitor.classify(value)
        self._log(
            {
                "phase": "anchor_margin",
                "anchor_margin_mode": "external",
                "value": value,
                "verdict": verdict,
                **self.anchor_margin_monitor.thresholds,
            }
        )
        if verdict == ANCHOR_MARGIN_HARD_GUARD:
            # implausibly favorable margin — a degenerate anchoring, never
            # a value to train past (fail-closed, D-V2-050)
            raise TrainingDivergenceError(
                f"anchor-margin hard guard tripped (value {value} <= "
                f"{self.anchor_margin_monitor.hard_guard}) — implausibly "
                "favorable anchoring is a degenerate/numerically broken "
                "state, failing closed"
            )
        return verdict

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch  # tensors are moved per-sub-batch in the loss methods

    def _validation_loss(self, val_loader: DataLoader) -> float:
        """The mean total loss over the pristine validation split.

        Eval-mode, no-grad, fail-closed on any non-finite value (#45 s8 —
        a diverging validation loss is a bug to surface, never to hide).
        Drives the #27 curriculum's milestone stage advancement.
        """
        if val_loader is None:
            raise TrainerStateError("a validation loader is required (fail-closed)")
        was_training = self.policy.training
        self.policy.eval()
        total = 0.0
        count = 0
        try:
            with torch.no_grad():
                for batch in val_loader:
                    losses = self.compute_loss(batch)
                    value = float(losses["total"].detach())
                    if not math.isfinite(value):
                        raise TrainingDivergenceError(
                            f"non-finite validation loss ({value}) — failing "
                            "closed (#45 s8)"
                        )
                    total += value
                    count += 1
        finally:
            if was_training:
                self.policy.train()
        if count == 0:
            raise TrainerStateError(
                "the validation loader yielded no batches (fail-closed)"
            )
        return total / count

    def _optimizer_step(self) -> None:
        grads = [p.grad for p in self.policy.parameters() if p.grad is not None]
        if not grads:
            return
        for grad in grads:
            self._assert_finite(grad, "gradient")
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.profile.gradient_clip_norm
        )
        with torch.no_grad():
            for param in self.policy.parameters():
                if param.grad is not None:
                    self._assert_finite(param, "parameter (post-update check pending)")
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for param in self.policy.parameters():
                if not bool(torch.isfinite(param).all()):
                    raise TrainingDivergenceError(
                        "a model parameter became non-finite after the "
                        "optimizer step — failing closed (#45 s8)"
                    )
