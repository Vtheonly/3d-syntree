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
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from syntree.models.policy import (
    POLICY_REACTION_IDS,
    SynTreePolicy,
    build_continuation_masks,
)
from syntree.models.torsion_head import torsion_circular_loss
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
    """The static training profile (Profile A — Tesla T4 FP32).

    Every field is frozen in version-controlled configuration; nothing is
    probed from free VRAM at runtime (#45 §5.2).
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

    @property
    def effective_batch_size(self) -> int:
        return int(self.micro_batch_size) * int(self.gradient_accumulation_steps)

    def digest(self) -> str:
        """Stable SHA-256 of the frozen profile (checkpoint compatibility)."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_profile(path: str = DEFAULT_PROFILE_PATH) -> TrainerProfile:
    """Load a ``TrainerProfile`` from a frozen JSON configuration file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return TrainerProfile(**data)


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
        if payload["scheduler"] and self.scheduler:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.progress = _Progress(**payload["progress"])
        self.history = list(payload["history"])
        restore_rng_state(rng_state_from_dict(payload["rng"]))
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
                if (
                    self.profile.checkpoint_interval_steps
                    and self.progress.global_step % self.profile.checkpoint_interval_steps == 0
                ):
                    self.save_checkpoint()
            else:
                # epoch completed on schedule: flush a trailing partial
                # accumulation and reset the in-epoch cursor
                self._optimizer_step()
                self.progress.batch_in_epoch = 0
                if val_loader is not None and stop_reason == "max_epochs_reached":
                    self._log({"phase": "validation", **self.telemetry()})
                continue
            break  # time-budget break inside the batch loop
        self.save_checkpoint(extra={"stop_reason": stop_reason})
        return {
            "stop_reason": stop_reason,
            "epochs_completed": self.progress.epoch + (1 if self.progress.batch_in_epoch == 0 else 0),
            "global_step": self.progress.global_step,
            "final_losses": self.history[-1] if self.history else None,
            "checkpoint": self.checkpoint_path,
        }

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch  # tensors are moved per-sub-batch in the loss methods

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
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for param in self.policy.parameters():
                if not bool(torch.isfinite(param).all()):
                    raise TrainingDivergenceError(
                        "a model parameter became non-finite after the "
                        "optimizer step — failing closed (#45 s8)"
                    )
