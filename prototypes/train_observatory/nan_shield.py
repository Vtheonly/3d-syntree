"""
nan_shield.py — numerical-stability guard for (mixed-precision) PyTorch training.

Fixes the failure mode seen at epoch 6 / batch 44: a single NaN batch silently
became 16 consecutive skipped batches, spread into the next epoch (45 skips,
NaN validation loss) while the old logic kept skipping forever. A skip-based
guard CANNOT recover once NaN reaches model parameters or optimizer state.

What this module does differently
---------------------------------
1. LOCALIZE  — every non-finite loss is attributed to the first bad component
   (reaction CE / synthon CE / torsion NLL / total) and logged with its recent
   value history, so you know WHERE the NaN entered.
2. STREAK ABORT — N consecutive non-finite events (loss OR grads) raise
   SystemicNanError instead of skipping forever. Infinitely skipping wastes
   GPU time and proves nothing.
3. GRAD/PARAM/OPTIMIZER GATES — gradients and parameters are checked BEFORE the
   optimizer step; a non-finite update is never applied. If parameters or
   optimizer state are already contaminated, skipping cannot help and the run
   aborts.
4. AMP-CORRECT STEP — guarded_step() implements the canonical FP16 pattern:
   scaler.scale(loss).backward() → scaler.unscale_() → grad-finiteness check →
   clip → scaler.step() → scaler.update(). Inf-grad batches are skipped the
   way GradScaler intends, and only healthy updates touch the weights.
5. VERIFIED CHECKPOINTS — guarded_save() saves, reloads on CPU and verifies
   finiteness of every tensor before the checkpoint is marked known-good and
   (in your pipeline) uploaded to HF. rollback() restores from the last
   known-good checkpoint and resets the failure state.
6. FORENSICS — on abort, a JSON snapshot (component histories, param health,
   CUDA state) is written to nan_forensics/ for offline root-cause analysis.

Nothing here swallows real code errors: shape bugs and other exceptions still
raise. Only non-finite numerics and CUDA OOM are intercepted.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch

try:  # optional integration — the shield works standalone too
    from train_observatory import _to_float  # noqa: F401
except Exception:  # pragma: no cover
    def _to_float(x, default=None):
        try:
            if x is None:
                return default
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().item()) if x.numel() == 1 else default
            if isinstance(x, (int, float)):
                return float(x)
            return default
        except Exception:
            return default


class SystemicNanError(RuntimeError):
    """Raised when non-finite events persist: training must stop and roll back."""


# --------------------------------------------------------------------- free helpers
def finite(v) -> bool:
    v = _to_float(v)
    return v is not None and math.isfinite(v)


def tensor_stats(t) -> dict:
    if not isinstance(t, torch.Tensor):
        return {"type": str(type(t))}
    try:
        tt = t.detach().cpu().float()
        finite_mask = torch.isfinite(tt)
        has_any = bool(finite_mask.any())
        return {"shape": list(t.shape), "dtype": str(t.dtype),
                "min": float(tt[finite_mask].min()) if has_any else None,
                "max": float(tt[finite_mask].max()) if has_any else None,
                "has_nan": bool(torch.isnan(tt).any()),
                "has_inf": bool(torch.isinf(tt).any())}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def params_finite(model):
    """(ok, first_bad_param_name). Read-only."""
    for name, p in model.named_parameters():
        if p.numel() and not bool(torch.isfinite(p).all()):
            return False, name
    return True, None


def grads_finite(model):
    """(ok, first_bad_param_name). Read-only; ignores params without grads."""
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.numel() and not bool(torch.isfinite(p.grad).all()):
            return False, name
    return True, None


def optimizer_state_finite(optimizer):
    """(ok, first_bad_entry_name). Checks exp_avg / exp_avg_sq / momentum etc."""
    for i, st in enumerate(optimizer.state.values()):
        for k, v in st.items():
            if isinstance(v, torch.Tensor) and v.numel() and not bool(torch.isfinite(v).all()):
                return False, f"state[{i}].{k}"
    return True, None


def verify_checkpoint(path, expect_keys=("model", "optimizer", "scheduler")):
    """Load a checkpoint on CPU and check every tensor is finite.
    Returns (ok, report). Works with {"model": sd, ...} dicts and raw
    state_dicts alike. Never mutates training state."""
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        return False, {"load_error": repr(e)}

    bad, checked = [], 0

    def walk(obj, prefix):
        nonlocal checked
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, torch.Tensor):
            checked += 1
            if obj.numel() and not bool(torch.isfinite(obj).all()):
                bad.append(prefix)

    walk(ck, "")
    keys_present = {k: (k in ck) for k in expect_keys} if isinstance(ck, dict) else \
        {k: None for k in expect_keys}
    return len(bad) == 0, {"checked_tensors": checked, "bad_tensors": bad[:10],
                           "keys_present": keys_present,
                           "size_mb": round(os.path.getsize(path) / 1e6, 2) if os.path.exists(path) else None}


# ------------------------------------------------------------------------- the shield
class NanShield:
    """Drop-in numerical guard around an existing training loop."""

    def __init__(self, monitor=None, max_consecutive=5, history=32,
                 max_consecutive_oom=3, param_check_every=1):
        self.monitor = monitor
        self.max_consecutive = int(max_consecutive)
        self.max_oom = int(max_consecutive_oom)
        self.param_check_every = int(param_check_every)  # check params every N batches
        self.streak = 0            # consecutive non-finite events (loss or grads)
        self.oom_streak = 0       # consecutive CUDA OOMs inside backward
        self.total_skips = 0      # historical count of skipped batches
        self.total_oom = 0
        self.first_failure = None  # dict describing the first non-finite event
        self.last_localized = None
        self.params_corrupted = False
        self.last_known_good = None  # path of last verified checkpoint
        self._hist = defaultdict(lambda: deque(maxlen=history))
        self._batches_seen = 0

    # ------------------------------------------------------------- loss checking
    def check_batch(self, total_loss, components=None, epoch=None, batch=None,
                    context=None, reset_streak_on_ok=True):
        """Check a loss (optionally per component) BEFORE backward.
        Returns 'ok' | 'skip_batch' | 'abort'.
        - records per-component history for forensics
        - localizes the first non-finite component
        - aborts after max_consecutive consecutive failures"""
        total = _to_float(total_loss)
        comps = {str(k): _to_float(v) for k, v in dict(components or {}).items()
                 if _to_float(v) is not None}
        bad = None
        # components FIRST: a NaN component is more informative than a NaN total
        for name, v in comps.items():
            self._hist[name].append(v)  # keep recent history for forensics
            if not math.isfinite(v):
                bad = (name, v)
                break
        if total is not None:
            self._hist["total"].append(total)
        if bad is None and total is not None and not math.isfinite(total):
            bad = ("total", total)
        if bad is None:
            if reset_streak_on_ok:
                self.streak = 0
            self._batches_seen += 1
            return "ok"
        return self._register_failure(bad[0], bad[1], epoch, batch, context)

    def _register_failure(self, component, value, epoch, batch, context,
                          reason=None):
        self.streak += 1
        self.total_skips += 1
        rec = {"component": component, "value": str(value), "reason": reason or "loss_nonfinite",
               "epoch": epoch, "batch": batch,
               "recent": [x for x in self._hist.get(component, [])][-8:]}
        if self.first_failure is None:
            self.first_failure = rec
        self.last_localized = component
        if self.monitor:
            try:
                self.monitor.event("nan_localized", flush=True, epoch=epoch, batch=batch,
                                   component=component, value=str(value),
                                   streak=self.streak, context=context,
                                   recent=rec["recent"],
                                   cuda_state=self.monitor.cuda_state())
            except Exception:
                pass
        if self.streak >= self.max_consecutive:
            if self.monitor:
                try:
                    self.monitor.event("systemic_nan_detected", flush=True,
                                       streak=self.streak, component=component,
                                       epoch=epoch, batch=batch,
                                       first_failure=self.first_failure,
                                       total_skips=self.total_skips,
                                       cuda_state=self.monitor.cuda_state())
                except Exception:
                    pass
            return "abort"
        return "skip_batch"

    # ------------------------------------------------------------ parameter health
    def check_params(self, model, optimizer=None, epoch=None, batch=None):
        """Detect NaN-contaminated weights / optimizer state. If parameters are
        already non-finite, batch-skipping can never recover — rollback is
        required (this is what made the epoch 6→7 cascade unrecoverable)."""
        ok, bad_name = params_finite(model)
        if ok and optimizer is not None:
            ok, bad_name = optimizer_state_finite(optimizer)
        if not ok and not self.params_corrupted:
            self.params_corrupted = True
            if self.monitor:
                try:
                    self.monitor.event("param_corruption", flush=True, param=bad_name,
                                       epoch=epoch, batch=batch,
                                       cuda_state=self.monitor.cuda_state())
                except Exception:
                    pass
        elif not ok and self.monitor and self._batches_seen % 50 == 0:
            pass  # already flagged; do not spam
        return ok

    # ------------------------------------------------------- canonical guarded step
    def guarded_step(self, loss, model, optimizer, scaler=None, clip=1.0,
                     scheduler=None, components=None, epoch=None, batch=None,
                     context=None, zero_grad_before=True):
        """The full, numerically-guarded optimizer update. Returns
        (action, grad_norm) where action ∈ {'ok', 'skipped', 'abort'}.

        AMP/FP16 path (pass a torch.cuda.amp.GradScaler as `scaler`):
            scaler.scale(loss).backward(); scaler.unscale_(optimizer);
            [finiteness gates]; clip; scaler.step(optimizer); scaler.update()
        FP32 path: backward → gates → clip → optimizer.step().
        Identical math to a plain clipped step when everything is finite."""
        # ---- A) loss-level check (localizes failing component) --------------
        action = self.check_batch(loss, components=components, epoch=epoch,
                                  batch=batch, context=context,
                                  reset_streak_on_ok=False)
        if action != "ok":
            if zero_grad_before:
                optimizer.zero_grad(set_to_none=True)
            return action, None

        if zero_grad_before:
            optimizer.zero_grad(set_to_none=True)
        # periodic parameter health (cheap on small models; tune for big ones)
        if self.param_check_every and self._batches_seen % max(1, self.param_check_every) == 0:
            if not self.check_params(model, optimizer, epoch, batch):
                return "abort", None  # params contaminated: rollback, don't skip

        # ---- B) backward (only OOM is intercepted; real bugs still raise) ---
        try:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)   # so clipping + gates see true grads
            else:
                loss.backward()
        except (torch.cuda.OutOfMemoryError, MemoryError) as e:  # noqa: PERF203
            self.oom_streak += 1
            self.total_oom += 1
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if self.monitor:
                try:
                    self.monitor.warn(f"OOM in backward @ep{epoch} b{batch}: {e}",
                                      oom_streak=self.oom_streak)
                except Exception:
                    pass
            if self.oom_streak >= self.max_oom:
                return "abort", None
            return "skipped", None

        # ---- C) gradient / param finiteness gates BEFORE any update --------
        gn = None
        if clip:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        gn_f = _to_float(gn) if gn is not None else None
        grads_ok, bad_grad = grads_finite(model)
        params_ok, _bad_param = params_finite(model)
        healthy = grads_ok and params_ok and (gn_f is None or math.isfinite(gn_f))
        if not healthy:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            reason = "grad_nonfinite" if not grads_ok else \
                     ("param_nonfinite" if not params_ok else "grad_norm_nonfinite")
            action = self._register_failure(reason, gn_f if gn_f is not None else 0.0,
                                            epoch, batch, context, reason=reason)
            return action, gn_f

        # ---- D) healthy: apply the update -----------------------------------
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        self.streak = 0
        self.oom_streak = 0
        self._batches_seen += 1
        return "ok", gn_f

    # ------------------------------------------------------------- checkpoints
    def guarded_save(self, path, model, optimizer=None, scheduler=None, epoch=None,
                     extra=None):
        """Save → reload on CPU → verify every tensor is finite → mark as
        known-good (or reject). Only verified checkpoints should be uploaded
        or used for rollback. Returns (ok, report)."""
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        ck = {"model": model.state_dict(), "epoch": epoch, "extra": extra or {}}
        if optimizer is not None:
            ck["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            ck["scheduler"] = scheduler.state_dict()
        torch.save(ck, path)
        ok, rep = verify_checkpoint(path)
        rep.update({"epoch": epoch, "verified": ok,
                    "bad": rep.get("bad_tensors", [])[:10]})
        health_path = path + ".health.json"
        with open(health_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        if ok:
            self.last_known_good = path
            if self.monitor:
                self.monitor.checkpoint_saved(path, epoch=epoch, health="verified")
        else:
            if self.monitor:
                self.monitor.event("checkpoint_rejected", flush=True, path=path,
                                   epoch=epoch, bad_tensors=rep["bad"])
        return ok, rep

    def rollback(self, model, optimizer=None, scheduler=None, path=None):
        """Restore model (+ optimizer/scheduler) from the last known-good
        checkpoint. Returns True on success. Resets streak/corruption state so
        training can resume; history counters are preserved for the audit trail."""
        path = str(path or self.last_known_good or "")
        if not path or not os.path.exists(path):
            return False
        ok, rep = verify_checkpoint(path)
        if not ok:
            if self.monitor:
                self.monitor.event("rollback_failed", flush=True, path=path,
                                   bad_tensors=rep.get("bad_tensors"))
            return False
        ck = torch.load(path, map_location="cpu")
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        if optimizer is not None and isinstance(ck, dict) and "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None and isinstance(ck, dict) and "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        self.streak = 0
        self.oom_streak = 0
        self.params_corrupted = False
        self._hist.clear()
        if self.monitor:
            self.monitor.checkpoint_loaded(path, epoch=ck.get("epoch") if isinstance(ck, dict) else None,
                                            reason="rollback")
            self.monitor.event("rollback_restored", flush=True, path=path,
                               epoch=ck.get("epoch") if isinstance(ck, dict) else None,
                               cuda_state=self.monitor.cuda_state())
        return True

    # ------------------------------------------------------------------ forensics
    def forensics(self, epoch, batch, components=None, model=None, extra=None):
        """Write a JSON snapshot of everything relevant to the failure:
        per-component recent histories, first failure record, param health,
        CUDA state. Stored under <run_dir>/nan_forensics/."""
        base = None
        if self.monitor is not None:
            base = Path(self.monitor.run_dir) / "nan_forensics"
        else:
            base = Path("nan_forensics")
        base.mkdir(parents=True, exist_ok=True)
        snap = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epoch": epoch, "batch": batch,
            "streak": self.streak, "total_skips": self.total_skips,
            "oom_streak": self.oom_streak,
            "first_failure": self.first_failure,
            "last_localized": self.last_localized,
            "params_corrupted": self.params_corrupted,
            "component_history": {k: list(v) for k, v in self._hist.items()},
            "components_current": {str(k): _to_float(v) for k, v in dict(components or {}).items()},
            "param_health": None, "bad_param_stats": None,
            "cuda_state": self.monitor.cuda_state() if self.monitor else {},
            "extra": extra or {},
        }
        if model is not None:
            ok, bad = params_finite(model)
            snap["param_health"] = {"finite": ok, "first_bad_param": bad}
            if not ok:
                for name, p in model.named_parameters():
                    if name == bad:
                        snap["bad_param_stats"] = {"name": name, **tensor_stats(p)}
                        break
        path = base / f"ep{epoch}_b{batch}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, default=str)
        return str(path)
