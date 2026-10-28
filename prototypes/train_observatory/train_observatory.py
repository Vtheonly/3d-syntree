"""
train_observatory.py — self-contained training observability for PyTorch pipelines.
====================================================================================
One file. Stdlib-only core; torch / numpy / matplotlib / psutil are used
opportunistically when present and everything degrades gracefully when absent.

RECORDED PER RUN — runs/<run_name>_<UTC timestamp>/
  manifest.json      environment + config snapshot (hardware, versions, config hash)
  config.json        the exact configuration dict
  events.jsonl       append-only audit trail: run_start, epoch_start / epoch_end /
                     epoch_aborted, validation, evaluation, checkpoint_saved/loaded,
                     warning, error (full traceback), numeric_alert (NaN/Inf watchdog),
                     grad_spike, note, run_end
  metrics.csv        one row per optimizer step: loss, acc, lr, grad_norm, batch time,
                     throughput, VRAM, per-component losses (JSON "extra" column)
  evaluations.csv    long-format eval metrics (epoch, level, metric, value) — accepts
                     anything you throw at it: reaction/synthon accuracy, top-k accuracy,
                     generation quality, per-difficulty-level metrics, ...
  system.csv         background-sampled CPU % / RAM usage
  checkpoints.jsonl   every checkpoint saved/loaded (path, size, epoch)
  summary.json       end-of-run digest (peaks, best epoch, totals, health)
  report.txt         human-readable one-page digest
  plots/*.png        loss, accuracy, lr schedule, grad norm, component losses,
                     throughput, resources, evaluation metrics, per-level heatmaps,
                     dashboard

GUARANTEES (why it cannot hurt your training)
  1. Every public method is exception-guarded: a monitoring bug degrades to a
     one-line stderr warning + an entry in `monitor.soft_failures`; training continues.
  2. No RNG consumption, no tensor mutation, no optimizer/scheduler interaction —
     identical trainings with and without the monitor are bit-identical
     (verified by prototype_test.py).
  3. Hot-loop cost is a few dict appends + an occasional buffered flush
     (prototype_test.py asserts < 2 ms/step on CPU).
  4. All heavy work (plots, summaries) happens in finish()/plot_all(), never in the loop.
  5. Nothing is swallowed: exceptions you raise are recorded and then re-raised.

QUICK START (inside your existing trainer — ~6 lines of changes)
    from train_observatory import RunMonitor, RunReport, compare_runs

    monitor = RunMonitor(run_name="stage1", config=cfg_dict)
    ...
    with monitor.epoch(epoch, n_batches=len(loader)):
        for i, batch in enumerate(loader):
            ...  # your existing forward / backward / step code
            monitor.step(epoch=epoch, batch=i, loss=loss, lr=lr, grad_norm=gn,
                         batch_size=len(x), acc=batch_acc,
                         components={"reaction_ce": r, "synthon_ce": s})
    monitor.validation(epoch, val_loss=vloss, val_acc=vacc)
    monitor.evaluation(epoch, metrics={"reaction_acc": a1, "synthon_EM": em})
    ...
    monitor.finish(status="completed")

  Post-hoc analysis — works even on runs that crashed before finish():
    rep = RunReport("runs/stage1_20261028_093000"); rep.plot_all(".../plots_reloaded")
    compare_runs(["runs/stage1_a", "runs/stage2_b"], "runs/comparison.png")
"""

from __future__ import annotations

import contextlib
import csv
import functools
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
import traceback
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    pass

_HAS_NUMPY = False
try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    pass

_HAS_PSUTIL = False
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    pass

_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg", force=False)  # no-op if a backend is already active (e.g. Colab inline)
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover
    pass

LIB_VERSION = "1.2.0"
SCHEMA_VERSION = 1

METRIC_COLS = ["ts", "global_step", "epoch", "batch", "n_batches", "loss", "acc", "lr",
               "grad_norm", "batch_time_ms", "steps_per_sec", "samples_per_sec",
               "vram_alloc_mb", "vram_reserved_mb", "extra"]
EVAL_COLS = ["ts", "epoch", "global_step", "subset", "level", "metric", "value"]
SYS_COLS = ["ts", "t_offset_s", "cpu_pct", "ram_used_mb", "ram_pct"]

_CRITICAL_EVENTS = {"error", "numeric_alert", "run_end", "run_start",
                    "systemic_nan_detected", "nan_localized", "rollback_restored",
                    "nan_skip", "epoch_aborted", "checkpoint_saved", "checkpoint_loaded"}


# ----------------------------------------------------------------------------- utils
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _to_float(x, default=None):
    """Tensor/numpy-safe scalar coercion. Never raises."""
    try:
        if x is None:
            return default
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            if x.numel() != 1:
                return default
            return float(x.detach().cpu().item())
        if _HAS_NUMPY and isinstance(x, _np.generic):
            return float(x)
        if isinstance(x, (int, float)):
            return float(x)
        return default
    except Exception:
        return default


def _json_default(o):
    if _HAS_TORCH and isinstance(o, torch.Tensor):
        try:
            return o.detach().cpu().tolist() if o.numel() > 1 else float(o.item())
        except Exception:
            return str(o)
    if _HAS_NUMPY and isinstance(o, _np.ndarray):
        return o.tolist()
    if _HAS_NUMPY and isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset, tuple, deque)):
        return list(o)
    if isinstance(o, BaseException):
        return f"{type(o).__name__}: {o}"
    return str(o)


def _dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


def _sha256_obj(obj) -> str:
    try:
        h = hashlib.sha256(_dumps({"o": obj}).encode("utf-8")).hexdigest()
    except Exception:
        h = "unavailable"
    return h


def _env_snapshot() -> dict:
    snap = {"hostname": platform.node(), "platform": platform.platform(),
            "python": platform.python_version(), "cpu_count": os.cpu_count()}
    if _HAS_PSUTIL:
        try:
            snap["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        except Exception:
            pass
    if _HAS_TORCH:
        snap["torch"] = torch.__version__
        try:
            if torch.cuda.is_available():
                snap["cuda"] = torch.version.cuda
                snap["gpu"] = torch.cuda.get_device_name(0)
                snap["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                snap["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        except Exception:
            pass
    snap["numpy"] = _np.__version__ if _HAS_NUMPY else None
    return snap


def _guard(method):
    """Decorator: turn any monitoring failure into a soft-fail instead of a crash."""
    @functools.wraps(method)
    def wrapper(self, *a, **k):
        try:
            return method(self, *a, **k)
        except Exception as e:  # noqa: BLE001 — deliberate: monitoring must never kill training
            self._soft_fail(method.__name__, e)
            return None
    return wrapper


# ----------------------------------------------------------------------- resource sampler
class _Sampler(threading.Thread):
    """Background CPU/RAM sampler. Owns system.csv exclusively (no locking needed)."""

    def __init__(self, path, interval):
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()   # NOTE: must NOT be named `_stop` (clobbers Thread._stop)
        self.interval = interval
        self.rows = []
        self._t0 = time.time()
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(SYS_COLS)
        self._f.flush()
        if _HAS_PSUTIL:
            try:
                psutil.cpu_percent(None)  # prime the counter
            except Exception:
                pass

    def run(self):
        if not _HAS_PSUTIL or not self.interval:
            return
        while not self._stop_evt.wait(self.interval):
            try:
                cpu = psutil.cpu_percent(None)
                vm = psutil.virtual_memory()
                self.rows.append([_now_iso(), round(time.time() - self._t0, 3),
                                  round(cpu, 2), round(vm.used / 1e6, 1), round(vm.percent, 2)])
                if len(self.rows) >= 20:
                    self._flush()
            except Exception:
                break  # a background sampler must never propagate

    def _flush(self):
        if self.rows:
            self._w.writerows(self.rows)
            self.rows.clear()
            self._f.flush()

    def close(self):
        self._stop_evt.set()
        self.join(timeout=2.0)
        self._flush()
        self._f.close()


# --------------------------------------------------------------------------- RunMonitor
class RunMonitor:
    """Records everything about one training run into a self-describing directory."""

    def __init__(self, run_name="run", config=None, run_root="runs", run_dir=None,
                 sampling_interval=1.0, flush_every=50, nan_guard=True,
                 grad_spike_guard=True, verbose=True, meta=None):
        # NOTE: __init__ is deliberately NOT guarded — a broken setup must fail loudly.
        self.run_name = str(run_name)
        self.config = dict(config or {})
        self.verbose = bool(verbose)
        self.nan_guard = bool(nan_guard)
        self.grad_spike_guard = bool(grad_spike_guard)
        self.flush_every = max(1, int(flush_every))

        ts = datetime.now(timezone.utc)
        rid = f"{self.run_name}_{ts:%Y%m%d_%H%M%S}"
        base = str(run_dir) if run_dir else os.path.join(str(run_root), rid)
        candidate = base
        i = 0
        while os.path.exists(candidate):
            i += 1
            candidate = f"{base}_{i}"
        self.run_dir = Path(candidate)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_dir.name

        # ---- state ---------------------------------------------------------------
        self.soft_failures = []            # (tag, repr(exc)) — internal monitor failures
        self._warned = set()
        self._g = 0                        # global step counter
        self._epoch = None
        self._epoch_t0 = None
        self._epoch_steps = 0
        self.warnings = 0
        self.errors = 0
        self.nan_total = 0
        self.nan_logged = 0
        self.spike_total = 0
        self.spike_logged = 0
        self.ckpts = 0
        self._t_start = time.perf_counter()
        self._t_last = None
        self._sps_ema = None
        self._hist = {}                    # name -> deque(recent values) for NaN context
        self._recent_grad = deque(maxlen=32)
        self._nan_alert_steps = []
        self._marks = []                   # (global_step, epoch) at each epoch_start
        self._train = []                   # in-memory metric rows (mirrors metrics.csv)
        self._mrows = []                   # write buffer for metrics.csv
        self._evalrows = []                # write buffer for evaluations.csv
        self._final_sys_rows = []          # sampler rows snapshot taken in finish()
        self._vals = []                    # validation rows
        self._evals = []                   # evaluation long rows
        self._orig_warn = None
        self._orig_hook = None
        self._grad_clip = None
        for key in ("grad_clip", "clip", "max_grad_norm", "clip_grad_norm", "clip_norm"):
            if key in self.config and self.config[key] is not None:
                self._grad_clip = _to_float(self.config[key])
                break

        # ---- files ---------------------------------------------------------------
        self._fe = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
        self._fm = open(self.run_dir / "metrics.csv", "w", newline="")
        self._mw = csv.DictWriter(self._fm, fieldnames=METRIC_COLS, extrasaction="ignore")
        self._mw.writeheader()
        self._fev = open(self.run_dir / "evaluations.csv", "w", newline="")
        self._ew = csv.DictWriter(self._fev, fieldnames=EVAL_COLS, extrasaction="ignore")
        self._ew.writeheader()
        self._fc = open(self.run_dir / "checkpoints.jsonl", "a", encoding="utf-8")
        self._sampler = _Sampler(str(self.run_dir / "system.csv"), sampling_interval)
        self._sampler.start()

        (self.run_dir / "manifest.json").write_text(
            _dumps({"schema": SCHEMA_VERSION, "library": LIB_VERSION, "run_id": self.run_id,
                    "created_at": _now_iso(), "env": _env_snapshot(),
                    "config": self.config, "config_sha256": _sha256_obj(self.config),
                    "meta": meta or {}}), encoding="utf-8")
        (self.run_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, default=_json_default), encoding="utf-8")

        self.event("run_start", flush=True, library=LIB_VERSION, schema=SCHEMA_VERSION,
                   torch=str(_env_snapshot().get("torch")), device="cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu")

    # ------------------------------------------------------------------ soft-fail
    def _soft_fail(self, tag, exc):
        self.soft_failures.append((tag, repr(exc)))
        if tag not in self._warned and self.verbose:
            self._warned.add(tag)
            try:
                print(f"[train_observatory] soft-fail in {tag}: {exc!r} "
                      f"(recorded in monitor.soft_failures; training continues)", file=sys.stderr)
            except Exception:
                pass

    # -------------------------------------------------------------------- events
    @_guard
    def event(self, kind, flush=False, **fields):
        row = {"ts": _now_iso(), "kind": str(kind), "step": self._g}
        row.update(fields)
        self._fe.write(_dumps(row) + "\n")
        if kind == "warning":
            self.warnings += 1
        elif kind == "error":
            self.errors += 1
        if flush or kind in _CRITICAL_EVENTS or self._fe.tell() > 65536:
            self._fe.flush()
        return row

    @_guard
    def warn(self, message, **fields):
        return self.event("warning", message=str(message), **fields)

    @_guard
    def error(self, message, exc=None, **fields):
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        else:
            tb = "".join(traceback.format_stack())
        return self.event("error", message=str(message), traceback=tb, flush=True, **fields)

    def note(self, kind="note", **fields):
        return self.event(kind, **fields)

    # ------------------------------------------------------------------- epochs
    def set_global_step(self, n):
        self._g = int(n)

    @_guard
    def epoch_start(self, epoch, n_batches=None):
        self._epoch = epoch
        self._epoch_t0 = time.perf_counter()
        self._epoch_steps = 0
        self._marks.append((self._g, epoch))
        return self.event("epoch_start", epoch=epoch, n_batches=n_batches)

    @_guard
    def epoch_end(self, epoch):
        dur = time.perf_counter() - self._epoch_t0 if self._epoch_t0 is not None else None
        row = self.event("epoch_end", epoch=epoch,
                         duration_s=round(dur, 4) if dur is not None else None,
                         steps=self._epoch_steps, flush=True)
        self._epoch = None
        self._epoch_t0 = None
        return row

    @contextlib.contextmanager
    def epoch(self, epoch, n_batches=None):
        """`with monitor.epoch(0, len(loader)):` — exceptions inside are recorded and re-raised."""
        self.epoch_start(epoch, n_batches)
        try:
            yield self
        except BaseException as e:
            self.event("epoch_aborted", epoch=epoch, error=repr(e), flush=True)
            raise
        finally:
            self.epoch_end(epoch)

    # ------------------------------------------------------------ numeric watchdog
    @_guard
    def check_value(self, value, name="loss", epoch=None, batch=None, context=None):
        """Returns True if `value` is finite. Logs a capped number of alerts (max 5/run)."""
        v = _to_float(value)
        if v is None:
            return True
        hist = self._hist.setdefault(name, deque(maxlen=32))
        hist.append(v)
        finite = math.isfinite(v)
        if not finite:
            self.nan_total += 1
            if self.nan_logged < 5:
                self.nan_logged += 1
                self._nan_alert_steps.append(self._g)
                self.event("numeric_alert", name=name, value=str(v), epoch=epoch, batch=batch,
                           context=context, recent=[x for x in hist][-8:],
                           cuda_state=self.cuda_state(), flush=True)
        return finite

    def is_healthy(self) -> bool:
        return self.nan_total == 0

    def cuda_state(self) -> dict:
        if not (_HAS_TORCH and torch.cuda.is_available()):
            return {}
        try:
            return {"allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}
        except Exception:
            return {}

    # ------------------------------------------------------------------ training
    @_guard
    def step(self, epoch=None, batch=None, n_batches=None, loss=None, lr=None,
             grad_norm=None, batch_size=None, acc=None, **extra):
        now = time.perf_counter()
        dt = None if self._t_last is None else now - self._t_last
        self._t_last = now

        loss_f = _to_float(loss)
        lr_f = _to_float(lr)
        gn_f = _to_float(grad_norm)
        acc_f = _to_float(acc)

        if self.nan_guard and loss_f is not None:
            self.check_value(loss_f, name="loss", epoch=epoch, batch=batch, context="train_step")

        if self.grad_spike_guard and gn_f is not None and math.isfinite(gn_f):
            self._recent_grad.append(gn_f)
            if len(self._recent_grad) >= 8:
                srt = sorted(self._recent_grad)
                med = srt[len(srt) // 2]
                if gn_f > max(10.0 * med, 5.0):
                    self.spike_total += 1
                    if self.spike_logged < 5:
                        self.spike_logged += 1
                        self.event("grad_spike", grad_norm=gn_f, median=med,
                                   epoch=epoch, batch=batch, flush=True)

        vram_alloc = vram_res = None
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                vram_alloc = round(torch.cuda.memory_allocated() / 1e6, 1)
                vram_res = round(torch.cuda.memory_reserved() / 1e6, 1)
            except Exception:
                pass

        sps = (1.0 / dt) if (dt and dt > 0) else None
        if sps is not None:
            self._sps_ema = sps if self._sps_ema is None else 0.7 * self._sps_ema + 0.3 * sps
        sps_out = round(self._sps_ema, 3) if self._sps_ema is not None else None
        sps_inst = round(sps, 3) if sps is not None else None
        samples_ps = round(batch_size / dt, 1) if (batch_size and dt and dt > 0) else None

        self._g += 1
        if self._epoch is not None:
            self._epoch_steps += 1

        row = {"ts": _now_iso(), "global_step": self._g, "epoch": epoch, "batch": batch,
               "n_batches": n_batches, "loss": loss_f, "acc": acc_f, "lr": lr_f,
               "grad_norm": gn_f, "batch_time_ms": round(dt * 1000, 3) if dt is not None else None,
               "steps_per_sec": sps_out, "samples_per_sec": samples_ps,
               "vram_alloc_mb": vram_alloc, "vram_reserved_mb": vram_res,
               "extra": _dumps(extra) if extra else ""}
        self._mrows_buf(row)
        self._train.append(row)
        return loss_f

    def _mrows_buf(self, row):
        """Buffer rows; DictWriter-friendly via a small queue."""
        if not hasattr(self, "_mrows") or self._mrows is None:
            self._mrows = []
        self._mrows.append(row)
        if len(self._mrows) >= self.flush_every:
            self._flush_metrics()

    @_guard
    def _flush_metrics(self):
        if getattr(self, "_mrows", None):
            self._mw.writerows(self._mrows)
            self._mrows.clear()
            self._fm.flush()

    # ---------------------------------------------------------------- validation
    @_guard
    def validation(self, epoch, val_loss, val_acc=None, **extra):
        row = {"epoch": epoch, "step": self._g,
               "val_loss": _to_float(val_loss), "val_acc": _to_float(val_acc)}
        self._vals.append(row)
        return self.event("validation", epoch=epoch, flush=True,
                          val_loss=row["val_loss"], val_acc=row["val_acc"], **extra)

    # ---------------------------------------------------------------- evaluation
    @_guard
    def evaluation(self, epoch, metrics, subset="val", level=None, flush=True):
        """Log any dict of eval metrics (reaction/synthon accuracy, top-k, generation
        quality, ...). With `level=k` the metric names get an "L{k}/" prefix so
        per-difficulty-level curves and heatmaps can be built automatically."""
        prefix = f"L{level}/" if level is not None else ""
        n = 0
        for name in sorted(metrics, key=str):
            v = _to_float(metrics[name])
            if v is None:
                continue
            row = {"ts": _now_iso(), "epoch": epoch, "global_step": self._g,
                   "subset": subset, "level": level,
                   "metric": prefix + str(name), "value": v}
            self._evals.append(row)
            if getattr(self, "_evalrows", None) is None:
                self._evalrows = []
            self._evalrows.append(row)
            n += 1
        if flush:
            self._flush_evals()
        return self.event("evaluation", epoch=epoch, level=level, subset=subset,
                          metrics={prefix + str(k): _to_float(v) for k, v in metrics.items()},
                          flush=True)

    @_guard
    def _flush_evals(self):
        if getattr(self, "_evalrows", None):
            self._ew.writerows(self._evalrows)
            self._evalrows.clear()
            self._fev.flush()

    # --------------------------------------------------------------- checkpoints
    @_guard
    def checkpoint_saved(self, path, epoch=None, health=None, **fields):
        try:
            size_mb = round(os.path.getsize(path) / 1e6, 2)
        except Exception:
            size_mb = None
        self.ckpts += 1
        rec = {"ts": _now_iso(), "event": "saved", "path": str(path), "epoch": epoch,
               "size_mb": size_mb, "health": health, "step": self._g}
        self._fc.write(_dumps(rec) + "\n")
        self._fc.flush()
        return self.event("checkpoint_saved", path=str(path), epoch=epoch,
                          size_mb=size_mb, health=health, flush=True, **fields)

    @_guard
    def checkpoint_loaded(self, path, epoch=None, **fields):
        rec = {"ts": _now_iso(), "event": "loaded", "path": str(path), "epoch": epoch,
               "step": self._g}
        self._fc.write(_dumps(rec) + "\n")
        self._fc.flush()
        return self.event("checkpoint_loaded", path=str(path), epoch=epoch, flush=True, **fields)

    # ------------------------------------------------------------------- misc io
    def flush(self):
        try:
            self._flush_metrics()
        except Exception as e:
            self._soft_fail("flush_metrics", e)
        try:
            self._flush_evals()
        except Exception as e:
            self._soft_fail("flush_evals", e)
        try:
            self._fe.flush()
        except Exception as e:
            self._soft_fail("flush_events", e)

    @_guard
    def capture_warnings(self):
        """Route Python warnings into the event log (restored in finish())."""
        if self._orig_warn is not None:
            return False
        self._orig_warn = warnings.showwarning
        mon = self

        def shim(message, category, filename, lineno, file=None, line=None):
            try:
                mon.warn(f"{category.__name__}: {message}", source=f"{filename}:{lineno}")
            except Exception:
                pass
            mon._orig_warn(message, category, filename, lineno, file, line)

        warnings.showwarning = shim
        return True

    @_guard
    def install_excepthook(self):
        """Log uncaught exceptions before the process dies (script mode). In notebooks
        IPython intercepts exceptions — use the try/except pattern shown in
        integration_snippet.py there instead."""
        if self._orig_hook is not None:
            return False
        self._orig_hook = sys.excepthook
        mon = self

        def hook(t, v, tb):
            try:
                mon.error("uncaught exception", exc=v if isinstance(v, BaseException) else None)
            finally:
                mon._orig_hook(t, v, tb)

        sys.excepthook = hook
        return True

    # -------------------------------------------------------------------- finish
    @_guard
    def finish(self, status="completed", plot=True, **fields):
        try:
            self._final_sys_rows = list(self._sampler.rows)
        except Exception:
            self._final_sys_rows = []
        self._sampler.close()
        dur = time.perf_counter() - self._t_start
        self.flush()

        generated = []
        if plot and _HAS_MPL:
            try:
                bundle = self._build_bundle()
                generated = plot_run(bundle, self.run_dir / "plots") or []
            except Exception as e:
                self._soft_fail("plot_all", e)

        summary = self._summary(status, dur, generated)
        try:
            (self.run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
            (self.run_dir / "report.txt").write_text(_render_report(summary), encoding="utf-8")
        except Exception as e:
            self._soft_fail("write_summary", e)

        self.event("run_end", status=status, duration_s=round(dur, 3),
                   summary_path=str(self.run_dir / "summary.json"), flush=True, **fields)

        if self._orig_warn is not None:
            try:
                warnings.showwarning = self._orig_warn
            except Exception:
                pass
            self._orig_warn = None
        if self._orig_hook is not None:
            try:
                sys.excepthook = self._orig_hook
            except Exception:
                pass
            self._orig_hook = None

        try:
            self._fe.close()
            self._fm.close()
            self._fev.close()
            self._fc.close()
        except Exception as e:
            self._soft_fail("close_files", e)
        if self.verbose:
            try:
                print(f"[train_observatory] run '{self.run_id}' {status} -> {self.run_dir}")
            except Exception:
                pass
        return summary

    def _build_bundle(self):
        return _bundle(rows=self._train, vals=self._vals, evals=self._evals,
                       marks=self._marks,
                       sys_rows=(self._final_sys_rows or list(self._sampler.rows)),
                       name=self.run_id, nan_alert_steps=list(self._nan_alert_steps),
                       grad_clip=self._grad_clip, status="")

    def _summary(self, status, duration_s, plots):
        vals = self._vals
        best = None
        finite_vals = [v for v in vals if v.get("val_loss") is not None and math.isfinite(v["val_loss"])]
        if finite_vals:
            bv = min(finite_vals, key=lambda v: v["val_loss"])
            best = {"epoch": bv["epoch"], "val_loss": bv["val_loss"], "val_acc": bv["val_acc"]}
        losses = [r["loss"] for r in self._train
                  if r["loss"] is not None and math.isfinite(r["loss"])]
        last10 = losses[-10:]
        lrs = [r["lr"] for r in self._train if r["lr"] is not None and math.isfinite(r["lr"])]
        gns = [r["grad_norm"] for r in self._train
               if r["grad_norm"] is not None and math.isfinite(r["grad_norm"])]
        sps = [r["steps_per_sec"] for r in self._train if r["steps_per_sec"]]
        bts = [r["batch_time_ms"] for r in self._train if r["batch_time_ms"]]
        va = [r["vram_alloc_mb"] for r in self._train if r["vram_alloc_mb"] is not None]
        vr = [r["vram_reserved_mb"] for r in self._train if r["vram_reserved_mb"] is not None]
        srt_gn = sorted(gns)
        evals_seen = sorted({e["metric"] for e in self._evals})
        return {
            "run_id": self.run_id, "run_dir": str(self.run_dir), "status": status,
            "library": LIB_VERSION, "started_utc": "_".join(self.run_id.split("_")[-2:]) if self.run_id else None,
            "duration_s": round(duration_s, 2), "total_steps": self._g,
            "epochs_seen": len({m[1] for m in self._marks}),
            "best_validation": best,
            "train_loss": {"first": losses[0] if losses else None,
                           "last10_mean": round(sum(last10) / len(last10), 6) if last10 else None,
                           "finite_records": len(losses)},
            "lr": {"peak": max(lrs) if lrs else None, "final": lrs[-1] if lrs else None},
            "grad_norm": {"median": round(srt_gn[len(srt_gn) // 2], 4) if srt_gn else None,
                          "max": round(max(gns), 4) if gns else None},
            "throughput": {"steps_per_sec_mean": round(sum(sps) / len(sps), 3) if sps else None,
                           "batch_time_ms_median": round(sorted(bts)[len(bts) // 2], 3) if bts else None},
            "vram": {"peak_alloc_mb": max(va) if va else None, "peak_reserved_mb": max(vr) if vr else None},
            "health": {"nan_alerts": self.nan_total, "grad_spikes": self.spike_total,
                       "warnings": self.warnings, "errors": self.errors,
                       "monitor_soft_fails": len(self.soft_failures)},
            "checkpoints": self.ckpts, "eval_metrics_logged": len(self._evals),
            "eval_metric_names": evals_seen[:64], "plots": [str(p) for p in plots],
        }

    def plot_all(self, outdir=None):
        """Manually trigger plot generation (also called by finish())."""
        if not _HAS_MPL:
            return []
        outdir = Path(outdir) if outdir else self.run_dir / "plots"
        bundle = self._build_bundle()
        return plot_run(bundle, outdir) or []

    # raw step data for post-hoc tooling
    @property
    def steps_recorded(self):
        return self._g


def _render_report(s: dict) -> str:
    def f(x, d="—"):
        return d if x is None else x

    best = s.get("best_validation") or {}
    lines = [
        f"RUN REPORT — {s.get('run_id')}",
        f"status          : {f(s.get('status'))}",
        f"duration        : {f(s.get('duration_s'))} s   steps: {f(s.get('total_steps'))}"
        f"   epochs: {f(s.get('epochs_seen'))}",
        f"best validation : loss {f(best.get('val_loss'))} @ epoch {f(best.get('epoch'))}"
        f"  (acc {f(best.get('val_acc'))})",
        f"train loss      : first {f((s.get('train_loss') or {}).get('first'))}"
        f" -> last10 mean {f((s.get('train_loss') or {}).get('last10_mean'))}",
        f"learning rate   : peak {f((s.get('lr') or {}).get('peak'))}"
        f" -> final {f((s.get('lr') or {}).get('final'))}",
        f"grad norm       : median {f((s.get('grad_norm') or {}).get('median'))}"
        f"  max {f((s.get('grad_norm') or {}).get('max'))}",
        f"throughput      : {f((s.get('throughput') or {}).get('steps_per_sec_mean'))} steps/s"
        f"  (median batch {f((s.get('throughput') or {}).get('batch_time_ms_median'))} ms)",
        f"VRAM peak       : alloc {f((s.get('vram') or {}).get('peak_alloc_mb'))} MB"
        f" / reserved {f((s.get('vram') or {}).get('peak_reserved_mb'))} MB",
        f"health          : nan_alerts {f((s.get('health') or {}).get('nan_alerts'))}"
        f"  grad_spikes {f((s.get('health') or {}).get('grad_spikes'))}"
        f"  warnings {f((s.get('health') or {}).get('warnings'))}"
        f"  errors {f((s.get('health') or {}).get('errors'))}",
        f"checkpoints     : {f(s.get('checkpoints'))}",
        f"plots           : {len(s.get('plots') or [])} files",
    ]
    return "\n".join(lines) + "\n"


# ============================================================================ PART 2
# Post-hoc loaders + plotting. All plotting operates on a plain "bundle" dict so
# RunMonitor (live) and RunReport (reloaded, possibly crashed) share one code path.
import re  # noqa: E402  (module-level import kept out of hot paths above)

if _HAS_NUMPY:
    np = _np


def _read_csv_rows(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


_STR_COLS = {"ts", "metric", "subset", "extra", "path", "event", "kind", "message",
             "status", "traceback", "context", "name", "recent"}


def _coerce(rows):
    out = []
    for r in rows:
        d = {}
        for k, v in (r or {}).items():
            if k is None or k in _STR_COLS:
                d[k] = v
            elif v is None or v == "":
                d[k] = None
            else:
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
        out.append(d)
    return out


def _bundle(rows, vals, evals, marks, sys_rows, name, nan_alert_steps,
            grad_clip=None, status=""):
    b = {"steps": [], "loss": [], "acc": [], "lr": [], "grad": [], "sps": [],
         "batch_ms": [], "vram_alloc": [], "vram_reserved": [],
         "val_steps": [], "val_loss": [], "val_acc": [],
         "marks": list(marks or []), "evals": {}, "components": {},
         "sys_t": [], "sys_cpu": [], "sys_ram_mb": [], "sys_ram_pct": [],
         "name": name, "nan_alerts": list(nan_alert_steps or []),
         "grad_clip": grad_clip, "status": status}
    for r in rows or []:
        gs = r.get("global_step")
        gs = int(gs) if gs is not None else 0
        b["steps"].append(gs)
        b["loss"].append(r.get("loss"))
        b["acc"].append(r.get("acc"))
        b["lr"].append(r.get("lr"))
        b["grad"].append(r.get("grad_norm"))
        b["sps"].append(r.get("steps_per_sec"))
        b["batch_ms"].append(r.get("batch_time_ms"))
        b["vram_alloc"].append(r.get("vram_alloc_mb"))
        b["vram_reserved"].append(r.get("vram_reserved_mb"))
        ex = r.get("extra")
        if ex:
            try:
                exd = json.loads(ex) if isinstance(ex, str) else ex
            except Exception:
                exd = {}
            comp = (exd or {}).get("components") if isinstance(exd, dict) else None
            if isinstance(comp, dict):
                for k, v in comp.items():
                    b["components"].setdefault(str(k), []).append((gs, _to_float(v)))
    for v in vals or []:
        b["val_steps"].append(int(v.get("step") or 0))
        b["val_loss"].append(v.get("val_loss"))
        b["val_acc"].append(v.get("val_acc"))
    for e in evals or []:
        m, ep, val = e.get("metric"), e.get("epoch"), e.get("value")
        if m is None or val is None:
            continue
        if m not in b["evals"]:
            b["evals"][m] = ([], [])
        xs, ys = b["evals"][m]
        if ep in xs:
            ys[xs.index(ep)] = val
        else:
            xs.append(ep)
            ys.append(val)
    for r in sys_rows or []:
        if isinstance(r, dict):
            b["sys_t"].append(r.get("t_offset_s"))
            b["sys_cpu"].append(r.get("cpu_pct"))
            b["sys_ram_mb"].append(r.get("ram_used_mb"))
            b["sys_ram_pct"].append(r.get("ram_pct"))
        else:
            b["sys_t"].append(r[1] if len(r) > 1 else None)
            b["sys_cpu"].append(r[2] if len(r) > 2 else None)
            b["sys_ram_mb"].append(r[3] if len(r) > 3 else None)
            b["sys_ram_pct"].append(r[4] if len(r) > 4 else None)
    return b


# ------------------------------------------------------------------ plot helpers
def _style(ax, title, xlabel="", ylabel="", size=11):
    ax.set_title(title, fontsize=size)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(8)


def _draw_marks(ax, b):
    marks = b.get("marks") or []
    if marks and len(marks) <= 60:
        for s, _e in marks:
            ax.axvline(s, color="gray", alpha=0.15, lw=0.8)


def _draw_nan_alerts(ax, b):
    for s in (b.get("nan_alerts") or []):
        ax.axvline(s, color="red", alpha=0.5, ls="--", lw=1.0)


def _has(series):
    return any(v is not None for v in series)


def _finish(fig, path, ax, own):
    if own:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)
    return ax


def _plot_loss(b, path=None, ax=None):
    if not _has(b["loss"]) and not _has(b["val_loss"]):
        return None
    own = ax is None
    if own:
        fig = plt.figure(figsize=(8, 4.6), constrained_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = None
    if _has(b["loss"]):
        ax.plot(b["steps"], b["loss"], lw=1.4, alpha=0.85, color="#1f77b4", label="train")
    if _has(b["val_loss"]):
        ax.plot(b["val_steps"], b["val_loss"], "o--", ms=5, lw=1.2, color="#ff7f0e",
                label="validation")
        fv = [v for v in b["val_loss"] if v is not None and math.isfinite(v)]
        if fv:
            by = min(fv)
            i = b["val_loss"].index(by)
            ax.plot([b["val_steps"][i]], [by], marker="*", ms=14, color="#d62728", zorder=5)
            ax.annotate(f"best {by:.4g}", xy=(b["val_steps"][i], by), fontsize=8, color="#d62728")
    _draw_marks(ax, b)
    _draw_nan_alerts(ax, b)
    _style(ax, "loss", "global step", "loss")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    return _finish(fig, path, ax, own)


def _plot_accuracy(b, path=None, ax=None):
    if not _has(b["acc"]) and not _has(b["val_acc"]):
        return None
    own = ax is None
    if own:
        fig = plt.figure(figsize=(8, 4.4), constrained_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = None
    if _has(b["acc"]):
        ax.plot(b["steps"], b["acc"], lw=1.3, alpha=0.85, color="#1f77b4", label="train acc")
    if _has(b["val_acc"]):
        ax.plot(b["val_steps"], b["val_acc"], "o--", ms=5, lw=1.2, color="#2ca02c",
                label="val acc")
    allv = [v for v in (b["acc"] + b["val_acc"]) if v is not None]
    if allv and 0.0 <= min(allv) and max(allv) <= 1.0:
        ax.set_ylim(-0.02, 1.02)
    _draw_marks(ax, b)
    _style(ax, "accuracy", "global step", "accuracy")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    return _finish(fig, path, ax, own)


def _plot_lr(b, path=None, ax=None):
    if not _has(b["lr"]):
        return None
    own = ax is None
    if own:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = None
    ax.plot(b["steps"], b["lr"], lw=1.4, color="#2ca02c")
    fv = [v for v in b["lr"] if v is not None and math.isfinite(v)]
    if fv:
        ax.annotate(f"peak {max(fv):.3g}", xy=(b["steps"][b["lr"].index(max(fv))], max(fv)),
                    fontsize=8, color="#2ca02c")
        ax.annotate(f"final {fv[-1]:.3g}", xy=(b["steps"][-1], fv[-1]), fontsize=8,
                    color="#2ca02c", xytext=(-40, -2), textcoords="offset points")
    _draw_marks(ax, b)
    _style(ax, "learning-rate schedule", "global step", "lr")
    return _finish(fig, path, ax, own)


def _plot_grad(b, path=None, ax=None):
    if not _has(b["grad"]):
        return None
    own = ax is None
    if own:
        fig = plt.figure(figsize=(8, 4.2), constrained_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = None
    ax.plot(b["steps"], b["grad"], lw=1.1, alpha=0.9, color="#9467bd")
    pos = [v for v in b["grad"] if v is not None and math.isfinite(v) and v > 0]
    if b.get("grad_clip"):
        ax.axhline(b["grad_clip"], color="red", ls="--", lw=1, alpha=0.7,
                   label=f"clip={b['grad_clip']:g}")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    if len(pos) > 4 and max(pos) / max(min(pos), 1e-12) > 100:
        ax.set_yscale("log")
    _draw_marks(ax, b)
    _style(ax, "gradient norm", "global step", "||grad||")
    return _finish(fig, path, ax, own)


def _plot_throughput(b, path=None, ax=None):
    if not _has(b["sps"]) and not _has(b["batch_ms"]):
        return None
    own = ax is None
    if own:
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = None
    inst = [(s, 1000.0 / m) for s, m in zip(b["steps"], b["batch_ms"])
            if m is not None and m > 0]
    if inst:
        ax.plot([x[0] for x in inst], [x[1] for x in inst], lw=0.6, alpha=0.3,
                color="#1f77b4", label="per-step")
    if _has(b["sps"]):
        ax.plot(b["steps"], b["sps"], lw=1.5, color="#1f77b4", label="EMA")
    _style(ax, "throughput", "global step", "steps / s")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    return _finish(fig, path, ax, own)


def _plot_resources(b, path=None, ax=None):
    if not _has(b["sys_cpu"]) and not _has(b["vram_alloc"]) and not _has(b["vram_reserved"]):
        return None
    if ax is not None:  # dashboard uses its own resource panel; keep this full-page form
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    if _has(b["sys_cpu"]):
        axes[0].plot(b["sys_t"], b["sys_cpu"], lw=1.2, color="#1f77b4", label="CPU %")
        axes[0].plot(b["sys_t"], b["sys_ram_pct"], lw=1.2, alpha=0.7, color="#ff7f0e",
                     label="RAM %")
        axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    _style(axes[0], "system resources (background samples)", "seconds since run start", "percent")
    if _has(b["vram_alloc"]) or _has(b["vram_reserved"]):
        axes[1].plot(b["steps"], b["vram_alloc"], lw=1.2, color="#2ca02c", label="allocated")
        axes[1].plot(b["steps"], b["vram_reserved"], lw=1.2, alpha=0.7, color="#d62728",
                     label="reserved")
        axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    _style(axes[1], "GPU VRAM", "global step", "MB")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_components(b, path=None, ax=None):
    comps = b.get("components") or {}
    if not comps:
        return None
    names = sorted(comps)
    ncols = min(2, len(names))
    nrows = max(1, math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 2.9 * nrows),
                              constrained_layout=True, squeeze=False)
    for i, name in enumerate(names):
        axx = axes[i // ncols][i % ncols]
        pts = comps[name]
        axx.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.2, color="#17becf")
        _style(axx, f"loss component: {name}", "global step", "")
    for j in range(len(names), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_evals(b, path=None, ax=None):
    em = b.get("evals") or {}
    names = [m for m in sorted(em) if not re.match(r"^L\d+/", m)]
    if not names:
        return None
    ncols = min(3, len(names))
    nrows = max(1, math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.9 * nrows),
                             constrained_layout=True, squeeze=False)
    for i, m in enumerate(names):
        axx = axes[i // ncols][i % ncols]
        xs, ys = em[m]
        axx.plot(xs, ys, marker="o", ms=4, lw=1.4, color="#1f77b4")
        allv = [v for v in ys if v is not None]
        if allv and 0.0 <= min(allv) and max(allv) <= 1.0:
            axx.set_ylim(-0.02, 1.02)
        _style(axx, m, "epoch", "", size=9)
    for j in range(len(names), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _plot_levels(b, path=None, ax=None):
    """Heatmaps of per-difficulty-level metrics (curriculum tracking): rows = epochs,
    columns = difficulty levels L1..L9. Greener = better (direction-aware)."""
    if not _HAS_NUMPY:
        return None
    em = b.get("evals") or {}
    lv_metrics = {m: v for m, v in em.items() if re.match(r"^L\d+/", m)}
    if not lv_metrics:
        return None
    levels = sorted({int(m.split("/")[0][1:]) for m in lv_metrics})
    prio = ["loss", "reaction_acc", "synthon_EM", "gen_validity", "tanimoto_mean"]
    suffixes = [s for s in prio if any(m.endswith("/" + s) for m in lv_metrics)]
    if not suffixes:
        suffixes = [sorted(lv_metrics)[0].split("/", 1)[1]]
    suffixes = suffixes[:2]
    n = len(suffixes)
    fig, axes = plt.subplots(1, n, figsize=(6.8 * n, 4.6), constrained_layout=True,
                             squeeze=False)
    for i, suf in enumerate(suffixes):
        axx = axes[0][i]
        per = {}
        epochs = set()
        for lv in levels:
            m = f"L{lv}/{suf}"
            if m in em:
                xs, ys = em[m]
                per[lv] = dict(zip(xs, ys))
                epochs.update(xs)
        epochs = sorted(e for e in epochs if e is not None)
        mat = np.full((max(len(epochs), 1), len(levels)), np.nan)
        for j, lv in enumerate(levels):
            for r, e in enumerate(epochs):
                v = per.get(lv, {}).get(e)
                if v is not None and math.isfinite(v):
                    mat[r, j] = v
        higher_better = suf not in ("loss", "nll", "error", "ce", "perplexity")
        cmap = "RdYlGn" if higher_better else "RdYlGn_r"
        vmin, vmax = np.nanmin(mat), np.nanmax(mat)
        if not (vmax > vmin):
            vmax = vmin + 1e-9
        im = axx.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axx.set_xticks(range(len(levels)))
        axx.set_xticklabels([f"L{lv}" for lv in levels], fontsize=8)
        if epochs:
            axx.set_yticks(range(len(epochs)))
            axx.set_yticklabels([str(int(e)) for e in epochs], fontsize=8)
        axx.set_xlabel("difficulty level", fontsize=9)
        axx.set_ylabel("epoch", fontsize=9)
        direction = "higher = better" if higher_better else "lower = better"
        axx.set_title(f"{suf} by difficulty level ({direction})", fontsize=11)
        if len(epochs) * len(levels) <= 120:
            span = vmax - vmin
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = mat[r, c]
                    if math.isfinite(v):
                        rel = (v - vmin) / span if span else 0.5
                        axx.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                 color="black" if 0.25 < rel < 0.75 else "white")
        fig.colorbar(im, ax=axx, fraction=0.046)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _dashboard_evals(b, path=None, ax=None):
    em = b.get("evals") or {}
    prio = ["reaction_acc", "reaction_topk@1", "reaction_topk@5", "synthon_rank_acc@1",
            "synthon_rank_acc@5", "synthon_EM", "gen_validity", "gen_uniqueness",
            "gen_diversity", "gen_novelty"]
    keys = [k for k in prio if k in em][:5]
    if not keys:
        keys = [k for k in sorted(em) if not re.match(r"^L\d+/", k)][:5]
    if not keys:
        return None
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    allv = []
    for i, k in enumerate(keys):
        xs, ys = em[k]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color=colors[i % len(colors)], label=k)
        allv += [v for v in ys if v is not None]
    if allv and 0.0 <= min(allv) and max(allv) <= 1.0:
        ax.set_ylim(-0.02, 1.02)
    _style(ax, "evaluation metrics", "epoch", "", size=10)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    return ax


def _plot_dashboard(b, path=None, ax=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [_plot_loss, _plot_accuracy, _plot_lr, _plot_grad, _plot_throughput,
              _dashboard_evals]
    for i, fn in enumerate(panels):
        axx = axes[i // 3][i % 3]
        try:
            r = fn(b, ax=axx)
        except Exception:
            r = None
        if r is None:
            axx.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9,
                     color="gray", transform=axx.transAxes)
            axx.set_xticks([])
            axx.set_yticks([])
    fig.suptitle(f"{b.get('name', 'run')} — training dashboard", fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


_PLOTTERS = [
    ("loss", _plot_loss), ("accuracy", _plot_accuracy), ("lr_schedule", _plot_lr),
    ("grad_norm", _plot_grad), ("throughput", _plot_throughput),
    ("resources", _plot_resources), ("component_losses", _plot_components),
    ("evaluation_metrics", _plot_evals), ("level_heatmaps", _plot_levels),
    ("dashboard", _plot_dashboard),
]


def plot_run(bundle, outdir):
    """Render every applicable plot for a bundle. Returns list of generated names."""
    if not _HAS_MPL:
        return []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    generated = []
    for name, fn in _PLOTTERS:
        try:
            p = fn(bundle, outdir / f"{name}.png")
            if p:
                generated.append(name)
        except Exception as e:  # noqa: BLE001
            try:
                print(f"[train_observatory] plot '{name}' failed: {e!r}", file=sys.stderr)
            except Exception:
                pass
    return generated


# ------------------------------------------------------------------------ RunReport
class RunReport:
    """Post-hoc loader for a run directory. Tolerates crashed/partial runs:
    every artifact is optional. Use it to reload, re-plot and compare any run."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.manifest = self._load_json(self.run_dir / "manifest.json") or {}
        self.summary = self._load_json(self.run_dir / "summary.json") or {}
        self.name = self.manifest.get("run_id") or self.run_dir.name
        self.events = self._load_jsonl(self.run_dir / "events.jsonl")
        self.checkpoints = self._load_jsonl(self.run_dir / "checkpoints.jsonl")
        self.metrics = _coerce(_read_csv_rows(self.run_dir / "metrics.csv"))
        self.evaluations = _coerce(_read_csv_rows(self.run_dir / "evaluations.csv"))
        self.system = _coerce(_read_csv_rows(self.run_dir / "system.csv"))
        self.validations = [e for e in self.events if e.get("kind") == "validation"]
        self.marks = [(e.get("step", 0), e.get("epoch"))
                      for e in self.events if e.get("kind") == "epoch_start"]
        self.nan_alert_steps = [e.get("step", 0) for e in self.events
                                if e.get("kind") == "numeric_alert"]
        cfg = self.manifest.get("config") or {}
        self.grad_clip = None
        for key in ("grad_clip", "clip", "max_grad_norm", "clip_grad_norm", "clip_norm"):
            if isinstance(cfg, dict) and cfg.get(key) is not None:
                self.grad_clip = _to_float(cfg[key])
                break

    @staticmethod
    def _load_json(p):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _load_jsonl(p):
        out = []
        p = Path(p)
        if not p.exists():
            return out
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            pass
        return out

    def bundle(self):
        vals = [{"epoch": e.get("epoch"), "step": e.get("step", 0),
                 "val_loss": e.get("val_loss"), "val_acc": e.get("val_acc")}
                for e in self.validations]
        return _bundle(rows=self.metrics, vals=vals, evals=self.evaluations,
                       marks=self.marks, sys_rows=self.system, name=self.name,
                       nan_alert_steps=self.nan_alert_steps,
                       grad_clip=self.grad_clip,
                       status=str(self.summary.get("status") or "?"))

    def plot_all(self, outdir=None):
        outdir = Path(outdir) if outdir else self.run_dir / "plots_reloaded"
        return plot_run(self.bundle(), outdir)

    def best_val(self):
        fv = [e for e in self.validations
              if isinstance(e.get("val_loss"), (int, float)) and math.isfinite(e["val_loss"])]
        if not fv:
            return None
        e = min(fv, key=lambda x: x["val_loss"])
        return {"epoch": e.get("epoch"), "step": e.get("step", 0),
                "val_loss": e["val_loss"], "val_acc": e.get("val_acc")}

    def __repr__(self):
        return (f"<RunReport {self.name}: {len(self.metrics)} steps, "
                f"{len(self.evaluations)} evals, status={self.summary.get('status', '?')}>")


def compare_runs(run_dirs, out="runs/comparison.png", metric="loss", title=None):
    """Overlay one metric (default: train loss) across several runs + summary table.
    Works on finished AND crashed runs (RunReport is crash-tolerant)."""
    if not _HAS_MPL:
        return []
    reps = [RunReport(d) for d in run_dirs]
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    table = []
    for rep in reps:
        rows = [r for r in rep.metrics if r.get(metric) is not None]
        if rows:
            xs = [int(r.get("global_step") or 0) for r in rows]
            ys = [r.get(metric) for r in rows]
            ax.plot(xs, ys, lw=1.3, alpha=0.85, label=rep.name)
        if rep.validations:
            vx = [int(e.get("step", 0)) for e in rep.validations
                  if isinstance(e.get("val_loss"), (int, float))]
            vy = [e.get("val_loss") for e in rep.validations
                  if isinstance(e.get("val_loss"), (int, float))]
            ax.plot(vx, vy, "o", ms=5, alpha=0.8, label=f"{rep.name} (val)")
        fl = [r.get(metric) for r in rep.metrics
              if r.get(metric) is not None and math.isfinite(r.get(metric))]
        bv = rep.best_val()
        table.append({"run": rep.name, "steps": len(rep.metrics),
                      "best_val_loss": bv and bv.get("val_loss"),
                      "final_train_loss": fl[-1] if fl else None,
                      "status": rep.summary.get("status", "?"),
                      "nan_alerts": len(rep.nan_alert_steps)})
    _style(ax, title or f"run comparison — {metric}", "global step", metric)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    txt = out.with_name(out.stem + "_summary.txt")
    lines = ["run,steps,best_val_loss,final_train_loss,status,nan_alerts"]
    for t in table:
        lines.append(",".join(str(t[k]) for k in
                              ("run", "steps", "best_val_loss", "final_train_loss",
                               "status", "nan_alerts")))
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


# ----------------------------------------------------------------------- smoke demo
if __name__ == "__main__":  # `python train_observatory.py` → 30-second smoke demo
    mon = RunMonitor(run_name="smoke_demo", config={"lr": 3e-3, "grad_clip": 1.0},
                    run_root="runs")
    total_steps = 30
    for ep in range(3):
        with mon.epoch(ep, n_batches=10):
            for i in range(10):
                g = mon.steps_recorded + 1
                loss = 2.5 * math.exp(-0.06 * g) + 0.05 * math.sin(g) + 0.35
                lr = 3e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * g / total_steps)))
                mon.step(epoch=ep, batch=i, n_batches=10, loss=loss, lr=lr,
                         grad_norm=0.5 + 0.4 * abs(math.sin(g)), batch_size=64,
                         acc=min(0.95, 0.4 + 0.02 * g),
                         components={"demo_ce": loss * 0.6, "demo_nll": loss * 0.4})
        mon.validation(ep, val_loss=2.0 - 0.3 * ep, val_acc=0.55 + 0.1 * ep)
        mon.evaluation(ep, {"reaction_acc": 0.5 + 0.1 * ep, "synthon_EM": 0.2 + 0.08 * ep,
                            "gen_validity": 0.9})
        for lv in range(1, 10):
            mon.evaluation(ep, {"loss": 1.9 - 0.25 * ep - lv * 0.02,
                                "reaction_acc": max(0.05, 0.4 + 0.08 * ep - lv * 0.01)},
                           level=lv)
    mon.finish("completed")
    print("demo run dir:", mon.run_dir)

