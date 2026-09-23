"""Structured JSON logging and experiment tracking.

Provides:
* :class:`StructuredLogger` – append-only JSONL metric logging with
  optional Weight & Biases mirroring when ``WANDB_API_KEY`` is present,
* :func:`configure_logging` – stdlib logging setup used by every entrypoint.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure console (+ optional file) logging on the root logger.

    Repeated calls rebuild the handler set, so a later call with a
    ``log_file`` takes effect even after an earlier console-only setup.
    """
    root = logging.getLogger()

    formatter = logging.Formatter(_LOG_FORMAT)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    root.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)


class StructuredLogger:
    """Append-only JSONL logger with optional WandB mirroring.

    Each ``log(record)`` call writes one JSON line to disk. When the
    ``WANDB_API_KEY`` environment variable is set and ``wandb`` imports,
    records are mirrored to a WandB run (created lazily).
    """

    def __init__(self, path: str, experiment_name: str = "3d-syntree"):
        self.path = str(path)
        self.experiment_name = str(experiment_name)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._wandb = None
        self._wandb_run = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb  # type: ignore

                self._wandb = wandb
            except ImportError:
                pass

    def log(self, record: Dict[str, Any]) -> None:
        """Write one structured record."""
        entry = dict(record)
        entry.setdefault("wall_time", time.time())
        entry.setdefault("experiment", self.experiment_name)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self._mirror_wandb(entry)

    def _mirror_wandb(self, entry: Dict[str, Any]) -> None:
        if self._wandb is None:
            return
        try:
            if self._wandb_run is None:
                self._wandb_run = self._wandb.init(
                    project=self.experiment_name,
                    name=os.path.basename(os.path.dirname(self.path)) or None,
                    resume="allow",
                )
            numeric = {
                k: v
                for k, v in entry.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric:
                self._wandb_run.log(numeric)
        except Exception:  # pragma: no cover - tracking must never crash training
            self._wandb = None

    def read_all(self):
        """Replay the log (for dashboards / tests)."""
        records = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return records


__all__ = ["StructuredLogger", "configure_logging"]
