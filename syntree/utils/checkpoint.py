"""Resilient checkpointing with bidirectional Hugging Face Hub sync.

Guarantees progress preservation across Colab/Kaggle runtime preemptions:

* every epoch (and every ``push_every_n_epochs`` epochs) a checkpoint
  containing model, optimizer, scheduler, RNG and metric state is written
  locally,
* checkpoints are asynchronously uploaded to a private HF Hub repository,
* ``restore_latest`` auto-resumes: local manifest first, HF Hub fallback,
* stale local checkpoints are pruned to ``keep_last_n``.

Security note: the write token is read exclusively from the ``HF_TOKEN``
environment variable (never from config files, which may be committed).
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Local + Hugging Face Hub checkpoint orchestrator."""

    MANIFEST_NAME = "manifest.json"

    def __init__(
        self,
        config: dict,
        keep_last_n: int = 3,
        ckpt_dir: str = "./checkpoints",
    ):
        self.config = config
        hf_cfg = config.get("huggingface", {})
        self.repo_id = hf_cfg.get("repo_id", "")
        self.enabled = bool(hf_cfg.get("enabled", False))
        self.private = bool(hf_cfg.get("private", True))
        self.push_freq = int(hf_cfg.get("push_every_n_epochs", 2))
        self.keep_last_n = max(1, int(keep_last_n))

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.token = os.environ.get("HF_TOKEN")
        self.api = None
        self._upload_lock = threading.Lock()
        if self.enabled:
            if not self.token:
                logger.warning(
                    "HF sync enabled but HF_TOKEN is not set; "
                    "checkpoints will remain local only."
                )
                self.enabled = False
            else:
                self._init_api()

    # ------------------------------------------------------------------
    # Hub initialisation
    # ------------------------------------------------------------------
    def _init_api(self) -> None:
        try:
            from huggingface_hub import HfApi, create_repo

            self.api = HfApi(token=self.token)
            if self.repo_id:
                create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    exist_ok=True,
                    private=self.private,
                )
            else:
                logger.warning("Empty hf repo_id; hub sync disabled.")
                self.enabled = False
        except Exception as exc:
            logger.warning("HF Hub initialisation failed (%s); local only.", exc)
            self.enabled = False

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
        final: bool = False,
        model_config: Optional[dict] = None,
    ) -> Optional[Path]:
        """Persist a checkpoint and (optionally) sync it to the Hub.

        Args:
            model_config: the effective ``config["model"]`` block the model
                was built with (may be GPU-auto-scaled). Stored so that
                ``generate`` / resume runs can rebuild the exact trained
                architecture before loading the state dict.

        Returns the local checkpoint path.
        """
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
        state = {
            "epoch": int(epoch),
            "step": int(step),
            "model_state_dict": _to_cpu_state(model.state_dict()),
            "model_config": model_config,
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "rng_states": self._capture_rng(),
            "metrics": metrics or {},
            "is_best": bool(is_best),
            "timestamp": time.time(),
        }
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, ckpt_path)

        manifest = {
            "latest_epoch": int(epoch),
            "latest_step": int(step),
            "best_metric": (metrics or {}).get("val_loss", None),
            "is_best": bool(is_best),
            "last_updated": time.asctime(),
        }
        if is_best:
            best_src = ckpt_path
            best_dst = self.ckpt_dir / "checkpoint_best.pt"
            torch.save(state, best_dst)
            _ = best_src, best_dst
        with open(self.ckpt_dir / self.MANIFEST_NAME, "w") as f:
            json.dump(manifest, f, indent=2)

        self._prune_old_checkpoints(keep=self.keep_last_n)

        should_push = self.enabled and self.api is not None and (
            final or is_best or (epoch % self.push_freq == 0)
        )
        if should_push:
            self._sync_to_hub_async(ckpt_path, epoch, manifest)
        return ckpt_path

    # ------------------------------------------------------------------
    # Restoration
    # ------------------------------------------------------------------
    def restore_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
    ) -> Tuple[int, int, float]:
        """Auto-resume from the freshest available checkpoint.

        Order of preference: local manifest -> HF Hub -> fresh start.

        Returns:
            ``(start_epoch, global_step, best_val_loss)``.
        """
        manifest_path = self.ckpt_dir / self.MANIFEST_NAME

        if not manifest_path.exists() and self.enabled and self.api is not None:
            self._pull_manifest_from_hub()

        if not manifest_path.exists():
            logger.info("No checkpoint found anywhere; starting from epoch 0.")
            return 0, 0, float("inf")

        with open(manifest_path) as f:
            manifest = json.load(f)
        latest_epoch = int(manifest["latest_epoch"])
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{latest_epoch}.pt"

        if not ckpt_path.exists():
            if not self._pull_checkpoint_from_hub(latest_epoch):
                logger.warning(
                    "Manifest points to missing checkpoint %s; starting fresh.",
                    ckpt_path,
                )
                return 0, 0, float("inf")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing or unexpected:
            logger.warning(
                "Checkpoint architecture mismatch: missing=%s unexpected=%s. "
                "New modules keep their initialized weights.",
                missing,
                unexpected,
            )
        if optimizer is not None and checkpoint.get("optimizer_state_dict"):
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, RuntimeError) as exc:
                # Legacy checkpoints may have a different parameter-group
                # topology after adding the reaction-family head.
                logger.warning(
                    "Skipping incompatible optimizer state from legacy checkpoint: %s",
                    exc,
                )
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Skipping incompatible scheduler state from legacy checkpoint: %s",
                    exc,
                )
        self._restore_rng(checkpoint.get("rng_states"))

        best = float("inf")
        if checkpoint.get("metrics"):
            val = checkpoint["metrics"].get("val_loss")
            if val is not None:
                best = float(val)
        elif manifest.get("best_metric") is not None:
            best = float(manifest["best_metric"])

        logger.info(
            "Restored checkpoint: epoch %d (step %d).", latest_epoch, checkpoint["step"]
        )
        return latest_epoch + 1, int(checkpoint["step"]), best

    def read_model_config(self) -> Optional[dict]:
        """Return the model architecture stored in the latest checkpoint.

        Used by ``main.py`` to rebuild the model with the exact (possibly
        GPU-auto-scaled) dimensions it was trained with, before the weights
        are restored. Returns ``None`` when no checkpoint exists or the
        stored checkpoint predates architecture persistence.
        """
        manifest_path = self.ckpt_dir / self.MANIFEST_NAME
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            ckpt_path = self.ckpt_dir / (
                f"checkpoint_epoch_{int(manifest['latest_epoch'])}.pt"
            )
            if not ckpt_path.exists():
                return None
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            return checkpoint.get("model_config")
        except Exception as exc:
            logger.warning("Could not read model config from checkpoint: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Hub transfer helpers
    # ------------------------------------------------------------------
    def _sync_to_hub_async(self, ckpt_path: Path, epoch: int, manifest: Dict) -> None:
        """Upload checkpoint + manifest on a daemon thread (never blocks
        the GPU training loop)."""

        def _upload() -> None:
            with self._upload_lock:
                try:
                    logger.info("Uploading epoch %d checkpoint to HF Hub...", epoch)
                    self.api.upload_file(
                        path_or_fileobj=str(ckpt_path),
                        path_in_repo=f"checkpoints/checkpoint_epoch_{epoch}.pt",
                        repo_id=self.repo_id,
                        repo_type="model",
                    )
                    self.api.upload_file(
                        path_or_fileobj=str(self.ckpt_dir / self.MANIFEST_NAME),
                        path_in_repo=self.MANIFEST_NAME,
                        repo_id=self.repo_id,
                        repo_type="model",
                    )
                    logger.info("HF Hub sync complete (epoch %d).", epoch)
                except Exception as exc:
                    logger.warning("HF Hub upload failed: %s. Local copy is safe.", exc)

        threading.Thread(target=_upload, daemon=True).start()

    def _pull_manifest_from_hub(self) -> None:
        try:
            logger.info("No local manifest; querying HF Hub for latest state...")
            self.api.hf_hub_download(
                repo_id=self.repo_id,
                filename=self.MANIFEST_NAME,
                local_dir=str(self.ckpt_dir),
            )
        except Exception:
            logger.info("No remote checkpoint on HF Hub; starting from epoch 0.")

    def _pull_checkpoint_from_hub(self, epoch: int) -> bool:
        try:
            self.api.hf_hub_download(
                repo_id=self.repo_id,
                filename=f"checkpoints/checkpoint_epoch_{epoch}.pt",
                local_dir=str(self.ckpt_dir),
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def _prune_old_checkpoints(self, keep: int) -> None:
        ckpts = sorted(
            self.ckpt_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: _epoch_from_name(p.name),
        )
        for old in ckpts[:-keep]:
            try:
                old.unlink()
            except OSError:  # pragma: no cover
                pass

    @staticmethod
    def _capture_rng() -> Dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    @staticmethod
    def _restore_rng(states: Optional[Dict]) -> None:
        if not states:
            return
        try:
            if "python" in states:
                random.setstate(states["python"])
            if "numpy" in states:
                np.random.set_state(states["numpy"])
            if "torch" in states:
                torch.set_rng_state(states["torch"])
            if states.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(states["cuda"])
        except Exception:  # pragma: no cover - RNG restore is best-effort
            logger.warning("RNG state restoration failed; continuing.")


def _epoch_from_name(name: str) -> int:
    try:
        return int(name.replace("checkpoint_epoch_", "").replace(".pt", ""))
    except ValueError:
        return -1


def _to_cpu_state(state_dict) -> Dict:
    return {
        k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in state_dict.items()
    }


def verify_hf_sync(repo_id: str) -> Dict:
    """Standalone verification used by the notebook's final cell.

    Queries the HF Hub for the latest checkpoint of ``repo_id`` and returns
    a status dict (never raises; failures are reported in ``sync_ok``).
    """
    token = os.environ.get("HF_TOKEN")
    status = {
        "repo_id": repo_id,
        "latest_remote_checkpoint": None,
        "epochs_completed": 0,
        "sync_ok": False,
    }
    if not token:
        status["error"] = "HF_TOKEN not set"
        return status
    try:
        from huggingface_hub import hf_hub_download

        local_manifest = hf_hub_download(
            repo_id=repo_id, filename=CheckpointManager.MANIFEST_NAME, token=token
        )
        with open(local_manifest) as f:
            manifest = json.load(f)
        status["latest_remote_checkpoint"] = f"checkpoint_epoch_{manifest['latest_epoch']}.pt"
        status["epochs_completed"] = int(manifest["latest_epoch"]) + 1
        status["sync_ok"] = True
    except Exception as exc:
        status["error"] = str(exc)
    return status


__all__ = ["CheckpointManager", "verify_hf_sync"]
