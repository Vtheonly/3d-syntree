"""Resilient checkpointing with authoritative Hugging Face Hub state.

Guarantees:
* every fully completed epoch is recorded in an atomic progress.json,
* progress.json must contain strictly contiguous completed epochs,
* when Hub sync is enabled, the remote Hub state is authoritative,
* stale local checkpoints are purged when the remote run was wiped/corrupt,
* regular uploads may run asynchronously, but final checkpoint uploads are
  synchronous and explicitly flush all earlier uploads before returning,
* checkpoint downloads from checkpoints/... are materialized into the manager's
  local checkpoint directory (never a nested directory).
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import shutil
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Local + Hugging Face Hub checkpoint orchestrator."""

    MANIFEST_NAME = "manifest.json"
    PROGRESS_NAME = "progress.json"
    PROGRESS_VERSION = 1

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
        self.private = bool(hf_cfg.get("private", False))
        self.push_freq = max(1, int(hf_cfg.get("push_every_n_epochs", 2)))
        self.keep_last_n = max(1, int(keep_last_n))

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.token = os.environ.get("HF_TOKEN")
        self.api = None
        self._upload_lock = threading.Lock()
        self._upload_executor: Optional[ThreadPoolExecutor] = None
        self._upload_futures: List[Future] = []

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

    def _ensure_upload_executor(self) -> ThreadPoolExecutor:
        if self._upload_executor is None:
            # Non-daemon executor workers cannot be abandoned mid-socket-write
            # when the Python interpreter exits.
            self._upload_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="syntree-hf-upload",
            )
        return self._upload_executor

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------
    @classmethod
    def _empty_progress(cls) -> Dict:
        return {
            "version": cls.PROGRESS_VERSION,
            "completed_epochs": [],
            "last_sequential_epoch": -1,
            "history": [],
        }

    def _read_progress(self) -> Dict:
        path = self.ckpt_dir / self.PROGRESS_NAME
        if not path.exists():
            return self._empty_progress()
        try:
            with open(path, "r", encoding="utf-8") as f:
                progress = json.load(f)
            return progress if isinstance(progress, dict) else self._empty_progress()
        except (OSError, json.JSONDecodeError, TypeError):
            return self._empty_progress()

    @classmethod
    def _validate_progress(cls, progress: Dict) -> Tuple[bool, int]:
        """Return (valid, last_contiguous_epoch)."""
        if not isinstance(progress, dict):
            return False, -1

        completed = progress.get("completed_epochs")
        last_seq = progress.get("last_sequential_epoch")

        if not isinstance(completed, list):
            return False, -1
        if any(
            not isinstance(epoch, int) or isinstance(epoch, bool)
            for epoch in completed
        ):
            return False, -1

        expected = list(range(len(completed)))
        if completed != expected:
            return False, -1

        expected_last = completed[-1] if completed else -1
        if last_seq != expected_last:
            return False, -1

        return True, expected_last

    def _record_progress(self, epoch: int, step: int, metrics: Dict) -> Dict:
        progress = self._read_progress()
        valid, last_seq = self._validate_progress(progress)
        if not valid:
            raise RuntimeError(
                "progress.json is missing/corrupt/non-contiguous; refusing to "
                f"record completed epoch {epoch}."
            )

        if epoch not in progress["completed_epochs"]:
            if epoch != last_seq + 1:
                raise RuntimeError(
                    "Non-sequential epoch completion detected: "
                    f"last completed={last_seq}, attempted={epoch}."
                )
            progress["completed_epochs"].append(epoch)
            progress["last_sequential_epoch"] = epoch
        elif epoch != progress["last_sequential_epoch"]:
            raise RuntimeError(
                f"Epoch {epoch} is already recorded out of order."
            )

        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

        history = progress.setdefault("history", [])
        if progress["completed_epochs"] and (
            not history or history[-1].get("epoch") != epoch
        ):
            history.append(
                {
                    "epoch": int(epoch),
                    "step": int(step),
                    "metrics": numeric_metrics,
                    "timestamp": time.time(),
                }
            )
        elif history and history[-1].get("epoch") == epoch:
            history[-1].update(
                {
                    "step": int(step),
                    "metrics": numeric_metrics,
                    "timestamp": time.time(),
                }
            )

        progress["version"] = self.PROGRESS_VERSION
        self._atomic_json_write(self.ckpt_dir / self.PROGRESS_NAME, progress)
        return progress

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
        """Persist a fully completed epoch and optionally sync it to the Hub."""
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("checkpoint epoch must be >= 0")

        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
        state = {
            "epoch": epoch,
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

        progress = self._record_progress(epoch, step, metrics or {})
        manifest = {
            "version": 2,
            "latest_epoch": epoch,
            "latest_step": int(step),
            "best_metric": (metrics or {}).get("val_loss", None),
            "is_best": bool(is_best),
            "last_sequential_epoch": int(progress["last_sequential_epoch"]),
            "last_updated": time.asctime(),
        }
        self._atomic_json_write(self.ckpt_dir / self.MANIFEST_NAME, manifest)

        if is_best:
            best_dst = self.ckpt_dir / "checkpoint_best.pt"
            best_tmp = best_dst.with_suffix(".pt.tmp")
            torch.save(state, best_tmp)
            os.replace(best_tmp, best_dst)

        self._prune_old_checkpoints(keep=self.keep_last_n)

        should_push = self.enabled and self.api is not None and (
            final or is_best or (epoch % self.push_freq == 0)
        )
        if should_push:
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            progress_bytes = json.dumps(progress, indent=2).encode("utf-8")
            if final:
                self._sync_to_hub_sync(
                    ckpt_path, epoch, manifest_bytes, progress_bytes
                )
            else:
                self._sync_to_hub_async(
                    ckpt_path, epoch, manifest_bytes, progress_bytes
                )

        return ckpt_path

    # ------------------------------------------------------------------
    # Restoration with remote authoritative verification
    # ------------------------------------------------------------------
    def restore_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
    ) -> Tuple[int, int, float]:
        """Restore the freshest verified checkpoint.

        With Hub sync enabled, the remote repository is authoritative:
        local state is never trusted until remote manifest + progress + latest
        checkpoint presence are verified.
        """
        manifest_path = self.ckpt_dir / self.MANIFEST_NAME

        if self.enabled and self.api is not None:
            remote_files = self._list_remote_files()
            if remote_files is None:
                logger.warning(
                    "[checkpoint] Unable to verify HF Hub state; local checkpoint "
                    "files are not authoritative, so stale local state is purged "
                    "and this process starts from epoch 0."
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

            remote_manifest_exists = self.MANIFEST_NAME in remote_files
            remote_progress_exists = self.PROGRESS_NAME in remote_files
            remote_checkpoint_files = {
                name
                for name in remote_files
                if name.startswith("checkpoints/checkpoint_epoch_")
                and name.endswith(".pt")
            }

            if (
                not remote_manifest_exists
                or not remote_progress_exists
                or not remote_checkpoint_files
            ):
                logger.warning(
                    "[checkpoint] Remote HF repository contains no complete "
                    "checkpoint state. Purging stale local checkpoint ghosts "
                    "and starting fresh from epoch 0."
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

            if not self._pull_manifest_from_hub() or not self._pull_progress_from_hub():
                logger.warning(
                    "[checkpoint] Failed to materialize verified remote metadata; "
                    "purging local checkpoint state and starting fresh."
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[checkpoint] Remote manifest is unreadable (%s); starting fresh.",
                    exc,
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

            progress = self._read_progress()
            progress_valid, last_seq = self._validate_progress(progress)
            if not progress_valid or last_seq < 0:
                logger.warning(
                    "[checkpoint] Remote progress.json is missing/corrupt/non-contiguous; "
                    "rejecting the remote state and starting fresh."
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

            try:
                latest_epoch = int(manifest["latest_epoch"])
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "[checkpoint] Remote manifest has no valid latest_epoch; starting fresh."
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

            if latest_epoch != last_seq:
                logger.warning(
                    "[checkpoint] Remote sequential progress ends at epoch %d, "
                    "but manifest claims epoch %d. Rejecting the state and "
                    "starting fresh.",
                    last_seq,
                    latest_epoch,
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

            latest_name = f"checkpoints/checkpoint_epoch_{latest_epoch}.pt"
            if latest_name not in remote_checkpoint_files:
                logger.warning(
                    "[checkpoint] Remote manifest points to missing checkpoint %s; "
                    "starting fresh.",
                    latest_name,
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

            self._purge_all_checkpoints()
            if not self._pull_checkpoint_from_hub(latest_epoch):
                logger.warning(
                    "[checkpoint] Verified remote checkpoint could not be downloaded; "
                    "starting fresh."
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

        if not manifest_path.exists():
            logger.info("No checkpoint found; starting from epoch 0.")
            return 0, 0, float("inf")

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            latest_epoch = int(manifest["latest_epoch"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "[checkpoint] Corrupt local manifest (%s); starting fresh.",
                exc,
            )
            self._purge_all_checkpoints()
            return 0, 0, float("inf")

        progress = self._read_progress()
        progress_valid, last_seq = self._validate_progress(progress)
        if not progress_valid or last_seq != latest_epoch:
            logger.warning(
                "[checkpoint] Local manifest/progress state is inconsistent; "
                "starting fresh."
            )
            self._purge_all_checkpoints()
            return 0, 0, float("inf")

        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{latest_epoch}.pt"
        if not ckpt_path.exists():
            if self.enabled and self.api is not None:
                if not self._pull_checkpoint_from_hub(latest_epoch):
                    logger.warning(
                        "Manifest points to missing checkpoint %s; starting fresh.",
                        ckpt_path,
                    )
                    self._purge_all_checkpoints()
                    return 0, 0, float("inf")
            else:
                logger.warning(
                    "Manifest points to missing checkpoint %s; starting fresh.",
                    ckpt_path,
                )
                self._purge_all_checkpoints()
                return 0, 0, float("inf")

        try:
            checkpoint = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            logger.warning(
                "[checkpoint] Failed to load %s (%s); starting fresh.",
                ckpt_path,
                exc,
            )
            self._purge_all_checkpoints()
            return 0, 0, float("inf")

        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        if checkpoint_epoch != latest_epoch:
            logger.warning(
                "[checkpoint] Checkpoint payload epoch=%d does not match manifest=%d; "
                "starting fresh.",
                checkpoint_epoch,
                latest_epoch,
            )
            self._purge_all_checkpoints()
            return 0, 0, float("inf")

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
                logger.warning(
                    "Skipping incompatible optimizer state from checkpoint: %s", exc
                )

        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Skipping incompatible scheduler state from checkpoint: %s", exc
                )

        self._restore_rng(checkpoint.get("rng_states"))

        best = float("inf")
        if checkpoint.get("metrics"):
            val = checkpoint["metrics"].get("val_loss")
            if val is not None:
                best = float(val)
        elif manifest.get("best_metric") is not None:
            best = float(manifest["best_metric"])

        next_epoch = latest_epoch + 1
        logger.info(
            "Restored verified checkpoint: epoch %d (step %d). Training resumes at epoch %d.",
            latest_epoch,
            int(checkpoint.get("step", 0)),
            next_epoch,
        )
        return next_epoch, int(checkpoint.get("step", 0)), best

    def read_model_config(self) -> Optional[dict]:
        """Return the model architecture stored in the authoritative state."""
        remote_files = None
        if self.enabled and self.api is not None:
            remote_files = self._list_remote_files()
            if remote_files is None:
                return None
            if (
                self.MANIFEST_NAME not in remote_files
                or self.PROGRESS_NAME not in remote_files
            ):
                self._purge_all_checkpoints()
                return None
            if not self._pull_manifest_from_hub() or not self._pull_progress_from_hub():
                self._purge_all_checkpoints()
                return None

        manifest_path = self.ckpt_dir / self.MANIFEST_NAME
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            progress = self._read_progress()
            valid, last_seq = self._validate_progress(progress)
            latest_epoch = int(manifest["latest_epoch"])
            if not valid or last_seq != latest_epoch:
                self._purge_all_checkpoints()
                return None

            remote_checkpoint_name = (
                f"checkpoints/checkpoint_epoch_{latest_epoch}.pt"
            )
            if remote_files is not None and remote_checkpoint_name not in remote_files:
                self._purge_all_checkpoints()
                return None

            # When Hub sync is authoritative, never reuse a local checkpoint
            # without first proving that this exact remote checkpoint exists.
            ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{latest_epoch}.pt"
            if remote_files is not None:
                self._purge_all_checkpoints()
                if not self._pull_checkpoint_from_hub(latest_epoch):
                    self._purge_all_checkpoints()
                    return None
            elif not ckpt_path.exists():
                return None

            checkpoint = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
            if int(checkpoint.get("epoch", -1)) != latest_epoch:
                self._purge_all_checkpoints()
                return None
            return checkpoint.get("model_config")
        except Exception as exc:
            logger.warning("Could not read model config from checkpoint: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Hub transfer helpers
    # ------------------------------------------------------------------
    def _list_remote_files(self) -> Optional[set]:
        if not self.api:
            return None
        try:
            return set(
                self.api.list_repo_files(
                    repo_id=self.repo_id,
                    repo_type="model",
                )
            )
        except Exception as exc:
            logger.warning("[checkpoint] HF Hub listing failed: %s", exc)
            return None

    def _sync_to_hub_async(
        self,
        ckpt_path: Path,
        epoch: int,
        manifest_bytes: bytes,
        progress_bytes: bytes,
    ) -> None:
        """Queue a serialized snapshot without daemon threads."""
        executor = self._ensure_upload_executor()
        future = executor.submit(
            self._upload_snapshot,
            ckpt_path,
            epoch,
            manifest_bytes,
            progress_bytes,
        )
        self._upload_futures.append(future)
        self._upload_futures = [
            item for item in self._upload_futures if not item.done()
        ]

    def _sync_to_hub_sync(
        self,
        ckpt_path: Path,
        epoch: int,
        manifest_bytes: bytes,
        progress_bytes: bytes,
    ) -> None:
        """Flush previous uploads, then synchronously upload the final state."""
        self.wait_for_uploads()
        self._upload_snapshot(
            ckpt_path, epoch, manifest_bytes, progress_bytes
        )

    def _upload_snapshot(
        self,
        ckpt_path: Path,
        epoch: int,
        manifest_bytes: bytes,
        progress_bytes: bytes,
    ) -> None:
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
                    path_or_fileobj=io.BytesIO(progress_bytes),
                    path_in_repo=self.PROGRESS_NAME,
                    repo_id=self.repo_id,
                    repo_type="model",
                )
                # The manifest is last so it never advertises a checkpoint
                # before that checkpoint and matching progress are on the Hub.
                self.api.upload_file(
                    path_or_fileobj=io.BytesIO(manifest_bytes),
                    path_in_repo=self.MANIFEST_NAME,
                    repo_id=self.repo_id,
                    repo_type="model",
                )
                logger.info("HF Hub sync complete (epoch %d).", epoch)
            except Exception as exc:
                logger.warning(
                    "HF Hub upload failed for epoch %d: %s. Local copy is safe.",
                    epoch, exc
                )

    def _download_hub_file(self, filename: str, destination: Path) -> bool:
        try:
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                token=self.token,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, destination)
            return True
        except Exception as exc:
            logger.warning(
                "[checkpoint] Failed to download %s from HF Hub: %s",
                filename, exc
            )
            return False

    def _pull_manifest_from_hub(self) -> bool:
        return self._download_hub_file(
            self.MANIFEST_NAME, self.ckpt_dir / self.MANIFEST_NAME
        )

    def _pull_progress_from_hub(self) -> bool:
        return self._download_hub_file(
            self.PROGRESS_NAME, self.ckpt_dir / self.PROGRESS_NAME
        )

    def _pull_checkpoint_from_hub(self, epoch: int) -> bool:
        return self._download_hub_file(
            f"checkpoints/checkpoint_epoch_{int(epoch)}.pt",
            self.ckpt_dir / f"checkpoint_epoch_{int(epoch)}.pt",
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def wait_for_uploads(self) -> None:
        """Block until all queued Hub uploads have finished."""
        if not self._upload_futures:
            return
        futures = list(self._upload_futures)
        for future in futures:
            try:
                future.result()
            except Exception as exc:  # pragma: no cover
                logger.warning("[checkpoint] background upload failed: %s", exc)
        self._upload_futures.clear()

    def _purge_all_checkpoints(self) -> None:
        """Delete local checkpoint artifacts, including stale metadata."""
        for pattern in ("checkpoint_epoch_*.pt", "checkpoint_best.pt", "*.pt.tmp"):
            for path in self.ckpt_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    pass
        for name in (self.MANIFEST_NAME, self.PROGRESS_NAME):
            path = self.ckpt_dir / name
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

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
    def _atomic_json_write(path: Path, payload: Dict) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

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
        except Exception:  # pragma: no cover
            logger.warning("RNG state restoration failed; continuing.")


def _epoch_from_name(name: str) -> int:
    try:
        return int(name.replace("checkpoint_epoch_", "").replace(".pt", ""))
    except ValueError:
        return -1


def _to_cpu_state(state_dict) -> Dict:
    return {
        key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
        for key, value in state_dict.items()
    }


def verify_hf_sync(repo_id: str) -> Dict:
    """Verify remote manifest and sequential progress, never raising."""
    status = {
        "repo_id": repo_id,
        "latest_remote_checkpoint": None,
        "epochs_completed": 0,
        "sync_ok": False,
    }
    token = os.environ.get("HF_TOKEN")
    if not token:
        status["error"] = "HF_TOKEN not set"
        return status

    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=token)
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        if CheckpointManager.MANIFEST_NAME not in files:
            status["error"] = "remote manifest.json not found"
            return status
        if CheckpointManager.PROGRESS_NAME not in files:
            status["error"] = "remote progress.json not found"
            return status

        manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename=CheckpointManager.MANIFEST_NAME,
            token=token,
        )
        progress_path = hf_hub_download(
            repo_id=repo_id,
            filename=CheckpointManager.PROGRESS_NAME,
            token=token,
        )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)

        valid, last_seq = CheckpointManager._validate_progress(progress)
        latest_epoch = int(manifest["latest_epoch"])
        checkpoint_name = f"checkpoints/checkpoint_epoch_{latest_epoch}.pt"

        status["latest_remote_checkpoint"] = f"checkpoint_epoch_{latest_epoch}.pt"
        status["epochs_completed"] = latest_epoch + 1
        status["sync_ok"] = (
            valid
            and last_seq == latest_epoch
            and checkpoint_name in files
        )
        if not status["sync_ok"]:
            status["error"] = "remote manifest/progress/checkpoint state is inconsistent"
    except Exception as exc:
        status["error"] = str(exc)
    return status


__all__ = ["CheckpointManager", "verify_hf_sync"]
