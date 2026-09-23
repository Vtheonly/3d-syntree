"""Resilient, wall-clock-budgeted training loop.

Features:
* 12-hour (configurable) time budget with graceful checkpoint-and-exit,
* gradient accumulation and mixed precision,
* cosine LR schedule with linear warmup,
* periodic checkpointing with automatic Hugging Face Hub sync,
* deterministic seeding for reproducibility,
* structured JSON metric logging.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.crossdocked import CrossDockedDataset
from syntree.models.torsion_head import ContinuousTorsionHead
from syntree.utils.checkpoint import CheckpointManager
from syntree.utils.hardware import autotune_batch_size, free_vram_bytes
from syntree.utils.logger import StructuredLogger

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed every RNG we rely on for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:  # pragma: no cover
        pass


class ResilientTrainer:
    """Time-budgeted trainer with checkpoint/resume support.

    Args:
        model: the :class:`~syntree.models.policy.SynTreePolicy`.
        config: full configuration dict (see ``configs/default_config.json``).
        device: torch device to train on.
        auto_resume: restore the latest checkpoint before training.
        output_dir: where metrics and experiment artifacts are written.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: torch.device,
        auto_resume: bool = True,
        output_dir: str = "./experiments",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = str(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        self.time_budget_sec = float(train_cfg.get("time_budget_hours", 11.5)) * 3600.0
        self.max_epochs = int(train_cfg.get("max_epochs", 40))
        self.accum_steps = max(1, int(data_cfg.get("accumulate_grad_batches", 1)))
        self.grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
        self.warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
        self.eval_interval = int(train_cfg.get("eval_interval_epochs", 1))
        self.loss_weights = dict(train_cfg.get("loss_weights", {}))
        self.keep_last_n = int(train_cfg.get("keep_last_n_checkpoints", 3))

        self.start_time = time.time()

        # Determinism.
        seed = int(config.get("system", {}).get("seed", 42))
        seed_everything(seed)
        self.struct_logger = StructuredLogger(os.path.join(self.output_dir, "metrics.jsonl"))

        # Data.
        self.catalog = SynthonCatalog(
            config["data"]["synthon_catalog_path"],
            embedding_dim=config["model"].get("synthon_embedding_dim", 128),
            min_fsp3=float(config.get("catalog", {}).get("min_fsp3", 0.42)),
            max_mw=float(config.get("catalog", {}).get("max_mw", 220.0)),
        )
        val_fraction = float(data_cfg.get("val_fraction", 0.1))
        self.dataset = CrossDockedDataset(
            data_cfg["data_dir"],
            split="train",
            catalog=self.catalog,
            num_synthetic=int(data_cfg.get("synthetic_samples", 100)),
        )
        self.val_dataset = CrossDockedDataset(
            data_cfg["data_dir"],
            split="val",
            catalog=self.catalog,
            num_synthetic=max(8, int(self.dataset.num_synthetic * val_fraction)),
        )

        # Mixed-precision flags (needed by the auto-scale probe below).
        self.use_amp = bool(config.get("system", {}).get("mixed_precision") in ("fp16", "bf16")) and \
            self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion = torch.nn.CrossEntropyLoss()

        # GPU auto-scaling: grow the batch until the card is ~85% full.
        self.batch_size = max(1, int(data_cfg.get("batch_size", 16)))
        self._autoscale_batch(train_cfg)

        self.loader = self._make_loader(self.dataset, shuffle=True)
        self.val_loader = self._make_loader(self.val_dataset, shuffle=False)

        # Optimization.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )
        self.total_steps = max(1, len(self.loader) // self.accum_steps * self.max_epochs)
        self.scheduler = self._make_scheduler()

        # Checkpointing.
        self.ckpt_manager = CheckpointManager(
            config, keep_last_n=self.keep_last_n, ckpt_dir=os.path.join(self.output_dir, "checkpoints")
        )

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        if auto_resume:
            self.start_epoch, self.global_step, self.best_val_loss = (
                self.ckpt_manager.restore_latest(
                    self.model, self.optimizer, self.scheduler
                )
            )
            # Advance the LR schedule to the resumed position.
            for _ in range(self.global_step):
                self.scheduler.step()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collate(items):
        """PyG collate; follow_batch guarantees pocket_batch exists."""
        from torch_geometric.data import Batch

        return Batch.from_data_list(list(items), follow_batch=["pocket_pos"])

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        """Plain DataLoader over the PyG dataset (collate via Batch.from_data_list)."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=int(self.config.get("system", {}).get("num_workers", 0)),
            collate_fn=self._collate,
            drop_last=False,
        )

    # ------------------------------------------------------------------
    # GPU auto-scaling
    # ------------------------------------------------------------------
    def _autoscale_batch(self, train_cfg: dict) -> None:
        """Grow ``batch_size`` until the GPU is filled to the target fraction.

        Probing runs real forward+backward passes on real batches and reads
        ``torch.cuda.max_memory_reserved`` after each one, so the chosen batch
        reflects the actual memory footprint (activations included), not a
        heuristic. The search is RNG-transparent: torch RNG state is snapshotted
        and restored, weights are never updated (no optimizer step), and
        gradients are zeroed between probes. Gradient accumulation is then
        rebalanced so ``batch x accum`` stays close to the configured
        effective batch. On CPU (or when disabled) this is a no-op.
        """
        auto = dict(train_cfg.get("auto_scale", {}))
        if not auto.get("enabled", False) or self.device.type != "cuda":
            return

        target_fraction = float(auto.get("target_vram_fraction", 0.85))
        max_batch = int(auto.get("max_batch_size", 8192))
        free_bytes = free_vram_bytes(self.device)
        if free_bytes <= 0:  # pragma: no cover - defensive
            return
        target_bytes = int(free_bytes * target_fraction)

        # --- RNG transparency -------------------------------------------------
        rng_state = torch.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

        def probe(batch_size: int) -> None:
            loader = DataLoader(
                self.dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, collate_fn=self._collate,
            )
            batch = next(iter(loader)).to(self.device)
            target = batch.target_synthon.clamp(max=len(self.catalog) - 1)
            rxn_mask = torch.zeros(
                batch.num_graphs, len(self.catalog),
                device=self.device, dtype=torch.float32,
            )
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                preds = self.model(
                    batch, self.catalog.embeddings.to(self.device), rxn_mask
                )
                loss = self.criterion(preds["synthon_logits"], target) + 0.5 * (
                    ContinuousTorsionHead.loss_fn(
                        preds["torsion_mu"], preds["torsion_kappa"],
                        batch.target_dihedral,
                    )
                )
            loss.backward()
            self.model.zero_grad(set_to_none=True)

        try:
            result = autotune_batch_size(
                probe,
                dataset_size=len(self.dataset),
                start_batch=self.batch_size,
                max_batch=max_batch,
                target_bytes=target_bytes,
                device=self.device,
            )
        except Exception as exc:  # pragma: no cover - keep training alive
            logger.warning("auto-scale probe failed (%s); keeping batch=%d",
                           exc, self.batch_size)
            result = {"batch_size": float(self.batch_size), "peak_bytes": 0.0}
        finally:
            # Restore RNG so probing never perturbs training determinism.
            torch.set_rng_state(rng_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            self.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        tuned = max(1, int(result["batch_size"]))
        peak_gb = float(result["peak_bytes"]) / 1024 ** 3
        free_gb = free_bytes / 1024 ** 3
        target_gb = target_bytes / 1024 ** 3

        if tuned <= self.batch_size and peak_gb < 0.5:  # probing produced nothing useful
            print(f"[trainer] auto-scale: disabled (probe inconclusive); "
                  f"batch={self.batch_size}")
            return

        # Rebalance accumulation to keep the effective batch roughly constant.
        effective = self.batch_size * self.accum_steps
        new_accum = max(1, int(round(effective / tuned)))
        old_batch, old_accum = self.batch_size, self.accum_steps
        self.batch_size, self.accum_steps = tuned, new_accum

        print(
            f"[trainer] auto-scale: free VRAM {free_gb:.2f} GB | "
            f"target {target_gb:.2f} GB"
        )
        print(
            f"[trainer] auto-scale: batch {old_batch} -> {tuned} | "
            f"accum {old_accum} -> {new_accum} | "
            f"probe peak {peak_gb:.2f} GB ({100.0 * peak_gb / max(free_gb, 1e-9):.0f}% of free)"
        )
        hard_cap = min(max_batch, len(self.dataset))
        if tuned >= hard_cap and peak_gb < 0.95 * target_gb:
            print(
                f"[trainer] auto-scale note: reached only {peak_gb:.1f} GB because "
                f"the batch is capped by the dataset size ({len(self.dataset)}). "
                "Increase data.synthetic_samples (or use the real CrossDocked "
                "data) for higher GPU utilization."
            )

    def _make_scheduler(self):
        total = max(1, self.total_steps)
        warmup = max(1, self.warmup_epochs * max(1, len(self.loader) // self.accum_steps))
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, float]:
        m = self.config.get("model", {})
        print(
            f"[trainer] starting run: budget={self.time_budget_sec / 3600:.2f}h, "
            f"epochs={self.max_epochs}, synthons={len(self.catalog)}, "
            f"train={len(self.dataset)}, val={len(self.val_dataset)}, "
            f"resume_epoch={self.start_epoch}"
        )
        print(
            f"[trainer] model: hidden_dim={m.get('hidden_dim', 128)}, "
            f"layers={m.get('num_equivariant_layers', 4)}, "
            f"heads={m.get('num_attention_heads', 4)} | "
            f"batch={self.batch_size} x accum={self.accum_steps} | "
            f"steps/epoch={len(self.loader)} | amp={self.use_amp}"
        )

        history = []
        for epoch in range(self.start_epoch, self.max_epochs):
            self.model.train()
            epoch_loss, epoch_synthon, epoch_torsion = 0.0, 0.0, 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(self.loader):
                elapsed = time.time() - self.start_time
                if elapsed >= self.time_budget_sec:
                    print(
                        f"[trainer] time budget expired after {elapsed / 3600:.2f}h; "
                        "checkpointing and exiting."
                    )
                    # Persist the last COMPLETED epoch (epoch-level resume
                    # granularity).
                    last_done = history[-1]["epoch"] if history else max(0, epoch - 1)
                    self._finalize(last_done, history)
                    return self._summary(history, floor=self.start_epoch)

                batch = batch.to(self.device)
                # Clamp targets into the catalog range (synthetic data may
                # have been built against a different catalog size).
                target = batch.target_synthon.clamp(max=len(self.catalog) - 1)

                rxn_mask = torch.zeros(
                    batch.num_graphs, len(self.catalog),
                    device=self.device, dtype=torch.float32,
                )

                with torch.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    preds = self.model(
                        batch, self.catalog.embeddings.to(self.device), rxn_mask
                    )
                    l_synthon = self.criterion(preds["synthon_logits"], target)
                    l_torsion = ContinuousTorsionHead.loss_fn(
                        preds["torsion_mu"], preds["torsion_kappa"], batch.target_dihedral
                    )
                    w_synthon = float(self.loss_weights.get("synthon_ce", 1.0))
                    w_torsion = float(self.loss_weights.get("torsion_nll", 0.5))
                    loss = (w_synthon * l_synthon + w_torsion * l_torsion) / self.accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == len(self.loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1

                epoch_loss += float(loss.item()) * self.accum_steps
                epoch_synthon += float(l_synthon.item())
                epoch_torsion += float(l_torsion.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            entry = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_synthon_ce": epoch_synthon / max(1, n_batches),
                "train_torsion_nll": epoch_torsion / max(1, n_batches),
                "lr": self.optimizer.param_groups[0]["lr"],
                "elapsed_hours": (time.time() - self.start_time) / 3600.0,
            }

            if (epoch + 1) % self.eval_interval == 0:
                entry.update(self.validate())

            history.append(entry)
            self.struct_logger.log({**entry, "phase": "train"})
            print(
                f"[trainer] epoch {epoch:03d} | loss {avg_loss:.4f} | "
                f"synthon_ce {entry['train_synthon_ce']:.4f} | "
                f"torsion_nll {entry['train_torsion_nll']:.4f} | "
                f"val_loss {entry.get('val_loss', float('nan')):.4f}"
            )

            is_best = entry.get("val_loss", float("inf")) < self.best_val_loss
            if is_best:
                self.best_val_loss = entry.get("val_loss", float("inf"))
            self.ckpt_manager.save_checkpoint(
                epoch=epoch,
                step=self.global_step,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=entry,
                is_best=is_best,
                model_config=self.config.get("model"),
            )
            self._write_latest_metrics(entry)

            if time.time() - self.start_time >= self.time_budget_sec:
                print("[trainer] budget reached at epoch boundary; exiting.")
                break

        last_epoch = history[-1]["epoch"] if history else self.start_epoch
        self._finalize(last_epoch, history)
        return self._summary(history, floor=self.start_epoch)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total, synthon, torsion, correct, n = 0.0, 0.0, 0.0, 0, 0
        for batch in self.val_loader:
            batch = batch.to(self.device)
            target = batch.target_synthon.clamp(max=len(self.catalog) - 1)
            rxn_mask = torch.zeros(
                batch.num_graphs, len(self.catalog),
                device=self.device, dtype=torch.float32,
            )
            preds = self.model(batch, self.catalog.embeddings.to(self.device), rxn_mask)
            l_synthon = self.criterion(preds["synthon_logits"], target)
            l_torsion = ContinuousTorsionHead.loss_fn(
                preds["torsion_mu"], preds["torsion_kappa"], batch.target_dihedral
            )
            total += float((l_synthon + 0.5 * l_torsion).item())
            synthon += float(l_synthon.item())
            torsion += float(l_torsion.item())
            correct += int((preds["synthon_logits"].argmax(-1) == target).sum().item())
            n += int(target.numel())
        self.model.train()
        return {
            "val_loss": total / max(1, len(self.val_loader)),
            "val_synthon_ce": synthon / max(1, len(self.val_loader)),
            "val_torsion_nll": torsion / max(1, len(self.val_loader)),
            "val_synthon_acc": correct / max(1, n),
        }

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    def _write_latest_metrics(self, entry: Dict[str, float]) -> None:
        path = os.path.join(self.output_dir, "latest_metrics.json")
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

    def _finalize(self, epoch: int, history) -> None:
        self.ckpt_manager.save_checkpoint(
            epoch=epoch,
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics=history[-1] if history else {},
            is_best=True,
            final=True,
            model_config=self.config.get("model"),
        )
        with open(os.path.join(self.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    @staticmethod
    def _summary(history, floor: int = 0) -> Dict[str, float]:
        if not history:
            # Nothing completed this session; report the resumed floor.
            return {"epochs_completed": floor, "final_train_loss": None,
                    "best_val_loss": None}
        return {
            "epochs_completed": max(floor, history[-1]["epoch"] + 1),
            "final_train_loss": history[-1]["train_loss"],
            "best_val_loss": min(
                (h.get("val_loss", float("inf")) for h in history), default=None
            ),
        }


__all__ = ["ResilientTrainer", "seed_everything"]
