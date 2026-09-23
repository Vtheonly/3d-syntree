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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from syntree.chemistry.catalog import SynthonCatalog
from syntree.chemistry.reactions import HANDLE_NAMES, REACTION_FAMILY_NAMES
from syntree.data.crossdocked import CrossDockedDataset
from syntree.data.hf_loader import (
    ShardedHuggingFaceDataset,
    ShardAwareShuffleSampler,
)
from syntree.data.trajectory import TrajectoryDataset
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
        self.reaction_loss_weight = float(self.loss_weights.get("reaction_ce", 0.5))
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
        trajectory_path = data_cfg.get("trajectory_dataset_path")
        self.data_backend = str(data_cfg.get("backend", "local")).lower()

        if self.data_backend == "huggingface":
            hf_cfg = dict(data_cfg.get("huggingface", {}))
            repo_id = str(hf_cfg.get("repo_id", "")).strip()
            if not repo_id:
                raise ValueError(
                    "data.huggingface.repo_id is required when data.backend='huggingface'"
                )
            token = os.environ.get("HF_TOKEN") or hf_cfg.get("token")
            self.dataset = ShardedHuggingFaceDataset(
                repo_id=repo_id,
                split="train",
                cache_dir=str(hf_cfg.get("cache_dir", "./hf_cache")),
                revision=str(hf_cfg.get("revision", "main")),
                token=token,
                max_cached_shards=int(hf_cfg.get("max_cached_shards", 2)),
            )
            self.val_dataset = ShardedHuggingFaceDataset(
                repo_id=repo_id,
                split="val",
                cache_dir=str(hf_cfg.get("cache_dir", "./hf_cache")),
                revision=str(hf_cfg.get("revision", "main")),
                token=token,
                max_cached_shards=int(hf_cfg.get("max_cached_shards", 2)),
            )
        elif trajectory_path:
            self.data_backend = "trajectory_pt"
            self.dataset = TrajectoryDataset(trajectory_path, split="train")
            self.val_dataset = TrajectoryDataset(trajectory_path, split="val")
        else:
            self.dataset = CrossDockedDataset(
                data_cfg["data_dir"],
                split="train",
                catalog=self.catalog,
                num_synthetic=int(data_cfg.get("synthetic_samples", 100)),
                synthetic_fallback=bool(data_cfg.get("synthetic_fallback", False)),
                split_manifest_path=data_cfg.get("split_manifest_path"),
            )
            self.val_dataset = CrossDockedDataset(
                data_cfg["data_dir"],
                split="val",
                catalog=self.catalog,
                num_synthetic=max(8, int(self.dataset.num_synthetic * val_fraction)),
                synthetic_fallback=bool(data_cfg.get("synthetic_fallback", False)),
                split_manifest_path=data_cfg.get("split_manifest_path"),
            )

        # Mixed-precision flags.
        self.use_amp = bool(config.get("system", {}).get("mixed_precision") in ("fp16", "bf16")) and \
            self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion = torch.nn.CrossEntropyLoss()

        # GPU auto-scaling.
        self.batch_size = max(1, int(data_cfg.get("batch_size", 16)))
        self.lr_scale_factor = 1.0
        self._autoscale_batch(train_cfg)

        self.loader = self._make_loader(self.dataset, shuffle=True)
        self.val_loader = self._make_loader(self.val_dataset, shuffle=False)

        # Optimization.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 3e-4)) * self.lr_scale_factor,
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
            for _ in range(self.global_step):
                self.scheduler.step()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collate(items):
        """Robust PyG batch collator that guards against 0-dim tensor errors."""
        from torch_geometric.data import Batch

        clean_items = []
        for item in items:
            data = item.clone()
            # Explicitly set num_nodes on each Data object to prevent PyG from
            # guessing and calling .size(cat_dim) on 0-dim scalar tensors.
            if hasattr(data, "pocket_pos") and data.pocket_pos is not None:
                data.num_nodes = data.pocket_pos.size(0)
            elif hasattr(data, "pos") and data.pos is not None:
                data.num_nodes = data.pos.size(0)

            # Convert any 0-dim scalar tensor to 1-dim so collate can concatenate cleanly.
            for k in list(data.keys()):
                v = data[k]
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    data[k] = v.unsqueeze(0)

            clean_items.append(data)

        batch = Batch.from_data_list(clean_items, follow_batch=["pocket_pos", "ligand_pos"])
        if not hasattr(batch, "pocket_batch") or batch.pocket_batch is None:
            batch.pocket_batch = getattr(batch, "pocket_pos_batch", None)
        if not hasattr(batch, "ligand_batch") or batch.ligand_batch is None:
            batch.ligand_batch = getattr(batch, "ligand_pos_batch", None)
        return batch

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        """DataLoader over PyG dataset with shard-aware sampling."""
        workers = int(self.config.get("system", {}).get("num_workers", 0))
        sampler = None
        if shuffle and isinstance(dataset, ShardedHuggingFaceDataset):
            sampler = ShardAwareShuffleSampler(
                dataset,
                seed=int(self.config.get("system", {}).get("seed", 42)),
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=workers,
            collate_fn=self._collate,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )

    # ------------------------------------------------------------------
    # GPU auto-scaling
    # ------------------------------------------------------------------
    def _autoscale_batch(self, train_cfg: dict) -> None:
        """Grow batch size until target VRAM fraction is reached."""
        auto = dict(train_cfg.get("auto_scale", {}))
        if not auto.get("enabled", False) or self.device.type != "cuda":
            return

        target_fraction = float(auto.get("target_vram_fraction", 0.85))
        max_batch = int(auto.get("max_batch_size", 8192))
        free_bytes = free_vram_bytes(self.device)
        if free_bytes <= 0:
            return
        target_bytes = int(free_bytes * target_fraction)

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
            target = batch.target_synthon.clamp(min=0, max=len(self.catalog) - 1)
            stop_index = len(self.catalog)
            target_stop = getattr(
                batch, "target_stop", torch.zeros_like(target, dtype=torch.bool)
            )
            target_action = torch.where(
                target_stop.bool(),
                torch.full_like(target, stop_index),
                target,
            )
            synthon_mask, reaction_mask = self._build_training_masks(batch)
            with torch.autocast(device_type="cuda", enabled=self.use_amp):
                preds = self.model(
                    batch,
                    self.catalog.embeddings.to(self.device),
                    synthon_mask,
                    reaction_mask,
                )
                loss = (
                    self.criterion(preds["reaction_logits"], batch.target_reaction_family_idx)
                    + self.criterion(preds["synthon_logits"], target_action)
                    + 0.5 * ContinuousTorsionHead.loss_fn(
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
                start_batch=min(self.batch_size, max(1, len(self.dataset))),
                max_batch=max_batch,
                target_bytes=target_bytes,
                device=self.device,
            )
        except Exception as exc:
            logger.warning("auto-scale probe failed (%s); keeping batch=%d", exc, self.batch_size)
            result = {"batch_size": float(self.batch_size), "peak_bytes": 0.0}
        finally:
            torch.set_rng_state(rng_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            self.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        tuned = max(1, int(result["batch_size"]))
        peak_gb = float(result["peak_bytes"]) / 1024 ** 3
        free_gb = free_bytes / 1024 ** 3
        target_gb = target_bytes / 1024 ** 3

        if tuned <= self.batch_size and peak_gb < 0.5:
            print(f"[trainer] auto-scale: disabled (probe inconclusive); batch={self.batch_size}")
            return

        effective = self.batch_size * self.accum_steps
        new_accum = max(1, int(round(effective / tuned)))
        old_batch, old_accum = self.batch_size, self.accum_steps
        old_effective = max(1, old_batch * old_accum)
        self.batch_size, self.accum_steps = tuned, new_accum
        new_effective = max(1, self.batch_size * self.accum_steps)
        auto_lr_rule = str(auto.get("lr_scale_rule", "linear")).lower()
        if auto_lr_rule == "linear":
            self.lr_scale_factor = new_effective / old_effective
        elif auto_lr_rule == "sqrt":
            self.lr_scale_factor = math.sqrt(new_effective / old_effective)
        else:
            self.lr_scale_factor = 1.0

        print(
            f"[trainer] auto-scale: free VRAM {free_gb:.2f} GB | target {target_gb:.2f} GB\n"
            f"[trainer] auto-scale: batch {old_batch} -> {tuned} | accum {old_accum} -> {new_accum} | "
            f"effective {old_effective} -> {new_effective} | lr scale {self.lr_scale_factor:.3f} | "
            f"probe peak {peak_gb:.2f} GB ({100.0 * peak_gb / max(free_gb, 1e-9):.0f}% of free)"
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
            if hasattr(getattr(self.loader, "sampler", None), "set_epoch"):
                self.loader.sampler.set_epoch(epoch)
            self.model.train()
            epoch_loss, epoch_reaction, epoch_synthon, epoch_torsion = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(self.loader):
                elapsed = time.time() - self.start_time
                if elapsed >= self.time_budget_sec:
                    print(
                        f"[trainer] time budget expired after {elapsed / 3600:.2f}h; "
                        "checkpointing and exiting."
                    )
                    last_done = history[-1]["epoch"] if history else max(0, epoch - 1)
                    self._finalize(last_done, history)
                    return self._summary(history, floor=self.start_epoch)

                batch = batch.to(self.device)
                target = batch.target_synthon.clamp(min=0, max=len(self.catalog) - 1)
                stop_index = len(self.catalog)
                target_stop = getattr(
                    batch, "target_stop", torch.zeros_like(target, dtype=torch.bool)
                )
                target_action = torch.where(
                    batch.target_stop.bool(),
                    torch.full_like(target, stop_index),
                    target,
                )
                synthon_mask, reaction_mask = self._build_training_masks(batch)

                torsion_synthon_emb = self.catalog.embeddings.to(self.device)[target]

                with torch.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    preds = self.model(
                        batch,
                        self.catalog.embeddings.to(self.device),
                        synthon_mask,
                        reaction_mask,
                        synthon_embedding_input=torsion_synthon_emb,
                    )
                    reaction_per_sample = F.cross_entropy(
                        preds["reaction_logits"],
                        batch.target_reaction_family_idx,
                        reduction="none",
                    )
                    non_stop = ~target_stop.bool()
                    l_reaction = (
                        reaction_per_sample[non_stop].mean()
                        if bool(non_stop.any().item())
                        else reaction_per_sample.new_zeros(())
                    )
                    l_synthon = self.criterion(preds["synthon_logits"], target_action)

                    torsion_loss_all = ContinuousTorsionHead.loss_fn(
                        preds["torsion_mu"], preds["torsion_kappa"], batch.target_dihedral
                    )
                    if bool(non_stop.any().item()):
                        torsion_log_probs = ContinuousTorsionHead.log_prob(
                            preds["torsion_mu"][non_stop],
                            preds["torsion_kappa"][non_stop],
                            batch.target_dihedral[non_stop],
                        )
                        l_torsion = -torsion_log_probs.mean()
                    else:
                        l_torsion = torsion_loss_all.new_zeros(())

                    w_synthon = float(self.loss_weights.get("synthon_ce", 1.0))
                    w_torsion = float(self.loss_weights.get("torsion_nll", 0.5))
                    loss = (
                        w_synthon * l_synthon
                        + self.reaction_loss_weight * l_reaction
                        + w_torsion * l_torsion
                    ) / self.accum_steps

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
                epoch_reaction += float(l_reaction.item())
                epoch_synthon += float(l_synthon.item())
                epoch_torsion += float(l_torsion.item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            entry = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_reaction_ce": epoch_reaction / max(1, n_batches),
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
                f"reaction_ce {entry['train_reaction_ce']:.4f} | "
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
    def _build_training_masks(self, batch):
        """Build reaction-family and synthon masks from validated targets."""
        reaction_masks = []
        synthon_masks = []
        for family_idx, handle_idx in zip(
            batch.target_reaction_family_idx.tolist(),
            batch.target_core_handle_idx.tolist(),
        ):
            family = REACTION_FAMILY_NAMES[int(family_idx)]
            handle = HANDLE_NAMES[int(handle_idx)]
            reaction_masks.append(
                self.catalog.get_reaction_family_compatibility_mask(
                    device=self.device, core_handle=handle
                )
            )
            synthon_masks.append(
                self.catalog.get_reaction_family_mask(
                    family, device=self.device, core_handle=handle
                )
            )
        return torch.stack(synthon_masks), torch.stack(reaction_masks)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        if len(self.val_dataset) == 0:
            self.model.train()
            return {
                "val_loss": float("inf"),
                "val_reaction_ce": float("nan"),
                "val_synthon_ce": float("nan"),
                "val_torsion_nll": float("nan"),
                "val_reaction_acc": float("nan"),
                "val_synthon_acc": float("nan"),
                "val_synthon_acc_oracle_family": float("nan"),
                "val_joint_action_acc": float("nan"),
            }

        total, reaction, synthon, torsion = 0.0, 0.0, 0.0, 0.0
        reaction_correct, synthon_correct = 0, 0
        oracle_synthon_correct_total, joint_action_correct, n = 0, 0, 0
        reaction_n = 0
        for batch in self.val_loader:
            batch = batch.to(self.device)
            target = batch.target_synthon.clamp(min=0, max=len(self.catalog) - 1)
            stop_index = len(self.catalog)
            target_stop = getattr(
                batch, "target_stop", torch.zeros_like(target, dtype=torch.bool)
            )
            target_action = torch.where(
                target_stop.bool(),
                torch.full_like(target, stop_index),
                target,
            )
            synthon_mask, reaction_mask = self._build_training_masks(batch)
            torsion_synthon_emb = self.catalog.embeddings.to(self.device)[target]

            preds = self.model(
                batch,
                self.catalog.embeddings.to(self.device),
                synthon_mask,
                reaction_mask,
                synthon_embedding_input=torsion_synthon_emb,
            )
            l_reaction = self.criterion(
                preds["reaction_logits"], batch.target_reaction_family_idx
            )
            l_synthon = self.criterion(preds["synthon_logits"], target_action)
            non_stop = ~target_stop.bool()
            if bool(non_stop.any().item()):
                l_torsion = -ContinuousTorsionHead.log_prob(
                    preds["torsion_mu"][non_stop],
                    preds["torsion_kappa"][non_stop],
                    batch.target_dihedral[non_stop],
                ).mean()
            else:
                l_torsion = preds["torsion_mu"].new_zeros(())
            total += float(
                (
                    self.reaction_loss_weight * l_reaction
                    + l_synthon
                    + 0.5 * l_torsion
                ).item()
            )
            reaction += float(l_reaction.item())
            synthon += float(l_synthon.item())
            torsion += float(l_torsion.item())

            predicted_family = preds["reaction_logits"].argmax(-1)
            reaction_correct += int(
                ((predicted_family == batch.target_reaction_family_idx) & non_stop).sum().item()
            )
            reaction_n += int(non_stop.sum().item())
            oracle_synthon = preds["synthon_logits"].argmax(-1)
            synthon_oracle_correct = (oracle_synthon == target_action)

            predicted_masks = []
            for family_idx, handle_idx in zip(
                predicted_family.tolist(),
                batch.target_core_handle_idx.tolist(),
            ):
                family = REACTION_FAMILY_NAMES[int(family_idx)]
                handle = HANDLE_NAMES[int(handle_idx)]
                predicted_masks.append(
                    self.catalog.get_reaction_family_mask(
                        family,
                        device=self.device,
                        core_handle=handle,
                    )
                )
            predicted_masks = torch.stack(predicted_masks)
            predicted_synthon_logits, _ = self.model.synthon_head(
                preds["pocket_context"],
                self.catalog.embeddings.to(self.device),
                predicted_masks,
            )
            joint_synthon = predicted_synthon_logits.argmax(-1)
            joint_synthon_correct = joint_synthon == target_action
            synthon_correct += int(joint_synthon_correct.sum().item())
            oracle_synthon_correct_total += int(synthon_oracle_correct.sum().item())
            joint_action_correct += int(
                (
                    (predicted_family == batch.target_reaction_family_idx)
                    & joint_synthon_correct
                ).sum().item()
            )
            n += int(target.numel())

        self.model.train()
        return {
            "val_loss": total / max(1, len(self.val_loader)),
            "val_reaction_ce": reaction / max(1, len(self.val_loader)),
            "val_synthon_ce": synthon / max(1, len(self.val_loader)),
            "val_torsion_nll": torsion / max(1, len(self.val_loader)),
            "val_reaction_acc": reaction_correct / max(1, reaction_n),
            "val_synthon_acc": synthon_correct / max(1, n),
            "val_synthon_acc_oracle_family": oracle_synthon_correct_total / max(1, n),
            "val_joint_action_acc": joint_action_correct / max(1, n),
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