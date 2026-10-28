"""
integration_snippet.py — wiring the verified stack into the real 3D-SynTree
trainer (stage 1 supervised + stage 2 RL), plus the exact recovery procedure
for the current epoch-6 NaN failure. Every component referenced here was
verified on the isolated prototype (prototype_test.py, 24/24 tests).

UPLOAD TO COLAB (4 files, no other dependencies):
    train_observatory.py, mol_metrics.py, curriculum.py, nan_shield.py

TOTAL DIFF TO YOUR TRAINER: ~15 lines per loop (shown below). Nothing else in
your pipeline changes — the interference test proved training is bit-identical
with the full stack attached.
"""

# ============================================================================
# PART 1 — STAGE 1 SUPERVISED LOOP (guarded, monitored, curriculum-ready)
# ============================================================================
import torch
from train_observatory import RunMonitor, RunReport, compare_runs
from nan_shield import NanShield, SystemicNanError, verify_checkpoint
import mol_metrics as mm
from curriculum import (grouped_split, difficulty_scores, assign_levels,
                        CurriculumSampler, LevelTracker, scaffold_key, parent_key)


def train_stage1(cfg, model, train_records, val_records, start_epoch=0):
    # ---- 1-time setup (3 lines) ----------------------------------------------
    monitor = RunMonitor(run_name="stage1", config=cfg, run_root="runs",
                         sampling_interval=1.0)          # env snapshot, LR/VRAM/loss trails
    shield = NanShield(monitor=monitor, max_consecutive=5,  # abort after 5 consecutive
                       param_check_every=10)                 # non-finite events, no infinite skips
    monitor.capture_warnings()

    # ---- data hygiene: leakage-safe grouped split FIRST (before any curriculum) --
    # Keep every scaffold / conformer / reaction family in ONE partition:
    #   train_idx, val_idx, test_idx, _ = grouped_split(
    #       all_records, key_fn=scaffold_key, val_frac=0.15, test_frac=0.05, seed=cfg["seed"])
    # For conformer families use parent_key (molecule identity) instead.

    # ---- curriculum (optional, Approach C): difficulty annotation + sampler ----
    #   scores  = difficulty_scores(train_records, feature_of=lambda r: r["smiles"])
    #   levels, _ = assign_levels(scores, n_levels=9)        # 9 quantile-balanced bins
    #   Extend molecular_difficulty_features() with your 3D features: rotatable
    #   bonds, ring systems, conformer counts, steric crowding, torsion targets.
    #   curriculum = CurriculumSampler(train_idx, levels,
    #                                   milestone_metric="val_loss", improve_thresh=0.01,
    #                                   patience=3)          # advances on milestones,
    #                                                         # never fully drops easy levels
    curriculum = None  # start your baseline run with None; A/B the curriculum later

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    # YOUR scheduler (unchanged), e.g. OneCycle / cosine:
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cfg["lr_lambda"])
    # FP16 on the T4 — the shield needs the scaler so the guard sees TRUE grads:
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("fp16", True))

    for epoch in range(start_epoch, cfg["epochs"]):
        try:
            with monitor.epoch(epoch, n_batches=len(cfg["loader"])):
                for i, batch in enumerate(cfg["loader"]):
                    # ... your existing forward pass ...
                    reaction_ce, synthon_ce, torsion_nll = cfg["forward_losses"](model, batch)
                    loss = reaction_ce + synthon_ce + torsion_nll

                    # ---- THE guarded step (replaces zero_grad/backward/step) ----
                    action, gn = shield.guarded_step(
                        loss, model, opt, scaler=scaler, clip=cfg["grad_clip"],
                        scheduler=sched,
                        components={"reaction_ce": reaction_ce.detach(),
                                    "synthon_ce": synthon_ce.detach(),
                                    "torsion_nll": torsion_nll.detach()},
                        epoch=epoch, batch=i)
                    if action == "abort":
                        raise SystemicNanError(f"systemic NaN at ep{epoch} b{i}")
                    if action in ("skipped", "skip_batch"):
                        continue          # this batch never touched the weights
                    monitor.step(epoch=epoch, batch=i, n_batches=len(cfg["loader"]),
                                 loss=loss.detach(), lr=sched.get_last_lr()[0],
                                 grad_norm=gn, batch_size=len(batch),
                                 components={"reaction_ce": float(reaction_ce.detach()),
                                             "synthon_ce": float(synthon_ce.detach()),
                                             "torsion_nll": float(torsion_nll.detach())})
        except SystemicNanError as e:
            monitor.error("systemic NaN — rolling back to last verified checkpoint",
                          exc=e, epoch=epoch)
            if not shield.rollback(model, opt, sched):
                monitor.finish(status="failed")
                raise
            continue                       # replay the epoch from the clean state

        # ---- end of epoch: REAL evaluation metrics (not just loss) -------------
        rxn_logits, rxn_targets = cfg["eval_reaction"](model, val_records)
        ranked_synthon_candidates, target_synthons = cfg["eval_synthon_ranking"](
            model, val_records)              # ranked candidate SMILES per sample
        generated_smiles = cfg["generate"](model, val_records)
        metrics = {}
        metrics.update(mm.reaction_metrics(rxn_logits, rxn_targets, ks=(1, 3, 5, 10)))
        metrics.update(mm.synthon_metrics(ranked_synthon_candidates, target_synthons,
                                          ks=(1, 3, 5)))
        metrics.update(mm.generation_metrics(
            generated_smiles, targets=target_synthons,
            reference_pool=cfg["train_smiles_pool"]))   # validity/uniqueness/novelty/diversity
        vloss = cfg["eval_loss"](model, val_records)
        monitor.validation(epoch, val_loss=vloss, val_acc=metrics["reaction_acc"])
        monitor.evaluation(epoch, metrics)

        # per-difficulty-level metrics (only if curriculum/levels are enabled):
        #   tracker = LevelTracker()
        #   for level, hit, n in cfg["per_level_hits"](model, val_records, levels):
        #       tracker.add_ratio(level, "reaction_acc", hit, n)
        #   for k, v in tracker.summary().items():
        #       m = re.match(r"^L(\d+)/(.+)$", k)
        #       monitor.evaluation(epoch, {m.group(2): v}, level=int(m.group(1)))
        #   monitor.plot_all() regenerates the per-level heatmap automatically.

        # ---- verified checkpoint -> THEN upload to HF --------------------------
        ok, rep = shield.guarded_save(f"ckpt_epoch{epoch}.pt", model, opt, sched,
                                      epoch=epoch)
        if ok:
            cfg["upload_to_hf"](f"ckpt_epoch{epoch}.pt")   # only verified ckpts upload

        if curriculum is not None:
            advanced = curriculum.maybe_advance({"val_loss": vloss})
            monitor.event("curriculum_stage", advanced=advanced, **curriculum.state())

    monitor.finish(status="completed")


# ============================================================================
# PART 2 — STAGE 2 RL (same pattern; keep one comparable benchmark)
# ============================================================================
# - Per PPO step:     monitor.step(epoch=episode, batch=t, loss=policy_loss,
#                                    lr=lr, grad_norm=gn,
#                                    reward=rollout_reward, kl=approx_kl,
#                                    entropy=entropy)      # extras -> JSON column
# - guarded_step works identically (scaler=None for typical FP32 RL runs).
# - After every N episodes, evaluate on the SAME held-out val set with the SAME
#   mol_metrics calls as stage 1 so curves are comparable across stages:
#       monitor.evaluation(episode, {**mm.reaction_metrics(...),
#                                     **mm.synthon_metrics(...)})
# - End of campaign: compare_runs(["runs/stage1_*", "runs/stage2_*"],
#                                 out="runs/stage_comparison.png")

# ============================================================================
# PART 3 — RECOVERY PROCEDURE FOR THE CURRENT EPOCH-6 NaN FAILURE
# ============================================================================
# Your log: batch 44 of epoch 6 produced loss=NaN, then 16 consecutive skips,
# and epoch 7 was NaN from batch 0 with val_loss=NaN. That pattern means the
# NaN reached the weights/optimizer state — batch-skipping alone cannot
# recover (exactly what the shield fixes).
#
# 1) STOP the run. It is burning T4 time and producing no usable updates.
# 2) Download the epoch-5 checkpoint from HF and VERIFY it before trusting it:
#        from nan_shield import verify_checkpoint
#        ok, rep = verify_checkpoint("epoch5.pt")      # checks model+opt+sched
#        print(ok, rep)   # ok=False -> fall back to epoch 4 / epoch 3 ...
# 3) Resume with the guarded loop above. The two changes that fix the cascade:
#       a. scaler.unscale_(opt) before clipping + a grad/param finiteness gate
#          BEFORE optimizer.step() — a NaN-grad batch is skipped WITHOUT the
#          weights ever being touched (this is what GradScaler's inf check
#          intends, now done explicitly);
#       b. max_consecutive=5: five consecutive non-finite events abort the
#          epoch and roll back to the last VERIFIED checkpoint instead of
#          skipping forever with a poisoned state.
# 4) Keep FP16, but WIRED THROUGH guarded_step(scaler=scaler). If alerts still
#    fire around the same batch, run a short FP32 diagnostic (fp16=False) over
#    that batch region — the per-component curves (plots/component_losses.png)
#    will show which of reaction CE / synthon CE / torsion NLL went non-finite
#    first. Prime suspects for torsion: log(0) from an all-zero angle bin,
#    division by a zero normalization term, or the earlier FP16 mask value.
# 5) When an abort happens, read nan_forensics/epX_bY.json: it contains the
#    failing component, its recent value history, param health and CUDA state
#    — enough to pin the bug offline instead of guessing from the tail of a log.
#
# Do NOT just lower the learning rate: if the root cause is an invalid torsion
# computation, LR only delays the same failure and destroys your schedule.
