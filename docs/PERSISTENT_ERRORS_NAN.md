# PERSISTENT ERRORS — Stage-1 NaN losses (UNRESOLVED, deliberately NOT fixed)

**Status as of 2026-10-29:** the error documented here still has no confirmed
root cause. Per the operator's explicit instruction, the 2026-10-29 iteration
did **not** attempt to solve, patch, or work around it. This document is the
complete evidence record and investigation plan for the next agent.

**Do not** treat the fresh restart (docs/HANDOFF_2026-10-29.md) as a fix: the
root cause is unidentified, so **the failure may recur in the fresh run**.
The restart only guarantees the new run does not INHERIT poisoned state.

---

## 1. Symptom (what is observed)

The Stage-1 trainer (Colab Tesla T4, fp16 + GradScaler) hits non-finite
training loss and then skips every batch indefinitely:

```
[trainer] WARNING: non-finite loss at batch <b> (loss=nan); skipping batch. (total skips: <n>)
```

Two distinct incidents are documented (full evidence in `docs/evidence/`):

| | Incident A | Incident B |
|---|---|---|
| Date | 2026-10-28 (earlier fresh run) | 2026-10-28 16:11 UTC (resumed run) |
| Onset | epoch 6, batch 44; 16 consecutive skips to the epoch end | batch 0 of resumed epoch 16 — immediately after `Restored verified checkpoint: epoch 15 (step 41)` |
| Spread | epoch 7 NaN from batch 0; cumulative skips 45 | every observed batch 0–47 NaN (47+ consecutive skips) |
| Validation | epoch-6 val loss NaN | (run aborted before validation) |
| Recovery | none — the skip loop never recovers on its own | none |

Confirmed facts (directly evidenced by logs):

1. Once the first NaN appears, the finite-loss guard skips every subsequent
   batch **of every following epoch** — the trainer never recovers.
2. A checkpoint saved after the onset is structurally valid (manifest/
   progress/epoch consistent) but numerically unusable: restoring it yields
   loss=NaN on batch 0 with no optimizer step in between.
3. The skip mechanism does NOT crash, does NOT repair, and (because
   `epoch_loss` aggregation only counts finite batches) reports nothing but
   the growing skip counter — the run keeps burning GPU time and uploading
   checkpoints indefinitely.
4. Before the onset (epochs 1–5), the run was healthy: train_loss 7.2804 →
   7.2794, val_loss plateau 7.4951, 0 OOM, 0 NaN skips, LR 3e-4.
5. The training-side numerics were ALREADY hardened before incident B
   (fp16-safe mask fill −1e9→−1e4, target unmasking, finite-loss guard,
   non-finite sample rejection in the collator, ±6e4 activation clamps,
   worst-case probe + pocket cap — commits 0bf156f and 95abd96) and the
   NaN still occurred. **The cause is not the original mask-overflow bug.**

## 2. Hypotheses (ranked, with the evidence for/against)

Nothing below is confirmed. They are ordered by current plausibility.

- **H1 — Poisoned optimizer/parameter state in the checkpoint (strongest
  for Incident B).** Batch 0 of the resumed run NaNs before any step; the
  only state that changes loss-before-step is restored model weights,
  restored GradScaler scale, or restored Adam moments. FOR: deterministic
  NaN from batch 0 across two independent processes. AGAINST: does not
  explain the ORIGINAL onset at epoch 6 / batch 44 (weights were finite
  during epochs 1–5); the poisoning mechanism between batch 43 and 44 of
  epoch 6 is unexplained (see H2–H5).
- **H2 — Finite-loss guard is not a gradient guard.** The skip happens
  after the forward pass but the code path `scaler.scale(loss).backward()`
  from a PREVIOUS finite batch could still deposit non-finite gradients
  into `.grad` without the scaler detecting inf (e.g. NaN produced by
  0*inf in fp16 backward, or a finite-but-huge update that overflows a
  weight to inf on the next step). GradScaler skips the step when found_inf
  fires, but finite loss + non-finite grads that survive clipping bounds
  only update the norm, and a weight that reaches ±inf turns every later
  loss into NaN. FOR: matches "sudden, permanent, all-batch" failure.
  AGAINST: unproven; needs the gradient/param health telemetry that was
  not running.
- **H3 — Torsion head numerics (von Mises / log Bessel) under fp16.**
  torsion NLL was flat for five epochs and involves `kappa`, `log_prob`,
  `log(bessel_i0)` — classic NaN producers at 0/∞ kappa. The training code
  runs torsion math inside autocast. FOR: the component was frozen (no
  learning progress) right before failure. AGAINST: no component-level
  loss was logged at onset, so this is inference, not evidence.
- **H4 — A specific data batch (epoch 6, batch 44) carries an input that
  survives the collator's non-finite check (e.g. finite coordinates with
  degenerate geometry: coincident atoms → r→0 divisions, or extreme
  charges) and overflows the forward in fp16.** The collator checks
  finiteness, not magnitude/geometry. FOR: onset at a specific batch
  mid-epoch. AGAINST: incident B NaNs from batch 0, so H4 alone cannot
  explain B unless H1/H2 contamination followed the first skip (the guard
  preserves accumulated grads from earlier batches in the window, which
  would then be stepped — plausibly stepping inf gradients into the
  weights: the exact interplay the prototype `nan_shield.py` was built to
  audit).
- **H5 — Restored GradScaler scale overflow.** A restored `scale` value can
  be large (growth 2x every growth_interval); the first backward under a
  huge scale can overflow grads to inf; the scaler then skips the step and
  halves the scale — normally self-healing within a few steps, but if the
  FORWARD itself saturates fp16 (activations near 65,504), loss is NaN in
  the forward, the scaler never gets to heal. FOR: incident-B batch-0 NaN.
  AGAINST: forward activation clamps (±6e4) were in place and should have
  bounded this; unverified.
- **H6 — Environment-level fp16 + torch 2.11.0+cu130 on compute 7.5
  instability.** Not yet tested (no fp32/​bf16 control run exists on the
  T4). The T4 (SM 7.5) cannot use bf16 autocast with fp32-like dynamic
  range, so the only escape would be fp32 at ~2× cost/time.

## 3. Affected components (where the code lives)

For the next agent — files/functions that govern the failing path:

| Component | Location | Role in the failure |
|---|---|---|
| Finite-loss skip guard | `syntree/engine/trainer.py :: ResilientTrainer.train` (the `non-finite loss ... skipping batch` branch) | Skips the batch but preserves accumulated grads from earlier batches in the accumulation window; never inspects parameters/optimizer state |
| Optimizer step path | `syntree/engine/trainer.py` (`scaler.unscale_ → clip_grad_norm_ → scaler.step → scaler.update`) | found_inf skip + scale-reduction logic; `scale_before/after` step-detection exists (Auditing F-18) |
| fp16 autocast region | `ResilientTrainer.train` forward + `syntree/models/*` | Where an overflow would occur; `finite_act_clamp` guards PaiNN layer boundaries |
| Torsion loss | `syntree/models/torsion_head.py :: ContinuousTorsionHead.loss_fn / log_prob` | H3 suspect; runs under autocast |
| Collator non-finite rejection | `ResilientTrainer.collate_batch` + `_has_nonfinite` | Rejects NaN/Inf inputs but not degenerate-but-finite geometry |
| Checkpoint restore | `syntree/utils/checkpoint.py :: restore_latest` | WAS the silent-poisoning entry (Incident B): verified structure, not finiteness. **Now refuses non-finite payloads loudly** (2026-10-29 safeguard, not a root-cause fix) |
| Checkpoint save | `CheckpointManager.save_checkpoint` | Saved and uploaded poisoned checkpoints; no finiteness gate on save |
| Auto-resume flow | `main.py` (`--resume-auto` handling) + `notebooks/run_3d_syntree.ipynb` cell 8 | Restored the poisoned state; now fresh-start by default with archive (2026-10-29) |
| Synthon catalog embeddings | `syntree/chemistry/catalog.py` (cached embeddings fed to every forward) | Unexamined input that is constant across batches — a NaN here would NaN every loss even with healthy weights (candidate to rule out cheaply) |

## 4. Known consequences

- All checkpoints ≥ epoch 6 of incident A (including the uploaded epoch-15
  state) must be assumed NaN-poisoned; do not fine-tune from them.
- The skip-forever behavior wastes the entire remaining budget (no circuit
  breaker existed at the time).
- The incident destroyed ~2/3 of the 2026-10-28 training budget and forced
  the from-scratch restart decision.

## 5. What the 2026-10-29 iteration did (and deliberately did NOT do)

Done (safeguards only — none of these address the root cause):

1. `CheckpointManager.restore_latest` now verifies model/optimizer/scaler
   finiteness and REFUSES non-finite payloads (loud RuntimeError, file kept).
2. Fresh start is the default policy for the restart: `--fresh` /
   `training.fresh_start` archives old state instead of deleting it, and
   fresh + `--resume-auto` is a hard error (`main.resolve_start_mode`).
3. Fresh runs log a param-health line (all-parameter finiteness + max |w|)
   at startup so a from-scratch run is provably clean at t=0.
4. The complete failure evidence is preserved (this file + `docs/evidence/`).

NOT done, per explicit instruction: no NaN-shield integration
(`prototypes/train_observatory/nan_shield.py` remains a prototype), no
gradient/parameter gate in the step path, no fp32 fallback, no torsion-head
numerical audit, no per-batch forensics dump.

## 6. Investigation plan for the next agent (evidence-first)

Ordered so each step either confirms or kills a hypothesis cheaply:

1. **Rule out the catalog embeddings (H-catalog, minutes).** On the Colab
   data: `torch.isfinite(SynthonCatalog(...).embeddings).all()` and per-row
   scan; also check the cached embedding file on disk.
2. **Forensically inspect the poisoned checkpoints (H1/H5, minutes).** They
   are preserved in the Colab workspace (or HF history revisions): load
   `checkpoint_epoch_15.pt`, run `_find_nonfinite_checkpoint_tensors`
   (`syntree/utils/checkpoint.py`), record which tensors (model? moments?
   scaler.scale?) are non-finite and their first indices. This single
   observation splits H1 into sub-cases and largely explains Incident B.
3. **Add batch-forensics capture at skip time (H2/H3/H4).** When the guard
   fires, dump per-component losses (reaction/synthon/torsion), grad
   finiteness, parameter finiteness, scaler scale, and the batch's sample
   indices + input tensor stats to a JSON (the prototype
   `nan_shield.py::NanShield` already implements all of this — integrate it
   or hand-roll a 30-line equivalent).
4. **Localization at onset (H2/H4).** In a controlled fresh run, wrap the
   first non-finite batch's inputs; re-run forward component-by-component in
   fp32 on CPU/GPU to find which term first produces inf/NaN.
5. **Controlled A/B (H6/H5).** Short (~6 epoch) runs on the T4: (a) fp16 as
   configured, (b) fp32 (`system.mixed_precision: "fp32"`). If fp32 runs
   clean to epoch 15 where fp16 died twice, the cause is fp16-specific
   numerics, not data or logic.
6. **Only after the above**: choose the minimal fix (guarded step à la
   nan_shield, torsion-head stabilization, or fp32 for T4) and re-verify
   with the five-file regression batch in docs/HANDOFF_2026-10-29.md.

## 7. Confirmed vs unconfirmed — one-glance summary

| Claim | Status |
|---|---|
| NaN persisted across all batches from first onset in both incidents | Confirmed (logs) |
| Restored epoch-15 checkpoint produces NaN from batch 0 | Confirmed (log) |
| "Verified" checkpoint label did not imply numeric health (pre-2026-10-29) | Confirmed (code + log) |
| Checkpoint ≥ epoch 6 contains non-finite tensors | Assumed; VERIFY with step 2 (not yet run) |
| GradScaler-skip / accumulation interplay allowed weight poisoning | Hypothesis (H2/H4) |
| Torsion head is the entry point | Hypothesis (H3) |
| fp16-specific instability on T4 | Hypothesis (H6); no control run yet |
| Any specific fix resolves the issue | **NOT claimed anywhere in this repo** |
