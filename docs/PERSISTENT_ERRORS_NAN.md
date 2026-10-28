# PERSISTENT ERRORS — Stage-1 NaN losses (UNRESOLVED, deliberately NOT fixed)

**Status as of 2026-10-29 (updated after the CPU pre-flight iteration):** the
error documented here still has no confirmed root cause for the ORIGINAL
onset. However, the 2026-10-29 forensic pass (§6 steps 1–2, executed — see
§6.5) CONFIRMED the checkpoint poisoning (H1) for Incident B, RULED OUT the
catalog and the raw data as sources, and discovered a NEW primary anomaly:
**the GradScaler scale had already collapsed to a denormal (1.84e-40) at the
last "healthy" checkpoint and reached exactly 0.0 by epoch 15** — the fp16
gradient path was broken from the start of the run, not from epoch 6.
Per the operator's explicit instruction, no fix has been applied.

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
  **UPDATE 2026-10-29: CONFIRMED for Incident B by direct forensics —
  see §6.5: the epoch-15 checkpoint contains 113/119 model tensors that
  are 100% NaN plus all matching optimizer moments; the restore-refusal
  safeguard behaves exactly as designed on this file. The remaining open
  question is the MECHANISM that poisoned the weights during epoch 6
  despite the found_inf-gated step path.**
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
  not running. **Still the leading candidate for the epoch-6 poisoning
  mechanism (H1 explains only Incident B).**
- **H3 — Torsion head numerics (von Mises / log Bessel) under fp16.**
  torsion NLL was flat for five epochs and involves `kappa`, `log_prob`,
  `log(bessel_i0)` — classic NaN producers at 0/∞ kappa. The training code
  runs torsion math inside autocast. FOR: the component was frozen (no
  learning progress) right before failure. AGAINST: no component-level
  loss was logged at onset, so this is inference, not evidence. **UPDATE
  2026-10-29: the torsion head itself is already fp32-isolated with kappa
  clamped to [0.1, 50] and a stable log_bessel_i0 (syntree/models/
  torsion_head.py) — the head's FORWARD is an unlikely source; the fp16
  risk is the BACKWARD through the PaiNN backbone (vector norms / sqrt /
  1/r terms), which remains unproven.**
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
  audit). **UPDATE 2026-10-29: every tensor of all 2,194 train + 286 val
  samples is finite with modest magnitudes (§6.5); a non-finite INPUT is
  ruled out — only degenerate-but-finite geometry remains possible.**
- **H5 — Restored GradScaler scale overflow.** A restored `scale` value can
  be large (growth 2x every growth_interval); the first backward under a
  huge scale can overflow grads to inf; the scaler then skips the step and
  halves the scale — normally self-healing within a few steps, but if the
  FORWARD itself saturates fp16 (activations near 65,504), loss is NaN in
  the forward, the scaler never gets to heal. FOR: incident-B batch-0 NaN.
  AGAINST: forward activation clamps (±6e4) were in place and should have
  bounded this; unverified. **UPDATE 2026-10-29: the real restored scale
  was the OPPOSITE of "huge" — 1.84e-40, a collapsed fp32 denormal (see
  the new scaler-collapse finding below). H5 as originally stated is
  disproven; the collapse direction is confirmed instead.**
- **H6 — Environment-level fp16 + torch 2.11.0+cu130 on compute 7.5
  instability.** Not yet tested (no fp32/​bf16 control run exists on the
  T4). The T4 (SM 7.5) cannot use bf16 autocast with fp32-like dynamic
  range, so the only escape would be fp32 at ~2× cost/time. **UPDATE
  2026-10-29: STRENGTHENED — a CPU fp32 smoke run on the SAME real data,
  same model code, same loss (2 epochs, small model) trains cleanly and
  decreases loss 7.77 → 6.59 with zero NaN/OOM skips
  (docs/evidence/2026-10-29_curriculum_realdata_verification.json). The
  fp16-vs-fp32 A/B on the T4 remains the decisive control.**
- **H7 (NEW, 2026-10-29) — GradScaler scale collapse as the PRIMARY
  anomaly.** Direct forensics on the preserved checkpoints show
  `scaler.scale = 1.83671e-40` (an fp32 denormal, ~2^-132) at the epoch-5
  "last healthy" save and `scale = 0.0` at epoch 15. Reaching 2^-132 from
  the initial 2^16 requires ~148 found_inf halvings — i.e. gradients
  overflowed fp16 REPEATEDLY from the very start of the run. Below a scale
  of ~2.9e-39, `1/scale` overflows fp32, and once the scale underflows to
  exactly 0.0 the scaler is in a permanently dead state (scaled loss ≡ 0;
  any inf in the raw backward becomes 0×inf = NaN; unscale divides by
  zero). Corroboration: the run's val accuracies were FROZEN from epoch
  0/1 (val_synthon_acc = 0.48951, val_reaction_acc = 0.76712 constant
  across epochs 0–5, see the remote progress.json) and train_loss moved
  only 7.353 → 7.279 after epoch 1 — the "healthy" phase was largely NOT
  learning because fp16 gradients underflowed to zero at denormal scales.
  The NaN at epoch 6 is the terminal symptom of this collapse (plus the
  still-open weight-poisoning mechanism). Additional detail: the remote
  progress.json shows a ~52-minute gap and an elapsed-hours RESET between
  the epoch-5 and epoch-6 entries — the onset happened in a RESTARTED
  process that had restored the epoch-5 state INCLUDING the collapsed
  scale.

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

## 6. Investigation plan (evidence-first) — status after the 2026-10-29 CPU pass

Ordered so each step either confirms or kills a hypothesis cheaply:

1. **[EXECUTED 2026-10-29 — RULED OUT]** Rule out the catalog embeddings.
   Result: all 2,680 × 256 fp32 embeddings finite (abs ∈
   [2.1e-8, 0.247]), zero rows: 0, ids_hash e90f58f92b14. Evidence:
   `docs/evidence/2026-10-29_catalog_and_data_finiteness.json`.
   Also scanned: every tensor of all 2,194 train + 286 val samples is
   finite with modest magnitudes (pocket_pos ≤ 21.2, ligand_pos ≤ 193.1,
   charges ≤ 1) — non-finite INPUTS ruled out.
2. **[EXECUTED 2026-10-29 — DECISIVE for H1]** Forensically inspect the
   poisoned checkpoints. Result: epoch-15 = 113/119 model tensors 100%
   NaN (all NaN, no ±inf; 5.8M bad elements) + 226/339 optimizer tensors
   100% NaN (11.6M bad elements; exp_avg AND exp_avg_sq of every poisoned
   param) + scaler scale = 0.0. Epoch-5 (last healthy) = everything
   finite, but scaler scale = 1.83671e-40 (denormal collapse, see H7).
   Epoch-5 → epoch-15 diff: 113 params changed, all to NaN. The
   production `restore_latest` REFUSES this file loudly (verified against
   the real file). Evidence:
   `docs/evidence/2026-10-29_checkpoint15_forensics.json`.
3. **[NEXT — Batch 2]** Add batch-forensics capture at skip time (H2/H7).
   When the guard fires, dump per-component losses (reaction/synthon/
   torsion), grad finiteness, parameter finiteness, scaler scale, and the
   batch's sample indices + input tensor stats to a JSON (the prototype
   `nan_shield.py::NanShield` already implements all of this — integrate
   it or hand-roll a 30-line equivalent). **With the scale-collapse
   finding, also log `scaler.get_scale()` at every step boundary — the
   collapse trajectory (2^16 → 0) is now the primary observable.**
4. **[NEXT — Batch 2]** Localization at onset (H2/H4). In a controlled
   fresh run, wrap the first non-finite batch's inputs; re-run forward
   component-by-component in fp32 on CPU/GPU to find which term first
   produces inf/NaN.
5. **[CRITICAL CONTROL — Batch 2]** Controlled A/B (H6/H7): short (~6
   epoch) runs on the T4: (a) fp16 as configured, (b) fp32
   (`system.mixed_precision: "fp32"`). If fp32 runs clean, the cause is
   fp16-specific numerics. **The CPU fp32 smoke run already passed
   cleanly on real data (H6 update above), so this A/B is now strongly
   expected to point at fp16; if the fresh fp16 run ALSO collapses the
   scale again (watch for it via the new scale logging), the decision
   rule in DECISIONS.md D14.2 applies: flip the T4 config to fp32.**
6. **Only after the above**: choose the minimal fix (guarded step à la
   nan_shield, backbone-side fp32 isolation of the inf-prone backward
   terms, or fp32 for T4) and re-verify with the regression batch.

## 6.5 New evidence summary (2026-10-29 CPU forensics pass)

All artifacts in `docs/evidence/`; raw JSON reports alongside this doc.

| Finding | Status | Implication |
|---|---|---|
| Catalog embeddings all finite (2,680×256) | Confirmed | Catalog ruled out as NaN source |
| All train/val sample tensors finite, modest magnitudes | Confirmed | Non-finite inputs ruled out |
| Epoch-15 checkpoint: 113/119 model tensors 100% NaN; all matching optimizer moments 100% NaN; scale=0.0 | Confirmed | H1 for Incident B; every checkpoint ≥ epoch 6 unusable (assumption → fact) |
| `restore_latest` refuses the real epoch-15 file with the expected message | Confirmed (live test) | The 2026-10-29 safeguard works on the actual poisoned artifact |
| Epoch-5 checkpoint fully finite BUT scaler scale = 1.83671e-40 | Confirmed | NEW H7: fp16 gradient path broken from run start |
| Scaler scale = 0.0 at epoch 15 (dead state) | Confirmed | Explains permanent all-batch skipping after onset |
| val accuracies frozen from epoch 0/1 (0.4895 / 0.7671 / 0.0944) | Confirmed (progress.json) | "Healthy" epochs were largely not learning (underflowed grads) |
| ~52-min gap + elapsed reset between epoch-5 and epoch-6 entries | Confirmed (progress.json) | Onset happened in a RESTARTED process carrying the collapsed scale |
| CPU fp32, real data, same code: 2 clean epochs, loss 7.77 → 6.59, 0 NaN/OOM | Confirmed | fp16-specific pathology strengthened (H6) |
| Early model predicts STOP class for every sample (constant argmax) | Confirmed (live check) | The frozen val_synthon_acc = 0.4895 is the constant-STOP baseline (48.4% of train / 48.9% of val samples are terminal stop steps) — NOT a metrics bug; expect it to persist until argmax predictions diversify |

## 7. Confirmed vs unconfirmed — one-glance summary

| Claim | Status |
|---|---|
| NaN persisted across all batches from first onset in both incidents | Confirmed (logs) |
| Restored epoch-15 checkpoint produces NaN from batch 0 | Confirmed (log) |
| Epoch-15 checkpoint contains non-finite model AND optimizer tensors (113/119 + 226/339, 100% NaN) | **Confirmed (2026-10-29 forensics — was "Assumed")** |
| Scaler scale was ALREADY collapsed (1.84e-40) at the last healthy checkpoint; 0.0 at epoch 15 | **Confirmed (2026-10-29 forensics — NEW)** |
| The "healthy" epochs 0–5 were largely not learning (frozen val accuracies, underflowed fp16 grads) | **Confirmed (progress.json + scale math)** |
| Catalog embeddings / raw data inputs are finite | **Confirmed (2026-10-29 scan — ruled out as sources)** |
| GradScaler-skip / accumulation interplay allowed weight poisoning | Hypothesis (H2) — the mechanism that applied NaN grads despite found_inf gating is STILL UNIDENTIFIED |
| fp16-specific instability on T4 | Hypothesis (H6), strongly strengthened by the clean CPU fp32 run; T4 A/B pending |
| Scaler collapse trajectory 2^16 → 0 driven by repeated found_inf (fp16 gradient overflow) from run start | Hypothesis (H7) — consistent with all evidence; needs step-level scale logging to confirm |
| Any specific fix resolves the issue | **NOT claimed anywhere in this repo** |
