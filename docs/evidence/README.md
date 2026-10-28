# Evidence Summary — 2026-10-28 Persistent NaN Errors (Stage 1, Colab T4)

This file consolidates the FACTS of the two NaN incidents from the
2026-10-28 training sessions. It is reconstructed from the preserved
conversation history and the verbatim log excerpts supplied by the operator;
raw console logs of the FIRST incident were not captured to a file at the
time, which is itself a documented gap (see "Evidence gaps" below).

- Full verbatim log of the second incident:
  `docs/evidence/2026-10-28_resume_epoch15_all_batches_nan.log`
- Prototype NaN-forensics drill (isolated sandbox, NOT the production run):
  `docs/evidence/prototype_nan_forensics_ep2_b4.json`

## Incident A — first NaN onset (earlier fresh run, epochs 1–7)

Observed facts (from session history):

1. Stage-1 fresh run on Colab T4 (fp16 + GradScaler, auto-scaled batch), seed
   slice dataset: train=2,194 / val=286, catalog 2,680 synthons.
2. Epochs 1–5 were numerically healthy:
   - train_loss 7.2804 (ep 4) -> 7.2794 (ep 5) (cumulative −0.997%),
   - val_loss plateaued at 7.4951 for consecutive epochs,
   - reaction CE ~0.5206→0.5227 (micro-rise), synthon CE ~6.1026→6.1003
     (micro-drop), torsion NLL ~1.8349→1.8356 (≈ flat for 5 epochs),
   - peak VRAM ~10.0 GB, 0 OOM skips, 0 NaN skips, LR 3e-4, effective
     batch 74 (epoch-5 report).
3. Epoch 6: from batch 44 the training loop logged
   `non-finite loss ... skipping batch` for EVERY subsequent batch — 16
   consecutive skips within epoch 6.
4. Epoch 7: NaN from batch 0; cumulative skip counter reached 45.
5. Epoch 6 validation loss was NaN.
6. The epoch-5 checkpoint was uploaded to the HF Hub during this run; the
   run later continued (checkpoints up to epoch 15 exist — see Incident B).

## Incident B — resume from epoch 15 (second run, 2026-10-28 16:11 UTC)

Verbatim log preserved in this directory. Key lines:

```
[main] restored model architecture from checkpoint (runtime flags preserved)
[trainer] auto-scale: batch 32 -> 37 | accum 2 -> 2 | effective 64 -> 74 | lr scale 1.156
2026-10-28 16:12:11,080 | INFO | syntree.utils.checkpoint | Restored verified checkpoint: epoch 15 (step 41). Training resumes at epoch 16.
[trainer] WARNING: non-finite loss at batch 0 (loss=nan); skipping batch. (total skips: 1)
... (identical warning for every observed batch 0..47)
```

Observed facts:

1. The run was launched in resume mode (`--resume-auto`) and restored the
   "verified" epoch-15 checkpoint (step 41), resuming at epoch 16.
2. EVERY observed batch of the resumed epoch produced loss=NaN, including
   batch 0 — i.e. the NaN is present before any optimizer step of this run.
3. The finite-loss guard skipped each such batch (47+ consecutive skips in
   the captured window); no crash, no exception, no recovery.
4. The checkpoint passed every structural verification implemented at the
   time (manifest/progress/epoch consistency + shape adaptation), yet was
   numerically unusable — "verified" did not include finiteness.

## Consequences

- Epochs 6+ of the first run and epoch 16+ of the resumed run produced no
  learning progress (every batch skipped), while still burning GPU time and
  checkpoint-upload bandwidth (epoch 6..15 checkpoints are NaN-poisoned).
- Any checkpoint at or after the first NaN onset must be assumed poisoned.
- This is the direct motivation for the 2026-10-29 decisions: restart from
  scratch with archival (never silent purge), and finiteness verification at
  restore time (see docs/HANDOFF_2026-10-29.md and docs/DECISIONS.md).

## Evidence gaps (honesty section)

- No per-component loss decomposition (reaction/synthon/torsion) was logged
  at the moment NaN appeared; the skip warning carries only loss=nan.
- No parameter/optimizer/gradient health snapshots exist for the incident.
- The precise first-NaN batch (epoch 6, batch 44) was never dumped to disk.
- The first-incident console log exists only as quoted text in the session
  history; only the second incident's log is preserved verbatim in-repo.
These gaps are what the observability prototype (`prototypes/train_observatory/`)
is designed to close when integrated.
