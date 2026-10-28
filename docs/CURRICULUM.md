# Curriculum Learning — 9-Level Difficulty-Aware Training (Stage 1)

**Implementation:** `syntree/engine/curriculum.py` (production module, wired
into `ResilientTrainer`), reference prototype `prototypes/train_observatory/curriculum.py`.

**Implementation status (honest):**

| Piece | State |
|---|---|
| Feature schema + scoring | Implemented; unit-tested on CPU (`tests/test_curriculum_features.py`) |
| 9-level quantile assignment + cache | Implemented; unit-tested (`tests/test_curriculum_levels.py`) |
| Mixed difficulty sampler (Approach C) | Implemented; unit-tested (`tests/test_curriculum_sampler.py`) |
| Trainer wiring (steps/epoch invariance, logs, checkpoint state, per-level val metrics) | Implemented; CPU-tested end-to-end on the synthetic smoke dataset (`tests/test_curriculum_trainer_integration.py`) |
| **Run on the real 2,194-sample HF dataset** | **NOT yet executed** — requires the Colab T4 run (user side) |
| Level-histogram sanity on real data (balanced bins, plausible difficulty ordering) | **NOT yet verified** — first lines of the run log will show it |
| Curriculum vs no-curriculum controlled comparison | **NOT yet run** — see §7 |

Do not describe curriculum learning as "working on the real data" until the
Colab run's log lines in §6 have been captured and reviewed.

---

## 1. Objective

The seed-slice dataset (train 2,194 / val 286) is small and heterogeneous in
difficulty (tiny rigid synthons vs large flexible pockets, rare reaction
families vs common ones). Training uniformly can spend most updates on a
hard tail the model cannot yet fit while under-training the mass of easy
structure. The curriculum biases the sampling distribution early toward
learnable examples and progressively shifts mass to harder levels as
validation milestones are met — **without ever removing hard examples**
(they are reachable from epoch 0) and **without dropping easy examples at
the end** (uniform floor).

## 2. Difficulty features (schema v1, `FEATURE_NAMES`)

Every training example receives a reproducible [0, 1] score from 11 features:

| Feature | Source | Meaning |
|---|---|---|
| `ligand_n_atoms` | PyG sample (`ligand_z`) | size of the ligand core to be predicted |
| `pocket_n_atoms` | PyG sample (`pocket_pos`/`pos`) | steric/complexity of the environment |
| `pocket_mean_abs_charge`, `ligand_mean_abs_charge` | PyG sample | electrostatic complexity |
| `synthon_n_atoms`, `synthon_n_rings`, `synthon_n_rotatable`, `synthon_n_hetero`, `synthon_symmetry_ratio` | RDKit on the target synthon's SMILES (via the catalog; cached per SMILES) | molecular difficulty: size, ring complexity, flexibility, heteroatoms, symmetry (fraction of unique canonical atom-rank classes) |
| `reaction_family_rarity`, `core_handle_rarity` | TRAIN-split statistics, `-log2(p)` | reaction complexity: rare classes are harder |

Design decisions:

- **Backend-agnostic.** Tensor features read only fields every data backend
  (local / HF shards / kaggle_offline / trajectory_pt) guarantees, so the
  curriculum works unchanged on all of them.
- **Catalog-powered chemistry features.** Ring/rotatable/symmetry features
  come from the catalog SMILES of `target_synthon` — computed once per
  synthon id (2680 RDKit parses, cached in-process + persisted in the JSON
  artifact), not once per epoch.
- **Train-only normalization.** Each feature is min-max normalized with the
  1st/99th percentile of the TRAIN split; validation rows are clipped to
  [0, 1] and can never influence the norm bounds (leakage rule).
- **Weights are config.** `curriculum.feature_weights` (see configs/
  `train_colab_12h.json`); defaults encode the relative-importance judgment
  (flexibility/sterics/rings weighted above pure size). The weights are part
  of the cache signature — changing them invalidates and recomputes levels.

Known limitation (deliberate, documented): the schema has no per-sample 3D
conformer count or explicit steric-clash measure — those were considered
(synthetic-tree discussion, 2026-10-28) but require per-sample ETKDG
conformer enumeration, which is too expensive to recompute per catalog/​config
change at the seed-slice stage and was deferred; `pocket_n_atoms` +
`synthon_n_rotatable` + charge features are the v1 proxies. Revisit when the
full 75k dataset is in use.

## 3. Level assignment (9 balanced bins)

- Train scores are cut into `n_levels = 9` quantile bins (L1 easiest … L9
  hardest), so bins are approximately balanced by construction
  (~244 train examples per level at 2,194 samples).
- **Bin edges come from the train split only.** Validation samples are
  assigned levels with those same edges, purely for per-level reporting.
- The train/val SPLIT ITSELF IS NEVER MODIFIED by the curriculum. (The
  stricter leakage-proof re-split — scaffold/reaction-family/conformer-
  grouped — remains the separate planned prerequisite task for the
  no-curriculum comparison; see HANDOFF §Next Steps.)
- The complete assignment (features, norms, edges, levels, counts) is
  persisted to `curriculum.cache_path` (default
  `./experiments/curriculum_levels_v1.json`) keyed by a signature of
  (schema, config, dataset sizes, catalog ids-hash). Rebuilds are automatic
  on mismatch, and log lines state whether levels were `computed` or
  `cache`-loaded — the assignment is therefore reproducible and auditable.

## 4. Sampling (Approach C — difficulty-aware MIXED sampling)

`CurriculumSampler` (a torch `Sampler`) replaces the train loader's shuffle:

- **Per-epoch draw count = `len(dataset)`** → steps/epoch, LR schedule, and
  time budget are IDENTICAL to the plain-shuffle baseline (asserted in
  tests). Sampling is with replacement (that is what allows level
  reweighting).
- **Stage mixture.** Stage `s ∈ [0, n_stages-1]` centers a Gaussian over
  level space (start: level 2 — easy but not trivial; end: hardest level)
  with spread `sigma`, blended with a uniform floor:
  `w(l) = (1-eps)·N(center_s, sigma)(l) + eps·U(l)`, `eps = min(min_weight·n_levels, 0.9)`.
  Every level always has weight ≥ `min_weight` (0.03) — hard examples are
  **never permanently removed**; at the final stage the mixture is dominated
  by hard levels but easy ones remain at the floor.
- **Milestone-driven advancement, never fixed epochs.** After each epoch's
  validation, `maybe_advance` moves to the next stage when
  `milestone_metric` (default `val_loss`) improves by `improve_thresh`
  (0.005) over the best-so-far of the current stage, OR when `patience`
  (2) epochs pass without a milestone (guarantees progress inside a 40-epoch
  / 11.5 h budget: worst case ~6·(1 + a few) epochs to reach the last stage).
  Missing/non-finite metric values never advance the stage except via
  patience.
- **Determinism.** The sampler uses its own `random.Random` seeded by
  `curriculum.seed + epoch` — it does not consume the global RNGs, so model
  init and any stochastic augmentation are unaffected; the same epoch always
  yields the same index sequence.
- **Checkpointing.** The stage / reference / patience counters ride in the
  checkpoint `extra_state["curriculum"]`; `--resume-auto` restores them and
  logs the restored stage. (Fresh restarts do not restore anything.)

## 5. Visibility (what the logs show)

Startup (after catalog + dataset load):

```
[curriculum] enabled: levels=9 stages=6 milestone=val_loss (improve>=0.005, patience=2) | assignment=computed
[curriculum] train level counts: L1=244 L2=243 ... L9=243 | val level counts: L1=31 L2=32 ...
[curriculum] stage 0/5 center=2.00 | level weights: L1=0.180 ... L9=0.050
```

Per epoch:

```
[curriculum] epoch 012 | stage 2/5 | sampled: L1=18 L2=31 ... L9=64
[curriculum] epoch 013: stage -> 3/5 (val_loss improved 7.4900 -> 7.4200 (>= 0.005 milestone)); new center=6.00
```

Metrics artifacts: every epoch entry in `metrics.jsonl` / `history.json` /
`latest_metrics.json` carries `curriculum_stage`, `curriculum_center`,
`curriculum_level_weights`, `curriculum_level_counts`, and per-level
validation metrics `L{1..9}/val_loss` and `L{1..9}/val_reaction_acc`.

Per-level validation detail: validation samples are aligned to levels via the
sequential val loader (shuffle=False, drop_last=False). If the collator
dropped any non-finite sample during a validation pass, index alignment
would be undefined, so the per-level summary REFUSES loudly for that epoch
instead of emitting misaligned numbers.

## 6. Verification protocol (user side, Colab T4)

Run `python main.py --mode train --config configs/train_colab_12h.json`
(notebook cell 9; `--fresh` comes from TRAIN_ARGS). Then check, in order:

1. **From-scratch confirmation** — log contains:
   `[main] FRESH START requested (config training.fresh_start)...`,
   `[trainer] FRESH START CONFIRMED: no checkpoint restored...`,
   `[trainer] FRESH START param health: all N parameter tensors finite`,
   `[trainer] run mode: FRESH (from-scratch weights...)`.
2. **Curriculum build** — the three `[curriculum]` startup lines; all 9
   train levels populated and roughly balanced (each ~244 ± small tolerance);
   assignment=computed on first run, cache on re-runs.
3. **Per-epoch lines** — `sampled:` counts sum ≈ 2,194; every level > 0;
   stage never decreases; at least one milestone or patience advance per
   few epochs.
4. **Per-level val metrics** — `L1/val_loss` … `L9/val_loss` present in
   `history.json`; expect a monotone-ish gradient (easier levels lower loss
   / higher accuracy) as a face-validity check of the difficulty ordering.
5. **Budget neutrality** — steps/epoch = 60 (unchanged), VRAM peak
   unchanged vs the 2026-10-28 runs.
6. **NaN watch** — if any `non-finite loss` warning appears, STOP and
   capture per docs/PERSISTENT_ERRORS_NAN.md §6 (the NaN error is documented,
   not fixed; the curriculum changes sampling only and cannot mask it).

## 7. Controlled comparison (planned, NOT yet run)

To prove the curriculum earns its complexity, run the same fresh seed with
`curriculum.enabled=false` (config `train_colab_12h_noCurr.json` — to be
created at comparison time) for the same epoch count, then compare per level:
success criteria are (a) L7–L9 val metrics improve with curriculum,
(b) L1–L3 do not regress, (c) overall val_loss at matched epoch ≥ baseline.
Decision rule: if (a) fails with (b) and (c) holding, keep the uniform
baseline and drop the curriculum (document the ablation result either way).

## 8. Config reference

`curriculum` block (see `configs/train_colab_12h.json` for the live values):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | false | master switch (true in the Colab config) |
| `n_levels` | 9 | difficulty bins; validated to [2, 20] |
| `n_stages` | 6 | curriculum stages (centers spread level 2 → hardest) |
| `min_weight` | 0.03 | per-level floor; validated to (0, 0.5] |
| `sigma` | 1.8 | Gaussian spread over level space |
| `seed` | 42 | sampler epoch-seed base |
| `milestone_metric` | val_loss | advancement metric (epoch metrics dict key) |
| `improve_thresh` | 0.005 | milestone improvement threshold |
| `patience` | 2 | epochs without milestone before forced advance |
| `feature_sample_stride` | 1 | feature-pass stride (>1 caps cost on huge datasets; default covers 100% of the train split) |
| `cache_path` | ./experiments/curriculum_levels_v1.json | level-assignment artifact |
| `feature_weights` | see config | per-feature difficulty weights (part of the cache signature) |

All keys are validated fail-closed in `main.validate_config` before any
dataset is touched.
