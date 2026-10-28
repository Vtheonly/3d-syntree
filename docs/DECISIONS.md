# Architecture & Engineering Decision Record — 3D-SynTree

**Scope:** every architectural and technical decision made across the
mission sessions (2026-10-27 → 2026-10-29), reconstructed so a fresh reader
can understand WHAT was decided, WHY, WHAT was rejected and why, and HOW
MUCH evidence stands behind each choice. Companion documents:
`docs/CURRICULUM.md` (curriculum design), `docs/PERSISTENT_ERRORS_NAN.md`
(the open NaN failure), `docs/HANDOFF_2026-10-29.md` (state + next steps),
`docs/ENGINEERING_AUDIT_REPORT.md` (the 2026-10-28 audit).

Confidence labels used below:
- **[EVIDENCE]** established by tests, measurements, or logs;
- **[JUDGMENT]** engineering judgment / best practice, no direct measurement;
- **[ASSUMPTION]** plausible but unverified — needs the listed validation.

---

## 1. The problem being solved

**Mission:** build and train **3D-SynTree**, a structure-based drug design
(SBDD) generative model that tears down the *synthesizability wall*: 3D
pocket-conditioned diffusion models generate chemically impossible chimeras
(<20% retrosynthetic solvability), while forward-synthesis models guarantee
synthesis but are blind to 3D binding geometry and drift to flat aromatic
chemistry. 3D-SynTree formulates molecular design as a Markov decision
process over a **reaction-constrained synthon action space with 3D torsion
guidance**: choose a certified building block, apply a validated medicinal
reaction, predict the continuous dihedral that minimizes steric clash in the
pocket. Every output is chemically valid by construction and carries a
synthesis recipe.

**Constraints:**

- Hardware: Colab Tesla T4 (14.56 GB VRAM, compute 7.5, no bf16 autocast
  benefit), fp16 + GradScaler; optional RTX 6000 Ada / H100 (bf16, TF32)
  and Kaggle 96 GB offline paths. Single-GPU, 11.5–12 h wall-clock budget.
- Data: verified open sources (CrossDocked2020 pocket10, PDBbind v2020
  refined, BindingMOAD, Enamine Rush-Delivery BBs); seed slice currently
  2,194 train / 286 val on 2,680 catalog synthons (full corpus ~75k
  planned); HF Hub as dataset/checkpoint transport (needs HF_TOKEN).
- Correctness bar: fail-closed guards everywhere — a silent wrong-data or
  wrong-state training run wastes an entire offline session.
- Operating model: the agent (Z.ai) writes/verifies code in a CPU sandbox;
  the human operator runs GPU tests on their own PC/Colab and returns logs;
  iterations proceed in ≤5-file test batches.

**Objectives (priority order):** (1) execution stability of Stage-1
training on the T4; (2) reliable, restorable, never-silently-corrupting
checkpointing; (3) consistent model/dataset identifiers (catalog
signatures, run IDs); (4) reproducible local testing; (5) curriculum
learning across the full dataset; (6) later: real evaluation metrics +
observability integration + Stage-2 RL at quality.

## 2. Core principles (decided, and why)

1. **Codebase-first** [JUDGMENT]: every fix lives in package modules; the
   notebook is a pure orchestrator. Rejected alternative: notebook-inline
   patching (used early, then deliberately removed in the audit) — it
   diverges from the repo, cannot be tested, and caused the "stale hotfix"
   confusion the audit had to clean up.
2. **Fail-closed over convenient fallback** [JUDGMENT, repeatedly EVIDENCE
   -backed]: unknown data backend, missing real data, non-finite checkpoint
   payloads, ambiguous fresh/resume flags — all raise instead of guessing.
   The alternative (silent fallback) is exactly what produced wasted GPU
   sessions (e.g. synthetic stub silently trained) and the poisoned
   checkpoint silently applied (Incident B).
3. **Historical artifacts are never silently destroyed** [JUDGMENT, born
   from a real incident]: the early `--fresh` purged local AND remote
   checkpoints and wiped the epoch-31 state; the 2026-10-29 policy replaces
   purge with ARCHIVE (local) + explicit `--purge-remote` opt-in.
4. **Prototype before integration for anything touching the training loop**
   [JUDGMENT]: observability/curriculum/NaN-shield were built and verified
   on an isolated mini-trainer first (bit-identical training, <2 ms/step
   overhead), and only the curriculum was promoted into production for this
   iteration; the observatory/nan-shield integration is deliberately
   deferred until the NaN evidence is collected.
5. **Evidence over claims** [JUDGMENT]: every status doc separates verified
   facts from hypotheses (see PERSISTENT_ERRORS_NAN §7); nothing is called
   fixed/final without a test or log line to cite.

## 3. System layers (final architecture)

```
notebooks/run_3d_syntree.ipynb   orchestration only (env vars, CLI calls)
main.py                          CLI: train | rl | generate | evaluate | info
syntree/
  chemistry/   catalog (Enamine synthons + pharm3D encoder), reactions,
               conformer, validator, hotspots
  data/        crossdocked | hf_loader (shards) | kaggle_loader | trajectory,
               splits, featurizer, fragmenter, curation
  models/      policy = PaiNN equivariant backbone + pocket/ligand encoders
               + reaction head + synthon head + continuous torsion head
  engine/      trainer (resilient Stage-1), rl (PPO), gflownet, dpo,
               generator, evaluator, environment, curriculum (NEW)
  core/        masking (fp16-safe constants), registry
  utils/       checkpoint (local+HF state machine), hardware (AMP/device
               resolution, auto-scale), logger
configs/       per-target JSON configs (colab 12h, kaggle 96 GB, h100, rl)
tests/         60+ test files, incl. the new 5-file batch
prototypes/    train_observatory (verified, not yet integrated)
```

## 4. Decision records

### D1 — Problem formulation: reaction-constrained synthon MDP + torsions

- **Decision.** Generate ligands as a decision sequence (reaction family →
  catalog synthon → dihedral) instead of atom-by-atom coordinate diffusion.
- **Reasoning.** Chemical validity and retrosynthetic solvability become
  structural properties (certified synthons + validated reactions + forward
  replay), while still conditioning on 3D pocket geometry through an
  equivariant backbone and torsion head.
- **Alternatives rejected.** (a) Continuous-coordinate diffusion
  (TargetDiff-style): strong docking numbers, but 85%+ outputs fail
  retrosynthesis and validity; (b) 2D forward-synthesis-only (SynNet-style):
  no pocket conditioning, flat outputs; (c) hybrid two-model pipelines:
  double the failure surface, no shared 3D state.
- **Consequences.** The action space is discrete+continuous mixed; losses
  are reaction CE + synthon CE + torsion NLL (von Mises). Catalog quality
  and reaction coverage become upstream dependencies of model quality.
- **Confidence.** [JUDGMENT] for the formulation (the mission brief's
  motivation), [EVIDENCE] that the mechanism works end-to-end at smoke
  scale (audit §6: forward-replay recovered a catalog-assembled ligand
  exactly).

### D2 — Two-stage training: supervised pretraining, then RL fine-tuning

- **Decision.** Stage 1 = supervised next-action prediction on real
  pocket–ligand–trajectory data. Stage 2 = RL (PPO default; GFlowNet-TB and
  DPO implemented as switchable alternatives) over generated molecules with
  3D/chemistry rewards.
- **Reasoning.** Offline data teaches the reaction grammar and pocket
  geometry; RL then optimizes reward (clash, contact, Fsp3, QED, validity)
  which has no tractable supervised target.
- **Alternatives rejected.** (a) RL from scratch — intractable (reward is
  sparse, grammar huge); (b) pure supervised — cannot optimize the design
  objective; (c) offline-only DPO — needs preference pairs at scale we
  don't have yet; kept implemented for later.
- **Consequences.** Stage-1 stability gates everything; Stage-2 must be
  memory-isolated on a T4 (see D6).
- **Confidence.** [EVIDENCE] Stage-2 ran cleanly (episodes 32–47 logs);
  Stage-1 currently blocked by the documented NaN failure.

### D3 — Model: PaiNN backbone + three heads, gradient checkpointing

- **Decision.** Equivariant PaiNN backbone (hidden 256, 8 layers, 8 heads on
  T4-scale configs) + reaction head, synthon head (catalog embedding
  scoring), continuous torsion head; `gradient_checkpointing=true` on
  16 GB-class GPUs (h100 config keeps it off).
- **Reasoning.** SE(3)-equivariant message passing is required to respect
  pocket geometry; PaiNN is a strong cost/quality point for the VRAM budget;
  checkpointing cut activation memory ~8× on the T4.
- **Alternatives rejected.** Non-equivariant GNN (breaks rotation
  invariance of torsion targets); larger backbones (won't fit); disabling
  checkpointing on T4 (14.1 GB → OOM in PaiNNInteraction, empirically).
- **Consequences.** Throughput drops ~20–30% with checkpointing — accepted
  for stability. Runtime-only flag must survive checkpoint restore
  (implemented in `restore_arch_with_runtime_flags`).
- **Confidence.** [EVIDENCE] OOM fix verified (95abd96 battery), [JUDGMENT]
  on backbone choice.

### D4 — Data: verified open sources, real-data guard, cluster-safe splits

- **Decision.** CrossDocked2020 (HF mirror, MD5-pinned), PDBbind refined,
  BindingMOAD via RCSB, Enamine BBs; MMseqs2 30%-identity sequence
  clustering for splits; strict trainer guard refuses synthetic/stub data
  in production mode; HF-sharded dataset backend with LRU shard cache.
- **Reasoning.** Every source was probed live (byte-exact sizes); leakage
  between splits is prevented at the protein-cluster level; stub datasets
  silently waste runs.
- **Alternatives rejected.** Registration-gated official mirrors (unusable);
  random splits (protein-family leakage); in-memory-only loading (won't
  scale to 75k).
- **Consequences.** Training data quality/quantity is bounded by the seed
  slice until the full shard upload runs (needs HF_TOKEN).
- **Confidence.** [EVIDENCE] for sources and split invariants (audit §2/§6).

### D5 — Mixed precision: fp16 + GradScaler on T4, bf16 on ≥A100

- **Decision.** `system.mixed_precision` selects fp16 (T4) or bf16 (RTX
  6000 Ada/H100), resolved through `resolve_amp_dtype` (explicit dtype —
  never autocast defaults); fp16 uses GradScaler; the fp16-unsafe `-1e9`
  mask fill was replaced by `-1e4` (`syntree/core/masking.py`), plus
  target-unmasking, finite-loss guard, non-finite sample rejection, ±6e4
  activation clamps, worst-case-pool auto-scale probe and `max_pocket_atoms`
  collator trim.
- **Reasoning.** fp16 on compute 7.5 has a 65,504 max finite magnitude; any
  mask/logit arithmetic near ±6e4 risks inf; the fixed constants keep every
  logit representable while keeping masked probability ≈ 0. bf16 has fp32
  dynamic range → no scaler, no fill hazards.
- **Alternatives rejected.** fp32 everywhere (2× slower on the 11.5 h
  budget — rejected before the NaN incident, a decision now under strain);
  pure −1e9 in fp32 only (correct, but one config flip from the exact
  overflow — fail-closed constants preferred).
- **Consequences / HONESTY.** Despite all guards, the NaN failure still
  occurred (PERSISTENT_ERRORS_NAN); the "fp16 is safe now" belief was
  falsified twice. The fp16-on-T4 decision is now [ASSUMPTION] pending the
  fp32 A/B control run — if fp32 runs clean, expect a policy change.
- **Confidence.** [EVIDENCE] for each individual guard (tested by
  `test_gpu_stability_fixes.py` / `test_oom_nan_resilience.py`);
  [ASSUMPTION] that the composition is sufficient.

### D6 — Checkpointing: authoritative state machine + (new) finiteness verification

- **Decision.** `CheckpointManager` keeps manifest.json + progress.json as
  the authoritative record (contiguous epoch bookkeeping, atomic writes),
  syncs to HF Hub, restores with shape-adaptive parameter loading. Added
  2026-10-29: (a) `restore_latest` REFUSES payloads with non-finite
  model/optimizer/scaler tensors (loud RuntimeError; file preserved);
  (b) `archive_local()` replaces silent purge on fresh starts.
- **Reasoning.** Progress/manifest consistency catches torn/corrupt state
  cheaply; shape adaptation tolerates catalog/reaction-count growth;
  Incident B proved structural verification ≠ numerical health, so the
  verification now covers finiteness too.
- **Alternatives rejected.** (a) Keep trusting structural checks — directly
  falsified by Incident B; (b) auto-purge poisoned checkpoints on restore —
  destroys forensic evidence and violates Principle 3; (c) silently fall
  back to epoch 0 on a poisoned payload — hides the failure and risks a
  "surprise" later; refused loudly instead.
- **Consequences.** `--resume-auto` on the poisoned epoch-15 state now
  crashes with an actionable message instead of training NaN forever.
- **Confidence.** [EVIDENCE] (unit-tested refusal path; the incident log is
  the motivating evidence).

### D7 — Fresh restart policy (2026-10-29 mandate)

- **Decision.** The next Stage-1 run starts completely from scratch:
  fresh weights, fresh optimizer/scheduler/scaler, epoch 0. Requested via
  `--fresh` OR `training.fresh_start=true` (the Colab config and notebook
  default it on). Local previous state is ARCHIVED to
  `checkpoints/archive/<UTC-stamp>_<reason>/`; the remote HF repo is left
  untouched unless the operator passes the new explicit `--purge-remote`;
  fresh + `--resume-auto` is a hard error (`resolve_start_mode`); fresh
  startup logs a parameter-health line proving the init state is finite.
- **Reasoning.** Old checkpoints ≥ epoch 6 are assumed poisoned; a silent
  resume path back into them is unacceptable; previous `--fresh` semantics
  (purge local + remote by default) destroyed the epoch-31 state once and
  must never repeat; the start mode must be unambiguous in logs so the
  operator can VERIFY the run is truly from scratch.
- **Alternatives rejected.** (a) Resume from epoch 5 (last healthy) — the
  poisoning timeline is unproven and the epoch-5→6 boundary batch is
  unknown; risk of re-entering a poisoned trajectory; (b) purge-by-default
  on fresh — destroys evidence, violates Principle 3; (c) keep silent
  "fresh overrides resume" precedence — the ambiguity is exactly how a
  poisoned state could sneak back in.
- **Consequences.** The fresh run will overwrite the remote HF checkpoints
  progressively (repo layout is fixed); previous revisions remain
  recoverable via HF commit history — documented in the startup log.
- **Confidence.** [EVIDENCE] for the mechanism (test batch);
  [ASSUMPTION] that from-scratch suffices — the NaN root cause is open and
  the failure may recur (PERSISTENT_ERRORS_NAN §6 step 5 is the control).

### D8 — Curriculum learning: 9 levels, Approach C mixed sampling

- **Decision.** Difficulty scoring (11-feature schema v1: tensor + catalog
  RDKit + train-only rarity features) → robust percentile normalization
  (train-only) → 9 balanced quantile bins → milestone-driven mixed sampling
  with a hard per-level floor (0.03), Gaussian level weights centered from
  level 2 → level 9 across 6 stages, advancing on `val_loss` improvement
  (0.005) or patience (2 epochs); exactly `len(dataset)` draws per epoch
  (budget-neutral); per-epoch + per-level metrics in logs and metrics.jsonl;
  level assignment cached in a signature-keyed JSON artifact; curriculum
  state rides in `extra_state["curriculum"]`.
- **Reasoning (why Approach C).** The dataset is small (2,194); staged
  subsets (Approach A: train on L1–L3 only, then widen) would leave rare
  hard families unseen for many epochs and crash the per-level validation
  signal; per-level loss weighting (Approach B) distorts the objective the
  baseline comparison must share and interacts badly with fp16 loss-scale
  dynamics. Mixed sampling (C) keeps every level present from epoch 0,
  changes only WHICH samples appear in a batch (not the loss), is exactly
  reversible, and is measurable per level.
- **Alternatives rejected (in detail).**
  - *A: staged subset inclusion* — coverage violates "cover the entire
    dataset"; catastrophic forgetting risk; per-level val metrics undefined
    for unseen levels.
  - *B: difficulty-weighted loss re-weighting* — changes gradients, so the
    no-curriculum baseline is no longer a clean control; reweighting
    interacts with GradScaler; harder to attribute effects.
  - *Per-epoch re-assignment of levels* — non-reproducible drift; fixed
    cached assignment is auditable.
  - *9 separate models / per-level LoRA* — absurd at 2,194 samples and one
    T4; explicitly rejected early.
  - *Fixed-epoch stage schedule* — wastes budget when val plateaus, or
    advances too early on slow runs; milestones + patience dominate.
  - *Training on ETKDG conformer counts as difficulty features* — too
    expensive per cache rebuild at this scale; deferred (documented
    limitation).
- **Layer-by-layer.** Features: `tensor_sample_features` (backend-agnostic
  PyG reads) + `catalog_synthon_features` (RDKit, cached) + rarity
  (`-log2(p)` on train). Scoring: `DifficultyModel` (robust 1st/99th
  percentile bounds, config weights). Assignment: `quantile_edges` /
  `level_for_score` (train edges, val by lookup). Sampling:
  `CurriculumSampler` (torch Sampler; per-epoch reseeding independent of
  global RNGs). Reporting: `LevelTracker` per-level val metrics + epoch
  counts. Persistence: JSON cache + checkpoint extra_state.
- **Consequences / HONESTY.** (a) Sampling is with replacement → some
  examples repeat per epoch, others are skipped — acceptable for a small
  dataset, and floors bound the bias; (b) per-level val metrics rest on ~32
  val samples/level → noisy; treat level trends, not single-epoch values;
  (c) the current split is sequence-clustered but NOT scaffold-grouped —
  per-level numbers may overstate generalization until the planned
  leakage-proof re-split lands; (d) all of this is CPU-verified on the
  synthetic smoke set only — real-data verification is the operator's
  next GPU run (CURRICULUM §6).
- **Confidence.** [EVIDENCE] for mechanics (65-test batch);
  [ASSUMPTION] that it helps — the controlled A/B (CURRICULUM §7) has not
  been run.

### D9 — NaN failure policy: document, don't fix (this iteration)

- **Decision.** The persistent NaN losses are documented exhaustively
  (PERSISTENT_ERRORS_NAN.md + preserved logs) and NOT patched; the next
  agent runs the evidence-first investigation (§6 there) before choosing a
  fix. The only safeguard added is the restore-time finiteness refusal
  (D6), which exists to satisfy "cannot silently restore old state" (D7).
- **Reasoning.** Two rounds of blind hardening (mask fill, clamps,
  rejection) did not stop the failure; adding a third (nan_shield step
  gates) before collecting batch-level forensics would change the failure
  surface and possibly mask the root cause. Evidence collection is cheap
  (the shield prototype already localizes + snapshots); fixes are not.
- **Alternatives rejected.** (a) Integrate `nan_shield` now — masks cause,
  violates the operator's explicit instruction; (b) switch to fp32 now —
  a real candidate FIX but also an experiment; it belongs in the planned
  A/B control, not smuggled into this iteration; (c) ignore the error —
  untenable, it consumed the last training budget.
- **Consequences.** The fresh curriculum run may NaN again; the operator
  knows to stop and capture evidence (CURRICULUM §6 step 6).
- **Confidence.** [JUDGMENT] on process; [EVIDENCE] that prior blind fixes
  were insufficient.

### D10 — Observability: prototype-first, integration deferred

- **Decision.** `train_observatory` (RunMonitor: manifests, events.jsonl,
  metrics/evaluations/system CSVs, summaries, plots) + `mol_metrics`
  (real evaluation metrics) were built and verified on an isolated
  mini-trainer with non-interference guarantees (bit-identical training,
  <2 ms/step); they are preserved in `prototypes/` and deliberately NOT yet
  wired into `ResilientTrainer`.
- **Reasoning.** The operator's mandate: no monitoring may perturb training
  or hide failures; the prototype-first policy (Principle 4) demands
  isolated verification first. Integration is sequenced AFTER the NaN
  evidence capture to avoid changing the observed system mid-investigation.
- **Alternatives rejected.** (a) Integrate immediately (2026-10-28 ask) —
  deferred for the same evidence-first reason as D9; (b) TensorBoard/W&B —
  external service deps in an offline-first project; the JSONL/CSV artifacts
  are diffable, grep-able, and survive crashes; (c) no observability — the
  incident happened precisely because per-component losses weren't logged.
- **Consequences.** The current trainer still lacks per-batch forensics —
  the gap the prototype closes; integration is the top post-restart task.
- **Confidence.** [EVIDENCE] prototype guarantees (prototype_test.py,
  comparison_summary.txt).

### D11 — Evaluation metrics: designed, prototyped, not yet in the training loop

- **Decision.** Real metrics (reaction accuracy, top-k synthon accuracy,
  synthon EM + Tanimoto, validity/uniqueness/novelty/diversity, per-level
  breakdowns) exist as the verified prototype; the trainer keeps its
  aggregate val metrics for now; per-level val loss/accuracy was added with
  the curriculum (2026-10-29) because curriculum validation cannot wait.
- **Reasoning / alternatives rejected.** Loss-only monitoring cannot detect
  degenerate optima (e.g. torsion NLL flat for 5 epochs while CE improved —
  exactly the pre-incident signature); full metric integration is scheduled
  with the observatory (D10) to avoid two separate trainer surgeries.
- **Confidence.** [JUDGMENT]; [ASSUMPTION] that per-level val metrics
  (added now) are statistically usable at 286 val samples (see D8 honesty).

### D12 — Stage-2 RL stability (T4)

- **Decision.** PPO with CPU-isolated rollouts (`PPOTransition` tensors
  `.detach().cpu()`-ed at construction), `minibatch_size=4`,
  `rollout_episodes=8`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  set before torch import, `empty_cache` at PPO epoch boundaries; all RNGs
  seeded at Stage-2 start.
- **Reasoning.** One forward graph per transition held 16 concurrent PaiNN
  graphs → 24 MiB alloc failure at minibatch 16; isolation + smaller
  minibatch + expandable segments removed the fragmentation.
- **Alternatives rejected.** Gradient accumulation in PPO epochs (wrong
  credit assignment); keeping rollouts on GPU (fragmentation); GFlowNet-TB
  as default (implemented + tested, kept as a switchable option pending
  comparative runs).
- **Consequences.** Stage-2 throughput is bounded by CPU rollout speed —
  accepted.
- **Confidence.** [EVIDENCE] episodes 32–47 ran clean after the fix.

### D13 — Testing & collaboration model

- **Decision.** Agent-side: CPU sandbox with vendored wheels, unit +
  integration suites, full-suite regression diffs against a pristine
  baseline (23 pre-existing env-drift failures documented byte-identical).
  Operator-side: runs the ≤5-file test batches on their own PC and returns
  complete outputs; the loop repeats. Every batch has explicit commands,
  expected outputs, and pass criteria.
- **Reasoning.** The operator's GPU/Colab is the only place runtime/GPU
  behavior can be verified; the sandbox can only certify CPU-reachable
  paths. Five files keep the human review load sane and the feedback loop
  tight.
- **Alternatives rejected.** (a) Agent guesses GPU behavior — falsified
  repeatedly (OOM probe, NaN); (b) full-suite runs on the operator side
  every iteration — too slow, wastes their GPU time; targeted batches only.
- **Consequences.** Anything GPU-only (fp16 dynamics, auto-scale, HF
  sync) is [ASSUMPTION] until an operator run confirms it.
- **Confidence.** [EVIDENCE] the process works (batch fixes landed);
  [JUDGMENT] the discipline.

### D14 — Known-questionable calls (self-critique, kept honest)

1. **Finite-loss guard preserves accumulated grads across skips** —
   motivated by "don't waste the window", but if a batch poisons grads
   without inf-detection, the following step applies them (H2/H4 in the
   NaN doc). The prototype shield gates this properly; not yet integrated.
2. **fp16 on T4 after two NaN incidents** — kept for budget reasons, but
   the burden of proof now sits with the planned fp32 A/B; if the fresh
   run NaNs again, flip to fp32 immediately (documented decision rule).
3. **9 levels with only 286 val samples** — level metrics will be noisy;
   accepted for trend-tracking, not epoch-level conclusions.
4. **Curriculum before the scaffold-grouped re-split** — the operator's
   mandate was curriculum now; the leakage-proof re-split (scaffold/
   reaction-family grouped) remains outstanding and should precede any
   publication-grade claims from per-level metrics.
5. **HF remote layout has no run-scoping** — fresh runs progressively
   overwrite remote checkpoints (mitigated by HF commit history + explicit
   `--purge-remote`); a per-run-id remote prefix was considered and
   rejected this iteration as too invasive for the checkpoint path.
6. **`--fresh` once purged remote by default** — a historical design error
   that cost the epoch-31 state; fixed by the archive/opt-in policy (D7).

## 5. What is settled vs open (one-glance)

| Topic | Status |
|---|---|
| Formulation, two-stage design, model family, data sources/splits, fail-closed guards, checkpoint state machine + finiteness refusal, archive-based fresh start, curriculum mechanics, RL isolation, prototype observability | **Settled & implemented** (CPU-verified; GPU verification pending where noted) |
| NaN root cause + the fp16-vs-fp32 policy on T4 | **OPEN** (PERSISTENT_ERRORS_NAN §6) |
| Observability integration, full metric battery in-loop | **OPEN** (designed + prototyped) |
| Leakage-proof scaffold/reaction-family re-split for the full dataset | **OPEN** (prerequisite for the no-curriculum comparison) |
| Curriculum benefit on real data | **OPEN** (A/B not run) |
| GFlowNet-TB / DPO vs PPO empirical ranking | **OPEN** (implemented, unranked) |
