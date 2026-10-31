# 3D-SynTree V2 — Durable Decision Log

> One decision per entry. Never silently change a decision: supersede it
> with a new entry and update the affected issues/docs. V1-era decisions
> live in `docs/legacy/DECISIONS.md` (historical, read-only); where a V1
> decision is still binding, it is re-adopted below with its V1 reference.

---

## D-2-001 — 2026-11-01 — V2 is a clean reconstruction on `3d-synV2`

- **Problem:** The repository history contains the full V1 system
  (fp16+GradScaler training, PPO/DPO Stage-2, synthetic CrossDockedDataset
  smoke path, Kaggle offline loader, VRAM-probing auto-scale) which #45
  §24 declares superseded by a unified V2 re-engineering.
- **Decision:** Treat `3d-synV2` (created at commit `44edb8b`, identical to
  `main` at branch time) as the V2 greenfield. The V1 code is *reference
  material*, not the V2 baseline: each capability is re-adopted only
  through its #5–#44 issue with V2 contracts.
- **Alternatives considered:** incremental patching of V1 (rejected: #45
  §24.1 explicitly forbids); keeping V1 modules "temporarily" active
  (rejected: no old/new side-by-side, #45 §24.3).
- **Evidence:** branch comparison `git log origin/main..3d-synV2` = empty
  at session start; issue #45 §24.
- **Affected:** all issues; README rewrite (this session).
- **V1 reference:** docs/legacy/DECISIONS.md global framing.

## D-2-002 — 2026-11-01 — Scope discipline for #5 ARCH-01

- **Problem:** Issue #5's literal scope is the purge list + "no dangling
  imports", while #45 §4 also demands the V2 layout and a first-class
  README as part of starting ARCH-01.
- **Decision:** #5 delivers: (a) the exact purge list, (b) removal of
  every dangling reference *necessary for the tree to import and the suite
  to run* (PPO/DPO dispatch in `main.py`, auto-scale call sites, kaggle/
  local backends, tests of purged subjects), (c) the V2 documentation
  skeleton (`docs/agent/*`, `docs/architecture/`, legacy archive,
  provenance archive), (d) the rewritten README. Anything deeper
  (reward re-engineering, FP32 engine, schema freezes) belongs to its own
  issue and is only *referenced* as debt.
- **Alternatives considered:** minimal literal purge leaving `main.py`
  importing deleted modules (rejected: broken CLI violates "clean
  deletions without dangling references"); expanding into #10/#30 work
  now (rejected: scope creep, #45 §22 "smallest coherent change").
- **Evidence:** acceptance-criteria greps (see WORK_LOG S-2026-11-01-A);
  837/849 test suite result with zero new failures vs. baseline.

## D-2-003 — 2026-11-01 — Interim data backend set = {huggingface, trajectory_pt}

- **Problem:** Purging `CrossDockedDataset` (local backend) and the Kaggle
  offline loaders leaves the trainer without its former default data path.
- **Decision:** `SUPPORTED_DATA_BACKENDS = ("huggingface", "trajectory_pt")`
  (both in `syntree/engine/trainer.py` and `syntree/core/registry.py`);
  default backend is `huggingface`; an unknown/legacy backend fails closed
  with a message pointing at #16 DATA-01 / #19 DATA-04. Trainer smoke tests
  use the `trajectory_pt` backend fed by
  `RetrosyntheticTrajectoryBuilder` states (real featurization path), via
  the session-scoped `smoke_trajectory_path` conftest fixture — replacing
  the purged synthetic CrossDocked smoke dataset.
- **Alternatives considered:** rebuilding a local loader now (rejected:
  that *is* #16 DATA-01); stubbing a fake dataset (rejected: #45 §11 "do
  not substitute synthetic point clouds / mock labels" for production; the
  trajectory fixture uses the real builder on real molecules).
- **Evidence:** `tests/test_engine_trainer.py` 23/23,
  `test_curriculum_trainer_integration.py` 10/10, `test_fresh_start.py`,
  `test_pipeline_integration.py` 14/14 green on the new fixture.
- **Affected:** #16 (will extend/replace), #26 (consumes), tests.

## D-2-004 — 2026-11-01 — Stage-2 algorithm set = GFlowNet-TB only

- **Problem:** `--mode rl` previously dispatched ppo | gflownet | dpo; PPO
  and DPO are purged by #5, and #45 §12.3 states PPO/DPO are not
  production alternatives in V2.
- **Decision:** `main.py --mode rl` accepts only
  `reinforcement_learning.algorithm = "gflownet"` (anything else exits
  code 2 with an explanatory message); `configs/rl_colab_12h.json` sets
  `algorithm: "gflownet"` and drops PPO-only keys (`ppo_epochs`,
  `clip_epsilon`, `gamma`, `gae_lambda`, `value_coef`, `entropy_coef`,
  `minibatch_size`). The rollout buffering trigger uses the pending-episode
  count (the GAE RolloutBuffer was PPO machinery).
- **Note:** the retained V1 `GFlowNetTrainer`/`ThreeDReward` code is
  *interim*: #38 FLOW-04 must resolve MDP/derivation-DAG semantics (the
  current engine's zero-backward-term assumption is only valid under the
  unique-parent derivation-tree view and must be re-proven), and #30
  FLOW-02 must remove the legacy **batch-relative Tanimoto diversity
  reward term** (`diversity_weight`) which violates #45 §12.4
  stationarity. Both are flagged in module docstrings.
- **Evidence:** `tests/test_engine_gflownet.py` (renamed from
  `test_engine_gflownet_dpo.py`; DPO classes removed, GFlowNet +
  checkpoint-plumbing + CLI tests retained, invalid-algorithm test now
  asserts the V2 rejection).
- **Affected:** #29, #30, #31, #38.

## D-2-005 — 2026-11-01 — Unsupported-hardware artifacts removed

- **Problem:** `configs/train_h100_full.json` (bf16, H100) and
  `configs/train_kaggle_96gb.json` (kaggle_offline backend) configure
  targets forbidden by #45 §5 (supported GPUs: Colab T4 / Kaggle T4 only)
  and/or purged backends; `CANONICAL_MODELS` in `syntree/core/registry.py`
  carried `3d-syntree-large` (V100/A10G) and `3d-syntree-h100` schemas.
- **Decision:** Delete both configs and the two non-T4 model schemas;
  keep `3d-syntree-base` (T4-portable profile). Canonical T4 training
  profiles will be defined by #26 TRAIN-01 and measured by #43 OPS-03.
- **Alternatives considered:** keeping them "for reference" (rejected:
  #45 §5 "Do not add configurations or documentation for other GPU
  targets").
- **Evidence:** #45 §5; configs deleted in this session's diff.
- **Affected:** #26, #43.

## D-2-006 — 2026-11-01 — Runtime OOM-skip guard retained (interim)

- **Problem:** Issue #5 purges `scale_model_config`/`autotune_batch_size`
  (VRAM-probing auto-scale). The trainer also contains a *runtime* OOM
  skip + adaptive epoch-end micro-batch halving guard that is related but
  not named by the issue.
- **Decision:** Remove the probing subsystem (`_autoscale_batch`,
  `_heaviest_probe_pool`, the `main.py` auto-scale branch, config
  `auto_scale` blocks) but **retain** the runtime OOM-skip/adaptive-rescale
  guard for now: it is failure resilience, not architecture mutation, and
  its fate is decided by #26 TRAIN-01 / #43 OPS-03 under measured T4
  profiles. A leftover `training.auto_scale` block is inert and only
  warned about.
- **Evidence:** trainer diff; `test_gpu_stability_fixes.py`,
  `test_oom_nan_resilience.py` green.
- **Affected:** #26, #43.

## D-2-007 — 2026-11-01 — Data-source provenance preserved outside configs

- **Problem:** `configs/download.yaml` (config of the purged
  `download/download_mode.py`) contains the only complete verified list of
  upstream raw-source URLs + checksums + verification dates.
- **Decision:** Archive that content as `docs/data/RAW_SOURCE_PROVENANCE.yaml`
  (header documents its provenance and freeze semantics), then delete the
  config. #16 DATA-01 consumes it as its source-of-record.
- **Evidence:** file diff; `docs/legacy/README.md` cross-reference.
- **Affected:** #16.

## D-2-008 — 2026-11-01 — Test suite strategy for purged capabilities

- **Problem:** ~30 test files referenced purged modules/symbols.
- **Decision:** Three-tier handling, per #45 §24.3 ("rewrite tests where
  necessary so they enforce the approved V2 contracts"):
  1. **Delete** tests whose *subject* is purged functionality
     (craft/download CLI, CrossDocked loader, Kaggle offline, PPO/DPO
     objectives, autoscale, builder script).
  2. **Rewrite** tests whose subject is *retained* but whose fixture used
     purged plumbing (policy/model tests now build synthetic PyG states
     directly; pipeline-integration builds shards via
     `RetrosyntheticTrajectoryBuilder` + `write_shards`; trainer smoke
     tests use the trajectory_pt backend).
  3. **Keep** everything else untouched.
  Every deleted capability gets an explicit successor note (STATUS.md /
  issue comments) so coverage returns with the V2 implementation.
- **Evidence:** baseline-vs-final failure comparison (identical failure
  set modulo removed subjects; zero new failures).
- **Affected:** #32 (Suite A re-engineering must restore builder-level
  coverage when #16 lands).
