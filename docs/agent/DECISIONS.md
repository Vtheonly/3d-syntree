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

## D-2-009 — 2026-11-01 — Schema module location: `syntree/core/schema.py`

- **Problem:** Issue #6 body specifies `syntree/core/schema.py` for the
  frozen state contract, while `docs/architecture/ARCHITECTURE.md` §2
  (written during #5) listed `syntree/data/schema.py` — an internal
  requirement conflict (#45 §0 requires documenting, not guessing).
- **Decision:** Follow the **issue body** (`syntree/core/schema.py`) and
  update the architecture doc to match. Rationale: the implementation
  issue is the binding task specification; the schema is consumed by the
  data, model and engine layers alike (a `core/` concern, beside
  `registry.py`/`masking.py`), not a data-layer-only module.
- **Alternatives considered:** `syntree/data/schema.py` (rejected: both
  consumers and the issue text point to `core/`); a new top-level
  `syntree/schema/` package (rejected: over-structuring for one module).
- **Evidence:** issue #6 §2 ("In `syntree/core/schema.py`, establish the
  canonical PyG batch schema"); ARCHITECTURE.md §2 row updated this
  session.
- **Affected:** #6, ARCHITECTURE.md.

## D-2-010 — 2026-11-01 — Strictness rules of the V2 state contract (#6)

- **Problem:** The V1 policy tolerated a zoo of malformed inputs (1-D
  reshape guessing by divisibility, 67-dim transparent splitting,
  `pocket_pos_batch` alias fallbacks, silent zeros defaults for
  `handle_pos`/`global_features`/`stop_mask`), violating #45 §7.1
  ("no implicit shape guessing or Franken-tensor compatibility parser").
- **Decision (binding contract rules, `syntree/core/schema.py`):**
  1. All 13 canonical fields are **required** on every state; the ligand
     sub-graph may be empty (`[0,3]`/`[0]`) but must exist explicitly
     (the t=0 state).
  2. `handle_features` is exactly `[B, 64]` float32 at the policy
     boundary; last-dim 67 raises
     `ValueError("Legacy 67D Franken-tensors are deprecated ...")`
     (exact prefix pinned by test) — the rejection takes precedence over
     missing-field diagnostics so legacy states get the actionable
     message.
  3. Only the **collator** performs documented normalisations: 0-dim
     tensors → `[1]`, 1-D `[64]` handle features → `[1, 64]` (explicit
     single-graph rule; 1-D `[67]` → Franken error). Nothing else
     reshapes.
  4. `collate_state_batch` is the **single** batching path: PyG
     `follow_batch=['pocket_pos','ligand_pos']` → renamed to canonical
     `pocket_batch`/`ligand_batch`, aliases deleted. Sample-carried
     batch vectors must be the implied zeros and are stripped (PyG
     double-increments `*_batch` keys and crashes on empty
     `ligand_batch`); pre-batched objects are rejected.
  5. `schema_version` is stamped by producers (`stamp_schema_version`,
     tensor form so it survives collation to `[B]`) and checked by the
     validator; unstamped states still pass full field validation
     (interim until #19 re-stamps shards).
  6. `handle_nodes` stays optional context (not in the 13-field core).
  7. `stop_mask` is `[B]` float32 (env and tests updated; 0-dim scalars
     were a legacy habit).
- **Alternatives considered:** validator-side implied-zeros for missing
  `pocket_batch` at B==1 (rejected: two-mode contract invites drift);
  keeping the 67→64+3 split in a load shim (rejected: #6 explicitly
  demands a hard error).
- **Evidence:** `tests/test_tensor_contracts.py` (43 tests incl. both #6
  acceptance criteria); full suite 880 passed / 12 pre-existing failures
  (unchanged failure set vs. baseline).
- **Affected:** #6, #21–#25 (model consumers), #19 (shard stamping),
  #32/#33.

## D-2-011 — 2026-11-01 — #7 scope: bipartite.py norm included, torsion_head delegated

- **Problem:** Issue #7 names `syntree/models/equivariant.py` (PaiNNInteraction/
  PaiNNMixing), but the same PaiNN geometry path computes `torch.norm(edge_vec)`
  in `bipartite.py`, and `torsion_head.py::_pool_vector_stats` also has a raw
  `torch.norm` on vector features.
- **Decision:** Route `bipartite.py`'s edge distance through `safe_norm` in #7
  (it is the direct feed of the interaction's edge geometry — the issue's
  motivating formula `edge_vec/(edge_dist+1e-8)` describes exactly this path),
  and **delegate** the torsion-head norm to #8 MATH-02 (that module is owned
  by #8's von Mises/torsion rework; cross-editing it here risks conflicting
  with #8's rewrite). Recorded in the #7 issue comment and STATUS debt table.
- **Also removed in #7:** the `edge_dist` parameter of `PaiNNInteraction` and
  `BatchedPaiNNLayer` (direction/distance now derive non-singularly from
  `edge_vec` inside the interaction — a dead input invites contract drift);
  `PaiNNMixing.epsilon` semantics changed from an additive bias inside the
  norm (`torch.norm(v + 1e-8)`) to the smoothing radius of `safe_norm`
  (default aligned to 1e-5).
- **Evidence:** `tests/test_mathematical_singularities.py` 26/26; suite
  906/12 (identical pre-existing failures).
- **Affected:** #7, #8 (inherits the torsion-norm debt), #21, #32.
