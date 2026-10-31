# 3D-SynTree V2 — Agent Work Log

> Chronological, append-only record of agent engineering sessions on
> `3d-synV2`. Each entry separates **attempted / implemented / tested
> locally / tested on T4 / scientifically validated** per #45 §19.1. No
> secrets are ever recorded here. V1-era history: `docs/legacy/` (esp.
> `docs/legacy/HANDOFF_2026-10-29.md`).

---

## S-2026-11-01-A — #5 ARCH-01: legacy purge + V2 foundation

- **Date/time:** 2026-11-01 (Asia/Shanghai), agent: Z.ai (GLM), autonomous
  session triggered by the standing #45 entry prompt.
- **Repository/branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `44edb8b` (= `origin/main` HEAD; branch had **zero**
  V2 commits — confirmed `git log origin/main..3d-synV2` empty).
- **Working tree at start:** clean (`git status --short` empty).

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full (71,488 chars) from the GitHub API.
- Fetched all issues (#5–#44 + #45): **all 40 backlog issues OPEN** — no V2
  issue work had started. Extracted every `Depends on`/`Blocks` edge
  (corrected for the `- **Depends on:**` bullet format; added-issue edges
  #38–#44 derived from reverse `Blocks` lists) → `docs/agent/DEPENDENCIES.md`.
- Cloned the repo on `3d-synV2`; inspected the tree (171 tracked files,
  V1 architecture: `craft/`, `download/`, `build_seed/`, `prototypes/`,
  `core/`, `syntree/`, `main.py`, 4 training configs, notebook, 58 test
  files).
- Read prior agent records: `docs/HANDOFF_2026-10-29.md` (V1 session: fp16
  NaN forensics, curriculum verification, fresh-restart machinery),
  `docs/DECISIONS.md` (V1 D1–D14), `docs/CURRICULUM.md`,
  `docs/PERSISTENT_ERRORS_NAN.md`, `docs/ENGINEERING_AUDIT_REPORT.md`.
  Conclusion: those sessions were V1 maintenance; **the V2 re-engineering
  had not begun.** Continuation point per #45 §25.2: **#5 ARCH-01**.
- Environment prepared (dev sandbox, CPU): `torch 2.15.0+cpu`,
  `torch_geometric 2.8.1`, `rdkit 2026.09.1`, `numpy 2.1.3`, `pyarrow
  26.0.0` — matches the V1 pre-flight environment exactly.

### 2. Baseline (before any change)

`python -m pytest tests/ -q` (excluding 3 slow CLI/e2e files):
**957 passed / 22 failed** in 81.5 s. All 22 failures recorded verbatim
(see STATUS.md §"Pre-existing test failures"); the V1 handoff had already
documented several as environment/version drift.

### 3. Implemented (this session)

**3a. Purge (issue #5 §2 — all targets removed; `git rm`):**
`craft/craft_mode.py`, `download/download_mode.py`, `download/__init__.py`
(`craft/__init__.py` did not exist in the tree — nothing to delete),
`scripts/build_full_dataset.py`, `scripts/build_full_dataset_fast.py`,
`scripts/prepare_multidataset.py`, `build_seed/` (5 files),
`prototypes/train_observatory/` (8 files), `syntree/engine/dpo.py`,
`syntree/models/pocket_encoder.py`, `syntree/data/crossdocked.py`,
`syntree/data/kaggle_loader.py` — 23 files total.

**3b. In-file symbol removals (issue #5 §2 comments):**
- `syntree/engine/rl.py`: deleted `PPOFineTuner`, `RolloutBuffer`,
  `PPOTransition`; `RolloutTransition` made a standalone dataclass;
  `joint_log_prob_and_entropy` type hint updated; docstring rewritten
  (Stage 2 = GFlowNet-TB; legacy diversity-reward debt flagged for #30).
- `syntree/utils/hardware.py`: deleted `scale_model_config`,
  `autotune_batch_size`, `MODEL_TIERS` (kept `free_vram_bytes` telemetry).

**3c. Dangling-reference cleanup:**
- `syntree/engine/trainer.py`: removed CrossDocked/kaggle imports; backend
  registry → `("huggingface", "trajectory_pt")` (default `huggingface`);
  removed the `kaggle_offline` branch and the local-CrossDocked
  fall-through (now a fail-closed `ValueError` pointing at #16/#19);
  removed `_autoscale_batch` + `_heaviest_probe_pool` and their call site
  (kept the runtime OOM-skip guard — D-2-006); guard messages no longer
  reference the deleted builder script.
- `main.py`: `--mode rl` is GFlowNet-only (ppo/dpo → exit 2 with an
  explanatory V2 message; `algorithm` default `gflownet`); removed the
  auto-scale branch (leftover `training.auto_scale` now inert + warned);
  removed `kaggle_offline` config validation; default backend
  `huggingface` in `validate_config`; fixed a pre-existing display-artifact
  assumption (no code change needed — "ain]" strings in tests are genuine
  substrings of "[main]").
- `syntree/core/registry.py`: backend registry synced; unsupported-HW
  model schemas (`3d-syntree-large`, `3d-syntree-h100`) removed (D-2-005).
- `syntree/data/__init__.py`: CrossDockedDataset export removed.
- `scripts/verify_setup.py`: builds its synthetic PyG smoke state directly
  (no CrossDockedDataset).
- `scripts/download_assets.py` + `syntree/engine/trainer.py` guard
  messages: point at V2 docs instead of the deleted builder.
- Docstring-only: `syntree/engine/gflownet.py`, `syntree/chemistry/reactions.py`.

**3d. Configs:** deleted `configs/craft.yaml`, `configs/download.yaml`
(provenance archived → `docs/data/RAW_SOURCE_PROVENANCE.yaml`, D-2-007),
`configs/train_h100_full.json`, `configs/train_kaggle_96gb.json`
(D-2-005); updated `configs/default_config.json` (backend
`huggingface`, `auto_scale` removed), `configs/train_colab_12h.json`
(`auto_scale` removed), `configs/rl_colab_12h.json` (`algorithm:
"gflownet"`, PPO-only keys removed).

**3e. Notebook:** `notebooks/run_3d_syntree.ipynb` cell 1/6/9 comments and
  titles de-PPO'd; CELL 4.5 build gate now fails closed with a #16 DATA-01
  pointer instead of invoking the purged builder.

**3f. Tests (three-tier strategy, D-2-008):**
- Deleted (subject purged): `test_autoscale.py`, `test_data_crossdocked.py`,
  `test_offline_kaggle.py`, `test_offline_e2e.py`,
  `test_download_craft_smoke.py`, `test_build_full_dataset.py`,
  `test_rl_objective.py`, `test_rl_rollout.py`.
- Renamed/edited: `test_engine_gflownet_dpo.py` → `test_engine_gflownet.py`
  (DPO classes removed; GFlowNet + checkpoint plumbing + Stage-2 CLI tests
  retained; invalid-algorithm test asserts the V2 rejection);
  `test_batch5_engine_eval.py` (RolloutBuffer/DPO tests removed);
  `test_batch2_core_policy.py` (autotune test removed; valid-config fixture
  updated for the huggingface backend); `test_fp16_safety.py`
  (CrossDocked-based synthetic-supervision class removed);
  `test_models_policy.py` (fresh-clone synthetic PyG fixture with
  empty-ligand pre-growth state); `test_models_heads.py` /
  `test_chemistry_biophysics.py` (PocketEncoder classes removed);
  `test_data_fragmenter.py` / `test_batch4_data_pipeline.py` /
  `test_integration_battery.py` (CrossDocked tests removed);
  `test_real_data_guard.py` (local-synthetic tests removed; trajectory
  backend test sets `backend="trajectory_pt"` explicitly; guard message
  assertion updated); `test_pipeline_integration.py` (fixture rewritten:
  trajectory builder → `write_shards` → FakeHub; trajectory-grouped
  80/10/10 split to keep the no-leak invariant green);
  `test_cli_smoke.py` (BuilderCLI removed; missing-catalog config fixed);
  `test_main_cli.py` (missing-catalog config fixed);
  `test_notebook_contract.py` (CELL 9 marker; build-gate contract = fail
  closed until #16); `test_gpu_stability_fixes.py` (T4-safety assertion →
  gflownet/rollout knobs); `test_curriculum_trainer_integration.py`
  (weight-sum tolerance 1e-6 → 1e-3 with rationale: recorded weights are
  rounded to 4 decimals by the curriculum history serializer).
- New shared fixture: `tests/conftest.py::smoke_trajectory_path` —
  session-scoped real trajectory `.pt` (12×4 real amide/ester products →
  ~96 states) replacing the purged synthetic smoke dataset; wired into
  `tiny_config_with_assets` via the `trajectory_pt` backend (D-2-003).

**3g. V2 documentation foundation:** `docs/agent/{STATUS, WORK_LOG,
DECISIONS, DEPENDENCIES, HANDOFF}.md`; `docs/architecture/ARCHITECTURE.md`;
`docs/legacy/` archive (+ README cross-index); `docs/data/
RAW_SOURCE_PROVENANCE.yaml`; README rewritten as a first-class V2
description (see §3h).

**3h. README.md** rewritten end-to-end as the V2 system description
(purpose, scientific formulation, MDP/DAG/GFlowNet distinction,
architecture, math foundations, reaction taxonomy, catalog, datasets,
leakage prevention, training stages, hardware policy, install/run,
evaluation, reproducibility, repo structure, limitations, honest
implementation-status table). No claims of unimplemented features.

### 4. Verification (all local CPU — no GPU/T4 claims)

- Acceptance greps: all 13 purge targets absent; **0** references in
  `syntree/` to `craft` / `download_mode` / `dpo` / `pocket_encoder` /
  `crossdocked` / `kaggle_loader`; **0** references anywhere to
  `PPOFineTuner` / `RolloutBuffer` / `PPOTransition` /
  `scale_model_config` / `autotune_batch_size` / `MODEL_TIERS`.
- Full suite after purge: **837 passed / 12 failed** (177.6 s, all files
  including CLI/e2e). Failure-for-failure comparison against the recorded
  baseline: **zero new failures**; 10 baseline failures disappeared because
  their subjects were purged; the remaining 12 are byte-identical
  pre-existing failures (7 discovery/checkpoint, 1 shard-record compat,
  4 von Mises/torsion-sampling — squarely #8 MATH-02 / #32 TEST-01 scope).
- Import checks: `syntree.*` packages and `main.py` import cleanly.
- NOT done here (honest scope): any GPU/T4 execution, HF artifact sync,
  reaction-taxonomy certification, catalog curation — those belong to
  their issues and require authorization/data.

### 5. Git record

- Single coherent commit on `3d-synV2` (message: purge + V2 foundation).
  SHA recorded in the issue #5 comment and in HANDOFF.md after push.
- `git diff --check` clean (no whitespace errors); staged set inspected
  with `git diff --stat` / `--unified=80` before committing.

### 6. GitHub updates

- Issue #5 updated with the completion comment (evidence + commit SHA) and
  closed after verification.
- Issues #6/#7 (next dependency-ready tasks) referenced from the handoff;
  no other issue state changed.

### 7. Next actions (for the next session)

1. `Next session: resume #6 ARCH-02` (see `HANDOFF.md` for the exact
   continuation contract).
2. Parallel-safe second task: #7 MATH-01.
3. #36 OPS-01 workflow *preparation* is allowed but the Colab SSH
   connection requires explicit user authentication/authorization.
4. Carry-forward debts (tracked in STATUS.md): von Mises test failures
   (#8), discovery/checkpoint test failures (#32), legacy
   batch-Tanimoto diversity reward (#30), AMP/GradScaler machinery still
   in trainer (#10).
