# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-01, session S-2026-11-01-A.

## 1. Continuation instruction (unambiguous)

```
Next session: implement #6 ARCH-02 (Establish Frozen State Tensor Schemas &
PyG Batch Contracts). Last verified commit: <SHA — filled after push>.
#5 ARCH-01 is implemented, verified locally, and closed. Do NOT redo the
purge. #7 MATH-01 is the independent parallel-safe second task. Before
coding #6, re-read issue #6 in full and re-check the branch HEAD matches
the commit above (someone may have advanced it).
```

## 2. Current project status

- Branch `3d-synV2`; #5 ARCH-01 delivered this session (purge + V2 docs
  foundation + README rewrite). All remaining backlog issues #6–#44 are
  OPEN; none started.
- Test suite state (local CPU): **837 passed / 12 failed**, all 12
  pre-existing before V2 work (failure IDs listed in STATUS.md; expected to
  be resolved by #8 MATH-02 and #32 TEST-01 — do not "fix" them inside
  unrelated issues without recording the decision).
- No GPU/T4 execution has happened for V2. No HF artifacts were synced.
  No training has been run on V2. Any claim otherwise would be false.

## 3. What the next agent must NOT redo

- The purge (23 files) and all dangling-reference fixes — complete and
  verified (grep evidence in WORK_LOG §4).
- `docs/agent/*`, `docs/architecture/ARCHITECTURE.md`, `docs/legacy/`
  archive, `docs/data/RAW_SOURCE_PROVENANCE.yaml` — established.
- README.md — rewritten as V2-first; extend it, don't restart it.
- `tests/conftest.py::smoke_trajectory_path` fixture and the
  `trajectory_pt`-backend test wiring — working; reuse it.

## 4. #6 ARCH-02 starting notes (from this session's reading of the code)

- Issue #6 demands the frozen tensor schema contract; the V1 "Franken-tensor"
  67-dim legacy path still exists as a tolerated branch in
  `syntree/models/policy.py` (`handle_features` 64 vs legacy 67) and in
  `syntree/data/featurizer.py` (`HANDLE_FEATURE_DIM`). #6 should freeze the
  64-dim contract, make schema/version checks fail closed, and delete the
  legacy tolerance.
- PyG batch contract today: `Batch.from_data_list(..., follow_batch=
  ["pocket_pos", "ligand_pos"])` with `pocket_batch`/`ligand_batch`
  fallbacks resolved inside `SynTreePolicy.forward` — that resolution logic
  is exactly the implicit guessing #45 §7.1 forbids; move it into an
  explicit, validated collate contract.
- Relevant existing tests to evolve: `tests/test_models_policy.py`
  (synthetic PyG fixture), `tests/test_data_featurizer.py`,
  `tests/conftest.py`.

## 5. Known debts carried forward (with owners)

| Debt | Owner issue |
|---|---|
| Legacy batch-Tanimoto diversity term in `ThreeDReward` (+ `diversity_weight: 0.2` in `configs/rl_colab_12h.json`) violates the stationary-reward contract | #30 FLOW-02 |
| AMP/GradScaler/`mixed_precision` machinery still in trainer/hardware | #10 MATH-04 |
| von Mises/log-Bessel + torsion sampling test failures (pre-existing) | #8 MATH-02 |
| core discovery/checkpoint test failures (pre-existing) | #32 TEST-01 |
| Zero-backward-term TB assumption + heuristic seed in `GFlowNetTrainer` | #38 FLOW-04 / #39 MODEL-06 |
| V1 curriculum engine present but not re-verified against V2 contracts | #27 TRAIN-02 |
| `SUPPORTED_DATA_BACKENDS` interim set `{huggingface, trajectory_pt}` | #16 DATA-01 / #19 DATA-04 |

## 6. Remote runtime / checkpoint state

None — no Colab/Kaggle session was opened (user authorization not
requested this session; #36 workflow will prepare instructions when
reached). No checkpoints exist for V2. The Hugging Face datasets
referenced by configs (`JJKK1212/3d-syntree-multidataset`) are V1-era
artifacts; their reuse must be checksum/schema-verified before any V2
training consumes them (#45 §15).

## 7. Session mechanics for the next agent

1. `git clone --branch 3d-synV2` (or fetch), confirm HEAD == the SHA in §1.
2. Install deps: `pip install torch --index-url
   https://download.pytorch.org/whl/cpu`, then `rdkit`, `torch_geometric`,
   `pyarrow`, `numpy`, `scipy`, `pandas` (or the vendored `wheels/`).
3. Baseline the suite: `python -m pytest tests/ -q` — expect 837 passed /
   12 failed (the exact failure list is in STATUS.md; any NEW failure is
   yours to fix before proceeding).
4. Work the issue per the #45 §18 protocol; end with the mandatory git
   diff review, commit+push, issue update, and refreshed handoff.
