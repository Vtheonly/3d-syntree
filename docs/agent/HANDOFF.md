# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-01, session S-2026-11-01-C (issues #6 + #7).

## 1. Continuation instruction (unambiguous)

```
Next session: implement #21 MODEL-01 (Bipartite SE(3) Equivariant Backbone)
— both hard deps are closed (#6 schema, #7 norms). Last verified commit:
bbd8e20d6e93d4f33b14776511f069ce2ba03939 (pushed to origin/3d-synV2,
verified). #5, #6 and #7 are implemented, locally verified, and CLOSED —
do NOT redo the purge, the schema, or the non-singular norms. Before
coding, re-read issue #21 in full and confirm branch HEAD matches the
recorded commit (someone may have advanced it). Parallel-safe alternatives
if #21 is taken: #8 MATH-02 (owns the torsion-head torch.norm debt + the 4
pre-existing von Mises failures), #9 MATH-03, #10 MATH-04 (fp16 clamps in
equivariant.py are documented interim for #10), #11 CHEM-01, #14 CHEM-04,
#16 DATA-01, #32 TEST-01.
```

## 2. Current project status

- Branch `3d-synV2`. Closed issues: **#5 ARCH-01**, **#6 ARCH-02**,
  **#7 MATH-01**. All remaining backlog issues #8–#44 OPEN.
- Test baseline (local CPU): **906 passed / 12 failed**; the 12 are the
  documented pre-existing failures (7 `test_core_discovery_checkpoint`,
  1 `test_core_manifest_checksums`, 4 von Mises/torsion) owned by
  **#8 MATH-02** and **#32 TEST-01** — do not "fix" them inside unrelated
  issues. Any NEW failure is yours to fix before proceeding.
- No GPU/T4 execution for V2 has happened. No training. No HF sync.
  Any claim otherwise would be false.

## 3. What the next agent must NOT redo

- The legacy purge (#5); the frozen tensor schema (#6):
  `syntree/core/schema.py` is the single state contract — extend it, do
  not fork it (13-field validation, canonical collator, 67-D rejection;
  43 tests in `tests/test_tensor_contracts.py`); the non-singular norm
  primitives (#7): `safe_norm`/`safe_unit_vector` in
  `syntree/models/equivariant.py` with `SAFE_NORM_EPS = 1e-5`, already
  routed through PaiNNInteraction/PaiNNMixing/PaiNNLayer/
  BatchedPaiNNLayer/bipartite.py (26 tests in
  `tests/test_mathematical_singularities.py`).
- README/architecture/agent-docs structure — update in place.

## 4. #21 MODEL-01 starting notes (from this session's reading of the code)

- `syntree/models/bipartite.py` (BipartitePaiNN) already implements the
  joint pocket+ligand union graph with pocket–ligand cross edges, learned
  node types, handle-node flags, pH-7.4 charges, gradient checkpointing,
  and now consumes the #6 schema states and #7 safe norms. #21 must
  verify it against its full acceptance criteria (formal equivariance
  proofs, contract checks, anything named in its issue body) and
  re-engineer whatever does not meet the V2 contract.
- Relevant existing tests: `tests/test_models_equivariant.py`,
  `tests/test_batch1_models.py`, `tests/test_models_architecture.py`,
  `tests/test_mathematical_singularities.py` (degenerate-geometry
  section covers BipartitePaiNN end-to-end).
- The 4 pre-existing von Mises/torsion failures belong to #8, not #21.

## 5. Known debts carried forward (with owners)

| Debt | Owner issue |
|---|---|
| Legacy batch-Tanimoto diversity term in `ThreeDReward` (+ `diversity_weight: 0.2` in `configs/rl_colab_12h.json`) violates the stationary-reward contract | #30 FLOW-02 |
| AMP/GradScaler/`mixed_precision` machinery still in trainer/hardware; `finite_act_clamp`/`FP16_ACT_CLAMP` in equivariant.py documented interim | #10 MATH-04 |
| von Mises/log-Bessel + torsion sampling test failures (pre-existing) | #8 MATH-02 |
| `torsion_head.py::_pool_vector_stats` still uses raw `torch.norm` (delegated from #7) | #8 MATH-02 |
| core discovery/checkpoint test failures (pre-existing) | #32 TEST-01 |
| Zero-backward-term TB assumption + heuristic seed in `GFlowNetTrainer` | #38 FLOW-04 / #39 MODEL-06 |
| V1 curriculum engine present but not re-verified against V2 contracts | #27 TRAIN-02 |
| `SUPPORTED_DATA_BACKENDS` interim set `{huggingface, trajectory_pt}` | #16 DATA-01 / #19 DATA-04 |
| Unstamped legacy shards: validator accepts absent `schema_version` (interim) until #19 re-stamps | #19 DATA-04 |

## 6. Remote runtime / checkpoint state

None — no Colab/Kaggle session was opened (user authorization not
requested; #36 workflow will prepare instructions when reached). No V2
checkpoints exist. The V1-era HF datasets
(`JJKK1212/3d-syntree-multidataset`) remain unverified for V2: any V2
training consumption must first pass the #6 schema contract — legacy
67-D shards now fail closed at collation (intended; #19 re-stamps or
reprocesses).

## 7. Session mechanics for the next agent

1. `git clone --branch 3d-synV2` (or fetch); confirm HEAD == the commit in
   §1. **Watch for a push race** (session S-B cloned 2 minutes before the
   previous agent's push landed; `git fetch` + compare with issue comments
   before assuming you are current).
2. Install deps: `pip install torch --index-url
   https://download.pytorch.org/whl/cpu`, then vendored
   `wheels/torch_geometric-*.whl`, `wheels/rdkit-*.whl`, then
   `pyarrow pandas scipy huggingface-hub tqdm h5py pyyaml pytest`.
3. Baseline the suite: `python -m pytest tests/ -q --tb=no` — expect
   **906 passed / 12 failed** (list in STATUS.md; any NEW failure is
   yours).
4. Work the issue per the #45 §18 protocol; end with the mandatory git
   diff review, commit+push, issue update, and refreshed handoff.
