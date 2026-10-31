# 3D-SynTree V2 — Issue & Project Status

> **Source of truth:** this table is synchronized with the GitHub issue
> tracker (repo `Vtheonly/3d-syntree`, branch `3d-synV2`). If this file and
> GitHub disagree, reconcile toward the *verified repository state*
> (#45 §25.1) and record the correction in `WORK_LOG.md`.
>
> Issue **#45** is the permanent operating constitution — always OPEN,
> never counted in the progress denominator.
>
> Last synchronized: **2026-11-01** (session S-2026-11-01-B, #6 ARCH-02).

## Legend

`Pending` → `Ready` (deps verified) → `In Progress` → `Implemented — Awaiting
Verification` → `Verified — Ready to Close` → `Closed`.
`Blocked` = external/hard blocker (reason stated).

## Implementation backlog status

| # | ID | Title (short) | Status | Module | Latest commit | Test evidence | Blocker / next action |
|---|---|---|---|---|---|---|---|
| 5 | ARCH-01 | Purge legacy pipelines; V2 foundation | **Closed** (2026-11-01) | repo-wide | `b4ba696` | Purge verified (13 targets absent, 0 dangling imports in `syntree/`); suite 837 passed / 12 pre-existing failures; README + agent docs landed | — |
| 6 | ARCH-02 | Frozen tensor schemas & PyG contracts | **Closed** (2026-11-01) | `syntree/core/schema.py` | `b5c86122249f0707467dcd0de1b9077c4eeacaf5` | `tests/test_tensor_contracts.py` 43/43; full suite 880 passed / 12 pre-existing; 67-D Franken rejection + fail-closed contract verified | — |
| 7 | MATH-01 | Non-singular PaiNN norms | **Ready (next recommended)** | `syntree/models/equivariant.py` | — | — | #6 done; #21 MODEL-01 now needs only #7 |
| 8 | MATH-02 | Stable von Mises backward (Amos ratio) | Ready (after #5) | `syntree/models/torsion_head.py` | — | — | Note: 3 pre-existing von Mises test failures exist (see §pre-existing) |
| 9 | MATH-03 | Continuous cosine cutoff envelopes | Ready (after #5) | `syntree/models/equivariant.py` | — | — | |
| 10 | MATH-04 | Fail-closed FP32 precision engine | Ready (after #5) | `syntree/utils/hardware.py`, `engine/trainer.py` | — | — | AMP/GradScaler code still present in trainer (legacy, must be removed here) |
| 11 | CHEM-01 | Certified 32-reaction taxonomy | Ready (after #5) | `syntree/chemistry/reactions.py` | — | — | |
| 12 | CHEM-02 | Enamine REAL 50k catalog + pharm3d | Blocked (needs #11, #40 + catalog access) | `syntree/chemistry/catalog.py` | — | — | Requires Enamine REAL subset access |
| 13 | CHEM-03 | Scaffold-constrained MMFF94 relaxation | Blocked (needs #11) | `syntree/chemistry/conformer.py` | — | — | |
| 14 | CHEM-04 | Soft-core LJ & volume floors | Ready (after #5) | `syntree/chemistry/` | — | — | |
| 15 | CHEM-05 | Canonical assembly ordering / decompositions | Blocked (needs #11) | `syntree/data/trajectory.py` | — | — | |
| 16 | DATA-01 | Multi-source preprocessing pipeline | Ready (after #5) | `syntree/data/curation.py`, `scripts/` | — | — | Provenance: `docs/data/RAW_SOURCE_PROVENANCE.yaml` |
| 17 | DATA-02 | MMseqs2 30% clustered splits | Blocked (needs #16) | `syntree/data/splits.py` | — | — | |
| 18 | DATA-03 | Trajectory extraction + exact replay | Blocked (needs #11, #12, #15, #17, #41) | `syntree/data/trajectory.py` | — | — | |
| 19 | DATA-04 | Compressed sharding engine | Blocked (needs #18) | `scripts/shard_and_upload.py`, `syntree/data/hf_loader.py` | — | — | |
| 20 | DATA-05 | Coordinate/dihedral augmentation | Blocked (needs #19) | `syntree/data/` | — | — | Train split only |
| 21 | MODEL-01 | BipartitePaiNN backbone | Blocked (needs #7 only) | `syntree/models/bipartite.py` | — | — | |
| 22 | MODEL-02 | Handle selection head | Blocked (needs #21) | `syntree/models/synthon_head.py` | — | — | |
| 23 | MODEL-03 | Cross-attention decoder | Blocked (needs #9, #21, #22) | `syntree/models/` | — | — | |
| 24 | MODEL-04 | Hierarchical heads + torsion space | Blocked (needs #11, #8, #23) | `syntree/models/` | — | — | |
| 25 | MODEL-05 | SynTreePolicy assembly + audit | Blocked (needs #21–#24, #39 interface) | `syntree/models/policy.py` | — | — | |
| 26 | TRAIN-01 | Pure FP32 T4 trainer | Blocked (needs #10, #19, #20, #25) | `syntree/engine/trainer.py` | — | — | T4 verification requires authorized Colab/Kaggle |
| 27 | TRAIN-02 | Quantile curriculum (Approach C) | Blocked (needs #26) | `syntree/engine/curriculum.py` | — | — | V1 curriculum exists; must be re-verified against V2 contracts |
| 28 | TRAIN-03 | 2D→3D transfer pretraining | Blocked (needs #11, #12, #27) | `syntree/engine/` | — | — | Requires licensed reaction data |
| 29 | FLOW-01 | GFlowNet TB engine | Blocked (needs #13, #25, #26, #38, #39) | `syntree/engine/gflownet.py` | — | — | V1 TB engine exists; #38 semantics must be resolved first |
| 30 | FLOW-02 | Stationary multi-objective reward | Blocked (needs #14, #29) | `syntree/engine/rl.py` (reward) | — | — | Legacy batch-Tanimoto diversity term must be REMOVED here |
| 31 | FLOW-03 | Stage-2 CLI + checkpoint state | Blocked (needs #29, #30) | `main.py` | — | — | |
| 32 | TEST-01 | Suite A capability regression | Ready (after #5) | `tests/` | — | — | |
| 33 | TEST-02 | Suite B invariant tests | Blocked (needs #7, #8, #21, #25, #26, #32) | `tests/` | — | — | |
| 34 | EVAL-01 | Independent AiZynthFinder evaluation | Blocked (needs #29) | `scripts/` | — | — | AiZynthFinder availability unverified |
| 35 | EVAL-02 | Comparative SBDD benchmark | Blocked (needs #34, #44) | `scripts/run_comparative_benchmark.py` | — | — | |
| 36 | OPS-01 | Colab SSH remote workflow | Pending (no code dep; **user authorization gate**) | `docs/`, notebook | — | — | Prepare workflow; do NOT claim any connection |
| 37 | OPS-02 | Hugging Face artifact sync | Blocked (needs #19, #26, #31, #35, #36) — final milestone | `syntree/utils/checkpoint.py` | — | — | |
| 38 | FLOW-04 | MDP state / derivation DAG semantics | Pending (implied deps #15, #24 interfaces) | `docs/architecture/` + engine | — | — | Design-first work item |
| 39 | MODEL-06 | Learned seed/anchor selection | Pending (implied deps #12, #21, #23, #25 interfaces) | `syntree/models/` | — | — | Design-first work item |
| 40 | CHEM-06 | Chemoselectivity filters | Blocked (needs #11) | `syntree/chemistry/` | — | — | |
| 41 | DATA-06 | Hashed inverted synthon index | Blocked (needs #11, #12) | `syntree/chemistry/` | — | — | |
| 42 | CHEM-07 | Sub-bin torsion refinement | Blocked (needs #13, #14, #24) | `syntree/chemistry/` | — | — | |
| 43 | OPS-03 | Measured T4 profiling + session-safe resume | Blocked (needs #26, #36) | `syntree/engine/trainer.py` | — | — | |
| 44 | EVAL-03 | Stock-aligned + independent retrosynthesis tracks | Blocked (needs #12, #34) | `scripts/` | — | — | |

**Progress denominator: 40 issues (#5–#44).** Closed: 2 (#5, #6). #45
excluded by standing rule.

## Pre-existing test failures (documented, NOT caused by V2 work)

The following 12 failures were failing on the branch **before** the #5
purge (baseline 2026-11-01, pre-change run: 957 passed / 22 failed; after
purge: 837 passed / 12 failed — zero new failures, 10 failure-subjects
removed with purged modules):

- `tests/test_core_discovery_checkpoint.py` — 7 failures (checkpoint/discovery
  contract drift)
- `tests/test_core_manifest_checksums.py::TestShardRecordCompat` — 1
- `tests/test_integration_battery.py::TestVonMisesMath::test_high_kappa_concentrates_at_mu` — 1
- `tests/test_math_verification.py` — 2 (log Bessel I₀ accuracy vs scipy;
  torsion sampling concentration)
- `tests/test_models_heads.py::TestContinuousTorsionHead::test_sampling_concentrates` — 1

The von Mises / torsion / Bessel failures are squarely in scope of
**#8 MATH-02** (stable von Mises implementation) — they are expected to be
resolved (or re-specified) there. The discovery/checkpoint failures fall to
**#32 TEST-01** (Suite A re-engineering).

Baseline after #6 (S-2026-11-01-B): **880 passed / 12 failed** — identical
failure set to the pre-#6 baseline (837/12) plus 43 new passing contract
tests; zero new failures.

## Session log pointer

Chronological agent activity: `WORK_LOG.md`. Durable decisions:
`DECISIONS.md`. Verified graph: `DEPENDENCIES.md`. Resumption instructions:
`HANDOFF.md`.
