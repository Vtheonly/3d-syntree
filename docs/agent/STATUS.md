# 3D-SynTree V2 — Issue & Project Status

> **Source of truth:** this table is synchronized with the GitHub issue
> tracker (repo `Vtheonly/3d-syntree`, branch `3d-synV2`). If this file and
> GitHub disagree, reconcile toward the *verified repository state*
> (#45 §25.1) and record the correction in `WORK_LOG.md`.
>
> Issue **#45** is the permanent operating constitution — always OPEN,
> never counted in the progress denominator.
>
> **Clean-room note (D-V2-001):** all pre-reset issue closures were voided
> on 2026-11-01; the branch was emptied (commit `075f86d`) and is being
> rebuilt from scratch. Statuses below track the **new** codebase only.
>
> Last synchronized: 2026-11-03 (session S-V2-11 close: #12 CHEM-02
> re-landed and CLOSED — funnel 24/24 counts exact, 0/50,000
> certification-label mismatches, selection `buffered_rows` exact, QR
> projection bit-exact, conformer-method split 17,493/32,507 exact,
> rebuilt catalog 49,998/50,000 overlap; #17 DATA-02 implemented, run
> full-scale, CLOSED — regenerated 204,005-record manifest, 22,387
> clusters, 14,958 components, 80.0/10.0/10.0, zero leakage; #32 TEST-01
> Suite A implemented and CLOSED — 293/293 green + canonical seeding /
> shard-primitive modules; #16 outputs regenerated full-scale; suite
> **1303 passed / 16 skipped**; clean-room closures now 20 of 40: #5,
> #6, #7, #8, #9, #10, #11, #12, #13, #14, #15, #16, #17, #21, #22,
> #23, #24, #32, #38, #40).

## Legend

`Pending` → `Ready` (deps verified) → `In Progress` → `Implemented —
Awaiting Verification` → `Verified — Ready to Close` → `Closed`.
`Blocked` = external/hard blocker (reason stated).

## Implementation backlog status

| # | ID | Title (short) | Status | Module | Latest commit | Test evidence | Blocker / next action |
|---|---|---|---|---|---|---|---|
| 5 | ARCH-01 | Clean-room purge + V2 foundation | **Closed** (2026-11-01) | repo-wide | `075f86d` (purge) + `0f5d74a` (foundation) | Trees API: 0 entries at purge commit; all legacy targets 404; 0 purged-module references in new `syntree/`; 138/138 tests green | — |
| 6 | ARCH-02 | Frozen tensor schemas & PyG contracts | **Closed** (2026-11-01) | `syntree/core/schema.py` | `0001087` | `tests/test_tensor_contracts.py` 93/93; both #6 acceptance criteria verified (conforming states pass; exact 67-D message raised via constructor/validator/collator) | — |
| 7 | MATH-01 | Non-singular PaiNN norms | **Closed** (2026-11-01) | `syntree/models/equivariant.py` | `00ec653` | `tests/test_mathematical_singularities.py` 45/45; `test_safe_norm_zero_distance` exact-0.0 value & gradient; SO(3) covariance over 5×200 vectors | — |
| 8 | MATH-02 | Stable von Mises backward (Amos ratio) | **Closed** (2026-11-02) | `syntree/models/torsion_head.py` | `ae2c615` | `tests/test_von_mises_torsion.py` 44/44; both acceptance criteria verified (forward vs `scipy.special.i0e` measured max 6.51e-6 over κ∈[0,100] < 1e-5; backward finite, ≥0, strictly <1 incl. κ=0→0) — D-V2-013 | — |
| 9 | MATH-03 | Continuous cosine cutoff envelopes | **Closed** (2026-11-01) | `syntree/models/cutoff.py` + `GaussianRadialBasis` integration | `bf735a6` | `tests/test_cosine_cutoff.py` 46/46; both #9 acceptance criteria verified (bitwise-zero radial vector at `d = r_cut + 1e-4`; gradient → 0 continuously as `d → r_cut⁻`); envelope integrated inside the RBF producer for the backbone (#21) and handle-head (#22) bases — D-V2-011; full suite 347/347 | — |
| 10 | MATH-04 | Fail-closed FP32 precision engine | **Closed** (2026-11-02) | `syntree/utils/hardware.py` | `5ce4f5e` | `tests/test_fp32_enforcement.py` 35/35; both acceptance criteria verified (simulated Tesla T4 cc 7.5 forces `precision="fp32"` + `grad_scaler=False`; `mixed_precision="fp16"` on cc 7.5 logs the FAIL-CLOSED warning and executes pure FP32) — D-V2-014; AST ban on GradScaler/autocast/torch.amp across `src/syntree` | — |
| 11 | CHEM-01 | Certified 32-reaction taxonomy | **Closed** (2026-11-02) | `syntree/chemistry/reactions.py` | `0e296b0` | `tests/test_32_reactions.py` 82/82; both acceptance criteria verified (100% round-trip forward replay on 48 standardized reagent pairs; 16 stereo pairs with CIP retention); `DEFAULT_REACTION_REGISTRY` = the certified taxonomy (interim superseded, D-V2-016) | — |
| 12 | CHEM-02 | Enamine REAL 50k catalog + pharm3d | **Closed** (2026-11-03, S-V2-11: build code re-landed and verified at full scale against the surviving artifacts — funnel 24/24 counts exact over all 290,648 records; 0/50,000 certification-label mismatches; buffered-rows list exact; projection/Q bit-exact; conformer-method split 17,493/32,507 with 0/50,000 mismatches; full rebuild 50,000 rows [50000,256] finite, mean Fsp3 0.7151, mean rot 1.5361, 49,998/50,000 artifact overlap) | `syntree/chemistry/synthon_encoder.py` + `scripts/build_synthon_catalog.py` | `b76422a` + `b8439bd` | `tests/test_synthon_encoder.py` 50/50; immutable HDF5 remains the authoritative catalog (rebuilt h5 = new version, sha256 `00a556ef...`); 3 FG-slot + vector-class-list + ETKDG-seed residuals documented | — |
| 13 | CHEM-03 | Scaffold-constrained MMFF94 relaxation | **Closed** (2026-11-02) | `syntree/chemistry/conformer.py` + `finalize_terminal_molecule` opt-in flags | `472f4df` | `tests/test_conformer_relaxation.py` 18/18; both acceptance criteria verified (amide planarity ≤1.0° restored exactly post-MMFF — a measured MMFF94 finding; seed RMSD <0.15 Å; junction bond <0.02 Å of MMFF equilibrium; pocket static ≤0.05 Å) — D-V2-017 | — |
| 14 | CHEM-04 | Soft-core LJ & volume floors | **Closed** (2026-11-02) | `syntree/chemistry/conformer.py` + engine wiring | `d61a7fc` | `tests/test_soft_core_chemistry.py` 50/50; both acceptance criteria verified (V(0.5)=484.85 kcal/mol < 500 for every supported element pair vs 1.58e9 rigid wall; 10-heavy-atom assembled molecule incurs the explicit heavy-atom penalty) — D-V2-015; full suite 534/534 incl. all 50 engine contracts | — |
| 15 | CHEM-05 | Canonical assembly ordering / decompositions | **Closed** (2026-11-02) | `syntree/data/fragmenter.py` | `864530d` | `tests/test_canonical_fragmenter.py` 21/21; acceptance verified (identical trajectories under 12–20 atom renumberings per ligand incl. the symmetric biphenyl) | — |
| 16 | DATA-01 | Multi-source preprocessing pipeline | **Closed** (2026-11-02) | `syntree/data/preprocessing.py` + `scripts/preprocess_multidataset.py` | `fccd670` | `tests/test_preprocessing.py` 28/28; both acceptance criteria verified at full scale: 204,005 complexes, zero unhandled exceptions (170,954 accepted / 33,051 exact-code rejections); manifest completeness (204,005 records, explicit documented nulls); suite 942/942 + 16 by-design skips — note: the run's output artifacts were partially lost to the sandbox recycle (47,483/170,954 pairs + `processed_manifest.jsonl`); regeneration is mechanical via the in-repo driver (raws + checksums survive) | — |
| 17 | DATA-02 | MMseqs2 30% clustered splits | **Closed** (2026-11-03, S-V2-11: #16 outputs regenerated full-scale first — 204,005 records exact total; then full-scale clustering + splits: 22,387 clusters, 14,958 components, **zero leakage**, train 136,764 / val 17,096 / test 17,095; evidence at `docs/data/splits/`) | `scripts/split_clusters.py` + `scripts/run_mmseqs_split.sh` | `02b09d2` (evidence `83a629e`) | `tests/test_split_clusters.py` 18/18; both acceptance criteria verified (programmatic zero-leakage graph check fails closed; split manifest with exact counts + cluster statistics + per-source counts) | — |
| 18 | DATA-03 | Trajectory extraction + exact replay | Blocked (needs #12, #17, #41 — #11 and #15 satisfied 2026-11-02) | `syntree/data/` | — | — | |
| 19 | DATA-04 | Compressed sharding engine | Blocked (needs #18) | `syntree/data/` | — | — | |
| 20 | DATA-05 | Coordinate/dihedral augmentation | Blocked (needs #19) | `syntree/data/` | — | — | Train split only |
| 21 | MODEL-01 | BipartitePaiNN backbone | **Closed** (2026-11-01) | `syntree/models/bipartite.py` | `e76d955` | `tests/test_bipartite_painn.py` 69/69 (both pinned acceptance criteria: scalar invariance + vector equivariance at ≤3e-13 over 36 rigid-motion trials; clean pocket-only N_l=0); full suite 207/207 | — |
| 22 | MODEL-02 | Handle selection head | **Closed** (2026-11-01) | `syntree/models/handle_head.py` | `433d303` | `tests/test_handle_selection.py` 44/44; #22 acceptance verified (valid normalized distributions over active handle sets for H > 1, on real `AssemblyEnvironment` menus at t=0 and grown intermediates); rigid-motion invariance 8.3e-16 (FP64) / 5.96e-08 (FP32); frozen profile 224,896 params; full suite 301/301 | — |
| 23 | MODEL-03 | Cross-attention decoder | **Closed** (2026-11-01) | `syntree/models/policy.py` | `dc4921e` | `tests/test_cross_attention_decoder.py` 58/58; both #23 acceptance criteria verified ([B,1,256]×[B,N,256]→[B,256]; padded atoms masked exactly — bit-identical under pad junk); SE(3) invariance 8.882e-16 (FP64) / 4.768e-07 (FP32); frozen profile 1,219,328 params; full suite 405/405 | — |
| 24 | MODEL-04 | Hierarchical heads + torsion space | **Closed** (2026-11-02) | `syntree/models/policy.py` + `syntree/models/torsion_head.py` | `74133c8` | `tests/test_hierarchical_heads.py` 100/100; both #24 acceptance criteria verified (STOP-isolation by forward hooks + stop-law invariance; all action components pure categorical laws whose logs compose in TB — kappa/mu invariance pinned); the three pre-V2 bugs pinned dead; full suite 755/755 — D-V2-018 | — |
| 25 | MODEL-05 | SynTreePolicy assembly + audit | **Implemented — Awaiting Verification** (CPU-verified 2026-11-02; the CUDA acceptance criterion is gated on the authorized Colab/Kaggle T4 runtime, #36) | `syntree/models/policy.py` | `84fd3e7` | `tests/test_policy_assembly.py` 48/48; parameter memory 43.5 MB < 50 MB verified by exact arithmetic (10,880,719 params, D-V2-019); full frozen-profile forward zero-NaN/Inf on CPU; the #39 seed/anchor component integrated (generator + factorized head + tied table); full suite 803/803 | CUDA/T4 execution pending #36 authorization |
| 26 | TRAIN-01 | Pure FP32 T4 trainer | Blocked (needs #19, #20, #25 — #10 closed 2026-11-02) | `syntree/engine/` | — | — | #10 satisfied (`RuntimeConfig` contract ready); T4 verification requires authorized Colab/Kaggle |
| 27 | TRAIN-02 | Quantile curriculum (Approach C) | Blocked (needs #26) | `syntree/engine/` | — | — | |
| 28 | TRAIN-03 | 2D→3D transfer pretraining | Blocked (needs #12, #27 — #11 satisfied) | `syntree/engine/` | — | — | Requires licensed reaction data |
| 29 | FLOW-01 | GFlowNet TB engine | Blocked (needs #25, #26, #39 — #13 satisfied 2026-11-02) | `syntree/engine/gflownet.py` | — | TB mathematics + uniform-leaf backward semantics implemented and flow-accounting-verified (D-V2-009); #38 closed; trainer + logZ conditioning remain | |
| 30 | FLOW-02 | Stationary multi-objective reward | Blocked (needs #29 — #14 closed 2026-11-02) | `syntree/engine/rl.py` | — | Stationary reward core + #14 volume floors implemented and tested (D-V2-009/D-V2-015): batch-invariant signature, cavity ratio + heavy-atom floors, complete telemetry; docking-surrogate production closure remains | `#14` satisfied |
| 31 | FLOW-03 | Stage-2 CLI + checkpoint state | Blocked (needs #29, #30) | `main.py` | — | — | |
| 32 | TEST-01 | Suite A capability regression | **Closed** (2026-11-03, S-V2-11: the five issue-pinned batteries under `tests/suite_a/` — reaction execution + isotope provenance (all 32 reactions), posebusters filters, conformer snapping (scaffold restraints), shard I/O (deterministic gzip + sha256 fail-closed), seeding determinism) | `tests/suite_a/` + `syntree/utils/seeding.py` + `syntree/data/shards.py` | `0e1bf30` | `pytest tests/suite_a/ -q` **293/293 green**; the seeding module + shard primitive are the canonical modules #26/#43 and #19 consume (no duplication) | — |
| 33 | TEST-02 | Suite B invariant tests | Blocked (needs #7, #8, #21, #25, #26, #32) | `tests/` | — | — | |
| 34 | EVAL-01 | Independent AiZynthFinder evaluation | Blocked (needs #29) | `scripts/` | — | — | AiZynthFinder availability unverified |
| 35 | EVAL-02 | Comparative SBDD benchmark | Blocked (needs #34, #44) | `scripts/` | — | — | |
| 36 | OPS-01 | Colab SSH remote workflow | Pending (no code dep; **user authorization gate**) | `docs/`, notebook | — | — | Prepare workflow; do NOT claim any connection |
| 37 | OPS-02 | Hugging Face artifact sync | Blocked (needs #19, #26, #31, #35, #36) — final milestone | `syntree/utils/` | — | — | |
| 38 | FLOW-04 | MDP state / derivation DAG semantics | **Closed** (2026-11-01) | `docs/architecture/MDP_STATE_SPACE.md` + `syntree/engine/` | (this session) | All 5 acceptance criteria: written spec distinguishing MDP vs DAG; proven + implemented uniform-leaf `P_B` (STOP term zero by proof); multi-path/symmetric flow tests; toy-DAG `P(x) = R(x)/Z` verification (19 states / 31 trajectories, residuals < 1e-9); #29 references #38's semantics | — |
| 39 | MODEL-06 | Learned seed/anchor selection | Partially implemented (2026-11-02 via #25: the deterministic anchor generator, the factorized SeedAnchorHead, masks, the seed log-probability contribution, and the audit entry exist in `syntree/models/policy.py`); remaining: the environment-side candidate-state rewiring, supervised seed labels, trajectory replay and GFlowNet accounting (blocked on #12's catalog) | `syntree/models/policy.py` | `84fd3e7` | `tests/test_policy_assembly.py` (generator determinism/equivariance, law validity, tied table, gradients) | Environment still commits the seed constructor-side |
| 40 | CHEM-06 | Chemoselectivity filters | **Closed** (2026-11-02) | `syntree/chemistry/chemoselectivity.py` | `044adca` | `tests/test_chemoselectivity.py` 111/111; all 5 #40 acceptance criteria verified (layer after basic validation; traceable named rules; protected substrates pass; full rule coverage; manifest impact reporting) — D-V2-020; full suite 914/914 | — |
| 41 | DATA-06 | Hashed inverted synthon index | Blocked (needs #11, #12) | `syntree/chemistry/` | — | — | |
| 42 | CHEM-07 | Terminal Sub-Bin Geometry Refinement (re-scoped) | Implemented — Awaiting Verification of remaining scope (#14 and #13 dependencies satisfied 2026-11-02: the production clash objective and the #13 opt-in stages exist; remaining open dep: **#24 only**) | `syntree/engine/environment.py` (terminal oracle) | — | Core implemented: coarse-scan-seeded Brent ±5° + MMFF94 restraints, bin-containment + dense-scan comparison + covalent-geometry preservation tests (D-V2-009/D-V2-015); the #13 stages integrate behind the D-V2-017 opt-in flags; T4 benchmarking + #24 production integration remain | |
| 43 | OPS-03 | Measured T4 profiling + session-safe resume | Blocked (needs #26, #36) | `syntree/engine/` | — | — | |
| 44 | EVAL-03 | Stock-aligned + independent retrosynthesis tracks | Blocked (needs #12, #34) | `scripts/` | — | — | |

**Progress denominator: 40 issues (#5–#44).** Closed: 20 (#5, #6, #7,
#8, #9, #10, #11, #12, #13, #14, #15, #16, #17, #21, #22, #23, #24,
#32, #38, #40 — clean-room era). Implemented-awaiting-verification: 1
(#25, CUDA criterion — #36 gate). #45 excluded by standing rule.

## Pre-existing failures

None. The clean-room suite is **1303 passed / 16 skipped** green (942 at
S-V2-10 + 18 #17 split tests + 293 Suite A + 50 #12 encoder tests, session
S-V2-11; 16 #11 skips are by design — reactions without a dedicated stereo
pair); there are no inherited failures.

## Session log pointer

Chronological agent activity: `WORK_LOG.md`. Durable decisions:
`DECISIONS.md`. Verified graph: `DEPENDENCIES.md`. Resumption
instructions: `HANDOFF.md`.
