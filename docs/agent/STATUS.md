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
> Last synchronized: 2026-11-02 (session S-V2-08 close: #11 CHEM-01, #13
> CHEM-03, #15 CHEM-05 implemented, verified, and closed — decisions
> D-V2-016/017; clean-room closures now #5, #6, #7, #8, #9, #10, #11,
> #13, #14, #15, #21, #22, #23, #38).

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
| 12 | CHEM-02 | Enamine REAL 50k catalog + pharm3d | Blocked (needs #40 + catalog access; #11 satisfied) | `syntree/chemistry/` | — | — | Mirror URL + SHA-256 pinned in provenance doc |
| 13 | CHEM-03 | Scaffold-constrained MMFF94 relaxation | **Closed** (2026-11-02) | `syntree/chemistry/conformer.py` + `finalize_terminal_molecule` opt-in flags | `472f4df` | `tests/test_conformer_relaxation.py` 18/18; both acceptance criteria verified (amide planarity ≤1.0° restored exactly post-MMFF — a measured MMFF94 finding; seed RMSD <0.15 Å; junction bond <0.02 Å of MMFF equilibrium; pocket static ≤0.05 Å) — D-V2-017 | — |
| 14 | CHEM-04 | Soft-core LJ & volume floors | **Closed** (2026-11-02) | `syntree/chemistry/conformer.py` + engine wiring | `d61a7fc` | `tests/test_soft_core_chemistry.py` 50/50; both acceptance criteria verified (V(0.5)=484.85 kcal/mol < 500 for every supported element pair vs 1.58e9 rigid wall; 10-heavy-atom assembled molecule incurs the explicit heavy-atom penalty) — D-V2-015; full suite 534/534 incl. all 50 engine contracts | — |
| 15 | CHEM-05 | Canonical assembly ordering / decompositions | **Closed** (2026-11-02) | `syntree/data/fragmenter.py` | `864530d` | `tests/test_canonical_fragmenter.py` 21/21; acceptance verified (identical trajectories under 12–20 atom renumberings per ligand incl. the symmetric biphenyl) | — |
| 16 | DATA-01 | Multi-source preprocessing pipeline | Ready (deps: #5) | `syntree/data/` + `scripts/` | — | — | Provenance: `docs/data/RAW_SOURCE_PROVENANCE.yaml` |
| 17 | DATA-02 | MMseqs2 30% clustered splits | Blocked (needs #16) | `syntree/data/` | — | — | |
| 18 | DATA-03 | Trajectory extraction + exact replay | Blocked (needs #12, #17, #41 — #11 and #15 satisfied 2026-11-02) | `syntree/data/` | — | — | |
| 19 | DATA-04 | Compressed sharding engine | Blocked (needs #18) | `syntree/data/` | — | — | |
| 20 | DATA-05 | Coordinate/dihedral augmentation | Blocked (needs #19) | `syntree/data/` | — | — | Train split only |
| 21 | MODEL-01 | BipartitePaiNN backbone | **Closed** (2026-11-01) | `syntree/models/bipartite.py` | `e76d955` | `tests/test_bipartite_painn.py` 69/69 (both pinned acceptance criteria: scalar invariance + vector equivariance at ≤3e-13 over 36 rigid-motion trials; clean pocket-only N_l=0); full suite 207/207 | — |
| 22 | MODEL-02 | Handle selection head | **Closed** (2026-11-01) | `syntree/models/handle_head.py` | `433d303` | `tests/test_handle_selection.py` 44/44; #22 acceptance verified (valid normalized distributions over active handle sets for H > 1, on real `AssemblyEnvironment` menus at t=0 and grown intermediates); rigid-motion invariance 8.3e-16 (FP64) / 5.96e-08 (FP32); frozen profile 224,896 params; full suite 301/301 | — |
| 23 | MODEL-03 | Cross-attention decoder | **Closed** (2026-11-01) | `syntree/models/policy.py` | `dc4921e` | `tests/test_cross_attention_decoder.py` 58/58; both #23 acceptance criteria verified ([B,1,256]×[B,N,256]→[B,256]; padded atoms masked exactly — bit-identical under pad junk); SE(3) invariance 8.882e-16 (FP64) / 4.768e-07 (FP32); frozen profile 1,219,328 params; full suite 405/405 | — |
| 24 | MODEL-04 | Hierarchical heads + torsion space | **Ready** (all deps satisfied: #11 closed 2026-11-02, #23 closed 2026-11-01, #8 closed 2026-11-02) | `syntree/models/` + `syntree/engine/environment.py` | — | Geometry contract (bin centres, resonance snap, determinism) verified in `tests/test_gflownet_reconciliation.py` (D-V2-009); circular torsion objective ready in `syntree/models/torsion_head.py` (D-V2-013); logit-mask constants ready in `syntree/utils/hardware.py` (D-V2-014); the certified reaction registry with per-family handle metadata ready in `syntree/chemistry/reactions.py` (D-V2-016); consumes `DecoderOutput.context` + per-row `handle_embeddings` | Next session's primary task |
| 25 | MODEL-05 | SynTreePolicy assembly + audit | Blocked (needs #24, #39 interface — #21/#22/#23 closed) | `syntree/models/policy.py` | — | — | #21, #22, #23 satisfied |
| 26 | TRAIN-01 | Pure FP32 T4 trainer | Blocked (needs #19, #20, #25 — #10 closed 2026-11-02) | `syntree/engine/` | — | — | #10 satisfied (`RuntimeConfig` contract ready); T4 verification requires authorized Colab/Kaggle |
| 27 | TRAIN-02 | Quantile curriculum (Approach C) | Blocked (needs #26) | `syntree/engine/` | — | — | |
| 28 | TRAIN-03 | 2D→3D transfer pretraining | Blocked (needs #12, #27 — #11 satisfied) | `syntree/engine/` | — | — | Requires licensed reaction data |
| 29 | FLOW-01 | GFlowNet TB engine | Blocked (needs #25, #26, #39 — #13 satisfied 2026-11-02) | `syntree/engine/gflownet.py` | — | TB mathematics + uniform-leaf backward semantics implemented and flow-accounting-verified (D-V2-009); #38 closed; trainer + logZ conditioning remain | |
| 30 | FLOW-02 | Stationary multi-objective reward | Blocked (needs #29 — #14 closed 2026-11-02) | `syntree/engine/rl.py` | — | Stationary reward core + #14 volume floors implemented and tested (D-V2-009/D-V2-015): batch-invariant signature, cavity ratio + heavy-atom floors, complete telemetry; docking-surrogate production closure remains | `#14` satisfied |
| 31 | FLOW-03 | Stage-2 CLI + checkpoint state | Blocked (needs #29, #30) | `main.py` | — | — | |
| 32 | TEST-01 | Suite A capability regression | Ready (deps: #5) | `tests/` | — | — | Aggregates per-issue capability tests as they land |
| 33 | TEST-02 | Suite B invariant tests | Blocked (needs #7, #8, #21, #25, #26, #32) | `tests/` | — | — | |
| 34 | EVAL-01 | Independent AiZynthFinder evaluation | Blocked (needs #29) | `scripts/` | — | — | AiZynthFinder availability unverified |
| 35 | EVAL-02 | Comparative SBDD benchmark | Blocked (needs #34, #44) | `scripts/` | — | — | |
| 36 | OPS-01 | Colab SSH remote workflow | Pending (no code dep; **user authorization gate**) | `docs/`, notebook | — | — | Prepare workflow; do NOT claim any connection |
| 37 | OPS-02 | Hugging Face artifact sync | Blocked (needs #19, #26, #31, #35, #36) — final milestone | `syntree/utils/` | — | — | |
| 38 | FLOW-04 | MDP state / derivation DAG semantics | **Closed** (2026-11-01) | `docs/architecture/MDP_STATE_SPACE.md` + `syntree/engine/` | (this session) | All 5 acceptance criteria: written spec distinguishing MDP vs DAG; proven + implemented uniform-leaf `P_B` (STOP term zero by proof); multi-path/symmetric flow tests; toy-DAG `P(x) = R(x)/Z` verification (19 states / 31 trajectories, residuals < 1e-9); #29 references #38's semantics | — |
| 39 | MODEL-06 | Learned seed/anchor selection | Pending (#21/#23 satisfied; implied deps #12, #25) | `syntree/models/` | — | — | Design-first work item; may consume the decoder's pocket-conditioned context (#23 addendum) |
| 40 | CHEM-06 | Chemoselectivity filters | **Ready** (needs #11 — satisfied 2026-11-02) | `syntree/chemistry/` | — | — | The registry exposes per-reaction handle metadata + junction patterns (D-V2-016 addendum) |
| 41 | DATA-06 | Hashed inverted synthon index | Blocked (needs #11, #12) | `syntree/chemistry/` | — | — | |
| 42 | CHEM-07 | Terminal Sub-Bin Geometry Refinement (re-scoped) | Implemented — Awaiting Verification of remaining scope (#14 and #13 dependencies satisfied 2026-11-02: the production clash objective and the #13 opt-in stages exist; remaining open dep: **#24 only**) | `syntree/engine/environment.py` (terminal oracle) | — | Core implemented: coarse-scan-seeded Brent ±5° + MMFF94 restraints, bin-containment + dense-scan comparison + covalent-geometry preservation tests (D-V2-009/D-V2-015); the #13 stages integrate behind the D-V2-017 opt-in flags; T4 benchmarking + #24 production integration remain | |
| 43 | OPS-03 | Measured T4 profiling + session-safe resume | Blocked (needs #26, #36) | `syntree/engine/` | — | — | |
| 44 | EVAL-03 | Stock-aligned + independent retrosynthesis tracks | Blocked (needs #12, #34) | `scripts/` | — | — | |

**Progress denominator: 40 issues (#5–#44).** Closed: 14 (#5, #6, #7, #8,
#9, #10, #11, #13, #14, #15, #21, #22, #23, #38 — clean-room era). #45
excluded by standing rule.

## Pre-existing failures

None. The clean-room suite is 655/655 green (534 at S-V2-07 + 82 taxonomy
(#11) + 18 relaxation (#13) + 21 fragmenter (#15) tests, session S-V2-08;
16 #11 skips are by design — reactions without a dedicated stereo pair);
there are no inherited failures.

## Session log pointer

Chronological agent activity: `WORK_LOG.md`. Durable decisions:
`DECISIONS.md`. Verified graph: `DEPENDENCIES.md`. Resumption
instructions: `HANDOFF.md`.
