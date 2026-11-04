# 3D-SynTree V2 — Agent Dependency Map (clean-room era)

> Verified dependency map for the implementation backlog **#5–#44**
> (issue #45 is the permanent constitution and is never part of this
> graph). Sources: the `Depends on` / `Blocks` fields of every issue body
> (re-fetched 2026-11-01 after the clean-room reset), plus reverse edges
> derived from `Blocks` lists for the added audit issues (#38–#44), which
> state their dependencies only implicitly.
>
> **Rule (#45 §17):** an issue is *dependency-ready* only when every hard
> dependency is **Verified — Ready to Close** (or Closed) **and its
> deliverable actually exists in the clean-room V2 codebase and has been
> inspected** — never merely because the blocking issue was closed at some
> point. All pre-reset closures were voided by the clean-room reset
> (decision D-V2-001); statuses below reflect the *new* codebase only.

## 1. Direct dependency edges (as stated in the issues)

| Issue | Depends on (hard) | Blocks |
|---|---|---|
| #5 ARCH-01 | — | #6, #16, #21, #29 |
| #6 ARCH-02 | #5 | #17, #21, #22 |
| #7 MATH-01 | #5 | #21, #32 |
| #8 MATH-02 | #5 | #24, #32 — **closed 2026-11-02**: `syntree/models/torsion_head.py` + 44 tests (D-V2-013) |
| #9 MATH-03 | #5 | #23 |
| #10 MATH-04 | #5 | #26 — **closed 2026-11-02**: `syntree/utils/hardware.py` + 35 tests (D-V2-014) |
| #11 CHEM-01 | #5 | #12, #18, #24, #40, #41 — **closed 2026-11-02**: `syntree/chemistry/reactions.py` + 82 tests (D-V2-016) |
| #12 CHEM-02 | #11, #40 — **closed 2026-11-03** (S-V2-11: build code re-landed and verified at full scale against the surviving artifacts — funnel 24/24 counts exact, 0/50,000 label mismatches, `buffered_rows` exact, projection bit-exact, conformer-method split 17,493/32,507 exact; commits `b76422a`+`b8439bd`; immutable HDF5 remains authoritative) | #18 (satisfied), #24 (satisfied), #39, #41, #44 |
| #13 CHEM-03 | #11 — **closed 2026-11-02**: `conformer.py` stages + opt-in flags, 18 tests (D-V2-017) | #29, #42 |
| #14 CHEM-04 | #5 | #30, #42 — **closed 2026-11-02**: `syntree/chemistry/conformer.py` + engine wiring + 50 tests (D-V2-015) |
| #15 CHEM-05 | #11 — **closed 2026-11-02**: `syntree/data/fragmenter.py` + 21 tests | #18 (its #38 dependency was already satisfied by the reconciliation, D-V2-009) |
| #16 DATA-01 | #5 — **closed 2026-11-02** (`fccd670`, 28 tests, full-scale run over 204,005 complexes; suite 942/16): `syntree/data/preprocessing.py` + `scripts/preprocess_multidataset.py` | #17 (satisfied) |
| #17 DATA-02 | #16 — **closed 2026-11-03** (`02b09d2`/`83a629e`, S-V2-11: #16 outputs regenerated full-scale first (204,005 records, exact total; +1 MOAD acceptance vs the original run attributed to upstream API drift); 22,387 clusters → 14,958 components; 80.0/10.0/10.0; zero leakage, fail-closed check; evidence at `docs/data/splits/`) | #18 (satisfied) |
| #18 DATA-03 | #11, #12, #15, #17, #41 — **all closed**; **CLOSED 2026-11-03** (S-V2-13: the full-scale run over all 170,955 accepted complexes — **194 accepted trajectories** (train 111 / val 50 / test 33), 176.8/s mean, closed failure vocabulary over 170,761 rejects; evidence `docs/data/trajectories/README.md` + `extraction_manifest_full_scale.json`) | #19 (satisfied) |
| #19 DATA-04 | #18 — **CLOSED 2026-11-03** (S-V2-13: `syntree/data/sharded.py` + `scripts/shard_and_upload.py` @ `b9b043e` — byte-bounded packing with fail-closed size verification, SHA-256 manifests, HF streaming, shard-aware unbiased sampler) | #26 (satisfied), #37 |
| #20 DATA-05 | #19 — **CLOSED 2026-11-03** (S-V2-14: `syntree/data/augmentation.py` @ `bc41319` — σ=0.15 Å ligand-only jitter + rigid ±5° non-junction dihedral perturbation, train-split only, stereo-preserving rejection resampling per D-V2-035; 34 tests) | #26 (satisfied) |
| #21 MODEL-01 | #6, #7 | #23, #24, #39 |
| #22 MODEL-02 | #6, #21 | #23 — **closed 2026-11-01**: `syntree/models/handle_head.py` + 44 tests (D-V2-010) |
| #23 MODEL-03 | #9, #21, #22 — **closed 2026-11-01 (session S-V2-06)**: `syntree/models/policy.py` + 58 tests (D-V2-012); all three deps were closed and inspected before implementation | #24, #39 |
| #24 MODEL-04 | #11, #8, #23 — **closed 2026-11-02** (`74133c8`, 100 tests, D-V2-018): `syntree/models/policy.py` hierarchical heads + `torsion_head.py` 36-bin categorical | #25 (satisfied), #29, #38 (satisfied), #42 |
| #25 MODEL-05 | #21, #22, #23, #24, #39* — **implemented 2026-11-02** (`84fd3e7`, 48 tests, D-V2-019; the #39 dependency satisfied by the integrated seed/anchor component); **remaining: the CUDA acceptance criterion (#36 gate)** | #26, #29 |
| #26 TRAIN-01 | #10 (closed 2026-11-02, D-V2-014), #19 (closed S-V2-13), #20 (closed S-V2-14), #25 (implemented — only its CUDA forward criterion is gated on the authorized #36 session); **implemented + CPU-verified 2026-11-03** (S-V2-14: `syntree/engine/trainer.py` + `syntree/data/supervised.py` + `configs/train_colab_t4.json` + `scripts/train_stage1.py` @ `e009127`; 40 tests + the end-to-end CLI smoke; the two T4 acceptance criteria — monotonic 2-epoch descent on T4, peak reserved VRAM < 9.5 GB — await the authorized #36 runtime; S-V2-19 fixed the loader blocker D-V2-043 at `b7e0f3d` (reactant-form graft matching — the product-graph matching blinded the pool to the nine graft classes) and staged the acceptance driver `scripts/runtime_train_chain.sh`) | #27, #33, #43 |
| #27 TRAIN-02 | #26 — **closed 2026-11-04** (S-V2-15: `syntree/engine/curriculum.py` + trainer/CLI integration @ `9197619`; the five issue-named difficulty axes, 9 quantile-balanced bins, the 6-stage floor-exact distribution (ε=0.03), exactly-N deterministic draws, the 0.005/patience-2 milestone machine, checkpointable state in the trainer payload; 40 tests incl. the `--curriculum` CLI smoke; both acceptance criteria verified — all 9 levels every epoch via histograms, 40-epoch smooth 6-stage progression) | #28 (satisfied) |
| #28 TRAIN-03 | #11, #12, #27 — **all closed**; the code path is dependency-ready; the *data* dependency is external: licensed, provenance-tracked reaction data for the 2D tier (#45 §12.2 forbids pretraining claims without it) — the owner must supply or authorize deferral | #29 |
| #29 FLOW-01 | #13, #25, #26, #38, #39, #42 — **ALL SATISFIED** (#13 closed 2026-11-02; #38 closed 2026-11-01 D-V2-009; #25 closed 2026-11-04 with the authorized-T4 evidence; #26 implemented + CPU-verified with its deliverable inspected; #39 closed 2026-11-03; #42's terminal-oracle deliverable exists, is tested, and is consumed end-to-end by the rollout driver); **CLOSED 2026-11-04** (S-V2-18: `LogPartitionHead` + the tensor log-prob path + `GFlowNetTrainer` + the production rollout bridge `syntree/engine/rollout.py` @ `6eccd41`; both acceptance criteria verified on the 19-state/31-trajectory DAG — exact-zero residuals at the optimal policy, learning convergence with logZ → the true partition function and R(x)/Z sampling to TV<0.15; D-V2-040/041) | #30 (satisfied), #31 (satisfied), #34 (unblocked) |
| #30 FLOW-02 | #14, #29 — **both closed** (#14 2026-11-02 D-V2-015; #29 2026-11-04); **CLOSED 2026-11-04** (S-V2-18: the stationary core was verified at S-V2-03/D-V2-009; the remaining implementable scope — the docking-backend injection point `DockingResult`/`surrogate_docking_scorer`/`ThreeDReward(docking_scorer=...)` — landed @ `90b1035`, D-V2-042; the real docking engine is #34/#35 scope and injects through the interface) | #31 (satisfied) |
| #31 FLOW-03 | #29, #30 — **both closed 2026-11-04**; **CLOSED 2026-11-04** (S-V2-18: `scripts/train_stage2.py` + `configs/stage2_colab_t4.json` @ `6760b3d` — the V2-canonical form of the legacy `main.py --mode rl` (D-V2-040); Stage-1 weight init with verified provenance, `--resume-auto` exact continuation, fail-closed paths; 6 CLI tests against a REAL Stage-1 fit) | #37 (one dep closer: #31 satisfied) |
| #32 TEST-01 | #5 — **closed 2026-11-03** (`0e1bf30`, S-V2-11: the five issue-pinned Suite A batteries 293/293 green + the canonical seeding module (`syntree/utils/seeding.py`) and shard-format primitive (`syntree/data/shards.py`) that #26/#43 and #19 consume) | #33 (satisfied) |
| #33 TEST-02 | #7, #8, #21, #25, #26, #32 — **implemented + CPU-verified 2026-11-04** (S-V2-15: the five issue-pinned batteries under `tests/suite_b/` @ `7a6ea3b`; `pytest tests/suite_b/ -v` → 47 passed / 1 skipped — the single skip is the battery-4 T4 memory measurement, which by the issue's own scope requires the authorized #36 runtime and executes there without modification) | #34 (satisfied at the CPU level; the memory battery rides #36) |
| #34 EVAL-01 | #29 | #35, #44 |
| #35 EVAL-02 | #34, #44 | #37 |
| #36 OPS-01 | — (no code deps; the **user authorization gate** is a hard external dependency, #45 §6/§29) — **prepared 2026-11-03** (S-V2-14: `notebooks/run_3d_syntree.ipynb` + `docs/ops/colab_ssh_workflow.md` @ `39e9a41`; the runtime-side init, the fail-closed T4 preflight, the operator `colab ssh` CLI flow and the §29.1 authentication handoff are ready; **no connection established — none claimed**) | #37, #43 (both ride the authorized session) |
| #37 OPS-02 | #19, #26, #31, #35, #36 | — (final milestone) |
| #38 FLOW-04 | implied: #15, #24 (from their `Blocks` lists) | — (**closed 2026-11-01**: spec delivered ahead of the cycle via the owner-directed reconciliation, D-V2-009; its deliverables live in `docs/architecture/MDP_STATE_SPACE.md` + `syntree/engine/`) |
| #39 MODEL-06 | implied: #12, #21, #23, #25 — **closed 2026-11-03** (S-V2-12: policy side via #25 `84fd3e7`; environment side `d1a9167` — pre-seed candidate state, explicit SeedAction, covariant pose D-V2-028, TB seed accounting Property 3b, supervised anchor labels; 24 tests) | #25's #39 dependency fully satisfied; #29's remaining deps: #25, #26 |
| #40 CHEM-06 | implied: #11 — **closed 2026-11-02** (`044adca`, 111 tests, D-V2-020) | #12 (satisfied), #44 |
| #41 DATA-06 | implied: #11, #12 — **closed 2026-11-03** (S-V2-12, `dea8809`: full-scale benchmark — exhaustive self-retrieval 68,417/0, brute-force equivalence 2,000/0, 888.7× reduction; the #18 consumer criterion demonstrated on real data) | #18 (satisfied) |
| #42 CHEM-07 | implied: #13, #14, #24 — **all satisfied 2026-11-02** (#24 closed `74133c8`); remaining scope: the T4 benchmarking measurement (**advanced**: the #29 rollout driver @ `6eccd41` now consumes the #24 torsion head + the #42 oracle in the integrated generation loop) | — |
| #43 OPS-03 | implied: #26, #36 | — |
| #44 EVAL-03 | implied: #12, #34 — **#12 closed 2026-11-03**; remaining: #34 | — |

\* #25 lists #39 and #29 lists #39 among dependencies (seed/anchor
selection feeds the policy assembly and the GFlowNet forward probability);
#29 also listed #38. These added-issue edges made the graph **cyclic** as
literally stated (#29 → #38 → #24 → … → #25 → #39 → #25). **The cycle is
now broken:** #38 was closed on 2026-11-01 by the owner-directed
reconciliation (D-V2-009) — its formal state-space specification, the
uniform-leaf backward policy, and their proofs landed ahead of #15/#24 as
design-first deliverables, so #29's #38 dependency is satisfied and #29's
remaining blockers are #13, #25, #26, #39. The #39 seed/anchor head
remains a design-first work item; re-verify edges in both directions
before starting any of #39–#44 (#45 §17). Update 2026-11-02 (S-V2-08):
#13 is also closed (D-V2-017), so #29's remaining blockers are #25, #26,
#39 and #42's remaining blocker is #24 only.

## 2. Critical path (longest dependency chain)

```
#5 → #6 → #21 → #23 → #24 → #25 → #26 → #27 → #28 → #29 → #30 → #31 → #37
```

Update (S-V2-18): #29/#30/#31 are closed — the remaining chain is
**#34 → #44 → #35 → #37** (evaluation) with **#36 → #26/#42/#43** as
the parallel authorized-hardware track and #28 the owner decision.

with the chemistry/data spur feeding in earlier:
`#5 → #11 → #12/#15 → #18 → #19 → #26` and `#11 → #13 → #29`.

## 3. Dependency-READY set (against the clean-room codebase)

Sessions S-V2-01 … S-V2-11 have closed the foundation (#5, #6), math
(#7, #8, #9, #10), chemistry (#11, #12, #13, #14, #15, #40), and data
(#16, #17) layers, the model stack (#21, #22, #23, #24), the Suite A
battery (#32), and the state-space spec (#38). The critical path now
runs: **#41 → #18 → #19 → #20 → #26 → #27 → #28 → #29 → #30 → #31 →
#37**. Dependency-READY set (verified against the clean-room codebase):

- **#41 DATA-06** (deps #11 + #12 both closed 2026-11-03): the hashed
  inverted index over the certified catalog — the catalog HDF5 (private
  HF `JJKK1212/3d-syntree-v2-catalog`, checksum-pinned) and the
  certified pool (152,489 blocks) are available; benchmark against a
  brute-force reference on an exhaustive subset per the issue. Closes
  the last #18 dependency.
- **#39 MODEL-06 environment side** (the policy-side component landed
  with #25; #12's catalog unblocked it): the environment-side
  candidate-state rewiring + supervised seed labels + TB accounting.
- **#36 OPS-01** — prepare-only (notebook + workflow docs); the remote
  session is user-authorization-gated. An authorized T4 session would
  ALSO close #25's remaining CUDA criterion (one full-profile
  SynTreePolicy forward with zero NaN/Infs on device) — run it first.

## 4. External (non-issue) dependencies

| Consumer | External dependency | Status |
|---|---|---|
| #12, #44 | Enamine REAL building-block catalog access (license, download) | Verified open mirror URL + SHA-256 on file (`docs/data/RAW_SOURCE_PROVENANCE.yaml`); curation counts must come from a manifest, never estimates |
| #16 | CrossDocked2020 / PDBbind refined / BindingMOAD raw sources | URLs + checksums + verification dates pinned in `docs/data/RAW_SOURCE_PROVENANCE.yaml` (verified 2026-10-28) |
| #17 | MMseqs2 binary | **Resolved** (S-V2-11: static binary used for the full-scale run; splits closed) |
| #26, #43 | Actual Tesla T4 runtime (Colab/Kaggle) | Requires user authorization per #45 §6/§28; NOT validatable in the CPU dev sandbox |
| #34, #44 | AiZynthFinder + stock files | Unverified availability |
| #37 | Hugging Face Hub write credentials (`HF_TOKEN` env var) | Available to authorized sessions only; never hardcode |
