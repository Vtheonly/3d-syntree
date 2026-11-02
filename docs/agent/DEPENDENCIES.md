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
| #12 CHEM-02 | #11, #40 — **both closed**; **catalog artifacts built full-scale 2026-11-02** (S-V2-10: 290,648 screened → 50,000 selected + encoded; all acceptance criteria verified; HDF5 + manifest on HF `JJKK1212/3d-syntree-v2-catalog`, evidence at `docs/data/catalog/`); remaining: re-land the build code (`synthon_encoder.py` + driver + tests) lost to the sandbox recycle before commit | #18, #24 (satisfied), #39, #41, #44 |
| #13 CHEM-03 | #11 — **closed 2026-11-02**: `conformer.py` stages + opt-in flags, 18 tests (D-V2-017) | #29, #42 |
| #14 CHEM-04 | #5 | #30, #42 — **closed 2026-11-02**: `syntree/chemistry/conformer.py` + engine wiring + 50 tests (D-V2-015) |
| #15 CHEM-05 | #11 — **closed 2026-11-02**: `syntree/data/fragmenter.py` + 21 tests | #18 (its #38 dependency was already satisfied by the reconciliation, D-V2-009) |
| #16 DATA-01 | #5 — **closed 2026-11-02** (`fccd670`, 28 tests, full-scale run over 204,005 complexes; suite 942/16): `syntree/data/preprocessing.py` + `scripts/preprocess_multidataset.py` | #17 (satisfied) |
| #17 DATA-02 | #16 — **closed 2026-11-02** (`fccd670`); dependency-ready once `processed_manifest.jsonl` is regenerated (the run's outputs were partially lost to the S-V2-10 sandbox recycle; the driver is in-repo and all raw sources + checksums survive); mmseqs binary staged in the dev workspace | #18 |
| #18 DATA-03 | #11, #12, #15, #17, #41 — **#11 and #15 closed 2026-11-02**; remaining: #12, #17, #41 | #19 |
| #19 DATA-04 | #18 | #26, #37 |
| #20 DATA-05 | #19 | #26 |
| #21 MODEL-01 | #6, #7 | #23, #24, #39 |
| #22 MODEL-02 | #6, #21 | #23 — **closed 2026-11-01**: `syntree/models/handle_head.py` + 44 tests (D-V2-010) |
| #23 MODEL-03 | #9, #21, #22 — **closed 2026-11-01 (session S-V2-06)**: `syntree/models/policy.py` + 58 tests (D-V2-012); all three deps were closed and inspected before implementation | #24, #39 |
| #24 MODEL-04 | #11, #8, #23 — **closed 2026-11-02** (`74133c8`, 100 tests, D-V2-018): `syntree/models/policy.py` hierarchical heads + `torsion_head.py` 36-bin categorical | #25 (satisfied), #29, #38 (satisfied), #42 |
| #25 MODEL-05 | #21, #22, #23, #24, #39* — **implemented 2026-11-02** (`84fd3e7`, 48 tests, D-V2-019; the #39 dependency satisfied by the integrated seed/anchor component); **remaining: the CUDA acceptance criterion (#36 gate)** | #26, #29 |
| #26 TRAIN-01 | #10, #19, #20, #25 — **#10 closed 2026-11-02** (`syntree/utils/hardware.py`, D-V2-014; `RuntimeConfig` ready); remaining open deps: #19, #20, #25 | #27, #33, #43 |
| #27 TRAIN-02 | #26 | #28 |
| #28 TRAIN-03 | #11, #12, #27 | #29 |
| #29 FLOW-01 | #13, #25, #26, #38, #39* — **#13 closed 2026-11-02** (D-V2-017); **#38 closed 2026-11-01** (D-V2-009); remaining: #25, #26, #39 | #30, #31 |
| #30 FLOW-02 | #14, #29 — **#14 closed 2026-11-02** (`syntree/chemistry/conformer.py` + `rl.py` volume floors, D-V2-015); remaining open deps: #29 | #31 |
| #31 FLOW-03 | #29, #30 | #37 |
| #32 TEST-01 | #5 | #33 |
| #33 TEST-02 | #7, #8, #21, #25, #26, #32 | #34 |
| #34 EVAL-01 | #29 | #35, #44 |
| #35 EVAL-02 | #34, #44 | #37 |
| #36 OPS-01 | — (user authorization required) | #37, #43 |
| #37 OPS-02 | #19, #26, #31, #35, #36 | — (final milestone) |
| #38 FLOW-04 | implied: #15, #24 (from their `Blocks` lists) | — (**closed 2026-11-01**: spec delivered ahead of the cycle via the owner-directed reconciliation, D-V2-009; its deliverables live in `docs/architecture/MDP_STATE_SPACE.md` + `syntree/engine/`) |
| #39 MODEL-06 | implied: #12, #21, #23, #25 — the **policy-side component is implemented** (2026-11-02 via #25: generator + factorized head + tied table + log-prob, `84fd3e7`); remaining: environment-side candidate-state rewiring + supervised labels + TB accounting, blocked on #12 | — |
| #40 CHEM-06 | implied: #11 — **closed 2026-11-02** (`044adca`, 111 tests, D-V2-020) | #12 (satisfied), #44 |
| #41 DATA-06 | implied: #11, #12 — **#11 closed 2026-11-02**; remaining: #12 | — |
| #42 CHEM-07 | implied: #13, #14, #24 — **all satisfied 2026-11-02** (#24 closed `74133c8`); remaining scope: T4 benchmarking + integrated-generation wiring | — |
| #43 OPS-03 | implied: #26, #36 | — |
| #44 EVAL-03 | implied: #12, #34 | — |

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

with the chemistry/data spur feeding in earlier:
`#5 → #11 → #12/#15 → #18 → #19 → #26` and `#11 → #13 → #29`.

## 3. Dependency-READY set (against the clean-room codebase)

With #5 (foundation), #6 (schema), #7 (norms), #9 (cosine cutoff
envelope), #21 (backbone), #22 (handle-selection head), #23
(cross-attention decoder), #8 (von Mises math), #10 (FP32 hardware
gate), #14 (soft-core clash + volume floors), #11 (certified 32-reaction
taxonomy), #13 (junction-torsion + constrained MMFF stages), #15
(canonical decomposer) implemented and verified (sessions S-V2-01 …
S-V2-08), and #38 closed with its state-space spec, the engine layer of
the reconciled two-level GFlowNet framework also exists
(`syntree/engine/`, D-V2-009/D-V2-015). With #16 closed (multi-source
preprocessing, full-scale run verified) and #12's catalog artifacts
built and verified (build code re-landing
pending), the model side of the critical path is complete through the
policy assembly and the data chain is unblocked through #17. The
critical path now runs: **#17 → #18 (needs #12-code + #41) → #19 → #20
→ #26**. Dependency-READY set (verified against the clean-room
codebase):

- **#12 code re-landing** (the highest-leverage open item): the
  artifacts, manifest, checksums, and every acceptance number are
  pinned (D-V2-021/022/023, `docs/data/catalog/`); the #40 screen
  machinery is in-repo; re-implement `synthon_encoder.py` +
  `scripts/build_synthon_catalog.py` + tests against those pins, then
  close #12 (which unblocks #39's environment side, #41, #44, and the
  #18 chemistry dependency).
- **#17 DATA-02** (deps: #16 closed): regenerate
  `processed_manifest.jsonl` first (in-repo driver + surviving raws
  with pinned checksums), then MMseqs2 30%-identity clustered splits
  (binary staged at `3d-syntree-work/tools/mmseqs` in the dev
  workspace; external-dep row below).
- **#32 TEST-01** (Suite A) — depends on #5 only; aggregates the
  per-issue capability tests (942/942 across the test-carrying closed
  issues) into the Suite A runner; parallel-safe.
- **#36 OPS-01** — prepare-only (notebook + workflow docs); the remote
  session is user-authorization-gated. An authorized T4 session would
  ALSO close #25's remaining CUDA criterion — run the full-profile
  forward first.

## 4. External (non-issue) dependencies

| Consumer | External dependency | Status |
|---|---|---|
| #12, #44 | Enamine REAL building-block catalog access (license, download) | Verified open mirror URL + SHA-256 on file (`docs/data/RAW_SOURCE_PROVENANCE.yaml`); curation counts must come from a manifest, never estimates |
| #16 | CrossDocked2020 / PDBbind refined / BindingMOAD raw sources | URLs + checksums + verification dates pinned in `docs/data/RAW_SOURCE_PROVENANCE.yaml` (verified 2026-10-28) |
| #17 | MMseqs2 binary | Must be installed in the execution environment |
| #26, #43 | Actual Tesla T4 runtime (Colab/Kaggle) | Requires user authorization per #45 §6/§28; NOT validatable in the CPU dev sandbox |
| #34, #44 | AiZynthFinder + stock files | Unverified availability |
| #37 | Hugging Face Hub write credentials (`HF_TOKEN` env var) | Available to authorized sessions only; never hardcode |
