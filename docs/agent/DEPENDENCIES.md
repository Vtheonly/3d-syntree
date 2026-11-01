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
| #8 MATH-02 | #5 | #24, #32 |
| #9 MATH-03 | #5 | #23 |
| #10 MATH-04 | #5 | #26 |
| #11 CHEM-01 | #5 | #12, #18, #24, #40, #41 |
| #12 CHEM-02 | #11, #40 | #18, #24, #39, #41, #44 |
| #13 CHEM-03 | #11 | #29, #42 |
| #14 CHEM-04 | #5 | #30, #42 |
| #15 CHEM-05 | #11 | #18, #38 (×2) |
| #16 DATA-01 | #5 | #17 |
| #17 DATA-02 | #16 | #18 |
| #18 DATA-03 | #11, #12, #15, #17, #41 | #19 |
| #19 DATA-04 | #18 | #26, #37 |
| #20 DATA-05 | #19 | #26 |
| #21 MODEL-01 | #6, #7 | #23, #24, #39 |
| #22 MODEL-02 | #6, #21 | #23 |
| #23 MODEL-03 | #9, #21, #22 | #24, #39 |
| #24 MODEL-04 | #11, #8, #23 | #25, #29, #38, #42 |
| #25 MODEL-05 | #21, #22, #23, #24, #39* | #26, #29 |
| #26 TRAIN-01 | #10, #19, #20, #25 | #27, #33, #43 |
| #27 TRAIN-02 | #26 | #28 |
| #28 TRAIN-03 | #11, #12, #27 | #29 |
| #29 FLOW-01 | #13, #25, #26, #38, #39* | #30, #31 |
| #30 FLOW-02 | #14, #29 | #31 |
| #31 FLOW-03 | #29, #30 | #37 |
| #32 TEST-01 | #5 | #33 |
| #33 TEST-02 | #7, #8, #21, #25, #26, #32 | #34 |
| #34 EVAL-01 | #29 | #35, #44 |
| #35 EVAL-02 | #34, #44 | #37 |
| #36 OPS-01 | — (user authorization required) | #37, #43 |
| #37 OPS-02 | #19, #26, #31, #35, #36 | — (final milestone) |
| #38 FLOW-04 | implied: #15, #24 (from their `Blocks` lists) | — (**closed 2026-11-01**: spec delivered ahead of the cycle via the owner-directed reconciliation, D-V2-009; its deliverables live in `docs/architecture/MDP_STATE_SPACE.md` + `syntree/engine/`) |
| #39 MODEL-06 | implied: #12, #21, #23, #25 | — |
| #40 CHEM-06 | implied: #11 | — |
| #41 DATA-06 | implied: #11, #12 | — |
| #42 CHEM-07 | implied: #13, #14, #24 | — |
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
before starting any of #39–#44 (#45 §17).

## 2. Critical path (longest dependency chain)

```
#5 → #6 → #21 → #23 → #24 → #25 → #26 → #27 → #28 → #29 → #30 → #31 → #37
```

with the chemistry/data spur feeding in earlier:
`#5 → #11 → #12/#15 → #18 → #19 → #26` and `#11 → #13 → #29`.

## 3. Dependency-READY set (against the clean-room codebase)

With #5 (foundation), #6 (schema), #7 (norms), #21 (backbone), and #38
(state-space semantics) implemented and verified (sessions S-V2-01 /
S-V2-02 / S-V2-03), the engine layer of the reconciled two-level GFlowNet
framework also exists (`syntree/engine/`, D-V2-009):

- **#22 MODEL-02** (handle selection head) — both hard deps satisfied
  (#6 schema at `0001087`, #21 backbone at `e76d955`); critical path;
  recommended next. Consumes `BackboneOutput` ligand node representations.
- **#8 MATH-02**, **#9 MATH-03**, **#10 MATH-04**, **#11 CHEM-01**,
  **#14 CHEM-04**, **#16 DATA-01**, **#32 TEST-01** — depend on #5 only;
  parallel-safe.
- **#36 OPS-01** — no code dependency, but the remote-Colab part is gated
  on the **user's** authentication/authorization: prepare the workflow,
  never claim a connection.

#23 MODEL-03 additionally needs #9 and #22 (both above); #24 MODEL-04
additionally needs #11 and #8 (both above) plus #23.

## 4. External (non-issue) dependencies

| Consumer | External dependency | Status |
|---|---|---|
| #12, #44 | Enamine REAL building-block catalog access (license, download) | Verified open mirror URL + SHA-256 on file (`docs/data/RAW_SOURCE_PROVENANCE.yaml`); curation counts must come from a manifest, never estimates |
| #16 | CrossDocked2020 / PDBbind refined / BindingMOAD raw sources | URLs + checksums + verification dates pinned in `docs/data/RAW_SOURCE_PROVENANCE.yaml` (verified 2026-10-28) |
| #17 | MMseqs2 binary | Must be installed in the execution environment |
| #26, #43 | Actual Tesla T4 runtime (Colab/Kaggle) | Requires user authorization per #45 §6/§28; NOT validatable in the CPU dev sandbox |
| #34, #44 | AiZynthFinder + stock files | Unverified availability |
| #37 | Hugging Face Hub write credentials (`HF_TOKEN` env var) | Available to authorized sessions only; never hardcode |
