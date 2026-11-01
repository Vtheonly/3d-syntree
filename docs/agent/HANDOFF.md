# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-01, session S-V2-02 close (#21 MODEL-01 —
clean-room BipartitePaiNN backbone, verified and closed).

## 1. Continuation instruction

```
Next session: implement #22 MODEL-02 (HandleSelectionHead) — both hard
deps are closed (#6 schema at commit 0001087, #21 backbone at commit
e76d955). Last verified commit: e76d955 (head of 3d-synV2, pushed; code
head; the S-V2-02 docs commit follows it). #5, #6, #7 and #21 are
implemented from scratch, locally verified (207/207 tests), and CLOSED —
do NOT redo the purge, the schema, the norms, or the backbone. Before
coding, re-read issue #22 in full, then `docs/architecture/ARCHITECTURE.md`
§3.2/§3.10 and this file, and confirm branch HEAD matches origin (push
races possible). #22 must: live in `syntree/models/handle_head.py`,
consume the ligand node representations + union metadata exposed by
`BackboneOutput` (never a second pocket encoder), cross-attend handle
queries over handle invariant descriptors [B_k, 64] and handle positions
relative to pocket centroids, and emit normalized logits over the active
handle set P(handle_t | s_t) for H > 1 exposed handles (acceptance: valid
normalized distributions). Parallel-safe alternatives if #22 is taken:
#8 MATH-02, #9 MATH-03, #10 MATH-04, #11 CHEM-01, #14 CHEM-04, #16 DATA-01,
#32 TEST-01. Note for #9: the backbone RBF is deliberately envelope-free
(decision D-V2-007) — #9 owns the cosine envelope for ALL spatial RBF
projections.
```

## 2. Binding context for every future session

- The branch is a **clean-room rebuild**: it was emptied at commit
  `075f86d` (2026-11-01) per the owner's directive. Never restore,
  re-copy, or "port" pre-reset code; carry forward specifications only
  (decision D-V2-001/D-V2-002). Pre-reset commits are forensic history.
- #45 is the permanent constitution: read it fully before any action;
  never close it.
- Pure FP32 on Colab/Kaggle Tesla T4 only; the dev sandbox is CPU-only
  and can never establish GPU claims (#45 §5–§6).
- Issue-scoped module paths and test file names pinned by issue bodies are
  binding (D-V2-002). New pinned names from #21: module
  `syntree/models/bipartite.py`, tests `tests/test_bipartite_painn.py`.
- Backbone contract highlights (D-V2-007/D-V2-008): `GaussianRadialBasis`
  emits the raw 20-Gaussian expansion with NO cosine envelope (#9 owns the
  envelope); the backbone accepts FP32 (production) and FP64 (local
  verification only, must match module dtype); `build_union_radius_graph`
  is the one authoritative edge construction — downstream heads must
  reuse it, not re-implement it.

## 3. What the next agent must NOT redo

- The clean-room purge (#5): the branch was emptied at `075f86d` and
  verified; **never** restore or port pre-reset code (D-V2-001) —
  pre-reset commits are forensic history only.
- The frozen tensor schema (#6): `syntree/core/schema.py` is the single
  state contract — extend it, never fork it. Its rules (12 canonical
  fields + version stamp, 67-D rejection, collator-only normalisations,
  single batching path) are binding; 93 tests in
  `tests/test_tensor_contracts.py` pin them.
- The non-singular primitives and PaiNN blocks (#7):
  `syntree/models/equivariant.py` (`safe_norm`, `safe_unit_vector`,
  `PaiNNInteraction`, `PaiNNMixing`, published-equation form per
  D-V2-004); 45 tests in `tests/test_mathematical_singularities.py` pin
  the singularities and equivariance. #21 composes these blocks —
  do not reimplement them.
- The union-graph backbone (#21): `syntree/models/bipartite.py`
  (`BipartitePaiNN` 8×d256, `GaussianRadialBasis`, exact
  `build_union_radius_graph`, `BackboneOutput` with the #39-facing pooled
  contexts, activation checkpointing, frozen profile = 5,999,104 params);
  69 tests in `tests/test_bipartite_painn.py` pin the contract (union
  ordering, three pair families, batching = single-graph, fail-closed
  dtype/atomic-number/pocket-coverage rules). Downstream heads consume
  `BackboneOutput` — no second pocket encoder.
- README / architecture / agent-docs structure — update in place.

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| No GPU/T4 execution, training, or HF artifact sync has occurred for V2 | #36/#43 gate | CPU sandbox cannot establish GPU claims |
| Dev environment not pinned in a lockfile | #26/#43 | `pyproject.toml` declares loose floors (torch≥2.1, PyG≥2.4); pin exact versions when the trainer lands |
| Schema `handle_nodes` optional context | #22/#25 | Policy-side extension point, deliberately outside the 12-field core |
| Unstamped states pass field validation (interim) | #19 | Remove the interim rule when shard producers stamp |
| Backbone RBF has no smooth cutoff envelope | #9 | Deliberate boundary (D-V2-007); #9 must apply the envelope to ALL spatial RBF projections |
| Backbone ≥1-pocket-atom-per-graph rule is stricter than schema | #21 | Documented in-module; a schema-level per-graph pocket rule may be worth adopting at #18/#19 |
| Activation-checkpointing T4 benefit unmeasured | #43 | Capability implemented + correctness-tested on CPU only |
| `num_graphs` for non-Batch inputs is derived (max batch index + 1) | #21 | A Batch's `num_graphs` is trusted; index-range + pocket-coverage checks catch inconsistency |

## 5. Remote runtime / checkpoint state

None — no Colab/Kaggle session has been opened (user authorization not
requested; #36 gate). No V2 checkpoints exist. No HF artifacts synced.

## 6. Session mechanics

1. `git clone --branch 3d-synV2 ...` → confirm HEAD matches §1's recorded
   commit; `git fetch` first if in doubt (push races possible).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]" &&
   pip install torch_geometric`.
3. Baseline the suite: `python -m pytest tests/ -q` — expected results
   recorded in §1 (207/207 as of S-V2-02).
4. Work per the #45 §18 per-issue protocol; end with the mandatory git
   diff review, commit+push (report exact SHAs), issue update, and
   refreshed handoff.
