# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-01, session S-V2-04 close (#22 MODEL-02
implemented, verified, and closed — decision D-V2-010).

## 1. Continuation instruction

```
Next session: implement #9 MATH-03 (continuous cosine cutoff envelopes) —
it depends on #5 only and is now the CRITICAL PATH: it is the last open
dependency of #23 MODEL-03 (#21 and #22 are closed), and per decision
D-V2-007 #9 owns the cosine envelope for ALL spatial RBF projections —
when integrating, it must cover both the backbone's edge-distance RBF
(syntree/models/bipartite.py::GaussianRadialBasis consumers) and the
handle head's centroid-distance RBF (syntree/models/handle_head.py,
[0, 12] A basis). Last verified code state: the S-V2-04 commits on
3d-synV2 (implementation `433d303`, pushed 589a843..433d303; suite
301/301). #5, #6, #7,
#21, #38, and #22 are implemented from scratch, locally verified, and
CLOSED — do NOT redo the purge, the schema, the norms, the backbone, the
GFlowNet state-space reconciliation, or the handle-selection head.
Before coding, re-read issue #9 in full, then
syntree/models/bipartite.py (the deliberately envelope-free RBF + the
pinned test that at d = r_cut the last Gaussian still evaluates to 1.0),
D-V2-007 in docs/agent/DECISIONS.md, and this file; confirm branch HEAD
matches origin (push races possible). Parallel-safe alternatives if #9
is taken: #8 MATH-02, #10 MATH-04, #11 CHEM-01, #14 CHEM-04, #16 DATA-01,
#32 TEST-01. After #9: #23 MODEL-03 becomes dependency-ready (it consumes
HandleSelectionOutput.handle_embeddings / pooled_handle_context as its
handle-side surface — see syntree/models/handle_head.py).
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
  New pinned names from #22: module `syntree/models/handle_head.py`, tests
  `tests/test_handle_selection.py`.
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
- The agency-enabled handle-selection head (#22, D-V2-010):
  `syntree/models/handle_head.py` (`HandleSelectionHead`, two-stage
  cross-attention over the active handle pool with the equivariant
  directional invariant `<v_atom, x_handle − centroid>`, per-graph
  segment log-softmax over exactly the active handle rows, `HandleSelectionOutput`
  with the #23-facing `handle_embeddings` / `pooled_handle_context`,
  fail-closed validation incl. the cross-field atom↔graph check and the
  legacy 67-D descriptor ban; frozen profile = 224,896 params); 44 tests
  in `tests/test_handle_selection.py` pin the contract, including the
  real `AssemblyEnvironment.active_handles()` menu integration (rows map
  1:1; argmax row ⇒ legal `AssemblyAction.handle_atom_idx`). The 64-D
  descriptor *producer* is owned by #12/#40; sampling/greedy utilities by
  #25. Do not reimplement or bypass it.
- The reconciled two-level GFlowNet engine layer (#38 close, D-V2-009):
  `syntree/engine/environment.py` (discrete Assembly MDP with deterministic
  bin-centre geometry + terminal reward oracle: coarse-scan-seeded Brent
  sub-bin refinement and 50 kcal/mol/Å² scaffold-constrained MMFF94),
  `syntree/engine/rl.py` (stationary `ThreeDReward`, no batch parameter by
  construction), `syntree/engine/gflownet.py` (Trajectory Balance with the
  uniform leaf deletion backward policy, STOP backward term zero by
  proof), and `docs/architecture/MDP_STATE_SPACE.md` (the formal #38
  specification + proofs). 50 tests in
  `tests/test_gflownet_reconciliation.py` pin the contracts, including the
  enumerable toy-DAG flow accounting (P(x) = R(x)/Z, residuals < 1e-9).
  Do NOT re-litigate the state-space debate — it is closed with proofs.
- README / architecture / agent-docs structure — update in place.

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| No GPU/T4 execution, training, or HF artifact sync has occurred for V2 | #36/#43 gate | CPU sandbox cannot establish GPU claims |
| Engine soft-core LJ/Coulomb, generic junction reactions, docking score are documented *surrogate* implementations with injection points | #14/#11/eval stack | D-V2-009 scope boundaries; production modules must supersede without changing the MDP/TB contracts |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | CPU-only sandbox; the refinement/relaxation timings need a T4 run |
| Bit-for-bit determinism is per RDKit build, not cross-version | #24/#42 | Documented in D-V2-009 and MDP_STATE_SPACE.md §6 |
| Dev environment not pinned in a lockfile | #26/#43 | `pyproject.toml` declares loose floors (torch≥2.1, PyG≥2.4); pin exact versions when the trainer lands |
| Schema `handle_nodes` optional context | #25 | Policy-side extension point, deliberately outside the 12-field core; #22 consumes the pool as explicit forward arguments (descriptors/pos/graph/atom) |
| 64-D handle descriptor producer | #12/#40 | `HandleSelectionHead` consumes `[H, 64]` rows; tests use synthetic descriptors until catalog featurization lands |
| Unstamped states pass field validation (interim) | #19 | Remove the interim rule when shard producers stamp |
| Backbone RBF has no smooth cutoff envelope | #9 | Deliberate boundary (D-V2-007); #9 must apply the envelope to ALL spatial RBF projections — including the handle head's centroid-distance RBF (D-V2-010) |
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
   recorded in §1 (301/301 as of S-V2-04; the handle head adds 44 tests
   to the 257 of S-V2-03).
4. Work per the #45 §18 per-issue protocol; end with the mandatory git
   diff review, commit+push (report exact SHAs), issue update, and
   refreshed handoff.
