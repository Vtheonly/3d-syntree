# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-01, session S-V2-06 close (#23 MODEL-03
implemented, verified, and closed — decision D-V2-012).

## 1. Continuation instruction

```
Next session: implement #8 MATH-02 (stable analytic Amos-ratio autograd
for the von Mises circular torsion objective) — it is dependency-ready
(deps: #5 only), it is the smallest remaining blocker of #24 MODEL-04 on
the critical path (#24 → #25 → #26 → ...), and the S-V2-06 dependency
review recommends it before #11 CHEM-01 (the other #24 blocker, a
larger unit: 32-reaction registry + representative reagent/product
tests per #45 §9.1). Read issue #8 in full first; module path per the
STATUS table: syntree/models/ (von Mises mathematics; #45 §8.2 pins the
bounded Amos approximation A(k) ≈ kappa / (0.5 + sqrt(kappa^2 + 2.25))
and demands comparison against a trusted numerical reference over the
relevant range, forward and backward, plus large/small-kappa behavior).
#24 becomes dependency-ready once BOTH #8 and #11 are closed. Last
verified code state: S-V2-06 commits on 3d-synV2 (suite 405/405; HEAD
of this session's docs commit). #5, #6, #7, #9, #21, #22, #23, #38 are
implemented from scratch, locally verified, and CLOSED — do NOT redo
the purge, the schema, the norms, the cosine cutoff envelope, the
backbone, the GFlowNet state-space reconciliation, the handle-selection
head, or the cross-attention decoder. Before coding, re-read issue #8
plus D-V2-004/D-V2-005 (numerical policy: no clamps, fail loudly) and
syntree/models/equivariant.py (the #7 precedent for math-primitive
modules and their test style); confirm branch HEAD matches origin (push
races possible). Parallel-safe alternatives if #8 is taken: #10 MATH-04,
#11 CHEM-01, #14 CHEM-04, #16 DATA-01, #32 TEST-01. #36 OPS-01 remains
gated on user authorization (Colab SSH). For #24 (once ready): consume
DecoderOutput.context [B, 256] (syntree/models/policy.py) + the per-row
HandleSelectionOutput.handle_embeddings; NOTE the closing LayerNorm —
each context row sums to the constant sum(beta), so head losses must be
per-component weighted (context.sum() has exactly zero gradient; see
tests/test_cross_attention_decoder.py header note and D-V2-012).
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
  binding (D-V2-002). Pinned so far: `syntree/core/schema.py` + tests
  `tests/test_tensor_contracts.py` (#6), `syntree/models/equivariant.py` +
  `tests/test_mathematical_singularities.py` (#7),
  `syntree/models/bipartite.py` + `tests/test_bipartite_painn.py` (#21),
  `syntree/models/handle_head.py` + `tests/test_handle_selection.py`
  (#22), `syntree/models/policy.py` (#23/#25; #23 delivered
  `CrossAttentionDecoderLayer` / `CrossAttentionDecoder` /
  `DecoderOutput` there with tests `tests/test_cross_attention_decoder.py`)
  and `syntree/models/cutoff.py::continuous_cosine_envelope` with tests
  `tests/test_cosine_cutoff.py` (#9).
- Backbone contract highlights (D-V2-007/D-V2-008/D-V2-011):
  `GaussianRadialBasis` emits the 20-Gaussian expansion MULTIPLIED by the
  continuous cosine cutoff envelope — integrated inside the producer,
  unconditionally (no raw expansion exists anywhere in V2; #23 resolved
  D-V2-011's conditional decoder clause by routing NO distances at all —
  decision D-V2-012); the backbone accepts FP32 (production) and
  FP64 (local verification only, must match module dtype);
  `build_union_radius_graph` is the one authoritative edge construction —
  downstream heads must reuse it, not re-implement it.
- Decoder contract highlights (D-V2-012, `syntree/models/policy.py`):
  dense interface `forward(query [B, 1, 256], memory [B, N, 256],
  key_padding_mask [B, N] bool) -> DecoderOutput` with `context
  [B, 256]`; V2-native `forward_state(backbone, handle)` builds the query
  from `pooled_handle_context` + `graph_context` + `pocket_context` (one
  authoritative embedding chain — never a second pocket/handle encoder);
  padded columns get exactly-zero attention and outputs are
  bit-identical under pad junk; fully-padded graphs fail closed; the
  module projects no spatial distances (any future distance bias MUST
  route through `GaussianRadialBasis`); frozen profile 1,219,328 params;
  **closing LayerNorm caveat for #24/#25**: each context row sums to the
  constant sum(β), so plain `context.sum()` has zero gradient — head
  losses must be per-component weighted.

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
- The continuous cosine cutoff envelope (#9, D-V2-011):
  `syntree/models/cutoff.py::continuous_cosine_envelope` (the issue-#9
  pinned form, fail-closed) integrated INSIDE
  `syntree/models/bipartite.py::GaussianRadialBasis.forward` — the one
  authoritative place for ALL spatial distance RBF projections (backbone
  edge distances; the handle head's [0, 12] Å centroid distances; any
  future decoder RBF). Enveloped behaviour is bitwise-zero features
  beyond the cutoff and C¹ taper at the boundary; parameter-free, frozen
  profiles unchanged. 46 tests in `tests/test_cosine_cutoff.py` plus the
  four rewritten legacy pins (two in `tests/test_bipartite_painn.py`,
  `test_radial_basis_is_enveloped` in `tests/test_handle_selection.py`)
  pin the contract. Do not re-implement, bypass, or add a raw-expansion
  escape hatch.
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
  segment log-softmax over exactly the active handle rows,
  `HandleSelectionOutput` with the #23-facing `handle_embeddings` /
  `pooled_handle_context`, fail-closed validation incl. the cross-field
  atom↔graph check and the legacy 67-D descriptor ban; frozen profile =
  224,896 params); 44 tests in `tests/test_handle_selection.py` pin the
  contract, including the real `AssemblyEnvironment.active_handles()`
  menu integration (rows map 1:1; argmax row ⇒ legal
  `AssemblyAction.handle_atom_idx`). The 64-D descriptor *producer* is
  owned by #12/#40; sampling/greedy utilities by #25. Do not reimplement
  or bypass it.
- The two-layer stacked cross-attention decoder (#23, D-V2-012):
  `syntree/models/policy.py` (`CrossAttentionDecoderLayer` with the
  issue-pinned MHA(256/8/0.1) + FFN 256→512→SiLU→Dropout→512→256 +
  pre-LN residuals; `CrossAttentionDecoder` with the dense
  `[B, 1, 256] × [B, N, 256] + key_padding_mask → [B, 256]` interface
  and `forward_state(backbone, handle)`; `DecoderOutput`; frozen
  profile = 1,219,328 params; SE(3)-invariant end-to-end 8.882e-16 /
  4.768e-07); 58 tests in `tests/test_cross_attention_decoder.py` pin
  the contract (exact masking, bit-identical under pad junk, fail-closed
  matrix, permutation/batch-composition invariance, gradient
  connectivity without cross-graph leaks). The decoder projects NO
  spatial distances (D-V2-012) — do not add a raw distance expansion; a
  future distance bias must route through `GaussianRadialBasis`. The
  closing LayerNorm makes each `context` row sum to `sum(β)` — head
  losses must be per-component weighted. Do not reimplement or bypass it.
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
| Backbone ≥1-pocket-atom-per-graph rule is stricter than schema | #21 | Documented in-module; a schema-level per-graph pocket rule may be worth adopting at #18/#19 |
| Activation-checkpointing T4 benefit unmeasured | #43 | Capability implemented + correctness-tested on CPU only |
| `num_graphs` for non-Batch inputs is derived (max batch index + 1) | #21 | A Batch's `num_graphs` is trusted; index-range + pocket-coverage checks catch inconsistency |
| `GaussianRadialBasis` buffers carry construction-dtype rounding | #21 (pre-existing) | `centers`/`_scale` are created in the module dtype (fp32 default) and merely cast by `.to(float64)`; fp64 verification therefore sees fp32-rounded buffers. Harmless for all pinned contracts (values shift ~1e-7 relative); analytic tests must derive references from the module's own buffers (see `tests/test_cosine_cutoff.py`). If bit-exact fp64 buffers are ever wanted, construct the module under `torch.set_default_dtype(torch.float64)` — revisit only if an issue demands it |
| fp32 `0.5·(cos+1)` cancellation floor near the envelope seam | #9 (documented, benign) | Values ≤ ~6e-8 within ~1e-4 of the cutoff in fp32 (≤1e-15 fp64); physically negligible (true values there are ~1e-10); pinned with appropriate tolerances in `tests/test_cosine_cutoff.py` |

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
   recorded in §1 (405/405 as of S-V2-06; the cross-attention decoder
   suite adds 58 tests to the 347 of S-V2-05).
4. Work per the #45 §18 per-issue protocol; end with the mandatory git
   diff review, commit+push (report exact SHAs), issue update, and
   refreshed handoff.
