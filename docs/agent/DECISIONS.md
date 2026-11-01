# 3D-SynTree V2 — Durable Decision Log (clean-room era)

> One decision per entry. Never silently change a decision: supersede it
> with a new entry and update the affected issues/docs. Decisions from the
> pre-reset era (recorded on the purged branch, commits `44edb8b..6dbb66a`)
> remain historically interesting but carry **no authority**; where a
> pre-reset decision encoded validated specification knowledge, it is
> explicitly *re-adopted* below with its provenance. Pre-reset code is not
> a reference: this branch started empty (commit `075f86d`).

---

## D-V2-001 — 2026-11-01 — Clean-room reset of `3d-synV2` (owner directive)

- **Problem:** The pre-reset approach implemented #5–#7 by incrementally
  repairing the legacy V1 codebase that remained on the branch (legacy
  `syntree/` tree, 800+ V1-era tests, legacy configs/wheels), which
  contradicts the owner's directive and #45 §24: V2 is a complete,
  clean, unified re-engineering, not a patch series.
- **Decision:** On the owner's explicit instruction (2026-11-01):
  (a) purge the branch completely — all 147 tracked files removed in
  commit `075f86d`, verified empty tree on origin, Git history preserved
  for forensics only; (b) reopen #5/#6/#7 with completion statuses voided;
  (c) rebuild every component newly, carrying forward *specifications
  only* (approved requirements, acceptance criteria, validated contract
  rules, scientific knowledge, the issue roadmap, data provenance).
- **Alternatives considered:** continuing the incremental approach
  (rejected by owner); selective deletion of "legacy-looking" files
  (rejected: no inherited structure may survive); orphan-branch restart
  (rejected: keep the `3d-synV2` branch ref and its history).
- **Evidence:** owner directive message; `git ls-files` empty at
  `075f86d`; GitHub tree API shows 0 entries at that SHA; issues #5/#6/#7
  reopened with rationale.
- **Affected:** everything; all issue statuses.

## D-V2-002 — 2026-11-01 — Pre-reset validated contracts re-adopted as specifications

- **Problem:** The purge removed the files, but a small set of contract
  details recorded on the pre-reset branch were *validated specification
  knowledge* (pinned by issue acceptance criteria or by tests against
  approved behaviour), not implementation choices.
- **Decision:** Re-adopt as binding V2 specifications (sources: issue
  bodies #5–#44 + pre-reset decision records D-2-009/D-2-010/D-2-011):
  1. The state schema lives at `syntree/core/schema.py` (issue #6 body is
     authoritative over any doc).
  2. Schema strictness rules (from D-2-010): 12 required canonical tensor
     fields + `schema_version` (`SCHEMA_VERSION = 2`); explicit empty
     ligand sub-graphs at t=0; `handle_features [B, 64]` float32;
     `stop_mask [B]` float32; hard 67-D rejection with exact message
     prefix `"Legacy 67D Franken-tensors are deprecated"` taking
     precedence over missing-field diagnostics; collator-only
     normalisations (0-dim → `[1]`, 1-D `[64]` → `[1, 64]`);
     `collate_state_batch` as the single batching path materialising
     canonical `pocket_batch`/`ligand_batch`; producer-side version
     stamping in tensor form.
  3. Norm primitive signatures (issue #7 body): `safe_norm(x, dim=-1,
     keepdim=False, eps=1e-5)` and `safe_unit_vector(vec, dim=-1,
     eps=1e-5) -> (unit, dist)` with `SAFE_NORM_EPS = 1e-5`; test file
     names pinned by the issues (`tests/test_tensor_contracts.py`,
     `tests/test_mathematical_singularities.py::test_safe_norm_zero_distance`).
- **Alternatives considered:** treating all pre-reset artifacts as toxic
  and re-deriving every constant (rejected: the issue bodies themselves
  pin these values — they *are* the approved requirements); copying
  pre-reset source files (forbidden by the owner directive).
- **Evidence:** issue bodies #6/#7; re-adoption verified by the new tests.
- **Affected:** #6, #7, all downstream consumers.

## D-V2-003 — 2026-11-01 — Package layout: src-layout, package `syntree`, version 2.0.0.dev0

- **Problem:** The fresh codebase needs a packaging layout; the pre-reset
  branch used a flat `syntree/` directory at the repository root with a
  build tag-along `core/` package and vendored wheels.
- **Decision:** Modern src-layout: package sources under `src/syntree/`
  (import path `syntree.*` — issue references such as
  `syntree/core/schema.py` are import paths and remain satisfied);
  package name `syntree`; version `2.0.0.dev0` (single source:
  `syntree/__init__.py` via PEP 621 `dynamic` version); dependencies
  declared minimally and grown per issue (`torch`, `torch_geometric`
  today); no vendored wheels (environments install from upstream indexes);
  no `core/` top-level straggler package.
- **Alternatives considered:** flat layout (rejected: weaker isolation
  against accidental import of unbuilt sources); separate top-level
  packages per layer (rejected: one coherent system, one package).
- **Affected:** pyproject.toml, all modules.

## D-V2-004 — 2026-11-01 — PaiNN blocks implement the published equations

- **Problem:** #7 requires `PaiNNInteraction`/`PaiNNMixing` to route all
  norms through the safe primitives. Two equation variants exist in the
  wild: the published PaiNN equations (Schütt, Unke & Gastegger, ICML
  2021, arXiv:2102.03150, Eqs. 7–10) with shared scalar transforms φ(s_j)
  *and* radial filter banks W(r_ij) gating the messages, and the simpler
  variant (used by some reference implementations and by the pre-reset
  code) with radial-only filters.
- **Decision:** Implement the **published equations faithfully**:
  message (Eq. 7–8): `∆s^m = Σ_j φ_s(s_j)∘W_s(r_ij)`,
  `∆v^m = Σ_j v_j∘φ_vv(s_j)∘W_vv(r_ij) + Σ_j φ_vs(s_j)∘W_vs'(r_ij)·r̂_ij`
  with a shared atomwise φ network (split afterwards) and bias-free
  radial filter banks; mixing (Eqs. 9–10):
  `∆s^u = a_ss(s,‖Vv‖) + a_sv(s,‖Vv‖)·⟨Uv,Vv⟩`,
  `∆v^u = a_vv(s,‖Vv‖)·Uv` with separate U/V linear maps and a shared
  scaling network over `[s, ‖Vv‖]`. Both blocks are residual updates
  (each returns features + delta); a PaiNN *layer* composes
  message-then-update (owned by #21). The singular paper direction
  `r_ij/‖r_ij‖` is replaced by `safe_unit_vector` (the #7 mandate).
  Equation form verified against the arXiv PDF (fetched 2026-11-01).
- **Alternatives considered:** the simplified radial-only variant
  (rejected: less expressive than the published architecture and no issue
  pins the simplification); adding cross-product geometric terms from the
  *output* gated-equivariant block (rejected: out of message/mixing scope).
- **Evidence:** arXiv:2102.03150v4 Eqs. 7–10 (extracted text on file in
  the session log); equivariance/singularity tests in
  `tests/test_mathematical_singularities.py`.
- **Affected:** #7, #21 (backbone composes these blocks), #32/#33.

## D-V2-005 — 2026-11-01 — No mixed-precision machinery in V2 by construction

- **Problem:** The pre-reset branch carried fp16 activation-clamp
  machinery in the equivariant path as "interim" debt for #10.
- **Decision:** The clean-room V2 contains **no** fp16/bf16 clamps,
  `nan_to_num`, or AMP anywhere: the production path is pure FP32
  (#45 §5.1), and genuinely poisoned inputs fail loudly (#45 §8). #10
  adds the *fail-closed enforcement* layer (device/dtype guards,
  configuration validation) rather than removing legacy clamps, because
  none exist.
- **Alternatives considered:** porting the interim clamps (rejected:
  carries V1 debt into a clean-room build).
- **Affected:** #7, #10, #26.

## D-V2-006 — 2026-11-01 — Test-suite discipline for the clean-room build

- **Problem:** The pre-reset suite carried 800+ V1-era tests (and 12
  documented pre-existing failures). Carrying them over is impossible
  (branch emptied) and undesirable (they encoded legacy behaviour).
- **Decision:** The V2 suite starts empty and grows exclusively from
  issue acceptance criteria; test file names pinned by issue bodies are
  honoured exactly (`tests/test_tensor_contracts.py`,
  `tests/test_mathematical_singularities.py`). Every issue's closure
  comment must state exact test commands and pass counts. No test may be
  deleted to make a run green; failures are fixed or the issue stays open.
- **Affected:** #32 (Suite A aggregation), #33 (Suite B), all issues.

## D-V2-007 — 2026-11-01 — Backbone RBF carries no smooth cutoff envelope (owned by #9)

- **Problem:** Issue #21 specifies "8 equivariant layers with 20 Gaussian
  RBFs" without an envelope, while #45 §8.3 and the debt table assign
  "cutoff envelope maths" to #9 MATH-03 ("Multiply all spatial distance RBF
  projections by this envelope"). Implementing the envelope inside #21 would
  duplicate #9's acceptance-tested mathematics and blur the issue boundary.
- **Decision:** `GaussianRadialBasis` (`syntree/models/bipartite.py`) emits
  the raw Gaussian expansion over centers `[0, r_cut]` with
  `sigma = r_cut / (num_radial - 1)` and **no** cosine envelope. When #9
  lands, it must apply `continuous_cosine_envelope` to *all* spatial RBF
  projections — the backbone's `GaussianRadialBasis` **and** the
  cross-attention RBF — in one authoritative place. Until then, edge
  truncation at `r_cut` is the hard graph-construction threshold (pinned by
  test: at `d = r_cut` the last Gaussian channel is exactly 1.0).
- **Alternatives considered:** implementing the envelope now inside the
  backbone (rejected: duplicates #9's mandate and its acceptance criteria);
  a smooth distance-weighted graph cut-off on edges (rejected: not pinned by
  any issue; changes the radius-graph semantics).
- **Evidence:** issue #21 body ("20 Gaussian RBFs", no envelope wording);
  issue #9 body (envelope formula + "multiply all spatial distance RBF
  projections"); `tests/test_bipartite_painn.py::
  TestGaussianRadialBasis::test_at_cutoff_last_channel_is_one`.
- **Affected:** #21, #9, #23, #32.

## D-V2-008 — 2026-11-01 — Backbone accepts FP32 (production) and FP64 (local verification) only

- **Problem:** The schema and production path are strictly FP32 (D-V2-005),
  but equivariance verification at float64 precision is standard numerical
  practice and was already used by the #7 block tests. The backbone needed
  an explicit dtype policy rather than an implicit one.
- **Decision:** `BipartitePaiNN` parameters default to FP32 (the production
  path on Colab/Kaggle T4). `forward` accepts exactly two input dtypes:
  `float32` (production) and `float64` (local numerical verification only),
  and the input dtype must **match the module parameter dtype** (cast the
  module with `.to(torch.float64)` to verify locally). FP16/BF16 and every
  other dtype raise `TensorContractError` fail-closed. No float64 training
  claim exists anywhere; the T4 production path remains pure FP32.
- **Alternatives considered:** accepting any dtype with silent casting
  (rejected: silent precision changes violate the fail-closed constitution);
  strictly FP32-only inputs (rejected: would weaken local equivariance
  verification to float32 tolerances).
- **Evidence:** `tests/test_bipartite_painn.py::TestFailClosed` (dtype
  matrix), `TestSE3InvarianceEquivariance` (float64 tolerances 1e-9;
  measured worst errors 2.98e-13 / 4.62e-14 over 36 rigid-motion trials).
- **Affected:** #21, #10 (the fail-closed precision engine enforces the
  production side), #26.
