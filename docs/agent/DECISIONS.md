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

## D-V2-009 — 2026-11-01 — Two-level GFlowNet reconciliation: discrete MDP + terminal reward oracle, uniform-leaf backward policy, stationary reward (owner task label "D21")

- **Problem:** The pre-reconciliation backlog contained three theoretical
  contradictions.  (1) *Geometry:* #24 mandates a discrete 36-bin torsion
  action for dimensionless Trajectory Balance, while #42 as originally
  scoped continuous sub-bin refinement + MMFF94 relaxation as an
  environment-side mutation of intermediate states — the resulting
  `s_{t+1}` would not be the state the policy transitioned to, a
  zero-probability continuous shift under a categorical measure, destroying
  flow conservation.  (2) *Backward:* #29's original `P_B = 1` assumption
  fails for multi-parent states (symmetric molecules, alternative reaction
  orders) while #38 demanded the semantics be proven first.  (3) *Reward:*
  #30's original batch-level Tanimoto diversity term made `R(x)` depend on
  rollout-batch siblings — a non-stationary target that destabilises
  `log Z` (its own addendum had already superseded the formula but no
  implementation existed).
- **Decision (owner-directed reconciliation task, 2026-11-01):**
  1. **Two-level architecture.**  The GFlowNet MDP state is strictly
     discrete; during rollout 3D coordinates are generated deterministically
     from the exact bin centres `phi_k = -pi + (k-0.5)*2*pi/36` with
     resonance-locked amide/ester junctions snapped planar trans.  *All*
     continuous physics (sub-bin refinement ±5°, scaffold-constrained
     MMFF94 with 50 kcal/mol/Å² seed restraints) runs **only upon STOP**
     as the terminal reward oracle, on a deep copy.  #42 is thereby
     re-scoped as "Terminal Sub-Bin Geometry Refinement".
  2. **Uniform Leaf Deletion backward policy.**  States are
     order-quotiented discrete assembly objects (canonical derivations);
     `P_B(s'|s) = 1/|RemovableLeafSynthons(s)|` over non-seed fragments
     with exactly one junction bond.  The STOP step's backward term is
     exactly zero *by proof* (absorbing uniqueness).  Specification,
     proofs, and the backward-path-measure identity:
     `docs/architecture/MDP_STATE_SPACE.md`.
  3. **Stationary reward.**  `R(x) = exp((S_dock + S_contact + S_key +
     S_props + S_vol)/T)` bounded below by `1e-4` — a function of the
     terminal molecule and the fixed pocket context only; the compute
     signature carries no batch parameter at all.  Diversity is evaluation
     telemetry, never a loss term.
  4. **Sub-bin optimiser naming (per #42's accuracy mandate):** the
     refinement is a *coarse-scan-seeded bracketed Brent optimiser* — a
     deterministic 9-point grid scan selects the bracket, then
     `scipy.optimize.minimize_scalar(method='bounded')` converges within
     it; the hybrid exists because the soft-core objective can be
     multimodal over the 10° interval.  It is verified to be at least as
     good as a dense 200-point reference scan on the same sequential
     baseline.
- **Alternatives considered:** continuous torsion densities summed with
  discrete probabilities (rejected: breaks the measure and the TB units —
  the pre-V2 κ→∞ failure); executing MMFF on intermediate states (rejected:
  the transition must be governed by the policy); `P_B = 1` with
  derivation-tree states (rejected: N! multiplicity inflation of
  multi-order molecules); `P_B = 1` with graph states (rejected: violates
  flow conservation at every multi-parent convergence); keeping the batch
  Tanimoto term (rejected: non-stationary target; already superseded by
  #30's addendum); plain Brent over the full ±5° interval (rejected:
  locks into local basins on the multimodal landscape).
- **Evidence:** `tests/test_gflownet_reconciliation.py` — 50 tests: pinned
  determinism (bit-for-bit, fresh catalogs), terminal-oracle isolation
  (counters, fail-closed non-terminal refusal, state immutability, rigid
  refinement preserving covalent geometry), reward stationarity
  (batch-composition invariance to 8 decimals / bit-identical, signature
  inspection, bounds, fail-closed), leaf semantics (branched periphery,
  symmetric convergence, renumbering invariance, path measure = 1),
  **enumerable toy-DAG flow accounting** (19 quotient states / 31
  trajectories: every TB residual < 1e-9 through the real engine with
  `log Z` set to the true total flow; terminal outcome probabilities equal
  `R(x)/Z` to 1e-9; two-order states split mass into exact halves), amide
  resonance snap + post-MMFF planarity, Brent-vs-dense-scan, and the
  pinned TB equation / STOP-zero-term / fail-closed contracts.  Full
  suite: 257/257.
- **Affected:** #24 (geometry contract implemented; heads remain open),
  #29 (TB mathematics + backward semantics pinned; trainer remains open),
  #30 (stationary reward core implemented; production physics remains
  open), #38 (closed — formal spec + proofs + tests), #42 (re-scoped and
  core implemented; T4 benchmarking + #11/#13/#14 integration remain open),
  #22–#25 (consume the environment's contracts), #45 §12.3/§12.4
  (satisfied by construction).
- **Scope boundaries recorded:** the engine's soft-core LJ / Coulomb
  surrogate, the generic junction-reaction registry, and the docking
  surrogate are *documented interim implementations with injection
  points*; #11/#13/#14 own the production chemistry and must supersede
  them without changing the MDP/TB contracts above.  Bit-for-bit
  determinism is claimed per RDKit build, not across versions.

## D-V2-010 — 2026-11-01 — Handle-selection head architecture: two-stage cross-attention over the active handle pool with equivariant directional invariants

- **Problem:** Issue #22 MODEL-02 requires `P(handle_t | s_t)` over the
  active handle pool — closing legacy "Vulnerability 3" (the pre-V2
  environment hardcoded `handles[0]`, so the agent had no spatial agency
  over where to grow). The head must consume the #21 backbone without a
  second pocket encoder, use handle invariant descriptors `[H, 64]` and
  positions relative to pocket centroids, and normalize over the *actually
  active* handle set. The open design question was how to make the
  centroid-relative Cartesian positions SE(3)-safe while keeping genuine
  3D directional agency (not just a radial distance).
- **Decision:** `HandleSelectionHead` (in
  `syntree/models/handle_head.py`, frozen profile: backbone d = 256, head
  width 128, 20 raw RBFs on [0, 12] Å, 224,896 parameters) builds a
  per-handle invariant embedding from four additive contributions: the
  64-D descriptor MLP, the backbone scalar of the attachment atom, the
  raw Gaussian RBF of the handle's distance to its graph's pocket
  centroid, and — the key architectural choice — the invariant
  directional features `<v_atom, x_handle − centroid_g>` obtained by
  contracting the **equivariant backbone vector channel** with the
  centroid-relative offset. A per-graph query (pooled ligand summary +
  pooled pocket context) then (stage 1) cross-attends over that graph's
  handle keys — the attention weights are themselves a valid per-graph
  distribution — pooling an attention-weighted handle context, and
  (stage 2) an additive Bahdanau-style scorer (query + key + pooled
  context) produces logits that are normalized by a per-graph
  `log_softmax` over exactly the active handle rows. The output exposes
  the auditable interface (logits / log_probs / probs / attention /
  entropy / counts) plus the handle embeddings and pooled handle context
  as the #23 decoder-facing surface. Inputs are fail-closed validated
  (FP32/FP64-only dtype policy per D-V2-008, device, shapes, index
  ranges, legacy 67-D rejection, and a cross-field check that each
  handle's attachment atom belongs to the handle's own graph).
- **Alternatives considered:** feeding the raw centroid-relative offset
  through ordinary linear layers (rejected: a Cartesian vector in scalar
  channels breaks invariance — the exact Franken-tensor mistake of #6);
  radial distance only (rejected: no directional agency — "open cavity vs
  wall" is precisely a direction question); pooling the equivariant
  vectors then contracting (rejected: pooled vector semantics are
  ill-motivated; the per-atom contraction is exact and local);
  softmax-attention-weights as the *final* law (rejected: a rank-1
  query·key score is too weak for competing handles; the two-stage design
  keeps the cross-attention character while letting every score see the
  competitor context); per-graph Python loops for normalization (rejected:
  the scatter-based segment log-softmax is vectorized, on-device, and
  numerically stable).
- **Evidence:** `tests/test_handle_selection.py` — 44 tests: the pinned
  #22 acceptance (valid normalized laws for H > 1, single-handle P = 1
  exactly, identical-handle uniform split with entropy ln 2, FP32 and
  FP64 paths); rigid-motion invariance worst error 8.3e-16 (float64, 6
  random rotations+translations of pocket+ligand+handle) and 5.96e-08
  (float32 production path); coincident-centroid finiteness (safe norm at
  r = 0); handle-row permutation equivariance, pocket-atom permutation
  invariance (end-to-end through the backbone), ligand-node permutation
  with consistent remap, batch-composition invariance; real
  `AssemblyEnvironment` integration at t = 0 (3-handle tri-chloro seed)
  and on grown intermediates — rows map 1:1 onto `active_handles()` and
  the argmax row's atom index is a legal `AssemblyAction.handle_atom_idx`;
  the log-probability feeds `TrajectoryStep.action_log_prob` directly;
  gradient connectivity into the checkpointed backbone and to positions;
  no gradient leaks across graphs; frozen profile pinned at 224,896
  parameters; audit interface; bit-for-bit determinism. Full suite:
  301/301.
- **Affected:** #22 (closed — all acceptance criteria verified), #23
  (consumes `HandleSelectionOutput.handle_embeddings` /
  `pooled_handle_context` as its handle-side surface), #24/#25 (the
  handle component of `log P_F` is now available for the hierarchical
  heads and the policy assembly), #12/#40 (own the 64-D descriptor
  *producer* consumed here), #9 (still owns the cosine envelope for the
  head's centroid-distance RBF, as for all spatial RBF projections).
- **Scope boundaries recorded:** the descriptor featurization (64-D) is
  not implemented here — synthetic descriptors are used in tests and the
  catalog issues own production featurization; `handle_pos` is documented
  (and integration-tested) as the reacting atom's conformer position but
  is deliberately *not* hard-equal-checked against the node position, so a
  future directional growth-point offset remains possible; sampling /
  greedy selection utilities are deferred to #25.

## D-V2-011 — 2026-11-01 — Cosine cutoff envelope integrated inside `GaussianRadialBasis` (the one authoritative place)

- **Problem:** Issue #9 MATH-03 mandates the `C¹`-continuous cosine
  cutoff envelope multiplied into *all* spatial distance RBF projections
  (`radial(d) = Gaussian(d) ⊙ f_cut(d)`), closing legacy "Vulnerability 7"
  (clamped raw RBFs gave every spectator from `r_cut + ε` to 50 Å an
  identical non-zero bias feature). Two integration designs existed:
  applying the envelope at each consumer's projection site (backbone
  forward, handle-head forward, future #23 decoder), or integrating it
  once inside the shared RBF producer. D-V2-007 had deferred the
  envelope from #21 with the note "#9 will integrate it in one
  authoritative place", which required an interpretation.
- **Decision:** Implement `continuous_cosine_envelope(dist, cutoff)`
  exactly in the issue's pinned form in a new dedicated module
  `syntree/models/cutoff.py` (fail-closed: non-tensor / non-floating /
  non-finite inputs and non-finite-or-non-positive cutoffs raise), and
  integrate it *inside* `GaussianRadialBasis.forward` — unconditionally,
  no opt-out flag. Rationale: both existing spatial RBF consumers (the
  #21 backbone edge-distance expansion and the #22 handle head's
  [0, 12] Å centroid-distance expansion) construct
  `GaussianRadialBasis` directly, so envelope-inside-the-producer covers
  every projection site in one place and makes the Vulnerability-7 class
  of bug structurally impossible to reintroduce (a future consumer
  cannot forget the envelope). The #23 decoder must likewise route its
  spatial RBF through `GaussianRadialBasis`. The envelope is
  parameter-free: frozen profiles are unchanged (backbone 5,999,104;
  handle head 224,896 — re-pinned by the existing tests).
- **Test-contract consequences (sanctioned by #45 §24.3 and D-V2-007's
  explicit "until #9 lands" clause):** the four pre-#9 raw-RBF pins were
  rewritten 1:1 to the enveloped contract, none deleted:
  `test_at_cutoff_last_channel_is_one` →
  `test_at_cutoff_features_vanish_and_beyond_is_bitwise_zero`
  (`tests/test_bipartite_painn.py`), `test_values_bounded` →
  `test_values_bounded_and_nonnegative`, `test_peaks_at_centers` →
  `test_peaks_at_centers_carry_the_envelope`, and
  `test_radial_basis_is_the_raw_gaussian_expansion` →
  `test_radial_basis_is_enveloped` (`tests/test_handle_selection.py`).
- **Numerical facts pinned by the new suite:** beyond the cutoff the
  features are *bitwise* zero (the `torch.where` branch) including the
  issue's acceptance probe `d = r_cut + 1e-4`; at `d = r_cut` the
  envelope evaluates to exactly `0.0` on the reference CPU platforms in
  both dtypes (elsewhere ≤ 1e-12); `f_cut(0) = 1` exactly so the first
  channel at `d = 0` remains exactly 1.0; the derivative decays
  linearly to zero as `d → r_cut⁻` (measured summed-gradient decay
  4.3e-2 → 3.5e-5 for offsets 1e-1 → 1e-4, matching
  `0.5 (π/r_cut)² δ`); the fp32 `0.5·(cos+1)` cancellation floor near
  the seam is ≤ ~6e-8 and the fp64 one ≤ ~1e-16 (documented in the
  tests; physically negligible). The envelope is a radial function, so
  SE(3) invariance/equivariance is unaffected (the full #21 suites
  re-run green against the enveloped path; a compact invariance test
  also lives in the new file).
- **Alternatives considered:** applying the envelope at each consumer
  site (rejected: duplicated application sites — exactly the
  forget-to-apply failure mode #9 exists to prevent); an
  `apply_envelope=False` escape flag (rejected: no V2 requirement needs
  a raw expansion, and a flag invites silent contract drift);
  implementing the envelope in `syntree/models/policy.py` as literally
  worded in the #9 body (rejected: that module is #23/#25's pinned
  deliverable and does not exist yet; the envelope must precede and
  outlive it); a smooth distance-weighted graph cutoff on edges
  (rejected: changes radius-graph semantics, out of #9 scope).
- **Evidence:** `tests/test_cosine_cutoff.py` — 46 tests: both pinned
  issue acceptance criteria; the analytic closed forms (value,
  derivative, reflection identity `f(d)+f(r_cut−d)=1`, monotonicity,
  quadratic near-cutoff decay, linear derivative decay); the
  product-rule analytic radial-derivative match in fp64 (reference
  derived from the module's own buffers so the pinned quantity is the
  envelope mathematics, not buffer rounding); bitwise determinism;
  dtype preservation; the fail-closed matrix (non-tensor, non-floating,
  NaN/Inf distances; zero/negative/NaN/Inf cutoffs); the enveloped-RBF
  integration (peaks carry the envelope, sup-norm ≤ envelope, no
  parameters added, state-dict roundtrip, unchanged fail-closed
  behaviour); the handle-head [0, 12] Å basis beyond-cutoff bitwise
  zeros (Vulnerability-7 regime dead); a backbone smoke test and a
  compact fp64 scalar-invariance / vector-equivariance check through
  the enveloped backbone. Full suite: **347/347**
  (`python -m pytest tests/ -q`, CPU sandbox, Python 3.12.16,
  torch 2.15.0+cpu, PyG 2.8.1).
- **Affected:** #9 (closed — all acceptance criteria verified), #21
  (`GaussianRadialBasis` semantics updated; frozen profile and
  equivariance contracts unchanged), #22 (centroid-distance RBF now
  enveloped at 12 Å; frozen profile unchanged), #23 (now
  dependency-ready; must route its spatial RBF through
  `GaussianRadialBasis` rather than re-implementing an expansion),
  #32/#33 (suite aggregation), D-V2-007 (superseded in its
  envelope-deferral clause; its "one authoritative place" intent is
  realized by this decision).
