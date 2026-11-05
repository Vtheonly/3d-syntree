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

## D-V2-012 — 2026-11-01 — Cross-attention decoder: per-graph handle query from the #22 surface, dense padded memory, no new spatial-RBF site

- **Problem:** Issue #23 pins the decoder's layer structure exactly
  (`nn.MultiheadAttention(256, 8, dropout=0.1)`, FFN
  `256→512→SiLU→Dropout→512→256`, pre-LayerNorm residuals on both
  sub-layers, ×2 stacked layers) and its dense acceptance shapes
  (`[B, 1, 256]` query × `[B, N, 256]` memory → `[B, 256]`, padded atoms
  masked via `key_padding_mask`), but three integration questions were
  open: (1) what the "handle query" is built from, given the #22 head
  emits both per-row handle embeddings `[H, 128]` and a pooled handle
  context `[B, 128]` while the decoder is 256-wide; (2) whether the
  decoder projects its own spatial distance RBF (D-V2-011 anticipated
  "the decoder's spatial RBF" but nothing in #23 pins one); (3) how the
  flat PyG-style union memory (`[N, d]` + `node_batch`) becomes the
  issue's dense padded `[B, N, d]` layout.
- **Decision:**
  1. **Per-graph query, `Lq = 1`** (the pinned acceptance shape):
     `q_g = W_handle(pooled_handle_context_g) + W_graph(graph_context_g)
     + W_pocket(pocket_context_g)` — the #22 pooled context is the
     attention-weighted aggregate of the authoritative handle embeddings
     (one handle encoder, D-V2-010), and the #21 pooled contexts are the
     backbone summaries (one pocket encoder). The per-row
     `handle_embeddings` stay available to the #24 heads for
     per-candidate scoring; the decoder does not re-encode them. The
     stack pins `Lq = 1`; the layer primitive accepts general `Lq`
     (pinned by test).
  2. **No new spatial-RBF site.** The decoder consumes no Cartesian
     coordinates: every spatial quantity it sees already passed through
     the enveloped `GaussianRadialBasis` upstream (backbone edge
     distances inside #21; centroid distances + directional invariants
     inside #22). D-V2-011's conditional rule is satisfied vacuously
     *and* structurally (audit flag `projects_spatial_distances: False`;
     no radial-basis submodule — pinned by test). Any future
     query-to-memory distance bias (e.g. #24/#39) MUST route through
     `GaussianRadialBasis`.
  3. **Order-preserving per-group padding** of the flat union memory
     into `[B, N_max, d]` with a bool `key_padding_mask` (True = padded,
     zero-filled slots). Padded columns receive exactly-zero attention
     in every layer/head/graph (train and eval); outputs are
     bit-identical under arbitrary padded-slot values; fully-padded
     graphs fail closed (softmax over an empty key set is undefined —
     impossible for valid backbone outputs, which guarantee ≥ 1 pocket
     atom per graph).
  4. **Final LayerNorm** after the two pre-LN layers (standard pre-LN
     stack practice; the issue pins the per-layer structure, the closing
     norm is the stack's). Consequence documented in the tests: each
     output row sums to the constant `sum(β)`, so `context.sum()` has
     exactly zero gradient — gradient tests (and any real head loss)
     use per-component weighted objectives.
- **Alternatives considered:** per-handle queries `[H, 256]` (rejected:
  contradicts the pinned `[B, 1, 256]` acceptance interface and would
  make the output per-handle rather than per-graph; #24 can consume the
  per-row embeddings directly); an RBF query-to-node attention bias
  (rejected: not pinned by any issue — speculative scope; both query and
  memory are already 3D-contextualized through enveloped producers, so
  content-based attention is 3D-aware on both sides); post-LN residuals
  (rejected: the issue pins pre-LN); no closing norm (rejected: the
  pre-LN residual stream would hand downstream heads unnormalized
  scale); accepting general `Lq` at stack level (rejected: the pinned
  shape is the contract; `Lq = 1` is the decoder's use); float
  `key_padding_mask` (rejected: bool is the exact PyTorch semantic;
  anything else invites silent mask misinterpretation).
- **Evidence:** `tests/test_cross_attention_decoder.py` — 58 tests: both
  pinned acceptance criteria (`[B, 1, 256]` × `[B, N, 256]` → `[B,
  256]`; padded columns exactly zero in every layer/head/graph, outputs
  bit-identical under pad junk, attention rows sum to 1 over real
  tokens, train-mode dropout keeps pads zero); the issue-pinned
  structure (MHA config, FFN widths/activation, pre-LN residual identity
  under zeroed sub-layers — which also distinguishes pre-LN from
  post-LN); padding roundtrip / order / mask exactness; the fail-closed
  matrix (dtype policy D-V2-005/008, `Lq ≠ 1`, non-bool mask,
  fully-padded graph, NaN/Inf, shape/batch/width mismatches,
  cross-field graph-count consistency, poisoned backbone scalar); SE(3)
  invariance end-to-end **8.882e-16** (FP64) / **4.768e-07** (FP32
  production) through the real backbone + handle head + decoder;
  translation, pocket/ligand permutation (with handle remap),
  memory-token permutation, and batch-composition invariance; gradient
  connectivity through the activation-checkpointed backbone and the
  handle head with **no cross-graph leaks**; frozen profile
  **1,219,328 parameters**; audit interface; eval determinism and
  state-dict roundtrip bit-exact; attention-footprint arithmetic
  (614,400 floats ≈ 0.59 MB at B = 12, N = 800 — below the issue's
  20 MB budget; a calculation, not a T4 measurement). Full suite:
  **405/405** (`python -m pytest tests/ -q`, CPU sandbox, Python
  3.12.16, torch 2.15.0+cpu, PyG 2.8.1).
- **Affected:** #23 (closed — all acceptance criteria verified), #24/#25
  (consume `DecoderOutput.context [B, 256]` plus the per-row
  `HandleSelectionOutput.handle_embeddings`), #39 (its addendum to #23
  allows consuming the decoder's pocket-conditioned context — exactly
  `context`), #32/#33 (suite aggregation), D-V2-011 (its conditional
  "decoder spatial RBF" clause is resolved: there is none, and the
  structural guarantee is pinned).
- **Scope boundaries recorded:** the decoder is a representation module
  — no action distributions are produced here (STOP / reaction family /
  reaction / synthon / stereo / torsion are #24's heads; assembly and
  sampling utilities are #25); no T4 execution or memory measurement has
  occurred (arithmetic only; #43 owns measured profiling); the pinned
  dropout 0.1 is active in training mode only (eval is bit-deterministic).

---

## D-V2-013 — 2026-11-02 — `LogBesselI0` implements the issue-pinned branches with exact gating and fail-closed domain validation instead of the snippet's dead clamps (#8)

- **Problem:** Issue #8 pins a reference snippet for
  `LogBesselI0(torch.autograd.Function)` whose forward/backward contain
  three defensive clamps (`k.clamp(min=1e-6)` twice, `tot.clamp(min=1e-30)`)
  and no input validation.  On the valid domain (`kappa >= 0`) every one of
  those clamps is provably dead code (the power-series total is `>= 1`; the
  asymptotic branch is only evaluated at `kappa >= 12`; the Amos ratio
  `k/(0.5 + sqrt(kappa^2 + 2.25))` has denominator `>= 2.0` and is exactly
  `0` at `kappa = 0`).  The constitution (#45 §8) forbids arbitrary
  clamping as a stability mechanism and demands fail-loud behaviour
  (D-V2-004/D-V2-005 precedent).
- **Decision:** Implement the pinned *mathematics* exactly (22-term power
  series below `kappa = 12`, the pinned 3-term asymptotic expansion at and
  above, float64 internals, dtype-preserving output; the analytic Amos
  ratio backward) while replacing the dead clamps with (a) **exact
  `torch.where` branch gating** — the inactive branch receives gated safe
  inputs (zeros for the series, ones for the logarithm), so no infinity is
  ever materialised — and (b) fail-closed domain validation in forward
  (`kappa` must be finite, floating, `>= 0`; NaN/Inf/negative raise).
  Values on the valid domain are identical to the pinned snippet; invalid
  inputs are rejected instead of silently clamped.  Additionally the
  public surface ships `log_bessel_i0`, `amos_ratio`, and the circular
  objective (`von_mises_log_prob` / `von_mises_nll` / `von_mises_loss`)
  that #24 MODEL-04 consumes — the head *architecture* remains #24's scope.
- **Alternatives considered:** porting the snippet verbatim with its dead
  clamps (rejected: violates the #45 §8 no-clamp policy and hides invalid
  inputs); raising on negative `kappa` *and* healing masked targets in the
  objective layer (out of #8 scope); wrapping the branches in try/except
  (rejected: catch blocks are explicitly forbidden).
- **Evidence:** `tests/test_von_mises_torsion.py` — 44 tests: forward vs
  `scipy.special.i0e` measured max error **6.51e-6** over `[0, 100]`
  (acceptance: `< 1e-5`), in both log-space and the scaled-space form the
  criterion names literally; backward finite/non-negative/strictly `< 1`
  everywhere incl. `kappa = 0` (exactly 0) and `kappa ∈ {1e3, 1e5}`;
  Amos-vs-`i1e/i0e` deviation measured **3.30e-2** max at `kappa ≈ 2.5`
  (7.9e-4 / 2.0e-4 / 7.4e-5 at 30/60/100) and pinned `< 4e-2`; seam tests
  at `kappa ∈ {11.999, 12, 12.001}` prove the legacy NaN-leak failure mode
  is structurally impossible; von Mises log-density vs
  `scipy.stats.vonmises` `< 1e-4`, numeric normalisation ≈ 1; AST-level ban
  on `clamp`/`nan_to_num`/`clip` inside the module.  Full suite 449/449.
- **Affected:** #8 (closed), #24 (consumes the objective; its `#8`
  dependency is satisfied — #11 CHEM-01 is the sole remaining blocker),
  #32/#33 (suite aggregation).

## D-V2-014 — 2026-11-02 — The hardware gate enforces pure FP32 everywhere and reports out-of-scope devices; simulated `DeviceInfo` inputs never touch global state (#10)

- **Problem:** Issue #10 pins the literal gate "if `device.type == cuda`
  and `compute_capability < 8.0`: force `mixed_precision = fp32`, disable
  autocast, remove `GradScaler`", while the constitution (#45 §5/§29.2)
  additionally restricts the *supported* hardware to Colab/Kaggle Tesla T4
  and mandates pure FP32 for the whole production path — including what
  happens on non-T4 CUDA devices (do not silently train).  The clean-room
  contains no AMP machinery at all (D-V2-005), so "#10 removes GradScaler"
  is enforced structurally rather than by deletion; the acceptance tests
  must run on a CPU-only sandbox and must not mutate global `torch.backends`.
- **Decision:** `configure_runtime_environment` resolves a device via real
  probing (`probe_device`) or an explicit **simulated** `DeviceInfo` (the
  acceptance-test path — never mutates global state); every non-`fp32`
  request token (`fp16`/`bf16`/`amp`/`mixed`) is rejected with a logged
  `FAIL-CLOSED` warning (cc < 8 cites the T4 GradScaler-collapse history;
  cc >= 8 additionally records the out-of-scope-device warning; CPU cites
  D-V2-005) and the frozen `RuntimeConfig` always carries
  `precision="fp32"`, `autocast_enabled=False`, `grad_scaler=False`.
  TF32 stays off by default, requires `tf32_justified=True` *and* cc >= 8
  (structurally impossible on the supported T4), and global backends are
  only written for *real* probed CUDA devices.  The issue-pinned logit-mask
  constants (`LOGIT_MASK_FILL = -1e4`, `LOGIT_MASK_LEGAL_ABOVE = -1e3`)
  ship with fail-closed `apply_logit_mask`, the pinned scatter
  target-unmasking invariant as a pure function
  (`unmask_target_actions`), and strict `validate_target_actions` for
  callers that refuse to heal invalid supervised targets (#45 §9.6).  A
  repo-wide AST regression test bans `GradScaler`/`autocast`/`torch.amp`
  and reduced-precision dtypes from `src/syntree`.
- **Alternatives considered:** honouring fp16 on cc >= 8 (rejected: the
  constitution's pure-FP32 mandate is device-independent); raising instead
  of warning on fp16 requests (rejected: the issue pins "logs a fail-closed
  warning and executes in pure FP32"); mutating global backends for
  simulated devices (rejected: test pollution).
- **Evidence:** `tests/test_fp32_enforcement.py` — 35 tests: both pinned
  acceptance criteria (simulated Tesla T4 cc 7.5 forces `fp32` + no
  GradScaler; `mixed_precision="fp16"` on cc 7.5 logs the FAIL-CLOSED
  warning and runs pure FP32); the gate over V100/K80/A100/H100/CPU;
  TF32 policy incl. structural impossibility on T4; global-state
  non-mutation for simulated infos; the pinned scatter verified
  bit-identical against the literal in-place snippet; the AST bans.  Full
  suite 484/484.
- **Affected:** #10 (closed), #26 (consumes `RuntimeConfig`; its `#10`
  dependency is satisfied — #19/#20/#25 remain), #24 (consumes the
  logit-mask constants), #43 (reuses `DeviceInfo`/`probe_device` for
  measured profiling).

## D-V2-015 — 2026-11-02 — Soft-core role separation: #14's flat-bottom form owns the clash/torsion objective and the volume floors; the 12-6 surrogate keeps the docking role (#14)

- **Problem:** D-V2-009 listed the engine's "soft-core LJ / Coulomb
  surrogate" as an interim with injection points for #14, but the engine
  surrogate served *two* distinct roles: (a) the sub-bin torsion
  refinement objective of #42 (needs a bounded repulsive clash penalty —
  exactly the issue #14 pinned flat-bottom form, per #14's follow-up
  addendum) and (b) the reward's `S_dock`/`S_contact` energies (need an
  *attractive* branch + Coulomb to express binding affinity).  The pinned
  #14 form is repulsive-only with a flat bottom: using it as `ΔG_dock`
  would delete the affinity semantics and identically zero the designed
  `S_contact` component of the #30 reward formulation.
- **Decision:** (1) `syntree/chemistry/conformer.py` owns the production
  flat-bottom soft-core clash potential — the issue-pinned form with
  `delta = 0.5 Å^6`, Bondi radii + D-V2-009 depths, Lorentz-Berthelot
  mixing **capped by the 500 kcal/mol bounded-spike ceiling**
  (`eps_ij = min(sqrt(eps_i eps_j), 500·delta/(4 sigma_ij^6))`), which
  makes the #14 acceptance bound (`V(0.5) < 500`) hold for *every*
  supported element pair by construction (measured: `V(0.5) = 484.85`,
  `V(0) = 500.00` vs `1.58e9` for the rigid wall); analytic gradient + a
  torch autograd mirror (the "differentiable potential"); a deterministic
  grid cavity-volume estimator; the ratio floor (0.35) and the
  heavy-atom floor (12, weight 2.0) penalties.  (2) The terminal sub-bin
  torsion refinement objective in
  `environment.py::finalize_terminal_molecule` now evaluates the
  production clash potential (the addendum's demand).  (3)
  `soft_core_pocket_energy` (12-6 + softened Coulomb) remains exported in
  its *docking-surrogate* role for `S_dock`/`S_contact` — a different
  injection point owned by #30 FLOW-02 / the evaluation stack.  (4)
  `ThreeDReward.S_vol` now carries both #14 floors with the cavity volume
  derived **once at construction** from the fixed pocket (stationarity
  preserved — the compute signature still has no batch parameter);
  explicit `volume_floor_angstrom3` overrides the derived floor; no
  pocket + no floor keeps the graph-only reward.  The Brent-vs-dense-scan
  engine test's dense reference was updated 1:1 to the production
  potential (contract preserved, #45 §24.3).
- **Alternatives considered:** using the flat-bottom form as `ΔG_dock`
  (rejected: removes affinity semantics, zeros `S_contact`); removing the
  12-6 surrogate entirely (rejected: the docking surrogate is a documented
  separate injection point per D-V2-009); per-element uncapped epsilon
  (rejected: the pinned `< 500` acceptance would fail for heavy pairs —
  the cap is the documented design); deriving the cavity volume per
  `compute()` call (rejected: would add work per trajectory and risk
  stationarity questions — construction-time derivation is the fixed
  pocket context).
- **Evidence:** `tests/test_soft_core_chemistry.py` — 50 tests: the
  pinned form (flat bottom bitwise zero, in-branch values to 1e-12,
  monotone repulsion, documented bounded contact jump); the universal
  bounded-spike bound over all element pairs; gradient vs finite
  differences and vs torch autograd; atom-typing tables pinned; unknown
  element policies; pair-matrix engine (totals, determinism,
  fail-closed); cavity estimator (positive, deterministic, monotone,
  resolution documented); both floor penalties' shapes/bounds; reward
  integration (10-heavy-atom *assembled* molecule incurs the explicit
  penalty `-0.3333`; 12-heavy dodecane does not; stationarity across
  engine instances bit-equal; complete telemetry); refinement objective
  proven equal to the production potential at the reproduced pose and
  *different* from the 12-6 surrogate.  Full suite **534/534** including
  all 50 engine contracts (toy-DAG flow accounting, batch-composition
  invariance, bit-for-bit rollout determinism).
- **Affected:** #14 (closed), #42 (its `#14` dependency is satisfied; the
  refinement objective is now the production potential; #11/#24 remain),
  #30 (its `#14` dependency is satisfied; #29 remains; the docking
  surrogate's production closure stays #30's scope), #24 (no change),
  #16 (cavity estimator calibration on real pockets arrives with DATA-01).

## D-V2-016 — 2026-11-02 — The certified 32-reaction taxonomy supersedes the engine's generic junction registry; the family-map conflict resolves to cross_coupling_cc; stereo retention is verified, not declared (#11)

- **Problem:** Three internal conflicts had to be resolved without
  guessing (constitution §0).  (1) The issue's family listing omits
  ``rxn_14``/``rxn_15`` — the two remaining C-C couplings after
  ``cross_coupling_cc``'s five — while the same issue mandates "map the
  32 transformations into the 9 canonical reaction families".
  (2) The engine's pinned single-junction MDP can never close a ring, so
  the six "heterocycle cyclizations" could not be expressed as ring-forming
  junctions.  (3) The issue's "explicit stereochemical retention tokens in
  all coupling templates" would, if implemented as hard ``@``/``@@``
  requirements inside reactant templates, reject valid enantiomeric and
  achiral reagents (SMARTS ``[C@]`` matches only that winding), the
  opposite of the intended retention guarantee.
- **Decision:**
  1. ``rxn_14`` (alkenyl Suzuki) and ``rxn_15`` (alkyl Negishi) are placed
     in ``cross_coupling_cc`` — the only C-C family — so the 9-family map
     covers all 27 policy reactions exactly as the issue demands.
  2. The heterocycle installations ship the ring **pre-cyclized on the
     synthon side** (the only representation compatible with the pinned
     MDP contracts); every entry documents the one-step cyclization
     provenance of its synthon class (amidines + beta-dicarbonyls for
     pyrimidines, Hantzsch for thiazoles, etc.).
  3. Stereo retention is implemented by RDKit's atom-map chirality
     transfer and **verified**: 16 stereo certification pairs compared as
     canonical isomeric SMILES plus explicit CIP-code tests — templates
     never hard-require a winding.
  4. ``DEFAULT_REACTION_REGISTRY`` **is** the certified taxonomy (27
     policy reactions in ``JunctionReaction`` format); the generic
     interim (``generic_coupling``/``amide_formation``/
     ``ester_formation``) is superseded and removed — no side-by-side
     versions (constitution §24.3).  All 50 pinned MDP/TB contracts stay
     green bit-for-bit (verified: full suite 616/616 at the #11 commit).
  5. Every policy reaction carries a product-side ``junction_pattern``
     (map ``:1`` = molecule-side attachment, ``:2`` = block-side) with a
     ``junction_priority``; bond-claim ambiguity resolves by
     ``(priority DESC, reaction_id ASC)`` — the metadata the #40
     chemoselectivity rules and #41 retrieval index require (the #11
     addendum) without weakening exact forward-replay validation.
- **Alternatives considered:** a 10th family for rxn_14/15 (rejected: the
  issue pins exactly 9); expressing cyclizations as engine ring-closure
  (rejected: would change the pinned ``JunctionRecord``/dihedral
  contracts); hard ``@`` tokens in reactant templates (rejected: rejects
  valid reagents); keeping the interim registry beside the certified one
  (rejected: §24.3 forbids old and new active side by side).
- **Evidence:** ``tests/test_32_reactions.py`` — 82 tests: taxonomy shape
  (32 = 8 C-N + 7 C-C + 6 C-O/C-S + 6 heterocycle + 5 deprotections;
  exact family membership incl. rxn_14/15); 100% round-trip forward-replay
  certification on 48 standardized reagent pairs (16 stereo, CIP
  retained); deprotections deterministic with no-op absence; engine
  end-to-end products through the real ``AssemblyEnvironment`` for 10
  reactions (certified trans locks snap to pi exactly, rotatable junctions
  honour the torsion bin, TB engine consumes certified-registry
  trajectories); fail-closed behaviour.  Full suite 616/616 → 655/655.
- **Affected:** #11 (closed), #12/#40/#41 (consume the handle metadata and
  junction patterns), #15 (consumes the patterns as its cut-eligibility
  source), #24 (dependency satisfied; consumes the registry), #42/#30
  (unlocked).

## D-V2-017 — 2026-11-02 — #13's geometry stages integrate behind opt-in flags whose defaults preserve the pinned legacy behaviour; MMFF94's weak amide torsions are restored exactly post-minimization (#13)

- **Problem:** (1) #13's body (written before the owner reconciliation)
  prescribes the cis/trans planar choice "via minimum Lennard-Jones pocket
  clash" and phi routing to the adjacent synthon-side bond — but the
  reconciliation tests pin the terminal oracle's legacy behaviour
  (locked junctions skipped, ``diagnostics["junctions"] == []``,
  ``subbin_refinements == 0``, drift measured from pi).  (2) RDKit's MMFF
  cannot type bare pocket atoms, so "minimize freely in MMFF94 within the
  static vdW field of the pocket" had no direct implementation.  (3) A
  measured finding: MMFF94's amide C-N torsion terms are weak — a bare
  amide sits ~3.7 deg off planar at the MMFF minimum, ~11.6 deg for the
  in-pipeline pose — so MMFF alone cannot deliver #13's pinned
  "planar within 1.0 degree" acceptance.
- **Decision:**
  1. ``apply_junction_torsion`` and ``relax_junction_constrained`` live in
     ``syntree/chemistry/conformer.py`` as terminal-oracle stages;
     ``finalize_terminal_molecule`` exposes them behind **opt-in flags**
     (``junction_torsion_stage`` / ``pocket_vdw_relaxation``, both default
     ``False``) so the 50 pinned reconciliation contracts stay green
     bit-for-bit — the same injection-point pattern as D-V2-009/D-V2-015.
     The #42 addendum ordering (sub-bin refinement before the #13 stages)
     is implemented and tested.
  2. The static pocket vdW field is realized by embedding the pocket heavy
     atoms into the MMFF system as **typed mini-fragments**
     (CH4/NH3/OH2/SH2 — hydrogen-completed, so MMFF types them),
     position-constrained (flat-bottom 0.05 A, k=5000) with their
     hydrogens: genuine ligand-pocket vdW coupling, pocket static within
     its tolerance.  Pocket metals receive RDKit MMFF's generic metal type
     (95) — a documented placeholder vdW shell, not an error (measured).
  3. The #13 planarity is **restored exactly after the free minimization**
     by the same LJ-chosen rigid snap (``route_phi=False`` mode): a pure
     rotation of the distal subtree, preserving the MMFF-relaxed internal
     geometry undistorted.  The "junction bond within 0.02 A of
     equilibrium" acceptance is implemented as convergence to the MMFF
     equilibrium (second-pass relaxation moves it < 0.02 A) — the engine's
     bin-centre placement deliberately uses covalent-radii sums, which are
     not MMFF equilibria (measured: N-C 1.47 A placed vs ~1.34 A MMFF
     amide).
- **Alternatives considered:** flipping the default behaviour on
  (rejected: breaks the pinned contracts silently); re-snapping only
  before MMFF (rejected: the measured 3.7-11.6 deg tilt shows the
  constraint does not survive the free minimization); post-hoc planarity
  by coordinate projection (rejected: would distort covalent geometry;
  the rigid rotation preserves it exactly).
- **Evidence:** ``tests/test_conformer_relaxation.py`` — 18 tests: LJ
  argmin self-consistency across two pockets (one crowding the trans
  region); no-pocket ties resolve to trans; phi routing verified on the
  exact deterministic reference convention; covalent geometry preserved
  under every rotation; the pinned acceptance criteria (planarity
  <= 1.0 deg, seed RMSD < 0.15 A, junction bond < 0.02 A of equilibrium,
  pocket static <= 0.05 A); combined pipeline ordering; legacy-default
  boundary.  Full suite 655/655.
- **Affected:** #13 (closed), #42 (its #13 dependency is satisfied; #24
  remains), #29 (its #13 dependency is satisfied; #25/#26/#39 remain),
  #30 (production physics closure unchanged in scope).

## D-V2-018 — 2026-11-02 — #24's hierarchical heads: reaction conditioning by support restriction; torsion conditioned on the tied synthon embedding; kappa by softplus with the MLE anchor; synthon masks as class prefilters (#24)

- **Problem:** (1) How should the 27-way reaction law be conditioned on
  the 9-way family selection?  (2) Which inputs condition the parity and
  torsion heads?  (3) How can the von Mises concentration be
  non-negative without clamping, and does its NLE growth risk the
  pre-V2 kappa-to-infinity failure?  (4) What does the synthon mask
  guarantee?
- **Decision:**
  1. **Support restriction realizes the hierarchy**: the reaction head
     produces 27 logits from the decoder context and its law is
     normalized over exactly the legal reactions *of the selected
     family* (`reaction_support_mask`).  This makes
     ``P(f, r | s) = P(f | s) · P(r | f, s)`` an exact factorization
     (pinned by test: summing the joint over all legal pairs recovers
     1) without a second representation path or a family-embedding
     plumbing that would duplicate the conditioning the context already
     carries.
  2. **Parity and torsion are conditioned on (context, selected-synthon
     embedding)** — the issue pins the torsion conditioning on ``B_t``
     explicitly; the parity head follows the same junction-level
     conditioning for interface symmetry.  Both consume the tied catalog
     embedding of the selection (`SynthonOutput.retrieved_embedding`),
     so gradients from both heads flow into the single catalog table.
  3. **kappa = softplus(raw)** — a smooth parameterization of the
     non-negative domain (mirroring the issue-pinned Amos form), not a
     clamp.  The pre-V2 "incentivizing kappa to infinity" failure was a
     *Trajectory-Balance measure-theory* violation (a rad^-1 density
     summed with dimensionless probabilities); in V2 the categorical is
     the only TB measure (pinned: step log-prob invariant under
     kappa/mu perturbation) and the supervised NLL anchors kappa at
     ``A^{-1}(cos Delta)`` for residual ``Delta`` — finite unless mu
     hits the target exactly, with #20's coordinate jitter keeping
     supervised angles non-degenerate.  ``mu`` is the atan2 of a 2D
     vector routed through the certified ``safe_unit_vector``
     primitive (#7) — bounded gradients at the zero vector.
  4. **The synthon legality mask is a class-level prefilter**
     (`synthon_legality_mask`: catalog handle-class vs the selected
     reaction's block class); the environment enforces the exact
     one-handle chemistry at execution and #40's curation guarantees
     curated blocks expose exactly one coupling handle.  Reaction-level
     availability must be consistent with the catalog (fail-closed
     otherwise).
- **Evidence:** `tests/test_hierarchical_heads.py` 100/100 — the exact
  factorization identity, STOP-isolation by forward hooks, TB
  composition through the real engine, kappa/mu invariance of the step
  log-prob, registry-mask cardinalities against real environment menus,
  the bin-centre identity with the engine's pinned geometry, and the
  end-to-end SE(3) invariance through the real stack.
- **Affected modules/issues:** `syntree/models/policy.py`,
  `syntree/models/torsion_head.py` (#24 closed at `74133c8`; #25/#26/#29
  consume the contracts).

## D-V2-019 — 2026-11-02 — The assembly's 10.88M frozen profile supersedes #25's ~4.8M estimate; the binding criterion is the < 50 MB memory bound (#25)

- **Problem:** issue #25 pins "total parameter count must settle at
  ~4.8M (~19.2 MB FP32)" but also "< 50 MB" — while the closed, frozen,
  and verified per-issue profiles already sum far above 4.8M
  (#21 backbone 5,999,104 + #22 handle head 224,896 + #23 decoder
  1,219,328 + #24 heads 3,272,975 incl. the 3.2M tied catalog table +
  #25's own seed/anchor head 164,416 = 10,880,719).
- **Decision:** the frozen per-issue profiles are authoritative
  (constitution #45 §5.2: architecture dimensions are frozen in
  version-controlled configuration, never re-derived to hit an
  estimate; re-shrinking the backbone/decoder to chase 4.8M would
  violate four closed issues' pinned contracts).  The **binding**
  acceptance criterion is the memory bound: 10,880,719 parameters =
  43,522,876 bytes FP32 = **43.5 MB < 50 MB** (87% of budget), pinned
  by test against the version-controlled
  ``SYNTREE_POLICY_PARAMETER_PROFILE`` and reported in the assembly
  audit.  The ~4.8M figure is recorded as a pre-implementation estimate
  superseded by the frozen dimensions.
- **Evidence:** `tests/test_policy_assembly.py` — profile equality,
  component-sum equality, tied-table counted once (identity), the
  memory arithmetic, and the audit fields.
- **Affected modules/issues:** `syntree/models/policy.py` (#25, commit
  `84fd3e7`); #26's trainer consumes the audit's memory numbers.

## D-V2-020 — 2026-11-02 — #40's exclusion matrix is per handle class (derived from the registry's consumers), orthogonal protection passes by SMARTS construction, and the 2,4-dihaloazine exemption is asymmetric (#40)

- **Problem:** (1) At what granularity are the chemoselectivity rules
  defined — per family, per reaction, or per handle class?  (2) How are
  protected multifunctional substrates distinguished from
  self-reactive ones without a protected-group whitelist bureaucracy?
  (3) When is a dihalo heteroarine a valid mono-coupling block rather
  than a regioisomer mixture?
- **Decision:**
  1. **Per block-side handle class** (the 19 registry classes), with the
     exclusion set derived from the union of the class's consuming
     reactions and their reagent natures (amine/alcohol/thiol
     nucleophiles, organozinc/acetylide bases, Pd/Sn couplings).  This
     is finer than the issue's "per family" phrasing and consistent
     with it (a family's rules are exactly its block classes' rules);
     the class is the curation unit and the environment's coupling unit.
  2. **Orthogonal protection is honoured by construction**: the
     free-group SMARTS match only free forms (acyl/carbamate/sulfonamide
     N, ester/silyl/benzyl O, thioether S, aromatic [nH] never match),
     so a Boc-protected diamine or silyl-protected bromoalcohol passes
     without any whitelist — pinned by the protected-substrate battery.
     Suzuki-tolerant functionality (free amines/alcohols/COOH on
     boronic acids — common real substrates) passes; only thiols are
     excluded there (catalyst poison).
  3. **The 2,4-dihalo-1,3-azine exemption is asymmetric**: a spectator
     ring halide is exempt only when it is separated from the handle by
     exactly one ring nitrogen AND flanked by *more* adjacent ring
     nitrogens than the handle (the less amine-labile position — the 2-
     vs 4-position of pyrimidines).  2,4-dichloropyrimidine passes with
     the handle declared at the amine-labile C4 and is excluded if
     declared at C2 (the amine reagent would attack C4 first); the
     equivalent 4,6-dichloro pattern never passes.  Undeclared
     multi-instance candidates resolve to the most amine-labile
     instance (fewest adjacent ring N) with ties excluded as ambiguous.
  4. **Competing electrophiles are uncatalyzed ones only**: aryl halides
     never compete under rxn_01-type amine conditions (no Pd present) —
     a bromo-aryl acyl chloride passes R5 (pinned by test).
- **Evidence:** `tests/test_chemoselectivity.py` 111/111 — vocabulary
  battery with negative controls, all 19 registry block classes with
  representative molecules, every rule positive/negative, the exemption
  in both declared directions, manifest impact counts/percentages,
  reason-code traceability.
- **Affected modules/issues:** `syntree/chemistry/chemoselectivity.py`
  (#40 closed at `044adca`; #12 consumes the screen + manifest inside
  catalog curation; #41 may reuse the handle detectors).

## D-V2-021/022/023 — 2026-11-02 — #12 catalog build decisions (artifact-pinned record; D-V2-021/022 rationale text lost to the S-V2-10 sandbox recycle)

> **Provenance warning (recorded per #45 §19.1):** the S-V2-10 session
> drafted these decisions but its working tree was recycled before the
> docs commit. `catalog_manifest.json::decision_log_refs` pins the IDs
> `["D-V2-021", "D-V2-022", "D-V2-023"]`; only D-V2-023's operative
> text survives verbatim (inside the manifest). This entry pins every
> binding fact that survives in the artifacts. When the build code is
> re-landed (#12's remaining scope), the re-implementation MUST satisfy
> these constraints and the restored rationale texts must be folded
> back into this entry.

- **D-V2-023 (survives verbatim in the manifest):** *"issue target 70%
  saturated / 30% aromatic; the certified saturated supply is the
  measured binding constraint: every certified saturated block is
  admitted and the remainder refilled from the aromatic stratum in
  canonical-SMILES order."* Selection arithmetic: 50,000 target =
  35,000-saturated stratum + aromatic refill; 5,000-row reserve buffer
  absorbed 16 fail-closed encode failures (`reserve_used: true`); final
  34,984 saturated (69.968%) / 15,016 aromatic (30.032%);
  `strata_fallback_used: false`.
- **Artifact-pinned binding facts** (attribution to D-V2-021 vs
  D-V2-022 to be restored with the code):
  1. **d_raw = 2,137, not the issue body's "2138"**: 2048 (Morgan,
     radius 2, normalized) + 47 (physicochemical scalars) + 6 (3D shape)
     + 5 (electrostatics) + 30 (handle direction vectors) + 1 (norm)
     = 2137 — the stated 2138 does not add up. Pinned by
     `normalization/mean` shape 2137 and `projection/Q` shape
     `[2137, 256]` in the HDF5.
  2. **Projection protocol:** per-feature z-normalization (statistics
     computed over the selected 50,000 rows and STORED in the h5
     `normalization/` group) → orthonormalized QR Johnson–Lindenstrauss
     matrix, **seed 2026** (`projection/seed`) → L2-normalization of the
     projected row (the issue's e_B formula). The seed is part of the
     frozen artifact identity; a rebuild must reproduce it bit-exactly.
  3. **Conformer/encoding policy:** ETKDGv3 embedding with full MMFF94
     relaxation where MMFF types the molecule; per-row provenance in
     `catalog/conformer_methods` (`etkdg3` 17,493 / `etkdg3+mmff94`
     32,507 — rows MMFF cannot type keep the ETKDG geometry, recorded
     honestly). Encoding is fail-closed with named reasons; the 16
     failures (12 ETKDG-no-conformer; 4 MMFF-untypeable `['B','K']`
     outside the documented B/Zn/Sn coupling-handle fallback) were
     replaced from the reserve, never silently dropped.
  4. **Screen-before-admission is mandatory** (audit addendum): the #40
     chemoselectivity screen (`screen_block`/`screen_catalog`) ran
     inside the curation loop — 39,150 exclusions with per-rule counts
     embedded in the manifest; basic filters alone never admit a
     synthon-role record.
  5. **The reaction-registry digest is part of the manifest**
     (`32ea6db9ce9d56747c6a3cc957e3dfa0c2413fa02a6a583cd07e23ff6f4c3f3d`)
     — the catalog is only valid against that certified taxonomy; any
     registry change invalidates the certification and requires a
     re-screen.
  6. **The HDF5 artifact is immutable** (`artifact.immutable: true`):
     any rebuild produces a NEW artifact version with a new checksum;
     this one is never mutated in place.
- **Evidence:** `docs/data/catalog/catalog_manifest.json` + the HDF5
  checksum sidecar (this commit); independent re-verification of all
  four acceptance criteria from the HDF5 (S-V2-10c, recorded in
  `WORK_LOG.md` §2.2); HF dataset `JJKK1212/3d-syntree-v2-catalog`.
  Unit tests to be re-landed with the build code (#12 open scope).
- **Affected modules/issues:** #12 (remaining scope: re-land
  `syntree/chemistry/synthon_encoder.py` +
  `scripts/build_synthon_catalog.py` + tests against these pins);
  consumers #18/#24/#39/#41/#44 (read the pinned schema; do not
  re-derive the descriptor or projection); #37 (the HF dataset repo is
  the catalog artifact channel).


## D-V2-024 — Continuous commit-and-push persistence

**Status:** accepted by explicit owner direction (2026-11-02).

**Decision:** Commit each bounded, coherent unit immediately after its applicable checks pass, then push it to `origin/3d-synV2` without waiting for session end. Push recovery checkpoints before risky or long-running work, before delegation, and after each independently recoverable batch. Verify every push by inspecting the remote ref and record the exact SHA and evidence in the existing work log or handoff. Preserve concurrent remote history; never force-push. Keep implementation changes on `3d-synV2`. Credentials must remain outside prompts and tracked files and must use secure credential storage.

**Rationale:** The S-V2-10 workspace recycle demonstrated that a session-end push can lose completed implementation work. Immediate pushes make verified progress visible and recoverable. A local commit without verified remote presence is not durable progress.

**Limits:** This does not permit knowingly broken commits, weakened tests, or recursive verification-only commits.

---

## D-V2-025 — 2026-11-03 — #12 re-landing: the artifact-recovered curation semantics (S-V2-11)

**Status:** accepted (artifact-pinned reconstruction; every pin verified at full scale).

**Decision:** The re-landed catalog build implements the curation
semantics exactly as recovered from the surviving S-V2-10 artifacts:

1. **Seed-role chemoselectivity is block-level.** A molecule carrying ANY
   free-group/electrophile self-reactive pair (the #40
   `SELF_REACTIVE_PAIRS` matrix evaluated between the molecule's *free*
   groups — the #40 free-form vocabulary — and the six uncatalyzed
   electrophiles: acid/sulfonyl/carbamoyl chlorides, chloroformate,
   alkyl bromide, aldehyde) loses the **entire** seed role, not just the
   firing class. Reproduces the pinned 757 fired blocks (760
   pair-firings) and the 118,683/39,150 seed-only/excluded counts
   exactly. Consequence: an amidine "amine" seed beside a certified
   alkyl bromide is safe (its N is not the free form), while a free
   primary amine beside any alkyl bromide is not.
2. **Synthon certification declares the first instance.** Each detected
   registry handle class is screened with
   `screen_block(mol, cls, handle_atom_idx=<lowest atom index>)`; the
   two programmatic boronic classes are screened without a declared atom
   (their cross-field validation is unsupported by `handle_class_of_atom`).
3. **Basic-filter stage order: MW → rotatable → heavy** (pinned by the
   76,970/21,691/56 count split).
4. **MMFF typing decision (pre-embedding, hydrogen-added molecule)**:
   relax iff `MMFFHasAllMoleculeParams` passes; otherwise keep the first
   ETKDG geometry iff the atoms outside the organic coverage fall within
   {B, Zn, Sn} (coupling handles) ∪ {F, Cl, Br, I} (halide counterions
   of the catalog's salt forms); anything else (e.g. potassium
   trifluoroborates, `['B','K']`) fails closed. Reproduces the stored
   17,493/32,507 etkdg3/mmff94 split with 0/50,000 row mismatches.
5. **Descriptor reconstruction residuals** (quantified, bounded): 3 of
   the 47 physchem functional-group slots (tertiary-amine, post-sulfone
   rare slot, one provably all-zero slot), the 30-D handle-vector class
   list (identity normalization in the artifact — no per-feature
   statistics exist to recover it), and the original run's ETKDG seed
   (stored energies underdetermine the protocol). The re-landed encoder
   is deterministic (seed 2026, 10 conformers, prune 0.5) and the
   immutable HDF5 remains the authoritative catalog.

**Rationale:** the surviving artifacts (screen funnel counts, catalog
labels, `buffered_rows` indices, stored Q/normalization, conformer
methods) form a complete oracle; the reconstruction was iterated at
full scale until every recoverable quantity matched exactly, and the
unrecoverable ones are explicitly bounded rather than guessed silently.

**Evidence:** 24/24 funnel counts exact over 290,648 records; 0/50,000
certification-label mismatches; `buffered_rows` reproduced exactly;
`projection/Q` bit-exact; conformer-method split exact (0/50,000);
rebuilt catalog 49,998/50,000 block-id overlap; `docs/data/catalog/`
README verification table; commits `b76422a` + `b8439bd`.

**Affected modules/issues:** #12 (closed), consumers #18/#24/#39/#41/#44
(consume the pinned schema + semantics; do not re-derive).

## D-V2-026 — 2026-11-03 — Registry-digest residual reconciled: the index binds the live digest + catalog sha256 (S-V2-12)

**Problem:** the re-landed `registry_digest()` (`08c9fb08…`) cannot
reproduce the immutable catalog's pinned `registry_digest` attribute
(`32ea6db9…`); a strict index↔catalog digest check would fail forever.

**Investigation:** forensics over 40+ plausible payload formats (field
subsets, orderings, whole-dataclass dumps) found no reproduction — the
original digest payload format was lost with the S-V2-10 build code
(the same residual class as the ETKDG seed, D-V2-025). The registry
**source** is provably unchanged since `0e296b0` (its only commit,
`git log --follow` verified), so certification validity is not in
question — only the historical hash format differs.

**Decision:** the #41 index binds (a) the catalog file **sha256**
(exact, unambiguous), (b) the **live** `registry_digest()` at build
time (any future registry change invalidates the index), and (c)
build-time justification of every certified class against the live
registry's block-side classes. The catalog's historical `32ea6db9`
attribute is preserved in the index metadata as provenance, not as an
active check.

**Affected:** #41 (closed), #12 (historical note), #44 (the stock-export
track inherits the same binding policy).

## D-V2-027 — 2026-11-03 — rxn_14/rxn_15 junction-pattern orientation defect corrected; attachment atoms are located by map number (S-V2-12)

**Problem:** #18's reactant reconstruction exposed that rxn_14/rxn_15's
junction patterns anchored `:1` on the **block** side (the aryl),
violating the registry's documented `:1 = molecule-side` convention
(the reaction SMARTS themselves map `:1` to the molecule-side atom). The
silent accomplice: RDKit `GetSubstructMatches` returns matches in
*template position* order, and all other patterns are written `[:1]`
first, so the positional assumption `match[0] == :1` cooperated with the
defect. rxn_23's symmetric `[c:1][c:2]` (orientation-agnostic, like
rxn_09) is not a defect.

**Decision:** corrected the two patterns to `:1`-first branch form
(`[CH1:1]([c:2])=[CH2]`, `[CX4;H1,H2:1]([c:2])~[#6]`) and exported
`junction_pattern_attachment_atoms()` as the one authoritative
map-number locator; the fragmenter and the trajectory extractor now
locate `:1`/`:2` by map number (behavior unchanged for the other 25
reactions). The registry **digest is unchanged** (the digest payload
covers smarts/mol_class/block_class, not junction patterns), so the #12
catalog certification and the #41 index bindings are unaffected. The
graft-spec derivation (`derive_junction_specs`) is itself a fail-closed
orientation check: a mis-oriented pattern can no longer derive specs.

**Evidence:** `tests/test_trajectories.py` (orientation regression
test); 27/27 specs derive; full suite 1384/16.

**Affected:** #11 (closed — defect correction, no digest change), #15
consumer path, #18.

## D-V2-028 — 2026-11-03 — The seed pose is a deterministic covariant environment-side function of (B_0, h_0) (#39)

**Problem:** #39 requires explicit pose/orientation variables "with a
probability measure consistent with the rest of the policy"; the policy
is strictly categorical (constitution §12.3 forbids continuous
densities in the TB sum).

**Decision:** the placed pose carries **no probability measure**: like
the #42 sub-bin torsion refinement, it is a deterministic
environment-side geometry operation of the categorical
`(B_0, h_0)` action. Rule: seed centroid at the anchor position; the
template's intrinsic frame aligned to an anchor frame built only from
covariantly-selected pocket vectors (farthest-assigned-atom rules with
charge/atomic-number/index tie-breaks — invariant under rigid motions
of a fixed atom list), giving exact equivariance up to the documented
collinear fallback (`_deterministic_perpendicular`). Pocket-free
fixtures keep the block's template pose bit-exactly (no conditioning
context exists).

**Evidence:** equivariance, determinism, centroid-at-anchor, and
extendability tests in `tests/test_seed_anchor_selection.py`.

**Affected:** #39 (closed), #29 (the TB accounting consumes the seed
step; no pose density ever enters).

## D-V2-029 — 2026-11-03 — Observation (not yet acted on): rxn_15's junction pattern is looser than its reactant template

The 11 `replay_mismatch` cases of the bounded run are all rxn_15
junctions on **secondary** alkyl carbons: the junction pattern
`[CX4;H1,H2:1](…)` recognises H1 centres, but the certified reactant
template is primary-only (`[CX4;H2,H3:1][Zn:4][Cl:5]`, documented "no
beta-hydride stereochemistry"). The reverse reconstruction builds the
secondary organozinc, the forward replay correctly rejects it, and the
complex fails closed with an explicit code — the system behaves as
designed. A future registry alignment refinement could tighten the
pattern to `H2` (matching the template exactly) so such bonds are never
claimed; **not changed this session** (scope discipline — it only
changes which failure code a complex receives). Recorded for the owner
and for #11's audit trail.

## D-V2-030 — 2026-11-03 — Supervised states are crystal intermediates in the ligand-centroid frame; pockets are capped by centroid-distance pruning (#26)

Stage-1 behavioural cloning trains on the **crystallographic
intermediates** — the induced subgraphs of the crystal ligand over the
atoms placed at each assembly step (the #20 motivation names the
pristine-crystal training distribution explicitly), re-derived from the
#15 canonical decomposition and cross-checked against the #18 record at
every step (seed graph, step count, each step's `product_smiles` — a
disagreeing record/artifact pair fails closed as corrupted).  The
pocket tensors come from the processed pocket artifact, centred on the
crystal ligand's centroid (the #45 §11 extraction frame) and capped at
the profile limit (Profile A: 700 atoms) by keeping the
centroid-nearest atoms (documented centroid-distance pruning, #45 §5.2).
The supervised anchor label is computed **on the trimmed pocket** so it
always refers to the anchor decomposition the state actually carries.
Pocket charges are deterministic per-element Gasteiger base values
(D-V2-035: the documented deterministic interim for the schema's pH-7.4
formal charges — raw PDB pocket lines carry no bond topology to compute
true partial charges on).

**Evidence:** `tests/test_supervised_dataset.py` (context
cross-checks, cap trimming, states pass the frozen tensor contract).

**Affected:** #26 (closed once T4-verified), #20 (the augmentation
operates on these states).

## D-V2-031 — 2026-11-03 — The intermediate's growth-handle pool = molecule-side reaction-template matches at non-junction atoms (#26)

The #22 handle head consumes the *active handle pool* of the
intermediate.  On crystal intermediates (no `BuildingBlock` templates
exist there) the pool is defined as: a growth handle exists at atom
`a` for reaction `r` iff `r`'s **molecule-side reactant template**
matches at `a` — the attachment atom located **by map number** (the
D-V2-027 rule; verified to resolve uniquely for all 27 policy
reactions) — **and** `a` is not already a junction atom of a formed
assembly bond (the environment's capacity bookkeeping mirrored on the
crystal side; the #40 curation guarantees one coupling per handle).
Pool rows are `(atom, molecule_handle_class)` pairs, deduplicated, in
deterministic reaction-id/atom order.  The block-side class vocabulary
(`ALL_HANDLE_CLASSES`) is *not* used for intermediates — it is the
catalog side; the dual molecule-side detection goes through the
certified reactant templates (exact reagent-level constraints, not
functional-group nicknames).

**Engineering hazard pinned:** RDKit **reaction-template** mols
(`rxn.GetReactantTemplate(0)`) segfault `GetSubstructMatches` in this
build (all 27 reactions, rc=-11); the safe equivalent is parsing the
reactant SMARTS *string* directly (`MolFromSmarts`) — pinned by test.

**Evidence:** `tests/test_supervised_dataset.py::TestHandlePool`.

**Affected:** #26; any future consumer of intermediate-side handle
semantics (#29, #34).

## D-V2-032 — 2026-11-03 — The state's canonical `handle_features` row is the *primary handle* (#26)

The frozen state contract carries one `handle_features [B, 64]` row
per graph.  For growth states it is the 64-D descriptor of the
**primary handle** — the next assembly step's junction atom under the
#15 canonical order (deterministic; the same junction the #18 record
consumed) — with `handle_pos` its position.  Terminal states with no
next step carry zeros (documented degenerate: the policy's STOP path
does not consume the primary-handle channel).  The full pool for the
#22 head travels beside the state (not inside it) as flat
`[H, 64]/[H, 3]/[H]/[H]` tensors with batched-ligand atom indexing.

**Affected:** #26.

## D-V2-033 — 2026-11-03 — Empty-pool terminal states: STOP is structurally forced and carries no stop loss (#26)

A completed ligand whose handle pool is empty admits only STOP; the
#22 head refuses empty pools by design ("only STOP is legal on such
states").  The dataset marks such terminal samples `forced=True` and
the trainer attaches **no stop-head loss** to them (the masked law is
deterministic — P(STOP)=1 by mask, no learned mass, no gradient);
non-forced terminal states with non-empty pools carry the normal STOP
cross-entropy (that is where the STOP decision is a real choice).  A
*step* row with an empty pool is a data inconsistency and fails
closed.

**Affected:** #26 (the STOP supervision semantics).

## D-V2-034 — 2026-11-03 — Parity labels are derived from the crystal CIP at the junction atom (#26)

The #18 record carries no parity label.  The supervised parity label
is the junction atom's CIP code under `AssignStereochemistryFrom3D` on
the crystal intermediate `M_t`, with `required=True` iff the junction
atom **becomes** a stereocentre in `M_t` while it was not one in
`M_{t-1}` (a junction that merely retains an existing centre carries
no parity action).

**Affected:** #26.

## D-V2-035 — 2026-11-03 — Naive σ=0.15 Å jitter inverts stereocentres; the augmenter is stereo-preserving by rejection resampling (#20)

**Measured finding (S-V2-14):** the chirality volume of a typical sp3
centre (~0.39 Å³ at ~1.5 Å bonds) is only ~2.5× the noise-induced
volume σ at the issue-pinned jitter level, and plain iid Gaussian
jitter **flipped the 3D CIP signature on 2 of 50 seeds** on the
fixture `C/C=C/C[C@H](N)C(=O)O` (RDKit `AssignStereochemistryFrom3D`).
Issue #20's acceptance ("does not invert chiral stereocenters") is
therefore **violated by naive jitter**.  Resolution: the composed
augmenter draws jitter until the full 3D stereo signature (per-atom
CIP labels + the isomeric SMILES from 3D, covering E/Z) matches the
pristine crystal signature — bounded redraws (~1.05 expected at the
measured ~4% flip rate), fail-closed if no preserving draw is found.
The rigid dihedral stage needs no such guard (provably
chirality-preserving: every stereocentre sits wholly on one side of
the rotation cut, or is centred on the axis atom whose fixed bond
vector lies along the rotation axis).

**Evidence:** `tests/test_augmentation.py` (the flip finding is
pinned as a regression test; the composed path is verified on 200
seeds).

**Affected:** #20 (closed), #26 (the loader consumes the composed
augmenter).

## D-V2-036 — 2026-11-03 — The trainer's legality masks come from the immutable catalog artifact; seed-only rows are never legal retrieval targets (#26)

The synthon-legality mask and the seed-eligibility mask are built from
the immutable #12 catalog HDF5 at CLI load time: per-row block handle
class (the first certified synthon class; seed-only rows carry the
out-of-vocab sentinel `len(BLOCK_HANDLE_CLASS_VOCAB)` so they can
never match a reaction's block class), the seed-eligibility row mask
(inverted into the True=illegal mask convention — the polarity bug
was caught by the CLI smoke: a masked selected seed produced the
9999 mask-fill loss), and the distinct available block classes.  The
policy's tied catalog table is sized to the artifact row count
exactly.

**Affected:** #26.

## D-V2-037 — 2026-11-04 — The #27 difficulty axes, their combination, and the per-item ingestion point

The five issue-named physical/chemical difficulty axes are computed
**per supervised item** (the state the policy actually sees, not per
complex): `pocket_atoms` (the capped pocket count), `rotatable_bonds`
(RDKit strict count of the crystal *intermediate*), `rings` (SSSR of
the intermediate; both via the sanitized `induced_submol` views),
`pocket_charge_magnitude` (`Σ|q|` over the pocket), and `handle_rarity`
(`1 − count(class)/max_count` of the item's primary handle class over
the train split's items; the pre-seed and terminal STOP items are
handle-free and sit at the common end `0.0`).  The axes are z-scored
over the training split — a constant column contributes nothing (an
axis on which the split is uniform carries no difficulty information)
— and combined with **equal weights** (the plain mean): the issue names
the axes but not weights, and only the induced *ranking* matters for
quantile binning.  Levels are a *train-split* device:
`difficulty_features` fails closed on any non-train split, mirroring
the #20 augmentation split guard.

**Alternatives considered:** per-complex features (rejected — coarser
than the state actually consumed; the per-item view naturally places
seed/early states at the easy end, matching #27's "bias early training
toward rigid, low-clash fragments"); learned difficulty (rejected —
not in the issue scope and unfalsifiable at Stage-1 scale).

**Evidence:** `tests/test_curriculum.py::TestDifficultyIngestion`,
`TestScoresAndBins` (balanced ±1 quantile counts, monotone levels,
degenerate-axis behaviour, tie determinism).

**Affected:** #27 (closed), #28 (consumes the sampler), #43 (the
checkpoint payload carries the sampler state).

## D-V2-038 — 2026-11-04 — The floor-exact level distribution and the milestone semantics (#27)

The stage distribution is `p(ℓ) = (1 − Lε)·kernel(ℓ; μ(s), σ=2) + ε`
with `L = 9`, `ε = 0.03`, and `μ(s) = 1 + (s−1)·8/5` — a **mixture
with the uniform distribution**: it sums to 1 *by construction* and
keeps every level at *exactly* `ε` or above at every stage, including
the extreme centres μ=1 and μ=9.  The naive alternative
(`max(kernel, ε)` then renormalize) can silently dilute the floor
below the promised bound (up to ε/1.27 when many levels sit at the
floor) and is rejected.  Milestone semantics: the running *minimum*
validation loss is the reference (it only tightens); improvement
`≥ 0.005` advances the stage and resets patience; two consecutive
non-improving epochs advance by the documented fallback; the stage
never regresses and is capped at 6.  Epoch draws are **stateless
deterministic** per `(seed, stage, epoch)` (NumPy `default_rng`
derivation mirroring the dataset's per-item streams), so an epoch
re-iterates bit-identically — the trainer's exact resume needs no
mid-epoch generator state, and the checkpointable state is exactly
`(stage, epoch, best, patience, transitions, histograms, config)`.

**Evidence:** `TestLevelDistribution` (floor exactness at every stage
incl. extremes, mass shift monotonicity), `TestMilestoneMachine`
(including the 40-epoch smooth-progression acceptance),
`TestEpochDraws` (exact N, all-9-levels per epoch, empirical
frequencies), `TestCurriculumState` (round-trip + fail-closed).

**Affected:** #27 (closed), #26 (the trainer's `fit` consumes the
milestone machine and persists the state), #43.

## D-V2-039 — 2026-11-04 — `--curriculum` is an additive CLI flag; the pinned #26 acceptance launch is unchanged (#27)

The canonical Stage-1 CLI gains `--curriculum` (builds the sampler
from the train split, wires `DataLoader(sampler=curriculum)`, passes
it to the trainer) as **the recommended production form** — but the
default remains off and the #36 notebook's Cell-8 launch command is
deliberately untouched: the #26 T4 acceptance run (monotonic 2-epoch
descent; peak reserved VRAM < 9.5 GB) must stay reproducible exactly
as prepared, and mixing its evidence with a curriculum-modified data
order would muddy the criterion.  The trainer enforces the wiring
fail-closed (sampler-driven loader + val-split validation signal +
curriculum⇒val_loader) and the checkpoint payload carries the
curriculum state with symmetric fail-closed restore — a curriculum
run cannot silently resume as a plain run or vice versa.

**Evidence:** `TestTrainerIntegration`, `TestCanonicalCLIWithCurriculum`
(end-to-end subprocess smoke; checkpoint carries the curriculum
state).

**Affected:** #27 (closed), #26 (trainer/CLI), #36 (notebook Cell 8
unchanged), #43.

## D-V2-040 — 2026-11-04 — The Stage-2 optimizer groups and the canonical CLI mapping (#29/#31)

The Stage-2 Trajectory Balance trainer owns **two optimizer param
groups**, pinned by the issues: the policy group continues the Stage-1
Profile-A setting (AdamW, lr 3e-4, weight decay 1e-5, clip 1.0) because
Stage 2 *resumes* the supervised-trained architecture and precision path
— changing optimizer semantics between stages would confound the
behavioural-prior starting point; the seed-conditioned log-partition
head ``log Z_theta(P, s_0)`` gets its **own group at lr_Z = 1e-2 with
zero weight decay** (the #29-pinned rate; the partition function is not
a regularization target and must be free to track the flow scale
quickly).  The legacy ``main.py --mode rl`` of issue #31 is mapped to
the V2-canonical ``scripts/train_stage2.py`` (the clean room has no
``main.py`` and no ``--algorithm`` flag: Stage 2 **is** GFlowNet-TB per
#45 §12.3 — PPO/DPO are not production alternatives; the branching CLI
is structurally absent, D-V2-001).  ``--resume-auto`` semantics are
fail-closed: resume the Stage-2 checkpoint if present, else initialize
from the verified Stage-1 checkpoint, else refuse — Stage 2 never
trains from random weights.

**Evidence:** `tests/test_gflownet_tb.py::TestToyDagFlowBalance`
(the lr_Z group pinned at 1e-2; both groups optimize; the loss drops
>100x with logZ → the true partition function),
`tests/test_stage2_cli.py` (init provenance + resume + fail-closed).

**Affected:** #29 (closed), #31 (closed), #26 (the Stage-1 weights the
CLI consumes), #43 (the graceful-stop contract the loop honors).

## D-V2-041 — 2026-11-04 — The rollout conventions of the production Stage-2 bridge (#29)

`syntree/engine/rollout.py` (the policy-driven rollout driver) fixes
the on-policy conventions, each mirroring an existing supervised
contract so the rollout distribution matches the training distribution:

* **Pocket frame and cap (D-V2-041a).**  Rollout environments carry a
  pocket **capped once by the documented centroid-distance pruning**
  (Profile-A 700, referenced to the pocket's own centroid — no crystal
  ligand exists at rollout time), so the environment's anchor menu and
  the policy's anchor candidates coincide exactly.  The pre-seed state
  is centred on the pocket centroid; growth states live in the
  **placed-seed centroid frame** (the rollout analog of the supervised
  crystal-ligand-centroid frame, D-V2-030).
* **Handle pool (D-V2-041b).**  The pool is the molecule-side
  growth-handle construction of the supervised loader (D-V2-031:
  reaction-template matches at non-junction atoms) **intersected with
  the environment's active capacity bookkeeping** — every sampled
  handle is executable, and the environment still enforces the exact
  chemistry fail-closed (mask drift is a loud error, never a silent
  skip).
* **Parity determination (D-V2-041c).**  Whether a junction creates a
  stereocentre is decided **before** the action commits, by stepping a
  *cloned environment state* (shallow clone: immutable members shared,
  per-trajectory state deep-copied) with ``chiral=None`` — the
  environment's own assembly code is the single source of truth, and
  the #24 rule (score the parity action exactly where the junction
  creates a stereocentre; ``chiral=None`` otherwise, so terminal
  objects stay a deterministic function of the scored action tuple) is
  preserved.  The prediction is verified fail-closed against the
  realized product.
* **Catalog bridges (D-V2-041d).**  Growth blocks derive their
  block-side ``HandleSpec`` (attachment + leaving atoms) by matching
  the certified registry's **block-side reactant SMARTS, parsed from
  the string** (the D-V2-031 RDKit segfault hazard: never template
  objects); the #40 curation guarantee (exactly one coupling handle
  per class) is enforced fail-closed.  Seed blocks — which are *not*
  curated coupling blocks and may expose several growth sites — derive
  their handles from the **molecule-side** construction
  (``grow_handle_pool`` semantics: one spec per (atom, class) site,
  capacity 1, no leaving atoms), which makes the environment's
  post-placement menu coincide with the policy pool by construction.
* **Undecided primary handle.**  Rollout decision states carry zeros
  in the ``handle_features``/``handle_pos`` policy-level fields (the
  same value supervised terminal/stop rows carry; the backbone does
  not consume these fields and the pool enters the #22 head through
  its own inputs — no exposure-bias surface exists).

**Evidence:** `tests/test_gflownet_tb.py::TestPolicyRolloutDriver`
(complete valid trajectories with grad-connected tensor log-probs,
determinism under a fixed generator, legality masks, forced stop,
clean backward into both groups, fail-closed contracts).

**Affected:** #29 (closed), #31 (the CLI drives the driver), #42 (the
driver consumes the #24 torsion head and the #42 terminal oracle in
the integrated generation loop — part of #42's remaining
integrated-generation wiring now exists), #22/#24/#39 (the heads the
driver samples from), #38 (the TB semantics the records feed).

## D-V2-042 — 2026-11-04 — The docking-backend injection point (#30)

``ThreeDReward(docking_scorer=...)`` makes the #30 statement "a real
docking engine replaces the surrogate without touching the MDP/TB
contracts" **structural**: the backend returns a ``DockingResult``
(``ΔG_dock`` + the attractive-only LJ component + a version-labelled
backend identity), the default is the documented surrogate
(:func:`surrogate_docking_scorer` — soft-core LJ + softened Coulomb,
bit-identical to the pre-injection computation), and the #30 clamp /
units / sign conventions are applied to **any** backend's scores
exactly as specified.  The backend is part of the frozen configuration
(#45 §12.4: swapping a backend is a changed training objective — a new,
explicitly-versioned instance, never a mutation); every reward's
telemetry records the backend identity (provenance).  Non-finite
scores, wrong return types, and non-callable scorers fail closed.  The
actual docking engine integration is owned by the evaluation stack
(#34/#35 — external software on the authorized runtime); #30's own
scope — the stationary multi-objective reward with no batch-relative
term, the floors, the caps, and the supersession interface — is
complete.

**Evidence:** `tests/test_reward_docking_backend.py` (12 tests:
default bit-identical, injected conventions incl. both clamp bounds,
stationarity under any backend, fail-closed contracts, adversarial
floor, provenance); the 50-test reward reconciliation battery +
Suite B stay green unchanged.

**Affected:** #30 (closed), #29 (the reward the TB loop consumes),
#34/#35 (the future real docking engine injects here), #26 (the
telemetry convention extended with ``docking_backend``).

## D-V2-043 — 2026-11-05 — Reactant-form graft matching in the molecule-side handle pool (#26/#29, D-V2-031 amendment)

The first canonical `train_stage1.py` launch on the authorized T4
(S-V2-19, after the bit-exact data reconstruction) surfaced a real
fail-closed stop: `pdbbind_4itp: the step-2 junction (atom 0, class
'alkylzinc') is not in the detected handle pool`.  Root cause: D-V2-031
matched the reaction's molecule-side **reactant template** against the
**product-form crystal graph** — but for the nine non-empty
molecule-side grafts of the certified taxonomy (rxn_09..15, rxn_23,
rxn_26: Br / Cl / ZnCl / B(OH)₂ leaving groups) the reactant's
distinguishing feature is, by construction, absent from every crystal
product graph (the leaving group departs in the reaction; a crystal
ligand contains no Zn).  Direct matching could therefore never see a
legal handle of those classes — the Negishi step of 4ITP (a product-form
methyl that reacts as CH₂–ZnCl) was unreachable, and the same blindness
silently applied to every aryl C-H (aryl-bromide/chloride/boronic-acid
classes) in every intermediate.

**The rule (the extractor's own model, mirrored):** an atom is a growth
position for a reaction iff the reaction's certified molecule-side graft
attaches there (`apply_graft`, the exact #18 reactant-reconstruction
path, H-count delta included) **and** the reactant template then matches
at that position; atoms whose graft cannot attach (no replaceable H,
valence, sanitization) are simply not handles — the extractor's
`graft_failed` vocabulary, not an error.  Empty-graft classes (amine /
alcohol / thiol) keep the direct product-graph semantics (unchanged).
The derivation is deliberately **not memoized**: the 4ITP ligand parsed
from the PDBbind mirror and built from MOAD PDB residues share one
canonical SMILES with different atom orderings, so a SMILES-keyed index
cache is unsound (measured first as an order-dependent loader failure);
the measured cost is ~1 ms per 27-reaction pool — no cache is warranted.
This corrects **both** consumers: the supervised pool (#26) and the #29
rollout driver's generation-time pool, which had the same blindness.

**Evidence:** `tests/test_supervised_dataset.py` (+2 regression tests:
the 4ITP methyl/alkylzinc case, aryl C-H graft visibility + the
quaternary-carbon negative; the direct-match blindness pinned);
`tests/test_trainer.py` (the CPU descent smoke widened to 6 epochs —
one optimizer step per epoch made the 2-epoch mean comparison
noise-dominated under the honest ~15-row action space; descent verified
at torch seeds 0..9); the loader materializes all 241 train + 101 val
items across 3 shuffle orders with the bit-exact 194-trajectory set +
regenerated pockets (0 failures).  Suite: **1615 passed / 20 skipped**.

**Affected:** #26 (the loader criterion — the training run unblocked),
#29 (the rollout pool), #18 (the graft-spec machinery is now consumed
by the supervised path), #42 (nothing — the terminal oracle is
untouched).

## D-V2-044 — 2026-11-05 — The evaluation-stack environment reconciliation (aizynthfinder 4.4.1 × rdkit 2026.09.1 × numpy 1.26) (#34/#44/#35)

**Problem.**  The independent retrosynthesis planner required by
#34/#44/#35 is `aizynthfinder` 4.4.1, whose metadata pins
`rdkit>=2023.9.1,<2024.0.0` and `numpy<2.0.0`.  The V2 chemistry stack
and its 1683-test suite are verified under modern rdkit (2026.09.1), and
the data pipeline needs a pyarrow that co-exists with numpy<2 (the
aizynthfinder dependency dask imports pyarrow; pyarrow ≥22 requires
numpy ≥2 at import time).

**Chosen approach.**  One environment running rdkit 2026.09.1 + numpy
1.26.4 + pyarrow 16.1.0 + aizynthfinder 4.4.1 (onnxruntime policies —
no TensorFlow).  The metadata conflict with aizynthfinder's
conservative rdkit pin is **runtime-verified rather than obeyed**: the
full planner pipeline (ONNX policy loading, rdchiral template
application, tree searches on both stocks, route extraction) was
exercised end-to-end under rdkit 2026.09.1 — acetanilide solved from a
mini stock, aspirin solved with 36 routes from the full 17.4M ZINC
stock, and the 194-record corpus evidence run completed without a
single rdchiral/rdkit failure.  The conflict is recorded here and in
`docs/eval/retrosynthesis_tracks.md` instead of silently downgrading
the project's chemistry stack (the repo's own floor
`rdkit>=2023.9.1` is satisfied either way, but the suite's verified
behavior is the modern one).

**Alternatives considered.**  (a) Obey the pin and downgrade to rdkit
2023.9.6 — satisfies all metadata but re-validates 1683 chemistry
tests against an unverified older toolchain; rejected.  (b) A separate
planner-side venv — the subprocess isolation already provides the
process boundary; a second interpreter adds operational complexity
without removing the version question; rejected.  (c) The legacy
TensorFlow policy backend — heavier and unmaintained; the ONNX backend
is the package's current default; rejected.

**Evidence.**  The evidence run of `scripts/run_benchmarks.py`
(2026-11-05): Track A solve 0.2667, Track B solve 0.7333 over 30 unique
structures, 0 planner failures; `pip check` reports exactly the one
documented conflict (aizynthfinder × rdkit).

**Affected:**  #34, #44, #35 (the evaluation stack), the dev
environment bootstrap (pyarrow pinned <22 until aizynthfinder lifts
numpy<2), future T4 runtime setup notes.

## D-V2-045 — 2026-11-05 — The evaluation corpus semantics: unique-structure denominators, two-track separation, and planner checkpoint-resume (#34/#44/#35)

**Problem.**  Three operational/scientific questions surfaced during
the first real evaluation run: (1) the #18 corpus is 194 records over
only **30 unique ligand structures** (the same crystal ligand is
accepted for many pockets — e.g. the inositol-type sugar 36×) — which
denominator defines a solve rate?; (2) how to keep two planner runs
inside bounded tool-call budgets without losing work; (3) how to report
per-target retrospective outcomes when several frozen targets share
one generated molecule.

**Chosen approach.**
1. **Planner targets are deduplicated unique canonical SMILES**;
   the solve-rate denominator is the unique-structure count (30), and
   the corpus multiplicity (194 records over 30 structures) is recorded
   in the report (`num_unique_valid_smiles` + the per-record corpus
   list) — never hidden, never silently weighted.
2. **Checkpoint-resume evaluation**: `run_benchmarks.py` /
   `run_comparative_benchmark.py` pass the aizynthcli `--checkpoint`
   protocol and retry across per-attempt timeouts; the aizynthcli
   checkpoint is **positional** (input order must be identical across
   resumes — guaranteed by the deterministic corpus order), and
   finished planner outputs are **reused** only when their target set
   exactly matches the corpus (#45 §15 artifact-reuse; mismatches
   recompute loudly).
3. **Per-target retro vectors map every frozen target whose structure
   file carries a molecule to that molecule's solve flag** — pockets
   sharing a generated molecule each see its outcome (33 test targets
   over 12 unique structures → 33 per-target entries).

**Alternatives considered.**  Per-record weighted solve rates (weight
duplicated molecules by multiplicity — hides the uniqueness structure
of the corpus behind a pocket-count artifact); id-keyed checkpoint
resume (the aizynthcli protocol is positional — we guarantee order
instead of patching the planner); no reuse (re-searching 194 targets
per invocation wastes ~30 min per call for identical results).

**Evidence.**  The Track A run (4 checkpoint-resumed calls, 53→99→147→
194 lines), the final both-track report (220 KB), and the comparative
run's `per_target_retro` counts (33 per track from 12 unique
structures).

**Affected:**  #34, #44, #35, the evidence artifacts in
`docs/eval/evidence/`, future evaluation reruns on the T4 runtime.

## D-V2-046 — 2026-11-05 — The profile-digest scope: operational scheduling knobs are not training semantics (#26/#43)

**Context.**  The authorized T4 T-chain's first run (S-V2-21) failed T2/T3
with `TrainerStateError: checkpoint profile digest does not match the
active profile`: a checkpoint saved under `--time-budget-seconds 40`
refused a budgetless resume.  `TrainerProfile.digest()` hashed
`asdict(self)` — every field, including session-scheduling configuration.

**Chosen approach.**  The digest hashes only the training-semantics fields
(`_DIGEST_FIELDS`: batch/accum/pocket-cap/LR/decay/clip/precision/seed/
profile-name).  `time_budget_seconds`, `num_workers`, `log_interval_steps`
and `checkpoint_interval_steps` are excluded: #43 requires configurable
time-aware checkpointing across sessions, so a checkpoint saved under one
session's budget must resume under another's.  The full profile (budget
included) is still recorded in the checkpoint payload for diagnostics.
Pinned by `test_operational_knobs_do_not_fragment_the_profile_digest` and
`test_resume_after_time_budget_stop_under_a_different_budget`.

**Alternatives considered.**  Passing the same budget on every resume
(pushes a scheduling detail into every continuation invocation and breaks
the moment a session's budget differs); hashing a "compatibility version"
constant instead of the fields (loses the per-field fail-closed property).

**Evidence.**  The T-chain run 1 log (profile-digest failure), run 3
(T1 `CONTROLLED_STOP_OK budget=15` → budgetless T2/T3 resumes →
`TRAIN_CHAIN_DONE`); the two regression tests; suite 1686/22 at `ee28713`.

**Affected:**  `syntree/engine/trainer.py`, #26, #43, every cross-session
checkpoint continuation.

## D-V2-047 — 2026-11-05 — Completed-run resume is a no-op: progress.epoch advances past a finished epoch (#43)

**Context.**  T-chain run 2 exposed a silent exact-continuation violation:
a step-42 `max_epochs_reached` checkpoint resumed and re-trained epoch 1 to
step 63 (21 repeated micro-batches).  `progress.epoch` held the index of
the last *started* epoch, so `range(start_epoch, max_epochs)` re-ran the
completed final epoch.  An existing test manually assigned the correct
state and thereby papered over the missing bookkeeping.

**Chosen approach.**  On epoch completion the trainer advances
`progress.epoch = epoch + 1` (mid-epoch stops keep the in-progress index,
which equals the completed count); `epochs_completed` is `progress.epoch`
directly.  Resuming a completed run is now a provable no-op — steps,
history and weights do not move (pinned by
`test_resume_of_a_completed_run_is_a_noop`); the manual-state test remains
as the epoch-extension contract.

**Alternatives considered.**  A separate `run_complete` flag on the
checkpoint (a second source of truth that can drift from the counters);
clamping `start_epoch` at resume time (hides the invariant instead of
maintaining it).

**Evidence.**  T-chain run 2 (step 42 → 63 repeat) vs run 3 (T3 resumes
from the killed run's step-12 checkpoint to exactly 42, `history_len` 42,
`epochs_completed: 2`); the regression test; suite 1687/22 at `925071d`.

**Affected:**  `syntree/engine/trainer.py`, #43, the T-chain's T3/T4
verifiers, `runtime_cross_session_continue.sh`'s prefix-identity
assertion.

## D-V2-048 — 2026-11-05 — max_epochs is run length, not a step semantic: cross-session continuation may extend it (#43)

**Context.**  Writing the #43 cross-session continuation driver exposed a
design tension: continuing a persisted run on a later session requires
extending `max_epochs` past the checkpoint's saved value, but `max_epochs`
was a digest field (after D-V2-046), so an extended resume would fail
closed.

**Chosen approach.**  `max_epochs` joins the operational set outside the
digest.  #43's premise is that full-scale training exceeds a single
session's lifetime; the epoch bound schedules run LENGTH, not what a step
IS.  The per-step contract (batch/accum/pocket-cap/LR/decay/clip/
precision/seed/profile-name) still fails closed — the regression test
moves `max_epochs` to the excluded-knobs assertion.

**Alternatives considered.**  A dedicated `--extend-epochs` resume flag
that rewrites the checkpoint profile under a compatibility exception
(special-case machinery for what is a legitimate continuation); keeping
`max_epochs` digested and requiring new sessions to continue under the
original bound (forbids the exact multi-session scaling #43 exists for).

**Evidence.**  `test_operational_knobs_do_not_fragment_the_profile_digest`
(max_epochs=7 digests equal; the eight semantics fields still differ);
suite 1687/22 at `778ff1c`.

**Affected:**  `syntree/engine/trainer.py`, #43,
`scripts/runtime_cross_session_continue.sh`.
