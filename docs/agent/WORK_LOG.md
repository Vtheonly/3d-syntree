# 3D-SynTree V2 — Agent Work Log (clean-room era)

> Chronological record of agent engineering sessions. Every entry
> distinguishes **attempted**, **implemented**, **tested locally**,
> **tested on T4**, and **scientifically validated** (#45 §19.1). Never
> paste secrets, credentials, or authentication links here.
>
> Pre-reset history (branch commits `44edb8b..6dbb66a`, issues #5–#7
> closure comments of 2026-10-31) is superseded by the clean-room reset;
> see decision D-V2-001. Git history retains it for forensics.

---

## S-V2-01 — 2026-11-01 — Clean-room reset + foundation (#5) + schema (#6) + norms (#7)

- **Agent/session:** autonomous lead-engineering session (Z.ai), UTC.
- **Repository / branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `6dbb66a` (pre-reset head, 147 files on branch).
- **Issues worked:** #5 ARCH-01, #6 ARCH-02, #7 MATH-01 (all reopened this
  session under the owner's clean-room directive; see below).

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full (all sections incl. §24 greenfield mandate, §25
  continuation protocol, §28/§29 remote-runtime rules).
- Fetched all 41 issues (#5–#45, both states) with comments; verified:
  pre-reset closures were #5, #6, #7 only; #8–#44 open; #45 open.
- Read pre-reset agent docs (STATUS/WORK_LOG/DECISIONS/DEPENDENCIES/
  HANDOFF), architecture doc, `syntree/core/schema.py`,
  `syntree/models/equivariant.py` — **as specification extraction only**
  (validated contract rules recorded in the new DECISIONS.md, D-V2-002).
- Owner directive received: the pre-reset "V2" (incremental repair of the
  legacy tree still carrying 88 legacy files + 800+ V1 tests) is rejected;
  the branch must be emptied and V2 rebuilt from zero.

### 2. Clean-room purge (owner directive; #5 core scope)

- `git rm -r -f .` → all 147 tracked files staged for deletion; working
  tree left with `.git` only.
- Commit `075f86d` "reset(v2)!: purge 3d-synV2 to empty root for
  clean-room V2 rebuild (#5)" — pushed to origin (fast-forward
  `6dbb66a..075f86d`), **no force-push, history preserved**.
- **Verification:** `git ls-files` empty; GitHub Trees API at `075f86d`
  returns 0 entries; spot-checked #5-named legacy targets
  (`craft/craft_mode.py`, `syntree/engine/dpo.py`,
  `syntree/models/pocket_encoder.py`, `syntree/data/crossdocked.py`)
  → HTTP 404 on the branch. No dangling imports possible (nothing left).
- **Issues reopened with rationale comments:** #5, #6, #7 (PATCH state =
  open + explanatory comment each); #45 verified still open (never
  touched).

### 3. Fresh V2 foundation (#5, in progress at time of this entry)

- New files (all newly written): `LICENSE` (MIT), `.gitignore`,
  `pyproject.toml` (src-layout, `syntree` 2.0.0.dev0, deps
  `torch`/`torch_geometric`, decision D-V2-003), `README.md`
  (first-class V2 description per #45 §4 with honest status table),
  `docs/architecture/ARCHITECTURE.md` (global spec, contract statuses),
  `docs/data/RAW_SOURCE_PROVENANCE.yaml` (verified upstream facts carried
  forward), `docs/agent/{STATUS,WORK_LOG,DECISIONS,DEPENDENCIES,HANDOFF}
  .md` (this suite).
- Dev environment: Python 3.12.16, torch 2.15.0+cpu, torch_geometric
  2.8.0.post1, pytest 9.0.2 (CPU sandbox — no GPU claims made).

### 4. #6 ARCH-02 — implemented, tested locally, closed

- `src/syntree/core/schema.py` (new, 745 lines): frozen
  `StateTensorContract` (12 canonical tensor fields + `schema_version`,
  `SCHEMA_VERSION = 2`), `validate_state_tensors` /
  `validate_state_batch` (fail-closed dtype/rank/shape, cross-field
  lengths, batch-index ranges, atomic-number bounds 1..100, finiteness),
  `collate_state_batch` (single batching path; canonical
  `pocket_batch`/`ligand_batch` materialised via PyG `follow_batch` with
  aliases dropped; explicit empty-ligand t=0 states; only two documented
  normalisations 0-dim→`[1]` and 1-D `[64]`→`[1,64]`; pre-batched /
  multi-graph samples rejected; per-sample stamps verified then stripped;
  batch stamped + re-validated), `stamp_schema_version`,
  `is_schema_version_compatible`, `TensorContractError(ValueError)`, and
  the hard 67-D rejection with the exact issue-pinned message taking
  precedence over missing-field diagnostics.
- `tests/test_tensor_contracts.py` (new, 93 tests): both acceptance
  criteria + full fail-closed contract + version stamping + every
  collator rule.
- Commit `0001087`, pushed (`0f5d74a..0001087`). Local CPU test run:
  `python -m pytest tests/test_tensor_contracts.py -q` → **93 passed**.

### 5. #7 MATH-01 — implemented, tested locally, closed

- `src/syntree/models/equivariant.py` (new, 348 lines): `SAFE_NORM_EPS =
  1e-5`, `safe_norm` / `safe_unit_vector` (exact issue signatures and
  formulas), `PaiNNInteraction` (paper Eqs. 7–8: shared φ(s_j) atomwise
  network split into three gates × bias-free radial filter banks; vector
  message = gated `v_j` + gated non-singular direction; scatter-add;
  residual), `PaiNNMixing` (paper Eqs. 9–10: U/V linear maps,
  `safe_norm(||Vv||)`, invariant `⟨Uv,Vv⟩` coupling, shared scaling
  network). Equation form verified against arXiv:2102.03150v4 (PDF
  fetched and Eqs. 7–10 extracted 2026-11-01; decision D-V2-004). No
  NaN-masking / fp16 clamps anywhere (D-V2-005).
- `tests/test_mathematical_singularities.py` (new, 45 tests): both
  acceptance criteria (exact pinned names
  `test_safe_norm_zero_distance`, `TestSO3Covariance::
  test_directional_vectors_exact_rotation_covariance`) plus analytic
  gradient identity, |∇| ≤ 1 bound across 11 magnitude scales,
  torch.norm asymptotics, unit boundedness across 30 orders of magnitude,
  degenerate geometry through the blocks (coincident atoms, zero edge
  vectors, zero vector features, empty edge sets) with finite bounded
  gradients (block-level documented bound H/ε, observed ~1.1e5), NaN
  propagation (fail-loudly), and invariance/equivariance/translation
  tests of interaction, mixing, and their composition under random rigid
  motions.
- Commit `00ec653`, pushed (`0001087..00ec653`). Local CPU test run:
  `python -m pytest tests/ -q` → **138 passed** (93 + 45).

### 6. Verification summary (all local CPU; no GPU/T4 claims)

- Environment: Python 3.12.16, torch 2.15.0+cpu, torch_geometric
  2.8.0.post1, pytest 9.0.2.
- `python -m pytest tests/ -q` → **138 passed, 0 failed, 0 warnings**.
- #5 acceptance: purge verified (Trees API 0 entries at `075f86d`;
  legacy targets 404); `rg 'craft|download_mode|dpo|pocket_encoder|
  crossdocked|kaggle_loader' src/syntree/` → 0 matches; clean git status.
- #6 acceptance: `TestConformingStates` 8/8; `TestLegacyFrankenRejection`
  8/8 (exact message text asserted verbatim).
- #7 acceptance: `test_safe_norm_zero_distance` 1/1 (exact 0.0 output and
  strictly-0.0 gradient — fp32 and fp64); `TestSO3Covariance` 3/3.
- Issues #5/#6/#7 closed with evidence comments (this session).

### 7. Commits / pushes (session total)

| Commit | Content | Push range |
|---|---|---|
| `075f86d` | reset(v2)!: purge branch to empty root | `6dbb66a..075f86d` |
| `0f5d74a` | feat(arch-01): fresh V2 foundation | `075f86d..0f5d74a` |
| `0001087` | feat(arch-02): frozen state tensor contract + canonical batch path | `0f5d74a..0001087` |
| `00ec653` | feat(math-01): non-singular norms + PaiNN blocks | `0001087..00ec653` |
| (this docs commit) | docs(agent): session record — statuses, SHAs, handoff | — |

### 8. GitHub issues updated

- Reopened with rationale: #5, #6, #7 (earlier this session).
- Closed with evidence comments: #5, #6, #7 (end of session).
- Verified untouched and OPEN: #45 (permanent).

### 9. Next actions

See `HANDOFF.md` §1: next dependency-ready task is #21 MODEL-01
(BipartitePaiNN backbone) on the critical path; parallel-safe:
#8, #9, #10, #11, #14, #16, #32; #36 requires user authorization.

---

## S-V2-02 — 2026-11-01 — BipartitePaiNN union-graph backbone (#21)

### 1. Session opening state discovery

- Read #45 in full; inspected branch `3d-synV2` (clean at `0692c23`, in
  sync with origin), `git status --short` empty; read STATUS /
  DEPENDENCIES / DECISIONS / HANDOFF / ARCHITECTURE / WORK_LOG; re-fetched
  issues #5–#44 states from the GitHub API (#5/#6/#7 closed, #8–#44 open,
  #45 open-permanent).
- Handoff S-V2-01 §1 directed: next task #21 MODEL-01 (both hard deps
  closed). Verified the deps' deliverables exist in the clean-room tree
  (`syntree/core/schema.py`, `syntree/models/equivariant.py`); re-read
  issue #21 in full plus boundary issues #9, #22, #23, #39 (interface
  consumers).
- Fresh dev environment installed (sandbox): Python 3.12.16,
  torch 2.15.0+cpu, torch_geometric 2.8.1, pytest 9.0.2; baseline
  reproduced **138/138 passed** before any change.

### 2. #21 MODEL-01 — implemented, tested locally, closed

- `src/syntree/models/bipartite.py` (new, 864 lines):
  - `BipartitePaiNN`: 8 PaiNN message-then-update layers composing the #7
    blocks (D-V2-004 equations), d=256 (vector 3×256), 20 Gaussian RBFs,
    r_cut=5.0 Å frozen as `BACKBONE_*` constants; shared atomic-number
    embedding (z∈[1,100], bound-matched to `schema.MAX_ATOMIC_NUMBER`,
    row 0 reserved), node-type embedding (0 pocket / 1 ligand), bias-free
    charge MLP, handle-flag embedding; vector features start at zero;
    union node ordering [pocket, ligand] (per-graph-major inside each
    block for collated batches); non-reentrant
    `torch.utils.checkpoint` per layer while gradients are enabled.
  - `build_union_radius_graph`: the one authoritative exact radius graph
    (both directions, no self-loops, no cross-graph edges, all three pair
    families); `safe_norm` threshold (ε-inclusive boundary, no_grad
    structure, differentiable `edge_vec`/`edge_dist` only); O(N²)
    transient memory documented.
  - `GaussianRadialBasis`: fixed buffer-only 20-Gaussian expansion on
    [0, r_cut], NO envelope (D-V2-007).
  - `BackboneOutput`: invariant scalars, equivariant vectors, union
    metadata, per-graph `graph_context`/`pocket_context` + slice
    properties — the #39-facing surface.
  - Fail-closed validation of the 8 consumed node fields: dtype
    (FP32/FP64 only, D-V2-008), device, atomic-number bounds, shapes,
    finiteness, batch ranges, ≥1 pocket atom per graph, handle flags.
- `src/syntree/models/__init__.py`: re-exports the backbone API.
- `tests/test_bipartite_painn.py` (new, 69 tests): both pinned acceptance
  criteria — scalar invariance / vector equivariance under random proper
  rotations (+translations, pooled contexts, handle flags, pocket-only,
  batched) at 1e-9 tolerances in float64, and clean N_l=0 pocket-only
  operation — plus frozen-profile pinning (exactly 5,999,104 parameters),
  default-profile FP32 smoke, brute-force graph equivalence, three pair
  families, cross-graph exclusion, coincident-atom safety, rigid-motion
  graph invariance, batched-equals-single-graph runs, pooling vs manual
  means, handle-flag union ordering, checkpointing math-equivalence
  (outputs & gradients identical with/without), inference under no_grad,
  determinism, input non-mutation, encoder decoupling from policy fields,
  and the fail-closed contract matrix.
- Two test-authoring mistakes found and fixed during development (both
  test-side, no module change): a "batch index out of range" case that
  legitimately derived B=2, and a handle-flag propagation assertion that
  ignored message passing.
- Commit `e76d955`, pushed (`0692c23..e76d955`). Issue #21 closed with
  evidence comment (id 6551179046).

### 3. Extra verification beyond the suite (local CPU)

- 36 rigid-motion trials across 12 independently seeded models (d=16,
  L=3): worst scalar invariance error **2.98e-13**, worst vector
  equivariance error **4.62e-14**.
- Empty-edge graph (isolated dimer > r_cut apart) through the full model:
  finite outputs. Default 256/8 profile end-to-end on a canonical
  `collate_state_batch` FP32 batch: forward + backward, finite grads.
- Default profile parameter audit: **5,999,104** (embeddings 25,856 +
  512 + 65,792 + 512; per layer 278,528 interaction + 459,776 mixing;
  × 8 layers) — hand-derived and measured, equal.

### 4. Verification summary (all local CPU; no GPU/T4 claims)

- Environment: Python 3.12.16, torch 2.15.0+cpu, torch_geometric 2.8.1,
  pytest 9.0.2.
- `python -m pytest tests/ -q` → **207 passed, 0 failed** (93 + 45 + 69).
- #21 acceptance 1 (invariance/equivariance): 8/8 SE(3) class tests +
  sweep above. #21 acceptance 2 (N_l=0): 3/3 + in-batch empty-ligand
  graphs.
- Mandatory pre-commit review executed: `git diff --check` (clean),
  `git diff --stat` (3 files, +1848/−2), `git diff --unified=80`
  (reviewed), `git status --short` (clean after commit).

### 5. Commits / pushes (session total)

| Commit | Content | Push range |
|---|---|---|
| `e76d955` | feat(model-01): BipartitePaiNN backbone, radius graph, RBF (#21) | `0692c23..e76d955` |
| (this docs commit) | docs(agent): S-V2-02 session record — statuses, decisions D-V2-007/008, handoff, README/architecture | — |

### 6. GitHub issues updated

- Closed with evidence comment: #21 (this session).
- Verified untouched and OPEN: #45 (permanent).
- STATUS.md table synchronized (4/40 closed: #5, #6, #7, #21).

### 7. Next actions

See `HANDOFF.md` §1: next dependency-ready task is #22 MODEL-02 (handle
selection head; critical path; consumes `BackboneOutput`); parallel-safe:
#8, #9 (owns the RBF envelope, D-V2-007), #10, #11, #14, #16, #32; #36
requires user authorization.

---

## S-V2-03 — 2026-11-01 — Owner-directed GFlowNet reconciliation: engine layer + #38 close (D-V2-009)

- **Agent/session:** autonomous lead-engineering session (Z.ai), UTC.
- **Repository / branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `6828747` (S-V2-02 docs head; code head `e76d955`).
- **Issues worked:** owner reconciliation task over #24, #29, #30, #38, #42
  (the task that resolves the three backlog contradictions: discrete TB vs
  continuous geometry optimisation, multi-parent backward semantics,
  reward non-stationarity).

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full (constitution §10.2/§12.3/§12.4/§18/§25 govern
  this task); fetched #24, #29, #30, #38, #42, #45 bodies (all open);
  read STATUS/DECISIONS/DEPENDENCIES/HANDOFF/WORK_LOG/ARCHITECTURE and the
  existing code head. Confirmed: clean-room closures #5/#6/#7/#21; next
  recommended task was #22; the owner's reconciliation task explicitly
  overrides the queue for this session (direct instruction, #45 §25.3).
- Baseline suite before any change: 207/207 green (re-cloned, `pip install
  -e .`, rdkit 2025.09.4 + scipy 1.14.1 + numpy 2.1.3 + torch 2.15.0+cpu
  installed in the sandbox).

### 2. Implemented (clean-room, from the task specification only)

- `src/syntree/engine/__init__.py`, `environment.py` (~1150 lines):
  - **Level 1 MDP**: `AssemblyEnvironment` with `reset/step/stop`,
    `AssemblyAction(h_t, r_t, B_t, c_t, k_t)`, `BuildingBlock` /
    `HandleSpec` / `JunctionReaction` (generic registry with resonance
    locks; #11 fills the certified taxonomy later), deterministic rigid
    placement realising the exact bin centres
    `phi_k = -pi + (k-0.5)*2*pi/36` with hybridisation angles and
    covalent-radii bond lengths, chirality-branch placement, junction
    history with index remapping, fail-closed validation everywhere.
  - **Leaf semantics (#38)**: `get_removable_leaves` (fragment junction
    graph, degree-1 non-seed fragments), `backward_leaf_probability` =
    uniform leaf deletion `1/N_leaves`.
  - **Level 2 oracle (#42/#13, terminal-only)**:
    `finalize_terminal_molecule` — deep copy, coarse-scan-seeded bracketed
    Brent sub-bin refinement (±5°, soft-core pocket objective,
    `scipy.optimize.minimize_scalar(method='bounded')`), scaffold-constrained
    MMFF94 with 50 kcal/mol/Å² position restraints on seed atoms,
    full diagnostics (angles, objectives, evaluation counts, seed RMSD,
    locked-dihedral drift). STOP-gated (fail-closed otherwise).
  - `PocketDefinition` + documented oracle-internal soft-core
    LJ/Coulomb surrogate (injection point for #14).
- `src/syntree/engine/rl.py` (~330 lines): `ThreeDReward` — stationary
  `R(x) = exp((S_dock + S_contact + S_key + S_props + S_vol)/T)`,
  `R >= 1e-4`, **no batch parameter in the signature at all**; component
  telemetry; fail-closed on non-finite physics and bad configuration.
- `src/syntree/engine/gflownet.py` (~330 lines):
  `TrajectoryBalanceEngine` — the exact pinned TB equation
  `L = (log Z + Σ log P_F − Σ log P_B − log R)²` with the uniform-leaf
  backward terms (`-log num_leaves` per assembly step; STOP term zero by
  proof), learnable `log_z` (FP32 default / FP64 verification, mirroring
  D-V2-008), batch loss (uncoupled mean), fail-closed contracts.
- `tests/test_gflownet_reconciliation.py` (new, 50 tests): the four
  pinned reconciliation tests (determinism bit-for-bit incl. fresh
  catalogs; terminal-oracle isolation incl. counters/state immutability/
  rigid covalent preservation; reward stationarity incl. 8-decimal
  batch-composition invariance + signature inspection; backward leaf
  probability incl. branched periphery + symmetric convergence +
  renumbering) **plus** the enumerable toy-DAG flow accounting (19
  quotient states / 31 trajectories: every TB residual < 1e-9 through
  the real engine with `log Z` at the true total flow; terminal outcome
  probabilities = `R(x)/Z` to 1e-9 relative; two-order mass split into
  exact halves; `Σ R(x) = Z`), amide resonance snap + post-MMFF
  planarity, Brent-vs-dense-200-point-scan on the same sequential
  baseline, and the pinned TB-equation/STOP-zero/fail-closed contracts.
- `docs/architecture/MDP_STATE_SPACE.md` (new, ~260 lines): the formal
  #38 specification — two-level architecture, order-quotiented state
  definition, the three backward-policy proofs (proper distribution,
  backward path measure identity by induction, STOP uniqueness), the TB
  theorem with proof, the worked multi-path example, adversarial-case
  coverage, interface/scope boundaries.
- `pyproject.toml`: dependencies grown per D-V2-003 (numpy, scipy, rdkit).
- `docs/agent/DECISIONS.md`: decision **D-V2-009** (owner task label
  "D21") — full rationale, alternatives, evidence, affected issues,
  scope boundaries.

### 3. Extra verification beyond the suite (local CPU)

- Probed the RDKit API surface before writing the oracle (MMFF
  position-constraint signature, covalent radii, dihedral accessors,
  ETKDG same-seed bit-for-bit determinism, AddHs/RemoveHs prop
  survival, QED/Volume) — all confirmed on the installed build.
- Diagnosed and fixed during development (all test-side or
  fixture-side; one module-side improvement): a broadcast error in the
  test pocket generator; stale handle indices in two fixtures; a
  2-vs-3-tuple unpack; an invalid naphthalene SMILES; the toy-DAG test
  initially forgot to set `log Z` to the true total flow and re-stepped
  an already-replayed env; the bond-length assertion moved from exact
  equality to 1e-12 (Rodrigues composition costs ~1 ULP).  One genuine
  module improvement: pure Brent over the full ±5° interval locked into
  a local basin on the multimodal soft-core landscape → replaced by the
  coarse-scan-seeded Brent hybrid (documented, and now *beats* the dense
  reference scan).

### 4. Verification summary (all local CPU; no GPU/T4 claims)

- Environment: Python 3.12.16, torch 2.15.0+cpu, torch_geometric 2.8.1,
  rdkit 2025.09.4, scipy 1.14.1, numpy 2.1.3, pytest 9.0.2.
- `python -m pytest tests/test_gflownet_reconciliation.py -q` →
  **50 passed, 0 warnings**.
- `python -m pytest tests/ -q` → **257 passed, 0 warnings** (93 + 45 + 69
  + 50).
- Mandatory pre-commit review: `git diff --check` (clean), `git diff
  --stat`, `git diff --unified=60` (reviewed), `git status --short`.

### 5. Commits / pushes (session total)

| Commit | Content | Push range |
|---|---|---|
| `d3d186a` | fix(gflownet): reconcile discrete TB flow balance with terminal geometry relaxation and stationary reward (engine layer + 50 tests + MDP_STATE_SPACE.md + D-V2-009 + deps) | `6828747..d3d186a` |
| (this docs commit) | docs(agent): S-V2-03 session record — statuses, dependency cycle broken, handoff refreshed | — |

### 6. GitHub issues updated

- **Closed with evidence comment: #38** (all five acceptance criteria
  verified — see STATUS.md row).
- Updated with re-alignment addenda + evidence comments: #24, #29, #30,
  #42 (all remain open with precise remaining scope).
- Verified untouched and OPEN: #45 (permanent).
- STATUS.md table synchronized (5/40 closed: #5, #6, #7, #21, #38).

### 7. Next actions

See `HANDOFF.md` §1: next dependency-ready task is #22 MODEL-02 (handle
selection head; critical path; consumes `BackboneOutput`; the engine now
exposes the concrete `active_handles()` menu it must score). Parallel-safe:
#8, #9 (owns the RBF envelope, D-V2-007), #10, #11, #14, #16, #32. #36
requires user authorization. #42's T4 benchmarking item remains gated on
#36/#43.

## S-V2-04 — 2026-11-01 — #22 MODEL-02 closed: agency-enabled handle-selection head (D-V2-010)

- **Agent/session:** autonomous lead-engineering session (Z.ai), UTC.
- **Repository / branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `589a843` (S-V2-03 docs head; branch verified equal
  to `origin/3d-synV2` before any change; working tree clean).

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full; fetched the #22 body and re-read issues #5–#44
  states from GitHub (clean-room closures confirmed: #5, #6, #7, #21, #38;
  open PR #46 unrelated to this task). Read STATUS / DECISIONS /
  DEPENDENCIES / HANDOFF / WORK_LOG / ARCHITECTURE / MDP_STATE_SPACE and
  the code head (`schema.py`, `equivariant.py`, `bipartite.py`,
  `environment.py`, `gflownet.py`, `rl.py`).
- Dependency check for #22: both hard deps closed (#6 @ `0001087`, #21 @
  `e76d955`); `BackboneOutput` and `AssemblyEnvironment.active_handles()`
  inspected as the concrete upstream/downstream surfaces.
- Baseline suite before any change (fresh sandbox install: torch 2.15.0+cpu,
  torch_geometric 2.8.1, rdkit 2026.9.1, scipy 1.14.1, numpy 2.1.3,
  pytest 9.0.2; Python 3.12.16): **257/257 green**, matching the S-V2-03
  handoff record.

### 2. Implemented (clean-room, from the issue + handoff specification)

- `src/syntree/models/handle_head.py` (new, ~640 lines):
  `HandleSelectionHead` — the two-stage cross-attention scorer of
  `P(handle_t | s_t)` over the active handle pool (decision D-V2-010):
  - per-handle invariant embedding = descriptor MLP ([H, 64] frozen schema
    width) + backbone scalar of the attachment atom + raw envelope-free
    Gaussian RBF of the handle-to-pocket-centroid distance (#9 owns the
    envelope, D-V2-007) + **equivariant directional invariant**
    `<v_atom, x_handle − centroid_g>` from the backbone vector channel;
  - per-graph query = pooled ligand summary + pooled pocket context;
  - stage 1: single-query-per-graph scaled dot-product attention over the
    graph's handle keys (attention weights themselves a valid per-graph
    law) with attention-weighted value pooling;
  - stage 2: additive Bahdanau-style scoring (query + key + pooled handle
    context) → per-graph `log_softmax` over **exactly the active handle
    rows** (segment softmax via scatter: max-shifted, on-device, stable);
  - `HandleSelectionOutput`: auditable logits / log_probs / probs /
    attention / entropy / counts + the #23-facing `handle_embeddings` and
    `pooled_handle_context`; `rows_for_graph` helper;
  - fail-closed validation (FP32/FP64-only per D-V2-008 with dtype/device
    matching, shapes, index ranges, legacy 67-D descriptor rejection,
    cross-field atom↔graph consistency, non-finite inputs, empty pool,
    pocket-only states) and post-forward finiteness assertions (#45 §16);
  - `audit()` parameter/dimension/trainability report; frozen production
    profile constants (backbone 256 / head 128 / 20 RBFs / 12 Å cutoff,
    **224,896 parameters**).
- `src/syntree/models/__init__.py`: exports grown accordingly.
- `tests/test_handle_selection.py` (new, 44 tests): pinned #22 acceptance
  (valid normalized laws for H > 1 in FP32 and FP64; H = 1 → P = 1
  exactly; identical-handle uniform split, entropy ln 2; mixed batches;
  far-handle tail stability), SE(3) invariance (rigid motions worst
  8.3e-16 @ FP64 over 6 trials; 5.96e-08 @ FP32; translation-only;
  coincident-centroid finiteness through the safe norm), permutation and
  batch consistency (handle-row permutation equivariance; pocket-atom
  permutation invariance end-to-end through the backbone; ligand-node
  permutation with consistent remap; batch-composition invariance),
  fail-closed contract matrix (13 cases), **real `AssemblyEnvironment`
  integration** (t = 0 tri-chloro seed menu and a grown 2-block
  intermediate: rows map 1:1 onto `active_handles()`, argmax row yields a
  legal `AssemblyAction.handle_atom_idx`, and the log-probability feeds
  `TrajectoryStep.action_log_prob` directly), gradient connectivity
  (into the checkpointed backbone and to positions; no cross-graph
  gradient leaks), frozen profile + audit + bit-for-bit determinism.
- `docs/architecture/ARCHITECTURE.md`: §3.11 added (`[VERIFIED — #22]`),
  module map + layout updated, header refreshed.
- `docs/agent/DECISIONS.md`: decision **D-V2-010** recorded.
- `README.md`: status table updated for #22 **and reconciled for the
  stale S-V2-03 rows** (#38 closure, engine layer, repository structure,
  limitations) — the S-V2-03 session had not refreshed the README (gap
  found during §25.1 discovery; corrected here per the constitution's
  reconcile-toward-verified-state rule).

### 3. Extra verification beyond the suite (local CPU)

- Hand-verified the frozen-profile parameter count (224,896) against the
  module audit before pinning it in the test.
- Debugged five initial test failures — all five were test-side bugs, none
  module-side: (a) a fixture generator-state divergence (`randperm` of
  different sizes desynchronised the descriptor draws between
  single-graph and batched constructions — fixed with a dedicated
  descriptor generator); (b) a pocket permutation that truncated the
  pocket block (12-of-21 nodes → the backbone's ≥1-pocket-atom rule
  correctly rejected it — fixed to permute the whole block); (c) a
  nonsense assertion comparing pre/post-remap atom indices; (d) a
  fail-closed test that tripped the wrong validator (descriptor dtype
  before the backbone dtype check); (e) an env-growth fixture using
  single-handle blocks that cannot grow the menu — replaced with
  4-aminophenol blocks whose surviving alcohol handle genuinely enlarges
  the active set. Each failure mode is now pinned by a corrected test.

### 4. Verification summary (all local CPU; no GPU/T4 claims)

- Environment: Python 3.12.16, torch 2.15.0+cpu, torch_geometric 2.8.1,
  rdkit 2026.9.1, scipy 1.14.1, numpy 2.1.3, pytest 9.0.2.
- `python -m pytest tests/test_handle_selection.py -q` → **44 passed,
  0 warnings**.
- `python -m pytest tests/ -q` → **301 passed** (257 baseline + 44 new).
- Measured evidence: rigid-motion invariance worst |Δ| = **8.3e-16**
  (float64, 6 random rotations + 10 Å-scale translations, over logits /
  log_probs / probs / attention / entropy); **5.96e-08** (float32
  production path); frozen profile = **224,896 parameters**.
- Mandatory pre-commit review: `git diff --check` (clean), `git diff
  --stat`, `git diff --unified=80` (reviewed), `git status --short`.

### 5. Commits / pushes (session total)

| Commit | Content | Push range |
|---|---|---|
| `433d303` | feat(model-02): agency-enabled handle-selection head P(handle\|s) (#22) — module + 44 tests + ARCHITECTURE §3.11 + D-V2-010 + README reconciliation | `589a843..433d303` |
| (this docs commit) | docs(agent): S-V2-04 session record — #22 closed at `433d303`, statuses, dependency map, handoff to #9 | — |

### 6. GitHub issues updated

- **Closed with evidence comment: #22** (acceptance criterion verified —
  see STATUS.md row and the issue comment for exact commands/results).
- Updated: #23's status note (last open dep is #9).
- Verified untouched and OPEN: #45 (permanent).
- STATUS.md synchronized (6/40 closed: #5, #6, #7, #21, #38, #22).

### 7. Next actions

See `HANDOFF.md` §1: next dependency-ready critical-path task is **#9
MATH-03** (continuous cosine cutoff envelopes — the last open dependency
of #23 MODEL-03, and #9 owns the envelope for ALL spatial RBF projections
per D-V2-007, including the head's centroid-distance RBF). Parallel-safe:
#8 MATH-02, #10 MATH-04, #11 CHEM-01, #14 CHEM-04, #16 DATA-01, #32
TEST-01. #36 remains gated on user authorization (Colab SSH).
