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

## S-V2-05 — 2026-11-01 — #9 MATH-03 closed: continuous cosine cutoff envelope integrated inside GaussianRadialBasis (D-V2-011)

- **Agent/session:** autonomous lead-engineering session (Z.ai), UTC.
- **Repository / branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `52849e9` (S-V2-04 docs head; branch verified equal
  to `origin/3d-synV2` before any change; working tree clean).

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full; fetched the #9 body from GitHub; re-read the
  #5–#44 issue states (clean-room closures confirmed: #5, #6, #7, #21,
  #38, #22). Read STATUS / DECISIONS / DEPENDENCIES / HANDOFF /
  WORK_LOG / ARCHITECTURE and the code head (`schema.py`,
  `equivariant.py`, `bipartite.py`, `handle_head.py`).
- The S-V2-04 handoff was unambiguous: next task **#9 MATH-03**
  (continuous cosine cutoff envelopes) — depends on #5 only, is the last
  open dependency of #23 MODEL-03, and per D-V2-007 owns the cosine
  envelope for ALL spatial RBF projections (the backbone's edge-distance
  RBF and the handle head's [0, 12] Å centroid-distance RBF).
- Baseline before any change: `python -m pytest tests/ -q` → **301/301
  passed** (matches the S-V2-04 record; environment Python 3.12.16,
  torch 2.15.0+cpu, torch_geometric 2.8.1, rdkit 2026.09.1, scipy 1.14.1,
  numpy 2.1.3, pytest 9.0.2).
- Pre-implementation numerical probe of the issue's pinned envelope
  formula on this platform: `f_cut(r_cut)` is exactly `0.0` in float32
  and float64 for cutoffs 5.0/12.0/3.7; beyond-cutoff values are bitwise
  `0.0` with bitwise-zero gradients; the derivative decays linearly
  (`f'(r_cut-δ) ≈ -0.5 (π/r_cut)² δ`: measured -1.97e-2/-1.97e-3/
  -1.97e-4/-1.97e-5 at δ = 1e-1/1e-2/1e-3/1e-4); the reflection identity
  `f(d)+f(r_cut-d)=1` holds to 2.2e-16 (machine epsilon).

### 2. Work performed (implementation)

- **New module `src/syntree/models/cutoff.py`** (the one authoritative
  envelope implementation): `continuous_cosine_envelope(dist, cutoff)` in
  the issue-#9 pinned form — `0.5 (cos(π d / r_cut) + 1)` for
  `d ≤ r_cut`, exact zeros beyond (`torch.where`), `clamp(min=0)`
  negative-roundoff semantics — with fail-closed validation
  (non-tensor / non-floating-dtype / NaN-Inf distances raise
  `TensorContractError`; non-finite-or-non-positive cutoffs raise
  `ValueError`).
- **Integration inside `GaussianRadialBasis`** (`bipartite.py`):
  `forward` now returns `Gaussian(d) ⊙ f_cut(d)` — unconditional, no
  opt-out flag, so every spatial RBF projection site (backbone
  edge-distance expansion; handle-head centroid-distance expansion over
  [0, 12] Å; the future #23 decoder, which must route through the same
  producer) is enveloped by construction. Module/class docstrings and
  the `handle_head.py` envelope references updated; the envelope is
  parameter-free so both frozen profiles are unchanged (re-pinned:
  backbone 5,999,104; handle head 224,896).
- **New test file `tests/test_cosine_cutoff.py`** — 46 tests in four
  classes: the two pinned issue acceptance criteria; the envelope
  primitive's analytic properties (closed-form value/derivative match,
  reflection identity, monotonicity, quadratic near-cutoff decay, linear
  derivative decay, bitwise determinism, dtype preservation, rank
  agnosticism, fail-closed matrix); the enveloped-RBF integration
  (peaks carry the envelope, sup-norm ≤ envelope, product-rule analytic
  radial-derivative match in fp64, no parameters added, state-dict
  roundtrip); downstream integration (handle-head basis beyond-cutoff
  bitwise zeros — the legacy "Vulnerability 7" regime dead; backbone
  smoke; compact fp64 scalar-invariance/vector-equivariance check
  through the enveloped backbone).
- **Test-contract migration (sanctioned by #45 §24.3 + D-V2-007's
  "until #9 lands" clause; D-V2-006 respected — nothing deleted, all
  rewritten 1:1):** `test_at_cutoff_last_channel_is_one` →
  `test_at_cutoff_features_vanish_and_beyond_is_bitwise_zero`,
  `test_values_bounded` → `test_values_bounded_and_nonnegative`,
  `test_peaks_at_centers` → `test_peaks_at_centers_carry_the_envelope`
  (all in `tests/test_bipartite_painn.py`);
  `test_radial_basis_is_the_raw_gaussian_expansion` →
  `test_radial_basis_is_enveloped` (`tests/test_handle_selection.py`).
- **Docs:** `docs/architecture/ARCHITECTURE.md` — new §3.12
  (`[VERIFIED — #9]`), §3.3/§3.11 raw-RBF wording updated, module map +
  repository layout updated; `docs/agent/DECISIONS.md` — **D-V2-011**
  recorded (envelope-inside-the-producer rationale, alternatives
  rejected, test-contract consequences, numerical facts, evidence);
  `README.md` — new "Continuous cosine cutoff envelope" paragraph in the
  mathematics section, planned-list pruned, module table row, repository
  structure, status table row + "last updated S-V2-05";
  `docs/agent/STATUS.md` / `DEPENDENCIES.md` / `HANDOFF.md` synchronized.

### 3. Extra verification beyond the suite (local CPU)

- Debugging notes for the record: three initial new-test failures were
  all test-side precision artifacts, none module-side — (a) the analytic
  reference re-derived `σ = r_cut/(R-1)` in full float64 while the
  module's `_scale`/`centers` buffers carry construction-dtype (float32)
  rounding before `.to(float64)` (a pre-existing #21 buffer property,
  not an envelope defect) — fixed by deriving the analytic reference
  from the module's own buffers so the pinned quantity is the envelope
  product-rule mathematics; (b) the quadratic near-cutoff bound needed
  the documented `0.5·(cos+1)` cancellation floor (≤ ~1e-15 fp64 /
  ~6e-8 fp32 near the seam — physically negligible, values there are
  ~1e-10); (c) a `BackboneOutput` attribute-name mistake
  (`scalar`/`vector`, not `node_scalar`) in the compact invariance test.
- Confirmed the four failing legacy tests after integration were exactly
  the four pre-#9 raw-RBF pins (no collateral damage: 297/301 green
  before the 1:1 rewrites).
- Confirmed the envelope cannot affect SE(3) contracts: it is a radial
  (distance-only) function; the full #21/#22 suites (equivariance ≤
  3e-13 over 36 trials, handle-head invariance 8.3e-16/5.96e-08) re-ran
  green against the enveloped path unchanged.

### 4. Verification summary (all local CPU; no GPU/T4 claims)

- Environment: Python 3.12.16, torch 2.15.0+cpu, torch_geometric 2.8.1,
  rdkit 2026.09.1, scipy 1.14.1, numpy 2.1.3, pytest 9.0.2.
- `python -m pytest tests/test_cosine_cutoff.py -q` → **46 passed**.
- `python -m pytest tests/ -q` → **347 passed** (301 baseline + 46 new;
  four legacy pins rewritten 1:1, none deleted).
- pyflakes clean on all touched files.
- Measured evidence: at `d = r_cut + 1e-4` (and 5.5, 50.0, 1e6) the
  radial feature vector is **bitwise zero** in both dtypes (issue
  acceptance criterion 1); at `d = r_cut` the features are exactly `0.0`
  on this platform (≤ 1e-12 pinned elsewhere); summed radial-gradient
  decay toward the boundary 4.3e-2 → 3.6e-3 → 3.5e-4 → 3.5e-5 for
  offsets 1e-1 → 1e-4, matching the analytic linear rate
  `0.5 (π/r_cut)² δ` (issue acceptance criterion 2); `f_cut(0) = 1`
  exactly (first channel exactly 1.0 at `d = 0`); frozen profiles
  unchanged.
- Mandatory pre-commit review: `git diff --check` (clean), `git diff
  --stat`, `git diff --unified=80` (reviewed), `git status --short`.

### 5. Commits / pushes (session total)

| Commit | Content | Push range |
|---|---|---|
| `bf735a6` | feat(math-03): continuous cosine cutoff envelope + integration inside GaussianRadialBasis (#9) — cutoff.py + 46 tests + 4 legacy-pin rewrites + ARCHITECTURE §3.12 + D-V2-011 + README | `52849e9..bf735a6` |
| (this docs commit) | docs(agent): S-V2-05 session record — #9 closed at `bf735a6`, statuses, dependency map, handoff to #23 | — |

### 6. GitHub issues updated

- **Closed with evidence comment: #9** (both acceptance criteria
  verified — see STATUS.md row and the issue comment for exact
  commands/results; D-V2-011 referenced).
- Updated: #23's status note (all dependencies closed — #23 is now
  dependency-ready and the next critical-path task).
- Verified untouched and OPEN: #45 (permanent).

### 7. Next actions

See `HANDOFF.md` §1: next dependency-ready critical-path task is **#23
MODEL-03** (two-layer stacked cross-attention decoder — consumes
`HandleSelectionOutput.handle_embeddings` / `pooled_handle_context` and
`BackboneOutput`; must route its spatial RBF through the enveloped
`GaussianRadialBasis`). Parallel-safe: #8 MATH-02, #10 MATH-04, #11
CHEM-01, #14 CHEM-04, #16 DATA-01, #32 TEST-01. #36 remains gated on
user authorization (Colab SSH).

## S-V2-06 — 2026-11-01 — #23 MODEL-03 closed: two-layer stacked cross-attention decoder (D-V2-012)

- **Agent/session:** autonomous lead-engineering session (Z.ai), UTC.
- **Repository / branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `fd61e14` (S-V2-05 docs head; branch verified equal
  to `origin/3d-synV2` before any change; working tree clean).

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full; read STATUS / DECISIONS / DEPENDENCIES /
  HANDOFF / WORK_LOG / ARCHITECTURE and the code head (`schema.py`,
  `equivariant.py`, `cutoff.py`, `bipartite.py`, `handle_head.py`).
  Fetched issue #23 + its comment thread (the owner's binding integration
  notes) and the live #5–#44 issue states from GitHub.
- The S-V2-05 handoff was unambiguous: next task **#23 MODEL-03** — all
  three hard deps (#9, #21, #22) closed with inspected deliverables;
  critical path #5 → #6 → #21 → #23 → #24 → #25 → #26 → …
- Dev environment rebuilt from scratch in the recycled sandbox: Python
  3.12.16, torch 2.15.0+cpu, torch_geometric 2.8.1, rdkit 2026.9.1,
  scipy 1.14.1, pytest 9.0.2 (`pip install torch --index-url
  https://download.pytorch.org/whl/cpu`, `pip install -e ".[test]"`,
  `pip install torch_geometric`).
- Baseline before any change: `python -m pytest tests/ -q` → **347/347
  passed** (matches the S-V2-05 record exactly).

### 2. Work performed (implementation)

- **New module `src/syntree/models/policy.py`** (the issue-pinned path):
  `CrossAttentionDecoderLayer` — `nn.MultiheadAttention(256, 8,
  dropout=0.1, batch_first=True)`, FFN `Linear(256→512) → SiLU → Dropout
  (0.1) → Linear(512→256)`, **pre-LayerNorm residuals on both
  sub-layers**; `CrossAttentionDecoder` — 2 stacked layers + closing
  LayerNorm, query projections from the #22 pooled handle surface and the
  #21 pooled contexts, order-preserving per-graph padding of the flat
  union memory into the dense `[B, N_max, d]` layout with a bool
  `key_padding_mask`; two forward interfaces (dense `[B, 1, 256] ×
  [B, N, 256] → [B, 256]` per the acceptance criteria; V2-native
  `forward_state(backbone, handle)`); `DecoderOutput` (context / sequence
  / per-layer attention maps / echoes / token counts); `audit()`
  (incl. `projects_spatial_distances: False`); fail-closed validation
  (FP32/FP64-only dtype policy matching module parameters per
  D-V2-005/008, device, shapes, `Lq = 1` pinning, bool-only mask,
  fully-padded-graph rejection, cross-field backbone↔handle graph-count
  consistency, NaN/Inf fail loudly); frozen `DECODER_*` constants
  (256/2/8/512/0.1); **1,219,328-parameter frozen profile**.
- **Design decision D-V2-012 recorded**: per-graph handle query
  (`W_handle(pooled_handle_context) + W_graph(graph_context) +
  W_pocket(pocket_context)` — one authoritative embedding chain, the
  per-row `handle_embeddings` stay with #24); **no new spatial-RBF site**
  (D-V2-011's conditional clause resolved vacuously + structurally: the
  decoder consumes no Cartesian coordinates at all; audit flag + no
  radial-basis submodule pinned by test); dense padded memory with exact
  masking semantics; closing LayerNorm (standard pre-LN practice) with
  the documented row-sum-constant caveat for downstream losses.
- **New test file `tests/test_cross_attention_decoder.py`** — 58 tests in
  six classes: the two pinned acceptance criteria (frozen-profile shape
  probe; padded columns exactly 0.0 in every layer/head/graph; outputs
  bit-identical under padded-slot junk 999/−777/1e6; attention rows sum
  to 1 over real tokens; train-mode dropout keeps pads exactly zero);
  issue-pinned layer structure (MHA/FFN config; pre-LN residual identity
  under zeroed sub-layers, which also distinguishes pre-LN from post-LN;
  general-`Lq` layer primitive; constructor validation); padding
  (roundtrip preserves rows and order; mask exactness; memory length;
  fully-padded and zero-node-graph rejection); dense-forward contract and
  fail-closed matrix (dtype policy, `Lq ≠ 1`, non-bool mask, shape/
  batch/width mismatches, NaN/Inf, empty memory; eval determinism and
  state-dict roundtrip bit-exact; seeded train-mode reproducibility;
  memory-token permutation invariance; **no cross-graph gradient leaks**
  — `context[0]`-loss leaves `memory.grad[1:]` exactly zero);
  V2-native `forward_state` (end-to-end through the real backbone +
  handle head; FP32 production path; plumbing equivalence bit-exact;
  gradients reach all three query surfaces; zero-handle graphs give
  finite context; single-graph batch; **SE(3) invariance 8.882e-16 FP64
  / 4.768e-07 FP32**; translation invariance; pocket/ligand permutation
  invariance with handle remap; batch-composition invariance; gradient
  flow through the checkpointed backbone and the handle head;
  cross-field/graph-count/width/dtype/poison fail-closed); frozen
  profile (constants; 1,219,328 parameters; audit; no radial-basis
  submodule; attention-footprint arithmetic 614,400 floats ≈ 0.59 MB at
  B=12/N≤800 — below the issue's 20 MB budget, a calculation not a T4
  measurement; full-production-profile integration smoke).
- **Exports:** `syntree/models/__init__.py` extended with the
  `DECODER_*` constants and the three new classes.
- **Docs:** `docs/architecture/ARCHITECTURE.md` — new §3.13
  (`[VERIFIED — #23]`), module map + repository layout updated, and the
  stale header (missing S-V2-05) corrected to S-V2-06;
  `docs/agent/DECISIONS.md` — **D-V2-012** recorded; `README.md` — new
  architecture-invariant bullet (one authoritative representation
  chain), module table row, repository structure, status table row,
  limitations, "last updated S-V2-06"; `docs/agent/STATUS.md` /
  `DEPENDENCIES.md` / `HANDOFF.md` synchronized.

### 3. Extra verification beyond the suite (local CPU)

- Test-side debugging (four initial failures, all test-side bugs, none
  module-side): (a) the junk-invariance test wrote junk into an
  *unmasked* slot of graph 2 — fixed by masking that slot; (b) the
  all-padded-graph probe omitted `dtype=torch.float64` on its fixtures,
  so the dtype-policy error fired before the mask check; (c) the
  wrong-width probe narrowed `backbone.scalar` *and* the contexts
  together, so the scalar error fired first — narrowed only the
  contexts; (d) the gradient tests used `context.sum()` / `context[0]
  .sum()` — degenerate losses because the closing LayerNorm makes each
  output row sum to the constant `sum(β)` (gradients exactly zero
  everywhere) — replaced with fixed linspace-weighted losses and the
  property documented in the test file header (this is a real interface
  fact for #24/#25, recorded in D-V2-012 and HANDOFF §2).
- Hand-verified the frozen-profile parameter arithmetic before pinning
  1,219,328 (MHA 263,168 + 2 LayerNorms 1,024 + FFN 262,912 = 527,104
  per layer; ×2 + query projections 164,608 + out_norm 512).
- Measured evidence script (kept at the session sandbox
  `scripts/evidence_model03.py`, not part of the repo): SE(3) worst
  |Δcontext| = **8.882e-16** (float64) / **4.768e-07** (float32
  production path) through the real backbone + handle head + decoder.

### 4. Verification summary (all local CPU; no GPU/T4 claims)

- Environment: Python 3.12.16, torch 2.15.0+cpu, torch_geometric 2.8.1,
  rdkit 2026.9.1, scipy 1.14.1, pytest 9.0.2.
- `python -m pytest tests/test_cross_attention_decoder.py -q` → **58
  passed, 0 warnings**.
- `python -m pytest tests/ -q` → **405 passed** (347 baseline + 58 new;
  nothing rewritten or deleted).
- pyflakes clean on all touched files.
- Mandatory pre-commit review: `git diff --check` (clean), `git diff
  --stat`, `git diff --unified=80` (reviewed; new files authored and
  reviewed in full), `git status --short`.

### 5. Commits / pushes (session total)

| Commit | Content | Push range |
|---|---|---|
| `dc4921e` | feat(model-03): two-layer stacked cross-attention decoder (#23) — module + 58 tests + ARCHITECTURE §3.13 + D-V2-012 + README | `fd61e14..dc4921e` |
| (this docs commit) | docs(agent): S-V2-06 session record — #23 closed at `dc4921e`, statuses, dependency map, handoff to #8 | — |

### 6. GitHub issues updated

- **Closed with evidence comment: #23** (both acceptance criteria
  verified — exact commands/results/measured values in
  https://github.com/Vtheonly/3d-syntree/issues/23#issuecomment-6559134923;
  D-V2-012 referenced).
- Updated: #24's status note (deps remaining: #11, #8; the decoder
  surfaces are available to its heads).
- Verified untouched and OPEN: #45 (permanent).
- STATUS.md synchronized (8/40 closed: #5, #6, #7, #9, #21, #22, #23,
  #38).

### 7. Next actions

See `HANDOFF.md` §1: next dependency-ready critical-path task is **#8
MATH-02** (analytic Amos-ratio autograd for the von Mises loss; deps #5
only) — together with #11 CHEM-01 it is the last blocker of #24
MODEL-04. Parallel-safe: #10 MATH-04, #11 CHEM-01, #14 CHEM-04, #16
DATA-01, #32 TEST-01. #36 remains gated on user authorization (Colab
SSH).

---

## S-V2-07 — 2026-11-02 — #8 MATH-02, #10 MATH-04, #14 CHEM-04 closed: von Mises math, FP32 gate, soft-core clash + volume floors (D-V2-013/014/015)

- **Agent/session:** autonomous lead-engineering session (Z.ai), UTC.
- **Repository / branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `8bd3e94` (S-V2-06 docs head; branch verified equal
  to `origin/3d-synV2`; working tree clean).

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full (1370 lines); read STATUS / DECISIONS /
  DEPENDENCIES / HANDOFF / WORK_LOG and the code head; fetched the live
  #5–#44 issue states from the GitHub API — closed = {5, 6, 7, 9, 21, 22,
  23, 38}, exactly matching the recorded status (no reconciliation gap).
- The S-V2-06 handoff was unambiguous: next task **#8 MATH-02** (deps: #5;
  the smallest remaining blocker of #24); parallel-safe alternatives #10
  / #11 / #14 / #16 / #32.  Session goal: solve at least three issues →
  **#8, #10, #14** chosen (all dependency-ready, all CPU-verifiable;
  #11 CHEM-01 deliberately left as the next session's focused unit — it
  is a large registry and deserves a full session).
- Spec archaeology per D-V2-002 (pre-reset files read as *specification*
  only, never code): legacy `syntree/models/torsion_head.py`
  (`log_bessel_i0` two-branch form, threshold 10, von Mises log_prob),
  legacy `syntree/utils/hardware.py` (the AMP machinery the issue orders
  removed: `resolve_amp_dtype`, auto-TF32, `cudnn.benchmark=True`), legacy
  `syntree/chemistry/conformer.py` (Bondi radii table, uniform
  `LJ_EPSILON = 0.10`, C¹-linear LJ continuation below 0.8σ — superseded
  by #14's pinned δ-form), and the legacy boundedness test
  (~650ε at full overlap).
- Dev environment rebuilt in the recycled sandbox: `pip install torch
  --index-url .../whl/cpu`, `pip install -e ".[test]"`,
  `pip install torch_geometric`, `pip install scipy`.  Baseline before
  any change: `python -m pytest tests/ -q` → **405/405 passed** (matches
  the S-V2-06 record).
- **Operational note:** the sandbox shell recycles between tool calls, so
  `GH_TOKEN` had to be re-exported per command; the initial clone had
  silently embedded an *empty* credential (the repo is public, so the
  anonymous clone succeeded and the first push failed with "Invalid
  username or token").  Fixed by `git remote set-url` with the token;
  pushes then succeeded.  Recorded here so a future session recognises
  the symptom immediately.

### 2. #8 MATH-02 — stable von Mises backward (commit `ae2c615`)

- **New module `src/syntree/models/torsion_head.py`** (issue-pinned path):
  `LogBesselI0(torch.autograd.Function)` — float64 two-branch forward
  (22-term power series below κ=12 with early exit; the pinned 3-term
  asymptotic expansion `κ − 0.5(log 2π + log κ) + log1p(u(1+u(4.5+37.5u)))`
  at/above), exact `torch.where` branch gating (no infinity ever
  materialised in an inactive branch), dtype-preserving output; backward =
  the **analytic Amos' ratio** `A(κ)=κ/(0.5+√(κ²+2.25))` (proven strictly
  in [0,1) ∀κ≥0; denominator ≥ 2; A(0)=0 exactly).  Public surface:
  `log_bessel_i0`, `amos_ratio`, `von_mises_log_prob`, `von_mises_nll`,
  `von_mises_loss`, constants.  The legacy "Singularity 2"
  (unrolled-series gradient divergence + `torch.where` NaN leakage) is
  structurally impossible: branch selection lives inside the untracked
  forward; the backward is one closed-form expression.
- **Decision D-V2-013 recorded:** the issue snippet's three defensive
  clamps are provably dead on κ≥0 — replaced by exact gating + fail-closed
  domain validation (finite/floating/κ≥0 else raise); identical values on
  the valid domain.
- **Measured evidence:** forward vs `scipy.special.i0e` over 20001 points
  on [0,100]: max error **6.51e-6** (< the pinned 1e-5, both in log-space
  and the literal scaled-space form); Amos backward vs the trusted
  reference ratio `i1e/i0e`: max **3.30e-2** at κ≈2.5 (7.9e-4/2.0e-4/
  7.4e-5 at 30/60/100 — O(κ⁻²) decay) — measured, documented, pinned.
- **Tests `tests/test_von_mises_torsion.py` — 44 tests / 4 classes:**
  forward accuracy (both acceptance forms, fp32-output floor documented),
  seam continuity, zero/small/large-κ behaviour, dtype/determinism,
  fail-closed matrix; backward boundedness (finite, ≥0, strictly <1 incl.
  κ=0→0 and κ∈{1e3,1e5}), monotonicity, trusted-reference deviation
  pinned < 4e-2, seam-gradient NaN-safety, batching/decomposition,
  `amos_ratio` helper consistency; the von Mises objective vs
  `scipy.stats.vonmises` (<1e-4), numeric normalisation ≈ 1 over
  [−π,π), peak-at-mean, κ=0 uniform exactly, gradient finiteness, an
  FD cross-check quantifying the pinned forward-exact/backward-approx
  design; AST-level ban on clamp/nan_to_num/clip in the module.
- Suite after #8: **449/449**.  Issue #8 closed with the evidence comment
  (GitHub issue comment 6561596269).

### 3. #10 MATH-04 — fail-closed FP32 precision engine (commit `5ce4f5e`)

- **New package `src/syntree/utils/` with `hardware.py`:**
  `configure_runtime_environment(device, *, mixed_precision="fp32",
  tf32=False, tf32_justified=False, device_info=None)` → frozen auditable
  `RuntimeConfig` (`precision` always "fp32"; `autocast_enabled`/
  `grad_scaler` always False; `warnings` tuple carries the verbatim
  FAIL-CLOSED rejections).  `DeviceInfo` + `probe_device()` serve both the
  real runtime and **simulated** hardware (the acceptance-test path —
  never mutates global `torch.backends`).  Gate semantics: cc<8 non-FP32
  request → FAIL-CLOSED warning citing the T4 GradScaler collapse
  (2^16→1.84e-40→0.0, weights→NaN); cc≥8 → pure FP32 still enforced *and*
  an out-of-scope-device warning per #45 §29.2; CPU → same rejection.
  TF32: off by default, needs `tf32_justified` AND cc≥8 (structurally
  impossible on the supported T4).  Issue-pinned logit-mask constants
  `LOGIT_MASK_FILL=-1e4` / `LOGIT_MASK_LEGAL_ABOVE=-1e3` with fail-closed
  `apply_logit_mask`; the pinned scatter target-unmasking invariant as a
  pure `unmask_target_actions` + strict `validate_target_actions`
  (#45 §9.6).  Legacy `scale_model_config`/`autotune_batch_size`/
  `resolve_amp_dtype` absent by construction (D-V2-005).
- **Decision D-V2-014 recorded** (see DECISIONS.md for the full
  alternatives/reasoning).
- **Tests `tests/test_fp32_enforcement.py` (issue-pinned name) — 35
  tests / 5 classes:** both acceptance criteria (simulated Tesla T4 cc
  7.5 forces fp32 + no GradScaler; fp16-on-cc7.5 logs the FAIL-CLOSED
  warning via caplog and executes pure FP32); the gate across
  V100/K80/A100/H100/CPU; every AMP token rejected; TF32 policy; global
  backends untouched by simulated infos; pinned constants; mask
  enforcement (masked logits never survive softmax; poisoned legal logits
  fail loudly); the pinned scatter verified bit-identical against the
  literal in-place snippet; strict target validation; **AST ban on
  GradScaler/autocast/torch.amp and fp16/bf16/half across `src/syntree`**
  (the "completely remove GradScaler" clause as a regression gate);
  legacy-helper absence.
- Suite after #10: **484/484**.  Issue #10 closed
  (comment 6561681322).

### 4. #14 CHEM-04 — soft-core LJ + volume floors (commit `d61a7fc`)

- **New package `src/syntree/chemistry/` with `conformer.py`:** the
  issue-pinned flat-bottom soft-core LJ `V=0` for `r≥σ`,
  `4εσ⁶/(r⁶+δ)` for `r<σ`, `δ=0.5 Å⁶`; atom typing = Bondi (1964)
  radii (re-adopted from the pre-reset spec, D-V2-002) + the D-V2-009
  oracle depths, Lorentz–Berthelot mixing **capped by the 500 kcal/mol
  bounded-spike ceiling** `ε_ij=min(√(ε_iε_j), 500δ/(4σ_ij⁶))` — the
  acceptance bound holds for every supported element pair by construction;
  analytic gradient (+ torch autograd mirror = the "differentiable
  potential"); `SoftCoreClashPotential` pair-matrix engine
  (fallback/raise unknown-element policies); deterministic grid
  `pocket_cavity_volume`; `volume_floor_penalty` (ratio 0.35, bounded
  [−2,0]) + `heavy_atom_floor_penalty` (floor 12, weight 2.0).
- **Engine wiring (decision D-V2-015):**
  `finalize_terminal_molecule`'s #42 sub-bin torsion objective now
  evaluates the production `soft_core_clash_energy` (the #14 addendum's
  demand); `soft_core_pocket_energy` keeps its **docking-surrogate** role
  for `S_dock`/`S_contact` (repulsive-only cannot express affinity; that
  injection point is #30/eval's); `ThreeDReward.S_vol` now carries both
  floors with `V_poc` derived once at construction (stationarity
  preserved — still no batch parameter; explicit
  `volume_floor_angstrom3` overrides; no pocket → graph-only reward);
  full telemetry (`volume_ratio`, `pocket_cavity_volume_angstrom3`,
  `heavy_atom_count`, `heavy_atom_penalty`, `ratio_penalty`).
- **Measured evidence:** C–C σ=3.40 Å, LB ε=0.086 → capped ε=0.0405;
  **V(0.5)=484.85 kcal/mol < 500** (acceptance 1) vs **1.58e9** for the
  rigid wall; max over all 10×10 supported element pairs = 484.85 (the
  ceiling construction makes the bound universal); V(0)=500.00 exactly;
  shell-pocket cavity 2325.4 Å³ (0.5 Å grid, deterministic); the
  engine-assembled two-step target (10 heavy atoms after both chlorides
  leave) incurs `heavy_atom_penalty = −0.3333` (acceptance 2 on a real
  synthesized molecule); `S_vol = −2.046`, `R = 0.0245 ≥ 1e-4`.
- **Tests `tests/test_soft_core_chemistry.py` — 50 tests / 6 classes**
  (reusing the verified engine-suite fixtures `_shell_pocket` /
  `_fresh_two_step_env` / `_rollout` by import): pinned form, universal
  bound, monotone repulsion, documented contact jump; gradient vs FD and
  vs torch autograd; analytic gradient maximum at r*=(5δ/7)^{1/6}; atom
  tables + mixing + ceiling; unknown-element policies; pair-matrix engine
  + fail-closed; cavity estimator (deterministic/monotone/resolution);
  both floor penalties' shapes; reward integration (stationarity across
  instances bit-equal, telemetry completeness, explicit-floor override,
  disabled-volume configuration); refinement-objective provenance (the
  oracle's `objective_before` equals the production potential at the
  reproduced pose and differs from the 12-6 surrogate).  The
  Brent-vs-dense-scan engine test's dense reference updated 1:1 to the
  production potential (contract unchanged, #45 §24.3).
- Suite after #14: **534/534** including all 50 engine contracts
  (toy-DAG flow accounting P(x)=R(x)/Z < 1e-9 residuals,
  batch-composition invariance, bit-for-bit determinism).  Issue #14
  closed (comment 6561847197).

### 5. Commits, pushes, issue updates

- `ae2c615` feat(math-02) — pushed; issue #8 closed with evidence.
- `5ce4f5e` feat(math-04) — pushed; issue #10 closed with evidence.
- `d61a7fc` feat(chem-04) — pushed; issue #14 closed with evidence.
- (this commit) docs(agent): S-V2-07 session record.
- All pushes verified against `origin/3d-synV2`; every commit was
  preceded by `git diff --check` / `git status --short` /
  `git diff --stat` review per #45 §3.

### 6. Verification summary

- `python -m pytest tests/ -q` → **534/534 passed** (CPU sandbox: Python
  3.12.16, torch 2.15.0+cpu, PyG 2.8.1, rdkit 2026.9.1, scipy 1.14.1,
  pytest 9.0.2).  No skipped, no xfail, no warnings beyond the ignored
  deprecation filters.
- Everything above is **tested locally** on CPU only — no GPU/T4
  execution, no training, no HF sync (unchanged gates: #36/#43).

### 7. Blockers / open risks (delta vs S-V2-06)

- #11 CHEM-01 is now the sole blocker of #24 MODEL-04 (critical path) and
  of the #12/#13/#15/#40/#41 chain — the recommended next session's task.
- The engine's generic junction-reaction registry and the docking
  surrogate remain documented interim implementations (D-V2-009/D-V2-015)
  pending #11 and #30 respectively.
- #42's T4 benchmarking acceptance item remains unmeasured (CPU sandbox).
- No new risks introduced: the three modules are pure additions plus two
  surgical engine integrations, all fail-closed, all pinned by tests.

### 8. Exact next actions

1. Next session: **#11 CHEM-01** (read the issue in full; module path
   `syntree/chemistry/`; 32 certified reactions + 9 coarse families +
   representative reagent/product tests per #45 §9.1; supersedes the
   generic junction registry without changing the MDP/TB contracts).
   Parallel-safe: #16 DATA-01, #32 TEST-01.
2. After #11: #24 MODEL-04 becomes dependency-ready (consumes
   `DecoderOutput.context` + `handle_embeddings` + `von_mises_loss` +
   `apply_logit_mask`/`unmask_target_actions`).
3. Do not redo #8/#10/#14 — their contracts are pinned by 129 tests.

---

## S-V2-08 — 2026-11-02 — #11 CHEM-01 + #13 CHEM-03 + #15 CHEM-05

- **Agent/session:** autonomous lead-engineering session (Z.ai), UTC.
- **Repository / branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `1db55e7` (S-V2-07 docs close).
- **Issues worked:** #11 (CHEM-01), #13 (CHEM-03), #15 (CHEM-05) — all
  three implemented, verified, pushed, and closed this session.

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full; confirmed the branch/HEAD matched the S-V2-07
  handoff (`git status` clean, HEAD `1db55e7` = origin/3d-synV2).
- Fetched issues #5–#45 with comments; verified open set against
  STATUS.md: closures at session start were #5, #6, #7, #8, #9, #10,
  #14, #21, #22, #23, #38 — full agreement, no reconciliation needed.
- Read DECISIONS D-V2-009 (engine scope boundaries) and D-V2-015 (role
  separation) as instructed by the handoff; inspected
  `syntree/engine/environment.py` (JunctionReaction /
  DEFAULT_REACTION_REGISTRY interim) and the 50 pinned
  `tests/test_gflownet_reconciliation.py` contracts.
- Baseline suite: **534/534 green** (matches the handoff).

### 2. #11 CHEM-01 — commit `0e296b0` (closed)

- Implemented `syntree/chemistry/reactions.py`: 32 certified reactions
  (8 C-N, 7 C-C, 6 C-O/C-S, 6 heterocycle installations, 5
  deprotections), 9 coarse families with the issue-pinned membership,
  48 standardized reagent/product certification pairs (16 stereo pairs),
  junction patterns + priorities for #15/#40/#41, deterministic
  deprotections (Boc/Alloc/Cbz/O-benzyl/TBDMS).
- **Conflict resolutions recorded as D-V2-016:** rxn_14/rxn_15 →
  cross_coupling_cc (the family list omitted them); heterocycles ship
  pre-cyclized on the synthon side (the pinned single-junction MDP can
  never close rings — one-step cyclization provenance documented per
  synthon class); stereo retention by atom-map chirality transfer,
  *verified* by CIP tests (hard `@` templates would reject valid
  enantiomeric/achiral reagents).
- **Empirical pre-verification loop** (scripts/verify_rxn_design.py —
  session-local, not committed): caught the pyrazine-vs-pyrimidine
  SMILES trap (`c1nccnc1` is pyrazine; the pyrimidine core is
  `c1ncccn1`), the boronic-acid `[OX2]` degree subtlety, the Stille
  template orientation, and the rxn_06/rxn_08 product-template typo —
  before any code was committed. All 32 reactions then certified 100%
  round-trip (48 pairs; symmetric-match duplicates deduplicated).
- Flipped the engine default: `DEFAULT_REACTION_REGISTRY` = the
  certified taxonomy (27 policy reactions); the generic interim was
  removed. Full suite after the flip: **616/616** — all 50 pinned
  contracts stay green bit-for-bit.
- Tests: `tests/test_32_reactions.py` 82/82 (16 skips by design).
  Issue #11 updated with evidence and closed (state_reason=completed).

### 3. #13 CHEM-03 — commit `472f4df` (closed)

- Extended `syntree/chemistry/conformer.py` in place:
  `apply_junction_torsion` (LJ-chosen planar snap for locked junctions
  via the production #14 flat-bottom clash; phi routes to the first
  adjacent non-conjugated single synthon-side bond through deterministic
  BFS; rigid rotations only) and `relax_junction_constrained` (MMFF94 in
  the static pocket vdW field — pocket heavy atoms embedded as
  hydrogen-completed typed mini-fragments CH4/NH3/OH2/SH2,
  position-constrained at 0.05 Å / k=5000; scaffold flat-bottom
  restraints pinned at 50 kcal/mol/Å² / 0.05 Å).
- **Measured findings recorded as D-V2-017:** MMFF94's amide torsions
  sit ~3.7° (bare amide) to ~11.6° (in-pipeline) off planar at their
  minimum, so planarity is *restored exactly post-minimization* by the
  same LJ-chosen rigid snap; the engine's covalent-radii placement is
  deliberately not an MMFF equilibrium, so the bond-length acceptance is
  implemented as MMFF convergence (< 0.02 Å second-pass).  Debugging
  also caught: the stale-snapshot clash() read, the bidirectional-
  adjacency BFS bug (the peripheral rotation swallowed the whole
  molecule), the multi-conformer zero-position trap (RWMol auto-extends
  existing conformers with (0,0,0) — strip conformers first), and the
  missing junction-property carry onto the returned molecule.
- `finalize_terminal_molecule` gained the opt-in flags
  `junction_torsion_stage` / `pocket_vdw_relaxation` (defaults preserve
  the legacy behaviour bit-for-bit; the 50 pinned contracts stay green).
- Tests: `tests/test_conformer_relaxation.py` 18/18 — acceptance
  criteria verified (planarity ≤ 1.0°, seed RMSD < 0.15 Å, junction
  bond < 0.02 Å of equilibrium, pocket static ≤ 0.05 Å).
  Full suite 634/634. Issue #13 updated with evidence and closed.

### 4. #15 CHEM-05 — commit `864530d` (closed)

- New package `syntree/data/` (first data-layer module);
  `fragmenter.py`: legal cuts from the #11 junction patterns (bridge
  bonds, both sides ≥ 2 heavy atoms, one reaction per bond by priority
  DESC / id ASC); canonical cut-application order with permanent
  rejection of atom-isolating cuts; the pinned seed rule (max ring
  system → highest degree → canonical SMILES ASC); periphery-to-core
  peeling (farthest-from-seed first) with the assembly order as the
  exact reverse. All labels canonical — identical trajectories under any
  atom ordering.
- Tests: `tests/test_canonical_fragmenter.py` 21/21 (12–20 random atom
  renumberings per ligand incl. the symmetric biphenyl; sequential
  rejection: amide beats the N-alkyl split of the same amine, ester
  beats the ether split of the same oxygen). Full suite **655/655**.
  Issue #15 updated with evidence and closed.

### 5. Verification and pushes

- Every issue followed the #45 §18 protocol: full issue read → deps
  inspected → implementation → tests → mandatory `git diff --check` /
  `--stat` / `--unified` review → commit → push to `3d-synV2` → GitHub
  issue comment (SHA + evidence) → close (state_reason=completed).
- Pushed commits: `0e296b0` (#11), `472f4df` (#13), `864530d` (#15) —
  all confirmed on origin (token embedded in the remote URL, verified
  99 chars per the S-V2-07 §1 note).
- Full-suite progression this session: 534/534 → 616/616 → 634/634 →
  655/655 (16 by-design skips: #11 stereo coverage).

### 6. Documentation (this commit)

- DECISIONS.md: D-V2-016, D-V2-017 appended.
- STATUS.md: #11/#13/#15 closed rows; #12/#18/#24/#28/#29/#40/#42 rows
  updated; denominator 14/40; pre-existing-failures section refreshed.
- DEPENDENCIES.md: graph rows updated; the READY set is now #24 (critical
  path), #40, #16, #32, #36 (authorization gate).
- HANDOFF.md: continuation instruction rewritten (next: #24 MODEL-04).
- README §14/§15: three new closed-capability rows + limitation wording.

### 7. Blockers, uncertainty, and next actions

- No new blockers. Unchanged gates: catalog access (#12), MMseqs2 binary
  (#17), T4 authorization (#36/#43), AiZynthFinder availability (#34/#44).
- Never verified this session: any GPU/T4 behaviour, training, HF sync
  (CPU sandbox; #45 §6 unchanged).
- **Next session: implement #24 MODEL-04** (all deps closed; the next
  critical-path node; unblocks #25 and retires #42's last open dep).
  Parallel-safe alternatives: #40 CHEM-06 (unblocks #12), #16 DATA-01,
  #32 TEST-01.

---

# Session S-V2-09 — 2026-11-02 — #24 closed, #25 implemented (CUDA-gated), #40 closed

## 1. Session parameters

- Agent session identifier: S-V2-09 (autonomous continuation from the
  S-V2-08 handoff; entry prompt = the permanent #45 pointer).
- Repository: `Vtheonly/3d-syntree`, branch `3d-synV2`.
- Starting commit: `6495485` (S-V2-08 close; clean tree verified with
  `git status --short`).
- Baseline suite: `python -m pytest tests/ -q` → **655 passed, 16
  skipped** (matches the handoff exactly; the 16 skips are the
  #11-designed stereo-pair gaps).
- Issues targeted (per the S-V2-08 handoff's continuation instruction):
  #24 MODEL-04 (primary, critical path), then #25 MODEL-05 (unblocked
  by #24), and #40 CHEM-06 (the parallel-safe alternative).
- Environment: CPU-only dev sandbox (Python 3.12.16, torch 2.15.0+cpu,
  torch_geometric 2.8.1, rdkit 2026.09.1, numpy 2.1.3, scipy 1.14.1).
  No GPU/T4 execution (per #45 §5–§6 the sandbox can never establish
  GPU claims).

## 2. Work performed

### 2.1 #24 MODEL-04 — decoupled hierarchical heads + discrete SO(2) torsion space

**Implemented** in `syntree/models/policy.py` (+~1,800 lines) and
`syntree/models/torsion_head.py` (+~530 lines):

- `TerminationHead` — binary `P(STOP|s)`; forced termination for states
  without legal continuation (mask-filled CONTINUE logit → `P(STOP)=1`
  exactly); structurally separate from every continuation head.
- `ReactionFamilyHead` (9-way) + `ReactionHead` (27-way) — the exact
  hierarchical factorization by support restriction over the certified
  registry metadata.
- `SynthonRetrievalHead` — K=50,000 tied catalog embeddings
  (`nn.Embedding(50_000, 64)`), scaled dot-product retrieval,
  block-class legality prefilter.
- `ChiralParityHead` — discrete `c_t ∈ {R,S}` scored exactly on junction
  stereocenters, conditioned on (context, synthon embedding).
- `TorsionHead` — strictly categorical 36-bin law `P(k|s,B)`; auxiliary
  (mu, kappa) via `safe_unit_vector`-routed atan2 + softplus, consumed
  only by `torsion_circular_loss` (von Mises NLL at bin centres).
- Mask factories consuming the environment `active_handles()` menu and
  the registry handle-class metadata; `HierarchicalPolicyHeads`
  container with the two-stage API + teacher-forced
  `forward_continue` (CONTINUE rows only).
- Frozen production profile: 3,272,975 parameters
  (`HIERARCHICAL_HEADS_PARAMETER_PROFILE`).

**Tests** — new `tests/test_hierarchical_heads.py`: **100 tests**.
Acceptance 1 (STOP isolation) pinned by forward-hook instrumentation,
stop-law invariance under synthon-table/torsion-parameter perturbation,
and the teacher-forced flow excluding STOP rows. Acceptance 2 (discrete
measures, clean TB) pinned by law-validity assertions, exact
`step_log_prob` decomposition, kappa/mu invariance, and a composed
trajectory through `TrajectoryBalanceEngine.trajectory_residual`.
Plus: registry metadata contracts, mask-factory semantics against real
`AssemblyEnvironment` menus + brute-force references, the exact
factorization identity, forced-STOP exactness, bin-centre identity with
the engine's pinned `bin_center(k)` for all 36 bins (cross-module
contract by test), circular-loss floor/antipode/seam-circularity,
fail-closed contracts, the frozen profile, gradient connectivity
(incl. the tied catalog table), determinism, end-to-end SE(3)
invariance through the real backbone+handle+decoder stack, and a real
`environment.step` proving the head indices map onto `AssemblyAction`
fields.

**Verification:** `python -m pytest tests/test_hierarchical_heads.py -q`
→ 100 passed; full suite 755 passed, 16 skipped. Diff reviewed
(`git diff --check/stat`), committed `74133c8`, pushed to origin
`3d-synV2` (verified).

**Issue action:** implementation report with both acceptance criteria
evidence posted to #24; **closed as completed** (all criteria verified;
no GPU dependency in scope).

### 2.2 #25 MODEL-05 — SynTreePolicy assembly + parameter audit (+ the #39 component)

**Implemented** in `syntree/models/policy.py` (+~1,100 lines):

- `SynTreePolicy` — one registered module with the full frozen
  production stack: `BipartitePaiNN` (activation checkpointing ON),
  `HandleSelectionHead`, `CrossAttentionDecoder`, the #24
  `HierarchicalPolicyHeads`, and the #39 seed/anchor component;
  `forward_state` composes backbone → handle head → decoder exactly;
  stages 0/1/2 delegate to the components.
- The **#39 seed/anchor component** (per #25's addendum):
  `pocket_anchor_candidates` — deterministic rotation-equivariant
  farthest-point anchor decomposition (invariant tie-breaks; separated
  subpockets get separate anchors); `aggregate_anchor_features` —
  invariant per-anchor means of the backbone pocket scalars;
  `SeedAnchorHead` — the factorized `P(anchor|pocket)·P(seed|anchor,pocket)`
  with the seed stage scoring through the SAME tied catalog embedding
  table (identical module instance, parameters counted once);
  `SeedAnchorOutput.step_log_prob` is the seed action's TB forward
  contribution.
- `audit()` — component breakdown, tied-table sharing, mask interfaces,
  the full action contract, memory arithmetic.

**Parameter-budget reconciliation (D-V2-019):** 10,880,719 parameters
= 43.5 MB FP32 < the 50 MB acceptance bound; the issue's ~4.8M estimate
is superseded by the closed per-issue frozen profiles (recorded as a
decision, not a silent deviation).

**Tests** — new `tests/test_policy_assembly.py`: **48 tests** —
assembly structure and per-issue frozen profiles, tied-table sharing
(identity, counted once), audit totals and memory, anchor generator
determinism/rigid-motion equivariance/subpocket separation/caps/
fail-closed domain, aggregation exactness and invariance (incl. a
cross-graph wiring error case), seed/anchor law validity and step
composition, exact `forward_state` equality with the manual composition,
stage delegation equality, the full frozen-profile end-to-end forward
with zero NaN/Infs (CPU; catalog K=50,000), gradients through the
checkpointed backbone, end-to-end SE(3) invariance of the seed/anchor
laws, fail-closed inputs, and an environment integration test.

**Verification:** 48 passed; full suite 803 passed, 16 skipped.
Committed `84fd3e7`, pushed (verified).

**Issue action:** report posted to #25 with the parameter-budget
reconciliation and the precise CUDA blocker; title updated to
"[MODEL-05] ... — Implemented, Awaiting CUDA Verification"; **left
OPEN** — the CUDA execution criterion is genuinely unverified in this
CPU sandbox and #45 §18 forbids closing on unverified criteria
(#36 authorization gate; never manufactured evidence).

### 2.3 #40 CHEM-06 — reaction-aware chemoselectivity filters

**Implemented** in `syntree/chemistry/chemoselectivity.py` (new, ~900
lines): the free-group vocabulary (orthogonal protection passes by
SMARTS construction), handle-class detection for all 19 registry block
classes (heteroaryl halides by ring classification with
pyrazine/pyridazine negative controls; boronic acids subclassed
programmatically), the documented per-class exclusion matrix (R1
self-reactive pairs, R1b competing nucleophiles, R2 organometallic
quenching, R3 competing identical handles, R4 competing SNAr positions
with the asymmetric differentiated 2,4-dihalo-1,3-azine exemption, R5
competing uncatalyzed electrophiles, R6 catalyst poisons), per-compound
records with named reason codes + detected groups + applicable registry
reactions, `screen_catalog` → `CurationManifest` with per-rule exclusion
counts/percentages, and `chemoselectivity_rule_report()`.

**Tests** — new `tests/test_chemoselectivity.py`: **111 tests** covering
all five acceptance criteria: the layer-after-validation pipeline,
reason-code traceability, the protected-substrate battery (Boc, silyl,
ester, thioether, benzyl ether, acetamide — all pass), Suzuki-tolerant
substrates, every rule positive/negative, the exemption in both declared
directions (2,4-dichloropyrimidine passes with the handle at the
amine-labile C4, excluded declared at C2; 4,6-dichloro excluded),
registry coverage of all 19 block classes, manifest impact reporting,
and fail-closed contracts.

**Verification:** 111 passed; full suite 914 passed, 16 skipped.
Committed `044adca`, pushed (verified). **Issue #40 closed** (all
criteria CPU-verifiable and verified).

## 3. Commits and pushes (this session)

| Commit | Content | Pushed |
|---|---|---|
| `74133c8` | #24 MODEL-04 implementation + tests | ✓ origin `3d-synV2` |
| `84fd3e7` | #25 MODEL-05 implementation + tests | ✓ origin `3d-synV2` |
| `044adca` | #40 CHEM-06 implementation + tests | ✓ origin `3d-synV2` |
| (this commit) | agent docs: DECISIONS D-V2-018/019/020, STATUS, WORK_LOG, DEPENDENCIES, HANDOFF, README | ✓ |

## 4. GitHub issue updates

- #24: implementation report comment; closed as completed.
- #25: implementation report comment; title annotated "Implemented,
  Awaiting CUDA Verification"; left open (precise blocker: the CUDA
  forward-pass criterion requires the authorized Colab/Kaggle T4
  runtime — #36 gate).
- #40: implementation report comment; closed as completed.
- #45: untouched (permanent constitution; never closed).

## 5. Final state and next actions

- Suite: **914 passed, 16 skipped** (was 655 at session start).
- Clean-room closures: 16 of 40 (+1 implemented-awaiting-CUDA).
- Next dependency-ready tasks (see the refreshed HANDOFF for the full
  analysis): #16 DATA-01 (deps: #5 only), #32 TEST-01 (deps: #5 only),
  #36 OPS-01 (prepare-only; user authorization gate), #12 CHEM-02
  (catalog access blocker), #26 TRAIN-01 (needs #19/#20 — data
  pipeline), #39's environment side (blocked on #12).
- Blockers unchanged: no GPU/T4 execution, no licensed catalog access,
  no HF artifact sync (all gated on authorization/access).

## 6. Honest verification ledger

- Attempted: all three issues.
- Implemented: all three (complete requirement scope, not placeholders).
- Tested locally: all three (259 new tests, all green; full suite
  914/914).
- Tested on T4: **none** — impossible in this sandbox (CPU-only); no
  GPU claim is made anywhere.
- Scientifically validated: chemistry-level (#40's rule matrix against
  the documented selectivity literature; #24's registry-mask
  cardinalities against the certified taxonomy). Training/validation
  on real data: not yet (data pipeline issues #16–#19 pending).

---

# Session S-V2-10 — 2026-11-02 — #16 DATA-01 closed; #12 catalog artifacts built full-scale (build code lost pre-commit — incident); continuation session pushed the surviving evidence

## 1. Session parameters

- Agent session identifier: S-V2-10 (autonomous continuation from the
  S-V2-09 handoff; entry prompt = the permanent #45 pointer), plus an
  S-V2-10c continuation (same date) executing the owner's "just push it"
  directive after the primary session hit its context limit.
- Repository: `Vtheonly/3d-syntree`, branch `3d-synV2`.
- Starting commit: `01f3bc0` (S-V2-09 docs close; clean tree).
- Baseline suite: 914 passed, 16 skipped (S-V2-09 close state).
- Issues targeted (per the S-V2-09 handoff's continuation instruction):
  #16 DATA-01 (primary, critical path), #12 CHEM-02 (the parallel-safe
  alternative — the handoff noted the licensed-mirror blocker; the
  mirror proved reachable).
- Environment: CPU-only dev sandbox (Python 3.12.16, torch 2.15.0+cpu,
  torch_geometric 2.8.1, rdkit 2026.09.1, numpy 2.1.3). **The sandbox
  recycled twice mid-session** (~02:44 UTC after the #16 close, and
  ~17:56 UTC during the #12 close-out) — this shaped the session's
  outcome (see §2.3).

## 2. Work performed

### 2.1 #16 DATA-01 — multi-source preprocessing pipeline (CLOSED)

**Implemented and executed full-scale**, committed `fccd670`, pushed
(verified), issue closed with the full evidence comment (2026-11-02):

- `src/syntree/data/preprocessing.py` — the canonical standardization
  library: 10 Å pocket sphere policy (documented
  water/additive/metal/modified-residue handling), ligand filters with
  exact rejection codes (MW 150–800, ≥10 heavy atoms, mandatory
  sanitization), PDB-ligand reconstruction via depositor CONECT
  connectivity + annotation-SMILES bond-order templates
  (`rdDetermineBonds` evaluated and REJECTED on RDKit 2026.09.1: wrong
  bond orders — benzene → `c1c#cc#cc#cc#1` — and spurious
  molecular-charge errors), full-chain sequence extraction (the #17
  clustering input), atomic deterministic gzip artifacts, append-only
  resumable manifest.
- `scripts/preprocess_multidataset.py` — the issue-pinned driver:
  streaming CrossDocked tar ingestion (no extraction), PDBbind zip with
  mol2 fallback (676 non-kekulizable type-4 SDFs recovered),
  BindingMOAD final CSV with TRUE per-row PDB ids (the old parquet's
  `name` column is a protein-family representative — a 24-PDB/367-row
  sample resolved only 13%; superseded, provenance updated), RCSB
  entry-API resolution/method re-verification (PDBbind X-ray ≤ 2.5 Å
  gate), bounded-parallel workers, resume-safe time budgets.
- `tests/test_preprocessing.py` — 28 tests.
- **Full-scale run:** 204,005 complexes across the three pinned mirrors
  (checksums verified) → **170,954 accepted / 33,051 rejected with
  exact reason codes, zero unhandled exceptions**; 341,913 atomic
  artifacts; `processed_manifest.jsonl` with 204,005 complete records
  (explicit documented nulls, never fabricated).
- Suite after #16: **942 passed / 16 skipped**.

### 2.2 #12 CHEM-02 — Enamine REAL 50k catalog: artifacts built full-scale and verified

The licensed-mirror blocker resolved (Enamine Rush-Delivery BB US
mirror reachable; SDF `..._290648cmpd_20260916.sdf`, sha256
`e91a26c9…` verified against the provenance record). The build was
**executed end-to-end at full scale**:

- **Curation funnel** (all 290,648 source records): 290,647 sanitizable
  → 290,356 unique → 191,639 basic-pass (MW ≤ 250, rotatable ≤ 3,
  heavy ∈ [4,18]) → registry handle detection over all 19 certified
  block classes → 152,489 certified pool → **39,150 chemoselectivity
  exclusions** via the #40 `screen_block`/`screen_catalog` machinery
  (competing_nucleophile 17,235; quenching_protic_group 7,397;
  competing_identical_handles 4,310; competing_snar_position 1,022;
  self_reactive_pair 771; seed_self_reactive_pair 757;
  competing_electrophile 150; catalyst_poison 44).
- **Stratified selection** (D-V2-023): 50,000 target — 35,000-saturated
  stratum + aromatic refill in canonical-SMILES order, 5,000-row
  reserve buffer; certified saturated supply is the measured binding
  constraint (every certified saturated block admitted; no strata
  fallback needed).
- **`pharm3d` encoding:** 2,137-dim raw descriptor (2048 Morgan + 47
  physicochemical + 6 3D-shape + 5 electrostatic + 30 handle direction
  vectors + 1 norm) → per-feature z-normalization → orthonormalized QR
  Johnson–Lindenstrauss projection (seed 2026) to 256 → L2-normalized;
  ETKDGv3 conformers with full MMFF94 relaxation where parameterizable
  (per-row `conformer_methods`: `etkdg3` 17,493 / `etkdg3+mmff94`
  32,507); fail-closed encoding with named reasons (16 failures: 12
  ETKDG-no-conformer, 4 MMFF-untypeable B/K outside the documented
  coupling-handle fallback).
- **Artifact:** `enamine_real_50k_pharm3d_d256.h5` — 67,797,608 bytes,
  sha256 `8b8c13a0581ec91db6ace643e13d9d46c8b29211e1a264b2bdd62bb90bf2ab7e`,
  schema `pharm3d-v2.0`/`catalog-v2.0`, pinned registry digest
  `32ea6db9…`, immutable. **All four #12 acceptance criteria verified**
  (50,000 unique entries; `[50000, 256]` all-finite; mean Fsp3 0.715094
  ≥ 0.45; mean rotatable 1.5361 ≤ 1.8; 69.968% saturated) —
  independently re-computed from the HDF5 during S-V2-10c, not just
  read from the manifest.

### 2.3 INCIDENT — sandbox recycle lost the #12 build code before commit

The session developed the #12 build code
(`syntree/chemistry/synthon_encoder.py` — handle detection + the
`pharm3d` descriptor + QR projection; `scripts/build_synthon_catalog.py`
— the resumable screen/select/encode driver; and its tests) in the
workspace repo clone and completed the full-scale build at 17:44 UTC.
The sandbox recycled at ~17:56 UTC **before the mandatory commit+push
step**; the untracked working clone was lost with it. What survived:
every build artifact (screen outcomes, selection, encode metadata, the
HDF5, the manifest), all raw sources, and the already-pushed #40 screen
machinery (`screen_block` / `screen_catalog` in
`syntree/chemistry/chemoselectivity.py`).

**S-V2-10c response (this continuation, under the owner's "just push
it" directive):**

1. Re-verified the artifact end-to-end: checksum match; independent
   HDF5 re-computation of every acceptance number (§2.2); strata,
   class-vocabulary, conformer-provenance and pinned-version audit.
2. Uploaded the catalog + manifest + funnel + checksum to the pinned
   artifact store — HF Hub dataset **`JJKK1212/3d-syntree-v2-catalog`**
   (private; owned by the project owner's account `JJKK1212`).
3. Committed the version-controllable evidence to the branch
   (`docs/data/catalog/`: manifest, funnel, checksum sidecar, README)
   together with this session record and the refreshed agent docs.
4. Posted the honest #12 status comment (evidence + incident +
   remaining scope). **#12 stays OPEN** — per #45 §18 the closure
   requires the code deliverables in-repo; the remaining scope is the
   re-landing of `synthon_encoder.py` + the build driver + tests,
   pinned against the surviving artifacts (schema, funnel semantics,
   selection order, and acceptance all pinned in the manifest).

The #16 full-scale **output** artifacts were also hit by the recycles:
47,483 of 170,954 accepted-complex artifact pairs survive plus the
chemcomp cache; `processed_manifest.jsonl` (the #17 input) is lost.
Regeneration is mechanical — the driver is in-repo (`fccd670`) and all
three raw sources survive with checksums pinned in the provenance
record (CrossDocked tar 1.6 GB, PDBbind zip 625 MB,
`bindingmoad_final.csv`).

## 3. Commits and pushes (this session)

| Commit | Content | Pushed |
|---|---|---|
| `fccd670` | #16 DATA-01 implementation + tests (primary session) | ✓ origin `3d-synV2` |
| (this commit) | S-V2-10 docs: #12 catalog evidence (`docs/data/catalog/`), DECISIONS D-V2-021/022/023 (artifact-pinned reconstruction), STATUS, DEPENDENCIES, HANDOFF, README | ✓ origin `3d-synV2` |

## 4. GitHub issue updates

- #16: full evidence comment; **closed as completed** (both acceptance
  criteria verified at full scale).
- #12: evidence + incident comment; **left OPEN** with the precise
  remaining scope (re-land the build code against the surviving
  artifacts).
- #45: untouched (permanent constitution; never closed).

## 5. Final state and next actions

- Suite: **942 passed, 16 skipped** (was 914 at session start).
- Clean-room closures: 17 of 40 (+1 implemented-awaiting-CUDA #25,
  +1 artifacts-built-awaiting-code #12).
- Next actions (see the refreshed HANDOFF): (1) re-land the #12 build
  code and close #12; (2) re-run `scripts/preprocess_multidataset.py`
  to regenerate `processed_manifest.jsonl` (raws + checksums survive);
  (3) #17 DATA-02 (mmseqs 2-binary already staged at
  `3d-syntree-work/tools/mmseqs` in the dev workspace); (4) #32 TEST-01;
  #36 OPS-01 remains user-authorization-gated (and would also close
  #25's CUDA criterion).

## 6. Honest verification ledger

- Attempted: #16, #12.
- Implemented: #16 fully (code + tests + full-scale execution, closed).
  #12: the deliverable **artifacts** were built at full scale and every
  acceptance criterion re-verified; the build **code** was lost to the
  sandbox recycle before commit and is NOT claimed as implemented —
  #12 remains open with the re-landing scope.
- Tested locally: #16 (28 new tests; suite 942/942 + 16 by-design
  skips). #12's tests were part of the lost working tree.
- Tested on T4: **none** — impossible in this sandbox (CPU-only); no
  GPU claim is made anywhere.
- Scientifically validated: data-level (#16's full-scale funnel with
  exact rejection codes over 204,005 complexes; #12's measured curation
  funnel over 290,648 source records with the #40 chemoselectivity
  screen applied before admission, per the audit addendum). Training/
  validation: not yet (#17–#20 pending).


## S-V2-11 — 2026-11-02 — Continuous push directive

Owner directive: commit each bounded, coherent unit immediately after its applicable checks pass and push to `origin/3d-synV2`; do not wait until session end. Push recovery checkpoints before risky work and delegation, and after each independently recoverable batch. Verify the remote ref after each push and record the SHA/evidence in existing session artifacts. Preserve concurrent history. Keep credentials out of tracked content. See D-V2-024 and the updated HANDOFF.

Documentation-only session: no implementation issues #5–#44 were changed or closed. Issue #45 remains open.

---

# Session S-V2-11 — 2026-11-02/03 — #12 re-landed and closed (artifact-exact), #16 outputs regenerated, #17 closed, #32 closed

- Agent session identifier: S-V2-11 (autonomous continuation from the
  #45 entry point; read issue #45 in full, HANDOFF/STATUS/DEPENDENCIES/
  DECISIONS + branch state first, per §25.1).
- Starting commit: `160c9dc` (S-V2-10 close). Ending commits:
  `02b09d2` (#17 code) → `83a629e` (#17 evidence) → `0e1bf30` (#32
  Suite A) → `b76422a` (#12 code) → `b8439bd` (#12 artifact-exact
  reconstruction) + this docs commit — all pushed and verified on
  `origin/3d-synV2` (remote-ref inspection after every push, D-V2-024).

## 1. What was done

1. **Raw sources re-acquired** (the S-V2-10 workspace was lost to the
   sandbox recycle): crossdocked tar 1,616,333,272 B md5
   `59416d06...` ✓, PDBbind zip 625,493,784 B md5 `56c61303...` ✓,
   `bindingmoad_final.csv` 523,117 B ✓, Enamine SDF zip sha256
   `4942df33...` ✓ (all against `RAW_SOURCE_PROVENANCE.yaml`), the
   immutable catalog HDF5 sha256 `8b8c13a0...` ✓ from the private HF
   dataset, plus a static mmseqs binary.
2. **#16 outputs regenerated full-scale** (~45 min, resumable driver):
   204,005 manifest records — exact total match; per-source accepted
   crossdocked 154,760 (exact), pdbbind 4,862 (exact), bindingmoad
   11,333 vs 11,332 (+1 complex, attributed to upstream RCSB/chemcomp
   API drift; recorded in `docs/data/splits/README.md`).
3. **#17 DATA-02** (`02b09d2` + evidence `83a629e`): fasta extraction
   (169,281 unique chain sequences, first-appearance order), MMseqs2
   `easy-cluster --min-seq-id 0.30 -c 0.80 --cov-mode 0 --cluster-mode
   2` (22,387 clusters / 11,873 multi-member), Union-Find over
   complexes sharing homologous chains (14,958 components, largest
   18,458), deterministic greedy largest-deficit 80/10/10 allocation
   (components by `(-size, min complex_id)`, ties train>val>test):
   **train 136,764 / val 17,096 / test 17,095; fail-closed
   zero-leakage check: 0 of 22,387 clusters span splits.**
   `unified_splits_30seq_id.parquet` + `split_manifest.json` (with
   per-source counts). 18 unit tests. **Closed.**
4. **#32 TEST-01** (`0e1bf30`): the five issue-pinned Suite A batteries
   (293 tests) + the two canonical primitives they exercise:
   `syntree/utils/seeding.py` (seed_everything / capture / restore —
   the one authoritative seeding for #26/#43) and
   `syntree/data/shards.py` (deterministic-gzip `.pt.gz` shard I/O +
   sha256 fail-closed — the #19 format foundation). **Closed.**
5. **#12 CHEM-02 re-landing** (`b76422a` + `b8439bd`) — the session's
   core forensic reconstruction, every step verified against the
   surviving artifacts at full scale:
   - **Certification model**: synthon role = `screen_block(mol, cls,
     handle_atom_idx=<first instance>)` pass (boronic classes without a
     declared atom); seed role = **block-level** screen (any #40
     `SELF_REACTIVE_PAIRS` firing between the molecule's *free* groups
     and the 6 uncatalyzed electrophiles kills the whole seed role) —
     **0/50,000 per-row mismatches** against the catalog labels.
   - **Funnel: 24/24 counts exact** over all 290,648 records (incl.
     seed_self_reactive_pair 757 fired blocks / 760 pair-firings; basic
     filter stage order MW→rotatable→heavy).
   - **Selection exact**: the 55k buffer (35k saturated + 20k aromatic,
     canonical-SMILES ascending, saturated-first) reproduces the
     artifact manifest's `buffered_rows` index list exactly.
   - **Projection bit-exact**: `default_rng(2026)` → QR →
     sign(diag(R)) reproduces `projection/Q` with max |Δ| = 0.0.
   - **Conformer decision tree**: MMFF typing checked on the H-added
     molecule pre-embedding; fallback {B,Zn,Sn} ∪ {F,Cl,Br,I} (coupling
     handles + halide counterions) reproduces the stored 17,493/32,507
     etkdg3/mmff94 split with **0/50,000 row mismatches**; potassium
     trifluoroborates fail closed on `['B','K']` (deterministic).
   - **Full rebuild**: 50,000 rows, [50000,256] float32 finite
     L2-normalized, mean Fsp3 0.7151, mean rotatable 1.5361 (both
     acceptance bounds pass), strata 34,986/15,014, methods 17,492/
     32,508, **49,998/50,000 (99.996%) block-id overlap with the
     immutable artifact**; rebuilt HDF5 sha256 `00a556ef...` recorded as
     a NEW artifact version (the original stays authoritative).
   - Documented residuals (quantified in the module docstring +
     evidence README): 3 of 47 FG slots (tertiary-amine, post-sulfone,
     provably-all-zero), the 30-D vector class list (identity
     normalization — statistically unrecoverable), the original ETKDG
     seed (energies underdetermine it; re-landed protocol deterministic
     at seed 2026). 50 unit tests. **Closed.**

## 2. Incidents / deviations

- Background downloads launched with `nohup ... &` did NOT survive the
  tool-call boundary (killed; partials vanished) — all large downloads
  were re-run in foreground bounded calls (fast: ~11 MB/s). Lesson
  recorded: no long-lived processes across tool calls in this sandbox.
- First funnel run had 3 count deltas (filter order + the per-class
  seed screen); second run 3 deltas (the block-level discovery); third
  run **24/24 exact**. Every iteration was full-scale (215 s/run).
- One encode overshoot (50,058 rows — the 50k cap was not enforced
  inside the pooled batch flush) — fixed, re-assembled from the encode
  cache without re-encoding.
- Owner pushed D-V2-024 (continuous commit-and-push) mid-session; the
  local branch was rebased onto it and every subsequent unit was
  committed and push-verified immediately.

## 3. Commands and verification (the exact evidence)

- Raw checksum verifications: `md5sum`/`sha256sum` against
  `RAW_SOURCE_PROVENANCE.yaml` (all OK).
- `python scripts/preprocess_multidataset.py --source ...` (full run,
  resumable): 204,005 records; per-source counts above.
- `MMSEQS_BIN=... bash scripts/run_mmseqs_split.sh ...`: 30 s;
  split_manifest.json + zero-leakage check in-run.
- `pytest tests/suite_a/ -q` → **293 passed**; full suite →
  **1303 passed / 16 skipped**.
- `python scripts/build_synthon_catalog.py --sdf ... --stage all`:
  funnel + selection + encode (chunked bounded calls; caches resumed).
- Funnel comparison: 24/24 counts exact vs `docs/data/catalog/
  screen_funnel.json`; certification-label check: 0/50,000 mismatches;
  `buffered_rows` list comparison: exact; `projection/Q`: max |Δ| = 0.0;
  conformer-method prediction: 17,493/32,507 exact, 0/50,000
  mismatches; rebuilt-artifact audit: [50000,256] finite, norms 1.0,
  overlap 49,998/50,000.

## 4. GitHub issue updates

- #12: full evidence comment; **closed as completed** (acceptance
  criteria verified with evidence + honest residual documentation).
- #17: full evidence comment; **closed as completed**.
- #32: full evidence comment; **closed as completed**.
- #45: untouched (permanent constitution).

## 5. Final state and next actions

- Suite: **1303 passed / 16 skipped**. Clean-room closures: **20 of
  40** (+1 implemented-awaiting-CUDA #25).
- Next dependency-ready: **#41 DATA-06** (deps #11+#12 closed — hashed
  inverted index over the catalog), then **#18 DATA-03** (deps #11,
  #12, #15, #17 closed + #41) — the data chain's next critical path;
  #39's environment-side rewiring unblocks with #12; #36 OPS-01
  prepare-only remains user-authorization-gated (would also close
  #25's CUDA criterion).
- Workspace artifacts (not in git, per policy): the regenerated
  `processed_manifest.jsonl` + pocket/ligand artifacts
  (`work/data/processed/`), `unified_splits_30seq_id.parquet`
  (`work/data/splits/`), the rebuilt catalog HDF5 + manifest
  (`work/catalog/`), raw sources (`work/raw/`), mmseqs binary
  (`work/tools/`). #37 (HF sync) uploads them when its deps close.

## 6. Honest verification ledger

- Attempted: #12 (re-landing), #16 (output regeneration), #17, #32.
- Implemented: all four (code + tests landed and push-verified).
- Tested locally: #12 (50 tests + full-scale funnel/selection/label/
  projection/conformer verification vs the artifacts), #17 (18 tests +
  full-scale run + in-run fail-closed leakage check), #32 (293 Suite A
  tests; full suite 1303/16).
- Tested on T4: **none** — CPU-only sandbox; no GPU claim is made.
- Scientifically validated: data/chemistry-level (#12's 24/24 funnel
  counts + 0/50,000 label checks + bit-exact projection + exact
  conformer-method split; #17's zero-leakage clustered splits at 30%
  identity over 169,281 sequences). Training/validation: not yet
  (#18–#20, #26+ pending).

---

# Session S-V2-12 — 2026-11-03 — #41 closed, #39 closed, #18 implemented + bounded-run verified

- Agent session identifier: S-V2-12 (autonomous continuation from the
  #45 entry point; read issue #45 in full, HANDOFF/STATUS/DEPENDENCIES/
  DECISIONS + branch state first, per §25.1).
- Starting commit: `4dddaa1` (S-V2-11 close). Ending commits:
  `dea8809` (#41 index) → `35b54ba` (#18 engine) → `95c12f2` (#18
  driver) → `d1a9167` (#39 environment side) + this docs commit — all
  pushed and remote-verified on `origin/3d-synV2` (D-V2-024).

## 1. What was done

1. **#41 DATA-06 CLOSED** (`dea8809`): the versioned hashed inverted
   index — `(family, handle_class, formula_hash)` postings + the seed
   namespace, salt→parent canonicalization, Hill-formula sha256/16,
   exact canonical-parent stage, brute-force reference; byte-
   deterministic atomic JSON; fail-closed schema/hash-version/catalog-
   sha/live-registry guards. Full-scale: build 8.26 s, 9,987 keys /
   64,451 postings, **exhaustive self-retrieval 68,417 checks / 0
   lost**, **brute-force equivalence 2,000 sampled queries / 0
   mismatches**, lookup 1.34 µs, **888.7× mean candidate reduction**,
   artifact sha `ea57e867…`. 28 tests.
2. **Registry forensics (D-V2-026/027)**: the re-landed
   `registry_digest()` cannot reproduce the immutable catalog's
   historical attribute (lost original payload format; registry source
   provably unchanged since `0e296b0`) — the index binds the live
   digest + catalog sha256 instead. #18's development then exposed the
   **rxn_14/rxn_15 junction-pattern orientation defect** (`:1` anchored
   on the block side) plus the deeper RDKit trap (match tuples are
   template-position-ordered, silently cooperating with the defect):
   corrected both patterns, exported
   `junction_pattern_attachment_atoms()` (map-number locator), and
   rewired the fragmenter + extractor to it. Registry digest unchanged
   (patterns are not in the digest payload).
3. **#18 DATA-03 implemented + verified** (`35b54ba` + `95c12f2`):
   the trajectory-extraction engine (auto-derived, cross-pair-verified
   leaving-group graft specs 27/27 — Cl/Br/B(OH)2/ZnCl/SnMe3/Li/
   aldehydic-O; index-backed discovery; **exact forward replay as the
   final acceptance**; closed failure vocabulary; crystal dihedral →
   36-bin torsion labels; documented rxn_13 protonated-form handling;
   seed_centroid for #39) + the resumable bounded driver. 29 tests.
   **Bounded real-data run** (CrossDocked md5-verified → 5,000 → 3,894
   accepted complexes → 688 MMseqs2 clusters / 0 leakage):
   **239.8 complexes/s**, 0/3,894 accepted — the honest scientific
   finding that certified-Enamine-BB reconstructibility is the binding
   constraint (95.7% `no_matched_seed`, 4.0% `no_matched_synthon`, 0.3%
   `replay_mismatch` — all 11 = rxn_15 secondary-alkyl junctions
   correctly rejected; D-V2-029 note). Evidence:
   `docs/data/trajectories/README.md`. **Stays open** for the
   full-scale run.
4. **#39 MODEL-06 environment side CLOSED** (`d1a9167`):
   `reset()` → the pre-seed candidate state (SeedMenu: anchors from the
   ONE authoritative generator × the seed pool); the explicit
   `SeedAction(B_0, h_0)` via `place_seed()` (step/stop/observe fail
   closed before it — no hidden heuristic on the production path); the
   deterministic covariant pose rule (D-V2-028; exact rigid-motion
   equivariance, pinned); placed-seed-leaves-growth-menu; the TB 'seed'
   step (forward log-prob in, backward term zero by the unique-parent
   proof — Property 3b added to MDP_STATE_SPACE.md);
   `supervised_anchor_label` (nearest-anchor to the crystal seed
   centroid, documented tie-breaks, supervision never pins inference).
   24 acceptance tests; 26 existing call sites rewired.
5. **Issue updates**: #41 + #39 closed with full evidence comments; #18
   updated with the implementation/bounded-run evidence and exact
   remaining scope. #45 untouched.

## 2. Incidents / deviations

- GitHub release-asset downloads 404 from this sandbox (mmseqs binary)
  — resolved via the official `mmseqs.com/latest` host.
- `GetSubstructMatches` uniquify=True silently dedupes symmetry-related
  matches AND the map-number/orientation trap (D-V2-027) — both now
  documented at the call sites that must not repeat them.
- An RWMol built from a cached template inherits its conformer: the
  placed-pose conformer must replace (not append to) it — caught by the
  centroid-at-anchor test, fixed with `RemoveAllConformers`.

## 3. Commands and verification (the exact evidence)

- `pytest tests/ -q` with `SYNTREE_CATALOG_H5=…h5` → **1384 passed /
  16 skipped** (baseline 1303/16 + 28 #41 + 29 #18 + 24 #39).
- `python scripts/build_synthon_index.py --catalog …h5 --out …
  --manifest … --sample-queries 2000` → the benchmark numbers above
  (manifest at `work/catalog/synthon_index_manifest.json`).
- `preprocess_multidataset.py --source crossdocked --limit 5000` →
  3,894/1,106; `run_mmseqs_split.sh` → 688 clusters, 0 leakage;
  `build_trajectories.py --max-seconds 420 --finalize` → 239.8/s,
  the failure histogram, `trajectories_{train,val,test}.pt` (0/0/0
  records — honest).

## 4. GitHub issue updates

- #41: full evidence comment; **closed as completed**.
- #39: full evidence comment; **closed as completed**.
- #18: implementation + bounded-run evidence comment; **open** (the
  full-scale run remains).
- #45: untouched (permanent constitution).

## 5. Final state and next actions

- Suite: **1384 passed / 16 skipped**. Clean-room closures: **22 of
  40** (+ #25 awaiting its CUDA criterion on #36; #42 awaiting T4
  benchmarking; #18 awaiting the full-scale run).
- Next dependency-ready: **#18 full-scale data chain** (re-acquire
  pdbbind + bindingmoad raws; full preprocessing; full splits; the
  resumable extraction driver to completion) — see HANDOFF; then #19
  (DATA-04, blocked only on #18) unblocks the whole TRAIN chain (#26 →
  #27 → #28/#29…).
- Workspace artifacts (not in git): the catalog HDF5 + index
  (`work/catalog/`, checksums verified), crossdocked tar
  (`work/raw/`), mmseqs binary (`work/tools/`), the bounded-run
  processed data + splits + trajectory outputs
  (`work/data/…` — re-creatable via `docs/data/trajectories/README.md`).

## 6. Honest verification ledger

- Attempted: #41, #18 (implementation + bounded real-data run), #39.
- Implemented: all three (code + tests landed and push-verified).
- Tested locally: #41 (28 tests + full-scale benchmark), #18 (29 tests
  + driver smoke + the 3,894-complex bounded run), #39 (24 tests +
  26 rewired call sites).
- Tested on T4: **none** — CPU-only sandbox; no GPU claim is made.
- Scientifically validated: chemistry/data-level (#41's exhaustive
  retrieval equivalence at full catalog scale; #18's replay acceptance
  invariant + the honest 0-acceptance census of the bounded subset;
  #39's anchor/pose semantics). Training/validation: not yet.

---

## Session S-V2-14 — 2026-11-03 — #20 closed, #26 implemented + CPU-verified, #36 prepared (auth gate)

### 0. State discovery (per #45 §25.1)

Read #45 in full; branch `3d-synV2` HEAD `b9b043e` (S-V2-13 had landed
#18's full-scale run `12b3a9f` and #19 `b9b043e` and closed both issues
on GitHub — the agent docs were stale at S-V2-12 state; reconciled this
session). Baseline suite: **1408 passed / 19 skipped** (catalog-artifact
skips only; the `work/` artifacts were lost to the sandbox recycle —
expected, re-creatable via the committed drivers).

### 1. Work performed

**#20 DATA-05 — implemented, verified, CLOSED** (`bc41319`):
`syntree/data/augmentation.py` + 34 tests. Issue-pinned σ=0.15 Å
ligand-only Gaussian jitter (pocket = fixed conditioning context,
bit-identical); rigid ±5° non-junction rotatable-dihedral perturbation
(Gauss-Seidel exact pristine+delta targeting — coupled rotations
handled; fail-closed bound verification; amide/ester/imidoyl
resonance-locked bonds excluded; junction bonds — the supervised torsion
actions — never deliberately rotated); train-split only (fail-closed on
val/test); single-graph state hook `augment_single_state`.
**Measured finding (D-V2-035):** naive jitter flips stereocentres (2/50
seeds on the E/Z fixture) → stereo-signature rejection resampling in
the composed augmenter (the finding is pinned as a regression test).

**#26 TRAIN-01 — implemented + CPU-verified, T4-gated** (`e009127`):
`syntree/data/supervised.py` (crystal-intermediate supervised states,
D-V2-030..034), `syntree/engine/trainer.py` (Profile A: B=12×6, pocket
cap 700, strict FP32 through the #10 gate, AdamW 3e-4/1e-5, clip 1.0;
fail-closed divergence — no nan-shield/batch-skipping; separate
host-RAM/GPU-VRAM telemetry; atomic SHA-256 checkpoints with exact
continuation state; time-budget graceful stop), `configs/
train_colab_t4.json`, `scripts/train_stage1.py` (the canonical CLI —
catalog-driven masks from the immutable #12 H5, fail-closed without it)
+ 40 tests incl. the end-to-end CLI smoke. RDKit hazard pinned:
reaction-template mols segfault `GetSubstructMatches` (D-V2-031).

**#36 OPS-01 — prepared, auth-gated** (`39e9a41`):
`notebooks/run_3d_syntree.ipynb` (fail-closed T4 preflight, aria2/
libxrender1/openssh-server, sshd start, exact-commit checkout, separate
RAM/VRAM preflight, GPU smoke, canonical training launch) +
`docs/ops/colab_ssh_workflow.md` (the operator's `colab ssh` CLI
patterns + the §29.1 authentication handoff). README status table
reconciled. **No connection established; none claimed** — the operator
was given the exact commands; the session criteria remain open.

### 2. Incidents / deviations

- The seed-legality mask polarity bug (eligibility vs True=illegal) was
  caught by the CLI smoke (a 9999 mask-fill loss) and fixed + pinned.
- S-V2-13 left the agent docs stale (closed #18/#19 without the session
  record); reconciled here — #18's evidence lives in
  `docs/data/trajectories/README.md`.

### 3. Commands and verification (the exact evidence)

- `python -m pytest tests/ -q` → **1482 passed / 19 skipped**
  (baseline 1408 + 34 #20 + 17 supervised + 23 trainer incl. the CLI
  smoke).
- `python scripts/train_stage1.py … --max-epochs 2` (the fixture
  end-to-end smoke) → RC 0; `telemetry.jsonl` + `trainer_state.pt` +
  `.sha256` written; final total loss 3.21 (seed 0.89, step 2.33,
  stop 0.0 forced-path).
- Loss-descent CPU smoke (6-epoch variant in-test): epoch-0 mean >
  last-epoch mean on the tiny fixture (the T4 criterion itself is
  gated, not claimed).

### 4. GitHub issue updates

- #20: full evidence comment; **closed as completed**.
- #26: full evidence comment (implemented + CPU-verified; T4 criteria
  explicitly pending); **open**.
- #36: preparation comment with the operator authentication
  instructions; **open**.
- #45: untouched (permanent constitution).

### 5. Final state and next actions

- Suite: **1482 passed / 19 skipped**. Clean-room closures: **23 of
  40** (+ #25/#26 implemented — T4-gated; #42 partially; #36 prepared —
  auth-gated).
- Next: **the authorized Colab T4 session (#36)** closes #25's CUDA
  criterion, #26's two T4 criteria and #36's session criteria in one
  run (then #27 → #33/#43…). CPU-side: #27 TRAIN-02 (needs #26 —
  implementable now against the CPU-verified trainer).

### 6. Honest verification ledger

- Attempted: #20, #26, #36 (preparation).
- Implemented: all three (code + tests landed and push-verified at
  `bc41319` / `e009127` / `39e9a41`).
- Tested locally: #20 (34 tests), #26 (40 tests + CLI smoke), #36 (no
  code tests — notebook/workflow docs; the runtime cells are for the
  authorized session).
- Tested on T4: **none** — CPU-only sandbox; no GPU claim is made.
- Scientifically verified: #20's augmentation invariants (chirality
  preservation guarantee, dihedral bounds, rigidity) and the measured
  flip finding; #26's supervised-state/label contracts (fail-closed
  record/artifact cross-checks) and checkpoint continuation state.

## Session S-V2-15 — 2026-11-04 — #27 closed, #33 implemented + CPU-verified, #20 re-audited, #36 auth handoff re-issued

### 0. State discovery (per #45 §25.1)

Read #45 in full (the whole constitution, all 29 sections); fresh clone
of `3d-synV2` at `600c1ae` (S-V2-14 close). Read every open issue's
body (#25–#37, #42–#44) and #36's comment trail; read
STATUS/HANDOFF/DEPENDENCIES/DECISIONS + the S-V2-14 work-log entry.
Confirmed the S-V2-14 claims against the tree: `syntree/data/
augmentation.py` + `tests/test_augmentation.py` present (34/34 green),
the #36 notebook (11 cells) + workflow doc present, the trainer + CLI
present. **Independent re-audit of #20 (the user's explicit request):
the closure is genuine** — fresh run `pytest tests/test_augmentation.py
-q` → 34 passed; baseline suite reproduced exactly
(1482 passed / 19 skipped) before any edit.

### 1. Work performed

**#27 TRAIN-02 — implemented, verified, CLOSED** (`9197619`):
`syntree/engine/curriculum.py` — the five issue-named difficulty axes
per supervised item (pocket atoms, rotatable bonds of the crystal
intermediate, rings, pocket charge magnitude, handle rarity;
z-scored, equal-weight, D-V2-037); 9 quantile-balanced level bins
(fail-closed below 9 items); the 6-stage level distribution
`p = (1−9ε)·kernel + ε`, ε = 0.03 exactly (the floor-exact mixture,
D-V2-038); `CurriculumSampler` drawing exactly N samples per epoch
from deterministic (seed, stage, epoch) NumPy streams (bit-identical
re-iteration — the exact-resume property); the milestone machine
(running-best reference, improvement ≥ 0.005 advances, patience-2
fallback, no regression, stage cap 6, non-finite val loss fails
closed); schema-versioned `state_dict`/`load_state_dict` with
config-mismatch fail-closed. Trainer integration in
`syntree/engine/trainer.py`: wiring guards (sampler-wired loader, val
split, curriculum ⇒ val loader), per-epoch `_validation_loss`
(eval-mode, no-grad, fail-closed on non-finite), curriculum telemetry
records (level histogram + probabilities + stage), checkpoint payload
key `"curriculum"` with symmetric fail-closed restore. Canonical CLI
`--curriculum` (additive; the #26 pinned launch unchanged — D-V2-039).
`tests/test_curriculum.py`: 40 tests — ingestion (real amide fixture +
split guard), balanced bins, floor exactness at every stage incl. the
extreme centres, exact-N, all-9-levels-per-epoch histograms at all six
stages, empirical frequency match, determinism/re-iteration, the
milestone machine incl. the 40-epoch smooth-progression acceptance,
state round-trips (JSON-able), trainer wiring guards, checkpoint
round-trip + both mismatch refusals, and the end-to-end
`--curriculum` CLI smoke (subprocess; checkpoint carries the
curriculum state).

**#33 TEST-02 — implemented + CPU-verified, one battery T4-gated** (`7a6ea3b`):
`tests/suite_b/` (a package — the issue-specified basenames would
otherwise collide with the root batteries):
(1) `test_mathematical_singularities.py` — coincident atoms d=0 finite
with exactly-zero gradients (`safe_norm`/`safe_unit_vector`), the von
Mises backward == the bounded Amos ratio A(κ) ∈ [0,1) over κ ∈ [0,100]
(extremes/monotonicity/asymptotics), the cosine cutoff bitwise-zero at
r_cut with continuous decay + the integrated RBF vanishing past it;
(2) `test_se3_equivariance.py` — scalar invariance + vector covariance
under 36 random proper SO(3) rotations and translations through
`BipartitePaiNN` (fp64), pure-translation, pocket-only N_l=0, batched;
(3) `test_tensor_contracts.py` — 64-D handle descriptors separate from
3-D coordinates; legacy 67-D Franken tensors rejected with the exact
schema message, incl. a literally-glued `[descriptors|xyz]` tensor;
(4) `test_t4_memory_boundary.py` — Profile-A CPU contracts (B=12/6/700,
budget 9.5 GiB, parameter arithmetic 10,880,719 params = 43.5 MB) +
the B=12×cap-700 full-profile fwd/bwd peak-reserved-VRAM measurement
that executes ONLY on the authorized Tesla T4 (loud, reason-pinned
skip elsewhere; non-T4 CUDA also refused per #45 §29.2);
(5) `test_gflownet_flow.py` — the deterministic uniform-leaf reverse
step on linear/symmetric fragment fixtures (seed exclusion, connected
predecessors, exactly 1/k) + the TB loss behaviour (exact identity,
seed/STOP zero backward, log(4/3) leaf sensitivity, zero loss at
exact flow, log-Z gradients, bit determinism, fail-closed set, batch
invariance, symmetric-ordering shared optimum).
`pytest tests/suite_b/ -v` → **47 passed / 1 skipped**.

**#20 DATA-05 — independent re-audit** (no code change needed):
fresh 34/34 + the full-suite baseline reproduction (1482/19 before
this session's additions). The user's session prompt asked to
"implement #20"; the honest finding is that S-V2-14 genuinely
implemented and closed it — this session verified it independently
rather than redoing it.

**#36 OPS-01 — verified prepared, auth handoff re-issued**: the
notebook (11 cells, fail-closed T4 preflight, sshd, exact-commit
checkout, separate RAM/VRAM preflight, GPU smoke, canonical launch)
and `docs/ops/colab_ssh_workflow.md` re-inspected; nothing further is
codeable — the remaining criterion is the operator's `colab ssh`
authentication (#45 §29.1: the agent must not fabricate or preempt the
link). The exact operator commands were re-issued in the session
reply.

### 2. Incidents / deviations

- Author-name mismatch on the #27 commit (`3D-SynTree V2 Agent` vs the
  repo's `autonomous-agent`; same email) — the amend ran after the
  author config change without `--reset-author`; pushed anyway
  (never force-push, D-V2-024). Cosmetic only.
- Three of my own test-side math errors were caught by the first
  curriculum run (patience semantics, 1-row z-score degeneracy,
  guard-order match) and two fixture/tolerance errors in Suite B
  (star-fragment id, FP32-vs-1e-7 tolerances, a sign, a log-prob
  mapping) — all fixed in the same session; the *implementation*
  needed no change after the initial write except the trainer/CLI
  integration which passed unchanged.

### 3. Commands and verification (the exact evidence)

- Baseline (before edits): `python -m pytest tests/ -q` → **1482
  passed / 19 skipped**; `python -m pytest tests/test_augmentation.py
  -q` → **34 passed** (the #20 audit).
- After #27: `python -m pytest tests/test_curriculum.py -q` → **40
  passed**; full suite → **1522 passed / 19 skipped**.
- After #33: `python -m pytest tests/suite_b/ -v` → **47 passed / 1
  skipped** (skip = the T4 memory measurement, reason pinned); full
  suite → **1569 passed / 20 skipped** (0 failures, no regressions;
  46 s–85 s runtime).
- Both #27 acceptance criteria verified in-test: all 9 levels sampled
  every epoch (histograms at all 6 stages) and the 40-epoch schedule
  transitioning smoothly through all 6 stages (monotone, no skips,
  5 transitions).
- `git diff --check` clean before every commit (D-V2-024 flow);
  remote ref verified after each push.

### 4. GitHub issue updates

- #27: full evidence comment; **closed as completed**.
- #33: full evidence comment (implemented + CPU-verified; the single
  remaining criterion is the on-T4 execution of the memory battery);
  **open** — mirrors #25/#26 treatment.
- #36: no new comment this session (the S-V2-14 comment remains the
  accurate prepared-awaiting-auth record; the handoff commands were
  re-issued in the chat reply).
- #20: no comment (re-audit evidence lives here + STATUS.md).
- #45: untouched (permanent constitution).

### 5. Final state and next actions

- Suite: **1569 passed / 20 skipped**. Clean-room closures: **24 of
  40** (+ #33 implemented — one battery gated on #36; #25/#26/#42
  unchanged).
- Commits this session: `9197619` (#27) and `7a6ea3b` (#33), both
  pushed and remote-verified; decisions D-V2-037/038/039 recorded.
- Next: **the authorized Colab T4 session (#36) is the single
  highest-value step** — it closes #25's CUDA criterion, #26's two T4
  criteria, #33's memory battery (`pytest tests/suite_b/ -v` → 48/48
  → close), #36's session criteria, and unblocks #43's measured
  profiling + cross-session resume. CPU-side if still unauthenticated:
  #29 FLOW-01's remaining scope is the logZ MLP + trainer integration
  (its TB mathematics + backward semantics are closed by #38/D-V2-009
  and now pinned by Suite B battery 5); #28 is externally blocked on
  licensed reaction data (owner decision).

### 6. Honest verification ledger

- Attempted: #27, #33, #20 (re-audit), #36 (verification + handoff).
- Implemented: #27 (code + tests + integration, closed), #33 (five
  batteries, one hardware-gated measurement).
- Tested locally: #27 (40 tests + CLI smoke), #33 (47 CPU tests),
  #20 (34 re-run), full suite 1569/20.
- Tested on T4: **none** — CPU-only sandbox; no GPU claim is made
  anywhere (the Suite-B memory battery and #25/#26/#42 criteria are
  explicitly gated).
- Scientifically verified: the curriculum's statistical contracts
  (floor exactness, balanced bins, all-level coverage, deterministic
  streams, milestone semantics) and Suite B's mathematical invariants
  (singularity behaviour, SE(3), schema separation, TB identities).

## Session S-V2-17 — 2026-11-04 — demo: real WebGL 3D protein graphics + protein-as-graph figures + zipped user deliverable

### 0. State discovery

Continued from HEAD `b923960` (the demo evidence commit `c5d9e2f`
plus the train-01 fix). The user requested the demo's protein be
rendered as an actual 3D structure ("draw the actual graph in 3d of
the protein or use software to render it") and the demo zipped for
delivery. Demo tree verified complete (38 files) before any edit;
`git status --short` clean at open.

### 1. Work performed

- Fetched the **full experimental structure** from RCSB
  (`https://files.rcsb.org/download/4ITP.pdb`, sha256
  `289c9a1f4df0c3a6fd30568fbea52f958fdd378219e6387b25228367624087db`)
  and proved the crystal-frame identity chain: TRP A5 NE1
  full-vs-pocket Δ = **0.000 Å**; all **145/145** pocket ATOM records
  match the deposition atom-for-atom (max dev 0.000 Å); the
  BindingMOAD ligand SDF equals the deposited 1GD HETATM coordinates
  (21/21 atoms, max same-element dev 0.000 Å).
- New committed demo section `demo/protein_to_molecule/
  protein_structure_3d/`: a **self-contained interactive 3Dmol.js
  2.4.0 (WebGL) viewer** (`protein_viewer.html`, engine + structures
  inlined, works offline in any browser; four scenes + spin), plus
  headless-Chromium-143/Playwright renders at 2400×1800 px
  (8×6 in @ 300 DPI) PNG+PDF for: full-protein cartoon (turbo
  per-residue coloring), active-site close-up (labelled catalytic
  residues, Zn²⁺), experimental-vs-reconstructed ligand overlay
  (RMSD 0.000 Å), VDW-surface view; a 36-frame 360° turntable GIF;
  and a Matplotlib **protein-as-a-3D-graph** figure (`pocket_graph_3d`):
  (a) Cα backbone graph — 257 nodes / 256 peptide edges; (b)
  binding-site atomic graph — 145 pocket atoms + 21 ligand atoms +
  Zn²⁺, covalent edges + dashed Zn coordination (Zn–N(sulfonamide)
  1.988 Å, His94/96/119 contacts in the provenance file).
- New committed raw source artifact
  `structures/raw/rcsb_4ITP_full.pdb.gz`; three new reproducible
  demo scripts (`render_protein_3d.py`, `render_protein_3d_shots.py`,
  `pocket_graph_3d.py`, self-locating `--demo-root` default);
  `render_provenance.json` (sources + sha256s, renderer versions,
  fail-closed checks, output sha256s, purpose statement: experimental
  reconstruction, **not** AI generation).
- README §2a added; directory tree + final-report Q9 updated.
- User deliverable zipped and integrity-verified
  (`unzip -t` clean; 38-file exact match vs disk).

### 2. Incidents / deviations (honest)

- **3Dmol.js LabelSpec gotcha:** the label background key is
  `backgroundColor` (default opaque *black*); a `background` key is
  silently ignored → labels rendered as solid black boxes. Found by
  inspecting the minified bundle's `Label` class; fixed with named
  colors + `bold` + thin borders.
- **Async surface race:** 3Dmol computes surfaces asynchronously by
  default, so screenshots raced the VDW pass (surface visible in one
  run, absent in the next at identical styles). Fixed with
  `$3Dmol.setSyncSurface(true)` + a 3 s wait for the surface scene —
  the render's nonwhite-pixel ratio jumped 0.571 → 0.956, proving the
  surface now renders deterministically.
- `'spectrum'` is not a valid cartoon colorscheme in 2.4.0 (console
  warnings, gray cartoon) → per-residue **turbo** colors computed in
  Python (261 `addStyle` calls).
- Close-up framing: `zoom(1.6)` after `zoomTo(ligand)` clipped the
  Zn²⁺ sphere and pushed label anchors out of frame — fixed by
  pulling back (zoom 0.72–0.82) and offsetting the Zn label along
  the ligand→Zn axis.
- **Render speed:** 2400×1800 with MSAA under SwiftShader ≈ 35–57 s
  per scene; the whole battery in one invocation exceeded the
  300 s tool timeout once → the shooter gained `--scenes` for
  per-scene invocations (fail-closed assertions re-run per
  invocation; provenance merges across partial runs).
- Visual QA used the z-ai VLM (glm-5.3-flash) on downscaled previews
  + full-res crops; iterated until all scenes passed (labels
  legible, Zn²⁺ visible, surface present, no clipped text).

### 3. Commands and verification (the exact evidence)

- `python3 scripts/protein_render_html.py --demo-root …` (builds the
  viewer; fetches 4ITP + 3dmol.min.js once, caches, then offline)
- `python3 scripts/protein_render_shoot.py --demo-root … --scenes
  full_protein|active_site,ligand_overlay,surface|gif`
- `python3 scripts/pocket_graph_3d.py --demo-root …`
- Zip: `zip -r -9 protein_to_molecule_demo.zip
  protein_to_molecule_demo` then `unzip -t` (clean) + exact 38-file
  set match vs disk.
- Every shooter invocation re-runs the fail-closed assertions:
  frame identity, 145/145 pocket atom identity, exp-vs-reconstructed
  RMSD **0.000 Å** (same-order + nearest-pair, 0 element
  mismatches), SDF-vs-deposited max dev **0.000 Å**.

### 4. GitHub issue updates

- #45: session work-log comment with the commit SHA (constitution
  §29-style record; no issue state changed).

### 5. Final state and next actions

- HEAD on `3d-synV2`: this session's commit (see `git log -1`); demo
  now carries `protein_structure_3d/` + 3 scripts + the full-protein
  raw artifact; workspace `download/` holds the verified zip.
- Next (unaffected): the Tier B fail-closed T4 driver for #26/#36
  continuation.

### 6. Honest verification ledger

- Attempted/implemented: real 3D molecular graphics for the
  protein→molecule demo (user-requested), rendered by committed
  scripts only.
- Tested locally: all renders verified by the shooter's assertions
  + VLM visual QA; zip integrity verified byte-for-byte.
- Tested on T4: **none** — rendering is CPU/WebGL-software work; no
  GPU claim is made.
- Scientifically verified: crystal-frame identity of every rendered
  structure (0.000 Å max deviation, 145 pocket + 21 ligand atoms);
  Zn²⁺ coordination measured from the deposition (1.99 Å to the
  sulfonamide N; His94/96/119 distances in `render_provenance.json`).
- NOT claimed: no AI-generated structure appears in any render; all
  coordinates are experimental (crystal) except the ETKDG block
  conformers, which are never drawn inside the pocket (unchanged
  demo §3 rules).

## Session S-V2-18 — 2026-11-04 — #29, #30, #31 CLOSED (the FLOW track lands); S-V2-16/17 records reconciled

### 0. State discovery (per #45 §25.1)

Read #45 in full; fresh clone of `3d-synV2` at `af115dc` (the S-V2-17
demo close). Read HANDOFF/STATUS/DEPENDENCIES/DECISIONS + the S-V2-14/15
work-log entries; fetched the live GitHub issue states for all 16 open
issues + the comment trails of #25/#26/#29/#30/#31/#36/#42/#43.

**Reconciliation findings (recorded here per §25.1):**

- The agent docs were **two sessions stale**: S-V2-16 (the first
  authorized T4 battery run — `cac741e`, the cross-device
  anchor-accumulator + CUDA-generator fixes) and S-V2-17 (the
  authorized-session evidence comments + the protein-3D demo work)
  landed commits and GitHub updates but no WORK_LOG entries and no
  STATUS/HANDOFF refresh. Consequences reconciled this session:
  - **#25 and #33 are CLOSED on GitHub** with the T4 evidence (full
    policy fwd/bwd on the live Tesla T4: peak reserved 3.758 GiB /
    allocated 3,295.6 MiB, zero NaN/Infs) — the STATUS rows said
    "Implemented — Awaiting Verification". Updated.
  - The live tracker state at session open: 28 of 40 closed; open =
    #26, #28, #29, #30, #31, #34, #35, #36, #37, #42, #43, #44 (+ #45).
  - The baseline suite is **1571 passed / 20 skipped** (not the 1569
    recorded at S-V2-15 — S-V2-16's `tests/__init__.py`/fp32-probe
    changes added 2 tests).
- The user's session prompt named #20 (already genuinely closed at
  S-V2-14, re-audited at S-V2-15 — not redone) and #36 (prepared; the
  authorization gate is the operator's). The dependency graph's
  critical path after #26's CPU-verified deliverables is
  **#29 → #30 → #31 → #37** — this session executed exactly that CPU
  segment.

### 1. Work performed

**#29 FLOW-01 — implemented, verified, CLOSED** (`6eccd41`):

- `LogPartitionHead` — the #29-specified lightweight 2-layer MLP over
  the pooled pocket latent (the backbone's `pocket_context`, 256) and
  the seed latent (the tied catalog embedding, 64); 82,433 params
  (0.76% of the policy assembly); zero-initialized output (documented
  deterministic start); fail-closed input validation.
- The differentiable tensor log-probability path: `TrajectoryStep`
  accepts 0-dim tensors beside floats; `TrajectoryBalanceEngine`
  gains the `log_z` override (the seed-conditioned partition function
  enters the pinned equation without touching its mathematics); the
  float verification path is unchanged (66 existing TB/Suite-B tests
  stay green — verified before commit).
- `GFlowNetTrainer` — strict FP32 via the #10 gate, AdamW with the
  two #29-pinned groups (policy 3e-4/1e-5 continuing Stage-1 Profile
  A; the log Z head at `lr_Z = 1e-2`, no weight decay — D-V2-040),
  fail-closed `TrainingDivergenceError` (no NaN shields, no batch
  skipping), gradient clipping with finite-gradient checks, separate
  host-RAM/GPU-VRAM telemetry, atomic SHA-256 checkpoints carrying
  policy + the log Z θ state + both optimizer groups + progress + RNG.
- `syntree/engine/rollout.py` (new) — the production rollout bridge
  `PolicyRolloutDriver`: batched lockstep rollouts (seed/anchor →
  STOP law → handle law → family → reaction → synthon → parity →
  torsion, all sampled from the policy's own normalized laws with the
  driver's `torch.Generator`), tensor forward log-probabilities per
  action component (the exact #29 decomposition), the Level-2 terminal
  oracle + the stationary `ThreeDReward`, the catalog bridges
  (block-side HandleSpecs from the certified reactant SMARTS —
  string-parsed per the D-V2-031 segfault hazard; seed blocks via the
  molecule-side construction), the parity pre-step determination via a
  cloned env state with post-hoc fail-closed verification. Conventions
  documented as D-V2-041 (capped-pocket frame, seed-centroid growth
  states, molecule-side pool ∩ env capacity, undecided primary handle).
- `AssemblyEnvironment.max_blocks` public property (the driver's STOP
  gate; no private access).

**#30 FLOW-02 — implemented, verified, CLOSED** (`90b1035`):

- The docking-backend injection point: `DockingResult` (ΔG +
  attractive-only LJ + optional surrogate telemetry + a version-labelled
  backend identity), `surrogate_docking_scorer` as the frozen default
  (bit-identical to the pre-injection computation — pinned), and
  `ThreeDReward(docking_scorer=...)` construction-time injection with
  the #30 clamp/units/sign conventions applied to any backend (D-V2-042).
  Non-finite scores, wrong return types, and non-callable scorers fail
  closed; every reward's telemetry records the backend provenance. The
  real docking engine is evaluation-stack scope (#34/#35) and injects
  through this interface — #30's own scope is complete.

**#31 FLOW-03 — implemented, verified, CLOSED** (`6760b3d`):

- `configs/stage2_colab_t4.json` (the frozen Stage-2 profile) +
  `scripts/train_stage2.py` (the V2-canonical form of the issue's
  `main.py --mode rl`; the branching `--algorithm` flag is structally
  absent — Stage 2 IS GFlowNet-TB, #45 §12.3): catalog-H5 mask surfaces
  + the environment BuildingBlock pool, the pocket pool from the #18
  records + processed artifacts (the Stage-1 loaders reused), Stage-1
  weight initialization with SHA-256 sidecar verification + exact
  provenance telemetry, `--resume-auto` fail-closed semantics (resume
  Stage-2 exactly / else Stage-1 / else refuse — never random weights),
  the on-policy TB loop with deterministic (seed, update) pocket
  permutations, per-update telemetry, checkpoint cadence + time-budget
  graceful stop (#43 contract).

### 2. Incidents / deviations (honest)

- My first rollout driver draft passed `log_probs` where
  `torch.multinomial` needs probabilities (RuntimeError on the first
  driver test) — caught by the battery, fixed at all five sampling
  sites.
- The first toy-DAG training test double-stepped the environment
  (`_replay(prefix)` + re-walking the prefix) and keyed action menus by
  path-dependent handle indices; fixed by walking a fresh env and
  identifying handles through order-invariant mapped-canonical-SMILES
  tags (two assembly orders reaching the same quotient state index the
  same law — the #38 semantics as an action identity; menu invariance
  verified across all 19 states before use).
- The first determinism test constructed a fresh policy per run
  (different random weights — of course the log-probs differed); fixed
  by sharing one instance.
- The first #31 CLI corrupt-sidecar test crashed with a traceback (rc
  1) instead of the fail-closed rc 2 — the init phase wasn't inside
  the fail-closed handler; wrapped.
- The toy exact-zero test needed fp64 tensors on the test side (a
  default-fp32 tensor loses ~4e-8 through the engine's dtype cast —
  documented behaviour, test corrected).
- Two cosmetic dead-code artifacts from drafting were removed before
  commit (`if False else` branches); `git diff --check` clean.

### 3. Commands and verification (the exact evidence)

- Baseline before edits: `python -m pytest tests/ -q` → **1571 passed
  / 20 skipped**.
- After #29: `pytest tests/test_gflownet_tb.py -q` → **26 passed**;
  the 66 existing TB/Suite-B tests re-run green; full suite →
  **1597 passed / 20 skipped**.
- After #30: `pytest tests/test_reward_docking_backend.py
  tests/test_gflownet_reconciliation.py tests/suite_b/
  tests/test_gflownet_tb.py -q` → **133 passed / 1 T4-gated skip**.
- After #31: `pytest tests/test_stage2_cli.py -q` → **6 passed**;
  final full suite → **1613 passed / 20 skipped** (1571 + 42 new, zero
  regressions; 4 m 08 s).
- #29 acceptance 1 (flow balance on synthetic DAG rollouts): every TB
  residual **exactly zero (< 1e-9)** at the analytically optimal policy
  + true logZ through the tensor path; acceptance 2 (clean backward):
  gradients reach BOTH param groups (production-path evidence), the
  loss drops >100× over 300 updates, the learned logZ converges to the
  true partition function (< 0.35), and 2000 sampled rollouts match
  R(x)/Z to **TV < 0.15**.
- #31 acceptance 1: the init record's policy-parameter SHA-256 equals
  the Stage-1 checkpoint's model exactly (recomputed in-test);
  acceptance 2: `--resume-auto` continues updates 2 → 4 with the full
  history intact.
- `git diff --check` before every commit; remote refs verified after
  every push (`6eccd41`, `90b1035`, `6760b3d`).

### 4. GitHub issue updates

- #29: full evidence comment; **closed as completed**.
- #30: full evidence comment (incl. the docking-engine supersession
  boundary statement); **closed as completed**.
- #31: full evidence comment; **closed as completed**.
- #36: operator authentication handoff re-issued (the remaining
  session criteria: the remote training launch + persisted telemetry +
  cross-session continuation); **open**.
- #45: session work-log comment (constitution §29-style record; no
  issue state changed).

### 5. Final state and next actions

- Suite: **1613 passed / 20 skipped**. Clean-room closures: **31 of
  40** (#29/#30/#31 this session). The FLOW track is complete; the
  critical path is now **#34 EVAL-01 → #44 → #35 → #37** with the
  #36-gated T4 work (#26's training run, #42's benchmark, #43's
  cross-session resume) as the parallel hardware track.
- Next session (CPU): **#34 EVAL-01** (deps #29 ✓ + #12 ✓; external
  dependency: AiZynthFinder availability) — the unlinked independent
  retrosynthesis evaluation pipeline; then #44 (both AiZynthFinder
  tracks). Next session (hardware): the authorized #36 runtime for the
  canonical Stage-1 launch (#26's two criteria), the #42 benchmark, and
  the #43 continuation demonstration — the Stage-2 CLI is now ready to
  chain there too.

### 6. Honest verification ledger

- Attempted: #29, #30, #31 (all three closed), plus the doc
  reconciliation of the unrecorded S-V2-16/17 sessions.
- Implemented: all three (code + tests landed and push-verified at
  `6eccd41` / `90b1035` / `6760b3d`).
- Tested locally: #29 (26 tests: the two acceptance criteria + the
  driver battery), #30 (12 tests + the existing 50-test reward
  battery), #31 (6 CLI tests against a REAL Stage-1 fit); full suite
  1613/20.
- Tested on T4: **none this session** — CPU-only sandbox; no GPU claim
  is made (the T4-gated items are #26's training-run criteria, #42's
  benchmark, #43's demonstration — all riding #36).
- Scientifically verified: the TB flow identity on the enumerable DAG
  (exact-zero residuals; reward-proportional sampling after learning),
  the stationarity/injectability contracts of the reward, and the
  Stage-1→Stage-2 initialization/resume provenance.
- NOT claimed: any Stage-2 training quality result (the loss-descent
  evidence is the toy-DAG acceptance fixture, not a production pocket
  campaign — that requires the authorized runtime and the #28-scale
  data decision); any docking-engine integration (the injection point
  exists; the engine is #34/#35 scope).

## Session S-V2-19 — 2026-11-05 — the authorized-T4 data reconstruction bit-exact; the D-V2-043 pool fix; the runtime-recycle recovery engineering (session in progress — the T4 training run pending on free-tier capacity)

### 0. State discovery

Read #45 + HANDOFF/STATUS/DEPENDENCIES/DECISIONS at `4dea80b` (the
S-V2-18 close). The authorized-hardware track B was the plan (HANDOFF
§1.B): #26's two training-run criteria + #36's remaining session
criteria + #43's profiling/resume, all riding the authorized runtime.

**Runtime reality:** the sandbox recycle since S-V2-17 also lost the
Colab CLI auth state; the OAuth bridge (`scripts/colab_auth_phase*.py`
+ token backup) restored it — the owner's refresh token survived.
The `syntree-t4` session itself was gone (free-tier recycle #1).

### 1. The full-chain bit-exact reproduction (runtime recycle #1)

`scripts/runtime_full_chain.sh` (committed this session, sanitized)
ran end-to-end on a fresh T4 runtime: env (torch 2.11.0+cu130) → §29.2
preflight (Tesla T4 14,912.7 MiB cc 7.5; host 12,975.5/11,303.6 MiB) →
catalog sha256 `8b8c13a0…` MATCH → raws md5-verified → **204,005
preprocess records exact** → splits **train 136,764 / val 17,096 /
test 17,095, 22,387 clusters, zero leakage** → index byte-deterministic
`ea57e867…` (888.7× reduction, 1.84 µs) → extraction 170,955/170,955
→ **TRAJECTORY_BITEXACT: PASS** (all three pinned .pt SHA-256s) →
portable bundle → HF persistence. Suite-B re-run on the T4 for the
record: 47 passed / 1 by-design skip; the memory battery
`B=12 × cap-700: peak reserved 3.758 GiB / allocated 3,295.6 MiB`
(exact reproduction of S-V2-16/17).

### 2. The D-V2-043 fix — the first canonical launch failed closed (the real bug)

`train_chain.sh` T1 crashed in the loader: `pdbbind_4itp: the step-2
junction (atom 0, class 'alkylzinc') is not in the detected handle
pool`.  Diagnosis (the census over all 161 train+val contexts, 20 step
views): D-V2-031 matched the molecule-side **reactant template**
against the **product-form crystal graph** — for the nine non-empty
molecule-side grafts (Br / Cl / ZnCl / B(OH)₂) the leaving group is by
construction absent from every crystal product graph, so those handle
classes were structurally invisible (4ITP's Negishi step reacts a
product-form methyl as CH₂–ZnCl; every aryl C-H was equally blind).
Fixed by reactant-form graft matching (`apply_graft` — the extractor's
own certified reconstruction; empty-graft classes unchanged; graft
incompatibility = not-a-handle, the `graft_failed` vocabulary).
An unsound SMILES-keyed instance cache (same graph, different atom
order — the PDBbind vs MOAD parses of the same 4ITP ligand) was caught
by the shuffled-order loader materialization and removed; ~1 ms/pool
measured.  The fix corrects both consumers: the #26 supervised pool
AND the #29 rollout driver's generation pool.  Evidence: +2 supervised
regression tests; the loader materializes 241 train + 101 val items
across 3 shuffle orders (0 failures); the CPU descent smoke widened to
6 seeded epochs (2-epoch means were noise-dominated at 1 optimizer
step/epoch under the honest ~15-row action space; descent verified at
torch seeds 0..9); **suite 1615 passed / 20 skipped**.  Commit
`b7e0f3d` (+ the decision D-V2-043 in DECISIONS.md).

### 3. The recovery engineering (runtime recycle #2 + upstream drift)

The runtime was recycled again ~2 min into the first diagnosis poll.
Stage-11's HF bundle enabled a ~10-minute recovery path — but the
bundle itself carried only ligands (the packer `break`-after-first-
suffix bug): 194 pockets were regenerated locally from the
checksummed raws (the deterministic (protein, heavy-atom-centroid,
10 Å) function; 159 CrossDocked streamed + 13 PDBbind + 22 MOAD RCSB
fetches), the packer fixed fail-closed at the source (refuses
incomplete bundles), and the **complete 388-artifact bundle
re-persisted to HF** (`data/training_bundle.tar`).  Upstream drift
observed and handled: the PDBbind mirror `photonmz/pbpp-2020` was
renamed `photonmz/pdbbindpp-2020` (identical bytes, md5-verified).
The three runtime drivers are committed sanitized (`02f4443`):
`runtime_full_chain.sh` / `runtime_restore_from_hf.sh` /
`runtime_train_chain.sh`.

### 4. Pending (free-tier T4 capacity)

The T4 assign endpoint has returned Service Unavailable through the
retry window (balance 0.00 — no compute units to buy priority).  The
training chain (`launch_final.sh`: restore → D-V2-043 precheck on the
runtime → suite-B → T1–T4) is staged and ready; **#26/#36/#43 remain
open pending that run** — nothing is claimed without it.

## Session S-V2-20 — 2026-11-05 — the evaluation stack: #34/#44 closed with real-planner evidence; #35 harness verified on the frozen test manifest (the CPU critical path of the S-V2-19 handoff)

### 0. State discovery

Read #45 in full (1,368 lines) + HANDOFF/STATUS/DEPENDENCIES at
`4affac2` + the GitHub issue states re-fetched via the API (31 of 40
closed; open: #26, #28, #34, #35, #36, #37, #42, #43, #44).  The
authorized-hardware track (A in the S-V2-19 handoff) is **operator-gated
and genuinely unavailable here**: no `ssh` binary and no `colab` CLI
exist in this sandbox, and §29.1 forbids the agent from authenticating
as the user — no GPU work was attempted and none is claimed.  The CPU
track (handoff §1.B) was the session plan: **#34 → #44 → #35**.

### 1. The external dependency resolution (#34's gate)

`aizynthfinder` 4.4.1 pip-installable and **runtime-verified**: the
public USPTO policy (Zenodo 7797465/7341155: ONNX rollout + ringbreaker
+ filter models, template libraries) downloaded; the official public
ZINC stock (figshare 23086469) downloaded.  The stock's legacy
fixed-format HDF5 cannot be unpickled inside the sandbox's 4 GB RAM —
solved by a **memory-bounded exact extraction**: the pickle byte-stream
is regex-scanned in windows anchored on the opcode + the 27-char
InChIKey body — 17,422,831 keys = the HDF5 row count exactly (the
first two window-boundary implementations over/under-counted by
125/97 — caught by the row-count check and fixed with absolute-position
tracking; the final SHA-256 `c80161d0…` is recorded).  Smoke
evidence: aspirin solved with 36 routes (acetic anhydride +
salicylic acid) from the full stock at 2.7 GB peak RSS.  Environment
reconciliations recorded as **D-V2-044** (rdkit 2026.09.1 vs the
package's conservative `<2024` pin — runtime-verified; numpy 1.26.4 ×
pyarrow 16.1.0).  The evaluation corpus was pulled from the HF
training bundle (`JJKK1212/3d-syntree-v2-runs`, tar sha-verified
contents: 194 ligand + 194 pocket artifacts + the three .pt trajectory
files 111/50/33) and the catalog re-downloaded + sha256-verified
(`8b8c13a0…`, matches the pinned value).  Baseline suite: **1616
passed / 19 skipped** (catalog-gated tests re-enabled).

### 2. The implementation (#34/#44/#35, commit `6c8c2bc`)

* `syntree/engine/evaluator.py` — the #34 library: the fail-closed
  corpus parser (multi-record, gzipped, closed rejection vocabulary,
  loud failure on wrong-file inputs), the physicochemical surface,
  the **isolated aizynthcli subprocess boundary** (file-delivered
  targets, flat-JSON results, non-zero/timeout/no-output all fail
  loudly), the #44 failure vocabulary (`stock_mismatch` vs
  `no_route_found` — never conflated), the optional fail-closed
  PoseBusters/Vina adapters, and the `evaluation_report.json`
  assembly with per-track provenance and the forward-replay block.
* `scripts/run_benchmarks.py` — the canonical #34 CLI (+
  checkpoint-resume and artifact-reuse per D-V2-045).
* #44: `scripts/export_aizynth_stock.py` (Track A stock export with
  the catalog-ID mapping + the SHA-256 tie) and
  `scripts/prepare_aizynth_tracks.py` (the two track configs +
  `tracks_manifest.json` with per-file checksums and the public
  source URLs).
* #35: `syntree/engine/comparative.py` (frozen-target loader,
  deterministic seeded bootstrap CIs, paired differences over the
  common-target intersection, the identical-evaluation-code path,
  markdown summaries) + `scripts/make_frozen_targets.py` (the manifest
  generator from the #18 test-split artifacts — **33 targets**, the
  honest actual count, with the 100-target design target recorded as
  not yet reachable) + `scripts/run_comparative_benchmark.py`.
* Tests: `tests/test_evaluator.py` (35), `tests/test_eval_stock_tracks.py`
  (11 + 3 real-planner gated on `SYNTREE_AIZYNTH_MODELS`), 
  `tests/test_comparative_benchmark.py` (22).  Full suite: **1683
  passed / 22 skipped**.

### 3. The evidence runs (all real, all labelled)

* **#34/#44**: the 194-record / **30-unique-structure** corpus through
  the isolated public-USPTO planner, both tracks, with PoseBusters:
  validity 194/194 (100%), mean Fsp3 0.5339, diversity 0.8934,
  PoseBusters (`mol`) 194/194; **Track A (generator-stock-aligned
  Enamine 50k): solve 0.2667** — 22 stock mismatches, 0 planner
  failures; **Track B (independent external ZINC 17,422,831): solve
  0.7333** — 8 stock mismatches, 0 planner failures.  The honest
  reading: every unsolved target is a classified stock limitation; the
  BB-filtered generator stock cannot cover the purchasable
  intermediates the USPTO routes need, while the 17.4M-compound
  external stock covers most.  The two rates answer different
  questions and are never merged.
* **#35**: the frozen 33-target manifest (test split) through the
  comparative harness: validity 1.000 [1.000, 1.000], Fsp3 0.536
  [0.437, 0.638], diversity 0.875, retro[stock_aligned] 0.250 [0.000,
  0.500], retro[independent_external] 0.667 [0.417, 0.917]; the
  baselines (TargetDiff/DiffSBDD/SyntheMol) are recorded as
  not_provided — their structures await operator export.
* Evidence copies: `docs/eval/evidence/` (evaluation_report.json 220 KB,
  comparative_benchmark.json + .md, frozen_targets.json, the stock
  manifests) + the provenance doc `docs/eval/retrosynthesis_tracks.md`.

### 4. Operational events (recorded for future sessions)

The sandbox recycles wiped the in-repo `.venv` **twice mid-session**
(a cleaner targeting repo-local venvs) — resolved by the persistent
`/home/z/my-project/evalenv` (outside the repo; survived every
subsequent boundary; the reinstall is ~4 min).  A stray planner-output
leak into the repo root from a relative-path test was caught by the
pre-commit diff review and fixed at the source (outputs now land next
to the report).  The corpus initially globbed both `.sdf` and
`.sdf.gz` copies (388 = 2×194) — caught by the unique-structure count
and cleaned.  Track A needed 4 checkpoint-resumed calls (53→99→147→194);
Track B and the comparative run completed in single calls after the
unique-target dedup (D-V2-045).

### 5. Final state and next actions

- Suite: **1683 passed / 22 skipped**.  Clean-room closures: **33 of
  40** (#34 and #44 this session, both with real-planner evidence).
- Open items: #26/#42/#43 (the authorized-T4 session — unchanged),
  #36 (the operator's `colab ssh` authentication), #28 (licensed
  reaction data — owner decision), #35 (baseline structures —
  operator export), #37 (blocked on #26 + #35).
- Next session (CPU): nothing on the critical path — the remaining
  open items are operator-gated; if the owner authenticates the T4
  runtime, run the S-V2-19 staged chain (restore → precheck →
  suite-B → T1–T4), which closes #26/#36/#43 (and #42's benchmark).
  If the owner exports baseline structures, #35 closes via
  `scripts/run_comparative_benchmark.py --method NAME=DIR`.

### 6. Honest verification ledger

- Attempted: #34, #44, #35 (all three implemented + tested; #34/#44
  closed; #35 harness verified, baseline data pending).
- Implemented: `syntree/engine/evaluator.py`,
  `syntree/engine/comparative.py`, five evaluation scripts, three
  test batteries (67 tests + 3 planner-gated), the evidence corpus +
  two-track planner runs, the frozen-target comparative run, the
  provenance docs.
- Tested locally: the full suite (1683/22 green with the catalog
  artifact present); the real-planner integration tests re-enabled
  with `SYNTREE_AIZYNTH_MODELS` (14/14 in that gate).
- Tested on T4: **none this session** — CPU-only sandbox; no GPU claim
  is made (the T4-gated items remain #26/#42/#43 riding #36).
- Scientifically verified: the independent retrosynthesis evidence on
  the honest #18 construction corpus (two tracks, failure-category
  classification, PoseBusters geometry, CIs) — **the corpus is NOT
  trained-policy output**, and no such claim is made anywhere.
- NOT claimed: any baseline comparison (the structures are not
  provided); any Vina docking score (the adapter is implemented and
  fail-closed but was not exercised in the evidence run — no receptor
  preparation was run on the pocket artifacts); any T4 behavior.

## Session S-V2-21 — 2026-11-05/06 — the authorized bridge fully operational: the T-chain run clean on the T4 (#26's run criteria + #36's remote-workflow criteria evidenced, #42 benchmarked); two live continuation bugs caught and fixed; #43's cross-session driver staged

### 0. State discovery

The prior session (S-V2-20's successor, uncommitted at recycle) had
completed the operator-consent OAuth flow for the official
`google-colab-cli` (refresh-token cache at `~/.config/colab-cli/`,
2026-11-05 17:29–17:50 UTC) and provisioned a first `syntree-t4` session
whose kernel-exec hardware battery (CLI history
`history/syntree-t4.jsonl`: Tesla T4 15360 MiB, host RAM 12.67 GiB
separate, torch 2.11.0+cu130, cc 7.5) was genuine — but its workflow-doc
update overstated the bridge as "root shell confirmed (SSH_BRIDGE_OK)"
with no transcript to back it.  This session re-verified everything from
scratch and corrected the record.

### 1. The bridge, verified for real this time

OAuth refresh still valid (no operator interaction needed).  Fresh T4
provisioned; the §29.2 battery re-run BOTH through the kernel channel and
through the real SSH-proxy path: `colab ssh --proxy-mode -s syntree-t4`
(the CLI reads `~/.ssh/id_ed25519.pub` directly when `--identity` is
absent — passing `-i` shells out to `ssh-keygen`, which this sandbox
lacks) + `scripts/colab_ssh_client.py --proxy` (paramiko) → **root shell
confirmed: `SSH_BRIDGE_OK`, root@b63e22748e63, Tesla T4 15360 MiB**
(transcripts: `verify_syntree-t4_1793902517.json`,
`verify_s-v2-21_1793902640.json` — full battery, mem pools separate,
policy gate PASS).  The `exec`/`put`/`get` surface then drove the remote
runtime end-to-end (clone at pinned commits, SFTP token staging at
`/root/.hf_token` mode 600, nohup orchestration, log polling).

**Operational finding (§29.2 discipline):** two of five provisioning
attempts returned *CPU-only* runtimes silently (the assign endpoint
answers `Service Unavailable` and the CLI falls back without failing).
The bootstrap now gates fail-closed on `nvidia-smi` AND torch-CUDA
concurring on Tesla T4 before any work launches
(`scripts/bootstrap_t4.sh`; non-T4 runtimes are disposed, never trained
on).

### 2. The T-chain — two live continuation bugs caught, fixed, re-run clean

Run 1 (commit `ff2290d`): T1 `CONTROLLED_STOP_OK budget=40`, then T2/T3
**failed** — `checkpoint profile digest does not match the active
profile`: the digest hashed `time_budget_seconds`
(**D-V2-046**, fixed at `ee28713`: digest scope = training semantics
only; operational knobs excluded; 2 regression tests; suite 1686/22).

Run 2 (at `ee28713`, on a second T4): all gates green superficially —
T1 OK, `INTEGRITY_AFTER_KILL: PASS`, T3 `max_epochs_reached`,
`STRICTLY_MONOTONIC_EPOCH_DESCENT: PASS`, `VRAM_UNDER_9_5_GIB: PASS`
(peak reserved 2.24 GiB), HF run persisted — **but two honesty problems**:
(i) run2 completed its 2 remaining steps before the 5 s-poll/10 s-delay
kill could land, so "abrupt interruption" was really a natural exit, and
(ii) T3 resumed the step-42 *completed* checkpoint and silently re-trained
epoch 1 to step 63 — 21 repeated micro-batches, a #43 exact-continuation
violation (**D-V2-047**, fixed at `925071d`: `progress.epoch` advances
past a completed epoch; completed-run resume is a provable no-op; the
T-chain driver made work-aware — T1 probes the total step count first and
re-picks a budget leaving ≥ 15 steps, T2 polls at 2 s and kills 2 s after
fresh telemetry, logging `ABRUPT_KILL_MID_TRAINING` only when the process
was alive with fresh telemetry; suite 1687/22).

Run 3 (at `6445929`, third T4): **clean end-to-end** —
T1 probe 60 s completes (42 micro-steps total) → 30 s budget leaves only
14 (rejected) → **15 s budget stops at step 12/42 (30 left)**;
T2 **SIGKILL mid-training** (telemetry lines 1→3 while alive) +
`INTEGRITY_AFTER_KILL: PASS`; T3 resumes the killed run's step-12
checkpoint to **exactly 42 steps, history_len 42, epochs_completed 2**
(zero repeats — D-V2-047 semantics visible: final `epoch: 2`);
T4 `STRICTLY_MONOTONIC_EPOCH_DESCENT: PASS` (per-epoch means
16.29 → 12.00; first/last micro-batch 27.61 → 0.0054),
`VRAM_UNDER_9_5_GIB: PASS` (peak reserved 2292 MiB = 2.24 GiB, peak
allocated 1501.8 MiB), last-completed logging (epoch 1, batch 20, step
42), `HF_RUN_UPLOAD_DONE: session1` (131 MB run.tar →
`runs/stage1_t4_session1/run.tar`), `TRAIN_CHAIN_DONE`.  Evidence bundle
in `ssh_evidence/session1_chain/` (chain + restore logs, telemetry.jsonl
— append-semantics confirmed across T1/T2/T3 — and the runtime's
hardware preflight).

### 3. #42's T4 execution-path benchmark (at `778ff1c`)

`scripts/benchmark_torsion_refinement.py` ran on the authorized T4
against **real pockets** loaded through the production loaders: 13
terminal states over 13 real pockets (24 sampled; 11 PDBbind metal
pockets fail-closed on MMFF94 typing — recorded in the report, never
hidden), rotatable rxn_07 junctions vs the locked rxn_01 contrast, bins
cycled 1–36 including the periodic boundary.  Per-junction: coarse scan
9 evals + **Brent 8–10 evals**, objective improvement median 1.10 (max
109.98, min 0.0), wall per terminal state mean 9.48 s (median 12.86 s —
dominated by the scaffold-constrained MMFF94 relaxation of ≤700-atom
pockets).  Report: `ssh_evidence/session1_chain/logs/torsion_refinement_benchmark.json`.

### 4. #43's cross-session continuation — driver staged, run blocked on capacity

`scripts/runtime_cross_session_continue.sh` (at `778ff1c`): fetches the
persisted session-1 run from HF, verifies the sidecar, resumes on a NEW
runtime with `max_epochs` extended (legalized by **D-V2-048** — run
length is not a step semantic), asserts the exact-continuation contract
(resumed history's first 42 records identical to the source; new records
present; fresh hardware baseline in the resumed telemetry) and persists
the cross-session run under its own HF tag.  **The execution is blocked
on free-tier GPU capacity**: after session 3 was released, every
provisioning attempt through the end of the session returned CPU-only
runtimes (assign endpoint `Service Unavailable`).  The source run is
persisted and the driver is one command away
(`bootstrap_t4.sh session2 continue`).

### 5. Session verification

Suite **1687 passed / 22 skipped** (baseline in this rebuilt evalenv:
1684/22; delta = the three new regression tests; an earlier flaky
8-failure run was diagnosed as `/tmp` exhaustion — pytest's retained
temp roots, Errno 28 — not code).  Commits: `ee28713` (D-V2-046),
`925071d` (D-V2-047 + the work-aware driver), `6445929` (RUN_TAG
parameterization), `778ff1c` (D-V2-048 + the continuation driver + the
#42 benchmark), all pushed to `origin/3d-synV2`.

## Session S-V2-22 — 2026-11-06 — the dead free tier replaced, the two-round T-chain reproduced on a new authorized account, and #43's cross-VM continuation finally executed (two latent driver defects caught: D-V2-049)

### 0. State discovery: the account died mid-session

The S-V2-22 bootstrap campaign (00:40–01:04 UTC, evidence:
`ssh_evidence/bootstrap/{new_attempt,retry_new,final_new,campaign3_new}*.log`)
hit the documented free-tier failure mode from the other side: the assign
endpoint answered `503 / outcome:2` on every POST while GETs still
returned `acc:t4`, and `ccu-info` reported `currentBalance: 0` —
the operator account's free GPU quota was exhausted (the operator
confirmed: "the free tier in that account is dead").  The sandbox
recycler had also wiped `~/.ssh/id_ed25519.pub` and the git credential
config (the bridge PRIVATE key survived in `.secrets/colab_ed25519`,
never re-generated — only its public half was re-derived).

### 1. Re-auth on a fresh operator account, bridge re-verified

The operator re-consented through the same split PKCE flow
(`scripts/colab_auth_flow.py` url → operator code → exchange; the
one-time code read from a 0600 file, never argv).  The new token
(refresh-enabled) provisioned a T4 on the FIRST attempt; the §29.2
gate PASSed (nvidia-smi AND torch-CUDA concurring: Tesla T4 15360 MiB,
torch 2.11.0+cu130, cc 7.5) and the SSH-proxy battery verified the
root shell (`ssh_evidence/verify_s-v2-newacct_1793929474.json`;
`colab ssh --proxy-mode` + paramiko, host RAM 12.67 GiB tracked
separate from GPU VRAM).  The dead account's token is archived
read-only at `.auth/colab_token_dead_account.json` — nothing revoked.

### 2. The two training rounds reproduced

`runtime_restore_from_hf.sh` restored the workspace bit-exact
(`HF_RESTORE_BITEXACT: PASS` — all three trajectory SHA-256s match the
S-V2-21 values; 388 processed artifacts; catalog checksum verified).
The T-chain then ran TWICE on the same authorized T4, pinned at
`5c65b77`, historical config untouched (`colab_t4_profile_a`: B=12 ×
accum 6, lr 3e-4, wd 1e-5, FP32 tf32-off, seed 2026, pocket cap 700,
max_epochs 2; entry `scripts/train_stage1.py --config
configs/train_colab_t4.json --device cuda`):

- **Round 1 (`RUN_TAG=repro-r1`)**: T1 probe 60 s completes (42
  micro-steps — the historical total) → 30 s budget stops at 27/42
  (15 left) `CONTROLLED_STOP_OK`; T2 `ABRUPT_KILL_MID_TRAINING`
  (telemetry 3→4 while alive) + `INTEGRITY_AFTER_KILL: PASS`; T3 exact
  resume to 42/42 `max_epochs_reached`; T4
  `STRICTLY_MONOTONIC_EPOCH_DESCENT: PASS` (17.155 → 12.190,
  first/last 27.785 → 0.102), `VRAM_UNDER_9_5_GIB: PASS` (peak
  reserved 2258 MiB, peak allocated 1501.8 MiB — the historical
  allocated figure exactly); `HF_RUN_UPLOAD_DONE: repro-r1`;
  `TRAIN_CHAIN_DONE` (01:51–01:54).
- **Round 2 (`RUN_TAG=repro-r2`)**: T1 30 s budget leaves 14
  (rejected, work-aware) → **15 s budget stops at step 12/42 (30
  left) — the exact S-V2-21 Run-3 signature**; T2 kill + integrity
  PASS; T3 42/42; T4 monotonic PASS (16.659 → 14.077, first/last
  26.824 → 0.061), VRAM PASS (peak reserved 2818 MiB, allocated
  1501.8 MiB again); `HF_RUN_UPLOAD_DONE: repro-r2`;
  `TRAIN_CHAIN_DONE` (01:55–01:59:59).

Evidence: `ssh_evidence/repro_r1_r2/` (chain log covering both rounds,
restore log, both outer logs, telemetry.jsonl, the checkpoint sidecar,
hardware_preflight.json).  Both run.tars persist on HF
(`runs/stage1_t4_repro-r1|repro-r2/run.tar`).

### 3. #43's cross-session continuation — executed at last, two latent defects caught (D-V2-049)

The driver staged at `778ff1c` had NEVER executed (S-V2-21 ended
capacity-blocked).  Its first live run died at X1 with
`NameError: SRC_TAG is not defined` — the DL/ME/TB/PU Python heredocs
are QUOTED, so bash never interpolates `${SRC_TAG}/${DST_TAG}/
${EXTRA_EPOCHS}/${N_TEL_BEFORE}` and the f-strings reached Python with
the literal `${...}` text.  After fixing that (env-var passing +
`os.environ` reads), a second latent defect surfaced: X1's preflight
guard checked `$SRC/hardware_preflight.json` (inside `stage1_t4/`)
while `runtime_train_chain.sh` tars the preflight at the ARCHIVE ROOT.
Both fixed at **`895d67c`** (D-V2-049), pushed to `origin/3d-synV2`.

The fixed driver then ran clean twice:

- **Same-VM verification** (`repro-r2 → session2-repro`, the debugging
  run): `CROSS_SESSION_CONTINUATION: PASS` (42 → 63, +21 new, prefix
  identical), `NEW_RUNTIME_TELEMETRY_BASELINE: PASS`, HF-persisted as
  `runs/stage1_t4_session2-repro/run.tar`.
- **The true cross-VM run** (the #43 letter): VM 1 released
  (`colab stop`), a FRESH T4 provisioned on the new account (first
  attempt), §29.2 gate PASS, repo cloned at `895d67c`, workspace
  restored bit-exact, and the continuation executed on the physically
  distinct runtime: `SOURCE_CHECKSUM: PASS` (sidecar), source history
  sha256 `e99b989b…` (identical fingerprint to the VM-1-side source),
  `CROSS_SESSION_CONTINUATION: PASS` (history 42 → 63, +21 new
  records, **prefix identical — no repeats/skips/rewrites**),
  `NEW_RUNTIME_TELEMETRY_BASELINE: PASS` (fresh host-RAM/GPU-VRAM
  baseline), persisted as **`runs/stage1_t4_session2/run.tar`** — the
  originally planned slot.  The epoch-3 final total loss
  (0.0102224592) is **bit-identical across the same-VM and cross-VM
  executions** — the seeded RNG state riding the checkpoint makes the
  resumed trajectory reproduce exactly on a different physical T4.

#43 OPS-03 closed (STATUS row + header updated; 37 of 40).

### 4. Session verification

The pytest suite was NOT re-run this session: the only code delta is
the bash driver fix, which the suite does not cover — its verification
is the live X1–X5 execution on the authorized T4 (bash -n + the
driver's own fail-closed gates all green).  Suite baseline stands at
1687/22 (S-V2-21).  Commits: `895d67c` (D-V2-049 + this record),
pushed to `origin/3d-synV2`.

## Session S-V2-23 — 2026-11-06 — the sandbox rebuilt from scratch (recycle wiped the workspace); the owner-directed R3 trainer profile implemented + verified; the GPU re-authentication staged (pending the operator's Google consent); the GitHub PAT rejected

### 1. State reconstruction (after a full sandbox recycle)

The recycler took **everything** this time: `repo/`, `evalenv/`,
`.secrets/`, `ssh_evidence/`, `~/.config/colab-cli/` (the OAuth cache),
`~/.ssh/` (the bridge keypair) and the untracked sandbox helpers
(`colab_auth_flow.py`, `bootstrap_t4.sh`, `t4_gate.py`).  Rebuilt from
the persisted sources: fresh clone at `5f493ed` (= the S-V2-22 head,
**#43 already closed** — 37/40); `evalenv` rebuilt per the §6.2 recipe
(torch 2.15 CPU + `-e ".[test]"` + h5py/pandas/psutil/huggingface_hub);
`google-colab-cli` 0.7.5 + paramiko reinstalled into `/home/z/.venv`;
a fresh bridge keypair generated (`scripts/gen_bridge_key.py`, private
key in `.secrets/`, 0600); and `scripts/colab_auth_flow.py`
**reconstructed from the CLI's own auth module** (split PKCE: step 1
prints the Google consent URL and persists the verifier; step 2
exchanges the operator-returned one-time code and writes the CLI's own
token cache — §29.1 honored: the agent never touches a password).

**The sandbox disk filled to 100% mid-session** (Errno 28 — the
trainer tests' mkdtemp checkpoints had stacked 7.3 GB in `/tmp` plus
767 pip-cache dirs); cleaned, 6.9 GB recovered — full-suite runs must
keep clearing `/tmp/syntree_trainer_test_*` too, not only
`/tmp/pytest-of-z`.

### 2. The owner-directed R3 trainer extension (D-V2-050)

Implemented `colab_t4_profile_a_r3` exactly as specified and verified:
AdamW `wd=1e-2`; cosine `3e-4 → 3e-5` with scale-aware warmup
(`warmup_fraction` of the frozen 252-step horizon — epoch extension
never reshapes it); validation early stopping (patience 5,
`min_delta=0.10` absolute in the measured ~12–16 loss units, relative
improvement recorded alongside); `best_state.pt` (the full atomic
SHA-256 continuation payload + `kind=best_val` provenance); the
anchor-margin monitor (warn −800 / graceful stop −700 / hard guard
−1000, degradation-increasing convention, loud calibration-pending
mode — **verified: no pre-existing −1000 guard exists in either
branch**, and no Stage-1 quantity lives on that scale; see D-V2-050
for the full analysis).  Profile A stays bit-compatible (digest
verified identical; all persisted checkpoints resumable).

**Verification (independently reproduced this session):**
`pytest tests/test_trainer.py` → **43/43** (26 existing + 17 new);
full suite → **1702 passed / 24 skipped / 0 failed** in 287.6 s.  The
git review (`diff --check`/`--stat`/`--unified=80`/`status`) is clean:
trainer +565/−27, tests +324, the new R3 config.

### 3. Honest ledger (what is NOT done and why)

- **The GitHub PAT the owner supplied is rejected (401 Unauthorized)**
  — verified against the API without printing it.  The R3 commit is
  **local only**; push is blocked until a fresh PAT arrives.
  Anonymous read works (the repo is public); authenticated push does
  not.
- **HF token verified valid (write role, `JJKK1212`)** — the run
  bundle / catalog / R3 run artifacts can persist to HF as before.
- **GPU authentication is staged, not complete**: the real Google
  consent URL was generated (split PKCE step 1 executed; the verifier
  is persisted) and handed to the operator with instructions.  The
  T4 provisioning + the R3 training run (the overfitting watch:
  per-epoch train/val telemetry + early stopping + best-state) execute
  after the operator returns the one-time authorization code.
- No GPU work was attempted or claimed this session yet; the sandbox
  remains CPU-only per #45 §5–§6.

---

## Session S-V2-24 — 2026-11-06 — GPU re-auth completed (two operator code rounds); the R3 T4 run EXECUTED — the overfitting watch answered with live evidence; the run persisted checksum-verified; the sandbox rebuilt from scratch (recycle #3)

### 1. State reconstruction (after a third full sandbox recycle)

The recycler again wiped everything (repo, `.secrets/`,
`~/.config/colab-cli/`, `~/.ssh/`, the untracked sandbox helpers).
Rebuilt: fresh clone at `f4fb715` (= the remote `3d-synV2` head — the
S-V2-23 "push pending" note is superseded: the R3 commit `bcddf7b` +
docs `f4fb715` were already on origin, verified via anonymous
`git ls-remote` + clone); `google-colab-cli` 0.7.5 + paramiko
reinstalled into `/home/z/.venv`; a fresh bridge keypair
(`scripts/gen_bridge_key.py` — `~/.ssh/id_ed25519(.pub)` + private
copy in `.secrets/colab_ed25519`, 0600); and
`scripts/colab_auth_flow.py` reconstructed again from the CLI's own
auth module (same split-PKCE design, now persisting state at the
canonical path AND a `.secrets/` mirror).  Issue #45 re-checked: no
new operator directives after S-V2-20; the constitution stands.

### 2. Operator round-trips (all §29.1-honored; nothing faked)

- **Google consent, attempt 1**: the operator returned a code for the
  S-V2-23-era challenge — orphaned by the recycle (documented failure
  mode; re-issued per protocol).  Attempt 2 (after the fresh URL):
  `invalid_grant` — the code expired while the operator gathered the
  other credentials (codes live ~10 min).  Attempt 3: **exchanged OK,
  refresh_token=yes**; live verification via userinfo
  (`colab_auth_flow.py status`), token values never printed.
- **Fresh GitHub PAT supplied + verified end-to-end** (authenticated
  push → "Everything up-to-date"); stored untracked 0600 at
  `/home/z/my-project/.auth/git-credentials`.
- **HF token supplied + verified** (`JJKK1212`, write role; `runs`
  repo accessible with auth); stored untracked 0600.

### 3. The R3 T4 execution (the owner's overfitting watch)

- **Provisioning + gate**: `colab new -s syntree-t4 --gpu T4` →
  session READY (free-tier balance 0.00, but the assign endpoint
  worked — not the account-death signature); fail-closed gate PASS:
  nvidia-smi AND torch-CUDA both report **Tesla T4 15360 MiB**
  (driver 580.82.07; torch 2.11.0+cu130, cc 7.5, fp32/tf32-off
  SUPPORTED).
- **Workspace restore** (`runtime_restore_from_hf.sh` with the token
  delivered via SFTP to `/root/.hf_token`): repo at `f4fb715` on the
  VM, catalog H5 SHA-256 pinned-MATCH, all three trajectory SHA-256s
  **MATCH — independently re-verified by direct `sha256sum`**, 388
  processed artifacts, `HF_RESTORE_BITEXACT: PASS`.
- **The run**: `train_stage1.py --config configs/train_colab_t4_r3.json
  --max-epochs 12 --device cuda` (241 train / 101 val samples).
  11 epochs completed, global_step 231/252, **stop_reason:
  `early_stopping`** (patience 5, min_delta 0.10, counter exhausted at
  epoch 11).  Validation: 14.2998, 14.1192, 13.9006, 14.5441,
  13.9487, **13.0775 (best, epoch 6)**, 13.358, 14.0625, 14.1656,
  15.3366, 15.4609.  Per-epoch train means: 20.77 → 18.93 → 13.97 →
  13.52 → 12.13 → 10.56 → 10.75 → 10.82 → 10.37 → 9.39.
  **Verdict: overfitting sets in after epoch 6** — train falls
  monotonically while validation rises five consecutive epochs; the
  R3 safeguards caught it live (graceful early stop + `best_state.pt`
  at the epoch-6 best: `kind=best_val`, full continuation payload @
  step 126, early-stopping state riding the checkpoint).  The LR
  cosine + warmup verified live in telemetry (3e-4 peak → decaying;
  warmup over the frozen 252-step horizon never reshaped by the
  epoch count).  The anchor-margin monitor ran in its documented loud
  calibration-pending mode (one telemetry record; thresholds recorded,
  inactive — no Stage-1 energy source, D-V2-050).
- **Persistence + evidence**: run tarred and uploaded to HF
  `JJKK1212/3d-syntree-v2-runs: runs/stage1_t4_r3/run.tar` (261,570,560
  bytes; **local tar sha256 == remote LFS sha256 ==
  `1dc00a5896964eda6b9554375a6960a72761b812c43772da6ee01ddfdec0f92b`**).
  §29.2 battery PASS post-run with the repo-state transcript
  (`ssh_evidence/verify_r3_final_1793978644.json`); small-file
  evidence pulled to `ssh_evidence/r3_run/` (telemetry.jsonl, both
  checkpoint sidecars, restore.log, r3_train.out with the final
  summary, hardware_preflight.json).  VM stopped after use
  (`colab stop`; no sessions left open).

### 4. Honest ledger

- **No issues closed or changed** — 37/40 stands; the R3 execution was
  an owner directive (D-V2-050), not an issue.  The remaining work is
  entirely operator-gated: #35 (baseline structure exports) → closes
  #35 + #37; #28 (licensed-data decision).
- **CPU suite not re-run this session** — this session's diff is
  docs-only; S-V2-23's independently-reproduced **1702 passed / 24
  skipped / 0 failed** at `f4fb715` carries over by construction.
- The `evalenv` dev environment was NOT rebuilt this session (the T4
  path did not need it; the next CPU-side session rebuilds it per
  HANDOFF §6.2).
- One bridge operational note: a detached `nohup` launch holds the
  paramiko channel open (the first launch call hit the channel
  read-timeout — the training itself was unaffected; status checks
  via fresh `exec` calls are the pattern that works).