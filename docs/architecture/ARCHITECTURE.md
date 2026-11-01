# 3D-SynTree V2 — Global Architecture Specification

> **Status:** authoritative target architecture per #45 §24.2 ("design
> globally before implementing locally"). Sections are marked `[TARGET]`
> (specified, not yet implemented), `[VERIFIED]` (implemented and tested on
> this branch, with evidence). This is a **clean-room document**: the V2
> branch was emptied on 2026-11-01 (commit `075f86d`) and every line of V2
> code is newly written against this specification and the issue roadmap
> #5–#44. Superseding decisions are recorded in
> `docs/agent/DECISIONS.md`.
>
> Last updated: 2026-11-01 (session S-V2-02: #5, #6, #7, #21).

## 1. System overview

3D-SynTree V2 generates synthesizable, pocket-aware drug candidates by
**reaction-constrained molecular growth in 3D**, trained in two stages:

```
raw structural data ──► ingestion/curation ──► clustered splits ──►
trajectory extraction (exact forward replay) ──► sharded datasets ──►
      │                                                        │
      │                              Stage 1: supervised behavioral cloning
      │                              (FP32, T4, quantile curriculum)
      ▼                                                        ▼
Enamine REAL synthon catalog ──► pharm3d descriptors + chemoselectivity ──┐
                                                                          │
   Stage 2: GFlowNet Trajectory Balance over the construction DAG  ◄──────┘
      │  (reward = stationary multi-objective pocket-conditioned score)
      ▼
evaluation: chemical validity / PoseBusters / docking / diversity /
            retrosynthesis (stock-aligned AND independent tracks)
```

**MDP / DAG / GFlowNet distinction (binding, #45 §2.3):** the MDP defines
the sequential decision process (state, action, transition, termination,
reward). The **construction DAG** is the acyclic state graph induced by the
growth rules. The GFlowNet learns a forward policy over that DAG. They are
complementary, never "alternative formulations". The molecule-level
proportional-sampling guarantee must handle derivation multiplicity
(canonical derivations, a correction term, or a graph-state backward
policy) — the exact formulation is #38 FLOW-04 `[TARGET]`.

## 2. Layered module map (target)

| Layer | Canonical module(s) | Backlog |
|---|---|---|
| Tensor schemas & batch contracts | `syntree/core/schema.py` (frozen, versioned) | #6 |
| Equivariant math kernels | `syntree/models/equivariant.py` | #7, #8, #9 |
| Precision engine (fail-closed FP32) | `syntree/utils/` | #10 |
| Reaction registry (32 reactions / 9 families) | `syntree/chemistry/` | #11 |
| Synthon catalog + pharm3d + chemoselectivity | `syntree/chemistry/` | #12, #40 |
| Conformer/geometry ops (MMFF94, soft-core LJ, torsion) | `syntree/chemistry/` | #13, #14, #42 |
| Canonical decompositions | `syntree/data/` | #15 |
| Ingestion / curation / splits | `syntree/data/` | #16, #17 |
| Trajectory extraction + exact replay | `syntree/data/` | #18 |
| Sharding + integrity | `syntree/data/` + `scripts/` | #19 |
| Augmentation (train split only) | `syntree/data/` | #20 |
| Backbone `BipartitePaiNN` | `syntree/models/bipartite.py` | #21 |
| Handle selection head | `syntree/models/` | #22 |
| Cross-attention decoder | `syntree/models/` | #23 |
| Hierarchical policy heads + torsion space | `syntree/models/` | #24 |
| Policy assembly + audit | `syntree/models/policy.py` | #25, #39 |
| Trainers (FP32 / curriculum / transfer) | `syntree/engine/` | #26, #27, #28 |
| GFlowNet TB engine | `syntree/engine/` | #29, #38 |
| Stationary reward | `syntree/engine/` | #30 |
| CLI / checkpoint state | `main.py` + `syntree/utils/` | #31, #43 |
| Test suites A/B | `tests/` | #32, #33 |
| Evaluation (retrosynthesis, SBDD benchmark) | `scripts/` | #34, #35, #44 |
| Remote execution + artifact sync | notebook + `syntree/utils/` | #36, #37 |

## 3. Binding cross-cutting contracts

### 3.1 Tensor schema separation `[VERIFIED — #6, 2026-11-01]`

Invariant chemical features (`[N, d]`), Cartesian positions (`[N, 3]`),
atomic numbers, and formal charges are **separate tensors**; never
concatenate descriptors with coordinates into one feature vector (the
legacy "Franken-tensor" 64+3 mistake). Schemas are frozen and versioned;
malformed shapes and schema-version mismatches fail closed.

Implemented in `syntree/core/schema.py`: the frozen `StateTensorContract`
(12 canonical tensor fields + `schema_version`, `SCHEMA_VERSION = 2`),
`validate_state_batch` (exact dtype/rank/shape, cross-field length and
batch-index consistency, atomic-number bounds 1..100, finiteness), the
canonical `collate_state_batch` (single batching path; materialises
`pocket_batch`/`ligand_batch`; explicit empty-ligand t=0 states; stamps and
re-validates the assembled batch), and the hard rejection of legacy 67-dim
tensors (`ValueError("Legacy 67D Franken-tensors are deprecated ...")`,
message prefix pinned by test). The collator performs the only documented
normalisations (0-dim → `[1]`, 1-D `[64]` → `[1, 64]`); everything else is
a hard error. Producers stamp `schema_version` via `stamp_schema_version`
(tensor form, survives collation to `[B]`); unstamped objects pass field
validation as an interim rule until shard producers exist (#19).

### 3.2 Equivariant geometry operators `[VERIFIED — #7, 2026-11-01]`

All norms and unit vectors route through
`syntree/models/equivariant.py::safe_norm` / `safe_unit_vector`
(`‖x‖_ε = sqrt(Σx² + ε²) − ε`, `ε = 1e-5`): exact `0.0` value and strictly
zero gradient at `x = 0`, bounded gradients everywhere, `‖unit‖ < 1`
strictly. The PaiNN message/mixing blocks are implemented from the
published equations (Schütt et al. 2021, Eqs. 7–10; see decision D-V2-004)
with the singular `r_ij/‖r_ij‖` replaced by the safe primitive. Scalar
invariance / vector equivariance under random rotations, translation
invariance, and coincident-coordinate finiteness are pinned by
`tests/test_mathematical_singularities.py`.

### 3.3 Union interaction graph & backbone `[VERIFIED — #21, 2026-11-01]`

`syntree/models/bipartite.py::BipartitePaiNN` is the single authoritative
molecular encoder over the union pocket∪ligand node set (ordering:
`[pocket nodes, ligand nodes]`, per-graph-major inside each block for
collated batches). 8 PaiNN message-then-update layers (d = 256, vector
3×256, 20 Gaussian RBFs, r_cut = 5.0 Å — frozen `BACKBONE_*` constants;
5,999,104 parameters pinned by test) compose the #7 blocks. Node features
are strictly invariant chemistry (shared z-embedding 1..100, node-type
embedding, bias-free charge MLP, handle-flag embedding); vector features
start at zero and acquire geometry only through message passing.
`build_union_radius_graph` is the **one** authoritative edge construction
(#45 §7.2): exact radius graph emitting pocket–pocket, ligand–ligand and
pocket–ligand pairs of the same state (both directions, no self-loops, no
cross-graph edges), `safe_norm` threshold, differentiable `edge_vec` /
`edge_dist` only. `GaussianRadialBasis` emits the raw 20-Gaussian expansion
with **no** cosine envelope — the envelope for all spatial RBF projections
is owned by #9 (decision D-V2-007). Every layer is wrapped in non-reentrant
activation checkpointing while gradients are enabled (T4 capability;
measured benefit deferred to #43). Inputs are fail-closed validated (FP32
production / FP64 local verification only, decision D-V2-008; atomic-number
bounds; ≥1 pocket atom per graph) and consume the canonical
`collate_state_batch` output. `BackboneOutput` exposes invariant scalars,
equivariant vectors, union metadata and per-graph pooled contexts — the
#39-facing surface; downstream heads must consume it, never introduce a
second pocket encoder. Verified: scalar invariance / vector equivariance
(worst errors ≤3e-13 over 36 rigid-motion trials), clean pocket-only
operation at N_l = 0, batching ≡ single-graph runs. Pinned by
`tests/test_bipartite_painn.py`.

### 3.4 Precision `[TARGET — #10]`

Production path on Colab/Kaggle Tesla T4: **pure FP32**. No FP16, no BF16,
no autocast, no `GradScaler`, no silent fallback. V2 contains no
mixed-precision machinery by construction (decision D-V2-005); #10 adds the
fail-closed enforcement layer (device/dtype guards, config validation,
runtime assertions) before the T4 trainer is certified (#26).

### 3.5 Memory policy `[TARGET — #43]`

Architecture dimensions are frozen in version-controlled configuration (no
VRAM-probing mutation). Explicit micro-batch / gradient-accumulation /
pocket-cap settings; activation checkpointing where measured to help. T4
profile targets (e.g. batch 12 × accum 6, pocket cap 700, peak reserved
VRAM < 9.5 GB) must be **measured** on real T4 hardware, never assumed.

### 3.6 Stage-2 algorithm `[VERIFIED as V2 policy — #45 §12.3]`

Stage 2 is **GFlowNet Trajectory Balance only**; PPO/DPO are not V2
production alternatives. The forward probability must include the seed,
STOP, handle, reaction, synthon, stereo, and torsion actions belonging to
the trajectory. The backward term is zero only if mathematically justified
for the actual state space — #38 owns that proof; no assumption is baked in
until then.

### 3.7 Reward stationarity `[TARGET — #30]`

Terminal TB reward must be a **stationary, reproducible** positive function
of the terminal object + fixed conditioning context. Batch-relative
diversity terms inside R(x) are forbidden; population diversity lives in
sampling / replay / archive monitoring outside the per-trajectory reward.

### 3.8 Data integrity & leakage prevention `[TARGET — #16–#19]`

Full multi-source ingestion (CrossDocked2020 Pocket10 with the RMSD
criterion, PDBbind refined, BindingMOAD), 10 Å pocket standardization
around the co-crystallized ligand centroid with a documented
waters/salts/additives/metals policy, MMseqs2 30%/80% cluster-connected
splits with whole-component allocation (no component crosses splits),
exact forward-replay validation of every supervised step, SHA-256-checked
compressed shards. Provenance of raw sources is pinned in
`docs/data/RAW_SOURCE_PROVENANCE.yaml`.

### 3.9 Torsion handling `[TARGET — #24, #42]`

36-bin categorical torsion action over [-π, π); the bin selects a coarse
rotamer region; environment-side bounded refinement (±5°, documented
optimizer) over a soft-core pocket objective; scaffold-constrained MMFF94
relaxation afterward. The refinement is a *geometry* operation attached to
the categorical action — no unnormalized continuous density inside the TB
log-probability.

### 3.10 Reproducibility & artifacts `[TARGET — #37, #43]`

Checkpoints persist model/optimizer/scheduler/global step/epoch/RNG
states/sampler-curriculum/data-progress/GFlowNet `log_Z` atomically with
integrity checks; session-safe resume proven by a real cross-session T4
continuation. Artifacts (datasets, manifests, catalog HDF5, index,
checkpoints, logs) sync to Hugging Face Hub with checksums; credentials
only via environment variables.

## 4. Repository layout (V2, growing with the roadmap)

```
src/syntree/              # the V2 package (src-layout, decision D-V2-003)
  core/                   # schema: frozen state contract & canonical batch path
  models/                 # equivariant math kernels; union-graph backbone;
                          # policy heads later
    equivariant.py        #   safe norms + PaiNN blocks (#7)
    bipartite.py          #   BipartitePaiNN + radius graph + RBF (#21)
docs/
  architecture/           # this document
  data/RAW_SOURCE_PROVENANCE.yaml
  agent/                  # STATUS, WORK_LOG, DECISIONS, DEPENDENCIES, HANDOFF
tests/                    # pytest suites, file names pinned by owning issues
```

Modules are added by their owning issues in dependency order; nothing is
pre-created speculatively.

## 5. Open architectural questions (tracked, not forgotten)

1. **Derivation multiplicity** in TB backward semantics (#38) — canonical
   derivations vs correction term vs graph-state backward policy.
2. **Seed/anchor action** integration (#39) — where the pocket-anchor
   variables live in the state and their probability measure.
3. **pharm3d projection versioning** (#12) — QR-projection seed/matrix
   persistence contract.
4. **Empty-ligand interaction graphs** at t=0 for the backbone —
   **resolved by #21**: the schema supports empty ligand sub-graphs
   explicitly and `BipartitePaiNN` runs cleanly as a pocket-only encoder
   when `N_l = 0` (pinned by `tests/test_bipartite_painn.py`).
