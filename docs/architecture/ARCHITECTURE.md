# 3D-SynTree V2 — Global Architecture Specification (living document)

> **Status:** authoritative target architecture per #45 §24.2 ("design
> globally before implementing locally"). Sections are marked
> `[TARGET]` (specified, not yet implemented), `[INTERIM]` (V1-inherited
> code present on the branch, re-verification pending), or
> `[VERIFIED]` (implemented + tested on this branch).
> Last updated: 2026-11-01 (S-2026-11-01-B, #6 ARCH-02). Update this file whenever a
> durable contract changes; supersede decisions in
> `docs/agent/DECISIONS.md`.

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
growth rules. The GFlowNet learns a forward policy over that DAG. These are
complementary, never "alternative formulations". The molecule-level
proportional-sampling guarantee must handle derivation multiplicity
(canonical derivations, a correction term, or graph-state backward policy) —
the exact formulation is #38 FLOW-04 `[TARGET]`.

## 2. Layered module map (target)

| Layer | Canonical module(s) | Backlog |
|---|---|---|
| Tensor schemas & batch contracts | `syntree/core/schema.py` (frozen, versioned) | #6 |
| Equivariant math kernels | `syntree/models/equivariant.py` | #7, #8, #9 |
| Precision engine | `syntree/utils/hardware.py` (fail-closed FP32) | #10 |
| Reaction registry (32 reactions / 9 families) | `syntree/chemistry/reactions.py` | #11 |
| Synthon catalog + pharm3d + chemoselectivity | `syntree/chemistry/catalog.py`, `synthon_encoder.py` | #12, #40 |
| Conformer/geometry ops (MMFF94, torsion, soft-core LJ) | `syntree/chemistry/conformer.py` | #13, #14, #42 |
| Canonical decompositions | `syntree/data/trajectory.py` | #15 |
| Ingestion / curation / splits | `syntree/data/curation.py`, `splits.py`, `multidataset.py` | #16, #17 |
| Trajectory extraction + exact replay | `syntree/data/trajectory.py` | #18 |
| Sharding + integrity | `scripts/shard_and_upload.py`, `syntree/data/hf_loader.py` | #19 |
| Augmentation (train split only) | `syntree/data/` | #20 |
| Backbone `BipartitePaiNN` | `syntree/models/bipartite.py` | #21 |
| Handle selection head | `syntree/models/synthon_head.py` | #22 |
| Cross-attention decoder | `syntree/models/` | #23 |
| Hierarchical policy heads + torsion space | `syntree/models/{reaction,torsion}_head.py` | #24 |
| Policy assembly + audit | `syntree/models/policy.py` | #25, #39 |
| Trainers (FP32 / curriculum / transfer) | `syntree/engine/trainer.py`, `curriculum.py` | #26, #27, #28 |
| GFlowNet TB engine | `syntree/engine/gflownet.py` | #29, #38 |
| Stationary reward | `syntree/engine/rl.py` (reward section) | #30 |
| CLI / checkpoint state | `main.py`, `syntree/utils/checkpoint.py` | #31, #43 |
| Test suites A/B | `tests/` | #32, #33 |
| Evaluation (retrosynthesis, SBDD benchmark) | `scripts/` | #34, #35, #44 |
| Remote execution + artifact sync | notebook, `syntree/utils/checkpoint.py` | #36, #37 |

## 3. Binding cross-cutting contracts

### 3.1 Tensor schema separation `[VERIFIED — #6, 2026-11-01]`

Invariant chemical features (`[N, d]`), Cartesian positions (`[N, 3]`),
atomic numbers, and formal charges are **separate tensors**; never
concatenate descriptors with coordinates into one feature vector (the V1
"Franken-tensor" 64+3 mistake). Schemas are frozen and versioned; malformed
shapes and schema-version mismatches fail closed.

Implemented in `syntree/core/schema.py` (#6 ARCH-02): the frozen
`StateTensorContract` (13 canonical fields, `SCHEMA_VERSION = 2`),
`validate_state_batch` (exact dtype/rank/shape, cross-field length and
batch-index consistency, atomic-number bounds, finiteness), the canonical
`collate_state_batch` (single batching path; materialises `pocket_batch`/
`ligand_batch`; empty-ligand t=0 states supported; stamps + validates),
and the hard rejection of legacy 67-dim tensors
(`ValueError("Legacy 67D Franken-tensors are deprecated ...")` — exact
message pinned by `tests/test_tensor_contracts.py`). `SynTreePolicy.forward`
now validates its input through this contract — all implicit fallbacks
(`pocket_pos_batch` aliasing, zeros defaults, 1-D reshape guessing, legacy
67→64+3 splitting) are deleted. Producers (`environment.py`,
`trajectory.py`) stamp `schema_version`; dimension constants
(`HANDLE_FEATURE_DIM=64`, `GLOBAL_FEATURE_DIM=4`) live in the schema module
and are re-exported by the featurizer.

### 3.2 Precision `[TARGET — #10]`

Production path on Colab/Kaggle Tesla T4: **pure FP32**. No FP16, no BF16,
no autocast, no `GradScaler`, no silent fallback. `[INTERIM]` V1 AMP
machinery (`resolve_amp_dtype`, `torch.amp.GradScaler`, autocast blocks in
`trainer.py`, `mixed_precision` config keys) is still present and must be
removed/re-engineered fail-closed by #10. The V1 fp16 NaN incident chain
(GradScaler scale collapse → all-NaN checkpoint; forensics in
`docs/legacy/PERSISTENT_ERRORS_NAN.md`) is the motivating evidence.

### 3.3 Memory policy `[TARGET — #43]`

Architecture dimensions are frozen in version-controlled config (no
VRAM-probing mutation — the auto-scale subsystem was purged in #5).
Explicit micro-batch/gradient-accumulation/pocket-cap settings; T4 profile
targets (e.g. batch 12 × accum 6, pocket cap 700, peak reserved VRAM
< 9.5 GB) must be **measured** on real T4 hardware, never assumed.
`[INTERIM]` Runtime OOM-skip + epoch-end adaptive micro-batch halving
retained (resilience, not architecture mutation) — fate decided by #26/#43.

### 3.4 Stage-2 algorithm `[VERIFIED as V2 policy — #5]`

Stage 2 is **GFlowNet Trajectory Balance only**; PPO and DPO were purged
(D-2-004). `[INTERIM]` The existing `GFlowNetTrainer` assumes a unique
parent (zero backward term) and a fixed heuristic seed; #38 must prove or
replace this (derivation multiplicity!), and #39 makes the seed/anchor an
explicit policy action included in the forward log-probability.

### 3.5 Reward stationarity `[TARGET — #30]`

Terminal TB reward must be a **stationary, reproducible** positive function
of the terminal object + fixed conditioning context. `[INTERIM]` The V1
`ThreeDReward` still contains a batch-relative Tanimoto `diversity_weight`
term (currently configured 0.2 in `configs/rl_colab_12h.json`) — a
documented violation of #45 §12.4 that #30 must remove; population
diversity moves to sampling/replay/archive monitoring outside R(x).

### 3.6 Data integrity & leakage prevention `[TARGET — #16–#19]`

Full multi-source ingestion (CrossDocked2020 Pocket10 with RMSD criterion,
PDBbind Refined, BindingMOAD), 10 Å pocket standardization around the
co-crystallized ligand centroid, MMseqs2 30%/80% cluster-connected splits
with whole-component allocation (no component crosses splits), exact
forward-replay validation of every supervised step, SHA-256-checked
compressed shards. Provenance of raw sources is pinned in
`docs/data/RAW_SOURCE_PROVENANCE.yaml`. `[INTERIM]` Backend set is
`("huggingface", "trajectory_pt")`; the V1 multidataset pipeline
(`scripts/preprocess_multidataset.py`, `syntree/data/multidataset.py`,
`splits.py`) exists but is not yet certified against the V2 criteria.

### 3.7 Torsion handling `[TARGET — #24, #42]`

36-bin categorical torsion action over [-π, π); the bin selects a coarse
rotamer region; environment-side bounded refinement (±5°, documented
optimizer) over a soft-core pocket objective; scaffold-constrained MMFF94
relaxation afterward. The refinement is a *geometry* operation attached to
the categorical action — no unnormalized continuous density inside the TB
log-probability. `[INTERIM]` V1 has a continuous von Mises torsion head —
to be replaced per #24; known von Mises numerical test failures are
pre-existing (see STATUS.md).

### 3.8 Reproducibility & artifacts `[TARGET — #37, #43]`

Checkpoints persist model/optimizer/scheduler/global step/epoch/RNG
states/sampler-curriculum/data-progress/GFlowNet `log_Z` atomically with
integrity checks; session-safe resume proven by a real cross-session T4
continuation. Artifacts (datasets, manifests, catalog HDF5, index,
checkpoints, logs) sync to Hugging Face Hub with checksums; credentials
only via environment variables.

## 4. Repository layout (V2)

```
main.py                    # CLI: train | rl | generate | evaluate | info
syntree/
  chemistry/               # reactions, catalog, conformer, validator, encoder
  core/                    # schema (frozen state contract), registry, masking
  data/                    # featurizer, fragmenter, curation, splits,
                           # trajectory, hf_loader, multidataset, synthon_library
  engine/                  # trainer, curriculum, environment, generator,
                           # evaluator, gflownet, rl (rollout+reward)
  models/                  # equivariant, bipartite, policy, heads
  utils/                   # hardware, checkpoint, logger
scripts/                   # download_assets, preprocess/split, shard_and_upload,
                           # build_synthon_catalog/trajectories, benchmarks, verify_*
configs/                   # default_config.json, train_colab_12h.json, rl_colab_12h.json
notebooks/run_3d_syntree.ipynb
tests/                     # pytest suite (A/B re-engineering: #32/#33)
docs/
  agent/                   # STATUS, WORK_LOG, DECISIONS, DEPENDENCIES, HANDOFF
  architecture/            # this document
  data/RAW_SOURCE_PROVENANCE.yaml
  legacy/                  # V1 historical records (read-only)
wheels/                    # vendored rdkit + torch_geometric wheels
```

## 5. Open architectural questions (tracked, not forgotten)

1. **Derivation multiplicity** in TB backward semantics (#38) — canonical
   derivations vs correction term vs graph-state backward policy.
2. **Seed/anchor action** integration (#39) — where the pocket-anchor
   variables live in the state and their probability measure.
3. **pharm3d projection versioning** (#12) — QR-projection seed/matrix
   persistence contract.
4. **Interim AMP removal order** (#10) — must land before #26's T4 trainer
   certification.
