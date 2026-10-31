# 3D-SynTree V2

**Reaction-constrained, pocket-conditioned molecular generation with an
auditable synthesis route — re-engineered as one coherent system.**

3D-SynTree grows a ligand inside a 3D protein binding pocket using only
*certified chemical reactions* and *purchasable building blocks*, so every
generated molecule comes with both a 3D pose and a verifiable construction
history. V2 is a clean re-engineering of the complete system — its
scientific formulation, numerical foundations, chemical constraints, data
pipeline, training objectives, and evaluation contracts were redesigned
together rather than patched onto the earlier prototype.

> **Status:** early V2 (foundation phase). See the honest
> [implementation-status table](#16-implementation-status) — nothing on
> this branch has been trained at production scale yet, and no benchmark
> numbers are claimed.

---

## 1. Purpose

Structure-based drug design (SBDD) needs molecules that are
simultaneously (a) shaped by the 3D binding environment and (b)
chemically valid and *synthesizable*. Purely coordinate-driven generators
can produce plausible poses with no synthesis route; purely
reaction-based 2D generators can encode chemistry while ignoring pocket
geometry. 3D-SynTree couples both: at every construction step the policy
chooses among **chemically applicable** actions — seed fragment and
pocket anchor, reactive handle, certified reaction, compatible catalog
synthon, torsion/rotamer action, or explicit termination — so the 3D
growth process *is* the synthetic route.

## 2. Scientific formulation

### 2.1 MDP, construction DAG, and GFlowNet — how they relate

These are three complementary views, not alternatives:

- **MDP** — the sequential decision problem: state (pocket + current
  ligand + handles), action (the chemistry-constrained choices above),
  transition (apply reaction, place fragment, refine geometry),
  termination (STOP), reward.
- **Construction DAG** — the directed acyclic graph of molecular
  construction states induced by the growth rules. The MDP *walks* this
  DAG; the DAG *is* the state space.
- **GFlowNet** — a forward sampling policy trained with **Trajectory
  Balance (TB)** so that terminal molecules are sampled in proportion to
  their reward under the fixed pocket context, preserving diversity
  instead of collapsing onto a single greedy hit.

A molecule can admit several construction histories (reaction orders);
the derivation-multiplicity semantics of TB over this DAG — canonical
derivations, a correction term, or a graph-state backward policy — is
being formalized explicitly (issue #38) rather than assumed away.

### 2.2 Two-stage training

1. **Stage 1 — supervised behavioral cloning** on expert trajectories
   extracted from real protein–ligand complexes and validated by exact
  forward replay (the internal reaction sequence must reconstruct the
   observed product graph exactly, including stereochemistry).
2. **Stage 2 — GFlowNet Trajectory Balance** fine-tuning with a
   *stationary* multi-objective terminal reward (pocket contact/clash,
   physicochemical terms). PPO and DPO are deliberately **not** part of
   the V2 design.

## 3. Architecture

```
inputs:  pocket (X_P, Z_P, q)           ligand state (X_L, Z_L, q)
              │                                │
              └────────► BipartitePaiNN ◄──────┘
              scalar s_i ∈ R^d   ·   vector v_i ∈ R^{3×d}
              (pocket–pocket, ligand–ligand, pocket–ligand interactions)
                          │
              2-layer cross-attention decoder
                          │
        hierarchical policy heads (STOP evaluated first, then
        seed/anchor · reaction family · reaction · handle · synthon ·
        stereo-parity · 36-bin torsion; invalid combinations masked;
        each distribution normalized over its valid alternatives)
                          │
        environment: apply reaction → place synthon → torsion refinement
        → scaffold-constrained MMFF94 relaxation → updated state
```

Key contracts (full specification in
[`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md)):

- **Strict feature/coordinate separation.** Invariant chemical features,
  Cartesian positions, atomic numbers, and formal charges are separate
  tensors — never concatenated into one mixed vector (the legacy
  "Franken-tensor" design is forbidden).
- **Bipartite SE(3)-equivariance.** One `BipartitePaiNN` over the union of
  pocket and ligand atoms, with scalar invariance and vector equivariance
  verified under random rigid motions.
- **STOP factorization.** STOP is an explicit absorbing action evaluated
  *before* continuation heads; if STOP is chosen, continuation heads are
  not scored.
- **Learned seeding.** The initial seed fragment and pocket anchor are
  explicit policy actions (not a fixed heuristic), included in the
  trajectory's forward log-probability.

## 4. Mathematical foundations

| Component | Requirement | Issue |
|---|---|---|
| Geometric norms | smooth ε-norm with well-defined zero-vector gradient (no division-by-zero) | #7 |
| Circular statistics | stable `log I₀(κ)` and the bounded Amos' ratio approximation `I₁/I₀ ≈ κ/(0.5+√(κ²+2.25))` for the torsion objective, validated against a trusted reference | #8 |
| Radial basis | continuous cosine cutoff envelopes that reach exactly zero at the cutoff | #9 |
| Precision | **pure FP32** production path on T4 (no fp16/bf16/autocast/GradScaler) | #10 |
| TB objective | `log Z + Σ log P_F − Σ log P_B − log R`, with backward semantics proven for the actual state space | #29, #38 |
| Reward | stationary, reproducible, positive; identical molecules in identical contexts receive identical rewards | #30 |

Numerical stability must come from mathematically appropriate functions —
never from exception swallowing, arbitrary clamping, or NaN substitution.
The V1 fp16 training incident (GradScaler collapse → fully NaN
checkpoints, forensics preserved in
[`docs/legacy/PERSISTENT_ERRORS_NAN.md`](docs/legacy/PERSISTENT_ERRORS_NAN.md))
is the standing motivation for the FP32 mandate.

## 5. Chemistry

- **Reaction registry** — 32 certified transformations in 9 coarse policy
  families, each with SMARTS templates, handle requirements, atom/bond
  mapping, stereochemical behavior, sanitization, and failure semantics
  (#11). Deterministic deprotections are documented explicitly.
- **Synthon catalog** — an Enamine REAL building-block subset (target:
  50,000 blocks) with documented MW / rotatable-bond / heavy-atom / Fsp3
  constraints and aryl coupling exemptions (#12). Every entry is a real
  catalog record — nothing is fabricated. `pharm3d` descriptors (2D,
  3D-shape, electrostatic, dipole, handle-directional) are projected
  through a versioned orthonormal QR basis and persisted in an immutable
  HDF5 artifact.
- **Chemoselectivity** — reaction-aware screening for competing
  functional groups, incompatible centers, and self-reactivity, with
  reason codes (#40).
- **Retrieval** — a versioned hashed inverted index (family × handle ×
  feature hash) prefilters candidates; every candidate still passes exact
  compatibility and reaction execution (#41).
- **Geometry** — categorical 36-bin torsion selection, bounded ±5°
  pocket-conditioned refinement over a soft-core clash objective, then
  scaffold-constrained MMFF94 relaxation that never escapes clashes by
  distorting valence (#13, #14, #42).

## 6. Data and leakage prevention

Sources (under their own licenses; provenance and checksums pinned in
[`docs/data/RAW_SOURCE_PROVENANCE.yaml`](docs/data/RAW_SOURCE_PROVENANCE.yaml)):
**CrossDocked2020 Pocket10**, **PDBbind Refined**, **BindingMOAD**.
Pockets are standardized around the co-crystallized ligand centroid
(10 Å radius) with a documented water/salt/metal policy. Splits are
generated with **MMseqs2 at 30% sequence identity / 80% coverage**;
homologous chains form connected components allocated whole into
80/10/10 partitions — **no connected component crosses splits**. Counts
are always read from manifests, never hardcoded. Augmentation
(coordinate jitter, dihedral exposure) applies to the training split only.

## 7. Hardware and precision policy

The only supported GPU environments are **Google Colab Tesla T4** and
**Kaggle Tesla T4** (pure FP32, frozen architecture dimensions in
version-controlled config, explicit micro-batch/gradient-accumulation
settings, measured — not assumed — memory profiles). A dedicated Colab
SSH workflow (#36) drives remote verification and training; cross-session
checkpoint continuation is a first-class requirement (#43).

## 8. Installation

```bash
git clone -b 3d-synV2 https://github.com/Vtheonly/3d-syntree.git
cd 3d-syntree
pip install -r requirements.txt        # or: pip install wheels/*.whl
python -m pytest tests/ -q             # local CPU verification
```

Core dependencies: PyTorch, PyTorch Geometric, RDKit, NumPy/SciPy/pandas,
pyarrow. Optional at runtime: MMseqs2 (splits), AutoDock Vina/GNINA
(docking scores), AiZynthFinder (retrosynthesis evaluation),
`huggingface_hub` (artifact sync; token via the `HF_TOKEN` environment
variable — never hardcoded).

## 9. Execution

```bash
python main.py --mode info                    # runtime/reaction diagnostics
python main.py --mode train --config configs/train_colab_12h.json
python main.py --mode rl    --config configs/rl_colab_12h.json --resume-auto
python main.py --mode generate --config ... --pocket-dir data/targets
python main.py --mode evaluate --config ...
```

Data assets are staged by `scripts/download_assets.py` (verified HF
sharded dataset). The canonical Colab entry point is
`notebooks/run_3d_syntree.ipynb`. The V2 multi-source dataset build
pipeline (#16) is not implemented yet; the notebook's build gate fails
closed until it is.

## 10. Evaluation and scientific honesty

- **Internal forward replay** proves the generator's own reaction chain
  reconstructs the molecule — it does *not* prove an independent planner
  can rediscover the route.
- **Retrosynthesis (AiZynthFinder)** is reported as two strictly separate
  tracks (#34, #44): *generator-stock-aligned* (the exact versioned
  synthon subset as stock) and *independent external-stock*. The two
  solve rates are never merged.
- **Comparative SBDD benchmark** (#35) reports chemical validity,
  PoseBusters validity, docking/affinity, diversity, and Fsp3 against
  documented baselines with frozen held-out targets.
- Held-out test data is never used for tuning.

## 11. Reproducibility

Versioned configs; seeded determinism; SHA-256-checked compressed data
shards with manifests; atomic checkpoints carrying model, optimizer,
scheduler, global step, RNG states, curriculum/sampler state, data
progress, and the GFlowNet partition function; artifacts synchronized to
Hugging Face Hub with checksums (#37). Agent workflow, decisions, and
verified status are documented under [`docs/agent/`](docs/agent/) — the
work log records what was *attempted / implemented / tested locally /
tested on T4 / scientifically validated*, separately.

## 12. Repository structure

```
main.py                 CLI (train | rl | generate | evaluate | info)
syntree/                chemistry/ · core/ · data/ · engine/ · models/ · utils/
scripts/                asset staging, preprocessing, sharding, benchmarks
configs/                default, Colab T4 training, Stage-2 GFlowNet configs
notebooks/              Colab entry point
tests/                  pytest suite (Suites A/B: #32/#33)
docs/                   agent/ · architecture/ · data/ (provenance) · legacy/
wheels/                 vendored RDKit + PyG wheels for offline runtimes
```

## 13. Limitations

- The scientific claims above describe the **V2 design**; much of it is
  not yet implemented (see the status table). In particular: no
  production-scale training, no measured T4 profiles, no benchmark
  results, and no wet-lab validation exist for V2.
- The reaction set and catalog constrain the reachable chemical space;
  infeasibility under this grammar is not evidence of general
  infeasibility.
- Docking scores used as reward components are surrogate physics, not
  experimental affinities.
- Comparative statements about other SBDD methods will only be made with
  traceable sources and fair configurations.

## 14. License

See [`LICENSE`](LICENSE).

## 15. Project governance

The permanent operating constitution for AI engineering agents is GitHub
issue **#45** (never closed). The implementation backlog is issues
**#5–#44**; current status is tracked in
[`docs/agent/STATUS.md`](docs/agent/STATUS.md) and synchronized with the
issue tracker.

---

## 16. Implementation status

Honest, evidence-linked state of this repository (updated 2026-11-01;
see `docs/agent/STATUS.md` for the per-issue detail):

| Capability | Status |
|---|---|
| Legacy purge & V2 foundation (repo layout, agent docs, architecture spec) | **Implemented + locally verified** (#5) |
| Tensor schema freeze / PyG batch contract | **Implemented + locally verified** (#6: `syntree/core/schema.py`, fail-closed, legacy 67-D rejected) |
| Non-singular norms, von Mises stability, cosine cutoffs, FP32 engine | Non-singular norms **implemented + locally verified** (#7); von Mises stability (#8, known V1 test failures open), cosine cutoffs (#9), FP32 engine (#10) pending |
| Certified reaction registry (32 reactions) | Partially implemented (V1 registry exists; V2 certification pending #11) |
| Enamine REAL 50k catalog + pharm3d + chemoselectivity + index | Planned (#12, #40, #41); V1 subset catalog exists (2.6k entries, V1 pipeline) |
| Conformer/torsion geometry stack | Partially implemented (V1 conformer engine; V2 contracts #13/#14/#42 pending) |
| Multi-source ingestion, MMseqs2 splits, exact-replay trajectories, sharding | Partially implemented (V1 pipeline; V2 re-engineering #16–#19 pending) |
| BipartitePaiNN backbone, heads, policy assembly | Partially implemented (V1 models exist; V2 contracts #21–#25, #39 pending) |
| Trainer (FP32/T4), curriculum, transfer pretraining | Partially implemented (V1 trainer/curriculum exist; #26–#28 pending; AMP machinery still present) |
| GFlowNet Trajectory Balance engine | Partially implemented (V1 TB engine exists; #38 semantics + #30 stationary reward pending) |
| Test Suites A/B | Partially implemented (suite: 906 passing / 12 pre-existing failures; #32/#33 pending) |
| Independent retrosynthesis evaluation, SBDD benchmark | Planned (#34, #35, #44) |
| Colab SSH remote workflow, HF artifact sync, T4 profiling/resume | Planned (#36, #37, #43); no GPU execution has occurred for V2 |
| Production training, benchmark results, wet-lab validation | **Not performed — no claims made** |

---

*3D-SynTree V2 — one coherent, chemically constrained, numerically stable,
reproducible system; every claim in this README is either implemented and
tested on this branch, or explicitly marked otherwise.*
