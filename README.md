# 3D-SynTree V2

**Structure-based molecular design via reaction-constrained, pocket-aware
synthon assembly in 3D.**

3D-SynTree V2 generates synthesizable, pocket-aware drug candidates by
combining two constraints that most generative approaches treat separately:

1. **3D geometric awareness** — growth is conditioned on the atomic
   environment of a protein binding pocket, processed by SE(3)-equivariant
   neural networks, and
2. **synthetic accessibility** — every construction step is a *certified
   chemical reaction* applied to purchasable building blocks, so each
   generated molecule comes with an auditable synthetic construction
   history rather than a post-hoc synthesizability guess.

> **Project status: clean-room V2 rebuild in progress.**
> The `3d-synV2` branch was deliberately emptied on 2026-11-01 (commit
> `075f86d`) and is being rebuilt as a completely new codebase from that
> empty starting point. Nothing in this repository is inherited from the
> previous implementation: every file is newly designed and newly written
> against the approved issue roadmap (#5–#44) and the permanent operating
> constitution (#45). See **Implementation status** below for exactly what
> exists, and read claims there — not in this overview — as the source of
> truth for what is implemented.

---

## 1. Scientific objective

Structure-based drug design (SBDD) needs molecular generation that respects
both the three-dimensional protein binding environment and chemical /
synthetic constraints. Purely coordinate-driven generation can produce
plausible-looking poses without guaranteeing chemically valid or
synthesizable structures; purely 2D reaction-based generation encodes
chemical transformations and purchasable building blocks but ignores the 3D
geometry of the pocket. 3D-SynTree performs **reaction-constrained molecular
growth conditioned on a 3D protein pocket**: at each construction step the
policy selects applicable chemical actions — an initial seed fragment and
pocket anchor, an eligible reactive handle, a certified reaction, a
compatible catalog synthon, a categorical torsion/rotamer action, or an
explicit termination — and the environment applies the valid reaction,
places the new fragment, refines its geometry, and updates the molecular
state. The system targets both a 3D ligand candidate and an auditable
synthetic construction history.

## 2. Formulation: MDP, construction DAG, and GFlowNet

These three concepts are complementary and are used precisely throughout
the project (this distinction is binding, see #45 §2.3):

- **MDP** — the sequential decision problem: state, action, transition,
  termination, reward.
- **Construction DAG** — the directed acyclic graph of valid molecular
  construction states and transitions induced by the MDP's growth rules.
- **GFlowNet** — learns a forward sampling policy over that construction
  state space, aiming to sample terminal objects in proportion to their
  positive reward under the fixed conditioning context (the target pocket).
- **Trajectory Balance (TB)** — the training objective enforcing flow
  consistency along sampled trajectories.

A molecular graph can have multiple valid construction histories. Uniqueness
of a derivation-tree parent does not by itself prove that molecule-level
samples follow P(x) ∝ R(x); derivation multiplicity must be handled
explicitly (canonical derivations, a mathematically justified correction, or
a graph-state / backward-policy formulation). This design item — issue #38 —
**is resolved in V2**: states are order-quotiented discrete assembly
objects, the backward policy is the uniform leaf deletion policy, the STOP
backward term is zero by proof, and the identities are verified on an
enumerable toy DAG through the real Trajectory Balance engine
(`docs/architecture/MDP_STATE_SPACE.md`, decision D-V2-009).

## 3. Architecture overview

The target V2 system (issue numbers mark ownership; see
`docs/architecture/ARCHITECTURE.md` for the full layered map):

| Concern | Canonical home | Issue |
|---|---|---|
| Frozen state tensor schemas & PyG batch contracts | `syntree.core.schema` | #6 |
| Non-singular equivariant math kernels | `syntree.models.equivariant` | #7, #8 |
| Continuous cosine cutoff envelope for all spatial RBFs | `syntree.models.cutoff` | #9 |
| Fail-closed pure-FP32 precision engine | `syntree.utils` | #10 |
| Bipartite SE(3) backbone over pocket ∪ ligand (`BipartitePaiNN`) | `syntree.models` | #21 |
| Agency-enabled handle-selection head (`P(handle \| s)`) | `syntree.models` | #22 |
| Two-layer cross-attention decoder + hierarchical policy heads | `syntree.models.policy` | #23, #24 |
| Certified 32-reaction taxonomy (9 coarse families) | `syntree.chemistry` | #11 |
| Enamine REAL synthon catalog + `pharm3d` descriptors | `syntree.chemistry` | #12, #40 |
| MMFF94 relaxation, soft-core LJ, torsion refinement | `syntree.chemistry` | #13, #14, #42 |
| Multi-source data pipeline, clustered splits, exact replay, sharding | `syntree.data` | #16–#20 |
| Supervised trainer, curriculum, transfer | `syntree.engine` | #26–#28 |
| GFlowNet Trajectory-Balance engine + stationary reward | `syntree.engine` | #29, #30, #38 |
| Test suites A (capability) and B (invariants) | `tests/` | #32, #33 |
| Retrosynthesis & comparative SBDD evaluation | `scripts/` | #34, #35, #44 |

Binding architectural invariants (details in the architecture document):

- **Strict separation of chemistry from geometry.** Invariant chemical
  features (`[N, d]`), Cartesian positions (`[N, 3]`), atomic numbers, and
  charges are separate tensors. The legacy "Franken-tensor" that glued 64
  chemical descriptors to 3 raw coordinates is a hard error everywhere.
- **One equivariant backbone** over the union of pocket and ligand atoms
  (pocket–pocket, ligand–ligand, and pocket–ligand interactions), never a
  pocket-only encoder blind to the growing ligand.
- **Non-singular geometry operators.** All norms and directional unit
  vectors use the ε-smoothed formulation below; no raw `torch.norm`
  divisions by near-zero distances anywhere in the model path.
- **One authoritative representation chain into the policy.** The
  cross-attention decoder (#23) refines a per-graph handle query built
  solely from the backbone's pooled contexts and the handle head's pooled
  surface (no second pocket/handle encoder) and projects no spatial
  distances itself — every geometric quantity it consumes already passed
  through the enveloped radial basis upstream.
- **Pure FP32 on Tesla T4 only** (see §7).

## 4. Mathematical foundations

**Non-singular norms and bounded unit vectors (#7, implemented).** All
geometric norms use the ε-smoothed form

```
||x||_eps = sqrt( sum_k x_k^2 + eps^2 ) - eps,        eps = 1e-5
grad_x ||x||_eps = x / sqrt( sum_k x_k^2 + eps^2 )
```

which equals `0.0` with a strictly-zero, bounded gradient at `x = 0` — the
raw L2 norm is undefined there and its gradient diverges, which corrupted
training in the previous system at coincident / clashing coordinates.
Directional unit vectors are computed as `v / sqrt(Σv² + ε²)`, which is
well-defined for every input (the zero vector maps to the zero direction)
and strictly bounded (`‖unit‖ < 1`). See `syntree.models.equivariant`.

**Equivariant message passing (PaiNN).** Scalar features `s ∈ R^d` and
vector features `v ∈ R^{3×d}` are propagated with the published PaiNN
update scheme (Schütt, Unke & Gastegger, ICML 2021): radial filter banks
and shared scalar transforms gate scalar and vector messages along
non-singular directions, and per-atom mixing blocks couple the channels
through the invariant quantities `‖Vv‖` and `⟨Uv, Vv⟩`. Scalar outputs are
provably invariant and vector outputs equivariant under rigid motions; both
properties are pinned by tests with random rotations.

**Continuous cosine cutoff envelope.** Every spatial distance RBF
projection is multiplied by the `C¹` cosine envelope
`f_cut(d) = 0.5 (cos(π d / r_cut) + 1)` that decays to zero at the cutoff
(`syntree.models.cutoff`, integrated inside `GaussianRadialBasis` — the
one authoritative producer for the backbone's edge distances and the
handle head's centroid distances alike, issue #9 / decision D-V2-011).
Beyond the cutoff the radial features are *identically* zero (bitwise),
killing the legacy failure mode where clamped raw Gaussians handed every
far spectator atom an identical non-zero bias feature; the derivative
approaches zero continuously at the boundary, so features taper smoothly
instead of switching abruptly.

**Planned, not yet implemented:** analytic Amos-ratio autograd for the von
Mises circular torsion objective (#8); the fail-closed FP32 precision
engine (#10). These are listed for design completeness — no claim of
existence is made until their issues close with evidence.

## 5. Chemistry: reactions, catalog, geometry

- **Reaction taxonomy (#11):** 32 certified reaction transformations
  grouped into 9 coarse policy families, each with SMARTS templates, handle
  requirements, atom/bond mapping, stereochemical behaviour, validity
  constraints, product sanitization, failure behaviour, and family
  membership. A SMARTS string executing is not certification.
- **Synthon catalog (#12, #40):** a curated 50,000-building-block Enamine
  REAL subset with `pharm3d` invariant descriptors (2D, 3D shape,
  electrostatic, dipole, handle-direction information) projected through a
  versioned orthonormal QR projection, persisted as an immutable HDF5
  artifact with a manifest. Reaction-aware chemoselectivity screening
  (competing handles, orthogonal protection) is part of curation — basic
  MW/Fsp3 filtering is explicitly insufficient.
- **Geometry (#13, #14, #42):** scaffold-constrained MMFF94 relaxation with
  resonance-locked geometry; soft-core Lennard-Jones pocket contact and
  volume-floor terms; 36-bin categorical torsion actions over [-π, π) with
  environment-side bounded (±5°) pocket-conditioned refinement inside the
  selected bin — refinement is a geometry operation attached to the
  categorical action, never an unnormalized continuous density inside the
  Trajectory-Balance log-probability.

## 6. Data, leakage prevention, and integrity

Sources (verified URLs, checksums, and dates in
`docs/data/RAW_SOURCE_PROVENANCE.yaml`): CrossDocked2020 Pocket10, PDBbind
refined (2020), and BindingMOAD. Pockets are standardized on the
co-crystallized ligand centroid (10 Å radius) with a documented policy for
waters, salts, additives, and metals. Splits are produced by MMseqs2
clustering at 30% sequence identity / 80% coverage across all protein
chains from all sources; complexes sharing homologous chains form connected
components that are allocated whole to 80/10/10 splits — **no homologous
component crosses a split**. Every supervised trajectory step is accepted
only under exact forward reaction replay (the selected reaction + catalog
synthon must reproduce the observed product graph exactly, with rejection
reasons recorded). Datasets are persisted as compressed shards with
SHA-256 manifests. Counts always come from manifests, never from
estimates.

## 7. Hardware and precision policy

The only supported GPU production environments are **Google Colab with an
NVIDIA Tesla T4** and **Kaggle with an NVIDIA Tesla T4**. The production
path is **pure FP32**: no FP16, no BF16, no autocast, no `GradScaler`, no
silent precision fallback. Architecture dimensions are frozen in
version-controlled configuration (no VRAM-probing auto-scaling); memory is
engineered explicitly (pocket/ligand atom caps, micro-batch ×
gradient-accumulation settings, activation checkpointing) and measured on
real T4 hardware before any throughput claim is made (#43). GPU-dependent
verification runs only on authorized T4 runtimes (#36); a CPU development
sandbox can never establish T4 behaviour.

## 8. Training stages

1. **Stage 1 — supervised behavioral cloning** on forward-replay-validated
   trajectories (#26): discrete action losses, circular torsion loss,
   Approach-C quantile curriculum with 9 difficulty levels (#27), 2D
   reaction-grammar pretraining → 3D fine-tuning transfer (#28),
   coordinate/dihedral augmentation on the training split only (#20).
2. **Stage 2 — GFlowNet Trajectory Balance** (#29, #38): the forward
   probability includes the seed, STOP, handle, reaction, synthon, stereo,
   and torsion actions; backward semantics are resolved by #38 (see §2).
   PPO/DPO are not production alternatives in V2. Implemented (#29
   closed): the seed-conditioned partition function `log Z_θ(P, s_0)`
   (a 2-layer MLP over the pooled pocket and seed latents, trained at
   `lr_Z = 1e-2`), the on-policy rollout driver
   (`syntree/engine/rollout.py` — the policy drives the assembly
   environment with its own categorical laws), the TB trainer
   (`GFlowNetTrainer`), and the canonical CLI
   `scripts/train_stage2.py` (#31) which initializes from Stage-1
   weights and resumes interrupted runs exactly.
3. **Reward (#30):** a stationary, reproducible, positive function of the
   terminal object and the fixed conditioning context only. Batch-relative
   diversity terms inside R(x) are forbidden; population diversity is
   managed outside the terminal reward. Hard chemical validity is never
   tradable against a favourable score.

## 9. Evaluation

Chemical validity, PoseBusters validity, docking/contact metrics,
diversity, and Fsp3 — plus two *separately reported* retrosynthesis tracks
(#34, #44): **Track A (generator-stock-aligned)** exports the exact
versioned Enamine subset the generator used as the planner stock; **Track B
(independent)** uses a separately sourced external stock. Solve rates are
never merged, and a missing building block is never treated as proof a
route is impossible. Comparative SBDD benchmarking against frozen held-out
targets (#35) reports statistical uncertainty and paired comparisons where
justified. Internal forward replay verifies internal construction
consistency only — it does not prove an independent planner can rediscover
a route. The measured two-track evidence run (2026-11-05, on the
194-record / 30-unique-structure #18 construction corpus) lives in
`docs/eval/` with full provenance: stock-aligned solve rate 0.267 vs
independent external-stock 0.733, every failure classified as a stock
mismatch (never a planner failure).

## 10. Installation (development)

```bash
git clone --branch 3d-synV2 https://github.com/Vtheonly/3d-syntree.git
cd 3d-syntree
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build on an authorized T4 runtime
pip install -e ".[test]"
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1, PyTorch Geometric ≥ 2.4.

## 11. Execution

There is **no end-to-end CLI yet** — the training/generation entry points
arrive with their owning issues (#26, #31). What can be executed today is
the library surface and its test suite:

```bash
python -m pytest tests/ -q          # full local (CPU) verification suite
```

GPU-dependent tests and training run only on authorized Colab/Kaggle T4
runtimes via the SSH workflow (#36) and are never claimed from a CPU
sandbox.

## 12. Reproducibility

Version-controlled frozen configurations; deterministic, checksum-verified
shards and artifacts; checkpoints persisting model / optimizer / scheduler
/ step / RNG / sampler-curriculum / data-progress / GFlowNet-partition
state atomically with integrity checks and session-safe resume (#43);
artifact synchronization to Hugging Face with checksums (#37). Credentials
are read from environment variables only (`HF_TOKEN`, `GITHUB_TOKEN`) and
never hardcoded or logged.

## 13. Repository structure

```
src/syntree/            # the V2 package
  core/                 # frozen state tensor contract & batch path (#6)
  models/               # equivariant math kernels (#7), cosine cutoff
                        # envelope (#9), union-graph backbone (#21),
                        # handle-selection head (#22), cross-attention
                        # decoder (#23), von Mises torsion mathematics
                        # (#8); remaining policy heads (later issues)
  utils/                # fail-closed FP32 hardware precision engine +
                        # standard logit masks (#10)
  chemistry/            # production pocket physics: flat-bottom soft-core
                        # LJ clash + cavity/heavy-atom volume floors (#14);
                        # reaction registry/catalog arrive with #11/#12
  engine/               # GFlowNet TB engine + rollout driver + trainer
                        # geometry / #38 semantics / #42+#13 terminal
                        # oracle (production clash objective per #14) /
                        # #30 stationary reward with #14 volume floors /
                        # #29 TB core)
docs/
  architecture/         # global architecture specification + MDP state space
  data/                 # verified raw-source provenance
  agent/                # STATUS, WORK_LOG, DECISIONS, DEPENDENCIES, HANDOFF
tests/                  # pytest suites (Suite A/B grow with #32/#33)
```

Agent-facing documentation (`docs/agent/`) is part of the deliverable, not
an afterthought: the project is built by autonomous engineering sessions
that must leave a precise, verified resumption point after every session.

## 14. Implementation status

Honest, evidence-linked status (synchronized with the GitHub issue
tracker; last updated 2026-11-02, session S-V2-09):

| Capability | Status | Evidence |
|---|---|---|
| Branch clean-room reset (purge of all legacy content) | **Done, verified, closed** | commit `075f86d` — branch tree verified empty on origin; foundation `0f5d74a` (#5) |
| Frozen state tensor contract & PyG batch path | **Done, verified, closed** | `syntree/core/schema.py` @ `0001087`, 93/93 tests (#6) |
| Non-singular PaiNN norms & bounded unit vectors | **Done, verified, closed** | `syntree/models/equivariant.py` @ `00ec653`, 45/45 tests (#7) |
| Continuous cosine cutoff envelope for all spatial RBF projections | **Done, verified, closed** | `syntree/models/cutoff.py` @ `bf735a6`, 46/46 tests — bitwise-zero features beyond the cutoff (`d = r_cut + 1e-4` probe), derivative → 0 continuously at the boundary, integrated inside `GaussianRadialBasis` for the backbone and handle-head bases (#9, D-V2-011) |
| Bipartite union-graph backbone (`BipartitePaiNN`) | **Done, verified, closed** | `syntree/models/bipartite.py` @ `e76d955`, 69/69 tests — SE(3) invariance/equivariance ≤3e-13 over 36 rigid-motion trials, clean pocket-only N_l=0, 5,999,104-param frozen profile (#21) |
| MDP / derivation-DAG semantics + TB backward policy | **Done, verified, closed** | `docs/architecture/MDP_STATE_SPACE.md` + `syntree/engine/` @ `d3d186a`, 50/50 tests — toy-DAG `P(x)=R(x)/Z` residuals < 1e-9 (#38, D-V2-009) |
| Agency-enabled handle-selection head | **Done, verified, closed** | `syntree/models/handle_head.py` @ `433d303` (S-V2-04), 44/44 tests — valid normalized laws on real `AssemblyEnvironment` menus, rigid-motion invariance 8.3e-16 (FP64) / 5.96e-08 (FP32), 224,896-param frozen profile (#22, D-V2-010) |
| Discrete assembly MDP + terminal reward oracle + TB engine (engine layer) | **Implemented, tested locally** — engine layer of the reconciled design; trainer/CLI integration open | `syntree/engine/environment.py` / `rl.py` / `gflownet.py` @ `d3d186a`, 50/50 tests (D-V2-009); #24/#29/#30/#42 remain open for their full scope |
| Two-layer stacked cross-attention decoder | **Done, verified, closed** | `syntree/models/policy.py` @ `dc4921e` (S-V2-06), 58/58 tests — `[B, 1, 256]` × `[B, N, 256]` → `[B, 256]` acceptance shapes, padded atoms masked exactly (bit-identical under pad junk), SE(3) invariance 8.882e-16 (FP64) / 4.768e-07 (FP32), 1,219,328-param frozen profile, no new spatial-RBF site (#23, D-V2-012) |
| Stable von Mises circular-torsion mathematics | **Done, verified, closed** | `syntree/models/torsion_head.py` @ `ae2c615` (S-V2-07), 44/44 tests — `log I_0` matches `scipy.special.i0e` to a measured 6.51e-6 over κ∈[0,100] (< 1e-5 acceptance); analytic Amos' ratio backward finite, ≥ 0, strictly < 1 everywhere (measured deviation vs `I₁/I₀`: 3.30e-2 max, O(κ⁻²) decay); the legacy NaN-leak/series-divergence failure is structurally eliminated (#8, D-V2-013) |
| Fail-closed FP32 hardware precision engine | **Done, verified, closed** | `syntree/utils/hardware.py` @ `5ce4f5e` (S-V2-07), 35/35 tests — simulated Tesla T4 (cc 7.5) forces `precision="fp32"` with no GradScaler/autocast; `mixed_precision="fp16"` requests log a FAIL-CLOSED warning and execute pure FP32; TF32 structurally off on supported hardware; repo-wide AST ban on AMP machinery (#10, D-V2-014) |
| Flat-bottom soft-core LJ clash + cavity/heavy-atom volume floors | **Done, verified, closed** | `syntree/chemistry/conformer.py` + engine wiring @ `d61a7fc` (S-V2-07), 50/50 tests — V(0.5 Å) = 484.85 kcal/mol < 500 for **every** supported element pair (bounded-spike ceiling by construction) vs 1.58×10⁹ for the rigid wall; the #42 torsion-refinement objective and the reward's `S_vol` (0.35 cavity ratio + 12-heavy-atom floors, stationary) consume it (#14, D-V2-015) |
| Certified 32-reaction taxonomy + 9-family coarse mapping + deterministic deprotections | **Done, verified, closed** | `syntree/chemistry/reactions.py` @ `0e296b0` (S-V2-08), 82/82 tests — 100% round-trip forward-replay certification on 48 standardized reagent pairs (16 stereo pairs with CIP retention); `DEFAULT_REACTION_REGISTRY` is the certified taxonomy (the generic interim was superseded and removed) while all 50 pinned MDP/TB contracts stay green (#11, D-V2-016) |
| Junction torsion stages + scaffold-constrained MMFF94 in the static pocket vdW field | **Done, verified, closed** | `syntree/chemistry/conformer.py` (extended) + `finalize_terminal_molecule` opt-in flags @ `472f4df` (S-V2-08), 18/18 tests — amide planarity ≤ 1.0° (restored exactly post-MMFF: MMFF94's amide torsions sit ~3.7–11.6° off planar at their minimum, a measured finding), seed-scaffold RMSD < 0.15 Å, junction bond settles < 0.02 Å of its MMFF equilibrium, pocket static within 0.05 Å behind D-V2-017 opt-in flags whose defaults preserve the legacy behaviour bit-for-bit (#13) |
| Canonical assembly ordering + seed-conditioned decompositions | **Done, verified, closed** | new package `syntree/data/fragmenter.py` @ `864530d` (S-V2-08), 21/21 tests — identical retrosynthetic trajectories under 12–20 random atom renumberings per ligand (the #15 acceptance criterion); the pinned seed rule (max ring system → highest degree → canonical SMILES) and periphery→core peeling with permanent rejection of atom-isolating cuts (#15) |
| Decoupled hierarchical policy heads + discrete 36-bin SO(2) torsion space | **Done, verified, closed** | `syntree/models/policy.py` + `syntree/models/torsion_head.py` @ `74133c8` (S-V2-09), 100/100 tests — STOP-isolation pinned by forward hooks (no continuation head executes for terminating rows); every action component a pure dimensionless categorical whose logs compose in TB (κ/μ-invariance pinned); the exact family→reaction factorization; the tied 50k catalog embedding table; the asymmetric resonance-lock torsion gating; the three pre-V2 policy bugs pinned dead (#24, D-V2-018) |
| Complete policy assembly (`SynTreePolicy`) + parameter audit | **Implemented, CPU-verified** — the CUDA forward-pass criterion awaits the authorized T4 runtime | `syntree/models/policy.py` @ `84fd3e7` (S-V2-09), 48/48 tests — one registered module with the full frozen stack (checkpointed backbone, handle head, decoder, #24 heads, #39 seed/anchor component); 10,880,719 params = 43.5 MB FP32 < the 50 MB bound (D-V2-019); full-profile forward zero-NaN/Inf on CPU; the tied table shared and counted once (#25) |
| Learnable seed-fragment & pocket-anchor selection (policy side) | **Implemented** (via #25): deterministic rotation-equivariant anchor generator, factorized `P(anchor\|pocket)·P(seed\|anchor,pocket)` head, masks, TB log-prob contribution | `syntree/models/policy.py` @ `84fd3e7`, pinned in `tests/test_policy_assembly.py` — generator determinism/equivariance, subpocket separation, law validity, gradients through the checkpointed backbone into the tied table; the environment-side candidate-state rewiring and supervised seed labels remain #39's open scope (blocked on #12) |
| Reaction-aware chemoselectivity & competing-handle curation screen | **Done, verified, closed** | `syntree/chemistry/chemoselectivity.py` @ `044adca` (S-V2-09), 111/111 tests — the free-group vocabulary with orthogonal protection passing by construction; the per-class exclusion matrix (self-reactive pairs, organometallic quenching, competing handles/electrophiles/SNAr positions, catalyst poisons); the asymmetric 2,4-dihalo-1,3-azine exemption; the catalog manifest with per-rule impact counts/percentages (#40, D-V2-020) |
| Enamine REAL 50k synthon catalog + `pharm3d` d256 embeddings | **Artifacts built, verified; build code re-landing pending** | S-V2-10 full-scale build: 290,648 source records screened (sha256-pinned mirror) → 152,489 certified pool (registry handle detection) → #40 chemoselectivity screen (39,150 exclusions) → 50,000 selected (34,984 saturated / 15,016 aromatic; D-V2-023) → 2,137-dim `pharm3d` → QR-JL projection (seed 2026) → `[50000, 256]` all-finite; mean Fsp3 0.7151, mean rotatable 1.5361 — all four #12 acceptance criteria independently re-verified. Artifact: `enamine_real_50k_pharm3d_d256.h5` (67,797,608 B, sha256 `8b8c13a0…`, immutable) on HF `JJKK1212/3d-syntree-v2-catalog`; evidence at `docs/data/catalog/`. The build **code** was lost to a sandbox recycle before commit — #12 stays open until `synthon_encoder.py` + the driver + tests are re-landed against the artifact pins |
| Hashed inverted synthon index | **Done, verified, closed** | `syntree/chemistry/synthon_index.py` + `scripts/build_synthon_index.py` (#41, S-V2-12): full-scale benchmark — exhaustive self-retrieval 68,417/0 lost, brute-force equivalence 2,000/0 mismatches, 888.7× mean candidate reduction, 1.34 µs lookups, byte-deterministic rebuilds |
| Multi-source preprocessing pipeline (CrossDocked / PDBbind / BindingMOAD → uniform pocket+ligand artifacts + provenance manifest) | **Done, verified, closed** | `syntree/data/preprocessing.py` + `scripts/preprocess_multidataset.py` @ `fccd670` (S-V2-10), 28/28 tests — full-scale run over 204,005 complexes with zero unhandled exceptions (170,954 accepted / 33,051 exact-code rejections); CONECT+template ligand reconstruction (`rdDetermineBonds` evaluated and rejected on RDKit 2026.09.1); the MOAD annotations mirror superseded by the final binding-data CSV (measured 13% resolution of family-representative ids); suite 942/942 (#16) |
| Clustered splits (MMseqs2 30%) | **Done, verified, closed** | `scripts/split_clusters.py` + `scripts/run_mmseqs_split.sh` (#17, S-V2-11): full-scale 204,005 records → 22,387 clusters / 14,958 components / **zero leakage**, train 136,764 / val 17,096 / test 17,095 — evidence at `docs/data/splits/` |
| Trajectory extraction + exact forward replay | **Done, verified, closed** | `syntree/data/trajectories.py` + `scripts/build_trajectories.py` (#18, S-V2-12/13): full-scale run over all 170,955 accepted complexes — **194 accepted trajectories** (train 111 / val 50 / test 33), every step exact-replay-verified; closed failure vocabulary over 170,761 rejects (97.78% `no_matched_seed` — the honest certified-BB reconstructibility census, `docs/data/trajectories/README.md`) |
| Compressed sharding engine | **Done, verified, closed** | `syntree/data/sharded.py` + `scripts/shard_and_upload.py` (#19, S-V2-13): byte-bounded greedy packing with fail-closed size verification, SHA-256 manifests, HF streaming loader, shard-aware unbiased shuffle sampler |
| Coordinate jitter & exposure-bias augmentation | **Done, verified, closed** | `syntree/data/augmentation.py` (#20): σ=0.15 Å ligand-only Gaussian jitter + rigid ±5° non-junction dihedral perturbation (Gauss-Seidel exact targets, fail-closed bound verification), train-split only; **measured finding (S-V2-14)**: naive jitter inverts stereocentres (~4% on the fixture) → stereo-signature rejection resampling guarantees no inversion |
| Supervised trainer (Profile A: pure FP32, B=12×6, pocket cap 700) | **Implemented, CPU-verified** — the two T4 acceptance criteria (monotonic 2-epoch descent on T4; peak reserved VRAM < 9.5 GB) await the authorized #36 runtime | `syntree/engine/trainer.py` + `syntree/data/supervised.py` + `configs/train_colab_t4.json` + `scripts/train_stage1.py` (#26): crystal-intermediate supervised states with fail-closed record/artifact cross-checks, the full Stage-1 loss stack (seed/stop/handle/family/reaction/synthon/parity CE + circular von Mises torsion loss), fail-closed divergence (no nan-shield/batch-skipping), separate host-RAM/GPU-VRAM telemetry, atomic SHA-256 checkpoints with exact continuation state, time-budget graceful stop; end-to-end CLI smoke green |
| Approach-C quantile curriculum (9 levels / 6 stages / 0.03 floor) | **Done, verified, closed** | `syntree/engine/curriculum.py` + trainer/CLI integration (#27, S-V2-15): per-item difficulty over the five issue-named axes (pocket atoms, rotatable bonds, rings, pocket charge magnitude, handle rarity — D-V2-037), 9 quantile-balanced bins, floor-exact 6-stage distribution (D-V2-038), exactly-N deterministic epoch draws, milestone machine (val improvement ≥ 0.005, patience-2 fallback), checkpointable sampler state in the trainer payload; canonical `--curriculum` CLI flag (D-V2-039); 40/40 tests incl. the CLI smoke — both #27 acceptance criteria verified |
| 2D → 3D transfer pretraining | Not started — code deps satisfied (#11/#12/#27 closed); **externally blocked on licensed reaction data** (owner decision) | #28 |
| Test suites A/B | Suite A **done, verified, closed** (293/293); Suite B **implemented + CPU-verified** (47 passed / 1 skipped — the skip is the T4 memory measurement, gated on the authorized #36 session) | #32 (`tests/suite_a/`); #33 (`tests/suite_b/`) |
| Retrosynthesis & comparative evaluation stack | **#34/#44 done, verified, closed with real-planner evidence**; #35 harness verified on the frozen 33-target test manifest — baseline structures (TargetDiff/DiffSBDD/SyntheMol) await operator export | `syntree/engine/evaluator.py` + `syntree/engine/comparative.py` + `scripts/run_benchmarks.py` / `export_aizynth_stock.py` / `prepare_aizynth_tracks.py` / `make_frozen_targets.py` / `run_comparative_benchmark.py` (#34/#44/#35): the fail-closed generated-SDF corpus parser, the isolated `aizynthcli` subprocess boundary (public USPTO policy), the two never-merged stock tracks (generator-aligned Enamine 50k with catalog-ID/checksum tie + independent external ZINC 17,422,831), stock-mismatch vs planner-failure classification (real-planner-tested), seeded bootstrap CIs, paired-over-common-targets comparisons, `evaluation_report.json` / `comparative_benchmark.json` + markdown; evidence run on the 194-record / 30-unique-structure #18 corpus: validity 100%, PoseBusters 194/194, Track A solve 0.267 (22 stock mismatches), Track B solve 0.733 — see `docs/eval/` |
| T4 GPU execution / training | **Prepared — awaiting operator authentication** | #36: `notebooks/run_3d_syntree.ipynb` (fail-closed T4 preflight, sshd setup, exact-commit checkout, separate RAM/VRAM preflight, canonical training launch) + `docs/ops/colab_ssh_workflow.md` (the operator's `colab ssh` CLI flow + the #45 §29.1 authorization handoff). No connection has been established and none is claimed; #43 profiling/resume rides the same session |
| HF artifact synchronization | **Partially exercised** — one-time preservation upload of the #12 catalog dataset to HF (`JJKK1212/3d-syntree-v2-catalog`, private) under the owner's directive; the automated sync pipeline is #37's gated scope | #37 |

No training has been run, no benchmark numbers exist, and no scientific
claim beyond the unit-tested contracts above is implied anywhere in this
repository.

## 15. Limitations

- The system is an early-stage rebuild: the foundation, the backbone, the
  handle-selection head, the cross-attention decoder, the von Mises
  torsion mathematics, the FP32 hardware gate, the soft-core clash and
  volume-floor physics, the reconciled GFlowNet engine layer, the certified
  32-reaction taxonomy, the #13 junction-torsion/MMFF stages, the
  canonical decomposer, the decoupled hierarchical policy heads with the
  discrete torsion space, the complete `SynTreePolicy` assembly with the
  seed/anchor selection component (CPU-verified; CUDA pending), the
  chemoselectivity curation screen, and the multi-source preprocessing
  pipeline are implemented; the #12 catalog **artifacts** are built and
  verified (its build code is being re-landed); everything else in
  this README describes the *target* design, clearly marked as such.
- The MDP/derivation-DAG semantics and the TB backward term are resolved
  (#38 closed with proofs and toy-DAG verification); the *trainer*
  integration of Stage 2 is implemented and verified (#29/#30/#31 closed:
  the logZ head + rollout driver + TB trainer, the docking-backend
  injection point of the stationary reward, and the Stage-2 CLI with
  exact resume).
- GPU behaviour (T4 memory profile, throughput) is unmeasured by design —
  it requires the authorized remote workflow.
- Data licensing and catalog access (Enamine REAL terms, PDBbind
  registration-gated distribution) constrain what can be reproduced
  verbatim; provenance records document the verified open mirrors used.

## 16. License

MIT — see [LICENSE](LICENSE).
