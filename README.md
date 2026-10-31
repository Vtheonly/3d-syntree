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
a graph-state / backward-policy formulation). This is a tracked,
*unresolved* design item — issue #38 — and no claim about the sampling
guarantee is made until it is proven and tested.

## 3. Architecture overview

The target V2 system (issue numbers mark ownership; see
`docs/architecture/ARCHITECTURE.md` for the full layered map):

| Concern | Canonical home | Issue |
|---|---|---|
| Frozen state tensor schemas & PyG batch contracts | `syntree.core.schema` | #6 |
| Non-singular equivariant math kernels | `syntree.models.equivariant` | #7, #8, #9 |
| Fail-closed pure-FP32 precision engine | `syntree.utils` | #10 |
| Bipartite SE(3) backbone over pocket ∪ ligand (`BipartitePaiNN`) | `syntree.models` | #21 |
| Cross-attention decoder + hierarchical policy heads | `syntree.models` | #22–#24 |
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

**Planned, not yet implemented:** analytic Amos-ratio autograd for the von
Mises circular torsion objective (#8); continuous cosine cutoff envelopes
for cross-attention radial bases (#9); the fail-closed FP32 precision engine
(#10). These are listed for design completeness — no claim of existence is
made until their issues close with evidence.

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
   PPO/DPO are not production alternatives in V2.
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
a route.

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
  models/               # equivariant math kernels (#7+), backbone & heads (later issues)
docs/
  architecture/         # global architecture specification
  data/                 # verified raw-source provenance
  agent/                # STATUS, WORK_LOG, DECISIONS, DEPENDENCIES, HANDOFF
tests/                  # pytest suites (Suite A/B grow with #32/#33)
```

Agent-facing documentation (`docs/agent/`) is part of the deliverable, not
an afterthought: the project is built by autonomous engineering sessions
that must leave a precise, verified resumption point after every session.

## 14. Implementation status

Honest, evidence-linked status (synchronized with the GitHub issue
tracker; last updated 2026-11-01):

| Capability | Status | Evidence |
|---|---|---|
| Branch clean-room reset (purge of all legacy content) | **Done, verified** | commit `075f86d` — branch tree verified empty on origin |
| Frozen state tensor contract & PyG batch path | **Implemented + locally tested** | `syntree/core/schema.py`, `tests/test_tensor_contracts.py` (#6) |
| Non-singular PaiNN norms & bounded unit vectors | **Implemented + locally tested** | `syntree/models/equivariant.py`, `tests/test_mathematical_singularities.py` (#7) |
| Reaction registry, catalog, chemistry engines | Not started | #11–#15, #40–#42 |
| Data pipeline, splits, replay, sharding | Not started | #16–#20 |
| Backbone, decoder, policy heads, policy assembly | Not started | #21–#25, #39 |
| Trainers, curriculum, transfer | Not started | #26–#28 |
| GFlowNet TB engine, reward, CLI | Not started | #29–#31, #38 |
| Test suites A/B (full scope) | Not started (only issue-scoped tests exist) | #32, #33 |
| Evaluation pipelines | Not started | #34, #35, #44 |
| T4 GPU execution / training | **Never attempted** — requires authorized runtime | #36, #43 |
| HF artifact synchronization | **Never attempted** | #37 |

No training has been run, no benchmark numbers exist, and no scientific
claim beyond the unit-tested contracts above is implied anywhere in this
repository.

## 15. Limitations

- The system is an early-stage rebuild: only the foundation issues are
  implemented; everything else in this README describes the *target*
  design, clearly marked as such.
- The MDP/derivation-DAG semantics and the TB backward term are unresolved
  (#38); until then no proportional-sampling claim is made.
- GPU behaviour (T4 memory profile, throughput) is unmeasured by design —
  it requires the authorized remote workflow.
- Data licensing and catalog access (Enamine REAL terms, PDBbind
  registration-gated distribution) constrain what can be reproduced
  verbatim; provenance records document the verified open mirrors used.

## 16. License

MIT — see [LICENSE](LICENSE).
