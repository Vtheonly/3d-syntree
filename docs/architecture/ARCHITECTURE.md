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
> Last updated: 2026-11-02 (sessions S-V2-02: #5, #6, #7, #21; S-V2-03:
> #38 + the reconciled engine layer; S-V2-04: #22; S-V2-05: #9; S-V2-06:
> #23; S-V2-07: #8, #10, #14).

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
| Handle selection head | `syntree/models/handle_head.py` | #22 |
| Cross-attention decoder | `syntree/models/policy.py` | #23 |
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
`edge_dist` only. `GaussianRadialBasis` emits the 20-Gaussian expansion
multiplied by the continuous cosine cutoff envelope integrated inside
the producer (#9 MATH-03 / decision D-V2-011 — see §3.12). Every layer is
wrapped in non-reentrant
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

### 3.6 Stage-2 algorithm `[VERIFIED — #38 semantics + #45 §12.3, D-V2-009]`

Stage 2 is **GFlowNet Trajectory Balance only**; PPO/DPO are not V2
production alternatives. The forward probability must include the seed,
STOP, handle, reaction, synthon, stereo, and torsion actions belonging to
the trajectory. The backward term is the **Uniform Leaf Deletion Policy**
`P_B(s'|s) = 1/|RemovableLeafSynthons(s)|` over the order-quotiented
discrete assembly states; the STOP step's backward term is zero *by proof*
(absorbing uniqueness). The full specification, proofs, and the enumerable
toy-DAG verification live in
`docs/architecture/MDP_STATE_SPACE.md` (implemented in
`syntree/engine/environment.py` + `syntree/engine/gflownet.py`).

### 3.7 Reward stationarity `[VERIFIED core — #30 contract implemented, D-V2-009]`

Terminal TB reward must be a **stationary, reproducible** positive function
of the terminal object + fixed conditioning context. Batch-relative
diversity terms inside R(x) are forbidden; population diversity lives in
sampling / replay / archive monitoring outside the per-trajectory reward.
The reconciled engine implements exactly this:
`syntree/engine/rl.py::ThreeDReward.compute(relaxed_mol)` accepts no batch
parameter at all (structurally stationary, tested across batch
compositions), with a hard floor `R >= 1e-4`. Full closure of #30 still
awaits its remaining deps (production soft-core #14, TB engine #29).

### 3.8 Data integrity & leakage prevention `[TARGET — #16–#19]`

Full multi-source ingestion (CrossDocked2020 Pocket10 with the RMSD
criterion, PDBbind refined, BindingMOAD), 10 Å pocket standardization
around the co-crystallized ligand centroid with a documented
waters/salts/additives/metals policy, MMseqs2 30%/80% cluster-connected
splits with whole-component allocation (no component crosses splits),
exact forward-replay validation of every supervised step, SHA-256-checked
compressed shards. Provenance of raw sources is pinned in
`docs/data/RAW_SOURCE_PROVENANCE.yaml`.

### 3.9 Torsion handling `[VERIFIED two-level contract — #24/#42 re-scoped, D-V2-009]`

36-bin categorical torsion action over [-π, π); the bin selects a coarse
rotamer region. The reconciled **two-level** split: during rollout the
environment realises the exact bin centres
`φ_k = -π + (k-½)·2π/36` deterministically (resonance-locked amide/ester
junctions snap planar trans), while the bounded refinement (±5°,
coarse-scan-seeded bracketed Brent) and the scaffold-constrained MMFF94
relaxation run **only at the terminal STOP as the reward oracle** — a
deterministic geometry operation attached to the categorical action with no
unnormalized continuous density inside the TB log-probability
(`docs/architecture/MDP_STATE_SPACE.md` §1).

### 3.10 Reproducibility & artifacts `[TARGET — #37, #43]`

Checkpoints persist model/optimizer/scheduler/global step/epoch/RNG
states/sampler-curriculum/data-progress/GFlowNet `log_Z` atomically with
integrity checks; session-safe resume proven by a real cross-session T4
continuation. Artifacts (datasets, manifests, catalog HDF5, index,
checkpoints, logs) sync to Hugging Face Hub with checksums; credentials
only via environment variables.

### 3.11 Handle-selection head `[VERIFIED — #22, 2026-11-01]`

`syntree/models/handle_head.py::HandleSelectionHead` implements the
agency-enabled handle action `P(handle_t | s_t)` (legacy "Vulnerability 3":
the pre-V2 environment hardcoded `handles[0]`, leaving the agent unable to
choose *where* to grow). It consumes the `BackboneOutput` of the single
authoritative encoder — never a second pocket encoder — plus the flattened
active-handle pool: invariant descriptors `[H, 64]` (the frozen schema
width; legacy 67-D Franken rows fail closed), reacting-atom positions
`[H, 3]` supplied separately from chemistry (#45 §7.1), graph assignment
`[H]`, and ligand-sub-graph attachment indices `[H]` that map 1:1 onto the
`AssemblyEnvironment.active_handles()` menu rows.

Per handle, the invariant embedding combines the 64-D descriptor MLP, the
backbone scalar of the attachment atom, the cosine-enveloped (#9 /
D-V2-011) Gaussian RBF of the handle's distance to its graph's pocket
centroid, and
the SE(3)-invariant directional features `<v_atom, x_handle − centroid>`
built from the *equivariant* backbone vector channel — so the head is
genuinely 3D-aware ("open cavity vs wall") while every scalar it produces
is rigid-motion invariant. A per-graph query (pooled ligand summary +
pocket context) cross-attends over that graph's handle keys (stage 1,
attention weights themselves a valid per-graph distribution), and an
additive Bahdanau-style scorer (query + key + pooled handle context)
produces the logits (stage 2). The final law is a per-graph `log_softmax`
over **exactly the active handle set** (#45 §7.3/§8.4 — normalization over
the actually valid alternatives; H = 1 degenerates to P = 1; graphs without
handles contribute no rows and zero entropy). The output exposes the
auditable tensor/log-probability interface (logits, log_probs, probs,
attention weights, entropies, counts) plus the handle embeddings and
pooled handle context for the #23 decoder. Frozen production profile:
backbone d = 256, head width 128, 20 enveloped RBFs on [0, 12] Å, 224,896
parameters (pinned by test). Verified: valid normalized distributions on
real `AssemblyEnvironment` menus at t = 0 and on grown intermediates
(argmax row ⇒ legal `AssemblyAction.handle_atom_idx`); rigid-motion
invariance worst error 8.3e-16 (float64, 6 random trials) / 5.96e-08
(float32); handle-row/pocket-atom/ligand-node permutation and batch-
composition invariance; gradient connectivity through the checkpointed
backbone; fail-closed validation (dtype/device/shape/index ranges,
cross-field graph consistency, legacy 67-D rejection). Pinned by
`tests/test_handle_selection.py`.

### 3.12 Continuous cosine cutoff envelope `[VERIFIED — #9, 2026-11-01]`

`syntree/models/cutoff.py::continuous_cosine_envelope` implements the
issue-#9 pinned `C¹` envelope `f_cut(d) = 0.5 (cos(π d / r_cut) + 1)`
for `d ≤ r_cut`, `0` beyond, and it is integrated **inside**
`GaussianRadialBasis.forward` — the one authoritative place for *all*
spatial distance RBF projections (decision D-V2-011): the #21 backbone's
edge-distance expansion and the #22 handle head's [0, 12] Å
centroid-distance expansion are enveloped by construction, and the #23
decoder must route its spatial RBF through the same producer instead of
re-implementing an expansion. The envelope is parameter-free (frozen
profiles unchanged) and a radial function, so SE(3)
invariance/equivariance is unaffected. Pinned by
`tests/test_cosine_cutoff.py`: bitwise-zero features for every
`d > r_cut` (including the issue's acceptance probe `d = r_cut + 1e-4` —
legacy "Vulnerability 7", far spectators receiving constant non-zero
bias, is structurally dead); derivative → 0 linearly as `d → r_cut⁻`;
`f_cut(0) = 1` exactly (first channel exactly 1.0 at `d = 0`); analytic
value/derivative closed forms, the reflection identity
`f(d) + f(r_cut − d) = 1`, and the product-rule radial derivative match
in fp64; fail-closed validation (non-tensor / non-floating / NaN-Inf
distances, invalid cutoffs).

### 3.13 Cross-attention decoder `[VERIFIED — #23, 2026-11-01]`

`syntree/models/policy.py::CrossAttentionDecoder` implements the two-layer
stacked cross-attention decoder block (issue #23; pre-reset decision D08
re-adopted as specification by D-V2-002). Each
`CrossAttentionDecoderLayer` is the issue-pinned structure —
`nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1,
batch_first=True)`, feed-forward refinement `Linear(256→512) → SiLU →
Dropout → Linear(512→256)`, and **pre-LayerNorm residual connections on
both sub-layers** — and the stack of two layers (layer 1: cavity bulk and
steric boundaries; layer 2: localized hydrogen-bond and salt-bridge
partners — a representational specialization of the same pinned
mechanism) closes with one final `LayerNorm` (standard pre-LN stack
practice). Frozen profile: d_model 256, 2 layers, 8 heads, FFN 512,
dropout 0.1 — **1,219,328 parameters** (pinned by test).

Two forward interfaces: the **dense padded** interface pinned by the #23
acceptance criteria (`query [B, 1, 256]` + `memory [B, N, 256]` +
`key_padding_mask [B, N]` → `context [B, 256]`; one query per graph is
the stack's contract, the layer primitive supports general `Lq`), and the
V2-native **`forward_state(backbone, handle)`** which consumes the #21
`BackboneOutput` (union scalars as the memory; pooled graph/pocket
contexts) and the #22 handle surface (`pooled_handle_context` — the
attention-weighted aggregate of the authoritative handle embeddings) to
build the per-graph handle query `W_handle(pooled_handle_context) +
W_graph(graph_context) + W_pocket(pocket_context)`. No second pocket
encoder, no second handle encoder; the per-row `handle_embeddings` remain
the #24 heads' surface. The flat union memory is padded per graph
(order-preserving gather) into the dense layout with a bool mask
(`True` = padded slot, zero-filled).

Masking semantics (pinned by tests): attention weights over padded
columns are **exactly 0** in every layer, head, and graph (train and
eval); outputs are **bit-identical** under arbitrary padded-slot values;
and a fully-padded graph fails closed (softmax over an empty key set is
undefined — impossible for valid backbone outputs, which guarantee ≥ 1
pocket atom per graph). The decoder projects **no spatial distances at
all** (decision D-V2-012): every consumed tensor is already invariant and
its geometry already passed through the enveloped `GaussianRadialBasis`
upstream (#21 edge distances; #22 centroid distances + directional
invariants), so D-V2-011's rule is satisfied vacuously and structurally —
any future query-to-memory distance bias must route through
`GaussianRadialBasis`. The output context is therefore SE(3)-invariant
end-to-end through the real backbone + handle head + decoder stack
(measured worst error **8.882e-16** FP64 / **4.768e-07** FP32).
Attention-map footprint at the documented T4 batch target (B = 12,
N ≤ 800, 2 layers × 8 heads, Lq = 1) is ≈ 0.59 MB FP32 **by arithmetic**
— two orders below the issue's 20 MB budget (a calculation, not a T4
measurement; #43 owns measured profiling). Inputs are fail-closed
validated (FP32/FP64-only dtype policy matching module parameters per
D-V2-008, device, shapes, index ranges, cross-field backbone↔handle
graph-count consistency, NaN/Inf fail loudly) and eval-mode forwards are
bit-deterministic (dropout is training-mode only). Pinned by
`tests/test_cross_attention_decoder.py` (58 tests; full suite 405/405
at S-V2-06).

### 3.14 Stable von Mises circular-torsion mathematics `[VERIFIED — #8, 2026-11-02]`

`syntree/models/torsion_head.py` implements the numerically stable
`log I_0(kappa)` primitive as the issue-pinned custom autograd Function
`LogBesselI0` (decision D-V2-013): a float64 two-branch forward — the
22-term power series below `kappa = 12` and the pinned three-term
asymptotic expansion at/above (whose coefficients match the true `I_0`
asymptotic series exactly through `O(kappa^-3)`, which is how the
threshold meets the pinned 1e-5 tolerance at the seam) — selected by
exact `torch.where` gating (no infinity is ever materialised in an
inactive branch), and the **analytic Amos' ratio backward**
`A(kappa) = kappa / (0.5 + sqrt(kappa^2 + 2.25))`, strictly bounded in
`[0, 1)` for every `kappa >= 0` (proof in the module docstring) with
`A(0) = 0` exactly and denominator bounded below by 2.0 — finite by
mathematics, no clamps. Measured: forward vs `scipy.special.i0e`
max error **6.51e-6** over `kappa ∈ [0, 100]`; Amos deviation vs the
trusted ratio `I_1/I_0 = i1e/i0e` max **3.30e-2** at `kappa ≈ 2.5`
(7.9e-4 / 2.0e-4 / 7.4e-5 at 30 / 60 / 100) — an approximation treated
honestly per #45 §8.2, pinned by tests. The legacy "Singularity 2"
(unrolled-series gradient divergence for `kappa > 10` plus NaN leakage
through inactive `torch.where` branches during backward) is
**structurally impossible**: the branch selection lives inside the
untracked forward of the custom Function and the backward is one
closed-form expression. The public surface ships the circular objective
`von_mises_log_prob` / `von_mises_nll` / `von_mises_loss` (verified
against `scipy.stats.vonmises`, numerically normalised over the circle)
that #24 MODEL-04's torsion loss consumes. Fail-closed domain:
finite, floating, `kappa >= 0`. Pinned by
`tests/test_von_mises_torsion.py` (44 tests; full suite 449/449 at
#8 close).

### 3.15 Fail-closed FP32 hardware precision engine `[VERIFIED — #10, 2026-11-02]`

`syntree/utils/hardware.py` implements the enforcement layer of the
constitution's pure-FP32 policy (decision D-V2-014).
`configure_runtime_environment` resolves the device by real probing
(`probe_device` — live name/capability/VRAM per #45 §28.4) or from an
explicit **simulated** `DeviceInfo` (the acceptance-test path, which
never mutates global `torch.backends` state) and applies the strict
gate: every non-`fp32` request token (`fp16`/`bf16`/`amp`/`mixed`) is
rejected with a logged `FAIL-CLOSED` warning — on compute capability
< 8 citing the pre-V2 T4 GradScaler collapse (scale `2^16 → 1.84e-40 →
0.0`, weights → NaN, legacy "Singularity 3") — and the frozen,
auditable `RuntimeConfig` always carries `precision="fp32"`,
`autocast_enabled=False`, `grad_scaler=False`. Devices outside the
supported Tesla T4 scope (cc ≠ 7.5) still enforce pure FP32 (the
mandate is device-independent) *and* record an out-of-scope warning per
#45 §29.2 so training never silently proceeds on unsupported hardware.
TF32 is off by default, requires an explicit measured justification
(`tf32_justified`) and compute capability ≥ 8 — structurally impossible
on the supported T4 (cc 7.5 has no TF32 units). The issue-pinned
logit-mask constants (`LOGIT_MASK_FILL = -1e4`,
`LOGIT_MASK_LEGAL_ABOVE = -1e3` — a 9,000-nat masked/legal separation)
ship with fail-closed `apply_logit_mask`, the pinned scatter
**target-unmasking invariant** as a pure function
(`unmask_target_actions`, bit-identical to the literal in-place snippet)
and strict `validate_target_actions` for supervised targets that were
illegal to begin with (#45 §9.6). The clean-room contains no AMP
machinery at all (D-V2-005 — the legacy `resolve_amp_dtype` /
auto-TF32 / `cudnn.benchmark` path and the older
`scale_model_config()` / `autotune_batch_size()` helpers were never
ported), and a **repo-wide AST regression test** bans
`GradScaler`/`autocast`/`torch.amp` and reduced-precision dtypes from
`src/syntree` — the "completely remove GradScaler" clause enforced as
a permanent gate. Pinned by `tests/test_fp32_enforcement.py` (35 tests;
full suite 484/484 at #10 close).

### 3.16 Flat-bottom soft-core clash + cavity/heavy-atom volume floors `[VERIFIED — #14, 2026-11-02]`

`syntree/chemistry/conformer.py` (the new `syntree/chemistry/` package)
implements the production pocket clash physics and volume floors of
issue #14 (decision D-V2-015). The **issue-pinned flat-bottom soft-core
Lennard-Jones potential** is `V(r) = 0` for `r >= sigma_ij` (the flat
bottom — bitwise zero; neutral geometry is never penalised) and
`V(r) = 4 eps sigma^6 / (r^6 + delta)` for `r < sigma_ij` with
`delta = 0.5 A^6` — bounded by construction. Atom typing: Bondi (1964)
radii (re-adopted from the pre-reset conformer spec, D-V2-002) plus the
D-V2-009 oracle well depths, mixed by Lorentz–Berthelot
(`sigma_ij = r_i + r_j`, `eps_ij = sqrt(eps_i eps_j)`) and **capped by
the bounded-spike ceiling** `eps_ij = min(sqrt(eps_i eps_j),
500 * delta / (4 sigma_ij^6))`, which makes the #14 acceptance bound
hold for *every* supported element pair by construction (measured:
`V(0.5 A) = 484.85 kcal/mol < 500` and `V(0) = 500.00` exactly for the
worst pair, versus `1.58e9 kcal/mol` for the rigid `r^-12` wall — the
shrinking-ligand driver). The potential is differentiable in both
senses: the analytic `dV/dr = -24 eps sigma^6 r^5 / (r^6 + delta)^2`
(finite everywhere, zero at `r = 0`, analytic maximum at
`r* = (5 delta/7)^{1/6}`) and a torch autograd mirror that matches it.
The **role separation** is binding (D-V2-015): this flat-bottom
repulsive-only form is the #42 sub-bin torsion refinement's local
objective (wired into `finalize_terminal_molecule` per the #14
follow-up addendum), while the attractive 12-6 + softened-Coulomb
surrogate (`soft_core_pocket_energy`) keeps its *docking-surrogate* role
for the reward's `S_dock`/`S_contact` — affinity semantics a clash
penalty cannot express, owned by #30/eval. The **volume floors** — the
second anti-shrinking flank — are the deterministic grid
cavity-volume estimator (envelope sphere minus vdW spheres; derived
once at reward construction from the fixed pocket, preserving
stationarity), the cavity filling floor (penalty below
`V_lig / V_poc = 0.35`, bounded `[-2, 0]`), and the explicit heavy-atom
guard (penalty below 12 heavy atoms, weight 2.0) — both wired into
`ThreeDReward.S_vol` with full telemetry, while the compute signature
still carries no batch parameter (batch-composition invariance
re-verified). Pinned by `tests/test_soft_core_chemistry.py` (50 tests;
full suite 534/534 at #14 close, including all 50 engine contracts).

## 4. Repository layout (V2, growing with the roadmap)

```
src/syntree/              # the V2 package (src-layout, decision D-V2-003)
  core/                   # schema: frozen state contract & canonical batch path
  models/                 # equivariant math kernels; union-graph backbone;
                          # policy heads
    equivariant.py        #   safe norms + PaiNN blocks (#7)
    cutoff.py             #   continuous cosine cutoff envelope (#9)
    bipartite.py          #   BipartitePaiNN + radius graph + enveloped RBF
                          #   (#21; envelope integrated per #9/D-V2-011)
    handle_head.py        #   agency-enabled handle selection P(handle|s) (#22)
    policy.py             #   two-layer stacked cross-attention decoder
                          #   (#23; decision D-V2-012 — no new spatial RBF
                          #   site, dense padded memory, exact masking)
    torsion_head.py       #   stable log I_0 + Amos backward + the von Mises
                          #   circular objective (#8; D-V2-013)
  utils/
    hardware.py           #   fail-closed FP32 gate + standard logit masks
                          #   (#10; D-V2-014 — RuntimeConfig, DeviceInfo,
                          #   target-unmasking invariant, AMP AST ban)
  chemistry/
    conformer.py          #   flat-bottom soft-core LJ clash + atom typing +
                          #   cavity/heavy-atom volume floors (#14; D-V2-015)
  engine/                 # the reconciled two-level GFlowNet framework
                          # (D-V2-009; docs/architecture/MDP_STATE_SPACE.md)
    environment.py        #   discrete Assembly MDP + terminal reward oracle
                          #   (#24 geometry, #38 leaves, #42/#13 oracle —
                          #   refinement objective = production clash
                          #   potential per #14/D-V2-015)
    rl.py                 #   stationary multi-objective terminal reward (#30)
                          #   with the #14 volume floors in S_vol
    gflownet.py           #   Trajectory Balance with uniform-leaf P_B (#29)
docs/
  architecture/           # this document + MDP_STATE_SPACE.md (the #38 spec)
  data/RAW_SOURCE_PROVENANCE.yaml
  agent/                  # STATUS, WORK_LOG, DECISIONS, DEPENDENCIES, HANDOFF
tests/                    # pytest suites, file names pinned by owning issues
```

Modules are added by their owning issues in dependency order; nothing is
pre-created speculatively.

## 5. Open architectural questions (tracked, not forgotten)

1. **Derivation multiplicity** in TB backward semantics (#38) —
   **resolved by D-V2-009**: states are order-quotiented discrete assembly
   objects (canonical derivations), the backward policy is the uniform leaf
   deletion policy, and the enumerable toy-DAG verification proves
   `P(x) = R(x)/Z` without multiplicity inflation
   (`docs/architecture/MDP_STATE_SPACE.md` §3, §5).
2. **Seed/anchor action** integration (#39) — where the pocket-anchor
   variables live in the state and their probability measure.
3. **pharm3d projection versioning** (#12) — QR-projection seed/matrix
   persistence contract.
4. **Empty-ligand interaction graphs** at t=0 for the backbone —
   **resolved by #21**: the schema supports empty ligand sub-graphs
   explicitly and `BipartitePaiNN` runs cleanly as a pocket-only encoder
   when `N_l = 0` (pinned by `tests/test_bipartite_painn.py`).
