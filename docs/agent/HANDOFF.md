# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-02, session S-V2-07 close (#8 MATH-02, #10
MATH-04, #14 CHEM-04 implemented, verified, and closed — decisions
D-V2-013/014/015).

## 1. Continuation instruction

```
Next session: implement #11 CHEM-01 (the certified 32-reaction taxonomy
grouped into 9 coarse policy families) — it is dependency-ready (deps:
#5 only), it is now the SOLE remaining blocker of #24 MODEL-04 on the
critical path (#24 → #25 → #26 → …), and it unblocks the whole
chemistry chain (#12 catalog → #13/#15 relaxation/decomposition → #40
chemoselectivity → #41 index). Read issue #11 in full first; module path
per the issue body: syntree/chemistry/ (the package now exists — created
by #14 with conformer.py; the reaction registry is a sibling module,
e.g. syntree/chemistry/reactions.py). Per #45 §9.1 each reaction must
define SMARTS templates, required handle types, atom/bond mapping and
provenance, stereochemical behaviour, validity constraints, product
sanitization, failure behaviour, and family membership; certification is
NOT "the SMARTS executes" — representative reagent/product tests with
documented chemical references are required; deterministic deprotections
must be explicitly documented. The new registry must supersede the
engine's generic junction-reaction model (AssemblyEnvironment's
reaction_registry / JunctionReaction interface in
syntree/engine/environment.py) WITHOUT changing the MDP/TB contracts
(D-V2-009): the 50 tests in tests/test_gflownet_reconciliation.py pin
those contracts and must stay green (extend/parameterize fixtures rather
than rewriting them where the richer registry needs new cases). Last
verified code state: S-V2-07 commits on 3d-synV2 (suite 534/534; HEAD of
this session's docs commit). #5, #6, #7, #8, #9, #10, #14, #21, #22,
#23, #38 are implemented from scratch, locally verified, and CLOSED —
do NOT redo the purge, the schema, the norms, the von Mises math, the
cosine cutoff, the FP32 gate, the soft-core clash/volume floors, the
backbone, the GFlowNet state-space reconciliation, the handle-selection
head, or the cross-attention decoder. Before coding, re-read issue #11
plus D-V2-009 (engine scope boundaries) and D-V2-015 (role separation:
flat-bottom clash vs docking surrogate), inspect
syntree/engine/environment.py's JunctionReaction/DEFAULT_REACTION_REGISTRY
(the interim interface the registry fills in), and confirm branch HEAD
matches origin (push races possible; note that sandbox shells recycle —
re-export GH_TOKEN per command and verify `git remote get-url origin` is
98+ chars before pushing, see WORK_LOG S-V2-07 §1). Parallel-safe
alternatives if #11 is taken: #16 DATA-01 (provenance record exists at
docs/data/RAW_SOURCE_PROVENANCE.yaml), #32 TEST-01 (aggregation of the
129 existing capability tests). #36 OPS-01 remains gated on user
authorization (Colab SSH). For #24 (once #11 lands): consume
DecoderOutput.context [B, 256] (syntree/models/policy.py) + per-row
HandleSelectionOutput.handle_embeddings; use
syntree/models/torsion_head.py::von_mises_loss for the circular torsion
loss and syntree/utils/hardware.py::apply_logit_mask /
unmask_target_actions / LOGIT_MASK_FILL / LOGIT_MASK_LEGAL_ABOVE for the
categorical heads; NOTE the closing LayerNorm — each context row sums to
the constant sum(beta), so head losses must be per-component weighted
(context.sum() has exactly zero gradient; see
tests/test_cross_attention_decoder.py header note and D-V2-012).
```

## 2. Binding context for every future session

- The branch is a **clean-room rebuild**: it was emptied at commit
  `075f86d` (2026-11-01) per the owner's directive. Never restore,
  re-copy, or "port" pre-reset code; carry forward specifications only
  (decision D-V2-001/D-V2-002). Pre-reset commits are forensic history.
- #45 is the permanent constitution: read it fully before any action;
  never close it.
- Pure FP32 on Colab/Kaggle Tesla T4 only; the dev sandbox is CPU-only
  and can never establish GPU claims (#45 §5–§6). The #10 enforcement
  layer (`syntree/utils/hardware.py`) is now the canonical gate: the
  trainer (#26) must consume `RuntimeConfig` and never construct
  autocast/GradScaler (an AST test bans them repo-wide).
- Issue-scoped module paths and test file names pinned by issue bodies are
  binding (D-V2-002). Pinned so far: `syntree/core/schema.py` + tests
  `tests/test_tensor_contracts.py` (#6), `syntree/models/equivariant.py`
  + tests `tests/test_mathematical_singularities.py` (#7),
  `syntree/models/bipartite.py` + tests `tests/test_bipartite_painn.py`
  (#21), `syntree/models/handle_head.py` + tests
  `tests/test_handle_selection.py` (#22), `syntree/models/policy.py` +
  tests `tests/test_cross_attention_decoder.py` (#23/#25),
  `syntree/models/cutoff.py::continuous_cosine_envelope` + tests
  `tests/test_cosine_cutoff.py` (#9), `syntree/models/torsion_head.py` +
  tests `tests/test_von_mises_torsion.py` (#8),
  `syntree/utils/hardware.py` + tests `tests/test_fp32_enforcement.py`
  (#10), `syntree/chemistry/conformer.py` + tests
  `tests/test_soft_core_chemistry.py` (#14).
- Backbone contract highlights (D-V2-007/D-V2-008/D-V2-011/D-V2-012):
  `GaussianRadialBasis` emits the 20-Gaussian expansion MULTIPLIED by the
  continuous cosine cutoff envelope — integrated inside the producer,
  unconditionally (no raw expansion exists anywhere in V2; the decoder
  projects NO distances — any future distance bias MUST route through
  `GaussianRadialBasis`); the backbone accepts FP32 (production) and
  FP64 (local verification only, must match module dtype);
  `build_union_radius_graph` is the one authoritative edge construction —
  downstream heads must reuse it, not re-implement it. Decoder
  highlights: dense `forward(query [B,1,256], memory [B,N,256],
  key_padding_mask [B,N]) -> DecoderOutput` with `context [B,256]`;
  V2-native `forward_state(backbone, handle)`; padded columns get
  exactly-zero attention; fully-padded graphs fail closed; frozen
  profile 1,219,328 params.
- Engine/physics contracts (D-V2-009/D-V2-015): the discrete Assembly MDP
  with bin-centre geometry; terminal-only sub-bin Brent refinement whose
  objective is now the **production flat-bottom soft-core clash
  potential** (`syntree/chemistry/conformer.py::soft_core_clash_energy`);
  scaffold-constrained MMFF94 (50 kcal/mol/Å² seed restraints); uniform
  leaf-deletion backward policy (STOP term zero by proof); stationary
  `ThreeDReward` whose `S_vol` carries the #14 cavity ratio floor (0.35,
  derived once at construction) + heavy-atom floor (12, weight 2.0) and
  whose `S_dock`/`S_contact` still consume the documented 12-6
  **docking surrogate** (`soft_core_pocket_energy` — a different
  injection point owned by #30/eval; do not conflate the two roles).
  `MDP/TB contracts are pinned by the 50 tests in
  tests/test_gflownet_reconciliation.py — supersede internals, never the
  contracts`.

## 3. What the next agent must NOT redo

- The clean-room purge (#5): the branch was emptied at `075f86d` and
  verified; **never** restore or port pre-reset code (D-V2-001) —
  pre-reset commits are forensic history only.
- The frozen tensor schema (#6): `syntree/core/schema.py` is the single
  state contract — extend it, never fork it. 93 tests in
  `tests/test_tensor_contracts.py` pin it.
- The non-singular primitives and PaiNN blocks (#7):
  `syntree/models/equivariant.py`; 45 tests pin the singularities and
  equivariance. #21 composes these blocks — do not reimplement them.
- The continuous cosine cutoff envelope (#9, D-V2-011):
  `syntree/models/cutoff.py::continuous_cosine_envelope` integrated
  INSIDE `GaussianRadialBasis.forward` — the one authoritative place for
  ALL spatial distance RBF projections. 46 + 4 rewritten legacy pins.
- The union-graph backbone (#21): `syntree/models/bipartite.py`
  (`BipartitePaiNN` 8×d256, exact `build_union_radius_graph`,
  activation checkpointing, frozen profile 5,999,104 params); 69 tests.
- The agency-enabled handle-selection head (#22, D-V2-010):
  `syntree/models/handle_head.py` (two-stage cross-attention over the
  active handle pool, per-graph segment log-softmax, fail-closed
  cross-field validation); 44 tests incl. the real
  `AssemblyEnvironment.active_handles()` menu integration.
- The two-layer stacked cross-attention decoder (#23, D-V2-012):
  `syntree/models/policy.py` (MHA 256/8/0.1 + FFN 256→512→SiLU→256,
  pre-LN; dense padded-memory interface; `forward_state`; SE(3)-invariant
  8.9e-16 FP64); 58 tests. The closing LayerNorm caveat for #24/#25 head
  losses is binding.
- The reconciled two-level GFlowNet engine layer (#38 close, D-V2-009):
  `syntree/engine/environment.py`, `rl.py`, `gflownet.py`, and
  `docs/architecture/MDP_STATE_SPACE.md` (formal spec + proofs). 50 tests
  incl. enumerable toy-DAG flow accounting. Do NOT re-litigate the
  state-space debate — it is closed with proofs.
- The von Mises torsion mathematics (#8, D-V2-013):
  `syntree/models/torsion_head.py` (`LogBesselI0` custom autograd
  Function — fp64 two-branch forward, analytic Amos' ratio backward,
  exact branch gating, fail-closed κ≥0 domain; `von_mises_log_prob` /
  `von_mises_loss` circular objective). 44 tests incl. the measured
  6.51e-6 forward accuracy vs `scipy.special.i0e` and the pinned 3.3e-2
  Amos deviation. #24 must CONSUME `von_mises_loss`, not re-derive it.
- The fail-closed FP32 precision engine (#10, D-V2-014):
  `syntree/utils/hardware.py` (`configure_runtime_environment` strict
  gate; `RuntimeConfig`; `DeviceInfo`/`probe_device` (simulated infos
  never mutate global backends); TF32 policy; `LOGIT_MASK_FILL=-1e4` /
  `LOGIT_MASK_LEGAL_ABOVE=-1e3` + `apply_logit_mask` /
  `unmask_target_actions` / `validate_target_actions`). 35 tests incl.
  the repo-wide AST ban on GradScaler/autocast/torch.amp/fp16/bf16. #26
  must CONSUME `RuntimeConfig`; #24 must CONSUME the logit-mask
  constants.
- The production soft-core clash + volume floors (#14, D-V2-015):
  `syntree/chemistry/conformer.py` (flat-bottom soft-core LJ with the
  500 kcal/mol bounded-spike ceiling; Bondi/oracle atom typing with
  LB-mixing + ceiling; analytic gradient + torch mirror;
  `SoftCoreClashPotential`; `pocket_cavity_volume`; the 0.35 ratio floor
  and 12-heavy-atom floor penalties) wired into the #42 refinement
  objective and `ThreeDReward.S_vol`. 50 tests. #30/#42/#16 CONSUME
  these; do not reimplement clash physics or volume floors elsewhere.
- README / architecture / agent-docs structure — update in place.

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| No GPU/T4 execution, training, or HF artifact sync has occurred for V2 | #36/#43 gate | CPU sandbox cannot establish GPU claims |
| Engine generic junction-reaction registry is a documented interim | #11 | The certified 32-reaction taxonomy must supersede it without changing the MDP/TB contracts |
| Docking surrogate (`S_dock`/`S_contact`) is a documented interim | #30/eval | D-V2-015 role separation; production closure is #30's remaining scope (needs #29) |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | CPU-only sandbox; the refinement now runs the production clash potential locally |
| Cavity-volume estimator is a floor-calibration scale (overestimates open surfaces) | #16 | Grid resolution 0.5 Å default; calibrate on real pockets with DATA-01 |
| Bit-for-bit determinism is per RDKit build, not cross-version | #24/#42 | Documented in D-V2-009 and MDP_STATE_SPACE.md §6 |
| Dev environment not pinned in a lockfile | #26/#43 | `pyproject.toml` declares loose floors; pin exact versions when the trainer lands |
| Schema `handle_nodes` optional context | #25 | Policy-side extension point, deliberately outside the 12-field core |
| 64-D handle descriptor producer | #12/#40 | `HandleSelectionHead` consumes `[H, 64]` rows; tests use synthetic descriptors until catalog featurization lands |
| Unstamped states pass field validation (interim) | #19 | Remove the interim rule when shard producers stamp |
| Backbone ≥1-pocket-atom-per-graph rule is stricter than schema | #21 | Documented in-module; a schema-level per-graph pocket rule may be worth adopting at #18/#19 |
| Activation-checkpointing T4 benefit unmeasured | #43 | Capability implemented + correctness-tested on CPU only |
| `GaussianRadialBasis` buffers carry construction-dtype rounding | #21 (pre-existing) | fp32-created buffers cast by `.to(float64)`; harmless for all pinned contracts (~1e-7 relative); analytic tests derive references from the module's own buffers |
| fp32 `0.5·(cos+1)` cancellation floor near the envelope seam | #9 (documented, benign) | Values ≤ ~6e-8 within ~1e-4 of the cutoff in fp32; pinned with appropriate tolerances |
| Amos backward is an approximation (max 3.30e-2 vs I1/I0 near κ≈2.5) | #8 (documented, pinned) | The issue-pinned design; forward is exact; deviation measured + test-pinned < 4e-2 |

## 5. Remote runtime / checkpoint state

None — no Colab/Kaggle session has been opened (user authorization not
requested; #36 gate). No V2 checkpoints exist. No HF artifacts synced.
No secrets are stored in the repository (the session PAT was used only
via the git remote URL and per-command env exports, never committed).

## 6. Session mechanics

1. `git clone --branch 3d-synV2 ...` → confirm HEAD matches §1's recorded
   commit; `git fetch` first if in doubt (push races possible).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]" &&
   pip install torch_geometric`.
3. Baseline the suite: `python -m pytest tests/ -q` — expected
   **534/534** as of S-V2-07 (405 baseline + 44 von Mises + 35
   FP32-enforcement + 50 soft-core chemistry tests).
4. Work per the #45 §18 per-issue protocol; end with the mandatory git
   diff review, commit+push (report exact SHAs), issue update, and
   refreshed handoff.  NOTE: sandbox shells recycle between tool calls —
   re-export `GH_TOKEN` per command; if a push fails with "Invalid
   username or token", check `git remote get-url origin | wc -c` (≈99
   means the token is embedded, ≈59 means it silently cloned anonymously
   — fix with `git remote set-url`).
