# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-02, session S-V2-08 close (#11 CHEM-01, #13
CHEM-03, #15 CHEM-05 implemented, verified, and closed — decisions
D-V2-016/017; suite 655/655).

## 1. Continuation instruction

```
Next session: implement #24 MODEL-04 (Decoupled Hierarchical Policy
Heads & Discrete SO(2) Torsion Space) — it is the next critical-path
node (#24 → #25 → #26 → …), ALL THREE of its dependencies are now
closed (#11 2026-11-02 D-V2-016, #8 2026-11-02 D-V2-013, #23 2026-11-01
D-V2-012), and its landing unblocks #25 (and retires #42's last open
dependency). Read issue #24 in full first. The module path is
syntree/models/ (heads) + syntree/engine/environment.py (geometry
contract carrier). Binding consumption notes from the closed
dependencies:

* Consume DecoderOutput.context [B, 256] (syntree/models/policy.py) and
  the per-row HandleSelectionOutput.handle_embeddings; per the closing
  LayerNorm caveat each context row sums to the constant sum(beta), so
  head losses must be per-component weighted — context.sum() has
  exactly zero gradient (tests/test_cross_attention_decoder.py header
  note and D-V2-012).
* Consume syntree/models/torsion_head.py::von_mises_loss for the
  circular torsion objective (D-V2-013) — never re-derive the von Mises
  mathematics.
* Consume syntree/utils/hardware.py::apply_logit_mask /
  unmask_target_actions / LOGIT_MASK_FILL / LOGIT_MASK_LEGAL_ABOVE for
  every categorical head (D-V2-014); STOP is evaluated BEFORE the
  continuation branches (#45 §7.3/§8.4) and must never score
  unavailable continuation heads.
* Consume the certified registry from #11:
  syntree/chemistry/reactions.py::POLICY_REACTIONS exposes the 27
  reaction ids, their 9 family names (the coarse family head normalizes
  over exactly these), per-reaction molecule/block handle classes, and
  resonance locks (D-V2-016). DEFAULT_REACTION_REGISTRY is already the
  certified taxonomy — no registry work is needed for #24.
* The junction geometry contract (bin centres, resonance snap,
  determinism) is already implemented and pinned by the 50 tests in
  tests/test_gflownet_reconciliation.py (D-V2-009) — supersede
  internals, never the contracts.

Last verified code state: S-V2-08 commits on 3d-synV2 (HEAD 864530d +
this docs commit; suite 655/655). #5, #6, #7, #8, #9, #10, #11, #13,
#14, #15, #21, #22, #23, #38 are implemented from scratch, locally
verified, and CLOSED — do NOT redo the purge, the schema, the norms, the
von Mises math, the cosine cutoff, the FP32 gate, the soft-core
clash/volume floors, the reaction taxonomy, the #13 geometry stages, the
canonical fragmenter, the backbone, the handle-selection head, the
cross-attention decoder, or the GFlowNet state-space reconciliation.
Before coding, re-read issue #24 plus D-V2-009 (geometry contract),
D-V2-012 (LayerNorm caveat), D-V2-013 (von Mises), D-V2-016 (registry
metadata), inspect syntree/models/policy.py + handle_head.py +
torsion_head.py, and confirm branch HEAD matches origin (push races
possible; sandbox shells recycle — re-export GH_TOKEN per command and
verify `git remote get-url origin | wc -c` is 99 before pushing, see
WORK_LOG S-V2-07 §1). Parallel-safe alternatives if #24 is taken: #40
CHEM-06 (unblocks #12; the registry metadata it needs is in
syntree/chemistry/reactions.py — POLICY_REACTIONS handle classes +
junction patterns), #16 DATA-01 (provenance record at
docs/data/RAW_SOURCE_PROVENANCE.yaml), #32 TEST-01 (aggregation of the
655 existing tests). #36 OPS-01 remains gated on user authorization
(Colab SSH). NOTE for #25 (after #24): its #39 dependency is a design
interface decision, not full #39 implementation — resolve per #45 §17
by inspecting what #25 actually consumes from the seed/anchor head.
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
  layer (`syntree/utils/hardware.py`) is the canonical gate: the
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
  `tests/test_soft_core_chemistry.py` (#14) **plus the #13 stages**
  (`apply_junction_torsion` / `relax_junction_constrained`) and tests
  `tests/test_conformer_relaxation.py` (#13, D-V2-017),
  `syntree/chemistry/reactions.py` + tests `tests/test_32_reactions.py`
  (#11, D-V2-016), `syntree/data/fragmenter.py` + tests
  `tests/test_canonical_fragmenter.py` (#15).
- Backbone contract highlights (D-V2-007/D-V2-008/D-V2-011/D-V2-012):
  `GaussianRadialBasis` emits the 20-Gaussian expansion MULTIPLIED by the
  continuous cosine cutoff envelope — integrated inside the producer,
  unconditionally; the backbone accepts FP32 (production) and FP64
  (local verification only); `build_union_radius_graph` is the one
  authoritative edge construction. Decoder: dense
  `forward(query [B,1,256], memory [B,N,256], key_padding_mask [B,N]) ->
  DecoderOutput` with `context [B,256]`; `forward_state(backbone, handle)`;
  padded columns get exactly-zero attention; fully-padded graphs fail
  closed; frozen profile 1,219,328 params.
- Engine/physics contracts (D-V2-009/D-V2-015/D-V2-016/D-V2-017): the
  discrete Assembly MDP with bin-centre geometry and certified
  resonance locks (trans for rxn_01/06/08/17/20); terminal-only
  sub-bin Brent refinement against the production flat-bottom clash
  potential; the #13 stages (`apply_junction_torsion`,
  `relax_junction_constrained` + the `finalize_terminal_molecule` opt-in
  flags `junction_torsion_stage` / `pocket_vdw_relaxation`) whose
  defaults preserve the legacy behaviour bit-for-bit; uniform
  leaf-deletion backward policy (STOP term zero by proof); stationary
  `ThreeDReward`. `MDP/TB contracts are pinned by the 50 tests in
  tests/test_gflownet_reconciliation.py — supersede internals, never
  the contracts`.
- Certified taxonomy contracts (D-V2-016): 32 reactions = 8 C-N +
  7 C-C + 6 C-O/C-S + 6 heterocycle + 5 deprotections; 9 families with
  rxn_14/rxn_15 in `cross_coupling_cc`; heterocycles pre-cyclized on
  the synthon side; stereo retention verified (16 stereo pairs + CIP
  tests); `junction_pattern`/`junction_priority` metadata for #40/#41;
  `DEFAULT_REACTION_REGISTRY` is the certified taxonomy (the generic
  interim was removed — never resurrect it).

## 3. What the next agent must NOT redo

- The clean-room purge (#5): never restore or port pre-reset code
  (D-V2-001).
- The frozen tensor schema (#6): `syntree/core/schema.py` — extend it,
  never fork it (93 tests pin it).
- The non-singular primitives and PaiNN blocks (#7) — #21 composes them.
- The continuous cosine cutoff envelope (#9, D-V2-011): integrated
  inside `GaussianRadialBasis.forward` — the one authoritative place for
  ALL spatial distance RBF projections.
- The union-graph backbone (#21): `BipartitePaiNN` 8×d256, exact
  `build_union_radius_graph`, frozen profile 5,999,104 params.
- The agency-enabled handle-selection head (#22, D-V2-010).
- The two-layer stacked cross-attention decoder (#23, D-V2-012) —
  including the closing-LayerNorm caveat binding #24/#25.
- The reconciled two-level GFlowNet engine layer (#38 close, D-V2-009).
- The von Mises torsion mathematics (#8, D-V2-013) — #24 must CONSUME
  `von_mises_loss`, not re-derive it.
- The fail-closed FP32 precision engine (#10, D-V2-014) — #26 must
  CONSUME `RuntimeConfig`; #24 must CONSUME the logit-mask constants.
- The production soft-core clash + volume floors (#14, D-V2-015).
- The certified 32-reaction taxonomy (#11, D-V2-016):
  `syntree/chemistry/reactions.py` — the registry, family map,
  junction patterns, deprotections, and certification pairs are
  verified; extend (e.g. #40 chemoselectivity rules on top), never
  fork.
- The #13 geometry stages (D-V2-017): `apply_junction_torsion` /
  `relax_junction_constrained` in `syntree/chemistry/conformer.py` and
  the `finalize_terminal_molecule` opt-in flags.
- The canonical decomposer (#15): `syntree/data/fragmenter.py` — #18
  consumes it for trajectory extraction.
- README / architecture / agent-docs structure — update in place.

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| No GPU/T4 execution, training, or HF artifact sync has occurred for V2 | #36/#43 gate | CPU sandbox cannot establish GPU claims |
| Docking surrogate (`S_dock`/`S_contact`) is a documented interim | #30/eval | D-V2-015 role separation; production closure is #30's remaining scope (needs #29) |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | CPU-only sandbox; the refinement runs the production clash potential locally |
| Cavity-volume estimator is a floor-calibration scale (overestimates open surfaces) | #16 | Grid resolution 0.5 Å default; calibrate on real pockets with DATA-01 |
| Bit-for-bit determinism is per RDKit build, not cross-version | #24/#42 | Documented in D-V2-009 and MDP_STATE_SPACE.md §6 |
| Dev environment not pinned in a lockfile | #26/#43 | `pyproject.toml` declares loose floors; pin exact versions when the trainer lands |
| Schema `handle_nodes` optional context | #25 | Policy-side extension point, deliberately outside the 12-field core |
| 64-D handle descriptor producer | #12/#40 | `HandleSelectionHead` consumes `[H, 64]` rows; tests use synthetic descriptors until catalog featurization lands |
| Unstamped states pass field validation (interim) | #19 | Remove the interim rule when shard producers stamp |
| MMFF94 amide/ester torsions sit ~3.7–11.6° off planar at their minimum (measured) | #13/#42/#30 | The #13 stages restore planarity exactly post-minimization (D-V2-017); any future force-field swap must keep that guarantee |
| Pocket metals get MMFF generic type 95 (placeholder vdW shell) | #13/#16 | Documented approximation of `relax_junction_constrained`; real pockets via #16 |
| Heterocycle junction cuts resolve to generic aryl reactions by (priority, id) | #40/#41 | Deterministic by design (D-V2-016); heterocycle-specific refinement belongs to #40/#41 |
| Activation-checkpointing T4 benefit unmeasured | #43 | Capability implemented + correctness-tested on CPU only |
| `GaussianRadialBasis` buffers carry construction-dtype rounding | #21 (pre-existing) | Harmless for all pinned contracts (~1e-7 relative) |
| fp32 `0.5·(cos+1)` cancellation floor near the envelope seam | #9 (documented, benign) | Values ≤ ~6e-8 within ~1e-4 of the cutoff in fp32; pinned with appropriate tolerances |
| Amos backward is an approximation (max 3.30e-2 vs I1/I0 near κ≈2.5) | #8 (documented, pinned) | The issue-pinned design; forward is exact |

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
   **655/655** as of S-V2-08 (534 baseline + 82 taxonomy + 18
   relaxation + 21 fragmenter; 16 taxonomy skips by design).
4. Work per the #45 §18 per-issue protocol; end with the mandatory git
   diff review, commit+push (report exact SHAs), issue update, and
   refreshed handoff. NOTE: sandbox shells recycle between tool calls —
   re-export `GH_TOKEN` per command; if a push fails with "Invalid
   username or token", check `git remote get-url origin | wc -c` (99
   means the token is embedded, 59 means it silently cloned anonymously
   — fix with `git remote set-url`).
