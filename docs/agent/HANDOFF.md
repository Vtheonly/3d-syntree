# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-02, session S-V2-09 close (#24 MODEL-04 and
#40 CHEM-06 implemented, verified, and CLOSED; #25 MODEL-05 implemented
and CPU-verified, left OPEN pending its CUDA acceptance criterion —
decisions D-V2-018/019/020; suite 914/914).

## 1. Continuation instruction

```
Next session: the critical path is blocked on data, so execute the
next dependency-ready issues in this order of preference:

1. #16 DATA-01 (multi-source preprocessing; deps: #5 only — closed and
   verified). This is the highest-leverage open issue: it unblocks
   #17 → #18 → #19 → #20 → #26 (the entire training critical path).
   Provenance record already exists at
   docs/data/RAW_SOURCE_PROVENANCE.yaml. Read issue #16 in full first.
2. #32 TEST-01 (Suite A aggregation; deps: #5 only). Aggregates the 914
   existing tests; mechanical, parallel-safe with #16.
3. #36 OPS-01 (prepare-only: notebook + workflow docs; the remote
   session itself is user-authorization-gated — never claim a
   connection). Note: an authorized T4 session would ALSO close #25's
   remaining CUDA criterion (one full-profile forward with zero
   NaN/Infs on device) — run it first if the owner authorizes.

Parallel-safe alternatives if the primary is taken: #12 CHEM-02 (now
unblocked on #40; the remaining blocker is licensed catalog access —
if the mirror is reachable, the chemoselectivity screen +
CurationManifest from syntree/chemistry/chemoselectivity.py are ready
to consume: screen_catalog produces the manifest), #39's remaining
environment-side scope (blocked on #12).

Last verified code state: S-V2-09 commits on 3d-synV2 (HEAD 044adca +
this docs commit; suite 914/914). Closed this session: #24 (74133c8),
#40 (044adca). Implemented-awaiting-verification: #25 (84fd3e7; CUDA
criterion only). Do NOT redo the hierarchical heads, the torsion head,
the policy assembly, the seed/anchor component, or the
chemoselectivity screen.

Binding consumption notes from this session:

* #24 heads: syntree/models/policy.py::HierarchicalPolicyHeads — the
  two-stage protocol (forward_stop FIRST; forward_continue for the
  CONTINUE subset only) is the pinned usage pattern (tests use
  forward-hook instrumentation; D-V2-018). The mask factories
  (build_continuation_masks / build_force_stop /
  reaction_support_mask / synthon_legality_mask) consume the
  environment active_handles() menu and the registry metadata — #25/#26
  trainers must route EVERY categorical through the #10 logit-mask
  machinery exactly as these heads do.
* #25 assembly: syntree/models/policy.py::SynTreePolicy — frozen
  profile 10,880,719 params = 43.5 MB FP32 (D-V2-019; the <50MB
  criterion is the binding one). The tied catalog table is ONE
  nn.Embedding shared by SynthonRetrievalHead and SeedAnchorHead
  (identity-pinned). The seed/anchor machinery:
  policy.anchor_candidates(...) → policy.anchor_features(...) →
  policy.forward_seed_anchor(...); SeedAnchorOutput.step_log_prob() is
  the seed action's TB forward contribution.
* #25's OPEN criterion (CUDA forward zero-NaN/Inf) needs one
  full-profile run on the authorized T4: construct SynTreePolicy(),
  run the test_frozen_profile_forward-style flow on device, post the
  evidence to #25, then close.
* #40 screen: syntree/chemistry/chemoselectivity.py — #12 must call
  screen_catalog AFTER its basic molecular validation and embed the
  CurationManifest (per-rule exclusion counts + percentages) in the
  catalog manifest (issue #40 acceptance). The exclusion matrix is
  per handle class (D-V2-020); the 2,4-dihaloazine exemption is
  ASYMMETRIC (spectator must be flanked by MORE adjacent ring N than
  the handle).
* Torsion: TorsionOutput.selected_log_prob is the ONLY TB measure;
  (mu, kappa) belong to torsion_circular_loss (supervision) alone.
  1-based bins match AssemblyAction.torsion_bin exactly.
```

## 2. Binding context for every future session

- The branch is a **clean-room rebuild**: emptied at `075f86d`
  (2026-11-01). Never restore pre-reset code; carry forward
  specifications only (D-V2-001/D-V2-002).
- #45 is the permanent constitution: read fully before any action;
  never close it.
- Pure FP32 on Colab/Kaggle Tesla T4 only; the dev sandbox is CPU-only
  and can never establish GPU claims (#45 §5–§6). The #10 enforcement
  layer (`syntree/utils/hardware.py`) is the canonical gate; the #26
  trainer must consume `RuntimeConfig` and never construct
  autocast/GradScaler (AST-banned).
- Issue-scoped module paths and test file names pinned by issue bodies
  are binding (D-V2-002). Pinned: `syntree/core/schema.py` +
  `tests/test_tensor_contracts.py` (#6), `syntree/models/equivariant.py`
  + `tests/test_mathematical_singularities.py` (#7),
  `syntree/models/bipartite.py` + `tests/test_bipartite_painn.py`
  (#21), `syntree/models/handle_head.py` +
  `tests/test_handle_selection.py` (#22), `syntree/models/policy.py` +
  `tests/test_cross_attention_decoder.py` (#23/#25) **+
  `tests/test_hierarchical_heads.py` (#24) +
  `tests/test_policy_assembly.py` (#25)**,
  `syntree/models/cutoff.py::continuous_cosine_envelope` +
  `tests/test_cosine_cutoff.py` (#9), `syntree/models/torsion_head.py`
  + `tests/test_von_mises_torsion.py` (#8) **+ the #24 torsion head
  (same file, D-V2-018)**, `syntree/utils/hardware.py` +
  `tests/test_fp32_enforcement.py` (#10),
  `syntree/chemistry/conformer.py` +
  `tests/test_soft_core_chemistry.py` (#14) + the #13 stages +
  `tests/test_conformer_relaxation.py` (#13),
  `syntree/chemistry/reactions.py` + `tests/test_32_reactions.py`
  (#11), `syntree/data/fragmenter.py` +
  `tests/test_canonical_fragmenter.py` (#15),
  `syntree/chemistry/chemoselectivity.py` +
  `tests/test_chemoselectivity.py` (#40).
- Backbone/decoder contracts (D-V2-007/008/011/012) and the engine/MDP
  contracts (D-V2-009/015/016/017) are unchanged this session — see the
  S-V2-08 handoff's section 2 for the full list (still binding).
- New this session (D-V2-018/019/020): the two-stage STOP/continuation
  protocol and its hook-pinned isolation; the exact family→reaction
  factorization by support restriction; parity/torsion conditioning on
  the tied synthon embedding; kappa-by-softplus with the MLE anchor;
  the 3,272,975 head profile; the 10,880,719 assembly profile and the
  <50MB binding interpretation; the per-class chemoselectivity matrix
  with the asymmetric 2,4-dihaloazine exemption.

## 3. What the next agent must NOT redo

- Everything in the S-V2-08 handoff's section 3 (unchanged).
- The #24 hierarchical heads + torsion head (`74133c8`) — including the
  mask factories, the teacher-forced protocol, and the circular loss.
- The #25 SynTreePolicy assembly + the #39 seed/anchor component
  (`84fd3e7`) — including the anchor generator, the factorized
  SeedAnchorHead, and the tied-table wiring.
- The #40 chemoselectivity screen + manifest (`044adca`).

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| No GPU/T4 execution, training, or HF artifact sync has occurred for V2 | #36/#43 gate | CPU sandbox cannot establish GPU claims |
| **#25's CUDA acceptance criterion is unverified** | #25/#36 | One full-profile forward on the authorized T4 closes it |
| Docking surrogate (`S_dock`/`S_contact`) is a documented interim | #30/eval | Production closure is #30's remaining scope (needs #29) |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | The #24 torsion head integration is now available |
| Cavity-volume estimator is a floor-calibration scale | #16 | Calibrate on real pockets with DATA-01 |
| Bit-for-bit determinism is per RDKit build | #24/#42 | Documented in D-V2-009 / MDP_STATE_SPACE.md §6 |
| Dev environment not pinned in a lockfile | #26/#43 | Pin when the trainer lands |
| Schema `handle_nodes` optional context | #25 | Policy-side extension point |
| 64-D handle descriptor producer | #12 | `HandleSelectionHead` consumes [H, 64]; tests use synthetic descriptors |
| Unstamped states pass field validation (interim) | #19 | Remove when shard producers stamp |
| MMFF94 amide/ester torsions ~3.7–11.6° off planar at minimum | #13/#42/#30 | #13 stages restore planarity exactly (D-V2-017) |
| Pocket metals get MMFF generic type 95 | #13/#16 | Real pockets via #16 |
| Heterocycle junction cuts resolve to generic aryl reactions | superseded | #40's chemoselectivity now governs curation-level selectivity |
| Activation-checkpointing T4 benefit unmeasured | #43 | Capability + gradient flow CPU-tested via #25 |
| `GaussianRadialBasis` buffers carry construction-dtype rounding | #21 (pre-existing) | Harmless (~1e-7 relative) |
| fp32 `0.5·(cos+1)` cancellation floor near the envelope seam | #9 (documented, benign) | Pinned with tolerances |
| Amos backward is an approximation (max 3.30e-2 vs I1/I0 near κ≈2.5) | #8 (documented, pinned) | Issue-pinned design |
| The chemoselectivity matrix encodes standard-condition chemistry | #40 | Corner-case reagents outside the rule model; reason codes name every feature for deliberate overrides |

## 5. Remote runtime / checkpoint state

None — no Colab/Kaggle session has been opened (user authorization not
requested; #36 gate). No V2 checkpoints exist. No HF artifacts synced.
No secrets are stored in the repository (the session PAT was used only
via the git remote URL and per-command env exports, never committed).

## 6. Session mechanics

1. `git clone --branch 3d-synV2 ...` → confirm HEAD matches §1's
   recorded commit; `git fetch` first if in doubt (push races possible).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]" &&
   pip install torch_geometric`.
3. Baseline the suite: `python -m pytest tests/ -q` — expected
   **914/914** as of S-V2-09 (655 at S-V2-08 + 100 hierarchical-heads
   + 48 policy-assembly + 111 chemoselectivity; 16 #11 skips by
   design).
4. Work per the #45 §18 per-issue protocol; end with the mandatory git
   diff review, commit+push (report exact SHAs), issue update, and
   refreshed handoff. NOTE: sandbox shells recycle between tool calls —
   re-export `GH_TOKEN` per command; if a push fails with "Invalid
   username or token", check `git remote get-url origin | wc -c` (93
   means the token is embedded, 43 means it silently cloned anonymously
   — fix with `git remote set-url`).
