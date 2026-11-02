# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-02, session S-V2-10 close (#16 DATA-01
implemented, executed full-scale, CLOSED at `fccd670` — suite 942/16;
#12 CHEM-02 catalog artifacts built full-scale with every acceptance
criterion verified and synced to the HF artifact store, but the build
code was lost to a sandbox recycle before commit — issue left OPEN on
the re-landing scope; decisions D-V2-021/022/023 artifact-pinned).

## 1. Continuation instruction

```
Next session: two recovery items come FIRST (both are mechanical-ish
and unblock the most downstream work), then the normal pipeline:

1. #12 code re-landing (highest leverage): re-implement
   syntree/chemistry/synthon_encoder.py + scripts/build_synthon_catalog.py
   + tests. EVERY design fact is pinned by the surviving artifacts
   (docs/data/catalog/catalog_manifest.json + the HDF5 on HF
   JJKK1212/3d-syntree-v2-catalog, private): d_raw 2137 (2048 Morgan +
   47 physchem + 6 3D-shape + 5 electrostatic + 30 handle vectors + 1
   norm; the issue's "2138" does not add up), z-normalization then QR
   Johnson-Lindenstrauss projection seed 2026 to 256, L2-normalized;
   ETKDGv3 + MMFF94 where parameterizable (per-row conformer_methods:
   etkdg3 17,493 / etkdg3+mmff94 32,507); fail-closed encode with named
   reasons; screen-before-admission via the in-repo #40
   chemoselectivity screen_block/screen_catalog (39,150 exclusions);
   stratified selection 35k-saturated + aromatic refill in
   canonical-SMILES order + 5k reserve (D-V2-023). Re-verify the
   re-implementation reproduces the manifest's funnel/selection
   semantics and pin unit tests to the artifact numbers; a byte-exact
   h5 rebuild is NOT required (the existing artifact is immutable and
   authoritative) — but the code path must regenerate an equivalent
   manifest from the same source SDF. Then close #12.
2. Regenerate the #16 outputs: re-run scripts/preprocess_multidataset.py
   (in-repo at fccd670) — the S-V2-10 full-scale run's output artifacts
   were partially lost to the recycle (47,483/170,954 complex pairs
   survive; processed_manifest.jsonl is gone). All three raw sources
   survive with checksums pinned in docs/data/RAW_SOURCE_PROVENANCE.yaml
   (crossdocked_pocket10.tar.gz 1.6 GB, pbpp-2020.zip 625 MB,
   bindingmoad_final.csv). The driver is resumable/append-only.
3. #17 DATA-02 (deps: #16 closed): MMseqs2 30%-identity clustered
   splits over the regenerated manifest's protein chains. The mmseqs
   binary is staged in the dev workspace (3d-syntree-work/tools/mmseqs)
   but the issue's acceptance needs it in the execution environment.
4. #32 TEST-01 (deps: #5 only) — aggregate the 942-test capability
   suite into the Suite A runner; parallel-safe with the above.
5. #36 OPS-01 prepare-only (user-authorization gate; never claim a
   connection). An authorized T4 session ALSO closes #25's remaining
   CUDA criterion (one full-profile SynTreePolicy forward with zero
   NaN/Infs on device) — run it first if the owner authorizes.

Last verified code state: S-V2-10 commits on 3d-synV2 (HEAD fccd670 +
this docs commit; suite 942 passed / 16 skipped). Closed this session:
#16 (fccd670). Artifacts-built/awaiting-code: #12 (evidence at
docs/data/catalog/, HDF5 on HF). Do NOT redo the #16 implementation
(only re-run its driver for outputs), do NOT rebuild the #12 catalog
artifacts (they are final, immutable, checksum-pinned), and do NOT redo
anything from S-V2-09 or earlier (see §3).

Binding consumption notes from this session:

* The #12 catalog HDF5 schema (pharm3d-v2.0 / catalog-v2.0) is frozen:
  catalog/ group (block_ids, smiles, synthon_classes — 51 pipe-joined
  combos incl. '' for seed-only rows, seed_classes — 29 combos, strata,
  mw, fsp3, heavy_atoms, rotatable_bonds, conformer_energies,
  conformer_methods), embeddings [50000, 256] float32, projection/Q
  [2137, 256] + projection/seed 2026, normalization/ mean+std [2137],
  versions/ pinned toolchain. The tied nn.Embedding(50_000, 64) in
  SynTreePolicy (#25) is the model-side consumer; its INITIALIZATION
  from these embeddings is a #26 trainer decision, not yet made.
* The catalog is only valid against the certified reaction registry
  digest 32ea6db9ce9d56747c6a3cc957e3dfa0c2413fa02a6a583cd07e23ff6f4c3f3d
  (recorded in the manifest) — any registry change invalidates
  certification and requires a re-screen.
* #16's uniform manifest schema (204,005 records: resolution/rmsd,
  ligand SMILES/MW/heavy counts/Fsp3, pocket stats, per-chain protein
  sequences = the #17 clustering input, artifact paths, provenance
  notes; nulls explicit, never fabricated) is pinned by the #16 close
  evidence in the issue tracker — regenerate before #17/#18 consume it.
* HF artifact channel: dataset repo JJKK1212/3d-syntree-v2-catalog
  (private, owner account). #37's automated sync remains the gated
  end-state; the catalog upload was a one-time preservation action
  under the owner's directive. Never commit tokens or *.h5 to git.
* From S-V2-09 (still binding): the #24 two-stage STOP/continuation
  protocol; every categorical routed through the #10 logit-mask
  machinery; TorsionOutput.selected_log_prob is the ONLY TB measure;
  the #25 assembly profile 10,880,719 params = 43.5 MB FP32; the #40
  per-class exclusion matrix with the asymmetric 2,4-dihaloazine
  exemption.
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
  `tests/test_chemoselectivity.py` (#40), **`syntree/data/preprocessing.py`
  + `scripts/preprocess_multidataset.py` +
  `tests/test_preprocessing.py` (#16), `syntree/chemistry/synthon_encoder.py`
  + `scripts/build_synthon_catalog.py` (#12 — to be re-landed)**.
- Backbone/decoder contracts (D-V2-007/008/011/012), the engine/MDP
  contracts (D-V2-009/015/016/017), and the S-V2-09 decisions
  (D-V2-018/019/020: two-stage STOP protocol, exact family→reaction
  factorization, 3,272,975 head profile, 10,880,719 assembly profile /
  43.5 MB < 50 MB binding interpretation, per-class chemoselectivity
  matrix with the asymmetric exemption) are unchanged — see the S-V2-08
  handoff's section 2 for the full list (still binding).
- New this session (D-V2-021/022/023, artifact-pinned — rationale text
  for 021/022 partially lost, see DECISIONS.md): d_raw 2137; QR-JL
  projection seed 2026 + stored z-normalization; ETKDGv3/MMFF94
  conformer policy with per-row provenance; fail-closed encode;
  screen-before-admission; registry-digest binding; immutable-HDF5
  policy; certified-saturated-supply as the stratification binding
  constraint with aromatic refill in canonical-SMILES order.
- **Sandbox-recycle protocol (new, hard-learned):** commit+push early
  and often — the S-V2-10 session lost a completed module to a recycle
  that hit between "work done" and "work pushed". For any multi-hour
  build, push the code BEFORE launching the long-running execution, and
  treat uncommitted work as at-risk.

## 3. What the next agent must NOT redo

- Everything in the S-V2-08 handoff's section 3 (unchanged).
- The #24 hierarchical heads + torsion head (`74133c8`), the #25
  SynTreePolicy assembly + the #39 seed/anchor component (`84fd3e7`),
  the #40 chemoselectivity screen + manifest (`044adca`) — S-V2-09.
- The #16 implementation (`fccd670`): preprocessing library + driver +
  tests are in-repo and closed. Only re-RUN the driver to regenerate
  the lost output artifacts (raws survive).
- The #12 catalog BUILD itself: the artifacts (HDF5 + manifest +
  funnel + checksums) are final, verified, immutable, and preserved
  (HF + `docs/data/catalog/`). Re-land only the CODE, pinned against
  them; do not re-curate, re-select, or re-encode.

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| **#12 build code lost pre-commit (S-V2-10 incident)** | #12 | Artifacts + manifest survive and are pinned; re-land code, then close |
| **#16 full-scale outputs partially lost** (47,483/170,954 pairs; `processed_manifest.jsonl` gone) | #17 | Mechanical regeneration via the in-repo driver; raws + checksums survive |
| No GPU/T4 execution, training, or HF artifact sync beyond the catalog | #36/#43 gate | CPU sandbox cannot establish GPU claims |
| **#25's CUDA acceptance criterion is unverified** | #25/#36 | One full-profile forward on the authorized T4 closes it |
| Docking surrogate (`S_dock`/`S_contact`) is a documented interim | #30/eval | Production closure is #30's remaining scope (needs #29) |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | The #24 torsion head integration is now available |
| Cavity-volume estimator is a floor-calibration scale | #16 | Real pocket stats were in the lost manifest; regenerate with the #16 re-run, then calibrate |
| Bit-for-bit determinism is per RDKit build | #24/#42 | Documented in D-V2-009 / MDP_STATE_SPACE.md §6 |
| Dev environment not pinned in a lockfile | #26/#43 | Pin when the trainer lands |
| Schema `handle_nodes` optional context | #25 | Policy-side extension point |
| 64-D handle descriptor producer | #12 re-landing | `HandleSelectionHead` consumes [H, 64]; the catalog h5 carries the 256-d table; the 64-d producer is part of the re-landed encoder scope (verify against `tests/test_handle_selection.py`'s synthetic descriptors) |
| Unstamped states pass field validation (interim) | #19 | Remove when shard producers stamp |
| MMFF94 amide/ester torsions ~3.7–11.6° off planar at minimum | #13/#42/#30 | #13 stages restore planarity exactly (D-V2-017) |
| Pocket metals get MMFF generic type 95 | #13/#16 | Re-verify on regenerated real pockets |
| Heterocycle junction cuts resolve to generic aryl reactions | superseded | #40's chemoselectivity governs curation-level selectivity |
| Activation-checkpointing T4 benefit unmeasured | #43 | Capability + gradient flow CPU-tested via #25 |
| `GaussianRadialBasis` buffers carry construction-dtype rounding | #21 (pre-existing) | Harmless (~1e-7 relative) |
| fp32 `0.5·(cos+1)` cancellation floor near the envelope seam | #9 (documented, benign) | Pinned with tolerances |
| Amos backward is an approximation (max 3.30e-2 vs I1/I0 near κ≈2.5) | #8 (documented, pinned) | Issue-pinned design |
| The chemoselectivity matrix encodes standard-condition chemistry | #40 | Corner-case reagents outside the rule model; reason codes name every feature for deliberate overrides |
| 16,578 CrossDocked poses rejected (non-kekulizable OpenBabel SDFs, no mol2 in the mirror) | #16 | Documented; recoverable if an alternative mirror is pinned |

## 5. Remote runtime / checkpoint state

No Colab/Kaggle session has been opened (user authorization not
requested; #36 gate). No V2 checkpoints exist. **One HF artifact is
synced:** the #12 catalog dataset `JJKK1212/3d-syntree-v2-catalog`
(private; uploaded 2026-11-02 under the owner's "just push it"
directive as a preservation action — full automation remains #37's
gated scope). No secrets are stored in the repository (session tokens
used only via git remote URLs / per-command env exports, never
committed).

## 6. Session mechanics

1. `git clone --branch 3d-synV2 ...` → confirm HEAD matches §1's
   recorded commit; `git fetch` first if in doubt (push races possible).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]" &&
   pip install torch_geometric`.
3. Baseline the suite: `python -m pytest tests/ -q` — expected
   **942 passed / 16 skipped** as of S-V2-10 (914 at S-V2-09 + 28
   preprocessing; 16 #11 skips by design).
4. Work per the #45 §18 per-issue protocol; **push code before any
   long-running execution** (§2 recycle protocol); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export `GH_TOKEN` per command; if a push fails with
   "Invalid username or token", check `git remote get-url origin | wc -c`
   (≈190 means the token is embedded across both remote lines, ≈86
   means it cloned anonymously — fix with `git remote set-url`).
