# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-03, session S-V2-11 close (**#12 CHEM-02
re-landed and closed** — artifact-exact: funnel 24/24 counts, 0/50,000
label mismatches, `buffered_rows` exact, projection bit-exact,
conformer-method split exact; **#17 DATA-02 closed** — full-scale
zero-leakage splits; **#32 TEST-01 closed** — Suite A 293/293; #16
outputs regenerated full-scale; suite **1303/16**; clean-room closures
**20/40**).

## 1. Continuation instruction

```
Next session: the data chain's next critical path is #41, then #18.

1. #41 DATA-06 (dependency-READY; deps #11+#12 closed 2026-11-03): the
   versioned hashed inverted index over the certified catalog. Inputs
   available: the immutable catalog HDF5 (private HF
   JJKK1212/3d-syntree-v2-catalog, sha256-pinned in
   docs/data/catalog/) and the certified pool (152,489 blocks; the
   rebuild driver's workdir cache or a re-run reproduces it exactly —
   see §5). Acceptance requires the brute-force equivalence benchmark
   on an exhaustive subset + measured throughput/storage/reduction +
   index version compatibility. Remember: index hits are candidate
   prefilters only (constitution §9.5).
2. #18 DATA-03 (deps #11+#12+#15+#17 closed; blocked ONLY on #41):
   trajectory extraction with exact forward-replay verification over
   the regenerated processed_manifest.jsonl + unified splits.
3. #39 MODEL-06 environment side (unblocked by #12): the environment
   candidate-state rewiring + supervised seed labels + TB accounting
   (the policy-side component landed with #25 at 84fd3e7).
4. #36 OPS-01 prepare-only remains user-authorization-gated (never
   claim a connection); an authorized T4 session ALSO closes #25's
   CUDA criterion (one full-profile SynTreePolicy forward, zero
   NaN/Inf on device) — run it first if authorized.

Do NOT redo: #12/#17/#32 (closed with evidence), the #16 outputs
(regenerated full-scale 2026-11-03: 204,005 records — crossdocked/pdbbind
per-source counts match the original exactly, +1 bindingmoad acceptance
attributed to upstream API drift and documented), or anything from
S-V2-10 or earlier (see §3 and the S-V2-10 handoff in git history).
```

## 2. Binding context for every future session

- The branch is a **clean-room rebuild**: emptied at `075f86d`
  (2026-11-01). Never restore pre-reset code; carry forward
  specifications only (D-V2-001/D-V2-002).
- #45 is the permanent constitution: read fully before any action;
  never close it.
- **D-V2-024 (owner directive, 2026-11-02): continuous commit-and-push**
  — commit each bounded coherent unit as soon as its checks pass and
  push immediately; verify the remote ref after every push; never
  force-push; no credentials in prompts/tracked files/logs.
- **Sandbox constraints (S-V2-11 lessons):** background processes do
  NOT survive tool-call boundaries (no nohup pipelines across calls;
  run long work in bounded foreground calls with resumable drivers);
  partial downloads vanish on recycles (verify checksums, resume).
- Pure FP32 on Colab/Kaggle Tesla T4 only; the dev sandbox is CPU-only
  and can never establish GPU claims (#45 §5–§6). The #10 enforcement
  layer (`syntree/utils/hardware.py`) is the canonical gate.
- Issue-scoped module paths and test file names pinned by issue bodies
  are binding (D-V2-002). The S-V2-11 additions:
  `syntree/chemistry/synthon_encoder.py` +
  `scripts/build_synthon_catalog.py` + `tests/test_synthon_encoder.py`
  (#12), `scripts/split_clusters.py` + `scripts/run_mmseqs_split.sh` +
  `tests/test_split_clusters.py` (#17), `tests/suite_a/` (#32),
  `syntree/utils/seeding.py` (the one authoritative seeding module,
  #26/#43 consume it), `syntree/data/shards.py` (the one authoritative
  shard format primitive, #19 builds `ShardedDataset` ON it — no
  re-implementation).
- **#12 semantics are artifact-pinned (D-V2-025)** — consumers must use
  them, not re-derive: seed screening is BLOCK-LEVEL (any #40
  `SELF_REACTIVE_PAIRS` firing between free groups and the six
  uncatalyzed electrophiles kills the whole seed role); synthon
  certification declares the first handle instance; basic-filter order
  MW→rotatable→heavy; MMFF typing checked pre-embedding with fallback
  {B,Zn,Sn}∪{F,Cl,Br,I}; the 2137-D descriptor layout, z-normalization
  (identity for Morgan/vector blocks), and the bit-exact seed-2026 QR
  projection are frozen. The immutable HDF5 (sha256 `8b8c13a0...` on
  HF) remains the authoritative catalog; the S-V2-11 rebuild
  (`00a556ef...`) is a new artifact version.
- Backbone/decoder contracts (D-V2-007/008/011/012), the engine/MDP
  contracts (D-V2-009/015/016/017), the S-V2-09 decisions
  (D-V2-018/019/020), and the catalog decisions (D-V2-021/022/023 +
  the S-V2-11 recovery D-V2-025) are unchanged — see DECISIONS.md.
- The reaction-registry digest binding
  (`32ea6db9...`) still governs catalog certification validity.

## 3. What the next agent must NOT redo

- Everything in the S-V2-10 handoff's section 3 (S-V2-09 and earlier).
- The S-V2-11 work: #17 (split engine + full-scale run, closed), #32
  (Suite A, closed), #12 (encoder + driver + tests + full-scale
  artifact-exact verification, closed), the #16 output regeneration.
- The #12 rebuild artifacts in `work/catalog/` (the encode cache is
  complete: re-running the driver re-assembles the manifest/HDF5 from
  cache in ~1 min without re-encoding).

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| **#25's CUDA acceptance criterion is unverified** | #25/#36 | One full-profile forward on the authorized T4 closes it |
| Docking surrogate (`S_dock`/`S_contact`) is a documented interim | #30/eval | Production closure is #30's remaining scope (needs #29) |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | The #24 torsion head integration is available |
| #12 descriptor reconstruction residuals (quantified) | #12 (closed) | 3 FG slots (tertiary-amine/post-sulfone/all-zero), the 30-D vector class list, and the original ETKDG seed — bounded deltas documented in the module docstring + rebuild manifest; the immutable HDF5 stays authoritative |
| +1 bindingmoad acceptance vs the S-V2-10 run (170,955 vs 170,954) | #16/#17 | Attributed to upstream RCSB/chemcomp API drift; the regenerated manifest is the going-forward input |
| Cavity-volume estimator is a floor-calibration scale | #16 | Real pocket stats are in the regenerated manifest; calibrate for #30 |
| Dev environment not pinned in a lockfile | #26/#43 | Pin when the trainer lands (versions group in the h5 pins the build toolchain) |
| Unstamped states pass field validation (interim) | #19 | Remove when shard producers stamp |
| No GPU/T4 execution, training, or HF artifact sync beyond the catalog | #36/#43 gate | CPU sandbox cannot establish GPU claims |

## 5. Remote runtime / workspace state

No Colab/Kaggle session has been opened (user authorization not
requested; #36 gate). No V2 checkpoints exist. One HF artifact synced:
the #12 catalog dataset `JJKK1212/3d-syntree-v2-catalog` (private,
S-V2-10). No secrets are stored in the repository (session tokens used
only via git remote URLs / per-command env exports, never committed).

Dev-workspace artifacts (not in git; under the session's
`work/` directory, re-creatable via the in-repo drivers):
`raw/` (all four sources + the catalog HDF5, checksum-verified),
`tools/mmseqs/bin/mmseqs` (static binary), `data/processed/`
(204,005-record manifest + accepted-complex artifacts, gzip layout),
`data/splits/` (fasta + cluster TSVs + parquet + split manifest),
`catalog/` (funnel cache + certified pool + encode cache + rebuilt
manifest/HDF5 + sha256 sidecar). #37 (HF sync) uploads them when its
deps close; a fresh sandbox re-creates them via the commands in the
S-V2-11 WORK_LOG entry (~90 min total).

## 6. Session mechanics

1. `git clone --branch 3d-synV2 ...` → confirm HEAD matches the
   recorded commit; `git fetch` first if in doubt (push races
   possible; rebase onto origin, never force-push).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]"`
   && `pip install torch_geometric h5py pandas pyarrow`.
3. Baseline the suite: `python -m pytest tests/ -q` — expected
   **1303 passed / 16 skipped** as of S-V2-11 (942 at S-V2-10 + 18
   #17 + 293 Suite A + 50 #12).
4. Work per the #45 §18 per-issue protocol; follow D-V2-024
   (commit+push each verified unit immediately); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export `GH_TOKEN` per command; no background
   processes across calls.
