# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-05, session S-V2-19 (in progress; see
WORK_LOG §S-V2-19): the authorized-T4 data reconstruction completed
**bit-exact** (204,005 records; splits 136,764/17,096/17,095
zero-leakage; index byte-deterministic; TRAJECTORY_BITEXACT: PASS on
all three pinned .pt SHA-256s); the first canonical training launch
failed closed on a real bug — **D-V2-043** fixed at `b7e0f3d`
(reactant-form graft matching in the molecule-side pool; suite
**1615/20**; the #29 rollout pool corrected too); the runtime-recycle
recovery is engineered and committed (`02f4443`: the sanitized
`scripts/runtime_{full_chain,restore_from_hf,train_chain}.sh` drivers
+ the complete 388-artifact training bundle persisted to HF
`JJKK1212/3d-syntree-v2-runs` `data/training_bundle.tar`).  **The
remaining work is exactly one authorized-T4 session: the training run
closure of #26/#36/#43** — blocked only by free-tier T4 capacity at
session close (the assign endpoint returned Service Unavailable
through the retry window; balance 0.00).

## 1. Continuation instruction

```
Next session — the authorized-hardware closure (everything is staged):

A. The one remaining authorized-hardware step (S-V2-19 left it exactly
   here): when a T4 is assignable, on the runtime run
   `scripts/runtime_restore_from_hf.sh` (env + preflight + catalog +
   the COMPLETE bundle: trajectories + splits + index + 194 ligand AND
   pocket artifacts, checksums re-verified) — then the staged
   launch_final pattern: the D-V2-043 precheck (materialize all 241
   train + 101 val items, 2 shuffle orders) → `pytest tests/suite_b/`
   → `scripts/runtime_train_chain.sh` (T1 time-budget stop; T2
   SIGKILL + sidecar integrity; T3 resume to max_epochs_reached; T4
   monotonic-descent + <9.5 GiB VRAM verification + HF run
   persistence).  That single chain closes #26 (both acceptance
   criteria), #36 (the launch + persisted telemetry + #43 continuation)
   and #43 (the measured profile + session-safe resume).  Set
   HF_TOKEN in the runtime env (never embedded — D-V2-024).  Free-tier
   reality: the assign endpoint may return Service Unavailable for
   hours; poll `colab new -s syntree-t4 --gpu T4` patiently (~2-minute
   intervals), and expect recycles mid-chain (the restore driver turns
   a recycle into ~10 minutes, not ~65).

B. CPU track (if no runtime): #34 EVAL-01 remains the
   dependency-ready critical path (deps closed; the external
   dependency is AiZynthFinder availability — check installability
   first).  Then #44 → #35 → #37.  #42's T4 benchmark rides the next
   authorized session alongside A.

If the operator has NOT authenticated and the CLI token is stale: do
NOT claim any GPU execution; re-issue the §29.1 handoff. The dev
sandbox is CPU-only by constitution (#45 §5–§6).

Do NOT redo: the S-V2-19 work (the bit-exact reconstruction, the
D-V2-043 fix `b7e0f3d`, the bundle/pocket recovery, the committed
runtime drivers), #29/#30/#31 (closed with evidence — rerun
`pytest tests/test_gflownet_tb.py tests/test_reward_docking_backend.py
tests/test_stage2_cli.py -q` if in doubt: 44 passed),
#27/#20/#18/#19/#41/#39/#25/#33 (closed), or anything from S-V2-18
and earlier (WORK_LOG §S-V2-14..18).
```

## 2. Binding context for every future session

- The branch is a **clean-room rebuild**: emptied at `075f86d`
  (2026-11-01). Never restore pre-reset code; carry forward
  specifications only (D-V2-001/D-V2-002).
- #45 is the permanent constitution: read fully before any action;
  never close it.
- **D-V2-024 (owner directive): continuous commit-and-push** — commit
  each bounded coherent unit as soon as its checks pass and push
  immediately; verify the remote ref after every push; never
  force-push; no credentials in prompts/tracked files/logs.
- **Sandbox constraints:** background processes do NOT survive
  tool-call boundaries (bounded foreground calls with resumable
  drivers only); partial downloads vanish on recycles (verify
  checksums, resume). GitHub release-asset URLs 404 — use upstream
  hosts or the API. The dev sandbox recycles between sessions: `work/`
  artifacts do not survive (the committed evidence copies pin the
  expected outputs).
- Pure FP32 on Colab/Kaggle Tesla T4 only; the dev sandbox is CPU-only
  and can never establish GPU claims (#45 §5–§6). The #10 enforcement
  layer (`syntree/utils/hardware.py`) is the canonical gate; both
  trainers assert their RuntimeConfig.
- **The #18 full-scale census is binding** (`docs/data/trajectories/
  README.md`): 194 accepted trajectories over 170,955 complexes — the
  certified-BB reconstructibility of crystal ligand space is ~0.11%,
  an honest property of the catalog/seed-vocabulary intersection, not
  a defect to silently "fix". Stage-1 training data = those 194
  trajectories (+ the #28 pretraining track for scale).
- **#20 semantics are binding** (D-V2-035): the composed augmenter is
  stereo-preserving by rejection resampling; never remove the guard.
- **#26/#27 contracts are binding** (D-V2-030..039): supervised states
  are crystal intermediates in the ligand-centroid frame; the
  growth-handle pool is molecule-side reaction-template matches at
  non-junction atoms (RDKit reaction-template mols SEGFAULT
  GetSubstructMatches — parse the reactant SMARTS string instead);
  parity labels come from the crystal CIP; the trainer's masks come
  from the immutable catalog H5; the curriculum's milestone machine
  consumes ONLY the val split; `--curriculum` is additive (the pinned
  #26 acceptance launch is unchanged).
- **The FLOW-track contracts are binding (new, D-V2-040/041/042):**
  the Stage-2 optimizer is two AdamW groups (policy 3e-4/1e-5
  continuing Profile A; the logZ head at the #29-pinned lr_Z = 1e-2,
  no weight decay); Stage 2 IS GFlowNet-TB — the legacy
  `main.py --mode rl` maps to `scripts/train_stage2.py` and no
  `--algorithm` flag exists; `--resume-auto` never trains from random
  weights (resume Stage-2 → else Stage-1 → else refuse). The rollout
  conventions: capped-pocket frame (env + policy anchor menus must
  coincide), placed-seed-centroid growth states, molecule-side pool ∩
  env capacity, the parity pre-step determination via a cloned env
  state (post-hoc fail-closed), block-side HandleSpecs from
  string-parsed reactant SMARTS (never template objects), seed blocks
  via the molecule-side construction. The reward's docking backend is
  frozen at construction (`ThreeDReward(docking_scorer=...)`,
  `DockingResult` contract; the default surrogate is bit-identical to
  the pre-injection computation).
- Backbone/decoder contracts (D-V2-007/008/011/012), the engine/MDP
  contracts (D-V2-009/015/016/017 + Property 3b), the S-V2-09 decisions
  (D-V2-018/019/020), the catalog decisions (D-V2-021/022/023/025) and
  the S-V2-12..15 records (D-V2-026..039) are unchanged — see
  DECISIONS.md.

## 3. What the next agent must NOT redo

- Everything in the S-V2-14/15 handoffs' section 3 (S-V2-13 and
  earlier).
- The S-V2-16/17 work (the authorized T4 battery fixes `cac741e`, the
  train-01 removeHs fix `b923960`, the demo work, the #25/#33 closures
  with T4 evidence) — reconciled into STATUS/DEPENDENCIES this session.
- The S-V2-18 work: #29 (`6eccd41`), #30 (`90b1035`), #31 (`6760b3d`)
  — all closed with evidence; the batteries are
  `tests/test_gflownet_tb.py` (26), `tests/test_reward_docking_backend.py`
  (12), `tests/test_stage2_cli.py` (6).

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| **#26's two training-run T4 criteria are unverified** | #26/#36 | The canonical `train_stage1.py` launch on the authorized runtime (monotonic 2-epoch descent; peak reserved VRAM < 9.5 GB over the full run). The S-V2-19 loader blocker (D-V2-043) is FIXED (`b7e0f3d`) and the launch is staged (`scripts/runtime_train_chain.sh`); only the T4 capacity is missing |
| **#36's remaining session criteria** | #36/operator | The remote training launch + persisted training telemetry + the #43 cross-session continuation (the authorized connection, T4/VRAM verification, preflight and fwd/bwd peaks are captured) |
| **#42's T4 benchmark is unmeasured** | #42/#36 | The oracle is implemented, tested, and now consumed end-to-end by the #29 rollout driver; the benchmark needs the runtime |
| Stage-1 training set is 194 trajectories (the honest census) | #28/owner | The #28 two-tier pretraining is the designed scale path; **externally blocked on licensed reaction data — owner decision** |
| #34's AiZynthFinder availability is unverified | #34 | The next critical-path issue; check installability before committing to a plan |
| Docking surrogate remains the active backend | #34/#35 | The #30 injection point exists and is tested; the real engine (evaluation stack) injects there |
| rxn_15 pattern looser than its reactant template (D-V2-029) | owner/#11 audit | Only changes which failure code a complex gets; not acted on |
| Dev environment not pinned in a lockfile | #26/#43 | Pin when the trainer lands on the T4 |
| The 19 accepted 0-step trajectories (whole-ligand seeds) | closed with #29 | SeedAction+STOP enter TB accounting naturally (the rollout driver handles single-candidate pools; verified in the driver battery) |
| Commit-author name drift on `9197619` | cosmetic | Same email; do not force-push to fix |

## 5. Remote runtime / workspace state

The authorized `syntree-t4` session is NOT currently open (free-tier
T4 capacity exhausted at S-V2-19 close; poll `colab new -s syntree-t4
--gpu T4`).  The CLI auth survives via the OAuth bridge backups
(`scripts/colab_token_backup.json` — restore per
`scripts/colab_ensure.sh`).  **Everything the training run needs is
persisted**: the complete 388-artifact training bundle on HF
(`JJKK1212/3d-syntree-v2-runs` `data/training_bundle.tar` — the three
bit-exact .pt trajectory files + splits + index + 194 ligand AND
pocket artifacts) and the catalog dataset
(`JJKK1212/3d-syntree-v2-catalog`); a runtime recycle costs ~10
minutes via `scripts/runtime_restore_from_hf.sh`.  No V2 checkpoints
exist outside the test fixtures.  No secrets are stored in the
repository (the runtime drivers read `HF_TOKEN` from the environment,
D-V2-024).

Dev-workspace artifacts (not in git; under `work/`): lost to the
sandbox recycle — re-creatable mechanically: raws via the committed
acquisition commands (`docs/data/trajectories/README.md`), the catalog
H5 via `scripts/build_synthon_catalog.py`, the index via
`scripts/build_synthon_index.py` (byte-deterministic), preprocessing /
splits / trajectories via the committed drivers. The committed evidence
copies (`docs/data/…`) pin the expected outputs (counts + SHA-256s).

The protein→molecule demo (S-V2-17) lives in git
(`demo/protein_to_molecule/` incl. `protein_structure_3d/`); the user's
zipped deliverable lived in the old sandbox `download/` dir.

## 6. Session mechanics

1. `git clone --branch 3d-synV2 …` → confirm HEAD matches the
   recorded commit; `git fetch` first if in doubt (push races
   possible; rebase onto origin, never force-push).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]"`
   && `pip install torch_geometric h5py pandas pyarrow psutil`.
3. Baseline the suite with `python -m pytest tests/ -q` — expected
   **1613 passed / 20 skipped** as of S-V2-18 (3 catalog-artifact skips
   re-enable with `SYNTREE_CATALOG_H5` pointing at a verified catalog;
   1 Suite-B skip = the T4 memory measurement, executed on the
   authorized runtime).
4. Work per the #45 §18 per-issue protocol; follow D-V2-024
   (commit+push each verified unit immediately); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export tokens per command; no background processes
   across calls.
