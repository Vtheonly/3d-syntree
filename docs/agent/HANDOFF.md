# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-04, session S-V2-18 close (**the FLOW track
landed and closed: #29 FLOW-01** — `LogPartitionHead` (the
seed-conditioned 2-layer logZ MLP, lr_Z = 1e-2), the differentiable
tensor log-probability path, `GFlowNetTrainer`, and the production
rollout bridge `syntree/engine/rollout.py` @ `6eccd41`, 26 tests, both
acceptance criteria verified on the 19-state DAG; **#30 FLOW-02** — the
docking-backend injection point @ `90b1035`, 12 tests; **#31 FLOW-03**
— `scripts/train_stage2.py` + `configs/stage2_colab_t4.json` @
`6760b3d`, 6 CLI tests; suite **1613/20**; clean-room closures **31/40**.
Also reconciled this session: the previously unrecorded S-V2-16/17
sessions — **#25 and #33 are CLOSED on GitHub with the authorized-T4
evidence** — and #20 in the user's prompt was already genuinely closed;
it was not redone.)

## 1. Continuation instruction

```
Next session — two parallel tracks; pick per authorization state:

A. CPU track (no authorization needed): #34 EVAL-01 is the
   dependency-ready critical-path issue (deps #29 ✓ closed S-V2-18,
   #12 ✓ closed; its external dependency is AiZynthFinder availability
   — check installability first; if unavailable, mark blocked with the
   precise reason and move to #44's preparatory scope or #35 prep).
   Then #44 (both AiZynthFinder tracks) → #35 → #37.

B. Authorized-hardware track (the operator must confirm a live Colab
   runtime per #45 §29.1 — re-issue the exact `colab ssh` commands and
   stop at the authentication boundary; S-V2-16/17 already captured a
   session `syntree-t4`, the T4/VRAM verification, the §28.4 preflight
   and the fwd/bwd peaks, so the REMAINING criteria are):
     1. the #26 data reconstruction on the runtime (raws →
        preprocessing → splits → index → trajectories → catalog, per
        docs/data/trajectories/README.md), then the pinned canonical
        launch:
        python scripts/train_stage1.py --config configs/train_colab_t4.json \
          --trajectories <dir> --processed <dir> --catalog <h5> \
          --run-dir /content/runs/stage1_t4 --device cuda \
          --time-budget-seconds 8400
        → closes #26's two criteria (monotonic 2-epoch descent; peak
        reserved VRAM < 9.5 GB with zero OOM skips) and #36's launch
        criteria.
     2. `pytest tests/suite_b/ -v` on the runtime (the memory battery
        already ran there once — re-run for the record).
     3. #43's cross-session continuation: stop at the time budget,
        reconnect in a later authorized session, resume, verify the
        exact next batch — then Stage-2 chaining via
        `scripts/train_stage2.py --resume-auto` (now implemented).
     4. #42's T4 benchmarking of the terminal refinement oracle.

If the operator has NOT authenticated: do track A. Do NOT claim any
GPU execution; the dev sandbox is CPU-only by constitution (#45 §5–§6).

Do NOT redo: #29/#30/#31 (closed with evidence — rerun
`pytest tests/test_gflownet_tb.py tests/test_reward_docking_backend.py
tests/test_stage2_cli.py -q` if in doubt: 44 passed), #27/#20/#18/#19/
#41/#39/#25/#33 (closed), the #26 implementation (only its T4
training-run criteria remain), the #36 preparation + the S-V2-16/17
authorized-session evidence, or anything from S-V2-15 and earlier
(WORK_LOG §S-V2-14/15).
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
| **#26's two training-run T4 criteria are unverified** | #26/#36 | The canonical `train_stage1.py` launch on the authorized runtime (monotonic 2-epoch descent; peak reserved VRAM < 9.5 GB over the full run). The micro-batch VRAM signal (3.758 GiB peak reserved) is already measured; the data reconstruction on the runtime was in progress at S-V2-17 |
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

No Colab/Kaggle session is currently open from THIS sandbox (the
S-V2-16/17 authorized session's CLI auth does not survive the sandbox
recycle; the operator must re-authenticate per #45 §29.1 — the exact
commands are re-issued in the session reply and
`docs/ops/colab_ssh_workflow.md`). No V2 checkpoints exist outside the
test fixtures. One HF artifact synced: the #12 catalog dataset
`JJKK1212/3d-syntree-v2-catalog` (private, S-V2-10). No secrets are
stored in the repository.

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
