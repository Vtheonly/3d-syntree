# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-05, session S-V2-20 (closed at commit `<this
session's final commit>`): the **CPU evaluation critical path is
complete** — #34 EVAL-01 and #44 EVAL-03 closed with real-planner
evidence (the 194-record / 30-unique-structure #18 corpus through the
isolated public-USPTO AiZynthFinder with the two never-merged stock
tracks; the evidence lives in `docs/eval/`), and #35 EVAL-02 is
implemented + harness-verified on the frozen 33-target test manifest
(the TargetDiff/DiffSBDD/SyntheMol baseline structures await operator
export).  Suite: **1683 passed / 22 skipped**.  Clean-room closures:
**33 of 40**.

## 1. Continuation instruction

```
Next session — everything remaining is operator-gated; choose by what
the operator provides:

A. If the operator authenticates the Colab T4 runtime (§29.1 — the
   agent must NOT authenticate as the user; re-issue the handoff in
   docs/ops/colab_ssh_workflow.md): run the S-V2-19 staged chain —
   scripts/runtime_restore_from_hf.sh → the D-V2-043 precheck →
   pytest tests/suite_b/ → scripts/runtime_train_chain.sh (T1–T4).
   That single chain closes #26 (both training-run criteria), #36
   (the remaining session criteria) and #43 (the measured profile +
   session-safe resume), and rides #42's T4 benchmark.  Set HF_TOKEN
   in the runtime env (never embedded — D-V2-024).

B. If the operator exports baseline structures (TargetDiff/DiffSBDD/
   SyntheMol: one SDF per frozen-target id — see the manifest
   docs/eval/evidence/frozen_targets.json): 
   scripts/run_comparative_benchmark.py --targets <manifest>
   --method NAME=DIR ... --retro-track stock_aligned=...
   --retro-track independent_external=... closes #35.  The public
   AiZynthFinder models + stocks live in the sandbox scratch
   (/home/z/my-project/scratch/aizynth_data + eval/) — re-download
   per docs/eval/retrosynthesis_tracks.md if recycled.

C. If neither: #28 needs the owner's licensed-reaction-data decision;
   #37 remains blocked on #26 + #35. There is NO dependency-ready
   code work left — do not invent work; verify state, reconcile
   documentation, and end with an honest handoff.

Do NOT redo: the S-V2-20 evaluation stack (evaluator.py, comparative.py,
the five scripts, the three batteries, docs/eval/, the evidence runs),
#34/#44 (closed with evidence), or anything from S-V2-19 and earlier
(WORK_LOG §S-V2-14..20). The dev sandbox is CPU-only by constitution
(#45 §5–§6); the persistent python env is /home/z/my-project/evalenv
(in-repo .venv dirs are wiped by the sandbox recycler — D-V2-044).
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
- **Sandbox constraints:** the recycler wipes in-repo `.venv` (use
  `/home/z/my-project/evalenv`); background processes do NOT survive
  tool-call boundaries (bounded foreground calls with resumable
  drivers only — the evaluation runs use the aizynthcli --checkpoint
  protocol the same way); `pkill -f aizynthcli` before long planner
  calls (orphaned planners hold the 2.7 GB ZINC stock in RAM).
- Pure FP32 on Colab/Kaggle Tesla T4 only; the dev sandbox is CPU-only
  and can never establish GPU claims (#45 §5–§6). The #10 enforcement
  layer (`syntree/utils/hardware.py`) is the canonical gate.
- **The #18 full-scale census is binding** (`docs/data/trajectories/
  README.md`): 194 accepted trajectories over 170,955 complexes —
  the certified-BB reconstructibility of crystal ligand space is
  ~0.11%, an honest property of the catalog/seed-vocabulary
  intersection, not a defect to silently "fix". Stage-1 training data
  = those 194 trajectories (+ the #28 pretraining track for scale).
  **Evaluation corpus note (D-V2-045): the 194 records cover 30
  unique ligand structures — solve-rate denominators are the unique
  structures, the multiplicity is recorded, never hidden.**
- **#20 semantics are binding** (D-V2-035): the composed augmenter is
  stereo-preserving by rejection resampling; never remove the guard.
- **#26/#27 contracts are binding** (D-V2-030..039) — see the S-V2-19
  handoff section 2 items (supervised states are crystal intermediates
  in the ligand-centroid frame; the growth-handle pool is molecule-side
  reactant-form graft matching (D-V2-043); parity labels from crystal
  CIP; trainer masks from the immutable catalog H5; the curriculum's
  milestone machine consumes ONLY the val split).
- **The FLOW-track contracts are binding (D-V2-040/041/042)** (Stage-2
  optimizer groups, --resume-auto semantics, rollout conventions, the
  frozen docking backend injection point).
- **The evaluation-stack contracts are binding (D-V2-044/045)**: the
  retrosynthesis tracks are never merged; planner targets are unique
  structures; stock mismatches are never planner failures; planner
  outputs are reused only on exact target-set match; the evaluation
  env runs aizynthfinder 4.4.1 under rdkit 2026.09.1 (runtime-verified
  — D-V2-044) with pyarrow pinned <22.
- Backbone/decoder contracts (D-V2-007/008/011/012), the engine/MDP
  contracts (D-V2-009/015/016/017 + Property 3b), the S-V2-09 decisions
  (D-V2-018/019/020), the catalog decisions (D-V2-021/022/023/025) and
  the S-V2-12..15 records (D-V2-026..039) are unchanged — see
  DECISIONS.md.

## 3. What the next agent must NOT redo

- Everything in the S-V2-14/15/16/17/18/19 handoffs (the FLOW track,
  the T4 battery fixes, the bit-exact reconstruction, the D-V2-043
  fix, the runtime drivers — see WORK_LOG §S-V2-14..19).
- **The S-V2-20 work**: the evaluation stack
  (`syntree/engine/evaluator.py` + `syntree/engine/comparative.py` +
  `scripts/run_benchmarks.py` / `export_aizynth_stock.py` /
  `prepare_aizynth_tracks.py` / `make_frozen_targets.py` /
  `run_comparative_benchmark.py` + the three test batteries +
  `docs/eval/` incl. the evidence copies) — #34/#44 are closed with
  evidence; #35's harness, frozen manifest and 3D-SynTree side are
  verified (only the operator's baseline structures are missing).

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| **#26's two training-run T4 criteria are unverified** | #26/#36 | The staged `scripts/runtime_train_chain.sh` chain rides the authorized runtime (unchanged from S-V2-19) |
| **#36's remaining session criteria** | #36/operator | The remote training launch + persisted telemetry + the #43 cross-session continuation |
| **#42's T4 benchmark is unmeasured** | #42/#36 | The oracle is implemented, tested, and consumed end-to-end by the #29 rollout driver |
| **#35's baselines** | operator | The harness is verified; TargetDiff/DiffSBDD/SyntheMol structures (one SDF per target_id) close it |
| Stage-1 training set is 194 trajectories (the honest census) | #28/owner | The #28 two-tier pretraining is the designed scale path; externally blocked on licensed reaction data |
| The Vina adapter is implemented but unexercised in the evidence run | #34/#35 | Optional backend, fail-closed; no docking score is claimed anywhere |
| rxn_15 pattern looser than its reactant template (D-V2-029) | owner/#11 audit | Only changes which failure code a complex gets; not acted on |
| Dev environment not pinned in a lockfile | #26/#43 | The evaluation-env reconciliation is documented (D-V2-044); pin when the trainer lands on the T4 |
| Commit-author name drift on `9197619` | cosmetic | Same email; do not force-push to fix |

## 5. Remote runtime / workspace state

The authorized `syntree-t4` session is NOT open (operator-gated; the
sandbox has no ssh/colab CLI and §29.1 forbids agent-side
authentication).  Everything the training run needs is persisted: the
complete 388-artifact bundle on HF (`JJKK1212/3d-syntree-v2-runs`
`data/training_bundle.tar`) + the catalog dataset
(`JJKK1212/3d-syntree-v2-catalog`); a recycle costs ~10 min via
`scripts/runtime_restore_from_hf.sh`.  No V2 checkpoints exist outside
the test fixtures.  No secrets are stored in the repository.

Dev-workspace state (not in git): the evaluation workspace lives under
`/home/z/my-project/scratch/` — the AiZynthFinder public models + the
extracted ZINC stock (`aizynth_data/`, re-downloadable per
`docs/eval/retrosynthesis_tracks.md`), the HF bundle/catalog copies
(`hf/`), and the run workspace (`eval/`: corpus, stocks, track
configs, planner outputs, reports).  The committed evidence copies
(`docs/eval/evidence/`) pin the expected outputs (counts + SHA-256s).

## 6. Session mechanics

1. `git clone --branch 3d-synV2 …` → confirm HEAD matches the
   recorded commit; `git fetch` first if in doubt (push races
   possible; rebase onto origin, never force-push).
2. Dev env: **`/home/z/my-project/evalenv`** (the recycler wipes
   in-repo `.venv`): `python3 -m venv /home/z/my-project/evalenv &&
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   && pip install -e ".[test]" && pip install h5py pandas pyarrow
   psutil huggingface_hub aizynthfinder "pyarrow==16.1.0" posebusters
   vina meeko gemmi && pip install rdkit==2026.9.1` (D-V2-044).
3. Baseline the suite with `python -m pytest tests/ -q` — expected
   **1683 passed / 22 skipped** (catalog present; +3 planner-gated
   tests re-enable with `SYNTREE_AIZYNTH_MODELS` pointing at the
   public-models dir; the 2 `work/catalog/…`-path gated tests
   re-enable by symlinking the catalog into `work/catalog/`).
4. Work per the #45 §18 per-issue protocol; follow D-V2-024
   (commit+push each verified unit immediately); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export tokens per command; no background processes
   across calls; `pkill -f aizynthcli` before long planner runs.
