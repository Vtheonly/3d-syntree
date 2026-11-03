# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-03, session S-V2-14 close (**#20 DATA-05
closed** — augmentation with the measured stereocentre-flip finding and
the rejection-resampling guarantee, `bc41319`; **#26 TRAIN-01
implemented + CPU-verified** — the supervised training loader + the
pure-FP32 Profile-A `ResilientTrainer` + the canonical CLI
`scripts/train_stage1.py`, `e009127`, 40 tests + end-to-end CLI smoke;
**#36 OPS-01 prepared** — `notebooks/run_3d_syntree.ipynb` +
`docs/ops/colab_ssh_workflow.md`, `39e9a41`, awaiting the operator's
`colab ssh` authentication; #18/#19 were closed by S-V2-13 and the docs
are now reconciled; suite **1482/19**; clean-room closures **23/40**).

## 1. Continuation instruction

```
Next session: the single highest-value step is the AUTHORIZED COLAB T4
SESSION (#36). The operator has been given the exact commands (final
reply of S-V2-14 + docs/ops/colab_ssh_workflow.md):

  1. The operator runs `colab ssh` (or the ProxyCommand form) in their
     LOCAL terminal, authenticates with their own Google account via
     the link Colab prints, and confirms authorization in chat.
  2. Only after that confirmation: connect, run the #45 §29.2
     verification battery (nvidia-smi, free -h, df -h, torch CUDA,
     exact-commit check), the §28.4 preflight, the GPU fwd/bwd smoke,
     then the canonical Stage-1 launch:
       python scripts/train_stage1.py --config configs/train_colab_t4.json \
         --trajectories <dir> --processed <dir> --catalog <h5> \
         --run-dir /content/runs/stage1_t4 --device cuda \
         --time-budget-seconds 8400
  3. That one session closes: #25's CUDA forward criterion, #26's two
     T4 criteria (monotonic 2-epoch descent; peak reserved VRAM < 9.5 GB
     — measure, never assume), #36's session criteria, and unblocks
     #27/#33/#43. If the T4 run diverges: TrainingDivergenceError is
     fail-closed BY DESIGN — diagnose, fix in source, push, re-pull;
     never add a shield.
  4. If the operator has NOT authenticated: work #27 TRAIN-02 (needs
     #26 — implementable now against the CPU-verified trainer; the
     quantile curriculum engine with per-level floor >= 0.03), then
     #33 TEST-02 (needs #7/#8/#21/#25/#26/#32 — #25/#26 are implemented
     + CPU-verified; its Suite-B batteries over the trainer are
     runnable on CPU, the T4 memory profile rides #36).

Do NOT redo: #20/#18/#19/#41/#39 (closed with evidence), the #26
implementation (landed + CPU-verified; only the T4 criteria remain),
the #36 preparation (landed; only the authorized session remains), or
anything from S-V2-13 and earlier (§3).
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
  hosts (mmseqs.com worked) or the API.
- Pure FP32 on Colab/Kaggle Tesla T4 only; the dev sandbox is CPU-only
  and can never establish GPU claims (#45 §5–§6). The #10 enforcement
  layer (`syntree/utils/hardware.py`) is the canonical gate; the
  trainer asserts its RuntimeConfig (fp32, no autocast, no
  GradScaler).
- **The #18 full-scale census is binding** (`docs/data/trajectories/
  README.md`): 194 accepted trajectories over 170,955 complexes — the
  certified-BB reconstructibility of crystal ligand space is ~0.11%,
  an honest property of the catalog/seed-vocabulary intersection, not
  a defect to silently "fix". Stage-1 training data = those 194
  trajectories (+ the #28 pretraining track for scale).
- **#20 semantics are binding** (D-V2-035): the composed augmenter is
  stereo-preserving by rejection resampling — naive σ=0.15 Å jitter
  measurably inverts stereocentres; never remove the guard.
- **#26 contracts are binding** (D-V2-030..036): supervised states are
  crystal intermediates in the ligand-centroid frame with fail-closed
  record/artifact cross-checks; the growth-handle pool is
  molecule-side reaction-template matches at non-junction atoms (RDKit
  reaction-template mols SEGFAULT GetSubstructMatches — parse the
  reactant SMARTS string instead); parity labels come from the crystal
  CIP; empty-pool terminal states carry no stop loss (forced STOP);
  the trainer's masks come from the immutable catalog H5 (seed-only
  rows are never legal retrieval targets).
- Backbone/decoder contracts (D-V2-007/008/011/012), the engine/MDP
  contracts (D-V2-009/015/016/017 + Property 3b), the S-V2-09 decisions
  (D-V2-018/019/020), the catalog decisions (D-V2-021/022/023/025) and
  the S-V2-12/13 records (D-V2-026..029) are unchanged — see
  DECISIONS.md.

## 3. What the next agent must NOT redo

- Everything in the S-V2-13 handoff's section 3 (S-V2-12 and earlier).
- The S-V2-14 work: #20 (closed with evidence), the #26 implementation
  (trainer + supervised loader + config + CLI; only the T4 criteria
  remain — rerun `pytest tests/test_trainer.py tests/test_supervised_dataset.py`
  if in doubt: 40/40), the #36 preparation (notebook + workflow docs),
  the README status reconciliation, and this session's decisions
  (D-V2-030..036).

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| **#25's CUDA forward criterion is unverified** | #25/#36 | One full-profile forward on the authorized T4 closes it (the #26 smoke cell in the notebook does exactly this) |
| **#26's two T4 criteria are unverified** | #26/#36 | Monotonic 2-epoch descent + peak reserved VRAM < 9.5 GB — measured on the authorized session only |
| Stage-1 training set is 194 trajectories (the honest census) | #28/owner | The #28 two-tier pretraining is the designed scale path; catalog extension is the owner's call |
| Docking surrogate (`S_dock`/`S_contact`) is a documented interim | #30/eval | Production closure is #30's remaining scope (needs #29) |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | The #24 torsion head integration is available |
| rxn_15 pattern looser than its reactant template (D-V2-029) | owner/#11 audit | Only changes which failure code a complex gets; not acted on |
| Dev environment not pinned in a lockfile | #26/#43 | Pin when the trainer lands on the T4 (next session) |
| The 19 accepted 0-step trajectories (whole-ligand seeds) | #29 | SeedAction+STOP under #39 semantics — they enter TB accounting naturally; noted for the #29 implementation |

## 5. Remote runtime / workspace state

No Colab/Kaggle session has been opened (the operator was given the
exact `colab ssh` commands at S-V2-14 close; #36 gate). No V2
checkpoints exist. One HF artifact synced: the #12 catalog dataset
`JJKK1212/3d-syntree-v2-catalog` (private, S-V2-10). No secrets are
stored in the repository (session tokens used only via git remote
URLs / per-command env exports, never committed).

Dev-workspace artifacts (not in git; under `work/`): **lost to the
sandbox recycle at S-V2-14 open** — re-creatable mechanically: raws via
the committed acquisition commands (`docs/data/trajectories/README.md`),
the catalog H5 via `scripts/build_synthon_catalog.py`, the index via
`scripts/build_synthon_index.py` (byte-deterministic), preprocessing /
splits / trajectories via the committed drivers. The committed evidence
copies (`docs/data/…`) pin the expected outputs (counts + SHA-256s).

## 6. Session mechanics

1. `git clone --branch 3d-synV2 …` → confirm HEAD matches the
   recorded commit; `git fetch` first if in doubt (push races
   possible; rebase onto origin, never force-push).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]"`
   && `pip install torch_geometric h5py pandas pyarrow psutil`.
3. Baseline the suite with `python -m pytest tests/ -q` — expected
   **1482 passed / 19 skipped** as of S-V2-14 (3 catalog-artifact skips
   re-enable with `SYNTREE_CATALOG_H5` pointing at a verified catalog).
4. Work per the #45 §18 per-issue protocol; follow D-V2-024
   (commit+push each verified unit immediately); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export tokens per command; no background processes
   across calls.
