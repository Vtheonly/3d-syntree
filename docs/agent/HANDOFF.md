# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-04, session S-V2-15 close (**#27 TRAIN-02
CLOSED** — the Approach C quantile curriculum engine
`syntree/engine/curriculum.py` + trainer/CLI integration, `9197619`,
40 tests incl. the `--curriculum` CLI smoke, both acceptance criteria
verified, D-V2-037/038/039; **#33 TEST-02 implemented + CPU-verified**
— the five Suite-B batteries under `tests/suite_b/`, `7a6ea3b`,
`pytest tests/suite_b/ -v` → 47 passed / 1 skipped, the single skip
being the battery-4 T4 memory measurement gated on the authorized #36
session; **#20 independently re-audited** (fresh 34/34 + baseline
suite reproduction — the S-V2-14 closure is genuine); **#36 verified
prepared** — the auth handoff re-issued to the operator; suite
**1569/20**; clean-room closures **24/40**).

## 1. Continuation instruction

```
Next session: the single highest-value step is STILL the AUTHORIZED
COLAB T4 SESSION (#36) — it is now even denser: one authorized run
closes #25's CUDA forward criterion, #26's two T4 criteria (monotonic
2-epoch descent; peak reserved VRAM < 9.5 GB — measure, never
assume), #33's remaining criterion (run `pytest tests/suite_b/ -v` on
the T4 → expect 48/48 → close #33 with the measured peak-VRAM figure),
#36's session criteria, and unblocks #43's measured profiling +
cross-session resume. The operator was re-issued the exact commands
(S-V2-15 final reply + docs/ops/colab_ssh_workflow.md):

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
     (add `--curriculum` for the #27 production form — D-V2-039; the
     bare form above is the pinned #26 acceptance launch).
  3. If the T4 run diverges: TrainingDivergenceError is fail-closed BY
     DESIGN — diagnose, fix in source, push, re-pull; never add a
     shield.

If the operator has NOT authenticated, the CPU-side critical path is:

  a. #29 FLOW-01's remaining scope: the seed-conditioned logZ MLP +
     GFlowNet trainer integration (its TB mathematics + uniform-leaf
     backward semantics are closed — #38/D-V2-009 — and are now pinned
     by Suite B battery 5; #25/#26 are implemented + CPU-verified, so
     the hard blockers are gone in practice).
  b. #28 TRAIN-03 is code-ready but EXTERNALLY BLOCKED on licensed,
     provenance-tracked reaction data — an owner decision, not an
     engineering task.
  c. #43's resume contract is already implemented in the trainer
     (atomic checkpoints with RNG + curriculum state); only its
     measured T4 demonstration remains, riding #36.

Do NOT redo: #27/#20/#18/#19/#41/#39 (closed with evidence), the #26
implementation (only the T4 criteria remain), the #33 batteries
(only the on-T4 execution remains), the #36 preparation (only the
authorized session remains), or anything from S-V2-13 and earlier
(WORK_LOG §S-V2-14/13).
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
- **#26/#27 contracts are binding** (D-V2-030..036 + 037/038/039):
  supervised states are crystal intermediates in the ligand-centroid
  frame; the growth-handle pool is molecule-side reaction-template
  matches at non-junction atoms (RDKit reaction-template mols SEGFAULT
  GetSubstructMatches — parse the reactant SMARTS string instead);
  parity labels come from the crystal CIP; empty-pool terminal states
  carry no stop loss; the trainer's masks come from the immutable
  catalog H5 (seed-only rows are never legal retrieval targets); the
  curriculum's milestone machine consumes ONLY the val split
  (fail-closed wiring), its checkpoint state rides the trainer payload
  key `"curriculum"` with symmetric fail-closed restore, and its
  epoch draws are stateless-deterministic per (seed, stage, epoch).
- Backbone/decoder contracts (D-V2-007/008/011/012), the engine/MDP
  contracts (D-V2-009/015/016/017 + Property 3b), the S-V2-09 decisions
  (D-V2-018/019/020), the catalog decisions (D-V2-021/022/023/025) and
  the S-V2-12/13/14 records (D-V2-026..036) are unchanged — see
  DECISIONS.md.

## 3. What the next agent must NOT redo

- Everything in the S-V2-13/S-V2-14 handoffs' section 3 (S-V2-12 and
  earlier).
- The S-V2-14 work: #20 (closed with evidence; re-audited green at
  S-V2-15), the #26 implementation (trainer + supervised loader +
  config + CLI; only the T4 criteria remain — rerun
  `pytest tests/test_trainer.py tests/test_supervised_dataset.py` if
  in doubt: 63/63), the #36 preparation (notebook + workflow docs).
- The S-V2-15 work: #27 (closed; rerun `pytest tests/test_curriculum.py
  -q` if in doubt: 40/40), the #33 batteries (rerun
  `pytest tests/suite_b/ -v` if in doubt: 47 passed / 1 hardware-gated
  skip), and this session's decisions (D-V2-037/038/039).

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| **#25's CUDA forward criterion is unverified** | #25/#36 | One full-profile forward on the authorized T4 closes it (the #26 smoke cell in the notebook does exactly this) |
| **#26's two T4 criteria are unverified** | #26/#36 | Monotonic 2-epoch descent + peak reserved VRAM < 9.5 GB — measured on the authorized session only |
| **#33's memory battery is unexecuted** | #33/#36 | `tests/suite_b/test_t4_memory_boundary.py` — CPU contracts green; the peak-VRAM measurement needs the T4 (runs there unchanged) |
| Stage-1 training set is 194 trajectories (the honest census) | #28/owner | The #28 two-tier pretraining is the designed scale path; **externally blocked on licensed reaction data — owner decision** |
| Docking surrogate (`S_dock`/`S_contact`) is a documented interim | #30/eval | Production closure is #30's remaining scope (needs #29) |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | The #24 torsion head integration is available |
| rxn_15 pattern looser than its reactant template (D-V2-029) | owner/#11 audit | Only changes which failure code a complex gets; not acted on |
| Dev environment not pinned in a lockfile | #26/#43 | Pin when the trainer lands on the T4 |
| The 19 accepted 0-step trajectories (whole-ligand seeds) | #29 | SeedAction+STOP under #39 semantics — they enter TB accounting naturally; noted for the #29 implementation |
| Commit-author name drift on `9197619` | cosmetic | `3D-SynTree V2 Agent` vs `autonomous-agent` (same email) — do not force-push to fix |

## 5. Remote runtime / workspace state

No Colab/Kaggle session has been opened (the operator was re-issued
the exact `colab ssh` commands at S-V2-15 close; #36 gate). No V2
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

S-V2-17 addition (in git): the protein→molecule demo now includes
`demo/protein_to_molecule/protein_structure_3d/` — a self-contained
3Dmol.js 2.4.0 WebGL viewer (`protein_viewer.html`) plus headless
Chromium renders (cartoon / active site / ligand overlay / VDW
surface PNG+PDF, 36-frame GIF, protein-as-graph figure) with
`render_provenance.json` fail-closed checks; reproduce via the three
committed `demo/protein_to_molecule/scripts/render_protein_3d*.py` /
`pocket_graph_3d.py` (first run fetches 4ITP from RCSB + 3Dmol.js
from jsDelivr, then offline; needs playwright+Chromium, pillow,
matplotlib). The user's zipped deliverable lives in the sandbox
`download/` dir, not in git.

## 6. Session mechanics

1. `git clone --branch 3d-synV2 …` → confirm HEAD matches the
   recorded commit; `git fetch` first if in doubt (push races
   possible; rebase onto origin, never force-push).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]"`
   && `pip install torch_geometric h5py pandas pyarrow psutil`.
3. Baseline the suite with `python -m pytest tests/ -q` — expected
   **1569 passed / 20 skipped** as of S-V2-15 (3 catalog-artifact skips
   re-enable with `SYNTREE_CATALOG_H5` pointing at a verified catalog;
   1 Suite-B skip = the T4 memory measurement, gated on #36).
4. Work per the #45 §18 per-issue protocol; follow D-V2-024
   (commit+push each verified unit immediately); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export tokens per command; no background processes
   across calls.
