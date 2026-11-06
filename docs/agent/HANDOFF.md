# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-06, session S-V2-22 (closing commit: this
docs commit on top of `895d67c`): **#43 closed — the cross-session
continuation executed on a second authorized T4.**  The S-V2-21
account's free tier died mid-session (assign POST 503 outcome:2,
currentBalance 0); a fresh operator account re-consented through the
same split PKCE flow, the bridge re-verified (T4 + §29.2 gate + SSH
battery), the workspace restored bit-exact from HF, and the two-round
T-chain reproduced clean (RUN_TAG repro-r1/repro-r2 — round 2's T1
stopped at the historical step-12/42 signature; all gates PASS, both
runs persisted to HF).  The never-before-executed continuation driver
then ran live and caught two latent defects (**D-V2-049**, fixed at
`895d67c`: quoted-heredoc `${VAR}` NameErrors + the X1 preflight guard
checking the wrong tar path) before completing the true cross-VM run:
history 42 → 63 with **prefix identity**, fresh runtime telemetry
baseline, epoch-3 final loss **bit-identical across VMs**, persisted
as `runs/stage1_t4_session2/run.tar`.  Suite baseline: **1687 passed /
22 skipped** (not re-run — the only delta is the bash-driver fix,
verified by its live X1–X5 execution).  Clean-room closures: **37 of
40** (#5–#27, #29–#34, #38–#44).

## 1. Continuation instruction

```
Next session — NO dependency-ready runtime actions remain; the work
left is operator-gated:

A. (DONE S-V2-22) The #43 cross-session continuation run — executed
   on a fresh second T4 (new operator account): CROSS_SESSION_
   CONTINUATION PASS (42 → 63, prefix identical), NEW_RUNTIME_
   TELEMETRY_BASELINE PASS, persisted as runs/stage1_t4_session2/
   run.tar; two latent driver defects caught + fixed (D-V2-049 at
   895d67c). Evidence: ssh_evidence/repro_r1_r2/ (both T-chain
   rounds, restore, both continuation runs, both VMs' preflights).
   Re-provisioning recipe if a future run needs a T4 again:
     colab new -s syntree-t4 --gpu T4   + scripts/t4_gate.py gate
     + runtime_restore_from_hf.sh  (HF_TOKEN via SFTP /root/.hf_token)
   KNOWN RISK (unchanged): free-tier provisioning can silently return
   CPU-only runtimes (assign endpoint 'Service Unavailable') — gate
   fail-closed on nvidia-smi AND torch-CUDA before any work (#45
   §29.2); the S-V2-22 account-death failure mode (POST 503 outcome:2
   + currentBalance 0 on every attempt) means: ask the operator for a
   fresh account re-consent — never fake or CPU-substitute.

B. If the operator exports baseline structures (TargetDiff/DiffSBDD/
   SyntheMol: one SDF per frozen-target id — see the manifest
   docs/eval/evidence/frozen_targets.json):
   scripts/run_comparative_benchmark.py --targets <manifest>
   --method NAME=DIR ... --retro-track stock_aligned=...
   --retro-track independent_external=... closes #35, and with it #37
   (the final milestone; its sync mechanics are already proven —
   bit-exact restores x3, checksum-verified uploads).

C. If neither: #28 needs the owner's licensed-reaction-data decision;
   there is no other dependency-ready work — verify state, reconcile
   documentation, end with an honest handoff.

Do NOT redo: the S-V2-22 two-round T-chain reproduction, the #43
cross-VM continuation, or the D-V2-049 fix (WORK_LOG §S-V2-22); nor
the S-V2-21 bridge/chain/benchmark work, the S-V2-20 evaluation
stack, or anything earlier. The dev sandbox is CPU-only by
constitution (#45 §5–§6); the persistent python env is
/home/z/my-project/evalenv (in-repo .venv dirs are wiped by the
sandbox recycler — D-V2-044; clear /tmp/pytest-of-z before full-suite
runs — Errno 28 exhaustion, S-V2-21). The colab CLI + paramiko live
in /home/z/.venv (google-colab-cli; ~/.ssh/id_ed25519.pub is the
bridge public key — regenerated from the preserved private key in
/home/z/my-project/.secrets/ if the recycler wipes it; do NOT pass
-i/--identity to `colab ssh`, it shells out to the absent ssh-keygen;
the private key stays in /home/z/my-project/.secrets/, never
committed). Git pushes need the persisted credential helper:
  git -c credential.helper='store --file=/home/z/my-project/.auth/
  git-credentials' push origin 3d-synV2
(the recycler wipes the global gitconfig that carried it).
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
- **The continuation contracts are binding (D-V2-046/047/048)**: the
  profile digest covers per-step training semantics ONLY (operational
  knobs — time budget, workers, telemetry/checkpoint cadence, max_epochs
  — are excluded); a completed-run resume is a no-op (progress.epoch
  advances past a finished epoch — never re-train a completed epoch);
  cross-session continuation may extend max_epochs. All three are pinned
  by regression tests in tests/test_trainer.py.
- Backbone/decoder contracts (D-V2-007/008/011/012), the engine/MDP
  contracts (D-V2-009/015/016/017 + Property 3b), the S-V2-09 decisions
  (D-V2-018/019/020), the catalog decisions (D-V2-021/022/023/025) and
  the S-V2-12..15 records (D-V2-026..039) are unchanged — see
  DECISIONS.md.

## 3. What the next agent must NOT redo

- Everything in the S-V2-14/15/16/17/18/19 handoffs (the FLOW track,
  the T4 battery fixes, the bit-exact reconstruction, the D-V2-043
  fix, the runtime drivers — see WORK_LOG §S-V2-14..19).
- **The S-V2-21 work**: the bridge (`scripts/colab_ssh_client.py` +
  `bootstrap_t4.sh` + the workflow doc), the executed T-chain and its
  fixes (`ee28713`/`925071d`/`6445929`/`778ff1c`), the #42 benchmark
  (`benchmark_torsion_refinement.py` + the on-T4 report), and the
  staged continuation driver — #26/#36/#42 are closed with evidence.
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
| **#43's cross-session continuation run** | #43/agent | Driver staged (`bootstrap_t4.sh session2 continue`); blocked only on free-tier GPU provisioning capacity (CPU-only fallback runtimes observed; the bootstrap gates fail-closed) |
| **#37 blocked on #35** | operator | The sync mechanics are proven (bit-exact restore x3, checksum-verified uploads); only the baseline comparison remains |
| **#35's baselines** | operator | The harness is verified; TargetDiff/DiffSBDD/SyntheMol structures (one SDF per target_id) close it |
| Stage-1 training set is 194 trajectories (the honest census) | #28/owner | The #28 two-tier pretraining is the designed scale path; externally blocked on licensed reaction data |
| The Vina adapter is implemented but unexercised in the evidence run | #34/#35 | Optional backend, fail-closed; no docking score is claimed anywhere |
| rxn_15 pattern looser than its reactant template (D-V2-029) | owner/#11 audit | Only changes which failure code a complex gets; not acted on |
| Dev environment not pinned in a lockfile | #26/#43 | The evaluation-env reconciliation is documented (D-V2-044); pin when the trainer lands on the T4 |
| Commit-author name drift on `9197619` | cosmetic | Same email; do not force-push to fix |

## 5. Remote runtime / workspace state

The bridge is OPERATIONAL (S-V2-21): OAuth refresh-token cache at
`~/.config/colab-cli/token.json` (no operator re-auth needed while it
stays valid), `colab` CLI + paramiko in `/home/z/.venv`, bridge pubkey
at `~/.ssh/id_ed25519.pub` (private key in
`/home/z/my-project/.secrets/colab_ed25519`, never committed). No
session is left open at handoff (VMs burn compute units; `colab stop`
after use). Everything the runtime needs is persisted: the
388-artifact training bundle + the session-1 run
(`JJKK1212/3d-syntree-v2-runs`: `data/training_bundle.tar`,
`runs/stage1_t4_session1/run.tar`) + the catalog
(`JJKK1212/3d-syntree-v2-catalog`); a fresh session restores in ~3 min
via `scripts/runtime_restore_from_hf.sh` (verified bit-exact x3).
Evidence transcripts live in `/home/z/my-project/ssh_evidence/`
(outside git; the summarized numbers are recorded in WORK_LOG/STATUS).

Dev-workspace state (not in git): `/home/z/my-project/evalenv` (the
rebuilt dev env — torch CPU + `-e '.[test]'` + h5py/pandas/pyarrow/
psutil/huggingface_hub + rdkit 2026.9.1; the AiZynthFinder stack is
NOT installed in this rebuild — the 3 planner-gated tests skip, the
baseline was 1684/22 pre-fixes, 1687/22 at `778ff1c`); the catalog H5
at `repo/work/catalog/` (re-downloadable); `repo/work/catalog` +
`~/.cache` sizes matter — clear `/tmp/pytest-of-z` before full-suite
runs (Errno 28). No secrets are stored in the repository.

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
   **1687 passed / 22 skipped** (catalog present at `work/catalog/`;
   +3 planner-gated tests re-enable with `SYNTREE_AIZYNTH_MODELS`
   after installing the aizynthfinder stack per D-V2-044 — without
   it the 1687 baseline holds with those skipped).
4. Work per the #45 §18 per-issue protocol; follow D-V2-024
   (commit+push each verified unit immediately); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export tokens per command; no background processes
   across calls; `pkill -f aizynthcli` before long planner runs.
