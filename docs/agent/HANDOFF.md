# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-06, session S-V2-24 (**the R3 T4 run
executed on an authorized Tesla T4 — the owner's overfitting watch is
answered with live evidence; D-V2-050 execution**):
`colab_t4_profile_a_r3` ran 11 of 12 epochs and stopped by
**validation early stopping** (patience 5, `min_delta` 0.10) —
per-epoch train means fell 20.77 → 9.39 while validation bottomed at
epoch 6 (**13.0775**) then rose five consecutive epochs (→ 15.46):
**overfitting detected after epoch 6 and handled live**
(`best_state.pt` = the epoch-6 best, `kind=best_val`, full
continuation payload @ step 126; LR cosine verified live over the
frozen 252-step horizon; the anchor monitor in its documented loud
calibration-pending mode).  Provisioning gated fail-closed
(nvidia-smi + torch-CUDA concur Tesla T4 15360 MiB, fp32/tf32-off
SUPPORTED); the workspace restored bit-exact from HF (trajectory
SHA-256s MATCH, independently re-verified); the run persisted
checksum-verified to HF `runs/stage1_t4_r3/run.tar` (LFS sha256
`1dc00a58…`, local == remote); §29.2 battery PASS (transcript in
`ssh_evidence/`); the VM stopped after use.  GPU auth re-completed
via the split-PKCE flow (two failed operator code rounds — one
orphaned by the recycle, one expired `invalid_grant` — then exchanged
OK; never faked); a fresh GitHub PAT + the HF token were supplied and
verified, and the S-V2-23 commits are confirmed on `origin/3d-synV2`
at `f4fb715` (the earlier "push pending" note is superseded).
Clean-room closures stand at **37 of 40** (#5–#27, #29–#34, #38–#44);
the remaining work is entirely operator-gated (#35 baselines →
#35+#37; #28 licensed-data decision).

## 1. Continuation instruction

```
Next session — no runtime action is pending; the remaining work is
operator-gated:

0. (DONE S-V2-24) GPU auth + the R3 T4 run + the evidence persistence
   all executed and recorded (WORK_LOG §S-V2-24; D-V2-050 execution).
   Resumable artifacts on HF: runs/stage1_t4_r3/run.tar — best_state.pt
   (the epoch-6 best, val 13.0775, full continuation payload @ step
   126) + trainer_state.pt (the early-stopped step 231; --resume
   continues from it, D-V2-047 no-op semantics apply to completed
   epochs).

1. THE OPERATOR EXPORTS THE #35 BASELINE STRUCTURES (one SDF per
   target_id — TargetDiff / DiffSBDD / SyntheMol), then:
   scripts/run_comparative_benchmark.py --targets <manifest>
   --method NAME=DIR ... --retro-track stock_aligned=...
   --retro-track independent_external=...
   → closes #35, and with it #37 (the final milestone; its sync
   mechanics are already proven — bit-exact restores x3,
   checksum-verified uploads).

2. #28 NEEDS THE OWNER'S licensed-reaction-data decision.  Context the
   R3 run just measured: on the honest 194-trajectory Stage-1 set
   (241 train / 101 val samples) the model overfits after ~6 epochs
   (val bottomed 13.0775 then rose 5 straight epochs while train fell
   20.77 → 9.39; early stopping fired at patience 5).  More data /
   the #28 pretraining track is the designed answer — not longer
   training on the same set.

3. IF A T4 SESSION IS NEEDED AGAIN: colab_auth_flow.py first
   (`status` — the refresh-token cache may still be valid; else `url`
   → operator code → `code`; never fake), then
   colab new -s syntree-t4 --gpu T4, the fail-closed gate (nvidia-smi
   AND torch-CUDA must concur Tesla T4 — CPU-only fallbacks are
   disposed, never trained on; account-death = POST 503 outcome:2 +
   currentBalance 0 → fresh operator re-consent), SFTP the HF token to
   /root/.hf_token, bash scripts/runtime_restore_from_hf.sh, then
   launch train_stage1.py --config configs/train_colab_t4_r3.json …
   VMs burn compute units — colab stop after use.  Operational note:
   a detached nohup launch holds the paramiko channel open (the
   launch call itself may time out — harmless; poll via fresh exec
   calls).

Do NOT redo: the S-V2-24 R3 T4 execution (persisted; the overfitting
verdict recorded), the S-V2-23 R3 implementation (CPU-verified), the
S-V2-22 two-round T-chain reproduction, the #43 cross-VM continuation,
or the D-V2-049 fix (WORK_LOG §S-V2-22); nor the S-V2-21
bridge/chain/benchmark work, the S-V2-20 evaluation stack, or
anything earlier. The dev sandbox is CPU-only by constitution
(#45 §5–§6); the persistent python env is /home/z/my-project/evalenv
(in-repo .venv dirs are wiped by the sandbox recycler — D-V2-044;
clear /tmp/pytest-of-z AND /tmp/syntree_trainer_test_* before AND
after full-suite runs — the S-V2-23 Errno-28 disk-exhaustion incident
filled 9.9 GB to 100%). The colab CLI + paramiko live in
/home/z/.venv (google-colab-cli; ~/.ssh/id_ed25519.pub is the bridge
public key — regenerate a FRESH keypair via scripts/gen_bridge_key.py
when the recycler wipes ~/.ssh AND .secrets/ (S-V2-24 proved a fresh
keypair always works — the pubkey rides the session-creation header,
no runtime pre-registration needed); do NOT pass -i/--identity to
`colab ssh`, it shells out to the absent ssh-keygen; the private key
stays in /home/z/my-project/.secrets/, never committed). Git pushes
need the persisted credential helper:
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
- **The S-V2-22/23/24 work**: the #43 cross-VM continuation + the
  D-V2-049 driver fix (S-V2-22); the R3 profile extension (D-V2-050,
  `bcddf7b`, CPU-verified S-V2-23); and the R3 T4 execution with the
  overfitting verdict + checksum-verified HF persistence (S-V2-24 —
  `runs/stage1_t4_r3/run.tar`; WORK_LOG §S-V2-24).

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| **Anchor-margin monitor stays calibration-pending** | owner/agent | Thresholds −800/−700/−1000 recorded verbatim (D-V2-050) but inactive until an environment-side energy source is wired (`anchor_margin_source="external"` + `anchor_margin_fn`); the R3 T4 run confirmed no Stage-1 quantity lives on that scale |
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
388-artifact training bundle + the session-1/2 runs + the R3 run
(`JJKK1212/3d-syntree-v2-runs`: `data/training_bundle.tar`,
`runs/stage1_t4_session1/run.tar`, `runs/stage1_t4_session2/run.tar`,
`runs/stage1_t4_r3/run.tar` — the R3 tar's local == remote sha256
verified S-V2-24) + the catalog (`JJKK1212/3d-syntree-v2-catalog`);
a fresh session restores in ~3 min via
`scripts/runtime_restore_from_hf.sh` (verified bit-exact x4).
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
   **1702 passed / 24 skipped / 0 failed** at `f4fb715` (S-V2-23's
   independently-reproduced R3-era baseline = the S-V2-21 1687/22 +
   the 17 R3 regressions; the 24 skips = 16 #11 stereo pairs + the
   catalog/planner artifact gates + the Suite-B T4 memory gate; a
   docs-only session may cite it without re-running, as S-V2-24 did).
4. Work per the #45 §18 per-issue protocol; follow D-V2-024
   (commit+push each verified unit immediately); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export tokens per command; no background processes
   across calls; `pkill -f aizynthcli` before long planner runs.
