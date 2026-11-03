# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-03, session S-V2-12 close (**#41 DATA-06
closed** — full-scale index benchmark, 888.7× reduction, brute-force
equivalence; **#39 MODEL-06 closed** — explicit seed/anchor action,
pre-seed candidate state, covariant pose, TB seed accounting; **#18
DATA-03 implemented + verified** — graft-spec reactant reconstruction,
indexed retrieval, exact forward replay, resumable driver, bounded
real-data run with the honest 0-acceptance finding; suite **1384/16**;
clean-room closures **22/40**).

## 1. Continuation instruction

```
Next session: the critical path is #18's FULL-SCALE RUN, then #19.

1. #18 DATA-03 full-scale (code is DONE at 35b54ba/95c12f2 — do NOT
   reimplement): re-acquire the pdbbind zip + bindingmoad csv raws
   (crossdocked tar, catalog HDF5 + index, and the mmseqs binary
   already verified in this workspace), run the full preprocessing
   (resumable, ~45 min in bounded calls), full splits (30 s), then
   build_trajectories.py to completion (resumable; ~240 complexes/s
   measured, so budget accordingly) and --finalize. Expected outcome
   per the bounded run: a small accepted set (single-digit % at best;
   the certified-BB reconstructibility constraint — see
   docs/data/trajectories/README.md). Close #18 when the full-scale
   counts + files exist with the failure histogram.
2. #19 DATA-04 (unblocks by #18): ShardedDataset ON the existing
   syntree/data/shards.py primitive (do not re-implement the format).
3. #26 TRAIN-01 then unblocks (needs #19, #20, #25 — #10 closed,
   #25 awaits only its #36-gated CUDA criterion; RuntimeConfig
   consumed from syntree/utils/hardware.py, seeding from
   syntree/utils/seeding.py).
4. #36 OPS-01 prepare-only remains user-authorization-gated (never
   claim a connection); an authorized T4 session ALSO closes #25's
   CUDA criterion — run it first if the user authorizes.

Do NOT redo: #41 (closed with evidence), #39 (closed with evidence),
the #18 engine/driver (landed; the remaining scope is ONLY the
full-scale data run), the S-V2-12 registry corrections (rxn_14/rxn_15
patterns — D-V2-027), or anything from S-V2-11 and earlier (§3).
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
  layer (`syntree/utils/hardware.py`) is the canonical gate.
- **#18 semantics are binding for its consumers** (D-V2-027 + the
  trajectories module docstring): attachment atoms are located BY MAP
  NUMBER (`junction_pattern_attachment_atoms` — RDKit match tuples are
  template-position-ordered); reactant reconstruction uses the
  auto-derived graft specs (`derive_junction_specs()` — itself a
  fail-closed orientation check); symmetric junction patterns must be
  matched with `uniquify=False`; rxn_13 lookups use the protonated cut
  fragment and replays use the catalog row's metalated form; the rxn_15
  secondary-alkyl observation is recorded as D-V2-029 (not yet acted
  on — owner's call).
- **#39 semantics are binding** (D-V2-028 + Property 3b in
  MDP_STATE_SPACE.md): the pre-seed state (SeedMenu) → explicit
  `SeedAction` → deterministic covariant pose (no probability measure);
  the seed TB step carries its forward log-prob and a ZERO backward
  term (unique-parent proof); the seed pool may overlap the growth
  catalog and the placed seed leaves the growth menu.
- The #41 index (`work/catalog/synthon_index.json`, sha `ea57e867…`) is
  the canonical retrieval surface for #18/#29/#44 consumers; it binds
  the immutable catalog sha256 `8b8c13a0…` + the live registry digest
  (D-V2-026 documents the historical-attribute residual).
- Backbone/decoder contracts (D-V2-007/008/011/012), the engine/MDP
  contracts (D-V2-009/015/016/017 + Property 3b), the S-V2-09 decisions
  (D-V2-018/019/020), and the catalog decisions (D-V2-021/022/023/025)
  are unchanged — see DECISIONS.md.

## 3. What the next agent must NOT redo

- Everything in the S-V2-11 handoff's section 3 (S-V2-09 and earlier).
- The S-V2-12 work: #41 (index + benchmark, closed), #39 (environment
  side, closed), the #18 engine + driver (landed and tested — only the
  full-scale RUN remains), the rxn_14/rxn_15 registry corrections, and
  the bounded-run workspace outputs (`work/data/…` — the manifest,
  splits, and trajectory accumulators are resumable state, not stale
  data; `docs/data/trajectories/README.md` records the run).

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| **#25's CUDA acceptance criterion is unverified** | #25/#36 | One full-profile forward on the authorized T4 closes it |
| **#18's full-scale run is pending** | #18 | Bounded-run evidence done; the 0-acceptance census makes the expected accepted count small — an honest outcome, not a blocker for #19 (which shards whatever exists) |
| Certified-BB reconstructibility is low (measured) | #12/#28/owner | 95.7% of the bounded subset fails the seed gate; catalog composition (acyl chlorides nearly absent) and the 7-class seed vocabulary are the binding constraints — informs #28's pretraining and any catalog-extension decision |
| Docking surrogate (`S_dock`/`S_contact`) is a documented interim | #30/eval | Production closure is #30's remaining scope (needs #29) |
| #42's T4 benchmarking acceptance item is unmeasured | #42/#43 gate | The #24 torsion head integration is available |
| rxn_15 pattern looser than its reactant template (D-V2-029) | owner/#11 audit | Only changes which failure code a complex gets; not acted on |
| Dev environment not pinned in a lockfile | #26/#43 | Pin when the trainer lands |
| No GPU/T4 execution, training, or HF artifact sync beyond the catalog | #36/#43 gate | CPU sandbox cannot establish GPU claims |

## 5. Remote runtime / workspace state

No Colab/Kaggle session has been opened (user authorization not
requested; #36 gate). No V2 checkpoints exist. One HF artifact synced:
the #12 catalog dataset `JJKK1212/3d-syntree-v2-catalog` (private,
S-V2-10). No secrets are stored in the repository (session tokens used
only via git remote URLs / per-command env exports, never committed).

Dev-workspace artifacts (not in git; under `work/`):
`raw/crossdocked_pocket10.tar.gz` (md5 verified) — **pdbbind +
bindingmoad raws still to be re-acquired**;
`catalog/enamine_real_50k_pharm3d_d256.h5` (sha verified) +
`synthon_index.json` + manifests; `tools/bin/mmseqs`;
`data/processed/` (3,894-complex bounded manifest + artifacts),
`data/splits/` (bounded splits, 688 clusters / 0 leakage),
`data/trajectories/` (bounded-run accumulators + finalized outputs +
manifest). All re-creatable via `docs/data/trajectories/README.md`.

## 6. Session mechanics

1. `git clone --branch 3d-synV2 …` → confirm HEAD matches the
   recorded commit; `git fetch` first if in doubt (push races
   possible; rebase onto origin, never force-push).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]"`
   && `pip install torch_geometric h5py pandas pyarrow`.
3. Baseline the suite with `SYNTREE_CATALOG_H5=work/catalog/
   enamine_real_50k_pharm3d_d256.h5 python -m pytest tests/ -q` —
   expected **1384 passed / 16 skipped** as of S-V2-12.
4. Work per the #45 §18 per-issue protocol; follow D-V2-024
   (commit+push each verified unit immediately); end with the
   mandatory git diff review, commit+push (report exact SHAs), issue
   update, and refreshed handoff. NOTE: sandbox shells recycle between
   tool calls — re-export tokens per command; no background processes
   across calls.
