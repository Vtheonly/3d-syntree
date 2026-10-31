# 3D-SynTree V2 — Agent Work Log (clean-room era)

> Chronological record of agent engineering sessions. Every entry
> distinguishes **attempted**, **implemented**, **tested locally**,
> **tested on T4**, and **scientifically validated** (#45 §19.1). Never
> paste secrets, credentials, or authentication links here.
>
> Pre-reset history (branch commits `44edb8b..6dbb66a`, issues #5–#7
> closure comments of 2026-10-31) is superseded by the clean-room reset;
> see decision D-V2-001. Git history retains it for forensics.

---

## S-V2-01 — 2026-11-01 — Clean-room reset + foundation (#5) + schema (#6) + norms (#7)

- **Agent/session:** autonomous lead-engineering session (Z.ai), UTC.
- **Repository / branch:** `Vtheonly/3d-syntree`, branch `3d-synV2`.
- **Starting commit:** `6dbb66a` (pre-reset head, 147 files on branch).
- **Issues worked:** #5 ARCH-01, #6 ARCH-02, #7 MATH-01 (all reopened this
  session under the owner's clean-room directive; see below).

### 1. State discovery (per #45 §25.1)

- Read issue #45 in full (all sections incl. §24 greenfield mandate, §25
  continuation protocol, §28/§29 remote-runtime rules).
- Fetched all 41 issues (#5–#45, both states) with comments; verified:
  pre-reset closures were #5, #6, #7 only; #8–#44 open; #45 open.
- Read pre-reset agent docs (STATUS/WORK_LOG/DECISIONS/DEPENDENCIES/
  HANDOFF), architecture doc, `syntree/core/schema.py`,
  `syntree/models/equivariant.py` — **as specification extraction only**
  (validated contract rules recorded in the new DECISIONS.md, D-V2-002).
- Owner directive received: the pre-reset "V2" (incremental repair of the
  legacy tree still carrying 88 legacy files + 800+ V1 tests) is rejected;
  the branch must be emptied and V2 rebuilt from zero.

### 2. Clean-room purge (owner directive; #5 core scope)

- `git rm -r -f .` → all 147 tracked files staged for deletion; working
  tree left with `.git` only.
- Commit `075f86d` "reset(v2)!: purge 3d-synV2 to empty root for
  clean-room V2 rebuild (#5)" — pushed to origin (fast-forward
  `6dbb66a..075f86d`), **no force-push, history preserved**.
- **Verification:** `git ls-files` empty; GitHub Trees API at `075f86d`
  returns 0 entries; spot-checked #5-named legacy targets
  (`craft/craft_mode.py`, `syntree/engine/dpo.py`,
  `syntree/models/pocket_encoder.py`, `syntree/data/crossdocked.py`)
  → HTTP 404 on the branch. No dangling imports possible (nothing left).
- **Issues reopened with rationale comments:** #5, #6, #7 (PATCH state =
  open + explanatory comment each); #45 verified still open (never
  touched).

### 3. Fresh V2 foundation (#5, in progress at time of this entry)

- New files (all newly written): `LICENSE` (MIT), `.gitignore`,
  `pyproject.toml` (src-layout, `syntree` 2.0.0.dev0, deps
  `torch`/`torch_geometric`, decision D-V2-003), `README.md`
  (first-class V2 description per #45 §4 with honest status table),
  `docs/architecture/ARCHITECTURE.md` (global spec, contract statuses),
  `docs/data/RAW_SOURCE_PROVENANCE.yaml` (verified upstream facts carried
  forward), `docs/agent/{STATUS,WORK_LOG,DECISIONS,DEPENDENCIES,HANDOFF}
  .md` (this suite).
- Dev environment: Python 3.12.16, torch 2.15.0+cpu, torch_geometric
  2.8.0.post1, pytest 9.0.2 (CPU sandbox — no GPU claims made).

### 4. Implementation records for #6 / #7

(To be finalized at session close: modules, tests, commands, results,
commit SHAs — see the issue comments for the authoritative evidence.)

### 5. Commands executed (session, abridged)

```
git clone https://github.com/Vtheonly/3d-syntree.git && git checkout 3d-synV2
<GitHub API via token: issue fetch/reopen/comment operations>
git rm -r -f . && git commit && git push <token-scoped URL> 3d-synV2
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric-2.8.0.post1.whl   # cached wheel, upstream artifact
python -m pytest tests/ -q                    # evidence recorded below at close
```

### 6. Next actions

Recorded in `HANDOFF.md` at session close.
