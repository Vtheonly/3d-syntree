# 3D-SynTree V2 — Agent Handoff (current resumption point)

> The single authoritative entry point for the next agent session, per
> #45 §25.4. Read order: issue #45 → this file → `docs/agent/STATUS.md` →
> `docs/agent/DEPENDENCIES.md` → `docs/agent/DECISIONS.md` →
> `docs/architecture/ARCHITECTURE.md` → `docs/agent/WORK_LOG.md`.

**Last updated:** 2026-11-01, session S-V2-01 close (#5, #6, #7 —
clean-room rebuild, verified and closed).

## 1. Continuation instruction

```
Next session: implement #21 MODEL-01 (BipartitePaiNN backbone) — both
hard deps are closed against the NEW codebase (#6 schema at commit
0001087, #7 norms/blocks at commit 00ec653). Last verified commit: the
head of 3d-synV2 after the S-V2-01 docs commit (see `git log -1`; code
head is 00ec653). #5, #6 and #7 are implemented from scratch, locally
verified (138/138 tests), and CLOSED — do NOT redo the purge, the
schema, or the norms. Before coding, re-read issue #21 in full, then
`docs/architecture/ARCHITECTURE.md` §3.1/§3.2 and this file, and
confirm branch HEAD matches origin (push races possible). #21 must:
compose PaiNNInteraction+PaiNNMixing (message-then-update) into 8
equivariant layers (d=256, 20 Gaussian RBFs, r_cut=5.0 Å) over the
union pocket∪ligand radius graph (pocket–pocket, ligand–ligand,
pocket–ligand), run cleanly at N_l=0, route every norm/unit vector
through safe_norm/safe_unit_vector, and satisfy
tests/test_bipartite_painn.py (rotation invariance/equivariance).
Parallel-safe alternatives if #21 is taken: #8 MATH-02, #9 MATH-03,
#10 MATH-04, #11 CHEM-01, #14 CHEM-04, #16 DATA-01, #32 TEST-01.
```

## 2. Binding context for every future session

- The branch is a **clean-room rebuild**: it was emptied at commit
  `075f86d` (2026-11-01) per the owner's directive. Never restore,
  re-copy, or "port" pre-reset code; carry forward specifications only
  (decision D-V2-001/D-V2-002). Pre-reset commits are forensic history.
- #45 is the permanent constitution: read it fully before any action;
  never close it.
- Pure FP32 on Colab/Kaggle Tesla T4 only; the dev sandbox is CPU-only
  and can never establish GPU claims (#45 §5–§6).
- Issue-scoped module paths and test file names pinned by issue bodies are
  binding (D-V2-002).

## 3. What the next agent must NOT redo

- The clean-room purge (#5): the branch was emptied at `075f86d` and
  verified; **never** restore or port pre-reset code (D-V2-001) —
  pre-reset commits are forensic history only.
- The frozen tensor schema (#6): `syntree/core/schema.py` is the single
  state contract — extend it, never fork it. Its rules (12 canonical
  fields + version stamp, 67-D rejection, collator-only normalisations,
  single batching path) are binding; 93 tests in
  `tests/test_tensor_contracts.py` pin them.
- The non-singular primitives and PaiNN blocks (#7):
  `syntree/models/equivariant.py` (`safe_norm`, `safe_unit_vector`,
  `PaiNNInteraction`, `PaiNNMixing`, published-equation form per
  D-V2-004); 45 tests in `tests/test_mathematical_singularities.py`
  pin the singularities and equivariance. #21 composes these blocks —
  do not reimplement them.
- README / architecture / agent-docs structure — update in place.

## 4. Known debts / open risks

| Item | Owner | Note |
|---|---|---|
| No GPU/T4 execution, training, or HF artifact sync has occurred for V2 | #36/#43 gate | CPU sandbox cannot establish GPU claims |
| Dev environment not pinned in a lockfile | #26/#43 | `pyproject.toml` declares loose floors (torch≥2.1, PyG≥2.4); pin exact versions when the trainer lands |
| Schema `handle_nodes` optional context | #22/#25 | Policy-side extension point, deliberately outside the 12-field core |
| Unstamped states pass field validation (interim) | #19 | Remove the interim rule when shard producers stamp |
| Radial basis / cosine cutoff construction not yet implemented | #9/#21 | `PaiNNInteraction` consumes `edge_rbf` as an input; the RBF producer is #21 (cutoff envelope maths is #9) |

## 5. Remote runtime / checkpoint state

None — no Colab/Kaggle session has been opened (user authorization not
requested; #36 gate). No V2 checkpoints exist. No HF artifacts synced.

## 6. Session mechanics

1. `git clone --branch 3d-syntree... ` → confirm HEAD matches §1's
   recorded commit; `git fetch` first if in doubt (push races possible).
2. Dev env: Python ≥ 3.10; `pip install torch --index-url
   https://download.pytorch.org/whl/cpu && pip install -e ".[test]"`.
3. Baseline the suite: `python -m pytest tests/ -q` — expected results
   recorded in §1.
4. Work per the #45 §18 per-issue protocol; end with the mandatory git
   diff review, commit+push (report exact SHAs), issue update, and
   refreshed handoff.
