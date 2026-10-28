# Train Observatory — prototype modules (2026-10-28/29)

These six standalone modules were built and verified IN ISOLATION (the
mandated "prototype-first" policy for the training-observability system).
They are preserved in-repo as the reference implementation for the next
integration step. They are NOT imported by the `syntree` package; nothing in
the production path depends on them.

| Module | Purpose | Status |
|---|---|---|
| `train_observatory.py` | RunMonitor / RunReport / compare_runs: per-run manifests, events.jsonl audit trail, metrics.csv, evaluations.csv (long format incl. per-level metrics), system.csv resource sampling, summary + plots | Prototype-verified (see below); trainer integration deferred |
| `curriculum.py` | Prototype of grouped leakage-safe splits, difficulty scoring, 9-level assignment, CurriculumSampler, LevelTracker | **Superseded** by `syntree/engine/curriculum.py` (2026-10-29), which ports the same math onto the real PyG datasets + synthon catalog |
| `nan_shield.py` | NaN localization, streak abort (SystemicNanError), grad/param/optimizer gates, verified checkpoints (save→reload→finiteness), rollback + `nan_forensics/` snapshots | Prototype-verified; trainer integration deliberately deferred (the persistent NaN error is documented, NOT fixed — see `docs/PERSISTENT_ERRORS_NAN.md`) |
| `mol_metrics.py` | Real evaluation metrics: reaction accuracy, top-k, synthon exact match + Tanimoto, generation validity/uniqueness/novelty/diversity | Prototype-verified; evaluator integration deferred |
| `prototype_test.py` | Isolated mini-"SynTree" sandbox exercising all of the above on synthetic SMILES chemistry | All assertions passed 2026-10-28 (see `comparison_summary.txt`) |
| `integration_snippet.py` | The exact ~6-line-per-hook wiring guide for the trainer | Reference only |

## Prototype verification evidence (2026-10-28)

- `python prototype_test.py` -> PASS/FAIL table, all green, exit 0.
- Non-interference guarantees measured on the prototype trainer:
  1. training with vs. without RunMonitor is **bit-identical**,
  2. per-step monitoring overhead **< 2 ms/step** (budget asserted in the suite),
  3. every public method is exception-guarded: a monitoring bug degrades to a
     stderr line + `soft_failures` entry, never a training crash.
- NaN-failure drill: a poisoned checkpoint (NaN weights) was caught by
  `nan_shield.verify_checkpoint`, a mid-run NaN batch was localized to its
  loss component and rolled back from the last known-good checkpoint
  (forensics snapshot: `docs/evidence/prototype_nan_forensics_ep2_b4.json`).
- `comparison_summary.txt` — two full prototype runs (baseline vs low-LR)
  compared through `compare_runs`, plus crash/corrupt-run recovery drills.

## Why they are not yet integrated

Per the 2026-10-29 handoff decision: (a) the immediate iteration scope was
restart-from-scratch + curriculum + error documentation; (b) the NaN
root-cause investigation is deliberately left open (documented in
`docs/PERSISTENT_ERRORS_NAN.md`), and integrating `nan_shield` would change
the failure surface before the evidence is collected. Integration of
`train_observatory` + `mol_metrics` is the next major work item — see
`docs/HANDOFF_2026-10-29.md` (Next Steps).
