# Legacy V1 Documentation Archive

This directory preserves the historical engineering record of the V1
implementation (pre-`3d-synV2`) — its decisions, audits, curriculum design,
NaN forensics, and evidence logs. These files are **read-only history**:

- They describe the *old* system: fp16 + GradScaler training, PPO/DPO
  Stage-2 candidates, the synthetic CrossDockedDataset smoke path, the
  Kaggle offline loader, VRAM-probing auto-scale, and the 2026-10-28/29
  incident analysis (GradScaler collapse → all-NaN checkpoint).
- None of the code they describe exists on `3d-synV2` anymore (purged in
  #5 ARCH-01).
- Their scientifically valuable conclusions (verified data-source
  provenance, the failure taxonomy, the curriculum *design rationale*)
  were carried forward into the V2 documentation:

| Legacy record | V2 successor |
|---|---|
| `DECISIONS.md` (V1 decisions D1–D14) | `docs/agent/DECISIONS.md` (V2 decisions reference them where still valid) |
| `HANDOFF_2026-10-29.md` | `docs/agent/HANDOFF.md` (last V1 session; V2 handoffs start 2026-11-01) |
| `CURRICULUM.md` | Consumed by #27 TRAIN-02 (Approach C quantile curriculum) |
| `PERSISTENT_ERRORS_NAN.md` + `evidence/` | Consumed by #10 MATH-04 (pure FP32 mandate) and #43 OPS-03 (session-safe resume); the fp16 root-cause chain is closed by V2's no-fp16 policy |
| `ENGINEERING_AUDIT_REPORT.md` | Consumed by the #5–#44 backlog design |
| `tasklist.md` | Superseded by the GitHub issue backlog #5–#44 |

**Verified data-source provenance** (upstream URLs, checksums, verification
dates for CrossDocked2020 / PDBbind / BindingMOAD / Enamine REAL) is
preserved separately in `docs/data/RAW_SOURCE_PROVENANCE.yaml` for the
V2 ingestion pipeline (#16 DATA-01).

Do not edit these historical records; append new V2 documentation under
`docs/agent/`, `docs/architecture/`, `docs/data/`, or `docs/experiments/`.
