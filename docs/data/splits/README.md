# V2 Split Build Evidence — #17 DATA-02 (session S-V2-11, 2026-11-02)

This directory pins the verified evidence of the #17 MMseqs2
sequence-clustered split generation over the regenerated
`processed_manifest.jsonl` (#16 re-run, S-V2-11: 204,005 records /
170,955 accepted — one MOAD acceptance drifted vs the S-V2-10 run's
170,954, attributed to upstream RCSB/chemcomp API drift; crossdocked and
pdbbind per-source counts match the original close evidence exactly).

## Files here

| File | Content |
|---|---|
| `split_manifest.json` | The authoritative split manifest: exact counts, per-source split counts, cluster/component statistics, the pinned mmseqs contract, the allocation rule, and the zero-leakage check result |

## Measured results (full scale)

* Input: 170,955 accepted complexes, 195,417 chain slots,
  169,281 unique chain sequences (first-appearance fasta order).
* MMseqs2 `easy-cluster --min-seq-id 0.30 -c 0.80 --cov-mode 0
  --cluster-mode 2`: 22,387 clusters (11,873 multi-member).
* Union-Find over complexes sharing homologous chains: **14,958
  connected components** (largest 18,458; 4,919 singletons).
* Deterministic allocation (components by `(-size, min complex_id)`,
  greedy largest-deficit, ties train>val>test):
  **train 136,764 (80.0%) / val 17,096 (10.0%) / test 17,095 (10.0%)**.
* Leakage check (fail-closed, executed in-run): **0 of 22,387 clusters
  span splits; components never split by construction.**

## Output artifact

`unified_splits_30seq_id.parquet` — 170,955 rows (`complex_id`,
`split`), one per accepted complex. The parquet lives in the processing
workspace (data artifacts are git-ignored); this evidence directory plus
the split manifest pin its content. The splitter is resumable and
deterministic: re-running `scripts/run_mmseqs_split.sh` on the same
manifest reproduces the table exactly.
