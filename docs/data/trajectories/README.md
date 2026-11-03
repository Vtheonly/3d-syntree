# DATA-03 trajectory extraction — bounded + full-scale evidence

> #18 DATA-03. Two real-data executions of the #18 extraction engine
> (#41-indexed candidate discovery + exact forward-replay acceptance):
> a bounded CrossDocked subset (S-V2-12, retained below for history) and
> the **full-scale run over all 170,955 accepted complexes of the three
> sources** (S-V2-13, 2026-11-03 — the run that closes the issue).

## Full-scale run (S-V2-13, 2026-11-03)

| Input | Value |
|---|---|
| Manifest | `work/data/processed/processed_manifest.jsonl` — 204,005 records (170,955 accepted / 33,050 rejected), regenerated this session; per-source accepted: crossdocked 154,760 / pdbbind 4,862 / bindingmoad 11,333 |
| Splits | `unified_splits_30seq_id.parquet` — 136,764 / 17,096 / 17,095; 22,387 clusters, 14,958 components, **0 leakage** (deterministic reproduction of the committed S-V2-11 evidence, bit-exact counts) |
| Catalog | immutable HDF5 sha256 `8b8c13a0581e…` (verified at load, HF sync) |
| Index | `synthon-index-v2.0` rebuilt from the catalog — byte-deterministic, sha256 `ea57e867…` (identical to the S-V2-12 artifact; exhaustive self-retrieval 68,417/0 lost, brute-force equivalence 2,000/0) |
| Extraction | `build_trajectories.py --max-seconds 430` × 4 resumable calls + `--finalize` (dev-sandbox CPU, single-threaded) |

### Measured results

| Metric | Value |
|---|---|
| Complexes processed | **170,955 / 170,955** (pending 0 at finalize) |
| Throughput | **176.8/s** mean across the 3 processing calls (77,995 + 76,024 + 16,740 in 966 s); all failure paths included |
| Accepted trajectories | **194** (train 111 / val 50 / test 33) |
| `no_matched_seed` | 166,983 (97.78%) — crossdocked 151,232, moad 11,011, pdbbind 4,740 |
| `no_matched_synthon` | 3,648 (2.13%) — crossdocked 3,261, moad 289, pdbbind 98 |
| `replay_mismatch` | 123 (0.07%) — crossdocked 108, moad 11, pdbbind 4 |
| `graft_failed` | 6 (pdbbind; exotic input graphs fail-closed post-guard) |
| `seed_fragment_unsanitizable` | 1 (pdbbind_1f5l — non-ring aromatic SDF bond; crash fixed this session, see commit `00ba1b8`) |
| Accepted by source | crossdocked 159, bindingmoad 22, pdbbind 13 |
| Accepted step histogram | 0 steps 176 (whole ligand = certified seed row: SeedAction+STOP under #39 semantics) / 1 step 16 / 2 steps 2 |
| Reactions in accepted steps | rxn_04 ×14, rxn_01 ×2, rxn_15 ×2, rxn_16 ×2 (20 steps total) |

Outputs (sha256):

- `trajectories_train.pt` — 111 records, 68,591 B, `8e95ed891b68c3dd78546115d2338f66ea521a81ac72af55f6a2a074000d5e85`
- `trajectories_val.pt` — 50 records, 28,195 B, `76fe93077306336898535eb46803e3274688a597c3d2b6a18834a6fa2b928d80`
- `trajectories_test.pt` — 33 records, 18,281 B, `4f67184914fa40fe5ff1e493594afa156928e15621e855f2cad8fbd032264db8`
- `extraction_manifest_full_scale.json` — committed evidence copy of the run manifest

### Acceptance-criteria verification (issue #18)

1. **100% of accepted trajectory steps pass forward-replay graph
   isomorphism.** By construction the extractor only emits steps it
   replayed (the exact stage re-executes the certified reaction and
   compares canonical graphs); additionally an **independent audit this
   session re-replayed all 20 accepted steps** from the recorded
   reactant SMILES through `run_certified_reaction` — 20/20 exact
   canonical-graph matches, 0 mismatches.
2. **Discarded complexes are logged with explicit failure codes.**
   170,761 failures over the closed vocabulary (`no_matched_seed`,
   `no_matched_synthon`, `replay_mismatch`, `graft_failed`,
   `seed_fragment_unsanitizable`), one JSONL record each with split,
   ligand SMILES, and detail. Full histogram above and in
   `extraction_manifest_full_scale.json`.

### Interpretation (honest, unchanged in kind from the bounded run)

The certified-BB reconstructibility fraction of crystal ligand space is
**0.11%** at full scale (194/170,955) — dominated by the seed gate
(97.8% of complexes have a canonical root fragment that is not a
certified 50k-catalog row). This is a property of
certified-BB-reconstructible chemistry, not a pipeline defect: the seed
action of #39 selects from certified catalog rows by design. Two
session-specific engineering findings, both fixed fail-closed this
session (commits `3cf5345`, `00ba1b8`):

- the chemcomp template fetch previously fabricated permanent
  no-template verdicts from transient RCSB outages (cache poisoning);
- an exotic non-ring aromatic SDF bond (pdbbind_1f5l) crashed the
  driver instead of yielding an explicit failure code.

Consequences for downstream: the supervised Stage-1 set is small
(194 trajectories; 20 growth steps) — this measurably raises the
importance of the #28 2D-pretraining track and any owner decision on
catalog composition/extensions (acyl chlorides, the 7-class seed
vocabulary). #19 shards whatever exists (as pinned by the handoff).

## Bounded-run evidence (S-V2-12, 2026-11-03 — retained for history)

| Input | Value |
|---|---|
| Source | CrossDocked pocket10 mirror (md5 `59416d06…` verified vs `RAW_SOURCE_PROVENANCE.yaml`) |
| Preprocessing | `preprocess_multidataset.py --source crossdocked --limit 5000 --workers 2` |
| Complexes | 5,000 considered → 3,894 accepted / 1,106 rejected (explicit #16 reason codes) |
| Splits | bounded 688 clusters, 0 clusters spanning splits |
| Throughput | 239.8 complexes/s; accepted 0 / 3,894; failures: `no_matched_seed` 3,727, `no_matched_synthon` 156, `replay_mismatch` 11 |
| Finding | the seed gate dominates (95.7%); acyl-chloride absence structurally limits amide acceptance; the 11 `replay_mismatch` are rxn_15 secondary-alkyl fail-closed behaviour (D-V2-029) |

## Verification commands (reproduce the full-scale run)

```bash
python scripts/preprocess_multidataset.py --source all --raw-dir work/raw \
    --out-dir work/data/processed --gzip --workers 2   # resumable; ~45 min
MMSEQS_BIN=work/tools/bin/mmseqs bash scripts/run_mmseqs_split.sh \
    work/data/processed/processed_manifest.jsonl work/data/splits
python scripts/build_synthon_index.py --catalog work/catalog/enamine_real_50k_pharm3d_d256.h5 \
    --out work/catalog/synthon_index.json --manifest work/catalog/index_manifest.json
python scripts/build_trajectories.py \
    --manifest work/data/processed/processed_manifest.jsonl \
    --splits work/data/splits/unified_splits_30seq_id.parquet \
    --catalog work/catalog/enamine_real_50k_pharm3d_d256.h5 \
    --index work/catalog/synthon_index.json \
    --out-dir work/data/trajectories --max-seconds 430   # re-run until pending 0
python scripts/build_trajectories.py ... --finalize     # assembles the .pt outputs
```

Unit-level acceptance evidence (the positive path): `tests/test_trajectories.py`
33/33 — real multi-step amide and Sonogashira trajectories extracted,
replay-verified, torsion-labelled end-to-end; every failure code
exercised including the `seed_fragment_unsanitizable` regression
(the verbatim pdbbind_1f5l molblock).
