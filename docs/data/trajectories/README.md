# DATA-03 bounded-run evidence (S-V2-12, 2026-11-03)

> First real-data execution of the #18 trajectory-extraction pipeline
> (#41-indexed candidate discovery + exact forward-replay acceptance).
> Bounded CrossDocked subset; the full-scale run (all three sources, the
> 204,005-record manifest) continues in later sessions via the resumable
> driver — nothing here is claimed at full scale.

## Run configuration

| Input | Value |
|---|---|
| Source | CrossDocked pocket10 mirror (md5 `59416d06…` verified vs `RAW_SOURCE_PROVENANCE.yaml`) |
| Preprocessing | `preprocess_multidataset.py --source crossdocked --limit 5000 --workers 2` |
| Complexes | 5,000 considered → **3,894 accepted / 1,106 rejected** (explicit #16 reason codes) |
| Splits | `run_mmseqs_split.sh` (mmseqs static build): 688 clusters, **0 clusters spanning splits** (fail-closed leakage check) |
| Catalog | immutable HDF5 sha256 `8b8c13a0…` (verified at load) |
| Index | `synthon-index-v2.0` (sha `ea57e867…`), registry digest verified at load |
| Extraction | `build_trajectories.py --max-seconds 420` (completed in 16.2 s) |

## Measured results (dev-sandbox CPU, single-threaded)

| Metric | Value |
|---|---|
| Throughput | **239.8 complexes/s** (incl. all failure paths) |
| Accepted trajectories | **0 / 3,894** |
| `no_matched_seed` | 3,727 (95.7%) |
| `no_matched_synthon` | 156 (4.0%) |
| `replay_mismatch` | 11 (0.3%) |

## Honest interpretation (the scientific finding)

1. **The seed gate dominates.** 95.7% of crystal ligands have a canonical
   decomposition whose root fragment is **not** an exact certified seed
   block of the 50k Enamine catalog: either the root carries no seed
   class at all (naphthalene/coumarin/benzodiazepine scaffolds — plain
   aromatics with no reactive group in the 7-class seed vocabulary), or
   it is a drug-sized scaffold that is not a purchasable building block
   (nucleosides, fused bicycles, …). This is a property of
   *certified-BB-reconstructible* chemistry, not a pipeline defect: the
   seed action of #39 selects from certified catalog rows by design.
2. **Growth-block composition is the second constraint.** Of the 167
   seed-matched complexes, 156 fail because the first reconstructed
   growth block is not a catalog row (e.g. `NC(=O)c1ccc(Br)nc1` ×96 — a
   nicotinamide drug intermediate, not a BB). Common acyl chlorides
   (`CCC(=O)Cl`, benzoyl chloride) are absent from the catalog (only 108
   `acid_chloride` rows exist — acyl chlorides are rarely stocked as
   shipping-stable building blocks), which structurally limits amide
   step acceptance.
3. **The 11 `replay_mismatch` cases are correct fail-closed behaviour.**
   All are rxn_15 (Negishi) junctions on **secondary** alkyl centres: the
   junction pattern recognises the bond, but the certified reactant
   template requires primary alkylzinc (`[CX4;H2,H3]`, documented
   "primary alkylzinc only") — the reverse reconstruction produces the
   secondary organozinc, the forward replay rejects it, and the complex
   is dropped with an explicit code. Observation recorded for a future
   registry alignment refinement (tighten rxn_15's junction pattern to
   `H2`): D-V2-029 note; not changed this session (scope discipline).
4. **Consequence for supervised training:** the reconstructible-from-
   certified-BBs fraction of crystal ligand space is small (single-digit
   percent at best under the current catalog). The supervised set
   (#18 full-scale) will be correspondingly small; the #28 2D
   pretraining track and catalog-composition decisions gain importance.
   This is reported as measured — the full-scale number lands when the
   remaining 200k complexes and the pdbbind/bindingmoad sources run.

## Verification commands (reproduce)

```bash
python scripts/preprocess_multidataset.py --source crossdocked \
    --raw-dir work/raw --out-dir work/data/processed --limit 5000 --workers 2
MMSEQS_BIN=work/tools/bin/mmseqs bash scripts/run_mmseqs_split.sh \
    work/data/processed/processed_manifest.jsonl work/data/splits
python scripts/build_trajectories.py \
    --manifest work/data/processed/processed_manifest.jsonl \
    --splits work/data/splits/unified_splits_30seq_id.parquet \
    --catalog work/catalog/enamine_real_50k_pharm3d_d256.h5 \
    --index work/catalog/synthon_index.json \
    --out-dir work/data/trajectories --finalize
```

Unit-level acceptance evidence (the positive path): `tests/test_trajectories.py`
29/29 — real multi-step amide and Sonogashira trajectories extracted,
replay-verified, torsion-labelled end-to-end, plus every failure code
exercised; the driver smoke chain additionally covers the accepted path
through `trajectories_{train,val}.pt`.
