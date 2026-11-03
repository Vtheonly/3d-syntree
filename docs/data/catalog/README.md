# V2 Catalog Build Evidence — #12 CHEM-02 (session S-V2-10, 2026-11-02)

This directory pins the **verified evidence of the #12 Enamine REAL
50,000-block synthon catalog build**. The immutable HDF5 artifact itself
is banned from git by the repository `.gitignore` (`*.h5`) and lives in
the pinned artifact store:

> **Hugging Face dataset (private):**
> <https://huggingface.co/datasets/JJKK1212/3d-syntree-v2-catalog>
> `enamine_real_50k_pharm3d_d256.h5` — 67,797,608 bytes,
> sha256 `8b8c13a0581ec91db6ace643e13d9d46c8b29211e1a264b2bdd62bb90bf2ab7e`

## Files here

| File | Content |
|---|---|
| `catalog_manifest.json` | The authoritative build manifest: source SDF + sha256, the full curation funnel, stratified selection, per-reason encode failures, acceptance block, reaction-registry digest, artifact checksum — `decision_log_refs` D-V2-021/022/023 |
| `screen_funnel.json` | The measured curation funnel over all 290,648 source records (embedded in the manifest; kept standalone for convenience) |
| `enamine_real_50k_pharm3d_d256.h5.sha256` | The artifact checksum sidecar |

## Verified acceptance (independently re-computed from the HDF5)

| Criterion (issue #12) | Measured | Pass |
|---|---|---|
| Catalog contains exactly 50,000 entries | 50,000 rows; 50,000 unique SMILES; 50,000 unique Enamine block ids | ✓ |
| Embedding matrix exactly `[50000, 256]`, zero NaN/Inf | `[50000, 256]` float32, all finite | ✓ |
| Mean Fsp3 ≥ 0.45 | 0.715094 | ✓ |
| Mean rotatable bonds ≤ 1.8 | 1.5361 | ✓ |

Stratification: 34,984 saturated (69.968%) / 15,016 aromatic (30.032%)
— the issue's 70/30 target under the certified-supply constraint
(D-V2-023). Conformer provenance per row: `etkdg3` 17,493 /
`etkdg3+mmff94` 32,507 (recorded in `catalog/conformer_methods`).

## Status caveat (honest, per #45 §18) — RESOLVED (S-V2-11, 2026-11-03)

The build **artifacts and manifest are final and verified**, but the
original build **code** was lost to a sandbox recycle before it could be
committed (session S-V2-10 incident, see `docs/agent/WORK_LOG.md`).

**S-V2-11 re-landing (commits `b76422a` + `b8439bd`): the code is
re-landed and verified against these artifacts at full scale.** The
re-implemented pipeline (`syntree/chemistry/synthon_encoder.py` +
`scripts/build_synthon_catalog.py` + `tests/test_synthon_encoder.py`,
50 tests) reproduces:

| Verification (full scale, 290,648 source records) | Result |
|---|---|
| Funnel: all 16 stage counts + all 8 per-rule exclusion counts | **24/24 exact** (incl. seed_self_reactive_pair 757 fired blocks / 760 pair-firings) |
| Certification labels vs the 50,000 catalog rows (synthon + seed classes) | **0/50,000 mismatches** |
| Strata assignment (`CalcNumAromaticRings > 0` → aromatic) | **0/50,000 mismatches** |
| Buffered selection (35k saturated + 20k aromatic, canonical-SMILES order, saturated first) | **buffered_rows index list reproduced exactly** |
| QR-JL projection (seed 2026, sign-normalized) | **projection/Q bit-exact (max Δ = 0.0)** |
| Conformer-method decision tree (MMFF pre-embed typing + B/Zn/Sn/halide fallback) | **17,493/32,507 split reproduced with 0/50,000 row mismatches** |
| Full rebuild: catalog size / embedding shape / finiteness / mean Fsp3 / mean rotatable | 50,000 / [50000,256] finite / 0.7151 / 1.5361 — all acceptance bounds pass |
| Rebuilt catalog overlap with the immutable artifact | **49,998/50,000 (99.996%)** |

Residual reconstruction ambiguities (documented in the module docstring
and the rebuild manifest): 3 of 47 physchem functional-group positions
(tertiary-amine slot: stored 0.070600 vs 0.079120; post-sulfone slot:
stored 0.001240/62 rows vs sulfoxide 0.002520/64 rows; one provably
all-zero slot), the 30-D handle-vector class list (identity-normalized in
the artifact — statistically unrecoverable), and the original run's ETKDG
seed (energies underdetermine it; the re-landed protocol is deterministic
with seed 2026). The **immutable HDF5 above remains the authoritative
catalog**; the rebuilt HDF5 is a new artifact version (sha256
`00a556ef2e3427f03fb562fa612080f20ec9fb86b041cfea78620b51a466ea2a`).

**#12 is closed.**
