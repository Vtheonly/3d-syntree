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

## Status caveat (honest, per #45 §18)

The build **artifacts and manifest are final and verified**, but the
build **code** (`syntree/chemistry/synthon_encoder.py`,
`scripts/build_synthon_catalog.py`, and their tests) was lost to a
sandbox recycle **before it could be committed** (session S-V2-10
incident, see `docs/agent/WORK_LOG.md`). Issue #12 therefore remains
**OPEN** until the code is re-landed and pinned against these artifacts;
do not treat the catalog as closed-scope until then. Consumers
(`SynthonRetrievalHead` / `SeedAnchorHead` tied table, #18/#39/#41/#44)
may already rely on the HDF5 **schema** pinned here (`pharm3d-v2.0` /
`catalog-v2.0`): `catalog/` group (block ids, SMILES, class strings,
strata, physicochemical scalars, conformer provenance),
`embeddings [50000, 256]` float32, `projection/Q [2137, 256]` (QR
Johnson–Lindenstrauss, seed 2026), `normalization/` (per-feature
mean/std), `versions/` (pinned toolchain).
