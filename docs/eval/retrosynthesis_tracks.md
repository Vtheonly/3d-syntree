# The two AiZynthFinder retrosynthesis evaluation tracks (#44 / #34)

> The authoritative provenance record for the retrosynthesis tracks of the
> V2 evaluation stack.  These are **two different scientific questions**
> and their solve rates are reported as separate metrics — never merged
> into a headline number (a structural property of
> `syntree/engine/evaluator.py` + `scripts/run_benchmarks.py`: there is
> no aggregated retro key anywhere in the report schema).

## Why two tracks

A retrosynthesis planner can report a false-negative route result if its
**stock** does not contain the building blocks the generator used
(#44 §Context).  Conversely, evaluating only against the generator's own
Enamine catalog does not establish broad, independent vendor-stock
synthesizability.  The two questions are kept separate:

* **Track A — generator-stock-aligned route verification.**  The planner
  runs with the exact versioned Enamine REAL building-block subset used
  by the model.  This verifies routes **aligned with the generator's own
  inventory** and is never labelled independent.
* **Track B — independent external-stock evaluation.**  The planner runs
  with a separately sourced external vendor stock (the official
  AiZynthFinder public ZINC snapshot).  This is the independent
  external-stock evidence (#34's anti-circularity requirement).

Both tracks share the planner's own knowledge base: the **public
USPTO-derived expansion policy** of the `aizynthfinder` package (ONNX
rollout + ringbreaker policies + filter policies, and their template
libraries).  This is external to the generator's certified 32-reaction
taxonomy — the planner never sees our forward reactions.

## Track A — the Enamine stock snapshot

| Item | Value |
|---|---|
| Source | the immutable #12 catalog HDF5 (`enamine_real_50k_pharm3d_d256.h5`, HF `JJKK1212/3d-syntree-v2-catalog`) |
| Catalog SHA-256 | `8b8c13a0581ec91db6ace643e13d9d46c8b29211e1a264b2bdd62bb90bf2ab7e` (matches the pinned checksum) |
| Upstream provenance | `Enamine_Rush-Delivery_Building_Blocks-US_290648cmpd_20260916.sdf` (checksum recorded in the catalog attrs) |
| Coverage | 50,000 building blocks → **50,000 unique InChIKeys** (0 duplicates) |
| Export format | AiZynthFinder `InMemoryInchiKeyQuery` text stock (`enamine_stock.txt`) |
| Catalog-ID preservation | `enamine_stock_mapping.csv` (`inchi_key,block_id,smiles`) — the #44 acceptance requirement |
| Checksum tie | `enamine_stock_manifest.json` records the HDF5 SHA-256, the export SHA-256s, row counts and tool versions |
| Producer | `scripts/export_aizynth_stock.py` (fail-closed on unparsable rows) |

## Track B — the independent external stock

| Item | Value |
|---|---|
| Source | the official AiZynthFinder public stock download (figshare deposit `23086469`, `zinc_stock.hdf5`) — the `aizynthfinder` package's documented `download_public_data` entry point fetches exactly this file |
| Vendor | ZINC database snapshot (AiZynthFinder public data release) |
| Coverage | **17,422,831 unique InChIKeys** |
| Preprocessing | a **memory-bounded format conversion** of the same official artifact (the 4 GB-RAM dev sandbox cannot unpickle the legacy fixed-format HDF5): the pickle byte-stream of the stock's object array is regex-scanned window-by-window (anchored on the pickle opcode + the 27-char InChIKey body); the extraction is **exact** — 17,422,831 keys = the HDF5 row count, verified — and the emitted text file carries SHA-256 `c80161d04f27cce1698c34fb14e5dcc437f73c187a47b08e0b40f3333ef1a460` |
| Access limitations | the sandbox conversion is a documented environment adaptation (the T4 runtime, with 12.9 GB RAM, can load the original HDF5 directly); the stock content is identical |

## The shared planner policy

| Item | Value |
|---|---|
| Expansion policies | `uspto` (ONNX rollout) + `ringbreaker` (ONNX) |
| Template libraries | the public USPTO unique templates + ringbreaker templates |
| Filter policy | the public USPTO filter model (ONNX) |
| Public sources | Zenodo records `7797465` (models) and `7341155` (templates) — URLs and per-file SHA-256s are recorded in `tracks_manifest.json` by `scripts/prepare_aizynth_tracks.py` |
| Planner invocation | an **isolated subprocess** (`aizynthcli`) with file-delivered targets and file-returned results — nothing in the generator's process is shared with the planner (#34's decoupling boundary) |
| Timeout / resume | bounded per-track budgets with the `aizynthcli --checkpoint` resume protocol (#43-style session safety applied to evaluation) |

## Stock-mismatch detection (never a planner failure)

`parse_aizynth_output` classifies every unsolved target:

* `stock_mismatch` — routes were found but their leaves are not
  purchasable in the active stock (`precursors_not_in_stock` non-empty):
  a **stock** limitation, reported with the missing precursor list;
* `no_route_found` — no routes at all: a genuine planner failure.

The distinction is pinned by tests (parser fixtures + a **real-planner
integration test** gated on `SYNTREE_AIZYNTH_MODELS`: acetanilide solves
on a mini stock, while aspirin's routes need out-of-stock leaves and is
classified `stock_mismatch`, never as a planner failure).

## The evidence runs (2026-11-05, dev sandbox, CPU)

Corpus: the 194 forward-replay-verified reconstructed crystal ligands of
the #18 census (111/50/33 over train/val/test — the current system's
construction corpus, **not trained-policy samples**; 194 records over
**30 unique structures** — the same crystal ligand is accepted for
multiple pockets).

| Metric | Value |
|---|---|
| Chemical validity | 194/194 (100%) |
| Mean Fsp3 | 0.5339 |
| Fingerprint diversity (mean pairwise Tanimoto distance) | 0.8934 |
| PoseBusters (`mol` config, ligand 3D geometry) | 194/194 passed |
| Track A solve rate (stock-aligned; denominator = 30 unique structures) | **0.2667** (8/30) — 22 `stock_mismatch`, 0 `no_route_found` |
| Track B solve rate (independent external ZINC stock) | **0.7333** (22/30) — 8 `stock_mismatch`, 0 `no_route_found` |

Reading (honest, evidence-backed): every unsolved target is a stock
mismatch, never a planner failure.  The independent ZINC stock (17.4M
purchasable compounds) solves substantially more than the generator's
own 50k building-block stock — the BB-filtered Enamine subset does not
cover the larger purchasable intermediates the USPTO routes need for
these mature crystal ligands.  Neither number is a headline; they
answer different questions (#44).

Artifacts: `evaluation_report.json` (+ the frozen-target comparative
run in `comparative_benchmark.json` / `.md`) — see
`docs/eval/evidence/`.

## Environment note (recorded per #45 §5/§20)

The dev environment runs `aizynthfinder` 4.4.1 whose metadata pins
`rdkit<2024`; the full pipeline (ONNX policies, rdchiral template
application, searches on both stocks) is **runtime-verified** under
rdkit 2026.09.1 in this environment — the combination is documented in
`DECISIONS.md` (D-V2-044) rather than silently downgrading the
project's chemistry stack.
