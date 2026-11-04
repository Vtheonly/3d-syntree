#!/usr/bin/env python
"""#44 EVAL-03 Track A — export the generator's Enamine REAL catalog as an
AiZynthFinder-compatible stock snapshot.

Track A of #44 is **generator-stock-aligned route verification**: the
exact versioned Enamine REAL building-block subset used by the model,
exported to the planner's stock format, with the catalog IDs preserved
and the export checksums tied to the immutable HDF5 catalog.  A route
found under this stock is *stock-aligned evidence* — it is never
reported as independent of the generator's own inventory (#44 and the
#34 audit addendum both forbid the merge).

Usage::

    python scripts/export_aizynth_stock.py \
        --catalog work/catalog/enamine_real_50k_pharm3d_d256.h5 \
        --output-dir work/eval/stocks

Outputs (fail-closed on any inconsistency):

* ``enamine_stock.txt`` — the AiZynthFinder text stock (one InChIKey
  per line; the planner's ``InMemoryInchiKeyQuery`` format);
* ``enamine_stock_mapping.csv`` — ``inchi_key,block_id,smiles`` — the
  catalog-ID preservation record (#44 acceptance);
* ``enamine_stock_manifest.json`` — the checksum tie: the HDF5
  catalog's own SHA-256, the catalog's embedded provenance attributes
  (source SDF + its SHA-256, registry digest, schema versions), the
  exported files' SHA-256s, row counts, and the export tool versions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExportError(ValueError):
    """Raised when the catalog or the export is inconsistent (fail-closed)."""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#44 Track A: export the Enamine catalog stock snapshot"
    )
    parser.add_argument(
        "--catalog",
        default=os.environ.get("SYNTREE_CATALOG_H5"),
        help="the immutable #12 catalog HDF5 (or SYNTREE_CATALOG_H5)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for the stock files and manifest",
    )
    args = parser.parse_args()
    if not args.catalog or not os.path.exists(args.catalog):
        raise SystemExit(
            f"error: catalog HDF5 not found: {args.catalog!r} "
            "(set --catalog or SYNTREE_CATALOG_H5)"
        )
    catalog_sha = _sha256_file(args.catalog)

    import h5py
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    keys: list[str] = []
    block_ids: list[str] = []
    smiles_list: list[str] = []
    duplicates: list[str] = []
    with h5py.File(args.catalog, "r") as h5:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h5.attrs.items()}
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in h5["catalog/block_ids"][:]]
        smiles = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in h5["catalog/smiles"][:]
        ]
        if len(ids) != len(smiles):
            raise SystemExit("error: catalog block_ids/smiles length mismatch")
        seen: set[str] = set()
        for block_id, smi in zip(ids, smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise SystemExit(
                    f"error: catalog block {block_id} has unparsable SMILES {smi!r}"
                )
            inchi = MolToInchi(mol)
            if not inchi:
                raise SystemExit(
                    f"error: catalog block {block_id} has no InChI: {smi!r}"
                )
            key = InchiToInchiKey(inchi)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            keys.append(key)
            block_ids.append(block_id)
            smiles_list.append(smi)

    stock_path = os.path.join(args.output_dir, "enamine_stock.txt")
    mapping_path = os.path.join(args.output_dir, "enamine_stock_mapping.csv")
    manifest_path = os.path.join(args.output_dir, "enamine_stock_manifest.json")
    with open(stock_path, "w") as handle:
        handle.write("\n".join(keys) + "\n")
    with open(mapping_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["inchi_key", "block_id", "smiles"])
        writer.writerows(zip(keys, block_ids, smiles_list))

    from syntree import __version__ as syntree_version
    from rdkit import rdBase

    manifest = {
        "track": "A — generator-stock-aligned (NEVER report as independent)",
        "stock_name": "enamine_stock",
        "catalog_path": os.path.abspath(args.catalog),
        "catalog_sha256": catalog_sha,
        "catalog_attributes": attrs,
        "num_rows": len(keys),
        "num_unique_inchikeys": len(set(keys)),
        "duplicate_inchikeys": duplicates,
        "stock_file": os.path.abspath(stock_path),
        "stock_file_sha256": _sha256_file(stock_path),
        "mapping_file": os.path.abspath(mapping_path),
        "mapping_file_sha256": _sha256_file(mapping_path),
        "format": "aizynthfinder InMemoryInchiKeyQuery text stock",
        "tool_versions": {
            "syntree": syntree_version,
            "rdkit": rdBase.rdkitVersion,
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": (
            "Track A verifies routes aligned with the generator's own "
            "building-block inventory; the solve rate under this stock is "
            "reported separately from any independent-stock track (#44)."
        ),
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[stock-a] exported {len(keys)} blocks ({len(set(keys))} unique keys)")
    print(f"[stock-a] stock:    {stock_path}")
    print(f"[stock-a] mapping: {mapping_path}")
    print(f"[stock-a] manifest:{manifest_path}")
    print(f"[stock-a] catalog sha256 tie: {catalog_sha}")
    print(f"[stock-a] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
