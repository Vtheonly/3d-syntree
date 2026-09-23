#!/usr/bin/env python3
"""Step 2: write the source manifest for preprocess_multidataset.py."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "./data/raw_pdb")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "./data/raw_pdb/manifest.csv")

rows = []
for prot in sorted(RAW.glob("*_protein.pdb")):
    stem = prot.stem[: -len("_protein")]
    lig = RAW / f"{stem}_ligand.pdb"
    if not lig.exists():
        continue
    rows.append(
        {
            "source": "pdbbind",
            "complex_id": stem,
            "protein_path": str(prot.resolve()),
            "ligand_path": str(lig.resolve()),
            # Publicly documented crystal metadata for the seed complexes.
            "experimental_method": "x-ray",
        }
    )

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["source", "complex_id", "protein_path", "ligand_path",
                       "experimental_method"]
    )
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} manifest rows -> {OUT}")
