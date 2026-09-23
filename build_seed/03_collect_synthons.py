#!/usr/bin/env python3
"""Step 3: extract the synthon candidates from the real seed ligands.

Uses the fragmenter's OWN disconnection + handle-decoration logic so the
catalog contains exactly the building blocks the retrosynthetic replay
needs - i.e., every decorated fragment is a real, purchasable-style
synthon (free amine, carboxylic acid, boronic acid, aryl bromide, ...).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/3d-syntree")

from rdkit import Chem

from syntree.data.fragmenter import (
    SUPPORTED_RETRO_FAMILIES,
    ReactionConstrainedFragmenter,
    canonical_smiles,
)
from syntree.chemistry.catalog import SynthonCatalog
from syntree.chemistry.reactions import ReactionEngine

MANIFEST = Path(sys.argv[1] if len(sys.argv) > 1 else "data/multidataset/processed_manifest.jsonl")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/raw_pdb/synthon_candidates.csv")

# A minimal catalog is needed only to instantiate the fragmenter for
# _family_pairs; the candidate collection itself never touches it.
dummy_csv = Path("/tmp/_dummy_catalog.csv")
dummy_csv.write_text(
    "id,smiles,fsp3,mw,primary_handle\n"
    "DUMMY-0000,OC1CCCCC1,1.0,100.16,alcohol\n"
)
catalog = SynthonCatalog(str(dummy_csv))
fragmenter = ReactionConstrainedFragmenter(catalog)

rows = [json.loads(l) for l in MANIFEST.open() if l.strip()]
seen = {}
for row in rows:
    lig = Chem.SDMolSupplier(row["processed_ligand_path"], removeHs=False)[0]
    if lig is None or lig.GetNumConformers() == 0:
        continue
    for family in SUPPORTED_RETRO_FAMILIES:
        for side_a, side_b, backend in fragmenter._family_pairs(family):
            for bond_idx in range(lig.GetNumBonds()):
                bond = lig.GetBondWithIdx(bond_idx)
                if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                    continue
                try:
                    fragments = ReactionConstrainedFragmenter._split_bond(
                        lig, bond_idx
                    )
                except Exception:
                    continue
                if len(fragments) != 2:
                    continue
                for frag, endpoint in fragments:
                    for handle in (side_a, side_b):
                        for decorated in ReactionConstrainedFragmenter._decorate_fragment(
                            frag, endpoint, handle
                        ):
                            smiles = canonical_smiles(decorated)
                            if smiles and smiles not in seen:
                                seen[smiles] = handle

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
    writer.writeheader()
    for i, (smiles, handle) in enumerate(sorted(seen.items())):
        writer.writerow({"id": f"SEED-{i:05d}", "smiles": smiles})

print(f"{len(seen)} unique synthon candidates -> {OUT}")
