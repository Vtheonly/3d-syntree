#!/usr/bin/env python3
"""Which seed complexes produce replayable trajectories?"""
import json, sys
sys.path.insert(0, "/home/z/my-project/3d-syntree")
from rdkit import Chem
from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.fragmenter import ReactionConstrainedFragmenter

catalog = SynthonCatalog("data/multidataset/enamine_3d_subset.parquet")
frag = ReactionConstrainedFragmenter(catalog)

rows = [json.loads(l) for l in open("data/multidataset/processed_manifest.jsonl") if l.strip()]
for row in rows:
    lig = Chem.SDMolSupplier(row["processed_ligand_path"], removeHs=False)[0]
    target = frag.find_target(lig) if lig is not None else None
    print(row["sample_id"], "->", "OK" if target else "no-decomposition")
