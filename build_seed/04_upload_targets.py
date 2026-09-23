#!/usr/bin/env python3
"""Upload RL target pocket PDBs to the dataset repo's targets/ folder.

Landmine 3 fix: the RL loop needs real pocket PDBs, which the HF shard
backend does not materialise as loose files. We stage the non-test-split
pockets (train+val complexes) so Stage 2 never sees the final evaluation
targets.
"""
import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = "JJKK1212/3d-syntree-multidataset"
ROOT = Path("/home/z/my-project/3d-syntree")

splits = pd.read_parquet(ROOT / "data/multidataset/unified_splits_30seq_id.parquet")
split_map = dict(zip(splits["sample_id"].astype(str), splits["split"].astype(str)))

api = HfApi(token=TOKEN)
uploaded = 0
for pocket in sorted((ROOT / "data/multidataset/pdbbind").glob("*_pocket.pdb")):
    complex_id = pocket.name[: -len("_pocket.pdb")]
    sample_id = f"pdbbind:{complex_id}"
    split = split_map.get(sample_id)
    if split == "test":
        print(f"skip {complex_id} (test split stays unseen by RL)")
        continue
    api.upload_file(
        path_or_fileobj=str(pocket),
        path_in_repo=f"targets/{pocket.name}",
        repo_id=REPO,
        repo_type="dataset",
        token=TOKEN,
    )
    print(f"uploaded targets/{pocket.name} ({split})")
    uploaded += 1
print(f"{uploaded} RL target pockets uploaded")
