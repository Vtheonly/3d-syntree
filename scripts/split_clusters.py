#!/usr/bin/env python3
"""Build leakage-safe split assignments from an MMseqs2 cluster TSV.

All complexes sharing any representative cluster are linked into one connected
component. Connected components, not individual complexes, are assigned to
train/validation/test.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--cluster-tsv", required=True)
    parser.add_argument("--output", default="./data/multidataset/unified_splits_30seq_id.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    args = parser.parse_args()

    chain_to_sample = {}
    with Path(args.fasta).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
                chain_to_sample[header] = header.rsplit("|chain=", 1)[0]

    cluster_df = pd.read_csv(
        args.cluster_tsv, sep="\t", header=None, names=["rep", "member"], dtype=str
    )
    member_to_rep = {}
    for row in cluster_df.itertuples(index=False):
        member_to_rep[str(row.member)] = str(row.rep)
        member_to_rep.setdefault(str(row.rep), str(row.rep))
    for header in chain_to_sample:
        member_to_rep.setdefault(header, header)

    sample_clusters = defaultdict(set)
    for header, sample_id in chain_to_sample.items():
        sample_clusters[sample_id].add(member_to_rep[header])

    samples = sorted(sample_clusters)
    uf = UnionFind(samples)
    cluster_owner = {}
    for sample_id, reps in sample_clusters.items():
        for rep in reps:
            prior = cluster_owner.get(rep)
            if prior is not None:
                uf.union(sample_id, prior)
            else:
                cluster_owner[rep] = sample_id

    components = defaultdict(list)
    for sample_id in samples:
        components[uf.find(sample_id)].append(sample_id)
    component_ids = list(components)

    rng = random.Random(args.seed)
    rng.shuffle(component_ids)
    n = len(component_ids)
    n_train = int(n * args.train_fraction)
    n_val = int(n * args.val_fraction)

    component_split = {}
    for i, component_id in enumerate(component_ids):
        component_split[component_id] = (
            "train"
            if i < n_train
            else "val"
            if i < n_train + n_val
            else "test"
        )

    rows = []
    for sample_id in samples:
        component_id = uf.find(sample_id)
        rows.append({
            "sample_id": sample_id,
            "split": component_split[component_id],
            "component_id": component_id,
            "cluster_representatives": sorted(sample_clusters[sample_id]),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output, index=False)

    summary = {
        "seed": args.seed,
        "min_sequence_identity": 0.30,
        "minimum_coverage": 0.80,
        "protein_chain_records": len(chain_to_sample),
        "complexes": len(samples),
        "linked_components": len(component_ids),
        "split_counts": frame["split"].value_counts().to_dict() if not frame.empty else {},
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
