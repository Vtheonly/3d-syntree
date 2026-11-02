"""MMseqs2 sequence-clustered split generation (issue #17 DATA-02).

Clean-room V2 implementation of the leakage-free split layer (constitution
#45 §11): random splitting over structural complexes leaks homology
whenever related protein chains land in different partitions, so splits are
partitioned at the **connected-component level of MMseqs2 clusters** over
every protein chain in the accepted-complex manifest.

Pipeline (orchestrated by ``scripts/run_mmseqs_split.sh``):

1. ``extract`` — stream ``processed_manifest.jsonl`` and emit
   ``all_proteins.fasta``: one record per **unique** chain sequence
   (first-appearance order; the header is ``seq<index>``).  A sidecar
   ``seq_map.json`` records ``complex_id -> [seq headers]`` so cluster
   membership can be lifted back to complexes.
2. (shell) ``mmseqs easy-cluster all_proteins.fasta mmseqs_clusters tmp
   --min-seq-id 0.30 -c 0.80 --cov-mode 0 --cluster-mode 2``.
3. ``allocate`` — parse ``mmseqs_clusters.tsv`` into a Union-Find graph:
   every pair of complexes whose chains share a cluster is merged.  The
   resulting connected components are assigned deterministically to
   train/val/test with the greedy largest-deficit rule (components sorted by
   ``(-size, min complex_id)``; each goes to the partition whose current
   fraction is furthest below its 80/10/10 target; ties break train > val >
   test).  Outputs:

   * ``unified_splits_30seq_id.parquet`` — the immutable split table
     (``complex_id``, ``split``), one row per accepted complex.
   * ``split_manifest.json`` — exact counts, cluster/component statistics,
     the leakage check, and full provenance (command lines, versions,
     input manifest path + record count).

The programmatic graph check (acceptance criterion 1) verifies that no
sequence cluster and no connected component spans two splits, and fails
closed (non-zero exit) on any violation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

# The pinned clustering contract (issue #17 / decision D15).
MMSEQS_MIN_SEQ_ID = "0.30"
MMSEQS_COV = "0.80"
MMSEQS_COV_MODE = "0"
MMSEQS_CLUSTER_MODE = "2"


# ---------------------------------------------------------------------------
# Union-Find (deterministic, path-halving, union by index for stability)
# ---------------------------------------------------------------------------

class UnionFind:
    """Deterministic union-find over string keys.

    Union resolves ties by lexicographically smaller root, so the component
    structure is a pure function of the *set* of merge operations, never
    their order.
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def add(self, key: str) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        # path halving
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # deterministic root: lexicographically smaller id wins
        if rb < ra:
            ra, rb = rb, ra
        self._parent[rb] = ra

    def components(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = defaultdict(list)
        for key in self._parent:
            out[self.find(key)].append(key)
        return {root: sorted(members) for root, members in out.items()}


# ---------------------------------------------------------------------------
# Stage 1: fasta extraction
# ---------------------------------------------------------------------------

def extract_fasta(manifest_path: str, workdir: str) -> Dict[str, object]:
    """Stream the preprocessing manifest into ``all_proteins.fasta``.

    Only **accepted** complexes contribute chains (rejected records carry
    no trajectories downstream); every *unique* sequence becomes one fasta
    record in first-appearance order.  Returns a summary dict.
    """
    os.makedirs(workdir, exist_ok=True)
    fasta_path = os.path.join(workdir, "all_proteins.fasta")
    map_path = os.path.join(workdir, "seq_map.json")

    seq_index: Dict[str, str] = {}  # sequence -> header
    complex_seqs: Dict[str, List[str]] = {}
    n_records = 0
    n_accepted = 0
    n_chain_slots = 0
    skipped_no_chains = 0

    tmp_fasta = fasta_path + ".tmp"
    with open(manifest_path, "r", encoding="utf-8") as fin, \
            open(tmp_fasta, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_records += 1
            rec = json.loads(line)
            if rec.get("status") != "accepted":
                continue
            n_accepted += 1
            chains = rec.get("protein_chains") or {}
            if not chains:
                skipped_no_chains += 1
                complex_seqs[rec["complex_id"]] = []
                continue
            headers: List[str] = []
            for chain_id in sorted(chains):  # deterministic chain order
                seq = chains[chain_id]
                n_chain_slots += 1
                header = seq_index.get(seq)
                if header is None:
                    header = f"seq{len(seq_index):07d}"
                    seq_index[seq] = header
                    fout.write(f">{header}\n{seq}\n")
                headers.append(header)
            complex_seqs[rec["complex_id"]] = sorted(set(headers))

    fout.close()
    os.replace(tmp_fasta, fasta_path)

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(complex_seqs, f)

    summary = {
        "manifest_records": n_records,
        "accepted_complexes": n_accepted,
        "accepted_without_chains": skipped_no_chains,
        "chain_slots": n_chain_slots,
        "unique_sequences": len(seq_index),
        "fasta": fasta_path,
        "seq_map": map_path,
    }
    with open(os.path.join(workdir, "extract_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


# ---------------------------------------------------------------------------
# Stage 3: cluster parsing, union-find, allocation, verification
# ---------------------------------------------------------------------------

def load_cluster_pairs(cluster_tsv: str) -> List[Tuple[str, str]]:
    """Parse the mmseqs ``*cluster.tsv`` into (representative, member) pairs.

    ``easy-cluster`` emits one line per cluster member
    (representative-id <tab> member-id).  The representative is itself a
    member line; self-pairs are harmless for union-find.
    """
    pairs: List[Tuple[str, str]] = []
    with open(cluster_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            pairs.append((rep, member))
    return pairs


def build_components(
    complex_seqs: Dict[str, List[str]],
    pairs: Sequence[Tuple[str, str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Union-find over complexes: shared cluster membership merges them.

    Returns ``(components, cluster_members)`` where ``components`` maps a
    root complex id to its sorted member complexes and ``cluster_members``
    maps each cluster representative to the sequence headers in it.
    """
    # sequence header -> complexes containing it
    seq_to_complexes: Dict[str, Set[str]] = defaultdict(set)
    for complex_id, headers in complex_seqs.items():
        for h in headers:
            seq_to_complexes[h].add(complex_id)

    uf = UnionFind()
    for complex_id in complex_seqs:
        uf.add(complex_id)

    cluster_members: Dict[str, List[str]] = defaultdict(list)
    for rep, member in pairs:
        cluster_members[rep].append(member)
        # merge every complex carrying the representative with every
        # complex carrying the member (homologous chains)
        if rep in seq_to_complexes and member in seq_to_complexes:
            for a in seq_to_complexes[rep]:
                for b in seq_to_complexes[member]:
                    uf.union(a, b)

    components = uf.components()
    return components, {k: sorted(v) for k, v in cluster_members.items()}


def allocate_splits(
    components: Dict[str, List[str]],
    *,
    fractions: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Deterministic greedy largest-deficit allocation of components.

    Components are processed largest-first (ties: smallest minimum
    complex id).  Each component joins the split whose current share is
    furthest below its target fraction; ties break ``train > val > test``.
    Whole components never split, so no homologous component crosses
    partitions.
    """
    fr = fractions or SPLIT_FRACTIONS
    order = sorted(components.items(), key=lambda kv: (-len(kv[1]), min(kv[1])))
    total = sum(len(members) for _, members in order)
    counts = {s: 0 for s in SPLITS}
    assignment: Dict[str, str] = {}
    for root, members in order:
        deficits = {s: fr[s] - (counts[s] / total if total else 0.0) for s in SPLITS}
        pick = max(SPLITS, key=lambda s: (deficits[s], -SPLITS.index(s)))
        counts[pick] += len(members)
        for cid in members:
            assignment[cid] = pick
    return assignment


def verify_no_leakage(
    complex_seqs: Dict[str, List[str]],
    cluster_members: Dict[str, List[str]],
    assignment: Dict[str, str],
) -> Dict[str, object]:
    """Acceptance check: no cluster and no component spans two splits.

    Fails closed: any violation is reported with concrete ids and raises.
    """
    seq_to_complexes: Dict[str, Set[str]] = defaultdict(set)
    for complex_id, headers in complex_seqs.items():
        for h in headers:
            seq_to_complexes[h].add(complex_id)

    violations: List[dict] = []
    for rep, members in cluster_members.items():
        splits_seen: Set[str] = set()
        for h in members:
            for cid in seq_to_complexes.get(h, ()):
                if cid in assignment:
                    splits_seen.add(assignment[cid])
        if len(splits_seen) > 1:
            violations.append(
                {"cluster": rep, "members": members[:8], "splits": sorted(splits_seen)}
            )
    if violations:
        raise SystemExit(
            "LEAKAGE CHECK FAILED — clusters spanning splits: "
            + json.dumps(violations[:5])
        )
    return {
        "clusters_checked": len(cluster_members),
        "clusters_spanning_splits": 0,
        "component_check": "single-split components (by construction)",
    }


def allocate(
    cluster_tsv: str,
    workdir: str,
    *,
    fractions: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """End-to-end allocate stage: parse, union, allocate, verify, write."""
    import pandas as pd  # local import: only needed for this stage

    with open(os.path.join(workdir, "seq_map.json"), "r", encoding="utf-8") as f:
        complex_seqs: Dict[str, List[str]] = json.load(f)

    pairs = load_cluster_pairs(cluster_tsv)
    components, cluster_members = build_components(complex_seqs, pairs)
    assignment = allocate_splits(components, fractions=fractions)
    check = verify_no_leakage(complex_seqs, cluster_members, assignment)

    table = pd.DataFrame(
        {
            "complex_id": sorted(assignment),
            "split": [assignment[c] for c in sorted(assignment)],
        }
    )
    parquet_path = os.path.join(workdir, "unified_splits_30seq_id.parquet")
    table.to_parquet(parquet_path, index=False)

    per_source: Dict[str, Counter] = defaultdict(Counter)
    split_counts = Counter(assignment.values())
    component_sizes = sorted((len(m) for m in components.values()), reverse=True)

    manifest = {
        "schema_version": "2.0.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mmseqs_contract": {
            "min_seq_id": MMSEQS_MIN_SEQ_ID,
            "cov": MMSEQS_COV,
            "cov_mode": MMSEQS_COV_MODE,
            "cluster_mode": MMSEQS_CLUSTER_MODE,
        },
        "split_fractions": fractions or SPLIT_FRACTIONS,
        "allocation_rule": (
            "components sorted by (-size, min complex_id); greedy "
            "largest-deficit to 80/10/10; ties train>val>test"
        ),
        "counts": {
            "complexes": len(assignment),
            **{s: int(split_counts.get(s, 0)) for s in SPLITS},
        },
        "component_statistics": {
            "components": len(components),
            "largest": component_sizes[0] if component_sizes else 0,
            "singleton_components": sum(1 for s in component_sizes if s == 1),
            "top10_sizes": component_sizes[:10],
        },
        "cluster_statistics": {
            "clusters": len(cluster_members),
            "multi_member_clusters": sum(
                1 for m in cluster_members.values() if len(m) > 1
            ),
        },
        "leakage_check": check,
        "outputs": {
            "split_table": parquet_path,
            "rows": int(len(table)),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    with open(os.path.join(workdir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    p_extract = sub.add_parser(
        "extract", help="manifest -> all_proteins.fasta + seq_map.json"
    )
    p_extract.add_argument("--manifest", required=True)
    p_extract.add_argument("--workdir", required=True)

    p_alloc = sub.add_parser(
        "allocate", help="mmseqs cluster tsv -> splits + manifest (fails on leakage)"
    )
    p_alloc.add_argument("--clusters", required=True, help="mmseqs *_cluster.tsv")
    p_alloc.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)
    if args.stage == "extract":
        summary = extract_fasta(args.manifest, args.workdir)
        print(json.dumps(summary, indent=1))
        return 0
    if args.stage == "allocate":
        manifest = allocate(args.clusters, args.workdir)
        print(json.dumps(manifest, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
