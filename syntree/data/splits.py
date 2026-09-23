"""Leakage-resistant protein-family splits for heterogeneous SBDD datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple


def _stable_hash(value: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest(),
        "big",
    )


def write_fasta(records: Iterable[Mapping[str, str]], fasta_path: str) -> Dict[str, str]:
    """Write unique protein sequences and return sequence_id -> sequence."""
    seen = {}
    os.makedirs(os.path.dirname(os.path.abspath(fasta_path)) or ".", exist_ok=True)
    with open(fasta_path, "w", encoding="utf-8") as handle:
        for record in records:
            seq_id = str(record["protein_key"])
            sequence = "".join(str(record["sequence"]).split()).upper()
            if not sequence:
                raise ValueError(f"Missing sequence for protein {seq_id}")
            if seq_id in seen and seen[seq_id] != sequence:
                raise ValueError(f"Conflicting sequences for protein_key={seq_id}")
            if seq_id in seen:
                continue
            seen[seq_id] = sequence
            handle.write(f">{seq_id}\n{sequence}\n")
    return seen


def cluster_with_mmseqs2(
    fasta_path: str,
    work_dir: str,
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    mmseqs_bin: str = "mmseqs",
) -> Dict[str, str]:
    """Cluster proteins with MMseqs2 and return member -> representative.

    Production clustering fails closed when MMseqs2 is unavailable or fails.
    """
    if not 0.0 < min_seq_id <= 1.0:
        raise ValueError("min_seq_id must be in (0, 1]")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    os.makedirs(work_dir, exist_ok=True)
    prefix = os.path.join(work_dir, "mmseqs_clusters")
    result = subprocess.run(
        [
            mmseqs_bin, "easy-cluster", fasta_path, prefix, work_dir,
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", "0",
            "--cluster-mode", "2",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "MMseqs2 clustering failed. Install MMseqs2 or provide a precomputed "
            f"cluster file. stderr: {result.stderr.strip()}"
        )
    cluster_tsv = prefix + "_cluster.tsv"
    if not os.path.exists(cluster_tsv):
        raise RuntimeError(f"MMseqs2 did not produce {cluster_tsv}")

    clusters: Dict[str, str] = {}
    with open(cluster_tsv, "r", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 2:
                continue
            representative, member = row[0].strip(), row[1].strip()
            clusters[member] = representative

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0]
                clusters.setdefault(seq_id, seq_id)

    if not clusters:
        raise RuntimeError("MMseqs2 cluster output was empty")
    return clusters


def deterministic_cluster_assignments(
    clusters: Mapping[str, str],
    record_rows: Sequence[Mapping[str, object]],
    seed: int = 42,
    fractions: Tuple[float, float, float] = (0.80, 0.10, 0.10),
) -> Dict[str, str]:
    """Assign whole protein clusters to train/val/test with no overlap."""
    if any(f <= 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("split fractions must be positive and sum to 1")

    members_by_cluster: MutableMapping[str, list] = defaultdict(list)
    for row in record_rows:
        key = str(row["protein_key"])
        if key not in clusters:
            raise ValueError(f"Missing MMseqs2 cluster assignment for {key}")
        members_by_cluster[clusters[key]].append(str(row["complex_id"]))

    total = max(1, len(record_rows))
    targets = [fractions[0] * total, fractions[1] * total, fractions[2] * total]
    counts = [0, 0, 0]
    names = ["train", "val", "test"]
    assignment: Dict[str, str] = {}

    cluster_items = sorted(
        members_by_cluster.items(),
        key=lambda item: (-len(item[1]), _stable_hash(item[0], seed)),
    )
    for cluster_id, complexes in cluster_items:
        deficits = [targets[i] - counts[i] for i in range(3)]
        best = max(range(3), key=lambda i: (deficits[i], -counts[i], -i))
        for complex_id in complexes:
            assignment[complex_id] = names[best]
        counts[best] += len(complexes)

    return assignment


def assign_with_locked_splits(
    clusters: Mapping[str, str],
    record_rows: Sequence[Mapping[str, object]],
    locked_splits: Mapping[str, str],
    seed: int = 42,
    fractions: Tuple[float, float, float] = (0.80, 0.10, 0.10),
) -> Dict[str, str]:
    """Assign train/val/test while keeping external benchmarks cluster-locked.

    A cluster containing a locked benchmark complex is entirely excluded from
    the train/val/test allocation and receives that benchmark split. This
    prevents a CASF-like test set from leaking through a homologous protein in
    another source.
    """
    cluster_locked: Dict[str, str] = {}
    for complex_id, split in locked_splits.items():
        row = next((r for r in record_rows if str(r["complex_id"]) == str(complex_id)), None)
        if row is None:
            raise ValueError(f"Locked complex {complex_id} is absent from records")
        cluster_id = clusters[str(row["protein_key"])]
        previous = cluster_locked.get(cluster_id)
        if previous is not None and previous != split:
            raise ValueError(
                f"Cluster {cluster_id} has conflicting locked splits: {previous} vs {split}"
            )
        cluster_locked[cluster_id] = str(split)

    unlocked_rows = [
        row for row in record_rows
        if clusters[str(row["protein_key"])] not in cluster_locked
    ]
    assignment = deterministic_cluster_assignments(
        clusters, unlocked_rows, seed=seed, fractions=fractions
    )

    for row in record_rows:
        cluster_id = clusters[str(row["protein_key"])]
        if cluster_id in cluster_locked:
            assignment[str(row["complex_id"])] = cluster_locked[cluster_id]

    return assignment


def assert_no_cluster_overlap(
    clusters: Mapping[str, str],
    complex_to_split: Mapping[str, str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    seen: Dict[str, str] = {}
    for row in rows:
        complex_id = str(row["complex_id"])
        protein_key = str(row["protein_key"])
        cluster_id = clusters[protein_key]
        split = complex_to_split[complex_id]
        previous = seen.get(cluster_id)
        if previous is not None and previous != split:
            raise AssertionError(
                f"Protein cluster {cluster_id} appears in both {previous} and {split}"
            )
        seen[cluster_id] = split


def save_split_manifest(
    path: str,
    complex_to_split: Mapping[str, str],
    clusters: Mapping[str, str],
    records: Sequence[Mapping[str, object]],
    *,
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    seed: int = 42,
) -> str:
    payload = {
        "version": 2,
        "split_policy": {
            "method": "MMseqs2",
            "min_sequence_identity": min_seq_id,
            "coverage": coverage,
            "seed": seed,
            "cluster_locked_benchmarks": sorted(
                {
                    value
                    for value in complex_to_split.values()
                    if value not in {"train", "val", "test"}
                }
            ),
        },
        "assignments": dict(sorted(complex_to_split.items())),
        "cluster_by_protein": dict(sorted(clusters.items())),
        "counts": {
            split: sum(1 for value in complex_to_split.values() if value == split)
            for split in sorted(set(complex_to_split.values()))
        },
        "sources": sorted({str(row["source"]) for row in records}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


__all__ = [
    "write_fasta",
    "cluster_with_mmseqs2",
    "deterministic_cluster_assignments",
    "assign_with_locked_splits",
    "assert_no_cluster_overlap",
    "save_split_manifest",
]
