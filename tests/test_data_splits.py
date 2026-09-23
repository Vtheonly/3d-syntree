"""Tests for leakage-resistant heterogeneous dataset splits."""
import json
from pathlib import Path

import pytest

from syntree.data.splits import (
    assert_no_cluster_overlap,
    assign_with_locked_splits,
    deterministic_cluster_assignments,
    write_fasta,
)


def _rows():
    return [
        {"complex_id": "a", "source": "crossdocked", "protein_key": "p1", "sequence": "AAAA"},
        {"complex_id": "b", "source": "pdbbind", "protein_key": "p2", "sequence": "BBBB"},
        {"complex_id": "c", "source": "bindingmoad", "protein_key": "p3", "sequence": "CCCC"},
        {"complex_id": "d", "source": "crossdocked", "protein_key": "p4", "sequence": "DDDD"},
    ]


def test_whole_clusters_never_split():
    rows = _rows()
    clusters = {"p1": "cluster-1", "p2": "cluster-1", "p3": "cluster-2", "p4": "cluster-3"}
    assignments = deterministic_cluster_assignments(
        clusters, rows, seed=42, fractions=(0.5, 0.25, 0.25)
    )
    assert assignments["a"] == assignments["b"]
    assert_no_cluster_overlap(clusters, assignments, rows)


def test_locked_benchmark_removes_entire_cluster():
    rows = _rows()
    clusters = {"p1": "cluster-1", "p2": "cluster-1", "p3": "cluster-2", "p4": "cluster-3"}
    assignments = assign_with_locked_splits(
        clusters, rows, {"c": "casf2016_test"}, seed=42
    )
    assert assignments["c"] == "casf2016_test"
    assert assignments["a"] == assignments["b"]
    assert assignments["a"] != "casf2016_test"
    assert_no_cluster_overlap(clusters, assignments, rows)


def test_locked_conflict_fails():
    rows = _rows()
    clusters = {"p1": "cluster-1", "p2": "cluster-1", "p3": "cluster-1", "p4": "cluster-3"}
    with pytest.raises(ValueError, match="conflicting locked splits"):
        assign_with_locked_splits(
            clusters, rows, {"a": "casf2016_test", "c": "other_test"}
        )


def test_write_fasta_rejects_conflicting_sequences(tmp_path: Path):
    rows = [
        {"protein_key": "same", "sequence": "AAAA"},
        {"protein_key": "same", "sequence": "BBBB"},
    ]
    with pytest.raises(ValueError, match="Conflicting sequences"):
        write_fasta(rows, str(tmp_path / "proteins.fasta"))


def test_split_manifest_is_valid_json(tmp_path: Path):
    path = tmp_path / "manifest.json"
    payload = {
        "assignments": {"a": "train", "b": "test"},
        "cluster_by_protein": {"p1": "c1", "p2": "c2"},
    }
    path.write_text(json.dumps(payload))
    assert json.loads(path.read_text())["assignments"]["a"] == "train"
