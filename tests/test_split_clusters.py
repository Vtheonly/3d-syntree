"""Tests for the #17 DATA-02 split layer: union-find, deterministic
allocation, leakage detection (fail-closed), and fasta extraction."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split_clusters import (  # noqa: E402
    UnionFind,
    allocate_splits,
    build_components,
    extract_fasta,
    load_cluster_pairs,
    verify_no_leakage,
)


# ---------------------------------------------------------------------------
# Union-Find determinism
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_merge_and_components(self):
        uf = UnionFind()
        for k in "abcdef":
            uf.add(k)
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("d", "e")
        comps = uf.components()
        sizes = sorted(sorted(m) for m in comps.values())
        assert sizes == [["a", "b", "c"], ["d", "e"], ["f"]]

    def test_deterministic_root_regardless_of_order(self):
        def build(pairs):
            uf = UnionFind()
            for k in "wxyz":
                uf.add(k)
            for a, b in pairs:
                uf.union(a, b)
            return {frozenset(m) for m in uf.components().values()}

        assert build([("w", "x"), ("x", "y")]) == build([("y", "x"), ("w", "x")])
        assert build([("w", "x"), ("x", "y")]) == build([("x", "w"), ("y", "x")])

    def test_self_union_noop(self):
        uf = UnionFind()
        uf.add("a")
        uf.union("a", "a")
        assert uf.components() == {"a": ["a"]}


# ---------------------------------------------------------------------------
# Components from clusters
# ---------------------------------------------------------------------------

def _complex_seqs(spec):
    return {cid: list(headers) for cid, headers in spec.items()}


class TestBuildComponents:
    def test_shared_cluster_merges_complexes(self):
        complex_seqs = _complex_seqs(
            {"c1": ["seq0"], "c2": ["seq1"], "c3": ["seq2"], "c4": ["seq3"]}
        )
        # cluster A (representative seq0) contains seq0+seq1 -> c1,c2 merge;
        # mmseqs representatives are themselves sequence ids (self-pair line)
        pairs = [("seq0", "seq0"), ("seq0", "seq1"),
                 ("seq2", "seq2"), ("seq3", "seq3")]
        comps, clusters = build_components(complex_seqs, pairs)
        merged = sorted(sorted(m) for m in comps.values())
        assert merged == [["c1", "c2"], ["c3"], ["c4"]]
        assert clusters["seq0"] == ["seq0", "seq1"]

    def test_transitive_merge_across_chains(self):
        # c1 has chains A,B; c2 has B; c3 has C; cluster links A<->C
        complex_seqs = _complex_seqs({"c1": ["sA", "sB"], "c2": ["sB"], "c3": ["sC"]})
        pairs = [("sA", "sA"), ("sA", "sC"), ("sB", "sB")]
        comps, _ = build_components(complex_seqs, pairs)
        merged = sorted(sorted(m) for m in comps.values())
        assert merged == [["c1", "c2", "c3"]]

    def test_diamond_homology_all_merge(self):
        complex_seqs = _complex_seqs({"c1": ["s1"], "c2": ["s1", "s2"], "c3": ["s2"]})
        pairs = [("s1", "s1"), ("s1", "s2")]
        comps, _ = build_components(complex_seqs, pairs)
        assert list(comps.values())[0] == ["c1", "c2", "c3"] or sorted(
            sorted(m) for m in comps.values()
        ) == [["c1", "c2", "c3"]]


# ---------------------------------------------------------------------------
# Deterministic 80/10/10 allocation
# ---------------------------------------------------------------------------

class TestAllocateSplits:
    def test_proportions_and_whole_components(self):
        comps = {f"root{i}": [f"c{i}"] for i in range(100)}
        assignment = allocate_splits(comps)
        from collections import Counter

        counts = Counter(assignment.values())
        assert counts["train"] == 80
        assert counts["val"] == 10
        assert counts["test"] == 10

    def test_large_component_never_split(self):
        comps = {"big": [f"c{i}" for i in range(50)]}
        comps.update({f"r{i}": [f"x{i}"] for i in range(50)})
        assignment = allocate_splits(comps)
        assert len({assignment[f"c{i}"] for i in range(50)}) == 1

    def test_deterministic_repeat(self):
        comps = {f"r{i}": [f"c{i}", f"d{i}"] for i in range(37)}
        a1 = allocate_splits(comps)
        a2 = allocate_splits(comps)
        assert a1 == a2

    def test_tie_break_prefers_train(self):
        # single component -> the first pick must be train (deficit tie)
        assignment = allocate_splits({"r": ["c0"]})
        assert assignment["c0"] == "train"

    def test_size_desc_ordering(self):
        # biggest component first: with 9 complexes in one component + 1
        # singleton, the big one takes train first and the singleton lands
        # where the deficit rule sends it
        comps = {"big": [f"c{i}" for i in range(9)], "s": ["x"]}
        assignment = allocate_splits(comps)
        assert len({assignment[f"c{i}"] for i in range(9)}) == 1


# ---------------------------------------------------------------------------
# Leakage check (fail-closed)
# ---------------------------------------------------------------------------

class TestVerifyNoLeakage:
    def test_clean_split_passes(self):
        complex_seqs = _complex_seqs({"c1": ["s0"], "c2": ["s1"]})
        clusters = {"r0": ["s0"], "r1": ["s1"]}
        assignment = {"c1": "train", "c2": "test"}
        check = verify_no_leakage(complex_seqs, clusters, assignment)
        assert check["clusters_spanning_splits"] == 0

    def test_leaky_split_raises(self):
        # s0 and s1 in one cluster, but their complexes are in different splits
        complex_seqs = _complex_seqs({"c1": ["s0"], "c2": ["s1"]})
        clusters = {"r0": ["s0", "s1"]}
        assignment = {"c1": "train", "c2": "test"}
        with pytest.raises(SystemExit, match="LEAKAGE"):
            verify_no_leakage(complex_seqs, clusters, assignment)

    def test_multi_chain_complex_cannot_leak_by_construction(self):
        # c1 has s0+s1: any valid allocation keeps c1's chains together
        complex_seqs = _complex_seqs({"c1": ["s0", "s1"], "c2": ["s1"]})
        comps, clusters = build_components(
            complex_seqs, [("s0", "s0"), ("s0", "s1")]
        )
        assignment = allocate_splits(comps)
        check = verify_no_leakage(complex_seqs, clusters, assignment)
        assert check["clusters_spanning_splits"] == 0
        assert assignment["c1"] == assignment["c2"]


# ---------------------------------------------------------------------------
# Fasta extraction
# ---------------------------------------------------------------------------

class TestExtractFasta:
    def _write_manifest(self, tmp_path, records):
        path = tmp_path / "processed_manifest.jsonl"
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return str(path)

    def test_extraction_semantics(self, tmp_path):
        records = [
            {
                "complex_id": "c1",
                "status": "accepted",
                "protein_chains": {"B": "SEQB", "A": "SEQA"},
            },
            {  # duplicate sequence -> deduplicated
                "complex_id": "c2",
                "status": "accepted",
                "protein_chains": {"A": "SEQA"},
            },
            {  # rejected -> excluded entirely
                "complex_id": "c3",
                "status": "rejected",
                "protein_chains": {"A": "SEQX"},
            },
            {  # accepted without chains -> recorded with empty list
                "complex_id": "c4",
                "status": "accepted",
                "protein_chains": {},
            },
        ]
        manifest = self._write_manifest(tmp_path, records)
        workdir = str(tmp_path / "work")
        summary = extract_fasta(manifest, workdir)

        assert summary["manifest_records"] == 4
        assert summary["accepted_complexes"] == 3
        assert summary["accepted_without_chains"] == 1
        assert summary["chain_slots"] == 3
        assert summary["unique_sequences"] == 2

        fasta = (Path(workdir) / "all_proteins.fasta").read_text()
        # deterministic chain order: A before B -> SEQA gets seq0000000
        assert ">seq0000000\nSEQA\n>seq0000001\nSEQB\n" == fasta

        seq_map = json.loads((Path(workdir) / "seq_map.json").read_text())
        assert seq_map == {"c1": ["seq0000000", "seq0000001"], "c2": ["seq0000000"], "c4": []}

    def test_fasta_headers_are_unique(self, tmp_path):
        records = [
            {"complex_id": f"c{i}", "status": "accepted",
             "protein_chains": {str(j): f"S{i}_{j}" for j in range(3)}}
            for i in range(20)
        ]
        manifest = self._write_manifest(tmp_path, records)
        workdir = str(tmp_path / "work")
        extract_fasta(manifest, workdir)
        headers = [
            line[1:]
            for line in (Path(workdir) / "all_proteins.fasta").read_text().splitlines()
            if line.startswith(">")
        ]
        assert len(headers) == len(set(headers)) == 60


# ---------------------------------------------------------------------------
# Cluster TSV parsing
# ---------------------------------------------------------------------------

class TestLoadClusterPairs:
    def test_parse_tsv(self, tmp_path):
        tsv = tmp_path / "clusters_cluster.tsv"
        tsv.write_text("rep1\tmem1\nrep1\trep1\nrep2\tmem2\n")
        pairs = load_cluster_pairs(str(tsv))
        assert pairs == [("rep1", "mem1"), ("rep1", "rep1"), ("rep2", "mem2")]

    def test_skips_malformed_lines(self, tmp_path):
        tsv = tmp_path / "c.tsv"
        tsv.write_text("rep1\tmem1\n\nonlyone\n")
        assert load_cluster_pairs(str(tsv)) == [("rep1", "mem1")]
