"""Tests for the hashed inverted synthon index (#41 DATA-06).

Issue-pinned acceptance criteria covered here:

1. Indexed retrieval returns the same exact valid reaction/synthon matches
   as the brute-force reference (exhaustive over synthetic catalogs;
   sampled + exhaustive-self-retrieval on the real 50k artifact when
   present).
2. Hash collisions cannot bypass exact chemical validation (a fabricated
   same-formula neighbor is returned by the prefilter but rejected by the
   exact stage).
3. Index and catalog version mismatches fail closed with a clear error
   (schema, formula-hash version, catalog sha256, registry digest).
4. Benchmarks report measured latency/throughput (the driver's protocol;
   the artifact-gated test re-runs it when the catalog HDF5 is present).

The real-artifact tests are gated on ``work/catalog/…h5`` being present
(the same gating convention as ``test_synthon_encoder.py``).
"""

from __future__ import annotations

import json
import os

import pytest

from syntree.chemistry.reactions import POLICY_REACTIONS
from syntree.chemistry.synthon_index import (
    FORMULA_HASH_VERSION,
    INDEX_SCHEMA_VERSION,
    SEED_KEY_FAMILY,
    SynthonIndex,
    SynthonIndexError,
    brute_force_lookup,
    brute_force_seed_lookup,
    exact_candidates,
    formula_hash,
    formula_hash_of_smiles,
    parent_formula,
    parent_fragment,
)

CATALOG_H5 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "work", "catalog", "enamine_real_50k_pharm3d_d256.h5",
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic catalogs (chemically real, small, exhaustive)
# ---------------------------------------------------------------------------

def _synthetic_catalog():
    """A small catalog exercising every key-path of the index.

    * ``acid_block`` — an acid chloride (salt-free) with a synthon class
      consumed by rxn_01/rxn_17/rxn_20.
    * ``acid_block_salt`` — the SAME parent in hydrochloride salt form
      (parent extraction must reduce to the shared parent).
    * ``same_formula_neighbor`` — a *different* molecule sharing the acid
      chloride's formula (the collision-prefilter case).
    * ``seed_amino`` — a seed-role amine (no synthon class).
    """
    return {
        "block_ids": ["B_ACID", "B_ACID_SALT", "B_NEIGHBOR", "B_SEED_AMINE"],
        "smiles": ["CCC(=O)Cl", "CCC(=O)Cl.Cl", "C(Cl)C(=O)C", "Br.CCN"],
        "synthon_classes": ["acid_chloride", "acid_chloride", "acid_chloride", ""],
        "seed_classes": ["", "", "", "amine"],
    }


def _build_synthetic():
    return SynthonIndex.build(
        catalog_sha256="0" * 64,
        catalog_schema_version="catalog-v2.0",
        catalog_registry_digest="historical",
        **_synthetic_catalog(),
    )


# ---------------------------------------------------------------------------
# Canonicalization and hashing
# ---------------------------------------------------------------------------

class TestCanonicalization:
    def test_parent_fragment_reduces_salts(self):
        assert parent_fragment("Br.BrCCN1CCC1") == "BrCCN1CCC1"
        assert parent_fragment("CCC(=O)Cl.Cl") == "CCC(=O)Cl"
        assert parent_fragment("CCN") == "CCN"

    def test_parent_fragment_tie_breaks_deterministically(self):
        # two equal-size fragments: most bonds, then canonical SMILES asc
        a = parent_fragment("CCO.CCN")
        assert a == "CCN"  # CCO (2 bonds) vs CCN (2 bonds): "CCN" < "CCO"

    def test_parent_fragment_rejects_unparseable(self):
        with pytest.raises(SynthonIndexError):
            parent_fragment("not-a-smiles")

    def test_formula_is_hill_order(self):
        assert parent_formula("CCC(=O)Cl") == "C3H5ClO"

    def test_formula_hash_is_versioned_sha256_16(self):
        assert formula_hash("C3H5ClO") == formula_hash("C3H5ClO")
        assert len(formula_hash("C3H5ClO")) == 16
        assert formula_hash("C3H5ClO") != formula_hash("C3H6ClO")
        assert FORMULA_HASH_VERSION == "hill-formula-sha256-16-v1"

    def test_query_hash_of_smiles_matches_parent_path(self):
        # salt-free query and salt-form catalog entry share the hash
        assert formula_hash_of_smiles("CCC(=O)Cl") == formula_hash_of_smiles(
            parent_fragment("CCC(=O)Cl.Cl")
        )


# ---------------------------------------------------------------------------
# Build, keys, and lookup semantics
# ---------------------------------------------------------------------------

class TestBuildAndLookup:
    def test_acid_chloride_posts_under_every_justified_family(self):
        index = _build_synthetic()
        # rxn_01 (amide_and_carbamate), rxn_17 & rxn_20 (ester_and_ether)
        for rid in ("rxn_01", "rxn_17", "rxn_20"):
            rows = index.lookup(rid, "CCC(=O)Cl")
            assert 0 in rows and 1 in rows  # both the salt and plain form

    def test_same_formula_neighbor_is_a_prefilter_candidate(self):
        index = _build_synthetic()
        rows = index.lookup("rxn_01", "CCC(=O)Cl")
        assert 2 in rows  # same formula, different structure: candidate only

    def test_seed_namespace_lookup(self):
        index = _build_synthetic()
        rows = index.seed_lookup("amine", "CCN")
        assert rows == (3,)

    def test_molecule_side_only_class_generates_no_postings(self):
        index = _build_synthetic()
        # 'amine' is a molecule-side class of rxn_01; no synthon posting exists
        assert all(
            not key.startswith("amide_and_carbamate|amine|") for key in index.postings
        )

    def test_unknown_reaction_fails_closed(self):
        index = _build_synthetic()
        with pytest.raises(SynthonIndexError, match="unknown reaction_id"):
            index.lookup("rxn_99", "CCC(=O)Cl")

    def test_unparseable_query_fails_closed(self):
        index = _build_synthetic()
        with pytest.raises(SynthonIndexError, match="unparseable"):
            index.lookup("rxn_01", "not-a-smiles")

    def test_build_fails_on_class_no_reaction_consumes(self):
        bad = _synthetic_catalog()
        bad["synthon_classes"] = ["acid_chloride", "acid_chloride", "acid_chloride",
                                  "no_such_class"]
        bad["seed_classes"] = ["", "", "", ""]
        with pytest.raises(SynthonIndexError, match="block side of no policy reaction"):
            SynthonIndex.build(
                catalog_sha256="0" * 64,
                catalog_schema_version="catalog-v2.0",
                **bad,
            )

    def test_build_fails_on_column_length_mismatch(self):
        bad = _synthetic_catalog()
        bad["block_ids"] = bad["block_ids"][:2]
        with pytest.raises(SynthonIndexError, match="disagree in length"):
            SynthonIndex.build(
                catalog_sha256="0" * 64,
                catalog_schema_version="catalog-v2.0",
                **bad,
            )


# ---------------------------------------------------------------------------
# Acceptance 1+2: brute-force equivalence and collision safety
# ---------------------------------------------------------------------------

class TestBruteForceEquivalence:
    def test_exhaustive_equivalence_on_synthetic_catalog(self):
        index = _build_synthetic()
        cat = _synthetic_catalog()
        queries = [
            (rid, parent)
            for rid in POLICY_REACTIONS
            for parent in ("CCC(=O)Cl", "C(Cl)C(=O)C", "CCN")
        ] + [("seed:amine", "CCN")]
        for reaction_id, parent in queries:
            if reaction_id.startswith("seed:"):
                got = index.seed_lookup(reaction_id[5:], parent)
                want = brute_force_seed_lookup(
                    seed_classes=cat["seed_classes"],
                    parents=list(index.parents),
                    seed_class=reaction_id[5:],
                    query_parent_smiles=parent,
                )
            else:
                got = index.lookup(reaction_id, parent)
                want = brute_force_lookup(
                    synthon_classes=cat["synthon_classes"],
                    parents=list(index.parents),
                    reaction_id=reaction_id,
                    query_parent_smiles=parent,
                )
            assert set(got) == set(want), (reaction_id, parent, got, want)

    def test_collision_cannot_bypass_exact_validation(self):
        # B_NEIGHBOR shares the acid chloride's formula but is a different
        # molecule: the prefilter returns it, the exact stage rejects it.
        index = _build_synthetic()
        rows = index.lookup("rxn_01", "CCC(=O)Cl")
        assert 2 in rows
        exact = exact_candidates(index, "CCC(=O)Cl", rows)
        assert 2 not in exact and set(exact) == {0, 1}
        # and a query for the neighbor itself returns only the neighbor
        rows_n = index.lookup("rxn_01", "C(Cl)C(=O)C")
        assert exact_candidates(index, "C(Cl)C(=O)C", rows_n) == (2,)

    def test_truncated_hash_collision_is_rejected_by_exact_stage(self):
        # Directly forge the worst case: two different formulas whose
        # hashes are forced equal by monkeypatching the hash to a constant
        # (collision by construction — every formula hashes the same).
        import syntree.chemistry.synthon_index as si

        original = si.formula_hash
        try:
            si.formula_hash = lambda formula: "collision"
            cat = _synthetic_catalog()
            # benzoyl chloride: a certified acid chloride with a DIFFERENT
            # formula than CCC(=O)Cl — under the forged hash both formulas
            # land in the same posting key.
            cat["smiles"] = ["CCC(=O)Cl", "CCC(=O)Cl.Cl", "C(Cl)C(=O)C", "O=C(Cl)c1ccccc1"]
            cat["synthon_classes"] = ["acid_chloride"] * 4
            cat["seed_classes"] = ["", "", "", ""]
            index = SynthonIndex.build(
                catalog_sha256="0" * 64,
                catalog_schema_version="catalog-v2.0",
                **cat,
            )
            rows = index.lookup("rxn_01", "CCC(=O)Cl")
            # the different-formula row collides into the posting list...
            assert 3 in rows
            assert si.parent_formula(index.parents[3]) != si.parent_formula("CCC(=O)Cl")
            # ...but the exact stage still rejects it.
            assert 3 not in index.exact_candidates("CCC(=O)Cl", rows)
        finally:
            si.formula_hash = original


# ---------------------------------------------------------------------------
# Acceptance 3: version fail-closed
# ---------------------------------------------------------------------------

class TestVersionFailClosed:
    def _saved(self, tmp_path):
        index = _build_synthetic()
        path = str(tmp_path / "index.json")
        index.save(path)
        return index, path

    def test_schema_version_mismatch_fails_closed(self, tmp_path):
        _index, path = self._saved(tmp_path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["schema_version"] = "synthon-index-v0.1"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with pytest.raises(SynthonIndexError, match="schema version mismatch"):
            SynthonIndex.load(path)

    def test_formula_hash_version_mismatch_fails_closed(self, tmp_path):
        _index, path = self._saved(tmp_path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["formula_hash_version"] = "old-recipe"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        with pytest.raises(SynthonIndexError, match="formula-hash version mismatch"):
            SynthonIndex.load(path)

    def test_catalog_sha_mismatch_fails_closed(self, tmp_path):
        _index, path = self._saved(tmp_path)
        fake = tmp_path / "other_catalog.h5"
        fake.write_bytes(b"not the catalog")
        with pytest.raises(SynthonIndexError, match="index/catalog version mismatch"):
            SynthonIndex.load(path, catalog_h5_path=str(fake))

    def test_registry_digest_mismatch_fails_closed(self, tmp_path, monkeypatch):
        index, path = self._saved(tmp_path)
        # simulate a registry change: live digest differs from the build's
        monkeypatch.setattr(
            "syntree.chemistry.synthon_index.registry_digest",
            lambda: "f" * 64,
        )
        with pytest.raises(SynthonIndexError, match="index/registry version mismatch"):
            SynthonIndex.load(path)

    def test_load_with_live_registry_passes(self, tmp_path):
        _index, path = self._saved(tmp_path)
        loaded = SynthonIndex.load(path)
        assert loaded.catalog_size == 4


# ---------------------------------------------------------------------------
# Persistence: determinism + atomicity
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_rebuild_is_byte_identical(self, tmp_path):
        first = _build_synthetic().save(str(tmp_path / "a.json"))
        second = _build_synthetic().save(str(tmp_path / "b.json"))
        assert first == second

    def test_save_is_atomic_no_tmp_leftover(self, tmp_path):
        path = str(tmp_path / "index.json")
        _build_synthetic().save(path)
        assert not os.path.exists(path + ".tmp")

    def test_roundtrip_preserves_lookups(self, tmp_path):
        index = _build_synthetic()
        path = str(tmp_path / "index.json")
        index.save(path)
        loaded = SynthonIndex.load(path)
        assert loaded.lookup("rxn_01", "CCC(=O)Cl") == index.lookup("rxn_01", "CCC(=O)Cl")
        assert loaded.stats() == index.stats()

    def test_payload_has_no_timestamp(self, tmp_path):
        path = str(tmp_path / "index.json")
        _build_synthetic().save(path)
        with open(path, "r", encoding="utf-8") as handle:
            blob = handle.read()
        assert "created" not in blob and "timestamp" not in blob


# ---------------------------------------------------------------------------
# The real 50k artifact (gated)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(CATALOG_H5), reason="catalog HDF5 not present (artifact-gated)")
class TestRealCatalog:
    def test_full_scale_build_and_exhaustive_self_retrieval(self, tmp_path):
        import hashlib
        import h5py

        with h5py.File(CATALOG_H5, "r") as h5:
            columns = {
                key: [
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in h5[f"catalog/{key}"][()]
                ]
                for key in ("block_ids", "smiles", "synthon_classes", "seed_classes")
            }
            attrs = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in h5.attrs.items()}
        digest = hashlib.sha256()
        with open(CATALOG_H5, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        catalog_sha = digest.hexdigest()

        index = SynthonIndex.build(
            catalog_sha256=catalog_sha,
            catalog_schema_version=attrs.get("catalog_schema_version", ""),
            catalog_registry_digest=attrs.get("registry_digest", ""),
            **columns,
        )
        stats = index.stats()
        assert stats["catalog_size"] == 50000
        assert stats["num_keys"] > 5000
        assert stats["distinct_parents"] > 45000

        # exhaustive self-retrieval: no (row, certified class) match is lost
        class_to_reactions = {}
        for rid, reaction in POLICY_REACTIONS.items():
            class_to_reactions.setdefault(str(reaction.block_handle_class), []).append(rid)
        checks = 0
        for row in range(index.catalog_size):
            for handle_class in (c for c in columns["synthon_classes"][row].split("|") if c):
                for rid in class_to_reactions[handle_class]:
                    assert row in index.lookup(rid, index.parents[row])
                    checks += 1
            for seed_class in (c for c in columns["seed_classes"][row].split("|") if c):
                assert row in index.seed_lookup(seed_class, index.parents[row])
                checks += 1
        assert checks > 60000

        # byte-deterministic rebuild
        path = str(tmp_path / "index.json")
        sha1 = index.save(path)
        again = SynthonIndex.build(
            catalog_sha256=catalog_sha,
            catalog_schema_version=attrs.get("catalog_schema_version", ""),
            catalog_registry_digest=attrs.get("registry_digest", ""),
            **columns,
        )
        assert again.save(str(tmp_path / "again.json")) == sha1

        # catalog sha fail-closed against the real file
        loaded = SynthonIndex.load(path, catalog_h5_path=CATALOG_H5)
        assert loaded.catalog_size == 50000

        # brute-force equivalence on a deterministic sample
        import random

        rng = random.Random(2026)
        pool = []
        for row in range(0, index.catalog_size, 25):
            for handle_class in (c for c in columns["synthon_classes"][row].split("|") if c):
                for rid in class_to_reactions[handle_class]:
                    pool.append((rid, index.parents[row]))
            for seed_class in (c for c in columns["seed_classes"][row].split("|") if c):
                pool.append((f"seed:{seed_class}", index.parents[row]))
        for reaction_id, parent in rng.sample(pool, min(250, len(pool))):
            if reaction_id.startswith("seed:"):
                got = index.seed_lookup(reaction_id[5:], parent)
                want = brute_force_seed_lookup(
                    seed_classes=columns["seed_classes"],
                    parents=list(index.parents),
                    seed_class=reaction_id[5:],
                    query_parent_smiles=parent,
                )
            else:
                got = index.lookup(reaction_id, parent)
                want = brute_force_lookup(
                    synthon_classes=columns["synthon_classes"],
                    parents=list(index.parents),
                    reaction_id=reaction_id,
                    query_parent_smiles=parent,
                )
            assert set(got) == set(want), (reaction_id, parent)

    def test_driver_benchmark_protocol(self, tmp_path):
        import subprocess
        import sys

        out = str(tmp_path / "index.json")
        manifest = str(tmp_path / "manifest.json")
        result = subprocess.run(
            [
                sys.executable,
                "scripts/build_synthon_index.py",
                "--catalog", CATALOG_H5,
                "--out", out,
                "--manifest", manifest,
                "--sample-queries", "500",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        with open(manifest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        measured = data["benchmark"]["measured"]
        # measured (not hardcoded) performance numbers are recorded
        assert measured["lookup_latency_us"] > 0
        assert measured["exact_stage_latency_us"] > 0
        assert measured["mean_candidates_per_query"] < 100  # real reduction
        assert measured["mean_candidate_reduction_factor"] > 100
        assert data["benchmark"]["protocol"]["exhaustive_self_retrieval_checks"] > 60000
        assert data["index_schema_version"] == INDEX_SCHEMA_VERSION
