"""Hashed inverted index for reaction-constrained synthon retrieval (#41).

Issue #41 (DATA-06): trajectory extraction (#18) and policy-time retrieval
must not linearly scan the 50,000-entry synthon catalog for every candidate
reaction.  This module builds and queries a **versioned inverted index**
keyed by ``(reaction_family, handle_type, molecular_formula_hash)`` and —
for the seed action of #39 (MODEL-06) — by the parallel
``("seed", seed_class, molecular_formula_hash)`` namespace.

The index is a **candidate prefilter only** (constitution #45 §9.5): hash
and formula matches are never chemical proof.  Every candidate that leaves
this index must still pass exact handle compatibility and the RDKit
reaction execution of the consumer (#18's exact forward replay); the
:func:`SynthonIndex.exact_candidates` stage performs the exact
canonical-SMILES filter and the *consumer* performs the reaction.

Determinism and versioning (issue-pinned)
-----------------------------------------
* **Canonicalization** — the catalog's salt/counterion form (e.g.
  ``"Br.BrCCN1CCC1"``) is reduced to its **parent fragment** by
  :func:`parent_fragment` (most heavy atoms; ties: most bonds, then
  canonical SMILES ascending — deterministic, documented).  The parent's
  molecular formula is computed in Hill order
  (:func:`parent_formula`) and hashed with SHA-256 truncated to 16 hex
  characters (:func:`formula_hash`).  Truncation makes a collision
  *possible in principle*; the exact stage makes it *harmless* (a colliding
  candidate is rejected by the canonical-SMILES comparison — pinned by
  test).
* **Version metadata** — the persisted artifact carries the index schema
  version, the formula-hash version, the catalog's SHA-256 and schema
  version, the live reaction-registry digest it was built against, and the
  catalog's historical registry-digest attribute.  Loading fails closed
  (a clear :class:`SynthonIndexError`) on any mismatch with the live
  registry or the catalog file it accompanies.  The historical
  ``32ea6db9…`` catalog attribute is preserved as provenance: it was
  produced by the lost original encoder build code and cannot be
  reproduced by the re-landed digest (the registry *source* is provably
  unchanged since ``0e296b0`` — its only commit — so certification
  validity is unaffected; decision D-V2-026 records the reconciliation).
* **Reproducible atomic builds** — the JSON payload is serialized with
  sorted keys and fixed separators, so identical inputs produce
  byte-identical artifacts (the build timestamp lives in the separate
  sidecar manifest, never in the index payload); writes go through
  ``<path>.tmp`` + :func:`os.replace`.

Key space
---------
``family`` is one of the 9 coarse policy families of
``syntree.chemistry.reactions.REACTION_FAMILIES`` (or ``"seed"``).  Only
**block-side** handle classes generate synthon postings: a catalog row is
posted once per (family, certified synthon handle class) pair it carries,
where the (family, class) pair must be justified by at least one live
policy reaction with that ``block_handle_class`` (validated at build
time, fail-closed).  Seed-role rows additionally post their certified
seed classes in the ``"seed"`` namespace.  Molecule-side-only classes
(e.g. a free amine on an electrophile block, useful to the environment's
handle menu after placement) generate no postings — they are not
retrievable block sides.

This module depends on ``rdkit``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from syntree.chemistry.reactions import POLICY_REACTIONS, CertifiedReaction
from syntree.chemistry.synthon_encoder import registry_digest

__all__ = [
    "INDEX_SCHEMA_VERSION",
    "FORMULA_HASH_VERSION",
    "SEED_KEY_FAMILY",
    "SynthonIndexError",
    "SynthonIndex",
    "parent_fragment",
    "parent_formula",
    "formula_hash",
    "formula_hash_of_smiles",
    "exact_candidates",
    "brute_force_lookup",
    "brute_force_seed_lookup",
]

#: The persisted-artifact schema version of the index itself.
INDEX_SCHEMA_VERSION = "synthon-index-v2.0"

#: The canonicalization + hash recipe version (Hill formula -> sha256/16).
FORMULA_HASH_VERSION = "hill-formula-sha256-16-v1"

#: The ``family`` namespace of seed-role keys (#39's seed retrieval).
SEED_KEY_FAMILY = "seed"


class SynthonIndexError(ValueError):
    """Fail-closed index error (version mismatch, malformed build, bad query)."""


# ---------------------------------------------------------------------------
# Canonicalization and hashing (versioned by FORMULA_HASH_VERSION)
# ---------------------------------------------------------------------------

def parent_fragment(smiles: str) -> str:
    """The canonical parent SMILES of a (possibly salt-form) catalog SMILES.

    Deterministic reduction: the connected fragment with the most heavy
    atoms wins; ties resolve by most bonds, then lexicographically smallest
    canonical SMILES.  Single-fragment SMILES pass through canonicalization
    unchanged (the query side of #18 reconstructs salt-free reactants, so
    build and query share this one canonicalization).  Memoized: the
    catalog's ~49k distinct parents repeat across every query of the
    equivalence benchmark (purely a performance cache — the function is
    deterministic in its string argument).
    """
    return _parent_fragment_cached(smiles)


@lru_cache(maxsize=None)
def _parent_fragment_cached(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SynthonIndexError(f"unparseable SMILES {smiles!r}")
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        raise SynthonIndexError(f"SMILES {smiles!r} has no molecular fragments")
    if len(frags) == 1:
        return Chem.MolToSmiles(frags[0])
    ranked = sorted(
        frags,
        key=lambda f: (-f.GetNumHeavyAtoms(), -f.GetNumBonds(), Chem.MolToSmiles(f)),
    )
    return Chem.MolToSmiles(ranked[0])


def parent_formula(parent_smiles: str) -> str:
    """The Hill-order molecular formula of a (parent) SMILES (memoized)."""
    return _parent_formula_cached(parent_smiles)


@lru_cache(maxsize=None)
def _parent_formula_cached(parent_smiles: str) -> str:
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        raise SynthonIndexError(f"unparseable parent SMILES {parent_smiles!r}")
    return rdMolDescriptors.CalcMolFormula(mol)


def formula_hash(formula: str) -> str:
    """SHA-256 of the Hill formula string, truncated to 16 hex characters."""
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()[:16]


def formula_hash_of_smiles(parent_smiles: str) -> str:
    """``formula_hash(parent_formula(parent_smiles))`` for salt-free queries."""
    return formula_hash(parent_formula(parent_smiles))


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthonIndex:
    """The versioned hashed inverted index over one immutable catalog.

    ``postings`` maps ``"family|handle_class|formula_hash"`` to the sorted
    ascending tuple of catalog row indices carrying that key.  ``parents``
    holds each row's canonical parent SMILES (the exact-match stage's
    comparison surface, so the index is self-contained without reopening
    the HDF5).  ``block_ids`` mirrors the catalog row order.
    """

    postings: Dict[str, Tuple[int, ...]]
    parents: Tuple[str, ...]
    block_ids: Tuple[str, ...]
    catalog_sha256: str
    catalog_schema_version: str
    registry_digest_value: str
    catalog_registry_digest: str
    catalog_size: int

    def __post_init__(self) -> None:
        if len(self.parents) != self.catalog_size or len(self.block_ids) != self.catalog_size:
            raise SynthonIndexError(
                f"catalog_size={self.catalog_size} disagrees with "
                f"{len(self.parents)} parents / {len(self.block_ids)} block ids"
            )
        if self.catalog_size and self.catalog_sha256 == "":
            raise SynthonIndexError("a non-empty index must pin its catalog sha256")
        if self.catalog_size and self.registry_digest_value == "":
            raise SynthonIndexError("a non-empty index must pin its registry digest")
        for key, rows in self.postings.items():
            if not rows:
                raise SynthonIndexError(f"posting key {key!r} is empty (fail-closed)")
            if any(not (0 <= r < self.catalog_size) for r in rows):
                raise SynthonIndexError(f"posting key {key!r} references an unknown row")
            if list(rows) != sorted(set(rows)):
                raise SynthonIndexError(
                    f"posting key {key!r} must be a sorted set of row indices"
                )

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        block_ids: Sequence[str],
        smiles: Sequence[str],
        synthon_classes: Sequence[str],
        seed_classes: Sequence[str],
        catalog_sha256: str,
        catalog_schema_version: str,
        catalog_registry_digest: str = "",
        reactions: Optional[Mapping[str, CertifiedReaction]] = None,
    ) -> "SynthonIndex":
        """Build the index from catalog columns (fail-closed at every step).

        Consumers use the *stored* certification labels (the artifact-pinned
        semantics of D-V2-025) — they are never re-derived here.  Every
        synthon posting key must be justified by a live policy reaction
        whose ``block_handle_class`` matches the certified class; a class
        no live reaction can consume as a block side fails the build (the
        catalog would be certified against chemistry the registry no longer
        contains).
        """
        n = len(block_ids)
        if not (len(smiles) == len(synthon_classes) == len(seed_classes) == n):
            raise SynthonIndexError("catalog columns disagree in length")
        if n == 0:
            raise SynthonIndexError("cannot build an index over an empty catalog")

        live_digest = registry_digest()
        registry = dict(reactions) if reactions is not None else dict(POLICY_REACTIONS)

        # (family, block_handle_class) -> reaction ids justifying the prefix.
        justified: Dict[Tuple[str, str], List[str]] = {}
        for reaction in registry.values():
            key = (str(reaction.family), str(reaction.block_handle_class))
            justified.setdefault(key, []).append(reaction.reaction_id)
        block_classes = {cls for (_family, cls) in justified}

        postings: Dict[str, List[int]] = {}
        parents: List[str] = []
        for row in range(n):
            parent = parent_fragment(_decode(smiles[row]))
            parents.append(parent)
            fhash = formula_hash(parent_formula(parent))
            for handle_class in _split_classes(synthon_classes[row]):
                if handle_class not in block_classes:
                    raise SynthonIndexError(
                        f"row {row} ({_decode(block_ids[row])}): certified handle "
                        f"class {handle_class!r} is the block side of no policy "
                        "reaction in the live registry — registry/catalog mismatch "
                        "(fail-closed)"
                    )
                for family in sorted(
                    f for (f, cls) in justified if cls == handle_class
                ):
                    postings.setdefault(f"{family}|{handle_class}|{fhash}", []).append(row)
            for seed_class in _split_classes(seed_classes[row]):
                postings.setdefault(f"{SEED_KEY_FAMILY}|{seed_class}|{fhash}", []).append(row)
        frozen = {key: tuple(sorted(set(rows))) for key, rows in sorted(postings.items())}
        return cls(
            postings=frozen,
            parents=tuple(parents),
            block_ids=tuple(_decode(b) for b in block_ids),
            catalog_sha256=str(catalog_sha256),
            catalog_schema_version=str(catalog_schema_version),
            registry_digest_value=live_digest,
            catalog_registry_digest=str(catalog_registry_digest),
            catalog_size=n,
        )

    # -- querying ------------------------------------------------------------

    def _key(self, family: str, handle_class: str, parent_smiles: str) -> str:
        return f"{family}|{handle_class}|{formula_hash_of_smiles(parent_smiles)}"

    def lookup(self, reaction_id: str, parent_smiles: str) -> Tuple[int, ...]:
        """Candidate rows for a reaction-constrained query (prefilter only).

        ``reaction_id`` must be a certified policy reaction; ``parent_smiles``
        must be the salt-free reactant form of the queried synthon.  Returns
        the posting list for ``(family, block_handle_class, formula_hash)``
        — a superset of the exact matches; consumers must run the exact
        stage (:meth:`exact_candidates`) and their own reaction validation.
        """
        reaction = POLICY_REACTIONS.get(reaction_id)
        if reaction is None:
            raise SynthonIndexError(
                f"unknown reaction_id {reaction_id!r}; certified: {sorted(POLICY_REACTIONS)}"
            )
        return self.postings.get(
            self._key(str(reaction.family), str(reaction.block_handle_class), parent_smiles),
            (),
        )

    def seed_lookup(self, seed_class: str, parent_smiles: str) -> Tuple[int, ...]:
        """Candidate rows for the seed action of #39, by certified seed class."""
        return self.postings.get(
            self._key(SEED_KEY_FAMILY, str(seed_class), parent_smiles), ()
        )

    def exact_candidates(
        self, parent_smiles: str, rows: Iterable[int]
    ) -> Tuple[int, ...]:
        """The exact canonical-parent-SMILES filter over candidate rows.

        This is the stage that makes formula-hash collisions harmless: a
        candidate whose *parent graph* differs from the query is rejected
        here, before any reaction chemistry runs.
        """
        target = parent_fragment(parent_smiles)
        return tuple(r for r in rows if self.parents[r] == target)

    # -- version guards ------------------------------------------------------

    def verify_registry(self) -> None:
        """Fail closed unless the live registry still matches the build."""
        live = registry_digest()
        if live != self.registry_digest_value:
            raise SynthonIndexError(
                f"index/registry version mismatch: the index was built against "
                f"registry digest {self.registry_digest_value!r} but the live "
                f"certified registry hashes to {live!r} — rebuild the index "
                "before querying (fail-closed)"
            )

    def verify_catalog_file(self, catalog_h5_path: str) -> None:
        """Fail closed unless the accompanying HDF5 matches the pinned sha256."""
        digest = hashlib.sha256()
        with open(catalog_h5_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        observed = digest.hexdigest()
        if observed != self.catalog_sha256:
            raise SynthonIndexError(
                f"index/catalog version mismatch: the index pins catalog "
                f"sha256 {self.catalog_sha256!r} but {catalog_h5_path!r} "
                f"hashes to {observed!r} — the row indices are not valid for "
                "this catalog file (fail-closed)"
            )

    # -- persistence ---------------------------------------------------------

    def to_payload(self) -> Dict[str, object]:
        """The byte-deterministic JSON payload (no timestamps)."""
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "formula_hash_version": FORMULA_HASH_VERSION,
            "catalog_schema_version": self.catalog_schema_version,
            "catalog_sha256": self.catalog_sha256,
            "catalog_registry_digest": self.catalog_registry_digest,
            "registry_digest": self.registry_digest_value,
            "catalog_size": self.catalog_size,
            "block_ids": list(self.block_ids),
            "parents": list(self.parents),
            "postings": {key: list(rows) for key, rows in sorted(self.postings.items())},
        }

    def save(self, path: str) -> str:
        """Atomically write the index JSON; returns its sha256."""
        blob = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SynthonIndex":
        """Reconstruct from a payload mapping (version-checked, fail-closed)."""
        try:
            schema = str(payload["schema_version"])
            hash_version = str(payload["formula_hash_version"])
        except KeyError as error:
            raise SynthonIndexError(f"index payload lacks the key {error}") from error
        if schema != INDEX_SCHEMA_VERSION:
            raise SynthonIndexError(
                f"index schema version mismatch: payload is {schema!r}, "
                f"this code implements {INDEX_SCHEMA_VERSION!r} (fail-closed)"
            )
        if hash_version != FORMULA_HASH_VERSION:
            raise SynthonIndexError(
                f"formula-hash version mismatch: payload is {hash_version!r}, "
                f"this code implements {FORMULA_HASH_VERSION!r} (fail-closed)"
            )
        postings_raw = payload.get("postings")
        parents = payload.get("parents")
        block_ids = payload.get("block_ids")
        if not isinstance(postings_raw, dict) or not isinstance(parents, list):
            raise SynthonIndexError("malformed index payload (postings/parents)")
        if not isinstance(block_ids, list):
            raise SynthonIndexError("malformed index payload (block_ids)")
        postings: Dict[str, Tuple[int, ...]] = {}
        for key, rows in postings_raw.items():
            if not isinstance(rows, list) or not all(
                isinstance(r, int) and not isinstance(r, bool) for r in rows
            ):
                raise SynthonIndexError(f"malformed posting list for key {key!r}")
            postings[str(key)] = tuple(int(r) for r in rows)
        catalog_size = int(payload.get("catalog_size", len(parents)))  # type: ignore[arg-type]
        return cls(
            postings=postings,
            parents=tuple(str(p) for p in parents),
            block_ids=tuple(str(b) for b in block_ids),
            catalog_sha256=str(payload.get("catalog_sha256", "")),
            catalog_schema_version=str(payload.get("catalog_schema_version", "")),
            registry_digest_value=str(payload.get("registry_digest", "")),
            catalog_registry_digest=str(payload.get("catalog_registry_digest", "")),
            catalog_size=catalog_size,
        )

    @classmethod
    def load(
        cls,
        path: str,
        *,
        catalog_h5_path: Optional[str] = None,
        verify_registry: bool = True,
    ) -> "SynthonIndex":
        """Load + version-check an index artifact (fail-closed).

        With ``catalog_h5_path`` the accompanying HDF5's sha256 is verified
        against the pinned catalog hash; with ``verify_registry`` the live
        certified registry digest is verified.  Both mismatches raise
        :class:`SynthonIndexError` with an explicit remediation message.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        index = cls.from_payload(payload)
        if verify_registry:
            index.verify_registry()
        if catalog_h5_path is not None:
            index.verify_catalog_file(catalog_h5_path)
        return index

    # -- statistics ----------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        """Index statistics for benchmarks and manifests."""
        num_postings = sum(len(rows) for rows in self.postings.values())
        sizes = [len(rows) for rows in self.postings.values()]
        return {
            "catalog_size": self.catalog_size,
            "num_keys": len(self.postings),
            "num_postings": num_postings,
            "mean_postings_per_key": (
                num_postings / len(self.postings) if self.postings else 0.0
            ),
            "max_postings_per_key": max(sizes) if sizes else 0,
            "seed_keys": sum(
                1 for k in self.postings if k.startswith(f"{SEED_KEY_FAMILY}|")
            ),
            "distinct_parents": len(set(self.parents)),
        }


# ---------------------------------------------------------------------------
# Exact stage (module level) and the brute-force reference
# ---------------------------------------------------------------------------

def exact_candidates(
    index: SynthonIndex, parent_smiles: str, rows: Iterable[int]
) -> Tuple[int, ...]:
    """Module-level exact canonical-SMILES filter (delegates to the index)."""
    return index.exact_candidates(parent_smiles, rows)


def brute_force_lookup(
    *,
    synthon_classes: Sequence[str],
    parents: Sequence[str],
    reaction_id: str,
    query_parent_smiles: str,
) -> Tuple[int, ...]:
    """Linear-scan ground truth for :meth:`SynthonIndex.lookup`.

    Scans every catalog row and keeps those whose certified synthon classes
    contain the reaction's block handle class AND whose parent **formula
    string** equals the query parent's formula — the semantics the index
    keys approximate, computed without touching the postings map or any
    hash.  A formula-hash collision would make the index return rows this
    reference does not (the discrepancy the #41 equivalence benchmark must
    expose).
    """
    reaction = POLICY_REACTIONS.get(reaction_id)
    if reaction is None:
        raise SynthonIndexError(f"unknown reaction_id {reaction_id!r}")
    target_formula = parent_formula(parent_fragment(query_parent_smiles))
    handle = str(reaction.block_handle_class)
    out: List[int] = []
    for row, (classes, parent) in enumerate(zip(synthon_classes, parents)):
        if handle in _split_classes(classes):
            if parent_formula(parent) == target_formula:
                out.append(row)
    return tuple(out)


def brute_force_seed_lookup(
    *,
    seed_classes: Sequence[str],
    parents: Sequence[str],
    seed_class: str,
    query_parent_smiles: str,
) -> Tuple[int, ...]:
    """Linear-scan ground truth for :meth:`SynthonIndex.seed_lookup`."""
    target_formula = parent_formula(parent_fragment(query_parent_smiles))
    out: List[int] = []
    for row, (classes, parent) in enumerate(zip(seed_classes, parents)):
        if str(seed_class) in _split_classes(classes):
            if parent_formula(parent) == target_formula:
                out.append(row)
    return tuple(out)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _split_classes(value: object) -> List[str]:
    text = _decode(value)
    return [part for part in text.split("|") if part]
