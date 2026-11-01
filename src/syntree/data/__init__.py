"""V2 data layer: canonical supervised decompositions (#15 CHEM-05).

Public surface (first module of the data layer; #16/#17/#18/#19/#20/#41
follow per the dependency map):

* :mod:`syntree.data.fragmenter` — deterministic canonical tree
  decomposition of ligand graphs into certified-reaction synthon
  fragments (#15 CHEM-05).
"""

from __future__ import annotations

from syntree.data.fragmenter import (
    CanonicalDecomposition,
    DisconnectionStep,
    FragmentRecord,
    canonical_decomposition,
    legal_disconnection_bonds,
)

__all__ = [
    "CanonicalDecomposition",
    "DisconnectionStep",
    "FragmentRecord",
    "canonical_decomposition",
    "legal_disconnection_bonds",
]
