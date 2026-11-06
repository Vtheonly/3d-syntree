"""V2 data layer: canonical supervised decompositions (#15 CHEM-05) and
multi-source preprocessing (#16 DATA-01).

Public surface (modules of the data layer; #17/#18/#19/#20/#41 follow per
the dependency map):

* :mod:`syntree.data.fragmenter` — deterministic canonical tree
  decomposition of ligand graphs into certified-reaction synthon
  fragments (#15 CHEM-05).
* :mod:`syntree.data.preprocessing` — multi-source standardization into
  uniform pocket/ligand artifacts with the unified provenance manifest
  (#16 DATA-01).
"""

from __future__ import annotations

from syntree.data.fragmenter import (
    CanonicalDecomposition,
    DisconnectionStep,
    FragmentRecord,
    canonical_decomposition,
    legal_disconnection_bonds,
)
from syntree.data.preprocessing import (
    LigandRejection,
    ManifestRecord,
    ManifestWriter,
    apply_ligand_filters,
    chain_sequences,
    extract_pocket_lines,
    ligand_from_mol_block,
    ligand_from_pdb_residue,
    standardize_complex,
)

__all__ = [
    "CanonicalDecomposition",
    "DisconnectionStep",
    "FragmentRecord",
    "canonical_decomposition",
    "legal_disconnection_bonds",
    "LigandRejection",
    "ManifestRecord",
    "ManifestWriter",
    "apply_ligand_filters",
    "chain_sequences",
    "extract_pocket_lines",
    "ligand_from_mol_block",
    "ligand_from_pdb_residue",
    "standardize_complex",
]
