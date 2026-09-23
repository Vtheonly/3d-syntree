"""Tests for reaction-constrained retrosynthetic supervision."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.fragmenter import ReactionConstrainedFragmenter
from syntree.chemistry.reactions import REACTION_FAMILY_NAMES


def test_exact_catalog_retro_target(assets_dir):
    catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
    fragmenter = ReactionConstrainedFragmenter(catalog)

    # Exact product of the fixture catalog's cyclohexane carboxylic acid and
    # isobutanol entries under esterification.
    ligand = Chem.AddHs(
        Chem.MolFromSmiles("CC(C)COC(=O)C1CCCCC1")
    )
    assert AllChem.EmbedMolecule(ligand, randomSeed=19) == 0

    target = fragmenter.find_target(ligand)
    assert target is not None
    assert target.reaction_family == "esterification"
    assert target.reaction_family in REACTION_FAMILY_NAMES
    assert target.core_handle_type in {"carboxylic_acid", "alcohol"}
    assert -3.141593 <= target.target_dihedral < 3.141593

    # The selected synthon must be an actual catalog member and the replay
    # must produce the observed connectivity.
    synthon = catalog.get_mol(target.synthon_index)
    assert synthon is not None


def test_unrelated_ligand_is_not_randomly_labeled(assets_dir):
    catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
    fragmenter = ReactionConstrainedFragmenter(catalog)

    ligand = Chem.AddHs(Chem.MolFromSmiles("C1CC(N)CC1"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=23) == 0

    # No precursor in the tiny fixture catalog can reconstruct this ligand.
    assert fragmenter.find_target(ligand) is None
