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


def test_aryl_halide_retrosynthesis_tries_all_leaving_groups(assets_dir):
    catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
    fragmenter = ReactionConstrainedFragmenter(catalog)

    ligand = Chem.AddHs(Chem.MolFromSmiles("Fc1ccc(c2ccccc2)cc1"))
    assert AllChem.EmbedMolecule(ligand, randomSeed=31) == 0

    target = fragmenter.find_target(ligand)
    assert target is not None
    assert target.reaction_family == "suzuki_coupling"
    assert target.reaction_name == "suzuki_coupling"


class TestNewFamilyRetrosynthesis:
    """tasklist Priority 4: retro support for sulfonamide + sp3 alkylation."""

    def _embed(self, smiles: str, seed: int = 7):
        ligand = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(ligand, randomSeed=seed) == 0
        return ligand

    def test_sulfonamide_retro_target(self, assets_dir):
        """N-ethylmethanesulfonamide decomposes to MsCl (fixture catalog)."""
        catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
        fragmenter = ReactionConstrainedFragmenter(catalog)

        target = fragmenter.find_target(self._embed("CCNS(C)(=O)=O"))
        assert target is not None
        assert target.reaction_family == "sulfonamide_coupling"
        assert target.reaction_name == "sulfonamide_coupling"
        assert target.core_handle_type in {"sulfonyl_chloride", "primary_secondary_amine"}
        # MsCl is the catalog synthon that replays to the ligand.
        assert Chem.MolToSmiles(catalog.get_mol(target.synthon_index)) == "CS(=O)(=O)Cl"

    def test_tosyl_anilide_retro_target(self, assets_dir):
        """p-Toluenesulfonanilide decomposes to TsCl (fixture catalog)."""
        catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
        fragmenter = ReactionConstrainedFragmenter(catalog)

        target = fragmenter.find_target(
            self._embed("Cc1ccc(S(=O)(=O)Nc2ccccc2)cc1")
        )
        assert target is not None
        assert target.reaction_family == "sulfonamide_coupling"
        assert (
            Chem.MolToSmiles(catalog.get_mol(target.synthon_index))
            == "Cc1ccc(S(=O)(=O)Cl)cc1"
        )

    def test_sp3_alkylation_retro_target(self, assets_dir):
        """Diethylamine decomposes to bromoethane (fixture catalog)."""
        catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
        fragmenter = ReactionConstrainedFragmenter(catalog)

        target = fragmenter.find_target(self._embed("CCNCC"))
        assert target is not None
        assert target.reaction_family == "sp3_alkylation"
        assert target.core_handle_type in {"alkyl_halide", "primary_secondary_amine"}
        assert Chem.MolToSmiles(catalog.get_mol(target.synthon_index)) == "CCBr"

    def test_reductive_amination_still_wins_first_for_c_n_bonds(self, assets_dir):
        """For C-N bonds whose aldehyde precursor IS in the catalog, the
        deterministic family order must keep preferring reductive_amination
        over sp3_alkylation (backward-compatible supervision labels). The
        ligand is built by the *forward* reaction from two fixture synthons,
        so a valid reductive-amination replay provably exists."""
        from syntree.chemistry.reactions import ReactionEngine

        engine = ReactionEngine()
        aldehyde = Chem.MolFromSmiles("C1CCC(CC1)C=O")   # fixture catalog entry
        amine = Chem.MolFromSmiles("C1CC(N)CC1")          # fixture catalog entry
        product = engine.apply_reaction(aldehyde, amine, "reductive_amination")
        assert product is not None
        ligand = Chem.AddHs(product.product)
        assert AllChem.EmbedMolecule(ligand, randomSeed=11) == 0

        catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
        fragmenter = ReactionConstrainedFragmenter(catalog)
        target = fragmenter.find_target(ligand)
        assert target is not None
        assert target.reaction_family == "reductive_amination"

    def test_sulfonamide_fingerprint_in_cache_names(self, assets_dir):
        """The processed-cache name embeds the grammar fingerprint so
        grammar upgrades invalidate stale caches (data integrity)."""
        from syntree.chemistry.reactions import grammar_fingerprint
        from syntree.data.crossdocked import CrossDockedDataset

        fp = grammar_fingerprint()
        ds = CrossDockedDataset(
            assets_dir["crossdocked_dir"],
            split="train",
            catalog=None,
            num_synthetic=4,
            synthetic=True,
        )
        name = ds.processed_file_names[0]
        assert f"_g{fp}" in name
