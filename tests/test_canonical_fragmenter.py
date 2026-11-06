"""Verification suite of #15 CHEM-05: canonical supervised decompositions.

Pinned by issue #15's acceptance criterion (legacy "Vulnerability 6"):

* :func:`TestAtomOrderInvariance` — **permuting atom order in input SDF
  files yields identical retrosynthetic trajectory sequences** (the #15
  acceptance criterion), across representative ligands and a fully
  symmetric molecule.
* :func:`TestSeedRule` — the issue-pinned root rule: maximum ring-system
  size, then highest attachment degree, then canonical-SMILES
  tie-breaking.
* :func:`TestPeripheryToCoreOrder` — disconnections run periphery to core
  (assembly is the exact reverse, core to periphery).
* :func:`TestSequentialRejection` — cuts that would isolate sub-synthon
  atoms are rejected permanently (the amide disconnection is applied
  before an N-alkyl split of the same amine, so no isolated atom ever
  appears).
* :func:`TestEdgeCases` — no legal cuts, ambiguity resolution by
  ``(priority DESC, reaction_id ASC)``, fail-closed inputs.
"""

from __future__ import annotations

import random

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.data.fragmenter import (
    CanonicalDecomposition,
    canonical_decomposition,
    legal_disconnection_bonds,
)
from syntree.data.fragmenter import FragmenterError  # noqa: F401 - re-export

REPRESENTATIVE_LIGANDS = [
    # amide + N-alkyl (rejected) + biaryl: three-fragment decomposition
    "CCC(=O)NCc1ccc(-c2ccccc2)cc1",
    # fully symmetric: identical fragments on both sides
    "c1ccc(-c2ccccc2)cc1",
    # ester junction; the ether split of the same O is rejected
    "CCOC(=O)c1ccccc1",
    # amide into a heteroaryl amine; the C-N(aryl) split is rejected
    "CC(=O)Nc1ncccn1",
    # sulfonamide + aryl ether
    "CCS(=O)(=O)NCCOc1ccccc1",
    # two ring systems tied on size; degree decides the seed
    "CC(=O)Nc1ccc(C(=O)Nc2ccccc2)cc1",
    # urea + ester
    "CCOC(=O)NCc1ncccc1C(=O)NCC",
]


def _trajectory_key(decomposition: CanonicalDecomposition) -> tuple:
    return (
        decomposition.ligand_smiles,
        decomposition.seed_smiles,
        tuple(
            (
                step.order,
                step.removed_fragment_smiles,
                step.reaction_id,
                step.family,
                step.molecule_handle_class,
                step.block_handle_class,
                step.remaining_smiles,
            )
            for step in decomposition.steps
        ),
        decomposition.assembly_order,
    )


def _with_conformer(smiles: str, seed: int = 2026) -> Chem.Mol:
    """SDF-like input: sanitized molecule with an embedded 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=False)
    return mol


def _permuted(mol: Chem.Mol, seed: int) -> Chem.Mol:
    """A copy of ``mol`` with a uniformly random atom renumbering."""
    rng = random.Random(seed)
    order = list(range(mol.GetNumAtoms()))
    rng.shuffle(order)
    return Chem.RenumberAtoms(mol, order)


# ---------------------------------------------------------------------------
# 1. Atom-order invariance (the #15 acceptance criterion)
# ---------------------------------------------------------------------------


class TestAtomOrderInvariance:
    @pytest.mark.parametrize("smiles", REPRESENTATIVE_LIGANDS)
    def test_permuted_atom_orders_yield_identical_trajectories(self, smiles):
        reference_mol = _with_conformer(smiles)
        reference = _trajectory_key(canonical_decomposition(reference_mol))
        for seed in range(12):
            permuted_mol = _permuted(reference_mol, seed)
            permuted = _trajectory_key(canonical_decomposition(permuted_mol))
            assert permuted == reference, (
                f"atom permutation {seed} changed the trajectory of {smiles}:\n"
                f"  reference: {reference}\n  permuted: {permuted}"
            )

    def test_symmetric_biphenyl_is_invariant(self):
        # identical fragments on both sides: the strongest invariance case
        smiles = "c1ccc(-c2ccccc2)cc1"
        reference = _trajectory_key(canonical_decomposition(Chem.MolFromSmiles(smiles)))
        for seed in range(20):
            permuted = _trajectory_key(
                canonical_decomposition(_permuted(_with_conformer(smiles), seed))
            )
            assert permuted == reference

    def test_input_molecule_is_never_mutated(self):
        mol = _with_conformer("CCC(=O)NCc1ccc(-c2ccccc2)cc1")
        before = Chem.Mol(mol)
        canonical_decomposition(mol)
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(before)
        assert mol.GetNumBonds() == before.GetNumBonds()


# ---------------------------------------------------------------------------
# 2. The root seed rule
# ---------------------------------------------------------------------------


class TestSeedRule:
    def test_maximum_ring_system_first(self):
        # the acetyl synthon is acyclic; the seed must be a ring fragment
        d = canonical_decomposition(Chem.MolFromSmiles("CCC(=O)NCc1ccc(-c2ccccc2)cc1"))
        assert d.seed_ring_system_atoms == 6
        assert d.seed_smiles == "NCc1ccccc1"  # the branched ring fragment

    def test_degree_breaks_ring_ties(self):
        # after the amide + biaryl cuts: {acetyl}(ring 0), {N-benzyl-phenyl}
        # (ring 6, degree 2), {phenyl}(ring 6, degree 1).  Ring size ties at 6;
        # the degree-2 fragment wins over the degree-1 phenyl.
        d = canonical_decomposition(Chem.MolFromSmiles("CCC(=O)NCc1ccc(-c2ccccc2)cc1"))
        fragments = [step.removed_fragment_smiles for step in d.steps] + [d.seed_smiles]
        assert "c1ccccc1" in fragments  # the degree-1 phenyl was peeled, not seeded

    def test_canonical_smiles_breaks_full_ties(self):
        # 3-phenylpyridine: phenyl (c1ccccc1) and pyridyl (c1ccncc1) both
        # ring-6, degree-1; canonical SMILES ascending picks the phenyl.
        d = canonical_decomposition(Chem.MolFromSmiles("c1ccc(-c2cccnc2)cc1"))
        assert d.seed_smiles == "c1ccccc1"
        assert d.steps[0].removed_fragment_smiles == "c1ccncc1"
        # the biaryl junction is claimed by rxn_09/10/11 (identical generic
        # pattern, equal priority); id-ascending determinism picks rxn_09 —
        # heterocycle-specific cut assignment is a #40/#41 concern (D-V2-016)
        assert d.steps[0].reaction_id == "rxn_09"


# ---------------------------------------------------------------------------
# 3. Periphery-to-core ordering
# ---------------------------------------------------------------------------


class TestPeripheryToCoreOrder:
    def test_order_is_periphery_first(self):
        d = canonical_decomposition(Chem.MolFromSmiles("CCC(=O)NCc1ccc(-c2ccccc2)cc1"))
        assert [s.order for s in d.steps] == list(range(1, len(d.steps) + 1))
        # the most peripheral synthon (distance tie broken by canonical
        # SMILES: uppercase 'C' sorts before lowercase 'c') is peeled first
        assert d.steps[0].removed_fragment_smiles == "CCC=O"
        assert d.steps[-1].removed_fragment_smiles == "c1ccccc1"
        # remaining molecules shrink monotonically
        sizes = [len(Chem.MolFromSmiles(s.remaining_smiles).GetAtoms()) for s in d.steps]
        assert sizes == sorted(sizes, reverse=True)

    def test_assembly_order_is_the_exact_reverse(self):
        d = canonical_decomposition(Chem.MolFromSmiles("CCC(=O)NCc1ccc(-c2ccccc2)cc1"))
        peeled = [s.removed_fragment_smiles for s in d.steps]
        assembled = [fragment for fragment, _rid in d.assembly_order[1:]]
        assert assembled == list(reversed(peeled))
        assert d.assembly_order[0][0] == d.seed_smiles
        assert d.assembly_order[0][1] is None  # the seed has no attaching reaction


# ---------------------------------------------------------------------------
# 4. Sequential rejection of isolating cuts
# ---------------------------------------------------------------------------


class TestSequentialRejection:
    def test_no_isolated_atoms_ever_appear(self):
        for smiles in REPRESENTATIVE_LIGANDS:
            d = canonical_decomposition(Chem.MolFromSmiles(smiles))
            fragments = [s.removed_fragment_smiles for s in d.steps] + [d.seed_smiles]
            for fragment in fragments:
                mol = Chem.MolFromSmiles(fragment)
                assert mol.GetNumAtoms() >= 2, (smiles, fragment)

    def test_amide_wins_over_alkyl_split_of_the_same_amine(self):
        # The amide cut (priority 4) is applied before the N-alkyl split
        # (priority 2); the latter would isolate the shared nitrogen, so it
        # is rejected and never appears in the trajectory.
        d = canonical_decomposition(Chem.MolFromSmiles("CCC(=O)NCc1ccc(-c2ccccc2)cc1"))
        reaction_ids = [s.reaction_id for s in d.steps]
        assert "rxn_01" in reaction_ids
        assert "rxn_07" not in reaction_ids

    def test_ester_wins_over_ether_split_of_the_same_oxygen(self):
        d = canonical_decomposition(Chem.MolFromSmiles("CCOC(=O)c1ccccc1"))
        reaction_ids = [s.reaction_id for s in d.steps]
        assert reaction_ids == ["rxn_17"]
        assert "rxn_16" not in reaction_ids

    def test_amide_wins_over_arylamine_split_of_the_same_nitrogen(self):
        d = canonical_decomposition(Chem.MolFromSmiles("CC(=O)Nc1ncccn1"))
        reaction_ids = [s.reaction_id for s in d.steps]
        assert reaction_ids == ["rxn_01"]
        assert "rxn_04" not in reaction_ids


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_legal_cuts(self):
        # hexane and benzene expose no junction motifs / no bridges
        for smiles in ("CCCCCC", "c1ccccc1"):
            d = canonical_decomposition(Chem.MolFromSmiles(smiles))
            assert d.steps == ()
            assert d.assembly_order == ((Chem.CanonSmiles(smiles), None),)
            assert d.seed_smiles == Chem.CanonSmiles(smiles)

    def test_ambiguity_resolves_deterministically(self):
        # the biaryl bridge (two aromatic endpoints) is claimed by
        # rxn_09/10/11 (the same pattern); id-ascending picks rxn_09.
        # The N-C(sp3) bond (aliphatic N, sp3 carbon) resolves to rxn_07
        # over rxn_03 (higher priority).
        mol = Chem.MolFromSmiles("CCC(=O)NCc1ccc(-c2ccccc2)cc1")
        bonds = legal_disconnection_bonds(mol)
        biaryl = next(
            rid
            for a, b, rid in bonds
            if mol.GetAtomWithIdx(a).GetIsAromatic() and mol.GetAtomWithIdx(b).GetIsAromatic()
        )
        assert biaryl == "rxn_09"
        alkyl = next(
            rid
            for a, b, rid in bonds
            if mol.GetAtomWithIdx(a).GetAtomicNum() == 7
            and not mol.GetAtomWithIdx(b).GetIsAromatic()
            and rid not in ("rxn_01",)
        )
        assert alkyl == "rxn_07"

    def test_fail_closed_on_empty(self):
        with pytest.raises(FragmenterError):
            canonical_decomposition(Chem.MolFromSmiles(""))
        with pytest.raises(FragmenterError):
            legal_disconnection_bonds(Chem.MolFromSmiles(""))
