"""Reaction-constrained retrosynthetic supervision for CrossDocked ligands.

The production dataset must never invent a synthon or torsion target. This
module searches the observed ligand graph for a bond that can be explained by
one of the supported forward reactions, reconstructs the corresponding
reaction handles, and requires an exact forward-reaction replay against a
catalog synthon before accepting the example.

Only transformations whose precursors can be recovered unambiguously from the
product graph are used for supervision. Reactions for which the product alone
does not identify the experimental class are represented by a shared reaction
family (for example SNAr/Buchwald aryl amination).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem

from syntree.chemistry.conformer import ConformerEngine
from syntree.chemistry.reactions import (
    REACTION_FAMILY_FOR,
    REACTION_FAMILY_MEMBERS,
    REACTION_SIDES,
    HANDLE_SMARTS,
    ReactionEngine,
)


# Product-only retrosynthesis is currently exact for these transformations.
# Urea formation and click chemistry are intentionally excluded because the
# simple product graph does not preserve enough information to reconstruct
# the actual two catalog precursors without extra reaction provenance.
SUPPORTED_RETRO_FAMILIES = (
    "amide_coupling",
    "reductive_amination",
    "suzuki_coupling",
    "aryl_amination",
    "esterification",
)


@dataclass(frozen=True)
class RetrosyntheticTarget:
    """Validated single-step training target extracted from a ligand pose."""

    synthon_index: int
    reaction_family: str
    reaction_name: str
    core_handle_type: str
    target_dihedral: float
    core_mol: Chem.Mol


def canonical_smiles(mol: Chem.Mol) -> str:
    """Canonical connectivity SMILES used for exact replay checks."""
    return Chem.MolToSmiles(
        Chem.RemoveHs(Chem.Mol(mol)),
        canonical=True,
        isomericSmiles=False,
    )


class ReactionConstrainedFragmenter:
    """Recover catalog synthons from an observed product molecule."""

    def __init__(self, catalog, reaction_engine: Optional[ReactionEngine] = None):
        self.catalog = catalog
        self.engine = reaction_engine or ReactionEngine()
        self._catalog_lookup = catalog.canonical_smiles_index
        self._family_members: Dict[str, Tuple[str, ...]] = {
            family: tuple(
                member
                for member in REACTION_FAMILY_MEMBERS.get(family, (family,))
                if member in REACTION_SIDES
            )
            for family in SUPPORTED_RETRO_FAMILIES
        }

    def find_target(self, ligand: Chem.Mol) -> Optional[RetrosyntheticTarget]:
        """Find the best validated catalog synthon decomposition.

        The search is deterministic. Candidates are ordered by:
        1. supported reaction-family order,
        2. molecular bond index,
        3. synthon catalog index.

        A candidate is accepted only when a catalog synthon and reconstructed
        core can be replayed through the reaction engine to the same molecular
        connectivity as the observed ligand.
        """
        if ligand is None or ligand.GetNumAtoms() == 0:
            return None
        if ligand.GetNumConformers() == 0:
            return None

        original = Chem.Mol(ligand)
        for family in SUPPORTED_RETRO_FAMILIES:
            for bond_idx in range(original.GetNumBonds()):
                bond = original.GetBondWithIdx(bond_idx)
                if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                    continue

                candidate = self._candidate_for_bond(
                    original, bond_idx, family
                )
                if candidate is not None:
                    return candidate
        return None

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------
    def _candidate_for_bond(
        self, ligand: Chem.Mol, bond_idx: int, family: str
    ) -> Optional[RetrosyntheticTarget]:
        bond = ligand.GetBondWithIdx(bond_idx)
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        fragments = self._split_bond(ligand, bond_idx)
        if len(fragments) != 2:
            return None

        family_pairs = self._family_pairs(family)
        for side_a, side_b, backend in family_pairs:
            # Try both orientations of the reaction roles.
            for a_is_fragment_zero in (True, False):
                frag_a, ep_a = fragments[0]
                frag_b, ep_b = fragments[1]
                if not a_is_fragment_zero:
                    frag_a, frag_b = frag_b, frag_a
                    ep_a, ep_b = ep_b, ep_a

                prepared_a = self._decorate_fragment(
                    frag_a, ep_a, side_a
                )
                prepared_b = self._decorate_fragment(
                    frag_b, ep_b, side_b
                )
                if not prepared_a or not prepared_b:
                    continue

                for prepared_a_mol in prepared_a:
                    for prepared_b_mol in prepared_b:
                        for synthon_role, core_role, synthon_mol, core_mol in (
                            (side_a, side_b, prepared_a_mol, prepared_b_mol),
                            (side_b, side_a, prepared_b_mol, prepared_a_mol),
                        ):
                            lookup = self._catalog_lookup.get(
                                canonical_smiles(synthon_mol), ()
                            )
                            if not lookup:
                                continue

                            handles = [
                                h for h in self.engine.detect_handles(core_mol)
                                if h.handle_type == core_role
                            ]
                            if not handles:
                                continue

                            for synthon_index in sorted(lookup):
                                synthon = self.catalog.get_mol(
                                    synthon_index, explicit_hs=False
                                )
                                replay = self.engine.apply_reaction(
                                    core_mol, synthon, backend
                                )
                                if replay is None:
                                    continue
                                if canonical_smiles(replay.product) != canonical_smiles(ligand):
                                    continue

                                target_dihedral = ConformerEngine.get_dihedral(
                                    ligand, (left, right)
                                )
                                if target_dihedral is None:
                                    continue

                                return RetrosyntheticTarget(
                                    synthon_index=synthon_index,
                                    reaction_family=REACTION_FAMILY_FOR[backend],
                                    reaction_name=backend,
                                    core_handle_type=core_role,
                                    target_dihedral=float(target_dihedral),
                                    core_mol=Chem.Mol(core_mol),
                                )
        return None

    @staticmethod
    def _split_bond(
        ligand: Chem.Mol, bond_idx: int
    ) -> Tuple[Tuple[Chem.Mol, int], Tuple[Chem.Mol, int]]:
        source = Chem.Mol(ligand)
        for atom in source.GetAtoms():
            atom.SetIntProp("_syntree_orig_idx", atom.GetIdx())

        bond = source.GetBondWithIdx(bond_idx)
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        rw = Chem.RWMol(source)
        rw.RemoveBond(left, right)
        cut = rw.GetMol()
        cut.UpdatePropertyCache(False)

        fragments = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)
        result = []
        for fragment in fragments:
            orig_to_idx = {
                atom.GetIntProp("_syntree_orig_idx"): atom.GetIdx()
                for atom in fragment.GetAtoms()
                if atom.HasProp("_syntree_orig_idx")
            }
            if left in orig_to_idx:
                endpoint = orig_to_idx[left]
            elif right in orig_to_idx:
                endpoint = orig_to_idx[right]
            else:
                continue
            for atom in fragment.GetAtoms():
                if atom.HasProp("_syntree_orig_idx"):
                    atom.ClearProp("_syntree_orig_idx")
            result.append((fragment, endpoint))

        return tuple(result)  # type: ignore[return-value]

    @staticmethod
    def _decorate_fragment(
        fragment: Chem.Mol, endpoint: int, handle_type: str
    ) -> Tuple[Chem.Mol, ...]:
        """Restore one or more possible reactant handles after a bond cut.

        The product graph does not preserve which aryl halide leaving group was
        used experimentally, so aryl_halide returns Br/Cl/I variants and the
        caller matches them against the actual catalog. Other handles have a
        unique graph reconstruction.
        """
        endpoint_atom = fragment.GetAtomWithIdx(endpoint)
        if handle_type == "aryl_halide":
            if not endpoint_atom.GetIsAromatic():
                return ()
            variants: List[Chem.Mol] = []
            for atomic_num in (35, 17, 53):
                rw = Chem.RWMol(fragment)
                xi = rw.AddAtom(Chem.Atom(atomic_num))
                rw.AddBond(endpoint, xi, Chem.BondType.SINGLE)
                mol = rw.GetMol()
                mol.UpdatePropertyCache(False)
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    continue
                variants.append(mol)
            return tuple(variants)

        rw = Chem.RWMol(fragment)
        endpoint_atom = rw.GetAtomWithIdx(endpoint)

        if handle_type == "carboxylic_acid":
            if endpoint_atom.GetAtomicNum() != 6:
                return ()
            oi = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(endpoint, oi, Chem.BondType.SINGLE)
        elif handle_type == "aldehyde":
            if endpoint_atom.GetAtomicNum() != 6:
                return ()
            oi = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(endpoint, oi, Chem.BondType.DOUBLE)
        elif handle_type == "boronic_acid":
            if not endpoint_atom.GetIsAromatic():
                return ()
            bi = rw.AddAtom(Chem.Atom(5))
            rw.AddBond(endpoint, bi, Chem.BondType.SINGLE)
            for _ in range(2):
                oi = rw.AddAtom(Chem.Atom(8))
                rw.AddBond(bi, oi, Chem.BondType.SINGLE)
        elif handle_type in ("primary_secondary_amine", "alcohol"):
            pass
        else:
            return ()

        mol = rw.GetMol()
        mol.UpdatePropertyCache(False)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return ()
        return (mol,)

    def _family_pairs(self, family: str):
        """Return (side_a, side_b, concrete backend) candidates."""
        members = self._family_members.get(family, ())
        result = []
        for backend in members:
            side_a, side_b = REACTION_SIDES[backend]
            result.append((side_a, side_b, backend))
        return result


__all__ = [
    "SUPPORTED_RETRO_FAMILIES",
    "RetrosyntheticTarget",
    "ReactionConstrainedFragmenter",
    "canonical_smiles",
]
