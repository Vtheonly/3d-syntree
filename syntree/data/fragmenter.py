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

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

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
#
# APPEND-ONLY family ordering: amide/sulfonamide/reductive_amination/
# sp3_alkylation precede the cross-coupling families so ordinary C-N and
# S-N disconnections are tried before metal-catalysed ones. Existing
# accepted targets keep their family assignment because the new families
# only *add* candidate disconnects (sulfonamide S-N bonds had no matching
# family before; C-N bonds still try reductive_amination first).
SUPPORTED_RETRO_FAMILIES = (
    "amide_coupling",
    "sulfonamide_coupling",
    "reductive_amination",
    "sp3_alkylation",
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
    junction_bond: Tuple[int, int]
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
                            lookup = self._catalog_candidates(synthon_mol)
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
                                    junction_bond=(int(left), int(right)),
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
    def _fold_explicit_hydrogens(
        fragment: Chem.Mol, endpoint: int
    ) -> Tuple[Chem.Mol, int]:
        """Fold explicit hydrogen neighbours into implicit H counts.

        Ligands loaded from SDF/MOL2 sometimes carry explicit hydrogens. The
        handle decorations below add atoms/bonds assuming implicit-H valence
        bookkeeping: decorating an explicit-H CH2 endpoint as an aldehyde
        (=O) would create a pentavalent carbon and silently fail. Folding Hs
        first is index-preserving for heavy atoms (``Chem.RemoveHs`` keeps
        heavy-atom order), so the returned endpoint stays valid.
        """
        if not any(atom.GetAtomicNum() == 1 for atom in fragment.GetAtoms()):
            return fragment, endpoint
        heavy_before = sum(
            1
            for atom in fragment.GetAtoms()
            if atom.GetAtomicNum() > 1 and atom.GetIdx() < endpoint
        )
        folded = Chem.RemoveHs(Chem.Mol(fragment))
        return folded, heavy_before

    @staticmethod
    def _decorate_fragment(
        fragment: Chem.Mol, endpoint: int, handle_type: str
    ) -> Tuple[Chem.Mol, ...]:
        """Restore one or more possible reactant handles after a bond cut.

        The product graph does not preserve which aryl halide leaving group was
        used experimentally, so aryl_halide returns Br/Cl/I variants and the
        caller matches them against the actual catalog. The same applies to the
        sp3 alkyl halide (tasklist Priority 4). Other handles have a unique
        graph reconstruction.
        """
        # Explicit-H ligands break implicit-valence decorations (e.g. the
        # aldehyde =O would over-valence a CH2 endpoint): fold Hs first.
        fragment, endpoint = ReactionConstrainedFragmenter._fold_explicit_hydrogens(
            fragment, endpoint
        )
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

        if handle_type == "alkyl_halide":
            # The cut C-N bond's carbon endpoint must be an aliphatic sp3
            # carbon (the alkyl_halide SMARTS [CX4][Br,I,Cl] is re-verified
            # implicitly because only decorated molecules that re-detect the
            # handle can appear in the catalog).
            if endpoint_atom.GetIsAromatic() or endpoint_atom.GetAtomicNum() != 6:
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
                # Guard: the decorated carbon must actually satisfy the
                # alkyl-halide handle (sp3, CX4). Vinyl/benzyl edge cases
                # are handled by the SMARTS itself.
                from syntree.chemistry.reactions import HANDLE_SMARTS

                pattern = Chem.MolFromSmarts(HANDLE_SMARTS["alkyl_halide"])
                if not mol.HasSubstructMatch(pattern):
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
        elif handle_type == "sulfonyl_chloride":
            # Cut S-N bond: re-attach Cl onto the sulfur endpoint to rebuild
            # the sulfonyl chloride precursor. Only a genuine sulfonamide S
            # (already bearing two double-bonded oxygens) can re-detect as
            # the sulfonyl_chloride handle, which the exact-replay check
            # enforces anyway.
            if endpoint_atom.GetAtomicNum() != 16:
                return ()
            ci = rw.AddAtom(Chem.Atom(17))
            rw.AddBond(endpoint, ci, Chem.BondType.SINGLE)
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

    def _catalog_candidates(
        self,
        synthon_mol: Chem.Mol,
        min_tanimoto: float = 0.85,
    ) -> Tuple[int, ...]:
        """Return exact catalog matches plus high-similarity candidates.

        Similarity is a candidate-discovery mechanism only. The caller still
        requires exact forward reaction replay against the observed product,
        so a Tanimoto-near miss can never become fabricated supervision.
        """
        exact = self._catalog_lookup.get(canonical_smiles(synthon_mol), ())
        if exact:
            return tuple(sorted(exact))

        fp = AllChem.GetMorganFingerprintAsBitVect(synthon_mol, radius=2, nBits=2048)
        scored = []
        for idx, smiles in enumerate(self.catalog.df["smiles"]):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            candidate_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(fp, candidate_fp)
            if similarity >= min_tanimoto:
                scored.append((float(similarity), int(idx)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(idx for _, idx in scored)

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
