"""Canonical assembly ordering and seed-conditioned decompositions (#15).

Issue #15 (legacy "Vulnerability 6"): the legacy fragmenter iterated bond
indices in arbitrary file order, so identical scaffolds with different
input atom numberings generated conflicting assembly supervision.  This
module produces a **unique, deterministic retrosynthetic trajectory for
any ligand graph, invariant to input SDF atom ordering**.

Algorithm (issue-pinned, every rule deterministic and documented)
------------------------------------------------------------------
1. **Legal disconnection bonds.**  For every policy reaction of the
   certified taxonomy (#11, ``syntree/chemistry/reactions.py``), the
   product-side ``junction_pattern`` SMARTS (map ``:1`` = molecule-side
   attachment atom, ``:2`` = block-side attachment atom) is matched
   against the ligand.  A bond is a legal cut when a pattern claims it,
   the bond is a *bridge* (never a ring bond), and removing it would
   leave >= 2 heavy atoms on both sides (methyls are never synthons).
   One bond can satisfy several reactions; the claim resolves by
   ``(junction_priority DESC, reaction_id ASC)`` — deterministic
   (D-V2-016).

2. **Cut application order.**  The legal cuts are applied one at a time
   in canonical order ``(junction_priority DESC, reaction_id ASC, bond
   endpoint canonical labels ASC)``.  After each cut the fragment set is
   recomputed; **a cut is rejected (permanently skipped) if it would
   create any fragment with fewer than 2 heavy atoms.**  This yields a
   synthon-sized fragment set without isolated single atoms (the amide
   disconnection is applied before an N-alkyl split of the same amine,
   so the nitrogen never becomes an isolated fragment).

3. **Root seed fragment** (issue-pinned rule): among the fragments pick
   ``argmax`` of
   ``(largest connected ring-system atom count DESC,
     attachment degree DESC,
     canonical SMILES ASC)`` —
   maximum ring-system size first, then the highest attachment degree,
   then canonical-SMILES tie-breaking (lexicographically smallest wins).
   Residual ties (fully symmetric fragments) produce identical outputs,
   so the trajectory stays deterministic; the final fallback is the
   smallest atom-index set.

4. **Disconnection order — periphery to core.**  The remaining fragments
   are peeled one at a time: among leaf fragments (degree 1 in the
   fragment graph, excluding the seed), the one **farthest from the seed**
   (graph distance) goes first — the most peripheral synthon — with
   canonical-SMILES tie-breaking.  Each peel records the removed
   synthon, the certified reaction that formed its junction, and the
   remaining molecule.  The **assembly order is the exact reverse**
   (core-to-periphery), which is what supervised trajectories consume.

5. **Atom-order invariance.**  Every recorded label is a canonical SMILES
   or a graph invariant (ring-system sizes, degrees, distances); the
   cut-application and tie-break keys never reference raw atom indices
   in an order-dependent way.  Permuting the input atom order therefore
   yields the identical trajectory sequence — the #15 acceptance
   criterion.

Supervision versus inference (issue addendum): the deterministic root
here is a mechanism for **reproducible supervised reference
trajectories**; it must not force inference to use the same heuristic
seed — inference seed/anchor selection is the explicit policy action of
#39 (MODEL-06).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rdkit import Chem

from syntree.chemistry.reactions import POLICY_REACTIONS, CertifiedReaction

__all__ = [
    "CanonicalDecomposition",
    "DisconnectionStep",
    "FragmentRecord",
    "canonical_decomposition",
    "legal_disconnection_bonds",
]

#: Minimum heavy-atom count for every fragment produced by a cut.
MIN_FRAGMENT_ATOMS = 2


class FragmenterError(ValueError):
    """Fail-closed error for malformed decomposition inputs."""


@dataclass(frozen=True)
class FragmentRecord:
    """One fragment of the final cut state."""

    smiles: str                 # canonical SMILES
    ring_system_atoms: int      # largest connected ring-system atom count
    attachment_degree: int      # number of incident cut bonds
    atom_indices: Tuple[int, ...]  # instance-level (ties need a fallback)


@dataclass(frozen=True)
class DisconnectionStep:
    """One periphery-to-core disconnection (reverse of assembly)."""

    order: int                      # 1-based, periphery first
    removed_fragment_smiles: str    # canonical SMILES of the peeled synthon
    reaction_id: str                # the certified reaction of its junction
    family: Optional[str]
    molecule_handle_class: Optional[str]  # the remaining side's handle class
    block_handle_class: str         # the removed synthon side's class
    remaining_smiles: str           # the remaining molecule after the peel


@dataclass(frozen=True)
class CanonicalDecomposition:
    """The full deterministic decomposition of one ligand graph."""

    ligand_smiles: str              # canonical SMILES of the input
    seed_smiles: str                # canonical SMILES of the root seed
    seed_ring_system_atoms: int
    steps: Tuple[DisconnectionStep, ...]          # periphery -> core
    assembly_order: Tuple[Tuple[str, Optional[str]], ...]  # core -> periphery:
    # (fragment_smiles, reaction_id that attached it)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)


def _ring_system_atoms(mol: Chem.Mol, atom_indices: Sequence[int]) -> int:
    """Largest connected ring-system atom count within a fragment.

    Ring systems are the connected components of the ring-bond subgraph
    (fused rings count as one system; bridge-linked rings are separate
    systems — the biaryl bridge is not a ring bond).
    """
    index_set = set(atom_indices)
    ring_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIdx() in index_set and atom.IsInRing()
    }
    if not ring_atoms:
        return 0
    adjacency: Dict[int, Set[int]] = {idx: set() for idx in ring_atoms}
    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if b in ring_atoms and e in ring_atoms and bond.IsInRing():
            adjacency[b].add(e)
            adjacency[e].add(b)
    best = 0
    unseen = set(ring_atoms)
    while unseen:
        stack = [next(iter(unseen))]
        component = {stack[0]}
        unseen.discard(stack[0])
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if nxt in unseen:
                    unseen.discard(nxt)
                    component.add(nxt)
                    stack.append(nxt)
        best = max(best, len(component))
    return best


def _components(mol: Chem.Mol) -> List[Set[int]]:
    """Connected components of the molecule's atom graph."""
    seen: Set[int] = set()
    out: List[Set[int]] = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() in seen:
            continue
        stack = [atom.GetIdx()]
        component = {atom.GetIdx()}
        seen.add(atom.GetIdx())
        while stack:
            node = stack.pop()
            for neighbor in mol.GetAtomWithIdx(node).GetNeighbors():
                if neighbor.GetIdx() not in seen:
                    seen.add(neighbor.GetIdx())
                    component.add(neighbor.GetIdx())
                    stack.append(neighbor.GetIdx())
        out.append(component)
    return out


def _submol(mol: Chem.Mol, atom_indices: Sequence[int]) -> Chem.Mol:
    """Extract the induced subgraph as an independent molecule."""
    keep = set(atom_indices)
    rw = Chem.RWMol()
    mapping: Dict[int, int] = {}
    for old in sorted(keep):
        mapping[old] = rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old)))
    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if b in mapping and e in mapping:
            rw.AddBond(mapping[b], mapping[e], bond.GetBondType())
    return rw.GetMol()


def _fragment_smiles(mol: Chem.Mol, atom_indices: Sequence[int]) -> str:
    return _canonical(_submol(mol, atom_indices))


def _endpoint_labels(mol: Chem.Mol, reaction: CertifiedReaction, a1: int, a2: int) -> Tuple[str, str]:
    """Canonical, atom-order-invariant sort key for a claimed bond."""
    return (
        _fragment_smiles(mol, [a1]),
        _fragment_smiles(mol, [a2]),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def legal_disconnection_bonds(mol: Chem.Mol) -> List[Tuple[int, int, str]]:
    """All legal (a1, a2, reaction_id) disconnections of ``mol``.

    ``a1`` is the molecule-side attachment atom (map ``:1``), ``a2`` the
    block-side attachment atom (map ``:2``) of the claiming reaction's
    junction pattern.  Ring bonds and both-side-tiny splits are never
    legal.  A bond claimed by several reactions resolves to the single
    ``(priority DESC, reaction_id ASC)`` winner.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise FragmenterError("canonical_decomposition requires a non-empty molecule")
    claims: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
    for reaction in POLICY_REACTIONS.values():
        pattern = Chem.MolFromSmarts(reaction.junction_pattern)
        if pattern is None:
            raise FragmenterError(
                f"{reaction.reaction_id}: unparseable junction pattern"
            )
        for match in mol.GetSubstructMatches(pattern):
            a1, a2 = int(match[0]), int(match[1])
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None or bond.IsInRing():
                continue
            component_sizes = _split_sizes(mol, a1, a2)
            if min(component_sizes) < MIN_FRAGMENT_ATOMS:
                continue
            claims.setdefault((min(a1, a2), max(a1, a2)), []).append(
                (reaction.junction_priority, reaction.reaction_id)
            )
    resolved: List[Tuple[int, int, str]] = []
    for (lo, hi), candidates in claims.items():
        priority, reaction_id = sorted(candidates, key=lambda c: (-c[0], c[1]))[0]
        # orientation: a1 = molecule side (:1), a2 = block side (:2)
        reaction = POLICY_REACTIONS[reaction_id]
        pattern = Chem.MolFromSmarts(reaction.junction_pattern)
        oriented: Optional[Tuple[int, int]] = None
        for match in mol.GetSubstructMatches(pattern):
            x, y = int(match[0]), int(match[1])
            if {x, y} == {lo, hi}:
                oriented = (x, y)
                break
        if oriented is None:  # pragma: no cover - the claim came from a match
            raise FragmenterError(f"{reaction_id}: lost the match for bond ({lo}, {hi})")
        resolved.append((oriented[0], oriented[1], reaction_id))
    return sorted(resolved, key=lambda item: (item[2], item[0], item[1]))


def _split_sizes(mol: Chem.Mol, a1: int, a2: int) -> Tuple[int, int]:
    """Heavy-atom counts of the two sides of the bond ``a1-a2``."""
    adjacency: Dict[int, Set[int]] = {
        atom.GetIdx(): {n.GetIdx() for n in atom.GetNeighbors()} for atom in mol.GetAtoms()
    }
    adjacency[a1].discard(a2)
    adjacency[a2].discard(a1)

    def side(start: int, blocked: int) -> Set[int]:
        stack, seen = [start], {start}
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if nxt != blocked and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    side_a = side(a1, a2) if a1 != a2 else set()
    side_b = set(range(mol.GetNumAtoms())) - side_a
    return len(side_a), len(side_b)


def canonical_decomposition(mol: Chem.Mol) -> CanonicalDecomposition:
    """Deterministic canonical decomposition of ``mol`` (see module notes)."""
    if mol is None or mol.GetNumAtoms() == 0:
        raise FragmenterError("canonical_decomposition requires a non-empty molecule")
    ligand_smiles = _canonical(mol)

    # -- 1. legal cuts, canonical application order --------------------------
    candidates = legal_disconnection_bonds(mol)
    order_key: Dict[str, int] = {}
    for reaction_id in sorted(POLICY_REACTIONS):
        order_key[reaction_id] = POLICY_REACTIONS[reaction_id].junction_priority
    candidates = sorted(
        candidates,
        key=lambda item: (
            -order_key[item[2]],
            item[2],
            _endpoint_labels(mol, POLICY_REACTIONS[item[2]], item[0], item[1]),
        ),
    )

    # -- 2. apply cuts; reject any cut that isolates a <2-atom fragment -------
    work = Chem.RWMol(Chem.Mol(mol))
    applied: List[Tuple[int, int, str]] = []
    for a1, a2, reaction_id in candidates:
        probe = Chem.RWMol(Chem.Mol(work))
        probe.RemoveBond(a1, a2)
        probe_mol = probe.GetMol()
        sizes = [len(c) for c in _components(probe_mol)]
        if min(sizes) < MIN_FRAGMENT_ATOMS:
            continue  # permanently rejected: would isolate a sub-synthon atom
        work = probe
        applied.append((a1, a2, reaction_id))
    cut_mol = work.GetMol()

    # -- 3. fragments and the root seed --------------------------------------
    components = _components(cut_mol)
    if not components:
        raise FragmenterError("decomposition produced no fragments")
    incident: Dict[frozenset, int] = {}
    for a1, a2, _reaction_id in applied:
        for component in components:
            if a1 in component:
                incident[frozenset(component)] = incident.get(frozenset(component), 0) + 1
                break
        for component in components:
            if a2 in component:
                incident[frozenset(component)] = incident.get(frozenset(component), 0) + 1
                break
    records: List[FragmentRecord] = []
    for component in components:
        key = frozenset(component)
        records.append(
            FragmentRecord(
                smiles=_fragment_smiles(cut_mol, sorted(component)),
                ring_system_atoms=_ring_system_atoms(cut_mol, sorted(component)),
                attachment_degree=incident.get(key, 0),
                atom_indices=tuple(sorted(component)),
            )
        )
    seed_record = sorted(
        records,
        key=lambda r: (
            -r.ring_system_atoms,
            -r.attachment_degree,
            r.smiles,
            r.atom_indices,
        ),
    )[0]
    seed_atoms = set(seed_record.atom_indices)

    # -- 4. periphery-to-core peeling ----------------------------------------
    # Fragment graph over the final components; parallel cut bonds collapse
    # to a single edge (degree counting uses distinct neighbours).
    neighbours: Dict[frozenset, Set[frozenset]] = {frozenset(c): set() for c in components}
    for a1, a2, _reaction_id in applied:
        side_a = next(c for c in components if a1 in c)
        side_b = next(c for c in components if a2 in c)
        if side_a != side_b:
            neighbours[frozenset(side_a)].add(frozenset(side_b))
            neighbours[frozenset(side_b)].add(frozenset(side_a))

    distances: Dict[frozenset, int] = {}
    queue = [frozenset(seed_atoms)]
    distances[frozenset(seed_atoms)] = 0
    while queue:
        current = queue.pop(0)
        for nxt in neighbours[current]:
            if nxt not in distances:
                distances[nxt] = distances[current] + 1
                queue.append(nxt)

    remaining = {frozenset(c) for c in components}
    current_atoms = set(range(cut_mol.GetNumAtoms()))
    steps: List[DisconnectionStep] = []
    order_index = 0
    while len(remaining) > 1:
        # leaf = degree-1 fragment, never the seed
        leaves = [
            key
            for key in remaining
            if key != frozenset(seed_atoms) and len(neighbours[key] & remaining) == 1
        ]
        if not leaves:
            # cyclic residue (parallel bonds) — deterministic fallback:
            # peel the farthest non-seed fragment
            leaves = [key for key in remaining if key != frozenset(seed_atoms)]
        chosen = sorted(
            leaves,
            key=lambda key: (
                -distances.get(key, 0),
                _fragment_smiles(cut_mol, sorted(key)),
                sorted(key),
            ),
        )[0]
        parent_set = neighbours[chosen] & (remaining - {chosen})
        # deterministic parent: canonical SMILES, then atom-index set
        parent_key = (
            sorted(
                parent_set,
                key=lambda k: (_fragment_smiles(cut_mol, sorted(k)), sorted(k)),
            )[0]
            if parent_set
            else None
        )
        # the cut bond between the chosen fragment and its deterministic
        # current parent (parallel bonds to other fragments stay for later)
        candidates_bond = [
            (a1, a2, rid)
            for a1, a2, rid in applied
            if (a1 in chosen and a2 in parent_key) or (a2 in chosen and a1 in parent_key)
        ] if parent_key is not None else []
        if not candidates_bond:
            # fall back to any bond out of the chosen fragment
            candidates_bond = [
                (a1, a2, rid)
                for a1, a2, rid in applied
                if (a1 in chosen and a2 in remaining and a2 not in chosen)
                or (a2 in chosen and a1 in remaining and a1 not in chosen)
            ]
        candidates_bond.sort(key=lambda b: (b[2], b[0], b[1]))
        bond = candidates_bond[0]
        reaction = POLICY_REACTIONS[bond[2]]
        order_index += 1
        remaining_smiles = _fragment_smiles(
            cut_mol, sorted(current_atoms - set(chosen))
        )
        steps.append(
            DisconnectionStep(
                order=order_index,
                removed_fragment_smiles=_fragment_smiles(cut_mol, sorted(chosen)),
                reaction_id=reaction.reaction_id,
                family=reaction.family,
                molecule_handle_class=reaction.molecule_handle_class,
                block_handle_class=reaction.block_handle_class,
                remaining_smiles=remaining_smiles,
            )
        )
        remaining.discard(chosen)
        current_atoms -= set(chosen)

    assembly_order: List[Tuple[str, Optional[str]]] = []
    for step in reversed(steps):
        assembly_order.append((step.removed_fragment_smiles, step.reaction_id))
    assembly_order.insert(0, (seed_record.smiles, None))

    return CanonicalDecomposition(
        ligand_smiles=ligand_smiles,
        seed_smiles=seed_record.smiles,
        seed_ring_system_atoms=seed_record.ring_system_atoms,
        steps=tuple(steps),
        assembly_order=tuple(assembly_order),
    )
