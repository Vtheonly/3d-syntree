"""3D conformer construction, dihedral rotation, and steric clash
evaluation.

The :class:`ConformerEngine` is responsible for the spatial half of every
generation step:

1. **Attach** – build 3D coordinates for a newly reacted synthon while
   locking the existing scaffold atoms in place.
2. **Rotate** – sweep or set the dihedral angle around the newly formed
   single bond.
3. **Score** – evaluate a soft Lennard-Jones clash penalty between ligand
   atoms and protein pocket atoms.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D

logger = logging.getLogger(__name__)

# Bondi van der Waals radii (Angstrom) for the elements relevant to
# protein-ligand modelling.
VDW_RADII: Dict[int, float] = {
    1: 1.10,   # H
    5: 1.92,   # B
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    11: 2.27,  # Na
    12: 1.73,  # Mg
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    19: 2.75,  # K
    20: 2.31,  # Ca
    30: 2.39,  # Zn
    34: 1.90,  # Se
    35: 1.85,  # Br
    53: 1.98,  # I
}
DEFAULT_VDW_RADIUS = 1.70  # carbon, used for unknown elements


def vdw_radius(atomic_num: int) -> float:
    """Bondi van der Waals radius for an element (carbon fallback)."""
    return VDW_RADII.get(int(atomic_num), DEFAULT_VDW_RADIUS)


def _dihedral_definition(
    mol: Chem.Mol, junction_bond: Tuple[int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """Resolve the 4-atom dihedral ``(d1, a1, a2, d4)`` around the junction.

    Preference order for the outer atoms: heavy atoms first, then the
    lowest index, ensuring a deterministic definition. Returns ``None`` for
    ring bonds (rotating them is chemically meaningless) or terminal bonds.
    """
    a1, a2 = junction_bond
    if a1 == a2 or a1 >= mol.GetNumAtoms() or a2 >= mol.GetNumAtoms():
        return None
    bond = mol.GetBondBetweenAtoms(a1, a2)
    if bond is None:
        return None
    if bond.IsInRing():
        return None  # rotatable single bonds only

    def neighbors_of(atom_idx: int, exclude: int) -> List[int]:
        nbrs = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if n.GetIdx() != exclude
        ]
        heavy = [i for i in nbrs if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        return sorted(heavy) + sorted(set(nbrs) - set(heavy))

    n1 = neighbors_of(a1, a2)
    n2 = neighbors_of(a2, a1)
    if not n1 or not n2:
        return None
    return (n1[0], a1, a2, n2[0])


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid alignment mapping points ``P`` onto ``Q`` (both ``[N, 3]``).

    Returns ``(R, t)`` such that ``P @ R.T + t ≈ Q`` with ``R`` a proper
    rotation (det = +1). Used to snap freshly embedded products onto the
    locked scaffold reference frame without tearing bonds.
    """
    assert P.shape == Q.shape
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    P0 = P - pc
    Q0 = Q - qc
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - pc @ R.T
    return R, t


class ConformerEngine:
    """Manages 3D geometry, dihedral bond rotations, and soft clash scores."""

    # ------------------------------------------------------------------
    # 1. Attachment: coordinates for a newly added synthon
    # ------------------------------------------------------------------
    @staticmethod
    def embed_product(
        product_mol: Chem.Mol,
        core_atom_map: Optional[Dict[int, int]] = None,
        core_reference: Optional[Chem.Mol] = None,
        random_seed: int = 42,
        max_attempts: int = 200,
        align_to_reference: bool = True,
    ) -> Optional[Chem.Mol]:
        """Embed 3D coordinates for ``product_mol``.

        When ``core_reference`` (the pre-reaction scaffold with a conformer)
        and ``core_atom_map`` (core atom idx -> product atom idx, as returned
        by :class:`~syntree.chemistry.reactions.ReactionEngine`) are given,
        the embedded product is first rigidly aligned (Kabsch) onto the
        scaffold's frame and the scaffold atoms are then locked exactly to
        their original positions, so the rest of the molecule grows around a
        rigid anchor without tearing any bond.

        Returns a molecule with exactly one conformer, or ``None`` when
        embedding fails.
        """
        if product_mol is None:
            raise ValueError("embed_product requires a valid Mol")

        mol = Chem.AddHs(Chem.Mol(product_mol))
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)
        params.useRandomCoords = True
        params.maxIterations = int(max_attempts)
        try:
            status = AllChem.EmbedMolecule(mol, params)
        except Exception:  # pragma: no cover - RDKit can raise on odd graphs
            status = -1
        if status != 0:
            # Retry once without ETKDG constraints.
            try:
                status = AllChem.EmbedMolecule(mol, randomSeed=int(random_seed) + 1)
            except Exception:
                status = -1
        if status != 0:
            logger.warning("Conformer embedding failed; returning None.")
            return None

        if core_reference is not None and core_atom_map:
            if core_reference.GetNumConformers() == 0:
                logger.warning("Core reference has no conformer; skipping lock.")
            else:
                pairs = [
                    (core_idx, prod_idx)
                    for core_idx, prod_idx in core_atom_map.items()
                    if core_idx < core_reference.GetNumAtoms()
                    and prod_idx < mol.GetNumAtoms()
                ]
                if pairs:
                    conf = mol.GetConformer()
                    ref_conf = core_reference.GetConformer()
                    prod_pos = np.array(conf.GetPositions(), dtype=np.float64)
                    ref_pos = np.array(ref_conf.GetPositions(), dtype=np.float64)

                    P = np.array([prod_pos[p] for _, p in pairs])
                    Q = np.array([ref_pos[c] for c, _ in pairs])

                    if align_to_reference and len(pairs) >= 3:
                        R, t = _kabsch(P, Q)
                        prod_pos = prod_pos @ R.T + t

                    # Hard-lock the scaffold atoms (also removes any
                    # residual alignment error).
                    for (core_idx, prod_idx) in pairs:
                        prod_pos[prod_idx] = ref_pos[core_idx]

                    for i in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(i, Point3D(*prod_pos[i]))

                    # Heal bond geometry: the ETKDG product may place the
                    # synthon slightly off after locking (its internal core
                    # conformation can differ from the reference). A short
                    # force-field relaxation with the scaffold frozen
                    # restores physical bond lengths.
                    ConformerEngine._relax_with_frozen_scaffold(mol, pairs)
        return mol

    @staticmethod
    def _relax_with_frozen_scaffold(
        mol: Chem.Mol,
        frozen_pairs,
        max_iterations: int = 300,
    ) -> bool:
        """Short force-field relaxation with the scaffold atoms frozen.

        Used after hard-locking scaffold coordinates so that bonds crossing
        the core/synthon boundary return to physical lengths without moving
        the scaffold. Returns True when a relaxation was performed.
        """
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                props = AllChem.MMFFGetMoleculeProperties(mol)
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff is None:
                return False
            for _, prod_idx in frozen_pairs:
                ff.AddFixedPoint(prod_idx)
            ff.Minimize(maxIts=max_iterations)
            return True
        except Exception as exc:  # pragma: no cover - best-effort healing
            logger.debug("Force-field relaxation skipped: %s", exc)
            return False

    # Kept for backwards compatibility with earlier specs.
    @classmethod
    def embed_and_align_synthon(
        cls,
        core_mol: Chem.Mol,
        product_mol: Chem.Mol,
        junction_bond: Tuple[int, int],
        core_atom_map: Optional[Dict[int, int]] = None,
    ) -> Optional[Chem.Mol]:
        """Embed ``product_mol`` while locking the scaffold of ``core_mol``.

        Derives ``core_atom_map`` from atom counts when not supplied (the
        first ``core_mol.GetNumAtoms()`` product atoms belong to the core
        under RDKit's default reaction atom ordering).
        """
        if core_atom_map is None:
            n_core = core_mol.GetNumAtoms()
            core_atom_map = {i: i for i in range(min(n_core, product_mol.GetNumAtoms()))}
        return cls.embed_product(product_mol, core_atom_map, core_mol)

    # ------------------------------------------------------------------
    # 2. Rotation: dihedral control around the junction bond
    # ------------------------------------------------------------------
    @staticmethod
    def set_dihedral(
        mol: Chem.Mol, junction_bond: Tuple[int, int], angle_rad: float
    ) -> Chem.Mol:
        """Rotate the synthon side of ``junction_bond`` to ``angle_rad``.

        The rotation moves only atoms on the synthon side of the bond; the
        scaffold side stays fixed. Returns the same molecule object for
        chaining (rotation is performed in place on the conformer).
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("set_dihedral requires a molecule with a conformer")
        conf = mol.GetConformer()
        a1, a2 = junction_bond

        definition = _dihedral_definition(mol, junction_bond)
        if definition is None:
            logger.debug("Cannot define dihedral around bond %s.", junction_bond)
            return mol

        d1, b1, b2, d4 = definition
        angle_deg = float(np.degrees(((angle_rad + np.pi) % (2 * np.pi)) - np.pi))
        try:
            rdMolTransforms.SetDihedralDeg(conf, d1, b1, b2, d4, angle_deg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Dihedral rotation failed (%s); leaving geometry unchanged.", exc)
        return mol

    @staticmethod
    def get_dihedral(
        mol: Chem.Mol, junction_bond: Tuple[int, int]
    ) -> Optional[float]:
        """Current dihedral angle (radians, in ``[-pi, pi)``) around the
        junction bond, or ``None`` when it cannot be defined."""
        if mol.GetNumConformers() == 0:
            return None
        definition = _dihedral_definition(mol, junction_bond)
        if definition is None:
            return None
        d1, b1, b2, d4 = definition
        conf = mol.GetConformer()
        deg = rdMolTransforms.GetDihedralDeg(conf, d1, b1, b2, d4)
        rad = np.radians(deg)
        return float((rad + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def optimize_dihedral(
        mol: Chem.Mol,
        junction_bond: Tuple[int, int],
        pocket_coords: Optional[np.ndarray] = None,
        pocket_vdw: Optional[np.ndarray] = None,
        ligand_vdw: Optional[np.ndarray] = None,
        grid_size: int = 36,
    ) -> Tuple[Chem.Mol, float, float]:
        """Grid-search the dihedral that minimises the steric clash.

        Args:
            mol: Molecule with a conformer.
            junction_bond: ``(core_atom, synthon_atom)`` indices.
            pocket_coords: ``[N_p, 3]`` pocket coordinates; when omitted the
                clash score is zero and only the torsion potential of the
                embedded geometry is returned.
            grid_size: Number of evenly spaced angles to scan.

        Returns:
            ``(mol, best_angle_rad, best_clash)`` — the molecule is rotated
            in place to the best angle.
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("optimize_dihedral requires a molecule with a conformer")

        conf = mol.GetConformer()
        ligand_coords = np.array(conf.GetPositions(), dtype=np.float64)

        if ligand_vdw is None:
            ligand_vdw = np.array(
                [vdw_radius(a.GetAtomicNum()) for a in mol.GetAtoms()], dtype=np.float64
            )

        if pocket_coords is None or len(pocket_coords) == 0:
            current = ConformerEngine.get_dihedral(mol, junction_bond)
            best_angle = current if current is not None else 0.0
            return mol, float(best_angle), 0.0

        if pocket_vdw is None:
            pocket_vdw = np.full(len(pocket_coords), DEFAULT_VDW_RADIUS, dtype=np.float64)

        # Identify the movable (synthon) side of the junction bond.
        a1, a2 = junction_bond
        movable = ConformerEngine._side_atoms(mol, a1, a2)
        if not movable:
            current = ConformerEngine.get_dihedral(mol, junction_bond)
            return mol, float(current or 0.0), 0.0

        best_angle, best_clash = 0.0, float("inf")
        for angle in np.linspace(-np.pi, np.pi, grid_size, endpoint=False):
            ConformerEngine.set_dihedral(mol, junction_bond, float(angle))
            clash = ConformerEngine.compute_steric_clash_np(
                np.array(mol.GetConformer().GetPositions(), dtype=np.float64),
                pocket_coords,
                ligand_vdw,
                pocket_vdw,
                subset_indices=movable,
            )
            if clash < best_clash:
                best_clash, best_angle = clash, float(angle)

        ConformerEngine.set_dihedral(mol, junction_bond, best_angle)
        return mol, best_angle, float(best_clash)

    @staticmethod
    def _side_atoms(mol: Chem.Mol, anchor: int, other: int) -> List[int]:
        """Atoms on ``other``'s side of the ``anchor–other`` bond."""
        visited = {anchor}
        stack = [other]
        side: List[int] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            side.append(node)
            for nbr in mol.GetAtomWithIdx(node).GetNeighbors():
                if nbr.GetIdx() not in visited:
                    stack.append(nbr.GetIdx())
        return side

    # ------------------------------------------------------------------
    # 3. Scoring: soft clash penalties
    # ------------------------------------------------------------------
    @staticmethod
    def compute_steric_clash_loss(
        ligand_coords: torch.Tensor,
        pocket_coords: torch.Tensor,
        ligand_vdw: Optional[torch.Tensor] = None,
        pocket_vdw: Optional[torch.Tensor] = None,
        clash_tolerance: float = 1.0,
    ) -> torch.Tensor:
        """Differentiable soft clash penalty.

        For every ligand-pocket atom pair,
        ``penalty = max(0, (r_i + r_j - tolerance) - d_ij)^2`` summed over
        all pairs (a smoothed, quadratic variant of the Lennard-Jones repulsive
        wall used as an auxiliary training loss).

        Args:
            ligand_coords: ``[N_l, 3]`` tensor.
            pocket_coords: ``[N_p, 3]`` tensor.
            ligand_vdw / pocket_vdw: optional per-atom radii tensors.
            clash_tolerance: softening margin subtracted from radius sums
                (Angstrom).
        """
        if ligand_coords.dim() != 2 or ligand_coords.size(1) != 3:
            raise ValueError(f"ligand_coords must be [N, 3], got {tuple(ligand_coords.shape)}")
        if pocket_coords.dim() != 2 or pocket_coords.size(1) != 3:
            raise ValueError(f"pocket_coords must be [M, 3], got {tuple(pocket_coords.shape)}")

        if ligand_vdw is None:
            ligand_vdw = torch.full(
                (ligand_coords.size(0),), DEFAULT_VDW_RADIUS,
                device=ligand_coords.device, dtype=ligand_coords.dtype,
            )
        if pocket_vdw is None:
            pocket_vdw = torch.full(
                (pocket_coords.size(0),), DEFAULT_VDW_RADIUS,
                device=pocket_coords.device, dtype=pocket_coords.dtype,
            )

        dists = torch.cdist(ligand_coords, pocket_coords)  # [N_l, N_p]
        thresholds = (ligand_vdw.unsqueeze(1) + pocket_vdw.unsqueeze(0)) - clash_tolerance
        clashes = torch.clamp(thresholds - dists, min=0.0)
        return torch.sum(clashes ** 2)

    @staticmethod
    def compute_steric_clash_np(
        ligand_coords: np.ndarray,
        pocket_coords: np.ndarray,
        ligand_vdw: Optional[np.ndarray] = None,
        pocket_vdw: Optional[np.ndarray] = None,
        clash_tolerance: float = 1.0,
        subset_indices: Optional[Sequence[int]] = None,
    ) -> float:
        """NumPy analogue of :meth:`compute_steric_clash_loss` for fast
        inference-time scoring (optionally restricted to a subset of ligand
        atoms, e.g. only the newly attached synthon)."""
        if subset_indices is not None:
            idx = np.asarray(subset_indices, dtype=int)
            ligand_coords = ligand_coords[idx]
            if ligand_vdw is not None:
                ligand_vdw = np.asarray(ligand_vdw, dtype=float)[idx]
        if ligand_vdw is None:
            ligand_vdw = np.full(len(ligand_coords), DEFAULT_VDW_RADIUS)
        if pocket_vdw is None:
            pocket_vdw = np.full(len(pocket_coords), DEFAULT_VDW_RADIUS)

        if len(ligand_coords) == 0 or len(pocket_coords) == 0:
            return 0.0

        diff = ligand_coords[:, None, :] - pocket_coords[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        thresholds = ligand_vdw[:, None] + pocket_vdw[None, :] - clash_tolerance
        clashes = np.clip(thresholds - dists, 0.0, None)
        return float((clashes ** 2).sum())


__all__ = [
    "ConformerEngine",
    "VDW_RADII",
    "vdw_radius",
]
