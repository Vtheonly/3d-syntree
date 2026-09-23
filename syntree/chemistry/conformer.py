"""3D conformer construction, dihedral rotation, and steric clash
evaluation.

The :class:`ConformerEngine` is responsible for the spatial half of every
generation step:

1. **Attach** – build 3D coordinates for a newly reacted synthon while
   keeping the existing scaffold *harmonically* restrained to its pocket
   frame and letting the junction atoms relax to physical bond geometry.
2. **Rotate** – sweep or set the dihedral angle around the newly formed
   single bond. Conjugated junctions (amides, esters) are resonance-locked
   to planar *trans/cis* geometries; the torsion is applied to the adjacent
   true single bond instead, exactly like protein φ/ψ angles.
3. **Score** – evaluate a soft Lennard-Jones interaction energy between
   ligand atoms and protein pocket atoms (repulsive wall *plus* the
   attractive dispersion well, so drifting into open solvent is no longer
   a free lunch).
"""

from __future__ import annotations

import copy
import logging
import math
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


# ---------------------------------------------------------------------------
# Resonance-locked (conjugated) junction detection.
#
# Amide C(=O)-N bonds have bond order ~1.4 and a ~18-22 kcal/mol rotation
# barrier: they are strictly planar at room temperature. Twisting them to an
# arbitrary policy angle produces an unphysical sp3 nitrogen and destroys
# the carbonyl resonance. The same holds for esters C(=O)-O.
# ---------------------------------------------------------------------------
_AMIDE_JUNCTION_SMARTS = "[C;R0]-[N;$(N[C]=O)]"
_CARBNONYL_SMARTS = "[C]=[O]"


def is_conjugated_junction(mol: Chem.Mol, junction_bond: Tuple[int, int]) -> bool:
    """True when ``junction_bond`` is resonance-locked (amide / ester /
    double bond / ring bond) and must not be freely rotated.

    An amide/ester junction is a *single* bond whose one end carries a
    carbonyl (C=O) double bond to O and whose partner is N or O.
    """
    if mol is None:
        return False
    a1, a2 = junction_bond
    if a1 >= mol.GetNumAtoms() or a2 >= mol.GetNumAtoms():
        return False
    bond = mol.GetBondBetweenAtoms(a1, a2)
    if bond is None:
        return False
    if bond.IsInRing():
        return True
    if bond.GetBondTypeAsDouble() != 1.0:
        return True  # double / aromatic bonds are not free rotors

    def _is_carbonyl_side(c_idx: int, x_idx: int) -> bool:
        c_atom = mol.GetAtomWithIdx(c_idx)
        for nbr in c_atom.GetNeighbors():
            if nbr.GetIdx() == x_idx:
                continue
            nb = mol.GetBondBetweenAtoms(c_idx, nbr.GetIdx())
            if (
                nbr.GetAtomicNum() == 8
                and nb is not None
                and nb.GetBondTypeAsDouble() == 2.0
            ):
                return True
        return False

    for c_idx, x_idx in ((a1, a2), (a2, a1)):
        c_atom, x_atom = mol.GetAtomWithIdx(c_idx), mol.GetAtomWithIdx(x_idx)
        if c_atom.GetAtomicNum() == 6 and x_atom.GetAtomicNum() in (7, 8):
            if _is_carbonyl_side(c_idx, x_idx):
                return True
    return False


def _adjacent_rotatable_bond(
    mol: Chem.Mol, junction_bond: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """Find the first true single bond adjacent to the synthon side of the
    junction (the protein-backbone-analogue φ/ψ rotation target)."""
    a1, a2 = junction_bond
    synthon_side = ConformerEngine._side_atoms(mol, a1, a2)
    candidates: List[Tuple[int, int]] = []
    for idx in synthon_side:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            if j == a1 or j == idx:
                continue
            if idx == a2 and j == a1:
                continue
            bond = mol.GetBondBetweenAtoms(idx, j)
            if bond is None or bond.IsInRing():
                continue
            if bond.GetBondTypeAsDouble() != 1.0:
                continue
            if is_conjugated_junction(mol, (idx, j)):
                continue
            # Must have neighbours on both sides to define a dihedral.
            n1 = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() != j]
            n2 = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != idx]
            if n1 and n2:
                candidates.append((idx, j))
    if not candidates:
        return None
    return min(candidates, key=lambda b: (b[0], b[1]))


def _junction_atoms_of(mol: Chem.Mol, core_atom_map: Dict[int, int]) -> set:
    """Product indices of core atoms that touch synthon-derived heavy atoms.

    These are the atoms the harmonic-restraint relaxation leaves completely
    free so the newly formed junction bond can relax to a physical length."""
    mapped = set(core_atom_map.values())
    junction: set = set()
    for prod_idx in mapped:
        if prod_idx >= mol.GetNumAtoms():
            continue
        for nbr in mol.GetAtomWithIdx(prod_idx).GetNeighbors():
            if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in mapped:
                junction.add(prod_idx)
                break
    return junction


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

        Protocol (computational-chemistry correct):

        1. Assign stereochemistry on the product graph and embed with ETKDG
           with ``enforceChirality = True`` so (R)/(S) synthons never invert
           during 3D generation.
        2. When ``core_reference`` (the pre-reaction scaffold with a
           conformer) and ``core_atom_map`` are given, embed with the
           scaffold coordinates as **embedding constraints** (the
           ``ConstrainedEmbed`` protocol: ETKDG's distance-geometry solver
           builds the synthon *around* the anchored core, so the newly
           formed junction bond starts at a physical length instead of
           being torn by a post-hoc snap).
        3. Relax with **harmonic position restraints** (spring constant
           50 kcal/mol/A^2) on the scaffold atoms instead of freezing them
           rigidly; the junction atoms stay loose so the newly formed bond
           can relax to a physical length/angle without tearing.

        Returns a molecule with exactly one conformer, or ``None`` when
        embedding fails.
        """
        if product_mol is None:
            raise ValueError("embed_product requires a valid Mol")

        mol = Chem.AddHs(Chem.Mol(product_mol))
        # Stereochemistry: make every stereocenter explicit on the graph, then
        # reject conformers that invert it during embedding.
        Chem.AssignStereochemistry(
            mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
        )

        # ------------------------------------------------------------------
        # Constrained embedding: hand the reference scaffold coordinates to
        # the distance-geometry solver (RDKit ``ConstrainedEmbed`` protocol).
        # ------------------------------------------------------------------
        coord_map = None
        if core_reference is not None and core_atom_map:
            if core_reference.GetNumConformers() > 0:
                ref_conf = core_reference.GetConformer()
                coord_map = {}
                for core_idx, prod_idx in core_atom_map.items():
                    if (
                        core_idx < core_reference.GetNumAtoms()
                        and prod_idx < mol.GetNumAtoms()
                    ):
                        p = ref_conf.GetAtomPosition(int(core_idx))
                        coord_map[int(prod_idx)] = Point3D(p.x, p.y, p.z)
                if not coord_map:
                    coord_map = None
            else:
                logger.warning("Core reference has no conformer; ignoring lock.")

        status = -1
        if coord_map:
            # RDKit's ``ConstrainedEmbed`` protocol: the distance-geometry
            # solver receives the scaffold coordinates as constraints via the
            # ``coordMap`` keyword (individual-argument call signature; the
            # ``ETKDGv3()`` parameter object has no ``coordMap`` attribute in
            # modern RDKit). Defaults of the kwargs form are ETKDG
            # (useExpTorsionAnglePrefs=True, useBasicKnowledge=True).
            try:
                status = AllChem.EmbedMolecule(
                    mol,
                    coordMap=coord_map,
                    randomSeed=int(random_seed),
                    enforceChirality=True,
                    useRandomCoords=False,
                )
            except Exception:  # pragma: no cover - RDKit can raise on odd graphs
                status = -1

        if status != 0:
            # Fallback: unconstrained ETKDG (+ Kabsch + snap below).
            try:
                mol = Chem.Mol(Chem.AddHs(Chem.Mol(product_mol)))
                Chem.AssignStereochemistry(
                    mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
                )
            except Exception:  # pragma: no cover
                pass
            params = AllChem.ETKDGv3()
            params.randomSeed = int(random_seed)
            params.useRandomCoords = True
            params.maxIterations = int(max_attempts)
            params.enforceChirality = True
            try:
                status = AllChem.EmbedMolecule(mol, params)
            except Exception:  # pragma: no cover - RDKit can raise on odd graphs
                status = -1
        if status != 0:
            # Retry once without ETKDG constraints.
            try:
                fallback = AllChem.ETKDGv3()
                fallback.randomSeed = int(random_seed) + 1
                fallback.enforceChirality = True
                status = AllChem.EmbedMolecule(mol, fallback)
            except Exception:
                status = -1
        if status != 0:
            logger.warning("Conformer embedding failed; returning None.")
            return None

        if core_reference is not None and core_atom_map and status == 0:
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

                    # With constrained embedding the product is already in the
                    # reference frame; a rigid alignment is only needed for the
                    # unconstrained fallback path. Detect a misaligned frame by
                    # centroid offset and Kabsch-align if necessary.
                    if align_to_reference and len(pairs) >= 3:
                        P = np.array([prod_pos[p] for _, p in pairs])
                        Q = np.array([ref_pos[c] for c, _ in pairs])
                        if (
                            float(np.linalg.norm(P.mean(0) - Q.mean(0))) > 0.5
                            or float(np.abs(P - Q).max()) > 1.0
                        ):
                            R, t = _kabsch(P, Q)
                            prod_pos = prod_pos @ R.T + t

                    # Snap scaffold atoms onto the reference frame (micro
                    # correction of ETKDG's residual coordinate drift).
                    for (core_idx, prod_idx) in pairs:
                        prod_pos[prod_idx] = ref_pos[core_idx]

                    for i in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(i, Point3D(*prod_pos[i]))

                    # Heal bond geometry with harmonic restraints (NOT rigid
                    # freezing): the junction atoms relax freely so the new
                    # bond reaches a physical length.
                    ConformerEngine._relax_with_harmonic_restraints(
                        mol, pairs, junction_atoms=_junction_atoms_of(
                            mol, core_atom_map
                        )
                    )
        return mol

    @staticmethod
    def _relax_with_harmonic_restraints(
        mol: Chem.Mol,
        frozen_pairs,
        junction_atoms: Optional[set] = None,
        max_iterations: int = 300,
        spring_constant: float = 50.0,
        max_displacement: float = 0.05,
    ) -> bool:
        """Relax a freshly attached synthon with harmonic scaffold restraints.

        Non-junction scaffold atoms are tethered to their current positions
        with a flat-bottom harmonic potential (spring constant
        ``spring_constant`` kcal/mol/A^2, free displacement
        ``max_displacement`` A) instead of being frozen rigidly, and the
        junction atoms are left completely free so the newly formed bond
        relaxes to physical lengths/angles. Returns True when a relaxation
        was performed.
        """
        junction_atoms = junction_atoms or set()
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                props = AllChem.MMFFGetMoleculeProperties(mol)
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                add_position = (
                    ff.MMFFAddPositionConstraint
                    if hasattr(ff, "MMFFAddPositionConstraint")
                    else None
                )
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol)
                add_position = (
                    ff.UFFAddPositionConstraint
                    if hasattr(ff, "UFFAddPositionConstraint")
                    else None
                )
            if ff is None:
                return False
            if add_position is None:  # pragma: no cover - very old RDKit
                for _, prod_idx in frozen_pairs:
                    ff.AddFixedPoint(prod_idx)
            else:
                for _, prod_idx in frozen_pairs:
                    if prod_idx in junction_atoms:
                        # Junction atoms relax under a loose tether: enough
                        # freedom to heal the new bond's length/angle, without
                        # letting the whole functional group migrate.
                        add_position(int(prod_idx), 0.25, float(spring_constant))
                    else:
                        add_position(
                            int(prod_idx), float(max_displacement), float(spring_constant)
                        )
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
        scaffold side stays fixed. Resonance-locked junctions (amides,
        esters, double and ring bonds) are **refused** - rotating them would
        violate molecular-orbital planarity - and the molecule is returned
        unchanged (use :meth:`apply_junction_torsion` for the chemistry-aware
        path). Returns the same molecule object for chaining.
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("set_dihedral requires a molecule with a conformer")
        if is_conjugated_junction(mol, junction_bond):
            logger.warning(
                "Refusing to rotate resonance-locked junction %s (amide/ester/"
                "ring/double bond); use apply_junction_torsion.",
                junction_bond,
            )
            return mol
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
    def apply_junction_torsion(
        mol: Chem.Mol,
        junction_bond: Tuple[int, int],
        angle_rad: float,
        pocket_coords: Optional[np.ndarray] = None,
        pocket_vdw: Optional[np.ndarray] = None,
    ) -> Tuple[Chem.Mol, float, Optional[Tuple[int, int]]]:
        """Apply a policy torsion with resonance-aware chemistry.

        * Free single-bond junction: rotate it directly to ``angle_rad``.
        * Amide / ester junction: snap the junction to the nearest planar
          geometry of {trans (180 deg), cis (0 deg)} - choosing whichever
          clashes less with the pocket when available - and propagate the
          requested torsion onto the first adjacent *true* single bond on
          the synthon side (the protein phi/psi analogue).

        Returns ``(mol, applied_junction_angle_rad, rotated_bond)`` where
        ``rotated_bond`` is the bond the torsion actually moved (``None``
        when no rotatable bond existed).
        """
        if mol.GetNumConformers() == 0:
            raise ValueError("apply_junction_torsion requires a conformer")

        if not is_conjugated_junction(mol, junction_bond):
            ConformerEngine.set_dihedral(mol, junction_bond, float(angle_rad))
            actual = ConformerEngine.get_dihedral(mol, junction_bond)
            return mol, float(actual if actual is not None else angle_rad), tuple(junction_bond)

        # --- Resonance-locked junction: planar snap + adjacent rotation ---
        targets = (math.pi, 0.0)  # trans first (amide/ester ground state)
        best_angle, best_energy = targets[0], None
        for candidate in targets:
            ConformerEngine._force_dihedral(mol, junction_bond, candidate)
            if pocket_coords is not None and len(pocket_coords) > 0:
                energy = ConformerEngine.compute_lennard_jones_np(
                    np.array(mol.GetConformer().GetPositions(), dtype=np.float64),
                    pocket_coords,
                    pocket_vdw=pocket_vdw,
                )
                best_energy = energy if best_energy is None else best_energy
                if best_energy is not None and energy <= best_energy:
                    best_angle, best_energy = candidate, energy
            else:
                best_angle = targets[0]  # prefer trans without a pocket

        ConformerEngine._force_dihedral(mol, junction_bond, best_angle)

        adjacent = _adjacent_rotatable_bond(mol, junction_bond)
        if adjacent is not None:
            ConformerEngine.set_dihedral(mol, adjacent, float(angle_rad))
        return mol, float(best_angle), adjacent

    @staticmethod
    def _force_dihedral(
        mol: Chem.Mol, junction_bond: Tuple[int, int], angle_rad: float
    ) -> None:
        """Rotate a (possibly conjugated) junction to a planar geometry.

        Only used for the discrete amide/ester snap where the target geometry
        is a resonance-legal planar state; moves the synthon side only.
        """
        definition = _dihedral_definition(mol, junction_bond)
        if definition is None:
            return
        d1, b1, b2, d4 = definition
        angle_deg = float(np.degrees(((angle_rad + math.pi) % (2 * math.pi)) - math.pi))
        try:
            rdMolTransforms.SetDihedralDeg(
                mol.GetConformer(), d1, b1, b2, d4, angle_deg
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Planar snap failed (%s); leaving geometry unchanged.", exc)

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
        rotatable = (
            junction_bond
            if not is_conjugated_junction(mol, junction_bond)
            else _adjacent_rotatable_bond(mol, junction_bond)
        )
        if rotatable is None:
            current = ConformerEngine.get_dihedral(mol, junction_bond)
            return mol, float(current or 0.0), 0.0
        for angle in np.linspace(-np.pi, np.pi, grid_size, endpoint=False):
            ConformerEngine.set_dihedral(mol, rotatable, float(angle))
            clash = ConformerEngine.compute_steric_clash_np(
                np.array(mol.GetConformer().GetPositions(), dtype=np.float64),
                pocket_coords,
                ligand_vdw,
                pocket_vdw,
                subset_indices=movable,
            )
            if clash < best_clash:
                best_clash, best_angle = clash, float(angle)

        ConformerEngine.set_dihedral(mol, rotatable, best_angle)
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
    # 3. Scoring: soft clash penalties + Lennard-Jones interaction energy
    # ------------------------------------------------------------------
    # Lennard-Jones well depth per ligand-pocket atom pair (kcal/mol).
    # Small on purpose: this is a smooth ML-facing surrogate of dispersion,
    # not a calibrated force field.
    LJ_EPSILON = 0.10
    # Fraction of sigma where the repulsive r^-12 wall is replaced by its
    # C1-linear continuation (the "linearized LJ" of coarse-grained MD).
    # Below 0.8*sigma (~2.7 A for a C-C pair - already a hard clash) the
    # potential stays exact-LJ, so the well and the zero-crossing at sigma
    # are untouched, while a full atom overlap is bounded at ~650*epsilon
    # with a constant, non-exploding gradient.
    LJ_REPULSIVE_SWITCH = 0.8

    @staticmethod
    def compute_lennard_jones_energy(
        ligand_coords: torch.Tensor,
        pocket_coords: torch.Tensor,
        ligand_vdw: Optional[torch.Tensor] = None,
        pocket_vdw: Optional[torch.Tensor] = None,
        epsilon: float = LJ_EPSILON,
        repulsive_switch: float = LJ_REPULSIVE_SWITCH,
    ) -> torch.Tensor:
        """Differentiable Lennard-Jones 6-12 interaction energy.

        ``V(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]`` per atom pair with
        ``sigma = r_i + r_j`` (the Bondi contact distance). For
        ``r < repulsive_switch * sigma`` the repulsive wall is replaced by
        its C1-continuous linear continuation ("linearized LJ"), so the
        energy and its gradient stay bounded under pathological overlaps
        without distorting the well.

        Unlike the pure repulsion-wall clash loss, this potential has an
        *attractive well* at ``r = 2^(1/6) sigma`` (depth ``-epsilon`` per
        pair): an atom sitting at van der Waals contact is favourable, an
        atom clashing is heavily penalised, and an atom drifting into open
        solvent returns to zero - so the global minimum is a well-packed
        pose, not "ligand teleports out of the pocket".

        Returns the summed pair energy (scalar tensor, kcal/mol surrogate).
        """
        if ligand_coords.dim() != 2 or ligand_coords.size(1) != 3:
            raise ValueError(
                f"ligand_coords must be [N, 3], got {tuple(ligand_coords.shape)}"
            )
        if pocket_coords.dim() != 2 or pocket_coords.size(1) != 3:
            raise ValueError(
                f"pocket_coords must be [M, 3], got {tuple(pocket_coords.shape)}"
            )
        if ligand_coords.numel() == 0 or pocket_coords.numel() == 0:
            return torch.zeros((), device=pocket_coords.device, dtype=torch.float32)

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

        coords_l = ligand_coords.to(torch.float32)
        coords_p = pocket_coords.to(torch.float32)
        r = torch.cdist(coords_l, coords_p).clamp(min=1e-6)     # [N, M]
        sigma = (
            ligand_vdw.to(torch.float32).unsqueeze(1)
            + pocket_vdw.to(torch.float32).unsqueeze(0)
        )
        eps = float(epsilon)
        c = float(repulsive_switch)

        # Exact 6-12 branch (used for r >= c*sigma).
        x = sigma / r
        x6 = x.pow(6)
        exact = 4.0 * eps * (x6 * x6 - x6)

        # C1-linear continuation below c*sigma: value V(c) + slope*(r - c*sigma)
        # evaluated per pair (the slope carries the 1/sigma factor).
        c6 = c ** -6
        c12 = c6 * c6
        v_switch = 4.0 * eps * (c12 - c6)
        slope = 24.0 * eps * (c ** -7 - 2.0 * c ** -13) / sigma
        linear = v_switch + slope * (r - c * sigma)

        energy = torch.where(r < c * sigma, linear, exact)
        return energy.sum()

    @staticmethod
    def compute_lennard_jones_np(
        ligand_coords: np.ndarray,
        pocket_coords: np.ndarray,
        ligand_vdw: Optional[np.ndarray] = None,
        pocket_vdw: Optional[np.ndarray] = None,
        epsilon: float = LJ_EPSILON,
        repulsive_switch: float = LJ_REPULSIVE_SWITCH,
    ) -> float:
        """NumPy twin of :meth:`compute_lennard_jones_energy`."""
        lig = np.asarray(ligand_coords, dtype=np.float64)
        poc = np.asarray(pocket_coords, dtype=np.float64)
        if lig.size == 0 or poc.size == 0:
            return 0.0
        if ligand_vdw is None:
            ligand_vdw = np.full(len(lig), DEFAULT_VDW_RADIUS)
        if pocket_vdw is None:
            pocket_vdw = np.full(len(poc), DEFAULT_VDW_RADIUS)

        diff = lig[:, None, :] - poc[None, :, :]
        r = np.sqrt((diff ** 2).sum(-1)).clip(min=1e-6)
        sigma = np.asarray(ligand_vdw, dtype=float)[:, None] + np.asarray(
            pocket_vdw, dtype=float
        )[None, :]
        eps = float(epsilon)
        c = float(repulsive_switch)

        x = sigma / r
        x6 = x ** 6
        exact = 4.0 * eps * (x6 * x6 - x6)

        c6 = c ** -6
        c12 = c6 * c6
        v_switch = 4.0 * eps * (c12 - c6)
        slope = 24.0 * eps * (c ** -7 - 2.0 * c ** -13) / sigma
        linear = v_switch + slope * (r - c * sigma)

        return float(np.where(r < c * sigma, linear, exact).sum())

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
    "is_conjugated_junction",
]
