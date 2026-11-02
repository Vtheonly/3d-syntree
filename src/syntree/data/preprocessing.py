"""Multi-source preprocessing into standardized pockets and ligands (#16 DATA-01).

Issue #16 (legacy "Vulnerability 1"): the legacy loaders relied on loose
directory scans, ad-hoc discovery and uncurated splits.  This module is
the canonical ingestion/standardization library consumed by
``scripts/preprocess_multidataset.py``; it converts the three raw sources
pinned in ``docs/data/RAW_SOURCE_PROVENANCE.yaml`` —

* CrossDocked2020 ``pocket10`` (archive-level RMSD <= 1.0 A filter),
* PDBbind v2020 Refined Set (X-ray, resolution <= 2.5 A),
* BindingMOAD (affinity-annotated biological complexes resolved via RCSB)

— into one uniform artifact triple per complex:

* ``<complex_id>_pocket.pdb``  — standardized binding pocket,
* ``<complex_id>_ligand.sdf``  — sanitized co-crystallized ligand,
* one line in the unified provenance manifest ``processed_manifest.jsonl``.

Standardization policies (issue-pinned and binding, D-V2-021)
-------------------------------------------------------------
**Pocket extraction.**  A clean sphere of radius
:data:`POCKET_RADIUS_ANGSTROM` (10.0 A) around the **ligand centroid**
(heavy-atom centroid of the sanitized ligand conformer).  Kept records:
``ATOM`` residues (standard amino acids), documented modified residues
:data:`MODIFIED_RESIDUES` (selenomethionine & co., integrated chain
members), and structural/catalytic metals :data:`METAL_KEPT_ELEMENTS`.
Stripped records: crystallographic waters :data:`WATER_RESNAMES`,
crystallization salts/buffers/additives :data:`ADDITIVE_RESNAMES`, and
every other ligand-like HETATM component that is not the ligand of
record ("non-drug HETATM components", issue #16 §2).  Metals are kept
deliberately: catalytic Zn/Mg/... are part of the binding environment
(constitution #45 §11 — no removal of biologically essential species
without a justified policy; the policy is documented here).

**Ligand standardization.**  Valid RDKit sanitization is mandatory;
``150 <= MW <= 800 Da`` (RDKit exact-free :func:`Descriptors.MolWt`),
``heavy atoms >= 10``.  Every rejection carries an exact machine-readable
reason code — never a silent drop, never a fabricated substitution
(constitution #45 §11).

**Metadata completeness.**  Every manifest record carries
``resolution_angstrom``, ``rmsd_angstrom`` and ``ligand_heavy_atoms``.
Where a source genuinely lacks a value the field is ``null`` with an
explicit ``provenance`` note (e.g. CrossDocked per-pose RMSD is not
carried by the pinned mirror — the archive-level RMSD <= 1.0 filter is
the documented guarantee).  Nulls are recorded, never invented.

**Determinism and resume.**  Artifacts are written atomically
(temp + rename); the manifest is append-only and resumable by
``complex_id``; identical inputs yield identical standardized artifacts
(no timestamps inside artifacts).

This module depends on ``rdkit`` only (``numpy`` for the centroid math).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

__all__ = [
    "PreprocessingError",
    "POCKET_RADIUS_ANGSTROM",
    "LIGAND_MW_RANGE",
    "LIGAND_MIN_HEAVY_ATOMS",
    "MANIFEST_SCHEMA_VERSION",
    "WATER_RESNAMES",
    "ADDITIVE_RESNAMES",
    "METAL_KEPT_ELEMENTS",
    "MODIFIED_RESIDUES",
    "RESOLUTION_RE_LIMIT_ANGSTROM",
    "LigandRejection",
    "LIGAND_REJECTION_CODES",
    "ManifestRecord",
    "ManifestWriter",
    "parse_resolution",
    "parse_experiment_method",
    "parse_pdb_residue_lines",
    "parse_conect_edges",
    "classify_resname",
    "ligand_centroid",
    "extract_pocket_lines",
    "chain_sequences",
    "ligand_from_mol_block",
    "ligand_from_pdb_residue",
    "apply_ligand_filters",
    "standardize_complex",
    "complex_id_for",
]

class PreprocessingError(ValueError):
    """Fail-closed error for malformed preprocessing inputs."""


#: Pocket extraction sphere radius around the ligand heavy-atom centroid.
POCKET_RADIUS_ANGSTROM = 10.0

#: Accepted ligand molecular-weight window (Da), issue #16 §2.
LIGAND_MW_RANGE = (150.0, 800.0)

#: Minimum ligand heavy-atom count, issue #16 §2.
LIGAND_MIN_HEAVY_ATOMS = 10

#: Manifest schema version (bumped on any breaking record change).
MANIFEST_SCHEMA_VERSION = "2.0.0"

#: PDBbind X-ray resolution limit (A), issue #16 §2.
RESOLUTION_RE_LIMIT_ANGSTROM = 2.5

#: Crystallographic water residue names.
WATER_RESNAMES = frozenset({"HOH", "DOD", "WAT", "H2O", "TIP", "TIP3", "SOL"})

#: Monatomic salt/counterion components (stripped as salts, not metals).
_SALT_COMPONENTS = frozenset(
    {"NA", "CL", "K", "BR", "I", "F", "LI", "CS", "RB", "NH4", "IOD", "BROM", "FLC"}
)

#: Crystallization buffers, cryoprotectants, additives and reagents whose
#: HETATM records are stripped from standardized pockets (documented list;
#: PDB three-letter codes, case-normalized on lookup).
ADDITIVE_RESNAMES = frozenset(
    {
        # polyols / cryoprotectants
        "GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "2PE", "PG0", "P6G", "P8G",
        "MPD", "ETH", "IPA", "DMSO", "DMF", "MOH", "ETOH",
        # buffers
        "MES", "TRS", "TRIS", "HEPES", "EPE", "PIPES", "MOPS", "MOPSO", "CHES",
        "BIS", "BIS-TRIS", "BTB", "CAC", "CACY", "IMD", "AZI", "BICINE",
        "TCEP", "DTT", "BME", "MESO",
        # carboxylic acids / salts
        "CIT", "ACT", "ACY", "FMT", "NO3", "SO4", "PO4", "PI", "PPO", "BCT",
        "CO3", "OXY", "SUL", "PHO", "PI4", "OLC", "OLA", "PLM", "STE",
        # detergents
        "BOG", "C8E", "LDA", "DDM", "LMN", "C12E", "MAL", "NAG",
        # common lab small molecules
        "GSH", "G.SSH", "DOD", "BEN", "PVC",
    }
)

#: Structural/catalytic metals deliberately KEPT in standardized pockets
#: (part of the binding environment; constitution #45 §11).  Alkali
#: halides and counterions are salts, not catalytic metals — see
#: :data:`_SALT_COMPONENTS`.
METAL_KEPT_ELEMENTS = frozenset(
    {
        "MG", "ZN", "CA", "FE", "MN", "CU", "NI", "CO", "CD", "HG", "MO", "W",
        "V", "CR", "PD", "PT", "AU", "AG", "SN", "AL", "GA", "SB", "BI",
        "SE", "TE", "TI", "ZR", "NB", "TA", "RE", "OS", "IR", "RH", "RU",
        "Y", "LA", "CE", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
        "YB", "LU", "TH",
    }
)

#: Documented modified residues that are integrated chain members (kept).
MODIFIED_RESIDUES = frozenset(
    {
        "MSE", "SEC", "PYL", "PCA", "CSO", "SEP", "TPO", "PTR", "KCX", "CME",
        "CSS", "CSX", "CSD", "OCS", "CSW", "CGU", "GLZ", "SNC", "LLY", "MLY",
        "HYP", "MLZ", "M3L", "ALY", "TRQ", "KYQ", "CAS", "CAF", "SNN", "MEN",
        "CSP", "NEP", "TYS", "ASX", "GLX", "SEC", "U2P", "4FB", "CSU", "SMC",
        "CSW", "CGA", "HLU", "HIC", "MHO", "TRN", "SRO", "NIY", "CGV",
    }
)

#: Standard amino acids (kept as pocket protein content).
STANDARD_RESIDUES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "UNK", "ASX", "GLX",
    }
)

#: Machine-readable ligand rejection codes (documented, exhaustive for #16).
LIGAND_REJECTION_CODES = frozenset(
    {
        "LIGAND_SDF_PARSE_FAILURE",
        "LIGAND_SANITIZATION_FAILURE",
        "LIGAND_MW_BELOW_RANGE",
        "LIGAND_MW_ABOVE_RANGE",
        "LIGAND_TOO_FEW_HEAVY_ATOMS",
        "LIGAND_BOND_PERCEPTION_FAILURE",
        "LIGAND_TEMPLATE_MISMATCH",
        "LIGAND_RESIDUE_NOT_FOUND",
        "LIGAND_MULTI_COMPONENT",
    }
)

#: Pocket-level rejection codes.
POCKET_REJECTION_CODES = frozenset({"POCKET_EMPTY", "POCKET_NO_ATOMS_IN_SPHERE"})

#: Source-level rejection codes.
SOURCE_REJECTION_CODES = frozenset(
    {
        "METHOD_NOT_XRAY",
        "RESOLUTION_MISSING",
        "RESOLUTION_ABOVE_LIMIT",
        "SOURCE_FILE_NOT_FOUND",
        "SOURCE_CHECKSUM_MISMATCH",
    }
)

_RESOLUTION_RE = re.compile(r"^REMARK   2 RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS", re.M)


@dataclass(frozen=True)
class LigandRejection(Exception):
    """A ligand filtered out with an exact reason code (fail-closed).

    Raised by :func:`apply_ligand_filters` / the ``ligand_*`` parsers;
    the caller converts it into a manifest ``rejected`` record — a
    rejection is a *result*, not an error (constitution #45 §11: record
    the source ID and exact failure code).
    """

    reason_code: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.reason_code}: {self.detail}"


# ---------------------------------------------------------------------------
# PDB text parsing
# ---------------------------------------------------------------------------

def parse_resolution(pdb_text: str) -> Optional[float]:
    """Parse ``REMARK   2 RESOLUTION. <x> ANGSTROMS``; ``None`` if absent.

    Not applicable / unresolved entries have no numeric REMARK 2; the
    caller records ``null`` with a provenance note — never a guess.
    """
    match = _RESOLUTION_RE.search(pdb_text)
    if match is None:
        return None
    return float(match.group(1))


def parse_experiment_method(pdb_text: str) -> Optional[str]:
    """Parse the first ``EXPDTA`` record (experiment method), or ``None``."""
    for line in pdb_text.splitlines():
        if line.startswith("EXPDTA"):
            return line[10:].strip()
    return None


def parse_pdb_residue_lines(
    pdb_text: str,
    *,
    resname: str,
    chain: Optional[str] = None,
    resseq: Optional[str] = None,
    model: int = 1,
) -> List[str]:
    """All HETATM lines of one residue (``resname[:chain:resseq]``).

    ``resseq`` is compared as a string after whitespace stripping (PDB
    columns 23-26); insertion codes (column 27) are ignored for matching
    but preserved in the returned lines.  Only the first model is read
    (NMR ensembles / multi-model files) unless ``model`` is overridden.
    """
    out: List[str] = []
    current_model = 1
    for line in pdb_text.splitlines():
        if line.startswith("MODEL "):
            try:
                current_model = int(line[10:14])
            except ValueError:
                current_model = 1
            continue
        if line.startswith("ENDMDL"):
            current_model = 1
            continue
        if current_model != model:
            continue
        if not (line.startswith("HETATM") or line.startswith("ATOM")):
            continue
        if line[17:20].strip().upper() != resname.upper():
            continue
        if chain is not None and line[21:22] != chain:
            continue
        if resseq is not None and line[22:26].strip() != resseq.strip():
            continue
        altloc = line[16:17]
        if altloc not in (" ", "A"):
            continue  # alternate conformers: keep the blank/primary/'A' set
        out.append(line)
    return out


def classify_resname(resname: str) -> str:
    """Classify a residue name into the pocket policy buckets.

    Returns one of ``'protein'`` (standard or documented modified
    residue), ``'water'``, ``'metal'``, ``'additive'`` (salts, buffers,
    cryoprotectants), ``'ligand'`` (anything else — ligand-like HETATM).
    """
    name = resname.strip().upper()
    if not name:
        return "ligand"
    if name in WATER_RESNAMES:
        return "water"
    if name in STANDARD_RESIDUES:
        return "protein"
    if name in MODIFIED_RESIDUES:
        return "protein"
    if name in METAL_KEPT_ELEMENTS and len(name) <= 3:
        # Metal HETATM: single-atom component named by element (ZN, MG, ...).
        if name not in _SALT_COMPONENTS:
            return "metal"
    if name in ADDITIVE_RESNAMES or name in _SALT_COMPONENTS:
        return "additive"
    return "ligand"


def _pdb_coordinates(line: str) -> Optional[Tuple[float, float, float]]:
    try:
        return (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    except ValueError:
        return None


def ligand_centroid(mol: Chem.Mol) -> np.ndarray:
    """Heavy-atom centroid of the ligand's first conformer (A)."""
    conf = mol.GetConformer()
    pos = np.asarray(
        [
            list(conf.GetAtomPosition(a.GetIdx()))
            for a in mol.GetAtoms()
            if a.GetAtomicNum() > 1
        ],
        dtype=np.float64,
    )
    if pos.size == 0:
        raise LigandRejection("LIGAND_TOO_FEW_HEAVY_ATOMS", "no heavy atoms")
    return pos.mean(axis=0)


def extract_pocket_lines(
    pdb_lines: Sequence[str],
    centroid: Sequence[float],
    *,
    radius: float = POCKET_RADIUS_ANGSTROM,
) -> List[str]:
    """The standardized pocket: kept records within ``radius`` of ``centroid``.

    Policy (module docstring): keep protein residues (standard +
    documented modified) and catalytic/structural metals; strip waters,
    documented additives/salts and ligand-like HETATM.  Kept records are
    returned in their original order (PDB files are chain/residue
    ordered), with coordinates untouched.
    """
    cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
    kept: List[str] = []
    for line in pdb_lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        resname = line[17:20].strip().upper()
        bucket = classify_resname(resname)
        if bucket in ("water", "additive", "ligand"):
            continue
        coords = _pdb_coordinates(line)
        if coords is None:
            continue  # malformed coordinate line — excluded, not guessed
        x, y, z = coords
        if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius:
            kept.append(line if line.endswith("\n") else line + "\n")
    return kept


_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "ASX": "B", "GLX": "Z", "UNK": "X",
}


def chain_sequences(
    pdb_lines: Sequence[str],
    *,
    restrict_to_chains: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Per-chain one-letter protein sequences from PDB ``ATOM`` records.

    Modified residues map to ``X``.  Residues are ordered by their first
    appearance (PDB files are chain/residue ordered); alternate location
    indicators are ignored (first occurrence wins).  When
    ``restrict_to_chains`` is given, only those chains are emitted —
    the standardized pocket's chains, read from the **full** protein
    file where available (PDBbind protein.pdb / MOAD full PDB) so the
    #17 sequence-clustered splitter sees full-length sequences instead
    of pocket fragments.  CrossDocked's mirror carries pocket-only
    files, so its sequences are pocket substrings by necessity
    (documented limitation; receptor PDB identity compensates there).
    """
    keep = None if restrict_to_chains is None else set(restrict_to_chains)
    seen: Dict[str, Dict[str, str]] = {}
    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip().upper()
        if classify_resname(resname) != "protein":
            continue
        chain = line[21:22] or "_"
        if keep is not None and chain not in keep:
            continue
        key = f"{line[22:26].strip()}{line[26:27].strip()}"
        per_chain = seen.setdefault(chain, {})
        if key not in per_chain:
            per_chain[key] = _THREE_TO_ONE.get(resname, "X")
    return {chain: "".join(res.values()) for chain, res in seen.items()}


# ---------------------------------------------------------------------------
# Ligand parsing and filters
# ---------------------------------------------------------------------------

def ligand_from_mol_block(
    mol_block: str, mol2_block: Optional[str] = None
) -> Chem.Mol:
    """Sanitized ligand from an SDF/MOL block; fail-closed with codes.

    Preserves the input 3D conformer (the docked/crystal pose).  When
    ``mol2_block`` is provided it is the TRIPOS mol2 form of the same
    ligand: the PDBbind mirror's V2000 SDFs mark fused heteroaromatic
    systems with bond type 4 in a form RDKit cannot kekulize (measured:
    676/5316 complexes, e.g. 185l/indole — the direct sanitize parse
    fails too), while the mol2 carries correct per-bond types; it is
    used as the documented fallback and the used format is NOT recorded
    separately (both encode the identical depositor molecule).
    """
    try:
        mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
        if mol is None:
            raise LigandRejection(
                "LIGAND_SDF_PARSE_FAILURE", "MolFromMolBlock returned None"
            )
        Chem.SanitizeMol(mol)
        return mol
    except LigandRejection:
        if not mol2_block:
            raise
        mol2 = Chem.MolFromMol2Block(mol2_block, sanitize=True, removeHs=False)
        if mol2 is not None:
            return mol2
        raise
    except Exception as sdf_error:  # noqa: BLE001 - fail closed with the code
        if mol2_block:
            mol2 = Chem.MolFromMol2Block(mol2_block, sanitize=True, removeHs=False)
            if mol2 is not None:
                return mol2
        raise LigandRejection(
            "LIGAND_SANITIZATION_FAILURE", f"{type(sdf_error).__name__}: {sdf_error}"
        ) from sdf_error


def parse_conect_edges(
    pdb_text: str, serials: Sequence[int]
) -> Dict[int, Tuple[int, ...]]:
    """CONECT bond records restricted to ``serials`` (authoritative ligand connectivity).

    RCSB depositions carry ``CONECT`` records for every HETATM residue —
    the depositor-declared bond table.  This is strictly more reliable
    than 3D proximity bonding, which misses long bonds in
    lower-resolution structures (measured: 1IEP/2.1 A imatinib loses two
    ring bonds under RDKit proximity bonding while its CONECT table is
    exact).
    """
    allowed = set(int(s) for s in serials)
    out: Dict[int, Tuple[int, ...]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("CONECT"):
            continue
        try:
            src = int(line[6:11])
        except ValueError:
            continue
        if src not in allowed:
            continue
        dsts = []
        for start in range(11, min(len(line), 31), 5):
            field = line[start : start + 5].strip()
            if not field:
                continue
            try:
                dst = int(field)
            except ValueError:
                continue
            if dst in allowed and dst != src:
                dsts.append(dst)
        if dsts:
            out[src] = tuple(sorted(set(dsts)))
    return out


def ligand_from_pdb_residue(
    hetatm_lines: Sequence[str],
    *,
    template_smiles: Optional[str] = None,
    conect: Optional[Dict[int, Sequence[int]]] = None,
) -> Chem.Mol:
    """Sanitized ligand from PDB HETATM lines (BindingMOAD path).

    Connectivity comes from the depositor's ``CONECT`` records when
    provided (:func:`parse_conect_edges`), otherwise from 3D proximity
    bonding (documented risk: long bonds in low-resolution structures
    may be missed).  Bond orders and formal charges are transferred from
    the annotation ``template_smiles``
    (:func:`AllChem.AssignBondOrdersFromTemplate`); symmetric molecules
    can admit several atom mappings and RDKit picks the first
    deterministically.  Coordinates come from the PDB records (the
    crystal pose).

    **No-template fallback is deliberately fail-closed.**
    ``rdDetermineBonds.DetermineBonds`` was evaluated on this RDKit
    build (2026.09.1) and rejected: on ``MolFromPDBBlock`` inputs it
    either fails with a spurious molecular-charge error (aspirin,
    ibuprofen, imatinib test cases) or silently returns wrong bond
    orders (benzene -> ``c1c#cc#cc#1``).  Geometry-only bond-order
    perception is therefore not trusted; a residue without a usable
    template is rejected with ``LIGAND_BOND_PERCEPTION_FAILURE`` — an
    exact, honest failure code (constitution #45 §11: record the source
    ID and failure reason, never fabricate chemistry).
    """
    if not hetatm_lines:
        raise LigandRejection("LIGAND_RESIDUE_NOT_FOUND", "no HETATM lines")
    block = "\n".join(hetatm_lines) + "\nEND\n"
    serials = []
    for line in hetatm_lines:
        try:
            serials.append(int(line[6:11]))
        except ValueError:
            raise LigandRejection(
                "LIGAND_RESIDUE_NOT_FOUND", f"unparseable serial: {line[:20]!r}"
            ) from None
    mol = Chem.MolFromPDBBlock(
        block, sanitize=False, removeHs=False, proximityBonding=False
    )
    if mol is None or mol.GetNumAtoms() == 0:
        raise LigandRejection("LIGAND_RESIDUE_NOT_FOUND", "empty PDB block")
    index_of = {serial: i for i, serial in enumerate(serials)}

    edges: set = set()
    if conect:
        for src, dsts in conect.items():
            src_idx = index_of.get(int(src))
            if src_idx is None:
                continue
            for dst in dsts:
                dst_idx = index_of.get(int(dst))
                if dst_idx is not None:
                    edges.add((min(src_idx, dst_idx), max(src_idx, dst_idx)))
    if not edges:
        # no CONECT table available: proximity bonding fallback
        proxied = Chem.MolFromPDBBlock(
            block, sanitize=False, removeHs=False, proximityBonding=True
        )
        if proxied is None or proxied.GetNumAtoms() != mol.GetNumAtoms():
            raise LigandRejection("LIGAND_RESIDUE_NOT_FOUND", "proximity bonding failed")
        mol = proxied
    else:
        editable = Chem.RWMol(mol)
        for i, j in sorted(edges):
            editable.AddBond(i, j, Chem.BondType.SINGLE)
        mol = editable.GetMol()

    template = None
    if template_smiles:
        candidate = Chem.MolFromSmiles(template_smiles)
        if candidate is not None:
            template = Chem.RemoveHs(candidate)
    if template is None:
        raise LigandRejection(
            "LIGAND_BOND_PERCEPTION_FAILURE",
            "no usable annotation SMILES template for bond-order perception",
        )
    if template.GetNumAtoms() != mol.GetNumAtoms():
        raise LigandRejection(
            "LIGAND_TEMPLATE_MISMATCH",
            f"template has {template.GetNumAtoms()} heavy atoms, "
            f"residue has {mol.GetNumAtoms()}",
        )
    try:
        ordered = AllChem.AssignBondOrdersFromTemplate(template, mol)
    except (ValueError, RuntimeError) as error:
        raise LigandRejection(
            "LIGAND_TEMPLATE_MISMATCH", f"template mapping failed: {error}"
        ) from error
    try:
        Chem.SanitizeMol(ordered)
    except Exception as error:  # noqa: BLE001
        raise LigandRejection(
            "LIGAND_SANITIZATION_FAILURE", f"template product: {error}"
        ) from error
    return ordered


def apply_ligand_filters(mol: Chem.Mol) -> Dict[str, object]:
    """Apply the issue-pinned ligand filters; return the ligand metadata.

    Raises :class:`LigandRejection` with the exact reason code when a
    filter fails; returns a metadata dict (MW, heavy atoms, rotatable
    bonds, formula, formal charge, canonical SMILES) on success.
    """
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception as error:  # noqa: BLE001
        raise LigandRejection(
            "LIGAND_SANITIZATION_FAILURE", f"SMILES export: {error}"
        ) from error
    heavy = mol.GetNumHeavyAtoms()
    if heavy < LIGAND_MIN_HEAVY_ATOMS:
        raise LigandRejection(
            "LIGAND_TOO_FEW_HEAVY_ATOMS", f"{heavy} < {LIGAND_MIN_HEAVY_ATOMS}"
        )
    mw = Descriptors.MolWt(mol)
    if mw < LIGAND_MW_RANGE[0]:
        raise LigandRejection("LIGAND_MW_BELOW_RANGE", f"{mw:.2f} < {LIGAND_MW_RANGE[0]}")
    if mw > LIGAND_MW_RANGE[1]:
        raise LigandRejection("LIGAND_MW_ABOVE_RANGE", f"{mw:.2f} > {LIGAND_MW_RANGE[1]}")
    return {
        "smiles": smiles,
        "molecular_weight_da": round(mw, 3),
        "heavy_atoms": heavy,
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "fraction_sp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
    }


# ---------------------------------------------------------------------------
# Complex ID scheme
# ---------------------------------------------------------------------------

def complex_id_for(source: str, *parts: str) -> str:
    """Deterministic complex id: ``<source>_<part...>`` (sanitized)."""
    joined = "_".join(str(p) for p in parts if p)
    return f"{source}_{joined}"


# ---------------------------------------------------------------------------
# Manifest records and writer
# ---------------------------------------------------------------------------

@dataclass
class ManifestRecord:
    """One complex's standardized outcome (accepted or rejected).

    Every field is JSON-serializable; ``null`` means *not available from
    this source* with the reason carried in ``provenance`` — never an
    invented value (constitution #45 §11).
    """

    complex_id: str
    source: str
    status: str  # 'accepted' | 'rejected'
    reason_code: Optional[str] = None
    detail: Optional[str] = None
    pdb_id: Optional[str] = None
    target_id: Optional[str] = None
    receptor_pdb_id: Optional[str] = None
    method: Optional[str] = None
    resolution_angstrom: Optional[float] = None
    rmsd_angstrom: Optional[float] = None
    affinity_uM: Optional[float] = None
    ligand: Optional[Dict[str, object]] = None
    pocket: Optional[Dict[str, object]] = None
    protein_chains: Optional[Dict[str, str]] = None
    artifacts: Optional[Dict[str, str]] = None
    provenance: List[str] = field(default_factory=list)
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


class ManifestWriter:
    """Append-only, resumable JSONL manifest.

    Resume semantics: ``__init__`` scans an existing manifest and indexes
    every well-formed line's ``complex_id`` (a truncated trailing line
    from an interrupted session is detected and excluded — the writer
    rewrites it cleanly on the next append).  ``already_processed``
    powers the script's ``--resume`` path so expensive sources are
    processed exactly once per complex across sessions.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._done: Dict[str, str] = {}
        self._valid_bytes = 0
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.endswith("\n"):
                        break  # interrupted trailing write
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    cid = record.get("complex_id")
                    if not isinstance(cid, str):
                        break
                    self._done[cid] = str(record.get("status", ""))
                    self._valid_bytes += len(line.encode("utf-8"))

    def already_processed(self, complex_id: str) -> bool:
        return complex_id in self._done

    def status_of(self, complex_id: str) -> Optional[str]:
        return self._done.get(complex_id)

    @property
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {"accepted": 0, "rejected": 0}
        for status in self._done.values():
            out[status] = out.get(status, 0) + 1
        return out

    def append(self, record: ManifestRecord) -> None:
        line = record.to_json_line() + "\n"
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        if size != self._valid_bytes:
            # truncated/garbage tail from an interrupted session — rebuild
            with open(self.path, "r", encoding="utf-8") as handle:
                good = handle.read(self._valid_bytes)
            with open(self.path, "w", encoding="utf-8") as handle:
                handle.write(good)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
        self._done[record.complex_id] = record.status
        self._valid_bytes += len(line.encode("utf-8"))


# ---------------------------------------------------------------------------
# Atomic artifact writers
# ---------------------------------------------------------------------------

def _open_text(path: str, gzip_output: bool):
    if gzip_output:
        # mtime=0: deterministic gzip headers (identical inputs ->
        # byte-identical artifacts; D-V2-021 reproducibility note).
        raw = gzip.GzipFile(filename=path, mode="wb", compresslevel=6, mtime=0)
        return io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
    return open(path, "wt", encoding="utf-8", newline="\n")


def _write_atomic(path: str, payload: str, *, gzip_output: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with _open_text(tmp, gzip_output) as handle:
        handle.write(payload)
    os.replace(tmp, path)


def _pocket_pdb_text(pocket_lines: Sequence[str]) -> str:
    body = "".join(
        (line if line.endswith("\n") else line + "\n") for line in pocket_lines
    )
    header = (
        "HEADER    3D-SYNTREE-V2 STANDARDIZED POCKET\n"
        "COMPND    POCKET (10.0 A SPHERE AROUND LIGAND CENTROID)\n"
    )
    return header + body + "TER\nEND\n"


def _ligand_sdf_text(mol: Chem.Mol, name: str) -> str:
    mol = Chem.Mol(mol)
    mol.SetProp("_Name", name)
    return Chem.MolToMolBlock(mol) + "$$$$\n"


# ---------------------------------------------------------------------------
# The core standardization
# ---------------------------------------------------------------------------

def standardize_complex(
    *,
    complex_id: str,
    source: str,
    protein_pdb_text: str,
    ligand: Chem.Mol,
    out_dir: str,
    pdb_id: Optional[str] = None,
    target_id: Optional[str] = None,
    receptor_pdb_id: Optional[str] = None,
    method: Optional[str] = None,
    resolution_angstrom: Optional[float] = None,
    rmsd_angstrom: Optional[float] = None,
    affinity_uM: Optional[float] = None,
    provenance: Optional[Sequence[str]] = None,
    gzip_output: bool = False,
    radius: float = POCKET_RADIUS_ANGSTROM,
) -> ManifestRecord:
    """Standardize one complex and write its artifacts.

    Pipeline: ligand filters (exact codes) -> centroid -> pocket sphere
    extraction -> chain sequences -> atomic artifact writes -> manifest
    record.  Rejections return a ``rejected`` record (artifacts are not
    written); malformed inputs raise :class:`PreprocessingError`.
    """
    notes = list(provenance or [])
    try:
        ligand_meta = apply_ligand_filters(ligand)
    except LigandRejection as rejection:
        return ManifestRecord(
            complex_id=complex_id,
            source=source,
            status="rejected",
            reason_code=rejection.reason_code,
            detail=rejection.detail[:300],
            pdb_id=pdb_id,
            target_id=target_id,
            receptor_pdb_id=receptor_pdb_id,
            method=method,
            resolution_angstrom=resolution_angstrom,
            rmsd_angstrom=rmsd_angstrom,
            affinity_uM=affinity_uM,
            provenance=notes,
        )

    centroid = ligand_centroid(ligand)
    protein_lines = protein_pdb_text.splitlines()
    pocket_lines = extract_pocket_lines(protein_lines, centroid, radius=radius)
    if not pocket_lines:
        return ManifestRecord(
            complex_id=complex_id,
            source=source,
            status="rejected",
            reason_code="POCKET_EMPTY",
            detail=f"no kept atoms within {radius} A of the ligand centroid",
            pdb_id=pdb_id,
            target_id=target_id,
            receptor_pdb_id=receptor_pdb_id,
            method=method,
            resolution_angstrom=resolution_angstrom,
            rmsd_angstrom=rmsd_angstrom,
            affinity_uM=affinity_uM,
            ligand=ligand_meta,
            provenance=notes,
        )

    chains = chain_sequences(
        protein_lines,
        restrict_to_chains=sorted(
            {(line[21:22] or "_") for line in pocket_lines}
        ),
    )
    pocket_atoms = sum(1 for line in pocket_lines if line.startswith(("ATOM", "HETATM")))
    pocket_residues = {
        (line[21:22], line[22:27].strip()) for line in pocket_lines
    }

    suffix_pocket = "_pocket.pdb.gz" if gzip_output else "_pocket.pdb"
    suffix_ligand = "_ligand.sdf.gz" if gzip_output else "_ligand.sdf"
    pocket_path = os.path.join(out_dir, f"{complex_id}{suffix_pocket}")
    ligand_path = os.path.join(out_dir, f"{complex_id}{suffix_ligand}")
    _write_atomic(pocket_path, _pocket_pdb_text(pocket_lines), gzip_output=gzip_output)
    _write_atomic(ligand_path, _ligand_sdf_text(ligand, complex_id), gzip_output=gzip_output)

    pocket_meta = {
        "n_atoms": pocket_atoms,
        "n_residues": len(pocket_residues),
        "n_chains": len(chains),
        "radius_angstrom": radius,
        "kept_het_types": sorted(
            {
                line[17:20].strip().upper()
                for line in pocket_lines
                if line.startswith("HETATM")
                and classify_resname(line[17:20].strip()) == "metal"
            }
        ),
    }
    ligand_meta["centroid"] = [round(float(x), 4) for x in centroid]

    return ManifestRecord(
        complex_id=complex_id,
        source=source,
        status="accepted",
        pdb_id=pdb_id,
        target_id=target_id,
        receptor_pdb_id=receptor_pdb_id,
        method=method,
        resolution_angstrom=resolution_angstrom,
        rmsd_angstrom=rmsd_angstrom,
        affinity_uM=affinity_uM,
        ligand=ligand_meta,
        pocket=pocket_meta,
        protein_chains=chains,
        artifacts={
            "pocket": os.path.relpath(pocket_path, out_dir),
            "ligand": os.path.relpath(ligand_path, out_dir),
        },
        provenance=notes,
    )
