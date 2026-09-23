"""Dataset curation primitives shared by all structural sources.

The curation layer deliberately converts heterogeneous public datasets into one
auditable representation before training. It does not invent missing labels,
and it treats crystallographic hetero components as removable unless explicitly
retained by the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import Descriptors, Lipinski


ARTIFACT_RESNAMES = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "MPD", "DMS",
    "ACT", "ACE", "FMT", "MES", "TRS", "HEP", "MPO", "PGE", "EOH", "IPA",
    "IMD", "NAG", "MAN", "BME", "TBU", "CL", "NA", "K", "CA", "MG", "ZN",
    "MN", "FE", "CO", "NI", "BR", "IOD", "I",
}


@dataclass(frozen=True)
class PocketExtractionConfig:
    radius_angstrom: float = 10.0
    remove_hetero_atoms: bool = True
    recenter: bool = True


@dataclass
class CuratedComplexRecord:
    complex_id: str
    source: str
    protein_pdb: str
    ligand_sdf: str
    pocket_pdb: str
    sequence: str
    protein_key: str
    affinity_type: Optional[str] = None
    affinity_value: Optional[float] = None
    affinity_unit: Optional[str] = None
    pose_rmsd: Optional[float] = None
    quality_flags: Optional[List[str]] = None
    split: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["quality_flags"] = list(self.quality_flags or [])
        return value


def _parse_pdb_atoms(path: str) -> List[dict]:
    atoms: List[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            atoms.append(
                {
                    "line": line.rstrip("\n"),
                    "record": line[:6].strip(),
                    "resname": line[17:20].strip().upper(),
                    "chain": line[21:22].strip(),
                    "resseq": line[22:26].strip(),
                    "icode": line[26:27].strip(),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
    return atoms


def extract_protein_sequence(protein_pdb: str) -> str:
    """Extract a deterministic structure-derived protein sequence."""
    residue_codes = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M",
    }
    seen = set()
    sequence = []
    for atom in _parse_pdb_atoms(protein_pdb):
        if atom["record"] != "ATOM" or atom["resname"] not in residue_codes:
            continue
        key = (atom["chain"], atom["resseq"], atom["icode"])
        if key in seen:
            continue
        seen.add(key)
        sequence.append(residue_codes[atom["resname"]])
    return "".join(sequence)


def _ligand_heavy_atom_coordinates(
    ligand: Chem.Mol,
) -> List[Tuple[float, float, float]]:
    if ligand.GetNumConformers() == 0:
        raise ValueError("Ligand must contain at least one 3D conformer")
    conf = ligand.GetConformer()
    coords = []
    for atom in ligand.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append((float(p.x), float(p.y), float(p.z)))
    if not coords:
        raise ValueError("Ligand contains no heavy atoms")
    return coords


def _ligand_centroid(
    coords: Sequence[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    n = float(len(coords))
    return (
        sum(x for x, _, _ in coords) / n,
        sum(y for _, y, _ in coords) / n,
        sum(z for _, _, z in coords) / n,
    )


def standardize_pocket(
    protein_pdb: str,
    ligand: Chem.Mol,
    output_pdb: str,
    config: PocketExtractionConfig = PocketExtractionConfig(),
) -> Dict[str, object]:
    """Extract one common radius-defined protein pocket from a raw PDB."""
    if config.radius_angstrom <= 0:
        raise ValueError("radius_angstrom must be positive")

    atoms = _parse_pdb_atoms(protein_pdb)
    ligand_xyz = _ligand_heavy_atom_coordinates(ligand)
    center = _ligand_centroid(ligand_xyz)
    radius2 = float(config.radius_angstrom) ** 2

    selected = []
    for atom in atoms:
        if config.remove_hetero_atoms and atom["record"] != "ATOM":
            continue
        dx = atom["x"] - center[0]
        dy = atom["y"] - center[1]
        dz = atom["z"] - center[2]
        if dx * dx + dy * dy + dz * dz <= radius2:
            selected.append(atom)

    if not selected:
        raise ValueError(
            f"No protein atoms found within {config.radius_angstrom:.1f} Å of ligand"
        )

    shift = center if config.recenter else (0.0, 0.0, 0.0)
    os.makedirs(os.path.dirname(os.path.abspath(output_pdb)) or ".", exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as handle:
        for serial, atom in enumerate(selected, start=1):
            x = atom["x"] - shift[0]
            y = atom["y"] - shift[1]
            z = atom["z"] - shift[2]
            line = atom["line"]
            line = f"{line[:6]}{serial:5d}{line[11:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            handle.write(line + "\n")
        handle.write("END\n")

    pocket_xyz = [
        (a["x"] - shift[0], a["y"] - shift[1], a["z"] - shift[2])
        for a in selected
    ]
    return {
        "atom_count": len(selected),
        "centroid": list(center),
        "radius_angstrom": config.radius_angstrom,
        "recentered": config.recenter,
        "bbox": {
            "min": [
                min(x for x, _, _ in pocket_xyz),
                min(y for _, y, _ in pocket_xyz),
                min(z for _, _, z in pocket_xyz),
            ],
            "max": [
                max(x for x, _, _ in pocket_xyz),
                max(y for _, y, _ in pocket_xyz),
                max(z for _, _, z in pocket_xyz),
            ],
        },
    }


def load_first_ligand(ligand_sdf: str) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(ligand_sdf, removeHs=False, sanitize=True)
    for mol in supplier:
        if mol is not None:
            return mol
    raise ValueError(f"No valid ligand found in {ligand_sdf}")


def ligand_quality_flags(ligand: Chem.Mol) -> List[str]:
    flags: List[str] = []
    if ligand.GetNumAtoms():
        residue_info = ligand.GetAtomWithIdx(0).GetPDBResidueInfo()
        if residue_info is not None:
            residue_name = residue_info.GetResidueName().strip().upper()
            if residue_name in ARTIFACT_RESNAMES:
                flags.append("crystallization_artifact")
    for prop_name in ("RESNAME", "PDB_RESNAME"):
        if ligand.HasProp(prop_name):
            if ligand.GetProp(prop_name).strip().upper() in ARTIFACT_RESNAMES:
                flags.append("crystallization_artifact")
    if ligand.GetNumConformers() == 0:
        flags.append("missing_3d_conformer")
    heavy = sum(1 for atom in ligand.GetAtoms() if atom.GetAtomicNum() > 1)
    if heavy < 5:
        flags.append("too_small")
    try:
        mw = float(Descriptors.MolWt(ligand))
        fsp3 = float(Lipinski.FractionCSP3(ligand))
    except Exception:
        return flags + ["descriptor_failure"]
    if mw < 80.0:
        flags.append("low_molecular_weight")
    if any(atom.GetAtomicNum() > 20 for atom in ligand.GetAtoms()):
        flags.append("metal_containing")
    if fsp3 == 0.0 and heavy >= 8:
        flags.append("fully_aromatic")
    return sorted(set(flags))


def write_jsonl(records: Iterable[CuratedComplexRecord], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return path


__all__ = [
    "ARTIFACT_RESNAMES",
    "PocketExtractionConfig",
    "CuratedComplexRecord",
    "extract_protein_sequence",
    "standardize_pocket",
    "load_first_ligand",
    "ligand_quality_flags",
    "write_jsonl",
]


def recenter_ligand(ligand: Chem.Mol, center: Sequence[float]) -> Chem.Mol:
    """Return a copy whose 3D conformer uses the same origin as the pocket."""
    if ligand.GetNumConformers() == 0:
        raise ValueError("Ligand must contain a 3D conformer")
    if len(center) != 3:
        raise ValueError("center must contain three coordinates")
    out = Chem.Mol(ligand)
    conf = out.GetConformer()
    for idx in range(out.GetNumAtoms()):
        point = conf.GetAtomPosition(idx)
        conf.SetAtomPosition(
            idx,
            Point3D(
                float(point.x - center[0]),
                float(point.y - center[1]),
                float(point.z - center[2]),
            ),
        )
    return out


def write_ligand_sdf(ligand: Chem.Mol, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    writer = Chem.SDWriter(path)
    writer.write(ligand)
    writer.close()
    return path


__all__ += ["recenter_ligand", "write_ligand_sdf"]
