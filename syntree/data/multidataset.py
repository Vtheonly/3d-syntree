"""Shared utilities for unified protein-ligand preprocessing.

Raw datasets are external inputs. This module normalizes common file layouts
into one manifest schema and extracts protein-only 10 A pockets.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

CRYSTAL_JUNK = {
    "HOH", "DOD", "WAT", "SO4", "PO4", "ACT", "EDO", "DMS", "FMT", "GOL",
    "PEG", "PG4", "1PE", "MPD", "BME", "CIT", "TRS", "MES", "HEPES", "IOD",
    "CL", "NA", "MG", "ZN", "CA", "MN", "FE", "K", "NI", "CU",
}

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "SEC":"U","PYL":"O","MSE":"M",
}


def stable_sample_id(source: str, complex_id: str) -> str:
    raw = f"{source}:{complex_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def read_ligand(path: str) -> Optional[Chem.Mol]:
    """Load the largest chemically plausible ligand component."""
    p = Path(path)
    molecules: List[Chem.Mol] = []
    try:
        if p.suffix.lower() in {".sdf", ".sd"}:
            supplier = Chem.SDMolSupplier(str(p), sanitize=False, removeHs=False)
            molecules = [m for m in supplier if m is not None]
        elif p.suffix.lower() == ".mol2":
            mol = Chem.MolFromMol2File(str(p), sanitize=False, removeHs=False)
            if mol is not None:
                molecules = [mol]
        elif p.suffix.lower() == ".pdb":
            mol = Chem.MolFromPDBFile(str(p), sanitize=False, removeHs=False)
            if mol is not None:
                molecules = [mol]
        else:
            return None
    except Exception:
        return None

    candidates: List[Tuple[int, Chem.Mol]] = []
    for mol in molecules:
        try:
            probe = Chem.Mol(mol)
            Chem.SanitizeMol(probe)
            for frag in Chem.GetMolFrags(probe, asMols=True, sanitizeFrags=True):
                if frag.GetNumHeavyAtoms() == 0:
                    continue
                carbon = sum(a.GetAtomicNum() == 6 for a in frag.GetAtoms())
                if carbon == 0:
                    continue
                if all(a.GetSymbol().upper() in CRYSTAL_JUNK for a in frag.GetAtoms()):
                    continue
                candidates.append((frag.GetNumHeavyAtoms(), frag))
        except Exception:
            continue

    if not candidates:
        return None
    _, ligand = max(candidates, key=lambda item: item[0])
    try:
        ligand = Chem.RemoveHs(ligand)
        Chem.AssignStereochemistry(ligand, cleanIt=True, force=True)
        return ligand if ligand.GetNumConformers() else None
    except Exception:
        return None


def validate_ligand(
    ligand: Optional[Chem.Mol],
    min_mw: float = 150.0,
    max_mw: float = 800.0,
    min_heavy_atoms: int = 10,
) -> Tuple[bool, Dict[str, float]]:
    if ligand is None:
        return False, {}
    try:
        mw = float(Descriptors.MolWt(ligand))
        heavy = int(ligand.GetNumHeavyAtoms())
    except Exception:
        return False, {}
    return (
        float(min_mw) <= mw <= float(max_mw) and heavy >= int(min_heavy_atoms),
        {"mw": mw, "heavy_atoms": float(heavy)},
    )


def _pdb_atom_key(line: str) -> Tuple[str, int, str, str]:
    chain = line[21].strip() or "_"
    try:
        resseq = int(line[22:26])
    except ValueError:
        resseq = 0
    return chain, resseq, line[26].strip(), line[12:16].strip()


def _pdb_xyz(line: str) -> Optional[np.ndarray]:
    try:
        return np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None


def extract_protein_pocket(
    protein_pdb_path: str,
    ligand_mol: Chem.Mol,
    output_pocket_pdb: str,
    cutoff_radius: float = 10.0,
    min_atoms: int = 100,
) -> bool:
    """Write a deterministic protein-ATOM sphere within cutoff Angstrom."""
    if ligand_mol is None or ligand_mol.GetNumConformers() == 0:
        return False
    lig_coords = np.asarray(ligand_mol.GetConformer().GetPositions(), dtype=np.float64)

    selected: Dict[Tuple[str, int, str, str], Tuple[float, str, np.ndarray]] = {}
    try:
        lines = Path(protein_pdb_path).read_text(errors="ignore").splitlines(True)
    except OSError:
        return False

    for line in lines:
        if not line.startswith("ATOM"):
            continue
        if line[17:20].strip().upper() in CRYSTAL_JUNK:
            continue
        element = (line[76:78].strip() or line[12:14].strip()).upper()
        if element == "H":
            continue
        xyz = _pdb_xyz(line)
        if xyz is None:
            continue
        try:
            occupancy = float(line[54:60])
        except ValueError:
            occupancy = 0.0
        key = _pdb_atom_key(line)
        previous = selected.get(key)
        if previous is None or occupancy > previous[0]:
            selected[key] = (occupancy, line, xyz)

    pocket_lines = []
    for _, line, xyz in selected.values():
        if float(np.linalg.norm(lig_coords - xyz, axis=1).min()) <= float(cutoff_radius):
            pocket_lines.append(line.rstrip("\n") + "\n")

    pocket_lines.sort(key=lambda ln: (ln[21], ln[22:26], ln[26], ln[12:16]))
    if len(pocket_lines) < int(min_atoms):
        return False

    out = Path(output_pocket_pdb)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(pocket_lines) + "END\n")
    return True


def extract_chain_sequences(pdb_path: str) -> Dict[str, str]:
    """Extract one-letter sequences from protein ATOM records."""
    chains: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}
    try:
        lines = Path(pdb_path).read_text(errors="ignore").splitlines()
    except OSError:
        return {}

    for line in lines:
        if not line.startswith("ATOM"):
            continue
        aa = AA3_TO_1.get(line[17:20].strip().upper())
        if aa is None:
            continue
        chain, resseq, icode, _ = _pdb_atom_key(line)
        resid = (resseq, icode)
        chains.setdefault(chain, [])
        seen.setdefault(chain, set())
        if resid not in seen[chain]:
            seen[chain].add(resid)
            chains[chain].append(aa)

    return {k: "".join(v) for k, v in chains.items() if v}


def iter_jsonl(path: str) -> Iterator[Dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


__all__ = [
    "CRYSTAL_JUNK", "AA3_TO_1", "read_ligand", "validate_ligand",
    "extract_protein_pocket", "extract_chain_sequences", "stable_sample_id",
    "iter_jsonl",
]
