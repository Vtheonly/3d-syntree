#!/usr/bin/env python3
"""Download real PDB complexes and split them into protein/ligand pairs.

Step 1 of the seed-dataset build (tasklist Landmine 1): fetch a handful of
genuinely real, high-resolution, drug-liganded complexes from the RCSB, and
write them in the <id>_protein.pdb / <id>_ligand.pdb layout that
preprocess_multidataset.py discovers.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "./data/raw_pdb")
OUT.mkdir(parents=True, exist_ok=True)

# Real, well-characterised protein-ligand complexes spanning unrelated
# protein families (kinase, nuclear receptor, aspartyl protease, HSP90,
# PDE, ...). Anything that fails the pipeline's own validation drops out.
CANDIDATES = [
    "1M17",  # EGFR tyrosine kinase + erlotinib
    "3ERT",  # estrogen receptor alpha + raloxifene
    "1HVR",  # HIV-1 protease + indinavir
    "1A9U",  # p38 MAP kinase + pyrimidinylimidazole
    "2V7A",  # HSP90 + aminopyrimidine inhibitor
    "1TOW",  # thymidine kinase + inhibitor
    "3NYR",  # CDK2-like kinase
    "1OWG",  # casein kinase 1
    "2GVW",  # KDR (VEGFR2) kinase
    "3H3B",  # PLK1 kinase
    "4F9M",  # AKT kinase
    "1Z9E",  # PDE10A
    "1Y6C",  # coagulation factor Xa
    "2BM2",  # urokinase-type plasminogen activator
]


def fetch(pdb_id: str):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read().decode("utf-8", errors="ignore")
        if "ATOM" not in data or "HETATM" not in data:
            return None
        return data
    except Exception as exc:
        print(f"  ! {pdb_id}: {exc}")
        return None


def split_complex(pdb_id: str, text: str):
    """Split ATOM records (protein) from the largest organic HETATM residue."""
    protein_lines, hetatm = [], {}
    for line in text.splitlines():
        if line.startswith("ATOM"):
            protein_lines.append(line)
        elif line.startswith("HETATM"):
            resname = line[17:20].strip()
            # Skip common non-drug hetero groups (waters, ions, sugars,
            # buffers, cofactors). The pipeline's own read_ligand /
            # validate_ligand applies the final chemistry filter.
            if resname in {
                "HOH", "WAT", "SO4", "PO4", "GOL", "EDO", "MES", "TRS",
                "Cl", "Na", "Mg", "Zn", "Ca", "K", "MN", "NAG", "NDG",
                "FUC", "ACT", "DMS", "BME", "PEG", "EPE", "CID", "NO3",
                "NH4", "FMT", "ACY", "IPA", "ANP", "ADP", "ATP", "GDP",
                "GNP", "GTP", "NAD", "NAP", "FAD", "FMN", "HEM", "HEME",
            }:
                continue
            hetatm.setdefault(resname, []).append(line)

    if not protein_lines or not hetatm:
        return None
    # Largest hetero residue by atom count = the ligand.
    resname, lig_lines = max(hetatm.items(), key=lambda kv: len(kv[1]))
    if len(lig_lines) < 10:  # too small to be a real ligand
        return None

    prot = OUT / f"{pdb_id.lower()}_protein.pdb"
    lig = OUT / f"{pdb_id.lower()}_ligand.pdb"
    prot.write_text("\n".join(protein_lines) + "\nTER\nEND\n")
    lig.write_text("\n".join(lig_lines) + "\nEND\n")
    return prot, lig


kept = 0
for pdb_id in CANDIDATES:
    print(f"fetching {pdb_id} ...")
    text = fetch(pdb_id)
    if text is None:
        continue
    result = split_complex(pdb_id, text)
    if result is None:
        print(f"  ! {pdb_id}: no usable protein/ligand split")
        continue
    print(f"  ok -> {result[0].name} / {result[1].name}")
    kept += 1

print(f"kept {kept}/{len(CANDIDATES)} complexes under {OUT}")
