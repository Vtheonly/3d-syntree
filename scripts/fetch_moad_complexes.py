#!/usr/bin/env python3
"""Fetch BindingMOAD-annotated complexes as protein PDB + ligand SDF pairs.

BindingMOAD's curated binding annotations (for example the ``moad.parquet``
export of the ``jglaser/binding_affinity`` pipeline, itself derived from
BindingMOAD's ``every.csv`` binding-data file) provide PDB IDs, ligand
residue identifiers (``RES:CHAIN:RESSEQ``), SMILES and affinity values. The
underlying crystal structures are distributed by the RCSB without
registration. This script closes that acquisition chain:

* loads the annotation table (parquet or csv),
* downloads each annotated PDB entry from ``files.rcsb.org`` (retrying),
* parses resolution / experimental method from the structure itself,
* extracts the annotated ligand residues into standalone 3D SDFs
  (bonds from the PDB CONECT records + proximity inference, bond orders
  assigned from the annotation SMILES template when possible),
* drops common crystallization artifacts (``ARTIFACT_RESNAMES``),
* writes ``<pdb_id>_protein.pdb`` / ``<pdb_id>_ligand.sdf`` plus a source
  manifest JSONL consumable by ``preprocess_multidataset.py
  --bindingmoad-manifest``.

Example:
    python scripts/fetch_moad_complexes.py \
        --annotations moad.parquet \
        --output-dir raw_workspace/bindingmoad/complexes \
        --manifest-out raw_workspace/bindingmoad/bindingmoad_manifest.jsonl \
        --max-entries 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from typing import Dict, Iterable, List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syntree.data.curation import ARTIFACT_RESNAMES  # noqa: E402

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
USER_AGENT = "3d-syntree-moad-acquisition/1.0"


def parse_ligand_name(ligand_name: str) -> Dict[str, str]:
    """Split a MOAD ``RES:CHAIN:RESSEQ`` identifier into components.

    Missing components are tolerated: ``LIG``, ``LIG:613`` and
    ``LIG:A:613`` are all accepted (chain/residue default to first match).
    """
    parts = [p.strip() for p in str(ligand_name).split(":") if p.strip()]
    out = {"resname": parts[0].upper() if parts else ""}
    if len(parts) >= 2:
        out["resseq"] = parts[-1]
    if len(parts) >= 3:
        out["chain"] = parts[1]
    else:
        out["chain"] = ""
        if len(parts) == 2:  # RES:RESSEQ form
            out["resseq"] = parts[1]
        else:
            out["resseq"] = ""
    return out


def fetch_pdb_text(pdb_id: str, retries: int = 4) -> Optional[str]:
    """Download one PDB entry from the RCSB (None on final failure)."""
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"  [rcsb] {pdb_id} attempt {attempt + 1} failed: {exc}")
            time.sleep(2 + attempt)
    return None


def parse_structure_metadata(pdb_text: str) -> Dict[str, object]:
    """Pull resolution and experimental method from the PDB header."""
    resolution = None
    method = ""
    for line in pdb_text.splitlines():
        if line.startswith("EXPDTA"):
            method = line[10:].strip()
        elif line.startswith("REMARK   2 RESOLUTION"):
            # e.g. "REMARK   2 RESOLUTION.    1.45 ANGSTROMS."
            token = line[26:].split()
            if token:
                try:
                    resolution = float(token[0])
                except ValueError:
                    pass  # "NOT APPLICABLE" (cryo-EM / NMR)
        if resolution is not None and method:
            break
    return {"resolution": resolution, "experimental_method": method}


def _ligand_pdb_block(
    pdb_text: str, resname: str, chain: str, resseq: str
) -> Optional[str]:
    """Extract the HETATM/TER lines of one ligand residue as a mini PDB."""
    hetatm: List[str] = []
    conect: List[str] = []
    serials = set()
    for line in pdb_text.splitlines(True):
        rec = line[:6]
        if rec == "HETATM" and line[17:20].strip().upper() == resname:
            if chain and line[21:22].strip() != chain:
                continue
            if resseq and line[22:26].strip() != resseq:
                continue
            hetatm.append(line)
            serials.add(line[6:11])
        elif rec == "CONECT" and line[6:11] in serials:
            conect.append(line)
    if not hetatm:
        return None
    block = ["HEADER    LIGAND EXTRACTION"]
    block += hetatm
    block.append("TER\nEND\n")
    block += conect
    return "".join(block)


def extract_ligand(
    pdb_text: str,
    ligand_name: str,
    template_smiles: Optional[str],
) -> Optional[Chem.Mol]:
    """Build a sanitized 3D ligand Mol for the annotated residue."""
    ident = parse_ligand_name(ligand_name)
    resname = ident["resname"]
    if not resname or resname in ARTIFACT_RESNAMES:
        return None
    block = _ligand_pdb_block(
        pdb_text, resname, ident.get("chain", ""), ident.get("resseq", "")
    )
    if block is None:
        return None

    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
        f.write(block)
        tmp_path = f.name
    try:
        mol = Chem.MolFromPDBFile(
            tmp_path, removeHs=False, sanitize=False, proximityBonding=True
        )
        if mol is None or mol.GetNumAtoms() < 4:
            return None
        template = None
        if template_smiles:
            try:
                template = Chem.MolFromSmiles(str(template_smiles))
            except Exception:
                template = None
        if template is not None:
            try:
                mol = AssignBondOrdersFromTemplate(template, mol)
            except Exception:
                pass  # keep proximity-bonded crystal connectivity
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        if mol.GetNumConformers() == 0:
            return None
        mol.SetProp("_Name", ligand_name)
        mol.SetProp("PDB_RESNAME", resname)
        return mol
    finally:
        os.unlink(tmp_path)


def iter_annotations(path: str) -> Iterable[Dict]:
    """Yield annotation rows with standardized keys."""
    frame = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    for _, row in frame.iterrows():
        pdb_id = str(row.get("name") or row.get("pdb") or row.get("pdb_id") or "").strip()
        ligand = str(row.get("ligand_name") or "").strip()
        if not pdb_id or not ligand or pdb_id.lower() == "none":
            continue
        yield {
            "pdb_id": pdb_id,
            "ligand_name": ligand,
            "smiles": row.get("smiles"),
            "affinity_uM": row.get("affinity_uM"),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire BindingMOAD-annotated complexes from the RCSB."
    )
    parser.add_argument("--annotations", required=True,
                        help="BindingMOAD annotation table (parquet/csv)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--max-entries", type=int, default=0,
                        help="Stop after N fetched complexes (0 = all)")
    parser.add_argument("--pocket-cache-dir", default="",
                        help="Optional directory holding pre-downloaded PDB files")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.abspath(args.manifest_out)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    accepted, skipped = [], {"annotation_skip": 0, "fetch_fail": 0,
                             "ligand_extract": 0, "artifact": 0, "no_xray": 0,
                             "duplicate": 0}
    fetched = 0
    seen_complexes = set()
    manifest = open(manifest_path, "w", encoding="utf-8")

    for row in iter_annotations(args.annotations):
        if args.max_entries and fetched >= args.max_entries:
            break
        pdb_id = row["pdb_id"].lower()
        # One complex per (pdb entry, ligand residue type): MOAD annotates
        # every symmetry/copy instance separately, but they share the same
        # protein and (usually) the same pocket, so only the first annotated
        # instance is materialized.
        ident = parse_ligand_name(row["ligand_name"])
        complex_key = (pdb_id, ident["resname"])
        if complex_key in seen_complexes:
            skipped["duplicate"] += 1
            continue
        seen_complexes.add(complex_key)
        cache_path = os.path.join(args.pocket_cache_dir, f"{pdb_id}.pdb") \
            if args.pocket_cache_dir else ""
        if cache_path and os.path.exists(cache_path):
            pdb_text = open(cache_path, encoding="utf-8", errors="replace").read()
        else:
            pdb_text = fetch_pdb_text(pdb_id)
        if pdb_text is None:
            skipped["fetch_fail"] += 1
            continue

        meta = parse_structure_metadata(pdb_text)
        method = str(meta["experimental_method"]).lower()
        if method and ("x-ray" not in method and "xray" not in method):
            skipped["no_xray"] += 1
            continue

        ligand = extract_ligand(pdb_text, row["ligand_name"], row["smiles"])
        if ligand is None:
            # Either the residue is a known artifact, absent, or unsalitizable.
            ident = parse_ligand_name(row["ligand_name"])
            skipped["artifact" if ident["resname"] in ARTIFACT_RESNAMES
                    else "ligand_extract"] += 1
            continue

        complex_id = f"{pdb_id}_{parse_ligand_name(row['ligand_name'])['resname'].lower()}"
        protein_path = os.path.join(args.output_dir, f"{pdb_id}_protein.pdb")
        ligand_path = os.path.join(args.output_dir, f"{complex_id}_ligand.sdf")
        if not os.path.exists(protein_path):
            with open(protein_path, "w", encoding="utf-8") as f:
                f.write(pdb_text)
        writer = Chem.SDWriter(ligand_path)
        writer.write(ligand)
        writer.close()

        record = {
            "source": "bindingmoad",
            "complex_id": complex_id,
            "protein_path": os.path.abspath(protein_path),
            "ligand_path": os.path.abspath(ligand_path),
            "resolution": meta["resolution"],
            "experimental_method": meta["experimental_method"],
            "affinity_uM": row.get("affinity_uM"),
        }
        manifest.write(json.dumps(record, sort_keys=True) + "\n")
        accepted.append(record)
        fetched += 1
        if fetched % 25 == 0:
            print(f"[moad] fetched {fetched} complexes "
                  f"(skipped: {skipped})", flush=True)

    manifest.close()
    summary = {
        "accepted": len(accepted),
        "skipped": skipped,
        "manifest": manifest_path,
        "output_dir": os.path.abspath(args.output_dir),
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(os.path.dirname(manifest_path),
                           "moad_fetch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
