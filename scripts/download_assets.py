#!/usr/bin/env python3
"""Headless asset retriever for 3D-SynTree.

Responsibilities:
1. **Real datasets** – attempt to download the CrossDocked2020 pocket-ligand
   benchmark and the Enamine 3D synthon subset when their remote sources are
   reachable (HF Hub / Zenodo).
2. **Offline fallback** – synthesize a chemically valid, deterministic
   catalog of building blocks (Fsp3/MW computed with RDKit, handles
   detected with the real reaction engine) plus sample protein pockets, so
   the entire pipeline runs end-to-end with zero network access.

Usage:
    python scripts/download_assets.py --target-dataset crossdocked2020 \
        --synthon-subset 3d-diversity-15k --output-dir ./data
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- sp3-rich building blocks (pass the Fsp3 >= 0.42 floor) ---------------
_SP3_RICH_SYNTHONS = [
    # (smiles, handle)
    # carboxylic acids
    ("CC1(C)CC(C(=O)O)C1", "carboxylic_acid"),          # gem-dimethyl cyclobutane acid
    ("OC(=O)C1CCCCC1", "carboxylic_acid"),              # cyclohexanecarboxylic acid
    ("CC(C)(C)C(=O)O", "carboxylic_acid"),              # pivalic acid
    ("CC1CCC(C(=O)O)CC1", "carboxylic_acid"),           # methylcyclohexane acid
    ("O=C(O)C1CC2CCCCC2C1", "carboxylic_acid"),         # bicyclo[2.2.2]octane acid
    # amines
    ("NC1CC2(CC1)COC2", "primary_secondary_amine"),     # spiro amine
    ("CC1(N)CCC(O)CC1", "primary_secondary_amine"),     # hydroxy cyclohexyl amine
    ("C1CC(N)CC1", "primary_secondary_amine"),          # cyclopentylamine
    ("NCC1CCCCC1", "primary_secondary_amine"),          # aminomethylcyclohexane
    ("CC1(C)NCCC1", "primary_secondary_amine"),         # 4,4-dimethylpiperidine
    ("NC1CCC(CC1)C(C)C", "primary_secondary_amine"),    # isopropyl cyclohexylamine
    # alcohols
    ("CC1(O)CCCCC1", "alcohol"),                        # methylcyclohexanol
    ("CC(C)CO", "alcohol"),                             # isobutanol
    ("OC(C)(C)C(C)(C)O", "alcohol"),                    # pinacol
    ("C1CCC(O)CC1", "alcohol"),                         # cyclohexanol
    ("CC(C)CCCCO", "alcohol"),                          # 5-methylhexan-1-ol
    # aldehydes
    ("CC1(C)CCC(C=O)C1", "aldehyde"),                   # gem-dimethyl cyclohexanal
    ("C1CCC(CC1)C=O", "aldehyde"),                      # cyclohexanecarboxaldehyde
    ("CCC(C)C(C)C=O", "aldehyde"),                      # branched aliphatic aldehyde
    # alkynes
    ("CCCCCC#C", "alkyne"),                             # 1-hexyne (Fsp3 0.67)
    ("CCC#CCCCO", "alkyne"),                            # 5-hexyn-1-ol (Fsp3 0.67)
    # azides
    ("CCCCCN=[N+]=[N-]", "azide"),                      # 1-azidopentane (Fsp3 1.0)
    ("CC(C)CCN=[N+]=[N-]", "azide"),                    # branched alkyl azide
]

# --- genuinely multifunctional synthons: at least two independently
# reactive handles so autoregressive assembly can continue after one reaction
# consumes one handle. -------------------------------------------------------
_BIFUNCTIONAL_SYNTHONS = [
    ("N[C@@H](C)C(=O)O", "carboxylic_acid"),
    ("N[C@@H](CC)C(=O)O", "carboxylic_acid"),
    ("N[C@@H](CC(C)C)C(=O)O", "carboxylic_acid"),
    ("N[C@@H](CO)C(=O)O", "carboxylic_acid"),
    ("N[C@@H](CCO)C(=O)O", "carboxylic_acid"),
    ("NCCO", "primary_secondary_amine"),
    ("NCCCO", "primary_secondary_amine"),
    ("NCC(C)O", "primary_secondary_amine"),
    ("NCCN", "primary_secondary_amine"),
    ("NCCCN", "primary_secondary_amine"),
    ("OCC(=O)O", "carboxylic_acid"),
    ("CC(O)C(=O)O", "carboxylic_acid"),
    ("OCCC(=O)O", "carboxylic_acid"),
    ("O=C(O)CC(=O)O", "carboxylic_acid"),
    ("O=C(O)CCC(=O)O", "carboxylic_acid"),
    ("OCCO", "alcohol"),
    ("OCCCO", "alcohol"),
    ("CC(O)CO", "alcohol"),
    ("C#CCO", "alkyne"),
    ("C#CCN", "alkyne"),
    ("N=[N+]=[N-]CCO", "azide"),
    ("N=[N+]=[N-]CC#C", "azide"),
    ("Brc1ccc(B(O)O)cc1", "aryl_halide"),     # aryl halide + boronic acid
    ("Brc1ccc(N)cc1", "aryl_halide"),         # aryl halide + amine
    ("Brc1ccc(O)cc1", "aryl_halide"),         # aryl halide + alcohol
    ("Brc1ccc(C(=O)O)cc1", "aryl_halide"),    # aryl halide + carboxylic acid
    ("OB(O)c1ccc(N)cc1", "boronic_acid"),     # boronic acid + amine
]

# --- aryl coupling partners (Fsp3-exempt: aromatic by chemistry) ----------
_ARYL_SYNTHONS = [
    ("BrC1=CC=C(Cl)C=C1", "aryl_halide"),
    ("Brc1ccc(F)cc1", "aryl_halide"),
    ("IC1=CC=CC=C1", "aryl_halide"),
    ("Clc1ccc(Cl)c(Cl)c1", "aryl_halide"),
    ("Brc1ccccc1C", "aryl_halide"),
    ("Fc1ccc(Br)cc1", "aryl_halide"),
    ("B(C1=CC=C(Cl)C=C1)(O)O", "boronic_acid"),
    ("B(c1ccccc1)(O)O", "boronic_acid"),
    ("B(c1ccc(C)cc1)(O)O", "boronic_acid"),
    ("OB(O)c1ccc(F)cc1", "boronic_acid"),
    ("B(c1ccccn1)(O)O", "boronic_acid"),
]

# A small protein-like pocket backbone (GLY-GLY-GLY) used for smoke tests.
_SAMPLE_POCKET_PDB = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C
ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O
ATOM      5  N   GLY A   2       3.315   1.545   0.000  1.00 20.00           N
ATOM      6  CA  GLY A   2       3.866   2.906   0.000  1.00 20.00           C
ATOM      7  C   GLY A   2       4.417   4.267   0.000  1.00 20.00           C
ATOM      8  O   GLY A   2       3.680   5.254   0.000  1.00 20.00           O
ATOM      9  N   GLY A   3       5.723   4.451   0.000  1.00 20.00           N
ATOM     10  CA  GLY A   3       6.274   5.812   0.000  1.00 20.00           C
ATOM     11  C   GLY A   3       6.825   7.173   0.000  1.00 20.00           C
ATOM     12  O   GLY A   3       6.088   8.160   0.000  1.00 20.00           O
ATOM     13  OXT GLY A   3       8.131   7.357   0.000  1.00 20.00           O
TER
END
"""


def build_synthetic_catalog(output_dir: str, num_copies: int = 40) -> str:
    """Create the offline Enamine-style catalog with honest descriptors.

    Every synthon's Fsp3 and molecular weight are computed with RDKit and
    its ``primary_handle`` verified against the reaction engine, so the
    catalog that ships out is exactly what :class:`SynthonCatalog` will
    re-validate on load.
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    from syntree.chemistry.reactions import ReactionEngine

    engine = ReactionEngine()
    rows = []
    all_synthons = _SP3_RICH_SYNTHONS + _BIFUNCTIONAL_SYNTHONS + _ARYL_SYNTHONS
    for base_idx, (smiles, handle) in enumerate(all_synthons):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  ! skipping unparseable SMILES: {smiles}")
            continue
        detected = engine.handle_types(mol)
        if handle not in detected:
            print(f"  ! {smiles}: declared handle '{handle}' not detected ({detected}); skipping")
            continue
        for copy_idx in range(num_copies):
            rows.append(
                {
                    "id": f"EN300-{base_idx:04d}-{copy_idx:03d}",
                    "smiles": smiles,
                    "fsp3": float(Lipinski.FractionCSP3(mol)),
                    "mw": float(Descriptors.MolWt(mol)),
                    "primary_handle": handle,
                }
            )

    if not rows:
        raise RuntimeError("Synthetic catalog construction produced zero rows")

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    catalog_path = os.path.join(output_dir, "enamine_3d_subset.parquet")
    df.to_parquet(catalog_path, index=False)

    n_sp3 = len(_SP3_RICH_SYNTHONS) * num_copies
    n_bifunctional = len(_BIFUNCTIONAL_SYNTHONS) * num_copies
    n_aryl = len(_ARYL_SYNTHONS) * num_copies
    print(f"  Synthetic Enamine 3D catalog: {catalog_path}")
    print(
        f"    {len(df)} synthons ({n_sp3} sp3-rich + "
        f"{n_bifunctional} multifunctional + {n_aryl} aryl partners)"
    )
    return catalog_path


def build_sample_pockets(output_dir: str, data_subdir: str = "crossdocked") -> str:
    """Write deterministic sample pocket PDB files for smoke runs."""
    crossdocked_dir = os.path.join(output_dir, data_subdir)
    os.makedirs(crossdocked_dir, exist_ok=True)
    pocket_path = os.path.join(crossdocked_dir, "sample_pocket.pdb")
    with open(pocket_path, "w") as f:
        f.write(_SAMPLE_POCKET_PDB)
    print(f"  Sample protein pocket: {pocket_path}")
    return pocket_path


def try_download_crossdocked(output_dir: str, data_subdir: str = "crossdocked") -> bool:
    """Attempt the real CrossDocked2020 download (best-effort).

    The full benchmark (~4 GB) is hosted on Zenodo; when the network or
    the dataset is unavailable we return False and the caller falls back
    to synthetic mode.
    """
    zenodo_url = os.environ.get(
        "CROSSDOCKED_URL",
        "https://zenodo.org/record/6458305/files/crossdocked_pocket10.tar.gz",
    )
    try:
        import urllib.request

        dest = os.path.join(output_dir, data_subdir, "crossdocked_pocket10.tar.gz")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            return True
        print(f"  Downloading CrossDocked2020 from {zenodo_url} ...")
        urllib.request.urlretrieve(zenodo_url, dest)
        print(f"  Downloaded to {dest}; extract it into {os.path.dirname(dest)}")
        return True
    except Exception as exc:
        print(f"  ! CrossDocked download unavailable ({exc}); using synthetic pockets.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="3D-SynTree asset downloader")
    parser.add_argument("--target-dataset", type=str, default="crossdocked2020")
    parser.add_argument("--synthon-subset", type=str, default="3d-diversity-15k")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--catalog-copies", type=int, default=40,
                        help="replication factor of the base synthon set for "
                             "the offline catalog")
    parser.add_argument("--skip-download", action="store_true",
                        help="always use the offline synthetic fallback")
    args = parser.parse_args()

    print(f"[assets] target dataset: {args.target_dataset}")
    print(f"[assets] synthon subset: {args.synthon_subset}")

    # 1. Synthon catalog (synthetic fallback is always available offline).
    catalog_path = os.path.join(args.output_dir, "enamine_3d_subset.parquet")
    if not os.path.exists(catalog_path):
        print("[assets] building offline synthon catalog (RDKit-verified) ...")
        build_synthetic_catalog(args.output_dir, num_copies=args.catalog_copies)
    else:
        print(f"[assets] catalog already present: {catalog_path}")

    # 2. Protein pockets.
    downloaded = False
    if not args.skip_download:
        downloaded = try_download_crossdocked(args.output_dir)
    pocket_dir = os.path.join(args.output_dir, "crossdocked")
    has_pockets = any(f.endswith("_pocket.pdb") for f in os.listdir(pocket_dir)) \
        if os.path.isdir(pocket_dir) else False
    if not downloaded and not has_pockets:
        print("[assets] writing synthetic sample pocket ...")
        build_sample_pockets(args.output_dir)

    # 3. Manifest for downstream consumers.
    manifest = {
        "target_dataset": args.target_dataset,
        "synthon_subset": args.synthon_subset,
        "catalog_path": catalog_path,
        "data_dir": pocket_dir,
        "offline_fallback_used": not downloaded,
    }
    manifest_path = os.path.join(args.output_dir, "assets_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[assets] manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
