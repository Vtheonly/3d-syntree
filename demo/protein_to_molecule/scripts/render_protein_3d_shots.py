#!/usr/bin/env python3
"""Render the 3Dmol.js protein viewer scenes headlessly with Playwright/Chromium.

For each scene: fresh page load (deterministic orientation), stage fixed at
2400x1800 px (8x6 in at 300 DPI), toolbar hidden, element screenshot.
Also builds a 36-frame rotation GIF and independent verification numbers.

Usage: python3 protein_render_shoot.py --demo-root <demo dir>
Writes into <demo-root>/protein_structure_3d/:
  full_protein_cartoon.png/.pdf, active_site_closeup.png/.pdf,
  ligand_overlay_experimental_vs_reconstructed.png/.pdf, protein_surface_pocket.png/.pdf,
  protein_rotation.gif, render_provenance.json
"""
import argparse
import datetime
import gzip
import importlib.metadata
import json
import math
import pathlib
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

STILL_W, STILL_H = 2400, 1800
GIF_W, GIF_H = 960, 720
GIF_FRAMES = 36
GIF_STEP_DEG = 10

SCENES = [
    ("full_protein", "full_protein_cartoon",
     "Full experimental protein (4ITP chain A, 261 residues) as rainbow-by-position "
     "cartoon, Zn2+ sphere, ligand 1GD sticks, pocket residues as thin sticks"),
    ("active_site", "active_site_closeup",
     "Active-site close-up: pocket residues as sticks, catalytic residue labels, "
     "1GD ball-and-stick, Zn2+ sphere"),
    ("ligand_overlay", "ligand_overlay_experimental_vs_reconstructed",
     "Experimental 1GD (cyan sticks) overlaid with the 2-step synthetic "
     "reconstruction (orange translucent spheres), pocket context"),
    ("surface", "protein_surface_pocket",
     "VDW surface of chain A (translucent white) with buried ligand 1GD as spheres"),
]


def png_nonwhite_ratio(path: pathlib.Path) -> float:
    im = Image.open(path).convert("L")
    hist = im.histogram()
    total = sum(hist)
    nonwhite = sum(hist[:246])
    return nonwhite / total


def png_to_pdf(png: pathlib.Path, pdf: pathlib.Path) -> None:
    im = Image.open(png)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(pdf, "PDF", resolution=300.0)


def parse_pdb_atoms(text: str):
    atoms = []
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atoms.append({
                "record": line[:6].strip(), "name": line[12:16].strip(),
                "resn": line[17:20].strip(), "chain": line[21],
                "resi": int(line[22:26]),
                "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                "elem": line[76:78].strip() or line[12:16].strip()[0],
            })
    return atoms


def parse_sdf(txt: str):
    lines = txt.splitlines()
    n_at, n_bo = int(lines[3][0:3]), int(lines[3][3:6])
    atoms = [(float(lines[4 + i][0:10]), float(lines[4 + i][10:20]),
              float(lines[4 + i][20:30]), lines[4 + i][31:34].strip())
             for i in range(n_at)]
    bonds = [(int(lines[4 + n_at + i][0:3]), int(lines[4 + n_at + i][3:6]))
             for i in range(n_bo)]
    return atoms, bonds


def compute_checks(demo: pathlib.Path) -> dict:
    full_pdb = gzip.decompress(
        (demo / "structures" / "raw" / "rcsb_4ITP_full.pdb.gz").read_bytes()).decode()
    pocket_pdb = (demo / "structures" / "protein.pdb").read_text()
    lig_exp = gzip.decompress(
        (demo / "structures" / "raw" / "bindingmoad_4ITP_1GD_ligand.sdf.gz").read_bytes()).decode()
    lig_rec = (demo / "structures" / "final_ligand.sdf").read_text()

    full_atoms = parse_pdb_atoms(full_pdb)
    pocket_atoms = parse_pdb_atoms(pocket_pdb)
    zn = [a for a in full_atoms if a["resn"] == "ZN"][0]
    deposited_lig = [a for a in full_atoms if a["resn"] == "1GD"]

    def get(atoms, **kw):
        out = []
        for a in atoms:
            if all(a[k] == v for k, v in kw.items()):
                out.append(a)
        return out

    # 1. frame identity full vs pocket
    trp_full = get(full_atoms, name="NE1", resn="TRP", chain="A", resi=5)[0]["xyz"]
    trp_pocket = get(pocket_atoms, name="NE1", resn="TRP", chain="A", resi=5)[0]["xyz"]
    d_trp = math.dist(trp_full, trp_pocket)

    # pocket residue set equality (protein residues in demo pocket vs in deposition)
    pocket_res = {(a["chain"], a["resi"]) for a in pocket_atoms if a["record"] == "ATOM"}
    full_res = {(a["chain"], a["resi"]) for a in full_atoms if a["record"] == "ATOM"}
    pocket_subset = pocket_res.issubset(full_res)
    # coordinate identity for every pocket atom (name+resi matched, min over altloc variants)
    n_pocket_atom_records = len([a for a in pocket_atoms if a["record"] == "ATOM"])
    mism, maxdev = 0, 0.0
    for a in pocket_atoms:
        if a["record"] != "ATOM":
            continue
        cand = get(full_atoms, name=a["name"], chain=a["chain"], resi=a["resi"])
        if not cand:
            mism += 1
            continue
        dev = min(math.dist(a["xyz"], c["xyz"]) for c in cand)
        maxdev = max(maxdev, dev)
        if dev > 1e-3:
            mism += 1

    # 2. Zn coordination
    lig_atoms, lig_bonds = parse_sdf(lig_exp)
    dmin, amin = min((math.dist(zn["xyz"], (a[0], a[1], a[2])), i) for i, a in enumerate(lig_atoms))
    zn_elem = lig_atoms[amin][3]
    his = {}
    for r in (94, 96, 119):
        cands = [a for a in full_atoms if a["resn"] == "HIS" and a["chain"] == "A"
                 and a["resi"] == r and a["name"] in ("NE2", "ND1")]
        if cands:
            his[f"His{r}"] = round(min(math.dist(zn["xyz"], c["xyz"]) for c in cands), 3)

    # 3. experimental vs reconstructed ligand
    rec_atoms, rec_bonds = parse_sdf(lig_rec)
    n = len(lig_atoms)
    rmsd_same = math.sqrt(sum(
        math.dist(lig_atoms[i][:3], rec_atoms[i][:3]) ** 2 for i in range(n)) / n)
    nearest = []
    for la in lig_atoms:
        d, ra = min((math.dist(la[:3], ra[:3]), ra) for ra in rec_atoms)
        nearest.append((d, la[3], ra[3]))
    rmsd_near = math.sqrt(sum(d * d for d, _, _ in nearest) / n)
    elem_mismatch = sum(1 for _, e1, e2 in nearest if e1 != e2)

    # 4. experimental SDF vs deposited 1GD HETATM coordinates
    dep_dev, dep_elem = 0.0, 0
    for la in lig_atoms:
        same_elem = [a for a in deposited_lig if a["elem"].upper() == la[3].upper()]
        if not same_elem:
            dep_elem += 1
            continue
        dep_dev = max(dep_dev, min(math.dist(la[:3], a["xyz"]) for a in same_elem))

    return {
        "crystal_frame_identity": {
            "trp_a5_ne1_full_vs_pocket_ang": round(d_trp, 4),
            "pocket_residues_subset_of_deposition": pocket_subset,
            "pocket_atoms_with_name_resi_match": n_pocket_atom_records - mism,
            "pocket_atoms_total": n_pocket_atom_records,
            "pocket_max_coord_deviation_ang": round(maxdev, 5),
        },
        "zn_coordination": {
            "zn_xyz": [round(v, 3) for v in zn["xyz"]],
            "closest_ligand_atom": {"index": amin, "element": zn_elem,
                                    "distance_ang": round(dmin, 3)},
            "zn_to_histidine_dist_ang": his,
        },
        "ligand_experimental_vs_reconstructed": {
            "atom_counts": [len(lig_atoms), len(rec_atoms)],
            "bond_counts": [len(lig_bonds), len(rec_bonds)],
            "rmsd_same_order_ang": round(rmsd_same, 4),
            "rmsd_nearest_pair_ang": round(rmsd_near, 4),
            "element_mismatches_nearest_pair": elem_mismatch,
        },
        "ligand_sdf_vs_deposited_hetatm": {
            "sdf_atoms": len(lig_atoms), "deposited_hetatm_atoms": len(deposited_lig),
            "max_same_element_deviation_ang": round(dep_dev, 4),
            "atoms_without_same_element_counterpart": dep_elem,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-root",
                    default=str(pathlib.Path(__file__).resolve().parents[1]))
    ap.add_argument("--scenes", default="all",
                    help="comma list of scene keys, 'gif', or 'all' (default)")
    args = ap.parse_args()
    demo = pathlib.Path(args.demo_root)
    out_dir = demo / "protein_structure_3d"
    html = out_dir / "protein_viewer.html"
    assert html.exists(), "run protein_render_html.py first"

    want = None if args.scenes == "all" else set(args.scenes.split(","))

    frames_dir = pathlib.Path("/tmp/protein_gif_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = pathlib.Path("/tmp/preview")
    preview_dir.mkdir(parents=True, exist_ok=True)

    import time

    outputs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
        chromium_version = browser.version

        def new_page(w, h):
            pg = browser.new_page(viewport={"width": w + 40, "height": h + 40})
            pg.goto(f"file://{html}")
            pg.wait_for_timeout(400)
            return pg

        # ---- still scenes -------------------------------------------------
        for scene_key, file_stem, desc in SCENES:
            if want is not None and scene_key not in want:
                continue
            t0 = time.time()
            pg = new_page(STILL_W, STILL_H)
            pg.evaluate(f"window.setStageSize({STILL_W}, {STILL_H})")
            pg.evaluate("window.hideToolbar(true)")
            pg.evaluate(f"window.setScene('{scene_key}')")
            wait_ms = 3000 if scene_key == "surface" else 900
            pg.wait_for_timeout(wait_ms)
            png = out_dir / f"{file_stem}.png"
            pg.locator("#stage").screenshot(path=str(png))
            pg.close()

            ratio = png_nonwhite_ratio(png)
            pdf = out_dir / f"{file_stem}.pdf"
            png_to_pdf(png, pdf)
            # small preview for inspection
            im = Image.open(png)
            im.thumbnail((760, 570))
            im.convert("RGB").save(preview_dir / f"{file_stem}.jpg", quality=82)
            outputs.append({
                "file": png.name, "scene": scene_key, "description": desc,
                "canvas_px": [STILL_W, STILL_H], "dpi_equivalent": 300,
                "nonwhite_pixel_ratio": round(ratio, 4),
                "png_bytes": png.stat().st_size, "pdf_bytes": pdf.stat().st_size,
            })
            print(f"[still] {png.name}: nonwhite={ratio:.3f} total={time.time()-t0:.1f}s")

        # ---- rotation GIF -------------------------------------------------
        frames = []
        if want is None or "gif" in want:
            t0 = time.time()
            pg = new_page(GIF_W, GIF_H)
            pg.evaluate(f"window.setStageSize({GIF_W}, {GIF_H})")
            pg.evaluate("window.hideToolbar(true)")
            pg.evaluate("window.setScene('full_protein')")
            pg.wait_for_timeout(900)
            for i in range(GIF_FRAMES):
                pg.evaluate(f"window.getViewer().rotate({GIF_STEP_DEG}, 'y')")
                pg.evaluate("window.getViewer().render()")
                pg.wait_for_timeout(70)
                fp = frames_dir / f"f{i:02d}.png"
                pg.locator("#stage").screenshot(path=str(fp))
                frames.append(fp)
            pg.close()
        browser.close()

    if want is None or "gif" in want:
        imgs = [Image.open(f).convert("RGB") for f in frames]
        gif_path = out_dir / "protein_rotation.gif"
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=110, loop=0, optimize=True, disposal=2)
        first_ratio = png_nonwhite_ratio(frames[0])
        outputs.append({
            "file": gif_path.name, "scene": "full_protein (rotation)",
            "description": f"360-degree turntable rotation of the full-protein scene, "
                           f"{GIF_FRAMES} frames x {GIF_STEP_DEG} deg",
            "canvas_px": [GIF_W, GIF_H], "frames": GIF_FRAMES,
            "frame_duration_ms": 110,
            "nonwhite_pixel_ratio_first_frame": round(first_ratio, 4),
            "gif_bytes": gif_path.stat().st_size,
        })
        print(f"[gif] {gif_path.name}: {gif_path.stat().st_size} bytes "
              f"total={time.time()-t0:.1f}s")

    # ---- checks & provenance ---------------------------------------------
    checks = compute_checks(demo)

    # sanity assertions (fail closed)
    c1 = checks["crystal_frame_identity"]
    assert c1["trp_a5_ne1_full_vs_pocket_ang"] < 1e-3, "frame mismatch"
    assert c1["pocket_atoms_with_name_resi_match"] == c1["pocket_atoms_total"], "pocket atom mismatch"
    assert c1["pocket_max_coord_deviation_ang"] < 1e-3, "pocket coords differ from deposition"
    ov = checks["ligand_experimental_vs_reconstructed"]
    assert ov["rmsd_same_order_ang"] < 1e-3 and ov["element_mismatches_nearest_pair"] == 0
    dl = checks["ligand_sdf_vs_deposited_hetatm"]
    assert dl["max_same_element_deviation_ang"] < 1e-3, "SDF ligand != deposited coordinates"

    import hashlib
    def sha(p):
        return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

    # merge outputs across partial runs (scene-by-scene invocations)
    prov_path = out_dir / "render_provenance.json"
    prior = {}
    if prov_path.exists():
        try:
            prior = json.loads(prov_path.read_text())
        except Exception:
            prior = {}
    merged = {o["file"]: o for o in prior.get("outputs", [])}
    for o in outputs:
        merged[o["file"]] = o
    outputs = [merged[k] for k in sorted(merged)]
    chromium_version = prior.get("renderer", {}).get("browser", "")
    if "Chromium" not in chromium_version:
        chromium_version = (f"Chromium {chromium_version} via Playwright "
                            f"{importlib.metadata.version('playwright')}")

    provenance = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": ("Real experimental protein 3D render for the protein-to-molecule demo. "
                    "This is a reconstruction/visualization of the experimentally observed "
                    "complex, NOT an AI generation."),
        "renderer": {
            "engine": "3Dmol.js (WebGL)",
            "version": "2.4.0",
            "source": "https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js",
            "browser": chromium_version,
            "pdf_conversion": f"Pillow {importlib.metadata.version('pillow')} (raster PDF, 300 dpi)",
        },
        "sources": {
            "protein_full": {
                "file": "structures/raw/rcsb_4ITP_full.pdb.gz",
                "fetched_from": "https://files.rcsb.org/download/4ITP.pdb",
                "citation": "Guzel-Akdemir et al., Bioorg. Med. Chem. 21, 6674 (2013), "
                            "DOI 10.1016/j.bmc.2013.08.011",
                "method": "X-RAY DIFFRACTION", "resolution_angstrom": 1.70,
            },
            "pocket": "structures/protein.pdb (standardized demo pocket, 10 A sphere, 32 residues incl. ZN)",
            "ligand_experimental": "structures/raw/bindingmoad_4ITP_1GD_ligand.sdf.gz (BindingMOAD)",
            "ligand_reconstructed": "structures/final_ligand.sdf (2-step: rxn_01 then rxn_15)",
            "zinc": "deposited ZN A 302 coordinates",
        },
        "verification": checks,
        "outputs": outputs,
        "output_sha256": {o["file"]: sha(out_dir / o["file"]) for o in outputs},
    }
    (out_dir / "render_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(checks, indent=2))
    print("OK: all render + verification assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
