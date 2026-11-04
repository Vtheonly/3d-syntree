#!/usr/bin/env python3
"""Build a self-contained interactive 3D protein viewer (3Dmol.js/WebGL) for the
protein-to-molecule demo, using ONLY real experimental coordinates:

  - full protein: PDB 4ITP chain A (RCSB deposition, X-ray 1.70 A)
  - catalytic Zn2+: deposited ZN A 302
  - experimental ligand 1GD: BindingMOAD artifact (structures/raw/...ligand.sdf.gz)
  - reconstructed ligand: 2-step synthetic rebuild (structures/final_ligand.sdf)

All coordinates are in the deposited crystal frame (verified elsewhere to 0.000 A).

Usage: python3 protein_render_html.py --demo-root <demo dir> [--workdir <cache>]
Writes: <demo-root>/protein_structure_3d/protein_viewer.html
        <demo-root>/structures/raw/rcsb_4ITP_full.pdb.gz  (new source artifact)
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402

RCSB_4ITP_URL = "https://files.rcsb.org/download/4ITP.pdb"
THREEDMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"
THREEDMOL_VERSION = "2.4.0"

# pocket residue numbers (chain A) from the demo's standardized pocket artifact
POCKET_RESI = [5, 62, 64, 67, 91, 92, 94, 96, 119, 121, 122, 123, 131, 132,
               134, 135, 136, 141, 143, 197, 198, 199, 200, 201, 202, 203,
               204, 205, 206, 207, 209]
# residues to label in the active-site close-up (hCA II catalytic machinery)
LABEL_RESI = [5, 62, 64, 92, 94, 96, 119, 199, 209]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: pathlib.Path) -> str:
    return sha256_bytes(p.read_bytes())


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "3d-syntree-demo/2.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def load_4itp(demo_root: pathlib.Path, workdir: pathlib.Path) -> tuple[str, str]:
    """Return (full_pdb_text, sha256_of_downloaded_pdb). Caches the gz artifact."""
    raw_gz = demo_root / "structures" / "raw" / "rcsb_4ITP_full.pdb.gz"
    if raw_gz.exists():
        data = gzip.decompress(raw_gz.read_bytes())
        return data.decode("ascii"), sha256_bytes(data)
    cache = workdir / "4ITP.pdb"
    if not cache.exists():
        workdir.mkdir(parents=True, exist_ok=True)
        data = fetch(RCSB_4ITP_URL)
        cache.write_bytes(data)
    else:
        data = cache.read_bytes()
    text = data.decode("ascii")
    raw_gz.parent.mkdir(parents=True, exist_ok=True)
    raw_gz.write_bytes(gzip.compress(data, 9))
    return text, sha256_bytes(data)


def load_3dmol(workdir: pathlib.Path) -> str:
    cache = workdir / "3dmol.min.js"
    if not cache.exists():
        workdir.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(fetch(THREEDMOL_URL))
    js = cache.read_text()
    if "</script" in js.lower():
        js = js.replace("</script", "<\\/script")
    return js


def split_models(full_pdb_text: str) -> dict:
    """M0 protein: ATOM chain A first-conformer. M3: Zn coordinates verbatim."""
    protein_lines, zn_lines, dropped_altloc = [], [], 0
    for line in full_pdb_text.splitlines():
        if line.startswith("ATOM  "):
            altloc = line[16]
            if altloc not in (" ", "A"):
                dropped_altloc += 1
                continue
            if altloc == "A":
                line = line[:16] + " " + line[17:]
            protein_lines.append(line)
        elif line.startswith("HETATM") and line[17:20].strip() == "ZN":
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            zn_lines.append(
                "HETATM%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00 30.00          %2s"
                % (1, "ZN ", "ZN ", "D", 1, x, y, z, "ZN")
            )
    protein_lines.append("TER")
    protein_lines.append("END")
    stats = {
        "protein_atoms": len([l for l in protein_lines if l.startswith("ATOM  ")]),
        "dropped_altloc_atoms": dropped_altloc,
        "zn_atoms": len(zn_lines),
    }
    return {
        "protein_pdb": "\n".join(protein_lines) + "\n",
        "zn_pdb": "\n".join(zn_lines) + "\nEND\n",
        "stats": stats,
    }


def sdf_to_pdbHetatm(sdf_text: str, resn: str, chain: str, resi: int) -> str:
    """Convert an (heavy-atom) SDF block into PDB HETATM + CONECT records so the
    molecule is cleanly selectable by resname/chain in 3Dmol.js. Coordinates are
    copied verbatim."""
    lines = sdf_text.splitlines()
    n_at, n_bo = int(lines[3][0:3]), int(lines[3][3:6])
    atoms, conect = [], []
    counts = {}
    serial = 0
    for i in range(n_at):
        x, y, z = float(lines[4 + i][0:10]), float(lines[4 + i][10:20]), float(lines[4 + i][20:30])
        elem = lines[4 + i][31:34].strip().upper()
        counts[elem] = counts.get(elem, 0) + 1
        name = f"{elem}{counts[elem]}"
        name_field = f" {name:<3s}" if len(elem) == 1 else f"{name:<4s}"
        serial += 1
        atoms.append(
            "HETATM%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
            % (serial, name_field, resn, chain, resi, x, y, z, elem)
        )
    for i in range(n_bo):
        l = lines[4 + n_at + i]
        a, b = int(l[0:3]), int(l[3:6])
        conect.append("CONECT%5d%5d" % (a, b))
    conect.append("END")
    return "\n".join(atoms + conect) + "\n"


def residue_colors(protein_pdb: str) -> list:
    """Per-residue cartoon colors, turbo colormap over chain-A residue order."""
    resis, seen = [], set()
    for line in protein_pdb.splitlines():
        if line.startswith("ATOM  ") and line[21] == "A":
            n = int(line[22:26])
            if n not in seen:
                seen.add(n)
                resis.append(n)
    resis.sort()
    n = len(resis)
    out = []
    for i, r in enumerate(resis):
        c = cm.turbo(i / max(1, n - 1))
        out.append([r, "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))])
    return out


def residue_label_anchors(protein_pdb: str, lig_centroid, label_resi) -> list:
    """Label positions: CA atom + offset pointing away from the ligand centroid,
    so labels sit outside the pocket instead of on top of the molecule."""
    ca = {}
    resn = {}
    for line in protein_pdb.splitlines():
        if line.startswith("ATOM  ") and line[21] == "A":
            n = int(line[22:26])
            resn[n] = line[17:20].strip()
            if line[12:16].strip() == "CA":
                ca[n] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    labels = []
    for r in label_resi:
        if r not in ca:
            continue
        p = ca[r]
        d = [p[k] - lig_centroid[k] for k in range(3)]
        norm = max(1e-6, sum(v * v for v in d) ** 0.5)
        off = 5.0
        pos = [round(p[k] + d[k] / norm * off, 3) for k in range(3)]
        labels.append({"text": resn[r].title() + str(r), "pos": pos})
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-root",
                    default=str(pathlib.Path(__file__).resolve().parents[1]))
    ap.add_argument("--workdir", default="/tmp/protein_render_cache")
    args = ap.parse_args()
    demo = pathlib.Path(args.demo_root)
    workdir = pathlib.Path(args.workdir)

    full_pdb, pdb_sha = load_4itp(demo, workdir)
    models = split_models(full_pdb)

    lig_exp_sdf = gzip.decompress(
        (demo / "structures" / "raw" / "bindingmoad_4ITP_1GD_ligand.sdf.gz").read_bytes()
    ).decode()
    lig_recon_sdf = (demo / "structures" / "final_ligand.sdf").read_text()

    lig_exp_pdb = sdf_to_pdbHetatm(lig_exp_sdf, "1GD", "B", 1)
    lig_recon_pdb = sdf_to_pdbHetatm(lig_recon_sdf, "LRC", "C", 1)

    def centroid_of_sdf(txt):
        lines = txt.splitlines()
        n = int(lines[3][0:3])
        pts = [(float(lines[4 + i][0:10]), float(lines[4 + i][10:20]),
                float(lines[4 + i][20:30])) for i in range(n)]
        return [round(sum(p[k] for p in pts) / n, 3) for k in range(3)]

    lig_centroid = centroid_of_sdf(lig_exp_sdf)
    zn_line = [l for l in models["zn_pdb"].splitlines() if l.startswith("HETATM")][0]
    zn_pos = [float(zn_line[30:38]), float(zn_line[38:46]), float(zn_line[46:54])]

    moldata = {
        "protein_pdb": models["protein_pdb"],
        "zn_pdb": models["zn_pdb"],
        "lig_exp_pdb": lig_exp_pdb,
        "lig_recon_pdb": lig_recon_pdb,
        "pocket_resi": POCKET_RESI,
        "residue_colors": residue_colors(models["protein_pdb"]),
        "residue_labels": residue_label_anchors(models["protein_pdb"], lig_centroid, LABEL_RESI),
        "lig_centroid": lig_centroid,
        "zn_pos": [round(v, 3) for v in zn_pos],
        "citation": {
            "pdb_id": "4ITP",
            "title": ("STRUCTURE OF HUMAN CARBONIC ANHYDRASE II BOUND TO A BENZENE "
                      "SULFONAMIDE"),
            "method": "X-RAY DIFFRACTION",
            "resolution_angstrom": 1.70,
            "authors": "O.GUZEL-AKDEMIR,S.BISWAS,K.LASTRA,R.MCKENNA,C.T.SUPURAN",
            "journal": "Bioorg. Med. Chem. 21, 6674 (2013)",
            "doi": "10.1016/j.bmc.2013.08.011",
        },
    }

    threemol_js = load_3dmol(workdir)

    app_js = r"""
// ---- 3D-SynTree V2 protein-to-molecule demo: interactive protein viewer ----
const D = JSON.parse(document.getElementById('moldata').textContent);
if ($3Dmol.setSyncSurface) $3Dmol.setSyncSurface(true); // synchronous surface gen
const stage = document.getElementById('stage');
const gldiv = document.getElementById('gldiv');
let viewer = null;
let currentScene = null;
let spinOn = false;
let SCALE = 1.0; // typography scale, 1.0 at 2400 px canvas width

const CAPTIONS = {
  full_protein: {
    title: 'Human Carbonic Anhydrase II - full experimental structure',
    sub: [
      'PDB 4ITP chain A (261 residues, X-ray 1.70 A), cartoon colored by residue position',
      'gray sphere = catalytic Zn2+   |   green = ligand 1GD   |   thin steel-blue sticks = 32 demo pocket residues',
    ],
  },
  active_site: {
    title: 'Active-site close-up - 1GD coordinated to Zn2+',
    sub: [
      'pocket residues as sticks (standardized 10 A demo pocket) | His94 / His96 / His119 + sulfonamide N coordinate Zn2+',
      'N(sulfonamide)-Zn2+ = 1.99 A | all coordinates in the deposited crystal frame',
    ],
  },
  ligand_overlay: {
    title: 'Experimental vs reconstructed ligand pose - RMSD 0.000 A',
    sub: [
      'cyan sticks = experimental 1GD (BindingMOAD 4ITP:1GD)',
      'orange translucent spheres = 2-step synthetic reconstruction (rxn_01 + rxn_15), 21/21 atoms coincident',
    ],
  },
  surface: {
    title: '1GD bound in the active-site cleft',
    sub: [
      'translucent surface = van der Waals surface of chain A | sulfonamide head buried at Zn2+, tail toward the cavity mouth',
      'catalytic Zn2+ shown as gray sphere below the ligand',
    ],
  },
};
function labelFont() { return Math.max(11, Math.round(44 * SCALE)); }

function setStageSize(w, h) {
  stage.style.width = w + 'px';
  stage.style.height = h + 'px';
  SCALE = w / 2400.0;
  document.documentElement.style.setProperty('--s', SCALE);
}

function makeViewer() {
  gldiv.innerHTML = '';
  viewer = $3Dmol.createViewer(gldiv, { backgroundColor: 'white', antialias: true });
}

function resetViewer() {
  if (!viewer) makeViewer();
  viewer.clear();
  if (viewer.removeAllSurfaces) viewer.removeAllSurfaces();
  if (viewer.removeAllLabels) viewer.removeAllLabels();
  if (viewer.removeAllShapes) viewer.removeAllShapes();
}

function addCoreModels() {
  viewer.addModel(D.protein_pdb, 'pdb');   // M0 protein (chain A)
  viewer.addModel(D.lig_exp_pdb, 'pdb');   // M1 experimental 1GD (chain B)
  viewer.addModel(D.lig_recon_pdb, 'pdb'); // M2 reconstruction (chain C)
  viewer.addModel(D.zn_pdb, 'pdb');        // M3 zinc (chain D)
}

function stylePocketSticks(radius, color) {
  for (const resi of D.pocket_resi) {
    viewer.addStyle({ chain: 'A', resi: resi },
                    { stick: { radius: radius, color: color } });
  }
}

function znSphere(radius) {
  viewer.addStyle({ resn: 'ZN' }, { sphere: { radius: radius, color: '0x5f5f5f' } });
}

function ligandSticks(colorScheme, radius) {
  viewer.addStyle({ resn: '1GD' }, { stick: { radius: radius, colorscheme: colorScheme } });
}

function ligandLabel() {
  viewer.addLabel('1GD (ligand)', {
    position: { x: D.lig_centroid[0], y: D.lig_centroid[1] + 3.4, z: D.lig_centroid[2] },
    fontSize: labelFont(), fontColor: 'black', bold: true,
    showBackground: true, backgroundColor: 'white',
    borderThickness: 1, borderColor: '0x555555', inFront: true,
  });
}

function znLabel(off) {
  off = (off === undefined) ? 2.8 : off;
  const dx = D.zn_pos[0] - D.lig_centroid[0], dy = D.zn_pos[1] - D.lig_centroid[1],
        dz = D.zn_pos[2] - D.lig_centroid[2];
  const n = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
  viewer.addLabel('Zn2+', {
    position: { x: D.zn_pos[0] + dx / n * off,
                y: D.zn_pos[1] + dy / n * off,
                z: D.zn_pos[2] + dz / n * off },
    fontSize: labelFont(), fontColor: 'black', bold: true,
    showBackground: true, backgroundColor: 'white',
    borderThickness: 1, borderColor: '0x555555', inFront: true,
  });
}

function setCaption(key) {
  const cap = CAPTIONS[key];
  document.getElementById('cap-title').textContent = cap.title;
  const sub = document.getElementById('cap-sub');
  sub.innerHTML = '';
  for (const line of cap.sub) {
    const d = document.createElement('div');
    d.textContent = line;
    sub.appendChild(d);
  }
}

function sceneFull() {
  resetViewer();
  addCoreModels();
  viewer.setStyle({ chain: 'A' }, { cartoon: {} });
  for (const [resi, col] of D.residue_colors) {
    viewer.addStyle({ chain: 'A', resi: resi }, { cartoon: { color: col } });
  }
  stylePocketSticks(0.06, '0x6b7f94');
  ligandSticks('greenCarbon', 0.24);
  viewer.addStyle({ resn: '1GD' }, { sphere: { scale: 0.26, colorscheme: 'greenCarbon' } });
  znSphere(1.5);
  viewer.zoomTo();
  viewer.rotate(-75, 'y');
  viewer.rotate(-10, 'x');
  viewer.render();
}

function sceneActiveSite() {
  resetViewer();
  addCoreModels();
  viewer.setStyle({ chain: 'A' }, { cartoon: { color: '0xd9d9d9', thickness: 0.6 } });
  stylePocketSticks(0.16, undefined);
  ligandSticks('greenCarbon', 0.26);
  viewer.addStyle({ resn: '1GD' }, { sphere: { scale: 0.22, colorscheme: 'greenCarbon' } });
  znSphere(1.45);
  for (const lab of D.residue_labels) {
    viewer.addLabel(lab.text, {
      position: { x: lab.pos[0], y: lab.pos[1], z: lab.pos[2] },
      fontSize: Math.max(11, Math.round(30 * SCALE)),
      fontColor: 'black', bold: true, showBackground: true,
      backgroundColor: 'white', borderThickness: 1, borderColor: '0x777777',
      inFront: true,
    });
  }
  ligandLabel();
  znLabel();
  viewer.zoomTo({ resn: '1GD' });
  viewer.zoom(0.72);
  viewer.rotate(-20, 'y');
  viewer.rotate(-10, 'x');
  viewer.render();
}

function sceneOverlay() {
  resetViewer();
  addCoreModels();
  viewer.setStyle({ chain: 'A' }, { cartoon: { color: '0xeeeeee', thickness: 0.5 } });
  stylePocketSticks(0.09, '0xb0b0b0');
  viewer.addStyle({ resn: '1GD' }, { stick: { radius: 0.24, colorscheme: 'cyanCarbon' } });
  viewer.addStyle({ resn: 'LRC' },
                  { sphere: { radius: 1.5, color: '0xff8c00', opacity: 0.42 } });
  znSphere(1.35);
  ligandLabel();
  znLabel(1.2);
  viewer.zoomTo({ resn: '1GD' });
  viewer.zoom(0.82);
  viewer.rotate(-25, 'y');
  viewer.rotate(-12, 'x');
  viewer.render();
}

function sceneSurface() {
  resetViewer();
  addCoreModels();
  viewer.setStyle({ chain: 'A' }, { cartoon: { color: '0xf0f0f0', thickness: 0.5 } });
  stylePocketSticks(0.08, '0xb8b8b8');
  ligandSticks('greenCarbon', 0.2);
  viewer.addStyle({ resn: '1GD' }, { sphere: { scale: 0.62, colorscheme: 'greenCarbon' } });
  znSphere(1.3);
  viewer.addSurface($3Dmol.SurfaceType.VDW,
                   { opacity: 0.75, color: '0xb0c4d8' }, { chain: 'A' });
  ligandLabel();
  znLabel(1.2);
  viewer.zoomTo({ resn: '1GD' });
  viewer.zoom(0.8);
  viewer.rotate(-25, 'y');
  viewer.rotate(-12, 'x');
  viewer.render();
}

const SCENES = {
  full_protein: sceneFull,
  active_site: sceneActiveSite,
  ligand_overlay: sceneOverlay,
  surface: sceneSurface,
};

window.setScene = function (key) {
  currentScene = key;
  setCaption(key);
  document.querySelectorAll('#toolbar button').forEach(b => {
    b.classList.toggle('active', b.dataset.scene === key);
  });
  SCENES[key]();
};

window.getViewer = function () { return viewer; };
window.setStageSize = setStageSize;
window.hideToolbar = function (hide) {
  document.getElementById('toolbar').style.display = hide ? 'none' : 'flex';
};

document.querySelectorAll('#toolbar button[data-scene]').forEach(b => {
  b.addEventListener('click', () => window.setScene(b.dataset.scene));
});
document.getElementById('btn-spin').addEventListener('click', () => {
  spinOn = !spinOn;
  if (viewer) viewer.spin(spinOn ? 'y' : false);
  document.getElementById('btn-spin').classList.toggle('active', spinOn);
});

function fitToWindow() {
  const w = Math.floor(window.innerWidth * 0.999);
  const h = Math.floor(window.innerHeight * 0.999);
  setStageSize(w, h);
}
fitToWindow();
window.setScene('full_protein');
"""

    html_doc = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>3D-SynTree V2 - protein viewer (PDB 4ITP / hCA II / ligand 1GD)</title>
<style>
  :root { --s: 1.0; }
  html, body { margin: 0; padding: 0; background: #ffffff; overflow: hidden;
               font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
  #stage { position: relative; width: 100vw; height: 100vh; }
  #gldiv { position: absolute; inset: 0; }
  #caption { position: absolute; top: calc(18px * var(--s)); left: calc(22px * var(--s));
             max-width: 74%; background: rgba(255,255,255,0.90);
             padding: calc(14px * var(--s)) calc(18px * var(--s));
             border-left: calc(6px * var(--s)) solid #1a6b1a; }
  #cap-title { margin: 0 0 calc(8px * var(--s)) 0; font-size: calc(46px * var(--s));
               font-weight: 700; color: #173a17; line-height: 1.15; }
  #cap-sub div { font-size: calc(26px * var(--s)); color: #333; margin: calc(3px * var(--s)) 0; }
  #credit { position: absolute; bottom: calc(12px * var(--s)); right: calc(16px * var(--s));
            text-align: right; font-size: calc(19px * var(--s)); color: #555;
            background: rgba(255,255,255,0.87); padding: calc(6px * var(--s)) calc(10px * var(--s)); }
  #toolbar { position: absolute; bottom: calc(14px * var(--s)); left: 50%;
             transform: translateX(-50%); display: flex; gap: calc(8px * var(--s)); }
  #toolbar button { font-size: calc(17px * var(--s)); padding: calc(8px * var(--s)) calc(14px * var(--s));
                    cursor: pointer; border: calc(1px * var(--s)) solid #1a6b1a; background: #fff;
                    color: #1a6b1a; border-radius: 4px; }
  #toolbar button.active { background: #1a6b1a; color: #fff; }
</style>
</head>
<body>
<div id="stage">
  <div id="gldiv"></div>
  <div id="caption">
    <h1 id="cap-title"></h1>
    <div id="cap-sub"></div>
  </div>
  <div id="toolbar">
    <button data-scene="full_protein">Full protein</button>
    <button data-scene="active_site">Active site</button>
    <button data-scene="ligand_overlay">Ligand overlay</button>
    <button data-scene="surface">Surface</button>
    <button id="btn-spin">Spin</button>
  </div>
  <div id="credit">
    <div>Rendered with 3Dmol.js """ + THREEDMOL_VERSION + """ (WebGL) - all coordinates in the deposited crystal frame of PDB 4ITP</div>
    <div>Guzel-Akdemir et al., <i>Bioorg. Med. Chem.</i> 21, 6674 (2013), DOI 10.1016/j.bmc.2013.08.011 - X-ray, 1.70 A</div>
  </div>
</div>
<script type="application/json" id="moldata">""" + json.dumps(moldata) + """</script>
<script>""" + threemol_js + """</script>
<script>""" + app_js + """</script>
</body>
</html>
"""

    out_dir = demo / "protein_structure_3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "protein_viewer.html"
    out.write_text(html_doc)

    manifest = {
        "html": str(out),
        "html_sha256": sha256_file(out),
        "rcsb_4itp_sha256": pdb_sha,
        "threemol_version": THREEDMOL_VERSION,
        "threemol_source": THREEDMOL_URL,
        "threemol_sha256": sha256_bytes(threemol_js.encode()),
        "model_split_stats": models["stats"],
        "residue_label_count": len(moldata["residue_labels"]),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
