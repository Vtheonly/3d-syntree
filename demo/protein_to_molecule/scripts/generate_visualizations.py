#!/usr/bin/env python3
"""Generate the 2D/3D demonstration figures for the extracted
protein-to-molecule construction example (bindingmoad_4ITP_1GD).

RDKit handles molecular graphs, reaction templates, 2D coordinates and
conformers; Matplotlib renders every figure.  No structure is drawn by
hand and no coordinate is fabricated:

* seed / intermediates / products / pocket ........ experimental X-ray
  coordinates (PDB 4ITP crystal frame, induced subgraphs);
* incoming building blocks ......................... deterministic RDKit
  ETKDG conformers (seed 2026), ALWAYS rendered outside the pocket frame
  and explicitly labelled as computationally generated.

Run AFTER scripts/replay_trajectory.py (which writes structures/ and
verification/).  Outputs reactions/step_XX/*.{png,pdf} plus the two
composite figures (300 DPI PNG + vector PDF).
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point2D

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
SDIR = os.path.join(DEMO, "structures")
RDIR = os.path.join(DEMO, "reactions")

ELEMENT_COLOR = {
    "C": "#4d4d4d", "N": "#2f5df8", "O": "#f83232", "S": "#f0d020",
    "Cl": "#33d633", "Br": "#a62929", "Zn": "#9d79a9", "F": "#7de57d",
    "H": "#e8e8e8",
}
ELEMENT_RADIUS = {"C": 0.77, "N": 0.75, "O": 0.73, "S": 1.02, "Cl": 0.99,
                  "Br": 1.14, "Zn": 1.22}
HIGHLIGHT = "#ff3355"       # reaction centre
NEW_BOND = "#e0115f"        # newly formed bond
LEAVING = "#ff8c00"         # leaving-group atoms (dashed ring)
EXPERIMENTAL_NOTE = "experimental X-ray coordinates (PDB 4ITP)"
COMPUTED_NOTE = "computational conformer (RDKit ETKDG, seed 2026) — no experimental position"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.6, "figure.dpi": 110,
})


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def load_mol(path: str) -> Chem.Mol:
    m = Chem.MolFromMolFile(path, removeHs=False, sanitize=True)
    if m is None:
        raise SystemExit(f"cannot parse {path}")
    return m


def coords_2d(mol: Chem.Mol, coord_map=None) -> np.ndarray:
    """2D coordinates on a COPY (never destroys the 3D conformer)."""
    m2 = Chem.Mol(mol)
    if coord_map:
        AllChem.Compute2DCoords(m2, coordMap=coord_map)
    else:
        AllChem.Compute2DCoords(m2)
    return m2.GetConformer().GetPositions()[:, :2]


def draw_mol_2d(ax, mol, xy, title="", subtitle="",
                highlight_atoms=(), highlight_bonds=(), leaving_atoms=(),
                bond_w=1.6, scale=1.0, label_ids=()):
    """Skeletal 2D rendering of a molecule on a matplotlib axis."""
    highlight_atoms, highlight_bonds = set(highlight_atoms), set(highlight_bonds)
    leaving_atoms = set(leaving_atoms)
    # bonds
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        x1, y1 = xy[i]; x2, y2 = xy[j]
        is_new = (i, j) in highlight_bonds or (j, i) in highlight_bonds
        color = NEW_BOND if is_new else "#222222"
        lw = (bond_w * 2.2 if is_new else bond_w)
        if b.GetBondType() == Chem.BondType.DOUBLE:
            dx, dy = x2 - x1, y2 - y1
            n = math.hypot(dx, dy) or 1.0
            ox, oy = -dy / n * 0.09, dx / n * 0.09
            ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], color=color, lw=lw,
                    zorder=2, solid_capstyle="round")
            ax.plot([x1 - ox, x2 - ox], [y1 - oy, y2 - oy], color=color, lw=lw,
                    zorder=2, solid_capstyle="round")
        else:
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2,
                    solid_capstyle="round")
    # atoms
    for a in mol.GetAtoms():
        i = a.GetIdx()
        sym = a.GetSymbol()
        x, y = xy[i]
        nh = sum(1 for _ in a.GetNeighbors())
        if sym == "C":
            ax.scatter([x], [y], s=14 * scale, color=ELEMENT_COLOR["C"], zorder=3)
        else:
            n_h = a.GetTotalNumHs()
            label = sym + (str(n_h) if (sym in ("N", "O", "S") and n_h) else "")
            ax.scatter([x], [y], s=250 * scale, color=ELEMENT_COLOR.get(sym, "gray"),
                       zorder=4, edgecolors="white", linewidths=0.8)
            ax.text(x, y, label, color="white", fontsize=7.5 * scale, ha="center",
                    va="center", zorder=5, fontweight="bold")
        if i in highlight_atoms:
            ax.scatter([x], [y], s=460 * scale, facecolors="none",
                       edgecolors=HIGHLIGHT, linewidths=2.0, zorder=6)
        if i in leaving_atoms:
            ax.scatter([x], [y], s=460 * scale, facecolors="none",
                       edgecolors=LEAVING, linewidths=2.0, zorder=6, linestyle="--")
        if i in label_ids:
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(8, 8), fontsize=6.5, color="#555555", zorder=7)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=14 if subtitle else 6)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.2, color="#35507a")
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")


def draw_mol_3d(ax, mol, conformer, title="", subtitle="",
                highlight_atoms=(), highlight_bonds=(), leaving_atoms=(),
                atom_scale=1.0, show_h=False):
    highlight_atoms, highlight_bonds = set(highlight_atoms), set(highlight_bonds)
    leaving_atoms = set(leaving_atoms)
    pos = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        is_new = (i, j) in highlight_bonds or (j, i) in highlight_bonds
        ax.plot(*zip(pos[i], pos[j]),
                color=NEW_BOND if is_new else "#333333",
                lw=3.4 if is_new else 1.4, zorder=3)
    for a in mol.GetAtoms():
        i = a.GetIdx()
        sym = a.GetSymbol()
        if sym == "H" and not show_h:
            continue
        r = ELEMENT_RADIUS.get(sym, 0.8)
        hot = i in highlight_atoms
        edge = HIGHLIGHT if hot else (LEAVING if i in leaving_atoms else "black")
        lw = 2.2 if (hot or i in leaving_atoms) else 0.5
        ax.scatter(*pos[i], s=(r * 210) * atom_scale,
                   color=ELEMENT_COLOR.get(sym, "gray"),
                   edgecolors=edge, linewidths=lw, zorder=4, alpha=0.95)
    if title:
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=2)
    if subtitle:
        ax.text2D(0.5, 0.02, subtitle, transform=ax.transAxes, ha="center",
                  fontsize=7.2, color="#35507a")
    ax.set_axis_off()
    _equal_box(ax, pos)


def _equal_box(ax, points, pad=1.5):
    points = np.asarray(points)
    mins, maxs = points.min(0), points.max(0)
    ctr, span = (mins + maxs) / 2, np.maximum((maxs - mins) / 2, 1e-6) + pad
    ax.set_xlim(ctr[0] - span[0], ctr[0] + span[0])
    ax.set_ylim(ctr[1] - span[1], ctr[1] + span[1])
    ax.set_zlim(ctr[2] - span[2], ctr[2] + span[2])
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def parse_pocket(path: str):
    atoms = []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:26].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            elem = name[0] if not name[0].isdigit() else name[1]
            if name.upper().startswith("ZN"):
                elem = "ZN"
            atoms.append({
                "name": name, "resname": resname, "chain": chain,
                "resseq": resseq, "pos": (x, y, z), "elem": elem.upper(),
                "het": line.startswith("HETATM"),
            })
    return atoms


def draw_pocket_3d(ax, pocket_atoms, ligand, lig_conf, title="", subtitle="",
                   highlight_atoms=(), highlight_bonds=(), annotate_contacts=True):
    # render strictly by zorder (matplotlib's 3D depth sort would otherwise
    # bury the annotation layer under nearby atom spheres)
    try:
        ax.computed_zorder = False
    except AttributeError:
        pass
    protein = [a for a in pocket_atoms if a["elem"] != "ZN"]
    zinc = [a for a in pocket_atoms if a["elem"] == "ZN"]
    pp = np.array([a["pos"] for a in protein])
    ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], s=20, color="#9fb3c8", alpha=0.6,
               edgecolors="none", zorder=2, label="protein atoms")
    lpos = np.array([list(lig_conf.GetAtomPosition(i)) for i in range(ligand.GetNumAtoms())])
    lig_ctr = lpos.mean(0)
    contacts = {}
    if annotate_contacts:
        for a in protein:
            d = np.linalg.norm(lpos - np.array(a["pos"]), axis=1).min()
            if d < 4.0:
                key = (a["chain"], a["resseq"], a["resname"])
                contacts[key] = min(contacts.get(key, 9e9), d)
    # label only the 4 nearest distinct residues, offset radially outward
    for (chain, resseq, resname), d in sorted(contacts.items(), key=lambda kv: kv[1])[:5]:
        sel = [a for a in protein if (a["chain"], a["resseq"], a["resname"]) == (chain, resseq, resname)]
        cp = np.array([a["pos"] for a in sel]).mean(0)
        direction = cp - lig_ctr
        n = np.linalg.norm(direction) or 1.0
        lp = cp + direction / n * 1.8
        ax.plot(*zip(cp, lp), color="#3a5a40", lw=0.7, alpha=0.55, zorder=3)
        ax.text(lp[0], lp[1], lp[2], f"{resname}{resseq}", fontsize=7.0,
                color="#3a5a40", zorder=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    for z in zinc:
        ax.scatter(*z["pos"], s=330, color=ELEMENT_COLOR["Zn"], edgecolors="black",
                   linewidths=1.0, zorder=5)
        ax.text(z["pos"][0], z["pos"][1], z["pos"][2] + 1.0, "Zn$^{2+}$",
                fontsize=8.5, ha="center", fontweight="bold", color="#6a3d9a",
                zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                    ec="none", alpha=0.9))
        # Zn - sulfonamide N coordination: dashed line + labelled distance
        for i, a in enumerate(ligand.GetAtoms()):
            if a.GetSymbol() == "N" and a.GetIdx() in highlight_atoms:
                dist = math.dist(z["pos"], lpos[i])
                zp, np_ = np.array(z["pos"]), lpos[i]
                ax.plot(*zip(zp, np_), ls="--", lw=2.2, color="#d81b60", zorder=6)
                mid = (zp + np_) / 2 + np.array([0.6, 0.6, 1.7])
                ax.text(mid[0], mid[1], mid[2], f"{dist:.2f} " + chr(0xC5),
                        fontsize=7.5, color="#b3005e", fontweight="bold", zorder=8,
                        ha="center", bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                               ec="none", alpha=0.9))
    for b in ligand.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        is_new = (i, j) in set(highlight_bonds) or (j, i) in set(highlight_bonds)
        ax.plot(*zip(lpos[i], lpos[j]), color=NEW_BOND if is_new else "#111111",
                lw=3.6 if is_new else 2.0, zorder=6)
    for i, a in enumerate(ligand.GetAtoms()):
        sym = a.GetSymbol()
        r = ELEMENT_RADIUS.get(sym, 0.8)
        hot = i in set(highlight_atoms)
        ax.scatter(*lpos[i], s=r * 150, color=ELEMENT_COLOR.get(sym, "gray"),
                   edgecolors=HIGHLIGHT if hot else "black",
                   linewidths=2.0 if hot else 0.5, zorder=5, alpha=0.96)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=2)
    if subtitle:
        ax.text2D(0.5, 0.02, subtitle, transform=ax.transAxes, ha="center",
                  fontsize=7.2, color="#35507a")
    ax.set_axis_off()
    # span-proportional box aspect (fills the canvas; no dead cube padding)
    allp = np.vstack([pp, lpos] + [[z["pos"]] for z in zinc])
    mins, maxs = allp.min(0), allp.max(0)
    ctr = (mins + maxs) / 2
    spans = np.maximum(maxs - mins, 1e-6) + 2.0
    ax.set_xlim(ctr[0] - spans[0] / 2, ctr[0] + spans[0] / 2)
    ax.set_ylim(ctr[1] - spans[1] / 2, ctr[1] + spans[1] / 2)
    ax.set_zlim(ctr[2] - spans[2] / 2, ctr[2] + spans[2] / 2)
    rel = spans / spans.max()
    try:
        ax.set_box_aspect(tuple(rel))
    except AttributeError:
        pass


def save(fig, path_noext):
    fig.savefig(path_noext + ".png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(path_noext + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path_noext + ".{png,pdf}")


def find_leaving(block: Chem.Mol, prod: Chem.Mol):
    """Minimal set of block atoms whose deletion lets the remainder
    substructure-match into the product = the block-side leaving atoms."""
    from itertools import combinations

    n = block.GetNumAtoms()
    for size in range(1, n):
        for subset in combinations(range(n), size):
            rw = Chem.RWMol(block)
            for i in sorted(subset, reverse=True):
                rw.RemoveAtom(i)
            core = rw.GetMol()
            try:
                Chem.SanitizeMol(core)
            except Exception:
                continue
            if prod.HasSubstructMatch(core):
                return list(subset)
    return []


def arrow_axes(fig, ax_id, label, sub=""):
    ax = fig.add_subplot(ax_id)
    ax.set_axis_off()
    ax.annotate("", xy=(0.94, 0.62), xytext=(0.06, 0.62),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#222222"))
    ax.text(0.5, 0.70, label, ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#8b1a1a")
    if sub:
        import textwrap
        wrapped = "\n".join(textwrap.wrap(sub, width=16))
        ax.text(0.5, 0.52, wrapped, ha="center", va="top", fontsize=6.4,
                color="#444444", wrap=True)
    return ax


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> int:
    traj = json.load(open(os.path.join(DEMO, "trajectory.json")))
    ver = json.load(open(os.path.join(DEMO, "verification.json")))
    steps = traj["steps"]
    geo = {g["order"]: g for g in ver["geometry"]}
    srep = {s["order"]: s for s in ver["steps"]}

    from syntree.chemistry.reactions import CERTIFIED_REACTIONS

    # ---- load structures (written by replay_trajectory.py) --------------
    final = load_mol(os.path.join(SDIR, "final_ligand.sdf"))
    seed = load_mol(os.path.join(SDIR, "seed.sdf"))
    mols = {
        "seed": seed,
        "step_01": {
            "react": load_mol(os.path.join(SDIR, "step_01", "reactant_intermediate.sdf")),
            "block": load_mol(os.path.join(SDIR, "step_01", "incoming_building_block.sdf")),
            "prod": load_mol(os.path.join(SDIR, "step_01", "product.sdf")),
        },
        "step_02": {
            "react": load_mol(os.path.join(SDIR, "step_02", "reactant_intermediate.sdf")),
            "block": load_mol(os.path.join(SDIR, "step_02", "incoming_building_block.sdf")),
            "prod": load_mol(os.path.join(SDIR, "step_02", "product.sdf")),
        },
        "final": final,
    }
    pocket = parse_pocket(os.path.join(SDIR, "protein.pdb"))

    # ---- consistent 2D layouts: final as reference, intermediates aligned
    xy_final = coords_2d(final)
    def aligned_2d(mol):
        m = final.GetSubstructMatches(mol)
        cmap = {a.GetIdx(): Point2D(*xy_final[img]) for a, img in
                ((a, m[0][a.GetIdx()]) for a in mol.GetAtoms())}
        return coords_2d(mol, coord_map=cmap)
    xy_seed = aligned_2d(seed)
    xy = {
        "seed": xy_seed,
        "step_01": {
            "react": xy_seed,
            "block": coords_2d(mols["step_01"]["block"]),
            "prod": aligned_2d(mols["step_01"]["prod"]),
        },
        "step_02": {
            "react": aligned_2d(mols["step_02"]["react"]),
            "block": coords_2d(mols["step_02"]["block"]),
            "prod": xy_final,
        },
    }

    # ---- reaction-centre bookkeeping (all via crystal-ligand indices) ---
    # the crystal ligand (atom order == final_ligand.sdf order: 21 heavy atoms)
    raw_lig_gz = os.path.join(SDIR, "raw", f"{traj['complex_id']}_ligand.sdf.gz")
    import gzip as _gzip

    lig = Chem.MolFromMolBlock(
        _gzip.open(raw_lig_gz, "rt").read(), removeHs=False, sanitize=True
    )

    def crystal_images(mol: Chem.Mol):
        """mol (SDF, induced-submol ascending order) -> crystal atom indices."""
        matches = lig.GetSubstructMatches(mol)
        if len(matches) != 1:
            raise SystemExit(f"non-unique crystal mapping for {mol.GetProp('_Name') if mol.HasProp('_Name') else 'mol'}: {len(matches)} matches")
        return matches[0]

    centres = {}
    prev_img = crystal_images(seed)
    for skey, order in (("step_01", 1), ("step_02", 2)):
        g = geo[order]
        inter, block, prod = mols[skey]["react"], mols[skey]["block"], mols[skey]["prod"]
        prod_img = crystal_images(prod)          # prod atom idx -> crystal idx
        img2prod = {c: i for i, c in enumerate(prod_img)}
        junction_crystal = g["junction_bond_crystal_indices"]
        jb = [img2prod[junction_crystal[0]], img2prod[junction_crystal[1]]]
        # handle atom: the reactant atom whose crystal image gains a neighbour
        # present in the product but absent from the reactant
        prod_atoms = set(prod_img)
        inter_img = crystal_images(inter)
        handle = None
        for i, c in enumerate(inter_img):
            for nb in lig.GetAtomWithIdx(c).GetNeighbors():
                nid = nb.GetIdx()
                if nid not in set(inter_img) and nid in prod_atoms:
                    handle = i
                    break
            if handle is not None:
                break
        leaving = find_leaving(block, prod)
        centres[skey] = {"handle": handle, "junction_bond": jb, "leaving": leaving}
        prev_img = prod_img

    # ==================================================================
    # per-step figures
    # ==================================================================
    for skey in ("step_01", "step_02"):
        order = 1 if skey == "step_01" else 2
        step = steps[order - 1]
        rx = CERTIFIED_REACTIONS[step["reaction_id"]]
        c = centres[skey]
        rdir = os.path.join(RDIR, skey)
        os.makedirs(rdir, exist_ok=True)
        inter, block, prod = mols[skey]["react"], mols[skey]["block"], mols[skey]["prod"]
        x_inter, x_block, x_prod = xy[skey]["react"], xy[skey]["block"], xy[skey]["prod"]

        # ---------------- (1) reaction_equation --------------------------
        fig = plt.figure(figsize=(13.6, 4.6))
        gs = fig.add_gridspec(1, 6, width_ratios=[2.2, 0.35, 1.3, 1.1, 0.35, 2.6])
        ax1 = fig.add_subplot(gs[0, 0])
        draw_mol_2d(ax1, inter, x_inter,
                    title=f"Current intermediate  (M{order - 1})",
                    subtitle=step["molecule_reactant_smiles"],
                    highlight_atoms=[c["handle"]] if c["handle"] is not None else [])
        axp = fig.add_subplot(gs[0, 1]); axp.set_axis_off()
        axp.text(0.5, 0.5, "+", ha="center", va="center", fontsize=17, fontweight="bold")
        ax2 = fig.add_subplot(gs[0, 2])
        draw_mol_2d(ax2, block, x_block,
                    title="Incoming building block",
                    subtitle=f"{step['block_id']} · {step['block_reactant_smiles']}",
                    leaving_atoms=c["leaving"])
        ax3 = arrow_axes(fig, gs[0, 3], step["reaction_id"], rx.name)
        ax4 = fig.add_subplot(gs[0, 5])
        draw_mol_2d(ax4, prod, x_prod,
                    title=f"Resulting intermediate  (M{order})",
                    subtitle=step["product_smiles"],
                    highlight_atoms=c["junction_bond"] or [],
                    highlight_bonds=[tuple(c["junction_bond"])] if c["junction_bond"] else [])
        fig.suptitle(
            f"Step {order} — actual certified reaction equation · "
            f"{rx.name} · family: {rx.family}",
            fontsize=11.5, fontweight="bold", y=1.00)
        note = (
            f"certified SMARTS: {rx.reaction_smarts}\n"
            f"new bond formed: {rx.product_bond_type} (junction); "
            f"leaving (block side): {rx.leaving_group_block}; "
            f"leaving (molecule side): {rx.leaving_group_molecule}\n"
            f"catalyst/conditions: "
            + ("Pd/Ni-catalysed per the registry mechanism note — not modelled by the template (unspecified)"
               if "catalysed" in rx.reference else "not specified by the certified template")
            + f"\nmechanism: {rx.reference}"
        )
        fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=7.4,
                 color="#333333", family="DejaVu Sans",
                 bbox=dict(boxstyle="round,pad=0.45", fc="#f4f6f8", ec="#b9c4cf"))
        save(fig, os.path.join(rdir, "reaction_equation"))

        # ---------------- (2) molecular_structures_2d --------------------
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
        draw_mol_2d(axes[0], inter, x_inter,
                    title=f"A · Before: M{order - 1} (crystal coordinates)",
                    subtitle=EXPERIMENTAL_NOTE,
                    highlight_atoms=[c["handle"]] if c["handle"] is not None else [])
        draw_mol_2d(axes[1], block, x_block,
                    title="B · Incoming building block",
                    subtitle=f"{step['block_id']} · {COMPUTED_NOTE}",
                    leaving_atoms=c["leaving"])
        draw_mol_2d(axes[2], prod, x_prod,
                    title=f"C · After: M{order} (crystal coordinates)",
                    subtitle=EXPERIMENTAL_NOTE,
                    highlight_atoms=c["junction_bond"] or [],
                    highlight_bonds=[tuple(c["junction_bond"])] if c["junction_bond"] else [])
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=HIGHLIGHT, markeredgewidth=2, markersize=11,
                   label="reaction centre (participating atoms)"),
            Line2D([0], [0], color=NEW_BOND, lw=3,
                   label="newly formed bond"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=LEAVING, markeredgewidth=2, linestyle="--",
                   markersize=11, label="leaving group (deleted)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"Step {order} — {rx.reaction_id}: {rx.name}  "
            f"(junction dihedral {step['dihedral_rad']:.4f} rad → torsion bin {step['torsion_bin']}/36)",
            fontsize=11, fontweight="bold", y=0.99)
        save(fig, os.path.join(rdir, "molecular_structures_2d"))

        # ---------------- (3) molecular_structures_3d --------------------
        fig = plt.figure(figsize=(13.6, 4.9))
        for k in range(3):
            fig.add_subplot(1, 3, k + 1, projection="3d")
        a1, a2, a3 = fig.axes
        draw_mol_3d(a1, inter, inter.GetConformer(),
                    title=f"A · Before: M{order - 1}",
                    subtitle=EXPERIMENTAL_NOTE,
                    highlight_atoms=[c["handle"]] if c["handle"] is not None else [])
        draw_mol_3d(a2, block, block.GetConformer(),
                    title="B · Incoming building block",
                    subtitle=COMPUTED_NOTE,
                    highlight_atoms=c["leaving"])
        draw_mol_3d(a3, prod, prod.GetConformer(),
                    title=f"C · After: M{order}",
                    subtitle=EXPERIMENTAL_NOTE,
                    highlight_atoms=c["junction_bond"] or [],
                    highlight_bonds=[tuple(c["junction_bond"])] if c["junction_bond"] else [])
        fig.suptitle(
            f"Step {order} — 3D molecular structures · {rx.reaction_id} ({rx.name})",
            fontsize=11, fontweight="bold", y=1.0)
        save(fig, os.path.join(rdir, "molecular_structures_3d"))

        # ---------------- (4) protein_pocket_3d --------------------------
        fig = plt.figure(figsize=(11.2, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=14, azim=-58)
        sulfonamide_n = [a.GetIdx() for a in prod.GetAtoms()
                         if a.GetSymbol() == "N"
                         and any(n.GetSymbol() == "S" for n in a.GetNeighbors())]
        draw_pocket_3d(
            ax, pocket, prod, prod.GetConformer(),
            title=f"Step {order} · M{order} inside the binding pocket — human carbonic anhydrase II (PDB 4ITP)",
            subtitle=EXPERIMENTAL_NOTE + " · pocket: 146 protein atoms + catalytic Zn$^{2+}$",
            highlight_atoms=sulfonamide_n,
            highlight_bonds=[tuple(c["junction_bond"])] if c["junction_bond"] else [])
        save(fig, os.path.join(rdir, "protein_pocket_3d"))

    # ==================================================================
    # composites
    # ==================================================================
    # ---- final_construction_pathway (2D) -------------------------------
    fig = plt.figure(figsize=(18.0, 4.6))
    gs = fig.add_gridspec(1, 7, width_ratios=[2.0, 1.25, 1.05, 2.15, 1.25, 1.05, 2.5],
                          wspace=0.22)
    a_seed = fig.add_subplot(gs[0, 0])
    draw_mol_2d(a_seed, seed, xy_seed,
                title="Initial seed (M0)",
                subtitle=f"{traj['seed_block_id']} · {traj['seed_smiles']}",
                highlight_atoms=[centres['step_01']['handle']])
    a_b1 = fig.add_subplot(gs[0, 2])
    draw_mol_2d(a_b1, mols["step_01"]["block"], xy["step_01"]["block"],
                title="+ block 1",
                subtitle=f"{steps[0]['block_id']} · {steps[0]['block_reactant_smiles']}",
                leaving_atoms=centres["step_01"]["leaving"])
    arrow_axes(fig, gs[0, 1], steps[0]["reaction_id"],
               CERTIFIED_REACTIONS[steps[0]["reaction_id"]].name)
    a_m1 = fig.add_subplot(gs[0, 3])
    draw_mol_2d(a_m1, mols["step_01"]["prod"], xy["step_01"]["prod"],
                title="Intermediate M1",
                subtitle=steps[0]["product_smiles"],
                highlight_atoms=[centres["step_02"]["handle"]])
    a_b2 = fig.add_subplot(gs[0, 5])
    draw_mol_2d(a_b2, mols["step_02"]["block"], xy["step_02"]["block"],
                title="+ block 2",
                subtitle=f"{steps[1]['block_id']} · {steps[1]['block_reactant_smiles']}",
                leaving_atoms=centres["step_02"]["leaving"])
    arrow_axes(fig, gs[0, 4], steps[1]["reaction_id"],
               CERTIFIED_REACTIONS[steps[1]["reaction_id"]].name)
    a_fin = fig.add_subplot(gs[0, 6])
    draw_mol_2d(a_fin, final, xy_final,
                title="Final ligand (observed)",
                subtitle=traj["ligand_smiles"])
    fig.suptitle(
        "Complete construction pathway — bindingmoad_4ITP_1GD "
        "(seed → rxn_01 → M1 → rxn_15 → final ligand)",
        fontsize=12, fontweight="bold", y=1.04)
    fig.text(0.5, -0.03,
             "All depicted molecules are the actual trajectory structures; 2D layouts are "
             "scaffold-aligned (the seed remains visually fixed as the molecule grows). "
             "Ringed atoms = reaction centres (the atom that reacts next); dashed rings = leaving groups.",
             ha="center", fontsize=7.5, color="#333333")
    save(fig, os.path.join(RDIR, "final_construction_pathway"))

    # ---- final_ligand_in_pocket_3d ------------------------------------
    fig = plt.figure(figsize=(11.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=14, azim=-58)
    sulf_n = [a.GetIdx() for a in final.GetAtoms()
              if a.GetSymbol() == "N" and any(n.GetSymbol() == "S" for n in a.GetNeighbors())]
    draw_pocket_3d(
        ax, pocket, final, final.GetConformer(),
        title="Final molecule inside its protein binding pocket — PDB 4ITP (human carbonic anhydrase II)",
        subtitle=EXPERIMENTAL_NOTE + " · ligand C15H16N2O3S · pocket 146 atoms + Zn$^{2+}$",
        highlight_atoms=sulf_n)
    ax.text2D(0.02, 0.95,
              "experimentally observed\n(crystal complex 4ITP)",
              transform=ax.transAxes, fontsize=8, color="#0a5f38",
              fontweight="bold", va="top")
    save(fig, os.path.join(RDIR, "final_ligand_in_pocket_3d"))

    print("\nAll demonstration figures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
