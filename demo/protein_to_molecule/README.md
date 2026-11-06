# Protein-to-Molecule Construction — an actual verified example

**A real, recovered, independently re-verified example of 3D-SynTree V2
reconstructing an experimentally observed ligand from certified Enamine
building blocks inside its protein binding pocket.**

- **Example:** `bindingmoad_4ITP_1GD` — one of the **2 two-step
  trajectories** of the full-scale #18 extraction (194 accepted
  trajectories over 170,955 complexes; step histogram 176 / 16 / 2).
- **Protein:** PDB **4ITP**, human carbonic anhydrase II (X-ray
  diffraction; pocket artifact = 146 protein atoms + the catalytic
  **Zn²⁺**).
- **Final ligand:** `NS(=O)(=O)c1ccc(CNC(=O)Cc2ccccc2)cc1`
  (C₁₅H₁₆N₂O₃S, 304.4 Da) — *N*-(4-sulfamoylbenzyl)-2-phenylacetamide,
  a primary-sulfonamide carbonic-anhydrase inhibitor whose
  sulfonamide N coordinates the catalytic Zn²⁺ at a measured
  **1.99 Å**.

> **What this demonstrates (and what it does NOT).** This is the
> **reconstruction capability**: the system selected certified catalog
> building blocks and re-executed certified reactions whose product is
> *exactly* the experimentally observed crystal ligand, in the correct
> protein pocket. It is **not** a claim that a trained policy generated
> this molecule, and not a claim that the full protein-conditioned
> generation pipeline is validated — the trajectory was extracted from
> supervised data (#18) and every reaction below is replayed by the
> project's certified implementation, not by a neural network.

---

## 1. The construction, step by step

### Step 0 — the seed (the anchor in the pocket)

| | |
|---|---|
| Catalog row | **REC0153714** (row 39,841 of the immutable 50k catalog) |
| Vendor form | `CC(=O)O.NCc1ccc(S(N)(=O)=O)cc1` (acetic-acid salt) |
| Free seed fragment | **`NCc1ccc(S(N)(=O)=O)cc1`** — 4-(aminomethyl)benzenesulfonamide |
| Certified seed class | `amine` (the row's seed classes are `alcohol\|amine`) |
| Coordinates | **experimental** — the seed is an induced subgraph of the crystal ligand |

The seed already carries the **primary sulfonamide warhead** — the
zinc-binding group that anchors the whole molecule 1.99 Å from the
catalytic Zn²⁺ of hCA II.

### Step 1 — amide formation (`rxn_01`, certified)

**Actual chemical equation (the certified implementation's own
template and reagents):**

```
Reactant 1 (M0):   NCc1ccc(S(N)(=O)=O)cc1            (the seed, crystal coordinates)
Reactant 2:        CC(=O)Cl                           (acetyl chloride — catalog REC0176965, row 4,713)
Reaction:          rxn_01  "amide formation (acid chloride + amine)"
SMARTS:            [NX3;H2,H1:4].[CX3:1](=[OX1:2])[Cl:3]>>[CX3:1](=[OX1:2])[NX3:4]
Product (M1):      CC(=O)NCc1ccc(S(N)(=O)=O)cc1       (canonical, verified)
```

- **Bond formed:** C(carbonyl)–N(benzylamine) single bond (junction).
- **Bonds broken / leaving group:** the block-side C–Cl (chloride
  deleted). The molecule side loses no atoms.
- **Bond orders:** the C=O double bond is retained (mapped `:2`
  through the template); the aromatic ring is untouched.
- **Reaction centre:** the benzylamine N of M0 and the carbonyl C of
  the incoming block (ringed in every figure).
- **Mechanism (registry note):** nucleophilic acyl substitution at an
  acid chloride; the amide C–N partial double bond pins the junction
  (`resonance_lock = trans`). Catalyst/conditions: **not specified by
  the certified template** (the transformation needs none).
- **Honest chemistry note:** the certified template admits **two**
  constitutional products here — N-acylation at the benzylamine
  (observed in the crystal) and N-acylation at the sulfonamide NH₂
  (not observed). The independent replay below reproduces *both*; the
  crystal structure disambiguates which one is the experimentally
  observed path. `replay_product_count = 2` matches the extraction
  record exactly.
- **Observed junction geometry (from the crystal):** dihedral
  O=C–N–CH₂ = **0.0103 rad** (planar amide) → torsion bin **19/36**.

### Step 2 — alkyl Negishi coupling (`rxn_15`, certified)

**Actual chemical equation:**

```
Reactant 1 (M1, certified reagent form):   NS(=O)(=O)c1ccc(CNC(=O)[CH2][Zn][Cl])cc1
Reactant 2:                                Brc1ccccc1   (bromobenzene — catalog REC0127681, row 35,887)
Reaction:                                  rxn_15  "alkyl Negishi coupling (alkylzinc + aryl bromide)"
SMARTS:                                    [CX4;H2,H3:1][Zn:4][Cl:5].[c:2][Br:3]>>[CX4:1][c:2]
Product (M2 = final ligand):               NS(=O)(=O)c1ccc(CNC(=O)Cc2ccccc2)cc1
```

- **Bond formed:** CH₂(sp³)–C(ipso, sp²) single bond (junction).
- **Bonds broken / leaving groups:** molecule-side C–Zn and Zn–Cl
  (the organozinc graft, deleted together), block-side C–Br (bromide
  deleted).
- **Reaction centre:** the acetyl-methylene carbon of M1 and the ipso
  carbon of bromobenzene.
- **Mechanism (registry note):** Pd/Ni-catalysed sp³–sp² coupling
  (Negishi variant for alkyl centres); primary alkylzinc only — no
  β-hydride stereochemistry reaches the junction. The **catalyst is
  named in the mechanism note but is not modelled by the SMARTS
  template** (metal coordination and conditions are unspecified
  template devices).
- **The `[CH2][Zn][Cl]` graft is the certified template's reagent
  form** (the extractor grafts the zinc reagent onto the molecule-side
  handle so the reaction class is exact). It has **no experimental
  coordinates and is never drawn inside the pocket** — see §3.
- **Observed junction geometry (from the crystal):** dihedral
  O=C–CH₂–C(ipso)–C(ortho) = **2.4810 rad** (≈142°) → torsion bin
  **33/36**.

### Result

The product of step 2 is **exactly the experimentally observed crystal
ligand** — canonical isomeric SMILES equality *and* full graph
isomorphism against the X-ray structure (see `verification.json`), with
the seed's sulfonamide N sitting 1.99 Å from the catalytic Zn²⁺ (S→Zn
3.09 Å) and the pocket's THR199 / HIS94 / HIS96 / HIS119 contacts
reproduced from the artifact coordinates.

---

## 2. Directory contents

```
demo/protein_to_molecule/
├── README.md                        (this file)
├── trajectory.json                  (the complete extracted record — every field)
├── verification.json                (18/18 independent verification checks)
├── structures/
│   ├── protein.pdb                  (crystal pocket: 146 atoms + Zn²⁺ — experimental)
│   ├── seed.sdf                     (M0 — crystal coordinates, experimental)
│   ├── step_01/
│   │   ├── reactant_intermediate.sdf    (M0, crystal coordinates)
│   │   ├── incoming_building_block.sdf  (acetyl chloride — ETKDG, computational)
│   │   └── product.sdf                  (M1, crystal coordinates)
│   ├── step_02/…                     (M1 / bromobenzene / final ligand)
│   ├── final_ligand.sdf             (M2 — the full crystal ligand, experimental)
│   └── raw/
│       ├── bindingmoad_4ITP_1GD_{ligand.sdf,pocket.pdb}.gz   (demo sources)
│       └── rcsb_4ITP_full.pdb.gz    (the FULL experimental structure, fetched from RCSB)
├── reactions/
│   ├── step_01/
│   │   ├── reaction_equation.{png,pdf}        (2D scheme + SMARTS + bond changes)
│   │   ├── molecular_structures_2d.{png,pdf}  (before / incoming / after, highlighted)
│   │   ├── molecular_structures_3d.{png,pdf}  (A/B/C in 3D, provenance-labelled)
│   │   └── protein_pocket_3d.{png,pdf}        (M1 inside the pocket + Zn)
│   ├── step_02/…                     (same four figures for step 2)
│   ├── final_construction_pathway.{png,pdf}  (seed → rxn_01 → M1 → rxn_15 → ligand)
│   └── final_ligand_in_pocket_3d.{png,pdf}   (final ligand in the pocket)
├── protein_structure_3d/            (see §2a — real WebGL molecular graphics)
│   ├── full_protein_cartoon.{png,pdf}              (the whole enzyme, cartoon)
│   ├── active_site_closeup.{png,pdf}               (pocket + Zn²⁺ + residue labels)
│   ├── ligand_overlay_experimental_vs_reconstructed.{png,pdf}
│   ├── protein_surface_pocket.{png,pdf}            (VDW surface, ligand in cleft)
│   ├── protein_rotation.gif                         (36-frame 360° turntable)
│   ├── pocket_graph_3d.{png,pdf}                   (the protein as a 3D graph)
│   ├── protein_viewer.html                          (interactive WebGL viewer)
│   └── render_provenance.json                       (sources, checks, sha256s)
├── scripts/
│   ├── replay_trajectory.py          (re-verification + structure generation)
│   ├── generate_visualizations.py    (reaction figures; RDKit + Matplotlib)
│   ├── render_protein_3d.py          (builds the self-contained WebGL viewer)
│   ├── render_protein_3d_shots.py    (headless Chromium renders + GIF + checks)
│   └── pocket_graph_3d.py            (the protein-as-graph figures)
└── verification/
    ├── reaction_replay.json          (per-step certified replay evidence)
    └── coordinate_provenance.json    (experimental vs computational coordinates)
```

All figures are 300-DPI PNG + vector PDF, generated *only* by the
committed scripts (never drawn by hand). RDKit handles every molecular
graph, reaction template, conformer and 2D depiction; Matplotlib only
renders.

---

## 2a. The protein itself — real 3D molecular graphics (3Dmol.js / WebGL)

The pocket scatter figures of §2 are complemented by genuine molecular
graphics renders of the **full experimental protein**, produced with
**3Dmol.js 2.4.0 (WebGL) executed in headless Chromium 143 via
Playwright**, plus a Matplotlib rendering of the protein **as an actual 3D
node-and-edge graph**. Every render uses only real coordinates, and the
full deposition was fetched from RCSB
(`https://files.rcsb.org/download/4ITP.pdb`, sha256 recorded in
`render_provenance.json`):

| File | Content |
|---|---|
| `full_protein_cartoon.{png,pdf}` | the whole enzyme — 261-residue cartoon colored by residue position (turbo), ligand 1GD, catalytic Zn²⁺, the 32 demo-pocket residues |
| `active_site_closeup.{png,pdf}` | pocket residues as sticks, labelled catalytic residues (His119, Thr199, Trp209, Gln92, …), ball-and-stick 1GD, gray Zn²⁺ sphere |
| `ligand_overlay_experimental_vs_reconstructed.{png,pdf}` | experimental 1GD (cyan sticks) vs the 2-step synthetic reconstruction (orange translucent spheres) — **RMSD 0.000 Å, 21/21 atoms coincident** |
| `protein_surface_pocket.{png,pdf}` | van der Waals surface of chain A (translucent) with 1GD nestled in the active-site cleft above Zn²⁺ |
| `protein_rotation.gif` | 36-frame 360° turntable of the full-protein scene |
| `pocket_graph_3d.{png,pdf}` | **the protein as a graph**: (a) Cα backbone graph — 257 nodes / 256 peptide edges; (b) binding-site atomic graph — 145 pocket atoms + 21 ligand atoms + Zn²⁺, covalent edges, dashed Zn coordination |
| `protein_viewer.html` | self-contained interactive WebGL viewer (no network needed) — four scenes + spin, open in any browser |
| `render_provenance.json` | renderer versions, source sha256s, frame checks, output sha256s |

**Frame verification (all fail-closed, re-run by
`render_protein_3d_shots.py`):** every one of the 145 pocket atoms matches
the full deposition atom-for-atom (max deviation **0.000 Å**); the
BindingMOAD ligand SDF equals the deposited 1GD HETATM coordinates (max
deviation **0.000 Å**, 21/21 atoms); Zn²⁺–N(sulfonamide) = **1.99 Å**;
Zn²⁺ is coordinated by His94 / His96 / His119 (distances in the
provenance file). Still renders are 2400×1800 px = 8×6 in at 300 DPI.

Reproduce with:

```bash
python scripts/render_protein_3d.py            # builds protein_viewer.html (fetches 4ITP + 3Dmol.js once)
python scripts/render_protein_3d_shots.py       # headless Chromium renders, GIF, provenance, checks
python scripts/pocket_graph_3d.py               # the protein-as-graph figures
```

(requires `playwright` with Chromium, `pillow`, `matplotlib`; the first
run needs network access to RCSB and jsDelivr, afterwards everything is
cached/offline).

These renders visualize the **experimentally observed complex** —
they are a reconstruction/visualization of crystallographic data, not an
AI generation (see `render_provenance.json` → `purpose`).

---

| Structure | Coordinates | Source |
|---|---|---|
| Protein pocket (146 atoms + Zn²⁺) | **experimental** | X-ray, PDB 4ITP pocket artifact |
| Seed M0, intermediate M1, final ligand M2 | **experimental** | induced subgraphs of the crystal ligand (the #26/D-V2-030 construction: intermediates *are* crystal subgraphs) |
| Incoming blocks (acetyl chloride, bromobenzene) | **computational** | deterministic RDKit ETKDG conformers (seed 2026); **no experimentally established pocket position — always rendered outside the pocket frame and labelled** |
| The `[CH2][Zn][Cl]` organozinc graft (rxn_15 reagent form) | none | a certified-template device; documented, never drawn as a placed structure |
| Junction dihedrals / torsion bins | **experimental** | recomputed from the crystal coordinates; both reproduce the record exactly (0.010335 rad / bin 19; 2.481023 rad / bin 33) |

No intermediate coordinate was fabricated anywhere in this demo. No
independently generated conformation is ever presented as
experimentally observed; no docking was performed.

---

## 4. Independent verification (18/18 checks — re-run them yourself)

`scripts/replay_trajectory.py` re-verifies the example from first
principles with the project's **own certified implementations**
(`syntree.chemistry.reactions.run_certified_reaction`,
`syntree.chemistry.synthon_encoder.read_catalog_h5`,
`syntree.data.trajectories.induced_submol`) — never a generic RDKit
substitute:

1. crystal ligand artifact parses, sanitizes, has one experimental conformer;
2. the catalog H5 is the pinned immutable build (sha256 `8b8c13a0…`);
3. the seed is a real certified row (REC0153714 = row 39,841, class `amine`);
4. the seed fragment derives from the catalog row's vendor salt form;
5. step-1 block REC0176965 = row 4,713, `acid_chloride` (rxn_01's required class);
6. step-2 block REC0127681 = row 35,887, `aryl_bromide` (rxn_15's required class);
7. both reactions belong to the certified 32-reaction registry;
8. **certified forward replay reproduces each recorded product graph** (step 1
   yields both constitutional isomers, matching the recorded
   `replay_product_count = 2`; step 2 yields exactly one product);
9. every intermediate is chemically valid (sanitized, neutral, formula-checked);
10. the final molecule **equals the experimentally observed crystal ligand**
    (canonical isomeric SMILES + full graph isomorphism);
11. atom identity / formal charges / stereochemistry per the acceptance
    contract (neutral; no stereocentres in this ligand);
12. both junction dihedrals and torsion bins **reproduce from the
    experimental coordinates** (crystal atom indices recorded in
    `verification.json`).

```bash
# reproduce everything (catalog from the pinned HF dataset or local store):
python scripts/replay_trajectory.py --catalog /path/to/enamine_real_50k_pharm3d_d256.h5
python scripts/generate_visualizations.py          # regenerates every figure
```

---

## 5. Final report (the twelve questions)

1. **Did you find an actual two-step example?** Yes — one of exactly two
   in the 194-trajectory dataset (both are the same crystal structure,
   4ITP, accepted through two different sources: BindingMOAD and
   PDBbind; this demo uses the BindingMOAD record; the PDBbind twin
   `pdbbind_4itp` is in the extraction output).
2. **Complex ID?** `bindingmoad_4ITP_1GD` (train split).
3. **Which protein?** PDB **4ITP** — human carbonic anhydrase II
   (X-ray; catalytic Zn²⁺ retained in the pocket artifact).
4. **Initial seed molecule?** `NCc1ccc(S(N)(=O)=O)cc1`
   (4-(aminomethyl)benzenesulfonamide; Enamine **REC0153714**,
   catalog row 39,841).
5. **Building blocks added?** `CC(=O)Cl` (acetyl chloride,
   **REC0176965**, row 4,713) and `Brc1ccccc1` (bromobenzene,
   **REC0127681**, row 35,887) — both real certified catalog rows.
6. **Certified reactions used?** `rxn_01` (amide formation, family
   `amide_and_carbamate`) and `rxn_15` (alkyl Negishi coupling, family
   `cross_coupling_cc`) — both members of the certified 32-reaction
   registry, executed by `run_certified_reaction`.
7. **Final molecule?** `NS(=O)(=O)c1ccc(CNC(=O)Cc2ccccc2)cc1`
   — *N*-(4-sulfamoylbenzyl)-2-phenylacetamide, C₁₅H₁₆N₂O₃S; **exactly
   the experimentally observed crystal ligand.**
8. **Did every reaction independently pass forward replay?** Yes —
   step 1 reproduces the recorded product among its two constitutional
   products (the crystal selects the benzylamine-N acylation); step 2
   reproduces its single recorded product. Product counts match the
   extraction record (`2` and `1`).
9. **Can we visualize the actual protein pocket and final ligand
   together?** Yes — `reactions/final_ligand_in_pocket_3d.{png,pdf}` and
   the per-step `protein_pocket_3d` figures: both in the original
   crystal coordinate frame, with the catalytic Zn²⁺, the 1.99 Å
   Zn–N(sulfonamide) coordination distance, and the THR199 / HIS94 /
   HIS96 / HIS119 contact residues measured from the artifact. The full
   protein itself is additionally rendered in §2a — cartoon, active-site
   close-up, VDW surface, a rotating GIF, and the protein **as a 3D
   graph** (`protein_structure_3d/`), all in the same crystal frame.
10. **Can we visualize the intermediate construction stages in 3D?**
    Yes — M0 and M1 carry **true experimental coordinates** (induced
    crystal subgraphs), shown per step in
    `molecular_structures_3d` panels A/C and inside the pocket in
    `protein_pocket_3d`. The incoming blocks are ETKDG conformers,
    rendered separately and labelled computational — their pocket
    position is **not** claimed.
11. **Which parts are experimentally observed, and which are
    reconstructed?** Experimental: the pocket, the crystal ligand, and
    every intermediate's coordinates (as crystal subgraphs), the
    junction dihedrals/torsion labels. Reconstructed (chemistry, not
    coordinates): the reaction events themselves and the incoming
    blocks' conformers. See §3 and
    `verification/coordinate_provenance.json`.
12. **Where are the reproducible artifacts stored?** This directory
    (`demo/protein_to_molecule/`, branch `3d-synV2`) — `trajectory.json`
    + `structures/` + `verification/` + the two scripts regenerate
    every figure deterministically. The raw extraction evidence is the
    committed `docs/data/trajectories/` evidence set; the trajectory
    `.pt` artifacts were rebuilt this session on the authorized T4 and
    verified against the pinned full-scale sha256s
    (`8e95ed89…` / `76fe9307…` / `4f671849…`).

---

## 6. Provenance chain (why these numbers are trustworthy)

```
PDB 4ITP (X-ray)
  └─ #16 preprocessing → bindingmoad_4ITP_1GD_{ligand.sdf,pocket.pdb}.gz   (md5-verified sources)
       └─ #18 extraction (exact forward replay + crystal-CIP labels)
            └─ trajectories_train.pt  (sha256 8e95ed89… — rebuilt on the authorized T4, byte-identical)
                 └─ this demo's record  →  18/18 independent re-verification checks
Certified Enamine 50k catalog (sha256 8b8c13a0…)  →  seed + both blocks (rows verified)
Certified 32-reaction registry (rxn_01, rxn_15)  →  independent replay
```

Every number in this README is either measured from these artifacts or
reproduced by `scripts/replay_trajectory.py`; nothing is quoted from
memory.
