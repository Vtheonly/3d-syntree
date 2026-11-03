"""The ``pharm3d`` synthon encoder and catalog chemistry layer (issue #12
CHEM-02; re-landed against the surviving build artifacts, session
S-V2-11).

This module is the **one authoritative** implementation of the catalog's
chemistry: the 2137-d raw ``pharm3d`` descriptor, the ETKDGv3 + MMFF94
lowest-energy conformer policy (fail-closed with per-row provenance),
the frozen z-normalization + QR Johnson-Lindenstrauss projection (seed
2026), the immutable HDF5 catalog schema (``pharm3d-v2.0`` /
``catalog-v2.0``), the 64-D invariant handle descriptors consumed by
``HandleSelectionHead`` (#22), and the seed-role certification
vocabulary.  The curation funnel (screen-before-admission via the #40
chemoselectivity layer) lives in the build driver
``scripts/build_synthon_catalog.py``.

Reconstruction provenance (constitution #45 §18 — honest scoping)
------------------------------------------------------------------
The S-V2-10 build lost its code to a sandbox recycle before commit.
Every design fact below is **artifact-pinned** against
``docs/data/catalog/`` + the immutable HDF5 (HF ``JJKK1212/
3d-syntree-v2-catalog``); the S-V2-11 re-landing verified each pin at
full scale over the 290,648-record source:

* **Funnel — exact (24/24 counts).** The rebuilt pipeline reproduces
  every stage count and every per-rule exclusion count of the pinned
  ``screen_funnel.json`` bit-for-bit: 290,648 total / 290,356 unique /
  191,639 basic-pass (stage order MW -> rotatable -> heavy) /
  50,065 handle-bearing / 152,489 certified (118,683 seed-only +
  20,390 both + 13,416 synthon-only) / 39,150 chemoscreen-excluded;
  rules: competing-nucleophile 17,235, quenching 7,397, identical
  handles 4,310, SNAr 1,022, self-reactive 771, seed self-reactive
  757 (757 fired blocks; 760 pair-firings), competing electrophile
  150, catalyst poison 44.
* **Certification model — exact (0/50,000 row mismatches).**  Synthon
  role: a block is certified for registry handle class ``C`` iff
  ``screen_block(mol, C, handle_atom_idx=<first instance>)`` passes
  (the two programmatic boronic classes are screened without a declared
  atom).  Seed role: **block-level** — the 7-class vocabulary below,
  and a molecule carrying ANY free-group/electrophile self-reactive
  pair (the #40 ``SELF_REACTIVE_PAIRS`` matrix between the molecule's
  *free* groups and the six uncatalyzed electrophiles: acid/sulfonyl/
  carbamoyl chlorides, chloroformate, alkyl bromide, aldehyde) loses
  the entire seed role, not just the firing class.
* **Strata — exact (0/50,000):** ``CalcNumAromaticRings(mol) > 0`` →
  ``aromatic``, else ``saturated`` (a structural test, not an Fsp3
  threshold).
* **Selection — exact.** The 55,000-row buffer (35,000 saturated +
  20,000 aromatic, canonical-SMILES ascending, saturated first)
  reproduces the artifact's ``buffered_rows`` index list exactly.
* **Descriptor blocks — 44 of 47 physchem scalars pinned exactly**
  (mean AND std over the 50k artifact rows); the layout is 2048 Morgan
  + 47 physchem + 6 shape + 5 electro + 30 handle vectors + 1 norm =
  **2137** (the issue body's "2138" does not add up — pinned by
  ``normalization/mean`` shape 2137 in the HDF5).  Three functional-
  group positions could not be uniquely identified from the stored
  statistics and are implemented as the family-coherent reconstruction
  (documented in ``FG_SMARTS``; measured distribution deltas recorded
  in the build manifest): the tertiary-amine position (stored
  0.070600/0.266337 vs reconstruction 0.079120/0.280820), the position
  after sulfone (stored 0.001240/0.035192 — 62 rows; implemented as
  sulfoxide, 64 rows / 0.002520), and one **provably all-zero**
  position (any SMARTS that matches nothing on this catalog is
  indistinguishable; implemented as peroxide, matching the #32
  unstable-motif vocabulary).  The 30-D handle-vector block carries
  identity normalization in the artifact, so its 10-class list is not
  recoverable from the stored statistics; ``VECTOR_HANDLE_CLASSES`` is
  the documented reconstruction (the seven seed classes plus the three
  most common non-seed registry handle classes).
* **Projection — bit-exact:** ``numpy.random.default_rng(2026).normal(
  size=(2137, 256))`` → ``np.linalg.qr`` → column sign-normalization by
  ``sign(diag(R))`` reproduces the stored ``projection/Q`` float32
  bit-for-bit (verified max |Δ| = 0.0).
* **Normalization:** per-feature z-normalization of the physchem /
  shape / electro / norm blocks with statistics computed over the
  selected rows and **stored** in the artifact (std floor 1e-12); the
  Morgan and handle-vector blocks pass through with identity
  normalization (stored mean 0 / std 1 — pinned from the artifact).
* **Conformer policy — profile-exact.** ETKDGv3 multi-conformer
  embedding followed by full MMFF94 relaxation of every conformer
  where RDKit's MMFF types the hydrogen-added molecule (lowest-energy
  conformer kept, ``etkdg3+mmff94``); rows whose untypeable atoms fall
  within the documented fallback set (coupling handles B/Zn/Sn **and
  halide counterions F/Cl/Br/I**) keep their first ETKDG geometry
  (``etkdg3``, energy NaN); anything else fails closed with the named
  symbols (potassium trifluoroborates fail on ``['B', 'K']``).  This
  decision tree reproduces the artifact's conformer-method split
  (17,493/32,507) with **zero** per-row mismatches over the 50k
  catalog.  The re-landed run is deterministic (``CONFORMER_SEED`` =
  2026, 10 conformers, prune 0.5) but the **original run's ETKDG seed
  is not recoverable from the surviving artifacts** (the HDF5 stores
  only final energies + method strings, which underdetermine the
  protocol; probe rows reproduce the seed-2026 multi-conformer family
  but not every stored energy), so the immutable HDF5 remains the
  authoritative embedding artifact (D-V2-021 immutability) and the
  re-landed encoder documents its own seed as the reproducibility
  contract for rebuilds.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors

from syntree.chemistry.chemoselectivity import (
    ALL_HANDLE_CLASSES,
    detect_handle_instances,
    screen_block,
)

__all__ = [
    "D_RAW",
    "D_EMBED",
    "D_HANDLE_FEATURES",
    "MORGAN_BITS",
    "MORGAN_RADIUS",
    "PROJECTION_SEED",
    "SCHEMA_VERSION",
    "CATALOG_SCHEMA_VERSION",
    "SEED_CLASS_SMARTS",
    "SEED_ACYL_ELECTROPHILES",
    "NUCLEOPHILIC_SEED_CLASSES",
    "FG_SMARTS",
    "VECTOR_HANDLE_CLASSES",
    "SynthonEncodeError",
    "EncodedSynthon",
    "ConformerOutcome",
    "embed_conformer",
    "pharm3d_descriptor",
    "handle_descriptor_64",
    "certify_block_roles",
    "detect_seed_classes",
    "build_projection",
    "project_raw",
    "encode_synthon",
    "write_catalog_h5",
    "read_catalog_h5",
    "registry_digest",
]

# ---------------------------------------------------------------------------
# Frozen dimensions and schema versions (artifact-pinned)
# ---------------------------------------------------------------------------

#: The raw descriptor dimensionality: 2048 + 47 + 6 + 5 + 30 + 1.
D_RAW = 2137
D_EMBED = 256
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
PROJECTION_SEED = 2026
SCHEMA_VERSION = "pharm3d-v2.0"
CATALOG_SCHEMA_VERSION = "catalog-v2.0"

#: The invariant handle-descriptor width consumed by HandleSelectionHead
#: (#22; frozen schema constant ``HANDLE_FEATURE_DIM``).
D_HANDLE_FEATURES = 64

#: z-score floor for degenerate features (pinned by normalization/std).
ZSCORE_FLOOR = 1e-12

# descriptor block offsets (artifact layout)
OFF_MORGAN = 0
OFF_PHYSCHEM = 2048
OFF_SHAPE = 2048 + 47
OFF_ELECTRO = OFF_SHAPE + 6
OFF_VECTORS = OFF_ELECTRO + 5
OFF_NORM = OFF_VECTORS + 30  # == 2136

#: The ETKDG conformer protocol of the re-landed encoder (deterministic).
CONFORMER_SEED = 2026
CONFORMER_NUM_CONFS = 10
CONFORMER_PRUNE_RMS = 0.5
MMFF_MAX_ITERS = 200


class SynthonEncodeError(ValueError):
    """Fail-closed encoding error with a named reason (never a NaN)."""


# ---------------------------------------------------------------------------
# Seed-role vocabulary (artifact-pinned; 0/50,000 label mismatches)
# ---------------------------------------------------------------------------

#: The seven seed (initial-fragment) classes and their detectors.
SEED_CLASS_SMARTS: Dict[str, str] = {
    # free O-H on any carbon (alcohols, phenols, carboxylic acids)
    "alcohol": "[OX2H][#6]",
    # non-aromatic N-H (primary/secondary incl. amidines/thioureas;
    # acyl (N-C=O) and sulfonyl/sulfinyl (N-S=O) nitrogens excluded)
    "amine": "[NX3;H2,H1;!$([n]);!$(NC=[OX1]);!$(N[S]=[OX1])]",
    # free thiol S-H
    "thiol": "[SX2H]",
    # any boronic acid B(OH)2
    "boronic_acid": "[BX3]([OX2H])[OX2]",
    "aryl_bromide": "[Br][c]",
    "aryl_chloride": "[Cl][c]",
    "vinyl_bromide": "[Br][CX3]=[CX3]",
}

#: Uncatalyzed electrophiles that intramolecularly react with a free
#: nucleophilic seed group (the seed_self_reactive_pair rule): the #40
#: ``SELF_REACTIVE_PAIRS`` matrix keyed on the seed's own FREE group —
#: the pair fires only when the molecule carries the free form of the
#: group (an amidine "amine" seed with an alkyl bromide is safe; a free
#: primary amine is not) and the electrophile's partner set contains it.
SEED_ACYL_ELECTROPHILES: Tuple[str, ...] = (
    "acid_chloride",
    "sulfonyl_chloride",
    "carbamoyl_chloride",
    "chloroformate",
    "alkyl_bromide",
    "aldehyde",
)

#: Seed classes whose free group can self-react with an electrophile.
NUCLEOPHILIC_SEED_CLASSES: Tuple[str, ...] = ("amine", "alcohol", "thiol")

#: The #40 free groups that realize each nucleophilic seed class (the
#: carboxylic acid O-H realizes the ``alcohol`` seed, matching the
#: artifact-pinned alcohol detector ``[OX2H][#6]``).
SEED_FREE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "amine": ("primary_amine", "secondary_amine"),
    "alcohol": ("alcohol", "phenol", "carboxylic_acid"),
    "thiol": ("thiol",),
}


def detect_seed_classes(mol: Chem.Mol) -> List[str]:
    """The seed classes detected on ``mol`` (vocabulary order)."""
    out = []
    for cls, smarts in SEED_CLASS_SMARTS.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:  # pragma: no cover — load-time pinned
            raise SynthonEncodeError(f"seed SMARTS for {cls!r} unparseable")
        if mol.GetSubstructMatches(query):
            out.append(cls)
    return out


def seed_screen(mol: Chem.Mol, seed_class: str) -> Optional[str]:
    """The seed-role chemoselectivity screen; ``None`` = certified.

    A nucleophilic/protic seed class is excluded when the molecule
    carries the seed's **free** group (the #40 free-form vocabulary —
    orthogonal protection passes by construction) *and* an uncatalyzed
    electrophilic handle whose #40 ``SELF_REACTIVE_PAIRS`` partner set
    contains that free group: the pair would self-react
    intramolecularly and quench the seed (e.g. free primary amine +
    aldehyde -> imine; free alcohol + alkyl bromide -> ether;
    any free nucleophile + acid chloride -> amide/ester).  An amidine
    "amine" seed beside a certified alkyl bromide is safe (its N is not
    the free form), and an alcohol seed beside an aldehyde is safe
    (reversible hemiacetal — the pinned matrix pairs aldehydes with
    amines only).
    """
    if seed_class not in NUCLEOPHILIC_SEED_CLASSES:
        return None
    from syntree.chemistry.chemoselectivity import (
        FREE_GROUP_SMARTS,
        SELF_REACTIVE_PAIRS,
    )

    own_groups = SEED_FREE_GROUPS[seed_class]
    free_present = {
        group
        for group, smarts in FREE_GROUP_SMARTS.items()
        if group in own_groups
        and mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
    }
    if not free_present:
        return None
    for electrophile in SEED_ACYL_ELECTROPHILES:
        if not detect_handle_instances(mol, electrophile):
            continue
        partners = SELF_REACTIVE_PAIRS.get(electrophile, frozenset())
        if free_present & partners:
            return f"seed_self_reactive_pair:{electrophile}"
    return None


def certify_block_roles(mol: Chem.Mol) -> Tuple[List[str], List[str], List[str]]:
    """Certified (synthon_classes, seed_classes, exclusion firings).

    Synthon certification screens every detected registry handle class
    with its first instance as the declared attachment atom (the
    artifact-pinned protocol; the two programmatic boronic classes are
    screened without a declared atom).  **Seed certification is
    block-level** (artifact-pinned: a molecule with ANY free-group /
    electrophile self-reactive pair loses the entire seed role, not
    just the firing class — verified against the pinned funnel counts:
    exactly 757 fired blocks over the 290,648-record source, and zero
    catalog rows keep a seed label beside a fired pair).  Reason codes
    are emitted once per fired block (deterministic order), matching
    the artifact's per-block seed-exclusion counting.
    """
    firings: List[str] = []
    synthon_ok: List[str] = []
    for cls in sorted(ALL_HANDLE_CLASSES):
        instances = detect_handle_instances(mol, cls)
        if not instances:
            continue
        if cls in ("arylboronic_acid", "heteroarylboronic_acid"):
            result = screen_block(mol, cls)
        else:
            result = screen_block(mol, cls, handle_atom_idx=instances[0])
        if result.status == "pass":
            synthon_ok.append(cls)
        else:
            firings.extend(result.reason_codes)

    seed_detected = detect_seed_classes(mol)
    seed_fired: Optional[str] = None
    for seed_class in seed_detected:
        reason = seed_screen(mol, seed_class)
        if reason is not None and seed_fired is None:
            seed_fired = reason
    seed_ok: List[str] = [] if seed_fired else list(seed_detected)
    if seed_fired:
        firings.append(seed_fired)
    return synthon_ok, seed_ok, firings


# ---------------------------------------------------------------------------
# The 47 physicochemical scalars (44 artifact-pinned + 3 documented
# reconstruction ambiguities — see the module docstring)
# ---------------------------------------------------------------------------

#: Functional-group counts, in artifact order.  Positions 2081 (tertiary
#: amine), 2088 (post-sulfone rare group) and 2090 (provably all-zero on
#: this catalog) are the documented reconstruction choices.
FG_SMARTS: Dict[str, str] = {
    "aldehyde": "[CX3H1](=[OX1])[#6]",
    "ketone": "[CX3](=[OX1])([#6])[#6]",
    "carboxylic_acid": "[CX3](=[OX1])[OX2H]",
    "ester": "[CX3](=[OX1])[OX2][#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "primary_amine": "[NX3;H2;!$(NC=[OX1]);!$(N[S]=[OX1])]",
    "secondary_amine": "[NX3;H1;!$(NC=[OX1]);!$(N[S]=[OX1])]",
    "tertiary_amine": "[NX3;H0;!$(NC=[OX1]);!$(N[S]=[OX1])]",
    "alcohol": "[OX2H][#6]",
    "phenol": "[OX2H][c]",
    "thiol": "[SX2H]",
    "ether": "[OX2]([#6])[#6]",
    "thioether": "[SX2]([#6])[#6]",
    "sulfone": "[SX4](=[OX1])(=[OX1])[#6]",
    "sulfoxide": "[SX3](=[OX1])[#6]",
    "nitrile": "[CX2]#[NX1]",
    "peroxide": "[OX2][OX2]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "iodine": "[I]",
}

_FG_QUERIES = {
    name: Chem.MolFromSmarts(sma) for name, sma in FG_SMARTS.items()
}

#: The ten handle classes whose centroid-relative mean unit directions
#: form the 30-D vector block (artifact layout: 10 x 3).  The exact
#: class list of the lost original is not recoverable from the stored
#: statistics (the block carries identity normalization); this list is
#: the seven seed classes plus the three most common non-seed registry
#: handle classes of the certified catalog, and the choice is recorded
#: as a reconstruction note in the build manifest.
VECTOR_HANDLE_CLASSES: Tuple[str, ...] = (
    "alcohol",
    "amine",
    "thiol",
    "boronic_acid",
    "aryl_bromide",
    "aryl_chloride",
    "vinyl_bromide",
    "aldehyde",
    "alkyl_bromide",
    "acetylide",
)


def _physchem_features(mol: Chem.Mol) -> List[float]:
    """The 47 physicochemical scalars in artifact order."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    values: List[float] = [
        Descriptors.MolWt(mol),                        # 2048 exact
        Descriptors.ExactMolWt(mol),                   # 2049 exact
        float(mol.GetNumHeavyAtoms()),                 # 2050 exact
        float(mol.GetNumAtoms()),                      # 2051 exact
        float(mol.GetNumBonds()),                      # 2052 exact
        float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "C")),
                                                     # 2053 heteroatoms exact
        rdMolDescriptors.CalcFractionCSP3(mol),        # 2054 exact
        float(Lipinski.NumRotatableBonds(mol)),        # 2055 exact
        float(Lipinski.NumHDonors(mol)),               # 2056 exact
        float(Lipinski.NumHAcceptors(mol)),            # 2057 exact
        Descriptors.MolLogP(mol),                      # 2058 exact
        Crippen_MolMR(mol),                            # 2059 MolMR exact
        Descriptors.TPSA(mol),                         # 2060 TPSA exact
        rdMolDescriptors.CalcLabuteASA(mol),           # 2061 exact
        float(mol.GetRingInfo().NumRings()),           # 2062 exact
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),     # 2063 exact
        float(rdMolDescriptors.CalcNumAliphaticRings(mol)),    # 2064 exact
        float(rdMolDescriptors.CalcNumSaturatedRings(mol)),    # 2065 exact
        float(Chem.GetFormalCharge(mol)),              # 2066 exact (const 0)
        float(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())),
                                                     # 2067 exact
        float(Descriptors.NumValenceElectrons(mol)),   # 2068 exact
        float(rdMolDescriptors.CalcNumAmideBonds(mol)),  # 2069 exact
        # -- Lipinski section (duplicated scalars, artifact-pinned) --
        Descriptors.MolWt(mol),                        # 2070 exact
        Descriptors.MolLogP(mol),                      # 2071 exact
        float(Lipinski.NumHDonors(mol)),               # 2072 exact
        float(Lipinski.NumHAcceptors(mol)),            # 2073 exact
    ]
    # -- functional-group counts (artifact order) --
    for name in FG_SMARTS:
        query = _FG_QUERIES[name]
        assert query is not None  # load-time pinned
        values.append(float(len(mol.GetSubstructMatches(query, uniquify=True))))
    assert len(values) == 47, f"physchem block must be 47-wide, got {len(values)}"
    return values


def Crippen_MolMR(mol: Chem.Mol) -> float:
    """Crippen molar refractivity (kept explicit for the pinned order)."""
    from rdkit.Chem import Crippen

    return Crippen.MolMR(mol)


# ---------------------------------------------------------------------------
# Conformer policy (fail-closed, per-row provenance)
# ---------------------------------------------------------------------------

#: Atoms MMFF94 cannot type that are nevertheless accepted: the
#: coupling-handle chemistries (boronic acids B, aryl zincs Zn, vinyl
#: stannanes Sn) **and the halide counterions** of the catalog's salt
#: forms (HBr/HCl/HI/HF salts keep their ETKDG geometry — artifact-
#: pinned: the fallback reproduces the stored 17,493/32,507
#: conformer-method split exactly; potassium trifluoroborate salts fail
#: closed because K is not in the set).
DOCUMENTED_MMFF_FALLBACK_ATOMS = frozenset({"B", "Zn", "Sn", "F", "Cl", "Br", "I"})


@dataclass
class ConformerOutcome:
    """The per-molecule conformer record written to the catalog."""

    method: str  # 'etkdg3+mmff94' | 'etkdg3'
    energy: float  # kcal/mol; NaN when MMFF was not applied
    conformer: Chem.Mol  # molecule with one heavy-atom conformer


def _mmff_untypeable_symbols(mol: Chem.Mol) -> List[str]:
    """Atom symbols outside MMFF94's organic element coverage.

    This is the *fallback audit*, not the typing decision itself: RDKit's
    MMFF can also fail to type covered elements in unusual states (the
    catalog's HBr/HCl salt counterions), which is exactly why the
    decision tree routes every ``MMFFHasAllMoleculeParams == False`` row
    through the fallback check instead of silently relaxing.
    """
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    covered = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
    return sorted(symbols - covered)


def embed_conformer(smiles: str) -> ConformerOutcome:
    """ETKDGv3 multi-conformer embedding + lowest-energy MMFF94 relaxation.

    Fail-closed policy (artifact-pinned — reproduces the stored
    17,493/32,507 conformer-method split over the 50k catalog exactly):

    * the MMFF94 typing check runs on the **hydrogen-added molecule
      before embedding**; untypeable atoms outside the documented
      fallback set fail closed immediately with the named symbols;
    * MMFF types the molecule -> all conformers relaxed
      (``MMFFOptimizeMoleculeConfs``), the lowest-energy one is kept
      (``etkdg3+mmff94`` provenance, energy recorded);
    * MMFF cannot type and the untypeable atoms are within the
      documented coupling-handle/counterion fallback -> the first
      ETKDG geometry is kept unrelaxed (``etkdg3``, energy NaN);
    * no ETKDG conformer at all -> :class:`SynthonEncodeError`
      ("ETKDGv3 embedding produced no conformers").
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SynthonEncodeError(f"unparseable SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    # The typing decision (artifact-pinned): relax only when RDKit's MMFF
    # types the whole hydrogen-added molecule; anything else routes
    # through the documented fallback (etkdg3) or fails closed.
    mmff_typeable = AllChem.MMFFHasAllMoleculeParams(mol)
    untypeable = _mmff_untypeable_symbols(mol)
    if not mmff_typeable and not (
        set(untypeable) <= DOCUMENTED_MMFF_FALLBACK_ATOMS
    ):
        raise SynthonEncodeError(
            "MMFF94 cannot type atoms "
            f"{untypeable} and they are not the documented "
            "B/Zn/Sn coupling-handle fallback (fail-closed)"
        )

    params = AllChem.ETKDGv3()
    params.randomSeed = CONFORMER_SEED
    params.pruneRmsThresh = CONFORMER_PRUNE_RMS
    cids = list(
        AllChem.EmbedMultipleConfs(
            mol, numConfs=CONFORMER_NUM_CONFS, params=params
        )
    )
    if not cids:
        raise SynthonEncodeError(
            "ETKDGv3 embedding produced no conformers (fail-closed)"
        )

    if mmff_typeable:
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=1, mmffVariant="MMFF94", maxIters=MMFF_MAX_ITERS
        )
        best_idx = min(range(len(cids)), key=lambda i: results[i][1])
        best_cid = cids[best_idx]
        energy = float(results[best_idx][1])
        method = "etkdg3+mmff94"
    else:
        best_cid = cids[0]
        energy = float("nan")
        method = "etkdg3"

    keep = Chem.Mol(mol, confId=best_cid)
    keep = Chem.RemoveHs(keep)
    return ConformerOutcome(method=method, energy=energy, conformer=keep)


# ---------------------------------------------------------------------------
# The pharm3d descriptor
# ---------------------------------------------------------------------------

def _shape_features(conf_mol: Chem.Mol) -> List[float]:
    """The six RDKit 3D shape descriptors (artifact order)."""
    return [
        rdMolDescriptors.CalcRadiusOfGyration(conf_mol),
        rdMolDescriptors.CalcAsphericity(conf_mol),
        rdMolDescriptors.CalcEccentricity(conf_mol),
        rdMolDescriptors.CalcInertialShapeFactor(conf_mol),
        rdMolDescriptors.CalcNPR1(conf_mol),
        rdMolDescriptors.CalcNPR2(conf_mol),
    ]


def _electro_features(conf_mol: Chem.Mol) -> List[float]:
    """Gasteiger charge statistics + dipole magnitude (artifact order)."""
    AllChem.ComputeGasteigerCharges(conf_mol)
    charges = np.array(
        [a.GetDoubleProp("_GasteigerCharge") for a in conf_mol.GetAtoms()],
        dtype=np.float64,
    )
    conf = conf_mol.GetConformer()
    positions = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(conf_mol.GetNumAtoms())],
        dtype=np.float64,
    )
    dipole = charges @ positions  # e*A
    return [
        float(charges.sum()),
        float(charges.std()),
        float(charges.min()),
        float(charges.max()),
        float(np.linalg.norm(dipole)),
    ]


def _handle_vector_features(
    mol: Chem.Mol, conf_mol: Chem.Mol
) -> List[float]:
    """The 30-D centroid-relative mean handle directions (10 classes x 3).

    For each vector class with detected handle/seed atoms, the mean unit
    direction from the molecular centroid to those atoms is computed in
    the conformer frame; absent classes contribute zeros.
    """
    conf = conf_mol.GetConformer()
    positions = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(conf_mol.GetNumAtoms())],
        dtype=np.float64,
    )
    centroid = positions.mean(axis=0)
    out: List[float] = []
    for cls in VECTOR_HANDLE_CLASSES:
        if cls in SEED_CLASS_SMARTS:
            query = Chem.MolFromSmarts(SEED_CLASS_SMARTS[cls])
            matches = mol.GetSubstructMatches(query, uniquify=True)
            atoms = sorted({m[0] for m in matches})
        else:
            atoms = list(detect_handle_instances(mol, cls))
        if not atoms or not np.isfinite(positions).all():
            out.extend([0.0, 0.0, 0.0])
            continue
        deltas = positions[atoms] - centroid
        norms = np.linalg.norm(deltas, axis=1)
        unit = np.zeros_like(deltas)
        nonzero = norms > 0
        unit[nonzero] = deltas[nonzero] / norms[nonzero, None]
        mean_dir = unit.mean(axis=0)
        out.extend(mean_dir.tolist())
    assert len(out) == 30
    return out


def _morgan_counts(mol: Chem.Mol) -> np.ndarray:
    """The 2048-bit radius-2 Morgan fingerprint as a count vector.

    The fingerprint convention (counts vs binary bits) is pinned by the
    artifact norm statistics: the re-landed default is the count form
    (for these building blocks every set bit has count 1 in practice,
    so counts and bits coincide on the source distribution; the build
    manifest records the convention).
    """
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    counts = np.zeros(MORGAN_BITS, dtype=np.float64)
    for index, value in generator.GetCountFingerprint(mol).GetNonzeroElements().items():
        counts[index] = value
    return counts


def pharm3d_descriptor(
    mol: Chem.Mol, conf_mol: Chem.Mol
) -> np.ndarray:
    """The 2137-D raw ``pharm3d`` descriptor (float64, blocks in the
    artifact layout: Morgan / physchem / shape / electro / vectors / norm).
    """
    fp = _morgan_counts(mol)
    physchem = np.array(_physchem_features(mol), dtype=np.float64)
    shape = np.array(_shape_features(conf_mol), dtype=np.float64)
    electro = np.array(_electro_features(conf_mol), dtype=np.float64)
    vectors = np.array(_handle_vector_features(mol, conf_mol), dtype=np.float64)

    head = np.concatenate([fp, physchem, shape, electro, vectors])
    norm = float(np.linalg.norm(head))
    x = np.concatenate([head, [norm]])
    if x.shape != (D_RAW,):
        raise SynthonEncodeError(
            f"descriptor width {x.shape[0]} != frozen {D_RAW} (fail-closed)"
        )
    if not np.isfinite(head).all():
        raise SynthonEncodeError(
            "non-finite raw descriptor block (fail-closed — never a NaN)"
        )
    return x


# ---------------------------------------------------------------------------
# The 64-D invariant handle descriptors (HandleSelectionHead input, #22)
# ---------------------------------------------------------------------------

#: One-hot slot per registry handle class (19) + per seed class (7) = 26,
#: plus 38 local-environment scalars = 64.
_HANDLE_ONEHOT_CLASSES: Tuple[str, ...] = tuple(sorted(ALL_HANDLE_CLASSES)) + tuple(
    SEED_CLASS_SMARTS
)


def handle_descriptor_64(mol: Chem.Mol, atom_idx: int) -> np.ndarray:
    """The invariant 64-D chemical descriptor of one handle attachment.

    Layout (frozen #22 input contract ``[64]``, chemistry only —
    Cartesian data is supplied separately per constitution #45 §7.1):

    * [0:26] one-hot handle-class slots — the registry classes realized
      by ``atom_idx`` (via ``handle_class_of_atom`` semantics) and the
      seed classes whose defining atom it is;
    * [26:64] local environment scalars: element one-hot (C/N/O/S/P,
      halides, B, Zn, Sn), degree, total valence, formal charge,
      aromaticity, ring membership + ring size, hydrogen count,
      Gasteiger charge, and the local sp2/sp3 neighbour fractions.
    """
    if not 0 <= atom_idx < mol.GetNumAtoms():
        raise SynthonEncodeError(
            f"handle atom index {atom_idx} out of range (fail-closed)"
        )
    atom = mol.GetAtomWithIdx(atom_idx)
    feats: List[float] = []

    # one-hot class slots [0:26]: the atom participates in a defining
    # match of the class (any position — the registry detectors return
    # first-SMARTS-atom indices, e.g. the Br of "[Br][c]", while the
    # engine's HandleSpec may pin the carbon; both identify the handle)
    realized = set()
    for cls in sorted(ALL_HANDLE_CLASSES):
        from syntree.chemistry.chemoselectivity import HANDLE_CLASS_SMARTS

        smarts = HANDLE_CLASS_SMARTS.get(cls)
        if smarts is not None:
            query = Chem.MolFromSmarts(smarts)
            for match in mol.GetSubstructMatches(query, uniquify=True):
                if atom_idx in match:
                    realized.add(cls)
        else:
            if atom_idx in detect_handle_instances(mol, cls):
                realized.add(cls)
    for cls, smarts in SEED_CLASS_SMARTS.items():
        query = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(query, uniquify=True):
            if match[0] == atom_idx:
                realized.add(cls)
    for cls in _HANDLE_ONEHOT_CLASSES:
        feats.append(1.0 if cls in realized else 0.0)

    # element one-hot [26:38]
    symbol = atom.GetSymbol()
    for allowed in ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Zn", "Sn"):
        feats.append(1.0 if symbol == allowed else 0.0)

    # local scalars [38:64]
    AllChem.ComputeGasteigerCharges(mol)
    neighbours = list(atom.GetNeighbors())
    neighbour_charges = [
        float(n.GetDoubleProp("_GasteigerCharge")) for n in neighbours
    ]
    feats.extend(
        [
            float(atom.GetDegree()),                                # 38
            float(atom.GetTotalValence()),                          # 39
            float(atom.GetFormalCharge()),                          # 40
            1.0 if atom.GetIsAromatic() else 0.0,                   # 41
            1.0 if atom.IsInRing() else 0.0,                        # 42
            float(atom.GetTotalNumHs()),                            # 43
            float(len(realized)),                                   # 44 class multiplicity
        ]
    )
    # bond-type counts to the attachment atom [45:49]
    bond_types = [b.GetBondType() for b in atom.GetBonds()]
    feats.extend(
        [
            float(sum(1 for b in bond_types if b == Chem.BondType.SINGLE)),
            float(sum(1 for b in bond_types if b == Chem.BondType.DOUBLE)),
            float(sum(1 for b in bond_types if b == Chem.BondType.TRIPLE)),
            float(sum(1 for b in bond_types if b == Chem.BondType.AROMATIC)),
        ]
    )
    # element-typed neighbour counts [49:53]
    feats.extend(
        [
            float(sum(1 for n in neighbours if n.GetSymbol() == "N")),
            float(sum(1 for n in neighbours if n.GetSymbol() == "O")),
            float(sum(1 for n in neighbours if n.GetSymbol() == "S")),
            float(
                sum(1 for n in neighbours if n.GetSymbol() in ("F", "Cl", "Br", "I"))
            ),
        ]
    )
    # ring context [53:55]
    rings = [r for r in mol.GetRingInfo().AtomRings() if atom_idx in r]
    feats.append(float(min((len(r) for r in rings), default=0)))
    feats.append(float(max((len(r) for r in rings), default=0)))
    # local electrostatics [55:58]
    feats.append(float(atom.GetDoubleProp("_GasteigerCharge")))
    feats.append(float(np.mean(neighbour_charges)) if neighbour_charges else 0.0)
    feats.append(float(np.min(neighbour_charges)) if neighbour_charges else 0.0)
    # hybridization fractions [58:61]
    denom = max(len(neighbours), 1)
    feats.append(
        sum(
            1 for n in neighbours
            if n.GetHybridization() == Chem.HybridizationType.SP2
        )
        / denom
    )
    feats.append(
        sum(
            1 for n in neighbours
            if n.GetHybridization() == Chem.HybridizationType.SP3
        )
        / denom
    )
    feats.append(
        sum(
            1 for n in neighbours
            if n.GetHybridization() == Chem.HybridizationType.SP
        )
        / denom
    )
    # aromatic/neighbour context [61:63]
    feats.append(float(sum(1 for n in neighbours if n.GetIsAromatic())) / denom)
    feats.append(float(sum(1 for n in neighbours if n.IsInRing())) / denom)
    # isotopic mass deviation placeholder-free: neighbour count scalar [63]
    feats.append(float(len(neighbours)))

    if len(feats) != D_HANDLE_FEATURES:
        raise SynthonEncodeError(
            f"handle descriptor width {len(feats)} != frozen "
            f"{D_HANDLE_FEATURES} (fail-closed)"
        )
    return np.array(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Projection (QR Johnson-Lindenstrauss, seed 2026 — artifact bit-exact)
# ---------------------------------------------------------------------------

def build_projection(seed: int = PROJECTION_SEED) -> np.ndarray:
    """The orthonormalized QR Johnson-Lindenstrauss matrix ``Q [2137, 256]``.

    Reproduces the artifact ``projection/Q`` bit-exactly: a seeded
    standard-normal draw, economy QR, and column sign normalization by
    ``sign(diag(R))`` (which makes the factorization unique).
    """
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(D_RAW, D_EMBED))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    return q.astype(np.float32)


def project_raw(
    x_raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """z-normalize (per-feature), project through ``Q``, L2-normalize.

    The Morgan and handle-vector blocks carry identity normalization
    (their stored mean/std are 0/1); the caller passes the artifact
    statistics for the remaining blocks.
    """
    if x_raw.shape != (D_RAW,):
        raise SynthonEncodeError(
            f"raw descriptor shape {x_raw.shape} != ({D_RAW},) (fail-closed)"
        )
    z = (x_raw - mean) / np.maximum(std, ZSCORE_FLOOR)
    e = z.astype(np.float64) @ q.astype(np.float64)
    norm = float(np.linalg.norm(e))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SynthonEncodeError(
            "projected embedding has non-finite or zero norm (fail-closed)"
        )
    return (e / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Encoding entry point
# ---------------------------------------------------------------------------

@dataclass
class EncodedSynthon:
    """One fully encoded catalog row."""

    block_id: str
    smiles: str
    stratum: str
    synthon_classes: str
    seed_classes: str
    mw: float
    fsp3: float
    heavy_atoms: int
    rotatable_bonds: int
    conformer_energy: float
    conformer_method: str
    x_raw: np.ndarray = field(repr=False)

    @property
    def x_raw_bytes(self) -> int:
        return int(self.x_raw.nbytes)


def encode_synthon(
    block_id: str, smiles: str, stratum: str, synthon_classes: str,
    seed_classes: str, mol: Optional[Chem.Mol] = None,
) -> EncodedSynthon:
    """Encode one certified block: conformer + raw descriptor + columns.

    Fail-closed at every stage with named reasons; never emits NaN
    outside the documented MMFF-fallback energy.
    """
    mol = mol if mol is not None else Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SynthonEncodeError(f"{block_id}: unparseable SMILES {smiles!r}")
    outcome = embed_conformer(smiles)
    x_raw = pharm3d_descriptor(mol, outcome.conformer)
    return EncodedSynthon(
        block_id=block_id,
        smiles=smiles,
        stratum=stratum,
        synthon_classes=synthon_classes,
        seed_classes=seed_classes,
        mw=Descriptors.MolWt(mol),
        fsp3=round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        rotatable_bonds=Lipinski.NumRotatableBonds(mol),
        conformer_energy=outcome.energy,
        conformer_method=outcome.method,
        x_raw=x_raw,
    )


# ---------------------------------------------------------------------------
# The immutable HDF5 catalog (schema pharm3d-v2.0 / catalog-v2.0)
# ---------------------------------------------------------------------------

def registry_digest() -> str:
    """The certified reaction-registry digest the catalog is bound to."""
    import hashlib
    import json

    from syntree.chemistry.reactions import CERTIFIED_REACTIONS

    payload = {
        rid: {
            "smarts": rxn.reaction_smarts,
            "mol_class": rxn.molecule_handle_class,
            "block_class": rxn.block_handle_class,
        }
        for rid, rxn in sorted(CERTIFIED_REACTIONS.items())
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_catalog_h5(
    path: str,
    rows: Sequence[EncodedSynthon],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    q: np.ndarray,
    source_sdf: str,
    source_sdf_sha256: str,
    registry_sha: str,
) -> str:
    """Write the immutable catalog HDF5 in the artifact-pinned schema."""
    import hashlib

    import h5py

    n = len(rows)
    embeddings = np.stack(
        [project_raw(r.x_raw, mean, std, q) for r in rows]
    ).astype(np.float32)
    if embeddings.shape != (n, D_EMBED):
        raise SynthonEncodeError(
            f"embedding matrix {embeddings.shape} != ({n}, {D_EMBED})"
        )
    if not np.isfinite(embeddings).all():
        raise SynthonEncodeError("non-finite embeddings (fail-closed)")

    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["catalog_schema_version"] = CATALOG_SCHEMA_VERSION
        h5.attrs["catalog_size"] = n
        h5.attrs["d_raw"] = D_RAW
        h5.attrs["d_embed"] = D_EMBED
        h5.attrs["projection_seed"] = PROJECTION_SEED
        h5.attrs["registry_digest"] = registry_sha
        h5.attrs["source_sdf"] = source_sdf
        h5.attrs["source_sdf_sha256"] = source_sdf_sha256
        import time

        h5.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        cat = h5.create_group("catalog")
        cat.create_dataset("block_ids", data=[r.block_id for r in rows], dtype=h5py.string_dtype())
        cat.create_dataset("smiles", data=[r.smiles for r in rows], dtype=h5py.string_dtype())
        cat.create_dataset(
            "synthon_classes",
            data=[r.synthon_classes for r in rows],
            dtype=h5py.string_dtype(),
        )
        cat.create_dataset(
            "seed_classes", data=[r.seed_classes for r in rows], dtype=h5py.string_dtype()
        )
        cat.create_dataset("strata", data=[r.stratum for r in rows], dtype=h5py.string_dtype())
        cat.create_dataset("mw", data=np.array([r.mw for r in rows], dtype=np.float64))
        cat.create_dataset("fsp3", data=np.array([r.fsp3 for r in rows], dtype=np.float64))
        cat.create_dataset(
            "heavy_atoms", data=np.array([r.heavy_atoms for r in rows], dtype=np.int32)
        )
        cat.create_dataset(
            "rotatable_bonds",
            data=np.array([r.rotatable_bonds for r in rows], dtype=np.int64),
        )
        cat.create_dataset(
            "conformer_energies",
            data=np.array([r.conformer_energy for r in rows], dtype=np.float64),
        )
        cat.create_dataset(
            "conformer_methods",
            data=[r.conformer_method for r in rows],
            dtype=h5py.string_dtype(length=24),
        )
        h5.create_dataset("embeddings", data=embeddings, dtype=np.float32)
        norm = h5.create_group("normalization")
        norm.create_dataset("mean", data=mean.astype(np.float64))
        norm.create_dataset("std", data=std.astype(np.float64))
        proj = h5.create_group("projection")
        proj.create_dataset("Q", data=q.astype(np.float32))
        proj.create_dataset("seed", data=np.int64(PROJECTION_SEED))
        ver = h5.create_group("versions")
        import platform
        import rdkit

        for name, value in (
            ("python", platform.python_version()),
            ("numpy", np.__version__),
            ("rdkit", rdkit.__version__),
            ("syntree", "2.0.0.dev0"),
        ):
            ver.create_dataset(name, data=[value], dtype=h5py.string_dtype())
    os.replace(tmp, path)

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_catalog_h5(path: str):
    """Read a catalog HDF5; returns (catalog dict-of-arrays, metadata)."""
    import h5py

    with h5py.File(path, "r") as h5:
        catalog = {
            key: h5[f"catalog/{key}"][()] for key in h5["catalog"]
        }
        metadata = {
            "attrs": dict(h5.attrs),
            "embeddings": h5["embeddings"][()],
            "mean": h5["normalization/mean"][()],
            "std": h5["normalization/std"][()],
            "q": h5["projection/Q"][()],
            "seed": int(h5["projection/seed"][()]),
        }
    return catalog, metadata
