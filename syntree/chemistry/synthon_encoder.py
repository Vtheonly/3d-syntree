"""Synthon embedding encoders (tasklist Priority 1).

The legacy ``morgan2d`` encoder embeds each building block as a seeded random
projection of its 2D Morgan bit fingerprint. It is chemically blind to
everything the tasklist criticises: 3D shape/volume, electrostatic surface,
and hydrogen-bond vector directions.

The default ``pharm3d`` encoder computes, fully offline and deterministically
(no external model downloads - essential for the offline RTX 6000 / Kaggle
environment):

* a 3D conformer per synthon (seeded ETKDG + MMFF94 relaxation, UFF fallback),
* Gasteiger partial charges -> charge statistics and dipole moment,
* 3D shape descriptors (radius of gyration, asphericity, eccentricity,
  inertial-shape factor, NPR1/NPR2 principal-axis ratios),
* pharmacophore counts (donor / acceptor / positive / negative / hydrophobic /
  aromatic),
* per-handle-type mean attachment direction vectors (handle atom -> molecular
  centroid, unit vectors averaged over instances),
* classical 2D descriptors (MW, TPSA, clogP, Fsp3, rotatable bonds, rings,
  HBD/HBA, element histogram, handle presence/counts) and the Morgan
  fingerprint (unit-normalised so 2D similarity does not drown the 3D block).

The concatenated raw descriptor is projected to ``embedding_dim`` with a
seeded *orthonormal* (QR-factorised) random matrix - a proper
Johnson-Lindenstrauss projection - and L2-normalised, exactly like the
legacy encoder's output contract. Everything is deterministic given
``(smiles, seed)`` so embeddings are reproducible across runs and machines,
and the expensive 3D conformer pass is cached on disk via
:class:`syntree.data.synthon_library.SynthonLibraryStore` (gracefully
skipping the cache when the catalog directory is read-only, e.g. Kaggle's
``/kaggle/input`` mount).

A truly *learned* foundation encoder (Uni-Mol / MACE style) requires
pre-trained weights that cannot be fetched in the offline environment; the
``pharm3d`` descriptor is the strongest deterministic offline upgrade and is
a drop-in input for any future learned projection because the raw descriptor
layout is versioned (``ENCODER_VERSION``).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
    rdPartialCharges,
)

from syntree.chemistry.reactions import HANDLE_NAMES, ReactionEngine

logger = logging.getLogger(__name__)

# Encoder registry. "morgan2d" reproduces the legacy projection bit-for-bit.
ENCODER_VERSION = "pharm3d-v1"
SUPPORTED_ENCODERS = ("pharm3d", "morgan2d")

_MORGAN_BITS = 2048
_MORGAN_RADIUS = 2

# ---------------------------------------------------------------------------
# Fixed descriptor layout (documented contract; versioned by ENCODER_VERSION)
# ---------------------------------------------------------------------------
# 2D engineered block (47 dims):
#   [ 0, 10)  element histogram / heavy atoms (C,N,O,S,F,Cl,Br,I,P,other)
#   [10, 21)  scalar descriptors:
#               mw/250, tpsa/100, clogp/5, fsp3, rotatable/10, heavy/30,
#               rings/4, aromatic rings/3, hbd/5, hba/10, |formal charge|/2
#   [21, 27)  pharmacophore fractions (per heavy atom):
#               donor, acceptor, positive, negative, hydrophobe, aromatic
#   [27, 37)  handle presence one-hot (grammar order)
#   [37, 47)  handle instance counts / 3 (grammar order)
# 3D block (42 dims, all zero when no conformer could be generated):
#   [47, 53)  shape: rgyr/4, asphericity, eccentricity,
#               inertial shape factor, npr1, npr2
#   [53, 54)  dipole magnitude / 8 Debye
#   [54, 58)  Gasteiger charge stats (mean, std, min, max)
#   [58, 88)  per-handle mean attachment direction (10 handles x 3 xyz),
#               zero vector when the handle type is absent
#   [88, 89)  vdw sphere volume / 300 A^3
# Magnitude scalar (1 dim): engineered-block norm / 3 (clipped), appended so
# unit-normalisation does not discard molecular size information.
_ELEMENTS = (6, 7, 8, 16, 9, 17, 35, 53, 15)  # C N O S F Cl Br I P (+other)
_N_ELEMENTS = len(_ELEMENTS) + 1               # 10
_ELEMENT_OFFSET = 0
_SCALAR_OFFSET = _ELEMENT_OFFSET + _N_ELEMENTS          # 10
_N_SCALARS = 11                                          # -> 21
_PHARM_OFFSET = _SCALAR_OFFSET + _N_SCALARS              # 21
_N_PHARM = 6                                             # -> 27
_HANDLE_PRESENCE_OFFSET = _PHARM_OFFSET + _N_PHARM       # 27
_N_HANDLES = len(HANDLE_NAMES)                           # 10 -> 37
_HANDLE_COUNT_OFFSET = _HANDLE_PRESENCE_OFFSET + _N_HANDLES  # 37
_2D_ENGINEERED_DIM = _HANDLE_COUNT_OFFSET + _N_HANDLES   # 47
_SHAPE_OFFSET = _2D_ENGINEERED_DIM                       # 47
_N_SHAPE = 6                                             # -> 53
_DIPOLE_OFFSET = _SHAPE_OFFSET + _N_SHAPE                # 53
_CHARGE_OFFSET = _DIPOLE_OFFSET + 1                      # 54
_N_CHARGE = 4                                            # -> 58
_HANDLE_DIR_OFFSET = _CHARGE_OFFSET + _N_CHARGE          # 58
_VOLUME_OFFSET = _HANDLE_DIR_OFFSET + _N_HANDLES * 3     # 88
_ENGINEERED_DIM = _VOLUME_OFFSET + 1                     # 89
_RAW_DIM = _MORGAN_BITS + _ENGINEERED_DIM + 1            # 2138

# Debye conversion for the Gasteiger point-charge dipole (e * Angstrom).
_DEBYE_PER_E_ANGSTROM = 4.80320

# SMARTS for coarse pharmacophore typing (dependency-light, deterministic).
# Acceptor = heteroatom without H (pure acceptor) or aromatic N.
_PHARMACOPHORE_SMARTS = {
    "donor": "[N,O,S;H1,H2]",
    "acceptor": "[N,O,S;H0,n]",
    "positive": "[+]",
    "negative": "[-]",
    "hydrophobe": "[#6;!$(C[N,O,S,F,Cl,Br,I])]",
    "aromatic": "[a]",
}
_PHARM_PATTERNS = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in _PHARMACOPHORE_SMARTS.items()
}

# vdW radii (Angstrom) for the sphere-volume estimate.
_VDW_RADII = {1: 1.10, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80,
              16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98}


def _vdw_volume(mol: Chem.Mol) -> float:
    total = 0.0
    for atom in mol.GetAtoms():
        r = _VDW_RADII.get(atom.GetAtomicNum(), 1.8)
        total += 4.0 / 3.0 * np.pi * r ** 3
    return float(total)


def _molecule_seed(seed: int, smiles: str) -> int:
    """Deterministic per-molecule conformer seed.

    Derived from the catalog seed and the SMILES content (crc32 - stable
    across processes, unlike Python's ``hash``), so identical molecules
    always receive identical 3D descriptors regardless of their row
    position, while different molecules explore different conformer seeds.
    """
    import zlib

    return (int(seed) * 1000003 + zlib.crc32(str(smiles).encode("utf-8"))) % 2147483647


def _clip(value: float, lo: float = -3.0, hi: float = 3.0) -> float:
    return float(min(max(value, lo), hi))


def _embed_conformer(mol_with_h: Chem.Mol, seed: int) -> Optional[Chem.Mol]:
    """Deterministically embed + force-field relax a molecule.

    Returns the *heavy-atom* molecule with the optimized conformer attached,
    or ``None`` when embedding failed (the caller falls back to a 2D-only
    descriptor). ETKDG with a fixed seed and MMFF94 are both deterministic
    given the same RDKit build.
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed % 2147483647)
    params.useRandomCoords = False
    try:
        status = AllChem.EmbedMolecule(mol_with_h, params)
    except Exception:
        status = -1
    if status != 0 or mol_with_h.GetNumConformers() == 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_with_h):
            AllChem.MMFFOptimizeMolecule(mol_with_h, mmffVariant="MMFF94", maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=500)
    except Exception:
        pass  # keep the ETKDG geometry; still deterministic
    heavy = Chem.RemoveHs(mol_with_h)
    return heavy if heavy.GetNumConformers() > 0 else None


def _morgan_fingerprint(mol: Chem.Mol) -> np.ndarray:
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS
        )
        fp = generator.GetFingerprint(mol)
    except (AttributeError, NameError):
        from rdkit import DataStructs

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=_MORGAN_RADIUS, nBits=_MORGAN_BITS)
        vec = np.zeros(_MORGAN_BITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, vec)
        return vec
    vec = np.zeros(_MORGAN_BITS, dtype=np.float32)
    for bit in fp.GetOnBits():
        vec[bit] = 1.0
    return vec


def _heavy_centroid(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    pos = np.array(
        [list(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()], dtype=np.float64
    )
    return pos.mean(axis=0)


def compute_pharm3d_descriptor(
    smiles: str,
    seed: int,
    engine: Optional[ReactionEngine] = None,
) -> Tuple[np.ndarray, bool]:
    """Raw pharm3d descriptor for one synthon.

    Returns ``(descriptor[_RAW_DIM], has_3d)``. The descriptor concatenates
    ``[unit-normalised Morgan | z-scaled engineered 2D+3D block | magnitude
    scalar]``. When conformer generation fails the 3D sub-block is zero and
    ``has_3d`` is False (deterministic, never NaN).
    """
    desc = np.zeros(_RAW_DIM, dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return desc, False

    engine = engine or ReactionEngine()
    heavy_atoms = max(1, mol.GetNumAtoms())

    # ---- Morgan block (unit-normalised) --------------------------------
    morgan = _morgan_fingerprint(mol)
    morgan_norm = float(np.linalg.norm(morgan))
    if morgan_norm > 0:
        morgan = morgan / morgan_norm

    # ---- engineered 2D block -------------------------------------------
    eng = np.zeros(_ENGINEERED_DIM, dtype=np.float64)

    hist = np.zeros(_N_ELEMENTS, dtype=np.float64)
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        slot = _ELEMENTS.index(z) if z in _ELEMENTS else _N_ELEMENTS - 1
        hist[slot] += 1.0
    eng[_ELEMENT_OFFSET:_ELEMENT_OFFSET + _N_ELEMENTS] = hist / heavy_atoms

    formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    scalars = [
        Descriptors.MolWt(mol) / 250.0,
        Descriptors.TPSA(mol) / 100.0,
        Descriptors.MolLogP(mol) / 5.0,
        Lipinski.FractionCSP3(mol),
        Lipinski.NumRotatableBonds(mol) / 10.0,
        mol.GetNumAtoms() / 30.0,
        mol.GetRingInfo().NumRings() / 4.0,
        rdMolDescriptors.CalcNumAromaticRings(mol) / 3.0,
        Lipinski.NumHDonors(mol) / 5.0,
        Lipinski.NumHAcceptors(mol) / 10.0,
        abs(formal_charge) / 2.0,
    ]
    eng[_SCALAR_OFFSET:_SCALAR_OFFSET + _N_SCALARS] = [
        _clip(v) for v in scalars
    ]

    for i, (name, pattern) in enumerate(_PHARM_PATTERNS.items()):
        if pattern is None:
            continue
        count = len(mol.GetSubstructMatches(pattern, uniquify=True))
        eng[_PHARM_OFFSET + i] = _clip(count / heavy_atoms)

    handle_infos = engine.detect_handles(mol)
    per_type: Dict[str, int] = {}
    for info in handle_infos:
        per_type[info.handle_type] = per_type.get(info.handle_type, 0) + 1
    for i, handle_name in enumerate(HANDLE_NAMES):
        count = per_type.get(handle_name, 0)
        eng[_HANDLE_PRESENCE_OFFSET + i] = 1.0 if count > 0 else 0.0
        eng[_HANDLE_COUNT_OFFSET + i] = _clip(count / 3.0)

    # ---- engineered 3D block -------------------------------------------
    has_3d = False
    try:
        mol_h = Chem.AddHs(Chem.Mol(mol))
        relaxed = _embed_conformer(mol_h, _molecule_seed(seed, smiles))
    except Exception:
        relaxed = None
    if relaxed is not None:
        has_3d = True
        try:
            shape = [
                rdMolDescriptors.CalcRadiusOfGyration(relaxed) / 4.0,
                rdMolDescriptors.CalcAsphericity(relaxed),
                rdMolDescriptors.CalcEccentricity(relaxed),
                rdMolDescriptors.CalcInertialShapeFactor(relaxed),
                rdMolDescriptors.CalcNPR1(relaxed),
                rdMolDescriptors.CalcNPR2(relaxed),
            ]
            eng[_SHAPE_OFFSET:_SHAPE_OFFSET + _N_SHAPE] = [_clip(v) for v in shape]
        except Exception:
            pass

        try:
            # Gasteiger charges on the heavy-atom copy of the relaxed
            # geometry: deterministic point-charge electrostatics.
            charged = Chem.Mol(relaxed)
            rdPartialCharges.ComputeGasteigerCharges(charged)
            q_heavy = np.array(
                [float(a.GetDoubleProp("_GasteigerCharge")) for a in charged.GetAtoms()],
                dtype=np.float64,
            )
            q_heavy = np.nan_to_num(q_heavy, nan=0.0, posinf=0.0, neginf=0.0)
            eng[_CHARGE_OFFSET] = _clip(float(q_heavy.mean()))
            eng[_CHARGE_OFFSET + 1] = _clip(float(q_heavy.std()))
            eng[_CHARGE_OFFSET + 2] = _clip(float(q_heavy.min()))
            eng[_CHARGE_OFFSET + 3] = _clip(float(q_heavy.max()))

            # Dipole from heavy-atom charges at the relaxed geometry
            # (e * Angstrom -> Debye). Deterministic, well-defined
            # approximation of the molecular dipole.
            conf = relaxed.GetConformer()
            pos = np.array(
                [list(conf.GetAtomPosition(a.GetIdx())) for a in relaxed.GetAtoms()],
                dtype=np.float64,
            )
            dipole_ea = float(np.linalg.norm((q_heavy[:, None] * pos).sum(axis=0)))
            eng[_DIPOLE_OFFSET] = _clip(dipole_ea * _DEBYE_PER_E_ANGSTROM / 8.0)
        except Exception:
            pass

        try:
            centroid = _heavy_centroid(relaxed)
            dirs = {name: [] for name in HANDLE_NAMES}
            for info in handle_infos:
                if info.handle_type not in dirs:
                    continue
                atom_idx = info.primary_atom
                if atom_idx >= relaxed.GetNumAtoms():
                    continue
                p = np.array(list(conf.GetAtomPosition(atom_idx)), dtype=np.float64)
                v = p - centroid
                norm = float(np.linalg.norm(v))
                if norm > 1e-8:
                    dirs[info.handle_type].append(v / norm)
            for i, name in enumerate(HANDLE_NAMES):
                vectors = dirs[name]
                if vectors:
                    mean_dir = np.mean(np.stack(vectors), axis=0)
                    norm = float(np.linalg.norm(mean_dir))
                    if norm > 1e-8:
                        mean_dir = mean_dir / norm
                    base = _HANDLE_DIR_OFFSET + 3 * i
                    eng[base:base + 3] = mean_dir
        except Exception:
            pass

        try:
            eng[_VOLUME_OFFSET] = _clip(_vdw_volume(relaxed) / 300.0)
        except Exception:
            pass

    # ---- assemble --------------------------------------------------------
    eng = np.nan_to_num(eng, nan=0.0, posinf=0.0, neginf=0.0)
    eng_norm = float(np.linalg.norm(eng))
    if eng_norm > 1e-8:
        eng_unit = eng / eng_norm
    else:
        eng_unit = eng
    magnitude = _clip(eng_norm / 3.0)

    desc[:_MORGAN_BITS] = morgan
    desc[_MORGAN_BITS:_MORGAN_BITS + _ENGINEERED_DIM] = eng_unit
    desc[_MORGAN_BITS + _ENGINEERED_DIM] = magnitude
    return np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), has_3d


def build_projection(raw_dim: int, embedding_dim: int, seed: int) -> np.ndarray:
    """Seeded orthonormal (QR) projection matrix ``[raw_dim, embedding_dim]``.

    The QR factor gives orthonormal columns (exact-norm-preserving on the
    column space) which are then scaled by ``sqrt(raw_dim / embedding_dim)``
    so the map is a proper Johnson-Lindenstrauss embedding: pairwise
    distances are preserved in expectation, ``E[||P x||^2] = ||x||^2``.
    Because the catalog L2-normalises every embedding row afterwards, the
    overall scale is cosmetic - it makes projected norms interpretable and
    the distance-preservation guarantee exact in expectation. Falls back to
    a scaled Gaussian when ``raw_dim`` is too small for a tall QR
    factorisation.
    """
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((raw_dim, embedding_dim)).astype(np.float64)
    if raw_dim >= embedding_dim and embedding_dim > 0:
        try:
            q, _ = np.linalg.qr(gaussian)
            jl_scale = float(np.sqrt(raw_dim / embedding_dim))
            return (q * jl_scale).astype(np.float32)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate dims
            pass
    projection = gaussian / np.sqrt(embedding_dim)
    return projection.astype(np.float32)


def encode_catalog_pharm3d(
    smiles_iter: Sequence[str],
    embedding_dim: int,
    seed: int,
    engine: Optional[ReactionEngine] = None,
    progress: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Encode a whole catalog with the pharm3d descriptor + QR projection.

    Returns ``(embeddings [K, d] float32 L2-normalised, info)`` where ``info``
    records the encoder name/version, how many synthons obtained a 3D
    conformer, and the projection seed - useful for cache signatures and
    logging.
    """
    engine = engine or ReactionEngine()
    raw = np.zeros((len(smiles_iter), _RAW_DIM), dtype=np.float32)
    n_3d = 0
    for i, smiles in enumerate(smiles_iter):
        descriptor, has_3d = compute_pharm3d_descriptor(str(smiles), seed, engine)
        raw[i] = descriptor
        if has_3d:
            n_3d += 1
        if progress and (i + 1) % 5000 == 0:
            logger.info("pharm3d encoder: %d/%d synthons (%d with 3D)", i + 1, len(smiles_iter), n_3d)

    projection = build_projection(_RAW_DIM, embedding_dim, seed)
    embedded = raw @ projection
    norms = np.linalg.norm(embedded, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embeddings = torch.from_numpy((embedded / norms).astype(np.float32))
    info = {
        "encoder": "pharm3d",
        "encoder_version": ENCODER_VERSION,
        "raw_dim": _RAW_DIM,
        "projection_seed": int(seed),
        "num_with_conformer": int(n_3d),
        "num_synthons": int(len(smiles_iter)),
    }
    return embeddings, info


def encode_catalog_morgan2d(
    smiles_iter: Sequence[str],
    embedding_dim: int,
    seed: int,
) -> torch.Tensor:
    """Legacy encoder: seeded Gaussian projection of Morgan bit fingerprints.

    Kept bit-for-bit compatible with the historical ``_compute_embeddings``
    implementation so old experiments remain reproducible when the catalog is
    configured with ``encoder="morgan2d"``.
    """
    from rdkit import DataStructs

    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((2048, embedding_dim)).astype(np.float32)
    projection /= np.sqrt(2048)

    rows = np.zeros((len(smiles_iter), embedding_dim), dtype=np.float32)
    for i, smiles in enumerate(smiles_iter):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        try:
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=_MORGAN_RADIUS, fpSize=_MORGAN_BITS
            )
            fp = generator.GetFingerprint(mol)
        except (AttributeError, NameError):
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=_MORGAN_RADIUS, nBits=_MORGAN_BITS
            )
        vec = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, vec)
        rows[i] = vec @ projection

    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return torch.from_numpy(rows / norms)


def catalog_ids_hash(ids: Sequence[str]) -> str:
    """Stable hash of the catalog id list for cache/checkpoint signatures."""
    payload = "\n".join(str(i) for i in ids)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "ENCODER_VERSION",
    "SUPPORTED_ENCODERS",
    "RAW_DIM",
    "ENGINEERED_DIM",
    "compute_pharm3d_descriptor",
    "encode_catalog_pharm3d",
    "encode_catalog_morgan2d",
    "build_projection",
    "catalog_ids_hash",
]

# Public aliases of the fixed layout constants.
RAW_DIM = _RAW_DIM
ENGINEERED_DIM = _ENGINEERED_DIM
