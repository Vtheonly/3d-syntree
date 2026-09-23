"""Enamine 3D synthon catalog: loading, filtering, and reaction-grammar
masking.

The catalog enforces the curated Enamine REAL 3D-Diversity constraints:

* fraction of sp3 carbons (Fsp3) >= ``min_fsp3`` (default 0.42),
* molecular weight <= ``max_mw`` Da (default 220),

with one chemically necessary exemption: aryl halides and boronic acids
(the mandatory coupling partners for Suzuki / Buchwald-Hartwig / SNAr
chemistry, whose reacting atoms are aromatic by definition) are exempt
from the Fsp3 floor. Without them the reaction grammar could not close.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

from syntree.chemistry.reactions import (
    REACTION_FAMILY_MEMBERS,
    REACTION_FAMILY_NAMES,
    REACTION_SIDES,
    HANDLE_SMARTS,
    ReactionEngine,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("id", "smiles", "fsp3", "mw", "primary_handle")

# Default catalog constraints (Enamine 3D-Diversity curation rules).
DEFAULT_MIN_FSP3 = 0.42
DEFAULT_MAX_MW = 220.0

# Handle types exempt from the Fsp3 floor (aryl coupling partners are
# necessarily flat: [c]-X bonds are required by the reaction SMARTS).
DEFAULT_EXEMPT_HANDLES = ("aryl_halide", "boronic_acid")


class SynthonCatalog:
    """Manages the certified 3D building-block library.

    Args:
        catalog_path: Path to a parquet or CSV catalog with columns
            ``id, smiles, fsp3, mw, primary_handle``.
        embedding_dim: Dimensionality of the synthon embedding table.
        min_fsp3: Minimum allowed fraction of sp3 carbons.
        max_mw: Maximum allowed molecular weight (Da).
        seed: Seed for the deterministic fingerprint projection.
        validate_handles: When True, re-detect handles from SMILES and drop
            rows whose ``primary_handle`` cannot be verified chemically.
        exempt_handles: Handle types allowed to bypass the Fsp3 floor
            (aryl coupling partners).
    """

    def __init__(
        self,
        catalog_path: str,
        embedding_dim: int = 128,
        min_fsp3: float = DEFAULT_MIN_FSP3,
        max_mw: float = DEFAULT_MAX_MW,
        seed: int = 42,
        validate_handles: bool = True,
        exempt_handles: Sequence[str] = DEFAULT_EXEMPT_HANDLES,
    ):
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.catalog_path = str(catalog_path)
        self.embedding_dim = int(embedding_dim)
        self.min_fsp3 = float(min_fsp3)
        self.max_mw = float(max_mw)
        self.seed = int(seed)
        self.exempt_handles = tuple(exempt_handles or ())

        self.engine = ReactionEngine()
        self.df = self._load_and_filter(validate_handles)
        self.num_synthons = len(self.df)
        if self.num_synthons == 0:
            raise ValueError(
                f"Catalog '{catalog_path}' is empty after filtering "
                f"(Fsp3 >= {min_fsp3}, MW <= {max_mw})."
            )

        self.embeddings = self._compute_embeddings()
        self.handle_masks, self._handle_counts = self._build_compatibility_indices()
        self._canonical_smiles_index = self._build_canonical_smiles_index()
        self._family_mask_cache: Dict[Tuple[str, Optional[str]], torch.Tensor] = {}
        self._smiles_cache: Dict[int, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_and_filter(self, validate_handles: bool) -> pd.DataFrame:
        if self.catalog_path.endswith(".parquet"):
            df = pd.read_parquet(self.catalog_path)
        elif self.catalog_path.endswith(".csv"):
            df = pd.read_csv(self.catalog_path)
        else:
            df = pd.read_csv(self.catalog_path)

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Catalog must contain columns: {sorted(REQUIRED_COLUMNS)}")

        n_raw = len(df)
        is_exempt = df["primary_handle"].isin(self.exempt_handles)
        passes_fsp3 = (df["fsp3"] >= self.min_fsp3) | is_exempt
        filtered = df[passes_fsp3 & (df["mw"] <= self.max_mw)].copy()
        n_filtered = len(filtered)

        dropped = []
        if validate_handles:
            keep: List[bool] = []
            for smiles, handle in zip(filtered["smiles"], filtered["primary_handle"]):
                ok = False
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None and handle in HANDLE_SMARTS:
                    ok = handle in self.engine.handle_types(mol)
                keep.append(ok)
                if not ok:
                    dropped.append(str(smiles))
            filtered = filtered[keep]
        filtered = filtered.reset_index(drop=True)

        if n_filtered < n_raw:
            logger.info(
                "Catalog filter (Fsp3 >= %.2f, MW <= %.1f): %d -> %d synthons",
                self.min_fsp3, self.max_mw, n_raw, n_filtered,
            )
        if dropped:
            logger.warning(
                "Dropped %d synthons with unverifiable primary_handle (e.g. %s)",
                len(dropped), dropped[:3],
            )
        return filtered

    # ------------------------------------------------------------------
    # Embeddings: deterministic Morgan-fingerprint projections
    # ------------------------------------------------------------------
    def _compute_embeddings(self) -> torch.Tensor:
        """Embed every synthon via a seeded random projection of its Morgan
        count fingerprint.

        The mapping is deterministic (fixed seed) and chemically informed:
        structurally similar synthons receive similar embeddings, while no
        gradient state is kept inside the catalog.
        """
        rng = np.random.default_rng(self.seed)
        projection = rng.standard_normal((2048, self.embedding_dim)).astype(np.float32)
        projection /= np.sqrt(2048)

        rows = np.zeros((self.num_synthons, self.embedding_dim), dtype=np.float32)
        for i, smiles in enumerate(self.df["smiles"]):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            try:
                # Modern API (RDKit >= 2022): dedicated generator objects.
                generator = rdFingerprintGenerator.GetMorganGenerator(
                    radius=2, fpSize=2048
                )
                fp = generator.GetFingerprint(mol)
            except (AttributeError, NameError):
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            vec = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, vec)
            rows[i] = vec @ projection

        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        emb = torch.from_numpy(rows / norms)
        return emb

    # ------------------------------------------------------------------
    # Compatibility masks
    # ------------------------------------------------------------------
    def _build_canonical_smiles_index(self) -> Dict[str, Tuple[int, ...]]:
        """Index exact canonical SMILES to filtered catalog row indices."""
        index: Dict[str, List[int]] = {}
        for i, smiles in enumerate(self.df["smiles"]):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            index.setdefault(key, []).append(i)
        return {key: tuple(values) for key, values in index.items()}

    @property
    def canonical_smiles_index(self) -> Dict[str, Tuple[int, ...]]:
        """Canonical-SMILES -> catalog indices for exact retro matching."""
        return self._canonical_smiles_index

    @property
    def reaction_family_names(self) -> Tuple[str, ...]:
        return REACTION_FAMILY_NAMES

    def get_reaction_family_mask(
        self,
        family: str,
        device: Optional[torch.device] = None,
        core_handle: Optional[str] = None,
        require_remaining_handle: bool = False,
        allow_terminal: bool = True,
    ) -> torch.Tensor:
        """Return a synthon compatibility mask for a reaction family.

        When ``require_remaining_handle`` is set, synthons are restricted to
        multifunctional building blocks so growth can continue after the
        reaction. If the catalog contains no multifunctional partner for the
        family, the filter falls back to the full partner set: a
        monofunctional cap that terminates the trajectory is strictly better
        than a dead-ended rollout that can never react.
        """
        if family not in REACTION_FAMILY_MEMBERS:
            raise ValueError(
                f"Unknown reaction family '{family}'. "
                f"Available: {list(REACTION_FAMILY_NAMES)}"
            )
        key = (family, core_handle)
        if key not in self._family_mask_cache:
            members = REACTION_FAMILY_MEMBERS[family]
            masks = [
                self.get_reaction_mask(name, core_handle=core_handle)
                for name in members
            ]
            self._family_mask_cache[key] = torch.stack(masks, dim=0).max(dim=0).values
        mask = self._family_mask_cache[key].clone()
        if require_remaining_handle or not allow_terminal:
            has_remaining = torch.from_numpy(self._handle_counts >= 2)
            grown = mask.clone()
            grown[~has_remaining] = -1e9
            if bool((grown > -1e8).any().item()):
                mask = grown
            else:
                # No multifunctional partner exists for this family: fall
                # back to monofunctional caps instead of dead-ending growth.
                logger.debug(
                    "No multifunctional synthon for family %s (handle %s); "
                    "allowing monofunctional partners.",
                    family, core_handle,
                )
        if device is not None:
            mask = mask.to(device)
        return mask

    def get_reaction_family_compatibility_mask(
        self,
        device: Optional[torch.device] = None,
        core_handle: Optional[str] = None,
        require_remaining_handle: bool = False,
        allow_terminal: bool = True,
    ) -> torch.Tensor:
        """Return a [F] mask for reaction families legal for a core handle."""
        values = []
        for family in REACTION_FAMILY_NAMES:
            legal = any(
                core_handle in REACTION_SIDES[reaction]
                for reaction in REACTION_FAMILY_MEMBERS[family]
            )
            if legal and (require_remaining_handle or not allow_terminal):
                family_mask = self.get_reaction_family_mask(
                    family,
                    core_handle=core_handle,
                    require_remaining_handle=True,
                    allow_terminal=allow_terminal,
                )
                legal = bool(torch.any(family_mask > -1e8).item())
            values.append(0.0 if legal else -1e9)
        mask = torch.tensor(values, dtype=torch.float32)
        return mask.to(device) if device is not None else mask

    def get_reaction_family_masks(
        self,
        device: Optional[torch.device] = None,
        core_handle: Optional[str] = None,
        require_remaining_handle: bool = False,
        allow_terminal: bool = True,
    ) -> torch.Tensor:
        """Return [F, K] synthon masks for all reaction families."""
        return torch.stack([
            self.get_reaction_family_mask(
                family,
                device=device,
                core_handle=core_handle,
                require_remaining_handle=require_remaining_handle,
                allow_terminal=allow_terminal,
            )
            for family in REACTION_FAMILY_NAMES
        ], dim=0)

    def _build_compatibility_indices(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Index every detected handle, not only ``primary_handle``."""
        masks = {
            handle: np.zeros(self.num_synthons, dtype=bool)
            for handle in HANDLE_SMARTS
        }
        handle_counts = np.zeros(self.num_synthons, dtype=np.int16)
        for idx, smiles in enumerate(self.df["smiles"]):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            detected = self.engine.detect_handles(mol)
            handle_counts[idx] = len(detected)
            for info in detected:
                masks[info.handle_type][idx] = True
        return masks, handle_counts

    def get_reaction_mask(
        self,
        target_reaction: str,
        device: Optional[torch.device] = None,
        core_handle: Optional[str] = None,
    ) -> torch.Tensor:
        """Logit mask over the catalog for ``target_reaction``.

        Args:
            target_reaction: Reaction name (key of :data:`REACTION_SIDES`).
            device: Torch device for the returned tensor.
            core_handle: Handle type of the growing ligand's attachment
                point, if known. The mask is then restricted to the
                *complementary* partner handle so the selected synthon can
                actually react with the core.

        Returns:
            Float tensor of shape ``[num_synthons]`` with ``0.0`` for legal
            synthons and ``-1e9`` for incompatible ones.
        """
        if target_reaction not in REACTION_SIDES:
            raise ValueError(
                f"Unknown reaction '{target_reaction}'. "
                f"Available: {sorted(REACTION_SIDES)}"
            )
        side_a, side_b = REACTION_SIDES[target_reaction]
        if core_handle == side_a:
            allowed = {side_b}
        elif core_handle == side_b:
            allowed = {side_a}
        else:
            allowed = {side_a, side_b}

        valid_indices = np.zeros(self.num_synthons, dtype=bool)
        for handle in allowed:
            if handle in self.handle_masks:
                valid_indices |= self.handle_masks[handle]

        mask = torch.zeros(self.num_synthons, dtype=torch.float32)
        mask[~valid_indices] = -1e9
        if device is not None:
            mask = mask.to(device)
        return mask

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_mol(self, synthon_idx: int, explicit_hs: bool = False) -> Chem.Mol:
        """Parse the SMILES of synthon ``synthon_idx``.

        Args:
            synthon_idx: Row index into the filtered catalog.
            explicit_hs: When True, add explicit hydrogens (needed before
                conformer embedding).
        """
        smiles = self.get_smiles(synthon_idx)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Synthon {synthon_idx} has unparseable SMILES: {smiles}")
        if explicit_hs:
            mol = Chem.AddHs(mol)
        return mol

    def get_smiles(self, synthon_idx: int) -> str:
        if not 0 <= synthon_idx < self.num_synthons:
            raise IndexError(
                f"synthon_idx {synthon_idx} out of range [0, {self.num_synthons})"
            )
        return str(self.df.iloc[synthon_idx]["smiles"])

    def get_id(self, synthon_idx: int) -> str:
        if not 0 <= synthon_idx < self.num_synthons:
            raise IndexError(
                f"synthon_idx {synthon_idx} out of range [0, {self.num_synthons})"
            )
        return str(self.df.iloc[synthon_idx]["id"])

    def synthon_indices_for_handles(
        self, handles: Iterable[str]
    ) -> np.ndarray:
        """Indices of synthons exposing at least one requested handle."""
        wanted = set(handles)
        valid = np.zeros(self.num_synthons, dtype=bool)
        for handle in wanted:
            if handle in self.handle_masks:
                valid |= self.handle_masks[handle]
        return np.nonzero(valid)[0]

    def get_handle_count(self, synthon_idx: int) -> int:
        """Return the number of reactive handle instances on a synthon."""
        if not 0 <= synthon_idx < self.num_synthons:
            raise IndexError(f"synthon_idx {synthon_idx} out of range")
        return int(self._handle_counts[synthon_idx])

    @property
    def multifunctional_indices(self) -> np.ndarray:
        """Catalog indices exposing at least two reactive handle instances."""
        return np.nonzero(self._handle_counts >= 2)[0]

    @property
    def available_handles(self) -> List[str]:
        """Handle types actually present in the catalog (>=1 synthon), sorted."""
        return sorted(
            handle for handle, mask in self.handle_masks.items() if bool(mask.any())
        )

    def __len__(self) -> int:
        return self.num_synthons

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"SynthonCatalog(path={self.catalog_path!r}, "
            f"n={self.num_synthons}, dim={self.embedding_dim}, "
            f"min_fsp3={self.min_fsp3}, max_mw={self.max_mw})"
        )


__all__ = [
    "SynthonCatalog",
    "REQUIRED_COLUMNS",
    "DEFAULT_MIN_FSP3",
    "DEFAULT_MAX_MW",
    "DEFAULT_EXEMPT_HANDLES",
]
