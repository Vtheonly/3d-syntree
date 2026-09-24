"""Enamine 3D synthon catalog: loading, filtering, and reaction-grammar
masking.

The catalog enforces the curated Enamine REAL 3D-Diversity constraints:

* fraction of sp3 carbons (Fsp3) >= ``min_fsp3`` (default 0.42),
* molecular weight <= ``max_mw`` Da (default 220),

with one chemically necessary exemption: aryl halides and boronic acids
(the mandatory coupling partners for Suzuki / Buchwald-Hartwig / SNAr
chemistry, whose reacting atoms are aromatic by definition) and sulfonyl
chlorides (aryl sulfonyl chlorides - tosylates, benzene sulfonamides - are
the workhorse precursors of the FDA-prevalent sulfonamide motif added in
tasklist Priority 4) are exempt from the Fsp3 floor. Without them the
reaction grammar could not close.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from syntree.chemistry.reactions import (
    REACTION_FAMILY_MEMBERS,
    REACTION_FAMILY_NAMES,
    REACTION_SIDES,
    HANDLE_SMARTS,
    ReactionEngine,
)
from syntree.chemistry.synthon_encoder import (
    ENCODER_VERSION,
    SUPPORTED_ENCODERS,
    catalog_ids_hash,
    encode_catalog_morgan2d,
    encode_catalog_pharm3d,
)
from syntree.data.synthon_library import SynthonLibraryStore

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("id", "smiles", "fsp3", "mw", "primary_handle")

# Default catalog constraints (Enamine 3D-Diversity curation rules).
DEFAULT_MIN_FSP3 = 0.42
DEFAULT_MAX_MW = 220.0

# Handle types exempt from the Fsp3 floor (aryl coupling partners are
# necessarily flat: [c]-X bonds are required by the reaction SMARTS;
# sulfonyl chlorides cover the aryl sulfonamide precursors).
DEFAULT_EXEMPT_HANDLES = ("aryl_halide", "boronic_acid", "sulfonyl_chloride")


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
        encoder: Synthon embedding encoder. ``"pharm3d"`` (default,
            tasklist Priority 1: deterministic 3D-pharmacophore descriptor
            with conformer shape/electrostatics/handle-direction features and
            an orthonormal QR projection, cached on disk) or ``"morgan2d"``
            (legacy seeded random projection of the Morgan bit fingerprint,
            bit-for-bit identical to the historical implementation).
        cache_embeddings: Persist pharm3d embeddings next to the catalog
            (``<catalog>_embeddings_<dim>d_<encoder>.h5``). The cache is
            validated against ids/dim/seed/encoder and silently skipped when
            the catalog directory is read-only (e.g. Kaggle ``/kaggle/input``).
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
        encoder: str = "pharm3d",
        cache_embeddings: bool = True,
    ):
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if encoder not in SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unknown synthon encoder '{encoder}'. "
                f"Supported: {list(SUPPORTED_ENCODERS)}"
            )
        self.catalog_path = str(catalog_path)
        self.embedding_dim = int(embedding_dim)
        self.min_fsp3 = float(min_fsp3)
        self.max_mw = float(max_mw)
        self.seed = int(seed)
        self.exempt_handles = tuple(exempt_handles or ())
        self.encoder_name = str(encoder)
        self.cache_embeddings = bool(cache_embeddings)

        self.engine = ReactionEngine()
        self.df = self._load_and_filter(validate_handles)
        self.num_synthons = len(self.df)
        if self.num_synthons == 0:
            raise ValueError(
                f"Catalog '{catalog_path}' is empty after filtering "
                f"(Fsp3 >= {min_fsp3}, MW <= {max_mw})."
            )

        self.embeddings, self.embedding_info = self._compute_embeddings()
        self.handle_masks, self._handle_counts = self._build_compatibility_indices()
        self._canonical_smiles_index = self._build_canonical_smiles_index()
        self._ids_hash = catalog_ids_hash(self.df["id"].astype(str).tolist())
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
    # Embeddings: 3D-pharmacophore encoder (default) or legacy Morgan
    # projection, with a validated on-disk cache for the expensive 3D pass.
    # ------------------------------------------------------------------
    def _compute_embeddings(self) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Embed every synthon with the configured encoder.

        Both encoders are deterministic for a fixed seed; ``pharm3d`` also
        caches the result next to the catalog so repeated training runs skip
        the conformer/force-field pass (the cache is skipped silently when
        the catalog directory is read-only, e.g. on Kaggle input mounts).
        """
        if self.encoder_name == "morgan2d":
            embeddings = encode_catalog_morgan2d(
                self.df["smiles"].tolist(), self.embedding_dim, self.seed
            )
            return embeddings, {"encoder": "morgan2d"}

        cache_path = self._embedding_cache_path()
        if self.cache_embeddings and cache_path is not None:
            cached = self._load_embedding_cache(cache_path)
            if cached is not None:
                logger.info(
                    "Loaded cached %s embeddings for %d synthons from %s",
                    self.encoder_name, self.num_synthons, cache_path,
                )
                return cached, {"encoder": self.encoder_name, "from_cache": True}

        embeddings, info = encode_catalog_pharm3d(
            self.df["smiles"].tolist(),
            self.embedding_dim,
            self.seed,
            engine=self.engine,
        )
        if self.cache_embeddings and cache_path is not None:
            self._save_embedding_cache(cache_path, embeddings, info)
        return embeddings, info

    def _embedding_cache_path(self) -> Optional[str]:
        base = os.path.splitext(self.catalog_path)[0]
        return f"{base}_embeddings_{self.embedding_dim}d_{self.encoder_name}.h5"

    def _load_embedding_cache(self, cache_path: str) -> Optional[torch.Tensor]:
        """Return cached embeddings when they match this catalog exactly."""
        if not os.path.exists(cache_path):
            return None
        store = SynthonLibraryStore(cache_path)
        try:
            payload = store.load()
        except Exception as exc:
            logger.warning("Embedding cache unreadable (%s); recomputing.", exc)
            return None
        meta = payload.get("meta", {}) or {}
        ids = list(payload.get("ids", []) or [])
        emb = payload.get("embeddings")
        expected_ids = self.df["id"].astype(str).tolist()
        if (
            meta.get("encoder") != self.encoder_name
            or meta.get("encoder_version") != ENCODER_VERSION
            or meta.get("seed") != self.seed
            or meta.get("dim") != self.embedding_dim
            or list(map(str, ids)) != expected_ids
            or emb is None
            or tuple(emb.shape) != (self.num_synthons, self.embedding_dim)
        ):
            logger.info("Embedding cache signature mismatch; recomputing.")
            return None
        return emb

    def _save_embedding_cache(
        self, cache_path: str, embeddings: torch.Tensor, info: Dict[str, object]
    ) -> None:
        """Atomically persist embeddings; skip gracefully on read-only dirs."""
        store = SynthonLibraryStore(cache_path)
        try:
            store.save(
                embeddings,
                ids=self.df["id"].astype(str).tolist(),
                smiles=self.df["smiles"].astype(str).tolist(),
                handles=self.df["primary_handle"].astype(str).tolist(),
                source_catalog=self.catalog_path,
                seed=self.seed,
                extra_meta={
                    "encoder": self.encoder_name,
                    "encoder_version": ENCODER_VERSION,
                    **{k: v for k, v in info.items() if k != "encoder"},
                },
            )
        except Exception as exc:
            # Read-only mount (Kaggle /kaggle/input) or missing h5py: the
            # in-memory embeddings are still valid, just not cached.
            logger.info(
                "Embedding cache not written to %s (%s); continuing in-memory.",
                cache_path, exc,
            )

    @property
    def embedding_signature(self) -> Dict[str, object]:
        """Signature of the embedding table for checkpoint validation."""
        return {
            "encoder": self.encoder_name,
            "encoder_version": ENCODER_VERSION if self.encoder_name == "pharm3d" else "legacy",
            "dim": self.embedding_dim,
            "seed": self.seed,
            "ids_hash": self._ids_hash,
            "num_synthons": self.num_synthons,
        }

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
            f"encoder={self.encoder_name}, "
            f"min_fsp3={self.min_fsp3}, max_mw={self.max_mw})"
        )


__all__ = [
    "SynthonCatalog",
    "REQUIRED_COLUMNS",
    "DEFAULT_MIN_FSP3",
    "DEFAULT_MAX_MW",
    "DEFAULT_EXEMPT_HANDLES",
]
