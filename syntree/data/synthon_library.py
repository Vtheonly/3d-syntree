"""Synthon library persistence: vector store backed by HDF5 / Parquet.

Provides an on-disk cache of catalog embeddings and metadata so repeated
training runs avoid re-computing Morgan fingerprint projections for every
synthon, plus a lightweight LMDB-free random-access reader.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MAGIC = "SYNTREE-EMB-V1"


class SynthonLibraryStore:
    """Persistence layer for synthon embeddings and catalog metadata.

    The store writes a single HDF5 file (falling back to a NumPy ``.npz``
    archive when h5py is unavailable) containing:

    * ``embeddings`` – ``[K, d]`` float32 matrix,
    * ``smiles`` / ``ids`` / ``handles`` – per-synthon metadata arrays,
    * ``meta`` – JSON blob with provenance (source path, dim, seed).
    """

    def __init__(self, path: str):
        self.path = str(path)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save(
        self,
        embeddings: torch.Tensor,
        ids,
        smiles,
        handles,
        source_catalog: str,
        seed: int,
    ) -> str:
        """Atomically persist the embedding table and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)

        emb = (
            embeddings.detach().cpu().numpy().astype(np.float32)
            if isinstance(embeddings, torch.Tensor)
            else np.asarray(embeddings, dtype=np.float32)
        )
        meta = json.dumps(
            {
                "magic": _MAGIC,
                "source_catalog": str(source_catalog),
                "seed": int(seed),
                "dim": int(emb.shape[1]),
                "num_synthons": int(emb.shape[0]),
            }
        )

        tmp_path = self.path + ".tmp"
        if self.path.endswith(".h5") or self.path.endswith(".hdf5"):
            try:
                import h5py

                with h5py.File(tmp_path, "w") as h5:
                    h5.create_dataset("embeddings", data=emb, compression="gzip")
                    dt = h5py.string_dtype(encoding="utf-8")
                    h5.create_dataset("ids", data=[str(i) for i in ids], dtype=dt)
                    h5.create_dataset("smiles", data=[str(s) for s in smiles], dtype=dt)
                    h5.create_dataset("handles", data=[str(h) for h in handles], dtype=dt)
                    h5.attrs["meta"] = meta
                os.replace(tmp_path, self.path)
                return self.path
            except ImportError:
                logger.info("h5py unavailable; falling back to .npz store.")

        np.savez_compressed(
            tmp_path,
            embeddings=emb,
            ids=np.array([str(i) for i in ids]),
            smiles=np.array([str(s) for s in smiles]),
            handles=np.array([str(h) for h in handles]),
            meta=np.array(meta),
        )
        # np.savez appends .npz when missing; normalize the name.
        if os.path.exists(tmp_path + ".npz") and not os.path.exists(tmp_path):
            os.replace(tmp_path + ".npz", tmp_path)
        os.replace(tmp_path, self.path)
        return self.path

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> Dict[str, object]:
        """Load the store. Returns a dict with ``embeddings`` (tensor) and
        metadata lists; raises ``FileNotFoundError`` when absent."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Synthon library store not found: {self.path}")

        if self.path.endswith((".h5", ".hdf5")):
            try:
                import h5py

                with h5py.File(self.path, "r") as h5:
                    emb = np.asarray(h5["embeddings"][...], dtype=np.float32)
                    ids = [s.decode() if isinstance(s, bytes) else s for s in h5["ids"][...]]
                    smiles = [
                        s.decode() if isinstance(s, bytes) else s for s in h5["smiles"][...]
                    ]
                    handles = [
                        s.decode() if isinstance(s, bytes) else s for s in h5["handles"][...]
                    ]
                    meta = json.loads(h5.attrs["meta"])
                return {
                    "embeddings": torch.from_numpy(emb),
                    "ids": ids,
                    "smiles": smiles,
                    "handles": handles,
                    "meta": meta,
                }
            except ImportError:
                logger.warning("h5py unavailable for %s; corrupt read.", self.path)
                raise

        archive = np.load(self.path, allow_pickle=False)
        emb = np.asarray(archive["embeddings"], dtype=np.float32)
        ids = [str(s) for s in archive["ids"].tolist()]
        smiles = [str(s) for s in archive["smiles"].tolist()]
        handles = [str(s) for s in archive["handles"].tolist()]
        meta = json.loads(str(archive["meta"].item()))
        return {
            "embeddings": torch.from_numpy(emb),
            "ids": ids,
            "smiles": smiles,
            "handles": handles,
            "meta": meta,
        }

    def exists(self) -> bool:
        return os.path.exists(self.path)

    @staticmethod
    def default_path(catalog_path: str, embedding_dim: int) -> str:
        """Canonical store location derived from a catalog path."""
        base = os.path.splitext(catalog_path)[0]
        return f"{base}_embeddings_{embedding_dim}d.h5"


__all__ = ["SynthonLibraryStore"]
