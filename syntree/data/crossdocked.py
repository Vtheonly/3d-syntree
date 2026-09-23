"""PyTorch Geometric dataset for CrossDocked2020 pocket-ligand complexes.

Two operating modes are supported:

**Real mode** – the directory ``root_dir`` contains ``*_pocket.pdb`` /
``*_ligand.sdf`` *pairs* produced by the CrossDocked2020 preprocessing
(RMSD < 1.0 A, 30% sequence-identity clustering). A directory holding only
pocket files (e.g. the generation-time ``sample_pocket.pdb``) does NOT
qualify – training targets require the ligand too.

**Synthetic mode** – when the directory holds no pocket files (or
``synthetic=True``), a deterministic, seeded mock dataset is produced so the
full training / generation / evaluation pipeline can be exercised end-to-end
without the multi-gigabyte external download (CI, smoke tests, Colab dry
runs).

Every sample is a :class:`torch_geometric.data.Data` object with:

======================  =====================================================
Field                   Content
======================  =====================================================
``pocket_pos``          ``[N_p, 3]`` centroid-centered pocket coordinates
``pocket_z``            ``[N_p]`` pocket atomic numbers
``handle_features``     ``[64]`` attachment-handle feature vector
``target_synthon``      scalar catalog index of the ground-truth synthon
``target_dihedral``     scalar target dihedral in ``[-pi, pi)``
======================  =====================================================
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from syntree.data.featurizer import MolecularFeaturizer

logger = logging.getLogger(__name__)


def _synthetic_pocket(rng: np.random.Generator, n_pocket: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic mock pocket: clustered C/N/O point cloud."""
    pos = torch.from_numpy(rng.normal(scale=3.0, size=(n_pocket, 3)).astype(np.float32))
    z = torch.from_numpy(rng.choice([6, 7, 8], size=n_pocket, p=[0.6, 0.2, 0.2]).astype(np.int64))
    pos = pos - pos.mean(dim=0, keepdim=True)
    return pos, z


class CrossDockedDataset(InMemoryDataset):
    """Pocket-ligand dataset with real and synthetic operating modes."""

    _POCKET_GLOB = "*_pocket.pdb"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        catalog=None,
        num_synthetic: int = 100,
        synthetic: Optional[bool] = None,
        seed: int = 42,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_rebuild: bool = False,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")
        self.root_dir = str(root_dir)
        self.split = split
        self.catalog = catalog
        self.seed = int(seed)
        self.force_rebuild = bool(force_rebuild)

        os.makedirs(self.root_dir, exist_ok=True)
        self.pocket_files = sorted(
            glob.glob(os.path.join(self.root_dir, self._POCKET_GLOB))
        )
        # Real mode requires pocket-ligand PAIRS (training data); pockets
        # without ligands (e.g. generation-time sample pockets) do not
        # qualify.
        self.pair_files = [
            p for p in self.pocket_files
            if os.path.exists(p.replace("_pocket.pdb", "_ligand.sdf"))
        ]
        self.use_synthetic = (
            (len(self.pair_files) == 0) if synthetic is None else bool(synthetic)
        )
        self.num_synthetic = int(num_synthetic)

        super().__init__(self.root_dir, transform, pre_transform, pre_filter)
        self._load_or_process()

    # ------------------------------------------------------------------
    # PyG plumbing
    # ------------------------------------------------------------------
    @property
    def raw_dir(self) -> str:
        return self.root_dir

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root_dir, "processed")

    @property
    def raw_file_names(self) -> List[str]:  # informational
        return [os.path.basename(p) for p in self.pocket_files]

    @property
    def processed_file_names(self) -> List[str]:
        mode = "synthetic" if self.use_synthetic else "real"
        return [f"{self.split}_{mode}.pt"]

    def download(self):  # handled by scripts/download_assets.py
        pass

    def process(self):
        data_list = self._build_samples()
        if hasattr(self, "save"):
            self.save(data_list, self.processed_paths[0])
        else:  # pragma: no cover - very old PyG
            self._data, self.slices = self.collate(data_list)

    def _load_or_process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        path = self.processed_paths[0]
        if self.force_rebuild or not os.path.exists(path):
            self.process()
        if hasattr(self, "load"):
            try:
                self.load(path)
                return
            except Exception:  # pragma: no cover - rebuild fallback
                self.process()
                self.load(path)
                return
        # PyG < 2.4 fallback
        try:
            self._data, self.slices = torch.load(path, weights_only=False)
        except Exception:
            self.process()
            self._data, self.slices = torch.load(path, weights_only=False)

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.num_synthetic
        return super().__len__()

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------
    def _build_samples(self) -> List[Data]:
        if self.use_synthetic:
            return self._build_synthetic()
        return self._build_real()

    def _build_synthetic(self) -> List[Data]:
        rng = np.random.default_rng(self.seed + (hash(self.split) % 10000))
        n_catalog = len(self.catalog) if self.catalog is not None else 50

        samples: List[Data] = []
        for _ in range(self.num_synthetic):
            n_pocket = int(rng.integers(24, 72))
            pos, z = _synthetic_pocket(rng, n_pocket)
            handle_feat = torch.from_numpy(rng.normal(size=64).astype(np.float32))
            samples.append(
                Data(
                    pocket_pos=pos,
                    pocket_z=z,
                    handle_features=handle_feat,
                    target_synthon=torch.tensor(
                        int(rng.integers(0, max(n_catalog, 1))), dtype=torch.long
                    ),
                    target_dihedral=torch.tensor(
                        float(rng.uniform(-np.pi, np.pi)), dtype=torch.float32
                    ),
                )
            )
        return samples

    def _build_real(self) -> List[Data]:
        if self.catalog is None:
            raise ValueError(
                "Real-mode CrossDockedDataset requires a SynthonCatalog to "
                "compute ground-truth synthon targets."
            )
        from syntree.chemistry.reactions import ReactionEngine

        engine = ReactionEngine()
        rng = np.random.default_rng(self.seed)
        samples: List[Data] = []

        for pocket_path in self.pair_files:
            ligand_path = pocket_path.replace("_pocket.pdb", "_ligand.sdf")
            pocket_mol = Chem.MolFromPDBFile(pocket_path, removeHs=False)
            if pocket_mol is None or pocket_mol.GetNumAtoms() == 0:
                continue
            ligand_mol = (
                Chem.SDMolSupplier(ligand_path, removeHs=False)[0]
                if os.path.exists(ligand_path)
                else None
            )

            try:
                feats = MolecularFeaturizer.featurize_pocket(pocket_mol)
            except Exception:
                continue

            handle_feat = torch.zeros(64, dtype=torch.float32)
            target_synthon = int(rng.integers(0, len(self.catalog)))
            target_dihedral = float(rng.uniform(-np.pi, np.pi))

            if ligand_mol is not None:
                try:
                    center = MolecularFeaturizer.ligand_center(ligand_mol)
                    feats = MolecularFeaturizer.featurize_pocket(
                        pocket_mol, center=center
                    )
                except Exception:
                    pass
                handles = engine.detect_handles(ligand_mol)
                if handles:
                    info = handles[0]
                    handle_feat = MolecularFeaturizer.featurize_handle(
                        ligand_mol, info.atom_indices, info.handle_type
                    )

            samples.append(
                Data(
                    pocket_pos=feats["pocket_pos"],
                    pocket_z=feats["pocket_z"],
                    handle_features=handle_feat,
                    target_synthon=torch.tensor(target_synthon, dtype=torch.long),
                    target_dihedral=torch.tensor(target_dihedral, dtype=torch.float32),
                )
            )

        if not samples:
            logger.warning(
                "No usable complexes under %s; falling back to synthetic mode.",
                self.root_dir,
            )
            self.use_synthetic = True
            return self._build_synthetic()
        return samples

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        """Human-readable dataset summary for logging."""
        return {
            "root": self.root_dir,
            "split": self.split,
            "mode": "synthetic" if self.use_synthetic else "real",
            "num_samples": len(self),
            "num_pocket_files": len(self.pocket_files),
        }


__all__ = ["CrossDockedDataset"]
