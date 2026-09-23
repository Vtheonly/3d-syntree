"""CrossDocked2020 dataset with reaction-validated supervision.

Real-mode examples are supervised only when a CrossDocked ligand can be
decomposed into a catalog synthon plus a reaction-compatible core and that
decomposition replays forward through the real RDKit reaction engine to the
same molecular connectivity. No random synthon or random dihedral labels are
ever created for real complexes.

Synthetic mode is retained strictly as a deterministic plumbing/smoke-test
dataset and is explicitly marked as synthetic in each sample.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from syntree.data.featurizer import MolecularFeaturizer
from syntree.chemistry.reactions import HANDLE_NAMES, REACTION_FAMILY_NAMES
from syntree.data.fragmenter import ReactionConstrainedFragmenter

logger = logging.getLogger(__name__)


def _synthetic_pocket(
    rng: np.random.Generator, n_pocket: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic mock pocket: clustered C/N/O point cloud."""
    pos = torch.from_numpy(
        rng.normal(scale=3.0, size=(n_pocket, 3)).astype(np.float32)
    )
    z = torch.from_numpy(
        rng.choice([6, 7, 8], size=n_pocket, p=[0.6, 0.2, 0.2]).astype(np.int64)
    )
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
        synthetic_fallback: bool = False,
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
        self.synthetic_fallback = bool(synthetic_fallback)
        self.force_rebuild = bool(force_rebuild)
        self.num_synthetic = int(num_synthetic)

        os.makedirs(self.root_dir, exist_ok=True)
        self.pocket_files = sorted(
            glob.glob(os.path.join(self.root_dir, self._POCKET_GLOB))
        )
        self.all_pair_files = [
            p
            for p in self.pocket_files
            if os.path.exists(p.replace("_pocket.pdb", "_ligand.sdf"))
        ]
        self.has_real_pairs = bool(self.all_pair_files)
        self.pair_files = [
            p for p in self.all_pair_files if self._belongs_to_split(p)
        ]

        # Synthetic data is explicit. With synthetic=None it is enabled
        # only when the caller explicitly opted into synthetic_fallback.
        self.use_synthetic = (
            bool(synthetic)
            if synthetic is not None
            else (not self.has_real_pairs and self.synthetic_fallback)
        )

        super().__init__(self.root_dir, transform, pre_transform, pre_filter)
        self._load_or_process()

    @property
    def raw_dir(self) -> str:
        return self.root_dir

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root_dir, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(p) for p in self.pocket_files]

    @property
    def processed_file_names(self) -> List[str]:
        mode = "synthetic" if self.use_synthetic else "real"
        catalog_key = len(self.catalog) if self.catalog is not None else 0
        return [
            f"{self.split}_{mode}_n{self.num_synthetic}_s{self.seed}_k{catalog_key}.pt"
        ]

    def download(self):
        pass

    def process(self):
        data_list = self._build_samples()
        # _build_samples() may explicitly switch to synthetic mode when the
        # caller opted into fallback. Recompute the processed path after that
        # decision so a fallback cache is never stored under a "real" filename.
        self.save(data_list, self.processed_paths[0])

    def _load_or_process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        path = self.processed_paths[0]
        if self.force_rebuild or not os.path.exists(path):
            self.process()
            path = self.processed_paths[0]
        try:
            self.load(path)
            return
        except Exception:
            self.process()
            path = self.processed_paths[0]
            self.load(path)

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.num_synthetic
        return super().__len__()

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------
    def _belongs_to_split(self, pocket_path: str) -> bool:
        """Stable 80/10/10 split with no cross-split duplication."""
        key = os.path.basename(pocket_path).encode("utf-8")
        bucket = zlib.crc32(key) % 1000
        assigned = (
            "train" if bucket < 800 else "val" if bucket < 900 else "test"
        )
        return assigned == self.split

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------
    def _build_samples(self) -> List[Data]:
        return self._build_synthetic() if self.use_synthetic else self._build_real()

    def _build_synthetic(self) -> List[Data]:
        rng = np.random.default_rng(
            self.seed + (zlib.crc32(self.split.encode()) % 10000)
        )
        n_catalog = len(self.catalog) if self.catalog is not None else 50

        g = torch.Generator().manual_seed(self.seed + 7919)
        n_feat = 72
        w_syn = torch.randn(n_feat, generator=g)
        w_dih = torch.randn(n_feat, generator=g)
        w_rxn = torch.randn(n_feat, generator=g)
        scale = float(math.sqrt(n_feat))

        family_handles = {
            "amide_coupling": ("carboxylic_acid", "primary_secondary_amine"),
            "reductive_amination": ("aldehyde", "primary_secondary_amine"),
            "suzuki_coupling": ("aryl_halide", "boronic_acid"),
            "aryl_amination": ("aryl_halide", "primary_secondary_amine"),
            "urea_formation": ("primary_secondary_amine",),
            "esterification": ("carboxylic_acid", "alcohol"),
            "click_triazole": ("alkyne", "azide"),
        }

        samples: List[Data] = []
        for _ in range(self.num_synthetic):
            n_pocket = int(rng.integers(24, 72))
            pos, z = _synthetic_pocket(rng, n_pocket)
            handle_position = pos[int(rng.integers(0, n_pocket))]
            handle_feat = torch.cat([
                torch.from_numpy(rng.normal(size=64).astype(np.float32)),
                handle_position,
            ])
            pocket_stats = torch.stack(
                [
                    pos[:, 0].mean(),
                    pos[:, 1].mean(),
                    pos[:, 2].mean(),
                    pos.abs().mean(),
                    z.float().mean(),
                ]
            )
            feats = torch.cat([handle_feat, pocket_stats])

            target_dihedral = math.pi * math.tanh(
                2.0 * torch.dot(feats, w_dih) / scale
            )
            family_idx = int(
                torch.floor(
                    torch.sigmoid(torch.dot(feats, w_rxn) / scale)
                    * len(REACTION_FAMILY_NAMES)
                ).clamp(0, len(REACTION_FAMILY_NAMES) - 1)
            )
            handles = family_handles[REACTION_FAMILY_NAMES[family_idx]]
            handle_idx = int(
                abs(int(torch.round(handle_feat[0] * 17).item()))
                % len(handles)
            )
            core_handle_idx = HANDLE_NAMES.index(handles[handle_idx])

            # Keep synthetic labels chemically compatible with the same
            # catalog grammar used by real training.
            if self.catalog is not None:
                from syntree.chemistry.reactions import REACTION_FAMILY_MEMBERS, REACTION_SIDES
                candidate_handles = set()
                for reaction in REACTION_FAMILY_MEMBERS[
                    REACTION_FAMILY_NAMES[family_idx]
                ]:
                    side_a, side_b = REACTION_SIDES[reaction]
                    core_handle = HANDLE_NAMES[core_handle_idx]
                    partner = side_b if core_handle == side_a else side_a if core_handle == side_b else None
                    if partner:
                        candidate_handles.add(partner)
                candidates = self.catalog.synthon_indices_for_handles(candidate_handles)
            else:
                candidates = np.arange(n_catalog, dtype=int)
            if len(candidates) == 0:
                candidates = np.arange(n_catalog, dtype=int)
            raw_idx = int(
                torch.floor(
                    torch.sigmoid(3.0 * torch.dot(feats, w_syn) / scale)
                    * len(candidates)
                ).clamp(0, len(candidates) - 1)
            )
            target_synthon = int(candidates[raw_idx])

            samples.append(
                Data(
                    pocket_pos=pos,
                    pocket_z=z,
                    handle_features=handle_feat,
                    target_synthon=torch.tensor(target_synthon, dtype=torch.long),
                    target_dihedral=torch.tensor(target_dihedral, dtype=torch.float32),
                    target_reaction_family_idx=torch.tensor(
                        family_idx, dtype=torch.long
                    ),
                    target_core_handle_idx=torch.tensor(
                        core_handle_idx, dtype=torch.long
                    ),
                    is_real_sample=torch.tensor(False, dtype=torch.bool),
                )
            )
        return samples

    def _build_real(self) -> List[Data]:
        if self.catalog is None:
            raise ValueError(
                "Real-mode CrossDockedDataset requires a SynthonCatalog "
                "to compute validated synthon targets."
            )

        fragmenter = ReactionConstrainedFragmenter(self.catalog)
        samples: List[Data] = []
        skipped_invalid = 0
        skipped_unmatched = 0

        for pocket_path in self.pair_files:
            ligand_path = pocket_path.replace("_pocket.pdb", "_ligand.sdf")
            try:
                pocket_mol = Chem.MolFromPDBFile(pocket_path, removeHs=False)
            except Exception:
                pocket_mol = None
            if pocket_mol is None or pocket_mol.GetNumAtoms() == 0:
                skipped_invalid += 1
                continue

            try:
                ligand_supplier = Chem.SDMolSupplier(
                    ligand_path, removeHs=False, sanitize=True
                )
                ligand_mol = next(
                    (m for m in ligand_supplier if m is not None), None
                )
            except Exception:
                ligand_mol = None

            if ligand_mol is None or ligand_mol.GetNumAtoms() == 0:
                skipped_invalid += 1
                continue
            if ligand_mol.GetNumConformers() == 0:
                skipped_invalid += 1
                continue

            try:
                center = MolecularFeaturizer.ligand_center(ligand_mol)
                feats = MolecularFeaturizer.featurize_pocket(
                    pocket_mol, center=center
                )
                target = fragmenter.find_target(ligand_mol)
            except Exception as exc:
                logger.debug(
                    "Skipping %s after target extraction failure: %s",
                    os.path.basename(pocket_path),
                    exc,
                )
                skipped_invalid += 1
                continue

            if target is None:
                skipped_unmatched += 1
                continue

            handle_info = next(
                (
                    h
                    for h in fragmenter.engine.detect_handles(target.core_mol)
                    if h.handle_type == target.core_handle_type
                ),
                None,
            )
            if handle_info is None:
                skipped_unmatched += 1
                continue

            samples.append(
                Data(
                    pocket_pos=feats["pocket_pos"],
                    pocket_z=feats["pocket_z"],
                    handle_features=MolecularFeaturizer.featurize_handle(
                        target.core_mol,
                        handle_info.atom_indices,
                        target.core_handle_type,
                        reference_center=center,
                    ),
                    target_synthon=torch.tensor(
                        target.synthon_index, dtype=torch.long
                    ),
                    target_dihedral=torch.tensor(
                        target.target_dihedral, dtype=torch.float32
                    ),
                    target_reaction_family_idx=torch.tensor(
                        REACTION_FAMILY_NAMES.index(target.reaction_family),
                        dtype=torch.long,
                    ),
                    target_core_handle_idx=torch.tensor(
                        HANDLE_NAMES.index(target.core_handle_type),
                        dtype=torch.long,
                    ),
                    is_real_sample=torch.tensor(True, dtype=torch.bool),
                )
            )

        logger.info(
            "CrossDocked %s split: pairs=%d, matched=%d, invalid=%d, unmatched=%d",
            self.split,
            len(self.pair_files),
            len(samples),
            skipped_invalid,
            skipped_unmatched,
        )

        if not samples and not self.has_real_pairs and not self.synthetic_fallback:
            raise RuntimeError(
                "No CrossDocked pocket-ligand pairs were found. Real training "
                "requires the actual CrossDocked pairs. Set synthetic=True or "
                "enable data.synthetic_fallback only for an explicit smoke run."
            )

        if not samples and self.pair_files and not self.synthetic_fallback:
            raise RuntimeError(
                f"No reaction-validated training targets were extracted from "
                f"{len(self.pair_files)} {self.split} CrossDocked pairs. "
                "Real data is never replaced with random or synthetic labels. "
                "Enable data.synthetic_fallback only for an explicit smoke run."
            )

        if not samples and self.pair_files and self.synthetic_fallback:
            logger.warning(
                "No real targets matched the catalog; explicit synthetic "
                "fallback is enabled for this run."
            )
            self.use_synthetic = True
            return self._build_synthetic()

        return samples

    def summary(self) -> Dict[str, object]:
        return {
            "root": self.root_dir,
            "split": self.split,
            "mode": "synthetic" if self.use_synthetic else "real",
            "num_samples": len(self),
            "num_pocket_files": len(self.pocket_files),
            "num_pair_files": len(self.all_pair_files),
            "num_split_pairs": len(self.pair_files),
        }


__all__ = ["CrossDockedDataset"]
