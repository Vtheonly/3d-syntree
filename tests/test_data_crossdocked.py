"""Unit tests for the CrossDocked dataset (synthetic + real modes)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.crossdocked import CrossDockedDataset


@pytest.fixture(scope="module")
def catalog(assets_dir):
    return SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)


class TestSyntheticMode:
    def test_synthetic_mode_is_explicit(self, tmp_path, catalog):
        with pytest.raises(RuntimeError, match="No CrossDocked pocket-ligand pairs"):
            CrossDockedDataset(
                str(tmp_path),
                split="train",
                catalog=catalog,
                num_synthetic=10,
            )

    def test_explicit_synthetic_mode(self, tmp_path, catalog):
        ds = CrossDockedDataset(
            str(tmp_path),
            synthetic=True,
            split="train",
            catalog=catalog,
            num_synthetic=10,
        )
        assert ds.use_synthetic
        assert len(ds) == 10

    def test_sample_structure(self, tmp_path, catalog):
        ds = CrossDockedDataset(str(tmp_path), synthetic=True, split="train", catalog=catalog,
                                num_synthetic=6, seed=7)
        sample = ds[0]
        assert sample.pocket_pos.shape[1] == 3
        assert sample.is_real_sample.item() is False
        assert 0 <= sample.target_synthon.item() < len(catalog)
        assert 0 <= sample.target_reaction_family_idx.item() < 7
        assert sample.pocket_z.shape[0] == sample.pocket_pos.shape[0]
        assert sample.handle_features.shape == (64,)
        assert sample.handle_pos.shape == (1, 3)
        assert sample.global_features.shape == (1, 4)
        assert sample.target_synthon.dtype == torch.long
        assert sample.target_dihedral.dtype == torch.float32
        assert sample.target_reaction_family_idx.dtype == torch.long
        assert sample.target_core_handle_idx.dtype == torch.long
        assert sample.is_real_sample.dtype == torch.bool
        assert -np.pi <= sample.target_dihedral.item() < np.pi

    def test_deterministic_given_seed(self, tmp_path, catalog):
        d1 = CrossDockedDataset(str(tmp_path / "a"), synthetic=True, catalog=catalog,
                                num_synthetic=5, seed=11)
        d2 = CrossDockedDataset(str(tmp_path / "b"), synthetic=True, catalog=catalog,
                                num_synthetic=5, seed=11)
        assert torch.allclose(d1[2].pocket_pos, d2[2].pocket_pos)
        assert d1[3].target_synthon == d2[3].target_synthon

    def test_different_seeds_differ(self, tmp_path, catalog):
        d1 = CrossDockedDataset(str(tmp_path / "a"), synthetic=True, catalog=catalog,
                                num_synthetic=5, seed=1)
        d2 = CrossDockedDataset(str(tmp_path / "b"), synthetic=True, catalog=catalog,
                                num_synthetic=5, seed=2)
        p1, p2 = d1[0].pocket_pos, d2[0].pocket_pos
        assert p1.shape != p2.shape or not torch.allclose(p1, p2)

    def test_splits_differ(self, tmp_path, catalog):
        train = CrossDockedDataset(str(tmp_path), synthetic=True, split="train", catalog=catalog,
                                   num_synthetic=8, seed=3)
        val = CrossDockedDataset(str(tmp_path), synthetic=True, split="val", catalog=catalog,
                                 num_synthetic=8, seed=3)
        p1, p2 = train[0].pocket_pos, val[0].pocket_pos
        assert p1.shape != p2.shape or not torch.allclose(p1, p2)

    def test_invalid_split_raises(self, tmp_path, catalog):
        with pytest.raises(ValueError, match="train/val/test"):
            CrossDockedDataset(str(tmp_path), synthetic=True, split="valid", catalog=catalog)

    def test_targets_within_catalog(self, tmp_path, catalog):
        ds = CrossDockedDataset(str(tmp_path), synthetic=True, catalog=catalog, num_synthetic=20)
        for i in range(len(ds)):
            assert 0 <= ds[i].target_synthon.item() < len(catalog)

    def test_processed_cache_reuse(self, tmp_path, catalog):
        CrossDockedDataset(str(tmp_path), synthetic=True, catalog=catalog, num_synthetic=10, seed=5)
        mtime = os.path.getmtime(os.path.join(str(tmp_path), "processed"))
        ds2 = CrossDockedDataset(str(tmp_path), synthetic=True, catalog=catalog,
                                 num_synthetic=10, seed=5)
        assert len(ds2) == 10
        # No reprocessing (same mtime bucket) — files unchanged.
        assert os.path.getmtime(os.path.join(str(tmp_path), "processed")) >= mtime

    def test_force_rebuild(self, tmp_path, catalog):
        CrossDockedDataset(str(tmp_path), synthetic=True, catalog=catalog, num_synthetic=10, seed=5)
        ds = CrossDockedDataset(str(tmp_path), synthetic=True, catalog=catalog, num_synthetic=10,
                                seed=5, force_rebuild=True)
        assert len(ds) == 10


class TestBatching:
    def test_batch_collation(self, tmp_path, catalog):
        ds = CrossDockedDataset(str(tmp_path), synthetic=True, catalog=catalog, num_synthetic=12)
        batch = Batch.from_data_list([ds[i] for i in range(4)],
                                     follow_batch=["pocket_pos"])
        assert batch.pocket_pos.shape[1] == 3
        assert batch.pocket_pos_batch.max().item() == 3
        assert batch.handle_features.numel() == 4 * 64
        assert batch.target_synthon.shape == (4,)

    def test_dataloader(self, tmp_path, catalog):
        ds = CrossDockedDataset(str(tmp_path), synthetic=True, catalog=catalog, num_synthetic=12)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(batches) == 3
        assert batches[0].pocket_pos.shape[0] > 0


class TestRealMode:
    def test_pocket_without_ligand_requires_explicit_synthetic(self, assets_dir, tmp_path):
        """A lone sample_pocket.pdb (generation target) must never be silently
        used as real training data: real mode fails closed, and the caller must
        explicitly opt into synthetic mode."""
        import shutil

        from syntree.chemistry.catalog import SynthonCatalog

        data_dir = tmp_path / "crossdocked"
        data_dir.mkdir()
        shutil.copy(
            os.path.join(assets_dir["crossdocked_dir"], "sample_pocket.pdb"),
            data_dir / "sample_pocket.pdb",
        )
        catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
        with pytest.raises(RuntimeError, match="No CrossDocked pocket-ligand pairs"):
            CrossDockedDataset(str(data_dir), catalog=catalog, num_synthetic=5)
        # Explicit opt-in keeps the smoke path available.
        ds = CrossDockedDataset(
            str(data_dir), catalog=catalog, num_synthetic=5, synthetic=True
        )
        assert ds.use_synthetic

    def test_real_mode_with_pair(self, assets_dir, tmp_path):
        """Pocket + ligand SDF pair activates real mode."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        data_dir = tmp_path / "crossdocked"
        data_dir.mkdir()
        pocket_lines = open(
            os.path.join(assets_dir["crossdocked_dir"], "sample_pocket.pdb")
        ).read()
        # "lig1" hashes (crc32 % 1000 = 67) into the train bucket.
        (data_dir / "lig1_pocket.pdb").write_text(pocket_lines)

        lig = Chem.AddHs(
            Chem.MolFromSmiles("CC(C)COC(=O)C1CCCCC1")
        )
        AllChem.EmbedMolecule(lig, randomSeed=42)
        writer = Chem.SDWriter(str(data_dir / "lig1_ligand.sdf"))
        writer.write(lig)
        writer.close()

        catalog = SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)
        ds = CrossDockedDataset(str(data_dir), catalog=catalog)
        assert not ds.use_synthetic
        assert len(ds) >= 1
        sample = ds[0]
        assert sample.pocket_pos.shape[1] == 3
        assert sample.is_real_sample.item() is True
        assert 0 <= sample.target_synthon.item() < len(catalog)
        assert 0 <= sample.target_reaction_family_idx.item() < 7
        assert 0 <= sample.target_core_handle_idx.item() < 8

    def test_real_mode_requires_catalog(self, assets_dir, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        data_dir = tmp_path / "crossdocked"
        data_dir.mkdir()
        pocket_lines = open(
            os.path.join(assets_dir["crossdocked_dir"], "sample_pocket.pdb")
        ).read()
        (data_dir / "lig1_pocket.pdb").write_text(pocket_lines)
        lig = Chem.AddHs(
            Chem.MolFromSmiles("CC(C)COC(=O)C1CCCCC1")
        )
        AllChem.EmbedMolecule(lig, randomSeed=42)
        writer = Chem.SDWriter(str(data_dir / "lig1_ligand.sdf"))
        writer.write(lig)
        writer.close()

        with pytest.raises(ValueError, match="SynthonCatalog"):
            CrossDockedDataset(str(data_dir), catalog=None)

    def test_summary(self, tmp_path, catalog):
        ds = CrossDockedDataset(str(tmp_path), synthetic=True, catalog=catalog, num_synthetic=7)
        s = ds.summary()
        assert s["mode"] == "synthetic"
        assert s["num_samples"] == 7
        assert s["split"] == "train"
