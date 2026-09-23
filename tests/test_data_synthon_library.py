"""Unit tests for the synthon embedding store (HDF5 / NPZ)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from syntree.data.synthon_library import SynthonLibraryStore


class TestHdf5Store:
    def test_roundtrip(self, tmp_path):
        store = SynthonLibraryStore(str(tmp_path / "lib.h5"))
        emb = torch.randn(10, 16)
        store.save(emb, ids=[f"id{i}" for i in range(10)],
                   smiles=["CCO"] * 10, handles=["alcohol"] * 10,
                   source_catalog="cat.parquet", seed=42)
        assert store.exists()

        loaded = store.load()
        assert torch.allclose(loaded["embeddings"], emb)
        assert loaded["ids"] == [f"id{i}" for i in range(10)]
        assert loaded["handles"] == ["alcohol"] * 10
        assert loaded["meta"]["seed"] == 42
        assert loaded["meta"]["dim"] == 16
        assert loaded["meta"]["num_synthons"] == 10

    def test_overwrite(self, tmp_path):
        store = SynthonLibraryStore(str(tmp_path / "lib.h5"))
        store.save(torch.zeros(3, 4), ["a", "b", "c"], ["C", "C", "C"],
                   ["x", "x", "x"], "src", 1)
        store.save(torch.ones(3, 4), ["a", "b", "c"], ["C", "C", "C"],
                   ["x", "x", "x"], "src", 1)
        loaded = store.load()
        assert torch.allclose(loaded["embeddings"], torch.ones(3, 4))

    def test_missing_file_raises(self, tmp_path):
        store = SynthonLibraryStore(str(tmp_path / "missing.h5"))
        assert not store.exists()
        with pytest.raises(FileNotFoundError):
            store.load()

    def test_npz_fallback(self, tmp_path):
        """The .npz path works even without h5py."""
        store = SynthonLibraryStore(str(tmp_path / "lib.npz"))
        emb = torch.randn(5, 8)
        store.save(emb, [f"i{i}" for i in range(5)], ["CC"] * 5,
                   ["h"] * 5, "src", 7)
        loaded = store.load()
        assert torch.allclose(loaded["embeddings"], emb)
        assert loaded["meta"]["seed"] == 7

    def test_default_path(self):
        path = SynthonLibraryStore.default_path("data/enamine.parquet", 128)
        assert path.endswith("enamine_embeddings_128d.h5")
