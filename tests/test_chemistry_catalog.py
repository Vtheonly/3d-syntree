"""Unit tests for the synthon catalog."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from syntree.chemistry.catalog import (
    DEFAULT_EXEMPT_HANDLES,
    REQUIRED_COLUMNS,
    SynthonCatalog,
)


@pytest.fixture(scope="module")
def catalog_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("catalog") / "test_catalog.parquet"
    rows = []
    synthons = [
        # sp3-rich (pass the floor)
        ("OC(=O)C1CCCCC1", "carboxylic_acid", 1.0, 128.0),
        ("C1CC(N)CC1", "primary_secondary_amine", 1.0, 100.0),
        ("CC(C)CO", "alcohol", 1.0, 74.0),
        ("C1CCC(CC1)C=O", "aldehyde", 0.857, 112.0),
        # aryl (Fsp3-exempt partners)
        ("Brc1ccc(F)cc1", "aryl_halide", 0.0, 175.0),
        ("B(c1ccccc1)(O)O", "boronic_acid", 0.0, 122.0),
        # must be filtered out: flat + non-exempt handle
        ("c1ccccc1N", "primary_secondary_amine", 0.0, 93.0),
        # must be filtered out: too heavy
        ("CCCCCCCCCCCCCCCCCC(=O)O", "carboxylic_acid", 1.0, 284.0),
    ]
    for i, (smiles, handle, fsp3, mw) in enumerate(synthons):
        rows.append(
            {
                "id": f"CAT-{i:03d}",
                "smiles": smiles,
                "fsp3": fsp3,
                "mw": mw,
                "primary_handle": handle,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def catalog(catalog_path):
    return SynthonCatalog(catalog_path, embedding_dim=16)


class TestLoading:
    def test_loads_and_filters(self, catalog):
        # 8 rows in, 2 filtered out (flat aniline + overweight acid).
        assert len(catalog) == 6

    def test_aryl_partners_exempt_from_fsp3_floor(self, catalog):
        handles = set(catalog.df["primary_handle"])
        assert "aryl_halide" in handles
        assert "boronic_acid" in handles

    def test_flat_nonexcluded_dropped(self, catalog):
        # The flat aniline is gone.
        assert "c1ccccc1N" not in set(catalog.df["smiles"])

    def test_missing_columns_raise(self, tmp_path):
        path = tmp_path / "bad.parquet"
        pd.DataFrame({"id": ["x"], "smiles": ["C"]}).to_parquet(path)
        with pytest.raises(ValueError, match="must contain columns"):
            SynthonCatalog(str(path))

    def test_empty_after_filter_raises(self, tmp_path):
        path = tmp_path / "empty.parquet"
        pd.DataFrame(
            {
                "id": ["x"],
                "smiles": ["c1ccccc1N"],
                "fsp3": [0.0],
                "mw": [93.0],
                "primary_handle": ["primary_secondary_amine"],
            }
        ).to_parquet(path)
        with pytest.raises(ValueError, match="empty after filtering"):
            SynthonCatalog(str(path))

    def test_invalid_embedding_dim_raises(self, catalog_path):
        with pytest.raises(ValueError):
            SynthonCatalog(catalog_path, embedding_dim=0)

    def test_wrong_handle_tag_dropped(self, tmp_path):
        """Row claiming aldehyde but actually an amine must be dropped."""
        path = tmp_path / "mislabeled.parquet"
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "smiles": ["C1CC(N)CC1", "C1CCC(CC1)C=O"],
                "fsp3": [1.0, 0.857],
                "mw": [100.0, 112.0],
                "primary_handle": ["aldehyde", "aldehyde"],
            }
        ).to_parquet(path)
        cat = SynthonCatalog(str(path), embedding_dim=8)
        assert len(cat) == 1  # only the real aldehyde survives


class TestEmbeddings:
    def test_shape(self, catalog):
        assert catalog.embeddings.shape == (len(catalog), 16)

    def test_deterministic(self, catalog_path):
        c1 = SynthonCatalog(catalog_path, embedding_dim=16)
        c2 = SynthonCatalog(catalog_path, embedding_dim=16)
        assert torch.allclose(c1.embeddings, c2.embeddings)

    def test_chemically_informed(self, catalog_path):
        """Similar synthons should embed closer than dissimilar ones
        (at the production embedding dimensionality)."""
        cat = SynthonCatalog(catalog_path, embedding_dim=128)
        emb = cat.embeddings
        # cyclopentylamine vs isobutanol (both small sp3) vs bromoaryl.
        smiles = list(cat.df["smiles"])
        amine_idx = smiles.index("C1CC(N)CC1")
        alcohol_idx = smiles.index("CC(C)CO")
        aryl_idx = smiles.index("Brc1ccc(F)cc1")
        d_sp3 = (emb[amine_idx] - emb[alcohol_idx]).norm()
        d_aryl = (emb[amine_idx] - emb[aryl_idx]).norm()
        assert d_sp3 < d_aryl

    def test_identical_smiles_identical_embedding(self, tmp_path):
        """Structural fingerprints must map deterministically."""
        path = tmp_path / "dup.parquet"
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "smiles": ["C1CC(N)CC1", "C1CC(N)CC1", "Brc1ccc(F)cc1"],
                "fsp3": [1.0, 1.0, 0.0],
                "mw": [100.0, 100.0, 175.0],
                "primary_handle": [
                    "primary_secondary_amine",
                    "primary_secondary_amine",
                    "aryl_halide",
                ],
            }
        ).to_parquet(path)
        cat = SynthonCatalog(str(path), embedding_dim=64)
        assert torch.equal(cat.embeddings[0], cat.embeddings[1])
        assert not torch.equal(cat.embeddings[0], cat.embeddings[2])

    def test_unit_norm(self, catalog):
        norms = catalog.embeddings.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


class TestReactionMasks:
    def test_amide_mask(self, catalog):
        mask = catalog.get_reaction_mask("amide_coupling")
        valid = set(
            catalog.df.iloc[i]["primary_handle"]
            for i in torch.nonzero(mask == 0).flatten().tolist()
        )
        assert valid == {"carboxylic_acid", "primary_secondary_amine"}

    def test_mask_blocks_incompatible(self, catalog):
        mask = catalog.get_reaction_mask("suzuki_coupling")
        valid = set(
            catalog.df.iloc[i]["primary_handle"]
            for i in torch.nonzero(mask == 0).flatten().tolist()
        )
        assert valid == {"aryl_halide", "boronic_acid"}

    def test_core_restricted_mask(self, catalog):
        mask = catalog.get_reaction_mask("amide_coupling", core_handle="carboxylic_acid")
        valid = set(
            catalog.df.iloc[i]["primary_handle"]
            for i in torch.nonzero(mask == 0).flatten().tolist()
        )
        assert valid == {"primary_secondary_amine"}

    def test_unknown_reaction_raises(self, catalog):
        with pytest.raises(ValueError, match="Unknown reaction"):
            catalog.get_reaction_mask("grignard")

    def test_device_transfer(self, catalog):
        mask = catalog.get_reaction_mask("snar", device=torch.device("cpu"))
        assert mask.device.type == "cpu"


class TestAccessors:
    def test_get_mol(self, catalog):
        mol = catalog.get_mol(0)
        assert mol is not None and mol.GetNumAtoms() > 0

    def test_get_mol_explicit_h(self, catalog):
        mol = catalog.get_mol(0, explicit_hs=True)
        assert any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())

    def test_get_smiles_and_id(self, catalog):
        assert isinstance(catalog.get_smiles(0), str)
        assert catalog.get_id(0).startswith("CAT-")

    @pytest.mark.parametrize("bad_idx", [-1, 100, 9999])
    def test_out_of_range_raises(self, catalog, bad_idx):
        with pytest.raises(IndexError):
            catalog.get_smiles(bad_idx)
        with pytest.raises(IndexError):
            catalog.get_mol(bad_idx)

    def test_synthon_indices_for_handles(self, catalog):
        idx = catalog.synthon_indices_for_handles(["carboxylic_acid"])
        assert len(idx) == 1
        idx = catalog.synthon_indices_for_handles(["aryl_halide", "boronic_acid"])
        assert len(idx) == 2

    def test_available_handles(self, catalog):
        assert set(catalog.available_handles) == {
            "carboxylic_acid", "primary_secondary_amine", "alcohol",
            "aldehyde", "aryl_halide", "boronic_acid",
        }

    def test_len_and_repr(self, catalog):
        assert len(catalog) == len(catalog.df)
        assert "SynthonCatalog" in repr(catalog)
