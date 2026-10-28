"""Difficulty feature extraction and scoring (curriculum stage 1).

Verifies the reproducibility and leakage rules of the difficulty schema:
  * tensor features are read defensively from any PyG sample shape,
  * catalog (RDKit) features are cached per SMILES and tolerate bad rows,
  * DifficultyModel percentiles are fitted on the TRAIN rows only,
  * scores are [0, 1], monotone in each weighted feature, and clipped for
    out-of-range (validation) rows.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syntree.engine.curriculum import (  # noqa: E402
    DEFAULT_FEATURE_WEIGHTS,
    DifficultyModel,
    tensor_sample_features,
    catalog_synthon_features,
)


def make_sample(pocket_n=10, ligand_n=5, family=1, handle=2, synthon=0):
    return Data(
        pocket_pos=torch.zeros(pocket_n, 3),
        pocket_z=torch.zeros(pocket_n, dtype=torch.long),
        pocket_charge=torch.full((pocket_n,), 0.5),
        ligand_pos=torch.zeros(ligand_n, 3),
        ligand_z=torch.zeros(ligand_n, dtype=torch.long),
        ligand_charge=torch.full((ligand_n,), 0.25),
        target_synthon=torch.tensor([synthon], dtype=torch.long),
        target_reaction_family_idx=torch.tensor([family], dtype=torch.long),
        target_core_handle_idx=torch.tensor([handle], dtype=torch.long),
        target_dihedral=torch.tensor([1.23], dtype=torch.float32),
    )


class TestTensorFeatures:
    def test_counts_and_charges(self):
        feats = tensor_sample_features(make_sample(pocket_n=10, ligand_n=5))
        assert feats["pocket_n_atoms"] == 10.0
        assert feats["ligand_n_atoms"] == 5.0
        assert feats["pocket_mean_abs_charge"] == pytest.approx(0.5)
        assert feats["ligand_mean_abs_charge"] == pytest.approx(0.25)
        assert feats["target_reaction_family_idx"] == 1
        assert feats["target_core_handle_idx"] == 2
        assert feats["target_synthon"] == 0

    def test_pos_z_fallback_naming(self):
        sample = Data(
            pos=torch.zeros(7, 3),
            z=torch.zeros(7, dtype=torch.long),
        )
        feats = tensor_sample_features(sample)
        assert feats["pocket_n_atoms"] == 7.0

    def test_deterministic(self):
        a = tensor_sample_features(make_sample(pocket_n=13, ligand_n=4))
        b = tensor_sample_features(make_sample(pocket_n=13, ligand_n=4))
        assert a == b

    def test_empty_ligand(self):
        feats = tensor_sample_features(make_sample(ligand_n=0))
        assert feats["ligand_n_atoms"] == 0.0
        assert feats["ligand_mean_abs_charge"] == 0.0


class TestCatalogFeatures:
    def test_phenethyl_alcohol(self):
        feats = catalog_synthon_features("c1ccccc1CCO")
        assert feats["synthon_n_atoms"] == 9.0  # 6 ring C + 2 side-chain C + O
        assert feats["synthon_n_rings"] == 1.0
        assert feats["synthon_n_hetero"] == 1.0  # the oxygen
        assert 0.0 < feats["synthon_symmetry_ratio"] <= 1.0

    def test_invalid_smiles_returns_zeros(self):
        feats = catalog_synthon_features("not_a_smiles!!")
        assert feats["synthon_n_atoms"] == 0.0
        assert feats["synthon_n_rings"] == 0.0

    def test_cache_returns_identical_rows(self):
        first = catalog_synthon_features("CCO")
        second = catalog_synthon_features("CCO")
        assert first == second


class TestDifficultyModel:
    def _rows(self, n=50):
        # pocket size varies smoothly; everything else constant
        return [
            {
                "pocket_n_atoms": float(10 + i),
                "ligand_n_atoms": 5.0,
                "pocket_mean_abs_charge": 0.1,
                "ligand_mean_abs_charge": 0.1,
                "synthon_n_atoms": 6.0,
                "synthon_n_rings": 0.0,
                "synthon_n_rotatable": 1.0,
                "synthon_n_hetero": 1.0,
                "synthon_symmetry_ratio": 0.8,
                "reaction_family_rarity": 1.0,
                "core_handle_rarity": 1.0,
            }
            for i in range(n)
        ]

    def test_fit_train_only_and_score_monotone(self):
        model = DifficultyModel({"pocket_n_atoms": 1.0})
        rows = self._rows()
        model.fit(rows)
        scores = [model.score(r) for r in rows]
        assert all(0.0 <= s <= 1.0 for s in scores)
        # bigger pocket = harder, given the single weighted feature
        assert scores[-1] > scores[0]
        assert scores == sorted(scores)

    def test_out_of_range_val_row_clipped(self):
        model = DifficultyModel({"pocket_n_atoms": 1.0})
        rows = self._rows()
        model.fit(rows)
        extreme = dict(rows[0])
        extreme["pocket_n_atoms"] = 10_000.0  # a validation outlier
        assert model.score(extreme) <= 1.0
        assert model.score(extreme) >= model.score(rows[-1])

    def test_default_weights_cover_schema(self):
        assert set(DEFAULT_FEATURE_WEIGHTS).issubset(
            set(DifficultyModel().weights)
        )

    def test_zero_weight_feature_ignored(self):
        model = DifficultyModel({"pocket_n_atoms": 1.0, "synthon_n_rings": 0.0})
        model.fit(self._rows())
        a = dict(self._rows()[0])
        b = dict(a)
        b["synthon_n_rings"] = 99.0
        assert model.score(a) == model.score(b)

    def test_higher_feature_means_higher_score(self):
        model = DifficultyModel()
        model.fit(self._rows())
        easy = dict(self._rows()[0])
        hard = dict(easy)
        for key in (
            "ligand_n_atoms", "pocket_n_atoms", "synthon_n_atoms",
            "synthon_n_rings", "synthon_n_rotatable", "synthon_n_hetero",
            "reaction_family_rarity", "core_handle_rarity",
        ):
            hard[key] = hard[key] + 50.0
        assert model.score(hard) > model.score(easy)
