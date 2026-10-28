"""9-level difficulty assignment (curriculum stage 2).

Verifies the level assignment contract from docs/CURRICULUM.md:
  * train split is cut into 9 approximately-balanced quantile bins,
  * bin edges depend on the TRAIN split only (validation rows can never
    shift the train levels - no leakage),
  * L1 holds the easiest and L9 the hardest examples,
  * assignment is deterministic and cacheable (identical on rebuild).
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syntree.engine.curriculum import CurriculumController  # noqa: E402


STUB_SMILES = [
    "CCO",           # 0: tiny
    "CCCC",          # 1
    "CCCCCC",        # 2
    "c1ccccc1",      # 3: ring
    "c1ccccc1CCO",   # 4: ring + chain
    "CCCCCCCCCCCC",  # 5: long chain
]


class StubDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        pocket_n, ligand_n, family, handle, synthon = self.rows[idx]
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
            target_dihedral=torch.tensor([1.0], dtype=torch.float32),
        )


class StubCatalog:
    def __init__(self):
        import pandas as pd

        self.df = pd.DataFrame({"smiles": STUB_SMILES})

    @property
    def embedding_signature(self):
        return {
            "encoder": "stub",
            "dim": 8,
            "ids_hash": "stub-ids-hash",
            "num_synthons": len(self.df),
        }


def make_rows(n, hard_step=2, seed_families=None):
    """Deterministic rows whose difficulty grows with the index."""
    rows = []
    for i in range(n):
        pocket_n = 10 + i * hard_step
        ligand_n = 4 + (i % 7)
        family = (seed_families or [0, 1, 2])[i % 3]
        handle = 1 + (i % 4)
        synthon = i % len(STUB_SMILES)
        rows.append((pocket_n, ligand_n, family, handle, synthon))
    return rows


def build_controller(train_rows, val_rows, tmp_path=None, **overrides):
    cfg = {
        "enabled": True,
        "n_levels": 9,
        "n_stages": 4,
        "seed": 42,
        "min_weight": 0.03,
        "sigma": 1.8,
        "milestone_metric": "val_loss",
        "patience": 2,
        "improve_thresh": 0.005,
        "cache_path": str(tmp_path / "levels.json") if tmp_path else None,
    }
    cfg.update(overrides)
    return CurriculumController.build(
        cfg,
        StubDataset(train_rows),
        StubDataset(val_rows),
        StubCatalog(),
        output_dir=str(tmp_path) if tmp_path else ".",
        log_fn=lambda *_: None,
    )


class TestLevelAssignment:
    def test_nine_balanced_bins(self, tmp_path):
        controller = build_controller(make_rows(90), make_rows(18), tmp_path)
        counts = controller.level_counts_train
        assert sorted(counts) == list(range(1, 10)), "all 9 levels must exist"
        for level, count in counts.items():
            assert 90 / 9 - 3 <= count <= 90 / 9 + 3, (
                f"level {level} not balanced: {count}"
            )
        assert sum(counts.values()) == 90

    def test_l1_easiest_l9_hardest(self, tmp_path):
        controller = build_controller(make_rows(90), make_rows(18), tmp_path)
        scores = [
            controller.model.score(row)
            for row in controller._feature_rows_train
        ]
        by_level = {lv: [] for lv in range(1, 10)}
        for score, lv in zip(scores, controller.levels_train):
            by_level[lv].append(score)
        mean_easy = sum(by_level[1]) / len(by_level[1])
        mean_hard = sum(by_level[9]) / len(by_level[9])
        assert mean_easy < mean_hard
        # every L1 example is easier than every L9 example under
        # smooth monotone features (quantile binning guarantees this)
        assert max(by_level[1]) <= min(by_level[9])

    def test_levels_are_1_to_9_ints(self, tmp_path):
        controller = build_controller(make_rows(45), make_rows(9), tmp_path)
        assert all(isinstance(lv, int) for lv in controller.levels_train)
        assert all(1 <= lv <= 9 for lv in controller.levels_train)
        assert all(1 <= lv <= 9 for lv in controller.levels_val)


class TestLeakageRules:
    def test_val_rows_never_change_train_levels(self, tmp_path):
        train_rows = make_rows(90)
        # First build: validation = same distribution.
        c1 = build_controller(train_rows, make_rows(18), tmp_path / "a")
        # Second build: same train rows, WILDLY different validation rows.
        val_outliers = [(5000, 50, 2, 3, 5)] * 18
        c2 = build_controller(train_rows, val_outliers, tmp_path / "b")
        assert c1.levels_train == c2.levels_train
        assert c1.edges == c2.edges
        # the extreme val rows are still capped at the hardest level
        assert all(lv == 9 for lv in c2.levels_val)

    def test_val_levels_use_train_edges(self, tmp_path):
        train_rows = make_rows(90)
        val_rows = [(12, 5, 1, 2, 1)] * 5  # easy rows
        controller = build_controller(train_rows, val_rows, tmp_path)
        # the easiest-possible rows must not be assigned the hardest level
        assert all(lv <= 3 for lv in controller.levels_val)

    def test_split_membership_untouched(self, tmp_path):
        """Curriculum must not re-split: train/val membership is fixed by the
        dataset objects, and the sampler only draws train indices."""
        train_rows = make_rows(90)
        controller = build_controller(train_rows, make_rows(18), tmp_path)
        pool = {
            idx
            for idxs in controller.sampler._by_level.values()
            for idx in idxs
        }
        assert pool == set(range(90)), "sampler may only see train indices"


class TestDeterminismAndCache:
    def test_rebuild_is_identical(self, tmp_path):
        rows = make_rows(60)
        c1 = build_controller(rows, make_rows(12), tmp_path / "one.json")
        c2 = build_controller(rows, make_rows(12), tmp_path / "two.json")
        assert c1.levels_train == c2.levels_train
        assert c1.levels_val == c2.levels_val

    def test_cache_round_trip(self, tmp_path):
        rows = make_rows(60)
        cache_path = tmp_path / "levels.json"
        c1 = build_controller(
            rows, make_rows(12), tmp_path=tmp_path, cache_path=str(cache_path)
        )
        assert cache_path.exists()
        assert not c1.from_cache
        payload = json.loads(cache_path.read_text())
        assert payload["signature"]["schema"] == 1
        assert len(payload["levels_train"]) == 60

        c2 = build_controller(
            rows, make_rows(12), tmp_path=tmp_path, cache_path=str(cache_path)
        )
        assert c2.from_cache, "second build must reuse the cache"
        assert c2.levels_train == c1.levels_train

    def test_cache_invalidates_on_config_change(self, tmp_path):
        rows = make_rows(60)
        cache_path = tmp_path / "levels.json"
        c1 = build_controller(
            rows, make_rows(12), tmp_path=tmp_path, cache_path=str(cache_path),
            n_levels=9,
        )
        c2 = build_controller(
            rows, make_rows(12), tmp_path=tmp_path, cache_path=str(cache_path),
            n_levels=5,
        )
        assert not c2.from_cache
        assert len(c2.edges) == 5
        assert all(1 <= lv <= 5 for lv in c2.levels_train)
