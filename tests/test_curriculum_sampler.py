"""CurriculumSampler semantics (Approach C - mixed difficulty sampling).

Verifies, per docs/CURRICULUM.md:
  * level weights sum to 1 and never fall below the uniform floor at ANY
    stage (hard examples are never permanently removed),
  * sampling mass moves from easy to hard as stages advance,
  * indices are drawn only from the train pools, are reproducible per
    epoch, and cover every level from every stage,
  * stage advancement is milestone-driven with a patience fallback and a
    hard cap at the final stage,
  * state() survives a save/load round trip.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from syntree.engine.curriculum import CurriculumSampler  # noqa: E402


def make_sampler(n_stages=6, min_weight=0.03, patience=2, seed=0,
                 per_level=30, n_levels=9, num_samples=270,
                 milestone_metric="val_loss", improve_thresh=0.005,
                 lower_better=True, missing_levels=()):
    by_level = {
        lv: [lv * per_level + i for i in range(per_level)]
        for lv in range(1, n_levels + 1)
        if lv not in missing_levels
    }
    return CurriculumSampler(
        by_level,
        num_samples=num_samples,
        n_levels=n_levels,
        n_stages=n_stages,
        min_weight=min_weight,
        sigma=1.8,
        seed=seed,
        milestone_metric=milestone_metric,
        lower_better=lower_better,
        improve_thresh=improve_thresh,
        patience=patience,
    )


class TestWeights:
    def test_weights_sum_to_one_every_stage(self):
        sampler = make_sampler()
        for stage in range(sampler.n_stages):
            sampler.stage = stage
            weights = sampler.weights()
            assert abs(sum(weights.values()) - 1.0) < 1e-9
            assert len(weights) == 9

    def test_floor_respected_every_stage(self):
        sampler = make_sampler(min_weight=0.05)
        for stage in range(sampler.n_stages):
            sampler.stage = stage
            for lv, w in sampler.weights().items():
                assert w >= 0.05 - 1e-12

    def test_stage_zero_favors_easy(self):
        sampler = make_sampler()
        weights = sampler.weights()
        assert weights[1] > weights[9]

    def test_final_stage_favors_hard(self):
        sampler = make_sampler()
        sampler.stage = sampler.n_stages - 1
        weights = sampler.weights()
        assert weights[9] > weights[1]

    def test_missing_levels_adapt(self):
        sampler = make_sampler(missing_levels=(4, 5))
        weights = sampler.weights()
        assert 4 not in weights and 5 not in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_empty_sampler_rejected(self):
        with pytest.raises(ValueError):
            CurriculumSampler({}, num_samples=10)


class TestSampling:
    def test_epoch_reproducibility(self):
        sampler = make_sampler()
        sampler.set_epoch(3)
        first = list(sampler)
        sampler.set_epoch(3)
        second = list(sampler)
        assert first == second
        assert len(first) == sampler.num_samples

    def test_different_epochs_differ(self):
        sampler = make_sampler(num_samples=500)
        sampler.set_epoch(1)
        a = list(sampler)
        sampler.set_epoch(2)
        b = list(sampler)
        assert a != b  # overwhelming probability with 500 draws

    def test_indices_stay_in_train_pools(self):
        sampler = make_sampler()
        sampler.set_epoch(0)
        valid = {
            idx
            for idxs in sampler._by_level.values()
            for idx in idxs
        }
        assert set(list(sampler)).issubset(valid)

    def test_all_levels_sampled_at_every_stage(self):
        # With 9 levels x 30 items and 4,500 draws per stage, every level
        # must receive samples at every stage (floor guarantee).
        sampler = make_sampler(num_samples=4500)
        for stage in range(sampler.n_stages):
            sampler.stage = stage
            sampler.set_epoch(stage)
            list(sampler)
            counts = sampler.last_sampled_counts
            assert len(counts) == 9, f"stage {stage}: {counts}"
            for lv in range(1, 10):
                assert counts[lv] > 0

    def test_counts_follow_weights(self):
        sampler = make_sampler(num_samples=9000)
        sampler.set_epoch(11)
        list(sampler)
        weights = sampler.weights()
        counts = sampler.last_sampled_counts
        for lv in range(1, 10):
            expected = weights[lv] * 9000
            assert abs(counts[lv] - expected) < 0.12 * expected, lv


class TestStageAdvancement:
    def test_milestone_improvement_advances(self):
        sampler = make_sampler(patience=50, improve_thresh=0.01)
        advanced, reason = sampler.maybe_advance({"val_loss": 7.50})
        assert not advanced
        advanced, reason = sampler.maybe_advance({"val_loss": 7.40})
        assert advanced and "improved" in reason
        assert sampler.stage == 1

    def test_no_improvement_updates_reference(self):
        sampler = make_sampler(patience=50, improve_thresh=0.01)
        sampler.maybe_advance({"val_loss": 7.50})
        # 7.499 does not clear the 0.01 threshold, but becomes the new best
        sampler.maybe_advance({"val_loss": 7.499})
        assert sampler.stage == 0
        advanced, _ = sampler.maybe_advance({"val_loss": 7.45})
        assert advanced  # 7.45 <= 7.499 - 0.01

    def test_patience_fallback_advances(self):
        # patience=2 -> a stage is abandoned after 2 epochs without a
        # milestone; the baseline epoch counts, so the 2nd call advances.
        sampler = make_sampler(patience=2, improve_thresh=1e9)
        advanced, _ = sampler.maybe_advance({"val_loss": 7.50})  # epoch 1
        assert not advanced
        advanced, reason = sampler.maybe_advance({"val_loss": 7.49})  # epoch 2
        assert advanced and "patience" in reason
        assert sampler.stage == 1
        # the counter resets after an advance
        advanced, _ = sampler.maybe_advance({"val_loss": 7.48})
        assert not advanced

    def test_stage_caps_at_final(self):
        sampler = make_sampler(n_stages=3, patience=1)
        for _ in range(10):
            sampler.maybe_advance({"val_loss": 1.0})
        assert sampler.stage == 2  # n_stages - 1, never beyond

    def test_higher_better_metric(self):
        sampler = make_sampler(
            milestone_metric="val_acc", lower_better=False,
            improve_thresh=0.01, patience=50,
        )
        sampler.maybe_advance({"val_acc": 0.50})
        advanced, _ = sampler.maybe_advance({"val_acc": 0.60})
        assert advanced

    def test_missing_metric_uses_patience(self):
        sampler = make_sampler(patience=1)
        advanced, reason = sampler.maybe_advance({})
        assert advanced and "patience" in reason

    def test_nonfinite_metric_ignored(self):
        sampler = make_sampler(patience=100)
        advanced, _ = sampler.maybe_advance({"val_loss": float("nan")})
        assert not advanced
        assert sampler._ref is None


class TestStateRoundTrip:
    def test_state_survives_save_load(self):
        sampler = make_sampler()
        for _ in range(3):
            sampler.maybe_advance({"val_loss": 0.1})
        state = sampler.state()

        fresh = make_sampler()
        fresh.load_state(state)
        assert fresh.stage == sampler.stage
        assert fresh._epochs_since == sampler._epochs_since
        assert fresh._ref == sampler._ref

    def test_bad_state_recovers_to_fresh(self):
        sampler = make_sampler()
        sampler.load_state({"stage": "bogus"})
        assert sampler.stage == 0

    def test_stage_clamped_on_load(self):
        sampler = make_sampler(n_stages=4)
        sampler.load_state({"stage": 99})
        assert sampler.stage == 3
