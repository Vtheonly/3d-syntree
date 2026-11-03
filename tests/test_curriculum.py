"""#27 TRAIN-02 — the Approach C quantile curriculum engine tests.

CPU-side verification of both issue acceptance criteria plus the trainer
integration contract:

1. **All 9 levels are sampled in every epoch** — asserted through the
   sampler's recorded per-epoch level histograms over a synthetic
   900-item difficulty table at every one of the 6 stages, and through
   the empirical level frequencies matching the stage distribution.
2. **Milestone-driven progression transitions smoothly through all 6
   stages over 40 epochs** — a 40-epoch validation-loss schedule (mixed
   milestones, sub-milestone improvements and plateaus) drives the stage
   machine through every stage, monotonically, one stage at a time.

Beyond the two pinned criteria: the floor-exactness of the level
distribution at every stage, balanced quantile binning, the milestone
machine's patience fallback and no-regression guarantees, exact step
count, determinism and exact-epoch re-iteration (the trainer's
resume contract), state-dict round-trips, fail-closed guards, and the
``ResilientTrainer`` integration (wiring guards, telemetry records,
checkpoint round-trips including the curriculum state).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from syntree.data.supervised import (  # noqa: E402
    SupervisedTrajectoryDataset,
    supervised_collate,
)
from syntree.engine.curriculum import (  # noqa: E402
    CURRICULUM_FEATURE_NAMES,
    CURRICULUM_NUM_LEVELS,
    CURRICULUM_NUM_STAGES,
    CURRICULUM_PROBABILITY_FLOOR,
    STAGE_PATIENCE,
    VAL_IMPROVEMENT_MILESTONE,
    CurriculumError,
    CurriculumSampler,
    difficulty_features,
    difficulty_scores,
    level_distribution,
    quantile_levels,
    stage_center,
    standardize_features,
)
from syntree.engine.trainer import (  # noqa: E402
    ResilientTrainer,
    TrainerProfile,
    TrainerStateError,
    load_profile,
)
from tests.test_supervised_dataset import _context  # noqa: E402  (the amide fixture)
from tests.test_trainer import _mini_catalog_tensors, _test_profile  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SYNTHETIC = 900


def _synthetic_features(n: int = N_SYNTHETIC, seed: int = 7) -> np.ndarray:
    """A varied ``[n, 5]`` difficulty table over the five issue-named axes."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.integers(50, 400, size=n).astype(np.float64),  # pocket_atoms
            rng.integers(0, 12, size=n).astype(np.float64),  # rotatable_bonds
            rng.integers(0, 6, size=n).astype(np.float64),  # rings
            rng.random(n) * 20.0,  # pocket_charge_magnitude
            rng.random(n),  # handle_rarity
        ]
    )


def _sampler(**overrides) -> CurriculumSampler:
    kwargs = dict(seed=2026)
    kwargs.update(overrides)
    return CurriculumSampler(_synthetic_features(), **kwargs)


def _advance_to_stage(sampler: CurriculumSampler, target: int) -> None:
    """Drive the milestone machine to ``target`` via strong milestones."""
    sampler.observe_validation_loss(8.0)  # the baseline observation
    value = 7.0
    while sampler.stage < target:
        sampler.observe_validation_loss(value)
        value -= 1.0


# ---------------------------------------------------------------------------
# Difficulty ingestion (the real supervised fixture)
# ---------------------------------------------------------------------------


class TestDifficultyIngestion:
    def test_feature_names_are_the_five_issue_axes(self):
        assert CURRICULUM_FEATURE_NAMES == (
            "pocket_atoms",
            "rotatable_bonds",
            "rings",
            "pocket_charge_magnitude",
            "handle_rarity",
        )

    def test_features_over_the_real_train_fixture(self):
        context = _context("train")
        dataset = SupervisedTrajectoryDataset(
            [context, context, context], split="train", augment=True, seed=2026
        )
        features = difficulty_features(dataset)
        assert features.shape == (len(dataset), 5)
        assert np.all(np.isfinite(features))
        assert np.all(features >= 0.0)
        # the pocket axis is the fixture's capped pocket atom count
        assert set(features[:, 0].tolist()) == {float(len(context.pocket_z))}
        # the pre-seed item (no intermediate) carries zero flexibility
        assert features[0, 1] == 0.0 and features[0, 2] == 0.0

    def test_non_train_split_is_refused(self):
        dataset = SupervisedTrajectoryDataset(
            [_context("val")], split="val", augment=False, seed=2026
        )
        with pytest.raises(CurriculumError, match="train split"):
            difficulty_features(dataset)
        with pytest.raises(CurriculumError, match="train split"):
            CurriculumSampler.from_dataset(dataset)

    def test_a_non_supervised_object_is_refused(self):
        from types import SimpleNamespace

        with pytest.raises(CurriculumError, match="item table"):
            difficulty_features(SimpleNamespace(split="train"))
        with pytest.raises(CurriculumError, match="train split"):
            difficulty_features([1, 2, 3])


# ---------------------------------------------------------------------------
# Standardization, scores, and the balanced quantile bins
# ---------------------------------------------------------------------------


class TestScoresAndBins:
    def test_constant_columns_contribute_nothing(self):
        features = np.zeros((90, 5))
        features[:, 0] = np.arange(90)
        standardized = standardize_features(features)
        assert np.allclose(standardized[:, 1:], 0.0)
        assert abs(standardized[:, 0].std() - 1.0) < 1e-12

    def test_scores_finite_and_ranked_by_difficulty(self):
        features = _synthetic_features()
        # two synthetic rows dominating / dominated on every axis
        features[0] = features.max(axis=0) + 1.0
        features[1] = features.min(axis=0) - 1.0
        scores = difficulty_scores(features)
        assert scores.shape == (features.shape[0],)
        assert np.all(np.isfinite(scores))
        assert scores[0] > scores[1]
        assert scores[0] == scores.max()
        assert scores[1] == scores.min()

    def test_bins_are_quantile_balanced(self):
        levels = quantile_levels(difficulty_scores(_synthetic_features()))
        counts = np.bincount(levels, minlength=10)[1:]
        assert counts.sum() == N_SYNTHETIC
        assert counts.max() - counts.min() <= 1
        assert (counts == N_SYNTHETIC // 9).all() or (counts == N_SYNTHETIC // 9 + 1).all()

    def test_levels_are_monotone_in_score(self):
        scores = difficulty_scores(_synthetic_features())
        levels = quantile_levels(scores)
        order = np.argsort(scores)
        assert np.all(np.diff(levels[order]) >= 0)

    def test_fewer_items_than_levels_fails_closed(self):
        with pytest.raises(CurriculumError, match="non-empty"):
            quantile_levels(np.zeros(8))

    def test_ties_are_deterministic(self):
        scores = np.zeros(90)
        assert np.array_equal(quantile_levels(scores), quantile_levels(scores))


# ---------------------------------------------------------------------------
# The 6-stage level distribution
# ---------------------------------------------------------------------------


class TestLevelDistribution:
    def test_issue_pinned_constants(self):
        assert CURRICULUM_NUM_LEVELS == 9
        assert CURRICULUM_NUM_STAGES == 6
        assert CURRICULUM_PROBABILITY_FLOOR == 0.03
        assert VAL_IMPROVEMENT_MILESTONE == 0.005
        assert STAGE_PATIENCE == 2

    def test_floor_is_exact_at_every_stage_including_extremes(self):
        for stage in range(1, 7):
            probabilities = level_distribution(stage)
            assert probabilities.shape == (9,)
            assert abs(probabilities.sum() - 1.0) < 1e-12
            assert probabilities.min() >= CURRICULUM_PROBABILITY_FLOOR - 1e-15

    def test_mass_centres_move_from_easy_to_hard(self):
        stage1, stage6 = level_distribution(1), level_distribution(6)
        assert int(np.argmax(stage1)) == 0  # level 1
        assert int(np.argmax(stage6)) == 8  # level 9
        assert stage1[:3].sum() > stage6[:3].sum()
        assert stage1[6:].sum() < stage6[6:].sum()
        centers = [stage_center(s) for s in range(1, 7)]
        assert centers[0] == 1.0 and abs(centers[-1] - 9.0) < 1e-12
        assert all(b > a for a, b in zip(centers, centers[1:]))

    def test_floors_out_of_contract_are_refused(self):
        with pytest.raises(CurriculumError, match="at least"):
            CurriculumSampler(_synthetic_features(), floor=0.029)
        with pytest.raises(CurriculumError, match="floor"):
            CurriculumSampler(_synthetic_features(), floor=0.2)  # >= 1/9

    def test_kernel_sigma_must_be_positive(self):
        with pytest.raises(CurriculumError, match="kernel_sigma"):
            CurriculumSampler(_synthetic_features(), kernel_sigma=0.0)

    def test_stage_out_of_range_fails_closed(self):
        with pytest.raises(CurriculumError, match="stage"):
            stage_center(7)
        with pytest.raises(CurriculumError, match="stage"):
            stage_center(0)


# ---------------------------------------------------------------------------
# The epoch draw: exact step count, coverage, determinism
# ---------------------------------------------------------------------------


class TestEpochDraws:
    def test_exact_step_count_every_epoch_every_stage(self):
        sampler = _sampler()
        for stage in range(1, 7):
            _advance_to_stage(sampler, stage)
            sampler.set_epoch(stage)
            drawn = list(iter(sampler))
            assert len(drawn) == sampler.num_samples == N_SYNTHETIC
            assert len(set(drawn)) <= N_SYNTHETIC
            assert all(0 <= index < N_SYNTHETIC for index in drawn)

    def test_all_nine_levels_sampled_every_epoch(self):
        """Issue acceptance criterion 1 (via the recorded histograms)."""
        sampler = _sampler()
        for stage in range(1, 7):
            _advance_to_stage(sampler, stage)
            sampler.set_epoch(10 + stage)
            drawn = np.asarray(list(iter(sampler)))
            histogram = np.bincount(sampler.levels[drawn], minlength=10)[1:]
            assert (histogram > 0).all(), f"stage {stage}: {histogram}"
            # the histogram equals the sampler's recorded level counts
            assert histogram.tolist() == sampler.last_level_counts
            assert sum(sampler.last_level_counts) == N_SYNTHETIC

    def test_empirical_frequencies_match_the_stage_distribution(self):
        sampler = _sampler()
        for stage in (1, 3, 6):
            _advance_to_stage(sampler, stage)
            sampler.set_epoch(stage)
            drawn = np.asarray(list(iter(sampler)))
            frequency = np.bincount(sampler.levels[drawn], minlength=10)[1:] / N_SYNTHETIC
            # 3-sigma-safe multinomial tolerance at N=900 (deterministic seed)
            assert np.allclose(frequency, sampler.level_probabilities(), atol=0.03)

    def test_reiteration_of_an_epoch_is_bit_identical(self):
        sampler = _sampler()
        sampler.set_epoch(4)
        first, second = list(iter(sampler)), list(iter(sampler))
        assert first == second
        # a different epoch draws a different stream
        sampler.set_epoch(5)
        assert list(iter(sampler)) != first

    def test_different_stages_shift_the_sampled_mass(self):
        sampler = _sampler()
        _advance_to_stage(sampler, 1)
        sampler.set_epoch(1)
        early = np.bincount(sampler.levels[np.asarray(list(iter(sampler)))], minlength=10)[1:]
        _advance_to_stage(sampler, 6)
        sampler.set_epoch(2)
        late = np.bincount(sampler.levels[np.asarray(list(iter(sampler)))], minlength=10)[1:]
        assert early[:3].sum() > late[:3].sum()
        assert early[6:].sum() < late[6:].sum()

    def test_constructor_guards(self):
        with pytest.raises(CurriculumError, match="features"):
            CurriculumSampler(np.zeros((90, 4)))
        with pytest.raises(CurriculumError, match="non-finite"):
            CurriculumSampler(
                np.concatenate([_synthetic_features(90), [[np.nan] * 5]])
            )
        with pytest.raises(CurriculumError, match="non-empty"):
            CurriculumSampler(_synthetic_features(8))


# ---------------------------------------------------------------------------
# The milestone machine
# ---------------------------------------------------------------------------


class TestMilestoneMachine:
    def test_first_observation_is_the_baseline(self):
        sampler = _sampler()
        assert sampler.observe_validation_loss(2.0) is False
        assert sampler.stage == 1 and sampler.best_val_loss == 2.0

    def test_milestone_advances_and_resets_patience(self):
        sampler = _sampler()
        sampler.observe_validation_loss(2.0)
        assert sampler.observe_validation_loss(1.994) is True  # 0.006 >= 0.005
        assert sampler.stage == 2 and sampler.patience == 0

    def test_sub_milestone_improvements_trigger_the_patience_fallback(self):
        sampler = _sampler()
        sampler.observe_validation_loss(2.0)
        assert sampler.observe_validation_loss(1.997) is False  # 0.003 < 0.005
        assert sampler.patience == 1
        assert sampler.observe_validation_loss(1.995) is True  # fallback at 2
        assert sampler.stage == 2 and sampler.patience == 0

    def test_worse_losses_never_regress_the_stage(self):
        sampler = _sampler()
        sampler.observe_validation_loss(2.0)  # the baseline
        sampler.observe_validation_loss(3.5)  # non-improving: patience 1
        assert sampler.stage == 1
        sampler.observe_validation_loss(4.0)  # patience 2: the fallback fires
        assert sampler.stage == 2
        sampler.observe_validation_loss(5.0)  # patience 1 again (reset)
        assert sampler.stage == 2  # neither advance nor regress
        assert sampler.best_val_loss == 2.0  # the running best only tightens

    def test_stage_is_capped_at_six(self):
        sampler = _sampler()
        _advance_to_stage(sampler, 6)
        assert sampler.stage == 6
        assert sampler.observe_validation_loss(0.0) is False
        assert sampler.stage == 6

    def test_non_finite_validation_loss_fails_closed(self):
        sampler = _sampler()
        with pytest.raises(CurriculumError, match="finite"):
            sampler.observe_validation_loss(float("nan"))
        with pytest.raises(CurriculumError, match="finite"):
            sampler.observe_validation_loss(float("inf"))

    def test_forty_epoch_schedule_transitions_smoothly_through_all_stages(self):
        """Issue acceptance criterion 2: all 6 stages over 40 epochs."""
        sampler = _sampler()
        schedule = (
            [2.000]  # epoch 0: the baseline
            + [1.996, 1.992, 1.988, 1.984]  # sub-milestone improvements
            + [1.964]  # a milestone (0.020)
            + [2.5, 3.0]  # a plateau of worsening losses
            + [1.9]  # a milestone over the running best
            + [1.9] * 31  # the annealed tail (capped at stage 6)
        )
        assert len(schedule) == 40
        stages = [sampler.stage]
        for epoch, value in enumerate(schedule, start=0):
            sampler.set_epoch(epoch)
            sampler.observe_validation_loss(value)
            stages.append(sampler.stage)
        assert set(stages) == {1, 2, 3, 4, 5, 6}
        assert all(b >= a for a, b in zip(stages, stages[1:]))  # monotone
        assert all(b - a <= 1 for a, b in zip(stages, stages[1:]))  # no skips
        assert stages[-1] == 6
        assert len(sampler.transitions) == 5


# ---------------------------------------------------------------------------
# The checkpointable state
# ---------------------------------------------------------------------------


class TestCurriculumState:
    def _worked_sampler(self) -> CurriculumSampler:
        sampler = _sampler()
        sampler.set_epoch(3)
        sampler.observe_validation_loss(2.0)
        sampler.observe_validation_loss(1.99)  # milestone advance
        list(iter(sampler))  # a recorded epoch histogram
        return sampler

    def test_state_roundtrip_is_exact_and_reproduces_draws(self):
        sampler = self._worked_sampler()
        state = sampler.state_dict()
        # JSON-ability (the checkpoint payload survives torch.save/serialization)
        json.dumps(state)
        restored = _sampler()
        restored.load_state_dict(json.loads(json.dumps(state)))
        assert restored.stage == sampler.stage
        assert restored.patience == sampler.patience
        assert restored.best_val_loss == sampler.best_val_loss
        assert restored.epoch == sampler.epoch
        assert restored.transitions == sampler.transitions
        assert restored.level_histogram(3) == sampler.level_histogram(3)
        sampler.set_epoch(3)
        restored.set_epoch(3)
        assert list(iter(sampler)) == list(iter(restored))

    def test_configuration_mismatch_fails_closed(self):
        state = self._worked_sampler().state_dict()
        other = CurriculumSampler(_synthetic_features(seed=8), seed=2027)
        with pytest.raises(CurriculumError, match="mismatch"):
            other.load_state_dict(state)

    def test_schema_version_mismatch_fails_closed(self):
        state = self._worked_sampler().state_dict()
        state["schema_version"] = 99
        with pytest.raises(CurriculumError, match="schema"):
            _sampler().load_state_dict(state)


# ---------------------------------------------------------------------------
# The trainer integration
# ---------------------------------------------------------------------------


def _mkdtmp() -> str:
    import tempfile

    return tempfile.mkdtemp(prefix="syntree_curriculum_test_")


def _trainer(run_dir: str, *, curriculum, profile=None, resume=False):
    block_classes, seed_rows, available = _mini_catalog_tensors()
    policy = __import__(
        "syntree.models.policy", fromlist=["SynTreePolicy"]
    ).SynTreePolicy(catalog_size=3)
    return ResilientTrainer(
        policy,
        profile if profile is not None else _test_profile(max_epochs=4),
        run_dir=run_dir,
        catalog_block_handle_classes=block_classes,
        catalog_seed_rows=seed_rows,
        available_block_classes=available,
        curriculum=curriculum,
        resume=resume,
    )


def _curriculum_loaders(batch_size: int = 4):
    train_context = _context("train")
    val_context = _context("val")
    train_dataset = SupervisedTrajectoryDataset(
        [train_context, train_context, train_context],
        split="train",
        augment=True,
        seed=2026,
    )
    val_dataset = SupervisedTrajectoryDataset(
        [val_context, val_context], split="val", augment=False, seed=2026
    )
    curriculum = CurriculumSampler.from_dataset(train_dataset, seed=2026)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=curriculum,
        collate_fn=supervised_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=supervised_collate
    )
    return curriculum, train_loader, val_loader


class TestTrainerIntegration:
    def test_fit_with_curriculum_logs_histograms_and_advances(self):
        run_dir = _mkdtmp()
        curriculum, train_loader, val_loader = _curriculum_loaders()
        trainer = _trainer(run_dir, curriculum=curriculum)
        result = trainer.fit(train_loader, val_loader=val_loader)
        assert result["stop_reason"] == "max_epochs_reached"
        assert result["curriculum"]["stage"] == curriculum.stage
        # the curriculum telemetry records carry the level histograms
        records = [
            json.loads(line)
            for line in open(os.path.join(run_dir, "telemetry.jsonl"))
        ]
        curriculum_records = [r for r in records if r.get("phase") == "curriculum"]
        assert len(curriculum_records) == 4  # one per completed epoch
        for record in curriculum_records:
            counts = record["level_counts"]
            assert len(counts) == 9
            assert sum(counts) == len(train_loader.dataset)
            assert math.isfinite(record["validation_loss"])
            assert len(record["level_probabilities"]) == 9
            assert min(record["level_probabilities"]) >= 0.03
        # 4 epochs = baseline + 3 further observations: at least one
        # patience-fallback or milestone advance must have fired
        assert curriculum.stage >= 2
        assert len(curriculum.transitions) >= 1

    def test_curriculum_requires_a_validation_loader(self):
        curriculum, train_loader, _ = _curriculum_loaders()
        trainer = _trainer(_mkdtmp(), curriculum=curriculum)
        with pytest.raises(TrainerStateError, match="validation loader"):
            trainer.fit(train_loader, val_loader=None)

    def test_train_loader_must_be_wired_with_the_sampler(self):
        curriculum, train_loader, val_loader = _curriculum_loaders()
        unwired = DataLoader(
            train_loader.dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=supervised_collate,
        )
        trainer = _trainer(_mkdtmp(), curriculum=curriculum)
        with pytest.raises(TrainerStateError, match="wired"):
            trainer.fit(unwired, val_loader=val_loader)

    def test_validation_signal_must_come_from_the_val_split(self):
        curriculum, train_loader, val_loader = _curriculum_loaders()
        train_as_val = DataLoader(
            train_loader.dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=supervised_collate,
        )
        trainer = _trainer(_mkdtmp(), curriculum=curriculum)
        with pytest.raises(TrainerStateError, match="val split"):
            trainer.fit(train_loader, val_loader=train_as_val)

    def test_checkpoint_roundtrip_restores_the_curriculum_state(self):
        run_dir = _mkdtmp()
        curriculum, train_loader, val_loader = _curriculum_loaders()
        profile = _test_profile(max_epochs=2)
        trainer = _trainer(run_dir, curriculum=curriculum, profile=profile)
        trainer.fit(train_loader, val_loader=val_loader)
        saved_stage = curriculum.stage
        saved_transitions = [dict(t) for t in curriculum.transitions]
        saved_best = curriculum.best_val_loss

        fresh_curriculum, fresh_train, fresh_val = _curriculum_loaders()
        resumed = _trainer(
            run_dir,
            curriculum=fresh_curriculum,
            profile=profile,
            resume=True,
        )
        assert resumed.progress.global_step == trainer.progress.global_step
        assert fresh_curriculum.stage == saved_stage
        assert fresh_curriculum.transitions == saved_transitions
        assert fresh_curriculum.best_val_loss == saved_best
        # and training continues from the restored curriculum (the frozen
        # profile is extended in place, mirroring the #26 continuation test)
        resumed.profile = _test_profile(max_epochs=4)
        result = resumed.fit(fresh_train, val_loader=fresh_val)
        assert result["stop_reason"] == "max_epochs_reached"
        assert fresh_curriculum.stage >= saved_stage

    def test_wiring_mismatches_fail_closed(self):
        run_dir = _mkdtmp()
        profile = _test_profile(max_epochs=1)
        curriculum, train_loader, val_loader = _curriculum_loaders()
        trainer = _trainer(run_dir, curriculum=curriculum, profile=profile)
        trainer.fit(train_loader, val_loader=val_loader)
        # a non-curriculum trainer refuses a curriculum checkpoint
        plain = _trainer(_mkdtmp(), curriculum=None, profile=profile)
        with pytest.raises(TrainerStateError, match="wiring mismatch"):
            plain.load_checkpoint(trainer.checkpoint_path)
        # a curriculum trainer refuses a plain checkpoint
        plain_run = _mkdtmp()
        plain_trainer = _trainer(plain_run, curriculum=None, profile=profile)
        plain_trainer.fit(train_loader, val_loader=val_loader)
        with pytest.raises(TrainerStateError, match="no curriculum state"):
            trainer.load_checkpoint(plain_trainer.checkpoint_path)

    def test_exact_epoch_reiteration_under_resume(self):
        """The sampler's deterministic streams make re-listing an epoch exact."""
        curriculum, _, _ = _curriculum_loaders()
        curriculum.set_epoch(2)
        _advance_to_stage(curriculum, 2)
        first = list(iter(curriculum))
        second = list(iter(curriculum))
        assert first == second  # re-listing the epoch reproduces the draws


# ---------------------------------------------------------------------------
# The canonical CLI end-to-end with --curriculum (scripts/train_stage1.py)
# ---------------------------------------------------------------------------


class TestCanonicalCLIWithCurriculum:
    def test_end_to_end_smoke(self, tmp_path):
        """The #27 production form of the canonical Stage-1 entry point.

        ``--curriculum`` wires the Approach C sampler into the canonical
        CLI: the train loader is sampler-driven, the validation loss
        drives stage advancement, and the checkpoint carries the
        curriculum state.
        """
        import subprocess
        import sys as _sys

        from rdkit import Chem

        from syntree.chemistry.synthon_encoder import (
            D_EMBED,
            encode_synthon,
            write_catalog_h5,
        )
        from syntree.data.trajectories import (
            CatalogView,
            extract_trajectory,
            trajectory_to_dict,
        )
        from tests.test_supervised_dataset import (
            AMIDE_CATALOG,
            LIGAND_SMILES,
            _build_index,
            _embed,
            _pocket,
        )

        tmp = str(tmp_path)
        traj_dir = os.path.join(tmp, "trajectories")
        processed = os.path.join(tmp, "processed")
        run_dir = os.path.join(tmp, "run")
        os.makedirs(traj_dir)
        os.makedirs(processed)

        records = {}
        for split in ("train", "val"):
            ligand = _embed(LIGAND_SMILES)
            record = extract_trajectory(
                ligand,
                complex_id="cx_amide",
                split=split,
                index=_build_index(AMIDE_CATALOG),
                catalog=CatalogView(
                    block_ids=tuple(r[0] for r in AMIDE_CATALOG),
                    smiles=tuple(r[1] for r in AMIDE_CATALOG),
                ),
            )
            records[split] = trajectory_to_dict(record)
        # three train records -> 12 supervised items -> all 9 levels populated
        torch.save(
            [records["train"]] * 3, os.path.join(traj_dir, "trajectories_train.pt")
        )
        torch.save([records["val"]], os.path.join(traj_dir, "trajectories_val.pt"))

        ligand = _embed(LIGAND_SMILES)
        with open(os.path.join(processed, "cx_amide_ligand.sdf"), "w") as handle:
            handle.write(Chem.MolToMolBlock(ligand))
        pocket = _pocket(ligand)
        lines = []
        for i, (pos, z) in enumerate(zip(pocket.positions, pocket.atomic_numbers)):
            element = {6: "C", 7: "N", 8: "O", 16: "S"}[int(z)]
            lines.append(
                f"ATOM  {i + 1:5d}  CA  ALA A{i % 9 + 1:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )
        with open(os.path.join(processed, "cx_amide_pocket.pdb"), "w") as handle:
            handle.write("\n".join(lines) + "\n")

        rows = [
            encode_synthon(block_id, smiles, stratum="fixture",
                           synthon_classes=syn, seed_classes=seed)
            for block_id, smiles, syn, seed in AMIDE_CATALOG
        ]
        x_raw = np.stack([r.x_raw for r in rows])
        mean = x_raw.mean(axis=0)
        std = np.maximum(x_raw.std(axis=0), 1e-12)
        rng = np.random.default_rng(2026)
        q, _ = np.linalg.qr(rng.standard_normal((x_raw.shape[1], D_EMBED)))
        catalog_path = os.path.join(tmp, "catalog_fixture.h5")
        write_catalog_h5(
            catalog_path, rows, mean=mean, std=std, q=q,
            source_sdf="fixture", source_sdf_sha256="0" * 64, registry_sha="0" * 64,
        )

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [
                _sys.executable,
                os.path.join(repo_root, "scripts", "train_stage1.py"),
                "--config", os.path.join(repo_root, "configs", "train_colab_t4.json"),
                "--trajectories", traj_dir,
                "--processed", processed,
                "--catalog", catalog_path,
                "--run-dir", run_dir,
                "--max-epochs", "2",
                "--micro-batch-size", "4",
                "--curriculum",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=repo_root,
        )
        assert result.returncode == 0, result.stderr[-3000:]
        assert "curriculum: 9 levels / 6 stages" in result.stdout
        assert os.path.exists(os.path.join(run_dir, "trainer_state.pt"))
        assert os.path.exists(os.path.join(run_dir, "trainer_state.pt.sha256"))
        telemetry = [
            json.loads(line)
            for line in open(os.path.join(run_dir, "telemetry.jsonl"))
        ]
        curriculum_records = [r for r in telemetry if r.get("phase") == "curriculum"]
        assert len(curriculum_records) == 2  # one per completed epoch
        for record in curriculum_records:
            assert len(record["level_counts"]) == 9
            assert sum(record["level_counts"]) == 12
        # the checkpoint carries the curriculum state (#45 s12.1/s13.6)
        payload = torch.load(
            os.path.join(run_dir, "trainer_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert payload["curriculum"]["schema_version"] == 1
        assert payload["curriculum"]["num_levels"] == 9
        assert payload["curriculum"]["num_stages"] == 6
