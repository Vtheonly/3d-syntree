"""#30 FLOW-02 — the docking-backend injection point (production supersession).

The stationary reward core of #30 was implemented and verified at S-V2-03
(D-V2-009): ``R(x) = exp((S_dock + S_contact + S_key + S_props + S_vol)/T)``
with the hard floor, no batch-relative term anywhere in the signature, and
fully labelled component telemetry.  With #29 FLOW-01 landed, this battery
pins the remaining implementable scope: the **docking-backend injection
point** that makes "a real docking engine replaces the surrogate without
touching the MDP/TB contracts" structural rather than aspirational —

* the default backend is the documented surrogate and the reward values
  are bit-identical to the pre-injection computation;
* an injected backend (the evaluation stack's real docking engine) is
  actually used, with the #30 clamp/units/sign conventions applied to its
  ``ΔG`` exactly as specified;
* the backend is part of the frozen configuration (stationarity and
  versioning: swapping a backend is a new, explicitly-versioned instance);
* non-finite backend scores fail closed (no NaN can reach ``log R``);
* reward reproducibility across batch compositions holds under an
  injected backend exactly as under the surrogate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from syntree.engine.environment import PocketDefinition
from syntree.engine.rl import (
    DockingResult,
    RewardError,
    ThreeDReward,
    surrogate_docking_scorer,
)
from tests.test_gflownet_reconciliation import _shell_pocket


def _target_molecule():
    """A small relaxed-ish terminal molecule with a conformer."""
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
    assert AllChem.EmbedMolecule(mol, randomSeed=2026) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


class TestDefaultBackendUnchanged:
    def test_default_is_the_documented_surrogate(self):
        reward = ThreeDReward(_shell_pocket())
        assert reward.docking_scorer is surrogate_docking_scorer

    def test_reward_bit_identical_to_direct_surrogate_computation(self):
        mol = _target_molecule()
        pocket = _shell_pocket()
        engine = ThreeDReward(pocket)
        value, components = engine.compute_with_components(mol)
        # recompute S_dock/S_contact directly from the surrogate physics
        positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
        numbers = np.asarray(
            [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64
        )
        direct = surrogate_docking_scorer(
            positions, numbers, engine._gasteiger_charges(mol), pocket
        )
        assert components["docking_backend"] == "surrogate_soft_core"
        assert components["delta_g_kcal_mol"] == pytest.approx(
            direct.delta_g_kcal_mol, rel=1e-12
        )
        assert components["S_dock"] == pytest.approx(
            min(max(-direct.delta_g_kcal_mol / 2.0, -3.0), 5.0), rel=1e-12
        )
        assert components["S_contact"] == pytest.approx(
            math.tanh(-direct.e_lj_attractive_kcal_mol / 5.0), rel=1e-12
        )
        assert value >= engine.reward_floor


class TestInjectedBackend:
    def _stub_backend(self, delta_g, e_attractive, calls):
        def scorer(positions, numbers, charges, pocket):
            calls.append(
                {
                    "positions_shape": tuple(np.asarray(positions).shape),
                    "pocket_atoms": pocket.num_atoms,
                }
            )
            return DockingResult(
                delta_g_kcal_mol=delta_g,
                e_lj_attractive_kcal_mol=e_attractive,
                backend="stub_engine_v9",
            )

        return scorer

    def test_injected_backend_is_used_with_the_pinned_conventions(self):
        calls = []
        engine = ThreeDReward(
            _shell_pocket(),
            docking_scorer=self._stub_backend(-8.0, -4.0, calls),
        )
        value, components = engine.compute_with_components(_target_molecule())
        assert len(calls) == 1
        assert calls[0]["pocket_atoms"] == engine.pocket.num_atoms
        # S_dock = clamp(-ΔG/2, -3, +5) with ΔG = -8 -> +4 (inside the clamp)
        assert components["S_dock"] == pytest.approx(4.0)
        # S_contact = tanh(-E_att/5) with E_att = -4
        assert components["S_contact"] == pytest.approx(math.tanh(4.0 / 5.0))
        assert components["docking_backend"] == "stub_engine_v9"
        # the other components are backend-independent
        assert 0.0 <= components["S_props"] <= 1.0
        assert value >= engine.reward_floor

    def test_clamp_bounds_hold_for_extreme_backend_scores(self):
        engine = ThreeDReward(
            _shell_pocket(),
            docking_scorer=self._stub_backend(-1e6, -1e6, []),
        )
        _, components = engine.compute_with_components(_target_molecule())
        assert components["S_dock"] == 5.0  # clamped above
        # tanh is bounded; at machine precision a huge argument saturates
        # to exactly 1.0 — the honest bound is the closed interval
        assert -1.0 < components["S_contact"] <= 1.0
        engine_pos = ThreeDReward(
            _shell_pocket(),
            docking_scorer=self._stub_backend(1e6, 0.0, []),
        )
        _, components_pos = engine_pos.compute_with_components(_target_molecule())
        assert components_pos["S_dock"] == -3.0  # clamped below
        assert components_pos["S_contact"] == 0.0

    def test_reward_stationary_across_batch_compositions_under_any_backend(self):
        """The #30 acceptance: identical molecule + pocket + frozen backend
        config -> identical reward, regardless of rollout-batch siblings."""
        mol = _target_molecule()
        engine = ThreeDReward(
            _shell_pocket(),
            docking_scorer=self._stub_backend(-6.0, -2.0, []),
        )
        alone = engine.compute(mol)
        # "siblings" cannot even be expressed: the signature has no batch
        # parameter; repeated calls amid other molecules stay identical
        other = Chem.MolFromSmiles("c1ccccc1")
        AllChem.EmbedMolecule(other, randomSeed=1)
        for _ in range(3):
            engine.compute(other)
        assert engine.compute(mol) == alone  # bit-identical

    def test_non_finite_backend_score_fails_closed(self):
        for bad_dg, bad_att in ((float("nan"), -1.0), (-1.0, float("inf"))):
            engine = ThreeDReward(
                _shell_pocket(),
                docking_scorer=self._stub_backend(bad_dg, bad_att, []),
            )
            with pytest.raises(RewardError, match="non-finite"):
                engine.compute_with_components(_target_molecule())

    def test_backend_must_return_the_contract_type(self):
        def bad_scorer(positions, numbers, charges, pocket):
            return (-8.0, -4.0)  # not a DockingResult

        engine = ThreeDReward(
            _shell_pocket(), docking_scorer=bad_scorer
        )
        with pytest.raises(RewardError, match="DockingResult"):
            engine.compute(_target_molecule())

    def test_non_callable_scorer_fails_closed_at_construction(self):
        with pytest.raises(RewardError, match="callable"):
            ThreeDReward(_shell_pocket(), docking_scorer="vinasaurus")

    def test_backend_is_frozen_configuration_provenance(self):
        engine = ThreeDReward(
            _shell_pocket(),
            docking_scorer=self._stub_backend(-5.0, -1.0, []),
        )
        _, components = engine.compute_with_components(_target_molecule())
        # the backend identity is recorded in every reward's telemetry —
        # the versioning the #30 addendum requires for objective changes
        assert components["docking_backend"] == "stub_engine_v9"
        # and the inactive (pocket-free) configuration reports it honestly
        graph_only = ThreeDReward()
        _, graph_components = graph_only.compute_with_components(
            _target_molecule()
        )
        assert graph_components["docking_backend"] == "inactive_no_pocket"
        assert graph_components["S_dock"] == 0.0


class TestRewardFloorsUnchanged:
    """The #30 floors/positivity through the injection surface."""

    def test_floor_enforced_with_adversarial_backend(self):
        def adversarial(positions, numbers, charges, pocket):
            return DockingResult(
                delta_g_kcal_mol=1e6,
                e_lj_attractive_kcal_mol=0.0,
                backend="adversarial_v1",
            )

        engine = ThreeDReward(
            _shell_pocket(),
            docking_scorer=adversarial,
            temperature=1.0,
        )
        value = engine.compute(_target_molecule())
        assert value >= engine.reward_floor
        assert math.isfinite(math.log(value))
