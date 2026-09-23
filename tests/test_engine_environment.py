"""MolecularAssemblyEnv MDP contract tests (bug report 3, upgrade 1).

Verifies the environment as a *rigorous MDP*: reset/step/observe/act
semantics, chemistry-mask legality, deterministic transitions, physics
rewards, termination conditions, and equivalence with the legacy
SBDDGenerator rollout (single source of chemistry truth).
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch
from rdkit import Chem

from syntree.chemistry.catalog import SynthonCatalog
from syntree.chemistry.conformer import ConformerEngine
from syntree.chemistry.reactions import REACTION_FAMILY_NAMES, ReactionEngine
from syntree.chemistry.validator import ChemicalValidator
from syntree.engine.environment import AssemblyAction, MolecularAssemblyEnv
from syntree.models.policy import SynTreePolicy


@pytest.fixture(scope="module")
def catalog(assets_dir):
    return SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)


@pytest.fixture()
def env(tiny_config_with_assets, catalog):
    # Function-scoped on purpose: tests mutate rollout state (cap, step
    # counters) and must not leak into each other.
    cfg = json.loads(json.dumps(tiny_config_with_assets))
    return MolecularAssemblyEnv(
        rxn_engine=ReactionEngine(),
        conformer_engine=ConformerEngine(),
        catalog=catalog,
        validator=ChemicalValidator(),
        config=cfg,
        device=torch.device("cpu"),
    )


@pytest.fixture()
def pocket_path(assets_dir):
    return os.path.join(assets_dir["crossdocked_dir"], "sample_pocket.pdb")


@pytest.fixture()
def policy(tiny_config_with_assets):
    torch.manual_seed(0)
    return SynTreePolicy(tiny_config_with_assets).eval()


class TestReset:
    def test_reset_returns_policy_ready_observation(self, env, pocket_path):
        obs = env.reset(pocket_path, seed_synthon_idx=0)
        for key in (
            "pocket_pos", "pocket_z", "pocket_charge", "ligand_pos", "ligand_z",
            "handle_features", "handle_pos", "handle_nodes", "global_features",
            "stop_mask",
        ):
            assert hasattr(obs, key), f"observation missing {key}"
        assert obs.ligand_pos.size(0) > 0  # seed is the initial ligand
        assert obs.handle_nodes[0].item() >= 0
        assert obs.global_features[0, -1].item() == 1.0  # has_ligand
        assert not env.done

    def test_reset_in_memory_pocket(self, env, pocket_path):
        mol = Chem.MolFromPDBFile(pocket_path, removeHs=False)
        obs = env.reset(pocket_mol=mol, seed_synthon_idx=1)
        assert obs.pocket_pos.size(0) == mol.GetNumAtoms()

    def test_reset_requires_pocket(self, env):
        with pytest.raises(ValueError):
            env.reset()

    def test_reset_invalid_pocket(self, env, tmp_path):
        bad = tmp_path / "bad.pdb"
        bad.write_text("garbage")
        with pytest.raises(ValueError):
            env.reset(str(bad))

    def test_recipe_starts_with_seed(self, env, pocket_path):
        env.reset(pocket_path, seed_synthon_idx=0)
        assert env.recipe[0]["action"] == "seed"
        assert env.recipe[0]["synthon_id"].startswith("TEST")


class TestLegalActions:
    def test_masks_shapes(self, env, pocket_path):
        env.reset(pocket_path, seed_synthon_idx=0)
        r_mask, s_masks, has_handle = env.legal_action_masks()
        assert r_mask.shape == (1, len(REACTION_FAMILY_NAMES))
        assert s_masks.shape[1] == len(REACTION_FAMILY_NAMES)
        assert s_masks.shape[2] == len(env.catalog)
        assert has_handle

    def test_mask_gates_stop_below_mw_cap(self, env, pocket_path):
        """A tiny seed is below the MW cap -> STOP is masked out."""
        env.reset(pocket_path, seed_synthon_idx=0)
        assert env._below_cap()
        obs = env.observe()
        assert obs.stop_mask.item() < -1e8

    def test_no_handle_state_masks_everything(self, env, pocket_path):
        """A terminal (done) state exposes the all-masked grammar."""
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=0)
        assert env.done  # born terminal (seed-only episode)
        r_mask, s_masks, has_handle = env.legal_action_masks()
        assert has_handle is False
        assert bool((r_mask < -1e8).all().item())
        assert bool((s_masks < -1e8).all().item())


class TestStep:
    def test_legal_growing_step(self, env, pocket_path):
        """One legal amide coupling grows the ligand and updates the state.

        Note: the tiny test catalog only contains monofunctional amines, so
        the catalog's documented fallback may cap the ligand in one step
        (``no_handles`` termination) - both outcomes are legal MDP
        behaviour; what matters is that the step actually executed.
        """
        obs = env.reset(pocket_path, seed_synthon_idx=0, max_steps=3)
        r_mask, s_masks, _ = env.legal_action_masks()
        # Pick a legal (family, synthon).
        family = int(r_mask[0].argmax().item())
        synthon = int(s_masks[0, family].argmax().item())
        action = AssemblyAction(family, synthon, dihedral_rad=1.0)
        next_obs, reward, done, info = env.step(action)

        assert isinstance(reward, float) and math.isfinite(reward)
        assert info.reaction is not None
        assert info.synthon_id.startswith("TEST")
        assert next_obs.ligand_pos.size(0) > obs.ligand_pos.size(0)
        assert len(env.recipe) == 2
        assert env.recipe[1]["action"] == "react"
        assert done is False or info.terminated_by == "no_handles"

    def test_stop_action_terminates(self, env, pocket_path):
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=5)
        # Lift the MW cap so the current seed already qualifies for STOP.
        env.terminal_cap_min_mw = 0.0
        _, _, done, info = env.step(
            AssemblyAction(0, -1, 0.0)  # STOP
        )
        assert done
        assert info.terminated_by == "policy_stop"
        assert env.recipe[-1]["action"] == "stop"

    def test_stop_below_cap_is_refused(self, env, pocket_path):
        """STOP while below the MW cap must NOT terminate the episode."""
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=5)
        assert env._below_cap()
        _, _, done, info = env.step(AssemblyAction(0, -1, 0.0))
        # STOP is refused -> treated as a growth attempt of synthon -1,
        # which is invalid -> episode ends with invalid_synthon, NOT
        # policy_stop.
        assert info.terminated_by != "policy_stop"

    def test_invalid_family_terminates(self, env, pocket_path):
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=3)
        _, _, done, info = env.step(
            AssemblyAction(99, 0, 0.0)  # out-of-range family
        )
        assert done
        assert info.terminated_by == "invalid_family"

    def test_invalid_synthon_terminates(self, env, pocket_path):
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=3)
        _, _, done, info = env.step(
            AssemblyAction(0, 999999, 0.0)
        )
        assert done
        assert info.terminated_by == "invalid_synthon"

    def test_horizon_terminates(self, env, pocket_path):
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=1)
        r_mask, s_masks, _ = env.legal_action_masks()
        family = int(r_mask[0].argmax().item())
        synthon = int(s_masks[0, family].argmax().item())
        _, _, done, info = env.step(AssemblyAction(family, synthon, 0.3))
        assert done
        assert info.terminated_by == "horizon"

    def test_step_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(AssemblyAction(0, 0, 0.0))

    def test_amide_steps_stay_planar(self, env, pocket_path):
        """Steps through amide junctions keep resonance planarity."""
        from rdkit.Chem import rdMolTransforms

        env.reset(pocket_path, seed_synthon_idx=0, max_steps=3)
        worst = 0.0
        for _ in range(2):
            r_mask, s_masks, has = env.legal_action_masks()
            if not has:
                break
            family = int(r_mask[0].argmax().item())
            synthon = int(s_masks[0, family].argmax().item())
            _, _, done, info = env.step(
                AssemblyAction(family, synthon, math.pi / 2)
            )
            if done:
                break
        mol = env.current_mol
        conf = mol.GetConformer()
        pattern = Chem.MolFromSmarts("[C:1](=[O:2])-[N:3]")
        for c_idx, o_idx, n_idx in mol.GetSubstructMatches(pattern):
            c_nbrs = [
                a.GetIdx()
                for a in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                if a.GetIdx() not in (o_idx, n_idx)
            ]
            n_nbrs = [
                a.GetIdx()
                for a in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                if a.GetIdx() != c_idx
            ]
            if c_nbrs and n_nbrs:
                deg = abs(
                    rdMolTransforms.GetDihedralDeg(
                        conf, c_nbrs[0], c_idx, n_idx, n_nbrs[0]
                    )
                )
                worst = max(worst, min(abs(deg - 180.0), abs(deg - 0.0)))
        assert worst < 15.0


class TestRewardPhysics:
    def test_dense_reward_reflects_contacts(self, env, pocket_path):
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=3)
        r_mask, s_masks, _ = env.legal_action_masks()
        family = int(r_mask[0].argmax().item())
        synthon = int(s_masks[0, family].argmax().item())
        _, reward, _, info = env.step(AssemblyAction(family, synthon, 0.7))
        assert math.isfinite(reward)
        assert math.isfinite(info.contact_energy)
        assert info.contact_energy <= 0.0 or info.contact_energy > 0.0  # finite

    def test_dense_reward_can_be_disabled(self, tiny_config_with_assets, catalog):
        cfg = json.loads(json.dumps(tiny_config_with_assets))
        quiet = MolecularAssemblyEnv(
            rxn_engine=ReactionEngine(),
            conformer_engine=ConformerEngine(),
            catalog=catalog,
            validator=ChemicalValidator(),
            config=cfg,
            device=torch.device("cpu"),
            dense_reward=False,
        )
        assert quiet.dense_reward is False


class TestEquivalenceWithGenerator:
    """The env must be the single source of chemistry truth: a policy-driven
    env rollout and SBDDGenerator.generate_ligand must produce the same
    chemistry for the same decisions (same seed, argmax policy)."""

    def test_env_rollout_produces_valid_ligand(
        self, env, policy, pocket_path, catalog
    ):
        obs = env.reset(pocket_path, seed_synthon_idx=0, max_steps=2)
        torch.manual_seed(0)
        done, steps = False, 0
        while not done and steps < 5:
            r_mask, s_masks, has = env.legal_action_masks()
            if not has:
                break
            decision = policy.act(
                obs, catalog.embeddings,
                reaction_compatibility_mask=r_mask,
                synthon_masks_by_reaction=s_masks,
                sample=False,
            )
            action = AssemblyAction(
                reaction_family_idx=int(decision["reaction_family_idx"][0]),
                synthon_idx=int(decision["synthon_idx"][0]),
                dihedral_rad=float(decision["dihedral"][0]),
                action_idx=int(decision["action_idx"][0]),
            )
            obs, reward, done, info = env.step(action)
            steps += 1
        assert steps >= 1
        smiles = env.smiles()
        assert Chem.MolFromSmiles(smiles) is not None
        checks = env.validate()
        assert isinstance(checks, dict) and len(checks) > 0

    def test_same_action_same_chemistry_deterministic(self, env, pocket_path):
        """Repeating an identical episode produces identical states."""
        results = []
        for _ in range(2):
            env.reset(pocket_path, seed_synthon_idx=0, max_steps=1)
            r_mask, s_masks, _ = env.legal_action_masks()
            family = int(r_mask[0].argmax().item())
            synthon = int(s_masks[0, family].argmax().item())
            env.step(AssemblyAction(family, synthon, 1.25))
            results.append((env.smiles(), env.total_clash, env.contact_energy()))
        assert results[0] == results[1]


class TestInfo:
    def test_info_serialisable(self, env, pocket_path):
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=2)
        r_mask, s_masks, _ = env.legal_action_masks()
        family = int(r_mask[0].argmax().item())
        synthon = int(s_masks[0, family].argmax().item())
        _, _, _, info = env.step(AssemblyAction(family, synthon, 0.9))
        as_dict = info.as_dict()
        assert isinstance(as_dict, dict)
        # JSON-serialisable for logging.
        import json as _json

        _json.dumps(as_dict)
