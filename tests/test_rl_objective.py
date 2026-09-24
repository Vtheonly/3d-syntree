"""Tests for the research-stage PPO objective."""
from __future__ import annotations

import torch

from syntree.engine.rl import PPOFineTuner


def test_terminal_reward_returns_decay_once():
    returns = PPOFineTuner.terminal_returns(1.0, horizon=3, gamma=0.5)
    assert torch.allclose(
        returns,
        torch.tensor([0.25, 0.5, 1.0]),
    )


def test_empty_terminal_returns():
    returns = PPOFineTuner.terminal_returns(1.0, horizon=0, gamma=0.99)
    assert returns.numel() == 0


class TestBatchDiversityReward:
    """tasklist Priority 6: batch Tanimoto diversity term."""

    def _mol(self, smiles):
        from rdkit import Chem

        return Chem.MolFromSmiles(smiles)

    def test_identical_molecules_have_zero_diversity(self):
        from syntree.engine.rl import ThreeDReward

        mols = [self._mol("CCO"), self._mol("CCO"), self._mol("CCO")]
        divs = ThreeDReward.batch_diversity(mols)
        assert all(d == 0.0 for d in divs)

    def test_disjoint_molecules_have_full_diversity(self):
        from syntree.engine.rl import ThreeDReward

        # Ethanol vs bromobenzene share no Morgan bits at radius 2.
        mols = [self._mol("CCO"), self._mol("Brc1ccccc1")]
        divs = ThreeDReward.batch_diversity(mols)
        assert all(abs(d - 1.0) < 1e-9 for d in divs)

    def test_diversity_between_zero_and_one(self):
        from syntree.engine.rl import ThreeDReward

        mols = [self._mol("CCO"), self._mol("CCCO"), self._mol("CCCCO")]
        for d in ThreeDReward.batch_diversity(mols):
            assert 0.0 <= d <= 1.0

    def test_symmetry_of_pairwise_diversity(self):
        from syntree.engine.rl import ThreeDReward

        mols = [self._mol("CCO"), self._mol("CCCO"), self._mol("c1ccccc1")]
        d = ThreeDReward.batch_diversity(mols)
        # mean_{j != i}(1 - T_ij) is symmetric: d[i] vs d[j] differ only by
        # the third molecule's contribution; for 3 mols the pairwise parts
        # cancel exactly, so all three values are equal.
        assert abs(d[0] - d[1]) < 1e-6

    def test_none_entries_score_zero(self):
        from syntree.engine.rl import ThreeDReward

        divs = ThreeDReward.batch_diversity([None, self._mol("CCO")])
        assert divs[0] == 0.0

    def test_singleton_batch_convention(self):
        from syntree.engine.rl import ThreeDReward

        assert ThreeDReward.batch_diversity([self._mol("CCO")]) == [1.0]

    def test_compute_uses_batch_reference_smiles(self):
        from syntree.engine.rl import ThreeDReward

        reward = ThreeDReward({"diversity_weight": 0.5, "docking_weight": 0.0,
                               "validity_weight": 0.0, "clash_weight": 0.0,
                               "contact_weight": 0.0, "fsp3_weight": 0.0,
                               "qed_weight": 0.0})
        mol = self._mol("CCO")
        result = reward.compute(mol, batch_reference_smiles=["CCO"])
        assert result["diversity"] == 0.0

        result2 = reward.compute(mol, batch_reference_smiles=["Brc1ccccc1"])
        assert abs(result2["diversity"] - 1.0) < 1e-9
        # reward includes 0.5 * diversity
        assert abs(result2["reward"] - 0.5 * result2["diversity"]) < 1e-9

    def test_diversity_default_weight_is_zero(self):
        """Legacy configs (no diversity key) keep the exact legacy reward."""
        from syntree.engine.rl import ThreeDReward

        reward = ThreeDReward({})
        assert reward.cfg.diversity_weight == 0.0
        assert reward.cfg.key_interaction_weight == 0.0
        mol = self._mol("CCO")
        result = reward.compute(mol, batch_reference_smiles=["CCCO"])
        assert result["diversity"] == 0.0

    def test_apply_terminal_rewards_adds_diversity_to_last_transition(self):
        import torch

        from syntree.engine.rl import RolloutTransition, ThreeDReward, apply_terminal_rewards

        reward = ThreeDReward({"diversity_weight": 0.3})
        t1 = RolloutTransition(
            state=None, reaction_mask=None, synthon_masks=None,
            family_idx=0, action_idx=0,
            old_log_prob=torch.tensor(0.0), old_value=torch.tensor(0.0),
            reward=0.0, done=True,
        )
        t2 = RolloutTransition(
            state=None, reaction_mask=None, synthon_masks=None,
            family_idx=0, action_idx=0,
            old_log_prob=torch.tensor(0.0), old_value=torch.tensor(0.0),
            reward=0.0, done=True,
        )
        base = {"reward": 1.0, "validity": 1.0}
        batch = [([t1], self._mol("CCO"), dict(base)),
                 ([t2], self._mol("Brc1ccccc1"), dict(base))]
        finalized = apply_terminal_rewards(batch, reward)
        assert finalized[0]["diversity"] == 1.0
        assert finalized[1]["diversity"] == 1.0
        # identical-to-others check: different molecules -> full diversity
        assert t1.reward == 1.0 + 0.3 * 1.0
        assert t2.reward == 1.0 + 0.3 * 1.0

    def test_apply_terminal_rewards_identical_batch(self):
        import torch

        from syntree.engine.rl import RolloutTransition, ThreeDReward, apply_terminal_rewards

        reward = ThreeDReward({"diversity_weight": 0.25})
        t = RolloutTransition(
            state=None, reaction_mask=None, synthon_masks=None,
            family_idx=0, action_idx=0,
            old_log_prob=torch.tensor(0.0), old_value=torch.tensor(0.0),
            reward=0.0, done=True,
        )
        batch = [([t], self._mol("CCO"), {"reward": 0.5})]
        finalized = apply_terminal_rewards(batch, reward)
        # single valid molecule in batch -> convention diversity 1.0
        assert finalized[0]["diversity"] == 1.0
        assert t.reward == 0.5 + 0.25


class TestKeyInteractionReward:
    """tasklist Priority 6: pharmacophore key-interaction bonus."""

    def _ligand_with_salt_bridge(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("C[N+](C)(C)CCO")  # choline-like cation
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=3)
        return mol

    def _pocket_with_acetate(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CC(=O)[O-]")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=3)
        return mol

    def test_count_key_interactions_detects_salt_bridge(self):
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import rdMolTransforms as rdmt

        from syntree.engine.rl import count_key_interactions

        lig = self._ligand_with_salt_bridge()
        pocket = self._pocket_with_acetate()
        # Align the acetate right next to the charged nitrogen.
        lig_conf, pk_conf = lig.GetConformer(), pocket.GetConformer()
        lig_n = next(a.GetIdx() for a in lig.GetAtoms()
                     if a.GetFormalCharge() > 0)
        pk_o = next(a.GetIdx() for a in pocket.GetAtoms()
                    if a.GetFormalCharge() < 0)
        lig_pos = np.array(lig_conf.GetAtomPosition(lig_n))
        pk_pos = np.array(pk_conf.GetAtomPosition(pk_o))
        translation = lig_pos + np.array([3.5, 0.0, 0.0]) - pk_pos
        for idx in range(pocket.GetNumAtoms()):
            pos = np.array(pk_conf.GetAtomPosition(idx)) + translation
            pk_conf.SetAtomPosition(idx, pos.tolist())

        n_salt, n_hbond = count_key_interactions(lig, pocket)
        assert n_salt >= 1

    def test_count_key_interactions_far_apart_gives_zero(self):
        import numpy as np
        from rdkit import Chem

        from syntree.engine.rl import count_key_interactions

        lig = self._ligand_with_salt_bridge()
        pocket = self._pocket_with_acetate()
        conf = pocket.GetConformer()
        for idx in range(pocket.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(idx)) + np.array([100.0, 0, 0])
            conf.SetAtomPosition(idx, pos.tolist())
        n_salt, n_hbond = count_key_interactions(lig, pocket)
        assert n_salt == 0
        assert n_hbond == 0

    def test_missing_inputs_give_zero(self):
        from syntree.engine.rl import count_key_interactions

        assert count_key_interactions(None, None) == (0, 0)

    def test_key_interaction_reward_math(self):
        from syntree.engine.rl import ThreeDReward, count_key_interactions

        reward = ThreeDReward({
            "key_interaction_weight": 1.5,
            "docking_weight": 0.0, "clash_weight": 0.0, "contact_weight": 0.0,
            "fsp3_weight": 0.0, "qed_weight": 0.0, "validity_weight": 0.0,
        })
        lig = self._ligand_with_salt_bridge()
        pocket = self._pocket_with_acetate()
        n_salt, n_hbond = count_key_interactions(lig, pocket)
        result = reward.compute(lig, pocket_mol=pocket)
        expected = min(1.0, n_salt + 0.5 * min(n_hbond, 2))
        assert abs(result["key_interaction"] - expected) < 1e-9
        assert abs(result["reward"] - 1.5 * expected) < 1e-6

    def test_key_interaction_weight_default_off(self):
        from syntree.engine.rl import ThreeDReward

        reward = ThreeDReward({})
        result = reward.compute(self._ligand_with_salt_bridge(),
                                pocket_mol=self._pocket_with_acetate())
        assert result["key_interaction"] == 0.0
        assert "n_salt_bridges" not in result
