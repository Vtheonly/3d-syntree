"""Policy-gradient fine-tuning for 3D-SynTree.

Stage 2 reuses the exact reaction-constrained generator. Chemistry execution is
non-differentiable, while the policy probabilities over reaction/synthon/STOP
actions remain differentiable, which makes PPO suitable for optimizing
black-box 3D rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED

from syntree.engine.evaluator import EvaluationPipeline


@dataclass
class ThreeDRewardConfig:
    docking_weight: float = 1.0
    clash_weight: float = 0.25
    fsp3_weight: float = 0.15
    qed_weight: float = 0.15
    validity_weight: float = 0.5
    docking_scale: float = 10.0
    clash_scale: float = 25.0


class ThreeDReward:
    """Bounded multi-objective reward for generated ligands."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.cfg = ThreeDRewardConfig(
            docking_weight=float(cfg.get("docking_weight", 1.0)),
            clash_weight=float(cfg.get("clash_weight", 0.25)),
            fsp3_weight=float(cfg.get("fsp3_weight", 0.15)),
            qed_weight=float(cfg.get("qed_weight", 0.15)),
            validity_weight=float(cfg.get("validity_weight", 0.5)),
            docking_scale=float(cfg.get("docking_scale", 10.0)),
            clash_scale=float(cfg.get("clash_scale", 25.0)),
        )

    def compute(
        self,
        mol: Chem.Mol,
        pocket_pdb_path: Optional[str] = None,
        clash_score: float = 0.0,
    ) -> Dict[str, float]:
        components: Dict[str, float] = {
            "docking": 0.0,
            "clash": 0.0,
            "fsp3": 0.0,
            "qed": 0.0,
            "validity": 0.0,
        }
        if mol is None:
            return {"reward": 0.0, **components}

        smiles = None
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
            Chem.SanitizeMol(Chem.Mol(mol))
            components["validity"] = 1.0
        except Exception:
            return {"reward": 0.0, **components}

        descriptors = {}
        try:
            from syntree.chemistry.validator import ChemicalValidator

            validator = ChemicalValidator()
            descriptors = validator.descriptors(mol)
            checks = validator.validate(mol)
            components["validity"] = 1.0 if all(checks.values()) else 0.0
        except Exception:
            descriptors = {}

        fsp3 = float(descriptors.get("fsp3", 0.0))
        components["fsp3"] = float(np.tanh(max(0.0, fsp3 - 0.42) / 0.20))
        try:
            components["qed"] = float(QED.qed(Chem.RemoveHs(Chem.Mol(mol))))
        except Exception:
            components["qed"] = 0.0

        clash_score = float(clash_score)
        components["clash"] = float(np.exp(-max(0.0, clash_score) / self.cfg.clash_scale))

        docking_engine = None
        if pocket_pdb_path:
            try:
                evaluator = EvaluationPipeline(
                    {
                        "evaluation": {
                            "docking_engine": "gnina",
                            "run_aizynthfinder": False,
                        }
                    },
                    output_dir="./experiments/rl_docking",
                )
                available = evaluator._tool_availability()
                docking_engine = "gnina" if available.get("gnina") else (
                    "vina" if available.get("vina") else None
                )
                if docking_engine:
                    result = evaluator._dock([mol], pocket_pdb_path)
                    scores = [s for s in result.get("scores", []) if s is not None]
                    if scores:
                        # More negative docking energies become larger bounded rewards.
                        components["docking"] = float(
                            np.tanh(-float(scores[0]) / self.cfg.docking_scale)
                        )
            except Exception:
                docking_engine = None

        reward = (
            self.cfg.docking_weight * components["docking"]
            + self.cfg.clash_weight * components["clash"]
            + self.cfg.fsp3_weight * components["fsp3"]
            + self.cfg.qed_weight * components["qed"]
            + self.cfg.validity_weight * components["validity"]
        )
        return {
            "reward": float(reward),
            **components,
            "docking_available": float(docking_engine is not None),
        }


@dataclass
class PPOTransition:
    """Single policy transition captured by SBDDGenerator."""

    state: object
    reaction_mask: torch.Tensor
    synthon_masks: torch.Tensor
    family_idx: int
    action_idx: int
    old_log_prob: torch.Tensor
    old_value: torch.Tensor


class PPOFineTuner:
    """Clipped PPO optimizer over chemistry-constrained SBDD trajectories."""

    def __init__(
        self,
        model: torch.nn.Module,
        catalog,
        device: torch.device,
        config: Optional[Dict] = None,
    ):
        cfg = config or {}
        self.model = model.to(device)
        self.catalog = catalog
        self.device = device
        self.clip_epsilon = float(cfg.get("clip_epsilon", 0.2))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.ppo_epochs = int(cfg.get("ppo_epochs", 4))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.get("learning_rate", 1e-5)),
            weight_decay=float(cfg.get("weight_decay", 1e-5)),
        )

    def _log_prob_and_value(self, transition: PPOTransition):
        batch = transition.state.to(self.device)
        reaction_mask = transition.reaction_mask.to(self.device)
        synthon_masks = transition.synthon_masks.to(self.device)
        family = int(transition.family_idx)
        selected_mask = synthon_masks[:, family, :]

        out = self.model(
            batch,
            self.catalog.embeddings.to(self.device),
            selected_mask,
            reaction_mask,
        )
        action_log_prob = out["synthon_log_probs"][0, transition.action_idx]
        if transition.action_idx == len(self.catalog):
            joint_log_prob = action_log_prob
        else:
            reaction_log_prob = out["reaction_log_probs"][0, family]
            joint_log_prob = reaction_log_prob + action_log_prob

        return joint_log_prob, out["state_value"][0], out["synthon_log_probs"][0]

    def update_episode(
        self,
        trace: Sequence[Dict],
        reward: float,
    ) -> Dict[str, float]:
        if not trace:
            return {"loss": 0.0, "reward": float(reward), "steps": 0.0}

        transitions = [
            PPOTransition(
                state=item["state"].cpu(),
                reaction_mask=item["reaction_mask"].cpu(),
                synthon_masks=item["synthon_masks"].cpu(),
                family_idx=int(item["family_idx"]),
                action_idx=int(item["action_idx"]),
                old_log_prob=item["old_log_prob"].detach().cpu(),
                old_value=item["old_value"].detach().cpu(),
            )
            for item in trace
        ]

        returns = []
        discounted = 0.0
        for _ in range(len(transitions)):
            discounted = float(reward) + self.gamma * discounted
            returns.append(discounted)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        old_values = torch.stack(
            [t.old_value.float() for t in transitions]
        ).to(self.device)
        advantages = returns_t - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        last_loss = 0.0
        for _ in range(self.ppo_epochs):
            policy_terms = []
            value_terms = []
            entropy_terms = []

            for idx, transition in enumerate(transitions):
                new_log_prob, new_value, synthon_log_probs = self._log_prob_and_value(
                    transition
                )
                old_log_prob = transition.old_log_prob.to(self.device)
                ratio = torch.exp(new_log_prob - old_log_prob)
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * adv
                policy_terms.append(-torch.minimum(unclipped, clipped))
                value_terms.append(F.mse_loss(new_value, returns_t[idx]))

                probs = synthon_log_probs.exp()
                entropy_terms.append(-(probs * synthon_log_probs).sum())

            policy_loss = torch.stack(policy_terms).mean()
            value_loss = torch.stack(value_terms).mean()
            entropy = torch.stack(entropy_terms).mean()
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            last_loss = float(loss.detach().item())

        return {
            "loss": last_loss,
            "reward": float(reward),
            "steps": float(len(transitions)),
            "mean_advantage": float(advantages.mean().detach().item()),
        }


__all__ = ["ThreeDRewardConfig", "ThreeDReward", "PPOTransition", "PPOFineTuner"]
