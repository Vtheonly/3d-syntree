"""Policy-gradient fine-tuning for 3D-SynTree.

Stage 2 reuses the chemistry-constrained generator. The environment is
non-differentiable, but the factorized policy

    pi(a_t|s_t) = pi(r_t|s_t) pi(b_t|r_t,s_t) pi(phi_t|b_t,r_t,s_t)

is differentiable with respect to its discrete action probabilities and the
terminal reward. PPO therefore optimizes the actual synthesis-constrained
policy rather than a post-hoc molecular score.

The implementation deliberately reports when docking is unavailable instead
of substituting a fabricated score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED

from syntree.engine.evaluator import EvaluationPipeline

logger = logging.getLogger(__name__)


@dataclass
class ThreeDRewardConfig:
    docking_weight: float = 1.0
    clash_weight: float = 0.25
    contact_weight: float = 0.25
    fsp3_weight: float = 0.15
    qed_weight: float = 0.15
    validity_weight: float = 0.5
    docking_scale: float = 10.0
    clash_scale: float = 25.0
    contact_scale: float = 5.0


class ThreeDReward:
    """Bounded terminal reward for a generated ligand."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.cfg = ThreeDRewardConfig(
            docking_weight=float(cfg.get("docking_weight", 1.0)),
            clash_weight=float(cfg.get("clash_weight", 0.25)),
            contact_weight=float(cfg.get("contact_weight", 0.25)),
            fsp3_weight=float(cfg.get("fsp3_weight", 0.15)),
            qed_weight=float(cfg.get("qed_weight", 0.15)),
            validity_weight=float(cfg.get("validity_weight", 0.5)),
            docking_scale=float(cfg.get("docking_scale", 10.0)),
            clash_scale=float(cfg.get("clash_scale", 25.0)),
            contact_scale=float(cfg.get("contact_scale", 5.0)),
        )

    def compute(
        self,
        mol: Chem.Mol,
        pocket_pdb_path: Optional[str] = None,
        clash_score: float = 0.0,
        contact_energy: float = 0.0,
    ) -> Dict[str, float]:
        components: Dict[str, float] = {
            "docking": 0.0,
            "clash": 0.0,
            "contact": 0.0,
            "fsp3": 0.0,
            "qed": 0.0,
            "validity": 0.0,
        }
        if mol is None:
            return {"reward": 0.0, **components, "docking_available": 0.0}

        try:
            heavy = Chem.RemoveHs(Chem.Mol(mol))
            Chem.SanitizeMol(Chem.Mol(heavy))
            components["validity"] = 1.0
        except Exception:
            return {"reward": 0.0, **components, "docking_available": 0.0}

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
            components["qed"] = float(QED.qed(heavy))
        except Exception:
            components["qed"] = 0.0

        clash_score = float(clash_score)
        components["clash"] = float(
            np.exp(-max(0.0, clash_score) / self.cfg.clash_scale)
        )

        # Lennard-Jones contact energy: negative when the ligand occupies the
        # pocket at van der Waals contact (favourable dispersion), positive
        # under steric overlap, and ~0 when the ligand drifts into solvent.
        # tanh maps it to a bounded reward that is maximal for a well-packed
        # pose and *negative* for a clashing one, so the agent can no longer
        # "win" by leaving the pocket.
        contact_energy = float(contact_energy)
        components["contact"] = float(
            np.tanh(-contact_energy / max(self.cfg.contact_scale, 1e-6))
        )

        docking_available = False
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
                engine = "gnina" if available.get("gnina") else (
                    "vina" if available.get("vina") else None
                )
                if engine:
                    docking_available = True
                    result = evaluator._dock([mol], pocket_pdb_path)
                    scores = [s for s in result.get("scores", []) if s is not None]
                    if scores:
                        components["docking"] = float(
                            np.tanh(-float(scores[0]) / self.cfg.docking_scale)
                        )
                    else:
                        docking_available = False
            except Exception:
                docking_available = False

        reward = (
            self.cfg.docking_weight * components["docking"]
            + self.cfg.clash_weight * components["clash"]
            + self.cfg.contact_weight * components["contact"]
            + self.cfg.fsp3_weight * components["fsp3"]
            + self.cfg.qed_weight * components["qed"]
            + self.cfg.validity_weight * components["validity"]
        )
        return {
            "reward": float(reward),
            **components,
            "docking_available": float(docking_available),
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


@dataclass
class RolloutTransition(PPOTransition):
    """A PPO transition plus its environment feedback.

    ``reward`` is the reward received *after* taking the action (dense
    physics shaping during growth; the terminal reward is added to the
    final transition of the episode).
    """

    reward: float = 0.0
    done: bool = False


class RolloutBuffer:
    """PPO rollout buffer across multiple episodes (bug report 2, Flaw 4).

    The old loop updated the policy on every single 2-3 transition episode,
    which destabilises PPO's importance-sampling ratio and causes
    catastrophic forgetting. This buffer accumulates ``rollout_episodes``
    episodes (32-64 in production) across *different* pockets; advantages
    are then estimated with Generalized Advantage Estimation and normalized
    across the entire buffer before any gradient step.
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.episodes: List[List[RolloutTransition]] = []

    # ------------------------------------------------------------------
    def add_episode(self, transitions: Sequence[RolloutTransition]) -> None:
        if transitions:
            self.episodes.append(list(transitions))

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def __len__(self) -> int:
        return sum(len(ep) for ep in self.episodes)

    # ------------------------------------------------------------------
    def compute_gae(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation over the whole buffer.

        For each episode with rewards ``r_1..r_T`` and values ``V_1..V_T``:

            delta_t = r_t + gamma * V_{t+1} - V_t      (V_{T+1} = 0)
            A_t     = delta_t + gamma * lambda * A_{t+1}
            return_t = A_t + V_t

        Returns:
            ``(advantages, returns)`` flat tensors of length
            ``len(buffer)`` in episode order.
        """
        advantages: List[float] = []
        returns: List[float] = []
        for episode in self.episodes:
            n = len(episode)
            values = [float(t.old_value.detach().float().item()) for t in episode]
            rewards = [float(t.reward) for t in episode]
            # Backward GAE recursion.
            adv_t = 0.0
            ep_adv = [0.0] * n
            ep_ret = [0.0] * n
            for t in range(n - 1, -1, -1):
                next_value = values[t + 1] if t + 1 < n else 0.0
                # Bootstrapping uses V(s_{t+1}); for terminal transitions
                # (done=True) the successor value is 0 by definition.
                if t + 1 < n and not episode[t].done:
                    next_value = values[t + 1]
                elif episode[t].done:
                    next_value = 0.0
                delta = rewards[t] + self.gamma * next_value - values[t]
                adv_t = delta + self.gamma * self.gae_lambda * adv_t
                ep_adv[t] = adv_t
            for t in range(n):
                ep_ret[t] = ep_adv[t] + values[t]
            advantages.extend(ep_adv)
            returns.extend(ep_ret)
        dev = device or torch.device("cpu")
        return (
            torch.tensor(advantages, dtype=torch.float32, device=dev),
            torch.tensor(returns, dtype=torch.float32, device=dev),
        )


class PPOFineTuner:
    """Clipped PPO optimizer over chemistry-constrained SBDD trajectories.

    Returns are terminal-reward returns: for a trajectory of length T, the
    reward at transition t is gamma**(T-1-t) * R. This avoids accidentally
    counting the same terminal reward once per preceding transition.
    """

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

        reaction_probs = out["reaction_log_probs"][0].exp()
        synthon_probs = out["synthon_log_probs"][0].exp()
        reaction_entropy = -(reaction_probs * out["reaction_log_probs"][0]).sum()
        synthon_entropy = -(synthon_probs * out["synthon_log_probs"][0]).sum()

        return (
            joint_log_prob,
            out["state_value"][0],
            reaction_entropy + synthon_entropy,
        )

    @staticmethod
    def terminal_returns(reward: float, horizon: int, gamma: float) -> torch.Tensor:
        """Return discounted terminal rewards for a trajectory."""
        if horizon <= 0:
            return torch.empty(0, dtype=torch.float32)
        return torch.tensor(
            [float(reward) * (float(gamma) ** (horizon - 1 - i))
             for i in range(horizon)],
            dtype=torch.float32,
        )

    def update_episode(
        self,
        trace: Sequence[Dict],
        reward: float,
    ) -> Dict[str, float]:
        if not trace:
            return {
                "loss": 0.0,
                "reward": float(reward),
                "steps": 0.0,
                "mean_advantage": 0.0,
            }

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

        horizon = len(transitions)
        returns_t = self.terminal_returns(
            reward, horizon, self.gamma
        ).to(self.device)
        old_values = torch.stack(
            [t.old_value.float() for t in transitions]
        ).to(self.device)
        advantages = returns_t - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        last_loss = 0.0
        last_entropy = 0.0
        for _ in range(self.ppo_epochs):
            policy_terms = []
            value_terms = []
            entropy_terms = []

            for idx, transition in enumerate(transitions):
                new_log_prob, new_value, entropy = self._log_prob_and_value(
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
                entropy_terms.append(entropy)

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
            last_entropy = float(entropy.detach().item())

        return {
            "loss": last_loss,
            "reward": float(reward),
            "steps": float(len(transitions)),
            "mean_advantage": float(advantages.mean().detach().item()),
            "policy_entropy": last_entropy,
        }

    # ------------------------------------------------------------------
    # Buffered PPO (bug report 2, Flaw 4)
    # ------------------------------------------------------------------
    def update_rollout(
        self,
        buffer: RolloutBuffer,
        minibatch_size: int = 32,
    ) -> Dict[str, float]:
        """PPO update over a multi-episode rollout buffer.

        Advantages are computed once with GAE over the whole buffer,
        normalized across ALL transitions (never per micro-batch), and the
        policy is then updated for ``ppo_epochs`` over shuffled minibatches
        of ``minibatch_size`` transitions - the standard, numerically stable
        PPO loop instead of the old 2-3-transition micro-batch descent.

        Returns summary statistics (losses, entropy, clip fractions).
        """
        if len(buffer) == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "mean_advantage": 0.0,
                "clip_fraction": 0.0,
                "transitions": 0,
                "episodes": 0,
            }

        advantages, returns = buffer.compute_gae(device=self.device)
        # Advantage normalization across the ENTIRE buffer.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        flat: List[RolloutTransition] = [
            t for episode in buffer.episodes for t in episode
        ]
        n = len(flat)
        minibatch_size = max(1, min(int(minibatch_size), n))

        last = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
        }
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n)
            epoch_losses, epoch_policy, epoch_value = [], [], []
            epoch_entropy, epoch_clip = [], []
            for start in range(0, n, minibatch_size):
                idx = perm[start : start + minibatch_size]
                policy_terms, value_terms, entropy_terms, clip_terms = (
                    [],
                    [],
                    [],
                    [],
                )
                for i in idx.tolist():
                    transition = flat[i]
                    new_log_prob, new_value, entropy = self._log_prob_and_value(
                        transition
                    )
                    old_log_prob = transition.old_log_prob.to(self.device)
                    ratio = torch.exp(new_log_prob - old_log_prob)
                    adv = advantages[i]
                    unclipped = ratio * adv
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    ) * adv
                    policy_terms.append(-torch.minimum(unclipped, clipped))
                    clip_terms.append(
                        (torch.abs(ratio - 1.0) > self.clip_epsilon).float()
                    )
                    value_terms.append(F.mse_loss(new_value, returns[i]))
                    entropy_terms.append(entropy)

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

                epoch_losses.append(float(loss.detach().item()))
                epoch_policy.append(float(policy_loss.detach().item()))
                epoch_value.append(float(value_loss.detach().item()))
                epoch_entropy.append(float(entropy.detach().item()))
                epoch_clip.append(
                    float(torch.stack(clip_terms).mean().item())
                )

            last = {
                "loss": sum(epoch_losses) / max(1, len(epoch_losses)),
                "policy_loss": sum(epoch_policy) / max(1, len(epoch_policy)),
                "value_loss": sum(epoch_value) / max(1, len(epoch_value)),
                "entropy": sum(epoch_entropy) / max(1, len(epoch_entropy)),
                "clip_fraction": sum(epoch_clip) / max(1, len(epoch_clip)),
            }

        return {
            **last,
            "mean_advantage": float(advantages.mean().item()),
            "transitions": n,
            "episodes": buffer.num_episodes,
        }


def collect_episode(
    env,
    model: torch.nn.Module,
    catalog,
    pocket_pdb_path: Optional[str] = None,
    terminal_reward_fn=None,
    sample: bool = True,
    temperature: float = 1.0,
    seed_synthon_idx: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Tuple[List[RolloutTransition], Dict]:
    """Roll out one episode of the assembly MDP under the current policy.

    Drives :class:`~syntree.engine.environment.MolecularAssemblyEnv` with
    ``model.act`` and records PPO transitions enriched with the per-step
    dense physics reward. The terminal reward (docking/QED/contact
    components) is computed by ``terminal_reward_fn(env) -> dict`` and added
    to the final transition, exactly as required by GAE bootstrapping.

    Returns:
        ``(transitions, reward_details)``.
    """
    from syntree.engine.environment import AssemblyAction

    observation = env.reset(
        pocket_pdb_path=pocket_pdb_path,
        seed_synthon_idx=seed_synthon_idx,
        max_steps=max_steps,
    )
    transitions: List[RolloutTransition] = []
    reward_details: Dict = {}
    while not env.done:
        reaction_mask, synthon_masks, has_handle = env.legal_action_masks()
        if not has_handle or not bool((reaction_mask > -1e8).any().item()):
            break
        decision = model.act(
            observation,
            catalog.embeddings,
            reaction_compatibility_mask=reaction_mask,
            synthon_masks_by_reaction=synthon_masks,
            sample=sample,
            temperature=temperature,
        )
        transition = RolloutTransition(
            state=observation.clone(),
            reaction_mask=reaction_mask.detach().clone(),
            synthon_masks=synthon_masks.detach().clone(),
            family_idx=int(decision["reaction_family_idx"][0].item()),
            action_idx=int(decision["action_idx"][0].item()),
            old_log_prob=decision["joint_log_prob"][0].detach(),
            old_value=decision["state_value"][0].detach(),
        )
        action = AssemblyAction(
            reaction_family_idx=int(decision["reaction_family_idx"][0].item()),
            synthon_idx=int(decision["synthon_idx"][0].item()),
            dihedral_rad=float(decision["dihedral"][0].item()),
            action_idx=int(decision["action_idx"][0].item()),
        )
        observation, step_reward, done, info = env.step(action)
        transition.reward = float(step_reward)
        transition.done = bool(done)
        transitions.append(transition)

    if terminal_reward_fn is not None:
        try:
            reward_details = terminal_reward_fn(env)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("terminal reward failed (%s); using zero.", exc)
            reward_details = {"reward": 0.0}
        terminal = float(reward_details.get("reward", 0.0))
        if transitions:
            transitions[-1].reward += terminal
    return transitions, reward_details


__all__ = [
    "ThreeDRewardConfig",
    "ThreeDReward",
    "PPOTransition",
    "RolloutTransition",
    "RolloutBuffer",
    "PPOFineTuner",
    "collect_episode",
]
