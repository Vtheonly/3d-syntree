"""Trajectory-Balance GFlowNet Stage-2 engine (tasklist RL modernisation).

Standard PPO maximises ``E[R]`` and is mathematically incentivised to
collapse the policy onto the single highest-scoring molecule it has found -
useless in drug discovery, where 50 structurally diverse chemical series are
worth more than one greedy hit. GFlowNets (Bengio et al., 2022) instead
train the policy to sample molecules *proportional to their reward*,

    P(x) = R(x) / Z,

so every high-reward mode (scaffold) is sampled according to its share of
the reward mass - the anti-mode-collapse objective the tasklist prescribes.

This module implements the **Trajectory Balance** objective (Malkin et al.,
2022). For a molecular-assembly trajectory ``tau = (s_0 -> s_1 -> ... ->
s_T = x)``:

    L_TB(tau) = ( log Z + sum_t log P_F(s_{t+1} | s_t)
                  - log R(x) - sum_t log P_B(s_t | s_{t+1}) )^2

When ``L_TB -> 0`` the sampling probability provably satisfies
``P(x) = R(x) / Z``.

Backward policy in this MDP: every forward step attaches exactly one synthon
through a single new bond, so the precursor of any reachable state is unique
(undo the last attachment). ``P_B`` is therefore deterministic and
``sum_t log P_B = 0``; the TB loss correspondingly simplifies to
``(log Z + sum log P_F - log R)^2``. The initial seed synthon is chosen by
the hotspot placement heuristic (not by the policy), so TB conditions on
``s_0`` exactly as the formulation requires.

The policy's factorized log-probability ``log P(reaction family) +
log P(synthon | family)`` is the forward flow contribution; the von Mises
torsion head stays supervised-only (identical to the PPO objective's scope).
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from syntree.engine.rl import joint_log_prob_and_entropy

logger = logging.getLogger(__name__)


class GFlowNetTrainer:
    """Trajectory Balance (TB) GFlowNet optimizer for 3D-SynTree.

    Args:
        model: the :class:`~syntree.models.policy.SynTreePolicy`.
        catalog: the :class:`~syntree.chemistry.catalog.SynthonCatalog`.
        device: torch device.
        config: ``reinforcement_learning`` config block. Recognised keys:
            ``learning_rate`` (default 1e-4), ``lr_z`` (log-Z learning rate,
            default 1e-2 - a higher rate for the normalizing constant is
            standard practice), ``weight_decay``, ``max_grad_norm``,
            ``reward_floor`` (default 1e-4, protects ``log R``),
            ``tb_diff_clip`` (optional clamp on the TB residual; default 40).
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
        self.lr = float(cfg.get("learning_rate", 1e-4))
        self.lr_z = float(cfg.get("lr_z", 1e-2))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.reward_floor = float(cfg.get("reward_floor", 1e-4))
        self.tb_diff_clip = float(cfg.get("tb_diff_clip", 40.0))

        # Learnable log-partition function log(Z) - a single scalar parameter
        # representing the total flow. No weight decay (it is a normalizer,
        # not a network weight).
        self.log_Z = torch.nn.Parameter(torch.zeros(1, device=device))
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.lr},
                {"params": [self.log_Z], "lr": self.lr_z, "weight_decay": 0.0},
            ],
            weight_decay=float(cfg.get("weight_decay", 1e-5)),
        )

    # ------------------------------------------------------------------
    # Trajectory Balance loss
    # ------------------------------------------------------------------
    def trajectory_balance_loss(
        self, trajectory_trace: Sequence, terminal_reward: float
    ) -> torch.Tensor:
        """Compute ``L_TB = (log Z + sum log P_F - log R - sum log P_B)^2``.

        Args:
            trajectory_trace: transitions of one sampled trajectory; each
                element must expose ``old_log_prob``-compatible state or be
                re-evaluated through :func:`joint_log_prob_and_entropy`. In
                practice the rollout buffer's
                :class:`~syntree.engine.rl.RolloutTransition` objects are
                passed, and the **current** policy log-probabilities are
                recomputed so the loss is differentiable (the
                collection-time log-probs are stale under a moving policy).
            terminal_reward: bounded terminal reward of the trajectory
                (floored at ``reward_floor`` to keep ``log R`` finite).

        Returns:
            Scalar TB loss for the trajectory.
        """
        reward = max(float(terminal_reward), self.reward_floor)
        log_R = torch.log(
            torch.tensor(reward, dtype=torch.float32, device=self.device)
        )

        sum_log_PF = torch.zeros(1, device=self.device)
        for step in trajectory_trace:
            log_pf, _, _ = joint_log_prob_and_entropy(
                self.model, self.catalog, self.device, step
            )
            sum_log_PF = sum_log_PF + log_pf

        # Deterministic backward policy in this chain-structured MDP:
        # log P_B(s_t | s_{t+1}) = 0 for every step (see module docstring).
        sum_log_PB = torch.zeros(1, device=self.device)

        diff = self.log_Z + sum_log_PF - log_R - sum_log_PB
        if self.tb_diff_clip and self.tb_diff_clip > 0:
            diff = torch.clamp(diff, -self.tb_diff_clip, self.tb_diff_clip)
        return diff.pow(2)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(
        self, batch_trajectories: Sequence[Tuple[Sequence, float]]
    ) -> Dict[str, float]:
        """Update the policy over a batch of sampled trajectories.

        Args:
            batch_trajectories: list of ``(transitions, terminal_reward)``
                pairs - exactly what the deferred-reward rollout loop
                produces.

        Returns:
            Summary statistics (mean TB loss, current ``log Z``, counts).
        """
        usable = [
            (trace, reward)
            for trace, reward in batch_trajectories
            if len(trace) > 0
        ]
        if not usable:
            return {"loss": 0.0, "log_Z": float(self.log_Z.item()),
                    "episodes": 0, "transitions": 0}

        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        for trace, reward in usable:
            losses.append(self.trajectory_balance_loss(trace, reward))
        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        params = list(self.model.parameters()) + [self.log_Z]
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(batch_loss.detach().item()),
            "log_Z": float(self.log_Z.detach().item()),
            "episodes": len(usable),
            "transitions": sum(len(t) for t, _ in usable),
        }

    # ------------------------------------------------------------------
    # Persistence (log_Z must survive checkpoints!)
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "log_Z": self.log_Z.detach().cpu().clone(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        if not state or "log_Z" not in state:
            return
        with torch.no_grad():
            self.log_Z.copy_(
                torch.as_tensor(state["log_Z"], device=self.log_Z.device)
                .to(self.log_Z.dtype)
                .view_as(self.log_Z)
            )
        if state.get("optimizer"):
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, RuntimeError) as exc:
                logger.warning("Skipping incompatible GFlowNet optimizer state: %s", exc)


__all__ = ["GFlowNetTrainer"]
