"""Direct Preference Optimization for 3D-SynTree (tasklist alternative).

DPO (Rafailov et al., 2023) adapts the policy directly on *preference
pairs* instead of reward maximisation: for a target pocket, sample two
candidate ligands, score both with the multi-objective oracle, label the
higher-scoring one ``M_w`` (preferred) and optimise

    L_DPO = -log sigma( beta * [ log pi_theta(M_w) - log pi_ref(M_w)
                                 - log pi_theta(M_l) + log pi_ref(M_l) ] )

There is no value network, no reward scaling, no GAE and no clipping -
training is as stable as supervised cross-entropy. ``log pi(M)`` is the
trajectory log-probability (sum of the factorized joint action
log-probabilities, the same quantity PPO and the GFlowNet engine train),
and ``pi_ref`` is a frozen copy of the policy at DPO initialisation.

Both candidates of a pair must come from the SAME pocket so the preference
reflects pocket-conditioned design quality rather than pocket difficulty.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from syntree.engine.rl import joint_log_prob_and_entropy

logger = logging.getLogger(__name__)


class DPOTrainer:
    """Pairwise-preference optimizer over chemistry-constrained trajectories.

    Args:
        model: the :class:`~syntree.models.policy.SynTreePolicy`.
        catalog: the :class:`~syntree.chemistry.catalog.SynthonCatalog`.
        device: torch device.
        config: ``reinforcement_learning`` config block. Recognised keys:
            ``dpo_beta`` (default 0.2), ``learning_rate`` (default 1e-5),
            ``weight_decay``, ``max_grad_norm``.
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
        self.beta = float(cfg.get("dpo_beta", 0.2))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.get("learning_rate", 1e-5)),
            weight_decay=float(cfg.get("weight_decay", 1e-5)),
        )
        # Frozen reference policy (pi_ref in the DPO objective). Detached
        # copy with gradients disabled; restored via `load_reference_state`.
        self.reference_model = copy.deepcopy(model).to(device)
        for param in self.reference_model.parameters():
            param.requires_grad_(False)
        self.reference_model.eval()

    # ------------------------------------------------------------------
    # Trajectory log-probabilities
    # ------------------------------------------------------------------
    def _trajectory_log_prob(
        self, model: torch.nn.Module, transitions: Sequence, no_grad: bool
    ) -> Optional[torch.Tensor]:
        if not transitions:
            return None
        total = torch.zeros(1, device=self.device)
        for step in transitions:
            log_prob, _, _ = joint_log_prob_and_entropy(
                model, self.catalog, self.device, step
            )
            total = total + log_prob
        return total

    # ------------------------------------------------------------------
    # DPO loss
    # ------------------------------------------------------------------
    def preference_loss(
        self,
        win_transitions: Sequence,
        lose_transitions: Sequence,
    ) -> Optional[torch.Tensor]:
        """DPO loss for one preference pair.

        Returns ``None`` when either trajectory is empty (nothing to prefer).
        """
        if not win_transitions or not lose_transitions:
            return None

        log_pi_w = self._trajectory_log_prob(self.model, win_transitions, False)
        log_pi_l = self._trajectory_log_prob(self.model, lose_transitions, False)
        with torch.no_grad():
            log_ref_w = self._trajectory_log_prob(
                self.reference_model, win_transitions, True
            )
            log_ref_l = self._trajectory_log_prob(
                self.reference_model, lose_transitions, True
            )

        margin = self.beta * (
            (log_pi_w - log_ref_w) - (log_pi_l - log_ref_l)
        )
        return -F.logsigmoid(margin)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(
        self, pairs: Sequence[Tuple[Sequence, Sequence]]
    ) -> Dict[str, float]:
        """Update the policy over a batch of preference pairs.

        Args:
            pairs: list of ``(win_transitions, lose_transitions)``.

        Returns:
            Summary statistics (mean DPO loss, mean margin, implicit reward
            gap, counts).
        """
        usable = [(w, l) for w, l in pairs if w and l]
        if not usable:
            return {"loss": 0.0, "margin": 0.0, "pairs": 0, "transitions": 0}

        self.optimizer.zero_grad(set_to_none=True)
        losses, margins = [], []
        for win, lose in usable:
            loss = self.preference_loss(win, lose)
            if loss is None:
                continue
            losses.append(loss)
            margins.append(float(loss.detach().item()))

        if not losses:
            return {"loss": 0.0, "margin": 0.0, "pairs": 0, "transitions": 0}

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return {
            "loss": float(batch_loss.detach().item()),
            "margin": float(
                torch.stack(
                    [torch.as_tensor(m, device=self.device) for m in margins]
                ).mean().item()
            ),
            "pairs": len(losses),
            "transitions": sum(len(w) + len(l) for w, l in usable),
        }

    # ------------------------------------------------------------------
    # Persistence: the reference policy must survive checkpoints, otherwise
    # a resumed run would compare against the *moved* policy.
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "reference_model": {
                k: v.detach().cpu().clone()
                for k, v in self.reference_model.state_dict().items()
            },
        }

    def load_state_dict(self, state: Dict) -> None:
        if not state:
            return
        if state.get("optimizer"):
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, RuntimeError) as exc:
                logger.warning("Skipping incompatible DPO optimizer state: %s", exc)
        reference = state.get("reference_model")
        if reference:
            self.reference_model.load_state_dict(reference, strict=False)


def build_preference_pairs(
    episodes: Sequence[Dict],
) -> List[Tuple[Sequence, Sequence]]:
    """Pair up consecutive episodes of a rollout batch by reward.

    Each pair holds two same-batch episodes ordered
    ``(win, lose)`` by terminal reward; exact ties are dropped (a preference
    between equal outcomes is undefined). Pairing within one batch keeps
    both candidates on comparable pockets.
    """
    pairs: List[Tuple[Sequence, Sequence]] = []
    for i in range(0, len(episodes) - 1, 2):
        a, b = episodes[i], episodes[i + 1]
        if not a.get("transitions") or not b.get("transitions"):
            continue
        reward_a = float(a.get("reward", 0.0))
        reward_b = float(b.get("reward", 0.0))
        if reward_a == reward_b:
            continue
        if reward_a > reward_b:
            pairs.append((a["transitions"], b["transitions"]))
        else:
            pairs.append((b["transitions"], a["transitions"]))
    return pairs


__all__ = ["DPOTrainer", "build_preference_pairs"]
