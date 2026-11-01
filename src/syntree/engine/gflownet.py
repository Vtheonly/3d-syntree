"""Trajectory Balance engine with the exact uniform-leaf backward policy.

Clean-room V2 implementation of the *reconciled* GFlowNet Trajectory
Balance core (issue #29 FLOW-01 as amended by its backward-policy
addendum; issue #38 FLOW-04 semantics; constitution #45 §12.3; decision
D-V2-009 in ``docs/agent/DECISIONS.md``).

**The loss.**  For a complete trajectory
``tau = (s_0, a_0, s_1, ..., a_{n-1}, s_n)`` ending in the terminal
molecule ``x`` (the last action is ``STOP``)::

    L_TB(tau) = ( log Z + sum_{t=0}^{n-1} log P_F(a_t | s_t)
                        - sum_{t=1}^{n}   log P_B(s_{t-1} | s_t)
                        - log R(x) )^2

All terms are dimensionless: every action component ``(h_t, r_t, B_t,
c_t, k_t)`` and the ``STOP`` decision are categorical probabilities
(issue #24), the reward enters as a single stationary scalar (issue #30),
and no continuous geometric density ever appears in the sum (the sub-bin
refinement of issue #42 is a deterministic environment-side operation of
the reward oracle, outside the probability measure).

**The backward policy (issue #38 — the multi-parent fix).**  The backward
transition is the *Uniform Leaf Deletion Policy* over removable leaf
synthons::

    P_B(s_{t-1} | s_t) = 1 / |RemovableLeafSynthons(s_t)|

so ``log P_B`` contributes ``-log(num_leaves)`` per assembly step, with
``num_leaves`` recorded on each step record by the environment (the count
of the state *after* the action, i.e. the state the backward process
stands on when undoing).  This is a proper probability distribution over
the valid parents of every non-initial state, which is exactly the
condition under which Trajectory Balance admits flows ``F(x) = R(x)`` on
terminal molecular graphs even when multiple reaction orderings or
symmetric disconnections converge to the same graph — the probability
mass of all derivations of ``x`` aggregates correctly (the ``N!``
multiplicity inflation of naive derivation-tree backward policies is
absent).  The identity is proven in
``docs/architecture/MDP_STATE_SPACE.md`` and verified on an enumerable toy
DAG in ``tests/test_gflownet_reconciliation.py``.

**The STOP backward term.**  The absorbing terminal reached by ``STOP``
has exactly one parent — the pre-STOP state — so
``log P_B(s_n | terminal) = log 1 = 0`` *by proof* (issue #38 acceptance:
``P_B = 1`` is used only where uniqueness is established for the actual
state representation).  The STOP step therefore contributes its forward
log-probability but no backward term; its recorded ``num_leaves`` is
telemetry only.

**Precision policy (decisions D-V2-005 / D-V2-008).**  Parameters default
to FP32 (the production T4 path).  ``float64`` is accepted only for local
numerical verification of the flow identities at tight tolerances (cast
with ``.to(torch.float64)``); every other dtype fails closed.

**Scope boundary.**  The seed-conditioned ``log Z_theta(P, s_0)`` MLP and
the trainer loop (``lr_Z = 1e-2`` per the #29 body) are owned by the full
#29/#26 trainer integration once the policy heads (#22–#25) exist; this
engine pins the exact TB mathematics and the backward semantics they must
consume.

This module depends on ``torch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch import nn

__all__ = [
    "TrajectoryBalanceEngine",
    "TrajectoryStep",
    "FlowError",
    "as_trajectory_steps",
]

#: Reward floor mirrored from the reward engine (log must stay finite).
_REWARD_FLOOR = 1e-4


class FlowError(ValueError):
    """Fail-closed Trajectory Balance error (malformed trajectory/reward)."""


@dataclass(frozen=True)
class TrajectoryStep:
    """One TB step record: forward log-prob + backward leaf count.

    ``action_log_prob`` is ``log P_F(a_t | s_t)`` — the *total* forward
    log-probability of the action tuple ``(h_t, r_t, B_t, c_t, k_t)``
    (sum of its component log-probabilities, issue #24) or of the STOP
    decision when ``action_type == 'stop'``.  ``num_leaves`` is the
    removable-leaf count of the state *after* the action — the
    backward-policy input ``|RemovableLeafSynthons(s_{t+1})|``.
    """

    action_type: str
    action_log_prob: float
    num_leaves: int = 0

    def __post_init__(self) -> None:
        if self.action_type not in ("add", "stop"):
            raise FlowError(
                f"action_type must be 'add' or 'stop', got {self.action_type!r}"
            )
        if self.action_type == "add":
            if self.num_leaves < 1:
                raise FlowError(
                    f"an 'add' step must record num_leaves >= 1 (the product state "
                    f"has at least the added fragment as a removable leaf); got "
                    f"{self.num_leaves}"
                )
        if not math.isfinite(self.action_log_prob):
            raise FlowError(
                f"action_log_prob must be finite, got {self.action_log_prob}"
            )


#: Anything mapping the documented keys is accepted as a step record.
StepLike = Union[Mapping[str, Any], TrajectoryStep, Any]


def as_trajectory_steps(trajectory: Sequence[StepLike]) -> List[TrajectoryStep]:
    """Normalise a trajectory (dicts or records) into validated records.

    Accepts ``TrajectoryStep`` instances, mappings with the keys
    ``action_type`` / ``action_log_prob`` / ``num_leaves``, or any object
    exposing those attributes (e.g. the environment's
    :class:`~syntree.engine.environment.StepRecord` after the policy has
    filled ``action_log_prob``).
    """
    if not trajectory:
        raise FlowError("an empty trajectory has no Trajectory Balance terms")
    steps: List[TrajectoryStep] = []
    for index, item in enumerate(trajectory):
        if isinstance(item, TrajectoryStep):
            steps.append(item)
            continue
        if isinstance(item, Mapping):
            try:
                steps.append(
                    TrajectoryStep(
                        action_type=item["action_type"],
                        action_log_prob=float(item["action_log_prob"]),
                        num_leaves=int(item.get("num_leaves", 0)),
                    )
                )
            except KeyError as error:
                raise FlowError(
                    f"trajectory step {index} lacks the required key {error}"
                ) from error
            continue
        action_type = getattr(item, "action_type", None)
        log_prob = getattr(item, "action_log_prob", None)
        if action_type is None or log_prob is None:
            raise FlowError(
                f"trajectory step {index} of type {type(item).__name__} exposes "
                f"neither the mapping keys nor the required attributes"
            )
        steps.append(
            TrajectoryStep(
                action_type=str(action_type),
                action_log_prob=float(log_prob),
                num_leaves=int(getattr(item, "num_leaves", 0) or 0),
            )
        )
    if steps[-1].action_type != "stop":
        raise FlowError(
            "a complete trajectory must end with the absorbing STOP action; got "
            f"{steps[-1].action_type!r}"
        )
    return steps


class TrajectoryBalanceEngine(nn.Module):
    """The Trajectory Balance loss with the uniform leaf backward policy.

    Owns the learnable log-partition function ``log Z`` (a scalar
    parameter here; the seed-conditioned MLP ``log Z_theta(P, s_0)`` of
    the full #29 trainer replaces it behind the same attribute once the
    backbone pools exist — the loss mathematics is unchanged).

    Usage::

        engine = TrajectoryBalanceEngine()
        loss = engine.trajectory_balance_loss(trajectory, reward)
        loss.backward()   # optimises log Z and (via policy graphs) P_F

    ``trajectory`` is a sequence of step records (see
    :func:`as_trajectory_steps`); ``reward`` is the stationary
    ``R(x) >= 1e-4`` from :class:`~syntree.engine.rl.ThreeDReward`.
    """

    def __init__(self, initial_log_z: float = 0.0) -> None:
        super().__init__()
        if not math.isfinite(initial_log_z):
            raise FlowError(f"initial_log_z must be finite, got {initial_log_z}")
        #: Learnable log-partition function.  lr_Z = 1e-2 is the #29-specified
        #: learning rate for this parameter in the trainer integration.
        self.log_z = nn.Parameter(torch.tensor(float(initial_log_z), dtype=torch.float32))

    @property
    def device(self) -> torch.device:
        return self.log_z.device

    @property
    def dtype(self) -> torch.dtype:
        return self.log_z.dtype

    def _log_reward(self, reward: float) -> torch.Tensor:
        if not math.isfinite(reward):
            raise FlowError(f"reward must be finite, got {reward}")
        if reward < _REWARD_FLOOR:
            raise FlowError(
                f"reward must be >= {_REWARD_FLOOR} (the hard floor keeps log R "
                f"finite; issue #30); got {reward}"
            )
        return torch.log(
            torch.tensor(float(reward), device=self.device, dtype=self.dtype)
        )

    def trajectory_residual(self, trajectory: Sequence[StepLike], reward: float) -> torch.Tensor:
        """The signed TB residual (zero when the flows are exact).

        ``log Z + sum log P_F - sum log P_B - log R(x)`` as a scalar tensor
        — exposed separately because the toy-DAG verification asserts it is
        exactly zero for the analytically optimal policy.
        """
        steps = as_trajectory_steps(trajectory)
        log_reward = self._log_reward(reward)

        sum_log_pf = torch.zeros((), device=self.device, dtype=self.dtype)
        sum_log_pb = torch.zeros((), device=self.device, dtype=self.dtype)
        for step in steps:
            # Forward: every action — including STOP — contributes its
            # dimensionless categorical log-probability (constitution §12.3).
            sum_log_pf = sum_log_pf + step.action_log_prob
            # Backward: uniform leaf deletion over the *product* state of an
            # 'add' action.  The STOP step's backward term is exactly zero:
            # the absorbing terminal has the unique pre-STOP parent (proved
            # in docs/architecture/MDP_STATE_SPACE.md §6).
            if step.action_type == "add":
                if step.num_leaves < 1:
                    raise FlowError(
                        f"an 'add' step must record num_leaves >= 1, got {step.num_leaves}"
                    )
                sum_log_pb = sum_log_pb - math.log(step.num_leaves)

        return self.log_z + sum_log_pf - sum_log_pb - log_reward

    def trajectory_balance_loss(
        self, trajectory: Sequence[StepLike], reward: float
    ) -> torch.Tensor:
        """Squared TB residual for one trajectory (the scalar loss)."""
        residual = self.trajectory_residual(trajectory, reward)
        return residual.pow(2)

    def batch_trajectory_balance_loss(
        self,
        trajectories: Sequence[Sequence[StepLike]],
        rewards: Sequence[float],
    ) -> Dict[str, torch.Tensor]:
        """Mean TB loss over a rollout batch, with per-trajectory residuals.

        Returns ``{'loss': mean, 'residuals': tensor[n]}``.  The batch is a
        *rollout grouping only*: no term couples the trajectories (the
        reward is stationary per molecule — issue #30).
        """
        if len(trajectories) != len(rewards):
            raise FlowError(
                f"trajectories ({len(trajectories)}) and rewards ({len(rewards)}) "
                f"disagree in length"
            )
        if not trajectories:
            raise FlowError("the batch is empty")
        residuals = torch.stack(
            [
                self.trajectory_residual(trajectory, reward)
                for trajectory, reward in zip(trajectories, rewards)
            ]
        )
        return {"loss": residuals.pow(2).mean(), "residuals": residuals}
