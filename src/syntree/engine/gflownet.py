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

**The seed backward term (issue #39).**  The placed-seed state is a
single-fragment molecule: it has no removable leaf synthons (every leaf
is non-seed by definition), so the only possible backward transition is
removing the seed entirely, returning to the empty pre-seed initial
state — the *unique* parent.  By the same uniqueness argument as STOP,
``log P_B(s_0 | placed-seed state) = log 1 = 0``: the seed step
contributes its forward log-probability (the categorical
``P(anchor | pocket) * P(seed | anchor, pocket)`` of the #39 head) and
no backward term, and the §4 induction base extends verbatim (backward
path measure from the placed-seed state to the empty state is 1).

**Precision policy (decisions D-V2-005 / D-V2-008).**  Parameters default
to FP32 (the production T4 path).  ``float64`` is accepted only for local
numerical verification of the flow identities at tight tolerances (cast
with ``.to(torch.float64)``); every other dtype fails closed.

**Differentiable log-probabilities (#29).**  ``action_log_prob`` accepts
a 0-dim ``torch.Tensor`` in addition to a Python float: a tensor value
keeps its autograd graph, so ``trajectory_residual``/``loss`` backpropagate
into the policy parameters that produced it (the on-policy TB training
path).  Float values remain the exact-verification form (the toy-DAG
identity checks).  Either way the value must be finite — a non-finite
log-probability is a fail-closed :class:`FlowError`, never a clipped or
silently skipped term.

**The seed-conditioned partition function (#29).**
:class:`LogPartitionHead` parameterizes ``log Z_theta(P, s_0)`` as a
lightweight 2-layer MLP over the *pooled pocket latent* (the backbone's
invariant ``pocket_context`` surface) concatenated with the *seed latent*
(the tied catalog embedding of the selected seed block) — exactly the two
conditioning surfaces the #39 seed stage already consumes.  Its output
enters the TB residual through the ``log_z`` override of
:meth:`TrajectoryBalanceEngine.trajectory_residual` (the equation's
``log Z`` term; the engine's own scalar parameter remains the analytic
verification form).  ``lr_Z = 1e-2`` is the #29-specified learning rate of
this head in :class:`GFlowNetTrainer`'s second optimizer param group.

**The trainer (#29).**  :class:`GFlowNetTrainer` owns the Stage-2
optimization loop: on-policy rollout batches (each a :class:`TBTrajectory`
— step records with tensor log-probabilities, the stationary reward, and
the trajectory's ``(pocket, seed)`` latent pair), the TB loss through the
engine, fail-closed divergence handling (no NaN shields, no batch
skipping — the #10/#26 convention), gradient clipping, and the atomic
SHA-256 checkpoint contract carrying policy weights, the ``log Z_theta``
MLP state, both optimizer param groups, progress and RNG states (#43
interface; #31 consumes it for Stage-2 CLI resume).

This module depends on ``torch``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch import nn

from syntree.engine.trainer import (
    TrainingDivergenceError,
    gpu_memory_telemetry,
    host_memory_telemetry,
)
from syntree.utils.hardware import configure_runtime_environment
from syntree.utils.seeding import (
    capture_rng_state,
    restore_rng_state,
    rng_state_from_dict,
    rng_state_to_dict,
    seed_everything,
)

__all__ = [
    "TrajectoryBalanceEngine",
    "TrajectoryStep",
    "FlowError",
    "as_trajectory_steps",
    "LogPartitionHead",
    "TBTrajectory",
    "GFlowNetProfile",
    "GFlowNetTrainer",
]

#: Reward floor mirrored from the reward engine (log must stay finite).
_REWARD_FLOOR = 1e-4


class FlowError(ValueError):
    """Fail-closed Trajectory Balance error (malformed trajectory/reward)."""


#: A forward log-probability: a Python float (exact verification form) or a
#: 0-dim ``torch.Tensor`` (the differentiable on-policy form, #29).
LogProb = Union[float, torch.Tensor]


def _check_log_prob_finite(value: LogProb) -> None:
    """Fail-closed finiteness check for either log-probability form."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise FlowError(
                f"a tensor action_log_prob must hold exactly one element, got "
                f"{tuple(value.shape)}"
            )
        if value.dtype not in (torch.float32, torch.float64):
            raise FlowError(
                f"a tensor action_log_prob must be float32 or float64, got "
                f"{value.dtype}"
            )
        if not bool(torch.isfinite(value).all()):
            raise FlowError(
                f"action_log_prob must be finite, got {float(value.detach())}"
            )
        return
    if not math.isfinite(float(value)):
        raise FlowError(f"action_log_prob must be finite, got {value}")


@dataclass(frozen=True)
class TrajectoryStep:
    """One TB step record: forward log-prob + backward leaf count.

    ``action_log_prob`` is ``log P_F(a_t | s_t)`` — the *total* forward
    log-probability of the action tuple ``(h_t, r_t, B_t, c_t, k_t)``
    (sum of its component log-probabilities, issue #24), of the STOP
    decision when ``action_type == 'stop'``, or of the categorical seed
    action ``P(anchor|pocket) * P(seed|anchor,pocket)`` when
    ``action_type == 'seed'`` (issue #39).  It is either a Python float
    (the exact-verification form) or a 0-dim ``torch.Tensor`` attached to
    the policy graph (the differentiable on-policy form, #29) — the sum
    in :meth:`TrajectoryBalanceEngine.trajectory_residual` keeps the
    graph either way.  ``num_leaves`` is the removable-leaf count of the
    state *after* the action — the backward-policy input
    ``|RemovableLeafSynthons(s_{t+1})|``; the seed and STOP steps carry
    no backward term (unique parents, proven in ``MDP_STATE_SPACE.md``)
    so their count is telemetry only.
    """

    action_type: str
    action_log_prob: LogProb
    num_leaves: int = 0

    def __post_init__(self) -> None:
        if self.action_type not in ("seed", "add", "stop"):
            raise FlowError(
                f"action_type must be 'seed', 'add', or 'stop', got {self.action_type!r}"
            )
        if self.action_type == "add":
            if self.num_leaves < 1:
                raise FlowError(
                    f"an 'add' step must record num_leaves >= 1 (the product state "
                    f"has at least the added fragment as a removable leaf); got "
                    f"{self.num_leaves}"
                )
        if self.action_type == "seed" and self.num_leaves != 0:
            raise FlowError(
                "a 'seed' step has no removable leaves (num_leaves must be 0)"
            )
        _check_log_prob_finite(self.action_log_prob)


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
                        action_log_prob=item["action_log_prob"],
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
                action_log_prob=log_prob,
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

    def _coerce_log_z(self, log_z: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Validate/cast a caller-supplied ``log Z`` term (the #29 override).

        Accepts ``None`` (use the engine's own scalar), a 0-dim/1-element
        tensor (one trajectory), or a ``[B]`` batch tensor; the value must
        be finite (float32/float64 only) and is cast to the engine's
        device/dtype.
        """
        if log_z is None:
            return None
        if not isinstance(log_z, torch.Tensor):
            raise FlowError(
                f"log_z must be a torch.Tensor or None, got {type(log_z).__name__}"
            )
        if log_z.dtype not in (torch.float32, torch.float64):
            raise FlowError(
                f"log_z must be float32 or float64, got {log_z.dtype}"
            )
        if log_z.ndim > 1:
            raise FlowError(
                f"log_z must be a scalar or [B] tensor, got {tuple(log_z.shape)}"
            )
        if not bool(torch.isfinite(log_z).all()):
            raise FlowError(
                f"log_z must be finite, got non-finite entries"
            )
        return log_z.to(device=self.device, dtype=self.dtype)

    def trajectory_residual(
        self,
        trajectory: Sequence[StepLike],
        reward: float,
        *,
        log_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The signed TB residual (zero when the flows are exact).

        ``log Z + sum log P_F - sum log P_B - log R(x)`` as a scalar tensor
        — exposed separately because the toy-DAG verification asserts it is
        exactly zero for the analytically optimal policy.

        ``log_z`` (the #29 seed-conditioned form): when supplied, the
        caller's tensor replaces the engine's own ``log_z`` parameter in
        the equation — a graph-connected value makes the residual
        differentiable through ``log Z_theta(P, s_0)`` while the pinned
        mathematics is unchanged.  ``None`` keeps the engine scalar (the
        analytic verification form).
        """
        steps = as_trajectory_steps(trajectory)
        log_reward = self._log_reward(reward)
        z_term = self._coerce_log_z(log_z)

        sum_log_pf = torch.zeros((), device=self.device, dtype=self.dtype)
        sum_log_pb = torch.zeros((), device=self.device, dtype=self.dtype)
        for step in steps:
            # Forward: every action — seed, assembly components, and STOP —
            # contributes its dimensionless categorical log-probability
            # (constitution §12.3; the #39 seed action included).  A tensor
            # value keeps its autograd graph (#29); a float broadcasts.
            value = step.action_log_prob
            if isinstance(value, torch.Tensor):
                if value.dtype not in (torch.float32, torch.float64):
                    raise FlowError(
                        f"a tensor action_log_prob must be float32/float64, "
                        f"got {value.dtype}"
                    )
                value = value.to(device=self.device, dtype=self.dtype)
            sum_log_pf = sum_log_pf + value
            # Backward: uniform leaf deletion over the *product* state of an
            # 'add' action.  The STOP step's backward term is exactly zero:
            # the absorbing terminal has the unique pre-STOP parent (proved
            # in docs/architecture/MDP_STATE_SPACE.md §6).  The seed step's
            # backward term is exactly zero by the same uniqueness argument:
            # the placed-seed state's only parent is the empty initial
            # state (no removable leaves exist to delete selectively).
            if step.action_type == "add":
                if step.num_leaves < 1:
                    raise FlowError(
                        f"an 'add' step must record num_leaves >= 1, got {step.num_leaves}"
                    )
                sum_log_pb = sum_log_pb - math.log(step.num_leaves)

        if z_term is not None:
            return z_term + sum_log_pf - sum_log_pb - log_reward
        return self.log_z + sum_log_pf - sum_log_pb - log_reward

    def trajectory_balance_loss(
        self,
        trajectory: Sequence[StepLike],
        reward: float,
        *,
        log_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Squared TB residual for one trajectory (the scalar loss)."""
        residual = self.trajectory_residual(trajectory, reward, log_z=log_z)
        return residual.pow(2)

    def batch_trajectory_balance_loss(
        self,
        trajectories: Sequence[Sequence[StepLike]],
        rewards: Sequence[float],
        *,
        log_z: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Mean TB loss over a rollout batch, with per-trajectory residuals.

        Returns ``{'loss': mean, 'residuals': tensor[n]}``.  The batch is a
        *rollout grouping only*: no term couples the trajectories (the
        reward is stationary per molecule — issue #30).

        ``log_z``: ``None`` (the engine scalar for every trajectory), a
        0-dim tensor (one shared value), or a ``[B]`` tensor of
        per-trajectory ``log Z_theta(P, s_0)`` values (#29) — each
        trajectory's residual uses its own conditioning value.
        """
        if len(trajectories) != len(rewards):
            raise FlowError(
                f"trajectories ({len(trajectories)}) and rewards ({len(rewards)}) "
                f"disagree in length"
            )
        if not trajectories:
            raise FlowError("the batch is empty")
        z_term = self._coerce_log_z(log_z)
        if z_term is not None and z_term.ndim == 1 and z_term.shape[0] != len(trajectories):
            raise FlowError(
                f"a batched log_z must have one entry per trajectory: "
                f"{tuple(z_term.shape)} vs {len(trajectories)}"
            )
        residuals = []
        for index, (trajectory, reward) in enumerate(zip(trajectories, rewards)):
            per_trajectory_z = (
                z_term[index] if z_term is not None and z_term.ndim == 1 else z_term
            )
            residuals.append(
                self.trajectory_residual(trajectory, reward, log_z=per_trajectory_z)
            )
        residuals = torch.stack(residuals)
        return {"loss": residuals.pow(2).mean(), "residuals": residuals}


# ---------------------------------------------------------------------------
# The seed-conditioned partition function (#29 FLOW-01)
# ---------------------------------------------------------------------------

#: Width of the pooled pocket latent surface (the frozen backbone width —
#: ``BackboneOutput.pocket_context`` / ``DECODER_D_MODEL``).
POCKET_LATENT_DIM = 256
#: Width of the seed latent surface (the tied catalog embedding —
#: ``SYNTON_EMBED_DIM``).
SEED_LATENT_DIM = 64


class LogPartitionHead(nn.Module):
    """The seed-conditioned partition function ``log Z_theta(P, s_0)`` (#29).

    A lightweight 2-layer MLP over the *pooled pocket latent* concatenated
    with the *seed latent*::

        log Z_theta(P, s_0) = W_2 . SiLU(W_1 . [pool(P) ; embed(s_0)] + b_1) + b_2

    * ``pool(P)`` — the backbone's invariant pooled pocket context
      ``[B, 256]`` (the exact surface the #39 seed/anchor head consumes);
    * ``embed(s_0)`` — the tied catalog embedding of the selected seed
      block ``[B, 64]`` (the exact surface the parity/torsion heads
      consume).

    The final projection is zero-initialized so the head starts at the
    documented deterministic value ``log Z = 0`` (a controlled, reproducible
    initial state; the first optimizer step moves it).  ``lr_Z = 1e-2`` is
    the #29-specified learning rate for this head in
    :class:`GFlowNetTrainer`'s dedicated optimizer param group.

    Parameter count (defaults): ``(256 + 64) * 256 + 256 + 256 + 1 =
    82,433`` — 0.76% of the policy assembly (10,880,719), "lightweight" as
    the issue requires.
    """

    def __init__(
        self,
        pocket_dim: int = POCKET_LATENT_DIM,
        seed_dim: int = SEED_LATENT_DIM,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        pocket_dim = int(pocket_dim)
        seed_dim = int(seed_dim)
        hidden_dim = int(hidden_dim)
        if pocket_dim < 1 or seed_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f"LogPartitionHead dims must be >= 1, got pocket={pocket_dim}, "
                f"seed={seed_dim}, hidden={hidden_dim}"
            )
        self.pocket_dim = pocket_dim
        self.seed_dim = seed_dim
        self.hidden_dim = hidden_dim
        self.trunk = nn.Linear(pocket_dim + seed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def extra_repr(self) -> str:
        return (
            f"pocket_dim={self.pocket_dim}, seed_dim={self.seed_dim}, "
            f"hidden_dim={self.hidden_dim}"
        )

    def forward(self, pocket_latent: torch.Tensor, seed_latent: torch.Tensor) -> torch.Tensor:
        """``log Z_theta(P, s_0)`` ``[B]`` for a batch of conditioning pairs.

        Both inputs are validated fail-closed (2-D, matching batch, finite,
        float32/float64) — a poisoned latent is never silently projected.
        """
        for name, tensor, dim in (
            ("pocket_latent", pocket_latent, self.pocket_dim),
            ("seed_latent", seed_latent, self.seed_dim),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise FlowError(
                    f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
                )
            if tensor.ndim != 2 or tensor.shape[1] != dim:
                raise FlowError(
                    f"{name} must be [B, {dim}], got {tuple(tensor.shape)}"
                )
            if tensor.dtype not in (torch.float32, torch.float64):
                raise FlowError(
                    f"{name} must be float32 or float64, got {tensor.dtype}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise FlowError(f"{name} contains non-finite values (fail-closed)")
        if pocket_latent.shape[0] != seed_latent.shape[0]:
            raise FlowError(
                f"pocket_latent and seed_latent disagree in batch: "
                f"{pocket_latent.shape[0]} vs {seed_latent.shape[0]}"
            )
        if pocket_latent.shape[0] < 1:
            raise FlowError("the log Z batch is empty (fail-closed)")
        concat = torch.cat(
            [pocket_latent.to(self.trunk.weight.dtype),
             seed_latent.to(self.trunk.weight.dtype)],
            dim=1,
        )
        out = self.output(torch.nn.functional.silu(self.trunk(concat)))
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# The rollout-trajectory container + the Stage-2 trainer (#29)
# ---------------------------------------------------------------------------


@dataclass
class TBTrajectory:
    """One complete on-policy rollout for Trajectory Balance (#29).

    ``steps`` are the environment's recorded steps with the policy's
    *tensor* forward log-probabilities filled in (the rollout driver's
    contract): ``action_type`` / ``action_log_prob`` (0-dim tensor or
    float) / ``num_leaves``.  ``reward`` is the stationary ``R(x)`` of the
    relaxed terminal molecule.  The latent pair is the trajectory's
    conditioning: ``pocket_latent`` — the pooled pocket context of the
    pre-seed backbone pass ``[pocket_dim]``; ``seed_latent`` — the tied
    catalog embedding of the selected seed ``[seed_dim]``.
    ``terminal_smiles`` is telemetry (the terminal object's identity).
    """

    steps: List[Any]
    reward: float
    pocket_latent: torch.Tensor
    seed_latent: torch.Tensor
    terminal_smiles: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise FlowError("a TBTrajectory needs at least the STOP step")
        last_type = getattr(self.steps[-1], "action_type", None)
        if isinstance(self.steps[-1], Mapping):
            last_type = self.steps[-1].get("action_type")
        if last_type != "stop":
            raise FlowError(
                "a TBTrajectory must end with the absorbing STOP action "
                f"(got {last_type!r}) — the seed/add/stop sequence is the TB unit"
            )
        if not math.isfinite(float(self.reward)):
            raise FlowError(f"reward must be finite, got {self.reward}")


@dataclass(frozen=True)
class GFlowNetProfile:
    """The frozen Stage-2 optimization profile (#29/#31; D-V2-040).

    ``log_z_learning_rate`` is the #29-pinned ``lr_Z = 1e-2``.  The policy
    group continues from the Stage-1 supervised setting (AdamW 3e-4 /
    1e-5, clip 1.0 — ``configs/train_colab_t4.json`` Profile A); Stage-2
    resumes the same architecture and precision path, so the policy
    optimizer hyperparameters carry over unchanged (documented in
    D-V2-040; the seed-conditioned log Z head is the only new parameter
    group).
    """

    learning_rate: float = 3e-4
    log_z_learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 0
    profile_name: str = "gflownet-stage2-a"

    def __post_init__(self) -> None:
        for name in ("learning_rate", "log_z_learning_rate"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not (math.isfinite(float(self.weight_decay)) and float(self.weight_decay) >= 0.0):
            raise ValueError(f"weight_decay must be finite and >= 0, got {self.weight_decay}")
        if not (math.isfinite(float(self.gradient_clip_norm)) and float(self.gradient_clip_norm) > 0.0):
            raise ValueError(
                f"gradient_clip_norm must be finite and > 0, got {self.gradient_clip_norm}"
            )

    def digest(self) -> str:
        import hashlib as _hashlib

        payload = json.dumps(asdict(self), sort_keys=True)
        return _hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _GFlowNetProgress:
    """The Stage-2 progress counter (global step + batches)."""

    def __init__(self, global_step: int = 0, updates: int = 0) -> None:
        self.global_step = int(global_step)
        self.updates = int(updates)


class GFlowNetTrainer:
    """The Stage-2 Trajectory Balance trainer (#29 FLOW-01).

    Owns the on-policy optimization loop over rollout batches: the TB loss
    through :class:`TrajectoryBalanceEngine` with per-trajectory
    ``log Z_theta(P, s_0)`` from :class:`LogPartitionHead`, the two-group
    AdamW optimizer (policy @ ``learning_rate``; the log Z head @
    ``log_z_learning_rate = 1e-2``, no weight decay on the partition
    head), fail-closed divergence handling, gradient clipping, and the
    atomic SHA-256 checkpoint contract (#43 interface; #31's Stage-2 CLI
    resumes through it).

    Usage (the #31 canonical entry point behind
    ``scripts/train_stage2.py``)::

        trainer = GFlowNetTrainer(policy, run_dir=..., device=...,
                                  catalog_block_handle_classes=...,
                                  catalog_seed_rows=...)
        for batch in rollout_batches:        # the rollout driver's output
            record = trainer.train_batch(batch)

    The rollout driver (``syntree/engine/rollout.py``) produces the
    :class:`TBTrajectory` batches by driving
    :class:`~syntree.engine.environment.AssemblyEnvironment` with the
    policy's categorical laws; this trainer never samples itself — the
    probability measure lives in the driver, the optimization lives here.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        run_dir: str,
        profile: Optional[GFlowNetProfile] = None,
        device: Optional[torch.device] = None,
        pocket_latent_dim: int = POCKET_LATENT_DIM,
        seed_latent_dim: int = SEED_LATENT_DIM,
        resume: bool = False,
    ) -> None:
        self.profile = profile or GFlowNetProfile()
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        # -- the strict hardware gate (#10 MATH-04; the #26 convention) ----
        self.runtime = configure_runtime_environment(device=device)
        if (
            self.runtime.precision != "fp32"
            or self.runtime.grad_scaler
            or self.runtime.autocast_enabled
        ):
            raise FlowError(
                "the hardware gate did not enforce strict FP32 without "
                f"GradScaler/autocast (got {self.runtime.summary()!r}) — "
                "refusing to train (#45 s5.1)"
            )
        self.device = torch.device(self.runtime.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        # -- determinism ------------------------------------------------------
        seed_everything(self.profile.seed, device=self.device)

        self.policy = policy.to(device=self.device, dtype=torch.float32)
        self.policy.train()
        # The seed-conditioned log Z head (#29): lr_Z = 1e-2, no weight
        # decay (the partition function is not a regularization target).
        self.log_partition = LogPartitionHead(
            pocket_dim=int(pocket_latent_dim), seed_dim=int(seed_latent_dim)
        ).to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(self.policy.parameters()),
                    "lr": float(self.profile.learning_rate),
                    "weight_decay": float(self.profile.weight_decay),
                    "name": "policy",
                },
                {
                    "params": list(self.log_partition.parameters()),
                    "lr": float(self.profile.log_z_learning_rate),
                    "weight_decay": 0.0,
                    "name": "log_z",
                },
            ]
        )
        # The pinned TB mathematics (the engine's own scalar log_z stays the
        # analytic-verification parameter; production log Z is the MLP).
        self.engine = TrajectoryBalanceEngine().to(
            device=self.device, dtype=torch.float32
        )

        self.progress = _GFlowNetProgress()
        self.history: List[Dict[str, Any]] = []
        self._rng_snapshot = capture_rng_state()
        self._started = time.monotonic()
        if resume:
            self.load_checkpoint(self.checkpoint_path)

    # -- paths ---------------------------------------------------------------

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, "gflownet_state.pt")

    # -- the optimization step -------------------------------------------------

    def train_batch(self, trajectories: Sequence[TBTrajectory]) -> Dict[str, Any]:
        """One TB update over a rollout batch; returns the telemetry record.

        Fail-closed contracts (the #10/#26 convention — no NaN shields, no
        batch skipping): every trajectory must carry finite latent pairs and
        a reward at or above the floor; a non-finite loss, gradient, or
        parameter raises :class:`TrainingDivergenceError` immediately.
        """
        if not trajectories:
            raise FlowError("the rollout batch is empty (fail-closed)")
        pocket = torch.stack(
            [t.pocket_latent.to(self.device, torch.float32) for t in trajectories]
        )
        seed = torch.stack(
            [t.seed_latent.to(self.device, torch.float32) for t in trajectories]
        )
        log_z = self.log_partition(pocket, seed)  # [B]
        steps_list = [list(t.steps) for t in trajectories]
        rewards = [float(t.reward) for t in trajectories]
        result = self.engine.batch_trajectory_balance_loss(
            steps_list, rewards, log_z=log_z
        )
        loss = result["loss"]
        if not bool(torch.isfinite(loss).all()):
            raise TrainingDivergenceError(
                f"non-finite TB loss at update {self.progress.updates} — "
                "a divergence is a bug to fix, never a batch to skip"
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # every trainable parameter group must carry finite gradients
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise TrainingDivergenceError(
                        f"non-finite gradient in the {group.get('name', '?')} "
                        f"group at update {self.progress.updates}"
                    )
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            float(self.profile.gradient_clip_norm),
        )
        self.optimizer.step()
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if not bool(torch.isfinite(parameter).all()):
                    raise TrainingDivergenceError(
                        f"non-finite parameter after the update "
                        f"{self.progress.updates} ({group.get('name', '?')} group)"
                    )
        self.progress.updates += 1
        self.progress.global_step += sum(len(s) for s in steps_list)
        record: Dict[str, Any] = {
            "update": self.progress.updates,
            "global_step": self.progress.global_step,
            "loss": float(loss.detach()),
            "residual_abs_mean": float(result["residuals"].detach().abs().mean()),
            "log_z_mean": float(log_z.detach().mean()),
            "log_z_std": float(log_z.detach().std(unbiased=False)) if len(trajectories) > 1 else 0.0,
            "batch_size": len(trajectories),
            "reward_mean": float(sum(rewards) / len(rewards)),
        }
        self.history.append(record)
        return record

    # -- telemetry (#26/#43 contract) -------------------------------------------

    def telemetry(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile": self.profile.profile_name,
            "updates": self.progress.updates,
            "global_step": self.progress.global_step,
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "runtime": self.runtime.summary(),
            "host_memory": host_memory_telemetry(),
            "gpu_memory": gpu_memory_telemetry(
                self.device if str(self.device).startswith("cuda") else None
            ),
        }
        if self.history:
            record["last_update"] = self.history[-1]
        return record

    def log_telemetry(self, record: Optional[Dict[str, Any]] = None) -> None:
        path = os.path.join(self.run_dir, "telemetry.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record or self.telemetry(), sort_keys=True) + "\n")

    # -- checkpoints (the #43 interface; #31 consumes them) ----------------------

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Atomically persist the full Stage-2 continuation state.

        Payload: the policy weights, **the neural partition function
        ``log Z_theta`` state** (the #31 requirement), the optimizer (both
        param groups — the ``lr_Z`` group included), progress, RNG states
        and the profile digest — written atomically with a SHA-256
        sidecar; :meth:`load_checkpoint` re-verifies before restoring.
        """
        path = path or self.checkpoint_path
        payload = {
            "format": "gflownet-stage2-v1",
            "profile": asdict(self.profile),
            "profile_digest": self.profile.digest(),
            "model": self.policy.state_dict(),
            "log_partition": self.log_partition.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "progress": {"global_step": self.progress.global_step,
                         "updates": self.progress.updates},
            "history": self.history,
            "rng": rng_state_to_dict(self._rng_snapshot),
            "runtime_summary": self.runtime.summary(),
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        digest = hashlib.sha256()
        with open(tmp, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        os.replace(tmp, path)
        sidecar = path + ".sha256"
        tmp_sidecar = sidecar + ".tmp"
        with open(tmp_sidecar, "w", encoding="utf-8") as handle:
            handle.write(digest.hexdigest() + "\n")
        os.replace(tmp_sidecar, sidecar)
        return path

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        """Restore the exact Stage-2 continuation state (fail-closed)."""
        path = path or self.checkpoint_path
        if not os.path.exists(path):
            raise FlowError(f"checkpoint not found: {path}")
        sidecar = path + ".sha256"
        if not os.path.exists(sidecar):
            raise FlowError(
                "checkpoint sidecar missing (integrity unverifiable) — "
                "refusing to resume"
            )
        expected = open(sidecar, "r", encoding="utf-8").read().strip()
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise FlowError(
                "checkpoint integrity check failed (SHA-256 mismatch) — "
                "refusing to resume from a corrupt file"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "gflownet-stage2-v1":
            raise FlowError(
                f"not a GFlowNetTrainer checkpoint (format={payload.get('format')!r})"
            )
        if payload["profile_digest"] != self.profile.digest():
            raise FlowError(
                "checkpoint profile digest does not match the active profile "
                "(frozen-profile incompatibility — do not mix profiles)"
            )
        self.policy.load_state_dict(payload["model"])
        self.policy.to(device=self.device, dtype=torch.float32)
        self.log_partition.load_state_dict(payload["log_partition"])
        self.log_partition.to(device=self.device, dtype=torch.float32)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.progress = _GFlowNetProgress(**payload["progress"])
        self.history = list(payload["history"])
        restore_rng_state(rng_state_from_dict(payload["rng"]))

    def snapshot_rng(self) -> None:
        """Refresh the checkpoint RNG snapshot (call after each rollout)."""
        self._rng_snapshot = capture_rng_state()
