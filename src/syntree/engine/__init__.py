"""The V2 engine layer: discrete assembly MDP, reward oracle, TB engine.

Public surface of the reconciled two-level GFlowNet framework (decision
D-V2-009; issues #24/#29/#30/#38/#42):

* :mod:`syntree.engine.environment` — the discrete Assembly MDP with
  deterministic bin-centre geometry and the terminal (STOP-gated)
  sub-bin refinement + scaffold-constrained MMFF94 reward oracle;
* :mod:`syntree.engine.rl` — the stationary multi-objective terminal
  reward ``R(x)``;
* :mod:`syntree.engine.gflownet` — the Trajectory Balance loss with the
  exact uniform-leaf backward policy;
* :mod:`syntree.engine.curriculum` — the #27 Approach C quantile
  curriculum sampler (9 levels / 6 stages / 0.03 floor).
"""

from syntree.engine.curriculum import (
    CurriculumError,
    CurriculumSampler,
    CURRICULUM_FEATURE_NAMES,
    CURRICULUM_NUM_LEVELS,
    CURRICULUM_NUM_STAGES,
    CURRICULUM_PROBABILITY_FLOOR,
    CURRICULUM_SPLIT,
)
from syntree.engine.environment import (
    AssemblyAction,
    AssemblyEnvironment,
    AssemblyError,
    BuildingBlock,
    FRAGMENT_PROP,
    HandleSpec,
    JunctionReaction,
    JunctionRecord,
    PocketDefinition,
    SEED_PROP,
    StepRecord,
    TORSION_BINS,
    TORSION_BIN_WIDTH,
    SUBBIN_HALF_WIDTH,
    backward_leaf_probability,
    bin_bounds,
    bin_center,
    finalize_terminal_molecule,
    get_removable_leaves,
    wrap_angle,
)
from syntree.engine.gflownet import (
    FlowError,
    TrajectoryBalanceEngine,
    TrajectoryStep,
    as_trajectory_steps,
)
from syntree.engine.rl import RewardError, ThreeDReward

__all__ = [
    "CurriculumError",
    "CurriculumSampler",
    "CURRICULUM_FEATURE_NAMES",
    "CURRICULUM_NUM_LEVELS",
    "CURRICULUM_NUM_STAGES",
    "CURRICULUM_PROBABILITY_FLOOR",
    "CURRICULUM_SPLIT",
    "AssemblyAction",
    "AssemblyEnvironment",
    "AssemblyError",
    "BuildingBlock",
    "FRAGMENT_PROP",
    "HandleSpec",
    "JunctionReaction",
    "JunctionRecord",
    "PocketDefinition",
    "SEED_PROP",
    "StepRecord",
    "TORSION_BINS",
    "TORSION_BIN_WIDTH",
    "SUBBIN_HALF_WIDTH",
    "backward_leaf_probability",
    "bin_bounds",
    "bin_center",
    "finalize_terminal_molecule",
    "get_removable_leaves",
    "wrap_angle",
    "FlowError",
    "TrajectoryBalanceEngine",
    "TrajectoryStep",
    "as_trajectory_steps",
    "RewardError",
    "ThreeDReward",
]
