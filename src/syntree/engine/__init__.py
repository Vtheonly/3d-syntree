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
  curriculum sampler (9 levels / 6 stages / 0.03 floor);
* :mod:`syntree.engine.comparative` — the #35 comparative benchmark
  harness: frozen target manifests, deterministic bootstrap CIs,
  paired-over-common-targets comparisons and the markdown summaries;
* :mod:`syntree.engine.evaluator` — the #34/#44/#35 evaluation stack:
  the fail-closed generated-SDF corpus parser, the isolated
  AiZynthFinder retrosynthesis boundary, the stock-mismatch failure
  vocabulary, the optional PoseBusters/Vina adapters and the
  ``evaluation_report.json`` assembly.
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
from syntree.engine.comparative import (
    ComparativeError,
    FrozenTarget,
    bootstrap_ci,
    bootstrap_rate_ci,
    comparative_report,
    load_frozen_targets,
    markdown_summary,
    method_section,
    paired_difference_ci,
)
from syntree.engine.evaluator import (
    EVAL_SCHEMA_VERSION,
    REJECTION_CODES,
    RETRO_FAILURE_CATEGORIES,
    EvaluationError,
    EvaluatedMolecule,
    RetroSearchResult,
    build_evaluation_report,
    fingerprint_diversity,
    load_generated_molecules,
    molecular_metrics,
    parse_aizynth_output,
    run_aizynthfinder,
)

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
    "EVAL_SCHEMA_VERSION",
    "REJECTION_CODES",
    "RETRO_FAILURE_CATEGORIES",
    "EvaluationError",
    "EvaluatedMolecule",
    "RetroSearchResult",
    "build_evaluation_report",
    "fingerprint_diversity",
    "load_generated_molecules",
    "molecular_metrics",
    "parse_aizynth_output",
    "run_aizynthfinder",
    "ComparativeError",
    "FrozenTarget",
    "bootstrap_ci",
    "bootstrap_rate_ci",
    "comparative_report",
    "load_frozen_targets",
    "markdown_summary",
    "method_section",
    "paired_difference_ci",
]
