"""V2 chemistry layer: production pocket physics and reaction machinery.

Public surface (issue #14 CHEM-04 and #11 CHEM-01 landed; #12/#13/#15/#40/#41
follow per the dependency map):

* :mod:`syntree.chemistry.conformer` — the flat-bottom soft-core
  Lennard-Jones clash potential (the #42 torsion-refinement objective),
  atom typing, and the cavity-volume / heavy-atom floor machinery wired
  into the stationary reward.
* :mod:`syntree.chemistry.reactions` — the certified 32-reaction taxonomy
  with its 9-family coarse mapping, standardized reagent/product
  certification pairs, deterministic deprotections, and the engine-format
  registry builder (#11 CHEM-01, D-V2-016).
* :mod:`syntree.chemistry.chemoselectivity` — the reaction-aware
  chemoselectivity and competing-handle screen for synthon curation
  (#40 CHEM-06): the free-group vocabulary with orthogonal protection
  honoured by construction, the per-class exclusion matrix (self-reactive
  pairs, organometallic quenching, competing handles/electrophiles/SNAr
  positions, catalyst poisons), the differentiated 2,4-dihalo-1,3-azine
  exemption, and the catalog manifest with per-rule impact reporting.
"""

from __future__ import annotations

from syntree.chemistry.chemoselectivity import (
    ALL_HANDLE_CLASSES,
    ChemoscreenResult,
    CurationManifest,
    ChemoselectivityError,
    chemoselectivity_rule_report,
    detect_free_groups,
    detect_handle_instances,
    handle_class_of_atom,
    screen_block,
    screen_catalog,
)
from syntree.chemistry.conformer import (
    CLASH_SPIKE_CEILING,
    DEFAULT_ATOM_TYPE,
    FLAT_BOTTOM_DELTA,
    HEAVY_ATOM_FLOOR,
    HEAVY_ATOM_FLOOR_WEIGHT,
    LJ_WELL_DEPTHS,
    VDW_RADII,
    VOLUME_RATIO_FLOOR,
    ChemistryError,
    SoftCoreClashPotential,
    atom_type,
    flat_bottom_soft_core_lj,
    flat_bottom_soft_core_lj_gradient,
    flat_bottom_soft_core_lj_torch,
    heavy_atom_floor_penalty,
    pair_epsilon,
    pocket_cavity_volume,
    soft_core_clash_energy,
    volume_fill_ratio,
    volume_floor_penalty,
)
from syntree.chemistry.reactions import (
    CERTIFIED_REACTIONS,
    DEPROTECTIONS,
    POLICY_REACTIONS,
    REACTION_CATEGORIES,
    REACTION_FAMILIES,
    ReactionRegistryError,
    apply_all_deprotections,
    apply_deprotection,
    build_junction_reaction_registry,
    certify_reaction,
    certify_registry,
)

__all__ = [
    "CLASH_SPIKE_CEILING",
    "DEFAULT_ATOM_TYPE",
    "FLAT_BOTTOM_DELTA",
    "HEAVY_ATOM_FLOOR",
    "HEAVY_ATOM_FLOOR_WEIGHT",
    "LJ_WELL_DEPTHS",
    "VDW_RADII",
    "VOLUME_RATIO_FLOOR",
    "ChemistryError",
    "SoftCoreClashPotential",
    "atom_type",
    "flat_bottom_soft_core_lj",
    "flat_bottom_soft_core_lj_gradient",
    "flat_bottom_soft_core_lj_torch",
    "heavy_atom_floor_penalty",
    "pair_epsilon",
    "pocket_cavity_volume",
    "soft_core_clash_energy",
    "volume_fill_ratio",
    "volume_floor_penalty",
    "CERTIFIED_REACTIONS",
    "DEPROTECTIONS",
    "POLICY_REACTIONS",
    "REACTION_CATEGORIES",
    "REACTION_FAMILIES",
    "ReactionRegistryError",
    "apply_all_deprotections",
    "apply_deprotection",
    "build_junction_reaction_registry",
    "certify_reaction",
    "certify_registry",
]
