"""V2 chemistry layer: production pocket physics and reaction machinery.

Public surface (issue #14 CHEM-04 landed first; #11/#12/#13/#15/#40/#41
follow per the dependency map):

* :mod:`syntree.chemistry.conformer` — the flat-bottom soft-core
  Lennard-Jones clash potential (the #42 torsion-refinement objective),
  atom typing, and the cavity-volume / heavy-atom floor machinery wired
  into the stationary reward.
"""

from __future__ import annotations

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
]
