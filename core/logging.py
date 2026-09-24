"""Formatted telemetry and measured attrition funnel reporter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class MeasuredDataFunnel:
    """Exact empirical counts measured at every physical filtering stage."""

    raw_source_records: int = 0
    structural_qc_input: int = 0
    structural_qc_passed: int = 0
    structural_qc_rejected: int = 0
    chemical_qc_input: int = 0
    chemical_qc_passed: int = 0
    chemical_qc_rejected: int = 0
    deduplication_before: int = 0
    deduplication_after: int = 0
    retrosynthesis_candidates: int = 0
    retrosynthesis_decomposed: int = 0
    retrosynthesis_failed: int = 0
    valid_complexes: int = 0
    trajectory_states_total: int = 0
    train_states: int = 0
    val_states: int = 0
    test_states: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "MeasuredDataFunnel":
        known = {k: int(v) for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def merge(self, previous: "MeasuredDataFunnel") -> None:
        """Add counts from a previous interrupted run (resume bookkeeping).

        Stage totals are cumulative counters, so a fresh run's counts are
        added on top of the persisted ones. Split/state totals and the
        dedup + trajectory counters are recomputed from manifests by the
        caller, so they are overwritten rather than summed.
        """
        cumulative = (
            "raw_source_records",
            "structural_qc_input",
            "structural_qc_passed",
            "structural_qc_rejected",
            "chemical_qc_input",
            "chemical_qc_passed",
            "chemical_qc_rejected",
            "deduplication_before",
            "deduplication_after",
            "retrosynthesis_candidates",
            "retrosynthesis_decomposed",
            "retrosynthesis_failed",
            "valid_complexes",
        )
        for field in cumulative:
            setattr(self, field, getattr(self, field) + getattr(previous, field))

    def print_report(self) -> None:
        report = f"""
======================================================================
                 MEASURED SCIENTIFIC DATA FUNNEL
======================================================================
RAW SOURCES
----------------------------------------------------------------------
Raw source records discovered:     {self.raw_source_records:,}

STRUCTURAL QC (pocket >= 100 atoms after 10A ligand-centroid re-trim,
               hetero/artifact removal, readable PDB + 3D ligand)
----------------------------------------------------------------------
Input:                             {self.structural_qc_input:,}
Passed:                            {self.structural_qc_passed:,}
Rejected:                          {self.structural_qc_rejected:,}

CHEMICAL QC (Ro5 Boundaries, Heavy Atoms >= 10, MW 150-800 Da,
             crystallization artifacts, metals, missing conformers)
----------------------------------------------------------------------
Input:                             {self.chemical_qc_input:,}
Passed:                            {self.chemical_qc_passed:,}
Rejected:                          {self.chemical_qc_rejected:,}

DEDUPLICATION (protein sequence identity clusters -> locked splits)
----------------------------------------------------------------------
Before:                            {self.deduplication_before:,}
After:                             {self.deduplication_after:,}
Redundancy Removed:                {self.deduplication_before - self.deduplication_after:,}

RETROSYNTHETIC DECOMPOSITION (10 Certified SMARTS Reaction Classes)
----------------------------------------------------------------------
Candidates Attempted:              {self.retrosynthesis_candidates:,}
Successfully Decomposed:           {self.retrosynthesis_decomposed:,}
Failed Decomposition / Unmatched:  {self.retrosynthesis_failed:,}
Acceptance Yield:                  {self.retrosynthesis_decomposed / max(1, self.retrosynthesis_candidates):.2%}

TRAJECTORY STATES
----------------------------------------------------------------------
Final Valid Complexes:             {self.valid_complexes:,}
Total Supervised States (S_t):     {self.trajectory_states_total:,}

FINAL DATASET SPLITS (Leakage-Safe 80/10/10 Cluster Partitioning)
----------------------------------------------------------------------
Train States:                      {self.train_states:,}
Validation States:                 {self.val_states:,}
Test States:                       {self.test_states:,}
Total Output States:               {self.train_states + self.val_states + self.test_states:,}
======================================================================
"""
        print(report)
