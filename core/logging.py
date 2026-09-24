"""Formatted telemetry and measured attrition funnel reporter."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def print_report(self) -> None:
        report = f"""
======================================================================
                 MEASURED SCIENTIFIC DATA FUNNEL
======================================================================
RAW SOURCES
----------------------------------------------------------------------
Raw source records discovered:     {self.raw_source_records:,}

STRUCTURAL QC (Resolution <= 2.5Å, Pockets >= 100 atoms, Artifacts)
----------------------------------------------------------------------
Input:                             {self.structural_qc_input:,}
Passed:                            {self.structural_qc_passed:,}
Rejected:                          {self.structural_qc_rejected:,}

CHEMICAL QC (Ro5 Boundaries, Heavy Atoms >= 10, MW 150-800 Da)
----------------------------------------------------------------------
Input:                             {self.chemical_qc_input:,}
Passed:                            {self.chemical_qc_passed:,}
Rejected:                          {self.chemical_qc_rejected:,}

DEDUPLICATION (Sequence Clustering & Ligand Canonical InChIKey)
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
Total Output States:               {self.trajectory_states_total:,}
======================================================================
"""
        print(report)