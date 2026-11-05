#!/usr/bin/env python3
"""benchmark_torsion_refinement.py — the #42 T4 execution-path benchmark.

Measures the terminal geometry-refinement reward oracle
(``AssemblyEnvironment.finalize_terminal_molecule``: bounded ±5° sub-bin
coarse-scan + bracketed Brent refinement of each rotatable junction
dihedral against the soft-core pocket potential, then scaffold-constrained
MMFF94 relaxation) on representative terminal states placed in REAL
pockets loaded from the #18 processed artifacts via the production
loaders (``train_stage1._load_complex``).

Chemistries: the canonical rotatable junction (rxn_07, alkyl bromide) —
the #42 refinement path — and the resonance-locked junction (rxn_01,
acid chloride) — the documented no-op contrast.  Torsion bins cycle
across the 36-bin space including the periodic boundary bins 1 and 36.

Records per state: selected bin, refined angle, objective before/after,
coarse + Brent evaluations, MMFF minimizations, wall time; aggregates
plus the runtime hardware baseline.  Evidence for #42's "benchmarked on
the supported T4 execution path" criterion (the authorized Colab T4
runtime; the refinement itself is CPU/RDKit — the benchmark measures
its cost profile on the deployment execution path).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402

from train_stage1 import _load_complex, _load_records  # noqa: E402
from tests.test_conformer_relaxation import (  # noqa: E402
    _ethyl_bromide,
    _propionyl_chloride,
    _seed_phenethylamine,
)
from syntree.engine.environment import (  # noqa: E402
    AssemblyAction,
    AssemblyEnvironment,
    PocketDefinition,
)

BIN_CYCLE = [1, 9, 18, 27, 36]  # includes the periodic boundary bins


def _capped_pocket(pocket: PocketDefinition, cap: int) -> PocketDefinition:
    """The production centroid-distance pocket cap (stage2's rule)."""
    if pocket.num_atoms <= cap:
        return pocket
    positions = np.asarray(pocket.positions, dtype=np.float64)
    centroid = positions.mean(axis=0)
    order = np.argsort(np.linalg.norm(positions - centroid, axis=1))
    keep = np.sort(order[:cap])
    return PocketDefinition(
        positions=positions[keep],
        atomic_numbers=np.asarray(pocket.atomic_numbers, dtype=np.int64)[keep],
        charges=np.asarray(pocket.charges, dtype=np.float64)[keep],
    )


def _build_terminal(pocket: PocketDefinition, chemistry: str, torsion_bin: int):
    if chemistry == "rxn_07":  # rotatable junction: the #42 refinement path
        synthons = {"etbr": _ethyl_bromide()}
        reaction, block = "rxn_07", "etbr"
    else:  # rxn_01: resonance-locked amide junction (the no-op contrast)
        synthons = {"propionyl": _propionyl_chloride()}
        reaction, block = "rxn_01", "propionyl"
    env = AssemblyEnvironment(
        _seed_phenethylamine(), synthons, pocket=pocket, max_blocks=4
    )
    env.reset()
    env.place_seed()
    handle = env.active_handles()[0]["handle_atom_idx"]
    env.step(AssemblyAction(handle, reaction, block, torsion_bin=torsion_bin))
    env.stop()
    return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trajectories", required=True,
                        help="directory holding trajectories_train.pt")
    parser.add_argument("--processed", required=True,
                        help="directory holding the processed complex artifacts")
    parser.add_argument("--pockets", type=int, default=24,
                        help="number of real pockets to benchmark")
    parser.add_argument("--max-pocket-atoms", type=int, default=700)
    parser.add_argument("--out", default="/content/logs/torsion_refinement_benchmark.json")
    args = parser.parse_args(argv)

    records = _load_records(args.trajectories, "train")[: args.pockets]
    print(f"[bench42] pockets: {len(records)}; chemistries: rxn_07 (rotatable) "
          f"+ rxn_01 (locked); bins cycle {BIN_CYCLE}")

    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        hardware["torch"] = torch.__version__
        hardware["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hardware["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        hardware["torch"] = "not installed"

    results, failures = [], []
    for index, record in enumerate(records):
        try:
            _ligand, pocket = _load_complex(args.processed, record)
            pocket = _capped_pocket(pocket, args.max_pocket_atoms)
            row = {
                "complex_id": record.get("complex_id", f"record_{index}"),
                "pocket_atoms": int(pocket.num_atoms),
            }
            for chemistry in ("rxn_07", "rxn_01"):
                torsion_bin = BIN_CYCLE[(index + (0 if chemistry == "rxn_07" else 2))
                                        % len(BIN_CYCLE)]
                env = _build_terminal(pocket, chemistry, torsion_bin)
                started = time.perf_counter()
                _relaxed, energy, diagnostics = env.finalize_terminal_molecule(
                    junction_torsion_stage=True,
                    pocket_vdw_relaxation=True,
                    return_diagnostics=True,
                )
                elapsed = time.perf_counter() - started
                junctions = diagnostics.get("junctions") or []
                row[chemistry] = {
                    "torsion_bin": torsion_bin,
                    "wall_seconds": round(elapsed, 4),
                    "energy": float(energy),
                    "junctions_refined": len(junctions),
                    "junctions": [
                        {
                            "selected_bin": entry.get("selected_bin"),
                            "initial_angle_rad": entry.get("initial_angle_rad"),
                            "refined_angle_rad": entry.get("refined_angle_rad"),
                            "bin_bounds_rad": entry.get("bin_bounds_rad"),
                            "objective_before": entry.get("objective_before"),
                            "objective_after": entry.get("objective_after"),
                            "coarse_evaluations": entry.get("coarse_evaluations"),
                            "brent_evaluations": entry.get("brent_evaluations"),
                        }
                        for entry in junctions
                    ],
                    "skipped_junctions": [
                        {k: v for k, v in entry.items() if k in
                         ("fragment_id", "selected_bin", "skipped")}
                        for entry in (diagnostics.get("junctions") or [])
                        if entry.get("skipped")
                    ],
                    "mmff_minimizations": env.stats.get("mmff_minimizations"),
                }
            results.append(row)
        except Exception as error:  # noqa: BLE001 — record and continue
            failures.append({"complex_id": record.get("complex_id", f"record_{index}"),
                             "error": repr(error)})
            continue

    if not results:
        print("[bench42] FAIL-CLOSED: no terminal state could be benchmarked",
              file=sys.stderr)
        for failure in failures:
            print("  ", failure, file=sys.stderr)
        return 2
    if len(failures) > len(records) // 2:
        print(f"[bench42] FAIL-CLOSED: {len(failures)}/{len(records)} pockets "
              "failed (majority)", file=sys.stderr)
        return 2

    rotatable = [row["rxn_07"] for row in results]
    refined = [j for row in rotatable for j in row["junctions"]]
    walls = [row["wall_seconds"] for row in rotatable]

    def stats(values):
        if not values:
            return None
        return {
            "count": len(values),
            "mean": round(statistics.fmean(values), 6),
            "median": round(statistics.median(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }

    objective_deltas = [
        float(j["objective_before"]) - float(j["objective_after"])
        for j in refined
        if j.get("objective_before") is not None and j.get("objective_after") is not None
    ]
    summary = {
        "pockets_benchmarked": len(results),
        "pockets_failed": len(failures),
        "rotatable_states": len(rotatable),
        "junctions_refined": len(refined),
        "brent_evaluations": stats([int(j["brent_evaluations"]) for j in refined
                                    if j.get("brent_evaluations") is not None]),
        "coarse_evaluations_per_junction": stats([int(j["coarse_evaluations"]) for j in refined
                                                  if j.get("coarse_evaluations") is not None]),
        "objective_improvement": stats(objective_deltas),
        "wall_seconds_per_terminal_state": stats(walls),
        "rotatable_wall_seconds": stats(walls),
        "locked_wall_seconds": stats([row["rxn_01"]["wall_seconds"] for row in results]),
        "mmff_minimizations_per_state": stats(
            [row["rxn_07"]["mmff_minimizations"] for row in results]),
        "hardware_baseline": hardware,
        "timing_source": "time.perf_counter",
    }

    report = {
        "issue": 42,
        "oracle": "AssemblyEnvironment.finalize_terminal_molecule "
                  "(sub-bin coarse-scan + bracketed Brent + constrained MMFF94)",
        "summary": summary,
        "results": results,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[bench42] junctions refined: {len(refined)} over {len(rotatable)} "
          f"rotatable terminal states ({len(results)} real pockets)")
    print(f"[bench42] wall/state: {summary['rotatable_wall_seconds']}")
    print(f"[bench42] objective improvement: {summary['objective_improvement']}")
    print(f"[bench42] Brent evals/junction: {summary['brent_evaluations']}")
    print(f"[bench42] report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
