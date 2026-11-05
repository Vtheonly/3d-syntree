#!/usr/bin/env python
"""#35 EVAL-02 — the comparative SBDD benchmark entry point.

Runs every provided method's exported structures through the **same**
evaluation code (``syntree/engine/comparative.py`` + the #34 evaluator)
on the **same frozen target manifest**, and emits
``comparative_benchmark.json`` plus the summary markdown tables with
bootstrap confidence intervals (the #35 acceptance artifacts)::

    python scripts/run_comparative_benchmark.py \
        --targets work/data/eval/frozen_targets.json \
        --method syntree=work/data/eval/syntree_structures \
        [--method targetdiff=...] [--method diffsbdd=...] \
        [--method synthemol=...] \
        [--retro-track stock_aligned=track_a_stock_aligned.yml] \
        [--retro-track independent_external=track_b_external_zinc.yml] \
        [--output comparative_benchmark.json] \
        [--markdown comparative_summary.md]

Per-method structure conventions (the external-baseline interface):

* each method provides a directory of generated structures, **one SDF
  per target named ``<target_id>.sdf``** (the manifest's identifiers);
* structures for targets a method did not generate are simply absent
  — counted as ``not_generated``, never fabricated;
* the baselines (TargetDiff, DiffSBDD, SyntheMol) are **external
  checkpoints with their own environments** — their structures must be
  exported by the operator; this harness only evaluates what is
  provided and reports absent methods as ``not_provided``.

The retrosynthesis tracks are the #44 two-track convention
(``stock_aligned`` and ``independent_external`` run separately; no
merged headline exists).  Paired differences are bootstrapped over the
**common targets** of each method pair — a fair comparison never mixes
target sets.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syntree.engine.comparative import (  # noqa: E402
    ComparativeError,
    comparative_report,
    load_frozen_targets,
    markdown_summary,
)
from syntree.engine.evaluator import (  # noqa: E402
    EvaluationError,
    load_generated_molecules,
    parse_aizynth_output,
    run_aizynthfinder,
)


def collect_method_structures(directory: str, target_ids: set[str]) -> list[str]:
    """Collect a method's per-target SDFs (absent targets stay absent)."""
    paths = []
    for tid in sorted(target_ids):
        for suffix in (".sdf", ".sdf.gz"):
            candidate = os.path.join(directory, tid + suffix)
            if os.path.exists(candidate):
                paths.append(candidate)
                break
    # stray files that do not match any target fail loudly (a typo'd
    # export must not silently inflate a method's corpus)
    allowed = {
        os.path.join(directory, tid + s)
        for tid in target_ids
        for s in (".sdf", ".sdf.gz")
    }
    present = set(glob.glob(os.path.join(directory, "*.sdf"))) | set(
        glob.glob(os.path.join(directory, "*.sdf.gz"))
    )
    stray = present - allowed
    if stray:
        raise EvaluationError(
            f"method directory {directory} contains structures not matching "
            f"any frozen target id: {sorted(stray)[:3]}"
        )
    return paths


def parse_method_specs(specs: list[str]) -> dict[str, str]:
    methods: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"error: method spec must be NAME=STRUCTURES_DIR, got {spec!r}"
            )
        name, directory = spec.split("=", 1)
        if not os.path.isdir(directory):
            raise SystemExit(f"error: method structures dir not found: {directory}")
        methods[name] = directory
    return methods


def _reusable(out_path: str, corpus_smiles: list[str]) -> bool:
    """True iff the planner artifact exists and covers exactly the corpus."""
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False
    try:
        existing = parse_aizynth_output(out_path)
    except (EvaluationError, json.JSONDecodeError, OSError):
        return False
    return set(existing) == set(corpus_smiles)


def _run_with_resume(
    smiles: list[str],
    config: str,
    output_path: str,
    python_exe: str,
    timeout_seconds: float,
    checkpoint_path: str | None,
    max_retries: int,
):
    """Run one track, resuming across timeouts via aizynthcli checkpoints."""
    extra: list[str] = []
    if checkpoint_path:
        extra = ["--checkpoint", checkpoint_path]
    attempts = 0
    while True:
        try:
            return run_aizynthfinder(
                smiles,
                config,
                output_path,
                python_exe=python_exe,
                timeout_seconds=timeout_seconds,
                extra_args=extra,
            )
        except EvaluationError as err:
            if not checkpoint_path or "timeout" not in str(err):
                raise
            if attempts >= max_retries:
                raise
            attempts += 1
            print(
                f"[bench] planner timeout; resuming from checkpoint "
                f"(attempt {attempts}/{max_retries})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#35 EVAL-02 comparative SBDD benchmark driver"
    )
    parser.add_argument(
        "--targets", required=True, help="the frozen target manifest JSON"
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="NAME=STRUCTURES_DIR (one SDF per <target_id>, repeatable)",
    )
    parser.add_argument(
        "--retro-track",
        action="append",
        default=[],
        help="NAME=AIZYNTH_CONFIG_YML (#44 track, repeatable)",
    )
    parser.add_argument(
        "--corpus-description",
        default=None,
        help="the honest corpus label (recorded verbatim)",
    )
    parser.add_argument(
        "--aizynth-python", default=sys.executable, help="planner interpreter"
    )
    parser.add_argument(
        "--planner-timeout", type=float, default=3600.0, help="per-track budget"
    )
    parser.add_argument(
        "--planner-checkpoint",
        action="store_true",
        help="resume long planner runs across interruptions (aizynthcli checkpoint)",
    )
    parser.add_argument(
        "--planner-retries", type=int, default=6, help="max resume attempts"
    )
    parser.add_argument(
        "--reuse-planner-output",
        action="store_true",
        help="reuse an existing planner output whose targets match (#45 §15)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="reference method name for paired comparisons (default: first)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="bootstrap RNG seed (deterministic)"
    )
    parser.add_argument(
        "--output", default="comparative_benchmark.json", help="output JSON"
    )
    parser.add_argument(
        "--markdown", default=None, help="summary markdown path (default: <output>.md)"
    )
    args = parser.parse_args()

    t0 = time.time()
    targets = load_frozen_targets(args.targets)
    target_ids = {t.target_id for t in targets}
    print(f"[bench] frozen targets: {len(targets)}")

    methods = parse_method_specs(args.method)
    method_corpora = {}
    for name, directory in methods.items():
        paths = collect_method_structures(directory, target_ids)
        if paths:
            corpus = load_generated_molecules(paths)
        else:
            corpus = []
        method_corpora[name] = corpus
        print(
            f"[bench] method {name!r}: {len(paths)} structures on the "
            f"frozen targets ({len(target_ids) - len(paths)} not generated)"
        )

    retro_tracks: dict[str, str] = {}
    for spec in args.retro_track:
        if "=" not in spec:
            raise SystemExit(f"error: retro track spec must be NAME=CONFIG: {spec!r}")
        name, config = spec.split("=", 1)
        if not os.path.exists(config):
            raise SystemExit(f"error: retro track config not found: {config}")
        retro_tracks[name] = config

    method_retro: dict[str, dict[str, dict]] = {name: {} for name in methods}
    if retro_tracks:
        output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        for method_name, corpus in method_corpora.items():
            valid = [m for m in corpus if m.valid]
            if not valid:
                continue
            # unique structures only: identical molecules need one search
            unique_smiles = list(dict.fromkeys(m.smiles for m in valid))
            for track_name, config in retro_tracks.items():
                out_path = os.path.join(
                    output_dir, f"aizynth_{method_name}_{track_name}.json"
                )
                if args.reuse_planner_output and _reusable(
                    out_path, unique_smiles
                ):
                    print(
                        f"[bench] retro[{track_name}] on {method_name!r}: "
                        f"reusing verified planner output {out_path}"
                    )
                    method_retro[method_name][track_name] = parse_aizynth_output(
                        out_path
                    )
                    continue
                print(
                    f"[bench] retro[{track_name}] on {method_name!r} "
                    f"({len(unique_smiles)} unique structures)"
                )
                method_retro[method_name][track_name] = _run_with_resume(
                    unique_smiles,
                    config,
                    out_path,
                    python_exe=args.aizynth_python,
                    timeout_seconds=args.planner_timeout,
                    checkpoint_path=(
                        os.path.join(
                            output_dir,
                            f"aizynth_{method_name}_{track_name}.checkpoint.jsonl",
                        )
                        if args.planner_checkpoint
                        else None
                    ),
                    max_retries=args.planner_retries,
                )

    description = args.corpus_description or (
        "exported generated structures evaluated on the frozen "
        f"{len(targets)}-target manifest; absent methods recorded as "
        "not_provided (#35 honesty contract)"
    )
    report = comparative_report(
        targets=targets,
        method_corpora=method_corpora,
        method_retro=method_retro,
        corpus_description=description,
        reference_method=args.reference,
        seed=args.seed,
    )
    report["provenance"] = {
        "wall_clock_s": round(time.time() - t0, 2),
        "retro_tracks": sorted(retro_tracks),
        "bootstrap_seed": args.seed,
    }
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    markdown_path = args.markdown or (args.output + ".md")
    with open(markdown_path, "w") as handle:
        handle.write(markdown_summary(report))
    print(f"[bench] report: {args.output}")
    print(f"[bench] markdown: {markdown_path}")
    for name, section in report["methods"].items():
        validity = section["validity"]
        print(
            f"[bench] {name}: validity {validity[0]:.3f} "
            f"[{validity[1]:.3f}, {validity[2]:.3f}] "
            f"({section['num_structures']} structures)"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationError, ComparativeError) as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(2)
