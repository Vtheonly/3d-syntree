#!/usr/bin/env python
"""#34 EVAL-01 — the generated-molecule benchmark entry point.

The canonical evaluation CLI of the V2 evaluation stack
(``syntree/engine/evaluator.py`` is the library; this script is the
operator-facing driver)::

    python scripts/run_benchmarks.py \
        --sdf work/data/eval/generated --sdf runs/rollout.sdf \
        --corpus-description "forward-replay-verified reconstructed
        crystal ligands of the #18 census (NOT trained-policy samples)" \
        --retro-track independent_external=path/to/track_config.yml \
        [--retro-track stock_aligned=path/to/track_config.yml] \
        [--replay-manifest docs/data/trajectories/extraction_manifest_full_scale.json] \
        [--posebusters-config mol] \
        [--output evaluation_report.json]

What it does (the #34 acceptance surface):

1. parses the generated SDF corpus (directories are walked for
   ``*.sdf`` / ``*.sdf.gz``; multi-record files supported) with the
   fail-closed parser;
2. computes the physicochemical metric surface (validity, Fsp³
   distribution, molecular weight, fingerprint diversity);
3. runs **one isolated AiZynthFinder subprocess per retrosynthesis
   track** (``--retro-track NAME=CONFIG_YML``, the #44 two-track
   convention — pass one track for the pure #34 independent run, two
   for the full #44 report; the tracks are never merged);
4. optionally records the internal forward-replay manifest block
   (#44: the report must distinguish replay from planner
   retrosynthesis);
5. optionally runs the PoseBusters physical-validity backend over the
   corpus structures (``--posebusters-config mol``; protein-paired
   ``redock`` busting requires ``--posebusters-pairs JSON`` with
   ``{ligand, protein}`` entries);
6. writes ``evaluation_report.json`` — corpus, per-molecule records,
   per-track solve rates with the failure-category split
   (``no_route_found`` vs ``stock_mismatch``), and the tool/provenance
   block — and prints the honest summary.

Every retrospective claim in the printed summary is labelled with its
track name and stock provenance; nothing is merged (#44).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from syntree.engine.evaluator import (  # noqa: E402
    EVAL_SCHEMA_VERSION,
    EvaluationError,
    build_evaluation_report,
    load_generated_molecules,
    parse_aizynth_output,
    posebusters_backend,
    run_aizynthfinder,
)


def collect_sdf_paths(inputs: list[str]) -> list[str]:
    """Expand files/directories into a sorted SDF path list."""
    paths: list[str] = []
    for entry in inputs:
        if os.path.isdir(entry):
            for pattern in ("*.sdf", "*.sdf.gz"):
                paths.extend(
                    sorted(glob.glob(os.path.join(entry, pattern)))
                )
        else:
            paths.append(entry)
    if not paths:
        raise EvaluationError(f"no SDF inputs found under: {inputs}")
    return paths


def parse_track_specs(specs: list[str]) -> dict[str, str]:
    """Parse ``NAME=CONFIG`` track specifications (fail-closed)."""
    tracks: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise EvaluationError(
                f"retro track spec must be NAME=CONFIG_YML, got: {spec!r}"
            )
        name, config = spec.split("=", 1)
        if not name or not os.path.exists(config):
            raise EvaluationError(f"retro track config not found: {spec!r}")
        tracks[name] = config
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#34 EVAL-01 generated-molecule benchmark driver"
    )
    parser.add_argument(
        "--sdf",
        action="append",
        required=True,
        help="generated SDF file or directory (repeatable)",
    )
    parser.add_argument(
        "--corpus-description",
        required=True,
        help="the honest label of what the corpus IS (recorded verbatim)",
    )
    parser.add_argument(
        "--retro-track",
        action="append",
        default=[],
        help="NAME=AIZYNTH_CONFIG_YML retrosynthesis track (repeatable)",
    )
    parser.add_argument(
        "--aizynth-python",
        default=sys.executable,
        help="interpreter with aizynthfinder installed (isolation env)",
    )
    parser.add_argument(
        "--planner-timeout",
        type=float,
        default=600.0,
        help="per-track planner wall-clock budget (seconds)",
    )
    parser.add_argument(
        "--planner-checkpoint",
        action="store_true",
        help=(
            "resume long planner runs across interruptions via the "
            "aizynthcli --checkpoint protocol (the #43 session-safe "
            "philosophy applied to evaluation)"
        ),
    )
    parser.add_argument(
        "--planner-retries",
        type=int,
        default=6,
        help="max resume attempts per track when checkpointing is on",
    )
    parser.add_argument(
        "--reuse-planner-output",
        action="store_true",
        help=(
            "reuse an existing planner output whose target set matches "
            "the corpus (#45 §15 artifact-reuse); mismatched outputs are "
            "recomputed loudly, never silently accepted"
        ),
    )
    parser.add_argument(
        "--replay-manifest",
        default=None,
        help="the #18 extraction manifest JSON (forward-replay block)",
    )
    parser.add_argument(
        "--posebusters-config",
        default=None,
        choices=["mol"],
        help="run the PoseBusters ligand-geometry backend over the corpus",
    )
    parser.add_argument(
        "--posebusters-pairs",
        default=None,
        help="JSON list of {ligand, protein} pairs for redock busting",
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="output report path (evaluation_report.json by default)",
    )
    parser.add_argument(
        "--aizynth-artifacts-dir",
        default=None,
        help="directory holding planner outputs (recorded as provenance)",
    )
    args = parser.parse_args()

    t0 = time.time()
    sdf_paths = collect_sdf_paths(args.sdf)
    print(f"[eval] corpus inputs: {len(sdf_paths)} SDF artifacts")
    corpus = load_generated_molecules(sdf_paths)
    valid = [m for m in corpus if m.valid]
    # planner targets are UNIQUE canonical SMILES: identical molecules
    # need one search (the corpus multiplicity is recorded, not hidden —
    # the #18 census accepts the same crystal ligand for many pockets)
    unique_smiles = list(dict.fromkeys(m.smiles for m in valid))
    print(
        f"[eval] parsed {len(corpus)} records "
        f"({len(valid)} valid, {len(corpus) - len(valid)} rejected; "
        f"{len(unique_smiles)} unique structures)"
    )

    retro_tracks: dict[str, dict] = {}
    track_provenance: dict[str, dict] = {}
    tracks = parse_track_specs(args.retro_track)
    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    for name, config in tracks.items():
        out_path = os.path.join(output_dir, f"aizynth_{name}.json")
        corpus_smiles = unique_smiles
        if args.reuse_planner_output and _reusable(out_path, corpus_smiles):
            print(
                f"[eval] retrosynthesis track {name!r}: reusing verified "
                f"planner output {out_path} (targets match the corpus)"
            )
            results = parse_aizynth_output(out_path)
        else:
            print(
                f"[eval] retrosynthesis track {name!r}: planner -> {out_path}"
            )
            results = _run_track_with_resume(
                corpus_smiles,
                config,
                out_path,
                python_exe=args.aizynth_python,
                timeout_seconds=args.planner_timeout,
                checkpoint_path=(
                    os.path.join(
                        output_dir, f"aizynth_{name}.checkpoint.jsonl"
                    )
                    if args.planner_checkpoint
                    else None
                ),
                max_retries=args.planner_retries,
            )
        retro_tracks[name] = results
        with open(config) as handle:
            config_payload = handle.read()
        track_provenance[name] = {
            "planner": "AiZynthFinder (aizynthcli)",
            "planner_config": config_payload,
            "planner_output_artifact": out_path,
            "aizynthfinder_version": _aizynthfinder_version(
                args.aizynth_python
            ),
        }

    forward_replay = None
    if args.replay_manifest:
        with open(args.replay_manifest) as handle:
            manifest = json.load(handle)
        totals = manifest.get("totals", {})
        counts = manifest.get("counts", {})
        forward_replay = {
            "source": os.path.abspath(args.replay_manifest),
            "accepted_trajectories": totals.get("accepted"),
            "rejected": totals.get("failed"),
            "per_split": {
                k: v.get("records")
                for k, v in manifest.items()
                if k.startswith("trajectories_")
                and isinstance(v, dict)
            },
            "failure_vocabulary": counts,
            "replay_verified": True,
            "note": (
                "internal forward-reaction replay (#18) — distinct from "
                "planner retrosynthesis; the two are never conflated (#44)"
            ),
        }

    extra_metrics: dict[str, dict] = {}
    if args.posebusters_config:
        records = {}
        for molecule in valid:
            try:
                outcome = posebusters_backend(
                    molecule.source_path,
                    config_name=args.posebusters_config,
                )
            except EvaluationError as err:
                outcome = {"error": str(err)}
            records[molecule.molecule_id] = outcome
        passed = sum(
            1 for r in records.values() if r.get("passed") is True
        )
        extra_metrics["posebusters"] = {
            "config": args.posebusters_config,
            "num_records": len(records),
            "num_passed": passed,
            "pass_rate": (passed / len(records)) if records else 0.0,
            "records": records,
        }
    if args.posebusters_pairs:
        with open(args.posebusters_pairs) as handle:
            pairs = json.load(handle)
        records = {}
        for pair in pairs:
            ligand = pair["ligand"]
            reference = pair.get("reference")
            try:
                outcome = posebusters_backend(
                    ligand, reference, config_name="redock"
                )
            except EvaluationError as err:
                outcome = {"error": str(err)}
            records[ligand] = outcome
        extra_metrics["posebusters_redock"] = {"records": records}

    report = build_evaluation_report(
        corpus=corpus,
        retro_tracks=retro_tracks,
        track_provenance=track_provenance,
        corpus_description=args.corpus_description,
        forward_replay_status=forward_replay,
        extra_metrics=extra_metrics or None,
    )
    report["provenance"] = {
        "schema": EVAL_SCHEMA_VERSION,
        "rdkit_version": _rdkit_version(),
        "generator_note": (
            "3D-SynTree V2 evaluation stack (#34/#44/#35); the "
            "retrosynthesis planner and its stocks are external to the "
            "generator's knowledge"
        ),
        "wall_clock_s": round(time.time() - t0, 2),
    }
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[eval] report written: {args.output}")

    # The honest printed summary — never a merged headline number.
    print(f"[eval] corpus: {report['corpus']['description']}")
    print(
        f"[eval] validity: {report['corpus']['validity_rate']:.4f} "
        f"({report['corpus']['num_valid']}/{report['corpus']['num_records']})"
    )
    print(
        f"[eval] mean Fsp3: {report['physicochemical']['fraction_sp3']['mean']:.4f}"
    )
    print(
        f"[eval] diversity: "
        f"{report['physicochemical']['fingerprint_diversity']:.4f}"
    )
    for name, section in report["retrosynthesis"].items():
        print(
            f"[eval] retro[{name}] solve_rate="
            f"{section['solve_rate']:.4f} "
            f"({section['num_solved']}/{section['num_targets']}) "
            f"stock_mismatch={section['num_stock_mismatch']} "
            f"no_route_found={section['num_no_route_found']}"
        )
    return 0


def _reusable(out_path: str, corpus_smiles: list[str]) -> bool:
    """True iff the planner artifact exists and covers exactly the corpus."""
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False
    try:
        existing = parse_aizynth_output(out_path)
    except (EvaluationError, json.JSONDecodeError, OSError):
        return False
    return set(existing) == set(corpus_smiles)


def _run_track_with_resume(
    smiles: list[str],
    config: str,
    output_path: str,
    python_exe: str,
    timeout_seconds: float,
    checkpoint_path: str | None,
    max_retries: int,
):
    """Run one track, resuming across timeouts via aizynthcli checkpoints.

    With ``--planner-checkpoint``, a timed-out attempt is not a failure:
    the checkpoint records every processed target, and the retry skips
    them — bounded foreground calls with resumable drivers (the same
    contract as the #43 session-safe training resume).  Without
    checkpointing, a timeout propagates as a loud error.
    """
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
                f"[eval] planner timeout; resuming from checkpoint "
                f"(attempt {attempts}/{max_retries})"
            )


def _aizynthfinder_version(python_exe: str) -> str:
    try:
        proc = __import__("subprocess").run(
            [python_exe, "-c", "import aizynthfinder; print(getattr(aizynthfinder, '__version__', 'unknown'))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — provenance must never kill the run
        return "unknown"


def _rdkit_version() -> str:
    try:
        from rdkit import rdBase

        return rdBase.rdkitVersion
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as err:
        print(f"error: {err}", file=sys.stderr)
        raise SystemExit(2)
