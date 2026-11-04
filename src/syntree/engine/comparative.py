"""#35 EVAL-02 — the comparative SBDD benchmark library.

The comparative harness of V2.  Issue #35's scientific claim — "3D-SynTree
breaks the synthesizability wall" — is only supportable **head-to-head on
identical test targets**: the same frozen target manifest, the same
evaluation code, the same stocks, the same planner.  This module provides
that machinery:

* :func:`load_frozen_targets` — the frozen target manifest loader
  (one entry per held-out pocket; strict schema, fail-closed on
  duplicates);
* :func:`bootstrap_ci` / :func:`bootstrap_rate_ci` — deterministic
  (seeded) percentile bootstrap confidence intervals for any per-target
  metric; the resample is over **targets**, the correct unit of
  replication for a benchmark on distinct pockets;
* :func:`paired_difference_ci` — the paired bootstrap over the
  **common** target set of two methods (a fair comparison never mixes
  target sets; an empty intersection fails loudly instead of silently
  degrading to unpaired);
* :func:`method_section` / :func:`comparative_report` — the per-method
  sections (validity, Fsp³, diversity, per-track retrosynthesis rates
  with the #44 stock-mismatch split) and the pairwise paired-difference
  tables — every method evaluated by the *same* code path (structural
  identity, not a promise);
* :func:`markdown_summary` — the summary tables with CIs (the #35
  acceptance artifact).

Honesty guards (mirroring #45 §14): the report records which methods
were actually provided; an empty method corpus is flagged
``not_provided`` rather than fabricated; and the two retrosynthesis
tracks (#44) are rendered as **separate metrics** — never a merged
headline solve rate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from syntree.engine.evaluator import (
    EvaluatedMolecule,
    RetroSearchResult,
    fingerprint_diversity,
    molecular_metrics,
)

#: Allowed top-level keys of a frozen target manifest entry.
TARGET_FIELDS = frozenset(
    {"target_id", "pocket", "reference_ligand_smiles", "notes"}
)


class ComparativeError(ValueError):
    """Raised on ill-defined comparative inputs (duplicate targets,
    empty method intersections, malformed manifests)."""


@dataclass(frozen=True)
class FrozenTarget:
    """One held-out benchmark target.

    Attributes:
        target_id: the stable target identifier (e.g. ``pdbbind_4itp``).
        pocket: absolute path to the pocket structure (PDB artifact).
        reference_ligand_smiles: the co-crystallised ligand (reference).
        notes: optional provenance notes.
    """

    target_id: str
    pocket: str
    reference_ligand_smiles: str
    notes: Optional[str] = None


def load_frozen_targets(path: str) -> List[FrozenTarget]:
    """Load and validate the frozen target manifest (fail-closed).

    The manifest is a JSON object ``{"split": ..., "targets": [...]}``;
    every entry must carry ``target_id``, ``pocket`` and
    ``reference_ligand_smiles``.  Duplicate target ids and pocket paths
    that do not exist raise :class:`ComparativeError`.
    """
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "targets" not in payload:
        raise ComparativeError(
            f"frozen target manifest must be a dict with a 'targets' list: {path}"
        )
    targets: List[FrozenTarget] = []
    seen: set[str] = set()
    for entry in payload["targets"]:
        missing = {"target_id", "pocket", "reference_ligand_smiles"} - set(entry)
        if missing:
            raise ComparativeError(
                f"frozen target entry missing fields {sorted(missing)}: {entry}"
            )
        if entry["target_id"] in seen:
            raise ComparativeError(
                f"duplicate target_id in manifest: {entry['target_id']}"
            )
        seen.add(entry["target_id"])
        pocket = entry["pocket"]
        if pocket and not os.path.exists(pocket):
            raise ComparativeError(f"target pocket not found: {pocket}")
        targets.append(
            FrozenTarget(
                target_id=entry["target_id"],
                pocket=pocket,
                reference_ligand_smiles=entry["reference_ligand_smiles"],
                notes=entry.get("notes"),
            )
        )
    if not targets:
        raise ComparativeError(f"frozen target manifest has no targets: {path}")
    return targets


# ---------------------------------------------------------------------------
# Deterministic bootstrap machinery
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of per-target values.

    Deterministic: the RNG is seeded from ``seed`` so the reported
    interval is exactly reproducible.  Empty input raises — a CI over
    nothing is not a statistic (#45 §14).
    """
    if len(values) == 0:
        raise ComparativeError("bootstrap over an empty value set")
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(arr.mean()), float(lo), float(hi)


def bootstrap_rate_ci(
    booleans: Sequence[bool],
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a rate (solve rate, validity rate).

    The unit of resampling is the target, so a rate of ``k/n`` carries
    an interval that honestly reflects the sample size.
    """
    if len(booleans) == 0:
        raise ComparativeError("bootstrap rate over an empty set")
    arr = np.asarray(booleans, dtype=float)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    rates = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(rates, [alpha, 1.0 - alpha])
    return float(arr.mean()), float(lo), float(hi)


def paired_difference_ci(
    values_a: Dict[str, float],
    values_b: Dict[str, float],
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired bootstrap CI of ``A - B`` over the **common** targets.

    The comparison is paired by ``target_id``; targets present in only
    one method are excluded from the difference (and reported as such —
    an unpaired silent degradation is forbidden).  An empty
    intersection raises :class:`ComparativeError`.
    """
    common = sorted(set(values_a) & set(values_b))
    if not common:
        raise ComparativeError(
            "paired comparison over an empty target intersection"
        )
    a = np.array([values_a[t] for t in common])
    b = np.array([values_b[t] for t in common])
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(common)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "metric": "A - B (paired, common targets)",
        "num_common_targets": n,
        "only_in_a": sorted(set(values_a) - set(values_b)),
        "only_in_b": sorted(set(values_b) - set(values_a)),
        "mean_difference": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "level": level,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Per-method evaluation (the single identical code path)
# ---------------------------------------------------------------------------


def method_section(
    method_name: str,
    corpus: List[EvaluatedMolecule],
    retro_tracks: Dict[str, Dict[str, RetroSearchResult]],
    target_ids: Optional[Dict[str, str]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate one method's corpus with the shared metric code.

    ``target_ids`` maps each corpus molecule's SMILES to its target id
    when structures are named per target (the per-target vectors used
    by the paired machinery).  The retrosynthesis rates are computed
    **per #44 track** — separate entries, never merged.  An empty
    corpus is flagged ``not_provided`` (an absent method is recorded,
    never fabricated).
    """
    valid = [m for m in corpus if m.valid]
    metrics = [molecular_metrics(m.smiles) for m in valid]
    section: Dict[str, Any] = {
        "method": method_name,
        "num_structures": len(corpus),
        "num_valid": len(valid),
        "not_provided": len(corpus) == 0,
        "validity": bootstrap_rate_ci(
            [m.valid for m in corpus], seed=seed
        ) if corpus else (0.0, 0.0, 0.0),
        "fraction_sp3": bootstrap_ci(
            [m["fraction_sp3"] for m in metrics], seed=seed
        ) if metrics else (0.0, 0.0, 0.0),
        "fingerprint_diversity": fingerprint_diversity(
            [m.smiles for m in valid]
        ),
        "retrosynthesis": {},
    }
    if target_ids:
        per_target: Dict[str, Dict[str, float]] = {}
        for track_name, results in retro_tracks.items():
            solved_flags: Dict[str, float] = {}
            for target, result in results.items():
                tid = target_ids.get(target, target)
                solved_flags[tid] = 1.0 if result.is_solved else 0.0
            per_target[track_name] = solved_flags
        section["per_target_retro"] = per_target
    for track_name, results in retro_tracks.items():
        section["retrosynthesis"][track_name] = bootstrap_rate_ci(
            [r.is_solved for r in results.values()], seed=seed
        ) if results else (0.0, 0.0, 0.0)
    return section


def comparative_report(
    targets: List[FrozenTarget],
    method_corpora: Dict[str, List[EvaluatedMolecule]],
    method_retro: Dict[str, Dict[str, Dict[str, RetroSearchResult]]],
    corpus_description: str,
    reference_method: Optional[str] = None,
    level: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """Assemble the ``comparative_benchmark.json`` payload.

    Every method is evaluated through :func:`method_section` — the
    identical code path (#35's fairness requirement is structural).
    Pairwise paired-difference tables are computed against the
    reference method (default: the first) for each retrosynthesis
    track.  Methods with no structures are recorded as
    ``not_provided`` — never silently dropped or fabricated.
    """
    if not method_corpora:
        raise ComparativeError("no method corpora provided")
    target_ids = {t.target_id for t in targets}
    sections: Dict[str, Any] = {}
    for name, corpus in method_corpora.items():
        id_map: Dict[str, str] = {}
        for molecule in corpus:
            stem = os.path.basename(molecule.source_path)
            tid = stem.replace(".sdf", "").replace(".gz", "")
            if tid in target_ids:
                id_map[molecule.smiles] = tid
        sections[name] = method_section(
            name,
            corpus,
            method_retro.get(name, {}),
            target_ids=id_map,
            seed=seed,
        )
    reference = reference_method or next(iter(sections))
    pairs: List[Dict[str, Any]] = []
    retro_tracks = sorted(
        {t for s in sections.values() for t in s.get("retrosynthesis", {})}
    )
    for name in sections:
        if name == reference or sections[name].get("not_provided"):
            continue
        if sections[reference].get("not_provided"):
            continue
        pairs.append(
            _pair(
                reference,
                name,
                sections,
                retro_tracks,
                level=level,
                seed=seed,
            )
        )
    return {
        "schema": "comparative-v2.0",
        "corpus_description": corpus_description,
        "frozen_targets": {
            "num_targets": len(targets),
            "target_ids": [t.target_id for t in targets],
        },
        "reference_method": reference,
        "methods": sections,
        "paired_comparisons": pairs,
        "notes": (
            "retrosynthesis rates are per-track (#44): "
            "stock-aligned and independent external-stock are separate "
            "metrics and are never merged into a headline number"
        ),
    }


def _pair(
    reference: str,
    name: str,
    sections: Dict[str, Any],
    retro_tracks: Sequence[str],
    level: float,
    seed: int,
) -> Dict[str, Any]:
    ref, other = sections[reference], sections[name]

    def per_target(section: Dict[str, Any], track: str) -> Dict[str, float]:
        return dict(section.get("per_target_retro", {}).get(track, {}))

    comparisons: Dict[str, Any] = {
        "reference": reference,
        "method": name,
        "level": level,
        "seed": seed,
        "metrics": {},
    }
    for track in retro_tracks:
        try:
            comparisons["metrics"][f"retro_{track}"] = paired_difference_ci(
                per_target(ref, track), per_target(other, track),
                level=level, seed=seed,
            )
        except ComparativeError as err:
            comparisons["metrics"][f"retro_{track}"] = {"error": str(err)}
    return comparisons


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _fmt_ci(triple: Tuple[float, float, float]) -> str:
    mean, lo, hi = triple
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def markdown_summary(report: Dict[str, Any]) -> str:
    """Render the comparative benchmark summary tables (with CIs)."""
    lines: List[str] = []
    lines.append("# Comparative SBDD benchmark summary")
    lines.append("")
    lines.append(f"Corpus: {report['corpus_description']}")
    lines.append(
        f"Frozen targets: {report['frozen_targets']['num_targets']} "
        f"(reference method: `{report['reference_method']}`)"
    )
    lines.append("")
    tracks = _track_names(report)
    header = (
        "| method | structures | validity | Fsp3 | diversity "
        + "".join(f" | retro[{t}] " for t in tracks)
        + "|"
    )
    lines.append(header)
    lines.append("|" + "---|" * (5 + len(tracks)))
    for name, section in report["methods"].items():
        cells = [
            name,
            str(section.get("num_structures", 0)),
            _fmt_ci(section["validity"]),
            _fmt_ci(section["fraction_sp3"]),
            f"{section['fingerprint_diversity']:.3f}",
        ]
        for track in tracks:
            cells.append(_fmt_ci(section["retrosynthesis"][track]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if report["paired_comparisons"]:
        lines.append("## Paired differences (reference − method, 95% CI)")
        lines.append("")
        lines.append("| comparison | metric | Δ | CI | common targets |")
        lines.append("|---|---|---|---|---|")
        for pair in report["paired_comparisons"]:
            for metric, stats in pair["metrics"].items():
                if "error" in stats:
                    lines.append(
                        f"| {pair['reference']} vs {pair['method']} "
                        f"| {metric} | — | {stats['error']} | 0 |"
                    )
                    continue
                lines.append(
                    f"| {pair['reference']} vs {pair['method']} "
                    f"| {metric} | {stats['mean_difference']:+.3f} "
                    f"| [{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}] "
                    f"| {stats['num_common_targets']} |"
                )
        lines.append("")
    lines.append(
        "> Retrosynthesis tracks are reported separately (#44); no merged "
        "headline solve rate exists."
    )
    return "\n".join(lines) + "\n"


def _track_names(report: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for section in report["methods"].values():
        for track in section.get("retrosynthesis", {}):
            if track not in names:
                names.append(track)
    return names


__all__ = [
    "TARGET_FIELDS",
    "ComparativeError",
    "FrozenTarget",
    "load_frozen_targets",
    "bootstrap_ci",
    "bootstrap_rate_ci",
    "paired_difference_ci",
    "method_section",
    "comparative_report",
    "markdown_summary",
]
