"""9-level difficulty-aware curriculum for Stage-1 training (Approach C).

Implemented per docs/CURRICULUM.md (see that document for the full design,
rationale, and verification protocol). Summary of the architecture:

1. DIFFICULTY SCORING - every training example gets a reproducible [0, 1]
   difficulty score from a fixed, versioned feature schema:

   * tensor features (backend-agnostic, read straight off the PyG ``Data``
     sample): ligand core atom count, pocket atom count (steric complexity),
     mean absolute pocket/ligand formal charge;
   * catalog features (RDKit, computed once per synthon id and cached): the
     target synthon's atom count, ring count, rotatable-bond count,
     heteroatom count, and symmetry ratio (fraction of unique canonical
     atom-rank classes);
   * rarity features (computed from the TRAIN split only): reaction-family
     and core-handle scarcity ``-log2(p)``.

   Every feature is robustly min-max normalised with the 1st/99th percentile
   of the TRAIN split, then combined via ``curriculum.feature_weights``.

2. LEVEL ASSIGNMENT - the train split is cut into ``n_levels`` (default 9)
   approximately-balanced quantile bins (L1 = easiest ... L9 = hardest).
   Bin edges come from the TRAIN split only; validation samples are assigned
   levels with those edges for per-level reporting without ever influencing
   the binning.  The existing train/val split itself is never modified by
   this module (leakage rule: curriculum touches only *sampling inside the
   train split*).

3. CURRICULUM SAMPLING (Approach C - difficulty-aware MIXED sampling) -
   a torch ``Sampler`` draws ``len(dataset)`` indices per epoch (identical
   steps/epoch to the plain shuffle path).  Stage ``s`` centers the sampling
   mass on difficulty level ``center(s)`` with a Gaussian spread (``sigma``)
   blended with a uniform floor (``min_weight``): easy levels are favoured
   early, hard levels dominate late, and NO level is ever fully removed -
   the hardest examples are reachable from epoch 0 and the easiest are never
   dropped at the end.  Stage advancement is milestone-driven (validation
   metric improvement threshold) with a patience-based fallback so the run
   can never stall inside a stage; it is never a fixed-epoch schedule.

4. VISIBILITY - per-level train/val counts, the active stage, the level
   weights, and the sampled per-level counts are printed at startup and once
   per epoch, and stored in the per-epoch metrics entry (metrics.jsonl).

Determinism: the sampler is seeded per epoch (``seed`` + epoch), independent
of the global RNGs, so model initialisation and data augmentation are
untouched.  The full feature table, bin edges, and level assignment are
cached to a JSON artifact keyed by a signature of (schema, config, dataset
sizes, catalog ids hash); rebuilds are automatic when any input changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

FEATURE_SCHEMA_VERSION = 1

# Ordered feature names of the difficulty schema (v1).  Cached artifacts are
# invalidated when this list or the schema version changes.
FEATURE_NAMES = (
    "ligand_n_atoms",
    "pocket_n_atoms",
    "pocket_mean_abs_charge",
    "ligand_mean_abs_charge",
    "synthon_n_atoms",
    "synthon_n_rings",
    "synthon_n_rotatable",
    "synthon_n_hetero",
    "synthon_symmetry_ratio",
    "reaction_family_rarity",
    "core_handle_rarity",
)

DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "ligand_n_atoms": 1.0,
    "pocket_n_atoms": 1.2,
    "pocket_mean_abs_charge": 0.6,
    "ligand_mean_abs_charge": 0.4,
    "synthon_n_atoms": 0.8,
    "synthon_n_rings": 1.0,
    "synthon_n_rotatable": 1.2,
    "synthon_n_hetero": 0.6,
    "synthon_symmetry_ratio": 0.4,
    "reaction_family_rarity": 0.8,
    "core_handle_rarity": 0.5,
}


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
def _first_present(data, names: Sequence[str]):
    for name in names:
        value = getattr(data, name, None)
        if value is not None:
            return value
    return None


def tensor_sample_features(data) -> Dict[str, float]:
    """Backend-agnostic structural features read directly from a PyG sample.

    Works for every data backend (local CrossDocked, Hugging Face shards,
    kaggle_offline, trajectory_pt, synthetic smoke) because it only touches
    fields every backend guarantees.  All values are plain Python floats so
    they are JSON-serialisable for the cache artifact.
    """
    pocket_pos = _first_present(data, ("pocket_pos", "pos"))
    pocket_z = _first_present(data, ("pocket_z", "z"))
    pocket_charge = getattr(data, "pocket_charge", None)
    ligand_z = _first_present(data, ("ligand_z",))
    ligand_charge = getattr(data, "ligand_charge", None)

    n_pocket = int(pocket_pos.size(0)) if pocket_pos is not None else (
        int(pocket_z.size(0)) if pocket_z is not None else 0
    )
    n_ligand = int(ligand_z.size(0)) if ligand_z is not None else 0

    def _mean_abs(t) -> float:
        if t is None or t.numel() == 0:
            return 0.0
        return float(t.detach().float().abs().mean().item())

    feats = {
        "ligand_n_atoms": float(n_ligand),
        "pocket_n_atoms": float(n_pocket),
        "pocket_mean_abs_charge": _mean_abs(pocket_charge),
        "ligand_mean_abs_charge": _mean_abs(ligand_charge),
    }
    # Target fields are used for the rarity features and synthon lookup.
    for key, names in (
        ("target_synthon", ("target_synthon",)),
        ("target_reaction_family_idx", ("target_reaction_family_idx",)),
        ("target_core_handle_idx", ("target_core_handle_idx",)),
        ("target_dihedral", ("target_dihedral",)),
    ):
        value = _first_present(data, names)
        if value is not None and value.numel() > 0:
            feats[key] = int(value[0].item()) if key != "target_dihedral" else float(value[0].item())
    return feats


def catalog_synthon_features(smiles: str) -> Dict[str, float]:
    """RDKit structural features of a catalog synthon (cached per SMILES).

    Returns zeros when RDKit cannot parse the SMILES - a single unparsable
    catalog row must never abort the curriculum build.
    """
    return _synthon_features_cached(str(smiles))


_SYNTHON_CACHE: Dict[str, Dict[str, float]] = {}


def _synthon_features_cached(smiles: str) -> Dict[str, float]:
    feats = _SYNTHON_CACHE.get(smiles)
    if feats is not None:
        return dict(feats)
    feats = {
        "synthon_n_atoms": 0.0,
        "synthon_n_rings": 0.0,
        "synthon_n_rotatable": 0.0,
        "synthon_n_hetero": 0.0,
        "synthon_symmetry_ratio": 0.0,
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            n_atoms = mol.GetNumAtoms()
            feats["synthon_n_atoms"] = float(n_atoms)
            feats["synthon_n_rings"] = float(mol.GetRingInfo().NumRings())
            feats["synthon_n_rotatable"] = float(
                rdMolDescriptors.CalcNumRotatableBonds(mol)
            )
            feats["synthon_n_hetero"] = float(
                sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (6, 1))
            )
            if n_atoms > 0:
                try:
                    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
                    feats["synthon_symmetry_ratio"] = float(
                        len(set(ranks)) / n_atoms
                    )
                except Exception:
                    feats["synthon_symmetry_ratio"] = 1.0
    except Exception:  # pragma: no cover - rdkit missing/broken row
        pass
    if len(_SYNTHON_CACHE) < 100_000:
        _SYNTHON_CACHE[smiles] = dict(feats)
    return dict(feats)


# ---------------------------------------------------------------------
# Difficulty model (train-only fitting)
# ---------------------------------------------------------------------
def _robust_bounds(values: List[float], lo_q: float = 0.01, hi_q: float = 0.99):
    """1st/99th percentile bounds of ``values`` (sorted-sample quantiles)."""
    if not values:
        return 0.0, 1.0
    ordered = sorted(values)
    n = len(ordered)
    lo = ordered[max(0, int(lo_q * n))]
    hi = ordered[min(n - 1, int(hi_q * n))]
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


class DifficultyModel:
    """Robust-normalised weighted difficulty score in [0, 1].

    Fitted on the TRAIN split only: the percentile bounds, and therefore the
    scores and the bin edges, never see validation/test statistics.
    """

    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        weights = dict(DEFAULT_FEATURE_WEIGHTS)
        if feature_weights:
            weights.update({str(k): float(v) for k, v in feature_weights.items()})
        self.weights = {k: v for k, v in weights.items() if k in FEATURE_NAMES}
        self.norms: Dict[str, Tuple[float, float]] = {}
        self._den = sum(v for v in self.weights.values() if v > 0) or 1.0

    def fit(self, feature_rows: List[Dict[str, float]]) -> None:
        self.norms = {}
        for name in FEATURE_NAMES:
            if name not in self.weights or self.weights[name] <= 0:
                continue
            values = [float(row.get(name, 0.0)) for row in feature_rows]
            self.norms[name] = _robust_bounds(values)

    def score(self, row: Dict[str, float]) -> float:
        total = 0.0
        for name, weight in self.weights.items():
            if weight <= 0:
                continue
            lo, hi = self.norms.get(name, (0.0, 1.0))
            raw = float(row.get(name, 0.0))
            clipped = min(max((raw - lo) / (hi - lo), 0.0), 1.0)
            total += weight * clipped
        return total / self._den


def quantile_edges(scores: List[float], n_levels: int) -> List[float]:
    """Balanced-quantile upper edges: ``edges[i]`` is the upper bound of level
    i+1.  ``edges[-1]`` is ``max(scores) + eps`` so every score is assigned."""
    if not scores:
        return []
    ordered = sorted(scores)
    n = len(ordered)
    edges = [
        ordered[min(n - 1, max(0, int(round((i / n_levels) * n))))]
        for i in range(1, n_levels)
    ]
    # Non-decreasing, distinct-enough edges guard (duplicate scores can make
    # quantile cut points collide; keep them strictly increasing by nudging).
    fixed = []
    for edge in edges:
        if fixed and edge <= fixed[-1]:
            edge = fixed[-1]
        fixed.append(float(edge))
    fixed.append(float(ordered[-1]) + 1e-12)
    return fixed


def level_for_score(score: float, edges: List[float], n_levels: int) -> int:
    """Level 1..n_levels for ``score`` given upper edges (train-fitted)."""
    if not edges:
        return 1
    for i, edge in enumerate(edges):
        if score < edge:
            return i + 1
    return n_levels


# ---------------------------------------------------------------------
# Curriculum sampler (Approach C)
# ---------------------------------------------------------------------
class CurriculumSampler(torch.utils.data.Sampler):  # type: ignore[misc]
    """Difficulty-aware mixed sampling with milestone-driven stages.

    * ``weights()`` - stage ``s`` centers a Gaussian on ``center(s)`` over the
      present levels, blended with a uniform floor so every level keeps at
      least ``min_weight``: ``w = (1-eps)*N(center, sigma) + eps*U``.  The
      floor cannot be diluted by renormalisation because the mixture already
      sums to one.
    * ``__iter__`` yields ``num_samples`` indices per epoch (with
      replacement), reseeded deterministically by ``set_epoch``.
    * ``maybe_advance(metrics)`` moves to the next stage when the milestone
      metric improves by ``improve_thresh`` (or degrades, for
      higher-is-better metrics), falling back to ``patience`` epochs so the
      run always progresses within the training budget.
    """

    def __init__(
        self,
        indices_by_level: Dict[int, List[int]],
        num_samples: int,
        n_levels: int = 9,
        n_stages: int = 6,
        min_weight: float = 0.03,
        sigma: float = 1.8,
        seed: int = 0,
        milestone_metric: str = "val_loss",
        lower_better: bool = True,
        improve_thresh: float = 0.005,
        patience: int = 2,
    ):
        self._by_level = {int(lv): list(map(int, idxs)) for lv, idxs in indices_by_level.items() if idxs}
        self.present_levels = sorted(self._by_level)
        if not self.present_levels:
            raise ValueError("CurriculumSampler needs at least one populated difficulty level")
        self.num_samples = max(1, int(num_samples))
        self.n_levels = max(2, int(n_levels))
        self.n_stages = max(1, int(n_stages))
        self.min_weight = float(min_weight)
        self.sigma = max(0.25, float(sigma))
        self.seed = int(seed)
        self.milestone_metric = str(milestone_metric)
        self.lower_better = bool(lower_better)
        self.improve_thresh = float(improve_thresh)
        self.patience = max(1, int(patience))

        self.stage = 0
        self._ref: Optional[float] = None
        self._epochs_since = 0
        self._epoch = 0
        self._rng = random.Random(self.seed)
        self.last_sampled_counts: Dict[int, int] = {}

        # Stage centers: start easy (first present level + 1), end at the
        # hardest present level.  With a single present level every center
        # collapses onto it.
        lo, hi = self.present_levels[0], self.present_levels[-1]
        if len(self.present_levels) == 1 or self.n_stages == 1:
            self._centers = [float(hi)] * self.n_stages
        else:
            start = min(lo + 1, hi)
            self._centers = [
                start + (hi - start) * i / max(1, self.n_stages - 1)
                for i in range(self.n_stages)
            ]

    # -- weights --------------------------------------------------------
    def weights(self) -> Dict[int, float]:
        center = self._centers[min(self.stage, len(self._centers) - 1)]
        raw = {
            lv: math.exp(-((lv - center) ** 2) / (2.0 * self.sigma ** 2))
            for lv in self.present_levels
        }
        total = sum(raw.values()) or 1.0
        n = len(self.present_levels)
        eps = min(self.min_weight * n, 0.9)
        return {
            lv: (1.0 - eps) * (raw[lv] / total) + eps / n
            for lv in self.present_levels
        }

    # -- torch Sampler API ------------------------------------------------
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        rng = random.Random((self.seed * 1_000_003 + self._epoch) % (2 ** 31))
        weights = self.weights()
        levels = list(weights)
        cumulative = []
        acc = 0.0
        for lv in levels:
            acc += weights[lv]
            cumulative.append(acc)
        counts: Dict[int, int] = defaultdict(int)
        out: List[int] = []
        for _ in range(self.num_samples):
            r = rng.random() * acc
            chosen = levels[-1]
            for j, bound in enumerate(cumulative):
                if r <= bound:
                    chosen = levels[j]
                    break
            pool = self._by_level[chosen]
            index = pool[rng.randrange(len(pool))]
            counts[chosen] += 1
            out.append(index)
        self.last_sampled_counts = dict(sorted(counts.items()))
        return iter(out)

    # -- stage advancement -------------------------------------------------
    def maybe_advance(self, metrics: Optional[Dict[str, float]]) -> Tuple[bool, Optional[str]]:
        self._epochs_since += 1
        advanced = False
        reason: Optional[str] = None
        value = None
        if metrics is not None:
            value = metrics.get(self.milestone_metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            value = float(value)
            if self._ref is None:
                self._ref = value
            elif self.stage < self.n_stages - 1:
                if self.lower_better and value <= self._ref - self.improve_thresh:
                    advanced, reason = True, (
                        f"{self.milestone_metric} improved {self._ref:.4f} -> {value:.4f} "
                        f"(>= {self.improve_thresh} milestone)"
                    )
                elif not self.lower_better and value >= self._ref + self.improve_thresh:
                    advanced, reason = True, (
                        f"{self.milestone_metric} improved {self._ref:.4f} -> {value:.4f} "
                        f"(>= {self.improve_thresh} milestone)"
                    )
            if not advanced and value is not None:
                if self.lower_better:
                    self._ref = min(self._ref, value)
                else:
                    self._ref = max(self._ref, value)
        if not advanced and self.stage < self.n_stages - 1 and self._epochs_since >= self.patience:
            advanced = True
            reason = f"patience {self.patience} epochs without a {self.milestone_metric} milestone"
        if advanced:
            self.stage += 1
            self._epochs_since = 0
            self._ref = None
        return advanced, reason

    # -- state ------------------------------------------------------------
    def state(self) -> Dict:
        return {
            "stage": self.stage,
            "n_stages": self.n_stages,
            "centers": self._centers,
            "ref": self._ref,
            "epochs_since_advance": self._epochs_since,
            "seed": self.seed,
            "present_levels": self.present_levels,
        }

    def load_state(self, state: Optional[Dict]) -> None:
        if not state:
            return
        try:
            self.stage = max(0, min(int(state["stage"]), self.n_stages - 1))
            self._epochs_since = max(0, int(state.get("epochs_since_advance", 0)))
            ref = state.get("ref")
            self._ref = float(ref) if isinstance(ref, (int, float)) else None
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("[curriculum] could not restore sampler state (%s); continuing fresh", exc)


# ---------------------------------------------------------------------
# Per-level metric tracker (validation reporting)
# ---------------------------------------------------------------------
class LevelTracker:
    """Accumulates per-level scalar means and ratio metrics, emitting
    ``"L{level}/{metric}"`` rows for the metrics entry / logs."""

    def __init__(self):
        self._scalars: Dict[Tuple[int, str], List[float]] = defaultdict(lambda: [0.0, 0])
        self._ratios: Dict[Tuple[int, str], List[float]] = defaultdict(lambda: [0.0, 0.0])

    def reset(self) -> None:
        self._scalars.clear()
        self._ratios.clear()

    def add_scalar(self, level: int, metric: str, value: Optional[float]) -> None:
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return
        slot = self._scalars[(int(level), str(metric))]
        slot[0] += float(value)
        slot[1] += 1

    def add_ratio(self, level: int, metric: str, numerator: float, denominator: float) -> None:
        slot = self._ratios[(int(level), str(metric))]
        slot[0] += float(numerator)
        slot[1] += float(denominator)

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for (lv, metric), (total, count) in sorted(self._scalars.items()):
            if count > 0:
                out[f"L{lv}/{metric}"] = total / count
        for (lv, metric), (num, den) in sorted(self._ratios.items()):
            if den > 0:
                out[f"L{lv}/{metric}"] = num / den
        return out


# ---------------------------------------------------------------------
# Controller: build from config + datasets, cache, wire into the trainer
# ---------------------------------------------------------------------
def _dataset_signature(dataset, catalog) -> Dict:
    signature = {"n_train": len(dataset)}
    try:
        signature["catalog_ids_hash"] = catalog.embedding_signature.get("ids_hash")
        signature["num_synthons"] = catalog.embedding_signature.get("num_synthons")
    except Exception:
        pass
    return signature


def _config_signature(cfg: Dict) -> Dict:
    keys = (
        "enabled", "n_levels", "n_stages", "min_weight", "sigma", "seed",
        "milestone_metric", "improve_thresh", "patience", "feature_weights",
        "feature_sample_stride",
    )
    return {k: cfg.get(k) for k in keys}


class CurriculumController:
    """Owns difficulty scoring, level assignment, the sampler, and reporting."""

    def __init__(
        self,
        cfg: Dict,
        feature_rows_train: List[Dict[str, float]],
        feature_rows_val: List[Dict[str, float]],
        model: DifficultyModel,
        edges: List[float],
        levels_train: List[int],
        levels_val: List[int],
        cache_path: Optional[str],
        from_cache: bool,
    ):
        self.cfg = dict(cfg)
        self.n_levels = int(cfg.get("n_levels", 9))
        self.model = model
        self.edges = edges
        self.levels_train = levels_train
        self.levels_val = levels_val
        self.cache_path = cache_path
        self.from_cache = from_cache
        # Kept for diagnostics/tests: the exact feature rows the model was
        # fitted on (train) and the val rows scored with the train edges.
        self._feature_rows_train = feature_rows_train
        self._feature_rows_val = feature_rows_val

        indices_by_level: Dict[int, List[int]] = defaultdict(list)
        for idx, lv in enumerate(levels_train):
            indices_by_level[int(lv)].append(int(idx))
        self.level_counts_train = {
            lv: len(idxs) for lv, idxs in sorted(indices_by_level.items())
        }
        counts_val: Dict[int, int] = defaultdict(int)
        for lv in levels_val:
            counts_val[int(lv)] += 1
        self.level_counts_val = dict(sorted(counts_val.items()))

        self.sampler = CurriculumSampler(
            indices_by_level,
            num_samples=len(levels_train),
            n_levels=self.n_levels,
            n_stages=int(cfg.get("n_stages", 6)),
            min_weight=float(cfg.get("min_weight", 0.03)),
            sigma=float(cfg.get("sigma", 1.8)),
            seed=int(cfg.get("seed", 0)),
            milestone_metric=str(cfg.get("milestone_metric", "val_loss")),
            lower_better=bool(cfg.get("lower_better", True)),
            improve_thresh=float(cfg.get("improve_thresh", 0.005)),
            patience=int(cfg.get("patience", 2)),
        )
        self.tracker = LevelTracker()
        self._val_tracking_ok = True

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        cfg: Dict,
        train_dataset,
        val_dataset,
        catalog,
        output_dir: str = "./experiments",
        log_fn=print,
    ) -> "CurriculumController":
        cfg = dict(cfg or {})
        n_levels = int(cfg.get("n_levels", 9))
        stride = max(1, int(cfg.get("feature_sample_stride", 1)))
        seed = int(cfg.get("seed", 0))

        def smiles_for(synthon_idx: int) -> Optional[str]:
            try:
                return str(catalog.df.iloc[int(synthon_idx)]["smiles"])
            except Exception:
                return None

        # ---- feature pass (deterministic; read-only dataset access) ----
        def collect(dataset) -> List[Dict[str, float]]:
            rows: List[Dict[str, float]] = []
            n = len(dataset)
            for idx in range(0, n, stride):
                try:
                    sample = dataset[idx]
                except Exception as exc:
                    logger.warning("[curriculum] feature pass skipped dataset[%d] (%s)", idx, exc)
                    rows.append({})
                    continue
                row = tensor_sample_features(sample)
                synthon_idx = row.get("target_synthon")
                if synthon_idx is not None:
                    smiles = smiles_for(synthon_idx)
                    if smiles is not None:
                        row.update(catalog_synthon_features(smiles))
                rows.append(row)
            return rows

        rows_train = collect(train_dataset)
        rows_val = collect(val_dataset) if len(val_dataset) else []

        # ---- rarity from the TRAIN split only (no leakage) ----
        def attach_rarity(rows: List[Dict[str, float]], train_stats: Optional[Tuple[Counter, Counter, int]]) -> None:
            if train_stats is None:
                for row in rows:
                    row["reaction_family_rarity"] = 0.0
                    row["core_handle_rarity"] = 0.0
                return
            family_counts, handle_counts, n = train_stats
            for row in rows:
                family = row.get("target_reaction_family_idx")
                handle = row.get("target_core_handle_idx")
                row["reaction_family_rarity"] = (
                    -math.log2(max(family_counts.get(int(family), 1), 1) / n)
                    if family is not None else 0.0
                )
                row["core_handle_rarity"] = (
                    -math.log2(max(handle_counts.get(int(handle), 1), 1) / n)
                    if handle is not None else 0.0
                )

        family_counts: Counter = Counter()
        handle_counts: Counter = Counter()
        counted = 0
        for row in rows_train:
            family = row.get("target_reaction_family_idx")
            handle = row.get("target_core_handle_idx")
            if family is not None:
                family_counts[int(family)] += 1
            if handle is not None:
                handle_counts[int(handle)] += 1
            counted += 1
        train_stats = (family_counts, handle_counts, max(1, counted))
        attach_rarity(rows_train, train_stats)
        attach_rarity(rows_val, train_stats)

        # ---- fit + bin (train only) ----
        model = DifficultyModel(cfg.get("feature_weights"))
        model.fit(rows_train)
        scores_train = [model.score(row) for row in rows_train]
        edges = quantile_edges(scores_train, n_levels)
        levels_train = [level_for_score(s, edges, n_levels) for s in scores_train]
        scores_val = [model.score(row) for row in rows_val]
        levels_val = [level_for_score(s, edges, n_levels) for s in scores_val]

        # ---- cache artifact (skip when disabled) ----
        cache_path = cfg.get("cache_path")
        from_cache = False
        signature = {
            "schema": FEATURE_SCHEMA_VERSION,
            "features": list(FEATURE_NAMES),
            "config": _config_signature(cfg),
            "dataset": _dataset_signature(train_dataset, catalog),
            "n_val": len(val_dataset),
        }
        if cache_path:
            path = os.path.join(output_dir, os.path.basename(str(cache_path))) \
                if not os.path.isabs(str(cache_path)) else str(cache_path)
            payload = {
                "signature": signature,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "feature_norms": {k: list(v) for k, v in model.norms.items()},
                "train_edges": edges,
                "levels_train": levels_train,
                "levels_val": levels_val,
                "level_counts_train": {},
                "level_counts_val": {},
            }
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if cached.get("signature") == signature:
                        levels_train = [int(x) for x in cached["levels_train"]]
                        levels_val = [int(x) for x in cached["levels_val"]]
                        edges = [float(x) for x in cached["train_edges"]]
                        from_cache = True
                        logger.info("[curriculum] loaded cached level assignment from %s", path)
                    else:
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(payload, f)
                        logger.info("[curriculum] feature/level cache regenerated (%s)", path)
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                        logger.info("[curriculum] feature/level cache written (%s)", path)
            except OSError as exc:
                logger.warning("[curriculum] cache artifact unavailable (%s); levels recomputed", exc)
            cfg["cache_path_resolved"] = path

        controller = cls(
            cfg, rows_train, rows_val, model, edges, levels_train, levels_val,
            cache_path, from_cache,
        )
        controller._signature = signature
        return controller

    # ------------------------------------------------------------------
    # Reporting / trainer hooks
    # ------------------------------------------------------------------
    def log_summary(self, log_fn=print) -> None:
        counts = " ".join(
            f"L{lv}={self.level_counts_train.get(lv, 0)}"
            for lv in range(1, self.n_levels + 1)
        )
        val_counts = " ".join(
            f"L{lv}={self.level_counts_val.get(lv, 0)}"
            for lv in range(1, self.n_levels + 1)
        )
        weights = self.sampler.weights()
        weights_txt = " ".join(
            f"L{lv}={weights[lv]:.3f}" for lv in sorted(weights)
        )
        source = "cache" if self.from_cache else "computed"
        log_fn(
            f"[curriculum] enabled: levels={self.n_levels} stages={self.sampler.n_stages} "
            f"milestone={self.sampler.milestone_metric} (improve>={self.sampler.improve_thresh}, "
            f"patience={self.sampler.patience}) | assignment={source}"
        )
        log_fn(
            f"[curriculum] train level counts: {counts} | "
            f"val level counts: {val_counts}"
        )
        log_fn(
            f"[curriculum] stage {self.sampler.stage}/{self.sampler.n_stages - 1} "
            f"center={self.sampler._centers[min(self.sampler.stage, len(self.sampler._centers) - 1)]:.2f} | "
            f"level weights: {weights_txt}"
        )

    def epoch_update(self, epoch: int, entry: Dict) -> Dict:
        """Call after the epoch entry is assembled (with val metrics merged).

        Returns the curriculum metric fields to merge into the entry.  Also
        advances the curriculum stage (milestone or patience) and reports the
        per-level sampled counts of the epoch that just ran.
        """
        advanced, reason = self.sampler.maybe_advance(entry)
        if advanced:
            print(
                f"[curriculum] epoch {epoch}: stage -> {self.sampler.stage}/"
                f"{self.sampler.n_stages - 1} ({reason}); new center="
                f"{self.sampler._centers[min(self.sampler.stage, len(self.sampler._centers) - 1)]:.2f}"
            )
            weights = self.sampler.weights()
            print(
                "[curriculum] new level weights: "
                + " ".join(f"L{lv}={weights[lv]:.3f}" for lv in sorted(weights))
            )
        weights = self.sampler.weights()
        counts_txt = " ".join(
            f"L{lv}={self.sampler.last_sampled_counts.get(lv, 0)}"
            for lv in sorted(weights)
        )
        print(
            f"[curriculum] epoch {epoch:03d} | stage {self.sampler.stage}/"
            f"{self.sampler.n_stages - 1} | sampled: {counts_txt}"
        )
        metrics = {
            "curriculum_stage": self.sampler.stage,
            "curriculum_center": self.sampler._centers[
                min(self.sampler.stage, len(self.sampler._centers) - 1)
            ],
            "curriculum_level_weights": {
                str(lv): round(w, 4) for lv, w in weights.items()
            },
            "curriculum_level_counts": {
                str(lv): c for lv, c in self.sampler.last_sampled_counts.items()
            },
        }
        return metrics

    # -- per-level validation tracking ------------------------------------
    def begin_val_pass(self) -> None:
        """Reset per-level accumulators at the start of a validation pass."""
        self.tracker.reset()
        self._val_tracking_ok = True

    def track_val_batch(
        self,
        start_index: int,
        per_sample_loss,
        per_sample_reaction_correct,
        non_stop_mask,
    ) -> None:
        """Accumulate per-level validation stats for one batch of samples.

        ``start_index`` is the dataset index of the first sample in the
        batch (validation loader is sequential, drop_last=False).  Tracking is
        skipped for a pass in which any sample was dropped by the collator
        (index alignment would be undefined); a warning is emitted once.
        """
        if not self._val_tracking_ok:
            return
        if start_index >= len(self.levels_val):
            self._val_tracking_ok = False
            return
        n = per_sample_loss.shape[0] if hasattr(per_sample_loss, "shape") else len(per_sample_loss)
        for offset in range(n):
            dataset_idx = start_index + offset
            if dataset_idx >= len(self.levels_val):
                break
            level = int(self.levels_val[dataset_idx])
            self.tracker.add_scalar(level, "val_loss", float(per_sample_loss[offset].item()))
            if non_stop_mask is not None:
                if bool(non_stop_mask[offset].item()):
                    self.tracker.add_ratio(
                        level, "val_reaction_acc",
                        1.0 if bool(per_sample_reaction_correct[offset].item()) else 0.0,
                        1.0,
                    )

    def level_metrics_summary(self, dropped_delta: int = 0) -> Dict[str, float]:
        """Per-level validation metrics (``L{level}/val_loss`` etc.).

        ``dropped_delta`` is the number of samples the collator dropped during
        this validation pass; any drop breaks the dataset-index alignment,
        so the summary refuses loudly instead of emitting misaligned numbers.
        """
        if int(dropped_delta) > 0:
            self._val_tracking_ok = False
        if not self._val_tracking_ok:
            print(
                "[curriculum] per-level validation metrics unavailable this "
                "epoch (non-finite samples dropped by the collator broke "
                "index alignment)"
            )
            return {}
        return self.tracker.summary()

    # -- checkpoint state ---------------------------------------------------
    def state(self) -> Dict:
        return {
            "sampler": self.sampler.state(),
            "n_levels": self.n_levels,
            "signature": getattr(self, "_signature", None),
        }

    def load_state(self, state: Optional[Dict]) -> None:
        if not isinstance(state, dict):
            return
        if state.get("n_levels") not in (None, self.n_levels):
            logger.warning(
                "[curriculum] checkpoint level count differs (%s vs %s); ignoring saved stage",
                state.get("n_levels"), self.n_levels,
            )
            return
        self.sampler.load_state(state.get("sampler"))


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "DEFAULT_FEATURE_WEIGHTS",
    "DifficultyModel",
    "CurriculumSampler",
    "CurriculumController",
    "LevelTracker",
    "tensor_sample_features",
    "catalog_synthon_features",
    "quantile_edges",
    "level_for_score",
]
