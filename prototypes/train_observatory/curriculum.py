"""
curriculum.py — leakage-safe data splits + 9-level difficulty annotation +
difficulty-aware curriculum sampling for molecular / synthesis-tree training.

Three independent pieces (use any of them alone):

1. LEAKAGE PREVENTION — grouped_split()
   Splits the dataset so that every *group* (scaffold / parent molecule /
   reaction family / conformer family) lands entirely in one partition.
   Prevents near-duplicate leakage between train and validation/test.
   Apply this BEFORE any curriculum, and never re-split afterwards.

2. DIFFICULTY ANNOTATION — difficulty_scores() + assign_levels()
   Computes a [0, 1] difficulty score per example from molecular/geometric
   features (size, branching, rotatable bonds, ring complexity, heteroatoms,
   symmetry, ...), then cuts the training set into N ≈ balanced difficulty
   bins (default 9 levels, L1 = easiest ... L9 = hardest) via quantiles.

3. CURRICULUM (Approach C — difficulty-aware MIXED training)
   CurriculumSampler: start with a mixture weighted toward easy/intermediate
   levels and gradually shift mass toward harder ones as *validation
   milestones* are reached (improvement threshold or patience), never fully
   removing easy examples (min level weight floor). LevelTracker accumulates
   per-level metrics so you can plot/compare every difficulty level
   separately and prove the curriculum helps where it should.

The transition policy is milestone-based, not epoch-based: the stage advances
when the milestone metric (default val_loss) improves by `improve_thresh`,
or when `patience` epochs pass without a milestone (safety net against
stalling). The final stage centers on the hardest level but still keeps a
small proportion of easy examples.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False

try:
    from mol_metrics import tokens, is_valid
except Exception:  # allow standalone use
    import re
    _TOK = re.compile(r"Cl|Br|\[[^\]]+\]|[BCNOSPFILbcnospfil]|[#\=\(\)\[\]\.\+\-%@]|\d")

    def tokens(smi):
        return _TOK.findall(str(smi)) if smi else []

    def is_valid(smi):
        return bool(smi)


# ------------------------------------------------------------------ group keys
def scaffold_key(smiles):
    """Murcko scaffold (RDKit). Fallback: main chain after removing branches."""
    if _HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "no_scaffold"
        except Exception:
            pass
    s = str(smiles)
    while "(" in s:  # strip simple branch groups (fallback only)
        try:
            i = s.index("(")
            depth = 0
            for j in range(i, len(s)):
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                    if depth == 0:
                        s = s[:i] + s[j + 1:]
                        break
            else:
                break
        except Exception:
            break
    return s.strip() or "no_scaffold"


def parent_key(record, field="smiles"):
    """Group key for conformer/parent families: the molecule identity itself."""
    return str(record.get(field) if isinstance(record, dict) else record)


# ------------------------------------------------------------- grouped splitting
def grouped_split(items, key_fn=None, val_frac=0.15, test_frac=0.15, seed=0,
                  shuffle=True):
    """Split indices of `items` so that no group (key_fn) crosses partitions.

    Returns (train_idx, val_idx, test_idx, groups) where groups maps
    key -> partition ('train' | 'val' | 'test'). Whole groups are assigned in
    shuffled order until per-partition item quotas are (approximately) met —
    quotas can be exceeded only by at most one group each.
    """
    n = len(items)
    if key_fn is None:
        key_fn = (lambda it: str(it))
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[key_fn(it)].append(i)
    rng = random.Random(seed)
    names = sorted(groups)
    rng.shuffle(names)
    n_test, n_val = int(round(test_frac * n)), int(round(val_frac * n))
    train_idx, val_idx, test_idx = [], [], []
    c_test = c_val = 0
    for g in names:
        idxs = list(groups[g])
        if c_test < max(n_test, 1) and n_test > 0:
            test_idx += idxs
            c_test += len(idxs)
        elif c_val < max(n_val, 1) and n_val > 0:
            val_idx += idxs
            c_val += len(idxs)
        else:
            train_idx += idxs
    if shuffle:
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
    part = {}
    for g in groups:
        if all(i in set(test_idx) for i in groups[g]):
            part[g] = "test"
        elif all(i in set(val_idx) for i in groups[g]):
            part[g] = "val"
        else:
            part[g] = "train"
    return train_idx, val_idx, test_idx, part


def assert_no_leakage(items, key_fn, splits):
    """Raises ValueError if any group appears in more than one partition."""
    train_idx, val_idx, test_idx = splits[0], splits[1], splits[2]
    seen = {}
    for part, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in idxs:
            g = key_fn(items[i])
            if g in seen and seen[g] != part:
                raise ValueError(f"leakage detected: group {g!r} in {seen[g]} and {part}")
            seen[g] = part
    return True


# ------------------------------------------------------------ difficulty scoring
DEFAULT_FEATURE_WEIGHTS = {
    "n_atoms": 1.0, "n_branches": 1.0, "n_hetero": 0.8, "n_rings": 1.2,
    "n_rotatable": 1.2, "n_chars": 0.5, "symmetry": 0.4,
}


def molecular_difficulty_features(smiles):
    """Structural difficulty features. RDKit adds rotatable-bond / ring counts;
    the fallback uses token statistics. Extend with your 3D-geometry features
    (conformer counts, steric crowding, torsion targets) for real datasets."""
    toks = tokens(smiles)
    feats = {
        "n_atoms": sum(1 for t in toks if t and (t[0].isalpha())),
        "n_branches": sum(1 for t in toks if t == "("),
        "n_hetero": sum(1 for t in toks if t and t[0] in "NOSPFinospfi"),
        "n_chars": len(str(smiles)),
        "n_rings": 0, "n_rotatable": 0, "symmetry": 0.0,
    }
    if _HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                feats["n_rings"] = mol.GetRingInfo().NumRings()
                feats["n_rotatable"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                branches = [t for t in toks if t == "("]
                feats["symmetry"] = len(branches) - len(set(branches)) if branches else 0.0
        except Exception:
            pass
    return feats


def difficulty_scores(records, feature_fn=molecular_difficulty_features,
                      feature_of=None, weights=None):
    """Robust-normalized weighted difficulty score in [0, 1] per record.
    `feature_of(record)` extracts the SMILES (default: record itself, or
    record["smiles"] for dicts). Normalization uses 1st/99th percentiles."""
    if feature_of is None:
        def feature_of(r):
            return r.get("smiles") if isinstance(r, dict) else r
    w = dict(DEFAULT_FEATURE_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})
    raw = [molecular_difficulty_features(feature_of(r)) if feature_fn is molecular_difficulty_features
           else feature_fn(r) for r in records]
    keys = sorted({k for d in raw for k in d})
    # robust min/max per feature
    norms = {}
    for k in keys:
        vals = sorted(d.get(k, 0.0) for d in raw)
        lo = vals[max(0, int(0.01 * len(vals)))]
        hi = vals[min(len(vals) - 1, int(0.99 * len(vals)))]
        norms[k] = (lo, hi if hi > lo else lo + 1e-9)
    scores = []
    for d in raw:
        num = sum(w.get(k, 0.0) * min(max((d.get(k, 0.0) - norms[k][0]) /
                                          (norms[k][1] - norms[k][0]), 0.0), 1.0)
                  for k in keys if w.get(k, 0.0) > 0)
        den = sum(w[k] for k in keys if w.get(k, 0.0) > 0) or 1.0
        scores.append(num / den)
    return scores


def assign_levels(scores, n_levels=9):
    """Quantile-binned difficulty levels 1..N (≈ balanced bins). Returns
    (levels list parallel to scores, bin edge values)."""
    if not scores:
        return [], []
    srt = sorted(scores)
    n = len(srt)
    edges = [srt[min(n - 1, max(0, int(round((i / n_levels) * n))))]
             for i in range(n_levels)] + [srt[-1] + 1e-12]
    levels = []
    for s in scores:
        lv = n_levels
        for i in range(n_levels):
            if s < edges[i + 1] or i == n_levels - 1:
                lv = i + 1
                break
        levels.append(lv)
    return levels, edges


# -------------------------------------------------------------- curriculum sampler
class CurriculumSampler:
    """Difficulty-aware mixed sampling (Approach C).

    Stage s of S centers sampling mass at difficulty level center(s) — starting
    easy and ending hard — with Gaussian spread (sigma) over neighboring levels
    and a hard floor (min_weight) so easy examples are never fully removed.
    Advancement is milestone-driven (see maybe_advance)."""

    def __init__(self, train_indices, levels, n_levels=9, seed=0, n_stages=6,
                 min_weight=0.03, sigma=1.8, milestone_metric="val_loss",
                 lower_better=True, improve_thresh=0.005, patience=3):
        self._by_level = defaultdict(list)
        for idx, lv in zip(train_indices, levels):
            self._by_level[int(lv)].append(int(idx))
        self.present_levels = sorted(self._by_level)
        self.n_levels = n_levels
        self.n_stages = max(1, int(n_stages))
        self.min_weight = float(min_weight)
        self.sigma = float(sigma)
        self.milestone_metric = milestone_metric
        self.lower_better = bool(lower_better)
        self.improve_thresh = float(improve_thresh)
        self.patience = int(patience)
        self._rng = random.Random(seed)
        self.stage = 0
        self.history = []
        self._ref = None
        self._epochs_since = 0
        # centers: from level 2 (easy but not trivial) up to hardest level
        if self.present_levels:
            lo = min(self.present_levels[0] + 1, self.present_levels[-1])
            hi = self.present_levels[-1]
            self._centers = [lo + (hi - lo) * i / max(1, self.n_stages - 1)
                             for i in range(self.n_stages)]
        else:
            self._centers = []

    # -- sampling ------------------------------------------------------------
    def weights(self):
        """Current level -> sampling weight (sums to exactly 1 over present levels,
        every weight >= min_weight). Implemented as a mixture
        (1-eps)*gaussian + eps*uniform so renormalization cannot dilute the floor."""
        if not self._centers:
            return {}
        c = self._centers[min(self.stage, len(self._centers) - 1)]
        raw = {lv: math.exp(-((lv - c) ** 2) / (2 * self.sigma ** 2))
               for lv in self.present_levels}
        total = sum(raw.values()) or 1.0
        n = len(self.present_levels)
        eps = min(self.min_weight * n, 0.9)  # total mass reserved for the uniform floor
        return {lv: (1.0 - eps) * (raw[lv] / total) + eps / n
                for lv in self.present_levels}

    def sample(self, n):
        """Draw n train indices from the difficulty-weighted mixture."""
        if not self._by_level or n <= 0:
            return []
        w = self.weights()
        lvs = list(w)
        cum = []
        acc = 0.0
        for lv in lvs:
            acc += w[lv]
            cum.append(acc)
        out = []
        for _ in range(n):
            r = self._rng.random() * acc
            lv = lvs[-1]
            for j, c in enumerate(cum):
                if r <= c:
                    lv = lvs[j]
                    break
            pool = self._by_level[lv]
            out.append(pool[self._rng.randrange(len(pool))])
        return out

    # -- stage advancement (milestone-based, never epoch-based) -------------
    def maybe_advance(self, epoch_metrics):
        """Call once per epoch with the epoch's evaluation metrics dict.
        Returns True if the curriculum advanced a stage."""
        self._epochs_since += 1
        advanced = False
        v = (epoch_metrics or {}).get(self.milestone_metric)
        if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
            self.history.append(float(v))
        if self.stage < self.n_stages - 1:
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                if self._ref is None:
                    self._ref = float(v)
                elif self.lower_better and v <= self._ref - self.improve_thresh:
                    self._advance()
                    advanced = True
                elif (not self.lower_better) and v >= self._ref + self.improve_thresh:
                    self._advance()
                    advanced = True
            if not advanced and self._epochs_since >= self.patience:
                self._advance()
                advanced = True
        return advanced

    def _advance(self):
        self.stage += 1
        self._epochs_since = 0
        self._ref = None  # re-baseline milestones at the new stage

    def state(self):
        return {"stage": self.stage, "n_stages": self.n_stages,
                "center": self._centers[min(self.stage, len(self._centers) - 1)] if self._centers else None,
                "weights": {str(k): round(v, 4) for k, v in self.weights().items()},
                "epochs_since_advance": self._epochs_since}


# --------------------------------------------------------------- per-level tracker
class LevelTracker:
    """Accumulate per-difficulty-level sums/counts and ratio metrics, then emit
    "L{level}/{metric}" rows for RunMonitor.evaluation()."""

    def __init__(self):
        self._scalars = defaultdict(lambda: [0.0, 0])   # (level, metric) -> [sum, n]
        self._ratios = defaultdict(lambda: [0.0, 0.0])   # (level, metric) -> [num, den]

    def add_scalar(self, level, metric, value):
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            return
        s = self._scalars[(int(level), str(metric))]
        s[0] += float(value)
        s[1] += 1

    def add_ratio(self, level, metric, numerator, denominator=1):
        r = self._ratios[(int(level), str(metric))]
        r[0] += float(numerator)
        r[1] += float(denominator)

    def reset(self):
        self._scalars.clear()
        self._ratios.clear()

    def summary(self):
        out = {}
        for (lv, m), (s, n) in self._scalars.items():
            if n > 0:
                out[f"L{lv}/{m}"] = s / n
        for (lv, m), (num, den) in self._ratios.items():
            if den > 0:
                out[f"L{lv}/{m}"] = num / den
        return out
