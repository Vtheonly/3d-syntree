"""
mol_metrics.py — actual evaluation metrics for synthesis-tree / retrosynthesis /
molecular-generation models (3D-SynTree style).

Every metric is a plain function over lists/arrays so you can call it from any
eval loop, and every value lands in [0, 1] so it can be fed directly into
RunMonitor.evaluation() for logging + plotting.

Metric families
---------------
1. Reaction prediction
   reaction_metrics(logits, targets, ks) -> reaction_acc (top-1), reaction_topk@k

2. Top-k prediction accuracy (generic, score/ranking-based)
   topk_accuracy(logits, targets, ks=(1, 3, 5, 10))

3. Synthon prediction (candidates are ranked SMILES lists per sample)
   synthon_metrics(ranked_preds, targets, ks) -> synthon_EM, synthon_topk_EM@k,
                                              synthon_tanimoto_{mean,max,median}

4. Molecular generation quality
   validity_rate, uniqueness_rate, novelty_rate, internal_diversity,
   rediscovery_rate -> generation_metrics() aggregates all of them.

Chemistry backends
------------------
RDKit optional. When present: canonical SMILES + Morgan fingerprints (radius 2,
2048 bits) — real chemistry. When absent: deterministic token n-gram hash
fingerprints + syntax validity checks — sufficient for prototyping and CI.
Both paths give identical *shapes* of results, so the same eval code runs
everywhere; only chemistry-aware canonicalization (e.g. "CCO" == "OCC")
requires RDKit.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from itertools import combinations

_HAS_RDKIT = False
try:
    from rdkit import Chem
    from rdkit import RDLogger
    # generation-metric evaluation intentionally feeds RDKit invalid candidate SMILES
    # (validity_rate measures exactly that) — silence the expected-parse-failure spam.
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        from rdkit.Chem import rdFingerprintGenerator
        _MORGAN_GENS = {}

        def _morgan_bits(mol, radius, nbits):
            key = (radius, nbits)
            if key not in _MORGAN_GENS:
                _MORGAN_GENS[key] = rdFingerprintGenerator.GetMorganGenerator(
                    radius=radius, fpSize=nbits)
            return set(_MORGAN_GENS[key].GetFingerprint(mol).GetOnBits())
    except Exception:  # older RDKit
        from rdkit.Chem import AllChem

        def _morgan_bits(mol, radius, nbits):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            return set(fp.GetOnBits())
    _HAS_RDKIT = True
except Exception:  # pragma: no cover — pure-python fallback used instead
    pass

_HAS_NUMPY = False
try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    pass

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- helpers
def _to_numpy_2d(logits):
    """Accept torch tensor / numpy array / nested list -> (list of rows, n_rows)."""
    if _HAS_TORCH and isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().tolist()
    elif _HAS_NUMPY and isinstance(logits, _np.ndarray):
        logits = logits.tolist()
    return [list(map(float, r)) for r in logits]


def _to_numpy_1d(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    elif _HAS_NUMPY and isinstance(x, _np.ndarray):
        x = x.tolist()
    return [int(v) for v in x]


_TOKEN_RE = re.compile(r"Cl|Br|\[[^\]]+\]|[BCNOSPFILbcnospfil]|[#\=\(\)\[\]\.\+\-%@]|\d")


def tokens(smi):
    """Simple SMILES tokenizer (works without RDKit)."""
    return _TOKEN_RE.findall(str(smi)) if smi else []


def _hash_ng(ng, nbits):
    return int(hashlib.md5("\x1f".join(ng).encode("utf-8")).hexdigest(), 16) % nbits


def token_fingerprint(smi, ngrams=(1, 2, 3), nbits=2048):
    """Deterministic n-gram hash fingerprint -> set of on-bit indices."""
    toks = tokens(smi)
    bits = set()
    for n in ngrams:
        for i in range(len(toks) - n + 1):
            bits.add(_hash_ng(toks[i:i + n], nbits))
    return bits


def fingerprint(smiles, nbits=2048, radius=2):
    """Morgan fingerprint (RDKit) or token-hash fingerprint (fallback) -> set of bits."""
    if _HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                return _morgan_bits(mol, radius, nbits)
        except Exception:
            pass
    return token_fingerprint(smiles, nbits=nbits)


def canonical(smiles):
    """Canonical SMILES (RDKit). Fallback: stripped raw string (no reordering)."""
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    if _HAS_RDKIT:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception:
            return None
    return s


_ATOM_CHARS = set("BCNOSPFILbcnospfil") | {"Cl", "Br"}
_ALLOWED_CHARS = set("BCNOSPFILbcnospfil()[]=#.+%@-0123456789ClBrH")


def _syntax_valid(smi):
    """Pure-python SMILES sanity check used when RDKit is unavailable:
    charset, bracket balance, ring-digit pairing, at least one atom."""
    if not smi:
        return False
    toks = tokens(smi)
    if not toks:
        return False
    depth = 0
    saw_atom = False
    digits = {}
    for t in toks:
        if t in ("(", "["):
            depth += 1
        elif t in (")", "]"):
            depth -= 1
            if depth < 0:
                return False
        elif t in _ATOM_CHARS:
            saw_atom = True
        elif t.isdigit():
            digits[t] = digits.get(t, 0) + 1
        for ch in t:
            if ch not in _ALLOWED_CHARS:
                return False
    if depth != 0 or not saw_atom:
        return False
    return all(v % 2 == 0 for v in digits.values())


def is_valid(smiles):
    """Chemical validity: RDKit parse test, or syntax check as fallback."""
    if not smiles:
        return False
    if _HAS_RDKIT:
        try:
            return Chem.MolFromSmiles(str(smiles)) is not None
        except Exception:
            return False
    return _syntax_valid(smiles)


# ----------------------------------------------------------------- top-k accuracy
def topk_accuracy(logits, targets, ks=(1, 3, 5, 10)):
    """Fraction of samples whose true class is within the top-k scored classes.
    `logits`: [B, C] scores (any ordering semantics — higher = better).
    Returns {"top1": 0.83, "top3": 0.95, ...}. ks larger than C are clamped."""
    rows = _to_numpy_2d(logits)
    tgt = _to_numpy_1d(targets)
    n = len(rows)
    if n == 0 or len(tgt) != n:
        return {f"top{k}": float("nan") for k in ks}
    ranks = []
    for row, t in zip(rows, tgt):
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))  # stable, deterministic
        ranks.append(order.index(t) + 1 if 0 <= t < len(row) else len(row) + 1)
    out = {}
    for k in ks:
        kk = max(1, min(int(k), len(rows[0])))
        out[f"top{k}"] = sum(1 for r in ranks if r <= kk) / n
    return out


def reaction_metrics(logits, targets, ks=(1, 3, 5, 10)):
    """Reaction-class prediction quality. `reaction_acc` IS top-1 accuracy."""
    tk = topk_accuracy(logits, targets, ks=ks)
    out = {"reaction_acc": tk.get("top1", float("nan"))}
    for k in ks:
        if int(k) != 1:
            out[f"reaction_topk@{k}"] = tk.get(f"top{k}", float("nan"))
    return out


# ---------------------------------------------------------------------------- tanimoto
def tanimoto(bits_a, bits_b):
    """Tanimoto of two bit-sets: |A∩B| / |A∪B|. Both empty -> 1.0 (identical)."""
    if not bits_a and not bits_b:
        return 1.0
    union = len(bits_a | bits_b)
    if union == 0:
        return 1.0
    return len(bits_a & bits_b) / union


def tanimoto_smiles(a, b):
    return tanimoto(fingerprint(a), fingerprint(b))


# ----------------------------------------------------------------- synthon prediction
def exact_match(pred, target):
    """Canonical-string equality (needs RDKit for true canonicalization)."""
    cp, ct = canonical(pred), canonical(target)
    return cp is not None and ct is not None and cp == ct


def synthon_metrics(ranked_preds, targets, ks=(1, 3, 5)):
    """Synthon prediction quality over ranked candidate lists.

    ranked_preds[i] : list of candidate synthon SMILES, best first (any length)
    targets[i]      : ground-truth synthon SMILES for sample i
    Returns synthon_EM (top-1 exact match), synthon_topk_EM@k (target anywhere
    in top-k), and Tanimoto similarity of the top-1 candidate to the target."""
    em_hits, topk_hits, tans = [], {k: [] for k in ks}, []
    for preds, tgt in zip(ranked_preds, targets):
        preds = list(preds or [])
        if not preds:
            em_hits.append(0.0)
            for k in ks:
                topk_hits[k].append(0.0)
            continue
        em_hits.append(float(exact_match(preds[0], tgt)))
        for k in ks:
            kk = min(int(k), len(preds))
            topk_hits[k].append(float(any(exact_match(p, tgt) for p in preds[:kk])))
        if is_valid(preds[0]) and is_valid(tgt):
            tans.append(tanimoto_smiles(preds[0], tgt))
    n = max(1, len(em_hits))
    out = {"synthon_EM": sum(em_hits) / n,
           "synthon_tanimoto_mean": (sum(tans) / len(tans)) if tans else 0.0,
           "synthon_tanimoto_max": max(tans) if tans else 0.0,
           "synthon_tanimoto_median": statistics.median(tans) if tans else 0.0}
    for k in ks:
        out[f"synthon_topk_EM@{k}"] = sum(topk_hits[k]) / len(topk_hits[k]) if topk_hits[k] else 0.0
    return out


# ------------------------------------------------------------------- generation quality
def validity_rate(smiles_list):
    n = len(smiles_list)
    if n == 0:
        return float("nan")
    return sum(1.0 for s in smiles_list if is_valid(s)) / n


def uniqueness_rate(smiles_list):
    n = len(smiles_list)
    if n == 0:
        return float("nan")
    uniq = {canonical(s) or str(s) for s in smiles_list}
    return len(uniq) / n


def novelty_rate(smiles_list, reference_pool):
    n = len(smiles_list)
    if n == 0 or not reference_pool:
        return float("nan") if n == 0 else 1.0
    ref = {canonical(s) or str(s) for s in reference_pool}
    return sum(1.0 for s in smiles_list if (canonical(s) or str(s)) not in ref) / n


def internal_diversity(smiles_list, max_pairs=2048):
    """1 - mean pairwise Tanimoto among generated samples (higher = more diverse)."""
    n = len(smiles_list)
    if n < 2:
        return float("nan") if n < 2 else 0.0
    fps = [fingerprint(s) for s in smiles_list]
    if n > 512:  # sample pairs for large sets
        idx = list(combinations(range(n), 2))
        step = max(1, len(idx) // max_pairs)
        idx = idx[::step]
    else:
        idx = list(combinations(range(n), 2))
    sims = [tanimoto(fps[a], fps[b]) for a, b in idx]
    return 1.0 - (sum(sims) / len(sims))


def rediscovery_rate(preds, targets):
    """Fraction of aligned (pred, target) pairs that are exact canonical matches."""
    n = len(preds)
    if n == 0 or len(targets) != n:
        return float("nan")
    return sum(1.0 for p, t in zip(preds, targets) if exact_match(p, t)) / n


def generation_metrics(generated, targets=None, reference_pool=None):
    """Aggregate molecular-generation quality of a list of generated SMILES.
    All values in [0, 1]; higher is better everywhere."""
    out = {"gen_validity": validity_rate(generated),
           "gen_uniqueness": uniqueness_rate(generated),
           "gen_diversity": internal_diversity(generated)}
    if reference_pool:
        out["gen_novelty"] = novelty_rate(generated, reference_pool)
    if targets is not None and len(targets) == len(generated) and targets:
        out["gen_rediscovery"] = rediscovery_rate(generated, targets)
        tan_pairs = [(p, t) for p, t in zip(generated, targets)
                     if is_valid(p) and is_valid(t)]
        if tan_pairs:
            out["gen_tanimoto_to_target"] = (sum(tanimoto_smiles(p, t)
                                                 for p, t in tan_pairs) / len(tan_pairs))
    return out


# ------------------------------------------------------------------- report helper
def summarize_metric_sets(*metric_dicts, prefix=""):
    """Merge eval dicts (later non-NaN wins) — handy before logging to the monitor."""
    merged = {}
    for d in metric_dicts:
        for k, v in (d or {}).items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            merged[f"{prefix}{k}"] = v
    return merged
