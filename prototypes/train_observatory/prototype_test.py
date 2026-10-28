"""
prototype_test.py — isolated mini-"SynTree" prototype + full self-test suite.
==============================================================================
A small, standalone stand-in for your 3D-SynTree trainer that exercises the
whole stack on synthetic-but-real SMILES chemistry (RDKit-verified paths when
available):

  * RunMonitor      — logging, LR/grad/VRAM tracking, plots, per-level evals
  * mol_metrics     — reaction accuracy, top-k accuracy, synthon EM + Tanimoto,
                      generation validity/uniqueness/novelty/diversity
  * curriculum      — leakage-safe grouped split, 9 difficulty levels,
                      milestone-driven difficulty-aware sampling, per-level
                      metric tracking
  * nan_shield      — NaN localization, streak abort, verified checkpoints,
                      rollback + forensics (the epoch-6 failure mode)

Nothing here touches your real codebase — this file is the verification
sandbox. `python prototype_test.py` prints a PASS/FAIL table and exits non-zero
on any failure. The tests include the two non-interference guarantees you
asked for: (1) bit-identical training with/without monitoring, and (2) a
measured per-step overhead budget.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import re
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "runs")

from train_observatory import RunMonitor, RunReport, compare_runs  # noqa: E402
import mol_metrics as mm                                             # noqa: E402
from curriculum import (grouped_split, assert_no_leakage, difficulty_scores,  # noqa: E402
                        assign_levels, CurriculumSampler, LevelTracker,
                        molecular_difficulty_features)
from nan_shield import (NanShield, SystemicNanError, finite,      # noqa: E402
                        verify_checkpoint, params_finite, grads_finite)

SEED = 0
N_RXN = 8
BATCH = 32

# =============================================================== toy chemistry
ATOMS = ["C", "N", "O", "S"]


def rand_molecule(rng, min_len=4, max_len=9):
    """Random simple organic chain (optionally one branch) — always valid SMILES."""
    n = rng.randint(min_len, max_len)
    smi = "".join(rng.choice(ATOMS) for _ in range(n))
    if rng.random() < 0.35 and n >= 4:
        pos = rng.randint(1, n - 2)
        bl = rng.randint(1, 2)
        smi = smi[:pos] + "(" + "".join(rng.choice(ATOMS) for _ in range(bl)) + ")" + smi[pos:]
    return smi


def synthon_cuts(smi):
    """All main-chain cut points: split the molecule into two valid synthons."""
    return [i for i in range(1, len(smi)) if smi[i - 1] in ATOMS and smi[i] in ATOMS]


def rxn_class(smi):
    """Synthetic reaction class: composition rule (learnable by construction)."""
    return (1 if "N" in smi else 0) * 4 + (1 if "O" in smi else 0) * 2 + (1 if "(" in smi else 0)


def make_records(n_molecules, seed):
    """Each parent molecule yields 2-3 records with DIFFERENT synthon cuts —
    a 'conformation family' analogue so grouped (leakage-safe) splitting is
    meaningful: the model cannot see a sibling cut of the same molecule in
    both train and validation."""
    rng = random.Random(seed)
    recs = []
    for _ in range(n_molecules):
        smi = rand_molecule(rng)
        cuts = synthon_cuts(smi)
        if not cuts:
            continue
        k = min(len(cuts), rng.choice([2, 2, 3]))
        for i in rng.sample(cuts, k):
            recs.append({"smiles": smi, "synthon_a": smi[:i], "synthon_b": smi[i:],
                         "rxn": rxn_class(smi), "cut": i})
    return recs


# ================================================================== vocab/tensors
VOCAB = ["<pad>", "<unk>"] + sorted(set("".join(r["smiles"] for r in make_records(400, 12345))))
CH2I = {c: i for i, c in enumerate(VOCAB)}
PAD, UNK = 0, 1


def encode(s, maxlen=None):
    ids = [CH2I.get(c, UNK) for c in s]
    if maxlen is not None:
        ids = ids[:maxlen] + [PAD] * max(0, maxlen - len(ids))
    return ids


def pad_to(seqs):
    L = max(len(s) for s in seqs)
    return [s + [PAD] * (L - len(s)) for s in seqs]


# ====================================================================== the model
class MiniSynTree(nn.Module):
    """Multi-task mini-'SynTree': reaction classification + synthon ranking +
    synthon generation (masked-token repair). Stands in for your real model;
    the metrics and monitoring are the deliverable, not this little net."""

    def __init__(self, vocab_size, n_rxn=N_RXN, hid=96):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 32)
        self.trunk = nn.Sequential(nn.Linear(32, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        self.head_rxn = nn.Linear(hid, n_rxn)
        self.head_rank = nn.Sequential(nn.Linear(hid + 32, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.head_gen = nn.Linear(64, vocab_size)

    def encode_target(self, ids):
        e = self.emb(ids)
        return self.trunk(e.mean(dim=1)), e

    def rank_scores(self, h, cand_ids):
        # h: [B, hid]; cand_ids: [B, C, L] -> scores [B, C]
        B, C, L = cand_ids.shape
        ce = self.emb(cand_ids.reshape(B * C, L)).mean(dim=1).reshape(B, C, 32)
        hh = h.unsqueeze(1).expand(B, C, h.shape[1])
        return self.head_rank(torch.cat([hh, ce], dim=-1)).squeeze(-1)

    def gen_logits(self, ids):
        out, _ = self.gru(self.emb(ids))
        return self.head_gen(out)


def corrupt_tokens(ids, rng, p=0.25):
    """Mask p of the tokens (the 'repair' generation task input)."""
    out = list(ids)
    for i in range(len(out)):
        if out[i] != PAD and rng.random() < p:
            out[i] = UNK
    return out


# ================================================================== global dataset
def build_world(n_molecules=400, seed=12345):
    recs = make_records(n_molecules, seed)
    # negatives pool for the ranking task (other molecules' synthons)
    rng = random.Random(999)
    pool = sorted({r["synthon_a"] for r in recs} | {r["synthon_b"] for r in recs})
    for r in recs:
        negs = []
        while len(negs) < 4:
            cand = rng.choice(pool)
            if cand != r["synthon_a"]:
                negs.append(cand)
        r["negatives"] = negs
    return recs


WORLD = build_world()


def split_world():
    """Leakage-safe grouped split: all cuts of one parent molecule stay together."""
    tr, va, te, parts = grouped_split(
        WORLD, key_fn=lambda r: r["smiles"], val_frac=0.15, test_frac=0.05, seed=SEED)
    assert_no_leakage(WORLD, lambda r: r["smiles"], (tr, va, te))
    return tr, va, te


TRAIN_IDX, VAL_IDX, TEST_IDX = split_world()


def difficulty_of_world():
    feats = [molecular_difficulty_features(r["smiles"]) for r in WORLD]
    keys = sorted(feats[0])
    # normalize per feature
    norm = {}
    for k in keys:
        vals = sorted(f[k] for f in feats)
        lo, hi = vals[0], vals[-1]
        norm[k] = (lo, hi if hi > lo else lo + 1e-9)
    scores = []
    for f in feats:
        num = sum(min(max((f[k] - norm[k][0]) / (norm[k][1] - norm[k][0]), 0.0), 1.0)
                  for k in keys)
        scores.append(num / len(keys))
    return scores


DIFF_SCORES = difficulty_of_world()
DIFF_LEVELS, _EDGES = assign_levels(DIFF_SCORES, n_levels=9)
LEVEL_OF = {i: DIFF_LEVELS[i] for i in range(len(WORLD))}

TRAIN_SMILES_POOL = [r["smiles"] for i in TRAIN_IDX for r in [WORLD[i]]]


# ==================================================================== evaluation
@torch.no_grad()
def evaluate(model, idxs, rng_seed=4242):
    """Full domain-metrics evaluation on a set of record indices. Returns
    (overall_metrics_dict, per_sample dict arrays, val_loss) — every value in [0,1]."""
    model.eval()
    rng = random.Random(rng_seed)
    recs = [WORLD[i] for i in idxs]
    n = len(recs)
    if n == 0:
        return {}, {}, float("nan")
    L = max(len(r["smiles"]) for r in recs)

    rxn_logits, rxn_targets = [], []
    rank_scores_all, gen_ranked, gen_tops = [], [], []
    sample_level = [LEVEL_OF.get(i, 0) for i in idxs]
    per_sample = {"rxn_hit": [], "rank_hit": [], "em_top1": [], "loss": [], "tanimoto": []}
    vloss_sum, vloss_n = 0.0, 0

    for bi in range(0, n, 64):
        batch = recs[bi:bi + 64]
        tids = torch.tensor([encode(r["smiles"], maxlen=L) for r in batch])
        rxny = torch.tensor([r["rxn"] for r in batch])
        h, _ = model.encode_target(tids)
        logits = model.head_rxn(h)

        # ranking: candidates = [synthon_a (positive), 4 negatives]
        cands = [[r["synthon_a"]] + r["negatives"] for r in batch]
        CL = max(len(c) for cs in cands for c in cs)
        cids = torch.tensor([[encode(c, maxlen=CL) for c in cs] for cs in cands])
        scores = model.rank_scores(h, cids)                       # [B, 5]

        # generation: repair corrupted synthon_a, 3 stochastic variants
        variants = [[] for _ in batch]
        var_scores = [[] for _ in batch]
        for seed in range(3):
            gsrc = [corrupt_tokens(encode(r["synthon_a"]), rng) for r in batch]
            gids = torch.tensor(pad_to(gsrc))
            gl = model.gen_logits(gids)                            # [B, L, V]
            logp = torch.log_softmax(gl, dim=-1)
            pred = gl.argmax(dim=-1)                               # [B, L]
            conf = []
            for j in range(len(batch)):
                toks = []
                lp = 0.0
                cnt = 0
                for pos in range(len(gsrc[j])):
                    if gsrc[j][pos] == UNK:
                        t = pred[j, pos].item()
                        lp += float(logp[j, pos, t])
                        cnt += 1
                        toks.append(VOCAB[t])
                    else:
                        toks.append(VOCAB[gsrc[j][pos]])
                s = "".join(t for t in toks if t != "<pad>")
                variants[j].append(s)
                var_scores[j].append(lp / cnt if cnt else -99.0)

        # loss on this val batch (for the val_loss curve)
        tgt_gen = torch.tensor(pad_to([encode(r["synthon_a"]) for r in batch]))
        src_gen = torch.tensor(pad_to([corrupt_tokens(encode(r["synthon_a"]), rng)
                                       for r in batch]))
        gl2 = model.gen_logits(src_gen)
        l_rxn = F.cross_entropy(logits, rxny)
        l_rank = F.cross_entropy(scores, torch.zeros(len(batch), dtype=torch.long))
        l_gen = F.cross_entropy(gl2.reshape(-1, gl2.shape[-1]), tgt_gen.reshape(-1),
                                ignore_index=PAD)
        vloss_sum += float(l_rxn + l_rank + l_gen)
        vloss_n += 1

        rxn_logits.append(logits)
        rxn_targets.append(rxny)
        rank_scores_all.append(scores)
        for j, r in enumerate(batch):
            per_sample["rxn_hit"].append(int(logits[j].argmax().item() == r["rxn"]))
            per_sample["rank_hit"].append(int(scores[j].argmax().item() == 0))
            order = sorted(range(3), key=lambda k: -var_scores[j][k])
            gen_ranked.append([variants[j][k] for k in order])
            gen_tops.append(variants[j][order[0]])
            per_sample["em_top1"].append(int(mm.exact_match(variants[j][order[0]],
                                                            r["synthon_a"])))
            t_val = None
            if mm.is_valid(variants[j][order[0]]) and mm.is_valid(r["synthon_a"]):
                t_val = mm.tanimoto(mm.fingerprint(variants[j][order[0]]),
                                    mm.fingerprint(r["synthon_a"]))
            per_sample["tanimoto"].append(t_val)

    all_logits = torch.cat(rxn_logits)
    all_targets = torch.cat(rxn_targets)
    all_scores = torch.cat(rank_scores_all)
    targets_a = [r["synthon_a"] for r in recs]

    m = mm.reaction_metrics(all_logits, all_targets, ks=(1, 3, 5))
    rank = mm.topk_accuracy(all_scores, torch.zeros(n, dtype=torch.long), ks=(1, 3, 5))
    for k, v in rank.items():
        m[f"synthon_rank_acc@{k.split('top')[1]}"] = v
    sm = mm.synthon_metrics(gen_ranked, targets_a, ks=(1, 3))
    m.update(sm)
    gm = mm.generation_metrics(gen_tops, targets=targets_a, reference_pool=TRAIN_SMILES_POOL)
    m.update(gm)
    m["val_loss_proxy"] = vloss_sum / max(1, vloss_n)
    model.train()
    return m, per_sample, vloss_sum / max(1, vloss_n)


# ==================================================================== training loop
def lr_factor(t, warmup, total):
    if t < warmup:
        return (t + 1) / warmup
    p = (t - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def train_once(monitor=None, shield=None, sampler=None, epochs=5, peak_lr=3e-3,
               warmup=15, batch=BATCH, injects=None, run_eval=True,
               ckpt_every=None, start_note=""):
    """One guarded training run. Returns a result dict.
    injects: set of (epoch, batch) where the gen loss is corrupted to NaN —
    simulating the epoch-6 torsion bug — to prove detection/skip/abort/rollback.
    """
    injects = set(injects or [])
    n_train = len(TRAIN_IDX)
    n_batches = math.ceil(n_train / batch)
    total_sched = epochs * n_batches
    torch.manual_seed(7)
    model = MiniSynTree(len(VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: lr_factor(t, warmup, total_sched))
    rng = random.Random(77)
    train_loss_hist, steps_logged = [], 0
    result = {"final_last10": None, "steps": 0, "epochs_done": 0,
              "aborted": None, "rollback": False}
    ckpt_dir = None
    if monitor is not None:
        ckpt_dir = os.path.join(str(monitor.run_dir), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

    def run_epoch(ep):
        nonlocal steps_logged
        idxs = sampler.sample(n_train) if sampler is not None else \
            torch.randperm(n_train, generator=torch.Generator().manual_seed(123)).tolist()
        for bi in range(0, n_train, batch):
            bidx = idxs[bi:bi + batch]
            recs = [WORLD[i] for i in bidx]
            L = max(len(r["smiles"]) for r in recs)
            tids = torch.tensor([encode(r["smiles"], maxlen=L) for r in recs])
            rxny = torch.tensor([r["rxn"] for r in recs])
            cands = [[r["synthon_a"]] + r["negatives"] for r in recs]
            CL = max(len(c) for cs in cands for c in cs)
            cids = torch.tensor([[encode(c, maxlen=CL) for c in cs] for cs in cands])
            tgt_gen = torch.tensor(pad_to([encode(r["synthon_a"]) for r in recs]))
            src_gen = torch.tensor(pad_to([corrupt_tokens(encode(r["synthon_a"]), rng)
                                           for r in recs]))

            h, _ = model.encode_target(tids)
            logits = model.head_rxn(h)
            scores = model.rank_scores(h, cids)
            gl = model.gen_logits(src_gen)
            loss_rxn = F.cross_entropy(logits, rxny)
            loss_rank = F.cross_entropy(scores, torch.zeros(len(recs), dtype=torch.long))
            loss_gen = F.cross_entropy(gl.reshape(-1, gl.shape[-1]), tgt_gen.reshape(-1),
                                       ignore_index=PAD)
            if (ep, bi // batch) in injects:
                loss_gen = torch.tensor(float("nan"))   # simulated torsion NaN
            components = {"reaction_ce": float(loss_rxn.detach()),
                          "synthon_ce": float(loss_rank.detach()),
                          "gen_ce": float(loss_gen.detach())}
            loss = loss_rxn + loss_rank + loss_gen
            batch_acc = float((logits.argmax(dim=1) == rxny).float().mean())

            action, gn = None, None
            if shield is not None:
                action, gn = shield.guarded_step(
                    loss, model, opt, scaler=None, clip=1.0, scheduler=sched,
                    components=components, epoch=ep, batch=bi // batch)
                if action == "abort":
                    path = shield.forensics(ep, bi // batch, components, model)
                    raise SystemicNanError(
                        f"systemic NaN at epoch {ep} batch {bi // batch}; forensics -> {path}")
                if action in ("skipped", "skip_batch"):
                    continue
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

            steps_logged += 1
            train_loss_hist.append(float(loss.detach()))
            if monitor is not None:
                gn_f = None if gn is None else float(gn.detach() if hasattr(gn, "detach") else gn)
                monitor.step(epoch=ep, batch=bi // batch, n_batches=n_batches,
                             loss=float(loss.detach()), lr=sched.get_last_lr()[0],
                             grad_norm=gn_f,
                             batch_size=len(recs), acc=batch_acc,
                             components=components)
            if math.isfinite(float(loss.detach())):
                result["final_last10"] = sum(train_loss_hist[-10:]) / \
                    len(train_loss_hist[-10:])

    ep = 0
    while ep < epochs:
        ctx = monitor.epoch(ep, n_batches=n_batches) if monitor is not None \
            else contextlib.nullcontext()
        try:
            with ctx:
                run_epoch(ep)
        except SystemicNanError as e:
            result["aborted"] = {"epoch": ep, "msg": str(e)}
            if monitor is not None:
                monitor.error("systemic NaN — stopping epoch", exc=e, epoch=ep)
            if shield is None or shield.last_known_good is None:
                raise
            ok = shield.rollback(model, opt, sched)
            result["rollback"] = bool(ok)
            if not ok:
                raise
            injects.clear()          # the fault injection has served its purpose
            continue                 # replay the same epoch from the clean checkpoint
        ep += 1
        # ---- end-of-epoch: eval + curriculum + checkpoint -------------------
        if run_eval and monitor is not None:
            metrics, per_sample, vloss = evaluate(model, VAL_IDX)
            overall = {k: v for k, v in metrics.items()
                       if isinstance(v, (int, float))}
            monitor.validation(ep - 1, val_loss=vloss,
                               val_acc=overall.get("reaction_acc"))
            monitor.evaluation(ep - 1, overall)
            tracker = LevelTracker()
            # per-sample arrays are aligned with VAL_IDX
            for j, i in enumerate(VAL_IDX):
                lv = LEVEL_OF.get(i)
                if not lv:
                    continue
                tracker.add_ratio(lv, "reaction_acc", per_sample["rxn_hit"][j], 1)
                tracker.add_ratio(lv, "synthon_EM", per_sample["em_top1"][j], 1)
                tracker.add_scalar(lv, "tanimoto_mean", per_sample["tanimoto"][j])
                tracker.add_ratio(lv, "loss_proxy", 1 - per_sample["rank_hit"][j], 1)
            for key, v in tracker.summary().items():
                m2 = re.match(r"^L(\d+)/(.+)$", key)
                if m2:
                    monitor.evaluation(ep - 1, {m2.group(2): v}, level=int(m2.group(1)))
            if sampler is not None:
                adv = sampler.maybe_advance({"val_loss": vloss})
                monitor.event("curriculum_stage", advanced=bool(adv),
                              **sampler.state())
        if monitor is not None and ckpt_every is not None and ep % ckpt_every == 0:
            path = os.path.join(ckpt_dir, f"epoch{ep}.pt")
            if shield is not None:
                shield.guarded_save(path, model, opt, sched, epoch=ep)
            else:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(), "epoch": ep}, path)
                monitor.checkpoint_saved(path, epoch=ep, health="unverified")
        result["epochs_done"] = ep

    result["steps"] = steps_logged
    result["model"] = model
    return result


# ================================================================ self-test suite
import csv as _csv
from collections import Counter


def jsonl(d, name):
    out = []
    with open(os.path.join(str(d), name), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def csv_rows(d, name):
    with open(os.path.join(str(d), name), newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def is_png(p, min_kb=4):
    if not os.path.exists(p):
        return False
    with open(p, "rb") as f:
        head = f.read(8)
    return head == b"\x89PNG\r\n\x1a\n" and os.path.getsize(p) > min_kb * 1024


G = {}


# ------------------------------------------------------------------ unit tests
def t_topk():
    r = mm.topk_accuracy([[3, 1, 2], [0, 5, 4]], [0, 2], ks=(1, 2))
    assert abs(r["top1"] - 0.5) < 1e-9, r
    assert abs(r["top2"] - 1.0) < 1e-9, r
    rt = mm.topk_accuracy(torch.tensor([[3.0, 1.0, 2.0]]), torch.tensor([0]), ks=(1,))
    assert abs(rt["top1"] - 1.0) < 1e-9
    return "top1=0.5 top2=1.0 (torch path ok)"


def t_tanimoto():
    a, b, c = {"a", "b", "c"}, {"a", "b", "d"}, {"x", "y"}
    assert mm.tanimoto(a, a) == 1.0
    assert abs(mm.tanimoto(a, b) - 0.5) < 1e-9
    assert mm.tanimoto(a, c) == 0.0
    return "identical=1.0 overlap=0.5 disjoint=0.0"


def t_exact_match():
    assert mm.exact_match("CCO", "CCO")
    if mm._HAS_RDKIT:
        assert mm.exact_match("CCO", "OCC"), "RDKit canonicalization expected"
        return "RDKit canonical: CCO == OCC"
    assert not mm.exact_match("CCO", "OCC")
    return "fallback (no RDKit): raw-string equality"


def t_validity():
    v = mm.validity_rate(["CCO", "C1CC1", "CXP9"])
    assert abs(v - 2 / 3) < 1e-9, v
    return "validity 2/3 (CXP9 rejected on both chemistry backends)"


def t_uniq_novelty():
    assert abs(mm.uniqueness_rate(["CCO", "CCO", "CCN"]) - 2 / 3) < 1e-9
    assert abs(mm.novelty_rate(["CCO", "CCN"], ["CCO", "CCC"]) - 0.5) < 1e-9
    return "uniqueness 2/3, novelty 1/2"


def t_diversity():
    d0 = mm.internal_diversity(["CCCC", "CCCC"])
    d1 = mm.internal_diversity(["CCCC", "NNNN"])
    assert d0 < 1e-6, d0
    assert d1 > 0.9, d1
    return f"identical={d0:.2f} disjoint~{d1:.2f}"


def t_metric_aggregators():
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])
    rm = mm.reaction_metrics(logits, [0, 1], ks=(1, 3))
    assert abs(rm["reaction_acc"] - 1.0) < 1e-9 and "reaction_topk@3" in rm
    sm = mm.synthon_metrics([["CCO", "CCC"], ["NNN", "CCO", "CCN"]], ["CCO", "CCO"],
                           ks=(1, 3))
    assert abs(sm["synthon_EM"] - 0.5) < 1e-9
    assert abs(sm["synthon_topk_EM@3"] - 1.0) < 1e-9
    assert 0.0 <= sm["synthon_tanimoto_mean"] <= 1.0
    return f"reaction_acc={rm['reaction_acc']:.2f} synthon_EM={sm['synthon_EM']:.2f} topk_EM@3={sm['synthon_topk_EM@3']:.2f}"


def t_grouped_split():
    assert_no_leakage(WORLD, lambda r: r["smiles"], (TRAIN_IDX, VAL_IDX, TEST_IDX))
    assert TRAIN_IDX and VAL_IDX and TEST_IDX
    assert not (set(TRAIN_IDX) & set(VAL_IDX))
    # negative control: a naive random split DOES leak sibling cuts
    rng = random.Random(1)
    idx = list(range(len(WORLD)))
    rng.shuffle(idx)
    naive_val = set(idx[:120])
    val_parents = {WORLD[i]["smiles"] for i in naive_val}
    leaks = sum(1 for i in idx[120:] if WORLD[i]["smiles"] in val_parents)
    assert leaks > 0, "expected naive split to leak"
    # reaction-class coverage in validation (needed for meaningful eval)
    cov = len({WORLD[i]["rxn"] for i in VAL_IDX})
    assert cov >= 6, cov
    return f"0 family leaks (naive split leaks {leaks}); val covers {cov}/8 reaction classes"


def t_levels():
    cnt = Counter(LEVEL_OF[i] for i in TRAIN_IDX)
    assert set(cnt) == set(range(1, 10)), cnt
    mx = max(cnt.values()) - min(cnt.values())
    assert mx <= max(3, len(TRAIN_IDX) // 9 // 3), cnt
    return f"9 levels, counts {sorted(cnt.items())}"


def t_curriculum_weights():
    lv = [LEVEL_OF[i] for i in TRAIN_IDX]
    s = CurriculumSampler(TRAIN_IDX, lv, seed=5, n_stages=6, min_weight=0.03, patience=99)
    w = s.weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert min(w.values()) >= 0.03 - 1e-9
    assert max(w, key=w.get) <= 3, "stage 0 must favor easy levels"
    s2 = CurriculumSampler(TRAIN_IDX, lv, seed=5, n_stages=6, patience=1)
    for _ in range(5):
        s2.maybe_advance({})
    w2 = s2.weights()
    assert max(w2, key=w2.get) == max(s2.present_levels), "final stage must favor hardest"
    return f"stage0 argmax=L{max(w, key=w.get)}, stage5 argmax=L{max(w2, key=w2.get)}, floor 0.03 kept"


def t_advance():
    lv = [LEVEL_OF[i] for i in TRAIN_IDX]
    s = CurriculumSampler(TRAIN_IDX, lv, seed=1, milestone_metric="val_loss",
                          improve_thresh=0.1, patience=100)
    assert not s.maybe_advance({"val_loss": 3.0})   # baseline
    assert s.maybe_advance({"val_loss": 2.9})       # improvement milestone
    assert not s.maybe_advance({"val_loss": 2.89})  # re-baselined
    sp = CurriculumSampler(TRAIN_IDX, lv, seed=1, improve_thresh=99.0, patience=2)
    assert not sp.maybe_advance({"val_loss": 1.0})
    assert sp.maybe_advance({"val_loss": 1.0})      # patience path
    return "milestone + patience policies verified"


def t_leveltracker():
    tr = LevelTracker()
    tr.add_scalar(2, "loss", 1.0)
    tr.add_scalar(2, "loss", 3.0)
    tr.add_ratio(2, "reaction_acc", 7, 10)
    tr.add_scalar(9, "loss", None)  # must be ignored
    s = tr.summary()
    assert abs(s["L2/loss"] - 2.0) < 1e-9 and abs(s["L2/reaction_acc"] - 0.7) < 1e-9
    assert "L9/loss" not in s
    return "L2/loss=2.0 L2/reaction_acc=0.7"


def t_shield_localize():
    sh = NanShield(max_consecutive=3)
    a = sh.check_batch(float("nan"), components={"reaction_ce": 1.0, "gen_ce": float("nan")})
    assert a == "skip_batch" and sh.last_localized == "gen_ce", (a, sh.last_localized)
    b = sh.check_batch(0.5, components={"reaction_ce": 1.0, "gen_ce": 0.4})
    assert b == "ok" and sh.streak == 0
    return "localized component=gen_ce; streak resets on recovery"


def t_shield_systemic():
    sh = NanShield(max_consecutive=3)
    assert sh.check_batch(float("nan")) == "skip_batch"
    assert sh.check_batch(float("nan")) == "skip_batch"
    assert sh.check_batch(float("nan")) == "abort"
    return "3 consecutive NaNs -> abort (no infinite skipping)"


def t_grads_helper():
    lin = nn.Linear(3, 2)
    assert grads_finite(lin)[0]
    for p in lin.parameters():
        p.grad = torch.full_like(p, float("nan"))
    ok, name = grads_finite(lin)
    assert not ok and name
    return f"detected {name}"


def t_verify_ckpt():
    lin = nn.Linear(3, 2)
    p1 = os.path.join(RUNS, "_u_good.pt")
    torch.save({"model": lin.state_dict()}, p1)
    ok, rep = verify_checkpoint(p1)
    assert ok and rep["checked_tensors"] >= 2
    p2 = os.path.join(RUNS, "_u_bad.pt")
    torch.save({"model": {"w": torch.tensor([float("nan")])}}, p2)
    ok2, rep2 = verify_checkpoint(p2)
    assert not ok2 and rep2["bad_tensors"]
    return f"healthy ckpt ok ({rep['checked_tensors']} tensors); NaN ckpt rejected"


def t_corruption():
    monC = RunMonitor(run_name="proto_corrupt", run_root=RUNS, sampling_interval=None,
                      verbose=False)
    shC = NanShield(monitor=monC, max_consecutive=3)
    modelC = nn.Linear(4, 4)
    optC = torch.optim.SGD(modelC.parameters(), lr=0.1)
    ck = os.path.join(str(monC.run_dir), "checkpoints", "good.pt")
    ck2 = os.path.join(str(monC.run_dir), "checkpoints", "bad.pt")
    ok, _ = shC.guarded_save(ck, modelC, optC, epoch=0)
    assert ok and os.path.exists(ck + ".health.json") and shC.last_known_good == ck
    with torch.no_grad():
        modelC.weight.data[0, 0] = float("nan")
    assert not shC.check_params(modelC, optC)
    assert "param_corruption" in [e["kind"] for e in jsonl(monC.run_dir, "events.jsonl")]
    ok2, _ = shC.guarded_save(ck2, modelC, optC, epoch=1)
    assert not ok2 and shC.last_known_good == ck  # rejected save must not become known-good
    assert "checkpoint_rejected" in [e["kind"] for e in jsonl(monC.run_dir, "events.jsonl")]
    assert shC.rollback(modelC, optC, path=ck)
    assert params_finite(modelC)[0]
    kinds = [e["kind"] for e in jsonl(monC.run_dir, "events.jsonl")]
    assert "rollback_restored" in kinds and "checkpoint_loaded" in kinds
    monC.finish(plot=False)
    return "NaN params detected; bad ckpt rejected; rollback restored finite weights"


# --------------------------------------------------------------- integration runs
def t_interference():
    lv = [LEVEL_OF[i] for i in TRAIN_IDX]
    cur0 = CurriculumSampler(TRAIN_IDX, lv, seed=5, n_stages=6, patience=99)
    base = train_once(monitor=None, shield=None, sampler=cur0, epochs=5, run_eval=False)
    cur1 = CurriculumSampler(TRAIN_IDX, lv, seed=5, n_stages=6, patience=99)
    monI = RunMonitor(run_name="proto_interference", run_root=RUNS,
                      sampling_interval=None, verbose=False)
    shI = NanShield(monitor=monI, max_consecutive=3)
    pair = train_once(monitor=monI, shield=shI, sampler=cur1, epochs=5, run_eval=False)
    monI.finish(plot=False)
    a, b = base["final_last10"], pair["final_last10"]
    assert a is not None and b is not None
    diff = abs(a - b)
    assert diff < 1e-6, f"monitored run diverged: {a} vs {b} (diff {diff})"
    assert monI.soft_failures == []
    G["base_final"] = a
    return f"final loss {a:.6f} == {b:.6f} (bit-identical, diff {diff:.2e})"


def t_full_run():
    lv = [LEVEL_OF[i] for i in TRAIN_IDX]
    curF = CurriculumSampler(TRAIN_IDX, lv, seed=5, n_stages=6,
                             milestone_metric="val_loss", improve_thresh=0.05, patience=2)
    monF = RunMonitor(run_name="proto_full", run_root=RUNS, sampling_interval=0.5,
                      config={"model": "MiniSynTree", "peak_lr": 3e-3, "warmup": 15,
                              "batch": BATCH, "grad_clip": 1.0, "epochs": 5,
                              "curriculum_levels": 9,
                              "note": "prototype verification run with injected NaNs"})
    shF = NanShield(monitor=monF, max_consecutive=3, param_check_every=10)
    monF.capture_warnings()
    resF = train_once(monitor=monF, shield=shF, sampler=curF, epochs=5, run_eval=True,
                      injects={(1, 1), (2, 2), (2, 3), (2, 4)}, ckpt_every=1)
    assert resF["aborted"] is not None, "expected the systemic NaN injection to abort"
    assert resF["rollback"] is True, "expected rollback to last known-good ckpt"
    assert resF["final_last10"] is not None and math.isfinite(resF["final_last10"])
    assert resF["epochs_done"] == 5
    # monitor-level NaN watchdog demo
    monF.check_value(float("nan"), name="watchdog_demo", epoch=9, batch=99,
                     context="self-test")
    # warning + handled error for the audit trail
    warnings.warn("synthetic deprecation demo")
    monF.warn("synthetic operator warning (handled)")
    try:
        raise ValueError("synthetic OOM-like failure for audit trail")
    except ValueError as e:
        monF.error("synthetic failure (handled, training continues)", exc=e)
    summF = monF.finish("completed")
    G.update(monF=monF, resF=resF, shF=shF, curF=curF, summF=summF,
             E=math.ceil(len(TRAIN_IDX) / BATCH))
    G["dirF"] = str(monF.run_dir)
    return (f"{resF['steps']} steps, 1 skip + 3-NaN abort @ep2, rollback+recovery, "
            f"final loss {resF['final_last10']:.4f}")


def _series(rep, metric):
    xs, ys = [], []
    for e in rep.evaluations:
        if e.get("metric") == metric and e.get("value") is not None:
            xs.append(int(e["epoch"]))
            ys.append(float(e["value"]))
    pairs = sorted(zip(xs, ys))
    return [(x, y) for x, y in pairs]


def t_full_artifacts():
    monF, resF, shF, curF, summF = (G["monF"], G["resF"], G["shF"], G["curF"],
                                    G["summF"])
    d = G["dirF"]
    E = G["E"]
    for f in ["manifest.json", "config.json", "events.jsonl", "metrics.csv",
              "evaluations.csv", "system.csv", "checkpoints.jsonl", "summary.json",
              "report.txt"]:
        assert os.path.exists(os.path.join(d, f)), f"missing {f}"
    assert os.path.exists(os.path.join(d, "nan_forensics", "ep2_b4.json"))
    assert monF.soft_failures == [], f"monitor soft-fails: {monF.soft_failures[:3]}"
    evs = jsonl(d, "events.jsonl")
    kinds = Counter(e["kind"] for e in evs)
    expect = {"run_start": 1, "epoch_start": 6, "epoch_end": 6, "epoch_aborted": 1,
              "error": 2, "numeric_alert": 1, "checkpoint_saved": 5,
              "checkpoint_loaded": 1, "rollback_restored": 1, "nan_localized": 4,
              "systemic_nan_detected": 1, "curriculum_stage": 5, "validation": 5,
              "run_end": 1}
    for k, v in expect.items():
        assert kinds.get(k, 0) == v, f"event {k}: {kinds.get(k, 0)} != {v} (kinds={kinds})"
    assert kinds.get("evaluation", 0) >= 100
    assert kinds.get("warning", 0) >= 2
    errs = [e for e in evs if e["kind"] == "error"]
    assert len(errs) == 2
    assert any("ValueError" in e.get("traceback", "") and "synthetic" in e.get("message", "")
               for e in errs)
    assert any("SystemicNanError" in e.get("traceback", "") for e in errs)
    fj = json.load(open(os.path.join(d, "nan_forensics", "ep2_b4.json")))
    assert fj["first_failure"]["component"] == "gen_ce", fj["first_failure"]
    assert len(fj.get("component_history", {}).get("gen_ce", [])) >= 3
    # metrics.csv integrity + exact LR schedule
    # NOTE: the single injected NaN batch (epoch 1) is skipped -> NOT logged and the
    # scheduler is NOT stepped for it, so epoch 1 contributes E-1 rows and the LR
    # timeline compresses by one scheduler position. Expected rows = 5E + 1.
    rows = csv_rows(d, "metrics.csv")
    assert len(rows) == resF["steps"] == 5 * E + 1, (len(rows), resF["steps"], 5 * E + 1)
    gs = [int(r["global_step"]) for r in rows]
    assert gs == list(range(1, len(rows) + 1))
    losses = [float(r["loss"]) for r in rows]
    assert all(math.isfinite(v) for v in losses)
    assert all(math.isfinite(float(r["grad_norm"])) for r in rows if r["grad_norm"])
    assert "components" in json.loads(rows[0]["extra"])

    def f(s):
        return 3e-3 * lr_factor(s, 15, 5 * E)
    exp = ([f(s) for s in range(1, E + 1)] +            # epoch 0: E steps
           [f(s) for s in range(E + 1, 2 * E)] +       # epoch 1: E-1 steps (1 NaN skip)
           [f(2 * E), f(2 * E + 1)] +                  # epoch 2 attempt: 2 steps before abort
           [f(s) for s in range(2 * E, 3 * E)] +        # epoch 2 replayed after rollback
           [f(s) for s in range(3 * E, 4 * E)] +        # epoch 3
           [f(s) for s in range(4 * E, 5 * E)])        # epoch 4
    logged = [float(r["lr"]) for r in rows]
    assert len(logged) == len(exp), (len(logged), len(exp))
    dmax = max(abs(a - b) for a, b in zip(logged, exp))
    assert dmax < 1e-9, dmax
    assert abs(max(logged) - 3e-3) < 1e-12, "peak lr must equal config peak"
    sysr = csv_rows(d, "system.csv")
    assert len(sysr) >= 1
    # evaluations + learning
    rep = RunReport(d)
    assert len(rep.evaluations) >= 200, len(rep.evaluations)
    rxn = _series(rep, "reaction_acc")
    assert len(rxn) == 5 and rxn[-1][1] >= 0.75 and rxn[-1][1] > rxn[0][1], rxn
    rank = _series(rep, "synthon_rank_acc@1")
    assert rank[-1][1] >= 0.40 and rank[-1][1] > rank[0][1], rank
    vals = sorted((e["epoch"], e["val_loss"]) for e in rep.validations
                  if isinstance(e.get("val_loss"), (int, float)))
    assert len(vals) == 5 and vals[-1][1] < vals[0][1], vals
    em = _series(rep, "synthon_EM")
    assert em and 0.05 <= em[-1][1] <= 1.0, em
    for name in ["gen_validity", "gen_uniqueness", "gen_novelty", "gen_diversity",
                 "gen_rediscovery", "synthon_tanimoto_mean", "reaction_topk@3",
                 "reaction_topk@5", "synthon_rank_acc@3", "synthon_rank_acc@5",
                 "synthon_topk_EM@3"]:
        s = _series(rep, name)
        assert s and all(0.0 <= v <= 1.0 for _, v in s), f"{name}: {s}"
    lvl = [e for e in rep.evaluations if str(e.get("metric", "")).startswith("L")]
    assert len(lvl) >= 100
    assert {"L1", "L9"} <= {str(e["metric"]).split("/")[0] for e in lvl}
    assert curF.stage >= 2, "curriculum must advance (patience guarantee)"
    # summary
    assert summF["status"] == "completed" and summF["best_validation"]
    assert summF["health"]["nan_alerts"] == 1 and summF["checkpoints"] == 5
    assert len(summF["plots"]) >= 9
    assert os.path.exists(os.path.join(d, "checkpoints", "epoch2.pt.health.json"))
    return (f"{len(rows)} rows, LR exact to {dmax:.0e}, rxn_acc {rxn[0][1]:.2f}->"
            f"{rxn[-1][1]:.2f}, rank@1 {rank[0][1]:.2f}->{rank[-1][1]:.2f}, "
            f"{len(rep.evaluations)} eval rows")


def t_full_plots():
    d = G["dirF"]
    names = ["loss.png", "accuracy.png", "lr_schedule.png", "grad_norm.png",
             "throughput.png", "component_losses.png", "evaluation_metrics.png",
             "level_heatmaps.png", "dashboard.png", "resources.png"]
    for n in names:
        p = os.path.join(d, "plots", n)
        assert is_png(p), f"missing/invalid plot {n}"
    return f"{len(names)}/{len(names)} valid PNGs (incl. per-level heatmaps + dashboard)"


def t_overhead():
    monB = RunMonitor(run_name="proto_bench", run_root=RUNS, sampling_interval=None,
                      flush_every=50, verbose=False)
    t0 = time.perf_counter()
    for i in range(2000):
        loss = 1.0 / (i + 1)
        lr = 3e-3
        gn = 0.5
        _ = float(loss) + lr + gn
    t_bare = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(2000):
        monB.step(epoch=0, batch=i % 20, n_batches=20, loss=1.0 / (i + 1), lr=3e-3,
                  grad_norm=0.5, batch_size=32, acc=0.5,
                  components={"reaction_ce": 0.3, "synthon_ce": 0.4, "gen_ce": 0.2})
    t_mon = time.perf_counter() - t0
    monB.finish(plot=False)
    per_ms = (t_mon - t_bare) / 2000 * 1000
    shB = NanShield(max_consecutive=3)
    t0 = time.perf_counter()
    for i in range(2000):
        shB.check_batch(0.5, components={"reaction_ce": 0.3, "synthon_ce": 0.4,
                                         "gen_ce": 0.2})
    t_sh = time.perf_counter() - t0
    sh_ms = t_sh / 2000 * 1000
    G["overhead_ms"] = per_ms
    G["shield_ms"] = sh_ms
    assert per_ms < 2.0, f"monitor overhead {per_ms:.3f} ms/step exceeds budget"
    assert sh_ms < 1.0, f"shield overhead {sh_ms:.3f} ms/batch exceeds budget"
    return f"monitor {per_ms:.3f} ms/step (budget 2.0), shield {sh_ms:.3f} ms/batch (budget 1.0)"


def t_crash_recovery():
    mon3 = RunMonitor(run_name="proto_crash", run_root=RUNS, sampling_interval=None,
                      verbose=False)
    for i in range(3):
        mon3.step(epoch=0, batch=i, loss=1.0 - 0.1 * i, lr=1e-3, grad_norm=0.4)
    mon3.flush()  # no finish() — simulate a hard crash after buffered writes
    rep3 = RunReport(str(mon3.run_dir))
    assert len(rep3.metrics) == 3
    assert rep3.best_val() is None and not rep3.summary
    gen3 = rep3.plot_all(os.path.join(str(mon3.run_dir), "plots_reloaded"))
    assert "loss" in gen3
    return f"RunReport recovered {len(rep3.metrics)} partial rows and re-plotted"


def t_compare():
    monL = RunMonitor(run_name="proto_lowLR", run_root=RUNS, sampling_interval=None,
                      verbose=False)
    resL = train_once(monitor=monL, shield=None, sampler=None, epochs=1,
                      peak_lr=3e-4, run_eval=True)
    assert resL["final_last10"] is not None
    monL.finish("completed")
    out = os.path.join(RUNS, "comparison.png")
    tbl = compare_runs([G["dirF"], str(monL.run_dir)], out=out)
    assert len(tbl) == 2 and os.path.exists(out)
    assert is_png(out, min_kb=8)
    assert os.path.exists(out.replace(".png", "_summary.txt"))
    return f"comparison.png + summary table for {len(tbl)} runs (lowLR final {resL['final_last10']:.3f})"


# ------------------------------------------------------------------------ runner
def main():
    t_all = time.perf_counter()
    os.makedirs(RUNS, exist_ok=True)
    print("=" * 100)
    print("PROTOTYPE SELF-TEST — training observatory + domain metrics + curriculum + NaN shield")
    print("=" * 100)
    print(f"world: {len(WORLD)} records | train {len(TRAIN_IDX)} / val {len(VAL_IDX)} / "
          f"test {len(TEST_IDX)} | rxn classes in val: "
          f"{len({WORLD[i]['rxn'] for i in VAL_IDX})}/8 | RDKit: {mm._HAS_RDKIT}")
    print("-" * 100)
    tests = [
        ("mol_metrics.topk_accuracy (hand-checked + tensor input)", t_topk),
        ("mol_metrics.tanimoto (identical / overlap / disjoint)", t_tanimoto),
        ("mol_metrics.exact_match canonicalization", t_exact_match),
        ("mol_metrics.validity (CCO / C1CC1 / CXP9)", t_validity),
        ("mol_metrics.uniqueness + novelty", t_uniq_novelty),
        ("mol_metrics.internal_diversity", t_diversity),
        ("reaction_metrics + synthon_metrics aggregators", t_metric_aggregators),
        ("grouped_split — zero family leakage (naive split leaks)", t_grouped_split),
        ("assign_levels — 9 balanced difficulty levels", t_levels),
        ("CurriculumSampler weights — easy start, hard end, floor kept", t_curriculum_weights),
        ("CurriculumSampler.maybe_advance — milestone + patience", t_advance),
        ("LevelTracker per-level summary keys", t_leveltracker),
        ("NanShield.check_batch — component localization + recovery", t_shield_localize),
        ("NanShield.check_batch — systemic abort (no infinite skipping)", t_shield_systemic),
        ("NanShield.grads_finite helper", t_grads_helper),
        ("verify_checkpoint — healthy vs NaN tensors", t_verify_ckpt),
        ("NanShield — corruption detected, bad ckpt rejected, rollback restores", t_corruption),
        ("NO INTERFERENCE — identical training with/without monitoring+shield", t_interference),
        ("FULL RUN — NaN injection -> abort -> rollback -> recovery", t_full_run),
        ("FULL RUN — artifacts, event trail, exact LR, learning happened", t_full_artifacts),
        ("FULL RUN — all plots valid PNGs", t_full_plots),
        ("OVERHEAD — monitor < 2 ms/step, shield < 1 ms/batch", t_overhead),
        ("CRASH RECOVERY — RunReport loads unfinished run", t_crash_recovery),
        ("compare_runs — cross-run overlay + summary table", t_compare),
    ]
    results = []
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            detail = fn() or ""
            ok = True
            print(f"PASS  {name}\n        {detail}  [{time.perf_counter() - t0:.1f}s]")
        except Exception as e:  # noqa: BLE001
            import traceback
            ok = False
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
            traceback.print_exc()
        results.append((name, ok))
    print("-" * 100)
    npass = sum(1 for _, ok in results if ok)
    print(f"RESULT: {npass}/{len(results)} PASSED  "
          f"(total {time.perf_counter() - t_all:.0f}s)")
    if "overhead_ms" in G:
        print(f"measured monitoring overhead: {G['overhead_ms']:.3f} ms/step | "
              f"shield: {G['shield_ms']:.3f} ms/batch")
    if npass == len(results):
        print(f"full run artifacts: {G.get('dirF')}")
    print("=" * 100)
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()

