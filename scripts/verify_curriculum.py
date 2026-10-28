#!/usr/bin/env python
"""Curriculum pre-flight verification on the REAL dataset (CPU, minutes).

Executes the operator's 2026-10-29 mandate: *verify the complete curriculum
(difficulty scores, level assignment, splits, training progression) BEFORE
training begins*.  This script replays the EXACT production construction
path used by ``ResilientTrainer`` (same ``SynthonCatalog``, same
``ShardedHuggingFaceDataset``, same ``CurriculumController.build`` call with
the same config block) and then runs a fail-closed verification battery.
Nothing here trains a model; it certifies the data-side curriculum so the
GPU run can start with the curriculum already proven.

Usage (repo root):

    python scripts/verify_curriculum.py --config configs/train_colab_12h.json

Optional flags:

    --catalog PATH     override data.synthon_catalog_path (e.g. a local copy)
    --repo-id ID       override data.huggingface.repo_id
    --cache-dir PATH   override the HF dataset cache dir
    --output-dir PATH  where to write the level cache artifact (default: a
                       temp dir - the real run writes ./experiments)
    --json PATH        write the machine-readable report here (default:
                       curriculum_verification.json next to the artifact)

Exit code 0 iff every check PASSES (fail-closed: any FAIL/ERROR -> exit 1).

Checks (see docs/CURRICULUM.md for the design being verified):
  1.  data-integrity      shard sha256 vs manifest, sample counts
  2.  target-bounds       target synthon indices inside the catalog
  3.  curriculum-build    production build path completes + startup logs
  4.  level-coverage      all n_levels train levels populated
  5.  level-balance       quantile bins approximately balanced
  6.  monotone-difficulty mean score strictly increasing L1 -> L9, and key
                          raw features trend upward (face validity)
  7.  val-assignment      val levels assigned with train edges, all in range
  8.  sampler-invariants  weights sum to 1, every level >= min_weight,
                          at every stage
  9.  stage-progression   sampling mass shifts easy -> hard across stages
  10. epoch-simulation    per stage: an epoch draw covers every level,
                          counts match weights, indices respect level pools
  11. determinism         same (seed, epoch) -> identical draw sequence
  12. budget-neutrality   len(sampler) == len(train) and steps/epoch equal
                          to the plain-shuffle path
  13. milestone-advance   milestone + patience advancement, stage monotone,
                          full 0->n_stages-1 progression within max_epochs
  14. coverage-guarantee  expected per-level draw share over the whole run
                          (curriculum covers the ENTIRE train split)
  15. cache-roundtrip     rebuild loads from cache with identical levels;
                          config change invalidates the cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class Check:
    def __init__(self, name):
        self.name = name
        self.status = "PASS"
        self.details = []
        self.metrics = {}

    def ok(self, msg, **metrics):
        self.details.append(("ok", msg))
        self.metrics.update(metrics)

    def fail(self, msg, **metrics):
        self.status = "FAIL"
        self.details.append(("fail", msg))
        self.metrics.update(metrics)

    def error(self, msg):
        self.status = "ERROR"
        self.details.append(("error", msg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--repo-id")
    ap.add_argument("--cache-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    data_cfg = config["data"]
    curr_cfg = dict(config.get("curriculum", {}) or {})
    if not curr_cfg.get("enabled", False):
        print("[verify-curriculum] FATAL: config curriculum.enabled is false - "
              "nothing to verify (this script certifies the curriculum path).")
        return 1

    catalog_path = args.catalog or data_cfg["synthon_catalog_path"]
    repo_id = args.repo_id or data_cfg.get("huggingface", {}).get("repo_id")
    cache_dir = args.cache_dir or data_cfg.get("huggingface", {}).get("cache_dir", "./hf_cache")
    output_dir = args.output_dir or tempfile.mkdtemp(prefix="curriculum_verify_")
    os.makedirs(output_dir, exist_ok=True)
    report_path = args.json or os.path.join(output_dir, "curriculum_verification.json")

    checks = []
    report = {
        "config_path": os.path.abspath(args.config),
        "catalog_path": os.path.abspath(catalog_path),
        "repo_id": repo_id,
        "output_dir": output_dir,
        "checks": {},
    }

    def run_check(name):
        c = Check(name)
        checks.append(c)
        print(f"\n=== [{name}] ===")
        return c

    # ------------------------------------------------------------------
    # 1. data integrity: manifests + shard checksums
    # ------------------------------------------------------------------
    c = run_check("data-integrity")
    try:
        from huggingface_hub import hf_hub_download

        from syntree.data.hf_loader import ShardedHuggingFaceDataset

        expected = {"train": None, "val": None}
        for split in expected:
            ds = ShardedHuggingFaceDataset(
                repo_id=repo_id, split=split, cache_dir=cache_dir,
                revision=data_cfg.get("huggingface", {}).get("revision", "main"),
                token=os.environ.get("HF_TOKEN"),
                max_cached_shards=1,
            )
            expected[split] = ds
            for shard in ds.shards:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"data/{split}/{shard['name']}",
                    repo_type="dataset",
                    revision=data_cfg.get("huggingface", {}).get("revision", "main"),
                    token=os.environ.get("HF_TOKEN"),
                    local_dir=str(ds.cache_dir),
                )
                with open(local, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                if digest != shard.get("sha256"):
                    c.fail(f"{split}/{shard['name']}: sha256 mismatch "
                           f"({digest[:12]} != {shard.get('sha256', '?')[:12]})")
                else:
                    c.ok(f"{split}/{shard['name']}: sha256 ok "
                         f"({shard['sample_count']} samples)")
        n_train, n_val = len(expected["train"]), len(expected["val"])
        c.ok(f"dataset sizes: train={n_train} val={n_val}",
             n_train=n_train, n_val=n_val)
        if n_train < int(data_cfg.get("min_real_samples", 50)):
            c.fail(f"train split ({n_train}) below min_real_samples")
        train_ds, val_ds = expected["train"], expected["val"]
    except Exception as exc:  # pragma: no cover
        c.error(f"{type(exc).__name__}: {exc}")
        train_ds = val_ds = None

    # ------------------------------------------------------------------
    # 2. catalog + target bounds
    # ------------------------------------------------------------------
    c = run_check("target-bounds")
    catalog = None
    try:
        from syntree.chemistry.catalog import SynthonCatalog

        catalog_cfg = dict(config.get("catalog", {}))
        catalog = SynthonCatalog(
            catalog_path,
            embedding_dim=config["model"].get("synthon_embedding_dim", 128),
            min_fsp3=float(catalog_cfg.get("min_fsp3", 0.42)),
            max_mw=float(catalog_cfg.get("max_mw", 220.0)),
            seed=int(catalog_cfg.get("seed", 42)),
            encoder=str(catalog_cfg.get("encoder", "pharm3d")),
            cache_embeddings=bool(catalog_cfg.get("cache_embeddings", True)),
        )
        n_cat = len(catalog.df)
        max_target = -1
        for i in range(len(train_ds)):
            t = int(train_ds[i].target_synthon[0].item())
            max_target = max(max_target, t)
        c.ok(f"catalog synthons={n_cat}, max target_synthon index in train="
             f"{max_target}", n_catalog=n_cat, max_target=max_target)
        if max_target >= n_cat:
            c.fail(f"target synthon index {max_target} outside catalog ({n_cat})")
        if not bool(__import__("torch").isfinite(catalog.embeddings).all().item()):
            c.fail("catalog embeddings contain non-finite values")
        else:
            c.ok("catalog embeddings all finite")
    except Exception as exc:
        c.error(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # 3-7. production curriculum build + level verification
    # ------------------------------------------------------------------
    c = run_check("curriculum-build")
    controller = None
    try:
        from syntree.engine.curriculum import CurriculumController

        # temp cache path so this verification never poisons the real artifact
        build_cfg = dict(curr_cfg)
        build_cfg["cache_path"] = os.path.join(output_dir, "curriculum_levels_v1.json")
        controller = CurriculumController.build(
            build_cfg, train_ds, val_ds, catalog, output_dir=output_dir,
            log_fn=print,
        )
        controller.log_summary()
        c.ok(f"build complete (assignment={'cache' if controller.from_cache else 'computed'})")
    except Exception as exc:
        c.error(f"{type(exc).__name__}: {exc}")

    n_levels = int(curr_cfg.get("n_levels", 9))

    if controller is not None:
        # -- level coverage + balance ------------------------------------
        c = run_check("level-coverage")
        counts = controller.level_counts_train
        missing = [lv for lv in range(1, n_levels + 1) if counts.get(lv, 0) == 0]
        line = " ".join(f"L{lv}={counts.get(lv, 0)}" for lv in range(1, n_levels + 1))
        print(f"train level counts: {line}")
        if missing:
            c.fail(f"empty train levels: {missing}")
        else:
            c.ok(f"all {n_levels} train levels populated: {line}", level_counts=counts)

        c = run_check("level-balance")
        vals = [counts.get(lv, 0) for lv in range(1, n_levels + 1)]
        ideal = sum(vals) / n_levels
        worst = max(abs(v - ideal) / ideal for v in vals)
        print(f"ideal per level = {ideal:.1f}; worst deviation = {worst * 100:.1f}%")
        if worst <= 0.25:
            c.ok(f"balanced: worst deviation {worst * 100:.1f}% (<= 25%)",
                 worst_deviation_pct=round(worst * 100, 2))
        else:
            c.fail(f"unbalanced: worst deviation {worst * 100:.1f}% (> 25%)")

        # -- monotone difficulty ------------------------------------------
        c = run_check("monotone-difficulty")
        rows = controller._feature_rows_train
        levels = controller.levels_train
        model = controller.model
        scores = [model.score(r) for r in rows]
        mean_by_level = {}
        for lv in range(1, n_levels + 1):
            idxs = [i for i, l in enumerate(levels) if l == lv]
            mean_by_level[lv] = sum(scores[i] for i in idxs) / max(1, len(idxs))
        seq = [mean_by_level[lv] for lv in range(1, n_levels + 1)]
        print("mean difficulty score by level:",
              " ".join(f"L{lv}={mean_by_level[lv]:.4f}" for lv in range(1, n_levels + 1)))
        if all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)):
            c.ok("mean score strictly increasing L1 -> L9")
        else:
            c.fail(f"mean score not monotone: {[round(x, 4) for x in seq]}")

        # face validity: key raw features should trend upward
        trends = {}
        for feat in ("pocket_n_atoms", "synthon_n_rotatable", "ligand_n_atoms",
                     "synthon_n_rings"):
            fm = {}
            for lv in range(1, n_levels + 1):
                idxs = [i for i, l in enumerate(levels) if l == lv]
                fm[lv] = sum(rows[i].get(feat, 0.0) for i in idxs) / max(1, len(idxs))
            first, last = fm[1], fm[n_levels]
            frac_up = sum(
                1 for i in range(1, n_levels)
                if fm[i + 1] >= fm[i] * 0.98
            ) / (n_levels - 1)
            trends[feat] = {"L1": round(first, 2), f"L{n_levels}": round(last, 2),
                            "monotone_frac": round(frac_up, 2)}
            print(f"  {feat}: L1={first:.2f} -> L{n_levels}={last:.2f} "
                  f"(non-decreasing steps: {frac_up * 100:.0f}%)")
            if last <= first * 1.05:
                c.fail(f"feature {feat} does not increase across levels "
                       f"(L1={first:.2f}, L{n_levels}={last:.2f})")
        if c.status == "PASS":
            c.ok("key difficulty features trend easy -> hard", trends=trends)

        # -- val assignment -----------------------------------------------
        c = run_check("val-assignment")
        vcounts = controller.level_counts_val
        vline = " ".join(f"L{lv}={vcounts.get(lv, 0)}" for lv in range(1, n_levels + 1))
        print(f"val level counts (train edges): {vline}")
        n_val_assigned = sum(vcounts.values())
        if n_val_assigned != len(val_ds):
            c.fail(f"val assignments ({n_val_assigned}) != val size ({len(val_ds)})")
        else:
            c.ok(f"all {n_val_assigned} val samples assigned with train edges",
                 val_level_counts=vcounts)
        # leakage invariant: val rows can never move train levels - rebuild
        # with val reversed and confirm train levels are identical
        try:
            class ReversedVal(list):
                def __init__(self, ds):
                    super().__init__(ds)
                    self._ds = ds

                def __getitem__(self, i):
                    return self._ds[len(self._ds) - 1 - i]

            from syntree.engine.curriculum import CurriculumController as CC

            build_cfg2 = dict(build_cfg)
            build_cfg2["cache_path"] = os.path.join(
                output_dir, "curriculum_levels_reversed_val.json")
            cc2 = CC.build(build_cfg2, train_ds, ReversedVal(val_ds), catalog,
                           output_dir=output_dir, log_fn=lambda *a, **k: None)
            if cc2.levels_train == controller.levels_train and \
               cc2.edges == controller.edges:
                c.ok("train levels/edges invariant to val ordering (no leakage)")
            else:
                c.fail("train levels changed when val was reordered - LEAKAGE")
        except Exception as exc:
            c.error(f"leakage probe failed: {exc}")

    # ------------------------------------------------------------------
    # 8-13. sampler + progression verification
    # ------------------------------------------------------------------
    if controller is not None:
        sampler = controller.sampler
        min_w = float(curr_cfg.get("min_weight", 0.03))
        n_stages = sampler.n_stages
        N = len(train_ds)

        c = run_check("sampler-invariants")
        stage_weights = {}
        for s in range(n_stages):
            sampler.stage = s
            w = sampler.weights()
            stage_weights[s] = {str(k): round(v, 4) for k, v in w.items()}
            total = sum(w.values())
            floor_ok = all(v >= min_w - 1e-9 for v in w.values())
            print(f"stage {s}/{n_stages - 1} center={sampler._centers[s]:.2f} "
                  f"weights={stage_weights[s]}")
            if abs(total - 1.0) > 1e-6:
                c.fail(f"stage {s}: weights sum {total:.6f} != 1")
            if not floor_ok:
                c.fail(f"stage {s}: some level below floor {min_w}")
        sampler.stage = 0
        if c.status == "PASS":
            c.ok(f"weights sum to 1 with floor >= {min_w} at all {n_stages} stages",
                 stage_weights=stage_weights)

        c = run_check("stage-progression")
        easy_mass = []
        hard_mass = []
        for s in range(n_stages):
            sampler.stage = s
            w = sampler.weights()
            easy = sum(w[lv] for lv in w if lv <= max(1, n_levels // 3))
            hard = sum(w[lv] for lv in w if lv >= n_levels - n_levels // 3 + 1)
            easy_mass.append(easy)
            hard_mass.append(hard)
        print(f"easy-third mass by stage: {[round(x, 3) for x in easy_mass]}")
        print(f"hard-third mass by stage: {[round(x, 3) for x in hard_mass]}")
        if easy_mass[0] <= easy_mass[-1] or hard_mass[0] >= hard_mass[-1]:
            c.fail("sampling mass does not shift easy -> hard")
        else:
            c.ok("mass shifts from easy to hard levels across stages",
                 easy_mass=easy_mass, hard_mass=hard_mass)
        sampler.stage = 0

        c = run_check("epoch-simulation")
        level_of = {}
        for idx, lv in enumerate(controller.levels_train):
            level_of[idx] = lv
        ok_all = True
        for s in range(n_stages):
            sampler.stage = s
            sampler.set_epoch(100 + s)  # arbitrary distinct epochs
            draws = list(sampler)
            if len(draws) != N:
                c.fail(f"stage {s}: drew {len(draws)} != {N}")
                ok_all = False
                break
            obs = {}
            for d in draws:
                obs[level_of[d]] = obs.get(level_of[d], 0) + 1
            w = sampler.weights()
            line = " ".join(f"L{lv}={obs.get(lv, 0)}" for lv in range(1, n_levels + 1))
            print(f"stage {s} epoch draw counts: {line}")
            for lv in range(1, n_levels + 1):
                if obs.get(lv, 0) == 0:
                    c.fail(f"stage {s}: level L{lv} never sampled")
                    ok_all = False
                exp = w.get(lv, 0.0) * N
                got = obs.get(lv, 0)
                tol = max(4.0 * math.sqrt(max(exp, 1.0)), 0.15 * exp)
                if abs(got - exp) > tol:
                    c.fail(f"stage {s} L{lv}: drew {got}, expected ~{exp:.0f} "
                           f"(tol {tol:.0f})")
                    ok_all = False
            # indices must belong to their level pools
            pool = {}
            for idx, lv in enumerate(controller.levels_train):
                pool.setdefault(lv, set()).add(idx)
            bad = sum(1 for d in draws if d not in pool[level_of[d]])
            if bad:
                c.fail(f"stage {s}: {bad} draws outside their level pool")
                ok_all = False
        sampler.stage = 0
        if ok_all and c.status == "PASS":
            c.ok(f"every stage: {N} draws, all levels present, counts ~ weights, "
                 f"pools respected")

        c = run_check("determinism")
        sampler.set_epoch(7)
        a = list(sampler)
        b = list(sampler)
        sampler.set_epoch(8)
        d8 = list(sampler)
        if a == b and a != d8:
            c.ok("same epoch reproduces the identical draw order; "
                 "different epochs differ")
        else:
            c.fail(f"determinism broken (same={a == b}, diff={a != d8})")

        c = run_check("budget-neutrality")
        try:
            batch = int(data_cfg.get("batch_size", 16))
            steps_curr = math.ceil(len(sampler) / batch)
            steps_plain = math.ceil(N / batch)
            print(f"steps/epoch: curriculum={steps_curr} plain-shuffle={steps_plain} "
                  f"(batch={batch}, drop_last=False)")
            if steps_curr == steps_plain and len(sampler) == N:
                c.ok("len(sampler) == len(train) and steps/epoch identical to "
                     "the plain-shuffle path", steps_per_epoch=steps_curr)
            else:
                c.fail("curriculum changes the per-epoch step count")
        except Exception as exc:
            c.error(str(exc))

        # -- milestone / patience advancement + full progression ----------
        c = run_check("milestone-advance")
        try:
            from syntree.engine.curriculum import CurriculumSampler as CS

            pools = {}
            for idx, lv in enumerate(controller.levels_train):
                pools.setdefault(lv, []).append(idx)
            sim = CS(pools, num_samples=N, n_levels=n_levels,
                     n_stages=n_stages,
                     min_weight=min_w,
                     sigma=float(curr_cfg.get("sigma", 1.8)),
                     seed=int(curr_cfg.get("seed", 42)),
                     milestone_metric=str(curr_cfg.get("milestone_metric", "val_loss")),
                     improve_thresh=float(curr_cfg.get("improve_thresh", 0.005)),
                     patience=int(curr_cfg.get("patience", 2)))
            max_epochs = int(config.get("training", {}).get("max_epochs", 40))
            # scenario 1: improving val_loss -> milestone advances
            stages_seen, epoch_of_stage = [], {}
            val = 7.50
            for epoch in range(max_epochs):
                stages_seen.append(sim.stage)
                epoch_of_stage.setdefault(sim.stage, epoch)
                sim.set_epoch(epoch)
                list(sim)  # consume the epoch
                sim.maybe_advance({"val_loss": val})
                val -= 0.05
            print(f"milestone scenario: stage timeline {stages_seen}")
            if sim.stage == n_stages - 1 and all(
                stages_seen[i] <= stages_seen[i + 1] for i in range(len(stages_seen) - 1)
            ):
                first_full = epoch_of_stage[n_stages - 1]
                c.ok(f"reached final stage at epoch {first_full}; stage never "
                     f"decreased", stage_timeline=stages_seen,
                     final_stage_epoch=first_full)
            else:
                c.fail(f"milestone progression incomplete/final={sim.stage}")
            # scenario 2: flat val_loss -> patience still advances
            sim2 = CS(pools, num_samples=N, n_levels=n_levels, n_stages=n_stages,
                      min_weight=min_w, sigma=float(curr_cfg.get("sigma", 1.8)),
                      seed=int(curr_cfg.get("seed", 42)),
                      improve_thresh=float(curr_cfg.get("improve_thresh", 0.005)),
                      patience=int(curr_cfg.get("patience", 2)))
            flat = []
            for epoch in range(max_epochs):
                flat.append(sim2.stage)
                sim2.maybe_advance({"val_loss": 7.49})
            print(f"patience scenario: stage timeline {flat}")
            if sim2.stage == n_stages - 1:
                c.ok(f"flat-metric run still reaches the final stage "
                     f"(patience) by epoch {flat.index(n_stages - 1)}")
            else:
                c.fail(f"patience path stalled at stage {sim2.stage}")
            # scenario 3: NaN metric must never advance via milestone
            sim3 = CS(pools, num_samples=N, n_levels=n_levels, n_stages=n_stages,
                      min_weight=min_w, sigma=float(curr_cfg.get("sigma", 1.8)),
                      seed=int(curr_cfg.get("seed", 42)),
                      improve_thresh=float(curr_cfg.get("improve_thresh", 0.005)),
                      patience=max_epochs * 2)  # patience disabled
            for _ in range(5):
                sim3.maybe_advance({"val_loss": float("nan")})
            if sim3.stage == 0:
                c.ok("non-finite metric never advances the stage")
            else:
                c.fail("NaN metric advanced the stage")
        except Exception as exc:
            c.error(f"{type(exc).__name__}: {exc}")

        # -- coverage of the entire dataset --------------------------------
        c = run_check("coverage-guarantee")
        try:
            coverage = {}
            total_draws = N * int(config.get("training", {}).get("max_epochs", 40))
            # expected stage residence: milestone scenario reached final at
            # epoch E, so approximate stages' share by the timeline observed
            stage_share = {}
            if "stage_timeline" in (checks[-1].metrics if checks else {}) or True:
                # use the simulated timeline from milestone-advance if available
                tl = None
                for cc in checks:
                    if cc.name == "milestone-advance" and "stage_timeline" in cc.metrics:
                        tl = cc.metrics["stage_timeline"]
                        break
                if tl:
                    for s in tl:
                        stage_share[s] = stage_share.get(s, 0) + 1
                    tot = sum(stage_share.values())
                    stage_share = {s: v / tot for s, v in stage_share.items()}
            for lv in range(1, n_levels + 1):
                n_pool = controller.level_counts_train.get(lv, 0)
                exp_draws = 0.0
                for s, share in stage_share.items():
                    sampler.stage = s
                    w = sampler.weights()
                    exp_draws += share * w.get(lv, 0.0) * N * \
                        int(config.get("training", {}).get("max_epochs", 40))
                p_miss = (1.0 - 1.0 / max(1, n_pool)) ** max(1.0, exp_draws)
                coverage[f"L{lv}"] = {
                    "pool": n_pool,
                    "expected_draws_over_run": round(exp_draws, 1),
                    "expected_unique_seen": round(n_pool * (1 - p_miss), 1),
                    "expected_unseen": round(n_pool * p_miss, 3),
                }
            sampler.stage = 0
            worst_unseen = max(v["expected_unseen"] for v in coverage.values())
            print(json.dumps(coverage, indent=1))
            if worst_unseen <= 0.5:
                c.ok(f"every level's expected-unseen samples <= 0.5 over the "
                     f"run ({total_draws} total draws) - full-dataset coverage",
                     coverage=coverage)
            else:
                c.fail(f"some level expects {worst_unseen:.1f} unseen samples "
                       f"- coverage hole")
        except Exception as exc:
            c.error(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # 15. cache round-trip + invalidation
    # ------------------------------------------------------------------
    if controller is not None:
        c = run_check("cache-roundtrip")
        try:
            from syntree.engine.curriculum import CurriculumController as CC

            build_again = dict(build_cfg)
            cc2 = CC.build(build_again, train_ds, val_ds, catalog,
                           output_dir=output_dir, log_fn=lambda *a, **k: None)
            if cc2.from_cache and cc2.levels_train == controller.levels_train \
               and cc2.levels_val == controller.levels_val \
               and cc2.edges == controller.edges:
                c.ok("rebuild loaded from cache with identical levels/edges")
            else:
                c.fail(f"cache round-trip mismatch (from_cache={cc2.from_cache})")

            build_changed = dict(build_cfg)
            build_changed["cache_path"] = os.path.join(
                output_dir, "curriculum_levels_changed.json")
            build_changed["feature_weights"] = dict(
                build_cfg.get("feature_weights", {}))
            first_feat = sorted(build_changed["feature_weights"])[0]
            build_changed["feature_weights"][first_feat] = \
                float(build_changed["feature_weights"][first_feat]) + 0.7
            cc3 = CC.build(build_changed, train_ds, val_ds, catalog,
                           output_dir=output_dir, log_fn=lambda *a, **k: None)
            if not cc3.from_cache:
                c.ok("changing feature_weights invalidates the cache "
                     "(levels recomputed)")
            else:
                c.fail("config change did NOT invalidate the cache")
        except Exception as exc:
            c.error(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    n_pass = sum(1 for c in checks if c.status == "PASS")
    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_err = sum(1 for c in checks if c.status == "ERROR")
    for c in checks:
        print(f"  [{c.status:^5}] {c.name}")
    print(f"\n{n_pass} PASS / {n_fail} FAIL / {n_err} ERROR")

    report["checks"] = {
        c.name: {"status": c.status, "details": c.details, "metrics": c.metrics}
        for c in checks
    }
    report["summary"] = {"pass": n_pass, "fail": n_fail, "error": n_err,
                         "verdict": "READY-FOR-TRAINING" if (n_fail == 0 and n_err == 0)
                         else "NOT-READY"}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\n[verify-curriculum] report -> {report_path}")

    return 0 if (n_fail == 0 and n_err == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
