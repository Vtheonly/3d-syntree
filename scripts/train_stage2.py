"""The canonical Stage-2 GFlowNet Trajectory Balance entry point (#31 FLOW-03).

This is the V2-clean-room form of the issue's ``main.py --mode rl``: the
legacy repository's branching CLI (``--algorithm ppo|dpo|gflownet``) is
structurally absent from V2 by decision D-V2-001, and the canonical
Stage-1 entry point established the ``scripts/train_stage*.py`` pattern
(#45 §28.3: "use the canonical CLI and configuration implemented in the
repository").  Stage 2 **is** GFlowNet-TB (#45 §12.3: "PPO/DPO are not
production alternatives in the specified V2 design") — there is no
algorithm flag to remove::

    python scripts/train_stage2.py \
        --config configs/stage2_colab_t4.json \
        --trajectories work/data/trajectories \
        --processed work/data/processed \
        --catalog work/data/catalog/catalog_v2.h5 \
        --stage1-run-dir runs/stage1_t4 \
        --run-dir runs/stage2_t4 \
        [--resume-auto] [--updates N] [--time-budget-seconds 8400] \
        [--device cuda]

The command:

1. loads the frozen Stage-2 profile (the two #29-pinned optimizer groups:
   policy AdamW 3e-4/1e-5 continuing Stage-1 Profile A, and the
   seed-conditioned ``log Z_theta`` head at ``lr_Z = 1e-2``);
2. loads the immutable #12 catalog HDF5 — the legality/eligibility mask
   surfaces (shared with Stage-1) **and** the environment's
   ``BuildingBlock`` pool (block-side handles derived from the certified
   reactant SMARTS; seed rows through the molecule-side growth-handle
   construction — ``syntree/engine/rollout.py``);
3. loads the pocket pool from the #18 trajectory records + processed
   artifacts (the same loaders as Stage-1), capped by the documented
   centroid-distance pruning;
4. initializes the policy from the **Stage-1 pretrained weights** (the
   #26 ``trainer_state.pt`` with its SHA-256 sidecar verified) unless
   ``--resume-auto`` finds an interrupted Stage-2 checkpoint to resume
   exactly (the #31 acceptance criteria; the ``log Z_theta`` state rides
   the Stage-2 checkpoint payload);
5. runs the on-policy TB loop: per update, a deterministic
   (seed, update)-derived pocket permutation selects the rollout batch,
   :class:`~syntree.engine.rollout.PolicyRolloutDriver` drives the
   environments, and :class:`~syntree.engine.gflownet.GFlowNetTrainer`
   applies the TB update — fail-closed divergence, atomic SHA-256
   checkpoints every ``updates_per_checkpoint`` updates and at the time
   budget (the #43 graceful-stop contract), separate host-RAM/GPU-VRAM
   telemetry.

``--resume-auto`` semantics (fail-closed): resume the Stage-2 checkpoint
if present; otherwise initialize from Stage-1; if neither exists, refuse
(Stage 2 never trains from random weights — the supervised behavioural
prior is the Stage-2 starting point by design).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from train_stage1 import _catalog_tensors, _load_complex, _load_records  # noqa: E402

from syntree.engine.environment import (  # noqa: E402
    AssemblyEnvironment,
    PocketDefinition,
)
from syntree.engine.gflownet import (  # noqa: E402
    FlowError,
    GFlowNetProfile,
    GFlowNetTrainer,
)
from syntree.engine.trainer import TrainingDivergenceError  # noqa: E402
from syntree.engine.rollout import (  # noqa: E402
    PolicyRolloutDriver,
    RolloutError,
    building_block_from_catalog_row,
    seed_block_from_catalog_row,
)
from syntree.models.policy import SynTreePolicy  # noqa: E402

DEFAULT_CONFIG = os.path.join("configs", "stage2_colab_t4.json")

#: The GFlowNetProfile field names the config may carry (the optimizer
#: semantics — the digest-gated frozen profile).
_PROFILE_FIELDS = {
    "profile_name",
    "learning_rate",
    "log_z_learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "seed",
}


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    unknown = set(config) - _PROFILE_FIELDS - {
        "rollout_batch_size",
        "max_blocks",
        "max_pocket_atoms",
        "updates_per_checkpoint",
    }
    if unknown:
        raise ValueError(f"unknown Stage-2 config keys: {sorted(unknown)}")
    return config


def _capped_pocket(pocket: PocketDefinition, cap: int) -> PocketDefinition:
    """The pocket capped by centroid-distance pruning (D-V2-041)."""
    import numpy as np

    if pocket.num_atoms <= cap:
        return pocket
    positions = np.asarray(pocket.positions, dtype=np.float64)
    centroid = positions.mean(axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    keep = np.argsort(distances, kind="stable")[:cap]
    return PocketDefinition(
        positions=positions[keep],
        atomic_numbers=np.asarray(pocket.atomic_numbers, dtype=np.int64)[keep],
        charges=np.asarray(pocket.charges, dtype=np.float64)[keep],
    )


def _verify_stage1_checkpoint(path: str) -> dict:
    """Load + integrity-verify a #26 Stage-1 checkpoint (weights only)."""
    sidecar = path + ".sha256"
    if not os.path.exists(path) or not os.path.exists(sidecar):
        raise FileNotFoundError(
            f"the Stage-1 checkpoint is incomplete: {path} (+ .sha256) — "
            "Stage 2 initializes from Stage-1 pretrained weights (no random "
            "initialization path exists by design)"
        )
    expected = open(sidecar, "r", encoding="utf-8").read().strip()
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"Stage-1 checkpoint integrity check failed: {path} (SHA-256 "
            "mismatch — refusing to initialize from a corrupt file)"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"not a Stage-1 trainer checkpoint: {path}")
    return payload


def _policy_param_sha(policy: SynTreePolicy) -> str:
    digest = hashlib.sha256()
    for key in sorted(policy.state_dict()):
        digest.update(key.encode("utf-8"))
        digest.update(policy.state_dict()[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trajectories", required=True,
                        help="directory holding trajectories_{train,val,test}.pt")
    parser.add_argument("--processed", required=True,
                        help="directory holding the processed complex artifacts")
    parser.add_argument("--catalog", default=os.environ.get("SYNTREE_CATALOG_H5"),
                        help="the immutable #12 catalog HDF5 (or SYNTREE_CATALOG_H5)")
    parser.add_argument("--stage1-run-dir", default=None,
                        help="the Stage-1 run directory (trainer_state.pt)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume-auto", action="store_true",
                        help="resume the Stage-2 checkpoint if present; else "
                             "initialize from Stage-1 (refuse if neither exists)")
    parser.add_argument("--updates", type=int, default=None,
                        help="stop after this many TB updates")
    parser.add_argument("--time-budget-seconds", type=int, default=None)
    parser.add_argument("--rollout-batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    profile = GFlowNetProfile(
        **{key: config[key] for key in _PROFILE_FIELDS if key in config}
    )
    rollout_batch = int(
        args.rollout_batch if args.rollout_batch is not None
        else config.get("rollout_batch_size", 12)
    )
    max_blocks = int(config.get("max_blocks", 4))
    max_pocket_atoms = int(config.get("max_pocket_atoms", 700))
    updates_per_checkpoint = int(config.get("updates_per_checkpoint", 50))
    if rollout_batch < 1 or max_blocks < 1 or max_pocket_atoms < 1:
        print("[train_stage2] FAIL-CLOSED: invalid batch/budget configuration",
              file=sys.stderr)
        return 2

    catalog_path = args.catalog or os.environ.get("SYNTREE_CATALOG_H5")
    if not catalog_path:
        print(
            "[train_stage2] FAIL-CLOSED: no catalog supplied (--catalog / "
            "SYNTREE_CATALOG_H5) — the legality masks and the environment "
            "block pool require the immutable #12 catalog artifact",
            file=sys.stderr,
        )
        return 2

    # -- the catalog surfaces + the environment block pool -------------------
    block_classes, seed_rows, available, catalog_size = _catalog_tensors(catalog_path)
    from syntree.chemistry.synthon_encoder import read_catalog_h5

    catalog, _metadata = read_catalog_h5(catalog_path)
    block_ids = [
        b.decode() if isinstance(b, bytes) else str(b) for b in catalog["block_ids"]
    ]
    smiles = [s.decode() if isinstance(s, bytes) else str(s) for s in catalog["smiles"]]
    synthon_classes = [
        s.decode() if isinstance(s, bytes) else str(s) for s in catalog["synthon_classes"]
    ]
    print(f"[train_stage2] building {catalog_size} environment blocks "
          f"({int(seed_rows.sum())} seed-eligible)...")
    try:
        seed_pool = {}
        growth_pool = {}
        for row, (block_id, smi, syn_class, is_seed) in enumerate(
            zip(block_ids, smiles, synthon_classes, seed_rows.tolist())
        ):
            block = (
                seed_block_from_catalog_row(block_id, smi)
                if is_seed
                else building_block_from_catalog_row(block_id, smi, syn_class)
            )
            if is_seed:
                seed_pool[block_id] = block
            else:
                growth_pool[block_id] = block
    except Exception as error:  # noqa: BLE001 - fail-closed with context
        print(f"[train_stage2] FAIL-CLOSED: catalog bridge: {error}", file=sys.stderr)
        return 2
    if not seed_pool:
        print("[train_stage2] FAIL-CLOSED: the catalog has no seed-eligible rows",
              file=sys.stderr)
        return 2
    full_pool = dict(growth_pool)
    full_pool.update(seed_pool)

    # -- the pocket pool (the #18 records + processed artifacts) --------------
    pockets = []
    for record in _load_records(args.trajectories, "train"):
        _ligand, pocket = _load_complex(args.processed, record)
        pockets.append(_capped_pocket(pocket, max_pocket_atoms))
    if not pockets:
        print("[train_stage2] FAIL-CLOSED: the train split has no trajectories "
              "(rollout pockets come from the #18 records)", file=sys.stderr)
        return 2
    print(f"[train_stage2] pocket pool: {len(pockets)}; rollout batch "
          f"{rollout_batch}; max_blocks {max_blocks}")

    # -- the policy + trainer --------------------------------------------------
    policy = SynTreePolicy(catalog_size=catalog_size)
    stage1_path = (
        os.path.join(args.stage1_run_dir, "trainer_state.pt")
        if args.stage1_run_dir
        else None
    )
    stage2_path = os.path.join(args.run_dir, "gflownet_state.pt")
    resumed = os.path.exists(stage2_path) and os.path.exists(stage2_path + ".sha256")
    try:
        if resumed:
            trainer = GFlowNetTrainer(
                policy, run_dir=args.run_dir, profile=profile,
                device=torch.device(args.device) if args.device else None,
                resume=True,
            )
            print(f"[train_stage2] resumed Stage-2 checkpoint: updates="
                  f"{trainer.progress.updates}")
        elif stage1_path is not None and os.path.exists(stage1_path):
            payload = _verify_stage1_checkpoint(stage1_path)
            policy.load_state_dict(payload["model"])
            init_sha = _policy_param_sha(policy)
            trainer = GFlowNetTrainer(
                policy, run_dir=args.run_dir, profile=profile,
                device=torch.device(args.device) if args.device else None,
            )
            print(f"[train_stage2] initialized from Stage-1 weights: {stage1_path}")
            trainer.log_telemetry({
                "event": "init",
                "stage1_checkpoint": stage1_path,
                "policy_param_sha256": init_sha,
                "resumed": False,
            })
        else:
            print(
                "[train_stage2] FAIL-CLOSED: no Stage-2 checkpoint to resume and no "
                "Stage-1 checkpoint to initialize from — Stage 2 starts from the "
                "supervised behavioural prior by design (pass --stage1-run-dir)",
                file=sys.stderr,
            )
            return 2
    except (FileNotFoundError, ValueError, FlowError) as error:
        print(f"[train_stage2] FAIL-CLOSED: {error}", file=sys.stderr)
        return 2

    driver = PolicyRolloutDriver(
        trainer.policy,
        catalog_block_ids=block_ids,
        catalog_block_handle_classes=block_classes,
        catalog_seed_rows=seed_rows,
        available_block_classes=available,
        device=trainer.device,
        max_pocket_atoms=max_pocket_atoms,
    )

    # -- the on-policy TB loop ---------------------------------------------------
    import numpy as np

    def pocket_batch(update: int):
        """A deterministic (seed, update)-derived pocket permutation slice."""
        rng = np.random.default_rng(
            (int(profile.seed) * 1_000_003 + int(update) * 7_919) % (2**63)
        )
        order = rng.permutation(len(pockets))
        return [pockets[int(i)] for i in order[:rollout_batch]]

    started = time.monotonic()
    target_updates = (
        args.updates if args.updates is not None else 10**9
    )
    stop_reason = "completed"
    try:
        while trainer.progress.updates < target_updates:
            if (
                args.time_budget_seconds is not None
                and time.monotonic() - started >= args.time_budget_seconds
            ):
                stop_reason = "time_budget"
                break
            envs = [
                AssemblyEnvironment(
                    seed_pool, full_pool, pocket=pocket, max_blocks=max_blocks
                )
                for pocket in pocket_batch(trainer.progress.updates)
            ]
            trajectories = driver.rollout(envs)
            trainer.snapshot_rng()
            record = trainer.train_batch(trajectories)
            trainer.log_telemetry({**trainer.telemetry(), "update_record": record})
            if trainer.progress.updates % updates_per_checkpoint == 0:
                trainer.save_checkpoint()
                print(f"[train_stage2] update {trainer.progress.updates}: "
                      f"loss={record['loss']:.6f} logZ={record['log_z_mean']:.4f} "
                      f"(checkpointed)")
    except (FlowError, TrainingDivergenceError, RolloutError) as error:
        # fail-closed: a divergence is a bug to fix, never a batch to skip
        print(f"[train_stage2] FAIL-CLOSED: {error}", file=sys.stderr)
        return 3
    trainer.save_checkpoint()
    print(f"[train_stage2] done ({stop_reason}): updates="
          f"{trainer.progress.updates}, final loss "
          f"{trainer.history[-1]['loss'] if trainer.history else float('nan'):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
