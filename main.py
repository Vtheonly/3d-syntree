#!/usr/bin/env python3
"""3D-SynTree unified CLI entrypoint.

Modes:
    train      – run the resilient, time-budgeted training loop.
    rl         – run Stage 2 chemistry-constrained PPO optimization.
    generate   – generate pocket-conditioned ligands + synthesis recipes.
    evaluate   – run the benchmark battery over generated molecules.
    info       – print runtime/hardware and configuration diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import torch


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (returns new dict)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str, runtime_config_path: Optional[str]) -> dict:
    with open(config_path) as f:
        config = json.load(f)
    if runtime_config_path:
        with open(runtime_config_path) as f:
            runtime = json.load(f)
        config = deep_update(config, runtime.get("config_override", {}))
    return config


def validate_config(config: dict) -> list:
    """Fail-closed structural validation of a merged config.

    Returns a list of human-readable error strings (empty = valid). This is
    deliberately lightweight - it catches the settings whose silent misuse
    would waste an entire offline RTX 6000 session (unknown data backend,
    unsupported mixed precision, impossible budgets, negative sizes)
    without duplicating every default.
    """
    from syntree.engine.trainer import SUPPORTED_DATA_BACKENDS

    errors = []

    system = config.get("system", {}) or {}
    mp = str(system.get("mixed_precision", "fp32")).lower()
    if mp not in ("fp32", "fp16", "bf16", "none", ""):
        errors.append(
            f"system.mixed_precision must be fp32 | fp16 | bf16, got '{mp}'"
        )

    data = config.get("data", {}) or {}
    backend = str(data.get("backend", "local")).lower()
    if backend not in SUPPORTED_DATA_BACKENDS:
        errors.append(
            f"data.backend must be one of {list(SUPPORTED_DATA_BACKENDS)}, "
            f"got '{backend}'"
        )
    if backend == "huggingface" and not str(
        (data.get("huggingface") or {}).get("repo_id", "")
    ).strip():
        errors.append(
            "data.huggingface.repo_id is required when data.backend='huggingface'"
        )
    if backend == "kaggle_offline" and not str(data.get("data_dir", "")).strip():
        errors.append(
            "data.data_dir is required when data.backend='kaggle_offline' "
            "(the mounted offline dataset directory)"
        )
    if backend == "trajectory_pt" and not data.get("trajectory_dataset_path"):
        errors.append(
            "data.trajectory_dataset_path is required when "
            "data.backend='trajectory_pt'"
        )
    for key in ("batch_size",):
        if key in data and int(data[key]) < 1:
            errors.append(f"data.{key} must be >= 1")
    # synthetic_samples == 0 is a valid production setting (no synthetic
    # fallback); only negatives are meaningless.
    if "synthetic_samples" in data and int(data["synthetic_samples"]) < 0:
        errors.append("data.synthetic_samples must be >= 0")
    if "accumulate_grad_batches" in data and int(data["accumulate_grad_batches"]) < 1:
        errors.append("data.accumulate_grad_batches must be >= 1")

    model = config.get("model", {}) or {}
    hidden = int(model.get("hidden_dim", 128))
    heads = int(model.get("num_attention_heads", 4))
    if hidden <= 0:
        errors.append("model.hidden_dim must be positive")
    if heads <= 0 or hidden % heads != 0:
        errors.append(
            f"model.hidden_dim ({hidden}) must be divisible by "
            f"model.num_attention_heads ({heads})"
        )
    if "synthon_embedding_dim" in model and int(model["synthon_embedding_dim"]) <= 0:
        errors.append("model.synthon_embedding_dim must be positive")

    training = config.get("training", {}) or {}
    budget = float(training.get("time_budget_hours", 11.5))
    if budget <= 0:
        errors.append("training.time_budget_hours must be positive")
    if int(training.get("max_epochs", 40)) < 1:
        errors.append("training.max_epochs must be >= 1")
    if float(training.get("learning_rate", 3e-4)) <= 0:
        errors.append("training.learning_rate must be positive")
    if "keep_last_n_checkpoints" in training and int(
        training["keep_last_n_checkpoints"]
    ) < 1:
        errors.append("training.keep_last_n_checkpoints must be >= 1")

    rl = config.get("reinforcement_learning", {}) or {}
    algorithm = str(rl.get("algorithm", "ppo")).strip().lower()
    if algorithm not in ("ppo", "gflownet", "dpo"):
        errors.append(
            f"reinforcement_learning.algorithm must be ppo | gflownet | dpo, "
            f"got '{algorithm}'"
        )
    if int(rl.get("episodes", 256)) < 1:
        errors.append("reinforcement_learning.episodes must be >= 1")

    catalog = config.get("catalog", {}) or {}
    encoder = str(catalog.get("encoder", "pharm3d")).strip().lower()
    if encoder not in ("pharm3d", "morgan2d"):
        errors.append(
            f"catalog.encoder must be pharm3d | morgan2d, got '{encoder}'"
        )

    return errors


def build_model(config: dict):
    from syntree.models.policy import SynTreePolicy

    return SynTreePolicy(config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="3D-SynTree SBDD Engine")
    parser.add_argument("--mode", choices=["train", "rl", "generate", "evaluate", "info"],
                        required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.json")
    parser.add_argument("--runtime-config", type=str, default=None,
                        help="JSON with a 'config_override' block (from notebooks)")
    parser.add_argument("--resume-auto", action="store_true",
                        help="auto-resume from the latest checkpoint (local or HF Hub)")
    parser.add_argument("--fresh", "--reset", action="store_true", dest="fresh",
                        help="start a completely fresh training run from epoch 0, ignoring and clearing any existing checkpoints")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training.max_epochs (e.g. to continue training past restored epochs)")
    parser.add_argument("--pocket", type=str, default=None,
                        help="pocket PDB for generation")
    parser.add_argument("--num-ligands", type=int, default=4)
    parser.add_argument("--sdf-dir", type=str, default="./outputs",
                        help="directory of generated SDFs for evaluation")
    parser.add_argument("--ckpt-dir", type=str, default="./experiments/checkpoints",
                        help="checkpoint directory for --resume-auto")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="override the output directory")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--pocket-dir", type=str, default=None,
        help="Directory of pocket PDBs for --mode rl (overrides the "
             "config and the data-dir scan).",
    )
    args = parser.parse_args(argv)

    from syntree.utils.logger import configure_logging
    configure_logging(level=os.environ.get("SYNTREE_LOG_LEVEL", "INFO"))

    # ---- configuration -------------------------------------------------
    if not os.path.exists(args.config):
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2
    config = load_config(args.config, args.runtime_config)
    if args.seed is not None:
        config.setdefault("system", {})["seed"] = int(args.seed)
    if args.output_dir is not None:
        config.setdefault("data", {})["output_dir"] = args.output_dir
    if args.epochs is not None:
        config.setdefault("training", {})["max_epochs"] = int(args.epochs)

    # Fresh mode overrides resume
    if args.fresh:
        args.resume_auto = False

    # Fail-closed config validation (before any hardware work or asset
    # staging: a session-wasting typo should die in milliseconds).
    validation_errors = validate_config(config)
    if validation_errors:
        for error in validation_errors:
            print(f"error: {error}", file=sys.stderr)
        return 2

    # ---- hardware ------------------------------------------------------
    from syntree.utils.hardware import configure_runtime_environment

    hw_info = configure_runtime_environment(config.get("system", {}))
    device = torch.device(hw_info["device"])
    print(f"[main] runtime profile: {json.dumps(hw_info, indent=2)}")

    # ---- dispatch -------------------------------------------------------
    if args.mode == "info":
        from syntree.chemistry.reactions import ReactionEngine, REACTION_TEMPLATES

        engine = ReactionEngine()
        print(f"[main] reactions registered: {len(engine.reactions)}")
        for name in sorted(REACTION_TEMPLATES):
            print(f"  - {name}")
        print(f"[main] device: {device} | precision: {hw_info['precision']}")
        return 0

    # Modes below need the data assets.
    data_dir = config["data"]["data_dir"]
    catalog_path = config["data"]["synthon_catalog_path"]
    if not os.path.exists(catalog_path):
        print(
            f"[main] catalog missing ({catalog_path}); "
            "run scripts/download_assets.py first.",
            file=sys.stderr,
        )
        return 2

    if args.mode == "train":
        from syntree.engine.trainer import ResilientTrainer
        from syntree.utils.checkpoint import CheckpointManager
        from syntree.utils.hardware import scale_model_config

        ckpt_dir = os.path.join(args.output_dir or "./experiments", "checkpoints")

        if args.fresh:
            print("[main] --fresh specified: resetting checkpoint state for a clean run from epoch 0")
            cleaner = CheckpointManager(config, ckpt_dir=ckpt_dir)
            cleaner.reset_all(purge_remote=bool(config.get("huggingface", {}).get("enabled", False)))

        if args.resume_auto and not args.fresh:
            reader = CheckpointManager(config, ckpt_dir=ckpt_dir)
            saved_model_cfg = reader.read_model_config()
            if saved_model_cfg:
                config["model"] = saved_model_cfg
                print("[main] restored model architecture from checkpoint")

        auto_cfg = config.get("training", {}).get("auto_scale", {})
        if (
            not args.resume_auto
            and auto_cfg.get("enabled", False)
            and auto_cfg.get("scale_model", True)
            and device.type == "cuda"
        ):
            scaled = scale_model_config(config.get("model", {}))
            if scaled != config.get("model", {}):
                print(
                    f"[main] auto-scale: model hidden "
                    f"{config['model'].get('hidden_dim', 128)} -> "
                    f"{scaled['hidden_dim']}, layers "
                    f"{config['model'].get('num_equivariant_layers', 4)} -> "
                    f"{scaled['num_equivariant_layers']}, heads "
                    f"{config['model'].get('num_attention_heads', 4)} -> "
                    f"{scaled['num_attention_heads']}"
                )
                config["model"] = scaled

        model = build_model(config).to(device)
        trainer = ResilientTrainer(
            model, config, device,
            auto_resume=args.resume_auto,
            output_dir=args.output_dir or "./experiments",
        )
        summary = trainer.train()
        print(f"[main] training summary: {json.dumps(summary, indent=2)}")
        return 0

    if args.mode == "rl":
        if not args.resume_auto:
            print("[main] --mode rl requires --resume-auto so Stage 2 starts from Stage 1 weights.", file=sys.stderr)
            return 2

        from syntree.engine.generator import SBDDGenerator
        from syntree.engine.rl import PPOFineTuner, ThreeDReward
        from syntree.utils.checkpoint import CheckpointManager

        rl_cfg = config.get("reinforcement_learning", {})
        if not bool(rl_cfg.get("enabled", False)):
            print("[main] reinforcement_learning.enabled is false; enable it for Stage 2.", file=sys.stderr)
            return 2

        stage1_manager = CheckpointManager(config, ckpt_dir=args.ckpt_dir)
        saved_model_cfg = stage1_manager.read_model_config()
        if saved_model_cfg:
            config["model"] = saved_model_cfg
            print("[main] restored model architecture from Stage 1 checkpoint")

        model = build_model(config).to(device)

        rl_output_dir = args.output_dir or "./rl_outputs"
        rl_ckpt_dir = os.path.join(rl_output_dir, "checkpoints")
        rl_checkpoint_config = {**config, "huggingface": {"enabled": False}}
        # Stage 2 checkpoints fire every `checkpoint_every` episodes, so
        # progress entries are monotonic but intentionally non-contiguous.
        rl_manager = CheckpointManager(
            rl_checkpoint_config, ckpt_dir=rl_ckpt_dir, progress_mode="episode"
        )
        rl_start_episode, _, _ = rl_manager.restore_latest(model)
        if rl_start_episode == 0:
            s1_epoch, s1_step, _ = stage1_manager.restore_latest(model)
            if s1_epoch > 0 or s1_step > 0:
                print(f"[main] Stage 2 initialized with Stage 1 checkpoint (epoch {s1_epoch - 1}, step {s1_step})")
            else:
                print("[main] Stage 2: No Stage 1 checkpoint found; starting with initialized weights.")
        print(f"[main] Stage 2 starting at episode {rl_start_episode}")

        generator = SBDDGenerator(
            model, config, device, output_dir=args.output_dir or "./rl_outputs"
        )
        reward_fn = ThreeDReward(rl_cfg.get("reward", {}))

        from syntree.engine.environment import MolecularAssemblyEnv
        from syntree.engine.rl import (
            RolloutBuffer,
            apply_terminal_rewards,
            collect_episode,
        )

        # Stage-2 algorithm selection (tasklist RL modernisation): PPO
        # (default, unchanged), Trajectory-Balance GFlowNet (samples all
        # reward modes, P(x) ~ R(x)), or pairwise DPO.
        algorithm = str(rl_cfg.get("algorithm", "ppo")).strip().lower()
        if algorithm not in ("ppo", "gflownet", "dpo"):
            print(
                f"[main] reinforcement_learning.algorithm must be one of "
                f"ppo | gflownet | dpo, got '{algorithm}'.",
                file=sys.stderr,
            )
            return 2

        if algorithm == "gflownet":
            from syntree.engine.gflownet import GFlowNetTrainer

            finetuner = GFlowNetTrainer(model, generator.catalog, device, rl_cfg)
            print(
                f"[main] Stage 2 algorithm: GFlowNet Trajectory Balance "
                f"(lr={finetuner.lr}, lr_z={finetuner.lr_z}, "
                f"reward_floor={finetuner.reward_floor})"
            )
        elif algorithm == "dpo":
            from syntree.engine.dpo import DPOTrainer, build_preference_pairs

            finetuner = DPOTrainer(model, generator.catalog, device, rl_cfg)
            print(
                f"[main] Stage 2 algorithm: Direct Preference Optimization "
                f"(beta={finetuner.beta})"
            )
        else:
            finetuner = PPOFineTuner(model, generator.catalog, device, rl_cfg)
            print("[main] Stage 2 algorithm: PPO")

        assembly_env = MolecularAssemblyEnv(
            rxn_engine=generator.rxn_engine,
            conformer_engine=generator.conformer_engine,
            catalog=generator.catalog,
            validator=generator.validator,
            config=config,
            device=device,
        )

        # Restore algorithm-specific state (GFlowNet log Z, DPO reference
        # policy) that lives outside the model parameters.
        def _algorithm_extra_state():
            if algorithm == "gflownet":
                return finetuner.state_dict()
            if algorithm == "dpo":
                return finetuner.state_dict()
            return None

        if getattr(rl_manager, "last_extra_state", None):
            try:
                if algorithm == "gflownet":
                    finetuner.load_state_dict(rl_manager.last_extra_state)
                    print(
                        f"[main] restored GFlowNet log_Z="
                        f"{float(finetuner.log_Z.item()):.4f} from checkpoint"
                    )
                elif algorithm == "dpo":
                    finetuner.load_state_dict(rl_manager.last_extra_state)
                    print("[main] restored DPO reference policy from checkpoint")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[main] could not restore algorithm state: {exc}")

        candidate_dirs = [
            args.pocket_dir,
            rl_cfg.get("pocket_dir"),
            "./data/test_pockets",
            os.path.join(os.path.dirname(data_dir.rstrip("/")), "test_pockets"),
            os.path.join(data_dir, "test_pockets"),
            data_dir,
        ]
        for manifest_candidate in ("./data/assets_manifest.json", os.path.join(os.path.dirname(data_dir.rstrip("/")), "assets_manifest.json")):
            if os.path.isfile(manifest_candidate):
                try:
                    with open(manifest_candidate) as f:
                        manifest_meta = json.load(f)
                    if manifest_meta.get("rl_pocket_dir"):
                        candidate_dirs.insert(0, manifest_meta["rl_pocket_dir"])
                except Exception:
                    pass

        pocket_paths = []
        pocket_source = None
        for candidate in candidate_dirs:
            if not candidate or not os.path.isdir(candidate):
                continue
            found = sorted(
                os.path.join(candidate, name)
                for name in os.listdir(candidate)
                if name.endswith("_pocket.pdb")
            )
            if found:
                pocket_paths = found
                pocket_source = candidate
                break

        if not pocket_paths:
            print(
                f"[main] no pocket PDB files found for --mode rl: tried {[c for c in candidate_dirs if c]}. "
                "Provide --pocket-dir or ensure test_pockets/ is staged.",
                file=sys.stderr,
            )
            return 2
        print(f"[main] RL using {len(pocket_paths)} pockets from {pocket_source}")

        episodes = int(rl_cfg.get("episodes", 256))
        temperature = float(rl_cfg.get("temperature", 1.0))
        checkpoint_every = int(rl_cfg.get("checkpoint_every", 16))
        rollout_episodes = max(1, int(rl_cfg.get("rollout_episodes", 32)))
        minibatch_size = max(1, int(rl_cfg.get("minibatch_size", 32)))
        history = []
        buffer = RolloutBuffer(
            gamma=float(rl_cfg.get("gamma", 0.99)),
            gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
        )
        last_stats: Dict[str, float] = {}
        # Deferred terminal rewards: base components are computed per episode
        # right after collection, and the batch Tanimoto diversity term is
        # applied when the rollout batch closes (tasklist Priority 6).
        pending: list = []

        for episode in range(rl_start_episode, episodes):
            pocket = pocket_paths[episode % len(pocket_paths)]
            model.eval()
            transitions, _ = collect_episode(
                assembly_env,
                model,
                generator.catalog,
                pocket_pdb_path=pocket,
                terminal_reward_fn=None,
                sample=True,
                temperature=temperature,
            )
            final_mol = assembly_env.current_mol
            base_details = reward_fn.compute(
                final_mol,
                pocket,
                clash_score=float(assembly_env.total_clash),
                contact_energy=float(assembly_env.contact_energy()),
                pocket_mol=assembly_env.pocket_mol,
            )
            buffer.add_episode(transitions)
            pending.append({
                "episode": episode,
                "pocket": os.path.basename(pocket),
                "transitions": transitions,
                "mol": final_mol,
                "base": base_details,
            })
            model.train()

            if (
                buffer.num_episodes >= rollout_episodes
                or episode == episodes - 1
            ):
                details_list = apply_terminal_rewards(
                    [(item["transitions"], item["mol"], item["base"])
                     for item in pending],
                    reward_fn,
                )
                if algorithm == "gflownet":
                    last_stats = finetuner.train_step(
                        [
                            (item["transitions"], details["reward"])
                            for item, details in zip(pending, details_list)
                        ]
                    )
                elif algorithm == "dpo":
                    pair_episodes = [
                        {"transitions": item["transitions"],
                         "reward": details["reward"]}
                        for item, details in zip(pending, details_list)
                    ]
                    last_stats = finetuner.train_step(
                        build_preference_pairs(pair_episodes)
                    )
                else:
                    last_stats = finetuner.update_rollout(
                        buffer, minibatch_size=minibatch_size
                    )
                buffer = RolloutBuffer(
                    gamma=float(rl_cfg.get("gamma", 0.99)),
                    gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
                )
                for item, details in zip(pending, details_list):
                    entry = {
                        "episode": item["episode"],
                        "pocket": item["pocket"],
                        **details,
                        "buffered_episodes": 0,
                        **{
                            k: v
                            for k, v in last_stats.items()
                            if isinstance(v, (int, float))
                        },
                    }
                    history.append(entry)
                    print(
                        f"[rl] episode {entry['episode']:04d} | "
                        f"reward={entry['reward']:.4f} | "
                        f"transitions={last_stats.get('transitions', 0)} | "
                        f"loss={entry.get('loss', 0.0):.4f} | "
                        f"clip={entry.get('clip_fraction', 0.0):.3f}"
                    )
                pending = []

            if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
                rl_manager.save_checkpoint(
                    epoch=episode,
                    step=episode + 1,
                    model=model,
                    optimizer=finetuner.optimizer,
                    metrics=history[-1] if history else {},
                    is_best=False,
                    model_config=config.get("model"),
                    catalog_signature=generator.catalog.embedding_signature,
                    extra_state=_algorithm_extra_state(),
                )

        rl_manager.save_checkpoint(
            epoch=max(0, episodes - 1),
            step=episodes,
            model=model,
            optimizer=finetuner.optimizer,
            metrics=history[-1] if history else {},
            is_best=True,
            final=True,
            model_config=config.get("model"),
            catalog_signature=generator.catalog.embedding_signature,
            extra_state=_algorithm_extra_state(),
        )
        rl_output = args.output_dir or "./rl_outputs"
        os.makedirs(rl_output, exist_ok=True)
        with open(os.path.join(rl_output, "rl_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[main] Stage 2 complete: {episodes} {algorithm.upper()} episodes")
        return 0

    if args.mode == "generate":
        from syntree.engine.generator import SBDDGenerator
        from syntree.utils.checkpoint import CheckpointManager

        manager = CheckpointManager(config, ckpt_dir=args.ckpt_dir)
        saved_model_cfg = manager.read_model_config()
        if saved_model_cfg:
            config["model"] = saved_model_cfg
            print("[main] restored model architecture from checkpoint")

        model = build_model(config).to(device)
        if args.resume_auto:
            start_epoch, _, _ = manager.restore_latest(model)
            if start_epoch > 0:
                print(f"[main] restored checkpoint weights (epoch {start_epoch - 1})")

        generator = SBDDGenerator(
            model, config, device,
            output_dir=args.output_dir or "./outputs",
        )
        pocket = args.pocket
        if not pocket or not os.path.exists(pocket):
            candidates = [
                "./data/test_pockets/1a9u_pocket.pdb",
                os.path.join(data_dir, "sample_pocket.pdb"),
            ]
            pocket = next((c for c in candidates if os.path.exists(c)), None)

        if not pocket:
            print(f"[main] no pocket provided and sample pocket missing.", file=sys.stderr)
            return 2

        generator.generate_batch(
            pocket_pdb_path=pocket, num_ligands=int(args.num_ligands)
        )
        return 0

    if args.mode == "evaluate":
        from syntree.engine.evaluator import EvaluationPipeline

        pipeline = EvaluationPipeline(
            config, output_dir=args.output_dir or "./experiments/eval"
        )
        mols, recipes = _load_generated(args.sdf_dir)
        if not mols:
            print(f"[main] no molecules found under {args.sdf_dir}", file=sys.stderr)
            return 2
        pocket = args.pocket or os.path.join(data_dir, "sample_pocket.pdb")
        report = pipeline.evaluate(
            mols, recipes,
            pocket_pdb_path=pocket if os.path.exists(pocket) else None,
        )
        scalars = {
            k: v for k, v in report.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        print(f"[main] evaluation summary: {json.dumps(scalars, indent=2)}")
        print(f"[main] full report: {report.get('report_path')}")
        return 0

    return 1


def _load_generated(sdf_dir: str):
    """Load generated molecules + recipes from an outputs directory."""
    from rdkit import Chem

    mols, recipes = [], []
    if not os.path.isdir(sdf_dir):
        return mols, recipes

    for name in sorted(os.listdir(sdf_dir)):
        if name.startswith("ligand_") and name.endswith(".sdf"):
            supplier = Chem.SDMolSupplier(os.path.join(sdf_dir, name), removeHs=False)
            mol = supplier[0] if len(supplier) else None
            mols.append(mol)
            recipe_path = os.path.join(
                sdf_dir, name.replace("ligand_", "recipe_").replace(".sdf", ".json")
            )
            if os.path.exists(recipe_path):
                with open(recipe_path) as f:
                    recipes.append(json.load(f))
            else:
                recipes.append(None)
    return mols, recipes


if __name__ == "__main__":
    raise SystemExit(main())