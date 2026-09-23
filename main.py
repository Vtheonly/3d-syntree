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

        if args.resume_auto:
            # Keep the same Hub/local authority rules for architecture discovery
            # that the trainer will use for actual checkpoint restoration.
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
        rl_manager = CheckpointManager(rl_checkpoint_config, ckpt_dir=rl_ckpt_dir)
        rl_start_episode, _, _ = rl_manager.restore_latest(model)
        if rl_start_episode == 0:
            # First time in Stage 2: load Stage 1 trained weights into policy
            s1_epoch, s1_step, _ = stage1_manager.restore_latest(model)
            if s1_step > 0:
                print(f"[main] Stage 2 initialized with Stage 1 checkpoint (step {s1_step})")
        print(f"[main] Stage 2 PPO starting at episode {rl_start_episode}")

        generator = SBDDGenerator(
            model, config, device, output_dir=args.output_dir or "./rl_outputs"
        )
        reward_fn = ThreeDReward(rl_cfg.get("reward", {}))
        finetuner = PPOFineTuner(model, generator.catalog, device, rl_cfg)

        from syntree.engine.environment import MolecularAssemblyEnv
        from syntree.engine.rl import RolloutBuffer, collect_episode

        assembly_env = MolecularAssemblyEnv(
            rxn_engine=generator.rxn_engine,
            conformer_engine=generator.conformer_engine,
            catalog=generator.catalog,
            validator=generator.validator,
            config=config,
            device=device,
        )

        # Robust multi-path pocket discovery for RL mode
        candidate_dirs = [
            args.pocket_dir,
            rl_cfg.get("pocket_dir"),
            "./data/test_pockets",
            os.path.join(os.path.dirname(data_dir.rstrip("/")), "test_pockets"),
            os.path.join(data_dir, "test_pockets"),
            data_dir,
        ]
        # Inspect assets_manifest.json if present
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
            gamma=finetuner.gamma,
            gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
        )
        last_stats: Dict[str, float] = {}

        for episode in range(rl_start_episode, episodes):
            pocket = pocket_paths[episode % len(pocket_paths)]
            model.eval()
            transitions, reward_details = collect_episode(
                assembly_env,
                model,
                generator.catalog,
                pocket_pdb_path=pocket,
                terminal_reward_fn=lambda env, p=pocket: reward_fn.compute(
                    env.current_mol,
                    p,
                    clash_score=float(env.total_clash),
                    contact_energy=float(env.contact_energy()),
                ),
                sample=True,
                temperature=temperature,
            )
            buffer.add_episode(transitions)
            model.train()

            if (
                buffer.num_episodes >= rollout_episodes
                or episode == episodes - 1
            ):
                last_stats = finetuner.update_rollout(
                    buffer, minibatch_size=minibatch_size
                )
                buffer = RolloutBuffer(
                    gamma=finetuner.gamma,
                    gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
                )

            model.eval()
            entry = {
                "episode": episode,
                "pocket": os.path.basename(pocket),
                **reward_details,
                "buffered_episodes": buffer.num_episodes,
                **{
                    k: v
                    for k, v in last_stats.items()
                    if isinstance(v, (int, float))
                },
            }
            history.append(entry)
            print(
                f"[rl] episode {episode:04d} | reward={entry['reward']:.4f} | "
                f"transitions={last_stats.get('transitions', 0)} | "
                f"loss={entry.get('loss', 0.0):.4f} | "
                f"clip={entry.get('clip_fraction', 0.0):.3f}"
            )
            if checkpoint_every > 0 and (episode + 1) % checkpoint_every == 0:
                rl_manager.save_checkpoint(
                    epoch=episode,
                    step=episode + 1,
                    model=model,
                    optimizer=finetuner.optimizer,
                    metrics=entry,
                    is_best=False,
                    model_config=config.get("model"),
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
        )
        rl_output = args.output_dir or "./rl_outputs"
        os.makedirs(rl_output, exist_ok=True)
        with open(os.path.join(rl_output, "rl_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[main] Stage 2 complete: {episodes} PPO episodes")
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