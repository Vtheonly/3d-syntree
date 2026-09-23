#!/usr/bin/env python3
"""3D-SynTree unified CLI entrypoint.

Modes:
    train      – run the resilient, time-budgeted training loop.
    generate   – generate pocket-conditioned ligands + synthesis recipes.
    evaluate   – run the benchmark battery over generated molecules.
    info       – print runtime/hardware and configuration diagnostics.

Examples:
    python main.py --mode train --config configs/default_config.json --resume-auto
    python main.py --mode generate --pocket data/crossdocked/sample_pocket.pdb --num-ligands 8
    python main.py --mode evaluate --sdf-dir outputs/
    python main.py --mode info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

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

        # A resumed run must keep the architecture it was trained with
        # (which may have been GPU-auto-scaled on a different card).
        if args.resume_auto:
            reader = CheckpointManager(
                {**config, "huggingface": {"enabled": False}}, ckpt_dir=ckpt_dir
            )
            saved_model_cfg = reader.read_model_config()
            if saved_model_cfg:
                config["model"] = saved_model_cfg
                print("[main] restored model architecture from checkpoint")

        # GPU auto-scaling: bigger card -> bigger PaiNN (only for fresh runs).
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

        # Keep PPO checkpoints separate from Stage 1 epoch checkpoints so the
        # two training stages never overwrite one another.
        rl_output_dir = args.output_dir or "./rl_outputs"
        rl_ckpt_dir = os.path.join(rl_output_dir, "checkpoints")
        rl_checkpoint_config = {**config, "huggingface": {"enabled": False}}
        rl_manager = CheckpointManager(rl_checkpoint_config, ckpt_dir=rl_ckpt_dir)
        rl_start_episode, _, _ = rl_manager.restore_latest(model)
        print(
            f"[main] Stage 2 PPO starting at episode {rl_start_episode}"
        )

        generator = SBDDGenerator(
            model, config, device, output_dir=args.output_dir or "./rl_outputs"
        )
        reward_fn = ThreeDReward(rl_cfg.get("reward", {}))
        finetuner = PPOFineTuner(model, generator.catalog, device, rl_cfg)

        pocket_paths = sorted(
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if name.endswith("_pocket.pdb")
        )
        if not pocket_paths:
            print(f"[main] no pocket PDB files found under {data_dir}", file=sys.stderr)
            return 2

        episodes = int(rl_cfg.get("episodes", 256))
        temperature = float(rl_cfg.get("temperature", 1.0))
        checkpoint_every = int(rl_cfg.get("checkpoint_every", 16))
        history = []
        for episode in range(rl_start_episode, episodes):
            pocket = pocket_paths[episode % len(pocket_paths)]
            model.eval()
            result = generator.generate_ligand(
                pocket_pdb_path=pocket,
                sample=True,
                temperature=temperature,
                return_trace=True,
            )
            reward_details = reward_fn.compute(
                result["rdkit_mol"],
                pocket,
                clash_score=float(result.get("clash_score", 0.0)),
            )
            model.train()
            stats = finetuner.update_episode(
                result.get("policy_trace") or [],
                reward_details["reward"],
            )
            model.eval()
            entry = {
                "episode": episode,
                "pocket": os.path.basename(pocket),
                **reward_details,
                **stats,
            }
            history.append(entry)
            print(
                f"[rl] episode {episode:04d} | reward={entry['reward']:.4f} | "
                f"loss={entry['loss']:.4f} | steps={entry['steps']:.0f}"
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

        manager.save_checkpoint(
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
        # Rebuild the exact trained architecture (auto-scaled runs store it).
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
        pocket = args.pocket or os.path.join(data_dir, "sample_pocket.pdb")
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
