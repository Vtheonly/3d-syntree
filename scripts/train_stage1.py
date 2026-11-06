"""The canonical Stage-1 supervised training entry point (#26 TRAIN-01).

This is the CLI the #36 Colab SSH workflow and #43 session-safe resume
drive on the authorized Tesla T4 runtime (constitution #45 §28.3: "Do
not invent a training command. Use the canonical CLI and configuration
implemented in the repository and documented in the README")::

    python scripts/train_stage1.py \
        --config configs/train_colab_t4.json \
        --trajectories work/data/trajectories \
        --processed work/data/processed \
        --run-dir runs/stage1_t4 \
        [--time-budget-seconds 8400] [--resume] [--device cuda] \
        [--curriculum]

The command:

1. loads the frozen static profile (Profile A — strict FP32, B=12,
   accumulation 6, pocket cap 700) and enforces it through the #10
   hardware gate;
2. loads the #18 trajectory records (``trajectories_{split}.pt``) and
   the processed complex artifacts (crystal ligand SDF + pocket PDB per
   complex, per the manifest), rebuilding the supervised contexts with
   fail-closed record/artifact cross-checks;
3. builds the train loader with the #20 augmentation (train split only)
   and the pristine validation loader; with ``--curriculum`` the train
   loader is driven by the #27 Approach C quantile curriculum sampler
   (9 levels / 6 stages / 0.03 floor, milestone-driven advancement on
   the validation loss — the recommended production form, D-V2-039);
4. runs :class:`~syntree.engine.trainer.ResilientTrainer` — fail-closed
   on divergence, atomic checkpoints with SHA-256 sidecars, complete
   host-RAM/GPU-VRAM telemetry — honoring the time budget with a
   controlled graceful stop for #43 cross-session continuation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from syntree.data.supervised import (  # noqa: E402
    SupervisedTrajectoryDataset,
    build_complex_context,
)
from syntree.data.trajectories import CatalogView  # noqa: E402
from syntree.engine.curriculum import CurriculumSampler  # noqa: E402
from syntree.engine.environment import PocketDefinition  # noqa: E402
from syntree.engine.trainer import (  # noqa: E402
    ResilientTrainer,
    TrainerProfile,
    load_profile,
)
from syntree.models.policy import SynTreePolicy  # noqa: E402


def _load_records(trajectories_dir: str, split: str):
    path = os.path.join(trajectories_dir, f"trajectories_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"trajectory file not found: {path}")
    records = torch.load(path, weights_only=False)
    if not isinstance(records, list) or not records:
        raise ValueError(f"no trajectory records in {path}")
    return records


def _load_complex(processed_dir: str, record: dict):
    """Load one complex's crystal ligand + pocket from the processed artifacts.

    The ligand is parsed exactly the way the #18 extractor parses it
    (``removeHs=True`` — ``build_trajectories.py``'s SDMolSupplier
    convention): PDBbind artifacts carry explicit hydrogens while
    CrossDocked/MOAD ones do not, and the canonical decomposition (and
    therefore the record's seed graph, step count and every
    ``product_smiles``) is defined on the heavy-atom graph.  Loading with
    ``removeHs=False`` re-derives a *different* seed for H-laden
    artifacts and trips the fail-closed record/artifact cross-check in
    ``build_complex_context`` (first observed on the authorized T4,
    S-V2-17: ``pdbbind_1px4``).
    """
    from rdkit import Chem

    complex_id = str(record["complex_id"])
    ligand_path = os.path.join(processed_dir, f"{complex_id}_ligand.sdf")
    pocket_path = os.path.join(processed_dir, f"{complex_id}_pocket.pdb")
    if os.path.exists(ligand_path + ".gz"):
        import gzip

        with gzip.open(ligand_path + ".gz", "rt", encoding="utf-8") as handle:
            ligand_block = handle.read()
        ligand = Chem.MolFromMolBlock(ligand_block, removeHs=True, sanitize=True)
    else:
        ligand = Chem.MolFromMolFile(ligand_path, removeHs=True, sanitize=True)
    if ligand is None:
        raise ValueError(f"{complex_id}: cannot parse the crystal ligand SDF")
    if not ligand.GetNumConformers():
        raise ValueError(f"{complex_id}: crystal ligand has no conformer")
    from syntree.data.supervised import pocket_from_pdb_text

    if os.path.exists(pocket_path + ".gz"):
        import gzip

        with gzip.open(pocket_path + ".gz", "rt", encoding="utf-8") as handle:
            pocket_text = handle.read()
    else:
        with open(pocket_path, "r", encoding="utf-8") as handle:
            pocket_text = handle.read()
    pocket = pocket_from_pdb_text(pocket_text)
    return ligand, pocket


def _catalog_tensors(catalog_path: str):
    """The catalog mask surfaces for the trainer (block classes + seed rows).

    Reads the immutable #12 catalog HDF5: per-row block handle classes
    (the first certified synthon class; seed-only rows carry the
    out-of-vocab sentinel so they are never legal retrieval targets), the
    seed-eligibility mask (non-empty ``seed_classes``), the distinct
    available block classes and the catalog size (the policy's
    ``catalog_size`` — the tied table must match the artifact exactly).
    """
    from syntree.chemistry.synthon_encoder import read_catalog_h5
    from syntree.models.policy import BLOCK_HANDLE_CLASS_VOCAB

    catalog, _metadata = read_catalog_h5(catalog_path)
    block_ids = [b.decode() if isinstance(b, bytes) else str(b) for b in catalog["block_ids"]]
    synthon_classes = [
        (s.decode() if isinstance(s, bytes) else str(s))
        for s in catalog["synthon_classes"]
    ]
    seed_classes = [
        (s.decode() if isinstance(s, bytes) else str(s))
        for s in catalog["seed_classes"]
    ]
    index_of = {name: i for i, name in enumerate(BLOCK_HANDLE_CLASS_VOCAB)}
    sentinel = len(BLOCK_HANDLE_CLASS_VOCAB)
    classes: List[int] = []
    available: set = set()
    for row in synthon_classes:
        primary = row.split(",")[0].strip()
        if primary in index_of:
            classes.append(index_of[primary])
            available.add(primary)
        else:
            classes.append(sentinel)  # seed-only row: never a legal synthon
    seed_rows = [bool(row.strip()) for row in seed_classes]
    return (
        torch.tensor(classes, dtype=torch.int64),
        torch.tensor(seed_rows, dtype=torch.bool),
        sorted(available),
        len(block_ids),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train_colab_t4.json")
    parser.add_argument("--trajectories", required=True,
                        help="directory holding trajectories_{train,val,test}.pt")
    parser.add_argument("--processed", required=True,
                        help="directory holding the processed complex artifacts")
    parser.add_argument("--catalog", default=os.environ.get("SYNTREE_CATALOG_H5"),
                        help="the immutable #12 catalog HDF5 (or SYNTREE_CATALOG_H5)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time-budget-seconds", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="enable the #27 Approach C quantile curriculum (9 levels / 6 "
        "stages / permanent 0.03 level floor; milestone-driven stage "
        "advancement on the validation loss)",
    )
    args = parser.parse_args(argv)

    profile: TrainerProfile = load_profile(args.config)
    if args.time_budget_seconds is not None:
        profile = type(profile)(
            **{**profile.__dict__, "time_budget_seconds": int(args.time_budget_seconds)}
        )
    if args.max_epochs is not None:
        profile = type(profile)(
            **{**profile.__dict__, "max_epochs": int(args.max_epochs)}
        )
    if args.micro_batch_size is not None:
        profile = type(profile)(
            **{**profile.__dict__, "micro_batch_size": int(args.micro_batch_size)}
        )

    from syntree.data.supervised import supervised_collate

    contexts = {"train": [], "val": []}
    for split in ("train", "val"):
        for record in _load_records(args.trajectories, split):
            ligand, pocket = _load_complex(args.processed, record)
            contexts[split].append(
                build_complex_context(record, ligand, pocket, profile.max_pocket_atoms)
            )
    train_dataset = SupervisedTrajectoryDataset(
        contexts["train"], split="train", augment=True, seed=profile.seed,
        max_pocket_atoms=profile.max_pocket_atoms,
    )
    val_dataset = (
        SupervisedTrajectoryDataset(
            contexts["val"], split="val", augment=False, seed=profile.seed,
            max_pocket_atoms=profile.max_pocket_atoms,
        )
        if contexts["val"]
        else None
    )
    curriculum = (
        CurriculumSampler.from_dataset(train_dataset, seed=profile.seed)
        if args.curriculum
        else None
    )
    if curriculum is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=profile.micro_batch_size,
            sampler=curriculum,
            num_workers=profile.num_workers,
            collate_fn=supervised_collate,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=profile.micro_batch_size,
            shuffle=True,
            num_workers=profile.num_workers,
            collate_fn=supervised_collate,
            generator=torch.Generator().manual_seed(profile.seed),
        )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=profile.micro_batch_size,
            shuffle=False,
            num_workers=profile.num_workers,
            collate_fn=supervised_collate,
        )
        if val_dataset is not None
        else None
    )

    catalog_path = args.catalog or os.environ.get("SYNTREE_CATALOG_H5")
    if not catalog_path:
        print(
            "[train_stage1] FAIL-CLOSED: no catalog supplied (--catalog / "
            "SYNTREE_CATALOG_H5) — the synthon legality mask and the seed "
            "eligibility mask require the immutable #12 catalog artifact",
            file=sys.stderr,
        )
        return 2
    block_classes, seed_rows, available, catalog_size = _catalog_tensors(catalog_path)
    policy = SynTreePolicy(catalog_size=catalog_size)
    trainer = ResilientTrainer(
        policy,
        profile,
        run_dir=args.run_dir,
        device=torch.device(args.device) if args.device else None,
        catalog_block_handle_classes=block_classes,
        catalog_seed_rows=seed_rows,
        available_block_classes=available,
        curriculum=curriculum,
        resume=bool(args.resume),
    )
    print(f"[train_stage1] profile: {json.dumps(profile.__dict__, sort_keys=True)}")
    print(f"[train_stage1] runtime: {trainer.runtime.summary()}")
    print(f"[train_stage1] train samples: {len(train_dataset)}; "
          f"val samples: {len(val_dataset) if val_dataset else 0}")
    if curriculum is not None:
        print(
            f"[train_stage1] curriculum: {curriculum.num_levels} levels / "
            f"{curriculum.num_stages} stages / floor "
            f"{curriculum.floor} (milestone {curriculum.milestone}, "
            f"patience {curriculum.stage_patience})"
        )
    result = trainer.fit(train_loader, val_loader=val_loader)
    print(f"[train_stage1] done: {json.dumps(result, sort_keys=True, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
