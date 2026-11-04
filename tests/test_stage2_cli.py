"""#31 FLOW-03 — the Stage-2 CLI + checkpoint state acceptance battery.

The issue's two acceptance criteria, in the V2-canonical CLI form
(``scripts/train_stage2.py`` — the clean-room replacement of the legacy
``main.py --mode rl``; the branching ``--algorithm`` flag is structurally
absent, D-V2-001, and Stage 2 *is* GFlowNet-TB per #45 §12.3):

1. *Initializing Stage 2 cleanly from Stage-1 checkpoints* — the CLI loads
   the #26 ``trainer_state.pt`` (SHA-256 sidecar verified) into the
   policy and records the exact initialization provenance; verified here
   against a REAL Stage-1 run on the amide fixture (a full in-process
   supervised fit, not a fabricated checkpoint).
2. *Resuming an interrupted RL run restores the exact ``log Z_theta``
   state* — ``--resume-auto`` resumes the Stage-2 checkpoint (bit-exact
   log-partition restore is pinned at the trainer level in
   ``test_gflownet_tb.py``); here the CLI wiring is verified end to end:
   updates continue, the checkpoint payload carries the neural partition
   function, and neither checkpoint source failing closed is ever
   silently tolerated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from syntree.data.supervised import (
    SupervisedTrajectoryDataset,
    supervised_collate,
)
from syntree.engine.trainer import ResilientTrainer, TrainerProfile
from syntree.models.policy import SynTreePolicy
from tests.test_supervised_dataset import (
    AMIDE_CATALOG,
    LIGAND_SMILES,
    _build_index,
    _context,
    _embed,
    _pocket,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_fixture(root: str) -> dict:
    """The Stage-1 CLI-smoke fixture: trajectories + processed + catalog H5."""
    from rdkit import Chem

    from syntree.chemistry.synthon_encoder import (
        D_EMBED,
        encode_synthon,
        write_catalog_h5,
    )
    from syntree.data.trajectories import (
        CatalogView,
        extract_trajectory,
        trajectory_to_dict,
    )

    dirs = {
        "trajectories": os.path.join(root, "trajectories"),
        "processed": os.path.join(root, "processed"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    records = {}
    for split in ("train", "val"):
        ligand = _embed(LIGAND_SMILES)
        record = extract_trajectory(
            ligand,
            complex_id="cx_amide",
            split=split,
            index=_build_index(AMIDE_CATALOG),
            catalog=CatalogView(
                block_ids=tuple(r[0] for r in AMIDE_CATALOG),
                smiles=tuple(r[1] for r in AMIDE_CATALOG),
            ),
        )
        records[split] = trajectory_to_dict(record)
    torch.save([records["train"]], os.path.join(dirs["trajectories"], "trajectories_train.pt"))
    torch.save([records["val"]], os.path.join(dirs["trajectories"], "trajectories_val.pt"))

    ligand = _embed(LIGAND_SMILES)
    with open(os.path.join(dirs["processed"], "cx_amide_ligand.sdf"), "w") as handle:
        handle.write(Chem.MolToMolBlock(ligand))
    pocket = _pocket(ligand)
    lines = []
    for i, (pos, z) in enumerate(zip(pocket.positions, pocket.atomic_numbers)):
        element = {6: "C", 7: "N", 8: "O", 16: "S"}[int(z)]
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA A{i % 9 + 1:4d}    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
            f"  1.00  0.00          {element:>2s}"
        )
    with open(os.path.join(dirs["processed"], "cx_amide_pocket.pdb"), "w") as handle:
        handle.write("\n".join(lines) + "\n")

    rows = [
        encode_synthon(block_id, smiles, stratum="fixture",
                       synthon_classes=syn, seed_classes=seed)
        for block_id, smiles, syn, seed in AMIDE_CATALOG
    ]
    x_raw = np.stack([r.x_raw for r in rows])
    mean = x_raw.mean(axis=0)
    std = np.maximum(x_raw.std(axis=0), 1e-12)
    rng = np.random.default_rng(2026)
    q, _ = np.linalg.qr(rng.standard_normal((x_raw.shape[1], D_EMBED)))
    catalog_path = os.path.join(root, "catalog_fixture.h5")
    write_catalog_h5(
        catalog_path, rows, mean=mean, std=std, q=q,
        source_sdf="fixture", source_sdf_sha256="0" * 64, registry_sha="0" * 64,
    )
    dirs["catalog"] = catalog_path
    return dirs


def _stage1_checkpoint(run_dir: str) -> str:
    """A REAL Stage-1 supervised checkpoint on the amide fixture."""
    from tests.test_trainer import _mini_catalog_tensors

    block_classes, seed_rows, available = _mini_catalog_tensors()
    context = _context("train")
    dataset = SupervisedTrajectoryDataset([context], split="train", augment=True, seed=2026)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=supervised_collate,
        generator=torch.Generator().manual_seed(2026),
    )
    policy = SynTreePolicy(catalog_size=3)
    trainer = ResilientTrainer(
        policy,
        TrainerProfile(
            micro_batch_size=4, gradient_accumulation_steps=1, max_epochs=1,
            log_interval_steps=1, checkpoint_interval_steps=10 ** 6,
        ),
        run_dir=run_dir,
        catalog_block_handle_classes=block_classes,
        catalog_seed_rows=seed_rows,
        available_block_classes=available,
    )
    trainer.fit(loader)
    return trainer.save_checkpoint()


def _stage2_config(path: str) -> str:
    config = {
        "profile_name": "gflownet-stage2-fixture",
        "learning_rate": 3e-4,
        "log_z_learning_rate": 1e-2,
        "weight_decay": 1e-5,
        "gradient_clip_norm": 1.0,
        "seed": 2026,
        "rollout_batch_size": 2,
        "max_blocks": 2,
        "max_pocket_atoms": 700,
        "updates_per_checkpoint": 1,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle)
    return path


def _run_stage2(fixture, config, run_dir, *, stage1=None, resume_auto=False,
                updates=2, expect_rc=0):
    command = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts", "train_stage2.py"),
        "--config", config,
        "--trajectories", fixture["trajectories"],
        "--processed", fixture["processed"],
        "--catalog", fixture["catalog"],
        "--run-dir", run_dir,
        "--updates", str(updates),
        "--rollout-batch", "2",
    ]
    if stage1 is not None:
        command += ["--stage1-run-dir", stage1]
    if resume_auto:
        command += ["--resume-auto"]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=900, cwd=REPO_ROOT
    )
    assert result.returncode == expect_rc, (
        f"rc={result.returncode}\nstdout tail: {result.stdout[-2000:]}\n"
        f"stderr tail: {result.stderr[-2000:]}"
    )
    return result


def _telemetry(run_dir: str):
    path = os.path.join(run_dir, "telemetry.jsonl")
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestStage2CLI:
    @pytest.fixture(scope="class")
    def fixture(self, tmp_path_factory):
        root = str(tmp_path_factory.mktemp("stage2_fixture"))
        fixture = _write_fixture(root)
        stage1_dir = os.path.join(root, "stage1_run")
        os.makedirs(stage1_dir, exist_ok=True)
        fixture["stage1_run"] = stage1_dir
        fixture["stage1_checkpoint"] = _stage1_checkpoint(stage1_dir)
        fixture["stage2_config"] = _stage2_config(os.path.join(root, "stage2.json"))
        return fixture

    def test_stage1_checkpoint_exists_with_sidecar(self, fixture):
        assert os.path.exists(fixture["stage1_checkpoint"])
        assert os.path.exists(fixture["stage1_checkpoint"] + ".sha256")

    def test_init_from_stage1_records_exact_provenance(self, fixture, tmp_path):
        """Acceptance 1: clean initialization from the Stage-1 checkpoint."""
        run_dir = str(tmp_path / "s2")
        result = _run_stage2(fixture, fixture["stage2_config"], run_dir,
                             stage1=fixture["stage1_run"], updates=2)
        assert "initialized from Stage-1 weights" in result.stdout
        init = [r for r in _telemetry(run_dir) if r.get("event") == "init"][0]
        assert init["resumed"] is False
        assert init["stage1_checkpoint"] == fixture["stage1_checkpoint"]
        # the recorded policy hash is EXACTLY the Stage-1 checkpoint's model
        stage1_payload = torch.load(
            fixture["stage1_checkpoint"], map_location="cpu", weights_only=False
        )
        policy = SynTreePolicy(catalog_size=3)
        policy.load_state_dict(stage1_payload["model"])
        import hashlib

        digest = hashlib.sha256()
        state = policy.state_dict()
        for key in sorted(state):
            digest.update(key.encode("utf-8"))
            digest.update(state[key].detach().cpu().numpy().tobytes())
        assert init["policy_param_sha256"] == digest.hexdigest()
        # the run wrote the Stage-2 checkpoint with the log Z theta state
        checkpoint = os.path.join(run_dir, "gflownet_state.pt")
        assert os.path.exists(checkpoint) and os.path.exists(checkpoint + ".sha256")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["format"] == "gflownet-stage2-v1"
        assert "log_partition" in payload
        assert payload["progress"]["updates"] == 2

    def test_resume_auto_restores_and_continues(self, fixture, tmp_path):
        """Acceptance 2: --resume-auto continues the interrupted run."""
        run_dir = str(tmp_path / "s2r")
        _run_stage2(fixture, fixture["stage2_config"], run_dir,
                    stage1=fixture["stage1_run"], updates=2)
        first = torch.load(
            os.path.join(run_dir, "gflownet_state.pt"),
            map_location="cpu", weights_only=False,
        )
        first_log_z = {
            k: v.clone() for k, v in first["log_partition"].items()
        }
        # the interrupted continuation: 2 more updates from the checkpoint
        _run_stage2(fixture, fixture["stage2_config"], run_dir,
                    resume_auto=True, updates=4)
        second = torch.load(
            os.path.join(run_dir, "gflownet_state.pt"),
            map_location="cpu", weights_only=False,
        )
        assert second["progress"]["updates"] == 4
        assert second["progress"]["global_step"] > first["progress"]["global_step"]
        # the log Z theta state was restored from the checkpoint (same
        # parameterization; the two further updates then moved it)
        assert set(second["log_partition"]) == set(first_log_z)
        # the history carries the full uninterrupted record
        assert [entry["update"] for entry in second["history"]] == [1, 2, 3, 4]
        # and the RNG snapshot progressed (no silent replay of pocket order)
        order_updates = [r for r in _telemetry(run_dir) if "update_record" in r]
        assert len(order_updates) >= 4

    def test_resume_auto_without_any_checkpoint_fails_closed(self, fixture, tmp_path):
        run_dir = str(tmp_path / "empty")
        os.makedirs(run_dir, exist_ok=True)
        result = _run_stage2(fixture, fixture["stage2_config"], run_dir,
                             resume_auto=True, updates=1, expect_rc=2)
        assert "FAIL-CLOSED" in result.stderr

    def test_resume_auto_with_only_stage1_initializes(self, fixture, tmp_path):
        run_dir = str(tmp_path / "s2only")
        _run_stage2(fixture, fixture["stage2_config"], run_dir,
                    resume_auto=True, stage1=fixture["stage1_run"], updates=1)
        init = [r for r in _telemetry(run_dir) if r.get("event") == "init"][0]
        assert init["resumed"] is False

    def test_corrupt_stage1_sidecar_fails_closed(self, fixture, tmp_path):
        run_dir = str(tmp_path / "s2corrupt")
        bad_dir = str(tmp_path / "stage1_bad")
        os.makedirs(bad_dir, exist_ok=True)
        import shutil

        shutil.copy(fixture["stage1_checkpoint"], os.path.join(bad_dir, "trainer_state.pt"))
        with open(os.path.join(bad_dir, "trainer_state.pt.sha256"), "w") as handle:
            handle.write("0" * 64 + "\n")
        result = _run_stage2(fixture, fixture["stage2_config"], run_dir,
                             stage1=bad_dir, updates=1, expect_rc=2)
        assert "integrity" in result.stderr
