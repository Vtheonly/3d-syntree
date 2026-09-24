"""Full-pipeline integration tests: builder -> shards -> HF streaming ->
trainer -> checkpoint/resume -> generation.

This module runs the production data path end-to-end against a fake Hugging
Face Hub (local directory + monkeypatched hf_hub_download), proving that:

* the dataset built by scripts/build_full_dataset.py streams losslessly
  through ShardedHuggingFaceDataset,
* ResilientTrainer accepts the real-shard dataset with the production
  fail-closed guard ENABLED (require_real_data=true, min_real_samples=50),
* checkpointing records contiguous epochs and auto-resume continues the run,
* the trained policy still drives the inference (generation) pipeline,
* identical seeds reproduce identical training losses on the HF backend.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import syntree.data.hf_loader as hf_loader  # noqa: E402
from scripts import build_full_dataset as bfd  # noqa: E402
from scripts.shard_and_upload import write_shards  # noqa: E402

POCKET_PDB = (
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
    "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
    "ATOM      3  C   GLY A   1       2.009   1.361   0.000  1.00 20.00           C\n"
    "ATOM      4  O   GLY A   1       1.272   2.348   0.000  1.00 20.00           O\n"
    "ATOM      5  N   GLY A   2       3.315   1.545   0.000  1.00 20.00           N\n"
    "ATOM      6  CA  GLY A   2       3.866   2.906   0.000  1.00 20.00           C\n"
    "ATOM      7  C   GLY A   2       4.417   4.267   0.000  1.00 20.00           C\n"
    "ATOM      8  O   GLY A   2       3.680   5.254   0.000  1.00 20.00           O\n"
    "END\n"
)

_CANDIDATE_PRODUCTS = [
    "O=C(NC1CCCCC1)C1CCCCC1",
    "CC(C)COC(=O)C1CCCCC1",
    "CC(C)(C)C(=O)NCC1CCCCC1",
    "CC(C)COC(=O)C(C)(C)C",
]


def _embedded(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=19) == 0
    return mol


def _write_sdf(mol: Chem.Mol, path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


@pytest.fixture(scope="module")
def real_shard_repo(tmp_path_factory, assets_dir):
    """Build a mini real dataset and serve it as a fake HF repo.

    40 pocket/ligand complexes are enough for the 80/10/10 split to place
    >= 50 training states and non-empty val/test splits, which lets the
    production guard (min_real_samples=50) run with its real threshold.
    """
    raw = tmp_path_factory.mktemp("pipe_raw") / "crossdocked"
    raw.mkdir(parents=True)
    n = 0
    for copy in range(10):
        for smi in _CANDIDATE_PRODUCTS:
            n += 1
            pdir = raw / f"protein_{n:03d}"
            pdir.mkdir()
            cid = f"1abc_{n}_1"
            (pdir / f"{cid}_pocket10.pdb").write_text(POCKET_PDB)
            _write_sdf(_embedded(smi), pdir / f"{cid}_ligand.sdf")
    (raw.parent / ".extracted_marker").touch()

    out_dir = tmp_path_factory.mktemp("pipe_shards")
    argv = [
        "build_full_dataset.py",
        "--raw-dir", str(raw.parent),
        "--catalog", assets_dir["catalog_path"],
        "--output-dir", str(out_dir),
        "--max-steps", "2",
        "--shard-mb", "1",
        "--max-target-pockets", "5",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = bfd.main()
    finally:
        sys.argv = old_argv
    assert rc == 0

    train_manifest = json.loads((out_dir / "train" / "manifest.json").read_text())
    assert train_manifest["total_samples"] >= 50, (
        "fixture too small to exercise the production guard at threshold 50: "
        f"{train_manifest['total_samples']}"
    )

    class FakeHub:
        """Serves builder output as an HF repo: repo paths `data/<split>/...`
        map to the local layout `<out>/<split>/...` (matching what
        upload_split places in the repository)."""

        def __init__(self, root: Path):
            self.root = root

        def download(self, **kwargs):
            filename = kwargs["filename"]
            local = self.root / filename
            if not local.exists() and filename.startswith("data/"):
                local = self.root / filename[len("data/"):]
            if not local.exists():
                raise FileNotFoundError(kwargs["filename"])
            return str(local)

    return {
        "hub": FakeHub(out_dir),
        "out_dir": out_dir,
        "catalog": out_dir / "enamine_3d_subset.parquet",
        "train_total": train_manifest["total_samples"],
        "targets": list((out_dir / "targets").glob("*.pdb")),
    }


def _patch_hub(monkeypatch, hub):
    monkeypatch.setattr(hf_loader, "hf_hub_download", hub.download)


def _hf_config(real_shard_repo, tmp_path, **data_overrides) -> dict:
    cfg = {
        "system": {
            "project_name": "3D-SynTree-Integration",
            "seed": 42,
            "device": "cpu",
            "mixed_precision": "fp32",
            "num_workers": 0,
        },
        "huggingface": {"enabled": False, "repo_id": "", "push_every_n_epochs": 2},
        "data": {
            "backend": "huggingface",
            "synthon_catalog_path": str(real_shard_repo["catalog"]),
            "huggingface": {
                "repo_id": "fake/repo",
                "revision": "main",
                "cache_dir": str(tmp_path / "hf_cache"),
                "max_cached_shards": 4,
            },
            "batch_size": 4,
            "accumulate_grad_batches": 1,
            "val_fraction": 0.1,
            "require_real_data": True,
            "min_real_samples": 50,
        },
        "model": {
            "hidden_dim": 32,
            "num_equivariant_layers": 2,
            "num_radial_basis": 12,
            "cutoff_radius": 5.0,
            "synthon_embedding_dim": 32,
            "num_attention_heads": 4,
            "max_atomic_number": 100,
            "dropout": 0.0,
        },
        "training": {
            "max_epochs": 1,
            "time_budget_hours": 0.25,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "lr_scheduler": "cosine_warmup",
            "warmup_epochs": 1,
            "grad_clip_norm": 1.0,
            "keep_last_n_checkpoints": 2,
            "eval_interval_epochs": 1,
            "auto_scale": {"enabled": False},
        },
    }
    cfg["data"].update(data_overrides)
    return cfg


def _make_trainer(cfg, tmp_path, auto_resume=False):
    from syntree.engine.trainer import ResilientTrainer
    from syntree.models.policy import SynTreePolicy

    torch.manual_seed(0)
    model = SynTreePolicy(cfg)
    return ResilientTrainer(
        model, cfg, torch.device("cpu"),
        auto_resume=auto_resume, output_dir=str(tmp_path),
    )


class TestBuilderToLoaderRoundTrip:
    def test_streamed_samples_match_manifest(self, monkeypatch, real_shard_repo, tmp_path):
        _patch_hub(monkeypatch, real_shard_repo["hub"])
        from syntree.data.hf_loader import ShardedHuggingFaceDataset

        ds = ShardedHuggingFaceDataset(
            "fake/repo", split="train",
            cache_dir=str(tmp_path / "cache"),
        )
        assert len(ds) == real_shard_repo["train_total"]
        # Visit every sample: every one must be a real, complete state.
        for i in range(len(ds)):
            sample = ds[i]
            assert bool(sample.is_real_sample.item()) is True
            assert sample.pocket_pos.size(0) > 0
            assert sample.ligand_pos.size(0) > 0
            assert torch.isfinite(sample.pocket_pos).all()
            assert int(sample.target_synthon.item()) >= 0

    def test_val_and_test_loadable(self, monkeypatch, real_shard_repo, tmp_path):
        _patch_hub(monkeypatch, real_shard_repo["hub"])
        from syntree.data.hf_loader import ShardedHuggingFaceDataset

        for split in ("val", "test"):
            ds = ShardedHuggingFaceDataset(
                "fake/repo", split=split,
                cache_dir=str(tmp_path / "cache"),
            )
            assert len(ds) > 0
            _ = ds[0]


class TestTrainerOnRealShards:
    @pytest.fixture()
    def trained(self, monkeypatch, real_shard_repo, tmp_path):
        _patch_hub(monkeypatch, real_shard_repo["hub"])
        cfg = _hf_config(real_shard_repo, tmp_path)
        trainer = _make_trainer(cfg, tmp_path)
        summary = trainer.train()
        return {
            "summary": summary,
            "output_dir": str(tmp_path),
            "cfg": cfg,
            "trainer": trainer,
        }

    def test_guard_passes_with_production_threshold(self, trained, real_shard_repo):
        """The exact production guard (min_real_samples=50) accepts a real
        dataset built by scripts/build_full_dataset.py."""
        assert len(trained["trainer"].dataset) >= 50
        assert trained["trainer"].data_backend == "huggingface"

    def test_training_completes_epoch(self, trained):
        summary = trained["summary"]
        assert summary["epochs_completed"] >= 1
        assert summary["final_train_loss"] is not None
        assert summary["final_train_loss"] > 0

    def test_losses_finite(self, trained):
        history = json.load(open(os.path.join(trained["output_dir"], "history.json")))
        for entry in history:
            for key in ("train_loss", "train_reaction_ce", "train_synthon_ce"):
                assert entry[key] == entry[key], f"{key} is NaN"

    def test_artifacts_written(self, trained):
        out = trained["output_dir"]
        for name in (
            "history.json", "latest_metrics.json", "metrics.jsonl",
            "checkpoints/manifest.json", "checkpoints/progress.json",
        ):
            assert os.path.exists(os.path.join(out, name)), f"missing {name}"

    def test_progress_records_contiguous_epochs(self, trained):
        ckpt_dir = os.path.join(trained["output_dir"], "checkpoints")
        progress = json.load(open(os.path.join(ckpt_dir, "progress.json")))
        completed = progress["completed_epochs"]
        assert completed == list(range(len(completed)))
        assert completed[0] == 0

    def test_checkpoint_payload_matches_manifest(self, trained):
        ckpt_dir = os.path.join(trained["output_dir"], "checkpoints")
        manifest = json.load(open(os.path.join(ckpt_dir, "manifest.json")))
        ckpt = torch.load(
            os.path.join(ckpt_dir, f"checkpoint_epoch_{manifest['latest_epoch']}.pt"),
            map_location="cpu", weights_only=False,
        )
        assert int(ckpt["epoch"]) == int(manifest["latest_epoch"])
        assert "model_state_dict" in ckpt

    def test_resume_continues_from_checkpoint(
        self, monkeypatch, real_shard_repo, tmp_path
    ):
        _patch_hub(monkeypatch, real_shard_repo["hub"])
        cfg = _hf_config(real_shard_repo, tmp_path)
        cfg["training"]["max_epochs"] = 1
        trainer = _make_trainer(cfg, tmp_path)
        trainer.train()

        # A second process (auto-resume) must start after the finished epoch.
        cfg2 = json.loads(json.dumps(cfg))
        cfg2["training"]["max_epochs"] = 1
        trainer2 = _make_trainer(cfg2, tmp_path, auto_resume=True)
        assert trainer2.start_epoch == 1
        summary = trainer2.train()
        assert summary["status"] == "already_completed"
        assert summary["epochs_completed"] == 1

    def test_resume_extends_training(
        self, monkeypatch, real_shard_repo, tmp_path
    ):
        _patch_hub(monkeypatch, real_shard_repo["hub"])
        cfg = _hf_config(real_shard_repo, tmp_path)
        cfg["training"]["max_epochs"] = 1
        trainer = _make_trainer(cfg, tmp_path)
        trainer.train()

        cfg2 = json.loads(json.dumps(cfg))
        cfg2["training"]["max_epochs"] = 2
        trainer2 = _make_trainer(cfg2, tmp_path, auto_resume=True)
        assert trainer2.start_epoch == 1
        summary = trainer2.train()
        assert summary["epochs_completed"] == 2
        ckpt_dir = os.path.join(str(tmp_path), "checkpoints")
        progress = json.load(open(os.path.join(ckpt_dir, "progress.json")))
        assert progress["completed_epochs"] == [0, 1]


class TestInferenceAfterTraining:
    def test_trained_policy_generates_from_staged_target(
        self, monkeypatch, real_shard_repo, tmp_path
    ):
        """Inference pipeline: after training on real shards, the policy
        drives SBDDGenerator on a pocket staged by the builder."""
        _patch_hub(monkeypatch, real_shard_repo["hub"])
        cfg = _hf_config(real_shard_repo, tmp_path)
        trainer = _make_trainer(cfg, tmp_path)
        trainer.train()

        from syntree.engine.generator import SBDDGenerator

        model = trainer.model
        model.eval()
        gen = SBDDGenerator(model, cfg, torch.device("cpu"), output_dir=str(tmp_path))
        assert real_shard_repo["targets"], "builder must stage RL target pockets"
        pocket = str(real_shard_repo["targets"][0])
        result = gen.generate_ligand(pocket, seed_synthon_idx=0, max_steps=1)
        assert result["recipe"], "generation must produce a recipe"
        assert result["recipe"][0]["action"] == "seed"
        assert result["contact_energy"] == result["contact_energy"]  # finite
        if result["rdkit_mol"] is not None:
            probe = Chem.Mol(result["rdkit_mol"])
            Chem.SanitizeMol(probe)  # must be chemically valid


class TestTrainingReproducibility:
    def test_identical_seeds_identical_losses(
        self, monkeypatch, real_shard_repo, tmp_path
    ):
        """Two independent HF-backend trainings with the same seed must
        produce identical loss histories (deterministic data order via
        ShardAwareShuffleSampler + seeded model init)."""
        losses = []
        for run in range(2):
            run_dir = tmp_path / f"run{run}"
            run_dir.mkdir()
            _patch_hub(monkeypatch, real_shard_repo["hub"])
            cfg = _hf_config(real_shard_repo, run_dir)
            torch.manual_seed(1234)
            from syntree.models.policy import SynTreePolicy
            from syntree.engine.trainer import ResilientTrainer

            model = SynTreePolicy(cfg)
            trainer = ResilientTrainer(
                model, cfg, torch.device("cpu"),
                auto_resume=False, output_dir=str(run_dir),
            )
            trainer.train()
            history = json.load(open(run_dir / "history.json"))
            losses.append([round(e["train_loss"], 6) for e in history])
        assert losses[0] == losses[1], (
            f"training not reproducible: {losses[0]} vs {losses[1]}"
        )


class TestNoDataLeak:
    def test_train_val_disjoint_by_trajectory(self, monkeypatch, real_shard_repo, tmp_path):
        """No trajectory may appear in more than one split (cluster-safe
        splits are approximated by whole-complex assignment in the builder)."""
        _patch_hub(monkeypatch, real_shard_repo["hub"])
        from syntree.data.hf_loader import ShardedHuggingFaceDataset

        seen = {}
        for split in ("train", "val", "test"):
            ds = ShardedHuggingFaceDataset(
                "fake/repo", split=split,
                cache_dir=str(tmp_path / "cache"),
            )
            for i in range(len(ds)):
                h = int(ds[i].trajectory_hash.item())
                seen.setdefault(h, set()).add(split)
        leaked = {h: s for h, s in seen.items() if len(s) > 1}
        assert not leaked, f"trajectories leaked across splits: {leaked}"

    def test_builder_summary_counts_match_shards(self, real_shard_repo):
        out = real_shard_repo["out_dir"]
        totals = {}
        for split in ("train", "val", "test"):
            manifest = json.loads((out / split / "manifest.json").read_text())
            declared = sum(s["sample_count"] for s in manifest["shards"])
            assert declared == manifest["total_samples"]
            totals[split] = manifest["total_samples"]
        assert sum(totals.values()) > 0
