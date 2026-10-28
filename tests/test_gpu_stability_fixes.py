"""Regression tests for the GPU-stability incident fixes (Colab T4 / Kaggle).

Covers the three acute production failures:

A. Stage-1 ``loss nan`` / ``val_loss 98,958,339``:
   * FP16-safe mask fills - the legacy ``-1e9`` overflows float16 to ``-inf``
     (binary16 max magnitude is 65,504); the new :data:`LOGIT_MASK_FILL`
     stays finite while still yielding exactly-zero softmax probability.
   * Target-unmasking invariant in ``ResilientTrainer._build_training_masks``:
     the ground-truth action is never masked, so cross-entropy can never see a
     ``+inf`` target logit (the source of NaN gradients and the ~1e8 val loss).
   * Finite-loss guard: a batch whose loss is inf/NaN is skipped, never
     backpropagated, and the epoch still reports finite metrics.

B. Stage-2 CUDA OOM in ``update_rollout``:
   * Every tensor stored inside a ``PPOTransition`` / ``RolloutTransition`` is
     ``.detach().cpu()``-ed by construction, so stored transitions never pin
     the forward autograd graph or VRAM between rollout steps.
   * ``configs/rl_colab_12h.json`` pins the T4-safe PPO geometry
     (minibatch_size <= 4, rollout_episodes <= 8) and main.py enables
     ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` before torch import.

C. Shape-adaptive checkpoint loading: restoring a 7-class checkpoint into a
   9-class head must copy overlapping rows, keep new rows at their fresh
   initialization, and never raise a size-mismatch RuntimeError.
"""

from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from syntree.core.masking import (
    LOGIT_MASK_FILL,
    LOGIT_MASK_LEGAL_ABOVE,
    has_any_legal,
)


# ======================================================================
# A1. FP16-safe mask constants
# ======================================================================
class TestFp16SafeMaskConstants:
    def test_fill_value_representable_in_fp16(self):
        """The new fill must survive the float16 cast; the legacy must not."""
        as_half = torch.tensor(LOGIT_MASK_FILL, dtype=torch.float16)
        assert torch.isfinite(as_half).item()
        # Documenting the original failure mode: 1e9 overflows binary16.
        legacy = torch.tensor(-1e9, dtype=torch.float16)
        assert not torch.isfinite(legacy).item()

    def test_masked_softmax_zero_probability_and_finite(self):
        """Masked entries get exactly-zero probability, everything finite."""
        torch.manual_seed(0)
        logits = torch.randn(4, 16).half()
        mask = torch.full((4, 16), LOGIT_MASK_FILL, dtype=torch.float16)
        mask[:, 3] = 0.0  # one legal entry per row
        probs = torch.softmax((logits + mask).float(), dim=-1)
        assert torch.isfinite(probs).all()
        assert (probs[:, 3] > 0.99).all()
        others = [i for i in range(16) if i != 3]
        assert (probs[:, others] < 1e-6).all()

    def test_ce_on_masked_then_unmasked_target_stays_finite(self):
        """The NaN mechanism end-to-end: masked target -> ~1e9 CE (the old
        failure); target-unmasking + fp16-safe fill -> bounded CE."""
        logits = torch.randn(2, 8)
        target = torch.tensor([5, 5])
        legacy_mask = torch.zeros(2, 8)
        legacy_mask[:, 5] = -1e9  # ground truth masked (the old bug)
        ce_masked = torch.nn.functional.cross_entropy(
            logits + legacy_mask, target
        )
        assert ce_masked.item() > 1e6

        fixed_mask = torch.full((2, 8), LOGIT_MASK_FILL)
        fixed_mask[:, 5] = 0.0  # target unmasked (the fix)
        ce_fixed = torch.nn.functional.cross_entropy(
            logits + fixed_mask, target
        )
        assert torch.isfinite(ce_fixed).item()
        assert ce_fixed.item() < 25.0

    def test_legality_threshold_separates_fill_from_legal(self):
        mask = torch.tensor([0.0, LOGIT_MASK_FILL])
        legal = mask > LOGIT_MASK_LEGAL_ABOVE
        assert legal.tolist() == [True, False]
        assert has_any_legal(mask)
        assert not has_any_legal(torch.full((3,), LOGIT_MASK_FILL))


# ======================================================================
# A2. Catalog / environment masks use the FP16-safe fill
# ======================================================================
class TestCatalogMaskFills:
    @pytest.fixture()
    def catalog(self, assets_dir):
        from syntree.chemistry.catalog import SynthonCatalog

        return SynthonCatalog(
            assets_dir["catalog_path"], embedding_dim=32,
            validate_handles=False, encoder="morgan2d",
        )

    def test_reaction_mask_fill_is_fp16_safe(self, catalog):
        # Suzuki with an amine core: only boronic-acid synthons are legal.
        mask = catalog.get_reaction_family_mask(
            "suzuki_coupling", core_handle="primary_secondary_amine"
        )
        assert (mask == LOGIT_MASK_FILL).any(), "some synthons must be masked"
        assert (mask == 0.0).any(), "boronic acids must stay legal"
        # The fill value itself must survive an fp16 cast (T4 autocast).
        assert torch.isfinite(mask.to(torch.float16)).all()

    def test_environment_empty_masks_use_fp16_safe_fill(
        self, catalog, tiny_config_with_assets, assets_dir
    ):
        from syntree.chemistry.conformer import ConformerEngine
        from syntree.chemistry.reactions import ReactionEngine
        from syntree.chemistry.validator import ChemicalValidator
        from syntree.engine.environment import MolecularAssemblyEnv

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        env = MolecularAssemblyEnv(
            rxn_engine=ReactionEngine(),
            conformer_engine=ConformerEngine(),
            catalog=catalog,
            validator=ChemicalValidator(),
            config=cfg,
            device=torch.device("cpu"),
        )
        pocket_path = os.path.join(
            assets_dir["crossdocked_dir"], "sample_pocket.pdb"
        )
        env.reset(pocket_path, seed_synthon_idx=0, max_steps=0)  # born terminal
        r_mask, s_masks, has_handle = env.legal_action_masks()
        assert has_handle is False
        assert torch.isfinite(r_mask.to(torch.float16)).all()
        assert torch.isfinite(s_masks.to(torch.float16)).all()
        assert not has_any_legal(r_mask)

        # The STOP mask on a below-cap seed uses the FP16-safe fill too.
        obs = env.observe()
        assert obs.stop_mask.item() == LOGIT_MASK_FILL


# ======================================================================
# A3. Trainer target-unmasking invariant
# ======================================================================
def _target_batch(family_idx, handle_idx, synthon_idx):
    return SimpleNamespace(
        target_reaction_family_idx=torch.tensor(list(family_idx)),
        target_core_handle_idx=torch.tensor(list(handle_idx)),
        target_synthon=torch.tensor(list(synthon_idx)),
    )


class TestTrainerTargetUnmasking:
    @pytest.fixture()
    def shell(self, assets_dir):
        from syntree.chemistry.catalog import SynthonCatalog
        from syntree.engine.trainer import ResilientTrainer

        return SimpleNamespace(
            device=torch.device("cpu"),
            catalog=SynthonCatalog(
                assets_dir["catalog_path"], embedding_dim=32,
                validate_handles=False, encoder="morgan2d",
            ),
            trainer_cls=ResilientTrainer,
        )

    def test_ground_truth_synthon_never_masked(self, shell):
        """Dataset annotation vs. catalog grammar disagreement: an amide
        family sample whose recorded target is a boronic acid (grammar would
        mask it). The invariant must unmask it while the rest stays masked."""
        from syntree.chemistry.reactions import HANDLE_NAMES, REACTION_FAMILY_NAMES

        amide = REACTION_FAMILY_NAMES.index("amide_coupling")
        amine = HANDLE_NAMES.index("primary_secondary_amine")
        boronic_row, azide_row = 10, 8  # conftest catalog rows

        batch = _target_batch([amide], [amine], [boronic_row])
        syn_mask, rxn_mask = shell.trainer_cls._build_training_masks(shell, batch)

        # Target unmasked (was LOGIT_MASK_FILL under pure grammar)...
        assert syn_mask[0, boronic_row].item() == 0.0
        # ...while grammar still constrains every non-target synthon.
        assert syn_mask[0, azide_row].item() == LOGIT_MASK_FILL
        assert rxn_mask[0, amide].item() == 0.0

    def test_ground_truth_family_never_masked(self, shell):
        """A recorded suzuki target on an amine-handle core is grammatically
        illegal; the invariant must still unmask the recorded family."""
        from syntree.chemistry.reactions import HANDLE_NAMES, REACTION_FAMILY_NAMES

        suzuki = REACTION_FAMILY_NAMES.index("suzuki_coupling")
        click = REACTION_FAMILY_NAMES.index("click_triazole")
        amine = HANDLE_NAMES.index("primary_secondary_amine")

        batch = _target_batch([suzuki], [amine], [10])
        syn_mask, rxn_mask = shell.trainer_cls._build_training_masks(shell, batch)

        assert rxn_mask[0, suzuki].item() == 0.0  # target family unmasked
        assert rxn_mask[0, click].item() == LOGIT_MASK_FILL  # others masked

    def test_out_of_range_targets_are_skipped_not_crashed(self, shell):
        """STOP-index / sentinel targets are bounds-checked, not indexed."""
        amide, amine = 0, 1
        batch = _target_batch([amide], [amine], [len(shell.catalog) + 7])
        syn_mask, rxn_mask = shell.trainer_cls._build_training_masks(shell, batch)
        assert torch.isfinite(syn_mask).all()
        assert torch.isfinite(rxn_mask).all()


# ======================================================================
# A4. Finite-loss guard (integration over the real training loop)
# ======================================================================
class _PoisonedPolicy:
    """Delegates everything to the wrapped policy but emits a +inf reaction
    logit for the first ``poison_calls`` forward calls - reproducing the
    exact class of batch that used to poison the whole run with NaN."""

    def __init__(self, inner: torch.nn.Module, poison_calls: int = 1):
        self.inner = inner
        self.poison_calls = int(poison_calls)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        self.calls += 1
        if self.calls <= self.poison_calls:
            out = dict(out)
            out["reaction_logits"] = out["reaction_logits"] + float("inf")
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def to(self, *args, **kwargs):
        """Keep the wrapper identity through ``model.to(device)`` (the
        trainer rebinds ``self.model = model.to(device)``)."""
        self.inner.to(*args, **kwargs)
        return self


class TestFiniteLossGuard:
    def test_poisoned_batch_is_skipped_and_run_stays_finite(
        self, tiny_config_with_assets, tmp_path
    ):
        from syntree.engine.trainer import ResilientTrainer
        from syntree.models.policy import SynTreePolicy

        cfg = json.loads(json.dumps(tiny_config_with_assets))
        cfg["data"]["synthetic_samples"] = 12
        cfg["training"]["max_epochs"] = 1

        torch.manual_seed(0)
        model = SynTreePolicy(cfg)
        poisoned = _PoisonedPolicy(model, poison_calls=1)
        trainer = ResilientTrainer(
            poisoned, cfg, torch.device("cpu"),
            auto_resume=False, output_dir=str(tmp_path),
        )
        summary = trainer.train()

        # The poisoned batch was skipped exactly once, never backpropagated.
        assert trainer.nan_skips == 1
        # The epoch still trained on the finite batches and reported them.
        assert summary["final_train_loss"] is not None
        assert math.isfinite(summary["final_train_loss"])
        assert summary["final_train_loss"] > 0.0

        history = json.load(open(os.path.join(str(tmp_path), "history.json")))
        entry = history[-1]
        assert entry["nonfinite_batches_skipped"] == 1
        for key in ("train_loss", "train_reaction_ce", "train_synthon_ce",
                    "train_torsion_nll"):
            assert math.isfinite(entry[key]), f"{key} must be finite"
        if "val_loss" in entry:
            assert math.isfinite(entry["val_loss"])
            assert entry["val_loss"] < 1e6, (
                "val_loss must never approach the 1e8 regime"
            )


# ======================================================================
# B1. Rollout transition tensor isolation (Stage-2 OOM fix)
# ======================================================================
class TestRolloutTransitionIsolation:
    def _grad_tensor(self):
        return (torch.randn(1, requires_grad=True) * 2.0).squeeze(0)

    def test_all_tensor_fields_detached_and_on_cpu(self):
        from syntree.engine.rl import RolloutTransition

        trans = RolloutTransition(
            state=None,
            reaction_mask=self._grad_tensor().unsqueeze(0).expand(1, 9),
            synthon_masks=self._grad_tensor().unsqueeze(0).expand(1, 9, 15),
            family_idx=0,
            action_idx=1,
            old_log_prob=self._grad_tensor(),
            old_value=self._grad_tensor(),
        )
        for field in ("reaction_mask", "synthon_masks", "old_log_prob", "old_value"):
            t = getattr(trans, field)
            assert t.grad_fn is None, f"{field} still pins an autograd graph"
            assert not t.requires_grad, f"{field} still requires grad"
            assert t.device.type == "cpu", f"{field} not isolated on CPU"

    def test_pyg_state_tensors_detached_and_on_cpu(self):
        from syntree.engine.rl import RolloutTransition

        state = Data(
            pocket_pos=torch.randn(5, 3),
            pocket_z=torch.randint(0, 10, (5,)),
            global_features=self._grad_tensor().unsqueeze(0).expand(1, 4),
        )
        trans = RolloutTransition(
            state=state,
            reaction_mask=torch.zeros(1, 9),
            synthon_masks=torch.zeros(1, 9, 15),
            family_idx=0,
            action_idx=0,
            old_log_prob=self._grad_tensor(),
            old_value=self._grad_tensor(),
        )
        stored = trans.state
        assert stored.pocket_pos.device.type == "cpu"
        assert stored.global_features.grad_fn is None
        assert not stored.global_features.requires_grad
        # Values survive isolation.
        assert torch.allclose(stored.pocket_pos, state.pocket_pos)

    def test_none_fields_pass_through(self):
        from syntree.engine.rl import RolloutTransition

        trans = RolloutTransition(
            state=None, reaction_mask=None, synthon_masks=None,
            family_idx=0, action_idx=0,
            old_log_prob=torch.tensor(-1.0), old_value=torch.tensor(0.5),
        )
        assert trans.state is None
        assert trans.reaction_mask is None
        assert trans.synthon_masks is None

    def test_values_preserved_after_isolation(self):
        from syntree.engine.rl import RolloutTransition

        lp = torch.tensor(-0.7, requires_grad=True)
        trans = RolloutTransition(
            state=None, reaction_mask=torch.zeros(1, 9),
            synthon_masks=torch.zeros(1, 9, 15),
            family_idx=2, action_idx=4,
            old_log_prob=lp, old_value=torch.tensor(1.25, requires_grad=True),
        )
        assert trans.old_log_prob.item() == pytest.approx(-0.7)
        assert trans.old_value.item() == pytest.approx(1.25)


# ======================================================================
# B2. T4-safe PPO geometry + allocator configuration
# ======================================================================
class TestRlOomConfiguration:
    def test_colab_rl_config_is_t4_safe(self, repo_root):
        cfg = json.load(open(os.path.join(repo_root, "configs", "rl_colab_12h.json")))
        rl = cfg["reinforcement_learning"]
        assert int(rl["minibatch_size"]) <= 4, (
            "PPO minibatch must stay <= 4 on 15 GB T4s "
            "(minibatch forward graphs are all alive until loss.backward)"
        )
        assert int(rl["rollout_episodes"]) <= 8

    def test_main_sets_expandable_segments_before_torch(self, repo_root):
        src = open(os.path.join(repo_root, "main.py")).read()
        assert "PYTORCH_CUDA_ALLOC_CONF" in src
        assert "expandable_segments:True" in src
        # Must be set BEFORE the torch import to reach the allocator init.
        assert src.index("PYTORCH_CUDA_ALLOC_CONF") < src.index("import torch")

    def test_notebook_sets_allocator_env_var(self, repo_root):
        nb = json.load(open(os.path.join(repo_root, "notebooks", "run_3d_syntree.ipynb")))
        joined = "".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        )
        assert "PYTORCH_CUDA_ALLOC_CONF" in joined
        # Codebase-first principle: the notebook must not patch sources inline.
        assert "open(hw_file)" not in joined
        assert "open(main_file)" not in joined


# ======================================================================
# C. Shape-adaptive checkpoint loading (7-class -> 9-class expansion)
# ======================================================================
class TestShapeAdaptiveCheckpoint:
    @pytest.fixture()
    def local_cfg(self, tiny_config, tmp_path):
        cfg = json.loads(json.dumps(tiny_config))
        cfg["huggingface"]["enabled"] = False
        return cfg

    def test_expand_head_7_to_9_classes(self, local_cfg, tmp_path):
        from syntree.utils.checkpoint import CheckpointManager

        ckpt_dir = str(tmp_path / "ck")
        mgr = CheckpointManager(local_cfg, ckpt_dir=ckpt_dir)

        torch.manual_seed(0)
        src = torch.nn.Linear(4, 7)  # 7-class head (legacy checkpoint)
        with torch.no_grad():
            src.weight.copy_(torch.randn(7, 4))
            src.bias.copy_(torch.randn(7))
        opt = torch.optim.Adam(src.parameters(), lr=1e-3)
        opt.step()  # populate optimizer state
        mgr.save_checkpoint(epoch=0, step=1, model=src, optimizer=opt,
                            metrics={"val_loss": 1.0})

        # Restore into an expanded 9-class head: must not raise.
        torch.manual_seed(1)
        dst = torch.nn.Linear(4, 9)
        dst_opt = torch.optim.Adam(dst.parameters(), lr=1e-3)
        start_epoch, step, best = mgr.restore_latest(dst, dst_opt)

        assert (start_epoch, step, best) == (1, 1, 1.0)
        assert dst.weight.shape == (9, 4)
        # Overlapping 7 rows are copied from the checkpoint...
        assert torch.allclose(dst.weight[:7], src.weight)
        assert torch.allclose(dst.bias[:7], src.bias)
        # ...and the 2 new rows keep their fresh (non-zero) initialization.
        new_rows = dst.weight[7:]
        assert torch.count_nonzero(new_rows) == new_rows.numel()
        assert not torch.allclose(new_rows, src.weight[:2])

    def test_shrink_head_9_to_7_classes(self, local_cfg, tmp_path):
        from syntree.utils.checkpoint import CheckpointManager

        ckpt_dir = str(tmp_path / "ck")
        mgr = CheckpointManager(local_cfg, ckpt_dir=ckpt_dir)

        torch.manual_seed(0)
        src = torch.nn.Linear(4, 9)
        with torch.no_grad():
            src.weight.copy_(torch.randn(9, 4))
            src.bias.copy_(torch.randn(9))
        mgr.save_checkpoint(epoch=0, step=1, model=src,
                            metrics={"val_loss": 1.0})

        dst = torch.nn.Linear(4, 7)
        start_epoch, _, _ = mgr.restore_latest(dst)
        assert start_epoch == 1
        assert torch.allclose(dst.weight, src.weight[:7])
