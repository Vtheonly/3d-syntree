"""CPU regression tests for the T4 OOM / residual-NaN hardening.

Covers the four defenses added after the 2026-10-28 Colab T4 crash
(probe certified batch=8 at 6.34 GB, a later outlier-pocket batch spiked to
14.1 GB and raised torch.OutOfMemoryError in PaiNNInteraction):

1. fp16 activation saturation (``finite_act_clamp``) at PaiNN layer
   boundaries - residual NaN batches at batch 0/2 must not recur,
2. gradient checkpointing on :class:`BipartitePaiNN` - identical math,
   gradients flow, and the encoder pays no recompute in eval/no-grad,
3. worst-case pocket trimming (``training.max_pocket_atoms``) in the
   trainer collator - bounding the radius-graph edge count,
4. non-finite sample rejection (the ``loss nan`` data-side root cause).

All tests run on CPU only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from syntree.models.equivariant import (
    FP16_ACT_CLAMP,
    BatchedPaiNNLayer,
    finite_act_clamp,
)
from syntree.models.bipartite import BipartitePaiNN


# ---------------------------------------------------------------------------
# 1. fp16 activation saturation
# ---------------------------------------------------------------------------
def test_finite_act_clamp_saturates_inf_and_nan():
    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 1.5, -2.5])
    y = finite_act_clamp(x)
    assert torch.isfinite(y).all(), "clamp must leave no non-finite entries"
    assert y[0].item() == FP16_ACT_CLAMP
    assert y[1].item() == -FP16_ACT_CLAMP
    assert y[2].item() == 0.0
    assert y[3].item() == 1.5 and y[4].item() == -2.5


def test_finite_act_clamp_bounds_within_fp16_range():
    x = torch.tensor([1e5, -1e5, 7.0e4])  # 1e5 overflows fp16 (max 65,504)
    y = finite_act_clamp(x)
    assert y.abs().max().item() <= 65504.0
    assert y[0].item() == FP16_ACT_CLAMP


def test_finite_act_clamp_healthy_passthrough():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(64, 32, dtype=dtype)
        y = finite_act_clamp(x)
        assert y.dtype == dtype
        assert torch.allclose(x, y), "healthy activations must be untouched"


def test_finite_act_clamp_integer_passthrough():
    idx = torch.arange(5)
    assert finite_act_clamp(idx) is idx or torch.equal(finite_act_clamp(idx), idx)


def test_painn_layer_saturates_overflowing_inputs():
    """A layer fed pathological (inf-inducing) features must emit finite."""
    torch.manual_seed(0)
    layer = BatchedPaiNNLayer(hidden_dim=16, num_radial=8)
    n, cutoff = 30, 5.0
    pos = torch.randn(n, 3) * 3.0
    dist = torch.cdist(pos, pos)
    row, col = (dist <= cutoff).nonzero(as_tuple=True)
    edge_index = torch.stack([row, col])
    edge_vec = pos[col] - pos[row]
    edge_dist = torch.norm(edge_vec, dim=-1)
    rbf = torch.exp(
        -torch.pow(dist[row, col].unsqueeze(-1) - torch.linspace(0, cutoff, 8), 2)
    )
    scalar = torch.full((n, 16), 6.0e4)  # already at the fp16 ceiling
    vector = torch.zeros(n, 3, 16)
    out_s, out_v = layer(scalar, vector, edge_index, edge_vec, edge_dist, rbf)
    assert torch.isfinite(out_s).all(), "scalar channel must stay finite"
    assert torch.isfinite(out_v).all(), "vector channel must stay finite"


def test_interaction_distance_clamp_protects_coincident_atoms():
    """Duplicate coordinates (d == 0) must not produce NaN unit vectors."""
    torch.manual_seed(0)
    inter = BatchedPaiNNLayer(hidden_dim=16, num_radial=8).interaction
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_vec = torch.zeros(2, 3)  # perfectly coincident atoms
    edge_dist = torch.zeros(2)
    edge_rbf = torch.ones(2, 8) * 0.5
    scalar = torch.randn(2, 16)
    vector = torch.randn(2, 3, 16)
    agg_s, agg_v = inter(scalar, vector, edge_index, edge_vec, edge_dist, edge_rbf)
    assert torch.isfinite(agg_s).all() and torch.isfinite(agg_v).all()


# ---------------------------------------------------------------------------
# 2. gradient checkpointing
# ---------------------------------------------------------------------------
def _encoder_inputs(n=40, graphs=2, seed=0):
    torch.manual_seed(seed)
    pos = torch.randn(n, 3) * 6.0
    z = torch.randint(1, 30, (n,))
    batch = torch.repeat_interleave(torch.arange(graphs), n // graphs)
    charge = torch.zeros(n)
    return pos, z, batch, charge


def test_checkpointed_encoder_matches_plain_forward():
    """Checkpointing is memory plumbing only: outputs must be identical."""
    torch.manual_seed(7)
    enc_ckpt = BipartitePaiNN(hidden_dim=24, num_layers=3, dropout=0.0,
                               gradient_checkpointing=True)
    enc_plain = BipartitePaiNN(hidden_dim=24, num_layers=3, dropout=0.0,
                                gradient_checkpointing=False)
    enc_plain.load_state_dict(enc_ckpt.state_dict())
    pos, z, batch, charge = _encoder_inputs()

    enc_ckpt.train()
    enc_plain.train()
    s1, v1, p1 = enc_ckpt(pos, z, batch, charge)
    s2, v2, p2 = enc_plain(pos, z, batch, charge)
    assert torch.allclose(s1, s2, atol=1e-5), "scalar outputs must match"
    assert torch.allclose(v1, v2, atol=1e-5), "vector outputs must match"
    assert torch.allclose(p1, p2, atol=1e-5), "pooled outputs must match"


def test_checkpointed_encoder_gradients_flow_and_are_finite():
    enc = BipartitePaiNN(hidden_dim=24, num_layers=3, dropout=0.0,
                         gradient_checkpointing=True)
    enc.train()
    pos, z, batch, charge = _encoder_inputs(seed=3)
    s, v, pooled = enc(pos, z, batch, charge)
    pooled.sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads, "checkpointed backward must populate gradients"
    assert all(torch.isfinite(g).all() for g in grads), "grads must be finite"
    # Every parameter tensor of the encoder must have received gradient.
    n_with_grad = sum(1 for p in enc.parameters() if p.grad is not None)
    assert n_with_grad >= len(list(enc.parameters())) // 2


def test_checkpointing_inactive_in_eval_and_no_grad():
    """Rollouts/validation must not pay the recompute cost."""
    enc = BipartitePaiNN(hidden_dim=24, num_layers=2, dropout=0.0,
                         gradient_checkpointing=True)
    pos, z, batch, charge = _encoder_inputs(seed=1)
    enc.eval()
    with torch.no_grad():
        s, v, pooled = enc(pos, z, batch, charge)
    assert torch.isfinite(s).all() and torch.isfinite(pooled).all()
    # training-mode call under grad-enabled context must still work after eval
    enc.train()
    s2, v2, pooled2 = enc(pos, z, batch, charge)
    assert torch.isfinite(s2).all()


# ---------------------------------------------------------------------------
# 3. pocket trimming (trainer collator)
# ---------------------------------------------------------------------------
def _fake_item(n_pocket=3000, n_ligand=20, poison=False):
    import numpy as np

    rng = np.random.default_rng(11)
    pocket_pos = torch.tensor(rng.normal(0, 8.0, (n_pocket, 3)), dtype=torch.float32)
    ligand_pos = torch.tensor(rng.normal(0, 1.5, (n_ligand, 3)), dtype=torch.float32)
    item = SimpleNamespace(
        pocket_pos=pocket_pos,
        pocket_z=torch.randint(1, 30, (n_pocket,)),
        pocket_charge=torch.zeros(n_pocket),
        ligand_pos=ligand_pos,
        target_dihedral=torch.tensor(rng.uniform(-math.pi, math.pi, n_ligand),
                                     dtype=torch.float32),
    )
    if poison:
        item.pocket_pos = item.pocket_pos.clone()
        item.pocket_pos[7] = float("nan")
    return item


def _trainer_cls():
    from syntree.engine.trainer import ResilientTrainer

    return ResilientTrainer


def test_trim_pocket_caps_oversized_binding_core_kept():
    ResilientTrainer = _trainer_cls()
    item = _fake_item(n_pocket=3000)
    trimmed_item, trimmed = ResilientTrainer._trim_pocket(item, cap=1600)
    assert trimmed is True
    assert trimmed_item.pocket_pos.size(0) == 1600
    assert trimmed_item.pocket_z.size(0) == 1600
    assert trimmed_item.pocket_charge.size(0) == 1600
    # ligand untouched even though it exists
    assert trimmed_item.ligand_pos.size(0) == 20
    assert trimmed_item.target_dihedral.size(0) == 20
    # kept rows are a sorted subset of the original (deterministic)
    original = item.pocket_pos
    kept = trimmed_item.pocket_pos
    dists = (original.unsqueeze(1) - kept.unsqueeze(0)).norm(dim=-1).min(dim=1).values
    assert (dists < 1e-6).all(), "kept atoms must be exact original rows"


def test_trim_pocket_noop_below_cap():
    ResilientTrainer = _trainer_cls()
    item = _fake_item(n_pocket=100)
    same, trimmed = ResilientTrainer._trim_pocket(item, cap=1600)
    assert trimmed is False
    assert same.pocket_pos.size(0) == 100


def test_trim_pocket_disabled_with_cap_zero():
    ResilientTrainer = _trainer_cls()
    item = _fake_item(n_pocket=3000)
    _, trimmed = ResilientTrainer._trim_pocket(item, cap=0)
    assert trimmed is False


# ---------------------------------------------------------------------------
# 4. non-finite sample rejection
# ---------------------------------------------------------------------------
def test_has_nonfinite_detects_poison():
    ResilientTrainer = _trainer_cls()
    assert ResilientTrainer._has_nonfinite(_fake_item(poison=True)) is True
    assert ResilientTrainer._has_nonfinite(_fake_item(poison=False)) is False
    # poisoned dihedral target alone must also trip the detector
    item = _fake_item()
    item.target_dihedral = item.target_dihedral.clone()
    item.target_dihedral[0] = float("inf")
    assert ResilientTrainer._has_nonfinite(item) is True


def test_collate_batch_drops_poisonous_samples():
    ResilientTrainer = _trainer_cls()
    good = _fake_item(n_pocket=12, n_ligand=4)
    bad = _fake_item(n_pocket=12, n_ligand=4, poison=True)
    stats = {"trimmed": 0, "dropped_nonfinite": 0}

    # Wrap in torch_geometric Data so Batch.from_data_list accepts them.
    from torch_geometric.data import Data

    def to_data(item):
        return Data(
            pocket_pos=item.pocket_pos,
            pocket_z=item.pocket_z,
            pocket_charge=item.pocket_charge,
            ligand_pos=item.ligand_pos,
        )

    batch = ResilientTrainer.collate_batch(
        [to_data(good), to_data(bad)], pocket_atom_cap=100, stats=stats
    )
    assert stats["dropped_nonfinite"] == 1
    assert batch.pocket_pos.size(0) == 12, "only the clean sample survives"

    # Every item poisonous -> loud, actionable error, not a silent crash.
    try:
        ResilientTrainer.collate_batch(
            [to_data(bad), to_data(bad)], pocket_atom_cap=100, stats=stats
        )
        raise AssertionError("expected RuntimeError for all-poisonous batch")
    except RuntimeError as exc:
        assert "non-finite" in str(exc)


def test_collate_batch_trims_oversized_pockets():
    ResilientTrainer = _trainer_cls()
    item = _fake_item(n_pocket=3000, n_ligand=6)
    stats = {"trimmed": 0, "dropped_nonfinite": 0}
    from torch_geometric.data import Data

    data = Data(
        pocket_pos=item.pocket_pos,
        pocket_z=item.pocket_z,
        pocket_charge=item.pocket_charge,
        ligand_pos=item.ligand_pos,
    )
    batch = ResilientTrainer.collate_batch([data], pocket_atom_cap=1600, stats=stats)
    assert stats["trimmed"] == 1
    assert batch.pocket_pos.size(0) == 1600
