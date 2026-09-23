"""Unit tests for the end-to-end policy network."""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Batch

from syntree.chemistry.catalog import SynthonCatalog
from syntree.data.crossdocked import CrossDockedDataset
from syntree.models.policy import SynTreePolicy


@pytest.fixture(scope="module")
def catalog(assets_dir):
    return SynthonCatalog(assets_dir["catalog_path"], embedding_dim=32)


@pytest.fixture(scope="module")
def dataset(assets_dir, catalog):
    return CrossDockedDataset(
        assets_dir["crossdocked_dir"], split="train", catalog=catalog,
        num_synthetic=10, seed=5,
    )


@pytest.fixture()
def model(tiny_config):
    torch.manual_seed(0)
    return SynTreePolicy(tiny_config)


@pytest.fixture()
def batch(dataset):
    return Batch.from_data_list([dataset[i] for i in range(4)],
                                follow_batch=["pocket_pos"])


class TestConstruction:
    def test_parameter_count(self, model):
        assert model.count_parameters() > 10_000

    def test_invalid_head_division_raises(self, tiny_config):
        cfg = dict(tiny_config)
        cfg["model"] = {**tiny_config["model"], "hidden_dim": 30,
                        "synthon_embedding_dim": 30}
        with pytest.raises(ValueError, match="divisible"):
            SynTreePolicy(cfg)

    def test_mismatched_synthon_dim_raises(self, tiny_config):
        cfg = dict(tiny_config)
        cfg["model"] = {**tiny_config["model"], "synthon_embedding_dim": 64}
        with pytest.raises(ValueError, match="synthon_embedding_dim"):
            SynTreePolicy(cfg)


class TestForward:
    def test_output_shapes(self, model, batch, catalog):
        out = model(batch, catalog.embeddings)
        assert out["synthon_logits"].shape == (4, len(catalog))
        assert out["synthon_log_probs"].shape == (4, len(catalog))
        assert out["torsion_mu"].shape == (4,)
        assert out["torsion_kappa"].shape == (4,)
        assert out["pocket_context"].shape == (4, 32)

    def test_single_graph_forward(self, model, dataset, catalog):
        data = dataset[0]
        out = model(data, catalog.embeddings)
        assert out["synthon_logits"].shape == (1, len(catalog))

    def test_mask_applied(self, model, batch, catalog):
        mask = torch.zeros(4, len(catalog))
        mask[:, 5:] = -1e9
        out = model(batch, catalog.embeddings, mask)
        assert (out["synthon_log_probs"][:, 5:] < -1e6).all()

    def test_backward_gradients(self, model, batch, catalog):
        out = model(batch, catalog.embeddings)
        loss = out["synthon_logits"].sum() + out["torsion_mu"].sum() \
            + out["torsion_kappa"].sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_bad_handle_features_raise(self, model, dataset, catalog):
        data = dataset[0]
        data.handle_features = torch.zeros(10)
        with pytest.raises(ValueError):
            model(data, catalog.embeddings)

    def test_batch_graph_mismatch_raises(self, model, dataset, catalog):
        data = dataset[0]
        data.handle_features = torch.randn(3, 64)  # 3 graphs, 1 pocket
        with pytest.raises(ValueError):
            model(data, catalog.embeddings)


class TestSE3Properties:
    """Scalar outputs must be invariant to pocket rotations/translations."""

    @staticmethod
    def _rotation(theta, axis=2):
        c, s = math.cos(theta), math.sin(theta)
        R = torch.eye(3)
        i, j = (axis + 1) % 3, (axis + 2) % 3
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        return R

    def test_rotation_invariance(self, model, dataset, catalog):
        model.eval()
        data = dataset[1]
        d1 = Batch.from_data_list([data], follow_batch=["pocket_pos"])
        d2 = Batch.from_data_list([data], follow_batch=["pocket_pos"])
        d2.pocket_pos = d2.pocket_pos @ self._rotation(1.1).T
        with torch.no_grad():
            o1 = model(d1, catalog.embeddings)
            o2 = model(d2, catalog.embeddings)
        assert torch.allclose(o1["synthon_logits"], o2["synthon_logits"],
                              atol=1e-4)

    def test_translation_invariance(self, model, dataset, catalog):
        model.eval()
        data = dataset[1]
        d1 = Batch.from_data_list([data], follow_batch=["pocket_pos"])
        d2 = Batch.from_data_list([data], follow_batch=["pocket_pos"])
        d2.pocket_pos = d2.pocket_pos + torch.tensor([7.0, -4.0, 2.0])
        with torch.no_grad():
            o1 = model(d1, catalog.embeddings)
            o2 = model(d2, catalog.embeddings)
        assert torch.allclose(o1["pocket_context"], o2["pocket_context"],
                              atol=1e-4)


class TestAct:
    def test_act_argmax(self, model, dataset, catalog):
        model.eval()
        data = dataset[0]
        decision = model.act(data, catalog.embeddings)
        assert decision["synthon_idx"].shape == (1,)
        assert 0 <= decision["synthon_idx"][0].item() < len(catalog)
        assert -math.pi <= decision["dihedral"][0].item() < math.pi

    def test_act_sampled_respects_mask(self, model, dataset, catalog):
        model.eval()
        data = dataset[0]
        mask = torch.zeros(1, len(catalog))
        mask[0, 3:] = -1e9
        for _ in range(5):
            decision = model.act(data, catalog.embeddings, mask, sample=True)
            assert decision["synthon_idx"][0].item() < 3

    def test_act_temperature(self, model, dataset, catalog):
        """Extreme low temperature must converge to the argmax."""
        model.eval()
        data = dataset[0]
        greedy = model.act(data, catalog.embeddings, sample=False)
        picks = set()
        for _ in range(10):
            d = model.act(data, catalog.embeddings, sample=True, temperature=1e-5)
            picks.add(d["synthon_idx"][0].item())
        assert picks == {greedy["synthon_idx"][0].item()}
