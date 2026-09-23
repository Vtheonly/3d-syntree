"""Architecture-correctness battery (bug reports 2 and 3).

These tests verify the *conceptual* fixes, not just shape compatibility:

* Flaw 1 (ghost ligand): the policy output must depend on the intermediate
  ligand it has already grown.
* Flaw 2 (torsion blindness): the torsion head must depend on the selected
  synthon's embedding.
* Flaw 3 (blind STOP): the termination logit must depend on global
  ligand-size features.
* Tell 1 (Franken tensor): chemical features and positions must travel in
  separate tensors.
* SE(3) equivariance of the joint encoding with the ligand present.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import Batch, Data

from syntree.models.bipartite import BipartitePaiNN
from syntree.models.policy import GLOBAL_FEATURE_DIM, SynTreePolicy
from syntree.models.torsion_head import ContinuousTorsionHead


@pytest.fixture(scope="module")
def pocket_mol():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CC(=O)NCC(=O)O")  # tiny "protein" fragment
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


@pytest.fixture()
def model(tiny_config):
    torch.manual_seed(0)
    m = SynTreePolicy(tiny_config)
    m.eval()
    return m


@pytest.fixture()
def state():
    """A single-graph state with pocket, ligand and a flagged handle node."""
    torch.manual_seed(7)
    pocket_pos = torch.randn(8, 3)
    pocket_z = torch.tensor([6, 7, 8, 6, 7, 8, 6, 7])
    ligand_pos = torch.randn(5, 3) * 0.5 + 1.0
    ligand_z = torch.tensor([6, 6, 7, 8, 6])
    return Data(
        pocket_pos=pocket_pos,
        pocket_z=pocket_z,
        pocket_charge=torch.zeros(8),
        pocket_batch=torch.zeros(8, dtype=torch.long),
        ligand_pos=ligand_pos,
        ligand_z=ligand_z,
        ligand_charge=torch.zeros(5),
        ligand_batch=torch.zeros(5, dtype=torch.long),
        handle_features=torch.randn(1, 64),
        handle_pos=torch.randn(1, 3),
        handle_nodes=torch.tensor([2]),
        global_features=torch.tensor([[0.4, 0.12, 0.05, 1.0]]),
        stop_mask=torch.tensor(0.0),
    )


@pytest.fixture()
def synthon_embeddings():
    torch.manual_seed(1)
    return torch.randn(7, 32)


class TestGhostLigand:
    """Flaw 1: the state must contain (and the policy must consume) the
    intermediate ligand, not only the reacting handle."""

    def test_policy_sees_the_ligand(self, model, state, synthon_embeddings):
        moved = Data(**{k: v for k, v in state.items()})
        moved.ligand_pos = state.ligand_pos + torch.tensor([3.0, 0.0, 0.0])
        with torch.no_grad():
            o1 = model(state, synthon_embeddings)
            o2 = model(moved, synthon_embeddings)
        diff = (o1["synthon_logits"] - o2["synthon_logits"]).abs().max().item()
        assert diff > 1e-5, (
            "The policy is blind to the intermediate ligand it has grown "
            f"(max logit change {diff:.2e} after moving the whole ligand)"
        )

    def test_policy_sees_ligand_elements(self, model, state, synthon_embeddings):
        swapped = Data(**{k: v for k, v in state.items()})
        swapped.ligand_z = torch.tensor([7, 7, 7, 7, 7])
        with torch.no_grad():
            o1 = model(state, synthon_embeddings)
            o2 = model(swapped, synthon_embeddings)
        diff = (o1["synthon_logits"] - o2["synthon_logits"]).abs().max().item()
        assert diff > 1e-5, "The policy ignores the ligand's element types"

    def test_empty_ligand_fallback(self, model, synthon_embeddings):
        """States without an intermediate ligand (seed step) still work."""
        torch.manual_seed(3)
        empty = Data(
            pocket_pos=torch.randn(8, 3),
            pocket_z=torch.tensor([6, 7, 8, 6, 7, 8, 6, 7]),
            pocket_batch=torch.zeros(8, dtype=torch.long),
            ligand_pos=torch.zeros(0, 3),
            ligand_z=torch.zeros(0, dtype=torch.long),
            handle_features=torch.randn(1, 64),
            handle_pos=torch.randn(1, 3),
            global_features=torch.zeros(1, GLOBAL_FEATURE_DIM),
            stop_mask=torch.tensor(0.0),
        )
        out = model(empty, synthon_embeddings)
        assert out["synthon_logits"].shape == (1, 8)

    def test_legacy_67_dim_features_split_transparently(
        self, model, synthon_embeddings
    ):
        torch.manual_seed(3)
        legacy = Data(
            pocket_pos=torch.randn(8, 3),
            pocket_z=torch.tensor([6, 7, 8, 6, 7, 8, 6, 7]),
            pocket_batch=torch.zeros(8, dtype=torch.long),
            handle_features=torch.randn(1, 67),
            stop_mask=torch.tensor(0.0),
        )
        out = model(legacy, synthon_embeddings)
        assert out["synthon_logits"].shape == (1, 8)

    def test_bipartite_encoder_ligand_edges(self):
        """Ligand atoms must actually exchange messages with pocket atoms."""
        torch.manual_seed(0)
        enc = BipartitePaiNN(hidden_dim=16, num_layers=2, num_radial=6).eval()
        pocket_pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]
        )
        pocket_z = torch.tensor([6, 7, 8])
        ligand_pos = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        ligand_z = torch.tensor([6, 6])
        batch = torch.zeros(5, dtype=torch.long)

        s0, v0, p0 = enc(pocket_pos, pocket_z, batch[:3])
        s1, v1, p1 = enc(
            pocket_pos, pocket_z, batch[:3], None, ligand_pos, ligand_z, batch[3:]
        )
        # Ligand nodes close to the pocket must influence pocket encodings.
        assert not torch.allclose(s0, s1[:3], atol=1e-6)
        # Ligand nodes are encoded too.
        assert s1.size(0) == 5

    def test_batching_with_unequal_ligand_sizes(self, model, synthon_embeddings):
        torch.manual_seed(3)
        batched = Data(
            pocket_pos=torch.randn(16, 3),
            pocket_z=torch.randint(6, 9, (16,)),
            pocket_charge=torch.zeros(16),
            pocket_batch=torch.tensor([0] * 8 + [1] * 8),
            ligand_pos=torch.randn(8, 3),
            ligand_z=torch.randint(6, 9, (8,)),
            ligand_charge=torch.zeros(8),
            ligand_batch=torch.tensor([0] * 5 + [1] * 3),
            handle_features=torch.randn(2, 64),
            handle_pos=torch.randn(2, 3),
            handle_nodes=torch.tensor([2, 1]),
            global_features=torch.randn(2, GLOBAL_FEATURE_DIM),
            stop_mask=torch.zeros(2),
        )
        out = model(batched, synthon_embeddings)
        assert out["synthon_logits"].shape == (2, 8)
        assert out["state_value"].shape == (2,)


class TestTorsionSynthonConditioning:
    """Flaw 2: the dihedral prediction must depend on the selected synthon."""

    def test_mu_depends_on_synthon(self, model, state, synthon_embeddings):
        with torch.no_grad():
            out = model(state, synthon_embeddings)
            context = out["pocket_context"]
            vectors = out["joint_vectors"]
            ub = out["union_batch"]
            mu_a, _ = model.torsion_head(context, vectors, ub, synthon_embeddings[:1])
            mu_b, _ = model.torsion_head(context, vectors, ub, synthon_embeddings[1:2])
        assert (mu_a - mu_b).abs().item() > 1e-5, (
            "The torsion head predicts the same angle for every synthon"
        )

    def test_act_dihedral_changes_with_selected_synthon(self, model, state):
        """End-to-end: two catalogs with different embeddings must be able
        to yield different dihedral predictions for the same state."""
        torch.manual_seed(11)
        emb_a = torch.randn(7, 32)
        emb_b = torch.randn(7, 32)
        with torch.no_grad():
            d_a = model.act(state, emb_a)
            d_b = model.act(state, emb_b)
        # The distributions differ in at least one of mu / action.
        differs = (
            (d_a["torsion_mu"] - d_b["torsion_mu"]).abs().item() > 1e-6
            or (d_a["action_idx"] - d_b["action_idx"]).abs().item() > 0
            or (d_a["torsion_kappa"] - d_b["torsion_kappa"]).abs().item() > 1e-6
        )
        assert differs

    def test_teacher_forcing_runs_through_forward(self, model, state, synthon_embeddings):
        out = model(
            state,
            synthon_embeddings,
            synthon_embedding_input=synthon_embeddings[3:4],
        )
        assert torch.isfinite(out["torsion_mu"]).all()
        assert torch.isfinite(out["torsion_kappa"]).all()


class TestStopGlobalFeatures:
    """Flaw 3: the STOP decision must see the ligand's global size."""

    def test_stop_logit_depends_on_global_features(
        self, model, state, synthon_embeddings
    ):
        grown = Data(**{k: v for k, v in state.items()})
        grown.global_features = torch.tensor([[2.0, 0.9, 1.5, 1.0]])  # big ligand
        with torch.no_grad():
            o1 = model(state, synthon_embeddings)
            o2 = model(grown, synthon_embeddings)
        stop_idx = synthon_embeddings.size(0)
        diff = (
            o1["synthon_logits"][0, stop_idx] - o2["synthon_logits"][0, stop_idx]
        ).abs().item()
        assert diff > 1e-5, (
            "The STOP logit ignores global ligand-size features"
        )

    def test_global_features_definition(self):
        from syntree.data.featurizer import MolecularFeaturizer

        empty = MolecularFeaturizer.ligand_global_features(None)
        assert empty.shape == (GLOBAL_FEATURE_DIM,)
        assert empty.abs().sum() == 0.0

        from rdkit import Chem

        lig = Chem.MolFromSmiles("C1CCCCC1C(=O)N2CCCCC2")
        feats = MolecularFeaturizer.ligand_global_features(lig, pocket_volume=1000.0)
        assert feats.shape == (GLOBAL_FEATURE_DIM,)
        assert feats[0] > 0.2  # MW / 500
        assert feats[1] > 0.05  # heavy atoms / 50
        assert 0.0 < feats[2] < 1.0  # volume ratio
        assert feats[3] == 1.0  # has_ligand


class TestNoFrankenTensors:
    """Tell 1: chemical features and positions live in separate tensors."""

    def test_handle_featurization_split(self, pocket_mol):
        from syntree.data.featurizer import (
            HANDLE_FEATURE_DIM,
            MolecularFeaturizer,
        )

        chem = MolecularFeaturizer.featurize_handle(pocket_mol, [0])
        pos = MolecularFeaturizer.featurize_handle_position(pocket_mol, [0])
        assert chem.shape == (HANDLE_FEATURE_DIM,)
        assert HANDLE_FEATURE_DIM == 64
        assert pos.shape == (3,)

    def test_ligand_featurization(self):
        from syntree.data.featurizer import MolecularFeaturizer
        from rdkit import Chem
        from syntree.chemistry.conformer import ConformerEngine

        mol = ConformerEngine.embed_product(Chem.MolFromSmiles("CCC(=O)N1CCCCC1"))
        feats = MolecularFeaturizer.featurize_ligand(
            mol, center=torch.zeros(3)
        )
        assert feats["ligand_pos"].shape[1] == 3
        assert feats["ligand_z"].shape[0] == feats["ligand_pos"].shape[0]
        assert feats["ligand_pos"].norm(dim=-1).max() < 10.0  # centered
        # heavy atom map points to valid ligand node indices
        n = feats["ligand_pos"].size(0)
        assert all(0 <= v < n for v in feats["heavy_atom_map"].values())
        assert len(feats["heavy_atom_map"]) == n


class TestJointSE3:
    """SE(3) invariance of the full joint (pocket + ligand) policy."""

    @staticmethod
    def _rotation(theta, axis=2):
        c, s = math.cos(theta), math.sin(theta)
        R = torch.eye(3)
        i, j = (axis + 1) % 3, (axis + 2) % 3
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        return R

    def test_rotation_invariance_with_ligand(
        self, model, state, synthon_embeddings
    ):
        rotated = Data(**{k: v for k, v in state.items()})
        R = self._rotation(1.1)
        rotated.pocket_pos = state.pocket_pos @ R.T
        rotated.ligand_pos = state.ligand_pos @ R.T
        rotated.handle_pos = state.handle_pos @ R.T
        with torch.no_grad():
            o1 = model(state, synthon_embeddings)
            o2 = model(rotated, synthon_embeddings)
        assert torch.allclose(
            o1["synthon_logits"], o2["synthon_logits"], atol=1e-4
        )

    def test_translation_invariance_with_ligand(
        self, model, state, synthon_embeddings
    ):
        shifted = Data(**{k: v for k, v in state.items()})
        t = torch.tensor([7.0, -4.0, 2.0])
        shifted.pocket_pos = state.pocket_pos + t
        shifted.ligand_pos = state.ligand_pos + t
        shifted.handle_pos = state.handle_pos + t
        with torch.no_grad():
            o1 = model(state, synthon_embeddings)
            o2 = model(shifted, synthon_embeddings)
        assert torch.allclose(
            o1["pocket_context"], o2["pocket_context"], atol=1e-4
        )


class TestTorsionHeadContract:
    """The synthon-conditioned torsion head keeps its mathematical contract."""

    def test_kappa_bounds(self, model, state, synthon_embeddings):
        with torch.no_grad():
            out = model(state, synthon_embeddings)
        assert (out["torsion_kappa"] >= 0.1).all()
        assert (out["torsion_kappa"] <= 50.0).all()
        assert (out["torsion_mu"] >= -math.pi).all()
        assert (out["torsion_mu"] < math.pi).all()

    def test_bad_synthon_shape_raises(self):
        head = ContinuousTorsionHead(32)
        ctx = torch.randn(2, 32)
        with pytest.raises(ValueError):
            head(ctx, synthon_embedding=torch.randn(3, 32))
        with pytest.raises(ValueError):
            head(ctx, synthon_embedding=torch.randn(2, 16))

    def test_null_synthon_used_when_missing(self):
        head = ContinuousTorsionHead(32)
        ctx = torch.randn(2, 32)
        mu1, k1 = head(ctx)
        mu2, k2 = head(
            ctx,
            synthon_embedding=head.null_synthon.unsqueeze(0).expand(2, -1),
        )
        assert torch.allclose(mu1, mu2)
        assert torch.allclose(k1, k2)
