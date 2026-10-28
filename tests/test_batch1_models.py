"""Batch 1 Test Suite: Equivariant Backbone, Mathematical Heads, and Bipartite Encoder.

Designed specifically for CPU execution on 8 GB RAM machines.
Tests expected behavior, edge cases, numerical precision, and known defects.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Limit CPU threads for Intel Core i3 stability
import torch
torch.set_num_threads(2)

# Prevent syntree/models/__init__.py from eagerly pulling in policy.py and the entire chemistry/RDKit cascade
sys.modules["syntree.models.policy"] = MagicMock()

from syntree.models.equivariant import (
    CosineCutoff,
    RadialBasis,
    PaiNNInteraction,
    PaiNNMixing,
    PaiNNLayer,
    BatchedPaiNNLayer,
    build_radius_graph,
)
from syntree.models.torsion_head import ContinuousTorsionHead, log_bessel_i0
from syntree.models.reaction_head import ReactionHead
from syntree.models.synthon_head import SynthonHead
from syntree.models.bipartite import BipartitePaiNN


class TestEquivariantBackbone(unittest.TestCase):
    """Tests for syntree/models/equivariant.py."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.hidden_dim = 32
        self.num_radial = 10
        self.cutoff = 5.0

    def test_01_radial_basis_cutoff_behavior(self):
        """Radial basis values must decay to exactly 0.0 beyond cutoff distance."""
        print("\n[TEST] Equivariant: RadialBasis Cutoff Envelope")
        rbf = RadialBasis(num_radial=self.num_radial, cutoff=self.cutoff)
        
        # Test distances: inside cutoff, exactly at cutoff, and far beyond
        distances = torch.tensor([1.0, 2.5, 4.99, 5.0, 5.5, 10.0], dtype=torch.float32)
        out = rbf(distances)
        
        self.assertEqual(out.shape, (6, self.num_radial))
        # Beyond cutoff (index 4 and 5) must be identically 0.0
        self.assertTrue(torch.all(out[4:] == 0.0), f"RBF did not zero out beyond cutoff: {out[4:]}")
        # Inside cutoff (index 0, 1) must have non-zero energy
        self.assertTrue(torch.all(out[:2].sum(dim=-1) > 0.0))
        print(f"  -> PASS: RadialBasis correctly zeroes beyond {self.cutoff} A.")

    def test_02_build_radius_graph_correctness_and_batch_isolation(self):
        """Radius graph must never create edges crossing different graphs in a batch."""
        print("\n[TEST] Equivariant: build_radius_graph Batch Separation")
        # Graph 0: 3 atoms clustered near (0, 0, 0)
        pos_g0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Graph 1: 3 atoms clustered near (100, 100, 100) -> far from g0, but close to each other
        pos_g1 = torch.tensor([[100.0, 100.0, 100.0], [101.0, 100.0, 100.0], [100.0, 101.0, 100.0]])
        
        pos = torch.cat([pos_g0, pos_g1], dim=0)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        
        edge_index = build_radius_graph(pos, batch, cutoff=self.cutoff)
        row, col = edge_index[0], edge_index[1]
        
        # Verify no edge connects an atom in Graph 0 to an atom in Graph 1
        cross_edges = (batch[row] != batch[col]).sum().item()
        self.assertEqual(cross_edges, 0, f"Found {cross_edges} cross-graph edges in radius graph!")
        print(f"  -> PASS: Generated {edge_index.size(1)} valid intra-graph edges, 0 cross-graph edges.")

    def test_03_painn_layer_equivariance(self):
        """Rotating input 3D coordinates must rotate vector features by the exact same rotation matrix."""
        print("\n[TEST] Equivariant: PaiNN Rotational Equivariance Check")
        layer = PaiNNLayer(hidden_dim=self.hidden_dim, num_radial=self.num_radial, cutoff=self.cutoff)
        layer.eval()

        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.3]])
        batch = torch.zeros(4, dtype=torch.long)
        scalar = torch.randn(4, self.hidden_dim)
        vector = torch.zeros(4, 3, self.hidden_dim)

        edge_index = build_radius_graph(pos, batch, cutoff=self.cutoff)

        with torch.no_grad():
            s1, v1 = layer(scalar, vector, edge_index, pos)

        # Apply a 90-degree rotation around the Z axis
        R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        pos_rot = pos @ R.T
        edge_index_rot = build_radius_graph(pos_rot, batch, cutoff=self.cutoff)

        with torch.no_grad():
            s2, v2 = layer(scalar, vector, edge_index_rot, pos_rot)

        # Scalar features must be strictly invariant: s1 == s2
        scalar_diff = torch.max(torch.abs(s1 - s2)).item()
        self.assertLess(scalar_diff, 1e-5, f"PaiNN scalar channel is not rotation invariant! Max diff: {scalar_diff}")

        # Vector features have shape [N, 3, d]; R has shape [3, 3] and rotates dimension 1 (spatial)
        v1_rotated = torch.einsum("ij,njd->nid", R, v1)
        vector_diff = torch.max(torch.abs(v2 - v1_rotated)).item()
        self.assertLess(vector_diff, 1e-4, f"PaiNN vector channel is not rotation equivariant! Max diff: {vector_diff}")
        print(f"  -> PASS: Scalar invariance diff={scalar_diff:.2e}, Vector equivariance diff={vector_diff:.2e}")

class TestTorsionHead(unittest.TestCase):
    """Tests for syntree/models/torsion_head.py and known mathematical issues."""

    def setUp(self):
        self.hidden_dim = 32
        self.head = ContinuousTorsionHead(hidden_dim=self.hidden_dim, min_kappa=0.1, max_kappa=50.0)

    def test_04_log_bessel_dtype_leak(self):
        """DIAGNOSTIC TEST (F-10): log_bessel_i0 must return float32 to prevent FP64 promotion."""
        print("\n[TEST] TorsionHead: Bessel Dtype Consistency (Auditing F-10)")
        kappa = torch.tensor([0.5, 2.0, 15.0], dtype=torch.float32)
        out = log_bessel_i0(kappa)
        
        print(f"  -> Diagnostic: Input dtype={kappa.dtype}, log_bessel_i0 output dtype={out.dtype}")
        if out.dtype == torch.float64:
            print("  -> [CONFIRMED VULNERABILITY F-10]: log_bessel_i0 promotes tensor to torch.float64!")
        self.assertEqual(out.dtype, torch.float32, "log_bessel_i0 returned float64; promotes outer training loss to float64!")

    def test_05_loss_fn_shape_handling(self):
        """DIAGNOSTIC TEST (F-08): loss_fn must accept target dihedrals whether shape is [B] or [B, 1]."""
        print("\n[TEST] TorsionHead: Target Dihedral Dimensionality (Auditing F-08)")
        b = 4
        mu = torch.zeros(b, dtype=torch.float32)
        kappa = torch.ones(b, dtype=torch.float32) * 2.0
        
        # Test 1: Shape [B]
        target_flat = torch.zeros(b, dtype=torch.float32)
        try:
            loss1 = ContinuousTorsionHead.loss_fn(mu, kappa, target_flat)
            self.assertEqual(loss1.dim(), 0)
            print("  -> Flat shape [B] evaluated successfully.")
        except Exception as exc:
            self.fail(f"loss_fn failed on standard [B] target: {exc}")

        # Test 2: Shape [B, 1] (as collated by certain DataLoader shards)
        target_2d = torch.zeros(b, 1, dtype=torch.float32)
        try:
            loss2 = ContinuousTorsionHead.loss_fn(mu, kappa, target_2d)
            self.assertEqual(loss2.dim(), 0)
            print("  -> 2D shape [B, 1] evaluated successfully.")
        except ValueError as exc:
            print(f"  -> [CONFIRMED VULNERABILITY F-08]: loss_fn rejected [B, 1] target: {exc}")
            raise

    def test_06_von_mises_sampling_range(self):
        """DIAGNOSTIC TEST (F-12): Sampled dihedrals must explore full circle [-pi, pi), not capped at 1 rad (57.3 deg)."""
        print("\n[TEST] TorsionHead: Best-Fisher Sampling Arc Range (Auditing F-12)")
        mu = torch.zeros(1000, dtype=torch.float32)
        kappa = torch.full((1000,), 0.1, dtype=torch.float32)  # Very low kappa = near-uniform on [-pi, pi)
        
        samples = ContinuousTorsionHead.sample(mu, kappa)
        max_angle = samples.abs().max().item()
        
        print(f"  -> Sampled 1000 dihedrals at kappa=0.1. Max observed absolute angle: {max_angle:.4f} rad ({math.degrees(max_angle):.1f} deg)")
        if max_angle <= 1.05:
            print("  -> [CONFIRMED VULNERABILITY F-12]: Samples never exceed 1.0 rad! Uniform sampler restricted to [0, 1) instead of [0, pi).")
        self.assertGreater(max_angle, 1.57, f"Sampled dihedrals restricted to < 90 deg under uniform concentration! Max angle: {max_angle:.3f} rad")


class TestReactionAndSynthonHeads(unittest.TestCase):
    """Tests for syntree/models/reaction_head.py and syntree/models/synthon_head.py."""

    def setUp(self):
        self.hidden_dim = 32
        self.num_families = 9
        self.num_synthons = 50
        self.batch_size = 3

    def test_07_reaction_head_masking(self):
        """Masked reaction families (-1e9) must receive zero probability after softmax."""
        print("\n[TEST] ReactionHead: Compatibility Mask Enforcement")
        head = ReactionHead(hidden_dim=self.hidden_dim, num_families=self.num_families)
        context = torch.randn(self.batch_size, self.hidden_dim)
        
        # Mask out family 0 and family 3 for all graphs
        mask = torch.zeros(self.batch_size, self.num_families)
        mask[:, 0] = -1e9
        mask[:, 3] = -1e9
        
        logits, log_probs = head(context, compatibility_mask=mask)
        probs = log_probs.exp()
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_families))
        self.assertTrue(torch.all(probs[:, 0] < 1e-6), f"Masked family 0 received non-zero prob: {probs[:, 0]}")
        self.assertTrue(torch.all(probs[:, 3] < 1e-6), f"Masked family 3 received non-zero prob: {probs[:, 3]}")
        print("  -> PASS: Incompatible reaction families masked out to prob ~ 0.0.")

    def test_08_synthon_head_forward_and_stop_action(self):
        """SynthonHead must output K+1 logits (index K = STOP) and respect stop_mask."""
        print("\n[TEST] SynthonHead: K+1 Action Output & STOP Masking")
        head = SynthonHead(hidden_dim=self.hidden_dim, num_heads=4)
        query = torch.randn(self.batch_size, self.hidden_dim)
        catalog_embeddings = torch.randn(self.num_synthons, self.hidden_dim)
        
        # Test without stop mask
        logits, log_probs = head(query, catalog_embeddings)
        self.assertEqual(logits.shape, (self.batch_size, self.num_synthons + 1), "SynthonHead output dim must be K + 1!")
        
        # Test with stop_mask = -1e9 (prohibiting STOP when molecule MW is below cap)
        stop_mask = torch.full((self.batch_size,), -1e9)
        logits_masked, log_probs_masked = head(query, catalog_embeddings, stop_mask=stop_mask)
        probs = log_probs_masked.exp()
        
        stop_prob = probs[:, -1]
        self.assertTrue(torch.all(stop_prob < 1e-6), f"STOP action received probability despite stop_mask: {stop_prob}")
        print("  -> PASS: STOP logit correctly appended at index K and suppressed when masked.")


class TestBipartitePaiNN(unittest.TestCase):
    """Tests for syntree/models/bipartite.py."""

    def setUp(self):
        self.hidden_dim = 32
        self.num_layers = 2
        self.num_radial = 8
        self.cutoff = 5.0
        self.encoder = BipartitePaiNN(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_radial=self.num_radial,
            cutoff=self.cutoff,
            max_atomic_number=50,
        )

    def test_09_bipartite_pocket_only_mode(self):
        """With ligand_pos=None, BipartitePaiNN must degenerate gracefully to pocket-only encoding."""
        print("\n[TEST] BipartitePaiNN: Pocket-Only Degeneration")
        n_pocket = 12
        pocket_pos = torch.randn(n_pocket, 3)
        pocket_z = torch.randint(1, 10, (n_pocket,))
        pocket_batch = torch.tensor([0]*6 + [1]*6, dtype=torch.long)
        
        scalar, vector, pooled = self.encoder(
            pocket_pos=pocket_pos,
            pocket_z=pocket_z,
            pocket_batch=pocket_batch,
        )
        
        self.assertEqual(scalar.shape, (n_pocket, self.hidden_dim))
        self.assertEqual(vector.shape, (n_pocket, 3, self.hidden_dim))
        self.assertEqual(pooled.shape, (2, self.hidden_dim))  # Batch size = 2
        print(f"  -> PASS: Pocket-only mode produced scalar {list(scalar.shape)}, pooled {list(pooled.shape)}.")

    def test_10_bipartite_joint_pocket_and_ligand_graph(self):
        """When ligand atoms are provided, the joint graph must concatenate nodes and honor formal charges."""
        print("\n[TEST] BipartitePaiNN: Joint Pocket + Ligand Encoding")
        n_p, n_l = 8, 4
        pocket_pos = torch.randn(n_p, 3)
        pocket_z = torch.randint(1, 10, (n_p,))
        pocket_batch = torch.zeros(n_p, dtype=torch.long)
        pocket_charge = torch.tensor([-0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        ligand_pos = torch.randn(n_l, 3)
        ligand_z = torch.randint(1, 10, (n_l,))
        ligand_batch = torch.zeros(n_l, dtype=torch.long)
        ligand_charge = torch.tensor([0.0, 1.0, 0.0, -1.0], dtype=torch.float32)

        handle_flag = torch.zeros(n_p + n_l, dtype=torch.bool)
        handle_flag[n_p + 1] = True  # Atom 1 of ligand is the reacting handle

        scalar, vector, pooled = self.encoder(
            pocket_pos=pocket_pos,
            pocket_z=pocket_z,
            pocket_batch=pocket_batch,
            pocket_charge=pocket_charge,
            ligand_pos=ligand_pos,
            ligand_z=ligand_z,
            ligand_batch=ligand_batch,
            ligand_charge=ligand_charge,
            handle_node=handle_flag,
        )

        n_total = n_p + n_l
        self.assertEqual(scalar.shape, (n_total, self.hidden_dim))
        self.assertEqual(vector.shape, (n_total, 3, self.hidden_dim))
        self.assertEqual(pooled.shape, (1, self.hidden_dim))
        self.assertFalse(torch.isnan(scalar).any(), "NaN detected in scalar features!")
        print(f"  -> PASS: Joint graph ({n_total} atoms) encoded cleanly with formal charges and handle node flag.")


if __name__ == "__main__":
    print("=" * 70)
    print(" 3D-SYNTREE LOCAL VERIFICATION SUITE: BATCH 1 (NEURAL CORE & HEADS)")
    print(f" Platform: {sys.platform} | PyTorch: {torch.__version__} | Device: CPU")
    print(" Hardware Target: Intel Core i3 (11th Gen) / 8 GB RAM")
    print("=" * 70)
    unittest.main(verbosity=2)