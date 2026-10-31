"""#6 ARCH-02 — frozen state tensor schema & PyG batch contract tests.

Acceptance criteria (issue #6):
* instantiating batches conforming to the schema passes type and shape
  checks;
* passing a 67-D tensor raises an explicit
  ``ValueError("Legacy 67D Franken-tensors are deprecated")``.

Beyond the two acceptance bullets, this file pins the full fail-closed
contract: dtype/shape enforcement, cross-field consistency, batch-index
ranges, the canonical collator (empty ligands at t=0, unequal ligand
sizes, 1-D [64] normalisation, batch-vector materialisation, schema
stamping) and the integration with ``SynTreePolicy.forward``.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from syntree.core.schema import (
    GLOBAL_FEATURE_DIM,
    HANDLE_FEATURE_DIM,
    LEGACY_FRANKEN_TENSOR_MESSAGE,
    LEGACY_HANDLE_FEATURE_DIM,
    SCHEMA_VERSION,
    StateTensorContract,
    TensorContractError,
    collate_state_batch,
    stamp_schema_version,
    validate_state_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(
    n_pocket: int = 10,
    n_ligand: int = 5,
    batch_size: int = 1,
    seed: int = 0,
    *,
    with_batch_vectors: bool = True,
) -> Data:
    """A conforming single-graph state (ligand may be empty)."""
    g = torch.Generator().manual_seed(seed)
    data = dict(
        pocket_pos=torch.randn(n_pocket, 3, generator=g),
        pocket_z=torch.randint(1, 30, (n_pocket,), generator=g),
        pocket_charge=torch.zeros(n_pocket),
        ligand_pos=torch.randn(n_ligand, 3, generator=g),
        ligand_z=torch.randint(6, 9, (n_ligand,), generator=g),
        ligand_charge=torch.zeros(n_ligand),
        handle_features=torch.randn(batch_size, HANDLE_FEATURE_DIM, generator=g),
        handle_pos=torch.randn(batch_size, 3, generator=g),
        handle_nodes=torch.tensor([0 if n_ligand else -1] * batch_size),
        global_features=torch.zeros(batch_size, GLOBAL_FEATURE_DIM),
        stop_mask=torch.zeros(batch_size),
    )
    if with_batch_vectors:
        data["pocket_batch"] = torch.zeros(n_pocket, dtype=torch.long)
        data["ligand_batch"] = torch.zeros(n_ligand, dtype=torch.long)
    return Data(num_nodes=n_pocket, **data)


def make_batched_state(
    pocket_sizes, ligand_sizes, seed: int = 0
) -> Data:
    """A conforming multi-graph batch (canonical batch vectors included)."""
    assert len(pocket_sizes) == len(ligand_sizes)
    b = len(pocket_sizes)
    g = torch.Generator().manual_seed(seed)
    pocket_batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(pocket_sizes)]
    )
    ligand_batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(ligand_sizes)]
    )
    return Data(
        pocket_pos=torch.randn(sum(pocket_sizes), 3, generator=g),
        pocket_z=torch.randint(1, 30, (sum(pocket_sizes),), generator=g),
        pocket_charge=torch.zeros(sum(pocket_sizes)),
        pocket_batch=pocket_batch,
        ligand_pos=torch.randn(sum(ligand_sizes), 3, generator=g),
        ligand_z=torch.randint(6, 9, (sum(ligand_sizes),), generator=g),
        ligand_charge=torch.zeros(sum(ligand_sizes)),
        ligand_batch=ligand_batch,
        handle_features=torch.randn(b, HANDLE_FEATURE_DIM, generator=g),
        handle_pos=torch.randn(b, 3, generator=g),
        handle_nodes=torch.tensor([0 if n else -1 for n in ligand_sizes]),
        global_features=torch.zeros(b, GLOBAL_FEATURE_DIM),
        stop_mask=torch.zeros(b),
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 1: conforming states pass type and shape checks
# ---------------------------------------------------------------------------


class TestContractInstantiation:
    def test_single_graph_state_passes(self):
        state = make_state()
        contract = validate_state_batch(state)
        assert isinstance(contract, StateTensorContract)
        assert contract.batch_size == 1
        assert contract.num_pocket == 10
        assert contract.num_ligand == 5
        assert contract.schema_version == SCHEMA_VERSION

    def test_empty_ligand_t0_state_passes(self):
        """The t=0 state (no ligand grown yet) conforms to the schema."""
        state = make_state(n_ligand=0)
        contract = validate_state_batch(state)
        assert contract.num_ligand == 0
        assert contract.ligand_pos.shape == (0, 3)

    def test_multi_graph_batch_passes(self):
        state = make_batched_state([8, 6, 5], [3, 0, 2])
        contract = validate_state_batch(state)
        assert contract.batch_size == 3
        assert contract.num_pocket == 19
        assert contract.num_ligand == 5

    def test_dataclass_direct_construction_validates(self):
        """StateTensorContract(...) validates on construction (#6 AC)."""
        state = make_state()
        contract = StateTensorContract(
            pocket_pos=state.pocket_pos,
            pocket_z=state.pocket_z,
            pocket_charge=state.pocket_charge,
            pocket_batch=state.pocket_batch,
            ligand_pos=state.ligand_pos,
            ligand_z=state.ligand_z,
            ligand_charge=state.ligand_charge,
            ligand_batch=state.ligand_batch,
            handle_features=state.handle_features,
            handle_pos=state.handle_pos,
            global_features=state.global_features,
            stop_mask=state.stop_mask,
        )
        assert contract.batch_size == 1

    def test_frozen_dataclass(self):
        contract = validate_state_batch(make_state())
        with pytest.raises(Exception):
            contract.pocket_pos = torch.zeros(1, 3)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: legacy 67-D Franken tensors raise hard ValueError
# ---------------------------------------------------------------------------


class TestLegacyFrankenTensorRejection:
    def test_67d_2d_tensor_raises_exact_message(self):
        state = make_state()
        state.handle_features = torch.randn(2, LEGACY_HANDLE_FEATURE_DIM)
        with pytest.raises(ValueError, match=r"Legacy 67D Franken-tensors are deprecated"):
            validate_state_batch(state)

    def test_67d_1d_tensor_raises(self):
        state = make_state()
        state.handle_features = torch.randn(LEGACY_HANDLE_FEATURE_DIM)
        with pytest.raises(ValueError, match=r"Legacy 67D Franken-tensors are deprecated"):
            validate_state_batch(state)

    def test_67d_rejected_even_when_other_fields_missing(self):
        """The Franken rejection takes precedence over missing-field
        diagnostics so legacy states get the actionable message."""
        state = Data(
            pocket_pos=torch.randn(8, 3),
            pocket_z=torch.randint(1, 9, (8,)),
            handle_features=torch.randn(1, LEGACY_HANDLE_FEATURE_DIM),
        )
        with pytest.raises(ValueError, match=r"Legacy 67D Franken-tensors are deprecated"):
            validate_state_batch(state)

    def test_67d_rejected_through_collate(self):
        state = make_state()
        state.handle_features = torch.randn(1, LEGACY_HANDLE_FEATURE_DIM)
        with pytest.raises(ValueError, match=r"Legacy 67D Franken-tensors are deprecated"):
            collate_state_batch([state])

    def test_67d_rejected_by_policy_forward(self, tiny_config):
        from syntree.models.policy import SynTreePolicy

        torch.manual_seed(0)
        model = SynTreePolicy(tiny_config).eval()
        state = make_state()
        state.handle_features = torch.randn(1, LEGACY_HANDLE_FEATURE_DIM)
        with pytest.raises(ValueError, match=r"Legacy 67D Franken-tensors are deprecated"):
            model(state, torch.randn(5, 32))

    def test_exact_message_text(self):
        """The deprecation string is a stable public constant."""
        assert LEGACY_FRANKEN_TENSOR_MESSAGE.startswith(
            "Legacy 67D Franken-tensors are deprecated"
        )


# ---------------------------------------------------------------------------
# Fail-closed validation of every field
# ---------------------------------------------------------------------------


class TestFailClosedValidation:
    def test_missing_required_field_names_it(self):
        state = make_state()
        del state.handle_pos
        with pytest.raises(TensorContractError, match="handle_pos"):
            validate_state_batch(state)

    def test_missing_handle_pos_only(self):
        """Explicit existence of handle_pos [B, 3] is enforced (#6)."""
        state = make_state()
        del state.handle_pos
        with pytest.raises(TensorContractError, match="handle_pos"):
            validate_state_batch(state)

    def test_wrong_handle_feature_dim(self):
        state = make_state()
        state.handle_features = torch.randn(1, 32)
        with pytest.raises(TensorContractError, match="handle_features"):
            validate_state_batch(state)

    def test_1d_handle_features_rejected_at_validation(self):
        """Policy-side validation refuses 1-D vectors (no implicit
        reshaping); only the collator performs the documented [64]->[1,64]
        normalisation."""
        state = make_state()
        state.handle_features = torch.randn(HANDLE_FEATURE_DIM)
        with pytest.raises(TensorContractError, match="2-D"):
            validate_state_batch(state)

    def test_wrong_dtypes_rejected(self):
        state = make_state()
        state.pocket_pos = state.pocket_pos.double()
        with pytest.raises(TensorContractError, match="pocket_pos"):
            validate_state_batch(state)

        state = make_state()
        state.pocket_z = state.pocket_z.to(torch.int32)
        with pytest.raises(TensorContractError, match="pocket_z"):
            validate_state_batch(state)

        state = make_state()
        state.stop_mask = state.stop_mask.to(torch.float64)
        with pytest.raises(TensorContractError, match="stop_mask"):
            validate_state_batch(state)

    def test_inconsistent_pocket_lengths(self):
        state = make_state(n_pocket=10)
        state.pocket_z = state.pocket_z[:7]
        with pytest.raises(TensorContractError, match="pocket_z"):
            validate_state_batch(state)

    def test_inconsistent_ligand_lengths(self):
        state = make_state(n_ligand=5)
        state.ligand_charge = state.ligand_charge[:3]
        with pytest.raises(TensorContractError, match="ligand_charge"):
            validate_state_batch(state)

    def test_inconsistent_batch_size(self):
        """Per-graph tensors must agree with B derived from handle_features."""
        state = make_state()
        state.global_features = torch.zeros(2, GLOBAL_FEATURE_DIM)
        state.handle_pos = torch.zeros(2, 3)
        state.stop_mask = torch.zeros(2)
        with pytest.raises(TensorContractError, match="handle_pos"):
            validate_state_batch(state)

        state = make_state()
        state.stop_mask = torch.zeros(3)
        with pytest.raises(TensorContractError, match="stop_mask"):
            validate_state_batch(state)

    def test_batch_index_out_of_range(self):
        state = make_batched_state([6, 6], [3, 3])
        state.pocket_batch[0] = 5  # B == 2
        with pytest.raises(TensorContractError, match="pocket_batch"):
            validate_state_batch(state)

        state = make_batched_state([6, 6], [3, 3])
        state.ligand_batch[-1] = 9
        with pytest.raises(TensorContractError, match="ligand_batch"):
            validate_state_batch(state)

    def test_atomic_number_bounds(self):
        state = make_state()
        state.pocket_z[0] = 0
        with pytest.raises(TensorContractError, match=r"\[1, 100\]"):
            validate_state_batch(state)

        state = make_state()
        state.pocket_z[0] = 101
        with pytest.raises(TensorContractError, match=r"\[1, 100\]"):
            validate_state_batch(state)

    def test_nonfinite_coordinates_rejected(self):
        state = make_state()
        state.pocket_pos[0, 0] = float("nan")
        with pytest.raises(TensorContractError, match="non-finite"):
            validate_state_batch(state)

    def test_none_state_rejected(self):
        with pytest.raises(TensorContractError):
            validate_state_batch(None)

    def test_empty_pocket_rejected(self):
        state = make_state(n_pocket=0, n_ligand=3)
        with pytest.raises(TensorContractError, match="at least one atom"):
            validate_state_batch(state)


# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------


class TestSchemaVersioning:
    def test_stamped_compatible_state_passes(self):
        state = stamp_schema_version(make_state())
        contract = validate_state_batch(state)
        assert contract.schema_version == SCHEMA_VERSION

    def test_incompatible_version_rejected(self):
        state = make_state()
        state.schema_version = torch.tensor([99])
        with pytest.raises(TensorContractError, match="schema_version"):
            validate_state_batch(state)

    def test_unstamped_state_still_field_validated(self):
        """Unstamped (legacy/interim) states are accepted only if every
        field check passes; a 67-dim unstamped tensor still fails hard."""
        state = make_state()
        contract = validate_state_batch(state)  # unstamped -> fields checked
        assert contract.batch_size == 1


# ---------------------------------------------------------------------------
# The canonical collator
# ---------------------------------------------------------------------------


class TestCollateStateBatch:
    def test_canonical_batch_vectors(self):
        batch = collate_state_batch(
            [make_state(8, 3, seed=1), make_state(6, 0, seed=2), make_state(5, 2, seed=3)]
        )
        assert isinstance(batch, Batch)
        assert batch.num_graphs == 3
        assert batch.pocket_batch.tolist() == [0] * 8 + [1] * 6 + [2] * 5
        # graph 1 has an empty ligand -> no ligand nodes attributed to it
        assert batch.ligand_batch.tolist() == [0] * 3 + [2] * 2
        assert batch.handle_features.shape == (3, HANDLE_FEATURE_DIM)
        assert batch.handle_pos.shape == (3, 3)
        assert batch.global_features.shape == (3, GLOBAL_FEATURE_DIM)
        assert batch.stop_mask.shape == (3,)

    def test_no_pos_batch_aliases_left(self):
        """Exactly one representation: canonical names only."""
        batch = collate_state_batch([make_state(), make_state(5, 2, seed=2)])
        assert hasattr(batch, "pocket_batch")
        assert hasattr(batch, "ligand_batch")
        assert "pocket_pos_batch" not in batch.keys()
        assert "ligand_pos_batch" not in batch.keys()

    def test_collated_batch_is_stamped_and_valid(self):
        batch = collate_state_batch([make_state(), make_state(5, 2, seed=2)])
        assert int(batch.schema_version.min()) == SCHEMA_VERSION
        contract = validate_state_batch(batch)
        assert contract.batch_size == 2

    def test_all_empty_ligands(self):
        batch = collate_state_batch(
            [make_state(5, 0, seed=1), make_state(6, 0, seed=2)]
        )
        assert batch.ligand_pos.shape == (0, 3)
        assert batch.ligand_batch.shape == (0,)
        contract = validate_state_batch(batch)
        assert contract.num_ligand == 0

    def test_1d_handle_features_normalised(self):
        """Trajectory-builder samples carry 1-D [64]; the collator
        performs the documented unambiguous [1, 64] normalisation."""
        state = make_state()
        state.handle_features = torch.randn(HANDLE_FEATURE_DIM)
        batch = collate_state_batch([state])
        assert batch.handle_features.shape == (1, HANDLE_FEATURE_DIM)

    def test_1d_wrong_size_rejected(self):
        state = make_state()
        state.handle_features = torch.randn(13)
        with pytest.raises(TensorContractError, match="handle_features"):
            collate_state_batch([state])

    def test_sample_carried_zero_batch_vectors_stripped(self):
        """Env-style states carry zeros(N) batch vectors; the collator
        verifies and strips them (PyG would double-increment)."""
        states = [make_state(8, 3, seed=1), make_state(6, 2, seed=2)]
        batch = collate_state_batch(states)
        assert batch.pocket_batch.tolist() == [0] * 8 + [1] * 6
        assert batch.ligand_batch.tolist() == [0] * 3 + [1] * 2

    def test_sample_carried_nonzero_batch_vector_rejected(self):
        state = make_state()
        state.pocket_batch[0] = 3
        with pytest.raises(TensorContractError, match="pre-batched"):
            collate_state_batch([state])

    def test_empty_input_rejected(self):
        with pytest.raises(TensorContractError):
            collate_state_batch([])

    def test_prebatched_object_rejected(self):
        batch = collate_state_batch([make_state(), make_state(5, 2, seed=2)])
        with pytest.raises(TensorContractError, match="already a Batch"):
            collate_state_batch([batch])

    def test_input_not_mutated(self):
        state = make_state()
        before = state.handle_features.clone()
        collate_state_batch([state])
        assert torch.equal(state.handle_features, before)
        assert state.handle_features.dim() == 2  # untouched

    def test_trainer_collate_delegates_to_schema(self):
        """ResilientTrainer.collate_batch must produce identical canonical
        batches (single batching path, #6)."""
        from syntree.engine.trainer import ResilientTrainer

        states = [make_state(8, 3, seed=1), make_state(6, 0, seed=2)]
        via_trainer = ResilientTrainer.collate_batch(
            list(states), pocket_atom_cap=0, stats=None
        )
        via_schema = collate_state_batch(
            [make_state(8, 3, seed=1), make_state(6, 0, seed=2)]
        )
        assert via_trainer.num_graphs == via_schema.num_graphs
        assert via_trainer.pocket_batch.tolist() == via_schema.pocket_batch.tolist()
        assert via_trainer.ligand_batch.tolist() == via_schema.ligand_batch.tolist()
        assert hasattr(via_trainer, "handle_pos")
        assert hasattr(via_trainer, "global_features")


# ---------------------------------------------------------------------------
# Integration with the policy (the contract's primary consumer)
# ---------------------------------------------------------------------------


class TestPolicyIntegration:
    @pytest.fixture()
    def model(self, tiny_config):
        from syntree.models.policy import SynTreePolicy

        torch.manual_seed(0)
        return SynTreePolicy(tiny_config).eval()

    def test_collated_batch_through_policy(self, model):
        batch = collate_state_batch(
            [make_state(9, 4, seed=1), make_state(7, 0, seed=2), make_state(6, 3, seed=3)]
        )
        with torch.no_grad():
            out = model(batch, torch.randn(6, 32))
        assert out["synthon_logits"].shape[0] == 3
        assert out["state_value"].shape[0] == 3

    def test_single_graph_state_through_policy(self, model):
        state = make_state(9, 4, seed=5)
        with torch.no_grad():
            out = model(state, torch.randn(6, 32))
        assert out["synthon_logits"].shape[0] == 1

    def test_missing_field_raises_through_policy(self, model):
        state = make_state(9, 4, seed=5)
        del state.global_features
        with pytest.raises(TensorContractError, match="global_features"):
            model(state, torch.randn(6, 32))

    def test_env_state_through_policy(self, model):
        """Environment-built states are stamped and conform."""
        state = stamp_schema_version(make_state(9, 3, seed=6))
        with torch.no_grad():
            decision = model.act(state, torch.randn(6, 32))
        assert decision["synthon_idx"].shape == (1,)
