"""Frozen state tensor contract & canonical PyG batch path — tests (#6 ARCH-02).

Acceptance criteria of issue #6 (clean-room V2 implementation):

1. Instantiating batches conforming to the schema passes type and shape
   checks (``tests/test_tensor_contracts.py``).
2. Passing a 67-D tensor raises an explicit
   ``ValueError("Legacy 67D Franken-tensors are deprecated")``.

Beyond the two pinned criteria, this suite enforces the full fail-closed
contract: dtype/rank/shape rules, cross-field consistency, batch-index
ranges, atomic-number bounds, finiteness, schema-version stamping, and
every documented rule of the canonical collator (decisions D-V2-002).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch_geometric.data import Batch, Data

from syntree.core.schema import (
    CONTRACT_FIELDS,
    GLOBAL_FEATURE_DIM,
    HANDLE_FEATURE_DIM,
    LEGACY_FRANKEN_TENSOR_MESSAGE,
    MAX_ATOMIC_NUMBER,
    SCHEMA_VERSION,
    StateTensorContract,
    TensorContractError,
    collate_state_batch,
    is_schema_version_compatible,
    stamp_schema_version,
    validate_state_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fields(
    num_pocket: int = 6,
    num_ligand: int = 4,
    batch_size: int = 1,
    *,
    seed: int = 0,
) -> dict:
    """Random but strictly contract-conforming field values."""
    g = torch.Generator().manual_seed(seed)
    n_p, n_l, b = num_pocket, num_ligand, batch_size
    return {
        "pocket_pos": torch.randn(n_p, 3, generator=g),
        "pocket_z": torch.randint(1, MAX_ATOMIC_NUMBER + 1, (n_p,), generator=g),
        "pocket_charge": torch.randn(n_p, generator=g),
        "pocket_batch": torch.randint(0, b, (n_p,), generator=g),
        "ligand_pos": torch.randn(n_l, 3, generator=g),
        "ligand_z": torch.randint(1, MAX_ATOMIC_NUMBER + 1, (n_l,), generator=g),
        "ligand_charge": torch.randn(n_l, generator=g),
        "ligand_batch": torch.randint(0, b, (n_l,), generator=g),
        "handle_features": torch.randn(b, HANDLE_FEATURE_DIM, generator=g),
        "handle_pos": torch.randn(b, 3, generator=g),
        "global_features": torch.randn(b, GLOBAL_FEATURE_DIM, generator=g),
        "stop_mask": torch.zeros(b),
    }


def _single_graph_data(
    num_pocket: int = 6, num_ligand: int = 4, *, seed: int = 0, stamp: bool = True
) -> Data:
    """A single-graph sample as produced by a V2 state producer."""
    f = _fields(num_pocket, num_ligand, batch_size=1, seed=seed)
    data = Data(**f)
    if stamp:
        stamp_schema_version(data)
    return data


def _batched(batch_size: int = 3, *, seed: int = 0) -> Batch:
    """A canonical batch of differently-sized single-graph states."""
    sizes = [(6, 4), (9, 0), (5, 7)][:batch_size]
    items = [
        _single_graph_data(n_p, n_l, seed=seed + i) for i, (n_p, n_l) in enumerate(sizes)
    ]
    return collate_state_batch(items)


# ---------------------------------------------------------------------------
# 1. Conforming states pass (acceptance criterion 1)
# ---------------------------------------------------------------------------


class TestConformingStates:
    def test_contract_from_fields_passes(self):
        contract = StateTensorContract(**_fields())
        assert contract.batch_size == 1
        assert contract.num_pocket == 6
        assert contract.num_ligand == 4

    def test_single_graph_data_object_passes(self):
        contract = validate_state_batch(_single_graph_data())
        assert isinstance(contract, StateTensorContract)
        assert contract.batch_size == 1

    def test_empty_ligand_t0_state_passes(self):
        contract = StateTensorContract(**_fields(num_ligand=0))
        assert contract.num_ligand == 0
        assert contract.ligand_batch.shape == (0,)

    def test_multi_graph_batch_passes(self):
        contract = validate_state_batch(_batched())
        assert contract.batch_size == 3
        assert contract.num_pocket == 6 + 9 + 5
        assert contract.num_ligand == 4 + 0 + 7

    def test_frozen_instance_rejects_mutation(self):
        contract = StateTensorContract(**_fields())
        with pytest.raises(dataclasses.FrozenInstanceError):
            contract.pocket_pos = torch.zeros(6, 3)

    def test_replace_produces_validated_copy(self):
        contract = StateTensorContract(**_fields())
        updated = dataclasses.replace(contract, pocket_charge=torch.zeros(6))
        assert torch.equal(updated.pocket_charge, torch.zeros(6))
        with pytest.raises(TensorContractError):
            dataclasses.replace(contract, pocket_charge=torch.zeros(7))  # wrong length

    def test_equality_compares_values_not_identity(self):
        a = StateTensorContract(**_fields(seed=1))
        b = StateTensorContract(**_fields(seed=1))
        c = StateTensorContract(**_fields(seed=2))
        assert a == b
        assert a != c
        assert a != "not-a-contract"

    def test_to_dict_has_twelve_canonical_fields(self):
        d = StateTensorContract(**_fields()).to_dict()
        assert len(d) == 12
        assert tuple(sorted(d)) == tuple(sorted(CONTRACT_FIELDS))


# ---------------------------------------------------------------------------
# 2. Legacy 67-D rejection (acceptance criterion 2)
# ---------------------------------------------------------------------------


class TestLegacyFrankenRejection:
    def test_2d_67_via_contract_constructor(self):
        f = _fields()
        f["handle_features"] = torch.randn(2, 67)
        with pytest.raises(ValueError, match=r"^Legacy 67D Franken-tensors"):
            StateTensorContract(**f)

    def test_2d_67_via_validate_state_batch(self):
        data = _single_graph_data()
        data.handle_features = torch.randn(1, 67)
        with pytest.raises(TensorContractError, match=r"^Legacy 67D Franken-tensors"):
            validate_state_batch(data)

    def test_2d_67_via_collate(self):
        data = _single_graph_data()
        data.handle_features = torch.randn(1, 67)
        with pytest.raises(TensorContractError, match=r"^Legacy 67D Franken-tensors"):
            collate_state_batch([data])

    def test_1d_67_via_validate_state_batch(self):
        data = _single_graph_data()
        data.handle_features = torch.randn(67)
        with pytest.raises(TensorContractError, match=r"^Legacy 67D Franken-tensors"):
            validate_state_batch(data)

    def test_1d_67_via_collate(self):
        data = _single_graph_data()
        data.handle_features = torch.randn(67)
        with pytest.raises(TensorContractError, match=r"^Legacy 67D Franken-tensors"):
            collate_state_batch([data])

    def test_exact_message_text(self):
        f = _fields()
        f["handle_features"] = torch.randn(3, 67)
        with pytest.raises(TensorContractError) as excinfo:
            StateTensorContract(**f)
        assert str(excinfo.value).startswith(
            "Legacy 67D Franken-tensors are deprecated"
        )
        assert str(excinfo.value) == LEGACY_FRANKEN_TENSOR_MESSAGE

    def test_rejection_precedes_missing_field_diagnostics(self):
        """A legacy state (67-D AND missing V2 fields) gets the actionable
        deprecation message, not the missing-field error."""
        data = Data(handle_features=torch.randn(2, 67))
        with pytest.raises(TensorContractError, match=r"^Legacy 67D Franken-tensors"):
            validate_state_batch(data)

    def test_rejection_is_plain_valueerror(self):
        f = _fields()
        f["handle_features"] = torch.randn(1, 67)
        with pytest.raises(ValueError):
            StateTensorContract(**f)


# ---------------------------------------------------------------------------
# 3. Fail-closed field validation
# ---------------------------------------------------------------------------


class TestFailClosedValidation:
    """Every malformed input raises TensorContractError with a precise message."""

    @pytest.mark.parametrize(
        "field, bad_value",
        [
            # dtype violations (V2 is strictly float32 / int64)
            ("pocket_pos", lambda f: f["pocket_pos"].double()),
            ("pocket_pos", lambda f: f["pocket_pos"].half()),
            ("pocket_z", lambda f: f["pocket_z"].to(torch.int32)),
            ("pocket_z", lambda f: f["pocket_z"].to(torch.float32)),
            ("pocket_batch", lambda f: f["pocket_batch"].to(torch.int32)),
            ("handle_features", lambda f: f["handle_features"].double()),
            ("stop_mask", lambda f: f["stop_mask"].double()),
            # rank violations
            ("pocket_pos", lambda f: f["pocket_pos"].reshape(-1)),
            ("handle_features", lambda f: f["handle_features"].reshape(-1)),
            ("pocket_z", lambda f: f["pocket_z"].unsqueeze(0)),
            ("global_features", lambda f: f["global_features"].reshape(-1)),
            # shape violations
            ("pocket_pos", lambda f: torch.randn(6, 2)),
            ("handle_features", lambda f: torch.randn(1, HANDLE_FEATURE_DIM - 1)),
            ("handle_features", lambda f: torch.randn(1, HANDLE_FEATURE_DIM + 1)),
            ("handle_pos", lambda f: torch.randn(1, 2)),
            ("global_features", lambda f: torch.randn(1, GLOBAL_FEATURE_DIM + 1)),
            ("stop_mask", lambda f: torch.zeros(2)),
            # cross-field length mismatches
            ("pocket_z", lambda f: torch.randint(1, 9, (5,))),
            ("pocket_charge", lambda f: torch.randn(7)),
            ("ligand_z", lambda f: torch.randint(1, 9, (3,))),
            ("ligand_batch", lambda f: torch.zeros(5, dtype=torch.long)),
            # batch-index range violations
            ("pocket_batch", lambda f: torch.full((6,), 1, dtype=torch.long)),
            ("pocket_batch", lambda f: torch.full((6,), -1, dtype=torch.long)),
            ("ligand_batch", lambda f: torch.full((4,), 2, dtype=torch.long)),
            # atomic-number bounds
            ("pocket_z", lambda f: torch.zeros(6, dtype=torch.long)),
            ("pocket_z", lambda f: torch.full((6,), MAX_ATOMIC_NUMBER + 1, dtype=torch.long)),
            ("ligand_z", lambda f: torch.zeros(4, dtype=torch.long)),
            # non-finite values
            ("pocket_pos", lambda f: _with_nan(f["pocket_pos"])),
            ("pocket_charge", lambda f: _with_inf(f["pocket_charge"])),
            ("handle_features", lambda f: _with_nan(f["handle_features"])),
            ("handle_pos", lambda f: _with_nan(f["handle_pos"])),
            ("global_features", lambda f: _with_inf(f["global_features"])),
            ("stop_mask", lambda f: _with_nan(f["stop_mask"])),
            # degenerate graph counts
            ("handle_features", lambda f: torch.randn(0, HANDLE_FEATURE_DIM)),
            ("pocket_pos", lambda f: torch.randn(0, 3)),
            # non-tensor inputs
            ("pocket_pos", lambda f: [[0.0, 0.0, 0.0]] * 6),
            ("pocket_z", lambda f: [6] * 6),
        ],
    )
    def test_malformed_field_raises(self, field, bad_value):
        f = _fields()
        f[field] = bad_value(f) if callable(bad_value) else bad_value
        with pytest.raises(TensorContractError):
            StateTensorContract(**f)

    def test_error_messages_name_the_field(self):
        f = _fields()
        f["pocket_pos"] = torch.randn(6, 2)
        with pytest.raises(TensorContractError, match=r"pocket_pos"):
            StateTensorContract(**f)

    def test_missing_field_on_data_object(self):
        data = _single_graph_data()
        del data.ligand_pos
        with pytest.raises(TensorContractError, match=r"ligand_pos"):
            validate_state_batch(data)

    def test_none_input_rejected(self):
        with pytest.raises(TensorContractError):
            validate_state_batch(None)


def _with_nan(t: torch.Tensor) -> torch.Tensor:
    t = t.clone()
    t.flatten()[0] = float("nan")
    return t


def _with_inf(t: torch.Tensor) -> torch.Tensor:
    t = t.clone()
    t.flatten()[0] = float("inf")
    return t


# ---------------------------------------------------------------------------
# 4. Schema-version stamping
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_stamped_state_passes(self):
        data = stamp_schema_version(_single_graph_data(stamp=False))
        assert torch.equal(data.schema_version, torch.tensor([SCHEMA_VERSION]))
        validate_state_batch(data)  # no raise

    def test_incompatible_int_stamp_rejected(self):
        data = _single_graph_data()
        data.schema_version = SCHEMA_VERSION + 1
        with pytest.raises(TensorContractError, match=r"schema_version"):
            validate_state_batch(data)

    def test_incompatible_tensor_stamp_rejected(self):
        data = _single_graph_data()
        data.schema_version = torch.tensor([SCHEMA_VERSION, SCHEMA_VERSION + 1])
        with pytest.raises(TensorContractError, match=r"schema_version"):
            validate_state_batch(data)

    def test_incompatible_contract_version_rejected(self):
        f = _fields()
        with pytest.raises(TensorContractError, match=r"schema_version"):
            StateTensorContract(**f, schema_version=99)

    def test_unstamped_state_passes_interim(self):
        data = _single_graph_data(stamp=False)
        # no schema_version attribute at all
        assert getattr(data, "schema_version", None) is None
        validate_state_batch(data)  # no raise (interim rule until #19)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, True),  # unstamped interim
            (SCHEMA_VERSION, True),
            (SCHEMA_VERSION + 1, False),
            (torch.tensor([SCHEMA_VERSION]), True),
            (torch.tensor([SCHEMA_VERSION, SCHEMA_VERSION]), True),
            (torch.tensor([SCHEMA_VERSION, 99]), False),
            (torch.tensor([], dtype=torch.long), False),
            ([SCHEMA_VERSION, SCHEMA_VERSION], True),
            ([SCHEMA_VERSION, 99], False),
            ([], False),
            (True, False),  # bool is not a version
            ("2", False),  # strings are not versions
        ],
    )
    def test_is_schema_version_compatible(self, value, expected):
        assert is_schema_version_compatible(value) is expected


# ---------------------------------------------------------------------------
# 5. The canonical collator
# ---------------------------------------------------------------------------


class TestCollateStateBatch:
    def test_output_is_batch_with_correct_graph_count(self):
        batch = _batched()
        assert isinstance(batch, Batch)
        assert batch.num_graphs == 3

    def test_batch_vectors_materialized(self):
        batch = _batched(batch_size=3)  # pocket sizes 6, 9, 5
        assert torch.equal(
            batch.pocket_batch,
            torch.tensor([0] * 6 + [1] * 9 + [2] * 5),
        )
        # ligand sizes 4, 0, 7 (graph 1 is the t=0 state)
        assert torch.equal(
            batch.ligand_batch,
            torch.tensor([0] * 4 + [1] * 0 + [2] * 7),
        )

    def test_follow_batch_aliases_removed(self):
        batch = _batched()
        assert not hasattr(batch, "pocket_pos_batch")
        assert not hasattr(batch, "ligand_pos_batch")

    def test_fields_concatenated_correctly(self):
        items = [
            _single_graph_data(6, 4, seed=1),
            _single_graph_data(9, 0, seed=2),
        ]
        batch = collate_state_batch(items)
        assert torch.equal(batch.pocket_z, torch.cat([items[0].pocket_z, items[1].pocket_z]))
        assert torch.equal(
            batch.pocket_pos, torch.cat([items[0].pocket_pos, items[1].pocket_pos])
        )
        assert torch.equal(
            batch.ligand_pos, torch.cat([items[0].ligand_pos, items[1].ligand_pos])
        )
        assert torch.equal(
            batch.handle_features,
            torch.cat([items[0].handle_features, items[1].handle_features]),
        )
        assert torch.equal(
            batch.global_features,
            torch.cat([items[0].global_features, items[1].global_features]),
        )
        assert torch.equal(batch.stop_mask, torch.zeros(2))

    def test_empty_ligand_among_graphs_supported(self):
        batch = _batched()  # includes a graph with 0 ligand atoms
        contract = validate_state_batch(batch)
        assert contract.num_ligand == 11

    def test_all_empty_ligands_supported(self):
        items = [
            _single_graph_data(5, 0, seed=1),
            _single_graph_data(6, 0, seed=2),
        ]
        batch = collate_state_batch(items)
        assert batch.ligand_pos.shape == (0, 3)
        assert batch.ligand_batch.shape == (0,)
        validate_state_batch(batch)

    def test_input_samples_not_mutated(self):
        items = [_single_graph_data(6, 4, seed=3), _single_graph_data(5, 2, seed=4)]
        before = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in items[0].to_dict().items()
        }
        had_stamp = hasattr(items[0], "schema_version")
        collate_state_batch(items)
        after = items[0].to_dict()
        assert set(before) == set(after)
        for k, v in before.items():
            assert torch.equal(v, after[k])
        # the sample's own attributes (incl. its stamp) are untouched
        assert had_stamp == hasattr(items[0], "schema_version")

    def test_1d_handle_features_normalized_to_single_row(self):
        data = _single_graph_data(stamp=False)
        data.handle_features = torch.randn(HANDLE_FEATURE_DIM)
        batch = collate_state_batch([data])
        assert batch.handle_features.shape == (1, HANDLE_FEATURE_DIM)

    def test_1d_wrong_width_rejected(self):
        data = _single_graph_data(stamp=False)
        data.handle_features = torch.randn(HANDLE_FEATURE_DIM + 3)
        with pytest.raises(TensorContractError, match=r"handle_features"):
            collate_state_batch([data])

    def test_0dim_tensors_normalized(self):
        data = _single_graph_data(stamp=False)
        data.stop_mask = torch.tensor(0.0)  # 0-dim scalar
        batch = collate_state_batch([data])
        assert batch.stop_mask.shape == (1,)

    def test_prebatched_sample_rejected(self):
        data = _single_graph_data()
        data.pocket_batch = torch.tensor([0, 1, 0, 1, 0, 1])  # non-zero
        with pytest.raises(TensorContractError, match=r"pre-batched"):
            collate_state_batch([data])

    def test_multirow_handle_features_rejected(self):
        data = _single_graph_data()
        data.handle_features = torch.randn(2, HANDLE_FEATURE_DIM)
        data.handle_pos = torch.randn(2, 3)
        data.global_features = torch.randn(2, GLOBAL_FEATURE_DIM)
        data.stop_mask = torch.zeros(2)
        with pytest.raises(TensorContractError, match=r"single-graph"):
            collate_state_batch([data])

    def test_batch_instance_rejected(self):
        batch = _batched()
        with pytest.raises(TensorContractError, match=r"Batch"):
            collate_state_batch([batch])

    def test_non_data_item_rejected(self):
        with pytest.raises(TensorContractError, match=r"Data"):
            collate_state_batch([{"pocket_pos": torch.zeros(1, 3)}])

    def test_none_item_rejected(self):
        with pytest.raises(TensorContractError):
            collate_state_batch([None])

    def test_empty_sequence_rejected(self):
        with pytest.raises(TensorContractError):
            collate_state_batch([])

    def test_output_stamped_and_validated(self):
        batch = _batched(batch_size=3)
        assert torch.equal(
            batch.schema_version, torch.tensor([SCHEMA_VERSION] * 3)
        )
        validate_state_batch(batch)  # full contract re-check passes

    def test_mixed_stamped_and_unstamped_samples(self):
        stamped = _single_graph_data(6, 4, seed=5, stamp=True)
        unstamped = _single_graph_data(5, 2, seed=6, stamp=False)
        batch = collate_state_batch([stamped, unstamped])
        assert torch.equal(
            batch.schema_version, torch.tensor([SCHEMA_VERSION] * 2)
        )

    def test_incompatible_sample_stamp_rejected_at_collate(self):
        data = _single_graph_data()
        data.schema_version = torch.tensor([SCHEMA_VERSION + 5])
        with pytest.raises(TensorContractError, match=r"schema_version"):
            collate_state_batch([data])

    def test_single_state_yields_zero_batch_vectors(self):
        batch = collate_state_batch([_single_graph_data(7, 3, seed=9)])
        assert torch.equal(batch.pocket_batch, torch.zeros(7, dtype=torch.long))
        assert torch.equal(batch.ligand_batch, torch.zeros(3, dtype=torch.long))
