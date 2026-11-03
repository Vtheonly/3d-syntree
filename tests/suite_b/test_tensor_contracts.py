"""Suite B / 3 — tensor-contract separation invariants (issue #33 TEST-02).

The issue-pinned battery: **separate 64-D chemical handle tensors from
3-D spatial coordinate tensors, with legacy 67-D concatenated shapes
raising exceptions** (#6 ARCH-02; constitution #45 §7.1 — "Do not
concatenate chemical descriptors and raw [x, y, z] coordinates into a
single feature vector").

The pre-V2 system glued 64 chemical features to 3 raw coordinates into
one 67-D "Franken tensor" and pushed it through ordinary linear layers
as if all channels transformed identically.  V2 keeps the
representations strictly separate — chemical invariants in
``handle_features [B, 64]``, Cartesian positions in ``handle_pos [B, 3]``
(and ``pocket_pos`` / ``ligand_pos``) — and the schema rejects any
legacy-shaped tensor at construction, validation, and collation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from syntree.core.schema import (  # noqa: E402
    GLOBAL_FEATURE_DIM,
    HANDLE_FEATURE_DIM,
    LEGACY_FRANKEN_TENSOR_MESSAGE,
    LEGACY_HANDLE_FEATURE_DIM,
    TensorContractError,
    collate_state_batch,
)

_GEN = torch.Generator().manual_seed(0)


def make_single_graph_state(*, handle_width: int = HANDLE_FEATURE_DIM) -> Data:
    """A canonical 12-field single-graph state (the #6 frozen schema)."""
    gen = _GEN
    return Data(
        pocket_pos=torch.randn(8, 3, generator=gen),
        pocket_z=torch.randint(1, 9, (8,), generator=gen),
        pocket_charge=torch.randn(8, generator=gen),
        ligand_pos=torch.randn(4, 3, generator=gen),
        ligand_z=torch.randint(1, 9, (4,), generator=gen),
        ligand_charge=torch.randn(4, generator=gen),
        handle_features=torch.randn(1, handle_width, generator=gen),
        handle_pos=torch.randn(1, 3, generator=gen),
        global_features=torch.randn(1, GLOBAL_FEATURE_DIM, generator=gen),
        stop_mask=torch.zeros(1),
    )


class TestTensorContractSeparation:
    def test_the_frozen_widths_are_separated(self):
        assert HANDLE_FEATURE_DIM == 64  # chemical descriptors only
        assert LEGACY_HANDLE_FEATURE_DIM == 67  # 64 + 3 xyz (deprecated)
        assert HANDLE_FEATURE_DIM != 3

    def test_valid_states_keep_features_and_coordinates_separate(self):
        batch = collate_state_batch(
            [make_single_graph_state(), make_single_graph_state()]
        )
        # chemical descriptors stay 64-wide, coordinates stay 3-wide
        assert batch.handle_features.shape == (2, HANDLE_FEATURE_DIM)
        assert batch.handle_pos.shape == (2, 3)
        assert batch.pocket_pos.shape[1] == 3
        assert batch.ligand_pos.shape[1] == 3
        # the two blocks never share a tensor or a width
        assert batch.handle_features.shape[1] != batch.handle_pos.shape[1]
        assert batch.handle_features.dtype.is_floating_point
        assert batch.handle_pos.dtype.is_floating_point

    def test_legacy_67d_franken_tensor_is_rejected_at_collation(self):
        legacy = make_single_graph_state(handle_width=LEGACY_HANDLE_FEATURE_DIM)
        with pytest.raises(TensorContractError) as error:
            collate_state_batch([legacy])
        message = str(error.value)
        assert message.startswith(LEGACY_FRANKEN_TENSOR_MESSAGE[:60])
        assert "67D" in message and "[B, 64]" in message

    def test_legacy_tensor_is_rejected_inside_a_batch_too(self):
        valid = make_single_graph_state()
        legacy = make_single_graph_state(handle_width=LEGACY_HANDLE_FEATURE_DIM)
        with pytest.raises(TensorContractError, match="67D"):
            collate_state_batch([valid, legacy])

    def test_a_concatenated_67d_projection_is_not_a_legal_feature_block(self):
        """The chemical block must never be constructed by gluing xyz on.

        A tensor built as ``[descriptors | coordinates]`` (67 wide) is
        indistinguishable from the legacy Franken tensor and is rejected
        by the same schema guard.
        """
        state = make_single_graph_state()
        glued = torch.cat(
            [state.handle_features, state.handle_pos], dim=-1
        )  # [1, 67]: exactly the pre-V2 construction
        assert glued.shape[1] == LEGACY_HANDLE_FEATURE_DIM
        franken = Data(
            pocket_pos=state.pocket_pos,
            pocket_z=state.pocket_z,
            pocket_charge=state.pocket_charge,
            ligand_pos=state.ligand_pos,
            ligand_z=state.ligand_z,
            ligand_charge=state.ligand_charge,
            handle_features=glued,
            handle_pos=state.handle_pos,
            global_features=state.global_features,
            stop_mask=state.stop_mask,
        )
        with pytest.raises(TensorContractError, match="67D"):
            collate_state_batch([franken])

    def test_coordinate_blocks_are_always_rank2_spatial(self):
        batch = collate_state_batch([make_single_graph_state()])
        for block in (batch.handle_pos, batch.pocket_pos, batch.ligand_pos):
            assert block.ndim == 2
            assert block.shape[1] == 3
            assert bool(torch.isfinite(block).all())
