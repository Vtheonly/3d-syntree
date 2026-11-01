"""Model components: equivariant math kernels (#7+), the union-graph
backbone (#21), the handle-selection head (#22), the cosine cutoff envelope
(#9), the cross-attention decoder (#23) and the remaining policy heads
(later issues)."""

from __future__ import annotations

from syntree.models.bipartite import (
    BACKBONE_CUTOFF,
    BACKBONE_HIDDEN_DIM,
    BACKBONE_NUM_LAYERS,
    BACKBONE_NUM_RADIAL,
    BackboneOutput,
    BipartitePaiNN,
    GaussianRadialBasis,
    build_union_radius_graph,
)
from syntree.models.cutoff import continuous_cosine_envelope
from syntree.models.equivariant import (
    SAFE_NORM_EPS,
    PaiNNInteraction,
    PaiNNMixing,
    safe_norm,
    safe_unit_vector,
)
from syntree.models.handle_head import (
    HANDLE_GEOM_CUTOFF,
    HANDLE_HEAD_HIDDEN_DIM,
    HANDLE_HEAD_NUM_RADIAL,
    HandleSelectionHead,
    HandleSelectionOutput,
)
from syntree.models.policy import (
    DECODER_D_MODEL,
    DECODER_DROPOUT,
    DECODER_FFN_DIM,
    DECODER_NUM_HEADS,
    DECODER_NUM_LAYERS,
    CrossAttentionDecoder,
    CrossAttentionDecoderLayer,
    DecoderOutput,
)

__all__ = [
    "BACKBONE_CUTOFF",
    "BACKBONE_HIDDEN_DIM",
    "BACKBONE_NUM_LAYERS",
    "BACKBONE_NUM_RADIAL",
    "BackboneOutput",
    "BipartitePaiNN",
    "GaussianRadialBasis",
    "build_union_radius_graph",
    "continuous_cosine_envelope",
    "SAFE_NORM_EPS",
    "PaiNNInteraction",
    "PaiNNMixing",
    "safe_norm",
    "safe_unit_vector",
    "HANDLE_GEOM_CUTOFF",
    "HANDLE_HEAD_HIDDEN_DIM",
    "HANDLE_HEAD_NUM_RADIAL",
    "HandleSelectionHead",
    "HandleSelectionOutput",
    "DECODER_D_MODEL",
    "DECODER_DROPOUT",
    "DECODER_FFN_DIM",
    "DECODER_NUM_HEADS",
    "DECODER_NUM_LAYERS",
    "CrossAttentionDecoder",
    "CrossAttentionDecoderLayer",
    "DecoderOutput",
]
