"""Core contracts: the frozen, versioned state tensor schema and the
canonical PyG batching path (#6 ARCH-02)."""

from __future__ import annotations

from syntree.core.schema import (
    CONTRACT_FIELDS,
    GLOBAL_FEATURE_DIM,
    HANDLE_FEATURE_DIM,
    LEGACY_FRANKEN_TENSOR_MESSAGE,
    LEGACY_HANDLE_FEATURE_DIM,
    MAX_ATOMIC_NUMBER,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    StateTensorContract,
    TensorContractError,
    collate_state_batch,
    is_schema_version_compatible,
    stamp_schema_version,
    validate_state_batch,
    validate_state_tensors,
)

__all__ = [
    "CONTRACT_FIELDS",
    "GLOBAL_FEATURE_DIM",
    "HANDLE_FEATURE_DIM",
    "LEGACY_FRANKEN_TENSOR_MESSAGE",
    "LEGACY_HANDLE_FEATURE_DIM",
    "MAX_ATOMIC_NUMBER",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "StateTensorContract",
    "TensorContractError",
    "collate_state_batch",
    "is_schema_version_compatible",
    "stamp_schema_version",
    "validate_state_batch",
    "validate_state_tensors",
]
