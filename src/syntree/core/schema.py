"""Frozen V2 state-tensor contract and the canonical PyG batching path.

This module is the single authoritative definition of the 3D-SynTree V2
policy state: the canonical tensors describing a (pocket, ligand, handle)
construction state, the rules that make them well-formed, and the one
sanctioned collator that turns single-graph samples into a validated
PyG :class:`~torch_geometric.data.Batch`.

Binding contracts (issue #6 ARCH-02; constitution #45 §7.1; decisions
D-V2-002/D-V2-003 in ``docs/agent/DECISIONS.md``):

* **Strict separation of invariant chemistry from geometry.** Chemical
  descriptors (``handle_features [B, 64]``) and Cartesian coordinates
  (``handle_pos [B, 3]``) are separate tensors. The deprecated "Franken
  tensor" that glued 64 chemical features to 3 raw coordinates (67 dims
  total) broke SE(3) equivariance — ordinary linear layers treated
  coordinate channels like features — and is a hard error everywhere in
  V2.
* **Fail-closed validation.** Every field is required with an exact dtype,
  rank and shape; cross-field consistency (atom counts, graph count,
  batch-index ranges, atomic-number bounds) is enforced; anything
  malformed raises :class:`TensorContractError` (a ``ValueError``) with a
  precise, actionable message. There is no implicit shape guessing and no
  legacy compatibility parser.
* **Versioned schema.** Producers stamp ``schema_version`` on every state
  via :func:`stamp_schema_version`; consumers reject incompatible versions
  explicitly. Unstamped objects pass field validation as a documented
  interim rule until shard producers exist (#19 DATA-04).
* **One collate path.** :func:`collate_state_batch` is the only sanctioned
  way to batch training states. It materialises the canonical
  ``pocket_batch``/``ligand_batch`` vectors itself, supports the empty
  ligand sub-graph of the t=0 state, performs the only two documented
  normalisations (0-dim → ``[1]``, 1-D ``[64]`` → ``[1, 64]``), rejects
  pre-batched inputs, and stamps + re-validates the assembled batch.

This module depends only on ``torch`` and ``torch_geometric``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Sequence

import torch
from torch_geometric.data import Batch, Data

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "HANDLE_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
    "LEGACY_HANDLE_FEATURE_DIM",
    "MAX_ATOMIC_NUMBER",
    "LEGACY_FRANKEN_TENSOR_MESSAGE",
    "CONTRACT_FIELDS",
    "TensorContractError",
    "StateTensorContract",
    "validate_state_tensors",
    "validate_state_batch",
    "collate_state_batch",
    "stamp_schema_version",
    "is_schema_version_compatible",
]

# ---------------------------------------------------------------------------
# Version and dimension constants (single source of truth for V2)
# ---------------------------------------------------------------------------

#: Name of the state schema (persisted alongside data as provenance).
SCHEMA_NAME = "syntree.state"

#: Version of the frozen V2 state contract. Bump only with a documented
#: breaking change and a superseding entry in ``docs/agent/DECISIONS.md``.
SCHEMA_VERSION = 2

#: Invariant chemical descriptor width of an attachment handle.
HANDLE_FEATURE_DIM = 64

#: Global ligand-state features: [MW/500, heavy/50, V_lig/V_poc, has_ligand].
GLOBAL_FEATURE_DIM = 4

#: Width of the deprecated pre-V2 "Franken tensor" (64 chemical + 3 xyz).
LEGACY_HANDLE_FEATURE_DIM = 67

#: Highest atomic number representable in ``pocket_z`` / ``ligand_z``
#: (must match the element embedding table of the backbone, #21).
MAX_ATOMIC_NUMBER = 100

#: Exact message raised for any 67-dim ``handle_features`` tensor. The
#: prefix is an acceptance criterion of issue #6 and is pinned by test.
LEGACY_FRANKEN_TENSOR_MESSAGE = (
    "Legacy 67D Franken-tensors are deprecated: `handle_features` must be "
    "[B, 64] invariant chemical descriptors with the reacting atom's "
    "Cartesian position supplied separately as `handle_pos` [B, 3] "
    "(issue #6 ARCH-02; #45 §7.1 forbids mixing features and coordinates)."
)

_FLOAT_DTYPES = (torch.float32,)
_LONG_DTYPES = (torch.int64, torch.long)


class TensorContractError(ValueError):
    """Raised when a state object violates the frozen V2 tensor contract.

    Subclasses :class:`ValueError` so generic ``except ValueError`` guards
    keep working.
    """


# ---------------------------------------------------------------------------
# The frozen contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateTensorContract:
    """The canonical tensors of one 3D-SynTree V2 policy state.

    Field semantics (issue #6):

    Pocket sub-graph (``N_p`` protein atoms):
        ``pocket_pos``      ``[N_p, 3]`` float32 — Cartesian coordinates,
                            centred on the co-crystallised ligand centroid.
        ``pocket_z``        ``[N_p]`` int64 — atomic numbers in ``[1, 100]``.
        ``pocket_charge``   ``[N_p]`` float32 — formal charges at pH 7.4.
        ``pocket_batch``    ``[N_p]`` int64 — graph assignment indices in
                            ``[0, B)``.

    Intermediate ligand sub-graph (``N_l`` atoms; **empty at t=0**, and the
    empty tensors must still be present explicitly):
        ``ligand_pos``      ``[N_l, 3]`` float32 — same frame as the pocket.
        ``ligand_z``        ``[N_l]`` int64.
        ``ligand_charge``   ``[N_l]`` float32.
        ``ligand_batch``    ``[N_l]`` int64 — indices in ``[0, B)``.

    Attachment handles & global context (``B`` graphs):
        ``handle_features`` ``[B, 64]`` float32 — invariant chemical
                            descriptors (never coordinates).
        ``handle_pos``      ``[B, 3]`` float32 — reacting atom position in
                            the pocket frame.
        ``global_features`` ``[B, 4]`` float32 — [MW/500, heavy/50,
                            V_lig/V_poc, has_ligand].
        ``stop_mask``       ``[B]`` float32 — additive STOP logit mask
                            (``0.0`` = allowed, large-negative = masked).

    Construction validates every rule (via ``__post_init__``); a malformed
    state raises :class:`TensorContractError`. Instances are immutable —
    use :func:`dataclasses.replace` to derive an updated copy (which
    re-validates). Equality compares tensor values (``torch.equal``) and
    the schema version, not identity.
    """

    # Pocket sub-graph.
    pocket_pos: torch.Tensor  # [N_p, 3] float32
    pocket_z: torch.Tensor  # [N_p] int64 in [1, 100]
    pocket_charge: torch.Tensor  # [N_p] float32
    pocket_batch: torch.Tensor  # [N_p] int64 in [0, B)

    # Intermediate ligand sub-graph (explicitly empty at t=0).
    ligand_pos: torch.Tensor  # [N_l, 3] float32
    ligand_z: torch.Tensor  # [N_l] int64 in [1, 100]
    ligand_charge: torch.Tensor  # [N_l] float32
    ligand_batch: torch.Tensor  # [N_l] int64 in [0, B)

    # Attachment handles & global context.
    handle_features: torch.Tensor  # [B, 64] float32
    handle_pos: torch.Tensor  # [B, 3] float32
    global_features: torch.Tensor  # [B, 4] float32
    stop_mask: torch.Tensor  # [B] float32

    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise TensorContractError(
                f"[state] incompatible schema_version "
                f"{self.schema_version!r}; this code implements "
                f"{SCHEMA_NAME} v{SCHEMA_VERSION} (issue #6 ARCH-02). "
                "Re-extract the data or migrate the producer."
            )
        validate_state_tensors(
            pocket_pos=self.pocket_pos,
            pocket_z=self.pocket_z,
            pocket_charge=self.pocket_charge,
            pocket_batch=self.pocket_batch,
            ligand_pos=self.ligand_pos,
            ligand_z=self.ligand_z,
            ligand_charge=self.ligand_charge,
            ligand_batch=self.ligand_batch,
            handle_features=self.handle_features,
            handle_pos=self.handle_pos,
            global_features=self.global_features,
            stop_mask=self.stop_mask,
        )

    # -- convenience read-outs --------------------------------------------

    @property
    def batch_size(self) -> int:
        """Number of graphs ``B`` in the state."""
        return int(self.handle_features.size(0))

    @property
    def num_pocket(self) -> int:
        """Number of pocket atoms ``N_p``."""
        return int(self.pocket_pos.size(0))

    @property
    def num_ligand(self) -> int:
        """Number of ligand atoms ``N_l`` (0 for the t=0 state)."""
        return int(self.ligand_pos.size(0))

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """The 12 canonical tensors as a plain dict (e.g. for ``Data``)."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclass_fields(self)
            if f.name != "schema_version"
        }

    # -- equality on tensor values (identity-free) --------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateTensorContract):
            return NotImplemented
        if self.schema_version != other.schema_version:
            return False
        for f in dataclass_fields(self):
            if f.name == "schema_version":
                continue
            a, b = getattr(self, f.name), getattr(other, f.name)
            if a.dtype != b.dtype or a.shape != b.shape:
                return False
            if a.numel() and not torch.equal(a, b):
                return False
        return True

    #: Tensor fields are not meaningfully hashable; forbid accidental use.
    __hash__ = None  # type: ignore[assignment]


#: The 12 canonical tensor fields, in declaration order.
CONTRACT_FIELDS: tuple = tuple(
    f.name for f in dataclass_fields(StateTensorContract)
    if f.name != "schema_version"
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_tensor(name: str, value: Any, *, where: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TensorContractError(
            f"[{where}] `{name}` must be a torch.Tensor, got "
            f"{type(value).__name__}"
        )
    return value


def _check_dtype(
    name: str, t: torch.Tensor, allowed: tuple, *, where: str
) -> None:
    if t.dtype not in allowed:
        expected = " or ".join(str(a) for a in allowed)
        raise TensorContractError(
            f"[{where}] `{name}` must have dtype {expected}, got {t.dtype}"
        )


def _check_shape(
    name: str,
    t: torch.Tensor,
    pattern: Sequence[int | None],
    *,
    where: str,
) -> None:
    """Check ``t.dim() == len(pattern)`` and fixed dims match the pattern.

    ``None`` entries are free (any size, including 0 where allowed).
    """
    if t.dim() != len(pattern):
        raise TensorContractError(
            f"[{where}] `{name}` must be {len(pattern)}-D with shape "
            f"[{', '.join('*' if p is None else str(p) for p in pattern)}], "
            f"got {tuple(t.shape)} (no implicit reshaping is performed)"
        )
    for axis, want in enumerate(pattern):
        if want is not None and t.size(axis) != want:
            raise TensorContractError(
                f"[{where}] `{name}` must have shape "
                f"[{', '.join('*' if p is None else str(p) for p in pattern)}], "
                f"got {tuple(t.shape)}"
            )


def _check_finite(name: str, t: torch.Tensor, *, where: str) -> None:
    if t.numel() and not bool(torch.isfinite(t).all()):
        raise TensorContractError(
            f"[{where}] `{name}` contains non-finite values (NaN/Inf); "
            "poisoned inputs fail closed — locate the producer"
        )


def _check_batch_indices(
    name: str, t: torch.Tensor, batch_size: int, *, where: str
) -> None:
    """Range check for a graph-assignment vector (may be empty)."""
    if t.numel() == 0:
        return
    lo, hi = int(t.min()), int(t.max())
    if hi >= batch_size or lo < 0:
        raise TensorContractError(
            f"[{where}] `{name}` indices must lie in [0, {batch_size}), "
            f"got range [{lo}, {hi}]"
        )


def _check_atomic_numbers(name: str, t: torch.Tensor, *, where: str) -> None:
    if t.numel() == 0:
        return
    lo, hi = int(t.min()), int(t.max())
    if lo < 1 or hi > MAX_ATOMIC_NUMBER:
        raise TensorContractError(
            f"[{where}] `{name}` atomic numbers must lie in "
            f"[1, {MAX_ATOMIC_NUMBER}], got range [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# Field validation (fail-closed heart of the contract)
# ---------------------------------------------------------------------------


def validate_state_tensors(
    *,
    pocket_pos: torch.Tensor,
    pocket_z: torch.Tensor,
    pocket_charge: torch.Tensor,
    pocket_batch: torch.Tensor,
    ligand_pos: torch.Tensor,
    ligand_z: torch.Tensor,
    ligand_charge: torch.Tensor,
    ligand_batch: torch.Tensor,
    handle_features: torch.Tensor,
    handle_pos: torch.Tensor,
    global_features: torch.Tensor,
    stop_mask: torch.Tensor,
    where: str = "state",
) -> None:
    """Validate the 12 canonical state tensors; raise on any violation.

    Enforces exact dtypes, exact ranks/shapes, cross-field length
    consistency, graph-index ranges, atomic-number bounds, finiteness, and
    the hard rejection of legacy 67-dim Franken tensors (checked first —
    see :func:`validate_state_batch` for why the precedence matters).
    """
    # -- handle block first: it defines B and carries the legacy check -----
    handle_features = _require_tensor(
        "handle_features", handle_features, where=where
    )
    if handle_features.dim() != 2:
        # 1-D legacy vectors are rejected by the caller's precedence check
        # when they are 67-wide; any other 1-D tensor is a rank error here.
        raise TensorContractError(
            f"[{where}] `handle_features` must be 2-D "
            f"[B, {HANDLE_FEATURE_DIM}], got {tuple(handle_features.shape)} "
            "(1-D vectors must be normalised to [1, 64] by the producer or "
            "by collate_state_batch; no implicit reshaping is performed here)"
        )
    _check_dtype("handle_features", handle_features, _FLOAT_DTYPES, where=where)
    if handle_features.size(-1) == LEGACY_HANDLE_FEATURE_DIM:
        raise TensorContractError(LEGACY_FRANKEN_TENSOR_MESSAGE)
    _check_shape(
        "handle_features",
        handle_features,
        [None, HANDLE_FEATURE_DIM],
        where=where,
    )
    batch_size = handle_features.size(0)
    if batch_size < 1:
        raise TensorContractError(
            f"[{where}] `handle_features` must contain at least one row "
            "(B >= 1)"
        )
    _check_finite("handle_features", handle_features, where=where)

    # -- pocket block -------------------------------------------------------
    pocket_pos = _require_tensor("pocket_pos", pocket_pos, where=where)
    _check_dtype("pocket_pos", pocket_pos, _FLOAT_DTYPES, where=where)
    _check_shape("pocket_pos", pocket_pos, [None, 3], where=where)
    num_pocket = pocket_pos.size(0)
    if num_pocket < 1:
        raise TensorContractError(
            f"[{where}] `pocket_pos` must contain at least one atom — "
            "a pocket-less state is meaningless"
        )
    _check_finite("pocket_pos", pocket_pos, where=where)

    pocket_z = _require_tensor("pocket_z", pocket_z, where=where)
    _check_dtype("pocket_z", pocket_z, _LONG_DTYPES, where=where)
    _check_shape("pocket_z", pocket_z, [num_pocket], where=where)
    _check_atomic_numbers("pocket_z", pocket_z, where=where)

    pocket_charge = _require_tensor("pocket_charge", pocket_charge, where=where)
    _check_dtype("pocket_charge", pocket_charge, _FLOAT_DTYPES, where=where)
    _check_shape("pocket_charge", pocket_charge, [num_pocket], where=where)
    _check_finite("pocket_charge", pocket_charge, where=where)

    pocket_batch = _require_tensor("pocket_batch", pocket_batch, where=where)
    _check_dtype("pocket_batch", pocket_batch, _LONG_DTYPES, where=where)
    _check_shape("pocket_batch", pocket_batch, [num_pocket], where=where)
    _check_batch_indices("pocket_batch", pocket_batch, batch_size, where=where)

    # -- ligand block (may be empty: the t=0 state) -------------------------
    ligand_pos = _require_tensor("ligand_pos", ligand_pos, where=where)
    _check_dtype("ligand_pos", ligand_pos, _FLOAT_DTYPES, where=where)
    _check_shape("ligand_pos", ligand_pos, [None, 3], where=where)
    num_ligand = ligand_pos.size(0)
    _check_finite("ligand_pos", ligand_pos, where=where)

    ligand_z = _require_tensor("ligand_z", ligand_z, where=where)
    _check_dtype("ligand_z", ligand_z, _LONG_DTYPES, where=where)
    _check_shape("ligand_z", ligand_z, [num_ligand], where=where)
    _check_atomic_numbers("ligand_z", ligand_z, where=where)

    ligand_charge = _require_tensor("ligand_charge", ligand_charge, where=where)
    _check_dtype("ligand_charge", ligand_charge, _FLOAT_DTYPES, where=where)
    _check_shape("ligand_charge", ligand_charge, [num_ligand], where=where)
    _check_finite("ligand_charge", ligand_charge, where=where)

    ligand_batch = _require_tensor("ligand_batch", ligand_batch, where=where)
    _check_dtype("ligand_batch", ligand_batch, _LONG_DTYPES, where=where)
    _check_shape("ligand_batch", ligand_batch, [num_ligand], where=where)
    _check_batch_indices("ligand_batch", ligand_batch, batch_size, where=where)

    # -- remaining per-graph tensors ----------------------------------------
    handle_pos = _require_tensor("handle_pos", handle_pos, where=where)
    _check_dtype("handle_pos", handle_pos, _FLOAT_DTYPES, where=where)
    _check_shape("handle_pos", handle_pos, [batch_size, 3], where=where)
    _check_finite("handle_pos", handle_pos, where=where)

    global_features = _require_tensor(
        "global_features", global_features, where=where
    )
    _check_dtype("global_features", global_features, _FLOAT_DTYPES, where=where)
    _check_shape(
        "global_features",
        global_features,
        [batch_size, GLOBAL_FEATURE_DIM],
        where=where,
    )
    _check_finite("global_features", global_features, where=where)

    stop_mask = _require_tensor("stop_mask", stop_mask, where=where)
    _check_dtype("stop_mask", stop_mask, _FLOAT_DTYPES, where=where)
    _check_shape("stop_mask", stop_mask, [batch_size], where=where)
    _check_finite("stop_mask", stop_mask, where=where)


# ---------------------------------------------------------------------------
# Schema-version stamping & compatibility
# ---------------------------------------------------------------------------


def is_schema_version_compatible(value: Any) -> bool:
    """True when a producer-stamped ``schema_version`` matches V2.

    ``None`` (unstamped) is accepted as an interim rule until shard
    producers exist (#19); int, tensor and list/tuple stamps are checked
    for exact equality with :data:`SCHEMA_VERSION`.
    """
    if value is None:
        return True
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        return False
    if isinstance(value, int):
        return value == SCHEMA_VERSION
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool((value == SCHEMA_VERSION).all().item())
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(
            is_schema_version_compatible(v) for v in value
        )
    return False


def stamp_schema_version(data: Data) -> Data:
    """Stamp ``schema_version`` (as a 1-element int64 tensor) on a state.

    Producers call this so consumers can reject incompatible schemas
    explicitly. The tensor form survives PyG collation (concatenating to
    ``[B]``) without special cases. Returns the same object for chaining.
    """
    data.schema_version = torch.tensor([SCHEMA_VERSION], dtype=torch.long)
    return data


# ---------------------------------------------------------------------------
# Validation of PyG Data / Batch objects (attribute containers)
# ---------------------------------------------------------------------------


def validate_state_batch(data: Any, *, where: str = "state") -> StateTensorContract:
    """Validate a PyG ``Data``/``Batch`` (or any attribute container).

    Enforces the full canonical-field contract plus the schema-version
    stamp (when present) and returns the frozen
    :class:`StateTensorContract` view of the tensors.

    Notes:
        * A batched object (``B > 1``) must carry explicit
          ``pocket_batch``/``ligand_batch`` vectors — there is no implicit
          fallback to PyG's ``follow_batch`` names (``pocket_pos_batch``)
          and no silent zeros default.
        * ``handle_nodes`` (index of the reacting handle within each
          graph's ligand nodes, ``-1`` when absent) is optional policy
          context, not part of the canonical contract.
    """
    if data is None:
        raise TensorContractError(
            f"[{where}] received None; a V2 state must be a Data/Batch "
            "object carrying all 12 canonical fields"
        )

    # The legacy Franken-tensor rejection takes precedence over every other
    # diagnostic: legacy states typically ALSO miss V2 fields, and the
    # actionable message is the deprecation itself (issue #6 acceptance).
    hf = getattr(data, "handle_features", None)
    if isinstance(hf, torch.Tensor) and (
        (hf.dim() == 2 and hf.size(-1) == LEGACY_HANDLE_FEATURE_DIM)
        or (hf.dim() == 1 and hf.size(0) == LEGACY_HANDLE_FEATURE_DIM)
    ):
        raise TensorContractError(LEGACY_FRANKEN_TENSOR_MESSAGE)

    missing = [
        name for name in CONTRACT_FIELDS if getattr(data, name, None) is None
    ]
    if missing:
        raise TensorContractError(
            f"[{where}] {type(data).__name__} object is missing required "
            f"state field(s): {', '.join(missing)}. The V2 contract "
            f"requires all {len(CONTRACT_FIELDS)} canonical tensors (see "
            "syntree/core/schema.py); legacy partial states are rejected."
        )

    stamped = getattr(data, "schema_version", None)
    if stamped is not None and not is_schema_version_compatible(stamped):
        raise TensorContractError(
            f"[{where}] incompatible state schema_version {stamped!r}; "
            f"this code implements {SCHEMA_NAME} v{SCHEMA_VERSION} "
            "(issue #6 ARCH-02). Re-extract the data or migrate the producer."
        )

    return StateTensorContract(
        pocket_pos=data.pocket_pos,
        pocket_z=data.pocket_z,
        pocket_charge=data.pocket_charge,
        pocket_batch=data.pocket_batch,
        ligand_pos=data.ligand_pos,
        ligand_z=data.ligand_z,
        ligand_charge=data.ligand_charge,
        ligand_batch=data.ligand_batch,
        handle_features=data.handle_features,
        handle_pos=data.handle_pos,
        global_features=data.global_features,
        stop_mask=data.stop_mask,
    )


# ---------------------------------------------------------------------------
# The canonical collator
# ---------------------------------------------------------------------------


def _prepare_single_graph_sample(item: Any, *, index: int) -> Data:
    """Normalise one single-graph sample before PyG batching.

    Applies only well-defined, documented normalisations:

    * 0-dim tensors become ``[1]`` (PyG cannot collate scalars).
    * a 1-D ``handle_features`` of exactly 64 becomes ``[1, 64]``
      (unambiguous for a single-graph sample); a 1-D 67-vector raises the
      legacy Franken-tensor error; any other width is a hard error.
    * per-sample ``schema_version`` stamps are compatibility-checked here
      and removed (the collator stamps the assembled batch itself, so
      mixed stamped/unstamped samples cannot desynchronise PyG's key
      union).
    * sample-carried ``pocket_batch``/``ligand_batch`` vectors must be the
      implied single-graph zeros and are stripped — the collator
      materialises the canonical vectors itself (PyG would otherwise
      double-increment them, and empty ``ligand_batch`` tensors crash
      PyG's ``__inc__``).
    * a multi-row ``handle_features`` (``B_k > 1``) is rejected: the
      collator batches single-graph states only.
    * ``num_nodes`` is derived from ``pocket_pos`` when absent.

    The item is cloned first; the caller's object is never mutated.
    """
    if item is None:
        raise TensorContractError(
            f"[collate] sample {index} is None; each item must be a "
            "single-graph Data object"
        )
    if isinstance(item, Batch) or bool(getattr(item, "is_batched", False)):
        raise TensorContractError(
            f"[collate] sample {index} is already a Batch; collate "
            "single-graph Data objects only (no re-batching)"
        )
    if not isinstance(item, Data):
        raise TensorContractError(
            f"[collate] sample {index} must be a torch_geometric Data "
            f"object, got {type(item).__name__}"
        )

    data = item.clone()
    for key in list(data.keys()):
        value = data[key]
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            data[key] = value.unsqueeze(0)

    handle_features = getattr(data, "handle_features", None)
    if handle_features is None:
        raise TensorContractError(
            f"[collate] sample {index} is missing `handle_features` "
            "(and every other canonical field — a V2 state must carry all "
            "12; see syntree/core/schema.py)"
        )
    if handle_features.dim() == 1:
        if handle_features.size(0) == LEGACY_HANDLE_FEATURE_DIM:
            raise TensorContractError(LEGACY_FRANKEN_TENSOR_MESSAGE)
        if handle_features.size(0) != HANDLE_FEATURE_DIM:
            raise TensorContractError(
                f"[collate] sample {index}: 1-D `handle_features` must "
                f"have exactly {HANDLE_FEATURE_DIM} elements, got "
                f"{handle_features.size(0)}"
            )
        data.handle_features = handle_features.unsqueeze(0)
    elif handle_features.dim() == 2 and handle_features.size(-1) == LEGACY_HANDLE_FEATURE_DIM:
        raise TensorContractError(LEGACY_FRANKEN_TENSOR_MESSAGE)
    elif handle_features.dim() == 2 and handle_features.size(0) != 1:
        raise TensorContractError(
            f"[collate] sample {index}: `handle_features` has "
            f"{handle_features.size(0)} rows — pre-batched/multi-graph "
            "samples are rejected; collate single-graph states only"
        )

    # Per-sample version stamps: verify compatibility, then strip (the
    # assembled batch is stamped uniformly below).
    stamped = getattr(data, "schema_version", None)
    if stamped is not None and not is_schema_version_compatible(stamped):
        raise TensorContractError(
            f"[collate] sample {index}: incompatible schema_version "
            f"{stamped!r}; this code implements {SCHEMA_NAME} "
            f"v{SCHEMA_VERSION} (issue #6 ARCH-02)."
        )
    if stamped is not None:
        del data.schema_version

    # Batch vectors: single-graph samples may carry the implied zeros.
    for key in ("pocket_batch", "ligand_batch"):
        vec = getattr(data, key, None)
        if vec is None:
            continue
        if not isinstance(vec, torch.Tensor) or vec.dtype not in _LONG_DTYPES or vec.dim() != 1:
            raise TensorContractError(
                f"[collate] sample {index}: `{key}` on a single-graph "
                f"sample must be a 1-D int64 tensor, got "
                f"{tuple(vec.shape) if isinstance(vec, torch.Tensor) else type(vec).__name__}"
            )
        if vec.numel() and int(vec.abs().max()) != 0:
            raise TensorContractError(
                f"[collate] sample {index}: `{key}` on a single-graph "
                "sample must be all zeros (pre-batched objects are "
                "rejected; the collator materialises batch vectors itself)"
            )
        del data[key]

    pocket_pos = getattr(data, "pocket_pos", None)
    if pocket_pos is not None and "num_nodes" not in data:
        # Pocket nodes define the graph's node count (handles are carried
        # separately as per-graph rows). Setting it explicitly avoids PyG's
        # heuristic (and warning) on our non-standard key set.
        data.num_nodes = int(pocket_pos.size(0))
    return data


def collate_state_batch(items: Sequence[Any]) -> Batch:
    """Collate single-graph states into a validated canonical ``Batch``.

    This is the one sanctioned batching path for V2 training states. It:

    1. normalises every sample (see
       :func:`_prepare_single_graph_sample`) — legacy 67-dim tensors fail
       here with the exact deprecation message;
    2. batches with ``follow_batch=['pocket_pos', 'ligand_pos']`` so the
       graph-assignment vectors exist even when the ligand sub-graph is
       empty (the t=0 state);
    3. renames PyG's ``pocket_pos_batch``/``ligand_pos_batch`` to the
       canonical ``pocket_batch``/``ligand_batch`` and drops the aliases
       (exactly one representation exists — no implicit name fallback
       anywhere downstream);
    4. stamps ``schema_version`` and validates the assembled batch against
       the full contract before returning it.
    """
    if not isinstance(items, (list, tuple)) or len(items) == 0:
        raise TensorContractError(
            "[collate] expected a non-empty list of single-graph states, "
            f"got {type(items).__name__}"
        )

    prepared = [
        _prepare_single_graph_sample(item, index=i)
        for i, item in enumerate(items)
    ]

    batch = Batch.from_data_list(
        prepared, follow_batch=["pocket_pos", "ligand_pos"]
    )

    # Materialise the canonical batch-vector names; drop PyG's follow_batch
    # aliases so exactly one representation exists.
    batch.pocket_batch = batch.pocket_pos_batch
    del batch.pocket_pos_batch
    if hasattr(batch, "ligand_pos_batch"):
        batch.ligand_batch = batch.ligand_pos_batch
        del batch.ligand_pos_batch
    else:
        # Every ligand sub-graph was empty: materialise the empty vector
        # explicitly so the field is always present.
        batch.ligand_batch = torch.zeros(
            batch.ligand_pos.size(0), dtype=torch.long
        )

    # Stamp + fail-closed validation of the assembled batch.
    batch.schema_version = torch.full(
        (batch.num_graphs,), SCHEMA_VERSION, dtype=torch.long
    )
    validate_state_batch(batch, where="collated batch")
    return batch
