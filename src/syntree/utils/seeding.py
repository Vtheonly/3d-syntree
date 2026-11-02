"""Canonical deterministic seed propagation (Suite A / #32 TEST-01).

The V2 reproducibility contract (constitution #45 §16 "deterministic seed
propagation", §13.6 RNG states in checkpoints) requires ONE authoritative
seeding implementation — every trainer, sampler, and environment draws its
RNG streams through this module, so that ``seed_everything(s)`` followed
by identical work reproduces identical results bit-exactly across
PyTorch, NumPy, and the Python standard library.

Consumers: the #26 trainer (RNG state checkpoints), the #27 curriculum
sampler, Suite A determinism tests.  Nothing here is CUDA-specific: the
CPU dev sandbox seeds identically to the T4 runtimes (device-conditional
CUDA generator seeding is included and verified on CUDA hardware only).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

__all__ = ["SeedState", "seed_everything", "capture_rng_state", "restore_rng_state"]


@dataclass(frozen=True)
class SeedState:
    """A fully materialized RNG seeding record (checkpointable)."""

    seed: int
    torch_seed: int
    numpy_seed: int
    python_seed: int

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "torch_seed": self.torch_seed,
            "numpy_seed": self.numpy_seed,
            "python_seed": self.python_seed,
        }


def seed_everything(seed: int, *, device: Optional[torch.device] = None) -> SeedState:
    """Seed every RNG stream deterministically from one integer.

    The three framework seeds are *derived* from ``seed`` by fixed offsets
    so the streams are independent by construction (not reseeded to the
    same value, which would correlate frameworks).  torch manual seeding
    covers the default CPU generator; a CUDA generator, when present on
    the requested device, is seeded with the torch seed as well.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    seed = int(seed)
    if not (0 <= seed <= 2**63 - 1):
        raise ValueError(f"seed out of range [0, 2^63-1]: {seed}")

    state = SeedState(
        seed=seed,
        torch_seed=(seed + 1) % (2**63),
        numpy_seed=(seed + 2) % (2**63),
        python_seed=(seed + 3) % (2**63),
    )
    torch.manual_seed(state.torch_seed)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(state.torch_seed)
    np.random.seed(state.numpy_seed % (2**32))
    random.seed(state.python_seed)
    return state


@dataclass(frozen=True)
class _RngSnapshot:
    """Point-in-time RNG state of all three frameworks."""

    torch: torch.Tensor
    numpy: dict
    python: tuple


def capture_rng_state() -> _RngSnapshot:
    """Capture the live RNG state of torch / NumPy / Python."""
    return _RngSnapshot(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
    )


def restore_rng_state(snapshot: _RngSnapshot) -> None:
    """Restore a captured RNG state exactly (checkpoint continuation)."""
    if not isinstance(snapshot, _RngSnapshot):
        raise TypeError(
            f"expected a capture_rng_state() snapshot, got {type(snapshot).__name__}"
        )
    torch.set_rng_state(snapshot.torch)
    np.random.set_state(snapshot.numpy)
    random.setstate(snapshot.python)
