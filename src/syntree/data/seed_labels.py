"""Supervised seed/anchor labels for canonical decompositions (#39 item 6).

The deterministic root of the #15 canonical decomposition is a
**supervision mechanism** — it must never force inference to start from
one heuristic seed (the #39 issue addendum).  The supervised label for
the explicit seed action ``a_seed = (B_0, h_0)`` is:

* **B_0** — the catalog seed block the #18 extractor matched to the
  decomposition's root fragment (an exact certified catalog row; the
  label is already carried by :class:`TrajectoryRecord`).
* **h_0** — the anchor whose position is nearest the *crystal* centroid
  of the root fragment, over the same deterministic anchor generator the
  environment and the policy use (one authoritative implementation).

Ambiguity handling (documented, deterministic):
* several anchors equidistant from the seed centroid -> the **lowest
  anchor index** wins (the generator's order is deterministic);
* several exact-matching seed rows (identical parent graphs) -> the #18
  extractor already resolves by lowest catalog row index;
* the anchor label is computed against the *crystal* geometry — the
  supervised convention "the seed action that reconstructs where the
  ligand actually sits" — while inference remains free to select any
  legal ``(B_0, h_0)`` the policy scores (the #39 acceptance: the
  deterministic decomposition never pins inference to one heuristic
  root).

This module depends on ``numpy``/``torch`` (through the generator) and
on :class:`~syntree.engine.environment.PocketDefinition`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from syntree.engine.environment import PocketDefinition

__all__ = ["SeedAnchorLabel", "supervised_anchor_label"]


@dataclass(frozen=True)
class SeedAnchorLabel:
    """The supervised anchor component of the seed action."""

    anchor_idx: int
    anchor_distance: float
    num_anchors: int
    ambiguity: str = ""

    @property
    def is_tie_break(self) -> bool:
        return "tie" in self.ambiguity.lower()


def supervised_anchor_label(
    pocket: PocketDefinition, seed_centroid: Tuple[float, float, float]
) -> SeedAnchorLabel:
    """The anchor label for one crystal seed centroid (#39 item 6).

    Enumerates the pocket's anchors with the authoritative generator
    (``syntree.models.policy.pocket_anchor_candidates``, default
    parameters — the same call the environment's pre-seed state and the
    policy's anchor stage use) and selects the anchor nearest the seed
    fragment's crystal centroid.  Ties resolve to the lowest anchor
    index (documented ambiguity handling).  Fail-closed on an empty
    anchor set (a non-empty pocket always yields >= 1 anchor).
    """
    import torch

    from syntree.models.policy import pocket_anchor_candidates

    positions, _assignment = pocket_anchor_candidates(
        torch.as_tensor(pocket.positions, dtype=torch.float64),
        torch.as_tensor(pocket.atomic_numbers, dtype=torch.float64),
        torch.as_tensor(pocket.charges, dtype=torch.float64),
    )
    anchors = np.ascontiguousarray(positions.numpy(), dtype=np.float64)
    if anchors.shape[0] == 0:
        raise ValueError("the anchor generator produced no anchors (invalid pocket)")
    target = np.asarray(seed_centroid, dtype=np.float64)
    distances = np.linalg.norm(anchors - target, axis=1)
    best = int(np.argmin(distances))  # argmin ties -> the lowest index
    tie = bool(np.count_nonzero(np.isclose(distances, distances[best], rtol=0.0, atol=1e-12)) > 1)
    return SeedAnchorLabel(
        anchor_idx=best,
        anchor_distance=float(distances[best]),
        num_anchors=int(anchors.shape[0]),
        ambiguity=(
            "equidistant anchors resolved to the lowest index"
            if tie
            else "unique nearest anchor"
        ),
    )
