"""SE(3)-equivariant geometry kernels and PaiNN message-passing blocks.

Clean-room V2 implementation of the non-singular geometric operators and
the PaiNN equivariant message-passing scheme (issue #7 MATH-01).

**Non-singular primitives (the #7 mandate, constitution #45 §8.1).**
All norms and directional unit vectors in the V2 model path route through
:func:`safe_norm` / :func:`safe_unit_vector`:

    ``||x||_eps = sqrt(sum_k x_k^2 + eps^2) - eps``           (eps = 1e-5)
    ``grad ||x||_eps = x / sqrt(sum_k x_k^2 + eps^2)``

Unlike the raw L2 norm, the gradient is bounded by 1 everywhere and is
*exactly zero* at ``x = 0`` — there is no ``0/0`` division and no NaN
gradient spike at coincident coordinates or clashing fragments (the
failure mode that corrupted training in the pre-V2 system, where
``edge_vec / (edge_dist + 1e-8)`` and ``torch.norm`` produced divergent
gradients as ``d_ij -> 0``). Directional unit vectors are computed as
``v / sqrt(sum v^2 + eps^2)`` — well-defined for every input including
the zero vector (which maps to the zero direction) and strictly bounded
(``||unit|| < 1``).

**PaiNN blocks (published equations).** ``PaiNNInteraction`` and
``PaiNNMixing`` implement the message and atomwise-update equations of
the PaiNN paper (Schütt, Unke & Gastegger, "Equivariant message passing
for the prediction of tensorial properties and molecular spectra", ICML
2021; arXiv:2102.03150), Eqs. 7–10 — see decision D-V2-004:

* message (Eqs. 7–8), for edges ``j -> i`` with direction ``r_hat_ij``::

    ∆s_i = Σ_j φ_s(s_j) ∘ W_s(||r_ij||)
    ∆v_i = Σ_j v_j ∘ φ_vv(s_j) ∘ W_vv(||r_ij||)
            + Σ_j φ_vs(s_j) ∘ W_vs'(||r_ij||) ⊗ r_hat_ij

  with one shared atomwise network ``φ`` on the neighbour scalars (split
  afterwards) and bias-free radial filter banks ``W`` over the radial
  basis. The paper's singular direction ``r_ij / ||r_ij||`` is replaced
  by :func:`safe_unit_vector`.
* mixing (Eqs. 9–10), atomwise::

    ∆s_i = a_ss(s_i, ||V v_i||) + a_sv(s_i, ||V v_i||) · ⟨U v_i, V v_i⟩
    ∆v_i = a_vv(s_i, ||V v_i||) · U v_i

  with separate linear maps ``U``/``V`` over the feature dimension and a
  shared scaling network ``a`` over ``[s, ||V v||]``. Vector-feature
  norms contract through :func:`safe_norm`.

Both blocks are *residual* updates (each returns features plus its
delta); a full PaiNN layer composes message-then-update (owned by the
backbone, issue #21). Scalar outputs are invariant and vector outputs
equivariant under rigid motions — pinned by tests with random rotations.

**Precision policy (#45 §5.1, decision D-V2-005).** This module contains
no mixed-precision machinery: no fp16 clamps, no ``nan_to_num``, no NaN
replacement. The production path is pure FP32; genuinely poisoned inputs
fail loudly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "SAFE_NORM_EPS",
    "safe_norm",
    "safe_unit_vector",
    "PaiNNInteraction",
    "PaiNNMixing",
]

#: Default smoothing radius of the non-singular norm primitives (#7).
#: Well below any physical inter-atomic distance (Ångström scale) and well
#: above fp32 rounding noise, so the approximation error is negligible for
#: real geometry while the gradient stays bounded (magnitude ≤ 1,
#: exactly 0 at the zero vector) at coincident coordinates.
SAFE_NORM_EPS = 1e-5


# ---------------------------------------------------------------------------
# Non-singular primitives (issue #7 specified signatures)
# ---------------------------------------------------------------------------


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = SAFE_NORM_EPS,
) -> torch.Tensor:
    """Non-singular norm with a smooth, bounded gradient as ``x -> 0``.

    Implements ``||x||_eps = sqrt(sum_k x_k^2 + eps^2) - eps`` with the
    gradient ``x / sqrt(sum_k x_k^2 + eps^2)`` (magnitude bounded by 1
    everywhere, *exactly* 0 at ``x = 0``). Unlike ``torch.norm``, this
    cannot produce ``0/0`` NaN gradients at coincident coordinates or
    clashing fragments (#7 MATH-01, #45 §8.1). Directional vectors must
    never be computed by dividing a zero vector by its own norm — use
    :func:`safe_unit_vector`.

    Args:
        x: tensor of vectors; the norm is reduced over dimension ``dim``.
        dim: reduction dimension (default: the last one).
        keepdim: keep the reduced dimension as size 1.
        eps: smoothing radius (default :data:`SAFE_NORM_EPS`).

    Returns:
        Tensor of ``eps``-smoothed norms (shape as ``torch.linalg.norm``
        would produce).
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=keepdim) + (eps * eps)) - eps


def safe_unit_vector(
    vec: torch.Tensor,
    dim: int = -1,
    eps: float = SAFE_NORM_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directional unit vectors and distances strictly bounded away from 0/0.

    Returns ``(unit, dist)`` where

    * ``unit = vec / sqrt(sum vec^2 + eps^2)`` — a well-defined direction
      for every input, including the zero vector (``unit = 0`` there);
      its norm is strictly less than 1;
    * ``dist = (sqrt(sum vec^2 + eps^2) - eps).clamp(min=0)`` — the
      :func:`safe_norm` distance, non-negative by construction.

    Both values and their gradients with respect to ``vec`` stay finite
    for coincident coordinates; the gradient magnitude is bounded by
    ``1/eps + 1`` instead of diverging.
    """
    norm = torch.sqrt(torch.sum(vec * vec, dim=dim, keepdim=True) + (eps * eps))
    unit = vec / norm
    dist = (norm.squeeze(dim) - eps).clamp(min=0.0)
    return unit, dist


# ---------------------------------------------------------------------------
# PaiNN message block (paper Eqs. 7–8)
# ---------------------------------------------------------------------------


class PaiNNInteraction(nn.Module):
    """Equivariant inter-atomic message passing block (PaiNN Eqs. 7–8).

    A shared atomwise network ``φ`` transforms the *neighbour* scalars
    ``s_j`` (its output is split into the three gates ``φ_s``, ``φ_vv``,
    ``φ_vs``), and bias-free radial filter banks ``W`` expand the radial
    basis of ``||r_ij||`` into three matching filter vectors. Scalar
    messages are gated products; vector messages combine (a) the
    neighbour vector features gated by ``φ_vv ∘ W_vv`` — propagating
    directional information from previous passes — with (b) the
    non-singular direction ``r_hat_ij`` scaled by ``φ_vs ∘ W_vs'`` —
    injecting fresh geometric directions. Messages are scattered into the
    receiving atoms and applied as a residual.

    All normalisation operations route through :func:`safe_unit_vector` /
    :func:`safe_norm`: coincident coordinates (``d_ij -> 0``) yield a zero
    direction with bounded gradients instead of the ``0/0`` division of
    the deprecated ``edge_vec / (edge_dist + 1e-8)`` guard (#7 MATH-01).

    Args:
        hidden_dim: feature dimension ``H`` of scalar (``[N, H]``) and
            vector (``[N, 3, H]``) representations.
        num_radial: number of radial basis functions ``R`` feeding the
            filter banks (``edge_rbf`` has shape ``[E, R]``).
    """

    def __init__(self, hidden_dim: int, num_radial: int) -> None:
        super().__init__()
        if hidden_dim < 1 or num_radial < 1:
            raise ValueError(
                f"PaiNNInteraction requires hidden_dim >= 1 and "
                f"num_radial >= 1, got {hidden_dim=} {num_radial=}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_radial = int(num_radial)
        # Shared atomwise network φ on neighbour scalars (output split into
        # φ_s, φ_vv, φ_vs afterwards — one transform, three gates).
        self.shared_scalar_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3 * self.hidden_dim),
        )
        # Radial filter banks: linear combinations of the radial basis
        # functions (paper: W_s, W_vv, W_vs'); bias-free by design.
        self.radial_filters = nn.Linear(self.num_radial, 3 * self.hidden_dim, bias=False)

    def forward(
        self,
        scalar: torch.Tensor,  # [N, H]
        vector: torch.Tensor,  # [N, 3, H]
        edge_index: torch.Tensor,  # [2, E]  (messages flow col j -> row i)
        edge_vec: torch.Tensor,  # [E, 3]   r_ij = r_j - r_i
        edge_rbf: torch.Tensor,  # [E, R]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the residual-updated ``(scalar + ∆s^m, vector + ∆v^m)``."""
        self._check_inputs(scalar, vector, edge_index, edge_vec, edge_rbf)
        row, col = edge_index[0], edge_index[1]

        # Shared gates: φ(s_j) split three ways, W(||r_ij||) split three ways.
        phi_j = self.shared_scalar_net(scalar[col])  # [E, 3H]
        phi_s, phi_vv, phi_vs = torch.split(phi_j, self.hidden_dim, dim=-1)
        w = self.radial_filters(edge_rbf)  # [E, 3H]
        w_s, w_vv, w_vs = torch.split(w, self.hidden_dim, dim=-1)

        # Non-singular direction r_hat_ij (zero, with bounded gradient, at
        # coincident coordinates) — the #7 core requirement.
        dir_vec, _ = safe_unit_vector(edge_vec, dim=-1)  # [E, 3]

        # Eq. 7: gated scalar message.
        msg_s = phi_s * w_s  # [E, H]
        # Eq. 8: gated vector message = propagated v_j ⊙ (φ_vv ∘ W_vv)
        #                          + fresh direction ⊗ (φ_vs ∘ W_vs').
        msg_v = (
            vector[col] * (phi_vv * w_vv).unsqueeze(1)
            + dir_vec.unsqueeze(-1) * (phi_vs * w_vs).unsqueeze(1)
        )  # [E, 3, H]

        # Scatter-add messages into receiving atoms; residual update.
        agg_s = torch.zeros_like(scalar).index_add(0, row, msg_s)
        agg_v = torch.zeros_like(vector).index_add(0, row, msg_v)
        return scalar + agg_s, vector + agg_v

    def _check_inputs(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> None:
        n, h = scalar.shape[0], self.hidden_dim
        e, r = None, self.num_radial
        if scalar.dim() != 2:
            raise ValueError(f"scalar must be [N, H], got {tuple(scalar.shape)}")
        if vector.dim() != 3 or vector.shape[1] != 3 or vector.shape[2] != h:
            raise ValueError(
                f"vector must be [N, 3, H={h}], got {tuple(vector.shape)}"
            )
        if scalar.shape[1] != h:
            raise ValueError(
                f"scalar feature width must be H={h}, got {scalar.shape[1]}"
            )
        if vector.shape[0] != n:
            raise ValueError(
                f"scalar and vector must have the same number of atoms, "
                f"got {n} vs {vector.shape[0]}"
            )
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must be [2, E], got {tuple(edge_index.shape)}"
            )
        e = edge_index.shape[1]
        if edge_vec.dim() != 2 or edge_vec.shape[1] != 3 or edge_vec.shape[0] != e:
            raise ValueError(
                f"edge_vec must be [E={e}, 3], got {tuple(edge_vec.shape)}"
            )
        if edge_rbf.dim() != 2 or edge_rbf.shape[1] != r or edge_rbf.shape[0] != e:
            raise ValueError(
                f"edge_rbf must be [E={e}, R={r}], got {tuple(edge_rbf.shape)}"
            )
        if e and (
            int(edge_index.max()) >= n or int(edge_index.min()) < 0
        ):
            raise ValueError(
                f"edge_index entries must lie in [0, N={n}), got range "
                f"[{int(edge_index.min())}, {int(edge_index.max())}]"
            )


# ---------------------------------------------------------------------------
# PaiNN atomwise mixing block (paper Eqs. 9–10)
# ---------------------------------------------------------------------------


class PaiNNMixing(nn.Module):
    """Equivariant atomwise update block mixing scalar and vector channels
    (PaiNN Eqs. 9–10).

    Two bias-free linear maps ``U``/``V`` act on the vector channels (over
    the feature dimension, shared across the three spatial components);
    the invariant norm ``||V v||`` — contracted through
    :func:`safe_norm`, never ``torch.norm`` — and the inner product
    ``⟨U v, V v⟩`` couple the vector channels back into the scalars. A
    shared scaling network ``a(s, ||V v||)`` produces the three gates::

        ∆s = a_ss + a_sv · ⟨U v, V v⟩
        ∆v = a_vv ⊙ U v

    applied as residuals. ``epsilon`` is the *smoothing radius* of the
    non-singular norm (not an additive bias inside the norm — the
    deprecated ``torch.norm(v + 1e-8)`` shifted every vector before
    taking its norm, which was mathematically wrong and still singular).

    Args:
        hidden_dim: feature dimension ``H`` (must match the interaction
            block).
        epsilon: smoothing radius for :func:`safe_norm` (default
            :data:`SAFE_NORM_EPS`).
    """

    def __init__(self, hidden_dim: int, epsilon: float = SAFE_NORM_EPS) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        self.hidden_dim = int(hidden_dim)
        self.epsilon = float(epsilon)
        self.vec_proj_u = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.vec_proj_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.shared_net = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3 * self.hidden_dim),
        )

    def forward(
        self, scalar: torch.Tensor, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the residual-updated ``(scalar + ∆s^u, vector + ∆v^u)``."""
        if scalar.dim() != 2 or vector.dim() != 3 or vector.shape[1] != 3:
            raise ValueError(
                f"unexpected feature shapes: scalar {tuple(scalar.shape)}, "
                f"vector {tuple(vector.shape)} (expected [N, H] and [N, 3, H])"
            )
        if scalar.shape[0] != vector.shape[0] or scalar.shape[1] != vector.shape[2]:
            raise ValueError(
                f"inconsistent scalar/vector shapes: {tuple(scalar.shape)} "
                f"vs {tuple(vector.shape)}"
            )

        u = self.vec_proj_u(vector)  # U v  -> [N, 3, H]
        v = self.vec_proj_v(vector)  # V v  -> [N, 3, H]
        # Non-singular vector-feature norms (#7): bounded gradients, and no
        # NaN-replacement clamps anywhere (fail loudly on poisoned inputs).
        v_norm = safe_norm(v, dim=1, eps=self.epsilon)  # [N, H]
        a = self.shared_net(torch.cat([scalar, v_norm], dim=-1))  # [N, 3H]
        a_ss, a_sv, a_vv = torch.split(a, self.hidden_dim, dim=-1)

        # Invariant coupling ⟨U v, V v⟩ over the spatial dimension.
        uv_dot = (u * v).sum(dim=1)  # [N, H]

        delta_s = a_ss + a_sv * uv_dot  # Eq. 9
        delta_v = a_vv.unsqueeze(1) * u  # Eq. 10
        return scalar + delta_s, vector + delta_v
