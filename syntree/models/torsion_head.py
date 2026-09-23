"""Continuous SO(2) dihedral torsion head.

Emits the parameters ``(mu, kappa)`` of a von Mises distribution over the
dihedral angle of the newly formed single bond:

    p(phi | mu, kappa) = exp(kappa * cos(phi - mu)) / (2 pi I_0(kappa))

``mu`` is the predicted mean angle and ``kappa > 0`` the concentration.
Training uses the exact circular negative log-likelihood with a numerically
stable log-modified-Bessel implementation.

AMP safety: ``torch.norm`` squares its inputs and fp16 overflows above
65,504, so vector features with per-component magnitude ~150 (routine after
a few epochs of training) overflow to ``inf`` inside an fp16 autocast region
and poison the loss with NaN. Every norm / statistic / loss computation in
this module therefore runs in fp32, with a saturation cap on pooled norms,
and the head's MLP executes inside an autocast-disabled region. The head is
three small linears on ``[B, d+3]`` -- the fp32 cost is negligible.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2.0 * math.pi)

# Saturation cap for pooled vector-feature norms. Norms only feed summary
# statistics into the head, so clamping them keeps the head finite even if
# the backbone's vector channel grows pathologically under fp16 training.
_NORM_CAP = 1e4


@contextmanager
def _autocast_disabled(device: torch.device):
    """Execute a block in fp32 even inside an outer autocast region.

    Falls back to running unguarded on backends without autocast support.
    """
    entered = False
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            entered = True
            yield
    except Exception:
        if entered:
            raise
        yield  # autocast unsupported on this backend; run without the guard


def log_bessel_i0(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(I_0(x))`` for ``x >= 0``.

    Small arguments use the power series ``log(sum (x^2/4)^k / (k!)^2)``;
    large arguments use the asymptotic expansion
    ``x - 0.5*log(2 pi x) + log(1 + 1/(8x) + ...)``.

    Always computed in float64 (a cheap 1-D tensor): the series/asymptotic
    arithmetic is precision-sensitive and must stay out of fp16/fp32
    regions; verified against scipy's ``i0e`` to <1e-5 absolute.
    """
    x = x.double().clamp(min=0.0)
    small = x < 10.0

    # Power series for small x (20 terms converge to <1e-8 for x < 10).
    xs = torch.where(small, x, torch.zeros_like(x))
    term = torch.ones_like(xs)
    total = term.clone()
    x2_over_4 = (xs * xs) / 4.0
    for k in range(1, 21):
        term = term * x2_over_4 / (k * k)
        total = total + term
    series = torch.log(total.clamp(min=1e-30))

    # Asymptotic expansion for large x.
    xl = torch.where(small, torch.full_like(x, 6.0), x)
    inv = 1.0 / (8.0 * xl)
    asym = xl - 0.5 * torch.log(2.0 * math.pi * xl) + torch.log1p(
        inv * (1.0 + inv * (4.5 + inv * 37.5))
    )

    return torch.where(small, series, asym)


class ContinuousTorsionHead(nn.Module):
    """Predicts von Mises parameters ``(mu, kappa)`` for one dihedral.

    The head consumes the scalar context vector, invariant summaries of
    the equivariant pocket vectors (per-graph mean norm statistics), and
    the **selected synthon's embedding** (bug report 2 / Flaw 2 fix: the
    optimal dihedral depends on the sterics of the specific group being
    attached - a methyl and a spiro-adamantyl must not share one angle).
    When no synthon embedding is available (e.g. a pure value probe), a
    learned null-synthon embedding is used so the input width stays fixed.
    """

    def __init__(self, hidden_dim: int, min_kappa: float = 0.1, max_kappa: float = 50.0):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.min_kappa = float(min_kappa)
        self.max_kappa = float(max_kappa)
        # Learned stand-in for "no synthon selected yet".
        self.null_synthon = nn.Parameter(torch.zeros(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # [cos_mu, sin_mu, log_kappa]
        )

    def forward(
        self,
        scalar_context: torch.Tensor,   # [B, d]
        vector_context: Optional[torch.Tensor] = None,  # [N, 3, d] pooled or raw
        batch: Optional[torch.Tensor] = None,           # [N]
        synthon_embedding: Optional[torch.Tensor] = None,  # [B, d] selected synthon
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict ``(mu, kappa)`` for every graph in the batch.

        Args:
            scalar_context: ``[B, d]`` per-graph scalar context.
            vector_context: optional equivariant vector features; norms are
                pooled per graph into 3 invariant statistics (mean, max, std).
            batch: node-to-graph assignment for ``vector_context``.
            synthon_embedding: ``[B, d]`` embedding of the synthon that will
                occupy the rotated bond (teacher forcing during training, the
                selected action at inference). Falls back to a learned null
                embedding when omitted.

        Returns:
            ``(mu, kappa)`` with shapes ``[B]`` and ``[B]``; ``mu`` lies in
            ``[-pi, pi)`` and ``kappa`` in ``[min_kappa, max_kappa]``.
        """
        if scalar_context.dim() != 2:
            raise ValueError(
                f"scalar_context must be [B, d], got {tuple(scalar_context.shape)}"
            )
        if scalar_context.size(-1) != self.hidden_dim:
            raise ValueError(
                f"scalar_context dim {scalar_context.size(-1)} != "
                f"hidden_dim {self.hidden_dim}"
            )

        b = scalar_context.size(0)
        device = scalar_context.device
        if synthon_embedding is None:
            synthon_embedding = self.null_synthon.unsqueeze(0).expand(b, -1)
        else:
            if synthon_embedding.dim() != 2 or synthon_embedding.size(0) != b:
                raise ValueError(
                    f"synthon_embedding must be [B, d] with B={b}, got "
                    f"{tuple(synthon_embedding.shape)}"
                )
            if synthon_embedding.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"synthon embedding dim {synthon_embedding.size(-1)} != "
                    f"hidden_dim {self.hidden_dim}"
                )

        stats = torch.zeros(b, 3, device=device, dtype=torch.float32)
        if vector_context is not None and vector_context.numel() > 0:
            if vector_context.dim() != 3 or vector_context.size(1) != 3:
                raise ValueError(
                    f"vector_context must be [N, 3, d], got {tuple(vector_context.shape)}"
                )
            if batch is None:
                batch = torch.zeros(
                    vector_context.size(0), dtype=torch.long,
                    device=vector_context.device,
                )
            stats = self._pool_vector_stats(vector_context, batch, b)

        # Run the head in fp32, immune to any surrounding fp16 autocast.
        with _autocast_disabled(device):
            h = torch.cat(
                [scalar_context.float(), stats, synthon_embedding.float()], dim=-1
            )
            out = self.mlp(h)

            cos_mu, sin_mu, raw_kappa = out[..., 0], out[..., 1], out[..., 2]
            norm = torch.sqrt(cos_mu ** 2 + sin_mu ** 2 + 1e-8)
            cos_mu = cos_mu / norm
            sin_mu = sin_mu / norm

            mu = torch.atan2(sin_mu, cos_mu)
            kappa = F.softplus(raw_kappa) + self.min_kappa
            kappa = kappa.clamp(max=self.max_kappa)
        return mu, kappa

    # ------------------------------------------------------------------
    # Vector-feature statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _pool_vector_stats(
        vector_context: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Per-graph ``(mean, max, std)`` of vector-feature norms, in fp32.

        Replaces the original per-graph Python loop (O(B) boolean masks per
        forward -- a serious drag at batch 640) with scatter ops, and hardens
        the math against fp16 overflow: ``torch.norm`` squares its inputs,
        and fp16 overflows above 65,504, so the raw ``pocket_v`` tensor of a
        growing backbone would turn ``inf`` and then ``std`` into NaN. All
        math here runs in fp32 with a saturation cap; empty graphs keep the
        historical all-zeros statistics.
        """
        v = vector_context.float()
        norms = torch.norm(v, dim=1)                                   # [N, d]
        norms = torch.nan_to_num(norms, nan=0.0, posinf=_NORM_CAP, neginf=0.0)
        norms = norms.clamp(max=_NORM_CAP)

        d_channels = norms.size(-1)
        node_counts = torch.bincount(batch, minlength=num_graphs)      # [B]
        entry_counts = (node_counts * d_channels).to(norms.dtype)     # [B]

        sums = norms.new_zeros(num_graphs, d_channels)
        sums.index_add_(0, batch, norms)                               # [B, d]
        mean_g = sums.sum(-1) / entry_counts.clamp(min=1.0)            # [B]

        max_v = norms.new_full((num_graphs, d_channels), float("-inf"))
        max_v.scatter_reduce_(
            0, batch.unsqueeze(-1).expand_as(norms), norms,
            reduce="amax", include_self=True,
        )
        max_g = max_v.max(dim=-1).values                               # [B]
        max_g = torch.where(node_counts > 0, max_g, torch.zeros_like(max_g))

        # Unbiased std over all n_g * d entries, two-pass for stability.
        dev = norms - mean_g[batch].unsqueeze(-1)                      # [N, d]
        sq = norms.new_zeros(num_graphs, d_channels)
        sq.index_add_(0, batch, dev * dev)
        denom = (entry_counts - 1.0).clamp(min=1.0)                    # [B]
        std_g = torch.sqrt((sq.sum(-1) / denom).clamp(min=0.0))        # [B]

        return torch.stack([mean_g, max_g, std_g], dim=-1)            # [B, 3]

    # ------------------------------------------------------------------
    # Distribution utilities
    # ------------------------------------------------------------------
    @staticmethod
    def log_prob(mu: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Exact von Mises log-density ``log p(phi | mu, kappa)`` (fp32)."""
        mu, kappa, phi = mu.float(), kappa.float(), phi.float()
        return kappa * torch.cos(phi - mu) - _LOG_2PI - log_bessel_i0(kappa)

    @classmethod
    def loss_fn(
        cls,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        target_dihedral: torch.Tensor,
    ) -> torch.Tensor:
        """Circular negative log-likelihood (mean over batch).

        The exact loss is
        ``-kappa * cos(phi - mu) + log(2 pi I_0(kappa))``
        which is fully differentiable thanks to the stable Bessel
        implementation.
        """
        if not (mu.shape == kappa.shape == target_dihedral.shape):
            raise ValueError(
                f"Shape mismatch: mu {tuple(mu.shape)}, kappa {tuple(kappa.shape)}, "
                f"target {tuple(target_dihedral.shape)}"
            )
        # fp32: the kappa * cos() and Bessel terms are precision-sensitive
        # and this is the loss that NaN-ed under fp16 autocast (Colab T4).
        mu = mu.float()
        kappa = kappa.float()
        target_dihedral = target_dihedral.float()
        return (-cls.log_prob(mu, kappa, target_dihedral)).mean()

    @staticmethod
    def sample(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Draw one dihedral per graph via rejection sampling.

        Best–Fisher-style rejection sampling on the cosine density; falls
        back to the mean ``mu`` when it fails to accept within 100 draws
        (vanishingly rare for ``kappa <= 50``).
        """
        n = mu.numel()
        device = mu.device
        mu = mu.float()
        kappa = kappa.float()
        phi = mu.clone()
        pending = torch.ones(n, dtype=torch.bool, device=device)
        r = 1.0 + torch.sqrt(1.0 + 4.0 * kappa ** 2)
        rho = (r - torch.sqrt(2.0 * r)) / (2.0 * kappa)
        rho = torch.clamp(rho, 1e-6, 1.0 - 1e-6)
        accepted = mu.new_zeros(n, dtype=torch.bool)
        for _ in range(100):
            idx = torch.nonzero(~accepted, as_tuple=True)[0]
            if idx.numel() == 0:
                break
            u = torch.rand(idx.numel(), device=device)
            v = torch.rand(idx.numel(), device=device)
            k_idx = kappa[idx]
            mu_idx = mu[idx]
            candidate = mu_idx + torch.atan2(
                rho[idx] * torch.sin(v), 1.0 - rho[idx] * torch.cos(v)
            ) / k_idx.clamp(min=1e-6)
            accept = u < torch.exp(k_idx * (torch.cos(candidate - mu_idx) - 1.0))
            acc_idx = idx[accept]
            phi[acc_idx] = candidate[accept]
            accepted[acc_idx] = True
        # Wrap into [-pi, pi).
        return (phi + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["ContinuousTorsionHead", "log_bessel_i0"]
