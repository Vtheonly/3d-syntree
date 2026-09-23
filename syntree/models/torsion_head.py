"""Continuous SO(2) dihedral torsion head.

Emits the parameters ``(mu, kappa)`` of a von Mises distribution over the
dihedral angle of the newly formed single bond:

    p(phi | mu, kappa) = exp(kappa * cos(phi - mu)) / (2 pi I_0(kappa))

``mu`` is the predicted mean angle and ``kappa > 0`` the concentration.
Training uses the exact circular negative log-likelihood with a numerically
stable log-modified-Bessel implementation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_2PI = math.log(2.0 * math.pi)


def log_bessel_i0(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(I_0(x))`` for ``x >= 0``.

    Small arguments use the power series ``log(sum (x^2/4)^k / (k!)^2)``;
    large arguments use the asymptotic expansion
    ``x - 0.5*log(2 pi x) + log(1 + 1/(8x) + ...)``.
    """
    x = x.clamp(min=0.0)
    small = x < 5.0

    # Power series for small x.
    xs = torch.where(small, x, torch.zeros_like(x))
    term = torch.ones_like(xs)
    total = term.clone()
    x2_over_4 = (xs * xs) / 4.0
    for k in range(1, 16):
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

    The head consumes the scalar context vector plus invariant summaries of
    the equivariant pocket vectors (per-graph mean norm statistics).
    """

    def __init__(self, hidden_dim: int, min_kappa: float = 0.1, max_kappa: float = 50.0):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.min_kappa = float(min_kappa)
        self.max_kappa = float(max_kappa)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict ``(mu, kappa)`` for every graph in the batch.

        Args:
            scalar_context: ``[B, d]`` per-graph scalar context.
            vector_context: optional equivariant vector features; norms are
                pooled per graph into 3 invariant statistics (mean, max, std).
            batch: node-to-graph assignment for ``vector_context``.

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
        stats = scalar_context.new_zeros(b, 3)
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
            norms = torch.norm(vector_context, dim=1)  # [N, d]
            for g in range(b):
                sel = norms[batch == g]
                if sel.numel() == 0:
                    continue
                stats[g, 0] = sel.mean()
                stats[g, 1] = sel.max()
                stats[g, 2] = sel.std() if sel.numel() > 1 else sel.new_zeros(())

        h = torch.cat([scalar_context, stats], dim=-1)
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
    # Distribution utilities
    # ------------------------------------------------------------------
    @staticmethod
    def log_prob(mu: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Exact von Mises log-density ``log p(phi | mu, kappa)``."""
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
