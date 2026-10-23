"""Low-rank, data-dependent cross-variable modulation.

The gate consumes a complete feature window with layout ``[B, T, C, D]``.
It pools the current window to one summary value per variable, maps that
summary through a factored ``C -> rank -> 2*C`` adapter, and applies a
per-variable FiLM-style residual to every time/feature position.  No
``C x C`` parameter is created, so the module remains small for high-
dimensional multivariate inputs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LowRankChannelGate(nn.Module):
    """Window-conditioned low-rank cross-variable gate.

    Parameters
    ----------
    channels:
        Number of variables ``C`` in the input.
    dim:
        Optional feature width ``D``.  When supplied, the forward method
        checks the last input dimension.  The gate itself uses a scalar
        summary per channel and therefore has no ``C x D`` expansion.
    rank:
        Rank of the channel adapter.  For ``C > 1`` it is capped at ``C-1``
        so the factorization cannot become a dense ``C x C`` map.  ``C=1``
        uses rank one as the degenerate, still-valid case.
    dropout:
        Dropout applied to the low-rank summary during training.
    eps:
        Numerical floor used by the input validation path.

    Notes
    -----
    The learnable ``residual_scale`` is initialized to zero (ReZero).  The
    initial module is consequently an identity map even though the gate
    parameters are trainable.  ``tanh`` bounds the learned residual strength
    after optimization.
    """

    def __init__(
        self,
        channels: int,
        dim: Optional[int] = None,
        rank: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-6,
        *,
        feature_dim: Optional[int] = None,
        acf_summary_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dim is not None and feature_dim is not None and int(dim) != int(feature_dim):
            raise ValueError("dim and feature_dim disagree")
        if dim is None:
            dim = feature_dim
        if dim is not None and int(dim) <= 0:
            raise ValueError("dim must be positive when supplied")

        requested_rank = int(rank)
        if requested_rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        if acf_summary_dim is not None and int(acf_summary_dim) <= 0:
            raise ValueError("acf_summary_dim must be positive when supplied")

        # C=1 has no cross-variable direction, but retaining a rank-one
        # adapter keeps the module shape-compatible with the general case.
        effective_rank = min(requested_rank, channels - 1) if channels > 1 else 1
        effective_rank = max(1, effective_rank)

        self.channels = channels
        self.dim = None if dim is None else int(dim)
        self.requested_rank = requested_rank
        self.rank = effective_rank
        self.eps = float(eps)
        self.acf_summary_dim = (
            None if acf_summary_dim is None else int(acf_summary_dim)
        )

        # The two factor layers are the only channel-mixing parameters.  The
        # output has two values per channel: FiLM scale and shift.
        self.summary_to_latent = nn.Linear(channels, effective_rank, bias=True)
        self.latent_to_affine = nn.Linear(effective_rank, 2 * channels, bias=True)
        if self.acf_summary_dim is not None:
            # The ACF descriptor is a train-split [C, F] table.  Mapping each
            # descriptor to a rank-sized row and contracting it with the
            # current-window channel activity gives an O(C*r) cross-variable
            # adapter without introducing a C-by-C parameter matrix.
            self.acf_summary_to_latent = nn.Sequential(
                nn.Linear(self.acf_summary_dim, effective_rank),
                nn.GELU(),
                nn.Linear(effective_rank, effective_rank),
            )
            # The outer residual_scale remains the ReZero safety gate.  This
            # coefficient only controls the relative contribution of the
            # static ACF latent after the residual has been opened.
            self.acf_latent_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.acf_summary_to_latent = None
            self.register_parameter("acf_latent_scale", None)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize a small, nonzero gate behind the zero residual."""

        nn.init.xavier_uniform_(self.summary_to_latent.weight)
        nn.init.zeros_(self.summary_to_latent.bias)
        # A small nonzero output gives the ReZero scalar a useful gradient on
        # the first update while keeping the initial perturbation exactly zero.
        nn.init.normal_(self.latent_to_affine.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.latent_to_affine.bias)
        if self.acf_summary_to_latent is not None:
            nn.init.xavier_uniform_(self.acf_summary_to_latent[0].weight)
            nn.init.zeros_(self.acf_summary_to_latent[0].bias)
            nn.init.xavier_uniform_(self.acf_summary_to_latent[2].weight)
            nn.init.zeros_(self.acf_summary_to_latent[2].bias)
        with torch.no_grad():
            self.residual_scale.zero_()

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "LowRankChannelGate expects [batch, time, channels, dim], "
                "received shape {}".format(tuple(x.shape))
            )
        if int(x.shape[2]) != self.channels:
            raise ValueError(
                "input has {} channels, gate was built for {}".format(
                    int(x.shape[2]), self.channels
                )
            )
        if self.dim is not None and int(x.shape[3]) != self.dim:
            raise ValueError(
                "input feature width {} does not match configured dim {}".format(
                    int(x.shape[3]), self.dim
                )
            )
        if int(x.shape[1]) <= 0 or int(x.shape[3]) <= 0:
            raise ValueError("time and feature dimensions must be non-empty")

    def _window_summary(self, x: torch.Tensor) -> torch.Tensor:
        """Pool only the current input window to shape ``[B, C]``."""

        # Averaging over time and feature dimensions leaves one value per
        # variable.  It cannot read labels, forecast targets, or persistent
        # state from another window.
        summary = x.mean(dim=(1, 3))
        return summary

    def compute_gate(
        self,
        x: torch.Tensor,
        acf_summary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw per-channel ``(scale, shift)`` for inspection.

        Both tensors have shape ``[B, C]``.  The scale is returned before the
        final ``tanh`` used in :meth:`forward`, so callers can inspect the
        adapter output without changing the residual contract.
        """

        self._validate_input(x)
        summary = self._window_summary(x)
        latent = F.gelu(self.summary_to_latent(summary))
        if acf_summary is not None:
            if self.acf_summary_to_latent is None:
                raise ValueError(
                    "acf_summary was supplied but this gate was not configured "
                    "with acf_summary_dim"
                )
            if acf_summary.ndim != 2 or acf_summary.shape != (
                self.channels,
                self.acf_summary_dim,
            ):
                raise ValueError(
                    "acf_summary must have shape [{}, {}]".format(
                        self.channels, self.acf_summary_dim
                    )
                )
            if not torch.isfinite(acf_summary).all():
                raise ValueError("acf_summary must be finite")
            # Use RMS activity rather than the signed mean.  The latter is
            # identically zero under mean-centered normalization and would
            # make the ACF branch disappear on an otherwise informative
            # window.  The descriptor itself is detached because it is a
            # training-split statistic, not a learned data path.
            activity = x.square().mean(dim=(1, 3)).clamp_min(self.eps).sqrt()
            descriptor = acf_summary.detach().to(
                device=x.device, dtype=activity.dtype
            )
            descriptor_latent = self.acf_summary_to_latent(descriptor)
            acf_latent = torch.einsum(
                "bc,cr->br", activity, descriptor_latent
            ) / float(self.channels) ** 0.5
            acf_scale = torch.tanh(self.acf_latent_scale).to(
                device=latent.device, dtype=latent.dtype
            )
            latent = latent + acf_scale * acf_latent.to(latent.dtype)
        latent = self.dropout(latent)
        scale, shift = self.latent_to_affine(latent).chunk(2, dim=-1)
        return scale, shift

    def forward(
        self,
        x: torch.Tensor,
        acf_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the low-rank gate and preserve the ``[B,T,C,D]`` shape."""

        scale, shift = self.compute_gate(x, acf_summary=acf_summary)
        batch = int(x.shape[0])
        scale = torch.tanh(scale).reshape(batch, 1, self.channels, 1)
        shift = shift.reshape(batch, 1, self.channels, 1)
        alpha = torch.tanh(self.residual_scale).to(dtype=x.dtype, device=x.device)
        return (1.0 + alpha * scale) * x + alpha * shift

    @property
    def acf_parameter_count(self) -> int:
        """Return parameters belonging only to the optional ACF adapter."""

        if self.acf_summary_to_latent is None:
            return 0
        return int(
            sum(parameter.numel() for parameter in self.acf_summary_to_latent.parameters())
            + self.acf_latent_scale.numel()
        )


__all__ = ["LowRankChannelGate"]
