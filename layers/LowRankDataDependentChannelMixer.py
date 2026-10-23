"""Small data-dependent channel mixer for encoded time-series tokens.

The module follows the useful part of CMamba's GDD-MLP contract: a complete
variable-token state is pooled, a low-rank adapter produces channel-wise affine
parameters, and the resulting modulation is applied to the encoded tokens.
There is deliberately no dense ``C x C`` parameter and the residual is
ReZero-initialized so an enabled module is exactly nested in the parent model
at initialization.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LowRankDataDependentChannelMixer(nn.Module):
    """Window-conditioned low-rank FiLM mixer over channel tokens.

    Parameters
    ----------
    channels:
        Number of input variables.
    dim:
        Encoded token width.
    rank:
        Bottleneck rank of the channel adapter.  It is capped below ``C`` for
        ``C > 1`` so the adapter stays strictly low-rank.
    dropout:
        Dropout applied to the pooled channel descriptor during training.

    Notes
    -----
    The input is ``[batch, channels, patches, dim]``.  The current
    implementation commonly uses one patch per variable (a full-window
    linear token), but keeping the patch axis explicit makes the module
    compatible with patch encoders and mirrors CMamba's state layout.
    """

    def __init__(
        self,
        channels: int,
        dim: int,
        rank: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-6,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        dim = int(dim)
        requested_rank = int(rank)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if requested_rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        try:
            init_scale = float(init_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("init_scale must be a finite value in [0, 1]") from exc
        if not torch.isfinite(torch.tensor(init_scale)) or not 0.0 <= init_scale <= 1.0:
            raise ValueError("init_scale must be a finite value in [0, 1]")

        effective_rank = min(requested_rank, channels - 1) if channels > 1 else 1
        effective_rank = max(1, effective_rank)
        self.channels = channels
        self.dim = dim
        self.requested_rank = requested_rank
        self.rank = effective_rank
        self.eps = float(eps)
        self.initial_scale = init_scale

        # Mean and RMS retain complementary information under common
        # mean-centering/normalization.  The adapter is O(C*r), not O(C^2).
        self.summary_to_latent = nn.Linear(2 * channels, effective_rank)
        self.latent_to_affine = nn.Linear(effective_rank, 2 * channels)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.summary_to_latent.weight)
        nn.init.zeros_(self.summary_to_latent.bias)
        nn.init.normal_(self.latent_to_affine.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.latent_to_affine.bias)
        with torch.no_grad():
            self.residual_scale.fill_(self.initial_scale)

    def _validate_input(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 4:
            raise ValueError(
                "LowRankDataDependentChannelMixer expects [batch, channels, "
                "patches, dim], received shape {}".format(tuple(tokens.shape))
            )
        if int(tokens.shape[1]) != self.channels:
            raise ValueError(
                "input has {} channels, mixer was built for {}".format(
                    int(tokens.shape[1]), self.channels
                )
            )
        if int(tokens.shape[3]) != self.dim:
            raise ValueError(
                "input feature width {} does not match configured dim {}".format(
                    int(tokens.shape[3]), self.dim
                )
            )
        if int(tokens.shape[2]) <= 0:
            raise ValueError("patch dimension must be non-empty")

    def channel_summary(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the data-dependent descriptor with shape ``[B, 2*C]``."""

        self._validate_input(tokens)
        mean = tokens.mean(dim=(2, 3))
        rms = tokens.square().mean(dim=(2, 3)).clamp_min(self.eps).sqrt()
        return torch.cat((mean, rms), dim=-1)

    def compute_residual(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return only the ReZero residual, preserving the input shape."""

        summary = self.channel_summary(tokens)
        latent = self.dropout(F.gelu(self.summary_to_latent(summary)))
        scale, shift = self.latent_to_affine(latent).chunk(2, dim=-1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        alpha = torch.tanh(self.residual_scale).to(
            dtype=tokens.dtype, device=tokens.device
        )
        return alpha * (scale * tokens + shift)

    def forward(
        self, tokens: torch.Tensor, *, return_residual: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Apply the mixer and optionally return its residual separately."""

        residual = self.compute_residual(tokens)
        mixed = tokens + residual
        if return_residual:
            return mixed, residual
        return mixed

    @property
    def parameter_count(self) -> int:
        """Number of learnable parameters owned by the mixer."""

        return int(sum(parameter.numel() for parameter in self.parameters()))


class LowRankGlobalChannelStateMixer(nn.Module):
    """Efficient global channel-state adapter for a shared temporal state.

    window has shape [B, T_w, C] and hidden has shape [B, T_h, D].
    The adapter can summarize each observed channel with mean and RMS, or
    consume an already available shared hidden summary.  It maps the summary
    through a rank-``r`` bottleneck and emits one hidden-space FiLM pair.  It
    therefore carries cross-variable information without materialising a
    ``C x C`` map or a per-channel ``C x D`` state.
    The zero-initialized ReZero coefficient makes the enabled path an exact
    identity at initialization.
    """

    def __init__(
        self,
        channels: int,
        dim: int,
        rank: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-6,
        summary_dim: int = None,
    ) -> None:
        super().__init__()
        channels = int(channels)
        dim = int(dim)
        requested_rank = int(rank)
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        if requested_rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        effective_rank = min(requested_rank, channels - 1) if channels > 1 else 1
        effective_rank = max(1, effective_rank)
        self.channels = channels
        self.dim = dim
        self.summary_dim = 2 * channels if summary_dim is None else int(summary_dim)
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be positive")
        self.requested_rank = requested_rank
        self.rank = min(
            effective_rank,
            self.summary_dim - 1 if self.summary_dim > 1 else 1,
        )
        self.rank = max(1, self.rank)
        self.eps = float(eps)
        self.summary_to_latent = nn.Linear(self.summary_dim, self.rank)
        self.latent_to_affine = nn.Linear(self.rank, 2 * dim)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.summary_to_latent.weight)
        nn.init.zeros_(self.summary_to_latent.bias)
        nn.init.normal_(self.latent_to_affine.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.latent_to_affine.bias)
        with torch.no_grad():
            self.residual_scale.zero_()

    def _validate(self, window: torch.Tensor, hidden: torch.Tensor) -> None:
        if window.ndim != 3 or hidden.ndim != 3:
            raise ValueError(
                "LowRankGlobalChannelStateMixer expects window [B,T,C] and "
                "hidden [B,T,D]"
            )
        if int(window.shape[0]) != int(hidden.shape[0]):
            raise ValueError("window and hidden batch dimensions must match")
        if int(window.shape[2]) != self.channels:
            raise ValueError(
                "window has {} channels, mixer was built for {}".format(
                    int(window.shape[2]), self.channels
                )
            )
        if int(hidden.shape[2]) != self.dim:
            raise ValueError(
                "hidden width {} does not match configured dim {}".format(
                    int(hidden.shape[2]), self.dim
                )
            )

    def channel_summary(self, window: torch.Tensor) -> torch.Tensor:
        """Return the ``[B, 2C]`` mean/RMS descriptor."""

        if window.ndim != 3 or int(window.shape[2]) != self.channels:
            raise ValueError("window must have shape [B,T,C]")
        mean = window.mean(dim=1)
        rms = window.square().mean(dim=1).clamp_min(self.eps).sqrt()
        return torch.cat((mean, rms), dim=-1)

    def compute_affine_from_summary(
        self, summary: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hidden-space scale and shift with shape [B,D]."""

        if summary.ndim != 2 or int(summary.shape[1]) != self.summary_dim:
            raise ValueError(
                "summary must have shape [B, {}]".format(self.summary_dim)
            )
        latent = self.dropout(F.gelu(self.summary_to_latent(summary)))
        scale, shift = self.latent_to_affine(latent).chunk(2, dim=-1)
        return torch.tanh(scale), shift

    def _fit_summary_width(self, summary: torch.Tensor) -> torch.Tensor:
        """Pad or truncate a raw channel summary for a custom adapter width."""

        width = int(summary.shape[1])
        if width < self.summary_dim:
            return F.pad(summary, (0, self.summary_dim - width))
        if width > self.summary_dim:
            return summary[:, : self.summary_dim]
        return summary

    def compute_affine(self, window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.summary_dim != 2 * self.channels:
            raise ValueError(
                "window summary is unavailable when summary_dim differs from 2*channels"
            )
        return self.compute_affine_from_summary(self.channel_summary(window))

    def forward_from_summary(
        self, summary: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        if hidden.ndim != 3 or int(hidden.shape[2]) != self.dim:
            raise ValueError(
                "hidden must have shape [B,T, {}]".format(self.dim)
            )
        if int(summary.shape[0]) != int(hidden.shape[0]):
            raise ValueError("summary and hidden batch dimensions must match")
        scale, shift = self.compute_affine_from_summary(summary)
        alpha = torch.tanh(self.residual_scale).to(
            dtype=hidden.dtype, device=hidden.device
        )
        return hidden + alpha * (
            scale.unsqueeze(1) * hidden + shift.unsqueeze(1)
        )

    def forward(self, window: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        self._validate(window, hidden)
        return self.forward_from_summary(
            self._fit_summary_width(self.channel_summary(window)), hidden
        )

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


__all__ = ["LowRankDataDependentChannelMixer", "LowRankGlobalChannelStateMixer"]
