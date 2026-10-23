"""Causal value and temporal embeddings for forecasting models.

The repository's historical :class:`layers.Embed.TokenEmbedding` uses circular
padding.  That is useful for some periodic encoders, but it lets the first
output token read the end of the input window.  The classes in this module keep
the existing embedding semantics while making the value convolution strictly
left padded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding


class CausalTokenEmbedding(nn.Module):
    """Project values with a convolution that only reads the current/past input.

    Parameters
    ----------
    c_in:
        Number of input variables.
    d_model:
        Output embedding width.
    kernel_size:
        Causal convolution width.  The default is three, matching the
        historical ``TokenEmbedding`` while replacing circular padding with
        zero padding on the left only.
    """

    def __init__(self, c_in: int, d_model: int, kernel_size: int = 3):
        super().__init__()
        c_in = int(c_in)
        d_model = int(d_model)
        kernel_size = int(kernel_size)
        if c_in <= 0 or d_model <= 0:
            raise ValueError("c_in and d_model must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.c_in = c_in
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.left_padding = kernel_size - 1
        # Padding is performed explicitly in forward so right padding cannot
        # accidentally be introduced by a future change to Conv1d options.
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.tokenConv.weight,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an embedded sequence with shape ``[B, T, d_model]``."""

        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, channels]")
        if x.shape[-1] != self.c_in:
            raise ValueError(
                "expected {} input channels, received {}".format(
                    self.c_in, x.shape[-1]
                )
            )

        values = x.transpose(1, 2)
        if self.left_padding:
            values = F.pad(values, (self.left_padding, 0))
        return self.tokenConv(values).transpose(1, 2)


class CausalDataEmbedding(nn.Module):
    """Causal value embedding plus the repository's temporal/position terms.

    The constructor mirrors ``layers.Embed.DataEmbedding``.  ``x_mark`` may be
    omitted or set to ``None``.  A cloned time-mark tensor is passed to the
    legacy temporal module because its ``TimeFeatureEmbedding`` implementation
    masks ``wo_freq`` columns in place; callers therefore observe no mutation.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        wo_freq: str = " ",
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.value_embedding = CausalTokenEmbedding(
            c_in=c_in,
            d_model=d_model,
            kernel_size=kernel_size,
        )
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type != "timeF":
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model,
                embed_type=embed_type,
                freq=freq,
                wo_freq=wo_freq,
            )
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model,
                embed_type=embed_type,
                freq=freq,
                wo_freq=wo_freq,
            )
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed ``x`` and optional time marks without mutating either input."""

        value = self.value_embedding(x)
        position = self.position_embedding(x)
        if x_mark is not None:
            # TimeFeatureEmbedding currently applies its frequency mask in
            # place.  Cloning here preserves that established behavior while
            # keeping the caller's marks unchanged.
            temporal = self.temporal_embedding(x_mark.clone())
            value = value + temporal
        return self.dropout(value + position)


__all__ = ["CausalTokenEmbedding", "CausalDataEmbedding"]
