"""A small, dependency-free iTransformer-style forecasting baseline.

The inverted representation treats each input variate as a token.  The
complete look-back window is projected to ``d_model`` for every token and a
standard Transformer encoder mixes information across variates.  A linear
head then predicts the requested horizon for each token.  This keeps the
baseline close to the published iTransformer design while making it
compatible with this repository's existing ``[B, T, C]`` trainer contract.
"""

from __future__ import annotations

import torch
from torch import nn


class Model(nn.Module):
    """Lightweight inverted Transformer for long/short-term forecasting."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError(
                "iTransformer baseline supports forecasting tasks only; "
                "received {!r}".format(self.task_name)
            )

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if self.enc_in <= 0 or self.c_out <= 0:
            raise ValueError("enc_in and c_out must be positive")
        if self.c_out > self.enc_in:
            raise ValueError(
                "iTransformer requires c_out <= enc_in for target-channel "
                "de-normalization (got c_out={}, enc_in={})".format(
                    self.c_out, self.enc_in
                )
            )

        d_model = int(getattr(configs, "d_model", 32))
        n_heads = int(getattr(configs, "n_heads", 4))
        e_layers = int(getattr(configs, "e_layers", 1))
        d_ff = int(getattr(configs, "d_ff", 4 * d_model))
        dropout = float(getattr(configs, "dropout", 0.1))
        if d_model <= 0 or n_heads <= 0 or e_layers <= 0 or d_ff <= 0:
            raise ValueError("d_model, n_heads, e_layers and d_ff must be positive")
        if d_model % n_heads:
            raise ValueError(
                "d_model ({}) must be divisible by n_heads ({})".format(
                    d_model, n_heads
                )
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = d_model
        # Inverted embedding: [B, C, T] -> [B, C, d_model].
        self.value_embedding = nn.Linear(self.seq_len, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.encoder_norm = nn.LayerNorm(d_model)
        # Each variable token independently predicts the future time axis.
        self.temporal_head = nn.Linear(d_model, self.pred_len)
        # ``MS`` uses all variables as input but predicts only the target
        # variable.  A small projection preserves that trainer contract.
        self.channel_projection = (
            nn.Linear(self.enc_in, self.c_out)
            if self.c_out != self.enc_in
            else nn.Identity()
        )

    def _normalise(self, x_enc: torch.Tensor):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        return centered / stdev, means, stdev

    def _encode(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_enc.ndim != 3:
            raise ValueError("x_enc must have shape [B, T, C]")
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(
                "expected input length {}, received {}".format(
                    self.seq_len, x_enc.shape[1]
                )
            )
        if x_enc.shape[2] != self.enc_in:
            raise ValueError(
                "expected {} input channels, received {}".format(
                    self.enc_in, x_enc.shape[2]
                )
            )
        normalized, means, stdev = self._normalise(x_enc)
        tokens = normalized.transpose(1, 2).contiguous()
        tokens = self.embedding_norm(self.value_embedding(tokens))
        encoded = self.encoder(tokens)
        return self.encoder_norm(encoded), means, stdev

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        encoded, means, stdev = self._encode(x_enc)
        # [B, C, H] -> [B, H, C].
        prediction = self.temporal_head(encoded).transpose(1, 2)
        prediction = self.channel_projection(prediction)

        # The custom dataset places the target at the final numeric column;
        # selecting the final c_out statistics handles S/M/MS consistently.
        target_means = means[:, :, -self.c_out:]
        target_stdev = stdev[:, :, -self.c_out:]
        prediction = prediction * target_stdev + target_means
        return prediction[:, -self.pred_len:, :]

    def get_embeddings(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """Expose variable-token representations for optional diagnostics."""

        encoded, _, _ = self._encode(x_enc)
        return encoded

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)


__all__ = ["Model"]
