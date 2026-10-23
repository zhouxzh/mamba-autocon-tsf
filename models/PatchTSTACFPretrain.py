"""Channel-count-agnostic PatchTST encoder for ACF representation pretraining."""

from __future__ import annotations

import torch
from torch import nn


class Model(nn.Module):
    """A shared channel-independent PatchTST forecasting encoder.

    The Transformer is applied to each channel's patch trajectory with shared
    weights. No parameter has a channel dimension, so source and target data
    with different variable counts use the same encoder and prediction head.
    """

    def __init__(self, seq_len=96, pred_len=96, patch_len=24, patch_stride=24,
                 d_model=32, d_ff=64, n_heads=4, e_layers=1, dropout=.1):
        super().__init__()
        if seq_len < patch_len or (seq_len - patch_len) % patch_stride:
            raise ValueError("PatchTSTACF requires full-coverage patch geometry")
        if d_model % n_heads:
            raise ValueError("d_model must divide n_heads")
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.patch_len, self.patch_stride = int(patch_len), int(patch_stride)
        self.patch_count = (self.seq_len - self.patch_len) // self.patch_stride + 1
        self.d_model = int(d_model)
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.patch_position = nn.Parameter(torch.zeros(1, 1, self.patch_count, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=int(n_heads), dim_feedforward=int(d_ff),
            dropout=float(dropout), batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(e_layers), norm=nn.LayerNorm(self.d_model))
        self.prediction_head = nn.Linear(self.patch_count * self.d_model, self.pred_len)

    def _normalize(self, x):
        mean = x.mean(dim=1, keepdim=True)
        scale = torch.sqrt((x - mean).square().mean(dim=1, keepdim=True) + 1e-5)
        return (x - mean) / scale, (mean, scale)

    def encode_normalized(self, normalized):
        batch, _, channels = normalized.shape
        patches = normalized.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        tokens = self.patch_embedding(patches) + self.patch_position
        encoded = self.encoder(tokens.reshape(batch * channels, self.patch_count, self.d_model))
        return encoded.reshape(batch, channels, self.patch_count, self.d_model)

    def normalized_forecast(self, normalized):
        encoded = self.encode_normalized(normalized)
        residual = self.prediction_head(encoded.flatten(start_dim=2))
        forecast = residual.transpose(1, 2) + normalized[:, -1:, :]
        return forecast, encoded

    def forward(self, x, return_encoded=False):
        normalized, (mean, scale) = self._normalize(x)
        forecast, encoded = self.normalized_forecast(normalized)
        restored = forecast * scale + mean
        return (restored, encoded) if return_encoded else restored
