"""Efficiency-constrained latent temporal Mamba forecaster with AutoCon.

The model keeps a DLinear-style full-channel forecast as its base, compresses
each observed time step into a small learned variable subspace, averages four
adjacent steps, and runs one causal Mamba-1 scan over that short temporal
sequence.  The decoded Mamba state is a ReZero forecast residual.  When
AutoCon is enabled, its time-indexed representation is constructed from the
same states and the same states directly affect the forecast, rather than
forming an auxiliary-only branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp

try:  # pragma: no cover - CUDA extension availability varies by environment
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class Model(nn.Module):
    """Short-token latent temporal Mamba with a full-channel linear base."""

    supports_return_repr_control = True
    temporal_stride = 4

    def __init__(self, configs):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("LatentTemporalMamba requires mamba_ssm Mamba-1{}".format(detail))
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        self.AutoCon = bool(getattr(configs, "AutoCon", False))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "MeanStd"))
        if min(self.seq_len, self.pred_len, self.enc_in, self.c_out) <= 0 or self.c_out > self.enc_in:
            raise ValueError("LatentTemporalMamba requires positive dimensions and c_out <= enc_in")
        if self.seq_len % self.temporal_stride:
            raise ValueError("LatentTemporalMamba requires seq_len divisible by {}".format(self.temporal_stride))

        d_model = int(getattr(configs, "d_model", 32))
        d_ff = int(getattr(configs, "d_ff", 64))
        self.latent_dim = min(16, d_model, self.enc_in)
        if min(d_model, d_ff, self.latent_dim) <= 0:
            raise ValueError("LatentTemporalMamba dimensions must be positive")
        d_state = min(8, int(getattr(configs, "mamba_d_state", 16)))
        d_conv = min(2, int(getattr(configs, "mamba_d_conv", 4)))
        expand = 1
        dropout = float(getattr(configs, "dropout", 0.1))

        self.decomposition = series_decomp(int(getattr(configs, "moving_avg", 25)))
        self.seasonal_head = nn.Linear(self.seq_len, self.pred_len)
        self.trend_head = nn.Linear(self.seq_len, self.pred_len)
        # These projections are the only variable-specific components; Mamba
        # itself runs on a 16-dimensional, 24-token temporal latent sequence.
        self.variable_compress = nn.Linear(self.enc_in, self.latent_dim, bias=False)
        self.token_norm = nn.LayerNorm(self.latent_dim)
        self.mamba = _Mamba1(
            d_model=self.latent_dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.state_norm = nn.LayerNorm(self.latent_dim)
        # A rank-one horizon-by-variable correction keeps the Mamba route in
        # the forecast while avoiding a dense [H,C] decoder on every step.
        self.horizon_decode = nn.Linear(self.latent_dim, self.pred_len, bias=False)
        self.variable_decode = nn.Linear(self.latent_dim, self.c_out, bias=False)
        self.forecast_scale = nn.Parameter(torch.zeros(()))

        if self.AutoCon:
            self.repr_projection = nn.Linear(self.latent_dim, d_ff)
            self.repr_norm = nn.LayerNorm(d_ff)
            self.repr_head = nn.Linear(d_ff, d_ff)
        else:
            self.repr_projection = None
            self.repr_norm = None
            self.repr_head = None

    def _normalize(self, x):
        mode = self.AutoCon_wnorm.lower()
        if mode in {"meanstd", "revin"}:
            mean = x.mean(dim=1, keepdim=True).detach()
            centered = x - mean
            scale = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            return centered / scale, (mean, scale)
        if mode == "mean":
            mean = x.mean(dim=1, keepdim=True).detach()
            return x - mean, mean
        if mode == "lastval":
            last = x[:, -1:, :].detach()
            return x - last, last
        raise ValueError("LatentTemporalMamba supports MeanStd, ReVIN, Mean, or LastVal normalization")

    def _restore(self, prediction, restore):
        if isinstance(restore, tuple):
            mean, scale = restore
            return prediction * scale[:, :, -self.c_out:] + mean[:, :, -self.c_out:]
        return prediction + restore[:, :, -self.c_out:]

    def _states(self, normalized):
        tokens = self.variable_compress(normalized)
        tokens = tokens.transpose(1, 2)
        tokens = F.avg_pool1d(tokens, kernel_size=self.temporal_stride, stride=self.temporal_stride)
        tokens = tokens.transpose(1, 2)
        try:
            states = self.mamba(self.token_norm(tokens))
        except RuntimeError as exc:
            if not tokens.is_cuda and ("cuda" in str(exc).lower() or "triton" in str(exc).lower()):
                raise RuntimeError("LatentTemporalMamba requires a CUDA-compatible mamba_ssm build") from exc
            raise
        return self.state_norm(states + tokens)

    def _base_forecast(self, normalized):
        seasonal, trend = self.decomposition(normalized)
        seasonal = self.seasonal_head(seasonal.transpose(1, 2)).transpose(1, 2)
        trend = self.trend_head(trend.transpose(1, 2)).transpose(1, 2)
        return seasonal + trend

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None):
        normalized, restore = self._normalize(x_enc)
        states = self._states(normalized)
        summary = states.mean(dim=1)
        correction = self.horizon_decode(summary).unsqueeze(-1) * self.variable_decode(summary).unsqueeze(1)
        prediction = self._base_forecast(normalized)[:, :, -self.c_out:]
        prediction = prediction + torch.tanh(self.forecast_scale) * correction
        prediction = self._restore(prediction, restore)
        need_repr = bool(onlyrepr or (self.AutoCon if return_repr is None else bool(return_repr)))
        representation = None
        if need_repr:
            grid = F.interpolate(
                states.transpose(1, 2), size=self.seq_len, mode="linear", align_corners=False
            ).transpose(1, 2)
            if self.AutoCon:
                representation = self.repr_head(self.repr_norm(self.repr_projection(grid)))
            else:
                representation = grid
        if onlyrepr:
            return None, representation
        if self.AutoCon or return_repr:
            return prediction, representation
        return prediction


__all__ = ["Model"]
