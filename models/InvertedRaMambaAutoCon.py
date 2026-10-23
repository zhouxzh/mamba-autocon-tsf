"""Inverted variable-token RaMamba-style forecaster with AutoCon support.

This is an efficiency-constrained local implementation inspired by public
variable-axis RaMamba/DMamba designs.  Each variable becomes one token after
an inverted temporal projection, and one forward Mamba-1 scan mixes those
tokens.  The trend branch is linear.  The bidirectional scan and larger FFN
from the reference family are intentionally omitted, so this module is not
claimed to be an equivalent reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp


try:  # pragma: no cover - depends on the local CUDA installation
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class _VariableMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("InvertedRaMamba requires mamba_ssm Mamba-1{}".format(detail))
        self.norm = nn.LayerNorm(int(d_model))
        self.mixer = _Mamba1(
            d_model=int(d_model), d_state=int(d_state), d_conv=int(d_conv),
            expand=int(expand),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) else nn.Identity()

    def forward(self, tokens):
        try:
            mixed = self.mixer(self.norm(tokens))
        except RuntimeError as exc:
            if not tokens.is_cuda and ("cuda" in str(exc).lower() or "triton" in str(exc).lower()):
                raise RuntimeError("InvertedRaMamba requires a CUDA-compatible mamba_ssm build") from exc
            raise
        return tokens + self.dropout(mixed)


class Model(nn.Module):
    """Inverted temporal embedding, one variable-axis Mamba scan, and trend head."""

    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        self.d_model = int(getattr(configs, "d_model", 32))
        self.d_ff = int(getattr(configs, "d_ff", 64))
        self.AutoCon = bool(getattr(configs, "AutoCon", False))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "MeanStd"))
        if min(self.seq_len, self.pred_len, self.enc_in, self.c_out, self.d_model, self.d_ff) <= 0:
            raise ValueError("InvertedRaMamba dimensions must be positive")
        if self.c_out > self.enc_in:
            raise ValueError("InvertedRaMamba requires c_out <= enc_in")

        d_state = int(getattr(configs, "mamba_d_state", 16))
        d_conv = int(getattr(configs, "mamba_d_conv", 4))
        expand = int(getattr(configs, "mamba_expand", 2))
        depth = max(1, int(getattr(configs, "e_layers", 1)))
        moving_avg = int(getattr(configs, "moving_avg", 25))
        if min(d_state, d_conv, expand, moving_avg) <= 0:
            raise ValueError("InvertedRaMamba Mamba/decomposition dimensions must be positive")

        self.decomposition = series_decomp(moving_avg)
        self.temporal_embedding = nn.Linear(self.seq_len, self.d_model)
        self.variable_blocks = nn.ModuleList([
            _VariableMamba(
                self.d_model, d_state, d_conv, expand,
                float(getattr(configs, "dropout", 0.1)),
            )
            for _ in range(depth)
        ])
        self.variable_norm = nn.LayerNorm(self.d_model)
        self.seasonal_head = nn.Linear(self.d_model, self.pred_len)
        self.trend_head = nn.Linear(self.seq_len, self.pred_len)

        if self.AutoCon:
            # Build a temporal [B, L, R] grid from the same variable-token
            # state used by the forecast head; gradients therefore reach the
            # forecast path even when the contrastive objective is auxiliary.
            self.time_embedding = nn.Linear(self.enc_in, self.d_model)
            self.repr_condition = nn.Linear(self.d_model, self.d_model, bias=False)
            self.repr_norm = nn.LayerNorm(self.d_model)
            self.repr_projection = nn.Linear(self.d_model, self.d_ff)
            self.repr_head = nn.Linear(self.d_ff, self.d_ff)
            self.repr_forecast = nn.Linear(self.d_ff, self.pred_len, bias=False)
            self.repr_forecast_scale = nn.Parameter(torch.zeros(()))
        else:
            self.time_embedding = None
            self.repr_condition = None
            self.repr_norm = None
            self.repr_projection = None
            self.repr_head = None
            self.repr_forecast = None
            self.register_parameter("repr_forecast_scale", None)

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
        raise ValueError("InvertedRaMamba supports MeanStd, ReVIN, Mean, or LastVal normalization")

    def _restore(self, prediction, restore):
        if isinstance(restore, tuple):
            mean, scale = restore
            return prediction * scale[:, :, -self.c_out:] + mean[:, :, -self.c_out:]
        return prediction + restore[:, :, -self.c_out:]

    def _variable_tokens(self, seasonal):
        tokens = self.temporal_embedding(seasonal.transpose(1, 2))
        for block in self.variable_blocks:
            tokens = block(tokens)
        return self.variable_norm(tokens)

    def _representation(self, seasonal, tokens):
        if not self.AutoCon:
            raise RuntimeError("plain InvertedRaMamba does not expose an AutoCon representation")
        time_tokens = self.time_embedding(seasonal)
        condition = self.repr_condition(tokens.mean(dim=1)).unsqueeze(1)
        return self.repr_head(self.repr_projection(self.repr_norm(time_tokens + condition)))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None):
        normalized, restore = self._normalize(x_enc)
        seasonal, trend = self.decomposition(normalized)
        tokens = self._variable_tokens(seasonal)
        seasonal_forecast = self.seasonal_head(tokens).transpose(1, 2)
        trend_forecast = self.trend_head(trend.transpose(1, 2)).transpose(1, 2)
        prediction = (seasonal_forecast + trend_forecast)[:, :, -self.c_out:]
        need_repr = bool(onlyrepr or (self.AutoCon if return_repr is None else bool(return_repr)))
        representation = self._representation(seasonal, tokens) if need_repr else None
        if self.AutoCon and representation is not None:
            correction = self.repr_forecast(representation.mean(dim=1)).unsqueeze(-1)
            correction = correction.expand(-1, -1, self.c_out)
            prediction = prediction + torch.tanh(self.repr_forecast_scale) * correction
        prediction = self._restore(prediction, restore)
        if onlyrepr:
            return None, representation
        if self.AutoCon:
            return prediction, representation
        return prediction


__all__ = ["Model"]
