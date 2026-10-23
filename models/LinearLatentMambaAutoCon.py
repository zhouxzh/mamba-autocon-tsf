"""NLinear base plus a low-rank latent Mamba residual with AutoCon.

The linear base preserves the strong persistence/extrapolation inductive bias
of NLinear.  Mamba is restricted to a small cross-variable latent space and a
short pooled temporal sequence; the same latent states produce the AutoCon
representation and a gated forecast residual.  This is a local model, not a
claim of equivalence to an external Mamba forecasting implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - depends on local CUDA installation
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class Model(nn.Module):
    """NLinear forecast with a low-rank, short-token Mamba correction."""

    supports_return_repr_control = True
    temporal_stride = 4

    def __init__(self, configs):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("LinearLatentMamba requires mamba_ssm Mamba-1{}".format(detail))
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        self.AutoCon = bool(getattr(configs, "AutoCon", False))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "LastVal"))
        if min(self.seq_len, self.pred_len, self.enc_in, self.c_out) <= 0 or self.c_out > self.enc_in:
            raise ValueError("LinearLatentMamba requires positive dimensions and c_out <= enc_in")
        if self.seq_len % self.temporal_stride:
            raise ValueError("LinearLatentMamba requires seq_len divisible by 4")

        requested_rank = int(getattr(configs, "mamba_temporal_rank", 8))
        requested_model = int(getattr(configs, "d_model", 32))
        self.latent_dim = max(1, min(requested_rank, requested_model, self.enc_in))
        self.d_ff = max(1, int(getattr(configs, "d_ff", self.latent_dim * 2)))
        d_state = max(1, min(4, int(getattr(configs, "mamba_d_state", 16))))
        d_conv = max(1, min(2, int(getattr(configs, "mamba_d_conv", 4))))
        expand = 1

        # NLinear-style persistence base.  The weight initialization is the
        # historical uniform extrapolator; optimization remains unconstrained.
        self.base_head = nn.Linear(self.seq_len, self.pred_len)
        nn.init.constant_(self.base_head.weight, 1.0 / self.seq_len)
        nn.init.zeros_(self.base_head.bias)

        self.variable_compress = nn.Linear(self.enc_in, self.latent_dim, bias=False)
        self.token_norm = nn.LayerNorm(self.latent_dim)
        self.mamba = _Mamba1(
            d_model=self.latent_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.state_norm = nn.LayerNorm(self.latent_dim)
        self.horizon_decode = nn.Linear(self.latent_dim, self.pred_len, bias=False)
        self.variable_decode = nn.Linear(self.latent_dim, self.c_out, bias=False)
        residual_init = float(getattr(configs, "mamba_ts_residual_init", 0.0))
        if not -0.99 < residual_init < 0.99:
            raise ValueError("mamba_ts_residual_init must be in (-0.99, 0.99)")
        self.forecast_scale = nn.Parameter(torch.tensor(residual_init))

        if self.AutoCon:
            self.repr_projection = nn.Linear(self.latent_dim, self.d_ff)
            self.repr_norm = nn.LayerNorm(self.d_ff)
            self.repr_head = nn.Linear(self.d_ff, self.d_ff)
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
        if mode == "lastval":
            last = x[:, -1:, :].detach()
            return x - last, last
        if mode == "mean":
            mean = x.mean(dim=1, keepdim=True).detach()
            return x - mean, mean
        raise ValueError("LinearLatentMamba supports LastVal, MeanStd, ReVIN, or Mean")

    def _restore(self, prediction, restore):
        if isinstance(restore, tuple):
            mean, scale = restore
            return prediction * scale[:, :, -self.c_out:] + mean[:, :, -self.c_out:]
        return prediction + restore[:, :, -self.c_out:]

    def _states(self, normalized):
        tokens = self.variable_compress(normalized)
        tokens = F.avg_pool1d(
            tokens.transpose(1, 2), kernel_size=self.temporal_stride,
            stride=self.temporal_stride,
        ).transpose(1, 2)
        try:
            mixed = self.mamba(self.token_norm(tokens))
        except RuntimeError as exc:
            if not tokens.is_cuda and ("cuda" in str(exc).lower() or "triton" in str(exc).lower()):
                raise RuntimeError("LinearLatentMamba requires a CUDA-compatible mamba_ssm build") from exc
            raise
        return self.state_norm(mixed + tokens)

    def _base_forecast(self, normalized):
        # Centering by the last value is exactly the NLinear forecast contract.
        return self.base_head(normalized.transpose(1, 2)).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None):
        normalized, restore = self._normalize(x_enc)
        states = self._states(normalized)
        summary = states.mean(dim=1)
        residual = self.horizon_decode(summary).unsqueeze(-1)
        residual = residual * self.variable_decode(summary).unsqueeze(1)
        prediction = self._base_forecast(normalized)[:, :, -self.c_out:]
        prediction = prediction + torch.tanh(self.forecast_scale) * residual

        need_repr = bool(onlyrepr or (self.AutoCon if return_repr is None else bool(return_repr)))
        representation = None
        if need_repr:
            # AutoCon reads the same latent Mamba states used by the forecast
            # residual; the contrastive gradient therefore reaches the shared
            # encoder without adding a second prediction-only backbone.
            grid = F.interpolate(
                states.transpose(1, 2), size=self.seq_len,
                mode="linear", align_corners=False,
            ).transpose(1, 2)
            if self.AutoCon:
                representation = self.repr_head(
                    self.repr_norm(self.repr_projection(grid))
                )
            else:
                representation = grid
        prediction = self._restore(prediction, restore)
        if onlyrepr:
            return None, representation
        if self.AutoCon or return_repr:
            return prediction, representation
        return prediction


__all__ = ["Model"]
