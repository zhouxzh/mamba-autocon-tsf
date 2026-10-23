"""S-Mamba/TimesMamba-inspired variable-token forecaster with AutoCon.

This is a local, mechanism-level implementation, not a reproduction of an
external checkpoint.  Each variable's observed history becomes one token; a
forward/reverse Mamba pair models inter-variable dependencies, then a shared
FFN and direct horizon head produce forecasts.  The optional AutoCon branch
keeps a genuine time-indexed representation by projecting every history step
and conditioning it on the variable-token Mamba state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # pragma: no cover - CUDA extension availability is environment-specific
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class _ResidualMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("MambaSM requires mamba_ssm Mamba-1{}".format(detail))
        self.norm = nn.LayerNorm(int(d_model))
        self.mixer = _Mamba1(
            d_model=int(d_model), d_state=int(d_state), d_conv=int(d_conv),
            expand=int(expand),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x):
        try:
            mixed = self.mixer(self.norm(x))
        except RuntimeError as exc:
            if not x.is_cuda and ("cuda" in str(exc).lower() or "triton" in str(exc).lower()):
                raise RuntimeError("MambaSM requires a CUDA-compatible mamba_ssm build") from exc
            raise
        return x + self.dropout(mixed)


class _BiVariableBlock(nn.Module):
    """Bidirectional variable-token Mamba followed by a shared FFN."""

    def __init__(self, d_model, d_state, d_conv, expand, d_ff, dropout):
        super().__init__()
        self.forward_mamba = _ResidualMamba(d_model, d_state, d_conv, expand, dropout)
        self.reverse_mamba = _ResidualMamba(d_model, d_state, d_conv, expand, dropout)
        self.norm = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), int(d_model)), nn.Dropout(float(dropout)),
        )

    def forward(self, tokens):
        forward = self.forward_mamba(tokens)
        reverse = torch.flip(self.reverse_mamba(torch.flip(tokens, dims=(1,))), dims=(1,))
        tokens = tokens + 0.5 * ((forward - tokens) + (reverse - tokens))
        return tokens + self.ffn(self.norm(tokens))


class Model(nn.Module):
    """Variable-token bidirectional Mamba with optional AutoCon output."""

    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        self.d_model = int(getattr(configs, "d_model", 32))
        self.d_ff = int(getattr(configs, "d_ff", 64))
        self.depth = max(1, int(getattr(configs, "e_layers", 1)))
        self.AutoCon = bool(getattr(configs, "AutoCon", False))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "MeanStd"))
        dropout = float(getattr(configs, "dropout", 0.1))
        if min(self.seq_len, self.pred_len, self.enc_in, self.c_out, self.d_model, self.d_ff) <= 0:
            raise ValueError("MambaSM dimensions must be positive")
        if self.c_out > self.enc_in:
            raise ValueError("MambaSM requires c_out <= enc_in")
        d_state = int(getattr(configs, "mamba_d_state", 16))
        d_conv = int(getattr(configs, "mamba_d_conv", 4))
        expand = int(getattr(configs, "mamba_expand", 2))
        if min(d_state, d_conv, expand) <= 0:
            raise ValueError("MambaSM Mamba dimensions must be positive")

        self.variable_embedding = nn.Linear(self.seq_len, self.d_model)
        self.blocks = nn.ModuleList([
            _BiVariableBlock(self.d_model, d_state, d_conv, expand, self.d_ff, dropout)
            for _ in range(self.depth)
        ])
        self.output_norm = nn.LayerNorm(self.d_model)
        self.forecast_head = nn.Linear(self.d_model, self.pred_len)
        if self.AutoCon:
            self.time_embedding = nn.Linear(self.enc_in, self.d_model)
            self.repr_condition = nn.Linear(self.d_model, self.d_model, bias=False)
            self.repr_norm = nn.LayerNorm(self.d_model)
            self.repr_projection = nn.Linear(self.d_model, self.d_ff)
            self.repr_head = nn.Linear(self.d_ff, self.d_ff)
        else:
            self.time_embedding = None
            self.repr_condition = None
            self.repr_norm = None
            self.repr_projection = None
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
        raise ValueError("MambaSM supports MeanStd, ReVIN, Mean, or LastVal normalization")

    def _restore(self, prediction, restore):
        if isinstance(restore, tuple):
            mean, scale = restore
            return prediction * scale[:, :, -self.c_out:] + mean[:, :, -self.c_out:]
        return prediction + restore[:, :, -self.c_out:]

    def _encode(self, normalized):
        tokens = self.variable_embedding(normalized.transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)

    def _representation(self, normalized, tokens):
        if not self.AutoCon:
            raise RuntimeError("plain MambaSM does not expose an AutoCon representation")
        time_tokens = self.time_embedding(normalized)
        condition = self.repr_condition(tokens.mean(dim=1)).unsqueeze(1)
        return self.repr_head(self.repr_projection(self.repr_norm(time_tokens + condition)))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None):
        normalized, restore = self._normalize(x_enc)
        tokens = self._encode(normalized)
        prediction = self.forecast_head(tokens).transpose(1, 2)[:, :, -self.c_out:]
        prediction = self._restore(prediction, restore)
        need_repr = bool(onlyrepr or (self.AutoCon if return_repr is None else bool(return_repr)))
        representation = self._representation(normalized, tokens) if need_repr else None
        if onlyrepr:
            return None, representation
        if self.AutoCon:
            return prediction, representation
        return prediction


__all__ = ["Model"]
