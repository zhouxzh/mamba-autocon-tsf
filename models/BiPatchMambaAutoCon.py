"""Bi-Mamba4TS-inspired patch Mamba forecaster with an AutoCon route.

This is a local implementation of the usable architectural ideas in
Bi-Mamba4TS, not a reproduction of its custom Mamba+ CUDA kernel.  It uses
the installed fused Mamba-1 operator in both directions over either patch
tokens within a channel or variable tokens within a patch.  The latter is the
channel-mixing route used for high-dimensional data.  Token-axis selection is
an explicit experiment setting; the original SRA decider depends on a
dataset-level correlation matrix and is deliberately not inferred from
validation/test samples here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


try:  # pragma: no cover - the CUDA extension is environment-dependent
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class _BiMambaBlock(nn.Module):
    """One native forward/reverse Mamba pair plus a small token MLP."""

    def __init__(self, d_model, d_state, d_conv, expand, d_ff, dropout):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("BiPatchMamba requires mamba_ssm Mamba-1{}".format(detail))
        kwargs = {
            "d_model": int(d_model), "d_state": int(d_state),
            "d_conv": int(d_conv), "expand": int(expand),
        }
        self.forward_norm = nn.LayerNorm(int(d_model))
        self.backward_norm = nn.LayerNorm(int(d_model))
        self.forward_mamba = _Mamba1(**kwargs)
        self.backward_mamba = _Mamba1(**kwargs)
        self.ffn_norm = nn.LayerNorm(int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), int(d_model)), nn.Dropout(float(dropout)),
        )

    def forward(self, tokens):
        try:
            forward = self.forward_mamba(self.forward_norm(tokens))
            backward = torch.flip(
                self.backward_mamba(torch.flip(self.backward_norm(tokens), dims=(1,))),
                dims=(1,),
            )
        except RuntimeError as exc:
            if not tokens.is_cuda and ("cuda" in str(exc).lower() or "triton" in str(exc).lower()):
                raise RuntimeError("BiPatchMamba requires a CUDA-compatible mamba_ssm build") from exc
            raise
        tokens = tokens + 0.5 * (forward + backward)
        return tokens + self.ffn(self.ffn_norm(tokens))


class Model(nn.Module):
    """Bidirectional patch/variable Mamba with optional AutoCon output."""

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
        self.repr_gate = bool(getattr(configs, "mamba_bipatch_repr_gate", False))
        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.patch_stride = int(getattr(configs, "patch_stride", 8))
        self.token_mode = str(getattr(configs, "mamba_bipatch_token_mode", "mixing")).strip().lower()
        if min(self.seq_len, self.pred_len, self.enc_in, self.c_out, self.d_model, self.d_ff, self.patch_len, self.patch_stride) <= 0:
            raise ValueError("BiPatchMamba dimensions must be positive")
        if self.patch_len > self.seq_len:
            raise ValueError("BiPatchMamba patch_len cannot exceed seq_len")
        if self.c_out > self.enc_in:
            raise ValueError("BiPatchMamba requires c_out <= enc_in")
        if self.token_mode not in {"mixing", "independent"}:
            raise ValueError("mamba_bipatch_token_mode must be mixing or independent")
        self.patch_count = 1 + (self.seq_len - self.patch_len) // self.patch_stride
        d_state = int(getattr(configs, "mamba_d_state", 16))
        d_conv = int(getattr(configs, "mamba_d_conv", 4))
        expand = int(getattr(configs, "mamba_expand", 2))
        if min(d_state, d_conv, expand) <= 0:
            raise ValueError("BiPatchMamba Mamba dimensions must be positive")
        dropout = float(getattr(configs, "dropout", 0.1))

        self.patch_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.patch_position = nn.Parameter(torch.zeros(1, 1, self.patch_count, self.d_model))
        self.channel_embedding = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.d_model))
        self.blocks = nn.ModuleList([
            _BiMambaBlock(self.d_model, d_state, d_conv, expand, self.d_ff, dropout)
            for _ in range(self.depth)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)
        self.forecast_head = nn.Linear(self.patch_count * self.d_model, self.pred_len)

        if self.AutoCon:
            # Both forecast correction and AutoCon consume this projection of
            # the very same channel-wise encoded patch states.
            self.repr_projection = nn.Linear(self.d_model, self.d_ff)
            self.repr_norm = nn.LayerNorm(self.d_ff)
            self.repr_head = nn.Linear(self.d_ff, self.d_ff)
            self.repr_forecast_time = nn.Linear(self.patch_count, self.pred_len, bias=False)
            self.repr_forecast_channel = nn.Linear(self.d_ff, 1, bias=False)
            self.repr_forecast_scale = nn.Parameter(torch.zeros(()))
        else:
            self.repr_projection = None
            self.repr_norm = None
            self.repr_head = None
            self.repr_forecast_time = None
            self.repr_forecast_channel = None
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
        raise ValueError("BiPatchMamba supports MeanStd, ReVIN, Mean, or LastVal normalization")

    def _restore(self, prediction, restore):
        if isinstance(restore, tuple):
            mean, scale = restore
            return prediction * scale[:, :, -self.c_out:] + mean[:, :, -self.c_out:]
        return prediction + restore[:, :, -self.c_out:]

    def _encode(self, normalized):
        patches = normalized.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        tokens = self.patch_embedding(patches) + self.patch_position + self.channel_embedding
        batch, channels, patches_count, width = tokens.shape
        if patches_count != self.patch_count:
            raise RuntimeError("BiPatchMamba patch count changed during forward")
        if self.token_mode == "mixing":
            sequence = tokens.transpose(1, 2).reshape(batch * patches_count, channels, width)
            for block in self.blocks:
                sequence = block(sequence)
            encoded = sequence.reshape(batch, patches_count, channels, width).transpose(1, 2)
        else:
            sequence = tokens.reshape(batch * channels, patches_count, width)
            for block in self.blocks:
                sequence = block(sequence)
            encoded = sequence.reshape(batch, channels, patches_count, width)
        return self.final_norm(encoded)

    def _representation(self, encoded):
        if not self.AutoCon:
            raise RuntimeError("plain BiPatchMamba does not expose an AutoCon representation")
        channel_repr = self.repr_head(self.repr_norm(self.repr_projection(encoded)))
        temporal = channel_repr.mean(dim=1)
        temporal = F.interpolate(temporal.transpose(1, 2), size=self.seq_len, mode="linear", align_corners=False).transpose(1, 2)
        return channel_repr, temporal

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None):
        normalized, restore = self._normalize(x_enc)
        encoded = self._encode(normalized)
        prediction = self.forecast_head(encoded.reshape(encoded.shape[0], self.enc_in, -1)).transpose(1, 2)
        need_repr = bool(onlyrepr or (self.AutoCon if return_repr is None else bool(return_repr)))
        representation = None
        if self.AutoCon and need_repr:
            channel_repr, representation = self._representation(encoded)
            correction = self.repr_forecast_channel(channel_repr).squeeze(-1)
            correction = self.repr_forecast_time(correction).transpose(1, 2)
            scale = torch.tanh(self.repr_forecast_scale)
            if self.repr_gate:
                # Keep the contrastive representation on the forecast path,
                # but bound its initial and learned influence to a small
                # multiplicative modulation of the supervised forecast.
                prediction = prediction * (
                    1.0 + 0.1 * scale * torch.tanh(correction)
                )
            else:
                prediction = prediction + scale * correction
        prediction = self._restore(prediction[:, :, -self.c_out:], restore)
        if onlyrepr:
            return None, representation
        if self.AutoCon:
            return prediction, representation
        return prediction


__all__ = ["Model"]
