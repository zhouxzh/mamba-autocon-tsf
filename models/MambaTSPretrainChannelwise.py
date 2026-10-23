"""Channelwise shared one-scan Mamba backbone for cross-domain pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class Model(nn.Module):
    """One shared Mamba module over independent channel patch sequences.

    The Mamba module is instantiated once and receives a flattened
    ``[batch * channels, patches, dimension]`` tensor.  Thus it preserves
    channel-specific temporal states without C-dependent parameters or a
    cross-channel recurrent scan.
    """

    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("MambaTSPretrainChannelwise requires mamba_ssm Mamba-1{}".format(detail))
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.patch_len = int(getattr(configs, "patch_len", 24))
        self.patch_stride = int(getattr(configs, "patch_stride", self.patch_len))
        self.d_model = int(getattr(configs, "d_model", 32))
        self.repr_dims = int(getattr(configs, "d_ff", self.d_model * 2))
        if min(self.seq_len, self.pred_len, self.patch_len, self.patch_stride, self.d_model, self.repr_dims) <= 0:
            raise ValueError("MambaTSPretrainChannelwise dimensions must be positive")
        if self.patch_len > self.seq_len:
            raise ValueError("patch_len cannot exceed seq_len")
        self.patch_count = 1 + (self.seq_len - self.patch_len) // self.patch_stride
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_count, self.d_model))
        self.encoder_norm = nn.LayerNorm(self.d_model)
        self.mamba = _Mamba1(
            d_model=self.d_model,
            d_state=int(getattr(configs, "mamba_d_state", 16)),
            d_conv=int(getattr(configs, "mamba_d_conv", 4)),
            expand=int(getattr(configs, "mamba_expand", 2)),
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.repr_projection = nn.Linear(self.d_model, self.repr_dims)
        self.repr_norm = nn.LayerNorm(self.repr_dims)
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)
        self.forecast_head = nn.Linear(self.patch_count * self.d_model, self.pred_len)
        self.reconstruction_head = nn.Linear(self.d_model, self.patch_len)
        self.register_buffer("representation_interpolation_matrix", self._interpolation_matrix(), persistent=False)

    def _interpolation_matrix(self):
        positions = ((torch.arange(self.seq_len, dtype=torch.float32) + 0.5)
                     * float(self.patch_count) / float(self.seq_len) - 0.5)
        left = positions.floor().long().clamp(0, self.patch_count - 1)
        right = (left + 1).clamp(0, self.patch_count - 1)
        weight = (positions - left.float()).clamp(0.0, 1.0)
        matrix = torch.zeros(self.seq_len, self.patch_count, dtype=torch.float32)
        matrix.scatter_add_(1, left[:, None], (1.0 - weight)[:, None])
        matrix.scatter_add_(1, right[:, None], weight[:, None])
        return matrix

    @staticmethod
    def _normalize(x):
        mean = x.mean(dim=1, keepdim=True).detach()
        centered = x - mean
        scale = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return centered / scale, mean, scale

    def encode(self, x, patch_mask=None):
        normalized, mean, scale = self._normalize(x)
        patches = normalized.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        tokens = self.patch_embedding(patches)
        batch, channels, patches_count, _ = tokens.shape
        if patch_mask is not None:
            if patch_mask.shape != (batch, self.patch_count):
                raise ValueError("patch_mask must have shape [batch, patch_count]")
            tokens = torch.where(
                patch_mask[:, None, :, None],
                self.mask_token[:, None, :, :],
                tokens,
            )
        sequence = self.encoder_norm(tokens + self.position_embedding).reshape(batch * channels, patches_count, self.d_model)
        try:
            sequence = self.mamba(sequence)
        except RuntimeError as exc:
            if not sequence.is_cuda and any(token in str(exc).lower() for token in ("cuda", "triton")):
                raise RuntimeError("MambaTSPretrainChannelwise requires a CUDA-compatible mamba_ssm build") from exc
            raise
        states = self.final_norm(sequence).reshape(batch, channels, patches_count, self.d_model)
        return states, normalized, mean, scale, patches

    def representation(self, states):
        temporal = states.mean(dim=1).transpose(1, 2)
        matrix = self.representation_interpolation_matrix.to(temporal.device, temporal.dtype)
        grid = torch.matmul(temporal, matrix.transpose(0, 1)).transpose(1, 2)
        return self.repr_head(self.repr_norm(self.repr_projection(grid)))

    def forecast(self, states, mean, scale):
        batch, channels = states.shape[:2]
        normalized_prediction = self.forecast_head(states.reshape(batch * channels, -1)).reshape(batch, channels, self.pred_len).transpose(1, 2)
        return normalized_prediction * scale + mean

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None, pretrain=False, mask=None):
        if pretrain:
            if mask is None:
                mask = torch.rand(x_enc.shape[0], self.patch_count, device=x_enc.device) < 0.25
            states, normalized, _, _, _ = self.encode(x_enc, patch_mask=mask)
            representation = self.representation(states)
            target = normalized.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
            return {
                "representation": representation,
                "reconstruction": self.reconstruction_head(states),
                "reconstruction_target": target,
                "mask": mask.to(device=x_enc.device, dtype=torch.bool),
                "states": states,
            }
        states, normalized, mean, scale, _ = self.encode(x_enc)
        representation = self.representation(states)
        prediction = self.forecast(states, mean, scale)
        if onlyrepr:
            return None, representation
        if return_repr:
            return prediction, representation
        return prediction


__all__ = ["Model"]
