"""Channel-count agnostic temporal Mamba backbone for AutoCon pretraining.

The encoder pools channel tokens before one temporal Mamba scan.  Model
parameters therefore do not depend on the number or identity of sensors and
the same checkpoint can be transferred across multivariate datasets.  The
pretraining-only reconstruction head is never used by ``forecast`` inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - CUDA extension availability varies by environment
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class Model(nn.Module):
    """Shared patch encoder with one temporal Mamba scan.

    ``forward`` is deliberately prediction-only by default.  Set
    ``pretrain=True`` to obtain a masked-patch reconstruction and the temporal
    representation consumed by the external AutoCon relation objective.
    """

    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("MambaTSPretrain requires mamba_ssm Mamba-1{}".format(detail))
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.patch_len = int(getattr(configs, "patch_len", 24))
        self.patch_stride = int(getattr(configs, "patch_stride", self.patch_len))
        self.d_model = int(getattr(configs, "d_model", 32))
        self.repr_dims = int(getattr(configs, "d_ff", self.d_model * 2))
        self.dropout_p = float(getattr(configs, "dropout", 0.1))
        if min(self.seq_len, self.pred_len, self.patch_len, self.patch_stride, self.d_model, self.repr_dims) <= 0:
            raise ValueError("MambaTSPretrain dimensions must be positive")
        if self.patch_len > self.seq_len:
            raise ValueError("patch_len cannot exceed seq_len")
        self.patch_count = 1 + (self.seq_len - self.patch_len) // self.patch_stride

        d_state = int(getattr(configs, "mamba_d_state", 16))
        d_conv = int(getattr(configs, "mamba_d_conv", 4))
        expand = int(getattr(configs, "mamba_expand", 2))
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_count, self.d_model))
        self.encoder_norm = nn.LayerNorm(self.d_model)
        self.mamba = _Mamba1(
            d_model=self.d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.repr_projection = nn.Linear(self.d_model, self.repr_dims)
        self.repr_norm = nn.LayerNorm(self.repr_dims)
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # Shared forecast decoder.  Channel-specific parameters are avoided:
        # each channel contributes only three observed statistics to a shared
        # amplitude head, while the temporal residual is common across sensors.
        self.forecast_head = nn.Linear(self.patch_count * self.d_model, self.pred_len)
        self.channel_head = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.forecast_scale = nn.Parameter(torch.zeros(()))

        # The reconstruction head is used only by the self-supervised
        # pretraining objective and adds no inference-time operation.
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
        """Return shared temporal states and per-channel patch states."""
        normalized, mean, scale = self._normalize(x)
        patches = normalized.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        per_channel = self.patch_embedding(patches)
        if patch_mask is not None:
            if patch_mask.shape != (per_channel.shape[0], self.patch_count):
                raise ValueError("patch_mask must have shape [batch, patch_count]")
            per_channel = torch.where(
                patch_mask[:, None, :, None],
                self.mask_token[:, None, :, :],
                per_channel,
            )
        temporal = per_channel.mean(dim=1) + self.position_embedding
        temporal = self.encoder_norm(temporal)
        try:
            temporal = self.mamba(temporal)
        except RuntimeError as exc:
            if not temporal.is_cuda and any(token in str(exc).lower() for token in ("cuda", "triton")):
                raise RuntimeError("MambaTSPretrain requires a CUDA-compatible mamba_ssm build") from exc
            raise
        temporal = self.final_norm(temporal)
        return temporal, per_channel, normalized, mean, scale

    def representation(self, states):
        matrix = self.representation_interpolation_matrix.to(states.device, states.dtype)
        grid = torch.matmul(states.transpose(1, 2), matrix.transpose(0, 1)).transpose(1, 2)
        return self.repr_head(self.repr_norm(self.repr_projection(grid)))

    def forecast(self, x, states=None, normalized=None, mean=None, scale=None):
        if states is None:
            states, _, normalized, mean, scale = self.encode(x)
        else:
            if normalized is None or mean is None or scale is None:
                normalized, mean, scale = self._normalize(x)
        temporal = self.forecast_head(states.reshape(states.shape[0], -1))
        stats = torch.stack(
            (normalized.mean(dim=1), normalized.std(dim=1, unbiased=False), normalized[:, -1]),
            dim=-1,
        )
        amplitude = self.channel_head(stats).squeeze(-1)
        residual = temporal.unsqueeze(-1) * amplitude.unsqueeze(1)
        base = normalized[:, -1:, :].expand(-1, self.pred_len, -1)
        residual_scale = torch.tanh(self.forecast_scale).to(dtype=x.dtype, device=x.device)
        normalized_prediction = base + residual_scale * residual
        return normalized_prediction * scale + mean

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None, pretrain=False, mask=None):
        if pretrain:
            if mask is None:
                mask = torch.rand(x_enc.shape[0], self.patch_count, device=x_enc.device) < 0.25
            states, masked_channel_patches, normalized, _, _ = self.encode(x_enc, patch_mask=mask)
            representation = self.representation(states)
            mask = mask.to(device=x_enc.device, dtype=torch.bool)
            reconstruction = self.reconstruction_head(states)
            target = normalized.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
            return {
                "representation": representation,
                "reconstruction": reconstruction,
                "reconstruction_target": target,
                "mask": mask,
                "states": states,
                "per_channel_patch": masked_channel_patches,
            }
        states, per_channel, normalized, mean, scale = self.encode(x_enc)
        representation = self.representation(states)
        prediction = self.forecast(
            x_enc, states=states, normalized=normalized, mean=mean, scale=scale
        )
        if onlyrepr:
            return None, representation
        if return_repr:
            return prediction, representation
        return prediction


__all__ = ["Model"]
