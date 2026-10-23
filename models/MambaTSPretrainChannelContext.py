"""One-scan channelwise Mamba with channel-count agnostic patch context."""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # pragma: no cover
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class Model(nn.Module):
    """Shared channel Mamba with a DeepSets mean/std patch context adapter."""

    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("MambaTSPretrainChannelContext requires mamba_ssm Mamba-1{}".format(detail))
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.patch_len = int(getattr(configs, "patch_len", 24))
        self.patch_stride = int(getattr(configs, "patch_stride", self.patch_len))
        self.d_model = int(getattr(configs, "d_model", 32))
        self.repr_dims = int(getattr(configs, "d_ff", self.d_model * 2))
        if min(self.seq_len, self.pred_len, self.patch_len, self.patch_stride, self.d_model, self.repr_dims) <= 0:
            raise ValueError("MambaTSPretrainChannelContext dimensions must be positive")
        if self.patch_len > self.seq_len:
            raise ValueError("patch_len cannot exceed seq_len")
        self.patch_count = 1 + (self.seq_len - self.patch_len) // self.patch_stride
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_count, self.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.context_projection = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model, bias=False), nn.GELU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
        )
        # The context gate starts at zero, preserving the channelwise control
        # at initialization.  Its projection must be nonzero, otherwise the
        # product of a zero gate and a zero projection has no gradient and the
        # advertised context branch is permanently dead.
        nn.init.xavier_uniform_(self.context_projection[-1].weight)
        self.context_scale = nn.Parameter(torch.zeros(()))
        self.domain_projection = nn.Sequential(
            nn.Linear(2, self.d_model, bias=False), nn.GELU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
        )
        nn.init.xavier_uniform_(self.domain_projection[-1].weight)
        self.domain_scale = nn.Parameter(torch.zeros(()))
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
        self.next_patch_head = nn.Linear(self.d_model, self.patch_len)
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

    def encode(self, x, patch_mask=None, data_interval_minutes=None):
        normalized, mean, scale = self._normalize(x)
        patches = normalized.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
        tokens = self.patch_embedding(patches)
        batch, channels, patch_count, _ = tokens.shape
        if patch_mask is not None:
            if patch_mask.shape != (batch, self.patch_count):
                raise ValueError("patch_mask must have shape [batch, patch_count]")
            tokens = torch.where(patch_mask[:, None, :, None], self.mask_token[:, None, :, :], tokens)
        base = tokens + self.position_embedding
        summary = torch.cat((base.mean(dim=1), base.std(dim=1, unbiased=False)), dim=-1)
        context = self.context_projection(summary)
        tokens = base + torch.tanh(self.context_scale).to(tokens.dtype) * context[:, None, :, :]
        if data_interval_minutes is not None:
            interval = torch.as_tensor(data_interval_minutes, device=x.device, dtype=x.dtype).reshape(-1)
            if interval.numel() == 1:
                interval = interval.expand(batch)
            if interval.numel() != batch or torch.any(interval <= 0):
                raise ValueError("data_interval_minutes must be positive scalar or [batch]")
            descriptor = torch.stack((torch.log1p(interval), torch.log1p(interval * self.seq_len)), dim=-1)
            domain = self.domain_projection(descriptor)
            tokens = tokens + torch.tanh(self.domain_scale).to(tokens.dtype) * domain[:, None, None, :]
        sequence = self.encoder_norm(tokens).reshape(batch * channels, patch_count, self.d_model)
        try:
            sequence = self.mamba(sequence)
        except RuntimeError as exc:
            if not sequence.is_cuda and any(token in str(exc).lower() for token in ("cuda", "triton")):
                raise RuntimeError("MambaTSPretrainChannelContext requires a CUDA-compatible mamba_ssm build") from exc
            raise
        states = self.final_norm(sequence).reshape(batch, channels, patch_count, self.d_model)
        return states, normalized, mean, scale

    def representation(self, states):
        temporal = states.mean(dim=1).transpose(1, 2)
        matrix = self.representation_interpolation_matrix.to(temporal.device, temporal.dtype)
        grid = torch.matmul(temporal, matrix.transpose(0, 1)).transpose(1, 2)
        return self.repr_head(self.repr_norm(self.repr_projection(grid)))

    def predict_next_patch(self, states):
        """Predict one normalized future patch from each channel's final state."""
        return self.next_patch_head(states[:, :, -1])

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None, pretrain=False, mask=None,
                data_interval_minutes=None):
        if pretrain:
            if mask is None:
                mask = torch.rand(x_enc.shape[0], self.patch_count, device=x_enc.device) < 0.25
            states, normalized, _, _ = self.encode(x_enc, patch_mask=mask, data_interval_minutes=data_interval_minutes)
            target = normalized.transpose(1, 2).unfold(-1, self.patch_len, self.patch_stride)
            return {"representation": self.representation(states), "reconstruction": self.reconstruction_head(states), "reconstruction_target": target, "mask": mask.to(device=x_enc.device, dtype=torch.bool), "states": states}
        states, _, mean, scale = self.encode(x_enc, data_interval_minutes=data_interval_minutes)
        batch, channels = states.shape[:2]
        normalized_prediction = self.forecast_head(states.reshape(batch * channels, -1)).reshape(batch, channels, self.pred_len).transpose(1, 2)
        prediction = normalized_prediction * scale + mean
        representation = self.representation(states)
        if onlyrepr:
            return None, representation
        if return_repr:
            return prediction, representation
        return prediction


__all__ = ["Model"]
