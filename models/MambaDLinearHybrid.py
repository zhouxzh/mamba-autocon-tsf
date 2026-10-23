"""DLinear-primary one-scan MambaTSv3 residual forecaster."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """DLinear is the exact zero-gated low-data control."""

    def __init__(self, mamba, linear, linear_only: bool = False,
                 mamba_correction_output: bool = False,
                 channelwise_residual_scale: bool = False,
                 centered_channelwise_residual_scale: bool = False,
                 channelwise_horizon_residual_scale: bool = False,
                 channelwise_residual_bias: bool = False,
                 shared_channel_statistic_residual_gate: bool = False,
                 shared_channel_statistic_gate_rank: int = 8):
        super().__init__()
        self.mamba = mamba
        self.linear = linear
        self.linear_only = bool(linear_only)
        self.mamba_correction_output = bool(mamba_correction_output)
        self.channelwise_residual_scale = bool(channelwise_residual_scale)
        self.centered_channelwise_residual_scale = bool(centered_channelwise_residual_scale)
        self.channelwise_horizon_residual_scale = bool(channelwise_horizon_residual_scale)
        self.channelwise_residual_bias = bool(channelwise_residual_bias)
        self.shared_channel_statistic_residual_gate = bool(shared_channel_statistic_residual_gate)
        if self.channelwise_horizon_residual_scale and not self.channelwise_residual_scale:
            raise ValueError("channelwise_horizon_residual_scale requires channelwise_residual_scale")
        if self.centered_channelwise_residual_scale and not self.channelwise_residual_scale:
            raise ValueError("centered_channelwise_residual_scale requires channelwise_residual_scale")
        if self.channelwise_residual_bias and not self.channelwise_residual_scale:
            raise ValueError("channelwise_residual_bias requires channelwise_residual_scale")
        if self.shared_channel_statistic_residual_gate and self.channelwise_residual_scale:
            raise ValueError("shared_channel_statistic_residual_gate is an alternative to channelwise_residual_scale")
        if self.shared_channel_statistic_residual_gate and int(shared_channel_statistic_gate_rank) <= 0:
            raise ValueError("shared_channel_statistic_gate_rank must be positive")
        # This target-only gate leaves the shared transferred core unchanged.
        # Scalar remains the historical default for checkpoint compatibility.
        shape = (
            (1, int(mamba.pred_len), int(mamba.c_out))
            if self.channelwise_horizon_residual_scale
            else (1, 1, int(mamba.c_out)) if self.channelwise_residual_scale else ()
        )
        self.residual_scale = nn.Parameter(torch.zeros(shape))
        if self.centered_channelwise_residual_scale:
            self.global_residual_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("global_residual_scale", None)
        if self.channelwise_residual_bias:
            bias_shape = (1, int(mamba.pred_len), int(mamba.c_out))
            self.residual_bias = nn.Parameter(torch.zeros(bias_shape))
        else:
            self.register_parameter("residual_bias", None)
        if self.shared_channel_statistic_residual_gate:
            # The gate has no channel-indexed parameters.  It learns when a
            # channel's local level, scale, and trend make the one-scan Mamba
            # correction reliable, while zero initialization preserves the
            # exact D/NLinear primary path at step zero.
            self.channel_statistic_gate_norm = nn.LayerNorm(4)
            self.channel_statistic_gate_hidden = nn.Linear(4, int(shared_channel_statistic_gate_rank))
            self.channel_statistic_gate_output = nn.Linear(int(shared_channel_statistic_gate_rank), 1)
            nn.init.zeros_(self.channel_statistic_gate_output.weight)
            nn.init.zeros_(self.channel_statistic_gate_output.bias)
        else:
            self.channel_statistic_gate_norm = None
            self.channel_statistic_gate_hidden = None
            self.channel_statistic_gate_output = None

    def _shared_channel_statistic_gate(self, x):
        """Return a channel-count-agnostic ``[batch, 1, channels]`` gate."""
        channel_features = torch.stack((
            x[:, -1, :],
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x[:, -1, :] - x[:, 0, :],
        ), dim=-1)
        hidden = torch.nn.functional.gelu(self.channel_statistic_gate_hidden(
            self.channel_statistic_gate_norm(channel_features)
        ))
        return self.channel_statistic_gate_output(hidden).transpose(1, 2)

    def forward(self, x):
        linear_prediction = self.linear(x, None, None, None)
        if self.linear_only:
            return linear_prediction
        output = self.mamba(x, return_repr=False)
        mamba_prediction = output[0] if isinstance(output, tuple) else output
        correction = (
            mamba_prediction
            if self.mamba_correction_output
            else mamba_prediction - linear_prediction
        )
        residual_scale = self.residual_scale
        if self.centered_channelwise_residual_scale:
            residual_scale = (
                self.global_residual_scale
                + residual_scale - residual_scale.mean(dim=-1, keepdim=True)
            )
        if self.shared_channel_statistic_residual_gate:
            residual_scale = residual_scale + self._shared_channel_statistic_gate(x)
        prediction = linear_prediction + torch.tanh(residual_scale) * correction
        return prediction if self.residual_bias is None else prediction + self.residual_bias
