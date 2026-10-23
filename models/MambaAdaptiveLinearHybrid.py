"""Adaptive DLinear/NLinear primary with one pretrained Mamba residual."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Blend linear inductive biases from trainable window statistics only."""

    def __init__(self, mamba, dlinear, nlinear, linear_only: bool = False):
        super().__init__()
        self.mamba = mamba
        self.dlinear = dlinear
        self.nlinear = nlinear
        self.linear_only = bool(linear_only)
        self.base_gate = nn.Sequential(nn.Linear(3, 8), nn.GELU(), nn.Linear(8, 1))
        # A neutral start is a 50/50 linear blend; the gate must learn from
        # target training windows rather than a validation-driven choice.
        nn.init.zeros_(self.base_gate[-1].weight)
        nn.init.zeros_(self.base_gate[-1].bias)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _features(x):
        centered = x - x.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1).sqrt().clamp_min(1e-6)
        delta = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(1, 2)) / scale.mean(dim=1)
        lag1 = (centered[:, 1:] * centered[:, :-1]).mean(dim=(1, 2)) / scale.square().mean(dim=1)
        curvature = (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]).abs().mean(dim=(1, 2)) / scale.mean(dim=1)
        return torch.stack((delta, lag1, curvature), dim=-1)

    def forward(self, x):
        dlinear = self.dlinear(x, None, None, None)
        nlinear = self.nlinear(x, None, None, None)
        gate = torch.sigmoid(self.base_gate(self._features(x))).view(-1, 1, 1)
        primary = gate * dlinear + (1.0 - gate) * nlinear
        if self.linear_only:
            return primary
        output = self.mamba(x, return_repr=False)
        mamba = output[0] if isinstance(output, tuple) else output
        return primary + torch.tanh(self.residual_scale) * (mamba - primary)
