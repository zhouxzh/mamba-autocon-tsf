"""RLinear baseline: reversible instance normalization plus linear mapping.

This is an independent implementation of the RLinear design described in
"Revisiting Long-term Time Series Forecasting: An Investigation on Linear
Mapping". It contains no AutoCon path because it is a robust linear
comparator for the Mamba + AutoCon candidate.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Per-window RevIN followed by a shared or channel-specific time map."""

    def __init__(self, configs, individual: bool = False):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.pred_len = (
            self.seq_len
            if self.task_name in {"classification", "anomaly_detection", "imputation"}
            else int(configs.pred_len)
        )
        self.channels = int(configs.enc_in)
        self.individual = bool(individual or getattr(configs, "individual", False))
        self.eps = 1e-5

        # RevIN's affine parameters are channel-specific and inverted before
        # restoring each input window's observed mean and standard deviation.
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.channels))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.channels))
        if self.individual:
            self.linear = nn.ModuleList(
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            )
        else:
            self.linear = nn.Linear(self.seq_len, self.pred_len)

    def _normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x - mean, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        normalized = (x - mean) / stdev
        return normalized * self.affine_weight + self.affine_bias, mean, stdev

    def _denormalize(self, x: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
        restored = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return restored * stdev + mean

    def _project(self, normalized: torch.Tensor) -> torch.Tensor:
        tokens = normalized.transpose(1, 2)
        if self.individual:
            projected = torch.stack(
                [layer(tokens[:, index, :]) for index, layer in enumerate(self.linear)],
                dim=1,
            )
        else:
            projected = self.linear(tokens)
        return projected.transpose(1, 2)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        normalized, mean, stdev = self._normalize(x_enc)
        return self._denormalize(self._project(normalized), mean, stdev)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc)[:, -self.pred_len :, :]
        raise NotImplementedError("RLinear is registered only for forecasting tasks")
