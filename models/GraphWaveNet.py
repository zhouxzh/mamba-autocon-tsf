"""Canonical-style Graph WaveNet baseline for graph traffic forecasting.

The implementation retains the material Graph WaveNet components: causal
dilated temporal gates, forward/reverse fixed graph diffusion, an adaptive
node-embedding support and multi-order graph convolution.  It uses this
repository's hashed edge-list contract and generic forecasting interface, so
it is a comparable local implementation rather than a claim of bit-for-bit
reproduction of the external repository.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalTemporalConv(nn.Module):
    """Dilated temporal convolution preserving the observed-window length."""

    def __init__(self, channels_in: int, channels_out: int, dilation: int):
        super().__init__()
        self.left_padding = int(dilation)
        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=(2, 1),
            dilation=(int(dilation), 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (0, 0, self.left_padding, 0)))


class _DiffusionGraphConv(nn.Module):
    """Order-two diffusion over fixed and learned Graph WaveNet supports."""

    def __init__(self, channels_in: int, channels_out: int, support_count: int, dropout: float):
        super().__init__()
        self.order = 2
        self.projection = nn.Conv2d(
            (1 + self.order * int(support_count)) * channels_in,
            channels_out,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, values: torch.Tensor, supports: tuple[torch.Tensor, ...]) -> torch.Tensor:
        states = [values]
        for support in supports:
            first = torch.einsum("bctn,nm->bctm", values, support)
            states.append(first)
            second = torch.einsum("bctn,nm->bctm", first, support)
            states.append(second)
        return self.dropout(self.projection(torch.cat(states, dim=1)))


class Model(nn.Module):
    """Graph WaveNet-style direct multi-horizon forecaster for traffic data."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = str(configs.task_name)
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.hidden = int(getattr(configs, "d_model", 24))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "LastVal"))
        if min(self.seq_len, self.pred_len, self.channels, self.hidden) <= 0:
            raise ValueError("GraphWaveNet dimensions must be positive")
        path = str(getattr(configs, "mamba_ts_graph_path", "") or "").strip()
        if not path:
            raise ValueError("GraphWaveNet requires mamba_ts_graph_path")
        forward, reverse = self._load_graph(
            path, getattr(configs, "mamba_ts_graph_expected_sha256", "")
        )
        self.register_buffer("graph_forward", forward, persistent=True)
        self.register_buffer("graph_reverse", reverse, persistent=True)

        dropout = float(getattr(configs, "dropout", 0.1))
        self.input_projection = nn.Conv2d(1, self.hidden, kernel_size=1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        # Two dilation stages preserve the WaveNet temporal mechanism without
        # inflating a compact H12 baseline into a capacity-only comparison.
        for dilation in (1, 2):
            self.filter_convs.append(_CausalTemporalConv(self.hidden, self.hidden, dilation))
            self.gate_convs.append(_CausalTemporalConv(self.hidden, self.hidden, dilation))
            self.residual_convs.append(nn.Conv2d(self.hidden, self.hidden, kernel_size=1))
            self.skip_convs.append(nn.Conv2d(self.hidden, self.hidden, kernel_size=1))
            self.graph_convs.append(_DiffusionGraphConv(self.hidden, self.hidden, 3, dropout))
        adaptive_rank = min(10, self.channels)
        self.nodevec_source = nn.Parameter(torch.randn(self.channels, adaptive_rank) * 0.1)
        self.nodevec_target = nn.Parameter(torch.randn(adaptive_rank, self.channels) * 0.1)
        self.end_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.hidden, 2 * self.hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2 * self.hidden, self.pred_len, kernel_size=1),
        )

    def _load_graph(self, path: str, expected_hash: str) -> tuple[torch.Tensor, torch.Tensor]:
        graph_path = Path(path).expanduser()
        if not graph_path.is_file():
            raise FileNotFoundError("GraphWaveNet graph contract is missing: {}".format(graph_path))
        expected_hash = str(expected_hash or "").strip().lower()
        if expected_hash:
            digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError("GraphWaveNet graph contract hash mismatch")
        with np.load(str(graph_path), allow_pickle=False) as graph:
            required = {"schema_version", "sensor_ids", "target", "source", "weight", "node_count"}
            missing = sorted(required.difference(graph.files))
            if missing:
                raise ValueError("GraphWaveNet graph contract is missing fields: {}".format(
                    ", ".join(missing)
                ))
            node_count = np.asarray(graph["node_count"])
            sensor_ids = np.asarray(graph["sensor_ids"])
            target = np.asarray(graph["target"])
            source = np.asarray(graph["source"])
            weight = np.asarray(graph["weight"], dtype=np.float32)
        if node_count.ndim != 0 or int(node_count.item()) != self.channels:
            raise ValueError("GraphWaveNet graph node count does not match enc_in")
        if sensor_ids.ndim != 1 or sensor_ids.size != self.channels:
            raise ValueError("GraphWaveNet graph sensor IDs do not match node count")
        if (
            target.ndim != 1 or source.ndim != 1 or weight.ndim != 1
            or target.size == 0 or target.size != source.size or target.size != weight.size
            or target.dtype.kind not in {"i", "u"} or source.dtype.kind not in {"i", "u"}
            or target.min() < 0 or source.min() < 0
            or target.max() >= self.channels or source.max() >= self.channels
            or not np.isfinite(weight).all() or np.any(weight < 0)
        ):
            raise ValueError("GraphWaveNet graph edge values are invalid")
        forward = np.zeros((self.channels, self.channels), dtype=np.float32)
        forward[source, target] = weight
        reverse = forward.T.copy()
        reverse_sum = reverse.sum(axis=1, keepdims=True)
        reverse /= np.maximum(reverse_sum, np.float32(1e-12))
        return torch.from_numpy(forward), torch.from_numpy(reverse)

    def _normalize(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if self.AutoCon_wnorm != "MeanStd":
            return x_enc, None
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - mean
        scale = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return centered / scale, (mean, scale)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        values, restore = self._normalize(x_enc)
        # [B, time, nodes] becomes the Graph WaveNet [B, feature, time, nodes]
        # layout.  Keeping nodes last is required by the diffusion einsums.
        state = self.input_projection(values.unsqueeze(1))
        adaptive = torch.softmax(
            F.relu(self.nodevec_source @ self.nodevec_target), dim=-1
        )
        supports = (self.graph_forward.to(state), self.graph_reverse.to(state), adaptive.to(state))
        skip = None
        for filter_conv, gate_conv, residual_conv, skip_conv, graph_conv in zip(
            self.filter_convs, self.gate_convs, self.residual_convs, self.skip_convs, self.graph_convs
        ):
            residual = state
            gated = torch.tanh(filter_conv(state)) * torch.sigmoid(gate_conv(state))
            skip_part = skip_conv(gated)
            skip = skip_part if skip is None else skip + skip_part
            state = graph_conv(gated, supports) + residual_conv(gated)
            state = state + residual
        # End projections emit [B, horizon, time, nodes]; selecting the final
        # causal time step already gives the repository forecast layout.
        output = self.end_projection(skip)[:, :, -1, :]
        if restore is not None:
            mean, scale = restore
            output = output * scale + mean
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc)[:, -self.pred_len :, :]
        if self.task_name == "imputation":
            return self.forecast(x_enc)
        raise ValueError("GraphWaveNet only supports forecasting and imputation")
