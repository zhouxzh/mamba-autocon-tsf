"""A compact STGCN-style traffic forecasting baseline.

This implementation follows the canonical STGCN block idea (temporal gated
convolution followed by graph propagation) while keeping the repository's
safe, hashed edge-list contract and the generic forecasting interface.  It is
an explicit baseline, not a claim of bit-for-bit reproduction of an external
repository.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _TemporalGLU(nn.Module):
    """Causal-shape-preserving temporal gated convolution."""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            channels_in,
            2 * channels_out,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        left, gate = self.conv(values).chunk(2, dim=1)
        return self.dropout(left * torch.sigmoid(gate))


class Model(nn.Module):
    """STGCN-style graph-temporal baseline for [batch, time, sensors] data."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(
            configs.seq_len
            if configs.task_name in {"classification", "anomaly_detection", "imputation"}
            else configs.pred_len
        )
        self.channels = int(configs.enc_in)
        self.hidden = int(getattr(configs, "d_model", 32))
        self.kernel_size = 3
        self.AutoCon_wnorm = getattr(configs, "AutoCon_wnorm", "LastVal")
        path = str(getattr(configs, "mamba_ts_graph_path", "") or "").strip()
        if not path:
            raise ValueError("STGCN requires mamba_ts_graph_path")
        target, source, weight = self._load_graph(
            path, getattr(configs, "mamba_ts_graph_expected_sha256", "")
        )
        self.register_buffer("graph_target", target, persistent=True)
        self.register_buffer("graph_source", source, persistent=True)
        self.register_buffer("graph_weight", weight, persistent=True)

        dropout = float(getattr(configs, "dropout", 0.1))
        self.temporal_in = _TemporalGLU(1, self.hidden, self.kernel_size, dropout)
        self.graph_projection = nn.Conv2d(self.hidden, self.hidden, kernel_size=1)
        self.temporal_out = _TemporalGLU(self.hidden, self.hidden, self.kernel_size, dropout)
        self.norm = nn.GroupNorm(1, self.hidden)
        self.head = nn.Linear(self.hidden * self.seq_len, self.pred_len)

    def _load_graph(self, path: str, expected_hash: str):
        graph_path = Path(path).expanduser()
        if not graph_path.is_file():
            raise FileNotFoundError("STGCN graph contract is missing: {}".format(graph_path))
        expected_hash = str(expected_hash or "").strip().lower()
        if expected_hash:
            digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError("STGCN graph contract hash mismatch")
        with np.load(str(graph_path), allow_pickle=False) as graph:
            required = {"schema_version", "sensor_ids", "target", "source", "weight", "node_count"}
            missing = sorted(required.difference(graph.files))
            if missing:
                raise ValueError("STGCN graph contract is missing fields: {}".format(", ".join(missing)))
            node_count = np.asarray(graph["node_count"])
            sensor_ids = np.asarray(graph["sensor_ids"])
            target = np.asarray(graph["target"])
            source = np.asarray(graph["source"])
            weight = np.asarray(graph["weight"], dtype=np.float32)
        if node_count.ndim != 0 or int(node_count.item()) != self.channels:
            raise ValueError("STGCN graph node count does not match enc_in")
        if sensor_ids.ndim != 1 or sensor_ids.size != self.channels:
            raise ValueError("STGCN graph sensor IDs do not match node count")
        if (
            target.ndim != 1 or source.ndim != 1 or weight.ndim != 1
            or target.size == 0 or target.size != source.size or target.size != weight.size
            or target.dtype.kind not in {"i", "u"} or source.dtype.kind not in {"i", "u"}
            or target.min() < 0 or source.min() < 0
            or target.max() >= self.channels or source.max() >= self.channels
            or not np.isfinite(weight).all() or np.any(weight < 0)
        ):
            raise ValueError("STGCN graph edge values are invalid")
        return (
            torch.as_tensor(target, dtype=torch.long),
            torch.as_tensor(source, dtype=torch.long),
            torch.as_tensor(weight, dtype=torch.float32),
        )

    def _propagate(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B, hidden, time, sensors].  Edge weights are row-normalized
        # in the dataset contract, so one sparse directed diffusion is enough
        # for the compact baseline.
        messages = values.index_select(-1, self.graph_source)
        messages = messages * self.graph_weight.to(values).view(1, 1, 1, -1)
        propagated = values.new_zeros(values.shape)
        propagated.index_add_(3, self.graph_target, messages)
        return propagated

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.AutoCon_wnorm == "MeanStd":
            means = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            values = centered / stdev
        else:
            means = None
            stdev = None
            values = x_enc

        # [batch, time, sensors] -> [batch, feature=1, time, sensors].
        values = values.unsqueeze(1)
        values = self.temporal_in(values)
        values = self.norm(values + self.graph_projection(self._propagate(values)))
        values = self.temporal_out(values)
        values = values.permute(0, 3, 1, 2).reshape(values.size(0), self.channels, -1)
        forecast = self.head(values).transpose(1, 2)
        if means is not None:
            forecast = forecast * stdev + means
        return forecast

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc)[:, -self.pred_len :, :]
        if self.task_name == "imputation":
            return self.forecast(x_enc)
        raise ValueError("STGCN only supports forecasting and imputation")
