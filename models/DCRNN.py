"""Canonical-style diffusion recurrent traffic forecasting baseline.

This compact implementation keeps the defining DCRNN ingredients: a
diffusion convolution over forward/reverse graph supports inside a gated
recurrent cell and a direct multi-horizon readout.  It uses the repository's
hashed edge-list contract and generic forecasting interface, so it is a
comparable local baseline rather than a bit-for-bit external reproduction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _DiffusionGRUCell(nn.Module):
    """A first-order forward/reverse diffusion GRU cell."""

    def __init__(self, node_count: int, hidden: int, dropout: float):
        super().__init__()
        self.node_count = int(node_count)
        self.hidden = int(hidden)
        # Input and recurrent states each contribute identity, forward and
        # reverse diffusion terms.  The three GRU gates share one projection.
        self.gate_projection = nn.Linear(6 * self.hidden, 3 * self.hidden)
        self.candidate_projection = nn.Linear(6 * self.hidden, self.hidden)
        self.dropout = nn.Dropout(float(dropout))

    @staticmethod
    def _diffuse(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        # Batched GEMM is numerically equivalent to the einsum and maps to a
        # faster CUDA kernel for the [batch, node, hidden] traffic shape.
        return torch.matmul(values.transpose(1, 2), support).transpose(1, 2)

    def _input_terms(
        self, values: torch.Tensor, supports: tuple[torch.Tensor, torch.Tensor]
    ) -> list[torch.Tensor]:
        """Build fixed input diffusion terms once per recurrent time step."""

        input_terms = [values]
        for support in supports:
            input_terms.append(self._diffuse(values, support))
        return input_terms

    def _features(
        self,
        input_terms: list[torch.Tensor],
        state: torch.Tensor,
        supports: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        state_terms = [state]
        for support in supports:
            state_terms.append(self._diffuse(state, support))
        return torch.cat(input_terms + state_terms, dim=-1)

    def forward(
        self,
        values: torch.Tensor,
        state: torch.Tensor,
        supports: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        input_terms = self._input_terms(values, supports)
        features = self._features(input_terms, state, supports)
        gates = self.gate_projection(features)
        update, reset, _ = gates.chunk(3, dim=-1)
        update = torch.sigmoid(update)
        reset = torch.sigmoid(reset)
        candidate_features = self._features(input_terms, reset * state, supports)
        candidate = torch.tanh(self.candidate_projection(candidate_features))
        next_state = update * state + (1.0 - update) * candidate
        return self.dropout(next_state)


class Model(nn.Module):
    """DCRNN-style graph recurrent baseline for [B, time, sensors] data."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = str(configs.task_name)
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(
            self.seq_len
            if self.task_name in {"classification", "anomaly_detection", "imputation"}
            else configs.pred_len
        )
        self.channels = int(configs.enc_in)
        self.hidden = int(getattr(configs, "d_model", 16))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "LastVal"))
        if min(self.seq_len, self.pred_len, self.channels, self.hidden) <= 0:
            raise ValueError("DCRNN dimensions must be positive")
        path = str(getattr(configs, "mamba_ts_graph_path", "") or "").strip()
        if not path:
            raise ValueError("DCRNN requires mamba_ts_graph_path")
        forward, reverse = self._load_graph(
            path, getattr(configs, "mamba_ts_graph_expected_sha256", "")
        )
        self.register_buffer("graph_forward", forward, persistent=True)
        self.register_buffer("graph_reverse", reverse, persistent=True)
        dropout = float(getattr(configs, "dropout", 0.1))
        self.input_projection = nn.Linear(1, self.hidden)
        self.cell = _DiffusionGRUCell(self.channels, self.hidden, dropout)
        self.readout = nn.Linear(self.hidden, self.pred_len)

    def _load_graph(self, path: str, expected_hash: str) -> tuple[torch.Tensor, torch.Tensor]:
        graph_path = Path(path).expanduser()
        if not graph_path.is_file():
            raise FileNotFoundError("DCRNN graph contract is missing: {}".format(graph_path))
        expected_hash = str(expected_hash or "").strip().lower()
        if expected_hash:
            digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError("DCRNN graph contract hash mismatch")
        with np.load(str(graph_path), allow_pickle=False) as graph:
            required = {"schema_version", "sensor_ids", "target", "source", "weight", "node_count"}
            missing = sorted(required.difference(graph.files))
            if missing:
                raise ValueError("DCRNN graph contract is missing fields: {}".format(", ".join(missing)))
            node_count = np.asarray(graph["node_count"])
            sensor_ids = np.asarray(graph["sensor_ids"])
            target = np.asarray(graph["target"])
            source = np.asarray(graph["source"])
            weight = np.asarray(graph["weight"], dtype=np.float32)
        if node_count.ndim != 0 or int(node_count.item()) != self.channels:
            raise ValueError("DCRNN graph node count does not match enc_in")
        if sensor_ids.ndim != 1 or sensor_ids.size != self.channels:
            raise ValueError("DCRNN graph sensor IDs do not match node count")
        if (
            target.ndim != 1 or source.ndim != 1 or weight.ndim != 1
            or target.size == 0 or target.size != source.size or target.size != weight.size
            or target.dtype.kind not in {"i", "u"} or source.dtype.kind not in {"i", "u"}
            or target.min() < 0 or source.min() < 0
            or target.max() >= self.channels or source.max() >= self.channels
            or not np.isfinite(weight).all() or np.any(weight < 0)
        ):
            raise ValueError("DCRNN graph edge values are invalid")
        adjacency = np.zeros((self.channels, self.channels), dtype=np.float32)
        adjacency[source, target] = weight
        row_sum = adjacency.sum(axis=1, keepdims=True)
        forward = adjacency / np.maximum(row_sum, np.float32(1e-12))
        reverse = forward.T.copy()
        reverse_sum = reverse.sum(axis=1, keepdims=True)
        reverse /= np.maximum(reverse_sum, np.float32(1e-12))
        return torch.from_numpy(forward), torch.from_numpy(reverse)

    def _normalize(self, x_enc: torch.Tensor):
        if self.AutoCon_wnorm.lower() != "meanstd":
            return x_enc, None
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - mean
        scale = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return centered / scale, (mean, scale)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        values, restore = self._normalize(x_enc)
        batch = values.shape[0]
        state = values.new_zeros(batch, self.channels, self.hidden)
        supports = (self.graph_forward.to(values), self.graph_reverse.to(values))
        for step in range(self.seq_len):
            step_values = self.input_projection(values[:, step, :].unsqueeze(-1))
            state = self.cell(step_values, state, supports)
        # Each node's final recurrent state emits all horizons directly.
        output = self.readout(state).permute(0, 2, 1)
        if restore is not None:
            mean, scale = restore
            output = output * scale + mean
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc)[:, -self.pred_len :, :]
        if self.task_name == "imputation":
            return self.forecast(x_enc)
        raise ValueError("DCRNN only supports forecasting and imputation")
