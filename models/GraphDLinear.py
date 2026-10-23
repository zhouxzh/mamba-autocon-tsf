"""Sparse-topology DLinear baseline for traffic forecasting.

This is a local baseline, not an implementation or rebranding of DCRNN,
Graph WaveNet, or another published graph forecaster.  It applies the same
validated row-normalized edge list used by the graph-Mamba candidate once to
the observed window, then retains DLinear's shared temporal decomposition and
linear heads.  It isolates the value of graph propagation from Mamba and
AutoCon representation learning.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from models.DLinear import Model as _DLinear


class Model(_DLinear):
    def __init__(self, configs, individual=False):
        super().__init__(configs, individual=individual)
        path = str(getattr(configs, "mamba_ts_graph_path", "") or "").strip()
        if not path:
            raise ValueError("GraphDLinear requires mamba_ts_graph_path")
        target, source, weight = self._load_graph(path, getattr(
            configs, "mamba_ts_graph_expected_sha256", ""
        ))
        self.register_buffer("graph_target", target, persistent=True)
        self.register_buffer("graph_source", source, persistent=True)
        self.register_buffer("graph_weight", weight, persistent=True)
        # ReZero initialization keeps the graph baseline nested in DLinear,
        # permitting a direct graph-off/on ablation from equal initialization.
        self.graph_spatial_scale = torch.nn.Parameter(torch.zeros(()))

    def _load_graph(self, path, expected_hash):
        graph_path = Path(path).expanduser()
        if not graph_path.is_file():
            raise FileNotFoundError("GraphDLinear graph contract is missing: {}".format(graph_path))
        expected_hash = str(expected_hash or "").strip().lower()
        if expected_hash:
            digest = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            if digest != expected_hash:
                raise ValueError("GraphDLinear graph contract hash mismatch")
        with np.load(str(graph_path), allow_pickle=False) as graph:
            required = {"schema_version", "sensor_ids", "target", "source", "weight", "node_count"}
            missing = sorted(required.difference(graph.files))
            if missing:
                raise ValueError("GraphDLinear graph contract is missing fields: {}".format(", ".join(missing)))
            node_count = np.asarray(graph["node_count"])
            sensor_ids = np.asarray(graph["sensor_ids"])
            target = np.asarray(graph["target"])
            source = np.asarray(graph["source"])
            weight = np.asarray(graph["weight"], dtype=np.float32)
        if node_count.ndim != 0 or int(node_count.item()) != self.channels:
            raise ValueError("GraphDLinear graph node count does not match enc_in")
        if sensor_ids.ndim != 1 or sensor_ids.size != self.channels:
            raise ValueError("GraphDLinear graph sensor IDs do not match node count")
        if (
            target.ndim != 1 or source.ndim != 1 or weight.ndim != 1
            or target.size == 0 or target.size != source.size or target.size != weight.size
            or target.dtype.kind not in {"i", "u"} or source.dtype.kind not in {"i", "u"}
            or target.min() < 0 or source.min() < 0
            or target.max() >= self.channels or source.max() >= self.channels
            or not np.isfinite(weight).all() or np.any(weight < 0)
        ):
            raise ValueError("GraphDLinear graph edge values are invalid")
        return (
            torch.as_tensor(target, dtype=torch.long),
            torch.as_tensor(source, dtype=torch.long),
            torch.as_tensor(weight, dtype=torch.float32),
        )

    def _propagate(self, values):
        messages = values.index_select(2, self.graph_source) * self.graph_weight.to(
            device=values.device, dtype=values.dtype
        ).view(1, 1, -1)
        propagated = values.new_zeros(values.shape)
        propagated.index_add_(2, self.graph_target, messages)
        scale = torch.tanh(self.graph_spatial_scale).to(values)
        return values + scale * (propagated - values)

    def forecast(self, x_enc):
        if self.AutoCon_wnorm != "MeanStd":
            return self.encoder(self._propagate(x_enc))
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        forecast = self.encoder(self._propagate(centered / stdev))
        return forecast * stdev + means
