"""Canonical-style AGCRN baseline for traffic forecasting.

The model follows the material AGCRN design: learned node embeddings induce
an adaptive graph, node-conditioned Chebyshev graph convolutions parameterize
GRU gates, and the last recurrent state emits every forecast horizon directly.
It is a local implementation behind the repository's generic forecasting
interface, not a bit-for-bit reproduction of the external training code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AdaptiveVertexGraphConv(nn.Module):
    """Node-conditioned Chebyshev graph convolution used by AGCRN cells."""

    def __init__(self, input_dim: int, output_dim: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.cheb_k = int(cheb_k)
        if self.cheb_k < 2:
            raise ValueError("AGCRN cheb_k must be at least 2")
        self.weight_pool = nn.Parameter(
            torch.empty(int(embed_dim), self.cheb_k, int(input_dim), int(output_dim))
        )
        self.bias_pool = nn.Parameter(torch.empty(int(embed_dim), int(output_dim)))
        nn.init.xavier_uniform_(self.weight_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def forward(self, values: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        # values: [batch, nodes, channels], node_embeddings: [nodes, embed].
        node_count = int(node_embeddings.shape[0])
        support = torch.softmax(
            F.relu(node_embeddings @ node_embeddings.transpose(0, 1)), dim=-1
        )
        supports = [torch.eye(node_count, device=values.device, dtype=values.dtype), support]
        for _ in range(2, self.cheb_k):
            supports.append(2.0 * support @ supports[-1] - supports[-2])
        stacked = torch.stack(supports, dim=0)
        weights = torch.einsum("ne,ekio->nkio", node_embeddings, self.weight_pool)
        bias = node_embeddings @ self.bias_pool
        graph_values = torch.einsum("knm,bmc->bknc", stacked, values).permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", graph_values, weights) + bias


class _AGCRNCell(nn.Module):
    """Adaptive graph-convolutional GRU cell from AGCRN."""

    def __init__(self, node_count: int, input_dim: int, hidden_dim: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.node_count = int(node_count)
        self.hidden_dim = int(hidden_dim)
        self.gates = _AdaptiveVertexGraphConv(
            input_dim + hidden_dim, 2 * hidden_dim, cheb_k, embed_dim
        )
        self.candidate = _AdaptiveVertexGraphConv(
            input_dim + hidden_dim, hidden_dim, cheb_k, embed_dim
        )

    def forward(self, values: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gates(torch.cat((values, state), dim=-1), node_embeddings))
        update, reset = gates.chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(torch.cat((values, update * state), dim=-1), node_embeddings)
        )
        return reset * state + (1.0 - reset) * candidate


class Model(nn.Module):
    """AGCRN-style direct multi-horizon traffic forecaster."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = str(configs.task_name)
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.hidden = int(getattr(configs, "d_model", 24))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "LastVal"))
        if min(self.seq_len, self.pred_len, self.channels, self.hidden) <= 0:
            raise ValueError("AGCRN dimensions must be positive")
        # The official implementation uses a learned node embedding and a
        # small Chebyshev support bank.  These fixed defaults keep the baseline
        # compact while d_model remains the registered recurrent width.
        self.embed_dim = min(10, self.channels)
        self.cheb_k = 2
        self.node_embeddings = nn.Parameter(torch.empty(self.channels, self.embed_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        self.input_projection = nn.Linear(1, self.hidden)
        self.cell = _AGCRNCell(
            self.channels, self.hidden, self.hidden, self.cheb_k, self.embed_dim
        )
        self.readout = nn.Linear(self.hidden, self.pred_len)

    def _normalize(self, x_enc: torch.Tensor):
        if self.AutoCon_wnorm.strip().lower() != "meanstd":
            return x_enc, None
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - mean
        scale = torch.sqrt(
            centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        return centered / scale, (mean, scale)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        values, restore = self._normalize(x_enc)
        batch = int(values.shape[0])
        state = values.new_zeros(batch, self.channels, self.hidden)
        embeddings = self.node_embeddings.to(device=values.device, dtype=values.dtype)
        for step in range(self.seq_len):
            step_values = self.input_projection(values[:, step, :].unsqueeze(-1))
            state = self.cell(step_values, state, embeddings)
        output = self.readout(state).permute(0, 2, 1)
        if restore is not None:
            mean, scale = restore
            output = output * scale + mean
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc)[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.forecast(x_enc)
        raise ValueError("AGCRN only supports forecasting and imputation")
