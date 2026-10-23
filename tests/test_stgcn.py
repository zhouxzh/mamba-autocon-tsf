import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.STGCN import Model
from models.GraphWaveNet import Model as GraphWaveNet
from models.DCRNN import Model as DCRNN, _DiffusionGRUCell
from models.AGCRN import Model as AGCRN


def _config(path: Path, expected_hash: str = ""):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=8,
        pred_len=3,
        enc_in=3,
        d_model=4,
        dropout=0.0,
        AutoCon_wnorm="MeanStd",
        mamba_ts_graph_path=str(path),
        mamba_ts_graph_expected_sha256=expected_hash,
    )


def _write_graph(path: Path, node_count: int = 3):
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        sensor_ids=np.asarray([str(i) for i in range(node_count)], dtype="U"),
        target=np.asarray([0, 1, 2], dtype=np.int64),
        source=np.asarray([0, 1, 2], dtype=np.int64),
        weight=np.ones(3, dtype=np.float32),
        node_count=np.asarray(node_count, dtype=np.int64),
    )


def test_stgcn_forward_shape_and_finiteness(tmp_path):
    graph = tmp_path / "graph.npz"
    _write_graph(graph)
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    model = Model(_config(graph, digest))
    output = model(torch.randn(2, 8, 3), None, None, None)
    assert output.shape == (2, 3, 3)
    assert torch.isfinite(output).all()


def test_stgcn_rejects_graph_hash_mismatch(tmp_path):
    graph = tmp_path / "graph.npz"
    _write_graph(graph)
    with pytest.raises(ValueError, match="hash mismatch"):
        Model(_config(graph, "0" * 64))


def test_stgcn_rejects_graph_channel_mismatch(tmp_path):
    graph = tmp_path / "graph.npz"
    _write_graph(graph, node_count=4)
    with pytest.raises(ValueError, match="node count"):
        Model(_config(graph))


def test_graph_wavenet_forward_shape_finiteness_and_gradients(tmp_path):
    graph = tmp_path / "graph.npz"
    _write_graph(graph, node_count=4)
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    config = _config(graph, digest)
    config.enc_in = 4
    model = GraphWaveNet(config)
    inputs = torch.randn(2, 8, 4, requires_grad=True)
    output = model(inputs, None, None, None)
    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.nodevec_source.grad is not None
    assert torch.isfinite(model.nodevec_source.grad).all()


def test_graph_wavenet_rejects_graph_hash_mismatch(tmp_path):
    graph = tmp_path / "graph.npz"
    _write_graph(graph)
    with pytest.raises(ValueError, match="hash mismatch"):
        GraphWaveNet(_config(graph, "0" * 64))


def test_dcrnn_forward_shape_finiteness_and_gradients(tmp_path):
    graph = tmp_path / "graph.npz"
    _write_graph(graph, node_count=4)
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    config = _config(graph, digest)
    config.enc_in = 4
    config.d_model = 3
    model = DCRNN(config)
    inputs = torch.randn(2, 8, 4, requires_grad=True)
    output = model(inputs, None, None, None)
    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.cell.gate_projection.weight.grad is not None
    assert torch.isfinite(model.cell.gate_projection.weight.grad).all()
    assert model.cell.candidate_projection.weight.grad is not None
    assert torch.isfinite(model.cell.candidate_projection.weight.grad).all()


def test_dcrnn_rejects_graph_hash_mismatch(tmp_path):
    graph = tmp_path / "graph.npz"
    _write_graph(graph)
    with pytest.raises(ValueError, match="hash mismatch"):
        DCRNN(_config(graph, "0" * 64))


def test_dcrnn_cached_input_diffusion_matches_repeated_formula():
    torch.manual_seed(17)
    cell = _DiffusionGRUCell(node_count=3, hidden=2, dropout=0.0)
    values = torch.randn(2, 3, 2)
    state = torch.randn(2, 3, 2)
    forward = torch.tensor(
        [[0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    reverse = forward.transpose(0, 1)
    supports = (forward, reverse)

    # Reference the pre-cache formula: it recomputes input diffusion for the
    # candidate branch. The optimized cell may only reuse those identical terms.
    reference_diffuse = lambda tensor, support: torch.einsum(
        "bnc,nm->bmc", tensor, support
    )
    input_terms = [values] + [reference_diffuse(values, support) for support in supports]
    state_terms = [state] + [reference_diffuse(state, support) for support in supports]
    gates = cell.gate_projection(torch.cat(input_terms + state_terms, dim=-1))
    update, reset, _ = gates.chunk(3, dim=-1)
    update, reset = torch.sigmoid(update), torch.sigmoid(reset)
    candidate_state = reset * state
    candidate_terms = [candidate_state] + [
        reference_diffuse(candidate_state, support) for support in supports
    ]
    expected = update * state + (1.0 - update) * torch.tanh(
        cell.candidate_projection(torch.cat(input_terms + candidate_terms, dim=-1))
    )
    assert torch.allclose(cell(values, state, supports), expected, atol=1e-7, rtol=1e-6)


def test_agcrn_forward_shape_finiteness_and_adaptive_graph_gradients():
    config = _config(Path("unused.npz"))
    config.enc_in = 4
    config.d_model = 3
    model = AGCRN(config)
    inputs = torch.randn(2, 8, 4, requires_grad=True)
    output = model(inputs, None, None, None)
    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.node_embeddings.grad is not None
    assert torch.isfinite(model.node_embeddings.grad).all()
    assert model.cell.gates.weight_pool.grad is not None
