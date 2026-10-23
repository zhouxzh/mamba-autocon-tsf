from types import SimpleNamespace

import torch

from layers.Embed import DataEmbedding
from layers.FourierCorrelation import FourierBlock
from models.ETSformer import Model as ETSformer
from models.FEDformer import Model as FEDformer


def _config(**overrides):
    values = dict(
        task_name="long_term_forecast",
        seq_len=16,
        label_len=8,
        pred_len=4,
        enc_in=4,
        dec_in=4,
        c_out=4,
        d_model=32,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        top_k=2,
        moving_avg=5,
        dropout=0.0,
        embed="timeF",
        freq="h",
        wo_freq="",
        activation="gelu",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_data_embedding_preserves_upstream_positional_dropout_api():
    embedding = DataEmbedding(2, 8, "timeF", "h", 0.23)
    assert embedding.dropout.p == 0.23


def test_fourier_block_uses_requested_head_count():
    block = FourierBlock(32, 32, seq_len=16, modes=4, n_heads=4)
    query = torch.randn(2, 16, 4, 8)
    output, attention = block(query, query, query, None)
    assert output.shape == (2, 4, 8, 16)
    assert attention is None


def test_fedformer_runs_with_non_default_four_heads():
    model = FEDformer(_config()).eval()
    x = torch.randn(2, 16, 4)
    mark = torch.randn(2, 16, 4)
    dec = torch.randn(2, 12, 4)
    dec_mark = torch.randn(2, 12, 4)
    with torch.no_grad():
        output = model(x, mark, dec, dec_mark)
    assert output.shape == (2, 4, 4)


def test_etsformer_accepts_legacy_embedding_call():
    model = ETSformer(_config()).eval()
    x = torch.randn(2, 16, 4)
    mark = torch.randn(2, 16, 4)
    dec = torch.randn(2, 12, 4)
    dec_mark = torch.randn(2, 12, 4)
    with torch.no_grad():
        output = model(x, mark, dec, dec_mark)
    assert output.shape == (2, 4, 4)
