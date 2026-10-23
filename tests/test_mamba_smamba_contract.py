"""Contracts for the S-Mamba-inspired variable-token AutoCon model."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import models.MambaSMambaAutoCon as sm_module


class _IdentityMamba(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def _config(**overrides):
    values = dict(
        seq_len=16, label_len=8, pred_len=6, enc_in=5, c_out=5,
        d_model=8, d_ff=12, e_layers=1, dropout=0.0,
        AutoCon=False, AutoCon_wnorm="MeanStd",
        mamba_d_state=4, mamba_d_conv=3, mamba_expand=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_smamba_plain_and_autocon_shape_gradient_contract():
    x = torch.randn(3, 16, 5)
    with patch.object(sm_module, "_Mamba1", _IdentityMamba):
        plain = sm_module.Model(_config())
        prediction = plain(x)
        assert prediction.shape == (3, 6, 5)

        autocon = sm_module.Model(_config(AutoCon=True))
        forecast, representation = autocon(x, return_repr=True)
        assert forecast.shape == (3, 6, 5)
        assert representation.shape == (3, 16, 12)
        representation.square().mean().backward()
        assert autocon.variable_embedding.weight.grad is not None
        assert autocon.time_embedding.weight.grad is not None


def test_smamba_return_repr_false_avoids_auxiliary_representation_for_training_loop():
    x = torch.randn(2, 16, 5)
    with patch.object(sm_module, "_Mamba1", _IdentityMamba):
        model = sm_module.Model(_config(AutoCon=True))
        forecast, representation = model(x, return_repr=False)
    assert forecast.shape == (2, 6, 5)
    assert representation is None
