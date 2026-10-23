"""Shape and AutoCon contract tests for the TimeMachine-style Mamba path."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import models.MambaTimeMachineAutoCon as tm_module
from models.MambaTM1AutoCon import Model as MambaTM1
from models.MambaTM2AutoCon import Model as MambaTM2


class _IdentityMamba(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def _config(**overrides):
    values = dict(
        seq_len=16,
        label_len=8,
        pred_len=6,
        enc_in=5,
        c_out=5,
        AutoCon=False,
        AutoCon_wnorm="MeanStd",
        dropout=0.0,
        d_ff=12,
        mamba_d_state=4,
        mamba_d_conv=3,
        mamba_expand=2,
        mamba_headdim=4,
        mamba_tm_n1=8,
        mamba_tm_n2=12,
        mamba_tm_repr_width=8,
        mamba_tm_channel_independent=False,
        mamba_tm_residual=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_time_machine_plain_and_autocon_shapes_and_gradient():
    x = torch.randn(3, 16, 5)
    with patch.object(tm_module, "_Mamba1", _IdentityMamba), patch.object(
        tm_module, "_Mamba2", _IdentityMamba
    ):
        plain = MambaTM1(_config())
        prediction = plain(x, None, None, None)
        assert prediction.shape == (3, 6, 5)

        autocon = MambaTM2(_config(AutoCon=True))
        forecast, representation = autocon(
            x, None, None, None, return_repr=True
        )
        assert forecast.shape == (3, 6, 5)
        assert representation.shape == (3, 16, 12)
        representation.pow(2).mean().backward()
        assert autocon.input_projection.weight.grad is not None
        assert torch.isfinite(autocon.input_projection.weight.grad).all()


def test_time_machine_channel_independent_path_and_multivariate_output():
    x = torch.randn(2, 16, 3)
    with patch.object(tm_module, "_Mamba1", _IdentityMamba):
        model = MambaTM1(
            _config(
                enc_in=3,
                c_out=1,
                mamba_tm_channel_independent=True,
            )
        )
        prediction = model(x, None, None, None)
    assert prediction.shape == (2, 6, 1)

