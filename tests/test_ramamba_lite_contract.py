"""Contract tests for the single-scan decomposed RaMambaLite candidate."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

import models.RaMambaLite as ramamba_lite
from run import MAMBA_AUTOCON_MODELS, MAMBA_PLAIN_MODELS, _mamba_runtime_contract_payload
from scripts.benchmark_model_resources import MODEL_MODULES
from scripts.run_training_plan import MODEL_SPECS


class _IdentityMamba(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def _config(**overrides):
    values = dict(
        seq_len=96, pred_len=12, enc_in=5, c_out=5, d_model=8, d_ff=16,
        e_layers=1, dropout=0.0, AutoCon=True, AutoCon_wnorm="MeanStd",
        mamba_d_state=4, mamba_d_conv=2, mamba_expand=2, moving_avg=25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ramamba_lite_uses_one_seasonal_scan_and_autocon_representation():
    x = torch.randn(2, 96, 5)
    with patch.object(ramamba_lite, "_Mamba1", _IdentityMamba):
        model = ramamba_lite.Model(_config())
        forecast, representation = model(x, return_repr=True)
        assert len(model.seasonal_blocks) == 1
        assert tuple(forecast.shape) == (2, 12, 5)
        assert tuple(representation.shape) == (2, 96, 16)
        assert float(model.repr_forecast_scale.detach()) == 0.0
        forecast.square().mean().backward()
        assert model.repr_forecast_scale.grad is not None
        with torch.no_grad():
            model.repr_forecast_scale.fill_(1.0)
        routed, routed_repr = model(x, return_repr=True)
    assert not torch.equal(forecast.detach(), routed.detach())
    assert tuple(routed_repr.shape) == (2, 96, 16)


def test_ramamba_lite_registry_and_runtime_contract():
    assert "RaMambaLite" in MAMBA_PLAIN_MODELS
    assert "RaMambaLiteAutoCon" in MAMBA_AUTOCON_MODELS
    assert MODEL_SPECS["RaMambaLiteAutoCon"]["autocon"] is True
    assert MODEL_MODULES["RaMambaLiteAutoCon"] == "RaMambaLite"
    payload = _mamba_runtime_contract_payload({
        "model": "RaMambaLiteAutoCon", "AutoCon": True, "features": "M",
        "seq_len": 96, "label_len": 48, "pred_len": 96, "enc_in": 7,
        "c_out": 7, "d_model": 32, "d_ff": 64, "e_layers": 1,
        "dropout": 0.1, "freq": "h", "mamba_d_state": 16,
        "mamba_d_conv": 4, "mamba_expand": 2,
    })
    assert payload["model"] == "RaMambaLiteAutoCon"
    assert payload["d_model"] == 32
