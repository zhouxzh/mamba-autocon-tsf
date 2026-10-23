"""Contract tests for the Bi-Mamba4TS-inspired local candidate."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

import models.BiPatchMambaAutoCon as bipatch
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
        patch_len=16, patch_stride=8, mamba_bipatch_token_mode="mixing",
        mamba_bipatch_repr_gate=False,
        mamba_d_state=4, mamba_d_conv=2, mamba_expand=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bipatch_mamba_supports_both_token_axes_and_direct_autocon_route():
    x = torch.randn(2, 96, 5)
    with patch.object(bipatch, "_Mamba1", _IdentityMamba):
        for mode in ("mixing", "independent"):
            model = bipatch.Model(_config(mamba_bipatch_token_mode=mode))
            forecast, representation = model(x, return_repr=True)
            assert tuple(forecast.shape) == (2, 12, 5)
            assert tuple(representation.shape) == (2, 96, 16)
            assert float(model.repr_forecast_scale.detach()) == 0.0
            forecast.square().mean().backward()
            assert model.repr_forecast_scale.grad is not None
            with torch.no_grad():
                model.repr_forecast_scale.fill_(1.0)
            routed, routed_representation = model(x, return_repr=True)
            assert not torch.equal(forecast.detach(), routed.detach())
            assert tuple(routed_representation.shape) == (2, 96, 16)


def test_bipatch_mamba_registry_and_runtime_contract():
    assert "BiPatchMamba" in MAMBA_PLAIN_MODELS
    assert "BiPatchMambaAutoCon" in MAMBA_AUTOCON_MODELS
    assert MODEL_SPECS["BiPatchMambaAutoCon"]["autocon"] is True
    assert MODEL_MODULES["BiPatchMambaAutoCon"] == "BiPatchMambaAutoCon"
    payload = _mamba_runtime_contract_payload({
        "model": "BiPatchMambaAutoCon", "AutoCon": True, "features": "M",
        "seq_len": 96, "label_len": 48, "pred_len": 96, "enc_in": 7,
        "c_out": 7, "d_model": 32, "d_ff": 64, "e_layers": 1,
        "dropout": 0.1, "freq": "h", "mamba_d_state": 16,
        "mamba_d_conv": 4, "mamba_expand": 2,
        "mamba_bipatch_token_mode": "mixing",
        "mamba_bipatch_repr_gate": True,
    })
    assert payload["model"] == "BiPatchMambaAutoCon"
    assert payload["mamba_bipatch_token_mode"] == "mixing"
    assert payload["mamba_bipatch_repr_gate"] is True


def test_bipatch_mamba_representation_gate_is_opt_in():
    with patch.object(bipatch, "_Mamba1", _IdentityMamba):
        plain = bipatch.Model(_config())
        gated = bipatch.Model(_config(mamba_bipatch_repr_gate=True))
    assert plain.repr_gate is False
    assert gated.repr_gate is True
