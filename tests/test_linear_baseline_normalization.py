"""Contracts for opt-in MeanStd normalization on linear baselines."""

from types import SimpleNamespace

import pytest
import torch

from models import DLinear, NLinear


def _configs(model_name, norm):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=8,
        pred_len=4,
        moving_avg=3,
        enc_in=3,
        individual=False,
        dropout=0.0,
        num_class=2,
        AutoCon_wnorm=norm,
    )


@pytest.mark.parametrize("module", [DLinear, NLinear])
def test_meanstd_linear_baseline_is_affine_equivariant(module):
    torch.manual_seed(7)
    model = module.Model(_configs(module.__name__, "MeanStd")).eval()
    x = torch.randn(2, 8, 3)
    with torch.no_grad():
        reference = model(x, None, None, None)
        transformed = model(2.5 * x + 1.75, None, None, None)
    torch.testing.assert_close(transformed, 2.5 * reference + 1.75, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("module", [DLinear, NLinear])
def test_linear_baseline_default_keeps_legacy_normalization_mode(module):
    model = module.Model(_configs(module.__name__, "LastVal")).eval()
    assert model.AutoCon_wnorm == "LastVal"
