import torch
import pytest
from torch import nn

from models.MambaDLinearHybrid import Model


class _Mamba(nn.Module):
    c_out = 3
    pred_len = 2

    def forward(self, x, return_repr=False):
        return x[:, :2, :]


class _Linear(nn.Module):
    def forward(self, x, *_):
        return torch.zeros_like(x[:, :2, :])


def test_channelwise_residual_scale_is_zero_initialized_and_target_local():
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
    global_gate = Model(_Mamba(), _Linear(), mamba_correction_output=True)
    channel_gate = Model(_Mamba(), _Linear(), mamba_correction_output=True,
                         channelwise_residual_scale=True)
    torch.testing.assert_close(global_gate(x), torch.zeros_like(x))
    torch.testing.assert_close(channel_gate(x), torch.zeros_like(x))
    assert tuple(channel_gate.residual_scale.shape) == (1, 1, 3)
    with torch.no_grad():
        channel_gate.residual_scale.copy_(torch.tensor([[[0.5, -0.5, 0.25]]]))
    expected = torch.tanh(channel_gate.residual_scale) * x
    torch.testing.assert_close(channel_gate(x), expected)
    channel_gate(x).sum().backward()
    assert channel_gate.residual_scale.grad is not None
    assert float(channel_gate.residual_scale.grad.abs().sum()) > 0


def test_channelwise_residual_bias_is_nested_and_requires_channelwise_scale():
    x = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="requires channelwise_residual_scale"):
        Model(_Mamba(), _Linear(), channelwise_residual_bias=True)
    model = Model(_Mamba(), _Linear(), mamba_correction_output=True,
                  channelwise_residual_scale=True, channelwise_residual_bias=True)
    torch.testing.assert_close(model(x), torch.zeros_like(x))
    assert tuple(model.residual_bias.shape) == (1, 2, 3)
    with torch.no_grad():
        model.residual_bias.copy_(torch.tensor([[[.1, .2, .3], [.4, .5, .6]]]))
    torch.testing.assert_close(model(x), model.residual_bias)


def test_horizon_channel_residual_scale_is_nested_and_has_prediction_shape():
    x = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="requires channelwise_residual_scale"):
        Model(_Mamba(), _Linear(), channelwise_horizon_residual_scale=True)
    model = Model(_Mamba(), _Linear(), mamba_correction_output=True,
                  channelwise_residual_scale=True,
                  channelwise_horizon_residual_scale=True)
    assert tuple(model.residual_scale.shape) == (1, 2, 3)
    torch.testing.assert_close(model(x), torch.zeros_like(x))
    with torch.no_grad():
        model.residual_scale.copy_(torch.tensor([[[.1, .2, .3], [.4, .5, .6]]]))
    torch.testing.assert_close(model(x), torch.tanh(model.residual_scale) * x)


def test_centered_channelwise_scale_preserves_global_path_and_zero_mean_deviations():
    x = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="requires channelwise_residual_scale"):
        Model(_Mamba(), _Linear(), centered_channelwise_residual_scale=True)
    model = Model(_Mamba(), _Linear(), mamba_correction_output=True,
                  channelwise_residual_scale=True,
                  centered_channelwise_residual_scale=True)
    torch.testing.assert_close(model(x), torch.zeros_like(x))
    with torch.no_grad():
        model.global_residual_scale.fill_(0.2)
        model.residual_scale.copy_(torch.tensor([[[0.3, 0.0, -0.3]]]))
    effective = 0.2 + model.residual_scale - model.residual_scale.mean(dim=-1, keepdim=True)
    torch.testing.assert_close(model(x), torch.tanh(effective) * x)


def test_shared_statistic_gate_is_zero_initialized_and_channel_count_agnostic():
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
    model = Model(_Mamba(), _Linear(), mamba_correction_output=True,
                  shared_channel_statistic_residual_gate=True,
                  shared_channel_statistic_gate_rank=5)
    torch.testing.assert_close(model(x), torch.zeros_like(x))
    assert tuple(model.residual_scale.shape) == ()
    assert model.channel_statistic_gate_hidden.in_features == 4
    assert model.channel_statistic_gate_hidden.out_features == 5
    with torch.no_grad():
        model.channel_statistic_gate_output.bias.fill_(0.25)
    expected = torch.tanh(torch.tensor(0.25)) * x
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert model.channel_statistic_gate_output.weight.grad is not None


def test_shared_statistic_gate_cannot_mix_with_channel_indexed_gate():
    with pytest.raises(ValueError, match="alternative to channelwise_residual_scale"):
        Model(_Mamba(), _Linear(), channelwise_residual_scale=True,
              shared_channel_statistic_residual_gate=True)
