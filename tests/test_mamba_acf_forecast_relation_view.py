"""Contracts for the direct forecast view used by global ACF relation loss."""

from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from layers.losses import ACFRelationLoss
from models.MambaACF import Model
from run import validate_mamba_acf_arguments
from tests.test_mamba_acf_contract import _args


def _forecast_relation_args():
    args = _args(channels=3, pred_len=12)
    args.mamba_use_acf_memory = False
    args.mamba_acf_relation_loss = True
    args.mamba_acf_relation_scope = "global"
    args.mamba_acf_relation_representation = "forecast"
    args.features = "M"
    return args


def _install_lightweight_encoder(model):
    """Avoid a backend-specific Mamba forward in graph-contract tests."""

    def encode(self, x, x_mark, *, return_base=False, return_full=True):
        hidden = x.mean(dim=-1, keepdim=True).expand(-1, -1, self.d_model)
        full = hidden if return_full else None
        return (hidden, full) if return_base else (full if return_full else hidden)

    model._encode_channels = MethodType(encode, model)


def test_forecast_relation_view_is_prediction_only_identity_and_routes_gradient():
    torch.manual_seed(20260903)
    off_args = _forecast_relation_args()
    off_args.mamba_acf_relation_representation = "encoder"
    off = Model(off_args).eval()

    torch.manual_seed(20260903)
    on = Model(_forecast_relation_args()).eval()
    _install_lightweight_encoder(off)
    _install_lightweight_encoder(on)

    shared = set(off.state_dict()).intersection(on.state_dict())
    assert shared
    for key in shared:
        assert torch.equal(off.state_dict()[key], on.state_dict()[key]), key

    values = torch.randn(3, off_args.seq_len, off_args.enc_in)
    with torch.no_grad():
        off_prediction, _ = off(
            values, None, None, None, return_repr=False
        )
        on_prediction, relation_view = on(
            values, None, None, None, return_repr=True
        )
        on_prediction_only, no_relation_view = on(
            values, None, None, None, return_repr=False
        )
    assert torch.equal(off_prediction, on_prediction)
    assert torch.equal(on_prediction, on_prediction_only)
    assert no_relation_view is None
    assert tuple(relation_view.shape) == (
        values.shape[0], off_args.pred_len, off_args.enc_in
    )

    prior = torch.zeros(off_args.enc_in, 64)
    prior[:, 24] = 0.9
    objective = ACFRelationLoss(
        prior,
        relation_scope="global",
        selected_lags=[24],
        min_lag=1,
    )
    on.zero_grad(set_to_none=True)
    _, relation_view = on(values, None, None, None, return_repr=True)
    _, global_loss = objective(
        F.normalize(relation_view, dim=-1),
        torch.tensor([0, 24, 48]),
    )
    global_loss.mean().backward()
    assert on.temporal_head.weight.grad is not None
    assert float(on.temporal_head.weight.grad.abs().sum()) > 0.0
    assert on.linear_residual.weight.grad is not None
    assert float(on.linear_residual.weight.grad.abs().sum()) > 0.0

    profile = on.resource_profile()
    assert profile["acf_relation_forecast_view_enabled"]
    assert profile["acf_relation_forecast_view_input_shape"] == "B,H,C"
    assert profile["acf_relation_forecast_view_global_pooling"] == "mean_over_horizon"
    assert not profile["acf_relation_forecast_view_is_stepwise_temporal_objective"]
    assert profile["acf_relation_forecast_view_prediction_only_macs_per_batch_item"] == 0


def test_forecast_relation_view_global_objective_is_window_level_and_rejects_one_channel():
    args = _forecast_relation_args()
    model = Model(args).eval()
    _install_lightweight_encoder(model)
    values = torch.randn(3, args.seq_len, args.enc_in)
    with torch.no_grad():
        _, relation_view = model(values, None, None, None, return_repr=True)
    prior = torch.zeros(args.enc_in, 64)
    prior[:, 24] = 0.9
    objective = ACFRelationLoss(
        prior,
        relation_scope="global",
        selected_lags=[24],
        min_lag=1,
    )
    labels = torch.tensor([0, 24, 48])
    _, original = objective(F.normalize(relation_view, dim=-1), labels)
    _, reversed_horizon = objective(
        F.normalize(relation_view.flip(1), dim=-1), labels)
    torch.testing.assert_close(original, reversed_horizon)

    invalid = _forecast_relation_args()
    invalid.enc_in = 1
    invalid.c_out = 1
    with pytest.raises(ValueError, match="enc_in>=2"):
        Model(invalid)
    invalid.model = "MambaACFAutoCon"
    with pytest.raises(ValueError, match="enc_in>=2"):
        validate_mamba_acf_arguments(invalid)


def test_forecast_relation_view_routes_gradient_into_active_acf_memory_gate():
    args = _forecast_relation_args()
    args.mamba_use_acf_memory = True
    args.mamba_acf_lags = [24]
    model = Model(args).eval()
    _install_lightweight_encoder(model)
    full_prior = torch.zeros(args.enc_in, 25)
    full_prior[:, 24] = 0.9
    model.set_acf_prior(full_prior, lags=[24])
    values = torch.randn(3, args.seq_len, args.enc_in)
    prior = torch.zeros(args.enc_in, 64)
    prior[:, 24] = 0.9
    objective = ACFRelationLoss(
        prior,
        relation_scope="global",
        selected_lags=[24],
        min_lag=1,
    )
    model.zero_grad(set_to_none=True)
    _, relation_view = model(values, None, None, None, return_repr=True)
    _, global_loss = objective(
        F.normalize(relation_view, dim=-1), torch.tensor([0, 24, 48])
    )
    global_loss.mean().backward()
    # The lightweight encoder used here can make the weight contribution
    # cancel across the pooled global anchors; the gate bias is the stable
    # direct-path witness and must receive a nonzero relation gradient.
    assert model.acf_gate.bias.grad is not None
    assert float(model.acf_gate.bias.grad.abs().sum()) > 0.0


def test_forecast_relation_view_requires_global_raw_relation_contract():
    invalid = _forecast_relation_args()
    invalid.mamba_acf_relation_scope = "both"
    try:
        Model(invalid)
    except ValueError as exc:
        assert "relation_scope=global" in str(exc)
    else:
        raise AssertionError("forecast relation view accepted local/both scope")

    invalid = _forecast_relation_args()
    invalid.AutoCon_acf_source = "residual"
    try:
        Model(invalid)
    except ValueError as exc:
        assert "AutoCon_acf_source=raw" in str(exc)
    else:
        raise AssertionError("forecast relation view accepted a non-raw ACF source")

    invalid = _forecast_relation_args()
    invalid.c_out = 1
    try:
        Model(invalid)
    except ValueError as exc:
        assert "c_out=enc_in" in str(exc)
    else:
        raise AssertionError("forecast relation view accepted a mismatched output shape")
