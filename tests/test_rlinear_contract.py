from pathlib import Path
from types import SimpleNamespace

import torch

from models.RLinear import Model
from scripts.benchmark_model_resources import MODEL_MODULES
from scripts.run_training_plan import MODEL_SPECS, _command, _job_specs, _validate_plan


def _configs(*, channels=3, seq_len=8, pred_len=4, individual=False):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        individual=individual,
    )


def test_rlinear_zero_projection_restores_window_mean():
    model = Model(_configs())
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    source = torch.randn(2, 8, 3)
    output = model(source, None, None, None)
    expected = source.mean(dim=1, keepdim=True).expand(-1, 4, -1)
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)


def test_rlinear_individual_projection_has_finite_gradients():
    model = Model(_configs(channels=2, individual=True))
    source = torch.randn(3, 8, 2, requires_grad=True)
    output = model(source, None, None, None)
    assert output.shape == (3, 4, 2)
    output.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert all(layer.weight.grad is not None for layer in model.linear)
    assert torch.isfinite(model.affine_weight.grad).all()


def test_rlinear_is_registered_for_plans_and_resources():
    assert MODEL_SPECS["RLinear"] == {"model": "RLinear", "autocon": False}
    assert MODEL_MODULES["RLinear"] == "RLinear"
    plan = {
        "version": 1,
        "schema_version": 1,
        "global": {"features": "M", "seq_len": 96, "label_len": 48},
        "phases": [{
            "name": "rlinear_validation",
            "datasets": ["ETTh1"],
            "models": ["RLinear"],
            "horizons": [96],
            "seeds": [2021],
            "skip_test": True,
        }],
    }
    assert _validate_plan(plan) == []
    job = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)[0]
    roots = {
        "checkpoints": Path("/tmp/rlinear-checkpoints"),
        "results": Path("/tmp/rlinear-results"),
        "test": Path("/tmp/rlinear-test"),
        "result_log": Path("/tmp/rlinear-results.txt"),
    }
    _, command = _command(job, roots)
    assert command[command.index("--model") + 1] == "RLinear"
    assert "--AutoCon" not in command
