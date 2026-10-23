from types import SimpleNamespace

import pytest
import torch

from models.TimeMixer import Model
from scripts.benchmark_model_resources import _normalise_config, _resource_signature
from scripts.run_training_plan import MODEL_SPECS, _command, _job_specs


def _config(**overrides):
    values = dict(
        task_name="long_term_forecast", seq_len=96, label_len=48, pred_len=24,
        enc_in=7, dec_in=7, c_out=7, d_model=16, d_ff=32, e_layers=1,
        dropout=0.0, embed="timeF", freq="h", moving_avg=25, top_k=5,
        timemixer_down_sampling_window=2, timemixer_down_sampling_layers=1,
        timemixer_channel_independence=False, timemixer_decomp_method="moving_avg",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("channel_independence", [False, True])
@pytest.mark.parametrize("decomp_method", ["moving_avg", "dft_decomp"])
def test_timemixer_forecast_shape_and_finiteness(channel_independence, decomp_method):
    model = Model(_config(
        timemixer_channel_independence=channel_independence,
        timemixer_decomp_method=decomp_method,
    ))
    output = model(torch.randn(2, 96, 7), None, None, None)
    assert output.shape == (2, 24, 7)
    assert torch.isfinite(output).all()


def test_timemixer_rejects_invalid_downsampling_contract():
    with pytest.raises(ValueError, match="fewer than two"):
        Model(_config(seq_len=16, timemixer_down_sampling_layers=4))


def test_timemixer_is_registered_and_serialized_in_plan_command(tmp_path):
    assert MODEL_SPECS["TimeMixer"] == {"model": "TimeMixer", "autocon": False}
    plan = {"global": {"features": "M", "seq_len": 96, "label_len": 48,
                        "d_model": 16, "d_ff": 32, "e_layers": 1,
                        "timemixer_down_sampling_window": 2,
                        "timemixer_down_sampling_layers": 1,
                        "timemixer_decomp_method": "dft_decomp",
                        "timemixer_channel_independence": True}}
    phase = {"name": "tm", "datasets": ["ETTh1"], "models": ["TimeMixer"],
             "horizons": [24], "seeds": [2021], "features": "M"}
    job = _job_specs(plan, phase, {}, allow_unresolved=True)[0]
    roots = {key: tmp_path / key for key in ("checkpoints", "results", "test", "logs")}
    roots["result_log"] = tmp_path / "result.txt"
    _, command = _command(job, roots, provenance_id="contract")
    assert "--timemixer_down_sampling_window" in command
    assert command[command.index("--timemixer_decomp_method") + 1] == "dft_decomp"
    assert "--timemixer_channel_independence" in command


def test_timemixer_resource_signature_includes_architecture_controls():
    common = dict(model="TimeMixer", seq_len=96, label_len=48, pred_len=24,
                  enc_in=7, dec_in=7, c_out=7, d_model=16, d_ff=32,
                  e_layers=1, timemixer_down_sampling_window=2,
                  timemixer_down_sampling_layers=1,
                  timemixer_channel_independence=False,
                  timemixer_decomp_method="moving_avg")
    assert _resource_signature(common) != _resource_signature(
        dict(common, timemixer_down_sampling_layers=2)
    )
    assert _normalise_config(dict(common, timemixer_decomp_method="dft_decomp"))["timemixer_decomp_method"] == "dft_decomp"
