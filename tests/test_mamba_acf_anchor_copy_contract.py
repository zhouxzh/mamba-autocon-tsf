"""Contract tests for the opt-in causal ACF anchor-copy branch."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from run import mamba_architecture_tag, validate_mamba_acf_arguments
from scripts.benchmark_model_resources import _manifest_elapsed_summary, _normalise_config, _parser
from scripts.run_training_plan import _command, _validate_mamba_options
from scripts.validate_training_phase import _validate_acf_prior_artifact, _validate_memory_lag_contract


ROOT = Path(__file__).resolve().parents[1]


def _runtime_args(**overrides):
    values = dict(
        model="MambaACF",
        seq_len=96,
        pred_len=96,
        mamba_scales=[1, 2],
        mamba_token_stride=1,
        mamba_acf_lags=[24, 168, 336],
        mamba_acf_anchor_copy=False,
        mamba_use_acf_memory=False,
        mamba_dynamic_acf=False,
        AutoCon_acf_source="raw",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _plan_config(**overrides):
    values = dict(
        seq_len=96,
        mamba_acf_lags=[24, 168, 336],
        mamba_acf_anchor_copy=True,
        mamba_dynamic_acf=False,
        autocon_acf_source="raw",
    )
    values.update(overrides)
    return values


def _roots(prefix="anchor-contract"):
    return {
        "checkpoints": Path("/tmp/{}-checkpoints".format(prefix)),
        "results": Path("/tmp/{}-results".format(prefix)),
        "test": Path("/tmp/{}-test".format(prefix)),
        "result_log": Path("/tmp/{}-results.txt".format(prefix)),
    }


def test_runtime_anchor_copy_is_opt_in_and_static_raw_only():
    validate_mamba_acf_arguments(_runtime_args(mamba_acf_anchor_copy=True))
    assert mamba_architecture_tag(_runtime_args()) != mamba_architecture_tag(
        _runtime_args(mamba_acf_anchor_copy=True)
    )

    with pytest.raises(ValueError, match="supported only by MambaACF"):
        validate_mamba_acf_arguments(
            _runtime_args(model="Mamba2", mamba_acf_anchor_copy=True)
        )
    with pytest.raises(ValueError, match="static ACF memory"):
        validate_mamba_acf_arguments(
            _runtime_args(mamba_acf_anchor_copy=True, mamba_dynamic_acf=True)
        )
    with pytest.raises(ValueError, match="AutoCon_acf_source=raw"):
        validate_mamba_acf_arguments(
            _runtime_args(mamba_acf_anchor_copy=True, AutoCon_acf_source="residual")
        )
    with pytest.raises(ValueError, match="non-empty mamba_acf_lags"):
        validate_mamba_acf_arguments(
            _runtime_args(mamba_acf_anchor_copy=True, mamba_acf_lags=[])
        )


def test_plan_anchor_copy_is_validated_and_forwarded():
    assert _validate_mamba_options(
        "anchor", _plan_config(), ["MambaACF"]
    ) == []
    assert any(
        "only MambaACF" in error
        for error in _validate_mamba_options(
            "anchor", _plan_config(), ["Mamba2"]
        )
    )
    assert any(
        "static ACF memory" in error
        for error in _validate_mamba_options(
            "anchor", _plan_config(mamba_dynamic_acf=True), ["MambaACF"]
        )
    )
    assert any(
        "autocon_acf_source=raw" in error
        for error in _validate_mamba_options(
            "anchor", _plan_config(autocon_acf_source="residual"), ["MambaACF"]
        )
    )
    assert any(
        "mamba_acf_lags" in error and "non-empty" in error
        for error in _validate_mamba_options(
            "anchor", _plan_config(mamba_acf_lags=[]), ["MambaACF"]
        )
    )

    job = {
        "phase": "anchor",
        "dataset": "ETTh1",
        "model": "MambaACF",
        "pred_len": 96,
        "seed": 2021,
        "autocon_lambda": None,
        "config": {"features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7, **_plan_config()},
    }
    _, command = _command(job, _roots())
    assert "--mamba_acf_anchor_copy" in command


def test_resource_normalization_and_manifest_keep_anchor_variants_separate():
    common = {
        "model": "MambaACF",
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 96,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "AutoCon_acf_source": "raw",
    }
    config = _normalise_config({**common, "mamba_acf_anchor_copy": True})
    assert config["mamba_acf_anchor_copy"] is True
    assert config["mamba_use_acf_memory"] is False
    with pytest.raises(ValueError, match="static ACF memory"):
        _normalise_config({**common, "mamba_acf_anchor_copy": True, "mamba_dynamic_acf": True})
    with pytest.raises(ValueError, match="AutoCon_acf_source=raw"):
        _normalise_config({**common, "mamba_acf_anchor_copy": True, "AutoCon_acf_source": "residual"})
    with pytest.raises(ValueError, match="only by MambaACF"):
        _normalise_config({**common, "model": "Mamba2", "mamba_acf_anchor_copy": True})

    args = _parser().parse_args([
        "--model", "MambaACF", "--dataset", "ETTh1", "--features", "M",
        "--seq-len", "96", "--label-len", "48", "--pred-len", "96",
        "--enc-in", "7", "--dec-in", "7", "--c-out", "7",
        "--mamba-acf-anchor-copy", "--autocon-acf-source", "raw",
        "--output-dir", "/tmp/anchor-resource-contract",
    ])
    assert args.mamba_acf_anchor_copy is True

    base_command = [
        "--seq_len", "96", "--pred_len", "96", "--batch_size", "32",
        "--train_epochs", "8", "--d_model", "32", "--d_ff", "32",
        "--e_layers", "1", "--mamba_d_state", "16", "--mamba_d_conv", "4",
        "--mamba_expand", "2", "--mamba_headdim", "16", "--AutoCon_acf_source", "raw",
    ]
    rows = []
    for seed, anchor in ((2021, False), (2022, True)):
        rows.append({
            "job_id": "anchor-{}".format(seed),
            "status": "ok",
            "model": "MambaACF",
            "dataset": "ETTh1",
            "features": "M",
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "seed": seed,
            "autocon_lambda": None,
            "elapsed_sec": 1.0,
            "command": base_command + (["--mamba_acf_anchor_copy"] if anchor else []),
        })
    summary = _manifest_elapsed_summary(rows)
    assert len(summary) == 2
    assert {row["mamba_acf_anchor_copy"] for row in summary} == {False, True}


def test_anchor_copy_forces_strict_sidecar_and_lag_provenance_without_memory():
    config = {
        "model": "MambaACF",
        "mamba_acf_anchor_copy": True,
        "mamba_use_acf_memory": False,
        "mamba_dynamic_acf": False,
        "mamba_causal_embedding": True,
        "mamba_bidirectional": False,
        "AutoCon_acf_source": "raw",
        "mamba_acf_lags": [3],
        "mamba_acf_prior_artifact": "acf_prior.npz",
        "mamba_acf_prior_sha256": "0" * 64,
        "seq_len": 4,
        "pred_len": 2,
        "enc_in": 2,
        "mamba_acf_temperature": 4.0,
    }
    job = {"model": "MambaACF", "phase": "anchor"}
    assert "mamba_acf_prior_artifact_missing" in _validate_acf_prior_artifact(
        config, ROOT / "does-not-exist", job, strict=True
    )
    assert _validate_memory_lag_contract(
        {
            **config,
            "mamba_acf_effective_lags": [3],
            "mamba_acf_train_peak_lags": [3],
            "mamba_acf_lag_mode": "train_peaks",
            "mamba_acf_effective_lag_source": "train_peaks",
        },
        job,
        plan={"acf_falsification_contract": {"fixed_placebo_arm": "placebo"}},
        strict=True,
    ) == []
