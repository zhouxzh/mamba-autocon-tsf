"""Contracts for the exploratory channel-specific ACF lag-bank screen."""

from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from scripts import analyze_mamba_acf_channel_lag_bank_screen as analyzer
from run import _mamba_runtime_contract_payload
from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/mamba_acf_channel_lag_bank_forecast_screen_plan_20260903.json"


def test_plan_contract_is_registered_and_unlaunched():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert analyzer._validate_plan_contract(plan) == []
    assert plan["status"] == analyzer.EXPECTED_PLAN_STATUS
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert plan["global"]["stream_metrics"] is False
    assert all(phase["stream_metrics"] is False for phase in plan["phases"])
    assert plan["comparison_contract"]["metric_source"] == analyzer.EXPECTED_METRIC_SOURCE
    assert (
        plan["comparison_contract"]["metric_source_description"]
        == analyzer.EXPECTED_METRIC_SOURCE_DESCRIPTION
    )
    assert [phase["name"] for phase in plan["phases"]] == [
        name for name, _, _ in analyzer.EXPECTED_PHASES
    ]
    assert sum(
        len(_job_specs(plan, phase, {}, allow_unresolved=True))
        for phase in plan["phases"]
    ) == analyzer.EXPECTED_JOBS


def test_plan_changes_only_registered_lag_bank_flag_and_forwards_it():
    plan = _load_json(PLAN_PATH)
    off_jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)
    on_jobs = _job_specs(plan, plan["phases"][1], {}, allow_unresolved=True)
    assert len(off_jobs) == len(on_jobs) == analyzer.EXPECTED_JOBS_PER_PHASE

    for off, on in zip(off_jobs, on_jobs):
        assert (off["dataset"], off["pred_len"], off["seed"]) == (
            on["dataset"], on["pred_len"], on["seed"]
        )
        off_config = dict(off["config"])
        on_config = dict(on["config"])
        assert off_config.pop("mamba_acf_channel_lag_bank") is False
        assert on_config.pop("mamba_acf_channel_lag_bank") is True
        for key in ("name", "purpose", "output_root"):
            off_config.pop(key, None)
            on_config.pop(key, None)
        assert off_config == on_config

    roots = {
        "results": ROOT / "tmp_lag_bank_results",
        "checkpoints": ROOT / "tmp_lag_bank_checkpoints",
        "test": ROOT / "tmp_lag_bank_test_results",
        "logs": ROOT / "tmp_lag_bank_logs",
        "result_log": ROOT / "tmp_lag_bank_results.txt",
    }
    _, off_command = _command(off_jobs[0], roots)
    _, on_command = _command(on_jobs[0], roots)
    assert "--mamba_acf_channel_lag_bank" not in off_command
    assert "--mamba_acf_channel_lag_bank" in on_command
    assert off_command[
        off_command.index("--mamba_acf_channel_lag_top_k") + 1
    ] == "2"
    assert on_command[
        on_command.index("--mamba_acf_channel_lag_top_k") + 1
    ] == "2"


def test_config_signature_ignores_dormant_runtime_lag_bank_top_k():
    """The requested top-k is common; runtime top-k is treatment-derived."""

    plan = _load_json(PLAN_PATH)
    off = dict(_job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)[0]["config"])
    on = dict(_job_specs(plan, plan["phases"][1], {}, allow_unresolved=True)[0]["config"])
    for config in (off, on):
        config.update({"model": "MambaACFAutoCon", "AutoCon": True})
        config["mamba_runtime_contract"] = _mamba_runtime_contract_payload(config)

    assert off["mamba_runtime_contract"]["mamba_acf_channel_lag_top_k"] is None
    assert on["mamba_runtime_contract"]["mamba_acf_channel_lag_top_k"] == 2
    assert analyzer._config_signature(off) == analyzer._config_signature(on)

    # A real active top-k change remains visible through the common, requested
    # configuration field even though the nested effective field is stripped.
    changed = dict(on)
    changed["mamba_acf_channel_lag_top_k"] = 1
    changed["mamba_runtime_contract"] = _mamba_runtime_contract_payload(changed)
    assert analyzer._config_signature(changed) != analyzer._config_signature(on)


def _support(arm: str) -> dict:
    enabled = arm == "on"
    return {
        "enabled": enabled,
        "top_k": analyzer.EXPECTED_TOP_K,
        "selection": analyzer.EXPECTED_SELECTION if enabled else None,
        "resolved_lags": [24, 48, 72],
        "legal_lags": [24, 48, 72],
        "legal_step_fractions": [1.0, 1.0, 1.0],
        "shape": [7, 3],
        "selected_slots": 14 if enabled else 0,
        "total_slots": 21 if enabled else 0,
        "density": 2.0 / 3.0 if enabled else 0.0,
        "computed_support_sha256": "b" * 64,
        "metadata_valid": True,
    }


def _synthetic_rows():
    rows = []
    for arm, offset in (("off", 0.0), ("on", -0.01)):
        for dataset in analyzer.EXPECTED_DATASETS:
            for seed in analyzer.EXPECTED_SEEDS:
                rows.append(
                    {
                        "phase": f"channel_lag_bank_{arm}_h96",
                        "arm": arm,
                        "enabled": arm == "on",
                        "dataset": dataset,
                        "seed": seed,
                        "horizon": analyzer.EXPECTED_HORIZON,
                        "mse": 1.0 + offset,
                        "mae": 0.5 + offset,
                        "elapsed_sec": 2.0,
                        "sidecar_sha256": "a" * 64,
                        "config_signature": {"common": "same"},
                        "support": _support(arm),
                        "job_id": f"{arm}-{dataset}-{seed}",
                    }
                )
    return rows


def test_synthetic_readout_computes_paired_contrasts_and_stop_rule():
    plan = copy.deepcopy(_load_json(PLAN_PATH))
    plan["_path"] = str(PLAN_PATH)
    result = analyzer.build_readout(plan, _synthetic_rows())

    contrast = result["contrasts"]["on_minus_off"]["mse"]
    assert contrast["n"] == analyzer.EXPECTED_JOBS_PER_PHASE
    assert contrast["wins_lower"] == analyzer.EXPECTED_JOBS_PER_PHASE
    assert contrast["mean"] == pytest.approx(-0.01)
    assert result["contrasts"]["by_dataset"]["ETTh1"]["mse"]["n"] == 3
    assert result["stopping_decision"]["meets_descriptive_screen"] is True
    assert "non-confirmatory" in result["evidence_boundary"]
    assert result["metric_source"]["mae_available_all_rows"] is True
    assert result["contrasts"]["on_minus_off"]["mae"]["n"] == analyzer.EXPECTED_JOBS_PER_PHASE


def test_validation_metrics_use_val_loss_and_omit_incomplete_mae():
    metrics = analyzer._validation_metrics(
        {"val_loss": [1.2, 0.9, 1.0]},
        Path("checkpoint/loss_logs.json"),
        "job-1",
    )
    assert metrics["mse"] == pytest.approx(0.9)
    assert metrics["mse_source"] == "val_loss"
    assert metrics["mae"] is None
    assert metrics["mae_source"] is None

    aliased = analyzer._validation_metrics(
        {
            "val_loss": [0.8, 0.7],
            "val_mse": [0.8, 0.7],
            "validation_mse": [0.8, 0.7],
            "val_mae": [0.4, 0.3],
        },
        Path("checkpoint/loss_logs.json"),
        "job-2",
    )
    assert aliased["mse_source"] == "val_loss"
    assert aliased["mse"] == pytest.approx(0.7)
    assert aliased["mse_aliases_verified"] == ["val_mse", "validation_mse"]
    assert aliased["mae"] == pytest.approx(0.3)
    assert aliased["mae_source"] == "val_mae"

    with pytest.raises(analyzer.NotRunError, match="missing validation trajectory val_loss"):
        analyzer._validation_metrics(
            {"val_mse": [0.8, 0.7]},
            Path("checkpoint/loss_logs.json"),
            "job-2-alias-only",
        )

    with pytest.raises(ValueError, match="validation MSE alias val_mse disagrees with val_loss"):
        analyzer._validation_metrics(
            {"val_loss": [0.8, 0.7], "val_mse": [0.8, 0.6]},
            Path("checkpoint/loss_logs.json"),
            "job-2-mismatch",
        )

    with pytest.raises(ValueError, match="validation MSE alias validation_mse disagrees with val_loss"):
        analyzer._validation_metrics(
            {"val_loss": [0.8, 0.7], "validation_mse": [0.8]},
            Path("checkpoint/loss_logs.json"),
            "job-2-length-mismatch",
        )

    rows = _synthetic_rows()
    for row in rows:
        row["mae"] = None
    plan = copy.deepcopy(_load_json(PLAN_PATH))
    plan["_path"] = str(PLAN_PATH)
    result = analyzer.build_readout(plan, rows)
    assert result["metric_source"]["mae_available_all_rows"] is False
    assert "mae" not in result["contrasts"]["on_minus_off"]
    assert all("mae" not in block for block in result["contrasts"]["by_dataset"].values())
    assert "omitted because val_mae/validation_mae" in analyzer.markdown(result)


def test_readout_metric_source_matches_canonical_validation_field():
    rows = _synthetic_rows()
    for row in rows:
        row["mse_source"] = "val_loss"
    plan = copy.deepcopy(_load_json(PLAN_PATH))
    plan["_path"] = str(PLAN_PATH)
    result = analyzer.build_readout(plan, rows)
    assert result["metric_source"]["primary"] == analyzer.EXPECTED_METRIC_SOURCE
    assert result["metric_source"]["mse_canonical_field"] == "val_loss"
    assert result["metric_source"]["mse_sources_observed"] == ["val_loss"]

    tampered = copy.deepcopy(rows)
    tampered[0]["mse_source"] = "val_mse"
    with pytest.raises(ValueError, match="readout MSE source must be val_loss"):
        analyzer.build_readout(plan, tampered)


def test_pairing_rejects_common_config_or_support_drift():
    rows = _synthetic_rows()
    tampered_config = copy.deepcopy(rows)
    tampered_config[1]["config_signature"] = {"common": "changed"}
    with pytest.raises(ValueError, match="common config differs"):
        analyzer.build_readout({}, tampered_config)

    tampered_support = copy.deepcopy(rows)
    tampered_support[1]["support"]["computed_support_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="computed support differs"):
        analyzer.build_readout({}, tampered_support)


def test_not_run_cli_does_not_create_readout_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_json = Path(temp_dir) / "readout.json"
        output_md = Path(temp_dir) / "readout.md"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/analyze_mamba_acf_channel_lag_bank_screen.py"),
                "--plan",
                str(PLAN_PATH),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 3
        assert "NOT_RUN" in completed.stderr
        assert not output_json.exists()
        assert not output_md.exists()
