"""Contract tests for the not-yet-authorized memory-gate mode screen."""

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan
from scripts import analyze_mamba_acf_memory_gate_mode_screen as analyzer


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/mamba_acf_memory_gate_mode_screen_plan_20260902.json"


def test_memory_gate_screen_is_unauthorized_and_exactly_two_arms():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert plan["require_acf_sidecar"] is True
    assert [phase["name"] for phase in plan["phases"]] == [
        "memory_gate_hidden_h96",
        "memory_gate_variable_token_h96",
    ]
    assert sum(
        len(_job_specs(plan, phase, {}, allow_unresolved=True))
        for phase in plan["phases"]
    ) == 12


def test_memory_gate_screen_changes_only_the_gate_mode():
    plan = _load_json(PLAN_PATH)
    first, second = plan["phases"]
    first_jobs = _job_specs(plan, first, {}, allow_unresolved=True)
    second_jobs = _job_specs(plan, second, {}, allow_unresolved=True)
    assert len(first_jobs) == len(second_jobs) == 6
    for left, right in zip(first_jobs, second_jobs):
        assert (left["dataset"], left["pred_len"], left["seed"]) == (
            right["dataset"], right["pred_len"], right["seed"]
        )
        left_config = dict(left["config"])
        right_config = dict(right["config"])
        assert left_config.pop("mamba_acf_memory_gate_mode") == "hidden"
        assert right_config.pop("mamba_acf_memory_gate_mode") == "variable_token"
        for metadata_key in ("name", "purpose", "output_root"):
            left_config.pop(metadata_key, None)
            right_config.pop(metadata_key, None)
        assert left_config == right_config

    roots = {
        "results": ROOT / "tmp_memory_gate_results",
        "checkpoints": ROOT / "tmp_memory_gate_checkpoints",
        "test": ROOT / "tmp_memory_gate_test_results",
        "logs": ROOT / "tmp_memory_gate_logs",
        "result_log": ROOT / "tmp_memory_gate_results.txt",
    }
    _, hidden_command = _command(first_jobs[0], roots)
    _, token_command = _command(second_jobs[0], roots)
    assert hidden_command[hidden_command.index("--mamba_acf_memory_gate_mode") + 1] == "hidden"
    assert token_command[token_command.index("--mamba_acf_memory_gate_mode") + 1] == "variable_token"


def test_plan_validator_rejects_variable_token_mode_without_static_mixer():
    plan = _load_json(PLAN_PATH)
    invalid = copy.deepcopy(plan)
    invalid["phases"][1]["mamba_variable_mixer"] = False
    errors = _validate_plan(invalid)
    assert any("memory_gate_mode=variable_token" in error for error in errors)


def _synthetic_rows():
    rows = []
    for arm, offset in (("hidden", 0.0), ("variable_token", -0.01)):
        for dataset in analyzer.EXPECTED_DATASETS:
            for seed in analyzer.EXPECTED_SEEDS:
                rows.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "seed": seed,
                        "mse": 1.0 + offset,
                        "mae": 0.5 + offset,
                        "elapsed_sec": 2.0,
                        "sidecar_sha256": "a" * 64,
                        "effective_lags": [24, 168, 336],
                        "relation_lags": [168, 336],
                        "config_signature": {"common": "same"},
                        "gate_state": {
                            "gate_parameter_count": 3168,
                            "state_numel": 100,
                            "gate_bias_mean": 0.0,
                            "gate_weight_l2": 1.0,
                            "variable_residual_scale": 0.0,
                        },
                        "runtime_diagnostics": {
                            "acf_runtime_gate_mean": {"mean": 0.5},
                            "acf_runtime_gate_std": {"mean": 0.1},
                            "acf_effective_memory_rms": {"mean": 0.2},
                        },
                        "relation_diagnostics": {
                            "acf_relation_global_sign_accuracy": {"mean": 0.9},
                        },
                        "job_id": f"{arm}-{dataset}-{seed}",
                    }
                )
    return rows


def test_synthetic_readout_applies_registered_stop_rule():
    plan = _load_json(PLAN_PATH)
    plan["_path"] = str(PLAN_PATH)
    result = analyzer.build_readout(plan, _synthetic_rows())
    contrast = result["contrasts"]["variable_token_minus_hidden"]["mse"]
    assert contrast["n"] == 6
    assert contrast["wins_lower"] == 6
    assert abs(contrast["mean"] + 0.01) < 1e-12
    assert result["stopping_decision"]["meets_descriptive_screen"] is True


def test_not_run_cli_does_not_create_readout_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_json = Path(temp_dir) / "readout.json"
        output_md = Path(temp_dir) / "readout.md"
        command = [
            sys.executable,
            str(ROOT / "scripts/analyze_mamba_acf_memory_gate_mode_screen.py"),
            "--plan",
            str(PLAN_PATH),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert completed.returncode == 3
        assert "NOT_RUN" in completed.stderr
        assert not output_json.exists()
        assert not output_md.exists()
