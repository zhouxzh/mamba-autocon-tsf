"""Contracts for the not-yet-authorized direct-anchor screen."""

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan
from scripts import analyze_mamba_acf_anchor_copy_screen as analyzer


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/mamba_acf_anchor_copy_screen_plan_20260902.json"


def test_anchor_screen_is_unauthorized_and_has_twelve_jobs():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert [phase["name"] for phase in plan["phases"]] == [
        "anchor_copy_off_h96",
        "anchor_copy_on_h96",
    ]
    assert sum(
        len(_job_specs(plan, phase, {}, allow_unresolved=True))
        for phase in plan["phases"]
    ) == 12


def test_anchor_screen_changes_only_anchor_flag_and_forwards_it():
    plan = _load_json(PLAN_PATH)
    first_jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)
    second_jobs = _job_specs(plan, plan["phases"][1], {}, allow_unresolved=True)
    assert len(first_jobs) == len(second_jobs) == 6
    for left, right in zip(first_jobs, second_jobs):
        assert (left["dataset"], left["pred_len"], left["seed"]) == (
            right["dataset"], right["pred_len"], right["seed"]
        )
        left_config = dict(left["config"])
        right_config = dict(right["config"])
        assert left_config.pop("mamba_acf_anchor_copy") is False
        assert right_config.pop("mamba_acf_anchor_copy") is True
        for config in (left_config, right_config):
            config.pop("name", None)
            config.pop("purpose", None)
            config.pop("output_root", None)
        assert left_config == right_config

    roots = {
        "results": ROOT / "tmp_anchor_results",
        "checkpoints": ROOT / "tmp_anchor_checkpoints",
        "test": ROOT / "tmp_anchor_test_results",
        "logs": ROOT / "tmp_anchor_logs",
        "result_log": ROOT / "tmp_anchor_results.txt",
    }
    _, off_command = _command(first_jobs[0], roots)
    _, on_command = _command(second_jobs[0], roots)
    assert "--mamba_acf_anchor_copy" not in off_command
    assert "--mamba_acf_anchor_copy" in on_command


def _synthetic_rows():
    rows = []
    for arm, offset in (("anchor_off", 0.0), ("anchor_on", -0.01)):
        for dataset in analyzer.EXPECTED_DATASETS:
            for seed in analyzer.EXPECTED_SEEDS:
                diagnostics = {
                    key: {"last": 0.1, "mean": 0.1, "count": 1}
                    for key in (
                        "acf_anchor_scale",
                        "acf_anchor_rms",
                        "acf_anchor_valid_fraction",
                        "acf_anchor_strength_mean",
                        "acf_anchor_lag_mean",
                    )
                }
                rows.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "seed": seed,
                        "mse": 1.0 + offset,
                        "mae": 0.5 + offset,
                        "elapsed_sec": 2.0,
                        "parameter_count": 100 + (1 if arm == "anchor_on" else 0),
                        "sidecar_sha256": "a" * 64,
                        "anchor_diagnostics": diagnostics,
                    }
                )
    return rows


def test_synthetic_readout_applies_anchor_stop_rule():
    plan = _load_json(PLAN_PATH)
    plan["_path"] = str(PLAN_PATH)
    result = analyzer.build_readout(plan, _synthetic_rows())
    contrast = result["contrasts"]["mse"]
    assert contrast["n"] == 6
    assert contrast["wins_lower"] == 6
    assert abs(contrast["mean"] + 0.01) < 1e-12
    assert result["stopping_decision"]["meets_descriptive_screen"] is True


def test_analyzer_not_run_does_not_create_outputs():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_json = Path(temp_dir) / "readout.json"
        output_md = Path(temp_dir) / "readout.md"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/analyze_mamba_acf_anchor_copy_screen.py"),
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


def test_anchor_plan_contract_rejects_gate_mode_change():
    plan = _load_json(PLAN_PATH)
    invalid = copy.deepcopy(plan)
    invalid["phases"][1]["mamba_acf_memory_gate_mode"] = "variable_token"
    errors = analyzer._plan_errors(invalid)
    assert any("mamba_acf_memory_gate_mode" in error for error in errors)
