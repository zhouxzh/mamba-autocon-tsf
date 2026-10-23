"""Contracts for the exploratory ACF anchor x soft-memory factorial screen."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from scripts import analyze_mamba_acf_anchor_memory_interaction_screen as analyzer
from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/mamba_acf_anchor_memory_interaction_screen_plan_20260904.json"


def _plan():
    return _load_json(PLAN_PATH)


def test_plan_is_valid_draft_and_expands_to_complete_factorial_matrix():
    plan = _plan()
    assert _validate_plan(plan) == []
    assert analyzer._plan_errors(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert plan["registered_job_counts"]["total"] == analyzer.EXPECTED_JOBS
    assert sum(
        len(_job_specs(plan, phase, {}, allow_unresolved=True))
        for phase in plan["phases"]
    ) == analyzer.EXPECTED_JOBS


def test_factorial_commands_change_only_anchor_and_memory_flags():
    plan = _plan()
    roots = {
        "results": ROOT / "tmp_interaction_results",
        "checkpoints": ROOT / "tmp_interaction_checkpoints",
        "test": ROOT / "tmp_interaction_test_results",
        "logs": ROOT / "tmp_interaction_logs",
        "result_log": ROOT / "tmp_interaction_results.txt",
    }
    jobs_by_arm = {}
    for phase in plan["phases"]:
        jobs = _job_specs(plan, phase, {}, allow_unresolved=True)
        assert len(jobs) == analyzer.EXPECTED_JOBS_PER_PHASE
        jobs_by_arm[phase["name"]] = jobs
    reference = jobs_by_arm["plain_h96"]
    for phase_name, arm, anchor, memory in analyzer.EXPECTED_PHASES:
        jobs = jobs_by_arm[phase_name]
        for ref, current in zip(reference, jobs):
            assert (ref["dataset"], ref["pred_len"], ref["seed"]) == (
                current["dataset"], current["pred_len"], current["seed"]
            )
            left = dict(ref["config"])
            right = dict(current["config"])
            assert right.pop("mamba_acf_anchor_copy") is anchor
            assert right.pop("mamba_use_acf_memory") is memory
            left.pop("mamba_acf_anchor_copy")
            left.pop("mamba_use_acf_memory")
            for config in (left, right):
                for key in ("name", "purpose", "output_root"):
                    config.pop(key, None)
            assert left == right

        _, command = _command(jobs[0], roots)
        assert analyzer._command_flag(command, "--mamba_use_acf_memory") is memory
        assert ("--mamba_acf_anchor_copy" in command) is anchor


def _synthetic_rows():
    values = {
        "plain": 1.00,
        "anchor_only": 0.98,
        "memory_only": 0.99,
        "anchor_memory": 0.96,
    }
    rows = []
    signature = ("shared-trunk", 96, "raw")
    sidecar = "a" * 64
    for _, arm, anchor, memory in analyzer.EXPECTED_PHASES:
        for dataset in analyzer.EXPECTED_DATASETS:
            for seed in analyzer.EXPECTED_SEEDS:
                rows.append(
                    {
                        "phase": f"{arm}_h96",
                        "arm": arm,
                        "anchor": anchor,
                        "memory": memory,
                        "dataset": dataset,
                        "seed": seed,
                        "horizon": 96,
                        "mse": values[arm],
                        "mae": values[arm] / 2.0,
                        "mse_source": "val_loss",
                        "sidecar_sha256": sidecar,
                        "config_signature": signature,
                        "job_id": f"{arm}-{dataset}-{seed}",
                    }
                )
    return rows


def test_synthetic_readout_reports_simple_effects_and_interaction():
    plan = copy.deepcopy(_plan())
    plan["_path"] = str(PLAN_PATH)
    result = analyzer.build_readout(plan, _synthetic_rows())

    pooled = result["contrasts"]["pooled"]
    assert pooled["anchor_at_memory_off"]["mean"] == pytest.approx(-0.02)
    assert pooled["anchor_at_memory_on"]["mean"] == pytest.approx(-0.03)
    assert pooled["memory_at_anchor_off"]["mean"] == pytest.approx(-0.01)
    assert pooled["memory_at_anchor_on"]["mean"] == pytest.approx(-0.02)
    interaction = result["contrasts"]["interaction"]["pooled"]
    assert interaction["n"] == analyzer.EXPECTED_DATASETS.__len__() * len(analyzer.EXPECTED_SEEDS)
    assert interaction["mean"] == pytest.approx(-0.01)
    assert interaction["wins_lower"] == analyzer.EXPECTED_JOBS_PER_PHASE
    assert result["contrasts"]["interaction"]["by_dataset"]["ETTh1"]["n"] == 3
    assert result["stopping_decision"]["meets_descriptive_screen"] is True
    assert result["stopping_decision"]["interaction_coherent_for_follow_up"] is True
    assert result["metric_source"]["test_metrics_read"] is False


def test_directionally_mixed_interaction_does_not_promote_follow_up():
    plan = copy.deepcopy(_plan())
    plan["_path"] = str(PLAN_PATH)
    rows = _synthetic_rows()
    for row in rows:
        if row["dataset"] == "ETTh1":
            row["mse"] = {
                "plain": 1.00,
                "anchor_only": 0.98,
                "memory_only": 0.99,
                "anchor_memory": 0.975,
            }[row["arm"]]
    result = analyzer.build_readout(plan, rows)
    decision = result["stopping_decision"]
    assert decision["meets_descriptive_screen"] is True
    assert decision["interaction_negative_in_both_datasets"] is False
    assert decision["interaction_coherent_for_follow_up"] is False
    assert "do not promote" in decision["action"]


def test_pairing_rejects_factor_or_sidecar_drift():
    rows = _synthetic_rows()
    changed_factor = copy.deepcopy(rows)
    changed_factor[0]["anchor"] = True
    with pytest.raises(ValueError, match="factor encoding differs"):
        analyzer.build_readout({}, changed_factor)

    changed_sidecar = copy.deepcopy(rows)
    changed_sidecar[1]["sidecar_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="sidecars differ"):
        analyzer.build_readout({}, changed_sidecar)

    changed_signature = copy.deepcopy(rows)
    changed_signature[1]["config_signature"] = ("different",)
    with pytest.raises(ValueError, match="common config differs"):
        analyzer.build_readout({}, changed_signature)


def test_runtime_contract_dormant_temperature_is_not_a_second_factor():
    base = {
        "model": "MambaACF",
        "mamba_acf_temperature": 4.0,
        "mamba_acf_anchor_copy": False,
        "mamba_use_acf_memory": False,
        "mamba_runtime_contract": {
            "mamba_acf_anchor_copy": False,
            "mamba_use_acf_memory": False,
            "mamba_acf_temperature": None,
            "mamba_variable_mixer": True,
        },
    }
    memory = copy.deepcopy(base)
    memory["mamba_use_acf_memory"] = True
    memory["mamba_runtime_contract"]["mamba_use_acf_memory"] = True
    memory["mamba_runtime_contract"]["mamba_acf_temperature"] = 4.0
    assert analyzer._config_signature(base) == analyzer._config_signature(memory)


def test_validation_metric_requires_val_loss_and_uses_best_epoch():
    metrics = analyzer._validation_metrics(
        {"val_loss": [1.2, 0.8, 0.9], "val_mae": [0.6, 0.4, 0.5]},
        Path("checkpoint/loss_logs.json"),
    )
    assert metrics["mse"] == pytest.approx(0.8)
    assert metrics["mse_source"] == "val_loss"
    assert metrics["mae"] == pytest.approx(0.4)
    with pytest.raises(analyzer.NotRunError, match="val_loss"):
        analyzer._validation_metrics({"val_mse": [0.8]}, Path("checkpoint/loss_logs.json"))


def test_not_run_cli_does_not_create_readout_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_json = Path(temp_dir) / "readout.json"
        output_md = Path(temp_dir) / "readout.md"
        # The registered screen may have been executed in a shared worktree;
        # make the NOT_RUN assertion independent of whether its real output
        # root currently contains completed artifacts.
        missing_plan = copy.deepcopy(_plan())
        missing_plan["phases"][0]["output_root"] = (
            "artifacts/exploratory/_missing_anchor_memory_interaction_test/plain_h96"
        )
        plan_path = Path(temp_dir) / "missing_plan.json"
        plan_path.write_text(json.dumps(missing_plan), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/analyze_mamba_acf_anchor_memory_interaction_screen.py"),
                "--plan", str(plan_path),
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


def test_plan_contract_rejects_unregistered_extra_factor():
    invalid = copy.deepcopy(_plan())
    invalid["phases"][0]["mamba_acf_temperature"] = 1.0
    invalid["phases"][1]["mamba_acf_temperature"] = 4.0
    assert any("mamba_acf_temperature" in error for error in analyzer._plan_errors(invalid))
