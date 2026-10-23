"""Contract tests for the unlaunched forecast-relation routing screen."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/mamba_acf_forecast_relation_view_screen_plan_20260903.json"


def _all_jobs(plan):
    return [
        job
        for phase in plan["phases"]
        for job in _job_specs(plan, phase, {}, allow_unresolved=True)
    ]


def _without_phase_metadata(config):
    result = dict(config)
    for key in (
        "name",
        "purpose",
        "output_root",
        "allow_zero_autocon_lambda",
        "autocon_lambda",
        "mamba_acf_relation_representation",
    ):
        result.pop(key, None)
    return result


def test_forecast_relation_view_screen_is_valid_unlaunched_and_four_armed():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert [phase["name"] for phase in plan["phases"]] == [
        "encoder_relation_null_h96",
        "encoder_relation_h96",
        "forecast_relation_null_h96",
        "forecast_relation_h96",
    ]
    assert plan["registered_job_counts"]["training_total"] == 24


def test_forecast_relation_view_screen_has_exactly_paired_coordinates_and_routes():
    plan = _load_json(PLAN_PATH)
    jobs = _all_jobs(plan)
    assert len(jobs) == 24
    by_phase = {}
    for job in jobs:
        by_phase.setdefault(job["phase"], []).append(job)
    assert {name: len(rows) for name, rows in by_phase.items()} == {
        "encoder_relation_null_h96": 6,
        "encoder_relation_h96": 6,
        "forecast_relation_null_h96": 6,
        "forecast_relation_h96": 6,
    }

    expected_coordinates = {
        (dataset, 96, seed)
        for dataset in ("ETTh1", "traffic")
        for seed in (2021, 2022, 2023)
    }
    for phase_jobs in by_phase.values():
        assert {
            (job["dataset"], job["pred_len"], job["seed"])
            for job in phase_jobs
        } == expected_coordinates
        assert {job["model"] for job in phase_jobs} == {"MambaACFAutoCon"}

    for phase_name in ("encoder_relation_null_h96", "encoder_relation_h96"):
        assert {
            job["config"]["mamba_acf_relation_representation"]
            for job in by_phase[phase_name]
        } == {"encoder"}
    for phase_name in ("forecast_relation_null_h96", "forecast_relation_h96"):
        assert {
            job["config"]["mamba_acf_relation_representation"]
            for job in by_phase[phase_name]
        } == {"forecast"}


def test_forecast_relation_view_screen_keeps_all_nonrouting_contracts_paired():
    plan = _load_json(PLAN_PATH)
    jobs = _all_jobs(plan)
    index = {
        (job["phase"], job["dataset"], job["pred_len"], job["seed"]): job
        for job in jobs
    }
    contract = plan["relation_representation_contract"]
    pairs = (
        (contract["encoder_null_phase"], contract["encoder_active_phase"], 0.0, 0.03),
        (contract["forecast_null_phase"], contract["forecast_active_phase"], 0.0, 0.03),
        (contract["encoder_active_phase"], contract["forecast_active_phase"], 0.03, 0.03),
    )
    for left_phase, right_phase, left_lambda, right_lambda in pairs:
        for dataset, horizon, seed in {
            (job["dataset"], job["pred_len"], job["seed"])
            for job in jobs
        }:
            left = index[(left_phase, dataset, horizon, seed)]
            right = index[(right_phase, dataset, horizon, seed)]
            assert left["autocon_lambda"] == left_lambda
            assert right["autocon_lambda"] == right_lambda
            assert _without_phase_metadata(left["config"]) == _without_phase_metadata(
                right["config"]
            )


def test_forecast_relation_view_screen_forwards_the_route_and_keeps_null_relation_contract():
    plan = _load_json(PLAN_PATH)
    jobs = _all_jobs(plan)
    roots = {
        "results": ROOT / "tmp_forecast_relation_results",
        "checkpoints": ROOT / "tmp_forecast_relation_checkpoints",
        "test": ROOT / "tmp_forecast_relation_test_results",
        "logs": ROOT / "tmp_forecast_relation_logs",
        "result_log": ROOT / "tmp_forecast_relation_results.txt",
    }
    for job in jobs:
        _, command = _command(job, roots)
        route_index = command.index("--mamba_acf_relation_representation")
        assert command[route_index + 1] == job["config"]["mamba_acf_relation_representation"]
        assert "--mamba_acf_relation_loss" in command
        lambda_index = command.index("--AutoCon_lambda")
        assert float(command[lambda_index + 1]) == job["autocon_lambda"]
        assert "--skip_test" in command
        assert "--stream_metrics" not in command
