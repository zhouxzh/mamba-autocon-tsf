"""Static contract tests for the high-dimensional AutoCon mechanism matrix."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


PLAN_PATH = Path("docs/research/mambatsv3_highdim_autocon_mechanism_h96_validation_20260928.json")


def test_highdim_autocon_mechanism_plan_is_complete_and_validation_only():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["global"]["skip_test"] is True
    assert plan["registered_job_counts"]["training_total"] == 18
    jobs = [job for phase in plan["phases"] for job in _job_specs(plan, phase, {}, allow_unresolved=False)]
    assert len(jobs) == 18
    expected = {(dataset, 96, seed) for dataset in ("electricity", "traffic") for seed in (3301, 3302, 3303)}
    for phase in ("active_fusion_h96", "fusion_off_h96", "lambda_zero_h96"):
        rows = [job for job in jobs if job["phase"] == phase]
        assert {(job["dataset"], job["pred_len"], job["seed"]) for job in rows} == expected
        assert all(job["model"] == "MambaTSv3AutoCon" for job in rows)


def test_highdim_autocon_mechanism_has_only_registered_one_factor_changes_and_no_test_command():
    plan = _load_json(PLAN_PATH)
    phases = {phase["name"]: phase for phase in plan["phases"]}
    active, off, null = phases["active_fusion_h96"], phases["fusion_off_h96"], phases["lambda_zero_h96"]
    assert active["mamba_ts_repr_forecast_fusion"] is True and active["autocon_lambda"] == 0.001
    assert off["mamba_ts_repr_forecast_fusion"] is False and off["autocon_lambda"] == 0.001
    assert null["mamba_ts_repr_forecast_fusion"] is True and null["autocon_lambda"] == 0.0
    roots = {"results": Path("tmp_r"), "checkpoints": Path("tmp_c"), "test": Path("tmp_t"), "logs": Path("tmp_l"), "result_log": Path("tmp_q")}
    for phase in plan["phases"]:
        for job in _job_specs(plan, phase, {}, allow_unresolved=False):
            _, command = _command(job, roots)
            assert "--AutoCon" in command
            assert "--skip_test" in command
            assert "--stream_metrics" not in command
            assert "--save_predictions" not in command
