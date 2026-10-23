"""Static contract checks for the high-dimensional linear baseline extension."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


PLAN_PATH = Path("docs/research/mambatsv3_dlinear_nlinear_highdim_h96_validation_20260928.json")


def test_highdim_linear_extension_is_validation_only_and_complete():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert plan["global"]["skip_test"] is True
    assert plan["global"]["save_predictions"] is False
    assert plan["registered_job_counts"]["training_total"] == 18


def test_highdim_linear_extension_pairs_candidate_and_linear_baselines_without_candidate_flags():
    plan = _load_json(PLAN_PATH)
    jobs = [job for phase in plan["phases"] for job in _job_specs(plan, phase, {}, allow_unresolved=False)]
    candidate = [job for job in jobs if job["phase"] == "candidate_h96_validation"]
    baselines = [job for job in jobs if job["phase"] == "linear_baselines_h96_validation"]
    coordinates = {(dataset, 96, seed) for dataset in ("electricity", "traffic") for seed in (3291, 3292, 3293)}
    assert len(candidate) == 6
    assert len(baselines) == 12
    assert {(job["dataset"], job["pred_len"], job["seed"]) for job in candidate} == coordinates
    assert {(job["dataset"], job["pred_len"], job["seed"]) for job in baselines} == coordinates
    assert all(job["model"] == "MambaTSv3AutoCon" for job in candidate)
    assert all(job["config"]["mamba_ts_repr_forecast_fusion"] is True for job in candidate)
    assert {job["model"] for job in baselines} == {"DLinear", "NLinear"}
    assert all(job["config"]["mamba_ts_repr_forecast_fusion"] is False for job in baselines)


def test_highdim_linear_extension_commands_cannot_read_test_or_save_predictions():
    plan = _load_json(PLAN_PATH)
    roots = {
        "results": Path("tmp_highdim_results"), "checkpoints": Path("tmp_highdim_checkpoints"),
        "test": Path("tmp_highdim_test"), "logs": Path("tmp_highdim_logs"),
        "result_log": Path("tmp_highdim_results.txt"),
    }
    for phase in plan["phases"]:
        for job in _job_specs(plan, phase, {}, allow_unresolved=False):
            _, command = _command(job, roots)
            assert "--skip_test" in command
            assert "--stream_metrics" not in command
            assert "--save_predictions" not in command
            if job["phase"] == "candidate_h96_validation":
                assert "--AutoCon" in command
                assert "--mamba_ts_repr_forecast_fusion" in command
            else:
                assert "--AutoCon" not in command
