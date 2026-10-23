"""Static contract tests for the plain Mamba-1 high-dimensional baseline."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


PLAN_PATH = Path("docs/research/mambatsv3_mamba1_baseline_highdim_h96_validation_20260929.json")


def test_mamba1_baseline_plan_is_complete_and_validation_only():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=False)
    assert len(jobs) == 6
    assert {job["model"] for job in jobs} == {"Mamba1"}
    assert {(job["dataset"], job["pred_len"], job["seed"]) for job in jobs} == {
        (dataset, 96, seed) for dataset in ("electricity", "traffic") for seed in (3351, 3352, 3353)
    }


def test_mamba1_baseline_commands_disable_candidate_features_and_test_access():
    plan = _load_json(PLAN_PATH)
    roots = {"results": Path("tmp_r"), "checkpoints": Path("tmp_c"), "test": Path("tmp_t"), "logs": Path("tmp_l"), "result_log": Path("tmp_q")}
    for job in _job_specs(plan, plan["phases"][0], {}, allow_unresolved=False):
        _, command = _command(job, roots)
        assert "--skip_test" in command
        assert "--AutoCon" not in command
        assert "--mamba_ts_repr_forecast_fusion" not in command
