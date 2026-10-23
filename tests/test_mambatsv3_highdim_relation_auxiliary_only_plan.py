"""Static contract tests for the relation-auxiliary-only confirmation matrix."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


PLAN_PATH = Path("docs/research/mambatsv3_highdim_relation_auxiliary_only_h96_validation_20260928.json")


def test_relation_auxiliary_only_plan_is_complete_and_validation_only():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["global"]["skip_test"] is True
    assert plan["registered_job_counts"]["training_total"] == 6
    jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=False)
    assert len(jobs) == 6
    assert {job["model"] for job in jobs} == {"MambaTSv3AutoCon"}
    assert {(job["dataset"], job["pred_len"], job["seed"]) for job in jobs} == {
        (dataset, 96, seed) for dataset in ("electricity", "traffic") for seed in (3291, 3292, 3293)
    }
    assert all(job.get("mamba_ts_repr_forecast_fusion", False) is False for job in jobs)
    assert all(job["autocon_lambda"] == 0.001 for job in jobs)


def test_relation_auxiliary_only_commands_keep_autocon_and_block_test_access():
    plan = _load_json(PLAN_PATH)
    roots = {"results": Path("tmp_r"), "checkpoints": Path("tmp_c"), "test": Path("tmp_t"), "logs": Path("tmp_l"), "result_log": Path("tmp_q")}
    for job in _job_specs(plan, plan["phases"][0], {}, allow_unresolved=False):
        _, command = _command(job, roots)
        assert "--AutoCon" in command
        assert "--skip_test" in command
        assert "--mamba_ts_repr_forecast_fusion" not in command
        assert "--stream_metrics" not in command
        assert "--save_predictions" not in command
