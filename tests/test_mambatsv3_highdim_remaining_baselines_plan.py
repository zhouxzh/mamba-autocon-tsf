"""Static checks for the same-source high-dimensional baseline continuation."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


PLAN_PATH = Path("docs/research/mambatsv3_highdim_remaining_strong_baselines_h96_validation_20260928.json")


def test_remaining_highdim_baselines_are_complete_validation_only_jobs():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["global"]["skip_test"] is True
    assert plan["registered_job_counts"]["training_total"] == 30
    jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=False)
    assert len(jobs) == 30
    assert {job["model"] for job in jobs} == {"RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2"}
    assert {(job["dataset"], job["pred_len"], job["seed"]) for job in jobs} == {
        (dataset, 96, seed) for dataset in ("electricity", "traffic") for seed in (3291, 3292, 3293)
    }


def test_remaining_highdim_baseline_commands_exclude_candidate_features_and_test_access():
    plan = _load_json(PLAN_PATH)
    roots = {"results": Path("tmp_r"), "checkpoints": Path("tmp_c"), "test": Path("tmp_t"), "logs": Path("tmp_l"), "result_log": Path("tmp_q")}
    for job in _job_specs(plan, plan["phases"][0], {}, allow_unresolved=False):
        _, command = _command(job, roots)
        assert "--skip_test" in command
        assert "--stream_metrics" not in command
        assert "--AutoCon" not in command
        assert "--mamba_ts_repr_forecast_fusion" not in command
