"""Static contract tests for fresh Solar relation-auxiliary-only validation."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


PLAN_PATH = Path("docs/research/mambatsv3_relation_auxiliary_solar_h96_validation_20260929.json")


def test_solar_relation_auxiliary_plan_is_complete_and_validation_only():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["registered_job_counts"]["total"] == 24
    phases = {phase["name"]: phase for phase in plan["phases"]}
    assert len(_job_specs(plan, phases["candidate_solar_h96_validation"], {}, allow_unresolved=False)) == 3
    assert len(_job_specs(plan, phases["baselines_solar_h96_validation"], {}, allow_unresolved=False)) == 21


def test_solar_relation_auxiliary_commands_keep_candidate_isolation_and_skip_test():
    plan = _load_json(PLAN_PATH)
    phases = {phase["name"]: phase for phase in plan["phases"]}
    roots = {"results": Path("tmp_r"), "checkpoints": Path("tmp_c"), "test": Path("tmp_t"), "logs": Path("tmp_l"), "result_log": Path("tmp_q")}
    for job in _job_specs(plan, phases["candidate_solar_h96_validation"], {}, allow_unresolved=False):
        _, command = _command(job, roots)
        assert "--AutoCon" in command and "--skip_test" in command
        assert "--mamba_ts_repr_forecast_fusion" not in command
    for job in _job_specs(plan, phases["baselines_solar_h96_validation"], {}, allow_unresolved=False):
        _, command = _command(job, roots)
        assert "--AutoCon" not in command and "--skip_test" in command
