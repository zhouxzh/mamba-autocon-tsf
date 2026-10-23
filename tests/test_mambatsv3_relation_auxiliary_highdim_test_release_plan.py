"""Static checks for the unapproved high-dimensional held-out draft."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


PLAN_PATH = Path("docs/research/mambatsv3_relation_auxiliary_highdim_test_release_draft_20260928.json")


def test_relation_auxiliary_highdim_draft_is_frozen_and_unapproved():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["status"] == "draft_test_release"
    assert plan["execution"]["authorized"] is False
    assert plan["registered_job_counts"]["total"] == 86
    phases = {phase["name"]: phase for phase in plan["phases"]}
    assert len(_job_specs(plan, phases["selection_relation_auxiliary_only_validation"], {}, allow_unresolved=False)) == 6
    assert len(_job_specs(plan, phases["confirmatory_relation_auxiliary_only_candidate_test"], {}, allow_unresolved=False)) == 10
    assert len(_job_specs(plan, phases["confirmatory_strong_baselines_test"], {}, allow_unresolved=False)) == 70


def test_relation_auxiliary_highdim_draft_keeps_test_boundary_and_candidate_isolation():
    plan = _load_json(PLAN_PATH)
    phases = {phase["name"]: phase for phase in plan["phases"]}
    roots = {"results": Path("tmp_r"), "checkpoints": Path("tmp_c"), "test": Path("tmp_t"), "logs": Path("tmp_l"), "result_log": Path("tmp_q")}
    selection = phases["selection_relation_auxiliary_only_validation"]
    candidate = phases["confirmatory_relation_auxiliary_only_candidate_test"]
    baseline = phases["confirmatory_strong_baselines_test"]
    for job in _job_specs(plan, selection, {}, allow_unresolved=False):
        _, command = _command(job, roots)
        assert "--skip_test" in command
        assert "--stream_metrics" not in command
    for job in _job_specs(plan, candidate, {}, allow_unresolved=False):
        _, command = _command(job, roots)
        assert "--AutoCon" in command
        assert "--skip_test" not in command
        assert "--stream_metrics" in command
        assert "--mamba_ts_repr_forecast_fusion" not in command
    for job in _job_specs(plan, baseline, {}, allow_unresolved=False):
        _, command = _command(job, roots)
        assert "--AutoCon" not in command
        assert "--skip_test" not in command
        assert "--stream_metrics" in command
