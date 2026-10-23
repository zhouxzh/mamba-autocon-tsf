"""Executable pre-registration checks for the unapproved traffic test extension."""

from pathlib import Path

from scripts.run_training_plan import _command, _job_specs, _load_json, _validate_plan


PLAN_PATH = Path("docs/research/metro_pemsbay_mambatsv3_strong_baseline_extension_test_release_draft_20260928.json")


def test_traffic_strong_baseline_extension_is_a_valid_unapproved_confirmation_draft():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert plan["status"] == "draft_test_release"
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert plan["registered_job_counts"]["training_total"] == 46


def test_traffic_strong_baseline_extension_has_exact_five_seed_test_pairing_and_isolates_flags():
    plan = _load_json(PLAN_PATH)
    jobs = [job for phase in plan["phases"] for job in _job_specs(plan, phase, {}, allow_unresolved=True)]
    by_phase = {}
    for job in jobs:
        by_phase.setdefault(job["phase"], []).append(job)
    assert {name: len(rows) for name, rows in by_phase.items()} == {
        "selection_fusion_candidate_validation": 6,
        "confirmatory_fusion_candidate_test": 10,
        "confirmatory_stgcn_baseline_test": 10,
        "confirmatory_timesnet_mamba2_baselines_test": 20,
    }
    expected_test_coordinates = {
        (dataset, 12, seed)
        for dataset in ("metr_la", "pems_bay")
        for seed in (3121, 3122, 3123, 3124, 3125)
    }
    for phase_name in (
        "confirmatory_fusion_candidate_test",
        "confirmatory_stgcn_baseline_test",
        "confirmatory_timesnet_mamba2_baselines_test",
    ):
        rows = by_phase[phase_name]
        assert {(row["dataset"], row["pred_len"], row["seed"]) for row in rows} == expected_test_coordinates
        assert all(row["config"]["skip_test"] is False for row in rows)
        assert all(row["config"]["stream_metrics"] is True for row in rows)
    assert all(row["config"]["mamba_ts_repr_forecast_fusion"] is True for row in by_phase["confirmatory_fusion_candidate_test"])
    assert all(row["config"]["mamba_ts_graph_spatial"] is True for row in by_phase["confirmatory_stgcn_baseline_test"])
    assert all(not row["config"].get("mamba_ts_graph_spatial", False) for row in by_phase["confirmatory_timesnet_mamba2_baselines_test"])
    assert all(not row["config"].get("AutoCon", False) for row in by_phase["confirmatory_timesnet_mamba2_baselines_test"])


def test_traffic_strong_baseline_extension_commands_preserve_test_and_graph_contracts():
    plan = _load_json(PLAN_PATH)
    roots = {
        "results": Path("tmp_extension_results"), "checkpoints": Path("tmp_extension_checkpoints"),
        "test": Path("tmp_extension_test"), "logs": Path("tmp_extension_logs"),
        "result_log": Path("tmp_extension_results.txt"),
    }
    for phase in plan["phases"]:
        for job in _job_specs(plan, phase, {}, allow_unresolved=True):
            _, command = _command(job, roots)
            if job["phase"] == "selection_fusion_candidate_validation":
                assert "--skip_test" in command
                assert "--stream_metrics" not in command
            else:
                assert "--skip_test" not in command
                assert "--stream_metrics" in command
            if job["phase"] == "confirmatory_stgcn_baseline_test":
                assert "--mamba_ts_graph_spatial" in command
                assert "--mamba_ts_graph_path" in command
                assert "--AutoCon" not in command
