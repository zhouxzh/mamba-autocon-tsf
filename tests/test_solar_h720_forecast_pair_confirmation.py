"""Contract checks for the independent Solar H720 forecast-pair confirmation."""

from pathlib import Path

from scripts.run_training_plan import _job_specs, _load_json, _validate_plan


PLAN = Path("docs/research/nlinear_variable_mamba_solar_h720_forecast_pair_confirmation_20260928.json")


def test_solar_h720_forecast_pair_confirmation_is_frozen_and_validation_only():
    plan = _load_json(PLAN)
    assert _validate_plan(plan) == []
    assert plan["execution"] == {
        "authorized": False,
        "training_started": False,
        "human_review_required": True,
    }
    assert plan["registered_job_counts"]["training_total"] == 6
    assert plan["global"]["skip_test"] is True
    assert plan["global"]["save_predictions"] is False


def test_solar_h720_forecast_pair_confirmation_has_exact_paired_treatment_only():
    plan = _load_json(PLAN)
    jobs = [job for phase in plan["phases"] for job in _job_specs(plan, phase, {}, allow_unresolved=True)]
    assert len(jobs) == 6
    by_phase = {name: [job for job in jobs if job["phase"] == name] for name in ("solar_h720_forecast_pair_off", "solar_h720_forecast_pair_on")}
    assert {name: len(rows) for name, rows in by_phase.items()} == {
        "solar_h720_forecast_pair_off": 3,
        "solar_h720_forecast_pair_on": 3,
    }
    for rows in by_phase.values():
        assert {(job["dataset"], job["pred_len"], job["seed"]) for job in rows} == {
            ("solar", 720, 2041), ("solar", 720, 2042), ("solar", 720, 2043)
        }
        assert {job["model"] for job in rows} == {"NLinearVariableMambaAutoCon"}
        assert all(job["config"]["skip_test"] is True for job in rows)
    for left, right in zip(sorted(by_phase["solar_h720_forecast_pair_off"], key=lambda row: row["seed"]), sorted(by_phase["solar_h720_forecast_pair_on"], key=lambda row: row["seed"])):
        assert left["seed"] == right["seed"]
        left_config = dict(left["config"])
        right_config = dict(right["config"])
        for key in (
            "name", "purpose", "output_root",
            "mamba_acf_forecast_pair_consistency", "mamba_acf_forecast_pair_weight",
        ):
            left_config.pop(key)
            right_config.pop(key)
        assert left_config == right_config
