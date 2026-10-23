import json
from pathlib import Path

from scripts.run_training_plan import _job_specs, _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/mamba_acf_step_reliability_screen_plan_20260903.json"


def _phase_config(plan, phase):
    config = dict(plan.get("global", {}))
    config.update(phase)
    return config


def test_step_reliability_plan_is_valid_and_unlaunched():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert plan["registered_job_counts"]["training_total"] == 12


def test_step_reliability_is_the_only_forecast_flag_difference():
    plan = _load_json(PLAN_PATH)
    off, on = plan["phases"]
    off_config = _phase_config(plan, off)
    on_config = _phase_config(plan, on)
    assert off["name"].endswith("off_h96")
    assert on["name"].endswith("on_h96")
    metadata_keys = {"name", "output_root", "purpose"}
    differences = {
        key
        for key in set(off_config) | set(on_config)
        if key not in metadata_keys
        if off_config.get(key) != on_config.get(key)
    }
    assert differences == {"mamba_acf_step_reliability"}
    assert off_config["mamba_acf_relation_loss"] is True
    assert on_config["mamba_acf_relation_loss"] is True
    assert off_config["autocon_acf_source"] == "raw"
    assert on_config["autocon_acf_source"] == "raw"


def test_step_reliability_plan_expands_exactly_paired_coordinates():
    plan = _load_json(PLAN_PATH)
    jobs = []
    for phase in plan["phases"]:
        jobs.extend(_job_specs(plan, phase, {}, allow_unresolved=True))
    assert len(jobs) == 12
    coordinates = {
        (job["dataset"], job["pred_len"], job["seed"])
        for job in jobs
    }
    assert len(coordinates) == 6
    assert {job["model"] for job in jobs} == {"MambaACFAutoCon"}
    assert {job["autocon_lambda"] for job in jobs} == {0.03}


def test_step_reliability_readout_requires_effective_memory_scale_diagnostic():
    plan = _load_json(PLAN_PATH)
    required = set(plan["comparison_contract"]["secondary_metrics"])
    assert "acf_runtime_gate_mean" in required
    assert "acf_effective_memory_rms" in required
    assert "acf_effective_memory_to_forecast_rms" in required
