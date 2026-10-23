from pathlib import Path

from scripts.run_training_plan import _job_specs, _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/metro_pemsbay_nlinear_variable_mamba_acf_selector_h12_screen_plan_20260927.json"


def test_frozen_selector_plan_is_validation_only_and_complete():
    plan = _load_json(PLAN_PATH)

    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is True
    assert plan["execution"]["training_started"] is False
    assert plan["global"]["skip_test"] is True
    assert [phase["name"] for phase in plan["phases"]] == [
        "selector_off_h12",
        "selector_on_h12",
        "external_context_h12",
    ]
    assert sum(
        len(_job_specs(plan, phase, {}, allow_unresolved=True))
        for phase in plan["phases"]
    ) == 48


def test_selector_pair_changes_only_the_registered_selector_flag():
    plan = _load_json(PLAN_PATH)
    off, on, external = plan["phases"]

    assert off["mamba_variable_acf_memory_selector"] is False
    assert on["mamba_variable_acf_memory_selector"] is True
    assert off["seeds"] == on["seeds"] == [2711, 2712, 2713]
    assert off["datasets"] == on["datasets"] == ["metr_la", "pems_bay"]
    assert external["mamba_acf_relation_loss"] is False
    assert external["mamba_variable_repr_fusion"] is False
