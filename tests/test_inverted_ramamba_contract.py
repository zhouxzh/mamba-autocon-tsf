from pathlib import Path


def test_inverted_ramamba_is_registered_and_uses_shared_token_states():
    model = Path("models/InvertedRaMambaAutoCon.py").read_text(encoding="utf-8")
    runner = Path("run.py").read_text(encoding="utf-8")
    resources = Path("scripts/benchmark_model_resources.py").read_text(encoding="utf-8")
    plan = Path("scripts/run_training_plan.py").read_text(encoding="utf-8")
    for token in ("temporal_embedding", "variable_blocks", "repr_condition", "repr_forecast_scale"):
        assert token in model
    assert "InvertedRaMambaAutoCon" in runner
    assert "InvertedRaMambaAutoCon" in resources
    assert "InvertedRaMambaAutoCon" in plan
