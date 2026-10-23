from pathlib import Path


def test_linear_latent_mamba_has_nlinear_base_and_shared_autocon_states():
    model = Path("models/LinearLatentMambaAutoCon.py").read_text(encoding="utf-8")
    runner = Path("run.py").read_text(encoding="utf-8")
    resources = Path("scripts/benchmark_model_resources.py").read_text(encoding="utf-8")
    for token in ("base_head", "variable_compress", "horizon_decode", "repr_projection", "same latent"):
        assert token in model
    assert "LinearLatentMambaAutoCon" in runner
    assert "LinearLatentMambaAutoCon" in resources
