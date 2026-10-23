from pathlib import Path


def test_channelwise_acf_probe_preserves_one_scan_and_validation_boundary():
    runner=Path("scripts/run_variable_mamba_acf_channelwise_probe.py").read_text(encoding="utf-8")
    trainer=Path("scripts/pretrain_variable_mamba_acf_core.py").read_text(encoding="utf-8")
    model=Path("models/NLinearVariableMambaAutoCon.py").read_text(encoding="utf-8")
    for token in ("validation_only", "test_metrics_produced", "run_gpu_guarded.py", "one variable-axis Mamba scan"):
        assert token in runner
    for token in ("--representation-channelwise-acf", "channelwise_global_contrastive_loss", "channel_peak_lags"):
        assert token in trainer
    assert "return_variable_tokens" in model
