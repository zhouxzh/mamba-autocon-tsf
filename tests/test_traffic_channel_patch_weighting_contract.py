from pathlib import Path


def test_traffic_strength_weighting_screen_preserves_training_and_inference_contracts():
    text = Path("scripts/run_traffic_d32_channel_patch_weighting_probe.py").read_text(encoding="utf-8")
    for token in (
        "validation_only", "test_metrics_produced", "acf_strength", "uniform",
        "--channel-patch-channel-weighting", "mamba_scan_count", "run_gpu_guarded.py",
    ):
        assert token in text
