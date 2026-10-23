from pathlib import Path


def test_traffic_locality_screen_is_target_prefix_and_validation_only():
    text = Path("scripts/run_traffic_d32_channel_patch_locality_probe.py").read_text(encoding="utf-8")
    for token in (
        "source_and_target", "source_only", "validation_only", "test_metrics_produced",
        "--calibration-channel-patch-relation-fraction", "mamba_scan_count", "run_gpu_guarded.py",
    ):
        assert token in text
