from pathlib import Path


def test_traffic_channel_patch_screen_is_validation_only_and_paired():
    text = Path("scripts/run_traffic_d32_channel_patch_probe.py").read_text(encoding="utf-8")
    for token in (
        "validation_only", "test_metrics_produced", "channel_patch_calibration",
        "--channel-patch-relation-fraction", "--calibration-channel-patch-relation-fraction",
        "mamba_scan_count", "run_gpu_guarded.py", "screen_rule",
    ):
        assert token in text
