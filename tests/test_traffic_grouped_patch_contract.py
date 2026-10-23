from pathlib import Path


def test_grouped_patch_screen_is_validation_only_and_replaces_channel_patch():
    text = Path("scripts/run_traffic_d32_grouped_patch_probe.py").read_text(encoding="utf-8")
    for token in (
        "validation_only", "test_metrics_produced", "grouped_patch_calibration",
        "--grouped-patch-relation-fraction", "--calibration-grouped-patch-relation-fraction",
        "--channel-patch-relation-fraction", "mamba_scan_count", "run_gpu_guarded.py",
    ):
        assert token in text
