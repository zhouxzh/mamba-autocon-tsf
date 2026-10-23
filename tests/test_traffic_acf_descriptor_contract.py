from pathlib import Path


def test_acf_descriptor_screen_is_role_local_validation_only_and_one_scan():
    text = Path("scripts/run_traffic_d32_acf_descriptor_probe.py").read_text(encoding="utf-8")
    for token in (
        "descriptor_channel_patch", "validation_only", "test_metrics_produced",
        "--mamba-ts-acf-channel-descriptor", "--channel-patch-relation-fraction",
        "mamba_scan_count", "run_gpu_guarded.py",
    ):
        assert token in text
