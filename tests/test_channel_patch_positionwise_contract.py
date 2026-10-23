from pathlib import Path


def test_positionwise_channel_patch_screen_is_target_excluded_and_one_scan():
    text = Path("scripts/run_pems03_channel_patch_positionwise_probe.py").read_text(encoding="utf-8")
    for token in ("channel-patch-similarity", "positionwise", "target_excluded_from_source_and_initialization", "mamba_scan_count", "run_gpu_guarded.py"):
        assert token in text


def test_positionwise_channel_patch_audit_checks_mode_support_and_boundary():
    text = Path("scripts/audit_pems03_channel_patch_positionwise_probe.py").read_text(encoding="utf-8")
    for token in ("positionwise mode missing", "channel_patch_relation_active_fraction", "target leakage", "screen_passes_expansion_margin"):
        assert token in text
