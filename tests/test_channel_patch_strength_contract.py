from pathlib import Path


def test_channel_patch_strength_screen_preserves_target_exclusion_and_one_scan():
    text = Path("scripts/run_pems03_channel_patch_strength_probe.py").read_text(encoding="utf-8")
    for token in ("channel-patch-relation-fraction", "target_excluded_from_source_and_initialization", "mamba_scan_count", "run_gpu_guarded.py", "screen_rule"):
        assert token in text


def test_channel_patch_strength_audit_requires_stronger_relation_and_support():
    text = Path("scripts/audit_pems03_channel_patch_strength_probe.py").read_text(encoding="utf-8")
    for token in ("fraction_0p10", "channel_patch_relation_active_fraction", "target leakage", "screen_passes_expansion_margin"):
        assert token in text
