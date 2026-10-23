from pathlib import Path


def test_official_timemixer_pems03_protocol_is_validation_only_and_guarded():
    text = Path("scripts/run_official_timemixer_pems03_channel_patch_25pct.py").read_text(encoding="utf-8")
    for token in ("validation_only", "test_metrics_produced", "run_gpu_guarded.py", "dataset/PEMS03/PEMS03.npz", "same_seed_pairing"):
        assert token in text


def test_official_timemixer_pems03_audit_checks_commit_and_label_budget():
    text = Path("scripts/audit_official_timemixer_pems03_channel_patch_25pct.py").read_text(encoding="utf-8")
    for token in ("official_commit", "label mismatch", "test boundary", "noninferior_at_one_se"):
        assert token in text
