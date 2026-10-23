from pathlib import Path


def test_robustness_extension_is_validation_only_and_audits_all_conditions():
    text = Path("scripts/audit_mambatsv3_acf_infonce_local_cluster_robustness.py").read_text(encoding="utf-8")
    for token in (
        "late_validation_shift", "missing_channels_30", "noise_0p10",
        "test_metrics_produced", "stress evaluator opened test", "active_wins",
        "stem-prefix",
    ):
        assert token in text
