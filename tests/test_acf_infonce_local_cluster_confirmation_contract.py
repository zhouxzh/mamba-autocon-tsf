from pathlib import Path


def test_confirmation_audit_requires_frozen_three_seed_gated_lifecycle():
    text = Path("scripts/audit_mambatsv3_acf_infonce_local_cluster_confirmation.py").read_text(encoding="utf-8")
    for token in (
        "5510, 5511, 5512", "development_seed_excluded", "development-seed-excluded", "contrastive_cluster",
        "channel_coverage", "global_relation_active_fraction", "paired_delta_ci95",
        "selection_done_before_test", "training_performed", "guard failure",
        "active/null pretraining changed", "mamba_scan_count", "expected-source",
    ):
        assert token in text
