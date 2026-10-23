from pathlib import Path


def test_local_cluster_candidate_audit_covers_mechanism_and_lifecycle():
    text = Path("scripts/audit_mambatsv3_acf_infonce_local_cluster_candidate.py").read_text(encoding="utf-8")
    for token in (
        "forecast_contrastive", "channel_coverage", "contrastive_cluster",
        "global_relation_active_fraction", "best_validation_mse",
        "selection_done_before_test", "training_performed", "GPU guard",
        "mamba_scan_count", "relation_loss_included",
    ):
        assert token in text
