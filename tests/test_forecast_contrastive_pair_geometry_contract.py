from pathlib import Path


def test_geometry_audit_uses_train_only_local_pairs_and_no_test_metrics():
    text = Path("scripts/audit_forecast_contrastive_pair_geometry.py").read_text(encoding="utf-8")
    for token in (
        "raw[:int(len(raw) * args.train_fraction)]", "channel_selected_lags",
        "contrastive_cluster", "batch_packing=\"local\"",
        "active_anchor_fraction_mean", "test_metrics_produced",
    ):
        assert token in text
