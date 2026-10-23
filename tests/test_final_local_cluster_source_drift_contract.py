from pathlib import Path


def test_source_drift_audit_checks_all_independent_manifests_without_training():
    text = Path("scripts/audit_final_local_cluster_source_drift.py").read_text(encoding="utf-8")
    for token in (
        "pems_active_null",
        "metr_active_null",
        "weather_active_null",
        "official_patchtst",
        "source_hashes_match",
        "training_performed",
        "new_test_reader_opened",
    ):
        assert token in text
