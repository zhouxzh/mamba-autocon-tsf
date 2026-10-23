from pathlib import Path


def test_independent_program_audit_keeps_domains_separate():
    text = Path("scripts/audit_local_cluster_independent_replication_program.py").read_text(encoding="utf-8")
    for token in ("pems_bay_h192_10", "metr_la_h192_10", "weather_h192_10", "pooled_effect_reported", "not pooled", "ratio_upper_1se", "human_gates"):
        assert token in text
