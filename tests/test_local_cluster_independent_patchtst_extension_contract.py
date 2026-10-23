from pathlib import Path


def test_independent_patchtst_extension_keeps_lifecycle_and_scope_boundaries():
    runner = Path("scripts/run_official_patchtst_local_cluster_independent_extension.py").read_text(encoding="utf-8")
    auditor = Path("scripts/audit_local_cluster_independent_patchtst_extension.py").read_text(encoding="utf-8")
    for token in ("registered_before_training", "open-test-after-validation", "validation-selected checkpoints", "post-release official PatchTST descriptive extension", "78", "85", "pems_bay_h192_10", "metr_la_h192_10", "weather_h192_10"):
        assert token in runner
    for token in ("pooled_effect_reported", "post-release descriptive official-code extension", "test lifecycle mismatch", "official model lock mismatch", "max_temperature_c"):
        assert token in auditor
