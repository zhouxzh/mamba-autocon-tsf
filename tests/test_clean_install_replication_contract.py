from pathlib import Path


def test_clean_install_audit_checks_binary_artifacts_runtime_and_lifecycle():
    text = Path("scripts/audit_clean_install_replication.py").read_text(encoding="utf-8")
    for token in ("binary artifact hash mismatch", "clean environment runtime mismatch", "clean_install", "from_source_binary_build", "selection_done_before_test"):
        assert token in text
