from pathlib import Path


def test_clean_environment_replication_audit_checks_hashes_and_lifecycle():
    text = Path("scripts/audit_clean_environment_replication.py").read_text(encoding="utf-8")
    for token in ("source/data hash mismatch", "same_machine_independent_interpreter", "independent_machine", "selection_done_before_test", "GPU guard failure"):
        assert token in text
