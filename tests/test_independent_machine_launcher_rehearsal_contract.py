from pathlib import Path


def test_launcher_rehearsal_audit_preserves_same_machine_boundary():
    text = Path("scripts/audit_independent_machine_launcher_rehearsal.py").read_text(encoding="utf-8")
    for token in ("launcher_rehearsal", "independent_machine", "test opened despite failed validation gate", "GPU guard failure"):
        assert token in text
