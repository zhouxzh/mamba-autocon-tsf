from pathlib import Path


def test_manuscript_draft_audit_excludes_non_result_numeric_text():
    text = Path("scripts/audit_local_cluster_manuscript_draft.py").read_text(encoding="utf-8")
    for token in ("Mermaid", "metadata", "references", "numeric_prose_lines", "claim:C", "evidence:E"):
        assert token in text
