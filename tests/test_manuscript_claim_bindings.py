from pathlib import Path

from scripts.audit_manuscript_claim_bindings import audit


def test_audit_ignores_numbered_questions_and_requires_inline_binding(tmp_path: Path):
    manuscript = tmp_path / "draft.md"
    manuscript.write_text(
        "1. Does the model improve?\n"
        "The MSE improves by 2.0%.\n"
        "The MSE improves by 3.0%. [evidence:E001]\n",
        encoding="utf-8",
    )

    report = audit(manuscript)

    assert report["numeric_claim_lines"] == 2
    assert report["bound_lines"] == 1
    assert report["unresolved_lines"] == 1
    assert report["unresolved"][0]["line"] == 2


def test_internal_manuscript_numeric_claims_have_machine_readable_bindings():
    manuscript = Path(__file__).resolve().parents[1] / "docs/research/manuscript_draft_20260928.md"
    report = audit(manuscript)

    assert report["status"] == "bound"
    assert report["numeric_claim_lines"] == report["bound_lines"]
    assert report["unresolved_lines"] == 0
