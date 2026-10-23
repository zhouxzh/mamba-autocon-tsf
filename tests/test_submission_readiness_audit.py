from pathlib import Path

from scripts.run_submission_readiness_audit import run


def test_audit_script_is_importable():
    root = Path(__file__).resolve().parents[1]
    code, _ = run(["python", "-c", "import scripts.run_submission_readiness_audit"], root)
    assert code == 0


def test_regression_summary_pattern_is_supported(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    log = tmp_path / "pytest.log"
    log.write_text("815 passed, 45 warnings in 41.32s\n", encoding="utf-8")
    code, output = run(
        [
            "python", "scripts/run_submission_readiness_audit.py",
            "--repo-root", ".",
            "--output-dir", str(tmp_path / "audit"),
            "--regression-log", str(log),
        ],
        root,
    )
    assert code == 0
    assert "checks_passed_human_approval_required" in output
