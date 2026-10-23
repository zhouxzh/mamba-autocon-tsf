"""Basic input validation for the stride-24 source-equivalence audit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_mambats_default_path_equivalence.py"


def test_equivalence_audit_rejects_missing_snapshot(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot-root", str(tmp_path / "missing"),
            "--output", str(tmp_path / "audit.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "audit.json").exists()
