"""Regression coverage for the stride-24 high-dimensional AutoCon audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_stride24_highdim_autocon_training.py"


def test_stride24_highdim_autocon_audit_rejects_missing_phase(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root", str(tmp_path / "missing"),
            "--output-dir", str(tmp_path / "output"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "output" / "readout.json").exists()


def test_stride24_highdim_autocon_audit_rejects_test_artifact(tmp_path):
    root = tmp_path / "candidate"
    (root / "logs").mkdir(parents=True)
    (root / "provenance.json").write_text(
        json.dumps({"source_tree_sha256": "a" * 64}), encoding="utf-8"
    )
    (root / "phase_validation.json").write_text(
        json.dumps({
            "valid": True,
            "phase": "sparse_candidate_h96_validation",
            "expected_jobs": 6,
        }),
        encoding="utf-8",
    )
    result_dir = root / "results" / "job"
    result_dir.mkdir(parents=True)
    (result_dir / "metrics.npy").write_bytes(b"test artifact")
    checkpoint_root = root / "checkpoints"
    record = {
        "job_id": "job",
        "status": "ok",
        "source_tree_sha256_before": "a" * 64,
        "source_tree_sha256_after": "a" * 64,
        "expected_artifact": str(result_dir / "metrics.npy"),
        "command": ["python", "--checkpoints", str(checkpoint_root)],
    }
    (root / "logs" / "job_manifest.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root", str(root),
            "--output-dir", str(tmp_path / "output"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "test artifact found" in (result.stderr + result.stdout)
