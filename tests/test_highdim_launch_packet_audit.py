"""Contract tests for the read-only high-dimensional launch packet audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_highdim_launch_packet.py"
PLAN = ROOT / "docs/research/mambatsv3_stride24_relation_auxiliary_highdim_test_release_draft_20261001.json"


def test_launch_packet_audit_rejects_stopped_stride24_packet(tmp_path):
    output = tmp_path / "packet.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(PLAN), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "review_packet_invalid"
    assert payload["authorized"] is False
    assert payload["launch_started"] is False
    assert any("status" in error for error in payload["errors"])


def test_launch_packet_audit_rejects_authorized_plan(tmp_path):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["execution"]["authorized"] = True
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "packet.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--plan", str(altered), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert any("authorized=false" in error for error in payload["errors"])
