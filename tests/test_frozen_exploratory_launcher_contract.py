"""Contracts for relocating exploratory output roots safely."""

import copy
from pathlib import Path

from scripts.run_frozen_exploratory_plan import (
    ROOT,
    _relocate_phase_output_roots,
)


def test_artifact_root_relocates_all_phase_roots_without_mutating_plan():
    plan = {
        "phases": [
            {"name": "alpha", "output_root": "artifacts/old/alpha"},
            {"name": "beta", "output_root": "artifacts/old/beta"},
        ]
    }
    original = copy.deepcopy(plan)
    relocated = _relocate_phase_output_roots(plan, Path("/tmp/frozen-smoke"))
    assert plan == original
    assert relocated["phases"][0]["output_root"] == "/tmp/frozen-smoke/alpha"
    assert relocated["phases"][1]["output_root"] == "/tmp/frozen-smoke/beta"


def test_artifact_root_preserves_nested_relative_phase_layout():
    plan = {
        "phases": [
            {"name": "one", "output_root": str(ROOT / "artifacts/old/one")},
            {"name": "two", "output_root": str(ROOT / "artifacts/old/group/two")},
        ]
    }
    relocated = _relocate_phase_output_roots(plan, Path("/tmp/frozen-smoke"))
    assert relocated["phases"][0]["output_root"] == "/tmp/frozen-smoke/one"
    assert relocated["phases"][1]["output_root"] == "/tmp/frozen-smoke/group/two"

