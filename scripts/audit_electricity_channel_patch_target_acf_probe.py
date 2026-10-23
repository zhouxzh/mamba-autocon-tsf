#!/usr/bin/env python3
"""Read-only audit for target-prefix ACF adaptation probe."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_target_acf_probe_20261006"
REFERENCE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/s6033_channel_patch_frozen_core.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    candidate = load(OUTPUT / "s6033_target_acf.json")
    reference = load(REFERENCE)
    if candidate.get("test_metrics_produced") or reference.get("test_metrics_produced"):
        raise ValueError("test boundary violated")
    left = float(candidate["best_validation_mse"]); right = float(reference["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False,
              "target_acf_mse": left, "channel_patch_plain_mse": right,
              "target_acf_vs_plain_percent": (left / right - 1.0) * 100.0,
              "target_acf_better": bool(left < right), "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
