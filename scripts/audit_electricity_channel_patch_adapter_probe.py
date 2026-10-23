#!/usr/bin/env python3
"""Read-only audit for channel-patch plus rank-8 target adapter."""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/electricity_channel_patch_adapter_probe_20261006"
REFERENCE = ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/s6033_channel_patch_frozen_core.json"

def load(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def main() -> int:
    candidate = load(OUTPUT / "s6033_channel_patch_adapter.json"); plain = load(REFERENCE)
    if candidate.get("test_metrics_produced") or plain.get("test_metrics_produced"): raise ValueError("test boundary violated")
    left, right = float(candidate["best_validation_mse"]), float(plain["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "adapter_mse": left, "plain_mse": right,
              "adapter_vs_plain_percent": (left / right - 1.0) * 100.0, "adapter_better": left < right, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
