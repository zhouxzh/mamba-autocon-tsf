#!/usr/bin/env python3
"""Audit validation-only 16/8 patch MambaTSv3 transfer probe."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_patch16_probe_20261003"
SEED = 5867


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan = _load(OUTPUT / "plan.json")
    active_pre, null_pre = _load(OUTPUT / "s5867_active_pretrain.json"), _load(OUTPUT / "s5867_null_pretrain.json")
    active, null = _load(OUTPUT / "s5867_active_frozen_core.json"), _load(OUTPUT / "s5867_null_frozen_core.json")
    if not plan.get("validation_only") or any(item.get("test_metrics_produced") for item in (active_pre, null_pre, active, null)):
        raise ValueError("probe must stay validation-only")
    for payload in (active_pre, null_pre, active, null):
        if int(payload["config"]["patch_len"]) != 16 or int(payload["config"]["patch_stride"]) != 8:
            raise ValueError("16/8 patch geometry mismatch")
    for payload in (active, null):
        if not payload["config"].get("freeze_pretrained_core"):
            raise ValueError("target core must remain frozen")
    value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "seed": SEED,
              "active": value, "exact_null": reference,
              "active_vs_null_percent": (value / reference - 1.0) * 100.0,
              "active_better": bool(value < reference),
              "next_step": "No test or external-baseline comparison is opened by one probe."}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
