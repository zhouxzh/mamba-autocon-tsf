#!/usr/bin/env python3
"""Audit the validation-only frozen-core target-head adaptation probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = _load(args.root / "plan.json")
    if not plan.get("validation_only") or plan.get("test_metrics_produced"):
        raise ValueError("probe must be validation-only")
    rows = []
    for arm in ("active", "null"):
        payload = _load(args.root / "traffic_s5770_{}_frozen_core.json".format(arm))
        if payload.get("test_metrics_produced"):
            raise ValueError("probe opened test metrics")
        if not payload["config"].get("freeze_pretrained_core"):
            raise ValueError("checkpoint did not freeze transferred core")
        history = payload["history"]
        if not all(row["mamba_core_frozen"] for row in history):
            raise ValueError("transferred core was unfrozen")
        if not all(row["mamba_forecast_head_trainable"] for row in history):
            raise ValueError("target forecast head was not trainable")
        rows.append({"arm": arm, "validation_mse": float(payload["best_validation_mse"])})
    active, null = rows[0]["validation_mse"], rows[1]["validation_mse"]
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "active_vs_null_percent": (active / null - 1.0) * 100.0,
              "active_better": bool(active < null),
              "next_step": "No test protocol is opened by this one-seed adaptation probe."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
