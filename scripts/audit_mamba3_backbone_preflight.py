#!/usr/bin/env python3
"""Read-only resource gate for official Mamba-3 backbone preflight."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mamba3_backbone_preflight_20261008"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan, mamba1, mamba3 = (load(OUTPUT / name) for name in ("plan.json", "mamba1.json", "mamba3.json"))
    for name in ("mamba1", "mamba3"):
        event = load(OUTPUT / (name + "_event.json"))
        if event.get("status") != "completed" or event.get("trigger") is not None:
            raise ValueError("GPU guard did not complete: {}".format(name))
    if not all(value["finite"] for value in (mamba1, mamba3)):
        raise ValueError("non-finite preflight output")
    if mamba1["shape"] != mamba3["shape"] or mamba1["shape"] != plan["shape"]:
        raise ValueError("shape mismatch")
    latency_ratio = float(mamba3["median_ms"] / mamba1["median_ms"])
    memory_ratio = float(mamba3["peak_memory_bytes"] / mamba1["peak_memory_bytes"])
    result = {
        "validation_only": True, "test_metrics_produced": False, "training_performed": False,
        "mamba1": mamba1, "mamba3": mamba3,
        "mamba3_vs_mamba1_latency_ratio": latency_ratio,
        "mamba3_vs_mamba1_memory_ratio": memory_ratio,
        "resource_gate_passes": bool(latency_ratio <= 1.5 and memory_ratio <= 1.5),
        "decision": "eligible_for_forecasting_integration" if latency_ratio <= 1.5 and memory_ratio <= 1.5 else "stopped_before_forecasting_training",
        "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
