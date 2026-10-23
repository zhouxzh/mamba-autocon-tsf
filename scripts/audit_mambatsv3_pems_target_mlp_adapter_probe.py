#!/usr/bin/env python3
"""Audit validation-only target temporal-MLP adapter probe."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_target_mlp_adapter_probe_20261003"
SEED = 5840


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan = _load(OUTPUT / "plan.json")
    active, null = _load(OUTPUT / "s5840_active_target_mlp.json"), _load(OUTPUT / "s5840_null_target_mlp.json")
    if not plan.get("validation_only") or any(item.get("test_metrics_produced") for item in (active, null)):
        raise ValueError("probe must stay validation-only")
    for payload in (active, null):
        if not payload["config"].get("mamba_ts_target_mlp_adapter") or not payload["config"].get("freeze_pretrained_core"):
            raise ValueError("target temporal adapter contract missing")
        if not all(row["mamba_core_frozen"] and row["mamba_target_mlp_adapter_trainable"] for row in payload["history"]):
            raise ValueError("temporal target adapter was not trainable with frozen core")
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
