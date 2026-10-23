#!/usr/bin/env python3
"""Audit validation-only discriminative-LR staged adaptation probe."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_pems_discriminative_lr_probe_20261003"
SEED = 5866


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    plan = _load(OUTPUT / "plan.json")
    active_pre, null_pre = _load(OUTPUT / "s5866_active_pretrain.json"), _load(OUTPUT / "s5866_null_pretrain.json")
    active, null = _load(OUTPUT / "s5866_active_staged_lr.json"), _load(OUTPUT / "s5866_null_staged_lr.json")
    if not plan.get("validation_only") or any(item.get("test_metrics_produced") for item in (active_pre, null_pre, active, null)):
        raise ValueError("probe must stay validation-only")
    for payload in (active, null):
        history = payload["history"]
        if history[0]["mamba_frozen"] is not True or any(row["mamba_frozen"] for row in history[1:]):
            raise ValueError("warm-up contract failed")
        if float(payload["config"]["mamba_core_learning_rate"]) != 0.0001:
            raise ValueError("discriminative core LR mismatch")
    value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "seed": SEED,
              "active": value, "exact_null": reference,
              "active_vs_null_percent": (value / reference - 1.0) * 100.0,
              "active_better": bool(value < reference),
              "next_step": "One probe cannot open test or establish strong-baseline parity."}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
