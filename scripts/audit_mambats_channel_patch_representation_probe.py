#!/usr/bin/env python3
"""Read-only audit for the channelwise patch-representation mechanism probe."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambats_channel_patch_representation_probe_20261006"
SEED = 6025


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for target in ("pems_bay", "metr_la"):
        records = {
            arm: _load(OUTPUT / target / f"s{SEED}_{arm}_frozen_core.json")
            for arm in ("null", "forecast", "channel_patch")
        }
        source = _load(OUTPUT / target / f"s{SEED}_channel_patch_pretrain.json")
        if any(record.get("test_metrics_produced") for record in (*records.values(), source)):
            raise ValueError("test boundary violated")
        config = source["config"]
        if float(config.get("channel_patch_relation_fraction", 0.0)) != .05:
            raise ValueError("channel patch fraction contract mismatch")
        telemetry = source["history"]
        if not telemetry or not any(float(row.get("channel_patch_relation_active_fraction", 0.0)) > 0 for row in telemetry):
            raise ValueError("channel patch objective had no active ACF anchors")
        values = {arm: float(record["best_validation_mse"]) for arm, record in records.items()}
        rows.append({"target": target, **values,
                     "channel_patch_vs_forecast_percent": (values["channel_patch"] / values["forecast"] - 1.0) * 100.0,
                     "channel_patch_vs_null_percent": (values["channel_patch"] / values["null"] - 1.0) * 100.0,
                     "channel_patch_better_than_forecast": values["channel_patch"] < values["forecast"]})
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "replication_eligible": bool(all(row["channel_patch_better_than_forecast"] for row in rows)),
              "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
