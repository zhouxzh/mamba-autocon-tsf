#!/usr/bin/env python3
"""Read locked strict-transfer checkpoints on their chronological test blocks.

This is a registered, read-only post-validation evaluation. It neither trains
nor selects models. Existing historical experiments have already used these
public dataset test partitions, so its results are protocol-specific
post-validation evidence, not a claim of globally pristine test isolation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_strict_frozen_core_postvalidation_test_20261003"
MATRICES = {
    "pems_bay_10": {
        "root": ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_probe_20261003",
        "dataset": ROOT / "dataset/PEMS-BAY/PEMS-BAY.csv", "seeds": (5800, 5801, 5802),
    },
    "pems_bay_25": {
        "root": ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_25_validation_20261003",
        "dataset": ROOT / "dataset/PEMS-BAY/PEMS-BAY.csv", "seeds": (5810, 5811, 5812),
    },
    "metr_la_10": {
        "root": ROOT / "artifacts/exploratory/mambatsv3_metr_frozen_core_validation_20261003",
        "dataset": ROOT / "dataset/METR-LA/METR-LA.csv", "seeds": (5820, 5821, 5822),
    },
    "weather_10": {
        "root": ROOT / "artifacts/exploratory/mambatsv3_weather_frozen_core_validation_20261003",
        "dataset": ROOT / "dataset/weather/weather.csv", "seeds": (5830, 5831, 5832),
    },
}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_test_read", "training_performed": False,
        "selection_performed": False, "checkpoint_mutation": False,
        "test_metrics_produced": True,
        "scope": "protocol-specific post-validation read-only test evaluation; not globally pristine due prior public-dataset test usage",
        "matrices": {name: {"checkpoint_root": str(spec["root"]), "dataset": str(spec["dataset"]), "seeds": list(spec["seeds"])} for name, spec in MATRICES.items()},
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for name, spec in MATRICES.items():
        for seed in spec["seeds"]:
            for arm in ("active", "null"):
                checkpoint = spec["root"] / "s{}_{}_frozen_core.pt".format(seed, arm)
                output = OUTPUT / "{}_s{}_{}.json".format(name, seed, arm)
                prefix = output.with_suffix("")
                command = [
                    sys.executable, "scripts/evaluate_lowdata_checkpoint_test.py",
                    "--dataset", str(spec["dataset"]), "--checkpoint", str(checkpoint),
                    "--output", str(output), "--kind", "hybrid", "--batch-size", "4", "--seed", str(seed),
                ]
                subprocess.run([
                    sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
                    "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
                    "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
                    "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
                ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
