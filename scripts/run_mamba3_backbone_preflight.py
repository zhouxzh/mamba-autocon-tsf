#!/usr/bin/env python3
"""Guarded, registered resource preflight for the official Mamba-3 block."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mamba3_backbone_preflight_20261008"
MAMBA1_PYTHON = Path("/home/zhong/anaconda3/envs/mamba/bin/python")
MAMBA3_PYTHON = Path("/home/zhong/anaconda3/envs/mamba3_probe_20261008/bin/python")
OFFICIAL_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


def guard(python: Path, command: list[str], stem: str) -> None:
    subprocess.run([
        str(python), str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(OUTPUT / (stem + "_telemetry.csv")),
        "--event-log", str(OUTPUT / (stem + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    if not MAMBA1_PYTHON.is_file() or not MAMBA3_PYTHON.is_file():
        raise FileNotFoundError("Mamba-1 or isolated Mamba-3 interpreter is missing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "registered_before_benchmark", "validation_only": True,
        "test_metrics_produced": False, "training_performed": False,
        "question": "Does the official Mamba-3 block meet the resource gate before any forecasting integration?",
        "official_repository": "https://github.com/state-spaces/mamba",
        "official_commit": OFFICIAL_COMMIT,
        "shape": [1300, 4, 24],
        "mamba1": {"d_state": 8, "d_conv": 2, "expand": 1},
        "mamba3": {"d_state": 16, "headdim": 24, "chunk_size": 32, "is_mimo": False},
        "measurement": {"warmup": 25, "iterations": 100, "dtype": "float32", "device_resident": True},
        "resource_gate": "stop before forecasting training if Mamba-3 median latency or peak memory exceeds 1.5x Mamba-1",
    }
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    common = ["scripts/benchmark_mamba_block_preflight.py", "--batch-channels", "1300", "--tokens", "4", "--d-model", "24", "--warmup", "25", "--iterations", "100"]
    guard(MAMBA1_PYTHON, [str(MAMBA1_PYTHON), *common, "--mixer", "mamba1", "--d-state", "8", "--output", str(OUTPUT / "mamba1.json")], "mamba1")
    guard(MAMBA3_PYTHON, [str(MAMBA3_PYTHON), "-W", "ignore", *common, "--mixer", "mamba3", "--d-state", "16", "--output", str(OUTPUT / "mamba3.json")], "mamba3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
