#!/usr/bin/env python3
"""Monitor an ACE2 simulation and GPU every 30 minutes.

The monitor records bounded summaries and terminates the guarded process group
if the configured thermal or fan limit is exceeded. The 5-second guard remains
the primary protection; this process is the requested long-interval audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def gpu_sample() -> dict[str, float]:
    fields = ("temperature.gpu", "fan.speed", "power.draw", "power.limit", "utilization.gpu", "memory.used", "memory.total")
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    values = [float(value.strip()) for value in result.stdout.strip().splitlines()[0].split(",")]
    if len(values) != len(fields):
        raise RuntimeError("unexpected nvidia-smi response")
    return dict(zip(fields, values))


def terminate_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def process_is_live(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


def ledger_summary(path: Path) -> dict[str, object]:
    result = {"tasks": 0, "completed": 0, "failed": 0, "last_task": None}
    if not path.is_file():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result["tasks"] += 1
            if row.get("status") == "COMPLETED":
                result["completed"] += 1
            else:
                result["failed"] += 1
            result["last_task"] = row.get("task_key")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval-sec", type=float, default=1800.0)
    parser.add_argument("--temperature-limit-c", type=float, default=78.0)
    parser.add_argument("--fan-limit-percent", type=float, default=85.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output / "results.jsonl"
    monitor_path = args.output / "monitor.jsonl"
    while True:
        if not process_is_live(args.pid):
            break
        sample = gpu_sample()
        summary = ledger_summary(ledger)
        trigger = None
        if sample["temperature.gpu"] > args.temperature_limit_c:
            trigger = f"temperature {sample['temperature.gpu']:.1f}C > {args.temperature_limit_c:.1f}C"
        elif sample["fan.speed"] > args.fan_limit_percent:
            trigger = f"fan {sample['fan.speed']:.1f}% > {args.fan_limit_percent:.1f}%"
        row = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **sample, **summary, "guard_trigger": trigger}
        with monitor_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if trigger:
            (args.output / "monitor_event.json").write_text(json.dumps({"status": "guard_triggered", "trigger": trigger, "row": row}, indent=2) + "\n", encoding="utf-8")
            terminate_group(args.pid)
            return 1
        time.sleep(args.interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
