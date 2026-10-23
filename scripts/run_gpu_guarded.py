#!/usr/bin/env python3
"""Launch one command with GPU thermal/fan guardrails and telemetry logging."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import time
from pathlib import Path


FIELDS = (
    "temperature.gpu", "fan.speed", "power.draw", "power.limit",
    "utilization.gpu", "memory.used", "memory.total",
)


def _sample() -> dict[str, float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    values = [value.strip() for value in result.stdout.strip().splitlines()[0].split(",")]
    if len(values) != len(FIELDS):
        raise RuntimeError("unexpected nvidia-smi telemetry response")
    return {field: float(value) for field, value in zip(FIELDS, values)}


def _terminate_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-limit-c", type=float, default=78.0)
    parser.add_argument("--fan-limit-percent", type=float, default=85.0)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--telemetry-csv", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--":
        raise ValueError("pass the guarded command after --")
    if not 0 < args.temperature_limit_c <= 95:
        raise ValueError("temperature limit must be in (0, 95]")
    if not 0 < args.fan_limit_percent <= 100:
        raise ValueError("fan limit must be in (0, 100]")
    if args.interval_sec <= 0:
        raise ValueError("interval must be positive")

    command = args.command[1:]
    args.telemetry_csv.parent.mkdir(parents=True, exist_ok=True)
    args.event_log.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "status": "running",
        "temperature_limit_c": args.temperature_limit_c,
        "fan_limit_percent": args.fan_limit_percent,
        "interval_sec": args.interval_sec,
        "command": command,
        "trigger": None,
        "returncode": None,
    }
    process = subprocess.Popen(command, start_new_session=True)
    with args.telemetry_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp_utc",) + FIELDS + ("guard_trigger",))
        writer.writeheader()
        while process.poll() is None:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                sample = _sample()
            except Exception as exc:
                event["status"] = "monitor_error"
                event["trigger"] = "telemetry_error: {}".format(exc)
                _terminate_group(process)
                break
            trigger = None
            if sample["temperature.gpu"] > args.temperature_limit_c:
                trigger = "temperature {:.1f}C > {:.1f}C".format(sample["temperature.gpu"], args.temperature_limit_c)
            elif sample["fan.speed"] > args.fan_limit_percent:
                trigger = "fan {:.1f}% > {:.1f}%".format(sample["fan.speed"], args.fan_limit_percent)
            writer.writerow({"timestamp_utc": timestamp, **sample, "guard_trigger": trigger or ""})
            handle.flush()
            if trigger:
                event["status"] = "guard_triggered"
                event["trigger"] = trigger
                _terminate_group(process)
                break
            time.sleep(args.interval_sec)
    event["returncode"] = process.poll()
    if event["status"] == "running":
        event["status"] = "completed" if event["returncode"] == 0 else "command_failed"
    import json
    args.event_log.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(event, sort_keys=True))
    return 0 if event["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
