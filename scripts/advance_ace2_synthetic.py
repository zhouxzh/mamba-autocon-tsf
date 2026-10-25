#!/usr/bin/env python3
"""Advance approved ACE2 synthetic phases only after the prior audit passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(input_dir: Path, expected: int) -> bool:
    analysis = input_dir / "analysis"
    subprocess.run([
        sys.executable, "scripts/analyze_ace2_simulation.py", "--input", str(input_dir),
        "--expected-tasks", str(expected), "--output", str(analysis),
    ], cwd=ROOT, check=True)
    return load(analysis / "audit.json").get("valid") is True


def launch_extension(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    guard = [
        sys.executable, "scripts/run_gpu_guarded.py", "--temperature-limit-c", "78",
        "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(output / "guard_telemetry.csv"), "--event-log", str(output / "guard_event.json"),
        "--", sys.executable, "scripts/ace2_simulation.py", "--phase", "S-extension",
        "--output", str(output), "--device", "cuda:0",
    ]
    with (output / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(guard, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (output / "simulation.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    monitor = subprocess.Popen([
        sys.executable, "scripts/monitor_ace2.py", "--output", str(output), "--pid", str(process.pid),
        "--interval-sec", "1800", "--temperature-limit-c", "78", "--fan-limit-percent", "85",
    ], cwd=ROOT, stdout=(output / "monitor.log").open("w", encoding="utf-8"), stderr=subprocess.STDOUT, start_new_session=True)
    (output / "monitor.pid").write_text(f"{monitor.pid}\n", encoding="utf-8")
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--poll-sec", type=float, default=60.0)
    args = parser.parse_args()
    core = args.core.resolve()
    extension = args.extension.resolve()
    status = {"status": "waiting_for_core", "core": str(core), "extension": str(extension)}
    (extension.parent / "program_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    event_path = core / "guard_event.json"
    while not event_path.is_file():
        time.sleep(args.poll_sec)
    event = load(event_path)
    if event.get("status") != "completed":
        status.update({"status": "core_guard_failed", "core_event": event})
        (extension.parent / "program_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        return 1
    if not audit(core, 14400):
        status.update({"status": "core_audit_failed"})
        (extension.parent / "program_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        return 1
    status.update({"status": "extension_running"})
    (extension.parent / "program_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    code = launch_extension(extension)
    event = load(extension / "guard_event.json") if (extension / "guard_event.json").is_file() else {"status": "missing_event"}
    valid = code == 0 and event.get("status") == "completed" and audit(extension, 43200)
    status.update({"status": "completed" if valid else "extension_failed", "extension_returncode": code, "extension_event": event})
    (extension.parent / "program_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
