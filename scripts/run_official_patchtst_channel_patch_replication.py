#!/usr/bin/env python3
"""Train official PatchTST on the locked channel-patch replication seeds only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/official_patchtst_channel_patch_replication_20261006"
REFERENCE = {
    "pems_bay": (ROOT / "artifacts/exploratory/official_patchtst_pems_mamba1_10pct_20261006", "pems_bay_h192_s5960_official_patchtst_event.json", (6026, 6027, 6028)),
    "metr_la": (ROOT / "artifacts/exploratory/official_patchtst_metr_mamba1_10pct_20261006", "metr_la_h192_s5970_official_patchtst_event.json", (6029, 6030, 6031)),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def replace(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command); out[out.index(flag) + 1] = value; return out


def guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def complete(path: Path) -> bool:
    return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(REFERENCE), action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); targets = tuple(args.target or sorted(REFERENCE))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps({
        "status": "registered_before_training", "validation_only": True, "test_metrics_produced": False,
        "question": "Same-seed official PatchTST comparator for channel-patch ACF SSL replication",
        "targets": {name: {"seeds": list(REFERENCE[name][2])} for name in targets},
        "official": "PatchTST commit 204c21efe0b39603ad6e2ca640ef5896646ab1a9",
        "contract": "official model with local chronological 60/20/20 adapter; 10% train labels; no test evaluation",
    }, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        reference_root, event_name, seeds = REFERENCE[target]
        event = reference_root / event_name
        for seed in seeds:
            prefix = OUTPUT / target / f"{target}_h192_s{seed}_official_patchtst"
            command = load(event)["command"]; command[0] = sys.executable
            command = replace(command, "--output", str(prefix.with_suffix(".pt")))
            command = replace(command, "--seed", str(seed))
            if "--evaluate-test" in command:
                raise ValueError("official baseline cannot read test rows")
            event_path = prefix.with_name(prefix.name + "_event.json")
            if not (args.resume and prefix.with_suffix(".pt").is_file() and complete(event_path)):
                prefix.parent.mkdir(parents=True, exist_ok=True)
                guard(command, prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
