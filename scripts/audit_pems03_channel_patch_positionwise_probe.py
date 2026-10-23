#!/usr/bin/env python3
"""Read-only audit for PEMS03 positionwise Channel-Patch screen."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_positionwise_probe_20261007"
SEED = 6083


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    trajectory = load(OUTPUT / f"s{SEED}_trajectory_frozen_core.json"); positionwise = load(OUTPUT / f"s{SEED}_positionwise_frozen_core.json"); pretrain = load(OUTPUT / f"s{SEED}_positionwise_pretrain.json"); event = load(OUTPUT / f"s{SEED}_positionwise_pretrain_event.json")
    if any(item.get("test_metrics_produced") for item in (trajectory, positionwise, pretrain)): raise ValueError("test boundary")
    if pretrain["config"].get("channel_patch_similarity") != "positionwise" or "--channel-patch-similarity" not in event["command"]: raise ValueError("positionwise mode missing")
    if any("pems03" in item.lower() for index, item in enumerate(event["command"][:-1]) if event["command"][index] in {"--source", "--initial-checkpoint"}): raise ValueError("target leakage")
    if not all(item["channel_patch_relation_active_fraction"] > 0 for item in pretrain["history"]): raise ValueError("patch relation inactive")
    plain, candidate = float(trajectory["best_validation_mse"]), float(positionwise["best_validation_mse"])
    result = {"validation_only": True, "test_metrics_produced": False, "trajectory_mse": plain, "positionwise_mse": candidate, "positionwise_vs_trajectory_percent": (candidate / plain - 1) * 100, "screen_passes_expansion_margin": bool(candidate / plain - 1 <= -.0025), "target_excluded_from_source_and_initialization": True, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
