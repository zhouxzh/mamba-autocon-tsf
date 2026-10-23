#!/usr/bin/env python3
"""Frozen read-only test extension for validated channel-patch SSL checkpoints.

This is deliberately an audit extension, not a new selection stage.  All
candidate, forecast-only, null, and official PatchTST checkpoints are hashed
into a manifest before a test reader is invoked.  The workspace has historical
public-test access, so results carry that explicit caveat.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/paper_final/channel_patch_frozen_test_extension_20261006"
TARGETS = {
    "pems_bay": {
        "dataset": "dataset/PEMS-BAY/PEMS-BAY.csv", "stem": "pems_bay_h192",
        "candidate": ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006/pems_bay",
        "baseline": ROOT / "artifacts/exploratory/official_patchtst_channel_patch_replication_20261006/pems_bay",
        "seeds": (6026, 6027, 6028),
    },
    "metr_la": {
        "dataset": "dataset/METR-LA/METR-LA.csv", "stem": "metr_la_h192",
        "candidate": ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006/metr_la",
        "baseline": ROOT / "artifacts/exploratory/official_patchtst_channel_patch_replication_20261006/metr_la",
        "seeds": (6029, 6030, 6031),
    },
    "weather": {
        "dataset": "dataset/weather/weather.csv", "stem": "weather_h192",
        "candidate": ROOT / "artifacts/exploratory/weather_channel_patch_replication_20261006",
        "controls": ROOT / "artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006",
        "control_arm_names": {"forecast": "active", "null": "null"},
        "baseline": ROOT / "artifacts/exploratory/official_patchtst_weather_mamba1_10pct_independent_replication_20261006",
        "seeds": (5991, 5992, 5993, 5994, 5995),
    },
}
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def payload(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("test_metrics_produced") or value.get("test_mse") is not None:
        raise ValueError(f"checkpoint has already opened test state: {path}")
    return value


def check_candidate(path: Path, expected_arm: str) -> dict:
    value = payload(path); cfg = value["config"]
    contract = (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]), float(cfg["few_shot_fraction"]), bool(cfg["freeze_pretrained_core"]))
    if contract != ("mamba1", 24, 8, .1, True):
        raise ValueError(f"candidate contract mismatch: {path}: {contract}")
    return {"arm": expected_arm, "checkpoint": str(path), "sha256": digest(path), "validation_mse": float(value["best_validation_mse"])}


def check_patch(path: Path) -> dict:
    value = payload(path); cfg = value["config"]
    contract = (int(cfg["patch_len"]), int(cfg["patch_stride"]), float(cfg["few_shot_fraction"]), value["official_commit"])
    if contract != (24, 8, .1, COMMIT):
        raise ValueError(f"official PatchTST contract mismatch: {path}: {contract}")
    return {"checkpoint": str(path), "sha256": digest(path), "validation_mse": float(value["best_validation_mse"]), "official_model_sha256": value["official_model_sha256"], "official_backbone_sha256": value["official_backbone_sha256"]}


def candidate_checkpoint(spec: dict, seed: int, arm: str) -> Path:
    if arm == "channel_patch":
        return spec["candidate"] / f"s{seed}_{arm}_frozen_core.pt"
    root = spec.get("controls", spec["candidate"])
    stored_arm = spec.get("control_arm_names", {}).get(arm, arm)
    return root / f"s{seed}_{stored_arm}_frozen_core.pt"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=None)
    args = parser.parse_args(); targets = tuple(args.target or sorted(TARGETS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "frozen_before_test", "targets": list(targets), "validation_only_selection_complete": True,
                "historical_public_test_access_caveat": True, "scope": "post-validation mechanism extension; test results cannot alter training", "candidate": {}, "official_patchtst": {}}
    for target in targets:
        spec = TARGETS[target]; manifest["candidate"][target] = []; manifest["official_patchtst"][target] = []
        for seed in spec["seeds"]:
            for arm in ("channel_patch", "forecast", "null"):
                checkpoint = candidate_checkpoint(spec, seed, arm)
                manifest["candidate"][target].append({"seed": seed, **check_candidate(checkpoint, arm)})
            baseline = spec["baseline"] / f"{spec['stem']}_s{seed}_official_patchtst.pt"
            manifest["official_patchtst"][target].append({"seed": seed, **check_patch(baseline)})
    (OUTPUT / "manifest_before_test.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        spec = TARGETS[target]
        target_output = OUTPUT / target; target_output.mkdir(parents=True, exist_ok=True)
        for seed in spec["seeds"]:
            for arm in ("channel_patch", "forecast", "null"):
                prefix = target_output / f"s{seed}_{arm}"
                guard([sys.executable, str(ROOT / "scripts/evaluate_lowdata_checkpoint_test.py"), "--dataset", spec["dataset"], "--checkpoint", str(candidate_checkpoint(spec, seed, arm)), "--output", str(prefix.with_name(prefix.name + "_test.json")), "--kind", "hybrid", "--batch-size", "4", "--seed", str(seed)], prefix)
            prefix = target_output / f"s{seed}_official_patchtst"
            guard([sys.executable, str(ROOT / "scripts/evaluate_official_patchtst_checkpoint_test.py"), "--dataset", spec["dataset"], "--checkpoint", str(spec["baseline"] / f"{spec['stem']}_s{seed}_official_patchtst.pt"), "--output", str(prefix.with_name(prefix.name + "_test.json")), "--batch-size", "4", "--seed", str(seed)], prefix)
    return 0


if __name__ == "__main__": raise SystemExit(main())
