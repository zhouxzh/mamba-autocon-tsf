#!/usr/bin/env python3
"""Read paired null checkpoints after the frozen active/PatchTST test release.

This does not select architecture or hyperparameters. It reads the matched
pre-registered null controls solely to quantify the held-out causal contrast.
"""

from __future__ import annotations

import hashlib
import json
import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/exploratory/mambatsv3_mamba1_crossdomain_replication_20261006"
RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006"
SEEDS = (5930, 5931, 5932)
TARGETS = {
    "pems_bay": "dataset/PEMS-BAY/PEMS-BAY.csv",
    "metr_la": "dataset/METR-LA/METR-LA.csv",
    "weather": "dataset/weather/weather.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    output = RELEASE / "paired_null_readout"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "paired_controls_frozen_before_readout", "selection_after_primary_test": False,
                "purpose": "held-out active/null causal contrast", "null_checkpoints": {}}
    for target in TARGETS:
        manifest["null_checkpoints"][target] = []
        for seed in SEEDS:
            checkpoint = CANDIDATES / target / f"s{seed}_null_frozen_core.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("test_metrics_produced") or payload.get("test_mse") is not None:
                raise ValueError(f"null checkpoint test state already opened: {checkpoint}")
            cfg = payload["config"]
            if (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]),
                    bool(cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, True):
                raise ValueError(f"null contract mismatch: {checkpoint}")
            manifest["null_checkpoints"][target].append({"checkpoint": str(checkpoint), "sha256": _sha256(checkpoint)})
    (output / "manifest_before_null_test.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for target, dataset in TARGETS.items():
        for seed in SEEDS:
            prefix = output / target / f"s{seed}_null"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            _guard([
                sys.executable, str(ROOT / "scripts/evaluate_lowdata_checkpoint_test.py"),
                "--dataset", dataset, "--checkpoint", str(CANDIDATES / target / f"s{seed}_null_frozen_core.pt"),
                "--output", str(prefix.with_name(prefix.name + "_test.json")),
                "--kind", "hybrid", "--batch-size", "4", "--seed", str(seed),
            ], prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
