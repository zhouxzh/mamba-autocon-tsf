#!/usr/bin/env python3
"""One-time read-only PEMS-BAY H192 10% frozen active/PatchTST test release."""

from __future__ import annotations

import hashlib
import json
import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006"
BASELINES = ROOT / "artifacts/exploratory/official_patchtst_pems_mamba1_10pct_20261006"
OUTPUT = ROOT / "artifacts/paper_final/mambatsv3_mamba1_pems10_frozen_test_release_20261006"
SEEDS = (5960, 5961, 5962)
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "frozen_before_test", "target": "PEMS-BAY", "horizon": 192,
                "few_shot_fraction": .10, "seeds": list(SEEDS), "validation_only_selection_complete": True,
                "candidate": [], "official_patchtst": [], "historical_public_test_access_caveat": True}
    for seed in SEEDS:
        candidate = CANDIDATES / f"s{seed}_active_frozen_core.pt"
        patch = BASELINES / f"pems_bay_h192_s{seed}_official_patchtst.pt"
        active = torch.load(candidate, map_location="cpu", weights_only=False)
        baseline = torch.load(patch, map_location="cpu", weights_only=False)
        if any((active.get("test_metrics_produced"), active.get("test_mse") is not None,
                baseline.get("test_metrics_produced"), baseline.get("test_mse") is not None)):
            raise ValueError("a checkpoint has already opened its test state")
        cfg, patch_cfg = active["config"], baseline["config"]
        if (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]),
                float(cfg["few_shot_fraction"]), bool(cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, .1, True):
            raise ValueError(f"candidate contract mismatch: {seed}")
        if (int(patch_cfg["patch_len"]), int(patch_cfg["patch_stride"]), float(patch_cfg["few_shot_fraction"]),
                baseline["official_commit"]) != (24, 8, .1, COMMIT):
            raise ValueError(f"PatchTST contract mismatch: {seed}")
        manifest["candidate"].append({"seed": seed, "checkpoint": str(candidate), "sha256": _sha256(candidate), "validation_mse": float(active["best_validation_mse"])})
        manifest["official_patchtst"].append({"seed": seed, "checkpoint": str(patch), "sha256": _sha256(patch), "validation_mse": float(baseline["best_validation_mse"]), "official_model_sha256": baseline["official_model_sha256"], "official_backbone_sha256": baseline["official_backbone_sha256"]})
    (OUTPUT / "manifest_before_test.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        prefix = OUTPUT / f"s{seed}"
        _guard([
            sys.executable, str(ROOT / "scripts/evaluate_lowdata_checkpoint_test.py"),
            "--dataset", "dataset/PEMS-BAY/PEMS-BAY.csv", "--checkpoint", str(CANDIDATES / f"s{seed}_active_frozen_core.pt"),
            "--output", str(prefix.with_name(prefix.name + "_active_test.json")), "--kind", "hybrid", "--batch-size", "4", "--seed", str(seed),
        ], prefix.with_name(prefix.name + "_active"))
        _guard([
            sys.executable, str(ROOT / "scripts/evaluate_official_patchtst_checkpoint_test.py"),
            "--dataset", "dataset/PEMS-BAY/PEMS-BAY.csv", "--checkpoint", str(BASELINES / f"pems_bay_h192_s{seed}_official_patchtst.pt"),
            "--output", str(prefix.with_name(prefix.name + "_official_patchtst_test.json")), "--batch-size", "4", "--seed", str(seed),
        ], prefix.with_name(prefix.name + "_official_patchtst"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
