#!/usr/bin/env python3
"""One-time read-only METR-LA H192 10% active/PatchTST test release."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006"
BASELINES = ROOT / "artifacts/exploratory/official_patchtst_metr_mamba1_10pct_20261006"
OUTPUT = ROOT / "artifacts/paper_final/mambatsv3_mamba1_metr10_frozen_test_release_20261006"
SEEDS = (5970, 5971, 5972)
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def guard(command: list[str], prefix: Path) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command], cwd=ROOT, check=True)

def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "frozen_before_test", "target": "METR-LA", "horizon": 192,
        "few_shot_fraction": .1, "seeds": list(SEEDS), "validation_only_selection_complete": True,
        "historical_public_test_access_caveat": True, "source_selection": "PEMS-BAY, PEMS07, PEMS08", "candidate": [], "official_patchtst": []}
    for seed in SEEDS:
        candidate = CANDIDATES / f"s{seed}_active_frozen_core.pt"
        patch = BASELINES / f"metr_la_h192_s{seed}_official_patchtst.pt"
        active_payload = torch.load(candidate, map_location="cpu", weights_only=False)
        patch_payload = torch.load(patch, map_location="cpu", weights_only=False)
        if any((active_payload.get("test_metrics_produced"), active_payload.get("test_mse") is not None, patch_payload.get("test_metrics_produced"), patch_payload.get("test_mse") is not None)):
            raise ValueError("checkpoint already opened test state")
        active_cfg, patch_cfg = active_payload["config"], patch_payload["config"]
        if (active_cfg.get("mamba_ts_mixer_type"), int(active_cfg["patch_len"]), int(active_cfg["patch_stride"]), float(active_cfg["few_shot_fraction"]), bool(active_cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, .1, True):
            raise ValueError(f"candidate contract mismatch: {seed}")
        if (int(patch_cfg["patch_len"]), int(patch_cfg["patch_stride"]), float(patch_cfg["few_shot_fraction"]), patch_payload["official_commit"]) != (24, 8, .1, COMMIT):
            raise ValueError(f"PatchTST contract mismatch: {seed}")
        manifest["candidate"].append({"seed": seed, "checkpoint": str(candidate), "sha256": digest(candidate)})
        manifest["official_patchtst"].append({"seed": seed, "checkpoint": str(patch), "sha256": digest(patch), "official_model_sha256": patch_payload["official_model_sha256"], "official_backbone_sha256": patch_payload["official_backbone_sha256"]})
    (OUTPUT / "manifest_before_test.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for seed in SEEDS:
        prefix = OUTPUT / f"s{seed}"
        guard([sys.executable, str(ROOT / "scripts/evaluate_lowdata_checkpoint_test.py"), "--dataset", "dataset/METR-LA/METR-LA.csv", "--checkpoint", str(CANDIDATES / f"s{seed}_active_frozen_core.pt"), "--output", str(prefix.with_name(prefix.name + "_active_test.json")), "--kind", "hybrid", "--batch-size", "4", "--seed", str(seed)], prefix.with_name(prefix.name + "_active"))
        guard([sys.executable, str(ROOT / "scripts/evaluate_official_patchtst_checkpoint_test.py"), "--dataset", "dataset/METR-LA/METR-LA.csv", "--checkpoint", str(BASELINES / f"metr_la_h192_s{seed}_official_patchtst.pt"), "--output", str(prefix.with_name(prefix.name + "_official_patchtst_test.json")), "--batch-size", "4", "--seed", str(seed)], prefix.with_name(prefix.name + "_official_patchtst"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
