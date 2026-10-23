#!/usr/bin/env python3
"""One-time read-only chronological test release for frozen Mamba-1 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/exploratory/mambatsv3_mamba1_crossdomain_replication_20261006"
BASELINES = ROOT / "artifacts/exploratory/official_patchtst_mamba1_crossdomain_replication_20261006"
OUTPUT = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006"
SEEDS = (5930, 5931, 5932)
TARGETS = {
    "pems_bay": ("dataset/PEMS-BAY/PEMS-BAY.csv", "pems_bay_h192"),
    "metr_la": ("dataset/METR-LA/METR-LA.csv", "metr_la_h192"),
    "weather": ("dataset/weather/weather.csv", "weather_h192"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _check_candidate(path: Path) -> dict:
    payload = _payload(path)
    cfg = payload["config"]
    if payload.get("test_metrics_produced") or payload.get("test_mse") is not None:
        raise ValueError(f"candidate test state already opened: {path}")
    required = (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]),
                int(cfg["patch_stride"]), bool(cfg["freeze_pretrained_core"]))
    if required != ("mamba1", 24, 8, True):
        raise ValueError(f"candidate contract mismatch: {path}")
    return {"checkpoint": str(path), "sha256": _sha256(path), "validation_mse": float(payload["best_validation_mse"])}


def _check_baseline(path: Path) -> dict:
    payload = _payload(path)
    cfg = payload["config"]
    if payload.get("test_metrics_produced") or payload.get("test_mse") is not None:
        raise ValueError(f"baseline test state already opened: {path}")
    if (int(cfg["patch_len"]), int(cfg["patch_stride"])) != (24, 8):
        raise ValueError(f"baseline geometry mismatch: {path}")
    return {"checkpoint": str(path), "sha256": _sha256(path), "validation_mse": float(payload["best_validation_mse"]),
            "official_commit": payload["official_commit"], "official_model_sha256": payload["official_model_sha256"],
            "official_backbone_sha256": payload["official_backbone_sha256"]}


def _guard(command: list[str], prefix: Path) -> None:
    subprocess.run([
        sys.executable, str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_telemetry.csv")),
        "--event-log", str(prefix.with_name(prefix.name + "_event.json")), "--", *command,
    ], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=None)
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "frozen_before_test", "targets": list(targets), "seeds": list(SEEDS),
                "validation_only_selection_complete": True, "candidate": {}, "official_patchtst": {},
                "historical_public_test_access_caveat": True}
    for target in targets:
        dataset, stem = TARGETS[target]
        manifest["candidate"][target] = []
        manifest["official_patchtst"][target] = []
        for seed in SEEDS:
            candidate = CANDIDATES / target / f"s{seed}_active_frozen_core.pt"
            baseline = BASELINES / target / f"{stem}_s{seed}_official_patchtst.pt"
            manifest["candidate"][target].append(_check_candidate(candidate))
            manifest["official_patchtst"][target].append(_check_baseline(baseline))
    (OUTPUT / "manifest_before_test.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        dataset, stem = TARGETS[target]
        for seed in SEEDS:
            prefix = OUTPUT / target / f"s{seed}"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            _guard([
                sys.executable, str(ROOT / "scripts/evaluate_lowdata_checkpoint_test.py"),
                "--dataset", dataset, "--checkpoint", str(CANDIDATES / target / f"s{seed}_active_frozen_core.pt"),
                "--output", str(prefix.with_name(prefix.name + "_active_test.json")),
                "--kind", "hybrid", "--batch-size", "4", "--seed", str(seed),
            ], prefix.with_name(prefix.name + "_active"))
            _guard([
                sys.executable, str(ROOT / "scripts/evaluate_official_patchtst_checkpoint_test.py"),
                "--dataset", dataset, "--checkpoint", str(BASELINES / target / f"{stem}_s{seed}_official_patchtst.pt"),
                "--output", str(prefix.with_name(prefix.name + "_official_patchtst_test.json")),
                "--batch-size", "4", "--seed", str(seed),
            ], prefix.with_name(prefix.name + "_official_patchtst"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
