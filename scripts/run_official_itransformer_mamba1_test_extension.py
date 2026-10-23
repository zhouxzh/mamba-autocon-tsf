#!/usr/bin/env python3
"""Read held-out test blocks for frozen post-release iTransformer baselines."""

from __future__ import annotations

import hashlib
import json
import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "artifacts/exploratory/official_itransformer_mamba1_postrelease_extension_20261006"
RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006"
OUTPUT = RELEASE / "official_itransformer_extension"
SEEDS = (5930, 5931, 5932)
TARGETS = {
    "pems_bay": ("dataset/PEMS-BAY/PEMS-BAY.csv", "pems_bay_h192"),
    "metr_la": ("dataset/METR-LA/METR-LA.csv", "metr_la_h192"),
    "weather": ("dataset/weather/weather.csv", "weather_h192"),
}
COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"


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
    parser.add_argument("--target", choices=sorted(TARGETS), action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.target or sorted(TARGETS))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest_before_test.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_test":
            raise ValueError("existing iTransformer test manifest is not frozen")
    else:
        manifest = {"status": "frozen_before_test", "selection_after_candidate_release": True,
                    "purpose": "post-release external iTransformer baseline extension", "baselines": {}}
    for target, (_dataset, stem) in TARGETS.items():
        if target in manifest["baselines"]:
            continue
        manifest["baselines"][target] = []
        for seed in SEEDS:
            checkpoint = BASELINES / target / f"{stem}_s{seed}_official_itransformer.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("test_metrics_produced") or payload.get("test_mse") is not None:
                raise ValueError(f"iTransformer checkpoint test state already opened: {checkpoint}")
            cfg = payload["config"]
            if (payload.get("official_commit"), int(cfg["seq_len"]), int(cfg["pred_len"]),
                    int(cfg["d_model"]), int(cfg["e_layers"])) != (COMMIT, 96, 192, 24, 1):
                raise ValueError(f"iTransformer contract mismatch: {checkpoint}")
            manifest["baselines"][target].append({
                "checkpoint": str(checkpoint), "sha256": _sha256(checkpoint),
                "validation_mse": float(payload["best_validation_mse"]),
                "source_hashes": {key: payload[key] for key in ("official_model_sha256", "official_embed_sha256", "official_attention_sha256", "official_encoder_sha256")},
            })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for target in targets:
        dataset, stem = TARGETS[target]
        for seed in SEEDS:
            prefix = OUTPUT / target / f"s{seed}_official_itransformer"
            prefix.parent.mkdir(parents=True, exist_ok=True)
            event = prefix.with_name(prefix.name + "_event.json")
            output = prefix.with_name(prefix.name + "_test.json")
            if args.resume and output.is_file() and event.is_file():
                state = json.loads(event.read_text(encoding="utf-8"))
                if state.get("status") == "completed" and state.get("trigger") is None:
                    continue
            _guard([
                sys.executable, str(ROOT / "scripts/evaluate_official_itransformer_checkpoint_test.py"),
                "--dataset", dataset, "--checkpoint", str(BASELINES / target / f"{stem}_s{seed}_official_itransformer.pt"),
                "--output", str(output),
                "--batch-size", "4", "--seed", str(seed),
            ], prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
