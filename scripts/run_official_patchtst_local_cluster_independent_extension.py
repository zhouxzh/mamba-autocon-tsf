#!/usr/bin/env python3
"""Run a post-release official PatchTST extension on the independent host.

The active/null local-cluster tests are already frozen.  This runner therefore
adds a descriptive external-code comparator only: it cannot select a new
candidate or alter the mechanism comparison.  Training is validation-only
until ``--open-test-after-validation`` explicitly invokes the read-only
checkpoint evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_independent_official_patchtst_20261008"
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
MODEL_HASH = "49d8bb865e1226d6338842f4b33c4bc6992cefb048dc2a4a5a8c410ca708586b"
BACKBONE_HASH = "df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26"
TARGETS = {
    "pems_bay_h192_10": {"target": "PEMS-BAY", "dataset": "dataset/PEMS-BAY/PEMS-BAY.csv", "seeds": (6130, 6131, 6132)},
    "metr_la_h192_10": {"target": "METR-LA", "dataset": "dataset/METR-LA/METR-LA.csv", "seeds": (6140, 6141, 6142)},
    "weather_h192_10": {"target": "Weather", "dataset": "dataset/weather/weather.csv", "seeds": (6150, 6151, 6152)},
}
TRACKED = (
    "scripts/official_patchtst_adapter.py",
    "scripts/evaluate_official_patchtst_checkpoint_test.py",
    "scripts/run_gpu_guarded.py",
    "data_provider/data_loader.py",
    "data_provider/__init__.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def complete(path: Path) -> bool:
    return path.is_file() and load(path).get("status") == "completed" and load(path).get("trigger") is None


def guarded(python: Path, command: list[str], prefix: Path, stage: str) -> None:
    subprocess.run([
        str(python), str(ROOT / "scripts/run_gpu_guarded.py"),
        "--temperature-limit-c", "78", "--fan-limit-percent", "85", "--interval-sec", "5",
        "--telemetry-csv", str(prefix.with_name(prefix.name + "_{}_telemetry.csv".format(stage))),
        "--event-log", str(prefix.with_name(prefix.name + "_{}_event.json".format(stage))),
        "--", *command,
    ], cwd=ROOT, check=True)


def training_command(python: Path, official_root: Path, dataset: str, output: Path, seed: int) -> list[str]:
    return [
        str(python), "scripts/official_patchtst_adapter.py",
        "--official-root", str(official_root), "--official-commit", COMMIT,
        "--dataset", dataset, "--output", str(output), "--pred-len", "192",
        "--few-shot-fraction", ".1", "--train-fraction", ".6", "--val-fraction", ".2",
        "--batch-size", "4", "--epochs", "5", "--max-train-windows", "64",
        "--d-model", "24", "--d-ff", "48", "--n-heads", "4", "--e-layers", "1",
        "--patch-len", "16", "--patch-stride", "8", "--seed", str(seed),
    ]


def test_command(python: Path, dataset: str, checkpoint: Path, output: Path, seed: int) -> list[str]:
    return [
        str(python), "scripts/evaluate_official_patchtst_checkpoint_test.py",
        "--dataset", dataset, "--checkpoint", str(checkpoint), "--output", str(output),
        "--batch-size", "4", "--seed", str(seed),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--open-test-after-validation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    python, official_root, output = args.python.resolve(), args.official_root.resolve(), args.output.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    if digest(official_root / "models/PatchTST.py") != MODEL_HASH or digest(official_root / "layers/PatchTST_backbone.py") != BACKBONE_HASH:
        raise ValueError("official PatchTST source lock mismatch")
    required = [ROOT / item for item in TRACKED] + [ROOT / spec["dataset"] for spec in TARGETS.values()]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError("missing protocol inputs: {}".format(", ".join(missing)))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "packet_only" if args.manifest_only else "registered_before_training",
        "machine_id": args.machine_id,
        "independent_machine": "unverified_operator_claim",
        "comparison_scope": "post-release official PatchTST descriptive extension; does not modify frozen active/null mechanism tests",
        "test_lifecycle": "validation-selected checkpoints only; explicit read-only test phase after all validation jobs complete",
        "test_open_requested": bool(args.open_test_after_validation),
        "official_repository": "yuqinie98/PatchTST",
        "official_commit": COMMIT,
        "official_model_sha256": MODEL_HASH,
        "official_backbone_sha256": BACKBONE_HASH,
        "official_root": str(official_root),
        "thermal_guard": {"temperature_limit_c": 78, "fan_limit_percent": 85, "interval_sec": 5},
        "contract": {"split": "60/20/20 chronological", "few_shot_fraction": .1, "seq_len": 96, "horizon": 192, "batch_size": 4, "epochs": 5, "max_train_windows": 64, "patch": "16/8", "d_model": 24, "d_ff": 48},
        "targets": {key: {**spec, "seeds": list(spec["seeds"]), "dataset_sha256": digest(ROOT / spec["dataset"])} for key, spec in TARGETS.items()},
        "source_hashes": {item: digest(ROOT / item) for item in TRACKED},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.manifest_only:
        return 0
    for key, spec in TARGETS.items():
        for seed in spec["seeds"]:
            prefix = output / key / "s{}_official_patchtst".format(seed)
            checkpoint = prefix.with_suffix(".pt")
            if not (args.resume and checkpoint.is_file() and complete(prefix.with_name(prefix.name + "_validation_event.json"))):
                guarded(python, training_command(python, official_root, spec["dataset"], checkpoint, seed), prefix, "validation")
    validation = []
    for key, spec in TARGETS.items():
        for seed in spec["seeds"]:
            prefix = output / key / "s{}_official_patchtst".format(seed)
            payload = load(prefix.with_suffix(".json"))
            valid = (not payload.get("test_metrics_produced") and math.isfinite(float(payload["best_validation_mse"])) and payload.get("official_commit") == COMMIT and payload.get("official_model_sha256") == MODEL_HASH and payload.get("official_backbone_sha256") == BACKBONE_HASH)
            validation.append({"target_key": key, "seed": seed, "valid": valid, "best_validation_mse": float(payload["best_validation_mse"])})
    passed = all(row["valid"] for row in validation)
    (output / "validation_gate.json").write_text(json.dumps({"passes": passed, "rows": validation, "test_metrics_produced": False}, indent=2) + "\n", encoding="utf-8")
    if args.open_test_after_validation and passed:
        for key, spec in TARGETS.items():
            for seed in spec["seeds"]:
                prefix = output / key / "s{}_official_patchtst".format(seed)
                test = prefix.with_name(prefix.name + "_test.json")
                if not (args.resume and test.is_file() and complete(prefix.with_name(prefix.name + "_test_event.json"))):
                    guarded(python, test_command(python, spec["dataset"], prefix.with_suffix(".pt"), test, seed), prefix, "test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
