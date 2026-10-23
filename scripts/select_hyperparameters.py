#!/usr/bin/env python
"""Select AutoCon lambda using validation trajectories only.

Each job contributes its best validation MSE.  To avoid letting a dataset with
the largest numerical scale dominate, losses are normalized within each
dataset/model/horizon/seed cell before averaging over the lambda candidates.
The resulting lambda is frozen before the confirmatory phase is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def _arg(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(result_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, str]:
    names = (
        "config.json", "metrics.npy", "original_metrics.npy", "channel_metrics.npy",
        "pred.npy", "true.npy", "training_complete.json", "acf_prior.npz",
    )
    paths = [result_dir / name for name in names]
    if checkpoint_dir is not None:
        paths.extend([checkpoint_dir / "checkpoint.pth", checkpoint_dir / "loss_logs.json"])
    return {path.name: _sha256(path) for path in paths if path.is_file()}


def _validate_gate(validation_path: Path, manifest_path: Path, plan_path: Path,
                   expected_jobs: int, phase_name: str, phase: dict,
                   repo_root: Path) -> dict:
    """Refuse lambda selection unless the strict phase validator passed."""
    if not validation_path.is_file():
        raise SystemExit("strict selection requires a phase_validation.json")
    with validation_path.open(encoding="utf-8") as handle:
        validation = json.load(handle)
    required = ("valid", "manifest", "manifest_sha256", "plan_sha256",
                "provenance", "provenance_id", "expected_jobs",
                "manifest_unique_jobs", "artifact_hashes_by_job")
    missing = [key for key in required if key not in validation]
    if missing:
        raise SystemExit("selection validation is missing strict fields: " + ", ".join(missing))
    if validation["valid"] is not True:
        raise SystemExit("selection phase failed validation; refusing lambda selection")
    if validation.get("phase") != phase_name:
        raise SystemExit("selection validation phase does not match --phase")
    if Path(validation["manifest"]).resolve() != manifest_path.resolve():
        raise SystemExit("selection validation refers to a different manifest")
    if validation["manifest_sha256"] != _sha256(manifest_path):
        raise SystemExit("selection manifest changed after validation")
    if validation["plan_sha256"] != _sha256(plan_path):
        raise SystemExit("training plan changed after selection validation")
    if int(validation["expected_jobs"]) != expected_jobs or int(validation["manifest_unique_jobs"]) != expected_jobs:
        raise SystemExit("selection validation coverage does not match the registered plan")
    provenance_path = Path(validation["provenance"])
    if not provenance_path.is_file():
        raise SystemExit("selection provenance sidecar is missing")
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if provenance.get("phase") != phase_name:
        raise SystemExit("selection provenance phase does not match --phase")
    if provenance.get("provenance_id") != validation["provenance_id"]:
        raise SystemExit("selection provenance id mismatch")
    snapshot_root = Path(provenance.get("source_snapshot", ""))
    for relative in (
        "scripts/run_training_plan.py",
        "scripts/validate_training_phase.py",
        "scripts/select_hyperparameters.py",
    ):
        snapshot_file = snapshot_root / relative
        live_file = repo_root / relative
        if snapshot_file.is_file() and _sha256(snapshot_file) != _sha256(live_file):
            raise SystemExit(f"live {relative} differs from the immutable selection snapshot")
    if provenance.get("selection_sha256") is not None:
        raise SystemExit("selection phase unexpectedly depends on a selection file")
    if len(validation["artifact_hashes_by_job"]) != expected_jobs:
        raise SystemExit("selection validation has incomplete artifact hash coverage")
    registered_root = (repo_root / phase.get("output_root", f"artifacts/paper_final/{phase_name}")).resolve()
    if registered_root not in manifest_path.parents:
        raise SystemExit("selection manifest is outside the registered phase output root")
    return validation


def _expected_cells(plan_path: Path, phase_name: str) -> tuple[set[tuple], set[float]]:
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    phase = next((item for item in plan.get("phases", []) if item.get("name") == phase_name), None)
    if phase is None:
        raise ValueError(f"selection phase not found in plan: {phase_name}")
    global_cfg = dict(plan.get("global", {}))
    cfg = dict(global_cfg)
    cfg.update(phase)
    candidates = {float(value) for value in cfg.get("autocon_lambdas", [])}
    if not candidates:
        raise ValueError(f"{phase_name}: autocon_lambdas must be non-empty")
    expected = set()
    horizons_by_dataset = cfg.get("horizons_by_dataset")
    for model in cfg.get("models", []):
        for dataset in cfg.get("datasets", []):
            horizons = horizons_by_dataset.get(dataset, []) if horizons_by_dataset else cfg.get("horizons", [])
            for pred_len in horizons:
                for seed in cfg.get("seeds", []):
                    expected.add((str(model), str(dataset), int(pred_len), int(seed)))
    if not expected:
        raise ValueError(f"{phase_name}: no expected selection cells")
    return expected, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/final_training_plan.json"))
    parser.add_argument("--phase", default="selection")
    parser.add_argument("--validation", type=Path, required=True,
                        help="strict phase_validation.json produced before selection")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (args.manifest if args.manifest.is_absolute() else repo_root / args.manifest).resolve()
    plan_path = (args.plan if args.plan.is_absolute() else repo_root / args.plan).resolve()
    validation_path = (args.validation if args.validation.is_absolute() else repo_root / args.validation).resolve()

    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    phase = next((item for item in plan.get("phases", []) if item.get("name") == args.phase), None)
    if phase is None:
        raise SystemExit(f"selection phase not found in plan: {args.phase}")
    expected_features = phase.get("features", plan.get("global", {}).get("features", "S"))

    # Keep only the last manifest record per job.  This makes an intentional
    # retry/resume idempotent and prevents duplicate rows from changing the
    # validation aggregate.
    manifest_records = {}
    malformed_lines = 0
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if "job_id" in record:
                manifest_records[record["job_id"]] = record

    expected_cells, expected_candidates = _expected_cells(plan_path, args.phase)
    # One row per candidate lambda is expected for each cell.
    validation_payload = _validate_gate(
        validation_path, manifest_path, plan_path,
        len(expected_cells) * len(expected_candidates), args.phase, phase, repo_root,
    )
    rows = []
    seen_coordinates = set()
    for record in manifest_records.values():
        if record.get("status") != "ok" or record.get("autocon_lambda") is None:
            continue
        command = record["command"]
        if "--skip_test" not in command:
            raise SystemExit("selection manifest contains a job that did not disable testing: " + record.get("job_id", "unknown"))
        checkpoint_root = Path(_arg(command, "--checkpoints"))
        matches = sorted(checkpoint_root.glob(f"*plan_{record['job_id']}*"))
        if not matches:
            continue
        result_root = Path(_arg(command, "--results_dir"))
        result_matches = sorted(result_root.glob(f"*plan_{record['job_id']}*/training_complete.json"))
        if not result_matches:
            raise SystemExit("selection result marker is missing: " + record.get("job_id", "unknown"))
        actual_hashes = _artifact_hashes(result_matches[0].parent, matches[0])
        expected_hashes = validation_payload["artifact_hashes_by_job"].get(record.get("job_id"))
        if expected_hashes != actual_hashes:
            raise SystemExit("selection artifact changed after validation: " + record.get("job_id", "unknown"))
        log_path = matches[0] / "loss_logs.json"
        if not log_path.is_file():
            continue
        with log_path.open(encoding="utf-8") as log_handle:
            logs = json.load(log_handle)
        validation = [float(value) for value in logs.get("val_loss", [])]
        if any(not math.isfinite(value) for value in validation):
            raise SystemExit("selection manifest contains non-finite validation loss: " + record.get("job_id", "unknown"))
        if not validation:
            continue
        row = {
            "model": record["model"], "dataset": record["dataset"],
            "pred_len": int(record["pred_len"]), "seed": int(record["seed"]),
            "lambda": float(record["autocon_lambda"]),
            "best_val_loss": min(validation),
        }
        coordinate = (row["model"], row["dataset"], row["pred_len"], row["seed"], row["lambda"])
        if coordinate in seen_coordinates:
            raise SystemExit("duplicate completed selection coordinate: " + repr(coordinate))
        seen_coordinates.add(coordinate)
        rows.append(row)

    if not rows:
        raise SystemExit("no completed validation trajectories found")

    cells = defaultdict(set)
    for row in rows:
        cells[(row["model"], row["dataset"], row["pred_len"], row["seed"])].add(row["lambda"])
    actual_cells = set(cells)
    missing_cells = expected_cells - actual_cells
    unexpected_cells = actual_cells - expected_cells
    incomplete = {key: sorted(values) for key, values in cells.items() if values != expected_candidates}
    if missing_cells or unexpected_cells or incomplete:
        problems = []
        if missing_cells:
            problems.append("missing=" + repr(sorted(missing_cells)[:4]))
        if unexpected_cells:
            problems.append("unexpected=" + repr(sorted(unexpected_cells)[:4]))
        if incomplete:
            problems.append("incomplete=" + repr(list(incomplete.items())[:4]))
        raise SystemExit("invalid selection coverage; refusing selection: " + "; ".join(problems))

    # Normalize within a cell so the pooled criterion is a relative validation
    # score, not a weighted average of incomparable dataset units.
    cells = defaultdict(list)
    for row in rows:
        cells[(row["model"], row["dataset"], row["pred_len"], row["seed"])].append(row)
    normalized = []
    for cell_rows in cells.values():
        mean_loss = sum(row["best_val_loss"] for row in cell_rows) / len(cell_rows)
        if not math.isfinite(mean_loss) or mean_loss <= 0:
            raise SystemExit("selection cell has invalid validation loss")
        for row in cell_rows:
            normalized.append({**row, "relative_val_loss": row["best_val_loss"] / max(mean_loss, 1e-12)})

    by_model_lambda = defaultdict(list)
    for row in normalized:
        by_model_lambda[(row["model"], row["lambda"])].append(row["relative_val_loss"])
    candidates = defaultdict(list)
    for (model, lambda_value), values in by_model_lambda.items():
        candidates[model].append({
            "lambda": lambda_value, "mean_relative_val_loss": sum(values) / len(values),
            "n_jobs": len(values),
        })

    selected = {}
    summary = {}
    for model, values in sorted(candidates.items()):
        values.sort(key=lambda item: (item["mean_relative_val_loss"], item["lambda"]))
        selected[model] = values[0]["lambda"]
        summary[model] = values

    output = (args.output if args.output.is_absolute() else repo_root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "criterion": "mean relative best validation MSE; test split never read",
        "plan": str(plan_path),
        "phase": args.phase,
        "features": expected_features,
        "malformed_manifest_lines_skipped": malformed_lines,
        "n_trajectories": len(rows),
        "selected_lambda_by_model": selected,
        "candidates": summary,
        "rows": rows,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"selected_lambda_by_model": selected, "n_trajectories": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
