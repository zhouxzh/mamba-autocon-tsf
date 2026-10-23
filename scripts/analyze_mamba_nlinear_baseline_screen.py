#!/usr/bin/env python3
"""Read and summarize a frozen validation-only NLinear baseline screen.

The screen deliberately reads only the pure forecasting validation trajectory
stored in ``loss_logs.json``.  It does not run inference and it never treats a
missing test artifact as a successful test result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _flag_value(command: Sequence[Any], flag: str) -> str:
    try:
        return str(command[list(command).index(flag) + 1])
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"manifest command is missing {flag}") from exc


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite: {value!r}")
    return result


def _read_loss_log(path: Path) -> Tuple[float, int, int]:
    logs = _json(path)
    trajectory = logs.get("val_loss")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"missing val_loss trajectory: {path}")
    values = [_finite(item, f"val_loss in {path}") for item in trajectory]
    best = min(values)
    best_index = values.index(best)
    best_epoch = int(logs.get("best_epoch", best_index + 1))
    epochs_ran = int(logs.get("epochs_ran", len(values)))
    return best, best_epoch, epochs_ran


def read_phase(phase_root: Path, expected_jobs: int = 16) -> List[Dict[str, Any]]:
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != expected_jobs:
        raise ValueError(f"expected {expected_jobs} manifest records, found {len(records)}")

    rows: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        if record.get("status") != "ok":
            raise ValueError(f"non-ok manifest record: {record.get('job_id')}")
        expected_artifact = Path(str(record["expected_artifact"])).resolve()
        result_dir = expected_artifact.parent
        config_path = result_dir / "config.json"
        config = _json(config_path)
        checkpoint_root = Path(_flag_value(record.get("command", []), "--checkpoints")).resolve()
        checkpoint_dir = checkpoint_root / result_dir.name
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        loss_path = checkpoint_dir / "loss_logs.json"
        if not checkpoint_path.is_file():
            raise ValueError(f"missing checkpoint: {checkpoint_path}")
        best, best_epoch, epochs_ran = _read_loss_log(loss_path)

        dataset = str(record["dataset"])
        pred_len = int(record["pred_len"])
        seed = int(record["seed"])
        key = (dataset, pred_len, seed)
        if key in seen:
            raise ValueError(f"duplicate coordinate: {key}")
        seen.add(key)
        rows.append({
            "dataset": dataset,
            "pred_len": pred_len,
            "seed": seed,
            "model": str(record.get("model", "NLinear")),
            "best_val_mse": best,
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "elapsed_sec": _finite(record.get("elapsed_sec", float("nan")), "elapsed_sec"),
            "autocon_enabled": bool(config.get("AutoCon", False)),
            "skip_test": bool(config.get("skip_test", True)),
            "test_metrics_produced": False,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
        })
    return sorted(rows, key=lambda row: (row["dataset"], row["pred_len"], row["seed"]))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not materialized:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for horizon in sorted({int(row["pred_len"]) for row in rows}):
        subset = [row for row in rows if int(row["pred_len"]) == horizon]
        values = [float(row["best_val_mse"]) for row in subset]
        elapsed = [float(row["elapsed_sec"]) for row in subset]
        output.append({
            "pred_len": horizon,
            "n": len(subset),
            "mean_best_val_mse": statistics.mean(values),
            "std_best_val_mse": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median_best_val_mse": statistics.median(values),
            "mean_elapsed_sec": statistics.mean(elapsed),
            "median_elapsed_sec": statistics.median(elapsed),
            "test_metrics_produced": False,
        })
    return output


def _read_reference(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        result.append({
            "dataset": str(row["dataset"]),
            "pred_len": int(row["pred_len"]),
            "seed": int(row["seed"]),
            "model": str(row["model"]),
            "best_val_mse": _finite(row["best_val_mse"], "reference best_val_mse"),
        })
    return result


def _pair(rows: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]], model: str) -> List[Dict[str, Any]]:
    left = {(r["dataset"], int(r["pred_len"]), int(r["seed"])): r for r in rows}
    right = {
        (r["dataset"], int(r["pred_len"]), int(r["seed"])): r
        for r in reference if str(r["model"]) == model
    }
    if set(left) != set(right):
        raise ValueError(
            f"exact coordinate mismatch against {model}: "
            f"left-only={sorted(set(left) - set(right))}, "
            f"right-only={sorted(set(right) - set(left))}"
        )
    output = []
    for key in sorted(left):
        candidate = float(left[key]["best_val_mse"])
        baseline = float(right[key]["best_val_mse"])
        output.append({
            "dataset": key[0],
            "pred_len": key[1],
            "seed": key[2],
            "candidate_model": str(left[key]["model"]),
            "reference_model": model,
            "candidate_val_mse": candidate,
            "reference_val_mse": baseline,
            "relative_change_pct": 100.0 * (candidate - baseline) / baseline,
            "candidate_wins": candidate < baseline,
        })
    return output


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, action="append", default=[])
    parser.add_argument("--reference-model", action="append", default=[])
    parser.add_argument("--expected-jobs", type=int, default=16)
    parser.add_argument(
        "--seed",
        dest="seeds",
        type=int,
        action="append",
        default=None,
        help="optionally retain only these registered seed coordinates",
    )
    args = parser.parse_args()

    rows = read_phase(args.phase_root.resolve(), expected_jobs=args.expected_jobs)
    if args.seeds is not None:
        wanted = set(args.seeds)
        rows = [row for row in rows if int(row["seed"]) in wanted]
        if not rows:
            raise ValueError(f"seed filter selected no rows: {sorted(wanted)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    summary = _summary(rows)
    _write_csv(output / "summary.csv", summary)

    lines = [
        "# NLinear PEMS baseline screen",
        "",
        "> Exploratory, validation-only readout. The registered jobs use `skip_test=true`; no test metric is claimed.",
        "",
        f"- Completed rows: **{len(rows)}**; unique coordinates: **{len({(r['dataset'], r['pred_len'], r['seed']) for r in rows})}**.",
        f"- AutoCon enabled rows: **{sum(bool(r['autocon_enabled']) for r in rows)}**.",
        "",
        "| Horizon | Rows | Mean best validation MSE | SD | Median MSE | Mean seconds | Median seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['pred_len']} | {item['n']} | {item['mean_best_val_mse']:.6f} | "
            f"{item['std_best_val_mse']:.6f} | {item['median_best_val_mse']:.6f} | "
            f"{item['mean_elapsed_sec']:.2f} | {item['median_elapsed_sec']:.2f} |"
        )

    references = list(args.reference_model)
    if references and len(references) not in (1, len(args.reference_csv)):
        raise ValueError("reference-model must have one value or one value per reference-csv")
    for index, reference_path in enumerate(args.reference_csv):
        reference = _read_reference(reference_path.resolve())
        models = sorted({str(row["model"]) for row in reference})
        model = references[0] if len(references) == 1 else (references[index] if references else (models[0] if len(models) == 1 else ""))
        if not model:
            raise ValueError(f"reference model is ambiguous for {reference_path}: {models}")
        paired = _pair(rows, reference, model)
        output_name = f"{_slug(model)}_deltas.csv"
        _write_csv(output / output_name, paired)
        for horizon in sorted({int(row["pred_len"]) for row in paired}):
            values = [float(row["relative_change_pct"]) for row in paired if int(row["pred_len"]) == horizon]
            wins = sum(value < 0 for value in values)
            lines.extend([
                "",
                f"## NLinear versus {model}, H={horizon}",
                "",
                f"Exact coordinates: **{len(values)}**; wins: **{wins}/{len(values)}**; mean relative change: **{statistics.mean(values):.3f}%**.",
            ])

    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed {len(rows)} NLinear validation rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
