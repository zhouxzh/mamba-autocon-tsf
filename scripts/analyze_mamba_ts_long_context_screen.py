#!/usr/bin/env python3
"""Read the validation-only long-context MambaTS v3/v4 screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _flag_value(command: list, flag: str) -> str:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"manifest command is missing {flag}") from exc


def read_phase(phase_root: Path) -> list[dict]:
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    with manifest_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != 8:
        raise ValueError(f"expected 8 manifest records, found {len(records)}")
    rows = []
    seen = set()
    for record in records:
        if record.get("status") != "ok":
            raise ValueError(f"non-ok manifest record: {record.get('job_id')}")
        result_dir = Path(str(record["expected_artifact"])).parent
        config = _json(result_dir / "config.json")
        checkpoint_root = Path(_flag_value(record.get("command", []), "--checkpoints"))
        checkpoint_dir = checkpoint_root / result_dir.name
        logs = _json(checkpoint_dir / "loss_logs.json")
        values = [float(item) for item in logs.get("val_loss", [])]
        if not values or not all(math.isfinite(item) for item in values):
            raise ValueError(f"invalid validation trajectory: {checkpoint_dir}")
        key = (
            str(record["dataset"]), int(record["pred_len"]),
            int(record["seed"]), str(record["model"]),
        )
        if key in seen:
            raise ValueError(f"duplicate coordinate: {key}")
        seen.add(key)
        rows.append({
            "dataset": key[0], "pred_len": key[1], "seed": key[2],
            "model": str(record["model"]), "seq_len": int(config.get("seq_len", 0)),
            "best_val_mse": min(values), "best_epoch": int(logs.get("best_epoch", values.index(min(values)) + 1)),
            "elapsed_sec": float(record.get("elapsed_sec", float("nan"))),
            "test_metrics_produced": False,
        })
    return sorted(rows, key=lambda row: (row["dataset"], row["pred_len"], row["model"], row["seed"]))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_phase(args.phase_root.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)

    by_key = {(row["dataset"], row["pred_len"], row["seed"], row["model"]): row for row in rows}
    deltas = []
    for dataset in sorted({row["dataset"] for row in rows}):
        key = (dataset, 96, 2022)
        control = by_key[key + ("MambaTSv3AutoCon",)]
        candidate = by_key[key + ("MambaTSv4AutoCon",)]
        delta = 100.0 * (candidate["best_val_mse"] - control["best_val_mse"]) / control["best_val_mse"]
        deltas.append({
            "dataset": dataset, "pred_len": 96, "seed": 2022,
            "control_val_mse": control["best_val_mse"],
            "candidate_val_mse": candidate["best_val_mse"],
            "relative_change_pct": delta,
            "control_elapsed_sec": control["elapsed_sec"],
            "candidate_elapsed_sec": candidate["elapsed_sec"],
        })
    _write_csv(output / "v4_vs_v3_deltas.csv", deltas)
    changes = [row["relative_change_pct"] for row in deltas]
    control_times = [row["control_elapsed_sec"] for row in deltas]
    candidate_times = [row["candidate_elapsed_sec"] for row in deltas]
    lines = [
        "# MambaTS long-context ACF screen", "",
        "Validation-only exact-coordinate screen with seq_len=384 and H=96; test metrics were withheld.",
        "The only paired comparator is the same-context MambaTSv3AutoCon control.", "",
        f"Rows: **{len(rows)}**; paired coordinates: **{len(deltas)}**; sequence length: **{rows[0]['seq_len']}**.", "",
        "| dataset | v3 val MSE | v4 val MSE | relative change | v3 sec | v4 sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {row['control_val_mse']:.6f} | {row['candidate_val_mse']:.6f} | "
            f"{row['relative_change_pct']:+.3f}% | {row['control_elapsed_sec']:.2f} | {row['candidate_elapsed_sec']:.2f} |"
        )
    lines.extend([
        "", f"Mean relative change: **{statistics.mean(changes):+.3f}%**; wins: **{sum(v < 0 for v in changes)}/{len(changes)}**.",
        f"Mean elapsed time: v3 **{statistics.mean(control_times):.2f}s**, v4 **{statistics.mean(candidate_times):.2f}s**.",
        "", "The screen is exploratory: one seed, four epochs, validation-only, and no matched TimesNet or full-budget confirmation.",
    ])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed {len(rows)} long-context validation rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
