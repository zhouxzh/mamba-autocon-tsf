#!/usr/bin/env python3
"""Analyze a validation-only MambaTS AutoCon lambda calibration matrix.

The zero-lambda AutoCon arm is the within-model control.  This script reports
coordinate-level deltas and descriptive lambda summaries without selecting a
test metric or treating time windows/channels as independent replicates.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found: {path}")
    required = {"dataset", "pred_len", "seed", "model", "autocon_lambda", "best_val_mse"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"rows are missing columns: {sorted(missing)}")
    parsed = []
    for row in rows:
        value = dict(row)
        value["pred_len"] = int(row["pred_len"])
        value["seed"] = int(row["seed"])
        value["autocon_lambda"] = float(row["autocon_lambda"])
        value["best_val_mse"] = float(row["best_val_mse"])
        if not math.isfinite(value["autocon_lambda"]) or value["autocon_lambda"] < 0:
            raise ValueError(f"invalid AutoCon lambda: {row}")
        if not math.isfinite(value["best_val_mse"]) or value["best_val_mse"] <= 0:
            raise ValueError(f"invalid validation MSE: {row}")
        parsed.append(value)
    return parsed


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def analyze(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    controls: dict[tuple[str, int, int], dict] = {}
    cells: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["pred_len"]), int(row["seed"]))
        if float(row["autocon_lambda"]) == 0.0:
            if key in controls:
                raise ValueError(f"duplicate zero-lambda control: {key}")
            controls[key] = row
        cells.setdefault(key, []).append(row)
    if set(controls) != set(cells):
        missing = sorted(set(cells) - set(controls))
        raise ValueError(f"coordinates without zero-lambda control: {missing}")

    deltas = []
    for key in sorted(cells):
        control = controls[key]
        control_mse = float(control["best_val_mse"])
        for row in sorted(cells[key], key=lambda item: float(item["autocon_lambda"])):
            mse = float(row["best_val_mse"])
            delta = mse - control_mse
            deltas.append({
                "dataset": key[0], "pred_len": key[1], "seed": key[2],
                "model": str(row["model"]),
                "autocon_lambda": float(row["autocon_lambda"]),
                "control_val_mse": control_mse,
                "candidate_val_mse": mse,
                "delta_candidate_minus_control": delta,
                "relative_change_pct": 100.0 * delta / control_mse,
                "best_epoch": row.get("best_epoch", ""),
                "best_epoch_autocon_loss": row.get("best_epoch_autocon_loss", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
            })

    grouped: dict[float, list[dict]] = {}
    for row in deltas:
        grouped.setdefault(float(row["autocon_lambda"]), []).append(row)
    summary = []
    for lambda_value, group in sorted(grouped.items()):
        changes = [float(item["relative_change_pct"]) for item in group]
        nonzero = [item for item in group if lambda_value > 0]
        summary.append({
            "autocon_lambda": lambda_value,
            "n_coordinates": len(group),
            "mean_relative_change_pct": sum(changes) / len(changes),
            "median_relative_change_pct": _median(changes),
            "wins_vs_zero": sum(float(item["delta_candidate_minus_control"]) < 0 for item in group),
            "ties_vs_zero": sum(float(item["delta_candidate_minus_control"]) == 0 for item in group),
            "max_relative_change_pct": max(changes),
            "min_relative_change_pct": min(changes),
            "mean_autocon_loss_at_best_epoch": (
                sum(float(item["best_epoch_autocon_loss"]) for item in nonzero
                    if str(item["best_epoch_autocon_loss"]).strip()) / len([
                        item for item in nonzero if str(item["best_epoch_autocon_loss"]).strip()
                    ]) if any(str(item["best_epoch_autocon_loss"]).strip() for item in nonzero) else ""
            ),
        })
    return deltas, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict], summary: list[dict]) -> None:
    positive = [row for row in summary if float(row["autocon_lambda"]) > 0]
    selected = min(positive, key=lambda row: (
        float(row["mean_relative_change_pct"]), float(row["autocon_lambda"])
    )) if positive else None
    lines = [
        "# MambaTS v3 AutoCon lambda calibration",
        "",
        "The matrix is validation-only. `lambda=0` is an AutoCon model with the auxiliary objective disabled and is the within-model control.",
        "No test metric, time window, or channel is used for selection.",
        "",
        f"Coordinate-lambda rows: **{len(rows)}**; coordinates: **{len({(r['dataset'], r['pred_len'], r['seed']) for r in rows})}**.",
        "",
        "## Lambda summary",
        "",
        "| lambda | coordinates | mean relative change % | median % | wins vs zero | min % | max % |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append("| {lam} | {n} | {mean} | {median} | {wins} | {minimum} | {maximum} |".format(
            lam=_fmt(float(item["autocon_lambda"])), n=item["n_coordinates"],
            mean=_fmt(float(item["mean_relative_change_pct"])),
            median=_fmt(float(item["median_relative_change_pct"])),
            wins=item["wins_vs_zero"], minimum=_fmt(float(item["min_relative_change_pct"])),
            maximum=_fmt(float(item["max_relative_change_pct"])),
        ))
    lines.extend(["", "## Interpretation", ""])
    if selected is not None:
        lines.append(
            "The descriptive lowest mean validation change among positive candidates is "
            f"`lambda={float(selected['autocon_lambda']):g}` at "
            f"{float(selected['mean_relative_change_pct']):.3f}% versus the zero control. "
            "This is a calibration result, not confirmatory evidence."
        )
    lines.extend(["", "## Coordinate deltas", "", "| dataset | horizon | seed | lambda | control MSE | candidate MSE | relative % |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append("| {dataset} | {pred_len} | {seed} | {lam} | {control} | {candidate} | {relative} |".format(
            dataset=row["dataset"], pred_len=row["pred_len"], seed=row["seed"],
            lam=_fmt(float(row["autocon_lambda"])), control=_fmt(float(row["control_val_mse"])),
            candidate=_fmt(float(row["candidate_val_mse"])),
            relative=_fmt(float(row["relative_change_pct"])),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = _read_rows(args.rows_csv.resolve())
    deltas, summary = analyze(rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "lambda_deltas.csv", deltas)
    _write_csv(output / "lambda_summary.csv", summary)
    _write_markdown(output / "lambda_readout.md", deltas, summary)
    print(f"Analyzed {len(deltas)} coordinate-lambda rows across {len(summary)} lambdas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
