#!/usr/bin/env python3
"""Pair calibrated MambaTS v3 AutoCon rows with frozen TimesNet rows.

Only exact dataset/horizon/seed coordinates are retained.  The input CSVs are
validation-only readouts; this tool does not select a model or read test data.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "pred_len", "seed", "model", "best_val_mse"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path} is missing columns: {sorted(required - set(rows[0]))}")
    parsed = []
    for row in rows:
        value = dict(row)
        value["pred_len"] = int(row["pred_len"])
        value["seed"] = int(row["seed"])
        value["best_val_mse"] = float(row["best_val_mse"])
        if not math.isfinite(value["best_val_mse"]) or value["best_val_mse"] <= 0:
            raise ValueError(f"invalid validation MSE in {path}: {row}")
        parsed.append(value)
    return parsed


def pair(candidate_rows: list[dict], reference_rows: list[dict], *, candidate_model: str, reference_model: str) -> list[dict]:
    def select(rows: list[dict], model: str) -> dict[tuple[str, int, int], dict]:
        result = {}
        for row in rows:
            if str(row["model"]) != model:
                continue
            key = (str(row["dataset"]), int(row["pred_len"]), int(row["seed"]))
            if key in result:
                raise ValueError(f"duplicate {model} coordinate: {key}")
            result[key] = row
        return result

    candidate = select(candidate_rows, candidate_model)
    reference = select(reference_rows, reference_model)
    if set(candidate) != set(reference):
        raise ValueError(
            f"exact coordinate mismatch: candidate-only={sorted(set(candidate) - set(reference))}, "
            f"reference-only={sorted(set(reference) - set(candidate))}"
        )
    result = []
    for key in sorted(candidate):
        left = float(candidate[key]["best_val_mse"])
        right = float(reference[key]["best_val_mse"])
        result.append({
            "dataset": key[0], "pred_len": key[1], "seed": key[2],
            "candidate": candidate_model, "reference": reference_model,
            "candidate_val_mse": left, "reference_val_mse": right,
            "delta_candidate_minus_reference": left - right,
            "relative_change_pct": 100.0 * (left - right) / right,
        })
    return result


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Calibrated MambaTS v3 AutoCon vs TimesNet",
        "",
        "Validation-only exact-coordinate pairing. Negative relative change favors the candidate; no test metric is used.",
        "",
        f"Paired coordinates: **{len(rows)}**; wins: **{sum(float(row['relative_change_pct']) < 0 for row in rows)}/{len(rows)}**.",
        "",
        "| split | coordinates | wins | mean relative % | median % |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, predicate in (("all", lambda row: True), ("H=12", lambda row: row["pred_len"] == 12), ("H=96", lambda row: row["pred_len"] == 96)):
        values = [float(row["relative_change_pct"]) for row in rows if predicate(row)]
        lines.append("| {} | {} | {} | {} | {} |".format(
            label, len(values), sum(value < 0 for value in values),
            _fmt(sum(values) / len(values)), _fmt(statistics.median(values)),
        ))
    lines.extend(["", "| dataset | horizon | seed | candidate MSE | TimesNet MSE | relative % |", "|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append("| {dataset} | {pred_len} | {seed} | {candidate} | {reference} | {relative} |".format(
            dataset=row["dataset"], pred_len=row["pred_len"], seed=row["seed"],
            candidate=_fmt(float(row["candidate_val_mse"])), reference=_fmt(float(row["reference_val_mse"])),
            relative=_fmt(float(row["relative_change_pct"])),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, action="append", required=True)
    parser.add_argument("--candidate-model", default="MambaTSv3AutoCon")
    parser.add_argument("--reference-model", default="TimesNet")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate_rows = _load(args.candidate_csv.resolve())
    reference_rows = []
    for path in args.reference_csv:
        reference_rows.extend(_load(path.resolve()))
    rows = pair(candidate_rows, reference_rows, candidate_model=args.candidate_model, reference_model=args.reference_model)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "timesnet_deltas.csv", rows)
    _write_markdown(output / "timesnet_readout.md", rows)
    print(f"Paired {len(rows)} exact coordinates; wins={sum(float(row['relative_change_pct']) < 0 for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
