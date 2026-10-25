#!/usr/bin/env python3
"""Summarize the completed ACE2 S-full synthetic experiment truthfully.

The analysis unit is an independent DGP replicate. Optimisation seeds are
averaged within each replicate before condition-level intervals are formed.
This script does not read benchmark datasets, L0/R1/P1 artifacts, or produce
claims beyond the generated synthetic DGP scope.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("S-core-v2-20261025", "S-extension-v2-20261025")
ARMS = ("A_signed", "N_null", "U_unsigned", "S_shuffle", "R_random")
CONTROLS = ARMS[1:]
MIGRATIONS = ("M0", "M1", "M2")
DGP_NAMES = tuple(f"G{i}" for i in range(1, 9))
T_975_DF19 = 2.093024054
COLORS = {"M0": "#0072B2", "M1": "#D55E00", "M2": "#009E73"}
MARKERS = {"M0": "o", "M1": "s", "M2": "^"}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) <= 1:
        return mean, 0.0, 0.0
    half = float(T_975_DF19 * array.std(ddof=1) / math.sqrt(len(array)))
    return mean, mean - half, mean + half


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    source_paths = []
    for phase in PHASES:
        path = ROOT / "artifacts" / "ace2_simulation" / phase / "results.jsonl"
        source_paths.append(str(path.relative_to(ROOT)))
        phase_rows = read_rows(path)
        if not phase_rows or any(row.get("status") != "COMPLETED" for row in phase_rows):
            raise RuntimeError(f"incomplete phase: {phase}")
        rows.extend(phase_rows)
    if len(rows) != 57600:
        raise RuntimeError(f"expected 57600 S-full rows, found {len(rows)}")

    run_groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in (
            "dgp", "migration", "noise", "replicate", "horizon",
            "label_fraction", "optimization_seed",
        ))
        run_groups[key][row["arm"]] = row
    incomplete = [key for key, arms in run_groups.items() if set(arms) != set(ARMS)]
    if incomplete:
        raise RuntimeError(f"incomplete five-arm run groups: {len(incomplete)}")

    seed_deltas = []
    for key, arms in sorted(run_groups.items()):
        active = float(arms["A_signed"]["mse"])
        for control in CONTROLS:
            seed_deltas.append({
                "dgp": key[0], "migration": key[1], "noise": key[2],
                "replicate": key[3], "horizon": key[4],
                "label_fraction": key[5], "optimization_seed": key[6],
                "control": control, "active_mse": active,
                "control_mse": float(arms[control]["mse"]),
                "delta_mse": active - float(arms[control]["mse"]),
            })

    replicate_groups: dict[tuple, list[float]] = defaultdict(list)
    for row in seed_deltas:
        key = tuple(row[field] for field in (
            "dgp", "migration", "noise", "replicate", "horizon",
            "label_fraction", "control",
        ))
        replicate_groups[key].append(row["delta_mse"])
    replicate_rows = []
    for key, deltas in sorted(replicate_groups.items()):
        if len(deltas) != 3:
            raise RuntimeError(f"expected three optimisation seeds for {key}, found {len(deltas)}")
        replicate_rows.append({
            "dgp": key[0], "migration": key[1], "noise": key[2],
            "replicate": key[3], "horizon": key[4],
            "label_fraction": key[5], "control": key[6],
            "optimization_seed_count": len(deltas),
            "mean_delta_mse": float(np.mean(deltas)),
        })

    condition_groups: dict[tuple, list[float]] = defaultdict(list)
    for row in replicate_rows:
        key = tuple(row[field] for field in (
            "dgp", "migration", "noise", "horizon", "label_fraction", "control",
        ))
        condition_groups[key].append(row["mean_delta_mse"])
    summary_rows = []
    for key, deltas in sorted(condition_groups.items()):
        if len(deltas) != 20:
            raise RuntimeError(f"expected 20 DGP replicates for {key}, found {len(deltas)}")
        mean, low, high = mean_ci(deltas)
        summary_rows.append({
            "dgp": key[0], "migration": key[1], "noise": key[2],
            "horizon": key[3], "label_fraction": key[4], "control": key[5],
            "dgp_replicate_count": len(deltas),
            "mean_delta_mse": mean, "ci95_low_t": low, "ci95_high_t": high,
            "interval": "two-sided 95% t interval across 20 independent DGP replicates; optimisation seeds averaged within replicate",
        })

    figure_rows = [row for row in summary_rows if row["control"] == "N_null"]
    strata = [(noise, label, horizon) for noise in ("low", "high") for label in (0.1, 0.25) for horizon in (96, 192)]
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.6), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.12, top=0.82, wspace=0.035, hspace=0.20)
    positions = np.arange(len(DGP_NAMES))
    offsets = {"M0": -0.22, "M1": 0.0, "M2": 0.22}
    for axis, (noise, label, horizon) in zip(axes.flat, strata):
        for migration in MIGRATIONS:
            values = []
            lower = []
            upper = []
            for dgp in DGP_NAMES:
                row = next(item for item in figure_rows if (
                    item["dgp"], item["migration"], item["noise"],
                    item["label_fraction"], item["horizon"],
                ) == (dgp, migration, noise, label, horizon))
                values.append(row["mean_delta_mse"])
                lower.append(row["mean_delta_mse"] - row["ci95_low_t"])
                upper.append(row["ci95_high_t"] - row["mean_delta_mse"])
            axis.errorbar(
                positions + offsets[migration], values, yerr=[lower, upper],
                color=COLORS[migration], marker=MARKERS[migration], linestyle="-",
                linewidth=1.1, markersize=4.5, capsize=2.2, label=migration,
            )
        axis.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--", zorder=0)
        axis.set_title(f"noise={noise}; label={label:.0%}; H={horizon}", fontsize=9)
        axis.set_xticks(positions, DGP_NAMES)
        axis.tick_params(axis="x", labelrotation=0, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axes[0, 0].set_ylabel("Mean signed - null MSE\n(negative favors signed)")
    axes[1, 0].set_ylabel("Mean signed - null MSE\n(negative favors signed)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Migration", loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False)
    fig.suptitle("ACE2 S-full Synthetic DGP: Signed vs Exact-Null", y=0.995, fontsize=12)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    png_path = figure_dir / "ace2_sfull_signed_vs_null.png"
    pdf_path = figure_dir / "ace2_sfull_signed_vs_null.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    write_csv(output / "seed_level_paired_deltas.csv", seed_deltas, list(seed_deltas[0]))
    write_csv(output / "replicate_level_paired_deltas.csv", replicate_rows, list(replicate_rows[0]))
    write_csv(output / "condition_summary.csv", summary_rows, list(summary_rows[0]))
    manifest = {
        "study_id": "ACE2",
        "protocol_version": "ace2-simulation-v1",
        "scope": "S-full synthetic DGP only; no L0/R1/P1 result is represented.",
        "source_result_ledgers": source_paths,
        "row_count": len(rows),
        "transformations": [
            "restrict to COMPLETED S-core-v2 and S-extension-v2 rows",
            "compute A_signed minus each control MSE per optimisation seed",
            "average the three optimisation-seed deltas within each DGP replicate",
            "compute condition mean and two-sided 95% t interval across 20 DGP replicates",
        ],
        "figure": {
            "path": str(png_path.relative_to(ROOT)),
            "control": "N_null",
            "unit": "independent DGP replicate",
            "uncertainty": "two-sided 95% t interval, df=19",
            "interpretation": "negative values favor signed; visualisation is descriptive and makes no external-validity claim",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": True, "rows": len(rows), "replicate_rows": len(replicate_rows), "condition_rows": len(summary_rows), "figure": str(png_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
