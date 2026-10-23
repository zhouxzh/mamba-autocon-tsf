#!/usr/bin/env python3
"""Build a paper-facing table from the frozen held-out release readouts.

The script only reads existing ``paired_test_summary.csv`` files. It does not
rerun a reader, training job, or metric computation, and writes outside the
frozen ``artifacts/paper_final`` roots by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


RELEASES = (
    {
        "release": "PEMS H12",
        "directory": "pems_mambatsv3_d24_test_release_20260926",
        "summary": "paired_test_summary.csv",
        "dataset_column": "dataset",
        "horizon_column": None,
    },
    {
        "release": "METR-LA/PEMS-BAY H12",
        "directory": "metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927",
        "summary": "paired_test_summary.csv",
        "dataset_column": "dataset",
        "horizon_column": None,
    },
    {
        "release": "Weather H96/H336/H720",
        "directory": "mambatsv3_weather_test_release_20260922",
        "summary": "paired_test_summary.csv",
        "dataset_column": None,
        "horizon_column": "horizon",
    },
    {
        "release": "Electricity/Traffic H96 CPU-4",
        "directory": "mambatsv3_stride24_relation_auxiliary_highdim_cpu4_test_release_20261001",
        "summary": "paired_test_summary.csv",
        "dataset_column": "dataset",
        "horizon_column": "horizon",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_rows(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for spec in RELEASES:
        path = repo_root / "artifacts" / "paper_final" / spec["directory"] / "readout" / spec["summary"]
        if not path.is_file():
            raise FileNotFoundError(path)
        rel_path = str(path.relative_to(repo_root))
        source_hashes[rel_path] = sha256_file(path)
        for item in read_rows(path):
            dataset = item[spec["dataset_column"]] if spec["dataset_column"] else "weather"
            horizon = item[spec["horizon_column"]] if spec["horizon_column"] else "12"
            rows.append(
                {
                    "release": spec["release"],
                    "dataset": dataset,
                    "horizon": int(horizon),
                    "baseline": item["baseline"],
                    "candidate_mse": float(item["candidate_mse_mean"]),
                    "baseline_mse": float(item["baseline_mse_mean"]),
                    "mean_mse_change_pct": float(item["mean_mse_change_pct"]),
                    "candidate_wins": int(item["candidate_wins"]),
                    "n": int(item["n"]),
                    "upper_ratio_95_one_sided": float(item["upper_ratio_95_one_sided"]),
                    "noninferior_ratio_1p02": item["noninferior_ratio_1p02"].lower() == "true",
                    "candidate_elapsed_mean_sec": float(item["candidate_elapsed_mean_sec"]),
                    "baseline_elapsed_mean_sec": float(item["baseline_elapsed_mean_sec"]),
                    "candidate_elapsed_faster": (
                        item.get("candidate_elapsed_faster", "").lower() == "true"
                        if item.get("candidate_elapsed_faster", "")
                        else None
                    ),
                }
            )
    rows.sort(key=lambda row: (row["release"], row["dataset"], row["horizon"], row["baseline"]))
    return rows, source_hashes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen Held-out Main Table",
        "",
        "Generated from existing `artifacts/paper_final` paired summaries. No test metric was recomputed.",
        "",
        "| Release | Dataset | H | Baseline | Candidate MSE | Baseline MSE | Change (%) | Wins | NI <=1.02 | Candidate faster |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        faster = "n/a" if row["candidate_elapsed_faster"] is None else str(row["candidate_elapsed_faster"])
        lines.append(
            f"| {row['release']} | {row['dataset']} | {row['horizon']} | {row['baseline']} | "
            f"{row['candidate_mse']:.6f} | {row['baseline_mse']:.6f} | {row['mean_mse_change_pct']:.3f} | "
            f"{row['candidate_wins']}/{row['n']} | {row['noninferior_ratio_1p02']} | {faster} |"
        )
    lines.extend(
        [
            "",
            "Scope: only the four named frozen held-out releases and their registered comparators. The Electricity/Traffic release retains its registered Traffic/iTransformer failure cell. This table does not support universal accuracy or speed claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/exploratory/paper_main_table_20260928"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    rows, source_hashes = build_rows(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "main_table.csv", rows)
    (output_dir / "main_table.md").write_text(render_markdown(rows), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "scope": "frozen held-out paper releases only",
        "row_count": len(rows),
        "source_summary_sha256": source_hashes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
