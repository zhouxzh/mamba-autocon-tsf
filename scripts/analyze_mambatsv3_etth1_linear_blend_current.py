#!/usr/bin/env python3
"""Strict reader for the current-source ETTh1 AutoCon linear-blend screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (3161, 3162, 3163)
ARMS = {
    "blend_h96": ("blend", 3, {"MambaTSv3AutoCon"}),
    "control_h96": ("control", 3, {"MambaTSv3AutoCon"}),
    "baselines_h96": ("baseline", 6, {"RLinear", "PatchTST"}),
}


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_arm(root: Path, kind: str, expected_jobs: int, expected_models: set[str]) -> tuple[list[dict], str]:
    phase = load(root / "phase_validation.json")
    if phase.get("valid") is not True or phase.get("expected_jobs") != expected_jobs:
        raise ValueError("phase validation failed: {}".format(root))
    source = load(root / "provenance.json").get("source_tree_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise ValueError("missing source hash")
    rows = []
    seen = set()
    for raw in (root / "logs" / "job_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = record.get("job_id")
        if job_id in seen or record.get("status") != "ok":
            raise ValueError("invalid manifest record: {}".format(job_id))
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError("source mismatch: {}".format(job_id))
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        model = str(config.get("model"))
        if model not in expected_models or config.get("skip_test") is not True:
            raise ValueError("model/test contract mismatch: {}".format(job_id))
        if any((result / name).exists() for name in ("metrics.npy", "original_metrics.npy", "stream_metrics.json")):
            raise ValueError("test artifact found: {}".format(job_id))
        if kind == "blend":
            if config.get("mamba_ts_repr_linear_expert") is not True or config.get("mamba_ts_repr_linear_expert_mode") != "blend":
                raise ValueError("blend contract mismatch: {}".format(job_id))
            if float(config.get("mamba_ts_repr_linear_blend_bias")) != -4.0:
                raise ValueError("blend bias mismatch: {}".format(job_id))
        elif config.get("mamba_ts_repr_linear_expert") is not False:
            raise ValueError("non-blend arm received expert: {}".format(job_id))
        command = record["command"]
        checkpoint_root = Path(command[command.index("--checkpoints") + 1]) / result.name
        losses = [float(value) for value in load(checkpoint_root / "loss_logs.json").get("val_loss", [])]
        if not losses:
            raise ValueError("missing validation loss: {}".format(job_id))
        rows.append({
            "dataset": str(config["data"]), "horizon": int(config["pred_len"]), "seed": int(config["seed"]),
            "model": model, "arm": kind, "best_val_mse": min(losses), "elapsed_sec": float(record["elapsed_sec"]),
        })
    if len(rows) != expected_jobs:
        raise ValueError("manifest count mismatch: {}".format(root))
    expected = {("ETTh1", 96, seed, model) for seed in SEEDS for model in expected_models}
    found = {(row["dataset"], row["horizon"], row["seed"], row["model"]) for row in rows}
    if found != expected:
        raise ValueError("coordinate mismatch: {}".format(root))
    return rows, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    source_hash = None
    for directory, (kind, count, models) in ARMS.items():
        arm_rows, source = read_arm(args.artifact_root / directory, kind, count, models)
        if source_hash is None:
            source_hash = source
        elif source != source_hash:
            raise ValueError("arms have different source snapshots")
        rows.extend(arm_rows)

    by_key = {(row["arm"], row["model"], row["seed"]): row for row in rows}
    blend = [by_key[("blend", "MambaTSv3AutoCon", seed)] for seed in SEEDS]
    control = [by_key[("control", "MambaTSv3AutoCon", seed)] for seed in SEEDS]
    baselines = {model: [by_key[("baseline", model, seed)] for seed in SEEDS] for model in ("RLinear", "PatchTST")}
    paired = []
    for seed, blend_row, control_row in zip(SEEDS, blend, control):
        paired.append({
            "seed": seed, "blend_mse": blend_row["best_val_mse"], "control_mse": control_row["best_val_mse"],
            "blend_vs_control_pct": 100.0 * (blend_row["best_val_mse"] / control_row["best_val_mse"] - 1.0),
        })
    blend_mean = statistics.fmean(row["best_val_mse"] for row in blend)
    control_mean = statistics.fmean(row["best_val_mse"] for row in control)
    baseline_means = {model: statistics.fmean(row["best_val_mse"] for row in values) for model, values in baselines.items()}
    strongest_model = min(baseline_means, key=baseline_means.get)
    strongest_mean = baseline_means[strongest_model]
    summary = {
        "blend_mean": blend_mean, "control_mean": control_mean, "blend_vs_control_pct": 100.0 * (blend_mean / control_mean - 1.0),
        "control_wins": sum(row["blend_mse"] < row["control_mse"] for row in paired), "strongest_baseline": strongest_model,
        "strongest_baseline_mean": strongest_mean, "blend_vs_strongest_pct": 100.0 * (blend_mean / strongest_mean - 1.0),
        "source_tree_sha256": source_hash,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / "paired_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paired_gate = summary["blend_vs_control_pct"] < 0.0 and summary["control_wins"] >= 2
    external_gate = summary["blend_vs_strongest_pct"] <= 2.0
    gate = paired_gate and external_gate
    lines = [
        "# Current-source ETTh1 AutoCon linear blend", "",
        "Validation-only strict readout. No test metric or prediction artifact was read or produced.", "",
        "| Arm | Mean validation MSE | SD | Mean job sec |", "| --- | ---: | ---: | ---: |",
        "| Blend | {:.6f} | {:.6f} | {:.3f} |".format(blend_mean, statistics.stdev(row["best_val_mse"] for row in blend), statistics.fmean(row["elapsed_sec"] for row in blend)),
        "| Exact v3 control | {:.6f} | {:.6f} | {:.3f} |".format(control_mean, statistics.stdev(row["best_val_mse"] for row in control), statistics.fmean(row["elapsed_sec"] for row in control)),
    ]
    for model, values in baselines.items():
        lines.append("| {} | {:.6f} | {:.6f} | {:.3f} |".format(model, baseline_means[model], statistics.stdev(row["best_val_mse"] for row in values), statistics.fmean(row["elapsed_sec"] for row in values)))
    lines.extend([
        "", "| Contrast | Result | Gate |", "| --- | ---: | --- |",
        "| Blend vs exact v3 | {:+.3f}% ({}/{}) | {} |".format(summary["blend_vs_control_pct"], summary["control_wins"], len(SEEDS), "PASS" if paired_gate else "FAIL"),
        "| Blend vs strongest baseline ({}) | {:+.3f}% | {} |".format(strongest_model, summary["blend_vs_strongest_pct"], "PASS" if external_gate else "FAIL"),
        "", "Overall gate: **{}**. Source snapshot: `{}`.".format("PASS" if gate else "FAIL", source_hash),
        "This is a single-domain validation screen and does not establish a causal AutoCon effect or authorize a held-out release.",
    ])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "gate": gate, "summary": summary, "test_metrics_produced": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
