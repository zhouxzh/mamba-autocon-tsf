#!/usr/bin/env python
"""Aggregate paper-matrix metrics, confidence intervals, and figures."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.acf_relation_contract import resolve_acf_source
from utils.model_identity import as_bool, canonical_model


MAMBA_BACKBONES = ("Mamba1", "Mamba2")


def markdown_table(frame):
    """Render a table even when the optional ``tabulate`` package is absent."""
    try:
        return frame.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        return frame.to_string(index=False)


def ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    return float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))


def load_trained(results_dir):
    rows = []
    for config_path in sorted(results_dir.glob("*/config.json")):
        result_dir = config_path.parent
        metrics_path = result_dir / "metrics.npy"
        if not metrics_path.exists():
            continue
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        values = np.asarray(np.load(metrics_path), dtype=float).reshape(-1)
        if values.size < 3:
            continue
        original_metrics_path = result_dir / "original_metrics.npy"
        original_values = (np.asarray(np.load(original_metrics_path), dtype=float).reshape(-1)
                           if original_metrics_path.is_file() else np.full(5, np.nan))
        if original_values.size < 5:
            original_values = np.pad(original_values, (0, 5 - original_values.size), constant_values=np.nan)
        model_name = canonical_model(config)
        is_autocon = as_bool(config.get("AutoCon", False))
        rows.append({
            "dataset": config.get("data"), "model": model_name,
            "features": config.get("features"), "seq_len": config.get("seq_len"),
            "pred_len": config.get("pred_len"), "seed": config.get("run_seed", config.get("seed")),
            "enc_in": config.get("enc_in"), "dec_in": config.get("dec_in"),
            "c_out": config.get("c_out"),
            "autocon": is_autocon,
            "acf_source": resolve_acf_source(config, default=""),
            # Keep lambda=0 for models that do not optimize AutoCon.  The
            # runner passes the default lambda to every job, but displaying
            # it on DLinear/Mamba2 would falsely imply an auxiliary loss.
            "autocon_lambda": config.get("AutoCon_lambda", 0.0) if is_autocon else 0.0,
            "mae": values[0], "mse": values[1], "rmse": values[2],
            "mape": values[3] if values.size > 3 else np.nan,
            "mspe": values[4] if values.size > 4 else np.nan,
            "original_mae": original_values[0], "original_mse": original_values[1],
            "original_rmse": original_values[2], "original_mape": original_values[3],
            "original_mspe": original_values[4],
            "source": "trained", "config_path": str(config_path),
        })
    return rows


def load_naive(path):
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "dataset": row["dataset"], "model": "NaiveLast" if row["model"] == "last" else "NaiveMean",
            "features": row["features"], "seq_len": int(row["seq_len"]),
            "pred_len": int(row["pred_len"]), "seed": 0, "autocon": False,
            "enc_in": row.get("enc_in", np.nan), "dec_in": row.get("dec_in", np.nan),
            "c_out": row.get("c_out", np.nan),
            "acf_source": "", "autocon_lambda": 0.0,
            "mae": row["scaled_mae"], "mse": row["scaled_mse"], "rmse": row["scaled_rmse"],
            "mape": np.nan, "mspe": np.nan, "source": "naive", "config_path": str(path),
            "original_mae": row.get("original_mae", np.nan),
            "original_mse": row.get("original_mse", np.nan),
            "original_rmse": row.get("original_rmse", np.nan),
            "original_mape": np.nan, "original_mspe": np.nan,
        })
    return rows


def summarize(frame):
    group_cols = ["dataset", "model", "features", "seq_len", "pred_len", "source", "autocon_lambda"]
    summaries = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        for metric in ("mae", "mse", "rmse"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = ci95(values)
        row["n_runs"] = int(len(group))
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return summary
    summary["rank_mse"] = summary.groupby(["dataset", "features", "seq_len", "pred_len"])["mse_mean"].rank(method="min")
    return summary.sort_values(["dataset", "pred_len", "mse_mean"])


def plot_summary(summary, output):
    if summary.empty:
        return
    datasets = list(summary["dataset"].drop_duplicates())
    cols = 2
    rows = int(math.ceil(len(datasets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.1 * rows), squeeze=False, constrained_layout=True)
    for axis, dataset in zip(axes.flat, datasets):
        subset = summary[summary["dataset"] == dataset]
        for (model, lambda_value), curve in subset.groupby(["model", "autocon_lambda"], dropna=False):
            curve = curve.sort_values("pred_len")
            label = str(model)
            if str(model).endswith("AutoCon"):
                label = f"{model} (lambda={float(lambda_value):g})"
            axis.errorbar(curve["pred_len"], curve["mse_mean"], yerr=curve["mse_ci95"],
                          marker="o", linewidth=1.2, capsize=2, label=label)
        axis.set_title(dataset)
        axis.set_xlabel("prediction horizon")
        axis.set_ylabel("MSE (train-standardized)")
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, frameon=False)
    for axis in axes.flat[len(datasets):]:
        axis.axis("off")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def relative_markdown_path(target, source_dir):
    return Path(os.path.relpath(target, start=source_dir)).as_posix()


def write_markdown(summary, output, artifact_dir):
    rank_summary = relative_markdown_path(artifact_dir / "rank_summary.csv", output.parent)
    lines = ["# Forecasting comparison", "", "Metrics are MSE/MAE in the train-standardized space used by the repository.",
             "Each trained row is a seed; `ci95` is a Student-t 95% interval across seeds.",
             "The summary includes exploratory AutoCon lambda variants; use [paper tables](paper_tables.md) and "
             f"[rank_summary.csv]({rank_summary}) for the fixed-lambda primary comparison.", ""]
    if summary.empty:
        lines.append("No metrics were found.")
    else:
        table = summary[["dataset", "model", "autocon_lambda", "pred_len", "n_runs", "mse_mean", "mse_ci95", "mae_mean", "mae_ci95", "rank_mse"]].copy()
        for column in ("mse_mean", "mse_ci95", "mae_mean", "mae_ci95"):
            table[column] = table[column].map(lambda value: f"{value:.6f}")
        lines.append(markdown_table(table))
        lines.extend(["", "## Interpretation limits", "", "- Existing GLD logs/checkpoints are not included: the source CSV is absent from this repository.",
                      "- Overlapping forecast windows are aggregated as individual windows; confidence intervals are across seeds, not windows.",
                      "- MAPE/MSPE on standardized series are not used for ranking.",
                      "- `rank_mse` is a descriptive rank over all rows (including naive and exploratory lambda variants); it is not the primary statistical ranking."])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_statistical_tables(frame, summary, output_dir):
    """Write paired Mamba ablations and improvement-vs-naive tables."""
    if frame.empty:
        return
    trained = frame[frame["source"] == "trained"].copy()
    naive = frame[frame["source"] == "naive"].copy()
    comparisons = []
    base_cols = ["dataset", "features", "seq_len", "pred_len"]
    for keys, group in trained.groupby(base_cols + ["autocon_lambda"], dropna=False):
        key = dict(zip(base_cols + ["autocon_lambda"], keys))
        base_key = {column: key[column] for column in base_cols}
        base = naive[(naive[base_cols] == pd.Series(base_key)).all(axis=1)]
        baseline_mse = float(base["mse"].min()) if not base.empty else np.nan
        for model, model_group in group.groupby("model"):
            value = float(model_group["mse"].mean())
            comparisons.append({**key, "model": model, "n_runs": len(model_group),
                                "mse_mean": value, "mse_ci95": ci95(model_group["mse"]),
                                "best_naive_mse": baseline_mse,
                                "improvement_vs_best_naive_pct":
                                100.0 * (baseline_mse - value) / baseline_mse if np.isfinite(baseline_mse) else np.nan})
    pd.DataFrame(comparisons).to_csv(output_dir / "improvement_vs_naive.csv", index=False)

    paired = []
    for keys, group in trained.groupby(base_cols, dropna=False):
        for backbone in MAMBA_BACKBONES:
            contrastive_model = f"{backbone}AutoCon"
            left = group[group["model"] == backbone].set_index("seed")["mse"]
            for lambda_value, right_group in group[group["model"] == contrastive_model].groupby("autocon_lambda"):
                right = right_group.set_index("seed")["mse"]
                joined = pd.concat([left.rename("plain"), right.rename("autocon")], axis=1).dropna()
                if joined.empty:
                    continue
                difference = joined["autocon"] - joined["plain"]
                row = dict(zip(base_cols, keys))
                row.update({"backbone": backbone, "plain_model": backbone,
                            "autocon_model": contrastive_model, "autocon_lambda": lambda_value,
                            "n_pairs": len(joined), "mean_mse_difference": float(difference.mean()),
                            "relative_change_pct": float(100.0 * difference.mean() / joined["plain"].mean())})
                if len(joined) >= 3:
                    from scipy.stats import ttest_rel, wilcoxon
                    row["paired_t_p"] = float(ttest_rel(joined["autocon"], joined["plain"]).pvalue)
                    try:
                        row["wilcoxon_p"] = float(wilcoxon(difference).pvalue)
                    except ValueError:
                        row["wilcoxon_p"] = np.nan
                else:
                    row["paired_t_p"] = np.nan
                    row["wilcoxon_p"] = np.nan
                paired.append(row)
    pd.DataFrame(paired).to_csv(output_dir / "mamba_paired_ablation.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="One explicit phase result directory; do not mix phases.")
    parser.add_argument("--naive", type=Path, default=Path("artifacts/baselines/naive_metrics.csv"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--docs_dir", type=Path, default=Path("docs/reports"),
                        help="Directory for the human-readable Markdown report.")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    naive_path = args.naive.resolve()
    output_dir = args.output_dir.resolve()
    docs_dir = args.docs_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    rows = load_trained(results_dir) + load_naive(naive_path)
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    frame.to_csv(output_dir / "raw_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_statistical_tables(frame, summary, output_dir)
    plot_summary(summary, output_dir / "mse_by_horizon.png")
    write_markdown(summary, docs_dir / "forecasting-comparison.md", output_dir)
    print(f"Collected {len(frame)} raw rows and {len(summary)} grouped rows into {output_dir}; report written to {docs_dir}")


if __name__ == "__main__":
    main()
