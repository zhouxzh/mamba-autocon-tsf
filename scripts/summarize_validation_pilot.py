#!/usr/bin/env python3
"""Summarize validation-only frozen exploratory runs.

The training runner deliberately omits ``metrics.npy`` when ``skip_test`` is
enabled.  This reader follows each successful manifest record to its result
configuration and checkpoint ``loss_logs.json`` and reports only the
forecasting validation loss.  It never reads test artifacts or selects a
model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_min(values: object) -> tuple[float | None, int | None]:
    if not isinstance(values, list):
        return None, None
    finite = [(float(value), index + 1) for index, value in enumerate(values)
              if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not finite:
        return None, None
    return min(finite, key=lambda item: item[0])


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _manifest_records(root: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    for manifest in sorted(root.glob("*/logs/job_manifest.jsonl")):
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid manifest JSON: {manifest}:{line_number}") from exc
                if not isinstance(record, dict) or not record.get("job_id"):
                    raise ValueError(f"invalid manifest record: {manifest}:{line_number}")
                latest[str(record["job_id"])] = record
    return list(latest.values())


def load_rows(root: Path) -> list[dict]:
    rows = []
    for record in _manifest_records(root):
        if record.get("status") != "ok":
            continue
        setting_value = record.get("setting_dir")
        if not setting_value:
            raise ValueError(f"successful record has no setting_dir: {record.get('job_id')}")
        result_dir = Path(str(setting_value)).resolve()
        config_path = result_dir / "config.json"
        phase = str(record.get("phase"))
        checkpoint_path = root / phase / "checkpoints" / result_dir.name / "loss_logs.json"
        config = _read_json(config_path)
        logs = _read_json(checkpoint_path)
        best_value, best_epoch = _finite_min(logs.get("val_loss"))
        if best_value is None:
            raise ValueError(f"no finite val_loss for {record.get('job_id')}")
        mse_values = logs.get("mse_loss", [])
        ac_values = logs.get("autocon_loss", [])
        index = (best_epoch - 1) if best_epoch is not None else -1
        rows.append({
            "job_id": str(record["job_id"]),
            "phase": phase,
            "dataset": str(config.get("data", record.get("dataset"))),
            "model": str(config.get("model", record.get("model"))),
            "features": str(config.get("features", record.get("features", ""))),
            "seq_len": int(config.get("seq_len", 0)),
            "pred_len": int(config.get("pred_len", record.get("pred_len", 0))),
            "seed": int(config.get("run_seed", config.get("seed", record.get("seed", 0)))),
            "enc_in": int(config.get("enc_in", record.get("enc_in", 0))),
            "c_out": int(config.get("c_out", record.get("c_out", 0))),
            "autocon": _as_bool(config.get("AutoCon", False)),
            "autocon_lambda": float(config.get("AutoCon_lambda", 0.0)) if _as_bool(config.get("AutoCon", False)) else 0.0,
            "best_val_mse": best_value,
            "best_epoch": int(best_epoch) if best_epoch is not None else "",
            "final_val_mse": float(logs["val_loss"][-1]),
            "best_epoch_train_mse": float(mse_values[index]) if isinstance(mse_values, list) and mse_values and -len(mse_values) <= index < len(mse_values) else "",
            "best_epoch_autocon_loss": float(ac_values[index]) if isinstance(ac_values, list) and ac_values and -len(ac_values) <= index < len(ac_values) else "",
            "elapsed_sec": float(record.get("elapsed_sec", float("nan"))),
            "config_path": str(config_path),
            "loss_log_path": str(checkpoint_path),
        })
    return sorted(rows, key=lambda row: (row["dataset"], row["pred_len"], row["model"], row["seed"]))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["features"], row["seq_len"],
               row["pred_len"], row["autocon_lambda"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        values = [float(row["best_val_mse"]) for row in group]
        result.append({
            "dataset": key[0], "model": key[1], "features": key[2],
            "seq_len": key[3], "pred_len": key[4], "autocon_lambda": key[5],
            "n_seeds": len(values),
            "seeds": ",".join(str(row["seed"]) for row in group),
            "best_val_mse_mean": _mean(values),
            "best_val_mse_std": _std(values),
            "best_val_mse_min": min(values),
        })
    return sorted(result, key=lambda row: (row["dataset"], row["pred_len"], row["best_val_mse_mean"]))


def paired_deltas(rows: list[dict]) -> list[dict]:
    index = {(row["dataset"], row["features"], row["seq_len"], row["pred_len"], row["seed"], row["model"]): row
             for row in rows}
    contrasts = (
        ("MambaTS", "Mamba2"),
        ("MambaTSAutoCon", "MambaTS"),
        ("MambaTSAutoCon", "Mamba2AutoCon"),
        ("MambaTSv2", "Mamba2"),
        ("MambaTSv2", "MambaTS"),
        ("MambaTSv2AutoCon", "MambaTSv2"),
        ("MambaTSv2AutoCon", "Mamba2AutoCon"),
        ("MambaTSv3", "Mamba2"),
        ("MambaTSv3", "MambaTS"),
        ("MambaTSv3", "DLinear"),
        ("MambaTSv3", "PatchTST"),
        ("MambaTSv3", "TimesNet"),
        ("MambaTSv3AutoCon", "MambaTSv3"),
        ("MambaTSv3AutoCon", "Mamba2AutoCon"),
    )
    result = []
    keys = {(row["dataset"], row["features"], row["seq_len"], row["pred_len"], row["seed"])
            for row in rows}
    for dataset, features, seq_len, pred_len, seed in sorted(keys):
        for candidate, reference in contrasts:
            left = index.get((dataset, features, seq_len, pred_len, seed, candidate))
            right = index.get((dataset, features, seq_len, pred_len, seed, reference))
            if left is None or right is None:
                continue
            delta = float(left["best_val_mse"]) - float(right["best_val_mse"])
            result.append({
                "dataset": dataset, "features": features, "seq_len": seq_len,
                "pred_len": pred_len, "seed": seed,
                "candidate": candidate, "reference": reference,
                "candidate_best_val_mse": float(left["best_val_mse"]),
                "reference_best_val_mse": float(right["best_val_mse"]),
                "delta_candidate_minus_reference": delta,
                "relative_change_pct": 100.0 * delta / float(right["best_val_mse"]),
            })
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict], summary: list[dict], deltas: list[dict]) -> None:
    lines = [
        "# PEMS validation-only exploratory readout",
        "",
        "Primary metric is the minimum validation MSE in `loss_logs.json`; each row is one seed.",
        "Test metrics are intentionally absent. This readout is descriptive and does not select a model.",
        "",
        f"Successful validation rows: **{len(rows)}**; paired rows available: **{len(deltas)}**.",
        "",
        "## Seed-level rows",
        "",
        "| dataset | model | horizon | seed | best val MSE | epoch | AutoCon lambda | elapsed s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {dataset} | {model} | {pred_len} | {seed} | {mse} | {epoch} | {lam} | {elapsed} |".format(
            dataset=row["dataset"], model=row["model"], pred_len=row["pred_len"], seed=row["seed"],
            mse=_fmt(float(row["best_val_mse"])), epoch=row["best_epoch"],
            lam=_fmt(float(row["autocon_lambda"])), elapsed=_fmt(float(row["elapsed_sec"])),
        ))
    lines.extend(["", "## Grouped rows", "", "| dataset | model | horizon | seeds | mean MSE | std | min |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in summary:
        lines.append("| {dataset} | {model} | {pred_len} | {n_seeds} ({seeds}) | {mean} | {std} | {minimum} |".format(
            dataset=row["dataset"], model=row["model"], pred_len=row["pred_len"], n_seeds=row["n_seeds"],
            seeds=row["seeds"], mean=_fmt(row["best_val_mse_mean"]), std=_fmt(row["best_val_mse_std"]),
            minimum=_fmt(row["best_val_mse_min"]),
        ))
    lines.extend(["", "## Exact paired deltas", "", "| dataset | horizon | seed | candidate | reference | candidate - reference | relative % |", "|---|---:|---:|---|---|---:|---:|"])
    for row in deltas:
        lines.append("| {dataset} | {pred_len} | {seed} | {candidate} | {reference} | {delta} | {relative} |".format(
            dataset=row["dataset"], pred_len=row["pred_len"], seed=row["seed"],
            candidate=row["candidate"], reference=row["reference"],
            delta=_fmt(row["delta_candidate_minus_reference"]), relative=_fmt(row["relative_change_pct"]),
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    summary = summarize(rows)
    deltas = paired_deltas(rows)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "validation_summary.csv", summary)
    _write_csv(output / "paired_deltas.csv", deltas)
    _write_markdown(output / "validation_readout.md", rows, summary, deltas)
    print(f"Collected {len(rows)} validation rows, {len(summary)} groups, {len(deltas)} paired deltas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
