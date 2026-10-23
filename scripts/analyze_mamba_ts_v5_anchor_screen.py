#!/usr/bin/env python3
"""Read the validation-only MambaTS v5 AutoCon anchor PEMS screen.

The reader uses the best pure forecasting validation loss, never test metrics,
and optionally pairs the result with exact-coordinate validation CSVs from
the nested v4 control.  Checkpoint probes report whether the learned ACF
memory and signed-anchor ReZero gates moved away from initialization.
"""

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


def _checkpoint_gate(path: Path) -> tuple[float, float]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if "ts_acf_memory_scale" not in state:
        raise ValueError(f"checkpoint has no v5/v4 ACF memory gate: {path}")
    values = state["ts_acf_memory_scale"].detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError(f"v5/v4 ACF memory gate is empty: {path}")
    return float(values.abs().mean()), float(values.abs().max())


def _checkpoint_repr_gate(path: Path) -> tuple[float, float]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    values = state.get("ts_acf_repr_scale")
    if values is None:
        return 0.0, 0.0
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError(f"v5 ACF representation gate is empty: {path}")
    return float(values.abs().mean()), float(values.abs().max())


def _checkpoint_anchor_gate(path: Path) -> tuple[float, float]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    values = state.get("ts_acf_anchor_scale")
    if values is None:
        raise ValueError(f"checkpoint has no v5 signed ACF anchor gate: {path}")
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError(f"v5 signed ACF anchor gate is empty: {path}")
    return float(values.abs().mean()), float(values.abs().max())


def read_phase(phase_root: Path) -> list[dict]:
    manifest_path = phase_root / "logs" / "job_manifest.jsonl"
    rows = []
    with manifest_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != 16:
        raise ValueError(f"expected 16 manifest records, found {len(records)}")
    seen = set()
    for record in records:
        if record.get("status") != "ok":
            raise ValueError(f"non-ok manifest record: {record.get('job_id')}")
        result_dir = Path(str(record["expected_artifact"])).parent
        config_path = result_dir / "config.json"
        config = _json(config_path)
        if str(config.get("mamba_ts_variant", "")).lower() != "v5":
            raise ValueError(f"expected v5 config: {config_path}")
        if str(record.get("model")) != "MambaTSv5AutoCon":
            raise ValueError(f"expected MambaTSv5AutoCon record: {record.get('job_id')}")
        checkpoint_root = Path(_flag_value(record.get("command", []), "--checkpoints"))
        checkpoint_dir = checkpoint_root / result_dir.name
        logs = _json(checkpoint_dir / "loss_logs.json")
        values = [float(item) for item in logs.get("val_loss", [])]
        if not values or not all(math.isfinite(item) for item in values):
            raise ValueError(f"invalid validation trajectory: {checkpoint_dir}")
        best = min(values)
        key = (str(record["dataset"]), int(record["pred_len"]), int(record["seed"]))
        if key in seen:
            raise ValueError(f"duplicate coordinate: {key}")
        seen.add(key)
        gate_mean, gate_max = _checkpoint_gate(checkpoint_dir / "checkpoint.pth")
        repr_gate_mean, repr_gate_max = _checkpoint_repr_gate(
            checkpoint_dir / "checkpoint.pth"
        )
        anchor_gate_mean, anchor_gate_max = _checkpoint_anchor_gate(
            checkpoint_dir / "checkpoint.pth"
        )
        rows.append({
            "dataset": key[0], "pred_len": key[1], "seed": key[2],
            "model": str(record["model"]), "best_val_mse": best,
            "best_epoch": int(logs.get("best_epoch", values.index(best) + 1)),
            "best_epoch_autocon_loss": float(logs.get("autocon_loss", [float("nan")])[values.index(best)]),
            "elapsed_sec": float(record.get("elapsed_sec", float("nan"))),
            "acf_gate_abs_mean": gate_mean, "acf_gate_abs_max": gate_max,
            "acf_repr_gate_abs_mean": repr_gate_mean,
            "acf_repr_gate_abs_max": repr_gate_max,
            "acf_anchor_gate_abs_mean": anchor_gate_mean,
            "acf_anchor_gate_abs_max": anchor_gate_max,
            "config_path": str(config_path),
            "test_metrics_produced": False,
        })
    return sorted(rows, key=lambda row: (row["dataset"], row["pred_len"], row["seed"]))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        result.append({
            "dataset": str(row["dataset"]), "pred_len": int(row["pred_len"]),
            "seed": int(row["seed"]), "model": str(row["model"]),
            "best_val_mse": float(row["best_val_mse"]),
        })
    return result


def pair(left: list[dict], right: list[dict], right_model: str) -> list[dict]:
    left_index = {(row["dataset"], row["pred_len"], row["seed"]): row for row in left}
    right_index = {
        (row["dataset"], row["pred_len"], row["seed"]): row
        for row in right if row["model"] == right_model
    }
    if set(left_index) != set(right_index):
        raise ValueError(
            f"exact coordinate mismatch against {right_model}: "
            f"left-only={sorted(set(left_index) - set(right_index))}, "
            f"right-only={sorted(set(right_index) - set(left_index))}"
        )
    rows = []
    for key in sorted(left_index):
        candidate = float(left_index[key]["best_val_mse"])
        reference = float(right_index[key]["best_val_mse"])
        rows.append({
            "dataset": key[0], "pred_len": key[1], "seed": key[2],
            "reference_model": right_model, "candidate_val_mse": candidate,
            "reference_val_mse": reference,
            "relative_change_pct": 100.0 * (candidate - reference) / reference,
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict]) -> list[dict]:
    output = []
    for horizon in (12, 96):
        values = [float(row["best_val_mse"]) for row in rows if row["pred_len"] == horizon]
        gates = [float(row["acf_gate_abs_mean"]) for row in rows if row["pred_len"] == horizon]
        repr_gates = [
            float(row.get("acf_repr_gate_abs_mean", 0.0))
            for row in rows if row["pred_len"] == horizon
        ]
        anchor_gates = [
            float(row.get("acf_anchor_gate_abs_mean", 0.0))
            for row in rows if row["pred_len"] == horizon
        ]
        output.append({
            "pred_len": horizon, "n": len(values),
            "mean_best_val_mse": statistics.mean(values) if values else "",
            "median_best_val_mse": statistics.median(values) if values else "",
            "mean_acf_gate_abs": statistics.mean(gates) if gates else "",
            "nonzero_gate_rows": sum(value > 1e-6 for value in gates),
            "mean_acf_repr_gate_abs": statistics.mean(repr_gates) if repr_gates else "",
            "nonzero_repr_gate_rows": sum(value > 1e-6 for value in repr_gates),
            "mean_acf_anchor_gate_abs": statistics.mean(anchor_gates) if anchor_gates else "",
            "nonzero_anchor_gate_rows": sum(value > 1e-6 for value in anchor_gates),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, action="append", default=[])
    parser.add_argument("--reference-model", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_phase(args.phase_root.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "validation_rows.csv", rows)
    _write_csv(output / "summary.csv", _summary(rows))
    lines = [
        "# MambaTS v5 AutoCon signed-anchor screen",
        "",
        "Validation-only exact-coordinate screen; test metrics were withheld.",
        "The ACF prior is fitted on the training split; the anchor reads only causal observed-window values, and all ReZero gates are read from each checkpoint.",
        "",
        f"Rows: **{len(rows)}**; coordinates: **{len({(r['dataset'], r['pred_len'], r['seed']) for r in rows})}**.",
        "",
        "| horizon | rows | mean best validation MSE | median MSE | mean abs memory gate | nonzero memory | mean abs repr gate | nonzero repr | mean abs anchor gate | nonzero anchor |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in _summary(rows):
        lines.append(
            f"| {item['pred_len']} | {item['n']} | {float(item['mean_best_val_mse']):.6f} | "
            f"{float(item['median_best_val_mse']):.6f} | {float(item['mean_acf_gate_abs']):.6f} | "
            f"{item['nonzero_gate_rows']} | {float(item['mean_acf_repr_gate_abs']):.6f} | "
            f"{item['nonzero_repr_gate_rows']} | {float(item['mean_acf_anchor_gate_abs']):.6f} | "
            f"{item['nonzero_anchor_gate_rows']} |"
        )
    reference_rows = []
    for reference_path in args.reference_csv:
        reference_rows.extend(read_csv(reference_path.resolve()))
    reference_models = args.reference_model or ["MambaTSv4AutoCon"]
    for reference_model in dict.fromkeys(reference_models):
        paired = pair(rows, reference_rows, reference_model)
        stem = reference_model.lower()
        _write_csv(output / f"{stem}_deltas.csv", paired)
        for horizon in (12, 96):
            values = [float(row["relative_change_pct"]) for row in paired if row["pred_len"] == horizon]
            lines.extend([
                "", f"## v5 versus {reference_model}, H={horizon}", "",
                f"Exact coordinates: **{len(values)}**; wins: **{sum(value < 0 for value in values)}/{len(values)}**; "
                f"mean relative change: **{statistics.mean(values):.3f}%**.",
            ])
    (output / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed {len(rows)} v5 validation rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
