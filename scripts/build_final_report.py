#!/usr/bin/env python
"""Build a confirmatory report from frozen-lambda results.

Unlike the historical pilot table builder, this script never chooses a lambda
from test MSE.  It reads the validation-selection JSON, filters the matching
contrastive rows, computes seed-paired deltas, and reports a non-overlapping
forecast-origin sensitivity check from saved predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_identity import as_bool, canonical_model


AUTOCON_MODELS = {
    "AutoConNetAutoCon",
    "Mamba1AutoCon",
    "Mamba2AutoCon",
    "MambaTSAutoCon",
    "MambaTSv2AutoCon",
    "MambaTSv3AutoCon",
    "MambaACFAutoCon",
    "MambaACFVariableAutoCon",
    "MambaACFLiteAutoCon",
    "MambaACF1AutoCon",
}
PLAIN_TO_AUTOCON = {
    "AutoConNet": "AutoConNetAutoCon",
    "Mamba1": "Mamba1AutoCon",
    "Mamba2": "Mamba2AutoCon",
    "MambaTS": "MambaTSAutoCon",
    "MambaTSv2": "MambaTSv2AutoCon",
    "MambaTSv3": "MambaTSv3AutoCon",
    "MambaACF": "MambaACFAutoCon",
    "MambaACFVariable": "MambaACFVariableAutoCon",
    "MambaACF1": "MambaACF1AutoCon",
}


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        return frame.to_string(index=False)


def _close(left: object, right: float) -> bool:
    try:
        return bool(np.isclose(float(left), float(right)))
    except (TypeError, ValueError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(result_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, str]:
    paths = [result_dir / "config.json", result_dir / "metrics.npy",
             result_dir / "original_metrics.npy", result_dir / "channel_metrics.npy",
             result_dir / "stream_metrics.json", result_dir / "stream_samples.npz",
             result_dir / "pred.npy", result_dir / "true.npy",
             result_dir / "training_complete.json", result_dir / "acf_prior.npz"]
    if checkpoint_dir is not None:
        paths.extend([checkpoint_dir / "checkpoint.pth", checkpoint_dir / "loss_logs.json"])
    return {path.name: _sha256(path) for path in paths if path.is_file()}


def _load_selection(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload.get("selected_lambda_by_model", payload)
    result = {str(key): float(value) for key, value in selected.items()}
    if any(not np.isfinite(value) or value <= 0 for value in result.values()):
        raise ValueError("selection contains non-finite or non-positive lambda")
    return result


def _load_selection_payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("selection payload must be a JSON object")
    return payload


def _load_plan(path: Path, phase_name: str) -> tuple[dict, dict]:
    with path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    phase = next((item for item in plan.get("phases", []) if item.get("name") == phase_name), None)
    if phase is None:
        raise ValueError(f"phase not found in plan: {phase_name}")
    return plan, phase


def _expected_keys(plan: dict, phase: dict, selected: dict[str, float]) -> set[tuple]:
    # Import the same expansion logic used by the runner, so the report and
    # execution cannot silently disagree about the registered matrix.
    import sys
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_training_plan import _job_specs
    jobs = _job_specs(plan, phase, selected, allow_unresolved=False)
    return {(job["dataset"], job["model"], job["config"]["features"],
             int(job["config"]["enc_in"]), int(job["config"]["dec_in"]),
             int(job["config"]["c_out"]), int(job["pred_len"]), int(job["seed"]),
             0.0 if job["autocon_lambda"] is None else float(job["autocon_lambda"])) for job in jobs}


def _manifest_coordinate(record: dict) -> tuple:
    return (record.get("dataset"), record.get("model"), record.get("features"),
            int(record.get("enc_in", 1)), int(record.get("dec_in", 1)),
            int(record.get("c_out", 1)), int(record.get("pred_len")),
            int(record.get("seed")), 0.0 if record.get("autocon_lambda") is None
            else float(record["autocon_lambda"]))


def _latest_manifest_records(path: Path) -> tuple[dict[str, dict], int]:
    """Read JSONL using the runner's last-record-wins resume semantics."""
    records = {}
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if "job_id" in record:
                records[record["job_id"]] = record
    return records, malformed


def _validated_manifest_paths(path: Path, expected_keys: set[tuple]) -> dict[tuple, Path]:
    records, malformed = _latest_manifest_records(path)
    if malformed:
        raise SystemExit("manifest contains malformed JSONL rows; refusing report")
    by_key = {}
    for record in records.values():
        key = _manifest_coordinate(record)
        if key not in expected_keys:
            raise SystemExit("manifest contains unregistered coordinate: " + repr(key))
        if key in by_key:
            raise SystemExit("manifest contains duplicate coordinate: " + repr(key))
        if record.get("status") != "ok":
            raise SystemExit("manifest contains non-successful record: " + repr(key))
        artifact = Path(record.get("expected_artifact", "")).resolve()
        if not artifact.is_file():
            raise SystemExit("manifest expected artifact missing: " + repr(key))
        by_key[key] = artifact.parent / "config.json"
    if set(by_key) != expected_keys:
        raise SystemExit("manifest coverage does not match the registered phase")
    return by_key


def _refresh_metrics_from_artifacts(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the manifest-linked metric arrays as the report's numeric source.

    ``raw_results.csv`` is retained as a convenient index, but it is not an
    authority: a stale or edited CSV must not change the confirmatory means.
    Metadata and the first three metrics are checked against each linked
    result directory before the refreshed values are returned.
    """
    refreshed = frame.copy()
    for index, row in refreshed.iterrows():
        config_path = Path(row["config_path"]).resolve()
        try:
            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
            values = np.asarray(np.load(config_path.parent / "metrics.npy"), dtype=float).reshape(-1)
            original_values = np.asarray(
                np.load(config_path.parent / "original_metrics.npy"), dtype=float
            ).reshape(-1)
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"unable to load validated metric artifact: {config_path}: {exc}") from exc
        if (values.size < 3 or original_values.size < 3
                or not np.all(np.isfinite(values[:3]))
                or not np.all(np.isfinite(original_values[:3]))):
            raise SystemExit(f"invalid metric artifact: {config_path.parent / 'metrics.npy'}")

        model_name = canonical_model(config)
        metadata = {
            "dataset": config.get("data"),
            "model": model_name,
            "features": config.get("features"),
            "enc_in": int(config.get("enc_in")),
            "dec_in": int(config.get("dec_in")),
            "c_out": int(config.get("c_out")),
            "seq_len": int(config.get("seq_len")),
            "pred_len": int(config.get("pred_len")),
            "seed": int(config.get("run_seed", config.get("seed"))),
        }
        for key, expected in metadata.items():
            actual = row[key]
            if key in {"enc_in", "dec_in", "c_out", "seq_len", "pred_len", "seed"}:
                okay = int(actual) == expected
            else:
                okay = actual == expected
            if not okay:
                raise SystemExit(
                    f"raw_results metadata disagrees with {config_path}: {key}={actual!r}, expected={expected!r}"
                )
        if as_bool(config.get("AutoCon", False)):
            if not _close(row["autocon_lambda"], float(config.get("AutoCon_lambda"))):
                raise SystemExit(f"raw_results lambda disagrees with {config_path}")

        # collect_results.py stores metrics in the repository order MAE, MSE,
        # RMSE, MAPE, MSPE.  Reassign every metric used by the report from the
        # checked artifact rather than trusting CSV payloads.
        for column, value in zip(("mae", "mse", "rmse", "mape", "mspe"), values):
            if column in refreshed.columns:
                if not np.isfinite(row[column]) or not np.isclose(float(row[column]), float(value), rtol=1e-6, atol=1e-8):
                    raise SystemExit(f"raw_results metric disagrees with {config_path}: {column}")
                refreshed.at[index, column] = float(value)
        for column, value in zip(
                ("original_mae", "original_mse", "original_rmse", "original_mape", "original_mspe"),
                original_values):
            if column in refreshed.columns:
                if not np.isfinite(row[column]) or not np.isclose(float(row[column]), float(value), rtol=1e-6, atol=1e-8):
                    raise SystemExit(f"raw_results original metric disagrees with {config_path}: {column}")
                refreshed.at[index, column] = float(value)
    return refreshed


def _channel_metrics(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export per-channel errors as descriptive heterogeneity, not replicates."""
    rows = []
    columns = ("mae", "mse", "rmse", "original_mae", "original_mse", "original_rmse")
    for _, record in primary.iterrows():
        config_path = Path(record["config_path"]).resolve()
        try:
            values = np.asarray(np.load(config_path.parent / "channel_metrics.npy"), dtype=float)
        except (OSError, ValueError, EOFError, ImportError) as exc:
            raise SystemExit(f"unable to load channel metrics: {config_path}: {exc}") from exc
        channels = int(record["c_out"])
        if values.shape != (channels, len(columns)) or not np.all(np.isfinite(values)):
            raise SystemExit(f"invalid channel metric artifact: {config_path.parent / 'channel_metrics.npy'}")
        for channel, metrics in enumerate(values):
            rows.append({
                "dataset": record["dataset"], "model": record["model"],
                "features": record["features"], "enc_in": int(record["enc_in"]),
                "dec_in": int(record["dec_in"]), "c_out": channels,
                "seq_len": int(record["seq_len"]), "pred_len": int(record["pred_len"]),
                "seed": int(record["seed"]), "channel": channel,
                **{name: float(value) for name, value in zip(columns, metrics)},
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail
    group_cols = ["dataset", "model", "features", "enc_in", "dec_in", "c_out", "seq_len", "pred_len", "channel"]
    summaries = []
    for keys, group in detail.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(len(group))
        for metric_name in columns:
            samples = group[metric_name].to_numpy(dtype=float)
            row[f"{metric_name}_mean"] = float(np.mean(samples))
            row[f"{metric_name}_std"] = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
        summaries.append(row)
    return detail, pd.DataFrame(summaries).sort_values(group_cols)


def _primary(frame: pd.DataFrame, selected: dict[str, float]) -> pd.DataFrame:
    trained = frame[frame["source"] == "trained"].copy()
    keep = ~trained["model"].isin(AUTOCON_MODELS)
    for model, value in selected.items():
        keep |= (trained["model"] == model) & trained["autocon_lambda"].map(lambda item: _close(item, value))
    return trained[keep].copy()


def _paired(raw: pd.DataFrame, selected: dict[str, float]) -> pd.DataFrame:
    primary = _primary(raw, selected)
    rows = []
    keys = ["dataset", "features", "enc_in", "dec_in", "c_out", "seq_len", "pred_len"]
    for group_keys, group in primary.groupby(keys, dropna=False):
        values = group.set_index(["model", "seed"])["mse"]
        base = dict(zip(keys, group_keys))
        for plain, contrastive in PLAIN_TO_AUTOCON.items():
            if plain not in group["model"].values or contrastive not in group["model"].values:
                continue
            plain_values = group[group["model"] == plain].set_index("seed")["mse"]
            contrastive_values = group[group["model"] == contrastive].set_index("seed")["mse"]
            joined = pd.concat([plain_values.rename("plain_mse"), contrastive_values.rename("autocon_mse")], axis=1).dropna()
            if joined.empty:
                continue
            difference = joined["autocon_mse"] - joined["plain_mse"]
            rows.append({**base, "plain_model": plain, "autocon_model": contrastive,
                         "autocon_lambda": selected.get(contrastive), "n_pairs": len(joined),
                         "mean_plain_mse": float(joined["plain_mse"].mean()),
                         "mean_autocon_mse": float(joined["autocon_mse"].mean()),
                         "mean_delta_mse": float(difference.mean()),
                         "relative_change_pct": float(100.0 * difference.mean() / joined["plain_mse"].mean()),
                         "seed_deltas": ";".join(f"{int(seed)}:{value:.8g}" for seed, value in difference.items())})
    return pd.DataFrame(rows)


def _origin_metrics(results_dir: Path, primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in primary.iterrows():
        config_path = Path(record["config_path"])
        result_dir = config_path.parent
        stream_path = result_dir / "stream_metrics.json"
        if stream_path.is_file():
            try:
                with stream_path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                origin = payload["origin"]["scaled"]
                n_origins = int(payload["origin_count"])
                origin_mse = float(origin["mse"])
                origin_mae = float(origin["mae"])
                origin_stride = payload.get("origin_stride")
                if (isinstance(origin_stride, bool) or not isinstance(origin_stride, int)
                        or origin_stride <= 0):
                    raise ValueError("origin_stride must be a positive integer")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SystemExit(f"invalid streaming origin artifact for {config_path}: {exc}") from exc
            if (n_origins <= 0 or not np.isfinite(origin_mse)
                    or not np.isfinite(origin_mae)):
                raise SystemExit(f"invalid streaming origin values for {config_path}")
            channels = int(record.get("c_out", 1))
            rows.append({"dataset": record["dataset"], "model": record["model"],
                         "features": record["features"],
                         "enc_in": int(record.get("enc_in", channels)),
                         "dec_in": int(record.get("dec_in", channels)),
                         "c_out": channels, "seq_len": int(record["seq_len"]),
                         "pred_len": int(record["pred_len"]), "seed": int(record["seed"]),
                         "origin_stride": int(origin_stride),
                         "n_origins": n_origins, "origin_mse_mean": origin_mse,
                         "origin_mae_mean": origin_mae})
            continue
        pred_path, true_path = result_dir / "pred.npy", result_dir / "true.npy"
        if not pred_path.is_file() or not true_path.is_file():
            raise SystemExit(f"missing prediction artifact for {config_path}")
        try:
            pred = np.asarray(np.load(pred_path), dtype=float)
            true = np.asarray(np.load(true_path), dtype=float)
        except (OSError, ValueError, EOFError, ImportError) as exc:
            raise SystemExit(f"invalid prediction artifact for {config_path}: {exc}") from exc
        horizon = int(record["pred_len"])
        channels = int(record.get("c_out", 1))
        if (pred.ndim != 3 or true.shape != pred.shape or horizon <= 0
                or tuple(pred.shape[1:]) != (horizon, channels)
                or pred.shape[0] == 0 or not np.all(np.isfinite(pred))
                or not np.all(np.isfinite(true))):
            raise SystemExit(f"invalid prediction shape/values for {config_path}")
        # Test windows are chronological one-step origins.  Taking every
        # horizon-th origin removes the dominant overlap for a sensitivity
        # analysis while preserving the repository's global metric as primary.
        selected_pred = pred[::horizon]
        selected_true = true[::horizon]
        errors = selected_pred - selected_true
        rows.append({"dataset": record["dataset"], "model": record["model"],
                     "features": record["features"],
                     "enc_in": int(record.get("enc_in", channels)),
                     "dec_in": int(record.get("dec_in", channels)),
                     "c_out": channels, "seq_len": int(record["seq_len"]),
                     "pred_len": horizon, "seed": int(record["seed"]),
                     "origin_stride": horizon,
                     "n_origins": int(errors.shape[0]),
                     "origin_mse_mean": float(np.mean(errors ** 2)),
                     "origin_mae_mean": float(np.mean(np.abs(errors)))})
    return pd.DataFrame(rows)


def _origin_paired(origin: pd.DataFrame, selected: dict[str, float]) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "features", "enc_in", "dec_in", "c_out", "seq_len", "pred_len", "origin_stride"]
    for group_keys, group in origin.groupby(keys, dropna=False):
        base = dict(zip(keys, group_keys))
        for plain, contrastive in PLAIN_TO_AUTOCON.items():
            left = group[group["model"] == plain].set_index("seed")["origin_mse_mean"]
            right = group[group["model"] == contrastive].set_index("seed")["origin_mse_mean"]
            joined = pd.concat([left.rename("plain"), right.rename("autocon")], axis=1).dropna()
            if joined.empty:
                continue
            delta = joined["autocon"] - joined["plain"]
            rows.append({**base, "plain_model": plain, "autocon_model": contrastive,
                         "autocon_lambda": selected[contrastive], "n_pairs": len(joined),
                         "mean_origin_delta_mse": float(delta.mean()),
                         "relative_change_pct": float(100.0 * delta.mean() / joined["plain"].mean()),
                         "seed_deltas": ";".join(
                             f"{int(seed)}:{value:.8g}" for seed, value in delta.items())})
    return pd.DataFrame(rows)


def _resources(manifest_path: Path) -> pd.DataFrame:
    records = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "job_id" in record:
                records[record["job_id"]] = record
    rows = []
    for record in records.values():
        if record.get("status") != "ok":
            continue
        params = np.nan
        log_path = Path(record.get("log", ""))
        if log_path.is_file():
            match = re.search(r"model parameters:(\d+)", log_path.read_text(encoding="utf-8"))
            if match:
                params = int(match.group(1))
        rows.append({"dataset": record["dataset"], "model": record["model"],
                     "features": record.get("features"),
                     "enc_in": int(record.get("enc_in", 1)),
                     "dec_in": int(record.get("dec_in", 1)),
                     "c_out": int(record.get("c_out", 1)),
                     "pred_len": int(record["pred_len"]), "elapsed_sec": record.get("elapsed_sec", np.nan),
                     "parameters": params})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return (frame.groupby(["model", "dataset", "features", "enc_in", "dec_in", "c_out", "pred_len"], dropna=False)
            .agg(runs=("elapsed_sec", "size"), elapsed_sec_mean=("elapsed_sec", "mean"),
                 elapsed_sec_std=("elapsed_sec", "std"), parameters=("parameters", "median"))
            .reset_index().sort_values(["dataset", "pred_len", "model"]))


def _markdown(summary: pd.DataFrame, paired: pd.DataFrame, origin_paired: pd.DataFrame,
              resources: pd.DataFrame, selected: dict[str, float], phase: dict) -> str:
    phase_name = phase["name"]
    title = "Confirmatory Core Report" if phase_name == "confirmatory_core" else f"{phase_name.replace('_', ' ').title()} Report"
    lines = [f"# {title}", "",
             "The primary rows use lambdas selected from the validation-only phase; no test metric was used for selection.",
             "The inferential replication unit is the random seed. Sliding windows are not treated as independent samples.", "",
             "## Frozen Lambdas", "", "```json", json.dumps(selected, indent=2), "```", ""]
    if summary.empty:
        lines.append("No completed primary metrics are available yet.")
    else:
        table = summary[["dataset", "model", "features", "enc_in", "c_out", "pred_len",
                         "n_runs", "mse_mean", "mse_ci95", "mae_mean", "mae_ci95",
                         "original_mse_mean", "original_mae_mean"]].copy()
        for column in ("mse_mean", "mse_ci95", "mae_mean", "mae_ci95",
                       "original_mse_mean", "original_mae_mean"):
            table[column] = table[column].map(lambda value: f"{value:.6f}")
        lines.extend(["## Primary Metrics", "", _markdown_table(table), ""])
    if not paired.empty:
        lines.extend(["## Seed-Paired Deltas", "", _markdown_table(paired), ""])
    if not origin_paired.empty:
        lines.extend(["## Non-overlapping Origin Sensitivity", "",
                      "Each seed uses the registered origin_stride; this is a sensitivity analysis, not an additional independent-sample test.",
                      "", _markdown_table(origin_paired), ""])
    if not resources.empty:
        lines.extend(["## Resource Summary", "", _markdown_table(resources), ""])
    caveats = ["Metrics are in the repository's train-standardized space; original-scale columns are descriptive.",
                  "- Primary test MSE averages all chronological windows; origin sensitivity uses every horizon-th origin.",
                  "- Parameter counts and environment details are retained in the phase manifest and environment snapshot."]
    datasets = set(phase.get("datasets", []))
    if "national_illness" in datasets:
        caveats.insert(2, "National Illness remains exploratory because its records are weekly.")
    if "weather" in datasets:
        caveats.insert(2, "Weather has duplicate timestamps and irregular gaps; this transfer phase is secondary, not a primary benchmark claim.")
    if phase_name == "confirmatory_core":
        caveats.insert(2, "Weather and National Illness are outside this core report.")
    lines.extend(["## Caveats", ""] + [f"- {item.lstrip('- ')}" for item in caveats])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_results", type=Path, required=True,
                        help="raw_results.csv produced by collect_results.py")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/final_training_plan.json"))
    parser.add_argument("--phase", default="confirmatory_core")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True,
                        help="valid=true JSON produced by validate_training_phase.py")
    parser.add_argument("--docs_output", type=Path, default=Path("docs/reports/confirmatory-core-fresh.md"))
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else Path(__file__).resolve().parents[1] / args.plan
    plan, phase = _load_plan(plan_path, args.phase)
    selection_path = args.selection if args.selection.is_absolute() else Path(__file__).resolve().parents[1] / args.selection
    selection_path = selection_path.resolve()
    selection_payload = _load_selection_payload(selection_path)
    if phase.get("autocon_lambda") != "selected":
        raise SystemExit(f"phase {args.phase} is not a selected-lambda confirmatory phase")
    expected_selection_phase = phase.get("selection_phase", "selection")
    if selection_payload.get("phase") != expected_selection_phase:
        raise SystemExit(
            f"selection phase mismatch: expected {expected_selection_phase!r}, "
            f"received {selection_payload.get('phase')!r}"
        )
    expected_features = phase.get("features", plan.get("global", {}).get("features", "S"))
    if selection_payload.get("features", expected_features) != expected_features:
        raise SystemExit("selection features do not match the report phase")
    raw = pd.read_csv(args.raw_results)
    selected = _load_selection(selection_path)
    validation_path = args.validation if args.validation.is_absolute() else Path(__file__).resolve().parents[1] / args.validation
    with validation_path.open(encoding="utf-8") as validation_handle:
        validation = json.load(validation_handle)
    if not validation.get("valid") or validation.get("phase") != args.phase:
        raise SystemExit("phase validation is missing or invalid; refusing confirmatory report")
    expected_keys = _expected_keys(plan, phase, selected)
    expected_autocon = {model for model in phase["models"] if model in AUTOCON_MODELS}
    if set(selected) != expected_autocon:
        raise SystemExit(f"selection keys do not match registered AutoCon models: {sorted(selected)} vs {sorted(expected_autocon)}")
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"results_dir does not exist: {results_dir}")
    manifest_path = args.manifest if args.manifest.is_absolute() else Path(__file__).resolve().parents[1] / args.manifest
    required_validation_fields = {
        "manifest", "manifest_sha256", "plan_sha256", "provenance",
        "provenance_id", "expected_jobs", "manifest_unique_jobs",
    }
    missing_validation_fields = sorted(field for field in required_validation_fields if field not in validation)
    if missing_validation_fields:
        raise SystemExit(
            "validation predates strict provenance schema; refusing report: "
            + ", ".join(missing_validation_fields)
        )
    if Path(validation["manifest"]).resolve() != manifest_path.resolve():
        raise SystemExit("phase validation refers to a different manifest")
    if validation["manifest_sha256"] != _sha256(manifest_path):
        raise SystemExit("manifest changed after phase validation; refusing report")
    if validation["plan_sha256"] != _sha256(plan_path):
        raise SystemExit("training plan changed after phase validation; refusing report")
    provenance_path = Path(validation["provenance"])
    if not provenance_path.is_file():
        raise SystemExit("validated provenance file is missing; refusing report")
    phase_root = manifest_path.parent.parent.resolve()
    provenance_path = provenance_path.resolve()
    if provenance_path.parent != phase_root:
        raise SystemExit(
            "validated provenance is outside the manifest phase root; refusing report"
        )
    if provenance_path != (phase_root / "provenance.json").resolve():
        raise SystemExit(
            "validated provenance is not the phase-root provenance.json; refusing report"
        )
    with provenance_path.open(encoding="utf-8") as provenance_handle:
        provenance = json.load(provenance_handle)
    if provenance.get("provenance_id") != validation["provenance_id"]:
        raise SystemExit("validation provenance id does not match provenance sidecar")
    # Confirmation reports must be tied to the approved launch decision that
    # was archived in the phase provenance.  Keep historical/exploratory plans
    # unchanged: the shared helper is a no-op unless confirmation_contract is
    # declared by the plan.
    if isinstance(plan.get("confirmation_contract"), dict):
        try:
            try:
                from scripts.run_training_plan import _validate_confirmation_launch_binding
            except ModuleNotFoundError:
                from run_training_plan import _validate_confirmation_launch_binding

            _validate_confirmation_launch_binding(
                plan, plan_path, provenance, provenance_path.parent
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"confirmation launch binding is missing or invalid; refusing report: {exc}"
            ) from exc
    snapshot_root = Path(provenance.get("source_snapshot", ""))
    for relative in (
        "scripts/run_training_plan.py",
        "scripts/validate_training_phase.py",
        "scripts/build_final_report.py",
        "scripts/validate_highdim_contract.py",
    ):
        snapshot_file = snapshot_root / relative
        live_file = Path(__file__).resolve().parents[1] / relative
        if snapshot_file.is_file() and _sha256(snapshot_file) != _sha256(live_file):
            raise SystemExit(f"live source differs from immutable provenance snapshot: {relative}")
    if provenance.get("selection_sha256") != _sha256(selection_path):
        raise SystemExit("selection file changed after phase validation; refusing report")
    if validation.get("expected_jobs") != len(expected_keys) or validation.get("manifest_unique_jobs") != len(expected_keys):
        raise SystemExit("phase validation coverage does not match the registered phase")
    manifest_paths = _validated_manifest_paths(manifest_path, expected_keys)
    validated_artifacts = validation.get("artifact_hashes_by_job", {})
    if not validated_artifacts:
        raise SystemExit("phase validation has no artifact hash snapshot; refusing report")
    manifest_rows, malformed = _latest_manifest_records(manifest_path)
    if malformed:
        raise SystemExit("manifest contains malformed JSONL rows; refusing report")
    manifest_rows = list(manifest_rows.values())
    manifest_job_ids = {row.get("job_id") for row in manifest_rows}
    if set(validated_artifacts) != manifest_job_ids:
        raise SystemExit("phase validation artifact hash coverage differs from manifest")
    for row in manifest_rows:
        hashes = validated_artifacts.get(row.get("job_id"))
        if hashes is None:
            raise SystemExit("phase validation artifact hash missing for " + str(row.get("job_id")))
        artifact = Path(row.get("expected_artifact", "")).resolve()
        checkpoint_root = next((Path(row["command"][i + 1]) for i, token in enumerate(row.get("command", [])) if token == "--checkpoints" and i + 1 < len(row["command"])), None)
        actual_hashes = _artifact_hashes(artifact.parent, checkpoint_root / artifact.parent.name if checkpoint_root else None)
        if actual_hashes != hashes:
            raise SystemExit("artifact changed after phase validation for " + str(row.get("job_id")))
    trained = raw[raw["source"] == "trained"].copy()
    path_errors = []
    for path_value in trained["config_path"].dropna().tolist():
        path = Path(path_value).resolve()
        if results_dir not in path.parents:
            path_errors.append(str(path))
    if path_errors:
        raise SystemExit("raw_results contains paths outside results_dir; refusing report: " + path_errors[0])
    trained["lambda_key"] = trained.apply(
        lambda row: 0.0 if not bool(row["autocon"]) else float(row["autocon_lambda"]), axis=1)
    trained["key"] = list(zip(trained["dataset"], trained["model"], trained["features"],
                               trained["enc_in"].astype(int), trained["dec_in"].astype(int),
                               trained["c_out"].astype(int), trained["pred_len"].astype(int),
                               trained["seed"].astype(int), trained["lambda_key"]))
    if len(trained) != len(set(trained["key"])):
        raise SystemExit("duplicate trained result coordinates found; refusing report")
    actual_keys = set(trained["key"])
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise SystemExit(f"result coverage mismatch: missing={missing[:3]} extra={extra[:3]}")
    raw_paths = {key: Path(path).resolve() for key, path in zip(trained["key"], trained["config_path"])}
    expected_paths = {key: path.resolve() for key, path in manifest_paths.items()}
    if raw_paths != expected_paths:
        raise SystemExit("raw_results paths do not exactly match the validated manifest")
    trained = _refresh_metrics_from_artifacts(trained)
    primary = _primary(trained, selected)
    if len(primary) != len(expected_keys):
        raise SystemExit("primary filtering changed registered result count")
    # Summary is recomputed after filtering, so exploratory lambdas cannot alter
    # means, ranks, or confidence intervals in the confirmatory report.
    groups = []
    for keys, group in primary.groupby(["dataset", "model", "features", "enc_in", "dec_in", "c_out", "seq_len", "pred_len"], dropna=False):
        row = dict(zip(["dataset", "model", "features", "enc_in", "dec_in", "c_out", "seq_len", "pred_len"], keys))
        row["autocon_lambda"] = selected.get(row["model"], 0.0)
        row["n_runs"] = len(group)
        for metric in ("mse", "mae", "rmse", "original_mse", "original_mae"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if len(values) > 1:
                from scipy import stats
                row[f"{metric}_ci95"] = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
            else:
                row[f"{metric}_ci95"] = 0.0
        groups.append(row)
    summary = pd.DataFrame(groups).sort_values(["dataset", "pred_len", "mse_mean"]) if groups else pd.DataFrame()
    paired = _paired(trained, selected)
    origin = _origin_metrics(results_dir, primary)
    if len(origin) != len(primary):
        raise SystemExit("origin sensitivity coverage does not match primary results")
    origin_paired = _origin_paired(origin, selected)
    resources = _resources(manifest_path)
    channel_detail, channel_summary = _channel_metrics(primary)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    paired.to_csv(output_dir / "paired_seed_deltas.csv", index=False)
    origin.to_csv(output_dir / "origin_robustness.csv", index=False)
    origin_paired.to_csv(output_dir / "origin_paired_deltas.csv", index=False)
    resources.to_csv(output_dir / "resource_summary.csv", index=False)
    channel_detail.to_csv(output_dir / "per_channel_metrics.csv", index=False)
    channel_summary.to_csv(output_dir / "per_channel_summary.csv", index=False)
    with (output_dir / "selected_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    markdown = _markdown(summary, paired, origin_paired, resources, selected, phase)
    (output_dir / f"{args.phase}.md").write_text(markdown, encoding="utf-8")
    docs_output = args.docs_output if args.docs_output.is_absolute() else Path(__file__).resolve().parents[1] / args.docs_output
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(primary)} primary rows, {len(paired)} paired cells, {len(origin)} origin rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
