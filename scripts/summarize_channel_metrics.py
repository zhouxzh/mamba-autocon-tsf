#!/usr/bin/env python
"""Summarize channel macro and overall micro errors by random seed.

This is a descriptive post-processing utility.  It consumes the validated
phase report ``per_channel_metrics.csv`` and/or compact ``stream_metrics.json``
artifacts without changing the phase report or any frozen result.  A row in
the seed table is one complete model run.  Channels and forecast windows are
dimensions of that run, not independent inferential replicates.

The preferred overall (micro) metric source is ``stream_metrics.json``.  A
phase ``raw_results.csv`` can provide the same exact global metrics when full
predictions were retained.  If neither is available, additive metrics are
derived from the equally weighted channel rows and explicitly labelled as a
``channel_equal_weight_proxy``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_identity import canonical_model


# ``channel_metrics.npy`` and the report use this stable order.  MAPE/MSPE are
# intentionally not included here: the streaming implementation only stores
# channel-wise MAE/MSE/RMSE and near-zero targets make percentage metrics
# unstable.  They remain available in the overall micro table when present.
CHANNEL_METRICS: tuple[str, ...] = (
    "mae", "mse", "rmse", "original_mae", "original_mse", "original_rmse",
)
MICRO_METRICS: tuple[str, ...] = (
    "mae", "mse", "rmse", "mape", "mspe",
    "original_mae", "original_mse", "original_rmse", "original_mape", "original_mspe",
)

META_COLUMNS: tuple[str, ...] = (
    "dataset", "model", "features", "seq_len", "pred_len", "seed",
    "enc_in", "dec_in", "c_out",
)
# All of these dimensions are part of a run identity.  In particular, do not
# merge two channels/seeds merely because they share dataset/model/horizon.
RUN_COLUMNS: tuple[str, ...] = META_COLUMNS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _int_value(value: object, name: str, *, default: int | None = None) -> int:
    if _is_missing(value) or value == "":
        if default is not None:
            return int(default)
        raise ValueError(f"missing integer field {name}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer field {name}: {value!r}") from exc
    if not np.isfinite(number) or int(number) != number:
        raise ValueError(f"invalid integer field {name}: {value!r}")
    return int(number)


def _model_name(mapping: Mapping[str, object]) -> str:
    value = mapping.get("model")
    if _is_missing(value):
        raise ValueError("missing model field")
    return canonical_model(mapping)


def _metadata(mapping: Mapping[str, object], *, c_out_default: int | None = None) -> dict[str, object]:
    """Normalize report/config metadata to a hashable run-coordinate schema."""
    dataset = mapping.get("dataset", mapping.get("data"))
    if _is_missing(dataset):
        raise ValueError("missing dataset/data field")
    features = mapping.get("features", "")
    if _is_missing(features):
        features = ""
    pred = mapping.get("pred_len", mapping.get("horizon"))
    pred_len = _int_value(pred, "pred_len")
    if pred_len <= 0:
        raise ValueError("pred_len must be positive")
    seed = mapping.get("seed", mapping.get("run_seed"))
    seed_value = _int_value(seed, "seed")
    c_out_raw = mapping.get("c_out", c_out_default)
    c_out = _int_value(c_out_raw, "c_out")
    if c_out <= 0:
        raise ValueError("c_out must be positive")
    enc_in = _int_value(mapping.get("enc_in", c_out), "enc_in", default=c_out)
    dec_in = _int_value(mapping.get("dec_in", c_out), "dec_in", default=c_out)
    seq_raw = mapping.get("seq_len", 0)
    seq_len = _int_value(seq_raw, "seq_len", default=0)
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    return {
        "dataset": str(dataset),
        "model": _model_name(mapping),
        "features": str(features),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "horizon": pred_len,
        "seed": seed_value,
        "enc_in": enc_in,
        "dec_in": dec_in,
        "c_out": c_out,
    }


def _run_key(mapping: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(mapping[column] for column in RUN_COLUMNS)


def _finite_metric(value: object, name: str, *, allow_missing: bool = True) -> float:
    if _is_missing(value) or value == "":
        if allow_missing:
            return float("nan")
        raise ValueError(f"missing metric {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid metric {name}: {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"non-finite metric {name}")
    return result


def _metric_column(frame: pd.DataFrame, name: str) -> str | None:
    """Find a metric column, accepting the optional ``channel_`` prefix."""
    for candidate in (name, f"channel_{name}"):
        if candidate in frame.columns:
            return candidate
    return None


def _validate_channel_rows(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"channel input is empty: {source}")
    required = {"dataset", "model", "seed", "channel"}
    if "pred_len" not in frame.columns and "horizon" not in frame.columns:
        required.add("pred_len")
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(f"channel input missing columns {missing}: {source}")
    metric_columns = {name: _metric_column(frame, name) for name in CHANNEL_METRICS}
    if all(column is None for column in metric_columns.values()):
        raise ValueError(f"channel input has no recognized metrics: {source}")

    rows: list[dict[str, object]] = []
    # Infer c_out per run before applying enc/dec defaults.  Applying defaults
    # first would make channel 0 look like a one-channel run and channel 1
    # like a two-channel run, which would split one real run into two groups.
    provisional: list[tuple[Mapping[str, object], int, tuple[object, ...], int | None, dict[str, float]]] = []
    for _, raw in frame.iterrows():
        channel = _int_value(raw.get("channel"), "channel")
        if channel < 0:
            raise ValueError(f"channel must be non-negative: {source}")
        raw_c_out = raw.get("c_out") if "c_out" in frame.columns else None
        explicit_c_out = None if _is_missing(raw_c_out) or raw_c_out == "" else _int_value(raw_c_out, "c_out")
        # Build a run identity from fields whose defaults do not depend on
        # c_out.  Explicit enc/dec values remain part of that identity.
        dataset = raw.get("dataset", raw.get("data"))
        if _is_missing(dataset):
            raise ValueError(f"missing dataset/data field: {source}")
        features = raw.get("features", "")
        if _is_missing(features):
            features = ""
        pred_len = _int_value(raw.get("pred_len", raw.get("horizon")), "pred_len")
        seed = _int_value(raw.get("seed", raw.get("run_seed")), "seed")
        seq_len = _int_value(raw.get("seq_len", 0), "seq_len", default=0)
        enc_marker = (None if "enc_in" not in frame.columns or _is_missing(raw.get("enc_in"))
                      else _int_value(raw.get("enc_in"), "enc_in"))
        dec_marker = (None if "dec_in" not in frame.columns or _is_missing(raw.get("dec_in"))
                      else _int_value(raw.get("dec_in"), "dec_in"))
        base_key = (str(dataset), _model_name(raw), str(features), seq_len, pred_len,
                    seed, enc_marker, dec_marker)
        # Scaled MAE/MSE/RMSE are the required channel metrics.  Original
        # scale columns are optional for legacy reports, but a present NaN in
        # a core metric is an invalid row rather than an ignorable missing
        # value.
        for name in ("mae", "mse", "rmse"):
            if metric_columns[name] is None:
                raise ValueError(f"channel input missing required metric {name}: {source}")
        values = {}
        for name, column in metric_columns.items():
            values[name] = _finite_metric(
                raw.get(column) if column else None,
                name,
                allow_missing=name not in {"mae", "mse", "rmse"},
            )
        provisional.append((raw, channel, base_key, explicit_c_out, values))

    # Fill omitted c_out from the maximum observed channel for each coordinate.
    maxima: dict[tuple[object, ...], int] = {}
    explicit_by_key: dict[tuple[object, ...], set[int]] = {}
    for _, channel, base_key, explicit_c_out, _ in provisional:
        maxima[base_key] = max(maxima.get(base_key, 0), channel + 1)
        if explicit_c_out is not None:
            explicit_by_key.setdefault(base_key, set()).add(explicit_c_out)
    for raw, channel, base_key, explicit_c_out, values in provisional:
        explicit_values = explicit_by_key.get(base_key, set())
        if len(explicit_values) > 1:
            raise ValueError(f"inconsistent c_out values for run {base_key!r}: {sorted(explicit_values)}")
        inferred = next(iter(explicit_values), maxima[base_key])
        if inferred < maxima[base_key]:
            raise ValueError(
                f"c_out={inferred} is smaller than observed channel range for run {base_key!r}"
            )
        metadata = _metadata(raw, c_out_default=inferred)
        rows.append({**metadata, "channel": channel, **values})

    result = pd.DataFrame(rows)
    # Reconcile enc/dec defaults and check coordinate uniqueness/completeness.
    for column in META_COLUMNS + ("channel",):
        if column not in result.columns:
            raise ValueError(f"normalized channel input lacks {column}: {source}")
    duplicate_mask = result.duplicated(list(RUN_COLUMNS) + ["channel"], keep=False)
    if bool(duplicate_mask.any()):
        duplicate = result.loc[duplicate_mask, list(RUN_COLUMNS) + ["channel"]].iloc[0].to_dict()
        raise ValueError(f"duplicate channel coordinate in {source}: {duplicate}")
    for key, group in result.groupby(list(RUN_COLUMNS), dropna=False, sort=False):
        channels = sorted(int(value) for value in group["channel"])
        c_out = int(group["c_out"].iloc[0])
        if c_out != len(set(channels)) or channels != list(range(c_out)):
            raise ValueError(
                f"channels for run {key!r} must be exactly 0..{c_out - 1}; got {channels}"
            )
    return result.sort_values(list(RUN_COLUMNS) + ["channel"]).reset_index(drop=True)


def load_per_channel_csv(path: Path | str) -> pd.DataFrame:
    """Load and validate a report ``per_channel_metrics.csv``."""
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"channel report does not exist: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"unable to read channel report {source}: {exc}") from exc
    return _validate_channel_rows(frame, source)


def _expand_paths(paths: Iterable[Path | str], *, filename: str) -> list[Path]:
    found: list[Path] = []
    for value in paths:
        path = Path(value).resolve()
        if path.is_file():
            if path.name == filename:
                found.append(path)
        elif path.is_dir():
            if (path / filename).is_file():
                found.append((path / filename).resolve())
            found.extend(sorted(path.rglob(filename)))
        else:
            raise ValueError(f"input path does not exist: {path}")
    unique = sorted(set(found))
    if not unique:
        raise ValueError(f"no {filename} files found in requested inputs")
    return unique


def _stream_record(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read stream artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"stream payload must be an object: {path}")
    if payload.get("schema_version") != 1 or payload.get("aggregation") != "streaming_window_weighted":
        raise ValueError(f"unsupported stream schema/aggregation: {path}")
    config_path = path.parent / "config.json"
    if not config_path.is_file():
        raise ValueError(f"stream artifact has no sibling config.json: {path}")
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read stream config {config_path}: {exc}") from exc
    channels_raw = payload.get("channel_metrics")
    try:
        channels = np.asarray(channels_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid channel_metrics array: {path}") from exc
    output_channels = _int_value(payload.get("output_channels"), "output_channels")
    if channels.shape != (output_channels, len(CHANNEL_METRICS)):
        raise ValueError(
            f"channel_metrics shape {channels.shape} != {(output_channels, len(CHANNEL_METRICS))}: {path}"
        )
    if not np.all(np.isfinite(channels)):
        raise ValueError(f"channel_metrics contains non-finite values: {path}")
    order = payload.get("channel_metric_order", list(CHANNEL_METRICS))
    if tuple(order) != CHANNEL_METRICS:
        raise ValueError(f"unexpected channel_metric_order in {path}: {order!r}")
    metadata = _metadata(config, c_out_default=output_channels)
    if metadata["c_out"] != output_channels:
        raise ValueError(f"stream/config c_out mismatch: {path}")
    rows = []
    for channel, values in enumerate(channels):
        rows.append({**metadata, "channel": channel,
                     **{name: float(value) for name, value in zip(CHANNEL_METRICS, values)}})

    micro: dict[str, object] = {**metadata,
                                "micro_source": "stream_metrics",
                                "stream_path": str(path),
                                "stream_sha256": _sha256(path),
                                "origin_stride": payload.get("origin_stride"),
                                "window_count": payload.get("window_count"),
                                "origin_count": payload.get("origin_count")}
    for section, prefix in (("scaled", ""), ("original", "original_")):
        values = payload.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"stream {section} section is not an object: {path}")
        for metric in ("mae", "mse", "rmse", "mape", "mspe"):
            key = f"{prefix}{metric}"
            micro[f"overall_micro_{key}"] = _finite_metric(values.get(metric), key)
    return rows, micro


def load_stream_artifacts(paths: Iterable[Path | str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load stream JSON files and return channel rows plus exact micro rows."""
    stream_paths = _expand_paths(paths, filename="stream_metrics.json")
    channel_rows: list[dict[str, object]] = []
    micro_rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for path in stream_paths:
        rows, micro = _stream_record(path)
        key = _run_key(micro)
        if key in seen:
            raise ValueError(f"duplicate stream run coordinate: {key!r}")
        seen.add(key)
        channel_rows.extend(rows)
        micro_rows.append(micro)
    channels = _validate_channel_rows(pd.DataFrame(channel_rows), Path("stream_metrics.json"))
    return channels, pd.DataFrame(micro_rows)


def _load_overall_csv(path: Path, channel_runs: set[tuple[object, ...]]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"overall metric CSV does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"unable to read overall metric CSV {path}: {exc}") from exc
    required = {"dataset", "model", "seed"}
    if "pred_len" not in frame.columns and "horizon" not in frame.columns:
        required.add("pred_len")
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(f"overall CSV missing columns {missing}: {path}")
    rows: list[dict[str, object]] = []
    file_sha = _sha256(path)
    for _, raw in frame.iterrows():
        # Naive rows or unrelated phases are not part of the channel report.
        if "source" in frame.columns and str(raw.get("source")) != "trained":
            continue
        try:
            metadata = _metadata(raw)
        except ValueError:
            # A malformed unrelated row should not hide a valid match, but a
            # row marked ``trained`` is expected to be complete and is an
            # error.  This distinction keeps naive/legacy side rows usable.
            if "source" in frame.columns and str(raw.get("source")) == "trained":
                raise
            continue
        if _run_key(metadata) not in channel_runs:
            continue
        row = {**metadata, "micro_source": "overall_csv",
               "overall_path": str(path), "overall_sha256": file_sha}
        for metric in MICRO_METRICS:
            row[f"overall_micro_{metric}"] = _finite_metric(raw.get(metric), metric)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    duplicates = result.duplicated(list(RUN_COLUMNS), keep=False)
    if bool(duplicates.any()):
        duplicate = result.loc[duplicates, list(RUN_COLUMNS)].iloc[0].to_dict()
        raise ValueError(
            f"overall CSV has multiple rows for channel run {duplicate}; filter it or use stream artifacts"
        )
    return result


def _derive_micro(group: pd.DataFrame) -> dict[str, float]:
    """Derive a transparent equal-channel proxy when no global artifact exists."""
    result = {f"overall_micro_{metric}": float("nan") for metric in MICRO_METRICS}
    # Every channel has the same number of forecast elements in this loader;
    # means of MAE/MSE are therefore equal-weight global aggregates.  RMSE is
    # recomputed from global MSE, never averaged from channel RMSEs.
    for metric in ("mae", "mse", "original_mae", "original_mse"):
        values = group[metric].to_numpy(dtype=float)
        if np.all(np.isfinite(values)):
            result[f"overall_micro_{metric}"] = float(np.mean(values))
    for prefix in ("", "original_"):
        mse = result[f"overall_micro_{prefix}mse"]
        if np.isfinite(mse) and mse >= 0:
            result[f"overall_micro_{prefix}rmse"] = float(math.sqrt(mse))
    return result


def _merge_micro(seed_row: dict[str, object], source: Mapping[str, object] | None,
                 group: pd.DataFrame) -> None:
    if source is None:
        seed_row.update(_derive_micro(group))
        seed_row["micro_source"] = "channel_equal_weight_proxy"
        seed_row["micro_note"] = "derived from equal-channel additive metrics"
        return
    seed_row["micro_source"] = str(source.get("micro_source", "unknown"))
    if source.get("stream_path"):
        seed_row["stream_path"] = source.get("stream_path")
        seed_row["stream_sha256"] = source.get("stream_sha256")
        for name in ("origin_stride", "window_count", "origin_count"):
            seed_row[name] = source.get(name)
    if source.get("overall_path"):
        seed_row["overall_path"] = source.get("overall_path")
        seed_row["overall_sha256"] = source.get("overall_sha256")
    for metric in MICRO_METRICS:
        key = f"overall_micro_{metric}"
        seed_row[key] = _finite_metric(source.get(key), key)


def make_seed_summary(channel_rows: pd.DataFrame, micro_rows: pd.DataFrame | None = None,
                      overall_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute one row per dataset/model/config/seed.

    ``channel_rows`` must contain one row for every channel of every run.  The
    optional micro tables use the same run coordinate and are never treated as
    extra observations.
    """
    validated = _validate_channel_rows(channel_rows.copy(), Path("channel_rows"))
    micro_map: dict[tuple[object, ...], Mapping[str, object]] = {}
    for source_frame in (micro_rows, overall_rows):
        if source_frame is None or source_frame.empty:
            continue
        for _, raw in source_frame.iterrows():
            key = _run_key(raw)
            # Stream is preferred over raw CSV.  Conflicting exact sources are
            # rejected rather than silently selecting one.
            previous = micro_map.get(key)
            if previous is None:
                micro_map[key] = raw.to_dict()
            elif str(previous.get("micro_source")) == "stream_metrics":
                for metric in MICRO_METRICS:
                    name = f"overall_micro_{metric}"
                    left, right = previous.get(name), raw.get(name)
                    if np.isfinite(float(left)) and np.isfinite(float(right)) and not np.isclose(float(left), float(right), rtol=1e-5, atol=1e-8):
                        raise ValueError(f"stream and overall metrics disagree for {key!r}: {name}")
            else:
                micro_map[key] = raw.to_dict()

    rows: list[dict[str, object]] = []
    for key, group in validated.groupby(list(RUN_COLUMNS), dropna=False, sort=False):
        metadata = dict(zip(RUN_COLUMNS, key))
        row: dict[str, object] = {**metadata, "horizon": int(metadata["pred_len"]),
                                  "n_channels": int(len(group)),
                                  "channel_source": str(group.get("source", pd.Series(["unknown"])).iloc[0])
                                  if "source" in group.columns else "report_or_stream"}
        for metric in CHANNEL_METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"channel_macro_{metric}"] = float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan")
        _merge_micro(row, micro_map.get(_run_key(metadata)), group)
        rows.append(row)
    if not rows:
        raise ValueError("no complete channel runs were found")
    result = pd.DataFrame(rows)
    result = result.sort_values(["dataset", "model", "pred_len", "seed"]).reset_index(drop=True)
    return result


def _summary_stats(values: Sequence[object], alpha: float) -> tuple[int, float, float, float, float]:
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    n = int(numeric.size)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(numeric))
    std = float(np.std(numeric, ddof=1)) if n > 1 else 0.0
    if n < 2:
        return n, mean, std, float("nan"), float("nan")
    half = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1) * std / math.sqrt(n))
    return n, mean, std, mean - half, mean + half


def make_group_summary(seed_rows: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Aggregate seed rows by dataset/model/horizon (and fixed config axes)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if seed_rows.empty:
        raise ValueError("seed summary is empty")
    group_columns = ["dataset", "model", "features", "seq_len", "pred_len", "c_out"]
    missing = [column for column in group_columns if column not in seed_rows.columns]
    if missing:
        raise ValueError(f"seed summary missing grouping columns: {missing}")
    rows: list[dict[str, object]] = []
    for key, group in seed_rows.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, key))
        row["horizon"] = int(row["pred_len"])
        seeds = sorted(int(value) for value in group["seed"])
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed in group {key!r}")
        row["n_seeds"] = len(seeds)
        row["seed_ids"] = ",".join(str(seed) for seed in seeds)
        row["n_channels_min"] = int(group["n_channels"].min())
        row["n_channels_max"] = int(group["n_channels"].max())
        sources = sorted(set(str(value) for value in group["micro_source"]))
        row["micro_sources"] = ";".join(sources)
        for prefix, metrics in (("channel_macro", CHANNEL_METRICS), ("overall_micro", MICRO_METRICS)):
            for metric in metrics:
                column = f"{prefix}_{metric}"
                n, mean, std, low, high = _summary_stats(group[column], alpha)
                row[f"{column}_n"] = n
                row[f"{column}_mean"] = mean
                row[f"{column}_std"] = std
                row[f"{column}_ci_low"] = low
                row[f"{column}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "model", "pred_len"]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        return frame.to_string(index=False)


def write_markdown(path: Path, seed_rows: pd.DataFrame, summary: pd.DataFrame,
                   *, inputs: Sequence[Path], validation_status: str, alpha: float) -> None:
    level = 100.0 * (1.0 - alpha)
    lines = [
        "# Channel Macro and Overall Micro Summary", "",
        "This is a descriptive post-processing table generated from validated phase artifacts.",
        "The inferential replication unit is the random seed; channels and sliding forecast windows are not independent replicates.",
        "",
        "## Definitions", "",
        "- `channel_macro_*` is the arithmetic mean across channels within one seed/run (equal channel weighting).",
        "- `overall_micro_*` is the all-element aggregate from `stream_metrics.json` when available, or the validated overall CSV.",
        "- `channel_equal_weight_proxy` is used only when no global artifact exists: additive MAE/MSE are averaged across channels and RMSE is recomputed as `sqrt(MSE)`. It must be labelled as a proxy in a manuscript.",
        f"- Means, sample standard deviations, and two-sided {level:g}% Student-t CIs are across seeds; a group with one seed has `NaN` CI bounds.",
        "- MAPE/MSPE are retained for diagnosis only and should not be the primary ranking metric, especially with near-zero targets.",
        "",
        "## Coverage", "",
        f"Validation status: `{validation_status}`.",
        f"Seed rows: **{len(seed_rows)}**; grouped rows: **{len(summary)}**.",
        f"Input files: {', '.join(f'`{path}`' for path in inputs)}.",
        "",
    ]
    if not summary.empty:
        columns = ["dataset", "model", "horizon", "n_seeds", "seed_ids", "n_channels_min", "n_channels_max", "micro_sources"]
        for prefix in ("channel_macro", "overall_micro"):
            for metric in ("mae", "mse", "rmse"):
                columns.extend([
                    f"{prefix}_{metric}_mean", f"{prefix}_{metric}_std",
                    f"{prefix}_{metric}_ci_low", f"{prefix}_{metric}_ci_high",
                ])
        table = summary[[column for column in columns if column in summary.columns]].copy()
        numeric = [column for column in table.columns if column.endswith(("_mean", "_std", "_ci_low", "_ci_high"))]
        for column in numeric:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
        lines.extend(["## Group Summary", "", _markdown_table(table), ""])
    lines.extend([
        "## Audit Notes", "",
        "The CSV `seed_channel_micro.csv` contains one row per complete seed/run; `summary_channel_micro.csv` contains the group-level estimates.",
        "Do not use the number of channels or forecast windows as the denominator of a confidence interval. They describe within-run aggregation only.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validation_status(phase_dir: Path | None, validation: Path | None, allow_unvalidated: bool) -> tuple[str, Path | None]:
    candidate = validation.resolve() if validation else None
    if candidate is None and phase_dir is not None:
        for name in ("phase_validation.json", "validation.json"):
            path = phase_dir / name
            if path.is_file():
                candidate = path.resolve()
                break
    if candidate is None:
        if not allow_unvalidated:
            raise ValueError("no phase validation JSON supplied; use --allow-unvalidated only for diagnostics")
        return "not_provided (diagnostic override)", None
    try:
        with candidate.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"unable to read validation JSON {candidate}: {exc}") from exc
    if payload.get("valid") is not True:
        if not allow_unvalidated:
            raise ValueError(f"phase validation is not valid: {candidate}")
        return "invalid (diagnostic override)", candidate
    return "valid=true", candidate


def _discover_phase_inputs(phase_dir: Path, channel_paths: list[Path], stream_paths: list[Path], overall_path: Path | None) -> tuple[list[Path], list[Path], Path | None]:
    if not phase_dir.is_dir():
        raise ValueError(f"phase directory does not exist: {phase_dir}")
    channels = list(channel_paths)
    report = phase_dir / "report" / "per_channel_metrics.csv"
    if report.is_file() and not channels:
        channels = [report]
    streams = list(stream_paths)
    if not streams:
        streams = sorted(phase_dir.rglob("stream_metrics.json"))
    overall = overall_path
    if overall is None:
        for candidate in (phase_dir / "raw_report" / "raw_results.csv", phase_dir / "raw_results.csv"):
            if candidate.is_file():
                overall = candidate
                break
    return channels, streams, overall


def _assert_channel_agreement(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Reject stale/mismatched channel CSV and stream rows at shared keys."""
    if left.empty or right.empty:
        return
    left_groups = {key: group for key, group in left.groupby(list(RUN_COLUMNS), dropna=False, sort=False)}
    right_groups = {key: group for key, group in right.groupby(list(RUN_COLUMNS), dropna=False, sort=False)}
    for key in sorted(set(left_groups) & set(right_groups), key=repr):
        a = left_groups[key].sort_values("channel")
        b = right_groups[key].sort_values("channel")
        if a["channel"].tolist() != b["channel"].tolist():
            raise ValueError(f"channel CSV/stream channel index mismatch for {key!r}")
        for metric in CHANNEL_METRICS:
            av = a[metric].to_numpy(dtype=float)
            bv = b[metric].to_numpy(dtype=float)
            finite = np.isfinite(av) & np.isfinite(bv)
            if finite.any() and not np.allclose(av[finite], bv[finite], rtol=1e-5, atol=1e-8):
                raise ValueError(f"channel CSV and stream metrics disagree for {key!r}: {metric}")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_dir = args.phase_dir.resolve() if args.phase_dir else None
    channel_paths, stream_paths, overall_path = _discover_phase_inputs(
        phase_dir, list(args.per_channel or []),
        args.stream or [], args.overall,
    ) if phase_dir else (list(args.per_channel or []), list(args.stream or []), args.overall)
    validation_status, validation_path = _validation_status(phase_dir, args.validation, args.allow_unvalidated)

    channel_rows: pd.DataFrame | None = None
    source_inputs: list[Path] = []
    if channel_paths:
        expanded = _expand_paths(channel_paths, filename="per_channel_metrics.csv")
        loaded = [load_per_channel_csv(path) for path in expanded]
        channel_rows = pd.concat(loaded, ignore_index=True)
        # Concatenating phases must still obey one coordinate/channel.
        channel_rows = _validate_channel_rows(channel_rows, Path("per_channel_metrics.csv"))
        source_inputs.extend(expanded)

    stream_channels = None
    stream_micro = None
    if stream_paths:
        stream_channels, stream_micro = load_stream_artifacts(stream_paths)
        source_inputs.extend(_expand_paths(stream_paths, filename="stream_metrics.json"))
        if channel_rows is None:
            channel_rows = stream_channels
        else:
            _assert_channel_agreement(channel_rows, stream_channels)
            # The report is the channel source when both exist.  Match stream
            # coordinates to it so stream remains an exact micro source; a
            # missing coordinate is harmless for mixed/legacy phases.
            report_keys = {_run_key(row) for _, row in channel_rows.drop_duplicates(list(RUN_COLUMNS)).iterrows()}
            stream_micro = stream_micro[stream_micro.apply(lambda row: _run_key(row) in report_keys, axis=1)].reset_index(drop=True)
    if channel_rows is None:
        raise ValueError("provide --phase-dir, --per-channel, or --stream")

    overall_rows = None
    if overall_path is not None:
        overall_path = overall_path.resolve()
        channel_keys = {_run_key(row) for _, row in channel_rows.drop_duplicates(list(RUN_COLUMNS)).iterrows()}
        overall_rows = _load_overall_csv(overall_path, channel_keys)
        source_inputs.append(overall_path)

    seed_rows = make_seed_summary(channel_rows, stream_micro, overall_rows)
    summary = make_group_summary(seed_rows, alpha=args.alpha)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "seed_channel_micro.csv"
    summary_path = output_dir / "summary_channel_micro.csv"
    markdown_path = output_dir / "summary_channel_micro.md"
    seed_rows.to_csv(seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(markdown_path, seed_rows, summary, inputs=source_inputs,
                   validation_status=validation_status, alpha=args.alpha)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "validation_status": validation_status,
        "validation_path": str(validation_path) if validation_path else None,
        "alpha": float(args.alpha),
        "seed_rows": int(len(seed_rows)),
        "group_rows": int(len(summary)),
        "seed_unit": "random_seed",
        "channel_unit": "descriptive_dimension_not_independent_replicate",
        "window_unit": "descriptive_aggregation_not_independent_replicate",
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in source_inputs],
        "outputs": {
            "seed_csv": str(seed_path), "summary_csv": str(summary_path),
            "markdown": str(markdown_path),
        },
    }
    (output_dir / "summary_channel_micro.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(seed_rows)} seed rows and {len(summary)} grouped rows to {output_dir}")
    return seed_rows, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path,
                        help="validated phase root; discovers report/per_channel_metrics.csv and stream artifacts")
    parser.add_argument("--per-channel", type=Path, action="append",
                        help="per_channel_metrics.csv (repeat for multiple explicitly selected files)")
    parser.add_argument("--stream", type=Path, action="append",
                        help="stream_metrics.json or result/phase directory (repeatable)")
    parser.add_argument("--overall", type=Path,
                        help="optional raw_results.csv with exact overall metrics")
    parser.add_argument("--validation", type=Path,
                        help="phase_validation.json; phase-dir auto-detects it")
    parser.add_argument("--allow-unvalidated", action="store_true",
                        help="allow diagnostic processing when no valid phase validation is present")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="two-sided CI alpha (default: 0.05)")
    args = parser.parse_args(argv)
    if not args.phase_dir and not args.per_channel and not args.stream:
        parser.error("one of --phase-dir, --per-channel, or --stream is required")
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
