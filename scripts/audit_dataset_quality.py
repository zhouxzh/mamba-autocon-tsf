#!/usr/bin/env python
"""Audit a forecasting CSV before it enters a paper phase.

The audit is deliberately independent of the training data loader.  It records
the exact input hash, schema, timestamp cadence, finite-value status, sentinel
counts, duplicate rows, constant columns, and per-channel scale/sparsity.  A
strict invocation exits non-zero when any registered quality condition fails;
the default invocation still writes a diagnostic snapshot for datasets that
need a documented cleaning step (for example Weather).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FREQ_TO_DELTA = {
    "h": pd.Timedelta(hours=1),
    "t": pd.Timedelta(minutes=15),
    "15min": pd.Timedelta(minutes=15),
    "d": pd.Timedelta(days=1),
    "w": pd.Timedelta(days=7),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    return value


def audit(
    path: Path,
    dataset: str,
    time_col: str,
    target: str,
    expected_channels: int | None,
    expected_freq: str | None,
    expected_sha256: str | None,
    sentinels: list[float],
    source_repository: str | None,
    source_revision: str | None,
    license_name: str | None,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sha256 = _sha256(path)
    frame = pd.read_csv(path)
    columns = [str(item) for item in frame.columns]
    schema_errors: list[str] = []
    if not columns or columns[0] != time_col:
        schema_errors.append(f"first_column={columns[0] if columns else None!r}, expected={time_col!r}")
    if target not in columns:
        schema_errors.append(f"target_missing={target!r}")
    value_columns = columns[1:] if columns and columns[0] == time_col else columns
    actual_channels = len(value_columns)
    if expected_channels is not None and actual_channels != expected_channels:
        schema_errors.append(f"channels={actual_channels}, expected={expected_channels}")

    timestamps = pd.to_datetime(frame[time_col], errors="coerce") if time_col in frame else pd.Series(dtype="datetime64[ns]")
    timestamp_errors: list[str] = []
    if timestamps.isna().any():
        timestamp_errors.append(f"invalid_timestamps={int(timestamps.isna().sum())}")
    duplicate_timestamps = int(timestamps.duplicated().sum())
    if duplicate_timestamps:
        timestamp_errors.append(f"duplicate_timestamps={duplicate_timestamps}")
    monotonic = bool(timestamps.is_monotonic_increasing) if len(timestamps) else False
    if not monotonic:
        timestamp_errors.append("timestamps_not_monotonic_increasing")
    deltas = timestamps.diff().dropna()
    delta_counts = {str(key): int(value) for key, value in deltas.value_counts().items()}
    expected_delta = FREQ_TO_DELTA.get(str(expected_freq).lower()) if expected_freq else None
    unexpected_intervals = int((deltas != expected_delta).sum()) if expected_delta is not None else 0
    if expected_delta is not None and unexpected_intervals:
        timestamp_errors.append(
            f"unexpected_intervals={unexpected_intervals}, expected={str(expected_delta)}"
        )

    numeric = frame[value_columns].apply(pd.to_numeric, errors="coerce") if value_columns else pd.DataFrame()
    matrix = numeric.to_numpy(dtype=np.float64, copy=False) if not numeric.empty else np.empty((len(frame), 0))
    nonnumeric = int(np.isnan(matrix).sum()) if matrix.size else 0
    nonfinite = int((~np.isfinite(matrix)).sum()) if matrix.size else 0
    sentinel_counts = {
        str(value): int(np.isclose(matrix, value, rtol=0.0, atol=0.0).sum())
        for value in sentinels
    }
    duplicate_rows = int(frame.duplicated().sum())
    constants = [value_columns[i] for i, std in enumerate(np.nanstd(matrix, axis=0)) if not math.isfinite(float(std)) or float(std) == 0.0]
    zero_fraction = np.mean(matrix == 0.0, axis=0) if matrix.size else np.empty(0)
    means = np.nanmean(matrix, axis=0) if matrix.size else np.empty(0)
    stds = np.nanstd(matrix, axis=0) if matrix.size else np.empty(0)
    finite_values = np.isfinite(matrix)
    quality_errors = list(schema_errors) + list(timestamp_errors)
    if nonfinite:
        quality_errors.append(f"nonfinite_values={nonfinite}")
    if duplicate_rows:
        quality_errors.append(f"duplicate_rows={duplicate_rows}")
    if constants:
        quality_errors.append(f"constant_columns={len(constants)}")
    if expected_sha256 and sha256 != expected_sha256:
        quality_errors.append(f"sha256={sha256}, expected={expected_sha256}")

    result = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": dataset,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256,
        "source": {
            "repository": source_repository,
            "revision": source_revision,
            "license": license_name,
        },
        "schema": {
            "time_column": time_col,
            "target": target,
            "columns": columns,
            "value_columns": value_columns,
            "rows": int(len(frame)),
            "channels": actual_channels,
            "expected_channels": expected_channels,
        },
        "time": {
            "start": timestamps.iloc[0].isoformat() if len(timestamps) and not pd.isna(timestamps.iloc[0]) else None,
            "end": timestamps.iloc[-1].isoformat() if len(timestamps) and not pd.isna(timestamps.iloc[-1]) else None,
            "expected_frequency": expected_freq,
            "expected_delta": str(expected_delta) if expected_delta is not None else None,
            "delta_counts": delta_counts,
            "duplicate_timestamps": duplicate_timestamps,
            "monotonic_increasing": monotonic,
            "unexpected_intervals": unexpected_intervals,
        },
        "values": {
            "numeric_coercion_nan": nonnumeric,
            "nonfinite": nonfinite,
            "sentinels": sentinel_counts,
            "duplicate_rows": duplicate_rows,
            "constant_columns": constants,
            "zero_fraction_max": float(np.max(zero_fraction)) if zero_fraction.size else None,
            "zero_fraction_ge_half": int(np.sum(zero_fraction >= 0.5)) if zero_fraction.size else 0,
            "per_channel": [
                {
                    "channel": index,
                    "name": value_columns[index],
                    "mean": means[index],
                    "std": stds[index],
                    "zero_fraction": zero_fraction[index],
                }
                for index in range(actual_channels)
            ],
        },
        "quality_pass": not quality_errors,
        "quality_errors": quality_errors,
    }
    return _json_value(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--time-col", default="date")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--expected-channels", type=int, default=None)
    parser.add_argument("--expected-freq", default=None)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--sentinel", type=float, action="append", default=[-9999.0])
    parser.add_argument("--source-repository", default=None)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--license", dest="license_name", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="exit 2 when quality_pass is false")
    args = parser.parse_args()
    try:
        result = audit(
            args.csv.resolve(), args.dataset, args.time_col, args.target,
            args.expected_channels, args.expected_freq, args.expected_sha256,
            args.sentinel, args.source_repository, args.source_revision,
            args.license_name,
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"dataset audit failed: {exc}") from exc
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset": result["dataset"],
        "rows": result["schema"]["rows"],
        "channels": result["schema"]["channels"],
        "sha256": result["sha256"],
        "quality_pass": result["quality_pass"],
        "quality_errors": len(result["quality_errors"]),
        "output": str(output),
    }, indent=2))
    if args.strict and not result["quality_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
