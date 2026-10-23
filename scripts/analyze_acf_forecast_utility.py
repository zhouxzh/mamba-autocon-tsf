#!/usr/bin/env python
"""Audit train-only ACF utility and causal seasonal baselines.

The script is intentionally descriptive.  It estimates signed, mean-centered
ACF values from the training split only, then evaluates deterministic last and
seasonal-lag baselines on train/test windows.  Test values are read only for
the reported error; no test value is used to fit an ACF, scaler, or choose a
lag.  Outputs are written outside the frozen ``artifacts/paper_final`` tree.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DATASETS = {
    "electricity": {
        "path": Path("dataset/electricity/electricity.csv"),
        "expected_channels": 321,
    },
    "traffic": {
        "path": Path("dataset/traffic/traffic.csv"),
        "expected_channels": 862,
    },
}
DEFAULT_LAGS = (1, 24, 48, 96, 168, 336, 720)
DEFAULT_HORIZONS = (96, 336, 720)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric value encountered")
    return value


def _validate_int_list(values: Sequence[int], name: str) -> Tuple[int, ...]:
    parsed = tuple(sorted({int(value) for value in values}))
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("{} must contain positive integers".format(name))
    return parsed


def _load_values(root: Path, dataset: str) -> Tuple[np.ndarray, Path, str]:
    """Load numeric columns without retaining the timestamp DataFrame column."""

    spec = DATASETS[dataset]
    path = (root / spec["path"]).resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("dataset file is missing or symlinked: {}".format(spec["path"]))

    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    if not columns or columns[0] != "date":
        raise ValueError("{} must start with a date column".format(spec["path"]))
    numeric_columns = columns[1:]
    expected_channels = int(spec["expected_channels"])
    if len(numeric_columns) != expected_channels:
        raise ValueError(
            "{} has {} value columns; expected {}".format(
                dataset, len(numeric_columns), expected_channels
            )
        )

    frame = pd.read_csv(path, usecols=numeric_columns, dtype=np.float32)
    values = frame.to_numpy(dtype=np.float32, copy=True)
    del frame
    if values.ndim != 2 or values.shape[1] != expected_channels:
        raise ValueError("{} numeric matrix has unexpected shape {}".format(dataset, values.shape))
    if not np.isfinite(values).all():
        raise ValueError("{} contains NaN or infinite numeric values".format(dataset))
    return values, path, _sha256(path)


def _split_bounds(n_rows: int, train_ratio: float, seq_len: int) -> Dict[str, int]:
    train_end = int(n_rows * train_ratio)
    test_rows = int(n_rows * 0.2)
    val_end = n_rows - test_rows
    test_start = val_end - seq_len
    if train_end <= seq_len:
        raise ValueError("training split is shorter than seq_len")
    if test_start < 0 or test_start >= n_rows:
        raise ValueError("test context boundary is invalid")
    return {
        "train_end": train_end,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": n_rows,
    }


def _signed_acf(values: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Return [channels, lags] biased, normalized signed ACF values."""

    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("values must have shape [time, channels]")
    max_lag = max(lags)
    if max_lag >= values.shape[0]:
        raise ValueError("candidate lag {} exceeds train length {}".format(max_lag, values.shape[0]))

    # Mean centering is performed on the train split only.  The denominator is
    # the lag-zero sum of squares, matching the biased normalized ACF used by
    # the repository's FFT helper while retaining the sign.
    centered = np.asarray(values, dtype=np.float64) - np.asarray(values, dtype=np.float64).mean(
        axis=0, keepdims=True
    )
    denominator = np.sum(centered * centered, axis=0, dtype=np.float64)
    result = np.zeros((values.shape[1], len(lags)), dtype=np.float64)
    for column, lag in enumerate(lags):
        numerator = np.sum(
            centered[int(lag) :] * centered[: -int(lag)], axis=0, dtype=np.float64
        )
        np.divide(numerator, denominator, out=result[:, column], where=denominator > 0.0)
    return np.clip(result, -1.0, 1.0)


def _acf_rows(dataset: str, acf_values: np.ndarray, lags: Sequence[int], train_n: int) -> List[dict]:
    rows = []
    for index, lag in enumerate(lags):
        values = acf_values[:, index]
        rows.append(
            {
                "dataset": dataset,
                "split": "train",
                "lag": int(lag),
                "train_observations": int(train_n),
                "channels": int(values.size),
                "mean_signed_acf": _finite_float(np.mean(values)),
                "median_signed_acf": _finite_float(np.median(values)),
                "std_signed_acf": _finite_float(np.std(values)),
                "q10_signed_acf": _finite_float(np.quantile(values, 0.10)),
                "q90_signed_acf": _finite_float(np.quantile(values, 0.90)),
                "min_signed_acf": _finite_float(np.min(values)),
                "max_signed_acf": _finite_float(np.max(values)),
                "mean_abs_acf": _finite_float(np.mean(np.abs(values))),
                "median_abs_acf": _finite_float(np.median(np.abs(values))),
                "positive_fraction": _finite_float(np.mean(values > 0.0)),
                "negative_fraction": _finite_float(np.mean(values < 0.0)),
            }
        )
    return rows


def _acf_channel_rows(dataset: str, acf_values: np.ndarray, lags: Sequence[int]) -> List[dict]:
    rows = []
    for channel in range(acf_values.shape[0]):
        for index, lag in enumerate(lags):
            rows.append(
                {
                    "dataset": dataset,
                    "channel_index": int(channel),
                    "lag": int(lag),
                    "signed_acf": _finite_float(acf_values[channel, index]),
                }
            )
    return rows


def _availability(seq_len: int, lag: int, horizon: int) -> np.ndarray:
    """Whether direct index L-1+h-lag is in the observed history window."""

    steps = np.arange(1, horizon + 1, dtype=np.int64)
    indices = int(seq_len) - 1 + steps - int(lag)
    return (indices >= 0) & (indices < int(seq_len))


def _availability_rows(dataset: str, seq_len: int, lags: Sequence[int], horizons: Sequence[int]) -> List[dict]:
    rows = []
    for lag in lags:
        for horizon in horizons:
            valid = _availability(seq_len, lag, horizon)
            valid_steps = np.flatnonzero(valid) + 1
            rows.append(
                {
                    "dataset": dataset,
                    "seq_len": int(seq_len),
                    "lag": int(lag),
                    "horizon": int(horizon),
                    "available_steps": int(valid.sum()),
                    "horizon_steps": int(horizon),
                    "availability_ratio": _finite_float(float(valid.mean())),
                    "first_available_step": int(valid_steps[0]) if valid_steps.size else None,
                    "last_available_step": int(valid_steps[-1]) if valid_steps.size else None,
                    "lag_within_input": bool(lag <= seq_len),
                }
            )
    return rows


class _MetricAccumulator:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, error: np.ndarray) -> None:
        error = np.asarray(error, dtype=np.float64)
        if error.size == 0:
            return
        if not np.isfinite(error).all():
            raise ValueError("baseline error became non-finite")
        self.abs_sum += float(np.abs(error).sum(dtype=np.float64))
        self.sq_sum += float(np.square(error).sum(dtype=np.float64))
        self.count += int(error.size)

    def update_reduced(self, abs_sum: float, sq_sum: float, count: int) -> None:
        """Add sufficient statistics already reduced over batch/time/channel.

        The vectorized baseline path uses this method so it does not create a
        second full-size array just to standardize an error tensor.  Keeping
        the accumulators in float64 preserves the same element-wise means as
        :meth:`update` while allowing the working tensors to remain float32.
        """

        if count < 0 or not math.isfinite(float(abs_sum)) or not math.isfinite(float(sq_sum)):
            raise ValueError("invalid reduced metric statistics")
        self.abs_sum += float(abs_sum)
        self.sq_sum += float(sq_sum)
        self.count += int(count)

    def metrics(self) -> Dict[str, Optional[float]]:
        if self.count == 0:
            return {"mae": None, "mse": None, "rmse": None, "elements": 0}
        mae = self.abs_sum / self.count
        mse = self.sq_sum / self.count
        return {
            "mae": _finite_float(mae),
            "mse": _finite_float(mse),
            "rmse": _finite_float(math.sqrt(mse)),
            "elements": int(self.count),
        }


def _window_count(length: int, seq_len: int, horizon: int) -> int:
    return max(0, int(length) - int(seq_len) - int(horizon) + 1)


def _iter_window_chunks(
    values: np.ndarray,
    seq_len: int,
    horizon: int,
    batch_size: int,
    origin_stride: int = 1,
):
    count = _window_count(values.shape[0], seq_len, horizon)
    if origin_stride <= 0:
        raise ValueError("origin_stride must be positive")
    history_offsets = np.arange(seq_len, dtype=np.int64)
    target_offsets = np.arange(horizon, dtype=np.int64)
    origins_all = np.arange(0, count, int(origin_stride), dtype=np.int64)
    for start in range(0, origins_all.size, batch_size):
        stop = min(origins_all.size, start + batch_size)
        origins = origins_all[start:stop]
        history = values[origins[:, None] + history_offsets[None, :], :]
        target = values[
            origins[:, None] + int(seq_len) + target_offsets[None, :], :
        ]
        yield history, target


def _error_row(
    *,
    dataset: str,
    split: str,
    baseline: str,
    evaluation_scope: str,
    lag: Optional[int],
    seq_len: int,
    horizon: int,
    n_windows: int,
    available_steps: int,
    availability_ratio: float,
    scaled: _MetricAccumulator,
    original: _MetricAccumulator,
) -> dict:
    scaled_metrics = scaled.metrics()
    original_metrics = original.metrics()
    return {
        "dataset": dataset,
        "split": split,
        "baseline": baseline,
        "evaluation_scope": evaluation_scope,
        "lag": int(lag) if lag is not None else None,
        "seq_len": int(seq_len),
        "horizon": int(horizon),
        "n_windows": int(n_windows),
        "available_steps_per_window": int(available_steps),
        "availability_ratio": _finite_float(availability_ratio),
        "evaluated_elements": int(scaled_metrics["elements"]),
        "scaled_mae": scaled_metrics["mae"],
        "scaled_mse": scaled_metrics["mse"],
        "scaled_rmse": scaled_metrics["rmse"],
        "original_mae": original_metrics["mae"],
        "original_mse": original_metrics["mse"],
        "original_rmse": original_metrics["rmse"],
    }


def _evaluate_split(
    *,
    dataset: str,
    split: str,
    values: np.ndarray,
    seq_len: int,
    horizons: Sequence[int],
    lags: Sequence[int],
    train_scale: np.ndarray,
    batch_size: int,
    horizon_chunk_size: int = 96,
    lag_batch_size: int = 4,
    origin_stride: int = 1,
) -> List[dict]:
    """Evaluate baselines with bounded batched indexing.

    A direct lag source is contiguous in forecast-step space.  We therefore
    gather a small ``[batch, horizon_chunk, lag_block, channel]`` tensor and
    reduce it immediately, instead of iterating over every window/horizon/lag
    in Python.  The resulting sufficient statistics are mathematically the
    same as the element-wise implementation and keep peak memory bounded.
    """

    if horizon_chunk_size <= 0 or lag_batch_size <= 0 or origin_stride <= 0:
        raise ValueError("horizon_chunk_size, lag_batch_size, and origin_stride must be positive")
    # Use float64 only for train-scaler moments and scalar accumulators.  The
    # source data remain float32, which bounds the temporary lag tensor.
    scale = np.asarray(train_scale, dtype=np.float64)
    inv_scale = 1.0 / scale
    inv_scale_sq = inv_scale * inv_scale

    def add_error(
        scaled_acc: _MetricAccumulator,
        original_acc: _MetricAccumulator,
        error: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Reduce one error tensor and update original/scaled accumulators."""

        if mask is not None:
            error = np.where(mask, error, 0.0)
            # ``mask`` is usually shaped [1, H, 1] and broadcasts over the
            # batch and channel axes.  Count the broadcasted elements so each
            # window contributes its available forecast steps.
            count = int(np.broadcast_to(mask, error.shape).sum(dtype=np.int64))
        else:
            count = int(error.size)
        if count == 0:
            return
        abs_by_channel = np.abs(error).sum(axis=tuple(range(error.ndim - 1)), dtype=np.float64)
        sq_by_channel = np.square(error).sum(axis=tuple(range(error.ndim - 1)), dtype=np.float64)
        # The final axis is channels; for a lag tensor the preceding reduction
        # axes leave [lags, channels], while a plain error leaves [channels].
        original_acc.update_reduced(
            float(abs_by_channel.sum(dtype=np.float64)),
            float(sq_by_channel.sum(dtype=np.float64)),
            count,
        )
        scaled_acc.update_reduced(
            float(np.sum(abs_by_channel * inv_scale, dtype=np.float64)),
            float(np.sum(sq_by_channel * inv_scale_sq, dtype=np.float64)),
            count,
        )

    rows: List[dict] = []
    for horizon in horizons:
        total_windows = _window_count(values.shape[0], seq_len, horizon)
        n_windows = (total_windows + int(origin_stride) - 1) // int(origin_stride)
        if n_windows == 0:
            # Preserve the matrix shape in short/synthetic inputs while making
            # the absence of windows explicit.
            empty_scaled = _MetricAccumulator()
            empty_original = _MetricAccumulator()
            rows.append(
                _error_row(
                    dataset=dataset, split=split, baseline="last",
                    evaluation_scope="all_steps", lag=None, seq_len=seq_len,
                    horizon=horizon, n_windows=0, available_steps=horizon,
                    availability_ratio=1.0, scaled=empty_scaled,
                    original=empty_original,
                )
            )
            for lag in lags:
                empty_scaled = _MetricAccumulator()
                empty_original = _MetricAccumulator()
                availability = _availability(seq_len, lag, horizon)
                rows.extend([
                    _error_row(
                        dataset=dataset, split=split, baseline="seasonal_lag",
                        evaluation_scope="available_only", lag=lag,
                        seq_len=seq_len, horizon=horizon, n_windows=0,
                        available_steps=int(availability.sum()),
                        availability_ratio=float(availability.mean()),
                        scaled=empty_scaled, original=empty_original,
                    ),
                    _error_row(
                        dataset=dataset, split=split, baseline="seasonal_lag",
                        evaluation_scope="last_fallback", lag=lag,
                        seq_len=seq_len, horizon=horizon, n_windows=0,
                        available_steps=int(availability.sum()),
                        availability_ratio=float(availability.mean()),
                        scaled=_MetricAccumulator(), original=_MetricAccumulator(),
                    ),
                ])
            continue

        last_scaled = _MetricAccumulator()
        last_original = _MetricAccumulator()
        seasonal_accumulators = {
            lag: {
                "available_scaled": _MetricAccumulator(),
                "available_original": _MetricAccumulator(),
                "fallback_scaled": _MetricAccumulator(),
                "fallback_original": _MetricAccumulator(),
            }
            for lag in lags
        }
        availability_by_lag = {lag: _availability(seq_len, lag, horizon) for lag in lags}
        availability_matrix = np.stack([availability_by_lag[lag] for lag in lags], axis=1)
        for history, target in _iter_window_chunks(
            values, seq_len, horizon, batch_size, origin_stride=origin_stride
        ):
            last = history[:, -1:, :]
            horizon_start = 0
            while horizon_start < horizon:
                horizon_stop = min(horizon, horizon_start + horizon_chunk_size)
                target_chunk = target[:, horizon_start:horizon_stop, :]
                last_chunk = np.broadcast_to(last, target_chunk.shape)
                last_error = last_chunk - target_chunk
                add_error(last_scaled, last_original, last_error)

                # Gather several lags at once.  ``history[:, indices, :]``
                # yields [batch, horizon_chunk, lag_block, channels].
                steps = np.arange(horizon_start + 1, horizon_stop + 1, dtype=np.int64)
                for lag_start in range(0, len(lags), lag_batch_size):
                    lag_stop = min(len(lags), lag_start + lag_batch_size)
                    lag_block = lags[lag_start:lag_stop]
                    indices = seq_len - 1 + steps[:, None] - np.asarray(
                        lag_block, dtype=np.int64
                    )[None, :]
                    safe_indices = np.clip(indices, 0, seq_len - 1)
                    seasonal = history[:, safe_indices, :]
                    seasonal_error = seasonal - target_chunk[:, :, None, :]
                    valid = availability_matrix[horizon_start:horizon_stop, lag_start:lag_stop]
                    valid_mask = valid[None, :, :, None]
                    for block_offset, lag in enumerate(lag_block):
                        state = seasonal_accumulators[lag]
                        # Slice one lag from the gathered tensor for the
                        # available-only accumulator.  The fallback tensor is
                        # kept batched across the lag block below.
                        one_error = seasonal_error[:, :, block_offset, :]
                        one_valid = valid[:, block_offset][None, :, None]
                        add_error(
                            state["available_scaled"],
                            state["available_original"],
                            one_error,
                            mask=one_valid,
                        )
                    fallback_error = np.where(
                        valid_mask,
                        seasonal_error,
                        last_error[:, :, None, :],
                    )
                    # Reduce each lag independently while retaining the same
                    # full-horizon denominator for every fallback baseline.
                    for block_offset, lag in enumerate(lag_block):
                        state = seasonal_accumulators[lag]
                        add_error(
                            state["fallback_scaled"],
                            state["fallback_original"],
                            fallback_error[:, :, block_offset, :],
                        )
                horizon_start = horizon_stop

        rows.append(
            _error_row(
                dataset=dataset, split=split, baseline="last",
                evaluation_scope="all_steps", lag=None, seq_len=seq_len,
                horizon=horizon, n_windows=n_windows, available_steps=horizon,
                availability_ratio=1.0, scaled=last_scaled,
                original=last_original,
            )
        )
        for lag in lags:
            valid = availability_by_lag[lag]
            state = seasonal_accumulators[lag]
            rows.extend([
                _error_row(
                    dataset=dataset, split=split, baseline="seasonal_lag",
                    evaluation_scope="available_only", lag=lag,
                    seq_len=seq_len, horizon=horizon, n_windows=n_windows,
                    available_steps=int(valid.sum()),
                    availability_ratio=float(valid.mean()),
                    scaled=state["available_scaled"],
                    original=state["available_original"],
                ),
                _error_row(
                    dataset=dataset, split=split, baseline="seasonal_lag",
                    evaluation_scope="last_fallback", lag=lag,
                    seq_len=seq_len, horizon=horizon, n_windows=n_windows,
                    available_steps=int(valid.sum()),
                    availability_ratio=float(valid.mean()),
                    scaled=state["fallback_scaled"],
                    original=state["fallback_original"],
                ),
            ])
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ensure_output_dir(root: Path, output_dir: Path, force: bool) -> Path:
    resolved = output_dir.resolve()
    repo_root = root.resolve()
    if repo_root != resolved and repo_root not in resolved.parents:
        raise ValueError("output_dir must remain inside the repository root")
    if "paper_final" in resolved.parts:
        raise ValueError("refusing to write under frozen artifacts/paper_final")
    if resolved.exists() and any(resolved.iterdir()) and not force:
        raise FileExistsError("output_dir is non-empty; pass --force to replace its contents")
    resolved.mkdir(parents=True, exist_ok=True)
    if force:
        for child in resolved.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                raise FileExistsError(
                    "--force will not recursively delete existing directory {}; choose a new output_dir".format(child)
                )
    return resolved


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("artifacts/exploratory/acf_forecast_utility"),
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--lags", nargs="+", type=int, default=list(DEFAULT_LAGS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--horizon_chunk_size", type=int, default=96,
        help="forecast steps gathered per bounded vectorized chunk",
    )
    parser.add_argument(
        "--lag_batch_size", type=int, default=4,
        help="candidate lags gathered together per vectorized chunk",
    )
    parser.add_argument(
        "--origin_stride", type=int, default=24,
        help=(
            "step between forecast origins used for descriptive baselines; "
            "1 evaluates every overlapping window, larger values bound runtime"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.seq_len <= 0 or args.batch_size <= 0:
        raise SystemExit("--seq_len and --batch_size must be positive")
    if args.horizon_chunk_size <= 0 or args.lag_batch_size <= 0 or args.origin_stride <= 0:
        raise SystemExit(
            "--horizon_chunk_size, --lag_batch_size, and --origin_stride must be positive"
        )
    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("--train_ratio must be between zero and one")
    try:
        lags = _validate_int_list(args.lags, "--lags")
        horizons = _validate_int_list(args.horizons, "--horizons")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    root = args.root.resolve()
    output_dir = _ensure_output_dir(
        root,
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir,
        args.force,
    )

    acf_rows: List[dict] = []
    acf_channel_rows: List[dict] = []
    availability_rows: List[dict] = []
    baseline_rows: List[dict] = []
    dataset_metadata: Dict[str, dict] = {}

    for dataset in args.datasets:
        print("Loading {} (numeric columns only)".format(dataset), flush=True)
        values, source_path, source_hash = _load_values(root, dataset)
        bounds = _split_bounds(values.shape[0], args.train_ratio, args.seq_len)
        train_values = values[: bounds["train_end"]]
        test_values = values[bounds["test_start"] : bounds["test_end"]]
        if max(lags) >= train_values.shape[0]:
            raise ValueError("candidate lag exceeds {} train observations".format(dataset))

        print("Computing train-only signed ACF for {}".format(dataset), flush=True)
        acf_values = _signed_acf(train_values, lags)
        acf_rows.extend(_acf_rows(dataset, acf_values, lags, train_values.shape[0]))
        acf_channel_rows.extend(_acf_channel_rows(dataset, acf_values, lags))
        availability_rows.extend(_availability_rows(dataset, args.seq_len, lags, horizons))

        train_scale = np.asarray(train_values, dtype=np.float64).std(axis=0, ddof=0)
        train_scale = np.where(train_scale > np.finfo(np.float64).eps, train_scale, 1.0)
        print("Evaluating descriptive train/test baselines for {}".format(dataset), flush=True)
        baseline_rows.extend(
            _evaluate_split(
                dataset=dataset, split="train", values=train_values,
                seq_len=args.seq_len, horizons=horizons, lags=lags,
                train_scale=train_scale,
                batch_size=args.batch_size,
                horizon_chunk_size=args.horizon_chunk_size,
                lag_batch_size=args.lag_batch_size,
                origin_stride=args.origin_stride,
            )
        )
        baseline_rows.extend(
            _evaluate_split(
                dataset=dataset, split="test", values=test_values,
                seq_len=args.seq_len, horizons=horizons, lags=lags,
                train_scale=train_scale,
                batch_size=args.batch_size,
                horizon_chunk_size=args.horizon_chunk_size,
                lag_batch_size=args.lag_batch_size,
                origin_stride=args.origin_stride,
            )
        )
        dataset_metadata[dataset] = {
            "source": str(source_path.relative_to(root)),
            "sha256": source_hash,
            "rows": int(values.shape[0]),
            "channels": int(values.shape[1]),
            "train_end_exclusive": int(bounds["train_end"]),
            "validation_end_exclusive": int(bounds["val_end"]),
            "test_context_start_in_full_series": int(bounds["test_start"]),
            "test_end_exclusive": int(bounds["test_end"]),
            "train_rows": int(train_values.shape[0]),
            "test_context_rows": int(test_values.shape[0]),
            "train_windows_by_horizon": {
                str(horizon): int(_window_count(train_values.shape[0], args.seq_len, horizon))
                for horizon in horizons
            },
            "test_windows_by_horizon": {
                str(horizon): int(_window_count(test_values.shape[0], args.seq_len, horizon))
                for horizon in horizons
            },
        }
        del values, train_values, test_values, acf_values

    _write_csv(
        output_dir / "signed_acf_summary.csv",
        acf_rows,
        [
            "dataset", "split", "lag", "train_observations", "channels",
            "mean_signed_acf", "median_signed_acf", "std_signed_acf",
            "q10_signed_acf", "q90_signed_acf", "min_signed_acf",
            "max_signed_acf", "mean_abs_acf", "median_abs_acf",
            "positive_fraction", "negative_fraction",
        ],
    )
    _write_csv(
        output_dir / "signed_acf_by_channel.csv",
        acf_channel_rows,
        ["dataset", "channel_index", "lag", "signed_acf"],
    )
    _write_csv(
        output_dir / "lag_availability.csv",
        availability_rows,
        [
            "dataset", "seq_len", "lag", "horizon", "available_steps",
            "horizon_steps", "availability_ratio", "first_available_step",
            "last_available_step", "lag_within_input",
        ],
    )
    _write_csv(
        output_dir / "descriptive_baseline_metrics.csv",
        baseline_rows,
        [
            "dataset", "split", "baseline", "evaluation_scope", "lag",
            "seq_len", "horizon", "n_windows", "available_steps_per_window",
            "availability_ratio", "evaluated_elements", "scaled_mae",
            "scaled_mse", "scaled_rmse", "original_mae", "original_mse",
            "original_rmse",
        ],
    )
    metadata = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "script": str(Path(__file__).resolve().relative_to(root)),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "seq_len": int(args.seq_len),
        "train_ratio": float(args.train_ratio),
        "candidate_lags": [int(value) for value in lags],
        "horizons": [int(value) for value in horizons],
        "window_batch_size": int(args.batch_size),
        "horizon_chunk_size": int(args.horizon_chunk_size),
        "lag_batch_size": int(args.lag_batch_size),
        "origin_stride": int(args.origin_stride),
        "origin_sampling_note": (
            "baselines use origins 0, origin_stride, ...; set origin_stride=1 "
            "to evaluate every overlapping window"
        ),
        "acf_definition": (
            "train-only mean-centered signed ACF; numerator=sum(x[t]x[t-lag]) "
            "and denominator=sum(x[t]^2) over the full train segment"
        ),
        "availability_definition": (
            "direct causal source index L-1+h-lag must lie in [0,L-1], "
            "for h=1..horizon"
        ),
        "baseline_definition": {
            "last": "repeat the final observed value for every forecast step",
            "seasonal_available_only": "use direct lag source only where available; omit unavailable forecast elements",
            "seasonal_last_fallback": "use direct lag source where available and last value otherwise",
        },
        "test_boundary_note": (
            "test targets are read only to score descriptive baselines; ACF, "
            "scaler moments, and lag choices use train rows only"
        ),
        "datasets": dataset_metadata,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True, allow_nan=False)

    print("Wrote train-only ACF utility outputs to {}".format(output_dir), flush=True)
    print("  ACF rows: {}; channel rows: {}; availability rows: {}; baseline rows: {}".format(
        len(acf_rows), len(acf_channel_rows), len(availability_rows), len(baseline_rows)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
