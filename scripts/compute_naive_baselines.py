#!/usr/bin/env python
"""Compute deterministic last-value and training-window-mean baselines.

The baselines use exactly the repository Dataset classes, including their
chronological split and train-only StandardScaler.  Metrics are reported both
in the standardized space used by ``run.py`` and after inverse transformation.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct execution via ``python scripts/compute_naive_baselines.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute


DATASETS = {
    "ETTh1": (Dataset_ETT_hour, "dataset/ETT-small/ETTh1.csv", "OT", "h"),
    "ETTh2": (Dataset_ETT_hour, "dataset/ETT-small/ETTh2.csv", "OT", "h"),
    "ETTm1": (Dataset_ETT_minute, "dataset/ETT-small/ETTm1.csv", "OT", "t"),
    "ETTm2": (Dataset_ETT_minute, "dataset/ETT-small/ETTm2.csv", "OT", "t"),
    "weather": (Dataset_Custom, "dataset/weather/weather.csv", "OT", "t"),
    "exchange_rate": (Dataset_Custom, "dataset/exchange_rate/exchange_rate.csv", "OT", "d"),
    "national_illness": (Dataset_Custom, "dataset/illness/national_illness.csv", "OT", "d"),
}

# High-dimensional files are intentionally opt-in.  Loading them by default
# would turn the lightweight baseline command into a multi-gigabyte job and
# could unexpectedly compete with a training run for GPU/CPU memory.
HIGH_DIM_DATASETS = {
    "electricity": (Dataset_Custom, "dataset/electricity/electricity.csv", "OT", "h"),
    "traffic": (Dataset_Custom, "dataset/traffic/traffic.csv", "OT", "h"),
}
ALL_DATASETS = {**DATASETS, **HIGH_DIM_DATASETS}


# The baseline implementation materializes several float64 forecast-shaped
# arrays while calculating metrics in both standardized and original units.
# This estimate deliberately over-approximates that peak so the automatic
# high-dimensional chunk stays safe when numpy/scikit-learn allocate ufunc
# intermediates. It is a per-window estimate, independent of the data.
DEFAULT_LOW_DIM_CHUNK_SIZE = 512
DEFAULT_HIGH_DIM_CHUNK_SIZE = 32
DEFAULT_HIGH_DIM_MAX_CHUNK_BYTES = 256 * 1024 * 1024
_PEAK_HISTORY_FLOAT64_ARRAYS = 2
_PEAK_HORIZON_FLOAT64_ARRAYS = 12
_PEAK_HORIZON_BOOLEAN_ARRAYS = 3


def estimated_peak_bytes_per_window(seq_len, horizon, channels):
    """Return a conservative temporary-memory estimate for one test window."""
    dimensions = {
        "seq_len": seq_len,
        "horizon": horizon,
        "channels": channels,
    }
    for name, value in dimensions.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")

    float64_bytes = np.dtype(np.float64).itemsize
    bool_bytes = np.dtype(np.bool_).itemsize
    history_values = _PEAK_HISTORY_FLOAT64_ARRAYS * int(seq_len)
    horizon_values = _PEAK_HORIZON_FLOAT64_ARRAYS * int(horizon)
    boolean_values = _PEAK_HORIZON_BOOLEAN_ARRAYS * int(horizon)
    return int(channels) * (
        float64_bytes * (history_values + horizon_values)
        + bool_bytes * boolean_values
    )


def resolve_chunk_size(
        dataset_name, horizon, seq_len, channels, chunk_size=None, max_chunk_bytes=None):
    """Choose a deterministic chunk size, preserving an explicit override.

    Ordinary datasets retain the historical 512-window default. Opt-in
    high-dimensional datasets use a conservative temporary-memory budget when
    no explicit chunk is given. A user-supplied ``max_chunk_bytes`` also
    constrains ordinary datasets, which is useful on memory-limited hosts.
    """
    if chunk_size is not None:
        chunk = int(chunk_size)
        if chunk <= 0:
            raise ValueError("chunk_size must be positive")
        return chunk

    is_high_dimensional = dataset_name in HIGH_DIM_DATASETS
    default_chunk = (
        DEFAULT_HIGH_DIM_CHUNK_SIZE if is_high_dimensional
        else DEFAULT_LOW_DIM_CHUNK_SIZE
    )
    if max_chunk_bytes is None:
        if not is_high_dimensional:
            return default_chunk
        max_chunk_bytes = DEFAULT_HIGH_DIM_MAX_CHUNK_BYTES

    max_chunk_bytes = int(max_chunk_bytes)
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    bytes_per_window = estimated_peak_bytes_per_window(seq_len, horizon, channels)
    memory_limited_chunk = max(1, max_chunk_bytes // bytes_per_window)
    return min(default_chunk, memory_limited_chunk)


def metrics(pred, true):
    error = pred - true
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def run_one(
        dataset_name, horizon, seq_len, features, root, chunk_size=None,
        max_chunk_bytes=None):
    dataset_cls, relative_path, target, freq = ALL_DATASETS[dataset_name]
    path = root / relative_path
    # Weather and ILI use the custom loader's target names from the CSV.
    if dataset_name == "national_illness":
        target = "OT"
    dataset = dataset_cls(
        root_path=str(path.parent), data_path=path.name, flag="test",
        size=[seq_len, max(1, seq_len // 2), horizon], features=features,
        target=target, timeenc=0, freq=freq, seasonal_patterns="Monthly",
        train_ratio=0.6,
    )
    # ``Dataset.__len__`` intentionally mirrors the training code but Python
    # rejects a negative length for very long horizons on the short ILI set.
    n = len(dataset.data_x) - seq_len - horizon + 1
    if n <= 0:
        return []

    # Accumulate sufficient statistics over all windows.  Averaging the
    # metric of each fixed-size chunk would weight a short final chunk the
    # same as every full chunk and would not match ``exp.test``'s global
    # element-wise mean.
    accum = {
        name: {"scaled_abs": 0.0, "scaled_sq": 0.0,
               "original_abs": 0.0, "original_sq": 0.0,
               "count": 0, "direction_correct": 0, "direction_count": 0}
        for name in ("last", "mean")
    }
    channels = int(dataset.data_x.shape[-1])
    chunk = resolve_chunk_size(
        dataset_name, horizon, seq_len, channels,
        chunk_size=chunk_size, max_chunk_bytes=max_chunk_bytes,
    )
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        xs = []
        ys = []
        for index in range(start, stop):
            _, x, y, _, _ = dataset[index]
            xs.append(x)
            ys.append(y[-horizon:])
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        last = np.repeat(x[:, -1:, :], horizon, axis=1)
        mean = np.repeat(x.mean(axis=1, keepdims=True), horizon, axis=1)
        for name, prediction in (("last", last), ("mean", mean)):
            # Dataset.inverse_transform expects the channel dimension intact.
            pred_original = dataset.inverse_transform(prediction.reshape(-1, prediction.shape[-1])).reshape(prediction.shape)
            y_original = dataset.inverse_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
            baseline = x[:, -1:, :]
            state = accum[name]
            scaled_error = prediction - y
            original_error = pred_original - y_original
            state["scaled_abs"] += float(np.abs(scaled_error).sum())
            state["scaled_sq"] += float(np.square(scaled_error).sum())
            state["original_abs"] += float(np.abs(original_error).sum())
            state["original_sq"] += float(np.square(original_error).sum())
            state["count"] += int(scaled_error.size)
            actual_delta = y - baseline
            valid_direction = np.abs(actual_delta) > 1e-12
            state["direction_correct"] += int(np.sum(
                (np.sign(prediction - baseline) == np.sign(actual_delta)) & valid_direction
            ))
            state["direction_count"] += int(np.sum(valid_direction))

    rows = []
    for name in ("last", "mean"):
        state = accum[name]
        count = max(1, state["count"])
        direction_count = max(1, state["direction_count"])
        row = {"dataset": dataset_name, "model": name, "features": features,
               "seq_len": seq_len, "pred_len": horizon, "n_windows": n,
               "enc_in": channels, "dec_in": channels, "chunk_size": chunk,
               "c_out": 1 if features == "MS" else channels}
        for prefix, abs_sum, sq_sum in (
                ("scaled", state["scaled_abs"], state["scaled_sq"]),
                ("original", state["original_abs"], state["original_sq"])):
            mse = sq_sum / count
            row[f"{prefix}_mae"] = float(abs_sum / count)
            row[f"{prefix}_mse"] = float(mse)
            row[f"{prefix}_rmse"] = float(np.sqrt(mse))
        row["directional_accuracy"] = float(state["direction_correct"] / direction_count)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baselines/naive_metrics.csv"))
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="dataset names (default: low-dimensional datasets; electricity/traffic are opt-in)",
    )
    parser.add_argument(
        "--include_highdim", action="store_true",
        help="append electricity and traffic to the default dataset list",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--features", choices=["S", "M", "MS"], default="S")
    parser.add_argument(
        "--chunk_size", type=int, default=None,
        help="windows processed per chunk; overrides automatic memory sizing",
    )
    parser.add_argument(
        "--max_chunk_bytes", type=int, default=None,
        help=(
            "maximum estimated temporary bytes per automatic chunk; defaults "
            "to 256 MiB for electricity/traffic and preserves low-dimensional defaults"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.datasets is None:
        selected_datasets = list(DATASETS)
        if args.include_highdim:
            selected_datasets.extend(HIGH_DIM_DATASETS)
    else:
        selected_datasets = list(args.datasets)
        unknown = sorted(set(selected_datasets) - set(ALL_DATASETS))
        if unknown:
            raise SystemExit("unknown dataset(s): " + ", ".join(unknown))
    rows = []
    for dataset in selected_datasets:
        for horizon in args.horizons:
            rows.extend(run_one(
                dataset, horizon, args.seq_len, args.features, root,
                chunk_size=args.chunk_size, max_chunk_bytes=args.max_chunk_bytes,
            ))
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} naive baseline rows to {output}")


if __name__ == "__main__":
    main()
