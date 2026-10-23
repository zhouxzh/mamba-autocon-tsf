import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from utils.dtwloss.dilate_loss import dilate_loss

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    denominator = np.maximum(np.abs(true), np.finfo(np.float32).eps)
    return np.mean(np.abs(pred - true) / denominator)


def MSPE(pred, true):
    denominator = np.maximum(np.abs(true), np.finfo(np.float32).eps)
    return np.mean(np.square((pred - true) / denominator))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def inverse_scaled_arrays(pred, true, scaler):
    """Inverse-transform saved forecast arrays for supported S/M/MS outputs."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    n_features = int(getattr(scaler, "n_features_in_", len(scaler.mean_)))
    channels = int(pred.shape[-1])
    if channels == n_features:
        shape = pred.shape
        pred_out = scaler.inverse_transform(pred.reshape(-1, channels)).reshape(shape)
        true_out = scaler.inverse_transform(true.reshape(-1, channels)).reshape(shape)
        return pred_out, true_out
    if channels == 1 and n_features >= 1:
        # MS keeps only the target (last registered value column) after the
        # forecast head, so apply the target column's affine transform.
        scale = float(np.asarray(scaler.scale_)[-1])
        mean = float(np.asarray(scaler.mean_)[-1])
        return pred * scale + mean, true * scale + mean
    raise ValueError(
        f"cannot inverse-transform {channels} output channels with {n_features} scaler features"
    )


def per_channel_metrics(pred, true):
    """Return MAE, MSE, and RMSE for each output channel."""
    residual = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return np.column_stack((
        np.mean(np.abs(residual), axis=(0, 1)),
        np.mean(residual ** 2, axis=(0, 1)),
        np.sqrt(np.mean(residual ** 2, axis=(0, 1))),
    ))


class StreamingMetricAccumulator:
    """Accumulate forecast metrics without retaining a full test tensor.

    High-dimensional datasets can make ``concatenate(preds)`` consume many GB
    even when prediction arrays are not requested.  This accumulator keeps only
    scalar/channel sums and a bounded number of windows for manual inspection.
    The reduction order is deterministic within a DataLoader traversal and
    matches :func:`metric` and :func:`per_channel_metrics` mathematically.
    """

    def __init__(self, scaler, output_channels, pred_len, origin_stride=None, sample_limit=4):
        self.output_channels = int(output_channels)
        self.pred_len = int(pred_len)
        if self.output_channels <= 0 or self.pred_len <= 0:
            raise ValueError("output_channels and pred_len must be positive")
        self.origin_stride = int(origin_stride or pred_len)
        if self.origin_stride <= 0:
            raise ValueError("origin_stride must be positive")
        self.sample_limit = max(0, int(sample_limit))
        self.window_count = 0
        self.element_count = 0
        self.origin_count = 0
        self._abs = np.zeros(self.output_channels, dtype=np.float64)
        self._sq = np.zeros(self.output_channels, dtype=np.float64)
        self._ape = np.zeros(self.output_channels, dtype=np.float64)
        self._spe = np.zeros(self.output_channels, dtype=np.float64)
        self._orig_abs = np.zeros(self.output_channels, dtype=np.float64)
        self._orig_sq = np.zeros(self.output_channels, dtype=np.float64)
        self._orig_ape = np.zeros(self.output_channels, dtype=np.float64)
        self._orig_spe = np.zeros(self.output_channels, dtype=np.float64)
        self._origin_abs = 0.0
        self._origin_sq = 0.0
        self._origin_ape = 0.0
        self._origin_spe = 0.0
        self._origin_orig_abs = 0.0
        self._origin_orig_sq = 0.0
        self._origin_orig_ape = 0.0
        self._origin_orig_spe = 0.0
        self._samples_pred = []
        self._samples_true = []

        scale = np.asarray(getattr(scaler, "scale_", []), dtype=np.float64).reshape(-1)
        mean = np.asarray(getattr(scaler, "mean_", []), dtype=np.float64).reshape(-1)
        if scale.size == self.output_channels and mean.size == self.output_channels:
            self._scale = scale
            self._mean = mean
        elif self.output_channels == 1 and scale.size and mean.size:
            # MS/S outputs retain the target (last registered column).
            self._scale = scale[-1:]
            self._mean = mean[-1:]
        else:
            raise ValueError(
                "scaler dimensions do not match output channels: "
                f"scale={scale.size}, mean={mean.size}, channels={self.output_channels}"
            )

    @staticmethod
    def _batch_sums(residual, truth, scale, mean):
        residual = np.asarray(residual, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        denominator = np.maximum(np.abs(truth), np.finfo(np.float32).eps)
        original_truth = truth * scale.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        original_residual = residual * scale.reshape(1, 1, -1)
        original_denominator = np.maximum(np.abs(original_truth), np.finfo(np.float32).eps)
        axes = (0, 1)
        return (
            np.sum(np.abs(residual), axis=axes),
            np.sum(residual * residual, axis=axes),
            np.sum(np.abs(residual) / denominator, axis=axes),
            np.sum((residual / denominator) ** 2, axis=axes),
            np.sum(np.abs(original_residual), axis=axes),
            np.sum(original_residual * original_residual, axis=axes),
            np.sum(np.abs(original_residual) / original_denominator, axis=axes),
            np.sum((original_residual / original_denominator) ** 2, axis=axes),
        )

    def update(self, pred, true):
        pred = np.asarray(pred)
        true = np.asarray(true)
        if pred.ndim != 3 or true.shape != pred.shape:
            raise ValueError(f"streaming update expects matching [B,T,C] arrays, got {pred.shape}/{true.shape}")
        if tuple(pred.shape[1:]) != (self.pred_len, self.output_channels):
            raise ValueError(
                f"streaming shape mismatch: got {pred.shape[1:]}, "
                f"expected {(self.pred_len, self.output_channels)}"
            )
        if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true)):
            raise ValueError("streaming update received non-finite predictions or targets")
        values = self._batch_sums(pred - true, true, self._scale, self._mean)
        (abs_sum, sq_sum, ape_sum, spe_sum,
         orig_abs_sum, orig_sq_sum, orig_ape_sum, orig_spe_sum) = values
        self._abs += abs_sum
        self._sq += sq_sum
        self._ape += ape_sum
        self._spe += spe_sum
        self._orig_abs += orig_abs_sum
        self._orig_sq += orig_sq_sum
        self._orig_ape += orig_ape_sum
        self._orig_spe += orig_spe_sum

        batch_size = int(pred.shape[0])
        start = self.window_count
        indices = np.arange(start, start + batch_size)
        origin_mask = (indices % self.origin_stride) == 0
        if np.any(origin_mask):
            origin_pred = pred[origin_mask]
            origin_true = true[origin_mask]
            origin_values = self._batch_sums(
                origin_pred - origin_true, origin_true, self._scale, self._mean
            )
            self._origin_abs += float(np.sum(origin_values[0]))
            self._origin_sq += float(np.sum(origin_values[1]))
            self._origin_ape += float(np.sum(origin_values[2]))
            self._origin_spe += float(np.sum(origin_values[3]))
            self._origin_orig_abs += float(np.sum(origin_values[4]))
            self._origin_orig_sq += float(np.sum(origin_values[5]))
            self._origin_orig_ape += float(np.sum(origin_values[6]))
            self._origin_orig_spe += float(np.sum(origin_values[7]))
            self.origin_count += int(origin_pred.shape[0])
            remaining = self.sample_limit - sum(item.shape[0] for item in self._samples_pred)
            if remaining > 0:
                self._samples_pred.append(np.asarray(origin_pred[:remaining], dtype=np.float32))
                self._samples_true.append(np.asarray(origin_true[:remaining], dtype=np.float32))
        self.window_count += batch_size
        self.element_count += int(pred.shape[0] * pred.shape[1] * pred.shape[2])

    @staticmethod
    def _metrics(abs_sum, sq_sum, ape_sum, spe_sum, count):
        if count <= 0:
            return [None] * 5
        mse = float(np.sum(sq_sum) / count)
        return [
            float(np.sum(abs_sum) / count),
            mse,
            float(np.sqrt(mse)),
            float(np.sum(ape_sum) / count),
            float(np.sum(spe_sum) / count),
        ]

    @staticmethod
    def _channel_metrics(abs_sum, sq_sum, ape_sum, spe_sum, count):
        if count <= 0:
            return np.full((len(abs_sum), 3), np.nan, dtype=np.float64)
        mse = sq_sum / count
        return np.column_stack((abs_sum / count, mse, np.sqrt(mse)))

    def to_payload(self):
        if self.element_count <= 0:
            raise ValueError("cannot finalize an empty streaming accumulator")
        scaled = self._metrics(self._abs, self._sq, self._ape, self._spe, self.element_count)
        original = self._metrics(
            self._orig_abs, self._orig_sq, self._orig_ape, self._orig_spe, self.element_count
        )
        scaled_channels = self._channel_metrics(
            self._abs, self._sq, self._ape, self._spe,
            self.window_count * self.pred_len,
        )
        original_channels = self._channel_metrics(
            self._orig_abs, self._orig_sq, self._orig_ape, self._orig_spe,
            self.window_count * self.pred_len,
        )
        # Origin accumulators sum all output channels, so their denominator
        # must count every selected time/channel element just like the full
        # streaming reduction above.
        origin_count = self.origin_count * self.pred_len * self.output_channels
        origin_scaled = self._metrics(
            np.asarray([self._origin_abs]), np.asarray([self._origin_sq]),
            np.asarray([self._origin_ape]), np.asarray([self._origin_spe]), origin_count
        ) if origin_count else [None] * 5
        origin_original = self._metrics(
            np.asarray([self._origin_orig_abs]), np.asarray([self._origin_orig_sq]),
            np.asarray([self._origin_orig_ape]), np.asarray([self._origin_orig_spe]), origin_count
        ) if origin_count else [None] * 5
        samples_pred = np.concatenate(self._samples_pred, axis=0) if self._samples_pred else np.empty((0, self.pred_len, self.output_channels), dtype=np.float32)
        samples_true = np.concatenate(self._samples_true, axis=0) if self._samples_true else np.empty((0, self.pred_len, self.output_channels), dtype=np.float32)
        return {
            "schema_version": 1,
            "aggregation": "streaming_window_weighted",
            "window_count": int(self.window_count),
            "element_count": int(self.element_count),
            "output_channels": int(self.output_channels),
            "pred_len": int(self.pred_len),
            "origin_stride": int(self.origin_stride),
            "origin_count": int(self.origin_count),
            "metric_order": ["mae", "mse", "rmse", "mape", "mspe"],
            "channel_metric_order": ["mae", "mse", "rmse", "original_mae", "original_mse", "original_rmse"],
            "scaled": {name: value for name, value in zip(["mae", "mse", "rmse", "mape", "mspe"], scaled)},
            "original": {name: value for name, value in zip(["mae", "mse", "rmse", "mape", "mspe"], original)},
            "channel_metrics": np.column_stack((scaled_channels, original_channels)).tolist(),
            "origin": {
                "scaled": {name: value for name, value in zip(["mae", "mse", "rmse", "mape", "mspe"], origin_scaled)},
                "original": {name: value for name, value in zip(["mae", "mse", "rmse", "mape", "mspe"], origin_original)},
            },
            "sample_pred": samples_pred,
            "sample_true": samples_true,
        }


def write_streaming_artifacts(result_dir, accumulator):
    """Write compact metric artifacts and return the JSON payload."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = accumulator.to_payload()
    metric_names = ["mae", "mse", "rmse", "mape", "mspe"]
    np.save(result_dir / "metrics.npy", np.asarray([payload["scaled"][name] for name in metric_names], dtype=np.float64))
    np.save(result_dir / "original_metrics.npy", np.asarray([payload["original"][name] for name in metric_names], dtype=np.float64))
    np.save(result_dir / "channel_metrics.npy", np.asarray(payload["channel_metrics"], dtype=np.float64))
    sample_pred = np.asarray(payload.pop("sample_pred"), dtype=np.float32)
    sample_true = np.asarray(payload.pop("sample_true"), dtype=np.float32)
    sample_path = result_dir / "stream_samples.npz"
    np.savez_compressed(sample_path, pred=sample_pred, true=sample_true)
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    payload["sample_artifact"] = {
        "path": str(sample_path),
        "sha256": digest,
        "shape": list(sample_pred.shape),
    }
    stream_path = result_dir / "stream_metrics.json"
    stream_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def shape_metric(pred, true, batch_size=100):
    dilate_e = 0.0
    shape_e = 0.0
    temp_e = 0.0
    pred = torch.tensor(pred).cuda()
    true = torch.tensor(true).cuda()
    for st in range(0, pred.shape[0], batch_size):
        st = st
        ed = st + batch_size
        # print(st, ed)
        sub_pred = pred[st:ed]
        sub_true = true[st:ed]
        n_data = sub_true.shape[0]
        with torch.no_grad():
            s_dilate_e, s_shape_DTW, s_temporal_DTW = dilate_loss(sub_pred, sub_true, 0.5, 0.01, 'cuda')

            dilate_e += s_dilate_e.cpu().item() * n_data
            shape_e += s_shape_DTW.cpu().item() * n_data
            temp_e += s_temporal_DTW.cpu().item() * n_data

    return dilate_e/pred.shape[0], shape_e/pred.shape[0], temp_e/pred.shape[0]
