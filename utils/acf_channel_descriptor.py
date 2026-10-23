"""Fixed train-prefix ACF channel descriptors for shared MambaTS conditioning."""

from __future__ import annotations

import numpy as np


def build_acf_channel_descriptor(acf_values, channel_peaks) -> np.ndarray:
    """Return a normalized, channel-count-independent ``[channels, 6]`` table."""
    acf = np.asarray(acf_values, dtype=np.float32)
    peaks = np.asarray(channel_peaks, dtype=np.int64)
    if acf.ndim != 2 or peaks.ndim != 2 or acf.shape[0] != peaks.shape[0]:
        raise ValueError("acf_values and channel_peaks must have matching [channels, ...] shapes")
    if acf.shape[1] < 2:
        raise ValueError("ACF descriptor requires at least one positive lag")
    descriptor = np.zeros((acf.shape[0], 6), dtype=np.float32)
    for channel in range(acf.shape[0]):
        lags = peaks[channel]
        lags = lags[(lags > 0) & (lags < acf.shape[1])]
        values = acf[channel, 1:]
        if len(lags):
            selected = acf[channel, lags]
            descriptor[channel, 0] = selected[0]
            descriptor[channel, 1] = np.median(np.abs(selected))
            descriptor[channel, 4] = float(lags[0]) / float(acf.shape[1] - 1)
            descriptor[channel, 5] = float(len(lags)) / float(peaks.shape[1])
        descriptor[channel, 2] = np.mean(np.abs(values))
        descriptor[channel, 3] = np.max(np.abs(values))
    if not np.isfinite(descriptor).all():
        raise ValueError("ACF descriptor must be finite")
    return ((descriptor - descriptor.mean(axis=0, keepdims=True)) /
            np.maximum(descriptor.std(axis=0, keepdims=True), 1e-6)).astype(np.float32)
