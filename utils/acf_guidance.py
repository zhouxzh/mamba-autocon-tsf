"""Leakage-safe autocorrelation priors for time-series forecasting.

The functions in this module deliberately separate *fitting* a prior from
using it at forecast time.  ``fit_signed_acf_prior`` must receive the training
split only.  The resulting per-channel, signed ACF peaks can be used either
as a compact seasonal configuration for a Mamba model or to construct causal
lag features for its prediction head.

No values after the end of ``history`` are read by :func:`causal_lag_features`.
Unavailable lags are filled with ``fill_value`` and reported by a boolean
availability mask, so callers can learn how much of a seasonal memory is
actually present instead of silently treating padding as an observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # Torch is available in the repository, but keep NumPy utilities usable alone.
    import torch
except ImportError:  # pragma: no cover - exercised only outside the project env.
    torch = None


ArrayLike = Union[np.ndarray, Sequence[float]]


def _strict_positive_integer(value, label: str) -> int:
    """Parse a positive integer without accepting bools or truncated floats."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a positive integer".format(label))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError("{} must be a positive integer".format(label))
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a positive integer".format(label)) from exc
    if not isinstance(value, (str, bytes)) and parsed != value:
        raise ValueError("{} must be a positive integer".format(label))
    if parsed <= 0:
        raise ValueError("{} must be a positive integer".format(label))
    return parsed


def _strict_nonnegative_integer(value, label: str) -> int:
    """Parse a non-negative integer without accepting bools or truncation."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a non-negative integer".format(label))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError("{} must be a non-negative integer".format(label))
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a non-negative integer".format(label)) from exc
    if not isinstance(value, (str, bytes)) and parsed != value:
        raise ValueError("{} must be a non-negative integer".format(label))
    if parsed < 0:
        raise ValueError("{} must be a non-negative integer".format(label))
    return parsed


def _strict_finite_float(value, label: str) -> float:
    """Parse a finite numeric option without treating bools as numbers."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a finite number".format(label))
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite number".format(label)) from exc
    if not np.isfinite(parsed):
        raise ValueError("{} must be a finite number".format(label))
    return parsed


def _contains_bool_scalar(value) -> bool:
    """Detect bool scalars before NumPy can coerce mixed lists to integers."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_bool_scalar(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype == object:
            return any(_contains_bool_scalar(item) for item in value.flat)
        return False
    if torch is not None and torch.is_tensor(value):
        return value.dtype == torch.bool
    return False


def compile_channel_lag_support(
    acf_values,
    lags,
    *,
    seq_len: int,
    horizon: int,
    top_k: int,
):
    """Compile the deterministic per-channel forecast-support mask.

    This NumPy reference compiler is shared by sidecar metadata and phase
    validation.  The model-side torch implementation uses the same float32
    formula and stable canonical-lag ordering; callers can compare its mask
    byte-for-byte with the returned ``uint8`` array.

    Returns ``(support, legal_step_fractions, canonical_lags)`` where support
    has shape ``[channels, lag_count]`` and exactly ``top_k`` entries per row.
    """

    seq_len = _strict_positive_integer(seq_len, "seq_len")
    horizon = _strict_positive_integer(horizon, "horizon")
    top_k = _strict_positive_integer(top_k, "top_k")
    compact = np.asarray(acf_values, dtype=np.float32)
    if compact.ndim != 2 or compact.shape[0] < 1 or compact.shape[1] < 1:
        raise ValueError("acf_values must have shape [channels, lag_count]")
    if not np.all(np.isfinite(compact)):
        raise ValueError("acf_values must be finite")
    raw_lags = np.asarray(lags).reshape(-1)
    if raw_lags.size != compact.shape[1]:
        raise ValueError("lags and acf_values width must match")
    parsed_lags = []
    for value in raw_lags.tolist():
        parsed_lags.append(_strict_positive_integer(value, "lags"))
    if len(set(parsed_lags)) != len(parsed_lags):
        raise ValueError("lags must be unique")
    order = np.argsort(np.asarray(parsed_lags, dtype=np.int64), kind="stable")
    canonical_lags = tuple(int(parsed_lags[index]) for index in order.tolist())
    if not np.array_equal(order, np.arange(len(parsed_lags))):
        compact = compact[:, order]

    steps = np.arange(1, horizon + 1, dtype=np.int64)[:, None]
    lag_array = np.asarray(canonical_lags, dtype=np.int64)[None, :]
    indices = seq_len - 1 + steps - lag_array
    legal_fraction = (
        ((indices >= 0) & (indices < seq_len))
        .astype(np.float32)
        .mean(axis=0, dtype=np.float32)
    )
    legal_slots = legal_fraction > 0.0
    if int(np.sum(legal_slots)) < top_k:
        raise ValueError(
            "requires at least {} lag slots with legal forecast support; "
            "resolved bank has {}".format(top_k, int(np.sum(legal_slots)))
        )
    scores = np.abs(compact) * legal_fraction[None, :]
    scores[:, ~legal_slots] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")
    support = np.zeros(compact.shape, dtype=np.uint8)
    rows = np.arange(compact.shape[0])[:, None]
    support[rows, order[:, :top_k]] = 1
    return support, legal_fraction, canonical_lags


@dataclass(frozen=True)
class ACFPeak:
    """A signed local ACF peak for one channel.

    ``value`` retains the sign of the correlation.  Peaks are ranked by
    ``abs(value)`` because both positive phase repetition and anti-phase
    repetition can be useful, while downstream code can still treat the two
    cases differently.
    """

    lag: int
    value: float

    @property
    def magnitude(self) -> float:
        """Return the absolute ACF strength at this lag."""

        return abs(self.value)


@dataclass(frozen=True)
class SignedACFPrior:
    """Training-only signed ACF and its selected peaks.

    Attributes
    ----------
    acf:
        Array with shape ``[channels, max_lag + 1]``.  Column zero is one for
        every non-degenerate channel.
    peaks:
        Tuple of per-channel tuples of :class:`ACFPeak`, ranked by magnitude.
    train_length:
        Number of observations used to estimate the prior.  It is metadata,
        not a source of values at inference time.
    """

    acf: np.ndarray
    peaks: Tuple[Tuple[ACFPeak, ...], ...]
    train_length: int

    @property
    def n_channels(self) -> int:
        """Number of channels represented by the prior."""

        return int(self.acf.shape[0])

    @property
    def max_lag(self) -> int:
        """Largest lag estimated by the prior."""

        return int(self.acf.shape[1] - 1)

    def channel_lags(self, channel: int, top_k: Optional[int] = None) -> Tuple[int, ...]:
        """Return selected positive lags for ``channel`` in rank order."""

        if channel < 0 or channel >= self.n_channels:
            raise IndexError("channel is outside the fitted ACF prior")
        selected = self.peaks[channel]
        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k must be non-negative")
            selected = selected[:top_k]
        return tuple(peak.lag for peak in selected)

    def channel_weights(self, channel: int, top_k: Optional[int] = None) -> Tuple[float, ...]:
        """Return signed ACF weights aligned with :meth:`channel_lags`."""

        if channel < 0 or channel >= self.n_channels:
            raise IndexError("channel is outside the fitted ACF prior")
        selected = self.peaks[channel]
        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k must be non-negative")
            selected = selected[:top_k]
        return tuple(float(peak.value) for peak in selected)

    def common_lags(self, top_k: Optional[int] = None) -> Tuple[int, ...]:
        """Return the strongest deterministic union of per-channel peak lags.

        This is useful for a shared Mamba head.  Per-channel models should use
        :meth:`channel_lags` instead, because a union can contain irrelevant
        lags for an individual channel.  Lags are ranked by their strongest
        absolute ACF value across channels, with the lag as a stable tie-break.
        """

        strengths = {}
        for channel_peaks in self.peaks:
            for peak in channel_peaks:
                strengths[peak.lag] = max(strengths.get(peak.lag, 0.0), peak.magnitude)
        lags = sorted(strengths, key=lambda lag: (-strengths[lag], lag))
        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k must be non-negative")
            lags = lags[:top_k]
        return tuple(lags)

    def availability_weighted_lags(
        self,
        *,
        history_length: int,
        horizon: int,
        top_k: int,
        min_separation: int = 1,
    ) -> Tuple[int, ...]:
        """Select train-only ACF peaks that can serve a direct forecast memory.

        A large autocorrelation at a lag that is unavailable for nearly every
        forecast step is not necessarily useful to a causal seasonal-memory
        residual.  This selector ranks the union of already fitted per-channel
        peaks by

        ``mean_channel_abs_acf(lag) * legal_forecast_step_fraction(lag)``.

        The availability factor depends only on the fixed input/history and
        forecast lengths, never on validation or test values.  Candidate lags
        remain restricted to the statistically selected training peaks, so the
        method does not turn trivial short lags into a new unconstrained search
        space.  It is intended for a forecasting-memory configuration; a
        relation objective may instead want the strongest temporal peaks even
        when they are sparse over a short horizon.
        """

        history_length = int(history_length)
        horizon = int(horizon)
        top_k = int(top_k)
        min_separation = int(min_separation)
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if min_separation < 1:
            raise ValueError("min_separation must be positive")
        if top_k == 0:
            return tuple()

        candidates = sorted({peak.lag for channel in self.peaks for peak in channel})
        if not candidates:
            return tuple()
        steps = np.arange(1, horizon + 1, dtype=np.int64)
        ranked = []
        for lag in candidates:
            if lag <= 0 or lag > self.max_lag:
                continue
            indices = history_length - 1 + steps - int(lag)
            availability = float(np.mean((indices >= 0) & (indices < history_length)))
            # A statistically strong lag with zero legal forecast positions
            # cannot contribute to the causal memory.  Do not spend one of
            # the limited lag slots on it; callers can fall back to their
            # configured lags when no train peak is usable.
            if availability <= 0.0:
                continue
            strength = float(np.mean(np.abs(self.acf[:, int(lag)])))
            ranked.append((strength * availability, strength, int(lag)))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = []
        for _, _, lag in ranked:
            if any(abs(lag - kept) < min_separation for kept in selected):
                continue
            selected.append(lag)
            if len(selected) >= top_k:
                break
        return tuple(selected)

    def relation_targets(
        self,
        origins: ArrayLike,
        *,
        reduction: str = "none",
        out_of_range: str = "zero",
        as_torch: bool = False,
        device=None,
    ):
        """Construct signed ACF relations between training-window origins.

        The returned matrix is suitable as an all-pair target for a relational
        contrastive objective.  With ``reduction='none'`` its shape is
        ``[batch, batch, channels]``; ``'mean'`` returns ``[batch, batch]``.
        ``out_of_range='zero'`` prevents an unmeasured distant pair from being
        assigned the last available ACF value.
        """

        return build_acf_relation_targets(
            origins,
            self.acf,
            reduction=reduction,
            out_of_range=out_of_range,
            as_torch=as_torch,
            device=device,
        )


def _as_time_channel_array(values: ArrayLike) -> np.ndarray:
    """Validate and normalize a time-major array to ``[time, channels]``."""

    if torch is not None and torch.is_tensor(values):
        # ACF fitting is an offline operation.  Moving a tensor to CPU here
        # keeps the fitted prior independent of the training device.
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("values must have shape [time] or [time, channels]")
    if array.shape[0] < 2:
        raise ValueError("at least two training observations are required")
    if not np.isfinite(array).all():
        raise ValueError("values contain NaN or infinite entries")
    return array


def _prepare_signal(values: np.ndarray, detrend: str) -> np.ndarray:
    """Apply an explicitly selected training-only centering/detrending step."""

    if detrend not in {"none", "mean", "linear"}:
        raise ValueError("detrend must be one of: none, mean, linear")
    signal = values.astype(np.float64, copy=True)
    if detrend == "mean":
        signal -= signal.mean(axis=0, keepdims=True)
    elif detrend == "linear":
        x = np.arange(signal.shape[0], dtype=np.float64)
        design = np.stack([x, np.ones_like(x)], axis=1)
        coefficients, _, _, _ = np.linalg.lstsq(design, signal, rcond=None)
        signal -= design @ coefficients
    return signal


def _fft_acf(
    values: np.ndarray,
    max_lag: int,
    *,
    channel_chunk_size: Optional[int] = 64,
) -> np.ndarray:
    """Compute biased, normalized (mean-centered) ACF with one FFT.

    Mean centering is part of the ACF definition, so it is applied even when
    ``detrend='none'``.  That option means that no additional linear trend
    removal is requested; it must not make a constant level look like perfect
    serial dependence.
    """

    values = values - values.mean(axis=0, keepdims=True)
    if channel_chunk_size is None:
        channel_chunk_size = values.shape[1]
    if not isinstance(channel_chunk_size, (int, np.integer)) or channel_chunk_size < 1:
        raise ValueError("channel_chunk_size must be a positive integer or None")

    n = values.shape[0]
    n_fft = 1 << (2 * n - 1).bit_length()
    # Work in channel chunks so high-dimensional datasets do not need a
    # second full-size complex FFT buffer for every channel simultaneously.
    acf = np.zeros((max_lag + 1, values.shape[1]), dtype=np.float64)
    for start in range(0, values.shape[1], int(channel_chunk_size)):
        stop = min(values.shape[1], start + int(channel_chunk_size))
        chunk = values[:, start:stop]
        spectrum = np.fft.rfft(chunk, n=n_fft, axis=0)
        correlation = np.fft.irfft(
            spectrum * np.conjugate(spectrum), n=n_fft, axis=0
        )[: max_lag + 1]
        variance = correlation[0].copy()
        non_degenerate = variance > np.finfo(np.float64).eps
        chunk_acf = acf[:, start:stop]
        chunk_acf[:, non_degenerate] = correlation[:, non_degenerate] / variance[non_degenerate]
        # A constant channel has no identifiable peaks; retain the conventional
        # lag-zero value and leave non-zero lags at zero.
        acf[0, start:stop] = 1.0
    return np.clip(acf.T, -1.0, 1.0)


def _select_peaks(
    acf: np.ndarray,
    *,
    sample_size: int,
    top_k: int,
    min_lag: int,
    min_abs_acf: float,
    confidence_z: Optional[float],
    min_separation: int,
) -> Tuple[ACFPeak, ...]:
    """Select separated local maxima while preserving ACF signs."""

    if top_k == 0:
        return tuple()
    # The approximate white-noise threshold depends on the number of training
    # observations, not on the number of lags retained in the prior.
    n = max(1, int(sample_size))
    threshold_floor = float(min_abs_acf)
    candidates: List[Tuple[float, int]] = []
    for lag in range(max(1, min_lag), acf.shape[0]):
        value = float(acf[lag])
        threshold = threshold_floor
        if confidence_z is not None:
            threshold = max(threshold, float(confidence_z) / np.sqrt(max(1, n)))
        if abs(value) < threshold:
            continue
        left = abs(float(acf[lag - 1])) if lag > 0 else -np.inf
        right = abs(float(acf[lag + 1])) if lag + 1 < acf.shape[0] else -np.inf
        if abs(value) >= left and abs(value) >= right:
            candidates.append((abs(value), lag))

    # A monotonic ACF can have no strict interior local maxima.  Retain the
    # strongest measured lags in that case so the prior remains usable.
    if not candidates:
        threshold = threshold_floor
        if confidence_z is not None:
            threshold = max(threshold, float(confidence_z) / np.sqrt(max(1, n)))
        candidates = [
            (abs(float(acf[lag])), lag)
            for lag in range(max(1, min_lag), acf.shape[0])
            if abs(float(acf[lag])) >= threshold
        ]

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: List[ACFPeak] = []
    for _, lag in candidates:
        if any(abs(lag - peak.lag) < min_separation for peak in selected):
            continue
        selected.append(ACFPeak(lag=lag, value=float(acf[lag])))
        if len(selected) >= top_k:
            break
    return tuple(selected)


def fit_signed_acf_prior(
    train_values: ArrayLike,
    max_lag: int,
    *,
    top_k: int = 8,
    min_lag: int = 1,
    min_abs_acf: float = 0.1,
    confidence_z: Optional[float] = 1.96,
    min_separation: int = 1,
    detrend: str = "mean",
    channel_chunk_size: Optional[int] = 64,
) -> SignedACFPrior:
    """Fit a per-channel signed ACF prior using the supplied train split only.

    Parameters
    ----------
    train_values:
        Time-major training observations, shape ``[time]`` or
        ``[time, channels]``.  The function does not inspect any validation or
        test data; callers should pass the split boundary explicitly.
    max_lag:
        Largest lag to estimate.  It must be smaller than the train length.
    top_k, min_lag, min_abs_acf:
        Controls for compact peak selection.  Peaks are ranked by magnitude,
        but their signed values are retained.
    confidence_z:
        Optional approximate white-noise threshold ``z/sqrt(n)``.  Set to
        ``None`` to disable it (useful for very short synthetic tests).
    min_separation:
        Minimum distance between selected peak lags.
    detrend:
        ``'mean'`` (default), ``'linear'``, or ``'none'``.
    channel_chunk_size:
        Number of channels processed per FFT chunk.  Smaller values reduce
        peak memory on high-dimensional datasets; ``None`` processes all
        channels together.

    Returns
    -------
    SignedACFPrior
        Immutable metadata plus an array of shape ``[channels, max_lag + 1]``.
    """

    values = _as_time_channel_array(train_values)
    max_lag = _strict_positive_integer(max_lag, "max_lag")
    if max_lag >= values.shape[0]:
        raise ValueError("max_lag must be smaller than the training length")
    top_k = _strict_nonnegative_integer(top_k, "top_k")
    min_lag = _strict_positive_integer(min_lag, "min_lag")
    if min_lag > max_lag:
        raise ValueError("min_lag must be in [1, max_lag]")
    min_abs_acf = _strict_finite_float(min_abs_acf, "min_abs_acf")
    if min_abs_acf < 0 or min_abs_acf > 1:
        raise ValueError("min_abs_acf must be in [0, 1]")
    if confidence_z is not None:
        confidence_z = _strict_finite_float(confidence_z, "confidence_z")
        if confidence_z < 0:
            raise ValueError("confidence_z must be non-negative or None")
    min_separation = _strict_positive_integer(min_separation, "min_separation")
    if channel_chunk_size is not None:
        channel_chunk_size = _strict_positive_integer(
            channel_chunk_size, "channel_chunk_size"
        )

    signal = _prepare_signal(values, detrend)
    acf = _fft_acf(signal, max_lag, channel_chunk_size=channel_chunk_size)
    peaks = tuple(
        _select_peaks(
            acf[channel],
            sample_size=values.shape[0],
            top_k=top_k,
            min_lag=min_lag,
            min_abs_acf=min_abs_acf,
            confidence_z=confidence_z,
            min_separation=min_separation,
        )
        for channel in range(acf.shape[0])
    )
    acf.setflags(write=False)
    return SignedACFPrior(acf=acf, peaks=peaks, train_length=int(values.shape[0]))


def _normalize_lags(lags, channels: int) -> np.ndarray:
    """Normalize global or per-channel lags to ``[channels, K]``."""

    if _contains_bool_scalar(lags):
        raise ValueError("lags must be integer-valued")
    array = np.asarray(lags)
    if array.ndim == 1:
        if array.size == 0:
            raise ValueError("lags must contain at least one lag")
        array = np.broadcast_to(array[None, :], (channels, array.size)).copy()
    elif array.ndim == 2:
        if array.shape[0] != channels or array.shape[1] == 0:
            raise ValueError("per-channel lags must have shape [channels, K]")
        array = array.copy()
    else:
        raise ValueError("lags must have shape [K] or [channels, K]")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.equal(array, np.floor(array)).all():
            raise ValueError("lags must be integer-valued")
        array = array.astype(np.int64)
    else:
        array = array.astype(np.int64, copy=False)
    # -1 is reserved as an explicit per-channel placeholder for an unused slot.
    if (array[array != -1] <= 0).any():
        raise ValueError("lags must be positive; only -1 may mark an unused slot")
    return array


def causal_lag_features(
    history,
    lags,
    horizon: Optional[int] = None,
    *,
    weights=None,
    fill_value: float = 0.0,
    return_mask: bool = True,
):
    """Build causal lag features for a history or a forecast horizon.

    ``history`` accepts NumPy or Torch arrays with shape ``[time, channels]``
    or ``[batch, time, channels]``.  If ``horizon`` is ``None``, features are
    aligned to every historical timestamp and only observations strictly
    earlier than that timestamp are eligible.  If ``horizon=H``, features are
    aligned to the next ``H`` timestamps after the history and use index
    ``history_length + h - lag``.  Thus a one-step, lag-one feature is exactly
    the last observed value.

    ``lags`` may be global ``[K]`` or channel-specific ``[channels, K]``; use
    ``-1`` in the latter form for an unused slot.  The output has shape
    ``[batch, output_length, channels, K]`` (or the corresponding unbatched
    shape).  A boolean mask with the same shape marks values backed by actual
    history.  Missing entries are filled with ``fill_value`` and never read
    from future targets.
    """

    is_torch = torch is not None and torch.is_tensor(history)
    if is_torch:
        if history.ndim == 2:
            source = history.unsqueeze(0)
            unbatched = True
        elif history.ndim == 3:
            source = history
            unbatched = False
        else:
            raise ValueError("history must have shape [time, channels] or [batch, time, channels]")
        batch, length, channels = source.shape
        if length < 1:
            raise ValueError("history must contain at least one observation")
        lag_matrix = _normalize_lags(lags, int(channels))
        out_length = int(length if horizon is None else horizon)
        if out_length < 1:
            raise ValueError("horizon must be positive when supplied")
        target = torch.arange(out_length, device=source.device, dtype=torch.long)
        if horizon is not None:
            target = target + int(length)
        lag_tensor = torch.as_tensor(lag_matrix, device=source.device, dtype=torch.long)
        indices = target[:, None, None] - lag_tensor[None, :, :]
        valid = (indices >= 0) & (indices < length) & (lag_tensor[None, :, :] != -1)
        safe_indices = indices.clamp(0, max(0, length - 1))
        gather_index = safe_indices.unsqueeze(0).expand(batch, -1, -1, -1)
        expanded = source.unsqueeze(-1).expand(-1, -1, -1, lag_matrix.shape[1])
        output = torch.gather(expanded, 1, gather_index)
        output = output.masked_fill(~valid.unsqueeze(0), fill_value)
        if weights is not None:
            weight = torch.as_tensor(weights, device=source.device, dtype=output.dtype)
            if weight.ndim == 1:
                if weight.numel() != lag_matrix.shape[1]:
                    raise ValueError("weights must align with the lag dimension")
                weight = weight.view(1, 1, 1, -1)
            elif weight.ndim == 2:
                if tuple(weight.shape) != tuple(lag_matrix.shape):
                    raise ValueError("per-channel weights must match lags shape")
                weight = weight.view(1, 1, channels, -1)
            else:
                raise ValueError("weights must have shape [K] or [channels, K]")
            output = output * weight
            # Keep the documented sentinel for unavailable memory.  Applying
            # a learned/signed weight to a padding value would otherwise turn
            # ``fill_value`` into a misleading observation.
            output = output.masked_fill(~valid.unsqueeze(0), fill_value)
        mask = valid.unsqueeze(0).expand(batch, -1, -1, -1)
        if unbatched:
            output, mask = output.squeeze(0), mask.squeeze(0)
        return (output, mask) if return_mask else output

    array = np.asarray(history)
    if array.ndim == 2:
        source = array[None, ...]
        unbatched = True
    elif array.ndim == 3:
        source = array
        unbatched = False
    else:
        raise ValueError("history must have shape [time, channels] or [batch, time, channels]")
    batch, length, channels = source.shape
    if length < 1:
        raise ValueError("history must contain at least one observation")
    lag_matrix = _normalize_lags(lags, int(channels))
    out_length = int(length if horizon is None else horizon)
    if out_length < 1:
        raise ValueError("horizon must be positive when supplied")
    target = np.arange(out_length, dtype=np.int64)
    if horizon is not None:
        target = target + int(length)
    indices = target[:, None, None] - lag_matrix[None, :, :]
    valid = (indices >= 0) & (indices < length) & (lag_matrix[None, :, :] != -1)
    safe_indices = np.clip(indices, 0, max(0, length - 1))
    expanded = np.broadcast_to(source[:, :, :, None], (batch, length, channels, lag_matrix.shape[1]))
    gather_index = np.broadcast_to(safe_indices[None, ...], (batch, out_length, channels, lag_matrix.shape[1]))
    output = np.take_along_axis(expanded, gather_index, axis=1)
    output = np.where(valid[None, ...], output, fill_value)
    if weights is not None:
        weight = np.asarray(weights, dtype=output.dtype)
        if weight.ndim == 1:
            if weight.size != lag_matrix.shape[1]:
                raise ValueError("weights must align with the lag dimension")
            output = output * weight[None, None, None, :]
        elif weight.ndim == 2:
            if tuple(weight.shape) != tuple(lag_matrix.shape):
                raise ValueError("per-channel weights must match lags shape")
            output = output * weight[None, None, :, :]
        else:
            raise ValueError("weights must have shape [K] or [channels, K]")
        output = np.where(valid[None, ...], output, fill_value)
    mask = np.broadcast_to(valid[None, ...], (batch, out_length, channels, lag_matrix.shape[1])).copy()
    if unbatched:
        output, mask = output.squeeze(0), mask.squeeze(0)
    return (output, mask) if return_mask else output


def build_acf_relation_targets(
    origins: ArrayLike,
    acf_values: ArrayLike,
    *,
    reduction: str = "none",
    out_of_range: str = "zero",
    as_torch: bool = False,
    device=None,
):
    """Return signed per-channel ACF relations for all pairs of origins.

    ``origins`` are integer positions of windows in the training timeline.
    For a pair ``(i, j)``, the target is the ACF at
    ``abs(origins[i] - origins[j])``.  ``acf_values`` accepts ``[lag]`` or
    ``[channels, lag]`` and therefore preserves heterogeneous channel
    periodicity when ``reduction='none'``.  ``reduction='mean'`` averages
    channels only after lookup.  Set ``as_torch=True`` to obtain a tensor on
    ``device`` for direct use in a PyTorch loss.
    """

    if _contains_bool_scalar(origins):
        raise ValueError("origins must be integer-valued")
    origin_array = np.asarray(origins)
    if origin_array.ndim != 1:
        raise ValueError("origins must be a one-dimensional sequence")
    if not np.issubdtype(origin_array.dtype, np.integer):
        if not np.equal(origin_array, np.floor(origin_array)).all():
            raise ValueError("origins must be integer-valued")
        origin_array = origin_array.astype(np.int64)
    else:
        origin_array = origin_array.astype(np.int64, copy=False)
    if (origin_array < 0).any():
        raise ValueError("origins must be non-negative")

    acf_array = np.asarray(acf_values, dtype=np.float64)
    if acf_array.ndim == 1:
        acf_array = acf_array[None, :]
    if acf_array.ndim != 2 or acf_array.shape[1] < 1:
        raise ValueError("acf_values must have shape [lag] or [channels, lag]")
    if not np.isfinite(acf_array).all():
        raise ValueError("acf_values contain NaN or infinite entries")
    if reduction not in {"none", "mean"}:
        raise ValueError("reduction must be 'none' or 'mean'")
    if out_of_range not in {"zero", "clip"}:
        raise ValueError("out_of_range must be 'zero' or 'clip'")

    distances = np.abs(origin_array[:, None] - origin_array[None, :])
    max_lag = acf_array.shape[1] - 1
    valid = distances <= max_lag
    lookup = np.minimum(distances, max_lag)
    relations = acf_array[:, lookup].transpose(1, 2, 0)
    if out_of_range == "zero":
        relations = np.where(valid[..., None], relations, 0.0)
    if reduction == "mean":
        relations = relations.mean(axis=-1)
    relations = relations.astype(np.float32, copy=False)
    if as_torch:
        if torch is None:
            raise ImportError("PyTorch is required when as_torch=True")
        return torch.as_tensor(relations, device=device)
    return relations


__all__ = [
    "ACFPeak",
    "SignedACFPrior",
    "fit_signed_acf_prior",
    "causal_lag_features",
    "build_acf_relation_targets",
]
