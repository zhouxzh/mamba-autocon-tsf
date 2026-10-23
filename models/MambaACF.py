"""ACF-guided Mamba forecasters for long-term time-series prediction.

This module is deliberately separate from the historical ``Mamba*AutoCon``
models.  Those models replace only the encoder in AutoConNet.  The default
model here uses one shared temporal state-space encoder for all channels,
optional multi-resolution and bidirectional Mamba blocks, a direct
multi-horizon prediction head, and a causal seasonal-memory residual driven by
signed window ACF values.  A channel-independent encoder is retained as an
explicit ablation, but is not the high-dimensional default because it repeats
the expensive Mamba scan once per variable.

The model returns ``(prediction, representation)`` when ``configs.AutoCon``
is enabled, which keeps the existing AutoCon trainer contract.  The default
relation view is an encoder projection; an opt-in forecast relation view uses
the final normalized forecast for the global ACFRelationLoss.  Its global
objective pools forecast steps, then sends a window-level relation gradient
directly through the forecasting branches without changing prediction-only
inference.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding
from layers.CausalEmbedding import CausalDataEmbedding
from layers.LowRankChannelGate import LowRankChannelGate
from layers.LowRankDataDependentChannelMixer import (
    LowRankDataDependentChannelMixer,
    LowRankGlobalChannelStateMixer,
)
from utils.acf_relation_contract import resolve_acf_source
from utils.acf_channel_order import signed_acf_profile_similarity_order
from utils.mamba_causality_contract import is_full_window_normalization


try:
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA1_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional CUDA dependency
    # Keep Mamba-1 availability independent from Mamba-2.  Some mamba_ssm
    # wheels expose only one API, and a missing Mamba2 must not disable the
    # otherwise usable Mamba-1 branch.
    _Mamba1 = None
    _MAMBA1_IMPORT_ERROR = exc

try:
    from mamba_ssm import Mamba2 as _Mamba2
    _MAMBA2_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional CUDA dependency
    # Likewise preserve Mamba-2 when an older wheel has no Mamba-1 symbol.
    _Mamba2 = None
    _MAMBA2_IMPORT_ERROR = exc


def _compatible_headdim(d_model: int, expand: int, requested: int) -> int:
    """Resolve a Mamba-2 head dimension for small experiment widths."""

    expanded = int(d_model) * int(expand)
    requested = int(requested)
    if expanded <= 0 or requested <= 0:
        raise ValueError("mamba dimensions must be positive")
    if expanded % requested == 0:
        return requested
    divisors = [value for value in range(min(expanded, requested), 0, -1)
                if expanded % value == 0]
    resolved = divisors[0]
    warnings.warn(
        "mamba_headdim={} is incompatible with expanded dimension {}; using "
        "headdim={} instead".format(requested, expanded, resolved),
        UserWarning,
    )
    return resolved


def _artifact_lags(payload_lags):
    """Parse NPZ lag metadata without silently truncating floating values."""

    raw_lags = np.asarray(payload_lags)
    if raw_lags.dtype.kind not in {"i", "u"}:
        raise ValueError("ACF prior lags must use an integer dtype")
    if raw_lags.dtype.kind == "u" and np.any(
        raw_lags > np.iinfo(np.int64).max
    ):
        raise ValueError("ACF prior lags exceed the supported integer range")
    return raw_lags.astype(np.int64, copy=False).reshape(-1).tolist()


def _strict_positive_integer(value, label):
    """Parse a positive integer without silently truncating configuration."""

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


def _normalize_acf_values(values, label):
    """Validate finite ACF inputs and preserve the historical normalization."""

    if not torch.isfinite(values).all():
        raise ValueError("{} must be finite".format(label))
    return values.clamp(-1.0, 1.0)


def _validate_acf_artifact_values(values, label):
    """Reject malformed sidecar correlations before attaching them."""

    if not torch.isfinite(values).all():
        raise ValueError("{} must be finite".format(label))
    if torch.any(values.abs() > 1.0 + 1e-6):
        raise ValueError("{} must be within [-1, 1]".format(label))


def _artifact_scalar_positive_integer(value, label):
    """Read a scalar NPZ integer field without truncating fractional values."""

    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError("{} must be a scalar positive integer".format(label))
    return _strict_positive_integer(raw.item(), label)


def _parse_positive_lags(values, label="lags", *, allow_empty=False):
    """Return positive, unique lag values while preserving caller order."""

    if isinstance(values, (str, bytes)):
        text = values.decode() if isinstance(values, bytes) else values
        values = [item for item in text.replace(",", " ").split() if item]
    elif torch.is_tensor(values):
        values = values.detach().cpu().reshape(-1).tolist()
    elif isinstance(values, np.ndarray):
        values = values.reshape(-1).tolist()
    else:
        try:
            values = list(values)
        except TypeError as exc:
            raise ValueError("{} must be a non-empty sequence".format(label)) from exc
    if not values:
        if allow_empty:
            return tuple()
        raise ValueError("{} must contain at least one positive value".format(label))
    parsed = tuple(
        _strict_positive_integer(value, "{} item".format(label)) for value in values
    )
    if len(set(parsed)) != len(parsed):
        raise ValueError("{} must contain unique values".format(label))
    return parsed


def _canonical_positive_lags(values, label="lags", *, allow_empty=False):
    """Return the canonical ascending lag tuple used by model metadata."""

    return tuple(sorted(_parse_positive_lags(values, label, allow_empty=allow_empty)))


def _validate_loaded_acf_prior_alignment(
    acf_values, full_acf_values, lags, channels
):
    """Reject a sidecar whose compact and complete ACF tables disagree.

    A complete table is retained for the AutoCon relation objective, whereas
    the compact table drives forecast memory.  Both must describe the same
    selected lags; otherwise a tampered artifact could silently use one prior
    for forecasting and another for representation learning.  Match the
    one-dimensional expansion accepted by ``set_compact_acf_prior`` so valid
    legacy sidecars keep their existing loading behavior.
    """

    compact = np.asarray(acf_values, dtype=np.float32)
    if compact.ndim == 1:
        compact = np.broadcast_to(compact, (int(channels), compact.shape[0]))
    if compact.ndim != 2 or compact.shape[0] != int(channels):
        raise ValueError("compact ACF prior must have shape [channels, selected_lags]")
    if compact.shape[1] == 0 or not np.isfinite(compact).all():
        raise ValueError("compact ACF prior must be non-empty and finite")

    complete = np.asarray(full_acf_values, dtype=np.float32)
    if complete.ndim == 1:
        complete = np.broadcast_to(complete, (int(channels), complete.shape[0]))
    if complete.ndim != 2 or complete.shape[0] != int(channels):
        raise ValueError("full ACF prior must have shape [channels, measured_lags]")
    if not np.isfinite(complete).all():
        raise ValueError("full ACF prior must be finite")

    parsed_lags = tuple(int(value) for value in lags)
    if (
        not parsed_lags
        or any(value <= 0 for value in parsed_lags)
        or len(set(parsed_lags)) != len(parsed_lags)
    ):
        raise ValueError("compact ACF prior lags must be positive and unique")
    if compact.shape[1] != len(parsed_lags):
        raise ValueError(
            "compact ACF prior lag count ({}) does not match coefficient "
            "width ({})".format(len(parsed_lags), compact.shape[1])
        )
    if complete.shape[1] <= max(parsed_lags):
        raise ValueError("full ACF prior does not contain every selected lag")

    selected_complete = complete[:, np.asarray(parsed_lags, dtype=np.int64)]
    if not np.allclose(selected_complete, compact, rtol=0.0, atol=1e-6):
        raise ValueError(
            "full ACF prior selected values do not match compact ACF prior"
        )


class _MambaFFNBlock(nn.Module):
    """Pre-normalized Mamba residual block with the FFN used by TS variants."""

    def __init__(self, d_model, d_state, d_conv, expand, headdim, dropout, kind, d_ff):
        super().__init__()
        if kind == "mamba1":
            mixer_cls = _Mamba1
            import_error = _MAMBA1_IMPORT_ERROR
            kind_label = "Mamba-1"
        elif kind == "mamba2":
            mixer_cls = _Mamba2
            import_error = _MAMBA2_IMPORT_ERROR
            kind_label = "Mamba-2"
        else:
            raise ValueError("unknown Mamba kind: {}".format(kind))
        if mixer_cls is None:
            detail = " ({})".format(import_error) if import_error else ""
            raise ImportError(
                "MambaACF requires {} support from the optional 'mamba_ssm' "
                "package{}".format(kind_label, detail)
            )

        self.norm_mixer = nn.LayerNorm(d_model)
        kwargs = {
            "d_model": int(d_model),
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
        }
        if kind == "mamba2":
            kwargs["headdim"] = int(headdim)
        self.mixer = mixer_cls(**kwargs)
        self.norm_ffn = nn.LayerNorm(d_model)
        ffn_width = max(int(d_ff), int(d_model))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_width, d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        try:
            x = x + self.mixer(self.norm_mixer(x))
        except RuntimeError as exc:
            message = str(exc).lower()
            if not x.is_cuda and ("cuda" in message or "triton" in message):
                raise RuntimeError(
                    "The installed mamba_ssm build requires CUDA for MambaACF; "
                    "run with --use_gpu or install a CPU-capable implementation."
                ) from exc
            raise
        return x + self.ffn(self.norm_ffn(x))


class _TemporalProjection(nn.Module):
    """Linear map from the encoder window to the forecast horizon.

    The full-rank path intentionally mirrors ``nn.Linear`` so the default
    MambaACF state-dict layout remains unchanged.  The optional factorized
    path uses two linear maps without a nonlinear bottleneck; it is therefore
    a rank-constrained temporal projection rather than an extra MLP.
    """

    def __init__(self, input_length, output_length, rank=None):
        super().__init__()
        input_length = int(input_length)
        output_length = int(output_length)
        if input_length <= 0 or output_length <= 0:
            raise ValueError("temporal projection dimensions must be positive")
        if rank is None:
            self.rank = None
            self.proj = nn.Linear(input_length, output_length)
            return

        rank = int(rank)
        if rank <= 0:
            raise ValueError("mamba_temporal_rank must be positive")
        if rank >= min(input_length, output_length):
            raise ValueError(
                "mamba_temporal_rank={} must be smaller than min(input_length={}, "
                "pred_len={}) when --mamba_low_rank_head is enabled".format(
                    rank, input_length, output_length
                )
            )
        self.rank = rank
        self.input_projection = nn.Linear(input_length, rank, bias=False)
        self.output_projection = nn.Linear(rank, output_length)

    @property
    def is_low_rank(self):
        return self.rank is not None

    def forward(self, x):
        if self.is_low_rank:
            return self.output_projection(self.input_projection(x))
        return self.proj(x)


class _HorizonLagAmplitudeCalibration(nn.Module):
    """Low-rank, horizon-aware amplitude adapter for ACF memory.

    The adapter is deliberately initialized at zero.  Its multiplicative
    amplitude is therefore exactly one at construction time, making the
    calibrated model a strict nested version of the existing memory path.
    ``rank=1`` is a constant lag amplitude and ``rank=2`` adds a fixed linear
    horizon slope.  The basis is deterministic and does not inspect train,
    validation, or test targets.
    """

    def __init__(self, horizon, lag_count, rank=2):
        super().__init__()
        self.horizon = int(horizon)
        self.lag_count = int(lag_count)
        self.rank = int(rank)
        if self.horizon <= 0 or self.lag_count <= 0:
            raise ValueError("horizon and lag_count must be positive")
        if self.rank not in {1, 2}:
            raise ValueError(
                "mamba_acf_horizon_calibration_rank must be 1 or 2"
            )
        self.coefficients = nn.Parameter(
            torch.zeros(self.lag_count, self.rank)
        )
        self.register_buffer(
            "basis",
            self._make_basis(self.horizon, self.rank),
            persistent=False,
        )

    @staticmethod
    def _make_basis(horizon, rank, device=None, dtype=None):
        steps = torch.arange(
            1, int(horizon) + 1, device=device, dtype=dtype or torch.float32
        )
        columns = [torch.ones_like(steps)]
        if int(rank) == 2:
            columns.append(2.0 * steps / (float(horizon) + 1.0) - 1.0)
        return torch.stack(columns, dim=-1)

    def amplitudes(self, horizon=None, device=None, dtype=None):
        horizon = self.horizon if horizon is None else int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if horizon == self.horizon:
            basis = self.basis
        else:
            basis = self._make_basis(horizon, self.rank, device=self.coefficients.device)
        basis = basis.to(
            device=self.coefficients.device if device is None else device,
            dtype=self.coefficients.dtype,
        )
        coefficients = self.coefficients
        logits = torch.matmul(basis, coefficients.transpose(0, 1))
        # Bound the learned multiplicative correction to [1/2, 2].  This keeps
        # a small auxiliary adapter from overwhelming the signed ACF signal.
        logits = logits.clamp(-math.log(2.0), math.log(2.0))
        amplitudes = torch.exp(logits)
        if dtype is not None:
            amplitudes = amplitudes.to(dtype=dtype)
        return amplitudes

    @property
    def parameter_count(self):
        return int(self.coefficients.numel())


def _resize_horizon_lag_calibration(adapter, horizon, lags, device=None):
    """Resize a calibration adapter after train-only lag resolution.

    ``MambaACF`` is constructed from requested lag slots, but a train-peak
    sidecar can resolve a smaller (or differently ordered) compact lag set.
    The sidecar is attached before the optimizer is created, so replacing the
    tiny adapter here keeps the optimizer/state-dict shape aligned with the
    actual forecast memory.  Coefficients are copied by lag identity when a
    caller replaces a prior on an already initialized model.
    """

    parsed_lags = tuple(int(value) for value in lags)
    if not parsed_lags:
        raise ValueError("horizon calibration requires at least one lag")
    old_lags = tuple(getattr(adapter, "lag_values", ()))
    if not old_lags:
        old_lags = tuple(range(int(adapter.lag_count)))
    if adapter.lag_count == len(parsed_lags) and old_lags == parsed_lags:
        return adapter
    dtype = adapter.coefficients.dtype
    old_device = adapter.coefficients.device if device is None else device
    replacement = _HorizonLagAmplitudeCalibration(
        horizon=int(horizon), lag_count=len(parsed_lags), rank=int(adapter.rank)
    ).to(device=old_device, dtype=dtype)
    old_index = {int(lag): index for index, lag in enumerate(old_lags)}
    with torch.no_grad():
        for new_index, lag in enumerate(parsed_lags):
            source_index = old_index.get(int(lag))
            if source_index is not None and source_index < adapter.lag_count:
                replacement.coefficients[new_index].copy_(
                    adapter.coefficients[source_index]
                )
    replacement.lag_values = parsed_lags
    return replacement


class _LocalPatchBlock(nn.Module):
    """Lightweight token mixer used by the channel-preserving local branch.

    The block operates on ``[B, C, P, D]`` patch tokens with weights shared
    across variables.  It deliberately contains no channel-by-channel
    parameters and no cross-variable attention, so adding the branch does not
    turn a high-dimensional Traffic input into a dense ``C x C`` operation.
    """

    def __init__(self, width, d_ff, dropout):
        super().__init__()
        width = int(width)
        hidden = max(width, int(d_ff))
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, padding=1, groups=width, bias=False
        )
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, width),
            nn.Dropout(float(dropout)),
        )

    def forward(self, tokens):
        normalized = self.norm(tokens)
        mixed = self.depthwise(normalized.transpose(1, 2)).transpose(1, 2)
        return tokens + mixed + self.ffn(self.norm(tokens + mixed))


class _MambaACFBase(nn.Module):
    """Shared implementation; subclasses select Mamba-1 or Mamba-2."""

    mamba_kind = "mamba2"
    # Callers that only need the deployed forecast can disable the auxiliary
    # AutoCon representation grid and projection without changing defaults.
    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.label_len = int(configs.label_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        self.d_model = int(getattr(configs, "d_model", 64))
        self.d_ff = int(getattr(configs, "d_ff", self.d_model * 2))
        self.depth = max(1, int(getattr(configs, "e_layers", 2)))
        self.dropout = float(getattr(configs, "dropout", 0.1))
        self.AutoCon = bool(getattr(configs, "AutoCon", False))
        self.AutoCon_wnorm = getattr(configs, "AutoCon_wnorm", "LastVal")
        self.embed = getattr(configs, "embed", "timeF")
        self.freq = getattr(configs, "freq", "h")
        self.wo_freq = getattr(configs, "wo_freq", "")
        self.acf_relation_representation = str(
            getattr(configs, "mamba_acf_relation_representation", "encoder")
        ).strip().lower()
        if self.acf_relation_representation not in {"encoder", "forecast"}:
            raise ValueError(
                "mamba_acf_relation_representation must be encoder or forecast"
            )
        self.forecast_relation_representation_enabled = bool(
            self.acf_relation_representation == "forecast"
        )
        if self.forecast_relation_representation_enabled:
            if not self.AutoCon:
                raise ValueError(
                    "mamba_acf_relation_representation=forecast requires AutoCon=true"
                )
            if not bool(getattr(configs, "mamba_acf_relation_loss", False)):
                raise ValueError(
                    "mamba_acf_relation_representation=forecast requires "
                    "mamba_acf_relation_loss=true"
                )
            relation_scope = str(
                getattr(configs, "mamba_acf_relation_scope", "both")
            ).strip().lower()
            if relation_scope != "global":
                raise ValueError(
                    "mamba_acf_relation_representation=forecast requires "
                    "mamba_acf_relation_scope=global"
                )
            if resolve_acf_source(configs) != "raw":
                raise ValueError(
                    "mamba_acf_relation_representation=forecast requires "
                    "AutoCon_acf_source=raw"
                )
            if str(getattr(configs, "features", "M")).strip().upper() != "M":
                raise ValueError(
                    "mamba_acf_relation_representation=forecast requires features=M"
                )
            if self.c_out != self.channels:
                raise ValueError(
                    "mamba_acf_relation_representation=forecast requires "
                    "c_out=enc_in"
                )
            if self.channels < 2:
                raise ValueError(
                    "mamba_acf_relation_representation=forecast requires enc_in>=2"
                )
            if bool(getattr(configs, "mamba_acf_repr_fusion", False)) or bool(
                getattr(configs, "mamba_variable_repr_fusion", False)
            ):
                raise ValueError(
                    "mamba_acf_relation_representation=forecast is incompatible "
                    "with encoder-only representation fusion options"
                )

        requested_state = int(getattr(configs, "mamba_d_state", 16))
        requested_conv = int(getattr(configs, "mamba_d_conv", 4))
        requested_expand = int(getattr(configs, "mamba_expand", 2))
        requested_head = int(getattr(configs, "mamba_headdim", 16))
        self.mamba_d_state = requested_state
        self.mamba_d_conv = requested_conv
        self.mamba_expand = requested_expand
        self.mamba_headdim = (
            _compatible_headdim(self.d_model, requested_expand, requested_head)
            if self.mamba_kind == "mamba2" else None
        )

        scales = getattr(configs, "mamba_scales", [1, 2])
        if isinstance(scales, str):
            scales = [int(item) for item in scales.replace(",", " ").split() if item]
        self.scales = tuple(sorted({int(item) for item in scales if int(item) >= 1}))
        if not self.scales or self.scales[0] != 1:
            self.scales = (1,) + self.scales
        if any(scale > self.seq_len for scale in self.scales):
            raise ValueError(
                "mamba_scales must not exceed seq_len (received {} for seq_len={})".format(
                    list(self.scales), self.seq_len
                )
            )
        # Mamba kernels have a noticeable fixed launch cost at short lookbacks.
        # This optional causal patch stride reduces the sequence presented to
        # the temporal SSM before the existing multi-scale branches run.  It
        # is deliberately separate from ``mamba_scales``: stride changes the
        # base tokenization, while scales still define the relative branches.
        # The default ``1`` preserves the historical full-resolution path.
        self.mamba_token_stride = int(
            getattr(configs, "mamba_token_stride", 1)
        )
        if self.mamba_token_stride <= 0:
            raise ValueError("mamba_token_stride must be positive")
        if self.mamba_token_stride > self.seq_len:
            raise ValueError(
                "mamba_token_stride={} cannot exceed seq_len={}".format(
                    self.mamba_token_stride, self.seq_len
                )
            )
        self.base_token_count = int(
            math.ceil(self.seq_len / self.mamba_token_stride)
        )
        if any(scale > self.base_token_count for scale in self.scales):
            raise ValueError(
                "mamba_scales must not exceed the {} base tokens produced by "
                "mamba_token_stride={} (received {})".format(
                    self.base_token_count,
                    self.mamba_token_stride,
                    list(self.scales),
                )
            )
        # Optional train-only ACF conditioning for the multi-scale fusion.
        # The attached prior defines which lags are trusted; the current
        # observed window supplies signed ACF evidence for each scale decision.
        # This keeps the branch input-dependent instead of adding a dataset-
        # constant bias that the ordinary scale gate could absorb.
        # A zero-initialized scalar keeps the option exactly nested at startup.
        self.acf_scale_conditioning_requested = bool(
            getattr(configs, "mamba_acf_scale_conditioning", False)
        )
        self.bidirectional = bool(getattr(configs, "mamba_bidirectional", False))
        self.channel_independent = bool(
            getattr(configs, "mamba_channel_independent", False)
        )
        self.causal_embedding = bool(getattr(configs, "mamba_causal_embedding", False))
        self.channel_gate_enabled = bool(getattr(configs, "mamba_channel_gate", False))
        self.channel_gate_rank = int(getattr(configs, "mamba_channel_gate_rank", 8))
        if self.channel_gate_rank <= 0:
            raise ValueError("mamba_channel_gate_rank must be positive")
        if self.channel_gate_enabled and self.channel_independent:
            raise ValueError(
                "mamba_channel_gate is a shared-mode cross-variable module and "
                "cannot be combined with mamba_channel_independent"
            )
        # Optional CMamba-inspired channel-state modulation.  The module
        # consumes a compact summary of the observed window and emits one
        # low-rank hidden-space FiLM residual.  It is deliberately separate
        # from ``LowRankChannelGate`` (which modulates raw scalar inputs) so
        # the forecast and AutoCon branches consume the same shared state.
        self.data_dependent_channel_mixer_enabled = bool(
            getattr(configs, "mamba_data_dependent_channel_mixer", False)
        )
        self.data_dependent_channel_mixer_rank = int(
            getattr(configs, "mamba_data_dependent_channel_mixer_rank", 8)
        )
        if self.data_dependent_channel_mixer_rank <= 0:
            raise ValueError(
                "mamba_data_dependent_channel_mixer_rank must be positive"
            )
        if (
            self.data_dependent_channel_mixer_enabled
            and self.channel_independent
        ):
            raise ValueError(
                "mamba_data_dependent_channel_mixer is a shared-mode "
                "cross-variable module and cannot be combined with "
                "mamba_channel_independent"
            )
        self.acf_channel_gate_enabled = bool(
            getattr(configs, "mamba_acf_channel_gate", False)
        )
        if self.acf_channel_gate_enabled and not self.channel_gate_enabled:
            raise ValueError(
                "mamba_acf_channel_gate requires mamba_channel_gate=true"
            )
        self.variable_mixer_enabled = bool(
            getattr(configs, "mamba_variable_mixer", False)
        )
        # Optional CMamba-lite modulation over the channel-preserving
        # variable-token state.  This is deliberately distinct from the
        # older global hidden-state adapter below: the latter emits one
        # [B,T,D] residual, while this branch keeps a separate state for every
        # channel and is therefore able to repair the high-dimensional
        # channel-compression bottleneck.
        self.variable_gdd_mixer_enabled = bool(
            getattr(configs, "mamba_variable_gdd_mixer", False)
        )
        self.variable_gdd_mixer_rank = int(
            getattr(configs, "mamba_variable_gdd_mixer_rank", 8)
        )
        # A strict zero gate makes the affine subnetwork gradient-free.  Keep
        # zero as the backward-compatible default, while allowing registered
        # candidate screens to use a small non-zero startup value.
        self.variable_gdd_mixer_init_scale = float(
            getattr(configs, "mamba_variable_gdd_init_scale", 0.0)
        )
        if not math.isfinite(self.variable_gdd_mixer_init_scale) or not (
            0.0 <= self.variable_gdd_mixer_init_scale <= 1.0
        ):
            raise ValueError(
                "mamba_variable_gdd_init_scale must be a finite value in [0, 1]"
            )
        if self.variable_gdd_mixer_rank <= 0:
            raise ValueError("mamba_variable_gdd_mixer_rank must be positive")
        if self.variable_gdd_mixer_enabled and not self.variable_mixer_enabled:
            raise ValueError(
                "mamba_variable_gdd_mixer requires mamba_variable_mixer=true"
            )
        self.variable_mixer_depth = max(
            1, int(getattr(configs, "mamba_variable_mixer_depth", 1))
        )
        # A grouped variable scan pools contiguous channel tokens before the
        # optional cross-variable Mamba branch.  ``1`` is the exact legacy
        # path: every channel remains one token and no grouping operation is
        # inserted.  Larger groups are useful for high-dimensional datasets
        # where scanning all variables would dominate the temporal encoder.
        self.variable_group_size = int(
            getattr(configs, "mamba_variable_group_size", 1)
        )
        if self.variable_group_size <= 0:
            raise ValueError("mamba_variable_group_size must be positive")
        if self.variable_group_size != 1 and not self.variable_mixer_enabled:
            raise ValueError(
                "mamba_variable_group_size requires mamba_variable_mixer=true"
            )
        if self.variable_mixer_enabled:
            self.variable_group_count = int(
                math.ceil(self.channels / self.variable_group_size)
            )
            self.variable_group_padding = int(
                self.variable_group_count * self.variable_group_size - self.channels
            )
        else:
            self.variable_group_count = 0
            self.variable_group_padding = 0
        self.variable_permutation = bool(
            getattr(configs, "mamba_variable_permutation", False)
        )
        # Variable-axis Mamba is sequence-order sensitive. ``input`` retains
        # the historical CSV order, while ``strength`` and ``similarity``
        # make deterministic train-prior orderings.  Both alternatives are
        # opt-in so existing runs keep their exact token order.
        self.acf_channel_order = str(
            getattr(configs, "mamba_acf_channel_order", "input")
        ).lower()
        if self.acf_channel_order not in {"input", "strength", "similarity"}:
            raise ValueError(
                "mamba_acf_channel_order must be input, strength, or similarity"
            )
        if self.acf_channel_order != "input" and not self.variable_mixer_enabled:
            raise ValueError(
                "mamba_acf_channel_order={} requires ".format(
                    self.acf_channel_order
                ) +
                "mamba_variable_mixer=true"
            )
        self.acf_channel_conditioning = bool(
            getattr(configs, "mamba_acf_channel_conditioning", False)
        )
        if self.acf_channel_conditioning and not self.variable_mixer_enabled:
            raise ValueError(
                "mamba_acf_channel_conditioning requires mamba_variable_mixer=true"
            )
        if self.variable_mixer_enabled and self.channel_independent:
            raise ValueError(
                "mamba_variable_mixer is a shared cross-variable branch and "
                "cannot be combined with mamba_channel_independent"
            )
        # Optional representation fusion lets the AutoCon relation objective
        # supervise the channel-preserving variable-token branch directly.
        # Keep this opt-in and structurally narrow: plain forecasts and models
        # without the variable branch retain the historical representation
        # path and do not allocate an unused adapter.
        self.variable_repr_fusion_requested = bool(
            getattr(configs, "mamba_variable_repr_fusion", False)
        )
        if self.variable_repr_fusion_requested and not self.AutoCon:
            raise ValueError(
                "mamba_variable_repr_fusion requires AutoCon=true"
            )
        if self.variable_repr_fusion_requested and not self.variable_mixer_enabled:
            raise ValueError(
                "mamba_variable_repr_fusion requires mamba_variable_mixer=true"
            )
        self.variable_repr_fusion_enabled = bool(
            self.variable_repr_fusion_requested
            and self.AutoCon
            and self.variable_mixer_enabled
        )

        self.patch_mixer_enabled = bool(
            getattr(configs, "mamba_patch_mixer", False)
        )
        self.patch_mixer_depth = max(
            1, int(getattr(configs, "mamba_patch_mixer_depth", 1))
        )
        self.patch_len = int(getattr(configs, "mamba_patch_len", 16))
        self.patch_stride = int(getattr(configs, "mamba_patch_stride", 8))
        self.patch_order = str(
            getattr(configs, "mamba_patch_order", "time_major")
        ).lower()
        self.patch_permutation = bool(
            getattr(configs, "mamba_patch_permutation", False)
        )
        if self.patch_mixer_enabled and self.channel_independent:
            raise ValueError(
                "mamba_patch_mixer is a shared cross-variable branch and "
                "cannot be combined with mamba_channel_independent"
            )
        if self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError("mamba_patch_len and mamba_patch_stride must be positive")
        if self.patch_len > self.seq_len:
            raise ValueError(
                "mamba_patch_len={} cannot exceed seq_len={}".format(
                    self.patch_len, self.seq_len
                )
            )
        if self.patch_order not in {"time_major", "channel_major"}:
            raise ValueError(
                "mamba_patch_order must be time_major or channel_major"
            )

        # Optional channel-preserving local pathway.  It encodes short temporal
        # patches independently for every variable with weights shared across
        # channels, then uses a factorized horizon projection.  Unlike the
        # variable-token branch this does not scan C tokens with Mamba and
        # therefore targets high-dimensional datasets such as Traffic.
        self.local_mixer_enabled = bool(
            getattr(configs, "mamba_local_mixer", False)
        )
        self.local_patch_len = int(
            getattr(configs, "mamba_local_patch_len", 16)
        )
        self.local_patch_stride = int(
            getattr(configs, "mamba_local_patch_stride", 8)
        )
        configured_local_width = getattr(configs, "mamba_local_width", None)
        self.local_width = (
            self.d_model
            if configured_local_width is None
            else int(configured_local_width)
        )
        self.local_rank = int(getattr(configs, "mamba_local_rank", 16))
        self.local_depth = max(
            1, int(getattr(configs, "mamba_local_depth", 1))
        )
        if self.local_mixer_enabled:
            if self.channel_independent:
                raise ValueError(
                    "mamba_local_mixer is a shared channel-preserving branch "
                    "and cannot be combined with mamba_channel_independent"
                )
            if self.local_patch_len <= 0 or self.local_patch_stride <= 0:
                raise ValueError(
                    "mamba_local_patch_len and mamba_local_patch_stride must be positive"
                )
            if self.local_patch_len > self.seq_len:
                raise ValueError(
                    "mamba_local_patch_len={} cannot exceed seq_len={}".format(
                        self.local_patch_len, self.seq_len
                    )
                )
            if self.local_width <= 0:
                raise ValueError("mamba_local_width must be positive")
            if self.local_rank <= 0:
                raise ValueError("mamba_local_rank must be positive")
            self.local_patch_count = max(
                1,
                math.ceil(
                    (self.seq_len - self.local_patch_len)
                    / self.local_patch_stride
                )
                + 1,
            )
            padded_length = (
                (self.local_patch_count - 1) * self.local_patch_stride
                + self.local_patch_len
            )
            self.local_right_pad = max(0, padded_length - self.seq_len)
            local_input_width = self.local_patch_count * self.local_width
            if self.local_rank >= min(local_input_width, self.pred_len):
                raise ValueError(
                    "mamba_local_rank={} must be smaller than min(local patch "
                    "features={}, pred_len={})".format(
                        self.local_rank, local_input_width, self.pred_len
                    )
                )
            self.local_patch_embedding = nn.Linear(
                self.local_patch_len, self.local_width
            )
            self.local_blocks = nn.ModuleList(
                [
                    _LocalPatchBlock(
                        self.local_width, self.d_ff, self.dropout
                    )
                    for _ in range(self.local_depth)
                ]
            )
            self.local_norm = nn.LayerNorm(self.local_width)
            self.local_head = _TemporalProjection(
                local_input_width, self.pred_len, self.local_rank
            )
            self.local_gate = nn.Linear(self.local_width, 1)
            nn.init.constant_(self.local_gate.bias, -2.0)
            self.local_residual_scale = nn.Parameter(torch.zeros(()))
            # A separate ReZero adapter lets AutoCon train the local summary
            # without forcing its forecast correction to be active at start.
            self.local_repr_projection = nn.Linear(
                self.local_width, self.d_model
            )
            self.local_repr_scale = nn.Parameter(torch.zeros(()))
            if not self.AutoCon:
                # The local representation adapter is only consumed by the
                # AutoCon objective.  Keep its keys for checkpoint
                # compatibility, but exclude dead parameters from plain runs.
                for parameter in self.local_repr_projection.parameters():
                    parameter.requires_grad_(False)
                self.local_repr_scale.requires_grad_(False)
        else:
            self.local_patch_count = None
            self.local_right_pad = 0
            self.local_patch_embedding = None
            self.local_blocks = None
            self.local_norm = None
            self.local_head = None
            self.local_gate = None
            self.register_parameter("local_residual_scale", None)
            self.local_repr_projection = None
            self.register_parameter("local_repr_scale", None)

        # The default embedding mixes channels once at each timestamp.  This
        # keeps the number of Mamba scans independent of the number of input
        # variables.  The independent path is useful for a controlled ablation
        # and keeps the per-channel representation semantics of the prototype.
        embedding_cls = CausalDataEmbedding if self.causal_embedding else DataEmbedding
        self.embedding = embedding_cls(
            1 if self.channel_independent else self.channels,
            self.d_model,
            self.embed,
            self.freq,
            self.wo_freq,
            self.dropout,
        )
        # The gate is applied to normalized values before temporal embedding.
        # Its scalar per-channel input keeps the added work linear in C and
        # avoids a second Mamba scan or a dense C-by-C projection.
        self.channel_gate = (
            LowRankChannelGate(
                self.channels,
                dim=1,
                rank=self.channel_gate_rank,
                dropout=self.dropout,
                acf_summary_dim=(4 if self.acf_channel_gate_enabled else None),
            )
            if self.channel_gate_enabled else None
        )
        if self.data_dependent_channel_mixer_enabled:
            # Reuse the shared scale-gate hidden summary rather than scanning
            # the raw window a second time.  Fork the RNG so enabling this
            # optional branch does not perturb initialization of the
            # historical backbone under the same seed.
            with torch.random.fork_rng(devices=[]):
                self.data_dependent_channel_mixer = (
                    LowRankGlobalChannelStateMixer(
                        self.channels,
                        self.d_model,
                        rank=self.data_dependent_channel_mixer_rank,
                        dropout=self.dropout,
                        summary_dim=self.d_model,
                    )
                )
            self.channel_state_embedding = None
        else:
            self.channel_state_embedding = None
            self.data_dependent_channel_mixer = None
        if self.variable_mixer_enabled:
            # Treat each variable's observed window as one token.  The
            # projection is shared across variables, so parameters remain
            # independent of C while the Mamba recurrence can model
            # cross-variable interactions in one scan.
            self.variable_embedding = nn.Linear(self.seq_len, self.d_model)
            self.variable_blocks = nn.ModuleList([
                _MambaFFNBlock(
                    self.d_model,
                    self.mamba_d_state,
                    self.mamba_d_conv,
                    self.mamba_expand,
                    self.mamba_headdim,
                    self.dropout,
                    self.mamba_kind,
                    self.d_ff,
                )
                for _ in range(self.variable_mixer_depth)
            ])
            self.variable_norm = nn.LayerNorm(self.d_model)
            if self.variable_gdd_mixer_enabled:
                # Keep optional-module initialization from changing the
                # historical variable-token backbone under the same seed.
                with torch.random.fork_rng(devices=[]):
                    self.variable_gdd_mixer = LowRankDataDependentChannelMixer(
                        channels=self.channels,
                        dim=self.d_model,
                        rank=self.variable_gdd_mixer_rank,
                        dropout=self.dropout,
                        init_scale=self.variable_gdd_mixer_init_scale,
                    )
            else:
                self.variable_gdd_mixer = None
            self.variable_head = nn.Linear(self.d_model, self.pred_len)
            self.variable_gate = nn.Linear(self.d_model, 1)
            nn.init.constant_(self.variable_gate.bias, -2.0)
            # ReZero keeps the plain temporal model nested inside this
            # optional branch and lets optimization decide whether the
            # cross-variable forecast is useful for a dataset.
            self.variable_residual_scale = nn.Parameter(torch.zeros(()))
            if self.acf_channel_conditioning:
                # The prior may retain a different number of selected lags for
                # each horizon.  Fixed summary statistics keep this adapter's
                # input width stable while preserving signed, per-channel
                # periodic information and peak-response information.
                self.acf_channel_conditioner = nn.Sequential(
                    nn.Linear(4, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                self.acf_conditioning_scale = nn.Parameter(torch.zeros(()))
            else:
                self.acf_channel_conditioner = None
                self.register_parameter("acf_conditioning_scale", None)
            if self.variable_repr_fusion_enabled:
                # A learned channel weighting keeps the summary channel-aware
                # while remaining permutation-invariant after any configured
                # train-only channel ordering/permutation is restored.
                # Do not let this optional adapter perturb initialization of
                # the shared forecast backbone.  The registered off/on screen
                # treats this as a one-factor contrast under a common seed.
                with torch.random.fork_rng(devices=[]):
                    self.variable_repr_pool = nn.Linear(self.d_model, 1)
                    self.variable_repr_projection = nn.Sequential(
                        nn.Linear(self.d_model, self.d_model),
                        nn.GELU(),
                        nn.Linear(self.d_model, self.d_model),
                    )
                # A small non-zero gate keeps the candidate near the legacy
                # representation at startup while ensuring the relation loss
                # reaches variable-token parameters on the first update.
                self.variable_repr_scale = nn.Parameter(torch.tensor(0.01))
            else:
                self.variable_repr_pool = None
                self.variable_repr_projection = None
                self.register_parameter("variable_repr_scale", None)
        else:
            self.variable_embedding = None
            self.variable_blocks = None
            self.variable_norm = None
            self.variable_gdd_mixer = None
            self.variable_head = None
            self.variable_gate = None
            self.register_parameter("variable_residual_scale", None)
            self.acf_channel_conditioner = None
            self.register_parameter("acf_conditioning_scale", None)
            self.variable_repr_pool = None
            self.variable_repr_projection = None
            self.register_parameter("variable_repr_scale", None)

        if self.patch_mixer_enabled:
            # Each channel keeps a short temporal patch sequence before the
            # shared Mamba scan.  The token order is explicit because a scan
            # over variables is not permutation invariant by construction.
            self.patch_count = max(
                1,
                math.ceil((self.seq_len - self.patch_len) / self.patch_stride) + 1,
            )
            padded_length = (
                (self.patch_count - 1) * self.patch_stride + self.patch_len
            )
            self.patch_right_pad = max(0, padded_length - self.seq_len)
            self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
            self.patch_blocks = nn.ModuleList([
                _MambaFFNBlock(
                    self.d_model,
                    self.mamba_d_state,
                    self.mamba_d_conv,
                    self.mamba_expand,
                    self.mamba_headdim,
                    self.dropout,
                    self.mamba_kind,
                    self.d_ff,
                )
                for _ in range(self.patch_mixer_depth)
            ])
            self.patch_norm = nn.LayerNorm(self.d_model)
            self.patch_head = nn.Linear(
                self.patch_count * self.d_model, self.pred_len
            )
            self.patch_gate = nn.Linear(self.d_model, 1)
            nn.init.constant_(self.patch_gate.bias, -2.0)
            self.patch_residual_scale = nn.Parameter(torch.zeros(()))
        else:
            self.patch_count = None
            self.patch_right_pad = 0
            self.patch_embedding = None
            self.patch_blocks = None
            self.patch_norm = None
            self.patch_head = None
            self.patch_gate = None
            self.register_parameter("patch_residual_scale", None)
        self.branches = nn.ModuleList()
        for _scale in self.scales:
            self.branches.append(nn.ModuleList([
                _MambaFFNBlock(
                    self.d_model,
                    self.mamba_d_state,
                    self.mamba_d_conv,
                    self.mamba_expand,
                    self.mamba_headdim,
                    self.dropout,
                    self.mamba_kind,
                    self.d_ff,
                )
                for _ in range(self.depth)
            ]))
        self.scale_gate = nn.Linear(self.d_model, len(self.scales))
        if self.acf_scale_conditioning_requested:
            scale_hidden = max(8, min(self.d_model, 32))
            # Keep the scale-on model nested in the scale-off initialization.
            # The new conditioner needs random weights, but consuming the
            # process-wide generator here would otherwise alter every later
            # shared layer under an identical experiment seed.
            with torch.random.fork_rng(devices=[]):
                self.acf_scale_conditioner = nn.Sequential(
                    nn.Linear(6, scale_hidden),
                    nn.GELU(),
                    nn.Linear(scale_hidden, len(self.scales)),
                )
            self.acf_scale_conditioning_scale = nn.Parameter(torch.zeros(()))
        else:
            self.acf_scale_conditioner = None
            self.register_parameter("acf_scale_conditioning_scale", None)
        self.fused_norm = nn.LayerNorm(self.d_model)

        # The forecasting state and the contrastive state are intentionally
        # separated.  AutoCon sees only the projection head output.
        self.value_head = nn.Linear(
            self.d_model, 1 if self.channel_independent else self.channels
        )
        # The temporal forecast projection consumes the tokenized Mamba state.
        # For the default stride=1 this is exactly ``seq_len``; an opt-in
        # stride therefore reduces both the recurrent state and this dense
        # horizon projection instead of restoring the state before forecasting.
        self.temporal_input_length = self.base_token_count
        self.low_rank_head = bool(getattr(configs, "mamba_low_rank_head", False))
        temporal_rank = getattr(configs, "mamba_temporal_rank", 16)
        if self.low_rank_head:
            self.temporal_rank = int(temporal_rank)
            # Both horizon projections are factorized.  Leaving the residual
            # full-rank would retain the same O(seq_len * pred_len) memory
            # cost the option is intended to measure.
            self.temporal_head = _TemporalProjection(
                self.temporal_input_length, self.pred_len, self.temporal_rank
            )
            self.linear_residual = _TemporalProjection(
                self.seq_len, self.pred_len, self.temporal_rank
            )
        else:
            self.temporal_rank = None
            # Keep the historical modules and state-dict keys byte-for-byte
            # compatible when the option is disabled.
            self.temporal_head = nn.Linear(
                self.temporal_input_length, self.pred_len
            )
            self.linear_residual = nn.Linear(self.seq_len, self.pred_len)
        self.residual_gate = nn.Linear(self.d_model, self.pred_len)
        nn.init.constant_(self.residual_gate.bias, -1.5)

        self.channel_pool = nn.Linear(self.d_model, 1)
        # ``channel_pool`` is only consumed when channel-independent states
        # are pooled for the AutoCon representation.  Keep the module for
        # checkpoint/API compatibility, but exclude it from plain or shared
        # temporal capacity where its output is never used.
        if not (self.channel_independent and self.AutoCon):
            for parameter in self.channel_pool.parameters():
                parameter.requires_grad_(False)
        self.repr_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        if not self.AutoCon:
            # The plain forecast does not consume this projection.  Freeze it
            # so resource counts and optimizer state reflect actual capacity.
            for parameter in self.repr_head.parameters():
                parameter.requires_grad_(False)

        # Optional ACF context for the representation branch.  The historical
        # memory residual is applied after the representation has already been
        # formed, so AutoCon would otherwise never see the ACF signal it is
        # meant to regularize.  Keep this adapter opt-in and ReZero-gated: the
        # default model and all existing checkpoints retain their exact path.
        self.acf_repr_fusion_requested = bool(
            getattr(configs, "mamba_acf_repr_fusion", False)
        )
        self.acf_repr_fusion_enabled = bool(
            self.acf_repr_fusion_requested and self.AutoCon
        )
        if self.acf_repr_fusion_enabled:
            self.acf_repr_projection = nn.Sequential(
                nn.Linear(4, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.acf_repr_scale = nn.Parameter(torch.zeros(()))
        else:
            self.acf_repr_projection = None
            self.register_parameter("acf_repr_scale", None)

        acf_lags = getattr(configs, "mamba_acf_lags", [24, 168, 336])
        # Keep the historical plain-forecast behavior where an explicitly
        # empty lag list disables the optional ACF branches.  Branch-specific
        # validation below still rejects an empty bank when it is enabled.
        self.acf_lags = _canonical_positive_lags(
            acf_lags, "mamba_acf_lags", allow_empty=True
        )
        self.use_acf_memory = bool(getattr(configs, "mamba_use_acf_memory", False))
        self.dynamic_acf = bool(getattr(configs, "mamba_dynamic_acf", False))
        # A shared lag bank is the historical forecast-memory contract.  The
        # optional channel mask below keeps that bank compact while allowing
        # each channel to use only its strongest train-only lag slots.  It is
        # deliberately opt-in: relation learning still sees the complete
        # prior, and all existing checkpoints retain the exact default path.
        self.acf_channel_lag_bank = bool(
            getattr(configs, "mamba_acf_channel_lag_bank", False)
        )
        raw_channel_lag_top_k = getattr(
            configs, "mamba_acf_channel_lag_top_k", 2
        )
        try:
            self.acf_channel_lag_top_k = _strict_positive_integer(
                raw_channel_lag_top_k, "mamba_acf_channel_lag_top_k"
            )
        except ValueError:
            raise
        if self.acf_channel_lag_bank:
            if not self.use_acf_memory:
                raise ValueError(
                    "mamba_acf_channel_lag_bank requires mamba_use_acf_memory=true"
                )
            if self.dynamic_acf:
                raise ValueError(
                    "mamba_acf_channel_lag_bank requires static train-only ACF "
                    "memory (mamba_dynamic_acf=false)"
                )
            if not self.acf_lags:
                raise ValueError(
                    "mamba_acf_channel_lag_bank requires non-empty mamba_acf_lags"
                )
            if self.acf_channel_lag_top_k > len(self.acf_lags):
                raise ValueError(
                    "mamba_acf_channel_lag_top_k cannot exceed the configured "
                    "mamba_acf_lags count"
                )
        # Plan files use the lowercase spelling while argparse exposes the
        # historical camel-case attribute.  Accept both so direct model API
        # construction cannot silently pair a raw local signal with a
        # non-raw prior merely because the spelling differed.
        self.acf_source = resolve_acf_source(configs)
        if self.acf_source not in {"raw", "residual", "trend"}:
            raise ValueError("AutoCon_acf_source must be raw, residual, or trend")
        # A local ACF estimated from only a handful of pairs is too noisy to
        # be a meaningful conditioning signal.  Keep this fixed and visible
        # in the resource profile so candidate plans cannot silently change
        # the statistical support of the branch.
        self.acf_scale_conditioning_min_pairs = 16
        if self.acf_scale_conditioning_requested:
            if self.acf_source != "raw":
                raise ValueError(
                    "mamba_acf_scale_conditioning requires AutoCon_acf_source=raw"
                )
            if not any(
                int(self.seq_len) - int(lag) >= self.acf_scale_conditioning_min_pairs
                for lag in self.acf_lags
            ):
                raise ValueError(
                    "mamba_acf_scale_conditioning requires at least one ACF lag "
                    "with 16 or more window pairs"
                )
        # Direct-anchor retrieval is deliberately narrower than the historical
        # multi-lag memory: it selects a signed, per-channel lag from the
        # immutable train-only prior and therefore requires the same raw signal
        # at fit and forecast time.  Keep this opt-in so all existing paths and
        # checkpoints retain their original behavior.
        self.acf_anchor_copy = bool(
            getattr(configs, "mamba_acf_anchor_copy", False)
        )
        if self.acf_anchor_copy:
            if self.dynamic_acf:
                raise ValueError(
                    "mamba_acf_anchor_copy requires static train-only ACF "
                    "memory (mamba_dynamic_acf=false)"
                )
            if self.acf_source != "raw":
                raise ValueError(
                    "mamba_acf_anchor_copy requires AutoCon_acf_source=raw"
                )
        try:
            self.acf_temperature = float(
                getattr(configs, "mamba_acf_temperature", 4.0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("mamba_acf_temperature must be finite and positive") from exc
        if not math.isfinite(self.acf_temperature) or self.acf_temperature <= 0:
            raise ValueError("mamba_acf_temperature must be positive")
        # The historical memory path uses one horizon-aggregated reliability
        # value.  The optional stepwise mode keeps that behavior reproducible
        # while allowing each forecast step to be gated by the lags that are
        # actually observable at that step.
        self.acf_step_reliability = bool(
            getattr(configs, "mamba_acf_step_reliability", False)
        )
        # Optional interpolation between the horizon-aggregated train-prior
        # reliability and the reliability of the lags legal at each step.
        # The scalar ReZero adapter keeps the branch exactly nested at
        # initialization while allowing the forecast loss to learn whether
        # absolute per-step ACF strength is useful.  It is intentionally
        # restricted to the static raw-prior path: no current-window estimate
        # or target value can influence this factor.
        self.acf_reliability_interpolation = bool(
            getattr(configs, "mamba_acf_reliability_interpolation", False)
        )
        if self.acf_reliability_interpolation:
            if not self.use_acf_memory:
                raise ValueError(
                    "mamba_acf_reliability_interpolation requires "
                    "mamba_use_acf_memory=true"
                )
            if self.dynamic_acf:
                raise ValueError(
                    "mamba_acf_reliability_interpolation requires static "
                    "train-only ACF memory (mamba_dynamic_acf=false)"
                )
            if self.acf_source != "raw":
                raise ValueError(
                    "mamba_acf_reliability_interpolation requires "
                    "AutoCon_acf_source=raw"
                )
            if self.acf_step_reliability:
                raise ValueError(
                    "mamba_acf_reliability_interpolation is incompatible with "
                    "mamba_acf_step_reliability=true"
                )
            self.acf_reliability_interpolation_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("acf_reliability_interpolation_scale", None)
        # Optional causal local-ACF agreement for the static train-prior
        # memory.  The branch only changes lag weights; it never changes the
        # prior sign used by the retrieved seasonal value.  Keep the minimum
        # pair count explicit so a long lag with one or two noisy pairs cannot
        # open a seemingly input-dependent path.
        self.acf_local_agreement_requested = bool(
            getattr(configs, "mamba_acf_local_agreement", False)
        )
        self.acf_local_min_pairs = int(
            getattr(configs, "mamba_acf_local_min_pairs", 16)
        )
        if self.acf_local_min_pairs <= 0:
            raise ValueError("mamba_acf_local_min_pairs must be positive")
        if self.acf_local_agreement_requested:
            if self.acf_source != "raw":
                raise ValueError(
                    "mamba_acf_local_agreement requires AutoCon_acf_source=raw"
                )
            self.acf_local_agreement_scale = nn.Parameter(torch.zeros(()))
            # Do not count a parameter in an incompatible/dormant forecast
            # branch as trainable capacity.  The state-dict key is retained so
            # toggling the option remains serialization-friendly.
            if not (self.use_acf_memory and not self.dynamic_acf):
                self.acf_local_agreement_scale.requires_grad_(False)
        else:
            self.register_parameter("acf_local_agreement_scale", None)
        self.acf_memory_gate_mode = str(
            getattr(configs, "mamba_acf_memory_gate_mode", "hidden")
        ).lower()
        if self.acf_memory_gate_mode not in {"hidden", "variable_token"}:
            raise ValueError(
                "mamba_acf_memory_gate_mode must be hidden or variable_token"
            )
        if self.acf_memory_gate_mode == "variable_token":
            if not self.variable_mixer_enabled:
                raise ValueError(
                    "mamba_acf_memory_gate_mode=variable_token requires "
                    "mamba_variable_mixer=true"
                )
            if not self.use_acf_memory or self.dynamic_acf:
                raise ValueError(
                    "mamba_acf_memory_gate_mode=variable_token requires "
                    "static ACF memory (mamba_use_acf_memory=true, "
                    "mamba_dynamic_acf=false)"
                )
        self.acf_horizon_calibration_enabled = bool(
            getattr(configs, "mamba_acf_horizon_calibration", False)
        )
        self.acf_horizon_calibration_rank = int(
            getattr(configs, "mamba_acf_horizon_calibration_rank", 2)
        )
        if self.acf_horizon_calibration_enabled:
            if not self.use_acf_memory:
                raise ValueError(
                    "mamba_acf_horizon_calibration requires mamba_use_acf_memory=true"
                )
            if self.dynamic_acf:
                raise ValueError(
                    "mamba_acf_horizon_calibration currently supports only "
                    "static train-only ACF memory (mamba_dynamic_acf=false)"
                )
            self.acf_horizon_calibration = _HorizonLagAmplitudeCalibration(
                max(1, self.pred_len), max(1, len(self.acf_lags)),
                rank=self.acf_horizon_calibration_rank,
            )
            self.acf_horizon_calibration.lag_values = tuple(self.acf_lags)
        else:
            self.acf_horizon_calibration = None
        # Forward-time diagnostics are intentionally ordinary Python-visible
        # scalars.  They are not model state and are never used to choose a
        # checkpoint; the training loop may copy them into audit logs.
        self._acf_runtime_diagnostics = {
            "acf_runtime_gate_mean": None,
            "acf_runtime_gate_std": None,
            "acf_memory_gate_variable_token": float(
                self.acf_memory_gate_mode == "variable_token"
            ),
            "acf_anchor_copy_enabled": float(self.acf_anchor_copy),
            "acf_anchor_scale": None,
            "acf_anchor_rms": None,
            "acf_anchor_valid_fraction": None,
            "acf_anchor_strength_mean": None,
            "acf_anchor_lag_mean": None,
            "acf_seasonal_rms": None,
            "acf_reliability_mean": None,
            "acf_effective_memory_rms": None,
            "acf_effective_memory_to_forecast_rms": None,
            "acf_calibration_amp_mean": None,
            "acf_calibration_amp_std": None,
            "acf_calibration_amp_min": None,
            "acf_calibration_amp_max": None,
            "acf_reliability_interpolation_mean": None,
            "acf_reliability_interpolation_std": None,
            "acf_reliability_interpolation_min": None,
            "acf_reliability_interpolation_max": None,
        }
        # Training logs consume scalar diagnostics after every batch.  Eval
        # and deployment paths do not need to synchronize five anchor
        # telemetry values from GPU to CPU, so keep that collection opt-in
        # outside training.  Resource benchmarks therefore measure forecast
        # computation rather than logging overhead.
        self.acf_collect_eval_diagnostics = bool(
            getattr(configs, "mamba_collect_eval_acf_diagnostics", False)
        )
        # The experiment wrapper may replace these buffers with a signed,
        # train-split ACF prior after the model is constructed.  Empty buffers
        # mean that the model falls back to a causal window estimate.
        self.register_buffer("acf_prior", torch.empty(0), persistent=False)
        # Keep the complete train-only table separately from the compact
        # forecast-lag table.  AutoCon's relation objective needs distances
        # beyond the selected forecast lags when a run is resumed from its
        # sidecar, while the forecast path only needs the compact table.
        self.register_buffer("acf_full_prior", torch.empty(0), persistent=False)
        self.register_buffer("acf_prior_lags", torch.empty(0, dtype=torch.long), persistent=False)
        # Static train-prior memory terms are populated by ``set_acf_prior``.
        # They contain no batch dimension and are deliberately non-persistent:
        # the experiment wrapper reattaches the train-only prior on every run.
        self.register_buffer(
            "acf_memory_static_coefficients", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_step_reliability", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_aggregate_reliability", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_indices", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_valid", torch.empty(0, dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_channel_support",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        # Per-channel direct-anchor terms are compiled from the same prior and
        # causal index grid as the soft memory.  They contain no batch axis and
        # are non-persistent train-split metadata.
        self.register_buffer(
            "acf_memory_static_anchor_indices",
            torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_anchor_lags",
            torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_anchor_sign",
            torch.empty(0), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_anchor_strength",
            torch.empty(0), persistent=False
        )
        self.register_buffer(
            "acf_memory_static_anchor_valid",
            torch.empty(0, dtype=torch.bool), persistent=False
        )
        self._acf_memory_static_length = None
        self._acf_memory_static_horizon = None
        self._acf_memory_static_lags = None
        self._acf_memory_static_temperature = None
        self._acf_memory_static_prior_identity = None
        # ``acf_prior`` is a public, non-persistent buffer and callers may
        # update it in-place after attachment.  Keep the tensor version used
        # to build the derived terms so such updates cannot leave stale
        # coefficients in the fast path.
        self._acf_memory_static_prior_version = None
        self._acf_memory_static_channel_lag_bank = None
        self._acf_memory_static_channel_lag_top_k = None
        if self.acf_anchor_copy:
            # A scalar ReZero adapter makes the new branch exactly nested at
            # initialization while giving the forecast loss a direct gradient
            # to the retrieved seasonal anchor from the first update.
            self.acf_anchor_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("acf_anchor_scale", None)
        # These are derived exclusively by ``set_acf_prior``. They are
        # non-persistent because they are train-split statistics rather than
        # learned model state, preserving old checkpoint compatibility.
        input_order = torch.arange(self.channels, dtype=torch.long)
        self.register_buffer(
            "acf_channel_order_indices", input_order.clone(), persistent=False
        )
        self.register_buffer(
            "acf_channel_inverse_indices", input_order.clone(), persistent=False
        )
        self.register_buffer(
            "acf_channel_dominant_peak", torch.zeros(self.channels), persistent=False
        )
        self.register_buffer(
            "acf_channel_dominant_peak_lag", torch.zeros(
                self.channels, dtype=torch.long
            ),
            persistent=False,
        )
        self.acf_gate = nn.Linear(self.d_model, self.pred_len)
        nn.init.constant_(self.acf_gate.bias, -4.0)
        # Preserve the state-dict key even when the optional memory branch is
        # disabled, while ensuring dead parameters are not counted or added to
        # the optimizer state for a plain forecast.
        if not (self.use_acf_memory and self.acf_lags):
            for parameter in self.acf_gate.parameters():
                parameter.requires_grad_(False)
        self.output_projection = (
            nn.Identity() if self.c_out == self.channels
            else nn.Linear(self.channels, self.c_out)
        )

    def set_acf_prior(self, acf_values, lags=None):
        """Attach a signed train-split ACF table to the forecasting model.

        ``acf_values`` may be ``[lag]`` or ``[channel, lag]``.  Only the
        configured ``mamba_acf_lags`` are retained.  This method is called by
        the AutoCon experiment after the training split has been loaded and is
        intentionally independent of validation/test data.
        """

        try:
            model_device = next(self.parameters()).device
        except StopIteration:  # pragma: no cover - the model always has parameters
            model_device = None
        prior = torch.as_tensor(acf_values, dtype=torch.float32, device=model_device)
        if prior.ndim == 1:
            prior = prior.unsqueeze(0).expand(self.channels, -1).clone()
        if prior.ndim != 2:
            raise ValueError("acf_values must have shape [lag] or [channel, lag]")
        prior = _normalize_acf_values(prior, "acf_values")
        if prior.shape[0] == 1 and self.channels != 1:
            prior = prior.expand(self.channels, -1).clone()
        if prior.shape[0] != self.channels:
            raise ValueError(
                "ACF prior has {} channels, model expects {}".format(
                    prior.shape[0], self.channels
                )
            )
        if prior.shape[1] == 0:
            raise ValueError("ACF prior must contain at least lag zero")
        if lags is not None:
            self.acf_lags = _canonical_positive_lags(lags, "lags")
        # Keep the temporary index tensor on the same device as the supplied
        # prior.  The prior is commonly fitted on CPU even when the model has
        # already been moved to CUDA.
        lags = torch.as_tensor(self.acf_lags, dtype=torch.long, device=prior.device)
        safe_lags = lags.clamp(min=0, max=prior.shape[1] - 1)
        selected = prior[:, safe_lags]
        selected = torch.where(
            lags.unsqueeze(0) < prior.shape[1], selected, torch.zeros_like(selected)
        )
        # A short synthetic prior may not contain the requested lag columns;
        # retain a full table only when every selected lag is represented.
        full_prior = (
            prior
            if lags.numel() and int(lags.max().item()) < prior.shape[1]
            else None
        )
        self.set_compact_acf_prior(selected, lags, full_acf_values=full_prior)

    def set_compact_acf_prior(self, acf_values, lags, full_acf_values=None):
        """Attach an already-selected, train-only compact ACF prior.

        This is the inverse of :meth:`export_acf_prior_state` and is used when
        a forecast is resumed from a checkpoint without refitting the source
        dataset.  ``acf_values`` must have shape ``[channels, n_lags]`` and
        must already correspond to ``lags``; no lag selection is performed.
        """

        try:
            model_device = next(self.parameters()).device
        except StopIteration:  # pragma: no cover - the model has parameters
            model_device = None
        selected = torch.as_tensor(
            acf_values, dtype=torch.float32, device=model_device
        )
        if selected.ndim == 1:
            selected = selected.unsqueeze(0).expand(self.channels, -1).clone()
        if selected.ndim != 2 or selected.shape[0] != self.channels:
            raise ValueError(
                "compact ACF prior must have shape [channels, selected_lags]"
            )
        if selected.shape[1] == 0:
            raise ValueError("compact ACF prior must be non-empty and finite")
        selected = _normalize_acf_values(selected, "compact ACF prior")
        raw_lags = _parse_positive_lags(lags, "compact ACF prior lags")
        if len(raw_lags) != selected.shape[1]:
            raise ValueError(
                "compact ACF prior lag count ({}) does not match coefficient "
                "width ({})".format(len(raw_lags), selected.shape[1])
            )
        # Keep lag/value alignment explicit for portable artifacts and callers
        # that supply an unsorted lag list.
        order = sorted(range(len(raw_lags)), key=lambda index: raw_lags[index])
        if order != list(range(len(raw_lags))):
            selected = selected.index_select(
                1, torch.as_tensor(order, dtype=torch.long, device=selected.device)
            )
        parsed_lags = tuple(raw_lags[index] for index in order)
        lag_tensor = torch.as_tensor(
            parsed_lags, dtype=torch.long, device=selected.device
        )
        full_prior = None
        if full_acf_values is not None:
            full_prior = torch.as_tensor(
                full_acf_values, dtype=torch.float32, device=selected.device
            )
            if full_prior.ndim == 1:
                full_prior = full_prior.unsqueeze(0).expand(self.channels, -1).clone()
            if full_prior.ndim != 2 or full_prior.shape[0] != self.channels:
                raise ValueError(
                    "full ACF prior must have shape [channels, measured_lags]"
                )
            if full_prior.shape[1] <= max(parsed_lags):
                raise ValueError(
                    "full ACF prior does not contain every selected lag"
                )
            full_prior = _normalize_acf_values(full_prior, "full ACF prior")
            # Keep the public direct API as strict as the NPZ loader: the
            # compact forecast table and complete relation table must refer to
            # the same lag/value pairs before either is attached to the model.
            # Convert only this one-time setup check to CPU NumPy; no tensor is
            # materialized in the forward path.
            _validate_loaded_acf_prior_alignment(
                selected.detach().cpu().numpy(),
                full_prior.detach().cpu().numpy(),
                parsed_lags,
                self.channels,
            )
        self.acf_lags = parsed_lags
        if getattr(self, "acf_horizon_calibration_enabled", False):
            self.acf_horizon_calibration = _resize_horizon_lag_calibration(
                self.acf_horizon_calibration,
                horizon=self.pred_len,
                lags=parsed_lags,
                device=selected.device,
            )
        self.acf_prior = selected.detach()
        self.acf_full_prior = (
            full_prior.detach()
            if full_prior is not None
            else torch.empty(0, device=selected.device)
        )
        self.acf_prior_lags = lag_tensor.detach()
        self._refresh_static_acf_memory_cache()
        self._set_acf_channel_order(selected, lag_tensor)

    def export_acf_prior_state(self):
        """Return the compact train-only prior in a serialization-friendly form."""

        if self.acf_prior.numel() == 0:
            return None
        if self.acf_prior.ndim != 2 or self.acf_prior.shape[0] != self.channels:
            raise RuntimeError("attached ACF prior has an invalid shape")
        if self.acf_prior_lags.numel() != self.acf_prior.shape[1]:
            raise RuntimeError("attached ACF prior lag metadata is inconsistent")
        full_prior = None
        if self.acf_full_prior.numel():
            if (
                self.acf_full_prior.ndim != 2
                or self.acf_full_prior.shape[0] != self.channels
                or not torch.isfinite(self.acf_full_prior).all()
            ):
                raise RuntimeError("attached full ACF prior has an invalid shape")
            full_prior = self.acf_full_prior.detach().cpu().numpy().astype(
                np.float32, copy=True
            )
        return {
            "schema_version": 1,
            "acf_values": self.acf_prior.detach().cpu().numpy().astype(
                np.float32, copy=True
            ),
            "acf_full_values": full_prior,
            "lags": self.acf_prior_lags.detach().cpu().numpy().astype(
                np.int64, copy=True
            ),
            "seq_len": int(self.seq_len),
            "pred_len": int(self.pred_len),
            "channels": int(self.channels),
            "acf_source": str(self.acf_source),
            "acf_temperature": float(self.acf_temperature),
        }

    def save_acf_prior(self, path):
        """Persist the selected train-only prior as a portable ``.npz`` file."""

        state = self.export_acf_prior_state()
        if state is None:
            return False
        payload = {
            "schema_version": np.asarray(state["schema_version"], dtype=np.int64),
            "acf_values": state["acf_values"],
            "lags": state["lags"],
            "seq_len": np.asarray(state["seq_len"], dtype=np.int64),
            "pred_len": np.asarray(state["pred_len"], dtype=np.int64),
            "channels": np.asarray(state["channels"], dtype=np.int64),
            "acf_source": np.asarray(state["acf_source"]),
            "acf_temperature": np.asarray(state["acf_temperature"], dtype=np.float64),
        }
        if state.get("acf_full_values") is not None:
            payload["acf_full_values"] = state["acf_full_values"]
        np.savez_compressed(str(path), **payload)
        return True

    def load_acf_prior(self, path):
        """Load a prior written by :meth:`save_acf_prior` and attach it."""

        with np.load(str(path), allow_pickle=False) as payload:
            required = {
                "schema_version", "acf_values", "lags", "seq_len", "pred_len",
                "channels", "acf_source", "acf_temperature",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(
                    "ACF prior artifact is missing fields: {}".format(", ".join(missing))
                )
            schema_version = _artifact_scalar_positive_integer(
                payload["schema_version"], "schema_version"
            )
            if schema_version != 1:
                raise ValueError(
                    "unsupported ACF prior artifact schema {}".format(schema_version)
                )
            values = np.asarray(payload["acf_values"], dtype=np.float32)
            full_values = (
                np.asarray(payload["acf_full_values"], dtype=np.float32)
                if "acf_full_values" in payload.files else None
            )
            lags = _artifact_lags(payload["lags"])
            _validate_acf_artifact_values(
                torch.as_tensor(values), "ACF prior artifact values"
            )
            if full_values is not None:
                _validate_acf_artifact_values(
                    torch.as_tensor(full_values),
                    "ACF prior artifact complete values",
                )
            contract = {
                "seq_len": _artifact_scalar_positive_integer(
                    payload["seq_len"], "seq_len"
                ),
                "pred_len": _artifact_scalar_positive_integer(
                    payload["pred_len"], "pred_len"
                ),
                "channels": _artifact_scalar_positive_integer(
                    payload["channels"], "channels"
                ),
                "acf_source": str(np.asarray(payload["acf_source"]).item()),
                "acf_temperature": float(np.asarray(payload["acf_temperature"]).item()),
            }
            if (
                not math.isfinite(contract["acf_temperature"])
                or contract["acf_temperature"] <= 0
            ):
                raise ValueError("ACF prior temperature must be finite and positive")
        expected_contract = {
            "seq_len": int(self.seq_len),
            "pred_len": int(self.pred_len),
            "channels": int(self.channels),
            "acf_source": str(self.acf_source),
            "acf_temperature": float(self.acf_temperature),
        }
        if contract["seq_len"] != expected_contract["seq_len"]:
            raise ValueError("ACF prior seq_len does not match the model")
        if contract["pred_len"] != expected_contract["pred_len"]:
            raise ValueError("ACF prior pred_len does not match the model")
        if contract["channels"] != expected_contract["channels"]:
            raise ValueError("ACF prior channel count does not match the model")
        if contract["acf_source"] != expected_contract["acf_source"]:
            raise ValueError("ACF prior source does not match the model")
        if not np.isclose(
            contract["acf_temperature"], expected_contract["acf_temperature"],
            rtol=0.0, atol=1e-12,
        ):
            raise ValueError("ACF prior temperature does not match the model")
        if full_values is not None:
            _validate_loaded_acf_prior_alignment(
                values, full_values, lags, self.channels
            )
        self.set_compact_acf_prior(
            values, lags, full_acf_values=full_values
        )
        return True

    def _build_static_acf_channel_support(
        self, device=None, *, length=None, horizon=None
    ):
        """Compile a deterministic per-channel mask over usable lag slots.

        The mask is derived only from the detached, train-split compact prior
        and the fixed forecast geometry.  A lag with no legal forecast
        position is excluded before ranking; the remaining slots are ranked
        by ``abs(ACF) * legal_step_fraction``.  Sorting ties by the existing
        ascending-lag bank order makes sidecar replay independent of backend
        ``topk`` tie behavior.
        """

        if (
            not getattr(self, "acf_channel_lag_bank", False)
            or self.acf_prior.numel() == 0
            or not self.acf_lags
        ):
            empty_device = self.acf_prior.device if device is None else device
            return torch.empty(0, dtype=torch.bool, device=empty_device)
        # The sidecar contract is float32.  Recompute selection in that dtype
        # even if a caller has manually replaced the non-persistent buffer
        # with float16, so model support hashes remain backend-independent.
        prior = self.acf_prior.to(dtype=torch.float32)
        if device is not None:
            prior = prior.to(device=device)
        lag_count = int(prior.shape[1])
        top_k = int(self.acf_channel_lag_top_k)
        if top_k > lag_count:
            raise ValueError(
                "mamba_acf_channel_lag_top_k cannot exceed the resolved "
                "ACF lag-bank count"
            )
        resolved_length = int(self.seq_len if length is None else length)
        resolved_horizon = int(self.pred_len if horizon is None else horizon)
        if resolved_length <= 0 or resolved_horizon <= 0:
            raise ValueError("ACF forecast geometry must be positive")
        lags = torch.as_tensor(
            self.acf_lags, device=prior.device, dtype=torch.long
        )
        steps = torch.arange(
            1, resolved_horizon + 1, device=prior.device, dtype=torch.long
        )
        indices = resolved_length - 1 + steps[:, None] - lags[None, :]
        legal_fraction = (
            (indices >= 0) & (indices < resolved_length)
        ).to(dtype=prior.dtype).mean(dim=0)
        legal_slots = legal_fraction > 0
        if int(legal_slots.sum().item()) < top_k:
            raise ValueError(
                "mamba_acf_channel_lag_bank requires at least {} lag slots "
                "with legal forecast support for seq_len={} and pred_len={}; "
                "resolved bank has {}".format(
                    top_k,
                    resolved_length,
                    resolved_horizon,
                    int(legal_slots.sum().item()),
                )
            )
        # ``argsort(stable=True)`` is available in the supported PyTorch
        # versions and gives a documented lag-order tie break.  Keep the
        # operation outside ``forward``; Traffic-sized masks are compiled once
        # when the train-only prior is attached or refreshed.
        scores = prior.abs() * legal_fraction.unsqueeze(0)
        scores = scores.masked_fill(~legal_slots.unsqueeze(0), -torch.inf)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        support = torch.zeros_like(prior, dtype=torch.bool)
        support.scatter_(1, order[:, :top_k], True)
        return support.detach()

    def _build_static_acf_memory_terms(self, length, horizon, device=None):
        """Build batch-independent causal-memory coefficients from the prior.

        The legacy implementation constructs ``[B,H,C,K]`` logits and weights
        for every batch.  Once a train-only prior is attached, all those terms
        depend only on ``(length, horizon, lags, temperature, prior)`` and can
        be compiled once.  Invalid lags remain explicit in the returned mask;
        this preserves the all-invalid-row semantics of the reference path.
        """

        if self.acf_prior.numel() == 0 or not self.acf_lags:
            return (
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
            )
        prior = self.acf_prior
        if device is not None:
            prior = prior.to(device=device)
        lags = torch.as_tensor(self.acf_lags, device=prior.device, dtype=torch.long)
        steps = torch.arange(1, int(horizon) + 1, device=prior.device, dtype=torch.long)
        indices = int(length) - 1 + steps[:, None] - lags[None, :]
        valid = (indices >= 0) & (indices < int(length))
        safe_indices = indices.clamp(0, max(0, int(length) - 1))

        # Match the reference order: masked softmax, availability mask, then
        # denominator normalization.  The resulting signed coefficients are
        # exactly the factors multiplying each historical lag value.  The
        # channel support mask is a pure train-prior restriction and is kept
        # separate from ``valid`` so audit reports can distinguish statistical
        # support from forecast-origin causality.
        channel_support = self._build_static_acf_channel_support(
            device=prior.device, length=length, horizon=horizon
        )
        if channel_support.numel():
            effective_valid = valid[:, None, :] & channel_support[None, :, :]
            logits = torch.log(prior.abs().clamp_min(1e-6)).unsqueeze(0)
            logits = logits / float(self.acf_temperature)
            logits = logits.masked_fill(
                ~effective_valid, torch.finfo(prior.dtype).min
            )
            valid_weight = effective_valid.to(dtype=prior.dtype)
        else:
            logits = torch.log(prior.abs().clamp_min(1e-6)).unsqueeze(0)
            logits = logits / float(self.acf_temperature)
            logits = logits.masked_fill(
                ~valid[:, None, :], torch.finfo(prior.dtype).min
            )
            valid_weight = valid[:, None, :].to(dtype=prior.dtype)
        weights = torch.softmax(logits, dim=-1) * valid_weight
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-4)
        coefficients = prior.sign().unsqueeze(0) * weights
        step_reliability = (
            weights * prior.abs().unsqueeze(0) * valid_weight
        ).sum(dim=-1).clamp(0.0, 1.0)
        if channel_support.numel():
            support_weight = channel_support.to(dtype=prior.dtype)
            availability_fraction = valid.to(dtype=prior.dtype).mean(dim=0)
            aggregate_reliability = (
                (prior.abs() * support_weight * availability_fraction.unsqueeze(0)).sum(dim=-1)
                / support_weight.sum(dim=-1).clamp_min(1.0)
            ).clamp(0.0, 1.0)
        else:
            availability_fraction = valid.to(dtype=prior.dtype).mean(dim=0)
            aggregate_reliability = (
                prior.abs() * availability_fraction.unsqueeze(0)
            ).mean(dim=-1).clamp(0.0, 1.0)
        return (
            coefficients.detach(),
            step_reliability.detach(),
            aggregate_reliability.detach(),
            safe_indices.detach(),
            valid.detach(),
        )

    def _build_static_acf_anchor_terms(self, length, horizon, device=None):
        """Compile causal per-channel maximum-|ACF| anchor lookups.

        The returned tensors describe one selected lag for every
        ``(forecast_step, channel)`` pair.  Selection is based only on the
        train-only compact prior; unavailable lag positions are masked before
        ``argmax`` and remain invalid in the returned mask.  No predicted
        value is ever fed back into the lookup.
        """

        if self.acf_prior.numel() == 0 or not self.acf_lags:
            empty_device = self.acf_prior.device if device is None else device
            return (
                torch.empty(0, dtype=torch.long, device=empty_device),
                torch.empty(0, dtype=torch.long, device=empty_device),
                torch.empty(0, device=empty_device),
                torch.empty(0, device=empty_device),
                torch.empty(0, dtype=torch.bool, device=empty_device),
            )
        prior = self.acf_prior
        if device is not None:
            prior = prior.to(device=device)
        lags = torch.as_tensor(self.acf_lags, device=prior.device, dtype=torch.long)
        steps = torch.arange(1, int(horizon) + 1, device=prior.device, dtype=torch.long)
        indices = int(length) - 1 + steps[:, None] - lags[None, :]
        valid = (indices >= 0) & (indices < int(length))
        safe_indices = indices.clamp(0, max(0, int(length) - 1))

        # ``prior`` is [C,K]; expose [H,C,K] so the selected lag is allowed to
        # vary by both horizon step and channel.  An all-invalid row selects a
        # safe placeholder, but ``anchor_valid`` below keeps its contribution
        # exactly zero.
        scores = prior.abs().unsqueeze(0).expand(int(horizon), -1, -1)
        scores = scores.masked_fill(
            ~valid[:, None, :], torch.finfo(scores.dtype).min
        )
        selected = scores.argmax(dim=-1)
        index_grid = safe_indices[:, None, :].expand(
            int(horizon), self.channels, -1
        )
        valid_grid = valid[:, None, :].expand(int(horizon), self.channels, -1)
        sign_grid = prior.sign().unsqueeze(0).expand(int(horizon), -1, -1)
        strength_grid = prior.abs().unsqueeze(0).expand(int(horizon), -1, -1)
        anchor_indices = index_grid.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        anchor_valid = valid_grid.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        anchor_sign = sign_grid.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        anchor_strength = strength_grid.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        anchor_lags = lags.view(1, 1, -1).expand(
            int(horizon), self.channels, -1
        ).gather(2, selected.unsqueeze(-1)).squeeze(-1)
        return (
            anchor_indices.detach(),
            anchor_lags.detach(),
            anchor_sign.detach(),
            anchor_strength.detach(),
            anchor_valid.detach(),
        )

    def _refresh_static_acf_memory_cache(self, device=None):
        """Refresh non-persistent memory terms after a prior/lag update.

        ``device`` is optional because training normally attaches the prior
        after the model has already been moved.  Resource probes and resumed
        evaluation can attach a CPU sidecar first and move the model later;
        allowing an explicit target device keeps the derived lookup tables
        resident with the tensors that consume them.
        """

        cache_device = self.acf_prior.device if device is None else torch.device(device)

        # Low-level callers/tests may construct the memory subsystem without
        # the full model config; preserve the historical default in that case.
        use_acf_memory = getattr(self, "use_acf_memory", True)
        acf_anchor_copy = getattr(self, "acf_anchor_copy", False)
        if use_acf_memory and self.acf_lags:
            terms = self._build_static_acf_memory_terms(
                self.seq_len, self.pred_len, device=cache_device
            )
        else:
            terms = (
                torch.empty(0, device=cache_device),
                torch.empty(0, device=cache_device),
                torch.empty(0, device=cache_device),
                torch.empty(0, dtype=torch.long, device=cache_device),
                torch.empty(0, dtype=torch.bool, device=cache_device),
            )
        (
            self.acf_memory_static_coefficients,
            self.acf_memory_static_step_reliability,
            self.acf_memory_static_aggregate_reliability,
            self.acf_memory_static_indices,
            self.acf_memory_static_valid,
        ) = terms
        if use_acf_memory and self.acf_lags:
            self.acf_memory_static_channel_support = (
                self._build_static_acf_channel_support(
                    device=cache_device,
                    length=self.seq_len,
                    horizon=self.pred_len,
                )
            )
        else:
            self.acf_memory_static_channel_support = torch.empty(
                0, dtype=torch.bool, device=cache_device
            )
        if acf_anchor_copy and self.acf_lags:
            anchor_terms = self._build_static_acf_anchor_terms(
                self.seq_len, self.pred_len, device=cache_device
            )
        else:
            anchor_terms = (
                torch.empty(0, dtype=torch.long, device=cache_device),
                torch.empty(0, dtype=torch.long, device=cache_device),
                torch.empty(0, device=cache_device),
                torch.empty(0, device=cache_device),
                torch.empty(0, dtype=torch.bool, device=cache_device),
            )
        (
            self.acf_memory_static_anchor_indices,
            self.acf_memory_static_anchor_lags,
            self.acf_memory_static_anchor_sign,
            self.acf_memory_static_anchor_strength,
            self.acf_memory_static_anchor_valid,
        ) = anchor_terms
        if (
            self.acf_prior.numel()
            and self.acf_lags
            and (use_acf_memory or acf_anchor_copy)
        ):
            self._acf_memory_static_length = int(self.seq_len)
            self._acf_memory_static_horizon = int(self.pred_len)
            self._acf_memory_static_lags = tuple(self.acf_lags)
            self._acf_memory_static_temperature = float(self.acf_temperature)
            self._acf_memory_static_prior_version = self._acf_prior_tensor_version(
                self.acf_prior
            )
            self._acf_memory_static_prior_identity = id(self.acf_prior)
            self._acf_memory_static_channel_lag_bank = bool(
                getattr(self, "acf_channel_lag_bank", False)
            )
            self._acf_memory_static_channel_lag_top_k = int(
                getattr(self, "acf_channel_lag_top_k", 2)
            )
        else:
            self._acf_memory_static_length = None
            self._acf_memory_static_horizon = None
            self._acf_memory_static_lags = None
            self._acf_memory_static_temperature = None
            self._acf_memory_static_prior_version = None
            self._acf_memory_static_prior_identity = None
            self._acf_memory_static_channel_lag_bank = None
            self._acf_memory_static_channel_lag_top_k = None

    @staticmethod
    def _acf_prior_tensor_version(prior):
        """Read a tensor version without failing for inference tensors."""

        try:
            return int(prior._version)
        except (AttributeError, RuntimeError):
            # Inference tensors intentionally do not expose a version counter;
            # treating them as uncacheable keeps derived coefficients correct.
            return None

    def _acf_local_agreement_support(self, length=None):
        """Return local-ACF lags and pair support for an observed window.

        A lag can be legal for a future memory lookup even when the current
        encoder window is too short to estimate its correlation.  Keep those
        two notions separate: CLAM only uses lags with at least the configured
        number of observed pairs.  The return value is batch-independent and
        therefore also suitable for the read-only resource profile.
        """

        length = int(self.seq_len if length is None else length)
        if length <= 0:
            return (), 0
        min_pairs = int(getattr(self, "acf_local_min_pairs", 16))
        eligible = tuple(
            int(lag)
            for lag in self.acf_lags
            if 0 < int(lag) < length
            and (length - int(lag)) >= min_pairs
        )
        pair_count = sum(length - lag for lag in eligible)
        return eligible, int(pair_count)

    def _acf_local_agreement_values(self, x):
        """Compute signed local ACF agreement with the static train prior.

        ``x`` is the observed, normalized encoder window.  For each eligible
        lag ``k``, the returned value is ``sign(P_k) * corr_x(k)`` multiplied
        by a pair-count confidence factor.  The prior magnitude still defines
        the baseline softmax support in :meth:`_static_acf_memory`; local
        evidence is only allowed to reinforce or suppress that support.

        The helper returns ``None`` when no lag has enough pairs, allowing the
        caller to execute the historical static-memory loop exactly.  A
        float32 accumulator is used for half/bfloat16 inputs to avoid a noisy
        denominator on high-dimensional data, then the result is cast back to
        the input dtype.
        """

        if not torch.is_tensor(x) or x.ndim != 3:
            raise ValueError("local ACF agreement expects x with shape [B,T,C]")
        batch, length, channels = x.shape
        if channels != self.channels:
            raise ValueError(
                "local ACF agreement input has {} channels; model expects {}".format(
                    channels, self.channels
                )
            )
        eligible_lags, pair_count = self._acf_local_agreement_support(length)
        if not eligible_lags:
            return None, torch.zeros(
                len(self.acf_lags), dtype=torch.bool, device=x.device
            ), 0
        if self.acf_prior.numel() == 0:
            return None, torch.zeros(
                len(self.acf_lags), dtype=torch.bool, device=x.device
            ), 0
        if self.acf_prior.ndim != 2 or tuple(self.acf_prior.shape) != (
            self.channels, len(self.acf_lags)
        ):
            raise RuntimeError(
                "attached ACF prior does not align with local agreement lags"
            )

        work = x if x.dtype in (torch.float32, torch.float64) else x.float()
        centered = work - work.mean(dim=1, keepdim=True)
        prior = self.acf_prior.to(device=x.device, dtype=work.dtype)
        prior_sign = prior.sign()
        eligible_set = set(eligible_lags)
        correlations = []
        eligible_mask = torch.zeros(
            len(self.acf_lags), dtype=torch.bool, device=x.device
        )
        for lag_index, lag_value in enumerate(self.acf_lags):
            lag = int(lag_value)
            if lag not in eligible_set:
                correlations.append(work.new_zeros((batch, channels)))
                continue
            eligible_mask[lag_index] = True
            left = centered[:, :-lag, :]
            right = centered[:, lag:, :]
            numerator = (left * right).sum(dim=1)
            denominator = torch.sqrt(
                (left.square().sum(dim=1) * right.square().sum(dim=1))
                .clamp_min(1e-8)
            )
            correlation = (numerator / denominator).clamp(-1.0, 1.0)
            # A short, but legal, overlap should have less influence than a
            # full-window estimate.  This factor is deterministic and uses no
            # target/future values.
            confidence = math.sqrt(float(length - lag) / float(length))
            correlations.append(
                correlation * prior_sign[:, lag_index].unsqueeze(0) * confidence
            )
        agreement = torch.stack(correlations, dim=-1)
        return agreement.to(dtype=x.dtype), eligible_mask, int(pair_count)

    def _acf_reliability_interpolation_amplitude(
        self, step_reliability, aggregate_reliability, *, dtype, device
    ):
        """Return a bounded per-step/channel strength interpolation.

        ``step_reliability`` is ``[H,C]`` and ``aggregate_reliability`` is
        ``[C]``.  Both are detached train-prior statistics.  Rows with no
        legal/supporting lag are assigned a neutral log-ratio so the option
        cannot amplify an all-invalid lookup.  The learned scalar is bounded
        by ``tanh`` and starts at zero, making the returned factor exactly one
        before optimization opens the branch.
        """

        if getattr(self, "acf_reliability_interpolation_scale", None) is None:
            return None
        step = step_reliability.to(device=device, dtype=torch.float32)
        aggregate = aggregate_reliability.to(device=device, dtype=torch.float32)
        if step.ndim != 2 or aggregate.ndim != 1:
            raise RuntimeError(
                "ACF reliability interpolation expects step [H,C] and "
                "aggregate [C] tensors"
            )
        if step.shape[-1] != aggregate.shape[-1]:
            raise RuntimeError(
                "ACF reliability interpolation channel dimensions do not match"
            )
        eps = 1e-4
        aggregate = aggregate.view(1, -1)
        supported = (step > eps) | (aggregate > eps)
        log_ratio = torch.log((step + eps) / (aggregate + eps))
        log_ratio = torch.where(
            supported,
            log_ratio.clamp(-math.log(2.0), math.log(2.0)),
            torch.zeros_like(log_ratio),
        )
        scale = torch.tanh(
            self.acf_reliability_interpolation_scale.to(
                device=device, dtype=torch.float32
            )
        )
        amplitude = torch.exp(scale * log_ratio).to(dtype=dtype)
        self._set_acf_runtime_diagnostic(
            "acf_reliability_interpolation_mean", amplitude
        )
        self._set_acf_runtime_diagnostic(
            "acf_reliability_interpolation_std",
            amplitude.float().std(unbiased=False),
        )
        self._set_acf_runtime_diagnostic(
            "acf_reliability_interpolation_min", amplitude.min()
        )
        self._set_acf_runtime_diagnostic(
            "acf_reliability_interpolation_max", amplitude.max()
        )
        return amplitude

    def _static_acf_memory(self, x, horizon):
        """Apply cached train-prior coefficients with a lag-wise accumulation."""

        if self.acf_prior.numel() == 0:
            return None, None
        length = int(x.shape[1])
        horizon = int(horizon)
        prior_version = self._acf_prior_tensor_version(self.acf_prior)
        # Setters refresh eagerly, but a caller can still mutate the public
        # prior buffer in-place.  Rebuild the model-shape cache immediately so
        # later calls do not repeatedly take the slow uncached path.
        if (
            length == int(self.seq_len)
            and horizon == int(self.pred_len)
            and (
                self._acf_memory_static_length != length
                or self._acf_memory_static_horizon != horizon
                or self._acf_memory_static_lags != tuple(self.acf_lags)
                or self._acf_memory_static_temperature != float(self.acf_temperature)
                or self._acf_memory_static_prior_version != prior_version
                or getattr(self, "_acf_memory_static_prior_identity", None)
                != id(self.acf_prior)
                or getattr(self, "_acf_memory_static_channel_lag_bank", None)
                != bool(getattr(self, "acf_channel_lag_bank", False))
                or getattr(self, "_acf_memory_static_channel_lag_top_k", None)
                != int(getattr(self, "acf_channel_lag_top_k", 2))
                or prior_version is None
            )
        ):
            self._refresh_static_acf_memory_cache(device=x.device)
        cached = (
            self.acf_memory_static_coefficients.ndim == 3
            and tuple(self.acf_memory_static_coefficients.shape)
            == (horizon, self.channels, len(self.acf_lags))
            and tuple(self.acf_memory_static_indices.shape)
            == (horizon, len(self.acf_lags))
            and self.acf_memory_static_indices.device == x.device
            and self._acf_memory_static_length == length
            and self._acf_memory_static_horizon == horizon
            and self._acf_memory_static_lags == tuple(self.acf_lags)
            and self._acf_memory_static_temperature == float(self.acf_temperature)
            and self._acf_memory_static_prior_version
            == prior_version
            and getattr(self, "_acf_memory_static_prior_identity", None)
            == id(self.acf_prior)
            and (
                not bool(getattr(self, "acf_channel_lag_bank", False))
                or (
                    self.acf_memory_static_channel_support.ndim == 2
                    and tuple(self.acf_memory_static_channel_support.shape)
                    == (self.channels, len(self.acf_lags))
                    and self.acf_memory_static_channel_support.device == x.device
                )
            )
            and getattr(self, "_acf_memory_static_channel_lag_bank", None)
            == bool(getattr(self, "acf_channel_lag_bank", False))
            and getattr(self, "_acf_memory_static_channel_lag_top_k", None)
            == int(getattr(self, "acf_channel_lag_top_k", 2))
            and prior_version is not None
        )
        if cached:
            coefficients = self.acf_memory_static_coefficients.to(
                device=x.device, dtype=x.dtype
            )
            indices = self.acf_memory_static_indices
            valid = self.acf_memory_static_valid
            step_reliability = self.acf_memory_static_step_reliability.to(
                device=x.device, dtype=x.dtype
            )
            aggregate_reliability = self.acf_memory_static_aggregate_reliability.to(
                device=x.device, dtype=x.dtype
            )
            channel_support = self.acf_memory_static_channel_support
        else:
            (
                coefficients,
                step_reliability,
                aggregate_reliability,
                indices,
                valid,
            ) = self._build_static_acf_memory_terms(length, horizon, device=x.device)
            channel_support = self._build_static_acf_channel_support(
                device=x.device, length=length, horizon=horizon
            )
            coefficients = coefficients.to(dtype=x.dtype)
            step_reliability = step_reliability.to(dtype=x.dtype)
            aggregate_reliability = aggregate_reliability.to(dtype=x.dtype)

        memory_source = self._memory_signal(x)
        calibration_amplitudes = None
        if getattr(self, "acf_horizon_calibration_enabled", False):
            if getattr(self, "acf_horizon_calibration", None) is None:
                raise RuntimeError(
                    "horizon calibration is enabled but its adapter is missing"
                )
            if len(self.acf_lags) != self.acf_horizon_calibration.lag_count:
                raise RuntimeError(
                    "horizon calibration lag slots do not match resolved ACF lags"
                )
            calibration_amplitudes = self.acf_horizon_calibration.amplitudes(
                horizon=horizon, device=x.device, dtype=x.dtype
            )
        local_agreement = None
        if (
            getattr(self, "acf_local_agreement_requested", False)
            and self.use_acf_memory
            and not self.dynamic_acf
        ):
            # This helper only inspects the observed encoder window.  A
            # missing/insufficient lag set returns ``None`` and deliberately
            # leaves the historical coefficient path untouched.
            local_agreement, _, _ = self._acf_local_agreement_values(x)
        seasonal = x.new_zeros((x.shape[0], horizon, self.channels))
        # Keeping the lag axis out of the live tensor graph is the main
        # high-dimensional memory/latency optimization.  Gradients still flow
        # through each indexed historical value into ``x``.
        if local_agreement is None:
            # Keep the disabled/no-support path byte-for-byte equivalent to
            # the established static train-prior memory implementation.
            for lag_index in range(len(self.acf_lags)):
                values = memory_source.index_select(1, indices[:, lag_index])
                contribution = values * coefficients[:, :, lag_index].unsqueeze(0)
                if calibration_amplitudes is not None:
                    contribution = contribution * calibration_amplitudes[
                        :, lag_index
                    ].view(1, horizon, 1)
                seasonal = seasonal + contribution
        else:
            # Reweight only the lag probabilities.  ``coefficients`` contains
            # the signed baseline factors, but ``abs(coefficients)`` is not a
            # valid unsigned support when a prior slot is exactly zero: the
            # masked softmax still assigns that slot a small mass via
            # ``clamp_min(1e-6)``.  Rebuild the unsigned baseline weights from
            # the prior and causal validity mask so a zero ReZero scale is
            # strictly nested in the historical static-memory path.
            # Keep the same native prior dtype used by
            # ``_build_static_acf_memory_terms`` for the softmax, then cast
            # once to the live input dtype.  Computing this in fp16/bfloat16
            # would make the zero-ReZero nested path diverge under AMP.
            prior = self.acf_prior.to(device=x.device)
            valid = valid.to(device=x.device)
            channel_support = self._build_static_acf_channel_support(
                device=x.device, length=length, horizon=horizon
            )
            logits = torch.log(prior.abs().clamp_min(1e-6)).unsqueeze(0)
            logits = logits / float(self.acf_temperature)
            if channel_support.numel():
                effective_valid = valid[:, None, :] & channel_support[None, :, :]
                logits = logits.masked_fill(
                    ~effective_valid, torch.finfo(logits.dtype).min
                )
                valid_weight = effective_valid.to(dtype=prior.dtype)
            else:
                logits = logits.masked_fill(
                    ~valid[:, None, :], torch.finfo(logits.dtype).min
                )
                valid_weight = valid[:, None, :].to(dtype=prior.dtype)
            base_weights = torch.softmax(logits, dim=-1) * valid_weight
            base_weights = base_weights / base_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-4)
            base_weights = base_weights.to(dtype=x.dtype)
            local_scale = torch.tanh(self.acf_local_agreement_scale).to(
                device=x.device, dtype=x.dtype
            )
            # The same bounded agreement factor is used for both the
            # denominator and the seasonal accumulation.  Cache it once so
            # CLAM does not launch a second per-lag exponential while keeping
            # the memory-saving lag-wise accumulation below.
            local_factors = torch.exp(
                (local_scale * local_agreement).clamp(-4.0, 4.0)
            )
            denominator = x.new_zeros((x.shape[0], horizon, self.channels))
            for lag_index in range(len(self.acf_lags)):
                factor = local_factors[:, :, lag_index]
                mass = base_weights[:, :, lag_index].unsqueeze(0) * factor[:, None, :]
                denominator = denominator + mass

            for lag_index in range(len(self.acf_lags)):
                values = memory_source.index_select(1, indices[:, lag_index])
                factor = local_factors[:, :, lag_index]
                mass = base_weights[:, :, lag_index].unsqueeze(0) * factor[:, None, :]
                weights = mass / denominator.clamp_min(1e-6)
                contribution = values * coefficients[:, :, lag_index].sign().unsqueeze(0)
                contribution = contribution * weights
                if calibration_amplitudes is not None:
                    contribution = contribution * calibration_amplitudes[
                        :, lag_index
                    ].view(1, horizon, 1)
                seasonal = seasonal + contribution
        if getattr(self, "acf_reliability_interpolation", False):
            amplitude = self._acf_reliability_interpolation_amplitude(
                step_reliability,
                aggregate_reliability,
                dtype=seasonal.dtype,
                device=seasonal.device,
            )
            if amplitude is not None:
                # Keep the reliability gate itself unchanged; this adapter
                # only adjusts the absolute seasonal amplitude available at a
                # particular forecast step and channel.
                seasonal = seasonal * amplitude.unsqueeze(0)
        if self.acf_step_reliability:
            reliability = step_reliability.unsqueeze(0)
        else:
            reliability = aggregate_reliability.unsqueeze(0).unsqueeze(1)
        if calibration_amplitudes is not None:
            self._set_acf_runtime_diagnostic(
                "acf_calibration_amp_mean", calibration_amplitudes.mean()
            )
            self._set_acf_runtime_diagnostic(
                "acf_calibration_amp_std", calibration_amplitudes.std(unbiased=False)
            )
            self._set_acf_runtime_diagnostic(
                "acf_calibration_amp_min", calibration_amplitudes.min()
            )
            self._set_acf_runtime_diagnostic(
                "acf_calibration_amp_max", calibration_amplitudes.max()
            )
        self._set_acf_runtime_diagnostic("acf_seasonal_rms", seasonal)
        self._set_acf_runtime_diagnostic("acf_reliability_mean", reliability)
        return seasonal, reliability

    def _static_acf_anchor_memory(self, x, horizon):
        """Retrieve a causal signed anchor for each step and channel.

        Unlike the soft memory path, this expert does not average lags or
        reuse the shared ACF gate.  The compact train-only prior chooses the
        available lag with the largest absolute correlation independently for
        every channel; the observed value at that lag is then multiplied by
        the prior sign.  Invalid positions are zeroed explicitly.
        """

        if self.acf_prior.numel() == 0 or not self.acf_lags:
            return None, None, None, None
        if getattr(self, "dynamic_acf", False):
            raise RuntimeError(
                "direct ACF anchor retrieval requires a static train-only prior"
            )
        if getattr(self, "acf_source", "raw") != "raw":
            raise RuntimeError(
                "direct ACF anchor retrieval requires AutoCon_acf_source=raw"
            )
        length = int(x.shape[1])
        horizon = int(horizon)
        prior_version = self._acf_prior_tensor_version(self.acf_prior)
        cached = (
            getattr(self, "acf_memory_static_anchor_indices", torch.empty(0)).ndim == 2
            and tuple(self.acf_memory_static_anchor_indices.shape)
            == (horizon, self.channels)
            and self.acf_memory_static_anchor_indices.device == x.device
            and getattr(self, "_acf_memory_static_length", None) == length
            and getattr(self, "_acf_memory_static_horizon", None) == horizon
            and getattr(self, "_acf_memory_static_lags", None) == tuple(self.acf_lags)
            and getattr(self, "_acf_memory_static_prior_version", None)
            == prior_version
            and getattr(self, "_acf_memory_static_prior_identity", None)
            == id(self.acf_prior)
            and prior_version is not None
        )
        if not cached:
            # A prior may have been attached before a model/device move.  In
            # that case refresh the non-persistent lookup tables once and use
            # them for all subsequent forwards instead of rebuilding the
            # horizon-by-channel selection on every call.
            if (
                length == int(self.seq_len)
                and horizon == int(self.pred_len)
                and prior_version is not None
            ):
                self._refresh_static_acf_memory_cache(device=x.device)
                cached = (
                    getattr(self, "acf_memory_static_anchor_indices", torch.empty(0)).ndim == 2
                    and tuple(self.acf_memory_static_anchor_indices.shape)
                    == (horizon, self.channels)
                    and self.acf_memory_static_anchor_indices.device == x.device
                    and getattr(self, "_acf_memory_static_length", None) == length
                    and getattr(self, "_acf_memory_static_horizon", None) == horizon
                    and getattr(self, "_acf_memory_static_lags", None) == tuple(self.acf_lags)
                    and getattr(self, "_acf_memory_static_prior_version", None)
                    == prior_version
                    and getattr(self, "_acf_memory_static_prior_identity", None)
                    == id(self.acf_prior)
                )
        if cached:
            anchor_indices = self.acf_memory_static_anchor_indices
            anchor_lags = self.acf_memory_static_anchor_lags
            anchor_sign = self.acf_memory_static_anchor_sign.to(
                device=x.device, dtype=x.dtype
            )
            anchor_strength = self.acf_memory_static_anchor_strength.to(
                device=x.device, dtype=x.dtype
            )
            anchor_valid = self.acf_memory_static_anchor_valid
        else:
            (
                anchor_indices,
                anchor_lags,
                anchor_sign,
                anchor_strength,
                anchor_valid,
            ) = self._build_static_acf_anchor_terms(
                length, horizon, device=x.device
            )
            anchor_sign = anchor_sign.to(dtype=x.dtype)
            anchor_strength = anchor_strength.to(dtype=x.dtype)

        if anchor_indices.numel() == 0:
            return None, None, None, None
        batch = int(x.shape[0])
        gather_index = anchor_indices.to(device=x.device).unsqueeze(0).expand(
            batch, -1, -1
        )
        # ``gather`` keeps the channel axis intact and avoids materializing a
        # [B,H,C,K] tensor, which matters for high-dimensional Traffic input.
        values = x.gather(dim=1, index=gather_index)
        valid = anchor_valid.to(device=x.device, dtype=x.dtype)
        anchor = values * anchor_sign.unsqueeze(0) * valid.unsqueeze(0)
        return anchor, anchor_lags.to(device=x.device), anchor_strength, valid

    def _set_acf_channel_order(self, selected, lags):
        """Derive a deterministic variable-token order from the train prior.

        The dominant peak remains signed for audit and conditioning.  The
        ``strength`` ordering uses its absolute magnitude so equally strong
        anti-correlated and positively correlated channels receive the same
        periodicity priority.  Ties are resolved by peak lag and then input
        channel index, which makes the ordering deterministic across hardware
        and repeated calls.
        """

        if selected.ndim != 2 or selected.shape != (self.channels, lags.numel()):
            raise RuntimeError(
                "selected ACF prior must have shape [channels, selected_lags]"
            )
        if lags.numel() == 0:
            input_order = torch.arange(
                self.channels, device=selected.device, dtype=torch.long
            )
            self.acf_channel_order_indices = input_order
            self.acf_channel_inverse_indices = input_order.clone()
            self.acf_channel_dominant_peak = selected.new_zeros(self.channels)
            self.acf_channel_dominant_peak_lag = torch.zeros_like(input_order)
            return
        peak_indices = selected.abs().argmax(dim=-1)
        dominant_peak = selected.gather(
            dim=-1, index=peak_indices.unsqueeze(-1)
        ).squeeze(-1)
        dominant_lag = lags.gather(dim=0, index=peak_indices)
        input_order = torch.arange(
            self.channels, device=selected.device, dtype=torch.long
        )

        if self.acf_channel_order == "strength":
            # Python's stable sort avoids depending on CUDA's stable-sort
            # implementation.  This setup runs once per train-split prior,
            # not in the forward path.
            peaks = dominant_peak.detach().cpu().tolist()
            peak_lags = dominant_lag.detach().cpu().tolist()
            ordered_channels = sorted(
                range(self.channels),
                key=lambda index: (-abs(float(peaks[index])), int(peak_lags[index]), index),
            )
            order = torch.as_tensor(
                ordered_channels, device=selected.device, dtype=torch.long
            )
        elif self.acf_channel_order == "similarity":
            # Build the order once from the compact, train-only ACF profile.
            # A signed cosine keeps positive and negative phase relationships
            # distinguishable while avoiding a dense channel mixer in every
            # forward pass.  The shared helper uses O(C^2 K) vectorized dot
            # products and O(CK) temporary memory, with deterministic tie
            # handling shared by offline replay.
            ordered_channels = signed_acf_profile_similarity_order(
                selected.detach().to(device="cpu", dtype=torch.float32).numpy()
            ).tolist()
            order = torch.as_tensor(
                ordered_channels, device=selected.device, dtype=torch.long
            )
        else:
            order = input_order

        inverse = torch.empty_like(order)
        inverse[order] = input_order
        self.acf_channel_order_indices = order.detach()
        self.acf_channel_inverse_indices = inverse.detach()
        self.acf_channel_dominant_peak = dominant_peak.detach()
        self.acf_channel_dominant_peak_lag = dominant_lag.detach()

    def _train_prior_channel_order(self, device):
        """Return static ACF order and inverse, or ``None`` for identity.

        No runtime-window ACF is used here.  Before ``set_acf_prior`` has
        attached a train-split table, the ACF-derived modes deliberately fall
        back to the input order rather than deriving an order from
        validation/test windows.
        """

        if self.acf_channel_order not in {"strength", "similarity"} or self.acf_prior.numel() == 0:
            return None, None
        order = self.acf_channel_order_indices.to(device=device, dtype=torch.long)
        inverse = self.acf_channel_inverse_indices.to(
            device=device, dtype=torch.long
        )
        if order.numel() != self.channels or inverse.numel() != self.channels:
            raise RuntimeError("ACF channel order has an invalid channel count")
        expected = torch.arange(self.channels, device=device, dtype=torch.long)
        if not torch.equal(order.index_select(0, inverse), expected):
            raise RuntimeError("ACF channel order inverse is invalid")
        return order, inverse

    def _acf_channel_summary(self, device, dtype):
        """Return fixed-width signed ACF features for each input channel.

        The summary is derived solely from the train-split prior attached by
        :meth:`set_acf_prior`.  Keeping the width fixed makes the model robust
        to horizon-specific peak selection while retaining phase and peak-response
        information.  A missing prior produces zeros, so an
        explicitly enabled branch cannot accidentally use validation/test
        observations as a fallback.
        """

        if self.acf_prior.numel() == 0:
            return torch.zeros((self.channels, 4), device=device, dtype=dtype)
        prior = self.acf_prior.to(device=device, dtype=dtype)
        if prior.ndim != 2 or prior.shape[0] != self.channels:
            raise RuntimeError(
                "attached ACF prior must have shape [channels, selected_lags]"
            )
        signed_mean = prior.mean(dim=-1)
        abs_mean = prior.abs().mean(dim=-1)
        if self.acf_prior_lags.numel() == prior.shape[1]:
            lags = self.acf_prior_lags.to(device=device, dtype=dtype)
            lag_scale = lags / lags.max().clamp_min(1.0)
            lag_weighted_signed = (prior * lag_scale.unsqueeze(0)).mean(dim=-1)
        else:
            lag_weighted_signed = signed_mean
        peak_index = prior.abs().argmax(dim=-1, keepdim=True)
        peak_signed = prior.gather(dim=-1, index=peak_index).squeeze(-1)
        return torch.stack(
            (signed_mean, abs_mean, lag_weighted_signed, peak_signed), dim=-1
        )

    def _acf_scale_summary(self, x):
        """Return six per-window signed ACF descriptors for scale fusion.

        The compact train-only prior supplies lag weights/signs, while the
        current normalized encoder window supplies the correlation values.
        Consequently the descriptor varies across windows and cannot be
        reduced to a learned dataset-level bias.  Only observed positions in
        ``x`` are used; no decoder values or targets enter this path.
        """

        if not torch.is_tensor(x) or x.ndim != 3:
            raise ValueError("scale conditioning expects x with shape [B,T,C]")
        batch, length, channels = x.shape
        if self.acf_prior.numel() == 0:
            return x.new_zeros((batch, 6))
        # A local correlation needs at least one pair of observations.  Long
        # forecast lags may still be valid for the causal memory lookup, but
        # they cannot provide a current-window statistic and are masked here.
        min_pairs = int(getattr(self, "acf_scale_conditioning_min_pairs", 16))
        if not any(
            0 < int(lag) and int(length) - int(lag) >= min_pairs
            for lag in self.acf_lags
        ):
            return x.new_zeros((batch, 6))
        prior = self.acf_prior.to(device=x.device, dtype=x.dtype)
        if prior.ndim != 2 or tuple(prior.shape) != (
            self.channels, len(self.acf_lags)
        ) or channels != self.channels:
            raise RuntimeError(
                "attached ACF prior does not align with scale conditioning"
            )
        if self.acf_prior_lags.numel() == len(self.acf_lags):
            lags = self.acf_prior_lags.to(device=x.device, dtype=torch.long)
        else:
            lags = torch.as_tensor(
                self.acf_lags, device=x.device, dtype=torch.long
            )

        centered = x - x.mean(dim=1, keepdim=True)
        correlations = []
        # ``self.acf_lags`` is a Python tuple kept in the same canonical
        # order as ``lags``.  Iterating it avoids a device-to-host sync from
        # ``Tensor.tolist()`` on every CUDA forward.
        for lag_value in self.acf_lags:
            lag = int(lag_value)
            if lag <= 0 or length - lag < min_pairs:
                correlations.append(x.new_zeros((batch, channels)))
                continue
            left = centered[:, :-lag, :]
            right = centered[:, lag:, :]
            numerator = (left * right).sum(dim=1)
            denominator = torch.sqrt(
                (left.square().sum(dim=1) * right.square().sum(dim=1))
                .clamp_min(1e-8)
            )
            correlations.append((numerator / denominator).clamp(-1.0, 1.0))
        corr = torch.stack(correlations, dim=-1)  # [B,C,K]

        prior_abs = prior.abs()
        valid = (
            (lags > 0)
            & (lags + int(min_pairs) <= int(length))
        ).to(dtype=x.dtype).view(1, -1)
        weights = prior_abs * valid
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights.unsqueeze(0)  # [1,C,K]
        lag_norm = lags.to(dtype=x.dtype) / float(max(int(self.seq_len), 1))
        lag_norm = lag_norm.view(1, 1, -1)
        signed = (corr * weights).sum(dim=(1, 2))
        absolute = (corr.abs() * weights).sum(dim=(1, 2))
        prior_sign_agreement = (
            corr * prior.sign().unsqueeze(0) * weights
        ).sum(dim=(1, 2))
        lag_weighted_absolute = (
            corr.abs() * weights * lag_norm
        ).sum(dim=(1, 2))
        peak_abs = corr.abs().amax(dim=-1).mean(dim=-1)
        high_lag_activity = (
            corr.abs()
            * weights
            * (lags.to(dtype=x.dtype).view(1, 1, -1) >= float(self.seq_len) * 0.5)
        ).sum(dim=(1, 2))
        return torch.stack(
            (
                signed,
                absolute,
                prior_sign_agreement,
                lag_weighted_absolute,
                peak_abs,
                high_lag_activity,
            ),
            dim=-1,
        )

    def _acf_scale_bias(self, x):
        """Map per-window train-prior ACF descriptors to scale-logit bias."""

        if (
            self.acf_scale_conditioner is None
            or self.acf_prior.numel() == 0
            or len(self.scales) <= 1
            or not any(
                0 < int(lag)
                and int(x.shape[1]) - int(lag)
                >= int(getattr(self, "acf_scale_conditioning_min_pairs", 16))
                for lag in self.acf_lags
            )
        ):
            return None
        summary = self._acf_scale_summary(x)
        bias = self.acf_scale_conditioner(summary)
        scale = torch.tanh(self.acf_scale_conditioning_scale).to(
            device=x.device, dtype=x.dtype
        )
        return scale * bias

    def _acf_conditioned_variable_tokens(self, variable_tokens, channel_indices=None):
        """Inject train-only per-channel ACF context into variable tokens."""

        if self.acf_channel_conditioner is None:
            return variable_tokens
        summary = self._acf_channel_summary(
            variable_tokens.device, variable_tokens.dtype
        )
        if channel_indices is not None:
            summary = summary.index_select(0, channel_indices)
        if summary.shape[0] != variable_tokens.shape[1]:
            raise RuntimeError(
                "ACF channel summary does not align with variable token order"
            )
        context = self.acf_channel_conditioner(summary)
        scale = torch.tanh(self.acf_conditioning_scale).to(
            dtype=variable_tokens.dtype, device=variable_tokens.device
        )
        return variable_tokens + scale * context.unsqueeze(0)

    def _group_variable_tokens(self, variable_tokens):
        """Pool contiguous variable tokens into fixed-width group tokens.

        The returned tensor has shape ``[B, groups, D]``.  A final partial
        group is zero-padded only for the reshape and divided by its number of
        real channels, so padding cannot dilute its mean.  Grouping is kept in
        a separate helper to make the channel-count/cost contract auditable
        without executing a Mamba kernel.
        """

        if self.variable_group_size <= 1:
            return variable_tokens
        batch, channels, width = variable_tokens.shape
        expected_groups = int(math.ceil(channels / self.variable_group_size))
        if expected_groups != self.variable_group_count:
            raise RuntimeError(
                "variable token channel count changed from model construction"
            )
        padding = expected_groups * self.variable_group_size - channels
        if padding:
            variable_tokens = F.pad(variable_tokens, (0, 0, 0, padding))
        grouped = variable_tokens.reshape(
            batch, expected_groups, self.variable_group_size, width
        )
        counts = variable_tokens.new_full(
            (expected_groups,), float(self.variable_group_size)
        )
        if padding:
            counts[-1] = float(self.variable_group_size - padding)
        return grouped.sum(dim=2) / counts.view(1, -1, 1).clamp_min(1.0)

    def _run_variable_branch(self, variable_tokens):
        """Run the variable Mamba branch, optionally with grouped scans.

        For grouped mode, ACF-conditioned per-channel tokens have already
        entered ``variable_tokens``.  The group-level Mamba context is then
        broadcast back to each original channel and added to that channel's
        token before the existing per-channel forecast head.  This preserves
        channel-specific outputs while reducing the recurrent sequence length
        from ``C`` to ``ceil(C / group_size)``.
        """

        if self.variable_group_size <= 1:
            return self._run_branch(variable_tokens, self.variable_blocks)
        grouped = self._group_variable_tokens(variable_tokens)
        grouped = self._run_branch(grouped, self.variable_blocks)
        context = grouped.repeat_interleave(self.variable_group_size, dim=1)
        context = context[:, : variable_tokens.shape[1], :]
        return variable_tokens + context

    def _variable_token_features(self, x):
        """Encode the observed window with the shared variable-token branch.

        The returned ``[B,C,D]`` states are in the original channel order.
        All optional ordering/permutation operations are kept in this helper
        so forecast and representation paths can share exactly one stochastic
        token realization when representation fusion is enabled.
        """

        if not self.variable_mixer_enabled:
            return None
        if x.ndim != 3 or int(x.shape[1]) != self.seq_len:
            raise ValueError(
                "variable-token branch expects [B, seq_len, C] input"
            )
        _, _, channels = x.shape
        if int(channels) != self.channels:
            raise ValueError(
                "variable-token branch input has {} channels; expected {}".format(
                    int(channels), self.channels
                )
            )
        variable_tokens = self.variable_embedding(x.transpose(1, 2))
        # Apply the static train-ACF order before any optional random
        # permutation or grouped scan.  Restoring the original order below
        # makes the representation summary independent of token bookkeeping.
        channel_order, channel_inverse = self._train_prior_channel_order(
            variable_tokens.device
        )
        if channel_order is not None:
            variable_tokens = variable_tokens.index_select(1, channel_order)
        variable_tokens = self._acf_conditioned_variable_tokens(
            variable_tokens, channel_indices=channel_order
        )
        permutation = None
        if self.variable_permutation and self.training and channels > 1:
            permutation = torch.randperm(channels, device=x.device)
            variable_tokens = variable_tokens[:, permutation, :]
        variable_tokens = self._run_variable_branch(variable_tokens)
        # Normalize the variable-token states before the optional CMamba-lite
        # adapter.  Applying FiLM before LayerNorm would largely remove its
        # per-token scale/shift and make the advertised channel mixer nearly
        # inert; CMamba applies its GDD block after the Mamba normalization.
        variable_tokens = self.variable_norm(variable_tokens)
        if self.variable_gdd_mixer is not None:
            # CMamba-lite channel-preserving modulation.  The existing mixer
            # consumes an explicit patch axis; a singleton axis here denotes
            # one full-window token per variable.  Both the forecast head and
            # AutoCon representation consume the same post-normalization state.
            variable_tokens = self.variable_gdd_mixer(
                variable_tokens.unsqueeze(2)
            ).squeeze(2)
        if permutation is not None:
            inverse = torch.argsort(permutation)
            variable_tokens = variable_tokens[:, inverse, :]
        if channel_inverse is not None:
            variable_tokens = variable_tokens.index_select(1, channel_inverse)
        return variable_tokens

    def _reshape_channels(self, x):
        """Flatten [B,T,C] into independent [B*C,T,1] sequences."""

        batch, length, channels = x.shape
        return x.transpose(1, 2).reshape(batch * channels, length, 1)

    def _embed(self, x, x_mark):
        batch, length, channels = x.shape
        if not self.channel_independent:
            return self.embedding(x, x_mark)
        values = self._reshape_channels(x)
        if x_mark is None:
            marks = None
        else:
            marks = x_mark.unsqueeze(1).expand(batch, channels, length, x_mark.shape[-1])
            marks = marks.reshape(batch * channels, length, x_mark.shape[-1])
        return self.embedding(values, marks)

    @staticmethod
    def _pool_scale(x, scale):
        scale = int(scale)
        if scale <= 0:
            raise ValueError("pool scale must be positive")
        if scale == 1:
            return x
        length = int(x.shape[1])
        if length == 0:
            return x

        # Keep complete chunks on the historical avg_pool1d path so divisible
        # inputs retain their previous floating-point behavior exactly.  The
        # final chunk is handled explicitly: it may be shorter than ``scale``
        # and must therefore be averaged over real observations only.
        full_length = (length // scale) * scale
        if full_length:
            pooled = F.avg_pool1d(
                x[:, :full_length, :].transpose(1, 2),
                kernel_size=scale,
                stride=scale,
                count_include_pad=False,
            ).transpose(1, 2)
        else:
            pooled = x[:, :0, :]
        if full_length == length:
            return pooled
        tail = x[:, full_length:, :].mean(dim=1, keepdim=True)
        return torch.cat((pooled, tail), dim=1)

    @staticmethod
    def _pool_scale_marks(x_mark, scale):
        """Use the terminal real mark for every causal pooled chunk.

        A pooled value token summarizes observations in
        ``[i * scale, (i + 1) * scale)`` (with a possibly shorter final
        chunk).  Calendar/time features are not averaged because doing so can
        create invalid dates or fractional categorical values; the mark at
        the final observed index is the causal representative of that same
        chunk.  This is a source-contract change for the historical
        full-resolution multi-scale path, which used the first mark.
        """

        scale = int(scale)
        if scale <= 0:
            raise ValueError("pool scale must be positive")
        if x_mark is None or scale == 1:
            return x_mark
        length = int(x_mark.shape[1])
        if length < 1:
            return x_mark
        sampled = x_mark[:, scale - 1::scale, :]
        expected = (length + scale - 1) // scale
        if sampled.shape[1] < expected:
            sampled = torch.cat([sampled, x_mark[:, -1:, :]], dim=1)
        return sampled

    @staticmethod
    def _sample_marks(x_mark, scale):
        """Backward-compatible alias for terminal pooled-mark selection."""

        return _MambaACFBase._pool_scale_marks(x_mark, scale)

    @staticmethod
    def _token_stride_marks(x_mark, stride):
        """Select terminal real marks for causal base-stride chunks."""

        return _MambaACFBase._pool_scale_marks(x_mark, stride)

    def _run_branch(self, hidden, blocks):
        original = hidden
        for block in blocks:
            forward = hidden
            for_block = forward
            for_block = block(for_block)
            hidden = for_block
        if self.bidirectional:
            # Both directions must consume the same pre-block input.  Reusing
            # the already transformed forward state would apply the blocks a
            # second time and is not a bidirectional encoder.
            reverse = torch.flip(original, dims=[1])
            for block in blocks:
                reverse = block(reverse)
            hidden = (hidden + torch.flip(reverse, dims=[1])) / math.sqrt(2.0)
        return hidden

    @staticmethod
    def _resize_temporal(hidden, target_length):
        """Resize a ``[batch, time, width]`` state only when necessary."""

        target_length = int(target_length)
        if hidden.shape[1] == target_length:
            return hidden
        return F.interpolate(
            hidden.transpose(1, 2),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _encode_channels(self, x, x_mark, *, return_base=False, return_full=True):
        """Encode the window and optionally retain the compressed forecast state.

        ``return_full`` controls whether a representation on the original
        encoder grid is materialized.  Forecasting can use the base token grid
        directly; AutoCon still receives a full-grid representation for
        compatibility with the existing relation-loss labels.
        """

        batch, length, channels = x.shape
        if length != self.seq_len:
            raise ValueError(
                "MambaACF expects seq_len={} input steps, received {}".format(
                    self.seq_len, length
                )
            )
        # Keep the observed signal used for the local ACF descriptor before
        # any learned cross-variable channel gate.  The train prior is fitted
        # from the raw/normalized source family, so a gated mixture would
        # silently change the meaning of ``AutoCon_acf_source=raw``.
        acf_scale_window = x
        if self.channel_gate is not None:
            acf_summary = None
            if self.acf_channel_gate_enabled and self.acf_prior.numel():
                # This summary is in the original input-channel order.  It is
                # deliberately independent of the optional variable-token
                # ordering, because the gate acts before token construction.
                acf_summary = self._acf_channel_summary(x.device, x.dtype)
            x = self.channel_gate(
                x.unsqueeze(-1), acf_summary=acf_summary
            ).squeeze(-1)
        # Tokenize the observed window once before all Mamba branches.  The
        # pooling never reads beyond the encoder window, and each pooled mark
        # denotes the final observed timestamp of its chunk.
        base_x = self._pool_scale(x, self.mamba_token_stride)
        base_length = base_x.shape[1]
        branches = []
        for scale, blocks in zip(self.scales, self.branches):
            # Aggregate every branch directly from the original observations.
            # Pooling an already pooled tail would give a partial base chunk
            # the same weight as a full chunk when it lands inside an outer
            # scale group (for example, L=105, stride=8, scale=2).
            effective_stride = int(self.mamba_token_stride) * int(scale)
            scaled_x = self._pool_scale(x, effective_stride)
            # Marks follow the same direct chunk boundaries as values.  The
            # terminal real mark remains causal for both default and strided
            # paths.
            scaled_mark = self._pool_scale_marks(x_mark, effective_stride)
            hidden = self._run_branch(self._embed(scaled_x, scaled_mark), blocks)
            # Multi-scale fusion only needs the common compressed grid.  The
            # old implementation restored every branch to ``length`` here,
            # which made stride reduce the scan but add a full-resolution
            # interpolation for every branch.
            hidden = self._resize_temporal(hidden, base_length)
            branches.append(hidden)

        stacked = torch.stack(branches, dim=-1)
        acf_scale_bias = self._acf_scale_bias(acf_scale_window)
        if self.channel_independent:
            # [B*C,T_base,D,S] -> [B,C,T_base,D]
            gate_input = stacked.mean(dim=-1).mean(dim=1)
            scale_logits = self.scale_gate(gate_input)
            if acf_scale_bias is not None:
                # Independent channels are flattened in B,C order by
                # ``_reshape_channels``.  Each window-level ACF bias is
                # repeated for its C independent streams.
                scale_logits = scale_logits + acf_scale_bias.repeat_interleave(
                    channels, dim=0
                )
            weights = torch.softmax(scale_logits, dim=-1)
            fused_base = (stacked * weights[:, None, None, :]).sum(dim=-1)
            fused_base = self.fused_norm(fused_base).reshape(
                batch, channels, base_length, self.d_model
            )
            if return_full and base_length != length:
                flat = fused_base.reshape(
                    batch * channels, base_length, self.d_model
                )
                full = self._resize_temporal(flat, length).reshape(
                    batch, channels, length, self.d_model
                )
            else:
                full = fused_base
        else:
            # Shared mode: [B,T_base,D,S], with one scale mixture per window.
            gate_input = stacked.mean(dim=-1).mean(dim=1)
            scale_logits = self.scale_gate(gate_input)
            if acf_scale_bias is not None:
                scale_logits = scale_logits + acf_scale_bias
            weights = torch.softmax(scale_logits, dim=-1)
            fused_base = (stacked * weights[:, None, None, :]).sum(dim=-1)
            fused_base = self.fused_norm(fused_base)
            full = (
                self._resize_temporal(fused_base, length)
                if return_full and base_length != length
                else fused_base
            )

        # CMamba-inspired channel-state modulation.  The adapter reuses the
        # shared scale-gate summary computed from the observed window, but only
        # its ReZero residual is added to the shared temporal state.  This keeps
        # the historical temporal encoder intact at initialization and,
        # importantly, makes the exact same mixed state available to both
        # forecasting and AutoCon representation paths.  The branch is
        # shared-mode only.
        fused_with_channel_state = self._apply_data_dependent_channel_mixer(
            x, fused_base, summary=gate_input
        )
        if fused_with_channel_state is not fused_base:
            fused_base = fused_with_channel_state
            if return_full:
                full = (
                    self._resize_temporal(fused_base, length)
                    if base_length != length
                    else fused_base
                )

        if return_base:
            return fused_base, full if return_full else None
        return full if return_full else fused_base

    def _apply_data_dependent_channel_mixer(self, x, hidden, *, summary=None):
        """Apply the shared global channel-state residual to hidden.

        x is the normalized observed window [B,T,C] and hidden is the fused
        temporal state [B,T,D].  The production path supplies the already
        computed [B,D] scale-gate summary; a direct call without ``summary``
        falls back to the module's mean/RMS descriptor.  It never reads target
        or validation values and returns the original object when disabled so
        the historical path remains allocation-free.
        """

        mixer = self.data_dependent_channel_mixer
        if mixer is None:
            return hidden
        if x.ndim != 3 or int(x.shape[1]) != self.seq_len:
            raise ValueError(
                "data-dependent channel mixer expects [B, seq_len, C] input"
            )
        if int(x.shape[2]) != self.channels:
            raise ValueError(
                "data-dependent channel mixer input has {} channels; expected {}".format(
                    int(x.shape[2]), self.channels
                )
            )
        if summary is None:
            return mixer(x, hidden)
        return mixer.forward_from_summary(summary, hidden)

    def _data_dependent_channel_residual(self, x):
        """Compatibility helper returning the residual for a zero state.

        Older contract probes call this method directly.  The production path
        uses _apply_data_dependent_channel_mixer so the residual is applied to
        the already-fused hidden state without allocating a second
        channel-token tensor.
        """

        mixer = self.data_dependent_channel_mixer
        if mixer is None:
            return None
        zero_hidden = x.new_zeros((x.shape[0], 1, self.d_model))
        # The compatibility path has no fused hidden summary.  Use a
        # deterministic mean summary padded/truncated to the configured width;
        # the production path always reuses the exact scale-gate summary.
        summary = x.mean(dim=1)
        if int(summary.shape[1]) < mixer.summary_dim:
            summary = F.pad(summary, (0, mixer.summary_dim - int(summary.shape[1])))
        elif int(summary.shape[1]) > mixer.summary_dim:
            summary = summary[:, :mixer.summary_dim]
        return self._apply_data_dependent_channel_mixer(
            x, zero_hidden, summary=summary
        )

    def _patch_forecast(self, x):
        """Forecast with channel-preserving temporal patch tokens.

        The branch is deliberately separate from the shared time-state path.
        It keeps one representation per input variable, while the Mamba scan
        is performed once over the explicitly ordered patch-token sequence.
        All values come from the normalized encoder window; right padding, if
        needed for the final patch, repeats the last observed value.
        """

        if not self.patch_mixer_enabled:
            return None
        batch, length, channels = x.shape
        values = x.transpose(1, 2)
        if self.patch_right_pad:
            values = F.pad(values, (0, self.patch_right_pad), mode="replicate")
        patches = values.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.patch_stride,
        )
        # [B,C,P,W] -> [B,C,P,D].
        tokens = self.patch_embedding(patches)
        permutation = None
        if self.patch_permutation and self.training and channels > 1:
            permutation = torch.randperm(channels, device=x.device)
            tokens = tokens[:, permutation, :, :]
        if self.patch_order == "time_major":
            tokens = tokens.permute(0, 2, 1, 3).reshape(
                batch, self.patch_count * channels, self.d_model
            )
        else:
            tokens = tokens.reshape(
                batch, channels * self.patch_count, self.d_model
            )
        tokens = self._run_branch(tokens, self.patch_blocks)
        tokens = self.patch_norm(tokens)
        if self.patch_order == "time_major":
            tokens = tokens.reshape(
                batch, self.patch_count, channels, self.d_model
            ).permute(0, 2, 1, 3)
        else:
            tokens = tokens.reshape(
                batch, channels, self.patch_count, self.d_model
            )
        if permutation is not None:
            inverse = torch.argsort(permutation)
            tokens = tokens[:, inverse, :, :]

        channel_tokens = tokens.reshape(
            batch, channels, self.patch_count * self.d_model
        )
        prediction = self.patch_head(channel_tokens).transpose(1, 2)
        weight = torch.sigmoid(
            self.patch_gate(tokens.mean(dim=2))
        ).transpose(1, 2).expand(-1, self.pred_len, -1)
        scale = torch.tanh(self.patch_residual_scale).to(
            dtype=prediction.dtype, device=prediction.device
        )
        return scale * weight * prediction

    def _local_patch_features(self, x):
        """Encode short temporal patches while preserving channel identity.

        ``x`` is normalized and has shape ``[B, T, C]``.  Patches are formed
        independently for each channel and processed by the same small block;
        the returned tensor is ``[B, C, P, D_local]``.  No target, decoder
        value, or cross-channel dense map enters this pathway.
        """

        if not self.local_mixer_enabled:
            return None
        batch, length, channels = x.shape
        values = x.transpose(1, 2)
        if self.local_right_pad:
            values = F.pad(values, (0, self.local_right_pad), mode="replicate")
        patches = values.unfold(
            dimension=-1,
            size=self.local_patch_len,
            step=self.local_patch_stride,
        )
        tokens = self.local_patch_embedding(patches)
        flat_tokens = tokens.reshape(
            batch * channels, self.local_patch_count, self.local_width
        )
        for block in self.local_blocks:
            flat_tokens = block(flat_tokens)
        tokens = self.local_norm(flat_tokens).reshape(
            batch, channels, self.local_patch_count, self.local_width
        )
        return tokens

    def _local_forecast(self, x, local_tokens=None):
        """Return a gated channel-preserving local forecast correction."""

        if not self.local_mixer_enabled:
            return None
        if local_tokens is None:
            local_tokens = self._local_patch_features(x)
        batch, channels, _, _ = local_tokens.shape
        flattened = local_tokens.reshape(batch, channels, -1)
        prediction = self.local_head(flattened).transpose(1, 2)
        weight = torch.sigmoid(
            self.local_gate(local_tokens.mean(dim=2))
        ).transpose(1, 2).expand(-1, self.pred_len, -1)
        scale = torch.tanh(self.local_residual_scale).to(
            dtype=prediction.dtype, device=prediction.device
        )
        return scale * weight * prediction

    def _local_representation_context(self, local_tokens):
        """Build the shared local summary once for all representation grids."""

        if not self.local_mixer_enabled or not self.AutoCon:
            return None
        context = local_tokens.mean(dim=(1, 2))
        return self.local_repr_projection(context).unsqueeze(1)

    def _fuse_local_representation(self, hidden, local_tokens, context=None):
        """Inject a channel-preserving local summary into AutoCon's state."""

        # The adapter is only meaningful for AutoCon; plain forecasting keeps
        # the local branch exclusively on the forecast path.
        if not self.local_mixer_enabled or not self.AutoCon:
            return hidden
        if hidden.ndim != 3:
            raise ValueError(
                "mamba_local_mixer requires shared hidden states with shape [B,T,D]"
            )
        if context is None:
            context = self._local_representation_context(local_tokens)
        scale = torch.tanh(self.local_repr_scale).to(
            dtype=hidden.dtype, device=hidden.device
        )
        return hidden + scale * context

    def _fuse_variable_representation(self, hidden, variable_tokens):
        """Inject variable-token states into the AutoCon representation grid.

        This adapter is representation-only: the forecasting branch already
        consumes the same variable-token states, while the returned summary is
        broadcast over the temporal grid used by the relation objective.  A
        small near-zero gate keeps the opt-in branch stable at startup while
        retaining a non-zero first-step gradient through the channel pooling,
        projection, and complete variable Mamba stack.
        """

        if not self.variable_repr_fusion_enabled:
            return hidden
        if variable_tokens is None:
            raise RuntimeError(
                "variable representation fusion requires variable-token states"
            )
        if variable_tokens.ndim != 3 or hidden.ndim != 3:
            raise ValueError(
                "variable representation fusion expects hidden [B,T,D] and "
                "variable tokens [B,C,D]"
            )
        if (
            int(variable_tokens.shape[0]) != int(hidden.shape[0])
            or int(variable_tokens.shape[2]) != self.d_model
        ):
            raise RuntimeError(
                "variable-token representation dimensions do not match hidden grid"
            )
        scores = self.variable_repr_pool(variable_tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        summary = (variable_tokens * weights).sum(dim=1, keepdim=True)
        context = self.variable_repr_projection(summary)
        scale = torch.tanh(self.variable_repr_scale).to(
            dtype=hidden.dtype, device=hidden.device
        )
        return hidden + scale * context

    @staticmethod
    def _signed_acf(x, lag):
        """Compute a differentiable, signed per-window ACF at one lag."""

        batch, length, channels = x.shape
        if lag >= length:
            return x.new_zeros((batch, channels))
        centered = x - x.mean(dim=1, keepdim=True)
        left = centered[:, :-lag, :]
        right = centered[:, lag:, :]
        numerator = (left * right).sum(dim=1)
        denominator = torch.sqrt(
            (left.square().sum(dim=1) * right.square().sum(dim=1)).clamp_min(1e-8)
        )
        return (numerator / denominator).clamp(-1.0, 1.0)

    def _causal_memory(self, x, horizon):
        """Return legal historical values for each horizon and selected lag.

        The returned tuple is ``(memory, signed_acf, valid)``.  ``valid`` is
        explicit because a short input window cannot supply every long lag;
        unavailable values are never treated as observations.
        """

        batch, length, channels = x.shape
        if not self.acf_lags:
            return None, None, None
        lags = torch.as_tensor(self.acf_lags, device=x.device, dtype=torch.long)
        steps = torch.arange(1, horizon + 1, device=x.device, dtype=torch.long)
        # Direct one-step lag semantics: for forecast step ``h`` (1-based),
        # read the observed position ``L - 1 + h - lag``.  Once that position
        # leaves the encoder window it is marked unavailable; silently
        # recycling an older cycle would turn an unavailable long lag into a
        # fabricated observation.
        indices = length - 1 + steps[:, None] - lags[None, :]
        valid = (indices >= 0) & (indices < length)
        safe_indices = indices.clamp(0, max(0, length - 1))
        # x[:, index, :] -> [B,H,K,C], then expose [B,H,C,K].
        memory_source = self._memory_signal(x)
        memory = memory_source[:, safe_indices, :].permute(0, 1, 3, 2)
        if self.dynamic_acf:
            # Dynamic coefficients must describe the same signal whose
            # historical values are supplied below.  In particular, a
            # residual/trend memory source must not be paired with raw-signal
            # correlations.
            corr = torch.stack(
                [self._signed_acf(memory_source, int(lag)) for lag in self.acf_lags],
                dim=-1,
            )
        elif self.acf_prior.numel():
            corr = self.acf_prior.unsqueeze(0).expand(batch, -1, -1)
        else:
            corr = x.new_zeros((batch, channels, len(self.acf_lags)))
        return memory, corr, valid

    def _memory_signal(self, x):
        """Return the same signal family used to fit the train-only ACF prior.

        The forecast is still restored against the original normalized window;
        this transformation only determines which observed lag residual is
        supplied to the optional seasonal memory branch.
        """

        if self.acf_source == "raw":
            return x
        kernel = max(3, self.seq_len + 1)
        if kernel % 2 == 0:
            kernel += 1
        values = x.transpose(1, 2)
        front = values[:, :, :1].expand(-1, -1, (kernel - 1) // 2)
        end = values[:, :, -1:].expand(-1, -1, (kernel - 1) // 2)
        padded = torch.cat([front, values, end], dim=-1)
        trend = F.avg_pool1d(padded, kernel_size=kernel, stride=1)
        if self.acf_source == "trend":
            return trend.transpose(1, 2)
        return (values - trend).transpose(1, 2)

    def _acf_representation_context(self, x):
        """Build a train-prior ACF context on the encoder grid.

        The lag lookup is causal: at position ``t`` only ``x[t-lag]`` is
        indexed.  When ``acf_source`` is ``residual`` or ``trend``, however,
        ``_memory_signal`` currently uses a centered full-window smoother, so
        the transformed signal itself is not strictly causal.  The profile
        reports those two properties separately rather than overstating the
        guarantee.  The signed prior is normalized by its per-channel
        absolute mass; the fourth summary records how much mass is available
        in the current prefix.
        """

        if (
            not self.acf_repr_fusion_enabled
            or self.acf_prior.numel() == 0
            or not self.acf_lags
        ):
            return None
        if x.ndim != 3:
            raise ValueError("ACF representation context expects [B,T,C] input")
        batch, length, channels = x.shape
        prior = self.acf_prior.to(device=x.device, dtype=x.dtype)
        if prior.ndim != 2 or tuple(prior.shape) != (
            self.channels, len(self.acf_lags)
        ):
            raise RuntimeError(
                "attached ACF prior does not align with representation lags"
            )
        lags = torch.as_tensor(self.acf_lags, device=x.device, dtype=torch.long)
        positions = torch.arange(length, device=x.device, dtype=torch.long)
        indices = positions.unsqueeze(0) - lags.unsqueeze(1)  # [K,T]
        valid = (indices >= 0) & (indices < length)
        safe_indices = indices.clamp(0, max(0, length - 1))

        # Advanced indexing gives a dense temporary [B,K,T,C]; expose the lag
        # axis last to align with the [C,K] prior.  This representation-only
        # path is intentionally kept separate from prediction-only inference;
        # a lag-wise accumulation would reduce peak memory but change the
        # floating-point addition order and therefore needs its own contract.
        lagged = x[:, safe_indices, :].permute(0, 2, 3, 1)
        memory_source = self._memory_signal(x)
        if memory_source is not x:
            lagged = memory_source[:, safe_indices, :].permute(0, 2, 3, 1)
        absolute_mass = prior.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        coefficients = prior / absolute_mass
        mask = valid.transpose(0, 1).to(dtype=x.dtype).view(1, length, 1, -1)
        seasonal = (
            lagged
            * coefficients.view(1, 1, channels, -1)
            * mask
        ).sum(dim=-1)
        available_mass = (
            prior.abs().view(1, 1, channels, -1) * mask
        ).sum(dim=-1) / absolute_mass.view(1, 1, channels)
        reliability = available_mass.mean(dim=-1).expand(batch, -1)
        summary = torch.stack(
            (
                seasonal.mean(dim=-1),
                seasonal.abs().mean(dim=-1),
                seasonal.square().mean(dim=-1).sqrt(),
                reliability,
            ),
            dim=-1,
        )
        return summary

    def _fuse_acf_representation(self, hidden, x):
        """Inject the optional ACF context into AutoCon's hidden grid only."""

        if not self.acf_repr_fusion_enabled:
            return hidden
        summary = self._acf_representation_context(x)
        if summary is None:
            return hidden
        context = self.acf_repr_projection(summary)
        scale = torch.tanh(self.acf_repr_scale).to(
            dtype=hidden.dtype, device=hidden.device
        )
        if hidden.ndim == 3:
            if tuple(hidden.shape[:2]) != tuple(context.shape[:2]):
                raise RuntimeError(
                    "ACF representation context does not match hidden grid"
                )
            return hidden + scale * context
        if hidden.ndim == 4:
            if hidden.shape[0] != context.shape[0] or hidden.shape[2] != context.shape[1]:
                raise RuntimeError(
                    "ACF representation context does not match channel-independent hidden grid"
                )
            return hidden + scale * context.unsqueeze(1)
        raise ValueError("hidden representation must have 3 or 4 dimensions")

    @staticmethod
    def _acf_memory_reliability(corr, lag_weights, valid, stepwise=False):
        """Return bounded ACF confidence for a memory residual.

        ``corr`` is ``[B,C,K]``, ``lag_weights`` is ``[B,H,C,K]`` and
        ``valid`` is ``[H,K]``.  Stepwise confidence is a weighted mean of
        the signed-ACF magnitudes over the legal lags for each horizon step;
        the legacy mode retains one availability-weighted value per channel.
        """

        valid_weight = valid.to(dtype=lag_weights.dtype)[None, :, None, :]
        corr_abs = corr.abs()[:, None, :, :]
        if stepwise:
            return (
                lag_weights * corr_abs * valid_weight
            ).sum(dim=-1).clamp(0.0, 1.0)
        available_fraction = valid_weight.mean(dim=1).squeeze(1)
        return (corr.abs() * available_fraction).mean(dim=-1).clamp(0.0, 1.0)

    def _reset_acf_runtime_diagnostics(self):
        diagnostics = getattr(self, "_acf_runtime_diagnostics", None)
        if diagnostics is None:
            diagnostics = {
                key: None for key in (
                    "acf_runtime_gate_mean", "acf_runtime_gate_std",
                    "acf_memory_gate_variable_token",
                    "acf_anchor_copy_enabled", "acf_anchor_scale",
                    "acf_anchor_rms", "acf_anchor_valid_fraction",
                    "acf_anchor_strength_mean", "acf_anchor_lag_mean",
                    "acf_seasonal_rms", "acf_reliability_mean",
                    "acf_effective_memory_rms",
                    "acf_effective_memory_to_forecast_rms",
                    "acf_calibration_amp_mean",
                    "acf_calibration_amp_std", "acf_calibration_amp_min",
                    "acf_calibration_amp_max",
                    "acf_reliability_interpolation_mean",
                    "acf_reliability_interpolation_std",
                    "acf_reliability_interpolation_min",
                    "acf_reliability_interpolation_max",
                )
            }
            self._acf_runtime_diagnostics = diagnostics
        for key in diagnostics:
            if key in {"acf_memory_gate_variable_token", "acf_anchor_copy_enabled"}:
                continue
            diagnostics[key] = None

    def _set_acf_runtime_diagnostic(self, key, value):
        # Training logs consume these scalars batch by batch.  Evaluation and
        # deployment should not pay a hidden GPU->CPU synchronization cost for
        # telemetry unless the caller explicitly requests it.  Keep detached
        # scalar tensors on-device until the complete snapshot is requested;
        # this lets the reader perform one batched transfer instead of one
        # synchronization per diagnostic field.
        if (
            not self.training
            and not getattr(self, "acf_collect_eval_diagnostics", True)
        ):
            return
        diagnostics = getattr(self, "_acf_runtime_diagnostics", None)
        if diagnostics is None:
            self._reset_acf_runtime_diagnostics()
            diagnostics = self._acf_runtime_diagnostics
        if key not in diagnostics:
            # Older structural tests construct a minimal object without the
            # full __init__; retaining a lazy key keeps the diagnostic helper
            # non-invasive for those objects.
            diagnostics[key] = None
        if value is None:
            self._acf_runtime_diagnostics[key] = None
            return
        if torch.is_tensor(value):
            value = value.detach().float()
            value = value.square().mean().sqrt() if key.endswith("rms") else value.mean()
            self._acf_runtime_diagnostics[key] = value.reshape(())
            return
        else:
            value = float(value)
        diagnostics[key] = value

    def acf_runtime_diagnostics(self):
        """Return a detached snapshot of the latest forecast-path diagnostics.

        Tensor-valued telemetry is reduced to scalar tensors by
        ``_set_acf_runtime_diagnostic``.  Grouping those scalars by device
        keeps the public API as ordinary Python values while amortizing the
        GPU-to-CPU synchronization across the whole snapshot.
        """

        diagnostics = dict(getattr(self, "_acf_runtime_diagnostics", {}))
        pending = {}
        for key, value in diagnostics.items():
            if torch.is_tensor(value):
                pending.setdefault(str(value.device), []).append((key, value))
        for values in pending.values():
            keys = [key for key, _ in values]
            stacked = torch.stack([value for _, value in values])
            host_values = stacked.detach().cpu().tolist()
            for key, host_value in zip(keys, host_values):
                diagnostics[key] = float(host_value)
        return diagnostics

    def _record_acf_effective_memory_strength(self, effective_memory, base_forecast):
        """Record ACF residual strength relative to the pre-memory forecast.

        The existing absolute RMS is useful only when all runs share the same
        normalization and target scale.  This ratio is a detached telemetry
        value; it is computed before the residual is added and therefore does
        not alter the prediction graph or forecast values.
        """

        if not self.training and not getattr(
            self, "acf_collect_eval_diagnostics", False
        ):
            return
        effective = effective_memory.detach().float()
        baseline = base_forecast.detach().float()
        effective_rms = effective.square().mean().sqrt()
        baseline_rms = baseline.square().mean().sqrt()
        relative_rms = effective_rms / baseline_rms.clamp_min(1e-12)
        self._set_acf_runtime_diagnostic(
            "acf_effective_memory_to_forecast_rms", relative_rms
        )

    def _forecast_normalized(
        self, hidden, x, local_tokens=None, variable_tokens=None
    ):
        self._reset_acf_runtime_diagnostics()
        if self.channel_independent:
            batch, channels, length, _ = hidden.shape
        else:
            variable_token_gate_source = None
            batch, length, _ = hidden.shape
            channels = x.shape[-1]
        if self.channel_independent:
            flat_hidden = hidden.reshape(batch * channels, length, self.d_model)
            values = self.value_head(flat_hidden).squeeze(-1)
            mamba_forecast = self.temporal_head(values)

            # A small learned residual preserves the strong linear inductive
            # bias for local continuation without making it the only path.
            flat_x = self._reshape_channels(x).squeeze(-1)
            residual_forecast = self.linear_residual(flat_x)
            gate = torch.sigmoid(self.residual_gate(flat_hidden[:, -1, :]))
            forecast = mamba_forecast + gate * residual_forecast
        else:
            # Shared temporal state; project to all channels only after the
            # Mamba scan.  Shapes are [B,T,C] and [B,C,H] respectively.
            values = self.value_head(hidden)
            mamba_forecast = self.temporal_head(values.transpose(1, 2)).transpose(1, 2)
            residual_forecast = self.linear_residual(x.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(self.residual_gate(hidden[:, -1, :])).unsqueeze(-1)
            forecast = mamba_forecast + gate * residual_forecast

            if self.variable_mixer_enabled:
                if variable_tokens is None:
                    variable_tokens = self._variable_token_features(x)
                variable_token_gate_source = variable_tokens
                variable_forecast = self.variable_head(variable_tokens).transpose(1, 2)
                variable_weight = torch.sigmoid(self.variable_gate(variable_tokens)).transpose(1, 2)
                variable_weight = variable_weight.expand(-1, self.pred_len, -1)
                variable_scale = torch.tanh(self.variable_residual_scale).to(
                    dtype=forecast.dtype, device=forecast.device
                )
                forecast = forecast + variable_scale * variable_weight * variable_forecast

            if self.patch_mixer_enabled:
                forecast = forecast + self._patch_forecast(x)

            if self.local_mixer_enabled:
                forecast = forecast + self._local_forecast(
                    x, local_tokens=local_tokens
                )

        if self.use_acf_memory and self.acf_lags:
            if self.acf_prior.numel() and not self.dynamic_acf:
                seasonal, reliability = self._static_acf_memory(
                    x, self.pred_len
                )
                hidden_gate = None
            elif not self.dynamic_acf:
                # A static-memory checkpoint is not self-contained unless its
                # train-only prior is reattached.  Failing here prevents a
                # standalone checkpoint load from silently changing the model
                # into an ACF-off forecast.
                raise RuntimeError(
                    "MambaACF static ACF memory is enabled but no train-only "
                    "prior is attached; load acf_prior.npz with "
                    "model.load_acf_prior(...) or construct the experiment "
                    "through its train-split prior attachment path"
                )
            else:
                memory, corr, valid = self._causal_memory(x, self.pred_len)
                seasonal = None
                reliability = None
                hidden_gate = None
                if memory is not None:
                    # Signed correlation controls direction, while a
                    # temperature-controlled softmax over magnitudes selects
                    # among available lags. Invalid entries are masked before
                    # normalization and cannot inject repeated padding.
                    valid_weight = valid.to(dtype=memory.dtype)[None, :, None, :]
                    abs_corr = corr.abs()[:, None, :, :]
                    lag_logits = torch.log(abs_corr.clamp_min(1e-6)) / self.acf_temperature
                    lag_logits = lag_logits.masked_fill(
                        ~valid[None, :, None, :], torch.finfo(memory.dtype).min
                    )
                    lag_weights = torch.softmax(lag_logits, dim=-1) * valid_weight
                    lag_weights = lag_weights / lag_weights.sum(dim=-1, keepdim=True).clamp_min(1e-4)
                    seasonal = (
                        memory * corr.sign()[:, None, :, :] * lag_weights
                    ).sum(dim=-1)
                    acf_reliability = self._acf_memory_reliability(
                        corr,
                        lag_weights,
                        valid,
                        stepwise=self.acf_step_reliability,
                    )
                    reliability = (
                        acf_reliability
                        if self.acf_step_reliability
                        else acf_reliability.unsqueeze(1)
                    )

            if seasonal is not None:
                if self.channel_independent:
                    hidden_gate = torch.sigmoid(self.acf_gate(flat_hidden[:, -1, :]))
                    # The independent forecast is flattened as [B*C,H].
                    seasonal = seasonal.permute(0, 2, 1).reshape(
                        batch * channels, self.pred_len
                    )
                    if reliability.shape[0] == 1 and batch != 1:
                        reliability = reliability.expand(
                            batch, -1, -1
                        )
                    if self.acf_step_reliability:
                        reliability = reliability.permute(0, 2, 1).reshape(
                            batch * channels, self.pred_len
                        )
                    else:
                        reliability = reliability.reshape(batch * channels, 1)
                    effective_memory = hidden_gate * reliability * seasonal
                    self._set_acf_runtime_diagnostic(
                        "acf_runtime_gate_mean", hidden_gate
                    )
                    self._set_acf_runtime_diagnostic(
                        "acf_runtime_gate_std", hidden_gate.std(unbiased=False)
                    )
                    self._set_acf_runtime_diagnostic(
                        "acf_effective_memory_rms", effective_memory
                    )
                    self._record_acf_effective_memory_strength(
                        effective_memory, forecast
                    )
                    forecast = forecast + effective_memory
                else:
                    if getattr(self, "acf_memory_gate_mode", "hidden") == "variable_token":
                        if variable_token_gate_source is None:
                            raise RuntimeError(
                                "variable-token ACF memory gate requires the "
                                "variable-token forecast branch"
                            )
                        hidden_gate = torch.sigmoid(
                            self.acf_gate(variable_token_gate_source)
                        ).transpose(1, 2)
                    else:
                        hidden_gate = torch.sigmoid(
                            self.acf_gate(hidden[:, -1, :])
                        ).unsqueeze(-1)
                    effective_memory = hidden_gate * reliability * seasonal
                    self._set_acf_runtime_diagnostic(
                        "acf_runtime_gate_mean", hidden_gate
                    )
                    self._set_acf_runtime_diagnostic(
                        "acf_runtime_gate_std", hidden_gate.std(unbiased=False)
                    )
                    self._set_acf_runtime_diagnostic(
                        "acf_effective_memory_rms", effective_memory
                    )
                    self._record_acf_effective_memory_strength(
                        effective_memory, forecast
                    )
                    forecast = forecast + effective_memory

        if getattr(self, "acf_anchor_copy", False):
            # This is an independent, strictly nested seasonal expert.  It
            # deliberately does not reuse the soft-memory reliability or gate:
            # the experiment therefore tests whether selecting one trustworthy
            # per-channel lag recovers signal that multi-lag averaging loses.
            anchor, anchor_lags, anchor_strength, anchor_valid = (
                self._static_acf_anchor_memory(x, self.pred_len)
            )
            if anchor is None:
                raise RuntimeError(
                    "direct ACF anchor retrieval requires an attached train-only "
                    "ACF prior"
                )
            anchor_scale = torch.tanh(self.acf_anchor_scale).to(
                dtype=forecast.dtype, device=forecast.device
            )
            if self.channel_independent:
                anchor_for_forecast = anchor.permute(0, 2, 1).reshape(
                    batch * channels, self.pred_len
                )
            else:
                anchor_for_forecast = anchor
            forecast = forecast + anchor_scale * anchor_for_forecast
            if self.training or self.acf_collect_eval_diagnostics:
                valid_float = anchor_valid.to(dtype=anchor_strength.dtype)
                valid_count = valid_float.sum().clamp_min(1.0)
                self._set_acf_runtime_diagnostic("acf_anchor_scale", anchor_scale)
                self._set_acf_runtime_diagnostic("acf_anchor_rms", anchor)
                self._set_acf_runtime_diagnostic(
                    "acf_anchor_valid_fraction", valid_float.mean()
                )
                self._set_acf_runtime_diagnostic(
                    "acf_anchor_strength_mean",
                    (anchor_strength * valid_float).sum() / valid_count,
                )
                self._set_acf_runtime_diagnostic(
                    "acf_anchor_lag_mean",
                    (anchor_lags.to(dtype=anchor_strength.dtype) * valid_float).sum()
                    / valid_count,
                )

        if self.channel_independent:
            return forecast.reshape(batch, channels, self.pred_len).transpose(1, 2)
        return forecast

    def _normalize(self, x):
        mode = getattr(self, "AutoCon_wnorm", "LastVal")
        if mode == "MeanStd":
            # Match PatchTST's per-window standardization exactly: the
            # population variance is used and the numerical floor is inside
            # the square root.  Keep this as a distinct mode so historical
            # ReVIN/LastVal checkpoints retain their original semantics.
            mean = x.mean(dim=1, keepdim=True).detach()
            centered = x - mean
            std = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            return centered / std, (mean, std)
        if mode == "ReVIN":
            mean = x.mean(dim=1, keepdim=True).detach()
            # Preserve the historical AutoCon/ReVIN sample-standard-deviation
            # contract.  PatchTST-compatible population statistics are exposed
            # separately as the explicit ``MeanStd`` mode.
            std = x.std(dim=1, keepdim=True, unbiased=True).detach()
            return (x - mean) / (std + 1e-5), (mean, std)
        if mode == "Mean":
            mean = x.mean(dim=1, keepdim=True).detach()
            return x - mean, mean
        if mode == "LastVal":
            last = x[:, -1:, :].detach()
            return x - last, last
        raise ValueError(
            "MambaACF supports AutoCon_wnorm in {MeanStd, ReVIN, Mean, LastVal}"
        )

    def _restore(self, prediction, restore):
        mode = getattr(self, "AutoCon_wnorm", "LastVal")
        if mode in {"ReVIN", "MeanStd"}:
            mean, std = restore
            if mode == "MeanStd":
                return prediction * std + mean
            return prediction * (std + 1e-5) + mean
        return prediction + restore

    def _pool_representation(self, hidden):
        if not self.channel_independent:
            return hidden
        # Learn a soft channel pooling for the contrastive representation while
        # retaining channel-specific states for the forecasting head.
        scores = self.channel_pool(hidden).squeeze(-1)  # [B,C,T]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)

    def _contrastive_repr(self, hidden):
        """Project the representation only for the AutoCon objective."""
        pooled = self._pool_representation(hidden)
        return self.repr_head(pooled) if self.AutoCon else pooled

    def resource_profile(self):
        """Return the effective channel/scan contract for fair comparisons.

        Total parameter count alone is misleading for high-dimensional inputs:
        shared mode compresses ``C`` variables into a ``d_model`` state, while
        channel-independent mode keeps one state per variable by flattening the
        batch.  The profile is intentionally read-only and contains no tensor
        values, so resource auditors can record the structural difference
        without changing the forward path.
        """

        # A causal ACF lag lookup alone does not make the returned AutoCon
        # representation pointwise causal.  Pooling summarizes a complete
        # chunk before it is interpolated back to the full time grid, and a
        # window-conditioned channel/scale adapter applies one summary to all
        # positions.  Keep these reasons explicit so resource reports cannot
        # turn a forecast-level causal-memory claim into a representation-level
        # causality claim.
        representation_causality_limits = []
        if not self.causal_embedding:
            representation_causality_limits.append("noncausal_token_embedding")
        if self.bidirectional:
            representation_causality_limits.append("bidirectional_temporal_scan")
        if self.mamba_token_stride != 1:
            representation_causality_limits.append(
                "strided_pooling_and_interpolation"
            )
        if any(scale != 1 for scale in self.scales):
            representation_causality_limits.append(
                "multiscale_pooling_and_interpolation"
            )
        if self.channel_gate is not None:
            representation_causality_limits.append(
                "window_conditioned_channel_gate"
            )
        if self.acf_scale_conditioner is not None:
            representation_causality_limits.append(
                "window_conditioned_acf_scale_gate"
            )
        if is_full_window_normalization(self.AutoCon_wnorm):
            representation_causality_limits.append(
                "window_full_normalization"
            )
        if getattr(self, "data_dependent_channel_mixer", None) is not None:
            representation_causality_limits.append(
                "window_conditioned_data_dependent_channel_mixer"
            )
        if self.local_mixer_enabled and self.AutoCon:
            representation_causality_limits.append(
                "window_pooled_local_representation_context"
            )
        if self.variable_repr_fusion_enabled:
            representation_causality_limits.append(
                "window_pooled_variable_token_representation_context"
            )
        if (
            getattr(self, "variable_gdd_mixer", None) is not None
            and self.variable_repr_fusion_enabled
        ):
            representation_causality_limits.append(
                "window_conditioned_variable_token_gdd_mixer"
            )
        if self.acf_repr_fusion_enabled and self.acf_source != "raw":
            representation_causality_limits.append(
                "centered_acf_representation_transform"
            )
        representation_grid_is_strictly_causal = not representation_causality_limits

        channel_branches = []
        if self.channel_independent:
            channel_mode = "channel_independent"
        else:
            channel_mode = "shared_temporal"
            if self.variable_mixer_enabled:
                channel_branches.append("variable_tokens")
            if self.patch_mixer_enabled:
                channel_branches.append("patch_tokens")
            if channel_branches:
                channel_mode += "_plus_" + "_plus_".join(channel_branches)

        # ``self.branches`` contains one temporal stack per scale.  A
        # channel-independent batch is physically expanded to B*C streams;
        # shared mode keeps one stream per batch item.  Bidirectionality runs a
        # second scan over each stream, but does not change the stream count.
        primary_streams = self.channels if self.channel_independent else 1
        primary_scan_streams = primary_streams * len(self.scales)
        variable_scan_tokens = (
            self.variable_group_count if self.variable_mixer_enabled else 0
        )
        # A branch is one ordered sequence per batch item; depth is a stack of
        # blocks applied to that sequence, not an additional independent
        # stream.  Keep streams and scan operations separate in the profile.
        variable_scan_streams = 1 if self.variable_mixer_enabled else 0
        patch_scan_tokens = (
            self.patch_count * self.channels if self.patch_mixer_enabled else 0
        )
        patch_scan_streams = 1 if self.patch_mixer_enabled else 0
        direction_multiplier = 2 if self.bidirectional else 1
        base_token_count = self.base_token_count
        main_tokens_per_scale = [
            int(math.ceil(base_token_count / scale)) for scale in self.scales
        ]
        local_agreement_lags, local_agreement_pair_count = (
            self._acf_local_agreement_support(self.seq_len)
        )
        local_agreement_parameter_count = int(
            self.acf_local_agreement_scale.numel()
            if getattr(self, "acf_local_agreement_scale", None) is not None
            else 0
        )
        local_agreement_path_active = bool(
            getattr(self, "acf_local_agreement_requested", False)
            and self.use_acf_memory
            and not self.dynamic_acf
            and self.acf_prior.numel() > 0
            and len(local_agreement_lags) > 0
        )
        channel_lag_bank_enabled = bool(
            getattr(self, "acf_channel_lag_bank", False)
        )
        channel_lag_bank_top_k = int(
            getattr(self, "acf_channel_lag_top_k", 2)
        )
        channel_lag_bank_prior_attached = bool(
            channel_lag_bank_enabled and self.acf_prior.numel() > 0
        )
        channel_lag_bank_path_active = bool(
            channel_lag_bank_prior_attached
            and self.use_acf_memory
            and not self.dynamic_acf
        )
        channel_lag_bank_slots = 0
        channel_lag_bank_total_slots = 0
        channel_lag_bank_bytes = 0
        channel_lag_bank_fp32_bytes = 0
        channel_lag_bank_legal_lags = []
        channel_lag_bank_legal_step_fractions = []
        if channel_lag_bank_enabled and self.acf_lags:
            lags = torch.as_tensor(self.acf_lags, dtype=torch.long)
            steps = torch.arange(1, self.pred_len + 1, dtype=torch.long)
            indices = self.seq_len - 1 + steps[:, None] - lags[None, :]
            legal_fraction = (
                ((indices >= 0) & (indices < self.seq_len)).float().mean(dim=0)
            )
            channel_lag_bank_legal_lags = [
                int(lag)
                for lag, fraction in zip(self.acf_lags, legal_fraction.tolist())
                if fraction > 0.0
            ]
            channel_lag_bank_legal_step_fractions = [
                float(fraction) for fraction in legal_fraction.tolist()
            ]
        if channel_lag_bank_prior_attached and self.acf_prior.ndim == 2:
            support = self.acf_memory_static_channel_support
            if support.ndim != 2 or tuple(support.shape) != (
                self.channels, len(self.acf_lags)
            ):
                support = self._build_static_acf_channel_support(
                    length=self.seq_len, horizon=self.pred_len
                )
            channel_lag_bank_slots = int(support.sum().item())
            channel_lag_bank_total_slots = int(support.numel())
            # ``support`` is a dense bool [C,K] buffer, not a sparse index
            # list.  Count its physical storage rather than selected entries.
            channel_lag_bank_bytes = int(support.numel() * support.element_size())
            channel_lag_bank_fp32_bytes = int(support.numel() * 4)

        # Report the two channel-preserving horizon projections separately.
        # These are architecture-level multiply counts (biases, activation
        # kernels, and backend fusion are excluded), so they must not be
        # presented as measured FLOPs or wall-clock time.  Keeping the
        # accounting here makes low-rank-head comparisons auditable even when
        # a GPU kernel launch dominates the measured latency.
        temporal_head_parameters = int(
            sum(parameter.numel() for parameter in self.temporal_head.parameters())
        )
        linear_residual_parameters = int(
            sum(parameter.numel() for parameter in self.linear_residual.parameters())
        )

        def projection_macs(input_length, output_length, rank):
            if rank is None:
                return int(input_length) * int(output_length)
            rank = int(rank)
            return int(input_length) * rank + rank * int(output_length)

        temporal_head_macs_per_channel = projection_macs(
            self.temporal_input_length, self.pred_len, self.temporal_rank
        )
        linear_residual_macs_per_channel = projection_macs(
            self.seq_len, self.pred_len, self.temporal_rank
        )
        temporal_projection_macs_per_window = int(self.channels) * (
            temporal_head_macs_per_channel + linear_residual_macs_per_channel
        )

        # The shared *Mamba component* of the forecast has the factorization
        # ``h[t,d] -> value[d,c]`` and therefore a channel-rank bound of
        # d_model.  The direct linear residual is already channel preserving,
        # so this is deliberately not claimed as a rank bound for the final
        # summed forecast.  Additive variable/patch branches are also reported
        # separately because they do not remove the shared component's bound.
        preserving_branches = []
        if self.channel_independent:
            preserving_branches.append("independent_temporal")
        if self.variable_mixer_enabled:
            preserving_branches.append("variable_tokens")
        if self.patch_mixer_enabled:
            preserving_branches.append("patch_tokens")
        if self.local_mixer_enabled:
            preserving_branches.append("local_patches")
        data_dependent_channel_mixer = getattr(
            self, "data_dependent_channel_mixer", None
        )
        channel_state_embedding = getattr(self, "channel_state_embedding", None)
        data_dependent_channel_mixer_active = bool(
            data_dependent_channel_mixer is not None
        )
        # Architecture-level cost proxy for the optional window-conditioned
        # adapter.  These counts cover only the added linear projections and
        # elementwise affine modulation; bias adds, reductions, nonlinearities,
        # kernel launches, the temporal Mamba stack, and data movement are
        # intentionally excluded.  They are not measured FLOPs or latency.
        data_mixer_state_embedding_macs = 0
        data_mixer_summary_projection_macs = 0
        data_mixer_hidden_affine_macs = 0
        data_mixer_adapter_linear_macs = 0
        data_mixer_affine_macs = 0
        data_mixer_tokenizer_scalar_ops = 0
        data_mixer_summary_ops = 0
        data_mixer_pooling_ops = 0
        data_mixer_activation_elements = 0
        data_mixer_tokenizer_input_width = 0
        data_mixer_summary_width = 0
        data_mixer_state_length = 0
        if data_dependent_channel_mixer_active:
            mixer_channels = int(self.channels)
            mixer_dim = int(self.d_model)
            mixer_rank = int(data_dependent_channel_mixer.rank)
            data_mixer_tokenizer_input_width = 0
            data_mixer_summary_width = int(data_dependent_channel_mixer.summary_dim)
            data_mixer_state_length = int(self.base_token_count)
            data_mixer_summary_projection_macs = (
                data_mixer_summary_width * mixer_rank
            )
            data_mixer_hidden_affine_macs = mixer_rank * (2 * mixer_dim)
            data_mixer_adapter_linear_macs = (
                data_mixer_summary_projection_macs
                + data_mixer_hidden_affine_macs
            )
            # The hidden-space FiLM applies one scale and one residual add at
            # every temporal state.  Count the scale multiply as the explicit
            # architecture proxy; bias adds and nonlinearities are excluded.
            data_mixer_affine_macs = data_mixer_state_length * mixer_dim
            # The production path reuses ``gate_input`` from the existing
            # scale-gate reduction.  It therefore adds no raw-window pooling
            # work; keep this incremental adapter cost at zero.  The field is
            # retained for compatibility with older resource manifests.
            data_mixer_tokenizer_scalar_ops = 0
            data_mixer_summary_ops = 0
            data_mixer_pooling_ops = 0
            # The [B,D] summary is owned by the existing scale gate and is not
            # counted as an incremental activation.  Count only adapter-owned
            # latent/affine tensors and the hidden residual state.
            data_mixer_activation_elements = (
                mixer_rank
                + 2 * mixer_dim
                + data_mixer_state_length * mixer_dim
            )
        variable_repr_fusion_active = bool(
            getattr(self, "variable_repr_fusion_enabled", False)
        )
        variable_repr_fusion_parameters = int(
            sum(
                parameter.numel()
                for module in (
                    self.variable_repr_pool,
                    self.variable_repr_projection,
                )
                if module is not None
                for parameter in module.parameters()
            )
            + (self.variable_repr_scale.numel() if variable_repr_fusion_active else 0)
        )
        variable_repr_fusion_trainable_parameters = int(
            sum(
                parameter.numel()
                for module in (
                    self.variable_repr_pool,
                    self.variable_repr_projection,
                )
                if module is not None
                for parameter in module.parameters()
                if parameter.requires_grad
            )
            + (
                self.variable_repr_scale.numel()
                if variable_repr_fusion_active
                and self.variable_repr_scale.requires_grad
                else 0
            )
        )
        # Cost proxy for the representation-only adapter.  It excludes bias,
        # activation, softmax, and kernel-launch overhead and is not a timing
        # claim.  The reduction and score projection scale linearly with C;
        # the two-layer D -> D adapter is independent of channel count.
        variable_repr_fusion_macs = int(
            self.channels * self.d_model
            + self.channels * self.d_model
            + 2 * self.d_model * self.d_model
            if variable_repr_fusion_active
            else 0
        )
        variable_repr_fusion_activation_elements = int(
            self.channels
            + self.channels * self.d_model
            + self.d_model
            if variable_repr_fusion_active
            else 0
        )
        variable_gdd_mixer = getattr(self, "variable_gdd_mixer", None)
        variable_gdd_mixer_active = bool(variable_gdd_mixer is not None)
        if variable_gdd_mixer_active:
            preserving_branches.append("variable_token_gdd")
        variable_gdd_mixer_parameters = int(
            sum(parameter.numel() for parameter in variable_gdd_mixer.parameters())
            if variable_gdd_mixer_active else 0
        )
        variable_gdd_mixer_trainable_parameters = int(
            sum(
                parameter.numel()
                for parameter in variable_gdd_mixer.parameters()
                if parameter.requires_grad
            ) if variable_gdd_mixer_active else 0
        )
        variable_gdd_mixer_rank = int(
            variable_gdd_mixer.rank if variable_gdd_mixer_active else 0
        )
        # These are deliberately architecture-level proxies, not measured
        # FLOPs.  The adapter pools mean/RMS over the singleton patch axis,
        # applies two low-rank projections to produce per-channel affine
        # parameters, and performs one feature-wise scale/add operation.
        variable_gdd_mixer_summary_macs = int(
            4 * self.channels * variable_gdd_mixer_rank
            if variable_gdd_mixer_active else 0
        )
        variable_gdd_mixer_affine_macs = int(
            self.channels * self.d_model if variable_gdd_mixer_active else 0
        )
        variable_gdd_mixer_macs = int(
            variable_gdd_mixer_summary_macs + variable_gdd_mixer_affine_macs
        )
        variable_gdd_mixer_activation_elements = int(
            variable_gdd_mixer_rank
            + 2 * self.channels
            + self.channels * self.d_model
            if variable_gdd_mixer_active else 0
        )
        data_mixer_total_macs = (
            data_mixer_state_embedding_macs
            + data_mixer_adapter_linear_macs
            + data_mixer_affine_macs
        )
        return {
            "channel_mode": channel_mode,
            "input_channels": self.channels,
            "state_width": self.d_model,
            "temporal_token_stride": int(self.mamba_token_stride),
            "temporal_token_count_before_multiscale": int(base_token_count),
            "temporal_head_input_length": int(self.temporal_input_length),
            "temporal_head_factorized": bool(self.low_rank_head),
            "temporal_projection_rank": (
                int(self.temporal_rank) if self.temporal_rank is not None else None
            ),
            "temporal_head_parameters": temporal_head_parameters,
            "linear_residual_parameters": linear_residual_parameters,
            "temporal_head_macs_per_channel": int(temporal_head_macs_per_channel),
            "linear_residual_macs_per_channel": int(linear_residual_macs_per_channel),
            "temporal_projection_macs_per_window": temporal_projection_macs_per_window,
            "representation_grid_length": int(self.seq_len),
            "representation_grid_is_strictly_causal": bool(
                representation_grid_is_strictly_causal
            ),
            "representation_grid_causality_limits": representation_causality_limits,
            "representation_normalization": str(self.AutoCon_wnorm),
            "representation_normalization_is_window_global": bool(
                is_full_window_normalization(self.AutoCon_wnorm)
            ),
            # The forecast relation view reuses the already-computed
            # normalized prediction only while ``return_repr`` is requested by
            # AutoCon training.  Prediction-only deployment therefore has no
            # new parameter, projection, or forward MAC budget.
            "acf_relation_representation": str(
                getattr(self, "acf_relation_representation", "encoder")
            ),
            "acf_relation_forecast_view_enabled": bool(
                getattr(self, "forecast_relation_representation_enabled", False)
            ),
            "acf_relation_forecast_view_input_shape": (
                "B,H,C"
                if getattr(self, "forecast_relation_representation_enabled", False)
                else None
            ),
            "acf_relation_forecast_view_scope": (
                "global"
                if getattr(self, "forecast_relation_representation_enabled", False)
                else None
            ),
            # ``ACFRelationLoss.global_contrastive_loss`` pools the relation
            # view over forecast steps.  This route therefore constrains a
            # window-level cross-channel forecast summary, not stepwise
            # forecast dynamics.
            "acf_relation_forecast_view_global_pooling": (
                "mean_over_horizon"
                if getattr(self, "forecast_relation_representation_enabled", False)
                else None
            ),
            "acf_relation_forecast_view_is_stepwise_temporal_objective": False,
            "acf_relation_forecast_view_parameters": 0,
            "acf_relation_forecast_view_prediction_only_macs_per_batch_item": 0,
            "acf_relation_forecast_view_prediction_only_extra_activations": 0,
            "acf_relation_forecast_view_direct_forecast_gradient": bool(
                getattr(self, "forecast_relation_representation_enabled", False)
            ),
            "embedding_input_channels": 1 if self.channel_independent else self.channels,
            "main_scan_streams_per_batch_item": int(primary_scan_streams * direction_multiplier),
            # Kept as the longest stream for compact tables.  The exact
            # multi-scale workload is available in the list below.
            "main_scan_tokens_per_stream": int(max(main_tokens_per_scale)),
            "main_scan_tokens_per_scale": main_tokens_per_scale,
            "main_scan_tokens_total_per_batch_item": int(
                sum(main_tokens_per_scale) * primary_streams * direction_multiplier
            ),
            "main_scan_blocks_per_stream": int(self.depth),
            "main_scan_operations_per_batch_item": int(
                primary_scan_streams * self.depth * direction_multiplier
            ),
            "acf_scale_conditioning_requested": bool(
                self.acf_scale_conditioning_requested
            ),
            "acf_scale_conditioning_path_active": bool(
                self.acf_scale_conditioner is not None
                and self.acf_prior.numel() > 0
                and len(self.scales) > 1
                and any(
                    0 < int(lag)
                    and int(self.seq_len) - int(lag)
                    >= int(getattr(self, "acf_scale_conditioning_min_pairs", 16))
                    for lag in self.acf_lags
                )
            ),
            "acf_scale_conditioning_min_pairs": int(
                getattr(self, "acf_scale_conditioning_min_pairs", 16)
            ),
            "acf_scale_conditioning_eligible_lag_count": int(
                sum(
                    0 < int(lag)
                    and int(self.seq_len) - int(lag)
                    >= int(getattr(self, "acf_scale_conditioning_min_pairs", 16))
                    for lag in self.acf_lags
                )
            ),
            "acf_scale_conditioning_window_pairs_per_channel": int(
                sum(
                    max(
                        0,
                        int(self.seq_len) - int(lag),
                    )
                    if int(self.seq_len) - int(lag)
                    >= int(getattr(self, "acf_scale_conditioning_min_pairs", 16))
                    else 0
                    for lag in self.acf_lags
                )
            ),
            "acf_scale_conditioning_summary_width": (
                6 if self.acf_scale_conditioner is not None else 0
            ),
            "acf_scale_conditioning_summary_source": (
                "current_normalized_window_signed_acf"
                if self.acf_scale_conditioner is not None else None
            ),
            "acf_scale_conditioning_is_window_dependent": bool(
                self.acf_scale_conditioner is not None
            ),
            "acf_scale_conditioning_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.acf_scale_conditioner.parameters()
                )
                + self.acf_scale_conditioning_scale.numel()
                if self.acf_scale_conditioner is not None
                else 0
            ),
            # CLAM is a forecast-head-only, input-dependent lag reweighting
            # branch.  The support counts are based on the observed encoder
            # length and the explicit minimum pair rule, not on future values.
            "acf_local_agreement_requested": bool(
                getattr(self, "acf_local_agreement_requested", False)
            ),
            "acf_local_agreement_path_active": local_agreement_path_active,
            "acf_local_agreement_active": local_agreement_path_active,
            "acf_local_agreement_min_pairs": int(
                getattr(self, "acf_local_min_pairs", 16)
            ),
            "acf_local_agreement_eligible_lag_count": int(
                len(local_agreement_lags)
            ),
            "acf_local_agreement_eligible_lags": list(local_agreement_lags),
            "acf_local_agreement_pair_count": int(local_agreement_pair_count),
            "acf_local_agreement_window_pairs_per_channel": int(
                local_agreement_pair_count
            ),
            "acf_local_agreement_total_pair_count": int(
                local_agreement_pair_count * self.channels
            ),
            "acf_local_agreement_parameters": local_agreement_parameter_count,
            "acf_local_agreement_trainable_parameters": int(
                self.acf_local_agreement_scale.numel()
                if getattr(self, "acf_local_agreement_scale", None) is not None
                and self.acf_local_agreement_scale.requires_grad
                else 0
            ),
            "acf_local_agreement_source": (
                "current_normalized_observed_window"
                if getattr(self, "acf_local_agreement_scale", None) is not None
                else None
            ),
            "acf_local_agreement_prior_sign_preserved": bool(
                getattr(self, "acf_local_agreement_scale", None) is not None
            ),
            "acf_local_agreement_forecast_origin_causal": bool(
                getattr(self, "acf_local_agreement_scale", None) is not None
            ),
            # The lag bank is a detached [C,K] train-prior support mask.  It
            # adds no learned parameters or dense cross-channel projection;
            # the byte count is reported separately from measured latency.
            "acf_channel_lag_bank_enabled": channel_lag_bank_enabled,
            "acf_channel_lag_bank_top_k": int(
                channel_lag_bank_top_k if channel_lag_bank_enabled else 0
            ),
            "acf_channel_lag_bank_prior_attached": channel_lag_bank_prior_attached,
            "acf_channel_lag_bank_path_active": channel_lag_bank_path_active,
            "acf_channel_lag_bank_source": (
                "train_only_compact_acf_prior"
                if channel_lag_bank_enabled else None
            ),
            "acf_channel_lag_bank_selection": (
                "abs_acf_times_legal_step_fraction"
                if channel_lag_bank_enabled else None
            ),
            "acf_channel_lag_bank_legal_lags": channel_lag_bank_legal_lags,
            "acf_channel_lag_bank_legal_lag_count": int(
                len(channel_lag_bank_legal_lags)
            ),
            "acf_channel_lag_bank_legal_step_fractions": (
                channel_lag_bank_legal_step_fractions
            ),
            "acf_channel_lag_bank_selected_slots": int(channel_lag_bank_slots),
            "acf_channel_lag_bank_total_slots": int(channel_lag_bank_total_slots),
            "acf_channel_lag_bank_density": float(
                channel_lag_bank_slots
                / max(1, channel_lag_bank_total_slots)
            ) if channel_lag_bank_enabled else 0.0,
            "acf_channel_lag_bank_parameters": 0,
            "acf_channel_lag_bank_trainable_parameters": 0,
            "acf_channel_lag_bank_mask_bytes": int(channel_lag_bank_bytes),
            "acf_channel_lag_bank_mask_bytes_fp32_equivalent": int(
                channel_lag_bank_fp32_bytes
            ),
            "variable_scan_streams_per_batch_item": int(variable_scan_streams * direction_multiplier),
            "variable_scan_tokens_per_stream": int(variable_scan_tokens),
            "variable_token_channels": int(self.channels if self.variable_mixer_enabled else 0),
            "variable_token_group_size": int(
                self.variable_group_size if self.variable_mixer_enabled else 0
            ),
            "variable_token_group_count": int(variable_scan_tokens),
            "variable_token_group_padding": int(
                self.variable_group_padding if self.variable_mixer_enabled else 0
            ),
            "variable_scan_token_reduction": float(
                1.0 - (variable_scan_tokens / self.channels)
            ) if self.variable_mixer_enabled and self.channels else 0.0,
            "variable_scan_blocks_per_stream": int(
                self.variable_mixer_depth if self.variable_mixer_enabled else 0
            ),
            "variable_scan_operations_per_batch_item": int(
                variable_scan_streams * self.variable_mixer_depth * direction_multiplier
                if self.variable_mixer_enabled else 0
            ),
            "variable_repr_fusion_requested": bool(
                getattr(self, "variable_repr_fusion_requested", False)
            ),
            "variable_repr_fusion_enabled": variable_repr_fusion_active,
            "variable_repr_fusion_path_active": variable_repr_fusion_active,
            "variable_repr_fusion_pooling": (
                "softmax_channel_weighted_mean" if variable_repr_fusion_active else None
            ),
            "variable_repr_fusion_input_shape": (
                "B,C,D" if variable_repr_fusion_active else None
            ),
            "variable_repr_fusion_context_shape": (
                "B,1,D -> B,T,D" if variable_repr_fusion_active else None
            ),
            "variable_repr_fusion_parameters": variable_repr_fusion_parameters,
            "variable_repr_fusion_trainable_parameters": (
                variable_repr_fusion_trainable_parameters
            ),
            "variable_repr_fusion_macs_per_batch_item": variable_repr_fusion_macs,
            "variable_repr_fusion_activation_elements_per_batch_item": (
                variable_repr_fusion_activation_elements
            ),
            "variable_repr_fusion_source": (
                "normalized_observed_window_variable_mamba_states"
                if variable_repr_fusion_active else None
            ),
            "variable_repr_fusion_forecast_path_shared": bool(
                variable_repr_fusion_active
            ),
            "variable_repr_fusion_relation_gradient_path": bool(
                variable_repr_fusion_active
            ),
            "variable_gdd_mixer_enabled": bool(
                getattr(self, "variable_gdd_mixer_enabled", False)
            ),
            "variable_gdd_mixer_path_active": variable_gdd_mixer_active,
            "variable_gdd_mixer_requested_rank": int(
                getattr(self, "variable_gdd_mixer_rank", 8)
            ) if getattr(self, "variable_gdd_mixer_enabled", False) else 0,
            "variable_gdd_mixer_init_scale": float(
                getattr(self, "variable_gdd_mixer_init_scale", 0.0)
            ) if getattr(self, "variable_gdd_mixer_enabled", False) else 0.0,
            "variable_gdd_mixer_rank": variable_gdd_mixer_rank,
            "variable_gdd_mixer_input_shape": (
                "B,C,1,D" if variable_gdd_mixer_active else None
            ),
            "variable_gdd_mixer_output_shape": (
                "B,C,D" if variable_gdd_mixer_active else None
            ),
            "variable_gdd_mixer_summary": (
                "per_channel_mean_and_rms_over_singleton_patch" 
                if variable_gdd_mixer_active else None
            ),
            "variable_gdd_mixer_channel_preserving": bool(
                variable_gdd_mixer_active
            ),
            "variable_gdd_mixer_dense_channel_matrix": False,
            "variable_gdd_mixer_parameters": variable_gdd_mixer_parameters,
            "variable_gdd_mixer_trainable_parameters": (
                variable_gdd_mixer_trainable_parameters
            ),
            "variable_gdd_mixer_summary_macs_per_batch_item": (
                variable_gdd_mixer_summary_macs
            ),
            "variable_gdd_mixer_affine_macs_per_batch_item": (
                variable_gdd_mixer_affine_macs
            ),
            "variable_gdd_mixer_macs_per_batch_item": variable_gdd_mixer_macs,
            "variable_gdd_mixer_activation_elements_per_batch_item": (
                variable_gdd_mixer_activation_elements
            ),
            "variable_gdd_mixer_autocon_shared_state": bool(
                variable_gdd_mixer_active and variable_repr_fusion_active
            ),
            "variable_gdd_mixer_cost_estimate_basis": (
                "two_low_rank_channel_projections_plus_featurewise_affine; "
                "excludes_reduction_nonlinearity_and_kernel_overhead"
                if variable_gdd_mixer_active else None
            ),
            # This adapter is a window-conditioned global channel-state mixer.
            # It emits a [B,T,D] hidden residual consumed by both forecasting
            # and AutoCon.
            "data_dependent_channel_mixer_enabled": bool(
                getattr(self, "data_dependent_channel_mixer_enabled", False)
            ),
            "data_dependent_channel_mixer_path_active": (
                data_dependent_channel_mixer_active
            ),
            "data_dependent_channel_mixer_requested_rank": int(
                getattr(self, "data_dependent_channel_mixer_rank", 8)
            ) if getattr(self, "data_dependent_channel_mixer_enabled", False)
            else 0,
            "data_dependent_channel_mixer_rank": int(
                data_dependent_channel_mixer.rank
                if data_dependent_channel_mixer_active else 0
            ),
            "data_dependent_channel_mixer_summary_width": int(
                data_mixer_summary_width
            ),
            "data_dependent_channel_mixer_state_length": int(
                data_mixer_state_length
            ),
            "data_dependent_channel_mixer_tokenizer": (
                "existing_scale_gate_hidden_summary"
                if data_dependent_channel_mixer_active else None
            ),
            "data_dependent_channel_mixer_tokenizer_input_width": int(
                data_mixer_tokenizer_input_width
            ),
            "data_dependent_channel_mixer_input_shape": (
                "B,T,D" if data_dependent_channel_mixer_active else None
            ),
            "data_dependent_channel_mixer_summary_shape": (
                "B,D" if data_dependent_channel_mixer_active else None
            ),
            "data_dependent_channel_mixer_residual_shape": (
                "B,T,D" if data_dependent_channel_mixer_active else None
            ),
            "data_dependent_channel_mixer_source": (
                "shared_temporal_gate_summary_from_normalized_observed_window"
                if data_dependent_channel_mixer_active else None
            ),
            "data_dependent_channel_mixer_state_embedding_parameters": int(
                sum(parameter.numel() for parameter in channel_state_embedding.parameters())
                if channel_state_embedding is not None else 0
            ),
            "data_dependent_channel_mixer_summary_projection_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in data_dependent_channel_mixer.summary_to_latent.parameters()
                ) if data_dependent_channel_mixer_active else 0
            ),
            "data_dependent_channel_mixer_hidden_affine_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in data_dependent_channel_mixer.latent_to_affine.parameters()
                ) if data_dependent_channel_mixer_active else 0
            ),
            "data_dependent_channel_mixer_parameters": int(
                sum(parameter.numel() for parameter in data_dependent_channel_mixer.parameters())
                if data_dependent_channel_mixer_active else 0
            ),
            "data_dependent_channel_mixer_trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in data_dependent_channel_mixer.parameters()
                    if parameter.requires_grad
                ) if data_dependent_channel_mixer_active else 0
            ),
            "data_dependent_channel_mixer_autocon_shared_state": bool(
                data_dependent_channel_mixer_active
            ),
            "data_dependent_channel_mixer_state_embedding_macs_per_batch_item": int(
                data_mixer_state_embedding_macs
            ),
            "data_dependent_channel_mixer_summary_projection_macs_per_batch_item": int(
                data_mixer_summary_projection_macs
            ),
            "data_dependent_channel_mixer_hidden_affine_macs_per_batch_item": int(
                data_mixer_hidden_affine_macs
            ),
            "data_dependent_channel_mixer_adapter_linear_macs_per_batch_item": int(
                data_mixer_adapter_linear_macs
            ),
            "data_dependent_channel_mixer_affine_macs_per_batch_item": int(
                data_mixer_affine_macs
            ),
            "data_dependent_channel_mixer_macs_per_batch_item": int(
                data_mixer_total_macs
            ),
            "data_dependent_channel_mixer_tokenizer_scalar_ops_per_batch_item": int(
                data_mixer_tokenizer_scalar_ops
            ),
            "data_dependent_channel_mixer_summary_scalar_ops_per_batch_item": int(
                data_mixer_summary_ops
            ),
            "data_dependent_channel_mixer_pooling_scalar_ops_per_batch_item": int(
                data_mixer_pooling_ops
            ),
            "data_dependent_channel_mixer_activation_elements_per_batch_item": int(
                data_mixer_activation_elements
            ),
            "data_dependent_channel_mixer_cost_estimate_basis": (
                "linear_and_affine_only; excludes_bias_nonlinearity_reduction_backend"
            ),
            "variable_tokens_use_train_acf_prior": bool(
                self.acf_channel_conditioning
            ),
            "acf_channel_summary_width": (
                4 if self.acf_channel_conditioning else 0
            ),
            "variable_token_order": self.acf_channel_order,
            "variable_token_order_uses_train_acf_prior": bool(
                self.acf_channel_order in {"strength", "similarity"}
            ),
            "variable_token_order_rule": {
                "input": "identity_csv_order",
                "strength": "dominant_abs_acf_peak",
                "similarity": "signed_acf_profile_cosine_greedy",
            }[self.acf_channel_order],
            "variable_token_order_prior_attached": bool(
                self.acf_prior.numel() > 0
            ),
            "acf_channel_gate_enabled": bool(self.acf_channel_gate_enabled),
            "acf_channel_gate_summary_width": (
                4 if self.acf_channel_gate_enabled else 0
            ),
            "acf_channel_gate_prior_attached": bool(
                self.acf_channel_gate_enabled and self.acf_prior.numel() > 0
            ),
            "acf_channel_gate_path_active": bool(
                self.acf_channel_gate_enabled and self.acf_prior.numel() > 0
            ),
            "acf_channel_gate_latent_rank": int(
                self.channel_gate.rank if self.acf_channel_gate_enabled else 0
            ),
            "acf_channel_gate_parameters": int(
                self.channel_gate.acf_parameter_count
                if self.acf_channel_gate_enabled and self.channel_gate is not None
                else 0
            ),
            "acf_representation_fusion_requested": bool(
                self.acf_repr_fusion_requested
            ),
            "acf_representation_fusion_enabled": bool(
                self.acf_repr_fusion_enabled
            ),
            "acf_representation_context_width": (
                4 if self.acf_repr_fusion_enabled else 0
            ),
            # The index lookup is causal for every source.  The residual/trend
            # transform is centered over the complete encoder window and is
            # therefore only strictly causal for the raw source.
            "acf_representation_lag_index_is_causal": bool(
                self.acf_repr_fusion_enabled
            ),
            "acf_representation_transform_is_causal": bool(
                self.acf_repr_fusion_enabled and self.acf_source == "raw"
            ),
            # This legacy field describes only the optional ACF descriptor.
            # ``representation_grid_is_strictly_causal`` above is the
            # conservative end-to-end state-grid property.
            "acf_representation_context_is_causal": bool(
                self.acf_repr_fusion_enabled and self.acf_source == "raw"
            ),
            "acf_representation_prior_attached": bool(
                self.acf_repr_fusion_enabled and self.acf_prior.numel() > 0
            ),
            "acf_representation_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.acf_repr_projection.parameters()
                )
                + self.acf_repr_scale.numel()
                if self.acf_repr_fusion_enabled
                else 0
            ),
            "acf_horizon_calibration_enabled": bool(
                self.acf_horizon_calibration_enabled
            ),
            "acf_horizon_calibration_rank": int(
                self.acf_horizon_calibration_rank
                if self.acf_horizon_calibration_enabled else 0
            ),
            "acf_horizon_calibration_lag_count": int(
                len(self.acf_lags) if self.acf_horizon_calibration_enabled else 0
            ),
            "acf_horizon_calibration_parameters": int(
                self.acf_horizon_calibration.parameter_count
                if self.acf_horizon_calibration is not None else 0
            ),
            "acf_horizon_calibration_basis": (
                "constant_linear"
                if self.acf_horizon_calibration_enabled else None
            ),
            "acf_horizon_calibration_static_only": bool(
                self.acf_horizon_calibration_enabled
            ),
            # Reliability is either one horizon-aggregated channel scalar or
            # a causal per-step/channel tensor.  Persist the effective mode so
            # resource and checkpoint audits do not infer it from a CLI default.
            "acf_step_reliability": bool(
                getattr(self, "acf_step_reliability", False)
            ),
            "acf_step_reliability_path_active": bool(
                getattr(self, "acf_step_reliability", False)
                and self.use_acf_memory
                and bool(self.acf_lags)
                and (self.dynamic_acf or self.acf_prior.numel() > 0)
            ),
            "acf_reliability_mode": (
                "stepwise" if getattr(self, "acf_step_reliability", False)
                else "aggregate"
            ),
            # Per-step/channel reliability interpolation is a scalar-gated
            # adapter over static train-prior statistics.  It has no dense
            # forecast projection or input-dependent ACF estimation cost.
            "acf_reliability_interpolation_enabled": bool(
                getattr(self, "acf_reliability_interpolation", False)
            ),
            "acf_reliability_interpolation_parameters": int(
                self.acf_reliability_interpolation_scale.numel()
                if getattr(self, "acf_reliability_interpolation_scale", None)
                is not None else 0
            ),
            "acf_reliability_interpolation_trainable_parameters": int(
                self.acf_reliability_interpolation_scale.numel()
                if getattr(self, "acf_reliability_interpolation_scale", None)
                is not None
                and self.acf_reliability_interpolation_scale.requires_grad
                else 0
            ),
            "acf_reliability_interpolation_static_only": bool(
                getattr(self, "acf_reliability_interpolation", False)
            ),
            "acf_reliability_interpolation_source": (
                "train_prior_and_causal_validity"
                if getattr(self, "acf_reliability_interpolation", False)
                else None
            ),
            "acf_reliability_interpolation_path_active": bool(
                getattr(self, "acf_reliability_interpolation", False)
                and self.use_acf_memory
                and not self.dynamic_acf
                and self.acf_prior.numel() > 0
            ),
            "acf_memory_gate_mode": getattr(self, "acf_memory_gate_mode", "hidden"),
            "acf_memory_gate_variable_token": bool(
                getattr(self, "acf_memory_gate_mode", "hidden") == "variable_token"
            ),
            # These are architecture-level costs for the shared ``D -> H``
            # gate.  They exclude bias adds and backend/kernel overhead.  A
            # variable-token gate emits one value per channel and horizon, so
            # its parameter count is unchanged but its activation/MAC budget
            # scales with ``C``; parameter equality must not be presented as
            # capacity or latency equality.
            "acf_memory_gate_parameters": int(
                sum(parameter.numel() for parameter in self.acf_gate.parameters())
                if hasattr(self, "acf_gate") else 0
            ),
            "acf_memory_gate_trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in self.acf_gate.parameters()
                    if parameter.requires_grad
                )
                if hasattr(self, "acf_gate") else 0
            ),
            "acf_memory_gate_output_elements_per_batch_item": int(
                self.pred_len
                * (self.channels
                   if getattr(self, "acf_memory_gate_mode", "hidden") == "variable_token"
                   else 1)
            ),
            "acf_memory_gate_activation_shape": (
                "B,H,C"
                if getattr(self, "acf_memory_gate_mode", "hidden") == "variable_token"
                else "B,H,1"
            ),
            "acf_memory_gate_input_tokens": int(
                self.channels
                if getattr(self, "acf_memory_gate_mode", "hidden") == "variable_token"
                else 1
            ),
            "acf_memory_gate_macs_per_batch_item": int(
                self.d_model * self.pred_len * (
                    self.channels
                    if getattr(self, "acf_memory_gate_mode", "hidden") == "variable_token"
                    else 1
                )
            ),
            "acf_memory_gate_activation_bytes_fp32_per_batch_item": int(
                self.pred_len
                * (self.channels
                   if getattr(self, "acf_memory_gate_mode", "hidden") == "variable_token"
                   else 1)
                * 4
            ),
            # Direct-anchor retrieval is an indexed, train-prior lookup rather
            # than a learned projection.  The scalar ReZero adapter is the only
            # added trainable capacity; selection/indexing has no dense MACs.
            "acf_anchor_copy_enabled": bool(
                getattr(self, "acf_anchor_copy", False)
            ),
            "acf_anchor_copy_parameters": int(
                self.acf_anchor_scale.numel()
                if getattr(self, "acf_anchor_scale", None) is not None else 0
            ),
            "acf_anchor_copy_trainable_parameters": int(
                self.acf_anchor_scale.numel()
                if getattr(self, "acf_anchor_scale", None) is not None
                and self.acf_anchor_scale.requires_grad else 0
            ),
            "acf_anchor_copy_selection": (
                "per_channel_max_abs_acf" if getattr(
                    self, "acf_anchor_copy", False
                ) else None
            ),
            # The sparse channel lag bank constrains only soft forecast
            # memory.  Direct-anchor selection intentionally remains an
            # independent full configured-lag expert; expose this explicitly
            # whenever both optional branches are requested.
            "acf_anchor_copy_lag_support": (
                "full_configured_lag_bank"
                if getattr(self, "acf_anchor_copy", False)
                else None
            ),
            "acf_anchor_copy_uses_channel_lag_bank": False,
            "acf_anchor_copy_source": (
                str(getattr(self, "acf_source", "raw"))
                if getattr(self, "acf_anchor_copy", False) else None
            ),
            "acf_anchor_copy_index_is_causal": bool(
                getattr(self, "acf_anchor_copy", False)
            ),
            "acf_anchor_copy_output_shape": (
                "B,H,C" if getattr(self, "acf_anchor_copy", False) else None
            ),
            "acf_anchor_copy_macs_per_batch_item": 0,
            "patch_scan_streams_per_batch_item": int(patch_scan_streams * direction_multiplier),
            "patch_scan_tokens_per_stream": int(patch_scan_tokens),
            "patch_scan_blocks_per_stream": int(
                self.patch_mixer_depth if self.patch_mixer_enabled else 0
            ),
            "patch_scan_operations_per_batch_item": int(
                patch_scan_streams * self.patch_mixer_depth * direction_multiplier
                if self.patch_mixer_enabled else 0
            ),
            "local_patch_enabled": bool(self.local_mixer_enabled),
            "local_patch_len": int(self.local_patch_len if self.local_mixer_enabled else 0),
            "local_patch_stride": int(
                self.local_patch_stride if self.local_mixer_enabled else 0
            ),
            "local_patch_count": int(
                self.local_patch_count if self.local_mixer_enabled else 0
            ),
            "local_patch_width": int(self.local_width if self.local_mixer_enabled else 0),
            "local_patch_depth": int(self.local_depth if self.local_mixer_enabled else 0),
            "local_patch_rank": int(self.local_rank if self.local_mixer_enabled else 0),
            "local_patch_tokens_per_batch_item": int(
                self.local_patch_count * self.channels
                if self.local_mixer_enabled else 0
            ),
            "local_patch_channel_preserving": bool(self.local_mixer_enabled),
            "local_patch_cross_channel_parameters": 0,
            "local_patch_forecast_parameters": int(
                sum(parameter.numel() for parameter in self.local_head.parameters())
                if self.local_mixer_enabled else 0
            ),
            "local_patch_representation_parameters": int(
                sum(parameter.numel() for parameter in self.local_repr_projection.parameters())
                + (self.local_repr_scale.numel() if self.local_mixer_enabled else 0)
                if self.local_mixer_enabled else 0
            ),
            "shared_mamba_component_channel_rank_bound": (
                0 if self.channel_independent else self.d_model
            ),
            "linear_residual_is_channel_preserving": True,
            "channel_preserving_forecast_branches": preserving_branches,
            "output_projection_cross_channel": bool(self.c_out != self.channels),
        }

    def forward(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False,
        return_repr=None,
    ):
        self._reset_acf_runtime_diagnostics()
        normalized, restore = self._normalize(x_enc)
        # ``None`` preserves the historical contract.  An explicit ``False``
        # is used on prediction-only paths to avoid materializing the full
        # representation grid and AutoCon projection head.
        need_repr = bool(onlyrepr or (
            self.AutoCon if return_repr is None else return_repr
        ))
        need_encoder_repr = bool(
            need_repr and not self.forecast_relation_representation_enabled
        )
        # Keep the compressed state on the forecast path.  AutoCon's temporal
        # labels still require a full-grid representation, so that grid is
        # requested only when the representation is actually returned.
        hidden_base, hidden_full = self._encode_channels(
            normalized,
            x_mark_enc,
            return_base=True,
            return_full=need_encoder_repr,
        )
        local_tokens = (
            self._local_patch_features(normalized)
            if self.local_mixer_enabled else None
        )
        # The local representation adapter is also part of the forecast state
        # for AutoCon models, so retain it even when the auxiliary output is
        # disabled.  The full-grid-only ACF adapter below remains skipped.
        local_context = self._local_representation_context(local_tokens)
        variable_tokens = (
            self._variable_token_features(normalized)
            if self.variable_repr_fusion_enabled and need_repr
            else None
        )
        hidden_base = self._fuse_local_representation(
            hidden_base, local_tokens, context=local_context
        )
        contrastive_repr = None
        if need_encoder_repr:
            if hidden_full is None:
                hidden_full = hidden_base
            hidden_full = self._fuse_local_representation(
                hidden_full, local_tokens, context=local_context
            )
            hidden_full = self._fuse_variable_representation(
                hidden_full, variable_tokens
            )
            # Relation supervision is computed from this representation.  The
            # optional causal ACF context is strictly an auxiliary path and is
            # therefore omitted on prediction-only inference.
            hidden_full = self._fuse_acf_representation(hidden_full, normalized)
            contrastive_repr = self._contrastive_repr(hidden_full)
        if onlyrepr and not self.forecast_relation_representation_enabled:
            return None, contrastive_repr
        prediction = self._forecast_normalized(
            hidden_base,
            normalized,
            local_tokens=local_tokens,
            variable_tokens=variable_tokens,
        )
        if self.forecast_relation_representation_enabled and need_repr:
            # This global-only relation view is the final normalized forecast,
            # including every enabled causal ACF forecast residual.  It has no
            # learned adapter, so relation gradients cannot bypass the
            # prediction heads through a separate representation projection.
            contrastive_repr = prediction
        if onlyrepr:
            return None, contrastive_repr
        prediction = self._restore(prediction, restore)
        if self.c_out != self.channels:
            prediction = self.output_projection(prediction)
        if self.AutoCon:
            return prediction, contrastive_repr
        return prediction

    def get_embeddings(
        self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False,
    ):
        if self.forecast_relation_representation_enabled:
            _, representation = self.forward(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                enc_self_mask=enc_self_mask,
                dec_self_mask=dec_self_mask,
                dec_enc_mask=dec_enc_mask,
                onlyrepr=True,
            )
            return representation
        self._reset_acf_runtime_diagnostics()
        normalized, _ = self._normalize(x_enc)
        _, hidden = self._encode_channels(
            normalized, x_mark_enc, return_base=True, return_full=True
        )
        local_tokens = (
            self._local_patch_features(normalized)
            if self.local_mixer_enabled else None
        )
        variable_tokens = (
            self._variable_token_features(normalized)
            if self.variable_repr_fusion_enabled else None
        )
        local_context = self._local_representation_context(local_tokens)
        hidden = self._fuse_local_representation(
            hidden, local_tokens, context=local_context
        )
        hidden = self._fuse_variable_representation(hidden, variable_tokens)
        hidden = self._fuse_acf_representation(hidden, normalized)
        return self._contrastive_repr(hidden)

    def get_embedddings(self, *args, **kwargs):
        return self.get_embeddings(*args, **kwargs)


class Model(_MambaACFBase):
    """Mamba-2 ACF-guided forecaster."""

    mamba_kind = "mamba2"

    def __init__(self, configs):
        super().__init__(configs)


__all__ = ["Model"]
