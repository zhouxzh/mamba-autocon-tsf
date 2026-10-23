"""MambaTS-inspired patch/variable-scan forecaster.

The public MambaTS implementation uses patch tokens, a variable-aware scan
and a per-variable linear prediction head.  This local implementation keeps
    those identifiable choices while using the repository's installed
    ``mamba_ssm`` package and the common forecasting/AutoCon interface.  It is a
    reproducible baseline, not a claim of bit-level equivalence to the external
    repository's vendored VAST/VPT implementation.

    ``mamba_ts_variant=v1`` is the original local baseline.  ``v2`` keeps the
    same patch and Mamba-1 components but scans each variable's patch sequence
    independently, adds zero-initialized variable/patch identity parameters,
    and exposes a zero-initialized persistence residual.  The separate
    variant is intentional: old v1 checkpoints and negative results remain
    reproducible after the architectural experiment.  ``v3`` keeps the
    efficient shared scan, adds a zero-initialized low-rank cross-variable
    context, and uses a per-channel persistence gate.  ``v4`` adds a
    separately gated, train-only static ACF residual to that backbone; the
    ACF path is causal at forecast lookup time and does not estimate
    statistics from validation or test windows.  ``v5`` keeps the v4 path and
    adds one nested signed ACF anchor: each forecast step/channel selects the
    strongest available train-only lag, retrieves only an observed value, and
    applies a single zero-initialized ReZero scale. ``v6`` uses a DLinear-style
    base prediction with a zero-initialized Mamba correction. ``v7`` combines
    that linear base with v4's train-only ACF memory and AutoCon selector.
    ``v8`` returns to the efficient v3 backbone and adds a causal per-window
    Fourier residual whose prediction gate reads the AutoCon representation.
    ``v9`` follows the decomposition principle used by recent forecasting
    Mamba work: one shared Mamba scan models the seasonal residual while a
    shared linear head projects the smooth trend.  Its AutoCon representation
    controls only the fusion correction, so contrastive representation
    learning has a direct, auditable prediction path without a second scan.
    ``v10`` keeps the v3 single scan but applies one train-only signed-ACF
    channel ordering before the scan and restores the original order after it.
    This is a fixed VAST-lite approximation, not a reproduction of the
    official learned ATSP/VPT procedure.
    ``v11`` implements the key public VPT mechanism: random per-sample
    variable orders during training accumulate prediction-loss transition
    costs, and evaluation uses a greedy or python-tsp ATSP-derived order.
    ``v12`` independently reimplements the high-level ms-Mamba idea: two
    Mamba-1 branches consume the same patch-token sequence with distinct
    Delta initialization ranges, then a learned softmax combines their states.
    It is not a copy of the external implementation and keeps the fused state
    as the sole forecast and AutoCon representation input.
    ``v13`` keeps one Mamba scan and adds two depthwise temporal filters over
    the resulting patch states.  This is a lightweight multi-scale residual,
    intended to test whether the scale benefit can be retained without the
    second Mamba branch cost.
    ``v14`` keeps one scan over full-coverage coarse patches and applies two
    depthwise filters to the observed samples before patching, providing a
    cheap local expert without a second Mamba branch.
    ``v15`` keeps v3's one-scan backbone and injects pooled observed calendar
    features into each patch token through a zero-initialized ReZero gate.
    The conditioned state is shared by the forecast and AutoCon heads.
    ``v16`` uses the same calendar conditioning but gathers one historical
    start marker per patch, avoiding an extra sliding-window reduction.

    The optional ``mamba_ts_repr_linear_expert_mode=blend`` route keeps the
    v3 backbone but uses the AutoCon representation to form a bounded convex
    mixture of the Mamba forecast and a shared NLinear-style forecast.  The
    existing ``residual`` mode remains the historical additive expert.
    ``nlinear_blend`` uses a low-rank NLinear-style projection to keep this
    mixture cheap on high-dimensional inputs.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Autoformer_EncDec import series_decomp
from layers.LowRankDataDependentChannelMixer import LowRankGlobalChannelStateMixer
from utils.acf_relation_contract import resolve_acf_source


try:  # pragma: no cover - optional CUDA dependency
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc

try:  # pragma: no cover - optional CUDA dependency
    from mamba_ssm import Mamba2 as _Mamba2
    _MAMBA2_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba2 = None
    _MAMBA2_IMPORT_ERROR = exc

try:  # pragma: no cover - only required for the official VPT ATSP variants
    from python_tsp.heuristics import (
        solve_tsp_lin_kernighan as _solve_tsp_lin_kernighan,
        solve_tsp_local_search as _solve_tsp_local_search,
        solve_tsp_simulated_annealing as _solve_tsp_simulated_annealing,
    )
    _PYTHON_TSP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _solve_tsp_lin_kernighan = None
    _solve_tsp_local_search = None
    _solve_tsp_simulated_annealing = None
    _PYTHON_TSP_IMPORT_ERROR = exc


class _ResidualMamba1(nn.Module):
    """Pre-normalized Mamba-1 block with an explicit CUDA error message."""

    def __init__(
        self, d_model, d_state, d_conv, expand, dropout,
        dt_min=0.001, dt_max=0.1,
    ):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError(
                "MambaTS requires the optional mamba_ssm package (Mamba-1){}".format(
                    detail
                )
            )
        self.norm = nn.LayerNorm(int(d_model))
        self.mixer = _Mamba1(
            d_model=int(d_model),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            dt_min=float(dt_min),
            dt_max=float(dt_max),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x):
        try:
            mixed = self.mixer(self.norm(x))
        except RuntimeError as exc:
            message = str(exc).lower()
            if not x.is_cuda and ("cuda" in message or "triton" in message):
                raise RuntimeError(
                    "The installed mamba_ssm Mamba-1 build requires CUDA for "
                    "MambaTS; run with --use_gpu on a CUDA GPU."
                ) from exc
            raise
        return x + self.dropout(mixed)


def _compatible_mamba2_headdim(d_model, expand, requested):
    """Resolve a Mamba-2 head size that divides the expanded model width."""

    expanded = int(d_model) * int(expand)
    requested = int(requested)
    if expanded <= 0 or requested <= 0:
        raise ValueError("Mamba-2 dimensions must be positive")
    if expanded % requested == 0:
        return requested
    return max(
        divisor for divisor in range(min(expanded, requested), 0, -1)
        if expanded % divisor == 0
    )


class _ResidualMamba2(nn.Module):
    """Pre-normalized Mamba-2 block for the optional v3 mixer replacement."""

    def __init__(self, d_model, d_state, d_conv, expand, dropout, headdim):
        super().__init__()
        if _Mamba2 is None:
            detail = " ({})".format(_MAMBA2_IMPORT_ERROR) if _MAMBA2_IMPORT_ERROR else ""
            raise ImportError(
                "MambaTS Mamba-2 mixer requires the optional mamba_ssm package{}".format(
                    detail
                )
            )
        self.norm = nn.LayerNorm(int(d_model))
        self.mixer = _Mamba2(
            d_model=int(d_model),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            headdim=_compatible_mamba2_headdim(d_model, expand, headdim),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x):
        try:
            mixed = self.mixer(self.norm(x))
        except RuntimeError as exc:
            message = str(exc).lower()
            if not x.is_cuda and ("cuda" in message or "triton" in message):
                raise RuntimeError(
                    "The installed mamba_ssm Mamba-2 build requires CUDA for "
                    "MambaTS; run with --use_gpu on a CUDA GPU."
                ) from exc
            raise
        return x + self.dropout(mixed)


class Model(nn.Module):
    """Patch-token Mamba-1 with optional AutoCon representation output."""

    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        self.d_model = int(getattr(configs, "d_model", 64))
        self.repr_dims = int(getattr(configs, "d_ff", self.d_model * 2))
        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.patch_stride = int(getattr(configs, "patch_stride", 8))
        self.dropout_p = float(getattr(configs, "dropout", 0.1))
        self.depth = max(1, int(getattr(configs, "e_layers", 1)))
        self.AutoCon = bool(getattr(configs, "AutoCon", False))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "LastVal"))
        self.variant = str(getattr(configs, "mamba_ts_variant", "v1")).strip().lower()
        self.mamba_ts_mixer_type = str(
            getattr(configs, "mamba_ts_mixer_type", "mamba1")
        ).strip().lower()
        if self.mamba_ts_mixer_type not in {"mamba1", "mamba2"}:
            raise ValueError(
                "mamba_ts_mixer_type must be 'mamba1' or 'mamba2'; received {}".format(
                    self.mamba_ts_mixer_type
                )
            )
        if self.variant not in {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
            raise ValueError(
                "MambaTS mamba_ts_variant must be 'v1' through 'v16'; received {}".format(
                    self.variant
                )
            )
        self.channel_independent = self.variant == "v2"
        self.ts_variable_permutation_enabled = bool(
            self.variant in {"v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}
            and getattr(configs, "mamba_ts_variable_permutation", False)
        )
        # Cross-domain self-supervision reuses the production v3 patch and
        # Mamba weights, but must not learn a source-specific channel count or
        # sensor identity. This flag is used only by the dedicated pretraining
        # driver; normal forecasting keeps the historical v3 path unchanged.
        self.ts_shared_pretrain_mode = bool(
            getattr(configs, "mamba_ts_shared_pretrain_mode", False)
        )
        self.ts_channel_agnostic_scan = bool(
            getattr(configs, "mamba_ts_channel_agnostic_scan", False)
        )
        # v13 only adds shared depthwise filters over channel-local patch
        # states, so it retains v3's channel-count-agnostic transfer contract.
        if self.ts_channel_agnostic_scan and self.variant not in {"v3", "v13"}:
            raise ValueError("mamba_ts_channel_agnostic_scan requires MambaTS v3 or v13")

        if min(self.seq_len, self.pred_len, self.enc_in, self.c_out) <= 0:
            raise ValueError("MambaTS dimensions must be positive")
        if self.c_out > self.enc_in:
            raise ValueError("MambaTS requires c_out <= enc_in")
        if self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError("MambaTS patch_len and patch_stride must be positive")
        if self.patch_len > self.seq_len:
            raise ValueError("MambaTS patch_len cannot exceed seq_len")
        self.patch_count = 1 + (self.seq_len - self.patch_len) // self.patch_stride
        # ``F.interpolate(..., mode='linear', align_corners=False)`` is used
        # to restore patch representations to the observed time grid.  The
        # grid is fixed by the config, so cache its exact sparse weights and
        # use one GEMM instead of launching an interpolation kernel per call.
        interpolation_positions = (
            (torch.arange(self.seq_len, dtype=torch.float32) + 0.5)
            * float(self.patch_count) / float(self.seq_len)
            - 0.5
        )
        interpolation_left = interpolation_positions.floor().long().clamp(
            0, self.patch_count - 1
        )
        interpolation_right = (interpolation_left + 1).clamp(
            0, self.patch_count - 1
        )
        interpolation_weight = (
            interpolation_positions - interpolation_left.to(torch.float32)
        ).clamp(0.0, 1.0)
        interpolation_matrix = torch.zeros(
            self.seq_len, self.patch_count, dtype=torch.float32
        )
        interpolation_matrix.scatter_add_(
            1, interpolation_left[:, None], (1.0 - interpolation_weight)[:, None]
        )
        interpolation_matrix.scatter_add_(
            1, interpolation_right[:, None], interpolation_weight[:, None]
        )
        self.register_buffer(
            "representation_interpolation_matrix",
            interpolation_matrix,
            # It is derived exactly from the immutable architecture config;
            # keeping it out of checkpoints preserves historical state-dict
            # compatibility while model.to(device) still moves the cache.
            persistent=False,
        )

        d_state = int(getattr(configs, "mamba_d_state", 16))
        d_conv = int(getattr(configs, "mamba_d_conv", 4))
        expand = int(getattr(configs, "mamba_expand", 2))
        headdim = int(getattr(configs, "mamba_headdim", 16))
        if min(d_state, d_conv, expand) <= 0:
            raise ValueError("MambaTS Mamba dimensions must be positive")
        if self.mamba_ts_mixer_type == "mamba2" and self.variant == "v12":
            raise ValueError("MambaTS v12 Delta branches currently require mamba1 mixer")
        mixer_block = (
            _ResidualMamba2 if self.mamba_ts_mixer_type == "mamba2" else _ResidualMamba1
        )

        # The official baseline embeds each observed patch independently and
        # flattens all variable patch tokens for one Mamba scan.  The scan is
        # channel-major, matching the published MambaTS tensor ordering.
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.ts_acf_channel_descriptor_enabled = bool(
            self.AutoCon and self.variant == "v3"
            and getattr(configs, "mamba_ts_acf_channel_descriptor", False)
        )
        self.register_buffer(
            "ts_acf_descriptor_values", torch.zeros(self.enc_in, 6), persistent=True
        )
        if self.ts_acf_channel_descriptor_enabled:
            self.ts_acf_channel_descriptor_projection = nn.Sequential(
                nn.LayerNorm(6), nn.Linear(6, self.d_model, bias=False),
            )
            self.ts_acf_channel_descriptor_scale = nn.Parameter(torch.zeros(()))
        else:
            self.ts_acf_channel_descriptor_projection = None
            self.register_parameter("ts_acf_channel_descriptor_scale", None)
        self.ts_shared_channel_summary_enabled = bool(
            self.AutoCon
            and self.variant == "v3"
            and getattr(configs, "mamba_ts_shared_channel_summary", False)
        )
        self.ts_channel_set_pooling_enabled = bool(
            self.AutoCon and self.variant == "v3"
            and getattr(configs, "mamba_ts_channel_set_pooling", False)
        )
        self.ts_channel_prototype_pooling_enabled = bool(
            self.AutoCon and self.variant == "v3"
            and getattr(configs, "mamba_ts_channel_prototype_pooling", False)
        )
        if sum((self.ts_shared_channel_summary_enabled, self.ts_channel_set_pooling_enabled,
                self.ts_channel_prototype_pooling_enabled)) > 1:
            raise ValueError("shared channel context modes are mutually exclusive")
        if self.ts_shared_channel_summary_enabled:
            self.ts_shared_channel_summary_projection = nn.Sequential(
                nn.Linear(2 * self.d_model, self.d_model, bias=False),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model, bias=False),
            )
            self.ts_shared_channel_summary_scale = nn.Parameter(torch.zeros(()))
        else:
            self.ts_shared_channel_summary_projection = None
            self.register_parameter("ts_shared_channel_summary_scale", None)
        if self.ts_channel_set_pooling_enabled:
            self.ts_channel_set_pooling_score = nn.Linear(self.d_model, 1)
            self.ts_channel_set_pooling_projection = nn.Sequential(
                nn.Linear(2 * self.d_model, self.d_model, bias=False),
                nn.GELU(), nn.Linear(self.d_model, self.d_model, bias=False),
            )
            self.ts_channel_set_pooling_scale = nn.Parameter(torch.zeros(()))
        else:
            self.ts_channel_set_pooling_score = None
            self.ts_channel_set_pooling_projection = None
            self.register_parameter("ts_channel_set_pooling_scale", None)
        if self.ts_channel_prototype_pooling_enabled:
            self.ts_channel_prototype_count = int(getattr(configs, "mamba_ts_channel_prototype_count", 4))
            if self.ts_channel_prototype_count <= 1:
                raise ValueError("mamba_ts_channel_prototype_count must exceed one")
            self.ts_channel_prototype_queries = nn.Parameter(
                torch.empty(self.ts_channel_prototype_count, self.d_model)
            )
            nn.init.normal_(self.ts_channel_prototype_queries, std=self.d_model ** -0.5)
            self.ts_channel_prototype_projection = nn.Sequential(
                nn.Linear(self.ts_channel_prototype_count * self.d_model, self.d_model, bias=False),
                nn.GELU(), nn.Linear(self.d_model, self.d_model, bias=False),
            )
            self.ts_channel_prototype_scale = nn.Parameter(torch.zeros(()))
        else:
            self.ts_channel_prototype_count = None
            self.register_parameter("ts_channel_prototype_queries", None)
            self.ts_channel_prototype_projection = None
            self.register_parameter("ts_channel_prototype_scale", None)
        self.ts_time_condition_enabled = self.variant in {"v15", "v16"}
        self.ts_time_condition_anchor_enabled = self.variant == "v16"
        if self.ts_time_condition_enabled:
            # TimeFeature supplies at most six normalized calendar values
            # (second, minute, hour, weekday, month-day, year-day). Padding
            # makes one checkpoint work across hourly, minute and daily data.
            self.ts_time_condition_projection = nn.Linear(6, self.d_model, bias=False)
            self.ts_time_condition_scale = nn.Parameter(torch.zeros(()))
            starts = torch.arange(
                0, self.seq_len - self.patch_len + 1, self.patch_stride,
                dtype=torch.long,
            )
            self.register_buffer("ts_time_patch_starts", starts, persistent=True)
        else:
            self.ts_time_condition_projection = None
            self.register_parameter("ts_time_condition_scale", None)
            self.register_buffer("ts_time_patch_starts", torch.empty(0, dtype=torch.long), persistent=True)
        self.ts_pre_local_enabled = self.variant == "v14"
        if self.ts_pre_local_enabled:
            self.ts_pre_short_filter = nn.Conv1d(
                self.enc_in, self.enc_in, kernel_size=3, padding=1,
                groups=self.enc_in, bias=False,
            )
            self.ts_pre_long_filter = nn.Conv1d(
                self.enc_in, self.enc_in, kernel_size=5, padding=2,
                groups=self.enc_in, bias=False,
            )
            self.ts_pre_local_logits = nn.Parameter(torch.zeros(2))
            self.ts_pre_local_scale = nn.Parameter(torch.zeros(()))
        else:
            self.ts_pre_short_filter = None
            self.ts_pre_long_filter = None
            self.register_parameter("ts_pre_local_logits", None)
            self.register_parameter("ts_pre_local_scale", None)
        self.ts_multiscale_enabled = self.variant == "v12"
        multipliers = getattr(configs, "mamba_ts_delta_multipliers", (1.0, 2.0))
        try:
            multipliers = tuple(float(value) for value in multipliers)
        except TypeError as exc:
            raise ValueError(
                "mamba_ts_delta_multipliers must be an iterable of positive floats"
            ) from exc
        if self.ts_multiscale_enabled:
            if len(multipliers) != 2 or multipliers[0] != 1.0 or multipliers[1] <= 1.0:
                raise ValueError(
                    "MambaTS v12 requires mamba_ts_delta_multipliers=(1.0, scale) with scale > 1"
                )
            self.ts_delta_multipliers = multipliers
            base_dt_min = float(getattr(configs, "mamba_ts_dt_min", 0.001))
            base_dt_max = float(getattr(configs, "mamba_ts_dt_max", 0.1))
            if not 0 < base_dt_min < base_dt_max:
                raise ValueError("MambaTS v12 requires 0 < mamba_ts_dt_min < mamba_ts_dt_max")
            self.ts_multiscale_encoder = nn.ModuleList([
                nn.ModuleList([
                    _ResidualMamba1(
                        self.d_model, d_state, d_conv, expand, self.dropout_p,
                        dt_min=base_dt_min * multiplier,
                        dt_max=base_dt_max * multiplier,
                    )
                    for _ in range(self.depth)
                ])
                for multiplier in self.ts_delta_multipliers
            ])
            # Equal logits reproduce ms-Mamba's initial averaging.
            self.ts_multiscale_logits = nn.Parameter(torch.zeros(len(multipliers)))
            self.encoder = None
        else:
            self.ts_delta_multipliers = None
            self.ts_multiscale_encoder = None
            self.register_parameter("ts_multiscale_logits", None)
            self.encoder = nn.ModuleList([
                mixer_block(
                    self.d_model, d_state, d_conv, expand, self.dropout_p,
                    headdim=headdim,
                ) if self.mamba_ts_mixer_type == "mamba2" else mixer_block(
                    self.d_model, d_state, d_conv, expand, self.dropout_p
                )
                for _ in range(self.depth)
            ])

        self.ts_lite_multiscale_enabled = self.variant == "v13"
        if self.ts_lite_multiscale_enabled:
            self.ts_lite_short_filter = nn.Conv1d(
                self.d_model, self.d_model, kernel_size=3, padding=1,
                groups=self.d_model, bias=False,
            )
            self.ts_lite_long_filter = nn.Conv1d(
                self.d_model, self.d_model, kernel_size=5, padding=2,
                groups=self.d_model, bias=False,
            )
            self.ts_lite_multiscale_logits = nn.Parameter(torch.zeros(2))
            self.ts_lite_multiscale_scale = nn.Parameter(torch.zeros(()))
        else:
            self.ts_lite_short_filter = None
            self.ts_lite_long_filter = None
            self.register_parameter("ts_lite_multiscale_logits", None)
            self.register_parameter("ts_lite_multiscale_scale", None)
        self.final_norm = nn.LayerNorm(self.d_model)
        self.forecast_head = nn.Linear(self.patch_count * self.d_model, self.pred_len)

        if self.variant in {"v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
            # These parameters start at zero so the first forward pass remains
            # a fair functional extension of v1.  Training can learn sensor
            # and patch identity without imposing an arbitrary channel order.
            self.patch_position_embedding = nn.Parameter(
                torch.zeros(1, 1, self.patch_count, self.d_model)
            )
            self.channel_embedding = nn.Parameter(
                torch.zeros(1, self.enc_in, 1, self.d_model)
            )
            # A shared persistence path is useful for traffic-like signals and
            # is deliberately gated from zero.  The initialization copies the
            # last normalized value to every forecast step; the gate and head
            # remain trainable and are included in the v2 parameter count.
            self.persistence_head = nn.Linear(self.seq_len, self.pred_len)
            with torch.no_grad():
                self.persistence_head.weight.zero_()
                self.persistence_head.weight[:, -1] = 1.0
                self.persistence_head.bias.zero_()
            scale_shape = () if self.variant == "v2" else (1, 1, self.enc_in)
            self.persistence_scale = nn.Parameter(torch.zeros(scale_shape))
        else:
            self.patch_position_embedding = None
            self.channel_embedding = None
            self.persistence_head = None
            self.persistence_scale = None

        if self.variant in {"v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
            context_rank = int(getattr(configs, "mamba_ts_context_rank", 16))
            if context_rank <= 0:
                raise ValueError("mamba_ts_context_rank must be positive")
            self.context_rank = min(context_rank, self.enc_in, self.d_model)
            self.context_projection = nn.Sequential(
                nn.Linear(self.enc_in, self.context_rank, bias=False),
                nn.Linear(self.context_rank, self.d_model, bias=False),
            )
            # This is a residual adapter.  Zero initialization isolates its
            # effect from the measured v1/v2 scan until optimization updates it.
            nn.init.zeros_(self.context_projection[-1].weight)
        else:
            self.context_rank = None
            self.context_projection = None

        # v9 keeps the expensive operator to one seasonal Mamba scan.  Trend
        # extrapolation is a channel-shared linear map, which is cheap in both
        # the high-dimensional and long-horizon settings.  The decomposition
        # is calculated only from each observed normalized input window.
        self.ts_decomposition_enabled = self.variant == "v9"
        if self.ts_decomposition_enabled:
            moving_avg = int(getattr(configs, "moving_avg", 25))
            if moving_avg <= 0:
                raise ValueError("moving_avg must be positive for MambaTS v9")
            self.ts_decomposition = series_decomp(moving_avg)
            self.ts_trend_head = nn.Linear(self.seq_len, self.pred_len)
            with torch.no_grad():
                self.ts_trend_head.weight.fill_(1.0 / float(self.seq_len))
                self.ts_trend_head.bias.zero_()
            if self.AutoCon:
                self.ts_decomposition_gate = nn.Linear(self.repr_dims, self.pred_len)
                self.ts_decomposition_scale = nn.Parameter(torch.zeros(()))
            else:
                self.ts_decomposition_gate = None
                self.register_parameter("ts_decomposition_scale", None)
        else:
            self.ts_decomposition = None
            self.ts_trend_head = None
            self.ts_decomposition_gate = None
            self.register_parameter("ts_decomposition_scale", None)

        # v6 starts from an explicit DLinear-style decomposition forecast and
        # lets Mamba learn only a ReZero residual correction.  This targets the
        # low-dimensional coordinates where linear baselines beat the direct
        # Mamba head, while keeping the Mamba encoder available to AutoCon.
        self.linear_residual_enabled = self.variant in {"v6", "v7"}
        if self.linear_residual_enabled:
            moving_avg = int(getattr(configs, "moving_avg", 25))
            self.linear_decomposition = series_decomp(moving_avg)
            self.linear_seasonal_head = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend_head = nn.Linear(self.seq_len, self.pred_len)
            with torch.no_grad():
                initial = torch.full(
                    (self.pred_len, self.seq_len), 1.0 / self.seq_len
                )
                self.linear_seasonal_head.weight.copy_(initial)
                self.linear_trend_head.weight.copy_(initial)
            residual_init = float(getattr(configs, "mamba_ts_residual_init", 0.0))
            if not -0.99 < residual_init < 0.99:
                raise ValueError("mamba_ts_residual_init must be in (-0.99, 0.99)")
            self.mamba_residual_scale = nn.Parameter(torch.tensor(residual_init))
        else:
            self.linear_decomposition = None
            self.linear_seasonal_head = None
            self.linear_trend_head = None
            self.register_parameter("mamba_residual_scale", None)

        # v4 keeps the v3 forecast backbone and adds a compact train-prior
        # memory residual.  The prior is attached by the experiment wrapper
        # after the train split is available; empty buffers keep construction
        # and checkpoint loading deterministic before that point.
        self.ts_acf_memory_enabled = bool(
            self.variant in {"v4", "v5", "v7"}
            and getattr(configs, "mamba_use_acf_memory", False)
        )
        self.ts_acf_lags = self._parse_lags(
            getattr(configs, "mamba_acf_lags", [24, 168, 336])
        )
        # Keep the compact-prior interface aligned with MambaACF so the
        # runner can persist and replay the train-only v4/v5/v7 prior. v1--v3
        # retain an empty state and therefore never produce a sidecar.
        self.acf_lags = self.ts_acf_lags
        self.ts_acf_source = resolve_acf_source(configs)
        if self.ts_acf_source not in {"raw", "residual", "trend"}:
            raise ValueError(
                "AutoCon_acf_source must be raw, residual, or trend"
            )
        if self.variant == "v5" and self.ts_acf_source != "raw":
            raise ValueError("MambaTS v5 ACF anchor requires AutoCon_acf_source=raw")
        self.ts_acf_temperature = float(
            getattr(configs, "mamba_acf_temperature", 4.0)
        )
        if self.ts_acf_temperature <= 0:
            raise ValueError("mamba_acf_temperature must be positive")
        self.register_buffer(
            "ts_acf_prior", torch.empty(0, 0), persistent=True
        )
        self.register_buffer(
            "ts_acf_lag_values", torch.empty(0, dtype=torch.long), persistent=True
        )
        self.ts_acf_order_enabled = bool(
            self.variant == "v10" and getattr(configs, "mamba_use_acf_memory", False)
        )
        self.register_buffer(
            "ts_acf_channel_order",
            torch.arange(self.enc_in, dtype=torch.long),
            persistent=True,
        )
        # v11 mirrors MambaTS VPT's train-time random variable paths and
        # loss-derived transition table, while using a deterministic greedy
        # decoder rather than the external Python ATSP solvers.
        self.ts_vpt_enabled = self.variant == "v11"
        self.ts_vpt_solver = str(
            getattr(configs, "mamba_ts_vpt_solver", "greedy")
        ).strip().lower()
        if self.ts_vpt_solver not in {"greedy", "sa", "ls", "lk"}:
            raise ValueError(
                "mamba_ts_vpt_solver must be one of {greedy, sa, ls, lk}"
            )
        if self.ts_vpt_enabled and self.ts_vpt_solver != "greedy" and _PYTHON_TSP_IMPORT_ERROR is not None:
            raise ImportError(
                "MambaTS v11 ATSP solvers require python-tsp; install python-tsp "
                "or select mamba_ts_vpt_solver=greedy"
            ) from _PYTHON_TSP_IMPORT_ERROR
        self.register_buffer(
            "ts_vpt_transition_cost", torch.zeros(self.enc_in, self.enc_in),
            persistent=True,
        )
        self.register_buffer(
            "ts_vpt_transition_count", torch.zeros(self.enc_in, self.enc_in),
            persistent=True,
        )
        self.register_buffer(
            "ts_vpt_channel_order", torch.arange(self.enc_in, dtype=torch.long),
            persistent=True,
        )
        self._ts_vpt_batch_transitions = None
        self._ts_vpt_order_dirty = False
        if self.ts_acf_memory_enabled:
            # ReZero keeps v4 nested in v3 at initialization and lets the
            # optimizer decide whether a dataset benefits from the prior.
            self.ts_acf_memory_scale = nn.Parameter(
                torch.zeros(1, 1, self.enc_in)
            )
        else:
            self.register_parameter("ts_acf_memory_scale", None)

        # A static ACF memory can help periodic channels but can also inject an
        # inappropriate lag signal into an irregular window.  This optional
        # correction uses the same AutoCon representation that is optimized by
        # the contrastive objective to select a small horizon-wise adjustment.
        # It adds no channel-dependent parameters and starts as an exact v4/v5
        # memory control because its ReZero scale is zero.
        self.ts_acf_memory_selector_enabled = bool(
            self.variant in {"v4", "v5", "v7"}
            and self.AutoCon
            and self.ts_acf_memory_enabled
            and getattr(configs, "mamba_ts_acf_memory_selector", False)
        )
        if self.ts_acf_memory_selector_enabled:
            self.ts_acf_memory_selector = nn.Linear(
                self.repr_dims, self.pred_len, bias=True
            )
            self.ts_acf_memory_selector_scale = nn.Parameter(torch.zeros(()))
        else:
            self.ts_acf_memory_selector = None
            self.register_parameter("ts_acf_memory_selector_scale", None)

        # The horizon-only selector cannot distinguish a periodic sensor from
        # an irregular one. This shared probe reads each variable's existing
        # Mamba state and adds one confidence logit without C-dependent params.
        self.ts_acf_memory_channel_selector_enabled = bool(
            self.ts_acf_memory_selector_enabled
            and getattr(configs, "mamba_ts_acf_memory_channel_selector", False)
        )
        if self.ts_acf_memory_channel_selector_enabled:
            self.ts_acf_memory_channel_selector = nn.Linear(self.d_model, 1, bias=True)
        else:
            self.ts_acf_memory_channel_selector = None

        self.ts_acf_anchor_enabled = bool(self.variant == "v5")
        if self.ts_acf_anchor_enabled:
            # One scalar keeps the new expert strictly nested in v4 at
            # initialization without adding a channel-by-horizon table.
            self.ts_acf_anchor_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("ts_acf_anchor_scale", None)

        # Optional AutoCon-only route: expose the same train-only ACF prior to
        # the temporal representation, without changing the plain forecast.
        # The scale is ReZero so the existing v4 forecast/representation is a
        # strict nested control at initialization.
        self.ts_acf_repr_fusion_enabled = bool(
            self.variant in {"v4", "v5", "v7"}
            and self.AutoCon
            and self.ts_acf_memory_enabled
            and getattr(configs, "mamba_acf_repr_fusion", False)
        )
        if self.ts_acf_repr_fusion_enabled:
            if self.ts_acf_source != "raw":
                raise ValueError(
                    "MambaTS v4/v5 ACF representation fusion requires raw source"
                )
            self.ts_acf_repr_projection = nn.Linear(4, self.d_model, bias=False)
            nn.init.xavier_uniform_(self.ts_acf_repr_projection.weight)
            self.ts_acf_repr_scale = nn.Parameter(torch.zeros(()))
        else:
            self.ts_acf_repr_projection = None
            self.register_parameter("ts_acf_repr_scale", None)

        if self.AutoCon:
            # Relation learning receives a time-indexed summary over channels,
            # while the forecast head retains channel-specific patch states.
            self.repr_projection = nn.Linear(self.d_model, self.repr_dims)
            self.repr_norm = nn.LayerNorm(self.repr_dims)
            self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)
        else:
            self.repr_projection = None
            self.repr_norm = None
            self.repr_head = None

        # v8 addresses the periodic component missed by the direct patch
        # forecast.  The candidate is computed from each observed window only;
        # its ReZero blend is nested in the v3 control at initialization.  When
        # AutoCon is enabled, the same representation trained by the contrastive
        # objective supplies horizon and channel confidence gates.
        self.ts_spectral_enabled = self.variant == "v8"
        if self.ts_spectral_enabled:
            self.ts_spectral_top_k = int(
                getattr(configs, "mamba_ts_spectral_top_k", 4)
            )
            if self.ts_spectral_top_k <= 0:
                raise ValueError("mamba_ts_spectral_top_k must be positive")
            self.ts_spectral_scale = nn.Parameter(torch.zeros(()))
            if self.AutoCon:
                self.ts_spectral_selector = nn.Linear(self.repr_dims, self.pred_len)
                self.ts_spectral_channel_selector = nn.Linear(self.d_model, 1)
            else:
                self.ts_spectral_selector = None
                self.ts_spectral_channel_selector = None
        else:
            self.ts_spectral_top_k = None
            self.register_parameter("ts_spectral_scale", None)
            self.ts_spectral_selector = None
            self.ts_spectral_channel_selector = None

        # Low-dimensional series repeatedly favored DLinear/PatchTST over the
        # shared Mamba scan.  This optional expert is deliberately a shared
        # linear extrapolator, not a C-dependent fallback table.  Its
        # horizon-wise gate reads the exact AutoCon representation.  ReZero
        # makes the first forward pass bit-identical to the v3 control.
        self.ts_repr_linear_expert_enabled = bool(
            self.AutoCon
            and self.variant in {"v3", "v4", "v5", "v6", "v7", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}
            and getattr(configs, "mamba_ts_repr_linear_expert", False)
        )
        if self.ts_repr_linear_expert_enabled:
            self.ts_repr_linear_expert_mode = str(
                getattr(configs, "mamba_ts_repr_linear_expert_mode", "residual")
            ).strip().lower()
            if self.ts_repr_linear_expert_mode not in {"residual", "blend", "nlinear_blend"}:
                raise ValueError(
                    "mamba_ts_repr_linear_expert_mode must be residual, blend, or nlinear_blend"
                )
            # Blend mode reuses v3's already-computed persistence head.  The
            # legacy residual mode retains its independent full linear expert
            # for checkpoint compatibility and historical ablations.
            self.ts_linear_expert = (
                None
                if self.ts_repr_linear_expert_mode in {"blend", "nlinear_blend"}
                else nn.Linear(self.seq_len, self.pred_len)
            )
            if self.ts_repr_linear_expert_mode == "nlinear_blend":
                requested_rank = int(
                    getattr(configs, "mamba_ts_repr_linear_expert_rank", 16)
                )
                if requested_rank <= 0:
                    raise ValueError("mamba_ts_repr_linear_expert_rank must be positive")
                self.ts_repr_linear_expert_rank = min(requested_rank, self.seq_len, self.pred_len)
                self.ts_nlinear_left = nn.Linear(
                    self.seq_len, self.ts_repr_linear_expert_rank, bias=False
                )
                self.ts_nlinear_right = nn.Linear(
                    self.ts_repr_linear_expert_rank, self.pred_len
                )
            else:
                self.ts_repr_linear_expert_rank = None
                self.ts_nlinear_left = None
                self.ts_nlinear_right = None
            self.ts_repr_linear_gate = nn.Linear(self.repr_dims, self.pred_len)
            self.ts_repr_linear_expert_scale = nn.Parameter(torch.zeros(()))
            self.ts_repr_linear_blend_bias = float(
                getattr(configs, "mamba_ts_repr_linear_blend_bias", -4.0)
            )
            if not -12.0 <= self.ts_repr_linear_blend_bias <= 0.0:
                raise ValueError(
                    "mamba_ts_repr_linear_blend_bias must be in [-12, 0]"
                )
            if self.ts_repr_linear_expert_mode in {"blend", "nlinear_blend"}:
                # Start from a small, data-independent linear contribution;
                # the representation gate can then learn a per-horizon blend
                # without the sign ambiguity of the legacy tanh residual scale.
                nn.init.zeros_(self.ts_repr_linear_gate.weight)
                nn.init.zeros_(self.ts_repr_linear_gate.bias)
        else:
            self.ts_linear_expert = None
            self.ts_repr_linear_gate = None
            self.register_parameter("ts_repr_linear_expert_scale", None)
            self.ts_repr_linear_expert_mode = "disabled"
            self.ts_repr_linear_blend_bias = None
            self.ts_repr_linear_expert_rank = None
            self.ts_nlinear_left = None
            self.ts_nlinear_right = None

        # Earlier MambaTS AutoCon variants exposed a representation only to the
        # auxiliary loss.  That made a small contrastive weight affect the
        # forecast indirectly, through every encoder parameter.  This optional
        # low-rank temporal residual makes the *same* representation a direct
        # forecasting input.  ReZero preserves the exact legacy forecast at
        # initialization, while the learned shared temporal component remains
        # inexpensive for high-dimensional data.
        self.ts_repr_forecast_fusion_enabled = bool(
            self.AutoCon
            and self.variant in {"v3", "v4", "v5", "v6", "v7", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}
            and getattr(configs, "mamba_ts_repr_forecast_fusion", False)
        )
        self.ts_repr_forecast_channelwise = bool(
            self.ts_repr_forecast_fusion_enabled
            and getattr(configs, "mamba_ts_repr_forecast_channelwise", False)
        )
        if self.ts_repr_forecast_fusion_enabled:
            self.ts_repr_forecast_time = nn.Linear(
                (
                    self.patch_count
                    if self.ts_repr_forecast_channelwise
                    else self.seq_len
                ),
                self.pred_len,
                bias=False,
            )
            if self.ts_repr_forecast_channelwise:
                # Shared channelwise projections retain a variable's encoded
                # trajectory while keeping parameter count independent of C.
                self.ts_repr_channel_projection = nn.Linear(
                    self.d_model, self.repr_dims
                )
                self.ts_repr_channel_norm = nn.LayerNorm(self.repr_dims)
                self.ts_repr_channel_head = nn.Linear(
                    self.repr_dims, self.repr_dims
                )
                self.ts_repr_forecast_channels = nn.Linear(
                    self.repr_dims, 1, bias=False
                )
            else:
                self.ts_repr_channel_projection = None
                self.ts_repr_channel_norm = None
                self.ts_repr_channel_head = None
                self.ts_repr_forecast_channels = nn.Linear(
                    self.repr_dims, self.c_out, bias=False
                )
            self.ts_repr_forecast_scale = nn.Parameter(
                torch.zeros(1, 1, self.c_out)
            )
        else:
            self.ts_repr_forecast_time = None
            self.ts_repr_channel_projection = None
            self.ts_repr_channel_norm = None
            self.ts_repr_channel_head = None
            self.ts_repr_forecast_channels = None
            self.register_parameter("ts_repr_forecast_scale", None)

        # A channel-preserving forecast residual that reuses the existing
        # shared Mamba scan. Unlike the earlier channelwise representation
        # route, it pools patch states before its small shared projection, so
        # activation growth is O(B * C * (d_model + rank + pred_len)), not
        # O(B * C * patch_count * d_ff). It is deliberately available only
        # with AutoCon: both losses therefore update the same encoded states.
        self.ts_channel_adapter_enabled = bool(
            self.AutoCon
            and self.variant == "v3"
            and getattr(configs, "mamba_ts_channel_adapter", False)
        )
        if self.ts_channel_adapter_enabled:
            requested_rank = int(getattr(configs, "mamba_ts_channel_adapter_rank", 8))
            if requested_rank <= 0:
                raise ValueError("mamba_ts_channel_adapter_rank must be positive")
            self.ts_channel_adapter_rank = min(requested_rank, self.d_model)
            self.ts_channel_adapter_in = nn.Linear(
                self.d_model, self.ts_channel_adapter_rank, bias=False
            )
            self.ts_channel_adapter_out = nn.Linear(
                self.ts_channel_adapter_rank, self.pred_len, bias=False
            )
            # Horizon-specific gates let the residual remain absent at any
            # forecast step where the shared state is not useful.
            self.ts_channel_adapter_scale = nn.Parameter(
                torch.zeros(1, self.pred_len, 1)
            )
        else:
            self.ts_channel_adapter_rank = None
            self.ts_channel_adapter_in = None
            self.ts_channel_adapter_out = None
            self.register_parameter("ts_channel_adapter_scale", None)

        # A target-time nonlinear decoder over the already computed patch
        # states. It is shared across channels, has no sensor-identity
        # parameters, and is ReZero-gated so enabling it preserves the frozen
        # encoder forecast at initialization. The branch never launches an
        # additional Mamba scan.
        self.ts_target_mlp_adapter_enabled = bool(
            self.AutoCon
            and self.variant == "v3"
            and getattr(configs, "mamba_ts_target_mlp_adapter", False)
        )
        if self.ts_target_mlp_adapter_enabled:
            requested_rank = int(getattr(configs, "mamba_ts_target_mlp_adapter_rank", 32))
            if requested_rank <= 0:
                raise ValueError("mamba_ts_target_mlp_adapter_rank must be positive")
            self.ts_target_mlp_adapter_rank = requested_rank
            self.ts_target_mlp_adapter_in = nn.Linear(
                self.patch_count * self.d_model, requested_rank, bias=False
            )
            self.ts_target_mlp_adapter_out = nn.Linear(
                requested_rank, self.pred_len, bias=False
            )
            self.ts_target_mlp_adapter_scale = nn.Parameter(
                torch.zeros(1, self.pred_len, 1)
            )
        else:
            self.ts_target_mlp_adapter_rank = None
            self.ts_target_mlp_adapter_in = None
            self.ts_target_mlp_adapter_out = None
            self.register_parameter("ts_target_mlp_adapter_scale", None)

        # Optional MambaMixer-inspired channel-state path.  It conditions the
        # already-computed shared temporal state on a low-rank summary of all
        # observed variables, without launching a second Mamba scan.  The
        # AutoCon representation and forecast head consume the same state, so
        # the auxiliary objective cannot optimize a disconnected branch.
        self.ts_global_channel_mixer_enabled = bool(
            self.AutoCon
            and self.variant == "v3"
            and getattr(configs, "mamba_ts_global_channel_mixer", False)
        )
        if self.ts_global_channel_mixer_enabled:
            mixer_rank = int(
                getattr(configs, "mamba_ts_global_channel_mixer_rank", 4)
            )
            if mixer_rank <= 0:
                raise ValueError(
                    "mamba_ts_global_channel_mixer_rank must be positive"
                )
            self.ts_global_channel_mixer_rank = mixer_rank
            self.ts_global_channel_mixer = LowRankGlobalChannelStateMixer(
                channels=self.enc_in,
                dim=self.d_model,
                rank=mixer_rank,
                dropout=self.dropout_p,
            )
        else:
            self.ts_global_channel_mixer_rank = None
            self.ts_global_channel_mixer = None
        self.ts_shared_postscan_mixer_enabled = bool(
            self.AutoCon and self.variant == "v3"
            and getattr(configs, "mamba_ts_shared_postscan_mixer", False)
        )
        if self.ts_shared_postscan_mixer_enabled:
            rank = int(getattr(configs, "mamba_ts_shared_postscan_mixer_rank", 4))
            if rank <= 0:
                raise ValueError("mamba_ts_shared_postscan_mixer_rank must be positive")
            # The mixer sees a [B,P,D] shared state, not a [B,P,C] channel
            # vector. Its parameter shapes therefore depend only on D and
            # transfer across source/target channel counts.
            self.ts_shared_postscan_mixer_rank = rank
            self.ts_shared_postscan_mixer = LowRankGlobalChannelStateMixer(
                channels=self.d_model, dim=self.d_model, rank=rank, dropout=self.dropout_p,
            )
        else:
            self.ts_shared_postscan_mixer_rank = None
            self.ts_shared_postscan_mixer = None

        # Traffic-graph extension for the fast v3 path.  The graph contract
        # stores a row-normalized directed edge list, so this branch aggregates
        # only observed neighboring sensors without materialising a dense
        # [C,C] matrix.  Its scalar is ReZero-initialized: graph-on remains
        # exactly nested in the non-graph v3 control before optimization.
        self.ts_graph_spatial_enabled = bool(
            self.AutoCon
            and self.variant == "v3"
            and getattr(configs, "mamba_ts_graph_spatial", False)
        )
        self.register_buffer(
            "ts_graph_target", torch.empty(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "ts_graph_source", torch.empty(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "ts_graph_weight", torch.empty(0), persistent=True
        )
        if self.ts_graph_spatial_enabled:
            graph_path = getattr(configs, "mamba_ts_graph_path", None)
            expected_hash = getattr(configs, "mamba_ts_graph_expected_sha256", "")
            target, source, weight = self._load_graph_contract(
                graph_path, expected_hash
            )
            self.ts_graph_target = target
            self.ts_graph_source = source
            self.ts_graph_weight = weight
            self.ts_graph_spatial_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("ts_graph_spatial_scale", None)

        # Optional reversible affine adapter after window normalization.  The
        # identity initialization keeps the enabled branch exactly nested in
        # the existing v3 AutoCon forecast while allowing per-channel scale
        # and shift to be learned from the training split.
        self.ts_revin_affine_enabled = bool(
            self.AutoCon
            and self.variant == "v3"
            and getattr(configs, "mamba_ts_revin_affine", False)
            and self.AutoCon_wnorm.strip().lower() in {"meanstd", "revin"}
        )
        if self.ts_revin_affine_enabled:
            self.ts_revin_log_scale = nn.Parameter(
                torch.zeros(1, 1, self.enc_in)
            )
            self.ts_revin_shift = nn.Parameter(torch.zeros(1, 1, self.enc_in))
        else:
            self.register_parameter("ts_revin_log_scale", None)
            self.register_parameter("ts_revin_shift", None)

        # Keep the ACF memory path auditable during training.  Values remain
        # detached scalar tensors until the experiment loop snapshots them, so
        # the diagnostic path does not add a per-batch host synchronization.
        self._ts_acf_runtime_diagnostics = {
            "acf_runtime_gate_mean": None,
            "acf_effective_memory_rms": None,
            "acf_effective_memory_to_forecast_rms": None,
            "spectral_runtime_gate_mean": None,
            "spectral_candidate_to_forecast_rms": None,
            "decomposition_runtime_gate_mean": None,
        }

    def _reset_ts_acf_runtime_diagnostics(self):
        for key in self._ts_acf_runtime_diagnostics:
            self._ts_acf_runtime_diagnostics[key] = None

    def _load_graph_contract(self, path, expected_hash):
        """Load a safe, aligned directed graph contract for the input nodes."""

        if path is None or not str(path).strip():
            raise ValueError("mamba_ts_graph_spatial requires mamba_ts_graph_path")
        graph_path = Path(str(path)).expanduser()
        if not graph_path.is_file():
            raise FileNotFoundError(
                "MambaTS graph contract is missing: {}".format(graph_path)
            )
        expected_hash = str(expected_hash or "").strip().lower()
        if expected_hash:
            digest = hashlib.sha256()
            with graph_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError("MambaTS graph contract hash mismatch")
        with np.load(str(graph_path), allow_pickle=False) as payload:
            required = {
                "schema_version", "sensor_ids", "target", "source", "weight",
                "node_count",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(
                    "MambaTS graph contract is missing fields: {}".format(
                        ", ".join(missing)
                    )
                )
            schema = np.asarray(payload["schema_version"])
            count = np.asarray(payload["node_count"])
            if schema.ndim != 0 or int(schema.item()) != 1:
                raise ValueError("unsupported MambaTS graph contract schema")
            if count.ndim != 0 or int(count.item()) != self.enc_in:
                raise ValueError(
                    "MambaTS graph node count does not match enc_in"
                )
            sensor_ids = np.asarray(payload["sensor_ids"])
            target = np.asarray(payload["target"])
            source = np.asarray(payload["source"])
            weight = np.asarray(payload["weight"], dtype=np.float32)
        if (
            sensor_ids.ndim != 1
            or sensor_ids.size != self.enc_in
            or len({str(value) for value in sensor_ids.tolist()}) != self.enc_in
        ):
            raise ValueError("MambaTS graph sensor IDs do not match node count")
        if (
            target.ndim != 1
            or source.ndim != 1
            or weight.ndim != 1
            or target.size == 0
            or target.size != source.size
            or target.size != weight.size
        ):
            raise ValueError("MambaTS graph edges must be non-empty aligned vectors")
        if (
            target.dtype.kind not in {"i", "u"}
            or source.dtype.kind not in {"i", "u"}
            or target.min() < 0
            or source.min() < 0
            or target.max() >= self.enc_in
            or source.max() >= self.enc_in
            or not np.isfinite(weight).all()
            or np.any(weight < 0)
        ):
            raise ValueError("MambaTS graph edge values are invalid")
        return (
            torch.as_tensor(target, dtype=torch.long),
            torch.as_tensor(source, dtype=torch.long),
            torch.as_tensor(weight, dtype=torch.float32),
        )

    def _graph_spatial_propagate(self, values):
        """Apply one directed edge-list propagation to an observed window."""

        if not self.ts_graph_spatial_enabled:
            return values
        if values.ndim != 3 or int(values.shape[-1]) != self.enc_in:
            raise ValueError(
                "MambaTS graph propagation expects [batch, time, enc_in]"
            )
        if self.ts_graph_target.numel() == 0:
            raise RuntimeError("MambaTS graph branch has no loaded edges")
        source = self.ts_graph_source.to(device=values.device)
        target = self.ts_graph_target.to(device=values.device)
        weight = self.ts_graph_weight.to(device=values.device, dtype=values.dtype)
        messages = values.index_select(2, source) * weight.view(1, 1, -1)
        propagated = values.new_zeros(values.shape)
        propagated.index_add_(2, target, messages)
        scale = torch.tanh(self.ts_graph_spatial_scale).to(
            device=values.device, dtype=values.dtype
        )
        return values + scale * (propagated - values)

    def _set_ts_acf_runtime_diagnostic(self, key, value):
        if value is None:
            self._ts_acf_runtime_diagnostics[key] = None
            return
        if torch.is_tensor(value):
            value = value.detach().float()
            if key.endswith("rms"):
                value = value.square().mean().sqrt()
            else:
                value = value.mean()
            self._ts_acf_runtime_diagnostics[key] = value.reshape(())
        else:
            self._ts_acf_runtime_diagnostics[key] = float(value)

    def acf_runtime_diagnostics(self):
        """Return scalar telemetry for the latest v4/v5/v7 forecast path."""

        diagnostics = dict(self._ts_acf_runtime_diagnostics)
        for key, value in diagnostics.items():
            if torch.is_tensor(value):
                diagnostics[key] = float(value.detach().cpu().item())
        vpt = self.vpt_runtime_diagnostics()
        if vpt is not None:
            diagnostics["vpt_transition_coverage"] = vpt["vpt_transition_coverage"]
            diagnostics["vpt_transition_count"] = vpt["vpt_transition_count"]
        return diagnostics

    @staticmethod
    def _parse_lags(values):
        if isinstance(values, (str, bytes)):
            values = values.decode() if isinstance(values, bytes) else values
            values = [item for item in values.replace(",", " ").split() if item]
        try:
            parsed = sorted({int(item) for item in list(values) if int(item) > 0})
        except (TypeError, ValueError) as exc:
            raise ValueError("mamba_acf_lags must contain positive integers") from exc
        return tuple(parsed)

    def set_ts_acf_prior(self, acf, lags):
        """Attach a signed ACF table fitted from the training split only."""

        prior = torch.as_tensor(acf, dtype=torch.float32).detach()
        parsed_lags = self._parse_lags(lags)
        if not parsed_lags:
            raise ValueError("MambaTS train-only ACF prior requires at least one lag")
        if prior.ndim != 2 or prior.shape[0] != self.enc_in:
            raise ValueError(
                "MambaTS train-only ACF prior must have shape [enc_in, lag_count]; got {}".format(
                    tuple(prior.shape)
                )
            )
        if prior.shape[1] != len(parsed_lags):
            # ``fit_signed_acf_prior`` returns the complete train-only table
            # including lag zero.  Compress it by the resolved lag identity so
            # the forecast path stores only the slots it actually consumes.
            if max(parsed_lags) >= prior.shape[1]:
                raise ValueError(
                    "MambaTS train-only ACF prior has {} columns but requires lag {}".format(
                        prior.shape[1], max(parsed_lags)
                    )
                )
            prior = prior.index_select(
                1, torch.as_tensor(parsed_lags, dtype=torch.long)
            )
        device = next(self.parameters()).device
        self.ts_acf_prior = prior.to(device=device)
        self.ts_acf_lag_values = torch.as_tensor(
            parsed_lags, dtype=torch.long, device=device
        )
        self.ts_acf_lags = parsed_lags
        self.acf_lags = parsed_lags
        if self.ts_acf_order_enabled:
            # Build the order once from the train-only profile.  The greedy
            # path is deterministic and avoids a per-forward Python search.
            profile = self.ts_acf_prior
            profile = profile / profile.norm(dim=1, keepdim=True).clamp_min(1e-6)
            similarity = profile @ profile.transpose(0, 1)
            start = int(profile.abs().sum(dim=1).argmax().item())
            visited = torch.zeros(self.enc_in, dtype=torch.bool, device=device)
            order = []
            current = start
            for _ in range(self.enc_in):
                order.append(current)
                visited[current] = True
                if len(order) == self.enc_in:
                    break
                scores = similarity[current].masked_fill(visited, -torch.inf)
                current = int(scores.argmax().item())
            self.ts_acf_channel_order = torch.as_tensor(order, dtype=torch.long, device=device)

    def export_acf_prior_state(self):
        """Return the compact v4/v5 train-only prior for sidecar storage."""

        if self.ts_acf_prior.numel() == 0:
            return None
        lags = self.ts_acf_lag_values.detach().cpu().numpy().astype(
            np.int64, copy=True
        )
        values = self.ts_acf_prior.detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        if (
            values.ndim != 2
            or values.shape[0] != self.enc_in
            or values.shape[1] != lags.size
            or lags.size == 0
            or not np.isfinite(values).all()
        ):
            raise RuntimeError("attached MambaTS v4/v5 ACF prior is invalid")
        return {
            "schema_version": 1,
            "acf_values": values,
            "acf_full_values": None,
            "lags": lags,
            "seq_len": int(self.seq_len),
            "pred_len": int(self.pred_len),
            "channels": int(self.enc_in),
            "acf_source": str(self.ts_acf_source),
            "acf_temperature": float(self.ts_acf_temperature),
        }

    def save_acf_prior(self, path):
        """Persist a compact train-only MambaTS v4/v5 prior as ``.npz``."""

        state = self.export_acf_prior_state()
        if state is None:
            return False
        np.savez_compressed(
            str(path),
            schema_version=np.asarray(state["schema_version"], dtype=np.int64),
            acf_values=state["acf_values"],
            lags=state["lags"],
            seq_len=np.asarray(state["seq_len"], dtype=np.int64),
            pred_len=np.asarray(state["pred_len"], dtype=np.int64),
            channels=np.asarray(state["channels"], dtype=np.int64),
            acf_source=np.asarray(state["acf_source"]),
            acf_temperature=np.asarray(state["acf_temperature"], dtype=np.float64),
        )
        return True

    def load_acf_prior(self, path):
        """Load a compact MambaTS v4/v5 sidecar without refitting ACF."""

        with np.load(str(path), allow_pickle=False) as payload:
            required = {
                "schema_version", "acf_values", "lags", "seq_len", "pred_len",
                "channels", "acf_source", "acf_temperature",
            }
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError("ACF prior artifact is missing fields: {}".format(", ".join(missing)))

            def scalar_int(name):
                value = np.asarray(payload[name])
                if value.ndim != 0 or value.dtype.kind not in {"i", "u"}:
                    raise ValueError("ACF prior artifact {} must be an integer scalar".format(name))
                return int(value.item())

            if scalar_int("schema_version") != 1:
                raise ValueError("unsupported ACF prior artifact schema")
            if scalar_int("seq_len") != self.seq_len or scalar_int("pred_len") != self.pred_len:
                raise ValueError("ACF prior artifact forecast geometry does not match MambaTS")
            if scalar_int("channels") != self.enc_in:
                raise ValueError("ACF prior artifact channel count does not match MambaTS")
            source = str(np.asarray(payload["acf_source"]).item())
            if source != self.ts_acf_source:
                raise ValueError("ACF prior artifact source does not match MambaTS")
            temperature = float(np.asarray(payload["acf_temperature"]).item())
            if not np.isfinite(temperature) or not math.isclose(
                temperature, self.ts_acf_temperature, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("ACF prior artifact temperature does not match MambaTS")
            lags_raw = np.asarray(payload["lags"])
            if lags_raw.dtype.kind not in {"i", "u"}:
                raise ValueError("ACF prior artifact lags must use an integer dtype")
            lags = self._parse_lags(lags_raw.reshape(-1).tolist())
            if tuple(lags_raw.reshape(-1).tolist()) != lags:
                raise ValueError("ACF prior artifact lags must be positive, unique, and ascending")
            values = np.asarray(payload["acf_values"], dtype=np.float32)
            if (
                values.ndim != 2
                or values.shape != (self.enc_in, len(lags))
                or not np.isfinite(values).all()
                or np.any(np.abs(values) > 1.0 + 1e-6)
            ):
                raise ValueError("ACF prior artifact values do not match MambaTS")
        self.set_ts_acf_prior(values, lags)
        return True

    def _ts_acf_memory_signal(self, x):
        if self.ts_acf_source == "raw":
            return x
        kernel = max(3, self.seq_len + 1)
        if kernel % 2 == 0:
            kernel += 1
        values = x.transpose(1, 2)
        pad = (kernel - 1) // 2
        padded = torch.cat(
            (
                values[:, :, :1].expand(-1, -1, pad),
                values,
                values[:, :, -1:].expand(-1, -1, pad),
            ),
            dim=-1,
        )
        trend = F.avg_pool1d(padded, kernel_size=kernel, stride=1)
        if self.ts_acf_source == "trend":
            return trend.transpose(1, 2)
        return (values - trend).transpose(1, 2)

    def _ts_acf_memory(self, normalized, apply_scale=True):
        if (
            not self.ts_acf_memory_enabled
            or self.ts_acf_prior.numel() == 0
            or self.ts_acf_lag_values.numel() == 0
        ):
            return normalized.new_zeros(
                normalized.shape[0], self.pred_len, self.enc_in
            )
        if self.ts_acf_prior.shape != (
            self.enc_in, self.ts_acf_lag_values.numel()
        ):
            raise RuntimeError("attached MambaTS v4/v5 ACF prior does not match model channels/lags")
        horizon = self.pred_len
        lags = self.ts_acf_lag_values.to(device=normalized.device)
        steps = torch.arange(
            1, horizon + 1, device=normalized.device, dtype=torch.long
        )
        indices = self.seq_len - 1 + steps[:, None] - lags[None, :]
        valid = (indices >= 0) & (indices < self.seq_len)
        safe_indices = indices.clamp(0, max(0, self.seq_len - 1))
        prior = self.ts_acf_prior.to(device=normalized.device, dtype=normalized.dtype)
        # Softmax weighting follows ACF magnitude while preserving sign.  The
        # static prior is immutable during forward and invalid pairs contribute
        # exactly zero.
        logits = torch.log(prior.abs().clamp_min(1e-6)).transpose(0, 1)
        logits = logits.unsqueeze(0).expand(horizon, -1, -1)
        logits = logits / self.ts_acf_temperature
        logits = logits.masked_fill(
            ~valid[:, :, None], torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits, dim=1)
        weights = weights * valid[:, :, None].to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-4)
        coefficients = weights * prior.transpose(0, 1).sign()
        source = self._ts_acf_memory_signal(normalized)
        memory = normalized.new_zeros(
            normalized.shape[0], horizon, self.enc_in
        )
        for lag_index in range(int(lags.numel())):
            values = source.index_select(1, safe_indices[:, lag_index])
            memory = memory + values * coefficients[:, lag_index, :].view(
                1, horizon, self.enc_in
            ) * valid[:, lag_index].view(1, horizon, 1).to(
                dtype=values.dtype
            )
        if not apply_scale:
            return memory
        scale = torch.tanh(self.ts_acf_memory_scale).to(
            device=memory.device, dtype=memory.dtype
        )
        return memory * scale

    def _ts_acf_anchor(self, normalized):
        """Retrieve a causal strongest-lag anchor for MambaTS v5.

        The selected lag comes only from the frozen train-split ACF prior.  A
        forecast step can use a historical input index only when that index is
        inside the observed window; invalid early steps are masked to zero.
        """

        if (
            not self.ts_acf_anchor_enabled
            or self.ts_acf_prior.numel() == 0
            or self.ts_acf_lag_values.numel() == 0
        ):
            return normalized.new_zeros(
                normalized.shape[0], self.pred_len, self.enc_in
            )
        prior = self.ts_acf_prior.to(
            device=normalized.device, dtype=normalized.dtype
        )
        lags = self.ts_acf_lag_values.to(device=normalized.device)
        horizon = int(self.pred_len)
        channels = int(self.enc_in)
        steps = torch.arange(
            1, horizon + 1, device=normalized.device, dtype=torch.long
        )
        indices = self.seq_len - 1 + steps[:, None] - lags[None, :]
        valid = (indices >= 0) & (indices < self.seq_len)
        safe_indices = indices.clamp(0, max(0, self.seq_len - 1))
        scores = prior.abs().unsqueeze(0).expand(horizon, -1, -1)
        scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
        selected = scores.argmax(dim=-1)
        index_grid = safe_indices[:, None, :].expand(horizon, channels, -1)
        anchor_indices = index_grid.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        sign_grid = prior.sign().unsqueeze(0).expand(horizon, -1, -1)
        anchor_sign = sign_grid.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        valid_grid = valid[:, None, :].expand(horizon, channels, -1)
        anchor_valid = valid_grid.gather(2, selected.unsqueeze(-1)).squeeze(-1)
        gathered = normalized.gather(
            dim=1,
            index=anchor_indices.unsqueeze(0).expand(normalized.shape[0], -1, -1),
        )
        scale = torch.tanh(self.ts_acf_anchor_scale).to(
            device=normalized.device, dtype=normalized.dtype
        )
        return gathered * anchor_sign.unsqueeze(0) * anchor_valid.unsqueeze(0).to(
            dtype=normalized.dtype
        ) * scale

    def _ts_acf_representation_context(self, normalized):
        """Build a causal train-prior context for the AutoCon representation."""

        if (
            not self.ts_acf_repr_fusion_enabled
            or self.ts_acf_prior.numel() == 0
            or self.ts_acf_lag_values.numel() == 0
        ):
            return None
        if normalized.ndim != 3 or normalized.shape[-1] != self.enc_in:
            raise ValueError("MambaTS v4/v5 ACF representation context expects [B,L,C]")
        batch, length, channels = normalized.shape
        prior = self.ts_acf_prior.to(
            device=normalized.device, dtype=normalized.dtype
        )
        lags = self.ts_acf_lag_values.to(device=normalized.device)
        if tuple(prior.shape) != (self.enc_in, int(lags.numel())):
            raise RuntimeError("attached MambaTS v4/v5 ACF prior does not align with representation lags")
        absolute_mass = prior.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        coefficients = prior / absolute_mass
        seasonal = normalized.new_zeros((batch, length, channels))
        lag_mass = prior.abs().mean(dim=0)
        lag_mass = lag_mass / lag_mass.sum().clamp_min(1e-6)
        reliability = normalized.new_zeros((length,))
        positions = torch.arange(length, device=normalized.device, dtype=torch.long)
        for lag_index in range(int(lags.numel())):
            lag = int(lags[lag_index].item())
            indices = positions - lag
            valid = indices >= 0
            values = normalized.index_select(1, indices.clamp_min(0))
            seasonal = seasonal + values * coefficients[:, lag_index].view(
                1, 1, channels
            ) * valid.view(1, length, 1).to(dtype=normalized.dtype)
            reliability = reliability + lag_mass[lag_index] * valid.to(
                dtype=normalized.dtype
            )
        summary = torch.stack(
            (
                seasonal.mean(dim=-1),
                seasonal.abs().mean(dim=-1),
                seasonal.square().mean(dim=-1).sqrt(),
                reliability.view(1, length).expand(batch, -1),
            ),
            dim=-1,
        )
        projected = self.ts_acf_repr_projection(summary)
        scale = torch.tanh(self.ts_acf_repr_scale).to(
            device=projected.device, dtype=projected.dtype
        )
        return projected * scale

    def _normalize(self, x):
        mode = self.AutoCon_wnorm.strip().lower()
        mean = x.mean(dim=1, keepdim=True).detach()
        if mode in {"meanstd", "revin"}:
            centered = x - mean
            scale = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            normalized = centered / scale
            if self.ts_revin_affine_enabled:
                gamma = torch.exp(self.ts_revin_log_scale).to(
                    device=normalized.device, dtype=normalized.dtype
                )
                beta = self.ts_revin_shift.to(
                    device=normalized.device, dtype=normalized.dtype
                )
                return normalized * gamma + beta, (mean, scale, gamma, beta)
            return normalized, (mean, scale)
        if mode == "mean":
            return x - mean, mean
        if mode == "lastval":
            last = x[:, -1:, :].detach()
            return x - last, last
        raise ValueError(
            "MambaTS supports AutoCon_wnorm in "
            "{MeanStd, ReVIN, Mean, LastVal}; received {}".format(
                self.AutoCon_wnorm
            )
        )

    @staticmethod
    def _restore(prediction, restore, c_out):
        # Forecast heads operate on all input channels so the multivariate
        # path stays channel-preserving; MS/S replay must slice the requested
        # target before broadcasting the stored normalization statistics.
        prediction = prediction[:, :, -c_out:]
        if isinstance(restore, tuple):
            if len(restore) == 4:
                mean, scale, gamma, beta = restore
                gamma = gamma[:, :, -c_out:]
                beta = beta[:, :, -c_out:]
                prediction = (prediction - beta) / gamma
                return prediction * scale[:, :, -c_out:] + mean[:, :, -c_out:]
            mean, scale = restore
            return prediction * scale[:, :, -c_out:] + mean[:, :, -c_out:]
        return prediction + restore[:, :, -c_out:]

    def _vpt_eval_order(self, device):
        """Resolve a VAST path from train-only transition cost."""

        if not self.ts_vpt_enabled:
            return None
        if self._ts_vpt_order_dirty:
            count = self.ts_vpt_transition_count
            if bool((count > 0).any()):
                mean_cost = self.ts_vpt_transition_cost / count.clamp_min(1.0)
                unseen = count <= 0
                penalty = mean_cost[~unseen].max().detach() if bool((~unseen).any()) else mean_cost.new_tensor(1.0)
                costs = mean_cost.masked_fill(unseen, penalty + 1.0)
                if self.ts_vpt_solver == "greedy":
                    diagonal = torch.diagonal(costs)
                    current = int(torch.argmin(diagonal).item())
                    visited = torch.zeros(self.enc_in, dtype=torch.bool, device=costs.device)
                    order = []
                    for _ in range(self.enc_in):
                        order.append(current)
                        visited[current] = True
                        if len(order) == self.enc_in:
                            break
                        candidates = costs[current].masked_fill(visited, torch.inf)
                        current = int(torch.argmin(candidates).item())
                else:
                    # Match the public MambaTS VPT route construction: choose
                    # a train-derived start node, adapt the directed matrix to
                    # python-tsp's cycle API, then restore the chosen start.
                    start = int(torch.argmin(torch.diagonal(costs)).item())
                    distance = costs.detach().float().cpu().numpy().copy()
                    np.fill_diagonal(distance, 0.0)
                    distance[:, start] = 0.0
                    distance[:, [start, 0]] = distance[:, [0, start]]
                    distance[[start, 0], :] = distance[[0, start], :]
                    initial = list(range(self.enc_in))
                    if self.ts_vpt_solver == "sa":
                        order, _ = _solve_tsp_simulated_annealing(distance, x0=initial)
                    elif self.ts_vpt_solver == "ls":
                        order, _ = _solve_tsp_local_search(distance, x0=initial)
                    else:
                        order, _ = _solve_tsp_lin_kernighan(distance, x0=initial)
                    order = list(order)
                    order[order.index(start)] = 0
                    order[0] = start
                self.ts_vpt_channel_order = torch.as_tensor(order, device=costs.device, dtype=torch.long)
            self._ts_vpt_order_dirty = False
        return self.ts_vpt_channel_order.to(device=device, dtype=torch.long)

    @torch.no_grad()
    def update_vpt_transition_cost(self, mse):
        """Accumulate public-VPT-style train prediction costs after a batch."""

        if not self.ts_vpt_enabled or not self.training or self._ts_vpt_batch_transitions is None:
            return
        if mse.ndim != 3 or mse.shape[0] != self._ts_vpt_batch_transitions.shape[0]:
            raise ValueError("MambaTS v11 VPT cost must have shape [batch, horizon, channels]")
        sample_cost = mse.detach().float().mean(dim=(1, 2))
        # The official code rescales each batch's detached per-sample error.
        # ``clamp_min`` retains that ordering while making batch size one safe.
        sample_cost = sample_cost / sample_cost.var(unbiased=False).sqrt().clamp_min(1e-6)
        transitions = self._ts_vpt_batch_transitions.to(device=sample_cost.device)
        src, dst = transitions[..., 0], transitions[..., 1]
        flat_index = (src * self.enc_in + dst).reshape(-1)
        self.ts_vpt_transition_cost.view(-1).scatter_add_(0, flat_index, sample_cost.repeat_interleave(src.shape[1]))
        self.ts_vpt_transition_count.view(-1).scatter_add_(0, flat_index, torch.ones_like(sample_cost).repeat_interleave(src.shape[1]))
        self._ts_vpt_order_dirty = True

    def vpt_runtime_diagnostics(self):
        """Return immutable training-state coverage for VPT audit logs."""

        if not self.ts_vpt_enabled:
            return None
        count = self.ts_vpt_transition_count
        return {
            "vpt_transition_coverage": float((count > 0).float().mean().detach().cpu()),
            "vpt_transition_count": float(count.sum().detach().cpu()),
            "vpt_channel_order": self.ts_vpt_channel_order.detach().cpu().tolist(),
            "vpt_solver": self.ts_vpt_solver,
        }

    def _encode(self, normalized, x_mark_enc=None):
        # [B,L,C] -> [B,C,P,K] -> [B*C,P,D].
        observed = normalized
        if self.ts_pre_local_enabled:
            local = normalized.transpose(1, 2)
            short = self.ts_pre_short_filter(local)
            long = self.ts_pre_long_filter(local)
            weights = torch.softmax(self.ts_pre_local_logits, dim=0)
            local = weights[0] * short + weights[1] * long
            observed = normalized + torch.tanh(self.ts_pre_local_scale) * local.transpose(1, 2)
        patches = observed.transpose(1, 2).unfold(
            dimension=-1, size=self.patch_len, step=self.patch_stride
        )
        tokens = self.patch_embedding(patches)
        batch, channels, patch_count, _ = tokens.shape
        if patch_count != self.patch_count:
            raise RuntimeError(
                "MambaTS patch count changed from {} to {}".format(
                    self.patch_count, patch_count
                )
            )
        if self.variant in {"v2", "v3", "v4", "v5", "v6", "v7", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
            tokens = tokens + self.patch_position_embedding
            if self.ts_acf_channel_descriptor_enabled:
                if self.ts_acf_descriptor_values.shape != (channels, 6):
                    raise ValueError("ACF channel descriptor does not match the input channel count")
                descriptor = self.ts_acf_channel_descriptor_projection(
                    self.ts_acf_descriptor_values.to(device=tokens.device, dtype=tokens.dtype)
                )
                scale = torch.tanh(self.ts_acf_channel_descriptor_scale).to(
                    device=tokens.device, dtype=tokens.dtype
                )
                tokens = tokens + scale * descriptor.view(1, channels, 1, self.d_model)
            if self.ts_shared_channel_summary_enabled:
                channel_mean = tokens.mean(dim=1)
                channel_std = tokens.std(dim=1, unbiased=False)
                summary = torch.cat((channel_mean, channel_std), dim=-1)
                summary = self.ts_shared_channel_summary_projection(summary)
                scale = torch.tanh(self.ts_shared_channel_summary_scale).to(
                    device=tokens.device, dtype=tokens.dtype
                )
                tokens = tokens + scale * summary.unsqueeze(1)
            if self.ts_channel_prototype_pooling_enabled:
                scores = torch.einsum("bcpd,kd->bcpk", tokens, self.ts_channel_prototype_queries)
                weights = torch.softmax(scores, dim=1)
                prototypes = torch.einsum("bcpk,bcpd->bpkd", weights, tokens)
                summary = self.ts_channel_prototype_projection(prototypes.flatten(start_dim=2))
                scale = torch.tanh(self.ts_channel_prototype_scale).to(device=tokens.device, dtype=tokens.dtype)
                tokens = tokens + scale * summary.unsqueeze(1)
            if self.ts_channel_set_pooling_enabled:
                scores = self.ts_channel_set_pooling_score(tokens).squeeze(-1)
                weights = torch.softmax(scores, dim=1)
                pooled_mean = (weights.unsqueeze(-1) * tokens).sum(dim=1)
                centered = tokens - pooled_mean.unsqueeze(1)
                pooled_var = (weights.unsqueeze(-1) * centered.square()).sum(dim=1)
                summary = torch.cat((pooled_mean, pooled_var.clamp_min(0).sqrt()), dim=-1)
                summary = self.ts_channel_set_pooling_projection(summary)
                scale = torch.tanh(self.ts_channel_set_pooling_scale).to(
                    device=tokens.device, dtype=tokens.dtype
                )
                tokens = tokens + scale * summary.unsqueeze(1)
            if not (self.ts_shared_pretrain_mode or self.ts_channel_agnostic_scan):
                tokens = tokens + self.channel_embedding
            if self.ts_time_condition_enabled and x_mark_enc is not None:
                if x_mark_enc.ndim != 3 or x_mark_enc.shape[:2] != (batch, self.seq_len):
                    raise ValueError(
                        "MambaTS v15 x_mark_enc must have shape [batch, seq_len, features]"
                    )
                marks = x_mark_enc.to(dtype=tokens.dtype)
                if marks.shape[-1] > 6:
                    marks = marks[..., :6]
                elif marks.shape[-1] < 6:
                    marks = F.pad(marks, (0, 6 - marks.shape[-1]))
                if self.ts_time_condition_anchor_enabled:
                    mark_patches = marks.index_select(1, self.ts_time_patch_starts)
                else:
                    mark_patches = marks.transpose(1, 2).unfold(
                        dimension=-1, size=self.patch_len, step=self.patch_stride
                    ).mean(dim=-1).transpose(1, 2)
                condition = self.ts_time_condition_projection(mark_patches)
                tokens = tokens + torch.tanh(self.ts_time_condition_scale) * condition[:, None, :, :]
            if self.variant == "v2":
                # A separate sequence per variable prevents the recurrent state
                # from leaking from one sensor's final patch into the next sensor.
                sequence = tokens.reshape(
                    batch * channels, patch_count, self.d_model
                )
            else:
                # v3 retains one efficient shared scan, but exposes a learned
                # cross-variable context at every temporal patch.
                if not (self.ts_shared_pretrain_mode or self.ts_channel_agnostic_scan):
                    context = self.context_projection(observed)
                    context = context.transpose(1, 2).unfold(
                        dimension=-1, size=self.patch_len, step=self.patch_stride
                    ).mean(dim=-1).transpose(1, 2)
                    tokens = tokens + context[:, None, :, :]
                inverse_permutation = None
                if self.ts_vpt_enabled:
                    if self.training:
                        permutation = torch.rand(
                            batch, channels, device=tokens.device
                        ).argsort(dim=1)
                        starts = permutation[:, :1]
                        route = torch.cat((starts, permutation), dim=1)
                        self._ts_vpt_batch_transitions = torch.stack(
                            (route[:, :-1], route[:, 1:]), dim=-1
                        ).detach()
                    else:
                        fixed = self._vpt_eval_order(tokens.device)
                        permutation = fixed.unsqueeze(0).expand(batch, -1)
                        self._ts_vpt_batch_transitions = None
                    gather = permutation[:, :, None, None].expand_as(tokens)
                    tokens = tokens.gather(1, gather)
                    inverse_permutation = torch.argsort(permutation, dim=1)
                elif self.ts_acf_order_enabled and self.ts_acf_channel_order.numel() == channels:
                    permutation = self.ts_acf_channel_order.to(device=tokens.device)
                    tokens = tokens.index_select(1, permutation)
                    inverse_permutation = torch.argsort(permutation)
                elif self.ts_variable_permutation_enabled and self.training:
                    # The shared channel-major scan sees a different variable
                    # ordering during training; output order is restored below.
                    permutation = torch.randperm(channels, device=tokens.device)
                    tokens = tokens.index_select(1, permutation)
                    inverse_permutation = torch.argsort(permutation)
                sequence = (
                    tokens.reshape(batch * channels, patch_count, self.d_model)
                    if (self.ts_shared_pretrain_mode or self.ts_channel_agnostic_scan)
                    else tokens.reshape(batch, channels * patch_count, self.d_model)
                )
        else:
            # [B,C,P,D] -> [B,C*P,D] is the v1 channel-major scan order.
            sequence = tokens.reshape(batch, channels * patch_count, self.d_model)
        if self.ts_multiscale_enabled:
            branch_states = []
            for branch in self.ts_multiscale_encoder:
                state = sequence
                for block in branch:
                    state = block(state)
                branch_states.append(state)
            weights = torch.softmax(self.ts_multiscale_logits, dim=0)
            sequence = sum(
                weight.to(dtype=state.dtype) * state
                for weight, state in zip(weights, branch_states)
            )
        else:
            for block in self.encoder:
                sequence = block(sequence)
        sequence = self.final_norm(sequence)
        encoded = sequence.reshape(batch, channels, patch_count, self.d_model)
        if self.ts_shared_postscan_mixer_enabled:
            shared_state = encoded.mean(dim=1)
            mixed_state = self.ts_shared_postscan_mixer(shared_state, shared_state)
            encoded = encoded + (mixed_state - shared_state).unsqueeze(1)
        if self.ts_lite_multiscale_enabled:
            local = encoded.reshape(batch * channels, patch_count, self.d_model).transpose(1, 2)
            short = self.ts_lite_short_filter(local).transpose(1, 2)
            long = self.ts_lite_long_filter(local).transpose(1, 2)
            weights = torch.softmax(self.ts_lite_multiscale_logits, dim=0)
            local = weights[0] * short + weights[1] * long
            encoded = encoded + torch.tanh(self.ts_lite_multiscale_scale) * local.reshape(
                batch, channels, patch_count, self.d_model
            )
        if self.variant in {"v3", "v4", "v5", "v6", "v7", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"} and inverse_permutation is not None:
            if inverse_permutation.ndim == 1:
                encoded = encoded.index_select(1, inverse_permutation)
            else:
                encoded = encoded.gather(
                    1, inverse_permutation[:, :, None, None].expand_as(encoded)
                )
        return encoded

    def _linear_forecast(self, normalized):
        """Return the shared DLinear-style forecast used by the v6 control."""

        if not self.linear_residual_enabled:
            raise RuntimeError("linear forecast is available only in MambaTS v6")
        seasonal, trend = self.linear_decomposition(normalized)
        seasonal = self.linear_seasonal_head(seasonal.transpose(1, 2))
        trend = self.linear_trend_head(trend.transpose(1, 2))
        return (seasonal + trend).transpose(1, 2)

    def _spectral_forecast(self, normalized):
        """Causally extrapolate dominant Fourier bins from the input window."""

        if not self.ts_spectral_enabled:
            raise RuntimeError("spectral forecast is available only in MambaTS v8")
        if normalized.ndim != 3 or normalized.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("MambaTS v8 spectral forecast expects [B, seq_len, enc_in]")
        values = normalized.transpose(1, 2)  # [B,C,L]
        spectrum = torch.fft.rfft(values, dim=-1)
        frequency_count = int(spectrum.shape[-1])
        candidate = spectrum[..., :1].real / float(self.seq_len)
        available = frequency_count - 1
        if available <= 0:
            return candidate.expand(-1, -1, self.pred_len).transpose(1, 2)
        count = min(self.ts_spectral_top_k, available)
        selected = spectrum[..., 1:].abs().topk(count, dim=-1).indices + 1
        coefficients = spectrum.gather(-1, selected)
        steps = torch.arange(
            self.seq_len,
            self.seq_len + self.pred_len,
            device=values.device,
            dtype=values.dtype,
        )
        angular = (
            2.0 * math.pi * selected.to(dtype=values.dtype).unsqueeze(-1)
            * steps.view(1, 1, 1, -1) / float(self.seq_len)
        )
        components = (
            coefficients.real.unsqueeze(-1) * torch.cos(angular)
            - coefficients.imag.unsqueeze(-1) * torch.sin(angular)
        )
        multiplicity = torch.full_like(selected, 2, dtype=values.dtype)
        if self.seq_len % 2 == 0:
            multiplicity = torch.where(
                selected == self.seq_len // 2,
                torch.ones_like(multiplicity), multiplicity,
            )
        candidate = candidate + (
            components * multiplicity.unsqueeze(-1)
        ).sum(dim=-2) / float(self.seq_len)
        return candidate.transpose(1, 2)

    def _representation(self, encoded, normalized=None):
        # Pool only observed-window tokens across variables; no target values
        # or test-time ACF statistics enter this auxiliary representation.
        # Interpolate patch states back to the observed time grid so the
        # legacy AutoCon local objective compares positions in raw timestamp
        # units rather than treating one patch step as one sample.
        if self.ts_repr_forecast_channelwise:
            # The contrastive representation is the channel summary of the
            # exact features used by the channel-preserving forecast route.
            # Thus both losses optimize shared representation parameters.
            temporal = self._channel_representation(encoded).mean(dim=1)
            temporal = self._restore_patch_grid(temporal.transpose(1, 2)).transpose(1, 2)
            return temporal
        temporal = encoded.mean(dim=1).transpose(1, 2)
        temporal = self._restore_patch_grid(temporal).transpose(1, 2)
        acf_context = (
            self._ts_acf_representation_context(normalized)
            if self.ts_acf_repr_fusion_enabled and normalized is not None
            else None
        )
        if acf_context is not None:
            temporal = temporal + acf_context
        temporal = self.repr_projection(temporal)
        temporal = self.repr_norm(temporal)
        return self.repr_head(temporal)

    def _restore_patch_grid(self, temporal):
        """Restore [B, D, patch_count] features to [B, D, seq_len]."""

        matrix = self.representation_interpolation_matrix.to(
            device=temporal.device, dtype=temporal.dtype
        )
        return torch.matmul(temporal, matrix.transpose(0, 1))

    def _representation_forecast(self, representation):
        """Project the AutoCon representation into a shared forecast residual."""

        if not self.ts_repr_forecast_fusion_enabled:
            return None
        if representation is None:
            raise RuntimeError("representation forecast fusion requires AutoCon features")
        if self.ts_repr_forecast_channelwise:
            raise RuntimeError(
                "channelwise representation forecast requires encoded channel states"
            )
        temporal = self.ts_repr_forecast_time(representation.transpose(1, 2))
        return self.ts_repr_forecast_channels(temporal.transpose(1, 2))

    def _channel_representation_forecast(self, encoded):
        """Return a channel-preserving forecast residual from encoded states."""

        if not self.ts_repr_forecast_channelwise:
            return None
        channels = self._channel_representation(encoded)
        channels = self.ts_repr_forecast_time(channels.transpose(2, 3)).transpose(2, 3)
        channels = self.ts_repr_forecast_channels(channels).squeeze(-1)
        return channels.transpose(1, 2)

    def _channel_representation(self, encoded):
        """Create the shared per-channel patch representation [B,C,P,R]."""

        if not self.ts_repr_forecast_channelwise:
            raise RuntimeError("channel representation is disabled")
        channels = encoded.reshape(
            encoded.shape[0] * encoded.shape[1], self.patch_count, self.d_model
        )
        channels = self.ts_repr_channel_projection(channels)
        channels = self.ts_repr_channel_norm(channels)
        channels = self.ts_repr_channel_head(channels)
        return channels.reshape(
            encoded.shape[0], encoded.shape[1], self.patch_count, self.repr_dims
        )

    def _channel_adapter_forecast(self, encoded):
        """Build the lightweight channel-preserving AutoCon Mamba residual."""

        if not self.ts_channel_adapter_enabled:
            return None
        # Pooling happens after the temporal Mamba scan; the adapter does not
        # add a second scan or a per-patch representation tensor.
        summary = encoded.mean(dim=2)
        residual = self.ts_channel_adapter_out(F.gelu(self.ts_channel_adapter_in(summary)))
        return residual.transpose(1, 2)

    def _target_mlp_adapter_forecast(self, encoded):
        """Decode per-channel patch states with no extra temporal scan."""

        if not self.ts_target_mlp_adapter_enabled:
            return None
        batch, channels = encoded.shape[:2]
        states = encoded.reshape(batch * channels, self.patch_count * self.d_model)
        residual = self.ts_target_mlp_adapter_out(
            F.gelu(self.ts_target_mlp_adapter_in(states))
        )
        return residual.reshape(batch, channels, self.pred_len).transpose(1, 2)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        onlyrepr=False,
        return_repr=None,
    ):
        self._reset_ts_acf_runtime_diagnostics()
        normalized, restore = self._normalize(x_enc)
        seasonal, trend = (
            self.ts_decomposition(normalized)
            if self.ts_decomposition_enabled
            else (normalized, None)
        )
        # Preserve the unmodified seasonal window for the persistence route.
        # The optional graph operator only enriches the temporal Mamba input,
        # and its ReZero scale makes this exactly the v3 control at startup.
        graph_seasonal = self._graph_spatial_propagate(seasonal)
        encoded = self._encode(graph_seasonal, x_mark_enc=x_mark_enc)
        if self.ts_global_channel_mixer_enabled:
            shared_state = encoded.mean(dim=1)
            mixed_state = self.ts_global_channel_mixer(seasonal, shared_state)
            encoded = encoded + (mixed_state - shared_state).unsqueeze(1)
        need_repr = bool(
            self.AutoCon
            and (
                onlyrepr
                or self.ts_repr_forecast_fusion_enabled
                or self.ts_acf_memory_selector_enabled
                # The gated linear expert consumes the AutoCon representation
                # internally, even when callers do not request it as output.
                or self.ts_repr_linear_expert_enabled
                or self.ts_decomposition_enabled
                or (self.AutoCon if return_repr is None else bool(return_repr))
            )
        )
        representation = (
            self._representation(encoded, normalized=seasonal)
            if need_repr else None
        )
        mamba_prediction = self.forecast_head(
            encoded.reshape(encoded.shape[0], self.enc_in, -1)
        ).transpose(1, 2)
        persistence = None
        if self.persistence_head is not None:
            persistence = self.persistence_head(seasonal.transpose(1, 2)).transpose(1, 2)
            mamba_prediction = mamba_prediction + self.persistence_scale * persistence
        if self.ts_decomposition_enabled:
            trend_prediction = self.ts_trend_head(trend.transpose(1, 2)).transpose(1, 2)
            prediction = mamba_prediction + trend_prediction
        elif self.linear_residual_enabled:
            linear_prediction = self._linear_forecast(normalized)
            correction_scale = torch.tanh(self.mamba_residual_scale).to(
                device=mamba_prediction.device, dtype=mamba_prediction.dtype
            )
            prediction = linear_prediction + correction_scale * (
                mamba_prediction - linear_prediction
            )
        else:
            prediction = mamba_prediction
        if self.ts_decomposition_enabled and self.AutoCon:
            if representation is None:
                raise RuntimeError("MambaTS v9 decomposition fusion requires AutoCon representation")
            fusion_gate = torch.sigmoid(
                self.ts_decomposition_gate(representation.mean(dim=1))
            ).unsqueeze(-1)
            fusion_scale = torch.tanh(self.ts_decomposition_scale).to(
                device=prediction.device, dtype=prediction.dtype
            )
            # The ReZero correction starts as the un-gated seasonal+trend
            # model.  It then lets the contrastively trained representation
            # select which forecast steps prefer trend over seasonal content.
            prediction = prediction + fusion_scale * (fusion_gate - 0.5) * (
                trend_prediction - mamba_prediction
            )
            self._set_ts_acf_runtime_diagnostic(
                "decomposition_runtime_gate_mean", fusion_gate
            )
        if self.variant in {"v4", "v5", "v7"}:
            memory = self._ts_acf_memory(normalized, apply_scale=False)
            memory_scale = torch.tanh(self.ts_acf_memory_scale).to(
                device=memory.device, dtype=memory.dtype
            )
            base_prediction = prediction
            effective_memory = memory * memory_scale
            if self.ts_acf_memory_selector_enabled:
                if representation is None:
                    raise RuntimeError("ACF memory selector requires AutoCon representation")
                selector_logits = self.ts_acf_memory_selector(
                    representation.mean(dim=1)
                ).unsqueeze(-1)
                if self.ts_acf_memory_channel_selector_enabled:
                    channel_logits = self.ts_acf_memory_channel_selector(
                        encoded.mean(dim=2)
                    ).transpose(1, 2)
                    selector_logits = selector_logits + channel_logits
                selector = torch.sigmoid(selector_logits)
                selector_scale = torch.tanh(self.ts_acf_memory_selector_scale).to(
                    device=memory.device, dtype=memory.dtype
                )
                # Centering makes this a correction to the static memory,
                # rather than silently halving it when the gate is neutral.
                effective_memory = effective_memory + memory * (selector - 0.5) * selector_scale
            self._set_ts_acf_runtime_diagnostic(
                "acf_runtime_gate_mean", memory_scale
            )
            self._set_ts_acf_runtime_diagnostic(
                "acf_effective_memory_rms", effective_memory
            )
            relative_rms = effective_memory.detach().float().square().mean().sqrt() / (
                base_prediction.detach().float().square().mean().sqrt().clamp_min(1e-12)
            )
            self._set_ts_acf_runtime_diagnostic(
                "acf_effective_memory_to_forecast_rms", relative_rms
            )
            prediction = prediction + effective_memory
        if self.variant == "v5":
            prediction = prediction + self._ts_acf_anchor(normalized)
        representation_forecast = (
            self._channel_representation_forecast(encoded)
            if self.ts_repr_forecast_channelwise
            else self._representation_forecast(representation)
        )
        if representation_forecast is not None:
            prediction = prediction + self.ts_repr_forecast_scale * representation_forecast
        channel_adapter_forecast = self._channel_adapter_forecast(encoded)
        if channel_adapter_forecast is not None:
            prediction = prediction + self.ts_channel_adapter_scale * channel_adapter_forecast
        target_mlp_adapter_forecast = self._target_mlp_adapter_forecast(encoded)
        if target_mlp_adapter_forecast is not None:
            prediction = prediction + self.ts_target_mlp_adapter_scale * target_mlp_adapter_forecast
        if self.ts_repr_linear_expert_enabled:
            if representation is None:
                raise RuntimeError("representation-gated linear expert requires AutoCon features")
            if self.ts_repr_linear_expert_mode == "blend":
                # v3 already computes this trainable persistence projection;
                # reuse it so the expert adds only the representation gate.
                linear_prediction = persistence
                if linear_prediction is None:
                    linear_prediction = normalized[:, -1:, :].expand(
                        -1, self.pred_len, -1
                    )
            elif self.ts_repr_linear_expert_mode == "nlinear_blend":
                last = normalized[:, -1:, :]
                centered = normalized - last
                projected = self.ts_nlinear_left(
                    centered.transpose(1, 2)
                )
                projected = self.ts_nlinear_right(projected).transpose(1, 2)
                linear_prediction = projected + last
            else:
                linear_prediction = self.ts_linear_expert(
                    normalized.transpose(1, 2)
                ).transpose(1, 2)
            gate_logits = self.ts_repr_linear_gate(representation.mean(dim=1))
            if self.ts_repr_linear_expert_mode in {"blend", "nlinear_blend"}:
                # This is a convex, non-negative mixture at every horizon:
                # the representation can select the cheap linear expert while
                # retaining the Mamba forecast whenever it is more useful.
                gate = torch.sigmoid(
                    gate_logits + self.ts_repr_linear_blend_bias
                ).unsqueeze(-1)
                prediction = prediction + gate * (linear_prediction - prediction)
            else:
                gate = torch.sigmoid(gate_logits).unsqueeze(-1)
                expert_scale = torch.tanh(self.ts_repr_linear_expert_scale).to(
                    device=prediction.device, dtype=prediction.dtype
                )
                prediction = prediction + expert_scale * gate * (linear_prediction - prediction)
        if self.ts_spectral_enabled:
            spectral_prediction = self._spectral_forecast(normalized)
            if representation is None:
                horizon_gate = prediction.new_ones(
                    prediction.shape[0], self.pred_len, 1
                )
            else:
                horizon_gate = torch.sigmoid(
                    self.ts_spectral_selector(representation.mean(dim=1))
                ).unsqueeze(-1)
                channel_gate = torch.sigmoid(
                    self.ts_spectral_channel_selector(encoded.mean(dim=2))
                ).transpose(1, 2)
                horizon_gate = horizon_gate * channel_gate
            spectral_scale = torch.tanh(self.ts_spectral_scale).to(
                device=prediction.device, dtype=prediction.dtype
            )
            self._set_ts_acf_runtime_diagnostic(
                "spectral_runtime_gate_mean", horizon_gate
            )
            relative_rms = spectral_prediction.detach().float().square().mean().sqrt() / (
                prediction.detach().float().square().mean().sqrt().clamp_min(1e-12)
            )
            self._set_ts_acf_runtime_diagnostic(
                "spectral_candidate_to_forecast_rms", relative_rms
            )
            prediction = prediction + spectral_scale * horizon_gate * spectral_prediction
        prediction = self._restore(prediction, restore, self.c_out)
        if onlyrepr:
            return None, representation
        if self.AutoCon or return_repr:
            return prediction, representation
        return prediction


__all__ = ["Model"]
