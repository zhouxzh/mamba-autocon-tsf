from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import (
    StreamingMetricAccumulator,
    inverse_scaled_arrays,
    metric,
    per_channel_metrics,
    shape_metric,
    write_streaming_artifacts,
)
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import hashlib
import os
import time
import warnings
import numpy as np

from statsmodels.tsa.api import acf
from models.AutoConNet import series_decomp
from layers.losses import (
    ACFRelationLoss,
    AutoCon,
    AutoConCI,
    ForecastACFConsistencyLoss,
    ForecastLagPairConsistencyLoss,
)
from utils.acf_guidance import fit_signed_acf_prior
from utils.acf_relation_contract import resolve_acf_source
from torch.onnx import OperatorExportTypes

warnings.filterwarnings('ignore')


class _ZeroAutoConLoss(nn.Module):
    """Replay-only placeholder for an AutoCon model with no live objective."""

    def forward(self, features, _timeindex):
        batch = int(features.shape[0])
        device = features.device
        dtype = features.dtype
        return (
            torch.zeros(batch, 1, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=dtype),
        )


# Keep batch aggregation, epoch reduction, and persisted loss logs aligned.
# Adding a diagnostic in one place must not silently drop it from the final
# training artifact.
ACF_RUNTIME_DIAGNOSTIC_KEYS = (
    'acf_runtime_gate_mean', 'acf_runtime_gate_std',
    'acf_anchor_copy_enabled', 'acf_anchor_scale',
    'acf_anchor_rms', 'acf_anchor_valid_fraction',
    'acf_anchor_strength_mean', 'acf_anchor_lag_mean',
    'acf_seasonal_rms', 'acf_reliability_mean',
    'acf_effective_memory_rms', 'acf_effective_memory_to_forecast_rms',
    'decomposition_runtime_gate_mean',
    'acf_calibration_amp_mean',
    'acf_calibration_amp_std', 'acf_calibration_amp_min',
    'acf_calibration_amp_max',
    'acf_reliability_interpolation_mean',
    'acf_reliability_interpolation_std',
    'acf_reliability_interpolation_min',
    'acf_reliability_interpolation_max',
    'vpt_transition_coverage', 'vpt_transition_count',
)

# ``signed_hinge_contribution`` already includes ``sign_weight``; the raw
# hinge remains available as ``*_sign_hinge`` for backward-compatible logs.
# Keep the component mapping in one place so batch, epoch, and JSON schemas
# cannot drift apart.
ACF_RELATION_COMPONENT_SPECS = (
    (
        'ACFRelation_global_magnitude_ce',
        'acf_relation_global_magnitude_ce',
        'last_global_magnitude_ce',
    ),
    (
        'ACFRelation_global_signed_hinge_contribution',
        'acf_relation_global_signed_hinge_contribution',
        'last_global_signed_hinge_contribution',
    ),
    (
        'ACFRelation_local_magnitude_ce',
        'acf_relation_local_magnitude_ce',
        'last_local_magnitude_ce',
    ),
    (
        'ACFRelation_local_signed_hinge_contribution',
        'acf_relation_local_signed_hinge_contribution',
        'last_local_signed_hinge_contribution',
    ),
)

# Relation coverage is telemetry only.  Keep its field names and source
# attributes together so the training loop can transfer all detached scalars
# in one operation while retaining the persisted JSON schema.
ACF_RELATION_DIAGNOSTIC_SPECS = (
    ('ACFRelation_global_valid_fraction', 'last_global_valid_fraction'),
    ('ACFRelation_global_active_anchor_fraction', 'last_global_active_anchor_fraction'),
    ('ACFRelation_global_candidate_fraction', 'last_global_candidate_fraction'),
    ('ACFRelation_global_candidate_active_anchor_fraction', 'last_global_candidate_active_anchor_fraction'),
    ('ACFRelation_global_candidate_count_mean', 'last_global_candidate_count_mean'),
    ('ACFRelation_global_contrastive_active_anchor_fraction', 'last_global_contrastive_active_anchor_fraction'),
    ('ACFRelation_global_degenerate_positive_anchor_fraction', 'last_global_degenerate_positive_anchor_fraction'),
    ('ACFRelation_global_positive_count_mean', 'last_global_positive_count_mean'),
    ('ACFRelation_global_high_acf_excluded_fraction', 'last_global_high_acf_excluded_fraction'),
    ('ACFRelation_global_high_acf_candidate_count', 'last_global_high_acf_candidate_count'),
    ('ACFRelation_global_high_acf_excluded_count', 'last_global_high_acf_excluded_count'),
    ('ACFRelation_global_high_acf_conditional_excluded_fraction', 'last_global_high_acf_conditional_excluded_fraction'),
    ('ACFRelation_global_sign_target_count', 'last_global_sign_target_count'),
    ('ACFRelation_global_sign_accuracy', 'last_global_sign_accuracy'),
    ('ACFRelation_global_sign_hinge', 'last_global_sign_hinge'),
    ('ACFRelation_local_target_fraction', 'last_local_target_fraction'),
    ('ACFRelation_local_active_position_fraction', 'last_local_active_position_fraction'),
    ('ACFRelation_local_sign_target_count', 'last_local_sign_target_count'),
    ('ACFRelation_local_sign_accuracy', 'last_local_sign_accuracy'),
    ('ACFRelation_local_sign_hinge', 'last_local_sign_hinge'),
)


def _read_detached_scalar_fields(source, specs):
    """Read detached scalar telemetry with at most one transfer per device."""

    values = {}
    pending = {}
    for output_key, attribute in specs:
        value = getattr(source, attribute)
        if torch.is_tensor(value):
            scalar = value.detach().to(dtype=torch.float32).reshape(())
            pending.setdefault(scalar.device, []).append((output_key, scalar))
        else:
            values[output_key] = float(value)
    for device_values in pending.values():
        keys = [key for key, _ in device_values]
        stacked = torch.stack([value for _, value in device_values])
        host_values = stacked.cpu().tolist()
        values.update(
            (key, float(host_value))
            for key, host_value in zip(keys, host_values)
        )
    return values


def _initialize_relation_component_logs(logs):
    """Register every relation-component series in the persisted log schema."""

    for _, log_key, _ in ACF_RELATION_COMPONENT_SPECS:
        logs[log_key] = []


def _initialize_relation_component_train_log(train_log):
    """Register one epoch's batch-level relation-component accumulators."""

    for train_key, _, _ in ACF_RELATION_COMPONENT_SPECS:
        train_log[train_key] = []


def _record_relation_component_batch(train_log, relation_loss):
    """Append detached ACF relation components from one training batch."""

    specs = [(train_key, attribute) for train_key, _, attribute in ACF_RELATION_COMPONENT_SPECS]
    values = _read_detached_scalar_fields(relation_loss, specs)
    for train_key, value in values.items():
        train_log[train_key].append(value)


def _reduce_relation_component_epoch(train_log):
    """Reduce batch telemetry, retaining explicit ``None`` for inactive loss."""

    for train_key, _, _ in ACF_RELATION_COMPONENT_SPECS:
        values = train_log[train_key]
        train_log[train_key] = float(np.mean(values)) if values else None


def _append_relation_component_epoch(logs, train_log):
    """Persist every reduced relation-component value for one epoch."""

    for train_key, log_key, _ in ACF_RELATION_COMPONENT_SPECS:
        logs[log_key].append(train_log[train_key])


class Exp_Long_Term_Forecast_with_AutoCon(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_with_AutoCon, self).__init__(args)

        self.AutoCon = args.AutoCon
        self.AutoCon_lambda = args.AutoCon_lambda
        self.ForecastACF_loss = None
        self.ForecastACF_weight = float(
            getattr(args, 'mamba_ts_forecast_acf_weight', 0.0)
        )
        self.ForecastLagPair_loss = None
        self.ForecastLagPair_weight = float(
            getattr(args, 'mamba_acf_forecast_pair_weight', 0.0)
        )
        st = time.time()
        self.AutoCon_loss = self.init_AutoCon(args) if self.AutoCon else None
        if self.AutoCon_loss is not None:
            # Relation buffers are train-only metadata, not parameters.  Move
            # them once so every batch avoids copying the complete ACF table
            # through ``tensor.to(device)`` in the lookup helpers.
            self.AutoCon_loss = self.AutoCon_loss.to(self.device)
        # ``init_AutoCon`` also constructs the optional forecast-level ACF
        # objective from the same train-only series.  It is stored separately
        # from the historical representation objective, so move it only after
        # that initializer has completed.
        if self.ForecastACF_loss is not None:
            self.ForecastACF_loss = self.ForecastACF_loss.to(self.device)
        if self.ForecastLagPair_loss is not None:
            self.ForecastLagPair_loss = self.ForecastLagPair_loss.to(self.device)
        ed = time.time()
        print(f'Autocorrelation calculation time: {ed-st:.4f}')

    def _init_forecast_lag_pair_consistency(
        self, args, acf_values, selected_lags
    ):
        """Attach the forecast-pair objective from the exact train-only prior."""

        if not bool(getattr(args, 'mamba_acf_forecast_pair_consistency', False)):
            return
        if not selected_lags:
            raise ValueError(
                'forecast lag-pair consistency requires resolved train ACF lags'
            )
        self.ForecastLagPair_loss = ForecastLagPairConsistencyLoss(
            acf_values,
            selected_lags=selected_lags,
            lag_tolerance=int(getattr(args, 'mamba_acf_relation_lag_tolerance', 0)),
        )

    def _load_mamba_sidecar_relation_loss(self, args, model):
        """Load a complete train-only Mamba ACF prior before AutoCon setup.

        AutoCon initialisation runs before the Mamba experiment mixin's
        post-construction hook.  Handling the sidecar here is therefore
        necessary: otherwise evaluation would refit from the train CSV (or
        fail when it is unavailable) and construct the relation objective from
        a different prior than the forecast branch.
        """

        configured_path = getattr(args, "mamba_acf_prior_artifact_path", None)
        if configured_path is None or not str(configured_path).strip():
            return None
        artifact_path = os.path.abspath(str(configured_path))
        if not os.path.isfile(artifact_path):
            raise RuntimeError(
                "configured MambaACF prior artifact is missing: {}".format(
                    artifact_path
                )
            )
        expected_sha_value = getattr(args, "mamba_acf_prior_expected_sha256", "")
        expected_sha = (
            "" if expected_sha_value is None
            else str(expected_sha_value).strip().lower()
        )
        if expected_sha:
            digest = hashlib.sha256()
            with open(artifact_path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_sha:
                raise RuntimeError(
                    "MambaACF prior artifact hash mismatch: {}".format(
                        artifact_path
                    )
                )
        model.load_acf_prior(artifact_path)
        # The Mamba subclass invokes its normal post-construction attachment
        # hook after this base initializer returns.  Mark the prior as already
        # attached so that hook does not read and validate the same sidecar a
        # second time.
        self._acf_prior_attached = True
        state = model.export_acf_prior_state()
        full_values = None if state is None else state.get("acf_full_values")
        if full_values is None:
            raise RuntimeError(
                "MambaACF AutoCon evaluation requires acf_prior.npz with the "
                "complete train-only ACF table; the compact-only artifact "
                "cannot reproduce the relation objective"
            )
        full_values = np.asarray(full_values, dtype=np.float32)
        model_channels = int(getattr(model, "channels", getattr(model, "enc_in", 0)))
        if full_values.ndim != 2 or full_values.shape[0] != model_channels:
            raise RuntimeError(
                "MambaACF sidecar complete ACF table has an invalid shape"
            )

        selected_lags = tuple(
            int(lag) for lag in getattr(
                model, "acf_lags", getattr(model, "acf_prior_lags", ())
            )
        )
        args.mamba_acf_effective_lags = list(selected_lags)
        args.mamba_acf_effective_lag_source = "persisted_sidecar"
        args.mamba_acf_effective_lag_fallback = False
        args.mamba_acf_prior_max_lag = int(full_values.shape[1] - 1)
        relation_min_lag = int(getattr(args, "mamba_acf_relation_min_lag", 0))
        if relation_min_lag > args.mamba_acf_prior_max_lag:
            raise ValueError(
                "mamba_acf_relation_min_lag exceeds the measured ACF range"
            )
        relation_lags = tuple(
            lag for lag in selected_lags if int(lag) >= relation_min_lag
        )
        relation_lag_fallback = False
        if not relation_lags and relation_min_lag > 0:
            if relation_min_lag <= args.mamba_acf_prior_max_lag:
                relation_lags = (relation_min_lag,)
                relation_lag_fallback = True
            elif bool(getattr(args, "mamba_acf_relation_lag_mask", False)):
                raise ValueError(
                    "mamba_acf_relation_min_lag exceeds the measured ACF range"
                )
        args.mamba_acf_relation_effective_lags = list(relation_lags)
        args.mamba_acf_relation_lag_fallback = bool(relation_lag_fallback)
        args.mamba_acf_relation_lag_source = (
            "min_lag_fallback" if relation_lag_fallback else "persisted_sidecar"
        )
        relation_target_mode = str(getattr(
            args, "mamba_acf_relation_target_mode", "hard_selected"
        )).lower()
        selected_for_loss = (
            relation_lags
            if bool(getattr(args, "mamba_acf_relation_lag_mask", False))
            or relation_target_mode in {"soft_multi_peak", "soft_full"}
            else None
        )
        args.mamba_acf_relation_selected_lags = list(
            relation_lags
            if bool(getattr(args, "mamba_acf_relation_lag_mask", False))
            else selected_lags
        )
        self._init_forecast_lag_pair_consistency(
            args, full_values, relation_lags
        )
        if not bool(getattr(args, "mamba_acf_relation_loss", False)):
            raise ValueError(
                "MambaACF AutoCon requires --mamba_acf_relation_loss; "
                "the legacy AutoCon objective is not valid for a compact ACF prior"
            )
        print("MambaACF complete train-only prior loaded from sidecar: {}".format(
            artifact_path
        ))
        return ACFRelationLoss(
            full_values,
            temperature=float(getattr(args, "AutoCon_temperature", 0.2)),
            relation_temperature=float(getattr(
                args, "AutoCon_relation_temperature", 0.25
            )),
            sign_weight=float(getattr(args, "AutoCon_sign_weight", 0.25)),
            sign_margin=float(getattr(args, "AutoCon_sign_margin", 0.1)),
            channel_reduce=str(getattr(
                args, "mamba_acf_relation_channel_reduce", "mean"
            )),
            relation_scope=str(getattr(args, "mamba_acf_relation_scope", "both")),
            selected_lags=selected_for_loss,
            lag_tolerance=int(getattr(args, "mamba_acf_relation_lag_tolerance", 0)),
            target_mode=relation_target_mode,
            soft_min_abs=float(getattr(
                args, "mamba_acf_relation_soft_min_abs", 0.1
            )),
            diagnostic_min_abs=float(getattr(
                args, "mamba_acf_relation_diagnostic_min_abs", 0.1
            )),
            selected_lag_boost=float(getattr(
                args, "mamba_acf_relation_selected_lag_boost", 0.0
            )),
            min_lag=relation_min_lag,
            normalize_global_active_anchors=bool(getattr(
                args, "mamba_acf_relation_active_anchor_normalization", False
            )),
        )

    def init_AutoCon(self, args):
        model_for_prior = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        configured_sidecar = getattr(args, "mamba_acf_prior_artifact_path", None)
        if (
            hasattr(model_for_prior, "load_acf_prior")
            and configured_sidecar is not None
            and str(configured_sidecar).strip()
            and (
                str(getattr(args, "model", "")).startswith("MambaACF")
                or bool(getattr(args, "mamba_acf_relation_loss", False))
            )
        ):
            return self._load_mamba_sidecar_relation_loss(args, model_for_prior)
        # MambaTS v4/v5 use a compact prior for forecast memory and optional
        # representation fusion, but do not expose MambaACF's complete
        # relation table. During checkpoint replay with the auxiliary objective
        # disabled, load the compact sidecar and avoid refitting a historical
        # AutoCon loss from the training CSV.
        if (
            hasattr(model_for_prior, "load_acf_prior")
            and configured_sidecar is not None
            and str(configured_sidecar).strip()
            and str(getattr(args, "model", "")).startswith(("MambaTSv4", "MambaTSv5"))
            and not bool(getattr(args, "mamba_acf_relation_loss", False))
        ):
            model_for_prior.load_acf_prior(str(configured_sidecar))
            self._acf_prior_attached = True
            return _ZeroAutoConLoss()
        target_data, _ = self._get_data(flag='train')
        target_data = target_data.data_x.copy()
        # Estimate the lag prior from the training split only.  The residual
        # is the default because the moving-average trend otherwise creates a
        # near-unit-root ACF and makes almost every lag look positive.
        kernel_size = max(3, int(args.seq_len) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoother = series_decomp(kernel_size)
        x = torch.from_numpy(target_data).unsqueeze(0)
        residual, trend = smoother(x)
        source = resolve_acf_source(args)
        if source not in {'residual', 'trend', 'raw'}:
            raise ValueError("AutoCon_acf_source must be one of: residual, trend, raw")
        if source == 'residual':
            target_data = residual
        elif source == 'trend':
            target_data = trend
        else:
            target_data = x
        target_data = target_data.squeeze(0).numpy()

        # MambaACF has a dedicated compact prior path.  Fit it once and share
        # the exact signed, train-only table with both the causal forecast
        # memory and the optional relation objective.  Historical AutoCon
        # models stay on the statsmodels path below for reproducibility.
        if hasattr(model_for_prior, 'set_acf_prior'):
            configured_lags = getattr(args, 'mamba_acf_lags', [24, 168, 336])
            if isinstance(configured_lags, str):
                configured_lags = [int(item) for item in configured_lags.replace(',', ' ').split() if item]
            configured_lags = tuple(sorted({int(item) for item in configured_lags if int(item) > 0}))
            max_configured = max(configured_lags, default=1)
            max_lag = min(
                len(target_data) - 1,
                max(max_configured, int(getattr(args, 'mamba_acf_max_lag', 720)), int(args.seq_len) - 1),
            )
            compact_prior = fit_signed_acf_prior(
                target_data,
                max_lag=max(1, max_lag),
                top_k=max(1, int(getattr(args, 'mamba_acf_top_k', 8)), len(configured_lags)),
                min_lag=min(max(1, int(getattr(args, 'mamba_acf_min_lag', 2))), max(1, max_lag)),
                min_abs_acf=float(getattr(args, 'mamba_acf_min_abs', 0.1)),
                confidence_z=float(getattr(args, 'mamba_acf_confidence_z', 1.96)),
                min_separation=max(1, int(getattr(args, 'mamba_acf_min_separation', 2))),
                detrend=str(getattr(args, 'mamba_acf_detrend', 'mean')),
                channel_chunk_size=int(getattr(args, 'mamba_acf_chunk', 64)),
            )
            # Keep the complete train-only peak union in the run metadata.  A
            # later validator must be able to distinguish a genuine peak
            # selection from a configured-lag fallback without recomputing the
            # prior (or accidentally touching validation/test values).
            train_peak_lags = tuple(
                sorted({
                    int(lag)
                    for lag in compact_prior.common_lags(top_k=None)
                    if 0 < int(lag) <= compact_prior.max_lag
                })
            )
            args.mamba_acf_train_peak_lags = list(train_peak_lags)
            # Persist the relation sampler's train-only candidate universe
            # under its explicit field as well as the shared memory field.
            # This keeps config.json and loss_logs.json independently
            # sufficient for audit/replay.
            args.mamba_acf_relation_train_peak_lags = list(train_peak_lags)
            lag_mode = str(getattr(args, 'mamba_acf_lag_mode', 'fixed')).lower()
            selected_lag_fallback = False
            if lag_mode == 'train_peaks':
                selected_lags = compact_prior.common_lags(top_k=max(1, len(configured_lags)))
                if not selected_lags:
                    selected_lags = configured_lags
                    selected_lag_fallback = True
            elif lag_mode == 'train_peaks_available':
                selected_lags = compact_prior.availability_weighted_lags(
                    history_length=int(args.seq_len),
                    horizon=int(args.pred_len),
                    top_k=max(1, len(configured_lags)),
                    min_separation=max(1, int(getattr(args, 'mamba_acf_min_separation', 2))),
                )
                if not selected_lags:
                    selected_lags = configured_lags
                    selected_lag_fallback = True
            elif lag_mode == 'fixed':
                selected_lags = configured_lags
            else:
                raise ValueError(
                    "mamba_acf_lag_mode must be 'fixed', 'train_peaks', or "
                    "'train_peaks_available'"
                )
            selected_lags = tuple(sorted({
                int(lag) for lag in selected_lags
                if 0 < int(lag) <= compact_prior.max_lag
            }))
            if not selected_lags:
                raise ValueError(
                    'no selected MambaACF lag is available in the train split; '
                    'reduce --mamba_acf_lags or use a train-peak lag mode'
                )
            if bool(getattr(args, 'mamba_variable_acf_memory_selector', False)):
                accessible_lags = tuple(
                    lag for lag in selected_lags if int(lag) <= int(args.seq_len)
                )
                if not accessible_lags:
                    raise ValueError(
                        'mamba_variable_acf_memory_selector requires at least '
                        'one selected train-only ACF lag <= seq_len; the '
                        'resolved lag set is inaccessible to the observed window'
                    )
            # Persist the resolved train-only choice on args so the relation
            # sampler, run metadata, and model all use the same lag set.
            args.mamba_acf_effective_lags = list(selected_lags)
            args.mamba_acf_effective_lag_source = (
                'configured_fallback' if selected_lag_fallback
                else 'fixed_configured' if lag_mode == 'fixed'
                else lag_mode
            )
            args.mamba_acf_effective_lag_fallback = bool(selected_lag_fallback)
            args.mamba_acf_prior_max_lag = int(compact_prior.max_lag)
            model_for_prior.set_acf_prior(compact_prior.acf, lags=selected_lags)
            self._acf_prior_attached = True
            relation_min_lag = int(getattr(args, 'mamba_acf_relation_min_lag', 0))
            if relation_min_lag > args.mamba_acf_prior_max_lag:
                raise ValueError(
                    'mamba_acf_relation_min_lag exceeds the measured ACF range; '
                    'lower the threshold or use a prior with a longer measured range'
                )
            # Pair batching uses available peaks as anchors.  The loss applies
            # the lower bound to every candidate, including unselected peaks.
            relation_lags = tuple(
                sorted(lag for lag in selected_lags if int(lag) >= relation_min_lag)
            )
            relation_lag_fallback = False
            if not relation_lags and relation_min_lag > 0:
                # A no-overlap control can still form useful batches even when
                # none of the forecast-memory peaks survives the bound.
                if relation_min_lag <= compact_prior.max_lag:
                    relation_lags = (relation_min_lag,)
                    relation_lag_fallback = True
                elif bool(getattr(args, 'mamba_acf_relation_lag_mask', False)):
                    raise ValueError(
                        'mamba_acf_relation_min_lag exceeds the measured ACF range; '
                        'lower the threshold or disable selected-lag masking'
                    )
            # Keep forecast-memory lags and relation lags separate: excluding
            # overlapping windows from the auxiliary objective must not remove
            # a short seasonal lag from the prediction path.
            args.mamba_acf_relation_effective_lags = list(relation_lags)
            if relation_lag_fallback:
                relation_lag_source = 'min_lag_fallback'
            elif (
                lag_mode in {'train_peaks', 'train_peaks_available'}
                and relation_lags
                and set(relation_lags).issubset(set(train_peak_lags))
                and not selected_lag_fallback
            ):
                relation_lag_source = 'train_peak_subset'
            elif relation_lags:
                relation_lag_source = 'configured_or_fixed'
            else:
                relation_lag_source = 'empty'
            args.mamba_acf_relation_lag_source = relation_lag_source
            args.mamba_acf_relation_lag_fallback = bool(relation_lag_fallback)
            args.mamba_acf_relation_selected_lags = list(
                relation_lags if bool(getattr(args, 'mamba_acf_relation_lag_mask', False))
                else selected_lags
            )
            print('MambaACF compact train-only prior: lags={}, max_lag={}, channels={}'.format(
                list(selected_lags), compact_prior.max_lag, compact_prior.n_channels
            ))
            if relation_lags != selected_lags:
                print('MambaACF relation lags (overlap control): {}'.format(list(relation_lags)))
            self._init_forecast_lag_pair_consistency(
                args, compact_prior.acf, relation_lags
            )
            if bool(getattr(args, 'mamba_acf_relation_loss', False)):
                relation_target_mode = str(getattr(
                    args, 'mamba_acf_relation_target_mode', 'hard_selected'
                )).lower()
                selected_for_loss = (
                    relation_lags
                    if bool(getattr(args, 'mamba_acf_relation_lag_mask', False))
                    or relation_target_mode in {'soft_multi_peak', 'soft_full'}
                    else None
                )
                return ACFRelationLoss(
                    compact_prior.acf,
                    temperature=float(getattr(args, 'AutoCon_temperature', 0.2)),
                    relation_temperature=float(getattr(args, 'AutoCon_relation_temperature', 0.25)),
                    sign_weight=float(getattr(args, 'AutoCon_sign_weight', 0.25)),
                    sign_margin=float(getattr(args, 'AutoCon_sign_margin', 0.1)),
                    channel_reduce=str(getattr(args, 'mamba_acf_relation_channel_reduce', 'mean')),
                    relation_scope=str(getattr(args, 'mamba_acf_relation_scope', 'both')),
                    selected_lags=selected_for_loss,
                    lag_tolerance=int(getattr(args, 'mamba_acf_relation_lag_tolerance', 0)),
                    target_mode=relation_target_mode,
                    soft_min_abs=float(getattr(
                        args, 'mamba_acf_relation_soft_min_abs', 0.1
                    )),
                    diagnostic_min_abs=float(getattr(
                        args, 'mamba_acf_relation_diagnostic_min_abs', 0.1
                    )),
                    selected_lag_boost=float(getattr(
                        args, 'mamba_acf_relation_selected_lag_boost', 0.0
                    )),
                    min_lag=relation_min_lag,
                    normalize_global_active_anchors=bool(getattr(
                        args, 'mamba_acf_relation_active_anchor_normalization', False
                    )),
                )
            # AutoCon assumes a complete ACF table and clamps out-of-range
            # window distances to its final entry.  MambaACF deliberately
            # stores a compact train-only table, so mixing the two would
            # silently supervise unrelated pairs with the same relation.
            raise ValueError(
                'MambaACF AutoCon requires --mamba_acf_relation_loss; '
                'the legacy AutoCon objective is not valid for a compact ACF prior'
            )

        acf_values = []
        signed_acf_values = []
        for i_ch in range(target_data.shape[-1]):
            values = acf(target_data[..., i_ch], nlags=max(1, len(target_data) - 1), fft=True)
            signed_acf_values.append(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0))
            acf_values.append(np.nan_to_num(np.abs(values), nan=0.0, posinf=1.0, neginf=0.0))

        if bool(getattr(args, 'mamba_ts_forecast_acf_consistency', False)):
            self.ForecastACF_loss = ForecastACFConsistencyLoss(
                np.stack(signed_acf_values, axis=0),
                horizon=int(args.pred_len),
                top_k=int(getattr(args, 'mamba_ts_forecast_acf_top_k', 4)),
                min_lag=int(getattr(args, 'mamba_ts_forecast_acf_min_lag', 2)),
                min_abs=float(getattr(args, 'mamba_ts_forecast_acf_min_abs', 0.10)),
            )

        # MambaACF consumes the signed train-split prior for its causal
        # seasonal-memory branch.  Existing AutoCon models retain their
        # historical absolute-ACF objective below for reproducibility.
        if self.args.model == 'AutoConCI':  # For multivariate forecasting
            acf_values = np.stack(acf_values, axis=0)
            loss = AutoConCI(args.batch_size, args.seq_len, acf_values, temperature=1.0, base_temperature=1.0)
            print(f'Auto-correlation values(abs):{acf_values[0, :2]} ~ {acf_values[0, -2:]}')
        else:
            acf_values = np.stack(acf_values, axis=0).mean(axis=0)
            loss = AutoCon(args.batch_size, args.seq_len, acf_values, temperature=1.0, base_temperature=1.0)
            print(f'Auto-correlation values(abs):{acf_values[:2]} ~ {acf_values[-2:]}')
        return loss

    def _forecast_output(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """Unwrap models with and without an auxiliary representation."""
        model_for_contract = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        if getattr(model_for_contract, "supports_return_repr_control", False):
            # AutoCon is needed for the training objective, but validation and
            # test only consume the forecast.  Keeping the default ``None``
            # behavior outside this branch preserves legacy model contracts.
            return_repr = bool(
                getattr(model_for_contract, "AutoCon", False)
                and model_for_contract.training
            )
            output = self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                return_repr=return_repr,
            )
        else:
            output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if isinstance(output, (tuple, list)):
            return output[0], output[1] if len(output) > 1 else None
        return output, None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(f'model parameters:{self.count_parameters(model)}')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        weight_decay = getattr(self.args, 'weight_decay', None)
        if weight_decay is None:
            weight_decay = 1e-5
        model_optim = optim.Adam(
            self.optimizer_parameter_groups(self.model, weight_decay),
            lr=self.args.learning_rate,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        # Aggregate squared error over all validation elements. Averaging batch
        # means would give a smaller final batch a different weight.
        total_squared_error = 0.0
        total_elements = 0
        self.model.eval()
        with torch.no_grad():
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                outputs, _ = self._forecast_output(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                residual = outputs - batch_y
                total_squared_error += float(torch.sum(residual * residual).item())
                total_elements += int(residual.numel())
        total_loss = (total_squared_error / total_elements
                      if total_elements else float('inf'))
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # Keep the validation trajectory beside the checkpoint.  This is used
        # for validation-only hyperparameter selection; test metrics must not
        # influence the choice of lambda or the stopping epoch.
        logs = {
            # Bump this when the relation objective's candidate/positive
            # semantics change; old screening logs must not be pooled with the
            # repaired objective.
            'acf_relation_objective_version': (
                int(self.AutoCon_loss.objective_version)
                if isinstance(self.AutoCon_loss, ACFRelationLoss) else None
            ),
            'acf_relation_target_mode': (
                self.AutoCon_loss.target_mode
                if isinstance(self.AutoCon_loss, ACFRelationLoss) else None
            ),
            'acf_relation_soft_min_abs': (
                float(self.AutoCon_loss.soft_min_abs)
                if isinstance(self.AutoCon_loss, ACFRelationLoss) else None
            ),
            'acf_relation_diagnostic_min_abs': (
                float(self.AutoCon_loss.diagnostic_min_abs)
                if isinstance(self.AutoCon_loss, ACFRelationLoss) else None
            ),
            # Resolved train-only lag metadata is duplicated in the loss log so
            # validators do not have to infer relation eligibility from a
            # mutable config or recompute the ACF table.
            'acf_relation_effective_lags': list(getattr(
                self.args, 'mamba_acf_relation_effective_lags', []
            )),
            'acf_relation_selected_lags': list(getattr(
                self.args, 'mamba_acf_relation_selected_lags', []
            )),
            'acf_relation_train_peak_lags': list(getattr(
                self.args, 'mamba_acf_train_peak_lags', []
            )),
            'acf_relation_lag_source': getattr(
                self.args, 'mamba_acf_relation_lag_source', None
            ),
            'acf_relation_lag_fallback': bool(getattr(
                self.args, 'mamba_acf_relation_lag_fallback', False
            )),
            'acf_relation_lag_balance': getattr(
                self.args, 'mamba_acf_relation_lag_balance', 'random'
            ),
            # The requested batch policy is not enough to recover the live
            # sampling distribution: all-candidate targets intentionally fall
            # back to ordinary shuffle even if ``lag_pairs`` was requested.
            # Persist the resolution produced by ``data_factory`` so a later
            # reader can audit the loss/sampler semantic match without replay.
            'acf_relation_batching_requested': getattr(
                self.args, 'mamba_acf_relation_batching_requested',
                getattr(self.args, 'mamba_acf_relation_batching', 'lag_pairs')
            ),
            'acf_relation_batching_effective': getattr(
                self.args, 'mamba_acf_relation_batching_effective', 'shuffle'
            ),
            'acf_relation_batching_reason': getattr(
                self.args, 'mamba_acf_relation_batching_reason',
                'not_resolved'
            ),
            'acf_relation_selected_lag_support': bool(getattr(
                self.args, 'mamba_acf_relation_selected_lag_support', False
            )),
            # Filled once per training epoch when the relation-aware sampler
            # is active.  Counts are kept as strings in JSON so arbitrary
            # integer lag values remain unambiguous object keys.
            'acf_relation_lag_pair_counts': [],
            'acf_relation_lag_pair_exposure': [],
            'train_loss': [],
            'val_loss': [],
            'mse_loss': [],
            'autocon_loss': [],
            'forecast_acf_loss': [],
            'forecast_lag_pair_loss': [],
            'forecast_lag_pair_fraction': [],
            'forecast_lag_pair_active_anchor_fraction': [],
            'forecast_lag_pair_count': [],
            'forecast_lag_pair_target_abs_mean': [],
            'effective_lambda': [],
            'acf_relation_global_valid_fraction': [],
            'acf_relation_global_active_anchor_fraction': [],
            'acf_relation_global_candidate_fraction': [],
            'acf_relation_global_candidate_active_anchor_fraction': [],
            'acf_relation_global_candidate_count_mean': [],
            'acf_relation_global_contrastive_active_anchor_fraction': [],
            'acf_relation_global_degenerate_positive_anchor_fraction': [],
            'acf_relation_global_positive_count_mean': [],
            'acf_relation_global_high_acf_excluded_fraction': [],
            'acf_relation_global_high_acf_candidate_count': [],
            'acf_relation_global_high_acf_excluded_count': [],
            'acf_relation_global_high_acf_conditional_excluded_fraction': [],
            'acf_relation_global_sign_target_count': [],
            'acf_relation_global_sign_accuracy': [],
            'acf_relation_global_sign_hinge': [],
            'acf_relation_local_target_fraction': [],
            'acf_relation_local_active_position_fraction': [],
            'acf_relation_local_sign_target_count': [],
            'acf_relation_local_sign_accuracy': [],
            'acf_relation_local_sign_hinge': [],
            'acf_runtime_gate_mean': [],
            'acf_runtime_gate_std': [],
            'acf_anchor_copy_enabled': [],
            'acf_anchor_scale': [],
            'acf_anchor_rms': [],
            'acf_anchor_valid_fraction': [],
            'acf_anchor_strength_mean': [],
            'acf_anchor_lag_mean': [],
            'acf_seasonal_rms': [],
            'acf_reliability_mean': [],
            'acf_effective_memory_rms': [],
            'acf_effective_memory_to_forecast_rms': [],
            'decomposition_runtime_gate_mean': [],
            'acf_calibration_amp_mean': [],
            'acf_calibration_amp_std': [],
            'acf_calibration_amp_min': [],
            'acf_calibration_amp_max': [],
            'acf_reliability_interpolation_mean': [],
            'acf_reliability_interpolation_std': [],
            'acf_reliability_interpolation_min': [],
            'acf_reliability_interpolation_max': [],
            'epoch_time_sec': [],
        }
        for diagnostic_key in ACF_RUNTIME_DIAGNOSTIC_KEYS:
            logs.setdefault(diagnostic_key, [])
        _initialize_relation_component_logs(logs)

        for epoch in range(self.args.train_epochs):
            batch_sampler = getattr(train_loader, 'batch_sampler', None)
            if hasattr(batch_sampler, 'set_epoch'):
                batch_sampler.set_epoch(epoch)
            iter_count = 0
            train_log = dict()
            train_log['loss'] = []
            train_log['MSE_loss'] = []
            train_log['AutoCon_loss'] = []
            train_log['ForecastACF_loss'] = []
            train_log['ForecastLagPair_loss'] = []
            train_log['ForecastLagPair_fraction'] = []
            train_log['ForecastLagPair_active_anchor_fraction'] = []
            train_log['ForecastLagPair_count'] = []
            train_log['ForecastLagPair_target_abs_mean'] = []
            train_log['ACFRelation_global_valid_fraction'] = []
            train_log['ACFRelation_global_active_anchor_fraction'] = []
            train_log['ACFRelation_global_candidate_fraction'] = []
            train_log['ACFRelation_global_candidate_active_anchor_fraction'] = []
            train_log['ACFRelation_global_candidate_count_mean'] = []
            train_log['ACFRelation_global_contrastive_active_anchor_fraction'] = []
            train_log['ACFRelation_global_degenerate_positive_anchor_fraction'] = []
            train_log['ACFRelation_global_positive_count_mean'] = []
            train_log['ACFRelation_global_high_acf_excluded_fraction'] = []
            train_log['ACFRelation_global_high_acf_candidate_count'] = []
            train_log['ACFRelation_global_high_acf_excluded_count'] = []
            train_log['ACFRelation_global_high_acf_conditional_excluded_fraction'] = []
            train_log['ACFRelation_global_sign_target_count'] = []
            train_log['ACFRelation_global_sign_accuracy'] = []
            train_log['ACFRelation_global_sign_hinge'] = []
            train_log['ACFRelation_local_target_fraction'] = []
            train_log['ACFRelation_local_active_position_fraction'] = []
            train_log['ACFRelation_local_sign_target_count'] = []
            train_log['ACFRelation_local_sign_accuracy'] = []
            train_log['ACFRelation_local_sign_hinge'] = []
            _initialize_relation_component_train_log(train_log)
            for diagnostic_key in ACF_RUNTIME_DIAGNOSTIC_KEYS:
                train_log[diagnostic_key] = []

            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                B, T, C = batch_x.shape

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs, repr = self._forecast_output(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                diagnostic_model = (
                    self.model.module
                    if isinstance(self.model, nn.DataParallel)
                    else self.model
                )
                diagnostic_reader = getattr(
                    diagnostic_model, 'acf_runtime_diagnostics', None
                )
                if callable(diagnostic_reader):
                    for diagnostic_key, diagnostic_value in diagnostic_reader().items():
                        if diagnostic_value is not None and np.isfinite(float(diagnostic_value)):
                            train_log.setdefault(diagnostic_key, []).append(
                                float(diagnostic_value)
                            )

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                MSE_loss = F.mse_loss(outputs, batch_y, reduction='none')
                vpt_updater = getattr(diagnostic_model, 'update_vpt_transition_cost', None)
                if callable(vpt_updater):
                    # VPT sees only the detached pure prediction loss, never
                    # AutoCon, validation, or test targets.
                    vpt_updater(MSE_loss)

                if self.args.AutoCon:
                    if repr is None:
                        raise RuntimeError('AutoCon training requires the model to return a representation')
                    features = F.normalize(repr, dim=-1)  # B, T, C
                    global_pos_labels = timeindex.long().to(self.device)
                    local_loss, global_loss = self.AutoCon_loss(features, global_pos_labels)

                    if self.args.model == 'AutoConCI':
                        local_mean = local_loss.reshape(B, C, -1).mean(dim=2).mean(dim=1)
                        autocon_loss = (local_mean + global_loss.mean(dim=0)) / 2.0
                    elif isinstance(self.AutoCon_loss, ACFRelationLoss):
                        # ACFRelationLoss exposes explicit scopes.  Average only
                        # active components so global-only/local-only ablations
                        # keep the same effective AutoCon lambda as ``both``.
                        components = []
                        scope = self.AutoCon_loss.relation_scope
                        if scope in {'both', 'local'}:
                            components.append(local_loss.mean(dim=1))
                        if scope in {'both', 'global'}:
                            components.append(global_loss)
                        autocon_loss = torch.stack(components, dim=0).mean(dim=0)
                    else:
                        autocon_loss = (local_loss.mean(dim=1) + global_loss) / 2.0
                    if isinstance(self.AutoCon_loss, ACFRelationLoss):
                        relation_diagnostics = _read_detached_scalar_fields(
                            self.AutoCon_loss,
                            ACF_RELATION_DIAGNOSTIC_SPECS,
                        )
                        for diagnostic_key, diagnostic_value in relation_diagnostics.items():
                            train_log[diagnostic_key].append(diagnostic_value)
                        _record_relation_component_batch(
                            train_log, self.AutoCon_loss
                        )
                else:
                    autocon_loss = torch.zeros(B, device=self.device)

                if self.ForecastACF_loss is not None:
                    forecast_acf_loss = self.ForecastACF_loss(outputs)
                else:
                    forecast_acf_loss = torch.zeros(B, device=self.device)
                if self.ForecastLagPair_loss is not None:
                    forecast_lag_pair_loss = self.ForecastLagPair_loss(
                        outputs, timeindex.long().to(self.device)
                    )
                    train_log['ForecastLagPair_fraction'].append(
                        self.ForecastLagPair_loss.last_pair_fraction
                    )
                    train_log['ForecastLagPair_active_anchor_fraction'].append(
                        self.ForecastLagPair_loss.last_active_anchor_fraction
                    )
                    train_log['ForecastLagPair_count'].append(
                        self.ForecastLagPair_loss.last_pair_count
                    )
                    train_log['ForecastLagPair_target_abs_mean'].append(
                        self.ForecastLagPair_loss.last_target_abs_mean
                    )
                else:
                    forecast_lag_pair_loss = torch.zeros(B, device=self.device)

                warmup_epochs = int(getattr(self.args, 'AutoCon_warmup_epochs', 0))
                if warmup_epochs > 0:
                    lambda_scale = min(1.0, float(epoch + 1) / warmup_epochs)
                else:
                    lambda_scale = 1.0
                effective_lambda = self.AutoCon_lambda * lambda_scale if self.args.AutoCon else 0.0
                loss = (
                    MSE_loss.mean()
                    + effective_lambda * autocon_loss.mean()
                    + self.ForecastACF_weight * forecast_acf_loss.mean()
                    + self.ForecastLagPair_weight * forecast_lag_pair_loss.mean()
                )

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                train_log['loss'].append(loss.item())
                train_log['AutoCon_loss'].append(float(autocon_loss.detach().mean().cpu()))
                train_log['ForecastACF_loss'].append(float(forecast_acf_loss.detach().mean().cpu()))
                train_log['ForecastLagPair_loss'].append(
                    float(forecast_lag_pair_loss.detach().mean().cpu())
                )
                train_log['MSE_loss'].append(float(MSE_loss.detach().mean().cpu()))

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_log['loss'] = np.average(train_log['loss'])
            train_log['AutoCon_loss'] = float(np.mean(train_log['AutoCon_loss']))
            train_log['ForecastACF_loss'] = float(np.mean(train_log['ForecastACF_loss']))
            train_log['ForecastLagPair_loss'] = float(
                np.mean(train_log['ForecastLagPair_loss'])
            )
            for telemetry_key in (
                'ForecastLagPair_fraction',
                'ForecastLagPair_active_anchor_fraction',
                'ForecastLagPair_count',
                'ForecastLagPair_target_abs_mean',
            ):
                values = train_log[telemetry_key]
                train_log[telemetry_key] = float(np.mean(values)) if values else None
            train_log['MSE_loss'] = float(np.mean(train_log['MSE_loss']))
            if train_log['ACFRelation_global_valid_fraction']:
                train_log['ACFRelation_global_valid_fraction'] = float(np.mean(
                    train_log['ACFRelation_global_valid_fraction']
                ))
                train_log['ACFRelation_global_active_anchor_fraction'] = float(np.mean(
                    train_log['ACFRelation_global_active_anchor_fraction']
                ))
                train_log['ACFRelation_global_candidate_fraction'] = float(np.mean(
                    train_log['ACFRelation_global_candidate_fraction']
                ))
                train_log['ACFRelation_global_candidate_active_anchor_fraction'] = float(np.mean(
                    train_log['ACFRelation_global_candidate_active_anchor_fraction']
                ))
                train_log['ACFRelation_global_candidate_count_mean'] = float(np.mean(
                    train_log['ACFRelation_global_candidate_count_mean']
                ))
                train_log['ACFRelation_global_contrastive_active_anchor_fraction'] = float(np.mean(
                    train_log['ACFRelation_global_contrastive_active_anchor_fraction']
                ))
                train_log['ACFRelation_global_degenerate_positive_anchor_fraction'] = float(np.mean(
                    train_log['ACFRelation_global_degenerate_positive_anchor_fraction']
                ))
                train_log['ACFRelation_global_positive_count_mean'] = float(np.mean(
                    train_log['ACFRelation_global_positive_count_mean']
                ))
                train_log['ACFRelation_global_high_acf_excluded_fraction'] = float(np.mean(
                    train_log['ACFRelation_global_high_acf_excluded_fraction']
                ))
                train_log['ACFRelation_global_high_acf_candidate_count'] = float(np.mean(
                    train_log['ACFRelation_global_high_acf_candidate_count']
                ))
                train_log['ACFRelation_global_high_acf_excluded_count'] = float(np.mean(
                    train_log['ACFRelation_global_high_acf_excluded_count']
                ))
                train_log['ACFRelation_global_high_acf_conditional_excluded_fraction'] = float(np.mean(
                    train_log['ACFRelation_global_high_acf_conditional_excluded_fraction']
                ))
                train_log['ACFRelation_global_sign_target_count'] = float(np.mean(
                    train_log['ACFRelation_global_sign_target_count']
                ))
                train_log['ACFRelation_global_sign_accuracy'] = float(np.mean(
                    train_log['ACFRelation_global_sign_accuracy']
                ))
                train_log['ACFRelation_global_sign_hinge'] = float(np.mean(
                    train_log['ACFRelation_global_sign_hinge']
                ))
                train_log['ACFRelation_local_target_fraction'] = float(np.mean(
                    train_log['ACFRelation_local_target_fraction']
                ))
                train_log['ACFRelation_local_active_position_fraction'] = float(np.mean(
                    train_log['ACFRelation_local_active_position_fraction']
                ))
                train_log['ACFRelation_local_sign_target_count'] = float(np.mean(
                    train_log['ACFRelation_local_sign_target_count']
                ))
                train_log['ACFRelation_local_sign_accuracy'] = float(np.mean(
                    train_log['ACFRelation_local_sign_accuracy']
                ))
                train_log['ACFRelation_local_sign_hinge'] = float(np.mean(
                    train_log['ACFRelation_local_sign_hinge']
                ))
            else:
                train_log['ACFRelation_global_valid_fraction'] = None
                train_log['ACFRelation_global_active_anchor_fraction'] = None
                train_log['ACFRelation_global_candidate_fraction'] = None
                train_log['ACFRelation_global_candidate_active_anchor_fraction'] = None
                train_log['ACFRelation_global_candidate_count_mean'] = None
                train_log['ACFRelation_global_contrastive_active_anchor_fraction'] = None
                train_log['ACFRelation_global_degenerate_positive_anchor_fraction'] = None
                train_log['ACFRelation_global_positive_count_mean'] = None
                train_log['ACFRelation_global_high_acf_excluded_fraction'] = None
                train_log['ACFRelation_global_high_acf_candidate_count'] = None
                train_log['ACFRelation_global_high_acf_excluded_count'] = None
                train_log['ACFRelation_global_high_acf_conditional_excluded_fraction'] = None
                train_log['ACFRelation_global_sign_target_count'] = None
                train_log['ACFRelation_global_sign_accuracy'] = None
                train_log['ACFRelation_global_sign_hinge'] = None
                train_log['ACFRelation_local_target_fraction'] = None
                train_log['ACFRelation_local_active_position_fraction'] = None
                train_log['ACFRelation_local_sign_target_count'] = None
                train_log['ACFRelation_local_sign_accuracy'] = None
                train_log['ACFRelation_local_sign_hinge'] = None

            for diagnostic_key in ACF_RUNTIME_DIAGNOSTIC_KEYS:
                values = train_log.get(diagnostic_key, [])
                train_log[diagnostic_key] = (
                    float(np.mean(values)) if values else None
                )
            _reduce_relation_component_epoch(train_log)

            # The lag-pair sampler records accepted disjoint pairs during its
            # iteration.  Persist the epoch-level exposure so a balanced
            # screen can be audited instead of inferred from its CLI flag.
            lag_counts = getattr(batch_sampler, 'last_lag_pair_counts', None)
            if isinstance(lag_counts, dict) and lag_counts:
                normalized_counts = {
                    str(int(lag)): int(count)
                    for lag, count in sorted(lag_counts.items(), key=lambda item: int(item[0]))
                }
                total_pairs = sum(normalized_counts.values())
                normalized_exposure = {
                    lag: (float(count) / float(total_pairs) if total_pairs else 0.0)
                    for lag, count in normalized_counts.items()
                }
                logs['acf_relation_lag_pair_counts'].append(normalized_counts)
                logs['acf_relation_lag_pair_exposure'].append(normalized_exposure)

            vali_loss = self.vali(vali_data, vali_loader, criterion)

            coverage_suffix = ''
            if train_log['ACFRelation_global_valid_fraction'] is not None:
                coverage_suffix = (
                    f", ACF Global Coverage:{train_log['ACFRelation_global_valid_fraction']:.2%}"
                    f", Active Anchors:{train_log['ACFRelation_global_active_anchor_fraction']:.2%}"
                    f", Candidates:{train_log['ACFRelation_global_candidate_fraction']:.2%}"
                    f", Candidate Anchors:{train_log['ACFRelation_global_candidate_active_anchor_fraction']:.2%}"
                    f", Candidate Count:{train_log['ACFRelation_global_candidate_count_mean']:.2f}"
                    f", Contrastive Anchors:{train_log['ACFRelation_global_contrastive_active_anchor_fraction']:.2%}"
                    f", Degenerate Positive:{train_log['ACFRelation_global_degenerate_positive_anchor_fraction']:.2%}"
                    f", Pos/Anchor:{train_log['ACFRelation_global_positive_count_mean']:.2f}"
                    f", High-ACF Excluded:{train_log['ACFRelation_global_high_acf_excluded_fraction']:.2%}"
                    f", High-ACF Conditional Excluded:{train_log['ACFRelation_global_high_acf_conditional_excluded_fraction']:.2%}"
                    f", Sign Acc:{train_log['ACFRelation_global_sign_accuracy']:.2%}"
                    f", Local Target:{train_log['ACFRelation_local_target_fraction']:.2%}"
                )
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} "
                  f"Train Loss: {train_log['loss']:.4f} (Forecasting Loss:{train_log['MSE_loss']:.4f} + "
                  f"AutoCon Loss:{train_log['AutoCon_loss']:.4f} x Lambda({effective_lambda:.6g}) + "
                  f"Forecast Pair:{train_log['ForecastLagPair_loss']:.4f} x Weight({self.ForecastLagPair_weight:.6g})), "
                  f"Vali MSE Loss: {vali_loss:.4f}{coverage_suffix}")
            logs['train_loss'].append(float(train_log['loss']))
            logs['val_loss'].append(float(vali_loss))
            logs['mse_loss'].append(float(train_log['MSE_loss']))
            logs['autocon_loss'].append(float(train_log['AutoCon_loss']))
            logs['forecast_acf_loss'].append(float(train_log['ForecastACF_loss']))
            logs['forecast_lag_pair_loss'].append(
                float(train_log['ForecastLagPair_loss'])
            )
            logs['forecast_lag_pair_fraction'].append(
                train_log['ForecastLagPair_fraction']
            )
            logs['forecast_lag_pair_active_anchor_fraction'].append(
                train_log['ForecastLagPair_active_anchor_fraction']
            )
            logs['forecast_lag_pair_count'].append(
                train_log['ForecastLagPair_count']
            )
            logs['forecast_lag_pair_target_abs_mean'].append(
                train_log['ForecastLagPair_target_abs_mean']
            )
            logs['effective_lambda'].append(float(effective_lambda))
            logs['acf_relation_global_valid_fraction'].append(
                train_log['ACFRelation_global_valid_fraction']
            )
            logs['acf_relation_global_active_anchor_fraction'].append(
                train_log['ACFRelation_global_active_anchor_fraction']
            )
            logs['acf_relation_global_candidate_fraction'].append(
                train_log['ACFRelation_global_candidate_fraction']
            )
            logs['acf_relation_global_candidate_active_anchor_fraction'].append(
                train_log['ACFRelation_global_candidate_active_anchor_fraction']
            )
            logs['acf_relation_global_candidate_count_mean'].append(
                train_log['ACFRelation_global_candidate_count_mean']
            )
            logs['acf_relation_global_contrastive_active_anchor_fraction'].append(
                train_log['ACFRelation_global_contrastive_active_anchor_fraction']
            )
            logs['acf_relation_global_degenerate_positive_anchor_fraction'].append(
                train_log['ACFRelation_global_degenerate_positive_anchor_fraction']
            )
            logs['acf_relation_global_positive_count_mean'].append(
                train_log['ACFRelation_global_positive_count_mean']
            )
            logs['acf_relation_global_high_acf_excluded_fraction'].append(
                train_log['ACFRelation_global_high_acf_excluded_fraction']
            )
            logs['acf_relation_global_high_acf_candidate_count'].append(
                train_log['ACFRelation_global_high_acf_candidate_count']
            )
            logs['acf_relation_global_high_acf_excluded_count'].append(
                train_log['ACFRelation_global_high_acf_excluded_count']
            )
            logs['acf_relation_global_high_acf_conditional_excluded_fraction'].append(
                train_log['ACFRelation_global_high_acf_conditional_excluded_fraction']
            )
            logs['acf_relation_global_sign_target_count'].append(
                train_log['ACFRelation_global_sign_target_count']
            )
            logs['acf_relation_global_sign_accuracy'].append(
                train_log['ACFRelation_global_sign_accuracy']
            )
            logs['acf_relation_global_sign_hinge'].append(
                train_log['ACFRelation_global_sign_hinge']
            )
            logs['acf_relation_local_target_fraction'].append(
                train_log['ACFRelation_local_target_fraction']
            )
            logs['acf_relation_local_active_position_fraction'].append(
                train_log['ACFRelation_local_active_position_fraction']
            )
            logs['acf_relation_local_sign_target_count'].append(
                train_log['ACFRelation_local_sign_target_count']
            )
            logs['acf_relation_local_sign_accuracy'].append(
                train_log['ACFRelation_local_sign_accuracy']
            )
            logs['acf_relation_local_sign_hinge'].append(
                train_log['ACFRelation_local_sign_hinge']
            )
            for diagnostic_key in ACF_RUNTIME_DIAGNOSTIC_KEYS:
                logs[diagnostic_key].append(train_log[diagnostic_key])
            _append_relation_component_epoch(logs, train_log)
            logs['epoch_time_sec'].append(float(time.time() - epoch_time))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        logs['best_epoch'] = int(np.argmin(logs['val_loss']) + 1) if logs['val_loss'] else None
        logs['epochs_ran'] = len(logs['val_loss'])
        with open(os.path.join(path, 'loss_logs.json'), 'w', encoding='utf-8') as handle:
            import json
            json.dump(logs, handle, indent=2)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            checkpoint = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        stream_metrics = bool(getattr(self.args, 'stream_metrics', False))
        origin_stride = getattr(self.args, 'origin_stride', None) or self.args.pred_len
        accumulator = (
            StreamingMetricAccumulator(
                test_data.scaler,
                int(self.args.c_out if self.args.features != 'MS' else 1),
                int(self.args.pred_len),
                origin_stride=int(origin_stride),
                sample_limit=int(getattr(self.args, 'stream_sample_windows', 4)),
            )
            if stream_metrics else None
        )
        preds = [] if not stream_metrics else None
        trues = [] if not stream_metrics else None
        test_root = getattr(self.args, 'test_results_dir', './test_results/')
        folder_path = os.path.join(test_root, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (timeindex, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs, _ = self._forecast_output(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                if stream_metrics:
                    accumulator.update(pred, true)
                else:
                    preds.append(pred)
                    trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

        if stream_metrics:
            result_root = getattr(self.args, 'results_dir', './results/')
            result_dir = os.path.join(result_root, setting)
            payload = write_streaming_artifacts(result_dir, accumulator)
            scaled = payload['scaled']
            mae, mse, rmse = scaled['mae'], scaled['mse'], scaled['rmse']
            mape, mspe = scaled['mape'], scaled['mspe']
            print('streamed test windows:', payload['window_count'])
            print(f'mse:{mse}, mae:{mae}, mape:{mape}, mspe:{mspe} '
                  'dilate:0.0000000, Shapedtw:0.0000000, Temporaldtw:0.0000000')
            result_log = getattr(self.args, 'result_log', 'result_long_term_forecast.txt')
            with open(result_log, 'a', encoding='utf-8') as handle:
                handle.write(setting + "  \n")
                handle.write('mse:{}, mae:{}'.format(mse, mae))
                handle.write('\n\n')
            return mse, mae, mape, mspe, 0.0, 0.0, 0.0

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        result_root = getattr(self.args, 'results_dir', './results/')
        folder_path = os.path.join(result_root, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # dilate_e, shape_e, temporal_e = shape_metric(preds, trues)  # These metrics take a long time to calculate.
        dilate_e, shape_e,temporal_e  = 0.0,  0.0,  0.0

        print(f'mse:{mse}, mae:{mae}, mape:{mape}, mspe:{mspe} dilate:{dilate_e:.7f}, Shapedtw:{shape_e:.7f}, Temporaldtw:{temporal_e:.7f}')
        result_log = getattr(self.args, 'result_log', 'result_long_term_forecast.txt')
        f = open(result_log, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        original_preds, original_trues = inverse_scaled_arrays(preds, trues, test_data.scaler)
        original_metrics = metric(original_preds, original_trues)
        channel_metrics = np.column_stack((
            per_channel_metrics(preds, trues),
            per_channel_metrics(original_preds, original_trues),
        ))
        np.save(folder_path + 'original_metrics.npy', np.asarray(original_metrics))
        np.save(folder_path + 'channel_metrics.npy', channel_metrics)
        if self.args.save:
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

                # Optional: export ONNX for visualization
        if getattr(self.args, 'export_onnx', False):
            try:
                export_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                export_model.eval()
                # get a sample batch
                sample = next(iter(test_loader))
                _, sx, sy, sxm, sym = sample
                # use a single sample to keep ONNX small
                sx = sx[:1].float().to(self.device)
                sy = sy[:1].float().to(self.device)
                sxm = sxm[:1].float().to(self.device)
                sym = sym[:1].float().to(self.device)

                # build decoder input same as during test
                dec_inp_sample = torch.zeros_like(sy[:, -self.args.pred_len:, :]).float()
                dec_inp_sample = torch.cat([sy[:, :self.args.label_len, :], dec_inp_sample], dim=1).float().to(self.device)

                onnx_dir = os.path.join(self.args.checkpoints, setting)
                os.makedirs(onnx_dir, exist_ok=True)
                onnx_path = os.path.join(onnx_dir, 'model.onnx')

                input_names = ["batch_x", "batch_x_mark", "dec_inp", "batch_y_mark"]
                output_names = ["output"]
                dynamic_axes = {
                    "batch_x": {0: "batch", 1: "seq_len"},
                    "batch_x_mark": {0: "batch", 1: "seq_len"},
                    "dec_inp": {0: "batch", 1: "total_len"},
                    "batch_y_mark": {0: "batch", 1: "total_len"},
                    "output": {0: "batch"}
                }

                torch.onnx.export(
                    export_model,
                    (sx, sxm, dec_inp_sample, sym),
                    onnx_path,
                    opset_version=17,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                )
                print(f"Exported ONNX model to: {onnx_path}")
            except Exception as e:
                print(f"Failed to export ONNX: {e}")

        return mse, mae, mape, mspe, dilate_e, shape_e, temporal_e
