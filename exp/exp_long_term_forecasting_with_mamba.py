"""Experiment entry points for the repository's Mamba-1 and Mamba-2 forecasters.

The forecasting loop is shared with the original long-term forecasting
experiment, but Mamba is kept in an explicit experiment module so its model
contract is visible and cannot accidentally be treated as an attention model.

The module exposes separate experiment classes for Mamba-1 and Mamba-2, with
and without the AutoCon objective.
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import os

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_term_forecasting_with_AutoCon import (
    Exp_Long_Term_Forecast_with_AutoCon,
)
from models import (
    Mamba1AutoCon,
    Mamba2AutoCon,
    MambaACF,
    MambaACF1,
    MambaTM1AutoCon,
    MambaTM2AutoCon,
    MambaSMambaAutoCon,
    RaMambaLite,
    BiPatchMambaAutoCon,
    LatentTemporalMamba,
    InvertedRaMambaAutoCon,
    LinearLatentMambaAutoCon,
    NLinearVariableMambaAutoCon,
    MambaTS,
    MambaTSv2,
    MambaTSv3,
    MambaTSv4,
    MambaTSv5,
    MambaTSv6,
    MambaTSv7,
    MambaTSv8,
    MambaTSv9,
    MambaTSv10,
    MambaTSv11,
    MambaTSv12,
    MambaTSv13,
    MambaTSv14,
    MambaTSv15,
    MambaTSv16,
)
from models.AutoConNet import series_decomp
from utils.acf_guidance import fit_signed_acf_prior
from utils.acf_relation_contract import (
    mamba_acf_prior_required,
    resolve_acf_source,
)


class _MambaExperimentMixin:
    """Build and validate an encoder-only Mamba forecasting model."""

    expected_model = None
    requires_autocon = None
    model_module = None
    backbone_name = None
    accepted_models = None

    def _attach_train_acf_prior(self):
        """Fit and attach a compact ACF prior without touching val/test data."""
        if getattr(self, "_acf_prior_attached", False):
            return
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        prior_setter = getattr(model, "set_acf_prior", None)
        if prior_setter is None:
            prior_setter = getattr(model, "set_ts_acf_prior", None)
        if prior_setter is None:
            return

        # A standalone evaluation may provide the exact sidecar emitted by a
        # prior run.  Load it before touching the dataset so evaluation is
        # reproducible even when the original train CSV is unavailable.
        artifact_path = getattr(self.args, "mamba_acf_prior_artifact_path", None)
        # Training keeps a prior for every MambaACF ablation so its provenance
        # and strict phase validators remain comparable.  Evaluation of a
        # plain ACF-off checkpoint does not consume that metadata, however;
        # avoid reopening the train split (which may not be present on an
        # inference-only machine) unless a live ACF branch or explicit sidecar
        # makes it necessary.  Missing flags in small direct-API fixtures are
        # handled conservatively by treating the call as training-like.
        is_training = getattr(self.args, "is_training", True)
        if isinstance(is_training, str):
            is_training = is_training.strip().lower() not in {
                "0", "false", "no", "off", ""
            }
        else:
            is_training = bool(is_training)
        needs_train_reconstruction = bool(
            getattr(model, "ts_acf_memory_enabled", False)
            or getattr(model, "ts_acf_order_enabled", False)
        )
        if (
            not is_training
            and not mamba_acf_prior_required(self.args)
            and not needs_train_reconstruction
        ):
            return
        if artifact_path is not None and str(artifact_path).strip():
            artifact_path = os.path.abspath(str(artifact_path))
            if not os.path.isfile(artifact_path):
                raise RuntimeError(
                    "configured MambaACF prior artifact is missing: {}".format(
                        artifact_path
                    )
                )
            expected_sha = str(
                getattr(self.args, "mamba_acf_prior_expected_sha256", "")
            ).strip().lower()
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
            # ``acf_lags`` is the compact forecast table.  Preserve the full
            # measured range in metadata when the sidecar carries it; using
            # only the largest selected lag makes a resumed evaluation look
            # as if the train ACF had been fitted to a shorter range.
            prior_state = model.export_acf_prior_state()
            full_values = (
                None if prior_state is None
                else prior_state.get("acf_full_values")
            )
            if full_values is not None:
                full_values = np.asarray(full_values)
                if full_values.ndim != 2 or full_values.shape[1] < 2:
                    raise RuntimeError(
                        "loaded MambaACF prior has an invalid complete table"
                    )
                prior_max_lag = int(full_values.shape[1] - 1)
            else:
                prior_max_lag = int(max(model.acf_lags, default=0))
            self.args.mamba_acf_effective_lags = list(model.acf_lags)
            self.args.mamba_acf_effective_lag_source = "persisted_sidecar"
            self.args.mamba_acf_effective_lag_fallback = False
            self.args.mamba_acf_prior_max_lag = prior_max_lag
            self._acf_prior_attached = True
            print("MambaACF prior loaded from sidecar: {}".format(artifact_path))
            return
        train_data, _ = self._get_data(flag="train")
        values = np.asarray(train_data.data_x, dtype=np.float64)
        # Keep plain and AutoCon MambaACF arms on the same train-only signal;
        # otherwise an apparent objective gain can be a prior-source effect.
        source = resolve_acf_source(self.args)
        if source not in {"residual", "trend", "raw"}:
            raise ValueError("AutoCon_acf_source must be one of: residual, trend, raw")
        if source != "raw":
            kernel_size = max(3, int(self.args.seq_len) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            decomposition = series_decomp(kernel_size)
            signal, trend = decomposition(torch.from_numpy(values).unsqueeze(0))
            values = (signal if source == "residual" else trend).squeeze(0).numpy()
        configured_lags = getattr(self.args, "mamba_acf_lags", [24, 168, 336])
        if isinstance(configured_lags, str):
            configured_lags = [int(item) for item in configured_lags.replace(",", " ").split() if item]
        max_configured_lag = max([int(item) for item in configured_lags if int(item) > 0], default=1)
        max_lag = min(
            values.shape[0] - 1,
            max(
                max_configured_lag,
                int(getattr(self.args, "mamba_acf_max_lag", 720)),
                int(self.args.seq_len) - 1,
            ),
        )
        prior = fit_signed_acf_prior(
            values,
            max_lag=max_lag,
            top_k=max(1, int(getattr(self.args, "mamba_acf_top_k", 8)), len(configured_lags)),
            min_lag=max(1, int(getattr(self.args, "mamba_acf_min_lag", 2))),
            min_abs_acf=float(getattr(self.args, "mamba_acf_min_abs", 0.1)),
            confidence_z=float(getattr(self.args, "mamba_acf_confidence_z", 1.96)),
            min_separation=max(1, int(getattr(self.args, "mamba_acf_min_separation", 2))),
            detrend=str(getattr(self.args, "mamba_acf_detrend", "mean")),
            channel_chunk_size=int(getattr(self.args, "mamba_acf_chunk", 64)),
        )
        train_peak_lags = tuple(sorted({
            int(lag) for lag in prior.common_lags(top_k=None)
            if 0 < int(lag) <= prior.max_lag
        }))
        self.args.mamba_acf_train_peak_lags = list(train_peak_lags)
        self.args.mamba_acf_relation_train_peak_lags = list(train_peak_lags)
        mode = str(getattr(self.args, "mamba_acf_lag_mode", "fixed")).lower()
        selected_lag_fallback = False
        if mode == "train_peaks":
            lags = prior.common_lags(top_k=max(1, len(configured_lags)))
            if not lags:
                lags = tuple(int(item) for item in configured_lags if int(item) > 0)
                selected_lag_fallback = True
        elif mode == "train_peaks_available":
            lags = prior.availability_weighted_lags(
                history_length=int(self.args.seq_len),
                horizon=int(self.args.pred_len),
                top_k=max(1, len(configured_lags)),
                min_separation=max(1, int(getattr(self.args, "mamba_acf_min_separation", 2))),
            )
            if not lags:
                lags = tuple(int(item) for item in configured_lags if int(item) > 0)
                selected_lag_fallback = True
        elif mode == "fixed":
            lags = tuple(int(item) for item in configured_lags if int(item) > 0)
        else:
            raise ValueError(
                "mamba_acf_lag_mode must be 'fixed', 'train_peaks', or "
                "'train_peaks_available'"
            )
        lags = tuple(sorted({
            int(lag) for lag in lags if 0 < int(lag) <= prior.max_lag
        }))
        if not lags:
            raise ValueError(
                "no selected MambaACF lag is available in the train split; "
                "reduce --mamba_acf_lags or use a train-peak lag mode"
            )
        self.args.mamba_acf_effective_lags = list(lags)
        self.args.mamba_acf_effective_lag_source = (
            "configured_fallback" if selected_lag_fallback
            else "fixed_configured" if mode == "fixed" else mode
        )
        self.args.mamba_acf_effective_lag_fallback = bool(selected_lag_fallback)
        self.args.mamba_acf_prior_max_lag = int(prior.max_lag)
        prior_setter(prior.acf, lags=lags)
        self._acf_prior_attached = True
        print("MambaACF train-only prior: lags={}, max_lag={}, channels={}".format(
            list(lags), max_lag, prior.n_channels
        ))

    def _validate_mamba_args(self, args):
        accepted_models = self.accepted_models or (self.expected_model,)
        if getattr(args, "model", None) not in accepted_models:
            raise ValueError(
                "{} must be used with --model {} (received {!r})".format(
                    self.__class__.__name__, ", ".join(accepted_models),
                    getattr(args, "model", None)
                )
            )

        using_autocon = bool(getattr(args, "AutoCon", False))
        if using_autocon != self.requires_autocon:
            if self.requires_autocon:
                raise ValueError(
                    "{} requires --AutoCon because it trains the contrastive "
                    "objective".format(self.__class__.__name__)
                )
            raise ValueError(
                "{} is the plain {} experiment; remove --AutoCon or use "
                "--model {}AutoCon".format(
                    self.__class__.__name__, self.expected_model, self.expected_model
                )
            )

        # Mamba is an encoder-only sequence mixer.  It returns a tensor (or a
        # prediction/representation tuple for AutoCon), never attention maps.
        # The generic experiment's output_attention branch indexes tensor[0],
        # which would incorrectly remove the batch dimension.
        if bool(getattr(args, "output_attention", False)):
            raise ValueError("Mamba experiments do not support --output_attention")

    def _build_model(self):
        if self.model_module is None or self.backbone_name is None:
            raise RuntimeError("Mamba experiment subclass must select a model module")
        model = self.model_module.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print("{} experiment model: {}".format(self.backbone_name, self.expected_model))
        print("model parameters:{}".format(self.count_parameters(model)))
        return model


class Exp_Long_Term_Forecast_with_Mamba1(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Plain Mamba-1 long-term forecasting with the prediction loss only."""

    expected_model = "Mamba1"
    requires_autocon = False
    model_module = Mamba1AutoCon
    backbone_name = "Mamba1"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_Mamba1_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """Mamba-1 long-term forecasting with the AutoCon objective."""

    expected_model = "Mamba1AutoCon"
    requires_autocon = True
    model_module = Mamba1AutoCon
    backbone_name = "Mamba1"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_Mamba2(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Plain Mamba-2 long-term forecasting with the prediction loss only."""

    expected_model = "Mamba2"
    requires_autocon = False
    model_module = Mamba2AutoCon
    backbone_name = "Mamba2"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_Mamba2_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """Mamba-2 long-term forecasting with the AutoCon objective."""

    expected_model = "Mamba2AutoCon"
    requires_autocon = True
    model_module = Mamba2AutoCon
    backbone_name = "Mamba2"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTS(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS-inspired patch/variable-scan forecast without AutoCon."""

    expected_model = "MambaTS"
    requires_autocon = False
    model_module = MambaTS
    backbone_name = "MambaTS"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTS_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS-inspired patch/variable-scan forecast with AutoCon."""

    expected_model = "MambaTSAutoCon"
    requires_autocon = True
    model_module = MambaTS
    backbone_name = "MambaTS"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv2(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v2 with independent per-variable temporal scans."""

    expected_model = "MambaTSv2"
    requires_autocon = False
    model_module = MambaTSv2
    backbone_name = "MambaTSv2"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv2_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v2 with the AutoCon representation objective."""

    expected_model = "MambaTSv2AutoCon"
    requires_autocon = True
    model_module = MambaTSv2
    backbone_name = "MambaTSv2"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv3(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v3 shared scan with low-rank cross-variable context."""

    expected_model = "MambaTSv3"
    requires_autocon = False
    model_module = MambaTSv3
    backbone_name = "MambaTSv3"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv3_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v3 with the AutoCon representation objective."""

    expected_model = "MambaTSv3AutoCon"
    requires_autocon = True
    model_module = MambaTSv3
    backbone_name = "MambaTSv3"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv4(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v4 with a train-only causal ACF forecast residual."""

    expected_model = "MambaTSv4"
    requires_autocon = False
    model_module = MambaTSv4
    backbone_name = "MambaTSv4"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaTSv4_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v4 with train-only ACF memory and AutoCon."""

    expected_model = "MambaTSv4AutoCon"
    requires_autocon = True
    model_module = MambaTSv4
    backbone_name = "MambaTSv4"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaTSv5(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v5 plain forecast with a causal train-only ACF anchor."""

    expected_model = "MambaTSv5"
    requires_autocon = False
    model_module = MambaTSv5
    backbone_name = "MambaTSv5"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaTSv5_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v5 with AutoCon encoder relation learning."""

    expected_model = "MambaTSv5AutoCon"
    requires_autocon = True
    model_module = MambaTSv5
    backbone_name = "MambaTSv5"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaTSv6(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v6 with a DLinear-style base and Mamba residual correction."""

    expected_model = "MambaTSv6"
    requires_autocon = False
    model_module = MambaTSv6
    backbone_name = "MambaTSv6"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv6_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v6 with its time-indexed AutoCon representation objective."""

    expected_model = "MambaTSv6AutoCon"
    requires_autocon = True
    model_module = MambaTSv6
    backbone_name = "MambaTSv6"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv7(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v7 with a linear base and train-only ACF memory."""

    expected_model = "MambaTSv7"
    requires_autocon = False
    model_module = MambaTSv7
    backbone_name = "MambaTSv7"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaTSv7_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v7 with AutoCon-selected train-only ACF correction."""

    expected_model = "MambaTSv7AutoCon"
    requires_autocon = True
    model_module = MambaTSv7
    backbone_name = "MambaTSv7"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaTSv8(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v8 with a causal per-window spectral residual."""

    expected_model = "MambaTSv8"
    requires_autocon = False
    model_module = MambaTSv8
    backbone_name = "MambaTSv8"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv8_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v8 with AutoCon-connected spectral gating."""

    expected_model = "MambaTSv8AutoCon"
    requires_autocon = True
    model_module = MambaTSv8
    backbone_name = "MambaTSv8"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv9(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v9 with a seasonal Mamba scan and trend-linear branch."""

    expected_model = "MambaTSv9"
    requires_autocon = False
    model_module = MambaTSv9
    backbone_name = "MambaTSv9"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv9_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v9 with AutoCon-connected trend-seasonal fusion."""

    expected_model = "MambaTSv9AutoCon"
    requires_autocon = True
    model_module = MambaTSv9
    backbone_name = "MambaTSv9"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv10(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v10 with a fixed train-ACF VAST-lite channel order."""

    expected_model = "MambaTSv10"
    requires_autocon = False
    model_module = MambaTSv10
    backbone_name = "MambaTSv10"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaTSv10_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v10 with AutoCon-connected representation fusion."""

    expected_model = "MambaTSv10AutoCon"
    requires_autocon = True
    model_module = MambaTSv10
    backbone_name = "MambaTSv10"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaTSv11(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """MambaTS v11 with VPT transition-cost order learning."""

    expected_model = "MambaTSv11"
    requires_autocon = False
    model_module = MambaTSv11
    backbone_name = "MambaTSv11"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv11_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """MambaTS v11 with VPT and an AutoCon-connected forecast residual."""

    expected_model = "MambaTSv11AutoCon"
    requires_autocon = True
    model_module = MambaTSv11
    backbone_name = "MambaTSv11"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv12(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Two-scale Delta-MambaTS forecast without the AutoCon objective."""

    expected_model = "MambaTSv12"
    requires_autocon = False
    model_module = MambaTSv12
    backbone_name = "MambaTSv12"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv12_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """Two-scale Delta-MambaTS with the shared AutoCon representation path."""

    expected_model = "MambaTSv12AutoCon"
    requires_autocon = True
    model_module = MambaTSv12
    backbone_name = "MambaTSv12"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv13(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """One-scan lightweight multi-scale MambaTS forecast."""

    expected_model = "MambaTSv13"
    requires_autocon = False
    model_module = MambaTSv13
    backbone_name = "MambaTSv13"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv13_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """One-scan lightweight multi-scale MambaTS with AutoCon."""

    expected_model = "MambaTSv13AutoCon"
    requires_autocon = True
    model_module = MambaTSv13
    backbone_name = "MambaTSv13"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv14(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """One-scan coarse-patch MambaTS with a local observed-window expert."""

    expected_model = "MambaTSv14"
    requires_autocon = False
    model_module = MambaTSv14
    backbone_name = "MambaTSv14"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv14_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """One-scan coarse-patch MambaTS with shared AutoCon representation."""

    expected_model = "MambaTSv14AutoCon"
    requires_autocon = True
    model_module = MambaTSv14
    backbone_name = "MambaTSv14"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv15(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """One-scan time-conditioned MambaTS forecast."""

    expected_model = "MambaTSv15"
    requires_autocon = False
    model_module = MambaTSv15
    backbone_name = "MambaTSv15"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv15_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """One-scan time-conditioned MambaTS with shared AutoCon representation."""

    expected_model = "MambaTSv15AutoCon"
    requires_autocon = True
    model_module = MambaTSv15
    backbone_name = "MambaTSv15"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv16(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """One-scan anchor-conditioned MambaTS forecast."""

    expected_model = "MambaTSv16"
    requires_autocon = False
    model_module = MambaTSv16
    backbone_name = "MambaTSv16"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTSv16_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """One-scan anchor-conditioned MambaTS with shared AutoCon representation."""

    expected_model = "MambaTSv16AutoCon"
    requires_autocon = True
    model_module = MambaTSv16
    backbone_name = "MambaTSv16"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTM1(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Plain TimeMachine-style four-path Mamba-1 forecast."""

    expected_model = "MambaTM1"
    requires_autocon = False
    model_module = MambaTM1AutoCon
    backbone_name = "MambaTM1"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTM1_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """TimeMachine-style Mamba-1 with the AutoCon representation objective."""

    expected_model = "MambaTM1AutoCon"
    requires_autocon = True
    model_module = MambaTM1AutoCon
    backbone_name = "MambaTM1"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTM2(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Plain TimeMachine-style four-path Mamba-2 forecast."""

    expected_model = "MambaTM2"
    requires_autocon = False
    model_module = MambaTM2AutoCon
    backbone_name = "MambaTM2"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaTM2_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """TimeMachine-style Mamba-2 with the AutoCon representation objective."""

    expected_model = "MambaTM2AutoCon"
    requires_autocon = True
    model_module = MambaTM2AutoCon
    backbone_name = "MambaTM2"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaSM(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """S-Mamba-inspired variable-token Mamba forecast."""

    expected_model = "MambaSM"
    requires_autocon = False
    model_module = MambaSMambaAutoCon
    backbone_name = "MambaSM"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaSM_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """S-Mamba-inspired variable-token Mamba with AutoCon."""

    expected_model = "MambaSMAutoCon"
    requires_autocon = True
    model_module = MambaSMambaAutoCon
    backbone_name = "MambaSM"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_RaMambaLite(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Single-scan decomposed variable Mamba forecast."""

    expected_model = "RaMambaLite"
    requires_autocon = False
    model_module = RaMambaLite
    backbone_name = "RaMambaLite"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_RaMambaLite_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """Single-scan decomposed variable Mamba with AutoCon representation learning."""

    expected_model = "RaMambaLiteAutoCon"
    requires_autocon = True
    model_module = RaMambaLite
    backbone_name = "RaMambaLite"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_BiPatchMamba(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Bi-Mamba4TS-inspired bidirectional patch/variable Mamba forecast."""

    expected_model = "BiPatchMamba"
    requires_autocon = False
    model_module = BiPatchMambaAutoCon
    backbone_name = "BiPatchMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_BiPatchMamba_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """Bi-Mamba4TS-inspired Mamba with a direct AutoCon representation route."""

    expected_model = "BiPatchMambaAutoCon"
    requires_autocon = True
    model_module = BiPatchMambaAutoCon
    backbone_name = "BiPatchMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_LatentTemporalMamba(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Compressed temporal Mamba residual with a linear forecast base."""

    expected_model = "LatentTemporalMamba"
    requires_autocon = False
    model_module = LatentTemporalMamba
    backbone_name = "LatentTemporalMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_LatentTemporalMamba_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """Compressed temporal Mamba with direct AutoCon representation forecast path."""

    expected_model = "LatentTemporalMambaAutoCon"
    requires_autocon = True
    model_module = LatentTemporalMamba
    backbone_name = "LatentTemporalMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_InvertedRaMamba(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Inverted variable-token RaMamba-style forecast."""

    expected_model = "InvertedRaMamba"
    requires_autocon = False
    model_module = InvertedRaMambaAutoCon
    backbone_name = "InvertedRaMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_InvertedRaMamba_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """Inverted variable-token RaMamba-style forecast with AutoCon."""

    expected_model = "InvertedRaMambaAutoCon"
    requires_autocon = True
    model_module = InvertedRaMambaAutoCon
    backbone_name = "InvertedRaMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_LinearLatentMamba(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """NLinear base plus low-rank latent Mamba residual."""

    expected_model = "LinearLatentMamba"
    requires_autocon = False
    model_module = LinearLatentMambaAutoCon
    backbone_name = "LinearLatentMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_LinearLatentMamba_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """NLinear base plus low-rank latent Mamba residual with AutoCon."""

    expected_model = "LinearLatentMambaAutoCon"
    requires_autocon = True
    model_module = LinearLatentMambaAutoCon
    backbone_name = "LinearLatentMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_NLinearVariableMamba(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """NLinear base plus one variable-axis Mamba residual."""

    expected_model = "NLinearVariableMamba"
    requires_autocon = False
    model_module = NLinearVariableMambaAutoCon
    backbone_name = "NLinearVariableMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_NLinearVariableMamba_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """NLinear base plus one variable-axis Mamba residual with AutoCon."""

    expected_model = "NLinearVariableMambaAutoCon"
    requires_autocon = True
    model_module = NLinearVariableMambaAutoCon
    backbone_name = "NLinearVariableMamba"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)


class Exp_Long_Term_Forecast_with_MambaACF(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Plain forecasting ablation for the proposed ACF-guided Mamba model."""

    expected_model = "MambaACF"
    requires_autocon = False
    model_module = MambaACF
    backbone_name = "MambaACF"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaACF_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """ACF-guided Mamba with the repository's AutoCon representation loss."""

    expected_model = "MambaACFAutoCon"
    accepted_models = ("MambaACFAutoCon", "MambaACFLiteAutoCon")
    requires_autocon = True
    model_module = MambaACF
    backbone_name = "MambaACF"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaACF1(
        _MambaExperimentMixin, Exp_Long_Term_Forecast):
    """Plain Mamba-1 variant of the ACF-guided forecaster."""

    expected_model = "MambaACF1"
    requires_autocon = False
    model_module = MambaACF1
    backbone_name = "MambaACF1"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


class Exp_Long_Term_Forecast_with_MambaACF1_AutoCon(
        _MambaExperimentMixin, Exp_Long_Term_Forecast_with_AutoCon):
    """Mamba-1 ACF-guided forecaster with AutoCon representation learning."""

    expected_model = "MambaACF1AutoCon"
    requires_autocon = True
    model_module = MambaACF1
    backbone_name = "MambaACF1"

    def __init__(self, args):
        self._validate_mamba_args(args)
        super().__init__(args)
        self._attach_train_acf_prior()


# Short aliases make the module convenient to import from small experiment
# scripts while the longer names match the repository's existing convention.
Exp_Long_Term_Forecast_with_Mamba = Exp_Long_Term_Forecast_with_Mamba2
Exp_Long_Term_Forecast_with_Mamba_AutoCon = Exp_Long_Term_Forecast_with_Mamba2_AutoCon
Exp_Long_Term_Forecast_Mamba = Exp_Long_Term_Forecast_with_Mamba2
Exp_Long_Term_Forecast_Mamba_AutoCon = Exp_Long_Term_Forecast_with_Mamba2_AutoCon


__all__ = [
    "Exp_Long_Term_Forecast_with_Mamba1",
    "Exp_Long_Term_Forecast_with_Mamba1_AutoCon",
    "Exp_Long_Term_Forecast_with_Mamba2",
    "Exp_Long_Term_Forecast_with_Mamba2_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTS",
    "Exp_Long_Term_Forecast_with_MambaTS_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv2",
    "Exp_Long_Term_Forecast_with_MambaTSv2_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv3",
    "Exp_Long_Term_Forecast_with_MambaTSv3_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv4",
    "Exp_Long_Term_Forecast_with_MambaTSv4_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv5",
    "Exp_Long_Term_Forecast_with_MambaTSv5_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv6",
    "Exp_Long_Term_Forecast_with_MambaTSv6_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv7",
    "Exp_Long_Term_Forecast_with_MambaTSv7_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv8",
    "Exp_Long_Term_Forecast_with_MambaTSv8_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv9",
    "Exp_Long_Term_Forecast_with_MambaTSv9_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv10",
    "Exp_Long_Term_Forecast_with_MambaTSv10_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv11",
    "Exp_Long_Term_Forecast_with_MambaTSv11_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv12",
    "Exp_Long_Term_Forecast_with_MambaTSv12_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv13",
    "Exp_Long_Term_Forecast_with_MambaTSv13_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv14",
    "Exp_Long_Term_Forecast_with_MambaTSv14_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv15",
    "Exp_Long_Term_Forecast_with_MambaTSv15_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTSv16",
    "Exp_Long_Term_Forecast_with_MambaTSv16_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTM1",
    "Exp_Long_Term_Forecast_with_MambaTM1_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaTM2",
    "Exp_Long_Term_Forecast_with_MambaTM2_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaSM",
    "Exp_Long_Term_Forecast_with_MambaSM_AutoCon",
    "Exp_Long_Term_Forecast_with_RaMambaLite",
    "Exp_Long_Term_Forecast_with_RaMambaLite_AutoCon",
    "Exp_Long_Term_Forecast_with_BiPatchMamba",
    "Exp_Long_Term_Forecast_with_BiPatchMamba_AutoCon",
    "Exp_Long_Term_Forecast_with_LatentTemporalMamba",
    "Exp_Long_Term_Forecast_with_LatentTemporalMamba_AutoCon",
    "Exp_Long_Term_Forecast_with_InvertedRaMamba",
    "Exp_Long_Term_Forecast_with_InvertedRaMamba_AutoCon",
    "Exp_Long_Term_Forecast_with_LinearLatentMamba",
    "Exp_Long_Term_Forecast_with_LinearLatentMamba_AutoCon",
    "Exp_Long_Term_Forecast_with_NLinearVariableMamba",
    "Exp_Long_Term_Forecast_with_NLinearVariableMamba_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaACF",
    "Exp_Long_Term_Forecast_with_MambaACF_AutoCon",
    "Exp_Long_Term_Forecast_with_MambaACF1",
    "Exp_Long_Term_Forecast_with_MambaACF1_AutoCon",
    "Exp_Long_Term_Forecast_with_Mamba",
    "Exp_Long_Term_Forecast_with_Mamba_AutoCon",
    "Exp_Long_Term_Forecast_Mamba",
    "Exp_Long_Term_Forecast_Mamba_AutoCon",
]
