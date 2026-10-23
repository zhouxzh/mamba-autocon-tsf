#!/usr/bin/env python
"""Execute the registered training plan one isolated job at a time.

The runner is intentionally conservative: every job receives its own model id,
log file and output roots; the test split can be disabled for selection; and a
JSONL manifest makes interrupted runs resumable without silently replacing a
completed result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.acf_relation_contract import resolve_acf_source
from utils.mamba_causality_contract import strict_representation_causality_errors


def _supports_variable_repr_fusion(model: object) -> bool:
    """Return whether a registered model has a prediction-connected fusion path."""

    name = str(model)
    return name == "NLinearVariableMambaAutoCon" or (
        "MambaACF" in name and name.endswith("AutoCon")
    )


def _variable_repr_fusion_requires_mixer(model: object) -> bool:
    """The NLinear-variable architecture owns its variable Mamba scan."""

    return str(model) != "NLinearVariableMambaAutoCon"

LAUNCH_AUTHORIZED_STATUSES = frozenset({"approved_for_launch", "launch_authorized"})
LAUNCH_AUTHORIZATION_KIND = "training_plan_launch_authorization"
LAUNCH_AUTHORIZATION_KINDS = frozenset({
    LAUNCH_AUTHORIZATION_KIND,
    # Kind used by the original confirmation protocol draft.
    "mamba_acf_confirmation_launch_authorization",
})
SOURCE_SNAPSHOT_PATHS = (
    Path("run.py"),
    Path("data_provider"),
    Path("exp"),
    Path("layers"),
    Path("models"),
    Path("utils"),
    Path("scripts/run_training_plan.py"),
    Path("scripts/select_hyperparameters.py"),
    Path("scripts/attest_selection.py"),
    Path("scripts/validate_training_phase.py"),
    Path("scripts/build_final_report.py"),
    Path("scripts/collect_results.py"),
    Path("scripts/compute_naive_baselines.py"),
    Path("scripts/analyze_datasets.py"),
    Path("scripts/audit_dataset_quality.py"),
    Path("scripts/validate_highdim_contract.py"),
    Path("scripts/summarize_channel_metrics.py"),
    # Keep analysis code bound to each confirmatory phase's immutable snapshot.
    Path("scripts/compute_paper_statistics.py"),
    Path("scripts/analyze_mamba_acf_confirmation.py"),
    Path("tests/test_confirmation_statistics.py"),
    Path("scripts/benchmark_model_resources.py"),
    # Keep exploratory mechanism readers reproducible with the phase source
    # snapshot when a later audit replays their hashes.
    Path("scripts/analyze_mamba_acf_horizon_calibration_screen.py"),
    Path("scripts/analyze_mamba_acf_fair_baseline_screen.py"),
    Path("scripts/analyze_mamba_acf_channel_mixer_electricity_h96.py"),
    Path("scripts/analyze_mambatsv3_d24_highdim_h12_transfer.py"),
    Path("scripts/analyze_metro_pemsbay_mambatsv3_autocon_h12_screen.py"),
    Path("scripts/analyze_graph_spatial_mambatsv3_screen.py"),
    Path("scripts/analyze_latent_temporal_mamba_highdim_validation.py"),
    Path("scripts/audit_latent_temporal_mamba_highdim_plan.py"),
    Path("scripts/smoke_latent_temporal_mamba_autocon.py"),
    Path("scripts/pretrain_mamba_autocon.py"),
    Path("scripts/finetune_mamba_pretrain.py"),
    Path("scripts/finetune_mambatsv3_core_transfer.py"),
)
DATASETS = {
    "ETTh1": {"root": "dataset/ETT-small", "path": "ETTh1.csv", "freq": "h", "target": "OT", "time_col": "date", "channels": 7},
    "ETTh2": {"root": "dataset/ETT-small", "path": "ETTh2.csv", "freq": "h", "target": "OT", "time_col": "date", "channels": 7},
    "ETTm1": {"root": "dataset/ETT-small", "path": "ETTm1.csv", "freq": "t", "target": "OT", "time_col": "date", "channels": 7},
    "ETTm2": {"root": "dataset/ETT-small", "path": "ETTm2.csv", "freq": "t", "target": "OT", "time_col": "date", "channels": 7},
    "weather": {"root": "dataset/weather", "path": "weather.csv", "freq": "t", "target": "OT", "time_col": "date", "channels": 21},
    "exchange_rate": {"root": "dataset/exchange_rate", "path": "exchange_rate.csv", "freq": "d", "target": "OT", "time_col": "date", "channels": 8},
    "national_illness": {"root": "dataset/illness", "path": "national_illness.csv", "freq": "d", "target": "OT", "time_col": "date", "channels": 7},
    "electricity": {"root": "dataset/electricity", "path": "electricity.csv", "freq": "h", "target": "OT", "time_col": "date", "channels": 321},
    "traffic": {"root": "dataset/traffic", "path": "traffic.csv", "freq": "h", "target": "OT", "time_col": "date", "channels": 862},
    # Optional dense-array benchmarks.  They are intentionally absent from
    # existing plans and are validated only when a new plan names them.  The
    # files are not downloaded by the runner; users must place the exact
    # source file at the registered path and record its hash in the plan.
    "solar": {
        "root": "dataset/solar", "path": "solar_AL.txt", "freq": "t",
        "target": "channel_136", "time_col": None, "channels": 137,
        "format": "txt", "data_key": None, "target_index": 136,
        "data_interval_minutes": 10,
        "source_hint": "Solar-Energy AL; header-free comma-delimited [time, channel] matrix",
    },
    "pems03": {
        "root": "dataset/PEMS03", "path": "PEMS03.npz", "freq": "t",
        "target": "channel_357", "time_col": None, "channels": 358,
        "format": "npz", "data_key": "data", "data_feature_index": 0,
        "target_index": 357,
        "data_interval_minutes": 5,
        "source_hint": "PEMS03; NPZ key data, shape [time, channel, feature] or [time, channel]",
    },
    "pems04": {
        "root": "dataset/PEMS04", "path": "PEMS04.npz", "freq": "t",
        "target": "channel_306", "channels": 307,
        "format": "npz", "data_key": "data", "data_feature_index": 0,
        "target_index": 306,
        "data_interval_minutes": 5, "time_col": None,
        "source_hint": "PEMS04; NPZ key data, shape [time, channel, feature] or [time, channel]",
    },
    "pems07": {
        "root": "dataset/PEMS07", "path": "PEMS07.npz", "freq": "t",
        "target": "channel_882", "time_col": None, "channels": 883,
        "format": "npz", "data_key": "data", "data_feature_index": 0,
        "target_index": 882,
        "data_interval_minutes": 5,
        "source_hint": "PEMS07; NPZ key data, shape [time, channel, feature] or [time, channel]",
    },
    "pems08": {
        "root": "dataset/PEMS08", "path": "PEMS08.npz", "freq": "t",
        "target": "channel_169", "time_col": None, "channels": 170,
        "format": "npz", "data_key": "data", "data_feature_index": 0,
        "target_index": 169,
        "data_interval_minutes": 5,
        "source_hint": "PEMS08; NPZ key data, shape [time, channel, feature] or [time, channel]",
    },
    "metr_la": {
        "root": "dataset/METR-LA", "path": "METR-LA.csv", "freq": "t",
        "target": "sensor_206", "time_col": None, "channels": 207,
        "format": "csv_numeric", "data_key": None, "target_index": 206,
        "data_interval_minutes": 5,
        "data_start_timestamp": "2012-03-01 00:00:00",
        "graph_path": "adjacency.npz",
        "graph_sha256": "97cfc5c63ab90bc6517108881f1aa3c26c62fbc548584b4ec76b28b0e16eac89",
        "source_hint": "METR-LA traffic-speed CSV; first timestamp column plus 207 sensor columns",
    },
    "pems_bay": {
        "root": "dataset/PEMS-BAY", "path": "PEMS-BAY.csv", "freq": "t",
        "target": "sensor_324", "time_col": None, "channels": 325,
        "format": "csv_numeric", "data_key": None, "target_index": 324,
        "data_interval_minutes": 5,
        "data_start_timestamp": "2017-01-01 00:00:00",
        "graph_path": "adjacency.npz",
        "graph_sha256": "3567957c59f994a678526725b810cf29031affe03dca99c72d0c4eba9e8761c7",
        "source_hint": "PEMS-BAY traffic-speed CSV; first timestamp column plus 325 sensor columns",
    },
}
MODEL_SPECS = {
    "DLinear": {"model": "DLinear", "autocon": False},
    "GraphDLinear": {"model": "GraphDLinear", "autocon": False},
    # Canonical STGCN-style graph-temporal baseline implemented with the
    # repository's safe edge-list contract and no PyG/DGL dependency.
    "STGCN": {"model": "STGCN", "autocon": False},
    # Graph WaveNet-style graph-temporal baseline with fixed and adaptive
    # graph supports, sharing the same hashed topology contract as STGCN.
    "GraphWaveNet": {"model": "GraphWaveNet", "autocon": False},
    # Canonical DCRNN-style diffusion recurrent baseline using the same
    # hashed traffic graph contract as the other graph comparators.
    "DCRNN": {"model": "DCRNN", "autocon": False},
    # Adaptive graph convolutional recurrent baseline.  Unlike the fixed
    # graph comparators it induces its support solely from learned nodes.
    "AGCRN": {"model": "AGCRN", "autocon": False},
    # NLinear is already implemented in ``models/NLinear.py`` and exposed by
    # the generic experiment.  Keep it as a first-class plain baseline so
    # extension plans cannot silently omit the normalized linear comparator.
    "NLinear": {"model": "NLinear", "autocon": False},
    # RLinear is a RevIN + linear-map comparator.  It remains plain by
    # design so AutoCon stays attributable to the Mamba candidate arm.
    "RLinear": {"model": "RLinear", "autocon": False},
    "PatchTST": {"model": "PatchTST", "autocon": False},
    "TimesNet": {"model": "TimesNet", "autocon": False},
    "TimeMixer": {"model": "TimeMixer", "autocon": False},
    # Inverted variable-token Transformer baseline.  It uses the generic
    # forecasting experiment and has no auxiliary AutoCon objective.
    "iTransformer": {"model": "iTransformer", "autocon": False},
    # Classical long-horizon references already shipped in the repository.
    # These are kept as explicit extension baselines so their results can be
    # paired with the Mamba rows without relying on legacy shell scripts.
    "Autoformer": {"model": "Autoformer", "autocon": False},
    "FEDformer": {"model": "FEDformer", "autocon": False},
    "ETSformer": {"model": "ETSformer", "autocon": False},
    "Transformer": {"model": "Transformer", "autocon": False},
    "Crossformer": {"model": "Crossformer", "autocon": False},
    "AutoConNet": {"model": "AutoConNet", "autocon": False},
    "AutoConNetAutoCon": {"model": "AutoConNet", "autocon": True},
    "Mamba1": {"model": "Mamba1", "autocon": False},
    "Mamba1AutoCon": {"model": "Mamba1AutoCon", "autocon": True},
    "Mamba2": {"model": "Mamba2", "autocon": False},
    "Mamba2AutoCon": {"model": "Mamba2AutoCon", "autocon": True},
    # MambaTS-inspired patch/variable-scan baseline and its AutoCon pair.
    "MambaTS": {"model": "MambaTS", "autocon": False},
    "MambaTSAutoCon": {"model": "MambaTSAutoCon", "autocon": True},
    # MambaTS v2 is a separate candidate so v1 checkpoints remain a stable
    # negative control.  The executable model and AutoCon contract are
    # registered explicitly in run.py.
    "MambaTSv2": {"model": "MambaTSv2", "autocon": False},
    "MambaTSv2AutoCon": {"model": "MambaTSv2AutoCon", "autocon": True},
    "MambaTSv3": {"model": "MambaTSv3", "autocon": False},
    "MambaTSv3AutoCon": {"model": "MambaTSv3AutoCon", "autocon": True},
    "MambaTSv4": {"model": "MambaTSv4", "autocon": False},
    "MambaTSv4AutoCon": {"model": "MambaTSv4AutoCon", "autocon": True},
    # MambaTS v5 adds the train-only causal signed ACF anchor while keeping
    # the v4 patch/shared-scan/AutoCon path as its exact architectural base.
    "MambaTSv5": {"model": "MambaTSv5", "autocon": False},
    "MambaTSv5AutoCon": {"model": "MambaTSv5AutoCon", "autocon": True},
    "MambaTSv6": {"model": "MambaTSv6", "autocon": False},
    "MambaTSv6AutoCon": {"model": "MambaTSv6AutoCon", "autocon": True},
    "MambaTSv7": {"model": "MambaTSv7", "autocon": False},
    "MambaTSv7AutoCon": {"model": "MambaTSv7AutoCon", "autocon": True},
    "MambaTSv8": {"model": "MambaTSv8", "autocon": False},
    "MambaTSv8AutoCon": {"model": "MambaTSv8AutoCon", "autocon": True},
    "MambaTSv9": {"model": "MambaTSv9", "autocon": False},
    "MambaTSv9AutoCon": {"model": "MambaTSv9AutoCon", "autocon": True},
    "MambaTSv10": {"model": "MambaTSv10", "autocon": False},
    "MambaTSv10AutoCon": {"model": "MambaTSv10AutoCon", "autocon": True},
    "MambaTSv11": {"model": "MambaTSv11", "autocon": False},
    "MambaTSv11AutoCon": {"model": "MambaTSv11AutoCon", "autocon": True},
    "MambaTSv12": {"model": "MambaTSv12", "autocon": False},
    "MambaTSv12AutoCon": {"model": "MambaTSv12AutoCon", "autocon": True},
    "MambaTSv13": {"model": "MambaTSv13", "autocon": False},
    "MambaTSv13AutoCon": {"model": "MambaTSv13AutoCon", "autocon": True},
    "MambaTSv14": {"model": "MambaTSv14", "autocon": False},
    "MambaTSv14AutoCon": {"model": "MambaTSv14AutoCon", "autocon": True},
    "MambaTSv15": {"model": "MambaTSv15", "autocon": False},
    "MambaTSv15AutoCon": {"model": "MambaTSv15AutoCon", "autocon": True},
    "MambaTSv16": {"model": "MambaTSv16", "autocon": False},
    "MambaTSv16AutoCon": {"model": "MambaTSv16AutoCon", "autocon": True},
    "LatentTemporalMamba": {"model": "LatentTemporalMamba", "autocon": False},
    "LatentTemporalMambaAutoCon": {"model": "LatentTemporalMambaAutoCon", "autocon": True},
    "InvertedRaMamba": {"model": "InvertedRaMamba", "autocon": False},
    "InvertedRaMambaAutoCon": {"model": "InvertedRaMambaAutoCon", "autocon": True},
    "LinearLatentMamba": {"model": "LinearLatentMamba", "autocon": False},
    "LinearLatentMambaAutoCon": {"model": "LinearLatentMambaAutoCon", "autocon": True},
    "NLinearVariableMamba": {"model": "NLinearVariableMamba", "autocon": False},
    "NLinearVariableMambaAutoCon": {"model": "NLinearVariableMambaAutoCon", "autocon": True},
    "MambaTSv6WarmAutoCon": {
        "model": "MambaTSv6AutoCon", "autocon": True,
        "config_overrides": {"mamba_ts_residual_init": 0.1},
    },
    # TimeMachine-style four-path Mamba baseline and its AutoCon companion.
    "MambaTM1": {"model": "MambaTM1", "autocon": False},
    "MambaTM1AutoCon": {"model": "MambaTM1AutoCon", "autocon": True},
    "MambaTM2": {"model": "MambaTM2", "autocon": False},
    "MambaTM2AutoCon": {"model": "MambaTM2AutoCon", "autocon": True},
    "MambaSM": {"model": "MambaSM", "autocon": False},
    "MambaSMAutoCon": {"model": "MambaSMAutoCon", "autocon": True},
    "RaMambaLite": {"model": "RaMambaLite", "autocon": False},
    "RaMambaLiteAutoCon": {"model": "RaMambaLiteAutoCon", "autocon": True},
    "BiPatchMamba": {"model": "BiPatchMamba", "autocon": False},
    "BiPatchMambaAutoCon": {"model": "BiPatchMambaAutoCon", "autocon": True},
    "MambaACF": {"model": "MambaACF", "autocon": False},
    # Runner-level arm alias for a paired variable-token screen.  The
    # executable model remains MambaACF; only these options differ from the
    # plain arm and are included in the per-job config/hash.
    "MambaACFVariable": {
        "model": "MambaACF", "autocon": False,
        "config_overrides": {
            "mamba_variable_mixer": True,
            "mamba_variable_group_size": 1,
        },
    },
    "MambaACFAutoCon": {"model": "MambaACFAutoCon", "autocon": True},
    # Paired variable-token AutoCon arm.  Keep the executable model name
    # unchanged so the experiment implementation remains a single source of
    # truth; the runner alias is persisted in the job/config provenance.
    "MambaACFVariableAutoCon": {
        "model": "MambaACFAutoCon", "autocon": True,
        "config_overrides": {
            "mamba_variable_mixer": True,
            "mamba_variable_group_size": 1,
        },
    },
    # A focused efficiency candidate: one Mamba-2 temporal scan, causal input
    # embedding, and factorized temporal/horizon projections.  The executable
    # model remains MambaACFAutoCon so AutoCon's train-only ACF relation path
    # and checkpoint contract stay shared with the reference arm.
    "MambaACFLiteAutoCon": {
        "model": "MambaACFAutoCon", "autocon": True,
        "config_overrides": {
            "mamba_scales": [1],
            "mamba_causal_embedding": True,
            "mamba_low_rank_head": True,
            "mamba_temporal_rank": 8,
            "mamba_variable_mixer": False,
            "mamba_patch_mixer": False,
            "mamba_local_mixer": False,
            "mamba_use_acf_memory": False,
        },
    },
    "MambaACF1": {"model": "MambaACF1", "autocon": False},
    "MambaACF1AutoCon": {"model": "MambaACF1AutoCon", "autocon": True},
}


# Frequencies are part of the dataset identity contract.  Keep the accepted
# values tied to the registry instead of accepting an arbitrary time-feature
# token that the loader may silently reinterpret.
VALID_DATASET_FREQUENCIES = frozenset(spec["freq"] for spec in DATASETS.values())

# These fields are consumed by ``run.py`` as integer budgets or dimensions.
# They are checked at the plan boundary so a malformed value cannot survive
# into command construction (where ``argparse`` would otherwise fail late).
PLAN_POSITIVE_INTEGER_FIELDS = (
    "seq_len", "label_len", "batch_size", "epochs", "train_epochs", "patience",
    "d_model", "d_ff", "e_layers", "n_heads", "d_layers", "factor", "top_k",
    "num_kernels", "moving_avg", "patch_len", "patch_stride",
    "timemixer_down_sampling_window", "timemixer_down_sampling_layers", "mamba_d_state",
    "mamba_d_conv", "mamba_expand", "mamba_headdim", "stream_sample_windows",
    "mamba_acf_local_min_pairs", "mamba_data_dependent_channel_mixer_rank",
    "mamba_variable_gdd_mixer_rank", "mamba_tm_n1", "mamba_tm_n2",
    "mamba_tm_repr_width", "mamba_ts_context_rank", "mamba_ts_spectral_top_k",
    "cpu_thread_limit",
)
PLAN_NONNEGATIVE_INTEGER_FIELDS = ("num_workers",)
PLAN_NONNEGATIVE_FLOAT_FIELDS = (
    "min_learning_rate", "weight_decay", "mamba_variable_gdd_init_scale",
)
PLAN_POSITIVE_FLOAT_FIELDS = ("learning_rate",)
PAIR_PHASE_METADATA_FIELDS = frozenset({"name", "purpose", "output_root"})
PAIRING_FIELDS = ("dataset", "horizon", "seed")
BASELINE_CLAIM_MODELS = frozenset({
    "DLinear", "GraphDLinear", "STGCN", "GraphWaveNet", "DCRNN", "AGCRN", "NLinear", "RLinear", "PatchTST", "TimesNet", "TimeMixer", "iTransformer",
    "Autoformer", "FEDformer", "ETSformer", "Transformer", "Crossformer",
    "Mamba1", "Mamba2", "MambaTS", "MambaTSv2", "MambaTSv3", "MambaTSv4", "MambaTSv5", "MambaTSv6", "MambaTSv7", "MambaTSv8", "MambaTSv9", "MambaTSv10", "MambaTSv11", "MambaTSv12", "MambaTSv13", "MambaTSv14", "MambaTSv15", "MambaTSv16", "MambaTM1", "MambaTM2", "MambaSM", "RaMambaLite", "BiPatchMamba", "LatentTemporalMamba", "InvertedRaMamba", "LinearLatentMamba", "NLinearVariableMamba",
})


def _dataset_header(dataset: str) -> list[str]:
    spec = DATASETS[dataset]
    path = REPO_ROOT / spec["root"] / spec["path"]
    if str(spec.get("format", "csv")).lower() not in {"csv", "csv_numeric"}:
        raise ValueError(
            "dataset {!r} uses format {!r}, so it has no CSV header".format(
                dataset, spec.get("format")
            )
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def _dataset_path(dataset: str) -> Path:
    spec = DATASETS[dataset]
    return REPO_ROOT / spec["root"] / spec["path"]


def _dataset_format(spec: dict) -> str:
    value = str(spec.get("format", "csv")).strip().lower()
    return value or "csv"


def _numeric_array_schema(path: Path, spec: dict) -> dict:
    """Inspect an optional numeric-array file through the shared loader."""

    from data_provider.data_loader import inspect_numeric_array_file

    return inspect_numeric_array_file(
        path,
        data_format=_dataset_format(spec),
        data_key=spec.get("data_key", "data"),
        data_feature_index=spec.get("data_feature_index", 0),
    )


def _validate_numeric_array_contract(dataset: str, contract: dict) -> list[str]:
    """Validate a lightweight contract for an optional NPZ/TXT dataset."""

    errors: list[str] = []
    if dataset not in DATASETS:
        return ["unknown dataset"]
    spec = DATASETS[dataset]
    expected_format = _dataset_format(spec)
    if expected_format == "csv":
        return ["numeric-array contract used for a CSV dataset"]
    if not isinstance(contract, dict):
        return ["contract entry must be an object"]
    relative_path = contract.get("relative_path")
    expected_path = Path(spec["root"]) / spec["path"]
    if Path(str(relative_path)) != expected_path:
        errors.append("relative_path does not match registered dataset path")
    if contract.get("format") != expected_format:
        errors.append("format must be {!r}".format(expected_format))
    if expected_format == "npz" and contract.get("data_key", spec.get("data_key", "data")) != spec.get("data_key", "data"):
        errors.append("data_key does not match the registered NPZ key")
    if contract.get("frequency") != spec["freq"]:
        errors.append("frequency does not match the registered frequency")
    if contract.get("time_column") not in (None, ""):
        errors.append("numeric-array datasets must not declare time_column")
    if contract.get("target_index", spec.get("target_index", -1)) != spec.get("target_index", -1):
        errors.append("target_index does not match the registered target index")
    if contract.get("data_interval_minutes", spec.get("data_interval_minutes")) != spec.get("data_interval_minutes"):
        errors.append("data_interval_minutes does not match the registered sampling interval")
    if contract.get("data_feature_index", spec.get("data_feature_index", 0)) != spec.get("data_feature_index", 0):
        errors.append("data_feature_index does not match the registered feature axis")
    expected_channels = contract.get("channels_including_target")
    if expected_channels is None:
        errors.append("channels_including_target is required")
    elif expected_channels != spec["channels"]:
        errors.append("channels_including_target disagrees with the registry")
    path = _dataset_path(dataset)
    if not path.is_file():
        hint = spec.get("source_hint")
        suffix = "; place the documented file ({})".format(hint) if hint else ""
        errors.append("missing dataset file {}{}".format(path, suffix))
        return errors
    expected_hash = contract.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        errors.append("sha256 must be a 64-character digest")
    else:
        try:
            actual_hash = _sha256(path)
        except OSError as exc:
            errors.append("dataset unreadable: {}".format(exc))
        else:
            if expected_hash.lower() != actual_hash:
                errors.append(
                    "sha256 mismatch (expected {}, got {})".format(
                        expected_hash, actual_hash
                    )
                )
    try:
        schema = _numeric_array_schema(path, spec)
    except (OSError, TypeError, ValueError) as exc:
        errors.append("invalid {} numeric dataset: {}".format(expected_format, exc))
    else:
        shape = schema.get("shape", [])
        actual_channels = int(shape[-1]) if len(shape) == 2 else -1
        if actual_channels != int(spec["channels"]):
            errors.append(
                "array channel count mismatch (expected {}, got {})".format(
                    spec["channels"], actual_channels
                )
            )
    return errors


def _feature_dimensions(dataset: str, features: str) -> tuple[int, int, int]:
    """Return (enc_in, dec_in, c_out) from the registered dataset schema."""
    if features not in {"S", "M", "MS"}:
        raise ValueError(f"unsupported features value: {features!r}")
    channels = int(DATASETS[dataset]["channels"])
    if features == "S":
        return 1, 1, 1
    if features == "M":
        return channels, channels, channels
    return channels, channels, 1


def _validate_data_fingerprints(fingerprints: dict) -> list[str]:
    """Validate lightweight dataset identity contracts used by screening plans.

    Screening plans may cover datasets whose legacy quality reports intentionally
    fail a strict cadence check (for example Weather).  They still need a hard
    identity gate so a substituted CSV cannot silently enter the matrix.
    """

    errors: list[str] = []
    if not isinstance(fingerprints, dict) or not fingerprints:
        return ["data_fingerprints must be a non-empty object"]
    for dataset, contract in fingerprints.items():
        if dataset not in DATASETS:
            errors.append(f"data_fingerprints[{dataset}]: unknown dataset")
            continue
        if not isinstance(contract, dict):
            errors.append(f"data_fingerprints[{dataset}]: entry must be an object")
            continue
        registered = DATASETS[dataset]
        registered_format = _dataset_format(registered)
        declared_format = contract.get("format")
        # ``csv_numeric`` is still a header-bearing CSV.  It differs from the
        # timestamp-aware loader only in how dates are synthesized, so it
        # must use the CSV identity and graph-order checks below rather than
        # the dense-array branch.
        if registered_format not in {"csv", "csv_numeric"}:
            if declared_format != registered_format:
                errors.append(
                    f"data_fingerprints[{dataset}]: format must be {registered_format!r}"
                )
            if registered_format == "npz":
                expected_key = registered.get("data_key", "data")
                if contract.get("data_key", expected_key) != expected_key:
                    errors.append(
                        f"data_fingerprints[{dataset}]: data_key must be {expected_key!r}"
                    )
        elif declared_format not in (None, "csv", "csv_numeric"):
            errors.append(
                f"data_fingerprints[{dataset}]: CSV dataset cannot declare format {declared_format!r}"
            )
        frequency = contract.get("frequency")
        if not isinstance(frequency, str) or not frequency.strip():
            errors.append(
                f"data_fingerprints[{dataset}]: frequency is required and must be a non-empty string"
            )
        elif frequency != frequency.strip() or frequency.lower() != frequency:
            errors.append(
                f"data_fingerprints[{dataset}]: frequency must be lowercase without surrounding whitespace"
            )
        elif frequency not in VALID_DATASET_FREQUENCIES:
            errors.append(
                f"data_fingerprints[{dataset}]: unsupported frequency {frequency!r}"
            )
        elif frequency != registered["freq"]:
            errors.append(
                f"data_fingerprints[{dataset}]: frequency mismatch (expected {registered['freq']!r}, got {frequency!r})"
            )
        relative_path = contract.get("relative_path")
        expected_path = Path(registered["root"]) / registered["path"]
        if relative_path is not None and Path(str(relative_path)) != expected_path:
            errors.append(
                f"data_fingerprints[{dataset}]: relative_path does not match registered dataset path"
            )
        path = REPO_ROOT / expected_path
        if not path.is_file():
            errors.append(f"data_fingerprints[{dataset}]: missing dataset file {path}")
            continue
        expected_hash = contract.get("sha256")
        if expected_hash is not None:
            try:
                actual_hash = _sha256(path)
            except OSError as exc:
                errors.append(f"data_fingerprints[{dataset}]: unreadable dataset: {exc}")
                continue
            if str(expected_hash).lower() != actual_hash:
                errors.append(
                    f"data_fingerprints[{dataset}]: sha256 mismatch (expected {expected_hash}, got {actual_hash})"
                )
        if registered_format not in {"csv", "csv_numeric"}:
            try:
                schema = _numeric_array_schema(path, registered)
            except (OSError, TypeError, ValueError) as exc:
                errors.append(
                    f"data_fingerprints[{dataset}]: invalid {registered_format} numeric dataset: {exc}"
                )
                continue
            expected_channels = contract.get("channels_including_target")
            shape = schema.get("shape", [])
            actual_channels = int(shape[-1]) if len(shape) == 2 else -1
            if expected_channels is not None and actual_channels != int(expected_channels):
                errors.append(
                    f"data_fingerprints[{dataset}]: channel count mismatch (expected {expected_channels}, got {actual_channels})"
                )
            if actual_channels != int(registered["channels"]):
                errors.append(
                    f"data_fingerprints[{dataset}]: registered channel count mismatch (expected {registered['channels']}, got {actual_channels})"
                )
            if contract.get("time_column") not in (None, ""):
                errors.append(
                    f"data_fingerprints[{dataset}]: numeric-array datasets must not declare a time_column"
                )
            expected_interval = registered.get("data_interval_minutes")
            if contract.get("data_interval_minutes", expected_interval) != expected_interval:
                errors.append(
                    f"data_fingerprints[{dataset}]: data_interval_minutes must be {expected_interval!r}"
                )
            expected_target_index = registered.get("target_index")
            if (
                contract.get("target_index", expected_target_index)
                != expected_target_index
            ):
                errors.append(
                    f"data_fingerprints[{dataset}]: target_index must be {expected_target_index!r}"
                )
            expected_feature_index = registered.get("data_feature_index", 0)
            if contract.get("data_feature_index", expected_feature_index) != expected_feature_index:
                errors.append(
                    f"data_fingerprints[{dataset}]: data_feature_index must be {expected_feature_index!r}"
                )
            continue
        try:
            header = _dataset_header(dataset)
        except (OSError, StopIteration, csv.Error) as exc:
            errors.append(f"data_fingerprints[{dataset}]: unreadable header: {exc}")
            continue
        expected_channels = contract.get("channels_including_target")
        if expected_channels is not None and len(header) - 1 != int(expected_channels):
            errors.append(
                f"data_fingerprints[{dataset}]: channel count mismatch (expected {expected_channels}, got {len(header) - 1})"
            )
        expected_time = contract.get("time_column")
        if expected_time is not None and (not header or header[0] != expected_time):
            errors.append(
                f"data_fingerprints[{dataset}]: time column mismatch (expected {expected_time!r})"
            )
        expected_target = contract.get("target")
        if expected_target is not None and expected_target not in header:
            errors.append(
                f"data_fingerprints[{dataset}]: target column {expected_target!r} is missing"
            )
        graph_fields = {
            "adjacency_relative_path", "adjacency_sha256", "directed_edge_count",
        }
        declared_graph_fields = graph_fields.intersection(contract)
        if not declared_graph_fields:
            continue
        if not {"adjacency_relative_path", "adjacency_sha256"}.issubset(
            declared_graph_fields
        ):
            errors.append(
                f"data_fingerprints[{dataset}]: adjacency path and sha256 must be declared together"
            )
            continue
        expected_graph_name = registered.get("graph_path")
        expected_graph_hash = registered.get("graph_sha256")
        if not expected_graph_name or not expected_graph_hash:
            errors.append(
                f"data_fingerprints[{dataset}]: dataset has no registered graph contract"
            )
            continue
        expected_graph_path = Path(registered["root"]) / expected_graph_name
        declared_graph_path = Path(str(contract["adjacency_relative_path"]))
        if declared_graph_path != expected_graph_path:
            errors.append(
                f"data_fingerprints[{dataset}]: adjacency_relative_path does not match registered graph path"
            )
            continue
        graph_path = REPO_ROOT / expected_graph_path
        if not graph_path.is_file():
            errors.append(f"data_fingerprints[{dataset}]: missing graph contract {graph_path}")
            continue
        declared_graph_hash = str(contract["adjacency_sha256"]).lower()
        if declared_graph_hash != str(expected_graph_hash).lower():
            errors.append(
                f"data_fingerprints[{dataset}]: adjacency_sha256 does not match registered graph hash"
            )
            continue
        actual_graph_hash = _sha256(graph_path)
        if actual_graph_hash != declared_graph_hash:
            errors.append(
                f"data_fingerprints[{dataset}]: adjacency sha256 mismatch (expected {declared_graph_hash}, got {actual_graph_hash})"
            )
            continue
        try:
            with np.load(str(graph_path), allow_pickle=False) as graph:
                required = {
                    "schema_version", "sensor_ids", "target", "source",
                    "weight", "node_count",
                }
                missing = sorted(required.difference(graph.files))
                if missing:
                    errors.append(
                        f"data_fingerprints[{dataset}]: graph contract is missing fields {', '.join(missing)}"
                    )
                    continue
                schema_version = np.asarray(graph["schema_version"])
                node_count = np.asarray(graph["node_count"])
                sensor_ids = np.asarray(graph["sensor_ids"]).astype(str).tolist()
                target = np.asarray(graph["target"])
                source = np.asarray(graph["source"])
                weight = np.asarray(graph["weight"])
        except (OSError, ValueError) as exc:
            errors.append(f"data_fingerprints[{dataset}]: unreadable graph contract: {exc}")
            continue
        expected_channels = int(registered["channels"])
        if schema_version.ndim != 0 or int(schema_version.item()) != 1:
            errors.append(f"data_fingerprints[{dataset}]: unsupported graph schema")
        if node_count.ndim != 0 or int(node_count.item()) != expected_channels:
            errors.append(f"data_fingerprints[{dataset}]: graph node count mismatch")
        if sensor_ids != [str(value) for value in header[1:]]:
            errors.append(f"data_fingerprints[{dataset}]: graph sensor IDs do not match CSV column order")
        if (
            target.ndim != 1 or source.ndim != 1 or weight.ndim != 1
            or target.size == 0 or target.size != source.size or target.size != weight.size
        ):
            errors.append(f"data_fingerprints[{dataset}]: graph edge vectors are invalid")
            continue
        declared_edge_count = contract.get("directed_edge_count")
        if declared_edge_count is not None and int(declared_edge_count) != int(target.size):
            errors.append(
                f"data_fingerprints[{dataset}]: directed_edge_count mismatch (expected {declared_edge_count}, got {target.size})"
            )
    return errors


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _hash_file_map(root: Path) -> tuple[dict[str, str], str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
            files[path.relative_to(root).as_posix()] = _sha256(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return files, hashlib.sha256(encoded).hexdigest()


def _artifact_hashes(result_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, str]:
    paths = [result_dir / "config.json", result_dir / "metrics.npy",
             result_dir / "original_metrics.npy", result_dir / "channel_metrics.npy",
             result_dir / "stream_metrics.json", result_dir / "stream_samples.npz",
             result_dir / "pred.npy", result_dir / "true.npy",
             result_dir / "training_complete.json", result_dir / "acf_prior.npz"]
    if checkpoint_dir is not None:
        paths.extend([checkpoint_dir / "checkpoint.pth", checkpoint_dir / "loss_logs.json"])
    return {path.name: _sha256(path) for path in paths if path.is_file()}


def _command_hash(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _is_confirmation_plan(plan: dict) -> bool:
    """Identify the strict boundary from the presence of a confirmation contract.

    A partial confirmation plan must remain strict and fail validation; it must
    never silently fall back to the exploratory runner path merely because a
    statistical-gate field is missing.
    """

    return isinstance(plan.get("confirmation_contract"), dict)


def _phase_has_autocon_model(phase: dict) -> bool:
    """Return whether a phase actually expands at least one AutoCon arm."""

    return any(
        isinstance(MODEL_SPECS.get(model), dict)
        and MODEL_SPECS[model].get("autocon") is True
        for model in phase.get("models", [])
    )


def _plan_is_exploratory(plan: dict) -> bool:
    """Require an explicit exploratory/screen status for null-control plans."""

    status = plan.get("status")
    if not isinstance(status, str):
        return False
    normalized = status.strip().lower()
    return "exploratory" in normalized or "screen" in normalized


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _required_prelaunch_gates(plan: dict, execution: dict) -> list[str]:
    """Return the distinct human-review gates that must have hash evidence."""

    gates: list[str] = []
    for field, value in (
        ("execution.required_before_launch", execution.get("required_before_launch")),
        ("prelaunch_gates", plan.get("prelaunch_gates")),
    ):
        if value is None:
            continue
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list when present")
        for gate in value:
            if not isinstance(gate, str) or not gate.strip():
                raise ValueError(f"{field} must contain non-empty strings")
            normalized = gate.strip()
            if normalized not in gates:
                gates.append(normalized)
    return gates


def _normalize_gate_evidence(sidecar: dict, required_gates: list[str]) -> dict[str, str]:
    """Check that a sidecar records a SHA256 for every declared launch gate."""

    evidence = sidecar.get("prelaunch_gate_evidence_hashes")
    if not isinstance(evidence, dict):
        raise ValueError("launch authorization sidecar requires prelaunch_gate_evidence_hashes")
    normalized: dict[str, str] = {}
    for gate, value in evidence.items():
        if not isinstance(gate, str) or not gate.strip():
            raise ValueError("launch authorization evidence keys must be non-empty strings")
        digest = value.get("sha256") if isinstance(value, dict) else value
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"launch authorization evidence for {gate!r} must be a SHA256")
        digest = digest.lower()
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"launch authorization evidence for {gate!r} must be a SHA256")
        normalized[gate.strip()] = digest
    missing = [gate for gate in required_gates if gate not in normalized]
    if missing:
        raise ValueError(
            "launch authorization sidecar lacks evidence hashes for required gates: "
            + "; ".join(missing)
        )
    return normalized


def _approval_record(sidecar: dict) -> dict[str, str | bool]:
    """Normalize the sidecar's human approval fields and require a UTC timestamp."""

    approval = sidecar.get("approval", sidecar)
    if not isinstance(approval, dict):
        raise ValueError("launch authorization sidecar approval must be an object")
    approved = approval.get("approved", sidecar.get("authorized"))
    if approved is not True:
        raise ValueError("launch authorization sidecar requires approval.approved=true")
    approved_by_value = approval.get("approved_by", sidecar.get("approved_by"))
    approved_at_value = approval.get("approved_at_utc", sidecar.get("approved_at_utc"))
    if not isinstance(approved_by_value, str) or not approved_by_value.strip():
        raise ValueError("launch authorization sidecar requires approved_by")
    if not isinstance(approved_at_value, str) or not approved_at_value.strip():
        raise ValueError("launch authorization sidecar requires approved_at_utc")
    approved_by = approved_by_value.strip()
    approved_at_utc = approved_at_value.strip()
    try:
        parsed = dt.datetime.fromisoformat(approved_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("launch authorization approved_at_utc must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("launch authorization approved_at_utc must include a UTC offset")
    if parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("launch authorization approved_at_utc must be UTC")
    return {
        "approved": True,
        "approved_by": approved_by,
        "approved_at_utc": approved_at_utc,
    }


def _validate_launch_authorization(
    plan: dict,
    plan_path: Path,
    sidecar_path: Path | None = None,
    *,
    enforce: bool = True,
) -> dict | None:
    """Validate the auditable launch gate for plans that declare ``execution``.

    Legacy plans without an execution object retain their historical runner
    behavior.  ``validate_only`` and ``dry_run`` callers may inspect a draft,
    but a real run must bind the exact plan hash to an approved sidecar and
    SHA256 evidence for every prelaunch gate declared by the plan.
    """

    strict_confirmation = _is_confirmation_plan(plan)
    if "execution" not in plan:
        if strict_confirmation:
            raise ValueError("confirmation plan requires an execution object")
        return None
    if strict_confirmation and not isinstance(plan.get("statistical_gate"), dict):
        raise ValueError("confirmation plan requires a statistical_gate object")
    execution = plan.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("plan execution must be an object")
    authorized = execution.get("authorized")
    if not isinstance(authorized, bool):
        raise ValueError("plan execution.authorized must be boolean")
    execution_status = execution.get("status")
    plan_status = plan.get("status")
    if strict_confirmation:
        if not isinstance(plan_status, str) or not plan_status.strip():
            raise ValueError("confirmation plan status must be a non-empty string")
        if plan_status != plan_status.strip():
            raise ValueError("confirmation plan status must not have surrounding whitespace")
        if not isinstance(execution_status, str) or not execution_status.strip():
            raise ValueError("confirmation plan execution.status must be a non-empty string")
        if execution_status != execution_status.strip():
            raise ValueError(
                "confirmation plan execution.status must not have surrounding whitespace"
            )
        if execution_status != plan_status:
            raise ValueError("plan execution.status disagrees with top-level status")
        status = plan_status
    else:
        if plan_status is not None and not isinstance(plan_status, str):
            raise ValueError("plan status must be a string")
        if execution_status is not None and (
            not isinstance(execution_status, str) or not execution_status.strip()
        ):
            raise ValueError("plan execution.status must be a non-empty string when present")
        if execution_status is not None and plan_status is not None and execution_status.strip() != plan_status.strip():
            raise ValueError("plan execution.status disagrees with top-level status")
        if plan_status is not None and plan_status.strip():
            status = plan_status.strip()
        elif execution_status is not None:
            status = execution_status.strip()
        else:
            # Exploratory plans may declare execution metadata before a formal
            # status vocabulary is frozen.  Preserve a deterministic binding
            # value while keeping real launches blocked by ``authorized``.
            status = "exploratory"
    required_gates = _required_prelaunch_gates(plan, execution)
    plan_path = plan_path.expanduser().resolve()
    binding = {
        "schema_version": 1,
        "kind": LAUNCH_AUTHORIZATION_KIND,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "plan_status": status,
        "authorized": authorized,
        "required_prelaunch_gates": required_gates,
        "enforced": bool(enforce),
    }
    if not enforce:
        return binding
    if authorized is not True:
        raise ValueError(
            "plan is not authorized: execution.authorized must be true "
            f"(status={status!r})"
        )
    if strict_confirmation and status not in LAUNCH_AUTHORIZED_STATUSES:
        raise ValueError(
            "confirmation plan status is not launch-approved: "
            f"expected one of {sorted(LAUNCH_AUTHORIZED_STATUSES)}, got {status!r}"
        )
    if sidecar_path is None:
        raise ValueError(
            "authorized plan launch requires --launch_sidecar with plan hash, approval, and gate evidence"
        )
    sidecar_path = _resolve_repo_path(sidecar_path)
    if not sidecar_path.is_file():
        raise ValueError(f"launch authorization sidecar is missing: {sidecar_path}")
    try:
        sidecar = _load_json(sidecar_path)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"launch authorization sidecar is invalid: {exc}") from exc
    if not isinstance(sidecar, dict):
        raise ValueError("launch authorization sidecar must be a JSON object")
    if sidecar.get("schema_version") != 1 or sidecar.get("kind") not in LAUNCH_AUTHORIZATION_KINDS:
        raise ValueError("launch authorization sidecar has an unsupported schema/kind")
    sidecar_approval = sidecar.get("approval")
    if sidecar.get("authorized") is False or (
        sidecar.get("authorized") is not True
        and not (isinstance(sidecar_approval, dict) and sidecar_approval.get("approved") is True)
    ):
        raise ValueError("launch authorization sidecar is not approved")
    declared_plan = sidecar.get("plan_path", sidecar.get("plan"))
    if declared_plan is None or _resolve_repo_path(str(declared_plan)) != plan_path:
        raise ValueError("launch authorization sidecar plan_path does not match --plan")
    if str(sidecar.get("plan_sha256", "")).lower() != binding["plan_sha256"]:
        raise ValueError("launch authorization sidecar plan_sha256 does not match plan")
    declared_status = sidecar.get("plan_status", sidecar.get("status"))
    if declared_status != status:
        raise ValueError("launch authorization sidecar status does not match plan")
    approval = _approval_record(sidecar)
    evidence = _normalize_gate_evidence(sidecar, required_gates)
    binding.update({
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "sidecar_kind": sidecar.get("kind"),
        "approval": approval,
        "prelaunch_gate_evidence_hashes": evidence,
    })
    return binding


def _validate_confirmation_launch_binding(
    plan: dict,
    plan_path: Path,
    provenance: dict,
    phase_root: Path,
) -> dict | None:
    """Validate the immutable launch binding archived in a confirmation phase.

    Training provenance is the authority for a completed phase, but report
    consumers must independently verify that it contains the *same* approved
    launch decision that the current plan describes.  This helper deliberately
    returns ``None`` for exploratory/legacy plans without a confirmation
    contract; a plan that declares that contract cannot opt out of the binding
    by omitting ``execution`` or the archived sidecar.
    """

    if not isinstance(plan.get("confirmation_contract"), dict):
        return None
    if not isinstance(provenance, dict):
        raise ValueError("confirmation provenance must be an object")
    if not isinstance(plan.get("statistical_gate"), dict):
        raise ValueError("confirmation plan requires a statistical_gate object")

    plan_path = plan_path.expanduser().resolve()
    phase_root = phase_root.expanduser().resolve()
    # ``enforce=False`` validates the shape and current plan hash while this
    # function applies the stronger post-launch approval and archive checks.
    plan_binding = _validate_launch_authorization(
        plan, plan_path, enforce=False
    )
    if not isinstance(plan_binding, dict):
        raise ValueError("confirmation plan requires an execution object")
    if plan_binding.get("authorized") is not True:
        raise ValueError("confirmation plan execution.authorized is not true")
    if plan_binding.get("plan_status") not in LAUNCH_AUTHORIZED_STATUSES:
        raise ValueError("confirmation plan status is not launch-approved")

    launch = provenance.get("launch_authorization")
    if not isinstance(launch, dict):
        raise ValueError("confirmation provenance has no launch authorization binding")
    # These fields are derived from the current plan and cannot be replaced by
    # values copied from an unrelated sidecar.
    for key in (
        "schema_version", "kind", "plan_path", "plan_sha256", "plan_status",
        "authorized", "required_prelaunch_gates",
    ):
        if launch.get(key) != plan_binding.get(key):
            raise ValueError(f"confirmation launch binding {key} disagrees with plan")
    if launch.get("enforced") is not True:
        raise ValueError("confirmation launch binding was not enforced")
    binding_hash = provenance.get("launch_authorization_binding_sha256")
    if binding_hash != _canonical_hash(launch):
        raise ValueError("confirmation launch authorization binding hash mismatch")

    copy_value = provenance.get("launch_authorization_copy")
    copy_hash = provenance.get("launch_authorization_copy_sha256")
    if not isinstance(copy_value, str) or not copy_value.strip():
        raise ValueError("confirmation provenance lacks archived launch sidecar")
    copy_path = Path(copy_value).expanduser().resolve()
    if not copy_path.is_file():
        raise ValueError("archived confirmation launch sidecar is missing")
    try:
        copy_path.relative_to(phase_root)
    except ValueError as exc:
        raise ValueError("archived confirmation launch sidecar is outside phase root") from exc
    actual_copy_hash = _sha256(copy_path)
    if not isinstance(copy_hash, str) or actual_copy_hash != copy_hash:
        raise ValueError("archived confirmation launch sidecar hash mismatch")
    if launch.get("sidecar_sha256") != actual_copy_hash:
        raise ValueError("archived sidecar hash differs from launch binding")

    archived = _load_json(copy_path)
    if not isinstance(archived, dict):
        raise ValueError("archived launch sidecar must be an object")
    if archived.get("schema_version") != 1 or archived.get("kind") not in LAUNCH_AUTHORIZATION_KINDS:
        raise ValueError("archived launch sidecar has an unsupported schema/kind")
    if launch.get("sidecar_kind") != archived.get("kind"):
        raise ValueError("archived sidecar kind differs from launch binding")
    declared_plan = archived.get("plan_path", archived.get("plan"))
    if declared_plan is None or _resolve_repo_path(str(declared_plan)) != plan_path:
        raise ValueError("archived sidecar plan_path does not match plan")
    if str(archived.get("plan_sha256", "")).lower() != plan_binding["plan_sha256"]:
        raise ValueError("archived sidecar plan hash does not match plan")
    if archived.get("plan_status", archived.get("status")) != plan_binding["plan_status"]:
        raise ValueError("archived sidecar status does not match plan")
    archived_approval = archived.get("approval")
    if archived.get("authorized") is False or (
        archived.get("authorized") is not True
        and not (isinstance(archived_approval, dict) and archived_approval.get("approved") is True)
    ):
        raise ValueError("archived launch sidecar is not approved")
    approval = _approval_record(archived)
    bound_approval = launch.get("approval")
    if not isinstance(bound_approval, dict) or approval != bound_approval:
        raise ValueError("archived sidecar approval differs from launch binding")
    evidence = _normalize_gate_evidence(
        archived, plan_binding.get("required_prelaunch_gates", [])
    )
    if evidence != launch.get("prelaunch_gate_evidence_hashes"):
        raise ValueError("archived sidecar gate evidence differs from launch binding")

    return {
        "launch_authorization": launch,
        "launch_authorization_binding_sha256": binding_hash,
        "launch_authorization_copy": str(copy_path),
        "launch_authorization_copy_sha256": actual_copy_hash,
    }


def _git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # pragma: no cover - diagnostics only
        return repr(exc)


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            if path.is_dir():
                path.chmod(0o555)
            else:
                path.chmod(0o444)
        except OSError:
            pass
    try:
        root.chmod(0o555)
    except OSError:
        pass


def _copy_source_snapshot(destination: Path) -> tuple[dict[str, str], str]:
    """Copy the uncommitted experiment source used by child jobs.

    A copy is deliberately used instead of ``git archive`` because the Mamba
    implementation and experiment changes are currently uncommitted.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"source snapshot already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix="source_snapshot.", dir=str(destination.parent)))
    try:
        for relative in SOURCE_SNAPSHOT_PATHS:
            source = REPO_ROOT / relative
            if not source.exists():
                raise FileNotFoundError(f"source snapshot input missing: {source}")
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                )
            else:
                shutil.copy2(source, target)
        files, tree_hash = _hash_file_map(temporary)
        _make_read_only(temporary)
        os.replace(temporary, destination)
        return files, tree_hash
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _phase_provenance(
    phase_root: Path,
    phase: dict,
    plan_path: Path,
    selection_path: Path | None,
    selection_binding: dict | None = None,
    launch_authorization: dict | None = None,
) -> dict:
    """Create or validate an immutable provenance record for a phase."""
    phase_root.mkdir(parents=True, exist_ok=True)
    provenance_path = phase_root / "provenance.json"
    plan_hash = _sha256(plan_path)
    selection_hash = _sha256(selection_path) if selection_path is not None else None
    selection_provenance_hash = (
        selection_binding.get("selection_provenance_sha256") if selection_binding is not None else None
    )
    selection_binding_hash = _canonical_hash(selection_binding) if selection_binding is not None else None
    dataset_hashes = {
        name: _sha256(REPO_ROOT / DATASETS[name]["root"] / DATASETS[name]["path"])
        for name in phase.get("datasets", [])
    }

    if provenance_path.exists():
        provenance = _load_json(provenance_path)
        expected = {
            "phase": phase["name"],
            "plan_sha256": plan_hash,
            "selection_sha256": selection_hash,
            "selection_provenance_sha256": selection_provenance_hash,
            "selection_binding_sha256": selection_binding_hash,
            "dataset_hashes": dataset_hashes,
        }
        # The launch binding is immutable for strict confirmation runs.  Keep
        # legacy exploratory provenance resumable when it predates this field.
        if launch_authorization is not None:
            expected["launch_authorization"] = launch_authorization
            expected["launch_authorization_copy"] = str(
                (phase_root / "provenance" / "launch_authorization.json").resolve()
            )
            expected["launch_authorization_copy_sha256"] = launch_authorization.get(
                "sidecar_sha256"
            )
            expected["launch_authorization_binding_sha256"] = _canonical_hash(
                launch_authorization
            )
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise RuntimeError(
                    f"immutable provenance mismatch for {phase['name']}: {key}; "
                    "use a new output_root instead of reusing this phase"
                )
        snapshot = Path(provenance.get("source_snapshot", ""))
        if not snapshot.is_dir():
            raise RuntimeError(f"source snapshot is missing: {snapshot}")
        files, tree_hash = _hash_file_map(snapshot)
        if tree_hash != provenance.get("source_tree_sha256") or files != provenance.get("source_files"):
            raise RuntimeError("source snapshot changed; refusing to resume phase")
        if launch_authorization is not None:
            copy_path = Path(str(provenance.get("launch_authorization_copy", ""))).expanduser().resolve()
            if not copy_path.is_file() or _sha256(copy_path) != provenance.get(
                "launch_authorization_copy_sha256"
            ):
                raise RuntimeError("launch authorization sidecar copy is missing or changed")
        return provenance

    existing = [item for item in phase_root.iterdir() if item.name not in {"provenance.json"}]
    if existing:
        raise RuntimeError(
            f"phase output already contains artifacts but no provenance: {phase_root}; "
            "move the legacy output aside and use a fresh phase root"
        )

    provenance_dir = phase_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=False)
    snapshot = provenance_dir / "source_snapshot"
    source_files, source_tree_hash = _copy_source_snapshot(snapshot)
    plan_copy = provenance_dir / "final_training_plan.json"
    shutil.copy2(plan_path, plan_copy)
    selection_copy = None
    selection_provenance_copy = None
    launch_authorization_copy = None
    launch_authorization_copy_hash = None
    if launch_authorization is not None:
        source_launch_sidecar = Path(launch_authorization["sidecar_path"])
        launch_authorization_copy = provenance_dir / "launch_authorization.json"
        shutil.copy2(source_launch_sidecar, launch_authorization_copy)
        launch_authorization_copy_hash = _sha256(launch_authorization_copy)
        if launch_authorization_copy_hash != launch_authorization.get("sidecar_sha256"):
            raise RuntimeError("launch authorization sidecar changed while creating provenance")
        launch_authorization_copy.chmod(0o444)
    if selection_path is not None:
        selection_copy = provenance_dir / "selected_hyperparameters.json"
        shutil.copy2(selection_path, selection_copy)
        if selection_binding is None:
            raise RuntimeError("selected phase is missing verified selection provenance")
        source_sidecar = Path(selection_binding["selection_provenance_path"])
        selection_provenance_copy = provenance_dir / "selected_hyperparameters.provenance.json"
        shutil.copy2(source_sidecar, selection_provenance_copy)
        if _sha256(selection_copy) != selection_hash:
            raise RuntimeError("selection artifact changed while creating provenance")
        if _sha256(selection_provenance_copy) != selection_provenance_hash:
            raise RuntimeError("selection provenance sidecar changed while creating provenance")
    package_freeze = _git_text([sys.executable, "-m", "pip", "freeze"])
    (provenance_dir / "pip-freeze.txt").write_text(package_freeze, encoding="utf-8")
    git_diff = _git_text(["git", "diff", "--no-ext-diff", "--binary", "HEAD"])
    (provenance_dir / "git-diff.patch").write_text(git_diff, encoding="utf-8")
    payload = {
        "schema": 1,
        "provenance_id": hashlib.sha256(
            json.dumps(
                {
                    "phase": phase["name"],
                    "plan_sha256": plan_hash,
                    "selection_sha256": selection_hash,
                    "selection_provenance_sha256": selection_provenance_hash,
                    "selection_binding_sha256": selection_binding_hash,
                    "dataset_hashes": dataset_hashes,
                    "launch_authorization": launch_authorization,
                    "source_tree_sha256": source_tree_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16],
        "phase": phase["name"],
        "created_at_utc": _utc_now(),
        "plan": str(plan_path),
        "plan_sha256": plan_hash,
        "selection": str(selection_path) if selection_path is not None else None,
        "selection_sha256": selection_hash,
        "selection_copy": str(selection_copy) if selection_copy else None,
        "selection_provenance": selection_binding,
        "selection_provenance_sha256": selection_provenance_hash,
        "selection_binding_sha256": selection_binding_hash,
        "selection_copy_sha256": selection_hash if selection_copy else None,
        "selection_provenance_copy_sha256": (
            selection_provenance_hash if selection_provenance_copy else None
        ),
        "selection_provenance_copy": str(selection_provenance_copy) if selection_provenance_copy else None,
        "launch_authorization": launch_authorization,
        "launch_authorization_copy": (
            str(launch_authorization_copy.resolve()) if launch_authorization_copy else None
        ),
        "launch_authorization_copy_sha256": launch_authorization_copy_hash,
        "launch_authorization_binding_sha256": (
            _canonical_hash(launch_authorization) if launch_authorization is not None else None
        ),
        "dataset_hashes": dataset_hashes,
        "source_snapshot": str(snapshot.resolve()),
        "source_files": source_files,
        "source_tree_sha256": source_tree_hash,
        "git_head": _git_text(["git", "rev-parse", "HEAD"]).strip(),
        "git_status": _git_text(["git", "status", "--short"]),
        "git_diff_sha256": hashlib.sha256(git_diff.encode("utf-8")).hexdigest(),
        "python": sys.executable,
        "python_version": sys.version,
    }
    # Exclusive creation prevents a second runner from silently replacing the
    # provenance record while the phase is active.
    with provenance_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    provenance_path.chmod(0o444)
    return payload


def _environment() -> dict:
    details = {"python": sys.executable, "python_version": sys.version}
    try:
        import torch

        details.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        details["torch_error"] = repr(exc)
    try:
        import mamba_ssm

        details["mamba_ssm"] = getattr(mamba_ssm, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - baseline environment support
        details["mamba_ssm_error"] = repr(exc)
    for command, key in ((["git", "rev-parse", "HEAD"], "git_commit"),
                         (["git", "status", "--short"], "git_status")):
        try:
            details[key] = subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()
        except Exception as exc:
            details[key] = repr(exc)
    return details


def _cpu_thread_environment(config: dict) -> dict[str, str]:
    """Return plan-bound BLAS/OpenMP limits for one isolated training job."""

    raw_limit = config.get("cpu_thread_limit")
    if raw_limit is None:
        return {}
    limit, error = _plan_positive_int(raw_limit, "cpu_thread_limit")
    if error or limit is None:
        raise ValueError(error or "cpu_thread_limit must be positive")
    value = str(limit)
    return {
        "AUTOCON_CPU_THREAD_LIMIT": value,
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _selected_lambdas(selection_file: Path | None) -> dict[str, float]:
    if selection_file is None:
        return {}
    payload = _load_json(selection_file)
    selected = payload.get("selected_lambda_by_model", payload)
    return {str(key): float(value) for key, value in selected.items()}


def _validate_selection_binding(
    selection_file: Path | None,
    phase: dict,
    plan: dict,
    plan_path: Path,
    selection_provenance_file: Path | None = None,
) -> dict | None:
    """Return a verified immutable binding for a selected-lambda phase."""
    if phase.get("autocon_lambda") != "selected":
        return None
    if selection_file is None or not selection_file.is_file():
        raise ValueError(f"{phase['name']} requires an explicit selection_file")
    payload = _load_json(selection_file)
    expected_phase = phase.get("selection_phase", "selection")
    if payload.get("phase") != expected_phase:
        raise ValueError(
            f"{phase['name']} requires selection phase {expected_phase!r}; "
            f"received {payload.get('phase')!r}"
        )
    expected_features = phase.get("features", plan.get("global", {}).get("features", "S"))
    actual_features = payload.get("features", expected_features)
    if actual_features != expected_features:
        raise ValueError(
            f"selection features {actual_features!r} do not match {phase['name']} features {expected_features!r}"
        )
    # Keep this import local so the runner remains usable for non-selected
    # baseline phases without loading the independent replay implementation.
    try:
        from attest_selection import verify_selection_attestation
    except ModuleNotFoundError:  # package-style imports used by unit tests/tools
        from scripts.attest_selection import verify_selection_attestation

    # Historical confirmatory phases predate this explicit sidecar.  The runner
    # requires it for every new launch; leaving it optional here preserves the
    # ability to inspect legacy phase validation artifacts without upgrading
    # their already immutable provenance.
    if selection_provenance_file is None:
        return None
    if not selection_provenance_file.is_file():
        raise ValueError(f"{phase['name']} selection provenance sidecar is missing")
    return verify_selection_attestation(
        selection_file, plan_path, phase, selection_provenance_file
    )


def _phase_horizons(phase: dict, dataset: str) -> list[int]:
    mapping = phase.get("horizons_by_dataset")
    if mapping is not None:
        values = mapping.get(dataset, [])
    else:
        values = phase.get("horizons", [])
    return [int(value) for value in values]


def _plan_bool(value, name: str) -> tuple[bool, str | None]:
    """Validate a plan boolean without accepting arbitrary truthy strings."""

    if isinstance(value, (bool, np.bool_)):
        return value, None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ) and value in (0, 1):
        return bool(value), None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "on"}:
            return True, None
        if token in {"false", "0", "no", "off"}:
            return False, None
    return False, f"{name} must be boolean"


def _plan_positive_int(value, name: str) -> tuple[int | None, str | None]:
    """Parse a positive integer plan field without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        return None, f"{name} must be a positive integer"
    if isinstance(value, (float, np.floating)) and (
        not math.isfinite(float(value)) or not float(value).is_integer()
    ):
        return None, f"{name} must be a positive integer"
    if isinstance(value, str) and not value.strip():
        return None, f"{name} must be a positive integer"
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a positive integer"
    # ``int(1.5)`` silently truncates, which is particularly dangerous for a
    # horizon or epoch budget.  String integers remain supported for replayed
    # JSON/CLI metadata, but decimal strings are rejected by ``int`` above.
    if not isinstance(value, str):
        try:
            if parsed != value:
                return None, f"{name} must be a positive integer"
        except (TypeError, ValueError):
            return None, f"{name} must be a positive integer"
    if parsed <= 0:
        return None, f"{name} must be a positive integer"
    return parsed, None


def _plan_nonnegative_int(value, name: str) -> tuple[int | None, str | None]:
    """Parse a non-negative integer while preserving the plan error style."""

    if isinstance(value, (bool, np.bool_)):
        return None, f"{name} must be a non-negative integer"
    if isinstance(value, (float, np.floating)) and (
        not math.isfinite(float(value)) or not float(value).is_integer()
    ):
        return None, f"{name} must be a non-negative integer"
    if isinstance(value, str) and not value.strip():
        return None, f"{name} must be a non-negative integer"
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a non-negative integer"
    if not isinstance(value, str):
        try:
            if parsed != value:
                return None, f"{name} must be a non-negative integer"
        except (TypeError, ValueError):
            return None, f"{name} must be a non-negative integer"
    if parsed < 0:
        return None, f"{name} must be a non-negative integer"
    return parsed, None


def _plan_float(
    value,
    name: str,
    *,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> tuple[float | None, str | None]:
    """Parse a finite floating-point plan field and enforce its interval."""

    if isinstance(value, (bool, np.bool_)):
        return None, f"{name} must be a finite number"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a finite number"
    if not math.isfinite(parsed):
        return None, f"{name} must be a finite number"
    if minimum is not None and (
        parsed < minimum or (not minimum_inclusive and parsed == minimum)
    ):
        operator = ">=" if minimum_inclusive else ">"
        return None, f"{name} must be {operator} {minimum}"
    if maximum is not None and (
        parsed > maximum or (not maximum_inclusive and parsed == maximum)
    ):
        operator = "<=" if maximum_inclusive else "<"
        return None, f"{name} must be {operator} {maximum}"
    return parsed, None


def _validate_numeric_plan_config(phase_name: str, config: dict) -> list[str]:
    """Validate runtime budgets and numeric ranges shared by all model arms."""

    errors: list[str] = []
    parsed: dict[str, int | float] = {}
    for key in PLAN_POSITIVE_INTEGER_FIELDS:
        if key not in config or config.get(key) is None:
            continue
        value, error = _plan_positive_int(config.get(key), key)
        if error:
            errors.append(f"{phase_name}: {error}")
        elif value is not None:
            parsed[key] = value
    for key in PLAN_NONNEGATIVE_INTEGER_FIELDS:
        if key not in config or config.get(key) is None:
            continue
        value, error = _plan_nonnegative_int(config.get(key), key)
        if error:
            errors.append(f"{phase_name}: {error}")
        elif value is not None:
            parsed[key] = value
    for key in PLAN_POSITIVE_FLOAT_FIELDS:
        if key not in config or config.get(key) is None:
            continue
        value, error = _plan_float(config.get(key), key, minimum=0.0, minimum_inclusive=False)
        if error:
            errors.append(f"{phase_name}: {error}")
        elif value is not None:
            parsed[key] = value
    for key in PLAN_NONNEGATIVE_FLOAT_FIELDS:
        if key not in config or config.get(key) is None:
            continue
        value, error = _plan_float(config.get(key), key, minimum=0.0)
        if error:
            errors.append(f"{phase_name}: {error}")
        elif value is not None:
            parsed[key] = value

    if "dropout" in config and config.get("dropout") is not None:
        _, error = _plan_float(
            config.get("dropout"), "dropout", minimum=0.0, maximum=1.0,
            maximum_inclusive=False,
        )
        if error:
            errors.append(f"{phase_name}: {error}")
    if "train_ratio" in config and config.get("train_ratio") is not None:
        _, error = _plan_float(
            config.get("train_ratio"), "train_ratio", minimum=0.0,
            minimum_inclusive=False, maximum=1.0, maximum_inclusive=False,
        )
        if error:
            errors.append(f"{phase_name}: {error}")
    if "min_learning_rate" in parsed and "learning_rate" in parsed:
        if float(parsed["min_learning_rate"]) > float(parsed["learning_rate"]):
            errors.append(
                f"{phase_name}: min_learning_rate cannot exceed learning_rate"
            )
    if "seq_len" in parsed and "label_len" in parsed:
        if int(parsed["label_len"]) > int(parsed["seq_len"]):
            errors.append(f"{phase_name}: label_len cannot exceed seq_len")
    raw_wnorm = config.get("autocon_wnorm")
    if raw_wnorm is None:
        raw_wnorm = config.get("AutoCon_wnorm", "LastVal")
    if str(raw_wnorm).strip().lower() == "revin":
        configured_seq_len = parsed.get("seq_len")
        if configured_seq_len is None:
            configured_seq_len, seq_len_error = _plan_positive_int(
                config.get("seq_len", 96), "seq_len"
            )
            if seq_len_error:
                configured_seq_len = None
        if configured_seq_len is not None and configured_seq_len < 2:
            errors.append(
                f"{phase_name}: AutoCon_wnorm=ReVIN requires seq_len >= 2 because it uses an unbiased standard deviation"
            )
    relation_loss, relation_error = _plan_bool(
        config.get("mamba_acf_relation_loss", False),
        "mamba_acf_relation_loss",
    )
    if relation_error:
        errors.append(f"{phase_name}: {relation_error}")
    relation_batching = str(config.get("mamba_acf_relation_batching", "lag_pairs"))
    if relation_loss and relation_batching == "lag_pairs" and "batch_size" in parsed:
        if int(parsed["batch_size"]) < 2:
            errors.append(
                f"{phase_name}: batch_size must be at least 2 for lag-pair relation batching"
            )
        elif int(parsed["batch_size"]) % 2:
            errors.append(
                f"{phase_name}: batch_size must be even for lag-pair relation batching"
            )
    return errors


def _config_bool(config: dict, key: str, default: bool = False) -> bool:
    """Read a plan boolean using the same semantics as plan validation."""

    value, error = _plan_bool(config.get(key, default), key)
    if error:
        raise ValueError(error)
    return value


def _validate_positive_integer_list(value, name: str) -> list[str]:
    """Validate a non-empty list of positive integer values."""

    if not isinstance(value, (list, tuple)) or not value:
        return [f"{name} must be a non-empty list of positive integers"]
    errors: list[str] = []
    parsed: list[int] = []
    for index, item in enumerate(value):
        parsed_value, error = _plan_positive_int(item, f"{name}[{index}]")
        if error:
            errors.append(error)
        elif parsed_value is not None:
            parsed.append(parsed_value)
    if len(parsed) == len(value) and len(set(parsed)) != len(parsed):
        errors.append(f"{name} must not contain duplicate values")
    return errors


def _validate_phase_horizon_contract(
    phase_name: str,
    phase: dict,
    datasets: list[str],
    inherited: dict | None = None,
) -> list[str]:
    """Validate list/mapping horizon declarations before job expansion.

    A JSON ``null`` in a phase means "inherit this field".  Resolve that
    convention here explicitly so a phase can use ``horizons: null`` while a
    global horizon declaration remains authoritative.
    """

    errors: list[str] = []
    inherited = inherited if isinstance(inherited, dict) else {}
    local_mapping = phase.get("horizons_by_dataset")
    local_horizons = phase.get("horizons")
    if local_mapping is not None:
        mapping = local_mapping
        horizons = local_horizons
    elif local_horizons is not None:
        mapping = None
        horizons = local_horizons
    else:
        mapping = inherited.get("horizons_by_dataset")
        horizons = inherited.get("horizons")
    if mapping is not None:
        if not isinstance(mapping, dict) or not mapping:
            return [f"{phase_name}: horizons_by_dataset must be a non-empty object"]
        dataset_names = {item for item in datasets if isinstance(item, str)}
        unknown = sorted(set(mapping) - dataset_names)
        errors.extend(
            f"{phase_name}: horizons_by_dataset contains dataset not in phase: {dataset}"
            for dataset in unknown
        )
        for dataset in datasets:
            if dataset not in mapping:
                errors.append(
                    f"{phase_name}: horizons_by_dataset is missing dataset {dataset!r}"
                )
                continue
            errors.extend(
                _validate_positive_integer_list(
                    mapping.get(dataset),
                    f"{phase_name}.horizons_by_dataset[{dataset}]",
                )
            )
        if horizons is not None:
            errors.append(
                f"{phase_name}: provide either horizons or horizons_by_dataset, not both"
            )
        return errors
    errors.extend(
        _validate_positive_integer_list(
            horizons, f"{phase_name}.horizons"
        )
    )
    return errors


def _canonical_value(value) -> str:
    """Return a stable representation for comparing nested plan values."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _strict_bool(value, name: str) -> list[str]:
    if not isinstance(value, bool):
        return [f"{name} must be a boolean"]
    return []


def _validate_pairing_unit(value, name: str) -> tuple[list[str], tuple[str, ...] | None]:
    """Validate the structured seed-pair coordinate declaration."""

    if not isinstance(value, dict):
        return [f"{name} must be an object with fields"], None
    errors: list[str] = []
    allowed_keys = {"fields", "mode"}
    unknown = sorted(set(value) - allowed_keys)
    errors.extend(f"{name} has unknown field {key!r}" for key in unknown)
    fields = value.get("fields")
    if not isinstance(fields, list) or not fields or any(
        not isinstance(item, str) for item in fields
    ):
        errors.append(f"{name}.fields must be a non-empty list of strings")
        parsed_fields = None
    else:
        parsed_fields = tuple(item.strip().lower() for item in fields)
        if any(not item for item in parsed_fields):
            errors.append(f"{name}.fields must not contain empty values")
        if len(set(parsed_fields)) != len(parsed_fields):
            errors.append(f"{name}.fields must not contain duplicates")
        if parsed_fields != PAIRING_FIELDS:
            errors.append(
                f"{name}.fields must be exactly {list(PAIRING_FIELDS)!r}"
            )
    mode = value.get("mode", "exact")
    if mode != "exact":
        errors.append(f"{name}.mode must be 'exact'")
    return errors, parsed_fields


def _validate_accuracy_gate_contract(
    gate, phase_datasets: set[str], name: str = "comparison_contract.accuracy_gate"
) -> list[str]:
    """Validate the structured, non-test-selection accuracy gate."""

    if not isinstance(gate, dict):
        return [f"{name} must be a structured object"]
    errors: list[str] = []
    allowed_keys = {
        "metric", "datasets", "direction", "aggregation", "uncertainty",
        "single_dataset_favorable_insufficient", "test_metric_for_selection",
        "description",
    }
    errors.extend(
        f"{name} has unknown field {key!r}"
        for key in sorted(set(gate) - allowed_keys)
    )
    if gate.get("metric") != "scaled_mse":
        errors.append(f"{name}.metric must be 'scaled_mse'")
    datasets = gate.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        errors.append(f"{name}.datasets must be a non-empty list")
    else:
        valid_dataset_names = [dataset for dataset in datasets if isinstance(dataset, str)]
        if len(valid_dataset_names) != len(datasets):
            errors.append(f"{name}.datasets must contain dataset names")
        unknown = sorted(set(valid_dataset_names) - phase_datasets)
        errors.extend(
            f"{name}.datasets contains dataset not covered by phases: {dataset}"
            for dataset in unknown
        )
        if len(set(valid_dataset_names)) != len(valid_dataset_names):
            errors.append(f"{name}.datasets must not contain duplicates")
    if gate.get("direction") != "non_worsening":
        errors.append(f"{name}.direction must be 'non_worsening'")
    if gate.get("aggregation") != "pooled_paired":
        errors.append(f"{name}.aggregation must be 'pooled_paired'")
    if gate.get("uncertainty") != "paired_seed":
        errors.append(f"{name}.uncertainty must be 'paired_seed'")
    errors.extend(
        _strict_bool(
            gate.get("single_dataset_favorable_insufficient"),
            f"{name}.single_dataset_favorable_insufficient",
        )
    )
    if gate.get("single_dataset_favorable_insufficient") is not True:
        errors.append(
            f"{name}.single_dataset_favorable_insufficient must be true"
        )
    errors.extend(
        _strict_bool(gate.get("test_metric_for_selection"), f"{name}.test_metric_for_selection")
    )
    if gate.get("test_metric_for_selection") is not False:
        errors.append(f"{name}.test_metric_for_selection must be false")
    if "description" in gate and (
        not isinstance(gate["description"], str) or not gate["description"].strip()
    ):
        errors.append(f"{name}.description must be a non-empty string when present")
    return errors


def _merged_phase_config(plan: dict, phase: dict) -> dict:
    """Merge global and phase settings, treating JSON null as inheritance.

    Keeping nulls out of the effective config is important at two boundaries:
    validation must inspect the inherited value, and command construction must
    never serialize the literal string ``None`` to an argparse option.
    """

    global_config = plan.get("global", {})
    config = {
        key: value
        for key, value in global_config.items()
        if value is not None
    } if isinstance(global_config, dict) else {}
    if isinstance(phase, dict):
        for key, value in phase.items():
            if value is not None:
                config[key] = value
        # Horizon declarations are mutually exclusive.  A non-null phase
        # override also clears an inherited declaration of the other form;
        # otherwise a harmless ``horizons_by_dataset`` override would leave a
        # stale global ``horizons`` key in the effective command config.
        if phase.get("horizons_by_dataset") is not None:
            config.pop("horizons", None)
        elif phase.get("horizons") is not None:
            config.pop("horizons_by_dataset", None)
    return config


def _horizon_declaration(plan: dict, phase: dict) -> dict:
    """Return the effective horizon declaration used for paired comparison."""

    config = _merged_phase_config(plan, phase)
    if config.get("horizons_by_dataset") is not None:
        return {"horizons_by_dataset": config.get("horizons_by_dataset")}
    return {"horizons": config.get("horizons")}


def _validate_scale_conditioning_contract(plan: dict, phases: list[dict]) -> list[str]:
    """Enforce the one-factor and structured-statistics contract for scale arms."""

    errors: list[str] = []
    phase_by_name = {
        phase.get("name"): phase
        for phase in phases
        if isinstance(phase, dict) and isinstance(phase.get("name"), str)
    }
    effective_flags: dict[str, bool] = {}
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        name = str(phase.get("name", "<unnamed>"))
        config = _merged_phase_config(plan, phase)
        flag, flag_error = _plan_bool(
            config.get("mamba_acf_scale_conditioning", False),
            f"{name}.mamba_acf_scale_conditioning",
        )
        if flag_error:
            errors.append(f"{name}: {flag_error}")
        else:
            effective_flags[name] = flag
    scale_requested = any(effective_flags.values())
    contract = plan.get("paired_phase_contract")
    if not scale_requested and contract is None:
        return errors
    if not isinstance(contract, dict):
        errors.append(
            "paired_phase_contract is required when scale conditioning is enabled"
        )
        return errors
    allowed_contract_keys = {"off_phase", "on_phase", "allowed_difference"}
    errors.extend(
        f"paired_phase_contract has unknown field {key!r}"
        for key in sorted(set(contract) - allowed_contract_keys)
    )
    off_name = contract.get("off_phase")
    on_name = contract.get("on_phase")
    if not isinstance(off_name, str) or not off_name.strip():
        errors.append("paired_phase_contract.off_phase must be a non-empty string")
    if not isinstance(on_name, str) or not on_name.strip():
        errors.append("paired_phase_contract.on_phase must be a non-empty string")
    if isinstance(off_name, str) and isinstance(on_name, str) and off_name == on_name:
        errors.append("paired_phase_contract.off_phase and on_phase must differ")
    allowed_difference = contract.get("allowed_difference")
    if allowed_difference != ["mamba_acf_scale_conditioning"]:
        errors.append(
            "paired_phase_contract.allowed_difference must be "
            "['mamba_acf_scale_conditioning']"
        )
    off = phase_by_name.get(off_name)
    on = phase_by_name.get(on_name)
    if off is None:
        errors.append(f"paired_phase_contract.off_phase is not registered: {off_name!r}")
    if on is None:
        errors.append(f"paired_phase_contract.on_phase is not registered: {on_name!r}")
    if off is None or on is None:
        return errors

    off_config = _merged_phase_config(plan, off)
    on_config = _merged_phase_config(plan, on)
    off_flag, off_error = _plan_bool(
        off_config.get("mamba_acf_scale_conditioning", False),
        f"{off_name}.mamba_acf_scale_conditioning",
    )
    on_flag, on_error = _plan_bool(
        on_config.get("mamba_acf_scale_conditioning", False),
        f"{on_name}.mamba_acf_scale_conditioning",
    )
    if off_error:
        errors.append(f"{off_name}: {off_error}")
    elif off_flag:
        errors.append(f"{off_name} must have mamba_acf_scale_conditioning=false")
    if on_error:
        errors.append(f"{on_name}: {on_error}")
    elif not on_flag:
        errors.append(f"{on_name} must have mamba_acf_scale_conditioning=true")
    if on_flag:
        source = resolve_acf_source(on_config, default="")
        if source != "raw":
            errors.append(
                f"{on_name}: mamba_acf_scale_conditioning requires autocon_acf_source=raw"
            )

    off_datasets = off_config.get("datasets")
    on_datasets = on_config.get("datasets")
    off_models = off_config.get("models")
    on_models = on_config.get("models")
    off_seeds = off_config.get("seeds")
    on_seeds = on_config.get("seeds")
    for field, left, right in (
        ("datasets", off_datasets, on_datasets),
        ("models", off_models, on_models),
        ("seeds", off_seeds, on_seeds),
    ):
        if left != right:
            errors.append(
                f"paired scale phases differ in {field}: "
                f"{off_name}={left!r}, {on_name}={right!r}"
            )
    off_horizons = _horizon_declaration(plan, off)
    on_horizons = _horizon_declaration(plan, on)
    if _canonical_value(off_horizons) != _canonical_value(on_horizons):
        errors.append(
            f"paired scale phases differ in horizon declaration: {off_name} vs {on_name}"
        )

    # The only allowed executable-config difference is the treatment flag.
    off_common = dict(off_config)
    on_common = dict(on_config)
    for config in (off_common, on_common):
        for key in PAIR_PHASE_METADATA_FIELDS:
            config.pop(key, None)
        config.pop("mamba_acf_scale_conditioning", None)
    if _canonical_value(off_common) != _canonical_value(on_common):
        differing_keys = sorted(
            key
            for key in set(off_common) | set(on_common)
            if _canonical_value(off_common.get(key))
            != _canonical_value(on_common.get(key))
        )
        errors.append(
            "paired scale phases differ outside the allowed flag: "
            + ", ".join(differing_keys)
        )

    extra_enabled = sorted(
        name for name, enabled in effective_flags.items() if enabled and name != on_name
    )
    if extra_enabled:
        errors.append(
            "scale conditioning is enabled in unpaired phase(s): "
            + ", ".join(extra_enabled)
        )

    comparison_fields = None
    comparison = plan.get("comparison_contract")
    if not isinstance(comparison, dict):
        errors.append("comparison_contract is required for scale conditioning plans")
    else:
        gate = comparison.get("accuracy_gate")
        phase_datasets = {
            dataset
            for phase in phases
            if isinstance(phase, dict)
            for dataset in (_merged_phase_config(plan, phase).get("datasets") or [])
            if isinstance(dataset, str)
        }
        errors.extend(_validate_accuracy_gate_contract(gate, phase_datasets))
        if "pairing_unit" in comparison:
            pairing_errors, comparison_fields = _validate_pairing_unit(
                comparison.get("pairing_unit"), "comparison_contract.pairing_unit"
            )
            errors.extend(pairing_errors)

    seed_policy = plan.get("seed_policy")
    if not isinstance(seed_policy, dict):
        errors.append("seed_policy is required for scale conditioning plans")
    else:
        pairing_errors, seed_fields = _validate_pairing_unit(
            seed_policy.get("pairing_unit"), "seed_policy.pairing_unit"
        )
        errors.extend(pairing_errors)
        if (
            comparison_fields is not None
            and seed_fields is not None
            and comparison_fields != seed_fields
        ):
            errors.append(
                "comparison_contract.pairing_unit.fields must match "
                "seed_policy.pairing_unit.fields"
            )
        declared_seeds = seed_policy.get("seeds")
        errors.extend(
            _validate_positive_integer_list(declared_seeds, "seed_policy.seeds")
        )
        if isinstance(declared_seeds, list) and isinstance(off_seeds, list):
            if declared_seeds != off_seeds:
                errors.append(
                    "seed_policy.seeds must match the paired phase seed list"
                )
    return errors


def _validate_fixed_variable_repr_fusion_contract(
    plan: dict, phases: list[dict], effective_flags: dict[str, bool]
) -> list[str]:
    """Validate representation fusion as a fixed component of another screen."""

    errors: list[str] = []
    contract = plan.get("fixed_variable_repr_fusion_contract")
    if not isinstance(contract, dict):
        errors.append(
            "fixed_variable_repr_fusion_contract must be an object when "
            "variable representation fusion is enabled without a paired screen"
        )
        return errors
    allowed_keys = {"phases", "state"}
    errors.extend(
        "fixed_variable_repr_fusion_contract has unknown field {!r}".format(key)
        for key in sorted(set(contract) - allowed_keys)
    )
    phase_names = contract.get("phases")
    if not isinstance(phase_names, list) or not phase_names:
        errors.append(
            "fixed_variable_repr_fusion_contract.phases must be a non-empty list"
        )
        phase_names = []
    if len(phase_names) != len(set(phase_names)):
        errors.append(
            "fixed_variable_repr_fusion_contract.phases must be unique"
        )
    if contract.get("state") != "enabled":
        errors.append(
            "fixed_variable_repr_fusion_contract.state must be 'enabled'"
        )
    phase_by_name = {
        phase.get("name"): phase
        for phase in phases
        if isinstance(phase, dict) and isinstance(phase.get("name"), str)
    }
    declared = set()
    for name in phase_names:
        if not isinstance(name, str) or not name.strip():
            errors.append(
                "fixed_variable_repr_fusion_contract.phases must contain names"
            )
            continue
        declared.add(name)
        phase = phase_by_name.get(name)
        if phase is None:
            errors.append(
                "fixed_variable_repr_fusion_contract references unknown phase: "
                f"{name!r}"
            )
            continue
        config = _merged_phase_config(plan, phase)
        if not effective_flags.get(name, False):
            errors.append(
                f"{name} must have mamba_variable_repr_fusion=true under the fixed contract"
            )
        models = config.get("models", [])
        if not isinstance(models, list) or not models or any(
            not _supports_variable_repr_fusion(model) for model in models
        ):
            errors.append(
                f"{name}: fixed variable representation fusion requires only "
                "registered prediction-connected AutoCon models"
            )
        if any(_variable_repr_fusion_requires_mixer(model) for model in models) and not bool(
            config.get("mamba_variable_mixer", False)
        ):
            errors.append(
                f"{name}: fixed variable representation fusion requires "
                "mamba_variable_mixer=true"
            )
    extra_enabled = sorted(
        name for name, enabled in effective_flags.items()
        if enabled and name not in declared
    )
    if extra_enabled:
        errors.append(
            "variable representation fusion is enabled outside the fixed "
            "contract phase(s): " + ", ".join(extra_enabled)
        )
    if len(phase_names) >= 2:
        reference = _merged_phase_config(plan, phase_by_name.get(phase_names[0], {}))
        for name in phase_names[1:]:
            phase = phase_by_name.get(name)
            if phase is None:
                continue
            config = _merged_phase_config(plan, phase)
            for field in ("datasets", "models", "seeds"):
                if reference.get(field) != config.get(field):
                    errors.append(
                        "fixed variable representation phases differ in "
                        f"{field}: {phase_names[0]} vs {name}"
                    )
            if _canonical_value(_horizon_declaration(plan, phase)) != _canonical_value(
                _horizon_declaration(plan, phase_by_name.get(phase_names[0], {}))
            ):
                errors.append(
                    "fixed variable representation phases differ in horizon "
                    f"declaration: {phase_names[0]} vs {name}"
                )
    return errors


def _validate_variable_repr_fusion_contract(
    plan: dict, phases: list[dict]
) -> list[str]:
    """Enforce a one-factor off/on contract for variable representation fusion.

    The adapter is intentionally screened as a paired mechanism change.  Keep
    the comparison executable at the plan boundary so a later edit to a
    phase's budget, data coordinates, or relation objective cannot be hidden
    behind the descriptive contrast in the JSON document.
    """

    errors: list[str] = []
    phase_by_name = {
        phase.get("name"): phase
        for phase in phases
        if isinstance(phase, dict) and isinstance(phase.get("name"), str)
    }
    effective_flags: dict[str, bool] = {}
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        name = str(phase.get("name", "<unnamed>"))
        flag, flag_error = _plan_bool(
            _merged_phase_config(plan, phase).get(
                "mamba_variable_repr_fusion", False
            ),
            f"{name}.mamba_variable_repr_fusion",
        )
        if flag_error:
            errors.append(f"{name}: {flag_error}")
        else:
            effective_flags[name] = flag

    requested = any(effective_flags.values())
    contract = plan.get("paired_variable_repr_fusion_contract")
    if contract is None and requested:
        errors.extend(
            _validate_fixed_variable_repr_fusion_contract(
                plan, phases, effective_flags
            )
        )
        return errors
    if not requested and contract is None:
        if plan.get("fixed_variable_repr_fusion_contract") is not None:
            errors.extend(
                _validate_fixed_variable_repr_fusion_contract(
                    plan, phases, effective_flags
                )
            )
        return errors
    if not isinstance(contract, dict):
        errors.append(
            "paired_variable_repr_fusion_contract is required when variable representation fusion is enabled"
        )
        return errors
    allowed_contract_keys = {"off_phase", "on_phase", "allowed_difference"}
    errors.extend(
        "paired_variable_repr_fusion_contract has unknown field {!r}".format(key)
        for key in sorted(set(contract) - allowed_contract_keys)
    )
    off_name = contract.get("off_phase")
    on_name = contract.get("on_phase")
    if not isinstance(off_name, str) or not off_name.strip():
        errors.append(
            "paired_variable_repr_fusion_contract.off_phase must be a non-empty string"
        )
    if not isinstance(on_name, str) or not on_name.strip():
        errors.append(
            "paired_variable_repr_fusion_contract.on_phase must be a non-empty string"
        )
    if isinstance(off_name, str) and isinstance(on_name, str) and off_name == on_name:
        errors.append(
            "paired_variable_repr_fusion_contract.off_phase and on_phase must differ"
        )
    allowed_difference = contract.get("allowed_difference")
    if allowed_difference != ["mamba_variable_repr_fusion"]:
        errors.append(
            "paired_variable_repr_fusion_contract.allowed_difference must be "
            "['mamba_variable_repr_fusion']"
        )
    off = phase_by_name.get(off_name)
    on = phase_by_name.get(on_name)
    if off is None:
        errors.append(
            "paired_variable_repr_fusion_contract.off_phase is not registered: "
            f"{off_name!r}"
        )
    if on is None:
        errors.append(
            "paired_variable_repr_fusion_contract.on_phase is not registered: "
            f"{on_name!r}"
        )
    if off is None or on is None:
        return errors

    off_config = _merged_phase_config(plan, off)
    on_config = _merged_phase_config(plan, on)
    off_flag, off_error = _plan_bool(
        off_config.get("mamba_variable_repr_fusion", False),
        f"{off_name}.mamba_variable_repr_fusion",
    )
    on_flag, on_error = _plan_bool(
        on_config.get("mamba_variable_repr_fusion", False),
        f"{on_name}.mamba_variable_repr_fusion",
    )
    if off_error:
        errors.append(f"{off_name}: {off_error}")
    elif off_flag:
        errors.append(
            f"{off_name} must have mamba_variable_repr_fusion=false"
        )
    if on_error:
        errors.append(f"{on_name}: {on_error}")
    elif not on_flag:
        errors.append(
            f"{on_name} must have mamba_variable_repr_fusion=true"
        )

    on_models = on_config.get("models", [])
    if not isinstance(on_models, list) or not on_models or any(
        not _supports_variable_repr_fusion(model) for model in on_models
    ):
        errors.append(
            f"{on_name}: variable representation fusion requires only registered "
            "prediction-connected AutoCon models"
        )
    if any(_variable_repr_fusion_requires_mixer(model) for model in on_models) and not bool(
        on_config.get("mamba_variable_mixer", False)
    ):
        errors.append(
            f"{on_name}: variable representation fusion requires mamba_variable_mixer=true"
        )

    for field in ("datasets", "models", "seeds"):
        if off_config.get(field) != on_config.get(field):
            errors.append(
                f"paired variable representation phases differ in {field}: "
                f"{off_name}={off_config.get(field)!r}, "
                f"{on_name}={on_config.get(field)!r}"
            )
    if _canonical_value(_horizon_declaration(plan, off)) != _canonical_value(
        _horizon_declaration(plan, on)
    ):
        errors.append(
            "paired variable representation phases differ in horizon declaration: "
            f"{off_name} vs {on_name}"
        )

    # Phase metadata and the treatment flag are the only permitted differences.
    off_common = dict(off_config)
    on_common = dict(on_config)
    for config in (off_common, on_common):
        for key in PAIR_PHASE_METADATA_FIELDS:
            config.pop(key, None)
        config.pop("mamba_variable_repr_fusion", None)
    if _canonical_value(off_common) != _canonical_value(on_common):
        differing_keys = sorted(
            key
            for key in set(off_common) | set(on_common)
            if _canonical_value(off_common.get(key))
            != _canonical_value(on_common.get(key))
        )
        errors.append(
            "paired variable representation phases differ outside the allowed flag: "
            + ", ".join(differing_keys)
        )

    extra_enabled = sorted(
        name for name, enabled in effective_flags.items()
        if enabled and name != on_name
    )
    if extra_enabled:
        errors.append(
            "variable representation fusion is enabled in unpaired phase(s): "
            + ", ".join(extra_enabled)
        )
    return errors


def _validate_variable_gdd_mixer_contract(
    plan: dict, phases: list[dict]
) -> list[str]:
    """Enforce a one-factor off/on contract for the variable-token GDD adapter.

    GDD is intentionally paired separately from the representation-fusion
    screen.  Both arms may therefore consume the same AutoCon representation;
    only the channel-preserving adapter flag (and its declared rank) may vary.
    """

    errors: list[str] = []
    phase_by_name = {
        phase.get("name"): phase
        for phase in phases
        if isinstance(phase, dict) and isinstance(phase.get("name"), str)
    }
    effective_flags: dict[str, bool] = {}
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        name = str(phase.get("name", "<unnamed>"))
        flag, flag_error = _plan_bool(
            _merged_phase_config(plan, phase).get(
                "mamba_variable_gdd_mixer", False
            ),
            f"{name}.mamba_variable_gdd_mixer",
        )
        if flag_error:
            errors.append(f"{name}: {flag_error}")
        else:
            effective_flags[name] = flag

    requested = any(effective_flags.values())
    contract = plan.get("paired_variable_gdd_mixer_contract")
    if not requested and contract is None:
        return errors
    if not isinstance(contract, dict):
        errors.append(
            "paired_variable_gdd_mixer_contract is required when variable GDD "
            "mixer is enabled"
        )
        return errors
    allowed_contract_keys = {"off_phase", "on_phase", "allowed_difference"}
    errors.extend(
        "paired_variable_gdd_mixer_contract has unknown field {!r}".format(key)
        for key in sorted(set(contract) - allowed_contract_keys)
    )
    off_name = contract.get("off_phase")
    on_name = contract.get("on_phase")
    if not isinstance(off_name, str) or not off_name.strip():
        errors.append(
            "paired_variable_gdd_mixer_contract.off_phase must be a non-empty string"
        )
    if not isinstance(on_name, str) or not on_name.strip():
        errors.append(
            "paired_variable_gdd_mixer_contract.on_phase must be a non-empty string"
        )
    if isinstance(off_name, str) and isinstance(on_name, str) and off_name == on_name:
        errors.append(
            "paired_variable_gdd_mixer_contract.off_phase and on_phase must differ"
        )
    allowed_difference = contract.get("allowed_difference")
    if allowed_difference != [
        "mamba_variable_gdd_mixer", "mamba_variable_gdd_mixer_rank"
    ]:
        errors.append(
            "paired_variable_gdd_mixer_contract.allowed_difference must be "
            "['mamba_variable_gdd_mixer', 'mamba_variable_gdd_mixer_rank']"
        )
    off = phase_by_name.get(off_name)
    on = phase_by_name.get(on_name)
    if off is None:
        errors.append(
            "paired_variable_gdd_mixer_contract.off_phase is not registered: "
            f"{off_name!r}"
        )
    if on is None:
        errors.append(
            "paired_variable_gdd_mixer_contract.on_phase is not registered: "
            f"{on_name!r}"
        )
    if off is None or on is None:
        return errors

    off_config = _merged_phase_config(plan, off)
    on_config = _merged_phase_config(plan, on)
    off_flag, off_error = _plan_bool(
        off_config.get("mamba_variable_gdd_mixer", False),
        f"{off_name}.mamba_variable_gdd_mixer",
    )
    on_flag, on_error = _plan_bool(
        on_config.get("mamba_variable_gdd_mixer", False),
        f"{on_name}.mamba_variable_gdd_mixer",
    )
    if off_error:
        errors.append(f"{off_name}: {off_error}")
    elif off_flag:
        errors.append(f"{off_name} must have mamba_variable_gdd_mixer=false")
    if on_error:
        errors.append(f"{on_name}: {on_error}")
    elif not on_flag:
        errors.append(f"{on_name} must have mamba_variable_gdd_mixer=true")

    for name, config in ((off_name, off_config), (on_name, on_config)):
        models = config.get("models", [])
        if not isinstance(models, list) or not models or any(
            not isinstance(model, str)
            or "MambaACF" not in model
            or not model.endswith("AutoCon")
            for model in models
        ):
            errors.append(
                f"{name}: variable GDD mixer requires only MambaACF AutoCon models"
            )
        if not bool(config.get("mamba_variable_mixer", False)):
            errors.append(
                f"{name}: variable GDD mixer requires mamba_variable_mixer=true"
            )

    for field in ("datasets", "models", "seeds"):
        if off_config.get(field) != on_config.get(field):
            errors.append(
                f"paired variable GDD phases differ in {field}: "
                f"{off_name}={off_config.get(field)!r}, "
                f"{on_name}={on_config.get(field)!r}"
            )
    if _canonical_value(_horizon_declaration(plan, off)) != _canonical_value(
        _horizon_declaration(plan, on)
    ):
        errors.append(
            "paired variable GDD phases differ in horizon declaration: "
            f"{off_name} vs {on_name}"
        )

    off_common = dict(off_config)
    on_common = dict(on_config)
    for config in (off_common, on_common):
        for key in PAIR_PHASE_METADATA_FIELDS:
            config.pop(key, None)
        for key in allowed_difference:
            config.pop(key, None)
    if _canonical_value(off_common) != _canonical_value(on_common):
        differing_keys = sorted(
            key
            for key in set(off_common) | set(on_common)
            if _canonical_value(off_common.get(key))
            != _canonical_value(on_common.get(key))
        )
        errors.append(
            "paired variable GDD phases differ outside the allowed flags: "
            + ", ".join(differing_keys)
        )

    extra_enabled = sorted(
        name for name, enabled in effective_flags.items()
        if enabled and name != on_name
    )
    if extra_enabled:
        errors.append(
            "variable GDD mixer is enabled in unpaired phase(s): "
            + ", ".join(extra_enabled)
        )
    return errors


def _validate_mamba_options(phase_name: str, config: dict, models: list[str]) -> list[str]:
    """Validate optional MambaACF architecture fields before job expansion."""

    mamba_models = [model for model in models if str(model).startswith("Mamba")]
    tm_models = [model for model in mamba_models if str(model).startswith("MambaTM")]
    ts_models = [model for model in mamba_models if str(model).startswith("MambaTS")]
    ts_context_models = [
        model for model in ts_models
        if str(model).startswith(("MambaTSv3", "MambaTSv4", "MambaTSv5", "MambaTSv6", "MambaTSv7", "MambaTSv8", "MambaTSv9", "MambaTSv10", "MambaTSv11", "MambaTSv12", "MambaTSv13", "MambaTSv14", "MambaTSv15", "MambaTSv16"))
    ]
    if ts_context_models:
        _, context_error = _plan_positive_int(
            config.get("mamba_ts_context_rank", 16), "mamba_ts_context_rank"
        )
        if context_error:
            return [f"{phase_name}: {context_error}"]
        mixer_type = str(config.get("mamba_ts_mixer_type", "mamba1")).strip().lower()
        if mixer_type not in {"mamba1", "mamba2"}:
            return [f"{phase_name}: mamba_ts_mixer_type must be mamba1 or mamba2"]
    v12_models = [model for model in ts_models if str(model).startswith("MambaTSv12")]
    if v12_models:
        if str(config.get("mamba_ts_mixer_type", "mamba1")).strip().lower() == "mamba2":
            return [f"{phase_name}: MambaTSv12 Delta branches currently require mamba1 mixer"]
        multipliers = config.get("mamba_ts_delta_multipliers", [1.0, 2.0])
        if not isinstance(multipliers, (list, tuple)) or len(multipliers) != 2:
            return [f"{phase_name}: mamba_ts_delta_multipliers must be [1.0, scale]"]
        first, first_error = _plan_float(multipliers[0], "mamba_ts_delta_multipliers[0]")
        second, second_error = _plan_float(multipliers[1], "mamba_ts_delta_multipliers[1]")
        if first_error or second_error or first != 1.0 or second is None or second <= 1.0:
            return [f"{phase_name}: MambaTSv12 requires Delta multipliers [1.0, scale] with scale > 1"]
        dt_min, dt_min_error = _plan_float(config.get("mamba_ts_dt_min", 0.001), "mamba_ts_dt_min")
        dt_max, dt_max_error = _plan_float(config.get("mamba_ts_dt_max", 0.1), "mamba_ts_dt_max")
        if dt_min_error or dt_max_error or dt_min is None or dt_max is None or not 0 < dt_min < dt_max:
            return [f"{phase_name}: MambaTSv12 requires 0 < mamba_ts_dt_min < mamba_ts_dt_max"]
    spectral_models = [model for model in ts_models if str(model).startswith("MambaTSv8")]
    if spectral_models:
        _, spectral_error = _plan_positive_int(
            config.get("mamba_ts_spectral_top_k", 4), "mamba_ts_spectral_top_k"
        )
        if spectral_error:
            return [f"{phase_name}: {spectral_error}"]
    if "mamba2_repr_forecast_fusion" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba2_repr_forecast_fusion"],
            "mamba2_repr_forecast_fusion",
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(str(model) != "Mamba2AutoCon" for model in models):
            return [
                f"{phase_name}: mamba2_repr_forecast_fusion requires only "
                "Mamba2AutoCon models"
            ]
    if "mamba_ts_residual_init" in config:
        try:
            residual_init = float(config["mamba_ts_residual_init"])
        except (TypeError, ValueError):
            return [f"{phase_name}: mamba_ts_residual_init must be numeric"]
        if not -0.99 < residual_init < 0.99:
            return [f"{phase_name}: mamba_ts_residual_init must be in (-0.99, 0.99)"]
        if any(
            not (
                str(model).startswith("MambaTSv6")
                or str(model).startswith("NLinearVariableMamba")
            )
            for model in models
        ):
            return [
                f"{phase_name}: mamba_ts_residual_init requires only MambaTSv6 "
                "or NLinearVariableMamba models"
            ]
    if "mamba_ts_repr_forecast_fusion" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_repr_forecast_fusion"],
            "mamba_ts_repr_forecast_fusion",
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(
            str(model) not in {
                "MambaTSv3AutoCon", "MambaTSv4AutoCon", "MambaTSv5AutoCon", "MambaTSv6AutoCon", "MambaTSv7AutoCon", "MambaTSv10AutoCon", "MambaTSv11AutoCon", "MambaTSv12AutoCon", "MambaTSv13AutoCon", "MambaTSv14AutoCon", "MambaTSv15AutoCon", "MambaTSv16AutoCon"
            }
            for model in models if str(model).startswith("MambaTS")
        ):
            return [
                f"{phase_name}: mamba_ts_repr_forecast_fusion requires only "
                "supported MambaTS AutoCon models"
            ]
    if "mamba_ts_repr_forecast_channelwise" in config:
        channelwise, channelwise_error = _plan_bool(
            config["mamba_ts_repr_forecast_channelwise"],
            "mamba_ts_repr_forecast_channelwise",
        )
        if channelwise_error:
            return [f"{phase_name}: {channelwise_error}"]
        if channelwise and not _config_bool(
            config, "mamba_ts_repr_forecast_fusion", False
        ):
            return [
                f"{phase_name}: mamba_ts_repr_forecast_channelwise requires "
                "mamba_ts_repr_forecast_fusion=true"
            ]
    if "mamba_ts_channel_adapter" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_channel_adapter"], "mamba_ts_channel_adapter"
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(str(model) != "MambaTSv3AutoCon" for model in models):
            return [
                f"{phase_name}: mamba_ts_channel_adapter requires only "
                "MambaTSv3AutoCon models"
            ]
        if enabled:
            _, rank_error = _plan_positive_int(
                config.get("mamba_ts_channel_adapter_rank", 8),
                "mamba_ts_channel_adapter_rank",
            )
            if rank_error:
                return [f"{phase_name}: {rank_error}"]
    if "mamba_ts_revin_affine" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_revin_affine"], "mamba_ts_revin_affine"
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(str(model) != "MambaTSv3AutoCon" for model in models):
            return [
                f"{phase_name}: mamba_ts_revin_affine requires only "
                "MambaTSv3AutoCon models"
            ]
        if enabled and str(
            config.get("autocon_wnorm", config.get("AutoCon_wnorm", "LastVal"))
        ).strip().lower() not in {"meanstd", "revin"}:
            return [
                f"{phase_name}: mamba_ts_revin_affine requires "
                "AutoCon_wnorm=MeanStd or ReVIN"
            ]
    if "mamba_ts_graph_spatial" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_graph_spatial"], "mamba_ts_graph_spatial"
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(
            str(model) not in {"MambaTSv3AutoCon", "GraphDLinear", "STGCN", "GraphWaveNet", "DCRNN"}
            for model in models
        ):
            return [
                f"{phase_name}: mamba_ts_graph_spatial requires only "
                "MambaTSv3AutoCon, GraphDLinear, STGCN, GraphWaveNet, or DCRNN models"
            ]
    if "mamba_ts_repr_linear_expert" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_repr_linear_expert"],
            "mamba_ts_repr_linear_expert",
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(
            str(model) not in {
                "MambaTSv3AutoCon", "MambaTSv4AutoCon", "MambaTSv5AutoCon", "MambaTSv6AutoCon", "MambaTSv7AutoCon"
            }
            for model in models
        ):
            return [
                f"{phase_name}: mamba_ts_repr_linear_expert requires only "
                "MambaTSv3/v4/v5 AutoCon models"
            ]
    if "mamba_ts_repr_linear_expert_mode" in config:
        mode = str(config.get("mamba_ts_repr_linear_expert_mode", "residual")).strip().lower()
        if mode not in {"residual", "blend", "nlinear_blend"}:
            return [
                f"{phase_name}: mamba_ts_repr_linear_expert_mode must be residual, blend, or nlinear_blend"
            ]
        if mode == "blend" and not _config_bool(config, "mamba_ts_repr_linear_expert", False):
            return [
                f"{phase_name}: blend mode requires mamba_ts_repr_linear_expert=true"
            ]
    if "mamba_ts_repr_linear_blend_bias" in config:
        bias, bias_error = _plan_float(
            config.get("mamba_ts_repr_linear_blend_bias", -4.0),
            "mamba_ts_repr_linear_blend_bias",
        )
        if bias_error:
            return [f"{phase_name}: {bias_error}"]
        if not -12.0 <= bias <= 0.0:
            return [
                f"{phase_name}: mamba_ts_repr_linear_blend_bias must be in [-12, 0]"
            ]
    if "mamba_ts_repr_linear_expert_rank" in config:
        _, rank_error = _plan_positive_int(
            config.get("mamba_ts_repr_linear_expert_rank", 16),
            "mamba_ts_repr_linear_expert_rank",
        )
        if rank_error:
            return [f"{phase_name}: {rank_error}"]
    if "mamba_ts_variable_permutation" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_variable_permutation"],
            "mamba_ts_variable_permutation",
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(
            str(model) not in {
                "MambaTSv3", "MambaTSv3AutoCon", "MambaTSv4", "MambaTSv4AutoCon",
                "MambaTSv5", "MambaTSv5AutoCon", "MambaTSv6", "MambaTSv6AutoCon",
                "MambaTSv7", "MambaTSv7AutoCon",
                "MambaTSv7", "MambaTSv7AutoCon",
            }
            for model in models
        ):
            return [
                f"{phase_name}: mamba_ts_variable_permutation requires only "
                "MambaTS v3/v4/v5/v7 models"
            ]
    if "mamba_ts_acf_memory_selector" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_acf_memory_selector"],
            "mamba_ts_acf_memory_selector",
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(
            str(model) not in {"MambaTSv4AutoCon", "MambaTSv5AutoCon", "MambaTSv7AutoCon"}
            for model in models
        ):
            return [
                f"{phase_name}: mamba_ts_acf_memory_selector requires only "
                "MambaTS v4/v5/v7 AutoCon models"
            ]
        if enabled and not _config_bool(config, "mamba_use_acf_memory", False):
            return [
                f"{phase_name}: mamba_ts_acf_memory_selector requires "
                "mamba_use_acf_memory=true"
            ]
    if "mamba_ts_acf_memory_channel_selector" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_acf_memory_channel_selector"],
            "mamba_ts_acf_memory_channel_selector",
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and not _config_bool(config, "mamba_ts_acf_memory_selector", False):
            return [
                f"{phase_name}: mamba_ts_acf_memory_channel_selector requires "
                "mamba_ts_acf_memory_selector=true"
            ]
    if "mamba_ts_forecast_acf_consistency" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_ts_forecast_acf_consistency"],
            "mamba_ts_forecast_acf_consistency",
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        if enabled and any(
            str(model) not in {
                "MambaTSv3AutoCon", "MambaTSv4AutoCon", "MambaTSv5AutoCon", "MambaTSv6AutoCon", "MambaTSv7AutoCon"
            }
            for model in models
        ):
            return [
                f"{phase_name}: mamba_ts_forecast_acf_consistency requires only "
                "MambaTS v3/v4/v5/v7 AutoCon models"
            ]
        if enabled:
            weight = config.get("mamba_ts_forecast_acf_weight", 0.0)
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                return [f"{phase_name}: mamba_ts_forecast_acf_weight must be numeric"]
            if not 0.0 < weight <= 1.0:
                return [
                    f"{phase_name}: mamba_ts_forecast_acf_weight must be in (0, 1]"
                ]
    if "mamba_acf_forecast_pair_consistency" in config:
        enabled, enabled_error = _plan_bool(
            config["mamba_acf_forecast_pair_consistency"],
            "mamba_acf_forecast_pair_consistency",
        )
        if enabled_error:
            return [f"{phase_name}: {enabled_error}"]
        try:
            weight = float(config.get("mamba_acf_forecast_pair_weight", 0.0))
        except (TypeError, ValueError):
            return [
                f"{phase_name}: mamba_acf_forecast_pair_weight must be numeric"
            ]
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            return [
                f"{phase_name}: mamba_acf_forecast_pair_weight must be in [0, 1]"
            ]
        if not enabled:
            if weight != 0.0:
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_weight requires "
                    "mamba_acf_forecast_pair_consistency=true"
                ]
        else:
            if any(str(model) != "NLinearVariableMambaAutoCon" for model in models):
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_consistency requires only "
                    "NLinearVariableMambaAutoCon"
                ]
            if not _config_bool(config, "mamba_acf_relation_loss", False):
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_consistency requires "
                    "mamba_acf_relation_loss=true"
                ]
            if str(config.get("mamba_acf_relation_batching", "lag_pairs")) != "lag_pairs":
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_consistency requires "
                    "mamba_acf_relation_batching=lag_pairs"
                ]
            if str(config.get("mamba_acf_relation_scope", "both")) not in {"both", "global"}:
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_consistency requires a global "
                    "relation scope"
                ]
            if not _config_bool(config, "mamba_acf_relation_lag_mask", False):
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_consistency requires "
                    "mamba_acf_relation_lag_mask=true"
                ]
            if str(config.get("mamba_acf_relation_target_mode", "hard_selected")) != "hard_selected":
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_consistency requires "
                    "mamba_acf_relation_target_mode=hard_selected"
                ]
            if resolve_acf_source(config) != "raw":
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_consistency requires "
                    "autocon_acf_source=raw"
                ]
            if not 0.0 < weight <= 1.0:
                return [
                    f"{phase_name}: mamba_acf_forecast_pair_weight must be in (0, 1]"
                ]
    if tm_models:
        errors: list[str] = []
        for key, default in (
            ("mamba_tm_n1", 64),
            ("mamba_tm_n2", 128),
            ("mamba_tm_repr_width", 64),
        ):
            value, value_error = _plan_positive_int(
                config.get(key, default), key
            )
            if value_error:
                errors.append(f"{phase_name}: {value_error}")
        for key, default in (
            ("mamba_tm_channel_independent", False),
            ("mamba_tm_residual", True),
        ):
            _, value_error = _plan_bool(config.get(key, default), key)
            if value_error:
                errors.append(f"{phase_name}: {value_error}")
        if errors:
            return errors
    relation_representation = str(
        config.get("mamba_acf_relation_representation", "encoder")
    ).lower()
    if relation_representation not in {"encoder", "forecast"}:
        return [
            f"{phase_name}: mamba_acf_relation_representation must be encoder or forecast"
        ]
    if not mamba_models:
        relation_min_lag, relation_min_lag_error = _plan_nonnegative_int(
            config.get("mamba_acf_relation_min_lag", 0),
            "mamba_acf_relation_min_lag",
        )
        if relation_min_lag_error:
            return [f"{phase_name}: {relation_min_lag_error}"]
        relation_capable_models = {"NLinearVariableMambaAutoCon"}
        relation_loss_enabled, relation_loss_error = _plan_bool(
            config.get("mamba_acf_relation_loss", False),
            "mamba_acf_relation_loss",
        )
        if relation_loss_error:
            return [f"{phase_name}: {relation_loss_error}"]
        if relation_min_lag != 0 and (
            not relation_loss_enabled
            or any(str(model) not in relation_capable_models for model in models)
        ):
            return [
                f"{phase_name}: mamba_acf_relation_min_lag requires "
                "mamba_acf_relation_loss=true on a relation-capable AutoCon model"
            ]
        if relation_representation != "encoder":
            return [
                f"{phase_name}: mamba_acf_relation_representation=forecast requires "
                "a phase containing only MambaACF AutoCon models"
            ]
        strict_representation, strict_error = _plan_bool(
            config.get("mamba_require_strict_representation_causality", False),
            "mamba_require_strict_representation_causality",
        )
        if strict_error:
            return [f"{phase_name}: {strict_error}"]
        if strict_representation:
            return [
                f"{phase_name}: mamba_require_strict_representation_causality "
                "requires a phase containing only MambaACF models"
            ]
        channel_lag_bank, channel_lag_bank_error = _plan_bool(
            config.get("mamba_acf_channel_lag_bank", False),
            "mamba_acf_channel_lag_bank",
        )
        if channel_lag_bank_error:
            return [f"{phase_name}: {channel_lag_bank_error}"]
        if channel_lag_bank:
            return [
                f"{phase_name}: mamba_acf_channel_lag_bank requires a phase containing only MambaACF models"
            ]
        data_mixer, data_mixer_error = _plan_bool(
            config.get("mamba_data_dependent_channel_mixer", False),
            "mamba_data_dependent_channel_mixer",
        )
        if data_mixer_error:
            return [f"{phase_name}: {data_mixer_error}"]
        if data_mixer:
            return [
                f"{phase_name}: mamba_data_dependent_channel_mixer requires a phase containing only MambaACF models"
            ]
        variable_gdd_mixer, variable_gdd_error = _plan_bool(
            config.get("mamba_variable_gdd_mixer", False),
            "mamba_variable_gdd_mixer",
        )
        if variable_gdd_error:
            return [f"{phase_name}: {variable_gdd_error}"]
        if variable_gdd_mixer:
            return [
                f"{phase_name}: mamba_variable_gdd_mixer requires a phase containing only MambaACF models"
            ]
        local_agreement_raw = config.get("mamba_acf_local_agreement", False)
        if local_agreement_raw is None:
            local_agreement_raw = False
        local_agreement, local_agreement_error = _plan_bool(
            local_agreement_raw, "mamba_acf_local_agreement"
        )
        if local_agreement_error:
            return [f"{phase_name}: {local_agreement_error}"]
        if local_agreement:
            return [
                f"{phase_name}: mamba_acf_local_agreement requires a phase containing only MambaACF models"
            ]
        reliability_interpolation, reliability_error = _plan_bool(
            config.get("mamba_acf_reliability_interpolation", False),
            "mamba_acf_reliability_interpolation",
        )
        if reliability_error:
            return [f"{phase_name}: {reliability_error}"]
        if reliability_interpolation:
            return [
                f"{phase_name}: mamba_acf_reliability_interpolation requires a phase containing only MambaACF models"
            ]
        memory_gate_mode = str(
            config.get("mamba_acf_memory_gate_mode", "hidden")
        ).lower()
        anchor_copy, anchor_error = _plan_bool(
            config.get("mamba_acf_anchor_copy", False),
            "mamba_acf_anchor_copy",
        )
        if anchor_error:
            return [f"{phase_name}: {anchor_error}"]
        if anchor_copy:
            return [
                f"{phase_name}: mamba_acf_anchor_copy requires a phase containing only MambaACF models"
            ]
        if memory_gate_mode not in {"hidden", "variable_token"}:
            return [
                f"{phase_name}: mamba_acf_memory_gate_mode must be hidden or variable_token"
            ]
        if memory_gate_mode != "hidden":
            return [
                f"{phase_name}: mamba_acf_memory_gate_mode=variable_token requires a phase containing only MambaACF models"
            ]
        token_stride, token_stride_error = _plan_positive_int(
            config.get("mamba_token_stride", 1), "mamba_token_stride"
        )
        if token_stride_error:
            return [f"{phase_name}: {token_stride_error}"]
        if token_stride != 1:
            return [
                f"{phase_name}: mamba_token_stride requires a phase containing only MambaACF models"
            ]
        group_size, group_error = _plan_positive_int(
            config.get("mamba_variable_group_size", 1),
            "mamba_variable_group_size",
        )
        if group_error:
            return [f"{phase_name}: {group_error}"]
        if group_size != 1:
            return [
                f"{phase_name}: mamba_variable_group_size requires a phase containing only MambaACF models"
            ]
        channel_order = str(config.get("mamba_acf_channel_order", "input")).lower()
        if channel_order not in {"input", "strength", "similarity"}:
            return [
                f"{phase_name}: mamba_acf_channel_order must be input, strength, or similarity"
            ]
        if channel_order != "input":
            models = list(config.get("models", []))
            compatible = {
                "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
                "MambaACFVariableAutoCon",
                "NLinearVariableMambaAutoCon",
            }
            if not models or any(model not in compatible for model in models):
                return [
                    f"{phase_name}: mamba_acf_channel_order={channel_order} requires a phase containing only MambaACF models or only NLinearVariableMambaAutoCon"
                ]
        acf_conditioning, acf_error = _plan_bool(
            config.get("mamba_acf_channel_conditioning", False),
            "mamba_acf_channel_conditioning",
        )
        if acf_error:
            return [f"{phase_name}: {acf_error}"]
        if acf_conditioning:
            return [
                f"{phase_name}: mamba_acf_channel_conditioning requires a phase containing only MambaACF models"
            ]
        acf_gate, acf_gate_error = _plan_bool(
            config.get("mamba_acf_channel_gate", False),
            "mamba_acf_channel_gate",
        )
        if acf_gate_error:
            return [f"{phase_name}: {acf_gate_error}"]
        if acf_gate:
            return [
                f"{phase_name}: mamba_acf_channel_gate requires a phase containing only MambaACF models"
            ]
        local_mixer, local_error = _plan_bool(
            config.get("mamba_local_mixer", False), "mamba_local_mixer"
        )
        if local_error:
            return [f"{phase_name}: {local_error}"]
        if local_mixer:
            return [
                f"{phase_name}: mamba_local_mixer requires a phase containing MambaACF models"
            ]
        acf_repr_fusion, acf_repr_error = _plan_bool(
            config.get("mamba_acf_repr_fusion", False),
            "mamba_acf_repr_fusion",
        )
        if acf_repr_error:
            return [f"{phase_name}: {acf_repr_error}"]
        if acf_repr_fusion:
            return [
                f"{phase_name}: mamba_acf_repr_fusion requires a phase containing only MambaACF AutoCon models"
            ]
        variable_repr_fusion, variable_repr_error = _plan_bool(
            config.get("mamba_variable_repr_fusion", False),
            "mamba_variable_repr_fusion",
        )
        if variable_repr_error:
            return [f"{phase_name}: {variable_repr_error}"]
        if variable_repr_fusion and not all(
            _supports_variable_repr_fusion(model) for model in models
        ):
            return [
                f"{phase_name}: mamba_variable_repr_fusion requires a phase containing only registered prediction-connected AutoCon models"
            ]
        variable_acf_selector, selector_error = _plan_bool(
            config.get("mamba_variable_acf_memory_selector", False),
            "mamba_variable_acf_memory_selector",
        )
        if selector_error:
            return [f"{phase_name}: {selector_error}"]
        if variable_acf_selector:
            if models != ["NLinearVariableMambaAutoCon"]:
                return [
                    f"{phase_name}: mamba_variable_acf_memory_selector requires only NLinearVariableMambaAutoCon"
                ]
            if not _config_bool(config, "AutoCon", True):
                return [
                    f"{phase_name}: mamba_variable_acf_memory_selector requires AutoCon=true"
                ]
            if not _config_bool(config, "mamba_use_acf_memory", False):
                return [
                    f"{phase_name}: mamba_variable_acf_memory_selector requires mamba_use_acf_memory=true"
                ]
            try:
                seq_len = int(config.get("seq_len", 0))
                relation_min_lag = int(config.get("mamba_acf_relation_min_lag", 0))
            except (TypeError, ValueError):
                return [
                    f"{phase_name}: selector requires valid seq_len and mamba_acf_relation_min_lag"
                ]
            if relation_min_lag >= seq_len:
                return [
                    f"{phase_name}: selector requires mamba_acf_relation_min_lag < seq_len"
                ]
        horizon_calibration, calibration_error = _plan_bool(
            config.get("mamba_acf_horizon_calibration", False),
            "mamba_acf_horizon_calibration",
        )
        if calibration_error:
            return [f"{phase_name}: {calibration_error}"]
        if horizon_calibration:
            return [
                f"{phase_name}: mamba_acf_horizon_calibration requires a phase containing only MambaACF models"
            ]
        calibration_rank, rank_error = _plan_positive_int(
            config.get("mamba_acf_horizon_calibration_rank", 2),
            "mamba_acf_horizon_calibration_rank",
        )
        if rank_error or calibration_rank not in {1, 2}:
            return [
                f"{phase_name}: mamba_acf_horizon_calibration_rank must be 1 or 2"
            ]
        return []
    errors: list[str] = []
    strict_representation, strict_representation_error = _plan_bool(
        config.get("mamba_require_strict_representation_causality", False),
        "mamba_require_strict_representation_causality",
    )
    if strict_representation_error:
        errors.append(f"{phase_name}: {strict_representation_error}")

    # MambaACF's normalization path intentionally has no decomposition
    # restore state.  Reject this combination while validating the plan so a
    # job cannot reach the forward pass and fail after resources are allocated.
    acf_models = [model for model in mamba_models if "MambaACF" in str(model)]
    acf_wnorm = str(
        config.get("autocon_wnorm", config.get("AutoCon_wnorm", "LastVal"))
    ).strip().lower()
    if acf_models and acf_wnorm == "decomp":
        errors.append(
            f"{phase_name}: MambaACF does not support AutoCon_wnorm=Decomp; "
            "use MeanStd, ReVIN, Mean, or LastVal"
        )

    def positive_int(key: str, default: int | None = None) -> int | None:
        if key not in config and default is None:
            return None
        raw_value = config.get(key, default)
        if raw_value is None:
            if default is None:
                return None
            raw_value = default
        value, value_error = _plan_positive_int(raw_value, key)
        if value_error:
            errors.append(f"{phase_name}: {value_error}")
            return None
        return value

    def bool_option(key: str, default: bool = False) -> bool:
        raw_value = config.get(key, default)
        if raw_value is None:
            raw_value = default
        value, error = _plan_bool(raw_value, key)
        if error:
            errors.append(f"{phase_name}: {error}")
        return value

    relation_loss = bool_option("mamba_acf_relation_loss")

    # ACF Mamba arms fit a train-only prior before the first batch.  Keep the
    # plan boundary as strict as the utility and runtime parsers so malformed
    # values cannot be recorded as a different effective experiment via
    # integer truncation or non-finite floating-point arithmetic.
    acf_model_phase = any("MambaACF" in str(model) for model in mamba_models)
    if acf_model_phase:
        acf_max_lag = positive_int("mamba_acf_max_lag", 720)
        acf_min_lag = positive_int("mamba_acf_min_lag", 2)
        acf_top_k = positive_int("mamba_acf_top_k", 8)
        acf_min_separation = positive_int("mamba_acf_min_separation", 2)
        acf_chunk = positive_int("mamba_acf_chunk", 64)
        if (
            acf_max_lag is not None
            and acf_min_lag is not None
            and acf_min_lag > acf_max_lag
        ):
            errors.append(
                f"{phase_name}: mamba_acf_min_lag cannot exceed mamba_acf_max_lag"
            )
        for key, default, minimum, maximum in (
            ("mamba_acf_min_abs", 0.1, 0.0, 1.0),
            ("mamba_acf_confidence_z", 1.96, 0.0, None),
            ("mamba_acf_temperature", 4.0, 0.0, None),
            ("mamba_variable_gdd_init_scale", 0.0, 0.0, 1.0),
        ):
            raw = config.get(key, default)
            _, value_error = _plan_float(
                raw,
                key,
                minimum=minimum,
                minimum_inclusive=(key != "mamba_acf_temperature"),
                maximum=maximum,
            )
            if value_error:
                errors.append(f"{phase_name}: {value_error}")
        if relation_loss:
            for key, default, minimum, maximum, inclusive in (
                ("AutoCon_temperature", 0.2, 0.0, None, False),
                ("AutoCon_relation_temperature", 0.25, 0.0, None, False),
                ("AutoCon_sign_weight", 0.25, 0.0, None, True),
                ("AutoCon_sign_margin", 0.1, 0.0, None, True),
            ):
                _, value_error = _plan_float(
                    config.get(key, default),
                    key,
                    minimum=minimum,
                    minimum_inclusive=inclusive,
                    maximum=maximum,
                )
                if value_error:
                    errors.append(f"{phase_name}: {value_error}")

    seq_len = positive_int("seq_len", 96) or 96
    if ts_models:
        ts_patch_len, ts_patch_len_error = _plan_positive_int(
            config.get("patch_len", 16), "patch_len"
        )
        ts_patch_stride, ts_patch_stride_error = _plan_positive_int(
            config.get("patch_stride", 8), "patch_stride"
        )
        if ts_patch_len_error:
            errors.append(f"{phase_name}: {ts_patch_len_error}")
        if ts_patch_stride_error:
            errors.append(f"{phase_name}: {ts_patch_stride_error}")
        if ts_patch_len is not None and ts_patch_len > seq_len:
            errors.append(
                f"{phase_name}: patch_len cannot exceed seq_len={seq_len} for MambaTS"
            )
        if ts_patch_len is not None and ts_patch_stride is not None:
            if 1 + (seq_len - ts_patch_len) // ts_patch_stride < 1:
                errors.append(
                    f"{phase_name}: MambaTS patch geometry yields no temporal tokens"
                )
    token_stride = positive_int("mamba_token_stride", 1)
    if token_stride is not None and token_stride > seq_len:
        errors.append(
            f"{phase_name}: mamba_token_stride cannot exceed seq_len={seq_len}"
        )
    parsed_scales = []
    scales = config.get("mamba_scales")
    if scales is not None:
        if isinstance(scales, (int, float)) and not isinstance(scales, bool):
            scales = [scales]
        if not isinstance(scales, (list, tuple)) or not scales:
            errors.append(f"{phase_name}: mamba_scales must be a non-empty list")
        else:
            parsed = []
            for value in scales:
                positive_int_value, value_error = _plan_positive_int(
                    value, "mamba_scales"
                )
                if value_error:
                    errors.append(f"{phase_name}: mamba_scales must contain positive integers")
                else:
                    parsed.append(positive_int_value)
            if parsed and any(value > seq_len for value in parsed):
                errors.append(f"{phase_name}: mamba_scales cannot exceed seq_len={seq_len}")
            parsed_scales = parsed

    channel_independent = bool_option("mamba_channel_independent")
    bool_option("mamba_bipatch_repr_gate")
    channel_gate = bool_option("mamba_channel_gate")
    data_mixer = bool_option("mamba_data_dependent_channel_mixer")
    variable_gdd_mixer = bool_option("mamba_variable_gdd_mixer")
    acf_channel_gate = bool_option("mamba_acf_channel_gate")
    channel_lag_bank = bool_option("mamba_acf_channel_lag_bank")
    channel_lag_top_k = positive_int("mamba_acf_channel_lag_top_k", 2)
    acf_scale_conditioning = bool_option("mamba_acf_scale_conditioning")
    variable_mixer = bool_option("mamba_variable_mixer")
    group_size = positive_int("mamba_variable_group_size", 1)
    channel_order = str(config.get("mamba_acf_channel_order", "input")).lower()
    if channel_order not in {"input", "strength", "similarity"}:
        errors.append(
            f"{phase_name}: mamba_acf_channel_order must be input, strength, or similarity"
        )
    acf_channel_conditioning = bool_option("mamba_acf_channel_conditioning")
    acf_repr_fusion = bool_option("mamba_acf_repr_fusion")
    variable_repr_fusion = bool_option("mamba_variable_repr_fusion")
    use_acf_memory = bool_option("mamba_use_acf_memory")
    dynamic_acf = bool_option("mamba_dynamic_acf")
    anchor_copy = bool_option("mamba_acf_anchor_copy")
    local_agreement = bool_option("mamba_acf_local_agreement")
    local_min_pairs = positive_int("mamba_acf_local_min_pairs", 16)
    horizon_calibration = bool_option("mamba_acf_horizon_calibration")
    calibration_rank = positive_int("mamba_acf_horizon_calibration_rank", 2)
    if calibration_rank is not None and calibration_rank not in {1, 2}:
        errors.append(
            f"{phase_name}: mamba_acf_horizon_calibration_rank must be 1 or 2"
        )
    patch_mixer = bool_option("mamba_patch_mixer")
    patch_permutation = bool_option("mamba_patch_permutation")
    local_mixer = bool_option("mamba_local_mixer")
    positive_int("mamba_channel_gate_rank", 8)
    positive_int("mamba_data_dependent_channel_mixer_rank", 8)
    positive_int("mamba_variable_gdd_mixer_rank", 8)
    positive_int("mamba_variable_mixer_depth", 1)
    patch_depth = positive_int("mamba_patch_mixer_depth", 1)
    patch_len = positive_int("mamba_patch_len", 16)
    positive_int("mamba_patch_stride", 8)
    local_patch_len = positive_int("mamba_local_patch_len", 16)
    positive_int("mamba_local_patch_stride", 8)
    local_width_value = config.get("mamba_local_width")
    local_width = (
        positive_int("mamba_local_width")
        if local_width_value is not None else None
    )
    positive_int("mamba_local_rank", 16)
    local_depth = positive_int("mamba_local_depth", 1)
    if any("MambaACF" in model for model in mamba_models) and patch_len is not None and patch_len > seq_len:
        errors.append(f"{phase_name}: mamba_patch_len cannot exceed seq_len={seq_len}")
    patch_order = str(config.get("mamba_patch_order", "time_major")).lower()
    if any("MambaACF" in model for model in mamba_models) and patch_order not in {"time_major", "channel_major"}:
        errors.append(f"{phase_name}: mamba_patch_order must be time_major or channel_major")
    if data_mixer and (not acf_models or len(acf_models) != len(models)):
        errors.append(
            f"{phase_name}: mamba_data_dependent_channel_mixer requires a phase containing only MambaACF models"
        )
    if variable_gdd_mixer and (not acf_models or len(acf_models) != len(models)):
        errors.append(
            f"{phase_name}: mamba_variable_gdd_mixer requires a phase containing only MambaACF models"
        )
    if variable_gdd_mixer and not variable_mixer:
        errors.append(
            f"{phase_name}: mamba_variable_gdd_mixer requires mamba_variable_mixer=true"
        )
    if channel_independent and (channel_gate or data_mixer or variable_mixer or patch_mixer):
        errors.append(
            f"{phase_name}: mamba_channel_independent is incompatible with shared gate/mixer branches"
        )
    if channel_independent and variable_gdd_mixer:
        errors.append(
            f"{phase_name}: mamba_variable_gdd_mixer is incompatible with mamba_channel_independent"
        )
    memory_gate_mode = str(
        config.get("mamba_acf_memory_gate_mode", "hidden")
    ).lower()
    if memory_gate_mode not in {"hidden", "variable_token"}:
        errors.append(
            f"{phase_name}: mamba_acf_memory_gate_mode must be hidden or variable_token"
        )
    elif memory_gate_mode == "variable_token":
        if not acf_models or len(acf_models) != len(models):
            errors.append(
                f"{phase_name}: mamba_acf_memory_gate_mode=variable_token requires a phase containing only MambaACF models"
            )
        if not variable_mixer:
            errors.append(
                f"{phase_name}: mamba_acf_memory_gate_mode=variable_token requires mamba_variable_mixer=true"
            )
        if not use_acf_memory or dynamic_acf:
            errors.append(
                f"{phase_name}: mamba_acf_memory_gate_mode=variable_token requires static ACF memory"
            )
    if anchor_copy:
        if not acf_models or len(acf_models) != len(models):
            errors.append(
                f"{phase_name}: mamba_acf_anchor_copy requires a phase containing only MambaACF models"
            )
        if dynamic_acf:
            errors.append(
                f"{phase_name}: mamba_acf_anchor_copy requires static ACF memory (mamba_dynamic_acf=false)"
            )
        acf_source = resolve_acf_source(config)
        if acf_source != "raw":
            errors.append(
                f"{phase_name}: mamba_acf_anchor_copy requires autocon_acf_source=raw"
            )
    acf_repr_fusion_models = {
        "MambaACFAutoCon", "MambaACFVariableAutoCon", "MambaACFLiteAutoCon",
        "MambaACF1AutoCon", "MambaTSv4AutoCon", "MambaTSv5AutoCon",
    }
    if acf_repr_fusion and (
        not models or any(model not in acf_repr_fusion_models for model in models)
    ):
        errors.append(
            f"{phase_name}: mamba_acf_repr_fusion requires a phase containing only "
            "MambaACF AutoCon models (or MambaTSv4/v5 AutoCon models for the TS adapters)"
        )
    if acf_repr_fusion and any(
        str(model) in {"MambaTSv4AutoCon", "MambaTSv5AutoCon", "MambaTSv7AutoCon"} for model in models
    ):
        if not use_acf_memory:
            errors.append(
                f"{phase_name}: MambaTSv4/v5/v7 ACF representation fusion requires "
                "mamba_use_acf_memory=true"
            )
        if resolve_acf_source(config) != "raw":
            errors.append(
                f"{phase_name}: MambaTSv4/v5/v7 ACF representation fusion requires "
                "autocon_acf_source=raw"
            )
    if variable_repr_fusion and (
        not models or any(not _supports_variable_repr_fusion(model) for model in models)
    ):
        errors.append(
            f"{phase_name}: mamba_variable_repr_fusion requires a phase containing only registered prediction-connected AutoCon models"
        )
    if variable_repr_fusion and any(
        _variable_repr_fusion_requires_mixer(model) for model in models
    ) and not variable_mixer:
        errors.append(
            f"{phase_name}: mamba_variable_repr_fusion requires mamba_variable_mixer=true"
        )
    if horizon_calibration and (
        not acf_models or len(acf_models) != len(models)
    ):
        errors.append(
            f"{phase_name}: mamba_acf_horizon_calibration requires a phase containing only MambaACF models"
        )
    if horizon_calibration and not use_acf_memory:
        errors.append(
            f"{phase_name}: mamba_acf_horizon_calibration requires mamba_use_acf_memory=true"
        )
    if horizon_calibration and dynamic_acf:
        errors.append(
            f"{phase_name}: mamba_acf_horizon_calibration requires mamba_dynamic_acf=false"
        )
    reliability_interpolation = bool_option(
        "mamba_acf_reliability_interpolation"
    )
    if reliability_interpolation and (
        not acf_models or len(acf_models) != len(models)
    ):
        errors.append(
            f"{phase_name}: mamba_acf_reliability_interpolation requires a phase containing only MambaACF models"
        )
    if reliability_interpolation and not use_acf_memory:
        errors.append(
            f"{phase_name}: mamba_acf_reliability_interpolation requires mamba_use_acf_memory=true"
        )
    if reliability_interpolation and dynamic_acf:
        errors.append(
            f"{phase_name}: mamba_acf_reliability_interpolation requires mamba_dynamic_acf=false"
        )
    if reliability_interpolation:
        reliability_source = resolve_acf_source(config)
        if reliability_source != "raw":
            errors.append(
                f"{phase_name}: mamba_acf_reliability_interpolation requires autocon_acf_source=raw"
            )
        if bool_option("mamba_acf_step_reliability"):
            errors.append(
                f"{phase_name}: mamba_acf_reliability_interpolation is incompatible with mamba_acf_step_reliability=true"
            )
    if channel_lag_bank:
        if not acf_models or len(acf_models) != len(models):
            errors.append(
                f"{phase_name}: mamba_acf_channel_lag_bank requires a phase containing only MambaACF models"
            )
        if not use_acf_memory:
            errors.append(
                f"{phase_name}: mamba_acf_channel_lag_bank requires mamba_use_acf_memory=true"
            )
        if dynamic_acf:
            errors.append(
                f"{phase_name}: mamba_acf_channel_lag_bank requires static ACF memory (mamba_dynamic_acf=false)"
            )
        if channel_lag_top_k is not None:
            configured_lags = config.get("mamba_acf_lags", (24, 168, 336))
            if isinstance(configured_lags, (int, float)) and not isinstance(
                configured_lags, bool
            ):
                configured_lags = [configured_lags]
            if not isinstance(configured_lags, (list, tuple)) or not configured_lags:
                errors.append(
                    f"{phase_name}: mamba_acf_channel_lag_bank requires non-empty mamba_acf_lags"
                )
            else:
                parsed_lags = []
                for lag in configured_lags:
                    parsed_lag, lag_error = _plan_positive_int(lag, "mamba_acf_lags")
                    if lag_error:
                        errors.append(f"{phase_name}: {lag_error}")
                    elif parsed_lag is not None:
                        parsed_lags.append(parsed_lag)
                if len(parsed_lags) != len(set(parsed_lags)):
                    errors.append(
                        f"{phase_name}: mamba_acf_lags must contain unique positive integers"
                    )
                unique_lag_count = len(set(parsed_lags))
                if channel_lag_top_k > unique_lag_count:
                    errors.append(
                        f"{phase_name}: mamba_acf_channel_lag_top_k cannot exceed mamba_acf_lags count"
                    )
    if acf_models and str(config.get("mamba_acf_detrend", "mean")).lower() == "linear":
        errors.append(
            f"{phase_name}: mamba_acf_detrend=linear is unsupported for MambaACF "
            "because runtime memory does not apply the same global detrend"
        )
    if strict_representation:
        if not acf_models or len(acf_models) != len(models):
            errors.append(
                f"{phase_name}: mamba_require_strict_representation_causality "
                "requires a phase containing only MambaACF models"
            )
        else:
            for reason in strict_representation_causality_errors(
                config, model=acf_models[0]
            ):
                errors.append(
                    f"{phase_name}: strict representation causality: {reason}"
                )
    if token_stride is not None and token_stride != 1 and (
        not acf_models or len(acf_models) != len(models)
    ):
        errors.append(
            f"{phase_name}: mamba_token_stride requires a phase containing only MambaACF models"
        )
    if token_stride is not None and acf_models:
        base_token_count = (seq_len + token_stride - 1) // token_stride
        if any(value > base_token_count for value in parsed_scales):
            errors.append(
                f"{phase_name}: mamba_scales cannot exceed {base_token_count} "
                f"base tokens after mamba_token_stride={token_stride}"
            )
    temporal_rank = positive_int("mamba_temporal_rank", 16)
    low_rank_head = bool_option("mamba_low_rank_head")
    if low_rank_head and temporal_rank is not None:
        effective_temporal_length = (
            (seq_len + token_stride - 1) // token_stride
            if token_stride is not None and acf_models
            else seq_len
        )
        pred_len = positive_int("pred_len", 96) or 96
        if temporal_rank >= min(effective_temporal_length, pred_len):
            errors.append(
                f"{phase_name}: mamba_temporal_rank must be smaller than "
                f"min(effective temporal tokens={effective_temporal_length}, pred_len={pred_len})"
            )
    if acf_channel_gate and (
        not acf_models or len(acf_models) != len(models)
    ):
        errors.append(
            f"{phase_name}: mamba_acf_channel_gate requires a phase containing only MambaACF models"
        )
    if acf_channel_gate and not channel_gate:
        errors.append(
            f"{phase_name}: mamba_acf_channel_gate requires mamba_channel_gate=true"
        )
    if acf_scale_conditioning and (
        not acf_models or len(acf_models) != len(models)
    ):
        errors.append(
            f"{phase_name}: mamba_acf_scale_conditioning requires a phase containing only MambaACF models"
        )
    if acf_scale_conditioning and len(set(parsed_scales)) < 2:
        errors.append(
            f"{phase_name}: mamba_acf_scale_conditioning requires at least two temporal scales"
        )
    if acf_scale_conditioning:
        acf_source = resolve_acf_source(config)
        if acf_source != "raw":
            errors.append(
                f"{phase_name}: mamba_acf_scale_conditioning requires autocon_acf_source=raw"
            )
        acf_lags_for_scale = config.get("mamba_acf_lags", (24, 168, 336))
        if isinstance(acf_lags_for_scale, (int, float)) and not isinstance(acf_lags_for_scale, bool):
            acf_lags_for_scale = [acf_lags_for_scale]
        try:
            eligible_scale_lags = [
                int(lag) for lag in acf_lags_for_scale
                if int(lag) > 0 and seq_len - int(lag) >= 16
            ]
        except (TypeError, ValueError):
            eligible_scale_lags = []
        if not eligible_scale_lags:
            errors.append(
                f"{phase_name}: mamba_acf_scale_conditioning requires at least one ACF lag with 16 or more window pairs"
            )
    if acf_channel_conditioning and (
        not acf_models or len(acf_models) != len(models)
    ):
        errors.append(
            f"{phase_name}: mamba_acf_channel_conditioning requires a phase containing only MambaACF models"
        )
    if acf_channel_conditioning and not variable_mixer:
        errors.append(
            f"{phase_name}: mamba_acf_channel_conditioning requires mamba_variable_mixer=true"
        )
    compatible_channel_order_models = {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
        "MambaACFVariableAutoCon",
        "NLinearVariableMambaAutoCon",
    }
    if channel_order != "input" and (
        not models or any(model not in compatible_channel_order_models for model in models)
    ):
        errors.append(
            f"{phase_name}: mamba_acf_channel_order={channel_order} requires a phase containing only MambaACF models or only NLinearVariableMambaAutoCon"
        )
    if (channel_order != "input" and not variable_mixer
            and any(model != "NLinearVariableMambaAutoCon" for model in models)):
        errors.append(
            f"{phase_name}: mamba_acf_channel_order={channel_order} requires mamba_variable_mixer=true"
        )
    if group_size is not None and group_size != 1:
        if not acf_models or len(acf_models) != len(models):
            errors.append(
                f"{phase_name}: mamba_variable_group_size>1 requires a phase containing only MambaACF models"
            )
        if not variable_mixer:
            errors.append(
                f"{phase_name}: mamba_variable_group_size requires mamba_variable_mixer=true"
            )
    if patch_mixer and (
        not acf_models or len(acf_models) != len(mamba_models)
    ):
        errors.append(
            f"{phase_name}: mamba_patch_mixer requires a phase containing only MambaACF models"
        )
    if patch_permutation and not patch_mixer:
        errors.append(f"{phase_name}: mamba_patch_permutation requires mamba_patch_mixer=true")
    if patch_depth is not None and patch_depth > 64:
        errors.append(f"{phase_name}: mamba_patch_mixer_depth is unreasonably large (>{64})")
    if local_mixer and (
        not acf_models or len(acf_models) != len(models)
    ):
        errors.append(
            f"{phase_name}: mamba_local_mixer requires a phase containing only MambaACF models"
        )
    if local_mixer and channel_independent:
        errors.append(
            f"{phase_name}: mamba_local_mixer is incompatible with mamba_channel_independent"
        )
    if local_mixer and local_patch_len is not None and local_patch_len > seq_len:
        errors.append(
            f"{phase_name}: mamba_local_patch_len cannot exceed seq_len={seq_len}"
        )
    if local_mixer and local_depth is not None and local_depth > 64:
        errors.append(
            f"{phase_name}: mamba_local_depth is unreasonably large (>{64})"
        )

    relation_batching = str(config.get("mamba_acf_relation_batching", "lag_pairs"))
    if relation_batching not in {"lag_pairs", "random"}:
        errors.append(f"{phase_name}: mamba_acf_relation_batching must be lag_pairs or random")
    relation_lag_balance = str(config.get("mamba_acf_relation_lag_balance", "random")).lower()
    if relation_lag_balance not in {"random", "balanced"}:
        errors.append(
            f"{phase_name}: mamba_acf_relation_lag_balance must be random or balanced"
        )
    if relation_lag_balance == "balanced" and relation_batching != "lag_pairs":
        errors.append(
            f"{phase_name}: balanced lag exposure requires mamba_acf_relation_batching=lag_pairs"
        )
    relation_reduce = str(config.get("mamba_acf_relation_channel_reduce", "mean"))
    if relation_reduce not in {"mean", "max_abs"}:
        errors.append(f"{phase_name}: mamba_acf_relation_channel_reduce must be mean or max_abs")
    relation_scope = str(config.get("mamba_acf_relation_scope", "both")).lower()
    if relation_scope not in {"both", "global", "local"}:
        errors.append(f"{phase_name}: mamba_acf_relation_scope must be both, global, or local")
    if relation_representation == "forecast":
        if (
            not acf_models
            or len(acf_models) != len(models)
            or any(not str(model).endswith("AutoCon") for model in acf_models)
        ):
            errors.append(
                f"{phase_name}: mamba_acf_relation_representation=forecast requires "
                "only MambaACF AutoCon models"
            )
        if not relation_loss:
            errors.append(
                f"{phase_name}: mamba_acf_relation_representation=forecast requires "
                "mamba_acf_relation_loss=true"
            )
        if relation_scope != "global":
            errors.append(
                f"{phase_name}: mamba_acf_relation_representation=forecast requires "
                "mamba_acf_relation_scope=global"
            )
        if resolve_acf_source(config) != "raw":
            errors.append(
                f"{phase_name}: mamba_acf_relation_representation=forecast requires "
                "autocon_acf_source=raw"
            )
        if str(config.get("features", "")).upper() != "M":
            errors.append(
                f"{phase_name}: mamba_acf_relation_representation=forecast requires features=M"
            )
        if config.get("enc_in") != config.get("c_out"):
            errors.append(
                f"{phase_name}: mamba_acf_relation_representation=forecast requires c_out=enc_in"
            )
        # ``enc_in`` is derived by the job expander for registered plans, so
        # the phase-level check below validates the dataset schema.  Keep an
        # explicit-config guard here for direct callers and synthetic tests.
        if config.get("enc_in") is not None:
            try:
                forecast_relation_channels = int(config.get("enc_in"))
            except (TypeError, ValueError):
                forecast_relation_channels = 0
            if forecast_relation_channels < 2:
                errors.append(
                    f"{phase_name}: mamba_acf_relation_representation=forecast requires enc_in>=2"
                )
        if acf_repr_fusion or variable_repr_fusion:
            errors.append(
                f"{phase_name}: mamba_acf_relation_representation=forecast is incompatible "
                "with encoder-only representation fusion options"
            )
    relation_lag_tolerance, tolerance_error = _plan_nonnegative_int(
        config.get("mamba_acf_relation_lag_tolerance", 0),
        "mamba_acf_relation_lag_tolerance",
    )
    if tolerance_error:
        errors.append(f"{phase_name}: {tolerance_error}")
    relation_lag_mask = bool_option("mamba_acf_relation_lag_mask")
    relation_min_lag, min_lag_error = _plan_nonnegative_int(
        config.get("mamba_acf_relation_min_lag", 0),
        "mamba_acf_relation_min_lag",
    )
    if min_lag_error:
        errors.append(f"{phase_name}: {min_lag_error}")
        relation_min_lag = 0
    elif not relation_loss and relation_min_lag != 0:
        errors.append(
            f"{phase_name}: mamba_acf_relation_min_lag requires "
            "mamba_acf_relation_loss=true"
        )
    relation_target_mode = str(
        config.get("mamba_acf_relation_target_mode", "hard_selected")
    ).lower()
    if relation_target_mode not in {"hard_selected", "soft_multi_peak", "soft_full"}:
        errors.append(
            f"{phase_name}: mamba_acf_relation_target_mode must be "
            "hard_selected, soft_multi_peak, or soft_full"
        )
    relation_soft_min_abs = config.get("mamba_acf_relation_soft_min_abs", 0.1)
    if isinstance(relation_soft_min_abs, bool):
        errors.append(
            f"{phase_name}: mamba_acf_relation_soft_min_abs must be in [0, 1]"
        )
    else:
        try:
            relation_soft_min_abs = float(relation_soft_min_abs)
            if (
                not math.isfinite(relation_soft_min_abs)
                or not 0 <= relation_soft_min_abs <= 1
            ):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f"{phase_name}: mamba_acf_relation_soft_min_abs must be in [0, 1]"
            )
    relation_diagnostic_min_abs = config.get(
        "mamba_acf_relation_diagnostic_min_abs", 0.1
    )
    if isinstance(relation_diagnostic_min_abs, bool):
        errors.append(
            f"{phase_name}: mamba_acf_relation_diagnostic_min_abs must be in [0, 1]"
        )
    else:
        try:
            relation_diagnostic_min_abs = float(relation_diagnostic_min_abs)
            if (
                not math.isfinite(relation_diagnostic_min_abs)
                or not 0 <= relation_diagnostic_min_abs <= 1
            ):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f"{phase_name}: mamba_acf_relation_diagnostic_min_abs must be in [0, 1]"
            )
    relation_selected_lag_boost = config.get(
        "mamba_acf_relation_selected_lag_boost", 0.0
    )
    if isinstance(relation_selected_lag_boost, bool):
        errors.append(
            f"{phase_name}: mamba_acf_relation_selected_lag_boost must be finite and non-negative"
        )
    else:
        try:
            relation_selected_lag_boost = float(relation_selected_lag_boost)
            if not math.isfinite(relation_selected_lag_boost) or relation_selected_lag_boost < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                f"{phase_name}: mamba_acf_relation_selected_lag_boost must be finite and non-negative"
            )
    bool_option("mamba_acf_step_reliability")
    acf_autocon_models = [
        model for model in acf_models
        if model in {
            "MambaACFAutoCon", "MambaACFVariableAutoCon", "MambaACFLiteAutoCon",
            "MambaACF1AutoCon",
        }
    ]
    if acf_autocon_models and not relation_loss:
        errors.append(
            f"{phase_name}: MambaACF AutoCon models require mamba_acf_relation_loss=true"
        )
    if relation_loss:
        if (
            not acf_models
            or any(not model.endswith("AutoCon") for model in acf_models)
            or len(acf_models) != len(mamba_models)
        ):
            errors.append(
                f"{phase_name}: mamba_acf_relation_loss requires a phase containing only MambaACF AutoCon models"
            )
    if (
        relation_loss
        and acf_autocon_models
        and token_stride is not None
        and token_stride > 1
        and relation_scope in {"both", "local"}
    ):
        errors.append(
            f"{phase_name}: mamba_token_stride={token_stride} is incompatible with "
            f"mamba_acf_relation_scope={relation_scope} when MambaACF AutoCon "
            "relation loss is enabled; use mamba_token_stride=1 or "
            "mamba_acf_relation_scope=global"
        )
    acf_lags = config.get("mamba_acf_lags")
    if acf_lags is not None:
        if isinstance(acf_lags, (int, float)) and not isinstance(acf_lags, bool):
            acf_lags = [acf_lags]
        if not isinstance(acf_lags, (list, tuple)) or not acf_lags:
            errors.append(f"{phase_name}: mamba_acf_lags must be a non-empty list")
        else:
            parsed_lags = []
            lag_values_valid = True
            for value in acf_lags:
                parsed_lag, lag_error = _plan_positive_int(value, "mamba_acf_lags")
                if lag_error:
                    errors.append(f"{phase_name}: mamba_acf_lags must contain positive integers")
                    lag_values_valid = False
                    break
                parsed_lags.append(parsed_lag)
            if (
                lag_values_valid
                and len(parsed_lags) == len(set(parsed_lags))
            ):
                pass
            elif lag_values_valid:
                errors.append(
                    f"{phase_name}: mamba_acf_lags must contain unique positive integers"
                )
    elif anchor_copy:
        errors.append(
            f"{phase_name}: mamba_acf_anchor_copy requires non-empty mamba_acf_lags"
        )
    if local_agreement and (
        not acf_models or len(acf_models) != len(models)
    ):
        errors.append(
            f"{phase_name}: mamba_acf_local_agreement requires a phase containing only MambaACF models"
        )
    if local_agreement and not use_acf_memory:
        errors.append(
            f"{phase_name}: mamba_acf_local_agreement requires mamba_use_acf_memory=true"
        )
    if local_agreement and dynamic_acf:
        errors.append(
            f"{phase_name}: mamba_acf_local_agreement requires static ACF memory (mamba_dynamic_acf=false)"
        )
    if local_agreement:
        acf_source = resolve_acf_source(config)
        if acf_source != "raw":
            errors.append(
                f"{phase_name}: mamba_acf_local_agreement requires autocon_acf_source=raw"
            )
        eligible_lags = []
        if local_min_pairs is not None and isinstance(acf_lags, (list, tuple)):
            for value in acf_lags:
                parsed_lag, lag_error = _plan_positive_int(value, "mamba_acf_lags")
                if lag_error or parsed_lag is None:
                    continue
                if seq_len - parsed_lag >= local_min_pairs:
                    eligible_lags.append(parsed_lag)
        if local_min_pairs is not None and not eligible_lags:
            errors.append(
                f"{phase_name}: mamba_acf_local_agreement requires at least one ACF lag with the configured minimum pair support"
            )
    lag_mode = str(config.get("mamba_acf_lag_mode", "fixed"))
    if lag_mode not in {"fixed", "train_peaks", "train_peaks_available"}:
        errors.append(
            f"{phase_name}: mamba_acf_lag_mode must be fixed, train_peaks, or train_peaks_available"
        )
    max_lag = positive_int("mamba_acf_max_lag", 720)
    min_lag = positive_int("mamba_acf_min_lag", 2)
    if max_lag is not None and min_lag is not None and min_lag > max_lag:
        errors.append(f"{phase_name}: mamba_acf_min_lag cannot exceed mamba_acf_max_lag")
    if (
        relation_loss
        and max_lag is not None
        and relation_min_lag > max_lag
    ):
        errors.append(
            f"{phase_name}: mamba_acf_relation_min_lag cannot exceed "
            "mamba_acf_max_lag when relation loss is enabled"
        )
    positive_int("mamba_acf_top_k", 8)
    positive_int("mamba_acf_min_separation", 2)
    positive_int("mamba_acf_chunk", 64)
    return errors


def _registered_job_count_errors(plan: dict, phases: list[dict]) -> list[str]:
    """Re-expand a plan and compare every registered count before launch.

    Count metadata is intentionally treated as a contract rather than a
    comment.  Selected-lambda phases use the unresolved one-lambda placeholder
    because selection changes the value, not the number of jobs.
    """

    registered = plan.get("registered_job_counts")
    if registered is None:
        if _is_confirmation_plan(plan):
            return ["confirmation plan requires registered_job_counts"]
        return []
    if not isinstance(registered, dict):
        return ["registered_job_counts must be an object"]
    errors: list[str] = []
    try:
        actual_by_phase = {
            str(phase["name"]): len(
                _job_specs(plan, phase, {}, allow_unresolved=True)
            )
            for phase in phases
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return [f"registered_job_counts cannot be recomputed: {exc}"]

    # Every registered phase must be present and must be an exact integer.
    for phase_name, actual in actual_by_phase.items():
        if phase_name not in registered:
            errors.append(f"registered_job_counts is missing phase {phase_name}")
            continue
        declared = registered[phase_name]
        if isinstance(declared, bool) or not isinstance(declared, int):
            errors.append(f"registered_job_counts[{phase_name}] must be an integer")
        elif declared != actual:
            errors.append(
                f"registered_job_counts[{phase_name}]={declared} disagrees with expanded jobs={actual}"
            )

    phase_names = set(actual_by_phase)
    all_total = sum(actual_by_phase.values())
    aggregate_values: dict[str, int] = {
        "total": all_total,
        "training_total": all_total,
    }
    metadata_keys = {"training_started", "resource_runs_not_in_training_matrix"}
    named_aggregate_keys = {"selection_total", "confirmation_total", "exploratory_total"}
    # Accept a named aggregate only when its prefix identifies at least one
    # registered phase.  This keeps legacy high-dimensional plans such as
    # ``highdim_selection_M`` extensible while rejecting silently invented
    # count fields in a strict confirmation plan.
    dynamic_aggregate_keys = {
        f"{token}_total"
        for phase_name in phase_names
        for token in str(phase_name).split("_")
        if token
    }
    allowed_keys = (
        phase_names
        | set(aggregate_values)
        | named_aggregate_keys
        | dynamic_aggregate_keys
        | metadata_keys
    )
    for key in registered:
        if not isinstance(key, str):
            errors.append(f"registered_job_counts key {key!r} must be a string")
        elif key not in allowed_keys:
            errors.append(f"registered_job_counts has unknown key {key!r}")
    for key in registered:
        if not isinstance(key, str) or key in aggregate_values or key in actual_by_phase or key in metadata_keys:
            continue
        if key == "selection_total":
            names = [name for name in phase_names if "selection" in name]
        elif key == "confirmation_total":
            # The frozen confirmation matrix includes its sensitivity phase.
            names = [
                name
                for name in phase_names
                if "confirm" in name or "sensitivity" in name
            ]
        elif key == "exploratory_total":
            status = str(plan.get("status", "")).lower()
            names = (
                list(phase_names)
                if "exploratory" in status
                else [name for name in phase_names if "exploratory" in name or "screen" in name]
            )
        elif key.endswith("_total"):
            token = key[: -len("_total")]
            names = [name for name in phase_names if token and token in name]
        else:
            continue
        if names:
            aggregate_values[key] = sum(actual_by_phase[name] for name in names)
        else:
            errors.append(f"registered_job_counts[{key}] has no matching phase group")

    for key, actual in aggregate_values.items():
        if key not in registered:
            continue
        declared = registered[key]
        if isinstance(declared, bool) or not isinstance(declared, int):
            errors.append(f"registered_job_counts[{key}] must be an integer")
        elif declared != actual:
            errors.append(
                f"registered_job_counts[{key}]={declared} disagrees with expanded jobs={actual}"
            )
    # Once a plan opts into the named confirmation totals, require the whole
    # aggregate contract; a partially documented matrix is not auditable.
    named_total_keys = {"selection_total", "confirmation_total", "training_total"}
    # The confirmation protocol declares all three totals as part of its
    # frozen matrix.  Require them even if a malformed plan happens to include
    # only one of the names; exploratory/legacy plans retain the older
    # partial-count compatibility rule.
    require_all_named = _is_confirmation_plan(plan)
    if require_all_named or named_total_keys.intersection(registered):
        required_named = {
            "selection_total": require_all_named or any("selection" in name for name in phase_names),
            "confirmation_total": require_all_named or any(
                "confirm" in name or "sensitivity" in name for name in phase_names
            ),
            "training_total": True,
        }
        for key, needed in required_named.items():
            if needed and key not in registered:
                errors.append(f"registered_job_counts is missing aggregate key {key}")
    return errors


def _contrast_endpoint_models(spec: str) -> tuple[str, ...] | None:
    """Extract model names from a frozen ``phase:model::phase:model`` spec."""

    value = str(spec).strip()
    if "=" in value:
        _, value = value.split("=", 1)
    if " versus " in value and "::" not in value:
        parts = value.split(" versus ", 1)
    else:
        parts = value.split("::")
    if len(parts) != 2:
        return None
    models = []
    for endpoint in parts:
        endpoint = endpoint.strip()
        if ":" not in endpoint:
            return None
        model = endpoint.rsplit(":", 1)[1].strip()
        if not model:
            return None
        models.append(model)
    return tuple(models)


def _validate_accuracy_claim_contract(plan: dict) -> list[str]:
    """Keep declared baseline claims aligned with the registered gate.

    This contract is optional for legacy/exploratory plans.  When present, it
    makes the distinction between primary non-inferiority comparators and
    descriptive baseline rows executable at plan validation time.
    """

    contract = plan.get("accuracy_claim_contract")
    if contract is None:
        return []
    errors: list[str] = []
    if not isinstance(contract, dict):
        return ["accuracy_claim_contract must be an object"]

    def _model_list(key: str, required: bool) -> list[str]:
        value = contract.get(key)
        if value is None and not required:
            return []
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            errors.append(
                "accuracy_claim_contract.{} must be a list of non-empty strings".format(key)
            )
            return []
        result = [item.strip() for item in value]
        if len(set(result)) != len(result):
            errors.append(
                "accuracy_claim_contract.{} must not contain duplicates".format(key)
            )
        unknown = sorted(set(result) - BASELINE_CLAIM_MODELS)
        errors.extend(
            "accuracy_claim_contract.{} contains unknown baseline {}".format(key, item)
            for item in unknown
        )
        return result

    primary = _model_list("primary_noninferiority_comparators", required=True)
    secondary = _model_list("secondary_descriptive_baselines", required=True)
    overlap = sorted(set(primary) & set(secondary))
    errors.extend(
        "accuracy_claim_contract primary/secondary overlap: {}".format(item)
        for item in overlap
    )
    gate = plan.get("statistical_gate")
    if not isinstance(gate, dict):
        errors.append("accuracy_claim_contract requires a statistical_gate object")
        return errors
    comparisons = gate.get("primary_comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        errors.append(
            "accuracy_claim_contract requires non-empty statistical_gate.primary_comparisons"
        )
        return errors
    endpoint_models: set[str] = set()
    for index, item in enumerate(comparisons):
        if not isinstance(item, str) or not item.strip():
            errors.append(
                "statistical_gate.primary_comparisons[{}] must be a non-empty string".format(index)
            )
            continue
        parsed = _contrast_endpoint_models(item)
        if parsed is None:
            errors.append(
                "statistical_gate.primary_comparisons[{}] has invalid endpoint syntax".format(index)
            )
            continue
        endpoint_models.update(parsed)
    missing = sorted(set(primary) - endpoint_models)
    errors.extend(
        "accuracy_claim_contract primary comparator {} is absent from primary_comparisons".format(item)
        for item in missing
    )
    leaked = sorted(set(secondary) & endpoint_models)
    errors.extend(
        "accuracy_claim_contract descriptive baseline {} appears in primary_comparisons".format(item)
        for item in leaked
    )

    phase_models = set()
    for phase in plan.get("phases", []):
        if not isinstance(phase, dict):
            continue
        models = phase.get("models", [])
        if isinstance(models, list):
            phase_models.update(item for item in models if isinstance(item, str))
    absent_from_matrix = sorted((set(primary) | set(secondary)) - phase_models)
    errors.extend(
        "accuracy_claim_contract baseline {} is absent from phase model matrix".format(item)
        for item in absent_from_matrix
    )
    return errors


def _validate_plan(plan: dict) -> list[str]:
    errors = []
    global_config = plan.get("global", {})
    if global_config is None:
        global_config = {}
    elif not isinstance(global_config, dict):
        errors.append("global must be an object")
        global_config = {}
    if plan.get("version") not in {1, 2}:
        errors.append("plan version must be 1 or 2")
    if _is_confirmation_plan(plan):
        # Confirmation is an opt-in strict contract.  Keep its schema checks
        # at the generic plan boundary too, so a malformed contract cannot be
        # interpreted as exploratory by a downstream runner or report tool.
        if not isinstance(plan.get("statistical_gate"), dict):
            errors.append("confirmation plan requires a statistical_gate object")
        execution = plan.get("execution")
        if not isinstance(execution, dict):
            errors.append("confirmation plan requires an execution object")
        else:
            if not isinstance(execution.get("authorized"), bool):
                errors.append("confirmation plan execution.authorized must be boolean")
            execution_status = execution.get("status")
            if not isinstance(execution_status, str) or not execution_status.strip():
                errors.append(
                    "confirmation plan execution.status must be a non-empty string"
                )
            elif execution_status != execution_status.strip():
                errors.append(
                    "confirmation plan execution.status must not have surrounding whitespace"
                )
            plan_status = plan.get("status")
            if not isinstance(plan_status, str) or not plan_status.strip():
                errors.append("confirmation plan status must be a non-empty string")
            elif plan_status != plan_status.strip():
                errors.append("confirmation plan status must not have surrounding whitespace")
            if (
                isinstance(execution_status, str)
                and isinstance(plan_status, str)
                and execution_status.strip()
                and plan_status.strip()
                and execution_status != plan_status
            ):
                errors.append("confirmation plan execution.status disagrees with top-level status")
    if "data_fingerprints" in plan:
        errors.extend(_validate_data_fingerprints(plan.get("data_fingerprints")))
    for dimension in ("enc_in", "dec_in", "c_out"):
        if dimension in global_config:
            errors.append(
                f"global {dimension} is derived from dataset schema and features; remove manual override"
            )
    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        errors.append("plan must contain a non-empty phases list")
        return errors
    # High-dimensional plans carry an immutable data contract.  Validate it at
    # the same boundary as the job matrix so a run cannot proceed on a
    # same-shaped but substituted CSV or an edited quality report.
    registered_contract = plan.get("dataset_contract")
    if registered_contract is not None:
        if not isinstance(registered_contract, dict) or not registered_contract:
            errors.append("dataset_contract must be a non-empty object")
        else:
            try:
                try:
                    # Direct ``python scripts/run_training_plan.py`` puts the
                    # scripts directory on sys.path; module execution does
                    # not, so retain both supported import forms.
                    from validate_highdim_contract import validate_dataset_contract
                except ImportError:
                    from scripts.validate_highdim_contract import validate_dataset_contract
                for dataset, contract in registered_contract.items():
                    if dataset in DATASETS and _dataset_format(DATASETS[dataset]) != "csv":
                        errors.extend(
                            f"dataset_contract[{dataset}]: {message}"
                            for message in _validate_numeric_array_contract(
                                str(dataset), contract
                            )
                        )
                    else:
                        result = validate_dataset_contract(str(dataset), contract, REPO_ROOT)
                        errors.extend(
                            f"dataset_contract[{dataset}]: {message}"
                            for message in result.get("errors", [])
                        )
            except (ImportError, OSError, TypeError, ValueError) as exc:
                errors.append(f"dataset_contract validator failed: {exc}")
    seen_phase_names = set()
    for phase in phases:
        if not isinstance(phase, dict):
            errors.append(f"phase must be an object: {phase!r}")
            continue
        name = phase.get("name")
        try:
            duplicate_name = name in seen_phase_names
        except TypeError:
            duplicate_name = False
        if not name or duplicate_name:
            errors.append(f"phase name is missing or duplicated: {name!r}")
        try:
            seen_phase_names.add(name)
        except TypeError:
            # Keep validation total for malformed JSON objects; subsequent
            # checks still report the invalid phase name without crashing.
            seen_phase_names.add(repr(name))
        phase_config = _merged_phase_config(plan, phase)
        phase_datasets = phase_config.get("datasets", [])
        phase_models = phase_config.get("models", [])
        phase_seeds = phase_config.get("seeds", [])
        if not phase_datasets or not phase_models or not phase_seeds:
            errors.append(f"{name}: datasets, models and seeds must be non-empty")
        if not isinstance(phase_datasets, list):
            errors.append(f"{name}: datasets must be a list")
            phase_datasets = []
        if not isinstance(phase_models, list):
            errors.append(f"{name}: models must be a list")
            phase_models = []
        if not isinstance(phase_seeds, list):
            errors.append(f"{name}: seeds must be a list")
            phase_seeds = []
        for dataset in phase_datasets:
            if not isinstance(dataset, str) or dataset not in DATASETS:
                errors.append(f"{name}: unknown dataset {dataset}")
            else:
                if isinstance(registered_contract, dict) and dataset not in registered_contract:
                    errors.append(f"{name}: dataset {dataset!r} is missing from dataset_contract")
                spec = DATASETS[dataset]
                path = _dataset_path(dataset)
                if not path.is_file():
                    hint = spec.get("source_hint")
                    suffix = f"; place the documented file ({hint})" if hint else ""
                    errors.append(f"{name}: missing dataset file {path}{suffix}")
                    continue
                if _dataset_format(spec) != "csv":
                    try:
                        schema = _numeric_array_schema(path, spec)
                    except (OSError, TypeError, ValueError) as exc:
                        errors.append(
                            f"{name}: invalid {spec.get('format')} dataset {path}: {exc}"
                        )
                        continue
                    shape = schema.get("shape", [])
                    actual_channels = int(shape[-1]) if len(shape) == 2 else -1
                    if actual_channels != int(spec["channels"]):
                        errors.append(
                            f"{name}: {dataset} has {actual_channels} array channels, expected {spec['channels']}"
                        )
                    continue
                try:
                    header = _dataset_header(dataset)
                except (OSError, StopIteration, csv.Error) as exc:
                    errors.append(f"{name}: unreadable dataset header {path}: {exc}")
                    continue
                if not header or header[0] != spec["time_col"]:
                    errors.append(
                        f"{name}: {dataset} must start with time column {spec['time_col']!r}"
                    )
                if spec["target"] not in header:
                    errors.append(f"{name}: {dataset} target {spec['target']!r} missing from CSV header")
                actual_channels = len(header) - 1
                if actual_channels != int(spec["channels"]):
                    errors.append(
                        f"{name}: {dataset} has {actual_channels} value columns, expected {spec['channels']}"
                    )
        features = phase_config.get("features", "S")
        if features not in {"S", "M", "MS"}:
            errors.append(f"{name}: features must be one of S, M, MS")
        for dimension in ("enc_in", "dec_in", "c_out"):
            if dimension in phase:
                errors.append(
                    f"{name}: {dimension} is derived from dataset schema and features; remove manual override"
                )
        for model in phase_models:
            if not isinstance(model, str) or model not in MODEL_SPECS:
                errors.append(f"{name}: unknown model {model}")
        errors.extend(_validate_numeric_plan_config(name, phase_config))
        errors.extend(
            _validate_phase_horizon_contract(
                name,
                phase,
                list(phase_datasets),
                global_config,
            )
        )
        errors.extend(_validate_mamba_options(name, phase_config, list(phase_models)))
        if str(phase_config.get("mamba_acf_relation_representation", "encoder")).lower() == "forecast":
            for dataset in phase_datasets:
                if not isinstance(dataset, str) or dataset not in DATASETS:
                    continue
                try:
                    forecast_relation_channels, _, _ = _feature_dimensions(
                        dataset, features
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if int(forecast_relation_channels) < 2:
                    errors.append(
                        f"{name}: mamba_acf_relation_representation=forecast requires "
                        f"at least two derived input channels for dataset {dataset}"
                    )
        seeds = phase_seeds
        if isinstance(seeds, list):
            parsed_seeds = []
            for index, seed in enumerate(seeds):
                parsed_seed, seed_error = _plan_nonnegative_int(seed, f"{name}.seeds[{index}]")
                if seed_error:
                    errors.append(f"{name}: {seed_error}")
                elif parsed_seed is not None:
                    parsed_seeds.append(parsed_seed)
            if len(parsed_seeds) == len(seeds) and len(parsed_seeds) != len(set(parsed_seeds)):
                errors.append(f"{name}: seeds must be unique")
        for dataset in phase_datasets:
            if not isinstance(dataset, str) or dataset not in DATASETS:
                continue
            try:
                horizons = _phase_horizons(phase_config, dataset)
            except (TypeError, ValueError, OverflowError):
                continue
            for horizon in horizons:
                if horizon <= 0:
                    errors.append(f"{name}: horizon must be positive ({horizon})")
                if dataset == "national_illness" and horizon > 192:
                    errors.append(f"{name}: ILI horizons above 192 are disallowed")
        root = phase_config.get("output_root", f"artifacts/paper_final/{name}")
        if root is None:
            root = f"artifacts/paper_final/{name}"
        if not isinstance(root, (str, Path)):
            errors.append(f"{name}: output_root must be a path string")
            root = f"artifacts/paper_final/{name}"
        root_path = (REPO_ROOT / root).resolve()
        paper_root = (REPO_ROOT / "artifacts/paper_final").resolve()
        allowed_roots = [paper_root]
        if _plan_is_exploratory(plan):
            # Exploratory screens are kept visibly separate from publication
            # artifacts.  Retain paper_final as a backwards-compatible root
            # for older draft fixtures, but prefer artifacts/exploratory for
            # new registrations.
            allowed_roots.append((REPO_ROOT / "artifacts/exploratory").resolve())
        if not any(root == allowed or allowed in root_path.parents for allowed in allowed_roots):
            expected = "artifacts/paper_final"
            if _plan_is_exploratory(plan):
                expected += " or artifacts/exploratory"
            errors.append(f"{name}: output_root must remain under {expected}")
        allow_zero_lambda = phase_config.get("allow_zero_autocon_lambda", False)
        if not isinstance(allow_zero_lambda, bool):
            errors.append(
                f"{name}: allow_zero_autocon_lambda must be boolean when present"
            )
            allow_zero_lambda = False
        if allow_zero_lambda and _is_confirmation_plan(plan):
            errors.append(
                f"{name}: allow_zero_autocon_lambda is exploratory-only and is not allowed in confirmation plans"
            )
        if allow_zero_lambda and not _plan_is_exploratory(plan):
            errors.append(
                f"{name}: allow_zero_autocon_lambda requires a top-level exploratory or screen plan status"
            )
        if allow_zero_lambda and not _phase_has_autocon_model(phase_config):
            errors.append(
                f"{name}: allow_zero_autocon_lambda requires at least one AutoCon model"
            )
        zero_lambda_seen = False
        for key in ("autocon_lambda", "autocon_lambdas"):
            values = phase_config.get(key)
            if values is None or values == "selected":
                if allow_zero_lambda and values == "selected":
                    errors.append(
                        f"{name}: allow_zero_autocon_lambda requires an explicit zero lambda and cannot be used with selected lambdas"
                    )
                continue
            if not isinstance(values, list):
                values = [values]
            invalid = []
            for value in values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                    or (float(value) == 0 and not allow_zero_lambda)
                ):
                    invalid.append(value)
                elif float(value) == 0:
                    zero_lambda_seen = True
            if invalid:
                if allow_zero_lambda:
                    errors.append(
                        f"{name}: AutoCon lambdas must be finite non-negative numbers"
                    )
                else:
                    errors.append(
                        f"{name}: AutoCon lambdas must be finite positive numbers"
                    )
        if allow_zero_lambda and not zero_lambda_seen:
            errors.append(
                f"{name}: allow_zero_autocon_lambda requires at least one explicit zero lambda"
            )
        if phase_config.get("autocon_lambda") == "selected":
            selection_phase = phase_config.get("selection_phase", "selection")
            phase_names = {
                item.get("name")
                for item in phases
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            }
            if selection_phase not in phase_names:
                errors.append(f"{name}: selection_phase {selection_phase!r} is not registered")
            if selection_phase == name:
                errors.append(f"{name}: selection_phase cannot refer to itself")
            selection_features = next(
                (item.get("features", plan.get("global", {}).get("features", "S"))
                for item in phases
                if isinstance(item, dict) and item.get("name") == selection_phase),
                None,
            )
            phase_features = phase_config.get("features", "S")
            if selection_features is not None and selection_features != phase_features:
                errors.append(
                    f"{name}: selected phase features {selection_features!r} do not match {phase_features!r}"
                )
    errors.extend(_validate_scale_conditioning_contract(plan, phases))
    errors.extend(_validate_variable_repr_fusion_contract(plan, phases))
    errors.extend(_validate_variable_gdd_mixer_contract(plan, phases))
    errors.extend(_validate_accuracy_claim_contract(plan))
    errors.extend(_registered_job_count_errors(plan, phases))
    return errors


def _job_specs(plan: dict, phase: dict, selected: dict[str, float], allow_unresolved: bool = False) -> list[dict]:
    phase_cfg = _merged_phase_config(plan, phase)
    # Keep the null-control escape hatch phase-local and explicit.  This check
    # is duplicated here because callers such as registered-job-count
    # validation can expand jobs without relying on the top-level validator.
    allow_zero_lambda = phase_cfg.get("allow_zero_autocon_lambda", False)
    if not isinstance(allow_zero_lambda, bool):
        raise ValueError(
            f"{phase.get('name', '<unnamed>')}: allow_zero_autocon_lambda must be boolean"
        )
    if allow_zero_lambda and _is_confirmation_plan(plan):
        raise ValueError(
            f"{phase.get('name', '<unnamed>')}: allow_zero_autocon_lambda is exploratory-only"
        )
    if allow_zero_lambda and not _plan_is_exploratory(plan):
        raise ValueError(
            f"{phase.get('name', '<unnamed>')}: allow_zero_autocon_lambda requires a top-level exploratory or screen plan status"
        )
    if allow_zero_lambda and not _phase_has_autocon_model(phase):
        raise ValueError(
            f"{phase.get('name', '<unnamed>')}: allow_zero_autocon_lambda requires an AutoCon model"
        )
    seeds = [int(value) for value in phase_cfg.get("seeds", [])]
    models = phase_cfg.get("models", [])
    jobs = []
    zero_lambda_seen = False
    for dataset in phase_cfg.get("datasets", []):
        features = str(phase_cfg.get("features", "S"))
        enc_in, dec_in, c_out = _feature_dimensions(dataset, features)
        for model_name in models:
            spec = MODEL_SPECS[model_name]
            # A registered runner arm may override architecture fields while
            # retaining the same executable model.  Resolve this before
            # hashing or persisting the job so provenance remains auditable.
            model_cfg = dict(phase_cfg)
            model_cfg.update(spec.get("config_overrides", {}))
            dataset_spec = DATASETS[dataset]
            model_cfg.setdefault(
                "data_format", dataset_spec.get("format", "auto")
            )
            model_cfg.setdefault(
                "data_key", dataset_spec.get("data_key") or "data"
            )
            model_cfg.setdefault(
                "data_feature_index", dataset_spec.get("data_feature_index", 0)
            )
            model_cfg.setdefault(
                "target_index", dataset_spec.get("target_index", -1)
            )
            if dataset_spec.get("data_interval_minutes") is not None:
                model_cfg.setdefault(
                    "data_interval_minutes",
                    dataset_spec["data_interval_minutes"],
                )
            horizons = _phase_horizons(phase_cfg, dataset)
            if spec["autocon"]:
                lambda_setting = phase_cfg.get("autocon_lambda", phase_cfg.get("autocon_lambdas", [0.03]))
                if lambda_setting == "selected":
                    if allow_zero_lambda:
                        raise ValueError(
                            f"{phase.get('name', '<unnamed>')}: allow_zero_autocon_lambda cannot be used with selected lambdas"
                        )
                    if model_name not in selected:
                        if not allow_unresolved:
                            raise ValueError(f"selection file has no lambda for {model_name}")
                        # Validation can count/inspect the registered plan
                        # before phase A exists; use a clearly marked default
                        # only for the dry validation expansion.
                        raw_lambdas = [0.03]
                    else:
                        raw_lambdas = [selected[model_name]]
                elif isinstance(lambda_setting, list):
                    raw_lambdas = list(lambda_setting)
                else:
                    raw_lambdas = [lambda_setting]
                normalized_lambdas = []
                for raw_lambda in raw_lambdas:
                    if isinstance(raw_lambda, bool):
                        raise ValueError(
                            f"{phase.get('name', '<unnamed>')}: AutoCon lambdas must be finite numbers"
                        )
                    try:
                        lambda_value = float(raw_lambda)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{phase.get('name', '<unnamed>')}: AutoCon lambdas must be finite numbers"
                        ) from exc
                    if (
                        not math.isfinite(lambda_value)
                        or lambda_value < 0
                        or (lambda_value == 0 and not allow_zero_lambda)
                    ):
                        requirement = "non-negative" if allow_zero_lambda else "positive"
                        raise ValueError(
                            f"{phase.get('name', '<unnamed>')}: AutoCon lambdas must be finite {requirement} numbers"
                        )
                    normalized_lambdas.append(lambda_value)
                    if lambda_value == 0:
                        zero_lambda_seen = True
                lambdas = normalized_lambdas
            else:
                lambdas = [None]
            for horizon in horizons:
                for seed in seeds:
                    for lambda_value in lambdas:
                        jobs.append({
                            "phase": phase["name"], "dataset": dataset, "model": model_name,
                            "pred_len": horizon, "seed": seed, "autocon_lambda": lambda_value,
                            "config": {**model_cfg, "features": features, "enc_in": enc_in,
                                       "dec_in": dec_in, "c_out": c_out},
                        })
    if allow_zero_lambda and not zero_lambda_seen:
        raise ValueError(
            f"{phase.get('name', '<unnamed>')}: allow_zero_autocon_lambda requires an explicit zero lambda"
        )
    return jobs


def _lambda_token(value: float | None) -> str:
    if value is None:
        return "none"
    return ("%.6g" % value).replace("-", "m").replace(".", "p")


def _config_hash(job: dict, provenance_id: str | None = None) -> str:
    payload = {
        "phase": job["phase"], "dataset": job["dataset"], "model": job["model"],
        "pred_len": job["pred_len"], "seed": job["seed"],
        "autocon_lambda": job["autocon_lambda"], "config": job["config"],
        "provenance_id": provenance_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _legacy_job_id(job: dict) -> str:
    """ID format used by the runner before configuration hashes were added."""
    return "{}_{}_{}_h{}_s{}_l{}".format(
        job["phase"], job["dataset"], job["model"], job["pred_len"],
        job["seed"], _lambda_token(job["autocon_lambda"]),
    )


def _cli_int_values(value, default):
    """Normalize scalar/list plan values before expanding an argparse list."""

    if value is None:
        value = default
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        values = [item for item in text.replace(",", " ").split() if item]
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        values = [value]
    else:
        values = list(value)
    if not values:
        raise ValueError("Mamba list argument cannot be empty")
    return [str(item) for item in values]


def _command(
    job: dict,
    roots: dict[str, Path],
    provenance_id: str | None = None,
    run_script: Path | None = None,
) -> tuple[str, list[str]]:
    cfg = job["config"]
    data = DATASETS[job["dataset"]]
    spec = MODEL_SPECS[job["model"]]
    config_hash = _config_hash(job, provenance_id)
    job_id = "{}_{}_{}_h{}_s{}_l{}_c{}".format(
        job["phase"], job["dataset"], job["model"], job["pred_len"],
        job["seed"], _lambda_token(job["autocon_lambda"]), config_hash,
    )
    features = cfg["features"]
    enc_in = int(cfg["enc_in"])
    dec_in = int(cfg["dec_in"])
    c_out = int(cfg["c_out"])
    # Older snapshots may persist an explicitly null local width.  The model
    # contract treats that value as "use d_model", so resolve it before
    # serialising an argparse integer rather than emitting the literal
    # string ``None``.
    local_width = cfg.get("mamba_local_width")
    if local_width is None:
        local_width = cfg.get("d_model", 32)
    command = [
        sys.executable, "-u", str(run_script or (REPO_ROOT / "run.py")), "--model_id", "plan_" + job_id,
        "--model", spec["model"], "--data", job["dataset"],
        "--root_path", data["root"], "--data_path", data["path"],
        "--features", features, "--target", data["target"], "--freq", data["freq"],
        "--data_format", str(cfg.get("data_format", data.get("format", "auto"))),
        "--data_key", str(cfg.get("data_key", data.get("data_key") or "data")),
        "--data_feature_index", str(cfg.get("data_feature_index", data.get("data_feature_index", 0))),
        "--target_index", str(cfg.get("target_index", data.get("target_index", -1))),
        "--data_start_timestamp", str(cfg.get("data_start_timestamp", data.get("data_start_timestamp", "2000-01-01"))),
        "--seq_len", str(cfg.get("seq_len", 96)), "--label_len", str(cfg.get("label_len", 48)),
        "--pred_len", str(job["pred_len"]), "--enc_in", str(enc_in), "--dec_in", str(dec_in),
        "--c_out", str(c_out), "--d_model", str(cfg.get("d_model", 32)),
        "--d_ff", str(cfg.get("d_ff", 32)), "--e_layers", str(cfg.get("e_layers", 1)),
        "--n_heads", str(cfg.get("n_heads", 8)), "--d_layers", str(cfg.get("d_layers", 1)),
        "--factor", str(cfg.get("factor", 1)), "--top_k", str(cfg.get("top_k", 5)),
        "--num_kernels", str(cfg.get("num_kernels", 6)),
        "--moving_avg", str(cfg.get("moving_avg", 25)),
        "--timemixer_down_sampling_window", str(cfg.get("timemixer_down_sampling_window", 2)),
        "--timemixer_down_sampling_layers", str(cfg.get("timemixer_down_sampling_layers", 1)),
        "--timemixer_decomp_method", str(cfg.get("timemixer_decomp_method", "moving_avg")),
        "--patch_len", str(cfg.get("patch_len", 16)),
        "--patch_stride", str(cfg.get("patch_stride", 8)),
        "--mamba_ts_variant",
        (
            "v16" if str(spec["model"]).startswith("MambaTSv16")
            else "v15" if str(spec["model"]).startswith("MambaTSv15")
            else "v14" if str(spec["model"]).startswith("MambaTSv14")
            else "v13" if str(spec["model"]).startswith("MambaTSv13")
            else "v12" if str(spec["model"]).startswith("MambaTSv12")
            else "v11" if str(spec["model"]).startswith("MambaTSv11")
            else "v10" if str(spec["model"]).startswith("MambaTSv10")
            else "v9" if str(spec["model"]).startswith("MambaTSv9")
            else "v8" if str(spec["model"]).startswith("MambaTSv8")
            else "v7" if str(spec["model"]).startswith("MambaTSv7")
            else "v7" if str(spec["model"]).startswith("MambaTSv7")
            else "v6" if str(spec["model"]).startswith("MambaTSv6")
            else "v5" if str(spec["model"]).startswith("MambaTSv5")
            else "v4" if str(spec["model"]).startswith("MambaTSv4")
            else "v3" if str(spec["model"]).startswith("MambaTSv3")
            else "v2" if str(spec["model"]).startswith("MambaTSv2")
            else "v1"
        ),
        "--mamba_ts_context_rank", str(cfg.get("mamba_ts_context_rank", 16)),
        "--mamba_ts_mixer_type", str(cfg.get("mamba_ts_mixer_type", "mamba1")),
        "--mamba_ts_vpt_solver", str(cfg.get("mamba_ts_vpt_solver", "greedy")),
        "--mamba_ts_delta_multipliers", *[str(value) for value in cfg.get("mamba_ts_delta_multipliers", [1.0, 2.0])],
        "--mamba_ts_dt_min", str(cfg.get("mamba_ts_dt_min", 0.001)),
        "--mamba_ts_dt_max", str(cfg.get("mamba_ts_dt_max", 0.1)),
        "--mamba_ts_global_channel_mixer_rank",
        str(cfg.get("mamba_ts_global_channel_mixer_rank", 4)),
        "--mamba_ts_spectral_top_k", str(cfg.get("mamba_ts_spectral_top_k", 4)),
        "--mamba_ts_residual_init", str(cfg.get("mamba_ts_residual_init", 0.0)),
        "--dropout", str(cfg.get("dropout", 0.1)),
        "--embed", str(cfg.get("embed", "timeF")),
        "--activation", str(cfg.get("activation", "gelu")),
        "--wo_freq", str(cfg.get("wo_freq", "")),
        "--mamba_d_state", str(cfg.get("mamba_d_state", 16)),
        "--mamba_d_conv", str(cfg.get("mamba_d_conv", 4)),
        "--mamba_expand", str(cfg.get("mamba_expand", 2)),
        "--mamba_headdim", str(cfg.get("mamba_headdim", 16)),
        "--mamba_tm_n1", str(cfg.get("mamba_tm_n1", 64)),
        "--mamba_tm_n2", str(cfg.get("mamba_tm_n2", 128)),
        "--mamba_tm_repr_width", str(cfg.get("mamba_tm_repr_width", 64)),
        "--mamba_tm_residual",
        _bool(_config_bool(cfg, "mamba_tm_residual", True)),
        "--mamba_bipatch_token_mode", str(cfg.get("mamba_bipatch_token_mode", "mixing")),
        "--mamba_scales", *_cli_int_values(cfg.get("mamba_scales"), [1]),
        "--mamba_token_stride", str(cfg.get("mamba_token_stride", 1)),
        "--mamba_temporal_rank", str(cfg.get("mamba_temporal_rank", 16)),
        "--mamba_channel_gate_rank", str(cfg.get("mamba_channel_gate_rank", 8)),
        "--mamba_data_dependent_channel_mixer_rank",
        str(cfg.get("mamba_data_dependent_channel_mixer_rank", 8)),
        "--mamba_variable_gdd_mixer_rank",
        str(cfg.get("mamba_variable_gdd_mixer_rank", 8)),
        "--mamba_variable_gdd_init_scale",
        str(cfg.get("mamba_variable_gdd_init_scale", 0.0)),
        "--mamba_variable_mixer_depth", str(cfg.get("mamba_variable_mixer_depth", 1)),
        "--mamba_variable_group_size", str(cfg.get("mamba_variable_group_size", 1)),
        "--mamba_acf_channel_order", str(cfg.get("mamba_acf_channel_order", "input")),
        "--mamba_patch_mixer_depth", str(cfg.get("mamba_patch_mixer_depth", 1)),
        "--mamba_patch_len", str(cfg.get("mamba_patch_len", 16)),
        "--mamba_patch_stride", str(cfg.get("mamba_patch_stride", 8)),
        "--mamba_patch_order", str(cfg.get("mamba_patch_order", "time_major")),
        "--mamba_local_patch_len", str(cfg.get("mamba_local_patch_len", 16)),
        "--mamba_local_patch_stride", str(cfg.get("mamba_local_patch_stride", 8)),
        "--mamba_local_width", str(local_width),
        "--mamba_local_rank", str(cfg.get("mamba_local_rank", 16)),
        "--mamba_local_depth", str(cfg.get("mamba_local_depth", 1)),
        "--mamba_acf_lags", *_cli_int_values(
            cfg.get("mamba_acf_lags"), [24, 168, 336]
        ),
        "--mamba_acf_channel_lag_top_k",
        str(cfg.get("mamba_acf_channel_lag_top_k", 2)),
        "--mamba_acf_lag_mode", str(cfg.get("mamba_acf_lag_mode", "fixed")),
        "--mamba_acf_max_lag", str(cfg.get("mamba_acf_max_lag", 720)),
        "--mamba_acf_top_k", str(cfg.get("mamba_acf_top_k", 8)),
        "--mamba_acf_min_lag", str(cfg.get("mamba_acf_min_lag", 2)),
        "--mamba_acf_min_abs", str(cfg.get("mamba_acf_min_abs", 0.1)),
        "--mamba_acf_confidence_z", str(cfg.get("mamba_acf_confidence_z", 1.96)),
        "--mamba_acf_min_separation", str(cfg.get("mamba_acf_min_separation", 2)),
        "--mamba_acf_detrend", str(cfg.get("mamba_acf_detrend", "mean")),
        "--mamba_acf_chunk", str(cfg.get("mamba_acf_chunk", 64)),
        "--mamba_use_acf_memory", _bool(_config_bool(cfg, "mamba_use_acf_memory")),
        "--mamba_dynamic_acf", _bool(_config_bool(cfg, "mamba_dynamic_acf")),
        "--mamba_acf_temperature", str(cfg.get("mamba_acf_temperature", 4.0)),
        "--mamba_acf_local_min_pairs", str(cfg.get("mamba_acf_local_min_pairs", 16)),
        "--mamba_acf_step_reliability", _bool(_config_bool(cfg, "mamba_acf_step_reliability")),
        "--mamba_acf_reliability_interpolation", _bool(_config_bool(cfg, "mamba_acf_reliability_interpolation")),
        "--mamba_collect_eval_acf_diagnostics",
        _bool(_config_bool(cfg, "mamba_collect_eval_acf_diagnostics")),
        "--mamba_acf_memory_gate_mode", str(cfg.get("mamba_acf_memory_gate_mode", "hidden")),
        "--mamba_acf_horizon_calibration_rank",
        str(cfg.get("mamba_acf_horizon_calibration_rank", 2)),
        "--mamba_acf_relation_batching", str(cfg.get("mamba_acf_relation_batching", "lag_pairs")),
        "--mamba_acf_relation_lag_balance", str(cfg.get("mamba_acf_relation_lag_balance", "random")),
        "--mamba_acf_relation_channel_reduce", str(cfg.get("mamba_acf_relation_channel_reduce", "mean")),
        "--mamba_acf_relation_scope", str(cfg.get("mamba_acf_relation_scope", "both")),
        "--mamba_acf_relation_representation",
        str(cfg.get("mamba_acf_relation_representation", "encoder")),
        "--mamba_acf_relation_lag_tolerance", str(cfg.get("mamba_acf_relation_lag_tolerance", 0)),
        "--mamba_acf_relation_min_lag", str(cfg.get("mamba_acf_relation_min_lag", 0)),
        "--mamba_acf_relation_target_mode", str(cfg.get("mamba_acf_relation_target_mode", "hard_selected")),
        "--mamba_acf_relation_soft_min_abs", str(cfg.get("mamba_acf_relation_soft_min_abs", 0.1)),
        "--mamba_acf_relation_diagnostic_min_abs", str(cfg.get("mamba_acf_relation_diagnostic_min_abs", 0.1)),
        "--mamba_acf_relation_selected_lag_boost", str(cfg.get("mamba_acf_relation_selected_lag_boost", 0.0)),
        "--AutoCon_multiscales", *[str(value) for value in (
            cfg.get("autocon_scale", 96) if isinstance(cfg.get("autocon_scale", 96), list)
            else [cfg.get("autocon_scale", 96)]
        )],
        # Accept the historical camel-case spelling used by older plans, but
        # give the canonical lower-case key precedence when both are present.
        "--AutoCon_wnorm", str(
            cfg.get("autocon_wnorm", cfg.get("AutoCon_wnorm", "LastVal"))
        ),
        "--AutoCon_warmup_epochs", str(cfg.get("autocon_warmup", 1)),
        "--AutoCon_acf_source", resolve_acf_source(cfg),
        "--batch_size", str(cfg.get("batch_size", 256)), "--train_epochs", str(cfg.get("epochs", 20)),
        "--patience", str(cfg.get("patience", 5)), "--learning_rate", str(cfg.get("learning_rate", 1e-3)),
        "--lradj", str(cfg.get("lradj", "type1")),
        "--min_learning_rate", str(cfg.get("min_learning_rate", 0.0)),
        "--weight_decay", str(cfg.get("weight_decay", 1e-5)), "--num_workers", str(cfg.get("num_workers", 0)),
        "--train_ratio", str(cfg.get("train_ratio", 0.6)),
        "--itr", "1", "--seed", str(job["seed"]), "--des", job["phase"],
        "--checkpoints", str(roots["checkpoints"]), "--results_dir", str(roots["results"]),
        "--test_results_dir", str(roots["test"]), "--result_log", str(roots["result_log"]),
        "--use_gpu", _bool(bool(cfg.get("use_gpu", True))), "--no_save",
        "--plan_job_id", job_id,
    ]
    if _config_bool(cfg, "mamba_ts_graph_spatial", False):
        graph_path = cfg.get("mamba_ts_graph_path") or data.get("graph_path")
        if not graph_path:
            raise ValueError(
                "mamba_ts_graph_spatial requires a registered graph_path"
            )
        resolved_graph_path = Path(graph_path)
        if not resolved_graph_path.is_absolute():
            resolved_graph_path = REPO_ROOT / data["root"] / resolved_graph_path
        graph_hash = (
            cfg.get("mamba_ts_graph_expected_sha256")
            or data.get("graph_sha256", "")
        )
        command.extend([
            "--mamba_ts_graph_spatial",
            "--mamba_ts_graph_path", str(resolved_graph_path),
            "--mamba_ts_graph_expected_sha256", str(graph_hash),
        ])
    if _config_bool(cfg, "timemixer_channel_independence", False):
        command.append("--timemixer_channel_independence")
    if (
        str(job["model"]).startswith("MambaTS")
        and _config_bool(cfg, "mamba_ts_repr_forecast_fusion", False)
    ):
        command.append("--mamba_ts_repr_forecast_fusion")
    if _config_bool(cfg, "mamba2_repr_forecast_fusion", False):
        command.append("--mamba2_repr_forecast_fusion")
    if _config_bool(cfg, "mamba_ts_repr_forecast_channelwise", False):
        command.append("--mamba_ts_repr_forecast_channelwise")
    if _config_bool(cfg, "mamba_ts_channel_adapter", False):
        command.extend([
            "--mamba_ts_channel_adapter",
            "--mamba_ts_channel_adapter_rank",
            str(cfg.get("mamba_ts_channel_adapter_rank", 8)),
        ])
    if _config_bool(cfg, "mamba_ts_revin_affine", False):
        command.append("--mamba_ts_revin_affine")
    if _config_bool(cfg, "mamba_ts_repr_linear_expert", False):
        command.extend([
            "--mamba_ts_repr_linear_expert",
            "--mamba_ts_repr_linear_expert_mode",
            str(cfg.get("mamba_ts_repr_linear_expert_mode", "residual")),
            "--mamba_ts_repr_linear_blend_bias",
            str(cfg.get("mamba_ts_repr_linear_blend_bias", -4.0)),
            "--mamba_ts_repr_linear_expert_rank",
            str(cfg.get("mamba_ts_repr_linear_expert_rank", 16)),
        ])
    if _config_bool(cfg, "mamba_ts_variable_permutation", False):
        command.append("--mamba_ts_variable_permutation")
    if _config_bool(cfg, "mamba_ts_acf_memory_selector", False):
        command.append("--mamba_ts_acf_memory_selector")
    if _config_bool(cfg, "mamba_ts_acf_memory_channel_selector", False):
        command.append("--mamba_ts_acf_memory_channel_selector")
    if _config_bool(cfg, "mamba_ts_forecast_acf_consistency", False):
        command.extend([
            "--mamba_ts_forecast_acf_consistency",
            "--mamba_ts_forecast_acf_weight",
            str(cfg.get("mamba_ts_forecast_acf_weight", 0.0)),
            "--mamba_ts_forecast_acf_top_k",
            str(cfg.get("mamba_ts_forecast_acf_top_k", 4)),
            "--mamba_ts_forecast_acf_min_lag",
            str(cfg.get("mamba_ts_forecast_acf_min_lag", 2)),
            "--mamba_ts_forecast_acf_min_abs",
            str(cfg.get("mamba_ts_forecast_acf_min_abs", 0.10)),
        ])
    if _config_bool(cfg, "mamba_acf_forecast_pair_consistency", False):
        command.extend([
            "--mamba_acf_forecast_pair_consistency",
            "--mamba_acf_forecast_pair_weight",
            str(cfg.get("mamba_acf_forecast_pair_weight", 0.0)),
        ])
    interval = cfg.get("data_interval_minutes", data.get("data_interval_minutes"))
    if interval is not None and _dataset_format(data) != "csv":
        command.extend(["--data_interval_minutes", str(interval)])
    if spec["autocon"]:
        command.extend(["--AutoCon", "--AutoCon_lambda", str(job["autocon_lambda"])])
        if _config_bool(cfg, "mamba_acf_relation_loss"):
            command.append("--mamba_acf_relation_loss")
            if _config_bool(cfg, "mamba_acf_relation_lag_mask"):
                command.append("--mamba_acf_relation_lag_mask")
            if _config_bool(cfg, "mamba_acf_relation_active_anchor_normalization"):
                command.append("--mamba_acf_relation_active_anchor_normalization")
            command.extend([
                "--AutoCon_temperature", str(cfg.get("AutoCon_temperature", 0.2)),
                "--AutoCon_relation_temperature", str(cfg.get("AutoCon_relation_temperature", 0.25)),
                "--AutoCon_sign_weight", str(cfg.get("AutoCon_sign_weight", 0.25)),
                "--AutoCon_sign_margin", str(cfg.get("AutoCon_sign_margin", 0.1)),
            ])
    if _config_bool(cfg, "save_predictions"):
        command = [item for item in command if item != "--no_save"]
    if _config_bool(cfg, "skip_test"):
        command.append("--skip_test")
    if _config_bool(cfg, "individual"):
        command.append("--individual")
    if _config_bool(cfg, "drop_last_train"):
        command.append("--drop_last_train")
    if _config_bool(cfg, "use_amp"):
        command.append("--use_amp")
    if _config_bool(cfg, "mamba_bidirectional"):
        command.append("--mamba_bidirectional")
    if _config_bool(cfg, "mamba_channel_independent"):
        command.append("--mamba_channel_independent")
    if _config_bool(cfg, "mamba_tm_channel_independent"):
        command.append("--mamba_tm_channel_independent")
    if _config_bool(cfg, "mamba_bipatch_repr_gate"):
        command.append("--mamba_bipatch_repr_gate")
    if _config_bool(cfg, "mamba_channel_gate"):
        command.append("--mamba_channel_gate")
    if _config_bool(cfg, "mamba_data_dependent_channel_mixer"):
        command.append("--mamba_data_dependent_channel_mixer")
    if _config_bool(cfg, "mamba_variable_gdd_mixer"):
        command.append("--mamba_variable_gdd_mixer")
    if _config_bool(cfg, "mamba_ts_global_channel_mixer"):
        command.append("--mamba_ts_global_channel_mixer")
    if _config_bool(cfg, "mamba_acf_scale_conditioning"):
        command.append("--mamba_acf_scale_conditioning")
    if _config_bool(cfg, "mamba_acf_local_agreement"):
        command.append("--mamba_acf_local_agreement")
    if _config_bool(cfg, "mamba_acf_channel_gate"):
        command.append("--mamba_acf_channel_gate")
    if _config_bool(cfg, "mamba_causal_embedding"):
        command.append("--mamba_causal_embedding")
    if _config_bool(cfg, "mamba_require_strict_representation_causality"):
        command.append("--mamba_require_strict_representation_causality")
    if _config_bool(cfg, "mamba_variable_mixer"):
        command.append("--mamba_variable_mixer")
    if _config_bool(cfg, "mamba_variable_repr_fusion"):
        command.append("--mamba_variable_repr_fusion")
    if _config_bool(cfg, "mamba_variable_acf_memory_selector"):
        command.append("--mamba_variable_acf_memory_selector")
    if _config_bool(cfg, "mamba_acf_channel_conditioning"):
        command.append("--mamba_acf_channel_conditioning")
    if _config_bool(cfg, "mamba_acf_repr_fusion"):
        command.append("--mamba_acf_repr_fusion")
    if _config_bool(cfg, "mamba_acf_horizon_calibration"):
        command.append("--mamba_acf_horizon_calibration")
    if _config_bool(cfg, "mamba_acf_anchor_copy"):
        command.append("--mamba_acf_anchor_copy")
    if _config_bool(cfg, "mamba_acf_channel_lag_bank"):
        command.append("--mamba_acf_channel_lag_bank")
    if _config_bool(cfg, "mamba_variable_permutation"):
        command.append("--mamba_variable_permutation")
    if _config_bool(cfg, "mamba_patch_mixer"):
        command.append("--mamba_patch_mixer")
    if _config_bool(cfg, "mamba_patch_permutation"):
        command.append("--mamba_patch_permutation")
    if _config_bool(cfg, "mamba_local_mixer"):
        command.append("--mamba_local_mixer")
    if _config_bool(cfg, "mamba_low_rank_head"):
        command.append("--mamba_low_rank_head")
    if _config_bool(cfg, "stream_metrics"):
        command.append("--stream_metrics")
        if cfg.get("origin_stride") is not None:
            command.extend(["--origin_stride", str(cfg["origin_stride"])])
        command.extend(["--stream_sample_windows", str(cfg.get("stream_sample_windows", 4))])
    return job_id, command


def _read_manifest(path: Path) -> dict[str, dict]:
    records = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A killed process can leave a partial final JSONL line;
                    # let the corresponding job retry on the next run.
                    continue
                if "job_id" in record:
                    records[record["job_id"]] = record
    return records


def _artifact_exists(record: dict) -> bool:
    expected = record.get("expected_artifact")
    if not expected or not Path(expected).exists():
        return False
    artifact_path = Path(expected)
    command = record.get("command", [])
    # Training is not considered resumable until its checkpoint and validation
    # trajectory are present. Confirmatory jobs additionally require the saved
    # predictions used by origin-level sensitivity analysis.
    checkpoint_root = None
    if "--checkpoints" in command:
        checkpoint_root = Path(command[command.index("--checkpoints") + 1])
    if checkpoint_root is not None:
        checkpoint_dir = checkpoint_root / artifact_path.parent.name
        if not (checkpoint_dir / "checkpoint.pth").is_file() or not (checkpoint_dir / "loss_logs.json").is_file():
            return False
    if "--stream_metrics" in command:
        if not (artifact_path.parent / "stream_metrics.json").is_file():
            return False
        if not (artifact_path.parent / "stream_samples.npz").is_file():
            return False
    if "--no_save" not in command:
        if not (artifact_path.parent / "pred.npy").is_file() or not (artifact_path.parent / "true.npy").is_file():
            return False
    # A resumable result must still match the hashes captured when the job
    # completed.  Merely checking that files exist can silently accept an
    # edited or partially replaced artifact and defer the failure until report
    # generation.
    expected_hashes = record.get("artifact_hashes")
    if expected_hashes:
        checkpoint_dir = checkpoint_root / artifact_path.parent.name if checkpoint_root is not None else None
        if _artifact_hashes(artifact_path.parent, checkpoint_dir) != expected_hashes:
            return False
    return True


def _normalized_command(command: list[str]) -> list[str]:
    """Remove identifiers that intentionally differ across runner versions."""
    normalized = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in {"--model_id", "--plan_job_id"}:
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def _legacy_record_matches(job: dict, current_command: list[str], record: dict) -> bool:
    """Allow old-format resume only when every training argument is identical."""
    old_command = record.get("command")
    if not isinstance(old_command, list):
        return False
    return _normalized_command(old_command) == _normalized_command(current_command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/final_training_plan.json"))
    parser.add_argument("--phase", action="append", help="phase name; repeat to run selected phases")
    parser.add_argument("--selection_file", type=Path, default=None,
                        help="JSON produced by select_hyperparameters.py")
    parser.add_argument("--selection_provenance", type=Path, default=None,
                        help="verified selection provenance sidecar for selected-lambda phases")
    parser.add_argument(
        "--launch_sidecar", type=Path, default=None,
        help="approved confirmation launch sidecar binding the exact plan hash",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="run at most N jobs after deterministic shuffling")
    parser.add_argument("--no_resume", action="store_true", help="rerun jobs marked successful in the manifest")
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else REPO_ROOT / args.plan
    plan = _load_json(plan_path)
    try:
        launch_authorization = _validate_launch_authorization(
            plan,
            plan_path,
            args.launch_sidecar,
            enforce=not args.validate_only and not args.dry_run,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"PLAN ERROR: launch authorization: {exc}", file=sys.stderr)
        return 2
    errors = _validate_plan(plan)
    if errors:
        for error in errors:
            print("PLAN ERROR:", error, file=sys.stderr)
        return 2
    selected_path = args.selection_file
    if selected_path is not None and not selected_path.is_absolute():
        selected_path = REPO_ROOT / selected_path
    selected_provenance_path = args.selection_provenance
    if selected_provenance_path is not None and not selected_provenance_path.is_absolute():
        selected_provenance_path = REPO_ROOT / selected_provenance_path
    phases = plan["phases"]
    if args.phase:
        wanted = set(args.phase)
        phases = [phase for phase in phases if phase["name"] in wanted]
        missing = wanted - {phase["name"] for phase in phases}
        if missing:
            print("PLAN ERROR: unknown phase(s): " + ", ".join(sorted(missing)), file=sys.stderr)
            return 2

    selected_phases = [phase for phase in phases if phase.get("autocon_lambda") == "selected"]
    # A real run must name one phase explicitly.  This prevents an S selection
    # file from being silently reused for an independent M phase.
    if selected_phases and not args.validate_only and not args.phase:
        print("PLAN ERROR: select one phase explicitly when selected lambdas are required", file=sys.stderr)
        return 2
    selection_bindings = {}
    if selected_phases and (not args.validate_only or selected_path is not None):
        if selected_path is not None and selected_provenance_path is None:
            print("PLAN ERROR: selected-lambda runs require --selection_provenance", file=sys.stderr)
            return 2
        for phase in selected_phases:
            try:
                selection_bindings[phase["name"]] = _validate_selection_binding(
                    selected_path, phase, plan, plan_path, selected_provenance_path
                )
            except ValueError as exc:
                print(f"PLAN ERROR: {exc}", file=sys.stderr)
                return 2
    selected = _selected_lambdas(selected_path)

    all_jobs = []
    for phase in phases:
        all_jobs.extend(_job_specs(plan, phase, selected, allow_unresolved=args.validate_only))
    order_seed = int(plan.get("job_order_seed", 1729))
    random.Random(order_seed).shuffle(all_jobs)
    if args.limit:
        all_jobs = all_jobs[:args.limit]
    print(f"Validated {len(all_jobs)} jobs (order_seed={order_seed})")
    if args.validate_only:
        return 0

    environment = _environment()
    failed_jobs = 0
    for phase in phases:
        phase_jobs = [job for job in all_jobs if job["phase"] == phase["name"]]
        if not phase_jobs:
            continue
        phase_root = REPO_ROOT / phase.get("output_root", f"artifacts/paper_final/{phase['name']}")
        selection_for_phase = selected_path if phase.get("autocon_lambda") == "selected" else None
        if args.dry_run:
            provenance = {
                "provenance_id": "dry-run",
                "source_snapshot": str(REPO_ROOT.resolve()),
                "source_tree_sha256": "dry-run",
            }
        else:
            provenance = _phase_provenance(
                phase_root,
                phase,
                plan_path,
                selection_for_phase,
                selection_bindings.get(phase["name"]),
                launch_authorization,
            )
        provenance_id = provenance["provenance_id"]
        run_script = Path(provenance["source_snapshot"]) / "run.py"
        roots = {
            "results": (phase_root / "results").resolve(),
            "checkpoints": (phase_root / "checkpoints").resolve(),
            "test": (phase_root / "test_results").resolve(),
            "logs": (phase_root / "logs").resolve(),
            "result_log": (phase_root / "results.txt").resolve(),
        }
        manifest_path = roots["logs"] / "job_manifest.jsonl"
        if not args.dry_run:
            for path in roots.values():
                if path.suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    path.mkdir(parents=True, exist_ok=True)
            environment_path = phase_root / "environment.json"
            if not environment_path.exists():
                environment_payload = {
                    "plan": str(plan_path), "phase": phase["name"],
                    "environment": environment, "provenance_id": provenance_id,
                    "provenance": str((phase_root / "provenance.json").resolve()),
                    "cpu_thread_environment": _cpu_thread_environment(
                        _merged_phase_config(plan, phase)
                    ),
                    "dataset_hashes": {name: _sha256(REPO_ROOT / DATASETS[name]["root"] / DATASETS[name]["path"])
                                       for name in phase.get("datasets", [])},
                }
                with environment_path.open("x", encoding="utf-8") as handle:
                    json.dump(environment_payload, handle, indent=2)
        previous = _read_manifest(manifest_path)
        mode = "a"
        manifest_handle = None if args.dry_run else manifest_path.open(mode, encoding="utf-8")
        try:
            for index, job in enumerate(phase_jobs, start=1):
                job_id, command = _command(
                    job, roots, provenance_id=provenance_id,
                    run_script=run_script,
                )
                log_path = roots["logs"] / f"{job_id}.log"
                prior = previous.get(job_id)
                if prior is None:
                    legacy = previous.get(_legacy_job_id(job))
                    if legacy and _legacy_record_matches(job, command, legacy):
                        prior = legacy
                if not args.no_resume and prior and prior.get("status") == "ok" and _artifact_exists(prior):
                    print(f"[{index}/{len(phase_jobs)}] skip {job_id} (resume)", flush=True)
                    continue
                record = {
                    "job_id": job_id, "phase": job["phase"], "dataset": job["dataset"],
                    "model": job["model"], "pred_len": job["pred_len"], "seed": job["seed"],
                    "autocon_lambda": job["autocon_lambda"], "command": command,
                    "features": job["config"]["features"],
                    "enc_in": int(job["config"]["enc_in"]),
                    "dec_in": int(job["config"]["dec_in"]),
                    "c_out": int(job["config"]["c_out"]),
                    "command_sha256": _command_hash(command),
                    "config_hash": _config_hash(job, provenance_id),
                    "provenance_id": provenance_id,
                    "source_tree_sha256_before": provenance["source_tree_sha256"],
                    "source_tree_sha256_after": None,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                    "log": str(log_path), "status": "planned",
                }
                thread_environment = _cpu_thread_environment(job["config"])
                if thread_environment:
                    record["environment_overrides"] = thread_environment
                if args.dry_run:
                    print(json.dumps(record, ensure_ascii=True))
                    continue
                print(f"[{index}/{len(phase_jobs)}] run {job_id}", flush=True)
                started = time.time()
                record["started_at_utc"] = _utc_now()
                with log_path.open("w", encoding="utf-8") as output:
                    child_environment = os.environ.copy()
                    child_environment.update(thread_environment)
                    completed = subprocess.run(command, cwd=REPO_ROOT, stdout=output,
                                               stderr=subprocess.STDOUT, text=True,
                                               env=child_environment)
                _, source_hash_after = _hash_file_map(Path(provenance["source_snapshot"]))
                record["source_tree_sha256_after"] = source_hash_after
                record["finished_at_utc"] = _utc_now()
                if source_hash_after != provenance["source_tree_sha256"]:
                    record["status"] = "invalid_provenance"
                    failed_jobs += 1
                    print(f"  source snapshot changed; see {log_path}", flush=True)
                setting_dirs = []
                for candidate in sorted(roots["results"].glob("*")):
                    config_path = candidate / "config.json"
                    if not config_path.is_file():
                        continue
                    try:
                        with config_path.open(encoding="utf-8") as config_handle:
                            config = json.load(config_handle)
                    except (OSError, ValueError):
                        continue
                    if config.get("plan_job_id") == job_id:
                        setting_dirs.append(candidate)
                expected = None
                if setting_dirs:
                    setting_dir = setting_dirs[0]
                    expected_name = "metrics.npy" if not bool(job["config"].get("skip_test", False)) else "training_complete.json"
                    expected = str(setting_dir / expected_name)
                record.update({"returncode": completed.returncode,
                               "elapsed_sec": round(time.time() - started, 3),
                               "expected_artifact": expected,
                               "status": record.get("status") if record.get("status") == "invalid_provenance" else
                               ("ok" if completed.returncode == 0 and expected and Path(expected).exists() else "failed")})
                if setting_dirs:
                    record["artifact_hashes"] = _artifact_hashes(
                        setting_dirs[0], roots["checkpoints"] / setting_dirs[0].name
                    )
                    record["artifact_hashes_captured_at_utc"] = _utc_now()
                if record["status"] != "ok" and record["status"] != "invalid_provenance":
                    failed_jobs += 1
                if setting_dirs:
                    loss_log_path = roots["checkpoints"] / setting_dirs[0].name / "loss_logs.json"
                    if loss_log_path.is_file():
                        try:
                            with loss_log_path.open(encoding="utf-8") as loss_handle:
                                loss_log = json.load(loss_handle)
                            record["best_epoch"] = loss_log.get("best_epoch")
                            record["epochs_ran"] = loss_log.get("epochs_ran", len(loss_log.get("val_loss", [])))
                        except (OSError, ValueError):
                            record["loss_log_error"] = str(loss_log_path)
                manifest_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                manifest_handle.flush()
                previous[job_id] = record
                if record["status"] != "ok":
                    print(f"  failed; see {log_path}", flush=True)
        finally:
            if manifest_handle is not None:
                manifest_handle.close()
    print(f"Training plan phase(s) complete; failed_jobs={failed_jobs}")
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    raise SystemExit(main())
