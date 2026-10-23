#!/usr/bin/env python
"""Collect architecture and steady-state inference resource measurements.

This is intentionally an analysis-side tool.  It never invokes ``run.py`` or
loads a dataset.  When a phase-validation file is supplied, selected configs
are checked against the immutable validation manifest and their phase source
snapshot is used for model construction.  GPU work is opt-in through
``--run-benchmark`` so inspecting a completed phase cannot accidentally
compete with training for a device.

The timing protocol measures device-resident synthetic forward passes after a
fixed warm-up.  It reports resource throughput only; synthetic input windows
and timing iterations are not treated as independent statistical observations.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.acf_relation_contract import (
    mamba_acf_prior_required,
    resolve_acf_source,
)
from utils.mamba_causality_contract import strict_representation_causality_errors
from utils.model_identity import canonical_model, executable_model

SCHEMA_VERSION = 1

# The confirmatory plan uses these nine public identifiers.  AutoConNet uses
# its shared module for both objective variants; Mamba models similarly share
# a module while ``AutoCon`` controls their returned representation.
MODEL_MODULES = {
    "DLinear": "DLinear",
    "GraphDLinear": "GraphDLinear",
    "STGCN": "STGCN",
    "GraphWaveNet": "GraphWaveNet",
    "DCRNN": "DCRNN",
    "AGCRN": "AGCRN",
    "NLinear": "NLinear",
    "RLinear": "RLinear",
    "PatchTST": "PatchTST",
    "TimesNet": "TimesNet",
    "TimeMixer": "TimeMixer",
    "iTransformer": "iTransformer",
    "Autoformer": "Autoformer",
    "FEDformer": "FEDformer",
    "ETSformer": "ETSformer",
    "Transformer": "Transformer",
    "Crossformer": "Crossformer",
    "AutoConNet": "AutoConNet",
    "AutoConNetAutoCon": "AutoConNet",
    "Mamba1": "Mamba1AutoCon",
    "Mamba1AutoCon": "Mamba1AutoCon",
    "Mamba2": "Mamba2AutoCon",
    "Mamba2AutoCon": "Mamba2AutoCon",
    "MambaTS": "MambaTS",
    "MambaTSAutoCon": "MambaTS",
    "MambaTSv2": "MambaTSv2",
    "MambaTSv2AutoCon": "MambaTSv2",
    "MambaTSv3": "MambaTSv3",
    "MambaTSv3AutoCon": "MambaTSv3",
    "MambaTSv4": "MambaTSv4",
    "MambaTSv4AutoCon": "MambaTSv4",
    "MambaTSv5": "MambaTSv5",
    "MambaTSv5AutoCon": "MambaTSv5",
    "MambaTSv6": "MambaTSv6",
    "MambaTSv6AutoCon": "MambaTSv6",
    "MambaTSv7": "MambaTSv7",
    "MambaTSv7AutoCon": "MambaTSv7",
    "MambaTSv8": "MambaTSv8",
    "MambaTSv8AutoCon": "MambaTSv8",
    "MambaTSv9": "MambaTSv9",
    "MambaTSv9AutoCon": "MambaTSv9",
    "MambaTSv10": "MambaTSv10",
    "MambaTSv10AutoCon": "MambaTSv10",
    "MambaTSv11": "MambaTSv11",
    "MambaTSv11AutoCon": "MambaTSv11",
    "MambaTSv12": "MambaTSv12",
    "MambaTSv12AutoCon": "MambaTSv12",
    "MambaTSv13": "MambaTSv13",
    "MambaTSv13AutoCon": "MambaTSv13",
    "MambaTSv14": "MambaTSv14",
    "MambaTSv14AutoCon": "MambaTSv14",
    "MambaTSv15": "MambaTSv15",
    "MambaTSv15AutoCon": "MambaTSv15",
    "MambaTSv16": "MambaTSv16",
    "MambaTSv16AutoCon": "MambaTSv16",
    "MambaTM1": "MambaTM1AutoCon",
    "MambaTM1AutoCon": "MambaTM1AutoCon",
    "MambaTM2": "MambaTM2AutoCon",
    "MambaTM2AutoCon": "MambaTM2AutoCon",
    "MambaSM": "MambaSMambaAutoCon",
    "MambaSMAutoCon": "MambaSMambaAutoCon",
    "RaMambaLite": "RaMambaLite",
    "RaMambaLiteAutoCon": "RaMambaLite",
    "BiPatchMamba": "BiPatchMambaAutoCon",
    "BiPatchMambaAutoCon": "BiPatchMambaAutoCon",
    "LatentTemporalMamba": "LatentTemporalMamba",
    "LatentTemporalMambaAutoCon": "LatentTemporalMamba",
    "InvertedRaMamba": "InvertedRaMambaAutoCon",
    "InvertedRaMambaAutoCon": "InvertedRaMambaAutoCon",
    "LinearLatentMamba": "LinearLatentMambaAutoCon",
    "LinearLatentMambaAutoCon": "LinearLatentMambaAutoCon",
    "NLinearVariableMamba": "NLinearVariableMambaAutoCon",
    "NLinearVariableMambaAutoCon": "NLinearVariableMambaAutoCon",
    "MambaACF": "MambaACF",
    # Runner-level aliases resolve to the same executable modules.  Keeping
    # them in the CLI registry lets an explicit resource request retain the
    # public arm identity while ``executable_model`` selects the module.
    "MambaACFVariable": "MambaACF",
    "MambaACFAutoCon": "MambaACF",
    "MambaACFVariableAutoCon": "MambaACF",
    "MambaACFLiteAutoCon": "MambaACF",
    "MambaACF1": "MambaACF1",
    "MambaACF1AutoCon": "MambaACF1",
}
AUTOCON_IDENTIFIERS = {
    "AutoConNetAutoCon",
    "Mamba1AutoCon",
    "Mamba2AutoCon",
    "MambaTSAutoCon",
    "MambaTSv2AutoCon",
    "MambaTSv3AutoCon",
    "MambaTSv4AutoCon",
    "MambaTSv5AutoCon",
    "MambaTSv6AutoCon",
    "MambaTSv7AutoCon",
    "MambaTSv8AutoCon",
    "MambaTSv9AutoCon",
    "MambaTSv10AutoCon",
    "MambaTSv11AutoCon",
    "MambaTSv12AutoCon",
    "MambaTSv13AutoCon",
    "MambaTSv14AutoCon",
    "MambaTSv15AutoCon",
    "MambaTSv16AutoCon",
    "MambaTM1AutoCon",
    "MambaTM2AutoCon",
    "MambaSMAutoCon",
    "RaMambaLiteAutoCon",
    "BiPatchMambaAutoCon",
    "LatentTemporalMambaAutoCon",
    "InvertedRaMambaAutoCon",
    "LinearLatentMambaAutoCon",
    "NLinearVariableMambaAutoCon",
    "MambaACFAutoCon",
    "MambaACFVariableAutoCon",
    "MambaACFLiteAutoCon",
    "MambaACF1AutoCon",
}
TIME_FEATURE_DIMS = {
    "h": 4,
    "t": 5,
    "s": 6,
    "m": 1,
    "a": 1,
    "w": 2,
    "d": 3,
    "b": 3,
}

# These mirror the model-facing defaults from run.py.  Explicit CLI mode
# requires the dimensional fields, while the remaining architectural knobs
# retain the corrected plan's values unless the caller makes them explicit.
EXPLICIT_DEFAULTS = {
    "task_name": "long_term_forecast",
    "AutoCon": False,
    "AutoCon_wnorm": "LastVal",
    "AutoCon_lambda": 0.05,
    "AutoCon_multiscales": [96],
    "seq_len": None,
    "label_len": None,
    "pred_len": None,
    "enc_in": None,
    "dec_in": None,
    "c_out": None,
    "d_model": 32,
    "n_heads": 8,
    "e_layers": 1,
    "d_layers": 1,
    "d_ff": 32,
    "patch_len": 16,
    "patch_stride": 8,
    "mamba_ts_variant": "v1",
    "mamba_ts_mixer_type": "mamba1",
    "mamba_ts_context_rank": 16,
    "mamba_ts_delta_multipliers": [1.0, 2.0],
    "mamba_ts_dt_min": 0.001,
    "mamba_ts_dt_max": 0.1,
    "mamba_ts_repr_forecast_fusion": False,
    "mamba_ts_repr_forecast_channelwise": False,
    "mamba_ts_channel_adapter": False,
    "mamba_ts_channel_adapter_rank": 8,
    "mamba_ts_revin_affine": False,
    "mamba_ts_global_channel_mixer": False,
    "mamba_ts_global_channel_mixer_rank": 4,
    "mamba_ts_graph_spatial": False,
    "mamba_ts_graph_path": "",
    "mamba_ts_graph_expected_sha256": "",
    "mamba_ts_repr_linear_expert": False,
    "mamba_ts_repr_linear_expert_mode": "residual",
    "mamba_ts_repr_linear_blend_bias": -4.0,
    "mamba_ts_repr_linear_expert_rank": 16,
    "mamba_ts_variable_permutation": False,
    "mamba_ts_acf_memory_selector": False,
    "mamba_ts_acf_memory_channel_selector": False,
    "mamba_ts_forecast_acf_consistency": False,
    "mamba_ts_forecast_acf_weight": 0.0,
    "mamba_ts_forecast_acf_top_k": 4,
    "mamba_ts_forecast_acf_min_lag": 2,
    "mamba_ts_forecast_acf_min_abs": 0.10,
    "mamba_ts_spectral_top_k": 4,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "mamba_headdim": 16,
    "mamba_tm_n1": 64,
    "mamba_tm_n2": 128,
    "mamba_tm_repr_width": 64,
    "mamba_tm_channel_independent": False,
    "mamba_tm_residual": True,
    "mamba_bipatch_token_mode": "mixing",
    "mamba_bipatch_repr_gate": False,
    "mamba_scales": [1, 2],
    "mamba_token_stride": 1,
    "mamba_bidirectional": False,
    "mamba_channel_independent": False,
    "mamba_channel_gate": False,
    "mamba_channel_gate_rank": 8,
    "mamba_data_dependent_channel_mixer": False,
    "mamba_data_dependent_channel_mixer_rank": 8,
    "mamba_acf_scale_conditioning": False,
    "mamba_acf_local_agreement": False,
    "mamba_acf_local_min_pairs": 16,
    "mamba_acf_channel_gate": False,
    "mamba_causal_embedding": False,
    "mamba_require_strict_representation_causality": False,
    "mamba_variable_mixer": False,
    "mamba_variable_gdd_mixer": False,
    "mamba_variable_gdd_mixer_rank": 8,
    "mamba_variable_gdd_init_scale": 0.0,
    "mamba_variable_mixer_depth": 1,
    "mamba_variable_group_size": 1,
    "mamba_variable_repr_fusion": False,
    "mamba_variable_acf_memory_selector": False,
    "mamba_variable_permutation": False,
    "mamba_acf_channel_order": "input",
    "mamba_acf_channel_conditioning": False,
    "mamba_acf_repr_fusion": False,
    "mamba_patch_mixer": False,
    "mamba_patch_mixer_depth": 1,
    "mamba_patch_len": 16,
    "mamba_patch_stride": 8,
    "mamba_patch_order": "time_major",
    "mamba_patch_permutation": False,
    "mamba_local_mixer": False,
    "mamba_local_patch_len": 16,
    "mamba_local_patch_stride": 8,
    "mamba_local_width": None,
    "mamba_local_rank": 16,
    "mamba_local_depth": 1,
    "mamba_low_rank_head": False,
    "mamba_temporal_rank": 16,
    "mamba_acf_lags": [24, 168, 336],
    "mamba_acf_channel_lag_bank": False,
    "mamba_acf_channel_lag_top_k": 2,
    "mamba_acf_lag_mode": "fixed",
    "mamba_acf_max_lag": 720,
    "mamba_acf_top_k": 8,
    "mamba_acf_min_lag": 2,
    "mamba_acf_min_abs": 0.1,
    "mamba_acf_confidence_z": 1.96,
    "mamba_acf_min_separation": 2,
    "mamba_acf_detrend": "mean",
    "mamba_acf_chunk": 64,
    "mamba_use_acf_memory": False,
    "mamba_dynamic_acf": False,
    "mamba_acf_temperature": 4.0,
    "mamba_acf_step_reliability": False,
    "mamba_acf_reliability_interpolation": False,
    "mamba_acf_memory_gate_mode": "hidden",
    "mamba_acf_anchor_copy": False,
    "mamba_acf_horizon_calibration": False,
    "mamba_acf_horizon_calibration_rank": 2,
    "mamba_acf_relation_loss": False,
    "mamba_acf_relation_batching": "lag_pairs",
    "mamba_acf_relation_lag_balance": "random",
    "mamba_acf_relation_channel_reduce": "mean",
    "mamba_acf_relation_scope": "both",
    "mamba_acf_relation_representation": "encoder",
    "mamba_acf_relation_lag_mask": False,
    "mamba_acf_relation_lag_tolerance": 0,
    "mamba_acf_relation_min_lag": 0,
    "mamba_acf_relation_target_mode": "hard_selected",
    "mamba_acf_relation_soft_min_abs": 0.1,
    "mamba_acf_relation_diagnostic_min_abs": 0.1,
    "mamba_acf_relation_selected_lag_boost": 0.0,
    "mamba_acf_relation_active_anchor_normalization": False,
    "AutoCon_warmup_epochs": 0,
    "AutoCon_acf_source": "residual",
    "AutoCon_temperature": 0.2,
    "AutoCon_relation_temperature": 0.25,
    "AutoCon_sign_weight": 0.25,
    "AutoCon_sign_margin": 0.1,
    "moving_avg": 25,
    "factor": 1,
    "top_k": 5,
    "num_kernels": 6,
    "timemixer_down_sampling_window": 2,
    "timemixer_down_sampling_layers": 1,
    "timemixer_channel_independence": False,
    "timemixer_decomp_method": "moving_avg",
    "dropout": 0.1,
    "embed": "timeF",
    "activation": "gelu",
    "freq": "h",
    "wo_freq": "",
    "output_attention": False,
    "individual": False,
    "features": "S",
    "data": "explicit",
}

RESOURCE_SIGNATURE_FIELDS = (
    "model",
    "AutoCon",
    "task_name",
    "features",
    "freq",
    "embed",
    "activation",
    "wo_freq",
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "dec_in",
    "c_out",
    "d_model",
    "n_heads",
    "e_layers",
    "d_layers",
    "d_ff",
    "patch_len",
    "patch_stride",
        "mamba_ts_variant",
        "mamba_ts_mixer_type",
        "mamba_ts_context_rank",
    "mamba_ts_repr_forecast_fusion",
    "mamba_ts_repr_forecast_channelwise",
    "mamba_ts_channel_adapter",
    "mamba_ts_channel_adapter_rank",
    "mamba_ts_revin_affine",
    "mamba_ts_global_channel_mixer",
    "mamba_ts_global_channel_mixer_rank",
    "mamba_ts_repr_linear_expert",
    "mamba_ts_repr_linear_expert_mode",
    "mamba_ts_repr_linear_blend_bias",
    "mamba_ts_repr_linear_expert_rank",
    "mamba_ts_variable_permutation",
    "mamba_ts_acf_memory_selector",
    "mamba_ts_acf_memory_channel_selector",
    "mamba_ts_forecast_acf_consistency",
    "mamba_ts_forecast_acf_weight",
    "mamba_ts_forecast_acf_top_k",
    "mamba_ts_forecast_acf_min_lag",
    "mamba_ts_forecast_acf_min_abs",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "mamba_headdim",
    "mamba_tm_n1",
    "mamba_tm_n2",
    "mamba_tm_repr_width",
    "mamba_tm_channel_independent",
    "mamba_tm_residual",
    "mamba_bipatch_token_mode",
    "mamba_scales",
    "mamba_token_stride",
    "mamba_bidirectional",
    "mamba_channel_independent",
    "mamba_channel_gate",
    "mamba_channel_gate_rank",
    "mamba_data_dependent_channel_mixer",
    "mamba_data_dependent_channel_mixer_rank",
    "mamba_acf_scale_conditioning",
    "mamba_acf_local_agreement",
    "mamba_acf_local_min_pairs",
    "mamba_acf_channel_gate",
    "mamba_causal_embedding",
    "mamba_require_strict_representation_causality",
    "mamba_variable_mixer",
    "mamba_variable_gdd_mixer",
    "mamba_variable_gdd_mixer_rank",
    "mamba_variable_gdd_init_scale",
    "mamba_variable_mixer_depth",
    "mamba_variable_group_size",
    "mamba_variable_repr_fusion",
    "mamba_variable_acf_memory_selector",
    "mamba_variable_permutation",
    "mamba_acf_channel_order",
    "mamba_acf_channel_conditioning",
    "mamba_acf_repr_fusion",
    "mamba_patch_mixer",
    "mamba_patch_mixer_depth",
    "mamba_patch_len",
    "mamba_patch_stride",
    "mamba_patch_order",
    "mamba_patch_permutation",
    "mamba_local_mixer",
    "mamba_local_patch_len",
    "mamba_local_patch_stride",
    "mamba_local_width",
    "mamba_local_rank",
    "mamba_local_depth",
    "mamba_low_rank_head",
    "mamba_temporal_rank",
    "mamba_acf_lags",
    "mamba_acf_channel_lag_bank",
    "mamba_acf_channel_lag_top_k",
    "mamba_acf_lag_mode",
    "mamba_acf_max_lag",
    "mamba_acf_top_k",
    "mamba_acf_min_lag",
    "mamba_acf_min_abs",
    "mamba_acf_confidence_z",
    "mamba_acf_min_separation",
    "mamba_acf_detrend",
    "mamba_acf_chunk",
    "mamba_use_acf_memory",
    "mamba_dynamic_acf",
    "mamba_acf_temperature",
    "mamba_acf_step_reliability",
    "mamba_acf_reliability_interpolation",
    "mamba_acf_memory_gate_mode",
    "mamba_acf_anchor_copy",
    "mamba_acf_horizon_calibration",
    "mamba_acf_horizon_calibration_rank",
    "mamba_acf_relation_loss",
    "mamba_acf_relation_batching",
    "mamba_acf_relation_lag_balance",
    "mamba_acf_relation_channel_reduce",
    "mamba_acf_relation_scope",
    "mamba_acf_relation_representation",
    "mamba_acf_relation_lag_mask",
    "mamba_acf_relation_lag_tolerance",
    "mamba_acf_relation_min_lag",
    "mamba_acf_relation_target_mode",
    "mamba_acf_relation_soft_min_abs",
    "mamba_acf_relation_diagnostic_min_abs",
    "mamba_acf_relation_selected_lag_boost",
    "mamba_acf_relation_active_anchor_normalization",
    "AutoCon_warmup_epochs",
    "AutoCon_acf_source",
    "AutoCon_temperature",
    "AutoCon_relation_temperature",
    "AutoCon_sign_weight",
    "AutoCon_sign_margin",
    "moving_avg",
    "factor",
    "top_k",
    "num_kernels",
    "timemixer_down_sampling_window",
    "timemixer_down_sampling_layers",
    "timemixer_channel_independence",
    "timemixer_decomp_method",
    "dropout",
    "AutoCon_wnorm",
    "AutoCon_multiscales",
    "output_attention",
    "individual",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_source_tree(root: Path) -> Tuple[Dict[str, str], str]:
    """Return the same file-map/tree hash convention used by the runner."""
    files: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files[path.relative_to(root).as_posix()] = _sha256(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return files, hashlib.sha256(encoded).hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read JSON: {}: {}".format(path, exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object: {}".format(path))
    return payload


def _resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _command_value(command: Sequence[Any], name: str) -> Optional[str]:
    try:
        index = list(command).index(name)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def _command_values(command: Sequence[Any], name: str) -> List[str]:
    """Read a variable-length CLI value list until the next option."""

    try:
        index = list(command).index(name)
    except ValueError:
        return []
    values: List[str] = []
    for value in list(command)[index + 1:]:
        if str(value).startswith("--"):
            break
        values.append(str(value))
    return values


def _command_flag(command: Sequence[Any], name: str) -> bool:
    return name in command


def _command_bool(command: Sequence[Any], name: str) -> Optional[bool]:
    """Read a boolean option recorded as a bare flag or explicit value."""

    if name not in command:
        return None
    value = _command_value(command, name)
    if value is None or value.startswith("--"):
        return True
    return _as_bool(value, name)


def _command_bool_or_default(command: Sequence[Any], name: str, default: bool) -> bool:
    value = _command_bool(command, name)
    return bool(default) if value is None else value


def _read_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    malformed: List[int] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError("cannot read manifest: {}: {}".format(path, exc)) from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue
            if not isinstance(record, dict) or not isinstance(record.get("job_id"), str):
                malformed.append(line_number)
                continue
            # This is the runner's resume rule: later records replace earlier
            # attempts for the same job id.
            records[record["job_id"]] = record
    if malformed:
        raise ValueError("manifest has malformed rows: {}".format(malformed[:10]))
    return records


def _model_identifier(config: Mapping[str, Any]) -> str:
    return canonical_model(config)


def _model_module(config: Mapping[str, Any]) -> str:
    identifier = executable_model(config)
    if identifier not in MODEL_MODULES:
        raise ValueError(
            "unsupported model {!r}; this resource tool supports: {}".format(
                identifier, ", ".join(sorted(MODEL_MODULES))
            )
        )
    return MODEL_MODULES[identifier]


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be an integer, got {!r}".format(name, value))
    if isinstance(value, (float, np.floating)) and (
        not math.isfinite(float(value)) or not float(value).is_integer()
    ):
        raise ValueError("{} must be an integer, got {!r}".format(name, value))
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be an integer, got {!r}".format(name, value)) from exc
    if not isinstance(value, str) and result != value:
        raise ValueError("{} must be an integer, got {!r}".format(name, value))
    if result <= 0:
        raise ValueError("{} must be positive, got {!r}".format(name, value))
    return result


def _as_nonnegative_int(value: Any, name: str) -> int:
    """Parse an integer setting that permits zero but rejects booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a non-negative integer, got {!r}".format(name, value))
    if isinstance(value, (float, np.floating)) and (
        not math.isfinite(float(value)) or not float(value).is_integer()
    ):
        raise ValueError("{} must be an integer, got {!r}".format(name, value))
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be an integer, got {!r}".format(name, value)) from exc
    if result < 0:
        raise ValueError("{} must be non-negative, got {!r}".format(name, value))
    return result


def _as_bool(value: Any, name: str) -> bool:
    """Parse a configuration boolean without treating ``"false"`` as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError("{} must be a boolean, got {!r}".format(name, value))


def _str2bool(value: Any) -> bool:
    """argparse adapter for explicit boolean values in resource configs."""

    return _as_bool(value, "mamba_tm_residual")


def _as_positive_int_list(value: Any, name: str, default: Sequence[int]) -> List[int]:
    """Normalize scalar/list integer settings while preserving deterministic order."""

    if value is None:
        value = default
    if isinstance(value, (str, bytes)):
        value = [item for item in str(value).replace(",", " ").split() if item]
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("{} must be a non-empty list".format(name))
    result = [_as_positive_int(item, name) for item in value]
    # The model always injects scale one when omitted.  Canonicalizing here
    # makes resource signatures represent the effective architecture.
    if name == "mamba_scales" and 1 not in result:
        result.append(1)
    return sorted(set(result))


def _normalise_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(EXPLICIT_DEFAULTS)
    merged.update(dict(config))
    if not merged.get("model"):
        raise ValueError("config has no model")
    for name in ("seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out"):
        merged[name] = _as_positive_int(merged.get(name), name)
    for name in (
        "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "patch_len", "patch_stride",
        "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_headdim", "moving_avg",
        "mamba_tm_n1", "mamba_tm_n2", "mamba_tm_repr_width",
        "factor", "top_k", "num_kernels", "timemixer_down_sampling_window",
        "timemixer_down_sampling_layers", "mamba_channel_gate_rank",
        "mamba_data_dependent_channel_mixer_rank", "mamba_variable_gdd_mixer_rank",
        "mamba_temporal_rank",
        "mamba_variable_mixer_depth", "mamba_acf_max_lag", "mamba_acf_top_k", "mamba_acf_min_lag",
        "mamba_acf_channel_lag_top_k",
        "mamba_variable_group_size",
        "mamba_acf_min_separation", "mamba_acf_chunk", "mamba_patch_mixer_depth",
        "mamba_patch_len", "mamba_patch_stride",
        "mamba_local_patch_len", "mamba_local_patch_stride",
        "mamba_local_rank", "mamba_local_depth", "mamba_token_stride",
        "mamba_acf_horizon_calibration_rank",
        "mamba_acf_local_min_pairs",
    ):
        merged[name] = _as_positive_int(merged.get(name), name)
    local_width = merged.get("mamba_local_width")
    if local_width is None:
        merged["mamba_local_width"] = merged["d_model"]
    else:
        merged["mamba_local_width"] = _as_positive_int(local_width, "mamba_local_width")
    merged["dropout"] = float(merged["dropout"])
    if not math.isfinite(merged["dropout"]) or merged["dropout"] < 0:
        raise ValueError("dropout must be finite and non-negative")
    merged["AutoCon"] = _as_bool(merged.get("AutoCon", False), "AutoCon")
    merged["timemixer_channel_independence"] = _as_bool(
        merged.get("timemixer_channel_independence", False),
        "timemixer_channel_independence",
    )
    merged["timemixer_decomp_method"] = str(
        merged.get("timemixer_decomp_method", "moving_avg")
    )
    if merged["timemixer_decomp_method"] not in {"moving_avg", "dft_decomp"}:
        raise ValueError("timemixer_decomp_method must be moving_avg or dft_decomp")
    if (
        str(merged.get("model")) == "TimeMixer"
        and merged["seq_len"] // (
            merged["timemixer_down_sampling_window"] ** merged["timemixer_down_sampling_layers"]
        ) < 2
    ):
        raise ValueError("TimeMixer down-sampling leaves fewer than two time steps")
    # Keep explicit resource measurements on the same fixed recipe as the
    # training-plan runner and direct ``run.py`` alias.  A plan-produced
    # config may retain the executable model name, so resolve the public arm
    # through its immutable plan-job provenance before applying the override.
    if canonical_model(merged) == "MambaACFLiteAutoCon":
        merged.update({
            "AutoCon": True,
            "mamba_scales": [1],
            "mamba_causal_embedding": True,
            "mamba_low_rank_head": True,
            "mamba_temporal_rank": 8,
            "mamba_variable_mixer": False,
            "mamba_patch_mixer": False,
            "mamba_local_mixer": False,
            "mamba_use_acf_memory": False,
            "mamba_acf_relation_loss": True,
        })
    merged["mamba_scales"] = _as_positive_int_list(
        merged.get("mamba_scales"), "mamba_scales", [1, 2]
    )
    if str(merged.get("model", "")).startswith("Mamba") and any(
        scale > merged["seq_len"] for scale in merged["mamba_scales"]
    ):
        raise ValueError("mamba_scales cannot exceed seq_len")
    if merged["mamba_token_stride"] > merged["seq_len"]:
        raise ValueError("mamba_token_stride cannot exceed seq_len")
    scales = merged.get("AutoCon_multiscales", [96])
    if isinstance(scales, (int, float)):
        scales = [scales]
    if not isinstance(scales, (list, tuple)) or not scales:
        raise ValueError("AutoCon_multiscales must be a non-empty list")
    merged["AutoCon_multiscales"] = [_as_positive_int(value, "AutoCon_multiscales") for value in scales]
    merged["freq"] = str(merged.get("freq", "h")).lower()
    merged["embed"] = str(merged.get("embed", "timeF"))
    merged["activation"] = str(merged.get("activation", "gelu"))
    merged["wo_freq"] = str(merged.get("wo_freq", ""))
    merged["features"] = str(merged.get("features", "S"))
    merged["data"] = str(merged.get("data", "explicit"))
    merged["task_name"] = str(merged.get("task_name", "long_term_forecast"))
    model_name = str(merged.get("model", ""))
    variant_default = (
        "v16" if model_name.lower().startswith("mambatsv16")
        else "v15" if model_name.lower().startswith("mambatsv15")
        else "v14" if model_name.lower().startswith("mambatsv14")
        else "v13" if model_name.lower().startswith("mambatsv13")
        else "v12" if model_name.lower().startswith("mambatsv12")
        else "v11" if model_name.lower().startswith("mambatsv11")
        else "v10" if model_name.lower().startswith("mambatsv10")
        else "v9" if model_name.lower().startswith("mambatsv9")
        else "v8" if model_name.lower().startswith("mambatsv8")
        else "v7" if model_name.lower().startswith("mambatsv7")
        else "v6" if model_name.lower().startswith("mambatsv6")
        else "v5" if model_name.lower().startswith("mambatsv5")
        else "v4" if model_name.lower().startswith("mambatsv4")
        else "v3" if model_name.lower().startswith("mambatsv3")
        else "v2" if model_name.lower().startswith("mambatsv2")
        else "v1"
    )
    requested_variant = (
        config["mamba_ts_variant"]
        if "mamba_ts_variant" in config
        else variant_default
    )
    merged["mamba_ts_variant"] = str(requested_variant).strip().lower()
    if merged["mamba_ts_variant"] not in {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
        raise ValueError("mamba_ts_variant must be v1 through v16")
    merged["mamba_ts_mixer_type"] = str(
        merged.get("mamba_ts_mixer_type", "mamba1")
    ).strip().lower()
    if merged["mamba_ts_mixer_type"] not in {"mamba1", "mamba2"}:
        raise ValueError("mamba_ts_mixer_type must be mamba1 or mamba2")
    if merged["mamba_ts_mixer_type"] == "mamba2" and merged["mamba_ts_variant"] == "v12":
        raise ValueError("MambaTS v12 Delta branches currently require mamba1 mixer")
    if model_name.lower().startswith("mambatsv16") and merged["mamba_ts_variant"] != "v16":
        raise ValueError("MambaTSv16 identifiers require mamba_ts_variant=v16")
    if model_name.lower().startswith("mambatsv15") and merged["mamba_ts_variant"] != "v15":
        raise ValueError("MambaTSv15 identifiers require mamba_ts_variant=v15")
    if model_name.lower().startswith("mambatsv14") and merged["mamba_ts_variant"] != "v14":
        raise ValueError("MambaTSv14 identifiers require mamba_ts_variant=v14")
    if model_name.lower().startswith("mambatsv13") and merged["mamba_ts_variant"] != "v13":
        raise ValueError("MambaTSv13 identifiers require mamba_ts_variant=v13")
    if model_name.lower().startswith("mambatsv12") and merged["mamba_ts_variant"] != "v12":
        raise ValueError("MambaTSv12 identifiers require mamba_ts_variant=v12")
    if model_name.lower().startswith("mambatsv11") and merged["mamba_ts_variant"] != "v11":
        raise ValueError("MambaTSv11 identifiers require mamba_ts_variant=v11")
    if model_name.lower().startswith("mambatsv10") and merged["mamba_ts_variant"] != "v10":
        raise ValueError("MambaTSv10 identifiers require mamba_ts_variant=v10")
    if model_name.lower().startswith("mambatsv9") and merged["mamba_ts_variant"] != "v9":
        raise ValueError("MambaTSv9 identifiers require mamba_ts_variant=v9")
    if model_name.lower().startswith("mambatsv8") and merged["mamba_ts_variant"] != "v8":
        raise ValueError("MambaTSv8 identifiers require mamba_ts_variant=v8")
    if model_name.lower().startswith("mambatsv7") and merged["mamba_ts_variant"] != "v7":
        raise ValueError("MambaTSv7 identifiers require mamba_ts_variant=v7")
    if model_name.lower().startswith("mambatsv6") and merged["mamba_ts_variant"] != "v6":
        raise ValueError("MambaTSv6 identifiers require mamba_ts_variant=v6")
    if model_name.lower().startswith("mambatsv5") and merged["mamba_ts_variant"] != "v5":
        raise ValueError("MambaTSv5 identifiers require mamba_ts_variant=v5")
    if model_name.lower().startswith("mambatsv4") and merged["mamba_ts_variant"] != "v4":
        raise ValueError("MambaTSv4 identifiers require mamba_ts_variant=v4")
    if model_name.lower().startswith("mambatsv2") and merged["mamba_ts_variant"] != "v2":
        raise ValueError("MambaTSv2 identifiers require mamba_ts_variant=v2")
    if model_name.lower().startswith("mambatsv3") and merged["mamba_ts_variant"] != "v3":
        raise ValueError("MambaTSv3 identifiers require mamba_ts_variant=v3")
    if model_name in {"MambaTS", "MambaTSAutoCon"} and merged["mamba_ts_variant"] != "v1":
        raise ValueError("MambaTS v1 identifiers require mamba_ts_variant=v1")
    merged["mamba_ts_context_rank"] = _as_positive_int(
        merged.get("mamba_ts_context_rank", 16), "mamba_ts_context_rank"
    ) if merged["mamba_ts_variant"] in {"v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"} else None
    merged["mamba_ts_spectral_top_k"] = _as_positive_int(
        merged.get("mamba_ts_spectral_top_k", 4), "mamba_ts_spectral_top_k"
    ) if merged["mamba_ts_variant"] == "v8" else None
    merged["mamba_ts_repr_forecast_fusion"] = bool(
        merged.get("mamba_ts_repr_forecast_fusion", False)
    )
    if merged["mamba_ts_repr_forecast_fusion"] and merged["mamba_ts_variant"] not in {"v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
        raise ValueError("mamba_ts_repr_forecast_fusion requires a supported MambaTS v3 through v16 variant")
    merged["mamba_ts_repr_forecast_channelwise"] = bool(
        merged.get("mamba_ts_repr_forecast_channelwise", False)
    )
    raw_multipliers = merged.get("mamba_ts_delta_multipliers", [1.0, 2.0])
    try:
        merged["mamba_ts_delta_multipliers"] = [float(value) for value in raw_multipliers]
    except TypeError as exc:
        raise ValueError("mamba_ts_delta_multipliers must be an iterable") from exc
    merged["mamba_ts_dt_min"] = float(merged.get("mamba_ts_dt_min", 0.001))
    merged["mamba_ts_dt_max"] = float(merged.get("mamba_ts_dt_max", 0.1))
    if merged["mamba_ts_variant"] == "v12":
        if (len(merged["mamba_ts_delta_multipliers"]) != 2
                or merged["mamba_ts_delta_multipliers"][0] != 1.0
                or merged["mamba_ts_delta_multipliers"][1] <= 1.0):
            raise ValueError("MambaTS v12 requires Delta multipliers [1.0, scale] with scale > 1")
        if not 0 < merged["mamba_ts_dt_min"] < merged["mamba_ts_dt_max"]:
            raise ValueError("MambaTS v12 requires 0 < mamba_ts_dt_min < mamba_ts_dt_max")
    merged["mamba_ts_channel_adapter"] = bool(
        merged.get("mamba_ts_channel_adapter", False)
    )
    merged["mamba_ts_channel_adapter_rank"] = _as_positive_int(
        merged.get("mamba_ts_channel_adapter_rank", 8),
        "mamba_ts_channel_adapter_rank",
    )
    merged["mamba_ts_revin_affine"] = bool(
        merged.get("mamba_ts_revin_affine", False)
    )
    merged["mamba_ts_global_channel_mixer"] = bool(
        merged.get("mamba_ts_global_channel_mixer", False)
    )
    merged["mamba_ts_global_channel_mixer_rank"] = _as_positive_int(
        merged.get("mamba_ts_global_channel_mixer_rank", 4),
        "mamba_ts_global_channel_mixer_rank",
    )
    merged["mamba_ts_graph_spatial"] = bool(
        merged.get("mamba_ts_graph_spatial", False)
    )
    merged["mamba_ts_graph_path"] = str(
        merged.get("mamba_ts_graph_path", "") or ""
    ).strip()
    merged["mamba_ts_graph_expected_sha256"] = str(
        merged.get("mamba_ts_graph_expected_sha256", "") or ""
    ).strip().lower()
    if merged["mamba_ts_global_channel_mixer"] and (
        not merged["AutoCon"] or merged["mamba_ts_variant"] != "v3"
    ):
        raise ValueError(
            "mamba_ts_global_channel_mixer requires AutoCon MambaTS v3"
        )
    if merged["mamba_ts_graph_spatial"] and model_name not in {"GraphDLinear", "STGCN", "GraphWaveNet", "DCRNN"} and (
        not merged["AutoCon"] or merged["mamba_ts_variant"] != "v3"
    ):
        raise ValueError("mamba_ts_graph_spatial requires AutoCon MambaTS v3, GraphDLinear, STGCN, GraphWaveNet, or DCRNN")
    if merged["mamba_ts_graph_spatial"] and not merged["mamba_ts_graph_path"]:
        raise ValueError("mamba_ts_graph_spatial requires mamba_ts_graph_path")
    merged["mamba_ts_repr_linear_expert"] = bool(
        merged.get("mamba_ts_repr_linear_expert", False)
    )
    merged["mamba_ts_repr_linear_expert_mode"] = str(
        merged.get("mamba_ts_repr_linear_expert_mode", "residual")
    ).strip().lower()
    if merged["mamba_ts_repr_linear_expert_mode"] not in {"residual", "blend", "nlinear_blend"}:
        raise ValueError("mamba_ts_repr_linear_expert_mode must be residual, blend, or nlinear_blend")
    merged["mamba_ts_repr_linear_blend_bias"] = float(
        merged.get("mamba_ts_repr_linear_blend_bias", -4.0)
    )
    if not -12.0 <= merged["mamba_ts_repr_linear_blend_bias"] <= 0.0:
        raise ValueError("mamba_ts_repr_linear_blend_bias must be in [-12, 0]")
    merged["mamba_ts_repr_linear_expert_rank"] = int(
        merged.get("mamba_ts_repr_linear_expert_rank", 16)
    )
    if merged["mamba_ts_repr_linear_expert_rank"] <= 0:
        raise ValueError("mamba_ts_repr_linear_expert_rank must be positive")
    if merged["mamba_ts_repr_linear_expert_mode"] == "blend" and not merged["mamba_ts_repr_linear_expert"]:
        raise ValueError("blend mode requires mamba_ts_repr_linear_expert")
    merged["mamba_ts_variable_permutation"] = bool(
        merged.get("mamba_ts_variable_permutation", False)
    )
    merged["mamba_ts_acf_memory_selector"] = bool(
        merged.get("mamba_ts_acf_memory_selector", False)
    )
    merged["mamba_ts_acf_memory_channel_selector"] = bool(
        merged.get("mamba_ts_acf_memory_channel_selector", False)
    )
    merged["mamba_ts_forecast_acf_consistency"] = bool(
        merged.get("mamba_ts_forecast_acf_consistency", False)
    )
    if merged["mamba_ts_variable_permutation"] and merged["mamba_ts_variant"] not in {"v3", "v4", "v5", "v6", "v7"}:
        raise ValueError("mamba_ts_variable_permutation requires mamba_ts_variant in {v3,v4,v5,v6}")
    if merged["mamba_ts_acf_memory_selector"] and (
        not merged["AutoCon"] or merged["mamba_ts_variant"] not in {"v4", "v5", "v7"}
        or not merged["mamba_use_acf_memory"]
    ):
        raise ValueError("mamba_ts_acf_memory_selector requires AutoCon MambaTS v4/v5 with ACF memory")
    if merged["mamba_ts_acf_memory_channel_selector"] and not merged["mamba_ts_acf_memory_selector"]:
        raise ValueError("mamba_ts_acf_memory_channel_selector requires mamba_ts_acf_memory_selector")
    if merged["mamba_ts_forecast_acf_consistency"]:
        if not merged["AutoCon"] or merged["mamba_ts_variant"] not in {"v3", "v4", "v5", "v6", "v7"}:
            raise ValueError("mamba_ts_forecast_acf_consistency requires AutoCon MambaTS v3/v4/v5")
        if not 0.0 < float(merged["mamba_ts_forecast_acf_weight"]) <= 1.0:
            raise ValueError("mamba_ts_forecast_acf_weight must be in (0, 1]")
    if merged["mamba_ts_repr_forecast_channelwise"] and not merged["mamba_ts_repr_forecast_fusion"]:
        raise ValueError("mamba_ts_repr_forecast_channelwise requires mamba_ts_repr_forecast_fusion")
    if merged["mamba_ts_channel_adapter"] and (
        not merged["AutoCon"] or merged["mamba_ts_variant"] != "v3"
    ):
        raise ValueError("mamba_ts_channel_adapter requires AutoCon MambaTS v3")
    if merged["mamba_ts_revin_affine"] and (
        not merged["AutoCon"] or merged["mamba_ts_variant"] != "v3"
    ):
        raise ValueError("mamba_ts_revin_affine requires AutoCon MambaTS v3")
    if merged["mamba_ts_revin_affine"] and str(
        merged.get("AutoCon_wnorm", "LastVal")
    ).strip().lower() not in {"meanstd", "revin"}:
        raise ValueError(
            "mamba_ts_revin_affine requires AutoCon_wnorm=MeanStd or ReVIN"
        )
    if merged["mamba_ts_repr_linear_expert"] and (
        not merged["AutoCon"] or merged["mamba_ts_variant"] not in {"v3", "v4", "v5", "v6", "v7"}
    ):
        raise ValueError("mamba_ts_repr_linear_expert requires AutoCon MambaTS v3/v4/v5")
    # Capability checks use the executable model name because runner-level
    # variable-token aliases intentionally share the MambaACF module.
    identifier = executable_model(merged)
    if identifier in {"MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"}:
        base_token_count = int(math.ceil(
            merged["seq_len"] / merged["mamba_token_stride"]
        ))
        if any(scale > base_token_count for scale in merged["mamba_scales"]):
            raise ValueError(
                "mamba_scales cannot exceed {} base tokens after "
                "mamba_token_stride={}".format(
                    base_token_count, merged["mamba_token_stride"]
                )
            )
    for name in (
        "mamba_bidirectional", "mamba_channel_independent", "mamba_channel_gate",
        "mamba_tm_channel_independent", "mamba_tm_residual",
        "mamba_data_dependent_channel_mixer",
        "mamba_acf_channel_lag_bank",
        "mamba_acf_scale_conditioning",
        "mamba_acf_local_agreement",
        "mamba_acf_channel_gate",
        "mamba_causal_embedding", "mamba_variable_mixer", "mamba_variable_gdd_mixer",
        "mamba_variable_repr_fusion",
        "mamba_variable_permutation",
        "mamba_require_strict_representation_causality",
        "mamba_acf_channel_conditioning",
        "mamba_acf_repr_fusion",
        "mamba_patch_mixer", "mamba_patch_permutation", "mamba_low_rank_head",
        "mamba_local_mixer",
        "mamba_use_acf_memory", "mamba_dynamic_acf", "mamba_acf_step_reliability",
        "mamba_acf_reliability_interpolation",
        "mamba_acf_anchor_copy",
        "mamba_acf_horizon_calibration",
        "mamba_acf_relation_loss", "mamba_acf_relation_lag_mask",
        "mamba_acf_relation_active_anchor_normalization",
    ):
        merged[name] = _as_bool(merged.get(name, False), name)
    merged["mamba_acf_lags"] = _as_positive_int_list(
        merged.get("mamba_acf_lags"), "mamba_acf_lags", [24, 168, 336]
    )
    if merged["mamba_acf_channel_lag_bank"]:
        if identifier not in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
        }:
            raise ValueError(
                "mamba_acf_channel_lag_bank is supported only by MambaACF models"
            )
        if not merged["mamba_use_acf_memory"]:
            raise ValueError(
                "mamba_acf_channel_lag_bank requires mamba_use_acf_memory=true"
            )
        if merged["mamba_dynamic_acf"]:
            raise ValueError(
                "mamba_acf_channel_lag_bank requires static train-only ACF memory"
            )
        if merged["mamba_acf_channel_lag_top_k"] > len(merged["mamba_acf_lags"]):
            raise ValueError(
                "mamba_acf_channel_lag_top_k cannot exceed mamba_acf_lags count"
            )
    merged["mamba_patch_order"] = str(merged.get("mamba_patch_order", "time_major")).lower()
    if merged["mamba_patch_order"] not in {"time_major", "channel_major"}:
        raise ValueError("mamba_patch_order must be time_major or channel_major")
    merged["mamba_acf_channel_order"] = str(
        merged.get("mamba_acf_channel_order", "input")
    ).lower()
    if merged["mamba_acf_channel_order"] not in {"input", "strength", "similarity"}:
        raise ValueError(
            "mamba_acf_channel_order must be input, strength, or similarity"
        )
    if "MambaACF" in identifier and merged["mamba_patch_len"] > merged["seq_len"]:
        raise ValueError("mamba_patch_len cannot exceed seq_len")
    merged["mamba_acf_lag_mode"] = str(merged.get("mamba_acf_lag_mode", "fixed"))
    if merged["mamba_acf_lag_mode"] not in {
        "fixed", "train_peaks", "train_peaks_available"
    }:
        raise ValueError(
            "mamba_acf_lag_mode must be fixed, train_peaks, or train_peaks_available"
        )
    merged["mamba_acf_memory_gate_mode"] = str(
        merged.get("mamba_acf_memory_gate_mode", "hidden")
    ).lower()
    if merged["mamba_acf_memory_gate_mode"] not in {"hidden", "variable_token"}:
        raise ValueError(
            "mamba_acf_memory_gate_mode must be hidden or variable_token"
        )
    merged["mamba_acf_detrend"] = str(merged.get("mamba_acf_detrend", "mean"))
    if merged["mamba_acf_detrend"] not in {"none", "mean", "linear"}:
        raise ValueError("mamba_acf_detrend must be none, mean, or linear")
    merged["mamba_acf_relation_batching"] = str(
        merged.get("mamba_acf_relation_batching", "lag_pairs")
    )
    if merged["mamba_acf_relation_batching"] not in {"lag_pairs", "random"}:
        raise ValueError("mamba_acf_relation_batching must be lag_pairs or random")
    merged["mamba_acf_relation_lag_balance"] = str(
        merged.get("mamba_acf_relation_lag_balance", "random")
    ).lower()
    if merged["mamba_acf_relation_lag_balance"] not in {"random", "balanced"}:
        raise ValueError("mamba_acf_relation_lag_balance must be random or balanced")
    if (
        merged["mamba_acf_relation_lag_balance"] == "balanced"
        and merged["mamba_acf_relation_batching"] != "lag_pairs"
    ):
        raise ValueError(
            "balanced lag exposure requires mamba_acf_relation_batching=lag_pairs"
        )
    merged["mamba_acf_relation_channel_reduce"] = str(
        merged.get("mamba_acf_relation_channel_reduce", "mean")
    )
    if merged["mamba_acf_relation_channel_reduce"] not in {"mean", "max_abs"}:
        raise ValueError("mamba_acf_relation_channel_reduce must be mean or max_abs")
    merged["mamba_acf_relation_scope"] = str(
        merged.get("mamba_acf_relation_scope", "both")
    ).lower()
    if merged["mamba_acf_relation_scope"] not in {"both", "global", "local"}:
        raise ValueError("mamba_acf_relation_scope must be both, global, or local")
    merged["mamba_acf_relation_representation"] = str(
        merged.get("mamba_acf_relation_representation", "encoder")
    ).lower()
    if merged["mamba_acf_relation_representation"] not in {"encoder", "forecast"}:
        raise ValueError(
            "mamba_acf_relation_representation must be encoder or forecast"
        )
    merged["mamba_acf_relation_lag_tolerance"] = _as_nonnegative_int(
        merged.get("mamba_acf_relation_lag_tolerance", 0),
        "mamba_acf_relation_lag_tolerance",
    )
    merged["mamba_acf_relation_min_lag"] = _as_nonnegative_int(
        merged.get("mamba_acf_relation_min_lag", 0),
        "mamba_acf_relation_min_lag",
    )
    if merged["mamba_acf_relation_min_lag"] > 0:
        if identifier not in {
            "MambaACFAutoCon", "MambaACF1AutoCon",
            "NLinearVariableMambaAutoCon",
        }:
            raise ValueError(
                "mamba_acf_relation_min_lag requires a relation-capable AutoCon model"
            )
    if (
        merged["mamba_acf_relation_loss"]
        and merged["mamba_acf_relation_min_lag"] > merged["mamba_acf_max_lag"]
    ):
        raise ValueError(
            "mamba_acf_relation_min_lag cannot exceed mamba_acf_max_lag "
            "when relation loss is enabled"
        )
    merged["mamba_acf_relation_target_mode"] = str(
        merged.get("mamba_acf_relation_target_mode", "hard_selected")
    ).lower()
    if merged["mamba_acf_relation_target_mode"] not in {
        "hard_selected", "soft_multi_peak", "soft_full"
    }:
        raise ValueError(
            "mamba_acf_relation_target_mode must be hard_selected, "
            "soft_multi_peak, or soft_full"
        )
    # JSON plans use the lower-case key while runtime CLI/config artifacts
    # historically use the mixed-case spelling.  Resolve once so resource
    # signatures and model construction follow the same precedence as the
    # runner and phase validator.
    merged["AutoCon_acf_source"] = resolve_acf_source(merged)
    if merged["AutoCon_acf_source"] not in {"residual", "trend", "raw"}:
        raise ValueError("AutoCon_acf_source must be residual, trend, or raw")
    merged["AutoCon_warmup_epochs"] = int(merged.get("AutoCon_warmup_epochs", 0))
    if merged["AutoCon_warmup_epochs"] < 0:
        raise ValueError("AutoCon_warmup_epochs must be non-negative")
    for name in (
        "mamba_acf_min_abs", "mamba_acf_confidence_z", "mamba_acf_temperature",
        "mamba_acf_relation_soft_min_abs", "mamba_acf_relation_diagnostic_min_abs",
        "mamba_acf_relation_selected_lag_boost",
        "mamba_variable_gdd_init_scale",
        "AutoCon_lambda", "AutoCon_temperature", "AutoCon_relation_temperature",
        "AutoCon_sign_weight", "AutoCon_sign_margin",
    ):
        merged[name] = float(merged[name])
        if not math.isfinite(merged[name]):
            raise ValueError("{} must be finite".format(name))
        if name in {
            "AutoCon_lambda", "AutoCon_sign_weight", "AutoCon_sign_margin",
            "mamba_acf_relation_selected_lag_boost",
        }:
            if merged[name] < 0:
                raise ValueError("{} must be non-negative".format(name))
        elif name in {
            "mamba_acf_relation_soft_min_abs",
            "mamba_acf_relation_diagnostic_min_abs",
            "mamba_variable_gdd_init_scale",
        }:
            if merged[name] < 0 or merged[name] > 1:
                raise ValueError(
                    "{} must be in [0, 1]".format(name)
                )
        elif merged[name] <= 0:
            raise ValueError("{} must be finite and positive".format(name))
    if merged["mamba_acf_min_lag"] > merged["mamba_acf_max_lag"]:
        raise ValueError("mamba_acf_min_lag cannot exceed mamba_acf_max_lag")
    if merged["mamba_channel_independent"] and (
        merged["mamba_channel_gate"] or merged["mamba_data_dependent_channel_mixer"]
        or merged["mamba_variable_mixer"]
        or merged["mamba_patch_mixer"] or merged["mamba_local_mixer"]
    ):
        raise ValueError(
            "mamba_channel_gate, mamba_data_dependent_channel_mixer, "
            "mamba_variable_mixer, mamba_patch_mixer, and "
            "mamba_local_mixer "
            "require shared channels; "
            "disable mamba_channel_independent"
        )
    if merged["mamba_variable_gdd_mixer"] and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
    }:
        raise ValueError(
            "mamba_variable_gdd_mixer is supported only by MambaACF models"
        )
    if merged["mamba_variable_gdd_mixer"] and not merged["mamba_variable_mixer"]:
        raise ValueError(
            "mamba_variable_gdd_mixer requires mamba_variable_mixer=true"
        )
    if merged["mamba_acf_channel_gate"] and not merged["mamba_channel_gate"]:
        raise ValueError("mamba_acf_channel_gate requires mamba_channel_gate=true")
    if merged["mamba_data_dependent_channel_mixer"] and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
    }:
        raise ValueError(
            "mamba_data_dependent_channel_mixer is supported only by MambaACF models"
        )
    if merged["mamba_acf_scale_conditioning"] and len(merged["mamba_scales"]) < 2:
        raise ValueError(
            "mamba_acf_scale_conditioning requires at least two temporal scales"
        )
    if merged["mamba_acf_scale_conditioning"] and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
    }:
        raise ValueError(
            "mamba_acf_scale_conditioning is supported only by MambaACF models"
        )
    if (
        merged["mamba_acf_scale_conditioning"]
        and merged["AutoCon_acf_source"] != "raw"
    ):
        raise ValueError(
            "mamba_acf_scale_conditioning requires AutoCon_acf_source=raw"
        )
    if merged["mamba_acf_scale_conditioning"] and not any(
        int(lag) > 0 and int(merged["seq_len"]) - int(lag) >= 16
        for lag in merged["mamba_acf_lags"]
    ):
        raise ValueError(
            "mamba_acf_scale_conditioning requires at least one ACF lag "
            "with 16 or more window pairs"
        )
    if merged["mamba_acf_local_agreement"] and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
    }:
        raise ValueError(
            "mamba_acf_local_agreement is supported only by MambaACF models"
        )
    if merged["mamba_acf_local_agreement"]:
        if not merged["mamba_use_acf_memory"]:
            raise ValueError(
                "mamba_acf_local_agreement requires mamba_use_acf_memory=true"
            )
        if merged["mamba_dynamic_acf"]:
            raise ValueError(
                "mamba_acf_local_agreement requires static ACF memory"
            )
        if merged["AutoCon_acf_source"] != "raw":
            raise ValueError(
                "mamba_acf_local_agreement requires AutoCon_acf_source=raw"
            )
        if not any(
            int(lag) > 0
            and int(merged["seq_len"]) - int(lag) >= int(merged["mamba_acf_local_min_pairs"])
            for lag in merged["mamba_acf_lags"]
        ):
            raise ValueError(
                "mamba_acf_local_agreement requires at least one ACF lag with "
                "the configured minimum pair support"
            )
    if merged["mamba_acf_channel_gate"] and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"
    }:
        raise ValueError("mamba_acf_channel_gate is supported only by MambaACF models")
    if merged["mamba_acf_horizon_calibration_rank"] not in {1, 2}:
        raise ValueError("mamba_acf_horizon_calibration_rank must be 1 or 2")
    if merged["mamba_acf_horizon_calibration"]:
        if identifier not in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
        }:
            raise ValueError(
                "mamba_acf_horizon_calibration is supported only by MambaACF models"
            )
        if not merged["mamba_use_acf_memory"]:
            raise ValueError(
                "mamba_acf_horizon_calibration requires mamba_use_acf_memory=true"
            )
        if merged["mamba_dynamic_acf"]:
            raise ValueError(
                "mamba_acf_horizon_calibration requires static ACF memory"
            )
    if merged["mamba_acf_reliability_interpolation"]:
        if identifier not in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
        }:
            raise ValueError(
                "mamba_acf_reliability_interpolation is supported only by MambaACF models"
            )
        if not merged["mamba_use_acf_memory"]:
            raise ValueError(
                "mamba_acf_reliability_interpolation requires mamba_use_acf_memory=true"
            )
        if merged["mamba_dynamic_acf"]:
            raise ValueError(
                "mamba_acf_reliability_interpolation requires static ACF memory"
            )
        if merged["AutoCon_acf_source"] != "raw":
            raise ValueError(
                "mamba_acf_reliability_interpolation requires AutoCon_acf_source=raw"
            )
        if merged["mamba_acf_step_reliability"]:
            raise ValueError(
                "mamba_acf_reliability_interpolation is incompatible with "
                "mamba_acf_step_reliability=true"
            )
    if merged["mamba_acf_memory_gate_mode"] == "variable_token":
        if not merged["mamba_variable_mixer"]:
            raise ValueError(
                "mamba_acf_memory_gate_mode=variable_token requires "
                "mamba_variable_mixer=true"
            )
        if not merged["mamba_use_acf_memory"] or merged["mamba_dynamic_acf"]:
            raise ValueError(
                "mamba_acf_memory_gate_mode=variable_token requires static ACF memory"
            )
    if merged["mamba_acf_anchor_copy"]:
        if identifier not in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
        }:
            raise ValueError(
                "mamba_acf_anchor_copy is supported only by MambaACF models"
            )
        if merged["mamba_dynamic_acf"]:
            raise ValueError(
                "mamba_acf_anchor_copy requires static ACF memory"
            )
        if merged["AutoCon_acf_source"] != "raw":
            raise ValueError(
                "mamba_acf_anchor_copy requires AutoCon_acf_source=raw"
            )
        if not merged["mamba_acf_lags"]:
            raise ValueError(
                "mamba_acf_anchor_copy requires non-empty mamba_acf_lags"
            )
    if merged["mamba_patch_mixer"] and identifier not in {"MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"}:
        raise ValueError("mamba_patch_mixer is supported only by MambaACF models")
    if merged["mamba_local_mixer"] and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"
    }:
        raise ValueError("mamba_local_mixer is supported only by MambaACF models")
    if merged["mamba_token_stride"] != 1 and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"
    }:
        raise ValueError("mamba_token_stride is supported only by MambaACF models")
    if merged["mamba_local_mixer"] and merged["mamba_local_patch_len"] > merged["seq_len"]:
        raise ValueError("mamba_local_patch_len cannot exceed seq_len")
    local_patch_count = max(
        1,
        math.ceil(
            (merged["seq_len"] - merged["mamba_local_patch_len"])
            / merged["mamba_local_patch_stride"]
        )
        + 1,
    )
    if merged["mamba_local_mixer"] and merged["mamba_local_rank"] >= min(
        local_patch_count * merged["mamba_local_width"], merged["pred_len"]
    ):
        raise ValueError(
            "mamba_local_rank must be smaller than the local patch feature width and pred_len"
        )
    if merged["mamba_variable_permutation"] and not merged["mamba_variable_mixer"]:
        raise ValueError("mamba_variable_permutation requires mamba_variable_mixer=true")
    if (merged["mamba_acf_channel_order"] != "input"
            and not merged["mamba_variable_mixer"]
            and identifier != "NLinearVariableMambaAutoCon"):
        raise ValueError(
            "mamba_acf_channel_order={} requires mamba_variable_mixer=true".format(
                merged["mamba_acf_channel_order"]
            )
        )
    if (merged["mamba_acf_channel_order"] != "input" and identifier not in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon",
            "NLinearVariableMambaAutoCon",
    }):
        raise ValueError(
                "mamba_acf_channel_order={} requires a compatible variable-token model".format(
                merged["mamba_acf_channel_order"]
            )
        )
    if merged["mamba_variable_group_size"] != 1 and not merged["mamba_variable_mixer"]:
        raise ValueError("mamba_variable_group_size requires mamba_variable_mixer=true")
    if merged["mamba_variable_group_size"] != 1 and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"
    }:
        raise ValueError("mamba_variable_group_size is supported only by MambaACF models")
    if merged["mamba_variable_repr_fusion"]:
        if identifier not in {
            "MambaACFAutoCon", "MambaACF1AutoCon", "NLinearVariableMambaAutoCon"
        }:
            raise ValueError(
                "mamba_variable_repr_fusion requires a supported AutoCon model"
            )
        if (
            identifier != "NLinearVariableMambaAutoCon"
            and not merged["mamba_variable_mixer"]
        ):
            raise ValueError(
                "mamba_variable_repr_fusion requires mamba_variable_mixer=true"
            )
    if merged["mamba_variable_acf_memory_selector"]:
        if identifier != "NLinearVariableMambaAutoCon":
            raise ValueError(
                "mamba_variable_acf_memory_selector requires NLinearVariableMambaAutoCon"
            )
        if not merged["AutoCon"] or not merged["mamba_use_acf_memory"]:
            raise ValueError(
                "mamba_variable_acf_memory_selector requires AutoCon and static ACF memory"
            )
    if merged["mamba_acf_repr_fusion"]:
        if identifier in {"MambaTSv4AutoCon", "MambaTSv5AutoCon", "MambaTSv7AutoCon"}:
            if not merged["mamba_use_acf_memory"]:
                raise ValueError(
                    "MambaTS v4/v5 ACF representation fusion requires "
                    "mamba_use_acf_memory=true"
                )
            if merged["AutoCon_acf_source"] != "raw":
                raise ValueError(
                    "MambaTS v4/v5 ACF representation fusion requires "
                    "AutoCon_acf_source=raw"
                )
        elif identifier not in {
            "MambaACFAutoCon", "MambaACF1AutoCon"
        }:
            raise ValueError(
                "mamba_acf_repr_fusion requires a MambaACF AutoCon or "
                "MambaTSv4/v5 AutoCon model"
            )
    if merged["mamba_acf_channel_conditioning"] and not merged["mamba_variable_mixer"]:
        raise ValueError("mamba_acf_channel_conditioning requires mamba_variable_mixer=true")
    if merged["mamba_acf_channel_conditioning"] and identifier not in {
        "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"
    }:
        raise ValueError("mamba_acf_channel_conditioning is supported only by MambaACF models")
    if merged["mamba_patch_permutation"] and not merged["mamba_patch_mixer"]:
        raise ValueError("mamba_patch_permutation requires mamba_patch_mixer=true")
    if merged["mamba_low_rank_head"]:
        effective_temporal_length = merged["seq_len"]
        if identifier in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"
        }:
            effective_temporal_length = int(math.ceil(
                merged["seq_len"] / merged["mamba_token_stride"]
            ))
        if merged["mamba_temporal_rank"] >= min(
            effective_temporal_length, merged["pred_len"]
        ):
            raise ValueError(
                "mamba_temporal_rank must be smaller than min(effective temporal "
                "tokens={}, pred_len={})".format(
                    effective_temporal_length, merged["pred_len"]
                )
            )
    if merged["task_name"] not in {"long_term_forecast", "short_term_forecast"}:
        raise ValueError("only forecasting task configs are supported")
    if merged["features"] not in {"S", "M", "MS"}:
        raise ValueError("features must be one of S, M, MS")
    if identifier not in MODEL_MODULES:
        _model_module(merged)  # raises the canonical error message
    if merged["model"] in {"Mamba1", "Mamba2", "MambaTS", "MambaTSv2", "MambaTSv4", "MambaTSv5", "MambaTSv6", "MambaTSv7"} and merged["AutoCon"]:
        raise ValueError("plain Mamba identifier cannot have AutoCon=true")
    if merged["model"] in {
        "Mamba1AutoCon", "Mamba2AutoCon", "MambaACFAutoCon",
        "MambaTSAutoCon", "MambaTSv2AutoCon", "MambaTSv4AutoCon",
        "MambaTSv5AutoCon", "MambaTSv6AutoCon", "MambaTSv7AutoCon",
        "MambaACFVariableAutoCon", "MambaACFLiteAutoCon", "MambaACF1AutoCon",
    } and not merged["AutoCon"]:
        raise ValueError("Mamba AutoCon identifier requires AutoCon=true")
    if identifier in {"MambaACFAutoCon", "MambaACF1AutoCon"} and not merged["mamba_acf_relation_loss"]:
        raise ValueError(
            "MambaACF AutoCon resource configs require mamba_acf_relation_loss=true"
        )
    if merged["mamba_acf_relation_loss"] and identifier not in {
        "MambaACFAutoCon", "MambaACF1AutoCon",
        "NLinearVariableMambaAutoCon",
    }:
        raise ValueError("mamba_acf_relation_loss requires a relation-capable AutoCon model")
    if merged["mamba_acf_relation_representation"] == "forecast":
        if identifier not in {"MambaACFAutoCon", "MambaACF1AutoCon"}:
            raise ValueError(
                "mamba_acf_relation_representation=forecast requires an AutoCon MambaACF model"
            )
        if not merged["mamba_acf_relation_loss"]:
            raise ValueError(
                "mamba_acf_relation_representation=forecast requires mamba_acf_relation_loss=true"
            )
        if merged["mamba_acf_relation_scope"] != "global":
            raise ValueError(
                "mamba_acf_relation_representation=forecast requires mamba_acf_relation_scope=global"
            )
        if merged["AutoCon_acf_source"] != "raw":
            raise ValueError(
                "mamba_acf_relation_representation=forecast requires AutoCon_acf_source=raw"
            )
        if str(merged["features"]).upper() != "M":
            raise ValueError(
                "mamba_acf_relation_representation=forecast requires features=M"
            )
        if int(merged["c_out"]) != int(merged["enc_in"]):
            raise ValueError(
                "mamba_acf_relation_representation=forecast requires c_out=enc_in"
            )
        if int(merged["enc_in"]) < 2:
            raise ValueError(
                "mamba_acf_relation_representation=forecast requires enc_in>=2"
            )
        if merged["mamba_acf_repr_fusion"] or merged["mamba_variable_repr_fusion"]:
            raise ValueError(
                "mamba_acf_relation_representation=forecast is incompatible with encoder-only representation fusion options"
            )
    if merged["mamba_require_strict_representation_causality"]:
        if identifier not in {
            "MambaACF", "MambaACF1", "MambaACFAutoCon", "MambaACF1AutoCon"
        }:
            raise ValueError(
                "mamba_require_strict_representation_causality is supported "
                "only by MambaACF models"
            )
        causality_errors = strict_representation_causality_errors(
            merged, model=identifier
        )
        if causality_errors:
            raise ValueError(
                "strict MambaACF representation causality contract failed: "
                + ", ".join(causality_errors)
            )
    return merged


def _freq_key(freq: str) -> str:
    value = str(freq).lower()
    if value in TIME_FEATURE_DIMS:
        return value
    if "min" in value or value.endswith("t"):
        return "t"
    if value.endswith("s"):
        return "s"
    if value.endswith("h"):
        return "h"
    if value.endswith("d"):
        return "d"
    if value.endswith("w"):
        return "w"
    if value.endswith("b"):
        return "b"
    if value.endswith("m"):
        return "m"
    raise ValueError("unsupported frequency for synthetic time marks: {!r}".format(freq))


def _time_feature_dim(config: Mapping[str, Any]) -> int:
    return TIME_FEATURE_DIMS[_freq_key(str(config["freq"]))]


def _resource_signature(config: Mapping[str, Any]) -> str:
    payload = {name: config.get(name) for name in RESOURCE_SIGNATURE_FIELDS}
    return _json_hash(payload)


def _setup_source_root(source_root: Path) -> None:
    """Make the frozen phase source take precedence for the next model import."""
    root = source_root.resolve()
    required_packages = ("models", "layers", "utils")
    missing_packages = [
        name for name in required_packages if not (root / name).is_dir()
    ]
    if missing_packages:
        raise ValueError(
            "source root does not contain {}: {}".format(
                ", ".join(name + "/" for name in missing_packages), root
            )
        )
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    # A single output may intentionally compare validated phases.  Clear every
    # repository package imported by model construction before moving to a
    # different immutable snapshot; leaving ``utils`` cached would combine a
    # frozen model with live ACF/causality helpers.
    for name in list(sys.modules):
        if any(
            name == package or name.startswith(package + ".")
            for package in required_packages
        ):
            del sys.modules[name]
    importlib.invalidate_caches()


def _model_class(source_root: Path, config: Mapping[str, Any]):
    _setup_source_root(source_root)
    module = importlib.import_module("models." + _model_module(config))
    return module.Model


def _validation_context(validation_path: Path) -> Dict[str, Any]:
    validation = _load_json(validation_path)
    if validation.get("valid") is not True or validation.get("errors"):
        raise ValueError("phase validation is not clean/valid: {}".format(validation_path))
    manifest_text = validation.get("manifest")
    manifest_hash = validation.get("manifest_sha256")
    if not manifest_text or not manifest_hash:
        raise ValueError("validation has no manifest path/hash: {}".format(validation_path))
    manifest_path = Path(str(manifest_text)).resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != str(manifest_hash):
        raise ValueError("manifest no longer matches the validated hash: {}".format(manifest_path))
    manifest = _read_manifest(manifest_path)
    expected_hashes = validation.get("artifact_hashes_by_job")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise ValueError("validation has no artifact hashes: {}".format(validation_path))
    provenance_text = validation.get("provenance")
    if not provenance_text:
        raise ValueError("validation has no provenance path: {}".format(validation_path))
    provenance_path = Path(str(provenance_text)).resolve()
    provenance = _load_json(provenance_path)
    if validation.get("provenance_id") != provenance.get("provenance_id"):
        raise ValueError("validation and provenance identifiers disagree")
    source_root = Path(str(provenance.get("source_snapshot", ""))).resolve()
    if not source_root.is_dir():
        raise ValueError("validated source snapshot is missing: {}".format(source_root))
    expected_source_files = provenance.get("source_files")
    expected_source_tree = provenance.get("source_tree_sha256")
    if not isinstance(expected_source_files, dict) or not expected_source_tree:
        raise ValueError("provenance has no source-tree attestation: {}".format(provenance_path))
    source_files, source_tree = _hash_source_tree(source_root)
    if source_files != expected_source_files or source_tree != expected_source_tree:
        raise ValueError("validated source snapshot changed: {}".format(source_root))

    by_config: Dict[Path, Dict[str, Any]] = {}
    for job_id, hashes in expected_hashes.items():
        if not isinstance(hashes, dict) or "config.json" not in hashes:
            raise ValueError("validation lacks config hash for job {}".format(job_id))
        record = manifest.get(str(job_id))
        if not record or record.get("status") != "ok":
            raise ValueError("validated manifest record is absent/non-successful: {}".format(job_id))
        artifact = record.get("expected_artifact")
        if not artifact:
            raise ValueError("manifest record has no artifact path: {}".format(job_id))
        config_path = (Path(str(artifact)).resolve().parent / "config.json").resolve()
        if not config_path.is_file():
            raise ValueError("validated config is missing: {}".format(config_path))
        if config_path in by_config:
            raise ValueError("validated manifest maps multiple jobs to one config: {}".format(config_path))
        by_config[config_path] = {
            "job_id": str(job_id),
            "expected_config_sha256": str(hashes["config.json"]),
            "source_root": source_root,
            "phase": validation.get("phase"),
            "provenance_id": validation.get("provenance_id"),
            "validation_path": validation_path,
        }
    return {
        "validation": validation,
        "manifest_path": manifest_path,
        "manifest_records": manifest,
        "by_config": by_config,
        "source_root": source_root,
    }


def _validated_requests(config_paths: Sequence[Path], context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    seen: set = set()
    for path in config_paths:
        config_path = _resolve_path(path)
        if config_path in seen:
            continue
        seen.add(config_path)
        item = context["by_config"].get(config_path)
        if item is None:
            raise ValueError(
                "config is not an artifact in the supplied phase validation: {}".format(config_path)
            )
        actual_hash = _sha256(config_path)
        if actual_hash != item["expected_config_sha256"]:
            raise ValueError("config hash changed after validation: {}".format(config_path))
        config = _normalise_config(_load_json(config_path))
        if config.get("plan_job_id") != item["job_id"]:
            raise ValueError("config plan_job_id disagrees with manifest: {}".format(config_path))
        requests.append({
            "config": config,
            "source_root": item["source_root"],
            "source": {
                "kind": "validated_config",
                "config_path": str(config_path),
                "config_sha256": actual_hash,
                "phase": item["phase"],
                "provenance_id": item["provenance_id"],
                "validation_path": str(item["validation_path"]),
                "job_id": item["job_id"],
            },
        })
    return requests


def _unverified_requests(config_paths: Sequence[Path], source_root: Path) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    seen: set = set()
    for path in config_paths:
        config_path = _resolve_path(path)
        if config_path in seen:
            continue
        seen.add(config_path)
        if not config_path.is_file():
            raise ValueError("config does not exist: {}".format(config_path))
        requests.append({
            "config": _normalise_config(_load_json(config_path)),
            "source_root": source_root,
            "source": {
                "kind": "unverified_config",
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
            },
        })
    return requests


def _explicit_request(args: argparse.Namespace, source_root: Path) -> Dict[str, Any]:
    required = ("seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError("explicit CLI config requires: {}".format(
            ", ".join("--" + name.replace("_", "-") for name in missing)
        ))
    config = dict(EXPLICIT_DEFAULTS)
    config.update({
        "model": args.model,
        "data": args.dataset,
        "features": args.features,
        "freq": args.freq,
        "embed": args.embed,
        "activation": args.activation,
        "wo_freq": args.wo_freq,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "enc_in": args.enc_in,
        "dec_in": args.dec_in,
        "c_out": args.c_out,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "patch_len": args.patch_len,
        "patch_stride": args.patch_stride,
        "mamba_ts_variant": (
            "v16" if str(args.model).lower().startswith("mambatsv16")
            else "v15" if str(args.model).lower().startswith("mambatsv15")
            else "v14" if str(args.model).lower().startswith("mambatsv14")
            else "v13" if str(args.model).lower().startswith("mambatsv13")
            else "v12" if str(args.model).lower().startswith("mambatsv12")
            else "v11" if str(args.model).lower().startswith("mambatsv11")
            else "v10" if str(args.model).lower().startswith("mambatsv10")
            else "v9" if str(args.model).lower().startswith("mambatsv9")
            else "v8" if str(args.model).lower().startswith("mambatsv8")
            else "v7" if str(args.model).lower().startswith("mambatsv7")
            else "v6" if str(args.model).lower().startswith("mambatsv6")
            else "v5" if str(args.model).lower().startswith("mambatsv5")
            else "v4" if str(args.model).lower().startswith("mambatsv4")
            else "v3" if str(args.model).lower().startswith("mambatsv3")
            else "v2" if str(args.model).lower().startswith("mambatsv2")
            else args.mamba_ts_variant
        ),
        "mamba_ts_context_rank": args.mamba_ts_context_rank,
        "mamba_ts_mixer_type": args.mamba_ts_mixer_type,
        "mamba_ts_spectral_top_k": args.mamba_ts_spectral_top_k,
        "mamba_ts_repr_forecast_fusion": args.mamba_ts_repr_forecast_fusion,
        "mamba_ts_repr_forecast_channelwise": args.mamba_ts_repr_forecast_channelwise,
        "mamba_ts_channel_adapter": args.mamba_ts_channel_adapter,
        "mamba_ts_channel_adapter_rank": args.mamba_ts_channel_adapter_rank,
        "mamba_ts_revin_affine": args.mamba_ts_revin_affine,
        "mamba_ts_global_channel_mixer": args.mamba_ts_global_channel_mixer,
        "mamba_ts_global_channel_mixer_rank": args.mamba_ts_global_channel_mixer_rank,
        "mamba_ts_graph_spatial": args.mamba_ts_graph_spatial,
        "mamba_ts_graph_path": args.mamba_ts_graph_path,
        "mamba_ts_graph_expected_sha256": args.mamba_ts_graph_expected_sha256,
        "mamba_ts_repr_linear_expert": args.mamba_ts_repr_linear_expert,
        "mamba_ts_repr_linear_expert_mode": args.mamba_ts_repr_linear_expert_mode,
        "mamba_ts_repr_linear_blend_bias": args.mamba_ts_repr_linear_blend_bias,
        "mamba_ts_repr_linear_expert_rank": args.mamba_ts_repr_linear_expert_rank,
        "mamba_ts_variable_permutation": args.mamba_ts_variable_permutation,
        "mamba_ts_acf_memory_selector": args.mamba_ts_acf_memory_selector,
        "mamba_ts_acf_memory_channel_selector": args.mamba_ts_acf_memory_channel_selector,
        "mamba_ts_forecast_acf_consistency": args.mamba_ts_forecast_acf_consistency,
        "mamba_ts_forecast_acf_weight": args.mamba_ts_forecast_acf_weight,
        "mamba_ts_forecast_acf_top_k": args.mamba_ts_forecast_acf_top_k,
        "mamba_ts_forecast_acf_min_lag": args.mamba_ts_forecast_acf_min_lag,
        "mamba_ts_forecast_acf_min_abs": args.mamba_ts_forecast_acf_min_abs,
        "mamba_d_state": args.mamba_d_state,
        "mamba_d_conv": args.mamba_d_conv,
        "mamba_expand": args.mamba_expand,
        "mamba_headdim": args.mamba_headdim,
        "mamba_tm_n1": args.mamba_tm_n1,
        "mamba_tm_n2": args.mamba_tm_n2,
        "mamba_tm_repr_width": args.mamba_tm_repr_width,
        "mamba_tm_channel_independent": args.mamba_tm_channel_independent,
        "mamba_tm_residual": args.mamba_tm_residual,
        "mamba_bipatch_token_mode": args.mamba_bipatch_token_mode,
        "mamba_scales": args.mamba_scales,
        "mamba_token_stride": args.mamba_token_stride,
        "mamba_bidirectional": args.mamba_bidirectional,
        "mamba_channel_independent": args.mamba_channel_independent,
        "mamba_channel_gate": args.mamba_channel_gate,
        "mamba_channel_gate_rank": args.mamba_channel_gate_rank,
        "mamba_data_dependent_channel_mixer": args.mamba_data_dependent_channel_mixer,
        "mamba_data_dependent_channel_mixer_rank": args.mamba_data_dependent_channel_mixer_rank,
        "mamba_acf_scale_conditioning": args.mamba_acf_scale_conditioning,
        "mamba_acf_local_agreement": args.mamba_acf_local_agreement,
        "mamba_acf_local_min_pairs": args.mamba_acf_local_min_pairs,
        "mamba_acf_channel_gate": args.mamba_acf_channel_gate,
        "mamba_causal_embedding": args.mamba_causal_embedding,
        "mamba_require_strict_representation_causality": (
            args.mamba_require_strict_representation_causality
        ),
        "mamba_variable_mixer": args.mamba_variable_mixer,
        "mamba_variable_gdd_mixer": args.mamba_variable_gdd_mixer,
        "mamba_variable_gdd_mixer_rank": args.mamba_variable_gdd_mixer_rank,
        "mamba_variable_gdd_init_scale": args.mamba_variable_gdd_init_scale,
        "mamba_variable_mixer_depth": args.mamba_variable_mixer_depth,
        "mamba_variable_group_size": args.mamba_variable_group_size,
        "mamba_variable_repr_fusion": args.mamba_variable_repr_fusion,
        "mamba_variable_acf_memory_selector": args.mamba_variable_acf_memory_selector,
        "mamba_acf_repr_fusion": args.mamba_acf_repr_fusion,
        "mamba_variable_permutation": args.mamba_variable_permutation,
        "mamba_acf_channel_order": args.mamba_acf_channel_order,
        "mamba_acf_channel_conditioning": args.mamba_acf_channel_conditioning,
        "mamba_patch_mixer": args.mamba_patch_mixer,
        "mamba_patch_mixer_depth": args.mamba_patch_mixer_depth,
        "mamba_patch_len": args.mamba_patch_len,
        "mamba_patch_stride": args.mamba_patch_stride,
        "mamba_patch_order": args.mamba_patch_order,
        "mamba_patch_permutation": args.mamba_patch_permutation,
        "mamba_local_mixer": args.mamba_local_mixer,
        "mamba_local_patch_len": args.mamba_local_patch_len,
        "mamba_local_patch_stride": args.mamba_local_patch_stride,
        "mamba_local_width": args.mamba_local_width,
        "mamba_local_rank": args.mamba_local_rank,
        "mamba_local_depth": args.mamba_local_depth,
        "mamba_low_rank_head": args.mamba_low_rank_head,
        "mamba_temporal_rank": args.mamba_temporal_rank,
        "mamba_acf_lags": args.mamba_acf_lags,
        "mamba_acf_channel_lag_bank": args.mamba_acf_channel_lag_bank,
        "mamba_acf_channel_lag_top_k": args.mamba_acf_channel_lag_top_k,
        "mamba_acf_lag_mode": args.mamba_acf_lag_mode,
        "mamba_acf_max_lag": args.mamba_acf_max_lag,
        "mamba_acf_top_k": args.mamba_acf_top_k,
        "mamba_acf_min_lag": args.mamba_acf_min_lag,
        "mamba_acf_min_abs": args.mamba_acf_min_abs,
        "mamba_acf_confidence_z": args.mamba_acf_confidence_z,
        "mamba_acf_min_separation": args.mamba_acf_min_separation,
        "mamba_acf_detrend": args.mamba_acf_detrend,
        "mamba_acf_chunk": args.mamba_acf_chunk,
        "mamba_use_acf_memory": args.mamba_use_acf_memory,
        "mamba_dynamic_acf": args.mamba_dynamic_acf,
        "mamba_acf_temperature": args.mamba_acf_temperature,
        "mamba_acf_step_reliability": args.mamba_acf_step_reliability,
        "mamba_acf_reliability_interpolation": args.mamba_acf_reliability_interpolation,
        "mamba_acf_memory_gate_mode": args.mamba_acf_memory_gate_mode,
        "mamba_acf_anchor_copy": args.mamba_acf_anchor_copy,
        "mamba_acf_horizon_calibration": args.mamba_acf_horizon_calibration,
        "mamba_acf_horizon_calibration_rank": args.mamba_acf_horizon_calibration_rank,
        "mamba_acf_relation_loss": args.mamba_acf_relation_loss,
        "mamba_acf_relation_batching": args.mamba_acf_relation_batching,
        "mamba_acf_relation_lag_balance": args.mamba_acf_relation_lag_balance,
        "mamba_acf_relation_channel_reduce": args.mamba_acf_relation_channel_reduce,
        "mamba_acf_relation_scope": args.mamba_acf_relation_scope,
        "mamba_acf_relation_representation": args.mamba_acf_relation_representation,
        "mamba_acf_relation_lag_mask": args.mamba_acf_relation_lag_mask,
        "mamba_acf_relation_lag_tolerance": args.mamba_acf_relation_lag_tolerance,
        "mamba_acf_relation_min_lag": args.mamba_acf_relation_min_lag,
        "mamba_acf_relation_target_mode": args.mamba_acf_relation_target_mode,
        "mamba_acf_relation_soft_min_abs": args.mamba_acf_relation_soft_min_abs,
        "mamba_acf_relation_diagnostic_min_abs": args.mamba_acf_relation_diagnostic_min_abs,
        "mamba_acf_relation_selected_lag_boost": args.mamba_acf_relation_selected_lag_boost,
        "moving_avg": args.moving_avg,
        "factor": args.factor,
        "top_k": args.top_k,
        "num_kernels": args.num_kernels,
        "timemixer_down_sampling_window": args.timemixer_down_sampling_window,
        "timemixer_down_sampling_layers": args.timemixer_down_sampling_layers,
        "timemixer_channel_independence": args.timemixer_channel_independence,
        "timemixer_decomp_method": args.timemixer_decomp_method,
        "dropout": args.dropout,
        "AutoCon_wnorm": args.autocon_wnorm,
        "AutoCon_lambda": args.autocon_lambda,
        "AutoCon_warmup_epochs": args.autocon_warmup_epochs,
        "AutoCon_acf_source": args.autocon_acf_source,
        "AutoCon_temperature": args.autocon_temperature,
        "AutoCon_relation_temperature": args.autocon_relation_temperature,
        "AutoCon_sign_weight": args.autocon_sign_weight,
        "AutoCon_sign_margin": args.autocon_sign_margin,
        "AutoCon_multiscales": args.autocon_scales,
        "individual": args.individual,
    })
    inferred_autocon = args.model in AUTOCON_IDENTIFIERS
    config["AutoCon"] = inferred_autocon if args.autocon is None else bool(args.autocon)
    if args.model == "AutoConNetAutoCon":
        config["model"] = "AutoConNet"
        config["AutoCon"] = True
    config = _normalise_config(config)
    return {
        "config": config,
        "source_root": source_root,
        "source": {
            "kind": "explicit_cli",
            "config_sha256": _json_hash(config),
        },
    }


def _deduplicate_requests(requests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduplicated: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for request in requests:
        key = (str(Path(request["source_root"]).resolve()), _resource_signature(request["config"]))
        if key not in deduplicated:
            item = dict(request)
            item["duplicate_sources"] = []
            deduplicated[key] = item
        else:
            deduplicated[key]["duplicate_sources"].append(request["source"])
    return list(deduplicated.values())


def _set_seeds(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_inputs(torch: Any, config: Mapping[str, Any], batch_size: int, device: Any) -> Tuple[Any, Any, Any, Any]:
    mark_dtype = torch.float32 if str(config["embed"]).lower() == "timef" else torch.long
    mark_dim = _time_feature_dim(config)
    x_enc = torch.randn(batch_size, int(config["seq_len"]), int(config["enc_in"]), device=device)
    x_dec = torch.zeros(
        batch_size,
        int(config["label_len"]) + int(config["pred_len"]),
        int(config["dec_in"]),
        dtype=torch.float32,
        device=device,
    )
    x_mark_enc = torch.zeros(batch_size, int(config["seq_len"]), mark_dim, dtype=mark_dtype, device=device)
    x_mark_dec = torch.zeros(
        batch_size,
        int(config["label_len"]) + int(config["pred_len"]),
        mark_dim,
        dtype=mark_dtype,
        device=device,
    )
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def _task_prediction(prediction: Any, config: Mapping[str, Any], torch: Any) -> Any:
    """Apply the same output-channel contract as the forecasting trainers."""

    if not isinstance(prediction, torch.Tensor):
        raise ValueError("model forward did not return a prediction tensor")
    if str(config.get("features", "M")) == "MS":
        if prediction.ndim != 3 or prediction.shape[-1] < 1:
            raise ValueError(
                "MS synthetic forward must return a [batch, time, channel] tensor before target slicing"
            )
        # All long-term trainers use f_dim=-1 for MS.  Mamba and some
        # baselines internally forecast enc_in channels before this task-level
        # target projection, so resource measurement must do the same slice.
        return prediction[:, :, -1:]
    return prediction


def _forward_prediction(
    model: Any,
    inputs: Tuple[Any, Any, Any, Any],
    torch: Any,
    config: Optional[Mapping[str, Any]] = None,
) -> Any:
    prediction, _ = _forward_outputs(
        model, inputs, torch, config=config, request_representation=False
    )
    return prediction


def _forward_outputs(
    model: Any,
    inputs: Tuple[Any, Any, Any, Any],
    torch: Any,
    config: Optional[Mapping[str, Any]] = None,
    request_representation: bool = False,
) -> Tuple[Any, Optional[Any]]:
    """Return the task prediction and optional AutoCon representation.

    Models declaring ``supports_return_repr_control`` receive an explicit
    boolean so the resource protocol can measure either the deployed forecast
    path or the representation-producing path.  Legacy models without that
    capability retain their default forward contract.
    """

    model_for_contract = getattr(model, "module", model)
    if getattr(model_for_contract, "supports_return_repr_control", False):
        output = model(*inputs, return_repr=bool(request_representation))
    else:
        output = model(*inputs)
    if isinstance(output, (tuple, list)):
        prediction = output[0]
        representation = output[1] if len(output) > 1 else None
    else:
        prediction = output
        representation = None
    return _task_prediction(prediction, config or {}, torch), representation


def _representation_shape(
    representation: Any,
    torch: Any,
    *,
    required: bool,
) -> Optional[List[int]]:
    """Validate and describe a returned representation without assuming rank."""

    if representation is None:
        if required:
            raise ValueError(
                "representation-inclusive benchmark requested, but the model "
                "returned no auxiliary representation"
            )
        return None
    if not isinstance(representation, torch.Tensor):
        raise ValueError("model auxiliary representation is not a tensor")
    if representation.ndim < 2:
        raise ValueError(
            "model auxiliary representation must have at least batch/time dimensions"
        )
    if not bool(torch.isfinite(representation).all().item()):
        raise ValueError("non-finite synthetic auxiliary representation")
    return [int(size) for size in representation.shape]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _benchmark_gpu(
    torch: Any,
    model: Any,
    config: Mapping[str, Any],
    batch_size: int,
    warmup_iters: int,
    timed_iters: int,
    input_seed: int,
    device_name: str,
    include_representation: bool = False,
) -> Dict[str, Any]:
    device = torch.device(device_name)
    # ``torch.device('cuda')`` has an unspecified index.  Resolve it before
    # selecting the device so the common CLI spelling behaves like ``cuda:0``
    # and the recorded resource contract is unambiguous.
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type != "cuda":
        raise ValueError("--run-benchmark requires a CUDA device, not {}".format(device_name))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot collect GPU peak memory or device latency")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("CUDA device index is out of range: {}".format(device_name))

    # Events and the default stream follow the current CUDA device.  Explicitly
    # select it so ``--device cuda:1`` measures the requested GPU rather than
    # recording an event on device 0.
    torch.cuda.set_device(device)
    # A single process may measure many deduplicated architectures.  Release
    # the previous model and allocator cache so reserved memory is not a
    # property of measurement order.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _set_seeds(torch, input_seed)
    # Warmup deliberately absorbs Mamba/Triton lazy initialization.  The
    # reported peak below is the subsequent steady-state forward peak.
    torch.backends.cudnn.benchmark = False
    model = model.to(device)
    model.eval()
    inputs = _make_inputs(torch, config, batch_size, device)
    model_for_contract = getattr(model, "module", model)
    supports_repr_control = bool(
        getattr(model_for_contract, "supports_return_repr_control", False)
    )
    torch.cuda.synchronize(device)
    allocated_after_setup = int(torch.cuda.memory_allocated(device))
    reserved_after_setup = int(torch.cuda.memory_reserved(device))

    with torch.inference_mode():
        prediction, representation = _forward_outputs(
            model,
            inputs,
            torch,
            config,
            request_representation=bool(include_representation),
        )
        expected_shape = (batch_size, int(config["pred_len"]), int(config["c_out"]))
        if tuple(prediction.shape) != expected_shape:
            raise ValueError(
                "unexpected prediction shape {}, expected {}".format(tuple(prediction.shape), expected_shape)
            )
        if not bool(torch.isfinite(prediction).all().item()):
            raise ValueError("non-finite synthetic forward prediction")
        representation_shape = _representation_shape(
            representation,
            torch,
            required=bool(include_representation),
        )
        del prediction, representation
        for _ in range(warmup_iters):
            prediction, representation = _forward_outputs(
                model,
                inputs,
                torch,
                config,
                request_representation=bool(include_representation),
            )
            if include_representation and representation is None:
                raise ValueError(
                    "representation-inclusive benchmark requested, but the model "
                    "returned no auxiliary representation"
                )
            del prediction, representation
    torch.cuda.synchronize(device)

    steady_state_allocated = int(torch.cuda.memory_allocated(device))
    steady_state_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    latencies_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(timed_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            prediction, representation = _forward_outputs(
                model,
                inputs,
                torch,
                config,
                request_representation=bool(include_representation),
            )
            end.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(float(start.elapsed_time(end)))
            # The first forward above validates type, shape and finiteness.
            # Avoid an additional reduction here so timed forwards and peak
            # memory represent model execution rather than validation work.
            if include_representation and representation is None:
                raise ValueError(
                    "representation-inclusive benchmark requested, but the model "
                    "returned no auxiliary representation"
                )
            del prediction, representation
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    mean_latency_ms = float(statistics.fmean(latencies_ms))
    throughput = 1000.0 * batch_size / mean_latency_ms if mean_latency_ms > 0 else float("inf")

    return {
        "status": "ok",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "benchmark_batch_size": int(batch_size),
        "warmup_forwards": int(warmup_iters),
        "timed_forwards": int(timed_iters),
        "input_seed": int(input_seed),
        "input_protocol": "synthetic device-resident input; transfer and data loading excluded",
        "forward_path": (
            "prediction_plus_auxiliary_representation"
            if include_representation and representation_shape is not None
            else (
                "prediction_only_auxiliary_representation_disabled"
                if supports_repr_control
                else "model_default_forward"
            )
        ),
        "auxiliary_representation_requested": bool(include_representation),
        "auxiliary_representation_controlled": supports_repr_control,
        "auxiliary_representation_returned": representation_shape is not None,
        "auxiliary_representation_shape": representation_shape,
        "relation_loss_included": False,
        "prediction_contract": (
            "trainer_MS_target_last_channel" if str(config.get("features", "M")) == "MS"
            else "model_output"
        ),
        "latency_ms_mean": mean_latency_ms,
        "latency_ms_median": float(statistics.median(latencies_ms)),
        "latency_ms_p95": _percentile(latencies_ms, 95.0),
        "synthetic_input_sequences_per_sec": throughput,
        "memory_after_model_and_inputs_allocated_mb": allocated_after_setup / (1024.0 ** 2),
        "memory_after_model_and_inputs_reserved_mb": reserved_after_setup / (1024.0 ** 2),
        "steady_state_allocated_mb": steady_state_allocated / (1024.0 ** 2),
        "steady_state_reserved_mb": steady_state_reserved / (1024.0 ** 2),
        "steady_state_peak_allocated_mb": peak_allocated / (1024.0 ** 2),
        "steady_state_peak_reserved_mb": peak_reserved / (1024.0 ** 2),
        "steady_state_peak_increment_mb": max(0, peak_allocated - steady_state_allocated) / (1024.0 ** 2),
        "cudnn_benchmark": False,
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _activate_resource_acf_channel_order(model, config: Mapping[str, Any], torch) -> str:
    """Attach synthetic ACF metadata required by a resource benchmark.

    This tool deliberately does not load a dataset, so it cannot fit the
    real train-split ACF table.  A deterministic synthetic table is sufficient
    to activate the same model branches and buffers as a train-only sidecar;
    it is never presented as forecasting evidence or a data-derived channel
    order.
    """

    # Keep resource construction aligned with training/replay.  In particular,
    # channel conditioning and representation fusion consume a train-only
    # prior even when static forecast memory is disabled.
    needs_prior = mamba_acf_prior_required(config)
    if not needs_prior:
        return "not_required"
    setter = getattr(model, "set_acf_prior", None)
    if setter is None:
        setter = getattr(model, "set_ts_acf_prior", None)
    if setter is None:
        return "unsupported"
    lags = [int(value) for value in config.get("mamba_acf_lags", [24, 168, 336])]
    lags = sorted({lag for lag in lags if lag > 0})
    if not lags:
        raise ValueError("ACF resource path requires positive ACF lags")
    channels = int(config["enc_in"])
    prior = torch.zeros((channels, max(lags) + 1), dtype=torch.float32)
    # Descending per-channel values create a non-identity order for C>1 while
    # avoiding an expensive or data-dependent fit in this analysis-only tool.
    prior[:, lags[0]] = torch.linspace(1.0, 0.0, channels, dtype=torch.float32)
    setter(prior, lags=lags)
    markers = []
    if str(config.get("mamba_acf_channel_order", "input")).lower() != "input":
        markers.append("order")
    if bool(config.get("mamba_acf_channel_gate", False)):
        markers.append("gate")
    if bool(config.get("mamba_acf_scale_conditioning", False)):
        markers.append("scale")
    if bool(config.get("mamba_acf_local_agreement", False)):
        markers.append("local_agreement")
    if bool(config.get("mamba_use_acf_memory", False)) and not bool(
        config.get("mamba_dynamic_acf", False)
    ):
        markers.append("memory")
    if bool(config.get("mamba_acf_reliability_interpolation", False)):
        markers.append("reliability_interpolation")
    if bool(config.get("mamba_acf_anchor_copy", False)):
        markers.append("anchor_copy")
    if bool(config.get("mamba_acf_channel_lag_bank", False)):
        markers.append("channel_lag_bank")
    if bool(config.get("mamba_acf_channel_conditioning", False)):
        markers.append("channel_conditioning")
    if bool(config.get("AutoCon", False)) and bool(
        config.get("mamba_acf_repr_fusion", False)
    ):
        markers.append("representation_fusion")
    if bool(config.get("mamba_ts_acf_memory_selector", False)):
        markers.append("representation_selected_memory")
    if bool(config.get("mamba_ts_acf_memory_channel_selector", False)):
        markers.append("channel_conditioned_selected_memory")
    if bool(config.get("mamba_acf_relation_loss", False)):
        markers.append("relation")
    if markers == ["order"]:
        return "synthetic_deterministic_activation"
    return "synthetic_deterministic_acf_prior+" + "+".join(markers)


def _measure_request(request: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Import torch only after CLI validation.  This makes --help and manifest
    # inspection safe on a CPU-only shell and avoids touching CUDA unless the
    # caller explicitly asks for a benchmark.
    import torch

    config = request["config"]
    _set_seeds(torch, int(args.input_seed))
    source_root = Path(request["source_root"]).resolve()
    model_module = _model_module(config)
    model_source_path = source_root / "models" / (model_module + ".py")
    if not model_source_path.is_file():
        raise ValueError(
            "model source file is missing for resource attestation: {}".format(
                model_source_path
            )
        )
    model_cls = _model_class(source_root, config)
    model = model_cls(SimpleNamespace(**config)).float()
    acf_channel_order_prior_source = _activate_resource_acf_channel_order(
        model, config, torch
    )
    model.eval()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    parameter_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))
    buffer_bytes = int(sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()))
    effective_mamba = {
        name: getattr(model, name, None)
        for name in (
            "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_headdim",
            "acf_channel_order",
        )
    }
    channel_profile = (
        model.resource_profile()
        if hasattr(model, "resource_profile") else None
    )

    benchmark: Dict[str, Any]
    if args.run_benchmark:
        benchmark = _benchmark_gpu(
            torch=torch,
            model=model,
            config=config,
            batch_size=int(args.benchmark_batch_size),
            warmup_iters=int(args.warmup_iters),
            timed_iters=int(args.timed_iters),
            input_seed=int(args.input_seed),
            device_name=str(args.device),
            include_representation=bool(args.include_representation),
        )
    else:
        benchmark = {
            "status": "not_requested",
            "reason": "GPU execution requires explicit --run-benchmark",
            "auxiliary_representation_requested": bool(args.include_representation),
            "auxiliary_representation_controlled": bool(
                getattr(getattr(model, "module", model), "supports_return_repr_control", False)
            ),
            "relation_loss_included": False,
        }

    architecture = {name: config.get(name) for name in RESOURCE_SIGNATURE_FIELDS}
    source_attestation = dict(request["source"])
    source_attestation.update({
        "model_module": model_module,
        "model_source_file": str(model_source_path.relative_to(source_root)),
        "model_source_sha256": _sha256(model_source_path),
        "benchmark_tool_sha256": _sha256(Path(__file__).resolve()),
    })
    mixer_source_path = source_root / "layers" / "LowRankDataDependentChannelMixer.py"
    if mixer_source_path.is_file():
        source_attestation["mixer_source_file"] = str(
            mixer_source_path.relative_to(source_root)
        )
        source_attestation["mixer_source_sha256"] = _sha256(mixer_source_path)
    return {
        "model": _model_identifier(config),
        "dataset": config.get("data"),
        "features": config.get("features"),
        "seq_len": int(config["seq_len"]),
        "label_len": int(config["label_len"]),
        "pred_len": int(config["pred_len"]),
        "enc_in": int(config["enc_in"]),
        "dec_in": int(config["dec_in"]),
        "c_out": int(config["c_out"]),
        "resource_signature_sha256": _resource_signature(config),
        "source_root": str(source_root),
        "source": source_attestation,
        "deduplicated_source_configs": request.get("duplicate_sources", []),
        "architecture": architecture,
        "effective_mamba_architecture": effective_mamba,
        "mamba_channel_profile": channel_profile,
        "acf_channel_order_prior_source": acf_channel_order_prior_source,
        "parameters": {
            "total": parameter_count,
            "trainable": trainable_parameter_count,
            "parameter_mb": parameter_bytes / (1024.0 ** 2),
            "buffer_mb": buffer_bytes / (1024.0 ** 2),
        },
        "gpu_inference": benchmark,
    }


def _manifest_elapsed_summary(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        try:
            elapsed = float(record["elapsed_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("successful manifest record has no valid elapsed_sec: {}".format(record.get("job_id"))) from exc
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("manifest elapsed_sec is invalid: {}".format(record.get("job_id")))
        command = record.get("command", [])
        if not isinstance(command, list):
            raise ValueError("manifest command is invalid: {}".format(record.get("job_id")))
        model = str(record.get("model", ""))
        dataset = str(record.get("dataset", ""))
        features = str(record.get("features", ""))
        seq_len = _command_value(command, "--seq_len")
        pred_len = _command_value(command, "--pred_len")
        batch_size = _command_value(command, "--batch_size")
        train_epochs = _command_value(command, "--train_epochs")
        lradj = _command_value(command, "--lradj") or "type1"
        lambda_value = record.get("autocon_lambda")
        lambda_key = None if lambda_value is None else float(lambda_value)
        architecture_values = {
            "d_model": _command_value(command, "--d_model"),
            "d_ff": _command_value(command, "--d_ff"),
            "e_layers": _command_value(command, "--e_layers"),
            "mamba_d_state": _command_value(command, "--mamba_d_state"),
            "mamba_d_conv": _command_value(command, "--mamba_d_conv"),
            "mamba_expand": _command_value(command, "--mamba_expand"),
            "mamba_headdim": _command_value(command, "--mamba_headdim"),
            "learning_rate": _command_value(command, "--learning_rate"),
            "min_learning_rate": _command_value(command, "--min_learning_rate"),
            "drop_last_train": _command_flag(command, "--drop_last_train"),
            "use_amp": _command_flag(command, "--use_amp"),
            "mamba_bidirectional": _command_bool_or_default(command, "--mamba_bidirectional", False),
            "mamba_channel_independent": _command_bool_or_default(command, "--mamba_channel_independent", False),
            "mamba_channel_gate": _command_bool_or_default(command, "--mamba_channel_gate", False),
            "mamba_data_dependent_channel_mixer": _command_bool_or_default(
                command, "--mamba_data_dependent_channel_mixer", False
            ),
            "mamba_acf_scale_conditioning": _command_bool_or_default(command, "--mamba_acf_scale_conditioning", False),
            "mamba_acf_local_agreement": _command_bool_or_default(command, "--mamba_acf_local_agreement", False),
            "mamba_acf_local_min_pairs": _command_value(command, "--mamba_acf_local_min_pairs") or 16,
            "mamba_acf_channel_gate": _command_bool_or_default(command, "--mamba_acf_channel_gate", False),
            "mamba_causal_embedding": _command_bool_or_default(command, "--mamba_causal_embedding", False),
            "mamba_require_strict_representation_causality": _command_bool_or_default(
                command, "--mamba_require_strict_representation_causality", False
            ),
            "mamba_variable_mixer": _command_bool_or_default(command, "--mamba_variable_mixer", False),
            "mamba_variable_repr_fusion": _command_bool_or_default(
                command, "--mamba_variable_repr_fusion", False
            ),
            "mamba_variable_acf_memory_selector": _command_bool_or_default(
                command, "--mamba_variable_acf_memory_selector", False
            ),
            "mamba_variable_gdd_init_scale": _command_value(
                command, "--mamba_variable_gdd_init_scale"
            ) or 0.0,
            "mamba_acf_repr_fusion": _command_bool_or_default(
                command, "--mamba_acf_repr_fusion", False
            ),
            "mamba_variable_group_size": _command_value(command, "--mamba_variable_group_size") or 1,
            "mamba_variable_permutation": _command_bool_or_default(command, "--mamba_variable_permutation", False),
            "mamba_acf_channel_order": _command_value(command, "--mamba_acf_channel_order") or "input",
            "mamba_acf_channel_conditioning": _command_bool_or_default(command, "--mamba_acf_channel_conditioning", False),
            "mamba_patch_mixer": _command_bool_or_default(command, "--mamba_patch_mixer", False),
            "mamba_patch_permutation": _command_bool_or_default(command, "--mamba_patch_permutation", False),
            "mamba_low_rank_head": _command_bool_or_default(command, "--mamba_low_rank_head", False),
            "mamba_channel_gate_rank": _command_value(command, "--mamba_channel_gate_rank"),
            "mamba_data_dependent_channel_mixer_rank": _command_value(
                command, "--mamba_data_dependent_channel_mixer_rank"
            ),
            "mamba_variable_mixer_depth": _command_value(command, "--mamba_variable_mixer_depth"),
            "mamba_patch_mixer_depth": _command_value(command, "--mamba_patch_mixer_depth"),
            "mamba_patch_len": _command_value(command, "--mamba_patch_len"),
            "mamba_patch_stride": _command_value(command, "--mamba_patch_stride"),
            "mamba_ts_variant": _command_value(command, "--mamba_ts_variant"),
            "mamba_ts_context_rank": _command_value(command, "--mamba_ts_context_rank"),
            "mamba_patch_order": _command_value(command, "--mamba_patch_order"),
            "mamba_local_mixer": _command_bool_or_default(command, "--mamba_local_mixer", False),
            "mamba_local_patch_len": _command_value(command, "--mamba_local_patch_len"),
            "mamba_local_patch_stride": _command_value(command, "--mamba_local_patch_stride"),
            "mamba_local_width": _command_value(command, "--mamba_local_width"),
            "mamba_local_rank": _command_value(command, "--mamba_local_rank"),
            "mamba_local_depth": _command_value(command, "--mamba_local_depth"),
            "mamba_temporal_rank": _command_value(command, "--mamba_temporal_rank"),
            "mamba_scales": tuple(_command_values(command, "--mamba_scales")),
            "mamba_token_stride": _command_value(command, "--mamba_token_stride") or 1,
            "mamba_acf_lags": tuple(_command_values(command, "--mamba_acf_lags")),
            "mamba_acf_lag_mode": _command_value(command, "--mamba_acf_lag_mode"),
            "mamba_acf_channel_lag_bank": _command_bool_or_default(
                command, "--mamba_acf_channel_lag_bank", False
            ),
            "mamba_acf_channel_lag_top_k": _command_value(
                command, "--mamba_acf_channel_lag_top_k"
            ) or 2,
            "mamba_acf_max_lag": _command_value(command, "--mamba_acf_max_lag"),
            "mamba_acf_top_k": _command_value(command, "--mamba_acf_top_k"),
            "mamba_acf_min_lag": _command_value(command, "--mamba_acf_min_lag"),
            "mamba_acf_min_abs": _command_value(command, "--mamba_acf_min_abs"),
            "mamba_acf_confidence_z": _command_value(command, "--mamba_acf_confidence_z"),
            "mamba_acf_min_separation": _command_value(command, "--mamba_acf_min_separation"),
            "mamba_acf_detrend": _command_value(command, "--mamba_acf_detrend"),
            "mamba_acf_chunk": _command_value(command, "--mamba_acf_chunk"),
            "mamba_use_acf_memory": _command_bool_or_default(command, "--mamba_use_acf_memory", False),
            "mamba_dynamic_acf": _command_bool_or_default(command, "--mamba_dynamic_acf", False),
            "mamba_acf_temperature": _command_value(command, "--mamba_acf_temperature"),
            "mamba_acf_step_reliability": _command_bool_or_default(command, "--mamba_acf_step_reliability", False),
            "mamba_acf_reliability_interpolation": _command_bool_or_default(command, "--mamba_acf_reliability_interpolation", False),
            "mamba_acf_memory_gate_mode": _command_value(command, "--mamba_acf_memory_gate_mode") or "hidden",
            "mamba_acf_anchor_copy": _command_bool_or_default(command, "--mamba_acf_anchor_copy", False),
            "mamba_acf_horizon_calibration": _command_bool_or_default(command, "--mamba_acf_horizon_calibration", False),
            "mamba_acf_horizon_calibration_rank": _command_value(command, "--mamba_acf_horizon_calibration_rank") or 2,
            "mamba_acf_relation_loss": _command_bool_or_default(command, "--mamba_acf_relation_loss", False),
            "mamba_acf_relation_batching": _command_value(command, "--mamba_acf_relation_batching"),
            "mamba_acf_relation_channel_reduce": _command_value(command, "--mamba_acf_relation_channel_reduce"),
            "mamba_acf_relation_scope": _command_value(command, "--mamba_acf_relation_scope") or "both",
            "mamba_acf_relation_representation": _command_value(
                command, "--mamba_acf_relation_representation"
            ) or "encoder",
            "mamba_acf_relation_lag_mask": _command_bool_or_default(command, "--mamba_acf_relation_lag_mask", False),
            "mamba_acf_relation_lag_tolerance": _command_value(command, "--mamba_acf_relation_lag_tolerance") or 0,
            "mamba_acf_relation_min_lag": _command_value(command, "--mamba_acf_relation_min_lag") or 0,
            "mamba_acf_relation_target_mode": _command_value(command, "--mamba_acf_relation_target_mode") or "hard_selected",
            "mamba_acf_relation_soft_min_abs": _command_value(command, "--mamba_acf_relation_soft_min_abs") or 0.1,
            "mamba_acf_relation_diagnostic_min_abs": _command_value(command, "--mamba_acf_relation_diagnostic_min_abs") or 0.1,
            "mamba_acf_relation_selected_lag_boost": _command_value(command, "--mamba_acf_relation_selected_lag_boost") or 0.0,
            "AutoCon_temperature": _command_value(command, "--AutoCon_temperature"),
            "AutoCon_relation_temperature": _command_value(command, "--AutoCon_relation_temperature"),
            "AutoCon_sign_weight": _command_value(command, "--AutoCon_sign_weight"),
            "AutoCon_sign_margin": _command_value(command, "--AutoCon_sign_margin"),
        }
        if not model or not dataset or not features or seq_len is None or pred_len is None:
            raise ValueError("manifest lacks resource coordinates: {}".format(record.get("job_id")))
        try:
            dimensions = tuple(int(record[key]) for key in ("enc_in", "dec_in", "c_out"))
            numeric_architecture = tuple(
                int(architecture_values[name]) if architecture_values[name] is not None else None
                for name in (
                    "d_model", "d_ff", "e_layers", "mamba_ts_context_rank", "mamba_d_state", "mamba_d_conv",
                    "mamba_expand", "mamba_headdim", "mamba_channel_gate_rank",
                    "mamba_data_dependent_channel_mixer_rank",
                    "mamba_acf_local_min_pairs",
                    "mamba_acf_channel_lag_top_k",
                    "mamba_variable_mixer_depth", "mamba_variable_group_size", "mamba_patch_mixer_depth", "mamba_patch_len",
                    "mamba_patch_stride", "mamba_local_patch_len", "mamba_local_patch_stride",
                    "mamba_local_width", "mamba_local_rank", "mamba_local_depth",
                    "mamba_temporal_rank", "mamba_token_stride", "mamba_acf_max_lag",
                    "mamba_acf_top_k", "mamba_acf_min_lag", "mamba_acf_min_separation",
                    "mamba_acf_chunk", "mamba_acf_relation_min_lag",
                    "mamba_acf_horizon_calibration_rank",
                )
            )
            key = (
                model, dataset, features, int(seq_len), int(pred_len), *dimensions,
                lambda_key, batch_size, train_epochs, lradj, *numeric_architecture,
                float(architecture_values["mamba_variable_gdd_init_scale"]),
                architecture_values["mamba_bidirectional"],
                architecture_values["mamba_channel_independent"],
                architecture_values["mamba_channel_gate"],
                architecture_values["mamba_data_dependent_channel_mixer"],
                architecture_values["mamba_acf_scale_conditioning"],
                architecture_values["mamba_acf_local_agreement"],
                architecture_values["mamba_acf_channel_gate"],
                architecture_values["mamba_causal_embedding"],
                architecture_values["mamba_require_strict_representation_causality"],
                architecture_values["mamba_variable_mixer"],
                architecture_values["mamba_variable_repr_fusion"],
                architecture_values["mamba_acf_repr_fusion"],
                architecture_values["mamba_variable_group_size"],
                architecture_values["mamba_variable_permutation"],
                architecture_values["mamba_acf_channel_order"],
                architecture_values["mamba_acf_channel_conditioning"],
                architecture_values["mamba_patch_mixer"],
                architecture_values["mamba_patch_permutation"],
                architecture_values["mamba_patch_order"],
                architecture_values["mamba_local_mixer"],
                architecture_values["mamba_low_rank_head"],
                architecture_values["mamba_scales"],
                architecture_values["mamba_acf_lags"],
                architecture_values["mamba_acf_lag_mode"],
                architecture_values["mamba_acf_channel_lag_bank"],
                architecture_values["mamba_use_acf_memory"],
                architecture_values["mamba_dynamic_acf"],
                architecture_values["mamba_acf_min_abs"],
                architecture_values["mamba_acf_confidence_z"],
                architecture_values["mamba_acf_detrend"],
                architecture_values["mamba_acf_temperature"],
                architecture_values["mamba_acf_step_reliability"],
                architecture_values["mamba_acf_reliability_interpolation"],
                architecture_values["mamba_acf_memory_gate_mode"],
                architecture_values["mamba_acf_anchor_copy"],
                architecture_values["mamba_acf_horizon_calibration"],
                architecture_values["mamba_acf_horizon_calibration_rank"],
                architecture_values["mamba_acf_relation_loss"],
                architecture_values["mamba_acf_relation_batching"],
                architecture_values["mamba_acf_relation_channel_reduce"],
                architecture_values["mamba_acf_relation_scope"],
                architecture_values["mamba_acf_relation_representation"],
                architecture_values["mamba_acf_relation_lag_mask"],
                architecture_values["mamba_acf_relation_lag_tolerance"],
                architecture_values["mamba_acf_relation_min_lag"],
                architecture_values["mamba_acf_relation_target_mode"],
                architecture_values["mamba_acf_relation_soft_min_abs"],
                architecture_values["mamba_acf_relation_diagnostic_min_abs"],
                architecture_values["mamba_acf_relation_selected_lag_boost"],
                architecture_values["AutoCon_temperature"],
                architecture_values["AutoCon_relation_temperature"],
                architecture_values["AutoCon_sign_weight"],
                architecture_values["AutoCon_sign_margin"],
                architecture_values["learning_rate"], architecture_values["min_learning_rate"],
                architecture_values["drop_last_train"], architecture_values["use_amp"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("manifest has invalid resource dimensions: {}".format(record.get("job_id"))) from exc
        row = grouped.setdefault(key, {
            "model": model,
            "dataset": dataset,
            "features": features,
            "seq_len": int(seq_len),
            "pred_len": int(pred_len),
            "enc_in": dimensions[0],
            "dec_in": dimensions[1],
            "c_out": dimensions[2],
            "autocon_lambda": lambda_key,
            "train_batch_size": int(batch_size) if batch_size is not None else None,
            "configured_train_epochs": int(train_epochs) if train_epochs is not None else None,
            "learning_rate_schedule": lradj,
            **{
                name: (
                    int(architecture_values[name])
                    if name in {
                        "d_model", "d_ff", "e_layers", "mamba_d_state", "mamba_d_conv",
                        "mamba_expand", "mamba_headdim", "mamba_channel_gate_rank",
                        "mamba_data_dependent_channel_mixer_rank",
                        "mamba_acf_local_min_pairs",
                        "mamba_acf_channel_lag_top_k",
                        "mamba_variable_mixer_depth", "mamba_variable_group_size", "mamba_patch_mixer_depth", "mamba_patch_len",
                        "mamba_patch_stride", "mamba_local_patch_len", "mamba_local_patch_stride",
                        "mamba_local_width", "mamba_local_rank", "mamba_local_depth",
                        "mamba_temporal_rank", "mamba_token_stride", "mamba_acf_max_lag",
                        "mamba_acf_top_k", "mamba_acf_min_lag", "mamba_acf_min_separation",
                        "mamba_acf_chunk", "mamba_acf_relation_min_lag",
                        "mamba_acf_horizon_calibration_rank",
                    }
                    and architecture_values[name] is not None
                    else architecture_values[name]
                )
                for name in (
                    "d_model", "d_ff", "e_layers", "mamba_ts_context_rank", "mamba_d_state", "mamba_d_conv",
                    "mamba_expand", "mamba_headdim", "mamba_channel_gate_rank",
                    "mamba_data_dependent_channel_mixer_rank",
                    "mamba_acf_channel_lag_top_k",
                    "mamba_variable_mixer_depth", "mamba_variable_group_size", "mamba_patch_mixer_depth", "mamba_patch_len",
                    "mamba_patch_stride", "mamba_local_patch_len", "mamba_local_patch_stride",
                    "mamba_local_width", "mamba_local_rank", "mamba_local_depth",
                    "mamba_temporal_rank", "mamba_token_stride", "mamba_acf_max_lag",
                    "mamba_acf_top_k", "mamba_acf_min_lag", "mamba_acf_min_separation",
                    "mamba_acf_chunk", "learning_rate", "min_learning_rate", "drop_last_train", "use_amp",
                )
            },
            "mamba_bidirectional": architecture_values["mamba_bidirectional"],
            "mamba_channel_independent": architecture_values["mamba_channel_independent"],
                    "mamba_channel_gate": architecture_values["mamba_channel_gate"],
                    "mamba_data_dependent_channel_mixer": architecture_values[
                        "mamba_data_dependent_channel_mixer"
                    ],
                    "mamba_acf_scale_conditioning": architecture_values["mamba_acf_scale_conditioning"],
                    "mamba_acf_local_agreement": architecture_values["mamba_acf_local_agreement"],
                    "mamba_acf_local_min_pairs": int(architecture_values["mamba_acf_local_min_pairs"]),
            "mamba_acf_channel_gate": architecture_values["mamba_acf_channel_gate"],
            "mamba_acf_channel_lag_bank": architecture_values[
                "mamba_acf_channel_lag_bank"
            ],
            "mamba_acf_channel_lag_top_k": int(
                architecture_values["mamba_acf_channel_lag_top_k"]
            ),
            "mamba_causal_embedding": architecture_values["mamba_causal_embedding"],
            "mamba_require_strict_representation_causality": architecture_values[
                "mamba_require_strict_representation_causality"
            ],
            "mamba_variable_mixer": architecture_values["mamba_variable_mixer"],
            "mamba_variable_repr_fusion": architecture_values[
                "mamba_variable_repr_fusion"
            ],
            "mamba_variable_gdd_init_scale": float(
                architecture_values["mamba_variable_gdd_init_scale"]
            ),
            "mamba_acf_repr_fusion": architecture_values[
                "mamba_acf_repr_fusion"
            ],
            "mamba_variable_group_size": int(architecture_values["mamba_variable_group_size"]),
            "mamba_variable_permutation": architecture_values["mamba_variable_permutation"],
            "mamba_acf_channel_order": architecture_values["mamba_acf_channel_order"],
            "mamba_acf_channel_conditioning": architecture_values["mamba_acf_channel_conditioning"],
            "mamba_patch_mixer": architecture_values["mamba_patch_mixer"],
            "mamba_patch_permutation": architecture_values["mamba_patch_permutation"],
            "mamba_patch_order": architecture_values["mamba_patch_order"],
            "mamba_local_mixer": architecture_values["mamba_local_mixer"],
            "mamba_low_rank_head": architecture_values["mamba_low_rank_head"],
            "mamba_scales": architecture_values["mamba_scales"],
            "mamba_token_stride": int(architecture_values["mamba_token_stride"]),
            "mamba_acf_lags": architecture_values["mamba_acf_lags"],
            "mamba_acf_lag_mode": architecture_values["mamba_acf_lag_mode"],
            "mamba_use_acf_memory": architecture_values["mamba_use_acf_memory"],
            "mamba_dynamic_acf": architecture_values["mamba_dynamic_acf"],
            "mamba_acf_min_abs": architecture_values["mamba_acf_min_abs"],
            "mamba_acf_confidence_z": architecture_values["mamba_acf_confidence_z"],
            "mamba_acf_detrend": architecture_values["mamba_acf_detrend"],
            "mamba_acf_temperature": architecture_values["mamba_acf_temperature"],
            "mamba_acf_step_reliability": architecture_values["mamba_acf_step_reliability"],
            "mamba_acf_reliability_interpolation": architecture_values["mamba_acf_reliability_interpolation"],
            "mamba_acf_memory_gate_mode": architecture_values["mamba_acf_memory_gate_mode"],
            "mamba_acf_anchor_copy": architecture_values["mamba_acf_anchor_copy"],
            "mamba_acf_horizon_calibration": architecture_values["mamba_acf_horizon_calibration"],
            "mamba_acf_horizon_calibration_rank": int(architecture_values["mamba_acf_horizon_calibration_rank"]),
            "mamba_acf_relation_loss": architecture_values["mamba_acf_relation_loss"],
            "mamba_acf_relation_batching": architecture_values["mamba_acf_relation_batching"],
            "mamba_acf_relation_channel_reduce": architecture_values["mamba_acf_relation_channel_reduce"],
            "mamba_acf_relation_scope": architecture_values["mamba_acf_relation_scope"],
            "mamba_acf_relation_representation": architecture_values[
                "mamba_acf_relation_representation"
            ],
            "mamba_acf_relation_lag_mask": architecture_values["mamba_acf_relation_lag_mask"],
            "mamba_acf_relation_lag_tolerance": int(architecture_values["mamba_acf_relation_lag_tolerance"]),
            "mamba_acf_relation_min_lag": int(architecture_values["mamba_acf_relation_min_lag"]),
            "mamba_acf_relation_target_mode": architecture_values["mamba_acf_relation_target_mode"],
            "mamba_acf_relation_soft_min_abs": float(architecture_values["mamba_acf_relation_soft_min_abs"]),
            "mamba_acf_relation_diagnostic_min_abs": float(architecture_values["mamba_acf_relation_diagnostic_min_abs"]),
            "mamba_acf_relation_selected_lag_boost": float(architecture_values["mamba_acf_relation_selected_lag_boost"]),
            "AutoCon_temperature": architecture_values["AutoCon_temperature"],
            "AutoCon_relation_temperature": architecture_values["AutoCon_relation_temperature"],
            "AutoCon_sign_weight": architecture_values["AutoCon_sign_weight"],
            "AutoCon_sign_margin": architecture_values["AutoCon_sign_margin"],
            "seed_ids": [],
            "elapsed_sec_values": [],
        })
        row["elapsed_sec_values"].append(elapsed)
        if record.get("seed") is not None:
            row["seed_ids"].append(int(record["seed"]))

    summaries: List[Dict[str, Any]] = []
    for row in grouped.values():
        values = row.pop("elapsed_sec_values")
        seed_ids = sorted(set(row.pop("seed_ids")))
        summaries.append({
            **row,
            "job_count": len(values),
            "unique_seed_count": len(seed_ids),
            "seed_ids": seed_ids,
            "elapsed_total_sec": float(sum(values)),
            "elapsed_mean_sec_per_job": float(statistics.fmean(values)),
            "elapsed_median_sec_per_job": float(statistics.median(values)),
            "elapsed_min_sec_per_job": float(min(values)),
            "elapsed_max_sec_per_job": float(max(values)),
            "unit_note": "one manifest job/seed; not sliding windows or inferential replicates",
        })
    return sorted(
        summaries,
        key=lambda item: (item["dataset"], item["features"], item["pred_len"], item["model"]),
    )


def _flatten_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for record in records:
        gpu = record["gpu_inference"]
        parameters = record["parameters"]
        source = record["source"]
        flattened.append({
            "model": record["model"],
            "dataset": record["dataset"],
            "features": record["features"],
            "seq_len": record["seq_len"],
            "label_len": record["label_len"],
            "pred_len": record["pred_len"],
            "enc_in": record["enc_in"],
            "dec_in": record["dec_in"],
            "c_out": record["c_out"],
            "resource_signature_sha256": record["resource_signature_sha256"],
            "acf_channel_order_prior_source": record.get("acf_channel_order_prior_source"),
            "parameter_total": parameters["total"],
            "parameter_trainable": parameters["trainable"],
            "parameter_mb": parameters["parameter_mb"],
            "buffer_mb": parameters["buffer_mb"],
            "effective_mamba_d_state": record["effective_mamba_architecture"].get("mamba_d_state"),
            "effective_mamba_d_conv": record["effective_mamba_architecture"].get("mamba_d_conv"),
            "effective_mamba_expand": record["effective_mamba_architecture"].get("mamba_expand"),
            "effective_mamba_headdim": record["effective_mamba_architecture"].get("mamba_headdim"),
            "effective_mamba_acf_channel_order": record["effective_mamba_architecture"].get("acf_channel_order"),
            **{
                "mamba_channel_" + name: value
                for name, value in (record.get("mamba_channel_profile") or {}).items()
            },
            # Keep the CSV self-describing for architecture ablations; the
            # complete canonical payload remains available under JSON
            # ``architecture``.
            **{
                "architecture_" + name: record["architecture"].get(name)
                for name in (
                    "mamba_scales", "mamba_token_stride", "mamba_bidirectional", "mamba_channel_independent",
                    "mamba_channel_gate", "mamba_channel_gate_rank",
                    "mamba_data_dependent_channel_mixer",
                    "mamba_data_dependent_channel_mixer_rank",
                    "mamba_causal_embedding",
                    "mamba_require_strict_representation_causality",
                    "mamba_acf_scale_conditioning",
                    "mamba_acf_local_agreement", "mamba_acf_local_min_pairs",
                    "mamba_acf_channel_gate",
                    "mamba_acf_channel_lag_bank",
                    "mamba_variable_mixer", "mamba_variable_mixer_depth", "mamba_variable_group_size",
                    "mamba_variable_gdd_mixer", "mamba_variable_gdd_mixer_rank",
                    "mamba_variable_gdd_init_scale",
                    "mamba_variable_repr_fusion", "mamba_acf_repr_fusion",
                    "mamba_variable_permutation", "mamba_acf_channel_order",
                    "mamba_acf_channel_conditioning",
                    "mamba_patch_mixer",
                    "mamba_patch_mixer_depth", "mamba_patch_len", "mamba_patch_stride",
                    "mamba_patch_order", "mamba_patch_permutation",
                    "mamba_local_mixer", "mamba_local_patch_len",
                    "mamba_local_patch_stride", "mamba_local_width",
                    "mamba_local_rank", "mamba_local_depth", "mamba_low_rank_head",
                    "mamba_temporal_rank", "mamba_acf_lags", "mamba_acf_lag_mode",
                    "mamba_acf_channel_lag_top_k",
                    "mamba_acf_max_lag", "mamba_acf_top_k", "mamba_acf_min_lag",
                    "mamba_acf_min_abs", "mamba_acf_confidence_z", "mamba_acf_min_separation",
                    "mamba_acf_detrend", "mamba_acf_chunk", "mamba_use_acf_memory",
                    "mamba_dynamic_acf", "mamba_acf_temperature", "mamba_acf_step_reliability",
                    "mamba_acf_reliability_interpolation",
                    "mamba_acf_memory_gate_mode",
                    "mamba_acf_anchor_copy",
                    "mamba_acf_horizon_calibration", "mamba_acf_horizon_calibration_rank",
                    "mamba_acf_relation_loss",
                    "mamba_acf_relation_batching", "mamba_acf_relation_channel_reduce",
                    "mamba_acf_relation_scope", "mamba_acf_relation_representation",
                    "mamba_acf_relation_lag_mask",
                    "mamba_acf_relation_lag_tolerance", "mamba_acf_relation_min_lag",
                    "mamba_acf_relation_target_mode", "mamba_acf_relation_soft_min_abs",
                    "mamba_acf_relation_diagnostic_min_abs",
                    "mamba_acf_relation_selected_lag_boost",
                )
            },
            "benchmark_status": gpu["status"],
            "benchmark_batch_size": gpu.get("benchmark_batch_size"),
            "benchmark_forward_path": gpu.get("forward_path"),
            "benchmark_auxiliary_representation_requested": gpu.get(
                "auxiliary_representation_requested"
            ),
            "benchmark_auxiliary_representation_controlled": gpu.get(
                "auxiliary_representation_controlled"
            ),
            "benchmark_auxiliary_representation_returned": gpu.get(
                "auxiliary_representation_returned"
            ),
            "benchmark_auxiliary_representation_shape": gpu.get(
                "auxiliary_representation_shape"
            ),
            "benchmark_relation_loss_included": gpu.get("relation_loss_included"),
            "latency_ms_mean": gpu.get("latency_ms_mean"),
            "latency_ms_median": gpu.get("latency_ms_median"),
            "latency_ms_p95": gpu.get("latency_ms_p95"),
            "synthetic_input_sequences_per_sec": gpu.get("synthetic_input_sequences_per_sec"),
            "steady_state_peak_allocated_mb": gpu.get("steady_state_peak_allocated_mb"),
            "steady_state_peak_reserved_mb": gpu.get("steady_state_peak_reserved_mb"),
            "steady_state_peak_increment_mb": gpu.get("steady_state_peak_increment_mb"),
            "source_kind": source["kind"],
            "config_path": source.get("config_path"),
            "config_sha256": source.get("config_sha256"),
            "phase": source.get("phase"),
            "provenance_id": source.get("provenance_id"),
            "job_id": source.get("job_id"),
        })
    return flattened


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _write_outputs(output_dir: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_resources.json"
    resource_csv = output_dir / "model_resources.csv"
    elapsed_csv = output_dir / "training_elapsed_summary.csv"
    targets = (json_path, resource_csv, elapsed_csv)
    if not overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite existing analysis files: {}".format(", ".join(existing)))
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_csv(resource_csv, _flatten_records(payload["resource_records"]))
    _write_csv(elapsed_csv, payload["training_elapsed_summary"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("source")
    source.add_argument("--config", action="append", type=Path,
                        help="result config.json; repeat for each architecture to benchmark")
    source.add_argument("--all-validated", action="store_true",
                        help="measure every config in --phase-validation (resource signatures are deduplicated)")
    source.add_argument("--phase-validation", type=Path,
                        help="validated phase_validation.json required to attest result configs")
    source.add_argument("--manifest", type=Path,
                        help="manifest for elapsed-time summary; inferred from --phase-validation when omitted")
    source.add_argument("--source-root", type=Path,
                        help="model source root for unverified/explicit input; defaults to repository root")

    explicit = parser.add_argument_group("explicit CLI architecture (alternative to --config)")
    explicit.add_argument("--model", choices=sorted(MODEL_MODULES))
    explicit.add_argument("--dataset", default="explicit")
    explicit.add_argument("--features", choices=("S", "M", "MS"), default="S")
    explicit.add_argument("--freq", default="h")
    explicit.add_argument("--embed", default="timeF")
    explicit.add_argument("--activation", default="gelu")
    explicit.add_argument("--wo-freq", default="")
    explicit.add_argument("--seq-len", type=int)
    explicit.add_argument("--label-len", type=int)
    explicit.add_argument("--pred-len", type=int)
    explicit.add_argument("--enc-in", type=int)
    explicit.add_argument("--dec-in", type=int)
    explicit.add_argument("--c-out", type=int)
    explicit.add_argument("--d-model", type=int, default=32)
    explicit.add_argument("--n-heads", type=int, default=8)
    explicit.add_argument("--e-layers", type=int, default=1)
    explicit.add_argument("--d-layers", type=int, default=1)
    explicit.add_argument("--d-ff", type=int, default=32)
    explicit.add_argument("--patch-len", type=int, default=16)
    explicit.add_argument("--patch-stride", type=int, default=8)
    explicit.add_argument("--timemixer-down-sampling-window",
                          dest="timemixer_down_sampling_window", type=int, default=2)
    explicit.add_argument("--timemixer-down-sampling-layers",
                          dest="timemixer_down_sampling_layers", type=int, default=1)
    explicit.add_argument("--timemixer-channel-independence",
                          dest="timemixer_channel_independence", action="store_true", default=False)
    explicit.add_argument("--timemixer-decomp-method",
                          dest="timemixer_decomp_method",
                          choices=("moving_avg", "dft_decomp"), default="moving_avg")
    explicit.add_argument("--mamba-ts-variant", dest="mamba_ts_variant",
                          choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"), default="v1",
                          help="MambaTS architecture variant")
    explicit.add_argument("--mamba-ts-context-rank", dest="mamba_ts_context_rank",
                          type=int, default=16,
                          help="low-rank cross-variable context width for MambaTS v3")
    explicit.add_argument("--mamba-ts-mixer-type", dest="mamba_ts_mixer_type",
                          choices=("mamba1", "mamba2"), default="mamba1",
                          help="MambaTS scan mixer for explicit resource candidates")
    explicit.add_argument("--mamba-ts-delta-multipliers", dest="mamba_ts_delta_multipliers",
                          type=float, nargs="+", default=[1.0, 2.0])
    explicit.add_argument("--mamba-ts-dt-min", dest="mamba_ts_dt_min", type=float, default=0.001)
    explicit.add_argument("--mamba-ts-dt-max", dest="mamba_ts_dt_max", type=float, default=0.1)
    explicit.add_argument("--mamba-ts-spectral-top-k", dest="mamba_ts_spectral_top_k",
                          type=int, default=4,
                          help="number of dominant causal Fourier bins for MambaTS v8")
    explicit.add_argument("--mamba-ts-repr-forecast-fusion",
                          dest="mamba_ts_repr_forecast_fusion", action="store_true",
                          default=False)
    explicit.add_argument("--mamba-ts-repr-forecast-channelwise",
                          dest="mamba_ts_repr_forecast_channelwise", action="store_true",
                          default=False)
    explicit.add_argument("--mamba-ts-channel-adapter",
                          dest="mamba_ts_channel_adapter", action="store_true",
                          default=False)
    explicit.add_argument("--mamba-ts-channel-adapter-rank",
                          dest="mamba_ts_channel_adapter_rank", type=int, default=8)
    explicit.add_argument("--mamba-ts-revin-affine",
                          dest="mamba_ts_revin_affine", action="store_true",
                          default=False,
                          help="enable the reversible per-channel affine adapter")
    explicit.add_argument("--mamba-ts-global-channel-mixer",
                          dest="mamba_ts_global_channel_mixer", action="store_true",
                          default=False,
                          help="enable the low-rank global channel-state mixer")
    explicit.add_argument("--mamba-ts-global-channel-mixer-rank",
                          dest="mamba_ts_global_channel_mixer_rank", type=int, default=4)
    explicit.add_argument("--mamba-ts-graph-spatial",
                          dest="mamba_ts_graph_spatial", action="store_true",
                          default=False,
                          help="enable sparse graph propagation before MambaTS v3")
    explicit.add_argument("--mamba-ts-graph-path",
                          dest="mamba_ts_graph_path", type=str, default="")
    explicit.add_argument("--mamba-ts-graph-expected-sha256",
                          dest="mamba_ts_graph_expected_sha256", type=str, default="")
    explicit.add_argument("--mamba-ts-repr-linear-expert",
                          dest="mamba_ts_repr_linear_expert", action="store_true",
                          default=False)
    explicit.add_argument("--mamba-ts-repr-linear-expert-mode",
                          dest="mamba_ts_repr_linear_expert_mode",
                          choices=("residual", "blend", "nlinear_blend"), default="residual")
    explicit.add_argument("--mamba-ts-repr-linear-blend-bias",
                          dest="mamba_ts_repr_linear_blend_bias",
                          type=float, default=-4.0)
    explicit.add_argument("--mamba-ts-repr-linear-expert-rank",
                          dest="mamba_ts_repr_linear_expert_rank",
                          type=int, default=16)
    explicit.add_argument("--mamba-ts-variable-permutation",
                          dest="mamba_ts_variable_permutation", action="store_true",
                          default=False,
                          help="apply train-time channel permutation in MambaTS v3/v4/v5")
    explicit.add_argument("--mamba-ts-acf-memory-selector",
                          dest="mamba_ts_acf_memory_selector", action="store_true",
                          default=False)
    explicit.add_argument("--mamba-ts-acf-memory-channel-selector",
                          dest="mamba_ts_acf_memory_channel_selector", action="store_true",
                          default=False)
    explicit.add_argument("--mamba-ts-forecast-acf-consistency",
                          dest="mamba_ts_forecast_acf_consistency", action="store_true",
                          default=False)
    explicit.add_argument("--mamba-ts-forecast-acf-weight",
                          dest="mamba_ts_forecast_acf_weight", type=float, default=0.0)
    explicit.add_argument("--mamba-ts-forecast-acf-top-k",
                          dest="mamba_ts_forecast_acf_top_k", type=int, default=4)
    explicit.add_argument("--mamba-ts-forecast-acf-min-lag",
                          dest="mamba_ts_forecast_acf_min_lag", type=int, default=2)
    explicit.add_argument("--mamba-ts-forecast-acf-min-abs",
                          dest="mamba_ts_forecast_acf_min_abs", type=float, default=0.10)
    explicit.add_argument("--mamba-d-state", type=int, default=16)
    explicit.add_argument("--mamba-d-conv", type=int, default=4)
    explicit.add_argument("--mamba-expand", type=int, default=2)
    explicit.add_argument("--mamba-headdim", type=int, default=16)
    explicit.add_argument("--mamba-tm-n1", dest="mamba_tm_n1", type=int, default=64,
                          help="TimeMachine first latent width")
    explicit.add_argument("--mamba-tm-n2", dest="mamba_tm_n2", type=int, default=128,
                          help="TimeMachine second latent width")
    explicit.add_argument("--mamba-tm-repr-width", dest="mamba_tm_repr_width", type=int,
                          default=64, help="TimeMachine AutoCon representation width")
    explicit.add_argument("--mamba-tm-channel-independent",
                          dest="mamba_tm_channel_independent", action="store_true",
                          default=False,
                          help="use the channel-independent TimeMachine branch")
    explicit.add_argument("--no-mamba-tm-channel-independent",
                          dest="mamba_tm_channel_independent", action="store_false")
    explicit.add_argument("--mamba-tm-residual", dest="mamba_tm_residual",
                          type=_str2bool, default=True,
                          help="enable TimeMachine residual branch fusion")
    explicit.add_argument("--mamba-bipatch-token-mode",
                          dest="mamba_bipatch_token_mode",
                          choices=("mixing", "independent"), default="mixing",
                          help="BiPatchMamba token axis")
    explicit.add_argument("--mamba-bipatch-repr-gate",
                          dest="mamba_bipatch_repr_gate", action="store_true",
                          default=False,
                          help="use bounded AutoCon representation forecast gate")
    explicit.add_argument("--mamba-scales", type=int, nargs="+", default=[1, 2])
    explicit.add_argument("--mamba-token-stride", type=int, default=1,
                          help="causal base patch stride before MambaACF temporal scans")
    explicit.add_argument("--mamba-bidirectional", dest="mamba_bidirectional",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-bidirectional", dest="mamba_bidirectional",
                          action="store_false")
    explicit.add_argument("--mamba-channel-independent", dest="mamba_channel_independent",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-channel-independent", dest="mamba_channel_independent",
                          action="store_false")
    explicit.add_argument("--mamba-channel-gate", dest="mamba_channel_gate",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-channel-gate", dest="mamba_channel_gate",
                          action="store_false")
    explicit.add_argument("--mamba-channel-gate-rank", type=int, default=8)
    explicit.add_argument("--mamba-data-dependent-channel-mixer",
                          dest="mamba_data_dependent_channel_mixer",
                          action="store_true", default=False,
                          help="enable low-rank window-conditioned cross-channel state modulation")
    explicit.add_argument("--no-mamba-data-dependent-channel-mixer",
                          dest="mamba_data_dependent_channel_mixer",
                          action="store_false")
    explicit.add_argument("--mamba-data-dependent-channel-mixer-rank",
                          dest="mamba_data_dependent_channel_mixer_rank",
                          type=int, default=8)
    explicit.add_argument("--mamba-acf-scale-conditioning", dest="mamba_acf_scale_conditioning",
                          action="store_true", default=False,
                          help="condition multi-scale fusion with train-prior-weighted current-window ACF")
    explicit.add_argument("--no-mamba-acf-scale-conditioning", dest="mamba_acf_scale_conditioning",
                          action="store_false")
    explicit.add_argument("--mamba-acf-local-agreement", dest="mamba_acf_local_agreement",
                          action="store_true", default=False,
                          help="reweight static forecast-memory lags with causal local ACF agreement")
    explicit.add_argument("--no-mamba-acf-local-agreement", dest="mamba_acf_local_agreement",
                          action="store_false")
    explicit.add_argument("--mamba-acf-local-min-pairs", dest="mamba_acf_local_min_pairs",
                          type=int, default=16,
                          help="minimum observed pairs for local ACF agreement")
    explicit.add_argument("--mamba-acf-channel-gate", dest="mamba_acf_channel_gate",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-acf-channel-gate", dest="mamba_acf_channel_gate",
                          action="store_false")
    explicit.add_argument("--mamba-causal-embedding", dest="mamba_causal_embedding",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-causal-embedding", dest="mamba_causal_embedding",
                          action="store_false")
    explicit.add_argument("--mamba-require-strict-representation-causality",
                          "--mamba_require_strict_representation_causality",
                          dest="mamba_require_strict_representation_causality",
                          action="store_true", default=False,
                          help="require a strictly causal full MambaACF AutoCon representation grid")
    explicit.add_argument("--mamba-variable-mixer", dest="mamba_variable_mixer",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-variable-mixer", dest="mamba_variable_mixer",
                          action="store_false")
    explicit.add_argument("--mamba-variable-repr-fusion",
                          dest="mamba_variable_repr_fusion",
                          action="store_true", default=False,
                          help="fuse variable-token states into the AutoCon representation")
    explicit.add_argument("--no-mamba-variable-repr-fusion",
                          dest="mamba_variable_repr_fusion",
                          action="store_false")
    explicit.add_argument("--mamba-variable-acf-memory-selector",
                          "--mamba_variable_acf_memory_selector",
                          dest="mamba_variable_acf_memory_selector",
                          action="store_true", default=False,
                          help="gate NLinearVariableMamba ACF memory with its AutoCon representation")
    explicit.add_argument("--mamba-variable-mixer-depth", type=int, default=1)
    explicit.add_argument("--mamba-variable-gdd-mixer", dest="mamba_variable_gdd_mixer",
                          action="store_true", default=False,
                          help="enable low-rank channel-preserving variable-token modulation")
    explicit.add_argument("--no-mamba-variable-gdd-mixer", dest="mamba_variable_gdd_mixer",
                          action="store_false")
    explicit.add_argument("--mamba-variable-gdd-mixer-rank",
                          dest="mamba_variable_gdd_mixer_rank", type=int, default=8)
    explicit.add_argument("--mamba-variable-gdd-init-scale",
                          dest="mamba_variable_gdd_init_scale", type=float, default=0.0,
                          help="initial raw residual scale for variable-token GDD (0..1)")
    explicit.add_argument("--mamba-variable-group-size", type=int, default=1,
                          help="contiguous channels per grouped variable-token Mamba scan")
    explicit.add_argument("--mamba-variable-permutation", dest="mamba_variable_permutation",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-variable-permutation", dest="mamba_variable_permutation",
                          action="store_false")
    explicit.add_argument("--mamba-acf-channel-order", choices=("input", "strength", "similarity"),
                          default="input",
                          help="variable-token order: CSV input, train-only ACF strength, or signed profile similarity")
    explicit.add_argument("--mamba-acf-channel-conditioning", dest="mamba_acf_channel_conditioning",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-acf-channel-conditioning", dest="mamba_acf_channel_conditioning",
                          action="store_false")
    explicit.add_argument("--mamba-acf-repr-fusion", "--mamba_acf_repr_fusion",
                          dest="mamba_acf_repr_fusion", action="store_true",
                          default=False,
                          help="fuse train-only ACF lag context into the AutoCon representation")
    explicit.add_argument("--no-mamba-acf-repr-fusion", "--no_mamba_acf_repr_fusion",
                          dest="mamba_acf_repr_fusion", action="store_false")
    explicit.add_argument("--mamba-patch-mixer", dest="mamba_patch_mixer",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-patch-mixer", dest="mamba_patch_mixer",
                          action="store_false")
    explicit.add_argument("--mamba-patch-mixer-depth", type=int, default=1)
    explicit.add_argument("--mamba-patch-len", type=int, default=16)
    explicit.add_argument("--mamba-patch-stride", type=int, default=8)
    explicit.add_argument("--mamba-patch-order", choices=("time_major", "channel_major"),
                          default="time_major")
    explicit.add_argument("--mamba-patch-permutation", dest="mamba_patch_permutation",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-patch-permutation", dest="mamba_patch_permutation",
                          action="store_false")
    explicit.add_argument("--mamba-local-mixer", dest="mamba_local_mixer",
                          action="store_true", default=False,
                          help="enable the channel-preserving local patch adapter")
    explicit.add_argument("--no-mamba-local-mixer", dest="mamba_local_mixer",
                          action="store_false")
    explicit.add_argument("--mamba-local-patch-len", type=int, default=16)
    explicit.add_argument("--mamba-local-patch-stride", type=int, default=8)
    explicit.add_argument("--mamba-local-width", type=int, default=None)
    explicit.add_argument("--mamba-local-rank", type=int, default=16)
    explicit.add_argument("--mamba-local-depth", type=int, default=1)
    explicit.add_argument("--mamba-low-rank-head", dest="mamba_low_rank_head",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-low-rank-head", dest="mamba_low_rank_head",
                          action="store_false")
    explicit.add_argument("--mamba-temporal-rank", type=int, default=16)
    explicit.add_argument("--mamba-acf-lags", type=int, nargs="+", default=[24, 168, 336])
    explicit.add_argument("--mamba-acf-channel-lag-bank",
                          "--mamba_acf_channel_lag_bank",
                          dest="mamba_acf_channel_lag_bank",
                          action="store_true", default=False,
                          help="restrict static ACF memory to per-channel top-k train-only lag slots")
    explicit.add_argument("--no-mamba-acf-channel-lag-bank",
                          "--no_mamba_acf_channel_lag_bank",
                          dest="mamba_acf_channel_lag_bank",
                          action="store_false")
    explicit.add_argument("--mamba-acf-channel-lag-top-k",
                          "--mamba_acf_channel_lag_top_k",
                          dest="mamba_acf_channel_lag_top_k",
                          type=int, default=2,
                          help="number of train-only lag slots retained per channel")
    explicit.add_argument("--mamba-acf-lag-mode", default="fixed")
    explicit.add_argument("--mamba-acf-max-lag", type=int, default=720)
    explicit.add_argument("--mamba-acf-top-k", type=int, default=8)
    explicit.add_argument("--mamba-acf-min-lag", type=int, default=2)
    explicit.add_argument("--mamba-acf-min-abs", type=float, default=0.1)
    explicit.add_argument("--mamba-acf-confidence-z", type=float, default=1.96)
    explicit.add_argument("--mamba-acf-min-separation", type=int, default=2)
    explicit.add_argument("--mamba-acf-detrend", choices=("none", "mean", "linear"), default="mean")
    explicit.add_argument("--mamba-acf-chunk", type=int, default=64)
    explicit.add_argument("--mamba-use-acf-memory", dest="mamba_use_acf_memory",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-use-acf-memory", dest="mamba_use_acf_memory",
                          action="store_false")
    explicit.add_argument("--mamba-dynamic-acf", dest="mamba_dynamic_acf",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-dynamic-acf", dest="mamba_dynamic_acf",
                          action="store_false")
    explicit.add_argument("--mamba-acf-temperature", type=float, default=4.0)
    explicit.add_argument("--mamba-acf-step-reliability", dest="mamba_acf_step_reliability",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-acf-step-reliability", dest="mamba_acf_step_reliability",
                          action="store_false")
    explicit.add_argument("--mamba-acf-reliability-interpolation",
                          "--mamba_acf_reliability_interpolation",
                          dest="mamba_acf_reliability_interpolation",
                          action="store_true", default=False,
                          help="learn bounded per-step/channel static ACF reliability interpolation")
    explicit.add_argument("--no-mamba-acf-reliability-interpolation",
                          "--no_mamba_acf_reliability_interpolation",
                          dest="mamba_acf_reliability_interpolation",
                          action="store_false")
    explicit.add_argument("--mamba-acf-memory-gate-mode", dest="mamba_acf_memory_gate_mode",
                          choices=("hidden", "variable_token"), default="hidden")
    explicit.add_argument("--mamba-acf-anchor-copy", "--mamba_acf_anchor_copy",
                          dest="mamba_acf_anchor_copy", action="store_true", default=False,
                          help="use the causal signed-ACF lag anchor in MambaACF")
    explicit.add_argument("--no-mamba-acf-anchor-copy", "--no_mamba_acf_anchor_copy",
                          dest="mamba_acf_anchor_copy", action="store_false")
    explicit.add_argument("--mamba-acf-horizon-calibration", dest="mamba_acf_horizon_calibration",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-acf-horizon-calibration", dest="mamba_acf_horizon_calibration",
                          action="store_false")
    explicit.add_argument("--mamba-acf-horizon-calibration-rank",
                          dest="mamba_acf_horizon_calibration_rank", type=int,
                          choices=(1, 2), default=2)
    explicit.add_argument("--mamba-acf-relation-loss", dest="mamba_acf_relation_loss",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-acf-relation-loss", dest="mamba_acf_relation_loss",
                          action="store_false")
    explicit.add_argument("--mamba-acf-relation-batching", choices=("lag_pairs", "random"),
                          default="lag_pairs")
    explicit.add_argument("--mamba-acf-relation-lag-balance", dest="mamba_acf_relation_lag_balance",
                          choices=("random", "balanced"), default="random",
                          help="record the relation sampler's lag-balance contract")
    explicit.add_argument("--mamba-acf-relation-channel-reduce", choices=("mean", "max_abs"),
                          default="mean")
    explicit.add_argument("--mamba-acf-relation-scope", choices=("both", "global", "local"),
                          default="both")
    explicit.add_argument("--mamba-acf-relation-representation",
                          choices=("encoder", "forecast"), default="encoder")
    explicit.add_argument("--mamba-acf-relation-lag-mask", dest="mamba_acf_relation_lag_mask",
                          action="store_true", default=False)
    explicit.add_argument("--no-mamba-acf-relation-lag-mask", dest="mamba_acf_relation_lag_mask",
                          action="store_false")
    explicit.add_argument("--mamba-acf-relation-lag-tolerance", type=int, default=0)
    explicit.add_argument("--mamba-acf-relation-min-lag", dest="mamba_acf_relation_min_lag",
                          type=int, default=0,
                          help="minimum selected lag for relation supervision; use seq_len for no overlap")
    explicit.add_argument("--mamba-acf-relation-target-mode",
                          choices=("hard_selected", "soft_multi_peak", "soft_full"),
                          default="hard_selected")
    explicit.add_argument("--mamba-acf-relation-soft-min-abs", type=float, default=0.1)
    explicit.add_argument("--mamba-acf-relation-diagnostic-min-abs", type=float, default=0.1)
    explicit.add_argument("--mamba-acf-relation-selected-lag-boost", type=float, default=0.0)
    explicit.add_argument("--moving-avg", type=int, default=25)
    explicit.add_argument("--factor", type=int, default=1)
    explicit.add_argument("--top-k", type=int, default=5)
    explicit.add_argument("--num-kernels", type=int, default=6)
    explicit.add_argument("--dropout", type=float, default=0.1)
    explicit.add_argument("--autocon", dest="autocon", action="store_true", default=None)
    explicit.add_argument("--no-autocon", dest="autocon", action="store_false")
    explicit.add_argument("--autocon-wnorm", default="LastVal")
    explicit.add_argument("--autocon-lambda", type=float, default=0.05)
    explicit.add_argument("--autocon-warmup-epochs", type=int, default=0)
    explicit.add_argument("--autocon-acf-source", choices=("residual", "trend", "raw"),
                          default="residual")
    explicit.add_argument("--autocon-temperature", type=float, default=0.2)
    explicit.add_argument("--autocon-relation-temperature", type=float, default=0.25)
    explicit.add_argument("--autocon-sign-weight", type=float, default=0.25)
    explicit.add_argument("--autocon-sign-margin", type=float, default=0.1)
    explicit.add_argument("--autocon-scales", type=int, nargs="+", default=[96])
    explicit.add_argument("--individual", action="store_true")

    benchmark = parser.add_argument_group("GPU benchmark")
    benchmark.add_argument("--run-benchmark", action="store_true",
                           help="explicitly execute GPU forwards; absent means parameter/manifest audit only")
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--benchmark-batch-size", type=int, default=32)
    benchmark.add_argument("--warmup-iters", type=int, default=25)
    benchmark.add_argument("--timed-iters", type=int, default=100)
    benchmark.add_argument("--input-seed", type=int, default=2718)
    benchmark.add_argument(
        "--include-representation",
        action="store_true",
        help=(
            "request and time the auxiliary AutoCon representation when the "
            "model exposes return_repr control; relation loss is excluded"
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, required=True)
    output.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if int(args.benchmark_batch_size) <= 0 or int(args.warmup_iters) < 0 or int(args.timed_iters) <= 0:
        raise SystemExit("benchmark batch size/timing iteration counts must be positive (warmup may be zero)")
    if args.all_validated and args.phase_validation is None:
        raise SystemExit("--all-validated requires --phase-validation")
    if args.all_validated and args.config:
        raise SystemExit("use either --all-validated or explicit --config paths, not both")
    has_config_source = bool(args.config) or bool(args.all_validated)
    if has_config_source == bool(args.model):
        raise SystemExit("provide exactly one source: validated configs, explicit --model, or --all-validated")

    validation_context: Optional[Dict[str, Any]] = None
    if args.phase_validation is not None:
        if args.model:
            raise SystemExit("--phase-validation attests result configs; use --config rather than explicit CLI mode")
        validation_context = _validation_context(_resolve_path(args.phase_validation))
        if args.source_root is not None and _resolve_path(args.source_root) != validation_context["source_root"]:
            raise SystemExit("--source-root conflicts with the validated provenance snapshot")
        config_paths = list(validation_context["by_config"]) if args.all_validated else args.config
        if not config_paths:
            raise SystemExit("--phase-validation requires --config or --all-validated")
        requests = _validated_requests(config_paths, validation_context)
        manifest_path = validation_context["manifest_path"]
        if args.manifest is not None and _resolve_path(args.manifest) != manifest_path:
            raise SystemExit("--manifest conflicts with the manifest recorded by --phase-validation")
    else:
        source_root = _resolve_path(args.source_root) if args.source_root is not None else REPO_ROOT
        if args.config:
            requests = _unverified_requests(args.config, source_root)
        else:
            requests = [_explicit_request(args, source_root)]
        manifest_path = _resolve_path(args.manifest) if args.manifest is not None else None

    requests = _deduplicate_requests(requests)
    if not requests:
        raise SystemExit("no unique resource configurations to measure")

    records = [_measure_request(request, args) for request in requests]
    training_summary: List[Dict[str, Any]] = []
    if manifest_path is not None:
        manifest_records = _read_manifest(manifest_path)
        training_summary = _manifest_elapsed_summary(manifest_records.values())

    try:
        import torch
        torch_environment = {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        torch_environment = {"torch_error": repr(exc)}
    package_environment = {}
    for package_name in ("mamba-ssm", "causal-conv1d"):
        try:
            package_environment[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_environment[package_name] = None
        except Exception as exc:  # pragma: no cover - diagnostics only
            package_environment[package_name + "_error"] = repr(exc)
    payload = {
        "schema": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source_attestation": {
            "phase_validation": str(_resolve_path(args.phase_validation)) if args.phase_validation else None,
            "manifest": str(manifest_path) if manifest_path is not None else None,
            "verified": validation_context is not None,
        },
        "benchmark_protocol": {
            "run_benchmark": bool(args.run_benchmark),
            "device": str(args.device) if args.run_benchmark else None,
            "benchmark_batch_size": int(args.benchmark_batch_size) if args.run_benchmark else None,
            "warmup_forwards": int(args.warmup_iters) if args.run_benchmark else None,
            "timed_forwards": int(args.timed_iters) if args.run_benchmark else None,
            "input_seed": int(args.input_seed),
            "auxiliary_representation_policy": (
                "pass return_repr=true and retain the auxiliary representation"
                if args.include_representation
                else "pass return_repr=false for models declaring support; otherwise use model default"
            ),
            "include_representation": bool(args.include_representation),
            "relation_loss_included": False,
            "statistical_unit_note": (
                "Timing forwards and synthetic windows are resource-load measurements only; "
                "they are not independent forecasting observations or inferential replicates."
            ),
        },
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            **torch_environment,
            **package_environment,
        },
        "resource_records": records,
        "training_elapsed_summary": training_summary,
    }
    _write_outputs(_resolve_path(args.output_dir), payload, bool(args.overwrite))
    print(
        "Wrote {} resource records and {} manifest elapsed groups to {}".format(
            len(records), len(training_summary), _resolve_path(args.output_dir)
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, RuntimeError, ImportError) as exc:
        raise SystemExit("RESOURCE BENCHMARK ERROR: {}".format(exc))
