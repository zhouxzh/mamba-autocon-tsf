import argparse
import hashlib
import json
import os
import re
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_long_term_forecasting_with_AutoCon import Exp_Long_Term_Forecast_with_AutoCon
from exp.exp_long_term_forecasting_with_mamba import (
    Exp_Long_Term_Forecast_with_Mamba1,
    Exp_Long_Term_Forecast_with_Mamba1_AutoCon,
    Exp_Long_Term_Forecast_with_Mamba2,
    Exp_Long_Term_Forecast_with_Mamba2_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTS,
    Exp_Long_Term_Forecast_with_MambaTS_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv2,
    Exp_Long_Term_Forecast_with_MambaTSv2_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv3,
    Exp_Long_Term_Forecast_with_MambaTSv3_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv4,
    Exp_Long_Term_Forecast_with_MambaTSv4_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv5,
    Exp_Long_Term_Forecast_with_MambaTSv5_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv6,
    Exp_Long_Term_Forecast_with_MambaTSv6_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv7,
    Exp_Long_Term_Forecast_with_MambaTSv7_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv8,
    Exp_Long_Term_Forecast_with_MambaTSv8_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv9,
    Exp_Long_Term_Forecast_with_MambaTSv9_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv10,
    Exp_Long_Term_Forecast_with_MambaTSv10_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv11,
    Exp_Long_Term_Forecast_with_MambaTSv11_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv12,
    Exp_Long_Term_Forecast_with_MambaTSv12_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv13,
    Exp_Long_Term_Forecast_with_MambaTSv13_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv14,
    Exp_Long_Term_Forecast_with_MambaTSv14_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv15,
    Exp_Long_Term_Forecast_with_MambaTSv15_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTSv16,
    Exp_Long_Term_Forecast_with_MambaTSv16_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTM1,
    Exp_Long_Term_Forecast_with_MambaTM1_AutoCon,
    Exp_Long_Term_Forecast_with_MambaTM2,
    Exp_Long_Term_Forecast_with_MambaTM2_AutoCon,
    Exp_Long_Term_Forecast_with_MambaSM,
    Exp_Long_Term_Forecast_with_MambaSM_AutoCon,
    Exp_Long_Term_Forecast_with_RaMambaLite,
    Exp_Long_Term_Forecast_with_RaMambaLite_AutoCon,
    Exp_Long_Term_Forecast_with_BiPatchMamba,
    Exp_Long_Term_Forecast_with_BiPatchMamba_AutoCon,
    Exp_Long_Term_Forecast_with_LatentTemporalMamba,
    Exp_Long_Term_Forecast_with_LatentTemporalMamba_AutoCon,
    Exp_Long_Term_Forecast_with_InvertedRaMamba,
    Exp_Long_Term_Forecast_with_InvertedRaMamba_AutoCon,
    Exp_Long_Term_Forecast_with_LinearLatentMamba,
    Exp_Long_Term_Forecast_with_LinearLatentMamba_AutoCon,
    Exp_Long_Term_Forecast_with_NLinearVariableMamba,
    Exp_Long_Term_Forecast_with_NLinearVariableMamba_AutoCon,
    Exp_Long_Term_Forecast_with_MambaACF,
    Exp_Long_Term_Forecast_with_MambaACF_AutoCon,
    Exp_Long_Term_Forecast_with_MambaACF1,
    Exp_Long_Term_Forecast_with_MambaACF1_AutoCon,
)
import random
import numpy as np
import scipy
from data_provider.data_factory import (
    resolve_acf_relation_batching,
    _record_acf_relation_batching,
)
from utils.acf_relation_contract import (
    mamba_acf_prior_required,
    resolve_acf_source,
)
from utils.acf_guidance import compile_channel_lag_support
from utils.mamba_causality_contract import strict_representation_causality_errors


def _configure_cpu_threads():
    """Honor a plan-bound CPU limit before any model work starts."""

    raw_limit = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if not raw_limit:
        return None
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError("AUTOCON_CPU_THREAD_LIMIT must be a positive integer") from exc
    if limit <= 0:
        raise ValueError("AUTOCON_CPU_THREAD_LIMIT must be a positive integer")
    torch.set_num_threads(limit)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The process may be embedded by a caller that already initialized
        # inter-op workers; intra-op limiting still constrains BLAS/OpenMP.
        pass
    return limit


_CPU_THREAD_LIMIT = _configure_cpu_threads()


MAMBA_PLAIN_MODELS = {
    'Mamba1', 'Mamba2', 'MambaTS', 'MambaTSv2', 'MambaTSv3', 'MambaTSv4', 'MambaTSv5', 'MambaTSv6', 'MambaTSv7', 'MambaTSv8', 'MambaTSv9', 'MambaTSv10', 'MambaTSv11', 'MambaTSv12', 'MambaTSv13', 'MambaTSv14', 'MambaTSv15', 'MambaTSv16', 'MambaACF', 'MambaACF1',
    'MambaTM1', 'MambaTM2', 'MambaSM', 'RaMambaLite', 'BiPatchMamba', 'LatentTemporalMamba', 'InvertedRaMamba', 'LinearLatentMamba', 'NLinearVariableMamba'
}
MAMBA_AUTOCON_MODELS = {
    'Mamba1AutoCon', 'Mamba2AutoCon', 'MambaTSAutoCon', 'MambaTSv2AutoCon', 'MambaTSv3AutoCon', 'MambaTSv4AutoCon', 'MambaTSv5AutoCon', 'MambaTSv6AutoCon', 'MambaTSv7AutoCon', 'MambaTSv8AutoCon', 'MambaTSv9AutoCon', 'MambaTSv10AutoCon', 'MambaTSv11AutoCon', 'MambaTSv12AutoCon', 'MambaTSv13AutoCon', 'MambaTSv14AutoCon', 'MambaTSv15AutoCon', 'MambaTSv16AutoCon', 'MambaACFAutoCon',
    'MambaACFLiteAutoCon', 'MambaACF1AutoCon', 'MambaTM1AutoCon',
    'MambaTM2AutoCon', 'MambaSMAutoCon', 'RaMambaLiteAutoCon', 'BiPatchMambaAutoCon', 'LatentTemporalMambaAutoCon', 'InvertedRaMambaAutoCon', 'LinearLatentMambaAutoCon', 'NLinearVariableMambaAutoCon'
}
MAMBA_MODELS = MAMBA_PLAIN_MODELS | MAMBA_AUTOCON_MODELS
MAMBA_EXPERIMENTS = {
    'Mamba1': Exp_Long_Term_Forecast_with_Mamba1,
    'Mamba1AutoCon': Exp_Long_Term_Forecast_with_Mamba1_AutoCon,
    'Mamba2': Exp_Long_Term_Forecast_with_Mamba2,
    'Mamba2AutoCon': Exp_Long_Term_Forecast_with_Mamba2_AutoCon,
    'MambaTS': Exp_Long_Term_Forecast_with_MambaTS,
    'MambaTSAutoCon': Exp_Long_Term_Forecast_with_MambaTS_AutoCon,
    'MambaTSv2': Exp_Long_Term_Forecast_with_MambaTSv2,
    'MambaTSv2AutoCon': Exp_Long_Term_Forecast_with_MambaTSv2_AutoCon,
    'MambaTSv3': Exp_Long_Term_Forecast_with_MambaTSv3,
    'MambaTSv3AutoCon': Exp_Long_Term_Forecast_with_MambaTSv3_AutoCon,
    'MambaTSv4': Exp_Long_Term_Forecast_with_MambaTSv4,
    'MambaTSv4AutoCon': Exp_Long_Term_Forecast_with_MambaTSv4_AutoCon,
    'MambaTSv5': Exp_Long_Term_Forecast_with_MambaTSv5,
    'MambaTSv5AutoCon': Exp_Long_Term_Forecast_with_MambaTSv5_AutoCon,
    'MambaTSv6': Exp_Long_Term_Forecast_with_MambaTSv6,
    'MambaTSv6AutoCon': Exp_Long_Term_Forecast_with_MambaTSv6_AutoCon,
    'MambaTSv7': Exp_Long_Term_Forecast_with_MambaTSv7,
    'MambaTSv7AutoCon': Exp_Long_Term_Forecast_with_MambaTSv7_AutoCon,
    'MambaTSv8': Exp_Long_Term_Forecast_with_MambaTSv8,
    'MambaTSv8AutoCon': Exp_Long_Term_Forecast_with_MambaTSv8_AutoCon,
    'MambaTSv9': Exp_Long_Term_Forecast_with_MambaTSv9,
    'MambaTSv9AutoCon': Exp_Long_Term_Forecast_with_MambaTSv9_AutoCon,
    'MambaTSv10': Exp_Long_Term_Forecast_with_MambaTSv10,
    'MambaTSv10AutoCon': Exp_Long_Term_Forecast_with_MambaTSv10_AutoCon,
    'MambaTSv11': Exp_Long_Term_Forecast_with_MambaTSv11,
    'MambaTSv11AutoCon': Exp_Long_Term_Forecast_with_MambaTSv11_AutoCon,
    'MambaTSv12': Exp_Long_Term_Forecast_with_MambaTSv12,
    'MambaTSv12AutoCon': Exp_Long_Term_Forecast_with_MambaTSv12_AutoCon,
    'MambaTSv13': Exp_Long_Term_Forecast_with_MambaTSv13,
    'MambaTSv13AutoCon': Exp_Long_Term_Forecast_with_MambaTSv13_AutoCon,
    'MambaTSv14': Exp_Long_Term_Forecast_with_MambaTSv14,
    'MambaTSv14AutoCon': Exp_Long_Term_Forecast_with_MambaTSv14_AutoCon,
    'MambaTSv15': Exp_Long_Term_Forecast_with_MambaTSv15,
    'MambaTSv15AutoCon': Exp_Long_Term_Forecast_with_MambaTSv15_AutoCon,
    'MambaTSv16': Exp_Long_Term_Forecast_with_MambaTSv16,
    'MambaTSv16AutoCon': Exp_Long_Term_Forecast_with_MambaTSv16_AutoCon,
    'MambaTM1': Exp_Long_Term_Forecast_with_MambaTM1,
    'MambaTM1AutoCon': Exp_Long_Term_Forecast_with_MambaTM1_AutoCon,
    'MambaTM2': Exp_Long_Term_Forecast_with_MambaTM2,
    'MambaTM2AutoCon': Exp_Long_Term_Forecast_with_MambaTM2_AutoCon,
    'MambaSM': Exp_Long_Term_Forecast_with_MambaSM,
    'MambaSMAutoCon': Exp_Long_Term_Forecast_with_MambaSM_AutoCon,
    'RaMambaLite': Exp_Long_Term_Forecast_with_RaMambaLite,
    'RaMambaLiteAutoCon': Exp_Long_Term_Forecast_with_RaMambaLite_AutoCon,
    'BiPatchMamba': Exp_Long_Term_Forecast_with_BiPatchMamba,
    'BiPatchMambaAutoCon': Exp_Long_Term_Forecast_with_BiPatchMamba_AutoCon,
    'LatentTemporalMamba': Exp_Long_Term_Forecast_with_LatentTemporalMamba,
    'LatentTemporalMambaAutoCon': Exp_Long_Term_Forecast_with_LatentTemporalMamba_AutoCon,
    'InvertedRaMamba': Exp_Long_Term_Forecast_with_InvertedRaMamba,
    'InvertedRaMambaAutoCon': Exp_Long_Term_Forecast_with_InvertedRaMamba_AutoCon,
    'LinearLatentMamba': Exp_Long_Term_Forecast_with_LinearLatentMamba,
    'LinearLatentMambaAutoCon': Exp_Long_Term_Forecast_with_LinearLatentMamba_AutoCon,
    'NLinearVariableMamba': Exp_Long_Term_Forecast_with_NLinearVariableMamba,
    'NLinearVariableMambaAutoCon': Exp_Long_Term_Forecast_with_NLinearVariableMamba_AutoCon,
    'MambaACF': Exp_Long_Term_Forecast_with_MambaACF,
    'MambaACFAutoCon': Exp_Long_Term_Forecast_with_MambaACF_AutoCon,
    'MambaACFLiteAutoCon': Exp_Long_Term_Forecast_with_MambaACF_AutoCon,
    'MambaACF1': Exp_Long_Term_Forecast_with_MambaACF1,
    'MambaACF1AutoCon': Exp_Long_Term_Forecast_with_MambaACF1_AutoCon,
}


_SETTING_COMPONENT_LIMIT = 180
_MAMBA_RUNTIME_CONTRACT_VERSION = 1


def _contract_source_value(source, name, default=None):
    """Read a runtime option from an argparse namespace or JSON mapping."""

    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _contract_bool(value, label):
    """Normalize persisted booleans without treating ``'false'`` as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            return True
        if token in {"false", "0", "no", "n", "off", ""}:
            return False
    raise ValueError("{} must be a boolean".format(label))


def _contract_int(value, label, *, minimum=0):
    """Normalize an integer contract field without truncating floats."""

    if minimum > 0:
        return _strict_positive_int(value, label)
    return _strict_nonnegative_int(value, label)


def _contract_float(value, label):
    """Normalize a finite floating-point contract field."""

    return _strict_bounded_float(value, label)


def _contract_int_list(value, label, *, allow_empty=False, preserve_order=False):
    """Normalize a list-valued model option using its model semantics."""

    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        values = [item for item in text.replace(",", " ").split() if item]
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError("{} must be a sequence of integers".format(label)) from exc
    if not values and not allow_empty:
        raise ValueError("{} must be non-empty".format(label))
    parsed = [_contract_int(item, label, minimum=1) for item in values]
    if preserve_order:
        return parsed
    return sorted(set(parsed))


def _effective_mamba_headdim(d_model, expand, requested):
    """Resolve the Mamba-2 head dimension used by model construction.

    Mamba-2 requires the expanded hidden width to be divisible by
    ``headdim``.  The model keeps the largest valid divisor not exceeding the
    requested value, so replay identity must compare this effective value
    rather than a syntactically different request that constructs the same
    operator.
    """

    expanded = int(d_model) * int(expand)
    requested = int(requested)
    if expanded <= 0 or requested <= 0:
        raise ValueError("Mamba dimensions must be positive")
    if expanded % requested == 0:
        return requested
    return max(
        divisor for divisor in range(min(expanded, requested), 0, -1)
        if expanded % divisor == 0
    )


def _mamba_runtime_contract_payload(source):
    """Return the normalized model/runtime identity for a Mamba checkpoint.

    Fields absent from legacy configs use the parser/model defaults.  The
    payload intentionally contains options that can alter inference or the
    AutoCon model construction while excluding device, output-directory and
    evaluation-only telemetry knobs.
    """

    model = _contract_source_value(source, "model", None)
    if model is None:
        return None
    model = str(model).strip()
    if model not in MAMBA_MODELS:
        return None

    is_acf = model.startswith("MambaACF")
    is_tm = model.startswith("MambaTM")
    is_sm = model.startswith("MambaSM")
    is_ra = model.startswith("RaMambaLite")
    is_inverted_ra = model.startswith("InvertedRaMamba")
    is_linear_latent = model.startswith("LinearLatentMamba")
    is_nlinear_variable = model.startswith("NLinearVariableMamba")
    is_bipatch = model.startswith("BiPatchMamba")
    is_ts = model.startswith("MambaTS")
    default_d_model = 64 if (is_acf or is_tm or is_ts) else (32 if (is_sm or is_ra or is_bipatch or is_inverted_ra or is_linear_latent or is_nlinear_variable) else 512)
    default_d_ff = 128 if is_tm else (64 if (is_sm or is_ra or is_bipatch or is_inverted_ra or is_linear_latent or is_nlinear_variable) else (default_d_model * 2 if (is_acf or is_ts) else 2048))
    payload = {
        "model": model,
        "AutoCon": _contract_bool(
            _contract_source_value(source, "AutoCon", False), "AutoCon"
        ),
        "AutoCon_wnorm": str(
            _contract_source_value(source, "AutoCon_wnorm", "LastVal")
        ).strip(),
        "features": str(
            _contract_source_value(source, "features", "M")
        ).strip(),
        "seq_len": _contract_int(
            _contract_source_value(source, "seq_len", 96), "seq_len", minimum=1
        ),
        "label_len": _contract_int(
            _contract_source_value(source, "label_len", 48), "label_len"
        ),
        "pred_len": _contract_int(
            _contract_source_value(source, "pred_len", 96), "pred_len", minimum=1
        ),
        "enc_in": _contract_int(
            _contract_source_value(source, "enc_in", 7), "enc_in", minimum=1
        ),
        "c_out": _contract_int(
            _contract_source_value(source, "c_out", 7), "c_out", minimum=1
        ),
        "d_model": _contract_int(
            _contract_source_value(source, "d_model", default_d_model),
            "d_model", minimum=1,
        ),
        "d_ff": _contract_int(
            _contract_source_value(source, "d_ff", default_d_ff),
            "d_ff", minimum=1,
        ),
        "e_layers": _contract_int(
            _contract_source_value(source, "e_layers", 2 if is_acf else 1),
            "e_layers", minimum=1,
        ),
        "dropout": _contract_float(
            _contract_source_value(source, "dropout", 0.1), "dropout"
        ),
        "embed": str(_contract_source_value(source, "embed", "timeF")).strip(),
        "freq": str(_contract_source_value(source, "freq", "h")).strip(),
        "wo_freq": str(_contract_source_value(source, "wo_freq", "") or "").strip(),
        "mamba_d_state": _contract_int(
            _contract_source_value(
                source, "mamba_d_state", 16 if (is_acf or is_ts or is_ra or is_bipatch or is_inverted_ra or is_linear_latent or is_nlinear_variable) else 64
            ),
            "mamba_d_state", minimum=1,
        ),
        "mamba_d_conv": _contract_int(
            _contract_source_value(source, "mamba_d_conv", 4),
            "mamba_d_conv", minimum=1,
        ),
        "mamba_expand": _contract_int(
            _contract_source_value(source, "mamba_expand", 2),
            "mamba_expand", minimum=1,
        ),
    }
    if is_bipatch:
        payload["mamba_bipatch_token_mode"] = str(
            _contract_source_value(source, "mamba_bipatch_token_mode", "mixing")
        ).strip().lower()
        if payload["mamba_bipatch_token_mode"] not in {"mixing", "independent"}:
            raise ValueError("mamba_bipatch_token_mode must be mixing or independent")
        payload["mamba_bipatch_repr_gate"] = _contract_bool(
            _contract_source_value(source, "mamba_bipatch_repr_gate", False),
            "mamba_bipatch_repr_gate",
        ) and payload["AutoCon"]
    if is_nlinear_variable:
        payload["mamba_ts_residual_init"] = _contract_float(
            _contract_source_value(source, "mamba_ts_residual_init", 0.0),
            "mamba_ts_residual_init",
        )
        payload["mamba_variable_repr_fusion"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_variable_repr_fusion", False),
                "mamba_variable_repr_fusion",
            )
            and payload["AutoCon"]
        )
        payload["mamba_variable_acf_memory_selector"] = bool(
            _contract_bool(
                _contract_source_value(
                    source, "mamba_variable_acf_memory_selector", False
                ),
                "mamba_variable_acf_memory_selector",
            )
            and payload["AutoCon"]
            and _contract_bool(
                _contract_source_value(source, "mamba_use_acf_memory", False),
                "mamba_use_acf_memory",
            )
        )
    if model in {"Mamba2", "Mamba2AutoCon"}:
        payload["mamba2_repr_forecast_fusion"] = bool(
            _contract_bool(
                _contract_source_value(
                    source, "mamba2_repr_forecast_fusion", False
                ),
                "mamba2_repr_forecast_fusion",
            )
            and payload["AutoCon"]
        )
    if is_ts:
        # ``MambaTSv2`` is a separate registered model so its construction
        # cannot collide with v1 checkpoints.  Legacy MambaTS names remain v1
        # even when an old config does not carry an explicit variant field.
        variant_default = (
            "v16" if model.lower().startswith("mambatsv16")
            else "v15" if model.lower().startswith("mambatsv15")
            else "v14" if model.lower().startswith("mambatsv14")
            else "v13" if model.lower().startswith("mambatsv13")
            else "v12" if model.lower().startswith("mambatsv12")
            else "v11" if model.lower().startswith("mambatsv11")
            else "v10" if model.lower().startswith("mambatsv10")
            else "v9" if model.lower().startswith("mambatsv9")
            else "v8" if model.lower().startswith("mambatsv8")
            else "v7" if model.lower().startswith("mambatsv7")
            else "v6" if model.lower().startswith("mambatsv6")
            else "v5" if model.lower().startswith("mambatsv5")
            else "v4" if model.lower().startswith("mambatsv4")
            else "v3" if model.lower().startswith("mambatsv3")
            else "v2" if model.lower().startswith("mambatsv2")
            else "v1"
        )
        payload["mamba_ts_variant"] = str(
            _contract_source_value(source, "mamba_ts_variant", variant_default)
        ).strip().lower()
        if payload["mamba_ts_variant"] not in {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
            raise ValueError("mamba_ts_variant must be v1 through v16")
        if model.lower().startswith("mambatsv16") and payload["mamba_ts_variant"] != "v16":
            raise ValueError("MambaTSv16 identifiers require mamba_ts_variant=v16")
        if model.lower().startswith("mambatsv15") and payload["mamba_ts_variant"] != "v15":
            raise ValueError("MambaTSv15 identifiers require mamba_ts_variant=v15")
        if model.lower().startswith("mambatsv14") and payload["mamba_ts_variant"] != "v14":
            raise ValueError("MambaTSv14 identifiers require mamba_ts_variant=v14")
        if model.lower().startswith("mambatsv13") and payload["mamba_ts_variant"] != "v13":
            raise ValueError("MambaTSv13 identifiers require mamba_ts_variant=v13")
        if model.lower().startswith("mambatsv12") and payload["mamba_ts_variant"] != "v12":
            raise ValueError("MambaTSv12 identifiers require mamba_ts_variant=v12")
        if model.lower().startswith("mambatsv11") and payload["mamba_ts_variant"] != "v11":
            raise ValueError("MambaTSv11 identifiers require mamba_ts_variant=v11")
        if model.lower().startswith("mambatsv10") and payload["mamba_ts_variant"] != "v10":
            raise ValueError("MambaTSv10 identifiers require mamba_ts_variant=v10")
        if model.lower().startswith("mambatsv9") and payload["mamba_ts_variant"] != "v9":
            raise ValueError("MambaTSv9 identifiers require mamba_ts_variant=v9")
        if model.lower().startswith("mambatsv8") and payload["mamba_ts_variant"] != "v8":
            raise ValueError("MambaTSv8 identifiers require mamba_ts_variant=v8")
        if model.lower().startswith("mambatsv7") and payload["mamba_ts_variant"] != "v7":
            raise ValueError("MambaTSv7 identifiers require mamba_ts_variant=v7")
        if model.lower().startswith("mambatsv6") and payload["mamba_ts_variant"] != "v6":
            raise ValueError("MambaTSv6 identifiers require mamba_ts_variant=v6")
        if model.lower().startswith("mambatsv5") and payload["mamba_ts_variant"] != "v5":
            raise ValueError("MambaTSv5 identifiers require mamba_ts_variant=v5")
        if model.lower().startswith("mambatsv4") and payload["mamba_ts_variant"] != "v4":
            raise ValueError("MambaTSv4 identifiers require mamba_ts_variant=v4")
        if model.lower().startswith("mambatsv2") and payload["mamba_ts_variant"] != "v2":
            raise ValueError("MambaTSv2 identifiers require mamba_ts_variant=v2")
        if model.lower().startswith("mambatsv3") and payload["mamba_ts_variant"] != "v3":
            raise ValueError("MambaTSv3 identifiers require mamba_ts_variant=v3")
        if model in {"MambaTS", "MambaTSAutoCon"} and payload["mamba_ts_variant"] != "v1":
            raise ValueError("MambaTS v1 identifiers require mamba_ts_variant=v1")
        payload["mamba_ts_mixer_type"] = str(
            _contract_source_value(source, "mamba_ts_mixer_type", "mamba1")
        ).strip().lower()
        if payload["mamba_ts_mixer_type"] not in {"mamba1", "mamba2"}:
            raise ValueError("mamba_ts_mixer_type must be mamba1 or mamba2")
        if (
            payload["mamba_ts_mixer_type"] == "mamba2"
            and payload["mamba_ts_variant"] == "v12"
        ):
            raise ValueError("MambaTS v12 Delta branches currently require mamba1 mixer")
        payload["mamba_ts_context_rank"] = None
        if payload["mamba_ts_variant"] in {"v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
            payload["mamba_ts_context_rank"] = _contract_int(
                _contract_source_value(source, "mamba_ts_context_rank", 16),
                "mamba_ts_context_rank", minimum=1,
            )
        payload["mamba_ts_channel_agnostic_scan"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_channel_agnostic_scan", False),
                "mamba_ts_channel_agnostic_scan",
            )
            and payload["mamba_ts_variant"] == "v3"
        )
        payload["mamba_ts_residual_init"] = _contract_float(
            _contract_source_value(source, "mamba_ts_residual_init", 0.0),
            "mamba_ts_residual_init",
        )
        if not -0.99 < payload["mamba_ts_residual_init"] < 0.99:
            raise ValueError("mamba_ts_residual_init must be in (-0.99, 0.99)")
        payload["mamba_ts_repr_forecast_fusion"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_repr_forecast_fusion", False),
                "mamba_ts_repr_forecast_fusion",
            )
            and payload["AutoCon"]
            and payload["mamba_ts_variant"] in {"v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12", "v13", "v14", "v15", "v16"}
        )
        payload["mamba_ts_repr_forecast_channelwise"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_repr_forecast_channelwise", False),
                "mamba_ts_repr_forecast_channelwise",
            )
            and payload["mamba_ts_repr_forecast_fusion"]
        )
        payload["mamba_ts_channel_adapter"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_channel_adapter", False),
                "mamba_ts_channel_adapter",
            )
            and payload["AutoCon"]
            and payload["mamba_ts_variant"] == "v3"
        )
        payload["mamba_ts_channel_adapter_rank"] = (
            _contract_int(
                _contract_source_value(source, "mamba_ts_channel_adapter_rank", 8),
                "mamba_ts_channel_adapter_rank", minimum=1,
            )
            if payload["mamba_ts_channel_adapter"] else None
        )
        payload["mamba_ts_revin_affine"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_revin_affine", False),
                "mamba_ts_revin_affine",
            )
            and payload["AutoCon"]
            and payload["mamba_ts_variant"] == "v3"
        )
        payload["mamba_ts_global_channel_mixer"] = bool(
            _contract_bool(
                _contract_source_value(
                    source, "mamba_ts_global_channel_mixer", False
                ),
                "mamba_ts_global_channel_mixer",
            )
            and payload["AutoCon"]
            and payload["mamba_ts_variant"] == "v3"
        )
        payload["mamba_ts_global_channel_mixer_rank"] = (
            _contract_int(
                _contract_source_value(
                    source, "mamba_ts_global_channel_mixer_rank", 4
                ),
                "mamba_ts_global_channel_mixer_rank",
                minimum=1,
            )
            if payload["mamba_ts_global_channel_mixer"] else None
        )
        payload["mamba_ts_graph_spatial"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_graph_spatial", False),
                "mamba_ts_graph_spatial",
            )
            and payload["AutoCon"]
            and payload["mamba_ts_variant"] == "v3"
        )
        payload["mamba_ts_graph_expected_sha256"] = (
            str(
                _contract_source_value(
                    source, "mamba_ts_graph_expected_sha256", ""
                )
                or ""
            ).strip().lower()
            if payload["mamba_ts_graph_spatial"] else None
        )
        payload["mamba_ts_graph_path"] = (
            str(
                _contract_source_value(source, "mamba_ts_graph_path", "")
                or ""
            ).strip()
            if payload["mamba_ts_graph_spatial"] else None
        )
        payload["mamba_ts_repr_linear_expert"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_repr_linear_expert", False),
                "mamba_ts_repr_linear_expert",
            )
            and payload["AutoCon"]
            and payload["mamba_ts_variant"] in {"v3", "v4", "v5", "v6", "v7", "v8"}
        )
        payload["mamba_ts_repr_linear_expert_mode"] = str(
            _contract_source_value(
                source, "mamba_ts_repr_linear_expert_mode", "residual"
            )
            or "residual"
        ).strip().lower()
        if payload["mamba_ts_repr_linear_expert_mode"] not in {"residual", "blend", "nlinear_blend"}:
            raise ValueError(
                "mamba_ts_repr_linear_expert_mode must be residual, blend, or nlinear_blend"
            )
        payload["mamba_ts_repr_linear_expert_rank"] = _contract_int(
            _contract_source_value(source, "mamba_ts_repr_linear_expert_rank", 16),
            "mamba_ts_repr_linear_expert_rank", minimum=1,
        )
        payload["mamba_ts_repr_linear_blend_bias"] = _contract_float(
            _contract_source_value(source, "mamba_ts_repr_linear_blend_bias", -4.0),
            "mamba_ts_repr_linear_blend_bias",
        )
        if not -12.0 <= payload["mamba_ts_repr_linear_blend_bias"] <= 0.0:
            raise ValueError("mamba_ts_repr_linear_blend_bias must be in [-12, 0]")
        payload["mamba_ts_variable_permutation"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_variable_permutation", False),
                "mamba_ts_variable_permutation",
            )
            and payload["mamba_ts_variant"] in {"v3", "v4", "v5", "v6", "v7"}
        )
        payload["mamba_ts_acf_memory_selector"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_acf_memory_selector", False),
                "mamba_ts_acf_memory_selector",
            )
            and payload["AutoCon"]
            and payload["mamba_ts_variant"] in {"v4", "v5", "v7"}
            and _contract_bool(
                _contract_source_value(source, "mamba_use_acf_memory", False),
                "mamba_use_acf_memory",
            )
        )
        payload["mamba_ts_acf_memory_channel_selector"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_acf_memory_channel_selector", False),
                "mamba_ts_acf_memory_channel_selector",
            )
            and payload["mamba_ts_acf_memory_selector"]
        )
        payload["mamba_ts_forecast_acf_consistency"] = bool(
            _contract_bool(
                _contract_source_value(source, "mamba_ts_forecast_acf_consistency", False),
                "mamba_ts_forecast_acf_consistency",
            )
            and payload["AutoCon"]
            and payload["mamba_ts_variant"] in {"v3", "v4", "v5"}
        )
        payload["mamba_ts_forecast_acf_weight"] = float(
            _contract_source_value(source, "mamba_ts_forecast_acf_weight", 0.0)
        )
        payload["mamba_ts_forecast_acf_top_k"] = _contract_int(
            _contract_source_value(source, "mamba_ts_forecast_acf_top_k", 4),
            "mamba_ts_forecast_acf_top_k", minimum=1,
        )
        payload["mamba_ts_forecast_acf_min_lag"] = _contract_int(
            _contract_source_value(source, "mamba_ts_forecast_acf_min_lag", 2),
            "mamba_ts_forecast_acf_min_lag", minimum=0,
        )
        payload["mamba_ts_forecast_acf_min_abs"] = float(
            _contract_source_value(source, "mamba_ts_forecast_acf_min_abs", 0.10)
        )
        payload["mamba_ts_spectral_top_k"] = _contract_int(
            _contract_source_value(source, "mamba_ts_spectral_top_k", 4),
            "mamba_ts_spectral_top_k", minimum=1,
        ) if payload["mamba_ts_variant"] == "v8" else None
        if payload["mamba_ts_variant"] == "v12":
            raw_multipliers = _contract_source_value(
                source, "mamba_ts_delta_multipliers", [1.0, 2.0]
            )
            if isinstance(raw_multipliers, (str, bytes)):
                text = raw_multipliers.decode() if isinstance(raw_multipliers, bytes) else raw_multipliers
                raw_multipliers = [item for item in text.replace(",", " ").split() if item]
            try:
                multipliers = tuple(
                    _contract_float(value, "mamba_ts_delta_multipliers")
                    for value in raw_multipliers
                )
            except TypeError as exc:
                raise ValueError("mamba_ts_delta_multipliers must be a sequence") from exc
            if len(multipliers) != 2 or multipliers[0] != 1.0 or multipliers[1] <= 1.0:
                raise ValueError(
                    "MambaTS v12 requires mamba_ts_delta_multipliers=(1.0, scale) with scale > 1"
                )
            payload["mamba_ts_delta_multipliers"] = multipliers
            payload["mamba_ts_dt_min"] = _contract_float(
                _contract_source_value(source, "mamba_ts_dt_min", 0.001),
                "mamba_ts_dt_min",
            )
            payload["mamba_ts_dt_max"] = _contract_float(
                _contract_source_value(source, "mamba_ts_dt_max", 0.1),
                "mamba_ts_dt_max",
            )
            if not 0 < payload["mamba_ts_dt_min"] < payload["mamba_ts_dt_max"]:
                raise ValueError("MambaTS v12 requires 0 < mamba_ts_dt_min < mamba_ts_dt_max")
        else:
            payload["mamba_ts_delta_multipliers"] = None
            payload["mamba_ts_dt_min"] = None
            payload["mamba_ts_dt_max"] = None
        payload["patch_len"] = _contract_int(
            _contract_source_value(source, "patch_len", 16),
            "patch_len", minimum=1,
        )
        payload["patch_stride"] = _contract_int(
            _contract_source_value(source, "patch_stride", 8),
            "patch_stride", minimum=1,
        )
        if payload["patch_len"] > payload["seq_len"]:
            raise ValueError("patch_len cannot exceed seq_len for MambaTS")
        if payload["mamba_ts_variant"] in {"v4", "v5", "v7", "v10"}:
            payload["mamba_use_acf_memory"] = _contract_bool(
                _contract_source_value(source, "mamba_use_acf_memory", False),
                "mamba_use_acf_memory",
            )
            payload["mamba_dynamic_acf"] = _contract_bool(
                _contract_source_value(source, "mamba_dynamic_acf", False),
                "mamba_dynamic_acf",
            )
            payload["mamba_acf_temperature"] = _contract_float(
                _contract_source_value(source, "mamba_acf_temperature", 4.0),
                "mamba_acf_temperature",
            )
            payload["mamba_acf_lags"] = _contract_int_list(
                _contract_source_value(source, "mamba_acf_lags", [24, 168, 336]),
                "mamba_acf_lags", allow_empty=True,
            )
            payload["AutoCon_acf_source"] = resolve_acf_source(source)
            payload["mamba_acf_repr_fusion"] = bool(
                _contract_bool(
                    _contract_source_value(source, "mamba_acf_repr_fusion", False),
                    "mamba_acf_repr_fusion",
                )
                and payload["AutoCon"]
                and payload["mamba_use_acf_memory"]
                and payload["mamba_ts_variant"] in {"v4", "v5", "v7"}
            )
        else:
            payload["mamba_use_acf_memory"] = None
            payload["mamba_dynamic_acf"] = None
            payload["mamba_acf_temperature"] = None
            payload["mamba_acf_lags"] = None
            payload["AutoCon_acf_source"] = None
            payload["mamba_acf_repr_fusion"] = None
    if model in {
        "Mamba2", "Mamba2AutoCon", "MambaACF", "MambaACFAutoCon",
        "MambaACFLiteAutoCon",
        "MambaTM2", "MambaTM2AutoCon",
    }:
        requested_headdim = _contract_int(
            _contract_source_value(source, "mamba_headdim", 16),
            "mamba_headdim", minimum=1,
        )
        payload["mamba_headdim"] = _effective_mamba_headdim(
            payload["d_model"], payload["mamba_expand"], requested_headdim
        )
    if is_acf:
        bool_fields = (
            "mamba_bidirectional", "mamba_channel_independent",
            "mamba_causal_embedding", "mamba_channel_gate",
            "mamba_data_dependent_channel_mixer",
            "mamba_acf_scale_conditioning", "mamba_acf_channel_gate",
            "mamba_variable_mixer",
            "mamba_variable_gdd_mixer",
            "mamba_variable_repr_fusion",
            # These switches affect the training-time token order.  They are
            # inactive during ``model.eval()``, but remain part of the
            # checkpoint construction/training provenance and must not be
            # silently changed on a replay command.
            "mamba_variable_permutation",
            "mamba_acf_channel_conditioning", "mamba_patch_mixer",
            "mamba_patch_permutation",
                "mamba_local_mixer",
                "mamba_low_rank_head", "mamba_acf_repr_fusion",
                "mamba_use_acf_memory", "mamba_dynamic_acf",
                "mamba_acf_relation_loss",
                "mamba_acf_step_reliability", "mamba_acf_reliability_interpolation",
            "mamba_acf_local_agreement", "mamba_acf_anchor_copy",
            "mamba_acf_horizon_calibration", "mamba_acf_channel_lag_bank",
        )
        for name in bool_fields:
            payload[name] = _contract_bool(
                _contract_source_value(source, name, False), name
            )
        # Representation fusion is constructed only for the AutoCon path
        # (the plain forecaster keeps the requested flag for provenance but
        # never reads it during inference).  Compare the effective branch so
        # a dormant compatibility flag cannot force a false replay mismatch
        # or an unnecessary sidecar requirement.
        payload["mamba_acf_repr_fusion"] = bool(
            payload["mamba_acf_repr_fusion"] and payload["AutoCon"]
        )
        # Variable-token representation fusion is an AutoCon-only adapter and
        # is meaningful only when the channel-preserving variable branch is
        # constructed.  Persist the effective state so dormant flags in old
        # plain checkpoints cannot create replay mismatches.
        payload["mamba_variable_repr_fusion"] = bool(
            payload["mamba_variable_repr_fusion"]
            and payload["AutoCon"]
            and payload["mamba_variable_mixer"]
        )
        # The channel-preserving GDD adapter is only constructed on the
        # variable-token branch.  Canonicalize a dormant request so replay
        # contracts describe the effective architecture rather than an
        # ignored compatibility flag.
        payload["mamba_variable_gdd_mixer"] = bool(
            payload["mamba_variable_gdd_mixer"]
            and payload["mamba_variable_mixer"]
        )
        int_fields = (
            ("mamba_token_stride", 1), ("mamba_channel_gate_rank", 8),
            ("mamba_data_dependent_channel_mixer_rank", 8),
            ("mamba_variable_gdd_mixer_rank", 8),
            ("mamba_variable_mixer_depth", 1), ("mamba_variable_group_size", 1),
            ("mamba_patch_mixer_depth", 1), ("mamba_patch_len", 16),
            ("mamba_patch_stride", 8), ("mamba_local_patch_len", 16),
            ("mamba_local_patch_stride", 8), ("mamba_local_rank", 16),
            ("mamba_local_depth", 1), ("mamba_temporal_rank", 16),
            ("mamba_acf_local_min_pairs", 16),
            ("mamba_acf_horizon_calibration_rank", 2),
            ("mamba_acf_channel_lag_top_k", 2),
        )
        # Numeric knobs that only parameterize an optional branch are
        # canonicalized to ``None`` while that branch is disabled.  This is
        # important for legacy replay: an inactive rank/patch width cannot
        # change the constructed forecast function, so its historical value
        # must not create a false contract mismatch.
        active_numeric = {
            "mamba_channel_gate_rank": payload["mamba_channel_gate"],
            "mamba_data_dependent_channel_mixer_rank": payload[
                "mamba_data_dependent_channel_mixer"
            ],
            "mamba_variable_gdd_mixer_rank": payload[
                "mamba_variable_gdd_mixer"
            ],
            "mamba_variable_mixer_depth": payload["mamba_variable_mixer"],
            "mamba_variable_group_size": payload["mamba_variable_mixer"],
            "mamba_patch_mixer_depth": payload["mamba_patch_mixer"],
            "mamba_patch_len": payload["mamba_patch_mixer"],
            "mamba_patch_stride": payload["mamba_patch_mixer"],
            "mamba_local_patch_len": payload["mamba_local_mixer"],
            "mamba_local_patch_stride": payload["mamba_local_mixer"],
            "mamba_local_rank": payload["mamba_local_mixer"],
            "mamba_local_depth": payload["mamba_local_mixer"],
            "mamba_acf_local_min_pairs": payload["mamba_acf_local_agreement"],
            "mamba_acf_horizon_calibration_rank": payload[
                "mamba_acf_horizon_calibration"
            ],
            "mamba_acf_channel_lag_top_k": payload["mamba_acf_channel_lag_bank"],
            "mamba_temporal_rank": payload["mamba_low_rank_head"],
        }
        for name, default in int_fields:
            parsed = _contract_int(
                _contract_source_value(source, name, default), name,
                minimum=1,
            )
            payload[name] = parsed if active_numeric.get(name, True) else None
        float_fields = (
            ("mamba_acf_temperature", 4.0),
            ("mamba_variable_gdd_init_scale", 0.0),
        )
        for name, default in float_fields:
            parsed = _contract_float(
                _contract_source_value(source, name, default), name
            )
            if name == "mamba_variable_gdd_init_scale":
                if not 0.0 <= parsed <= 1.0:
                    raise ValueError(
                        "mamba_variable_gdd_init_scale must be in [0, 1]"
                    )
                payload[name] = (
                    parsed if payload["mamba_variable_gdd_mixer"] else None
                )
            else:
                payload[name] = (
                    parsed
                    if payload["mamba_use_acf_memory"]
                    else None
                )
        payload["AutoCon_acf_source"] = resolve_acf_source(source)
        payload["mamba_acf_lags"] = _contract_int_list(
            _contract_source_value(source, "mamba_acf_lags", [24, 168, 336]),
            "mamba_acf_lags", allow_empty=True,
        )
        payload["mamba_scales"] = _contract_int_list(
            _contract_source_value(source, "mamba_scales", [1, 2]),
            "mamba_scales",
        )
        if 1 not in payload["mamba_scales"]:
            payload["mamba_scales"].insert(0, 1)
        payload["mamba_scales"] = sorted(set(payload["mamba_scales"]))
        payload["mamba_acf_lag_mode"] = str(
            _contract_source_value(source, "mamba_acf_lag_mode", "fixed")
        ).strip().lower()
        payload["mamba_acf_memory_gate_mode"] = str(
            _contract_source_value(source, "mamba_acf_memory_gate_mode", "hidden")
        ).strip().lower()
        relation_representation = str(
            _contract_source_value(
                source, "mamba_acf_relation_representation", "encoder"
            )
        ).strip().lower()
        if relation_representation not in {"encoder", "forecast"}:
            raise ValueError(
                "mamba_acf_relation_representation must be encoder or forecast"
            )
        # The alternate view is consumed only by the ACF relation objective.
        # Canonicalize dormant requests so old plain forecasts and relation-off
        # controls do not acquire a meaningless replay distinction.
        payload["mamba_acf_relation_representation"] = (
            relation_representation
            if payload["mamba_acf_relation_loss"] else "encoder"
        )
        payload["mamba_acf_channel_order"] = str(
            _contract_source_value(source, "mamba_acf_channel_order", "input")
        ).strip().lower()
        if payload["mamba_acf_channel_order"] not in {
            "input", "strength", "similarity"
        }:
            raise ValueError(
                "mamba_acf_channel_order must be input, strength, or similarity"
            )
        payload["mamba_patch_order"] = str(
            _contract_source_value(source, "mamba_patch_order", "time_major")
        ).strip().lower()
        raw_local_width = _contract_source_value(source, "mamba_local_width", None)
        if payload["mamba_local_mixer"]:
            # The model resolves ``None`` to d_model, so compare effective
            # width rather than the raw spelling used by old configs.
            payload["mamba_local_width"] = (
                payload["d_model"] if raw_local_width is None else
                _contract_int(raw_local_width, "mamba_local_width", minimum=1)
            )
        else:
            if raw_local_width is not None:
                _contract_int(raw_local_width, "mamba_local_width", minimum=1)
            payload["mamba_local_width"] = None
    else:
        payload["AutoCon_multiscales"] = _contract_int_list(
            _contract_source_value(source, "AutoCon_multiscales", [96]),
            "AutoCon_multiscales", preserve_order=True,
        )
    if is_tm:
        payload["mamba_tm_n1"] = _contract_int(
            _contract_source_value(source, "mamba_tm_n1", 64),
            "mamba_tm_n1", minimum=1,
        )
        payload["mamba_tm_n2"] = _contract_int(
            _contract_source_value(source, "mamba_tm_n2", 128),
            "mamba_tm_n2", minimum=1,
        )
        payload["mamba_tm_repr_width"] = _contract_int(
            _contract_source_value(source, "mamba_tm_repr_width", 64),
            "mamba_tm_repr_width", minimum=1,
        )
        payload["mamba_tm_channel_independent"] = _contract_bool(
            _contract_source_value(source, "mamba_tm_channel_independent", False),
            "mamba_tm_channel_independent",
        )
        payload["mamba_tm_residual"] = _contract_bool(
            _contract_source_value(source, "mamba_tm_residual", True),
            "mamba_tm_residual",
        )
    return payload


def _mamba_acf_prior_required(source):
    """Return whether an ACF train prior is consumed by the live path.

    The implementation lives in ``utils.acf_relation_contract`` so replay
    preflight and experiment construction cannot drift apart.
    """

    return mamba_acf_prior_required(source)


def _preflight_mamba_acf_sidecar(metadata_dir, args, previous_config):
    """Check sidecar identity before constructing a MambaACF experiment."""

    if not _mamba_acf_prior_required(args):
        return None
    artifact_name = str(
        previous_config.get("mamba_acf_prior_artifact", "")
    ).strip()
    if artifact_name != "acf_prior.npz":
        raise RuntimeError(
            "MambaACF checkpoint evaluation requires acf_prior.npz sidecar "
            "for the enabled train-prior branch; refusing to refit it"
        )
    artifact_path = os.path.join(metadata_dir, artifact_name)
    if not os.path.isfile(artifact_path):
        raise RuntimeError(
            "MambaACF checkpoint evaluation requires an existing acf_prior.npz "
            "sidecar; refusing to refit it"
        )
    expected_sha = str(
        previous_config.get("mamba_acf_prior_sha256", "")
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise RuntimeError(
            "MambaACF checkpoint evaluation requires a valid 64-hex "
            "mamba_acf_prior_sha256"
        )
    digest = hashlib.sha256()
    with open(artifact_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            "existing MambaACF prior artifact hash mismatch: {}".format(
                artifact_path
            )
        )
    return artifact_path


def _contract_values_equal(left, right):
    """Compare normalized contract values, including finite float scalars."""

    if isinstance(left, float) or isinstance(right, float):
        try:
            return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return left == right


def _mamba_runtime_contract_mismatches(requested, persisted):
    mismatches = []
    for key in sorted(set(requested) | set(persisted)):
        if key not in requested or key not in persisted:
            mismatches.append(key)
        elif not _contract_values_equal(requested[key], persisted[key]):
            mismatches.append(key)
    return mismatches


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        value = float(np.mean(a)) if n else float('nan')
        return value, 0.0, 0.0
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, -h, +h


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {'1', 'true', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('expected a boolean value')


def _canonical_positive_lags(values, label='lags'):
    """Return a deterministic, validated lag list for metadata contracts."""

    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError('{} must be a sequence of positive integers'.format(label))
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError('{} must be a sequence of positive integers'.format(label)) from exc
    parsed = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('{} must contain positive integers'.format(label))
        if isinstance(value, (float, np.floating)) and (
            not np.isfinite(value) or not float(value).is_integer()
        ):
            raise ValueError('{} must contain positive integers'.format(label))
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError('{} must contain positive integers'.format(label)) from exc
        if integer <= 0 or integer != value:
            raise ValueError('{} must contain positive integers'.format(label))
        parsed.append(integer)
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError('{} must contain unique positive integers'.format(label))
    return sorted(parsed)


def _strict_positive_int(value, label):
    """Parse a positive integer without silently truncating decimal input."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError('{} must be a positive integer'.format(label))
    if isinstance(value, (float, np.floating)) and (
        not np.isfinite(value) or not value.is_integer()
    ):
        raise ValueError('{} must be a positive integer'.format(label))
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('{} must be a positive integer'.format(label)) from exc
    if not isinstance(value, str) and parsed != value:
        raise ValueError('{} must be a positive integer'.format(label))
    if parsed <= 0:
        raise ValueError('{} must be a positive integer'.format(label))
    return parsed


def _strict_nonnegative_int(value, label):
    """Parse a non-negative integer without bool coercion or truncation."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError('{} must be a non-negative integer'.format(label))
    if isinstance(value, (float, np.floating)) and (
        not np.isfinite(value) or not float(value).is_integer()
    ):
        raise ValueError('{} must be a non-negative integer'.format(label))
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('{} must be a non-negative integer'.format(label)) from exc
    if not isinstance(value, (str, bytes)) and parsed != value:
        raise ValueError('{} must be a non-negative integer'.format(label))
    if parsed < 0:
        raise ValueError('{} must be a non-negative integer'.format(label))
    return parsed


def _strict_bounded_float(value, label, *, minimum=None, maximum=None):
    """Parse a finite float and enforce inclusive bounds."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError('{} must be a finite number'.format(label))
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('{} must be a finite number'.format(label)) from exc
    if not np.isfinite(parsed):
        raise ValueError('{} must be finite'.format(label))
    if minimum is not None and parsed < minimum:
        if maximum is None:
            raise ValueError('{} must be at least {}'.format(label, minimum))
        raise ValueError('{} must be in [{}, {}]'.format(label, minimum, maximum))
    if maximum is not None and parsed > maximum:
        if minimum is None:
            raise ValueError('{} must be at most {}'.format(label, maximum))
        raise ValueError('{} must be in [{}, {}]'.format(label, minimum, maximum))
    return parsed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def persist_run_config(metadata_dir, args, run_seed, setting):
    """Write requested and resolved experiment settings for later auditing."""

    # ``args.seed`` is the base seed when ``--itr`` runs several replicas.
    # Store both values explicitly and make the conventional ``seed`` field
    # the actual replica seed so downstream readers cannot silently pair every
    # replica with the first seed.
    base_seed = int(getattr(args, 'seed', run_seed))
    resolved_seed = int(run_seed)
    ephemeral = {
        "mamba_acf_prior_artifact_path",
        "mamba_acf_prior_expected_sha256",
    }
    # Resolve this before the first config write.  The training loader is
    # constructed later inside ``Exp.train``; computing the same data-free
    # decision here keeps config.json and loss_logs.json auditable even when a
    # run stops before the first batch is consumed.
    relation_decision = resolve_acf_relation_batching(args)
    if bool(getattr(args, 'mamba_acf_relation_loss', False)):
        _record_acf_relation_batching(args, relation_decision)
    payload = {
        **{
            key: value for key, value in vars(args).items()
            if key not in ephemeral
        },
        'base_seed': base_seed,
        'seed': resolved_seed,
        'run_seed': resolved_seed,
        'setting': setting,
        'plan_job_id': args.plan_job_id,
    }
    runtime_contract = _mamba_runtime_contract_payload(args)
    if runtime_contract is not None:
        payload['mamba_runtime_contract_version'] = _MAMBA_RUNTIME_CONTRACT_VERSION
        payload['mamba_runtime_contract'] = runtime_contract
    with open(os.path.join(metadata_dir, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, default=str)


def persist_acf_prior_artifact(metadata_dir, args, experiment):
    """Persist a MambaACF train-only prior and bind it to run metadata.

    The forecast checkpoint contains learned weights, while the compact ACF
    table is a deterministic train-split input resolved by the experiment
    wrapper.  Keeping it as a sidecar makes standalone evaluation explicit and
    prevents a checkpoint from silently losing its seasonal-memory branch.
    """

    model = getattr(experiment, "model", None)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    saver = getattr(model, "save_acf_prior", None)
    exporter = getattr(model, "export_acf_prior_state", None)
    if not callable(saver) or not callable(exporter):
        return False

    state = exporter()
    if state is None:
        return False
    artifact_name = "acf_prior.npz"
    artifact_path = os.path.join(metadata_dir, artifact_name)
    if not saver(artifact_path) or not os.path.isfile(artifact_path):
        raise RuntimeError("MambaACF prior export did not create {}".format(artifact_path))
    digest = hashlib.sha256()
    with open(artifact_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    values = np.asarray(state["acf_values"])
    lags = np.asarray(state["lags"], dtype=np.int64).reshape(-1)
    _bind_acf_prior_metadata(
        args,
        artifact_name,
        digest.hexdigest(),
        values,
        lags,
        full_values=state.get("acf_full_values"),
    )
    _bind_acf_channel_lag_bank_metadata(args, values, lags, model)
    return True


def _bind_acf_prior_metadata(args, artifact_name, digest, values, lags, full_values=None):
    """Attach sidecar identity fields to an argparse namespace."""

    args.mamba_acf_prior_artifact = str(artifact_name)
    args.mamba_acf_prior_sha256 = str(digest)
    args.mamba_acf_prior_shape = [int(value) for value in np.asarray(values).shape]
    args.mamba_acf_prior_lags = _canonical_positive_lags(
        np.asarray(lags).reshape(-1).tolist(), 'sidecar lags'
    )
    if full_values is not None:
        args.mamba_acf_prior_full_shape = [
            int(value) for value in np.asarray(full_values).shape
        ]
    else:
        vars(args).pop("mamba_acf_prior_full_shape", None)


def _bind_acf_channel_lag_bank_metadata(args, values, lags, model=None):
    """Persist the deterministic forecast-support mask beside an ACF sidecar.

    The mask is not serialized separately because it is a pure function of
    the compact train-only prior, resolved lags, fixed forecast geometry, and
    top-k setting.  Recording its exact hash and inputs closes that replay
    contract without adding a second mutable artifact.
    """

    if not bool(getattr(args, "mamba_acf_channel_lag_bank", False)):
        return
    try:
        top_k = _strict_positive_int(
            getattr(args, "mamba_acf_channel_lag_top_k", 2),
            "mamba_acf_channel_lag_top_k",
        )
        seq_len = _strict_positive_int(
            getattr(args, "seq_len", 0), "seq_len"
        )
        pred_len = _strict_positive_int(
            getattr(args, "pred_len", 0), "pred_len"
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid channel lag-bank metadata inputs") from exc
    if top_k <= 0 or seq_len <= 0 or pred_len <= 0:
        raise RuntimeError("channel lag-bank top-k and forecast geometry must be positive")

    try:
        support, legal_fraction, canonical_lags = compile_channel_lag_support(
            values,
            lags,
            seq_len=seq_len,
            horizon=pred_len,
            top_k=top_k,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    compact = np.asarray(values, dtype=np.float32)
    if compact.ndim != 2 or compact.shape[1] != len(canonical_lags):
        raise RuntimeError("channel lag-bank compact prior does not align with sidecar lags")
    resolved_lags = list(canonical_lags)
    support_hash = hashlib.sha256(support.tobytes(order="C")).hexdigest()

    if model is not None:
        model_support = getattr(model, "acf_memory_static_channel_support", None)
        if torch.is_tensor(model_support) and model_support.numel():
            observed = model_support.detach().to("cpu", dtype=torch.uint8).numpy()
            if observed.shape != support.shape or not np.array_equal(observed, support):
                raise RuntimeError(
                    "MambaACF channel lag-bank mask does not match persisted sidecar metadata"
                )

    args.mamba_acf_channel_lag_bank_selection = (
        "abs_acf_times_legal_step_fraction"
    )
    args.mamba_acf_channel_lag_bank_resolved_lags = list(resolved_lags)
    args.mamba_acf_channel_lag_bank_legal_lags = [
        int(lag)
        for lag, fraction in zip(resolved_lags, legal_fraction.tolist())
        if fraction > 0.0
    ]
    args.mamba_acf_channel_lag_bank_legal_step_fractions = [
        float(fraction) for fraction in legal_fraction.tolist()
    ]
    args.mamba_acf_channel_lag_bank_support_shape = [
        int(support.shape[0]), int(support.shape[1])
    ]
    args.mamba_acf_channel_lag_bank_selected_slots = int(support.sum())
    args.mamba_acf_channel_lag_bank_total_slots = int(support.size)
    args.mamba_acf_channel_lag_bank_density = float(
        support.sum() / max(1, support.size)
    )
    args.mamba_acf_channel_lag_bank_support_sha256 = support_hash


def validate_persisted_acf_runtime_contract(args, previous_config):
    """Reject evaluation settings that would change a persisted Mamba path.

    The model is constructed before a prior sidecar can be loaded.  Therefore
    every option that can alter the Mamba forecast/representation function is
    compared against the original run config first; checking only the sidecar
    after construction would allow a checkpoint to be evaluated with a
    different function while still accepting the same state-dict.
    """

    if not isinstance(previous_config, dict):
        return
    requested_model = str(getattr(args, "model", "")).strip()
    persisted_model = str(previous_config.get("model", "")).strip()
    # A Mamba checkpoint cannot be replayed through a non-Mamba experiment
    # (or through a different Mamba variant).  Check the persisted identity
    # before the requested-model early return so a mistaken CLI model fails
    # before any experiment/model constructor runs.
    if (
        persisted_model in MAMBA_MODELS
        and requested_model != persisted_model
    ):
        raise RuntimeError(
            "evaluation model {} does not match the persisted Mamba model {}".format(
                requested_model or "<missing>", persisted_model
            )
        )
    if requested_model not in MAMBA_MODELS:
        return

    try:
        requested_contract = _mamba_runtime_contract_payload(args)
        persisted_contract = _mamba_runtime_contract_payload(previous_config)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "cannot validate persisted Mamba runtime contract: {}".format(exc)
        ) from exc
    if requested_contract is None or persisted_contract is None:
        raise RuntimeError("persisted Mamba config has no valid model identity")
    persisted_version = previous_config.get("mamba_runtime_contract_version")
    persisted_nested = previous_config.get("mamba_runtime_contract")
    if persisted_version is not None:
        try:
            version = _strict_positive_int(
                persisted_version, "mamba_runtime_contract_version"
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if version != _MAMBA_RUNTIME_CONTRACT_VERSION:
            raise RuntimeError(
                "unsupported persisted Mamba runtime contract version {}".format(
                    version
                )
            )
        if not isinstance(persisted_nested, dict):
            raise RuntimeError(
                "persisted Mamba runtime contract must be an object"
            )
        nested_mismatches = _mamba_runtime_contract_mismatches(
            persisted_contract, persisted_nested
        )
        if nested_mismatches:
            raise RuntimeError(
                "persisted Mamba runtime contract disagrees with config fields: {}".format(
                    ", ".join(nested_mismatches)
                )
            )
    mismatches = _mamba_runtime_contract_mismatches(
        requested_contract, persisted_contract
    )
    if mismatches:
        raise RuntimeError(
            "evaluation Mamba runtime contract does not match the persisted "
            "config: {}".format(", ".join(mismatches))
        )

    # Retain the explicit lag-bank wording used by existing callers/tests.
    persisted_raw = previous_config.get("mamba_acf_channel_lag_bank", False)
    if not requested_model.startswith("MambaACF"):
        return
    try:
        persisted_raw = _contract_bool(
            persisted_raw, "previous config mamba_acf_channel_lag_bank"
        )
        requested = _contract_bool(
            getattr(args, "mamba_acf_channel_lag_bank", False),
            "mamba_acf_channel_lag_bank",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if requested != persisted_raw:
        raise RuntimeError(
            "evaluation mamba_acf_channel_lag_bank={} does not match the "
            "persisted config value {}".format(requested, persisted_raw)
        )
    if not persisted_raw:
        return
    persisted_top_k = _strict_positive_int(
        previous_config.get("mamba_acf_channel_lag_top_k", 2),
        "previous config mamba_acf_channel_lag_top_k",
    )
    requested_top_k = _strict_positive_int(
        getattr(args, "mamba_acf_channel_lag_top_k", 2),
        "mamba_acf_channel_lag_top_k",
    )
    if requested_top_k != persisted_top_k:
        raise RuntimeError(
            "evaluation mamba_acf_channel_lag_top_k={} does not match the "
            "persisted config value {}".format(requested_top_k, persisted_top_k)
        )


def _mamba_acf_model_and_prior_path(metadata_dir, experiment):
    """Resolve a MambaACF model and its canonical evaluation sidecar path."""

    model = getattr(experiment, "model", None)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    artifact_name = "acf_prior.npz"
    artifact_path = os.path.join(metadata_dir, artifact_name)
    return model, artifact_name, artifact_path


def _bind_loaded_acf_prior_metadata(
    metadata_dir, args, experiment, expected_sha=None, actual_sha=None
):
    """Hash-check and bind metadata for a sidecar already mounted on a model.

    AutoCon Mamba experiments attach the sidecar while constructing the
    relation loss.  The outer replay path still needs the same provenance
    fields, but must not invoke ``model.load_acf_prior`` a second time.
    """

    model, artifact_name, artifact_path = _mamba_acf_model_and_prior_path(
        metadata_dir, experiment
    )
    exporter = getattr(model, "export_acf_prior_state", None)
    if not callable(exporter):
        return False
    if not os.path.isfile(artifact_path):
        return False
    if actual_sha is None:
        digest = hashlib.sha256()
        with open(artifact_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
    if expected_sha and str(expected_sha).strip().lower() != actual_sha:
        raise RuntimeError(
            "existing MambaACF prior artifact hash mismatch: {}".format(artifact_path)
        )
    state = exporter()
    if state is None:
        raise RuntimeError("loaded MambaACF prior artifact produced an empty state")
    _bind_acf_prior_metadata(
        args,
        artifact_name,
        actual_sha,
        state["acf_values"],
        state["lags"],
        full_values=state.get("acf_full_values"),
    )
    _bind_acf_channel_lag_bank_metadata(
        args, state["acf_values"], state["lags"], model
    )
    return True


def load_acf_prior_artifact(metadata_dir, args, experiment, expected_sha=None):
    """Load, hash-check, and bind an existing MambaACF prior sidecar."""

    model, artifact_name, artifact_path = _mamba_acf_model_and_prior_path(
        metadata_dir, experiment
    )
    loader = getattr(model, "load_acf_prior", None)
    exporter = getattr(model, "export_acf_prior_state", None)
    if not callable(loader) or not callable(exporter):
        return False
    if not os.path.isfile(artifact_path):
        return False
    digest = hashlib.sha256()
    with open(artifact_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if expected_sha and str(expected_sha).strip().lower() != actual_sha:
        raise RuntimeError(
            "existing MambaACF prior artifact hash mismatch: {}".format(artifact_path)
        )
    loader(artifact_path)
    if not _bind_loaded_acf_prior_metadata(
        metadata_dir, args, experiment, expected_sha=actual_sha,
        actual_sha=actual_sha,
    ):
        raise RuntimeError("loaded MambaACF prior artifact produced an empty state")
    return True


_ACF_PRIOR_METADATA_KEYS = (
    "mamba_acf_prior_artifact",
    "mamba_acf_prior_sha256",
    "mamba_acf_prior_shape",
    "mamba_acf_prior_lags",
    "mamba_acf_prior_full_shape",
    "mamba_acf_channel_lag_bank_selection",
    "mamba_acf_channel_lag_bank_resolved_lags",
    "mamba_acf_channel_lag_bank_legal_lags",
    "mamba_acf_channel_lag_bank_legal_step_fractions",
    "mamba_acf_channel_lag_bank_support_shape",
    "mamba_acf_channel_lag_bank_selected_slots",
    "mamba_acf_channel_lag_bank_total_slots",
    "mamba_acf_channel_lag_bank_density",
    "mamba_acf_channel_lag_bank_support_sha256",
)

_ACF_RESOLUTION_METADATA_KEYS = (
    "mamba_acf_train_peak_lags",
    "mamba_acf_relation_train_peak_lags",
    "mamba_acf_effective_lags",
    "mamba_acf_effective_lag_source",
    "mamba_acf_effective_lag_fallback",
    "mamba_acf_prior_max_lag",
    "mamba_acf_relation_effective_lags",
    "mamba_acf_relation_lag_source",
    "mamba_acf_relation_lag_fallback",
    "mamba_acf_relation_selected_lags",
)


def clear_acf_prior_metadata(args):
    """Remove resolved ACF metadata before a new run is persisted.

    ``argparse`` reuses the same namespace across ``--itr`` replicas and the
    evaluation replay path.  Both the sidecar identity and the lag-selection
    fields are derived outputs, so retaining either group can leak the prior
    replica's train-only statistics into a new ``config.json``.
    """

    for key in _ACF_PRIOR_METADATA_KEYS + _ACF_RESOLUTION_METADATA_KEYS:
        vars(args).pop(key, None)


def preserve_acf_prior_metadata(args, previous_config):
    """Carry a validated run's ACF metadata through an ACF-off evaluation.

    A plain ``MambaACF`` checkpoint does not need to load its train-only prior
    when all ACF branches are disabled.  The original training config still
    owns that sidecar identity, though, so an evaluation-only config rewrite
    must not erase it and make the phase appear unverifiable later.
    """

    if not isinstance(previous_config, dict):
        return
    for key in _ACF_PRIOR_METADATA_KEYS + _ACF_RESOLUTION_METADATA_KEYS:
        if key in previous_config:
            setattr(args, key, previous_config[key])


def prepare_eval_acf_prior_args(
    args,
    *,
    prior_required,
    preflight_artifact_path=None,
    expected_sha=None,
):
    """Prepare only the ACF inputs consumed by an evaluation model.

    A checkpoint may retain a historical ACF sidecar for provenance even when
    its live forecast and representation paths do not use it.  In that
    ACF-off case, an explicit or inherited construction path must not make
    evaluation depend on the sidecar again.
    """

    clear_acf_prior_metadata(args)
    # Construction-only paths may have been inherited from a caller or from
    # a previous ``--itr`` namespace.  They are valid only when they were
    # produced by this evaluation's sidecar preflight below.
    vars(args).pop("mamba_acf_prior_artifact_path", None)
    vars(args).pop("mamba_acf_prior_expected_sha256", None)
    if not prior_required:
        return
    if preflight_artifact_path is not None:
        args.mamba_acf_prior_artifact_path = preflight_artifact_path
        args.mamba_acf_prior_expected_sha256 = str(expected_sha or "")


def load_eval_acf_prior_if_required(
    metadata_dir,
    args,
    experiment,
    *,
    expected_sha=None,
    prior_required,
):
    """Load a persisted sidecar only for a live ACF-dependent eval branch."""

    if not prior_required:
        return False
    # AutoCon Mamba experiments load the sidecar during their base
    # initialization so the relation objective is built from the same full
    # train-only table.  Avoid loading the identical artifact a second time in
    # the outer evaluation path; bind its provenance from the already-mounted
    # model state instead.  The artifact hash is checked again against the
    # copied evaluation directory before metadata is written.
    if bool(getattr(experiment, "_acf_prior_attached", False)):
        return _bind_loaded_acf_prior_metadata(
            metadata_dir, args, experiment, expected_sha=expected_sha
        )
    return load_acf_prior_artifact(
        metadata_dir,
        args,
        experiment,
        expected_sha=expected_sha,
    )


def restore_acf_resolution_metadata(args, previous_config, *, strict=True):
    """Restore resolved train-only lag metadata after sidecar loading.

    The sidecar contains the coefficients and selected forecast lags; the
    config records the peak-selection provenance and relation diagnostics.  A
    resumed evaluation must retain both, while rejecting a sidecar/config pair
    that names different forecast lags.
    """

    if not isinstance(previous_config, dict):
        return
    actual_lags = [
        int(value) for value in getattr(args, "mamba_acf_prior_lags", [])
    ]
    try:
        actual_lags = _canonical_positive_lags(actual_lags, 'sidecar lags') or []
    except ValueError as exc:
        if strict:
            raise RuntimeError(str(exc)) from exc
        actual_lags = []
    expected_lags = previous_config.get("mamba_acf_effective_lags")
    if expected_lags is not None:
        try:
            expected_lags = _canonical_positive_lags(
                expected_lags, 'previous config effective lags'
            )
        except ValueError as exc:
            if strict:
                raise RuntimeError(str(exc)) from exc
            expected_lags = None
        if expected_lags is not None and actual_lags and expected_lags != actual_lags:
            raise RuntimeError(
                "MambaACF sidecar lags do not match the previous config"
            )
    try:
        persisted_bank = _contract_bool(
            previous_config.get("mamba_acf_channel_lag_bank", False),
            "previous config mamba_acf_channel_lag_bank",
        )
        current_bank = _contract_bool(
            getattr(args, "mamba_acf_channel_lag_bank", False),
            "mamba_acf_channel_lag_bank",
        )
    except ValueError as exc:
        if strict:
            raise RuntimeError(str(exc)) from exc
        persisted_bank = False
        current_bank = False
    if current_bank != persisted_bank:
        raise RuntimeError(
            "MambaACF channel lag-bank setting does not match the previous config"
        )
    if persisted_bank:
        bank_keys = (
            "mamba_acf_channel_lag_bank_selection",
            "mamba_acf_channel_lag_bank_resolved_lags",
            "mamba_acf_channel_lag_bank_legal_lags",
            "mamba_acf_channel_lag_bank_legal_step_fractions",
            "mamba_acf_channel_lag_bank_support_shape",
            "mamba_acf_channel_lag_bank_selected_slots",
            "mamba_acf_channel_lag_bank_total_slots",
            "mamba_acf_channel_lag_bank_density",
            "mamba_acf_channel_lag_bank_support_sha256",
        )
        for key in bank_keys:
            expected = previous_config.get(key)
            actual = getattr(args, key, None)
            if expected is None:
                continue
            if key == "mamba_acf_channel_lag_bank_legal_step_fractions":
                try:
                    same = np.allclose(
                        np.asarray(expected, dtype=float),
                        np.asarray(actual, dtype=float),
                        rtol=0.0,
                        atol=1e-12,
                    )
                except (TypeError, ValueError):
                    same = False
            elif key == "mamba_acf_channel_lag_bank_density":
                try:
                    same = bool(np.isclose(
                        float(expected), float(actual), rtol=0.0, atol=1e-12
                    ))
                except (TypeError, ValueError):
                    same = False
            else:
                same = expected == actual
            if not same:
                raise RuntimeError(
                    "MambaACF channel lag-bank metadata does not match the "
                    "previous config: {}".format(key)
                )
    for key in _ACF_RESOLUTION_METADATA_KEYS:
        if key in previous_config:
            value = previous_config[key]
            if isinstance(value, list):
                if key in {
                    "mamba_acf_train_peak_lags",
                    "mamba_acf_effective_lags",
                    "mamba_acf_relation_effective_lags",
                    "mamba_acf_relation_selected_lags",
                }:
                    try:
                        value = _canonical_positive_lags(value, key)
                    except ValueError as exc:
                        if strict:
                            raise RuntimeError(str(exc)) from exc
                else:
                    value = list(value)
            setattr(args, key, value)


def compact_setting(setting, limit=_SETTING_COMPONENT_LIMIT):
    """Keep a result/checkpoint directory within portable component limits.

    The full requested configuration is persisted in ``config.json``.  A
    bounded directory name therefore prevents long plan job identifiers and
    architecture tags from failing before training starts, while a suffix hash
    preserves a stable, collision-resistant mapping to the original setting.
    """

    setting = str(setting)
    limit = int(limit)
    if limit < 24:
        raise ValueError("setting component limit must be at least 24")
    if len(setting) <= limit:
        return setting
    digest = hashlib.sha256(setting.encode("utf-8")).hexdigest()[:16]
    prefix_length = limit - len(digest) - 2
    return "{}_h{}".format(setting[:prefix_length], digest)


def experiment_setting(args, iteration, run_seed):
    """Build the historical setting string, compacting only overlong names."""

    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, iteration)
    setting += '_seed{}'.format(run_seed)
    if args.model in MAMBA_MODELS:
        if (
            args.model.startswith('Mamba1')
            or args.model.startswith('MambaACF1')
            or args.model.startswith('MambaTM1')
        ):
            setting += '_m1s{}x{}_ex{}'.format(
                args.mamba_d_state, args.mamba_d_conv, args.mamba_expand)
        elif args.model.startswith('MambaTS'):
            setting += '_mts{}x{}_ex{}_p{}x{}'.format(
                args.mamba_d_state, args.mamba_d_conv, args.mamba_expand,
                args.patch_len, args.patch_stride)
        else:
            setting += '_ms{}x{}_st{}x{}'.format(
                args.mamba_d_state, args.mamba_d_conv,
                args.mamba_expand, args.mamba_headdim)
    if args.AutoCon:
        setting += '_ac{}_{}'.format(args.AutoCon_lambda, args.AutoCon_acf_source)
    setting += mamba_architecture_tag(args)
    return compact_setting(setting)


def mamba_architecture_tag(args):
    """Return a short collision-resistant tag for optional MambaACF knobs."""

    model_name = str(getattr(args, 'model', ''))
    if model_name.startswith('MambaTM'):
        payload = {
            name: getattr(args, name, None)
            for name in (
                'mamba_tm_n1', 'mamba_tm_n2', 'mamba_tm_repr_width',
                'mamba_tm_channel_independent', 'mamba_tm_residual',
                'mamba_d_state', 'mamba_d_conv', 'mamba_expand',
                'mamba_headdim', 'AutoCon_wnorm', 'AutoCon_lambda',
            )
        }
        encoded = json.dumps(
            payload, sort_keys=True, default=str, separators=(',', ':')
        ).encode()
        return '_tm' + hashlib.sha256(encoded).hexdigest()[:8]
    if model_name.startswith('MambaTS'):
        payload = {
            name: getattr(args, name, None)
            for name in (
                'mamba_ts_variant', 'mamba_ts_context_rank', 'mamba_ts_residual_init',
                'mamba_ts_mixer_type',
                'mamba_ts_spectral_top_k',
                'mamba_ts_delta_multipliers', 'mamba_ts_dt_min', 'mamba_ts_dt_max',
                'mamba_ts_repr_forecast_fusion',
                'mamba_ts_repr_forecast_channelwise',
                'mamba_ts_channel_adapter',
                'mamba_ts_channel_adapter_rank',
                'mamba_ts_revin_affine',
                'mamba_ts_global_channel_mixer',
                'mamba_ts_global_channel_mixer_rank',
                'mamba_ts_graph_spatial',
                'mamba_ts_graph_path',
                'mamba_ts_graph_expected_sha256',
                'mamba_ts_repr_linear_expert',
                'mamba_ts_repr_linear_expert_mode',
                'mamba_ts_repr_linear_blend_bias',
                'mamba_ts_repr_linear_expert_rank',
                'mamba_ts_variable_permutation', 'AutoCon_wnorm',
                'mamba_ts_acf_memory_selector',
                'mamba_ts_forecast_acf_consistency',
                'mamba_ts_forecast_acf_weight', 'mamba_ts_forecast_acf_top_k',
                'mamba_ts_forecast_acf_min_lag', 'mamba_ts_forecast_acf_min_abs',
                'AutoCon_lambda',
            )
        }
        encoded = json.dumps(
            payload, sort_keys=True, default=str, separators=(',', ':')
        ).encode()
        return '_ts' + hashlib.sha256(encoded).hexdigest()[:8]
    if model_name == 'Mamba2AutoCon':
        payload = {
            name: getattr(args, name, None)
            for name in (
                'mamba_d_state', 'mamba_d_conv', 'mamba_expand',
                'mamba_headdim', 'mamba2_repr_forecast_fusion',
                'AutoCon_wnorm', 'AutoCon_lambda',
            )
        }
        encoded = json.dumps(
            payload, sort_keys=True, default=str, separators=(',', ':')
        ).encode()
        return '_m2' + hashlib.sha256(encoded).hexdigest()[:8]
    if model_name.startswith('BiPatchMamba'):
        payload = {
            name: getattr(args, name, None)
            for name in (
                'mamba_bipatch_token_mode', 'mamba_bipatch_repr_gate',
                'mamba_d_state', 'mamba_d_conv', 'mamba_expand',
                'AutoCon_wnorm', 'AutoCon_lambda',
            )
        }
        encoded = json.dumps(
            payload, sort_keys=True, default=str, separators=(',', ':')
        ).encode()
        return '_bp' + hashlib.sha256(encoded).hexdigest()[:8]
    if model_name.startswith('NLinearVariableMamba'):
        payload = {
            name: getattr(args, name, None)
            for name in (
                'mamba_d_state', 'mamba_d_conv', 'mamba_expand',
                'mamba_ts_residual_init', 'mamba_variable_repr_fusion',
                'mamba_variable_acf_memory_selector',
                'mamba_use_acf_memory', 'AutoCon_wnorm', 'AutoCon_lambda',
                'AutoCon_acf_source', 'mamba_acf_relation_loss',
                'mamba_acf_relation_scope', 'mamba_acf_relation_lag_mask',
                'mamba_acf_relation_min_lag', 'mamba_acf_relation_target_mode',
                'mamba_acf_forecast_pair_consistency',
                'mamba_acf_forecast_pair_weight',
            )
        }
        encoded = json.dumps(
            payload, sort_keys=True, default=str, separators=(',', ':')
        ).encode()
        return '_nv' + hashlib.sha256(encoded).hexdigest()[:8]
    if not model_name.startswith('MambaACF'):
        return ''
    payload = {
        name: getattr(args, name, None)
        for name in (
            'mamba_scales', 'mamba_token_stride', 'mamba_bidirectional', 'mamba_channel_independent',
            'mamba_channel_gate', 'mamba_channel_gate_rank',
            'mamba_data_dependent_channel_mixer',
            'mamba_data_dependent_channel_mixer_rank',
            'mamba_variable_gdd_mixer', 'mamba_variable_gdd_mixer_rank',
            'mamba_variable_gdd_init_scale',
            'mamba_causal_embedding', 'mamba_low_rank_head', 'mamba_temporal_rank',
            'mamba_variable_mixer', 'mamba_variable_mixer_depth',
            'mamba_variable_group_size',
            'mamba_variable_repr_fusion',
            'mamba_variable_permutation',
            'mamba_acf_channel_order',
            'mamba_acf_channel_conditioning',
            'mamba_acf_scale_conditioning',
            'mamba_acf_local_agreement', 'mamba_acf_local_min_pairs',
            'mamba_acf_repr_fusion',
            'mamba_acf_channel_gate',
            'mamba_patch_mixer', 'mamba_patch_mixer_depth', 'mamba_patch_len',
            'mamba_patch_stride', 'mamba_patch_order', 'mamba_patch_permutation',
            'mamba_local_mixer', 'mamba_local_patch_len',
            'mamba_local_patch_stride', 'mamba_local_width',
            'mamba_local_rank', 'mamba_local_depth',
            'mamba_acf_lags', 'mamba_acf_lag_mode',
            'mamba_acf_max_lag', 'mamba_acf_top_k', 'mamba_acf_min_lag',
            'mamba_acf_min_abs', 'mamba_acf_confidence_z',
            'mamba_acf_min_separation', 'mamba_acf_detrend', 'mamba_acf_chunk',
            'mamba_use_acf_memory', 'mamba_dynamic_acf', 'mamba_acf_temperature',
            'mamba_acf_step_reliability',
            'mamba_acf_reliability_interpolation',
            'AutoCon_wnorm', 'AutoCon_acf_source', 'AutoCon_multiscales',
            'AutoCon_warmup_epochs', 'AutoCon_temperature',
            'AutoCon_relation_temperature', 'AutoCon_sign_weight',
            'AutoCon_sign_margin',
            'mamba_acf_relation_loss', 'mamba_acf_relation_batching',
            'mamba_acf_relation_lag_balance',
            'mamba_acf_relation_channel_reduce', 'mamba_acf_relation_scope',
            'mamba_acf_relation_representation',
            'mamba_acf_relation_lag_mask', 'mamba_acf_relation_lag_tolerance',
            'mamba_acf_relation_min_lag', 'mamba_acf_relation_target_mode',
            'mamba_acf_relation_soft_min_abs',
            'mamba_acf_relation_diagnostic_min_abs',
            'mamba_acf_relation_selected_lag_boost',
            'mamba_acf_relation_active_anchor_normalization',
            'mamba_acf_horizon_calibration',
            'mamba_acf_horizon_calibration_rank',
            'mamba_acf_memory_gate_mode',
            'mamba_acf_anchor_copy',
            'mamba_acf_channel_lag_bank', 'mamba_acf_channel_lag_top_k',
            'mamba_require_strict_representation_causality',
        )
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(',', ':')).encode()
    return '_a' + hashlib.sha256(encoded).hexdigest()[:8]


def validate_mamba_acf_arguments(args):
    """Reject inactive or structurally invalid base-token stride settings."""

    stride = _strict_positive_int(
        getattr(args, 'mamba_token_stride', 1), 'mamba_token_stride'
    )
    seq_len = _strict_positive_int(
        getattr(args, 'seq_len', 96), 'seq_len'
    )
    window_normalization = str(
        getattr(args, 'AutoCon_wnorm', 'LastVal')
    ).strip().lower()
    # All ReVIN implementations use PyTorch's unbiased sample standard
    # deviation.  A one-step window therefore produces NaN before model
    # construction; reject that configuration at the public CLI boundary.
    if window_normalization == 'revin' and seq_len < 2:
        raise ValueError(
            'AutoCon_wnorm=ReVIN requires seq_len >= 2 because it uses an '
            'unbiased standard deviation'
        )
    setting_override = getattr(args, 'setting_override', None)
    if setting_override is not None:
        if int(getattr(args, 'is_training', 1)) != 0:
            raise ValueError('setting_override is available only when is_training=0')
        setting_override = str(setting_override).strip()
        if not setting_override:
            raise ValueError('setting_override must be non-empty when provided')
        override_path = os.path.normpath(setting_override)
        if (
            override_path != setting_override
            or os.path.isabs(setting_override)
            or setting_override in {'.', '..'}
            or '/' in setting_override
            or '\\' in setting_override
        ):
            raise ValueError(
                'setting_override must be a relative single directory name'
            )
    relation_min_lag = _strict_nonnegative_int(
        getattr(args, 'mamba_acf_relation_min_lag', 0),
        'mamba_acf_relation_min_lag',
    )
    relation_lag_tolerance = _strict_nonnegative_int(
        getattr(args, 'mamba_acf_relation_lag_tolerance', 0),
        'mamba_acf_relation_lag_tolerance',
    )
    memory_gate_mode = str(
        getattr(args, 'mamba_acf_memory_gate_mode', 'hidden')
    ).lower()
    anchor_copy = bool(getattr(args, 'mamba_acf_anchor_copy', False))
    model_name = str(getattr(args, 'model', ''))
    relation_loss = bool(getattr(args, 'mamba_acf_relation_loss', False))
    relation_batching = str(
        getattr(args, 'mamba_acf_relation_batching', 'lag_pairs')
    ).strip().lower()
    relation_scope = str(
        getattr(args, 'mamba_acf_relation_scope', 'both')
    ).strip().lower()
    relation_representation = str(
        getattr(args, 'mamba_acf_relation_representation', 'encoder')
    ).strip().lower()
    acf_scale_conditioning = bool(
        getattr(args, 'mamba_acf_scale_conditioning', False)
    )
    acf_local_agreement = bool(
        getattr(args, 'mamba_acf_local_agreement', False)
    )
    acf_reliability_interpolation = bool(
        getattr(args, 'mamba_acf_reliability_interpolation', False)
    )
    channel_lag_bank = bool(
        getattr(args, 'mamba_acf_channel_lag_bank', False)
    )
    channel_lag_top_k = _strict_positive_int(
        getattr(args, 'mamba_acf_channel_lag_top_k', 2),
        'mamba_acf_channel_lag_top_k',
    )
    data_dependent_channel_mixer = bool(
        getattr(args, 'mamba_data_dependent_channel_mixer', False)
    )
    data_dependent_channel_mixer_rank = _strict_positive_int(
        getattr(args, 'mamba_data_dependent_channel_mixer_rank', 8),
        'mamba_data_dependent_channel_mixer_rank',
    )
    variable_gdd_mixer = bool(
        getattr(args, 'mamba_variable_gdd_mixer', False)
    )
    variable_gdd_mixer_rank = _strict_positive_int(
        getattr(args, 'mamba_variable_gdd_mixer_rank', 8),
        'mamba_variable_gdd_mixer_rank',
    )
    variable_gdd_init_scale = _strict_bounded_float(
        getattr(args, 'mamba_variable_gdd_init_scale', 0.0),
        'mamba_variable_gdd_init_scale', minimum=0, maximum=1,
    )
    variable_repr_fusion = bool(
        getattr(args, 'mamba_variable_repr_fusion', False)
    )
    variable_acf_memory_selector = bool(
        getattr(args, 'mamba_variable_acf_memory_selector', False)
    )
    acf_repr_fusion = bool(
        getattr(args, 'mamba_acf_repr_fusion', False)
    )
    ts_repr_forecast_fusion = bool(
        getattr(args, 'mamba_ts_repr_forecast_fusion', False)
    )
    mamba2_repr_forecast_fusion = bool(
        getattr(args, 'mamba2_repr_forecast_fusion', False)
    )
    ts_repr_forecast_channelwise = bool(
        getattr(args, 'mamba_ts_repr_forecast_channelwise', False)
    )
    ts_channel_agnostic_scan = bool(
        getattr(args, 'mamba_ts_channel_agnostic_scan', False)
    )
    ts_channel_adapter = bool(
        getattr(args, 'mamba_ts_channel_adapter', False)
    )
    ts_revin_affine = bool(
        getattr(args, 'mamba_ts_revin_affine', False)
    )
    ts_global_channel_mixer = bool(
        getattr(args, 'mamba_ts_global_channel_mixer', False)
    )
    ts_graph_spatial = bool(
        getattr(args, 'mamba_ts_graph_spatial', False)
    )
    ts_repr_linear_expert = bool(
        getattr(args, 'mamba_ts_repr_linear_expert', False)
    )
    ts_repr_linear_expert_mode = str(
        getattr(args, 'mamba_ts_repr_linear_expert_mode', 'residual')
    ).strip().lower()
    ts_repr_linear_blend_bias = _strict_bounded_float(
        getattr(args, 'mamba_ts_repr_linear_blend_bias', -4.0),
        'mamba_ts_repr_linear_blend_bias', minimum=-12.0, maximum=0.0,
    )
    ts_variable_permutation = bool(
        getattr(args, 'mamba_ts_variable_permutation', False)
    )
    ts_forecast_acf_consistency = bool(
        getattr(args, 'mamba_ts_forecast_acf_consistency', False)
    )
    forecast_lag_pair_consistency = bool(
        getattr(args, 'mamba_acf_forecast_pair_consistency', False)
    )
    forecast_lag_pair_weight = _strict_bounded_float(
        getattr(args, 'mamba_acf_forecast_pair_weight', 0.0),
        'mamba_acf_forecast_pair_weight', minimum=0, maximum=1,
    )
    ts_acf_memory_selector = bool(
        getattr(args, 'mamba_ts_acf_memory_selector', False)
    )
    ts_acf_memory_channel_selector = bool(
        getattr(args, 'mamba_ts_acf_memory_channel_selector', False)
    )

    # The shared Mamba-1/2 head emits one seasonal channel per input channel.
    # ``features=MS`` is the historical exception: the trainer intentionally
    # slices the final channel after the model returns.  Reject every other
    # mismatched public shape before construction instead of allowing silent
    # broadcasting in the trend head.
    if model_name in {
        'Mamba1', 'Mamba1AutoCon', 'Mamba2', 'Mamba2AutoCon',
        'MambaTS', 'MambaTSAutoCon', 'MambaTSv2', 'MambaTSv2AutoCon',
        'MambaTSv3', 'MambaTSv3AutoCon', 'MambaTSv4', 'MambaTSv4AutoCon',
        'MambaTSv5', 'MambaTSv5AutoCon', 'MambaTSv6', 'MambaTSv6AutoCon',
        'MambaTSv7', 'MambaTSv7AutoCon',
        'MambaTSv8', 'MambaTSv8AutoCon', 'MambaTSv9', 'MambaTSv9AutoCon',
        'MambaTSv10', 'MambaTSv10AutoCon', 'MambaTSv11', 'MambaTSv11AutoCon',
        'MambaTSv12', 'MambaTSv12AutoCon', 'MambaTSv13', 'MambaTSv13AutoCon',
        'MambaTSv14', 'MambaTSv14AutoCon', 'MambaTSv15', 'MambaTSv15AutoCon',
        'MambaTSv16', 'MambaTSv16AutoCon',
        'MambaTM1', 'MambaTM1AutoCon', 'MambaTM2', 'MambaTM2AutoCon',
        'MambaSM', 'MambaSMAutoCon',
        'RaMambaLite', 'RaMambaLiteAutoCon',
        'BiPatchMamba', 'BiPatchMambaAutoCon',
        'InvertedRaMamba', 'InvertedRaMambaAutoCon',
        'LinearLatentMamba', 'LinearLatentMambaAutoCon',
        'NLinearVariableMamba', 'NLinearVariableMambaAutoCon',
    }:
        features = str(getattr(args, 'features', '') or '').strip().upper()
        raw_enc_in = getattr(args, 'enc_in', None)
        raw_c_out = getattr(args, 'c_out', None)
        if features and raw_enc_in is not None and raw_c_out is not None:
            enc_in = _strict_positive_int(raw_enc_in, 'enc_in')
            c_out = _strict_positive_int(raw_c_out, 'c_out')
            if features == 'MS':
                if c_out != 1:
                    raise ValueError(
                    'Mamba1/Mamba2/MambaTS/MambaTM/MambaSM features=MS requires c_out=1; '
                        'the trainer selects the final input channel'
                    )
            elif features in {'S', 'M'} and c_out != enc_in:
                raise ValueError(
                    'Mamba1/Mamba2/MambaTS/MambaTM/MambaSM features={} requires c_out=enc_in '
                    '(got c_out={}, enc_in={})'.format(features, c_out, enc_in)
                )
    if model_name.startswith('MambaTM'):
        for name in ('mamba_tm_n1', 'mamba_tm_n2', 'mamba_tm_repr_width'):
            _strict_positive_int(getattr(args, name, None), name)
    relation_models = {
        'MambaACFAutoCon', 'MambaACFLiteAutoCon', 'MambaACF1AutoCon',
        'NLinearVariableMambaAutoCon',
    }
    if relation_loss and model_name not in relation_models:
        raise ValueError(
            'mamba_acf_relation_loss requires --model MambaACFAutoCon, '
            'MambaACFLiteAutoCon, MambaACF1AutoCon, or '
            'NLinearVariableMambaAutoCon together with --AutoCon'
        )
    if model_name in relation_models:
        if not bool(getattr(args, 'AutoCon', False)):
            raise ValueError('{} requires --AutoCon'.format(model_name))
        if not relation_loss:
            raise ValueError(
                '{} requires --mamba_acf_relation_loss; use MambaACF or '
                'MambaACF1 for a plain forecast'.format(model_name)
            )
    if (
        str(getattr(args, 'model', '')).startswith('MambaACF')
        and str(getattr(args, 'AutoCon_wnorm', 'LastVal')).strip().lower() == 'decomp'
    ):
        raise ValueError(
            'MambaACF does not support AutoCon_wnorm=Decomp; use '
            'MeanStd, ReVIN, Mean, or LastVal'
        )
    if memory_gate_mode not in {'hidden', 'variable_token'}:
        raise ValueError(
            'mamba_acf_memory_gate_mode must be hidden or variable_token'
        )
    if relation_min_lag < 0:
        raise ValueError('mamba_acf_relation_min_lag must be non-negative')
    if relation_scope not in {'both', 'global', 'local'}:
        raise ValueError(
            'mamba_acf_relation_scope must be both, global, or local'
        )
    if relation_representation not in {'encoder', 'forecast'}:
        raise ValueError(
            'mamba_acf_relation_representation must be encoder or forecast'
        )
    if relation_loss and relation_batching == 'lag_pairs':
        batch_size = _strict_positive_int(
            getattr(args, 'batch_size', 32), 'batch_size'
        )
        if batch_size < 2:
            raise ValueError(
                'batch_size must be at least 2 for lag-pair relation batching'
            )
        if batch_size % 2:
            raise ValueError(
                'batch_size must be even for lag-pair relation batching'
            )
    if stride <= 0:
        raise ValueError('mamba_token_stride must be positive')
    if stride > seq_len:
        raise ValueError('mamba_token_stride cannot exceed seq_len={}'.format(seq_len))
    if (
        relation_loss
        and model_name in relation_models
        and stride > 1
        and relation_scope in {'both', 'local'}
    ):
        raise ValueError(
            'mamba_token_stride={} is incompatible with '
            'mamba_acf_relation_scope={} when MambaACF AutoCon relation loss '
            'is enabled; use mamba_token_stride=1 or '
            'mamba_acf_relation_scope=global'.format(stride, relation_scope)
        )
    if not str(getattr(args, 'model', '')).startswith('MambaACF'):
        if relation_representation != 'encoder':
            raise ValueError(
                'mamba_acf_relation_representation is supported only by '
                'MambaACF AutoCon models'
            )
        if stride != 1:
            raise ValueError('mamba_token_stride is supported only by MambaACF models')
        if relation_min_lag != 0 and model_name not in relation_models:
            raise ValueError(
                'mamba_acf_relation_min_lag requires a relation-capable AutoCon model'
            )
        if bool(getattr(args, 'mamba_acf_horizon_calibration', False)):
            raise ValueError(
                'mamba_acf_horizon_calibration is supported only by MambaACF models'
            )
        if memory_gate_mode != 'hidden':
            raise ValueError(
                'mamba_acf_memory_gate_mode is supported only by MambaACF models'
            )
        if anchor_copy:
            raise ValueError(
                'mamba_acf_anchor_copy is supported only by MambaACF models'
            )
        if acf_scale_conditioning:
            raise ValueError(
                'mamba_acf_scale_conditioning is supported only by MambaACF models'
            )
        if acf_local_agreement:
            raise ValueError(
                'mamba_acf_local_agreement is supported only by MambaACF models'
            )
        if acf_reliability_interpolation:
            raise ValueError(
                'mamba_acf_reliability_interpolation is supported only by '
                'MambaACF models'
            )
        if bool(getattr(args, 'mamba_require_strict_representation_causality', False)):
            raise ValueError(
                'mamba_require_strict_representation_causality is supported '
                'only by MambaACF models'
            )
        if channel_lag_bank:
            raise ValueError(
                'mamba_acf_channel_lag_bank is supported only by MambaACF models'
            )
        channel_order = str(
            getattr(args, 'mamba_acf_channel_order', 'input')
        ).strip().lower()
        if channel_order not in {'input', 'strength', 'similarity'}:
            raise ValueError(
                'mamba_acf_channel_order must be input, strength, or similarity'
            )
        if channel_order != 'input' and model_name not in relation_models:
            raise ValueError(
                'mamba_acf_channel_order requires a relation-capable AutoCon model'
            )
        if data_dependent_channel_mixer:
            raise ValueError(
                'mamba_data_dependent_channel_mixer is supported only by '
                'MambaACF models'
            )
        if variable_gdd_mixer:
            raise ValueError(
                'mamba_variable_gdd_mixer is supported only by MambaACF models'
            )
        if variable_repr_fusion:
            if model_name not in {
                'MambaACFAutoCon', 'MambaACFLiteAutoCon',
                'MambaACF1AutoCon', 'NLinearVariableMambaAutoCon',
                }:
                    raise ValueError(
                        'mamba_variable_repr_fusion is supported only by MambaACF '
                        'AutoCon models; NLinearVariableMambaAutoCon is also '
                        'supported as a variable-token AutoCon model'
                    )
            if not bool(getattr(args, 'AutoCon', False)):
                raise ValueError('mamba_variable_repr_fusion requires --AutoCon')
        if acf_repr_fusion:
            if model_name not in {'MambaTSv4AutoCon', 'MambaTSv5AutoCon', 'MambaTSv7AutoCon'}:
                raise ValueError(
                    'mamba_acf_repr_fusion is supported only by MambaACF AutoCon '
                    'or MambaTSv4/v5/v7 AutoCon models'
                )
            if not bool(getattr(args, 'mamba_use_acf_memory', False)):
                raise ValueError(
                    'MambaTS v4/v5/v7 ACF representation fusion requires '
                    'mamba_use_acf_memory=true'
                )
            if resolve_acf_source(args) != 'raw':
                raise ValueError(
                    'MambaTS v4/v5/v7 ACF representation fusion requires '
                    'AutoCon_acf_source=raw'
                )
    if ts_repr_forecast_fusion:
            if model_name not in {
                'MambaTSv3AutoCon', 'MambaTSv4AutoCon', 'MambaTSv5AutoCon', 'MambaTSv6AutoCon', 'MambaTSv7AutoCon', 'MambaTSv10AutoCon', 'MambaTSv11AutoCon', 'MambaTSv12AutoCon', 'MambaTSv13AutoCon', 'MambaTSv14AutoCon', 'MambaTSv15AutoCon', 'MambaTSv16AutoCon'
            }:
                raise ValueError(
                    'mamba_ts_repr_forecast_fusion requires a supported MambaTS '
                    'AutoCon model'
                )
            if not bool(getattr(args, 'AutoCon', False)):
                raise ValueError(
                    'mamba_ts_repr_forecast_fusion requires --AutoCon'
                )
    if ts_channel_agnostic_scan and model_name not in {
            'MambaTSv3', 'MambaTSv3AutoCon'}:
        raise ValueError(
            'mamba_ts_channel_agnostic_scan requires MambaTSv3 or MambaTSv3AutoCon'
        )
    if mamba2_repr_forecast_fusion:
        if model_name != 'Mamba2AutoCon':
            raise ValueError(
                'mamba2_repr_forecast_fusion requires --model Mamba2AutoCon'
            )
        if not bool(getattr(args, 'AutoCon', False)):
            raise ValueError('mamba2_repr_forecast_fusion requires --AutoCon')
        if ts_repr_forecast_channelwise and not ts_repr_forecast_fusion:
            raise ValueError(
                'mamba_ts_repr_forecast_channelwise requires '
                'mamba_ts_repr_forecast_fusion=true'
            )
        if ts_channel_adapter:
            if model_name != 'MambaTSv3AutoCon':
                raise ValueError(
                    'mamba_ts_channel_adapter requires MambaTSv3AutoCon'
                )
            if not bool(getattr(args, 'AutoCon', False)):
                raise ValueError('mamba_ts_channel_adapter requires --AutoCon')
            _strict_positive_int(
                getattr(args, 'mamba_ts_channel_adapter_rank', 8),
                'mamba_ts_channel_adapter_rank',
            )
        if ts_revin_affine:
            if model_name != 'MambaTSv3AutoCon':
                raise ValueError(
                    'mamba_ts_revin_affine requires MambaTSv3AutoCon'
                )
            if not bool(getattr(args, 'AutoCon', False)):
                raise ValueError('mamba_ts_revin_affine requires --AutoCon')
            if str(getattr(args, 'AutoCon_wnorm', 'LastVal')).strip().lower() not in {
                'meanstd', 'revin'
            }:
                raise ValueError(
                    'mamba_ts_revin_affine requires AutoCon_wnorm=MeanStd or ReVIN'
                )
        if ts_global_channel_mixer:
            if model_name != 'MambaTSv3AutoCon':
                raise ValueError(
                    'mamba_ts_global_channel_mixer requires MambaTSv3AutoCon'
                )
            if not bool(getattr(args, 'AutoCon', False)):
                raise ValueError(
                    'mamba_ts_global_channel_mixer requires --AutoCon'
                )
            _strict_positive_int(
                getattr(args, 'mamba_ts_global_channel_mixer_rank', 4),
                'mamba_ts_global_channel_mixer_rank',
            )
        if ts_repr_linear_expert:
            if model_name not in {
                'MambaTSv3AutoCon', 'MambaTSv4AutoCon', 'MambaTSv5AutoCon', 'MambaTSv6AutoCon', 'MambaTSv7AutoCon'
            }:
                raise ValueError(
                    'mamba_ts_repr_linear_expert requires a MambaTS v3/v4/v5 '
                    'AutoCon model'
                )
            if not bool(getattr(args, 'AutoCon', False)):
                raise ValueError('mamba_ts_repr_linear_expert requires --AutoCon')
        if ts_repr_linear_expert_mode not in {'residual', 'blend', 'nlinear_blend'}:
            raise ValueError(
                'mamba_ts_repr_linear_expert_mode must be residual, blend, or nlinear_blend'
            )
        if ts_repr_linear_expert_mode == 'blend' and not ts_repr_linear_expert:
            raise ValueError(
                'mamba_ts_repr_linear_expert_mode=blend requires '
                '--mamba_ts_repr_linear_expert'
            )
        _strict_positive_int(
            getattr(args, 'mamba_ts_repr_linear_expert_rank', 16),
            'mamba_ts_repr_linear_expert_rank',
        )
    if ts_variable_permutation and model_name not in {
            'MambaTSv3', 'MambaTSv3AutoCon', 'MambaTSv4', 'MambaTSv4AutoCon',
            'MambaTSv5', 'MambaTSv5AutoCon', 'MambaTSv6', 'MambaTSv6AutoCon',
            'MambaTSv7', 'MambaTSv7AutoCon',
        }:
            raise ValueError(
            'mamba_ts_variable_permutation requires a MambaTS v3/v4/v5 model'
        )
    if ts_global_channel_mixer and not mamba2_repr_forecast_fusion:
        if model_name != 'MambaTSv3AutoCon':
            raise ValueError(
                'mamba_ts_global_channel_mixer requires MambaTSv3AutoCon'
            )
        if not bool(getattr(args, 'AutoCon', False)):
            raise ValueError('mamba_ts_global_channel_mixer requires --AutoCon')
        _strict_positive_int(
            getattr(args, 'mamba_ts_global_channel_mixer_rank', 4),
            'mamba_ts_global_channel_mixer_rank',
        )
    if ts_graph_spatial:
        if model_name not in {'MambaTSv3AutoCon', 'GraphDLinear', 'STGCN', 'GraphWaveNet', 'DCRNN'}:
            raise ValueError(
                'mamba_ts_graph_spatial requires MambaTSv3AutoCon, GraphDLinear, STGCN, GraphWaveNet, or DCRNN'
            )
        if model_name == 'MambaTSv3AutoCon' and not bool(getattr(args, 'AutoCon', False)):
            raise ValueError('mamba_ts_graph_spatial requires --AutoCon')
        graph_path = str(getattr(args, 'mamba_ts_graph_path', '') or '').strip()
        if not graph_path:
            raise ValueError(
                'mamba_ts_graph_spatial requires --mamba_ts_graph_path'
            )
        expected_hash = str(
            getattr(args, 'mamba_ts_graph_expected_sha256', '') or ''
        ).strip().lower()
        if expected_hash and (
            len(expected_hash) != 64
            or any(character not in '0123456789abcdef' for character in expected_hash)
        ):
            raise ValueError(
                'mamba_ts_graph_expected_sha256 must be a 64-character SHA-256'
            )
    if ts_forecast_acf_consistency:
        if model_name not in {
            'MambaTSv3AutoCon', 'MambaTSv4AutoCon', 'MambaTSv5AutoCon', 'MambaTSv6AutoCon', 'MambaTSv7AutoCon',
            'NLinearVariableMambaAutoCon',
        }:
            raise ValueError(
                'mamba_ts_forecast_acf_consistency requires a supported AutoCon model'
            )
        if not bool(getattr(args, 'AutoCon', False)):
            raise ValueError('mamba_ts_forecast_acf_consistency requires --AutoCon')
        weight = _strict_bounded_float(
            getattr(args, 'mamba_ts_forecast_acf_weight', 0.0),
            'mamba_ts_forecast_acf_weight', minimum=0, maximum=1,
        )
        if weight <= 0:
            raise ValueError('mamba_ts_forecast_acf_consistency requires a positive weight')
        if int(getattr(args, 'mamba_ts_forecast_acf_top_k', 4)) < 1:
            raise ValueError('mamba_ts_forecast_acf_top_k must be positive')
        min_lag = int(getattr(args, 'mamba_ts_forecast_acf_min_lag', 2))
        if min_lag < 0 or min_lag >= int(getattr(args, 'pred_len', 0)):
            raise ValueError('mamba_ts_forecast_acf_min_lag must be in [0, pred_len)')
        min_abs = _strict_bounded_float(
            getattr(args, 'mamba_ts_forecast_acf_min_abs', 0.10),
            'mamba_ts_forecast_acf_min_abs', minimum=0, maximum=1,
        )
        args.mamba_ts_forecast_acf_weight = weight
        args.mamba_ts_forecast_acf_min_abs = min_abs
    if forecast_lag_pair_consistency:
        if model_name != 'NLinearVariableMambaAutoCon':
            raise ValueError(
                'mamba_acf_forecast_pair_consistency requires '
                '--model NLinearVariableMambaAutoCon'
            )
        if not bool(getattr(args, 'AutoCon', False)) or not relation_loss:
            raise ValueError(
                'mamba_acf_forecast_pair_consistency requires --AutoCon and '
                '--mamba_acf_relation_loss'
            )
        if relation_batching != 'lag_pairs':
            raise ValueError(
                'mamba_acf_forecast_pair_consistency requires '
                'mamba_acf_relation_batching=lag_pairs'
            )
        if relation_scope not in {'both', 'global'}:
            raise ValueError(
                'mamba_acf_forecast_pair_consistency requires a global '
                'relation scope'
            )
        if not bool(getattr(args, 'mamba_acf_relation_lag_mask', False)):
            raise ValueError(
                'mamba_acf_forecast_pair_consistency requires '
                'mamba_acf_relation_lag_mask=true'
            )
        if str(getattr(args, 'mamba_acf_relation_target_mode', '')).lower() != 'hard_selected':
            raise ValueError(
                'mamba_acf_forecast_pair_consistency requires '
                'mamba_acf_relation_target_mode=hard_selected'
            )
        if resolve_acf_source(args) != 'raw':
            raise ValueError(
                'mamba_acf_forecast_pair_consistency requires '
                'AutoCon_acf_source=raw'
            )
        if forecast_lag_pair_weight <= 0:
            raise ValueError(
                'mamba_acf_forecast_pair_consistency requires a positive weight'
            )
        args.mamba_acf_forecast_pair_weight = forecast_lag_pair_weight
    elif forecast_lag_pair_weight != 0:
        raise ValueError(
            'mamba_acf_forecast_pair_weight requires '
            '--mamba_acf_forecast_pair_consistency'
        )
    if ts_acf_memory_selector:
        if model_name not in {'MambaTSv4AutoCon', 'MambaTSv5AutoCon', 'MambaTSv7AutoCon'}:
            raise ValueError(
                'mamba_ts_acf_memory_selector requires a MambaTS v4/v5/v7 AutoCon model'
            )
        if not bool(getattr(args, 'mamba_use_acf_memory', False)):
            raise ValueError('mamba_ts_acf_memory_selector requires mamba_use_acf_memory=true')
        if not bool(getattr(args, 'AutoCon', False)):
            raise ValueError('mamba_ts_acf_memory_selector requires --AutoCon')
    if ts_acf_memory_channel_selector and not ts_acf_memory_selector:
        raise ValueError(
            'mamba_ts_acf_memory_channel_selector requires '
            'mamba_ts_acf_memory_selector=true'
        )
        if model_name.startswith('MambaTSv5'):
            if not bool(getattr(args, 'mamba_use_acf_memory', False)):
                raise ValueError('MambaTSv5 requires mamba_use_acf_memory=true')
            if resolve_acf_source(args) != 'raw':
                raise ValueError('MambaTSv5 requires AutoCon_acf_source=raw')
        return

    if relation_representation == 'forecast':
        if model_name not in relation_models or not relation_loss:
            raise ValueError(
                'mamba_acf_relation_representation=forecast requires '
                '--model MambaACFAutoCon, MambaACFLiteAutoCon, or '
                'MambaACF1AutoCon together with '
                '--AutoCon and --mamba_acf_relation_loss'
            )
        if relation_scope != 'global':
            raise ValueError(
                'mamba_acf_relation_representation=forecast requires '
                'mamba_acf_relation_scope=global'
            )
        if resolve_acf_source(args) != 'raw':
            raise ValueError(
                'mamba_acf_relation_representation=forecast requires '
                'AutoCon_acf_source=raw'
            )
        if str(getattr(args, 'features', '')).strip().upper() != 'M':
            raise ValueError(
                'mamba_acf_relation_representation=forecast requires features=M'
            )
        enc_in = _strict_positive_int(getattr(args, 'enc_in', None), 'enc_in')
        c_out = _strict_positive_int(getattr(args, 'c_out', None), 'c_out')
        if c_out != enc_in:
            raise ValueError(
                'mamba_acf_relation_representation=forecast requires c_out=enc_in'
            )
        if enc_in < 2:
            raise ValueError(
                'mamba_acf_relation_representation=forecast requires enc_in>=2'
            )
        if bool(getattr(args, 'mamba_acf_repr_fusion', False)) or bool(
            getattr(args, 'mamba_variable_repr_fusion', False)
        ):
            raise ValueError(
                'mamba_acf_relation_representation=forecast is incompatible '
                'with encoder-only representation fusion options'
            )

    if data_dependent_channel_mixer and bool(
        getattr(args, 'mamba_channel_independent', False)
    ):
        raise ValueError(
            'mamba_data_dependent_channel_mixer is a shared-mode '
            'cross-variable module and cannot be combined with '
            'mamba_channel_independent'
        )
    if variable_gdd_mixer:
        if bool(getattr(args, 'mamba_channel_independent', False)):
            raise ValueError(
                'mamba_variable_gdd_mixer is a shared-mode channel-preserving '
                'module and cannot be combined with mamba_channel_independent'
            )
        if not bool(getattr(args, 'mamba_variable_mixer', False)):
            raise ValueError(
                'mamba_variable_gdd_mixer requires --mamba_variable_mixer'
            )
    if variable_repr_fusion:
        if not bool(getattr(args, 'AutoCon', False)):
            raise ValueError(
                'mamba_variable_repr_fusion requires --AutoCon'
            )
        is_nlinear_variable_model = model_name == 'NLinearVariableMambaAutoCon'
        if not is_nlinear_variable_model and not bool(
            getattr(args, 'mamba_variable_mixer', False)
        ):
            raise ValueError(
                'mamba_variable_repr_fusion requires '
                '--mamba_variable_mixer'
            )
        if model_name not in relation_models:
            raise ValueError(
                'mamba_variable_repr_fusion is supported only by MambaACF '
                'AutoCon models or --model '
                'MambaACFAutoCon, MambaACFLiteAutoCon, MambaACF1AutoCon, '
                'or NLinearVariableMambaAutoCon'
            )
    if variable_acf_memory_selector:
        if model_name != 'NLinearVariableMambaAutoCon':
            raise ValueError(
                'mamba_variable_acf_memory_selector requires '
                '--model NLinearVariableMambaAutoCon'
            )
        if not bool(getattr(args, 'AutoCon', False)):
            raise ValueError(
                'mamba_variable_acf_memory_selector requires --AutoCon'
            )
        if not bool(getattr(args, 'mamba_use_acf_memory', False)):
            raise ValueError(
                'mamba_variable_acf_memory_selector requires '
                'mamba_use_acf_memory=true'
            )
        if int(getattr(args, 'mamba_acf_relation_min_lag', 0)) >= int(
            getattr(args, 'seq_len', 0)
        ):
            raise ValueError(
                'mamba_variable_acf_memory_selector requires '
                'mamba_acf_relation_min_lag < seq_len so the fitted ACF memory '
                'contains at least one legal observed lag'
            )
    if acf_repr_fusion:
        if not bool(getattr(args, 'AutoCon', False)):
            raise ValueError(
                'mamba_acf_repr_fusion requires --AutoCon'
            )
        if model_name not in relation_models:
            raise ValueError(
                'mamba_acf_repr_fusion requires --model '
                'MambaACFAutoCon, MambaACFLiteAutoCon, or MambaACF1AutoCon'
            )

    # Every MambaACF training run fits a train-only prior, including
    # plain/memory ablations.  Validate the fitter's complete numeric contract here so a
    # direct Namespace invocation cannot defer a malformed value to a later
    # ``int(...)``/``float(...)`` conversion inside the experiment wrapper.
    max_lag = _strict_positive_int(
        getattr(args, 'mamba_acf_max_lag', 720), 'mamba_acf_max_lag'
    )
    acf_top_k = _strict_positive_int(
        getattr(args, 'mamba_acf_top_k', 8), 'mamba_acf_top_k'
    )
    acf_min_lag = _strict_positive_int(
        getattr(args, 'mamba_acf_min_lag', 2), 'mamba_acf_min_lag'
    )
    acf_min_separation = _strict_positive_int(
        getattr(args, 'mamba_acf_min_separation', 2),
        'mamba_acf_min_separation',
    )
    acf_chunk = _strict_positive_int(
        getattr(args, 'mamba_acf_chunk', 64), 'mamba_acf_chunk'
    )
    acf_min_abs = _strict_bounded_float(
        getattr(args, 'mamba_acf_min_abs', 0.1),
        'mamba_acf_min_abs', minimum=0.0, maximum=1.0,
    )
    acf_confidence_z = _strict_bounded_float(
        getattr(args, 'mamba_acf_confidence_z', 1.96),
        'mamba_acf_confidence_z', minimum=0.0,
    )
    if acf_min_lag > max_lag:
        raise ValueError('mamba_acf_min_lag cannot exceed mamba_acf_max_lag')

    if anchor_copy:
        source = resolve_acf_source(args)
        if source != 'raw':
            raise ValueError(
                'mamba_acf_anchor_copy requires AutoCon_acf_source=raw'
            )
        if bool(getattr(args, 'mamba_dynamic_acf', False)):
            raise ValueError(
                'mamba_acf_anchor_copy requires static ACF memory '
                '(mamba_dynamic_acf=false)'
            )
        lags = getattr(args, 'mamba_acf_lags', None)
        if isinstance(lags, (str, bytes)):
            lag_values = [item for item in str(lags).replace(',', ' ').split() if item]
        else:
            try:
                lag_values = [] if lags is None else list(lags)
            except TypeError:
                # Keep malformed direct invocations on the same actionable
                # validation path as an omitted lag list instead of leaking a
                # low-level ``TypeError`` from ``list(int)``.
                lag_values = []
        if not lag_values:
            raise ValueError(
                'mamba_acf_anchor_copy requires non-empty mamba_acf_lags'
            )

    if memory_gate_mode == 'variable_token':
        if not bool(getattr(args, 'mamba_variable_mixer', False)):
            raise ValueError(
                'mamba_acf_memory_gate_mode=variable_token requires '
                'mamba_variable_mixer=true'
            )
        if not bool(getattr(args, 'mamba_use_acf_memory', False)) or bool(
            getattr(args, 'mamba_dynamic_acf', False)
        ):
            raise ValueError(
                'mamba_acf_memory_gate_mode=variable_token requires static '
                'ACF memory (mamba_use_acf_memory=true, mamba_dynamic_acf=false)'
            )

    if acf_reliability_interpolation:
        source = resolve_acf_source(args)
        if not bool(getattr(args, 'mamba_use_acf_memory', False)):
            raise ValueError(
                'mamba_acf_reliability_interpolation requires '
                'mamba_use_acf_memory=true'
            )
        if bool(getattr(args, 'mamba_dynamic_acf', False)):
            raise ValueError(
                'mamba_acf_reliability_interpolation requires static ACF memory '
                '(mamba_dynamic_acf=false)'
            )
        if source != 'raw':
            raise ValueError(
                'mamba_acf_reliability_interpolation requires '
                'AutoCon_acf_source=raw'
            )
        if bool(getattr(args, 'mamba_acf_step_reliability', False)):
            raise ValueError(
                'mamba_acf_reliability_interpolation is incompatible with '
                'mamba_acf_step_reliability=true'
            )

    if channel_lag_bank:
        if not bool(getattr(args, 'mamba_use_acf_memory', False)):
            raise ValueError(
                'mamba_acf_channel_lag_bank requires mamba_use_acf_memory=true'
            )
        if bool(getattr(args, 'mamba_dynamic_acf', False)):
            raise ValueError(
                'mamba_acf_channel_lag_bank requires static train-only ACF memory '
                '(mamba_dynamic_acf=false)'
            )
        lags = getattr(args, 'mamba_acf_lags', None)
        if isinstance(lags, (str, bytes)):
            lags = [item for item in str(lags).replace(',', ' ').split() if item]
        try:
            canonical_lags = _canonical_positive_lags(lags, 'mamba_acf_lags')
        except ValueError:
            canonical_lags = None
        lag_count = len(canonical_lags) if canonical_lags is not None else 0
        if lag_count <= 0:
            raise ValueError(
                'mamba_acf_channel_lag_bank requires non-empty mamba_acf_lags'
            )
        if channel_lag_top_k > lag_count:
            raise ValueError(
                'mamba_acf_channel_lag_top_k cannot exceed mamba_acf_lags count'
            )

    if str(getattr(args, 'mamba_acf_detrend', 'mean')).lower() == 'linear':
        raise ValueError(
            'mamba_acf_detrend=linear is unsupported for MambaACF because '
            'the runtime memory path does not apply the same global linear '
            'detrend; use mean/none or implement a window-consistent prior'
        )

    acf_temperature = _strict_bounded_float(
        getattr(args, 'mamba_acf_temperature', 4.0),
        'mamba_acf_temperature', minimum=0.0,
    )
    if acf_temperature <= 0:
        raise ValueError('mamba_acf_temperature must be finite and positive')

    if relation_loss:
        relation_defaults = {
            'AutoCon_temperature': 0.2,
            'AutoCon_relation_temperature': 0.25,
            'AutoCon_sign_weight': 0.25,
            'AutoCon_sign_margin': 0.1,
        }

        def _finite_relation_option(name, *, positive=False):
            value = _strict_bounded_float(
                getattr(args, name, relation_defaults[name]), name,
                minimum=0.0,
            )
            if positive and value <= 0:
                qualifier = 'finite and positive' if positive else 'finite and non-negative'
                raise ValueError('{} must be {}'.format(name, qualifier))
            return value

        _finite_relation_option('AutoCon_temperature', positive=True)
        _finite_relation_option('AutoCon_relation_temperature', positive=True)
        _finite_relation_option('AutoCon_sign_weight')
        _finite_relation_option('AutoCon_sign_margin')

    relation_target_mode = str(
        getattr(args, 'mamba_acf_relation_target_mode', 'hard_selected')
    ).lower()
    if relation_target_mode not in {'hard_selected', 'soft_multi_peak', 'soft_full'}:
        raise ValueError(
            'mamba_acf_relation_target_mode must be hard_selected, '
            'soft_multi_peak, or soft_full'
        )
    soft_min_abs = _strict_bounded_float(
        getattr(args, 'mamba_acf_relation_soft_min_abs', 0.1),
        'mamba_acf_relation_soft_min_abs', minimum=0.0, maximum=1.0,
    )
    diagnostic_min_abs = _strict_bounded_float(
        getattr(args, 'mamba_acf_relation_diagnostic_min_abs', 0.1),
        'mamba_acf_relation_diagnostic_min_abs', minimum=0.0, maximum=1.0,
    )
    selected_lag_boost = _strict_bounded_float(
        getattr(args, 'mamba_acf_relation_selected_lag_boost', 0.0),
        'mamba_acf_relation_selected_lag_boost', minimum=0.0,
    )
    horizon_calibration = bool(
        getattr(args, 'mamba_acf_horizon_calibration', False)
    )
    horizon_calibration_rank = _strict_positive_int(
        getattr(args, 'mamba_acf_horizon_calibration_rank', 2),
        'mamba_acf_horizon_calibration_rank',
    )
    if horizon_calibration_rank not in {1, 2}:
        raise ValueError(
            'mamba_acf_horizon_calibration_rank must be 1 or 2'
        )
    if horizon_calibration:
        if not str(getattr(args, 'model', '')).startswith('MambaACF'):
            raise ValueError(
                'mamba_acf_horizon_calibration is supported only by MambaACF models'
            )
        if not bool(getattr(args, 'mamba_use_acf_memory', False)):
            raise ValueError(
                'mamba_acf_horizon_calibration requires mamba_use_acf_memory=true'
            )
        if bool(getattr(args, 'mamba_dynamic_acf', False)):
            raise ValueError(
                'mamba_acf_horizon_calibration requires static train-only ACF memory '
                '(mamba_dynamic_acf=false)'
            )
    if relation_loss:
        if relation_min_lag > max_lag:
            raise ValueError(
                'mamba_acf_relation_min_lag cannot exceed mamba_acf_max_lag '
                'when relation loss is enabled'
            )

    base_tokens = (seq_len + stride - 1) // stride
    scales = getattr(args, 'mamba_scales', [1, 2])
    if isinstance(scales, (str, bytes)):
        raise ValueError('mamba_scales must contain positive integers')
    try:
        scales = [
            _strict_positive_int(scale, 'mamba_scales')
            for scale in list(scales)
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError('mamba_scales must contain positive integers') from exc
    if not scales:
        raise ValueError('mamba_scales must contain positive integers')
    if any(scale > base_tokens for scale in scales):
        raise ValueError(
            'mamba_scales cannot exceed {} base tokens after mamba_token_stride={}'.format(
                base_tokens, stride
            )
        )
    if bool(getattr(args, 'mamba_require_strict_representation_causality', False)):
        if not str(getattr(args, 'model', '')).startswith('MambaACF'):
            raise ValueError(
                'mamba_require_strict_representation_causality is supported '
                'only by MambaACF models'
            )
        causality_errors = strict_representation_causality_errors(args)
        if causality_errors:
            raise ValueError(
                'strict MambaACF representation causality contract failed: '
                + ', '.join(causality_errors)
            )
    if acf_scale_conditioning and len(set(scales)) < 2:
        raise ValueError(
            'mamba_acf_scale_conditioning requires at least two temporal scales'
        )
    if acf_scale_conditioning:
        source = resolve_acf_source(args)
        if source != 'raw':
            raise ValueError(
                'mamba_acf_scale_conditioning requires AutoCon_acf_source=raw'
            )
        try:
            acf_lags = _canonical_positive_lags(
                getattr(args, 'mamba_acf_lags', (24, 168, 336)),
                'mamba_acf_lags',
            )
            eligible_lags = [
                lag for lag in acf_lags if seq_len - lag >= 16
            ]
        except (TypeError, ValueError):
            eligible_lags = []
        if not eligible_lags:
            raise ValueError(
                'mamba_acf_scale_conditioning requires at least one ACF lag '
                'with 16 or more window pairs'
            )
    if acf_local_agreement:
        if not bool(getattr(args, 'mamba_use_acf_memory', False)):
            raise ValueError(
                'mamba_acf_local_agreement requires mamba_use_acf_memory=true'
            )
        if bool(getattr(args, 'mamba_dynamic_acf', False)):
            raise ValueError(
                'mamba_acf_local_agreement requires static ACF memory '
                '(mamba_dynamic_acf=false)'
            )
        source = resolve_acf_source(args)
        if source != 'raw':
            raise ValueError(
                'mamba_acf_local_agreement requires AutoCon_acf_source=raw'
            )
        min_pairs = _strict_positive_int(
            getattr(args, 'mamba_acf_local_min_pairs', 16),
            'mamba_acf_local_min_pairs',
        )
        try:
            acf_lags = _canonical_positive_lags(
                getattr(args, 'mamba_acf_lags', (24, 168, 336)),
                'mamba_acf_lags',
            )
            eligible_lags = [
                lag for lag in acf_lags if seq_len - lag >= min_pairs
            ]
        except (TypeError, ValueError):
            eligible_lags = []
        if not eligible_lags:
            raise ValueError(
                'mamba_acf_local_agreement requires at least one ACF lag with '
                'the configured minimum pair support'
            )
    if bool(getattr(args, 'mamba_low_rank_head', False)):
        rank = _strict_positive_int(
            getattr(args, 'mamba_temporal_rank', 16), 'mamba_temporal_rank'
        )
        pred_len = _strict_positive_int(
            getattr(args, 'pred_len', 96), 'pred_len'
        )
        if rank >= min(base_tokens, pred_len):
            raise ValueError(
                'mamba_temporal_rank must be smaller than min(effective temporal tokens={}, pred_len={})'.format(
                    base_tokens, int(getattr(args, 'pred_len', 96))
                )
            )


def select_experiment_class(args):
    """Select a task-specific experiment implementation.

    Mamba1 and Mamba2 have a tensor/representation output contract that differs from
    attention models, so it is routed through the dedicated experiment module
    instead of being hidden in the generic model dictionary.
    """

    if args.task_name != 'long_term_forecast':
        raise Exception('Other tasks (e.g., Imputation, Classification) are available at TimesNet Repository')

    if args.model in MAMBA_PLAIN_MODELS:
        if args.AutoCon:
            raise ValueError(
                'Use --model {}AutoCon together with --AutoCon for contrastive {} training'.format(
                    args.model, args.model
                )
            )
        return MAMBA_EXPERIMENTS[args.model]

    if args.model in MAMBA_AUTOCON_MODELS:
        if not args.AutoCon:
            raise ValueError('{} requires --AutoCon'.format(args.model))
        return MAMBA_EXPERIMENTS[args.model]

    return Exp_Long_Term_Forecast_with_AutoCon if args.AutoCon else Exp_Long_Term_Forecast


def apply_mamba_public_alias_defaults(args):
    """Expand fixed public MambaACF recipes before runtime validation.

    The training-plan runner normally resolves ``MambaACFLiteAutoCon`` to the
    executable ``MambaACFAutoCon`` plus these overrides.  Applying the same
    contract for direct CLI use keeps the public alias reproducible outside the
    plan runner as well.
    """

    model = getattr(args, "model", None)
    if str(model).startswith("MambaTSv16"):
        args.mamba_ts_variant = "v16"
    elif str(model).startswith("MambaTSv15"):
        args.mamba_ts_variant = "v15"
    elif str(model).startswith("MambaTSv14"):
        args.mamba_ts_variant = "v14"
    elif str(model).startswith("MambaTSv13"):
        args.mamba_ts_variant = "v13"
    elif str(model).startswith("MambaTSv12"):
        args.mamba_ts_variant = "v12"
    elif str(model).startswith("MambaTSv11"):
        args.mamba_ts_variant = "v11"
    elif str(model).startswith("MambaTSv10"):
        args.mamba_ts_variant = "v10"
    elif str(model).startswith("MambaTSv9"):
        args.mamba_ts_variant = "v9"
    elif str(model).startswith("MambaTSv8"):
        args.mamba_ts_variant = "v8"
    elif str(model).startswith("MambaTSv7"):
        args.mamba_ts_variant = "v7"
    elif str(model).startswith("MambaTSv6"):
        args.mamba_ts_variant = "v6"
    elif str(model).startswith("MambaTSv5"):
        args.mamba_ts_variant = "v5"
    elif str(model).startswith("MambaTSv4"):
        args.mamba_ts_variant = "v4"
    elif str(model).startswith("MambaTSv3"):
        args.mamba_ts_variant = "v3"
    elif str(model).startswith("MambaTSv2"):
        args.mamba_ts_variant = "v2"
    elif str(model).startswith("MambaTS"):
        args.mamba_ts_variant = "v1"
    if model != "MambaACFLiteAutoCon":
        return args
    args.AutoCon = True
    args.mamba_scales = [1]
    args.mamba_causal_embedding = True
    args.mamba_low_rank_head = True
    args.mamba_temporal_rank = 8
    args.mamba_variable_mixer = False
    args.mamba_patch_mixer = False
    args.mamba_local_mixer = False
    args.mamba_use_acf_memory = False
    args.mamba_acf_relation_loss = True
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoCon Argument Parser')

    # AutoCon config
    parser.add_argument('--AutoCon', action='store_true', default=False, help='')
    parser.add_argument('--AutoCon_wnorm', type=str, default='LastVal', help='Window Normalization Techniques: {MeanStd | ReVIN | Mean | LastVal | Decomp}')

    parser.add_argument('--AutoCon_lambda', type=float, default=0.05, help='weight of the AutoCon objective')
    parser.add_argument('--AutoCon_warmup_epochs', type=int, default=0,
                        help='linearly warm up the AutoCon weight for this many epochs (0 disables warmup)')
    parser.add_argument('--AutoCon_acf_source', type=str, default='residual',
                        choices=['residual', 'trend', 'raw'],
                        help='training-split signal used to estimate the ACF prior')
    parser.add_argument('--mamba_acf_relation_loss', action='store_true', default=False,
                        help='use the all-pair signed ACF relation objective for MambaACF AutoCon')
    parser.add_argument('--mamba_acf_relation_batching', type=str,
                        choices=['lag_pairs', 'random'], default='lag_pairs',
                        help='batch construction for MambaACF signed ACF relation learning')
    parser.add_argument('--mamba_acf_relation_lag_balance', type=str,
                        choices=['random', 'balanced'], default='random',
                        help='opt-in selected-lag exposure policy for lag-pair batching')
    parser.add_argument('--mamba_acf_relation_channel_reduce', type=str,
                        choices=['mean', 'max_abs'], default='mean',
                        help='how per-channel signed ACF relations supervise a shared representation')
    parser.add_argument('--mamba_acf_relation_scope', type=str,
                        choices=['both', 'global', 'local'], default='both',
                        help='active signed ACF relation component(s) for MambaACF AutoCon')
    parser.add_argument('--mamba_acf_relation_representation', type=str,
                        choices=['encoder', 'forecast'], default='encoder',
                        help='relation view: encoder representation or final normalized forecast')
    parser.add_argument('--mamba_acf_relation_lag_mask', action='store_true', default=False,
                        help='restrict positive relation targets to the resolved train ACF lags')
    parser.add_argument('--mamba_acf_relation_lag_tolerance', type=int, default=0,
                        help='integer tolerance around selected ACF relation lags')
    parser.add_argument('--mamba_acf_relation_min_lag', type=int, default=0,
                        help='minimum resolved ACF lag used by relation supervision and lag-pair batching; '
                             'set to seq_len for a no-overlap relation control')
    parser.add_argument('--mamba_acf_relation_target_mode', type=str,
                        choices=['hard_selected', 'soft_multi_peak', 'soft_full'],
                        default='hard_selected',
                        help='positive-target semantics for signed ACF relation learning')
    parser.add_argument('--mamba_acf_relation_soft_min_abs', type=float, default=0.1,
                        help='minimum absolute ACF strength for soft_multi_peak targets')
    parser.add_argument('--mamba_acf_relation_diagnostic_min_abs', type=float, default=0.1,
                        help='independent absolute ACF threshold used by high-ACF diagnostics')
    parser.add_argument('--mamba_acf_relation_selected_lag_boost', type=float, default=0.0,
                        help='non-negative multiplicative target-weight boost for selected lags; 1.0 doubles their weight')
    parser.add_argument('--mamba_acf_relation_active_anchor_normalization',
                        action='store_true', default=False,
                        help='opt in to averaging global relation loss over active anchors only')
    parser.add_argument('--mamba_acf_forecast_pair_consistency',
                        action='store_true', default=False,
                        help='match forecasts from train-ACF lag-paired windows during training')
    parser.add_argument('--mamba_acf_forecast_pair_weight', type=float, default=0.0,
                        help='training-only weight for forecast lag-pair ACF consistency')
    parser.add_argument('--AutoCon_temperature', type=float, default=0.2,
                        help='representation temperature for the MambaACF relation objective')
    parser.add_argument('--AutoCon_relation_temperature', type=float, default=0.25,
                        help='ACF relation-distribution temperature for MambaACF')
    parser.add_argument('--AutoCon_sign_weight', type=float, default=0.25,
                        help='sign-aware phase penalty weight for MambaACF relation loss')
    parser.add_argument('--AutoCon_sign_margin', type=float, default=0.1,
                        help='minimum signed similarity margin for MambaACF relation loss')
    parser.add_argument('--AutoCon_multiscales', type=int, nargs='+', default=[96], help='e.g., [96, 192], [96, 192, 720]')

    parser.add_argument('--train_ratio', type=float, default=0.6, help='For custom dataset except for ETT datasets')
    parser.add_argument('--save', action='store_true', default=True, help='Save predictions')
    parser.add_argument('--no_save', action='store_false', dest='save', help='do not save prediction arrays')
    parser.add_argument('--skip_test', action='store_true',
                        help='train and save the best checkpoint without reading the test split')
    parser.add_argument('--stream_metrics', action='store_true',
                        help='stream test metrics and retain only compact samples instead of full pred/true arrays')
    parser.add_argument('--origin_stride', type=int, default=None,
                        help='stride in test windows for compact origin sensitivity metrics (defaults to pred_len)')
    parser.add_argument('--stream_sample_windows', type=int, default=4,
                        help='number of non-overlapping origin windows retained for manual inspection')
    parser.add_argument('--seed', type=int, default=2021, help='base random seed; itr adds the run index')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='AutoConNet',
                        help='model name; baselines include [DLinear, GraphDLinear, STGCN, GraphWaveNet, DCRNN, AGCRN, NLinear, RLinear, PatchTST, TimesNet, TimeMixer, iTransformer, Autoformer, FEDformer, ETSformer, Transformer, Crossformer]; Mamba options: [Mamba1, Mamba1AutoCon, Mamba2, Mamba2AutoCon, MambaTS, MambaTSAutoCon, MambaTSv2, MambaTSv2AutoCon, MambaTSv3, MambaTSv3AutoCon, MambaTM1, MambaTM1AutoCon, MambaTM2, MambaTM2AutoCon, MambaSM, MambaSMAutoCon, MambaACF, MambaACFAutoCon, MambaACFLiteAutoCon, MambaACF1, MambaACF1AutoCon]')
    parser.add_argument('--plan_job_id', type=str, default=None,
                        help='opaque plan-runner job identifier stored in config metadata')
    parser.add_argument('--setting_override', type=str, default=None,
                        help='evaluation-only setting directory binding for replaying an existing artifact')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', '--feature', dest='features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--data_format', type=str, choices=['auto', 'npz', 'txt', 'csv_numeric'], default='auto',
                        help='input format for optional numeric-array datasets; CSV datasets ignore this option')
    parser.add_argument('--data_key', type=str, default='data',
                        help='array key for --data_format npz (default: data)')
    parser.add_argument('--data_feature_index', type=int, default=0,
                        help='feature axis index for [time, channel, feature] npz arrays')
    parser.add_argument('--target_index', type=int, default=-1,
                        help='target channel index for numeric-array datasets in S or MS mode')
    parser.add_argument('--data_interval_minutes', type=int, default=None,
                        help='actual sampling interval for numeric arrays with freq=t (for example 5 for PEMS, 10 for Solar)')
    parser.add_argument('--data_start_timestamp', type=str, default='2000-01-01',
                        help='calendar origin for numeric-array time features; use the documented source start when available')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--wo_freq', type=str, default='',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results/', help='directory for metrics and predictions')
    parser.add_argument('--test_results_dir', type=str, default='./test_results/', help='directory for temporary test outputs')
    parser.add_argument('--result_log', type=str, default='result_long_term_forecast.txt', help='plain-text result log')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (percent)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--timemixer_down_sampling_window', type=int, default=2,
                        help='TimeMixer down-sampling ratio')
    parser.add_argument('--timemixer_down_sampling_layers', type=int, default=1,
                        help='TimeMixer number of additional scales')
    parser.add_argument('--timemixer_channel_independence', action='store_true', default=False,
                        help='use TimeMixer channel-independent embedding/projection')
    parser.add_argument('--timemixer_decomp_method', choices=['moving_avg', 'dft_decomp'],
                        default='moving_avg', help='TimeMixer decomposition')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--patch_len', type=int, default=16, help='PatchTST patch length')
    parser.add_argument('--patch_stride', type=int, default=8, help='PatchTST patch stride')
    parser.add_argument('--mamba_ts_variant', type=str, choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16'], default='v1',
                        help='MambaTS architecture variant; registry aliases set this automatically')
    parser.add_argument('--mamba_ts_mixer_type', choices=['mamba1', 'mamba2'],
                        default='mamba1',
                        help='MambaTS scan mixer; mamba2 is an explicit new architecture candidate')
    parser.add_argument('--mamba_ts_context_rank', type=int, default=16,
                        help='low-rank cross-variable context width for MambaTS v3')
    parser.add_argument('--mamba_ts_channel_agnostic_scan', action='store_true',
                        default=False,
                        help='run the one MambaTSv3 block on shared per-channel patch sequences without channel embeddings/context')
    parser.add_argument('--mamba_ts_vpt_solver', choices=['greedy', 'sa', 'ls', 'lk'],
                        default='greedy',
                        help='MambaTS v11 evaluation route solver; SA/LS/LK reproduce the python-tsp VPT path family')
    parser.add_argument('--mamba_ts_delta_multipliers', type=float, nargs='+', default=[1.0, 2.0],
                        help='MambaTS v12 Delta multipliers: 1.0 followed by a value greater than 1.0')
    parser.add_argument('--mamba_ts_dt_min', type=float, default=0.001,
                        help='MambaTS v12 base Delta lower initialization bound')
    parser.add_argument('--mamba_ts_dt_max', type=float, default=0.1,
                        help='MambaTS v12 base Delta upper initialization bound')
    parser.add_argument('--mamba_ts_spectral_top_k', type=int, default=4,
                        help='number of dominant causal Fourier bins for MambaTS v8')
    parser.add_argument('--mamba_ts_residual_init', type=float, default=0.0,
                        help='initial raw ReZero scale for the MambaTS v6 linear residual')
    parser.add_argument('--mamba_ts_repr_forecast_fusion', action='store_true',
                        default=False,
                        help='feed the AutoCon representation through a ReZero shared forecast residual in MambaTS v3/v4/v5')
    parser.add_argument('--mamba2_repr_forecast_fusion', action='store_true',
                        default=False,
                        help='feed the Mamba2AutoCon representation through a ReZero shared forecast residual')
    parser.add_argument('--mamba_ts_repr_forecast_channelwise', action='store_true',
                        default=False,
                        help='preserve each channel in the MambaTS AutoCon representation forecast residual')
    parser.add_argument('--mamba_ts_channel_adapter', action='store_true',
                        default=False,
                        help='add a low-rank channel-preserving residual after the existing MambaTS v3 scan; requires AutoCon')
    parser.add_argument('--mamba_ts_channel_adapter_rank', type=int, default=8,
                        help='shared low-rank width for the MambaTS v3 channel adapter')
    parser.add_argument('--mamba_ts_revin_affine', action='store_true',
                        default=False,
                        help='learn a reversible per-channel affine adapter after MeanStd/ReVIN normalization; requires MambaTSv3AutoCon')
    parser.add_argument('--mamba_ts_global_channel_mixer', action='store_true',
                        default=False,
                        help='condition the shared MambaTS v3 state on a low-rank cross-variable summary; requires AutoCon')
    parser.add_argument('--mamba_ts_global_channel_mixer_rank', type=int, default=4,
                        help='low-rank width for the MambaTS v3 global channel-state mixer')
    parser.add_argument('--mamba_ts_graph_spatial', action='store_true', default=False,
                        help='bind the traffic graph contract for MambaTS v3, GraphDLinear, STGCN, GraphWaveNet, or DCRNN; MambaTS requires AutoCon')
    parser.add_argument('--mamba_ts_graph_path', type=str, default='',
                        help='safe NPZ graph contract aligned to MambaTS input channels')
    parser.add_argument('--mamba_ts_graph_expected_sha256', type=str, default='',
                        help='optional SHA-256 required for the MambaTS graph contract')
    parser.add_argument('--mamba_ts_repr_linear_expert', action='store_true',
                        default=False,
                        help='use the AutoCon representation to gate a shared linear expert in MambaTS v3/v4/v5')
    parser.add_argument('--mamba_ts_repr_linear_expert_mode', type=str,
                        choices=['residual', 'blend', 'nlinear_blend'], default='residual',
                        help='linear expert fusion: additive residual, bounded persistence blend, or NLinear-style blend')
    parser.add_argument('--mamba_ts_repr_linear_blend_bias', type=float,
                        default=-4.0,
                        help='initial logit bias for the bounded linear/Mamba blend')
    parser.add_argument('--mamba_ts_repr_linear_expert_rank', type=int,
                        default=16,
                        help='low-rank width for the NLinear-style blend expert')
    parser.add_argument('--mamba_ts_variable_permutation', action='store_true',
                        default=False,
                        help='randomly permute MambaTS v3/v4/v5 variable patch-token order during training and restore output order')
    parser.add_argument('--mamba_ts_acf_memory_selector', action='store_true',
                        default=False,
                        help='use the MambaTS v4/v5 AutoCon representation to select a horizon-wise correction to train-prior ACF memory')
    parser.add_argument('--mamba_ts_acf_memory_channel_selector', action='store_true',
                        default=False,
                        help='add a shared per-variable Mamba-state confidence to the ACF memory selector')
    parser.add_argument('--mamba_ts_forecast_acf_consistency', action='store_true',
                        default=False,
                        help='match MambaTS v3/v4/v5 AutoCon forecast temporal ACF to the train-only ACF prior during training')
    parser.add_argument('--mamba_ts_forecast_acf_weight', type=float, default=0.0,
                        help='training-only weight for MambaTS forecast ACF consistency')
    parser.add_argument('--mamba_ts_forecast_acf_top_k', type=int, default=4,
                        help='number of train-only ACF lags retained for forecast consistency')
    parser.add_argument('--mamba_ts_forecast_acf_min_lag', type=int, default=2,
                        help='minimum temporal lag for MambaTS forecast ACF consistency')
    parser.add_argument('--mamba_ts_forecast_acf_min_abs', type=float, default=0.10,
                        help='minimum train ACF magnitude eligible for forecast consistency')
    parser.add_argument('--mamba_d_state', type=int, default=64, help='Mamba2 state dimension')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba2 local convolution width')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba2 expansion factor')
    parser.add_argument('--mamba_headdim', type=int, default=16, help='Mamba2 SSM head dimension')
    parser.add_argument('--mamba_tm_n1', type=int, default=64,
                        help='TimeMachine first projected width')
    parser.add_argument('--mamba_tm_n2', type=int, default=128,
                        help='TimeMachine second projected width')
    parser.add_argument('--mamba_tm_repr_width', type=int, default=64,
                        help='TimeMachine temporal AutoCon representation width')
    parser.add_argument('--mamba_tm_channel_independent', action='store_true',
                        default=False,
                        help='scan compressed temporal tokens independently per channel')
    parser.add_argument('--mamba_tm_residual', type=str2bool, default=True,
                        help='include TimeMachine projected-state residual paths')
    parser.add_argument('--mamba_bipatch_token_mode', type=str, default='mixing',
                        choices=['mixing', 'independent'],
                        help='BiPatchMamba token axis: channel mixing per patch or independent patches per channel')
    parser.add_argument('--mamba_bipatch_repr_gate', action='store_true', default=False,
                        help='use a bounded representation-conditioned forecast gate in BiPatchMamba AutoCon')
    parser.add_argument('--mamba_scales', type=int, nargs='+', default=[1, 2],
                        help='temporal downsampling scales for MambaACF')
    parser.add_argument('--mamba_token_stride', type=int, default=1,
                        help='causal base patch stride before MambaACF temporal scans')
    parser.add_argument('--mamba_bidirectional', action='store_true', default=False,
                        help='add a reverse Mamba scan in MambaACF')
    parser.add_argument('--mamba_channel_independent', action='store_true', default=False,
                        help='run one Mamba scan per channel (slow ablation; shared mode is default)')
    parser.add_argument('--mamba_channel_gate', action='store_true', default=False,
                        help='enable low-rank, input-window-conditioned cross-variable modulation')
    parser.add_argument('--mamba_channel_gate_rank', type=int, default=8,
                        help='rank of the low-rank cross-variable modulation adapter')
    parser.add_argument('--mamba_data_dependent_channel_mixer', action='store_true',
                        default=False,
                        help='enable low-rank window-conditioned cross-channel state modulation')
    parser.add_argument('--mamba_data_dependent_channel_mixer_rank', type=int, default=8,
                        help='rank of the data-dependent cross-channel state adapter')
    parser.add_argument('--mamba_acf_scale_conditioning', action='store_true', default=False,
                        help='condition multi-scale temporal fusion with train-prior-weighted current-window ACF')
    parser.add_argument('--mamba_acf_local_agreement', action='store_true', default=False,
                        help='reweight static forecast-memory lags using causal local ACF agreement')
    parser.add_argument('--mamba_acf_local_min_pairs', type=int, default=16,
                        help='minimum observed pairs required for local ACF agreement')
    parser.add_argument('--mamba_causal_embedding', action='store_true', default=False,
                        help='use strictly left-padded value embedding in MambaACF')
    parser.add_argument('--mamba_require_strict_representation_causality',
                        action='store_true', default=False,
                        help='require a strictly causal full MambaACF AutoCon representation grid')
    parser.add_argument('--mamba_variable_mixer', action='store_true', default=False,
                        help='scan variable tokens with a shared Mamba branch')
    parser.add_argument('--mamba_variable_gdd_mixer', action='store_true',
                        default=False,
                        help='apply a low-rank CMamba-style data-dependent modulation to variable-token states')
    parser.add_argument('--mamba_variable_gdd_mixer_rank', type=int, default=8,
                        help='bottleneck rank of the variable-token GDD modulation')
    parser.add_argument('--mamba_variable_gdd_init_scale', type=float, default=0.0,
                        help='initial raw residual scale for variable-token GDD (0..1)')
    parser.add_argument('--mamba_variable_repr_fusion', action='store_true', default=False,
                        help='fuse variable-token states into the MambaACF AutoCon representation')
    parser.add_argument('--mamba_variable_acf_memory_selector', action='store_true', default=False,
                        help='use the AutoCon representation to gate train-only ACF memory in NLinearVariableMamba')
    parser.add_argument('--mamba_variable_mixer_depth', type=int, default=1,
                        help='number of blocks in the optional variable-token Mamba branch')
    parser.add_argument('--mamba_variable_group_size', type=int, default=1,
                        help='contiguous channels per grouped variable-token Mamba scan (1 keeps one token per channel)')
    parser.add_argument('--mamba_variable_permutation', action='store_true', default=False,
                        help='randomly permute variable tokens during training and restore output order')
    parser.add_argument('--mamba_acf_channel_order', type=str,
                        choices=['input', 'strength', 'similarity'], default='input',
                        help='variable-token order: input CSV, train-only ACF strength, or signed ACF-profile similarity order')
    parser.add_argument('--mamba_acf_channel_conditioning', action='store_true', default=False,
                        help='condition variable-token forecasts on train-only signed ACF channel summaries')
    parser.add_argument('--mamba_acf_repr_fusion', action='store_true', default=False,
                        help='inject causal train-only ACF lag context into the AutoCon representation path')
    parser.add_argument('--mamba_acf_channel_gate', action='store_true', default=False,
                        help='inject train-only signed ACF summaries into the low-rank channel gate (no variable-token scan)')
    parser.add_argument('--mamba_patch_mixer', action='store_true', default=False,
                        help='preserve per-variable temporal patches and scan them with a shared Mamba branch')
    parser.add_argument('--mamba_patch_mixer_depth', type=int, default=1,
                        help='number of blocks in the optional channel-patch Mamba branch')
    parser.add_argument('--mamba_patch_len', type=int, default=16,
                        help='temporal patch length for the optional channel-patch branch')
    parser.add_argument('--mamba_patch_stride', type=int, default=8,
                        help='temporal patch stride for the optional channel-patch branch')
    parser.add_argument('--mamba_patch_order', choices=['time_major', 'channel_major'],
                        default='time_major', help='order of channel-patch tokens in the Mamba scan')
    parser.add_argument('--mamba_patch_permutation', action='store_true', default=False,
                        help='randomly permute channels within patch-token groups during training')
    parser.add_argument('--mamba_local_mixer', action='store_true', default=False,
                        help='add a channel-preserving local patch adapter with a low-rank horizon head')
    parser.add_argument('--mamba_local_patch_len', type=int, default=16,
                        help='temporal patch length for the channel-preserving local adapter')
    parser.add_argument('--mamba_local_patch_stride', type=int, default=8,
                        help='temporal patch stride for the channel-preserving local adapter')
    parser.add_argument('--mamba_local_width', type=int, default=None,
                        help='local patch feature width (defaults to d_model)')
    parser.add_argument('--mamba_local_rank', type=int, default=16,
                        help='rank of the local adapter horizon projection')
    parser.add_argument('--mamba_local_depth', type=int, default=1,
                        help='number of shared local patch mixer blocks')
    parser.add_argument('--mamba_low_rank_head', action='store_true', default=False,
                        help='factorize the temporal and linear-residual heads (optional ablation)')
    parser.add_argument('--mamba_temporal_rank', type=int, default=16,
                        help='rank used by --mamba_low_rank_head; must be below min(seq_len, pred_len)')
    parser.add_argument('--mamba_acf_lags', type=int, nargs='+', default=[24, 168, 336],
                        help='causal seasonal-memory lags for MambaACF')
    parser.add_argument('--mamba_acf_channel_lag_bank', action='store_true', default=False,
                        help='restrict static ACF memory to per-channel top-k train-only lag slots')
    parser.add_argument('--mamba_acf_channel_lag_top_k', type=int, default=2,
                        help='number of train-only lag slots retained per channel in the optional lag bank')
    parser.add_argument('--mamba_acf_lag_mode', type=str,
                        choices=['fixed', 'train_peaks', 'train_peaks_available'], default='fixed',
                        help='use configured lags, strongest train peaks, or peaks weighted by legal forecast availability')
    parser.add_argument('--mamba_acf_max_lag', type=int, default=720,
                        help='maximum lag evaluated when fitting the train-only ACF prior')
    parser.add_argument('--mamba_acf_top_k', type=int, default=8,
                        help='number of candidate train-split ACF peaks retained by the prior fitter')
    parser.add_argument('--mamba_acf_min_lag', type=int, default=2,
                        help='minimum lag considered by the train-only ACF prior fitter')
    parser.add_argument('--mamba_acf_min_abs', type=float, default=0.1,
                        help='minimum absolute ACF magnitude for a selected peak')
    parser.add_argument('--mamba_acf_confidence_z', type=float, default=1.96,
                        help='normal-approximation threshold used for ACF peak selection')
    parser.add_argument('--mamba_acf_min_separation', type=int, default=2,
                        help='minimum separation between selected ACF peaks')
    parser.add_argument('--mamba_acf_detrend', type=str, choices=['none', 'mean', 'linear'], default='mean',
                        help='detrending mode for the train-only signed ACF prior')
    parser.add_argument('--mamba_acf_chunk', type=int, default=64,
                        help='channel chunk size for train-only FFT ACF estimation')
    parser.add_argument('--mamba_use_acf_memory', type=str2bool, default=False,
                        help='enable the causal ACF seasonal-memory residual')
    parser.add_argument('--mamba_dynamic_acf', type=str2bool, default=False,
                        help='estimate signed ACF from each input window when no fitted prior is attached')
    parser.add_argument('--mamba_acf_temperature', type=float, default=4.0,
                        help='softmax temperature for ACF memory reliability')
    parser.add_argument('--mamba_acf_step_reliability', type=str2bool, default=False,
                        help='gate each forecast step by the ACF lags legal at that step')
    parser.add_argument('--mamba_acf_reliability_interpolation', type=str2bool, default=False,
                        help='learn a bounded interpolation from aggregate to per-step/channel static ACF reliability')
    parser.add_argument('--mamba_collect_eval_acf_diagnostics', type=str2bool, default=False,
                        help='collect ACF forecast-path diagnostics during evaluation/test (adds GPU-to-CPU telemetry sync)')
    parser.add_argument('--mamba_acf_memory_gate_mode', type=str,
                        choices=['hidden', 'variable_token'], default='hidden',
                        help='ACF memory gate source; variable_token uses restored variable-token states')
    parser.add_argument('--mamba_acf_anchor_copy', action='store_true', default=False,
                        help='use a causal signed-ACF lag anchor in the MambaACF forecast path')
    parser.add_argument('--mamba_acf_horizon_calibration', action='store_true', default=False,
                        help='learn a low-rank, train-only horizon amplitude calibration for static ACF memory')
    parser.add_argument('--mamba_acf_horizon_calibration_rank', type=int, choices=[1, 2], default=2,
                        help='rank of the fixed constant/linear horizon basis used by ACF memory calibration')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--drop_last_train', action='store_true', default=False,
                        help='drop an incomplete training batch')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Adam weight decay; omitted preserves the legacy trainer defaults')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=0.0,
                        help='floor for the cosine learning-rate schedule')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use gpu (true/false)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # ONNX export
    parser.add_argument('--export_onnx', action='store_true', default=False, help='export the trained model to ONNX')

    args = parser.parse_args()
    apply_mamba_public_alias_defaults(args)
    validate_mamba_acf_arguments(args)
    seed_everything(args.seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = select_experiment_class(args)

    if args.is_training:
        results = []
        base_seed = int(args.seed)
        for ii in range(args.itr):
            run_seed = base_seed + ii
            args.run_seed = run_seed
            seed_everything(run_seed)
            # Keep the directory name bounded; ``config.json`` retains every
            # requested field for full provenance.
            setting = experiment_setting(args, ii, run_seed)
            metadata_dir = os.path.join(args.results_dir, setting)
            os.makedirs(metadata_dir, exist_ok=True)
            clear_acf_prior_metadata(args)
            persist_run_config(metadata_dir, args, run_seed, setting)

            exp = Exp(args)  # set experiments
            # A MambaACF experiment resolves train-peak lags while building
            # its train-only prior.  Rewrite the metadata once that choice is
            # known, without changing any requested setting.
            persist_acf_prior_artifact(metadata_dir, args, exp)
            persist_run_config(metadata_dir, args, run_seed, setting)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if args.skip_test:
                marker = os.path.join(metadata_dir, 'training_complete.json')
                with open(marker, 'w', encoding='utf-8') as handle:
                    json.dump({'status': 'trained_without_test', 'setting': setting,
                               'run_seed': run_seed}, handle, indent=2)
                print('>>>>>>>test skipped by --skip_test<<<<<<<<<<<<<<<<<<<<<<<<<')
                torch.cuda.empty_cache()
                continue

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae, mape, mspe, dilate_e, shape_e, temporal_e = exp.test(setting)
            results.append((mse, mae, mape, mspe, dilate_e, shape_e, temporal_e))
            torch.cuda.empty_cache()

        if not results:
            print('Training complete; no test metrics were requested.')
            raise SystemExit(0)

        m_mse, mse_nh, mse_ph = mean_confidence_interval([res[0] for res in results])
        m_mae, mae_nh, mae_ph = mean_confidence_interval([res[1] for res in results])
        m_mape, mape_nh, mape_ph = mean_confidence_interval([res[2] for res in results])
        m_mspe, mspe_nh, mspe_ph = mean_confidence_interval([res[3] for res in results])
        m_shape_e, shape_e_nh, shape_e_ph = mean_confidence_interval([res[5] for res in results])
        m_temporal_e, temporal_e_nh, temporal_e_ph = mean_confidence_interval([res[6] for res in results])

        print(f'MSE:{m_mse:.4f}+-{mse_ph:.5f}, MAE:{m_mae:.4f}+-{mae_ph:.5f}, MAPE:{m_mape:.4f}+-{mape_ph:.5f}'
              f', MSPE:{m_mspe:.4f}+-{mspe_ph:.5f}, SHAPEDTW:{m_shape_e:.4f}+-{shape_e_ph:.5f}, TEMPDTW:{m_temporal_e:.4f}+-{temporal_e_ph:.5f}')
    else:
        args.run_seed = int(args.seed)
        ii = 0
        setting = str(getattr(args, 'setting_override', '') or '').strip()
        if not setting:
            setting = experiment_setting(args, ii, args.seed)
        metadata_dir = os.path.join(args.results_dir, setting)
        os.makedirs(metadata_dir, exist_ok=True)
        # Preserve the previously recorded sidecar hash before rewriting the
        # CLI metadata.  Evaluation should consume the exact prior paired with
        # an existing result, not silently replace it with a newly fitted one.
        previous_config = {}
        previous_config_error = None
        previous_config_path = os.path.join(metadata_dir, 'config.json')
        if os.path.isfile(previous_config_path):
            try:
                with open(previous_config_path, encoding='utf-8') as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    previous_config = loaded
                else:
                    previous_config_error = 'is not a JSON object'
            except (OSError, ValueError, TypeError):
                previous_config_error = 'is unreadable or invalid JSON'
        else:
            previous_config_error = 'is missing'
        checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        if (
            str(getattr(args, 'model', '')).strip() in MAMBA_MODELS
            and os.path.isfile(checkpoint_path)
            and previous_config_error is not None
        ):
            raise RuntimeError(
                'Mamba checkpoint evaluation requires a valid persisted '
                'config.json before model construction; {} {}'.format(
                    previous_config_path, previous_config_error
                )
            )
        # Validate construction-time Mamba branch flags before creating the
        # experiment/model.  A sidecar can restore coefficients and lags, but
        # it cannot retroactively undo a model built with a different forecast
        # function.
        validate_persisted_acf_runtime_contract(args, previous_config)
        prior_required = _mamba_acf_prior_required(args)
        preflight_artifact_path = None
        if os.path.isfile(checkpoint_path):
            preflight_artifact_path = _preflight_mamba_acf_sidecar(
                metadata_dir, args, previous_config
            )
        prepare_eval_acf_prior_args(
            args,
            prior_required=prior_required,
            preflight_artifact_path=preflight_artifact_path,
            expected_sha=previous_config.get('mamba_acf_prior_sha256'),
        )
        exp = Exp(args)  # set experiments
        loaded_sidecar = load_eval_acf_prior_if_required(
            metadata_dir,
            args,
            exp,
            expected_sha=previous_config.get('mamba_acf_prior_sha256'),
            prior_required=prior_required,
        )
        if loaded_sidecar:
            restore_acf_resolution_metadata(args, previous_config, strict=True)
        elif (
            os.path.isfile(checkpoint_path)
            and not prior_required
        ):
            # ACF-off replay does not consume the train prior, but the
            # existing checkpoint's config must retain its sidecar identity
            # for later provenance audits and branch replays.
            preserve_acf_prior_metadata(args, previous_config)
        # These are construction-time inputs only; keeping an absolute path in
        # the persisted config would make an otherwise portable result
        # directory machine-specific.
        vars(args).pop('mamba_acf_prior_artifact_path', None)
        vars(args).pop('mamba_acf_prior_expected_sha256', None)
        if not loaded_sidecar and prior_required:
            if (
                os.path.isfile(checkpoint_path)
                and str(getattr(args, 'model', '')).startswith('MambaACF')
                and prior_required
            ):
                raise RuntimeError(
                    'MambaACF checkpoint evaluation requires a validated '
                    'acf_prior.npz sidecar; refusing to refit or silently '
                    'change the ACF objective'
                )
            persist_acf_prior_artifact(metadata_dir, args, exp)
        # Write the resolved evaluation metadata only after sidecar loading and
        # all consistency checks succeed.  A failed replay must leave the
        # original config (including its sidecar identity) intact so a later
        # retry can still discover the artifact.
        persist_run_config(metadata_dir, args, args.seed, setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
