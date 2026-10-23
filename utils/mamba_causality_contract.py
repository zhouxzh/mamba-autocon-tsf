"""Configuration-only causality contracts for MambaACF.

The forecasting memory can be causal even when the representation grid is not.
Keep this check independent of PyTorch so plan, runtime, and artifact validators
share the same conservative interpretation of a strict representation claim.
"""

from __future__ import annotations

from collections.abc import Mapping

from utils.acf_relation_contract import resolve_acf_source


_MAMBA_ACF_MODELS = frozenset(
    {
        "MambaACF",
        "MambaACFAutoCon",
        "MambaACF1",
        "MambaACF1AutoCon",
        "MambaACFVariable",
        "MambaACFVariableAutoCon",
        "MambaACFLiteAutoCon",
    }
)

# Every normalization mode currently implemented by MambaACF derives its
# centering (and, for ReVIN/MeanStd, scale) from the complete input window.  None of
# these modes can therefore support a strict prefix-by-prefix representation
# causality claim.  Keep the set here so runtime, plan, and artifact checks use
# the same interpretation.
_FULL_WINDOW_NORMALIZATION_MODES = frozenset(
    {"meanstd", "revin", "mean", "lastval"}
)


def _value(config, key, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _bool(value):
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _scales(config):
    raw = _value(config, "mamba_scales", (1, 2))
    if isinstance(raw, str):
        raw = [item for item in raw.replace(",", " ").split() if item]
    try:
        return tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError):
        return None


def resolve_autocon_wnorm(config, default="LastVal"):
    """Resolve the plan/runtime spelling of the AutoCon window norm mode."""

    # JSON plans use the lower-case spelling; runtime namespaces historically
    # use ``AutoCon_wnorm``.  Match the plan command builder when both exist.
    raw = _value(config, "autocon_wnorm", None)
    if raw is None:
        raw = _value(config, "AutoCon_wnorm", default)
    return str(raw).strip()


def is_full_window_normalization(value):
    """Return whether a normalization mode reads the complete input window."""

    return str(value).strip().lower() in _FULL_WINDOW_NORMALIZATION_MODES


def strict_representation_causality_errors(config, model=None):
    """Return conservative errors for a strict MambaACF grid claim.

    The check describes the full AutoCon representation grid, not merely the
    indexed forecast-memory lookup.  It intentionally does not reject
    variable-token or patch forecast branches because those branches do not
    change the temporal hidden grid used by the relation objective.
    """

    model_name = str(model if model is not None else _value(config, "model", ""))
    if model_name not in _MAMBA_ACF_MODELS:
        return []
    errors = []
    if not _bool(_value(config, "mamba_causal_embedding", False)):
        errors.append("mamba_causal_embedding_required")
    if _bool(_value(config, "mamba_bidirectional", False)):
        errors.append("mamba_bidirectional_forbidden")
    try:
        stride = int(_value(config, "mamba_token_stride", 1))
    except (TypeError, ValueError):
        stride = None
    if stride != 1:
        errors.append("mamba_token_stride_must_be_one")
    scales = _scales(config)
    if scales != (1,):
        errors.append("mamba_scales_must_be_single_scale_one")
    if _bool(_value(config, "mamba_channel_gate", False)):
        errors.append("mamba_channel_gate_forbidden")
    if _bool(_value(config, "mamba_data_dependent_channel_mixer", False)):
        errors.append("mamba_data_dependent_channel_mixer_forbidden")
    if _bool(_value(config, "mamba_acf_scale_conditioning", False)):
        errors.append("mamba_acf_scale_conditioning_forbidden")
    if is_full_window_normalization(resolve_autocon_wnorm(config)):
        errors.append("window_full_normalization_forbidden")
    if (
        _bool(_value(config, "mamba_local_mixer", False))
        and model_name.endswith("AutoCon")
    ):
        errors.append("mamba_local_mixer_forbidden")
    if _bool(_value(config, "mamba_variable_repr_fusion", False)):
        errors.append("mamba_variable_repr_fusion_forbidden")
    if _bool(_value(config, "mamba_acf_repr_fusion", False)):
        source = resolve_acf_source(config)
        if source != "raw":
            errors.append("autocon_acf_source_raw_required")
    return sorted(set(errors))


__all__ = [
    "is_full_window_normalization",
    "resolve_autocon_wnorm",
    "strict_representation_causality_errors",
]
