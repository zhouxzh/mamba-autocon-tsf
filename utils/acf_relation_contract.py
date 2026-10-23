"""Data-independent contracts for MambaACF relation sampling.

This module intentionally has no training-framework imports.  The data
provider and artifact validator both use the resolver so a JSON audit does
not require importing the CSV datasets or PyTorch.
"""

from __future__ import annotations

from collections.abc import Mapping

_ACF_RELATION_MODELS = frozenset({
    "MambaACFAutoCon", "MambaACF1AutoCon", "NLinearVariableMambaAutoCon",
})
_ACF_RELATION_TARGET_MODES = frozenset(
    {"hard_selected", "soft_multi_peak", "soft_full"}
)
_ACF_RELATION_SCOPES = frozenset({"both", "global", "local"})


def resolve_acf_source(config, default: str = "residual") -> str:
    """Resolve the ACF source aliases with plan-key precedence.

    ``autocon_acf_source`` is the canonical spelling in JSON plans.  The
    historical ``AutoCon_acf_source`` spelling remains accepted for runtime
    namespaces and older artifacts, but it is consulted only when the
    canonical key is absent (or explicitly ``None``).
    """

    if isinstance(config, Mapping):
        value = config.get("autocon_acf_source")
        if value is None:
            value = config.get("AutoCon_acf_source", default)
    else:
        value = (
            getattr(config, "autocon_acf_source", None)
            if config is not None else None
        )
        if value is None:
            value = (
                getattr(config, "AutoCon_acf_source", default)
                if config is not None else default
            )
    return str(value).strip().lower()


def _as_bool(value):
    """Parse config booleans without treating the string ``'false'`` as true."""

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _config_value(config, name, default=None):
    """Read a flag from either a runtime namespace or a JSON mapping."""

    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _has_configured_acf_lags(config):
    """Return whether the configured ACF memory bank has any lag slots.

    The model treats an explicitly empty ``mamba_acf_lags`` sequence as a
    disabled forecast-memory branch.  Keep the predicate aligned with that
    construction behavior while remaining fail-closed for malformed values;
    argument/plan validators report malformed lag types separately.
    """

    raw = _config_value(config, "mamba_acf_lags", None)
    if raw is None:
        # Omitted values use the model's non-empty parser default.
        return True
    if isinstance(raw, (str, bytes)):
        text = raw.decode() if isinstance(raw, bytes) else raw
        return bool(text.replace(",", " ").split())
    try:
        return bool(list(raw))
    except (TypeError, ValueError):
        # Do not let a malformed config bypass a required-sidecar check.
        return True


def mamba_acf_prior_required(config) -> bool:
    """Return whether a MambaACF path consumes train-only ACF metadata.

    The predicate is shared by the CLI replay preflight and the experiment
    wrapper.  Keeping it here prevents an evaluation-only path from fitting a
    train prior when every ACF-dependent forecast/representation branch is
    disabled, while still requiring the sidecar for any live branch.
    """

    model = str(_config_value(config, "model", "")).strip()
    is_mamba_acf = model.startswith("MambaACF")
    is_mambats_acf = model.startswith(("MambaTSv4", "MambaTSv5"))
    is_nlinear_relation = (
        model == "NLinearVariableMambaAutoCon"
        and _as_bool(_config_value(config, "mamba_acf_relation_loss", False))
    )
    if not is_mamba_acf and not is_mambats_acf and not is_nlinear_relation:
        return False
    if is_nlinear_relation:
        return True
    # MambaTS v4/v5 use the compact prior only for forecast memory and the
    # optional representation adapter. They do not use MambaACF's complete
    # relation-loss table, so retain the latter's stricter branches below.
    if is_mambats_acf:
        return (
            _as_bool(_config_value(config, "mamba_use_acf_memory", False))
            and _has_configured_acf_lags(config)
        ) or (
            _as_bool(_config_value(config, "AutoCon", False))
            and _as_bool(_config_value(config, "mamba_acf_repr_fusion", False))
        )
    if _as_bool(_config_value(config, "mamba_acf_relation_loss", False)):
        return True
    # Dynamic ACF memory is recomputed from each evaluation window and does
    # not consume the train-only sidecar.  Only the static memory branch needs
    # a persisted prior here; other ACF-dependent branches below can still
    # require it independently.
    if (
        _as_bool(_config_value(config, "mamba_use_acf_memory", False))
        and not _as_bool(_config_value(config, "mamba_dynamic_acf", False))
        and _has_configured_acf_lags(config)
    ):
        return True
    for name in (
        "mamba_acf_anchor_copy",
        "mamba_acf_channel_gate",
        "mamba_acf_channel_conditioning",
        "mamba_acf_scale_conditioning",
    ):
        if _as_bool(_config_value(config, name, False)):
            return True
    return (
        _as_bool(_config_value(config, "AutoCon", False))
        and _as_bool(_config_value(config, "mamba_acf_repr_fusion", False))
    ) or str(
        _config_value(config, "mamba_acf_channel_order", "input")
    ).strip().lower() in {"strength", "similarity"}


def resolve_acf_relation_batching(args):
    """Resolve whether lag-pair batching is meaningful for the live loss.

    ``ACFRelationLoss`` has two different target contracts.  Hard-selected
    mode only uses the configured lag set when ``lag_mask`` is enabled.  Soft
    modes supervise all sufficiently correlated candidates; selected lags
    affect that objective only when an explicit positive boost is configured.
    Pairing windows at selected lags in the other cases changes the sampling
    distribution without changing the loss target, so the ordinary shuffled
    loader is the faithful default.

    The returned mapping is deliberately data-independent so it can be tested
    without constructing a CSV-backed dataset.
    """

    requested = str(
        getattr(args, "mamba_acf_relation_batching", "lag_pairs")
    ).strip().lower()
    target_mode = str(
        getattr(args, "mamba_acf_relation_target_mode", "hard_selected")
    ).strip().lower()
    lag_mask = _as_bool(getattr(args, "mamba_acf_relation_lag_mask", False))
    relation_loss = _as_bool(getattr(args, "mamba_acf_relation_loss", False))
    relation_scope = str(
        getattr(args, "mamba_acf_relation_scope", "both")
    ).strip().lower()
    model = str(getattr(args, "model", ""))
    try:
        selected_lag_boost = float(
            getattr(args, "mamba_acf_relation_selected_lag_boost", 0.0)
        )
    except (TypeError, ValueError):
        # Argument validation reports malformed values with its own precise
        # error.  Treating one as unsupported here avoids accidentally
        # enabling a sampler before that validation runs.
        selected_lag_boost = 0.0

    hard_selected_support = target_mode == "hard_selected" and lag_mask
    soft_selected_support = (
        target_mode in {"soft_multi_peak", "soft_full"}
        and selected_lag_boost > 0.0
    )
    selected_lag_support = bool(hard_selected_support or soft_selected_support)
    global_relation_support = relation_scope in {"both", "global"}

    if not relation_loss:
        reason = "relation_loss_disabled"
    elif model not in _ACF_RELATION_MODELS:
        reason = "non_mamba_acf_autocon_model"
    elif requested != "lag_pairs":
        reason = "requested_random_batching"
    elif relation_scope not in _ACF_RELATION_SCOPES:
        reason = "unsupported_relation_scope"
    elif not global_relation_support:
        reason = "local_scope_no_global_pairs"
    elif target_mode not in _ACF_RELATION_TARGET_MODES:
        reason = "unsupported_relation_target_mode"
    elif selected_lag_support:
        reason = (
            "hard_selected_lag_mask"
            if hard_selected_support else "soft_selected_lag_boost"
        )
    else:
        reason = "all_candidate_targets"

    use_lag_pairs = bool(
        relation_loss
        and model in _ACF_RELATION_MODELS
        and requested == "lag_pairs"
        and global_relation_support
        and selected_lag_support
    )
    return {
        "requested": requested,
        "effective": "lag_pairs" if use_lag_pairs else "shuffle",
        "use_lag_pairs": use_lag_pairs,
        "selected_lag_support": selected_lag_support,
        "target_mode": target_mode,
        "relation_scope": relation_scope,
        "lag_mask": lag_mask,
        "selected_lag_boost": selected_lag_boost,
        "reason": reason,
    }


__all__ = [
    "mamba_acf_prior_required",
    "resolve_acf_relation_batching",
    "resolve_acf_source",
]
