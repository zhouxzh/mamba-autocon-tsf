"""Canonical labels for persisted forecasting-model identities.

The training-plan runner has a small number of arm aliases whose executable
module intentionally differs from their registered model name.  In
particular, variable-token MambaACF arms execute ``MambaACF`` or
``MambaACFAutoCon`` and carry the registered alias in ``plan_job_id``.  Result
collectors must retain that distinction or they can pool different arms.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


EXECUTABLE_MODEL_ALIASES = {
    "MambaACFVariable": "MambaACF",
    "MambaACFVariableAutoCon": "MambaACFAutoCon",
    "MambaACFLiteAutoCon": "MambaACFAutoCon",
}


def as_bool(value: Any, default: bool = False) -> bool:
    """Parse persisted CLI booleans without treating ``"false"`` as true."""

    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _job_has_alias(plan_job_id: str, alias: str) -> bool:
    """Match an alias as a complete runner job-id token."""

    return re.search(rf"(?:^|_){re.escape(alias)}(?:_|$)", plan_job_id) is not None


def canonical_model(config_or_name: Any, autocon: Any = None) -> str:
    """Return a reporting key while preserving runner-level arm aliases.

    The registered MambaACF arms are not independent Python modules.  They
    resolve to the normal executable models with a configuration override.
    Recover an alias only when its immutable plan-job token agrees; this avoids
    relabelling unrelated exploratory experiments.
    """

    getter = getattr(config_or_name, "get", None)
    if isinstance(config_or_name, Mapping) or callable(getter):
        name = str(getter("model", "unknown"))
        # CSV/report loaders historically used the lower-case ``autocon``
        # spelling; persisted run configs use ``AutoCon``.
        auto_value = getter("AutoCon", getter("autocon", False))
        enabled = as_bool(auto_value)
        plan_job_id = str(getter("plan_job_id", ""))
        variable_mixer = as_bool(getter("mamba_variable_mixer", False))

        if variable_mixer:
            if (
                name == "MambaACFAutoCon"
                and enabled
                and _job_has_alias(plan_job_id, "MambaACFVariableAutoCon")
            ):
                return "MambaACFVariableAutoCon"
            if (
                name == "MambaACF"
                and not enabled
                and _job_has_alias(plan_job_id, "MambaACFVariable")
            ):
                return "MambaACFVariable"
        if (
            name == "MambaACFAutoCon"
            and enabled
            and _job_has_alias(plan_job_id, "MambaACFLiteAutoCon")
        ):
            return "MambaACFLiteAutoCon"
    else:
        name = str(config_or_name)
        enabled = as_bool(autocon, False)

    if enabled and not name.endswith("AutoCon"):
        name += "AutoCon"
    return name


def executable_model(config_or_name: Any, autocon: Any = None) -> str:
    """Map a canonical reporting name to its Python executable model name."""

    canonical = canonical_model(config_or_name, autocon)
    return EXECUTABLE_MODEL_ALIASES.get(canonical, canonical)
