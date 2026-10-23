"""MambaTS v2 architecture wrapper.

The implementation lives in :mod:`models.MambaTS` so the forecast and
representation contracts stay identical.  This wrapper only selects the v2
variant, which keeps v1 checkpoints and comparisons unambiguous.
"""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """MambaTS with independent temporal scans and a gated persistence path."""

    def __init__(self, configs):
        # The experiment namespace is intentionally mutable throughout this
        # repository; recording the effective variant here makes config.json
        # and runtime identity agree with the constructed module.
        setattr(configs, "mamba_ts_variant", "v2")
        super().__init__(configs)


__all__ = ["Model"]
