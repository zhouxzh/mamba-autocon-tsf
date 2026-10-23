"""MambaTS v13: one-scan lightweight multi-scale patch mixer."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the single-scan lightweight multi-scale residual path."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v13")
        super().__init__(configs)


__all__ = ["Model"]
