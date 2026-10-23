"""MambaTS v9: decomposed single-scan Mamba with AutoCon fusion."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the v9 seasonal-Mamba and trend-linear architecture."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v9")
        super().__init__(configs)


__all__ = ["Model"]
