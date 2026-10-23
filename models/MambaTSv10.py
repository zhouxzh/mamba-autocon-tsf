"""MambaTS v10: single-scan VAST-lite ordering with AutoCon fusion."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the fixed train-ACF channel ordering architecture."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v10")
        super().__init__(configs)


__all__ = ["Model"]
