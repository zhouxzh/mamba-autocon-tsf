"""MambaTS v7: DLinear base plus AutoCon-selected ACF memory."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the v7 nested linear-base and ACF-selector architecture."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v7")
        super().__init__(configs)


__all__ = ["Model"]
