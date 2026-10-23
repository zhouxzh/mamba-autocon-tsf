"""MambaTS v6: DLinear-initialized Mamba residual correction."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Keep AutoCon's Mamba representation while nesting in DLinear at init."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v6")
        super().__init__(configs)


__all__ = ["Model"]
