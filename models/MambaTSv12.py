"""MambaTS v12: independent two-scale Delta-Mamba with AutoCon support."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the independently implemented ms-Mamba-inspired v12 path."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v12")
        super().__init__(configs)


__all__ = ["Model"]
