"""MambaTS v11: VPT edge-cost training and greedy VAST inference order."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the single-scan learned transition-order architecture."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v11")
        super().__init__(configs)


__all__ = ["Model"]
