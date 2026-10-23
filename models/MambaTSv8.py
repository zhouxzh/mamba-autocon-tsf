"""MambaTS v8: causal spectral residual with AutoCon-connected gating."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the v8 shared-scan Fourier residual architecture."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v8")
        super().__init__(configs)


__all__ = ["Model"]
