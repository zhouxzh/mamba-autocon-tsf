"""MambaTS v4: shared scan plus train-only causal ACF memory."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """MambaTS v3 backbone with a gated static ACF forecast residual."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v4")
        super().__init__(configs)


__all__ = ["Model"]
