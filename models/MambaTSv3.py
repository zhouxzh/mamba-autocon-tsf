"""MambaTS v3 wrapper for the shared-scan cross-variable candidate."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Shared MambaTS scan with low-rank context and channel gates."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v3")
        super().__init__(configs)


__all__ = ["Model"]
