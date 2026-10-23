"""MambaTS v14: full-coverage coarse scan with a local observed-window expert."""

from __future__ import annotations

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the one-scan local-before-patching path."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v14")
        super().__init__(configs)


__all__ = ["Model"]
