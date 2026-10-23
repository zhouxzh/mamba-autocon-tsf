"""MambaTS v5: patch Mamba with a causal train-only ACF anchor."""

from .MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """Select the v5 nested ACF-anchor variant without duplicating the model."""

    def __init__(self, configs):
        setattr(configs, "mamba_ts_variant", "v5")
        super().__init__(configs)


__all__ = ["Model"]
