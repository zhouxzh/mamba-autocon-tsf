"""Low-overhead anchor-conditioned single-scan MambaTS v16 wrapper."""

from models.MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """MambaTS v16 binds the direct patch-start calendar conditioning."""

    def __init__(self, configs):
        if str(getattr(configs, "mamba_ts_variant", "v16")).lower() != "v16":
            raise ValueError("MambaTSv16 requires mamba_ts_variant=v16")
        super().__init__(configs)


__all__ = ["Model"]
