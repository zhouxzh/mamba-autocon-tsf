"""Time-conditioned single-scan MambaTS v15 wrapper."""

from models.MambaTS import Model as _MambaTSModel


class Model(_MambaTSModel):
    """MambaTS v15 binds the calendar-conditioned single-scan variant."""

    def __init__(self, configs):
        if str(getattr(configs, "mamba_ts_variant", "v15")).lower() != "v15":
            raise ValueError("MambaTSv15 requires mamba_ts_variant=v15")
        super().__init__(configs)


__all__ = ["Model"]
