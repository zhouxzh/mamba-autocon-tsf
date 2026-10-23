"""Mamba-1 variant of the ACF-guided forecasting model."""

from models.MambaACF import _MambaACFBase


class Model(_MambaACFBase):
    """Use the Mamba-1 kernel with the shared ACF-guided architecture."""

    mamba_kind = "mamba1"


__all__ = ["Model"]
