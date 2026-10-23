"""Mamba-1 backbone with the shared AutoConNet forecasting head.

``Mamba1`` and ``Mamba1AutoCon`` share this module.  The model returns a
prediction tensor for the plain forecasting objective and
``(prediction, representation)`` when ``configs.AutoCon`` is enabled, matching
the contract used by the AutoCon trainer.
"""

import torch.nn as nn

from models.Mamba2AutoCon import _MambaForecastModel


try:
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA1_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the local extension
    _Mamba1 = None
    _MAMBA1_IMPORT_ERROR = exc


class _Mamba1ResidualBlock(nn.Module):
    """Pre-normalized residual Mamba-1 block for stacked sequence encoding."""

    def __init__(self, d_model, d_state, d_conv, expand, headdim, dropout):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA1_IMPORT_ERROR) if _MAMBA1_IMPORT_ERROR else ""
            raise ImportError(
                "Mamba1AutoCon requires the optional 'mamba_ssm' package "
                "(Mamba-1){}; install it in the active training environment.".format(detail)
            )
        self.norm = nn.LayerNorm(d_model)
        self.mixer = _Mamba1(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        try:
            mixed = self.mixer(self.norm(x))
        except RuntimeError as exc:
            message = str(exc).lower()
            if not x.is_cuda and ("cuda" in message or "triton" in message):
                raise RuntimeError(
                    "The installed mamba_ssm/causal_conv1d build requires a "
                    "CUDA device; run Mamba1AutoCon with --use_gpu on a CUDA "
                    "GPU or install a CPU-capable Mamba-1 implementation."
                ) from exc
            raise
        return x + self.dropout(mixed)


class Model(_MambaForecastModel):
    """AutoConNet-compatible forecasting model with a Mamba-1 encoder."""

    residual_block_cls = _Mamba1ResidualBlock
    uses_mamba_headdim = False
