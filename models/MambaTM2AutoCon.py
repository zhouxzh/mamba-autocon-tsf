"""TimeMachine-style four-path Mamba-2 forecaster."""

from models.MambaTimeMachineAutoCon import _TimeMachineModel


class Model(_TimeMachineModel):
    def __init__(self, configs):
        super().__init__(configs, "mamba2")


__all__ = ["Model"]
