"""TimeMachine-style multi-path Mamba forecaster with an AutoCon interface.

The forecast path follows the public TimeMachine implementation: two feature
widths are projected from the look-back window and each width is processed by
two independent Mamba branches before a residual fusion head predicts the
whole horizon.  The repository's AutoCon objective receives a separate,
time-indexed representation derived from the first shared projection.  This
keeps the contrastive signal connected to the forecasting path without using
future targets or changing the plain inference output.

Reference implementation:
https://github.com/Atik-Ahamed/TimeMachine
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:  # pragma: no cover - availability depends on the active environment
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA1_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA1_IMPORT_ERROR = exc

try:  # pragma: no cover - availability depends on the active environment
    from mamba_ssm import Mamba2 as _Mamba2
    _MAMBA2_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba2 = None
    _MAMBA2_IMPORT_ERROR = exc


def _compatible_headdim(d_model, expand, requested):
    """Return the largest valid Mamba-2 head dimension not above requested."""

    expanded = int(d_model) * int(expand)
    requested = int(requested)
    if expanded <= 0 or requested <= 0:
        raise ValueError("Mamba dimensions must be positive")
    if expanded % requested == 0:
        return requested
    return max(
        divisor for divisor in range(min(expanded, requested), 0, -1)
        if expanded % divisor == 0
    )


class _ResidualMamba(nn.Module):
    """Pre-normalized residual Mamba block with a clear CPU failure mode."""

    def __init__(self, kind, d_model, d_state, d_conv, expand, headdim, dropout):
        super().__init__()
        if kind == "mamba1":
            mixer_cls = _Mamba1
            import_error = _MAMBA1_IMPORT_ERROR
        elif kind == "mamba2":
            mixer_cls = _Mamba2
            import_error = _MAMBA2_IMPORT_ERROR
        else:
            raise ValueError("kind must be mamba1 or mamba2")
        if mixer_cls is None:
            detail = " ({})".format(import_error) if import_error else ""
            raise ImportError(
                "MambaTimeMachine requires the optional mamba_ssm package "
                "{}{}".format(kind, detail)
            )
        self.norm = nn.LayerNorm(int(d_model))
        kwargs = {
            "d_model": int(d_model),
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
        }
        if kind == "mamba2":
            kwargs["headdim"] = _compatible_headdim(
                int(d_model), int(expand), int(headdim)
            )
        self.mixer = mixer_cls(**kwargs)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x):
        try:
            mixed = self.mixer(self.norm(x))
        except RuntimeError as exc:
            message = str(exc).lower()
            if not x.is_cuda and ("cuda" in message or "triton" in message):
                raise RuntimeError(
                    "The installed mamba_ssm build requires CUDA for "
                    "MambaTimeMachine; run with --use_gpu on a CUDA GPU."
                ) from exc
            raise
        return x + self.dropout(mixed)


class _TimeMachineModel(nn.Module):
    """Shared Mamba-1/Mamba-2 implementation used by the experiment registry."""

    supports_return_repr_control = True

    def __init__(self, configs, kind):
        super().__init__()
        self.kind = str(kind)
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if self.enc_in <= 0 or self.c_out <= 0 or self.c_out > self.enc_in:
            raise ValueError(
                "MambaTimeMachine requires 0 < c_out <= enc_in, got "
                "c_out={}, enc_in={}".format(self.c_out, self.enc_in)
            )

        self.AutoCon = bool(getattr(configs, "AutoCon", False))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "LastVal"))
        self.dropout_p = float(getattr(configs, "dropout", 0.1))
        self.n1 = int(getattr(configs, "mamba_tm_n1", 64))
        self.n2 = int(getattr(configs, "mamba_tm_n2", 128))
        self.repr_width = int(getattr(configs, "mamba_tm_repr_width", 64))
        self.repr_dims = int(getattr(configs, "d_ff", 128))
        self.channel_independent = bool(
            getattr(configs, "mamba_tm_channel_independent", False)
        )
        self.residual = bool(getattr(configs, "mamba_tm_residual", True))
        if min(self.n1, self.n2, self.repr_width, self.repr_dims) <= 0:
            raise ValueError("TimeMachine widths must be positive")

        d_state = int(getattr(configs, "mamba_d_state", 16))
        d_conv = int(getattr(configs, "mamba_d_conv", 4))
        expand = int(getattr(configs, "mamba_expand", 2))
        headdim = int(getattr(configs, "mamba_headdim", 16))

        # The official TimeMachine path uses two independent Mamba branches at
        # each of two projected widths.  In the multivariate mode the scan axis
        # is the variable axis; in channel-independent mode the paired branch
        # scans the compressed temporal axis instead.
        low_a_dim = self.n1
        low_b_dim = 1 if self.channel_independent else self.n1
        high_a_dim = self.n2
        high_b_dim = 1 if self.channel_independent else self.n2
        self.low_a = _ResidualMamba(
            self.kind, low_a_dim, d_state, d_conv, expand, headdim, self.dropout_p
        )
        self.low_b = _ResidualMamba(
            self.kind, low_b_dim, d_state, d_conv, expand, headdim, self.dropout_p
        )
        self.high_a = _ResidualMamba(
            self.kind, high_a_dim, d_state, d_conv, expand, headdim, self.dropout_p
        )
        self.high_b = _ResidualMamba(
            self.kind, high_b_dim, d_state, d_conv, expand, headdim, self.dropout_p
        )

        self.input_projection = nn.Linear(self.seq_len, self.n1)
        self.width_projection = nn.Linear(self.n1, self.n2)
        self.high_to_low = nn.Linear(self.n2, self.n1)
        self.forecast_head = nn.Linear(2 * self.n1, self.pred_len)

        # AutoCon sees a temporal grid.  It reuses the first forecast
        # projection, maps each compressed time token across variables, and
        # interpolates back to the original look-back length.  Plain models do
        # not construct this auxiliary branch at all: this keeps their memory,
        # parameter count, and resource measurements faithful to inference.
        if self.AutoCon:
            self.repr_channel_projection = nn.Linear(self.enc_in, self.repr_width)
            self.repr_norm = nn.LayerNorm(self.repr_width)
            self.repr_mamba = _ResidualMamba(
                self.kind,
                self.repr_width,
                d_state,
                d_conv,
                expand,
                headdim,
                self.dropout_p,
            )
            self.repr_projection = nn.Linear(self.repr_width, self.repr_dims)
            self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)
            self.repr_dropout = nn.Dropout(self.dropout_p)
        else:
            self.repr_channel_projection = None
            self.repr_norm = None
            self.repr_mamba = None
            self.repr_projection = None
            self.repr_head = None
            self.repr_dropout = None

    def _normalize(self, x_enc):
        mode = self.AutoCon_wnorm.lower()
        if mode == "meanstd":
            mean = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - mean
            scale = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            return centered / scale, (mean, scale)
        if mode == "revin":
            mean = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - mean
            scale = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            return centered / scale, (mean, scale)
        if mode == "mean":
            mean = x_enc.mean(dim=1, keepdim=True).detach()
            return x_enc - mean, mean
        if mode == "lastval":
            last = x_enc[:, -1:, :].detach()
            return x_enc - last, last
        raise ValueError(
            "MambaTimeMachine supports AutoCon_wnorm in "
            "{MeanStd, ReVIN, Mean, LastVal}; received {}".format(
                self.AutoCon_wnorm
            )
        )

    def _scan_pair(self, low, first, second):
        if not self.channel_independent:
            return first(low), second(low)
        batch, channels, width = low.shape
        first_out = first(low.reshape(batch * channels, 1, width)).reshape(
            batch, channels, width
        )
        second_input = low.reshape(batch * channels, width, 1)
        second_out = second(second_input).reshape(batch, channels, width)
        return first_out, second_out

    def _forecast_state(self, normalized):
        channel_tokens = normalized.transpose(1, 2).contiguous()
        low_input = self.input_projection(channel_tokens)
        low_a, low_b = self._scan_pair(low_input, self.low_a, self.low_b)
        high_input = self.width_projection(low_input)
        high_a, high_b = self._scan_pair(high_input, self.high_a, self.high_b)
        low_state = low_a + low_b
        high_state = high_a + high_b
        if self.residual:
            low_state = low_state + low_input
            high_state = high_state + high_input
        fused = torch.cat((low_state, self.high_to_low(high_state)), dim=-1)
        return self.forecast_head(fused).transpose(1, 2), low_input

    def _representation(self, low_input):
        if not self.AutoCon:
            raise RuntimeError(
                "MambaTimeMachine plain models do not expose an AutoCon "
                "representation; construct the AutoCon variant instead"
            )
        # [B, C, n1] -> [B, n1, C] gives AutoCon a temporal token axis.
        tokens = low_input.transpose(1, 2).contiguous()
        tokens = self.repr_channel_projection(tokens)
        tokens = tokens + self.repr_mamba(self.repr_norm(tokens))
        tokens = self.repr_projection(tokens)
        if tokens.shape[1] != self.seq_len:
            tokens = F.interpolate(
                tokens.transpose(1, 2),
                size=self.seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return self.repr_head(self.repr_dropout(tokens))

    @staticmethod
    def _restore(prediction, restore, c_out):
        if isinstance(restore, tuple):
            mean, scale = restore
            mean = mean[:, :, -c_out:]
            scale = scale[:, :, -c_out:]
            return prediction * scale + mean
        return prediction + restore[:, :, -c_out:]

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        onlyrepr=False,
        return_repr=None,
    ):
        normalized, restore = self._normalize(x_enc)
        raw_prediction, low_input = self._forecast_state(normalized)
        prediction = raw_prediction[:, :, -self.c_out:]
        prediction = self._restore(prediction, restore, self.c_out)

        need_repr = bool(
            onlyrepr
            or (self.AutoCon if return_repr is None else bool(return_repr))
        )
        representation = self._representation(low_input) if need_repr else None
        if onlyrepr:
            return None, representation
        if self.AutoCon:
            return prediction, representation
        return prediction

    def get_embeddings(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        onlyrepr=False,
    ):
        normalized, _ = self._normalize(x_enc)
        _, low_input = self._forecast_state(normalized)
        return self._representation(low_input)

    def get_embedddings(self, *args, **kwargs):
        """Compatibility alias used by older AutoCon callers."""

        return self.get_embeddings(*args, **kwargs)


class Model(_TimeMachineModel):
    """Default wrapper; concrete registry modules choose Mamba-1 or Mamba-2."""

    def __init__(self, configs):
        super().__init__(configs, getattr(configs, "mamba_tm_kind", "mamba2"))


__all__ = ["Model", "_TimeMachineModel"]
