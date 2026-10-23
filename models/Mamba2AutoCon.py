"""Mamba-2 backbone with the AutoConNet forecasting head.

The module intentionally keeps the public model contract used by
``Exp_Long_Term_Forecast_with_AutoCon``: when ``configs.AutoCon`` is true,
``forward`` returns ``(prediction, representation)``; otherwise it returns
the prediction only.  Mamba-2 operates on ``[batch, time, d_model]`` and the
existing AutoConNet trend/seasonal head is kept unchanged after projecting
the Mamba output to ``d_ff``.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.CausalEmbedding import CausalDataEmbedding
from layers.Embed import DataEmbedding


try:
    # Import lazily at construction time from the caller's environment.  The
    # optional dependency should not prevent the repository's other models
    # from being imported on a CPU-only or baseline environment.
    from mamba_ssm import Mamba2 as _Mamba2
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the local extension
    _Mamba2 = None
    _MAMBA_IMPORT_ERROR = exc


class moving_avg(nn.Module):
    """Moving-average block used by AutoConNet's series decomposition."""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Replicate the end points so the decomposition preserves sequence
        # length (the original AutoConNet implementation uses this scheme).
        # Split an even kernel's padding asymmetrically; otherwise floor
        # padding on both sides would silently shorten the trend sequence.
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        front = x[:, 0:1, :].repeat(1, left, 1)
        end = x[:, -1:, :].repeat(1, right, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Series decomposition into residual and moving-average trend."""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


def _compatible_headdim(d_model, expand, requested):
    """Return a valid Mamba2 head size for the expanded hidden dimension.

    Mamba2 requires ``expand * d_model`` to be divisible by ``headdim``.
    Existing AutoConNet experiments use very small ``d_model`` values, so a
    default head size that works for a larger model can otherwise fail during
    model construction.  An explicitly compatible value is left untouched;
    an incompatible value is reduced to the largest divisor not exceeding it
    and a warning records the effective setting in experiment logs.
    """

    expanded = int(d_model) * int(expand)
    requested = int(requested)
    if expanded <= 0:
        raise ValueError("mamba_expand * d_model must be positive")
    if requested <= 0:
        raise ValueError("mamba_headdim must be a positive integer")
    if expanded % requested == 0:
        return requested

    candidates = [d for d in range(min(requested, expanded), 0, -1) if expanded % d == 0]
    resolved = candidates[0]
    warnings.warn(
        "mamba_headdim={} is incompatible with expanded dimension {}; using "
        "headdim={} instead.".format(requested, expanded, resolved),
        UserWarning,
    )
    return resolved


class _Mamba2ResidualBlock(nn.Module):
    """Pre-normalized residual Mamba2 block for stacked sequence encoding."""

    def __init__(self, d_model, d_state, d_conv, expand, headdim, dropout):
        super().__init__()
        if _Mamba2 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError(
                "Mamba2AutoCon requires the optional 'mamba_ssm' package "
                "(Mamba2){}; install it in the active training environment.".format(detail)
            )
        self.norm = nn.LayerNorm(d_model)
        self.mixer = _Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        try:
            mixed = self.mixer(self.norm(x))
        except RuntimeError as exc:
            # Some mamba-ssm wheels expose only CUDA causal-conv/Triton
            # kernels.  Preserve the original exception as the cause while
            # making an accidental CPU run immediately actionable.
            message = str(exc).lower()
            if not x.is_cuda and ("cuda" in message or "triton" in message):
                raise RuntimeError(
                    "The installed mamba_ssm/causal_conv1d build requires a "
                    "CUDA device; run Mamba2AutoCon with --use_gpu on a CUDA "
                    "GPU or install a CPU-capable Mamba2 implementation."
                ) from exc
            raise
        return x + self.dropout(mixed)


class _MambaForecastModel(nn.Module):
    """Shared AutoConNet-compatible forecasting model for Mamba encoders."""

    residual_block_cls = _Mamba2ResidualBlock
    uses_mamba_headdim = True
    # Callers that only need the deployed forecast can disable the auxiliary
    # AutoCon projection without changing the default forward contract.
    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_dims = configs.d_model
        self.repr_dims = configs.d_ff
        self.depth = max(1, int(getattr(configs, "e_layers", 1)))

        self.AutoCon = getattr(configs, "AutoCon", True)
        self.AutoCon_wnorm = getattr(configs, "AutoCon_wnorm", "LastVal")
        multiscales = getattr(configs, "AutoCon_multiscales", [96])
        self.AutoCon_multiscales = (
            [int(multiscales)] if isinstance(multiscales, int) else list(multiscales)
        )
        dropout = float(getattr(configs, "dropout", 0.1))

        # Mamba-specific knobs are kept separate from the repository's
        # d_model/d_ff settings so experiments can report them explicitly.
        self.mamba_d_state = int(getattr(configs, "mamba_d_state", 64))
        self.mamba_d_conv = int(getattr(configs, "mamba_d_conv", 4))
        self.mamba_expand = int(getattr(configs, "mamba_expand", 2))
        requested_headdim = int(getattr(configs, "mamba_headdim", 16))
        self.mamba_headdim = (
            _compatible_headdim(self.hidden_dims, self.mamba_expand, requested_headdim)
            if self.uses_mamba_headdim else None
        )

        # Keep the historical circular value embedding by default for exact
        # backward compatibility.  When explicitly requested, use the shared
        # left-padded embedding so the public causal flag is effective for
        # plain Mamba-1/Mamba-2 models as well as MambaACF.
        self.causal_embedding = bool(
            getattr(configs, "mamba_causal_embedding", False)
        )
        embedding_cls = CausalDataEmbedding if self.causal_embedding else DataEmbedding
        self.enc_embedding = embedding_cls(
            configs.enc_in,
            self.hidden_dims,
            getattr(configs, "embed", "timeF"),
            getattr(configs, "freq", "h"),
            wo_freq=getattr(configs, "wo_freq", ""),
            dropout=dropout,
        )
        self.feature_extractor = nn.ModuleList(
            [
                self.residual_block_cls(
                    self.hidden_dims,
                    self.mamba_d_state,
                    self.mamba_d_conv,
                    self.mamba_expand,
                    self.mamba_headdim,
                    dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.repr_projection = (
            nn.Linear(self.hidden_dims, self.repr_dims)
            if self.hidden_dims != self.repr_dims
            else nn.Identity()
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        # Projection used only for the AutoCon representation contract.  The
        # forecasting head continues to consume the pre-projection backbone
        # representation, so enabling AutoCon does not perturb its input.
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)
        if not self.AutoCon:
            # Preserve the state-dict key while excluding a dead projection
            # from plain-model trainable capacity and optimizer updates.
            for parameter in self.repr_head.parameters():
                parameter.requires_grad_(False)

        # Existing AutoConNet prediction head (multi-scale trend + seasonal
        # linear extrapolation), unchanged apart from the encoder above.
        self.ch_mlps = nn.ModuleList(
            [nn.Linear(self.repr_dims, self.c_out) for _ in self.AutoCon_multiscales]
        )
        self.length_mlp = nn.Linear(self.seq_len, self.pred_len)
        self.trend_decoms = nn.ModuleList(
            [series_decomp(kernel_size=dlen + 1) for dlen in self.AutoCon_multiscales]
        )

        self.input_decom = series_decomp(25)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        # Construct this optional branch after every legacy prediction layer.
        # The ordering matters: with the same seed, all common control/fusion
        # parameters must start bit-identically for a paired screen to make a
        # causal statement about the new branch.
        self.mamba2_repr_forecast_fusion_enabled = bool(
            self.AutoCon
            and getattr(configs, "mamba2_repr_forecast_fusion", False)
        )
        if self.mamba2_repr_forecast_fusion_enabled:
            self.mamba2_repr_forecast_time = nn.Linear(
                self.seq_len, self.pred_len, bias=False
            )
            self.mamba2_repr_forecast_channels = nn.Linear(
                self.repr_dims, self.c_out, bias=False
            )
            self.mamba2_repr_forecast_scale = nn.Parameter(
                torch.zeros(1, 1, self.c_out)
            )
        else:
            self.mamba2_repr_forecast_time = None
            self.mamba2_repr_forecast_channels = None
            self.register_parameter("mamba2_repr_forecast_scale", None)

    def _normalize_window(self, x_enc):
        """Apply the same window normalization choices as AutoConNet."""

        if self.AutoCon_wnorm == "MeanStd":
            # Exact population-statistic parity with PatchTST.  This mode is
            # explicit so the historical ReVIN denominator remains stable.
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - seq_mean
            seq_std = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            short_x = centered / seq_std
            long_x = short_x.clone()
            restore = (seq_mean, seq_std)
        elif self.AutoCon_wnorm == "ReVIN":
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
            restore = (seq_mean, seq_std)
        elif self.AutoCon_wnorm == "Mean":
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = x_enc - seq_mean
            long_x = short_x.clone()
            restore = seq_mean
        elif self.AutoCon_wnorm == "Decomp":
            short_x, long_x = self.input_decom(x_enc)
            restore = None
        elif self.AutoCon_wnorm == "LastVal":
            seq_last = x_enc[:, -1:, :].detach()
            short_x = x_enc - seq_last
            long_x = short_x.clone()
            restore = seq_last
        else:
            raise ValueError(
                "Not Supported Window Normalization:{}. Use {{MeanStd | ReVIN | Mean | LastVal | Decomp}}.".format(
                    self.AutoCon_wnorm
                )
            )
        return short_x, long_x, restore

    def _encode(self, long_x, x_mark_enc):
        enc_out = self.enc_embedding(long_x, x_mark_enc)
        for block in self.feature_extractor:
            enc_out = block(enc_out)
        return self.repr_dropout(self.repr_projection(enc_out))

    def _contrastive_repr(self, backbone_repr):
        """Return the representation exposed to the AutoCon objective."""
        return self.repr_head(backbone_repr) if self.AutoCon else backbone_repr

    def _forecast_from_repr(self, repr, short_x, restore):
        # Preserve AutoConNet's trend head exactly: representation is projected
        # across time first, then each requested trend scale is channel-mixed.
        len_out = F.gelu(repr.transpose(1, 2))
        len_out = self.length_mlp(len_out).transpose(1, 2)

        trend_outs = []
        for trend_decom, ch_mlp in zip(self.trend_decoms, self.ch_mlps):
            _, dec_out = trend_decom(len_out)
            dec_out = F.gelu(dec_out)
            dec_out = ch_mlp(dec_out)
            _, trend_out = trend_decom(dec_out)
            trend_outs.append(trend_out)

        # An empty multiscale list is not useful for AutoCon, but handling it
        # explicitly gives a clear error instead of an opaque torch.stack one.
        if not trend_outs:
            raise ValueError("AutoCon_multiscales must contain at least one scale")
        trend_outs = torch.stack(trend_outs, dim=-1).sum(dim=-1)
        season_out = self.Linear(short_x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.AutoCon_wnorm in {"ReVIN", "MeanStd"}:
            seq_mean, seq_std = restore
            scale = seq_std if self.AutoCon_wnorm == "MeanStd" else seq_std + 1e-5
            return (season_out + trend_outs) * scale + seq_mean
        if self.AutoCon_wnorm == "Mean":
            return season_out + trend_outs + restore
        if self.AutoCon_wnorm == "Decomp":
            return season_out + trend_outs
        if self.AutoCon_wnorm == "LastVal":
            return season_out + trend_outs + restore
        raise RuntimeError("Unexpected window normalization mode")

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        onlyrepr=False,
        return_repr=None,
    ):
        # x_enc: [Batch, Input length, Channel]
        short_x, long_x, restore = self._normalize_window(x_enc)
        backbone_repr = self._encode(long_x, x_mark_enc)
        # ``None`` preserves the historical behavior: AutoCon models return
        # their representation, while plain models do not expose one.  An
        # explicit ``False`` is used by validation/deployment to skip the
        # auxiliary projection head.
        need_repr = bool(onlyrepr or (
            self.AutoCon if return_repr is None else return_repr
        ))
        contrastive_repr = (
            self._contrastive_repr(backbone_repr)
            if need_repr or self.mamba2_repr_forecast_fusion_enabled
            else None
        )

        if onlyrepr:
            return None, contrastive_repr

        pred = self._forecast_from_repr(backbone_repr, short_x, restore)
        if self.mamba2_repr_forecast_fusion_enabled:
            if contrastive_repr is None:
                raise RuntimeError(
                    "Mamba2 representation forecast fusion requires AutoCon features"
                )
            residual = self.mamba2_repr_forecast_time(
                contrastive_repr.transpose(1, 2)
            ).transpose(1, 2)
            residual = self.mamba2_repr_forecast_channels(residual)
            pred = pred + self.mamba2_repr_forecast_scale * residual
        if self.AutoCon and need_repr:
            return pred, contrastive_repr
        if self.AutoCon:
            return pred, None
        return pred

    def get_embeddings(
        self,
        x_enc,
        x_mark_enc,
        x_dec=None,
        x_mark_dec=None,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        onlyrepr=False,
    ):
        """Return the time-indexed representation used by AutoCon."""

        _, long_x, _ = self._normalize_window(x_enc)
        return self._contrastive_repr(self._encode(long_x, x_mark_enc))

    # Match the misspelled compatibility method in the original AutoConCI.
    def get_embedddings(
        self,
        x_enc,
        x_mark_enc,
        x_dec=None,
        x_mark_dec=None,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        onlyrepr=False,
    ):
        return self.get_embeddings(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            enc_self_mask,
            dec_self_mask,
            dec_enc_mask,
            onlyrepr,
        )


class Model(_MambaForecastModel):
    """AutoConNet-compatible forecasting model with a Mamba-2 encoder."""

    residual_block_cls = _Mamba2ResidualBlock
    uses_mamba_headdim = True
