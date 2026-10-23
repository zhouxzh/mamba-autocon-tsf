"""TimeMixer baseline for long-term forecasting.

This is a compact, forecasting-only adaptation of the official TimeMixer
architecture.  It keeps its defining multi-scale seasonal bottom-up and trend
top-down MLP mixing, while using the repository's existing data embedding and
normalization conventions.
"""

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos


class _DFTDecomposition(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = max(1, int(top_k))

    def forward(self, x):
        spectrum = torch.fft.rfft(x, dim=1)
        magnitude = spectrum.abs()
        magnitude[:, 0, :] = 0
        keep = min(self.top_k, magnitude.shape[1])
        threshold = torch.topk(magnitude, keep, dim=1).values[:, -1:, :]
        seasonal = torch.fft.irfft(
            spectrum.masked_fill(magnitude < threshold, 0), n=x.shape[1], dim=1
        )
        return seasonal, x - seasonal


class _SeasonMixer(nn.Module):
    def __init__(self, lengths):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(high, low), nn.GELU(), nn.Linear(low, low))
            for high, low in zip(lengths[:-1], lengths[1:])
        )

    def forward(self, values):
        current = values[0]
        output = [current]
        for index, layer in enumerate(self.layers):
            current = values[index + 1] + layer(current.transpose(1, 2)).transpose(1, 2)
            output.append(current)
        return output


class _TrendMixer(nn.Module):
    def __init__(self, lengths):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(low, high), nn.GELU(), nn.Linear(high, high))
            for high, low in zip(reversed(lengths[:-1]), reversed(lengths[1:]))
        )

    def forward(self, values):
        reversed_values = list(reversed(values))
        current = reversed_values[0]
        output = [current]
        for index, layer in enumerate(self.layers):
            current = reversed_values[index + 1] + layer(
                current.transpose(1, 2)
            ).transpose(1, 2)
            output.append(current)
        return list(reversed(output))


class _PastDecomposableMixing(nn.Module):
    def __init__(self, configs, lengths):
        super().__init__()
        if configs.timemixer_decomp_method == "moving_avg":
            self.decomposition = series_decomp(configs.moving_avg)
        elif configs.timemixer_decomp_method == "dft_decomp":
            self.decomposition = _DFTDecomposition(configs.top_k)
        else:
            raise ValueError("timemixer_decomp_method must be moving_avg or dft_decomp")
        self.channel_independence = bool(configs.timemixer_channel_independence)
        self.cross = (
            nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff), nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
            )
            if not self.channel_independence else None
        )
        self.season_mixer = _SeasonMixer(lengths)
        self.trend_mixer = _TrendMixer(lengths)
        self.output = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff), nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, values):
        seasonal, trend = [], []
        for value in values:
            season, low_frequency = self.decomposition(value)
            if self.cross is not None:
                season, low_frequency = self.cross(season), self.cross(low_frequency)
            seasonal.append(season)
            trend.append(low_frequency)
        mixed_season = self.season_mixer(seasonal)
        mixed_trend = self.trend_mixer(trend)
        if self.channel_independence:
            return [original + self.output(season + trend)
                    for original, season, trend in zip(values, mixed_season, mixed_trend)]
        return [season + trend for season, trend in zip(mixed_season, mixed_trend)]


class Model(nn.Module):
    """Official TimeMixer-style multi-scale MLP forecasting baseline."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("TimeMixer currently supports forecasting tasks only")
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        self.down_window = int(configs.timemixer_down_sampling_window)
        self.down_layers = int(configs.timemixer_down_sampling_layers)
        self.channel_independence = bool(configs.timemixer_channel_independence)
        if self.down_window < 2 or self.down_layers < 0:
            raise ValueError("TimeMixer down-sampling window must be >=2 and layers >=0")
        self.lengths = [self.seq_len // (self.down_window ** index)
                        for index in range(self.down_layers + 1)]
        if min(self.lengths) < 2:
            raise ValueError("TimeMixer down-sampling leaves fewer than two time steps")
        self.embed = DataEmbedding_wo_pos(
            1 if self.channel_independence else self.channels,
            configs.d_model, configs.embed, configs.freq, configs.dropout,
        )
        self.blocks = nn.ModuleList(
            _PastDecomposableMixing(configs, self.lengths)
            for _ in range(max(1, int(configs.e_layers)))
        )
        self.predictors = nn.ModuleList(
            nn.Linear(length, self.pred_len) for length in self.lengths
        )
        self.projection = nn.Linear(configs.d_model, 1 if self.channel_independence else self.c_out)
        if not self.channel_independence:
            self.preprocess = series_decomp(configs.moving_avg)
            self.residual_layers = nn.ModuleList(nn.Linear(length, length) for length in self.lengths)
            self.regression_layers = nn.ModuleList(
                nn.Linear(length, self.pred_len) for length in self.lengths
            )
        else:
            self.preprocess = None
            self.residual_layers = None
            self.regression_layers = None

    def _normalise(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(torch.var(x - mean, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return (x - mean) / scale, mean, scale

    def _scales(self, x):
        values = [x]
        current = x.transpose(1, 2)
        for _ in range(self.down_layers):
            current = torch.nn.functional.avg_pool1d(current, self.down_window, self.down_window)
            values.append(current.transpose(1, 2))
        return values

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        normalized, mean, scale = self._normalise(x_enc)
        source_scales = self._scales(normalized)
        if self.channel_independence:
            batch = normalized.shape[0]
            encoded = []
            for value in source_scales:
                reshaped = value.transpose(1, 2).reshape(-1, value.shape[1], 1)
                encoded.append(self.embed(reshaped, None))
        else:
            seasonal, trend = zip(*(self.preprocess(value) for value in source_scales))
            encoded = [self.embed(value, None) for value in seasonal]
        for block in self.blocks:
            encoded = block(encoded)
        forecasts = []
        for index, value in enumerate(encoded):
            projected = self.predictors[index](value.transpose(1, 2)).transpose(1, 2)
            projected = self.projection(projected)
            if self.channel_independence:
                projected = projected.reshape(batch, self.channels, self.pred_len).transpose(1, 2)
            else:
                residual = trend[index].transpose(1, 2)
                residual = self.regression_layers[index](self.residual_layers[index](residual)).transpose(1, 2)
                projected = projected + residual
            forecasts.append(projected)
        return sum(forecasts) * scale[:, :, :self.c_out] + mean[:, :, :self.c_out]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
