import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import DataEmbedding
from layers.dilated_conv import DilatedConvEncoder


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # Asymmetric endpoint padding keeps the output length equal to the
        # input for both odd and even moving-average kernels.
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        front = x[:, 0:1, :].repeat(1, left, 1)
        end = x[:, -1:, :].repeat(1, right, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_dims = configs.d_model
        self.repr_dims = configs.d_ff
        self.depth = configs.e_layers

        self.AutoCon = configs.AutoCon

        self.AutoCon_wnorm = configs.AutoCon_wnorm
        self.AutoCon_multiscales = configs.AutoCon_multiscales

        enc_in = 1  # Channel Independence

        self.enc_embedding = DataEmbedding(enc_in, self.hidden_dims, configs.embed, configs.freq, dropout=configs.dropout)
        self.feature_extractor = DilatedConvEncoder(
            self.hidden_dims,
            [self.hidden_dims] * self.depth + [self.repr_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)
        if not self.AutoCon:
            # Keep the module for checkpoint compatibility, but exclude the
            # unused contrastive projection from plain-model capacity counts.
            for parameter in self.repr_head.parameters():
                parameter.requires_grad_(False)

        self.ch_mlps = nn.ModuleList([nn.Linear(self.repr_dims, 1) for _ in self.AutoCon_multiscales])
        self.length_mlp = nn.Linear(self.seq_len, self.pred_len)
        self.trend_decoms = nn.ModuleList([series_decomp(kernel_size=dlen + 1) for dlen in self.AutoCon_multiscales])

        self.input_decom = series_decomp(25)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def _contrastive_repr(self, backbone_repr):
        """Project representations only for the auxiliary AutoCon objective."""
        return self.repr_head(backbone_repr) if self.AutoCon else backbone_repr

    def _normalize_window(self, x_enc):
        """Return normalized views and the values needed to restore predictions."""
        if self.AutoCon_wnorm == 'MeanStd':
            # Match PatchTST's per-window population statistics.  Keeping the
            # epsilon inside the square root makes this mode numerically
            # identical to the shared Mamba/AutoConNet implementations.
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            centered = x_enc - seq_mean
            seq_std = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            short_x = centered / seq_std
            return short_x, short_x.clone(), ('MeanStd', seq_mean, seq_std)
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            return short_x, short_x.clone(), ('ReVIN', seq_mean, seq_std)
        if self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = x_enc - seq_mean
            return short_x, short_x.clone(), ('Mean', seq_mean)
        if self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
            return short_x, long_x, ('Decomp',)
        if self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:, -1:, :].detach()
            short_x = x_enc - seq_last
            return short_x, short_x.clone(), ('LastVal', seq_last)
        raise ValueError(
            'Not Supported Window Normalization:{}. Use {{MeanStd | ReVIN | Mean | LastVal | Decomp}}.'.format(
                self.AutoCon_wnorm
            )
        )

    def _encode_channels(self, long_x, x_mark_enc):
        """Encode [B,T,C] in the same batch-major order as the forecast path."""
        batch, length, channels = long_x.shape
        values = long_x.permute(0, 2, 1).reshape(batch * channels, length, 1)
        if x_mark_enc is None:
            marks = None
        else:
            if x_mark_enc.ndim == 4:
                if x_mark_enc.shape[1] != 1:
                    raise ValueError(
                        'AutoConCI expects x_mark_enc with shape [B,T,F] or [B,1,T,F]'
                    )
                x_mark_enc = x_mark_enc[:, 0]
            if x_mark_enc.ndim != 3 or x_mark_enc.shape[0] != batch or x_mark_enc.shape[1] != length:
                raise ValueError('AutoConCI x_mark_enc must align with x_enc as [B,T,F]')
            marks = x_mark_enc.repeat_interleave(channels, dim=0)
        encoded = self.enc_embedding(values, marks)
        return self.repr_dropout(
            self.feature_extractor(encoded.transpose(1, 2))
        ).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        # x: [Batch, Input length, Channel]

        short_x, long_x, restore = self._normalize_window(x_enc)

        B, T, C = long_x.shape
        backbone_repr = self._encode_channels(long_x, x_mark_enc)
        contrastive_repr = self._contrastive_repr(backbone_repr)

        if onlyrepr:
            return None, contrastive_repr

        len_out = F.gelu(backbone_repr.transpose(1, 2))
        len_out = self.length_mlp(len_out).transpose(1, 2)  # (B * C, O, D)

        trend_outs = []
        for trend_decom, ch_mlp in zip(self.trend_decoms, self.ch_mlps):
            _, dec_out = trend_decom(len_out)
            dec_out = F.gelu(dec_out)
            dec_out = ch_mlp(dec_out)  # (B, I, D) > (B, I, C)
            _, trend_out = trend_decom(dec_out)
            trend_outs.append(trend_out)

        trend_outs = torch.stack(trend_outs, dim=-1).sum(dim=-1)
        trend_outs = trend_outs.reshape(B, C, self.pred_len).permute(0, 2, 1)
        season_out = self.Linear(short_x.permute(0, 2, 1)).permute(0, 2, 1)

        if restore[0] in {'ReVIN', 'MeanStd'}:
            scale = restore[2] if restore[0] == 'MeanStd' else restore[2] + 1e-5
            pred = (season_out + trend_outs) * scale + restore[1]
        elif restore[0] == 'Mean':
            pred = season_out + trend_outs + restore[1]
        elif restore[0] == 'Decomp':
            pred = season_out + trend_outs
        elif restore[0] == 'LastVal':
            pred = season_out + trend_outs + restore[1]
        else:
            raise Exception()

        if self.AutoCon:
            return pred, contrastive_repr  # [Batch, Output length, Channel]
        else:
            return pred


    def get_embeddings(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        """Return the same projected representation exposed by ``forward``."""
        _, long_x, _ = self._normalize_window(x_enc)
        return self._contrastive_repr(self._encode_channels(long_x, x_mark_enc))

    def get_embedddings(self, *args, **kwargs):
        """Backward-compatible alias for the historical misspelling."""
        return self.get_embeddings(*args, **kwargs)
