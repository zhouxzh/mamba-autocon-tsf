"""NLinear base with one variable-axis Mamba scan and AutoCon support.

Each variable's complete observed window is embedded as one token and a single
Mamba-1 scan mixes variables.  A shared NLinear-style forecast supplies the
base prediction.  The Mamba route is a ReZero residual, while AutoCon reads a
time grid conditioned on the same variable-token state.  This deliberately
omits decomposition, reverse scans, and FFNs so its resource cost can be
compared directly with compact inverted-token baselines.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils.acf_channel_order import signed_acf_profile_similarity_order

try:  # pragma: no cover - depends on the local CUDA installation
    from mamba_ssm import Mamba as _Mamba1
    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    _Mamba1 = None
    _MAMBA_IMPORT_ERROR = exc


class Model(nn.Module):
    """Shared NLinear forecast plus a one-scan variable-token Mamba residual."""

    supports_return_repr_control = True

    def __init__(self, configs):
        super().__init__()
        if _Mamba1 is None:
            detail = " ({})".format(_MAMBA_IMPORT_ERROR) if _MAMBA_IMPORT_ERROR else ""
            raise ImportError("NLinearVariableMamba requires mamba_ssm Mamba-1{}".format(detail))
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.c_out = int(configs.c_out)
        self.d_model = int(getattr(configs, "d_model", 32))
        self.d_ff = int(getattr(configs, "d_ff", 64))
        self.AutoCon = bool(getattr(configs, "AutoCon", False))
        self.AutoCon_wnorm = str(getattr(configs, "AutoCon_wnorm", "LastVal"))
        self.use_acf_memory = bool(getattr(configs, "mamba_use_acf_memory", False))
        self.variable_repr_fusion_enabled = bool(
            self.AutoCon and getattr(configs, "mamba_variable_repr_fusion", False)
        )
        self.variable_mixer_enabled = bool(
            getattr(configs, "mamba_variable_mixer", False)
        )
        self.bidirectional_enabled = bool(
            getattr(configs, "mamba_bidirectional", False)
        )
        self.acf_channel_order = str(
            getattr(configs, "mamba_acf_channel_order", "input")
        ).lower()
        if self.acf_channel_order not in {"input", "strength", "similarity"}:
            raise ValueError(
                "mamba_acf_channel_order must be input, strength, or similarity"
            )
        if min(self.seq_len, self.pred_len, self.enc_in, self.c_out, self.d_model, self.d_ff) <= 0:
            raise ValueError("NLinearVariableMamba dimensions must be positive")
        if self.c_out > self.enc_in:
            raise ValueError("NLinearVariableMamba requires c_out <= enc_in")

        d_state = max(1, min(8, int(getattr(configs, "mamba_d_state", 16))))
        d_conv = max(1, min(2, int(getattr(configs, "mamba_d_conv", 4))))
        residual_init = float(getattr(configs, "mamba_ts_residual_init", 0.0))
        if not -0.99 < residual_init < 0.99:
            raise ValueError("mamba_ts_residual_init must be in (-0.99, 0.99)")

        self.base_head = nn.Linear(self.seq_len, self.pred_len)
        nn.init.constant_(self.base_head.weight, 1.0 / self.seq_len)
        nn.init.zeros_(self.base_head.bias)
        self.temporal_embedding = nn.Linear(self.seq_len, self.d_model)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.mamba = _Mamba1(
            d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=1
        )
        self.reverse_mamba = (
            _Mamba1(d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=1)
            if self.bidirectional_enabled else None
        )
        if self.variable_mixer_enabled:
            # A factored channel map supplies order-robust cross-variable
            # interaction without materialising a dense C x C matrix.  The
            # ReZero scale keeps the original variable-axis Mamba as the
            # exact initialization and lets the forecast loss open this path.
            mixer_rank = max(
                1,
                min(
                    self.enc_in - 1 if self.enc_in > 1 else 1,
                    int(getattr(configs, "mamba_variable_gdd_mixer_rank", 8)),
                ),
            )
            self.variable_mixer_input = nn.Parameter(
                torch.empty(mixer_rank, self.enc_in)
            )
            self.variable_mixer_output = nn.Parameter(
                torch.empty(self.enc_in, mixer_rank)
            )
            self.variable_mixer_scale = nn.Parameter(torch.tensor(0.0))
            nn.init.normal_(self.variable_mixer_input, mean=0.0, std=0.02)
            nn.init.normal_(self.variable_mixer_output, mean=0.0, std=0.02)
        else:
            self.variable_mixer_input = None
            self.variable_mixer_output = None
            self.variable_mixer_scale = None
        self.variable_head = nn.Linear(self.d_model, self.pred_len)
        self.forecast_scale = nn.Parameter(torch.tensor(residual_init))
        # Populated once from the training split by the experiment wrapper.
        # Empty means the optional causal ACF memory is inactive.
        self.register_buffer("acf_prior", torch.empty(0), persistent=False)
        self.register_buffer("acf_full_prior", torch.empty(0), persistent=False)
        self.register_buffer("acf_prior_lags", torch.empty(0, dtype=torch.long), persistent=False)
        input_order = torch.arange(self.enc_in, dtype=torch.long)
        self.register_buffer("acf_channel_order_indices", input_order.clone(), persistent=False)
        self.register_buffer("acf_channel_inverse_indices", input_order.clone(), persistent=False)
        self.acf_memory_scale = nn.Parameter(torch.tensor(0.0))

        if self.AutoCon:
            self.repr_norm = nn.LayerNorm(self.d_model)
            self.repr_projection = nn.Linear(self.d_model, self.d_ff)
            self.repr_head = nn.Linear(self.d_ff, self.d_ff)
            if self.variable_repr_fusion_enabled:
                # A global temporal representation modulates the shared
                # variable-token state.  The scalar starts at zero so fusion
                # is a strict ReZero extension of the base forecast.
                self.repr_fusion_gate = nn.Linear(self.d_ff, self.d_model)
                self.repr_fusion_scale = nn.Parameter(torch.tensor(0.0))
            else:
                self.repr_fusion_gate = None
                self.repr_fusion_scale = None
        else:
            self.repr_norm = None
            self.repr_projection = None
            self.repr_head = None
            self.repr_fusion_gate = None
            self.repr_fusion_scale = None

        # When the train-only ACF memory and prediction-connected
        # representation fusion are both enabled, let the same representation
        # select whether the static memory should be trusted at each forecast
        # step/channel.  The two-step horizon basis keeps the adapter small
        # for long horizons, and the zero ReZero scale makes this an exact
        # nested extension of the existing memory path at initialization.
        self.acf_memory_selector_enabled = bool(
            self.AutoCon
            and self.use_acf_memory
            and getattr(configs, "mamba_variable_acf_memory_selector", False)
        )
        if self.acf_memory_selector_enabled:
            self.acf_memory_horizon_selector = nn.Linear(self.d_ff, 2)
            self.acf_memory_channel_selector = nn.Linear(self.d_model, 1)
            self.acf_memory_selector_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.acf_memory_horizon_selector = None
            self.acf_memory_channel_selector = None
            self.register_parameter("acf_memory_selector_scale", None)

    def _normalize(self, x):
        mode = self.AutoCon_wnorm.lower()
        if mode == "lastval":
            last = x[:, -1:, :].detach()
            return x - last, last
        if mode in {"meanstd", "revin"}:
            mean = x.mean(dim=1, keepdim=True).detach()
            centered = x - mean
            scale = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            return centered / scale, (mean, scale)
        if mode == "mean":
            mean = x.mean(dim=1, keepdim=True).detach()
            return x - mean, mean
        raise ValueError("NLinearVariableMamba supports LastVal, MeanStd, ReVIN, or Mean")

    def _restore(self, prediction, restore):
        if isinstance(restore, tuple):
            mean, scale = restore
            return prediction * scale[:, :, -self.c_out:] + mean[:, :, -self.c_out:]
        return prediction + restore[:, :, -self.c_out:]

    @staticmethod
    def _restore_history(normalized, restore):
        """Recover the full observed history for raw-scale ACF retrieval."""
        if isinstance(restore, tuple):
            mean, scale = restore
            return normalized * scale + mean
        return normalized + restore

    def _tokens(self, normalized):
        tokens = self.temporal_embedding(normalized.transpose(1, 2))
        order, inverse = self._train_prior_channel_order(tokens.device)
        if order is not None:
            tokens = tokens.index_select(1, order)
        if self.variable_mixer_enabled:
            latent = torch.einsum(
                "rc,bcd->brd", self.variable_mixer_input, tokens
            )
            cross_variable = torch.einsum(
                "cr,brd->bcd", self.variable_mixer_output, latent
            )
            tokens = tokens + torch.tanh(self.variable_mixer_scale) * cross_variable
        try:
            hidden = self.token_norm(tokens)
            forward = self.mamba(hidden)
            if self.bidirectional_enabled:
                reverse = torch.flip(
                    self.reverse_mamba(torch.flip(hidden, dims=(1,))), dims=(1,)
                )
                tokens = tokens + 0.5 * (forward + reverse)
            else:
                tokens = tokens + forward
            if inverse is not None:
                tokens = tokens.index_select(1, inverse)
            return tokens
        except RuntimeError as exc:
            if not tokens.is_cuda and ("cuda" in str(exc).lower() or "triton" in str(exc).lower()):
                raise RuntimeError("NLinearVariableMamba requires a CUDA-compatible mamba_ssm build") from exc
            raise

    def _representation(self, normalized, tokens):
        if not self.AutoCon:
            raise RuntimeError("plain NLinearVariableMamba does not expose an AutoCon representation")
        # Each time state is the normalized observed channel vector projected
        # onto the same Mamba token states consumed by ``variable_head``.  It
        # deliberately has no independent input encoder: AutoCon gradients
        # must enter the variable-token Mamba route that corrects the forecast.
        latent = torch.bmm(normalized, tokens) / float(self.enc_in) ** 0.5
        latent = self.repr_norm(latent)
        return self.repr_head(self.repr_projection(latent))

    def set_acf_prior(self, acf_values, lags):
        """Attach a signed, train-only compact ACF prior for forecast memory."""
        prior = torch.as_tensor(acf_values, dtype=torch.float32)
        lag_tensor = torch.as_tensor(list(lags), dtype=torch.long)
        if prior.ndim == 1:
            prior = prior.unsqueeze(0).expand(self.enc_in, -1).clone()
        if prior.ndim != 2 or prior.shape[0] != self.enc_in:
            raise ValueError("NLinearVariableMamba ACF prior must be [enc_in, lag]")
        if lag_tensor.numel() == 0:
            raise ValueError("NLinearVariableMamba ACF prior requires at least one lag")
        full_prior = prior.detach().clone()
        if prior.shape[1] != lag_tensor.numel():
            # ``fit_signed_acf_prior`` exposes the complete train-only table;
            # accept it and retain only the configured lag columns.
            if int(lag_tensor.max()) >= prior.shape[1]:
                raise ValueError("NLinearVariableMamba ACF prior/lags mismatch")
            prior = prior.index_select(1, lag_tensor)
        elif int(lag_tensor.max()) >= prior.shape[1]:
            raise ValueError("NLinearVariableMamba ACF prior does not contain every lag")
        if not torch.isfinite(prior).all() or (lag_tensor <= 0).any():
            raise ValueError("NLinearVariableMamba ACF prior must be finite and positive-lag")
        self.acf_prior = prior.detach()
        self.acf_full_prior = full_prior
        self.acf_prior_lags = lag_tensor.detach()
        self._set_acf_channel_order(self.acf_prior, self.acf_prior_lags)

    def set_compact_acf_prior(self, acf_values, lags, full_acf_values=None):
        """Attach a sidecar compact prior without recomputing train statistics."""
        selected = torch.as_tensor(acf_values, dtype=torch.float32)
        if selected.ndim == 1:
            selected = selected.unsqueeze(0).expand(self.enc_in, -1).clone()
        if selected.ndim != 2 or selected.shape[0] != self.enc_in:
            raise ValueError("compact ACF prior must have shape [enc_in, selected_lags]")
        lag_tensor = torch.as_tensor(list(lags), dtype=torch.long)
        if selected.shape[1] != lag_tensor.numel() or lag_tensor.numel() == 0:
            raise ValueError("compact ACF prior lag metadata is inconsistent")
        if (lag_tensor <= 0).any() or not torch.isfinite(selected).all():
            raise ValueError("compact ACF prior must be finite and positive-lag")
        order = torch.argsort(lag_tensor)
        selected = selected.index_select(1, order)
        lag_tensor = lag_tensor.index_select(0, order)
        if full_acf_values is None:
            full = selected.detach().clone()
        else:
            full = torch.as_tensor(full_acf_values, dtype=torch.float32)
            if full.ndim == 1:
                full = full.unsqueeze(0).expand(self.enc_in, -1).clone()
            if full.ndim != 2 or full.shape[0] != self.enc_in:
                raise ValueError("full ACF prior must have shape [enc_in, measured_lags]")
            if int(lag_tensor.max()) >= full.shape[1] or not torch.isfinite(full).all():
                raise ValueError("full ACF prior does not contain every selected lag")
            expected = full.index_select(1, lag_tensor)
            if not torch.allclose(expected, selected, atol=1e-5, rtol=0.0):
                raise ValueError("compact and full ACF prior values are inconsistent")
        self.acf_prior = selected.detach()
        self.acf_full_prior = full.detach()
        self.acf_prior_lags = lag_tensor.detach()
        self._set_acf_channel_order(self.acf_prior, self.acf_prior_lags)

    def _set_acf_channel_order(self, selected, lags):
        """Resolve a deterministic train-prior variable order once per fit."""

        input_order = torch.arange(
            self.enc_in, device=selected.device, dtype=torch.long
        )
        if self.acf_channel_order == "strength":
            peaks = selected.abs().amax(dim=-1).detach().cpu().tolist()
            peak_index = selected.abs().argmax(dim=-1)
            peak_lags = lags.index_select(0, peak_index).detach().cpu().tolist()
            ordered = sorted(
                range(self.enc_in),
                key=lambda index: (-float(peaks[index]), int(peak_lags[index]), index),
            )
            order = torch.as_tensor(ordered, device=selected.device, dtype=torch.long)
        elif self.acf_channel_order == "similarity":
            ordered = signed_acf_profile_similarity_order(
                selected.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            order = torch.as_tensor(ordered, device=selected.device, dtype=torch.long)
        else:
            order = input_order
        inverse = torch.empty_like(order)
        inverse[order] = input_order
        self.acf_channel_order_indices = order.detach()
        self.acf_channel_inverse_indices = inverse.detach()

    def _train_prior_channel_order(self, device):
        if self.acf_channel_order == "input" or self.acf_prior.numel() == 0:
            return None, None
        order = self.acf_channel_order_indices.to(device=device, dtype=torch.long)
        inverse = self.acf_channel_inverse_indices.to(device=device, dtype=torch.long)
        expected = torch.arange(self.enc_in, device=device, dtype=torch.long)
        if order.numel() != self.enc_in or not torch.equal(order.index_select(0, inverse), expected):
            raise RuntimeError("NLinearVariableMamba ACF channel order is invalid")
        return order, inverse

    def export_acf_prior_state(self):
        if self.acf_prior.numel() == 0:
            return None
        full = self.acf_full_prior if self.acf_full_prior.numel() else self.acf_prior
        return {
            "schema_version": 1,
            "acf_values": self.acf_prior.detach().cpu().numpy().astype(np.float32, copy=True),
            "acf_full_values": full.detach().cpu().numpy().astype(np.float32, copy=True),
            "lags": self.acf_prior_lags.detach().cpu().numpy().astype(np.int64, copy=True),
            "seq_len": int(self.seq_len),
            "pred_len": int(self.pred_len),
            "channels": int(self.enc_in),
            "acf_source": "train_only",
            "acf_temperature": 1.0,
        }

    def save_acf_prior(self, path):
        state = self.export_acf_prior_state()
        if state is None:
            return False
        np.savez_compressed(
            str(path), schema_version=np.asarray(1, dtype=np.int64),
            acf_values=state["acf_values"], acf_full_values=state["acf_full_values"],
            lags=state["lags"], seq_len=np.asarray(self.seq_len, dtype=np.int64),
            pred_len=np.asarray(self.pred_len, dtype=np.int64),
            channels=np.asarray(self.enc_in, dtype=np.int64),
            acf_source=np.asarray("train_only"), acf_temperature=np.asarray(1.0),
        )
        return True

    def load_acf_prior(self, path):
        with np.load(str(path), allow_pickle=False) as payload:
            required = {"schema_version", "acf_values", "lags", "seq_len", "pred_len", "channels"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError("ACF prior artifact is missing fields: {}".format(", ".join(missing)))
            if int(np.asarray(payload["schema_version"]).item()) != 1:
                raise ValueError("unsupported ACF prior artifact schema")
            if int(np.asarray(payload["seq_len"]).item()) != self.seq_len or int(np.asarray(payload["pred_len"]).item()) != self.pred_len:
                raise ValueError("ACF prior forecast geometry does not match the model")
            if int(np.asarray(payload["channels"]).item()) != self.enc_in:
                raise ValueError("ACF prior channel count does not match the model")
            full = payload["acf_full_values"] if "acf_full_values" in payload.files else None
            self.set_compact_acf_prior(payload["acf_values"], payload["lags"], full)
        return True

    def _acf_memory(self, normalized):
        batch, length, channels = normalized.shape
        memory = normalized.new_zeros((batch, self.pred_len, channels))
        if not self.use_acf_memory or self.acf_prior.numel() == 0:
            return memory
        prior = self.acf_prior.to(device=normalized.device, dtype=normalized.dtype)
        weights = prior.abs()
        weights = prior / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        for slot, lag_value in enumerate(self.acf_prior_lags.tolist()):
            lag = int(lag_value)
            if lag > length:
                continue
            pattern = normalized[:, -lag:, :]
            repeats = (self.pred_len + lag - 1) // lag
            seasonal = pattern.repeat(1, repeats, 1)[:, :self.pred_len, :]
            memory = memory + seasonal * weights[:, slot].view(1, 1, channels)
        return memory

    def _select_acf_memory(self, memory, representation, tokens):
        """Apply a representation-conditioned, zero-initialized memory gate."""
        if not self.acf_memory_selector_enabled or representation is None:
            return memory
        batch, horizon, channels = memory.shape
        summary = representation.mean(dim=1)
        coefficients = self.acf_memory_horizon_selector(summary)
        # A fixed affine horizon basis avoids a d_ff x pred_len parameter
        # table while still allowing early/late forecast trust to differ.
        basis = torch.linspace(
            -1.0, 1.0, horizon, device=memory.device, dtype=memory.dtype
        )
        horizon_logits = coefficients[:, :1] + coefficients[:, 1:2] * basis
        channel_tokens = tokens[:, -self.c_out:, :]
        channel_logits = self.acf_memory_channel_selector(channel_tokens).squeeze(-1)
        channel_logits = channel_logits.unsqueeze(1)
        logits = horizon_logits.unsqueeze(-1) + channel_logits
        gate = torch.sigmoid(logits)
        scale = torch.tanh(self.acf_memory_selector_scale).to(memory.dtype)
        return memory * (1.0 + scale * (2.0 * gate - 1.0))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                onlyrepr=False, return_repr=None, return_variable_tokens=False):
        normalized, restore = self._normalize(x_enc)
        tokens = self._tokens(normalized)
        need_repr = bool(
            onlyrepr
            or self.acf_memory_selector_enabled
            or (self.AutoCon if return_repr is None else bool(return_repr))
        )
        # Fusion must construct the same representation consumed by AutoCon
        # even for prediction-only forwards, otherwise its MSE path would be
        # disconnected from the contrastive representation parameters.
        representation = (
            self._representation(normalized, tokens)
            if need_repr or self.variable_repr_fusion_enabled else None
        )
        forecast_tokens = tokens
        if self.variable_repr_fusion_enabled:
            context = representation.mean(dim=1)
            gain = torch.tanh(self.repr_fusion_gate(context)).unsqueeze(1)
            forecast_tokens = tokens * (
                1.0 + torch.tanh(self.repr_fusion_scale) * gain
            )
        base = self.base_head(normalized.transpose(1, 2)).transpose(1, 2)
        residual = self.variable_head(forecast_tokens).transpose(1, 2)
        forecast_normalized = base[:, :, -self.c_out:] + torch.tanh(self.forecast_scale) * residual[:, :, -self.c_out:]
        raw_history = self._restore_history(normalized, restore)
        acf_memory = self._acf_memory(raw_history)[:, :, -self.c_out:]
        acf_memory = self._select_acf_memory(
            acf_memory, representation, forecast_tokens[:, :, :]
        )
        prediction = self._restore(forecast_normalized, restore)
        prediction = prediction + torch.tanh(self.acf_memory_scale) * acf_memory
        if onlyrepr:
            if return_variable_tokens:
                return None, representation, tokens
            return None, representation
        if self.AutoCon or return_repr:
            if return_variable_tokens:
                return prediction, representation, tokens
            return prediction, representation
        if return_variable_tokens:
            return prediction, tokens
        return prediction


__all__ = ["Model"]
