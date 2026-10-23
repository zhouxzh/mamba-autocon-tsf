#!/usr/bin/env python3
"""Run a small CUDA contract smoke for the MambaACF forecast relation view.

This is an implementation check, not a training benchmark.  It constructs
both the Mamba-2 and Mamba-1 ACF models, attaches a deterministic train-only
style prior, runs one forward/loss/backward pass, and verifies that the
forecast relation objective reaches the forecast head.  No checkpoints,
metrics, or other experiment artifacts are written.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# Support both ``python scripts/...`` and module-style invocation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layers.losses import ACFRelationLoss
from models.MambaACF import Model as Mamba2ACF
from models.MambaACF1 import Model as Mamba1ACF


def _arguments(channels: int, pred_len: int) -> SimpleNamespace:
    """Return the smallest explicit config shared by both ACF backends."""

    return SimpleNamespace(
        seq_len=96,
        label_len=48,
        pred_len=pred_len,
        enc_in=channels,
        c_out=channels,
        d_model=16,
        d_ff=32,
        e_layers=1,
        mamba_d_state=8,
        mamba_d_conv=2,
        mamba_expand=2,
        mamba_headdim=8,
        AutoCon=True,
        AutoCon_wnorm="LastVal",
        AutoCon_acf_source="raw",
        embed="timeF",
        freq="h",
        wo_freq="",
        dropout=0.0,
        features="M",
        mamba_scales=[1],
        mamba_token_stride=1,
        mamba_bidirectional=False,
        mamba_channel_independent=False,
        mamba_channel_gate=False,
        mamba_channel_gate_rank=8,
        mamba_data_dependent_channel_mixer=False,
        mamba_data_dependent_channel_mixer_rank=8,
        mamba_acf_scale_conditioning=False,
        mamba_acf_channel_gate=False,
        mamba_causal_embedding=False,
        mamba_variable_mixer=False,
        mamba_variable_mixer_depth=1,
        mamba_variable_group_size=1,
        mamba_variable_permutation=False,
        mamba_acf_channel_order="input",
        mamba_acf_channel_conditioning=False,
        mamba_acf_repr_fusion=False,
        mamba_variable_repr_fusion=False,
        mamba_patch_mixer=False,
        mamba_patch_mixer_depth=1,
        mamba_patch_len=16,
        mamba_patch_stride=8,
        mamba_patch_order="time_major",
        mamba_patch_permutation=False,
        mamba_local_mixer=False,
        mamba_local_patch_len=16,
        mamba_local_patch_stride=8,
        mamba_local_width=16,
        mamba_local_rank=8,
        mamba_local_depth=1,
        mamba_low_rank_head=False,
        mamba_temporal_rank=16,
        mamba_acf_lags=[24],
        mamba_use_acf_memory=True,
        mamba_dynamic_acf=False,
        mamba_acf_temperature=4.0,
        mamba_acf_step_reliability=False,
        mamba_acf_reliability_interpolation=False,
        mamba_acf_horizon_calibration=False,
        mamba_acf_horizon_calibration_rank=2,
        mamba_acf_memory_gate_mode="hidden",
        mamba_acf_anchor_copy=False,
        mamba_acf_local_agreement=False,
        mamba_acf_relation_loss=True,
        mamba_acf_relation_batching="lag_pairs",
        mamba_acf_relation_lag_balance="balanced",
        mamba_acf_relation_channel_reduce="mean",
        mamba_acf_relation_scope="global",
        mamba_acf_relation_representation="forecast",
        mamba_acf_relation_lag_mask=True,
        mamba_acf_relation_lag_tolerance=0,
        mamba_acf_relation_min_lag=1,
        mamba_acf_relation_target_mode="hard_selected",
        mamba_acf_relation_soft_min_abs=0.1,
        mamba_acf_relation_diagnostic_min_abs=0.1,
        mamba_acf_relation_selected_lag_boost=0.0,
        mamba_acf_relation_active_anchor_normalization=False,
    )


def _run_one(name, model_cls, *, channels, pred_len, batch_size, device):
    args = _arguments(channels, pred_len)
    model = model_cls(args).to(device).train()
    # The sidecar-shaped full table is deliberately attached through the
    # public API, matching the train-only artifact contract used by training.
    prior = torch.zeros(channels, 25, dtype=torch.float32, device=device)
    prior[:, 24] = 0.8
    model.set_acf_prior(prior, lags=[24])

    x = torch.randn(batch_size, args.seq_len, channels, device=device)
    x_mark = torch.randn(batch_size, args.seq_len, 4, device=device)
    decoder = torch.zeros(
        batch_size, args.label_len + pred_len, channels, device=device
    )
    decoder_mark = torch.randn(batch_size, args.label_len + pred_len, 4, device=device)
    prediction, relation_view = model(
        x, x_mark, decoder, decoder_mark, return_repr=True
    )
    relation_prior = torch.zeros(channels, 64, dtype=torch.float32, device=device)
    relation_prior[:, 24] = 0.8
    objective = ACFRelationLoss(
        relation_prior,
        relation_scope="global",
        selected_lags=[24],
        min_lag=1,
    )
    _, relation_loss = objective(
        F.normalize(relation_view, dim=-1),
        torch.arange(batch_size, device=device, dtype=torch.long) * 24,
    )
    loss = prediction.square().mean() + 0.03 * relation_loss.mean()
    loss.backward()
    torch.cuda.synchronize(device)
    gradient = model.temporal_head.weight.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("forecast relation gradient did not reach temporal_head")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(loss)):
        raise RuntimeError("non-finite prediction or loss")
    profile = model.resource_profile()
    result = {
        "backend": name,
        "prediction_shape": list(prediction.shape),
        "relation_view_shape": list(relation_view.shape),
        "loss": float(loss.detach().cpu()),
        "temporal_head_gradient_l1": float(gradient.detach().abs().sum().cpu()),
        "forecast_view_global_pooling": profile.get(
            "acf_relation_forecast_view_global_pooling"
        ),
        "prediction_only_extra_macs": profile.get(
            "acf_relation_forecast_view_prediction_only_macs_per_batch_item"
        ),
    }
    del objective, prediction, relation_view, loss, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    if args.channels < 2:
        parser.error("--channels must be at least 2 for a channel relation objective")
    if args.pred_len < 1 or args.batch_size < 2:
        parser.error("--pred-len must be positive and --batch-size at least 2")
    if not torch.cuda.is_available():
        print("CUDA is required by the installed mamba_ssm backend", flush=True)
        return 2
    torch.manual_seed(20260903)
    device = torch.device("cuda")
    results = [
        _run_one(
            name,
            model_cls,
            channels=args.channels,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            device=device,
        )
        for name, model_cls in (
            ("MambaACF(Mamba2)", Mamba2ACF),
            ("MambaACF1(Mamba1)", Mamba1ACF),
        )
    ]
    print(json.dumps({"status": "pass", "results": results}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
