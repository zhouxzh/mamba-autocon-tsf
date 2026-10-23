#!/usr/bin/env python3
"""Closed low-data DLinear-primary / one-scan pretrained-Mamba comparison."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DLinear, MambaTS, NLinear
from models.MambaDLinearHybrid import Model as Hybrid
from scripts.finetune_mambatsv3_core_transfer import Windows, config, transfer_core, json_compatible
from data_provider.samplers import ACFLagPairBatchSampler
from data_provider.data_loader import load_numeric_array_file
from layers.losses import ForecastLagPairAffineLoss
from utils.acf_guidance import fit_signed_acf_prior
from utils.acf_channel_descriptor import build_acf_channel_descriptor


LINEAR_MODELS = {"DLinear": DLinear.Model, "NLinear": NLinear.Model}
TRANSFERABLE_CORE_PREFIXES = (
    "patch_embedding.", "patch_position_embedding", "encoder.", "final_norm.", "repr_", "ts_shared_channel_summary_", "ts_channel_set_pooling_", "ts_channel_prototype_", "ts_lite_",
    "ts_acf_channel_descriptor_", "ts_shared_postscan_mixer.",
)


def _load_target_values(path: Path) -> np.ndarray:
    """Read CSV or repository-supported numeric-array targets as [time, channel]."""
    return load_numeric_array_file(path, data_format="auto")


class OriginWindows(Dataset):
    """Forecast windows retaining target-train origins for ACF pairing."""

    def __init__(self, values, seq_len, pred_len, starts):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len, self.pred_len = int(seq_len), int(pred_len)
        self.starts = np.asarray(starts, dtype=np.int64)
        if not len(self.starts):
            raise ValueError("no target affine windows")

    def __len__(self): return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return start, self.values[start:start + self.seq_len], self.values[start + self.seq_len:start + self.seq_len + self.pred_len]


def _paired_starts(total, max_windows, lags):
    starts = np.arange(total, dtype=np.int64)
    if len(starts) <= max_windows: return starts
    lags = sorted({int(lag) for lag in lags if 0 < int(lag) < len(starts)}) or [1]
    selected = set()
    for index in range(max_windows // 2):
        lag = lags[index % len(lags)]
        anchor = int(round(index * (len(starts) - lag - 1) / max(1, max_windows // 2 - 1)))
        selected.update((anchor, anchor + lag))
    for start in np.linspace(0, len(starts) - 1, max_windows * 2, dtype=np.int64):
        if len(selected) >= max_windows: break
        selected.add(int(start))
    return np.asarray(sorted(selected)[:max_windows], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=48)
    parser.add_argument("--mamba-d-state", type=int, default=8)
    parser.add_argument("--mamba-d-conv", type=int, default=2)
    parser.add_argument("--mamba-expand", type=int, default=1)
    parser.add_argument("--mamba-headdim", type=int, default=None,
                        help="Mamba mixer head width; defaults to d-model to match source pretraining")
    parser.add_argument("--mamba-ts-mixer-type", choices=["mamba1", "mamba2"], default="mamba1")
    parser.add_argument("--mamba-ts-variant", choices=["v3", "v13"], default="v3",
                        help="channel-count-agnostic temporal backbone family member")
    parser.add_argument("--context-rank", type=int, default=16)
    parser.add_argument("--channel-agnostic-scan", action="store_true")
    parser.add_argument("--mamba-ts-repr-forecast-fusion", action="store_true",
                        help="zero-gated AutoCon representation-to-forecast residual from the existing scan")
    parser.add_argument("--mamba-ts-repr-forecast-channelwise", action="store_true",
                        help="use shared per-channel representation projection; requires forecast fusion")
    parser.add_argument("--mamba-ts-shared-channel-summary", action="store_true",
                        help="inject channel-count-agnostic mean/std patch summary before the shared scan")
    parser.add_argument("--mamba-ts-channel-set-pooling", action="store_true",
                        help="learn permutation-equivariant channel pooling before the shared scan")
    parser.add_argument("--mamba-ts-channel-prototype-pooling", action="store_true")
    parser.add_argument("--mamba-ts-channel-prototype-count", type=int, default=4)
    parser.add_argument("--mamba-ts-acf-channel-descriptor", action="store_true",
                        help="use a fixed target-train-prefix ACF descriptor in the shared patch encoder")
    parser.add_argument("--mamba-ts-acf-channel-descriptor-tune-scale", action="store_true",
                        help="allow only the zero-gated ACF descriptor scale to adapt on target labels")
    parser.add_argument("--mamba-ts-shared-postscan-mixer", action="store_true")
    parser.add_argument("--mamba-ts-shared-postscan-mixer-rank", type=int, default=4)
    parser.add_argument("--mamba-ts-channel-adapter", action="store_true",
                        help="train a zero-gated low-rank target residual from the existing one-scan state")
    parser.add_argument("--mamba-ts-channel-adapter-rank", type=int, default=8)
    parser.add_argument("--mamba-ts-target-mlp-adapter", action="store_true",
                        help="train a zero-gated shared nonlinear decoder from existing one-scan patch states")
    parser.add_argument("--mamba-ts-target-mlp-adapter-rank", type=int, default=32)
    parser.add_argument("--mamba-ts-revin-affine", action="store_true",
                        help="learn target-only per-channel affine calibration after MeanStd normalization")
    parser.add_argument("--transfer-prediction-head", action="store_true")
    parser.add_argument("--prediction-head-transfer-mode", choices=["tile", "exact"], default="exact")
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--linear-model", choices=sorted(LINEAR_MODELS), default="DLinear")
    parser.add_argument("--mamba-correction-output", action="store_true")
    parser.add_argument("--mamba-channelwise-residual-scale", action="store_true",
                        help="learn a zero-initialized target-channel gate for the existing Mamba correction")
    parser.add_argument("--mamba-centered-channelwise-residual-scale", action="store_true",
                        help="learn zero-mean channel deviations around a shared Mamba correction scale")
    parser.add_argument("--mamba-channelwise-horizon-residual-scale", action="store_true",
                        help="make the target-channel correction gate prediction-step specific")
    parser.add_argument("--mamba-channelwise-residual-bias", action="store_true",
                        help="learn a zero-initialized target channel/horizon calibration after the correction gate")
    parser.add_argument("--mamba-shared-channel-statistic-residual-gate", action="store_true",
                        help="learn a channel-count-agnostic residual gate from local per-channel window statistics")
    parser.add_argument("--mamba-shared-channel-statistic-gate-rank", type=int, default=8)
    parser.add_argument("--target-affine-fraction", type=float, default=0.0)
    parser.add_argument("--target-affine-pairing", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--few-shot-fraction", type=float, choices=[0.1, 0.25, 1.0], required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=64)
    parser.add_argument("--freeze-mamba-epochs", type=int, default=1)
    parser.add_argument("--freeze-pretrained-core", action="store_true",
                        help="keep transferred shared encoder weights frozen while tuning target forecast parameters")
    parser.add_argument("--mamba-core-learning-rate", type=float, default=None,
                        help="optional discriminative LR for transferred shared core after warm-up")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.mamba_headdim is None:
        args.mamba_headdim = args.d_model
    if args.mamba_headdim <= 0:
        raise ValueError("mamba-headdim must be positive")
    if args.mamba_ts_repr_forecast_channelwise and not args.mamba_ts_repr_forecast_fusion:
        raise ValueError("mamba-ts-repr-forecast-channelwise requires mamba-ts-repr-forecast-fusion")
    if args.mamba_ts_acf_channel_descriptor_tune_scale and not args.mamba_ts_acf_channel_descriptor:
        raise ValueError("mamba-ts-acf-channel-descriptor-tune-scale requires descriptor conditioning")
    if args.mamba_ts_shared_postscan_mixer_rank <= 0:
        raise ValueError("mamba-ts-shared-postscan-mixer-rank must be positive")
    if args.linear_only and args.source_checkpoint:
        raise ValueError("linear-only control must not receive a source checkpoint")
    if args.mamba_channelwise_residual_bias and not args.mamba_channelwise_residual_scale:
        raise ValueError("mamba-channelwise-residual-bias requires mamba-channelwise-residual-scale")
    if args.mamba_channelwise_horizon_residual_scale and not args.mamba_channelwise_residual_scale:
        raise ValueError("mamba-channelwise-horizon-residual-scale requires mamba-channelwise-residual-scale")
    if args.mamba_centered_channelwise_residual_scale and not args.mamba_channelwise_residual_scale:
        raise ValueError("mamba-centered-channelwise-residual-scale requires mamba-channelwise-residual-scale")
    if args.mamba_shared_channel_statistic_residual_gate and args.mamba_channelwise_residual_scale:
        raise ValueError("mamba-shared-channel-statistic-residual-gate is an alternative to mamba-channelwise-residual-scale")
    if args.mamba_shared_channel_statistic_gate_rank <= 0:
        raise ValueError("mamba-shared-channel-statistic-gate-rank must be positive")
    if args.freeze_pretrained_core and args.linear_only:
        raise ValueError("freeze-pretrained-core is incompatible with linear-only")
    if args.freeze_pretrained_core and not args.source_checkpoint:
        raise ValueError("freeze-pretrained-core requires a source checkpoint")
    if args.mamba_ts_channel_adapter_rank <= 0:
        raise ValueError("mamba-ts-channel-adapter-rank must be positive")
    if args.mamba_ts_target_mlp_adapter_rank <= 0:
        raise ValueError("mamba-ts-target-mlp-adapter-rank must be positive")
    if args.mamba_core_learning_rate is not None and args.mamba_core_learning_rate <= 0:
        raise ValueError("mamba-core-learning-rate must be positive")
    if not 0.0 <= args.target_affine_fraction <= 1.0:
        raise ValueError("target-affine-fraction must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    raw = _load_target_values(args.dataset)
    train_end = int(len(raw) * args.train_fraction); val_end = int(len(raw) * (args.train_fraction + args.val_fraction)); few_end = int(train_end * args.few_shot_fraction)
    mean = raw[:few_end].mean(axis=0, keepdims=True); std = raw[:few_end].std(axis=0, keepdims=True)
    values = (raw - mean) / np.sqrt(std * std + 1e-5)
    starts = np.arange(0, few_end - args.seq_len - args.pred_len + 1)
    if len(starts) > args.max_train_windows:
        starts = np.linspace(0, len(starts) - 1, args.max_train_windows, dtype=np.int64)
    val_starts = np.arange(max(0, train_end - args.seq_len), val_end - args.seq_len - args.pred_len + 1)
    val_starts = val_starts[val_starts + args.seq_len >= train_end]
    test_starts = np.arange(max(0, val_end - args.seq_len), len(raw) - args.seq_len - args.pred_len + 1)
    test_starts = test_starts[test_starts + args.seq_len >= val_end]
    target_relation = None
    if args.target_affine_fraction or args.target_affine_pairing:
        target_prior = fit_signed_acf_prior(raw[:few_end], max_lag=min(720, few_end - 1), top_k=8, min_lag=1, min_abs_acf=.05, confidence_z=1.96, min_separation=2, detrend="mean", channel_chunk_size=64)
        lags = list(target_prior.common_lags(top_k=8)) or [1]
        paired = _paired_starts(few_end - args.seq_len - args.pred_len + 1, args.max_train_windows, lags)
        paired_dataset = OriginWindows(values, args.seq_len, args.pred_len, paired)
        train_loader = DataLoader(paired_dataset, batch_sampler=ACFLagPairBatchSampler(paired_dataset, batch_size=args.batch_size, lags=lags, lag_balance="balanced", seed=args.seed, origins=paired), num_workers=0)
        target_relation = (
            ForecastLagPairAffineLoss(target_prior.acf, selected_lags=lags).cuda()
            if args.target_affine_fraction else None
        )
    else:
        train_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, starts), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, val_starts), batch_size=args.batch_size)
    mamba = MambaTS.Model(config(args, raw.shape[1])).cuda()
    if args.mamba_ts_acf_channel_descriptor:
        descriptor_prior = fit_signed_acf_prior(
            raw[:few_end], max_lag=min(720, few_end - 1), top_k=8, min_lag=1,
            min_abs_acf=.05, confidence_z=1.96, min_separation=2,
            detrend="mean", channel_chunk_size=64,
        )
        descriptor_peaks = np.zeros((raw.shape[1], 8), dtype=np.int64)
        for channel in range(raw.shape[1]):
            selected = descriptor_prior.channel_lags(channel, top_k=8)
            descriptor_peaks[channel, :len(selected)] = selected
        with torch.no_grad():
            mamba.ts_acf_descriptor_values.copy_(torch.as_tensor(
                build_acf_channel_descriptor(descriptor_prior.acf, descriptor_peaks),
                device=mamba.ts_acf_descriptor_values.device,
            ))
    transfer = None if args.source_checkpoint is None else transfer_core(mamba, args.source_checkpoint, args.transfer_prediction_head, args.prediction_head_transfer_mode)
    if args.mamba_correction_output:
        if transfer is None:
            raise ValueError("mamba-correction-output requires a residual-target source checkpoint")
        raw_checkpoint = torch.load(args.source_checkpoint[0], map_location="cpu", weights_only=False)
        if raw_checkpoint.get("prediction_target") != "last_value_residual":
            raise ValueError("mamba-correction-output requires prediction_target=last_value_residual")
    linear_config = config(args, raw.shape[1]); linear_config.AutoCon_wnorm = "MeanStd"; linear_config.moving_avg = 25; linear_config.individual = False
    model = Hybrid(mamba, LINEAR_MODELS[args.linear_model](linear_config).cuda(),
                   linear_only=args.linear_only, mamba_correction_output=args.mamba_correction_output,
                   channelwise_residual_scale=args.mamba_channelwise_residual_scale,
                   centered_channelwise_residual_scale=args.mamba_centered_channelwise_residual_scale,
                   channelwise_horizon_residual_scale=args.mamba_channelwise_horizon_residual_scale,
                   channelwise_residual_bias=args.mamba_channelwise_residual_bias,
                   shared_channel_statistic_residual_gate=args.mamba_shared_channel_statistic_residual_gate,
                   shared_channel_statistic_gate_rank=args.mamba_shared_channel_statistic_gate_rank).cuda()
    core_parameters = [
        parameter for name, parameter in model.mamba.named_parameters()
        if name.startswith(TRANSFERABLE_CORE_PREFIXES)
    ]
    core_ids = {id(parameter) for parameter in core_parameters}
    target_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in core_ids
    ]
    optimizer_groups = [{"params": target_parameters, "lr": 1e-3}]
    if core_parameters:
        optimizer_groups.append({
            "params": core_parameters,
            "lr": args.mamba_core_learning_rate or 1e-3,
        })
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-5)
    best, best_state, history = float("inf"), None, []
    for epoch in range(args.epochs):
        frozen = not args.linear_only and epoch < args.freeze_mamba_epochs
        core_frozen = bool(args.freeze_pretrained_core and transfer is not None)
        forecast_head_trainable = False
        channel_adapter_trainable = False
        target_mlp_adapter_trainable = False
        descriptor_scale_trainable = False
        for name, parameter in model.mamba.named_parameters():
            parameter.requires_grad = not (
                frozen or (core_frozen and name.startswith(TRANSFERABLE_CORE_PREFIXES))
            )
            if (args.mamba_ts_acf_channel_descriptor_tune_scale
                    and name == "ts_acf_channel_descriptor_scale"):
                parameter.requires_grad = True
            if name.startswith("forecast_head") and parameter.requires_grad:
                forecast_head_trainable = True
            if name.startswith("ts_channel_adapter") and parameter.requires_grad:
                channel_adapter_trainable = True
            if name.startswith("ts_target_mlp_adapter") and parameter.requires_grad:
                target_mlp_adapter_trainable = True
            if name == "ts_acf_channel_descriptor_scale" and parameter.requires_grad:
                descriptor_scale_trainable = True
        model.train(); losses = []; affine_losses = []; weighted_affine_losses = []; target_active = []
        for packed in train_loader:
            if args.target_affine_fraction or args.target_affine_pairing:
                origins, x, y = packed
            else:
                x, y = packed; origins = None
            prediction = model(x.cuda())
            mse = F.mse_loss(prediction, y.cuda())
            if target_relation is None:
                affine = mse.detach() * 0.0; weighted_affine = affine
            else:
                affine = target_relation(prediction, origins.long().cuda()).mean()
                weight = args.target_affine_fraction * mse.detach() / affine.detach().abs().clamp_min(1e-6)
                weighted_affine = weight * affine
                target_active.append(float(target_relation.last_active_anchor_fraction))
            loss = mse + weighted_affine
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            affine_losses.append(float(affine.detach()))
            weighted_affine_losses.append(float(weighted_affine.detach()))
        model.eval(); errors = []
        with torch.no_grad():
            for x, y in val_loader:
                errors.append(F.mse_loss(model(x.cuda()), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        val = float(torch.cat(errors).mean())
        residual_scale = torch.tanh(model.residual_scale).detach()
        global_residual_scale = (
            float(torch.tanh(model.global_residual_scale).detach())
            if model.global_residual_scale is not None else None
        )
        residual_bias = model.residual_bias.detach() if model.residual_bias is not None else None
        statistic_gate = (
            model._shared_channel_statistic_gate(x.cuda()).detach()
            if model.shared_channel_statistic_residual_gate else None
        )
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_mse": val, "mamba_frozen": frozen, "mamba_core_frozen": core_frozen, "mamba_forecast_head_trainable": forecast_head_trainable, "mamba_channel_adapter_trainable": channel_adapter_trainable, "mamba_target_mlp_adapter_trainable": target_mlp_adapter_trainable, "acf_descriptor_scale_trainable": descriptor_scale_trainable, "acf_descriptor_scale": float(model.mamba.ts_acf_channel_descriptor_scale.detach()) if model.mamba.ts_acf_channel_descriptor_scale is not None else None, "mamba_core_learning_rate": args.mamba_core_learning_rate, "channelwise_residual_scale": bool(args.mamba_channelwise_residual_scale), "centered_channelwise_residual_scale": bool(args.mamba_centered_channelwise_residual_scale), "global_residual_scale": global_residual_scale, "channelwise_horizon_residual_scale": bool(args.mamba_channelwise_horizon_residual_scale), "channelwise_residual_bias": bool(args.mamba_channelwise_residual_bias), "shared_channel_statistic_residual_gate": bool(args.mamba_shared_channel_statistic_residual_gate), "shared_channel_statistic_gate_rank": int(args.mamba_shared_channel_statistic_gate_rank), "statistic_gate_abs_max": float(statistic_gate.abs().max()) if statistic_gate is not None else 0.0, "residual_scale": float(residual_scale.mean()), "residual_scale_abs_max": float(residual_scale.abs().max()), "residual_scale_shape": list(residual_scale.shape), "residual_bias_abs_max": float(residual_bias.abs().max()) if residual_bias is not None else 0.0, "residual_bias_shape": list(residual_bias.shape) if residual_bias is not None else [], "target_affine_loss": float(np.mean(affine_losses)), "target_weighted_affine_loss": float(np.mean(weighted_affine_losses)), "target_affine_fraction": args.target_affine_fraction, "target_affine_active_fraction": float(np.mean(target_active) if target_active else 0.0)})
        if val < best: best, best_state = val, {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    test_mse = None
    if args.evaluate_test:
        model.load_state_dict(best_state, strict=True); model.eval(); errors = []
        test_loader = DataLoader(Windows(values, args.seq_len, args.pred_len, test_starts), batch_size=args.batch_size)
        with torch.no_grad():
            for x, y in test_loader:
                errors.append(F.mse_loss(model(x.cuda()), y.cuda(), reduction="none").mean(dim=(1, 2)).cpu())
        test_mse = float(torch.cat(errors).mean())
    payload = {"model": best_state, "config": json_compatible(vars(args)), "best_validation_mse": best, "test_mse": test_mse, "history": history, "transfer": transfer, "target_channels": int(raw.shape[1]), "few_train_rows": few_end, "original_train_rows": train_end, "validation_rows": val_end-train_end, "test_rows": len(raw)-val_end, "test_metrics_produced": bool(args.evaluate_test)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({**payload, "model": None}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_mse": best, "test_metrics_produced": bool(args.evaluate_test)}, sort_keys=True))


if __name__ == "__main__":
    main()
