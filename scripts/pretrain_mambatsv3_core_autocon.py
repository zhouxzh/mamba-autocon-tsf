#!/usr/bin/env python3
"""Self-supervise transferable MambaTSv3 core weights with masked patches and AutoCon.

The production MambaTSv3 scan is reused exactly once per input window. During
pretraining, channel-count-specific embedding/context parameters are bypassed;
only patch embedding, patch positions, temporal encoder, final norm, and the
AutoCon representation projection are shared across source domains. A masked
patch decoder is training-only and is deliberately excluded from the transfer
checkpoint. Source ACF priors and relation labels are fitted exclusively from
each chronological source-training prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import load_numeric_array_file
from data_provider.samplers import ACFLagPairBatchSampler
from layers.losses import (
    ACFRelationLoss, ForecastLagPairAffineLoss, ForecastLagPairConsistencyLoss,
    ForecastLagPairContrastiveLoss, ChannelPatchLagContrastiveLoss, PooledPatchLagContrastiveLoss,
    PooledPatchVarianceLoss,
)
from models import MambaTS
from utils.acf_guidance import fit_signed_acf_prior
from utils.acf_channel_descriptor import build_acf_channel_descriptor


CORE_PREFIXES = (
    "patch_embedding.", "patch_position_embedding", "encoder.", "final_norm.", "repr_",
    "ts_shared_channel_summary_", "ts_channel_set_pooling_", "ts_channel_prototype_", "ts_lite_",
    "ts_acf_channel_descriptor_", "ts_shared_postscan_mixer.",
)


class Windows(Dataset):
    def __init__(self, values: np.ndarray, seq_len: int, max_windows: int,
                 next_patch: bool = False, patch_len: int = 0,
                 relation_lags: list[int] | None = None,
                       relation_pair_policy: str = "legacy"):
        if values.ndim != 2 or len(values) <= seq_len:
            raise ValueError("source train prefix is too short for the requested window")
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len = int(seq_len)
        self.next_patch = bool(next_patch)
        self.patch_len = int(patch_len)
        if self.next_patch and self.patch_len <= 0:
            raise ValueError("next-patch source windows require a positive patch_len")
        starts = np.arange(
            len(values) - seq_len - (self.patch_len if self.next_patch else 0) + 1,
            dtype=np.int64,
        )
        self.starts = self._select_starts(
            starts, max_windows, relation_lags or [], relation_pair_policy
        )

    @staticmethod
    def _select_starts(starts: np.ndarray, max_windows: int,
                       relation_lags: list[int], pair_policy: str = "legacy"):
        if len(starts) <= max_windows:
            return starts
        usable = sorted({int(lag) for lag in relation_lags if 0 < int(lag) < len(starts)})
        if not usable:
            return np.linspace(0, len(starts) - 1, max_windows, dtype=np.int64)
        if pair_policy not in {"legacy", "round_robin", "contrastive_cluster"}:
            raise ValueError("relation_pair_policy must be 'legacy', 'round_robin', or 'contrastive_cluster'")
        selected = set()
        if pair_policy == "contrastive_cluster":
            # A forecast InfoNCE mini-batch needs two local positive pairs
            # plus non-positive in-range candidates.  Form four-origin
            # clusters [a, a+d, a+L, a+L+d], where L is a selected ACF lag
            # and d is deliberately not one of those lags.  The sampler's
            # local packing keeps each cluster together without new scans.
            primary = usable[0]
            offset = max(1, primary // 2)
            while offset in usable:
                offset += 1
            cluster_span = primary + offset
            anchors = np.linspace(
                0, len(starts) - cluster_span - 1,
                max(1, max_windows // 4), dtype=np.int64,
            )
            for anchor in anchors:
                cluster = [
                    int(starts[anchor]), int(starts[anchor + offset]),
                    int(starts[anchor + primary]), int(starts[anchor + primary + offset]),
                ]
                if len(selected) + 4 > max_windows:
                    break
                selected.update(cluster)
            for start in np.linspace(0, len(starts) - 1, max_windows * 2, dtype=np.int64):
                if len(selected) >= max_windows:
                    break
                selected.add(int(starts[start]))
            return np.asarray(sorted(selected)[:max_windows], dtype=np.int64)
        if pair_policy == "legacy":
            # Preserve the historical source-window contract used by all
            # audited releases. Experimental pair policies must opt in.
            for index in range(max_windows // 2):
                lag = usable[index % len(usable)]
                anchor = int(round(index * (len(starts) - lag - 1) / max(1, max_windows // 2 - 1)))
                selected.update((int(starts[anchor]), int(starts[anchor + lag])))
            for start in np.linspace(0, len(starts) - 1, max_windows * 2, dtype=np.int64):
                if len(selected) >= max_windows:
                    break
                selected.add(int(starts[start]))
            return np.asarray(sorted(selected)[:max_windows], dtype=np.int64)
        # Preserve both members of every seeded relation pair.  Sorting and
        # slicing a mixed anchor/partner set can silently discard the distant
        # partner, leaving a contrastive batch without a positive example.
        pair_budget = max_windows // 2
        candidate_pairs = {}
        density = max(pair_budget * 4, len(usable) * 2)
        for lag_index, lag in enumerate(usable):
            anchors = np.linspace(0, len(starts) - lag - 1, density, dtype=np.int64)
            offset = lag_index % max(1, len(anchors))
            candidate_pairs[lag] = [
                (int(starts[anchor]), int(starts[anchor + lag]))
                for anchor in np.roll(anchors, -offset)
            ]
        # Interleave lags before accepting pairs.  A lag-sorted greedy pass
        # lets small periods consume every origin, starving channel-specific
        # daily/weekly positives from an InfoNCE denominator.
        for index in range(density):
            rotated_lags = usable[index % len(usable):] + usable[:index % len(usable)]
            for lag in rotated_lags:
                anchor, partner = candidate_pairs[lag][index]
                if len(selected) + 2 > max_windows:
                    break
                if anchor not in selected and partner not in selected:
                    selected.update((anchor, partner))
                if len(selected) >= 2 * pair_budget:
                    break
            if len(selected) >= 2 * pair_budget:
                break
        for start in np.linspace(0, len(starts) - 1, max_windows * 2, dtype=np.int64):
            if len(selected) >= max_windows:
                break
            selected.add(int(starts[start]))
        return np.asarray(sorted(selected)[:max_windows], dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        values = self.values[start:start + self.seq_len]
        if self.next_patch:
            future = self.values[start + self.seq_len:start + self.seq_len + self.patch_len]
            return start, values, future
        return start, values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_values(path: Path, data_format: str) -> np.ndarray:
    if data_format == "csv":
        values = pd.read_csv(path).iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        values = load_numeric_array_file(path, data_format=data_format)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("source data must be a finite [time, channels] array: {}".format(path))
    return values


def _source_spec(raw: str) -> tuple[str, Path, str]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--source must be NAME:PATH:csv|npz|txt")
    return parts[0], Path(parts[1]).expanduser().resolve(), parts[2]


def _checkpoint_spec(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _config(args, channels: int):
    return SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.source_pred_len, enc_in=channels, c_out=channels,
        d_model=args.d_model, d_ff=args.d_ff, patch_len=args.patch_len,
        patch_stride=args.patch_stride, dropout=0.1, e_layers=1,
        mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand, mamba_headdim=args.mamba_headdim,
        mamba_ts_variant=args.mamba_ts_variant, mamba_ts_mixer_type=args.mamba_ts_mixer_type,
        mamba_ts_context_rank=args.context_rank, mamba_ts_repr_forecast_fusion=False,
        mamba_ts_shared_pretrain_mode=True, mamba_ts_channel_agnostic_scan=True,
        mamba_ts_shared_channel_summary=bool(getattr(args, "mamba_ts_shared_channel_summary", False)),
        mamba_ts_channel_set_pooling=bool(getattr(args, "mamba_ts_channel_set_pooling", False)),
        mamba_ts_channel_prototype_pooling=bool(getattr(args, "mamba_ts_channel_prototype_pooling", False)),
        mamba_ts_channel_prototype_count=int(getattr(args, "mamba_ts_channel_prototype_count", 4)),
        mamba_ts_shared_postscan_mixer=bool(getattr(args, "mamba_ts_shared_postscan_mixer", False)),
        mamba_ts_shared_postscan_mixer_rank=int(getattr(args, "mamba_ts_shared_postscan_mixer_rank", 4)),
        mamba_ts_acf_channel_descriptor=bool(getattr(args, "mamba_ts_acf_channel_descriptor", False)),
        AutoCon=True,
        AutoCon_wnorm="MeanStd", task_name="long_term_forecast",
    )


def _core_keys(state: dict[str, torch.Tensor]) -> list[str]:
    return sorted(
        key for key in state
        if key.startswith(CORE_PREFIXES) and key != "ts_acf_descriptor_values"
    )


def _copy_core(target: nn.Module, source: nn.Module, keys: list[str]) -> None:
    target_state = target.state_dict()
    source_state = source.state_dict()
    with torch.no_grad():
        for key in keys:
            target_state[key].copy_(source_state[key])


def _masked_input(model, values: torch.Tensor, patch_mask: torch.Tensor, mask_token: torch.Tensor):
    normalized, _ = model._normalize(values)
    patches = normalized.transpose(1, 2).unfold(-1, model.patch_len, model.patch_stride)
    if patch_mask.shape != (values.shape[0], model.patch_count):
        raise ValueError("patch mask shape mismatch")
    sample_mask = patch_mask.repeat_interleave(model.patch_len, dim=1)
    if sample_mask.shape[1] != model.seq_len:
        raise RuntimeError("masked pretraining requires non-overlapping full-coverage patches")
    replacement = mask_token.repeat(model.patch_count).view(1, model.seq_len, 1)
    masked = torch.where(sample_mask[:, :, None], replacement, normalized)
    return masked, patches


def _relation_loss(relation, representation: torch.Tensor, starts: torch.Tensor):
    local, global_loss = relation(F.normalize(representation, dim=-1), starts)
    return (local.mean(dim=1) + global_loss).mean() / 2.0


def _pooled_patch_representation(encoded: torch.Tensor, projector: nn.Module) -> torch.Tensor:
    """Pool shared encoder states without channel-count-specific parameters."""
    if encoded.ndim != 4:
        raise ValueError("encoded states must have shape [batch, channels, patches, embedding]")
    summary = torch.cat(
        (encoded.mean(dim=1), encoded.std(dim=1, unbiased=False)), dim=-1
    )
    return projector(summary)


def _channel_peak_matrix(prior, top_k: int = 8) -> np.ndarray:
    """Pack variable-length channel peak lists; zero is non-positive padding."""
    peaks = np.zeros((prior.n_channels, int(top_k)), dtype=np.int64)
    for channel in range(prior.n_channels):
        selected = prior.channel_lags(channel, top_k=top_k)
        peaks[channel, :len(selected)] = selected
    return peaks


def _channel_peak_stability_weights(train: np.ndarray, channel_peaks: np.ndarray) -> np.ndarray:
    """Score selected lags by signed-ACF agreement across train-prefix halves."""
    if train.ndim != 2 or channel_peaks.ndim != 2 or train.shape[1] != channel_peaks.shape[0]:
        raise ValueError("train and channel_peaks shapes are incompatible")
    midpoint = len(train) // 2
    if midpoint < 3:
        raise ValueError("train prefix is too short for stability estimation")
    max_lag = min(720, midpoint - 1)
    options = dict(max_lag=max_lag, top_k=8, min_lag=1, min_abs_acf=0.05,
                   confidence_z=1.96, min_separation=2, detrend="mean", channel_chunk_size=64)
    left = fit_signed_acf_prior(train[:midpoint], **options).acf
    right = fit_signed_acf_prior(train[midpoint:], **options).acf
    weights = np.zeros(channel_peaks.shape, dtype=np.float32)
    for channel in range(channel_peaks.shape[0]):
        for slot, lag in enumerate(channel_peaks[channel]):
            lag = int(lag)
            if 0 < lag <= max_lag:
                first, second = float(left[channel, lag]), float(right[channel, lag])
                weights[channel, slot] = max(0.0, 1.0 - abs(first - second)) if first * second > 0.0 else 0.0
    return weights


def _channel_peak_strength_weights(channel_acf: np.ndarray, channel_peaks: np.ndarray) -> np.ndarray:
    """Train-prefix ACF reliability for channel-wise loss aggregation.

    Each channel receives the median magnitude at its selected ACF peaks. The
    mean-one, clipped vector changes only how active channel losses are pooled;
    it never changes pairs, source windows, or model parameters.
    """
    values = np.asarray(channel_acf, dtype=np.float32)
    peaks = np.asarray(channel_peaks, dtype=np.int64)
    if values.ndim != 2 or peaks.ndim != 2 or values.shape[0] != peaks.shape[0]:
        raise ValueError("channel_acf and channel_peaks shapes are incompatible")
    strength = np.zeros(peaks.shape[0], dtype=np.float32)
    for channel in range(peaks.shape[0]):
        lags = peaks[channel]
        lags = lags[(lags > 0) & (lags < values.shape[1])]
        if len(lags):
            strength[channel] = float(np.median(np.abs(values[channel, lags])))
    positive = strength > 0
    if not np.any(positive):
        return np.ones_like(strength)
    normalized = strength / float(strength[positive].mean())
    return np.clip(normalized, .25, 4.0).astype(np.float32)


def _forecast_target_acf_table(train: np.ndarray, seq_len: int, horizon: int,
                               channel_peaks: np.ndarray, max_pairs: int = 64):
    """Fit future-residual ACF coefficients at fixed train-prefix peak lags.

    The affine forecast loss compares centered future trajectories, whereas a
    conventional ACF is fitted to the observed signal. For every already
    selected raw-signal peak, estimate the correlation between centered
    last-value residual future segments at that same window lag. The sampled
    segments are all contained in ``train`` and no pair, lag, model, or
    inference path is changed.
    """
    values = np.asarray(train, dtype=np.float32)
    peaks = np.asarray(channel_peaks, dtype=np.int64)
    if values.ndim != 2 or peaks.ndim != 2 or values.shape[1] != peaks.shape[0]:
        raise ValueError("train and channel_peaks shapes are incompatible")
    if seq_len <= 0 or horizon <= 1 or max_pairs < 2:
        raise ValueError("forecast-target ACF geometry is invalid")
    start_count = len(values) - int(seq_len) - int(horizon) + 1
    if start_count < 3:
        raise ValueError("train prefix is too short for forecast-target ACF")
    max_lag = min(720, len(values) - 1)
    table = np.zeros((values.shape[1], max_lag + 1), dtype=np.float32)
    table[:, 0] = 1.0
    lags = sorted({int(lag) for lag in peaks.ravel() if 0 < int(lag) <= max_lag})
    horizon_offsets = np.arange(int(horizon), dtype=np.int64)
    for lag in lags:
        pair_count = start_count - lag
        if pair_count < 2:
            continue
        origins = np.linspace(0, pair_count - 1, min(int(max_pairs), pair_count), dtype=np.int64)
        left = values[origins[:, None] + int(seq_len) + horizon_offsets[None, :]]
        right_origins = origins + lag
        right = values[right_origins[:, None] + int(seq_len) + horizon_offsets[None, :]]
        left = left - values[origins + int(seq_len) - 1][:, None, :]
        right = right - values[right_origins + int(seq_len) - 1][:, None, :]
        # Match ForecastLagPairAffineLoss: remove each trajectory's horizon mean.
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        numerator = np.sum(left * right, axis=(0, 1), dtype=np.float64)
        denominator = np.sqrt(
            np.sum(np.square(left), axis=(0, 1), dtype=np.float64)
            * np.sum(np.square(right), axis=(0, 1), dtype=np.float64)
        )
        table[:, lag] = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
        ).clip(-1.0, 1.0).astype(np.float32)
    lag_index = peaks.reshape(-1)
    channel_index = np.repeat(np.arange(peaks.shape[0]), peaks.shape[1])
    valid = (lag_index > 0) & (lag_index <= max_lag)
    task_values = table[channel_index[valid], lag_index[valid]]
    return table, {
        "candidate_lag_count": int(len(lags)),
        "evaluated_pair_slots": int(valid.sum()),
        "mean_abs_task_coefficient": float(np.abs(task_values).mean()) if len(task_values) else 0.0,
    }


def _acf_grouped_patch_prior(train: np.ndarray, channel_peaks: np.ndarray, group_count: int):
    """Derive fixed train-prefix ACF groups and their pooled ACF priors."""
    if group_count < 2 or group_count > train.shape[1]:
        raise ValueError("group_count must be in [2, channel_count]")
    dominant = np.where(channel_peaks[:, 0] > 0, channel_peaks[:, 0], np.iinfo(np.int64).max)
    order = np.argsort(dominant, kind="stable")
    assignment = np.empty(train.shape[1], dtype=np.int64)
    for group, indices in enumerate(np.array_split(order, group_count)):
        assignment[indices] = group
    normalized = (train - train.mean(axis=0, keepdims=True)) / np.sqrt(
        train.var(axis=0, keepdims=True) + 1e-5
    )
    grouped = np.stack([normalized[:, assignment == group].mean(axis=1) for group in range(group_count)], axis=1)
    prior = fit_signed_acf_prior(
        grouped, max_lag=min(720, len(grouped) - 1), top_k=8, min_lag=1,
        min_abs_acf=.05, confidence_z=1.96, min_separation=2, detrend="mean", channel_chunk_size=64,
    )
    peaks = _channel_peak_matrix(prior, top_k=8)
    return assignment, prior.acf.astype(np.float32), peaks


def _grouped_patch_representation(encoded: torch.Tensor, assignment: np.ndarray, group_count: int) -> torch.Tensor:
    """Mean-pool shared patch states by a fixed train-only ACF group."""
    if encoded.ndim != 4 or encoded.shape[1] != len(assignment):
        raise ValueError("encoded states and ACF group assignment are incompatible")
    index = torch.as_tensor(assignment, device=encoded.device, dtype=torch.long)
    pooled = encoded.new_zeros((encoded.shape[0], group_count, encoded.shape[2], encoded.shape[3]))
    pooled.index_add_(1, index, encoded)
    counts = torch.bincount(index, minlength=group_count).to(encoded).view(1, group_count, 1, 1)
    return pooled / counts.clamp_min(1)


def _coverage_pair_lags(channel_peaks: np.ndarray, min_support: float) -> tuple[list[int], dict[int, float]]:
    """Select train-only source lags that activate many channelwise positives."""
    if channel_peaks.ndim != 2 or channel_peaks.shape[0] == 0:
        raise ValueError("channel_peaks must have shape [channels, peaks]")
    if not 0.0 < float(min_support) <= 1.0:
        raise ValueError("min_channel_positive_support must be in (0, 1]")
    values, counts = np.unique(channel_peaks[channel_peaks > 0], return_counts=True)
    if not len(values):
        return [1], {1: 0.0}
    support = {
        int(lag): float(count) / float(channel_peaks.shape[0])
        for lag, count in zip(values.tolist(), counts.tolist())
    }
    selected = sorted(lag for lag, value in support.items() if value >= float(min_support))
    if not selected:
        best = min(support, key=lambda lag: (-support[lag], lag))
        selected = [int(best)]
    return selected, {lag: support[lag] for lag in sorted(support)}


def _restrict_channel_peaks(channel_peaks: np.ndarray, allowed_lags: list[int]) -> np.ndarray:
    """Retain only globally supported positives for each channel.

    The historical ``channel_coverage`` option constrained source-window
    sampling, but still retained rare channel-only lags in the InfoNCE
    numerator. This stricter option is opt-in so audited checkpoints remain
    byte-for-byte reproducible.
    """
    peaks = np.asarray(channel_peaks, dtype=np.int64)
    if peaks.ndim != 2:
        raise ValueError("channel_peaks must have shape [channels, peaks]")
    allowed = np.asarray(sorted({int(lag) for lag in allowed_lags if int(lag) > 0}), dtype=np.int64)
    if not len(allowed):
        raise ValueError("allowed_lags must contain at least one positive lag")
    restricted = peaks.copy()
    restricted[~np.isin(restricted, allowed)] = 0
    if not np.any(restricted > 0):
        raise ValueError("coverage restriction removed every channelwise positive")
    return restricted


def _validate_patch_geometry(seq_len: int, patch_len: int, patch_stride: int,
                             pretrain_task: str) -> None:
    """Validate patch coverage without forbidding overlapping next-patch tokens."""
    if patch_len <= 0 or patch_stride <= 0 or patch_len > seq_len:
        raise ValueError("patch length/stride must be positive and fit the sequence")
    if pretrain_task == "masked_reconstruction":
        if patch_len != patch_stride or seq_len % patch_len:
            raise ValueError("masked reconstruction requires non-overlapping full-coverage patches")
    elif pretrain_task == "next_patch":
        if patch_stride > patch_len:
            raise ValueError("next-patch pretraining requires overlapping or contiguous patch coverage")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="NAME:PATH:csv|npz|txt")
    parser.add_argument("--calibration-source", default=None,
                        help=("optional target-domain NAME:PATH:csv|npz|txt; only its earliest "
                              "train-fraction x calibration-fraction prefix is used"))
    parser.add_argument("--calibration-fraction", type=float, default=None,
                        help="few-shot fraction of the chronological training prefix for --calibration-source")
    parser.add_argument("--initial-checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=24)
    parser.add_argument("--source-pred-len", type=int, default=24,
                        help="next-segment length and transferable source prediction-head horizon")
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--d-ff", type=int, default=48)
    parser.add_argument("--mamba-d-state", type=int, default=8)
    parser.add_argument("--mamba-d-conv", type=int, default=2)
    parser.add_argument("--mamba-expand", type=int, default=1)
    parser.add_argument("--mamba-headdim", type=int, default=None,
                        help="Mamba mixer head width; defaults to d-model for source/target compatibility")
    parser.add_argument("--mamba-ts-mixer-type", choices=["mamba1", "mamba2"], default="mamba1",
                        help="select the state-space mixer while retaining the MambaTSv3 outer contract")
    parser.add_argument("--mamba-ts-variant", choices=["v3", "v13"], default="v3",
                        help="channel-count-agnostic temporal backbone family member")
    parser.add_argument("--mamba-ts-shared-channel-summary", action="store_true",
                        help="inject channel-count-agnostic mean/std patch summary before the shared scan")
    parser.add_argument("--mamba-ts-channel-set-pooling", action="store_true",
                        help="learn permutation-equivariant channel attention and pool mean/std before the shared scan")
    parser.add_argument("--mamba-ts-channel-prototype-pooling", action="store_true",
                        help="learn multiple shared channel-set prototypes before the existing scan")
    parser.add_argument("--mamba-ts-channel-prototype-count", type=int, default=4)
    parser.add_argument("--context-rank", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--source-update-policy", choices=["sequential", "cyclic"], default="sequential",
                        help="source-domain optimizer ordering; cyclic removes fixed last-source recency across epochs")
    parser.add_argument("--relation-pair-policy", choices=["legacy", "round_robin", "contrastive_cluster"],
                        default="legacy", help="source relation-window policy; legacy preserves audited releases")
    parser.add_argument("--forecast-contrastive-pairing",
                        choices=["channel_union", "channel_coverage"],
                        default="channel_union",
                        help="source pair lag set for forecast InfoNCE; channel_coverage avoids rare channel-only lags")
    parser.add_argument("--min-channel-positive-support", type=float, default=0.5,
                        help="minimum source-channel peak support for channel_coverage pairing")
    parser.add_argument("--forecast-contrastive-positive-lags",
                        choices=["all_channel_peaks", "coverage_only"],
                        default="all_channel_peaks",
                        help=("whether channel_coverage also restricts InfoNCE positive labels; "
                              "the historical default preserves existing releases"))
    parser.add_argument("--forecast-contrastive-acf-weighting", action="store_true",
                        help="weight forecast InfoNCE positives by train-only ACF magnitude")
    parser.add_argument("--forecast-contrastive-stability-weighting", action="store_true",
                        help="weight forecast InfoNCE positives by two-half train-prefix ACF stability")
    parser.add_argument("--forecast-target-acf", action="store_true",
                        help=("replace raw-signal affine ACF coefficients with train-only "
                              "future-residual trajectory correlations at the same selected lags"))
    parser.add_argument("--forecast-target-acf-max-pairs", type=int, default=64,
                        help="maximum train-only source-window pairs per lag for forecast-target ACF fitting")
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--predictive-weight", type=float, default=1.0,
                        help="weight of the next-patch prediction loss")
    parser.add_argument("--prediction-target", choices=["raw", "last_value_residual"],
                        default="raw", help="source next-segment target for the transferable head")
    parser.add_argument("--pretrain-task", choices=["masked_reconstruction", "acf_only", "next_patch"],
                        default="masked_reconstruction")
    parser.add_argument("--relation-view", choices=["full", "forecast_state", "channelwise_forecast_state", "forecast_patch", "forecast_affine", "forecast_contrastive", "pooled_patch_contrastive"], default="full",
                        help="ACF relation representation: full temporal grid or final forecast state")
    parser.add_argument("--mamba-lambda", type=float, default=0.1)
    parser.add_argument("--relation-loss-fraction", type=float, default=0.0,
                        help="target weighted ACF relation fraction of the predictive loss; 0 keeps fixed lambda")
    parser.add_argument("--representation-relation-fraction", type=float, default=0.0,
                        help="additional prediction-scaled ACF loss on transferred repr_ states; zero preserves forecast-only InfoNCE")
    parser.add_argument("--channel-patch-relation-fraction", type=float, default=0.0,
                        help=("additional prediction-scaled channelwise ACF InfoNCE on full encoder "
                              "patch trajectories; training-only and zero preserves the forecast-only contract"))
    parser.add_argument("--calibration-channel-patch-relation-fraction", type=float, default=None,
                        help=("optional Channel-Patch fraction applied only to --calibration-source; "
                              "defaults to channel-patch-relation-fraction for historical behavior"))
    parser.add_argument("--channel-patch-acf-confidence-weighting", action="store_true",
                        help="weight channel-patch InfoNCE positives by train-only ACF magnitude")
    parser.add_argument("--channel-patch-stability-weighting", action="store_true",
                        help="weight channel-patch InfoNCE positives by two-half train-prefix ACF stability")
    parser.add_argument("--channel-patch-similarity", choices=["trajectory", "positionwise", "last_patch"], default="trajectory",
                        help="full trajectory, same-position average, or final causal patch similarity for Channel-Patch InfoNCE")
    parser.add_argument("--channel-patch-projector", choices=["identity", "mlp"], default="identity",
                        help=("shared train-only projection before Channel-Patch InfoNCE; the MLP has no "
                              "channel-specific parameters and is omitted from inference"))
    parser.add_argument("--mamba-ts-acf-channel-descriptor", action="store_true",
                        help=("inject a zero-gated shared projection of each channel's fixed train-prefix "
                              "ACF descriptor into patch tokens before the existing single scan"))
    parser.add_argument("--mamba-ts-shared-postscan-mixer", action="store_true",
                        help="apply a channel-count-independent low-rank FiLM mixer after the one shared scan")
    parser.add_argument("--mamba-ts-shared-postscan-mixer-rank", type=int, default=4)
    parser.add_argument("--channel-patch-channel-weighting", choices=["none", "acf_strength"], default="none",
                        help=("train-only channel aggregation for Channel-Patch InfoNCE; acf_strength "
                              "uses normalized selected-peak ACF magnitude without changing pairs"))
    parser.add_argument("--grouped-patch-relation-fraction", type=float, default=0.0,
                        help="prediction-scaled train-only ACF-grouped patch InfoNCE fraction")
    parser.add_argument("--calibration-grouped-patch-relation-fraction", type=float, default=None,
                        help="optional grouped-patch fraction for the target calibration prefix")
    parser.add_argument("--grouped-patch-group-count", type=int, default=4,
                        help="fixed train-prefix ACF groups for grouped patch InfoNCE")
    parser.add_argument("--calibration-relation-loss-fraction", type=float, default=None,
                        help=("optional relation-loss fraction used only for the target-calibration prefix; "
                              "both active/null retain the same predictive calibration task"))
    parser.add_argument("--pooled-patch-variance-weight", type=float, default=0.0,
                        help="pretraining-only anti-collapse weight for pooled_patch_contrastive")
    parser.add_argument("--pooled-patch-variance-target-std", type=float, default=0.2,
                        help="minimum batch standard deviation for each pooled patch feature")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=3601)
    args = parser.parse_args()
    if args.mamba_headdim is None:
        args.mamba_headdim = args.d_model
    if args.mamba_headdim <= 0:
        raise ValueError("mamba-headdim must be positive")
    if args.mamba_ts_shared_postscan_mixer_rank <= 0:
        raise ValueError("mamba-ts-shared-postscan-mixer-rank must be positive")
    _validate_patch_geometry(args.seq_len, args.patch_len, args.patch_stride, args.pretrain_task)
    if not 0 < args.train_fraction < 1 or not 0 <= args.mask_ratio < 1 or args.max_windows <= 0:
        raise ValueError("invalid source split, mask ratio, or window limit")
    if (args.calibration_source is None) != (args.calibration_fraction is None):
        raise ValueError("calibration-source and calibration-fraction must be supplied together")
    if args.calibration_fraction is not None and not 0 < args.calibration_fraction <= 1:
        raise ValueError("calibration-fraction must be in (0, 1]")
    if args.calibration_relation_loss_fraction is not None and args.calibration_source is None:
        raise ValueError("calibration-relation-loss-fraction requires calibration-source")
    if args.calibration_channel_patch_relation_fraction is not None and args.calibration_source is None:
        raise ValueError("calibration-channel-patch-relation-fraction requires calibration-source")
    if args.calibration_grouped_patch_relation_fraction is not None and args.calibration_source is None:
        raise ValueError("calibration-grouped-patch-relation-fraction requires calibration-source")
    if args.reconstruction_weight < 0 or args.predictive_weight < 0:
        raise ValueError("reconstruction-weight and predictive-weight must be non-negative")
    if args.pretrain_task == "masked_reconstruction" and args.reconstruction_weight == 0 and args.mamba_lambda <= 0:
        raise ValueError("reconstruction-weight must be non-negative and at least one objective must be active")
    if args.pretrain_task == "next_patch" and args.predictive_weight == 0 and args.mamba_lambda <= 0:
        raise ValueError("next_patch requires a positive predictive-weight or mamba-lambda")
    if args.pretrain_task == "acf_only" and args.mamba_lambda <= 0:
        raise ValueError("acf_only pretraining requires a positive mamba-lambda")
    if args.source_pred_len <= 0:
        raise ValueError("source-pred-len must be positive")
    if not 0.0 <= args.relation_loss_fraction <= 1.0:
        raise ValueError("relation-loss-fraction must be in [0, 1]")
    if not 0.0 <= args.representation_relation_fraction <= 1.0:
        raise ValueError("representation-relation-fraction must be in [0, 1]")
    if not 0.0 <= args.channel_patch_relation_fraction <= 1.0:
        raise ValueError("channel-patch-relation-fraction must be in [0, 1]")
    if not 0.0 <= args.grouped_patch_relation_fraction <= 1.0:
        raise ValueError("grouped-patch-relation-fraction must be in [0, 1]")
    if (args.calibration_channel_patch_relation_fraction is not None
            and not 0.0 <= args.calibration_channel_patch_relation_fraction <= 1.0):
        raise ValueError("calibration-channel-patch-relation-fraction must be in [0, 1]")
    if (args.calibration_grouped_patch_relation_fraction is not None
            and not 0.0 <= args.calibration_grouped_patch_relation_fraction <= 1.0):
        raise ValueError("calibration-grouped-patch-relation-fraction must be in [0, 1]")
    if (args.calibration_relation_loss_fraction is not None
            and not 0.0 <= args.calibration_relation_loss_fraction <= 1.0):
        raise ValueError("calibration-relation-loss-fraction must be in [0, 1]")
    if not 0.0 < args.min_channel_positive_support <= 1.0:
        raise ValueError("min-channel-positive-support must be in (0, 1]")
    if args.relation_loss_fraction and args.mamba_lambda <= 0:
        raise ValueError("relation-loss-fraction requires a positive mamba-lambda")
    if args.representation_relation_fraction and args.mamba_lambda <= 0:
        raise ValueError("representation-relation-fraction requires a positive mamba-lambda")
    calibration_channel_patch_fraction = (
        args.channel_patch_relation_fraction
        if args.calibration_channel_patch_relation_fraction is None
        else args.calibration_channel_patch_relation_fraction
    )
    calibration_grouped_patch_fraction = (
        args.grouped_patch_relation_fraction
        if args.calibration_grouped_patch_relation_fraction is None
        else args.calibration_grouped_patch_relation_fraction
    )
    any_channel_patch_fraction = max(
        float(args.channel_patch_relation_fraction), float(calibration_channel_patch_fraction)
    )
    any_grouped_patch_fraction = max(
        float(args.grouped_patch_relation_fraction), float(calibration_grouped_patch_fraction)
    )
    if any_channel_patch_fraction and args.mamba_lambda <= 0:
        raise ValueError("channel-patch-relation-fraction requires a positive mamba-lambda")
    if any_channel_patch_fraction and args.relation_view != "forecast_contrastive":
        raise ValueError("channel-patch-relation-fraction requires relation_view=forecast_contrastive")
    if any_grouped_patch_fraction and (args.mamba_lambda <= 0 or args.relation_view != "forecast_contrastive"):
        raise ValueError("grouped-patch-relation-fraction requires positive forecast_contrastive AutoCon")
    if any_grouped_patch_fraction and any_channel_patch_fraction:
        raise ValueError("grouped-patch and channel-patch objectives are mutually exclusive")
    if args.channel_patch_acf_confidence_weighting and args.channel_patch_stability_weighting:
        raise ValueError("only one channel-patch positive weighting mode may be enabled")
    if (args.channel_patch_acf_confidence_weighting or args.channel_patch_stability_weighting) and not any_channel_patch_fraction:
        raise ValueError("channel-patch positive weighting requires channel-patch-relation-fraction")
    if (args.channel_patch_acf_confidence_weighting or args.channel_patch_stability_weighting) and args.relation_view != "forecast_contrastive":
        raise ValueError("channel-patch positive weighting requires relation_view=forecast_contrastive")
    if args.channel_patch_channel_weighting != "none" and not any_channel_patch_fraction:
        raise ValueError("channel-patch channel weighting requires channel-patch-relation-fraction")
    if args.calibration_relation_loss_fraction and args.mamba_lambda <= 0:
        raise ValueError("calibration-relation-loss-fraction requires a positive mamba-lambda")
    if args.pooled_patch_variance_weight < 0:
        raise ValueError("pooled-patch-variance-weight must be non-negative")
    if args.pooled_patch_variance_weight and args.relation_view != "pooled_patch_contrastive":
        raise ValueError("pooled-patch-variance-weight requires pooled_patch_contrastive")
    if args.pooled_patch_variance_weight and args.mamba_lambda <= 0:
        raise ValueError("pooled-patch-variance-weight requires active AutoCon")
    if sum((args.mamba_ts_shared_channel_summary, args.mamba_ts_channel_set_pooling,
            args.mamba_ts_channel_prototype_pooling)) > 1:
        raise ValueError("shared channel context modes are mutually exclusive")
    if args.mamba_ts_channel_prototype_pooling and args.mamba_ts_channel_prototype_count <= 1:
        raise ValueError("mamba-ts-channel-prototype-count must exceed one")
    if args.forecast_contrastive_acf_weighting and args.forecast_contrastive_stability_weighting:
        raise ValueError("only one forecast-contrastive positive weighting mode may be enabled")
    if (args.forecast_contrastive_acf_weighting or args.forecast_contrastive_stability_weighting) and args.relation_view != "forecast_contrastive":
        raise ValueError("forecast-contrastive positive weighting requires relation_view=forecast_contrastive")
    if args.forecast_target_acf and not (
            args.pretrain_task == "next_patch"
            and args.relation_view in {"forecast_affine", "forecast_contrastive"}
    ):
        raise ValueError(
            "forecast-target-acf requires next_patch pretraining with "
            "relation_view=forecast_affine or forecast_contrastive"
        )
    if args.forecast_target_acf_max_pairs < 2:
        raise ValueError("forecast-target-acf-max-pairs must be at least two")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cpu_limit = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu_limit:
        torch.set_num_threads(int(cpu_limit))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    source_specs = [(*_source_spec(item), "source", None) for item in args.source]
    source_names = {name for name, _path, _format, _role, _fraction in source_specs}
    if len(source_names) != len(source_specs):
        raise ValueError("--source names must be unique")
    if args.calibration_source is not None:
        calibration_name, calibration_path, calibration_format = _source_spec(args.calibration_source)
        if calibration_name in source_names:
            raise ValueError("calibration-source must not duplicate a --source name")
        source_specs.append((
            calibration_name, calibration_path, calibration_format,
            "target_calibration", float(args.calibration_fraction),
        ))

    sources = []
    for name, path, data_format, role, calibration_fraction in source_specs:
        raw = _load_values(path, data_format)
        prefix_fraction = args.train_fraction * (
            calibration_fraction if role == "target_calibration" else 1.0
        )
        train = raw[:int(len(raw) * prefix_fraction)]
        prior = fit_signed_acf_prior(
            train, max_lag=min(720, len(train) - 1), top_k=8, min_lag=1,
            min_abs_acf=0.05, confidence_z=1.96, min_separation=2,
            detrend="mean", channel_chunk_size=64,
        )
        selected_lags = [lag for lag in prior.common_lags(top_k=8) if lag < len(train)] or [1]
        channel_peaks = _channel_peak_matrix(prior, top_k=8)
        if args.forecast_target_acf:
            relation_channel_acf, target_acf_metadata = _forecast_target_acf_table(
                train, args.seq_len, args.source_pred_len, channel_peaks,
                max_pairs=args.forecast_target_acf_max_pairs,
            )
            peak_channel_index = np.repeat(np.arange(channel_peaks.shape[0]), channel_peaks.shape[1])
            peak_lag_index = channel_peaks.reshape(-1)
            peak_valid = peak_lag_index > 0
            raw_peak_values = prior.acf[peak_channel_index[peak_valid], peak_lag_index[peak_valid]]
            task_peak_values = relation_channel_acf[peak_channel_index[peak_valid], peak_lag_index[peak_valid]]
            target_acf_metadata.update({
                "mean_abs_raw_task_delta": float(np.abs(raw_peak_values - task_peak_values).mean()),
                "raw_task_sign_agreement": float(np.mean(np.sign(raw_peak_values) == np.sign(task_peak_values))),
            })
        else:
            relation_channel_acf = prior.acf.astype(np.float32)
            target_acf_metadata = None
        acf_descriptor = build_acf_channel_descriptor(prior.acf, channel_peaks)
        channel_pair_lags = sorted({int(lag) for lag in channel_peaks.ravel() if lag > 0}) or [1]
        coverage_pair_lags, channel_lag_support = _coverage_pair_lags(
            channel_peaks, args.min_channel_positive_support
        )
        forecast_pair_lags = (
            coverage_pair_lags
            if args.forecast_contrastive_pairing == "channel_coverage"
            else channel_pair_lags
        )
        contrastive_channel_lags = channel_peaks
        if (args.relation_view == "forecast_contrastive"
                and args.forecast_contrastive_pairing == "channel_coverage"
                and args.forecast_contrastive_positive_lags == "coverage_only"):
            contrastive_channel_lags = _restrict_channel_peaks(channel_peaks, forecast_pair_lags)
        contrastive_channel_weights = (
            _channel_peak_stability_weights(train, contrastive_channel_lags)
            if (args.forecast_contrastive_stability_weighting or args.channel_patch_stability_weighting) else None
        )
        channel_patch_channel_weights = (
            _channel_peak_strength_weights(prior.acf, contrastive_channel_lags)
            if args.channel_patch_channel_weighting == "acf_strength" else None
        )
        if any_grouped_patch_fraction:
            group_assignment, grouped_channel_acf, grouped_channel_peaks = _acf_grouped_patch_prior(
                train, channel_peaks, args.grouped_patch_group_count
            )
            group_sizes = np.bincount(
                group_assignment, minlength=args.grouped_patch_group_count
            ).astype(int).tolist()
        else:
            group_assignment = grouped_channel_acf = grouped_channel_peaks = None
            group_sizes = None
        sources.append({
            "name": name, "path": path, "format": data_format,
            "dataset": Windows(
                train, args.seq_len, args.max_windows,
                next_patch=args.pretrain_task == "next_patch", patch_len=args.source_pred_len,
                relation_lags=(
                    forecast_pair_lags
                    if args.relation_view == "forecast_contrastive"
                    else channel_pair_lags
                    if args.relation_view in {"channelwise_forecast_state", "forecast_patch", "forecast_affine", "forecast_contrastive"}
                    else selected_lags
                ),
                relation_pair_policy=args.relation_pair_policy,
            ),
            "acf": prior.acf.mean(axis=0).astype(np.float32), "selected_lags": selected_lags,
            "channel_acf": relation_channel_acf,
            "channel_selected_lags": channel_peaks,
            "contrastive_channel_selected_lags": contrastive_channel_lags,
            "contrastive_channel_positive_weights": contrastive_channel_weights,
            "acf_descriptor": acf_descriptor,
            "channel_patch_channel_weights": channel_patch_channel_weights,
            "group_assignment": group_assignment,
            "grouped_channel_acf": grouped_channel_acf,
            "grouped_channel_peaks": grouped_channel_peaks,
            "metadata": {"name": name, "path": str(path), "sha256": _sha256(path),
                         "role": role, "calibration_fraction": calibration_fraction,
                         "acf_channel_descriptor": bool(args.mamba_ts_acf_channel_descriptor),
                         "forecast_target_acf": bool(args.forecast_target_acf),
                         "forecast_target_acf_metadata": target_acf_metadata,
                         "relation_loss_fraction": float(
                             args.calibration_relation_loss_fraction
                             if role == "target_calibration" and args.calibration_relation_loss_fraction is not None
                             else args.relation_loss_fraction
                         ),
                         "channel_patch_relation_fraction": float(
                             calibration_channel_patch_fraction
                             if role == "target_calibration"
                             else args.channel_patch_relation_fraction
                         ),
                         "train_prefix_fraction": float(prefix_fraction),
                         "raw_rows": int(len(raw)), "train_rows": int(len(train)),
                         "channels": int(raw.shape[1]), "selected_lags": selected_lags,
                         "forecast_pair_lags": forecast_pair_lags,
                         "forecast_positive_pair_lags": sorted({int(lag) for lag in contrastive_channel_lags.ravel() if lag > 0}),
                         "stability_weight_mean": float(contrastive_channel_weights[contrastive_channel_lags > 0].mean())
                         if contrastive_channel_weights is not None and np.any(contrastive_channel_lags > 0) else None,
                         "channel_patch_channel_weighting": args.channel_patch_channel_weighting,
                         "channel_patch_channel_weight_min": float(channel_patch_channel_weights.min())
                         if channel_patch_channel_weights is not None else None,
                         "channel_patch_channel_weight_max": float(channel_patch_channel_weights.max())
                         if channel_patch_channel_weights is not None else None,
                         "channel_patch_channel_weight_mean": float(channel_patch_channel_weights.mean())
                         if channel_patch_channel_weights is not None else None,
                         "grouped_patch_relation_fraction": float(
                             calibration_grouped_patch_fraction
                             if role == "target_calibration" else args.grouped_patch_relation_fraction
                         ),
                         "grouped_patch_group_count": (
                             int(args.grouped_patch_group_count) if any_grouped_patch_fraction else None
                         ),
                         "grouped_patch_group_sizes": group_sizes,
                         "channel_lag_support": channel_lag_support},
        })
        sources[-1]["channel_pair_lags"] = channel_pair_lags
        sources[-1]["forecast_pair_lags"] = forecast_pair_lags

    checkpoints = [_checkpoint_spec(item) for item in args.initial_checkpoint]
    initial_states = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu")
        initial_states.append(payload.get("model", payload))
    common = set.intersection(*(set(state) for state in initial_states))
    initial_core = {
        key: torch.stack([state[key].float() for state in initial_states]).mean(dim=0)
        for key in sorted(common) if key.startswith(CORE_PREFIXES)
        and all(state[key].shape == initial_states[0][key].shape for state in initial_states)
    }
    if not any(key.startswith("encoder.") for key in initial_core):
        raise ValueError("initial checkpoints have no compatible MambaTSv3 encoder core")

    models = []
    for source in sources:
        model = MambaTS.Model(_config(args, source["metadata"]["channels"])).cuda().train()
        state = model.state_dict()
        with torch.no_grad():
            for key, value in initial_core.items():
                if key in state and state[key].shape == value.shape:
                    state[key].copy_(value.to(device=state[key].device, dtype=state[key].dtype))
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(CORE_PREFIXES)
        if args.mamba_ts_acf_channel_descriptor:
            with torch.no_grad():
                model.ts_acf_descriptor_values.copy_(
                    torch.as_tensor(source["acf_descriptor"], device=model.ts_acf_descriptor_values.device)
                )
        models.append(model)
    master = models[0]
    core_keys = _core_keys(master.state_dict())
    decoder = nn.Linear(master.patch_count * args.d_model, args.source_pred_len).cuda()
    pooled_patch_projector = nn.Sequential(
        nn.Linear(2 * args.d_model, args.d_model), nn.GELU(),
        nn.LayerNorm(args.d_model), nn.Linear(args.d_model, args.d_model),
    ).cuda()
    channel_patch_projector = (
        nn.Identity()
        if args.channel_patch_projector == "identity"
        else nn.Sequential(
            nn.LayerNorm(args.d_model), nn.Linear(args.d_model, args.d_model),
            nn.GELU(), nn.Linear(args.d_model, args.d_model),
        )
    ).cuda()
    mask_token = nn.Parameter(torch.zeros(args.patch_len, device="cuda"))
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in master.named_parameters() if parameter.requires_grad]
        + list(decoder.parameters()) + list(pooled_patch_projector.parameters())
        + list(channel_patch_projector.parameters()) + [mask_token],
        lr=args.learning_rate, weight_decay=1e-5,
    )

    history = []
    for epoch in range(args.epochs):
        losses, reconstruction_losses, relation_losses, weighted_relation_losses, active = [], [], [], [], []
        representation_relation_losses, weighted_representation_relation_losses, representation_active = [], [], []
        channel_patch_relation_losses, weighted_channel_patch_relation_losses, channel_patch_active = [], [], []
        grouped_patch_relation_losses, weighted_grouped_patch_relation_losses, grouped_patch_active = [], [], []
        variance_losses, pooled_stds = [], []
        source_order = list(range(len(sources)))
        if args.source_update_policy == "cyclic" and len(source_order) > 1:
            shift = epoch % len(source_order)
            source_order = source_order[shift:] + source_order[:shift]
        for source_index in source_order:
            source, model = sources[source_index], models[source_index]
            source_relation_loss_fraction = float(source["metadata"]["relation_loss_fraction"])
            source_channel_patch_fraction = float(source["metadata"]["channel_patch_relation_fraction"])
            source_grouped_patch_fraction = float(source["metadata"]["grouped_patch_relation_fraction"])
            loader = DataLoader(
                source["dataset"],
                batch_sampler=ACFLagPairBatchSampler(
                    source["dataset"], batch_size=args.batch_size,
                    lags=(
                        source["forecast_pair_lags"]
                        if args.relation_view == "forecast_contrastive"
                        else source["channel_pair_lags"]
                        if args.relation_view in {"channelwise_forecast_state", "forecast_patch", "forecast_affine", "forecast_contrastive"}
                        else source["selected_lags"]
                    ), drop_last=False,
                    seed=args.seed + epoch, lag_balance="balanced",
                    origins=source["dataset"].starts,
                    batch_packing=(
                        "local" if args.relation_pair_policy == "contrastive_cluster" else "random"
                    ),
                ),
                num_workers=0,
            )
            relation = None if args.mamba_lambda == 0 else ACFRelationLoss(
                source["channel_acf"] if args.relation_view == "channelwise_forecast_state" else source["acf"],
                temperature=0.2, relation_temperature=0.25,
                sign_weight=0.25, sign_margin=0.1,
                relation_scope="global" if args.relation_view != "full" else "both",
                selected_lags=source["selected_lags"], lag_tolerance=0,
                target_mode="hard_selected", soft_min_abs=0.05,
                diagnostic_min_abs=0.05, min_lag=0,
                channel_selected_lags=(
                    source["channel_selected_lags"]
                    if args.relation_view == "channelwise_forecast_state" else None
                ),
                normalize_global_active_anchors=args.relation_view == "channelwise_forecast_state",
            ).cuda()
            forecast_relation = None
            channel_patch_relation = None
            grouped_patch_relation = None
            pooled_patch_relation = None
            pooled_patch_variance = None
            if args.relation_view in {"forecast_patch", "forecast_affine", "forecast_contrastive"}:
                relation_class = (
                    ForecastLagPairAffineLoss
                    if args.relation_view == "forecast_affine"
                    else (
                        ForecastLagPairContrastiveLoss
                        if args.relation_view == "forecast_contrastive"
                        else ForecastLagPairConsistencyLoss
                    )
                )
                relation_kwargs = {
                    "selected_lags": source["channel_pair_lags"],
                }
                if args.relation_view == "forecast_contrastive":
                    relation_kwargs["channel_selected_lags"] = source["contrastive_channel_selected_lags"]
                    relation_kwargs["acf_confidence_weighting"] = args.forecast_contrastive_acf_weighting
                    relation_kwargs["channel_positive_weights"] = source["contrastive_channel_positive_weights"]
                forecast_relation = relation_class(source["channel_acf"], **relation_kwargs).cuda()
                if source_channel_patch_fraction:
                    channel_patch_relation = ChannelPatchLagContrastiveLoss(
                        source["channel_acf"], selected_lags=source["channel_pair_lags"],
                        channel_selected_lags=source["contrastive_channel_selected_lags"],
                        patch_similarity=args.channel_patch_similarity,
                        acf_confidence_weighting=args.channel_patch_acf_confidence_weighting,
                        channel_positive_weights=(
                            source["contrastive_channel_positive_weights"]
                            if args.channel_patch_stability_weighting else None
                        ),
                        channel_weights=source["channel_patch_channel_weights"],
                    ).cuda()
                if source_grouped_patch_fraction:
                    grouped_patch_relation = ChannelPatchLagContrastiveLoss(
                        source["grouped_channel_acf"], selected_lags=source["channel_pair_lags"],
                        channel_selected_lags=source["grouped_channel_peaks"],
                        patch_similarity=args.channel_patch_similarity,
                    ).cuda()
            elif args.relation_view == "pooled_patch_contrastive" and args.mamba_lambda:
                pooled_patch_relation = PooledPatchLagContrastiveLoss(
                    source["acf"], selected_lags=source["selected_lags"], temperature=0.2,
                ).cuda()
                if args.pooled_patch_variance_weight:
                    pooled_patch_variance = PooledPatchVarianceLoss(
                        target_std=args.pooled_patch_variance_target_std,
                    ).cuda()
            for packed in loader:
                if args.pretrain_task == "next_patch":
                    starts, values, future = packed
                else:
                    starts, values = packed
                    future = None
                if model is not master:
                    _copy_core(model, master, core_keys)
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                values = values.cuda(non_blocking=True)
                if args.pretrain_task == "next_patch":
                    normalized, restore = model._normalize(values)
                    encoded = model._encode(normalized)
                    mean, scale = restore[:2] if isinstance(restore, tuple) else (None, None)
                    if mean is None:
                        raise RuntimeError("next_patch requires MeanStd or ReVIN normalization")
                    normalized_future = (future.cuda(non_blocking=True) - mean) / scale
                    last = normalized[:, -1:, :]
                    target = (
                        normalized_future - last
                        if args.prediction_target == "last_value_residual"
                        else normalized_future
                    )
                    prediction = decoder(encoded.reshape(encoded.shape[0], encoded.shape[1], -1))
                    reconstruction_loss = F.smooth_l1_loss(prediction, target.transpose(1, 2))
                elif args.pretrain_task == "masked_reconstruction" and args.reconstruction_weight:
                    mask = torch.rand(values.shape[0], master.patch_count, device=values.device) < args.mask_ratio
                    if not bool(mask.any()):
                        mask[:, 0] = True
                    encoded_input, target = _masked_input(model, values, mask, mask_token)
                    encoded = model._encode(encoded_input)
                    reconstruction = nn.functional.linear(
                        encoded, decoder.weight[:master.patch_len, :master.d_model],
                        decoder.bias[:master.patch_len],
                    )
                    error = (reconstruction - target).square().mean(dim=-1)
                    reconstruction_loss = error.masked_select(mask[:, None, :].expand_as(error)).mean()
                else:
                    normalized, _ = model._normalize(values)
                    encoded = model._encode(normalized)
                    reconstruction_loss = encoded.square().mean() * 0.0
                if relation is None:
                    contrastive_loss = reconstruction_loss.detach() * 0.0
                else:
                    if args.relation_view in {"forecast_patch", "forecast_affine", "forecast_contrastive"}:
                        if args.pretrain_task != "next_patch":
                            raise ValueError("forecast relation view requires next_patch pretraining")
                        relation_forecast = (
                            prediction + last.transpose(1, 2)
                            if args.prediction_target == "last_value_residual"
                            else prediction
                        )
                        contrastive_loss = forecast_relation(
                            relation_forecast.transpose(1, 2), starts.long().cuda(non_blocking=True)
                        ).mean()
                        active.append(float(forecast_relation.last_active_anchor_fraction))
                    elif args.relation_view == "pooled_patch_contrastive":
                        relation_features = _pooled_patch_representation(encoded, pooled_patch_projector)
                        contrastive_loss = pooled_patch_relation(
                            relation_features,
                            starts.long().cuda(non_blocking=True),
                        ).mean()
                        variance_loss = (
                            pooled_patch_variance(relation_features)
                            if pooled_patch_variance is not None
                            else contrastive_loss.detach() * 0.0
                        )
                        contrastive_loss = contrastive_loss + args.pooled_patch_variance_weight * variance_loss
                        variance_losses.append(float(variance_loss.detach()))
                        if pooled_patch_variance is not None:
                            pooled_stds.append(float(pooled_patch_variance.last_mean_std))
                        active.append(float(pooled_patch_relation.last_active_anchor_fraction))
                    else:
                        relation_features = model._representation(encoded)
                        if args.relation_view == "forecast_state":
                            relation_features = relation_features[:, -1:, :]
                        if args.relation_view == "channelwise_forecast_state":
                            channel_states = encoded[:, :, -1, :]
                            channel_features = model.repr_head(
                                model.repr_norm(model.repr_projection(channel_states))
                            )
                            contrastive_loss = relation.channelwise_global_contrastive_loss(
                                channel_features, starts.long().cuda(non_blocking=True)
                            ).mean()
                        else:
                            contrastive_loss = _relation_loss(
                                relation, relation_features, starts.long().cuda(non_blocking=True)
                            )
                        active.append(float(relation.last_global_contrastive_active_anchor_fraction.detach().cpu()))
                if args.representation_relation_fraction and relation is not None:
                    representation_loss = relation.global_contrastive_loss(
                        F.normalize(model._representation(encoded), dim=-1),
                        starts.long().cuda(non_blocking=True),
                    ).mean()
                    representation_active.append(
                        float(relation.last_global_contrastive_active_anchor_fraction.detach().cpu())
                    )
                else:
                    representation_loss = reconstruction_loss.detach() * 0.0
                if source_channel_patch_fraction:
                    channel_patch_loss = channel_patch_relation(
                        channel_patch_projector(encoded), starts.long().cuda(non_blocking=True)
                    ).mean()
                    channel_patch_active.append(float(channel_patch_relation.last_active_anchor_fraction))
                else:
                    channel_patch_loss = reconstruction_loss.detach() * 0.0
                if source_grouped_patch_fraction:
                    grouped_features = _grouped_patch_representation(
                        encoded, source["group_assignment"], args.grouped_patch_group_count
                    )
                    grouped_patch_loss = grouped_patch_relation(
                        grouped_features, starts.long().cuda(non_blocking=True)
                    ).mean()
                    grouped_patch_active.append(float(grouped_patch_relation.last_active_anchor_fraction))
                else:
                    grouped_patch_loss = reconstruction_loss.detach() * 0.0
                task_weight = (
                    args.predictive_weight if args.pretrain_task == "next_patch"
                    else args.reconstruction_weight
                )
                if source_relation_loss_fraction and args.mamba_lambda:
                    relation_weight = (
                        source_relation_loss_fraction
                        * (task_weight * reconstruction_loss).detach()
                        / contrastive_loss.detach().abs().clamp_min(1e-6)
                    )
                else:
                    relation_weight = contrastive_loss.new_tensor(args.mamba_lambda)
                weighted_relation = relation_weight * contrastive_loss
                if args.representation_relation_fraction:
                    representation_weight = (
                        args.representation_relation_fraction
                        * (task_weight * reconstruction_loss).detach()
                        / representation_loss.detach().abs().clamp_min(1e-6)
                    )
                else:
                    representation_weight = representation_loss.new_zeros(())
                weighted_representation_relation = representation_weight * representation_loss
                if source_channel_patch_fraction:
                    channel_patch_weight = (
                        source_channel_patch_fraction
                        * (task_weight * reconstruction_loss).detach()
                        / channel_patch_loss.detach().abs().clamp_min(1e-6)
                    )
                else:
                    channel_patch_weight = channel_patch_loss.new_zeros(())
                weighted_channel_patch_relation = channel_patch_weight * channel_patch_loss
                if source_grouped_patch_fraction:
                    grouped_patch_weight = (
                        source_grouped_patch_fraction * (task_weight * reconstruction_loss).detach()
                        / grouped_patch_loss.detach().abs().clamp_min(1e-6)
                    )
                else:
                    grouped_patch_weight = grouped_patch_loss.new_zeros(())
                weighted_grouped_patch_relation = grouped_patch_weight * grouped_patch_loss
                loss = (
                    task_weight * reconstruction_loss + weighted_relation
                    + weighted_representation_relation + weighted_channel_patch_relation
                    + weighted_grouped_patch_relation
                )
                loss.backward()
                if model is not master:
                    master_parameters = dict(master.named_parameters())
                    for name, parameter in model.named_parameters():
                        if name in master_parameters and parameter.requires_grad:
                            master_parameters[name].grad = parameter.grad.detach().clone() if parameter.grad is not None else None
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                reconstruction_losses.append(float(reconstruction_loss.detach()))
                relation_losses.append(float(contrastive_loss.detach()))
                weighted_relation_losses.append(float(weighted_relation.detach()))
                representation_relation_losses.append(float(representation_loss.detach()))
                weighted_representation_relation_losses.append(float(weighted_representation_relation.detach()))
                channel_patch_relation_losses.append(float(channel_patch_loss.detach()))
                weighted_channel_patch_relation_losses.append(float(weighted_channel_patch_relation.detach()))
                grouped_patch_relation_losses.append(float(grouped_patch_loss.detach()))
                weighted_grouped_patch_relation_losses.append(float(weighted_grouped_patch_relation.detach()))
        record = {
            "epoch": epoch + 1, "mean_loss": float(np.mean(losses)),
            "mean_reconstruction_loss": float(np.mean(reconstruction_losses)),
            "mean_autocon_loss": float(np.mean(relation_losses)),
            "mean_weighted_autocon_loss": float(np.mean(weighted_relation_losses)),
            "mean_representation_relation_loss": float(np.mean(representation_relation_losses)),
            "mean_weighted_representation_relation_loss": float(np.mean(weighted_representation_relation_losses)),
            "relation_loss_fraction": float(args.relation_loss_fraction),
            "representation_relation_fraction": float(args.representation_relation_fraction),
            "channel_patch_relation_fraction": float(args.channel_patch_relation_fraction),
            "grouped_patch_relation_fraction": float(args.grouped_patch_relation_fraction),
            "channel_patch_similarity": args.channel_patch_similarity,
            "channel_patch_channel_weighting": args.channel_patch_channel_weighting,
            "channel_patch_projector": args.channel_patch_projector,
            "calibration_channel_patch_relation_fraction": (
                None if args.calibration_channel_patch_relation_fraction is None
                else float(args.calibration_channel_patch_relation_fraction)
            ),
            "source_channel_patch_relation_fractions": {
                source["metadata"]["name"]: float(source["metadata"]["channel_patch_relation_fraction"])
                for source in sources
            },
            "source_grouped_patch_relation_fractions": {
                source["metadata"]["name"]: float(source["metadata"]["grouped_patch_relation_fraction"])
                for source in sources
            },
            "source_relation_loss_fractions": {
                source["metadata"]["name"]: float(source["metadata"]["relation_loss_fraction"])
                for source in sources
            },
            "source_update_order": [sources[index]["metadata"]["name"] for index in source_order],
            "global_relation_active_fraction": float(np.mean(active)) if active else 0.0,
            "representation_relation_active_fraction": float(np.mean(representation_active)) if representation_active else 0.0,
            "mean_channel_patch_relation_loss": float(np.mean(channel_patch_relation_losses)),
            "mean_weighted_channel_patch_relation_loss": float(np.mean(weighted_channel_patch_relation_losses)),
            "channel_patch_relation_active_fraction": float(np.mean(channel_patch_active)) if channel_patch_active else 0.0,
            "mean_grouped_patch_relation_loss": float(np.mean(grouped_patch_relation_losses)),
            "mean_weighted_grouped_patch_relation_loss": float(np.mean(weighted_grouped_patch_relation_losses)),
            "grouped_patch_relation_active_fraction": float(np.mean(grouped_patch_active)) if grouped_patch_active else 0.0,
            "mean_pooled_patch_variance_loss": float(np.mean(variance_losses)) if variance_losses else 0.0,
            "mean_pooled_patch_feature_std": float(np.mean(pooled_stds)) if pooled_stds else 0.0,
            "batches": len(losses),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    shared_state = {key: value.detach().cpu() for key, value in master.state_dict().items() if key in core_keys}
    payload = {
        "model": shared_state,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sources": [source["metadata"] for source in sources],
        "initial_checkpoints": [str(path) for path in checkpoints],
        "initial_checkpoint_sha256": [_sha256(path) for path in checkpoints],
        "shared_keys": core_keys, "history": history,
        "test_metrics_produced": False,
        "prediction_target": args.prediction_target,
    }
    if args.pretrain_task == "next_patch":
        payload["pretrain_prediction_head"] = {
            key: value.detach().cpu() for key, value in decoder.state_dict().items()
        }
    if args.relation_view == "pooled_patch_contrastive":
        payload["pretrain_pooled_patch_projector"] = {
            key: value.detach().cpu() for key, value in pooled_patch_projector.state_dict().items()
        }
    if args.channel_patch_projector == "mlp":
        payload["pretrain_channel_patch_projector"] = {
            key: value.detach().cpu() for key, value in channel_patch_projector.state_dict().items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    json_payload = {key: value for key, value in payload.items() if key not in {"model", "pretrain_prediction_head", "pretrain_pooled_patch_projector", "pretrain_channel_patch_projector"}}
    if "pretrain_prediction_head" in payload:
        json_payload["pretrain_prediction_head"] = {
            key: {"shape": list(value.shape), "sha256": _tensor_digest(value)}
            for key, value in payload["pretrain_prediction_head"].items()
        }
    if "pretrain_pooled_patch_projector" in payload:
        json_payload["pretrain_pooled_patch_projector"] = {
            key: {"shape": list(value.shape), "sha256": _tensor_digest(value)}
            for key, value in payload["pretrain_pooled_patch_projector"].items()
        }
    if "pretrain_channel_patch_projector" in payload:
        json_payload["pretrain_channel_patch_projector"] = {
            key: {"shape": list(value.shape), "sha256": _tensor_digest(value)}
            for key, value in payload["pretrain_channel_patch_projector"].items()
        }
    args.output.with_suffix(".json").write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "test_metrics_produced": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
