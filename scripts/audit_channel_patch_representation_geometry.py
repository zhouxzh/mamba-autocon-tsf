#!/usr/bin/env python3
"""Read-only source-train representation geometry audit for channel-patch SSL."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import load_numeric_array_file
from models import MambaTS
from utils.acf_guidance import fit_signed_acf_prior


DEFAULT_ROOT = ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006/pems_bay"
DEFAULT_OUTPUT = ROOT / "artifacts/exploratory/channel_patch_representation_geometry_20261007"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def config(saved: dict, channels: int) -> SimpleNamespace:
    values = saved["config"]
    return SimpleNamespace(
        seq_len=int(values["seq_len"]), pred_len=int(values["source_pred_len"]),
        enc_in=channels, c_out=channels, d_model=int(values["d_model"]),
        d_ff=int(values["d_ff"]), patch_len=int(values["patch_len"]),
        patch_stride=int(values["patch_stride"]), dropout=0.1, e_layers=1,
        mamba_d_state=int(values["mamba_d_state"]), mamba_d_conv=int(values["mamba_d_conv"]),
        mamba_expand=int(values["mamba_expand"]), mamba_headdim=int(values["mamba_headdim"]),
        mamba_ts_variant=str(values.get("mamba_ts_variant", "v3")),
        mamba_ts_mixer_type=str(values.get("mamba_ts_mixer_type", "mamba1")),
        mamba_ts_context_rank=int(values["context_rank"]),
        mamba_ts_shared_pretrain_mode=True, mamba_ts_channel_agnostic_scan=True,
        AutoCon=True, AutoCon_wnorm="MeanStd", task_name="long_term_forecast",
    )


def source_spec(raw: str) -> tuple[str, Path, str]:
    name, path, data_format = raw.split(":", 2)
    return name, Path(path).resolve(), data_format


def load_model(saved: dict, channels: int) -> torch.nn.Module:
    model = MambaTS.Model(config(saved, channels)).cuda().eval()
    model.load_state_dict(saved["model"], strict=False)
    return model


def pair_shifts(prior, max_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = prior.n_channels
    positive = np.empty(channels, dtype=np.int64)
    negative = np.empty(channels, dtype=np.int64)
    signs = np.empty(channels, dtype=np.float32)
    for channel, peaks in enumerate(prior.peaks):
        allowed = [peak for peak in peaks if peak.lag <= max_lag]
        if allowed:
            peak = allowed[0]
        else:
            lag = int(np.argmax(np.abs(prior.acf[channel, 1:max_lag + 1])) + 1)
            peak = SimpleNamespace(lag=lag, value=float(prior.acf[channel, lag]))
        positive[channel], signs[channel] = int(peak.lag), float(np.sign(peak.value) or 1.0)
        candidates = np.arange(1, max_lag + 1, dtype=np.int64)
        candidates = candidates[candidates != positive[channel]]
        negative[channel] = int(candidates[np.argmin(np.abs(prior.acf[channel, candidates]))])
    return positive, negative, signs


@torch.no_grad()
def geometry(model, values: np.ndarray, prior, seed: int, anchors: int, batch_size: int) -> dict:
    seq_len, horizon = model.seq_len, model.pred_len
    max_lag = min(720, prior.max_lag, len(values) - seq_len - horizon - 1)
    pos, neg, signs = pair_shifts(prior, max_lag)
    safe_end = len(values) - seq_len - horizon - max(int(pos.max()), int(neg.max()))
    origins = np.linspace(0, safe_end, anchors, dtype=np.int64)
    total_positive, total_negative, count = 0.0, 0.0, 0
    sign_tensor = torch.as_tensor(signs, device="cuda")
    for origin in origins:
        shifts = np.unique(np.concatenate(([0], pos, neg))).astype(np.int64)
        encoded = {}
        for begin in range(0, len(shifts), batch_size):
            part = shifts[begin:begin + batch_size]
            windows = torch.as_tensor(
                np.stack([values[origin + shift:origin + shift + seq_len] for shift in part]),
                dtype=torch.float32, device="cuda",
            )
            normalized, _ = model._normalize(windows)
            states = model._encode(normalized).flatten(start_dim=2)
            for index, shift in enumerate(part.tolist()):
                encoded[int(shift)] = states[index]
        anchor = encoded[0]
        channel_index = torch.arange(anchor.shape[0], device="cuda")
        positive = torch.stack([encoded[int(shift)][channel] for channel, shift in enumerate(pos.tolist())])
        negative = torch.stack([encoded[int(shift)][channel] for channel, shift in enumerate(neg.tolist())])
        pos_cos = F.cosine_similarity(anchor, positive, dim=-1) * sign_tensor
        neg_cos = F.cosine_similarity(anchor, negative, dim=-1).abs()
        total_positive += float(pos_cos.sum())
        total_negative += float(neg_cos.sum())
        count += int(anchor.shape[0])
    positive_mean = total_positive / count
    negative_mean = total_negative / count
    return {"signed_positive_cosine": positive_mean, "low_acf_negative_abs_cosine": negative_mean,
            "alignment_margin": positive_mean - negative_mean, "channel_pairs": count,
            "anchors": int(len(origins)), "max_lag": int(max_lag)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-checkpoint", type=Path, default=DEFAULT_ROOT / "s6026_forecast_pretrain.pt")
    parser.add_argument("--channel-patch-checkpoint", type=Path, default=DEFAULT_ROOT / "s6026_channel_patch_pretrain.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "audit.json")
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.anchors < 2 or args.batch_size < 1:
        raise ValueError("anchors must be >=2 and batch-size positive")
    cpu = os.environ.get("AUTOCON_CPU_THREAD_LIMIT", "").strip()
    if cpu:
        torch.set_num_threads(int(cpu))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    forecast = torch.load(args.forecast_checkpoint, map_location="cpu", weights_only=False)
    channel = torch.load(args.channel_patch_checkpoint, map_location="cpu", weights_only=False)
    if any(item.get("test_metrics_produced") for item in (forecast, channel)):
        raise ValueError("geometry audit requires validation-only checkpoints")
    if float(channel["config"].get("channel_patch_relation_fraction", 0.0)) != .05:
        raise ValueError("channel-patch objective contract mismatch")
    source_specs = channel["config"]["source"]
    rows = []
    for raw_spec in source_specs:
        name, path, data_format = source_spec(raw_spec)
        raw = load_numeric_array_file(path, data_format=data_format).astype(np.float32)
        train = raw[:int(len(raw) * float(channel["config"]["train_fraction"]))]
        prior = fit_signed_acf_prior(train, max_lag=min(720, len(train) - 1), top_k=8,
                                    min_lag=1, min_abs_acf=.05, confidence_z=1.96,
                                    min_separation=2, detrend="mean", channel_chunk_size=64)
        forecast_model = load_model(forecast, train.shape[1])
        channel_model = load_model(channel, train.shape[1])
        forecast_stats = geometry(forecast_model, train, prior, args.seed, args.anchors, args.batch_size)
        channel_stats = geometry(channel_model, train, prior, args.seed, args.anchors, args.batch_size)
        rows.append({"source": name, "train_rows": int(len(train)), "channels": int(train.shape[1]),
                     "forecast": forecast_stats, "channel_patch": channel_stats,
                     "margin_change": channel_stats["alignment_margin"] - forecast_stats["alignment_margin"]})
        del forecast_model, channel_model
        torch.cuda.empty_cache()
    result = {"validation_only": True, "test_metrics_produced": False,
              "scope": "source chronological train prefixes only; no target validation/test windows",
              "rows": rows,
              "mean_margin_change": float(np.mean([row["margin_change"] for row in rows])),
              "representation_geometry_supported": bool(all(row["margin_change"] > 0 for row in rows))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
