#!/usr/bin/env python3
"""Run the approved ACE2 synthetic S-pilot with an auditable task ledger.

The pilot is deliberately self-contained: it generates the registered DGPs,
trains a small channel-wise lag forecaster, and evaluates all five pre-registered
relation arms. It does not read repository datasets, checkpoints, or test
artifacts. The implementation is a pilot operationalization, not evidence for
the historical L0 results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


PHASE = "S-pilot"
DGP_NAMES = tuple(f"G{i}" for i in range(1, 9))
ARMS = ("A_signed", "N_null", "U_unsigned", "S_shuffle", "R_random")
CONFIG = {
    "burn_in": 2048,
    "source_train_length": 8192,
    "target_train_length": 8192,
    "target_val_length": 2048,
    "target_test_length": 4096,
    "channels": 8,
    "input_length": 192,
    "horizon": 96,
    "label_fraction": 0.10,
    "replicates": 4,
    "optimization_seeds": (7101, 7102),
    "epochs": 12,
    "batch_size": 64,
    "learning_rate": 2e-3,
    "relation_weight": 0.05,
    "max_lag": 96,
}


def phase_config(phase: str) -> dict[str, object]:
    if phase == "S-pilot":
        return {"migrations": ("M0",), "noise_levels": ("low",), "labels": (0.10,), "horizons": (96,), "replicates": 4, "optimization_seeds": (7101, 7102), "expected": 320}
    if phase == "S-core":
        return {"migrations": ("M0", "M1", "M2"), "noise_levels": ("low",), "labels": (0.10,), "horizons": (96, 192), "replicates": 20, "optimization_seeds": (8101, 8102, 8103), "expected": 14400}
    if phase == "S-extension":
        return {"migrations": ("M0", "M1", "M2"), "noise_levels": ("low", "high"), "labels": (0.10, 0.25), "horizons": (96, 192), "replicates": 20, "optimization_seeds": (8101, 8102, 8103), "expected": 43200}
    if phase == "S-full":
        return {"migrations": ("M0", "M1", "M2"), "noise_levels": ("low", "high"), "labels": (0.10, 0.25), "horizons": (96, 192), "replicates": 20, "optimization_seeds": (8101, 8102, 8103), "expected": 57600}
    raise ValueError(f"unsupported phase {phase}")


@dataclass(frozen=True)
class Relation:
    lags: np.ndarray
    signs: np.ndarray


def _stable_seed(*parts: object) -> int:
    raw = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def _oscillator(rng: np.random.Generator, length: int, period: float, channels: int) -> np.ndarray:
    out = np.zeros((length, channels), dtype=np.float32)
    radius = 0.995
    a1 = 2.0 * radius * math.cos(2.0 * math.pi / period)
    a2 = -(radius**2)
    noise = rng.normal(size=(length, channels)).astype(np.float32)
    out[0] = noise[0]
    out[1] = noise[1]
    for t in range(2, length):
        out[t] = a1 * out[t - 1] + a2 * out[t - 2] + 0.15 * noise[t]
    return out


def _ar1(rng: np.random.Generator, length: int, coeff: float, channels: int) -> np.ndarray:
    out = np.zeros((length, channels), dtype=np.float32)
    noise = rng.normal(size=(length, channels)).astype(np.float32)
    out[0] = noise[0]
    for t in range(1, length):
        out[t] = coeff * out[t - 1] + noise[t]
    return out


def generate_series(name: str, seed: int, length: int, channels: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "G1":
        return rng.normal(size=(length, channels)).astype(np.float32)
    if name == "G2":
        return _ar1(rng, length, 0.8, channels)
    if name == "G3":
        return _ar1(rng, length, -0.8, channels)
    if name == "G4":
        return _oscillator(rng, length, 64.0, channels)
    if name == "G5":
        return (_oscillator(rng, length, 64.0, channels) + _oscillator(rng, length, 128.0, channels)) / math.sqrt(2.0)
    if name == "G6":
        split = min(length, CONFIG["burn_in"] + 8192 + 2048)
        first = _oscillator(rng, length, 64.0, channels)
        second = _oscillator(rng, length, 96.0, channels)
        return np.concatenate((first[:split], second[split:]), axis=0)
    if name == "G7":
        out = np.zeros((length, channels), dtype=np.float32)
        innovations = rng.standard_t(5, size=(length, channels)).astype(np.float32) / math.sqrt(5.0 / 3.0)
        for t in range(1, length):
            sigma = 1.0 if ((t - CONFIG["burn_in"]) // 512) % 2 == 0 else 2.0
            out[t] = 0.6 * out[t - 1] + sigma * innovations[t]
        return out
    if name == "G8":
        out = np.zeros((length, channels), dtype=np.float32)
        drivers = _oscillator(rng, length, 64.0, 4)
        out[:, :4] = drivers
        noise = rng.normal(size=(length, 4)).astype(np.float32)
        for channel in range(4):
            out[:, 4 + channel] = drivers[:, channel] + noise[:, channel]
        return out
    raise ValueError(f"unknown DGP {name}")


def generate_target(name: str, seed: int, migration: str, length: int, channels: int) -> np.ndarray:
    """Generate the registered target-side migration condition."""
    if migration == "M0":
        return generate_series(name, seed, length, channels)
    rng = np.random.default_rng(seed)
    if name == "G1":
        scale = 1.5 if migration == "M1" else 1.0
        return (scale * rng.normal(size=(length, channels))).astype(np.float32)
    if name in {"G2", "G3"}:
        coeff = 0.5 if migration == "M1" else (0.8 if name == "G3" else -0.8)
        if migration == "M2":
            coeff = -0.8 if name == "G2" else 0.8
        return _ar1(rng, length, coeff, channels)
    if name in {"G4", "G5"}:
        if migration == "M2":
            return rng.normal(size=(length, channels)).astype(np.float32)
        multiplier = 1.25 if migration == "M1" else 1.0
        first = _oscillator(rng, length, 64.0 * multiplier, channels)
        if name == "G5":
            first = (first + _oscillator(rng, length, 128.0 * multiplier, channels)) / math.sqrt(2.0)
        return first
    if name == "G6":
        split = CONFIG["burn_in"] + (8192 if migration == "M1" else 8192 + 2048)
        first = _oscillator(rng, length, 64.0, channels)
        if migration == "M2":
            second = rng.normal(size=(length, channels)).astype(np.float32)
        else:
            second = _oscillator(rng, length, 96.0, channels)
        return np.concatenate((first[:split], second[split:]), axis=0)
    if name == "G7":
        out = np.zeros((length, channels), dtype=np.float32)
        innovations = rng.standard_t(5, size=(length, channels)).astype(np.float32) / math.sqrt(5.0 / 3.0)
        coeff = 0.0 if migration == "M2" else 0.6
        sigma_scale = 1.5 if migration == "M1" else 1.0
        for t in range(1, length):
            sigma = (1.0 if ((t - CONFIG["burn_in"]) // 512) % 2 == 0 else 2.0) * sigma_scale
            out[t] = coeff * out[t - 1] + sigma * innovations[t]
        return out
    if name == "G8":
        drivers = _oscillator(rng, length, 64.0, 4)
        out = np.zeros((length, channels), dtype=np.float32)
        out[:, :4] = drivers
        noise = rng.normal(size=(length, 4)).astype(np.float32)
        lag = 16 if migration == "M1" else 8
        sign = -1.0 if migration == "M2" else 1.0
        for channel in range(4):
            shifted = np.roll(drivers[:, channel], lag)
            out[:, 4 + channel] = sign * shifted + noise[:, channel]
        return out
    raise ValueError(f"unknown DGP {name}")


def make_windows(values: np.ndarray, start: int, length: int, input_length: int, horizon: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    end = start + length
    origins = np.arange(start + input_length, end - horizon + 1, dtype=np.int64)
    if fraction < 1.0:
        count = max(1, int(math.floor(len(origins) * fraction)))
        origins = origins[:count]
    x = np.stack([values[t - input_length:t] for t in origins]).astype(np.float32)
    y = np.stack([values[t:t + horizon] for t in origins]).astype(np.float32)
    return x, y


def estimate_relation(source: np.ndarray, max_lag: int) -> Relation:
    centered = source - source.mean(axis=0, keepdims=True)
    denom = np.sum(centered * centered, axis=0) + 1e-8
    scores = []
    for lag in range(1, max_lag + 1):
        scores.append(np.sum(centered[lag:] * centered[:-lag], axis=0) / denom)
    score = np.asarray(scores)
    lag = np.argmax(np.abs(score), axis=0) + 1
    # Use one lag/sign per channel. G8 naturally exposes cross-channel limits
    # because this descriptor is intentionally univariate.
    signs = np.sign(np.asarray([score[l - 1, channel] for channel, l in enumerate(lag)], dtype=np.float32))
    signs[signs == 0] = 1
    return Relation(lags=lag.astype(np.int64), signs=signs.astype(np.float32))


class LagForecaster(nn.Module):
    def __init__(self, channels: int, input_length: int, horizon: int):
        super().__init__()
        self.channels = channels
        self.input_length = input_length
        self.horizon = horizon
        self.weight = nn.Parameter(torch.zeros(channels, input_length, horizon))
        self.bias = nn.Parameter(torch.zeros(channels, horizon))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bcl,clh->bch", x, self.weight) + self.bias.unsqueeze(0)


def _arm_relation(base: Relation, arm: str, seed: int) -> Relation:
    rng = np.random.default_rng(seed)
    if arm == "A_signed":
        return base
    if arm == "N_null":
        return Relation(base.lags.copy(), np.zeros_like(base.signs))
    if arm == "U_unsigned":
        return Relation(base.lags.copy(), np.ones_like(base.signs))
    if arm == "S_shuffle":
        return Relation(base.lags.copy(), rng.permutation(base.signs))
    if arm == "R_random":
        return Relation(rng.integers(1, CONFIG["max_lag"] + 1, size=base.lags.shape), rng.choice([-1.0, 1.0], size=base.signs.shape).astype(np.float32))
    raise ValueError(arm)


def train_and_evaluate(target: np.ndarray, relation: Relation, seed: int, device: torch.device) -> dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_start = 0
    train_len = CONFIG["target_train_length"]
    val_start = train_len
    val_len = CONFIG["target_val_length"]
    test_start = train_len + val_len
    test_len = CONFIG["target_test_length"]
    x_train, y_train = make_windows(target, train_start, train_len, CONFIG["input_length"], CONFIG["horizon"], CONFIG["label_fraction"])
    x_val, y_val = make_windows(target, val_start, val_len, CONFIG["input_length"], CONFIG["horizon"], 1.0)
    x_test, y_test = make_windows(target, test_start, test_len, CONFIG["input_length"], CONFIG["horizon"], 1.0)
    xtr = torch.from_numpy(x_train).permute(0, 2, 1).to(device)
    ytr = torch.from_numpy(y_train).permute(0, 2, 1).to(device)
    xva = torch.from_numpy(x_val).permute(0, 2, 1).to(device)
    yva = torch.from_numpy(y_val).permute(0, 2, 1).to(device)
    xte = torch.from_numpy(x_test).permute(0, 2, 1).to(device)
    yte = torch.from_numpy(y_test).permute(0, 2, 1).to(device)
    model = LagForecaster(CONFIG["channels"], CONFIG["input_length"], CONFIG["horizon"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    lag_index = torch.as_tensor(CONFIG["input_length"] - relation.lags, device=device).clamp(0, CONFIG["input_length"] - 1)
    signs = torch.as_tensor(relation.signs, device=device)
    best_state = None
    best_val = float("inf")
    for _ in range(CONFIG["epochs"]):
        model.train()
        order = torch.randperm(xtr.shape[0], device=device)
        for batch in order.split(CONFIG["batch_size"]):
            pred = model(xtr[batch])
            loss = (pred - ytr[batch]).square().mean()
            if torch.any(signs != 0):
                selected = model.weight[torch.arange(CONFIG["channels"], device=device), lag_index, 0]
                loss = loss + CONFIG["relation_weight"] * ((selected - signs).square() * (signs != 0)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val = float((model(xva) - yva).square().mean().item())
        if not math.isfinite(val):
            raise FloatingPointError("non-finite validation MSE")
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pred = model(xte)
        mse = float((pred - yte).square().mean().item())
        mae = float((pred - yte).abs().mean().item())
    if not math.isfinite(mse) or not math.isfinite(mae):
        raise FloatingPointError("non-finite test metric")
    return {"best_val_mse": best_val, "mse": mse, "mae": mae, "train_windows": int(len(x_train)), "val_windows": int(len(x_val)), "test_windows": int(len(x_test))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=("S-pilot", "S-core", "S-extension", "S-full"), default="S-pilot")
    parser.add_argument("--limit-tasks", type=int, default=0, help="debug limit; zero runs the full approved pilot")
    args = parser.parse_args()
    phase = phase_config(args.phase)
    CONFIG["replicates"] = int(phase["replicates"])
    CONFIG["optimization_seeds"] = tuple(phase["optimization_seeds"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    total_length = CONFIG["burn_in"] + CONFIG["source_train_length"] + CONFIG["target_train_length"] + CONFIG["target_val_length"] + CONFIG["target_test_length"]
    rows = []
    for dgp in DGP_NAMES:
        for migration in phase["migrations"]:
            for noise_level in phase["noise_levels"]:
                for label_fraction in phase["labels"]:
                    for horizon in phase["horizons"]:
                        if args.phase == "S-extension" and noise_level == "low" and label_fraction == 0.10:
                            continue
                        CONFIG["horizon"] = int(horizon)
                        CONFIG["label_fraction"] = float(label_fraction)
                        for replicate in range(CONFIG["replicates"]):
                            data_seed = _stable_seed(args.phase, dgp, migration, noise_level, replicate)
                            source = generate_series(dgp, data_seed, total_length, CONFIG["channels"])
                            target = generate_target(dgp, data_seed + 1, migration, total_length, CONFIG["channels"])
                            noise_fraction = 0.1 if noise_level == "low" else 1.0
                            source_scale = float(np.var(source[CONFIG["burn_in"]:CONFIG["burn_in"] + CONFIG["source_train_length"]]))
                            target_scale = float(np.var(target[CONFIG["burn_in"]:CONFIG["burn_in"] + CONFIG["target_train_length"]]))
                            noise_rng = np.random.default_rng(_stable_seed(data_seed, "observation-noise"))
                            source = source + noise_rng.normal(size=source.shape).astype(np.float32) * math.sqrt(max(noise_fraction * source_scale, 1e-8))
                            target = target + noise_rng.normal(size=target.shape).astype(np.float32) * math.sqrt(max(noise_fraction * target_scale, 1e-8))
                            offset = CONFIG["burn_in"]
                            source_prefix = source[offset:offset + CONFIG["source_train_length"]]
                            target_body = target[offset:]
                            base_relation = estimate_relation(source_prefix, CONFIG["max_lag"])
                            for opt_seed in CONFIG["optimization_seeds"]:
                                for arm in ARMS:
                                    if args.limit_tasks and len(rows) >= args.limit_tasks:
                                        break
                                    key = f"{args.phase}/v1/{dgp}/{migration}/{noise_level}/replicate-{replicate}/H{horizon}/label-{label_fraction:.2f}/{arm}/{opt_seed}"
                                    started = time.time()
                                    status = "COMPLETED"
                                    metrics: dict[str, float | int] = {}
                                    try:
                                        relation = _arm_relation(base_relation, arm, _stable_seed(key, "relation"))
                                        metrics = train_and_evaluate(target_body, relation, opt_seed, device)
                                    except torch.cuda.OutOfMemoryError:
                                        status = "OOM"
                                        if device.type == "cuda":
                                            torch.cuda.empty_cache()
                                    except (FloatingPointError, RuntimeError, ValueError) as exc:
                                        status = "NUMERICAL_FAILURE" if isinstance(exc, FloatingPointError) else "DATA_ERROR"
                                        metrics["error"] = str(exc)
                                    row = {
                                        "study_id": "ACE2", "protocol_version": "ace2-simulation-v1", "phase": args.phase,
                                        "task_key": key, "dgp": dgp, "migration": migration, "noise": noise_level,
                                        "observation_noise_variance_fraction": noise_fraction,
                                        "replicate": replicate, "data_seed": data_seed, "optimization_seed": opt_seed,
                                        "arm": arm, "horizon": CONFIG["horizon"], "label_fraction": CONFIG["label_fraction"],
                                        "status": status, "attempt": 1, "elapsed_sec": round(time.time() - started, 3),
                                        "device": str(device), **metrics,
                                    }
                                    rows.append(row)
                                    with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                if args.limit_tasks and len(rows) >= args.limit_tasks:
                                    break
                            if args.limit_tasks and len(rows) >= args.limit_tasks:
                                break
                        if args.limit_tasks and len(rows) >= args.limit_tasks:
                            break
                    if args.limit_tasks and len(rows) >= args.limit_tasks:
                        break
                if args.limit_tasks and len(rows) >= args.limit_tasks:
                    break
            if args.limit_tasks and len(rows) >= args.limit_tasks:
                break
        if args.limit_tasks and len(rows) >= args.limit_tasks:
            break
    fields = sorted({key for row in rows for key in row})
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "study_id": "ACE2", "protocol_version": "ace2-simulation-v1", "phase": args.phase,
        "status": "completed", "task_count": len(rows), "expected_task_count": int(phase["expected"]),
        "config": CONFIG, "dgp_names": DGP_NAMES, "arms": ARMS, "device": str(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "observation_noise_variance_fractions": {"low": 0.1, "high": 1.0},
        "scope": "synthetic pilot only; no repository datasets, checkpoints, or historical test artifacts",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=list) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "tasks": len(rows), "completed": sum(row["status"] == "COMPLETED" for row in rows), "device": str(device)}, sort_keys=True))
    return 0 if all(row["status"] == "COMPLETED" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
