#!/usr/bin/env python3
"""Run the registered low-cost AutoCon pretraining direction pilot serially."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(name: str, command: list[str], output: Path) -> None:
    log = output / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError("pilot arm failed: {}".format(name))


def main() -> int:
    parser_mode = os.environ.get("MAMBATS_PRETRAIN_ENCODER_MODE", "pooled").strip().lower()
    if parser_mode not in {"pooled", "channelwise", "channel_context"}:
        raise ValueError("MAMBATS_PRETRAIN_ENCODER_MODE must be pooled, channelwise, or channel_context")
    suffix = {"pooled": "", "channelwise": "_channelwise", "channel_context": "_channel_context"}[parser_mode]
    run_label = os.environ.get("MAMBATS_PRETRAIN_PILOT_LABEL", "").strip()
    relation_batching = os.environ.get("MAMBATS_PRETRAIN_RELATION_BATCHING", "random").strip().lower()
    pretrain_task = os.environ.get("MAMBATS_PRETRAIN_TASK", "masked_reconstruction").strip().lower()
    if relation_batching not in {"random", "lag_pairs"}:
        raise ValueError("MAMBATS_PRETRAIN_RELATION_BATCHING must be random or lag_pairs")
    if pretrain_task not in {"masked_reconstruction", "next_patch"}:
        raise ValueError("MAMBATS_PRETRAIN_TASK must be masked_reconstruction or next_patch")
    if pretrain_task == "next_patch" and parser_mode != "channel_context":
        raise ValueError("next_patch pilot requires channel_context encoder")
    pretrain_seq_len = "72" if pretrain_task == "next_patch" else "96"
    if run_label and not run_label.replace("_", "").isalnum():
        raise ValueError("MAMBATS_PRETRAIN_PILOT_LABEL must be alphanumeric/underscore")
    label_suffix = "" if not run_label else "_" + run_label
    output = ROOT / ("artifacts/exploratory/mambats_autocon_pretraining_traffic_pilot_20261002" + suffix + label_suffix)
    output.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    common_sources = [
        "--dataset", "ETTh1:dataset/ETT-small/ETTh1.csv:csv:60",
        "--dataset", "Weather:dataset/weather/weather.csv:csv:10",
        "--dataset", "PEMS03:dataset/PEMS03/PEMS03.npz:npz:5",
        "--seq-len", pretrain_seq_len, "--patch-len", "24", "--patch-stride", "24",
        "--d-model", "16", "--d-ff", "32", "--encoder-mode", parser_mode,
        "--batch-size", "4" if parser_mode == "channelwise" else "16",
        "--epochs", "3", "--max-windows", "128" if parser_mode == "channelwise" else "512",
        "--contiguous-windows", "--pretrain-task", pretrain_task,
        "--next-patch-loss", "huber", "--seed", "3501",
    ]
    auto_ckpt = output / "autocon_pretrain.pt"
    recon_ckpt = output / "reconstruction_pretrain.pt"
    auto_args = [py, "scripts/pretrain_mamba_autocon.py", *common_sources, "--mamba-lambda", "0.001", "--output", str(auto_ckpt)]
    if relation_batching == "lag_pairs":
        auto_args.extend(["--relation-batching", "lag_pairs"])
    run("autocon_pretrain", auto_args, output)
    run("reconstruction_pretrain", [py, "scripts/pretrain_mamba_autocon.py", *common_sources, "--mamba-lambda", "0", "--output", str(recon_ckpt)], output)
    finetune_common = [
        py, "scripts/finetune_mamba_pretrain.py", "--dataset", "dataset/traffic/traffic.csv",
        "--format", "csv", "--seq-len", "96", "--pred-len", "96", "--patch-len", "24",
        "--patch-stride", "24", "--d-model", "16", "--d-ff", "32", "--few-shot-fraction", "0.1",
        "--encoder-mode", parser_mode, "--batch-size", "2" if parser_mode == "channelwise" else "16",
        "--epochs", "5", "--max-train-windows", "64" if parser_mode == "channelwise" else "256",
        "--freeze-encoder-epochs", "1", "--data-interval-minutes", "60", "--seed", "3501",
    ]
    run("random_init", [*finetune_common, "--output", str(output / "random_init.pt")], output)
    run("reconstruction_finetune", [*finetune_common, "--checkpoint", str(recon_ckpt), "--output", str(output / "reconstruction_finetune.pt")], output)
    run("autocon_finetune", [*finetune_common, "--checkpoint", str(auto_ckpt), "--output", str(output / "autocon_finetune.pt")], output)
    summaries = {}
    for name in ("random_init", "reconstruction_finetune", "autocon_finetune"):
        with (output / (name + ".json")).open(encoding="utf-8") as handle:
            summaries[name] = json.load(handle)["best_validation_mse"]
    (output / "summary.json").write_text(json.dumps({"validation_only": True, "encoder_mode": parser_mode, "pretrain_task": pretrain_task, "relation_batching": relation_batching, "target": "traffic", "few_shot_fraction": 0.1, "mse": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "mse": summaries}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
