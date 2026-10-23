#!/usr/bin/env python
"""Run a real-CUDA forward/backward contract check for graph MambaTSv3.

This is an implementation smoke test, not a training result and not a
benchmark. It verifies that the sparse graph branch, one Mamba-1 scan, and the
AutoCon representation output are all finite and connected to gradients.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import MambaTSv3
from data_provider.data_loader import Dataset_Array
from layers.losses import AutoCon
from utils.acf_guidance import fit_signed_acf_prior


DATASETS = {
    "metr_la": {
        "root": REPO_ROOT / "dataset/METR-LA",
        "csv": "METR-LA.csv",
        "graph": "adjacency.npz",
        "contract": "adjacency_contract.json",
        "start": "2012-03-01 00:00:00",
    },
    "pems_bay": {
        "root": REPO_ROOT / "dataset/PEMS-BAY",
        "csv": "PEMS-BAY.csv",
        "graph": "adjacency.npz",
        "contract": "adjacency_contract.json",
        "start": "2017-01-01 00:00:00",
    },
}


def _model(dataset: str, seed: int):
    if not torch.cuda.is_available():
        raise RuntimeError("graph MambaTSv3 CUDA smoke requires an available GPU")
    spec = DATASETS[dataset]
    contract = json.loads((spec["root"] / spec["contract"]).read_text(encoding="utf-8"))
    graph_path = spec["root"] / spec["graph"]
    channels = int(contract["node_count"])
    config = SimpleNamespace(
        task_name="long_term_forecast", seq_len=96, pred_len=12,
        enc_in=channels, c_out=channels, d_model=24, d_ff=48,
        patch_len=24, patch_stride=24, dropout=0.1, e_layers=1,
        AutoCon=True, AutoCon_wnorm="LastVal", mamba_ts_variant="v3",
        mamba_d_state=8, mamba_d_conv=2, mamba_expand=1, mamba_headdim=8,
        mamba_ts_context_rank=16, mamba_ts_graph_spatial=True,
        mamba_ts_graph_path=str(graph_path),
        mamba_ts_graph_expected_sha256=contract["adjacency_npz_sha256"],
        mamba_ts_repr_forecast_fusion=False,
        mamba_ts_repr_forecast_channelwise=False, mamba_ts_channel_adapter=False,
        mamba_ts_global_channel_mixer=False, mamba_ts_repr_linear_expert=False,
        mamba_ts_variable_permutation=False, mamba_ts_acf_memory_selector=False,
        mamba_ts_acf_memory_channel_selector=False,
        mamba_ts_forecast_acf_consistency=False,
        mamba_ts_forecast_acf_weight=0.0, mamba_ts_forecast_acf_top_k=4,
        mamba_ts_forecast_acf_min_lag=2, mamba_ts_forecast_acf_min_abs=0.1,
        mamba_ts_residual_init=0.0, moving_avg=25,
    )
    torch.manual_seed(seed)
    return MambaTSv3.Model(config).cuda().train(), contract


def run(dataset: str, batch_size: int, seed: int) -> dict:
    model, contract = _model(dataset, seed)
    channels = int(contract["node_count"])
    inputs = torch.randn(batch_size, 96, channels, device="cuda")
    torch.cuda.synchronize()
    forecast, representation = model(inputs, return_repr=True)
    loss = forecast.square().mean() + 0.01 * representation.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    scale_grad = model.ts_graph_spatial_scale.grad
    if scale_grad is None or not torch.isfinite(scale_grad).all():
        raise RuntimeError("graph spatial scale did not receive a finite gradient")
    return {
        "dataset": dataset,
        "device": str(forecast.device),
        "device_name": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "seq_len": 96,
        "pred_len": 12,
        "channels": channels,
        "graph_edges": int(model.ts_graph_target.numel()),
        "forecast_shape": list(forecast.shape),
        "representation_shape": list(representation.shape),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "graph_scale_initial": float(model.ts_graph_spatial_scale.detach().cpu()),
        "graph_scale_gradient_abs": float(scale_grad.detach().abs().cpu()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "graph_sha256": contract["adjacency_npz_sha256"],
        "smoke_contract": "real_cuda_forward_backward; no dataset windows or test metrics",
    }


def run_real_data_batch(dataset: str, batch_size: int, seed: int) -> dict:
    """Check one actual training batch without producing a training result."""

    spec = DATASETS[dataset]
    model, contract = _model(dataset, seed)
    channels = int(contract["node_count"])
    train_data = Dataset_Array(
        root_path=str(spec["root"]), data_path=spec["csv"], flag="train",
        size=[96, 48, 12], features="M", freq="t", timeenc=1,
        train_ratio=0.6, data_format="csv_numeric", target_index=channels - 1,
        data_interval_minutes=5, data_start_timestamp=spec["start"],
    )
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    origins, inputs, targets, _, _ = next(iter(loader))
    inputs = inputs.float().cuda(non_blocking=True)
    targets = targets[:, -12:, :].float().cuda(non_blocking=True)

    # This is the same train-only source class used by AutoCon.  Its ACF is
    # reduced across sensor channels, as the historical AutoCon objective does.
    prior = fit_signed_acf_prior(
        np.asarray(train_data.data_x, dtype=np.float64), max_lag=95,
        top_k=8, min_lag=2, min_abs_acf=0.1, confidence_z=1.96,
        min_separation=2, detrend="mean",
    )
    autocon = AutoCon(
        batch_size=batch_size, seq_len=96, acf_values=prior.acf.mean(axis=0),
        temperature=1.0, base_temperature=1.0,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    forecast, representation = model(inputs, return_repr=True)
    mse = F.mse_loss(forecast, targets)
    local, global_ = autocon(F.normalize(representation, dim=-1), origins.long().cuda())
    contrastive = (local.mean(dim=1) + global_).mean() / 2.0
    contrastive_gradient = torch.autograd.grad(
        contrastive, model.ts_graph_spatial_scale, retain_graph=True, allow_unused=True,
    )[0]
    loss = mse + 0.001 * contrastive
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    graph_gradient = model.ts_graph_spatial_scale.grad
    if graph_gradient is None or contrastive_gradient is None:
        raise RuntimeError("graph scale is disconnected from forecast or AutoCon loss")
    if not torch.isfinite(loss) or not torch.isfinite(contrastive_gradient).all():
        raise RuntimeError("real-data graph smoke has a non-finite loss or gradient")
    return {
        "dataset": dataset,
        "device": str(forecast.device),
        "device_name": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "train_split_windows": len(train_data),
        "origin_range": [int(origins.min()), int(origins.max())],
        "forecast_shape": list(forecast.shape),
        "representation_shape": list(representation.shape),
        "mse_loss": float(mse.detach().cpu()),
        "autocon_contrastive_loss": float(contrastive.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
        "graph_scale_initial": 0.0,
        "graph_scale_autocon_gradient_abs": float(contrastive_gradient.detach().abs().cpu()),
        "graph_scale_total_gradient_abs": float(graph_gradient.detach().abs().cpu()),
        "graph_scale_after_step": float(model.ts_graph_spatial_scale.detach().cpu()),
        "acf_train_only_shape": list(prior.acf.shape),
        "graph_sha256": contract["adjacency_npz_sha256"],
        "smoke_contract": "single real train batch plus AutoCon backward; no validation/test metrics or checkpoint",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--real-data-batch", action="store_true",
                        help="run one train-split batch with AutoCon; no validation/test metrics")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    runner = run_real_data_batch if args.real_data_batch else run
    results = [runner(name, args.batch_size, args.seed) for name in (args.dataset or sorted(DATASETS))]
    payload = {"schema_version": 1, "results": results}
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
