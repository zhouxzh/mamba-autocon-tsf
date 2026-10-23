#!/usr/bin/env python3
"""CUDA forward/backward smoke for the latent temporal AutoCon Mamba path."""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from layers.losses import ACFRelationLoss
from models import LatentTemporalMamba


def _gradient_norm(module: torch.nn.Module) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).norm().item()) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--channels", type=int, default=321)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=3361)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.pred_len, enc_in=args.channels,
        c_out=args.channels, d_model=32, d_ff=64, dropout=0.1,
        moving_avg=25, mamba_d_state=16, mamba_d_conv=4,
        AutoCon=True, AutoCon_wnorm="MeanStd", task_name="long_term_forecast",
    )
    model = LatentTemporalMamba.Model(config).cuda().train()
    inputs = torch.randn(args.batch_size, args.seq_len, args.channels, device="cuda")
    target = torch.randn(args.batch_size, args.pred_len, args.channels, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    prediction, representation = model(inputs, return_repr=True)
    if tuple(prediction.shape) != (args.batch_size, args.pred_len, args.channels):
        raise RuntimeError("unexpected prediction shape: {}".format(tuple(prediction.shape)))
    expected_repr_shape = (args.batch_size, args.seq_len, 64)
    if tuple(representation.shape) != expected_repr_shape:
        raise RuntimeError("unexpected representation shape: {}".format(tuple(representation.shape)))
    acf = torch.full((720,), 0.2, device="cuda")
    relation = ACFRelationLoss(
        acf, temperature=0.2, relation_temperature=0.25,
        relation_scope="both", selected_lags=[24, 168, 336], min_lag=0,
    ).cuda()
    features = F.normalize(representation, dim=-1)
    labels = torch.arange(args.batch_size, device="cuda", dtype=torch.long) * 24
    local_loss, global_loss = relation(features, labels)
    loss = F.mse_loss(prediction, target) + 0.001 * (local_loss.mean() + global_loss.mean()) / 2.0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    payload = {
        "prediction_shape": list(prediction.shape),
        "representation_shape": list(representation.shape),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "local_loss_finite": bool(torch.isfinite(local_loss).all().item()),
        "global_loss_finite": bool(torch.isfinite(global_loss).all().item()),
        "mamba_grad_norm": _gradient_norm(model.mamba),
        "compress_grad_norm": _gradient_norm(model.variable_compress),
        "repr_projection_grad_norm": _gradient_norm(model.repr_projection),
        "forecast_scale_grad": float(model.forecast_scale.grad.detach().item()),
    }
    if not all(payload[key] for key in ("loss_finite", "local_loss_finite", "global_loss_finite")):
        raise RuntimeError("non-finite smoke loss")
    if any(payload[key] <= 0.0 for key in ("mamba_grad_norm", "compress_grad_norm", "repr_projection_grad_norm")):
        raise RuntimeError("representation gradient path is zero")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
