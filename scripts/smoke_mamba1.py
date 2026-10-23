#!/usr/bin/env python
"""Minimal CUDA contract test for the Mamba1 + AutoCon model."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.losses import AutoCon
from models.Mamba1AutoCon import Model


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required by the installed mamba_ssm wheel")
    args = SimpleNamespace(
        seq_len=96, label_len=48, pred_len=24, enc_in=1, c_out=1,
        d_model=16, d_ff=16, e_layers=1, mamba_d_state=16,
        mamba_d_conv=4, mamba_expand=2, mamba_headdim=16,
        AutoCon=True, AutoCon_wnorm="LastVal", AutoCon_multiscales=[24],
        embed="timeF", freq="h", wo_freq="", dropout=0.0,
    )
    model = Model(args).cuda()
    x = torch.randn(2, args.seq_len, 1, device="cuda")
    x_mark = torch.randn(2, args.seq_len, 4, device="cuda")
    dec = torch.zeros(2, args.label_len + args.pred_len, 1, device="cuda")
    dec_mark = torch.randn_like(x_mark[:, :dec.shape[1]])
    pred, representation = model(x, x_mark, dec, dec_mark)
    if pred.shape != (2, args.pred_len, 1) or representation.shape != (2, args.seq_len, args.d_ff):
        raise AssertionError((pred.shape, representation.shape))
    acf_values = np.ones(args.seq_len, dtype=np.float32)
    contrastive = AutoCon(2, args.seq_len, acf_values, temperature=1.0, base_temperature=1.0).cuda()
    local, global_ = contrastive(F.normalize(representation, dim=-1), torch.arange(2, device="cuda"))
    objective = pred.square().mean() + 0.01 * (local.mean() + global_.mean())
    objective.backward()
    if not torch.isfinite(objective) or not torch.isfinite(pred).all():
        raise AssertionError("non-finite Mamba1 smoke output")
    print("Mamba1 smoke ok:", tuple(pred.shape), tuple(representation.shape), float(objective.detach().cpu()))


if __name__ == "__main__":
    main()
