#!/usr/bin/env python3
"""Verify a CUDA/Mamba clean-install environment before a frozen replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


WHEELS = {
    "mamba_ssm": (Path("artifacts/reproducibility_wheels_20261003/mamba_ssm-2.3.1-cp39-cp39-linux_x86_64.whl"), "e5efa9866d44ab54572d105e5a6b2c6cd0ec8aea623373c3d3c1310638079d41"),
    "causal_conv1d": (Path("artifacts/reproducibility_wheels_20261003/causal_conv1d-1.6.2.post1-cp39-cp39-linux_x86_64.whl"), "435cfb57c204eec017e051fdf51b3d3f22edfcb624f394f18b3f1c46b67f4e28"),
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import causal_conv1d
    import mamba_ssm
    from mamba_ssm.modules.mamba_simple import Mamba
    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("requires torch 2.8.0+cu128 with CUDA available")
    if mamba_ssm.__version__ != "2.3.1":
        raise RuntimeError("requires mamba_ssm 2.3.1")
    for name, (wheel, expected) in WHEELS.items():
        if sha256(wheel) != expected:
            raise RuntimeError("wheel hash mismatch: {}".format(name))
    model = Mamba(d_model=24, d_state=8, d_conv=2, expand=1).cuda().eval()
    x = torch.randn(2, 4, 24, device="cuda")
    with torch.inference_mode():
        y = model(x)
    torch.cuda.synchronize()
    result = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "mamba_ssm": mamba_ssm.__version__, "causal_conv1d": causal_conv1d.__file__,
        "smoke_shape": list(y.shape), "wheels": {name: str(path) for name, (path, _hash) in WHEELS.items()},
        "clean_environment_ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
