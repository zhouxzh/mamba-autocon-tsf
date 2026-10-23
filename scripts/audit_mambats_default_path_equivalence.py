#!/usr/bin/env python3
"""Compare the frozen and current default MambaTSv3 CUDA execution paths."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROBE = r'''
import hashlib
import json
import os
import sys
import base64
from types import SimpleNamespace

import torch

sys.path.insert(0, os.getcwd())
from models.MambaTSv3 import Model


def digest_tensors(values):
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


config = SimpleNamespace(
    task_name="long_term_forecast", seq_len=96, pred_len=96,
    enc_in=321, c_out=321, d_model=32, d_ff=64,
    patch_len=24, patch_stride=24, dropout=0.0, e_layers=1,
    AutoCon=True, AutoCon_wnorm="MeanStd", AutoCon_lambda=0.001,
    mamba_ts_variant="v3", mamba_ts_mixer_type="mamba1",
    mamba_ts_context_rank=16, mamba_d_state=16, mamba_d_conv=4,
    mamba_expand=2, mamba_headdim=16, mamba_ts_repr_forecast_fusion=False,
    mamba_scales=[1], mamba_causal_embedding=True,
)
torch.manual_seed(20261001)
torch.cuda.manual_seed_all(20261001)
device = torch.device("cuda:0")
model = Model(config).to(device).eval()
state_digest = digest_tensors(model.state_dict())
x = torch.randn(2, 96, 321, device=device)
with torch.inference_mode():
    prediction, representation = model(x, return_repr=True)
payload = {
    "state_sha256": state_digest,
    "prediction_sha256": digest_tensors({"prediction": prediction}),
    "representation_sha256": digest_tensors({"representation": representation}),
    "prediction_shape": list(prediction.shape),
    "representation_shape": list(representation.shape),
    "prediction_sum": float(prediction.float().sum().cpu()),
    "representation_sum": float(representation.float().sum().cpu()),
    "representation_bytes_b64": base64.b64encode(
        representation.detach().contiguous().cpu().numpy().tobytes()
    ).decode("ascii"),
    # Older immutable snapshots predate the explicit field. Their model
    # construction always used Mamba-1, which is the compatibility default.
    "mixer": getattr(model, "mamba_ts_mixer_type", "mamba1"),
    "variant": model.variant,
}
print(json.dumps(payload, sort_keys=True))
'''


def _probe(source_root: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "probe failed for {}: {}".format(source_root, result.stderr.strip())
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("probe produced invalid JSON for {}".format(source_root)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot_root = args.snapshot_root.resolve()
    if not (snapshot_root / "models" / "MambaTS.py").is_file():
        raise ValueError("snapshot root does not contain models/MambaTS.py")
    frozen = _probe(snapshot_root)
    current = _probe(ROOT)
    import base64
    import numpy as np

    frozen_representation = np.frombuffer(
        base64.b64decode(frozen.pop("representation_bytes_b64")), dtype=np.float32
    )
    current_representation = np.frombuffer(
        base64.b64decode(current.pop("representation_bytes_b64")), dtype=np.float32
    )
    if frozen_representation.shape != current_representation.shape:
        raise ValueError("representation buffers have different element counts")
    representation_max_abs = float(
        np.max(np.abs(frozen_representation - current_representation))
    )
    compared = ("state_sha256", "prediction_sha256")
    payload = {
        "snapshot_root": str(snapshot_root),
        "current_root": str(ROOT),
        "frozen": frozen,
        "current": current,
        "equal": {key: frozen[key] == current[key] for key in compared},
        "representation_max_abs": representation_max_abs,
        "representation_tolerance": 2e-6,
    }
    payload["passed"] = all(payload["equal"].values()) and (
        frozen["prediction_shape"] == current["prediction_shape"]
        and frozen["representation_shape"] == current["representation_shape"]
        and frozen["mixer"] == current["mixer"] == "mamba1"
        and frozen["variant"] == current["variant"] == "v3"
        and representation_max_abs <= payload["representation_tolerance"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "output": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
