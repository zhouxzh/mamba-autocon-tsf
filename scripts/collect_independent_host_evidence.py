#!/usr/bin/env python3
"""Collect read-only CUDA/Mamba evidence from a proposed independent host."""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
from pathlib import Path


REMOTE_PROBE = r'''
import json
import platform
import torch
import mamba_ssm
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print(json.dumps({
    "hostname": platform.node(),
    "cuda_available": True,
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "mamba_ssm": mamba_ssm.__version__,
}, sort_keys=True))
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True,
                        help="remote SSH destination; commands are read-only")
    parser.add_argument("--machine-id", required=True,
                        help="operator-provided identifier, not a proof of independence")
    parser.add_argument("--remote-python", default="python",
                        help="remote Python interpreter used only for the metadata probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    remote_command = shlex.join([args.remote_python, "-c", REMOTE_PROBE])
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "--", args.ssh_host, remote_command],
        check=True, text=True, capture_output=True,
    )
    remote = json.loads(completed.stdout.strip().splitlines()[-1])
    if not remote.get("cuda_available") or not remote.get("gpu"):
        raise RuntimeError("remote CUDA/Mamba probe returned incomplete evidence")
    local_hostname = platform.node()
    result = {
        "machine_id": args.machine_id,
        "ssh_host": args.ssh_host,
        "local_hostname": local_hostname,
        "remote": remote,
        "distinct_from_local_hostname": remote["hostname"] != local_hostname,
        "read_only_collection": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
