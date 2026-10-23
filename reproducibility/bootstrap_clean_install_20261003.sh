#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/new/conda/environment" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_PATH=$1
CONDA_BIN=${CONDA_EXE:-conda}

cd "$REPO_ROOT"
sha256sum -c reproducibility/clean_install_replication_checksums_20261003.txt
"$CONDA_BIN" create -y -p "$ENV_PATH" python=3.9 pip
"$CONDA_BIN" install -y -p "$ENV_PATH" -c nvidia cuda-nvcc=12.8
"$ENV_PATH/bin/python" -m pip install \
  torch==2.8.0 numpy==1.23.5 pandas==2.3.3 scikit-learn==1.6.1 \
  einops==0.8.1 packaging==25.0 ninja==1.13.0 transformers==4.57.6
"$ENV_PATH/bin/python" -m pip install --no-deps \
  artifacts/reproducibility_wheels_20261003/causal_conv1d-1.6.2.post1-cp39-cp39-linux_x86_64.whl \
  artifacts/reproducibility_wheels_20261003/mamba_ssm-2.3.1-cp39-cp39-linux_x86_64.whl
"$ENV_PATH/bin/python" scripts/verify_clean_install_environment.py \
  --output artifacts/reproducibility/independent_machine_environment_check.json
