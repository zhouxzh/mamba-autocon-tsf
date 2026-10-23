# ACF-InfoNCE Reproducibility Guide

## Verified Runtime

The audited runs used the `mamba` conda environment on an RTX 5090 D:

| Component | Version |
| --- | --- |
| Python | 3.9.25 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| NVIDIA driver | 595.80 |
| mamba-ssm | 2.3.1 |
| causal-conv1d | 1.6.2.post1 |
| Triton | 3.4.0 |
| NumPy / pandas / SciPy | 1.23.5 / 2.3.3 / 1.13.1 |

Verify a target environment before any run:

```bash
conda run -n mamba python -c "import torch,mamba_ssm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), mamba_ssm.__version__)"
```

`mamba-ssm` and `causal-conv1d` include CUDA extensions. Installing them on a
different driver, CUDA toolkit, or Python version is an environment rebuild,
not a bitwise replay of the audited runtime. Record the rebuilt package hashes
and rerun the complete test suite before rerunning experiments.

## Data Contract

The audited source-selection and release JSONs record SHA-256 values for every
input dataset. Do not substitute a different preprocessing variant with the
same dataset label. The main audited release inputs are:

- `dataset/PEMS-BAY/PEMS-BAY.csv`
- `dataset/METR-LA/METR-LA.csv`
- `dataset/weather/weather.csv`
- source pools listed in each release audit's `source_selection` field

## Read-Only Audit

The following commands do not train or mutate checkpoints:

```bash
conda run -n mamba python scripts/build_mambatsv3_acf_infonce_core_bundle.py
conda run -n mamba python scripts/audit_mambatsv3_acf_infonce_horizon_release.py
conda run -n mamba python scripts/audit_mambatsv3_acf_infonce_metr_la_h192_release.py
conda run -n mamba python scripts/audit_mambatsv3_acf_infonce_weather_h192_release.py
conda run -n mamba python -m pytest -q
```

The bundle will fail closed if a referenced audit lacks `audit_valid=true` or
if a full-target-prefix validation artifact unexpectedly contains a test
metric.

## External Source Locks

The post-release external-model extensions are also bound to source-file and
requirements hashes, not only Git commit strings. Before rerunning an adapter,
download the pinned repositories, then verify them with:

```bash
conda run -n mamba python scripts/verify_external_code_locks.py \
  --lock docs/research/mambatsv3_acf_infonce_external_code_locks_20261003.json \
  --checkout patchtst=PATH_TO_PATCHTST_ROOT \
  --checkout itransformer=PATH_TO_ITRANSFORMER_ROOT \
  --checkout timesnet=PATH_TO_TSLIB_ROOT \
  --checkout timemixer=PATH_TO_TIMEMIXER_ROOT
```

The verification must return `valid=true` before an adapter run is compared to
the audited external-code extension. A changed source file or requirements
file requires a new external-code experiment root.

## Release Interpretation

Three audited held-out releases are indexed by
`artifacts/paper_final/mambatsv3_acf_infonce_core_bundle_20261003/bundle.json`.
PEMS-BAY TimesNet, TimeMixer, and RLinear comparisons are post-release
descriptive extensions. They must not be described as members of the original
frozen confirmatory test matrix.

The core bundle also retains negative boundaries: Electricity transfer failure,
Traffic 25% failure, Weather 25% instability, and the stopped forecast-affine
loss. Reproduction or manuscript tables must retain those limits.
