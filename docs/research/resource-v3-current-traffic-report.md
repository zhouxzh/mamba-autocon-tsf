# Current Resource Re-test: Traffic H96

## Scope

This is an analysis-only resource re-test of the current source tree. It does
not train a model, load a dataset window, or compare forecast accuracy. The
measured path is a device-resident synthetic forward pass.

| Setting | Value |
| --- | --- |
| Dataset shape | Traffic, multivariate (`C=862`) |
| Encoder / horizon | `L=96`, `H=96` |
| Benchmark batch | `B=8` |
| Shared width/depth | `d_model=32`, `e_layers=1`, `d_ff=64` |
| Mamba core | `d_state=16`, `d_conv=4`, `expand=2`, `headdim=16` |
| Device | NVIDIA GeForce RTX 5090 D (`cuda:0`) |
| Software | Python 3.9.25, PyTorch 2.8.0+cu128, CUDA 12.8 |
| Timing | 50 warm-up forwards, 200 timed forwards, CUDA events + synchronization |
| Input | fixed synthetic device-resident tensor, seed `2718`; transfers excluded |

PatchTST, Mamba2, MambaACF plain, and MambaACF memory all use the same input
shape and forecasting dimensions. This is not parameter/FLOP equivalence:
PatchTST uses attention, Mamba2 and MambaACF have different heads and residual
paths, and their parameter counts differ. The MambaACF memory arm attaches a
deterministic synthetic ACF prior so that its memory path is actually active;
it is not a train-split ACF measurement.

## Results

Values are from
[`model_resources.json`](../../artifacts/resource_v3_current/traffic_c862_h96_b8_w50_t200/model_resources.json).
Latency is the mean per batch forward; p95 is over the 200 timed forwards;
throughput is `B / latency` and is synthetic sequences per second.

| Model | Total / trainable params | Mean latency (ms) | Median (ms) | P95 (ms) | Synthetic seq/s | Peak allocated (MB) | Peak increment (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PatchTST | 46,080 / 46,080 | 0.886401 | 0.885824 | 0.891467 | 9,025.3 | 110.446 | 95.188 |
| Mamba2 | 171,722 / 167,562 | 0.703623 | 0.702608 | 0.717635 | 11,369.7 | 39.351 | 23.614 |
| MambaACF plain | 150,700 / 145,387 | 0.685274 | 0.684640 | 0.699413 | 11,674.2 | 31.445 | 15.760 |
| MambaACF memory | 150,700 / 148,555 | 0.809002 | 0.808272 | 0.817942 | 9,888.7 | 42.827 | 25.861 |

Relative mean latency to PatchTST is `-20.62%` for Mamba2, `-22.69%` for
MambaACF plain, and `-8.73%` for MambaACF memory. Within the MambaACF pair,
memory adds `18.06%` latency and `36.20%` peak allocated memory relative to
plain. These are point estimates from one benchmark invocation, not
statistical replicates.

## Reproduction

The four explicit architecture configs are in
[`artifacts/resource_v3_current/configs/`](../../artifacts/resource_v3_current/configs/).
The preflight (construction and output-shape check without GPU timing) was:

```bash
PYTHONPATH=. conda run --no-capture-output -n mamba \
  python scripts/benchmark_model_resources.py \
  --source-root . \
  --config artifacts/resource_v3_current/configs/patchtst_traffic_c862_h96_b8.json \
  --config artifacts/resource_v3_current/configs/mamba2_traffic_c862_h96_b8.json \
  --config artifacts/resource_v3_current/configs/mambaacf_plain_traffic_c862_h96_b8.json \
  --config artifacts/resource_v3_current/configs/mambaacf_memory_traffic_c862_h96_b8.json \
  --output-dir artifacts/resource_v3_current/preflight
```

The timed run was:

```bash
PYTHONPATH=. conda run --no-capture-output -n mamba \
  python scripts/benchmark_model_resources.py \
  --source-root . \
  --config artifacts/resource_v3_current/configs/patchtst_traffic_c862_h96_b8.json \
  --config artifacts/resource_v3_current/configs/mamba2_traffic_c862_h96_b8.json \
  --config artifacts/resource_v3_current/configs/mambaacf_plain_traffic_c862_h96_b8.json \
  --config artifacts/resource_v3_current/configs/mambaacf_memory_traffic_c862_h96_b8.json \
  --run-benchmark --device cuda:0 --benchmark-batch-size 8 \
  --warmup-iters 50 --timed-iters 200 --input-seed 2718 \
  --output-dir artifacts/resource_v3_current/traffic_c862_h96_b8_w50_t200
```

## Interpretation Boundary

The result supports only the statement that, under this GPU and synthetic
device-resident shape, the current Mamba forward implementations enqueue and
execute faster than the measured PatchTST configuration. It does **not** show
that Mamba trains faster, reaches the same accuracy, uses less wall-clock time
per epoch, or has better end-to-end throughput. Data loading, host-to-device
transfers, optimizer work, backward passes, checkpointing, and validation are
excluded. A publication-grade efficiency claim still requires a matched
training protocol and separately measured end-to-end wall-clock and memory
statistics.
