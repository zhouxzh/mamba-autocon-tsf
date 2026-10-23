# Direct-anchor resource smoke (2026-09-02)

This is an architecture-only resource check for the opt-in
`mamba_acf_anchor_copy` branch. It does not load a dataset, fit an ACF prior,
train a checkpoint, or measure forecasting accuracy. The benchmark uses the
same deterministic synthetic prior in both arms and keeps all inputs on the
GPU.

## Protocol

- GPU: NVIDIA GeForce RTX 5090 D
- model: `MambaACFAutoCon`, shared temporal scan plus variable-token mixer
- `seq_len=96`, `pred_len=96`, `d_model=32`, one Mamba scale, static memory
- 50 warm-up forwards and 500 timed forwards after lazy initialization
- ETTh1: `C=7`, batch 2; Traffic: `C=862`, batch 1
- evaluation telemetry is lazy by default, so the timing excludes avoidable
  GPU-to-CPU scalar logging synchronization

## Readout

| dataset | anchor | parameters | static buffer (MB) | median (ms) | p95 (ms) | peak increment (MB) |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | off | 58,947 | 0.6324 | 1.2759 | 1.3139 | 1.15 |
| ETTh1 | on | 58,948 | 0.6484 | 1.3274 | 1.9449 | 1.15 |
| Traffic | off | 169,242 | 3.0199 | 1.3049 | 1.3284 | 3.93 |
| Traffic | on | 169,243 | 4.9928 | 1.3616 | 1.3789 | 3.93 |

The median overhead is approximately `+4.04%` for ETTh1 and `+4.35%` for
Traffic. The treatment adds one trainable scalar (`tanh` ReZero scale) and
uses indexed gathers rather than a dense `[B,H,C,K]` candidate tensor. The
post-model steady-state peak increment is the same in both arms, while the
anchor's static buffers increase by `0.0160 MB` for ETTh1 and `1.9730 MB` for
Traffic; the latter must be reported rather than hidden behind the relative
peak-increment measure.

Each JSON/CSV record also stores the SHA-256 of the exact model source file
and benchmark script used for the measurement: `models/MambaACF.py` is
`41281fc7aee6c1b76519ea098c9314c71322b7a9cbd9e39ca668405b5510de16` and
`scripts/benchmark_model_resources.py` is
`e16833f8db960e187065c8b1179a2b8a517aa1ecfb5b328f132c0bc5c13fee64`. The
branch is therefore a small resource addition, but it is not faster than
the nested no-anchor model in either measurement. These are resource
feasibility results only; they are not evidence that the anchor improves
accuracy or end-to-end training time. The cache-aware raw JSON/CSV records are kept in
[`artifacts/exploratory/mamba_acf_anchor_copy_resources_current_20260902`](../../artifacts/exploratory/mamba_acf_anchor_copy_resources_current_20260902/).
