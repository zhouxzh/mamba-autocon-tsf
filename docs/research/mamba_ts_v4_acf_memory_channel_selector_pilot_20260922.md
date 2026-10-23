# MambaTS v4 Channel-Conditioned ACF Memory Pilot (2026-09-22)

## Protocol

This is an eight-job, validation-only mechanism pilot on the four long-horizon
coordinates where the global AutoCon representation selector was weakest. The
control is the frozen global selector; the treatment adds a shared linear probe
over each variable's encoded Mamba state. Both phases use the same train-only
ACF prior, AutoCon objective, seed (`2021`), budget, and data contract.

Plan: [`mamba_ts_v4_acf_memory_channel_selector_pilot_20260922.json`](./mamba_ts_v4_acf_memory_channel_selector_pilot_20260922.json)

Both phase validators are `valid=true` with four unique jobs, matching source
hashes, and no test artifacts. The readout command is recorded in
[`analyze_mamba_ts_v4_acf_memory_channel_selector_pilot.py`](../../scripts/analyze_mamba_ts_v4_acf_memory_channel_selector_pilot.py).

## Results

| Dataset | Horizon | Global selector MSE | Channel-conditioned MSE | Relative change |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 336 | 1.323683 | 1.332409 | `+0.659%` |
| ETTh1 | 720 | 1.611436 | 1.607068 | `-0.271%` |
| Traffic | 336 | 0.373741 | 0.372587 | `-0.309%` |
| Traffic | 720 | 0.397292 | 0.396265 | `-0.258%` |

The mean change over four one-seed pairs is `-0.045%`. The treatment repairs
the prior Traffic regressions and ETTh1/H720, but worsens ETTh1/H336. This is
not enough evidence for a general claim or model selection: it is one seed,
selected coordinates, and no external baseline or test evaluation.

## Decision

Keep the implementation as a gated mechanism candidate, but do not promote it
to the paper model yet. A full confirmation would need a pre-registered
multi-dataset, multi-seed matrix including H96/H336/H720 and exact strong
baselines. Until then, the defensible result is that channel-conditioned
confidence is a plausible explanation for the global selector's Traffic
failure, not a demonstrated forecasting improvement.
