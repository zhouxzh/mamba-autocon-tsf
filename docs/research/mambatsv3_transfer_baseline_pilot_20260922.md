# MambaTS v3 Transfer Baseline Pilot (2026-09-22)

## Protocol

This is a newly authorized, validation-only 54-job transfer pilot. It uses one
frozen source snapshot and the same M-feature, MeanStd, eight-epoch, three-seed
contract for DLinear, PatchTST, TimesNet, iTransformer, Mamba-2, and
MambaTSv3AutoCon. The candidate retains the repository's AutoCon
representation objective with `lambda=0.001`; tests are disabled for every
job.

The phase validator reports `valid=true`, 54 expected and unique jobs, and 54
`ok` manifest rows. All result configurations have `skip_test=true`; no test
metric arrays exist. The plan and launch binding are
[`mambatsv3_transfer_baseline_pilot_20260922.json`](./mambatsv3_transfer_baseline_pilot_20260922.json)
and
[`mambatsv3_transfer_baseline_pilot_launch_sidecar_20260922.json`](./mambatsv3_transfer_baseline_pilot_launch_sidecar_20260922.json).

## Validation Results

| Dataset | Horizon | Best mean validation MSE | MambaTS v3 AutoCon | vs. PatchTST | vs. TimesNet | vs. Mamba-2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Weather | 96 | MambaTS v3 AutoCon `0.445116` | `0.445116` | `-10.010%`, 3/3 | `-3.174%`, 3/3 | `-6.615%`, 3/3 |
| Exchange-Rate | 96 | PatchTST `0.261374` | `0.276657` | `+5.873%`, 0/3 | `-20.429%`, 3/3 | `-18.587%`, 3/3 |
| ILI | 24 | DLinear `0.236295` | `0.252986` | `+6.789%`, 0/3 | `+0.895%`, 1/3 | `-11.295%`, 3/3 |

Negative changes favor MambaTS v3 AutoCon. The result is deliberately mixed:
the candidate is robustly best on Weather, but loses to PatchTST on
Exchange-Rate and to DLinear/PatchTST on ILI. The pilot therefore adds useful
domain coverage and a credible Weather candidate signal; it does not establish
general accuracy non-inferiority, nor does it isolate the incremental benefit
of AutoCon because this pilot has no `lambda=0` MambaTS v3 null arm.

## Prediction-Only Resources

On Weather/C21/H96/B32, FP32 device-resident input, 50 warm-up and 200 timed
forwards, the following matched configuration records were measured:

| Model | Trainable parameters | Median latency | p95 latency | Peak increment |
| --- | ---: | ---: | ---: | ---: |
| MambaTS v3 AutoCon | 62,053 | `0.267 ms` | `0.275 ms` | `15.04 MB` |
| PatchTST | 46,080 | `0.283 ms` | `0.292 ms` | `9.11 MB` |
| Mamba-2 | 32,193 | `0.694 ms` | `0.704 ms` | `13.55 MB` |
| TimesNet | 1,193,589 | `2.765 ms` | `2.777 ms` | `44.00 MB` |

The candidate is about 5.8% faster than PatchTST for this one input shape, but
has more parameters and a larger peak allocation. These are isolated forward
measurements from validated result configurations, not data-loading,
backpropagation, optimizer, or end-to-end training speed.

## Decision

Expand only the Weather branch to H336 and H720 with the same three seeds, an
exact `lambda=0` MambaTS v3 null, and the retained strong baselines. Do not
expand Exchange-Rate or ILI for this candidate without a materially different,
pre-registered architecture hypothesis. Any paper-level table still requires a
full-budget, multi-seed, test-isolated confirmation and a train-step resource
benchmark.
