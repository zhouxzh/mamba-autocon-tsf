# Residual-Target AutoCon Validation

## Candidate

The candidate preserves one MambaTSv3 Mamba-1 block per shared channel patch
sequence. It pretrains a channel-count-agnostic core and an exact-H96
prediction head to predict the residual to the final observed normalized
value. Direct signed affine ACF alignment is calibrated to about 10% of the
source next-segment loss and uses train-only source ACF lag pairs. At target
adaptation, a train-prefix inner calibration tail chooses DLinear or NLinear
as the primary forecast; the pretrained Mamba head is a zero-gated residual
correction. No ACF loss or representation is computed at inference.

## Leakage-Safe Selection

Target source ranking now uses only the declared few-shot target prefix, not
the full target training partition. Traffic source pool is
Electricity/METR-LA/PEMS-BAY; Weather source pool is
METR-LA/PEMS-BAY/Electricity. The target-inner calibration tail consistently
chooses DLinear with uniform windows for Traffic and NLinear with uniform
windows for Weather. Main validation and target tests are not accessed during
either selection.

## Three-Seed Validation

| Target | Selected primary | Control MSE | Residual-AutoCon MSE | Change | Wins | Paired upper ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Traffic H96, 10% | DLinear | .806555 | .794837 | -1.453% | 3/3 | .9866 |
| Weather H96, 10% | NLinear | 4.288621 | 4.207431 | -1.893% | 3/3 | .9832 |

The three paired seeds are `5101/5103/5105` for Traffic and
`5102/5104/5106` for Weather. Every job used the same 10% target prefix,
five target epochs, 64 train windows, one frozen-Mamba warm-up epoch, and a
closed chronological validation partition. No target test metric or
prediction artifact was produced.

## Stress And Efficiency

On a fresh selected-primary target pair, the hybrid improved all six closed
validation stress conditions relative to its selected linear control:
in-distribution, later validation half, 10%/30% missing channels, and
5%/10% signal-relative noise. The reductions range from .68% to 2.00%.
These are one-seed stress observations and need replication.

Prediction-only CUDA measurements explicitly use `return_repr=false`, return
no representation, construct no relation loss, and use one Mamba scan. Fixed
RTX 5090 D `B=8,L=96,H=96` median forward latency is .6965 ms for Traffic
and .3107 ms for Weather. This is a fixed-shape forward result, not end-to-end
throughput.

## Evidence Boundary

This is a strong validation candidate, not a submission-ready or held-out
result. Before any held-out release, a registered plan must freeze the source
selection artifacts, primary-selection procedure, source/target data hashes,
three target seeds, external baseline set, stress seeds, and one-sided
non-inferiority gate. The first held-out metric must remain unseen until the
validation bundle passes all of those prelaunch checks.

Artifacts:

- `artifacts/exploratory/mambatsv3_fewshot_source_selection_20261003`
- `artifacts/exploratory/mambatsv3_clean_fewshot_residual_20261003`
- `artifacts/exploratory/mambatsv3_clean_fewshot_residual_replication_20261003`
- `artifacts/exploratory/mambatsv3_clean_fewshot_residual_replication3_20261003`
- `artifacts/exploratory/mambatsv3_residual_hybrid_robustness_20261003`
- `artifacts/exploratory/mambatsv3_residual_hybrid_inference_20261003`
