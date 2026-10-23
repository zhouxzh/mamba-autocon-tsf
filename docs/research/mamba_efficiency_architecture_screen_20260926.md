# Mamba efficiency architecture screen (2026-09-26)

## Question and protocol

The earlier PEMS H=12 result established validation accuracy for `MambaTSv3AutoCon`, but its PEMS07 prediction path (`B=32`, `L=96`, `C=883`, `H=12`) required about `2.189 ms` and `647.8 MB`. This screen asked whether a Mamba architecture can retain AutoCon representation learning while materially reducing this cost.

All accuracy runs below use PEMS07 H=12, `M` features, MeanStd normalization, CUDA, 20 epochs, and `skip_test=true`. They are validation-only exploratory results; no test results were created or inspected.

## Token-sparse MambaTS v3

The dominant cost of the original patch-token MambaTS v3 is its shared scan over `C * P` tokens. For PEMS07, the original `patch_len=16`, `patch_stride=8` produces `883 * 11 = 9,713` tokens. A resource-only probe selected fewer patch tokens before any fresh architecture-screen seed was observed.

| Configuration | Parameters | Mean latency (ms) | Peak allocated (MB) |
|---|---:|---:|---:|
| Original v3 (`stride=8`, `d_model=32`) | 66,491 | 2.181 | 647.9 |
| Sparse v3 (`stride=24`, `d_model=16`) | 27,459 | 0.475 | 105.2 |
| Sparse v3 (`stride=32`, `d_model=16`) | 27,251 | 0.397 | 88.4 |
| TimeMixer reference | 158,643 | 0.601 | 97.0 |
| Mamba2 reference | 158,839 | 0.753 | 73.6 |
| iTransformer reference | 12,172 | 0.915 | 57.3 |

The sparse configurations meet the forward-latency objective, but accuracy did not survive the reduction.

### Stride-32 PEMS07 probe

Three fresh paired seeds gave active/null validation MSE means of `0.067500/0.067355`; active was about `0.22%` worse than null. Both are about `3.6%` worse than the frozen PEMS07 TimeMixer reference (`0.065095`). This route was rejected before four-dataset confirmation.

### Stride-24 PEMS07 probe

Three fresh paired seeds gave active/null means of `0.065725/0.065775`. Active improved null by about `0.075%` and was within `0.97%` of the frozen TimeMixer reference. This justified the separately frozen four-PEMS confirmation.

### Stride-24 four-PEMS confirmation

Fresh seeds `2051/2052/2053`, active lambda `0.001`, and the exact lambda-zero graph completed `24/24` jobs; both phase validators reported complete manifests with zero errors.

| Dataset | Null MSE | Active MSE | Active vs null | Best frozen external baseline | Active vs baseline |
|---|---:|---:|---:|---:|---:|
| PEMS03 | 0.066358 | 0.066334 | -0.036% | TimesNet 0.063845 | +3.90% |
| PEMS04 | 0.086084 | 0.086068 | -0.018% | TimeMixer 0.080217 | +7.29% |
| PEMS07 | 0.065689 | 0.065736 | +0.072% | TimeMixer 0.065095 | +0.98% |
| PEMS08 | 0.095683 | 0.095609 | -0.077% | TimeMixer 0.092892 | +2.93% |

The paired active/null average across all 12 coordinates is a small `-0.0148%` change, but the candidate violates the predeclared 1% baseline non-inferiority gate on PEMS03, PEMS04, and PEMS08. It is therefore a reproducible speed-positive, cross-dataset accuracy-negative result and must not be test-released or promoted as the paper candidate.

Plan and artifacts:

- [`four-PEMS plan`](pems_mambatsv3_stride24_autocon_h12_confirmation_20260926.json)
- [`active phase validation`](../../artifacts/exploratory/pems_mambatsv3_stride24_autocon_h12_confirmation_20260926/active/phase_validation.json)
- [`resource result`](../../artifacts/exploratory/pems07_mambatsv3_stride24_resource_final_20260926/model_resources.csv)

## Variable-token NLinear Mamba

The alternative architecture embeds each variable's full history as one token and makes one Mamba-1 scan across variables. Its fusion-enabled prediction path uses the same temporal representation for AutoCon and the forecast residual. The pre-training PEMS07 resource measurement is favorable: `6,379` parameters, `0.339 ms` mean latency, and `65.1 MB` peak allocation.

The first two launches exposed contract failures before usable evidence:

1. The model requires `mamba_acf_relation_loss=true`; the original plan omitted it and all jobs failed argument validation before training.
2. With the relation loss enabled but fixed lags, phase validation rejected every completed null coordinate because relation-lag provenance requires train-derived peaks. Those results are invalid and isolated under the old output directory.

The corrected train-peak launch logged train-peak telemetry for completed null jobs, but it did not complete phase validation and triggered the plan stopping condition before its active arm: best validation MSE was approximately `0.075`, over 15% worse than the TimeMixer PEMS07 reference. Its ACF relation telemetry additionally showed `Contrastive Anchors=0%` and `Degenerate Positive=100%`, so the requested relation contrast was not informative in this geometry. The process was terminated before active results could be treated as evidence.

Therefore the variable-token route is currently a forward-efficiency candidate only, not an accuracy candidate. A future attempt requires a relation representation/lag design that yields non-degenerate positives before another full training run.

### Corrected width-64 relation rerun

The required selected-lag mask was subsequently enabled for a fresh `d_model=64`, `d_ff=128` PEMS07/H12 active/null three-seed screen. The objective then had usable contrastive support (final global contrastive-active anchors `68.8%--69.7%`; degenerate positives `17.9%--18.4%`) and both phase validators were `valid=true`. Thus the old objective-degeneracy diagnosis is resolved for this rerun.

It did not resolve the forecast gap: active/null mean validation MSE was `.067201/.067309`, compared with the frozen same-snapshot TimeMixer mean `.065095` (`+3.24%/+3.40%`). The active relation objective improved its exact null by only `0.16%`. The d64 prediction path remains fast (`.336 ms`, `56,155` parameters, `96.2 MB` peak allocated at PEMS07 `B32/L96/C883/H12`), but this is a speed-positive, accuracy-negative result and is stopped before a multi-dataset or test expansion. The plan is [`pems07_nlinear_variable_mamba_d64_relation_paired_screen_20260926.json`](pems07_nlinear_variable_mamba_d64_relation_paired_screen_20260926.json).

## Current decision

## Full-coverage four-token follow-up

The earlier stride-24 screen used `patch_len=16`, which left gaps in the 96-step input window. A fixed follow-up replaced it with `patch_len=24, patch_stride=24`: four tokens remain, but all input time points are represented. The PEMS07 resource preflight (`B=32,L=96,C=883,H=12`) measured 27,587 parameters, 0.489 ms median (0.490 ms mean), and 105.2 MB peak allocation, approximately 18.5% faster than the same-snapshot TimeMixer reference at 0.601 ms.

Fresh paired seeds 2091/2092/2093 completed as validation-only active/null phases. Both validators were valid with 3/3 unique jobs and no test artifacts. Best validation MSE by seed was null `0.063536/0.063983/0.063578` (mean `0.063699`) and active lambda=0.001 `0.063499/0.064017/0.063626` (mean `0.063714`). The paired active change was `+0.024%`, with only one of three seeds improving, so the AutoCon effect is not directionally reproducible. Both arms nevertheless beat the historical TimeMixer mean `0.065095` by about 2.1% on this coordinate.

This is a promising local architecture result, not yet a general Mamba or contrastive-learning result. The candidate is eligible for a fresh four-PEMS confirmation; active/null must remain a separate ablation, and no test release is permitted until that matrix is complete.

### Four-PEMS confirmation outcome

The fresh-seed confirmation (`2101/2102/2103`) is now complete. Both active and null phases are `valid=true` with 12/12 jobs, zero errors, and no test metrics. The active model records PEMS03/04/07/08 validation MSE of `.064881/.082387/.064088/.093733`, respectively. Against the frozen external best baselines, this is `+1.623%/+2.705%/-1.546%/+0.905%`. The predeclared 1% non-inferiority rule therefore fails on PEMS03 and PEMS04.

The exact paired active/null changes are `-0.109%/-0.012%/+0.116%/+0.002%`. Because their signs are inconsistent, no stable AutoCon accuracy benefit is established. The full-coverage patch repair has a real PEMS07 local speed/accuracy signal, but it does not preserve accuracy across all four PEMS datasets and is stopped before test release. See the [confirmation readout](pems_fullcover_p24_s24_mambatsv3_autocon_h12_confirmation_readout_20260926.md).

This screen narrows the engineering conclusion:

- Reducing patch-token count can make MambaTS v3 faster than TimeMixer, Mamba2, and iTransformer on the high-dimensional PEMS07 forward shape, but loses cross-PEMS accuracy.
- A variable-token Mamba can be faster with a prediction-connected AutoCon representation. Corrected selected-lag supervision removes the prior objective degeneracy, but the width-64 PEMS07 result still fails the accuracy non-inferiority gate.
- The original wide MambaTSv3AutoCon remains the accuracy candidate for PEMS H=12 validation, but it has not passed the efficiency requirement. None of the architectures in this screen should enter test release or a high-level paper claim.
