# Full-Coverage MambaTS v3 PEMS H12 Confirmation Readout

## Scope

This is a validation-only exploratory confirmation of the frozen `MambaTSv3AutoCon` full-coverage four-token graph: `patch_len=24`, `patch_stride=24`, `d_model=16`, `d_ff=32`, one causal Mamba scale, and `AutoCon_lambda=0.001`. The exact lambda-zero graph is the null control. All jobs use `M` features, `seq_len=96`, `pred_len=12`, MeanStd normalization, 20 epochs, CUDA, and `skip_test=true`.

The earlier PEMS07 selection probe used seeds `2091/2092/2093`; this confirmation uses fresh paired seeds `2101/2102/2103` and no test metric was created or inspected. Both phase validators reported `valid=true`, 12 expected/12 unique jobs, and zero errors.

## Results

`MSE` is the minimum pure-forecast validation loss. The frozen external baseline is taken from the same-source PEMS H12 baseline readout. A negative difference is better.

| Dataset | Null MSE | Active MSE | Active vs null | Seed wins | Frozen best external baseline | Active vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| PEMS03 | 0.064952 | 0.064881 | -0.109% | 3/3 | TimesNet 0.063845 | +1.623% |
| PEMS04 | 0.082397 | 0.082387 | -0.012% | 2/3 | TimeMixer 0.080217 | +2.705% |
| PEMS07 | 0.064014 | 0.064088 | +0.116% | 1/3 | TimeMixer 0.065095 | -1.546% |
| PEMS08 | 0.093731 | 0.093733 | +0.002% | 2/3 | TimeMixer 0.092892 | +0.905% |

The active/null changes have no common direction: the active objective improves PEMS03 and PEMS04 slightly, but degrades PEMS07 and PEMS08. The PEMS03 difference is directionally 3/3, but its external-baseline comparison fails; the other effects are near zero. It is therefore invalid to claim a stable AutoCon accuracy contribution.

## Decision

The preregistered non-inferiority margin is 1% relative validation MSE per dataset. Active exceeds it on PEMS03 (+1.623%) and PEMS04 (+2.705%), so this candidate is stopped before test release. Its PEMS07 result is favorable, and its PEMS07 prediction-only preflight was fast (0.489 ms median versus TimeMixer 0.601 ms), but a single dataset cannot override a failed four-dataset matrix.

This result remains useful as evidence that full temporal coverage is materially better than the earlier gapful sparse patch design, but it is not a paper-ready main model. No new tuning, seed reuse, or test release may use this confirmation matrix.

## Audit Artifacts

- [Frozen plan](pems_fullcover_p24_s24_mambatsv3_autocon_h12_confirmation_20260926.json)
- [Null phase validation](../../artifacts/exploratory/pems_fullcover_p24_s24_mambatsv3_autocon_h12_confirmation_20260926/null/phase_validation.json)
- [Active phase validation](../../artifacts/exploratory/pems_fullcover_p24_s24_mambatsv3_autocon_h12_confirmation_20260926/active/phase_validation.json)
- [PEMS07 resource preflight](../../artifacts/exploratory/pems07_mambatsv3_fullcover_p24_s24_d16_resource_preflight_20260926/model_resources.json)
