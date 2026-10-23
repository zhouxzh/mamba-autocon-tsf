# PEMS d32 full-coverage MambaTS v3 AutoCon H96 confirmation

Date: 2026-09-26  
Protocol: validation-only, train-only normalization, `skip_test=true`  
Plan: [`pems_fullcover_p24_s24_d32_mambatsv3_autocon_h96_confirmation_20260926.json`](./pems_fullcover_p24_s24_d32_mambatsv3_autocon_h96_confirmation_20260926.json)

## Scope

This is the fresh four-PEMS confirmation of the fixed full-coverage graph
`patch_len=24`, `patch_stride=24`, `d_model=32`, `d_ff=64`, `d_state=8`,
`d_conv=2`, `expand=1`. The active arm uses `AutoCon_lambda=0.001`; the paired
null arm uses the identical graph and `AutoCon_lambda=0`. There are 12 jobs per
arm (four datasets and three seeds), and no test metric is produced.

## Validation result

The active and null phases each passed the strict 12/12 manifest and provenance
validation. The table reports the minimum pure-forecast validation MSE averaged
over the three paired seeds. Relative values are `active / reference - 1`.

| Dataset | Null MSE | Active MSE | Active vs null | Historical H96 reference | Active vs reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS03 | 0.158796 | 0.158803 | +0.004% | 0.162275 (TimesNet) | -2.140% |
| PEMS04 | 0.200069 | 0.200145 | +0.038% | 0.195812 (TimesNet) | +2.213% |
| PEMS07 | 0.213677 | 0.213702 | +0.012% | 0.215844 (MambaTS v3) | -0.992% |
| PEMS08 | 0.368927 | 0.365468 | -0.937% | 0.375467 (TimesNet) | -2.663% |

The historical-reference stopping rule was not triggered: only PEMS04 is more
than 2% worse. However, the active-vs-null contrast is not directionally
consistent: three datasets are slightly worse and one improves. Therefore this
run does **not** establish a causal AutoCon accuracy gain. It only supports
continuing with a same-source baseline and resource audit for the d32 candidate.

## Reproducibility and boundary

- Fresh paired seeds: `2171`, `2172`, `2173`.
- Dataset fingerprints and all launch settings are frozen in the plan JSON.
- Phase validation: `active/phase_validation.json` and `null/phase_validation.json`,
  both `valid=true`, `expected_jobs=12`, `manifest_unique_jobs=12`.
- Artifact roots:
  [`active`](../../artifacts/exploratory/pems_fullcover_p24_s24_d32_mambatsv3_autocon_h96_confirmation_20260926/active/)
  and
  [`null`](../../artifacts/exploratory/pems_fullcover_p24_s24_d32_mambatsv3_autocon_h96_confirmation_20260926/null/).
- No test-set, cross-horizon, or publication-level superiority claim is allowed
  from this confirmation.

## Same-shape resource check

Using the same synthetic device-resident input protocol on the RTX 5090 D
(batch 32, 25 warm-up and 100 timed forwards), the d32 candidate measured
`0.625584 ms` median latency, `69,707` total parameters and `138.03 MB`
steady-state peak increment. A same-shape TimeMixer (`d_model=32`, `d_ff=64`)
measured `0.718736 ms`, `183,171` parameters and `99.78 MB` peak increment.
Thus the candidate is about `12.96%` faster and `61.96%` smaller in parameter
count in this forward-only load test, at the cost of about `38.3 MB` more peak
incremental allocation. This is an engineering measurement, not end-to-end
training speed or a statistical forecasting result.

Resource artifacts:
[`MambaTS v3`](../../artifacts/exploratory/pems07_mambatsv3_fullcover_p24_s24_d32_h96_resource_preflight_20260926/model_resources.json),
[`TimeMixer`](../../artifacts/exploratory/pems07_timemixer_h96_b32_d32_resource_preflight_20260926/model_resources.json).
