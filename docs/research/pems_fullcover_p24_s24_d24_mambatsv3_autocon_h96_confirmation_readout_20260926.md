# D24 Full-Coverage MambaTS v3 PEMS H96 Confirmation Readout

## Scope and Integrity

The architecture, optimizer, normalization, and AutoCon setting are fixed from the H12-qualified d24 candidate; only `pred_len=96` changes. PEMS03/04/07/08 use fresh paired seeds `2141/2142/2143`, 20 epochs, CUDA, and `skip_test=true`. Both active and strict lambda-zero null phases are complete and `valid=true`, with 12 expected/12 unique jobs and no test artifacts.

## Validation MSE

| Dataset | Null MSE | Active MSE | Active vs null | Seed wins | Historical H96 strong reference | Active vs reference |
|---|---:|---:|---:|---:|---:|---:|
| PEMS03 | 0.168119 | 0.168134 | +0.009% | 1/3 | TimesNet 0.162275 | +3.610% |
| PEMS04 | 0.215007 | 0.214917 | -0.042% | 3/3 | TimesNet 0.195812 | +9.757% |
| PEMS07 | 0.219293 | 0.218978 | -0.144% | 3/3 | MambaTSv3 0.215844 | +1.452% |
| PEMS08 | 0.367345 | 0.367357 | +0.003% | 1/3 | TimesNet 0.375467 | -2.160% |

The historical references are a screening boundary rather than a same-source comparison, but they are sufficient for the registered stop rule. PEMS03 and PEMS04 exceed their references by more than 2%, so two dataset means fail the predeclared long-horizon gate. The active/null direction is also inconsistent. No H96 test release, same-source baseline expansion, or further tuning of this d24 four-token graph is permitted.

The PEMS07 prediction-only resource preflight remained favorable (`0.567 ms` median, 54,891 parameters, B32/L96/C883/H96), but resource efficiency cannot compensate for the failed cross-dataset accuracy rule.

## Artifacts

- [Frozen H96 plan](pems_fullcover_p24_s24_d24_mambatsv3_autocon_h96_confirmation_20260926.json)
- [Null phase validation](../../artifacts/exploratory/pems_fullcover_p24_s24_d24_mambatsv3_autocon_h96_confirmation_20260926/null/phase_validation.json)
- [Active phase validation](../../artifacts/exploratory/pems_fullcover_p24_s24_d24_mambatsv3_autocon_h96_confirmation_20260926/active/phase_validation.json)
- [PEMS07 H96 resource preflight](../../artifacts/exploratory/pems07_mambatsv3_fullcover_p24_s24_d24_h96_resource_preflight_20260926/model_resources.json)
