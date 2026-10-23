# D24 Full-Coverage MambaTS v3 PEMS H12 Confirmation Readout

## Scope

This validation-only confirmation freezes a four-token, full-history `MambaTSv3AutoCon` graph: `patch_len=24`, `patch_stride=24`, `d_model=24`, `d_ff=48`, one causal Mamba-1 scale, and `AutoCon_lambda=0.001`. Its strict null keeps every graph and training parameter identical but sets the effective lambda to zero. PEMS03/04/07/08 all use `M`, `L=96`, `H=12`, MeanStd normalization, 20 epochs, CUDA, and `skip_test=true`.

The architecture was selected only by the PEMS07 resource preflight plus a separate PEMS07 seed set. This confirmation uses fresh paired seeds `2131/2132/2133`. Both phases are complete and valid: 12 expected/12 unique jobs, zero validator errors, and no test metrics.

## Accuracy

`MSE` is the minimum pure-forecast validation loss. Negative external-baseline differences are improvements. The listed external reference is the strongest frozen model from the same-source PEMS H12 screen.

| Dataset | Null MSE | Active MSE | Active vs null | Seed wins | Frozen best external baseline | Active vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| PEMS03 | 0.063102 | 0.063093 | -0.015% | 2/3 | TimesNet 0.063845 | -1.178% |
| PEMS04 | 0.080197 | 0.080203 | +0.007% | 1/3 | TimeMixer 0.080217 | -0.018% |
| PEMS07 | 0.062497 | 0.062459 | -0.061% | 2/3 | TimeMixer 0.065095 | -4.049% |
| PEMS08 | 0.093078 | 0.093008 | -0.075% | 3/3 | TimeMixer 0.092892 | +0.125% |

The candidate is within the predeclared 1% non-inferiority margin on every dataset and better than the frozen external reference on three of four means. PEMS08 is `+0.125%`, within tolerance rather than a win. AutoCon active/null improvements are small: they are directionally favorable on PEMS03, PEMS07, and PEMS08, while PEMS04 is effectively neutral/slightly adverse. This supports retaining AutoCon as a shared representation-learning component and reporting an exact null ablation; it does not establish a large independent contrastive-loss effect.

## Prediction-Only GPU Resources

All resource values are serial GPU measurements on `cuda:0` (RTX 5090 D), device-resident synthetic `B=32,L=96,H=12` input, 25 warm-up and 100 timed forwards, with AutoCon representation return disabled. They exclude training, data loading, and the contrastive loss. The PEMS07 TimeMixer record is the frozen same-source reference with the same batch/shape protocol; PEMS03/04/08 use the same d24 capacity protocol.

| Dataset | Candidate params | Candidate median (ms) | TimeMixer params | TimeMixer median (ms) | Candidate speed advantage |
|---|---:|---:|---:|---:|---:|
| PEMS03 (C=358) | 21,270 | 0.303 | 75,790 | 1.431 | 78.8% |
| PEMS04 (C=307) | 19,587 | 0.289 | 70,843 | 1.277 | 77.4% |
| PEMS07 (C=883) | 38,595 | 0.530 | 158,643 | 0.599 | 11.5% |
| PEMS08 (C=170) | 15,066 | 0.268 | 57,554 | 0.869 | 69.2% |

The candidate is faster and smaller than TimeMixer on every PEMS H12 shape. This is a forward-path result; it must not be described as end-to-end training or full system speed.

## Decision

The d24 full-coverage candidate passes the H12 four-PEMS accuracy non-inferiority and TimeMixer forward-latency gates. It is therefore eligible for a fresh H96/long-horizon confirmation and only then a separately authorized held-out-test release. The four-dataset H12 matrix itself remains validation-only and cannot substantiate a publication claim.

## Auditable Artifacts

- [Frozen plan](pems_fullcover_p24_s24_d24_mambatsv3_autocon_h12_confirmation_20260926.json)
- [Null phase validation](../../artifacts/exploratory/pems_fullcover_p24_s24_d24_mambatsv3_autocon_h12_confirmation_20260926/null/phase_validation.json)
- [Active phase validation](../../artifacts/exploratory/pems_fullcover_p24_s24_d24_mambatsv3_autocon_h12_confirmation_20260926/active/phase_validation.json)
- [PEMS03 candidate resource](../../artifacts/exploratory/pems03_mambatsv3_fullcover_p24_s24_d24_resource_serial_20260926/model_resources.json)
- [PEMS04 candidate resource](../../artifacts/exploratory/pems04_mambatsv3_fullcover_p24_s24_d24_resource_serial_20260926/model_resources.json)
- [PEMS07 candidate resource](../../artifacts/exploratory/pems07_mambatsv3_fullcover_p24_s24_d24_resource_preflight_20260926/model_resources.json)
- [PEMS08 candidate resource](../../artifacts/exploratory/pems08_mambatsv3_fullcover_p24_s24_d24_resource_serial_20260926/model_resources.json)
