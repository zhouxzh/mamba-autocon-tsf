# ACF Stability-Weighted Positive Pair Probe

Status: completed validation-only negative mechanism result. No target test
rows were read.

This probe is distinct from raw ACF-magnitude weighting. For every selected
channel-lag pair, the source training prefix was split chronologically in two;
the positive weight is its signed-ACF agreement across those halves. Peak
selection, source windows, 24/8 Mamba-1 shared encoder, target adaptation,
and `relation_loss_fraction=0.10` remained fixed. The stability score is
train-only and is consumed only while constructing the pretraining InfoNCE
loss, so the deployed model has no extra parameter, operation, or scan.

| Target | Null MSE | Plain InfoNCE | Stability weighted | Stability vs plain | Stability vs null |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192, 10% | 2.857066 | 2.780203 | 2.780629 | +0.015% | -2.675% |
| METR-LA H192, 10% | 1.594552 | 1.589495 | 1.589518 | +0.001% | -0.316% |

Source diagnostic weights were nondegenerate (mean `0.91--0.96`, more than
98% nonzero selected peaks), and active relation coverage remained at least
0.8. The stability arm nevertheless lost to ordinary InfoNCE in both domains.
The cross-domain gate fails; no replication or test release is authorized.

Together with the relation-fraction and raw-magnitude probes, this closes the
class of simple positive-pair reweighting mechanisms. Further work must alter
the shared encoder's transferable cross-channel state construction rather than
reweighting the same relation pairs.

Artifacts:

- `artifacts/exploratory/mambatsv3_mamba1_acf_stability_probe_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_mamba1_acf_stability_probe_20261006/audit.json`
- `scripts/run_mambatsv3_mamba1_acf_stability_probe.py`
- `scripts/audit_mambatsv3_mamba1_acf_stability_probe.py`

All stages were serial guarded jobs. Peak telemetry was 52C/0% fan on
PEMS-BAY and 48C/0% fan on METR-LA; no guard event triggered.
