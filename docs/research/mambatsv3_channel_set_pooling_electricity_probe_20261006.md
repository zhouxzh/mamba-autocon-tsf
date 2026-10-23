# Learned Channel Set-Pooling on Electricity

Status: completed validation-only high-dimensional mechanism probe. No test
reader was executed.

The frozen Electricity H96 strong-calibration protocol was used unchanged:
Traffic source relation fraction `0.1`, Electricity's chronological 25%
few-shot calibration prefix at active fraction `0.3`, a frozen transferred
Mamba-1 24/8 core, and one deployed Mamba scan. The only change is the learned
permutation-equivariant channel set-pooling context, compared at one fresh
seed against both its pooling null and the ordinary strong-calibration active
encoder.

| Arm | Validation MSE | Comparison |
| --- | ---: | --- |
| Plain active | 8.166603 | Reference |
| Pooling active | 8.181410 | +0.181% vs plain active |
| Pooling null | 8.225519 | Pooling active is -0.536% |

The learned pooling branch preserves an active/null ACF relation benefit, but
does not improve the already selected plain strong-calibration active model.
It is therefore a negative high-dimensional structure result and is not
replicated or evaluated on test data. This agrees with its two-domain
replication boundary: pooling can preserve the pretraining effect but is not
a universal accuracy upgrade.

All six guarded stages completed serially below 49C with no fan escalation or
guard trigger.

Artifacts:

- `artifacts/exploratory/mambatsv3_electricity_mamba1_set_pooling_strong_calibration_probe_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_electricity_mamba1_set_pooling_strong_calibration_probe_20261006/audit.json`
- `scripts/run_mambatsv3_electricity_set_pooling_strong_calibration_probe.py`
- `scripts/audit_mambatsv3_electricity_set_pooling_strong_calibration_probe.py`
