# Multi-Prototype Channel Pooling on Electricity

Status: completed validation-only negative high-dimensional probe. No target
test rows were read.

This is the final scan-pre-context variant after fixed mean/std and learned
single-summary pooling. Four shared prototype queries attend over channels at
each patch and form four channel-subgroup descriptors, which are projected
back to the token width before the existing Mamba-1 scan. The query count and
parameters are independent of channel count; channel order is permutation
equivariant; no additional Mamba scan is created.

The Electricity H96 strong-calibration protocol at fresh seed 5989 produced:

| Arm | Validation MSE | Comparison |
| --- | ---: | --- |
| Plain active | 8.172686 | Reference |
| Prototype active | 8.188600 | +0.195% vs plain active |
| Prototype null | 8.236143 | Prototype active is -0.577% |

Prototype pooling preserves a causal active/null gain but does not improve the
matched plain AutoCon model. It fails the pre-registered high-dimensional
selection gate; no replication, external baseline run, or test release is
authorized. Together with the fixed summary and single-attention summary
probes, this closes the channel-summary-before-scan family under the current
protocol.

All guarded stages completed below 48C with no fan escalation or guard event.

Artifacts:

- `artifacts/exploratory/mambatsv3_electricity_mamba1_prototype_pooling_probe_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_electricity_mamba1_prototype_pooling_probe_20261006/audit.json`
- `scripts/run_mambatsv3_electricity_prototype_pooling_probe.py`
- `scripts/audit_mambatsv3_electricity_prototype_pooling_probe.py`
