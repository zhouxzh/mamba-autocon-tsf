# d32/24 Traffic Transfer Bridge

Status: validation-only negative symmetry probe. No test row was read.

This probe mirrors the Electricity d32/24 bridge: compatible PEMS d32/24
initial cores, Electricity source prefix plus Traffic 25% calibration prefix,
next-patch pretraining, train-only forecast-level ACF relations, frozen target
core, and one Mamba scan. The active calibration relation fraction is `0.3`;
the null arm has identical predictive calibration data and zero relation loss.

| Quantity | Result |
| --- | ---: |
| Active validation MSE | 0.816451 |
| Null validation MSE | 0.812202 |
| Active vs null | +0.523% |
| Previous d24/8 source-only active | 0.817744 |
| d32/24 vs d24/8 absolute active | -0.158% |

The larger/full-coverage geometry slightly improves the active model's
absolute error, but improves the null more, reversing the ACF causal effect.
The active relation anchor fraction remains 0.875 in the final source epoch,
so this is a genuine domain-specific negative result rather than inactive
relation sampling. No replication, external baseline training, or test release
is authorized for this Traffic few-shot bridge.

All guarded stages completed below 51C with no fan escalation.

Artifacts:

- `artifacts/exploratory/mambatsv3_traffic_mamba1_d32_stride24_transfer_probe_20261006/audit.json`
- `scripts/run_mambatsv3_traffic_d32_stride24_transfer_probe.py`
- `scripts/audit_mambatsv3_traffic_d32_stride24_transfer_probe.py`
