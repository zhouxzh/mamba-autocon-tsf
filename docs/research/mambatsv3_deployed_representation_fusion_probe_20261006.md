# Deployed Representation-Fusion Transfer Probe

Status: validation-only mechanism probe. No test rows were read.

This probe connects the source-pretrained representation to deployed
forecasting without another Mamba scan. It freezes the encoder and transferred
`repr_` states, then trains only a zero-gated target fusion decoder on the
same few-shot target data. The direct-representation-pretrained source core is
compared with an otherwise identical forecast-only source core; both use the
same target fusion capacity.

| Target | Plain source + fusion | Direct-repr source + fusion | Direct repr vs plain fusion |
| --- | ---: | ---: | ---: |
| PEMS-BAY H192, 10% | 2.763633 | 2.759782 | -0.139% |
| METR-LA H192, 10% | 1.590175 | 1.591708 | +0.096% |

The target fusion was active: the learned absolute fusion-scale means were
`0.0234/0.0322` for plain/direct representation on PEMS-BAY and
`0.0216/0.0206` on METR-LA. Thus the mixed result is not a zero-gate failure.
The direct representation arm does not consistently improve the deployed
forecast, so it is not replicated or released.

The deployed path still constructs no AutoCon loss and invokes only the
existing Mamba scan, but it internally computes the representation required by
the fusion decoder. This is a validation-only boundary, not a latency claim.

Artifacts:

- `artifacts/exploratory/mambatsv3_mamba1_repr_fusion_transfer_probe_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_mamba1_repr_fusion_transfer_probe_20261006/audit.json`

All four guarded target-adaptation stages completed below 48C without a guard
trigger.
