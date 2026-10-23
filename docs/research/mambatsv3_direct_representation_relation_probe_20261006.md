# Direct Representation-Relation Pretraining Probe

Status: completed validation-only mechanism probe. No target test reader ran.

The ordinary forecast-level InfoNCE trains the source prediction head and
shared encoder through predicted future segments. This probe adds a second,
prediction-scale-calibrated ACF relation objective directly on the transferred
`repr_projection`, `repr_norm`, and `repr_head` output from the same encoded
Mamba states. It adds no Mamba scan and is absent from prediction-only
deployment.

| Target | Null MSE | Forecast-only active | Direct representation active | Representation vs forecast-only |
| --- | ---: | ---: | ---: | ---: |
| PEMS-BAY H192, 10% | 2.860845 | 2.771663 | 2.767588 | -0.147% |
| METR-LA H192, 10% | 1.596676 | 1.590776 | 1.591882 | +0.069% |

The new representation loss was not inert: PEMS-BAY had 0.833 active anchors
in every source epoch and METR-LA had 1.0, with nonzero calibrated weighted
losses throughout. It improves one target but loses on the other, failing the
cross-domain selection gate. No seed replication or test release is permitted.

This closes the direct-global-representation ACF relation variant at fraction
0.05. A future representation objective must change the positive/negative
semantics or target adaptation contract, rather than merely increasing this
same loss.

Artifacts:

- `artifacts/exploratory/mambatsv3_mamba1_representation_relation_probe_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_mamba1_representation_relation_probe_20261006/audit.json`
- `scripts/run_mambatsv3_mamba1_representation_relation_probe.py`
- `scripts/audit_mambatsv3_mamba1_representation_relation_probe.py`

All 12 guarded stages completed below 50C with no fan escalation or guard
trigger.
