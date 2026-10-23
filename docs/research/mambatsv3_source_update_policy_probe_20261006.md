# Source-Domain Update-Order Probe

Status: validation-only mechanism boundary. No test rows were read.

The shared pretraining encoder was previously optimized source by source in a
fixed order. This probe added an explicit `cyclic` policy that rotates the
source order once per epoch; it changes no model parameter, no relation label,
and no deployed scan. Sequential and cyclic active/null arms used the same
fresh seed and exact 10% target coordinates.

| Target | Sequential active vs null | Cyclic active vs null | Cyclic vs sequential active |
| --- | ---: | ---: | ---: |
| PEMS-BAY H192 | -2.475% | -2.001% | +0.209% |
| METR-LA H192 | -0.383% | -0.665% | -0.422% |

Cyclic source updates improve METR-LA but degrade PEMS-BAY, so the registered
cross-domain gate fails. The default remains `source_update_policy=sequential`
for compatibility with the frozen release. The cyclic policy is retained as a
supported, explicitly auditable option, not promoted to a main result.

The audit also verified that cyclic source orders were genuinely rotated:
`[source0, source1, source2]`, `[source1, source2, source0]`, and
`[source2, source0, source1]` across the three pretraining epochs. Peak
telemetry was 52C/0% fan; no guard event triggered.

Artifacts:

- `artifacts/exploratory/mambatsv3_mamba1_source_update_policy_probe_20261006/plan.json`
- `artifacts/exploratory/mambatsv3_mamba1_source_update_policy_probe_20261006/audit.json`
- `scripts/run_mambatsv3_mamba1_source_update_policy_probe.py`
- `scripts/audit_mambatsv3_mamba1_source_update_policy_probe.py`
