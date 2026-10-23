# Independent PEMS-BAY External Robustness Replication

Status: independently registered five-seed, validation-only replication. It
uses seeds `6001--6005` and is not pooled with the prior three-seed extension.
No test row was read.

Mamba-1 active/null and locked official PatchTST were retrained under the same
PEMS-BAY H192, 10% chronological protocol, then evaluated under identical
validation stresses.

| Condition | Active vs null | Active vs official PatchTST | PatchTST upper ratio | Active wins vs PatchTST |
| --- | ---: | ---: | ---: | ---: |
| In distribution | -2.908%, 5/5 | -1.650% | 0.98560 | 5/5 |
| Late shift | -2.655%, 5/5 | -1.635% | 0.98565 | 5/5 |
| Missing channels 10% | -2.617%, 5/5 | -1.505% | 0.98678 | 5/5 |
| Missing channels 30% | -2.052%, 5/5 | -1.174% | 0.98979 | 5/5 |
| Noise 0.05 | -2.912%, 5/5 | -1.684% | 0.98523 | 5/5 |
| Noise 0.10 | -2.923%, 5/5 | -1.790% | 0.98411 | 5/5 |

All six cells pass the one-standard-error external noninferiority criterion.
This independently resolves the original three-seed PEMS-BAY extension's
near-threshold upper bounds without pooling samples or changing the model.

The same independent candidate seeds also outperform a separately trained,
pinned official TimesNet baseline by `10.366%` (5/5 wins; upper 0.91665) on
ordinary validation; see `mamba1_pems_independent_timesnet_20261006.md`.

Artifacts:

- `artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006/audit.json`
- `artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006/robustness_validation/external_patchtst_audit.json`
- `scripts/run_mambatsv3_pems_mamba1_10pct_independent_replication.py`
- `scripts/run_pems_mamba1_10pct_independent_robustness.py`
- `scripts/run_official_patchtst_pems_mamba1_10pct_independent_replication.py`
- `scripts/run_official_patchtst_pems_mamba1_10pct_independent_robustness.py`

All guarded stages ran below 52C; no fan escalation or guard event occurred.
