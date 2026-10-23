# PEMS-BAY H336 Local-Cluster Confirmation Readout

| Seed | Active validation | Null validation | Active test | Null test | Active vs null |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5600 | 2.404347 | 2.428002 | 2.290376 | 2.308820 | -0.799% |
| 5601 | 2.393448 | 2.418023 | 2.280559 | 2.300299 | -0.858% |
| 5602 | 2.419077 | 2.450778 | 2.298170 | 2.322950 | -1.067% |
| Mean | - | - | 2.289701 | 2.310690 | -0.908% |

All validation gates passed before test access. The paired test-MSE difference
is -0.020988 with normal-approximation 95% interval [-0.024776, -0.017200].
This independent long-horizon confirmation is separate from H192 results.

Prediction-only inference reports one Mamba scan, no AutoCon representation,
and no relation loss; batch-four median latency is 0.315024 ms with 20.84 MB
peak allocated memory. Audit: [`audit.json`](../../artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_h336_confirmation_20261003/audit.json).

The closed validation robustness extension also favors active in 3/3 seeds for
every condition: ordinary validation -1.095%, late origins -0.989%, 10%/30%
missing channels -0.974%/-0.746%, and 0.05/0.10 noise -1.091%/-1.079%.
These post-confirmation read-only results are recorded in
[`pems_bay_h336_robustness_audit.json`](../../artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_h336_confirmation_20261003/pems_bay_h336_robustness_audit.json).
