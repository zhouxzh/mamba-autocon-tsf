# Local-Cluster Robustness Readout

This post-confirmation extension evaluates only closed validation blocks. The
active and exact-null checkpoints were frozen before these input perturbations;
no checkpoint training, model selection, or test access occurred here.

| Target | Ordinary | Late shift | Missing 10% | Missing 30% | Noise .05 | Noise .10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY | -1.714% (3/3) | -1.564% (3/3) | -1.545% (3/3) | -1.177% (3/3) | -1.710% (3/3) | -1.702% (3/3) |
| METR-LA | -0.477% (3/3) | -0.414% (3/3) | -0.425% (3/3) | -0.328% (3/3) | -0.476% (3/3) | -0.473% (3/3) |
| Weather | -0.213% (3/3) | -0.257% (3/3) | -0.157% (3/3) | -0.092% (3/3) | -0.212% (3/3) | -0.209% (3/3) |

Values are active versus exact-null mean MSE; negative is better. The exact
artifacts are [`PEMS-BAY`](../../artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_robustness_extension_20261003/pems_bay_robustness_audit.json),
[`METR-LA`](../../artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_robustness_extension_20261003/metr_la_robustness_audit.json),
and [`Weather`](../../artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_robustness_extension_20261003/weather_robustness_audit.json).

This is evidence for stable direction under the listed input perturbations, not
evidence that the method solves arbitrary distribution shift or missing-data
imputation. The magnitude remains domain-dependent, especially on Weather.
