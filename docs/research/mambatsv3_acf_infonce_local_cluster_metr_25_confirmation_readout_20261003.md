# METR-LA 25% Label Confirmation Readout

| Seed | Active validation | Null validation | Active test | Null test | Active vs null |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5580 | 1.184043 | 1.189409 | 1.925583 | 1.929626 | -0.210% |
| 5581 | 1.186327 | 1.192714 | 1.932363 | 1.938369 | -0.310% |
| 5582 | 1.191474 | 1.198321 | 1.937030 | 1.944674 | -0.393% |
| Mean | - | - | 1.931659 | 1.937556 | -0.304% |

All three predeclared validation gates passed before test access. The paired
test-MSE difference is -0.005898 with normal-approximation 95% interval
[-0.007939, -0.003857]. This is a separate 25%-label confirmation and is not
pooled with the 10%-label confirmation.

The prediction-only checkpoint uses one Mamba scan, no returned AutoCon
representation, and no relation loss; batch-four median forward latency is
0.310976 ms. Audit: [`audit.json`](../../artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_metr_25_confirmation_20261003/audit.json).
