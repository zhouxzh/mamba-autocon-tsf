# PEMS-BAY 25% Label Confirmation Readout

| Seed | Active validation | Null validation | Active test | Null test | Active vs null |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5540 | 2.356439 | 2.387263 | 2.281244 | 2.304630 | -1.015% |
| 5541 | 2.340603 | 2.372531 | 2.270849 | 2.294834 | -1.045% |
| 5542 | 2.334530 | 2.373790 | 2.267676 | 2.297178 | -1.284% |
| Mean | - | - | 2.273256 | 2.298880 | -1.115% |

All three active validation checkpoints passed the predeclared gate, and all
three active test checkpoints won their exact-null controls. The paired test
difference is -0.025624 with a normal-approximation 95% interval of
[-0.029439, -0.021809]. This is a separate 25%-label confirmation and is not
pooled with the 10%-label confirmation.

The prediction-only benchmark remains one Mamba scan with no AutoCon relation
loss or representation returned; batch-4 median latency is 0.315520 ms. The
full lifecycle audit is [`audit.json`](../../artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_25_confirmation_20261003/audit.json).
