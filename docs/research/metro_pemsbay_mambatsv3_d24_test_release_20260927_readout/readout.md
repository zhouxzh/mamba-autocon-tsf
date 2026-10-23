# Frozen METR-LA / PEMS-BAY MambaTSv3 Test Release

Five fresh paired seeds; one-sided 95% upper MSE-ratio bounds use the registered 1.02 non-inferiority threshold.

| Dataset | Baseline | Candidate MSE | Baseline MSE | Change | Wins | Upper ratio | Non-inferior |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| metr_la | DLinear | 0.432489 | 0.430784 | +0.397% | 2/5 | 1.0113 | True |
| metr_la | NLinear | 0.432489 | 0.431782 | +0.165% | 3/5 | 1.0059 | True |
| metr_la | PatchTST | 0.432489 | 0.427167 | +1.248% | 0/5 | 1.0203 | False |
| metr_la | TimeMixer | 0.432489 | 0.420330 | +2.900% | 0/5 | 1.0394 | False |
| metr_la | iTransformer | 0.432489 | 0.438699 | -1.395% | 3/5 | 1.0015 | True |
| metr_la | MambaTSv3 | 0.432489 | 0.430351 | +0.498% | 1/5 | 1.0106 | True |
| pems_bay | DLinear | 0.344955 | 0.436691 | -21.007% | 5/5 | 0.8019 | True |
| pems_bay | NLinear | 0.344955 | 0.436555 | -20.983% | 5/5 | 0.8018 | True |
| pems_bay | PatchTST | 0.344955 | 0.403969 | -14.602% | 5/5 | 0.8701 | True |
| pems_bay | TimeMixer | 0.344955 | 0.375111 | -7.999% | 5/5 | 0.9463 | True |
| pems_bay | iTransformer | 0.344955 | 0.358730 | -3.840% | 5/5 | 0.9756 | True |
| pems_bay | MambaTSv3 | 0.344955 | 0.347258 | -0.665% | 4/5 | 1.0031 | True |

This release is restricted to non-graph METR-LA/PEMS-BAY H12. DLinear/NLinear are descriptive speed boundaries; no universal efficiency claim follows.
