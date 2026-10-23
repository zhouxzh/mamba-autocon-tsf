# BiPatchMamba AutoCon H12 Transfer Readout

Validation-only. No held-out test metric was generated or read.

| Dataset | Active | Null | Plain | Active vs null | wins/null | Strongest baseline | Active vs baseline | External gate | Transfer gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| ETTh1 | 0.320870 | 0.322460 | 0.316512 | -0.493% | 2/2 | MambaTSv3 (0.312156) | +2.791% | False | False |
| electricity | 0.122824 | 0.125729 | 0.108246 | -2.311% | 2/2 | TimeMixer (0.105503) | +16.417% | False | False |
| traffic | 0.343654 | 0.347799 | 0.324494 | -1.192% | 2/2 | iTransformer (0.324679) | +5.844% | False | False |
| weather | 0.248507 | 0.246464 | 0.220066 | +0.829% | 1/2 | TimeMixer (0.220755) | +12.572% | False | False |

Source snapshot: `e36b643dd16d6c52f9fbc700c8aa6d17a0fbe86ec2bfbdfcd4ea6d0cf4cb1c8e`.
