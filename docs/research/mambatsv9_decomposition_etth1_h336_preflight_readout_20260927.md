# MambaTS v9 Decomposition ETTh1 H336 Preflight Readout

## Result

The frozen validation-only preflight completed all nine jobs: MambaTSv9AutoCon
at exact `lambda=0` and `.001`, plus same-contract PatchTST, each on fresh
ETTh1/H336 seeds `2191--2193`. All jobs use `skip_test=true`; no test metric
was generated or read. The three phase manifests have one identical source
snapshot (`275e85b8b5e180a0c3e23b81269fc8d79c5a8cad9b15bcccb82704c8b27595dd`),
zero failed jobs, and nine `training_complete.json` files.

| v9 null MSE | v9 active MSE | PatchTST MSE | active vs null | active wins/null | active vs PatchTST | active wins/PatchTST |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.330833 | 1.330459 | 1.279328 | -0.028% | 1/3 | +3.997% | 0/3 |

The model-level change is a single seasonal-residual Mamba scan plus a shared
trend linear head. In the active arm, the time-indexed AutoCon representation
also drives a ReZero horizon gate between trend and seasonal predictions. The
new model passes shape, zero-initialization, gradient-connectivity, runtime
contract, resource-registry, and training-plan registration tests. It is an
independent adaptation of the decomposition principle in the public
[DMamba repository](https://github.com/Csusheu/DMamba), not a reproduction.

## Decision

The pre-registered continuation rule required both: active no worse than
PatchTST on mean validation MSE, and active lower than its exact null on at
least two seed pairs. It passes neither condition. Stop the v9 branch: no
post-hoc moving-average/lambda/width tuning, no resource benchmark, no
multi-dataset expansion, and no test release.

The machine-readable readout and per-seed values are in
[`artifacts/exploratory/mambatsv9_decomposition_etth1_h336_preflight_20260927/readout`](../../artifacts/exploratory/mambatsv9_decomposition_etth1_h336_preflight_20260927/readout/).
