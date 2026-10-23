# MambaTS v9 Decomposition ETTh1 H336 Preflight

## Question

Can a single seasonal Mamba scan plus a shared trend extrapolator close the
ETTh1/H336 error gap while retaining a direct prediction path from the
AutoCon representation?

## Fixed Model

MambaTS v9 decomposes only the observed, MeanStd-normalized 96-step input
window with `moving_avg=25`. The seasonal residual is encoded by the existing
single shared Mamba-1 patch scan (`16/8`, `d_model=32`, `d_ff=64`, one layer).
A channel-shared linear map predicts the trend. The two forecasts are summed.
The AutoCon representation then supplies a horizon gate for a zero-initialized
correction between those two forecasts. This is an independent adaptation of
DMamba's decomposition principle, not a reproduction of its variable-direction
encoder or MLP stack.

## Matrix

Three exact paired fresh seeds (`2191--2193`) compare MambaTSv9AutoCon at
`lambda=0` and `.001`; same-contract PatchTST is the external reference.
Every arm uses ETTh1 multivariate H336, eight epochs maximum, batch 32,
MeanStd normalization, cosine learning rate schedule, and `skip_test=true`.

The candidate advances only if active is no worse than PatchTST on mean best
validation MSE and improves over its exact null on at least two seeds. A pass
does not release tests or establish speed; it only permits an independent
multi-dataset confirmation and matched resource benchmark.

The machine-readable frozen plan is
[`mambatsv9_decomposition_etth1_h336_preflight_20260927.json`](mambatsv9_decomposition_etth1_h336_preflight_20260927.json).
