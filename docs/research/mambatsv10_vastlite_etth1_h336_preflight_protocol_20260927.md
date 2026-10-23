# MambaTS v10 VAST-lite ETTh1 H336 Preflight

## Question

Can a channel order learned once from train-only signed ACF profiles improve
the existing single-scan MambaTS path on ETTh1/H336, while the AutoCon
representation remains a direct forecast input?

## Fixed Model

The public MambaTS implementation uses VPT during training and a learned
transition-cost ordering at inference. v10 deliberately does not claim to
replicate that implementation. It fits signed ACF profiles on the training
split only at lags `24,48`, normalizes each profile, and forms one deterministic
greedy channel-similarity path. The existing channel-major patch tokens are
permuted by this fixed path before their sole Mamba-1 scan, then restored to
the original channel order before forecasting. No dynamic validation/test
statistic, second scan, attention layer, or per-batch order search is used.

AutoCon's time-indexed representation continues to feed the existing
zero-initialized forecast residual. The null and active arms therefore differ
only in `AutoCon_lambda`.

## Matrix And Gate

ETTh1 multivariate H336 uses three exact paired fresh seeds (`2201--2203`),
eight maximum epochs, batch 32, MeanStd normalization, cosine schedule, and
`skip_test=true`. The arms are v10 lambda zero, v10 lambda `.001`, and
same-contract PatchTST.

v10 advances only if active is no worse than PatchTST on mean best validation
MSE and improves over its exact null on at least two seed pairs. A failure
stops the branch: it does not permit order/lambda retuning, resource claims,
or test release.

The frozen machine-readable contract is
[`mambatsv10_vastlite_etth1_h336_preflight_20260927.json`](mambatsv10_vastlite_etth1_h336_preflight_20260927.json).
