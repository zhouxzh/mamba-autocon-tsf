# MambaTS v10 VAST-lite ETTh1 H336 Readout

## Decision

Stop this candidate. The replacement preflight completed all nine validation
jobs, but v10 violates both frozen continuation gates. It must not receive
post-hoc ordering, ACF-lag, lambda, width, resource, multi-dataset, or test
search.

## Validated Results

| v10 null MSE | v10 active MSE | PatchTST MSE | active vs null | active wins/null | active vs PatchTST | active wins/PatchTST |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.333936 | 1.337945 | 1.276616 | +0.301% | 0/3 | +4.804% | 0/3 |

All `9/9` jobs completed with one shared frozen source snapshot
`5ed9dd62cb52a6c76bff15ad2332d9cf6db235e87e9d50886eabd13ca1fd4379`.
All runs set `skip_test=true`; strict inspection found zero test metric files.
The six v10 runs emitted a train-only ACF sidecar with the same SHA256,
`dafe21da1094f679537909e993816ba378665221de6ef2e3e5fee0f639678295`.

The direct AutoCon forecast fusion was present in both v10 arms. Therefore
the result does not support a claim that this fixed ACF-derived VAST-lite
ordering or the active AutoCon objective closes the ETTh1/H336 gap. It also
does not evaluate official learned VPT/VAST, which uses a transition-cost
inference order rather than this fixed local approximation.

The machine-readable strict readout is
[`v10 retry readout`](../../artifacts/exploratory/mambatsv10_vastlite_etth1_h336_preflight_retry_20260927/readout/readout.md).

## Superseded Attempt

The initially launched plan had valid candidate jobs but invalid PatchTST
commands because a Mamba-only fusion flag was inherited from global plan
configuration. Its three baseline jobs failed before training. That entire
attempt is excluded from the table above and preserved only as an engineering
audit artifact. The replacement plan isolated candidate-only flags before
launch and is the only v10 evidence used here.
