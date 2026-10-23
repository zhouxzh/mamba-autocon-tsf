# MambaACF Channel-State Mixer: Electricity H96

## Status

The frozen 27-job validation-only factorial passed the registered branch gate. No test metric was generated or read.

## Registered Design

The candidate is the existing `MambaACFAutoCon` path with train-only raw ACF memory and prediction-connected AutoCon representation, plus `LowRankGlobalChannelStateMixer(rank=4)`. The mixer conditions a shared hidden state on the observed window through a low-rank FiLM residual. Its ReZero scale starts at zero, and both forecasting and AutoCon consume the same post-mixer state. It adds no Mamba scan and no dense channel-by-channel matrix.

The experiment used Electricity H96, 321 channels, sequence length 96, three fresh seeds (`2271--2273`), five external baselines, and the exact 2x2 factorial of mixer off/on and relation lambda 0/.001. Every job used `skip_test=true`; the source snapshot was identical across all 27 jobs.

## Result

| Condition | Mean best validation MSE |
| --- | ---: |
| Mixer off, relation null | 0.157653 |
| Mixer off, relation active | 0.157668 |
| Mixer on, relation null | 0.155525 |
| Mixer on, relation active | 0.155544 |
| Strongest external baseline, TimesNet | 0.169221 |

The primary candidate improves the exact mixer-off active control by `1.347%` and wins `2/3` paired seeds. It is `8.082%` below TimesNet. The mixer effect is also present under the relation null (`1.349%`), so the attributable result is the window-conditioned channel-state mixer rather than the relation objective.

The active relation objective is non-inferior to its exact mixer-on null (`+0.012%`, `0/3` paired wins, within the registered 1% margin), but it does not improve accuracy. The paper must not claim an independent AutoCon relation gain from this screen.

## Resource Boundary

The rank-4 mixer adds 453 trainable parameters in the Electricity shape. Exact-shape synthetic prediction-only median latency was `0.883 ms` off and `0.860 ms` on. This is a no-obvious-regression preflight only, not a speed-superiority result; the candidate remains slower than simple linear baselines in the wider resource matrix.

## Decision

Keep this architecture as the next confirmation candidate and register it on at least one independent dataset under the same factorial and external-baseline contract. Do not release held-out test metrics, tune the rank or lambda, or claim AutoCon relation attribution before that confirmation.

Plan and authorization: [`plan`](mamba_acf_channel_mixer_electricity_h96_20260927.json), [`launch sidecar`](mamba_acf_channel_mixer_electricity_h96_20260927_launch_sidecar.json). Strict machine readout: [`readout`](mamba_acf_channel_mixer_electricity_h96_20260927_readout/readout.md). Training artifacts: `../../artifacts/exploratory/mamba_acf_channel_mixer_electricity_h96_20260927/`.
