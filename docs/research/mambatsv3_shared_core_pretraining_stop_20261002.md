# MambaTSv3 Shared-Core AutoCon Pretraining Stop Record

## Question and invariant

This study tested whether a train-only self-supervised AutoCon stage can turn
the existing one-Mamba-block MambaTSv3 core into a transferable low-data
encoder. Traffic was excluded from eight source domains (PEMS03/04/07/08,
METR-LA, PEMS-BAY, Electricity, Weather). Source data were restricted to each
chronological 60% training prefix; Traffic adaptation used 10% of its training
prefix and its existing validation block. No target test row, metric, or
prediction was opened.

All source stages reused only the transferable patch embedding, patch position,
Mamba encoder, final normalization and AutoCon representation parameters.
Target channel embedding/context and forecasting head were new. All jobs used
the same `d_model=24`, `d_ff=48`, one Mamba-1 block, `L=96`, non-overlapping
24-point patches, four CPU threads and a `78C`/`85%` GPU guard.

## Results

| Hypothesis | Target protocol | Traffic validation MSE | Change versus matched initial core | Decision |
| --- | --- | ---: | ---: | --- |
| Initial four-PEMS averaged core | Core frozen for all five target epochs | 0.930572 | reference | baseline |
| Eight-source masked reconstruction | Core frozen for all five target epochs | 0.943316 | +1.369% | fail |
| Eight-source reconstruction + AutoCon | Core frozen for all five target epochs | 0.938982 | +0.904% | fail |
| Initial four-PEMS averaged core | One frozen epoch, four core-calibration epochs | 0.910321 | reference | baseline |
| Eight-source reconstruction + AutoCon | One frozen epoch, four core-calibration epochs | 0.913417 | +0.340% | fail |
| Eight-source pure AutoCon, channel-major v3 scan | One frozen epoch, four core-calibration epochs | 0.914932 | +0.506% | fail |
| Channel-agnostic initial core | One shared Mamba block per channel patch sequence | 0.907492 | reference | baseline |
| Eight-source pure AutoCon, channel-agnostic scan | Same channel-agnostic target protocol | 0.911449 | +0.436% | fail |

The relation loss was not inactive. In eight-source pretraining, selected-lag
pair batches yielded `74.2%--77.7%` active global contrastive anchors and a
finite mean AutoCon loss near `2.03--2.24`. Masked reconstruction decreased
from `1.126` to `0.799` in its control. Thus these are performance failures,
not a data-leakage, empty-loss, or dead-gradient explanation.

## Structural result

The new `mamba_ts_channel_agnostic_scan` path is channel permutation
equivariant, bypasses channel-count-specific embedding and context, and uses
the same one Mamba block batched over channel patch sequences. It is therefore
an honest channel-count-agnostic encoder implementation. The plain initial
core improved from `.910321` to `.907492` under this topology, but source ACF
self-supervision still worsened it to `.911449`.

## Stop decision

All four tested variants fail the Traffic development promotion rule. Weather
is therefore not opened as an external gate, and no 25% target, robustness,
latency, baseline, or held-out matrix is justified for this candidate. The
current AutoCon relation objective must not be claimed as a transferable
pretraining mechanism or a source of the earlier multi-source core-transfer
gain. The existing narrow held-out MambaTSv3 results remain unchanged.

Artifacts:

- `artifacts/exploratory/mambatsv3_shared_core_pretraining_traffic_20261002`
- `artifacts/exploratory/mambatsv3_channel_agnostic_pretraining_traffic_20261002`
- `artifacts/exploratory/arbor_mambatsv3_shared_core_pretraining_20261002/.arbor/tree.json`

## 2026-10-03 Predictive follow-up: stop

The original next-patch artifacts under the earlier lightweight encoder are
not evidence for the production MambaTSv3 core. A first implementation of the
new production-core predictive branch also incorrectly multiplied its
next-patch loss by the masked-reconstruction weight when that weight was zero.
Those initial production-core predictive artifacts are audit-only and are not
included below. The implementation now has an independent
`--predictive-weight` and a regression test; all results in this section used
an active Smooth L1 next-24-patch loss plus a train-only ACF objective.

The target remained Traffic 10%, source pool remained the eight
leave-Traffic-out domains, and target tests remained closed. Each treatment
was compared only with its matched channel-agnostic initial core using the
same seed and target protocol.

| Source objective / relation view | Initial core MSE | Pretrained MSE | Change | Decision |
| --- | ---: | ---: | ---: | --- |
| Next patch + final-state, channel-mean ACF | .895390 | .903894 | +.949% | fail |
| Next patch + per-channel ACF values, source-wide positive lags | .904790 | .912186 | +.817% | fail |
| Next patch + channel-specific ACF peak support and physical-origin pairs | .900674 | .904734 | +.451% | fail |

The last branch corrected two potential coverage defects: sparse window sets
stored original training origins, and the pair sampler resolved lags in those
physical units; each channel used only its own train-prefix statistically
selected ACF peaks. Its active channel-window relation fraction was sparse
(`.12%--.48%`) because high-dimensional sensors have heterogeneous periods,
but the loss was normalized over active channel anchors and remained finite.
Despite true next-patch optimization, final-state prediction connection,
per-channel signed ACF, and physical lag pairing, all three variants worsened
the matched low-data target core. This closes the current ACF-pair pretraining
family: do not launch the Weather gate, 25% adaptation, shift/noise/missing
sensor matrix, latency matrix, or held-out release for it.

Artifacts:

- `artifacts/exploratory/mambatsv3_predictive_pretraining_corrected_traffic_20261003`
- `artifacts/exploratory/mambatsv3_channelwise_acf_predictive_traffic_20261003`
- `artifacts/exploratory/mambatsv3_channelpeak_origin_predictive_traffic_20261003`
- `artifacts/exploratory/arbor_mambatsv3_predictive_pretraining_20261003/.arbor/tree.json`
