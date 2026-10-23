# Submission Readiness Audit: 2026-10-03

## Objective Audit

| Original requirement | Current evidence | Status |
| --- | --- | --- |
| Freeze a fastest stable one-scan MambaTSv3 backbone | Production MambaTSv3 remains one scan; deployed checkpoints report `mamba_scan_count=1`. | Met |
| Channel-count agnostic shared patch encoder | Shared core transfers across PEMS, METR-LA, PEMS-BAY, Weather, Electricity, and Traffic shapes; channel-specific parameters are bypassed during source pretraining. | Met in implementation |
| Train-only ACF positive/negative pretraining | Signed forecast-level InfoNCE uses train-prefix ACF lag pairs and in-batch negatives; exact active/null controls are positive on three main long-horizon coordinates. | Met for stated scope |
| Low-data 10%/25% evidence | PEMS-BAY H192/H336 and METR-LA H192 retain 25% active/null gains. Weather H192 25% is only 1/3; Traffic and Electricity are negative boundaries. | Partially met, scoped |
| Shift/missing/noise robustness | All six closed validation conditions are 3/3 positive on PEMS-BAY H192/H336, METR-LA H192, and Weather H192. | Met for main coordinates |
| Clean inference | Prediction-only benchmarks omit relation loss and pairing. Candidate is faster than TimesNet/TimeMixer but slower than RLinear. | Met with speed boundary |

## Audited Held-Out Coordinates

| Target | Horizon | NLinear | PatchTST | iTransformer | TimesNet | Mechanism active/null |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY | 192 | -10.172% | -3.391% | -8.459% | post-release -7.613% | -1.395%, 3/3 |
| PEMS-BAY | 336 | -8.449% | -3.380% | -11.739% | post-release -10.509% | -1.956%, 3/3 |
| METR-LA | 192 | -4.209% | -5.634% | -8.660% | -9.736% | -0.540%, 3/3 |
| Weather | 192 | -3.189% | -5.338% | -10.926% | -6.283% | -0.253%, 3/3 |

All frozen-release comparisons in this table are three paired seeds. The
PEMS-BAY TimesNet, TimeMixer, and RLinear extensions were trained after the
candidate test release and must remain descriptive, not be folded into the
original frozen significance statement.

## Honest Q1 Position

The work now has a credible Q1-journal research story only under a narrow
claim: train-only ACF contrastive pretraining improves low-data long-horizon
forecasting in ACF-compatible periodic traffic/sensor targets, and it does so
without inference-time contrastive work or a second Mamba scan.

It is not ready to claim a universal time-series foundation model. Electricity
does not transfer, Traffic is unstable at 25%, and Weather has unstable 25%
mechanism evidence. The paper must report these boundaries, not omit them.

The current local-cluster evidence additionally gives a train-prefix
compatibility hypothesis: five target-level points have a descriptive
compatibility/gain correlation of `r=-0.765`. This is not a submission claim
until it is registered and tested on new domains or chronological slices.

Before submitting to a Q1 journal, the remaining non-negotiable work is:

1. Publish a minimal reproducibility package: environments, data acquisition
   instructions, source-selection manifests, frozen commands, and release
   audit readers.
2. State the local baseline implementation status and either reproduce against
   official external implementations or provide a documented published-score
   comparison protocol.
3. Build the manuscript around the pre-registered-like audited release tables,
   not post-release baseline extensions or exploratory screens.
4. Complete accountable author evidence review, venue policy review, and all
   declarations before final submission. The separate-machine rerun requirement
   is now met by the independent PEMS-BAY H192 replication recorded below.

The bundle at
`artifacts/paper_final/mambatsv3_acf_infonce_core_bundle_20261003/` is the
starting point for that reproducibility package.

The current ACF source-ranking heuristic has a recorded negative sensitivity
(`+2.032%`, 0/3 against a heterogeneous counterfactual pool). It must be
reported as a heuristic boundary, not presented as a validated source-domain
selection contribution.

## 2026-10-08 Maintenance Update

The three primary held-out audit readers were rerun unchanged after later
exploratory work and each retained `audit_valid=true`. Rebuilding the core
bundle again bound the same three releases, retained all five pre-existing
negative boundaries, and kept the full-prefix reference test-free. This is a
read-only integrity check, not a new result.

Two subsequent validation-only short-horizon mechanism probes also sharpened
the scope rather than changing the primary release:

| Probe | Coordinate | Result | Decision |
| --- | --- | ---: | --- |
| Future-residual affine coefficient table | Traffic H96, seed 6120 | `0.795736` vs raw ACF `0.795702` (`+0.004%`) | Do not replace the frozen long-horizon signed InfoNCE coefficients with this statistic |
| Future-residual signed InfoNCE coefficient table | Weather H192, seed 6121 | `4.516603` vs raw-peak sign `4.515470` (`+0.025%`) | Do not extend this statistic to the evidence-bearing InfoNCE recipe |

Both probes use only chronological validation data and retain
`test_metrics_produced=false`. They found substantial raw-signal versus
future-residual ACF disagreement for METR-LA but no transfer improvement after
correcting it. These negative results reinforce the existing claim boundary:
the paper must report the audited signed-InfoNCE release as an empirically
scoped protocol, not as a universal consequence of any particular ACF
coefficient estimator.

## 2026-10-08 Independent-Machine Packet

A dedicated portable packet now targets the central PEMS-BAY H192/10%
local-cluster active/null mechanism using fresh seeds `6130--6132`. It fixes
the source pool, H192 exact head, local contrastive-cluster geometry, source
and target budgets, validation-before-test gate, input hashes, and per-stage
GPU thermal guard. The local `--manifest-only` check created a packet with all
14 required source/data/checkpoint hashes and with test reading disabled.

The packet was subsequently executed on the verified distinct host
`zhong-Precision-Tower-7810` (RTX 3060 Ti, Torch `2.6.0+cu126`, Mamba `2.2.4`).
All 14 tracked code/data/checkpoint hashes matched the packet manifest. The
three fresh test seeds give active/null `2.638441/2.670296` (`-1.193%`), 3/3
wins, and a one-SE upper ratio of `.99151`; all 18 guarded stages completed
without a trigger, with maximum `63C` and `31%` fan. This is independent
support evidence, not a replacement or pooled extension of the primary local
effect estimate. See [the packet instructions](mambatsv3_local_cluster_independent_machine_packet_20261008.md) and
`artifacts/reproducibility/mambatsv3_local_cluster_independent_machine_20261008/independent_audit.json`.

The same independent host also replicated METR-LA H192/10% with its fixed
PEMS-BAY/PEMS07/PEMS08 source pool. Fresh seeds `6140--6142` give active/null
`2.622863/2.630788` (`-0.301%`), 3/3 wins and one-SE upper ratio `.99762`.
All 18 guarded stages completed without a trigger (peak `60C`, `30%` fan).
This is separate support evidence for a second target domain, not a pooled
replacement of the primary local estimate. Audit:
`artifacts/reproducibility/mambatsv3_local_cluster_metr_independent_machine_20261008/independent_audit.json`.

Weather H192/10% was also independently replicated with its fixed
METR-LA/PEMS-BAY/Electricity source pool. Fresh seeds `6150--6152` give
active/null `1.016437/1.017655` (`-0.120%`), 3/3 wins and one-SE upper ratio
`.99903`, with all 18 guarded stages completing normally (peak `60C`, `30%`
fan). This is a small but directionally consistent third-domain replication;
it remains separate from, and is not pooled into, the primary local estimates.
Audit:
`artifacts/reproducibility/mambatsv3_local_cluster_weather_independent_machine_20261008/independent_audit.json`.

The official PatchTST compatibility audit is recorded in
`mambatsv3_acf_infonce_external_baseline_protocol_20261003.md`: its unmodified
70/10/20 split and automatic train-then-test entry point cannot be compared
directly to the audited release. An isolated adapter is required before an
official-code comparison may be claimed.
