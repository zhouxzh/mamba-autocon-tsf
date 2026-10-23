# Core Evidence: ACF-Contrastive MambaTSv3 Pretraining

## Claim Under Test

The method uses a channel-count agnostic shared patch encoder with one MambaTSv3
scan. Source pretraining predicts a future residual and applies signed
forecast-level InfoNCE to train-only ACF-peak window pairs. At target time, an
exact horizon-matched head initializes a one-scan Mamba residual correction on
top of a selected linear primary. ACF pairing and contrastive loss are absent
from inference.

The positive claim is deliberately limited to ACF-compatible, periodic,
low-data traffic forecasting. It is not a universal claim over all datasets.
The current train-prefix ACF source-ranking heuristic is not claimed as causal:
an H192 source-pool ablation favors a heterogeneous counterfactual pool by
`2.032%` (0/3 for the ACF top-3 pool).

## Main Held-Out Results

Each main result is three paired seeds, 10% target data, five target epochs,
64 training windows, best validation checkpoint selected before test reading,
and GPU guard completion. The NLinear, PatchTST, and iTransformer baselines
are part of each frozen release.

| Target | Horizon | Active MSE | NLinear | PatchTST | iTransformer | Active vs NLinear | Active vs PatchTST | Active vs iTransformer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY | 192 | 2.621235 | 2.918066 | 2.713254 | 2.863448 | -10.172% | -3.391% | -8.459% |
| PEMS-BAY | 336 | 2.275405 | 2.485398 | 2.355005 | 2.578049 | -8.449% | -3.380% | -11.739% |
| METR-LA | 192 | 2.617203 | 2.732207 | 2.773458 | 2.865344 | -4.209% | -5.634% | -8.660% |
| Weather | 192 | 1.027734 | 1.061585 | 1.085683 | 1.153800 | -3.189% | -5.338% | -10.926% |

Every table comparison is 3/3 paired wins. One-standard-error upper ratios
against PatchTST are `.9690`, `.9695`, and `.9475`, respectively.

METR-LA also includes TimesNet in its frozen release: active `2.617203` versus
TimesNet `2.899513` (`-9.736%`, 3/3). PEMS-BAY TimesNet/TimeMixer were added
after its candidate test release, so they are descriptive post-release
extensions: active is lower than TimesNet by `7.613%/10.509%` and TimeMixer by
`4.145%/5.463%` at H192/H336, all 3/3.

RLinear was likewise added as a post-release descriptive extension. Active is
lower by `13.461%/12.593%` at H192/H336, all 3/3. Its fixed-shape latency is
`.065/.067 ms`, however, versus active `.384/.402 ms`; the candidate must not
claim speed superiority over robust linear baselines.

An external-code PatchTST extension imports the official Apache-2.0 network
from `yuqinie98/PatchTST` commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9`
through a local chronological split adapter. On PEMS-BAY H192, active is
`2.621235` versus official PatchTST `2.652867` (`-1.192%`, 3/3; one-SE upper
ratio `.9903`). This is a post-release external-model-code extension, not a
member of the original frozen primary matrix.

The same pinned official model and adapter at PEMS-BAY H336 gives active
`2.275405` versus official PatchTST `2.314149` (`-1.674%`, 3/3; one-SE upper
ratio `.9892`). Both external-code coordinates retain their post-release
status; they corroborate the local PatchTST comparison but do not alter the
original release selection rule.

On the independent METR-LA H192 target, the same official PatchTST commit and
adapter give active `2.617203` versus official PatchTST `2.743069` (`-4.589%`,
3/3; one-SE upper ratio `.9559`). This confirms the official PatchTST result
across two traffic/sensor targets while retaining post-release status.

On Weather H192, active is `1.027734` versus the same official PatchTST
`1.059068` (`-2.959%`, 3/3; one-SE upper ratio `.9809`). Official PatchTST
confirmation therefore spans all three main held-out targets.

An external-code iTransformer extension imports the MIT network from
`thuml/iTransformer` commit `c2426e68ca13f74aaec08045c5c724d8ad328124` through
the same chronological adapter contract. Active is `-7.054%` at H192 and
`-11.360%` at H336, both 3/3, with one-SE upper ratios `.9325/.8922`. These
are also post-release external-model-code extensions and do not replace the
original frozen primary matrix.

On METR-LA H192, the same official iTransformer commit and adapter give active
`2.617203` versus official iTransformer `2.869208` (`-8.783%`, 3/3; one-SE
upper ratio `.9158`). Thus official PatchTST and iTransformer confirmations
cover both PEMS-BAY and METR-LA, while official TimesNet/TimeMixer confirmations
currently cover PEMS-BAY only.

An external TimesNet extension imports `models/TimesNet.py` from
`thuml/Time-Series-Library` commit `4e938a1767106324dd753b2a44832bf870a0252e`
through the same boundary adapter. Active is `-7.523%` at H192 and `-11.493%`
at H336, both 3/3, with one-SE upper ratios `.9277/.8889`. Like the other
external-code comparisons, these are post-release extensions.

An external TimeMixer extension imports `models/TimeMixer.py` from
`kwuking/TimeMixer` commit `e24610583b36fdd8c76cc17a8df4e65759a5f460` through
the same boundary adapter. Active is `-3.564%` at H192 and `-5.831%` at H336,
both 3/3, with one-SE upper ratios `.9720/.9452`. These external-code results
also remain post-release extensions.

Weather also includes TimesNet in its frozen H192 release: active `1.027734`
versus TimesNet `1.096633` (`-6.283%`, 3/3).

## Mechanism And Robustness

Source active/null controls keep source pools, source initialization, seed,
residual target, exact head, and target fine-tuning fixed. The only change is
the signed InfoNCE term. The new pair-preserving sampler yields active anchors
above 89% in the positive PEMS-BAY/METR-LA long-horizon matrices.

| Target | Horizon | Active vs exact null | Paired wins |
| --- | ---: | ---: | ---: |
| PEMS-BAY | 192 | -1.395% | 3/3 |
| PEMS-BAY | 336 | -1.956% | 3/3 |
| METR-LA | 192 | -0.540% | 3/3 |
| Weather | 192 | -0.253% | 3/3 |

For all three main coordinates, active wins every three-seed closed-validation
condition: in distribution, later time block, 10%/30% masked channels, and
0.05/0.10 relative noise. The smallest main-coordinate gain is METR-LA H192
with 30% missing channels, `-0.375%`, still 3/3 wins.

At 25% target data, the exact source checkpoints retain the effect:
PEMS-BAY H192 `-1.339%`, PEMS-BAY H336 `-1.949%`, and METR-LA H192 `-0.554%`,
all 3/3 active wins.

A full-target-prefix PEMS-BAY H192 validation reference uses 100% of the
declared target training prefix, with the shared fixed-budget sampler expanded
to 128 windows. It gives active `1.427664` versus NLinear `1.642596`, PatchTST
`1.516147`, iTransformer `1.548894`, and TimesNet `1.496053`, corresponding to
`-13.085%/-5.836%/-7.827%/-4.571%`, all 3/3. This is a validation reference,
not a claim of convergence under exhaustive full-prefix optimization.

Weather H192 is 3/3 positive in every closed stress condition, but its 25%
active/null result is only `-0.108%` with 1/3 wins. It supports held-out
accuracy and low-data robustness, but not a stable 25% mechanism claim.

## Inference Boundary

Batch-8, device-resident, FP32, 25 warm-up and 100 timed forward passes on an
RTX 5090 D give:

| Horizon | Active median ms | TimesNet ms | TimeMixer ms | Active Mamba scans | Relation loss |
| ---: | ---: | ---: | ---: | ---: | --- |
| 192 | .384 | .665 | .516 | 1 | excluded |
| 336 | .402 | .649 | .520 | 1 | excluded |

This is fixed-shape prediction latency only. It excludes data loading,
host-device transfer, backward propagation, and optimizer work.

RLinear remains materially faster at `.065/.067 ms` for H192/H336. The valid
efficiency claim is therefore lower latency than the measured nonlinear
TimesNet/TimeMixer baselines while retaining one Mamba scan, not universal
inference-speed superiority.

## Negative Boundaries

- The prior forecast-affine ACF loss fails active/null and is stopped.
- Electricity H96 signed InfoNCE is `+0.316%`, 0/3 active wins; it is excluded
  from positive claims. An explicit round-robin lag sampler increased active
  anchor coverage to 51%--57% but worsened active/null to `+0.548%`, 0/3;
  pair coverage alone does not explain the negative transfer.
- PEMS-BAY H192 source ranking: ACF top-3 is `+2.032%` versus a fixed
  heterogeneous counterfactual pool, 0/3; source selection is not a claimed
  causal mechanism.
- Traffic H96 is small at 10% (`-0.127%`, 2/3) and turns negative at 25%
  (`+0.216%`, 0/3), so it is supporting low-data evidence only.
- Weather H192 is a held-out positive coordinate, but its 25% mechanism
  comparison is unstable (`-0.108%`, 1/3), so it cannot support the central
  data-efficiency claim.

## Submission Readiness

This evidence supports a focused journal paper on ACF-compatible low-data
traffic forecasting with causal active/null evidence, held-out accuracy,
robustness, and clean one-scan inference. It is not yet enough to claim a
general-purpose foundation forecasting model or universal performance. A Q1
submission still needs an external implementation or published-score protocol
for all strong baselines, a public reproducibility bundle, and an explicit
scope statement that retains the Electricity/Traffic negative boundaries.

## Audits

- PEMS-BAY H192/H336: `artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003/audit.json`
- METR-LA H192: `artifacts/paper_final/mambatsv3_acf_infonce_metr_la_h192_confirmatory_20261003/audit.json`
- Weather H192: `artifacts/paper_final/mambatsv3_acf_infonce_weather_h192_confirmatory_20261003/audit.json`
- Official PatchTST H192 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_h192_extension_20261003/audit.json`
- Official PatchTST H336 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_h336_extension_20261003/audit.json`
- Official PatchTST METR-LA H192 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_metr_la_h192_extension_20261003/audit.json`
- Official PatchTST Weather H192 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_weather_h192_extension_20261003/audit.json`
- Official iTransformer H192 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_h192_extension_20261003/audit.json`
- Official iTransformer H336 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_h336_extension_20261003/audit.json`
- Official iTransformer METR-LA H192 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_metr_la_h192_extension_20261003/audit.json`
- Official TimesNet H192 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_timesnet_h192_extension_20261003/audit.json`
- Official TimesNet H336 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_timesnet_h336_extension_20261003/audit.json`
- Official TimeMixer H192 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_timemixer_h192_extension_20261003/audit.json`
- Official TimeMixer H336 extension: `artifacts/paper_final/mambatsv3_acf_infonce_official_timemixer_h336_extension_20261003/audit.json`
- Mechanism history: `mambatsv3_acf_contrastive_repair_screen_20261003.md`
