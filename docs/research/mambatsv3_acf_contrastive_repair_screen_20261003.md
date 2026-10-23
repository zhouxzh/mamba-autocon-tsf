# ACF Contrastive Repair Screen: 2026-10-03

## Why a New Objective Was Needed

The four-domain residual-transfer release is positive, but its source
forecast-affine ACF objective fails an exact active/null control: active is
`+0.282%` on METR-LA and `+1.036%` on PEMS-BAY. That affine loss is therefore
stopped as an AutoCon mechanism claim.

The audit identified a second issue in the source subsampler. It seeded
lag-separated pairs but then sorted and truncated individual window origins,
which could retain one pair member and remove the other. This made explicit
contrastive objectives sparsely active. `Windows._select_starts` now keeps
complete pairs and has dedicated regression tests.

## New Candidate

`ForecastLagPairContrastiveLoss` is a signed forecast-level InfoNCE objective.
For each channel, train-only ACF peak-separated source windows form positives;
all other in-range, non-identical windows in the batch are denominator
negatives. Negative ACF positives use sign-flipped cosine scores in both
numerator and denominator. The score is prediction-connected through the
shared Mamba encoder and transferred H96 head; it does not run at inference.

The loss is calibrated to 10% of the source predictive loss. It produces zero
gradient if a batch lacks both an ACF positive and a competing negative.

## Validation-Only Active/Null Screen

The source domains, initialization, seed, residual target, exact H96 head,
target fine-tuning, and target validation split are identical between active
and null. The only change is the source InfoNCE term. No test metrics were
produced.

| Target | Null mean | Active mean | Active vs null | Wins | Active anchor coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY | 2.599308 | 2.593343 | -0.229% | 3/3 | 94.8%--96.9% |
| METR-LA | 1.424067 | 1.422929 | -0.080% | 3/3 | 91.1%--95.8% |

This began as an early mechanism signal. The Weather replication below extends
it to a non-traffic target, but the full evidence remains validation-only and
must not be presented as a held-out result.

## Weather Replication and 25% Data

Weather uses its pre-existing few-shot-only source selection
`METR-LA/PEMS-BAY/Electricity`; its main validation and test partitions remain
closed during selection. At 10%, active is `-0.081%` relative to null (2/3
wins). At 25%, active is `-0.094%` (2/3 wins). The effect is small but remains
non-inferior in a non-traffic target.

The corresponding 25% active/null results are:

| Target | Active vs null | Wins |
| --- | ---: | ---: |
| PEMS-BAY | -0.364% | 3/3 |
| METR-LA | -0.095% | 3/3 |
| Weather | -0.094% | 2/3 |

## Closed Validation Robustness

At 10%, active/null comparisons use the same frozen checkpoints under closed
validation stresses. PEMS-BAY and METR-LA win all three paired seeds in every
condition. Weather is non-inferior in every aggregate, with its weakest
condition (30% missing channels) at `-0.034%`, 2/3 wins.

| Target | In distribution | Late shift | Missing 10% | Missing 30% | Noise 0.05 | Noise 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY | -0.229%, 3/3 | -0.215%, 3/3 | -0.206%, 3/3 | -0.158%, 3/3 | -0.231%, 3/3 | -0.233%, 3/3 |
| METR-LA | -0.080%, 3/3 | -0.066%, 3/3 | -0.071%, 3/3 | -0.054%, 3/3 | -0.080%, 3/3 | -0.081%, 3/3 |
| Weather | -0.081%, 2/3 | -0.092%, 3/3 | -0.058%, 2/3 | -0.034%, 2/3 | -0.081%, 2/3 | -0.079%, 2/3 |

## Horizon Extension

PEMS-BAY uses separately pretrained and exactly transferred heads for each
horizon. The H192 and H336 active/null matrices use three fresh paired seeds,
with test metrics disabled.

| Horizon | Null mean | Active mean | Active vs null | Wins |
| --- | ---: | ---: | ---: | ---: |
| H96 | 2.599308 | 2.593343 | -0.229% | 3/3 |
| H192 | 2.783732 | 2.744908 | -1.395% | 3/3 |
| H336 | 2.434082 | 2.386473 | -1.956% | 3/3 |

H336 prediction-only inference remains a single Mamba scan: batch-8 median
latency is `.400 ms`, with no returned AutoCon representation and no relation
loss. This is a fixed-shape forward benchmark, not end-to-end throughput.

## PEMS-BAY H192/H336 Held-Out Release

The H192/H336 validation checkpoints were frozen before the chronological
test partition was opened. Each test reader restores the validation-selected
state and emits `training_performed=false`. The release includes NLinear,
PatchTST, and iTransformer at the same low-data budget.

| Horizon | Active | NLinear | PatchTST | iTransformer | vs NLinear | vs PatchTST | vs iTransformer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H192 | 2.621235 | 2.918066 | 2.713254 | 2.863448 | -10.172% | -3.391% | -8.459% |
| H336 | 2.275405 | 2.485398 | 2.355005 | 2.578049 | -8.449% | -3.380% | -11.739% |

All six paired comparisons beat each baseline. The one-standard-error upper
ratios against PatchTST are `.9690/.9695`; they remain below the 1.02
non-inferiority boundary. Closed validation stress also gives 3/3 active
wins in every H192/H336 shift, missing-channel, and noise condition. The
strict artifact is
`artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003/audit.json`.

The same frozen source checkpoints continue to improve at 25% target data on
closed validation: H192 is `-1.339%` and H336 is `-1.949%`, both 3/3 active
wins against their exact null. Thus the long-horizon signal is not confined to
the 10% setting, although no full-data claim has been tested.

## Traffic Boundary

Traffic uses its few-shot-only selected sources Electricity/METR-LA/PEMS-BAY
and DLinear primary. At 10%, H96 active/null is `-0.127%` (2/3 wins), with
the late-block shift condition `-0.144%` (3/3). Other closed stresses remain
non-inferior but only 2/3 wins. At 25%, active is `+0.216%` (0/3). Traffic is
therefore supporting low-data evidence only and is not a valid long-horizon
release target under the current recipe.

## Electricity Boundary

Electricity provides a necessary negative transfer check. Its few-shot-only
source selector chose Traffic/METR-LA/Weather and its inner calibration chose
DLinear. The same H96 active/null protocol yields `+0.316%` (0/3 active wins),
with only 39%--49% active relation anchors. Electricity is excluded from the
positive claim and establishes that the current source-selection plus InfoNCE
recipe is not universally effective.

## METR-LA H192 Independent Held-Out Release

METR-LA is the second independent long-prediction target. Its source selector
chooses PEMS-BAY/PEMS07/PEMS08 from only the target few-shot prefix. H192
active/null validation is `-0.540%`, 3/3 wins, and all six closed validation
stresses are 3/3 positive. The frozen held-out release gives:

| Active | NLinear | PatchTST | iTransformer | TimesNet | vs NLinear | vs PatchTST | vs iTransformer | vs TimesNet |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.617203 | 2.732207 | 2.773458 | 2.865344 | 2.899513 | -4.209% | -5.634% | -8.660% | -9.736% |

Every comparison is 3/3 active wins. The one-standard-error upper ratios are
`.9607/.9475/.9164/.9057`, all inside a 1.02 non-inferiority limit. At 25%
target data, H192 remains `-0.554%`, 3/3 active wins versus null. The strict
audit is
`artifacts/paper_final/mambatsv3_acf_infonce_metr_la_h192_confirmatory_20261003/audit.json`.

## TimesNet Extension And Speed

TimesNet was added after the PEMS-BAY H192/H336 frozen candidate release, so
its PEMS-BAY test comparison is explicitly descriptive rather than part of
the original confirmatory audit. It is nevertheless 3/3 worse than active at
both H192/H336 by `+7.613%/+10.509%`. Same-shape device-resident forward
benchmarks at batch 8 show active H192/H336 `.384/.402 ms` versus TimesNet
`.665/.649 ms`, a `42.3%/38.1%` lower latency. These timings exclude data,
relation loss, and host transfer; candidate records keep `mamba_scan_count=1`.

## Invalidated Screen

`artifacts/exploratory/mambatsv3_acf_infonce_screen_20261003/` was produced
before correcting signed-score normalization. It is implementation-invalid and
must not be used. The v2 screen corrected score normalization but preceded the
pair-preservation fix and likewise is not a mechanism result. Only the v3
PEMS-BAY screen and the METR-LA screen are valid for the table above.

## Artifacts

- PEMS-BAY: `artifacts/exploratory/mambatsv3_acf_infonce_screen_v3_pairs_20261003/`
- METR-LA: `artifacts/exploratory/mambatsv3_acf_infonce_metr_la_v1_20261003/`
- Weather: `artifacts/exploratory/mambatsv3_acf_infonce_weather_v1_20261003/`
- 25% validation: `artifacts/exploratory/mambatsv3_acf_infonce_25_validation_20261003/`
- Stress validation: `artifacts/exploratory/mambatsv3_acf_infonce_stress_validation_20261003/`
- H192/H336: `artifacts/exploratory/mambatsv3_acf_infonce_h192_screen_20261003/`, `artifacts/exploratory/mambatsv3_acf_infonce_h336_screen_20261003/`
- H192/H336 held-out audit: `artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003/`
- Electricity negative boundary: `artifacts/exploratory/mambatsv3_acf_infonce_electricity_v1_20261003/`
- Traffic boundary: `artifacts/exploratory/mambatsv3_acf_infonce_traffic_v1_20261003/`, `artifacts/exploratory/mambatsv3_acf_infonce_traffic_25_validation_20261003/`
- METR-LA H192 held-out audit: `artifacts/paper_final/mambatsv3_acf_infonce_metr_la_h192_confirmatory_20261003/`
- PEMS-BAY resource extension: `artifacts/paper_final/mambatsv3_acf_infonce_horizon_resource_20261003/`
- Implementation: `layers/losses.py`, `scripts/pretrain_mambatsv3_core_autocon.py`
