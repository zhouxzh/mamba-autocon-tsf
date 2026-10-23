# Frozen Mamba-1 ACF Test Release

Status: completed read-only chronological test release. Candidate and official
PatchTST checkpoints were SHA-256 locked in a manifest before any release
reader ran. The test readers did not train or select a model.

## Main Result

| Target | Labels | Active test MSE | Official PatchTST test MSE | Active vs PatchTST | Wins | One-SE ratio upper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192 | 25% | 2.289210 | 2.327682 | -1.653% | 3/3 | 0.98849 |
| METR-LA H192 | 25% | 1.940196 | 2.059119 | -5.775% | 3/3 | 0.94365 |
| Weather H192 | 10% | 1.027531 | 1.059570 | -3.024% | 3/3 | 0.97537 |

The candidate is a one-scan Mamba-1 shared patch encoder with 24/8 patches,
target-frozen transferred core, source next-patch pretraining, and train-only
ACF forecast-contrastive positives/negatives. The official baseline is
PatchTST commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9`, source-hash locked
before the test readers ran.

## Post-release iTransformer Extension

A second external baseline was trained under the same validation split,
label budget, seed list, model width, epoch count, and target-window budget
after the candidate release was frozen. The pinned official iTransformer commit
is `c2426e68ca13f74aaec08045c5c724d8ad328124`; four source files and every
checkpoint were hash checked before its read-only test evaluation.

| Target | Active vs official iTransformer | Active wins | One-SE ratio upper |
| --- | ---: | ---: | ---: |
| PEMS-BAY | -7.588% | 3/3 | 0.92884 |
| METR-LA | -9.575% | 3/3 | 0.90794 |
| Weather | -11.286% | 3/3 | 0.89473 |

This is a post-release baseline extension: it strengthens external comparison
coverage but did not affect candidate architecture, source selection, or the
primary test-release selection.

## Post-release TimeMixer Extension

Official TimeMixer commit `e24610583b36fdd8c76cc17a8df4e65759a5f460` was
added under the same frozen post-release protocol. TimeMixer, its decomposition
layer, embedding layer, and normalization layer were hash locked before each
read-only test reader ran.

| Target | Active vs official TimeMixer | Active wins | One-SE ratio upper |
| --- | ---: | ---: | ---: |
| PEMS-BAY | -3.184% | 3/3 | 0.97077 |
| METR-LA | -4.169% | 3/3 | 0.96433 |
| Weather | -2.127% | 2/3 | 0.99425 |

This extension follows the same rule as iTransformer: it strengthens the
external comparison record but is not part of the original candidate-selection
matrix.

## Causal Control

The paired null checkpoints were read after the primary frozen active/PatchTST
release solely to quantify the pre-registered active/null contrast. They did
not affect model selection.

| Target | Active vs paired null | Active wins | One-SE ratio upper |
| --- | ---: | ---: | ---: |
| PEMS-BAY | -2.449% | 3/3 | 0.97896 |
| METR-LA | -0.430% | 3/3 | 0.99612 |
| Weather | -0.397% | 3/3 | 0.99635 |

## PEMS-BAY 10% Extension

A separately frozen PEMS-BAY H192 10%-label active/PatchTST release recorded
active MSE `2.655247` versus official PatchTST `2.675938` (`-0.773%`, 3/3
wins, one-SE ratio upper `0.99547`). Candidate and PatchTST checkpoint hashes
were recorded before the reader ran. This release has the same
historical-public-test-access caveat as the primary release.

A frozen METR-LA H192 10%-label release used a fresh 10%-prefix ACF source
selection (PEMS-BAY, PEMS07, PEMS08). It recorded active MSE `2.626573`
versus official PatchTST `2.740553` (`-4.159%`, 3/3 wins, one-SE upper
`0.96296`). This read-only release carries the same historical-access caveat.

## Robustness And Runtime

The independent validation replication contains 18 stress cells: three
targets times later origins, 10%/30% channel masking, and 0.05/0.10 relative
noise. Every cell has 3/3 active wins and a one-standard-error ratio upper
bound below 1.0. Inference uses one Mamba block invocation per forecast and
constructs neither AutoCon loss nor an AutoCon representation.

A later hash-checked official PatchTST validation extension applies the same
six stresses to the locked 10%-label checkpoints. PEMS-BAY and METR-LA pass
the one-SE noninferiority rule in all conditions; METR-LA wins every seed in
every cell. The initial Weather three-seed extension had marginal upper-bound
misses, but a separately registered five-seed Weather replication passes all
six conditions. This remains validation-only robustness evidence, not a new
test release. See `mamba1_external_robustness_patchtst_20261006.md` and
`mamba1_weather_independent_external_robustness_20261006.md`.

The independently registered METR-LA H192/10% validation replication now also
has five fresh candidate/PatchTST seed pairs. Active MSE is `1.591615` versus
its paired null `1.600515` (`-0.556%`, 5/5 wins) and official PatchTST
`1.659457` (`-4.088%`, 5/5 wins; one-SE ratio upper `0.96354`). This supports
the low-label validation record only; it neither reopens nor changes the
frozen test release.

Batch-four device-resident median latencies are PEMS-BAY 0.333 ms, METR-LA
0.315 ms, and Weather 0.314 ms. These numbers demonstrate a clean one-scan
path; they do not establish a universal external-baseline latency advantage.
On the same PEMS-BAY-shaped batch-four microbenchmark, Mamba-1 was 0.331 ms,
official PatchTST 0.255 ms, and official iTransformer 0.237 ms. The candidate
is therefore slower than both named external baselines, but faster than the
project's Mamba-2 accuracy arm (0.766 ms) and uses no AutoCon loss or extra
Mamba scan.

## Required Limits

This release carries a historical-public-test-access caveat because this
repository contains prior public test artifacts. It also compares externally
to PatchTST in the primary release, with iTransformer added only afterward as
a locked post-release extension. Therefore these results are strong evidence
for the stated method but are not by themselves a complete SCI Q1 submission
package. Electricity and Traffic remain explicit
high-dimensional failure domains. Under the same optimized Mamba-1 24/8,
frozen-core protocol and similarity-selected single source, Electricity with
Traffic source was worse than its exact null by 1.221% (8.24315 vs 8.14374)
and Traffic with Electricity source was worse by 0.161% (0.81774 vs 0.81643).
An Electricity target-prefix calibration follow-up also remained worse than
its exact null by 0.579% (8.17911 vs 8.13206). Neither high-dimensional probe
or calibration follow-up opened test data. Shared rank-8 channel and rank-32
full-patch target MLP adapters likewise failed their matched nulls by 0.763%
and 0.103%, respectively. Direct channelwise representation-to-forecast fusion
also failed its null by 0.111%.
The valid transferred-and-frozen channel-count-agnostic mean/std summary
context also failed its null by 0.070% (8.34717 vs 8.34137).

Audits:

- `artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006/audit.json`
- `artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006/paired_null_readout/audit.json`
- `artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006/official_itransformer_extension/audit.json`
- `artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006/official_timemixer_extension/audit.json`
- `artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006/inference_comparison_pems_bay/audit.json`
- `artifacts/paper_final/mambatsv3_mamba1_pems10_frozen_test_release_20261006/audit.json`
- `artifacts/paper_final/mambatsv3_mamba1_metr10_frozen_test_release_20261006/audit.json`

The test release and paired-null readout are separate because the primary
active/PatchTST checkpoints were frozen and recorded before any release reader
ran. The null controls were already paired checkpoints, but their test reading
occurred after the primary release and is reported only as a causal-control
extension, never as a post-test model-selection signal.
