# Cross-Domain Mamba-2 AutoCon Pretraining Readout

Status: exploratory validation evidence only. No test partition was opened by
any artifact in this document.

## Question

Can a channel-count-agnostic shared patch encoder, pretrained with train-only
ACF positive/negative relation pairs and then frozen, provide low-label gains
without placing AutoCon loss or an additional Mamba scan in inference?

The common candidate is MambaTSv3 with one Mamba-2 encoder block, `d_model=24`,
two state-space heads (`headdim=12`), 24/8 patches, and a DLinear/NLinear
residual forecasting head. Source pretraining uses next-patch prediction plus
a 10% scaled ACF forecast-contrastive loss. Target adaptation freezes the
transferred patch encoder, position tokens, Mamba block, final norm, and
representation head; only target-specific forecasting parameters train.

## Validation Matrix

| Target | Labels | Source policy | Active vs null | Active/null wins | Active vs official PatchTST | Active/PatchTST wins | One-SE ratio upper |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY H192 | 25% | fixed PEMS07/08/04 pool | -3.251% | 3/3 | -1.067% | 2/3 | 0.99973 |
| METR-LA H192 | 25% | target-few-shot ACF top-3: PEMS-BAY/Weather/PEMS08 | -0.681% | 3/3 | -5.619% | 3/3 | 0.94785 |
| Weather H192 | 10% | fixed METR-LA/PEMS-BAY/Electricity pool | -0.478% | 3/3 | -2.940% | 3/3 | 0.97826 |

The official comparator is PatchTST commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9`, retrained under each target's same
chronological split, target-label budget, five-epoch/64-window budget, and
24/8 patch geometry. All comparison artifacts contain `test_metrics_produced:
false`.

## Robustness

PEMS-BAY and source-selected METR-LA both retain three-seed active/null gains
under later validation origins, fixed 10%/30% channel masking, and relative
Gaussian noise 0.05/0.10. The METR-LA range is -0.47% to -0.69%; PEMS-BAY's
range is -2.32% to -3.28%. In each audited stress condition, active wins 3/3
and the one-standard-error ratio upper bound is below 1.

The channel-mask operation follows the existing normalized-input stress
contract. Absolute MSE across different stresses is not compared; only the
matched active/null contrast within a condition is interpreted.

## Inference Contract And Boundary

Runtime hooks for the PEMS-BAY candidate observed exactly 250 Mamba encoder
block invocations for 250 prediction calls. It returned no AutoCon
representation and constructed no relation loss. AutoCon is therefore a
pretraining-only mechanism at inference, and the deployed model contains one
Mamba scan.

This does not establish an unconditional speed advantage. On the named
PEMS-BAY batch-four CUDA benchmark, the Mamba-2 candidate's median prediction
latency was 0.766 ms, slower than same-geometry official PatchTST at 0.254 ms.
The Mamba-1 candidate was faster at 0.319-0.331 ms but did not produce the
same cross-domain accuracy evidence. A paper must present this as an explicit
accuracy-efficiency tradeoff, not as a universal speed claim.

## Required Next Gate

The architecture, `headdim`, patch geometry, source-selection rule, source
candidate pool, target domains, seed lists, baselines, robustness conditions,
and inference metric must be frozen before any independent validation
replication. The existing results are adaptive and exploratory; no test metric
may select or promote this candidate.
