# AutoCon Mamba Cross-Domain Pretraining Protocol

_Registered design for a new pretraining research line. This document contains no accuracy result and authorizes no large training run._

---

## Research question

Can a channel-count agnostic, single-scan Mamba patch encoder pretrained with train-only AutoCon relations retain or improve forecasting quality when only 10% or 25% of each target domain's training prefix is available, while preserving prediction-only inference cost?

## Architecture

`MambaTSPretrain` uses a shared patch projection, one temporal Mamba-1 scan, and a channel mean before the scan. Parameter shapes therefore do not depend on sensor count. Dataset-local batches retain their original channel dimensions; no cross-dataset sensor alignment is assumed.

During pretraining, masked-patch reconstruction prevents trivial representation collapse and train-only signed ACF relations define AutoCon positive/negative structure. During fine-tuning and inference, reconstruction and AutoCon losses are disabled; prediction uses the same one-scan encoder and a shared channel-statistic forecast head.

The predictive alternative uses 72 observed points to forecast the next 24-point patch. Its cross-domain source loss is Smooth L1 on the context-normalized future patch, so an isolated high-variance source window cannot dominate the shared encoder update.

## Data partitioning

| Role | Datasets | Permitted rows |
| --- | --- | --- |
| Pretraining source pool | PEMS03/04/07/08, METR-LA, PEMS-BAY, Electricity, Traffic, Weather | Each source dataset's chronological train prefix only |
| Target adaptation | Held-out target domain selected before launch | 10% or 25% of that target's original chronological train prefix |
| Validation | Existing validation partition | Model selection only |
| Test | Existing untouched test partition | Read only after a separately authorized confirmatory protocol |

No source prior may use target validation or test rows. A target cannot appear in a source pretraining pool for a leave-one-domain-out transfer cell.

## Controls

Every target/fraction/seed cell must compare:

1. Randomly initialized one-scan shared encoder, forecast fine-tuning only.
2. Reconstruction-only pretraining, identical source pool and model.
3. AutoCon-plus-reconstruction pretraining, identical source pool and model.
4. Target-only AutoCon training control.
5. Strong external baselines selected before launch: DLinear, PatchTST, iTransformer, TimesNet, TimeMixer, and Mamba2 where supported.

The primary mechanism contrast is AutoCon-plus-reconstruction versus reconstruction-only pretraining. It isolates relation learning from generic masked reconstruction.

## Robustness matrix

| Stressor | Registered setting | Primary outcome |
| --- | --- | --- |
| Low data | 10%, 25% target train prefix | Validation scaled MSE and five-seed paired ratios |
| Temporal shift | Later chronological validation block | MSE degradation relative to in-distribution validation |
| Missing sensors | Fixed 10% and 30% channel dropout at inference | MSE degradation and failure rate |
| Additive noise | Fixed signal-relative Gaussian noise levels | MSE degradation |
| Efficiency | RTX 5090 D, FP32, device-resident B32/L96/H96 | Median prediction-only latency, parameters, peak memory |

The stressor seed, channels, noise scale, timing warm-up, and target domains must be registered before any target test metric is read.

## Promotion rule

A candidate moves to an authorized held-out release only if all of the following hold on at least two leave-one-domain-out targets:

- AutoCon-plus-reconstruction is non-inferior to reconstruction-only pretraining under the registered one-sided 1.02 ratio gate;
- it improves at least one low-data or robustness endpoint over both random initialization and reconstruction-only control;
- it remains non-inferior to the strongest registered external baseline on the target validation split;
- the prediction-only one-scan latency is no greater than the selected nonlinear reference budget; and
- all seeds, source hashes, pretraining checkpoint hashes, split prefixes, and source exclusions are auditable.

Failure on any criterion stops the candidate rather than triggering target-specific hyperparameter search.

The reconstruction-only control passes `mamba-lambda=0`, which skips relation-loss construction and computation. Fine-tuning may freeze the shared encoder for a registered warm-up interval, but the freeze interval must be identical across pretraining conditions.

## Current implementation smoke

The local smoke validated one shared model across ETTh1 (7 channels), Weather (21), and PEMS03 (358), using only each dataset's train prefix. It performed masked reconstruction plus train-only ACF relation learning and completed under the GPU guard. This proves shape and gradient compatibility only, not transfer quality.

The complementary reconstruction-only smoke passed `mamba-lambda=0` and completed without constructing the relation objective. A 10% ETTh1 fine-tuning smoke restored 30 compatible shared encoder keys from the pretraining checkpoint and evaluated only the chronological validation block. These are functional checks, not performance evidence.

An initial Traffic 10% direction pilot exposed an invalid masking implementation: the encoder saw patches whose errors were masked only at loss aggregation. The result is audit-only and was superseded by true mask-token encoding. The corrected channelwise masked-patch pilot then ran with a source pool excluding Traffic and failed: reconstruction and AutoCon-plus-reconstruction were `+2.029%` and `+2.190%` above random initialization. A final channel-context variant activated ACF lag-pair relations in `94.5--96.7%` of global anchors and improved over reconstruction-only, but remained `+0.985%` above random initialization. The full matrix remains unauthorized and the pretraining candidate is stopped. See [`Traffic pilot`](mambats_autocon_pretraining_traffic_pilot_20261002.md).

The prediction-aligned next-patch alternative also failed under the same source exclusion and lag-pair mechanism. Predictive-only and predictive-plus-AutoCon were `+2.701%` and `+2.594%` above random initialization, despite `95.7--96.7%` active relation anchors. The pretraining line is stopped rather than tuned after this result.

## Follow-up: retained MambaTSv3 core transfer

The failed standalone shared encoder was not substituted into the forecasting
model. A separate exploratory study instead retained the validated one-scan
MambaTSv3 feature path and averaged only channel-invariant source-core weights.
It improved two 10% target validation conditions, but the exact source
AutoCon-active versus lambda-zero ablation was only `-0.026%` on Traffic.
Consequently this protocol's AutoCon pretraining claim remains stopped: the
observed transfer gain cannot be attributed to AutoCon. The full bounded
readout is [`MambaTSv3 core transfer`](mambatsv3_core_transfer_lowdata_20261002.md).
