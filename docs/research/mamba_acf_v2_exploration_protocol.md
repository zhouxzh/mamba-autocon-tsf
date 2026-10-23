# MambaACF v2 Exploration Protocol (v3 relation objective addendum)

**Status:** launch-time exploratory protocol; not a retrospective registration.
**Created:** 2026-08-31.
**Scope:** a new, independently versioned MambaACF experiment only. This document does
not amend `corrected_training_plan.json`, `highdim_training_plan.json`, or any completed
artifact.

**Version boundary:** the original matrix and its relation cells are v2. The current
source contains `ACFRelationLoss.objective_version=3`; the v3 semantics and pilot below
are an addendum for a new launch. They do not relabel completed v2 jobs, and any formal
v3 selection must use a new plan identifier, provenance sidecar, and output root after
the candidate grid is frozen.

## 1. Question, claim boundary, and exclusions

The research question is narrowly defined:

> Can a direct Mamba-2 forecaster that uses train-only signed ACF information for
> cross-variable representation, legal lag memory, and AutoCon relation learning match a
> validation-selected forecasting baseline on Electricity and Traffic while being more
> efficient than comparable nonlinear baselines?

The candidate is a **Mamba-centered direct forecaster**, not a literal model containing
only Mamba operators: it includes an input embedding, direct horizon head, residual path,
and optional ACF modules. A result may support a conditional statement about this exact
implementation and data contract; it cannot support universal Mamba, ACF, or efficiency
claims.

The following evidence is explicitly excluded from all selection, final aggregation, and
headline claims under this protocol:

- the completed `artifacts/paper_final/corrected/highdim_*_v3` outputs, whose selection
  provenance is `posthoc_verified` and `publication_ready=false`;
- all pre-existing Traffic/H96 MambaACF pilots, including the three-seed selector replay;
- historical Mamba1/2+AutoCon results, which use a different AutoCon forecasting head;
- any run launched before the source, data, candidate list, and seed lists below are
  frozen into a new provenance sidecar.

Those artifacts remain useful diagnostics, but never enter a table, threshold, or tie
break here. If this protocol is changed after a validation result is inspected, the changed
protocol receives a new identifier and all affected selection work restarts.

## 2. Fixed data contract

| Dataset | CSV | Channels | Features | Primary selection horizon | Frozen evaluation horizons |
| --- | --- | ---: | --- | ---: | --- |
| Electricity | `dataset/electricity/electricity.csv` | 321 | `M` | 96 | 96, 336, 720 |
| Traffic | `dataset/traffic/traffic.csv` | 862 | `M` | 96 | 96, 336, 720 |

Before the first selection job, the runner must record and verify the following source
hashes, CSV schema, `date` column, `OT` target column, hourly cadence, finite values, and
the repository loader split boundaries:

| Dataset | Required SHA256 |
| --- | --- |
| Electricity | `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad` |
| Traffic | `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16` |

All models use `seq_len=96`, `label_len=48`, the same train/validation/test boundaries,
the same scaling fitted on train rows, and `features=M`. The H96-only selection stage is
deliberate: H336 and H720 must not influence architecture choice. Their validation split
is still used for early stopping after the architecture is frozen, but never to change
the chosen structure.

## 3. Current implementation contract

The experiment distinguishes implemented features from a proposed feature. A proposed
 feature must not be silently treated as present merely because it appears in this protocol.

| Component | Current status | Exact role and constraint |
| --- | --- | --- |
| Variable-token Mamba | Implemented: `mamba_variable_mixer=true` | One window per channel becomes a variable token; the scan is order-sensitive and uses a shared Mamba branch. |
| Train-only ACF conditioning | Implemented: `mamba_acf_channel_conditioning=true` | Adds four signed, per-channel train-prior statistics to variable tokens through a zero-initialized residual adapter: signed mean, absolute mean, lag-weighted signed mean, and strongest signed peak. Requires the variable mixer. |
| Causal ACF memory | Implemented: `mamba_use_acf_memory=true`, `mamba_acf_lag_mode=train_peaks_available` | Selects train-peak lags by mean absolute ACF times legal forecast-step availability. Unavailable positions are masked; future targets and recursive predictions are never supplied as memory. |
| Grouped variable scan | Implemented: `mamba_variable_group_size=g` | Pools **contiguous** variable tokens before the variable scan and broadcasts group context back to channels. `g=1` is the ungrouped control. Requires the variable mixer. |
| Random variable permutation | Implemented: `mamba_variable_permutation=true` | Training-only order-robustness control. It is not an ACF ordering method and is not combined with deterministic ACF ordering in the initial screen. |
| ACF channel ordering | Implemented: `mamba_acf_channel_order=strength` | Uses the train-only signed ACF prior to sort variable tokens by dominant absolute peak strength, then peak lag and source-column index; the inverse order restores the original channel layout. It is opt-in and requires the variable mixer. |
| ACF-conditioned low-rank channel gate | Implemented: `mamba_acf_channel_gate=true` | Injects a fixed-width train-only ACF summary into the existing low-rank channel gate. It requires `mamba_channel_gate`, does not require a variable-token scan, and introduces no dense `C x C` parameter. |
| Static causal ACF memory cache | Implemented for attached train priors | Precomputes horizon/lag coefficients and legal indices once per `(length, horizon, lags, temperature)` contract, then accumulates one lag at a time. The cached path is algebraically equivalent to the masked reference path and keeps reliability batch-independent for broadcast. |
| AutoCon relation objective | Implemented: `MambaACFAutoCon` + `mamba_acf_relation_loss=true` | Current source is `ACFRelationLoss.objective_version=3`: separate projection head, signed relation, explicit candidate/target masks, and lag-paired batching. It is a separate model arm, not an unreported extra loss. |

### ACFRelationLoss v3 contract

For any new v3 run, a relation candidate is a non-identity window pair whose
distance is inside the measured ACF table and is at least
`mamba_acf_relation_min_lag`. The lower bound therefore filters both the
denominator candidate set and the supervised target set; it does not remove a
lag from forecast memory. `selected_lags` are resolved from the train-only
prior and have the following target semantics:

| `target_mode` | Positive target support | Role of selected lags |
| --- | --- | --- |
| `hard_selected` | selected lags only when supplied; otherwise legacy all-candidate fallback | v2-compatible hard positives |
| `soft_multi_peak` | candidates with `abs(signed_acf) > soft_min_abs` | multiplicative weight `1 + selected_lag_boost` |
| `soft_full` | every candidate with non-zero ACF | sensitivity control; threshold ignored |

Magnitude is fitted with `abs(cosine)` and direction with a separate sign
hinge. `relation_scope=global` is a clean scope for the v3 semantic/mechanism
control; `local` and `both` are ablations and may have no active positions for
lags at or above the window length. Local loss is normalized over active
temporal positions. The 30-run screen did not show a stable accuracy gain for
`soft_multi_peak`, so no target mode is treated as an accuracy recommendation.
Every v3 log must record objective version, target mode, candidate fraction,
candidate-active-anchor fraction, positive-count, high-ACF-excluded fraction,
and local target/active-position fractions.

### Deterministic ACF channel order contract

The implemented initial definition is `acf_strength_desc_v1`:

1. Fit the signed residual ACF only on the train split and resolve the same
   `train_peaks_available` lag set `L` used by the candidate's memory branch.
2. For each channel, identify its strongest selected signed ACF peak. Rank channels by
   descending absolute peak magnitude, then ascending peak lag, then ascending source
   column index. The signed peak remains available to conditioning and memory; it is not
   discarded merely because ranking uses magnitude.
3. Persist the resulting permutation, inverse permutation, selected lags, dominant peak
   values, and train-data hash.
4. Apply the permutation to variable tokens and corresponding ACF summary rows; restore
   forecast output to the original column order before metrics. If `group_size > 1`, sort
   first and form contiguous groups second. Padding introduced by the last group has no
   score and no target.
5. Do not enable `mamba_variable_permutation` in this arm. Randomizing the very sequence
   whose deterministic order is the hypothesis would make the ablation uninterpretable.

The implementation gate is covered by tests demonstrating: deterministic repeatability;
no dependence on validation/test values; exact inverse restoration; source-order and
ACF-order shape equivalence; and permutation equivariance of the **ordering procedure**:
when input columns, target columns, and ACF-prior rows are permuted together, then
`acf_strength_desc_v1` is recomputed from stable source-column identities and the restored
forecast/loss is unchanged. This does not assert that an arbitrary fixed Mamba scan is
order invariant. If a future implementation changes this definition or a contract fails,
remove the arm rather than replacing it with an undocumented approximation.

## 4. Candidate matrix

All candidates use Mamba-2, one temporal block, `mamba_d_conv=4`,
`mamba_expand=2`, `mamba_headdim=16`, `mamba_scales=[1]`, and a common train-only residual
ACF source. During validation-only selection, `d_model` is restricted to `{32, 64}` with
`d_ff=2*d_model`; the selected width is part of the frozen configuration. No test metric
may choose width, lag, group size, loss weight, or ordering.

| ID | Model and enabled components | Purpose |
| --- | --- | --- |
| C0 | `MambaACF`, no optional forecast-ACF, variable, or AutoCon branch | Direct-forecaster control. A trainer may attach a train prior, but C0 cannot consume it in its forecast path. |
| C1 | C0 + variable mixer, `group_size=1`, source column order | Cross-variable-token control. |
| C2 | C1 + train-only ACF conditioning | Tests whether the four-statistic prior adds information beyond variable tokens. |
| C3 | C2 + `train_peaks_available` causal memory | Main non-AutoCon v2 trunk. |
| C4-g | C3 with `group_size` in `{4, 8}`; source column order | Cost/accuracy trade-off. `group_size=1` is C3, not a duplicate. |
| C4-o | C3 with `group_size=1` and `acf_strength_desc_v1` order | Deterministic train-prior order hypothesis; opt-in and not the default. |
| G1 | C0 + low-rank channel gate + `mamba_acf_channel_gate=true`; no variable-token scan | Tests whether a compact train-only ACF descriptor improves cross-variable modulation without paying for a variable-axis recurrent scan. |
| C5 | Selected C3/C4 trunk as `MambaACFAutoCon` plus signed ACF relation loss | Full AutoCon representation-learning candidate. |
| R1 | C3 + training-only random variable permutation | Order-sensitivity robustness control; not eligible as the primary ACF-order result. |

For C5, the original v2 validation-only relation grid was
`AutoCon_lambda in {0.003, 0.01, 0.03}`, with
`mamba_acf_relation_batching=lag_pairs`, `mamba_acf_relation_scope=both`,
`mamba_acf_relation_channel_reduce=mean`, and `mamba_acf_relation_lag_mask=true`.
That grid is historical for this protocol. A new v3 launch must freeze and record a
separate target-mode grid. `soft_multi_peak` may be included as a semantic/mechanism
control with `relation_scope=global` and `lag_mask=false`, but it is not a precision
recommendation: the subsequent three-seed screen did not show a stable gain. Its
`soft_min_abs`, `selected_lag_boost`, and `min_lag` values must be predeclared rather
than inherited from a pilot. The selected lambda and trunk are global across both
datasets at H96.
If relation-pair coverage is zero for any selected cell, C5 is infeasible for that cell
and cannot be promoted by substituting an unlogged random-batch objective. Coverage,
active-anchor fraction, target mode, forecast loss, and relation loss must be retained
per epoch.

### v3 relation mechanism pilot, excluded from selection

The latest v3 hard/soft result is deliberately excluded from this protocol's selection
and final aggregation, as required by Section 1. It used Traffic/H96, seed `2021`, three
epochs, a global-only `MambaACFAutoCon` relation loss, and no forecast ACF memory:

| Target | scaled MSE | scaled MAE | positive targets/anchor | high-ACF excluded |
| --- | ---: | ---: | ---: | ---: |
| `hard_selected`, `lag_mask=true` | 0.617183 | 0.400212 | 0.944 | 0.681 |
| `soft_multi_peak`, `soft_min_abs=0.1`, `selected_lag_boost=1.0`, `lag_mask=false` | **0.609619** | **0.397436** | 4.287 | 0.000 |

The soft target is about `1.23%` lower in scaled MSE in this early paired implementation
pilot, but the later 30-run screen did not show a stable gain. It is one seed, one horizon
and a three-epoch budget, so it only motivates a predeclared v3 validation grid; it cannot
select C5, establish a soft-target gain, or be pooled with v2/memory artifacts. See
[`artifacts/mamba_acf_soft_relation_pilot_20260831/`](../../artifacts/mamba_acf_soft_relation_pilot_20260831/).

### v3 30-run screen summary

The exact readout contains five phases x two datasets x three seeds (`30` successful
runs). Each dataset-phase estimate has `n=3`; means and SDs are descriptive and the
pooled six-run row is not six independent replications. The phase means (MSE / MAE) are:
This screen is a post-v2 exploratory readout and does not amend the protocol's original
selection grid or make any target mode eligible by itself.

| Phase | ETTh1 MSE | Traffic MSE | ETTh1 MAE | Traffic MAE |
| --- | ---: | ---: | ---: | ---: |
| plain_screen | 0.389664 ± 0.002807 | 0.570445 ± 0.003733 | 0.411648 ± 0.002041 | 0.389375 ± 0.002401 |
| available_memory_screen | 0.389005 ± 0.001678 | 0.555458 ± 0.005670 | 0.408272 ± 0.002361 | 0.384809 ± 0.003573 |
| hard_relation_nonoverlap_screen | 0.391957 ± 0.002591 | 0.565820 ± 0.008408 | 0.413486 ± 0.003919 | 0.384731 ± 0.004063 |
| soft_relation_nonoverlap_screen | 0.392103 ± 0.001567 | 0.568679 ± 0.004372 | 0.412781 ± 0.002968 | 0.388632 ± 0.008201 |
| available_memory_soft_relation_nonoverlap_screen | 0.395972 ± 0.005431 | 0.551952 ± 0.004095 | 0.414361 ± 0.006776 | 0.381977 ± 0.007107 |

Soft versus hard mean MSE changes are `+0.04%` (ETTh1) and `+0.51%` (Traffic),
so the three-seed screen does not support a stable soft-target accuracy gain.
Memory+soft versus available memory is `+1.79%` on ETTh1 and `-0.63%` on Traffic;
the Traffic direction is favorable but remains exploratory. Full seed-level values,
tests, and provenance are in [`mamba_acf_v3_screen_readout.md`](./mamba_acf_v3_screen_readout.md).

The architecture-selection order is intentionally staged rather than a full factorial:

1. Screen C0--C4, G1, and R1 at H96 validation only.
2. Freeze one non-relation trunk using the rule in Section 5.
3. Compare that trunk with C5 over the fixed relation grid, still on H96 validation only.
4. Freeze either the full C5 configuration or record that the combined AutoCon candidate
   failed. A plain trunk must not be relabeled as an AutoCon method if C5 loses.

This keeps the AutoCon contribution mandatory for the proposed combined method while
allowing a valid negative conclusion if it does not improve the validation criterion.

## 5. Phase A: validation-only architecture selection

### Separation and seeds

- Selection seeds: `1101, 1102, 1103`.
- Selection uses `--skip_test`; its process must not instantiate or read a test loader.
- Each phase receives a new output root, immutable source snapshot, complete command
  manifest, data hashes, environment capture, and validation trajectories.
- No selection seed or selected configuration is reused from a prior artifact.

### Baselines and validation budget

Every selection cell includes the following reference models:

`DLinear`, `NLinear`, `PatchTST`, `TimesNet`, `Mamba1`, and `Mamba2`.

The following is the fixed validation-only capacity grid; an executable manifest must
encode it exactly before jobs launch:

| Family | Candidate settings |
| --- | --- |
| DLinear, NLinear | `individual=false`, `individual=true` |
| PatchTST, TimesNet | `d_model in {32,64}`, `d_ff=2*d_model`, `e_layers=1` |
| Mamba1, Mamba2 | `d_model in {32,64}`, `d_ff=2*d_model`, `e_layers=1`, `d_state=16`, `d_conv=4`, `expand=2`; Mamba-2 additionally uses `headdim=16` |
| C0--C5, G1, and R1 | `d_model in {32,64}`, `d_ff=2*d_model`, with every component setting fixed in Section 4 |

The runner records parameter counts for every selected setting; this grid is a controlled
model-selection budget, not a claim that the resulting parameter counts are exactly
matched. DLinear/NLinear are retained as accuracy and cost anchors, even though a
nonlinear Mamba model cannot credibly promise lower latency than these simple linear
predictors.

All selection jobs use the same optimizer family, `epochs=50`, `patience=10`, cosine
schedule, `learning_rate=1e-3`, `weight_decay=1e-5`, `batch_size=16`, and the fixed
dataset split unless a model has an explicitly documented incompatibility. OOM handling is
limited to a predeclared batch-size reduction; it cannot change model width, candidate
membership, or epochs after validation metrics have been read.

### Selection rule

For each dataset `d`, compute the mean best-checkpoint validation MSE across the three
selection seeds. Let `B_d` be the lowest such validation MSE among the six baseline
families after their validation-only choices are frozen. For candidate `a`, compute:

\[
  q(a) = \max_{d \in \{Electricity, Traffic\}}
  \frac{\overline{MSE}_{val}(a,d)}{B_d}.
\]

The selected non-relation trunk is the eligible candidate with the lowest `q(a)`. A
candidate is eligible only when it completes every seed, passes all ACF/model contracts,
and has `q(a) <= 1.02`. Ties within 0.5% are resolved by lower median end-to-end epoch
time, then lower peak allocated GPU memory, then the lexicographic candidate ID. This is
a deterministic operational rule, not a significance test.

C5 is retained only if its global `q(a)` is no larger than the frozen trunk's `q(a)` and
its relation diagnostics are valid in all cells. Otherwise the protocol records a failed
combined AutoCon candidate; it does not choose a relation weight from test performance.
If no candidate meets the eligibility rule, Phase B does not start and the result is an
honest negative finding.

## 6. Freeze gate before any test read

Before a single final-test metric is generated or viewed, create an immutable selection
record containing:

- the exact selected candidate ID, all command arguments, selected width, group size,
  effective lag set, ACF order (if applicable), and AutoCon lambda;
- validation seed-level table and the deterministic calculation of `q(a)`;
- dataset, source-tree, plan, and environment hashes; GPU, PyTorch, `mamba_ssm`,
  `causal-conv1d`, and Triton versions;
- parameter count; peak allocated/reserved GPU memory; and median epoch-time methodology;
- failed jobs, OOM adjustments, relation coverage, and the decision result;
- a signed/attested `selection_provenance` sidecar that binds the final-stage job matrix
  to the selection output.

Any rerun caused by a code, dataset, driver, command, or selected-config change receives a
new phase root and repeats Phase A. Training failures cannot be silently dropped; a failed
final seed is reported and prevents confirmatory aggregate inference for that cell.

## 7. Phase B: frozen five-seed evaluation

Only after the freeze gate passes, run the five final seeds on both datasets at H96, H336,
and H720. The final matrix contains all six baselines plus the following controls:

1. C0 direct MambaACF control;
2. the frozen non-relation v2 trunk;
3. the frozen C5 MambaACF + AutoCon relation candidate, if C5 passed Phase A.

The selected configuration is constant across all final seeds. H336/H720 use the same
frozen architecture; their train-only priors, resolved lag sets, and optional channel
order are recomputed from their own train split and horizon and persisted, but they do not
open a new architecture search. Checkpoints are trained with validation early stopping
only. Then validate the complete five-seed training manifest before performing a batch
test evaluation. This prevents early test cells from influencing later training decisions.

The primary accuracy comparator for each dataset/horizon is the **single baseline family
named by H96 validation before test read**; `PatchTST` and `Mamba2` are mandatory
secondary accuracy comparators regardless of which family is selected. All six baseline
results remain in the final table. The comparator identity and its selected configuration
are part of the freeze record; it cannot be changed after tests are visible.

## 8. Analysis, efficiency claim, and stopping conditions

### Accuracy

The primary endpoint is train-standardized test MSE, with original-scale MAE/MSE/RMSE and
channel-level metrics as required diagnostics. The inference unit is the paired seed, not
sliding window or channel. For each dataset/horizon, report the five seed-level values,
paired log-MSE ratio, mean paired delta, and a multiplicity-adjusted confidence interval
across the six dataset-horizon cells. Holm adjustment is used for the fixed primary and
secondary comparisons; BH-FDR is an additional sensitivity analysis.

The combined C5 method can claim predeclared **non-inferiority** only if the upper bound
of the one-sided paired 95% t confidence interval (calculated on seed-level log-MSE
ratios, with Holm-adjusted alpha across the six cells) versus the frozen primary
comparator is at most `1.01` for every Electricity/Traffic and H96/H336/H720 cell. A
superiority claim requires the corresponding upper bound below `1.00`. Five seeds may
fail to establish either condition; absence of evidence is reported as inconclusive, not
as equivalence.

### Efficiency

No nonlinear candidate will claim to be faster than DLinear or NLinear. The valid
efficiency comparators are `PatchTST`, `TimesNet`, and `Mamba2` under identical device,
precision, batch shape, sequence length, horizon, warm-up, and timed-iteration settings.
Measure parameter count, peak allocated/reserved memory, device-resident forward latency,
forecast throughput, and end-to-end epoch time including data loading. At least 50 warm-up
and 200 timed GPU iterations are required; report median and p90 rather than a single
timing.

An efficiency headline requires both accuracy non-inferiority above and at least 10% lower
median end-to-end epoch time than the named nonlinear comparator for every claimed cell.
Synthetic forward timing is implementation evidence only and cannot replace this result.

### Mandatory negative reporting and no-go conditions

The final report must include all candidates that completed Phase A, all failed/infeasible
relation cells, exact ACF coverage, selected lags/orders, and per-channel error summaries.
It must stop short of a combined-method publication claim when any of the following occurs:

- C5 fails validation or relation coverage, so AutoCon was not part of the selected model;
- any selected final cell lacks all five valid seed outputs;
- the required ACF-order implementation/test gate fails;
- C5 fails the predeclared non-inferiority condition in any claimed cell;
- the speed condition is not met against the named nonlinear comparator;
- a data, source, or selection-provenance hash changes after freeze.

In those cases, the correct deliverable is a reproducible ablation/negative-results report,
not an assertion that MambaACF v2 is generally more accurate or faster.

## 9. Human review checklist

Before calling the result suitable for a paper, verify:

1. The new phase roots and provenance files exist and do not reference historical
   post-hoc artifacts as inputs.
2. Selection logs have no test loader access and all Phase A choices are reproducible from
   validation trajectories alone.
3. The selected model configuration agrees with this document, including ordering, group
   size, ACF source, effective lags, memory mask, and relation settings.
4. Final metrics, resource figures, and confidence intervals are seed-level and include
   all six baselines and all required controls.
5. The final prose limits claims to the cells satisfying the registered accuracy and
   efficiency conditions, while preserving null and negative findings.

Related implementation and evidence context: [MambaACF code review](autocon_mamba_acf_core_review.md),
[external Mamba architecture review](external-mamba-ts-architecture-review.md), and
[high-dimensional evidence boundary](highdim-extension-protocol.md).
