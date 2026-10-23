# MambaACF v2 Screening Plan (with v3 relation addendum)

`mamba_acf_v2_benchmark_plan.json` is a launch-time screening matrix for the
new local patch branch and the inverted Transformer baseline. It is separate
from the historical `corrected` and `strict` phases; its short one-seed runs
cannot be used as confirmatory evidence. The JSON remains a **v2** plan: its
relation phase was launched with `scope=both` and `lag_mask=true`. The v3
relation objective in the current source is documented below and must receive a
new plan identifier/output root if it is launched; it does not retroactively
change completed v2 artifacts.

The matrix has four deliberately separate phases:

- `baseline_screen_M`: DLinear, NLinear, PatchTST, TimesNet, iTransformer,
  Mamba-1 and Mamba-2 on ETTh1, Weather and Exchange-Rate;
- `highdim_screen_M`: DLinear, PatchTST, iTransformer and Mamba-2 on
  Electricity and Traffic;
- `acf_local_memory_screen_M`: MambaACF with channel-preserving local patches
  and train-only, availability-weighted signed-ACF memory;
- `acf_local_relation_screen_M`: MambaACFAutoCon with the same local branch and
  signed ACF relation loss. In the v2 launch, selected train-only lags define
  positives; in current v3 code, resolved lags define selected targets/boosts
  while all in-range non-identity candidates (subject to `min_lag`) remain in
  the denominator.

Every phase uses `H=96`, a fixed `seq_len=96`, the same optimizer settings and
one seed for screening. A follow-up confirmatory plan must freeze the architecture
and AutoCon lambda from validation only, then repeat at least five seeds and
measure paired deltas, channel-level errors, peak GPU memory and end-to-end
epoch time.

The Weather CSV is retained as a screening dataset but is not bound to the
strict high-dimensional quality-contract validator: its source file contains a
known duplicate timestamp and a 10-minute cadence while the legacy registry
uses the generic `t` frequency label. The exact SHA256 is recorded in the JSON,
and this exception must be resolved or explicitly reported before Weather is
used in a confirmatory table.

## v3 relation objective boundary

`ACFRelationLoss.objective_version=3` separates denominator candidates from
positive targets. A candidate pair must be non-identity, within the measured ACF
table, and at least `mamba_acf_relation_min_lag` apart. The target mode is
explicitly recorded as one of:

- `hard_selected`: v2-compatible selected-lag positives when a selected-lag
  mask is supplied; otherwise it retains the legacy all-candidate fallback;
- `soft_multi_peak`: every candidate with `abs(signed_acf) > soft_min_abs` is a
  positive, with selected lags weighted by `1 + selected_lag_boost`;
- `soft_full`: every non-zero ACF candidate is a positive sensitivity control.

`soft_multi_peak` is retained as a semantic/mechanism control, not a precision
recommendation: the subsequent three-seed screen did not show a stable accuracy
gain. Any new-run target mode, scope, `lag_mask`, `soft_min_abs`,
`selected_lag_boost`, and `min_lag` must be explicitly recorded before launch.
`min_lag=seq_len` provides a no-overlap global control and does not alter
forecast-memory lags. Local/both scopes remain valid ablations, but their active
positions can be empty for long lags at `seq_len=96`. v3 logs must contain
objective version, target mode, candidate and active-anchor coverage,
positive-count, high-ACF-excluded fraction, and local active-position diagnostics.

## Latest v3 pilot (not part of this v2 matrix)

Traffic/H96, seed `2021`, three epochs, same `MambaACFAutoCon` backbone and
global scope, with forecast ACF memory disabled:

| Target | scaled MSE | scaled MAE | Interpretation |
| --- | ---: | ---: | --- |
| `hard_selected` (v3 compatibility) | 0.617183 | 0.400212 | 0.944 positive targets/anchor; 68.1% of strong candidates excluded |
| `soft_multi_peak` | **0.609619** | **0.397436** | 4.287 positive targets/anchor; no strong candidates excluded |

The soft row is about `1.23%` lower in scaled MSE than the hard row in this early
single-seed, three-epoch mechanism pilot, but the later 30-run screen did not
replicate a stable soft-target gain. It is not comparable to the memory-only or
historical AutoCon tables and is not evidence of publication readiness. Artifacts:
[`hard`](../../artifacts/mamba_acf_soft_relation_pilot_20260831/hard/) and
[`soft`](../../artifacts/mamba_acf_soft_relation_pilot_20260831/soft/).

### v3 30-run screen readout

The exact readout contains five phases, two datasets, and three seeds per
dataset-phase (`30` successful runs total). Each dataset-phase estimate has
`n=3`; the pooled `ALL` row is descriptive and is not six independent
replications. Mean ± seed-level SD:

| Phase | ETTh1 MSE | Traffic MSE | ETTh1 MAE | Traffic MAE |
| --- | ---: | ---: | ---: | ---: |
| plain_screen | 0.389664 ± 0.002807 | 0.570445 ± 0.003733 | 0.411648 ± 0.002041 | 0.389375 ± 0.002401 |
| available_memory_screen | 0.389005 ± 0.001678 | 0.555458 ± 0.005670 | 0.408272 ± 0.002361 | 0.384809 ± 0.003573 |
| hard_relation_nonoverlap_screen | 0.391957 ± 0.002591 | 0.565820 ± 0.008408 | 0.413486 ± 0.003919 | 0.384731 ± 0.004063 |
| soft_relation_nonoverlap_screen | 0.392103 ± 0.001567 | 0.568679 ± 0.004372 | 0.412781 ± 0.002968 | 0.388632 ± 0.008201 |
| available_memory_soft_relation_nonoverlap_screen | 0.395972 ± 0.005431 | 0.551952 ± 0.004095 | 0.414361 ± 0.006776 | 0.381977 ± 0.007107 |

Relative to hard relation, soft relation changes mean MSE by `+0.04%` on
ETTh1 and `+0.51%` on Traffic, with inconsistent seed directions. Relative to
available memory, memory+soft changes mean MSE by `+1.79%` on ETTh1 and
`-0.63%` on Traffic. These are exploratory paired deltas, not confirmation.

Validate the plan without launching jobs:

```bash
conda run --no-capture-output -n mamba python scripts/run_training_plan.py \
  --plan docs/research/mamba_acf_v2_benchmark_plan.json --validate_only
```

Audit completed outputs without pooling them with other phases:

```bash
conda run --no-capture-output -n mamba python scripts/audit_benchmark_matrix.py \
  --plan docs/research/mamba_acf_v2_benchmark_plan.json \
  --artifact-root artifacts --output-dir artifacts/benchmark_audit_v2
```
