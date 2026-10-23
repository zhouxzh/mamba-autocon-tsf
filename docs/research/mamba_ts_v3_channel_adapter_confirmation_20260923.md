# MambaTS v3 AutoCon Channel Adapter: Independent Confirmation Boundary

The exploratory adapter screen passed its registered rule on PEMS03/04/07 but
was run with only two seeds. This confirmation matrix is frozen before any
test release and adds strong plain baselines.

## Matrix

- Datasets: PEMS03, PEMS04, PEMS07, PEMS08, with the existing NPZ hashes and
  feature-axis contracts.
- Horizons: 12 and 96; input length 96; M features.
- Seeds: `2021, 2022, 2023, 2024, 2025`.
- Arms: DLinear, PatchTST, TimesNet, Mamba2, MambaTSv3AutoCon null, and
  MambaTSv3AutoCon rank-8 channel adapter.
- Total: `4 x 2 x 5 x 6 = 240` validation-only jobs.
- Shared budget: 20 epochs maximum, patience 5, identical optimizer and
  normalization per coordinate, `skip_test=true`.

## Frozen gates

The adapter can proceed to a test-release proposal only if all conditions hold:

1. It is non-inferior to the best of the four non-adapter Mamba/forecasting
   baselines under a predeclared 1.02 relative-MSE margin in at least 6 of 8
   dataset-horizon cells.
2. It improves over its exact MambaTSv3 AutoCon null in at least 6 of 8 cells,
   with no cell worse than 1%.
3. It remains within 10% of the null in the fixed device-resident latency
   protocol and reports synchronized train wall time, parameter count, and peak
   GPU memory.
4. PEMS08 is reported as a mandatory failure-analysis cell even if the above
   aggregate gates pass.

If any gate fails, stop architecture expansion and do not use the test set to
rescue the candidate. A separate approved launch sidecar is required; this
document itself authorizes no training.

## Completed result (2026-09-23)

The authorized frozen plan completed `240/240` validation-only jobs with zero
failures. The independent reader accepted all records: all six arms used the
same source snapshot
`f2a2ca1f2b3cc0ce7a6e7e33c42b6c9000b9d5be8be19b5578d69cdd1364ddab`, every
cell has five seeds, and no test metric artifact was produced. The raw,
machine-readable readout is
[`readout.md`](../../artifacts/exploratory/mamba_ts_v3_channel_adapter_confirmation_20260923/readout/readout.md).

| Dataset | H | Adapter vs. best baseline | Adapter vs. exact null |
| --- | ---: | ---: | ---: |
| PEMS03 | 12 | -0.257% | -0.257% |
| PEMS03 | 96 | +2.003% | -2.953% |
| PEMS04 | 12 | -0.268% | -0.268% |
| PEMS04 | 96 | +9.820% | -1.216% |
| PEMS07 | 12 | -0.395% | -0.395% |
| PEMS07 | 96 | -2.264% | -2.264% |
| PEMS08 | 12 | +1.244% | +1.244% |
| PEMS08 | 96 | +2.160% | +2.160% |

Negative values favor the adapter. It improves over the exact null in 6/8
cells, but fails the no-regression condition on both PEMS08 cells. It is also
not non-inferior to the best external baseline in PEMS04/H96: its MSE is
9.820% higher than TimesNet, well beyond the predeclared 2% margin. Thus gates
1 and 2 fail, even though the adapter is better than its null in six cells.

The synchronized PEMS07/H96 device-resident prediction-only benchmark
(`B=2`, 50 warm-up, 200 timed forwards, RTX 5090 D) records null/adapter
median latency `0.32693/0.34517 ms`, a `+5.58%` increase; parameter totals
are `104,291/105,411`, and both have `38.854 MB` peak allocated increment.
Across the eight matched validation cells, the recorded mean training wall time
is `70.648/71.286 s` (null/adapter, `+0.90%`).
The latency gate passes, but it cannot rescue the failed accuracy gates.
Artifacts are
[`model_resources.csv`](../../artifacts/exploratory/mamba_ts_v3_channel_adapter_confirmation_20260923/resource_pems07_h96/model_resources.csv)
and [`training_elapsed_summary.csv`](../../artifacts/exploratory/mamba_ts_v3_channel_adapter_confirmation_20260923/resource_pems07_h96/training_elapsed_summary.csv).

### Decision

Stop this rank-8 channel-adapter branch. Do not open the test set, tune PEMS08,
or claim a general AutoCon/ACF accuracy improvement from it. The valid narrow
finding is that reusing the AutoCon-trained MambaTS-v3 channel states can
improve some PEMS long-horizon validation cells at small inference cost, but
the effect is dataset-dependent and does not satisfy the frozen publication
gate.
