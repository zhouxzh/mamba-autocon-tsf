# Dataset and Baseline Inventory (2026-09-03)

This is a read-only inventory of the datasets, model implementations, persisted
forecast metrics, and resource measurements currently present in the repository.
It is intended for human review. It does not merge results across provenance,
select hyperparameters, or authorize another training run.

## Scope and audit source

The counts below come from the post-contract 2026-09-03 benchmark audit:

```text
conda run --no-capture-output -n mamba python scripts/audit_benchmark_matrix.py \
  --plan-dir docs/research \
  --artifact-root artifacts \
  --output-dir artifacts/benchmark_audit_20260903_post_contract
```

The audit found `5,882` persisted `config.json` files and `2,977` usable metric
artifacts. A metric is counted only when `metrics.npy` is finite and has at least
MAE/MSE/RMSE, or when `stream_metrics.json` contains a finite scaled MSE. There
are `226` usable metric artifacts outside registered phase roots; these remain
historical or exploratory and are not silently promoted into a paper table.

## Datasets on disk

The nine datasets listed below are present locally and are the datasets used by
the completed matrices. They have a valid `date` column, the expected `OT`
target, and the expected number of value columns. The broader runner registry
also contains PEMS03/04/07/08 and Solar entries, but their local source files
are absent in this workspace and they are not eligible for a launch without a
separate data-contract review. Row counts below exclude the header. Frequency
is the repository's loader/plan label, not a claim that all datasets have
identical sampling semantics.

| Dataset | File | Rows | Channels | Frequency label | Features used in completed matrices |
| --- | --- | ---: | ---: | --- | --- |
| ETTh1 | `dataset/ETT-small/ETTh1.csv` | 17,420 | 7 | h | S, M |
| ETTh2 | `dataset/ETT-small/ETTh2.csv` | 17,420 | 7 | h | S, M |
| ETTm1 | `dataset/ETT-small/ETTm1.csv` | 69,680 | 7 | t | S, M |
| ETTm2 | `dataset/ETT-small/ETTm2.csv` | 69,680 | 7 | t | S, M |
| Electricity | `dataset/electricity/electricity.csv` | 26,304 | 321 | h | M |
| Traffic | `dataset/traffic/traffic.csv` | 17,544 | 862 | h | M |
| Weather | `dataset/weather/weather.csv` | 52,696 | 21 | t | S, M (M only in one-seed screen) |
| Exchange-Rate | `dataset/exchange_rate/exchange_rate.csv` | 7,587 | 8 | d | S, M (M only in one-seed screen) |
| National Illness | `dataset/illness/national_illness.csv` | 966 | 7 | d | S |

The loader uses fixed ETT boundaries and ratio-based custom boundaries. Therefore
the same numeric horizon is not automatically the same calendar duration across
hourly, sub-hourly, and daily data. Results should remain grouped by dataset and
horizon.

## Implemented model families

The generic experiment registry exposes the established forecasting baselines
and AutoCon models in `exp/exp_basic.py`. Mamba variants are dispatched through
`exp/exp_long_term_forecasting_with_mamba.py`.

| Family | Implemented entry points | Role in a fair comparison |
| --- | --- | --- |
| DLinear | `DLinear` | Required plain baseline |
| NLinear | `NLinear` | Required normalized-linear baseline; currently under-covered |
| PatchTST | `PatchTST` | Required strong attention/patch baseline |
| TimesNet | `TimesNet` | Required convolution/periodicity baseline |
| Mamba-1 | `Mamba1`, `Mamba1AutoCon` | Plain backbone and AutoCon objective ablation |
| Mamba-2 | `Mamba2`, `Mamba2AutoCon` | Plain backbone and AutoCon objective ablation |
| AutoConNet | `AutoConNet`, `AutoConNetAutoCon` | Repository representation-learning control |
| ACF-guided Mamba | `MambaACF`, `MambaACFAutoCon`, `MambaACF1`, plus variable-token aliases | Proposed method/ablations; not a substitute for plain baselines |
| MambaTS-inspired | `MambaTS`, `MambaTSv2`, `MambaTSv3`, `MambaTSv4`, `MambaTSv5` and AutoCon pairs | Patch/shared-scan candidates; v5 signed ACF anchor is currently a neutral ablation |
| iTransformer | `iTransformer` | Optional extra baseline; only a small one-seed screen exists |

An `AutoCon` suffix is an objective/configuration change, not a matched plain
baseline. A proposed ACF row must retain a same-backbone plain or null control.

## Persisted metric coverage

The following are the main registered completed phases. Counts are expected
coordinates and usable metrics under the phase's own output root; they must not
be pooled with similarly named historical directories.

| Phase | Dataset(s) | Features | Horizons | Seeds | Models in phase | Coverage | Important limitation |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| `corrected_confirmatory_S` | ETTh1/2, ETTm1/2 | S | 96, 336, 720 (ETTh1/2); 96, 336 (ETTm1/2) | 5 | DLinear, PatchTST, TimesNet, AutoConNet, AutoConNetAutoCon, Mamba1, Mamba1AutoCon, Mamba2, Mamba2AutoCon | 450/450 | No NLinear |
| `corrected_confirmatory_M` | ETTh1/2, ETTm1/2 | M | same ETT horizon grid | 5 | same nine models | 450/450 | No NLinear |
| `highdim_confirmatory_M_v3` | Electricity, Traffic | M | 96, 336, 720 | 5 | same nine models | 270/270 | No NLinear; high-dimensional Mamba is dataset-dependent |
| `secondary_transfer` | Weather, Exchange-Rate | S | 96, 336 | 5 | same nine models | 180/180 | Transfer phase, not a full multivariate comparison |
| `exploratory_ili` | National Illness | S | 24, 36, 48, 60 | 5 | same nine models | 180/180 | Short daily series; exploratory by plan |
| `mamba_acf_v2` `baseline_screen_M` | ETTh1, Weather, Exchange-Rate | M | 96 | 1 | DLinear, NLinear, PatchTST, TimesNet, iTransformer, Mamba1, Mamba2 | 21/21 | One seed; screening only |
| `mamba_acf_fair_baseline_screen` `plain_baselines_h96` | ETTh1, Traffic | M | 96 | 3 | DLinear, PatchTST, Mamba1, Mamba2 | 24/24 | No NLinear or TimesNet; exploratory |
| `extension_highdim_plain_M` | Electricity, Traffic | M | 96, 336, 720 | 5 | DLinear, NLinear, PatchTST, TimesNet, Mamba1, Mamba2 | 0/180 | Historical registered phase remains unrun; a separate fresh 216-job three-seed screen now covers both high-dimensional datasets |

The corrected and high-dimensional confirmatory matrices are complete relative
to their own manifests, but their model lists omit NLinear. The audit therefore
reports the phases as complete without treating them as complete six-baseline
matrices.

### Model and dataset totals across all usable artifacts

These totals include every registered and unregistered historical artifact, so
they are useful for locating evidence but are not a comparative table. Model
counts use the audit's canonical identity rule: variable-token aliases are kept
as separate registered arms when their immutable `plan_job_id` supports that
identity, even though they execute the same Python module.

| Dataset | Usable metrics |
| --- | ---: |
| ETTh1 | 781 |
| ETTh2 | 540 |
| ETTm1 | 360 |
| ETTm2 | 360 |
| Electricity | 140 |
| Traffic | 415 |
| National Illness | 180 |
| Weather | 97 |
| Exchange-Rate | 97 |

| Model family/entry | Usable metrics |
| --- | ---: |
| DLinear | 283 |
| NLinear | 4 |
| PatchTST | 284 |
| TimesNet | 273 |
| Mamba1 | 280 |
| Mamba1AutoCon | 270 |
| Mamba2 | 282 |
| Mamba2AutoCon | 271 |
| AutoConNet / AutoConNetAutoCon | 270 / 270 |
| MambaACF | 53 |
| MambaACF1 | 1 |
| MambaACFAutoCon | 69 |
| MambaACFVariable | 126 |
| MambaACFVariableAutoCon | 229 |
| iTransformer | 5 |

The four NLinear metrics are specifically: one Traffic/H96/seed-2021 extension
pilot and three one-seed H96 rows for ETTh1, Weather, and Exchange-Rate in
`artifacts/paper_final/mamba_acf_v2_baseline_screen_M`. There is no completed
NLinear row in the corrected ETT, high-dimensional, Weather/Exchange transfer,
or ILI five-seed matrices.

## Reusable result and resource artifacts

The following artifacts are safe starting points for audit or method selection,
with the stated boundaries:

- `artifacts/paper_final/corrected/highdim_confirmatory_M_v3/report/highdim_confirmatory_M.md` and `.../raw_report/summary.csv`: five-seed Electricity/Traffic comparison for the nine-model high-dimensional phase. It is complete for its manifest but lacks NLinear.
- `artifacts/paper_final/corrected/confirmatory_M/statistics/paper_statistics.json` and the corresponding S file: corrected ETT seed-paired AutoCon objective statistics; these are not ACF-guided Mamba results.
- `artifacts/extension_pilot_20260831/traffic_h96_summary/summary.csv`: one-seed Traffic/H96 pilot containing DLinear, NLinear, PatchTST, Mamba2, and several MambaACF variants. It is useful for smoke-level direction only.
- `artifacts/resource_v3_current/traffic_c862_h96_b8_w50_t200/model_resources.csv`: unified synthetic forward measurement for PatchTST, Mamba2, MambaACF plain, and MambaACF memory at Traffic/C862/H96/B8. It excludes data loading, backward, optimizer, and end-to-end epoch time.
- `artifacts/resource_extension/`: multiple one-row architecture/resource measurements, including DLinear, NLinear, Mamba2, PatchTST, and MambaACF variants. Timing protocols and source snapshots vary, so these rows are not one fair ranking.

The existing high-dimensional report gives a useful warning: Mamba-2 is close to
PatchTST on Electricity but substantially worse on Traffic at all three long-term
horizons. That is evidence against a universal Mamba claim, not evidence that
the ACF method has failed or succeeded everywhere.

## Missing fair comparisons

1. **NLinear is missing from the completed five-seed ETT and high-dimensional
   matrices.** This is the clearest baseline gap. The registered
   `extension_highdim_plain_M` phase explicitly asks for 180 rows and currently
   has `0/180` usable metrics.
2. **No single provenance contains all six plain baselines across multiple
   datasets and all three long horizons.** Existing one-seed NLinear rows cannot
   be copied into a five-seed table.
3. **ACF-guided Mamba does not yet have a same-protocol, multi-dataset main
   table with the six plain baselines.** Most ACF screens are ETTh1/Traffic,
   H96, three seeds, and exploratory.
4. **Efficiency evidence is forward-only or mixed-protocol.** There is no
   publication-grade end-to-end comparison including loading, backward,
   optimizer, validation, peak GPU memory, and wall-clock per epoch under one
   hardware/software contract.
5. **The secondary transfer and ILI phases are S-feature results.** They should
   not be presented as evidence for high-dimensional multivariate forecasting.
6. **Existing AutoCon lambda choices are tied to their phase provenance.** A
   proposed ACF method needs a validation-only selection rule in the same new
   plan; test metrics must not choose the lambda or architecture.

## Recommended minimum fair matrix

For a focused next paper-grade experiment, create one new immutable plan and
provenance root rather than filling old roots. The smallest useful matrix is:

| Axis | Recommended fixed values |
| --- | --- |
| Datasets | ETTh1, ETTm1, Electricity, Traffic; add Weather and Exchange-Rate as an external-domain check if budget allows |
| Features | M for all four core datasets; report S separately, never pooled with M |
| Horizons | 96, 336, 720 (daily ILI uses its own 24/36/48/60 grid and should remain secondary) |
| Seeds | 2021-2025 for final test; separate validation seeds/selection phase |
| Plain baselines | DLinear, NLinear, PatchTST, TimesNet, Mamba1, Mamba2 |
| Representation controls | AutoConNet and AutoConNetAutoCon, with a same-backbone plain/null control |
| Proposed rows | MambaACF plain; MambaACF + AutoCon; final selected structural variant only after validation |
| Metrics | scaled MSE/MAE/RMSE as primary; original-scale diagnostics; per-channel summary for Electricity/Traffic |
| Efficiency | identical GPU/software, forward latency plus full train/validation wall-clock, peak allocated/reserved memory, active and total parameters |
| Statistical unit | dataset x horizon x seed pairing; no treating sliding windows as independent replicates |

This core matrix is `4 datasets x 3 horizons x 5 seeds x 6 plain baselines =
360` plain runs, plus the explicitly registered AutoCon and ACF rows. If the
budget cannot support all 360 plain runs, reduce the number of datasets or
horizons before reducing the baseline set; otherwise the main claim will remain
underidentified.

## Decision boundary

Current repository evidence is sufficient to say that Mamba implementations,
AutoCon controls, and ACF-guided candidate branches are runnable and auditable.
It is not sufficient to say that an ACF-Mamba model is uniformly at least as
accurate as the baselines, or faster end-to-end. The next defensible gate is to
approve a new plan/provenance, complete the missing NLinear/plain-baseline
coordinates, and then evaluate the ACF candidate against the frozen six-model
matrix.

## 2026-09-26 PEMS baseline inventory update

The inventory above is the 2026-09-03 CSV-focused snapshot and must not be read
as the current count of local data sources. PEMS03/04/07/08 NPZ traffic-flow
arrays are now present and have independent data-contract audits. In addition,
the iTransformer and TimeMixer PEMS baseline extensions have each completed
40/40 five-seed, validation-only coordinates (H12/H96): 80 new baseline jobs
in total. Their shared modern snapshot differs from the earlier MambaTS v3
candidate snapshot, so they increase baseline coverage but cannot be merged
into a candidate comparison table. The audit, per-coordinate means and
provenance boundary are recorded in
[`pems_modern_baseline_extensions_audit_20260926.md`](pems_modern_baseline_extensions_audit_20260926.md).

为补齐这个缺口，已登记 [`mamba_multidataset_plain_baseline_expansion_plan_20260903.json`](mamba_multidataset_plain_baseline_expansion_plan_20260903.json)：4 个核心数据集、3 个 horizon、3 个 seed，共 216 个 validation-only 作业。它只是补齐 NLinear 和观察数据集依赖性的起点，不是 5-seed 确认矩阵；当前未授权、未训练。

## 2026-09-10 NLinear update

The separate frozen plan
[`mamba_pems_nlinear_baseline_screen_20260910.json`](mamba_pems_nlinear_baseline_screen_20260910.json)
completed 16/16 validation-only jobs on PEMS03/04/07/08 (H12/H96, seeds
2022/2023). This closes the NLinear gap for the small PEMS exploratory screen,
but it does not change the historical five-seed ETT/Electricity/Traffic
coverage above. The detailed result and coordinate-restricted comparison are
in [`mamba_pems_nlinear_baseline_screen_20260910.md`](mamba_pems_nlinear_baseline_screen_20260910.md).

The PEMS screen shows NLinear approximately tied with DLinear and substantially
behind MambaTSv3, PatchTST, and TimesNet at H96. It remains a required control;
the result is not a test-set claim and should not be pooled with the older
confirmatory artifacts.

## 2026-09-10 six-baseline execution smoke

The fresh plan [`mamba_multidataset_plain_baseline_smoke_plan_20260910.json`](mamba_multidataset_plain_baseline_smoke_plan_20260910.json)
completed a deterministic 24-job execution smoke before any larger launch. It
covered ETTh1, ETTm1, Electricity, and Traffic across H96/H336/H720 and
successfully exercised DLinear, NLinear, PatchTST, TimesNet, Mamba1, and
Mamba2. All jobs used `skip_test=true`, so the audit records configuration and
validation completion but zero publishable metric artifacts. High-dimensional
H720 jobs took roughly 108--162 seconds in observed examples; the full 216-job
expansion therefore remains a multi-hour, human-approval decision rather than
an automatic next step. Details are in
[`mamba_multidataset_plain_baseline_smoke_20260910.md`](mamba_multidataset_plain_baseline_smoke_20260910.md).

## 2026-09-10 complete six-baseline screen

The fresh plan [`mamba_multidataset_plain_baseline_full_screen_20260910.json`](mamba_multidataset_plain_baseline_full_screen_20260910.json)
subsequently completed all `216/216` validation-only coordinates under one
source snapshot: ETTh1, ETTm1, Electricity, Traffic; H96/H336/H720; seeds
2021/2022/2023; DLinear, NLinear, PatchTST, TimesNet, Mamba1, and Mamba2.
The phase validator reports `valid=true`, `status_counts.ok=216`, and the
strict readout contains 216 seed rows and 72 dataset/horizon/model groups.
The results are not a universal Mamba win: PatchTST is best on ETTh1 at all
three horizons, while TimesNet is best on Electricity and Traffic H720;
Mamba-2 is competitive with or better than linear baselines mainly on the
high-dimensional datasets. The complete validation-only result and paired
seed deltas are in
[`mamba_multidataset_plain_baseline_full_screen_20260910.md`](mamba_multidataset_plain_baseline_full_screen_20260910.md).
The old `extension_highdim_plain_M` registration remains unrun and must not be
retroactively mixed with this fresh screen.
