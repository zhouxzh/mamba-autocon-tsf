# MambaACF Fair Baseline Screen

Status: **exploratory; not confirmatory**.

All arms use ETTh1/traffic, H96, seeds 2021-2023, the same split and eight-epoch budget. The three ACF arms are staged mechanism pairings; DLinear, PatchTST, Mamba-1 and Mamba-2 are external reference baselines and are not parameter-count matched.

The `relation - memory` row is **not** a pure relation-loss effect: the memory arm executes `MambaACF`, while the relation arm executes `MambaACFAutoCon` and enables the AutoCon objective. Treat that row as a confounded staged comparison; the same-architecture `lambda=.03 - lambda=0` null control in the variable-token screen is the interpretable relation contrast.

## Staged Contrasts

| Contrast (treatment - reference) | n | MSE delta | MSE relative | MAE delta | MAE relative |
|---|---:|---:|---:|---:|---:|
| memory - DLinear | 6 | -0.0607251 | -11.290% | -0.0107685 | -2.628% |
| memory - PatchTST | 6 | 0.0089515 | 1.912% | 0.0239139 | 6.375% |
| memory - Mamba1 | 6 | -0.0494504 | -9.391% | -0.014214 | -3.439% |
| memory - Mamba2 | 6 | -0.056681 | -10.618% | -0.0208627 | -4.968% |
| relation - memory | 6 | 0.00149797 | 0.314% | 0.00129416 | 0.324% |
| calibrated - relation | 6 | -0.000651916 | -0.136% | -0.000254799 | -0.064% |

Negative accuracy deltas favor the treatment. Any p-value in the JSON is an unadjusted descriptive paired t-test and must not be used for model selection.

## Diagnostics

| Arm | Checkpoint state elements (mean) | Effective memory RMS | Relation sign accuracy | Calibration amplitude |
|---|---:|---:|---:|---:|
| DLinear | 18624.0 | n/a | n/a | n/a |
| PatchTST | 206080.0 | n/a | n/a | n/a |
| Mamba1 | 264962.5 | n/a | n/a | n/a |
| Mamba2 | 262894.5 | n/a | n/a | n/a |
| memory | 255552.5 | n/a | n/a | n/a |
| relation | 255552.5 | 0.160928 | 0.911705 | n/a |
| calibrated | 255558.5 | 0.171735 | 0.911762 | 1.16672 |

## Boundaries

- Checkpoint state elements are reported as a transparent size diagnostic, not as a trainable-parameter claim.
- The staged ACF comparisons require identical raw train-only sidecar hashes and lag resolution at each dataset-seed coordinate.
- Short training, six paired coordinates per contrast, non-matched external capacity, and no independent replication prevent a publication-level claim.
