# MambaTSv3 Low-Data Core-Transfer Readout

## Scope

This is an exploratory, validation-only transfer study. It does not read a
target test split. The target model retains the existing one-scan MambaTSv3
forecasting path. Only source parameters with both channel-independent names
and identical tensor shapes are transferred: patch embedding/position,
temporal encoder, final normalization, and representation layers. Channel
embeddings, channel context, forecast heads, and persistence scales are newly
initialized for the target.

The target uses 10% of its chronological training prefix, then selects the
best of five fixed epochs on the pre-existing chronological validation block.
Every job used the GPU guard (`78C`, `85%` fan) and four CPU threads.

## Source-Diverse Transfer

Four PEMS source checkpoints (PEMS03/04/07/08) were trained with the existing
AutoCon MambaTSv3 H12/d24 recipe. Their compatible core tensors were averaged
before target adaptation. The target forecast head was never transferred.

| Target condition | Random initialization MSE | Four-source core transfer MSE | Relative change |
| --- | ---: | ---: | ---: |
| Traffic 10%, development condition | 0.895753 | 0.880226 | -1.733% |
| Weather 10%, fresh external gate | 4.469334 | 4.417528 | -1.159% |

Weather was selected as the fresh gate because the earlier single-source
experiment had already used Electricity to reject a source-specific result.
No target test rows were opened. The transfer result therefore supports a
bounded statement: source-diverse, channel-invariant MambaTSv3 initialization
can improve two low-data validation conditions while preserving one-scan
prediction-time cost. It is not a held-out forecasting release and is not a
general cross-domain claim.

## AutoCon Source Ablation

To distinguish source AutoCon from ordinary source-supervised transfer, a
strict H96/d24 source checkpoint control was used. PEMS03/04/07/08, source
seed, source architecture, horizon, averaging rule, target windows, and target
seed were fixed. The only changed source-training factor was
`AutoCon_lambda`: `0` versus `0.001`.

| Traffic 10% development condition | Four-source lambda-zero MSE | Four-source active AutoCon MSE | Active relative change |
| --- | ---: | ---: | ---: |
| Same target run, seed 3501 | 0.900750 | 0.900515 | -0.026% |

The sign favors active AutoCon, but a one-seed `0.026%` change is not a
credible causal effect. The larger source-diverse transfer result must not be
attributed to AutoCon. It is more parsimoniously explained by source-supervised
core initialization plus averaging. This ablation stops target-specific tuning
of the current auxiliary-loss design.

## Decision

The lightweight standalone shared-encoder pretraining line remains rejected.
The multi-source core-transfer recipe is retained as a low-data baseline and a
possible future component. A new AutoCon architecture is justified only if its
relation objective is prediction-connected at source training and produces a
material active-versus-null effect across multiple target domains and seeds.

Artifacts:

- `artifacts/exploratory/arbor_mambatsv3_core_average_traffic_dev_retry_20261002`
- `artifacts/exploratory/arbor_mambatsv3_core_average_weather_gate_20261002`
- `artifacts/exploratory/arbor_mambatsv3_autocon_source_ablation_traffic_20261002`
- `artifacts/exploratory/arbor_mambats_pretraining_20261002/.arbor/tree.json`
