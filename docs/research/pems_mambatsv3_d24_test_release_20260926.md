# PEMS d24 MambaTS v3 AutoCon Held-Out Test Release (2026-09-26)

## Protocol integrity

This is a frozen PEMS H12 held-out release for the four-token, full-coverage
`MambaTSv3AutoCon` candidate (`patch_len=24`, `patch_stride=24`, `d_model=24`,
`d_ff=48`, one causal Mamba scan). It retains AutoCon's shared
representation-learning objective with `lambda=0.001`; prediction-only
deployment disables representation return and does not execute the training
loss.

The validation-only contract phase (`12/12`), candidate test phase (`20/20`),
and seven-baseline test phase (`140/140`) all passed strict phase validation
with zero errors. Every test phase used five fresh paired seeds
`2131--2135`, streaming held-out metrics, immutable source snapshots, and an
enforced launch-sidecar binding. Test metrics were read only after all three
phases were complete and valid.

## Held-out forecasting accuracy

The table shows mean streaming scaled MSE. Every listed cell is `5/5` paired
seed wins for the candidate. The registered primary comparators are PatchTST
and Mamba2: their one-sided 95% paired upper MSE-ratio bounds are all below
the pre-registered `1.02` non-inferiority threshold.

| Dataset | Candidate | PatchTST | Mamba2 | TimeMixer | iTransformer |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS03 | .069361 | .098339 | .076299 | .071618 | .078095 |
| PEMS04 | .080189 | .116095 | .082468 | .082309 | .096220 |
| PEMS07 | .063274 | .092364 | .069001 | .066941 | .073909 |
| PEMS08 | .083224 | .109606 | .088140 | .085529 | .089992 |

The candidate also has lower held-out MSE than DLinear, NLinear, and TimesNet
on all four datasets. Its improvement over TimeMixer is `3.14%`, `2.57%`,
`5.47%`, and `2.70%` for PEMS03/04/07/08, respectively. The full 28-row
paired test table, including one-sided bounds and all descriptive baselines,
is in [`paired_test_summary.csv`](../../artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/readout/paired_test_summary.csv).

## Matched prediction-only resources

All resource rows use the validated source snapshots on RTX 5090 D, FP32,
device-resident `B=32,L=96,H=12` inputs, 50 warm-up forwards, and 200 timed
forwards. They exclude data loading, optimization, and the contrastive loss.

| Dataset | Candidate ms | PatchTST ms | TimeMixer ms | Mamba2 ms | iTransformer ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| PEMS03 | .303 | .542 | 1.431 | .736 | .358 |
| PEMS04 | .285 | .476 | 1.274 | .723 | .303 |
| PEMS07 | .530 | 1.194 | 2.883 | .750 | 1.578 |
| PEMS08 | .269 | .327 | .871 | .740 | .194 |

The candidate is faster than PatchTST, TimesNet, TimeMixer, and Mamba2 on all
four matched PEMS shapes, and faster than iTransformer on PEMS03/04/07. It is
slower than DLinear/NLinear everywhere and slower than iTransformer on
PEMS08. Candidate parameter counts range from `15,066` to `38,595`, versus
TimeMixer's `57,554--126,715` and Mamba2's `26,549--112,822`; parameter count
is not uniformly lower than PatchTST or iTransformer. Thus the supported
efficiency claim is a named-comparator forward-latency result, not universal
speed or parameter superiority. Full records are in
[`candidate resources`](../../artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/resources_candidate/model_resources.json)
and [`baseline resources`](../../artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/resources_baselines/model_resources.json).

## Interpretation and remaining gap

This is the strongest current evidence in the repository: a four-dataset,
five-seed, held-out short-horizon result with matched modern baselines and
reproducible resource measurements. It supports a narrow manuscript result
for non-graph PEMS H12 forecasting.

It does not satisfy the full high-level-paper target. The same d24 route
failed its four-dataset H96 validation gate, so it cannot be presented as a
long-horizon or general time-series method. Earlier active-versus-null PEMS
screens also show that the incremental AutoCon loss is small and inconsistent;
the paper may truthfully state that the final model uses prediction-connected
AutoCon representation learning and report the null ablation, but may not
claim that AutoCon alone caused the held-out gain. A publication-grade next
stage requires a new, frozen long-horizon mechanism that passes the H96 gate
before another test release.

Artifacts:

- [Frozen plan](pems_mambatsv3_d24_test_release_20260926.json)
- [Launch binding](pems_mambatsv3_d24_test_release_20260926_launch_sidecar.json)
- [Machine-readable test readout](../../artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/readout/readout.md)
